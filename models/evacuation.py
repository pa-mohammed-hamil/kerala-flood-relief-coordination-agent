from __future__ import annotations

"""
backend/models/evacuation.py

Kerala Flood Relief Coordination Agent

SQLAlchemy 2.x models for emergency evacuation
planning, approval, coordination and tracking.

Features:
- Flood evacuation requests
- Evacuation plans and priority classification
- Vulnerable population tracking
- Pickup points and destinations
- Shelter and rescue coordination
- Vehicle and transport assignments
- Route safety verification
- Human authorization and approval
- Evacuation progress and headcounts
- Emergency incident references
- Evacuation event history
- Optimistic concurrency control

SAFETY:
These models do not initiate emergency dispatch,
issue evacuation orders, reserve shelter capacity
or control vehicles.

AI recommendations are advisory. Operational
evacuations require authorized human coordination,
verified routes and destination confirmation.
"""


import enum
import math
import uuid

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    event,
)

from sqlalchemy.ext.mutable import (
    MutableDict,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from backend.database.base import Base


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(
    enum_class: type[enum.Enum],
) -> SQLEnum:
    return SQLEnum(
        enum_class,
        values_callable=lambda members: [
            member.value for member in members
        ],
        native_enum=False,
        validate_strings=True,
        length=80,
    )


def validate_non_negative_integer(
    value: int | None,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a non-negative integer"
        )

    return value


def validate_coordinate(
    value: float | None,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"Invalid {field_name}"
        )

    return float(value)


def require_text(
    value: str,
    field_name: str,
    max_length: int = 500,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise ValueError(
            f"{field_name} exceeds maximum length"
        )

    return result


# ============================================================
# ENUMERATIONS
# ============================================================

class EvacuationStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    PLANNING = "planning"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    ARRIVED = "arrived"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class EvacuationPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvacuationReason(str, enum.Enum):
    FLOOD_RISK = "flood_risk"
    ACTIVE_FLOODING = "active_flooding"
    FLASH_FLOOD = "flash_flood"
    LANDSLIDE_RISK = "landslide_risk"
    DAM_RELEASE_WARNING = "dam_release_warning"
    BUILDING_UNSAFE = "building_unsafe"
    MEDICAL_EMERGENCY = "medical_emergency"
    INFRASTRUCTURE_FAILURE = (
        "infrastructure_failure"
    )
    OTHER = "other"


class EvacuationType(str, enum.Enum):
    PREVENTIVE = "preventive"
    EMERGENCY = "emergency"
    MEDICAL = "medical"
    ASSISTED = "assisted"
    MASS_EVACUATION = "mass_evacuation"
    RELOCATION = "relocation"


class EvacuationTransportMode(str, enum.Enum):
    WALKING = "walking"
    BUS = "bus"
    VAN = "van"
    CAR = "car"
    AMBULANCE = "ambulance"
    BOAT = "boat"
    RESCUE_VEHICLE = "rescue_vehicle"
    HELICOPTER = "helicopter"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EvacuationRouteStatus(str, enum.Enum):
    NOT_ASSESSED = "not_assessed"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED_SAFE = "verified_safe"
    RESTRICTED = "restricted"
    UNSAFE = "unsafe"
    EXPIRED = "expired"


class EvacuationDestinationType(str, enum.Enum):
    RELIEF_SHELTER = "relief_shelter"
    HOSPITAL = "hospital"
    TEMPORARY_CAMP = "temporary_camp"
    SAFE_ASSEMBLY_POINT = "safe_assembly_point"
    OTHER = "other"


class EvacuationAssignmentStatus(str, enum.Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACCEPTED = "accepted"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    TRANSPORTING = "transporting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class EvacuationEventType(str, enum.Enum):
    CREATED = "created"
    VERIFIED = "verified"
    PRIORITY_CHANGED = "priority_changed"
    PLAN_UPDATED = "plan_updated"
    ROUTE_VERIFIED = "route_verified"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    TEAM_ASSIGNED = "team_assigned"
    VEHICLE_ASSIGNED = "vehicle_assigned"
    STARTED = "started"
    PAUSED = "paused"
    RESUMED = "resumed"
    HEADCOUNT_UPDATED = "headcount_updated"
    DESTINATION_REACHED = "destination_reached"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    NOTE_ADDED = "note_added"


# ============================================================
# EVACUATION MASTER MODEL
# ============================================================

class Evacuation(Base):
    __tablename__ = "evacuations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    evacuation_number: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )

    district_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "districts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    incident_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    sos_request_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    rescue_mission_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reason: Mapped[EvacuationReason] = mapped_column(
        enum_type(EvacuationReason),
        nullable=False,
    )

    evacuation_type: Mapped[
        EvacuationType
    ] = mapped_column(
        enum_type(EvacuationType),
        nullable=False,
    )

    priority: Mapped[
        EvacuationPriority
    ] = mapped_column(
        enum_type(EvacuationPriority),
        default=EvacuationPriority.MEDIUM,
        nullable=False,
        index=True,
    )

    status: Mapped[
        EvacuationStatus
    ] = mapped_column(
        enum_type(EvacuationStatus),
        default=EvacuationStatus.DRAFT,
        nullable=False,
        index=True,
    )

    source_location_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    source_address: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    source_latitude: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    source_longitude: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    taluk: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    panchayat: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    ward: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    destination_type: Mapped[
        EvacuationDestinationType | None
    ] = mapped_column(
        enum_type(EvacuationDestinationType),
        nullable=True,
    )

    destination_name: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    destination_address: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    destination_latitude: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    destination_longitude: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    shelter_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    shelter_reservation_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    destination_confirmed: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    destination_confirmed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    estimated_people: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    children_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    elderly_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    disabled_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    pregnant_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    medical_assistance_count: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_evacuated: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_arrived: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_unaccounted_for: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    transport_mode: Mapped[
        EvacuationTransportMode
    ] = mapped_column(
        enum_type(EvacuationTransportMode),
        default=EvacuationTransportMode.UNKNOWN,
        nullable=False,
    )

    route_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    route_status: Mapped[
        EvacuationRouteStatus
    ] = mapped_column(
        enum_type(EvacuationRouteStatus),
        default=EvacuationRouteStatus.NOT_ASSESSED,
        nullable=False,
    )

    route_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    route_verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    route_valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    requires_medical_support: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_accessible_transport: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_rescue_team: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_human_approval: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approval_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    approved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    coordinator_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    estimated_start_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    estimated_completion_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    started_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    arrived_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    cancelled_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    cancellation_reason: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    safety_notes: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    operational_notes: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    assignments: Mapped[
        list["EvacuationAssignment"]
    ] = relationship(
        "EvacuationAssignment",
        back_populates="evacuation",
        cascade="save-update, merge",
    )

    checkpoints: Mapped[
        list["EvacuationCheckpoint"]
    ] = relationship(
        "EvacuationCheckpoint",
        back_populates="evacuation",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["EvacuationEvent"]
    ] = relationship(
        "EvacuationEvent",
        back_populates="evacuation",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_evacuation_district",
        ),
        CheckConstraint(
            "estimated_people > 0",
            name="ck_evacuation_estimated_people",
        ),
        CheckConstraint(
            "children_count >= 0",
            name="ck_evacuation_children",
        ),
        CheckConstraint(
            "elderly_count >= 0",
            name="ck_evacuation_elderly",
        ),
        CheckConstraint(
            "disabled_count >= 0",
            name="ck_evacuation_disabled",
        ),
        CheckConstraint(
            "pregnant_count >= 0",
            name="ck_evacuation_pregnant",
        ),
        CheckConstraint(
            "medical_assistance_count >= 0",
            name="ck_evacuation_medical",
        ),
        CheckConstraint(
            "people_evacuated >= 0",
            name="ck_evacuation_people_evacuated",
        ),
        CheckConstraint(
            "people_arrived >= 0",
            name="ck_evacuation_people_arrived",
        ),
        CheckConstraint(
            "people_unaccounted_for >= 0",
            name="ck_evacuation_unaccounted",
        ),
        CheckConstraint(
            "people_arrived <= people_evacuated",
            name="ck_evacuation_arrival_count",
        ),
        CheckConstraint(
            "source_latitude IS NULL OR "
            "(source_latitude >= -90 "
            "AND source_latitude <= 90)",
            name="ck_evacuation_source_latitude",
        ),
        CheckConstraint(
            "source_longitude IS NULL OR "
            "(source_longitude >= -180 "
            "AND source_longitude <= 180)",
            name="ck_evacuation_source_longitude",
        ),
        CheckConstraint(
            "destination_latitude IS NULL OR "
            "(destination_latitude >= -90 "
            "AND destination_latitude <= 90)",
            name="ck_evacuation_destination_latitude",
        ),
        CheckConstraint(
            "destination_longitude IS NULL OR "
            "(destination_longitude >= -180 "
            "AND destination_longitude <= 180)",
            name="ck_evacuation_destination_longitude",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_evacuation_version",
        ),
        Index(
            "ix_evacuation_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_evacuation_priority_status",
            "priority",
            "status",
        ),
        Index(
            "ix_evacuation_created_at",
            "created_at",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "source_latitude",
        "destination_latitude",
    )
    def validate_latitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -90, 90
        )

    @validates(
        "source_longitude",
        "destination_longitude",
    )
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -180, 180
        )

    @validates(
        "children_count",
        "elderly_count",
        "disabled_count",
        "pregnant_count",
        "medical_assistance_count",
        "people_evacuated",
        "people_arrived",
        "people_unaccounted_for",
    )
    def validate_count(
        self,
        key: str,
        value: int,
    ) -> int:
        result = validate_non_negative_integer(
            value, key
        )
        if result is None:
            raise ValueError(f"{key} is required")
        return result

    @validates("estimated_people")
    def validate_estimated_people(
        self,
        key: str,
        value: int,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                "estimated_people must be positive"
            )

        return value

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            EvacuationStatus.COMPLETED,
            EvacuationStatus.CANCELLED,
            EvacuationStatus.FAILED,
        }

    @property
    def has_valid_route(self) -> bool:
        if self.route_status != (
            EvacuationRouteStatus.VERIFIED_SAFE
        ):
            return False

        if self.route_verified_by_user_id is None:
            return False

        if self.route_valid_until is None:
            return False

        return self.route_valid_until > utc_now()

    @property
    def has_operational_approval(self) -> bool:
        return (
            self.approved_by_user_id is not None
            and self.approved_at is not None
            and bool(self.approval_reference)
        )

    @property
    def can_begin(self) -> bool:
        return (
            self.status in {
                EvacuationStatus.APPROVED,
                EvacuationStatus.READY,
            }
            and self.is_verified
            and self.destination_confirmed
            and self.has_valid_route
            and self.has_operational_approval
        )

    @property
    def completion_percentage(self) -> float:
        if self.estimated_people <= 0:
            return 0.0

        return round(
            min(
                self.people_arrived
                / self.estimated_people
                * 100,
                100.0,
            ),
            2,
        )

    def submit_for_verification(self) -> None:
        if self.status != EvacuationStatus.DRAFT:
            raise ValueError(
                "Only draft evacuations can be submitted"
            )

        self.status = (
            EvacuationStatus.PENDING_VERIFICATION
        )

    def verify(
        self,
        *,
        verifier_id: int,
    ) -> None:
        if self.status != (
            EvacuationStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Evacuation is not pending verification"
            )

        if verifier_id <= 0:
            raise ValueError("Invalid verifier")

        if (
            self.created_by_user_id is not None
            and verifier_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent verification required"
            )

        self.is_verified = True
        self.verified_by_user_id = verifier_id
        self.verified_at = utc_now()
        self.status = EvacuationStatus.VERIFIED

    def start_planning(self) -> None:
        if self.status != EvacuationStatus.VERIFIED:
            raise ValueError(
                "Evacuation must be verified"
            )

        self.status = EvacuationStatus.PLANNING

    def submit_for_approval(self) -> None:
        if self.status != EvacuationStatus.PLANNING:
            raise ValueError(
                "Evacuation must be in planning"
            )

        if not self.destination_confirmed:
            raise ValueError(
                "Destination must be confirmed"
            )

        if not self.has_valid_route:
            raise ValueError(
                "A currently verified safe route is required"
            )

        self.status = (
            EvacuationStatus.PENDING_APPROVAL
        )

    def approve(
        self,
        *,
        approver_id: int,
        approval_reference: str,
    ) -> None:
        if self.status != (
            EvacuationStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Evacuation is not pending approval"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if (
            self.created_by_user_id is not None
            and approver_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent approval required"
            )

        if not approval_reference.strip():
            raise ValueError(
                "Approval reference required"
            )

        if not self.destination_confirmed:
            raise ValueError(
                "Destination is not confirmed"
            )

        if not self.has_valid_route:
            raise ValueError(
                "Route verification is missing or expired"
            )

        self.approved_by_user_id = approver_id
        self.approval_reference = (
            approval_reference.strip()
        )
        self.approved_at = utc_now()
        self.status = EvacuationStatus.APPROVED

    def start(self) -> None:
        if not self.can_begin:
            raise ValueError(
                "Evacuation is not authorized or ready"
            )

        self.status = EvacuationStatus.IN_PROGRESS
        self.started_at = utc_now()

    def pause(self, reason: str) -> None:
        if self.status != EvacuationStatus.IN_PROGRESS:
            raise ValueError(
                "Only active evacuations can be paused"
            )

        self.status = EvacuationStatus.PAUSED
        self.safety_notes = require_text(
            reason, "reason", 2000
        )

    def resume(self) -> None:
        if self.status != EvacuationStatus.PAUSED:
            raise ValueError(
                "Evacuation is not paused"
            )

        if not self.has_valid_route:
            raise ValueError(
                "Route must be reverified before resuming"
            )

        self.status = EvacuationStatus.IN_PROGRESS

    def record_headcount(
        self,
        *,
        evacuated: int,
        arrived: int,
        unaccounted_for: int,
    ) -> None:
        if self.status not in {
            EvacuationStatus.IN_PROGRESS,
            EvacuationStatus.PAUSED,
            EvacuationStatus.ARRIVED,
        }:
            raise ValueError(
                "Headcount updates require an active "
                "or arrived evacuation"
            )

        for field_name, value in (
            ("evacuated", evacuated),
            ("arrived", arrived),
            ("unaccounted_for", unaccounted_for),
        ):
            validate_non_negative_integer(
                value, field_name
            )

        if arrived > evacuated:
            raise ValueError(
                "Arrived count cannot exceed evacuated"
            )

        if unaccounted_for > evacuated:
            raise ValueError(
                "Unaccounted count cannot exceed evacuated"
            )

        if arrived + unaccounted_for > evacuated:
            raise ValueError(
                "Headcount categories overlap"
            )

        self.people_evacuated = evacuated
        self.people_arrived = arrived
        self.people_unaccounted_for = unaccounted_for

    def mark_arrived(self) -> None:
        if self.status != EvacuationStatus.IN_PROGRESS:
            raise ValueError(
                "Evacuation must be in progress"
            )

        if self.people_arrived <= 0:
            raise ValueError(
                "No arrivals have been recorded"
            )

        self.status = EvacuationStatus.ARRIVED
        self.arrived_at = utc_now()

    def complete(self) -> None:
        if self.status != EvacuationStatus.ARRIVED:
            raise ValueError(
                "Evacuation has not reached its destination"
            )

        if self.people_unaccounted_for != 0:
            raise ValueError(
                "Unaccounted people require resolution "
                "before completion"
            )

        if self.people_arrived != self.estimated_people:
            raise ValueError(
                "Arrival count does not match the "
                "planned population"
            )

        self.status = EvacuationStatus.COMPLETED
        self.completed_at = utc_now()

    def cancel(self, reason: str) -> None:
        if self.is_terminal:
            raise ValueError(
                "Terminal evacuation cannot be cancelled"
            )

        if self.status in {
            EvacuationStatus.IN_PROGRESS,
            EvacuationStatus.PAUSED,
            EvacuationStatus.ARRIVED,
        }:
            raise ValueError(
                "Active evacuations require coordinated "
                "stand-down before cancellation"
            )

        self.cancellation_reason = require_text(
            reason, "reason", 2000
        )
        self.status = EvacuationStatus.CANCELLED
        self.cancelled_at = utc_now()

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "evacuation_number": self.evacuation_number,
            "district_id": self.district_id,
            "reason": self.reason.value,
            "evacuation_type": self.evacuation_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "estimated_people": self.estimated_people,
            "people_evacuated": self.people_evacuated,
            "people_arrived": self.people_arrived,
            "people_unaccounted_for": (
                self.people_unaccounted_for
            ),
            "route_status": self.route_status.value,
            "destination_confirmed": (
                self.destination_confirmed
            ),
            "completion_percentage": (
                self.completion_percentage
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<Evacuation("
            f"id={self.id!r}, "
            f"status={self.status.value!r}, "
            f"priority={self.priority.value!r}"
            f")>"
        )


# ============================================================
# EVACUATION ASSIGNMENTS
# ============================================================

class EvacuationAssignment(Base):
    __tablename__ = "evacuation_assignments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    evacuation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "evacuations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    rescue_team_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            nullable=True,
        )
    )

    volunteer_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            nullable=True,
        )
    )

    vehicle_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    transport_mode: Mapped[
        EvacuationTransportMode
    ] = mapped_column(
        enum_type(EvacuationTransportMode),
        nullable=False,
    )

    vehicle_capacity: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    assigned_people: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    status: Mapped[
        EvacuationAssignmentStatus
    ] = mapped_column(
        enum_type(EvacuationAssignmentStatus),
        default=EvacuationAssignmentStatus.PROPOSED,
        nullable=False,
    )

    assigned_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    accepted_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    departed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    evacuation: Mapped["Evacuation"] = relationship(
        "Evacuation",
        back_populates="assignments",
    )

    __table_args__ = (
        CheckConstraint(
            "vehicle_capacity IS NULL OR "
            "vehicle_capacity >= 0",
            name="ck_evac_assignment_capacity",
        ),
        CheckConstraint(
            "assigned_people >= 0",
            name="ck_evac_assignment_people",
        ),
        CheckConstraint(
            "vehicle_capacity IS NULL OR "
            "assigned_people <= vehicle_capacity",
            name="ck_evac_assignment_capacity_limit",
        ),
        Index(
            "ix_evac_assignment_status",
            "evacuation_id",
            "status",
        ),
    )


# ============================================================
# EVACUATION CHECKPOINTS
# ============================================================

class EvacuationCheckpoint(Base):
    __tablename__ = "evacuation_checkpoints"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    evacuation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "evacuations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    checkpoint_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    checkpoint_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    is_mandatory: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    is_verified_safe: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    expected_arrival_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    actual_arrival_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    recorded_people: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    evacuation: Mapped["Evacuation"] = relationship(
        "Evacuation",
        back_populates="checkpoints",
    )

    __table_args__ = (
        UniqueConstraint(
            "evacuation_id",
            "checkpoint_number",
            name="uq_evacuation_checkpoint_number",
        ),
        CheckConstraint(
            "checkpoint_number > 0",
            name="ck_evac_checkpoint_number",
        ),
        CheckConstraint(
            "recorded_people IS NULL OR "
            "recorded_people >= 0",
            name="ck_evac_checkpoint_people",
        ),
    )


# ============================================================
# EVACUATION EVENT HISTORY
# ============================================================

class EvacuationEvent(Base):
    __tablename__ = "evacuation_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    evacuation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "evacuations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        EvacuationEventType
    ] = mapped_column(
        enum_type(EvacuationEventType),
        nullable=False,
    )

    previous_status: Mapped[
        EvacuationStatus | None
    ] = mapped_column(
        enum_type(EvacuationStatus),
        nullable=True,
    )

    new_status: Mapped[
        EvacuationStatus | None
    ] = mapped_column(
        enum_type(EvacuationStatus),
        nullable=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    actor_agent_name: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    evacuation: Mapped["Evacuation"] = relationship(
        "Evacuation",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_evac_event_time",
            "evacuation_id",
            "occurred_at",
        ),
    )


# ============================================================
# ORM EVENTS
# ============================================================

@event.listens_for(
    Evacuation,
    "before_update",
)
def update_evacuation_timestamp(
    mapper: Any,
    connection: Any,
    target: Evacuation,
) -> None:
    target.updated_at = utc_now()


@event.listens_for(
    EvacuationAssignment,
    "before_update",
)
def update_assignment_timestamp(
    mapper: Any,
    connection: Any,
    target: EvacuationAssignment,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

EVACUATION_MODELS = (
    Evacuation,
    EvacuationAssignment,
    EvacuationCheckpoint,
    EvacuationEvent,
)


__all__ = [
    "Evacuation",
    "EvacuationAssignment",
    "EvacuationCheckpoint",
    "EvacuationEvent",
    "EvacuationStatus",
    "EvacuationPriority",
    "EvacuationReason",
    "EvacuationType",
    "EvacuationTransportMode",
    "EvacuationRouteStatus",
    "EvacuationDestinationType",
    "EvacuationAssignmentStatus",
    "EvacuationEventType",
    "EVACUATION_MODELS",
]
