from __future__ import annotations

"""
backend/models/rescue_mission.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x ORM models for flood rescue missions.

Features:
- Rescue mission registration and tracking
- SOS and incident coordination
- District-based rescue operations
- Rescue team and vehicle assignments
- Boat, ambulance and emergency transport support
- Rescue priority and risk classification
- Victim and vulnerable-person tracking
- Route verification and safety assessments
- Rescue mission checkpoints
- Live operational progress
- Rescue outcome and headcount recording
- Equipment requirements
- Mission evidence and reports
- Human approval and authorization
- AI-assisted rescue recommendations
- Audit events and optimistic concurrency

Safety:
- AI recommendations never authorize dispatch.
- Only authorized personnel may approve operations.
- Routes must be assessed for current conditions.
- Headcounts require reconciliation.
- Medical and victim information is sensitive.
- Physical rescue decisions remain with qualified
  emergency response personnel.
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

from sqlalchemy.ext.mutable import MutableDict
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


def require_text(
    value: str,
    field_name: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} is required"
        )

    result = value.strip()

    if len(result) > maximum:
        raise ValueError(
            f"{field_name} exceeds {maximum} characters"
        )

    return result


def validate_non_negative_int(
    value: int | None,
    field_name: str,
    *,
    allow_none: bool = False,
) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required")

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


def validate_probability(
    value: float | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )

    return float(value)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class RescueMissionStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    PLANNING = "planning"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    READY = "ready"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    RESCUE_IN_PROGRESS = "rescue_in_progress"
    EVACUATING = "evacuating"
    RETURNING = "returning"
    PAUSED = "paused"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RescueMissionType(str, enum.Enum):
    FLOOD_RESCUE = "flood_rescue"
    FLASH_FLOOD_RESCUE = "flash_flood_rescue"
    WATER_RESCUE = "water_rescue"
    BOAT_RESCUE = "boat_rescue"
    LANDSLIDE_RESCUE = "landslide_rescue"
    BUILDING_EVACUATION = "building_evacuation"
    MEDICAL_EVACUATION = "medical_evacuation"
    SEARCH_AND_RESCUE = "search_and_rescue"
    MISSING_PERSON_SEARCH = "missing_person_search"
    VEHICLE_RESCUE = "vehicle_rescue"
    ELDERLY_ASSISTANCE = "elderly_assistance"
    DISABILITY_ASSISTANCE = "disability_assistance"
    CHILD_RESCUE = "child_rescue"
    ANIMAL_RESCUE = "animal_rescue"
    OTHER = "other"


class RescueMissionPriority(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    LIFE_THREATENING = "life_threatening"


class RescueRiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class RescueTransportMode(str, enum.Enum):
    BOAT = "boat"
    RESCUE_BOAT = "rescue_boat"
    AMBULANCE = "ambulance"
    RESCUE_VEHICLE = "rescue_vehicle"
    TRUCK = "truck"
    VAN = "van"
    HELICOPTER = "helicopter"
    ON_FOOT = "on_foot"
    MIXED = "mixed"
    OTHER = "other"


class RescueRouteStatus(str, enum.Enum):
    NOT_ASSESSED = "not_assessed"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED_SAFE = "verified_safe"
    RESTRICTED = "restricted"
    UNSAFE = "unsafe"
    EXPIRED = "expired"


class RescueAssignmentStatus(str, enum.Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACCEPTED = "accepted"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    ACTIVE = "active"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RescueCheckpointStatus(str, enum.Enum):
    PENDING = "pending"
    REACHED = "reached"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    FAILED = "failed"


class RescueEquipmentCategory(str, enum.Enum):
    LIFE_JACKET = "life_jacket"
    RESCUE_ROPE = "rescue_rope"
    RESCUE_RING = "rescue_ring"
    STRETCHER = "stretcher"
    FIRST_AID_KIT = "first_aid_kit"
    MEDICAL_KIT = "medical_kit"
    OXYGEN = "oxygen"
    BOAT = "boat"
    MOTOR = "motor"
    RADIO = "radio"
    SATELLITE_PHONE = "satellite_phone"
    FLASHLIGHT = "flashlight"
    PPE = "ppe"
    DRONE = "drone"
    THERMAL_CAMERA = "thermal_camera"
    GPS_DEVICE = "gps_device"
    OTHER = "other"


class RescueEquipmentStatus(str, enum.Enum):
    REQUIRED = "required"
    RESERVED = "reserved"
    ALLOCATED = "allocated"
    CHECKED = "checked"
    DEPLOYED = "deployed"
    RETURNED = "returned"
    DAMAGED = "damaged"
    UNAVAILABLE = "unavailable"


class RescueOutcome(str, enum.Enum):
    PENDING = "pending"
    ALL_RESCUED = "all_rescued"
    PARTIALLY_RESCUED = "partially_rescued"
    NO_PERSON_FOUND = "no_person_found"
    TRANSFERRED_TO_OTHER_TEAM = "transferred_to_other_team"
    UNSAFE_TO_PROCEED = "unsafe_to_proceed"
    CANCELLED = "cancelled"
    OTHER = "other"


class RescueEvidenceType(str, enum.Enum):
    PHOTO = "photo"
    VIDEO = "video"
    DOCUMENT = "document"
    GPS_TRACK = "gps_track"
    FIELD_REPORT = "field_report"
    MEDICAL_HANDOVER = "medical_handover"
    OFFICIAL_REPORT = "official_report"
    OTHER = "other"


class RescueMissionEventType(str, enum.Enum):
    CREATED = "created"
    VERIFIED = "verified"
    PLAN_UPDATED = "plan_updated"
    RISK_UPDATED = "risk_updated"
    ROUTE_VERIFIED = "route_verified"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    TEAM_ASSIGNED = "team_assigned"
    EQUIPMENT_ALLOCATED = "equipment_allocated"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    ARRIVED_ON_SCENE = "arrived_on_scene"
    RESCUE_STARTED = "rescue_started"
    PERSON_RESCUED = "person_rescued"
    HEADCOUNT_UPDATED = "headcount_updated"
    MEDICAL_HANDOVER = "medical_handover"
    EVACUATION_STARTED = "evacuation_started"
    RETURN_STARTED = "return_started"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SAFETY_ESCALATION = "safety_escalation"
    EVIDENCE_ADDED = "evidence_added"
    NOTE_ADDED = "note_added"


# ============================================================
# RESCUE MISSION
# ============================================================

class RescueMission(Base):
    __tablename__ = "rescue_missions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    mission_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
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

    evacuation_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    missing_person_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            nullable=True,
        )
    )

    mission_type: Mapped[
        RescueMissionType
    ] = mapped_column(
        enum_type(RescueMissionType),
        nullable=False,
    )

    priority: Mapped[
        RescueMissionPriority
    ] = mapped_column(
        enum_type(RescueMissionPriority),
        default=RescueMissionPriority.NORMAL,
        nullable=False,
        index=True,
    )

    risk_level: Mapped[
        RescueRiskLevel
    ] = mapped_column(
        enum_type(RescueRiskLevel),
        default=RescueRiskLevel.UNKNOWN,
        nullable=False,
    )

    status: Mapped[
        RescueMissionStatus
    ] = mapped_column(
        enum_type(RescueMissionStatus),
        default=RescueMissionStatus.DRAFT,
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    mission_objective: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    location_name: Mapped[str | None] = (
        mapped_column(
            String(255),
            nullable=True,
        )
    )

    address: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
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

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    target_area_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    estimated_people_at_risk: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_confirmed_at_risk: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_rescued: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_evacuated: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_transferred_to_medical: Mapped[
        int
    ] = mapped_column(
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

    children_at_risk: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    elderly_at_risk: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_with_disabilities_at_risk: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_requiring_medical_support: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    requires_medical_team: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_water_rescue: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_specialized_equipment: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_technical_rescue: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    transport_mode: Mapped[
        RescueTransportMode
    ] = mapped_column(
        enum_type(RescueTransportMode),
        default=RescueTransportMode.OTHER,
        nullable=False,
    )

    route_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    route_status: Mapped[
        RescueRouteStatus
    ] = mapped_column(
        enum_type(RescueRouteStatus),
        default=RescueRouteStatus.NOT_ASSESSED,
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

    destination_shelter_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    destination_hospital_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    destination_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    destination_confirmed: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_ai_recommended: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    ai_agent_name: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    ai_model_version: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    ai_confidence_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    ai_decision_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
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

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    created_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    mission_commander_user_id: Mapped[
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

    approval_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    dispatched_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    outcome: Mapped[
        RescueOutcome
    ] = mapped_column(
        enum_type(RescueOutcome),
        default=RescueOutcome.PENDING,
        nullable=False,
    )

    outcome_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    outcome_verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    outcome_summary: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    safety_notes: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    cancellation_reason: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    failure_reason: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    planned_start_at: Mapped[
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

    dispatched_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    arrived_on_scene_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    started_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    completed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    idempotency_key: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        unique=True,
        nullable=True,
    )

    audit_reference: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
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
        list["RescueMissionAssignment"]
    ] = relationship(
        "RescueMissionAssignment",
        back_populates="mission",
        cascade="save-update, merge",
    )

    checkpoints: Mapped[
        list["RescueMissionCheckpoint"]
    ] = relationship(
        "RescueMissionCheckpoint",
        back_populates="mission",
        cascade="save-update, merge",
    )

    equipment: Mapped[
        list["RescueMissionEquipment"]
    ] = relationship(
        "RescueMissionEquipment",
        back_populates="mission",
        cascade="save-update, merge",
    )

    evidence: Mapped[
        list["RescueMissionEvidence"]
    ] = relationship(
        "RescueMissionEvidence",
        back_populates="mission",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["RescueMissionEvent"]
    ] = relationship(
        "RescueMissionEvent",
        back_populates="mission",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_rescue_mission_district",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_rescue_mission_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_rescue_mission_longitude",
        ),
        CheckConstraint(
            "estimated_people_at_risk >= 0",
            name="ck_rescue_mission_estimated",
        ),
        CheckConstraint(
            "people_confirmed_at_risk >= 0",
            name="ck_rescue_mission_confirmed",
        ),
        CheckConstraint(
            "people_rescued >= 0",
            name="ck_rescue_mission_rescued",
        ),
        CheckConstraint(
            "people_evacuated >= 0",
            name="ck_rescue_mission_evacuated",
        ),
        CheckConstraint(
            "people_transferred_to_medical >= 0",
            name="ck_rescue_mission_medical",
        ),
        CheckConstraint(
            "people_unaccounted_for >= 0",
            name="ck_rescue_mission_unaccounted",
        ),
        CheckConstraint(
            "children_at_risk >= 0",
            name="ck_rescue_mission_children",
        ),
        CheckConstraint(
            "elderly_at_risk >= 0",
            name="ck_rescue_mission_elderly",
        ),
        CheckConstraint(
            "people_with_disabilities_at_risk >= 0",
            name="ck_rescue_mission_disability",
        ),
        CheckConstraint(
            "people_requiring_medical_support >= 0",
            name="ck_rescue_mission_medical_support",
        ),
        CheckConstraint(
            "ai_confidence_score IS NULL OR "
            "(ai_confidence_score >= 0 "
            "AND ai_confidence_score <= 1)",
            name="ck_rescue_mission_ai_confidence",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_rescue_mission_version",
        ),
        Index(
            "ix_rescue_mission_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_rescue_mission_priority_status",
            "priority",
            "status",
        ),
        Index(
            "ix_rescue_mission_incident",
            "incident_id",
            "created_at",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "mission_number",
        "title",
        "description",
    )
    def validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "mission_number": 120,
            "title": 255,
            "description": 10000,
        }

        return require_text(
            value,
            key,
            limits[key],
        )

    @validates("latitude")
    def validate_latitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -90, 90
        )

    @validates("longitude")
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -180, 180
        )

    @validates("ai_confidence_score")
    def validate_ai_confidence(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_probability(value, key)

    @validates(
        "estimated_people_at_risk",
        "people_confirmed_at_risk",
        "people_rescued",
        "people_evacuated",
        "people_transferred_to_medical",
        "people_unaccounted_for",
        "children_at_risk",
        "elderly_at_risk",
        "people_with_disabilities_at_risk",
        "people_requiring_medical_support",
    )
    def validate_headcount(
        self,
        key: str,
        value: int,
    ) -> int:
        result = validate_non_negative_int(
            value, key
        )
        assert result is not None
        return result

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            RescueMissionStatus.COMPLETED,
            RescueMissionStatus.PARTIALLY_COMPLETED,
            RescueMissionStatus.FAILED,
            RescueMissionStatus.CANCELLED,
        }

    @property
    def is_human_approved(self) -> bool:
        return (
            self.approved_by_user_id is not None
            and self.approved_at is not None
            and bool(self.approval_reference)
        )

    @property
    def has_valid_route(self) -> bool:
        if (
            self.route_status
            != RescueRouteStatus.VERIFIED_SAFE
        ):
            return False

        if self.route_verified_by_user_id is None:
            return False

        if self.route_valid_until is None:
            return False

        return (
            as_utc(self.route_valid_until)
            > utc_now()
        )

    @property
    def has_approved_team(self) -> bool:
        return any(
            assignment.status in {
                RescueAssignmentStatus.APPROVED,
                RescueAssignmentStatus.ACCEPTED,
            }
            for assignment in self.assignments
        )

    @property
    def can_dispatch(self) -> bool:
        return (
            self.status == RescueMissionStatus.READY
            and self.is_verified
            and self.is_human_approved
            and self.has_valid_route
            and self.has_approved_team
            and self.risk_level
            != RescueRiskLevel.UNKNOWN
        )

    def submit_for_verification(self) -> None:
        if self.status != RescueMissionStatus.DRAFT:
            raise ValueError(
                "Only draft missions can be submitted"
            )

        self.status = (
            RescueMissionStatus.PENDING_VERIFICATION
        )

    def verify(self, verifier_id: int) -> None:
        if self.status != (
            RescueMissionStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Mission is not pending verification"
            )

        if verifier_id <= 0:
            raise ValueError(
                "Invalid verifier"
            )

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
        self.status = RescueMissionStatus.VERIFIED

    def begin_planning(self) -> None:
        if self.status != RescueMissionStatus.VERIFIED:
            raise ValueError(
                "Mission must be verified"
            )

        self.status = RescueMissionStatus.PLANNING

    def request_approval(self) -> None:
        if self.status != RescueMissionStatus.PLANNING:
            raise ValueError(
                "Mission is not in planning"
            )

        self.status = (
            RescueMissionStatus.PENDING_APPROVAL
        )

    def approve(
        self,
        *,
        approver_id: int,
        approval_reference: str,
    ) -> None:
        if self.status != (
            RescueMissionStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Mission is not pending approval"
            )

        if approver_id <= 0:
            raise ValueError(
                "Invalid approver"
            )

        if approver_id in {
            self.created_by_user_id,
            self.verified_by_user_id,
        }:
            raise ValueError(
                "Independent approval required"
            )

        self.approval_reference = require_text(
            approval_reference,
            "approval_reference",
            500,
        )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = RescueMissionStatus.APPROVED

    def mark_ready(self) -> None:
        if self.status != RescueMissionStatus.APPROVED:
            raise ValueError(
                "Mission is not approved"
            )

        self.status = RescueMissionStatus.READY

    def dispatch(self, dispatcher_id: int) -> None:
        """
        The service must independently verify:
        - Dispatcher permissions
        - Rescue team availability and qualifications
        - Equipment and vehicle readiness
        - Current weather and water conditions
        - Route safety and operational authorization
        - Audit transaction and external dispatch result
        """
        if not self.can_dispatch:
            raise ValueError(
                "Mission does not meet dispatch requirements"
            )

        if dispatcher_id <= 0:
            raise ValueError(
                "Invalid dispatcher"
            )

        self.dispatched_by_user_id = dispatcher_id
        self.dispatched_at = utc_now()
        self.status = RescueMissionStatus.DISPATCHED

    def mark_en_route(self) -> None:
        if self.status != RescueMissionStatus.DISPATCHED:
            raise ValueError(
                "Mission has not been dispatched"
            )

        self.status = RescueMissionStatus.EN_ROUTE

    def mark_on_scene(self) -> None:
        if self.status != RescueMissionStatus.EN_ROUTE:
            raise ValueError(
                "Mission is not en route"
            )

        self.arrived_on_scene_at = utc_now()
        self.status = RescueMissionStatus.ON_SCENE

    def start_rescue(self) -> None:
        if self.status != RescueMissionStatus.ON_SCENE:
            raise ValueError(
                "Mission team is not on scene"
            )

        self.started_at = utc_now()
        self.status = (
            RescueMissionStatus.RESCUE_IN_PROGRESS
        )

    def record_headcount(
        self,
        *,
        rescued: int,
        evacuated: int,
        medical_transfers: int,
        unaccounted_for: int,
    ) -> None:
        if self.status not in {
            RescueMissionStatus.RESCUE_IN_PROGRESS,
            RescueMissionStatus.EVACUATING,
            RescueMissionStatus.RETURNING,
        }:
            raise ValueError(
                "Mission is not in an active rescue phase"
            )

        values = {
            "rescued": rescued,
            "evacuated": evacuated,
            "medical_transfers": medical_transfers,
            "unaccounted_for": unaccounted_for,
        }

        for key, value in values.items():
            validate_non_negative_int(value, key)

        if evacuated > rescued:
            raise ValueError(
                "Evacuated count cannot exceed rescued count"
            )

        if medical_transfers > rescued:
            raise ValueError(
                "Medical transfers cannot exceed rescued count"
            )

        if (
            self.people_confirmed_at_risk > 0
            and rescued + unaccounted_for
            > self.people_confirmed_at_risk
        ):
            raise ValueError(
                "Headcount exceeds confirmed people at risk"
            )

        self.people_rescued = rescued
        self.people_evacuated = evacuated
        self.people_transferred_to_medical = (
            medical_transfers
        )
        self.people_unaccounted_for = (
            unaccounted_for
        )

    def pause(self, reason: str) -> None:
        if self.status not in {
            RescueMissionStatus.EN_ROUTE,
            RescueMissionStatus.ON_SCENE,
            RescueMissionStatus.RESCUE_IN_PROGRESS,
            RescueMissionStatus.EVACUATING,
            RescueMissionStatus.RETURNING,
        }:
            raise ValueError(
                "Mission cannot be paused"
            )

        self.safety_notes = require_text(
            reason,
            "reason",
            5000,
        )
        self.status = RescueMissionStatus.PAUSED

    def complete(
        self,
        *,
        verifier_id: int,
        outcome: RescueOutcome,
        summary: str,
    ) -> None:
        if self.status not in {
            RescueMissionStatus.RESCUE_IN_PROGRESS,
            RescueMissionStatus.EVACUATING,
            RescueMissionStatus.RETURNING,
        }:
            raise ValueError(
                "Mission cannot be completed"
            )

        if verifier_id <= 0:
            raise ValueError(
                "Invalid outcome verifier"
            )

        if outcome not in {
            RescueOutcome.ALL_RESCUED,
            RescueOutcome.PARTIALLY_RESCUED,
            RescueOutcome.NO_PERSON_FOUND,
            RescueOutcome.TRANSFERRED_TO_OTHER_TEAM,
            RescueOutcome.UNSAFE_TO_PROCEED,
            RescueOutcome.OTHER,
        }:
            raise ValueError(
                "Invalid completion outcome"
            )

        if outcome == RescueOutcome.ALL_RESCUED:
            if self.people_unaccounted_for != 0:
                raise ValueError(
                    "Unaccounted persons remain"
                )

            if (
                self.people_confirmed_at_risk > 0
                and self.people_rescued
                < self.people_confirmed_at_risk
            ):
                raise ValueError(
                    "Rescued count is below confirmed count"
                )

        self.outcome_summary = require_text(
            summary,
            "summary",
            10000,
        )

        self.outcome = outcome
        self.outcome_verified_by_user_id = verifier_id
        self.outcome_verified_at = utc_now()
        self.completed_at = utc_now()

        if outcome == RescueOutcome.ALL_RESCUED:
            self.status = RescueMissionStatus.COMPLETED
        else:
            self.status = (
                RescueMissionStatus.PARTIALLY_COMPLETED
            )

    def cancel(self, reason: str) -> None:
        if self.is_terminal:
            raise ValueError(
                "Terminal mission cannot be cancelled"
            )

        if self.status in {
            RescueMissionStatus.DISPATCHED,
            RescueMissionStatus.EN_ROUTE,
            RescueMissionStatus.ON_SCENE,
            RescueMissionStatus.RESCUE_IN_PROGRESS,
            RescueMissionStatus.EVACUATING,
            RescueMissionStatus.RETURNING,
            RescueMissionStatus.PAUSED,
        }:
            raise ValueError(
                "Active rescue missions require "
                "command-authorized stand-down procedures"
            )

        self.cancellation_reason = require_text(
            reason,
            "reason",
            5000,
        )
        self.outcome = RescueOutcome.CANCELLED
        self.status = RescueMissionStatus.CANCELLED

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mission_number": self.mission_number,
            "district_id": self.district_id,
            "mission_type": self.mission_type.value,
            "priority": self.priority.value,
            "risk_level": self.risk_level.value,
            "status": self.status.value,
            "title": self.title,
            "people_rescued": self.people_rescued,
            "people_evacuated": self.people_evacuated,
            "people_unaccounted_for": (
                self.people_unaccounted_for
            ),
            "is_verified": self.is_verified,
            "is_human_approved": self.is_human_approved,
            "outcome": self.outcome.value,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }


# ============================================================
# RESCUE TEAM ASSIGNMENT
# ============================================================

class RescueMissionAssignment(Base):
    __tablename__ = "rescue_mission_assignments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    mission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_missions.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    rescue_team_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )

    team_leader_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    vehicle_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    transport_mode: Mapped[
        RescueTransportMode
    ] = mapped_column(
        enum_type(RescueTransportMode),
        nullable=False,
    )

    assigned_member_count: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    available_rescue_capacity: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    status: Mapped[
        RescueAssignmentStatus
    ] = mapped_column(
        enum_type(RescueAssignmentStatus),
        default=RescueAssignmentStatus.PROPOSED,
        nullable=False,
    )

    assignment_reason: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    requested_by_user_id: Mapped[
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

    approved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    accepted_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    completed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
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

    mission: Mapped["RescueMission"] = relationship(
        "RescueMission",
        back_populates="assignments",
    )

    __table_args__ = (
        UniqueConstraint(
            "mission_id",
            "rescue_team_id",
            name="uq_rescue_mission_team",
        ),
        CheckConstraint(
            "assigned_member_count >= 0",
            name="ck_rescue_assignment_members",
        ),
        CheckConstraint(
            "available_rescue_capacity >= 0",
            name="ck_rescue_assignment_capacity",
        ),
        Index(
            "ix_rescue_assignment_team_status",
            "rescue_team_id",
            "status",
        ),
    )

    def approve(self, approver_id: int) -> None:
        if self.status != (
            RescueAssignmentStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Assignment is not pending approval"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if (
            self.requested_by_user_id is not None
            and approver_id == self.requested_by_user_id
        ):
            raise ValueError(
                "Independent approval required"
            )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = RescueAssignmentStatus.APPROVED


# ============================================================
# RESCUE CHECKPOINT
# ============================================================

class RescueMissionCheckpoint(Base):
    __tablename__ = "rescue_mission_checkpoints"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    mission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_missions.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    checkpoint_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
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

    status: Mapped[
        RescueCheckpointStatus
    ] = mapped_column(
        enum_type(RescueCheckpointStatus),
        default=RescueCheckpointStatus.PENDING,
        nullable=False,
    )

    is_mandatory: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
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

    route_safe_at_arrival: Mapped[
        bool | None
    ] = mapped_column(
        Boolean,
        nullable=True,
    )

    verified_by_user_id: Mapped[
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

    mission: Mapped["RescueMission"] = relationship(
        "RescueMission",
        back_populates="checkpoints",
    )

    __table_args__ = (
        UniqueConstraint(
            "mission_id",
            "checkpoint_number",
            name="uq_rescue_mission_checkpoint",
        ),
        CheckConstraint(
            "checkpoint_number > 0",
            name="ck_rescue_checkpoint_number",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_rescue_checkpoint_lat",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_rescue_checkpoint_lon",
        ),
    )


# ============================================================
# RESCUE EQUIPMENT
# ============================================================

class RescueMissionEquipment(Base):
    __tablename__ = "rescue_mission_equipment"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    mission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_missions.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    equipment_category: Mapped[
        RescueEquipmentCategory
    ] = mapped_column(
        enum_type(RescueEquipmentCategory),
        nullable=False,
    )

    equipment_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    required_quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    allocated_quantity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    status: Mapped[
        RescueEquipmentStatus
    ] = mapped_column(
        enum_type(RescueEquipmentStatus),
        default=RescueEquipmentStatus.REQUIRED,
        nullable=False,
    )

    is_safety_critical: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    condition_verified: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
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

    mission: Mapped["RescueMission"] = relationship(
        "RescueMission",
        back_populates="equipment",
    )

    __table_args__ = (
        CheckConstraint(
            "required_quantity > 0",
            name="ck_rescue_equipment_required",
        ),
        CheckConstraint(
            "allocated_quantity >= 0",
            name="ck_rescue_equipment_allocated",
        ),
        CheckConstraint(
            "allocated_quantity <= required_quantity",
            name="ck_rescue_equipment_allocation_limit",
        ),
    )


# ============================================================
# RESCUE EVIDENCE
# ============================================================

class RescueMissionEvidence(Base):
    __tablename__ = "rescue_mission_evidence"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    mission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_missions.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    evidence_type: Mapped[
        RescueEvidenceType
    ] = mapped_column(
        enum_type(RescueEvidenceType),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    storage_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    content_sha256: Mapped[str | None] = (
        mapped_column(
            String(64),
            nullable=True,
        )
    )

    contains_sensitive_person_data: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    malware_scan_passed: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    uploaded_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    observed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
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

    mission: Mapped["RescueMission"] = relationship(
        "RescueMission",
        back_populates="evidence",
    )

    __table_args__ = (
        CheckConstraint(
            "content_sha256 IS NULL OR "
            "length(content_sha256) = 64",
            name="ck_rescue_evidence_sha256",
        ),
    )


# ============================================================
# RESCUE EVENT HISTORY
# ============================================================

class RescueMissionEvent(Base):
    __tablename__ = "rescue_mission_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    mission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_missions.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        RescueMissionEventType
    ] = mapped_column(
        enum_type(RescueMissionEventType),
        nullable=False,
    )

    previous_status: Mapped[
        RescueMissionStatus | None
    ] = mapped_column(
        enum_type(RescueMissionStatus),
        nullable=True,
    )

    new_status: Mapped[
        RescueMissionStatus | None
    ] = mapped_column(
        enum_type(RescueMissionStatus),
        nullable=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
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

    mission: Mapped["RescueMission"] = relationship(
        "RescueMission",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_rescue_event_mission_time",
            "mission_id",
            "occurred_at",
        ),
    )


# ============================================================
# ORM TIMESTAMP EVENT
# ============================================================

@event.listens_for(
    RescueMission,
    "before_update",
)
def update_rescue_mission_timestamp(
    mapper: Any,
    connection: Any,
    target: RescueMission,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

RESCUE_MISSION_MODELS = (
    RescueMission,
    RescueMissionAssignment,
    RescueMissionCheckpoint,
    RescueMissionEquipment,
    RescueMissionEvidence,
    RescueMissionEvent,
)


__all__ = [
    "RescueMission",
    "RescueMissionAssignment",
    "RescueMissionCheckpoint",
    "RescueMissionEquipment",
    "RescueMissionEvidence",
    "RescueMissionEvent",
    "RescueMissionStatus",
    "RescueMissionType",
    "RescueMissionPriority",
    "RescueRiskLevel",
    "RescueTransportMode",
    "RescueRouteStatus",
    "RescueAssignmentStatus",
    "RescueCheckpointStatus",
    "RescueEquipmentCategory",
    "RescueEquipmentStatus",
    "RescueOutcome",
    "RescueEvidenceType",
    "RescueMissionEventType",
    "RESCUE_MISSION_MODELS",
]
