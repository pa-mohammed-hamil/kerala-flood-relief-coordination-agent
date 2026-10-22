
"""
backend/models/incident.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x ORM models for emergency incident
management and response coordination.

Features:
- Flood and disaster incident registration
- District and geographic tracking
- Incident severity, priority and status
- SOS and flood alert references
- Human verification and approval
- Rescue and evacuation coordination
- Incident impact assessments
- Resource requirements
- Incident assignments
- Timeline and audit events
- AI-generated incident assessments
- Human-reviewed resolution and closure
- Optimistic concurrency control

SAFETY:
AI-generated incident assessments are advisory.
This module does not automatically dispatch teams,
issue evacuation orders or publish public warnings.

Live response decisions require authorized human
oversight and verified operational information.
"""

from __future__ import annotations

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
    max_length: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise ValueError(
            f"{field_name} exceeds {max_length} characters"
        )

    return result


def validate_non_negative(
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


# ============================================================
# ENUMERATIONS
# ============================================================

class IncidentType(str, enum.Enum):
    FLOOD = "flood"
    FLASH_FLOOD = "flash_flood"
    URBAN_FLOODING = "urban_flooding"
    RIVER_OVERFLOW = "river_overflow"
    DAM_RELATED_FLOOD = "dam_related_flood"
    LANDSLIDE = "landslide"
    BUILDING_COLLAPSE = "building_collapse"
    ROAD_BLOCKAGE = "road_blockage"
    BRIDGE_DAMAGE = "bridge_damage"
    POWER_FAILURE = "power_failure"
    WATER_SUPPLY_FAILURE = "water_supply_failure"
    MEDICAL_EMERGENCY = "medical_emergency"
    STRANDED_PEOPLE = "stranded_people"
    MISSING_PERSON = "missing_person"
    RESCUE_REQUIRED = "rescue_required"
    SHELTER_EMERGENCY = "shelter_emergency"
    RELIEF_SHORTAGE = "relief_shortage"
    COMMUNICATION_FAILURE = "communication_failure"
    OTHER = "other"


class IncidentSeverity(str, enum.Enum):
    UNKNOWN = "unknown"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


class IncidentPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"


class IncidentStatus(str, enum.Enum):
    REPORTED = "reported"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    TRIAGED = "triaged"
    PENDING_RESPONSE_APPROVAL = (
        "pending_response_approval"
    )
    RESPONSE_APPROVED = "response_approved"
    RESPONSE_IN_PROGRESS = "response_in_progress"
    STABILIZED = "stabilized"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    CANCELLED = "cancelled"


class IncidentSourceType(str, enum.Enum):
    SOS_REQUEST = "sos_request"
    CITIZEN_REPORT = "citizen_report"
    VOLUNTEER = "volunteer"
    RESCUE_TEAM = "rescue_team"
    DISTRICT_CONTROL_ROOM = "district_control_room"
    OFFICIAL_AUTHORITY = "official_authority"
    WEATHER_STATION = "weather_station"
    RIVER_STATION = "river_station"
    DAM_MONITORING = "dam_monitoring"
    SATELLITE = "satellite"
    DRONE = "drone"
    AI_MODEL = "ai_model"
    AI_AGENT = "ai_agent"
    MANUAL_ENTRY = "manual_entry"
    OTHER = "other"


class IncidentVerificationStatus(str, enum.Enum):
    UNVERIFIED = "unverified"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    REJECTED = "rejected"


class IncidentResourceType(str, enum.Enum):
    RESCUE_TEAM = "rescue_team"
    RESCUE_BOAT = "rescue_boat"
    AMBULANCE = "ambulance"
    MEDICAL_TEAM = "medical_team"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    HELICOPTER = "helicopter"
    SHELTER = "shelter"
    FOOD_KIT = "food_kit"
    DRINKING_WATER = "drinking_water"
    MEDICINE = "medicine"
    FIRST_AID_KIT = "first_aid_kit"
    GENERATOR = "generator"
    COMMUNICATION_EQUIPMENT = (
        "communication_equipment"
    )
    HEAVY_EQUIPMENT = "heavy_equipment"
    VOLUNTEER = "volunteer"
    OTHER = "other"


class IncidentAssignmentStatus(str, enum.Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACCEPTED = "accepted"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class IncidentEventType(str, enum.Enum):
    CREATED = "created"
    UPDATED = "updated"
    VERIFICATION_REQUESTED = "verification_requested"
    VERIFIED = "verified"
    REJECTED = "rejected"
    TRIAGED = "triaged"
    PRIORITY_CHANGED = "priority_changed"
    SEVERITY_CHANGED = "severity_changed"
    ASSESSMENT_ADDED = "assessment_added"
    RESPONSE_REQUESTED = "response_requested"
    RESPONSE_APPROVED = "response_approved"
    RESOURCE_REQUESTED = "resource_requested"
    TEAM_ASSIGNED = "team_assigned"
    RESPONSE_STARTED = "response_started"
    STABILIZED = "stabilized"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"
    MARKED_DUPLICATE = "marked_duplicate"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"
    NOTE_ADDED = "note_added"


# ============================================================
# INCIDENT MASTER
# ============================================================

class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    incident_number: Mapped[str] = mapped_column(
        String(100),
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

    sos_request_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    flood_alert_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    evacuation_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    rescue_mission_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            nullable=True,
        )
    )

    incident_type: Mapped[IncidentType] = (
        mapped_column(
            enum_type(IncidentType),
            nullable=False,
            index=True,
        )
    )

    severity: Mapped[IncidentSeverity] = (
        mapped_column(
            enum_type(IncidentSeverity),
            default=IncidentSeverity.UNKNOWN,
            nullable=False,
            index=True,
        )
    )

    priority: Mapped[IncidentPriority] = (
        mapped_column(
            enum_type(IncidentPriority),
            default=IncidentPriority.MEDIUM,
            nullable=False,
            index=True,
        )
    )

    status: Mapped[IncidentStatus] = mapped_column(
        enum_type(IncidentStatus),
        default=IncidentStatus.REPORTED,
        nullable=False,
        index=True,
    )

    verification_status: Mapped[
        IncidentVerificationStatus
    ] = mapped_column(
        enum_type(IncidentVerificationStatus),
        default=IncidentVerificationStatus.UNVERIFIED,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    location_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
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

    estimated_affected_people: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    people_stranded: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    people_injured: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    people_missing: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    fatalities_reported: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    people_rescued: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    requires_rescue: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_evacuation: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    requires_medical_response: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_relief_supplies: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    immediate_danger: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    road_accessible: Mapped[bool | None] = (
        mapped_column(
            Boolean,
            nullable=True,
        )
    )

    source_type: Mapped[
        IncidentSourceType
    ] = mapped_column(
        enum_type(IncidentSourceType),
        nullable=False,
    )

    source_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    reporter_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    is_ai_generated: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    ai_confidence: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    verified_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    requires_human_approval: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    response_approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    response_approval_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    response_approved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    assigned_coordinator_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    duplicate_of_incident_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "incidents.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    resolution_summary: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    resolution_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    closure_approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    audit_reference: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    occurred_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    reported_at: Mapped[datetime] = (
        mapped_column(
            DateTime(timezone=True),
            default=utc_now,
            nullable=False,
        )
    )

    response_started_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    stabilized_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    resolved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    closed_at: Mapped[datetime | None] = (
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

    assessments: Mapped[
        list["IncidentAssessment"]
    ] = relationship(
        "IncidentAssessment",
        back_populates="incident",
        cascade="save-update, merge",
    )

    resource_requests: Mapped[
        list["IncidentResourceRequest"]
    ] = relationship(
        "IncidentResourceRequest",
        back_populates="incident",
        cascade="save-update, merge",
    )

    assignments: Mapped[
        list["IncidentAssignment"]
    ] = relationship(
        "IncidentAssignment",
        back_populates="incident",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["IncidentEvent"]
    ] = relationship(
        "IncidentEvent",
        back_populates="incident",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_incident_district",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_incident_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_incident_longitude",
        ),
        CheckConstraint(
            "ai_confidence IS NULL OR "
            "(ai_confidence >= 0 AND ai_confidence <= 1)",
            name="ck_incident_ai_confidence",
        ),
        CheckConstraint(
            "estimated_affected_people IS NULL OR "
            "estimated_affected_people >= 0",
            name="ck_incident_affected",
        ),
        CheckConstraint(
            "people_stranded IS NULL OR people_stranded >= 0",
            name="ck_incident_stranded",
        ),
        CheckConstraint(
            "people_injured IS NULL OR people_injured >= 0",
            name="ck_incident_injured",
        ),
        CheckConstraint(
            "people_missing IS NULL OR people_missing >= 0",
            name="ck_incident_missing",
        ),
        CheckConstraint(
            "fatalities_reported IS NULL OR "
            "fatalities_reported >= 0",
            name="ck_incident_fatalities",
        ),
        CheckConstraint(
            "people_rescued IS NULL OR people_rescued >= 0",
            name="ck_incident_rescued",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_incident_version",
        ),
        Index(
            "ix_incident_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_incident_priority_status",
            "priority",
            "status",
        ),
        Index(
            "ix_incident_type_reported",
            "incident_type",
            "reported_at",
        ),
        Index(
            "ix_incident_severity_priority",
            "severity",
            "priority",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "incident_number",
        "title",
        "description",
        "location_name",
    )
    def validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "incident_number": 100,
            "title": 255,
            "description": 10000,
            "location_name": 255,
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

    @validates("ai_confidence")
    def validate_ai_confidence(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_probability(
            value, key
        )

    @validates(
        "estimated_affected_people",
        "people_stranded",
        "people_injured",
        "people_missing",
        "fatalities_reported",
        "people_rescued",
    )
    def validate_population_count(
        self,
        key: str,
        value: int | None,
    ) -> int | None:
        return validate_non_negative(
            value, key
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            IncidentStatus.CLOSED,
            IncidentStatus.REJECTED,
            IncidentStatus.DUPLICATE,
            IncidentStatus.CANCELLED,
        }

    @property
    def is_response_approved(self) -> bool:
        return (
            self.response_approved_by_user_id is not None
            and self.response_approved_at is not None
            and bool(self.response_approval_reference)
        )

    def request_verification(self) -> None:
        if self.status != IncidentStatus.REPORTED:
            raise ValueError(
                "Only reported incidents can request verification"
            )

        self.status = IncidentStatus.PENDING_VERIFICATION
        self.verification_status = (
            IncidentVerificationStatus.UNDER_REVIEW
        )

    def verify(
        self,
        *,
        verifier_id: int,
    ) -> None:
        if self.status != (
            IncidentStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Incident is not pending verification"
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
        self.verification_status = (
            IncidentVerificationStatus.VERIFIED
        )
        self.status = IncidentStatus.VERIFIED

    def triage(
        self,
        *,
        priority: IncidentPriority,
        severity: IncidentSeverity,
    ) -> None:
        if self.status != IncidentStatus.VERIFIED:
            raise ValueError(
                "Incident must be verified before triage"
            )

        self.priority = priority
        self.severity = severity
        self.status = IncidentStatus.TRIAGED

    def request_response_approval(self) -> None:
        if self.status != IncidentStatus.TRIAGED:
            raise ValueError(
                "Incident must be triaged"
            )

        self.status = (
            IncidentStatus.PENDING_RESPONSE_APPROVAL
        )

    def approve_response(
        self,
        *,
        approver_id: int,
        approval_reference: str,
    ) -> None:
        if self.status != (
            IncidentStatus.PENDING_RESPONSE_APPROVAL
        ):
            raise ValueError(
                "Incident is not pending response approval"
            )

        if approver_id <= 0:
            raise ValueError(
                "Invalid approver"
            )

        if (
            self.created_by_user_id is not None
            and approver_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent response approval required"
            )

        self.response_approval_reference = (
            require_text(
                approval_reference,
                "approval_reference",
                500,
            )
        )

        self.response_approved_by_user_id = approver_id
        self.response_approved_at = utc_now()
        self.status = IncidentStatus.RESPONSE_APPROVED

    def start_response(self) -> None:
        if self.status != IncidentStatus.RESPONSE_APPROVED:
            raise ValueError(
                "Incident response is not approved"
            )

        if not self.is_response_approved:
            raise ValueError(
                "Missing response authorization"
            )

        self.status = IncidentStatus.RESPONSE_IN_PROGRESS
        self.response_started_at = utc_now()

    def stabilize(self) -> None:
        if self.status != (
            IncidentStatus.RESPONSE_IN_PROGRESS
        ):
            raise ValueError(
                "Response must be in progress"
            )

        self.status = IncidentStatus.STABILIZED
        self.stabilized_at = utc_now()

    def resolve(
        self,
        *,
        summary: str,
        verifier_id: int,
    ) -> None:
        if self.status != IncidentStatus.STABILIZED:
            raise ValueError(
                "Incident must be stabilized first"
            )

        if verifier_id <= 0:
            raise ValueError(
                "Invalid resolution verifier"
            )

        self.resolution_summary = require_text(
            summary,
            "summary",
            10000,
        )

        self.resolution_verified_by_user_id = (
            verifier_id
        )
        self.status = IncidentStatus.RESOLVED
        self.resolved_at = utc_now()

    def close(
        self,
        *,
        approver_id: int,
    ) -> None:
        if self.status != IncidentStatus.RESOLVED:
            raise ValueError(
                "Incident must be resolved"
            )

        if approver_id <= 0:
            raise ValueError(
                "Invalid closure approver"
            )

        if approver_id == (
            self.resolution_verified_by_user_id
        ):
            raise ValueError(
                "Closure approval must be independent"
            )

        self.closure_approved_by_user_id = approver_id
        self.status = IncidentStatus.CLOSED
        self.closed_at = utc_now()

    def mark_duplicate(
        self,
        original_incident_id: str,
    ) -> None:
        if self.is_terminal:
            raise ValueError(
                "Terminal incident cannot be marked duplicate"
            )

        original_incident_id = require_text(
            original_incident_id,
            "original_incident_id",
            36,
        )

        if original_incident_id == self.id:
            raise ValueError(
                "Incident cannot duplicate itself"
            )

        self.duplicate_of_incident_id = (
            original_incident_id
        )
        self.status = IncidentStatus.DUPLICATE

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "incident_number": self.incident_number,
            "district_id": self.district_id,
            "incident_type": self.incident_type.value,
            "severity": self.severity.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "title": self.title,
            "location_name": self.location_name,
            "is_verified": self.is_verified,
            "is_ai_generated": self.is_ai_generated,
            "requires_rescue": self.requires_rescue,
            "requires_evacuation": (
                self.requires_evacuation
            ),
            "requires_medical_response": (
                self.requires_medical_response
            ),
            "is_response_approved": (
                self.is_response_approved
            ),
            "reported_at": (
                self.reported_at.isoformat()
                if self.reported_at
                else None
            ),
        }


# ============================================================
# INCIDENT ASSESSMENT
# ============================================================

class IncidentAssessment(Base):
    __tablename__ = "incident_assessments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    incident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "incidents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    assessed_severity: Mapped[
        IncidentSeverity
    ] = mapped_column(
        enum_type(IncidentSeverity),
        nullable=False,
    )

    recommended_priority: Mapped[
        IncidentPriority
    ] = mapped_column(
        enum_type(IncidentPriority),
        nullable=False,
    )

    assessment_summary: Mapped[str] = (
        mapped_column(
            Text,
            nullable=False,
        )
    )

    recommended_actions: Mapped[dict] = (
        mapped_column(
            MutableDict.as_mutable(JSON),
            default=dict,
            nullable=False,
        )
    )

    identified_risks: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    evidence_references: Mapped[dict] = (
        mapped_column(
            MutableDict.as_mutable(JSON),
            default=dict,
            nullable=False,
        )
    )

    confidence_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    uncertainty_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    is_ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    agent_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    requires_human_review: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=True,
            nullable=False,
        )
    )

    is_human_verified: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    assessed_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    verified_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    reviewed_at: Mapped[datetime | None] = (
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

    incident: Mapped["Incident"] = relationship(
        "Incident",
        back_populates="assessments",
    )

    __table_args__ = (
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 "
            "AND confidence_score <= 1)",
            name="ck_incident_assessment_confidence",
        ),
        CheckConstraint(
            "uncertainty_score IS NULL OR "
            "(uncertainty_score >= 0 "
            "AND uncertainty_score <= 1)",
            name="ck_incident_assessment_uncertainty",
        ),
        Index(
            "ix_incident_assessment_time",
            "incident_id",
            "created_at",
        ),
    )

    @validates(
        "confidence_score",
        "uncertainty_score",
    )
    def validate_assessment_score(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_probability(
            value, key
        )


# ============================================================
# INCIDENT RESOURCE REQUEST
# ============================================================

class IncidentResourceRequest(Base):
    __tablename__ = "incident_resource_requests"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    incident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "incidents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    resource_type: Mapped[
        IncidentResourceType
    ] = mapped_column(
        enum_type(IncidentResourceType),
        nullable=False,
    )

    requested_quantity: Mapped[int] = (
        mapped_column(
            Integer,
            nullable=False,
        )
    )

    approved_quantity: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
            nullable=False,
        )
    )

    allocated_quantity: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
            nullable=False,
        )
    )

    fulfilled_quantity: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
            nullable=False,
        )
    )

    unit: Mapped[str] = mapped_column(
        String(50),
        default="units",
        nullable=False,
    )

    justification: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    requested_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    approved_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    approved_at: Mapped[datetime | None] = (
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

    incident: Mapped["Incident"] = relationship(
        "Incident",
        back_populates="resource_requests",
    )

    __table_args__ = (
        CheckConstraint(
            "requested_quantity > 0",
            name="ck_incident_resource_requested",
        ),
        CheckConstraint(
            "approved_quantity >= 0",
            name="ck_incident_resource_approved",
        ),
        CheckConstraint(
            "allocated_quantity >= 0",
            name="ck_incident_resource_allocated",
        ),
        CheckConstraint(
            "fulfilled_quantity >= 0",
            name="ck_incident_resource_fulfilled",
        ),
        CheckConstraint(
            "approved_quantity <= requested_quantity",
            name="ck_incident_resource_approval_limit",
        ),
        CheckConstraint(
            "allocated_quantity <= approved_quantity",
            name="ck_incident_resource_allocation_limit",
        ),
        CheckConstraint(
            "fulfilled_quantity <= allocated_quantity",
            name="ck_incident_resource_fulfillment_limit",
        ),
        Index(
            "ix_incident_resource_type",
            "incident_id",
            "resource_type",
        ),
    )


# ============================================================
# INCIDENT ASSIGNMENT
# ============================================================

class IncidentAssignment(Base):
    __tablename__ = "incident_assignments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    incident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "incidents.id",
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

    assigned_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    assignment_role: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    status: Mapped[
        IncidentAssignmentStatus
    ] = mapped_column(
        enum_type(IncidentAssignmentStatus),
        default=IncidentAssignmentStatus.PROPOSED,
        nullable=False,
    )

    assigned_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    approved_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
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

    incident: Mapped["Incident"] = relationship(
        "Incident",
        back_populates="assignments",
    )

    __table_args__ = (
        Index(
            "ix_incident_assignment_status",
            "incident_id",
            "status",
        ),
    )


# ============================================================
# INCIDENT EVENT HISTORY
# ============================================================

class IncidentEvent(Base):
    __tablename__ = "incident_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    incident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "incidents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        IncidentEventType
    ] = mapped_column(
        enum_type(IncidentEventType),
        nullable=False,
    )

    previous_status: Mapped[
        IncidentStatus | None
    ] = mapped_column(
        enum_type(IncidentStatus),
        nullable=True,
    )

    new_status: Mapped[
        IncidentStatus | None
    ] = mapped_column(
        enum_type(IncidentStatus),
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

    actor_agent_name: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    audit_reference: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
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

    incident: Mapped["Incident"] = relationship(
        "Incident",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_incident_event_time",
            "incident_id",
            "occurred_at",
        ),
    )


# ============================================================
# ORM EVENTS
# ============================================================

@event.listens_for(
    Incident,
    "before_update",
)
def update_incident_timestamp(
    mapper: Any,
    connection: Any,
    target: Incident,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

INCIDENT_MODELS = (
    Incident,
    IncidentAssessment,
    IncidentResourceRequest,
    IncidentAssignment,
    IncidentEvent,
)


__all__ = [
    "Incident",
    "IncidentAssessment",
    "IncidentResourceRequest",
    "IncidentAssignment",
    "IncidentEvent",
    "IncidentType",
    "IncidentSeverity",
    "IncidentPriority",
    "IncidentStatus",
    "IncidentSourceType",
    "IncidentVerificationStatus",
    "IncidentResourceType",
    "IncidentAssignmentStatus",
    "IncidentEventType",
    "INCIDENT_MODELS",
]
