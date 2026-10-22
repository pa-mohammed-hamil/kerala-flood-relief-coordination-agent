from __future__ import annotations

"""
SOS request models for Kerala Flood Relief Coordination Agent.

Python 3.11+ | SQLAlchemy 2.x

Features:
- Emergency SOS registration
- Flood rescue and medical emergency requests
- GPS and location tracking
- Emergency priority and triage
- Human-verified dispatch authorization
- Rescue team assignment
- SOS status tracking
- Emergency contact notifications
- SOS evidence attachments
- Duplicate and false-alarm management
- Escalation and response history
- AI-assisted risk assessment
- Audit events

Safety:
- AI risk scores are advisory only.
- Emergency dispatch requires authorized human approval.
- SOS reports must not be silently discarded.
- Contact details and exact locations require access control.
- Emergency response integrations belong in the service layer.
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
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(cls: type[enum.Enum]) -> SQLEnum:
    return SQLEnum(
        cls,
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
        raise ValueError(f"{field_name} is required")

    result = value.strip()

    if len(result) > maximum:
        raise ValueError(
            f"{field_name} exceeds {maximum} characters"
        )

    return result


def require_user_id(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError("Valid user ID required")

    return value


def finite_number(
    value: float | int | None,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"{field_name} must be a finite number"
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise ValueError(
            f"{field_name} is below minimum"
        )

    if maximum is not None and result > maximum:
        raise ValueError(
            f"{field_name} exceeds maximum"
        )

    return result


# ============================================================
# ENUMS
# ============================================================

class SOSRequestType(str, enum.Enum):
    FLOOD_RESCUE = "flood_rescue"
    STRANDED_PERSON = "stranded_person"
    MEDICAL_EMERGENCY = "medical_emergency"
    EVACUATION = "evacuation"
    LANDSLIDE = "landslide"
    BUILDING_COLLAPSE = "building_collapse"
    DROWNING_RISK = "drowning_risk"
    MISSING_PERSON = "missing_person"
    FIRE = "fire"
    FOOD_WATER = "food_water"
    SHELTER = "shelter"
    ACCESSIBILITY_ASSISTANCE = "accessibility_assistance"
    OTHER = "other"


class SOSRequestStatus(str, enum.Enum):
    RECEIVED = "received"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    TRIAGED = "triaged"
    DISPATCH_PENDING = "dispatch_pending"
    DISPATCH_APPROVED = "dispatch_approved"
    TEAM_ASSIGNED = "team_assigned"
    RESPONDING = "responding"
    ON_SCENE = "on_scene"
    RESCUE_IN_PROGRESS = "rescue_in_progress"
    RESCUED = "rescued"
    REFERRED = "referred"
    RESOLVED = "resolved"
    CLOSED = "closed"
    DUPLICATE = "duplicate"
    FALSE_ALARM = "false_alarm"
    CANCELLED = "cancelled"


class SOSPriority(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    LIFE_THREATENING = "life_threatening"


class SOSVerificationStatus(str, enum.Enum):
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"


class SOSSource(str, enum.Enum):
    MOBILE_APP = "mobile_app"
    WEB_APP = "web_app"
    PHONE_CALL = "phone_call"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMERGENCY_HOTLINE = "emergency_hotline"
    RESCUE_TEAM = "rescue_team"
    VOLUNTEER = "volunteer"
    GOVERNMENT_OFFICER = "government_officer"
    IOT_DEVICE = "iot_device"
    AUTOMATED_SENSOR = "automated_sensor"
    OTHER = "other"


class SOSLocationSource(str, enum.Enum):
    GPS = "gps"
    NETWORK = "network"
    MANUAL = "manual"
    SHARED_LOCATION = "shared_location"
    RESPONDER_VERIFIED = "responder_verified"
    UNKNOWN = "unknown"


class SOSAssignmentStatus(str, enum.Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACCEPTED = "accepted"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class SOSEvidenceType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    LOCATION = "location"
    SENSOR_DATA = "sensor_data"
    OTHER = "other"


class SOSEvidenceReviewStatus(str, enum.Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class SOSNotificationChannel(str, enum.Enum):
    SMS = "sms"
    PUSH = "push"
    EMAIL = "email"
    PHONE_CALL = "phone_call"
    WHATSAPP = "whatsapp"
    INTERNAL = "internal"


class SOSNotificationStatus(str, enum.Enum):
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    CANCELLED = "cancelled"


class SOSEscalationStatus(str, enum.Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class SOSEventType(str, enum.Enum):
    CREATED = "created"
    VERIFIED = "verified"
    TRIAGED = "triaged"
    PRIORITY_CHANGED = "priority_changed"
    LOCATION_UPDATED = "location_updated"
    EVIDENCE_ADDED = "evidence_added"
    DISPATCH_REQUESTED = "dispatch_requested"
    DISPATCH_APPROVED = "dispatch_approved"
    TEAM_ASSIGNED = "team_assigned"
    TEAM_ACCEPTED = "team_accepted"
    RESPONDER_EN_ROUTE = "responder_en_route"
    RESPONDER_ARRIVED = "responder_arrived"
    RESCUE_STARTED = "rescue_started"
    PERSON_RESCUED = "person_rescued"
    ESCALATED = "escalated"
    NOTIFICATION_SENT = "notification_sent"
    REFERRED = "referred"
    RESOLVED = "resolved"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    DUPLICATE_MARKED = "duplicate_marked"
    FALSE_ALARM_MARKED = "false_alarm_marked"
    NOTE_ADDED = "note_added"


# ============================================================
# SOS REQUEST
# ============================================================

class SOSRequest(Base):
    __tablename__ = "sos_requests"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    request_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
        index=True,
    )

    request_type: Mapped[SOSRequestType] = mapped_column(
        enum_type(SOSRequestType),
        nullable=False,
    )

    source: Mapped[SOSSource] = mapped_column(
        enum_type(SOSSource),
        nullable=False,
    )

    status: Mapped[SOSRequestStatus] = mapped_column(
        enum_type(SOSRequestStatus),
        default=SOSRequestStatus.RECEIVED,
        nullable=False,
        index=True,
    )

    priority: Mapped[SOSPriority] = mapped_column(
        enum_type(SOSPriority),
        default=SOSPriority.MODERATE,
        nullable=False,
        index=True,
    )

    verification_status: Mapped[
        SOSVerificationStatus
    ] = mapped_column(
        enum_type(SOSVerificationStatus),
        default=SOSVerificationStatus.UNVERIFIED,
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

    district_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "districts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    taluk: Mapped[str | None] = mapped_column(
        String(150),
    )

    village: Mapped[str | None] = mapped_column(
        String(150),
    )

    panchayat: Mapped[str | None] = mapped_column(
        String(150),
    )

    location_description: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
    )

    location_accuracy_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    location_source: Mapped[
        SOSLocationSource
    ] = mapped_column(
        enum_type(SOSLocationSource),
        default=SOSLocationSource.UNKNOWN,
        nullable=False,
    )

    location_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    location_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    location_verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    location_updated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    reporter_user_id: Mapped[int | None] = mapped_column(
        Integer,
    )

    encrypted_reporter_contact_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    encrypted_emergency_contact_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    affected_people_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
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

    injured_people_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    people_with_disabilities_count: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    is_immediate_danger: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_medical_response: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_boat: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_ambulance: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_accessibility_support: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_person_trapped: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_water_level_rising: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    water_depth_m: Mapped[float | None] = mapped_column(
        Float,
    )

    ai_risk_score: Mapped[float | None] = mapped_column(
        Float,
    )

    ai_priority_recommendation: Mapped[
        SOSPriority | None
    ] = mapped_column(
        enum_type(SOSPriority),
    )

    ai_model_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    ai_assessment_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    human_triaged_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    triaged_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    dispatch_approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    dispatch_approved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    assigned_rescue_team_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    rescue_mission_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    incident_id: Mapped[str | None] = mapped_column(
        String(36),
    )

    evacuation_id: Mapped[str | None] = mapped_column(
        String(36),
    )

    duplicate_of_request_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "sos_requests.id",
            ondelete="RESTRICT",
        ),
    )

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    response_started_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    responder_arrived_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    rescued_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    resolved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    closed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    closed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    resolution_notes: Mapped[
        str | None
    ] = mapped_column(
        Text,
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

    locations: Mapped[
        list["SOSLocationUpdate"]
    ] = relationship(
        "SOSLocationUpdate",
        back_populates="sos_request",
    )

    assignments: Mapped[
        list["SOSRescueAssignment"]
    ] = relationship(
        "SOSRescueAssignment",
        back_populates="sos_request",
    )

    evidence: Mapped[
        list["SOSEvidence"]
    ] = relationship(
        "SOSEvidence",
        back_populates="sos_request",
    )

    notifications: Mapped[
        list["SOSNotification"]
    ] = relationship(
        "SOSNotification",
        back_populates="sos_request",
    )

    escalations: Mapped[
        list["SOSEscalation"]
    ] = relationship(
        "SOSEscalation",
        back_populates="sos_request",
    )

    events: Mapped[
        list["SOSEvent"]
    ] = relationship(
        "SOSEvent",
        back_populates="sos_request",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_sos_district",
        ),
        CheckConstraint(
            "affected_people_count >= 0 AND "
            "children_count >= 0 AND "
            "elderly_count >= 0 AND "
            "injured_people_count >= 0 AND "
            "people_with_disabilities_count >= 0",
            name="ck_sos_people_counts",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude BETWEEN -90 AND 90)",
            name="ck_sos_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude BETWEEN -180 AND 180)",
            name="ck_sos_longitude",
        ),
        CheckConstraint(
            "location_accuracy_m IS NULL OR "
            "location_accuracy_m >= 0",
            name="ck_sos_location_accuracy",
        ),
        CheckConstraint(
            "water_depth_m IS NULL OR water_depth_m >= 0",
            name="ck_sos_water_depth",
        ),
        CheckConstraint(
            "ai_risk_score IS NULL OR "
            "(ai_risk_score >= 0 AND ai_risk_score <= 1)",
            name="ck_sos_ai_risk",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_sos_version",
        ),
        Index(
            "ix_sos_district_status_priority",
            "district_id",
            "status",
            "priority",
        ),
        Index(
            "ix_sos_received_status",
            "received_at",
            "status",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "request_number",
        "title",
        "description",
    )
    def validate_required(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "request_number": 120,
            "title": 255,
            "description": 10000,
        }
        return require_text(value, key, limits[key])

    @validates(
        "latitude",
        "longitude",
        "location_accuracy_m",
        "water_depth_m",
        "ai_risk_score",
    )
    def validate_numeric(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        bounds = {
            "latitude": (-90, 90),
            "longitude": (-180, 180),
            "location_accuracy_m": (0, None),
            "water_depth_m": (0, None),
            "ai_risk_score": (0, 1),
        }

        lower, upper = bounds[key]
        return finite_number(value, key, lower, upper)

    @property
    def is_active(self) -> bool:
        return self.status not in {
            SOSRequestStatus.RESOLVED,
            SOSRequestStatus.CLOSED,
            SOSRequestStatus.DUPLICATE,
            SOSRequestStatus.FALSE_ALARM,
            SOSRequestStatus.CANCELLED,
        }

    @property
    def requires_urgent_attention(self) -> bool:
        return (
            self.priority in {
                SOSPriority.CRITICAL,
                SOSPriority.LIFE_THREATENING,
            }
            or self.is_immediate_danger
            or self.is_person_trapped
        )

    def request_verification(self) -> None:
        if self.status != SOSRequestStatus.RECEIVED:
            raise ValueError(
                "SOS cannot enter verification state"
            )

        self.status = SOSRequestStatus.PENDING_VERIFICATION
        self.verification_status = (
            SOSVerificationStatus.PENDING
        )

    def verify(self, verifier_id: int) -> None:
        require_user_id(verifier_id)

        if self.status not in {
            SOSRequestStatus.RECEIVED,
            SOSRequestStatus.PENDING_VERIFICATION,
        }:
            raise ValueError(
                "SOS cannot be verified in current state"
            )

        self.verified_by_user_id = verifier_id
        self.verified_at = utc_now()
        self.verification_status = (
            SOSVerificationStatus.VERIFIED
        )
        self.status = SOSRequestStatus.VERIFIED

    def triage(
        self,
        *,
        priority: SOSPriority,
        triaged_by_user_id: int,
    ) -> None:
        require_user_id(triaged_by_user_id)

        if self.status != SOSRequestStatus.VERIFIED:
            raise ValueError(
                "SOS must be verified before triage"
            )

        self.priority = priority
        self.human_triaged_by_user_id = triaged_by_user_id
        self.triaged_at = utc_now()
        self.status = SOSRequestStatus.TRIAGED

    def request_dispatch(self) -> None:
        if self.status != SOSRequestStatus.TRIAGED:
            raise ValueError(
                "SOS must be triaged before dispatch request"
            )

        self.status = SOSRequestStatus.DISPATCH_PENDING

    def approve_dispatch(
        self,
        authorized_user_id: int,
    ) -> None:
        require_user_id(authorized_user_id)

        if self.status != (
            SOSRequestStatus.DISPATCH_PENDING
        ):
            raise ValueError(
                "Dispatch is not pending approval"
            )

        if (
            self.human_triaged_by_user_id is not None
            and authorized_user_id
            == self.human_triaged_by_user_id
        ):
            raise ValueError(
                "Independent dispatch approval required"
            )

        self.dispatch_approved_by_user_id = (
            authorized_user_id
        )
        self.dispatch_approved_at = utc_now()
        self.status = SOSRequestStatus.DISPATCH_APPROVED

    def assign_team(
        self,
        rescue_team_id: str,
    ) -> None:
        if self.status != (
            SOSRequestStatus.DISPATCH_APPROVED
        ):
            raise ValueError(
                "Dispatch approval required"
            )

        self.assigned_rescue_team_id = require_text(
            rescue_team_id,
            "rescue_team_id",
            36,
        )
        self.status = SOSRequestStatus.TEAM_ASSIGNED

    def mark_responding(self) -> None:
        if self.status != SOSRequestStatus.TEAM_ASSIGNED:
            raise ValueError(
                "Rescue team must be assigned"
            )

        self.response_started_at = utc_now()
        self.status = SOSRequestStatus.RESPONDING

    def mark_on_scene(self) -> None:
        if self.status != SOSRequestStatus.RESPONDING:
            raise ValueError(
                "Response must be underway"
            )

        self.responder_arrived_at = utc_now()
        self.status = SOSRequestStatus.ON_SCENE

    def mark_rescue_started(self) -> None:
        if self.status != SOSRequestStatus.ON_SCENE:
            raise ValueError(
                "Responder arrival must be confirmed"
            )

        self.status = SOSRequestStatus.RESCUE_IN_PROGRESS

    def mark_rescued(self) -> None:
        if self.status != (
            SOSRequestStatus.RESCUE_IN_PROGRESS
        ):
            raise ValueError(
                "Rescue must be in progress"
            )

        self.rescued_at = utc_now()
        self.status = SOSRequestStatus.RESCUED

    def resolve(
        self,
        *,
        authorized_user_id: int,
        notes: str,
    ) -> None:
        require_user_id(authorized_user_id)

        if self.status not in {
            SOSRequestStatus.RESCUED,
            SOSRequestStatus.REFERRED,
        }:
            raise ValueError(
                "SOS cannot be resolved in current state"
            )

        self.resolution_notes = require_text(
            notes,
            "notes",
            10000,
        )
        self.resolved_at = utc_now()
        self.status = SOSRequestStatus.RESOLVED

    def close(
        self,
        *,
        authorized_user_id: int,
    ) -> None:
        require_user_id(authorized_user_id)

        if self.status != SOSRequestStatus.RESOLVED:
            raise ValueError(
                "SOS must be resolved before closure"
            )

        self.closed_by_user_id = authorized_user_id
        self.closed_at = utc_now()
        self.status = SOSRequestStatus.CLOSED

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Minimal response for authorized operational views.
        Public endpoints should further restrict case data.
        """
        return {
            "id": self.id,
            "request_number": self.request_number,
            "request_type": self.request_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "district_id": self.district_id,
            "affected_people_count": (
                self.affected_people_count
            ),
            "requires_urgent_attention": (
                self.requires_urgent_attention
            ),
            "received_at": (
                self.received_at.isoformat()
                if self.received_at
                else None
            ),
        }


# ============================================================
# SOS LOCATION UPDATE
# ============================================================

class SOSLocationUpdate(Base):
    __tablename__ = "sos_location_updates"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    sos_request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "sos_requests.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    latitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    longitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    accuracy_m: Mapped[float | None] = mapped_column(
        Float,
    )

    location_source: Mapped[
        SOSLocationSource
    ] = mapped_column(
        enum_type(SOSLocationSource),
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
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    received_at: Mapped[datetime] = mapped_column(
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

    sos_request: Mapped["SOSRequest"] = relationship(
        "SOSRequest",
        back_populates="locations",
    )

    __table_args__ = (
        CheckConstraint(
            "latitude BETWEEN -90 AND 90",
            name="ck_sos_location_lat",
        ),
        CheckConstraint(
            "longitude BETWEEN -180 AND 180",
            name="ck_sos_location_lon",
        ),
        CheckConstraint(
            "accuracy_m IS NULL OR accuracy_m >= 0",
            name="ck_sos_location_accuracy",
        ),
        Index(
            "ix_sos_location_request_time",
            "sos_request_id",
            "observed_at",
        ),
    )


# ============================================================
# RESCUE ASSIGNMENT
# ============================================================

class SOSRescueAssignment(Base):
    __tablename__ = "sos_rescue_assignments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    sos_request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "sos_requests.id",
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

    rescue_mission_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    status: Mapped[
        SOSAssignmentStatus
    ] = mapped_column(
        enum_type(SOSAssignmentStatus),
        default=SOSAssignmentStatus.PROPOSED,
        nullable=False,
    )

    is_ai_recommended: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    recommendation_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    proposed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    accepted_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    approved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    accepted_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
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

    sos_request: Mapped["SOSRequest"] = relationship(
        "SOSRequest",
        back_populates="assignments",
    )

    __table_args__ = (
        Index(
            "ix_sos_assignment_team_status",
            "rescue_team_id",
            "status",
        ),
    )

    def request_approval(self) -> None:
        if self.status != SOSAssignmentStatus.PROPOSED:
            raise ValueError(
                "Assignment is not proposed"
            )

        self.status = SOSAssignmentStatus.PENDING_APPROVAL

    def approve(self, approver_id: int) -> None:
        require_user_id(approver_id)

        if self.status != (
            SOSAssignmentStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Assignment is not pending approval"
            )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = SOSAssignmentStatus.APPROVED

    def accept(self, responder_id: int) -> None:
        require_user_id(responder_id)

        if self.status != SOSAssignmentStatus.APPROVED:
            raise ValueError(
                "Assignment requires approval"
            )

        self.accepted_by_user_id = responder_id
        self.accepted_at = utc_now()
        self.status = SOSAssignmentStatus.ACCEPTED


# ============================================================
# SOS EVIDENCE
# ============================================================

class SOSEvidence(Base):
    __tablename__ = "sos_evidence"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    sos_request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "sos_requests.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    evidence_type: Mapped[
        SOSEvidenceType
    ] = mapped_column(
        enum_type(SOSEvidenceType),
        nullable=False,
    )

    storage_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    file_hash_sha256: Mapped[
        str | None
    ] = mapped_column(
        String(64),
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(150),
    )

    file_size_bytes: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
    )

    review_status: Mapped[
        SOSEvidenceReviewStatus
    ] = mapped_column(
        enum_type(SOSEvidenceReviewStatus),
        default=SOSEvidenceReviewStatus.PENDING,
        nullable=False,
    )

    is_ai_analyzed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    ai_analysis_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    uploaded_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    reviewed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    reviewed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    sos_request: Mapped["SOSRequest"] = relationship(
        "SOSRequest",
        back_populates="evidence",
    )

    __table_args__ = (
        CheckConstraint(
            "file_size_bytes IS NULL OR "
            "file_size_bytes >= 0",
            name="ck_sos_evidence_size",
        ),
    )


# ============================================================
# SOS NOTIFICATION
# ============================================================

class SOSNotification(Base):
    __tablename__ = "sos_notifications"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    sos_request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "sos_requests.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    channel: Mapped[
        SOSNotificationChannel
    ] = mapped_column(
        enum_type(SOSNotificationChannel),
        nullable=False,
    )

    status: Mapped[
        SOSNotificationStatus
    ] = mapped_column(
        enum_type(SOSNotificationStatus),
        default=SOSNotificationStatus.QUEUED,
        nullable=False,
    )

    recipient_encrypted_reference: Mapped[
        str
    ] = mapped_column(
        String(500),
        nullable=False,
    )

    message_template: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    provider_message_id: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    last_error_code: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    delivered_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    acknowledged_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    sos_request: Mapped["SOSRequest"] = relationship(
        "SOSRequest",
        back_populates="notifications",
    )

    __table_args__ = (
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_sos_notification_attempts",
        ),
    )


# ============================================================
# SOS ESCALATION
# ============================================================

class SOSEscalation(Base):
    __tablename__ = "sos_escalations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    sos_request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "sos_requests.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    escalation_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[
        SOSEscalationStatus
    ] = mapped_column(
        enum_type(SOSEscalationStatus),
        default=SOSEscalationStatus.OPEN,
        nullable=False,
    )

    escalated_to_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    escalated_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    acknowledged_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    acknowledged_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    resolved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    resolved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    sos_request: Mapped["SOSRequest"] = relationship(
        "SOSRequest",
        back_populates="escalations",
    )

    __table_args__ = (
        CheckConstraint(
            "escalation_level > 0",
            name="ck_sos_escalation_level",
        ),
    )


# ============================================================
# SOS AUDIT EVENTS
# ============================================================

class SOSEvent(Base):
    __tablename__ = "sos_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    sos_request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "sos_requests.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        SOSEventType
    ] = mapped_column(
        enum_type(SOSEventType),
        nullable=False,
    )

    previous_status: Mapped[
        SOSRequestStatus | None
    ] = mapped_column(
        enum_type(SOSRequestStatus),
    )

    new_status: Mapped[
        SOSRequestStatus | None
    ] = mapped_column(
        enum_type(SOSRequestStatus),
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_user_id: Mapped[int | None] = mapped_column(
        Integer,
    )

    actor_agent_name: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
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

    sos_request: Mapped["SOSRequest"] = relationship(
        "SOSRequest",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_sos_event_request_time",
            "sos_request_id",
            "occurred_at",
        ),
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

SOS_REQUEST_MODELS = (
    SOSRequest,
    SOSLocationUpdate,
    SOSRescueAssignment,
    SOSEvidence,
    SOSNotification,
    SOSEscalation,
    SOSEvent,
)


__all__ = [
    "SOSRequest",
    "SOSLocationUpdate",
    "SOSRescueAssignment",
    "SOSEvidence",
    "SOSNotification",
    "SOSEscalation",
    "SOSEvent",
    "SOSRequestType",
    "SOSRequestStatus",
    "SOSPriority",
    "SOSVerificationStatus",
    "SOSSource",
    "SOSLocationSource",
    "SOSAssignmentStatus",
    "SOSEvidenceType",
    "SOSEvidenceReviewStatus",
    "SOSNotificationChannel",
    "SOSNotificationStatus",
    "SOSEscalationStatus",
    "SOSEventType",
    "SOS_REQUEST_MODELS",
]
