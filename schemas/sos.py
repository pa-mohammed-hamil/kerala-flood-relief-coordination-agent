from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/sos.py

Pydantic v2 schemas for emergency SOS management.

Features:
- Emergency SOS registration
- GPS and manual location reporting
- Flood rescue and medical emergencies
- Emergency contact information
- Vulnerable-person assistance
- SOS triage and priority recommendations
- Duplicate detection
- Rescue coordination
- SOS acknowledgment and status tracking
- Escalation and cancellation
- Emergency communication
- Human approval
- SOS dashboard and analytics
- Audit logging

SAFETY:
- All valid SOS requests must enter the emergency queue.
- AI must never automatically reject emergency requests.
- Priority scores are advisory, not medical assessments.
- Acknowledgment does not mean rescue was dispatched.
- Rescue dispatch requires authorized confirmation.
- Never expose private SOS information publicly.
- False-positive cancellation must be authenticated.
"""


from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


KERALA_DISTRICTS = {
    "thiruvananthapuram",
    "kollam",
    "pathanamthitta",
    "alappuzha",
    "kottayam",
    "idukki",
    "ernakulam",
    "thrissur",
    "palakkad",
    "malappuram",
    "kozhikode",
    "wayanad",
    "kannur",
    "kasaragod",
}


def normalize_district(value: str) -> str:
    district = value.strip().lower()

    if district not in KERALA_DISTRICTS:
        raise ValueError(
            f"Invalid Kerala district: {value}"
        )

    return district


def validate_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Datetime must include timezone information"
        )

    return value


# ============================================================
# BASE SCHEMA
# ============================================================

class SOSBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ============================================================
# ENUMS
# ============================================================

class SOSType(str, Enum):
    FLOOD_RESCUE = "flood_rescue"
    MEDICAL_EMERGENCY = "medical_emergency"
    EVACUATION = "evacuation"
    TRAPPED_PERSON = "trapped_person"
    MISSING_PERSON = "missing_person"
    FOOD_WATER = "food_water"
    SHELTER = "shelter"
    OTHER = "other"


class SOSPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNASSESSED = "unassessed"


class SOSStatus(str, Enum):
    RECEIVED = "received"
    TRIAGED = "triaged"
    AWAITING_REVIEW = "awaiting_review"
    ACKNOWLEDGED = "acknowledged"
    ASSIGNED = "assigned"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class SOSSource(str, Enum):
    WEB = "web"
    MOBILE = "mobile"
    PHONE = "phone"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    CONTROL_ROOM = "control_room"
    FIELD_RESPONDER = "field_responder"
    OTHER = "other"


class LocationSource(str, Enum):
    GPS = "gps"
    MANUAL = "manual"
    NETWORK = "network"
    RESPONDER = "responder"
    UNKNOWN = "unknown"


class TriageStatus(str, Enum):
    PENDING = "pending"
    AI_RECOMMENDED = "ai_recommended"
    HUMAN_REVIEWED = "human_reviewed"
    ESCALATED = "escalated"


class SOSNotificationType(str, Enum):
    REQUEST_RECEIVED = "request_received"
    REQUEST_ACKNOWLEDGED = "request_acknowledged"
    PRIORITY_UPDATED = "priority_updated"
    RESCUE_ASSIGNED = "rescue_assigned"
    DISPATCH_CONFIRMED = "dispatch_confirmed"
    RESCUE_IN_PROGRESS = "rescue_in_progress"
    REQUEST_RESOLVED = "request_resolved"
    REQUEST_CANCELLED = "request_cancelled"
    ESCALATION = "escalation"


class SOSNotificationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    UNKNOWN = "unknown"


class SOSEscalationReason(str, Enum):
    NO_RESPONSE = "no_response"
    DETERIORATING_CONDITIONS = "deteriorating_conditions"
    MEDICAL_EMERGENCY = "medical_emergency"
    INCREASED_WATER_LEVEL = "increased_water_level"
    RESCUE_DELAY = "rescue_delay"
    NEW_DANGER = "new_danger"
    MANUAL_ESCALATION = "manual_escalation"
    OTHER = "other"


class SOSCancellationReason(str, Enum):
    FALSE_ALARM = "false_alarm"
    REQUESTER_SAFE = "requester_safe"
    DUPLICATE = "duplicate"
    RESOLVED_EXTERNALLY = "resolved_externally"
    OTHER = "other"


class SOSVerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    INCONCLUSIVE = "inconclusive"


# ============================================================
# LOCATION
# ============================================================

class SOSLocation(SOSBase):
    district: str

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
    )

    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
    )

    address: str | None = Field(
        default=None,
        max_length=1000,
    )

    landmark: str | None = None

    panchayat: str | None = None
    municipality: str | None = None
    ward: str | None = None

    location_source: LocationSource = (
        LocationSource.UNKNOWN
    )

    accuracy_m: float | None = Field(
        default=None,
        ge=0,
    )

    recorded_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @model_validator(mode="after")
    def validate_coordinates(self):
        if (
            self.latitude is None
        ) != (
            self.longitude is None
        ):
            raise ValueError(
                "Latitude and longitude must be supplied together"
            )

        if (
            self.latitude is None
            and not self.address
            and not self.landmark
        ):
            raise ValueError(
                "Provide coordinates, an address, or a landmark"
            )

        return self


# ============================================================
# EMERGENCY CONTACT
# ============================================================

class SOSEmergencyContact(SOSBase):
    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    relationship: str | None = None

    phone_number: str = Field(
        min_length=7,
        max_length=20,
    )

    authorized_to_contact: bool = False


# ============================================================
# PEOPLE REQUIRING ASSISTANCE
# ============================================================

class SOSPeopleInformation(SOSBase):
    total_people: int = Field(
        default=1,
        ge=1,
    )

    children_count: int = Field(
        default=0,
        ge=0,
    )

    elderly_count: int = Field(
        default=0,
        ge=0,
    )

    injured_count: int = Field(
        default=0,
        ge=0,
    )

    people_with_disabilities: int = Field(
        default=0,
        ge=0,
    )

    pregnant_people_count: int = Field(
        default=0,
        ge=0,
    )

    missing_people_count: int = Field(
        default=0,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_counts(self):
        counts = [
            self.children_count,
            self.elderly_count,
            self.injured_count,
            self.people_with_disabilities,
            self.pregnant_people_count,
        ]

        if any(
            count > self.total_people
            for count in counts
        ):
            raise ValueError(
                "Individual subgroup counts cannot exceed total_people"
            )

        return self


# ============================================================
# EMERGENCY CONDITIONS
# ============================================================

class SOSEmergencyConditions(SOSBase):
    trapped: bool = False

    immediate_danger: bool = False

    medical_emergency: bool = False

    rising_water: bool = False

    building_damage: bool = False

    road_access_blocked: bool = False

    communication_difficulties: bool = False

    drinking_water_unavailable: bool = False

    reported_water_depth_m: float | None = Field(
        default=None,
        ge=0,
    )

    additional_hazards: list[str] = Field(
        default_factory=list
    )


# ============================================================
# CREATE SOS REQUEST
# ============================================================

class SOSCreate(SOSBase):
    sos_type: SOSType

    location: SOSLocation

    description: str = Field(
        min_length=5,
        max_length=5000,
    )

    requester_name: str | None = Field(
        default=None,
        max_length=150,
    )

    requester_phone: str | None = Field(
        default=None,
        max_length=20,
    )

    people: SOSPeopleInformation = Field(
        default_factory=SOSPeopleInformation
    )

    conditions: SOSEmergencyConditions = Field(
        default_factory=SOSEmergencyConditions
    )

    emergency_contacts: list[
        SOSEmergencyContact
    ] = Field(default_factory=list)

    source: SOSSource = SOSSource.WEB

    preferred_language: str = "ml"

    additional_notes: str | None = Field(
        default=None,
        max_length=3000,
    )

    @field_validator("preferred_language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in {"en", "ml"}:
            raise ValueError(
                "Supported languages: en, ml"
            )
        return value


# ============================================================
# SOS RESPONSE
# ============================================================

class SOSResponse(SOSCreate):
    sos_id: UUID = Field(
        default_factory=uuid4
    )

    tracking_reference: str

    status: SOSStatus = SOSStatus.RECEIVED

    priority: SOSPriority = (
        SOSPriority.UNASSESSED
    )

    triage_status: TriageStatus = (
        TriageStatus.PENDING
    )

    verification_status: SOSVerificationStatus = (
        SOSVerificationStatus.UNVERIFIED
    )

    assigned_team_id: UUID | None = None

    rescue_mission_id: UUID | None = None

    acknowledged_at: datetime | None = None

    dispatched_at: datetime | None = None

    resolved_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# PUBLIC SOS SUBMISSION RESPONSE
# ============================================================

class SOSSubmissionResponse(SOSBase):
    sos_id: UUID

    tracking_reference: str

    status: SOSStatus = SOSStatus.RECEIVED

    received_at: datetime = Field(
        default_factory=utc_now
    )

    message: str = (
        "SOS request received. This does not confirm "
        "that a rescue team has been dispatched."
    )


# ============================================================
# AI SOS TRIAGE RECOMMENDATION
# ============================================================

class AIOSTriageRecommendation(SOSBase):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    sos_id: UUID

    suggested_priority: SOSPriority

    priority_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    reasons: list[str] = Field(
        default_factory=list
    )

    warning_signs: list[str] = Field(
        default_factory=list
    )

    recommended_actions: list[str] = Field(
        default_factory=list
    )

    model_name: str | None = None

    model_version: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    human_review_required: bool = True

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# HUMAN TRIAGE ASSESSMENT
# ============================================================

class SOSTriageAssessment(SOSBase):
    assessment_id: UUID = Field(
        default_factory=uuid4
    )

    sos_id: UUID

    priority: SOSPriority

    assessment_notes: str = Field(
        min_length=1,
        max_length=3000,
    )

    assessed_by: str

    assessed_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(
        cls,
        value: SOSPriority,
    ) -> SOSPriority:
        if value == SOSPriority.UNASSESSED:
            raise ValueError(
                "Human triage requires an assessed priority"
            )

        return value


# ============================================================
# SOS DUPLICATE DETECTION
# ============================================================

class SOSDuplicateCandidate(SOSBase):
    sos_id: UUID

    possible_duplicate_id: UUID

    similarity_score: float = Field(
        ge=0,
        le=1,
    )

    matching_factors: list[str] = Field(
        default_factory=list
    )

    reviewed: bool = False

    generated_at: datetime = Field(
        default_factory=utc_now
    )


class SOSDuplicateReview(SOSBase):
    sos_id: UUID

    possible_duplicate_id: UUID

    confirmed_duplicate: bool

    reviewed_by: str

    reason: str = Field(
        min_length=5,
        max_length=2000,
    )


# ============================================================
# SOS ACKNOWLEDGMENT
# ============================================================

class SOSAcknowledgment(SOSBase):
    sos_id: UUID

    acknowledged_by: str

    acknowledgment_message: str

    acknowledged_at: datetime = Field(
        default_factory=utc_now
    )

    rescue_dispatched: bool = False


# ============================================================
# SOS STATUS UPDATE
# ============================================================

class SOSStatusUpdate(SOSBase):
    sos_id: UUID

    status: SOSStatus

    notes: str | None = Field(
        default=None,
        max_length=3000,
    )

    updated_by: str


# ============================================================
# SOS ESCALATION
# ============================================================

class SOSEscalationRequest(SOSBase):
    sos_id: UUID

    reason: SOSEscalationReason

    description: str = Field(
        min_length=5,
        max_length=3000,
    )

    updated_conditions: SOSEmergencyConditions | None = None


class SOSEscalationResponse(SOSBase):
    escalation_id: UUID = Field(
        default_factory=uuid4
    )

    sos_id: UUID

    reason: SOSEscalationReason

    escalated_at: datetime = Field(
        default_factory=utc_now
    )

    human_review_required: bool = True

    acknowledged_by: str | None = None


# ============================================================
# SOS CANCELLATION
# ============================================================

class SOSCancellationRequest(SOSBase):
    sos_id: UUID

    reason: SOSCancellationReason

    description: str | None = None


class SOSCancellationResponse(SOSBase):
    sos_id: UUID

    status: SOSStatus

    cancelled_by: str

    cancelled_at: datetime = Field(
        default_factory=utc_now
    )

    cancellation_verified: bool = False


# ============================================================
# SOS RESOLUTION
# ============================================================

class SOSResolutionReport(SOSBase):
    sos_id: UUID

    resolution_summary: str = Field(
        min_length=5,
        max_length=5000,
    )

    people_assisted: int = Field(
        ge=0,
    )

    rescue_mission_id: UUID | None = None

    assistance_confirmed_by: str

    evidence_references: list[str] = Field(
        default_factory=list
    )

    resolved_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# SOS LOCATION UPDATE
# ============================================================

class SOSLocationUpdate(SOSBase):
    sos_id: UUID

    location: SOSLocation

    updated_by: str | None = None

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# SOS COMMUNICATION
# ============================================================

class SOSNotification(SOSBase):
    notification_id: UUID = Field(
        default_factory=uuid4
    )

    sos_id: UUID

    notification_type: SOSNotificationType

    channel: str

    recipient_reference: str

    message: str

    status: SOSNotificationStatus = (
        SOSNotificationStatus.PENDING
    )

    provider_message_id: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    sent_at: datetime | None = None

    delivered_at: datetime | None = None


# ============================================================
# SOS TRACKING
# ============================================================

class SOSTrackingEvent(SOSBase):
    event_id: UUID = Field(
        default_factory=uuid4
    )

    sos_id: UUID

    status: SOSStatus

    description: str

    timestamp: datetime = Field(
        default_factory=utc_now
    )


class SOSTrackingResponse(SOSBase):
    sos_id: UUID

    tracking_reference: str

    current_status: SOSStatus

    priority: SOSPriority

    events: list[SOSTrackingEvent] = Field(
        default_factory=list
    )

    last_updated: datetime = Field(
        default_factory=utc_now
    )

    rescue_dispatch_confirmed: bool = False


# ============================================================
# SOS DASHBOARD
# ============================================================

class SOSDashboard(SOSBase):
    district: str | None = None

    total_requests: int | None = Field(
        default=None,
        ge=0,
    )

    received_requests: int | None = Field(
        default=None,
        ge=0,
    )

    awaiting_review: int | None = Field(
        default=None,
        ge=0,
    )

    critical_requests: int | None = Field(
        default=None,
        ge=0,
    )

    high_priority_requests: int | None = Field(
        default=None,
        ge=0,
    )

    assigned_requests: int | None = Field(
        default=None,
        ge=0,
    )

    dispatched_requests: int | None = Field(
        default=None,
        ge=0,
    )

    in_progress_requests: int | None = Field(
        default=None,
        ge=0,
    )

    resolved_requests: int | None = Field(
        default=None,
        ge=0,
    )

    average_acknowledgment_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    average_dispatch_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    data_verified: bool = False

    @field_validator("district")
    @classmethod
    def validate_district(
        cls,
        value: str | None,
    ) -> str | None:
        return (
            normalize_district(value)
            if value is not None
            else None
        )


# ============================================================
# SOS AUDIT
# ============================================================

class SOSAuditEvent(SOSBase):
    event_id: UUID = Field(
        default_factory=uuid4
    )

    sos_id: UUID

    action: str

    performed_by: str

    # Avoid storing names, phone numbers,
    # precise coordinates or medical information.
    details: dict[str, Any] = Field(
        default_factory=dict
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# GENERIC API RESPONSE
# ============================================================

class SOSAPIResponse(SOSBase):
    success: bool

    message: str

    data: dict[str, Any] | None = None

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "SOSType",
    "SOSPriority",
    "SOSStatus",
    "SOSSource",
    "LocationSource",
    "TriageStatus",
    "SOSNotificationType",
    "SOSNotificationStatus",
    "SOSEscalationReason",
    "SOSCancellationReason",
    "SOSVerificationStatus",
    "SOSLocation",
    "SOSEmergencyContact",
    "SOSPeopleInformation",
    "SOSEmergencyConditions",
    "SOSCreate",
    "SOSResponse",
    "SOSSubmissionResponse",
    "AIOSTriageRecommendation",
    "SOSTriageAssessment",
    "SOSDuplicateCandidate",
    "SOSDuplicateReview",
    "SOSAcknowledgment",
    "SOSStatusUpdate",
    "SOSEscalationRequest",
    "SOSEscalationResponse",
    "SOSCancellationRequest",
    "SOSCancellationResponse",
    "SOSResolutionReport",
    "SOSLocationUpdate",
    "SOSNotification",
    "SOSTrackingEvent",
    "SOSTrackingResponse",
    "SOSDashboard",
    "SOSAuditEvent",
    "SOSAPIResponse",
    "normalize_district",
    "validate_datetime",
]
