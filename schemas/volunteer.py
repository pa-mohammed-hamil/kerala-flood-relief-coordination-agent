from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/volunteer.py

Pydantic v2 schemas for volunteer coordination.

Modules:
- Volunteer registration and verification
- Volunteer profiles and skills
- Availability and shift scheduling
- Volunteer organizations and teams
- Emergency task creation and assignment
- AI volunteer matching recommendations
- Human-approved volunteer deployment
- Volunteer GPS check-ins
- Attendance and working hours
- Training and certification
- Safety and incident reporting
- Volunteer performance and feedback
- Volunteer analytics and audit logging

SAFETY:
- AI recommendations do not authorize deployment.
- Volunteers must be verified before operational assignment.
- Specialist rescue/medical tasks require verified credentials.
- Volunteer availability must be explicitly confirmed.
- Location sharing must be consent-based.
- Do not publicly expose volunteer contact details.
- Physical deployment requires authorized confirmation.
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

class VolunteerBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ============================================================
# ENUMS
# ============================================================

class VolunteerStatus(str, Enum):
    REGISTERED = "registered"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    AVAILABLE = "available"
    ASSIGNED = "assigned"
    DEPLOYED = "deployed"
    RESTING = "resting"
    UNAVAILABLE = "unavailable"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"


class VolunteerSkill(str, Enum):
    GENERAL_RELIEF = "general_relief"
    FOOD_DISTRIBUTION = "food_distribution"
    WATER_DISTRIBUTION = "water_distribution"
    SHELTER_MANAGEMENT = "shelter_management"
    INVENTORY_MANAGEMENT = "inventory_management"
    LOGISTICS = "logistics"
    DRIVING = "driving"
    BOAT_OPERATION = "boat_operation"
    WATER_RESCUE = "water_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    FIRST_AID = "first_aid"
    MEDICAL_SUPPORT = "medical_support"
    NURSING = "nursing"
    COUNSELLING = "counselling"
    CHILD_SUPPORT = "child_support"
    ELDERLY_SUPPORT = "elderly_support"
    DISABILITY_SUPPORT = "disability_support"
    TRANSLATION = "translation"
    COMMUNICATION = "communication"
    DATA_ENTRY = "data_entry"
    GIS_MAPPING = "gis_mapping"
    TECHNOLOGY_SUPPORT = "technology_support"
    DAMAGE_ASSESSMENT = "damage_assessment"
    CLEANUP = "cleanup"
    OTHER = "other"


class VolunteerAvailabilityStatus(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFF_DUTY = "off_duty"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class VolunteerTaskType(str, Enum):
    FOOD_DISTRIBUTION = "food_distribution"
    WATER_DISTRIBUTION = "water_distribution"
    RELIEF_PACKING = "relief_packing"
    SHELTER_SUPPORT = "shelter_support"
    INVENTORY_SUPPORT = "inventory_support"
    TRANSPORTATION = "transportation"
    COMMUNICATION = "communication"
    DATA_COLLECTION = "data_collection"
    TRANSLATION = "translation"
    CLEANUP = "cleanup"
    FAMILY_REUNIFICATION = "family_reunification"
    MEDICAL_SUPPORT = "medical_support"
    SEARCH_AND_RESCUE = "search_and_rescue"
    WATER_RESCUE = "water_rescue"
    DAMAGE_ASSESSMENT = "damage_assessment"
    OTHER = "other"


class VolunteerTaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VolunteerTaskStatus(str, Enum):
    OPEN = "open"
    PENDING_ASSIGNMENT = "pending_assignment"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class AssignmentStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    VOLUNTEER_ACCEPTED = "volunteer_accepted"
    DEPLOYMENT_CONFIRMED = "deployment_confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VolunteerShiftStatus(str, Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    COMPLETED = "completed"
    MISSED = "missed"
    CANCELLED = "cancelled"


class TrainingStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


class VolunteerIncidentType(str, Enum):
    INJURY = "injury"
    MEDICAL_EMERGENCY = "medical_emergency"
    UNSAFE_CONDITIONS = "unsafe_conditions"
    EQUIPMENT_FAILURE = "equipment_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    MISSING_VOLUNTEER = "missing_volunteer"
    HARASSMENT = "harassment"
    OTHER = "other"


class VolunteerIncidentStatus(str, Enum):
    REPORTED = "reported"
    UNDER_REVIEW = "under_review"
    ACTION_REQUIRED = "action_required"
    RESOLVED = "resolved"
    CLOSED = "closed"


# ============================================================
# LOCATION
# ============================================================

class VolunteerLocation(VolunteerBase):
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

    address: str | None = None
    landmark: str | None = None
    panchayat: str | None = None

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
                "Latitude and longitude must be provided together"
            )

        return self


# ============================================================
# VOLUNTEER REGISTRATION
# ============================================================

class VolunteerRegistration(VolunteerBase):
    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    age: int = Field(
        ge=18,
        le=100,
    )

    email: str | None = None

    phone_number: str = Field(
        min_length=7,
        max_length=20,
    )

    home_district: str

    current_location: VolunteerLocation | None = None

    skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    languages: list[str] = Field(
        default_factory=lambda: ["ml"]
    )

    organization_id: UUID | None = None

    emergency_contact_name: str | None = None

    emergency_contact_phone: str | None = None

    has_transport: bool = False

    willing_to_travel: bool = False

    maximum_travel_distance_km: float | None = Field(
        default=None,
        ge=0,
    )

    consent_to_contact: bool

    consent_to_location_sharing: bool = False

    @field_validator("home_district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @model_validator(mode="after")
    def validate_registration(self):
        if not self.consent_to_contact:
            raise ValueError(
                "Contact consent is required for registration"
            )

        return self


# ============================================================
# VOLUNTEER PROFILE
# ============================================================

class VolunteerProfile(VolunteerRegistration):
    volunteer_id: UUID = Field(
        default_factory=uuid4
    )

    status: VolunteerStatus = (
        VolunteerStatus.PENDING_VERIFICATION
    )

    verification_status: VerificationStatus = (
        VerificationStatus.PENDING
    )

    verified_skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    availability_status: VolunteerAvailabilityStatus = (
        VolunteerAvailabilityStatus.UNKNOWN
    )

    completed_tasks: int = Field(
        default=0,
        ge=0,
    )

    total_service_hours: float = Field(
        default=0,
        ge=0,
    )

    registered_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


class VolunteerProfileUpdate(VolunteerBase):
    email: str | None = None

    phone_number: str | None = None

    current_location: VolunteerLocation | None = None

    skills: list[VolunteerSkill] | None = None

    languages: list[str] | None = None

    has_transport: bool | None = None

    willing_to_travel: bool | None = None

    maximum_travel_distance_km: float | None = Field(
        default=None,
        ge=0,
    )

    consent_to_location_sharing: bool | None = None


# ============================================================
# VOLUNTEER VERIFICATION
# ============================================================

class VolunteerVerificationRequest(VolunteerBase):
    volunteer_id: UUID

    verification_status: VerificationStatus

    verified_skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    reason: str | None = None


class VolunteerVerificationResponse(VolunteerBase):
    volunteer_id: UUID

    verification_status: VerificationStatus

    verified_skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    verified_by: str

    verified_at: datetime = Field(
        default_factory=utc_now
    )

    authorization_verified: bool = False


# ============================================================
# CERTIFICATIONS
# ============================================================

class VolunteerCertification(VolunteerBase):
    certification_id: UUID = Field(
        default_factory=uuid4
    )

    volunteer_id: UUID

    certification_name: str = Field(
        min_length=2,
        max_length=200,
    )

    issuing_organization: str

    certificate_reference: str | None = None

    issued_at: datetime

    expires_at: datetime | None = None

    verified: bool = False

    verified_by: str | None = None

    @model_validator(mode="after")
    def validate_dates(self):
        validate_datetime(self.issued_at)

        if self.expires_at is not None:
            validate_datetime(self.expires_at)

            if self.expires_at <= self.issued_at:
                raise ValueError(
                    "Certification expiry must follow issue date"
                )

        return self


# ============================================================
# TRAINING
# ============================================================

class VolunteerTraining(VolunteerBase):
    training_id: UUID = Field(
        default_factory=uuid4
    )

    volunteer_id: UUID

    training_name: str

    provider: str

    status: TrainingStatus = (
        TrainingStatus.NOT_STARTED
    )

    completed_at: datetime | None = None

    expires_at: datetime | None = None

    verified: bool = False


# ============================================================
# VOLUNTEER AVAILABILITY
# ============================================================

class VolunteerAvailability(VolunteerBase):
    volunteer_id: UUID

    status: VolunteerAvailabilityStatus

    available_from: datetime | None = None

    available_until: datetime | None = None

    preferred_districts: list[str] = Field(
        default_factory=list
    )

    confirmed_by_volunteer: bool = False

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("preferred_districts")
    @classmethod
    def validate_districts(
        cls,
        values: list[str],
    ) -> list[str]:
        return [
            normalize_district(value)
            for value in values
        ]

    @model_validator(mode="after")
    def validate_availability(self):
        if self.available_from is not None:
            validate_datetime(self.available_from)

        if self.available_until is not None:
            validate_datetime(self.available_until)

        if (
            self.available_from is not None
            and self.available_until is not None
            and self.available_until <= self.available_from
        ):
            raise ValueError(
                "Availability end must follow start"
            )

        return self


# ============================================================
# VOLUNTEER ORGANIZATION
# ============================================================

class VolunteerOrganizationCreate(VolunteerBase):
    name: str = Field(
        min_length=2,
        max_length=200,
    )

    organization_type: str

    district: str

    contact_name: str | None = None

    contact_phone: str | None = None

    email: str | None = None

    description: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class VolunteerOrganizationResponse(
    VolunteerOrganizationCreate
):
    organization_id: UUID = Field(
        default_factory=uuid4
    )

    verified: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# VOLUNTEER TEAM
# ============================================================

class VolunteerTeamCreate(VolunteerBase):
    team_name: str = Field(
        min_length=2,
        max_length=200,
    )

    organization_id: UUID | None = None

    district: str

    leader_id: UUID | None = None

    member_ids: list[UUID] = Field(
        default_factory=list
    )

    skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class VolunteerTeamResponse(VolunteerTeamCreate):
    team_id: UUID = Field(
        default_factory=uuid4
    )

    verified: bool = False

    active: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# VOLUNTEER TASK
# ============================================================

class VolunteerTaskCreate(VolunteerBase):
    title: str = Field(
        min_length=5,
        max_length=200,
    )

    description: str = Field(
        min_length=10,
        max_length=5000,
    )

    task_type: VolunteerTaskType

    priority: VolunteerTaskPriority = (
        VolunteerTaskPriority.MEDIUM
    )

    location: VolunteerLocation

    required_skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    required_volunteers: int = Field(
        ge=1,
    )

    planned_start: datetime | None = None

    planned_end: datetime | None = None

    shelter_id: UUID | None = None

    incident_id: UUID | None = None

    safety_instructions: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_schedule(self):
        if self.planned_start is not None:
            validate_datetime(self.planned_start)

        if self.planned_end is not None:
            validate_datetime(self.planned_end)

        if (
            self.planned_start is not None
            and self.planned_end is not None
            and self.planned_end <= self.planned_start
        ):
            raise ValueError(
                "Task end must follow task start"
            )

        return self


class VolunteerTaskResponse(VolunteerTaskCreate):
    task_id: UUID = Field(
        default_factory=uuid4
    )

    status: VolunteerTaskStatus = (
        VolunteerTaskStatus.OPEN
    )

    assigned_volunteers: list[UUID] = Field(
        default_factory=list
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# AI VOLUNTEER MATCHING
# ============================================================

class VolunteerMatch(VolunteerBase):
    volunteer_id: UUID

    task_id: UUID

    match_score: float = Field(
        ge=0,
        le=100,
    )

    matched_skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    missing_skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    distance_km: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_travel_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    identity_verified: bool = False

    required_credentials_verified: bool = False

    availability_confirmed: bool = False

    route_safety_verified: bool = False

    reasons: list[str] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )


class VolunteerAssignmentRecommendation(VolunteerBase):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    task_id: UUID

    recommended_volunteers: list[
        VolunteerMatch
    ] = Field(default_factory=list)

    reasoning_summary: str

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    human_approval_required: bool = True

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# VOLUNTEER ASSIGNMENT
# ============================================================

class VolunteerAssignmentCreate(VolunteerBase):
    task_id: UUID

    volunteer_id: UUID

    recommendation_id: UUID | None = None

    assignment_notes: str | None = None


class VolunteerAssignmentResponse(
    VolunteerAssignmentCreate
):
    assignment_id: UUID = Field(
        default_factory=uuid4
    )

    status: AssignmentStatus = (
        AssignmentStatus.PENDING_APPROVAL
    )

    approved_by: str | None = None

    approved_at: datetime | None = None

    accepted_by_volunteer_at: datetime | None = None

    deployment_confirmed_by: str | None = None

    deployment_confirmed_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.status in {
            AssignmentStatus.DEPLOYMENT_CONFIRMED,
            AssignmentStatus.IN_PROGRESS,
            AssignmentStatus.COMPLETED,
        }:
            if not (
                self.approved_by
                and self.approved_at
                and self.accepted_by_volunteer_at
                and self.deployment_confirmed_by
                and self.deployment_confirmed_at
            ):
                raise ValueError(
                    "Deployment requires authorized approval, "
                    "volunteer acceptance and confirmation"
                )

        return self


# ============================================================
# VOLUNTEER SHIFT
# ============================================================

class VolunteerShiftCreate(VolunteerBase):
    volunteer_id: UUID

    task_id: UUID | None = None

    shelter_id: UUID | None = None

    start_time: datetime

    end_time: datetime

    @model_validator(mode="after")
    def validate_shift(self):
        validate_datetime(self.start_time)
        validate_datetime(self.end_time)

        if self.end_time <= self.start_time:
            raise ValueError(
                "Shift end must follow start"
            )

        return self


class VolunteerShiftResponse(VolunteerShiftCreate):
    shift_id: UUID = Field(
        default_factory=uuid4
    )

    status: VolunteerShiftStatus = (
        VolunteerShiftStatus.SCHEDULED
    )

    checked_in_at: datetime | None = None

    checked_out_at: datetime | None = None

    confirmed_service_hours: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# ATTENDANCE
# ============================================================

class VolunteerCheckIn(VolunteerBase):
    volunteer_id: UUID

    shift_id: UUID

    location: VolunteerLocation | None = None

    confirmed_by: str | None = None

    checked_in_at: datetime = Field(
        default_factory=utc_now
    )


class VolunteerCheckOut(VolunteerBase):
    volunteer_id: UUID

    shift_id: UUID

    notes: str | None = None

    confirmed_by: str | None = None

    checked_out_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# CONSENT-BASED GPS UPDATE
# ============================================================

class VolunteerLocationUpdate(VolunteerBase):
    volunteer_id: UUID

    location: VolunteerLocation

    accuracy_m: float | None = Field(
        default=None,
        ge=0,
    )

    consent_confirmed: bool

    recorded_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_consent(self):
        if not self.consent_confirmed:
            raise ValueError(
                "Location sharing requires consent"
            )

        if (
            self.location.latitude is None
            or self.location.longitude is None
        ):
            raise ValueError(
                "GPS update requires coordinates"
            )

        return self


# ============================================================
# VOLUNTEER SAFETY INCIDENT
# ============================================================

class VolunteerSafetyIncidentCreate(VolunteerBase):
    volunteer_id: UUID

    assignment_id: UUID | None = None

    incident_type: VolunteerIncidentType

    location: VolunteerLocation | None = None

    description: str = Field(
        min_length=5,
        max_length=5000,
    )

    immediate_assistance_required: bool = False

    reported_at: datetime = Field(
        default_factory=utc_now
    )


class VolunteerSafetyIncidentResponse(
    VolunteerSafetyIncidentCreate
):
    incident_id: UUID = Field(
        default_factory=uuid4
    )

    status: VolunteerIncidentStatus = (
        VolunteerIncidentStatus.REPORTED
    )

    assigned_reviewer: str | None = None

    resolved_at: datetime | None = None


# ============================================================
# TASK COMPLETION
# ============================================================

class VolunteerTaskCompletion(VolunteerBase):
    task_id: UUID

    volunteer_id: UUID

    completion_summary: str = Field(
        min_length=5,
        max_length=3000,
    )

    evidence_references: list[str] = Field(
        default_factory=list
    )

    completed_at: datetime = Field(
        default_factory=utc_now
    )

    verified_by: str | None = None

    completion_verified: bool = False


# ============================================================
# VOLUNTEER FEEDBACK
# ============================================================

class VolunteerFeedback(VolunteerBase):
    feedback_id: UUID = Field(
        default_factory=uuid4
    )

    volunteer_id: UUID

    task_id: UUID | None = None

    rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    feedback: str | None = Field(
        default=None,
        max_length=3000,
    )

    submitted_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# HUMAN APPROVAL
# ============================================================

class VolunteerApprovalRequest(VolunteerBase):
    assignment_id: UUID

    approve: bool

    reason: str = Field(
        min_length=5,
        max_length=2000,
    )


class VolunteerApprovalResponse(VolunteerBase):
    assignment_id: UUID

    approved: bool

    reviewed_by: str

    reviewed_at: datetime = Field(
        default_factory=utc_now
    )

    authorization_verified: bool = False


# ============================================================
# VOLUNTEER AUDIT
# ============================================================

class VolunteerAuditEvent(VolunteerBase):
    event_id: UUID = Field(
        default_factory=uuid4
    )

    entity_id: UUID

    entity_type: str

    action: str

    performed_by: str

    # Store non-sensitive metadata only.
    details: dict[str, Any] = Field(
        default_factory=dict
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# PUBLIC VOLUNTEER SUMMARY
# ============================================================

class PublicVolunteerSummary(VolunteerBase):
    volunteer_id: UUID

    district: str

    skills: list[VolunteerSkill] = Field(
        default_factory=list
    )

    verification_status: VerificationStatus

    availability_status: VolunteerAvailabilityStatus

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# VOLUNTEER DASHBOARD
# ============================================================

class VolunteerDashboard(VolunteerBase):
    district: str | None = None

    registered_volunteers: int | None = Field(
        default=None,
        ge=0,
    )

    verified_volunteers: int | None = Field(
        default=None,
        ge=0,
    )

    available_volunteers: int | None = Field(
        default=None,
        ge=0,
    )

    deployed_volunteers: int | None = Field(
        default=None,
        ge=0,
    )

    active_tasks: int | None = Field(
        default=None,
        ge=0,
    )

    completed_tasks: int | None = Field(
        default=None,
        ge=0,
    )

    pending_assignments: int | None = Field(
        default=None,
        ge=0,
    )

    active_shifts: int | None = Field(
        default=None,
        ge=0,
    )

    verified_service_hours: float | None = Field(
        default=None,
        ge=0,
    )

    unresolved_safety_incidents: int | None = Field(
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
# GENERIC API RESPONSE
# ============================================================

class VolunteerAPIResponse(VolunteerBase):
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
    "VolunteerStatus",
    "VerificationStatus",
    "VolunteerSkill",
    "VolunteerAvailabilityStatus",
    "VolunteerTaskType",
    "VolunteerTaskPriority",
    "VolunteerTaskStatus",
    "AssignmentStatus",
    "VolunteerShiftStatus",
    "TrainingStatus",
    "VolunteerIncidentType",
    "VolunteerIncidentStatus",
    "VolunteerLocation",
    "VolunteerRegistration",
    "VolunteerProfile",
    "VolunteerProfileUpdate",
    "VolunteerVerificationRequest",
    "VolunteerVerificationResponse",
    "VolunteerCertification",
    "VolunteerTraining",
    "VolunteerAvailability",
    "VolunteerOrganizationCreate",
    "VolunteerOrganizationResponse",
    "VolunteerTeamCreate",
    "VolunteerTeamResponse",
    "VolunteerTaskCreate",
    "VolunteerTaskResponse",
    "VolunteerMatch",
    "VolunteerAssignmentRecommendation",
    "VolunteerAssignmentCreate",
    "VolunteerAssignmentResponse",
    "VolunteerShiftCreate",
    "VolunteerShiftResponse",
    "VolunteerCheckIn",
    "VolunteerCheckOut",
    "VolunteerLocationUpdate",
    "VolunteerSafetyIncidentCreate",
    "VolunteerSafetyIncidentResponse",
    "VolunteerTaskCompletion",
    "VolunteerFeedback",
    "VolunteerApprovalRequest",
    "VolunteerApprovalResponse",
    "VolunteerAuditEvent",
    "PublicVolunteerSummary",
    "VolunteerDashboard",
    "VolunteerAPIResponse",
    "normalize_district",
    "validate_datetime",
]
