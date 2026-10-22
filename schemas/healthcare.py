from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/healthcare.py

Pydantic v2 schemas for disaster healthcare coordination.

Modules:
- Medical emergency requests
- Patient triage
- Ambulance availability and dispatch
- Hospital capacity monitoring
- Emergency medical teams
- Medical supply inventory
- Patient transport and referrals
- Medical assistance tracking
- Healthcare analytics
- Human approval and audit records

SAFETY:
- Triage levels are assigned by qualified personnel.
- AI-generated suggestions are advisory only.
- Ambulance dispatch requires authorized confirmation.
- Do not expose patient data through public APIs.
- Missing capacity information is not zero capacity.
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

class HealthcareBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ============================================================
# ENUMS
# ============================================================

class MedicalEmergencyType(str, Enum):
    INJURY = "injury"
    DROWNING = "drowning"
    BREATHING_DIFFICULTY = "breathing_difficulty"
    CARDIAC_EMERGENCY = "cardiac_emergency"
    UNCONSCIOUS = "unconscious"
    PREGNANCY_COMPLICATION = "pregnancy_complication"
    INFECTION = "infection"
    CHRONIC_CONDITION = "chronic_condition"
    MEDICATION_REQUIRED = "medication_required"
    MOBILITY_ASSISTANCE = "mobility_assistance"
    OTHER = "other"


class TriageLevel(str, Enum):
    IMMEDIATE = "immediate"
    URGENT = "urgent"
    DELAYED = "delayed"
    MINIMAL = "minimal"
    EXPECTANT = "expectant"
    UNASSESSED = "unassessed"


class MedicalRequestStatus(str, Enum):
    RECEIVED = "received"
    AWAITING_TRIAGE = "awaiting_triage"
    TRIAGED = "triaged"
    AWAITING_ASSIGNMENT = "awaiting_assignment"
    ASSIGNED = "assigned"
    RESPONSE_CONFIRMED = "response_confirmed"
    IN_PROGRESS = "in_progress"
    TRANSPORTING = "transporting"
    AT_HOSPITAL = "at_hospital"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AmbulanceStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    ON_SCENE = "on_scene"
    TRANSPORTING = "transporting"
    UNAVAILABLE = "unavailable"
    OFFLINE = "offline"


class AmbulanceType(str, Enum):
    BASIC_LIFE_SUPPORT = "basic_life_support"
    ADVANCED_LIFE_SUPPORT = "advanced_life_support"
    PATIENT_TRANSPORT = "patient_transport"
    BOAT_AMBULANCE = "boat_ambulance"
    OTHER = "other"


class DispatchStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class HospitalStatus(str, Enum):
    OPERATIONAL = "operational"
    LIMITED = "limited"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class MedicalTeamStatus(str, Enum):
    AVAILABLE = "available"
    ASSIGNED = "assigned"
    DEPLOYED = "deployed"
    RESTING = "resting"
    UNAVAILABLE = "unavailable"


class SupplyCategory(str, Enum):
    MEDICINE = "medicine"
    FIRST_AID = "first_aid"
    IV_FLUID = "iv_fluid"
    OXYGEN = "oxygen"
    PPE = "ppe"
    SANITATION = "sanitation"
    DIAGNOSTIC = "diagnostic"
    EQUIPMENT = "equipment"
    OTHER = "other"


class ReferralStatus(str, Enum):
    REQUESTED = "requested"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TRANSPORTING = "transporting"
    ARRIVED = "arrived"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ============================================================
# LOCATION
# ============================================================

class HealthcareLocation(HealthcareBase):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    district: str

    address: str | None = None
    landmark: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# PATIENT INFORMATION
# ============================================================

class PatientInformation(HealthcareBase):
    patient_id: UUID = Field(
        default_factory=uuid4
    )

    full_name: str | None = Field(
        default=None,
        max_length=150,
    )

    age: int | None = Field(
        default=None,
        ge=0,
        le=125,
    )

    contact_phone: str | None = None

    emergency_contact_phone: str | None = None

    medical_conditions: list[str] = Field(
        default_factory=list
    )

    known_allergies: list[str] = Field(
        default_factory=list
    )

    current_medications: list[str] = Field(
        default_factory=list
    )

    mobility_limitations: bool | None = None

    pregnancy_related: bool | None = None

    notes: str | None = Field(
        default=None,
        max_length=3000,
    )


# ============================================================
# MEDICAL EMERGENCY REQUEST
# ============================================================

class MedicalEmergencyCreate(HealthcareBase):
    sos_request_id: UUID | None = None

    emergency_type: MedicalEmergencyType

    location: HealthcareLocation

    description: str = Field(
        min_length=5,
        max_length=5000,
    )

    patient: PatientInformation | None = None

    patient_count: int = Field(
        default=1,
        ge=1,
    )

    conscious: bool | None = None
    breathing: bool | None = None

    severe_bleeding_reported: bool = False

    trapped: bool = False

    access_difficulties: list[str] = Field(
        default_factory=list
    )

    reported_by: str | None = None

    reported_at: datetime = Field(
        default_factory=utc_now
    )


class MedicalEmergencyResponse(MedicalEmergencyCreate):
    request_id: UUID = Field(
        default_factory=uuid4
    )

    status: MedicalRequestStatus = (
        MedicalRequestStatus.RECEIVED
    )

    triage_level: TriageLevel = (
        TriageLevel.UNASSESSED
    )

    assigned_team_id: UUID | None = None
    assigned_ambulance_id: UUID | None = None

    assigned_hospital_id: UUID | None = None

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# MEDICAL TRIAGE
# ============================================================

class MedicalTriageAssessment(HealthcareBase):
    assessment_id: UUID = Field(
        default_factory=uuid4
    )

    request_id: UUID

    triage_level: TriageLevel

    assessment_notes: str = Field(
        min_length=1,
        max_length=5000,
    )

    assessed_by: str = Field(
        min_length=1,
        max_length=200,
    )

    assessor_qualification: str | None = None

    assessed_at: datetime = Field(
        default_factory=utc_now
    )

    reassessment_required: bool = False

    @field_validator("triage_level")
    @classmethod
    def validate_assessed_level(
        cls,
        value: TriageLevel,
    ) -> TriageLevel:
        if value == TriageLevel.UNASSESSED:
            raise ValueError(
                "Completed triage requires an assessed level"
            )

        return value


class AITriageRecommendation(HealthcareBase):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    request_id: UUID

    suggested_level: TriageLevel = (
        TriageLevel.UNASSESSED
    )

    reasoning_summary: str

    warning_signs: list[str] = Field(
        default_factory=list
    )

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    model_name: str | None = None
    model_version: str | None = None

    human_review_required: bool = True

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# AMBULANCE
# ============================================================

class AmbulanceCreate(HealthcareBase):
    vehicle_registration: str = Field(
        min_length=3,
        max_length=30,
    )

    ambulance_type: AmbulanceType

    operator_name: str = Field(
        min_length=2,
        max_length=200,
    )

    district: str

    equipment: list[str] = Field(
        default_factory=list
    )

    capacity: int = Field(
        default=1,
        ge=1,
    )

    contact_phone: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class AmbulanceResponse(AmbulanceCreate):
    ambulance_id: UUID = Field(
        default_factory=uuid4
    )

    status: AmbulanceStatus = (
        AmbulanceStatus.UNAVAILABLE
    )

    current_location: HealthcareLocation | None = None

    assigned_request_id: UUID | None = None

    last_location_update: datetime | None = None

    verified: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# AMBULANCE DISPATCH
# ============================================================

class AmbulanceDispatchRequest(HealthcareBase):
    medical_request_id: UUID
    ambulance_id: UUID

    pickup_location: HealthcareLocation

    destination_hospital_id: UUID | None = None

    priority_notes: str | None = None


class AmbulanceDispatchResponse(HealthcareBase):
    dispatch_id: UUID = Field(
        default_factory=uuid4
    )

    medical_request_id: UUID
    ambulance_id: UUID

    status: DispatchStatus = (
        DispatchStatus.PENDING_APPROVAL
    )

    recommended_at: datetime = Field(
        default_factory=utc_now
    )

    approved_by: str | None = None
    approved_at: datetime | None = None

    confirmed_by_operator: str | None = None
    confirmed_at: datetime | None = None

    dispatched_at: datetime | None = None

    estimated_arrival_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_confirmation(self):
        if (
            self.status in {
                DispatchStatus.CONFIRMED,
                DispatchStatus.IN_PROGRESS,
                DispatchStatus.COMPLETED,
            }
            and (
                not self.confirmed_by_operator
                or self.confirmed_at is None
            )
        ):
            raise ValueError(
                "Confirmed dispatch requires operator confirmation"
            )

        return self


# ============================================================
# HOSPITAL
# ============================================================

class HospitalCreate(HealthcareBase):
    name: str = Field(
        min_length=2,
        max_length=250,
    )

    location: HealthcareLocation

    contact_phone: str | None = None

    emergency_department: bool = False

    specialties: list[str] = Field(
        default_factory=list
    )

    services: list[str] = Field(
        default_factory=list
    )


class HospitalResponse(HospitalCreate):
    hospital_id: UUID = Field(
        default_factory=uuid4
    )

    status: HospitalStatus = HospitalStatus.UNKNOWN

    verified: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# HOSPITAL CAPACITY
# ============================================================

class HospitalCapacity(HealthcareBase):
    hospital_id: UUID

    total_beds: int | None = Field(
        default=None,
        ge=0,
    )

    occupied_beds: int | None = Field(
        default=None,
        ge=0,
    )

    available_beds: int | None = Field(
        default=None,
        ge=0,
    )

    icu_available: int | None = Field(
        default=None,
        ge=0,
    )

    emergency_beds_available: int | None = Field(
        default=None,
        ge=0,
    )

    ventilators_available: int | None = Field(
        default=None,
        ge=0,
    )

    oxygen_supported_beds_available: int | None = Field(
        default=None,
        ge=0,
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    verified: bool = False

    @model_validator(mode="after")
    def validate_capacity(self):
        if (
            self.total_beds is not None
            and self.occupied_beds is not None
            and self.occupied_beds > self.total_beds
        ):
            raise ValueError(
                "Occupied beds cannot exceed total beds"
            )

        if (
            self.total_beds is not None
            and self.available_beds is not None
            and self.available_beds > self.total_beds
        ):
            raise ValueError(
                "Available beds cannot exceed total beds"
            )

        if all(
            value is not None
            for value in (
                self.total_beds,
                self.occupied_beds,
                self.available_beds,
            )
        ):
            if (
                self.occupied_beds + self.available_beds
                > self.total_beds
            ):
                raise ValueError(
                    "Occupied plus available beds exceeds total"
                )

        return self


# ============================================================
# MEDICAL RESPONSE TEAM
# ============================================================

class MedicalTeamCreate(HealthcareBase):
    team_name: str = Field(
        min_length=2,
        max_length=200,
    )

    district: str

    organization: str

    specializations: list[str] = Field(
        default_factory=list
    )

    personnel_count: int = Field(
        ge=1,
    )

    equipment: list[str] = Field(
        default_factory=list
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class MedicalTeamResponse(MedicalTeamCreate):
    team_id: UUID = Field(
        default_factory=uuid4
    )

    status: MedicalTeamStatus = (
        MedicalTeamStatus.UNAVAILABLE
    )

    current_location: HealthcareLocation | None = None

    assigned_request_id: UUID | None = None

    verified: bool = False

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# MEDICAL INVENTORY
# ============================================================

class MedicalInventoryItem(HealthcareBase):
    item_id: UUID = Field(
        default_factory=uuid4
    )

    item_name: str = Field(
        min_length=1,
        max_length=200,
    )

    category: SupplyCategory

    facility_id: UUID

    quantity_available: float = Field(
        ge=0,
    )

    unit: str

    minimum_stock_level: float | None = Field(
        default=None,
        ge=0,
    )

    expiry_date: datetime | None = None

    batch_number: str | None = None

    verified: bool = False

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# PATIENT REFERRAL
# ============================================================

class PatientReferralCreate(HealthcareBase):
    medical_request_id: UUID

    source_hospital_id: UUID | None = None

    destination_hospital_id: UUID

    reason: str = Field(
        min_length=5,
        max_length=3000,
    )

    requested_by: str

    transport_required: bool = True

    clinical_handover_reference: str | None = None


class PatientReferralResponse(PatientReferralCreate):
    referral_id: UUID = Field(
        default_factory=uuid4
    )

    status: ReferralStatus = ReferralStatus.REQUESTED

    accepted_by: str | None = None
    accepted_at: datetime | None = None

    arrival_confirmed_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# MEDICAL STATUS UPDATE
# ============================================================

class MedicalStatusUpdate(HealthcareBase):
    request_id: UUID

    status: MedicalRequestStatus

    notes: str | None = Field(
        default=None,
        max_length=3000,
    )


# ============================================================
# MEDICAL EVENT AUDIT
# ============================================================

class MedicalAuditEvent(HealthcareBase):
    event_id: UUID = Field(
        default_factory=uuid4
    )

    entity_id: UUID

    entity_type: str

    action: str

    performed_by: str

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# HEALTHCARE DASHBOARD
# ============================================================

class HealthcareDashboard(HealthcareBase):
    district: str | None = None

    active_medical_requests: int | None = Field(
        default=None,
        ge=0,
    )

    awaiting_triage: int | None = Field(
        default=None,
        ge=0,
    )

    available_ambulances: int | None = Field(
        default=None,
        ge=0,
    )

    active_ambulance_missions: int | None = Field(
        default=None,
        ge=0,
    )

    operational_hospitals: int | None = Field(
        default=None,
        ge=0,
    )

    available_hospital_beds: int | None = Field(
        default=None,
        ge=0,
    )

    available_icu_beds: int | None = Field(
        default=None,
        ge=0,
    )

    deployed_medical_teams: int | None = Field(
        default=None,
        ge=0,
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    data_verified: bool = False


# ============================================================
# GENERIC API RESPONSE
# ============================================================

class HealthcareAPIResponse(HealthcareBase):
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
    "MedicalEmergencyType",
    "TriageLevel",
    "MedicalRequestStatus",
    "AmbulanceStatus",
    "AmbulanceType",
    "DispatchStatus",
    "HospitalStatus",
    "MedicalTeamStatus",
    "SupplyCategory",
    "ReferralStatus",
    "HealthcareLocation",
    "PatientInformation",
    "MedicalEmergencyCreate",
    "MedicalEmergencyResponse",
    "MedicalTriageAssessment",
    "AITriageRecommendation",
    "AmbulanceCreate",
    "AmbulanceResponse",
    "AmbulanceDispatchRequest",
    "AmbulanceDispatchResponse",
    "HospitalCreate",
    "HospitalResponse",
    "HospitalCapacity",
    "MedicalTeamCreate",
    "MedicalTeamResponse",
    "MedicalInventoryItem",
    "PatientReferralCreate",
    "PatientReferralResponse",
    "MedicalStatusUpdate",
    "MedicalAuditEvent",
    "HealthcareDashboard",
    "HealthcareAPIResponse",
    "normalize_district",
    "validate_datetime",
]
