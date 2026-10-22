from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/shelter.py

Pydantic v2 schemas for flood relief shelter management.

Modules:
- Shelter registration and verification
- Shelter capacity and occupancy
- Shelter facilities and accessibility
- Evacuee registration
- Family/group accommodation
- Shelter allocation recommendations
- Human-approved shelter assignments
- Check-in and check-out
- Shelter transfers
- Shelter supplies and shortages
- Shelter safety inspections
- Shelter analytics and audit logs

SAFETY:
- AI allocation recommendations are advisory only.
- Shelter availability must be verified before assignment.
- Do not treat missing capacity data as available capacity.
- Shelter assignments require authorized confirmation.
- Protect evacuee identities, contact details and locations.
- Do not publicly expose individual resident records.
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


def validate_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Datetime must include timezone information"
        )
    return value


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


# ============================================================
# BASE SCHEMA
# ============================================================

class ShelterBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ============================================================
# ENUMS
# ============================================================

class ShelterType(str, Enum):
    SCHOOL = "school"
    COMMUNITY_HALL = "community_hall"
    GOVERNMENT_BUILDING = "government_building"
    RELIEF_CAMP = "relief_camp"
    TEMPORARY_SHELTER = "temporary_shelter"
    RELIGIOUS_BUILDING = "religious_building"
    SPORTS_COMPLEX = "sports_complex"
    OTHER = "other"


class ShelterStatus(str, Enum):
    PENDING_VERIFICATION = "pending_verification"
    OPERATIONAL = "operational"
    LIMITED = "limited"
    FULL = "full"
    CLOSED = "closed"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class ShelterFacility(str, Enum):
    DRINKING_WATER = "drinking_water"
    FOOD = "food"
    TOILETS = "toilets"
    BATHROOMS = "bathrooms"
    ELECTRICITY = "electricity"
    GENERATOR = "generator"
    MEDICAL_SUPPORT = "medical_support"
    FIRST_AID = "first_aid"
    WHEELCHAIR_ACCESS = "wheelchair_access"
    ACCESSIBLE_TOILETS = "accessible_toilets"
    CHILD_FRIENDLY_SPACE = "child_friendly_space"
    WOMEN_SAFE_SPACE = "women_safe_space"
    INFANT_CARE = "infant_care"
    ELDERLY_SUPPORT = "elderly_support"
    INTERNET = "internet"
    CHARGING_STATIONS = "charging_stations"
    SECURITY = "security"
    PET_FRIENDLY = "pet_friendly"
    OTHER = "other"


class ResidentStatus(str, Enum):
    REGISTERED = "registered"
    WAITLISTED = "waitlisted"
    ASSIGNED = "assigned"
    CHECKED_IN = "checked_in"
    TRANSFER_PENDING = "transfer_pending"
    TRANSFERRED = "transferred"
    CHECKED_OUT = "checked_out"
    CANCELLED = "cancelled"


class AllocationStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CAPACITY_RESERVED = "capacity_reserved"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TransferStatus(str, Enum):
    REQUESTED = "requested"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ShelterPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupplyStatus(str, Enum):
    SUFFICIENT = "sufficient"
    LOW = "low"
    OUT_OF_STOCK = "out_of_stock"
    UNKNOWN = "unknown"


class InspectionStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    CONDITIONAL = "conditional"
    FAILED = "failed"


# ============================================================
# LOCATION
# ============================================================

class ShelterLocation(ShelterBase):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    district: str

    address: str | None = None
    landmark: str | None = None
    panchayat: str | None = None
    municipality: str | None = None
    ward: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# SHELTER REGISTRATION
# ============================================================

class ShelterCreate(ShelterBase):
    name: str = Field(
        min_length=2,
        max_length=250,
    )

    shelter_type: ShelterType

    location: ShelterLocation

    description: str | None = Field(
        default=None,
        max_length=3000,
    )

    managing_organization: str | None = None

    manager_name: str | None = None

    manager_contact: str | None = None

    total_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    facilities: list[ShelterFacility] = Field(
        default_factory=list
    )

    accessible: bool = False

    family_accommodation: bool = False

    medical_support_available: bool = False

    registration_notes: str | None = None


class ShelterResponse(ShelterCreate):
    shelter_id: UUID = Field(
        default_factory=uuid4
    )

    status: ShelterStatus = (
        ShelterStatus.PENDING_VERIFICATION
    )

    verified: bool = False

    verified_by: str | None = None

    verified_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


class ShelterUpdate(ShelterBase):
    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=250,
    )

    description: str | None = None

    manager_name: str | None = None

    manager_contact: str | None = None

    total_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    facilities: list[ShelterFacility] | None = None

    accessible: bool | None = None

    family_accommodation: bool | None = None

    medical_support_available: bool | None = None


# ============================================================
# SHELTER CAPACITY
# ============================================================

class ShelterCapacity(ShelterBase):
    shelter_id: UUID

    total_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    current_occupancy: int | None = Field(
        default=None,
        ge=0,
    )

    reserved_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    available_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    occupancy_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    verified: bool = False

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_capacity(self):
        total = self.total_capacity
        occupied = self.current_occupancy
        reserved = self.reserved_capacity
        available = self.available_capacity

        if total is not None:
            if occupied is not None and occupied > total:
                raise ValueError(
                    "Occupancy exceeds total capacity"
                )

            if reserved is not None and reserved > total:
                raise ValueError(
                    "Reserved capacity exceeds total capacity"
                )

            if available is not None and available > total:
                raise ValueError(
                    "Available capacity exceeds total capacity"
                )

            if occupied is not None and reserved is not None:
                if occupied + reserved > total:
                    raise ValueError(
                        "Occupancy plus reservations exceeds capacity"
                    )

            if all(
                value is not None
                for value in (
                    occupied,
                    reserved,
                    available,
                )
            ):
                if occupied + reserved + available != total:
                    raise ValueError(
                        "Capacity balance is inconsistent"
                    )

        if (
            total is not None
            and occupied is not None
            and self.occupancy_percent is not None
        ):
            expected = (
                occupied / total * 100
                if total > 0
                else 0.0
            )

            if abs(
                expected - self.occupancy_percent
            ) > 0.1:
                raise ValueError(
                    "Occupancy percentage is inconsistent"
                )

        return self


class ShelterCapacityUpdate(ShelterBase):
    shelter_id: UUID

    total_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    reason: str = Field(
        min_length=5,
        max_length=2000,
    )


# ============================================================
# RESIDENT / EVACUEE INFORMATION
# ============================================================

class ShelterResidentCreate(ShelterBase):
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

    home_district: str | None = None

    home_address: str | None = None

    emergency_contact_name: str | None = None

    emergency_contact_phone: str | None = None

    requires_accessibility_support: bool = False

    requires_medical_support: bool = False

    medical_support_notes: str | None = None

    registration_reference: str | None = None

    @field_validator("home_district")
    @classmethod
    def validate_home_district(
        cls,
        value: str | None,
    ) -> str | None:
        return (
            normalize_district(value)
            if value is not None
            else None
        )


class ShelterResidentResponse(ShelterResidentCreate):
    resident_id: UUID = Field(
        default_factory=uuid4
    )

    shelter_id: UUID | None = None

    status: ResidentStatus = (
        ResidentStatus.REGISTERED
    )

    registered_at: datetime = Field(
        default_factory=utc_now
    )

    checked_in_at: datetime | None = None

    checked_out_at: datetime | None = None


# ============================================================
# FAMILY / GROUP REGISTRATION
# ============================================================

class ShelterFamilyRegistration(ShelterBase):
    family_id: UUID = Field(
        default_factory=uuid4
    )

    primary_contact_name: str | None = None

    contact_phone: str | None = None

    members: list[ShelterResidentCreate] = Field(
        min_length=1
    )

    requested_shelter_id: UUID | None = None

    keep_family_together: bool = True

    special_requirements: list[str] = Field(
        default_factory=list
    )

    registered_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# SHELTER ALLOCATION REQUEST
# ============================================================

class ShelterAllocationRequest(ShelterBase):
    request_id: UUID = Field(
        default_factory=uuid4
    )

    sos_request_id: UUID | None = None

    family_id: UUID | None = None

    resident_ids: list[UUID] = Field(
        default_factory=list
    )

    people_count: int = Field(
        ge=1
    )

    current_location: ShelterLocation

    preferred_district: str | None = None

    accessibility_required: bool = False

    medical_support_required: bool = False

    family_accommodation_required: bool = False

    priority: ShelterPriority = ShelterPriority.MEDIUM

    notes: str | None = None

    @field_validator("preferred_district")
    @classmethod
    def validate_preferred_district(
        cls,
        value: str | None,
    ) -> str | None:
        return (
            normalize_district(value)
            if value is not None
            else None
        )

    @model_validator(mode="after")
    def validate_residents(self):
        if (
            self.resident_ids
            and len(self.resident_ids) != self.people_count
        ):
            raise ValueError(
                "Resident ID count must match people_count"
            )

        if len(set(self.resident_ids)) != len(
            self.resident_ids
        ):
            raise ValueError(
                "Duplicate resident IDs are not allowed"
            )

        return self


# ============================================================
# AI SHELTER MATCH
# ============================================================

class ShelterMatch(ShelterBase):
    shelter_id: UUID

    match_score: float = Field(
        ge=0,
        le=100,
    )

    distance_km: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_travel_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    available_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    capacity_verified: bool = False

    shelter_operational_verified: bool = False

    route_safety_verified: bool = False

    accessibility_suitable: bool | None = None

    medical_support_suitable: bool | None = None

    reasons: list[str] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )


class ShelterAllocationRecommendation(ShelterBase):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    request_id: UUID

    recommended_shelter_id: UUID | None = None

    alternatives: list[ShelterMatch] = Field(
        default_factory=list
    )

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
# SHELTER ASSIGNMENT
# ============================================================

class ShelterAssignmentCreate(ShelterBase):
    request_id: UUID

    shelter_id: UUID

    resident_ids: list[UUID] = Field(
        default_factory=list
    )

    people_count: int = Field(
        ge=1
    )

    recommendation_id: UUID | None = None

    assignment_notes: str | None = None

    @model_validator(mode="after")
    def validate_assignment(self):
        if (
            self.resident_ids
            and len(self.resident_ids) != self.people_count
        ):
            raise ValueError(
                "Resident count does not match assignment size"
            )

        return self


class ShelterAssignmentResponse(
    ShelterAssignmentCreate
):
    assignment_id: UUID = Field(
        default_factory=uuid4
    )

    status: AllocationStatus = (
        AllocationStatus.PENDING_APPROVAL
    )

    approved_by: str | None = None

    approved_at: datetime | None = None

    capacity_reserved_at: datetime | None = None

    confirmed_by_shelter: str | None = None

    confirmed_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_confirmation(self):
        if self.status == AllocationStatus.CONFIRMED:
            if not (
                self.approved_by
                and self.approved_at
                and self.capacity_reserved_at
                and self.confirmed_by_shelter
                and self.confirmed_at
            ):
                raise ValueError(
                    "Confirmed assignment requires approval, "
                    "capacity reservation and shelter confirmation"
                )

        return self


# ============================================================
# CHECK-IN / CHECK-OUT
# ============================================================

class ShelterCheckInRequest(ShelterBase):
    assignment_id: UUID

    resident_ids: list[UUID] = Field(
        min_length=1
    )

    confirmed_by: str

    notes: str | None = None


class ShelterCheckInResponse(ShelterBase):
    assignment_id: UUID

    checked_in_resident_ids: list[UUID]

    confirmed_by: str

    checked_in_at: datetime = Field(
        default_factory=utc_now
    )


class ShelterCheckOutRequest(ShelterBase):
    shelter_id: UUID

    resident_ids: list[UUID] = Field(
        min_length=1
    )

    destination_description: str | None = None

    reason: str | None = None

    confirmed_by: str


class ShelterCheckOutResponse(ShelterBase):
    shelter_id: UUID

    checked_out_resident_ids: list[UUID]

    confirmed_by: str

    checked_out_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# SHELTER TRANSFER
# ============================================================

class ShelterTransferCreate(ShelterBase):
    source_shelter_id: UUID

    destination_shelter_id: UUID

    resident_ids: list[UUID] = Field(
        min_length=1
    )

    reason: str = Field(
        min_length=5,
        max_length=2000,
    )

    transportation_required: bool = False

    @model_validator(mode="after")
    def validate_transfer(self):
        if (
            self.source_shelter_id
            == self.destination_shelter_id
        ):
            raise ValueError(
                "Source and destination shelters must differ"
            )

        if len(set(self.resident_ids)) != len(
            self.resident_ids
        ):
            raise ValueError(
                "Duplicate resident IDs are not allowed"
            )

        return self


class ShelterTransferResponse(ShelterTransferCreate):
    transfer_id: UUID = Field(
        default_factory=uuid4
    )

    status: TransferStatus = (
        TransferStatus.REQUESTED
    )

    approved_by: str | None = None

    approved_at: datetime | None = None

    destination_confirmed_by: str | None = None

    completed_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# SHELTER SUPPLIES
# ============================================================

class ShelterSupplyItem(ShelterBase):
    item_id: UUID = Field(
        default_factory=uuid4
    )

    shelter_id: UUID

    item_name: str = Field(
        min_length=1,
        max_length=200,
    )

    quantity_available: float | None = Field(
        default=None,
        ge=0,
    )

    minimum_required: float | None = Field(
        default=None,
        ge=0,
    )

    unit: str

    status: SupplyStatus = SupplyStatus.UNKNOWN

    verified: bool = False

    updated_at: datetime = Field(
        default_factory=utc_now
    )


class ShelterSupplyRequest(ShelterBase):
    shelter_id: UUID

    item_name: str

    quantity_required: float = Field(
        gt=0
    )

    unit: str

    priority: ShelterPriority = (
        ShelterPriority.MEDIUM
    )

    reason: str | None = None


# ============================================================
# SAFETY INSPECTION
# ============================================================

class ShelterSafetyInspection(ShelterBase):
    inspection_id: UUID = Field(
        default_factory=uuid4
    )

    shelter_id: UUID

    inspection_status: InspectionStatus

    structural_safety_verified: bool = False

    electrical_safety_verified: bool = False

    drinking_water_verified: bool = False

    sanitation_verified: bool = False

    fire_safety_verified: bool = False

    flood_access_safety_verified: bool = False

    findings: list[str] = Field(
        default_factory=list
    )

    corrective_actions: list[str] = Field(
        default_factory=list
    )

    inspected_by: str

    inspected_at: datetime = Field(
        default_factory=utc_now
    )

    next_inspection_due: datetime | None = None

    @model_validator(mode="after")
    def validate_inspection(self):
        if self.inspection_status == InspectionStatus.PASSED:
            checks = (
                self.structural_safety_verified,
                self.electrical_safety_verified,
                self.drinking_water_verified,
                self.sanitation_verified,
                self.fire_safety_verified,
                self.flood_access_safety_verified,
            )

            if not all(checks):
                raise ValueError(
                    "Passed inspection requires all safety "
                    "checks to be verified"
                )

        return self


# ============================================================
# HUMAN APPROVAL
# ============================================================

class ShelterApprovalRequest(ShelterBase):
    resource_id: UUID

    resource_type: str

    approve: bool

    reason: str = Field(
        min_length=5,
        max_length=2000,
    )


class ShelterApprovalResponse(ShelterBase):
    resource_id: UUID

    approved: bool

    reviewed_by: str

    reviewed_at: datetime = Field(
        default_factory=utc_now
    )

    authorization_verified: bool = False


# ============================================================
# SHELTER AUDIT
# ============================================================

class ShelterAuditEvent(ShelterBase):
    event_id: UUID = Field(
        default_factory=uuid4
    )

    entity_id: UUID

    entity_type: str

    action: str

    performed_by: str

    # Store only non-sensitive metadata here.
    details: dict[str, Any] = Field(
        default_factory=dict
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# PUBLIC SHELTER INFORMATION
# ============================================================

class PublicShelterSummary(ShelterBase):
    shelter_id: UUID

    name: str

    shelter_type: ShelterType

    district: str

    status: ShelterStatus

    total_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    available_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    facilities: list[ShelterFacility] = Field(
        default_factory=list
    )

    capacity_verified: bool = False

    last_updated: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# SHELTER DASHBOARD
# ============================================================

class ShelterDashboard(ShelterBase):
    district: str | None = None

    total_shelters: int | None = Field(
        default=None,
        ge=0,
    )

    operational_shelters: int | None = Field(
        default=None,
        ge=0,
    )

    full_shelters: int | None = Field(
        default=None,
        ge=0,
    )

    shelters_needing_inspection: int | None = Field(
        default=None,
        ge=0,
    )

    total_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    current_occupancy: int | None = Field(
        default=None,
        ge=0,
    )

    reserved_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    available_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    occupancy_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    pending_allocations: int | None = Field(
        default=None,
        ge=0,
    )

    pending_transfers: int | None = Field(
        default=None,
        ge=0,
    )

    shelters_with_supply_shortages: int | None = Field(
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

class ShelterAPIResponse(ShelterBase):
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
    "ShelterType",
    "ShelterStatus",
    "ShelterFacility",
    "ResidentStatus",
    "AllocationStatus",
    "TransferStatus",
    "ShelterPriority",
    "SupplyStatus",
    "InspectionStatus",
    "ShelterLocation",
    "ShelterCreate",
    "ShelterResponse",
    "ShelterUpdate",
    "ShelterCapacity",
    "ShelterCapacityUpdate",
    "ShelterResidentCreate",
    "ShelterResidentResponse",
    "ShelterFamilyRegistration",
    "ShelterAllocationRequest",
    "ShelterMatch",
    "ShelterAllocationRecommendation",
    "ShelterAssignmentCreate",
    "ShelterAssignmentResponse",
    "ShelterCheckInRequest",
    "ShelterCheckInResponse",
    "ShelterCheckOutRequest",
    "ShelterCheckOutResponse",
    "ShelterTransferCreate",
    "ShelterTransferResponse",
    "ShelterSupplyItem",
    "ShelterSupplyRequest",
    "ShelterSafetyInspection",
    "ShelterApprovalRequest",
    "ShelterApprovalResponse",
    "ShelterAuditEvent",
    "PublicShelterSummary",
    "ShelterDashboard",
    "ShelterAPIResponse",
    "normalize_district",
    "validate_datetime",
]
