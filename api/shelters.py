from __future__ import annotations

"""
backend/api/shelters.py

Kerala Flood Relief Coordination Agent
Shelter Management & Emergency Accommodation API

Python 3.11+ | FastAPI | Pydantic v2

Features:
    - Shelter discovery and capacity monitoring
    - Shelter registration and updates
    - Resident check-in and check-out
    - Household management
    - Bed and room allocation
    - Occupancy tracking
    - Facility management
    - Resource requests
    - Safety inspections
    - Shelter alerts
    - Resident welfare checks
    - Shelter transfer coordination
    - Audit event history

Operational changes are delegated to a configured
ShelterService implementation.

Important:
    - No fabricated shelter availability.
    - No automatic resident transfers.
    - No unverified occupancy updates.
    - Resident PII requires strict authorization.
"""


import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Protocol

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.dependencies import get_current_user


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/shelters",
    tags=["Shelter Management"],
)

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# ROLES
# ============================================================

ADMIN_ROLES = {
    "admin",
    "super_admin",
}

SHELTER_ROLES = {
    "shelter_manager",
    "shelter_coordinator",
    "shelter_operator",
}

COMMAND_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "incident_commander",
}

MEDICAL_ROLES = {
    "medical_officer",
    "healthcare_coordinator",
}

READ_ROLES = (
    ADMIN_ROLES
    | SHELTER_ROLES
    | COMMAND_ROLES
    | MEDICAL_ROLES
    | {"auditor"}
)

WRITE_ROLES = (
    ADMIN_ROLES
    | SHELTER_ROLES
    | COMMAND_ROLES
)

APPROVAL_ROLES = (
    ADMIN_ROLES
    | COMMAND_ROLES
    | {"shelter_manager"}
)

RESIDENT_ROLES = (
    ADMIN_ROLES
    | SHELTER_ROLES
    | {"authorized_case_worker"}
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class ShelterType(str, Enum):
    GOVERNMENT = "government"
    SCHOOL = "school"
    COMMUNITY_HALL = "community_hall"
    RELIGIOUS_FACILITY = "religious_facility"
    TEMPORARY_CAMP = "temporary_camp"
    HOSPITAL_ADJACENT = "hospital_adjacent"
    OTHER = "other"


class ShelterStatus(str, Enum):
    PLANNED = "planned"
    INSPECTION_PENDING = "inspection_pending"
    READY = "ready"
    OPEN = "open"
    AT_CAPACITY = "at_capacity"
    RESTRICTED = "restricted"
    CLOSED = "closed"


class ResidentStatus(str, Enum):
    REGISTERED = "registered"
    CHECKED_IN = "checked_in"
    TEMPORARILY_ABSENT = "temporarily_absent"
    TRANSFERRED = "transferred"
    CHECKED_OUT = "checked_out"


class BedStatus(str, Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    OUT_OF_SERVICE = "out_of_service"


class FacilityType(str, Enum):
    DRINKING_WATER = "drinking_water"
    TOILET = "toilet"
    BATHROOM = "bathroom"
    ELECTRICITY = "electricity"
    KITCHEN = "kitchen"
    MEDICAL_AREA = "medical_area"
    ACCESSIBLE_FACILITY = "accessible_facility"
    CHILD_FRIENDLY_AREA = "child_friendly_area"
    SECURITY = "security"
    OTHER = "other"


class ShelterPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResourceCategory(str, Enum):
    FOOD = "food"
    WATER = "water"
    MEDICINE = "medicine"
    HYGIENE = "hygiene"
    BEDDING = "bedding"
    CLOTHING = "clothing"
    SANITATION = "sanitation"
    EQUIPMENT = "equipment"
    OTHER = "other"


class InspectionOutcome(str, Enum):
    PASS = "pass"
    CONDITIONAL = "conditional"
    FAIL = "fail"


class ShelterAlertType(str, Enum):
    OVERCROWDING = "overcrowding"
    FOOD_SHORTAGE = "food_shortage"
    WATER_SHORTAGE = "water_shortage"
    MEDICAL_EMERGENCY = "medical_emergency"
    SAFETY_RISK = "safety_risk"
    SANITATION_ISSUE = "sanitation_issue"
    SECURITY_INCIDENT = "security_incident"
    OTHER = "other"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ShelterCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    district_id: int = Field(ge=1)

    name: str = Field(
        min_length=3,
        max_length=255,
    )

    shelter_type: ShelterType

    location_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    total_capacity: int = Field(ge=1)

    accessible_capacity: int = Field(
        default=0,
        ge=0,
    )

    contact_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    description: str | None = Field(
        default=None,
        max_length=3000,
    )

    @model_validator(mode="after")
    def validate_capacity(self) -> "ShelterCreate":
        if self.accessible_capacity > self.total_capacity:
            raise ValueError(
                "Accessible capacity cannot exceed total capacity"
            )
        return self


class ShelterUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        min_length=3,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
        max_length=3000,
    )

    total_capacity: int | None = Field(
        default=None,
        ge=1,
    )

    accessible_capacity: int | None = Field(
        default=None,
        ge=0,
    )


class ShelterStatusUpdate(BaseModel):
    status: ShelterStatus

    reason: str = Field(
        min_length=10,
        max_length=3000,
    )

    safety_verified: bool = False

    capacity_verified: bool = False


class ShelterRoomCreate(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=150,
    )

    capacity: int = Field(ge=1)

    accessible: bool = False


class ShelterBedCreate(BaseModel):
    room_id: str

    bed_reference: str = Field(
        min_length=2,
        max_length=150,
    )

    accessible: bool = False

    @field_validator("room_id")
    @classmethod
    def validate_room_id(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))


class ShelterFacilityCreate(BaseModel):
    facility_type: FacilityType

    quantity: int = Field(ge=0)

    operational: bool = True

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )


class ShelterResidentCheckIn(BaseModel):
    resident_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    household_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    bed_id: str | None = None

    check_in_at: datetime

    consent_recorded: bool = False

    @field_validator("bed_id")
    @classmethod
    def validate_bed_id(
        cls,
        value: str | None,
    ) -> str | None:
        return str(uuid.UUID(value)) if value else None


class ShelterResidentCheckOut(BaseModel):
    resident_id: str

    checked_out_at: datetime

    destination_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    reason: str = Field(
        min_length=5,
        max_length=3000,
    )

    @field_validator("resident_id")
    @classmethod
    def validate_resident_id(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))


class ShelterTransferRequest(BaseModel):
    resident_id: str

    destination_shelter_id: str

    reason: str = Field(
        min_length=10,
        max_length=3000,
    )

    destination_capacity_verified: bool = False

    transport_ready: bool = False

    authorized_approval: bool = False

    @field_validator(
        "resident_id",
        "destination_shelter_id",
    )
    @classmethod
    def validate_ids(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))


class ShelterOccupancySnapshotCreate(BaseModel):
    observed_at: datetime

    occupied: int = Field(ge=0)

    available: int = Field(ge=0)

    reserved: int = Field(ge=0)

    source_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    verified: bool = False


class ShelterResourceRequestCreate(BaseModel):
    category: ResourceCategory

    item_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    quantity: int = Field(ge=1)

    priority: ShelterPriority

    justification: str = Field(
        min_length=10,
        max_length=3000,
    )


class ShelterInspectionCreate(BaseModel):
    inspected_at: datetime

    outcome: InspectionOutcome

    findings: str = Field(
        min_length=10,
        max_length=5000,
    )

    fire_safety_verified: bool = False

    structural_safety_verified: bool = False

    sanitation_verified: bool = False

    accessibility_verified: bool = False

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )


class ShelterAlertCreate(BaseModel):
    alert_type: ShelterAlertType

    priority: ShelterPriority

    description: str = Field(
        min_length=10,
        max_length=3000,
    )

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )


class ShelterWelfareCheckCreate(BaseModel):
    resident_id: str

    checked_at: datetime

    notes: str = Field(
        min_length=10,
        max_length=3000,
    )

    urgent_assistance_required: bool = False

    @field_validator("resident_id")
    @classmethod
    def validate_resident_id(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ShelterResponse(BaseModel):
    id: str

    district_id: int

    name: str

    status: str

    total_capacity: int

    occupied: int = 0

    available: int = 0

    created_at: datetime | None = None


class ShelterListResponse(BaseModel):
    items: list[ShelterResponse]

    total: int

    limit: int

    offset: int


class ShelterStatisticsResponse(BaseModel):
    total_shelters: int

    by_status: dict[str, int]

    total_capacity: int

    occupied_capacity: int

    available_capacity: int

    generated_at: datetime


class ShelterResidentResponse(BaseModel):
    id: str

    shelter_id: str

    status: str

    checked_in_at: datetime | None = None


class ShelterRoomResponse(BaseModel):
    id: str

    shelter_id: str

    name: str

    capacity: int


class ShelterBedResponse(BaseModel):
    id: str

    shelter_id: str

    status: str


class ShelterFacilityResponse(BaseModel):
    id: str

    shelter_id: str

    facility_type: str

    operational: bool


class ShelterOccupancyResponse(BaseModel):
    id: str

    shelter_id: str

    occupied: int

    available: int

    observed_at: datetime


class ShelterResourceResponse(BaseModel):
    id: str

    shelter_id: str

    category: ResourceCategory

    status: str


class ShelterInspectionResponse(BaseModel):
    id: str

    shelter_id: str

    outcome: InspectionOutcome

    inspected_at: datetime


class ShelterAlertResponse(BaseModel):
    id: str

    shelter_id: str

    alert_type: ShelterAlertType

    priority: ShelterPriority


class ShelterEventResponse(BaseModel):
    id: str

    shelter_id: str

    event_type: str

    created_at: datetime


class ShelterHealthResponse(BaseModel):
    service: str

    status: str

    timestamp: datetime


# ============================================================
# SERVICE CONTRACT
# ============================================================

class ShelterService(Protocol):
    """
    Domain service must enforce:

    - Shelter existence and district access
    - Verified capacity and occupancy
    - Atomic bed reservation and assignment
    - Resident and household privacy
    - Duplicate check-in prevention
    - Safeguarding of vulnerable residents
    - Authorized resident transfers
    - Verified safety inspections
    - Resource request tracking
    - Human approval for operational changes
    - Immutable audit events
    """

    async def health(self) -> ShelterHealthResponse:
        ...

    async def list_shelters(
        self,
        district_id: int | None,
        limit: int,
        offset: int,
        actor_id: str,
    ) -> ShelterListResponse:
        ...

    async def statistics(
        self,
        district_id: int | None,
        actor_id: str,
    ) -> ShelterStatisticsResponse:
        ...

    async def create_shelter(
        self,
        payload: ShelterCreate,
        actor_id: str,
    ) -> ShelterResponse:
        ...

    async def get_shelter(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> ShelterResponse | None:
        ...

    async def update_shelter(
        self,
        shelter_id: str,
        payload: ShelterUpdate,
        actor_id: str,
    ) -> ShelterResponse:
        ...

    async def change_status(
        self,
        shelter_id: str,
        payload: ShelterStatusUpdate,
        actor_id: str,
    ) -> ShelterResponse:
        ...

    async def list_rooms(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterRoomResponse]:
        ...

    async def create_room(
        self,
        shelter_id: str,
        payload: ShelterRoomCreate,
        actor_id: str,
    ) -> ShelterRoomResponse:
        ...

    async def list_beds(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterBedResponse]:
        ...

    async def create_bed(
        self,
        shelter_id: str,
        payload: ShelterBedCreate,
        actor_id: str,
    ) -> ShelterBedResponse:
        ...

    async def list_facilities(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterFacilityResponse]:
        ...

    async def create_facility(
        self,
        shelter_id: str,
        payload: ShelterFacilityCreate,
        actor_id: str,
    ) -> ShelterFacilityResponse:
        ...

    async def list_residents(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterResidentResponse]:
        ...

    async def check_in_resident(
        self,
        shelter_id: str,
        payload: ShelterResidentCheckIn,
        actor_id: str,
    ) -> ShelterResidentResponse:
        ...

    async def check_out_resident(
        self,
        shelter_id: str,
        payload: ShelterResidentCheckOut,
        actor_id: str,
    ) -> ShelterResidentResponse:
        ...

    async def transfer_resident(
        self,
        shelter_id: str,
        payload: ShelterTransferRequest,
        actor_id: str,
    ) -> ShelterResidentResponse:
        ...

    async def list_occupancy(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterOccupancyResponse]:
        ...

    async def record_occupancy(
        self,
        shelter_id: str,
        payload: ShelterOccupancySnapshotCreate,
        actor_id: str,
    ) -> ShelterOccupancyResponse:
        ...

    async def list_resources(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterResourceResponse]:
        ...

    async def request_resources(
        self,
        shelter_id: str,
        payload: ShelterResourceRequestCreate,
        actor_id: str,
    ) -> ShelterResourceResponse:
        ...

    async def list_inspections(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterInspectionResponse]:
        ...

    async def create_inspection(
        self,
        shelter_id: str,
        payload: ShelterInspectionCreate,
        actor_id: str,
    ) -> ShelterInspectionResponse:
        ...

    async def list_alerts(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterAlertResponse]:
        ...

    async def create_alert(
        self,
        shelter_id: str,
        payload: ShelterAlertCreate,
        actor_id: str,
    ) -> ShelterAlertResponse:
        ...

    async def welfare_check(
        self,
        shelter_id: str,
        payload: ShelterWelfareCheckCreate,
        actor_id: str,
    ) -> dict[str, Any]:
        ...

    async def list_events(
        self,
        shelter_id: str,
        actor_id: str,
    ) -> list[ShelterEventResponse]:
        ...


# ============================================================
# DEPENDENCIES
# ============================================================

def get_shelter_service(
    request: Request,
) -> ShelterService:
    service = getattr(
        request.app.state,
        "shelter_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Shelter service is not configured",
        )

    return service


ShelterServiceDependency = Annotated[
    ShelterService,
    Depends(get_shelter_service),
]


# ============================================================
# SECURITY HELPERS
# ============================================================

def validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid UUID",
        ) from None


def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return str(identifier)


def role_of(user: Any) -> str:
    value = getattr(user, "role", "")

    if isinstance(value, Enum):
        value = value.value

    return str(value).lower()


def require_role(
    user: Any,
    roles: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) not in roles:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions",
        )


def require_district_access(
    user: Any,
    district_id: int,
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) in ADMIN_ROLES:
        return

    assigned = getattr(
        user,
        "district_id",
        None,
    )

    if assigned is None or int(assigned) != district_id:
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )


def scoped_district(
    user: Any,
    district_id: int | None,
) -> int | None:
    require_role(user, READ_ROLES)

    if (
        getattr(user, "is_superuser", False)
        or role_of(user) in ADMIN_ROLES
    ):
        return district_id

    assigned = getattr(
        user,
        "district_id",
        None,
    )

    if assigned is None:
        raise HTTPException(
            status_code=403,
            detail="District assignment required",
        )

    if district_id is not None and district_id != int(assigned):
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )

    return int(assigned)


async def require_shelter_access(
    service: ShelterService,
    shelter_id: str,
    user: Any,
) -> ShelterResponse:
    shelter = await service.get_shelter(
        shelter_id=shelter_id,
        actor_id=actor_id(user),
    )

    if shelter is None:
        raise HTTPException(
            status_code=404,
            detail="Shelter not found",
        )

    require_district_access(
        user,
        shelter.district_id,
    )

    return shelter


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=ShelterHealthResponse,
)
async def shelter_health(
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterHealthResponse:
    require_role(current_user, READ_ROLES)

    return await service.health()


# ============================================================
# LIST SHELTERS
# ============================================================

@router.get(
    "",
    response_model=ShelterListResponse,
)
async def list_shelters(
    current_user: CurrentUser,
    service: ShelterServiceDependency,
    district_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=20, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
) -> ShelterListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.list_shelters(
        district_id=district_id,
        limit=limit,
        offset=offset,
        actor_id=actor_id(current_user),
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=ShelterStatisticsResponse,
)
async def shelter_statistics(
    current_user: CurrentUser,
    service: ShelterServiceDependency,
    district_id: int | None = Query(default=None, ge=1),
) -> ShelterStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.statistics(
        district_id=district_id,
        actor_id=actor_id(current_user),
    )


# ============================================================
# CREATE SHELTER
# ============================================================

@router.post(
    "",
    response_model=ShelterResponse,
    status_code=201,
)
async def create_shelter(
    payload: ShelterCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterResponse:
    require_role(current_user, WRITE_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    return await service.create_shelter(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# SHELTER DETAILS
# ============================================================

@router.get(
    "/{shelter_id}",
    response_model=ShelterResponse,
)
async def get_shelter(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterResponse:
    require_role(current_user, READ_ROLES)

    return await require_shelter_access(
        service,
        validate_uuid(shelter_id),
        current_user,
    )


# ============================================================
# UPDATE SHELTER
# ============================================================

@router.patch(
    "/{shelter_id}",
    response_model=ShelterResponse,
)
async def update_shelter(
    shelter_id: str,
    payload: ShelterUpdate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.update_shelter(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# CHANGE SHELTER STATUS
# ============================================================

@router.post(
    "/{shelter_id}/status",
    response_model=ShelterResponse,
)
async def change_shelter_status(
    shelter_id: str,
    payload: ShelterStatusUpdate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterResponse:
    require_role(current_user, APPROVAL_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    if payload.status == ShelterStatus.OPEN:
        if not (
            payload.safety_verified
            and payload.capacity_verified
        ):
            raise HTTPException(
                status_code=422,
                detail="Shelter opening checks incomplete",
            )

    return await service.change_status(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# ROOMS
# ============================================================

@router.get(
    "/{shelter_id}/rooms",
    response_model=list[ShelterRoomResponse],
)
async def list_shelter_rooms(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterRoomResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_rooms(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/rooms",
    response_model=ShelterRoomResponse,
    status_code=201,
)
async def create_shelter_room(
    shelter_id: str,
    payload: ShelterRoomCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterRoomResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.create_room(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# BEDS
# ============================================================

@router.get(
    "/{shelter_id}/beds",
    response_model=list[ShelterBedResponse],
)
async def list_shelter_beds(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterBedResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_beds(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/beds",
    response_model=ShelterBedResponse,
    status_code=201,
)
async def create_shelter_bed(
    shelter_id: str,
    payload: ShelterBedCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterBedResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.create_bed(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# FACILITIES
# ============================================================

@router.get(
    "/{shelter_id}/facilities",
    response_model=list[ShelterFacilityResponse],
)
async def list_shelter_facilities(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterFacilityResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_facilities(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/facilities",
    response_model=ShelterFacilityResponse,
    status_code=201,
)
async def create_shelter_facility(
    shelter_id: str,
    payload: ShelterFacilityCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterFacilityResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.create_facility(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# RESIDENTS
# ============================================================

@router.get(
    "/{shelter_id}/residents",
    response_model=list[ShelterResidentResponse],
)
async def list_shelter_residents(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterResidentResponse]:
    require_role(current_user, RESIDENT_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_residents(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/residents/check-in",
    response_model=ShelterResidentResponse,
    status_code=201,
)
async def shelter_resident_check_in(
    shelter_id: str,
    payload: ShelterResidentCheckIn,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterResidentResponse:
    require_role(current_user, RESIDENT_ROLES)

    identifier = validate_uuid(shelter_id)

    shelter = await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    if shelter.status not in {
        ShelterStatus.OPEN.value,
        ShelterStatus.READY.value,
    }:
        raise HTTPException(
            status_code=409,
            detail="Shelter is not accepting residents",
        )

    return await service.check_in_resident(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/residents/check-out",
    response_model=ShelterResidentResponse,
)
async def shelter_resident_check_out(
    shelter_id: str,
    payload: ShelterResidentCheckOut,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterResidentResponse:
    require_role(current_user, RESIDENT_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.check_out_resident(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/residents/transfer",
    response_model=ShelterResidentResponse,
)
async def transfer_shelter_resident(
    shelter_id: str,
    payload: ShelterTransferRequest,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterResidentResponse:
    require_role(current_user, APPROVAL_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    if not all(
        [
            payload.destination_capacity_verified,
            payload.transport_ready,
            payload.authorized_approval,
        ]
    ):
        raise HTTPException(
            status_code=422,
            detail="Resident transfer checks incomplete",
        )

    return await service.transfer_resident(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# OCCUPANCY
# ============================================================

@router.get(
    "/{shelter_id}/occupancy",
    response_model=list[ShelterOccupancyResponse],
)
async def list_shelter_occupancy(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterOccupancyResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_occupancy(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/occupancy",
    response_model=ShelterOccupancyResponse,
    status_code=201,
)
async def record_shelter_occupancy(
    shelter_id: str,
    payload: ShelterOccupancySnapshotCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterOccupancyResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(shelter_id)

    shelter = await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    if not payload.verified:
        raise HTTPException(
            status_code=422,
            detail="Occupancy snapshot must be verified",
        )

    if (
        payload.occupied
        + payload.available
        + payload.reserved
        > shelter.total_capacity
    ):
        raise HTTPException(
            status_code=422,
            detail="Occupancy exceeds shelter capacity",
        )

    return await service.record_occupancy(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# RESOURCE REQUESTS
# ============================================================

@router.get(
    "/{shelter_id}/resources",
    response_model=list[ShelterResourceResponse],
)
async def list_shelter_resources(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterResourceResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_resources(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/resources",
    response_model=ShelterResourceResponse,
    status_code=201,
)
async def request_shelter_resources(
    shelter_id: str,
    payload: ShelterResourceRequestCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterResourceResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.request_resources(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# SAFETY INSPECTIONS
# ============================================================

@router.get(
    "/{shelter_id}/inspections",
    response_model=list[ShelterInspectionResponse],
)
async def list_shelter_inspections(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterInspectionResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_inspections(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/inspections",
    response_model=ShelterInspectionResponse,
    status_code=201,
)
async def create_shelter_inspection(
    shelter_id: str,
    payload: ShelterInspectionCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterInspectionResponse:
    require_role(current_user, APPROVAL_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.create_inspection(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# SHELTER ALERTS
# ============================================================

@router.get(
    "/{shelter_id}/alerts",
    response_model=list[ShelterAlertResponse],
)
async def list_shelter_alerts(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterAlertResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_alerts(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{shelter_id}/alerts",
    response_model=ShelterAlertResponse,
    status_code=201,
)
async def create_shelter_alert(
    shelter_id: str,
    payload: ShelterAlertCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> ShelterAlertResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.create_alert(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# RESIDENT WELFARE
# ============================================================

@router.post(
    "/{shelter_id}/welfare-checks",
)
async def create_resident_welfare_check(
    shelter_id: str,
    payload: ShelterWelfareCheckCreate,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, RESIDENT_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.welfare_check(
        shelter_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get(
    "/{shelter_id}/events",
    response_model=list[ShelterEventResponse],
)
async def list_shelter_events(
    shelter_id: str,
    current_user: CurrentUser,
    service: ShelterServiceDependency,
) -> list[ShelterEventResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(shelter_id)

    await require_shelter_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_events(
        shelter_id=identifier,
        actor_id=actor_id(current_user),
    )


__all__ = [
    "router",
    "ShelterService",
    "ShelterCreate",
    "ShelterUpdate",
    "ShelterStatusUpdate",
    "ShelterRoomCreate",
    "ShelterBedCreate",
    "ShelterFacilityCreate",
    "ShelterResidentCheckIn",
    "ShelterResidentCheckOut",
    "ShelterTransferRequest",
    "ShelterOccupancySnapshotCreate",
    "ShelterResourceRequestCreate",
    "ShelterInspectionCreate",
    "ShelterAlertCreate",
    "ShelterWelfareCheckCreate",
    "ShelterResponse",
    "ShelterListResponse",
    "ShelterStatisticsResponse",
]
