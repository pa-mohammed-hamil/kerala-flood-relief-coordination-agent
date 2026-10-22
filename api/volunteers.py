from __future__ import annotations

"""
backend/api/volunteers.py

Kerala Flood Relief Coordination Agent
Volunteer Management & Coordination API

Python 3.11+ | FastAPI | Pydantic v2

Endpoints:
    GET    /volunteers/health
    GET    /volunteers
    GET    /volunteers/statistics
    POST   /volunteers
    GET    /volunteers/{volunteer_id}
    PATCH  /volunteers/{volunteer_id}
    POST   /volunteers/{volunteer_id}/verify
    GET    /volunteers/{volunteer_id}/skills
    POST   /volunteers/{volunteer_id}/skills
    GET    /volunteers/{volunteer_id}/training
    POST   /volunteers/{volunteer_id}/training
    GET    /volunteers/{volunteer_id}/availability
    POST   /volunteers/{volunteer_id}/availability
    GET    /volunteers/{volunteer_id}/assignments
    POST   /volunteers/{volunteer_id}/assignments
    GET    /volunteers/{volunteer_id}/shifts
    POST   /volunteers/{volunteer_id}/shifts
    GET    /volunteers/{volunteer_id}/locations
    POST   /volunteers/{volunteer_id}/locations
    GET    /volunteers/{volunteer_id}/equipment
    POST   /volunteers/{volunteer_id}/equipment
    GET    /volunteers/{volunteer_id}/welfare-checks
    POST   /volunteers/{volunteer_id}/welfare-checks
    GET    /volunteers/{volunteer_id}/events

Safety:
    - Volunteers cannot self-approve verification.
    - Rescue assignments require verified training.
    - Assignments require human authorization.
    - Volunteer location history is restricted.
    - Availability is not equivalent to deployment readiness.
    - All operational changes require audit logging.
"""


from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Protocol
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
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
    prefix="/volunteers",
    tags=["Volunteer Management"],
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

COORDINATOR_ROLES = {
    "volunteer_coordinator",
    "district_officer",
    "emergency_coordinator",
    "incident_commander",
}

FIELD_ROLES = {
    "field_officer",
    "rescue_coordinator",
    "shelter_coordinator",
    "relief_coordinator",
}

READ_ROLES = (
    ADMIN_ROLES
    | COORDINATOR_ROLES
    | FIELD_ROLES
    | {"auditor"}
)

MANAGE_ROLES = (
    ADMIN_ROLES
    | COORDINATOR_ROLES
)

ASSIGNMENT_ROLES = MANAGE_ROLES | {
    "rescue_coordinator",
    "shelter_coordinator",
    "relief_coordinator",
}

WELFARE_ROLES = MANAGE_ROLES | {
    "field_officer",
    "medical_officer",
}

LOCATION_ROLES = MANAGE_ROLES | {
    "field_officer",
    "rescue_coordinator",
}

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class VolunteerStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"


class VolunteerSkillType(str, Enum):
    FIRST_AID = "first_aid"
    CPR = "cpr"
    WATER_RESCUE = "water_rescue"
    BOAT_OPERATION = "boat_operation"
    SEARCH_AND_RESCUE = "search_and_rescue"
    MEDICAL_SUPPORT = "medical_support"
    SHELTER_MANAGEMENT = "shelter_management"
    FOOD_DISTRIBUTION = "food_distribution"
    LOGISTICS = "logistics"
    DRIVING = "driving"
    COMMUNICATION = "communication"
    TRANSLATION = "translation"
    DATA_ENTRY = "data_entry"
    OTHER = "other"


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    CERTIFIED = "certified"


class AvailabilityStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LIMITED = "limited"


class AssignmentType(str, Enum):
    RESCUE_SUPPORT = "rescue_support"
    SHELTER_SUPPORT = "shelter_support"
    RELIEF_DISTRIBUTION = "relief_distribution"
    LOGISTICS = "logistics"
    MEDICAL_SUPPORT = "medical_support"
    COMMUNICATION = "communication"
    DATA_COLLECTION = "data_collection"
    COMMUNITY_OUTREACH = "community_outreach"


class AssignmentStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ShiftStatus(str, Enum):
    SCHEDULED = "scheduled"
    CHECKED_IN = "checked_in"
    CHECKED_OUT = "checked_out"
    CANCELLED = "cancelled"


class WelfareStatus(str, Enum):
    SAFE = "safe"
    NEEDS_ASSISTANCE = "needs_assistance"
    UNREACHABLE = "unreachable"
    INJURED = "injured"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class VolunteerCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    district_id: int = Field(ge=1)

    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    contact_reference: str = Field(
        min_length=5,
        max_length=150,
    )

    organization_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    preferred_language: str | None = Field(
        default=None,
        max_length=50,
    )

    emergency_contact_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    consent_recorded: bool = False


class VolunteerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    preferred_language: str | None = Field(
        default=None,
        max_length=50,
    )

    organization_reference: str | None = Field(
        default=None,
        max_length=150,
    )


class VolunteerVerificationRequest(BaseModel):
    approved: bool

    verification_notes: str = Field(
        min_length=10,
        max_length=3000,
    )

    identity_verified: bool = False
    eligibility_verified: bool = False
    safeguarding_verified: bool = False


class VolunteerSkillCreate(BaseModel):
    skill_type: VolunteerSkillType

    level: SkillLevel

    certificate_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    verified: bool = False


class VolunteerTrainingCreate(BaseModel):
    training_name: str = Field(
        min_length=3,
        max_length=255,
    )

    provider_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    completed_at: datetime

    expires_at: datetime | None = None

    certificate_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_training_dates(
        self,
    ) -> "VolunteerTrainingCreate":
        if (
            self.expires_at is not None
            and self.expires_at <= self.completed_at
        ):
            raise ValueError(
                "Training expiry must be after completion"
            )
        return self


class VolunteerAvailabilityCreate(BaseModel):
    status: AvailabilityStatus

    available_from: datetime
    available_until: datetime

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )

    @model_validator(mode="after")
    def validate_period(
        self,
    ) -> "VolunteerAvailabilityCreate":
        if self.available_until <= self.available_from:
            raise ValueError(
                "Availability end must be after start"
            )
        return self


class VolunteerAssignmentCreate(BaseModel):
    assignment_type: AssignmentType

    district_id: int = Field(ge=1)

    mission_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    location_reference: str = Field(
        min_length=3,
        max_length=500,
    )

    instructions: str = Field(
        min_length=10,
        max_length=3000,
    )

    starts_at: datetime

    ends_at: datetime | None = None

    training_verified: bool = False
    availability_verified: bool = False
    safety_briefing_completed: bool = False
    supervisor_approved: bool = False

    @model_validator(mode="after")
    def validate_assignment_dates(
        self,
    ) -> "VolunteerAssignmentCreate":
        if (
            self.ends_at is not None
            and self.ends_at <= self.starts_at
        ):
            raise ValueError(
                "Assignment end must be after start"
            )
        return self


class VolunteerShiftCreate(BaseModel):
    assignment_id: UUID

    starts_at: datetime
    ends_at: datetime

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )

    @model_validator(mode="after")
    def validate_shift(
        self,
    ) -> "VolunteerShiftCreate":
        if self.ends_at <= self.starts_at:
            raise ValueError(
                "Shift end must be after start"
            )
        return self


class VolunteerLocationCreate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    observed_at: datetime

    accuracy_m: float | None = Field(
        default=None,
        ge=0,
        le=100000,
    )

    consent_verified: bool = False

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "Timezone-aware timestamp required"
            )
        if value > datetime.now(timezone.utc):
            raise ValueError(
                "Location timestamp cannot be in the future"
            )
        return value


class VolunteerEquipmentCreate(BaseModel):
    equipment_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    quantity: int = Field(ge=1)

    issued_at: datetime

    inspection_verified: bool = False

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )


class VolunteerWelfareCheckCreate(BaseModel):
    checked_at: datetime

    status: WelfareStatus

    notes: str = Field(
        min_length=5,
        max_length=3000,
    )

    immediate_assistance_required: bool = False


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class VolunteerResponse(BaseModel):
    id: UUID
    district_id: int
    full_name: str
    status: VolunteerStatus
    created_at: datetime
    updated_at: datetime | None = None


class VolunteerListResponse(BaseModel):
    items: list[VolunteerResponse]
    total: int
    limit: int
    offset: int


class VolunteerStatisticsResponse(BaseModel):
    total_volunteers: int
    by_status: dict[str, int]
    active_assignments: int
    generated_at: datetime


class VolunteerSkillResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    skill_type: VolunteerSkillType
    level: SkillLevel
    verified: bool


class VolunteerTrainingResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    training_name: str
    completed_at: datetime


class VolunteerAvailabilityResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    status: AvailabilityStatus
    available_from: datetime
    available_until: datetime


class VolunteerAssignmentResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    assignment_type: AssignmentType
    status: AssignmentStatus
    created_at: datetime


class VolunteerShiftResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    assignment_id: UUID
    status: ShiftStatus


class VolunteerLocationResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    latitude: float
    longitude: float
    observed_at: datetime


class VolunteerEquipmentResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    equipment_reference: str
    quantity: int


class VolunteerWelfareResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    status: WelfareStatus
    checked_at: datetime


class VolunteerEventResponse(BaseModel):
    id: UUID
    volunteer_id: UUID
    event_type: str
    created_at: datetime


class VolunteerHealthResponse(BaseModel):
    service: str
    status: str
    timestamp: datetime


# ============================================================
# SERVICE CONTRACT
# ============================================================

class VolunteerService(Protocol):
    """
    Implement in backend/services/volunteer_service.py.

    Responsibilities:
      - Volunteer identity verification
      - District and ownership authorization
      - Certification validation
      - Assignment suitability and safety checks
      - Consent-based location collection
      - Shift and availability conflict detection
      - Equipment inventory reconciliation
      - Welfare escalation
      - Immutable audit logging
    """

    async def health(self) -> VolunteerHealthResponse: ...

    async def list_volunteers(
        self,
        actor_id: str,
        district_id: int | None,
        limit: int,
        offset: int,
    ) -> VolunteerListResponse: ...

    async def statistics(
        self,
        actor_id: str,
        district_id: int | None,
    ) -> VolunteerStatisticsResponse: ...

    async def create_volunteer(
        self,
        payload: VolunteerCreate,
        actor_id: str,
    ) -> VolunteerResponse: ...

    async def get_volunteer(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> VolunteerResponse | None: ...

    async def update_volunteer(
        self,
        volunteer_id: str,
        payload: VolunteerUpdate,
        actor_id: str,
    ) -> VolunteerResponse: ...

    async def verify_volunteer(
        self,
        volunteer_id: str,
        payload: VolunteerVerificationRequest,
        actor_id: str,
    ) -> VolunteerResponse: ...

    async def list_skills(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerSkillResponse]: ...

    async def add_skill(
        self,
        volunteer_id: str,
        payload: VolunteerSkillCreate,
        actor_id: str,
    ) -> VolunteerSkillResponse: ...

    async def list_training(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerTrainingResponse]: ...

    async def add_training(
        self,
        volunteer_id: str,
        payload: VolunteerTrainingCreate,
        actor_id: str,
    ) -> VolunteerTrainingResponse: ...

    async def list_availability(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerAvailabilityResponse]: ...

    async def add_availability(
        self,
        volunteer_id: str,
        payload: VolunteerAvailabilityCreate,
        actor_id: str,
    ) -> VolunteerAvailabilityResponse: ...

    async def list_assignments(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerAssignmentResponse]: ...

    async def add_assignment(
        self,
        volunteer_id: str,
        payload: VolunteerAssignmentCreate,
        actor_id: str,
    ) -> VolunteerAssignmentResponse: ...

    async def list_shifts(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerShiftResponse]: ...

    async def add_shift(
        self,
        volunteer_id: str,
        payload: VolunteerShiftCreate,
        actor_id: str,
    ) -> VolunteerShiftResponse: ...

    async def list_locations(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerLocationResponse]: ...

    async def add_location(
        self,
        volunteer_id: str,
        payload: VolunteerLocationCreate,
        actor_id: str,
    ) -> VolunteerLocationResponse: ...

    async def list_equipment(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerEquipmentResponse]: ...

    async def add_equipment(
        self,
        volunteer_id: str,
        payload: VolunteerEquipmentCreate,
        actor_id: str,
    ) -> VolunteerEquipmentResponse: ...

    async def list_welfare_checks(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerWelfareResponse]: ...

    async def add_welfare_check(
        self,
        volunteer_id: str,
        payload: VolunteerWelfareCheckCreate,
        actor_id: str,
    ) -> VolunteerWelfareResponse: ...

    async def list_events(
        self,
        volunteer_id: str,
        actor_id: str,
    ) -> list[VolunteerEventResponse]: ...


# ============================================================
# DEPENDENCIES
# ============================================================

def get_volunteer_service(
    request: Request,
) -> VolunteerService:
    service = getattr(
        request.app.state,
        "volunteer_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Volunteer service is not configured",
        )

    return service


VolunteerServiceDependency = Annotated[
    VolunteerService,
    Depends(get_volunteer_service),
]


# ============================================================
# SECURITY HELPERS
# ============================================================

def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return str(identifier)


def role_of(user: Any) -> str:
    role = getattr(user, "role", "")

    if isinstance(role, Enum):
        role = role.value

    return str(role).lower()


def require_role(
    user: Any,
    allowed: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) not in allowed:
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


async def require_volunteer_access(
    service: VolunteerService,
    volunteer_id: UUID,
    user: Any,
) -> VolunteerResponse:
    volunteer = await service.get_volunteer(
        volunteer_id=str(volunteer_id),
        actor_id=actor_id(user),
    )

    if volunteer is None:
        raise HTTPException(
            status_code=404,
            detail="Volunteer not found",
        )

    require_district_access(
        user,
        volunteer.district_id,
    )

    return volunteer


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=VolunteerHealthResponse,
)
async def volunteer_health(
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerHealthResponse:
    require_role(current_user, READ_ROLES)
    return await service.health()


# ============================================================
# LIST VOLUNTEERS
# ============================================================

@router.get(
    "",
    response_model=VolunteerListResponse,
)
async def list_volunteers(
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
    district_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> VolunteerListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.list_volunteers(
        actor_id=actor_id(current_user),
        district_id=district_id,
        limit=limit,
        offset=offset,
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=VolunteerStatisticsResponse,
)
async def volunteer_statistics(
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
    district_id: int | None = Query(default=None, ge=1),
) -> VolunteerStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.statistics(
        actor_id=actor_id(current_user),
        district_id=district_id,
    )


# ============================================================
# CREATE VOLUNTEER
# ============================================================

@router.post(
    "",
    response_model=VolunteerResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_volunteer(
    payload: VolunteerCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerResponse:
    require_role(current_user, MANAGE_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    if not payload.consent_recorded:
        raise HTTPException(
            status_code=422,
            detail="Volunteer consent is required",
        )

    return await service.create_volunteer(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# VOLUNTEER DETAILS
# ============================================================

@router.get(
    "/{volunteer_id}",
    response_model=VolunteerResponse,
)
async def get_volunteer(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerResponse:
    require_role(current_user, READ_ROLES)

    return await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )


# ============================================================
# UPDATE VOLUNTEER
# ============================================================

@router.patch(
    "/{volunteer_id}",
    response_model=VolunteerResponse,
)
async def update_volunteer(
    volunteer_id: UUID,
    payload: VolunteerUpdate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.update_volunteer(
        volunteer_id=str(volunteer_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# VERIFY VOLUNTEER
# ============================================================

@router.post(
    "/{volunteer_id}/verify",
    response_model=VolunteerResponse,
)
async def verify_volunteer(
    volunteer_id: UUID,
    payload: VolunteerVerificationRequest,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    if payload.approved and not all([
        payload.identity_verified,
        payload.eligibility_verified,
        payload.safeguarding_verified,
    ]):
        raise HTTPException(
            status_code=422,
            detail="Volunteer verification incomplete",
        )

    return await service.verify_volunteer(
        volunteer_id=str(volunteer_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# SKILLS
# ============================================================

@router.get(
    "/{volunteer_id}/skills",
    response_model=list[VolunteerSkillResponse],
)
async def list_volunteer_skills(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerSkillResponse]:
    require_role(current_user, READ_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_skills(
        str(volunteer_id),
        actor_id(current_user),
    )


@router.post(
    "/{volunteer_id}/skills",
    response_model=VolunteerSkillResponse,
    status_code=201,
)
async def add_volunteer_skill(
    volunteer_id: UUID,
    payload: VolunteerSkillCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerSkillResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.add_skill(
        str(volunteer_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# TRAINING
# ============================================================

@router.get(
    "/{volunteer_id}/training",
    response_model=list[VolunteerTrainingResponse],
)
async def list_volunteer_training(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerTrainingResponse]:
    require_role(current_user, READ_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_training(
        str(volunteer_id),
        actor_id(current_user),
    )


@router.post(
    "/{volunteer_id}/training",
    response_model=VolunteerTrainingResponse,
    status_code=201,
)
async def add_volunteer_training(
    volunteer_id: UUID,
    payload: VolunteerTrainingCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerTrainingResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.add_training(
        str(volunteer_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# AVAILABILITY
# ============================================================

@router.get(
    "/{volunteer_id}/availability",
    response_model=list[VolunteerAvailabilityResponse],
)
async def list_volunteer_availability(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerAvailabilityResponse]:
    require_role(current_user, READ_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_availability(
        str(volunteer_id),
        actor_id(current_user),
    )


@router.post(
    "/{volunteer_id}/availability",
    response_model=VolunteerAvailabilityResponse,
    status_code=201,
)
async def add_volunteer_availability(
    volunteer_id: UUID,
    payload: VolunteerAvailabilityCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerAvailabilityResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.add_availability(
        str(volunteer_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# ASSIGNMENTS
# ============================================================

@router.get(
    "/{volunteer_id}/assignments",
    response_model=list[VolunteerAssignmentResponse],
)
async def list_volunteer_assignments(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerAssignmentResponse]:
    require_role(current_user, READ_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_assignments(
        str(volunteer_id),
        actor_id(current_user),
    )


@router.post(
    "/{volunteer_id}/assignments",
    response_model=VolunteerAssignmentResponse,
    status_code=201,
)
async def assign_volunteer(
    volunteer_id: UUID,
    payload: VolunteerAssignmentCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerAssignmentResponse:
    require_role(current_user, ASSIGNMENT_ROLES)

    volunteer = await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    require_district_access(
        current_user,
        payload.district_id,
    )

    if volunteer.status not in {
        VolunteerStatus.VERIFIED,
        VolunteerStatus.ACTIVE,
    }:
        raise HTTPException(
            status_code=409,
            detail="Volunteer is not verified or active",
        )

    if not all([
        payload.training_verified,
        payload.availability_verified,
        payload.safety_briefing_completed,
        payload.supervisor_approved,
    ]):
        raise HTTPException(
            status_code=422,
            detail="Assignment readiness checks incomplete",
        )

    return await service.add_assignment(
        str(volunteer_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# SHIFTS
# ============================================================

@router.get(
    "/{volunteer_id}/shifts",
    response_model=list[VolunteerShiftResponse],
)
async def list_volunteer_shifts(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerShiftResponse]:
    require_role(current_user, READ_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_shifts(
        str(volunteer_id),
        actor_id(current_user),
    )


@router.post(
    "/{volunteer_id}/shifts",
    response_model=VolunteerShiftResponse,
    status_code=201,
)
async def add_volunteer_shift(
    volunteer_id: UUID,
    payload: VolunteerShiftCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerShiftResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.add_shift(
        str(volunteer_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# LOCATION TRACKING
# ============================================================

@router.get(
    "/{volunteer_id}/locations",
    response_model=list[VolunteerLocationResponse],
)
async def list_volunteer_locations(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerLocationResponse]:
    require_role(current_user, LOCATION_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_locations(
        str(volunteer_id),
        actor_id(current_user),
    )


@router.post(
    "/{volunteer_id}/locations",
    response_model=VolunteerLocationResponse,
    status_code=201,
)
async def add_volunteer_location(
    volunteer_id: UUID,
    payload: VolunteerLocationCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerLocationResponse:
    require_role(current_user, LOCATION_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    if not payload.consent_verified:
        raise HTTPException(
            status_code=422,
            detail="Location consent required",
        )

    return await service.add_location(
        str(volunteer_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# EQUIPMENT
# ============================================================

@router.get(
    "/{volunteer_id}/equipment",
    response_model=list[VolunteerEquipmentResponse],
)
async def list_volunteer_equipment(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerEquipmentResponse]:
    require_role(current_user, READ_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_equipment(
        str(volunteer_id),
        actor_id(current_user),
    )


@router.post(
    "/{volunteer_id}/equipment",
    response_model=VolunteerEquipmentResponse,
    status_code=201,
)
async def add_volunteer_equipment(
    volunteer_id: UUID,
    payload: VolunteerEquipmentCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerEquipmentResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    if not payload.inspection_verified:
        raise HTTPException(
            status_code=422,
            detail="Equipment inspection required",
        )

    return await service.add_equipment(
        str(volunteer_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# WELFARE CHECKS
# ============================================================

@router.get(
    "/{volunteer_id}/welfare-checks",
    response_model=list[VolunteerWelfareResponse],
)
async def list_volunteer_welfare_checks(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerWelfareResponse]:
    require_role(current_user, WELFARE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_welfare_checks(
        str(volunteer_id),
        actor_id(current_user),
    )


@router.post(
    "/{volunteer_id}/welfare-checks",
    response_model=VolunteerWelfareResponse,
    status_code=201,
)
async def add_volunteer_welfare_check(
    volunteer_id: UUID,
    payload: VolunteerWelfareCheckCreate,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> VolunteerWelfareResponse:
    require_role(current_user, WELFARE_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.add_welfare_check(
        str(volunteer_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get(
    "/{volunteer_id}/events",
    response_model=list[VolunteerEventResponse],
)
async def list_volunteer_events(
    volunteer_id: UUID,
    current_user: CurrentUser,
    service: VolunteerServiceDependency,
) -> list[VolunteerEventResponse]:
    require_role(current_user, READ_ROLES)

    await require_volunteer_access(
        service,
        volunteer_id,
        current_user,
    )

    return await service.list_events(
        str(volunteer_id),
        actor_id(current_user),
    )


__all__ = [
    "router",
    "VolunteerService",
    "VolunteerCreate",
    "VolunteerUpdate",
    "VolunteerVerificationRequest",
    "VolunteerSkillCreate",
    "VolunteerTrainingCreate",
    "VolunteerAvailabilityCreate",
    "VolunteerAssignmentCreate",
    "VolunteerShiftCreate",
    "VolunteerLocationCreate",
    "VolunteerEquipmentCreate",
    "VolunteerWelfareCheckCreate",
    "VolunteerResponse",
    "VolunteerListResponse",
    "VolunteerStatisticsResponse",
]
