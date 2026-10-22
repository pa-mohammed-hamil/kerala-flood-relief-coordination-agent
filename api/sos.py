
"""
backend/api/sos.py

Emergency SOS Coordination API
Kerala Flood Relief Coordination Agent

Requirements:
    Python 3.11+
    FastAPI
    Pydantic v2

The service implementation must provide:
    - Durable SOS persistence
    - Reporter ownership and district authorization
    - Duplicate request detection
    - Trusted location updates
    - Human-verified rescue assignments
    - Escalation and notification delivery tracking
    - Audit events and idempotent state transitions

No request is represented as dispatched, rescued, or
notified until the corresponding service confirms it.
"""

from __future__ import annotations

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


router = APIRouter(
    prefix="/sos",
    tags=["Emergency SOS"],
)

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# PERMISSIONS
# ============================================================

ADMIN_ROLES = {"admin", "super_admin"}

COORDINATOR_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "incident_commander",
    "sos_coordinator",
    "rescue_coordinator",
}

RESPONDER_ROLES = {
    "rescue_team_leader",
    "rescue_operator",
    "field_officer",
}

READ_ROLES = (
    ADMIN_ROLES
    | COORDINATOR_ROLES
    | RESPONDER_ROLES
    | {"auditor"}
)

MANAGE_ROLES = ADMIN_ROLES | COORDINATOR_ROLES
ASSIGN_ROLES = MANAGE_ROLES
ESCALATE_ROLES = MANAGE_ROLES


# ============================================================
# ENUMERATIONS
# ============================================================

class SOSCategory(str, Enum):
    FLOOD_TRAPPED = "flood_trapped"
    WATER_RESCUE = "water_rescue"
    MEDICAL_EMERGENCY = "medical_emergency"
    LANDSLIDE = "landslide"
    BUILDING_COLLAPSE = "building_collapse"
    STRANDED = "stranded"
    MISSING_PERSON = "missing_person"
    EVACUATION_ASSISTANCE = "evacuation_assistance"
    OTHER = "other"


class SOSPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    LIFE_THREATENING = "life_threatening"


class SOSStatus(str, Enum):
    REPORTED = "reported"
    TRIAGED = "triaged"
    VERIFIED = "verified"
    ASSIGNED = "assigned"
    RESCUE_IN_PROGRESS = "rescue_in_progress"
    RESCUED = "rescued"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    FALSE_ALARM = "false_alarm"


class SOSLocationSource(str, Enum):
    GPS = "gps"
    NETWORK = "network"
    MANUAL = "manual"
    RESPONDER = "responder"


class SOSEscalationReason(str, Enum):
    LIFE_THREATENING = "life_threatening"
    NO_RESPONSE = "no_response"
    DELAYED_RESCUE = "delayed_rescue"
    CONDITION_WORSENING = "condition_worsening"
    RESOURCE_SHORTAGE = "resource_shortage"
    OTHER = "other"


class SOSEvidenceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"


class SOSResolutionType(str, Enum):
    RESCUED = "rescued"
    SELF_EVACUATED = "self_evacuated"
    REFERRED = "referred"
    DUPLICATE = "duplicate"
    FALSE_ALARM = "false_alarm"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class SOSCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    district_id: int = Field(ge=1)

    category: SOSCategory

    description: str = Field(
        min_length=10,
        max_length=5000,
    )

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    location_source: SOSLocationSource

    location_accuracy_m: float | None = Field(
        default=None,
        ge=0,
        le=100000,
    )

    location_description: str | None = Field(
        default=None,
        max_length=500,
    )

    people_at_risk: int = Field(
        default=1,
        ge=1,
        le=10000,
    )

    children_present: bool = False
    elderly_present: bool = False
    medical_assistance_required: bool = False

    reported_priority: SOSPriority = SOSPriority.HIGH

    contact_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    idempotency_key: str = Field(
        min_length=8,
        max_length=128,
    )


class SOSUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(
        default=None,
        min_length=10,
        max_length=5000,
    )

    people_at_risk: int | None = Field(
        default=None,
        ge=1,
        le=10000,
    )

    medical_assistance_required: bool | None = None


class SOSLocationUpdateCreate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    source: SOSLocationSource

    accuracy_m: float | None = Field(
        default=None,
        ge=0,
        le=100000,
    )

    observed_at: datetime

    notes: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator("observed_at")
    @classmethod
    def check_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "Timezone-aware timestamp required"
            )

        if value > datetime.now(timezone.utc):
            raise ValueError(
                "Observation cannot be in the future"
            )

        return value


class SOSTriageRequest(BaseModel):
    priority: SOSPriority

    notes: str = Field(
        min_length=10,
        max_length=3000,
    )

    location_verified: bool = False
    emergency_details_verified: bool = False


class SOSAssignmentCreate(BaseModel):
    rescue_team_id: UUID

    mission_id: UUID | None = None

    assignment_notes: str = Field(
        min_length=10,
        max_length=3000,
    )

    team_availability_verified: bool = False
    team_capability_verified: bool = False
    route_safety_verified: bool = False
    team_leader_acknowledged: bool = False


class SOSEvidenceCreate(BaseModel):
    evidence_type: SOSEvidenceType

    evidence_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    captured_at: datetime | None = None

    source_verified: bool = False


class SOSEscalationCreate(BaseModel):
    reason: SOSEscalationReason

    notes: str = Field(
        min_length=10,
        max_length=3000,
    )

    escalation_target_reference: str = Field(
        min_length=3,
        max_length=150,
    )


class SOSStatusUpdate(BaseModel):
    status: SOSStatus

    reason: str = Field(
        min_length=10,
        max_length=3000,
    )

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )


class SOSResolutionRequest(BaseModel):
    resolution: SOSResolutionType

    notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    people_rescued: int = Field(
        default=0,
        ge=0,
    )

    responder_confirmation: bool = False
    evidence_verified: bool = False


class SOSCancellationRequest(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=3000,
    )

    active_team_notified: bool = False


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class SOSResponse(BaseModel):
    id: UUID
    district_id: int
    category: SOSCategory
    status: SOSStatus
    priority: SOSPriority
    people_at_risk: int
    created_at: datetime
    updated_at: datetime | None = None


class SOSListResponse(BaseModel):
    items: list[SOSResponse]
    total: int
    limit: int
    offset: int


class SOSLocationResponse(BaseModel):
    id: UUID
    sos_request_id: UUID
    latitude: float
    longitude: float
    observed_at: datetime


class SOSAssignmentResponse(BaseModel):
    id: UUID
    sos_request_id: UUID
    rescue_team_id: UUID
    status: str
    created_at: datetime


class SOSEvidenceResponse(BaseModel):
    id: UUID
    sos_request_id: UUID
    evidence_type: SOSEvidenceType
    created_at: datetime


class SOSEscalationResponse(BaseModel):
    id: UUID
    sos_request_id: UUID
    reason: SOSEscalationReason
    status: str
    created_at: datetime


class SOSNotificationResponse(BaseModel):
    id: UUID
    sos_request_id: UUID
    channel: str
    delivery_status: str
    created_at: datetime


class SOSEventResponse(BaseModel):
    id: UUID
    sos_request_id: UUID
    event_type: str
    created_at: datetime


class SOSStatisticsResponse(BaseModel):
    total_requests: int
    by_status: dict[str, int]
    by_priority: dict[str, int]
    active_requests: int
    generated_at: datetime


class SOSHealthResponse(BaseModel):
    service: str
    status: str
    timestamp: datetime


# ============================================================
# SERVICE INTERFACE
# ============================================================

class SOSService(Protocol):
    """
    Implement in backend/services/sos_service.py.

    Requirements:
        - Transactional persistence
        - Verified caller identity or safe guest intake
        - Ownership checks
        - District access enforcement
        - Request deduplication
        - Authorized status transitions
        - Audit logs
        - Trusted responder assignments
        - Real notification delivery receipts
    """

    async def health(self) -> SOSHealthResponse: ...

    async def create_request(
        self,
        payload: SOSCreate,
        actor_id: str,
    ) -> SOSResponse: ...

    async def list_requests(
        self,
        actor_id: str,
        district_id: int | None,
        limit: int,
        offset: int,
        status_filter: SOSStatus | None,
    ) -> SOSListResponse: ...

    async def get_statistics(
        self,
        actor_id: str,
        district_id: int | None,
    ) -> SOSStatisticsResponse: ...

    async def get_request(
        self,
        sos_id: str,
        actor_id: str,
    ) -> SOSResponse | None: ...

    async def update_request(
        self,
        sos_id: str,
        payload: SOSUpdate,
        actor_id: str,
    ) -> SOSResponse: ...

    async def add_location(
        self,
        sos_id: str,
        payload: SOSLocationUpdateCreate,
        actor_id: str,
    ) -> SOSLocationResponse: ...

    async def list_locations(
        self,
        sos_id: str,
        actor_id: str,
    ) -> list[SOSLocationResponse]: ...

    async def triage(
        self,
        sos_id: str,
        payload: SOSTriageRequest,
        actor_id: str,
    ) -> SOSResponse: ...

    async def add_assignment(
        self,
        sos_id: str,
        payload: SOSAssignmentCreate,
        actor_id: str,
    ) -> SOSAssignmentResponse: ...

    async def list_assignments(
        self,
        sos_id: str,
        actor_id: str,
    ) -> list[SOSAssignmentResponse]: ...

    async def add_evidence(
        self,
        sos_id: str,
        payload: SOSEvidenceCreate,
        actor_id: str,
    ) -> SOSEvidenceResponse: ...

    async def list_evidence(
        self,
        sos_id: str,
        actor_id: str,
    ) -> list[SOSEvidenceResponse]: ...

    async def escalate(
        self,
        sos_id: str,
        payload: SOSEscalationCreate,
        actor_id: str,
    ) -> SOSEscalationResponse: ...

    async def list_escalations(
        self,
        sos_id: str,
        actor_id: str,
    ) -> list[SOSEscalationResponse]: ...

    async def update_status(
        self,
        sos_id: str,
        payload: SOSStatusUpdate,
        actor_id: str,
    ) -> SOSResponse: ...

    async def resolve(
        self,
        sos_id: str,
        payload: SOSResolutionRequest,
        actor_id: str,
    ) -> SOSResponse: ...

    async def cancel(
        self,
        sos_id: str,
        payload: SOSCancellationRequest,
        actor_id: str,
    ) -> SOSResponse: ...

    async def list_notifications(
        self,
        sos_id: str,
        actor_id: str,
    ) -> list[SOSNotificationResponse]: ...

    async def list_events(
        self,
        sos_id: str,
        actor_id: str,
    ) -> list[SOSEventResponse]: ...


def get_sos_service(request: Request) -> SOSService:
    service = getattr(
        request.app.state,
        "sos_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="SOS service is not configured",
        )

    return service


SOSServiceDependency = Annotated[
    SOSService,
    Depends(get_sos_service),
]


# ============================================================
# AUTHORIZATION HELPERS
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
    value = getattr(user, "role", "")

    if isinstance(value, Enum):
        value = value.value

    return str(value).lower()


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

    assigned = getattr(user, "district_id", None)

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

    assigned = getattr(user, "district_id", None)

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


def normalized_id(value: UUID) -> str:
    return str(value)


async def require_sos_access(
    service: SOSService,
    sos_id: UUID,
    user: Any,
    *,
    operational: bool = False,
) -> SOSResponse:
    sos = await service.get_request(
        sos_id=normalized_id(sos_id),
        actor_id=actor_id(user),
    )

    if sos is None:
        raise HTTPException(
            status_code=404,
            detail="SOS request not found",
        )

    if operational:
        require_role(user, READ_ROLES)
        require_district_access(user, sos.district_id)

    # The service must verify ownership for ordinary
    # callers before returning any private SOS data.
    return sos


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=SOSHealthResponse,
)
async def sos_health(
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSHealthResponse:
    require_role(current_user, READ_ROLES)
    return await service.health()


# ============================================================
# CREATE SOS
# ============================================================

@router.post(
    "",
    response_model=SOSResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_sos(
    payload: SOSCreate,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSResponse:
    """
    Create a persisted emergency request.

    Authentication is required in this version.
    A separate verified guest intake workflow can
    be added for unauthenticated emergency callers.

    Service must validate district and location.
    """
    return await service.create_request(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# LIST SOS REQUESTS
# ============================================================

@router.get(
    "",
    response_model=SOSListResponse,
)
async def list_sos_requests(
    current_user: CurrentUser,
    service: SOSServiceDependency,
    district_id: int | None = Query(default=None, ge=1),
    status_filter: SOSStatus | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SOSListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.list_requests(
        actor_id=actor_id(current_user),
        district_id=district_id,
        limit=limit,
        offset=offset,
        status_filter=status_filter,
    )


# ============================================================
# SOS STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=SOSStatisticsResponse,
)
async def sos_statistics(
    current_user: CurrentUser,
    service: SOSServiceDependency,
    district_id: int | None = Query(default=None, ge=1),
) -> SOSStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.get_statistics(
        actor_id=actor_id(current_user),
        district_id=district_id,
    )


# ============================================================
# SOS DETAILS
# ============================================================

@router.get(
    "/{sos_id}",
    response_model=SOSResponse,
)
async def get_sos(
    sos_id: UUID,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSResponse:
    return await require_sos_access(
        service,
        sos_id,
        current_user,
    )


# ============================================================
# UPDATE SOS
# ============================================================

@router.patch(
    "/{sos_id}",
    response_model=SOSResponse,
)
async def update_sos(
    sos_id: UUID,
    payload: SOSUpdate,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSResponse:
    sos = await require_sos_access(
        service,
        sos_id,
        current_user,
    )

    if sos.status in {
        SOSStatus.RESOLVED,
        SOSStatus.CANCELLED,
        SOSStatus.FALSE_ALARM,
    }:
        raise HTTPException(
            status_code=409,
            detail="Finalized SOS cannot be edited",
        )

    return await service.update_request(
        sos_id=normalized_id(sos_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# LOCATION UPDATES
# ============================================================

@router.get(
    "/{sos_id}/locations",
    response_model=list[SOSLocationResponse],
)
async def list_sos_locations(
    sos_id: UUID,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> list[SOSLocationResponse]:
    await require_sos_access(
        service,
        sos_id,
        current_user,
    )

    return await service.list_locations(
        normalized_id(sos_id),
        actor_id(current_user),
    )


@router.post(
    "/{sos_id}/locations",
    response_model=SOSLocationResponse,
    status_code=201,
)
async def update_sos_location(
    sos_id: UUID,
    payload: SOSLocationUpdateCreate,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSLocationResponse:
    await require_sos_access(
        service,
        sos_id,
        current_user,
    )

    return await service.add_location(
        normalized_id(sos_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# TRIAGE
# ============================================================

@router.post(
    "/{sos_id}/triage",
    response_model=SOSResponse,
)
async def triage_sos(
    sos_id: UUID,
    payload: SOSTriageRequest,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    if not (
        payload.location_verified
        and payload.emergency_details_verified
    ):
        raise HTTPException(
            status_code=422,
            detail="Triage verification checks incomplete",
        )

    return await service.triage(
        normalized_id(sos_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# RESCUE ASSIGNMENTS
# ============================================================

@router.get(
    "/{sos_id}/assignments",
    response_model=list[SOSAssignmentResponse],
)
async def list_sos_assignments(
    sos_id: UUID,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> list[SOSAssignmentResponse]:
    await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    return await service.list_assignments(
        normalized_id(sos_id),
        actor_id(current_user),
    )


@router.post(
    "/{sos_id}/assignments",
    response_model=SOSAssignmentResponse,
    status_code=201,
)
async def assign_sos_rescue_team(
    sos_id: UUID,
    payload: SOSAssignmentCreate,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSAssignmentResponse:
    require_role(current_user, ASSIGN_ROLES)

    sos = await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    if sos.status not in {
        SOSStatus.TRIAGED,
        SOSStatus.VERIFIED,
    }:
        raise HTTPException(
            status_code=409,
            detail="SOS is not ready for assignment",
        )

    if not all([
        payload.team_availability_verified,
        payload.team_capability_verified,
        payload.route_safety_verified,
        payload.team_leader_acknowledged,
    ]):
        raise HTTPException(
            status_code=422,
            detail="Rescue assignment checks incomplete",
        )

    return await service.add_assignment(
        normalized_id(sos_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# EVIDENCE
# ============================================================

@router.get(
    "/{sos_id}/evidence",
    response_model=list[SOSEvidenceResponse],
)
async def list_sos_evidence(
    sos_id: UUID,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> list[SOSEvidenceResponse]:
    await require_sos_access(
        service,
        sos_id,
        current_user,
    )

    return await service.list_evidence(
        normalized_id(sos_id),
        actor_id(current_user),
    )


@router.post(
    "/{sos_id}/evidence",
    response_model=SOSEvidenceResponse,
    status_code=201,
)
async def add_sos_evidence(
    sos_id: UUID,
    payload: SOSEvidenceCreate,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSEvidenceResponse:
    await require_sos_access(
        service,
        sos_id,
        current_user,
    )

    return await service.add_evidence(
        normalized_id(sos_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# ESCALATIONS
# ============================================================

@router.get(
    "/{sos_id}/escalations",
    response_model=list[SOSEscalationResponse],
)
async def list_sos_escalations(
    sos_id: UUID,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> list[SOSEscalationResponse]:
    await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    return await service.list_escalations(
        normalized_id(sos_id),
        actor_id(current_user),
    )


@router.post(
    "/{sos_id}/escalations",
    response_model=SOSEscalationResponse,
    status_code=201,
)
async def escalate_sos(
    sos_id: UUID,
    payload: SOSEscalationCreate,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSEscalationResponse:
    require_role(current_user, ESCALATE_ROLES)

    await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    return await service.escalate(
        normalized_id(sos_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# STATUS UPDATES
# ============================================================

@router.post(
    "/{sos_id}/status",
    response_model=SOSResponse,
)
async def update_sos_status(
    sos_id: UUID,
    payload: SOSStatusUpdate,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSResponse:
    require_role(current_user, MANAGE_ROLES)

    await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    # Prevent bypassing verification and dispatch.
    if payload.status in {
        SOSStatus.ASSIGNED,
        SOSStatus.RESCUE_IN_PROGRESS,
        SOSStatus.RESCUED,
        SOSStatus.RESOLVED,
    }:
        raise HTTPException(
            status_code=409,
            detail=(
                "Use verified rescue assignment "
                "or resolution workflow"
            ),
        )

    return await service.update_status(
        normalized_id(sos_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# RESOLUTION
# ============================================================

@router.post(
    "/{sos_id}/resolve",
    response_model=SOSResponse,
)
async def resolve_sos(
    sos_id: UUID,
    payload: SOSResolutionRequest,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSResponse:
    require_role(current_user, MANAGE_ROLES)

    sos = await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    if sos.status in {
        SOSStatus.RESOLVED,
        SOSStatus.CANCELLED,
        SOSStatus.FALSE_ALARM,
    }:
        raise HTTPException(
            status_code=409,
            detail="SOS is already finalized",
        )

    if payload.resolution == SOSResolutionType.RESCUED:
        if not (
            payload.responder_confirmation
            and payload.evidence_verified
        ):
            raise HTTPException(
                status_code=422,
                detail="Rescue verification is required",
            )

    return await service.resolve(
        normalized_id(sos_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# CANCELLATION
# ============================================================

@router.post(
    "/{sos_id}/cancel",
    response_model=SOSResponse,
)
async def cancel_sos(
    sos_id: UUID,
    payload: SOSCancellationRequest,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> SOSResponse:
    sos = await require_sos_access(
        service,
        sos_id,
        current_user,
    )

    if sos.status in {
        SOSStatus.RESOLVED,
        SOSStatus.CANCELLED,
        SOSStatus.FALSE_ALARM,
    }:
        raise HTTPException(
            status_code=409,
            detail="SOS is already finalized",
        )

    if sos.status in {
        SOSStatus.ASSIGNED,
        SOSStatus.RESCUE_IN_PROGRESS,
    }:
        require_role(current_user, MANAGE_ROLES)

        if not payload.active_team_notified:
            raise HTTPException(
                status_code=422,
                detail="Active rescue team must be notified",
            )

    return await service.cancel(
        normalized_id(sos_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# NOTIFICATIONS
# ============================================================

@router.get(
    "/{sos_id}/notifications",
    response_model=list[SOSNotificationResponse],
)
async def list_sos_notifications(
    sos_id: UUID,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> list[SOSNotificationResponse]:
    await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    return await service.list_notifications(
        normalized_id(sos_id),
        actor_id(current_user),
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get(
    "/{sos_id}/events",
    response_model=list[SOSEventResponse],
)
async def list_sos_events(
    sos_id: UUID,
    current_user: CurrentUser,
    service: SOSServiceDependency,
) -> list[SOSEventResponse]:
    await require_sos_access(
        service,
        sos_id,
        current_user,
        operational=True,
    )

    return await service.list_events(
        normalized_id(sos_id),
        actor_id(current_user),
    )


__all__ = [
    "router",
    "SOSService",
    "SOSCreate",
    "SOSUpdate",
    "SOSLocationUpdateCreate",
    "SOSTriageRequest",
    "SOSAssignmentCreate",
    "SOSEvidenceCreate",
    "SOSEscalationCreate",
    "SOSStatusUpdate",
    "SOSResolutionRequest",
    "SOSCancellationRequest",
    "SOSResponse",
    "SOSListResponse",
    "SOSStatisticsResponse",
]
