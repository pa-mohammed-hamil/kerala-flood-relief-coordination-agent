from __future__ import annotations

"""
backend/api/evacuation.py

Kerala Flood Relief Coordination Agent
Evacuation Planning & Coordination API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Features:
    - Evacuation operation management
    - District-level access control
    - Evacuation assignments
    - Checkpoint tracking
    - Evacuation status monitoring
    - Human approval for operational actions
    - Paginated operations
    - Operational statistics
    - Audit-ready workflow boundaries

Endpoints:
    GET    /evacuation
    GET    /evacuation/statistics
    POST   /evacuation
    GET    /evacuation/{evacuation_id}
    PATCH  /evacuation/{evacuation_id}
    POST   /evacuation/{evacuation_id}/approve
    POST   /evacuation/{evacuation_id}/activate
    POST   /evacuation/{evacuation_id}/complete
    POST   /evacuation/{evacuation_id}/cancel
    GET    /evacuation/{evacuation_id}/assignments
    POST   /evacuation/{evacuation_id}/assignments
    GET    /evacuation/{evacuation_id}/checkpoints
    POST   /evacuation/{evacuation_id}/checkpoints
    GET    /evacuation/{evacuation_id}/events

Safety:
    AI recommendations do not authorize evacuations.
    Activation requires verified human approval.
    Route safety must be checked against current conditions.
    Personal location data requires restricted access.
"""


import logging
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from sqlalchemy import (
    func,
    select,
)

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.evacuation import (
    Evacuation,
    EvacuationAssignment,
    EvacuationCheckpoint,
    EvacuationEvent,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/evacuation",
    tags=["Evacuation Management"],
)

DbSession = Annotated[
    AsyncSession,
    Depends(get_db),
]

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# CONFIGURATION
# ============================================================

ADMIN_ROLES = {
    "admin",
    "super_admin",
}

COMMAND_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "incident_commander",
}

FIELD_ROLES = {
    "rescue_coordinator",
    "field_officer",
    "rescue_team_leader",
}

READ_ROLES = (
    ADMIN_ROLES
    | COMMAND_ROLES
    | FIELD_ROLES
    | {"operator", "analyst"}
)

WRITE_ROLES = (
    ADMIN_ROLES
    | COMMAND_ROLES
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class EvacuationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvacuationStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AssignmentStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EN_ROUTE = "en_route"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    REACHED = "reached"
    BLOCKED = "blocked"
    BYPASSED = "bypassed"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class EvacuationCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    district_id: int = Field(ge=1)

    title: str = Field(
        min_length=5,
        max_length=255,
    )

    description: str = Field(
        min_length=10,
        max_length=10000,
    )

    priority: EvacuationPriority = (
        EvacuationPriority.MEDIUM
    )

    incident_id: str | None = None

    origin_reference: str = Field(
        min_length=3,
        max_length=500,
    )

    destination_shelter_id: str | None = None

    estimated_people: int = Field(
        ge=0,
        le=1_000_000,
    )

    planned_start_at: datetime | None = None

    route_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    ai_recommended: bool = False

    @field_validator(
        "incident_id",
        "destination_shelter_id",
    )
    @classmethod
    def validate_optional_uuid(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return str(uuid.UUID(value))


class EvacuationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None,
        min_length=5,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
        min_length=10,
        max_length=10000,
    )

    priority: EvacuationPriority | None = None

    estimated_people: int | None = Field(
        default=None,
        ge=0,
    )

    planned_start_at: datetime | None = None


class EvacuationApprovalRequest(BaseModel):
    approved: bool

    reason: str = Field(
        min_length=10,
        max_length=5000,
    )

    route_safety_verified: bool = False

    destination_capacity_verified: bool = False


class EvacuationActivationRequest(BaseModel):
    confirmation_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    route_safety_verified: bool

    shelter_readiness_verified: bool

    communication_ready: bool


class EvacuationCompletionRequest(BaseModel):
    actual_people_evacuated: int = Field(
        ge=0,
    )

    notes: str = Field(
        min_length=10,
        max_length=5000,
    )


class EvacuationCancelRequest(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=5000,
    )


class EvacuationAssignmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rescue_team_id: str

    assigned_people: int = Field(
        ge=0,
    )

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )

    @field_validator("rescue_team_id")
    @classmethod
    def validate_team_id(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))


class EvacuationCheckpointCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=3,
        max_length=255,
    )

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

    sequence_number: int = Field(ge=1)

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class EvacuationResponse(BaseModel):
    id: str
    district_id: int
    status: str
    priority: str | None = None
    created_at: datetime | None = None


class EvacuationListResponse(BaseModel):
    items: list[EvacuationResponse]
    total: int
    limit: int
    offset: int


class EvacuationAssignmentResponse(BaseModel):
    id: str
    evacuation_id: str
    status: str | None = None


class EvacuationCheckpointResponse(BaseModel):
    id: str
    evacuation_id: str
    status: str | None = None


class EvacuationStatisticsResponse(BaseModel):
    total_evacuations: int
    by_status: dict[str, int]
    generated_at: datetime


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid UUID",
        ) from None


def role_of(user: Any) -> str:
    return enum_value(
        getattr(user, "role", "")
    ).lower()


def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return str(identifier)


def require_role(
    user: Any,
    allowed_roles: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) not in allowed_roles:
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

    assigned_district = getattr(
        user,
        "district_id",
        None,
    )

    if (
        assigned_district is None
        or int(assigned_district) != district_id
    ):
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

    assigned_district = getattr(
        user,
        "district_id",
        None,
    )

    if assigned_district is None:
        raise HTTPException(
            status_code=403,
            detail="District assignment required",
        )

    if (
        district_id is not None
        and district_id != int(assigned_district)
    ):
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )

    return int(assigned_district)


async def get_evacuation_or_404(
    db: AsyncSession,
    evacuation_id: str,
    user: Any,
) -> Evacuation:
    identifier = validate_uuid(evacuation_id)

    result = await db.execute(
        select(Evacuation).where(
            Evacuation.id == identifier
        )
    )

    evacuation = result.scalar_one_or_none()

    if evacuation is None:
        raise HTTPException(
            status_code=404,
            detail="Evacuation not found",
        )

    require_district_access(
        user,
        evacuation.district_id,
    )

    return evacuation


def evacuation_response(
    evacuation: Evacuation,
) -> EvacuationResponse:
    return EvacuationResponse(
        id=str(evacuation.id),
        district_id=evacuation.district_id,
        status=enum_value(evacuation.status),
        priority=enum_value(
            getattr(
                evacuation,
                "priority",
                None,
            )
        ) or None,
        created_at=getattr(
            evacuation,
            "created_at",
            None,
        ),
    )


def require_editable(
    evacuation: Evacuation,
) -> None:
    if enum_value(evacuation.status) not in {
        "draft",
        "pending_approval",
    }:
        raise HTTPException(
            status_code=409,
            detail="Evacuation cannot be edited",
        )


# ============================================================
# LIST EVACUATIONS
# ============================================================

@router.get(
    "",
    response_model=EvacuationListResponse,
)
async def list_evacuations(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> EvacuationListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(Evacuation)

    count_query = (
        select(func.count())
        .select_from(Evacuation)
    )

    if district_id is not None:
        query = query.where(
            Evacuation.district_id == district_id
        )

        count_query = count_query.where(
            Evacuation.district_id == district_id
        )

    total_result = await db.execute(
        count_query
    )

    result = await db.execute(
        query.order_by(
            Evacuation.created_at.desc()
        )
        .offset(offset)
        .limit(limit)
    )

    return EvacuationListResponse(
        items=[
            evacuation_response(item)
            for item in result.scalars().all()
        ],
        total=int(total_result.scalar_one()),
        limit=limit,
        offset=offset,
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=EvacuationStatisticsResponse,
)
async def evacuation_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> EvacuationStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(
        Evacuation.status,
        func.count(Evacuation.id),
    )

    if district_id is not None:
        query = query.where(
            Evacuation.district_id == district_id
        )

    query = query.group_by(
        Evacuation.status
    )

    result = await db.execute(query)

    by_status = {
        enum_value(row[0]): int(row[1])
        for row in result.all()
    }

    return EvacuationStatisticsResponse(
        total_evacuations=sum(
            by_status.values()
        ),
        by_status=by_status,
        generated_at=utc_now(),
    )


# ============================================================
# CREATE EVACUATION
# ============================================================

@router.post(
    "",
    response_model=EvacuationResponse,
    status_code=201,
)
async def create_evacuation(
    payload: EvacuationCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> EvacuationResponse:
    require_role(current_user, WRITE_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    # Domain service must validate:
    # - Incident ownership
    # - Shelter destination and capacity
    # - Route availability
    # - District authorization
    # - Draft lifecycle
    # - Audit event creation
    raise HTTPException(
        status_code=501,
        detail="Evacuation creation service is not configured",
    )


# ============================================================
# GET EVACUATION
# ============================================================

@router.get(
    "/{evacuation_id}",
    response_model=EvacuationResponse,
)
async def get_evacuation(
    evacuation_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> EvacuationResponse:
    require_role(current_user, READ_ROLES)

    evacuation = await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    return evacuation_response(evacuation)


# ============================================================
# UPDATE EVACUATION
# ============================================================

@router.patch(
    "/{evacuation_id}",
    response_model=EvacuationResponse,
)
async def update_evacuation(
    evacuation_id: str,
    payload: EvacuationUpdate,
    db: DbSession,
    current_user: CurrentUser,
) -> EvacuationResponse:
    require_role(current_user, WRITE_ROLES)

    evacuation = await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    require_editable(evacuation)

    raise HTTPException(
        status_code=501,
        detail="Evacuation update service is not configured",
    )


# ============================================================
# APPROVAL
# ============================================================

@router.post("/{evacuation_id}/approve")
async def approve_evacuation(
    evacuation_id: str,
    payload: EvacuationApprovalRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    evacuation = await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    if enum_value(evacuation.status) not in {
        "draft",
        "pending_approval",
    }:
        raise HTTPException(
            status_code=409,
            detail="Evacuation cannot be reviewed",
        )

    if payload.approved:
        if not (
            payload.route_safety_verified
            and payload.destination_capacity_verified
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Route safety and destination capacity "
                    "must be verified before approval"
                ),
            )

    raise HTTPException(
        status_code=501,
        detail="Human approval workflow is not configured",
    )


# ============================================================
# ACTIVATION
# ============================================================

@router.post("/{evacuation_id}/activate")
async def activate_evacuation(
    evacuation_id: str,
    payload: EvacuationActivationRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    evacuation = await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    if enum_value(evacuation.status) != "approved":
        raise HTTPException(
            status_code=409,
            detail="Evacuation must be approved first",
        )

    if not all(
        [
            payload.route_safety_verified,
            payload.shelter_readiness_verified,
            payload.communication_ready,
        ]
    ):
        raise HTTPException(
            status_code=422,
            detail="Operational readiness checks are incomplete",
        )

    raise HTTPException(
        status_code=501,
        detail="Evacuation activation service is not configured",
    )


# ============================================================
# COMPLETION
# ============================================================

@router.post("/{evacuation_id}/complete")
async def complete_evacuation(
    evacuation_id: str,
    payload: EvacuationCompletionRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    evacuation = await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    if enum_value(evacuation.status) not in {
        "active",
        "paused",
    }:
        raise HTTPException(
            status_code=409,
            detail="Evacuation is not active",
        )

    raise HTTPException(
        status_code=501,
        detail="Evacuation completion service is not configured",
    )


# ============================================================
# CANCELLATION
# ============================================================

@router.post("/{evacuation_id}/cancel")
async def cancel_evacuation(
    evacuation_id: str,
    payload: EvacuationCancelRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    evacuation = await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    if enum_value(evacuation.status) in {
        "completed",
        "cancelled",
    }:
        raise HTTPException(
            status_code=409,
            detail="Evacuation is already finalized",
        )

    raise HTTPException(
        status_code=501,
        detail="Evacuation cancellation service is not configured",
    )


# ============================================================
# ASSIGNMENTS
# ============================================================

@router.get("/{evacuation_id}/assignments")
async def list_evacuation_assignments(
    evacuation_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[EvacuationAssignmentResponse]:
    require_role(current_user, READ_ROLES)

    await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    result = await db.execute(
        select(EvacuationAssignment).where(
            EvacuationAssignment.evacuation_id
            == validate_uuid(evacuation_id)
        )
    )

    return [
        EvacuationAssignmentResponse(
            id=str(item.id),
            evacuation_id=str(
                item.evacuation_id
            ),
            status=enum_value(
                getattr(
                    item,
                    "status",
                    None,
                )
            ) or None,
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/{evacuation_id}/assignments",
    status_code=201,
)
async def create_evacuation_assignment(
    evacuation_id: str,
    payload: EvacuationAssignmentCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    raise HTTPException(
        status_code=501,
        detail="Rescue team assignment service is not configured",
    )


# ============================================================
# CHECKPOINTS
# ============================================================

@router.get("/{evacuation_id}/checkpoints")
async def list_evacuation_checkpoints(
    evacuation_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[EvacuationCheckpointResponse]:
    require_role(current_user, READ_ROLES)

    await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    result = await db.execute(
        select(EvacuationCheckpoint).where(
            EvacuationCheckpoint.evacuation_id
            == validate_uuid(evacuation_id)
        )
    )

    return [
        EvacuationCheckpointResponse(
            id=str(item.id),
            evacuation_id=str(
                item.evacuation_id
            ),
            status=enum_value(
                getattr(
                    item,
                    "status",
                    None,
                )
            ) or None,
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/{evacuation_id}/checkpoints",
    status_code=201,
)
async def create_evacuation_checkpoint(
    evacuation_id: str,
    payload: EvacuationCheckpointCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    raise HTTPException(
        status_code=501,
        detail="Checkpoint creation service is not configured",
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get("/{evacuation_id}/events")
async def evacuation_events(
    evacuation_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_evacuation_or_404(
        db,
        evacuation_id,
        current_user,
    )

    result = await db.execute(
        select(EvacuationEvent)
        .where(
            EvacuationEvent.evacuation_id
            == validate_uuid(evacuation_id)
        )
        .order_by(
            EvacuationEvent.created_at.desc()
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "evacuation_id": str(
                item.evacuation_id
            ),
            "created_at": item.created_at,
        }
        for item in result.scalars().all()
    ]


__all__ = [
    "router",
    "EvacuationCreate",
    "EvacuationUpdate",
    "EvacuationApprovalRequest",
    "EvacuationActivationRequest",
    "EvacuationCompletionRequest",
    "EvacuationCancelRequest",
    "EvacuationAssignmentCreate",
    "EvacuationCheckpointCreate",
    "EvacuationResponse",
    "EvacuationListResponse",
    "EvacuationStatisticsResponse",
]
