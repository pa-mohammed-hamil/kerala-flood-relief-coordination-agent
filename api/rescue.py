
"""
backend/api/rescue.py

Kerala Flood Relief Coordination Agent
Rescue Operations & Emergency Mission API

Endpoints:
    GET    /rescue/health
    GET    /rescue/teams
    GET    /rescue/teams/{team_id}
    GET    /rescue/teams/{team_id}/members
    GET    /rescue/teams/{team_id}/equipment
    GET    /rescue/teams/{team_id}/locations
    GET    /rescue/missions
    GET    /rescue/statistics
    POST   /rescue/missions
    GET    /rescue/missions/{mission_id}
    PATCH  /rescue/missions/{mission_id}
    GET    /rescue/missions/{mission_id}/assignments
    POST   /rescue/missions/{mission_id}/assignments
    GET    /rescue/missions/{mission_id}/checkpoints
    POST   /rescue/missions/{mission_id}/checkpoints
    GET    /rescue/missions/{mission_id}/equipment
    POST   /rescue/missions/{mission_id}/equipment
    GET    /rescue/missions/{mission_id}/evidence
    POST   /rescue/missions/{mission_id}/evidence
    POST   /rescue/missions/{mission_id}/approve
    POST   /rescue/missions/{mission_id}/dispatch
    POST   /rescue/missions/{mission_id}/status
    POST   /rescue/missions/{mission_id}/complete
    POST   /rescue/missions/{mission_id}/cancel
    GET    /rescue/missions/{mission_id}/events

Safety:
    - AI suggestions never dispatch teams.
    - Human approval is mandatory.
    - Team availability and safety must be verified.
    - SOS and incident references must be validated.
    - Mission status changes must be audited.
    - Sensitive responder locations are restricted.
"""

from __future__ import annotations

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
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.rescue_team import (
    RescueTeam,
    RescueTeamMember,
    RescueTeamEquipment,
    RescueTeamLocation,
)
from backend.models.rescue_mission import (
    RescueMission,
    RescueMissionAssignment,
    RescueMissionCheckpoint,
    RescueMissionEquipment,
    RescueMissionEvidence,
    RescueMissionEvent,
)


router = APIRouter(
    prefix="/rescue",
    tags=["Rescue Operations"],
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
# ROLES
# ============================================================

ADMIN_ROLES = {"admin", "super_admin"}

COMMAND_ROLES = {
    "incident_commander",
    "emergency_coordinator",
    "district_officer",
    "rescue_coordinator",
}

FIELD_ROLES = {
    "rescue_team_leader",
    "rescue_operator",
    "field_officer",
}

READ_ROLES = (
    ADMIN_ROLES
    | COMMAND_ROLES
    | FIELD_ROLES
    | {"auditor"}
)

WRITE_ROLES = ADMIN_ROLES | COMMAND_ROLES

APPROVAL_ROLES = ADMIN_ROLES | {
    "incident_commander",
    "district_officer",
    "rescue_coordinator",
}

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMS
# ============================================================

class RescuePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RescueMissionStatus(str, Enum):
    REPORTED = "reported"
    TRIAGED = "triaged"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    RESCUE_IN_PROGRESS = "rescue_in_progress"
    EVACUATING = "evacuating"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RescueType(str, Enum):
    WATER_RESCUE = "water_rescue"
    BOAT_RESCUE = "boat_rescue"
    BUILDING_EVACUATION = "building_evacuation"
    LANDSLIDE_RESCUE = "landslide_rescue"
    MEDICAL_EVACUATION = "medical_evacuation"
    SEARCH_AND_RESCUE = "search_and_rescue"
    ANIMAL_RESCUE = "animal_rescue"
    OTHER = "other"


class CheckpointType(str, Enum):
    DEPARTURE = "departure"
    ARRIVAL = "arrival"
    ROUTE_BLOCKED = "route_blocked"
    RESCUE_STARTED = "rescue_started"
    RESCUE_COMPLETED = "rescue_completed"
    RETURNED_TO_BASE = "returned_to_base"


class ReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


# ============================================================
# REQUEST MODELS
# ============================================================

class RescueMissionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    district_id: int = Field(ge=1)
    title: str = Field(min_length=5, max_length=255)
    description: str = Field(
        min_length=10,
        max_length=5000,
    )

    rescue_type: RescueType
    priority: RescuePriority = RescuePriority.HIGH

    incident_id: str | None = None
    sos_request_id: str | None = None

    location_reference: str = Field(
        min_length=3,
        max_length=500,
    )

    estimated_people_at_risk: int = Field(
        default=0,
        ge=0,
    )

    special_requirements: str | None = Field(
        default=None,
        max_length=3000,
    )

    @field_validator(
        "incident_id",
        "sos_request_id",
    )
    @classmethod
    def validate_reference(
        cls,
        value: str | None,
    ) -> str | None:
        return str(uuid.UUID(value)) if value else None


class RescueMissionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None,
        min_length=5,
        max_length=255,
    )
    description: str | None = Field(
        default=None,
        min_length=10,
        max_length=5000,
    )
    priority: RescuePriority | None = None
    estimated_people_at_risk: int | None = Field(
        default=None,
        ge=0,
    )


class RescueAssignmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rescue_team_id: str
    assignment_notes: str = Field(
        min_length=10,
        max_length=3000,
    )
    team_availability_verified: bool = False
    equipment_verified: bool = False
    team_leader_acknowledged: bool = False

    @field_validator("rescue_team_id")
    @classmethod
    def validate_team(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))


class RescueCheckpointCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_type: CheckpointType
    observed_at: datetime
    location_reference: str = Field(
        min_length=3,
        max_length=500,
    )
    notes: str = Field(
        min_length=5,
        max_length=3000,
    )

    @field_validator("observed_at")
    @classmethod
    def validate_time(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "Timezone-aware timestamp required"
            )
        if value > datetime.now(timezone.utc):
            raise ValueError(
                "Checkpoint time cannot be in the future"
            )
        return value


class RescueEquipmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equipment_reference: str = Field(
        min_length=3,
        max_length=150,
    )
    quantity: int = Field(ge=1)
    safety_inspection_verified: bool = False
    notes: str | None = Field(
        default=None,
        max_length=2000,
    )


class RescueEvidenceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_reference: str = Field(
        min_length=5,
        max_length=500,
    )
    description: str = Field(
        min_length=10,
        max_length=3000,
    )
    captured_at: datetime
    source_verified: bool = False


class RescueApprovalRequest(BaseModel):
    action: ReviewAction
    review_notes: str = Field(
        min_length=10,
        max_length=3000,
    )
    risk_assessment_verified: bool = False
    route_safety_verified: bool = False
    rescue_resources_verified: bool = False


class RescueDispatchRequest(BaseModel):
    dispatch_notes: str = Field(
        min_length=10,
        max_length=3000,
    )
    team_ready: bool = False
    communication_verified: bool = False
    route_verified: bool = False
    safety_briefing_completed: bool = False


class RescueStatusUpdate(BaseModel):
    status: RescueMissionStatus
    reason: str = Field(
        min_length=10,
        max_length=3000,
    )


class RescueCompletionRequest(BaseModel):
    people_rescued: int = Field(ge=0)
    people_transferred: int = Field(ge=0)
    completion_notes: str = Field(
        min_length=10,
        max_length=5000,
    )
    team_return_status_verified: bool = False
    survivor_handover_verified: bool = False
    evidence_verified: bool = False


class RescueCancellationRequest(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=3000,
    )
    active_team_recall_confirmed: bool = False


# ============================================================
# RESPONSE MODELS
# ============================================================

class RescueTeamResponse(BaseModel):
    id: str
    district_id: int
    name: str | None = None
    status: str | None = None


class RescueMissionResponse(BaseModel):
    id: str
    district_id: int
    title: str
    status: str
    priority: str | None = None
    created_at: datetime | None = None


class RescueMissionListResponse(BaseModel):
    items: list[RescueMissionResponse]
    total: int
    limit: int
    offset: int


class RescueStatisticsResponse(BaseModel):
    total_missions: int
    by_status: dict[str, int]
    total_teams: int
    generated_at: datetime


class RescueHealthResponse(BaseModel):
    service: str
    status: str
    timestamp: datetime


# ============================================================
# RESCUE SERVICE CONTRACT
# ============================================================

class RescueService(Protocol):
    """
    Service must enforce:
      - Human approval before dispatch
      - District and team authorization
      - Verified SOS/incident references
      - Team availability and capability checks
      - Equipment and transport readiness
      - Route safety and communications
      - Idempotent state transitions
      - Immutable audit logging
      - No autonomous emergency dispatch
    """

    async def health(self) -> dict[str, Any]: ...

    async def create_mission(
        self,
        payload: RescueMissionCreate,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def update_mission(
        self,
        mission_id: str,
        payload: RescueMissionUpdate,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def add_assignment(
        self,
        mission_id: str,
        payload: RescueAssignmentCreate,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def add_checkpoint(
        self,
        mission_id: str,
        payload: RescueCheckpointCreate,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def add_equipment(
        self,
        mission_id: str,
        payload: RescueEquipmentCreate,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def add_evidence(
        self,
        mission_id: str,
        payload: RescueEvidenceCreate,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def review_mission(
        self,
        mission_id: str,
        payload: RescueApprovalRequest,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def dispatch_mission(
        self,
        mission_id: str,
        payload: RescueDispatchRequest,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def update_status(
        self,
        mission_id: str,
        payload: RescueStatusUpdate,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def complete_mission(
        self,
        mission_id: str,
        payload: RescueCompletionRequest,
        actor_id: str,
    ) -> dict[str, Any]: ...

    async def cancel_mission(
        self,
        mission_id: str,
        payload: RescueCancellationRequest,
        actor_id: str,
    ) -> dict[str, Any]: ...


def get_rescue_service(
    request: Request,
) -> RescueService:
    service = getattr(
        request.app.state,
        "rescue_service",
        None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Rescue service is not configured",
        )
    return service


RescueServiceDependency = Annotated[
    RescueService,
    Depends(get_rescue_service),
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


def enum_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


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


async def get_team_or_404(
    db: AsyncSession,
    team_id: str,
    user: Any,
) -> RescueTeam:
    result = await db.execute(
        select(RescueTeam).where(
            RescueTeam.id == validate_uuid(team_id)
        )
    )
    team = result.scalar_one_or_none()
    if team is None:
        raise HTTPException(
            status_code=404,
            detail="Rescue team not found",
        )
    require_district_access(user, team.district_id)
    return team


async def get_mission_or_404(
    db: AsyncSession,
    mission_id: str,
    user: Any,
) -> RescueMission:
    result = await db.execute(
        select(RescueMission).where(
            RescueMission.id == validate_uuid(mission_id)
        )
    )
    mission = result.scalar_one_or_none()
    if mission is None:
        raise HTTPException(
            status_code=404,
            detail="Rescue mission not found",
        )
    require_district_access(user, mission.district_id)
    return mission


def team_response(team: RescueTeam) -> RescueTeamResponse:
    return RescueTeamResponse(
        id=str(team.id),
        district_id=team.district_id,
        name=getattr(team, "name", None),
        status=enum_value(
            getattr(team, "status", None)
        ) or None,
    )


def mission_response(
    mission: RescueMission,
) -> RescueMissionResponse:
    return RescueMissionResponse(
        id=str(mission.id),
        district_id=mission.district_id,
        title=mission.title,
        status=enum_value(mission.status),
        priority=enum_value(
            getattr(mission, "priority", None)
        ) or None,
        created_at=getattr(mission, "created_at", None),
    )


async def related_records(
    db: AsyncSession,
    model: Any,
    column: Any,
    identifier: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    result = await db.execute(
        select(model)
        .where(column == identifier)
        .limit(limit)
    )
    return [
        {"id": str(item.id)}
        for item in result.scalars().all()
    ]


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
async def rescue_health(
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, READ_ROLES)
    return await service.health()


# ============================================================
# RESCUE TEAMS
# ============================================================

@router.get(
    "/teams",
    response_model=list[RescueTeamResponse],
)
async def list_rescue_teams(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[RescueTeamResponse]:
    district_id = scoped_district(current_user, district_id)

    query = select(RescueTeam)
    if district_id is not None:
        query = query.where(
            RescueTeam.district_id == district_id
        )

    result = await db.execute(
        query.order_by(RescueTeam.id).limit(limit)
    )
    return [
        team_response(item)
        for item in result.scalars().all()
    ]


@router.get(
    "/teams/{team_id}",
    response_model=RescueTeamResponse,
)
async def get_rescue_team(
    team_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> RescueTeamResponse:
    require_role(current_user, READ_ROLES)
    team = await get_team_or_404(db, team_id, current_user)
    return team_response(team)


@router.get("/teams/{team_id}/members")
async def rescue_team_members(
    team_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, COMMAND_ROLES | ADMIN_ROLES)
    await get_team_or_404(db, team_id, current_user)
    return await related_records(
        db,
        RescueTeamMember,
        RescueTeamMember.team_id,
        validate_uuid(team_id),
    )


@router.get("/teams/{team_id}/equipment")
async def rescue_team_equipment(
    team_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)
    await get_team_or_404(db, team_id, current_user)
    return await related_records(
        db,
        RescueTeamEquipment,
        RescueTeamEquipment.team_id,
        validate_uuid(team_id),
    )


@router.get("/teams/{team_id}/locations")
async def rescue_team_locations(
    team_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, COMMAND_ROLES | ADMIN_ROLES)
    await get_team_or_404(db, team_id, current_user)
    return await related_records(
        db,
        RescueTeamLocation,
        RescueTeamLocation.team_id,
        validate_uuid(team_id),
        limit=50,
    )


# ============================================================
# MISSIONS
# ============================================================

@router.get(
    "/missions",
    response_model=RescueMissionListResponse,
)
async def list_rescue_missions(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> RescueMissionListResponse:
    district_id = scoped_district(current_user, district_id)

    query = select(RescueMission)
    count_query = select(func.count(RescueMission.id))

    if district_id is not None:
        query = query.where(
            RescueMission.district_id == district_id
        )
        count_query = count_query.where(
            RescueMission.district_id == district_id
        )

    total = (await db.execute(count_query)).scalar_one()

    result = await db.execute(
        query.order_by(RescueMission.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    return RescueMissionListResponse(
        items=[
            mission_response(item)
            for item in result.scalars().all()
        ],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/statistics",
    response_model=RescueStatisticsResponse,
)
async def rescue_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(default=None, ge=1),
) -> RescueStatisticsResponse:
    district_id = scoped_district(current_user, district_id)

    query = select(
        RescueMission.status,
        func.count(RescueMission.id),
    ).group_by(RescueMission.status)

    team_query = select(func.count(RescueTeam.id))

    if district_id is not None:
        query = query.where(
            RescueMission.district_id == district_id
        )
        team_query = team_query.where(
            RescueTeam.district_id == district_id
        )

    rows = (await db.execute(query)).all()

    by_status = {
        enum_value(row[0]): int(row[1])
        for row in rows
    }

    total_teams = (
        await db.execute(team_query)
    ).scalar_one()

    return RescueStatisticsResponse(
        total_missions=sum(by_status.values()),
        by_status=by_status,
        total_teams=int(total_teams),
        generated_at=datetime.now(timezone.utc),
    )


@router.post("/missions", status_code=201)
async def create_rescue_mission(
    payload: RescueMissionCreate,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)
    require_district_access(current_user, payload.district_id)
    return await service.create_mission(
        payload, actor_id(current_user)
    )


@router.get(
    "/missions/{mission_id}",
    response_model=RescueMissionResponse,
)
async def get_rescue_mission(
    mission_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> RescueMissionResponse:
    require_role(current_user, READ_ROLES)
    mission = await get_mission_or_404(
        db, mission_id, current_user
    )
    return mission_response(mission)


@router.patch("/missions/{mission_id}")
async def update_rescue_mission(
    mission_id: str,
    payload: RescueMissionUpdate,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)
    mission = await get_mission_or_404(
        db, mission_id, current_user
    )

    if enum_value(mission.status) in {
        "completed", "cancelled", "failed"
    }:
        raise HTTPException(
            status_code=409,
            detail="Finalized mission cannot be edited",
        )

    return await service.update_mission(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# ASSIGNMENTS
# ============================================================

@router.get("/missions/{mission_id}/assignments")
async def list_mission_assignments(
    mission_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)
    await get_mission_or_404(db, mission_id, current_user)
    return await related_records(
        db,
        RescueMissionAssignment,
        RescueMissionAssignment.mission_id,
        validate_uuid(mission_id),
    )


@router.post(
    "/missions/{mission_id}/assignments",
    status_code=201,
)
async def add_mission_assignment(
    mission_id: str,
    payload: RescueAssignmentCreate,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)
    await get_mission_or_404(db, mission_id, current_user)

    if not all([
        payload.team_availability_verified,
        payload.equipment_verified,
        payload.team_leader_acknowledged,
    ]):
        raise HTTPException(
            status_code=422,
            detail="Team readiness verification incomplete",
        )

    return await service.add_assignment(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# CHECKPOINTS
# ============================================================

@router.get("/missions/{mission_id}/checkpoints")
async def list_mission_checkpoints(
    mission_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)
    await get_mission_or_404(db, mission_id, current_user)
    return await related_records(
        db,
        RescueMissionCheckpoint,
        RescueMissionCheckpoint.mission_id,
        validate_uuid(mission_id),
    )


@router.post(
    "/missions/{mission_id}/checkpoints",
    status_code=201,
)
async def add_mission_checkpoint(
    mission_id: str,
    payload: RescueCheckpointCreate,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(
        current_user,
        WRITE_ROLES | FIELD_ROLES,
    )
    await get_mission_or_404(db, mission_id, current_user)
    return await service.add_checkpoint(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# MISSION EQUIPMENT
# ============================================================

@router.get("/missions/{mission_id}/equipment")
async def list_mission_equipment(
    mission_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)
    await get_mission_or_404(db, mission_id, current_user)
    return await related_records(
        db,
        RescueMissionEquipment,
        RescueMissionEquipment.mission_id,
        validate_uuid(mission_id),
    )


@router.post(
    "/missions/{mission_id}/equipment",
    status_code=201,
)
async def add_mission_equipment(
    mission_id: str,
    payload: RescueEquipmentCreate,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)
    await get_mission_or_404(db, mission_id, current_user)

    if not payload.safety_inspection_verified:
        raise HTTPException(
            status_code=422,
            detail="Equipment inspection required",
        )

    return await service.add_equipment(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# EVIDENCE
# ============================================================

@router.get("/missions/{mission_id}/evidence")
async def list_mission_evidence(
    mission_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)
    await get_mission_or_404(db, mission_id, current_user)
    return await related_records(
        db,
        RescueMissionEvidence,
        RescueMissionEvidence.mission_id,
        validate_uuid(mission_id),
    )


@router.post(
    "/missions/{mission_id}/evidence",
    status_code=201,
)
async def add_mission_evidence(
    mission_id: str,
    payload: RescueEvidenceCreate,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(
        current_user,
        WRITE_ROLES | FIELD_ROLES,
    )
    await get_mission_or_404(db, mission_id, current_user)
    return await service.add_evidence(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# APPROVAL
# ============================================================

@router.post("/missions/{mission_id}/approve")
async def approve_rescue_mission(
    mission_id: str,
    payload: RescueApprovalRequest,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, APPROVAL_ROLES)

    mission = await get_mission_or_404(
        db, mission_id, current_user
    )

    if enum_value(mission.status) != "pending_approval":
        raise HTTPException(
            status_code=409,
            detail="Mission is not pending approval",
        )

    if payload.action == ReviewAction.APPROVE:
        if not all([
            payload.risk_assessment_verified,
            payload.route_safety_verified,
            payload.rescue_resources_verified,
        ]):
            raise HTTPException(
                status_code=422,
                detail="Approval safety checks incomplete",
            )

    return await service.review_mission(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# DISPATCH
# ============================================================

@router.post("/missions/{mission_id}/dispatch")
async def dispatch_rescue_mission(
    mission_id: str,
    payload: RescueDispatchRequest,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, APPROVAL_ROLES)

    mission = await get_mission_or_404(
        db, mission_id, current_user
    )

    if enum_value(mission.status) != "approved":
        raise HTTPException(
            status_code=409,
            detail="Mission requires prior approval",
        )

    if not all([
        payload.team_ready,
        payload.communication_verified,
        payload.route_verified,
        payload.safety_briefing_completed,
    ]):
        raise HTTPException(
            status_code=422,
            detail="Dispatch readiness checks incomplete",
        )

    return await service.dispatch_mission(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# STATUS UPDATE
# ============================================================

@router.post("/missions/{mission_id}/status")
async def update_rescue_mission_status(
    mission_id: str,
    payload: RescueStatusUpdate,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(
        current_user,
        WRITE_ROLES | FIELD_ROLES,
    )

    await get_mission_or_404(
        db, mission_id, current_user
    )

    # Service must enforce permitted transitions.
    # This endpoint must not bypass dispatch approval.
    return await service.update_status(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# COMPLETE
# ============================================================

@router.post("/missions/{mission_id}/complete")
async def complete_rescue_mission(
    mission_id: str,
    payload: RescueCompletionRequest,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, APPROVAL_ROLES)

    mission = await get_mission_or_404(
        db, mission_id, current_user
    )

    if enum_value(mission.status) not in {
        "on_scene",
        "rescue_in_progress",
        "evacuating",
    }:
        raise HTTPException(
            status_code=409,
            detail="Mission is not ready for completion",
        )

    if not all([
        payload.team_return_status_verified,
        payload.survivor_handover_verified,
        payload.evidence_verified,
    ]):
        raise HTTPException(
            status_code=422,
            detail="Completion verification incomplete",
        )

    return await service.complete_mission(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# CANCEL
# ============================================================

@router.post("/missions/{mission_id}/cancel")
async def cancel_rescue_mission(
    mission_id: str,
    payload: RescueCancellationRequest,
    db: DbSession,
    current_user: CurrentUser,
    service: RescueServiceDependency,
) -> dict[str, Any]:
    require_role(current_user, APPROVAL_ROLES)

    mission = await get_mission_or_404(
        db, mission_id, current_user
    )

    if enum_value(mission.status) in {
        "completed",
        "cancelled",
    }:
        raise HTTPException(
            status_code=409,
            detail="Mission already finalized",
        )

    if enum_value(mission.status) in {
        "dispatched",
        "en_route",
        "on_scene",
        "rescue_in_progress",
        "evacuating",
    } and not payload.active_team_recall_confirmed:
        raise HTTPException(
            status_code=422,
            detail="Active team recall confirmation required",
        )

    return await service.cancel_mission(
        validate_uuid(mission_id),
        payload,
        actor_id(current_user),
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get("/missions/{mission_id}/events")
async def rescue_mission_events(
    mission_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)
    await get_mission_or_404(
        db, mission_id, current_user
    )
    return await related_records(
        db,
        RescueMissionEvent,
        RescueMissionEvent.mission_id,
        validate_uuid(mission_id),
    )


__all__ = [
    "router",
    "RescueService",
    "RescueMissionCreate",
    "RescueMissionUpdate",
    "RescueAssignmentCreate",
    "RescueCheckpointCreate",
    "RescueEquipmentCreate",
    "RescueEvidenceCreate",
    "RescueApprovalRequest",
    "RescueDispatchRequest",
    "RescueStatusUpdate",
    "RescueCompletionRequest",
    "RescueCancellationRequest",
]
