from __future__ import annotations
"""
Agent orchestration API for Kerala Flood Relief Coordination Agent.

Routes:
    GET    /agents
    GET    /agents/health
    GET    /agents/decisions
    GET    /agents/decisions/{decision_id}
    POST   /agents/decisions/{decision_id}/review
    POST   /agents/decisions/{decision_id}/execute
    POST   /agents/decisions/{decision_id}/cancel
    GET    /agents/executions/{execution_id}

Security:
- JWT authentication through backend.dependencies
- Role-based authorization
- Human approval for consequential actions
- Audit logging
- AI recommendations are advisory
"""


import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user
from backend.models.agent_decision import (
    AgentDecision,
    AgentDecisionReview,
    AgentExecution,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["AI Agents"],
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

AGENT_REGISTRY: dict[str, dict[str, Any]] = {
    "supervisor_agent": {
        "name": "Supervisor Agent",
        "description": "Coordinates flood response agents",
        "category": "orchestration",
    },
    "flood_forecast_agent": {
        "name": "Flood Forecast Agent",
        "description": "Analyzes rainfall and flood predictions",
        "category": "forecasting",
    },
    "risk_assessment_agent": {
        "name": "Risk Assessment Agent",
        "description": "Assesses flood and landslide risks",
        "category": "risk",
    },
    "sos_triage_agent": {
        "name": "SOS Triage Agent",
        "description": "Prioritizes emergency assistance requests",
        "category": "emergency",
    },
    "rescue_dispatch_agent": {
        "name": "Rescue Dispatch Agent",
        "description": "Recommends rescue team assignments",
        "category": "rescue",
    },
    "evacuation_agent": {
        "name": "Evacuation Agent",
        "description": "Recommends evacuation plans",
        "category": "evacuation",
    },
    "shelter_agent": {
        "name": "Shelter Agent",
        "description": "Assesses shelter capacity and needs",
        "category": "shelter",
    },
    "relief_logistics_agent": {
        "name": "Relief Logistics Agent",
        "description": "Recommends relief supply distribution",
        "category": "logistics",
    },
    "medical_agent": {
        "name": "Medical Agent",
        "description": "Supports medical resource coordination",
        "category": "healthcare",
    },
    "volunteer_agent": {
        "name": "Volunteer Agent",
        "description": "Recommends volunteer allocations",
        "category": "volunteers",
    },
    "damage_assessment_agent": {
        "name": "Damage Assessment Agent",
        "description": "Assists with damage evaluation",
        "category": "recovery",
    },
    "notification_agent": {
        "name": "Notification Agent",
        "description": "Drafts public safety notifications",
        "category": "communications",
    },
    "verification_agent": {
        "name": "Verification Agent",
        "description": "Checks evidence and recommendations",
        "category": "verification",
    },
}


# ============================================================
# SCHEMAS
# ============================================================

class DecisionReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class DecisionReviewRequest(BaseModel):
    action: DecisionReviewAction
    reason: str = Field(
        min_length=5,
        max_length=5000,
    )


class DecisionExecutionRequest(BaseModel):
    idempotency_key: str = Field(
        min_length=8,
        max_length=150,
    )
    dry_run: bool = True


class DecisionCancelRequest(BaseModel):
    reason: str = Field(
        min_length=5,
        max_length=5000,
    )


class AgentInfo(BaseModel):
    agent_id: str
    name: str
    description: str
    category: str


class AgentHealthResponse(BaseModel):
    service: str
    status: str
    registered_agents: int
    timestamp: datetime


class AgentDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_name: str | None = None
    status: str
    created_at: datetime | None = None


class PaginatedDecisions(BaseModel):
    items: list[AgentDecisionResponse]
    total: int
    limit: int
    offset: int


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def user_identifier(user: Any) -> str:
    value = getattr(user, "id", None)

    if value is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authenticated user",
        )

    return str(value)


def require_role(
    user: Any,
    allowed_roles: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    role = enum_value(getattr(user, "role", ""))

    if role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )


def validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid identifier",
        ) from None


async def get_decision_or_404(
    db: AsyncSession,
    decision_id: str,
) -> AgentDecision:
    identifier = validate_uuid(decision_id)

    result = await db.execute(
        select(AgentDecision).where(
            AgentDecision.id == identifier
        )
    )

    decision = result.scalar_one_or_none()

    if decision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent decision not found",
        )

    return decision


def decision_response(
    decision: AgentDecision,
) -> AgentDecisionResponse:
    return AgentDecisionResponse(
        id=str(decision.id),
        agent_name=getattr(
            decision,
            "agent_name",
            None,
        ),
        status=enum_value(decision.status),
        created_at=getattr(
            decision,
            "created_at",
            None,
        ),
    )


# ============================================================
# AGENT DISCOVERY
# ============================================================

@router.get(
    "",
    response_model=list[AgentInfo],
)
async def list_agents(
    current_user: CurrentUser,
) -> list[AgentInfo]:
    return [
        AgentInfo(
            agent_id=agent_id,
            **metadata,
        )
        for agent_id, metadata in AGENT_REGISTRY.items()
    ]


@router.get(
    "/health",
    response_model=AgentHealthResponse,
)
async def agents_health(
    current_user: CurrentUser,
) -> AgentHealthResponse:
    return AgentHealthResponse(
        service="agent_orchestration",
        status="api_available",
        registered_agents=len(AGENT_REGISTRY),
        timestamp=utc_now(),
    )


# ============================================================
# AGENT DECISIONS
# ============================================================

@router.get(
    "/decisions",
    response_model=PaginatedDecisions,
)
async def list_agent_decisions(
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> PaginatedDecisions:
    require_role(
        current_user,
        {
            "admin",
            "district_officer",
            "emergency_coordinator",
            "operator",
            "reviewer",
        },
    )

    count_result = await db.execute(
        select(func.count()).select_from(AgentDecision)
    )

    total = count_result.scalar_one()

    result = await db.execute(
        select(AgentDecision)
        .order_by(AgentDecision.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    decisions = result.scalars().all()

    return PaginatedDecisions(
        items=[
            decision_response(decision)
            for decision in decisions
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/decisions/{decision_id}",
    response_model=AgentDecisionResponse,
)
async def get_agent_decision(
    decision_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> AgentDecisionResponse:
    require_role(
        current_user,
        {
            "admin",
            "district_officer",
            "emergency_coordinator",
            "operator",
            "reviewer",
        },
    )

    decision = await get_decision_or_404(
        db,
        decision_id,
    )

    return decision_response(decision)


# ============================================================
# HUMAN REVIEW
# ============================================================

@router.post(
    "/decisions/{decision_id}/review",
)
async def review_agent_decision(
    decision_id: str,
    payload: DecisionReviewRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        {
            "admin",
            "district_officer",
            "emergency_coordinator",
            "reviewer",
        },
    )

    decision = await get_decision_or_404(
        db,
        decision_id,
    )

    actor_id = user_identifier(current_user)

    current_status = enum_value(decision.status)

    if current_status not in {
        "pending_review",
        "pending_approval",
        "proposed",
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Decision cannot be reviewed in its current state",
        )

    if str(
        getattr(decision, "created_by_user_id", "")
    ) == actor_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Independent review is required",
        )

    # Delegate the state transition to the domain model/service.
    # The model's actual review method and column names must be
    # reconciled before enabling persistence.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Decision review requires integration with "
            "the AgentDecision domain workflow"
        ),
    )


# ============================================================
# EXECUTION
# ============================================================

@router.post(
    "/decisions/{decision_id}/execute",
)
async def execute_agent_decision(
    decision_id: str,
    payload: DecisionExecutionRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        {
            "admin",
            "district_officer",
            "emergency_coordinator",
        },
    )

    decision = await get_decision_or_404(
        db,
        decision_id,
    )

    if enum_value(decision.status) not in {
        "approved",
        "ready_for_execution",
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Decision is not approved for execution",
        )

    if payload.dry_run:
        return {
            "decision_id": decision.id,
            "dry_run": True,
            "execution_started": False,
            "message": (
                "Decision is eligible for execution, "
                "but no external action was performed"
            ),
        }

    # Real execution must go through an idempotent, audited,
    # allowlisted action dispatcher with per-action permissions.
    # Never execute arbitrary AI-generated tool instructions.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Approved action dispatcher is not configured",
    )


@router.post(
    "/decisions/{decision_id}/cancel",
)
async def cancel_agent_decision(
    decision_id: str,
    payload: DecisionCancelRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        {
            "admin",
            "district_officer",
            "emergency_coordinator",
        },
    )

    decision = await get_decision_or_404(
        db,
        decision_id,
    )

    if enum_value(decision.status) in {
        "executed",
        "completed",
        "cancelled",
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Decision cannot be cancelled",
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Cancellation requires integration with "
            "the AgentDecision domain workflow"
        ),
    )


# ============================================================
# EXECUTION HISTORY
# ============================================================

@router.get(
    "/executions/{execution_id}",
)
async def get_agent_execution(
    execution_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        {
            "admin",
            "district_officer",
            "emergency_coordinator",
            "operator",
            "reviewer",
        },
    )

    identifier = validate_uuid(execution_id)

    result = await db.execute(
        select(AgentExecution).where(
            AgentExecution.id == identifier
        )
    )

    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent execution not found",
        )

    return {
        "id": str(execution.id),
        "decision_id": str(execution.decision_id),
        "status": enum_value(execution.status),
        "created_at": getattr(
            execution,
            "created_at",
            None,
        ),
    }


__all__ = [
    "router",
    "AGENT_REGISTRY",
    "AgentInfo",
    "AgentHealthResponse",
    "AgentDecisionResponse",
    "DecisionReviewRequest",
    "DecisionExecutionRequest",
    "DecisionCancelRequest",
]