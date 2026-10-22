from __future__ import annotations
"""
backend/api/dams.py

Kerala Flood Relief Coordination Agent
Dam Monitoring and Reservoir Intelligence API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Endpoints:
    GET  /dams
    GET  /dams/statistics
    GET  /dams/{dam_id}
    GET  /dams/{dam_id/water-levels
    POST /dams/{dam_id}/water-levels
    GET  /dams/{dam_id}/gates
    GET  /dams/{dam_id}/gates/{gate_id}/observations
    POST /dams/{dam_id}/gates/{gate_id}/observations
    GET  /dams/{dam_id}/alerts
    GET  /dams/{dam_id}/advisories

Important:
    - All AI predictions are advisory.
    - Gate-control commands are never exposed.
    - Alert publication requires separate human authorization.
    - Sensor readings are not automatically verified.
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
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.dam import (
    Dam,
    DamWaterLevelReading,
    DamGate,
    DamGateObservation,
    DamAlert,
    DamOperationalAdvisory,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dams",
    tags=["Dam Monitoring"],
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

READ_ROLES = {
    "admin",
    "super_admin",
    "district_officer",
    "emergency_coordinator",
    "dam_operator",
    "hydrologist",
    "analyst",
    "reviewer",
}

WRITE_ROLES = {
    "admin",
    "super_admin",
    "dam_operator",
    "hydrologist",
}

MAX_PAGE_SIZE = 100


# ============================================================
# SCHEMAS
# ============================================================

class DamResponse(BaseModel):
    id: str
    district_id: int | None = None
    name: str | None = None
    status: str | None = None


class DamListResponse(BaseModel):
    items: list[DamResponse]
    total: int
    limit: int
    offset: int


class WaterLevelReadingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    water_level_m: float = Field(ge=0)
    source: str = Field(
        min_length=2,
        max_length=80,
    )
    inflow_cumecs: float | None = Field(
        default=None,
        ge=0,
    )
    outflow_cumecs: float | None = Field(
        default=None,
        ge=0,
    )


class GateObservationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    opening_percentage: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    opening_height_m: float | None = Field(
        default=None,
        ge=0,
    )
    discharge_cumecs: float | None = Field(
        default=None,
        ge=0,
    )
    notes: str | None = Field(
        default=None,
        max_length=2000,
    )


class DamStatisticsResponse(BaseModel):
    total_dams: int
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
            detail="Invalid identifier",
        ) from None


def require_role(
    user: Any,
    roles: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    role = enum_value(
        getattr(user, "role", "")
    ).lower()

    if role not in roles:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions",
        )


def require_district_access(
    user: Any,
    district_id: int | None,
) -> None:
    if getattr(user, "is_superuser", False):
        return

    role = enum_value(
        getattr(user, "role", "")
    ).lower()

    if role in {"admin", "super_admin"}:
        return

    user_district = getattr(
        user,
        "district_id",
        None,
    )

    if (
        district_id is None
        or user_district is None
        or int(user_district) != int(district_id)
    ):
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )


def dam_response(dam: Dam) -> DamResponse:
    return DamResponse(
        id=str(dam.id),
        district_id=getattr(
            dam,
            "district_id",
            None,
        ),
        name=getattr(
            dam,
            "name",
            None,
        ),
        status=enum_value(
            getattr(dam, "status", None)
        ),
    )


async def get_dam_or_404(
    db: AsyncSession,
    dam_id: str,
    user: Any,
) -> Dam:
    result = await db.execute(
        select(Dam).where(
            Dam.id == validate_uuid(dam_id)
        )
    )

    dam = result.scalar_one_or_none()

    if dam is None:
        raise HTTPException(
            status_code=404,
            detail="Dam not found",
        )

    require_district_access(
        user,
        getattr(dam, "district_id", None),
    )

    return dam


async def get_gate_or_404(
    db: AsyncSession,
    dam_id: str,
    gate_id: str,
) -> DamGate:
    result = await db.execute(
        select(DamGate).where(
            DamGate.id == validate_uuid(gate_id),
            DamGate.dam_id == validate_uuid(dam_id),
        )
    )

    gate = result.scalar_one_or_none()

    if gate is None:
        raise HTTPException(
            status_code=404,
            detail="Dam gate not found",
        )

    return gate


# ============================================================
# LIST DAMS
# ============================================================

@router.get(
    "",
    response_model=DamListResponse,
)
async def list_dams(
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
) -> DamListResponse:
    require_role(current_user, READ_ROLES)

    role = enum_value(
        getattr(current_user, "role", "")
    ).lower()

    if (
        not getattr(current_user, "is_superuser", False)
        and role not in {"admin", "super_admin"}
    ):
        user_district = getattr(
            current_user,
            "district_id",
            None,
        )

        if user_district is None:
            raise HTTPException(
                status_code=403,
                detail="District assignment required",
            )

        if (
            district_id is not None
            and district_id != int(user_district)
        ):
            raise HTTPException(
                status_code=403,
                detail="District access denied",
            )

        district_id = int(user_district)

    query = select(Dam)
    count_query = select(func.count()).select_from(Dam)

    if district_id is not None:
        query = query.where(
            Dam.district_id == district_id
        )
        count_query = count_query.where(
            Dam.district_id == district_id
        )

    total_result = await db.execute(count_query)

    result = await db.execute(
        query.offset(offset).limit(limit)
    )

    return DamListResponse(
        items=[
            dam_response(dam)
            for dam in result.scalars().all()
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
    response_model=DamStatisticsResponse,
)
async def dam_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> DamStatisticsResponse:
    require_role(current_user, READ_ROLES)

    role = enum_value(
        getattr(current_user, "role", "")
    ).lower()

    if (
        not getattr(current_user, "is_superuser", False)
        and role not in {"admin", "super_admin"}
    ):
        user_district = getattr(
            current_user,
            "district_id",
            None,
        )

        if user_district is None:
            raise HTTPException(
                status_code=403,
                detail="District assignment required",
            )

        if (
            district_id is not None
            and district_id != int(user_district)
        ):
            raise HTTPException(
                status_code=403,
                detail="District access denied",
            )

        district_id = int(user_district)

    query = select(
        Dam.status,
        func.count(Dam.id),
    )

    if district_id is not None:
        query = query.where(
            Dam.district_id == district_id
        )

    query = query.group_by(Dam.status)

    result = await db.execute(query)

    by_status = {
        enum_value(row[0]): int(row[1])
        for row in result.all()
    }

    return DamStatisticsResponse(
        total_dams=sum(by_status.values()),
        by_status=by_status,
        generated_at=utc_now(),
    )


# ============================================================
# GET DAM
# ============================================================

@router.get(
    "/{dam_id}",
    response_model=DamResponse,
)
async def get_dam(
    dam_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> DamResponse:
    require_role(current_user, READ_ROLES)

    dam = await get_dam_or_404(
        db,
        dam_id,
        current_user,
    )

    return dam_response(dam)


# ============================================================
# WATER LEVELS
# ============================================================

@router.get("/{dam_id}/water-levels")
async def list_water_levels(
    dam_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_dam_or_404(
        db,
        dam_id,
        current_user,
    )

    result = await db.execute(
        select(DamWaterLevelReading)
        .where(
            DamWaterLevelReading.dam_id
            == validate_uuid(dam_id)
        )
        .order_by(
            DamWaterLevelReading.observed_at.desc()
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "dam_id": str(item.dam_id),
            "observed_at": item.observed_at,
            "water_level_m": getattr(
                item,
                "water_level_m",
                None,
            ),
        }
        for item in result.scalars().all()
    ]


@router.post(
    "/{dam_id}/water-levels",
    status_code=201,
)
async def create_water_level(
    dam_id: str,
    payload: WaterLevelReadingCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_dam_or_404(
        db,
        dam_id,
        current_user,
    )

    # Observation creation must use the exact model fields,
    # trusted sensor ingestion, deduplication and audit trail.
    raise HTTPException(
        status_code=501,
        detail="Water-level ingestion service is not configured",
    )


# ============================================================
# GATES
# ============================================================

@router.get("/{dam_id}/gates")
async def list_dam_gates(
    dam_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_dam_or_404(
        db,
        dam_id,
        current_user,
    )

    result = await db.execute(
        select(DamGate).where(
            DamGate.dam_id == validate_uuid(dam_id)
        )
    )

    return [
        {
            "id": str(gate.id),
            "dam_id": str(gate.dam_id),
            "status": enum_value(
                getattr(gate, "status", None)
            ),
        }
        for gate in result.scalars().all()
    ]


@router.get(
    "/{dam_id}/gates/{gate_id}/observations",
)
async def list_gate_observations(
    dam_id: str,
    gate_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_dam_or_404(
        db,
        dam_id,
        current_user,
    )

    await get_gate_or_404(
        db,
        dam_id,
        gate_id,
    )

    result = await db.execute(
        select(DamGateObservation)
        .where(
            DamGateObservation.gate_id
            == validate_uuid(gate_id)
        )
        .order_by(
            DamGateObservation.observed_at.desc()
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "gate_id": str(item.gate_id),
            "observed_at": item.observed_at,
        }
        for item in result.scalars().all()
    ]


@router.post(
    "/{dam_id}/gates/{gate_id}/observations",
    status_code=201,
)
async def create_gate_observation(
    dam_id: str,
    gate_id: str,
    payload: GateObservationCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_dam_or_404(
        db,
        dam_id,
        current_user,
    )

    await get_gate_or_404(
        db,
        dam_id,
        gate_id,
    )

    # Observation only. No physical gate actuation.
    raise HTTPException(
        status_code=501,
        detail="Gate observation ingestion is not configured",
    )


# ============================================================
# ALERTS
# ============================================================

@router.get("/{dam_id}/alerts")
async def list_dam_alerts(
    dam_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_dam_or_404(
        db,
        dam_id,
        current_user,
    )

    result = await db.execute(
        select(DamAlert)
        .where(
            DamAlert.dam_id == validate_uuid(dam_id)
        )
        .limit(limit)
    )

    return [
        {
            "id": str(alert.id),
            "dam_id": str(alert.dam_id),
            "status": enum_value(
                getattr(alert, "status", None)
            ),
            "severity": enum_value(
                getattr(alert, "severity", None)
            ),
        }
        for alert in result.scalars().all()
    ]


# ============================================================
# OPERATIONAL ADVISORIES
# ============================================================

@router.get("/{dam_id}/advisories")
async def list_dam_advisories(
    dam_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_dam_or_404(
        db,
        dam_id,
        current_user,
    )

    result = await db.execute(
        select(DamOperationalAdvisory)
        .where(
            DamOperationalAdvisory.dam_id
            == validate_uuid(dam_id)
        )
        .limit(limit)
    )

    return [
        {
            "id": str(advisory.id),
            "dam_id": str(advisory.dam_id),
            "status": enum_value(
                getattr(advisory, "status", None)
            ),
        }
        for advisory in result.scalars().all()
    ]


__all__ = [
    "router",
    "DamResponse",
    "DamListResponse",
    "WaterLevelReadingCreate",
    "GateObservationCreate",
    "DamStatisticsResponse",
]