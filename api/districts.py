from __future__ import annotations

"""
backend/api/districts.py

Kerala Flood Relief Coordination Agent
District Administration & Flood Risk Intelligence API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Endpoints:
    GET    /districts
    GET    /districts/statistics
    GET    /districts/{district_id}
    GET    /districts/{district_id}/divisions
    GET    /districts/{district_id}/risk-profile
    GET    /districts/{district_id}/risk-history
    GET    /districts/{district_id}/resources
    GET    /districts/{district_id}/resource-history
    GET    /districts/{district_id}/status-history
    POST   /districts/{district_id}/risk-profile
    POST   /districts/{district_id}/resources
    POST   /districts/{district_id}/status

Notes:
    - District IDs are integers.
    - Operational records require authenticated access.
    - District-scoped roles cannot access other districts.
    - AI-generated flood-risk assessments are advisory.
    - Official warnings require authorized human review.
    - Write operations require integration with validated
      domain services before deployment.
"""


import logging

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
)

from sqlalchemy import (
    func,
    select,
)

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.district import (
    District,
    DistrictAdministrativeDivision,
    DistrictFloodRiskProfile,
    DistrictResourceSnapshot,
    DistrictStatusHistory,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/districts",
    tags=["District Administration"],
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

DISTRICT_ROLES = {
    "district_officer",
    "emergency_coordinator",
}

ANALYST_ROLES = {
    "analyst",
    "hydrologist",
    "risk_analyst",
}

OPERATOR_ROLES = {
    "operator",
    "field_officer",
}

READ_ROLES = (
    ADMIN_ROLES
    | DISTRICT_ROLES
    | ANALYST_ROLES
    | OPERATOR_ROLES
)

WRITE_ROLES = (
    ADMIN_ROLES
    | DISTRICT_ROLES
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class FloodRiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class DistrictOperationalStatus(str, Enum):
    NORMAL = "normal"
    MONITORING = "monitoring"
    ALERT = "alert"
    EMERGENCY = "emergency"
    RECOVERY = "recovery"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class DistrictRiskProfileCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    risk_level: FloodRiskLevel

    risk_score: float = Field(
        ge=0,
        le=100,
    )

    rainfall_risk_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    river_risk_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    landslide_risk_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    population_exposure_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    assessment_notes: str | None = Field(
        default=None,
        max_length=5000,
    )

    source_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    ai_generated: bool = False

    model_version: str | None = Field(
        default=None,
        max_length=150,
    )


class DistrictResourceSnapshotCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available_rescue_teams: int = Field(ge=0)

    available_ambulances: int = Field(ge=0)

    available_boats: int = Field(ge=0)

    available_shelters: int = Field(ge=0)

    available_volunteers: int = Field(ge=0)

    available_food_kits: int = Field(ge=0)

    available_water_liters: float = Field(ge=0)

    recorded_at: datetime

    source_reference: str | None = Field(
        default=None,
        max_length=500,
    )


class DistrictStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DistrictOperationalStatus

    reason: str = Field(
        min_length=10,
        max_length=5000,
    )

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class DistrictResponse(BaseModel):
    id: int
    name: str
    code: str | None = None
    status: str | None = None


class DistrictListResponse(BaseModel):
    items: list[DistrictResponse]
    total: int
    limit: int
    offset: int


class DistrictDivisionResponse(BaseModel):
    id: str
    district_id: int
    name: str | None = None
    division_type: str | None = None


class DistrictRiskResponse(BaseModel):
    id: str
    district_id: int
    risk_level: str | None = None
    risk_score: float | None = None
    created_at: datetime | None = None


class DistrictResourceResponse(BaseModel):
    id: str
    district_id: int
    recorded_at: datetime | None = None


class DistrictStatusHistoryResponse(BaseModel):
    id: str
    district_id: int
    status: str | None = None
    created_at: datetime | None = None


class DistrictStatisticsResponse(BaseModel):
    total_districts: int
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


def role_of(user: Any) -> str:
    return enum_value(
        getattr(user, "role", "")
    ).lower()


def require_role(
    user: Any,
    roles: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
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

    if assigned_district is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="District assignment required",
        )

    if int(assigned_district) != district_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
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


async def get_district_or_404(
    db: AsyncSession,
    district_id: int,
    user: Any,
) -> District:
    require_district_access(
        user,
        district_id,
    )

    result = await db.execute(
        select(District).where(
            District.id == district_id
        )
    )

    district = result.scalar_one_or_none()

    if district is None:
        raise HTTPException(
            status_code=404,
            detail="District not found",
        )

    return district


def district_response(
    district: District,
) -> DistrictResponse:
    return DistrictResponse(
        id=district.id,
        name=str(
            getattr(
                district,
                "name",
                getattr(
                    district,
                    "district_name",
                    "",
                ),
            )
        ),
        code=getattr(
            district,
            "code",
            None,
        ),
        status=enum_value(
            getattr(
                district,
                "status",
                None,
            )
        ) or None,
    )


# ============================================================
# LIST DISTRICTS
# ============================================================

@router.get(
    "",
    response_model=DistrictListResponse,
)
async def list_districts(
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
) -> DistrictListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(District)

    count_query = (
        select(func.count())
        .select_from(District)
    )

    if district_id is not None:
        query = query.where(
            District.id == district_id
        )

        count_query = count_query.where(
            District.id == district_id
        )

    total_result = await db.execute(
        count_query
    )

    result = await db.execute(
        query.order_by(District.id)
        .offset(offset)
        .limit(limit)
    )

    return DistrictListResponse(
        items=[
            district_response(district)
            for district in result.scalars().all()
        ],
        total=int(total_result.scalar_one()),
        limit=limit,
        offset=offset,
    )


# ============================================================
# DISTRICT STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=DistrictStatisticsResponse,
)
async def district_statistics(
    db: DbSession,
    current_user: CurrentUser,
) -> DistrictStatisticsResponse:
    district_id = scoped_district(
        current_user,
        None,
    )

    query = select(District)

    if district_id is not None:
        query = query.where(
            District.id == district_id
        )

    result = await db.execute(query)

    districts = result.scalars().all()

    by_status: dict[str, int] = {}

    for district in districts:
        current_status = (
            enum_value(
                getattr(
                    district,
                    "status",
                    None,
                )
            )
            or "unknown"
        )

        by_status[current_status] = (
            by_status.get(
                current_status,
                0,
            ) + 1
        )

    return DistrictStatisticsResponse(
        total_districts=len(districts),
        by_status=by_status,
        generated_at=utc_now(),
    )


# ============================================================
# DISTRICT DETAILS
# ============================================================

@router.get(
    "/{district_id}",
    response_model=DistrictResponse,
)
async def get_district(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
) -> DistrictResponse:
    require_role(current_user, READ_ROLES)

    district = await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    return district_response(district)


# ============================================================
# ADMINISTRATIVE DIVISIONS
# ============================================================

@router.get(
    "/{district_id}/divisions",
    response_model=list[DistrictDivisionResponse],
)
async def list_district_divisions(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
) -> list[DistrictDivisionResponse]:
    require_role(current_user, READ_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    result = await db.execute(
        select(DistrictAdministrativeDivision)
        .where(
            DistrictAdministrativeDivision.district_id
            == district_id
        )
        .limit(limit)
    )

    return [
        DistrictDivisionResponse(
            id=str(item.id),
            district_id=item.district_id,
            name=getattr(
                item,
                "name",
                None,
            ),
            division_type=enum_value(
                getattr(
                    item,
                    "division_type",
                    None,
                )
            ) or None,
        )
        for item in result.scalars().all()
    ]


# ============================================================
# FLOOD RISK PROFILE
# ============================================================

@router.get(
    "/{district_id}/risk-profile",
)
async def get_latest_risk_profile(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, READ_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    result = await db.execute(
        select(DistrictFloodRiskProfile)
        .where(
            DistrictFloodRiskProfile.district_id
            == district_id
        )
        .order_by(
            DistrictFloodRiskProfile.created_at.desc()
        )
        .limit(1)
    )

    profile = result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Risk profile not available",
        )

    return {
        "id": str(profile.id),
        "district_id": district_id,
        "risk_level": enum_value(
            getattr(
                profile,
                "risk_level",
                None,
            )
        ),
        "risk_score": getattr(
            profile,
            "risk_score",
            None,
        ),
        "created_at": profile.created_at,
        "advisory_only": True,
    }


@router.get(
    "/{district_id}/risk-history",
)
async def get_risk_history(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=30,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    result = await db.execute(
        select(DistrictFloodRiskProfile)
        .where(
            DistrictFloodRiskProfile.district_id
            == district_id
        )
        .order_by(
            DistrictFloodRiskProfile.created_at.desc()
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "district_id": district_id,
            "risk_level": enum_value(
                getattr(
                    item,
                    "risk_level",
                    None,
                )
            ),
            "risk_score": getattr(
                item,
                "risk_score",
                None,
            ),
            "created_at": item.created_at,
        }
        for item in result.scalars().all()
    ]


@router.post(
    "/{district_id}/risk-profile",
    status_code=201,
)
async def create_risk_profile(
    district_id: int,
    payload: DistrictRiskProfileCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    # Requires model-specific field mapping, evidence
    # validation and authorized human approval.
    raise HTTPException(
        status_code=501,
        detail=(
            "Risk profile creation service "
            "is not configured"
        ),
    )


# ============================================================
# DISTRICT RESOURCES
# ============================================================

@router.get(
    "/{district_id}/resources",
)
async def get_latest_resources(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, READ_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    result = await db.execute(
        select(DistrictResourceSnapshot)
        .where(
            DistrictResourceSnapshot.district_id
            == district_id
        )
        .order_by(
            DistrictResourceSnapshot.created_at.desc()
        )
        .limit(1)
    )

    snapshot = result.scalar_one_or_none()

    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail="Resource snapshot not available",
        )

    return {
        "id": str(snapshot.id),
        "district_id": district_id,
        "created_at": snapshot.created_at,
        "source": "district_resource_snapshot",
    }


@router.get(
    "/{district_id}/resource-history",
)
async def resource_history(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=30,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    result = await db.execute(
        select(DistrictResourceSnapshot)
        .where(
            DistrictResourceSnapshot.district_id
            == district_id
        )
        .order_by(
            DistrictResourceSnapshot.created_at.desc()
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "district_id": district_id,
            "created_at": item.created_at,
        }
        for item in result.scalars().all()
    ]


@router.post(
    "/{district_id}/resources",
    status_code=201,
)
async def create_resource_snapshot(
    district_id: int,
    payload: DistrictResourceSnapshotCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    raise HTTPException(
        status_code=501,
        detail=(
            "Resource snapshot ingestion "
            "service is not configured"
        ),
    )


# ============================================================
# DISTRICT STATUS HISTORY
# ============================================================

@router.get(
    "/{district_id}/status-history",
)
async def district_status_history(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    result = await db.execute(
        select(DistrictStatusHistory)
        .where(
            DistrictStatusHistory.district_id
            == district_id
        )
        .order_by(
            DistrictStatusHistory.created_at.desc()
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "district_id": district_id,
            "status": enum_value(
                getattr(
                    item,
                    "status",
                    None,
                )
            ),
            "created_at": item.created_at,
        }
        for item in result.scalars().all()
    ]


@router.post(
    "/{district_id}/status",
)
async def update_district_status(
    district_id: int,
    payload: DistrictStatusUpdate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_district_or_404(
        db,
        district_id,
        current_user,
    )

    # Status changes require a validated domain transition,
    # authorization checks, and an immutable audit record.
    raise HTTPException(
        status_code=501,
        detail=(
            "District operational status workflow "
            "is not configured"
        ),
    )


__all__ = [
    "router",
    "DistrictResponse",
    "DistrictListResponse",
    "DistrictDivisionResponse",
    "DistrictRiskResponse",
    "DistrictResourceResponse",
    "DistrictStatusHistoryResponse",
    "DistrictStatisticsResponse",
    "DistrictRiskProfileCreate",
    "DistrictResourceSnapshotCreate",
    "DistrictStatusUpdate",
]
