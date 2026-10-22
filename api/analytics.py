from __future__ import annotations
"""
Kerala Flood Relief Coordination Agent
backend/api/analytics.py

Analytics API for:
- Emergency operations dashboard
- SOS and incident statistics
- Rescue and evacuation analytics
- Shelter occupancy
- Volunteer deployment
- Inventory and relief delivery
- Flood alerts and rainfall monitoring
- District-level analytics
- Time-series analytics
- Operational KPI summaries

Python 3.11+ | FastAPI | SQLAlchemy 2.x
"""


import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy import (
    and_,
    case,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.district import District
from backend.models.flood_alert import FloodAlert
from backend.models.incident import Incident
from backend.models.rescue_mission import RescueMission
from backend.models.evacuation import Evacuation
from backend.models.shelter import Shelter
from backend.models.shelter_resident import ShelterResident
from backend.models.volunteer import Volunteer
from backend.models.inventory import InventoryItem
from backend.models.relief_delivery import ReliefDelivery
from backend.models.weather_station import (
    WeatherStation,
    RainfallReading,
)
from backend.models.sos_request import SOSRequest


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
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
# SCHEMAS
# ============================================================

class Metric(BaseModel):
    name: str
    value: int | float
    unit: str | None = None


class StatusCount(BaseModel):
    status: str
    count: int


class DistrictMetric(BaseModel):
    district_id: int
    district_name: str
    total: int


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: int | float


class DashboardResponse(BaseModel):
    generated_at: datetime
    district_id: int | None = None
    metrics: dict[str, int | float]
    warnings: list[str] = Field(default_factory=list)


class AnalyticsSummary(BaseModel):
    generated_at: datetime
    total: int
    by_status: list[StatusCount]


class RainfallSummary(BaseModel):
    generated_at: datetime
    district_id: int | None = None
    window_hours: int
    station_count: int
    reading_count: int
    maximum_reading_mm: float | None
    total_recorded_mm: float


class ShelterOccupancySummary(BaseModel):
    generated_at: datetime
    total_shelters: int
    current_residents: int
    residents_by_status: list[StatusCount]


# ============================================================
# SECURITY
# ============================================================

ALLOWED_ROLES = {
    "admin",
    "district_officer",
    "emergency_coordinator",
    "operator",
    "analyst",
    "reviewer",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def require_analytics_access(
    current_user: Any,
    district_id: int | None = None,
) -> None:
    if getattr(current_user, "is_superuser", False):
        return

    role = normalize_value(
        getattr(current_user, "role", "")
    ).lower()

    if role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analytics access denied",
        )

    user_district = getattr(
        current_user,
        "district_id",
        None,
    )

    if role == "district_officer":
        if user_district is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="District assignment required",
            )

        if district_id is not None and (
            int(user_district) != district_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cross-district analytics denied",
            )


def scoped_district(
    current_user: Any,
    district_id: int | None,
) -> int | None:
    require_analytics_access(
        current_user,
        district_id,
    )

    role = normalize_value(
        getattr(current_user, "role", "")
    ).lower()

    if (
        role == "district_officer"
        and not getattr(current_user, "is_superuser", False)
    ):
        return int(current_user.district_id)

    return district_id


# ============================================================
# DATABASE HELPERS
# ============================================================

async def count_records(
    db: AsyncSession,
    model: Any,
    *,
    district_id: int | None = None,
    filters: list[Any] | None = None,
) -> int:
    query = select(func.count()).select_from(model)

    conditions = list(filters or [])

    if district_id is not None:
        district_column = getattr(
            model,
            "district_id",
            None,
        )

        if district_column is None:
            raise ValueError(
                f"{model.__name__} has no district_id"
            )

        conditions.append(
            district_column == district_id
        )

    if conditions:
        query = query.where(and_(*conditions))

    result = await db.execute(query)

    return int(result.scalar_one() or 0)


async def group_by_status(
    db: AsyncSession,
    model: Any,
    *,
    district_id: int | None = None,
    extra_filters: list[Any] | None = None,
) -> list[StatusCount]:
    query = (
        select(
            model.status,
            func.count().label("total"),
        )
        .select_from(model)
    )

    conditions = list(extra_filters or [])

    if district_id is not None:
        district_column = getattr(
            model,
            "district_id",
            None,
        )

        if district_column is None:
            raise ValueError(
                f"{model.__name__} has no district_id"
            )

        conditions.append(
            district_column == district_id
        )

    if conditions:
        query = query.where(and_(*conditions))

    query = query.group_by(model.status)

    result = await db.execute(query)

    return [
        StatusCount(
            status=normalize_value(row[0]),
            count=int(row[1]),
        )
        for row in result.all()
    ]


def status_total(
    groups: list[StatusCount],
    statuses: set[str],
) -> int:
    return sum(
        item.count
        for item in groups
        if item.status in statuses
    )


# ============================================================
# DASHBOARD
# ============================================================

@router.get(
    "/dashboard",
    response_model=DashboardResponse,
)
async def get_dashboard(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> DashboardResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    incident_groups = await group_by_status(
        db,
        Incident,
        district_id=district_id,
    )

    rescue_groups = await group_by_status(
        db,
        RescueMission,
        district_id=district_id,
    )

    evacuation_groups = await group_by_status(
        db,
        Evacuation,
        district_id=district_id,
    )

    volunteer_groups = await group_by_status(
        db,
        Volunteer,
        district_id=district_id,
    )

    sos_groups = await group_by_status(
        db,
        SOSRequest,
        district_id=district_id,
    )

    total_shelters = await count_records(
        db,
        Shelter,
        district_id=district_id,
    )

    total_weather_stations = await count_records(
        db,
        WeatherStation,
        district_id=district_id,
    )

    metrics = {
        "total_incidents": sum(
            item.count for item in incident_groups
        ),
        "active_incidents": status_total(
            incident_groups,
            {
                "open",
                "active",
                "in_progress",
                "assigned",
            },
        ),
        "total_sos_requests": sum(
            item.count for item in sos_groups
        ),
        "pending_sos_requests": status_total(
            sos_groups,
            {
                "pending",
                "received",
                "new",
                "verified",
            },
        ),
        "total_rescue_missions": sum(
            item.count for item in rescue_groups
        ),
        "active_rescue_missions": status_total(
            rescue_groups,
            {
                "active",
                "dispatched",
                "in_progress",
            },
        ),
        "total_evacuations": sum(
            item.count for item in evacuation_groups
        ),
        "active_evacuations": status_total(
            evacuation_groups,
            {
                "active",
                "in_progress",
                "ongoing",
            },
        ),
        "total_shelters": total_shelters,
        "total_volunteers": sum(
            item.count for item in volunteer_groups
        ),
        "deployed_volunteers": status_total(
            volunteer_groups,
            {
                "deployed",
                "on_duty",
            },
        ),
        "weather_stations": total_weather_stations,
    }

    return DashboardResponse(
        generated_at=utc_now(),
        district_id=district_id,
        metrics=metrics,
    )


# ============================================================
# SOS ANALYTICS
# ============================================================

@router.get(
    "/sos",
    response_model=AnalyticsSummary,
)
async def sos_analytics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> AnalyticsSummary:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    groups = await group_by_status(
        db,
        SOSRequest,
        district_id=district_id,
    )

    return AnalyticsSummary(
        generated_at=utc_now(),
        total=sum(x.count for x in groups),
        by_status=groups,
    )


# ============================================================
# INCIDENT ANALYTICS
# ============================================================

@router.get(
    "/incidents",
    response_model=AnalyticsSummary,
)
async def incident_analytics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> AnalyticsSummary:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    groups = await group_by_status(
        db,
        Incident,
        district_id=district_id,
    )

    return AnalyticsSummary(
        generated_at=utc_now(),
        total=sum(x.count for x in groups),
        by_status=groups,
    )


# ============================================================
# RESCUE ANALYTICS
# ============================================================

@router.get(
    "/rescue",
    response_model=AnalyticsSummary,
)
async def rescue_analytics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> AnalyticsSummary:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    groups = await group_by_status(
        db,
        RescueMission,
        district_id=district_id,
    )

    return AnalyticsSummary(
        generated_at=utc_now(),
        total=sum(x.count for x in groups),
        by_status=groups,
    )


# ============================================================
# EVACUATION ANALYTICS
# ============================================================

@router.get(
    "/evacuation",
    response_model=AnalyticsSummary,
)
async def evacuation_analytics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> AnalyticsSummary:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    groups = await group_by_status(
        db,
        Evacuation,
        district_id=district_id,
    )

    return AnalyticsSummary(
        generated_at=utc_now(),
        total=sum(x.count for x in groups),
        by_status=groups,
    )


# ============================================================
# VOLUNTEER ANALYTICS
# ============================================================

@router.get(
    "/volunteers",
    response_model=AnalyticsSummary,
)
async def volunteer_analytics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> AnalyticsSummary:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    groups = await group_by_status(
        db,
        Volunteer,
        district_id=district_id,
    )

    return AnalyticsSummary(
        generated_at=utc_now(),
        total=sum(x.count for x in groups),
        by_status=groups,
    )


# ============================================================
# SHELTER ANALYTICS
# ============================================================

@router.get(
    "/shelters",
    response_model=ShelterOccupancySummary,
)
async def shelter_analytics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> ShelterOccupancySummary:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    total_shelters = await count_records(
        db,
        Shelter,
        district_id=district_id,
    )

    groups = await group_by_status(
        db,
        ShelterResident,
        district_id=district_id,
    )

    current_residents = status_total(
        groups,
        {
            "admitted",
            "active",
            "temporarily_away",
        },
    )

    return ShelterOccupancySummary(
        generated_at=utc_now(),
        total_shelters=total_shelters,
        current_residents=current_residents,
        residents_by_status=groups,
    )


# ============================================================
# RAINFALL ANALYTICS
# ============================================================

@router.get(
    "/rainfall",
    response_model=RainfallSummary,
)
async def rainfall_analytics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
    ),
) -> RainfallSummary:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    cutoff = utc_now() - timedelta(hours=hours)

    query = (
        select(
            func.count(RainfallReading.id),
            func.count(
                func.distinct(
                    RainfallReading.station_id
                )
            ),
            func.max(
                RainfallReading.rainfall_mm
            ),
            func.sum(
                RainfallReading.rainfall_mm
            ),
        )
        .join(
            WeatherStation,
            RainfallReading.station_id
            == WeatherStation.id,
        )
        .where(
            RainfallReading.observed_at >= cutoff
        )
    )

    if district_id is not None:
        query = query.where(
            WeatherStation.district_id == district_id
        )

    result = await db.execute(query)

    row = result.one()

    return RainfallSummary(
        generated_at=utc_now(),
        district_id=district_id,
        window_hours=hours,
        reading_count=int(row[0] or 0),
        station_count=int(row[1] or 0),
        maximum_reading_mm=(
            float(row[2])
            if row[2] is not None
            else None
        ),
        total_recorded_mm=float(row[3] or 0),
    )


# ============================================================
# DISTRICT ANALYTICS
# ============================================================

@router.get(
    "/districts/incidents",
    response_model=list[DistrictMetric],
)
async def incidents_by_district(
    db: DbSession,
    current_user: CurrentUser,
) -> list[DistrictMetric]:
    district_id = scoped_district(
        current_user,
        None,
    )

    query = (
        select(
            District.id,
            District.name,
            func.count(Incident.id),
        )
        .outerjoin(
            Incident,
            Incident.district_id == District.id,
        )
        .group_by(
            District.id,
            District.name,
        )
        .order_by(
            func.count(Incident.id).desc()
        )
    )

    if district_id is not None:
        query = query.where(
            District.id == district_id
        )

    result = await db.execute(query)

    return [
        DistrictMetric(
            district_id=int(row[0]),
            district_name=str(row[1]),
            total=int(row[2]),
        )
        for row in result.all()
    ]


# ============================================================
# TIME SERIES
# ============================================================

@router.get(
    "/timeseries/incidents",
    response_model=list[TimeSeriesPoint],
)
async def incident_timeseries(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
    days: int = Query(
        default=7,
        ge=1,
        le=90,
    ),
) -> list[TimeSeriesPoint]:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    cutoff = utc_now() - timedelta(days=days)

    # Database-independent aggregation:
    # group timestamps into UTC calendar dates in Python.
    query = select(
        Incident.created_at
    ).where(
        Incident.created_at >= cutoff
    )

    if district_id is not None:
        query = query.where(
            Incident.district_id == district_id
        )

    result = await db.execute(query)

    daily_counts: dict[str, int] = {}

    for observed_at in result.scalars():
        if observed_at is None:
            continue

        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(
                tzinfo=timezone.utc
            )

        day = (
            observed_at
            .astimezone(timezone.utc)
            .date()
            .isoformat()
        )

        daily_counts[day] = (
            daily_counts.get(day, 0) + 1
        )

    return [
        TimeSeriesPoint(
            timestamp=datetime.fromisoformat(
                f"{day}T00:00:00+00:00"
            ),
            value=count,
        )
        for day, count in sorted(
            daily_counts.items()
        )
    ]


# ============================================================
# OPERATIONAL SUMMARY
# ============================================================

@router.get(
    "/operations/summary",
)
async def operations_summary(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> dict[str, Any]:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    models = {
        "incidents": Incident,
        "sos_requests": SOSRequest,
        "rescue_missions": RescueMission,
        "evacuations": Evacuation,
        "shelters": Shelter,
        "volunteers": Volunteer,
        "weather_stations": WeatherStation,
    }

    totals: dict[str, int] = {}

    for name, model in models.items():
        totals[name] = await count_records(
            db,
            model,
            district_id=district_id,
        )

    return {
        "generated_at": utc_now().isoformat(),
        "district_id": district_id,
        "totals": totals,
        "data_source": "operational_database",
        "note": (
            "Counts reflect stored records and may not "
            "represent independently verified live conditions."
        ),
    }


__all__ = [
    "router",
    "DashboardResponse",
    "AnalyticsSummary",
    "RainfallSummary",
    "ShelterOccupancySummary",
    "DistrictMetric",
    "TimeSeriesPoint",
]