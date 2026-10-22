from __future__ import annotations

"""
backend/api/rainfall.py

Kerala Flood Relief Coordination Agent
Rainfall Monitoring & Weather Station API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Endpoints:
    GET  /rainfall/stations
    GET  /rainfall/statistics
    GET  /rainfall/districts/{district_id}/summary
    GET  /rainfall/stations/{station_id}
    GET  /rainfall/stations/{station_id}/readings
    POST /rainfall/stations/{station_id}/readings
    GET  /rainfall/stations/{station_id}/observations
    GET  /rainfall/stations/{station_id}/health
    GET  /rainfall/stations/{station_id}/alerts
    GET  /rainfall/stations/{station_id}/thresholds
    POST /rainfall/stations/{station_id}/thresholds
    GET  /rainfall/stations/{station_id}/maintenance
    GET  /rainfall/stations/{station_id}/events
    POST /rainfall/forecast

Safety:
    - Rainfall observations must come from verified sources.
    - Unverified readings must not be presented as official.
    - Sensor faults and stale data must be identified.
    - Threshold exceedance does not automatically
      constitute an official flood warning.
    - Forecasts are advisory.
"""


import logging
import uuid

from datetime import datetime, timedelta, timezone
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

from sqlalchemy import (
    func,
    select,
)

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.weather_station import (
    WeatherStation,
    WeatherObservation,
    RainfallReading,
    WeatherThreshold,
    WeatherStationAlert,
    WeatherStationHealth,
    WeatherStationMaintenance,
    WeatherStationEvent,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rainfall",
    tags=["Rainfall Monitoring"],
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

WEATHER_ROLES = {
    "meteorologist",
    "weather_operator",
    "hydrologist",
}

COMMAND_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "risk_analyst",
}

READ_ROLES = (
    ADMIN_ROLES
    | WEATHER_ROLES
    | COMMAND_ROLES
    | {"operator", "analyst"}
)

WRITE_ROLES = (
    ADMIN_ROLES
    | WEATHER_ROLES
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class RainfallIntensity(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    VERY_HEAVY = "very_heavy"
    EXTREME = "extreme"


class ReadingSource(str, Enum):
    AUTOMATIC_STATION = "automatic_station"
    MANUAL_OBSERVATION = "manual_observation"
    VERIFIED_INTEGRATION = "verified_integration"


class ReadingQuality(str, Enum):
    UNVERIFIED = "unverified"
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"


class SensorOperationalStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class RainfallReadingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime

    rainfall_mm: float = Field(
        ge=0,
        le=5000,
    )

    measurement_period_minutes: int = Field(
        ge=1,
        le=1440,
    )

    source: ReadingSource

    source_reference: str = Field(
        min_length=3,
        max_length=500,
    )

    quality: ReadingQuality = (
        ReadingQuality.UNVERIFIED
    )

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
                "Observation time cannot be in the future"
            )

        return value


class WeatherThresholdCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold_name: str = Field(
        min_length=3,
        max_length=150,
    )

    rainfall_mm: float = Field(
        gt=0,
        le=5000,
    )

    duration_minutes: int = Field(
        ge=1,
        le=10080,
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    effective_from: datetime

    effective_until: datetime | None = None

    @model_validator(mode="after")
    def validate_dates(
        self,
    ) -> "WeatherThresholdCreate":
        if (
            self.effective_until is not None
            and self.effective_until
            <= self.effective_from
        ):
            raise ValueError(
                "Threshold end must be after start"
            )

        return self


class RainfallForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    district_id: int = Field(ge=1)

    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=72,
    )

    include_uncertainty: bool = True


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class WeatherStationResponse(BaseModel):
    id: str

    district_id: int

    name: str | None = None

    status: str | None = None


class WeatherStationListResponse(BaseModel):
    items: list[WeatherStationResponse]

    total: int

    limit: int

    offset: int


class RainfallReadingResponse(BaseModel):
    id: str

    station_id: str

    rainfall_mm: float | None = None

    observed_at: datetime | None = None

    quality: str | None = None


class WeatherObservationResponse(BaseModel):
    id: str

    station_id: str

    observed_at: datetime | None = None


class WeatherThresholdResponse(BaseModel):
    id: str

    station_id: str


class WeatherAlertResponse(BaseModel):
    id: str

    station_id: str

    status: str | None = None


class WeatherHealthResponse(BaseModel):
    id: str

    station_id: str

    status: str | None = None


class WeatherMaintenanceResponse(BaseModel):
    id: str

    station_id: str


class RainfallStatisticsResponse(BaseModel):
    total_stations: int

    total_readings: int

    generated_at: datetime


class DistrictRainfallSummary(BaseModel):
    district_id: int

    station_count: int

    reading_count: int

    total_recorded_mm: float

    period_start: datetime

    period_end: datetime

    generated_at: datetime


class RainfallForecastResponse(BaseModel):
    district_id: int

    horizon_hours: int

    predicted_total_mm: float = Field(ge=0)

    model_reference: str

    generated_at: datetime

    data_cutoff_at: datetime

    advisory_only: bool = True


# ============================================================
# FORECAST SERVICE CONTRACT
# ============================================================

class RainfallForecastService(Protocol):

    async def forecast_rainfall(
        self,
        district_id: int,
        horizon_hours: int,
        include_uncertainty: bool,
        actor_id: str,
    ) -> RainfallForecastResponse:
        ...


def get_forecast_service(
    request: Request,
) -> RainfallForecastService:
    service = getattr(
        request.app.state,
        "rainfall_forecast_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Rainfall forecasting service is not configured",
        )

    return service


ForecastServiceDependency = Annotated[
    RainfallForecastService,
    Depends(get_forecast_service),
]


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


def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return str(identifier)


def role_of(user: Any) -> str:
    return enum_value(
        getattr(user, "role", "")
    ).lower()


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


async def get_station_or_404(
    db: AsyncSession,
    station_id: str,
    user: Any,
) -> WeatherStation:
    identifier = validate_uuid(station_id)

    result = await db.execute(
        select(WeatherStation).where(
            WeatherStation.id == identifier
        )
    )

    station = result.scalar_one_or_none()

    if station is None:
        raise HTTPException(
            status_code=404,
            detail="Weather station not found",
        )

    require_district_access(
        user,
        station.district_id,
    )

    return station


def station_response(
    station: WeatherStation,
) -> WeatherStationResponse:
    return WeatherStationResponse(
        id=str(station.id),
        district_id=station.district_id,
        name=getattr(
            station,
            "name",
            None,
        ),
        status=enum_value(
            getattr(station, "status", None)
        ) or None,
    )


# ============================================================
# WEATHER STATIONS
# ============================================================

@router.get(
    "/stations",
    response_model=WeatherStationListResponse,
)
async def list_weather_stations(
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
) -> WeatherStationListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(WeatherStation)

    count_query = (
        select(func.count())
        .select_from(WeatherStation)
    )

    if district_id is not None:
        query = query.where(
            WeatherStation.district_id == district_id
        )

        count_query = count_query.where(
            WeatherStation.district_id == district_id
        )

    total_result = await db.execute(
        count_query
    )

    result = await db.execute(
        query.order_by(WeatherStation.id)
        .offset(offset)
        .limit(limit)
    )

    return WeatherStationListResponse(
        items=[
            station_response(item)
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
    response_model=RainfallStatisticsResponse,
)
async def rainfall_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> RainfallStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    station_query = select(
        func.count(WeatherStation.id)
    )

    reading_query = (
        select(func.count(RainfallReading.id))
        .join(
            WeatherStation,
            RainfallReading.station_id
            == WeatherStation.id,
        )
    )

    if district_id is not None:
        station_query = station_query.where(
            WeatherStation.district_id == district_id
        )

        reading_query = reading_query.where(
            WeatherStation.district_id == district_id
        )

    stations_result = await db.execute(
        station_query
    )

    readings_result = await db.execute(
        reading_query
    )

    return RainfallStatisticsResponse(
        total_stations=int(
            stations_result.scalar_one()
        ),
        total_readings=int(
            readings_result.scalar_one()
        ),
        generated_at=utc_now(),
    )


# ============================================================
# DISTRICT RAINFALL SUMMARY
# ============================================================

@router.get(
    "/districts/{district_id}/summary",
    response_model=DistrictRainfallSummary,
)
async def district_rainfall_summary(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
    ),
) -> DistrictRainfallSummary:
    require_role(current_user, READ_ROLES)

    require_district_access(
        current_user,
        district_id,
    )

    period_end = utc_now()
    period_start = period_end - timedelta(
        hours=hours
    )

    station_result = await db.execute(
        select(func.count(WeatherStation.id))
        .where(
            WeatherStation.district_id == district_id
        )
    )

    reading_result = await db.execute(
        select(
            func.count(RainfallReading.id),
            func.coalesce(
                func.sum(RainfallReading.rainfall_mm),
                0,
            ),
        )
        .join(
            WeatherStation,
            RainfallReading.station_id
            == WeatherStation.id,
        )
        .where(
            WeatherStation.district_id == district_id,
            RainfallReading.observed_at >= period_start,
            RainfallReading.observed_at <= period_end,
        )
    )

    row = reading_result.one()

    return DistrictRainfallSummary(
        district_id=district_id,
        station_count=int(
            station_result.scalar_one()
        ),
        reading_count=int(row[0]),
        total_recorded_mm=float(row[1]),
        period_start=period_start,
        period_end=period_end,
        generated_at=utc_now(),
    )


# ============================================================
# STATION DETAILS
# ============================================================

@router.get(
    "/stations/{station_id}",
    response_model=WeatherStationResponse,
)
async def get_weather_station(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> WeatherStationResponse:
    require_role(current_user, READ_ROLES)

    station = await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    return station_response(station)


# ============================================================
# RAINFALL READINGS
# ============================================================

@router.get(
    "/stations/{station_id}/readings",
    response_model=list[RainfallReadingResponse],
)
async def list_rainfall_readings(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
) -> list[RainfallReadingResponse]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(RainfallReading)
        .where(
            RainfallReading.station_id
            == validate_uuid(station_id)
        )
        .order_by(
            RainfallReading.observed_at.desc()
        )
        .limit(limit)
    )

    return [
        RainfallReadingResponse(
            id=str(item.id),
            station_id=str(item.station_id),
            rainfall_mm=getattr(
                item,
                "rainfall_mm",
                None,
            ),
            observed_at=getattr(
                item,
                "observed_at",
                None,
            ),
            quality=enum_value(
                getattr(item, "quality", None)
            ) or None,
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/stations/{station_id}/readings",
    status_code=201,
)
async def create_rainfall_reading(
    station_id: str,
    payload: RainfallReadingCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    # Requires:
    # - Trusted station credentials
    # - Source verification
    # - Duplicate observation detection
    # - Timestamp and measurement validation
    # - Sensor anomaly detection
    # - Audit logging
    raise HTTPException(
        status_code=501,
        detail="Verified rainfall ingestion service is not configured",
    )


# ============================================================
# WEATHER OBSERVATIONS
# ============================================================

@router.get(
    "/stations/{station_id}/observations",
)
async def weather_observations(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
) -> list[WeatherObservationResponse]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(WeatherObservation)
        .where(
            WeatherObservation.station_id
            == validate_uuid(station_id)
        )
        .order_by(
            WeatherObservation.observed_at.desc()
        )
        .limit(limit)
    )

    return [
        WeatherObservationResponse(
            id=str(item.id),
            station_id=str(item.station_id),
            observed_at=getattr(
                item,
                "observed_at",
                None,
            ),
        )
        for item in result.scalars().all()
    ]


# ============================================================
# SENSOR HEALTH
# ============================================================

@router.get(
    "/stations/{station_id}/health",
)
async def station_health(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> WeatherHealthResponse | None:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(WeatherStationHealth)
        .where(
            WeatherStationHealth.station_id
            == validate_uuid(station_id)
        )
        .limit(1)
    )

    item = result.scalar_one_or_none()

    if item is None:
        return None

    return WeatherHealthResponse(
        id=str(item.id),
        station_id=str(item.station_id),
        status=enum_value(
            getattr(item, "status", None)
        ) or None,
    )


# ============================================================
# STATION ALERTS
# ============================================================

@router.get(
    "/stations/{station_id}/alerts",
)
async def station_alerts(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[WeatherAlertResponse]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(WeatherStationAlert)
        .where(
            WeatherStationAlert.station_id
            == validate_uuid(station_id)
        )
        .limit(limit)
    )

    return [
        WeatherAlertResponse(
            id=str(item.id),
            station_id=str(item.station_id),
            status=enum_value(
                getattr(item, "status", None)
            ) or None,
        )
        for item in result.scalars().all()
    ]


# ============================================================
# THRESHOLDS
# ============================================================

@router.get(
    "/stations/{station_id}/thresholds",
)
async def station_thresholds(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[WeatherThresholdResponse]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(WeatherThreshold)
        .where(
            WeatherThreshold.station_id
            == validate_uuid(station_id)
        )
    )

    return [
        WeatherThresholdResponse(
            id=str(item.id),
            station_id=str(item.station_id),
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/stations/{station_id}/thresholds",
    status_code=201,
)
async def create_station_threshold(
    station_id: str,
    payload: WeatherThresholdCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    # Thresholds must be calibrated and approved.
    raise HTTPException(
        status_code=501,
        detail="Rainfall threshold management service is not configured",
    )


# ============================================================
# MAINTENANCE
# ============================================================

@router.get(
    "/stations/{station_id}/maintenance",
)
async def station_maintenance(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[WeatherMaintenanceResponse]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(WeatherStationMaintenance)
        .where(
            WeatherStationMaintenance.station_id
            == validate_uuid(station_id)
        )
    )

    return [
        WeatherMaintenanceResponse(
            id=str(item.id),
            station_id=str(item.station_id),
        )
        for item in result.scalars().all()
    ]


# ============================================================
# STATION EVENTS
# ============================================================

@router.get(
    "/stations/{station_id}/events",
)
async def station_events(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(WeatherStationEvent)
        .where(
            WeatherStationEvent.station_id
            == validate_uuid(station_id)
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "station_id": str(item.station_id),
        }
        for item in result.scalars().all()
    ]


# ============================================================
# RAINFALL FORECAST INTEGRATION
# ============================================================

@router.post(
    "/forecast",
    response_model=RainfallForecastResponse,
)
async def rainfall_forecast(
    payload: RainfallForecastRequest,
    current_user: CurrentUser,
    service: ForecastServiceDependency,
) -> RainfallForecastResponse:
    require_role(current_user, READ_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    result = await service.forecast_rainfall(
        district_id=payload.district_id,
        horizon_hours=payload.horizon_hours,
        include_uncertainty=payload.include_uncertainty,
        actor_id=actor_id(current_user),
    )

    if result.district_id != payload.district_id:
        raise HTTPException(
            status_code=502,
            detail="Forecast district mismatch",
        )

    return result


__all__ = [
    "router",
    "RainfallForecastService",
    "RainfallReadingCreate",
    "WeatherThresholdCreate",
    "RainfallForecastRequest",
    "RainfallForecastResponse",
    "WeatherStationResponse",
    "WeatherStationListResponse",
    "RainfallReadingResponse",
    "RainfallStatisticsResponse",
    "DistrictRainfallSummary",
]
