from __future__ import annotations
"""
backend/api/rivers.py

Kerala Flood Relief Coordination Agent
River Monitoring & Water-Level Intelligence API

Python 3.11+ | FastAPI | Pydantic v2 | SQLAlchemy 2.x

Endpoints:
    GET  /rivers/health
    GET  /rivers/stations
    GET  /rivers/statistics
    GET  /rivers/districts/{district_id}/summary
    GET  /rivers/stations/{station_id}
    GET  /rivers/stations/{station_id}/levels
    GET  /rivers/stations/{station_id}/latest
    POST /rivers/stations/{station_id}/levels
    GET  /rivers/stations/{station_id}/forecasts
    POST /rivers/stations/{station_id}/forecast
    GET  /rivers/stations/{station_id}/sensors
    GET  /rivers/stations/{station_id}/advisories
    POST /rivers/stations/{station_id}/advisories
    GET  /rivers/stations/{station_id}/maintenance
    GET  /rivers/stations/{station_id}/events

Safety:
    - Never fabricate river readings.
    - Forecasts are advisory, not official warnings.
    - Verified observations must come from trusted sources.
    - Datum and measurement units must be validated.
    - Public flood alerts require human approval.
    - No physical dam or river infrastructure controls.
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

from sqlalchemy import (
    func,
    select,
)

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.river_station import (
    RiverStation,
    RiverWaterLevelReading,
    RiverStationSensor,
    RiverLevelForecast,
    RiverStationAdvisory,
    RiverStationMaintenance,
    RiverStationEvent,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/rivers",
    tags=["River Monitoring"],
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
# ROLE CONFIGURATION
# ============================================================

ADMIN_ROLES = {
    "admin",
    "super_admin",
}

HYDROLOGY_ROLES = {
    "hydrologist",
    "river_monitoring_officer",
    "water_resources_officer",
    "sensor_operator",
}

COMMAND_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "incident_commander",
    "risk_analyst",
}

READ_ROLES = (
    ADMIN_ROLES
    | HYDROLOGY_ROLES
    | COMMAND_ROLES
    | {"auditor"}
)

INGEST_ROLES = (
    ADMIN_ROLES
    | {
        "hydrologist",
        "river_monitoring_officer",
        "sensor_operator",
    }
)

ADVISORY_ROLES = (
    ADMIN_ROLES
    | {
        "hydrologist",
        "water_resources_officer",
        "district_officer",
    }
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class WaterLevelUnit(str, Enum):
    METERS = "m"
    CENTIMETERS = "cm"


class ReadingQuality(str, Enum):
    UNVERIFIED = "unverified"
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"


class ReadingSource(str, Enum):
    AUTOMATIC_SENSOR = "automatic_sensor"
    MANUAL_GAUGE = "manual_gauge"
    VERIFIED_INTEGRATION = "verified_integration"


class RiverTrend(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    UNKNOWN = "unknown"


class RiverRiskLevel(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    DANGER = "danger"
    EXTREME = "extreme"


class SensorStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class AdvisoryStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    CANCELLED = "cancelled"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class RiverLevelReadingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime

    water_level: float = Field(
        ge=-500,
        le=10000,
    )

    unit: WaterLevelUnit = WaterLevelUnit.METERS

    datum_reference: str = Field(
        min_length=3,
        max_length=200,
    )

    source: ReadingSource

    source_reference: str = Field(
        min_length=3,
        max_length=500,
    )

    quality: ReadingQuality = (
        ReadingQuality.UNVERIFIED
    )

    sensor_id: str | None = None

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
                "Observation timestamp cannot be in the future"
            )

        return value

    @field_validator("sensor_id")
    @classmethod
    def validate_sensor_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return str(uuid.UUID(value))


class RiverForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=72,
    )

    include_uncertainty: bool = True

    model_reference: str | None = Field(
        default=None,
        max_length=150,
    )


class RiverAdvisoryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=5,
        max_length=255,
    )

    description: str = Field(
        min_length=10,
        max_length=5000,
    )

    risk_level: RiverRiskLevel

    evidence_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    effective_from: datetime

    effective_until: datetime | None = None

    @model_validator(mode="after")
    def validate_period(
        self,
    ) -> "RiverAdvisoryCreate":
        if (
            self.effective_until is not None
            and self.effective_until <= self.effective_from
        ):
            raise ValueError(
                "Advisory end must be after start"
            )

        return self


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class RiverStationResponse(BaseModel):
    id: str
    district_id: int
    name: str | None = None
    river_name: str | None = None
    status: str | None = None


class RiverStationListResponse(BaseModel):
    items: list[RiverStationResponse]
    total: int
    limit: int
    offset: int


class RiverLevelResponse(BaseModel):
    id: str
    station_id: str
    observed_at: datetime | None = None
    water_level: float | None = None
    unit: str | None = None
    quality: str | None = None


class RiverSensorResponse(BaseModel):
    id: str
    station_id: str
    status: str | None = None


class RiverForecastResponse(BaseModel):
    station_id: str
    horizon_hours: int
    predicted_level_m: float
    model_reference: str
    generated_at: datetime
    data_cutoff_at: datetime
    advisory_only: bool = True


class RiverAdvisoryResponse(BaseModel):
    id: str
    station_id: str
    status: str | None = None
    risk_level: str | None = None


class RiverStatisticsResponse(BaseModel):
    total_stations: int
    total_readings: int
    generated_at: datetime


class DistrictRiverSummary(BaseModel):
    district_id: int
    station_count: int
    reading_count: int
    generated_at: datetime


class RiverHealthResponse(BaseModel):
    service: str
    status: str
    timestamp: datetime


# ============================================================
# SERVICE CONTRACT
# ============================================================

class RiverMonitoringService(Protocol):
    """
    Domain service responsibilities:

    - Authenticate station ingestion sources
    - Validate measurement datum and units
    - Detect duplicate readings
    - Detect stale and anomalous sensor data
    - Produce verified forecasts
    - Store model and observation provenance
    - Enforce human review of advisories
    - Audit operational changes
    """

    async def health(
        self,
    ) -> RiverHealthResponse:
        ...

    async def ingest_level(
        self,
        station_id: str,
        payload: RiverLevelReadingCreate,
        actor_id: str,
    ) -> RiverLevelResponse:
        ...

    async def forecast(
        self,
        station_id: str,
        payload: RiverForecastRequest,
        actor_id: str,
    ) -> RiverForecastResponse:
        ...

    async def create_advisory(
        self,
        station_id: str,
        payload: RiverAdvisoryCreate,
        actor_id: str,
    ) -> RiverAdvisoryResponse:
        ...


def get_river_service(
    request: Request,
) -> RiverMonitoringService:
    service = getattr(
        request.app.state,
        "river_monitoring_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="River monitoring service is not configured",
        )

    return service


RiverServiceDependency = Annotated[
    RiverMonitoringService,
    Depends(get_river_service),
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

    assigned = getattr(
        user,
        "district_id",
        None,
    )

    if (
        assigned is None
        or int(assigned) != district_id
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

    if (
        district_id is not None
        and district_id != int(assigned)
    ):
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )

    return int(assigned)


async def get_station_or_404(
    db: AsyncSession,
    station_id: str,
    user: Any,
) -> RiverStation:
    identifier = validate_uuid(station_id)

    result = await db.execute(
        select(RiverStation).where(
            RiverStation.id == identifier
        )
    )

    station = result.scalar_one_or_none()

    if station is None:
        raise HTTPException(
            status_code=404,
            detail="River station not found",
        )

    require_district_access(
        user,
        station.district_id,
    )

    return station


def station_response(
    station: RiverStation,
) -> RiverStationResponse:
    return RiverStationResponse(
        id=str(station.id),
        district_id=station.district_id,
        name=getattr(
            station,
            "name",
            None,
        ),
        river_name=getattr(
            station,
            "river_name",
            None,
        ),
        status=enum_value(
            getattr(station, "status", None)
        ) or None,
    )


def level_response(
    reading: RiverWaterLevelReading,
) -> RiverLevelResponse:
    return RiverLevelResponse(
        id=str(reading.id),
        station_id=str(reading.station_id),
        observed_at=getattr(
            reading,
            "observed_at",
            None,
        ),
        water_level=getattr(
            reading,
            "water_level",
            None,
        ),
        unit=enum_value(
            getattr(reading, "unit", None)
        ) or None,
        quality=enum_value(
            getattr(reading, "quality", None)
        ) or None,
    )


async def related_records(
    db: AsyncSession,
    model: Any,
    station_id: str,
    limit: int = 100,
) -> list[Any]:
    result = await db.execute(
        select(model)
        .where(
            model.station_id == station_id
        )
        .limit(limit)
    )

    return list(result.scalars().all())


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=RiverHealthResponse,
)
async def river_health(
    current_user: CurrentUser,
    service: RiverServiceDependency,
) -> RiverHealthResponse:
    require_role(current_user, READ_ROLES)

    return await service.health()


# ============================================================
# STATION LIST
# ============================================================

@router.get(
    "/stations",
    response_model=RiverStationListResponse,
)
async def list_river_stations(
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
) -> RiverStationListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(RiverStation)

    count_query = select(
        func.count(RiverStation.id)
    )

    if district_id is not None:
        query = query.where(
            RiverStation.district_id == district_id
        )

        count_query = count_query.where(
            RiverStation.district_id == district_id
        )

    total = (
        await db.execute(count_query)
    ).scalar_one()

    result = await db.execute(
        query.order_by(RiverStation.id)
        .offset(offset)
        .limit(limit)
    )

    return RiverStationListResponse(
        items=[
            station_response(item)
            for item in result.scalars().all()
        ],
        total=int(total),
        limit=limit,
        offset=offset,
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=RiverStatisticsResponse,
)
async def river_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> RiverStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    station_query = select(
        func.count(RiverStation.id)
    )

    reading_query = (
        select(
            func.count(RiverWaterLevelReading.id)
        )
        .join(
            RiverStation,
            RiverWaterLevelReading.station_id
            == RiverStation.id,
        )
    )

    if district_id is not None:
        station_query = station_query.where(
            RiverStation.district_id == district_id
        )

        reading_query = reading_query.where(
            RiverStation.district_id == district_id
        )

    station_count = (
        await db.execute(station_query)
    ).scalar_one()

    reading_count = (
        await db.execute(reading_query)
    ).scalar_one()

    return RiverStatisticsResponse(
        total_stations=int(station_count),
        total_readings=int(reading_count),
        generated_at=datetime.now(timezone.utc),
    )


# ============================================================
# DISTRICT SUMMARY
# ============================================================

@router.get(
    "/districts/{district_id}/summary",
    response_model=DistrictRiverSummary,
)
async def district_river_summary(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
) -> DistrictRiverSummary:
    require_role(current_user, READ_ROLES)

    require_district_access(
        current_user,
        district_id,
    )

    stations = (
        await db.execute(
            select(func.count(RiverStation.id))
            .where(
                RiverStation.district_id == district_id
            )
        )
    ).scalar_one()

    readings = (
        await db.execute(
            select(
                func.count(RiverWaterLevelReading.id)
            )
            .join(
                RiverStation,
                RiverWaterLevelReading.station_id
                == RiverStation.id,
            )
            .where(
                RiverStation.district_id == district_id
            )
        )
    ).scalar_one()

    return DistrictRiverSummary(
        district_id=district_id,
        station_count=int(stations),
        reading_count=int(readings),
        generated_at=datetime.now(timezone.utc),
    )


# ============================================================
# STATION DETAILS
# ============================================================

@router.get(
    "/stations/{station_id}",
    response_model=RiverStationResponse,
)
async def get_river_station(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> RiverStationResponse:
    require_role(current_user, READ_ROLES)

    station = await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    return station_response(station)


# ============================================================
# WATER LEVEL HISTORY
# ============================================================

@router.get(
    "/stations/{station_id}/levels",
    response_model=list[RiverLevelResponse],
)
async def list_river_levels(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
) -> list[RiverLevelResponse]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(RiverWaterLevelReading)
        .where(
            RiverWaterLevelReading.station_id
            == validate_uuid(station_id)
        )
        .order_by(
            RiverWaterLevelReading.observed_at.desc()
        )
        .limit(limit)
    )

    return [
        level_response(item)
        for item in result.scalars().all()
    ]


# ============================================================
# LATEST WATER LEVEL
# ============================================================

@router.get(
    "/stations/{station_id}/latest",
    response_model=RiverLevelResponse,
)
async def latest_river_level(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> RiverLevelResponse:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    result = await db.execute(
        select(RiverWaterLevelReading)
        .where(
            RiverWaterLevelReading.station_id
            == validate_uuid(station_id)
        )
        .order_by(
            RiverWaterLevelReading.observed_at.desc()
        )
        .limit(1)
    )

    reading = result.scalar_one_or_none()

    if reading is None:
        raise HTTPException(
            status_code=404,
            detail="No water-level readings available",
        )

    return level_response(reading)


# ============================================================
# INGEST WATER LEVEL
# ============================================================

@router.post(
    "/stations/{station_id}/levels",
    response_model=RiverLevelResponse,
    status_code=201,
)
async def ingest_river_level(
    station_id: str,
    payload: RiverLevelReadingCreate,
    db: DbSession,
    current_user: CurrentUser,
    service: RiverServiceDependency,
) -> RiverLevelResponse:
    require_role(current_user, INGEST_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    return await service.ingest_level(
        station_id=validate_uuid(station_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# FORECAST HISTORY
# ============================================================

@router.get(
    "/stations/{station_id}/forecasts",
)
async def river_forecast_history(
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

    records = await related_records(
        db,
        RiverLevelForecast,
        validate_uuid(station_id),
        limit,
    )

    return [
        {
            "id": str(item.id),
            "station_id": str(item.station_id),
        }
        for item in records
    ]


# ============================================================
# GENERATE FORECAST
# ============================================================

@router.post(
    "/stations/{station_id}/forecast",
    response_model=RiverForecastResponse,
)
async def forecast_river_level(
    station_id: str,
    payload: RiverForecastRequest,
    db: DbSession,
    current_user: CurrentUser,
    service: RiverServiceDependency,
) -> RiverForecastResponse:
    require_role(
        current_user,
        HYDROLOGY_ROLES | ADMIN_ROLES,
    )

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    return await service.forecast(
        station_id=validate_uuid(station_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# SENSORS
# ============================================================

@router.get(
    "/stations/{station_id}/sensors",
    response_model=list[RiverSensorResponse],
)
async def river_station_sensors(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[RiverSensorResponse]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    records = await related_records(
        db,
        RiverStationSensor,
        validate_uuid(station_id),
    )

    return [
        RiverSensorResponse(
            id=str(item.id),
            station_id=str(item.station_id),
            status=enum_value(
                getattr(item, "status", None)
            ) or None,
        )
        for item in records
    ]


# ============================================================
# ADVISORIES
# ============================================================

@router.get(
    "/stations/{station_id}/advisories",
    response_model=list[RiverAdvisoryResponse],
)
async def river_advisories(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[RiverAdvisoryResponse]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    records = await related_records(
        db,
        RiverStationAdvisory,
        validate_uuid(station_id),
    )

    return [
        RiverAdvisoryResponse(
            id=str(item.id),
            station_id=str(item.station_id),
            status=enum_value(
                getattr(item, "status", None)
            ) or None,
            risk_level=enum_value(
                getattr(item, "risk_level", None)
            ) or None,
        )
        for item in records
    ]


@router.post(
    "/stations/{station_id}/advisories",
    response_model=RiverAdvisoryResponse,
    status_code=201,
)
async def create_river_advisory(
    station_id: str,
    payload: RiverAdvisoryCreate,
    db: DbSession,
    current_user: CurrentUser,
    service: RiverServiceDependency,
) -> RiverAdvisoryResponse:
    require_role(current_user, ADVISORY_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    # Creation must produce a draft, not an
    # automatically published official warning.
    return await service.create_advisory(
        station_id=validate_uuid(station_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# MAINTENANCE
# ============================================================

@router.get(
    "/stations/{station_id}/maintenance",
)
async def river_station_maintenance(
    station_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_station_or_404(
        db,
        station_id,
        current_user,
    )

    records = await related_records(
        db,
        RiverStationMaintenance,
        validate_uuid(station_id),
    )

    return [
        {
            "id": str(item.id),
            "station_id": str(item.station_id),
        }
        for item in records
    ]


# ============================================================
# EVENTS
# ============================================================

@router.get(
    "/stations/{station_id}/events",
)
async def river_station_events(
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

    records = await related_records(
        db,
        RiverStationEvent,
        validate_uuid(station_id),
        limit,
    )

    return [
        {
            "id": str(item.id),
            "station_id": str(item.station_id),
        }
        for item in records
    ]


__all__ = [
    "router",
    "RiverMonitoringService",
    "RiverLevelReadingCreate",
    "RiverForecastRequest",
    "RiverAdvisoryCreate",
    "RiverStationResponse",
    "RiverLevelResponse",
    "RiverForecastResponse",
    "RiverStatisticsResponse",
    "DistrictRiverSummary",
]