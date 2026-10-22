from __future__ import annotations

"""
backend/api/flood_forecast.py

Kerala Flood Relief Coordination Agent
AI-Powered Flood Forecasting API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Features:
    - District-level flood forecasting
    - Rainfall prediction
    - River water-level prediction
    - Flood probability and severity estimation
    - ML/DL model selection
    - Forecast uncertainty
    - Multi-horizon predictions
    - Historical forecast retrieval
    - Model evaluation
    - Forecast comparison
    - Human-reviewed advisories
    - District-based authorization
    - Audit-ready prediction requests

Endpoints:
    GET  /flood-forecast/models
    GET  /flood-forecast/health
    POST /flood-forecast/predict
    POST /flood-forecast/rainfall
    POST /flood-forecast/river-level
    POST /flood-forecast/risk-assessment
    POST /flood-forecast/compare
    GET  /flood-forecast/history
    GET  /flood-forecast/districts/{district_id}
    GET  /flood-forecast/evaluations
    POST /flood-forecast/evaluations
    POST /flood-forecast/advisories

Important:
    - Forecasts are advisory, not official warnings.
    - No fabricated model predictions.
    - Missing ML services return 503.
    - Official alerts require human authorization.
    - Predictions must include provenance and timestamps.
"""


import logging
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
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user
from backend.models.district import District


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/flood-forecast",
    tags=["Flood Forecasting"],
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

FORECAST_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "hydrologist",
    "meteorologist",
    "risk_analyst",
    "analyst",
}

READ_ROLES = (
    ADMIN_ROLES
    | FORECAST_ROLES
    | {"operator", "reviewer"}
)

EVALUATION_ROLES = (
    ADMIN_ROLES
    | {
        "hydrologist",
        "meteorologist",
        "risk_analyst",
    }
)

ADVISORY_ROLES = (
    ADMIN_ROLES
    | {
        "district_officer",
        "emergency_coordinator",
    }
)

SUPPORTED_HORIZONS = {
    1, 3, 6, 12, 24, 48, 72,
}

MAX_BATCH_DISTRICTS = 14


# ============================================================
# ENUMERATIONS
# ============================================================

class ForecastModelType(str, Enum):
    AUTO = "auto"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    TEMPORAL_FUSION_TRANSFORMER = (
        "temporal_fusion_transformer"
    )
    ENSEMBLE = "ensemble"
    HYDROLOGICAL = "hydrological"


class FloodRiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class ForecastSource(str, Enum):
    WEATHER_STATION = "weather_station"
    RIVER_STATION = "river_station"
    SATELLITE = "satellite"
    RADAR = "radar"
    HISTORICAL = "historical"
    MULTI_SOURCE = "multi_source"


class ForecastStatus(str, Enum):
    GENERATED = "generated"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    INVALIDATED = "invalidated"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class FloodForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    district_id: int = Field(ge=1)

    horizon_hours: int = Field(
        default=24,
        description="Prediction horizon in hours",
    )

    model_type: ForecastModelType = (
        ForecastModelType.AUTO
    )

    include_uncertainty: bool = True

    include_explanations: bool = True

    use_latest_sensor_data: bool = True

    rainfall_override_mm: float | None = Field(
        default=None,
        ge=0,
        le=5000,
    )

    river_level_override_m: float | None = Field(
        default=None,
        ge=0,
        le=1000,
    )

    @model_validator(mode="after")
    def validate_horizon(
        self,
    ) -> "FloodForecastRequest":
        if self.horizon_hours not in SUPPORTED_HORIZONS:
            raise ValueError(
                "Supported horizons: "
                "1, 3, 6, 12, 24, 48, 72 hours"
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

    model_type: ForecastModelType = (
        ForecastModelType.AUTO
    )

    include_uncertainty: bool = True


class RiverLevelForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    river_station_id: str

    horizon_hours: int = Field(
        default=12,
        ge=1,
        le=72,
    )

    model_type: ForecastModelType = (
        ForecastModelType.AUTO
    )

    include_uncertainty: bool = True

    @model_validator(mode="after")
    def validate_station(
        self,
    ) -> "RiverLevelForecastRequest":
        self.river_station_id = str(
            uuid.UUID(self.river_station_id)
        )

        return self


class RiskAssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    district_id: int = Field(ge=1)

    rainfall_mm: float = Field(
        ge=0,
        le=5000,
    )

    river_level_m: float | None = Field(
        default=None,
        ge=0,
        le=1000,
    )

    soil_saturation_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    reservoir_storage_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=72,
    )


class ForecastCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    district_id: int = Field(ge=1)

    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=72,
    )

    model_types: list[ForecastModelType] = Field(
        min_length=2,
        max_length=5,
    )


class ForecastEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(
        min_length=2,
        max_length=150,
    )

    dataset_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    evaluation_window_start: datetime

    evaluation_window_end: datetime

    metrics: list[str] = Field(
        default_factory=lambda: [
            "mae",
            "rmse",
            "precision",
            "recall",
            "f1",
            "brier_score",
        ],
    )

    @model_validator(mode="after")
    def validate_window(
        self,
    ) -> "ForecastEvaluationRequest":
        if (
            self.evaluation_window_end
            <= self.evaluation_window_start
        ):
            raise ValueError(
                "Evaluation end must be after start"
            )

        return self


class ForecastAdvisoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_id: str

    district_id: int = Field(ge=1)

    title: str = Field(
        min_length=5,
        max_length=255,
    )

    recommendations: list[str] = Field(
        min_length=1,
        max_length=20,
    )

    review_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    @model_validator(mode="after")
    def validate_forecast_id(
        self,
    ) -> "ForecastAdvisoryRequest":
        self.forecast_id = str(
            uuid.UUID(self.forecast_id)
        )

        return self


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ForecastPoint(BaseModel):
    timestamp: datetime

    predicted_value: float

    lower_bound: float | None = None

    upper_bound: float | None = None


class ForecastProvenance(BaseModel):
    model_id: str
    model_version: str
    generated_at: datetime
    data_cutoff_at: datetime
    sources: list[str]
    verified: bool = False


class FloodForecastResponse(BaseModel):
    forecast_id: str

    district_id: int

    horizon_hours: int

    flood_probability: float = Field(
        ge=0,
        le=1,
    )

    risk_level: FloodRiskLevel

    predicted_rainfall_mm: float | None = None

    predicted_river_level_m: float | None = None

    confidence_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    predictions: list[ForecastPoint] = Field(
        default_factory=list,
    )

    explanations: list[str] = Field(
        default_factory=list,
    )

    provenance: ForecastProvenance

    advisory_only: bool = True


class RainfallForecastResponse(BaseModel):
    district_id: int
    horizon_hours: int
    predicted_total_mm: float = Field(ge=0)
    predictions: list[ForecastPoint]
    provenance: ForecastProvenance


class RiverLevelForecastResponse(BaseModel):
    river_station_id: str
    horizon_hours: int
    predicted_peak_level_m: float
    predictions: list[ForecastPoint]
    provenance: ForecastProvenance


class RiskAssessmentResponse(BaseModel):
    district_id: int
    risk_score: float = Field(ge=0, le=100)
    risk_level: FloodRiskLevel
    contributing_factors: dict[str, float]
    provenance: ForecastProvenance
    advisory_only: bool = True


class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    version: str
    is_available: bool
    supported_horizons: list[int]


class ForecastHealthResponse(BaseModel):
    service: str
    status: str
    timestamp: datetime
    model_service_available: bool


# ============================================================
# FORECAST ENGINE CONTRACT
# ============================================================

class ForecastEngine(Protocol):
    """
    Contract for a separately implemented forecasting
    service.

    The engine must use actual validated models and data.
    It must not invent predictions when inputs are missing.
    """

    async def health(self) -> bool:
        ...

    async def list_models(self) -> list[ModelInfo]:
        ...

    async def predict_flood(
        self,
        payload: FloodForecastRequest,
        actor_id: str,
    ) -> FloodForecastResponse:
        ...

    async def predict_rainfall(
        self,
        payload: RainfallForecastRequest,
        actor_id: str,
    ) -> RainfallForecastResponse:
        ...

    async def predict_river_level(
        self,
        payload: RiverLevelForecastRequest,
        actor_id: str,
    ) -> RiverLevelForecastResponse:
        ...

    async def assess_risk(
        self,
        payload: RiskAssessmentRequest,
        actor_id: str,
    ) -> RiskAssessmentResponse:
        ...

    async def compare_models(
        self,
        payload: ForecastCompareRequest,
        actor_id: str,
    ) -> dict[str, Any]:
        ...

    async def list_history(
        self,
        district_id: int | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        ...

    async def list_evaluations(
        self,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        ...

    async def evaluate(
        self,
        payload: ForecastEvaluationRequest,
        actor_id: str,
    ) -> dict[str, Any]:
        ...

    async def create_advisory(
        self,
        payload: ForecastAdvisoryRequest,
        actor_id: str,
    ) -> dict[str, Any]:
        ...


# ============================================================
# DEPENDENCIES
# ============================================================

def get_forecast_engine(
    request: Request,
) -> ForecastEngine:
    """
    Register the forecasting engine on app.state:

        app.state.forecast_engine = engine

    Missing services fail explicitly instead of returning
    fabricated model outputs.
    """

    engine = getattr(
        request.app.state,
        "forecast_engine",
        None,
    )

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting engine is not configured",
        )

    return engine


ForecastEngineDependency = Annotated[
    ForecastEngine,
    Depends(get_forecast_engine),
]


# ============================================================
# SECURITY HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> str:
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


async def validate_district(
    db: AsyncSession,
    district_id: int,
) -> None:
    result = await db.execute(
        select(District.id).where(
            District.id == district_id
        )
    )

    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404,
            detail="District not found",
        )


async def validate_forecast_district(
    db: AsyncSession,
    user: Any,
    district_id: int,
) -> None:
    require_district_access(user, district_id)
    await validate_district(db, district_id)


# ============================================================
# MODEL DISCOVERY
# ============================================================

@router.get(
    "/models",
    response_model=list[ModelInfo],
)
async def list_forecast_models(
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
) -> list[ModelInfo]:
    require_role(current_user, READ_ROLES)

    return await engine.list_models()


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=ForecastHealthResponse,
)
async def forecast_health(
    current_user: CurrentUser,
    request: Request,
) -> ForecastHealthResponse:
    require_role(current_user, READ_ROLES)

    engine = getattr(
        request.app.state,
        "forecast_engine",
        None,
    )

    available = False

    if engine is not None:
        try:
            available = bool(
                await engine.health()
            )
        except Exception:
            logger.exception(
                "Forecast engine health check failed"
            )

    return ForecastHealthResponse(
        service="flood_forecasting",
        status=(
            "operational"
            if available
            else "unavailable"
        ),
        timestamp=utc_now(),
        model_service_available=available,
    )


# ============================================================
# FLOOD PREDICTION
# ============================================================

@router.post(
    "/predict",
    response_model=FloodForecastResponse,
)
async def predict_flood(
    payload: FloodForecastRequest,
    db: DbSession,
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
) -> FloodForecastResponse:
    require_role(current_user, READ_ROLES)

    await validate_forecast_district(
        db,
        current_user,
        payload.district_id,
    )

    prediction = await engine.predict_flood(
        payload=payload,
        actor_id=actor_id(current_user),
    )

    if prediction.district_id != payload.district_id:
        raise HTTPException(
            status_code=502,
            detail="Forecast district mismatch",
        )

    return prediction


# ============================================================
# RAINFALL FORECAST
# ============================================================

@router.post(
    "/rainfall",
    response_model=RainfallForecastResponse,
)
async def predict_rainfall(
    payload: RainfallForecastRequest,
    db: DbSession,
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
) -> RainfallForecastResponse:
    require_role(current_user, READ_ROLES)

    await validate_forecast_district(
        db,
        current_user,
        payload.district_id,
    )

    result = await engine.predict_rainfall(
        payload=payload,
        actor_id=actor_id(current_user),
    )

    if result.district_id != payload.district_id:
        raise HTTPException(
            status_code=502,
            detail="Rainfall forecast district mismatch",
        )

    return result


# ============================================================
# RIVER LEVEL FORECAST
# ============================================================

@router.post(
    "/river-level",
    response_model=RiverLevelForecastResponse,
)
async def predict_river_level(
    payload: RiverLevelForecastRequest,
    db: DbSession,
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
) -> RiverLevelForecastResponse:
    require_role(current_user, READ_ROLES)

    # The river station's district must be resolved and
    # checked before invoking the forecasting engine.
    # The RiverStation model mapping is required here.
    raise HTTPException(
        status_code=501,
        detail=(
            "River station district authorization "
            "is not configured"
        ),
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

@router.post(
    "/risk-assessment",
    response_model=RiskAssessmentResponse,
)
async def assess_flood_risk(
    payload: RiskAssessmentRequest,
    db: DbSession,
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
) -> RiskAssessmentResponse:
    require_role(current_user, READ_ROLES)

    await validate_forecast_district(
        db,
        current_user,
        payload.district_id,
    )

    result = await engine.assess_risk(
        payload=payload,
        actor_id=actor_id(current_user),
    )

    if result.district_id != payload.district_id:
        raise HTTPException(
            status_code=502,
            detail="Risk assessment district mismatch",
        )

    return result


# ============================================================
# MODEL COMPARISON
# ============================================================

@router.post("/compare")
async def compare_forecast_models(
    payload: ForecastCompareRequest,
    db: DbSession,
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
) -> dict[str, Any]:
    require_role(current_user, EVALUATION_ROLES)

    await validate_forecast_district(
        db,
        current_user,
        payload.district_id,
    )

    if len(set(payload.model_types)) != len(
        payload.model_types
    ):
        raise HTTPException(
            status_code=422,
            detail="Model types must be unique",
        )

    return await engine.compare_models(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# FORECAST HISTORY
# ============================================================

@router.get("/history")
async def forecast_history(
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> list[dict[str, Any]]:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await engine.list_history(
        district_id=district_id,
        limit=limit,
        offset=offset,
    )


# ============================================================
# DISTRICT FORECAST HISTORY
# ============================================================

@router.get(
    "/districts/{district_id}",
)
async def district_forecasts(
    district_id: int,
    db: DbSession,
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> list[dict[str, Any]]:
    await validate_forecast_district(
        db,
        current_user,
        district_id,
    )

    return await engine.list_history(
        district_id=district_id,
        limit=limit,
        offset=0,
    )


# ============================================================
# MODEL EVALUATIONS
# ============================================================

@router.get("/evaluations")
async def list_forecast_evaluations(
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, EVALUATION_ROLES)

    return await engine.list_evaluations(
        limit=limit,
        offset=offset,
    )


@router.post(
    "/evaluations",
    status_code=202,
)
async def evaluate_forecast_model(
    payload: ForecastEvaluationRequest,
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
) -> dict[str, Any]:
    require_role(current_user, EVALUATION_ROLES)

    return await engine.evaluate(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# HUMAN-REVIEWED ADVISORIES
# ============================================================

@router.post(
    "/advisories",
    status_code=202,
)
async def create_forecast_advisory(
    payload: ForecastAdvisoryRequest,
    db: DbSession,
    current_user: CurrentUser,
    engine: ForecastEngineDependency,
) -> dict[str, Any]:
    require_role(current_user, ADVISORY_ROLES)

    await validate_forecast_district(
        db,
        current_user,
        payload.district_id,
    )

    # The forecasting engine must verify:
    # - Forecast exists and belongs to district
    # - Model provenance is valid
    # - Reviewer has appropriate authority
    # - Recommendations are evidence-grounded
    # - Advisory is not published as an official alert
    # - An immutable audit event is recorded

    return await engine.create_advisory(
        payload=payload,
        actor_id=actor_id(current_user),
    )


__all__ = [
    "router",
    "ForecastEngine",
    "ForecastModelType",
    "FloodRiskLevel",
    "FloodForecastRequest",
    "FloodForecastResponse",
    "RainfallForecastRequest",
    "RainfallForecastResponse",
    "RiverLevelForecastRequest",
    "RiverLevelForecastResponse",
    "RiskAssessmentRequest",
    "RiskAssessmentResponse",
    "ForecastCompareRequest",
    "ForecastEvaluationRequest",
    "ForecastAdvisoryRequest",
    "ForecastProvenance",
    "ForecastPoint",
    "ModelInfo",
    "ForecastHealthResponse",
]
