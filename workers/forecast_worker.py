from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/workers/forecast_worker.py

AI Flood Forecasting Worker

Features:
- Rainfall forecasting
- River-level forecasting
- Flood-risk prediction
- District-level forecasting
- Configurable ML model inference
- Forecast uncertainty tracking
- Celery background execution
- Redis caching
- Forecast history
- Data validation
- Model version tracking
- Human-reviewed alert recommendations

Safety:
Predictions are advisory.
No official flood warnings are issued automatically.
"""


import hashlib
import json
import logging
import math
import os
import pickle
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from redis import Redis
from redis.exceptions import RedisError

from backend.core.config import get_settings


# ============================================================
# CONFIGURATION
# ============================================================

settings = get_settings()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models" / "flood_prediction"

FORECAST_CACHE_TTL = 60 * 15

FORECAST_HISTORY_TTL = 60 * 60 * 24 * 30

SUPPORTED_HORIZONS = {1, 3, 6, 12, 24, 48, 72}


# ============================================================
# CELERY
# ============================================================

celery_app = Celery(
    "kerala_flood_forecast_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    task_soft_time_limit=240,
    task_time_limit=300,
)


# ============================================================
# REDIS
# ============================================================

redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


# ============================================================
# ENUMS
# ============================================================

class ForecastType(str, Enum):
    RAINFALL = "rainfall"
    RIVER_LEVEL = "river_level"
    FLOOD_RISK = "flood_risk"
    DISTRICT = "district"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ForecastStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    INSUFFICIENT_DATA = "insufficient_data"


# ============================================================
# KERALA DISTRICTS
# ============================================================

KERALA_DISTRICTS = {
    "thiruvananthapuram",
    "kollam",
    "pathanamthitta",
    "alappuzha",
    "kottayam",
    "idukki",
    "ernakulam",
    "thrissur",
    "palakkad",
    "malappuram",
    "kozhikode",
    "wayanad",
    "kannur",
    "kasaragod",
}


def normalize_district(value: str) -> str:
    district = value.strip().lower()

    if district not in KERALA_DISTRICTS:
        raise ValueError(
            f"Unsupported Kerala district: {value}"
        )

    return district


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


# ============================================================
# INPUT SCHEMAS
# ============================================================

class WeatherObservation(BaseModel):
    timestamp: datetime

    rainfall_mm: float = Field(ge=0, le=2000)

    temperature_c: float | None = Field(
        default=None,
        ge=-30,
        le=65,
    )

    humidity_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    pressure_hpa: float | None = Field(
        default=None,
        ge=800,
        le=1100,
    )

    wind_speed_mps: float | None = Field(
        default=None,
        ge=0,
        le=150,
    )


class RiverObservation(BaseModel):
    timestamp: datetime

    station_id: str = Field(min_length=1)

    water_level_m: float

    discharge_m3s: float | None = Field(
        default=None,
        ge=0,
    )


class ForecastRequest(BaseModel):
    district: str

    horizon_hours: int = 24

    weather_history: list[WeatherObservation] = Field(
        default_factory=list
    )

    river_history: list[RiverObservation] = Field(
        default_factory=list
    )

    station_id: str | None = None

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @field_validator("horizon_hours")
    @classmethod
    def validate_horizon(cls, value: int) -> int:
        if value not in SUPPORTED_HORIZONS:
            raise ValueError(
                "Supported forecast horizons: "
                "1, 3, 6, 12, 24, 48, 72 hours"
            )
        return value

    @model_validator(mode="after")
    def validate_timestamps(self):
        for history in (
            self.weather_history,
            self.river_history,
        ):
            timestamps = [
                item.timestamp for item in history
            ]

            if any(
                timestamp.tzinfo is None
                for timestamp in timestamps
            ):
                raise ValueError(
                    "Observation timestamps must "
                    "include timezone information"
                )

            if timestamps != sorted(timestamps):
                raise ValueError(
                    "Observations must be ordered "
                    "from oldest to newest"
                )

        return self


class ForecastResult(BaseModel):
    forecast_id: str
    request_id: str
    district: str

    forecast_type: ForecastType
    horizon_hours: int

    status: ForecastStatus

    predicted_value: float | None = None
    unit: str | None = None

    risk_level: RiskLevel = RiskLevel.UNKNOWN

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    prediction_interval: list[float] | None = None

    model_name: str | None = None
    model_version: str | None = None

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    version: str
    feature_names: tuple[str, ...]
    output_unit: str


MODEL_REGISTRY = {
    ForecastType.RAINFALL: ModelSpec(
        name="rainfall_forecaster",
        filename="rainfall_model.joblib",
        version="1.0",
        feature_names=(
            "rainfall_latest",
            "rainfall_mean",
            "rainfall_max",
            "rainfall_sum",
            "rainfall_std",
            "horizon_hours",
        ),
        output_unit="mm",
    ),
    ForecastType.RIVER_LEVEL: ModelSpec(
        name="river_level_forecaster",
        filename="river_level_model.joblib",
        version="1.0",
        feature_names=(
            "river_level_latest",
            "river_level_mean",
            "river_level_max",
            "river_level_trend",
            "rainfall_sum",
            "horizon_hours",
        ),
        output_unit="m",
    ),
    ForecastType.FLOOD_RISK: ModelSpec(
        name="flood_risk_predictor",
        filename="flood_risk_model.joblib",
        version="1.0",
        feature_names=(
            "rainfall_sum",
            "rainfall_max",
            "river_level_latest",
            "river_level_trend",
            "horizon_hours",
        ),
        output_unit="probability",
    ),
}


# ============================================================
# SAFE MODEL LOADING
# ============================================================

class ModelUnavailableError(RuntimeError):
    pass


_model_cache: dict[str, Any] = {}


def load_model(
    forecast_type: ForecastType,
) -> tuple[Any, ModelSpec]:
    """
    Load only locally trusted, verified model artifacts.

    joblib uses pickle internally and is unsafe
    for untrusted files. Model provenance and
    checksums must be verified before deployment.
    """

    import joblib

    spec = MODEL_REGISTRY[forecast_type]

    model_path = MODEL_DIR / spec.filename

    if not model_path.is_file():
        raise ModelUnavailableError(
            f"Model not installed: {spec.name}"
        )

    cache_key = str(model_path)

    if cache_key not in _model_cache:
        logger.info(
            "Loading trusted model: %s",
            spec.name,
        )

        _model_cache[cache_key] = joblib.load(
            model_path
        )

    return _model_cache[cache_key], spec


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def rainfall_features(
    history: list[WeatherObservation],
    horizon_hours: int,
) -> dict[str, float]:

    if len(history) < 3:
        raise ValueError(
            "At least 3 weather observations required"
        )

    values = np.asarray(
        [item.rainfall_mm for item in history],
        dtype=np.float64,
    )

    return {
        "rainfall_latest": float(values[-1]),
        "rainfall_mean": float(np.mean(values)),
        "rainfall_max": float(np.max(values)),
        "rainfall_sum": float(np.sum(values)),
        "rainfall_std": float(np.std(values)),
        "horizon_hours": float(horizon_hours),
    }


def river_features(
    history: list[RiverObservation],
    weather_history: list[WeatherObservation],
    horizon_hours: int,
) -> dict[str, float]:

    if len(history) < 3:
        raise ValueError(
            "At least 3 river observations required"
        )

    levels = np.asarray(
        [item.water_level_m for item in history],
        dtype=np.float64,
    )

    rainfall_sum = sum(
        item.rainfall_mm
        for item in weather_history
    )

    return {
        "river_level_latest": float(levels[-1]),
        "river_level_mean": float(np.mean(levels)),
        "river_level_max": float(np.max(levels)),
        "river_level_trend": float(
            levels[-1] - levels[0]
        ),
        "rainfall_sum": float(rainfall_sum),
        "horizon_hours": float(horizon_hours),
    }


def flood_risk_features(
    request: ForecastRequest,
) -> dict[str, float]:

    rain = rainfall_features(
        request.weather_history,
        request.horizon_hours,
    )

    river = river_features(
        request.river_history,
        request.weather_history,
        request.horizon_hours,
    )

    return {
        "rainfall_sum": rain["rainfall_sum"],
        "rainfall_max": rain["rainfall_max"],
        "river_level_latest": river["river_level_latest"],
        "river_level_trend": river["river_level_trend"],
        "horizon_hours": float(
            request.horizon_hours
        ),
    }


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_inference(
    forecast_type: ForecastType,
    features: dict[str, float],
) -> tuple[float, ModelSpec]:

    model, spec = load_model(forecast_type)

    missing = set(spec.feature_names) - set(features)

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    vector = np.asarray(
        [
            features[name]
            for name in spec.feature_names
        ],
        dtype=np.float64,
    ).reshape(1, -1)

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "Model features must be finite"
        )

    if forecast_type == ForecastType.FLOOD_RISK:
        if not hasattr(model, "predict_proba"):
            raise ModelUnavailableError(
                "Flood-risk model must support "
                "predict_proba"
            )

        probabilities = model.predict_proba(vector)

        classes = list(model.classes_)

        if 1 not in classes:
            raise ModelUnavailableError(
                "Flood-risk positive class not found"
            )

        positive_index = classes.index(1)

        prediction = float(
            probabilities[0][positive_index]
        )
    else:
        prediction = float(
            model.predict(vector)[0]
        )

    if not math.isfinite(prediction):
        raise ValueError(
            "Model returned non-finite prediction"
        )

    if forecast_type == ForecastType.FLOOD_RISK:
        if not 0 <= prediction <= 1:
            raise ValueError(
                "Invalid flood-risk probability"
            )
    elif forecast_type == ForecastType.RAINFALL:
        if prediction < 0:
            raise ValueError(
                "Rainfall prediction cannot be negative"
            )

    return prediction, spec


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(
    probability: float,
) -> RiskLevel:

    if not 0 <= probability <= 1:
        return RiskLevel.UNKNOWN

    # Illustrative model-score categories only.
    # These are not official flood-warning thresholds.

    if probability >= 0.90:
        return RiskLevel.CRITICAL

    if probability >= 0.75:
        return RiskLevel.HIGH

    if probability >= 0.50:
        return RiskLevel.MODERATE

    return RiskLevel.LOW


# ============================================================
# REDIS STORAGE
# ============================================================

def forecast_cache_key(
    request: ForecastRequest,
    forecast_type: ForecastType,
) -> str:

    payload = request.model_dump(
        mode="json",
        exclude={"request_id"},
    )

    payload["forecast_type"] = forecast_type.value

    raw = json.dumps(
        payload,
        sort_keys=True,
    ).encode("utf-8")

    fingerprint = hashlib.sha256(raw).hexdigest()

    return f"forecast:cache:{fingerprint}"


def save_forecast(
    result: ForecastResult,
) -> None:

    payload = result.model_dump(mode="json")

    redis_client.set(
        f"forecast:result:{result.forecast_id}",
        json.dumps(payload),
        ex=FORECAST_HISTORY_TTL,
    )

    redis_client.lpush(
        f"forecast:history:{result.district}",
        result.forecast_id,
    )

    redis_client.ltrim(
        f"forecast:history:{result.district}",
        0,
        999,
    )


def get_cached_forecast(
    key: str,
) -> dict[str, Any] | None:

    raw = redis_client.get(key)

    if not raw:
        return None

    return json.loads(raw)


def cache_forecast(
    key: str,
    result: ForecastResult,
) -> None:

    redis_client.set(
        key,
        result.model_dump_json(),
        ex=FORECAST_CACHE_TTL,
    )


# ============================================================
# FORECAST ENGINE
# ============================================================

def generate_forecast(
    request: ForecastRequest,
    forecast_type: ForecastType,
) -> ForecastResult:

    forecast_id = str(uuid.uuid4())

    try:
        if forecast_type == ForecastType.RAINFALL:
            features = rainfall_features(
                request.weather_history,
                request.horizon_hours,
            )

        elif forecast_type == ForecastType.RIVER_LEVEL:
            features = river_features(
                request.river_history,
                request.weather_history,
                request.horizon_hours,
            )

        elif forecast_type == ForecastType.FLOOD_RISK:
            features = flood_risk_features(request)

        else:
            raise ValueError(
                "Unsupported forecast type"
            )

        prediction, spec = run_inference(
            forecast_type,
            features,
        )

        risk_level = RiskLevel.UNKNOWN

        if forecast_type == ForecastType.FLOOD_RISK:
            risk_level = classify_risk(prediction)

        result = ForecastResult(
            forecast_id=forecast_id,
            request_id=request.request_id,
            district=request.district,
            forecast_type=forecast_type,
            horizon_hours=request.horizon_hours,
            status=ForecastStatus.COMPLETED,
            predicted_value=prediction,
            unit=spec.output_unit,
            risk_level=risk_level,
            model_name=spec.name,
            model_version=spec.version,
            warnings=[
                "AI-generated advisory forecast. "
                "Not an official emergency warning."
            ],
            metadata={
                "features": features,
                "station_id": request.station_id,
            },
        )

    except ModelUnavailableError as exc:
        result = ForecastResult(
            forecast_id=forecast_id,
            request_id=request.request_id,
            district=request.district,
            forecast_type=forecast_type,
            horizon_hours=request.horizon_hours,
            status=ForecastStatus.MODEL_UNAVAILABLE,
            warnings=[str(exc)],
        )

    except ValueError as exc:
        result = ForecastResult(
            forecast_id=forecast_id,
            request_id=request.request_id,
            district=request.district,
            forecast_type=forecast_type,
            horizon_hours=request.horizon_hours,
            status=ForecastStatus.INSUFFICIENT_DATA,
            warnings=[str(exc)],
        )

    return result


# ============================================================
# CELERY FORECAST TASK
# ============================================================

@celery_app.task(
    bind=True,
    name="forecast.generate",
    max_retries=3,
    default_retry_delay=30,
)
def forecast_task(
    self,
    request_data: dict[str, Any],
    forecast_type: str,
) -> dict[str, Any]:

    request = ForecastRequest.model_validate(
        request_data
    )

    kind = ForecastType(forecast_type)

    cache_key = forecast_cache_key(
        request,
        kind,
    )

    try:
        cached = get_cached_forecast(cache_key)

        if cached:
            cached["cache_hit"] = True
            return cached

        result = generate_forecast(
            request,
            kind,
        )

        save_forecast(result)

        if result.status == ForecastStatus.COMPLETED:
            cache_forecast(
                cache_key,
                result,
            )

        payload = result.model_dump(mode="json")
        payload["cache_hit"] = False

        return payload

    except RedisError as exc:
        logger.exception(
            "Redis error during forecasting"
        )

        raise self.retry(exc=exc)

    except SoftTimeLimitExceeded:
        logger.error(
            "Forecast task exceeded time limit"
        )
        raise


# ============================================================
# RAINFALL FORECAST
# ============================================================

@celery_app.task(
    name="forecast.rainfall",
)
def forecast_rainfall(
    request_data: dict[str, Any],
) -> dict[str, Any]:

    return forecast_task.run(
        request_data,
        ForecastType.RAINFALL.value,
    )


# ============================================================
# RIVER FORECAST
# ============================================================

@celery_app.task(
    name="forecast.river_level",
)
def forecast_river_level(
    request_data: dict[str, Any],
) -> dict[str, Any]:

    return forecast_task.run(
        request_data,
        ForecastType.RIVER_LEVEL.value,
    )


# ============================================================
# FLOOD RISK FORECAST
# ============================================================

@celery_app.task(
    name="forecast.flood_risk",
)
def forecast_flood_risk(
    request_data: dict[str, Any],
) -> dict[str, Any]:

    return forecast_task.run(
        request_data,
        ForecastType.FLOOD_RISK.value,
    )


# ============================================================
# DISTRICT FORECAST
# ============================================================

@celery_app.task(
    name="forecast.district",
)
def forecast_district(
    request_data: dict[str, Any],
) -> dict[str, Any]:

    request = ForecastRequest.model_validate(
        request_data
    )

    results = {}

    for forecast_type in (
        ForecastType.RAINFALL,
        ForecastType.RIVER_LEVEL,
        ForecastType.FLOOD_RISK,
    ):
        result = generate_forecast(
            request,
            forecast_type,
        )

        save_forecast(result)

        results[forecast_type.value] = (
            result.model_dump(mode="json")
        )

    return {
        "district": request.district,
        "request_id": request.request_id,
        "horizon_hours": request.horizon_hours,
        "generated_at": utc_iso(),
        "forecasts": results,
        "official_warning": False,
    }


# ============================================================
# ALERT REVIEW INTEGRATION
# ============================================================

@celery_app.task(
    name="forecast.review_flood_risk",
)
def review_flood_risk(
    forecast_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a recommendation for human review.

    Does not send an alert or trigger rescue dispatch.
    """

    result = ForecastResult.model_validate(
        forecast_result
    )

    if result.status != ForecastStatus.COMPLETED:
        return {
            "review_required": False,
            "reason": "Forecast incomplete",
        }

    if result.forecast_type != ForecastType.FLOOD_RISK:
        return {
            "review_required": False,
            "reason": "Not a flood-risk forecast",
        }

    if result.risk_level not in {
        RiskLevel.HIGH,
        RiskLevel.CRITICAL,
    }:
        return {
            "review_required": False,
            "risk_level": result.risk_level.value,
        }

    review_record = {
        "forecast_id": result.forecast_id,
        "district": result.district,
        "risk_level": result.risk_level.value,
        "predicted_value": result.predicted_value,
        "status": "pending_human_review",
        "created_at": utc_iso(),
        "official_warning": False,
    }

    redis_client.xadd(
        "forecast:human_review",
        {
            "record": json.dumps(review_record)
        },
    )

    return review_record


# ============================================================
# MODEL HEALTH
# ============================================================

@celery_app.task(
    name="forecast.model_health",
)
def model_health() -> dict[str, Any]:

    models = {}

    for forecast_type, spec in MODEL_REGISTRY.items():
        path = MODEL_DIR / spec.filename

        models[forecast_type.value] = {
            "name": spec.name,
            "version": spec.version,
            "installed": path.is_file(),
        }

    return {
        "status": "checked",
        "models": models,
        "timestamp": utc_iso(),
    }


# ============================================================
# WORKER HEALTH
# ============================================================

@celery_app.task(
    name="forecast.worker_health",
)
def worker_health() -> dict[str, Any]:

    redis_client.ping()

    return {
        "worker": "forecast_worker",
        "status": "healthy",
        "timestamp": utc_iso(),
    }
