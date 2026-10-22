from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/workers/weather_worker.py

Weather Data Collection and Monitoring Worker

Features:
- Weather API integration
- Rainfall monitoring
- Temperature and humidity monitoring
- Wind speed and atmospheric pressure
- Kerala district-level weather observations
- Weather data validation
- Redis caching
- Historical weather observations
- Celery background tasks
- Weather anomaly detection
- Forecast worker integration
- Data-source verification
- Worker health monitoring

Safety:
Weather observations are not official flood warnings.
No synthetic observations are used as live weather data.
"""


import hashlib
import json
import logging
import math
import os
import uuid

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import httpx

from celery import Celery
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

WEATHER_CACHE_TTL = 900
WEATHER_HISTORY_TTL = 60 * 60 * 24 * 30

WEATHER_API_TIMEOUT = 20

# Open-Meteo provides weather model/reanalysis data.
# It is not an official IMD flood-warning feed.
WEATHER_API_BASE = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# CELERY
# ============================================================

celery_app = Celery(
    "kerala_flood_weather_worker",
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
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    task_soft_time_limit=90,
    task_time_limit=120,
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

class WeatherSource(str, Enum):
    OPEN_METEO = "open_meteo"
    IMD = "imd"
    WEATHER_STATION = "weather_station"


class WeatherStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class WeatherRisk(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    UNKNOWN = "unknown"


# ============================================================
# KERALA DISTRICT COORDINATES
# ============================================================

# Approximate district headquarters coordinates.
# They do not represent the entire district.

KERALA_DISTRICTS = {
    "thiruvananthapuram": (8.5241, 76.9366),
    "kollam": (8.8932, 76.6141),
    "pathanamthitta": (9.2648, 76.7870),
    "alappuzha": (9.4981, 76.3388),
    "kottayam": (9.5916, 76.5222),
    "idukki": (9.8494, 76.9720),
    "ernakulam": (9.9816, 76.2999),
    "thrissur": (10.5276, 76.2144),
    "palakkad": (10.7867, 76.6548),
    "malappuram": (11.0510, 76.0711),
    "kozhikode": (11.2588, 75.7804),
    "wayanad": (11.6854, 76.1320),
    "kannur": (11.8745, 75.3704),
    "kasaragod": (12.4996, 74.9869),
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
# WEATHER SCHEMAS
# ============================================================

class WeatherLocation(BaseModel):
    district: str

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class WeatherObservation(BaseModel):
    observation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    district: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    source: WeatherSource

    observed_at: datetime

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

    rainfall_mm: float | None = Field(
        default=None,
        ge=0,
    )

    rainfall_period_hours: float | None = Field(
        default=None,
        gt=0,
    )

    wind_speed_kmh: float | None = Field(
        default=None,
        ge=0,
    )

    pressure_hpa: float | None = Field(
        default=None,
        ge=800,
        le=1100,
    )

    cloud_cover_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    weather_code: int | None = None

    is_forecast: bool = False
    verified: bool = False

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @model_validator(mode="after")
    def validate_timestamp(self):
        if self.observed_at.tzinfo is None:
            raise ValueError(
                "Observation timestamp must include timezone"
            )

        if (
            self.rainfall_mm is not None
            and self.rainfall_period_hours is None
        ):
            raise ValueError(
                "Rainfall accumulation period is required"
            )

        return self


class WeatherCollectionResult(BaseModel):
    district: str
    status: WeatherStatus

    observation: WeatherObservation | None = None

    collected_at: datetime = Field(
        default_factory=utc_now
    )

    error: str | None = None

    cache_hit: bool = False


# ============================================================
# CACHE KEYS
# ============================================================

def weather_cache_key(district: str) -> str:
    return f"weather:latest:{district}"


def weather_history_key(district: str) -> str:
    return f"weather:history:{district}"


def weather_lock_key(district: str) -> str:
    return f"weather:lock:{district}"


# ============================================================
# WEATHER DATA PROVIDER
# ============================================================

class WeatherProviderError(RuntimeError):
    pass


def fetch_open_meteo_weather(
    district: str,
) -> WeatherObservation:
    """
    Fetch current model-derived weather data.

    Rainfall is retrieved as the last completed
    hourly precipitation accumulation, not
    interpreted as an instantaneous rainfall rate.

    Data is marked unverified for official use.
    """

    district = normalize_district(district)

    latitude, longitude = KERALA_DISTRICTS[district]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "surface_pressure,"
            "cloud_cover,"
            "weather_code"
        ),
        "hourly": "precipitation",
        "past_hours": 3,
        "forecast_hours": 1,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }

    try:
        with httpx.Client(
            timeout=WEATHER_API_TIMEOUT,
            follow_redirects=False,
        ) as client:
            response = client.get(
                WEATHER_API_BASE,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

    except (
        httpx.HTTPError,
        ValueError,
    ) as exc:
        raise WeatherProviderError(
            "Weather provider request failed"
        ) from exc

    current = data.get("current") or {}
    hourly = data.get("hourly") or {}

    timestamp_text = current.get("time")

    if not timestamp_text:
        raise WeatherProviderError(
            "Weather provider omitted observation time"
        )

    observed_at = datetime.fromisoformat(
        timestamp_text.replace("Z", "+00:00")
    )

    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(
            tzinfo=timezone.utc
        )

    rainfall_mm = None
    rainfall_period_hours = None

    hourly_times = hourly.get("time") or []
    hourly_rainfall = hourly.get("precipitation") or []

    if len(hourly_times) == len(hourly_rainfall):
        completed = []

        for time_text, value in zip(
            hourly_times,
            hourly_rainfall,
        ):
            if value is None:
                continue

            hour_time = datetime.fromisoformat(
                time_text.replace("Z", "+00:00")
            )

            if hour_time.tzinfo is None:
                hour_time = hour_time.replace(
                    tzinfo=timezone.utc
                )

            if hour_time + timedelta(hours=1) <= utc_now():
                completed.append(
                    (hour_time, float(value))
                )

        if completed:
            rainfall_mm = completed[-1][1]
            rainfall_period_hours = 1.0

    return WeatherObservation(
        district=district,
        latitude=latitude,
        longitude=longitude,
        source=WeatherSource.OPEN_METEO,
        observed_at=observed_at,
        temperature_c=current.get("temperature_2m"),
        humidity_percent=current.get(
            "relative_humidity_2m"
        ),
        rainfall_mm=rainfall_mm,
        rainfall_period_hours=rainfall_period_hours,
        wind_speed_kmh=current.get("wind_speed_10m"),
        pressure_hpa=current.get("surface_pressure"),
        cloud_cover_percent=current.get("cloud_cover"),
        weather_code=current.get("weather_code"),
        is_forecast=False,
        verified=False,
    )


# ============================================================
# IMD ADAPTER
# ============================================================

def fetch_imd_weather(
    district: str,
) -> WeatherObservation:
    """
    Integration point for an authorized IMD feed.

    Do not fabricate official IMD measurements.
    """

    raise WeatherProviderError(
        "Official IMD adapter is not configured. "
        "Implement integrations/imd_adapter.py"
    )


# ============================================================
# WEATHER PROVIDER ROUTER
# ============================================================

def fetch_weather(
    district: str,
    source: WeatherSource = WeatherSource.OPEN_METEO,
) -> WeatherObservation:

    if source == WeatherSource.OPEN_METEO:
        return fetch_open_meteo_weather(district)

    if source == WeatherSource.IMD:
        return fetch_imd_weather(district)

    raise WeatherProviderError(
        f"Unsupported automatic weather source: {source}"
    )


# ============================================================
# REDIS STORAGE
# ============================================================

def save_weather(
    observation: WeatherObservation,
) -> None:

    district = observation.district
    payload = observation.model_dump_json()

    redis_client.set(
        weather_cache_key(district),
        payload,
        ex=WEATHER_CACHE_TTL,
    )

    redis_client.zadd(
        weather_history_key(district),
        {
            payload: observation.observed_at.timestamp()
        },
    )

    cutoff = (
        utc_now() - timedelta(days=30)
    ).timestamp()

    redis_client.zremrangebyscore(
        weather_history_key(district),
        "-inf",
        cutoff,
    )

    redis_client.expire(
        weather_history_key(district),
        WEATHER_HISTORY_TTL,
    )


def get_cached_weather(
    district: str,
) -> WeatherObservation | None:

    raw = redis_client.get(
        weather_cache_key(district)
    )

    if not raw:
        return None

    return WeatherObservation.model_validate_json(raw)


def get_weather_history(
    district: str,
    limit: int = 100,
) -> list[dict[str, Any]]:

    district = normalize_district(district)

    records = redis_client.zrevrange(
        weather_history_key(district),
        0,
        max(0, limit - 1),
    )

    return [
        json.loads(record)
        for record in records
    ]


# ============================================================
# DATA QUALITY
# ============================================================

def validate_weather_freshness(
    observation: WeatherObservation,
    max_age_minutes: int = 90,
) -> bool:

    age = utc_now() - observation.observed_at

    return (
        timedelta(minutes=-5)
        <= age
        <= timedelta(minutes=max_age_minutes)
    )


def check_weather_anomalies(
    observation: WeatherObservation,
) -> list[str]:
    """
    Simple data-quality checks.

    These are not official meteorological
    warning thresholds.
    """

    issues = []

    if not validate_weather_freshness(observation):
        issues.append("Observation is stale or future-dated")

    if observation.rainfall_mm is None:
        issues.append("Rainfall measurement unavailable")

    if observation.temperature_c is None:
        issues.append("Temperature measurement unavailable")

    if observation.humidity_percent is None:
        issues.append("Humidity measurement unavailable")

    if observation.wind_speed_kmh is None:
        issues.append("Wind speed measurement unavailable")

    return issues


# ============================================================
# WEATHER COLLECTION TASK
# ============================================================

@celery_app.task(
    bind=True,
    name="weather.collect_district",
    max_retries=3,
    default_retry_delay=30,
)
def collect_district_weather(
    self,
    district: str,
    source: str = "open_meteo",
    force_refresh: bool = False,
) -> dict[str, Any]:

    district = normalize_district(district)

    provider = WeatherSource(source)

    try:
        if not force_refresh:
            cached = get_cached_weather(district)

            if (
                cached
                and cached.source == provider
                and validate_weather_freshness(cached)
            ):
                return WeatherCollectionResult(
                    district=district,
                    status=WeatherStatus.SUCCESS,
                    observation=cached,
                    cache_hit=True,
                ).model_dump(mode="json")

        observation = fetch_weather(
            district,
            provider,
        )

        if not validate_weather_freshness(observation):
            return WeatherCollectionResult(
                district=district,
                status=WeatherStatus.STALE,
                observation=observation,
                error="Weather observation is not current",
            ).model_dump(mode="json")

        save_weather(observation)

        issues = check_weather_anomalies(
            observation
        )

        logger.info(
            "Weather collected: district=%s source=%s issues=%s",
            district,
            provider.value,
            issues,
        )

        return WeatherCollectionResult(
            district=district,
            status=WeatherStatus.SUCCESS,
            observation=observation,
        ).model_dump(mode="json")

    except WeatherProviderError as exc:
        logger.warning(
            "Weather provider unavailable: %s",
            exc,
        )

        return WeatherCollectionResult(
            district=district,
            status=WeatherStatus.UNAVAILABLE,
            error=str(exc),
        ).model_dump(mode="json")

    except RedisError as exc:
        logger.exception(
            "Redis failure during weather collection"
        )

        raise self.retry(
            exc=exc,
            countdown=min(
                30 * (2 ** self.request.retries),
                300,
            ),
        )


# ============================================================
# COLLECT ALL DISTRICTS
# ============================================================

@celery_app.task(
    name="weather.collect_all_districts",
)
def collect_all_districts(
    source: str = "open_meteo",
) -> dict[str, Any]:

    tasks = {}

    for district in sorted(KERALA_DISTRICTS):
        task = collect_district_weather.apply_async(
            kwargs={
                "district": district,
                "source": source,
            }
        )

        tasks[district] = task.id

    return {
        "status": "queued",
        "district_count": len(tasks),
        "tasks": tasks,
        "timestamp": utc_iso(),
    }


# ============================================================
# WEATHER SUMMARY
# ============================================================

@celery_app.task(
    name="weather.district_summary",
)
def district_weather_summary(
    district: str,
) -> dict[str, Any]:

    district = normalize_district(district)

    observation = get_cached_weather(district)

    if observation is None:
        return {
            "district": district,
            "status": WeatherStatus.UNAVAILABLE.value,
            "message": "No weather data available",
        }

    is_fresh = validate_weather_freshness(
        observation
    )

    return {
        "district": district,
        "status": (
            WeatherStatus.SUCCESS.value
            if is_fresh
            else WeatherStatus.STALE.value
        ),
        "temperature_c": observation.temperature_c,
        "humidity_percent": observation.humidity_percent,
        "rainfall_mm": observation.rainfall_mm,
        "rainfall_period_hours": (
            observation.rainfall_period_hours
        ),
        "wind_speed_kmh": observation.wind_speed_kmh,
        "pressure_hpa": observation.pressure_hpa,
        "cloud_cover_percent": (
            observation.cloud_cover_percent
        ),
        "source": observation.source.value,
        "verified": observation.verified,
        "observed_at": observation.observed_at.isoformat(),
        "data_quality_issues": check_weather_anomalies(
            observation
        ),
    }


# ============================================================
# FORECAST INTEGRATION
# ============================================================

@celery_app.task(
    name="weather.prepare_forecast_data",
)
def prepare_forecast_data(
    district: str,
    limit: int = 24,
) -> dict[str, Any]:
    """
    Prepare weather history for forecasting.

    Forecast models must validate observation
    intervals, data completeness, and units.
    """

    district = normalize_district(district)

    records = get_weather_history(
        district,
        limit=limit,
    )

    records.reverse()

    observations = []

    for record in records:
        observation = WeatherObservation.model_validate(
            record
        )

        if observation.rainfall_mm is None:
            continue

        if observation.rainfall_period_hours != 1.0:
            continue

        observations.append({
            "timestamp": observation.observed_at.isoformat(),
            "rainfall_mm": observation.rainfall_mm,
            "temperature_c": observation.temperature_c,
            "humidity_percent": (
                observation.humidity_percent
            ),
            "pressure_hpa": observation.pressure_hpa,
            "wind_speed_mps": (
                observation.wind_speed_kmh / 3.6
                if observation.wind_speed_kmh is not None
                else None
            ),
        })

    return {
        "district": district,
        "observation_count": len(observations),
        "weather_history": observations,
        "source_verified": False,
        "ready_for_forecast": len(observations) >= 3,
    }


# ============================================================
# WORKER HEALTH
# ============================================================

@celery_app.task(
    name="weather.worker_health",
)
def worker_health() -> dict[str, Any]:

    redis_client.ping()

    return {
        "worker": "weather_worker",
        "status": "healthy",
        "timestamp": utc_iso(),
        "districts_supported": len(KERALA_DISTRICTS),
        "weather_provider": "open_meteo",
        "official_warning_source": False,
    }


# ============================================================
# RUN WORKER
# ============================================================

if __name__ == "__main__":
    print(
        "Run with:\n"
        "celery -A backend.workers.weather_worker.celery_app "
        "worker --loglevel=info --pool=solo"
    )
