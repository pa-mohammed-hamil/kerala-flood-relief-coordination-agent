"""Placeholder module: integrations/weather_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Weather Integration Adapter

File: backend/integrations/weather_adapter.py

Python 3.11+
Dependency: pip install httpx

Supports:
    - Weather observations
    - Hourly forecasts
    - Rainfall records
    - Weather warnings
    - Kerala district filtering
    - Configurable provider endpoints
    - Configurable JSON field mapping
    - Data freshness validation
    - Rainfall aggregation
    - Advisory rainfall indicators
    - RAG record preparation
    - Async HTTP, retries, timeouts
    - Source provenance and audit events

IMPORTANT:
    Provider field mappings are illustrative.
    Configure them against the actual authorized
    API response contract.

    Rainfall indicators are NOT official IMD
    warning classifications or flood predictions.

    Never automatically issue evacuation orders,
    dispatch rescue teams, or publish warnings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    httpx = None


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================
# EXCEPTIONS
# ============================================================

class WeatherAdapterError(Exception):
    pass


class WeatherConfigurationError(WeatherAdapterError):
    pass


class WeatherValidationError(WeatherAdapterError):
    pass


class WeatherAuthenticationError(WeatherAdapterError):
    pass


class WeatherRequestError(WeatherAdapterError):
    pass


class WeatherResponseError(WeatherAdapterError):
    pass


class WeatherSourceError(WeatherAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class WeatherAuthType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"


class WeatherRecordType(str, Enum):
    OBSERVATION = "observation"
    FORECAST = "forecast"
    RAINFALL = "rainfall"
    WARNING = "warning"


class WeatherDataStatus(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    FUTURE_TIMESTAMP = "future_timestamp"


class WeatherVerification(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class WeatherWarningLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    UNKNOWN = "unknown"


class RainfallIndicator(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"


# ============================================================
# DISTRICTS
# ============================================================

KERALA_DISTRICTS = (
    "Thiruvananthapuram",
    "Kollam",
    "Pathanamthitta",
    "Alappuzha",
    "Kottayam",
    "Idukki",
    "Ernakulam",
    "Thrissur",
    "Palakkad",
    "Malappuram",
    "Kozhikode",
    "Wayanad",
    "Kannur",
    "Kasaragod",
)

DISTRICT_LOOKUP = {
    name.casefold(): name
    for name in KERALA_DISTRICTS
}


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def normalize_district(value: str) -> str:
    if not isinstance(value, str):
        raise WeatherValidationError(
            "District must be a string"
        )

    result = DISTRICT_LOOKUP.get(
        value.strip().casefold()
    )

    if result is None:
        raise WeatherValidationError(
            "Unknown Kerala district"
        )

    return result


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value

    elif isinstance(value, str):
        text = value.strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise WeatherValidationError(
                "Invalid ISO timestamp"
            ) from exc

    else:
        raise WeatherValidationError(
            "Unsupported timestamp"
        )

    if (
        result.tzinfo is None
        or result.utcoffset() is None
    ):
        raise WeatherValidationError(
            "Timestamp must include timezone"
        )

    return result.astimezone(timezone.utc)


def optional_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None

    return parse_timestamp(value)


def optional_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        raise WeatherValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise WeatherValidationError(
            f"Invalid {name}"
        ) from exc

    if not math.isfinite(number):
        raise WeatherValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and number < minimum:
        raise WeatherValidationError(
            f"{name} below minimum"
        )

    if maximum is not None and number > maximum:
        raise WeatherValidationError(
            f"{name} above maximum"
        )

    return number


def nested_get(
    data: Mapping[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, Mapping):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class WeatherAdapterConfig:
    base_url: str

    observations_endpoint: str = "/observations"
    forecasts_endpoint: str = "/forecasts"
    rainfall_endpoint: str = "/rainfall"
    warnings_endpoint: str = "/warnings"

    auth_type: WeatherAuthType = WeatherAuthType.NONE

    api_key: str | None = None
    api_key_header: str = "X-API-Key"

    bearer_token: str | None = None

    timeout_seconds: float = 20.0

    max_retries: int = 2
    retry_base_seconds: float = 0.5

    max_response_bytes: int = 5_000_000
    max_records: int = 5000

    observation_max_age_minutes: int = 180
    forecast_max_age_minutes: int = 360
    warning_max_age_minutes: int = 180

    source_verified: bool = False
    require_verified_source: bool = True

    allowed_hosts: tuple[str, ...] = ()

    allow_local_http_testing: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)

        if not parsed.hostname:
            raise WeatherConfigurationError(
                "Invalid base URL"
            )

        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise WeatherConfigurationError(
                "Base URL cannot contain credentials, "
                "query or fragment"
            )

        if parsed.scheme != "https":
            local_testing = (
                self.allow_local_http_testing
                and parsed.scheme == "http"
                and parsed.hostname in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
            )

            if not local_testing:
                raise WeatherConfigurationError(
                    "HTTPS required"
                )

        if (
            self.allowed_hosts
            and parsed.hostname not in self.allowed_hosts
        ):
            raise WeatherConfigurationError(
                "Provider host not allowlisted"
            )

        for endpoint in (
            self.observations_endpoint,
            self.forecasts_endpoint,
            self.rainfall_endpoint,
            self.warnings_endpoint,
        ):
            if (
                not endpoint.startswith("/")
                or endpoint.startswith("//")
                or "://" in endpoint
                or "?" in endpoint
                or "#" in endpoint
                or ".." in endpoint.split("/")
            ):
                raise WeatherConfigurationError(
                    "Invalid endpoint path"
                )

        if (
            self.auth_type == WeatherAuthType.API_KEY
            and not self.api_key
        ):
            raise WeatherConfigurationError(
                "API key required"
            )

        if (
            self.auth_type == WeatherAuthType.BEARER
            and not self.bearer_token
        ):
            raise WeatherConfigurationError(
                "Bearer token required"
            )

        if self.timeout_seconds <= 0:
            raise WeatherConfigurationError(
                "Timeout must be positive"
            )

        if self.max_retries < 0:
            raise WeatherConfigurationError(
                "Retries cannot be negative"
            )

        for name in (
            "max_response_bytes",
            "max_records",
            "observation_max_age_minutes",
            "forecast_max_age_minutes",
            "warning_max_age_minutes",
        ):
            if getattr(self, name) <= 0:
                raise WeatherConfigurationError(
                    f"{name} must be positive"
                )


# ============================================================
# FIELD MAPPING
# ============================================================

@dataclass(frozen=True)
class WeatherFieldMapping:
    """
    Generic JSON mapping.

    Example expected provider structure:

    {
        "data": [
            {
                "station_id": "ST001",
                "district": "Thrissur",
                "observed_at": "...+05:30",
                "rainfall_mm": 12.5
            }
        ]
    }

    Actual IMD field names may differ.
    """

    records_path: str = "data"

    record_id: str = "id"
    station_id: str = "station_id"
    station_name: str = "station_name"

    district: str = "district"

    observed_at: str = "observed_at"
    issued_at: str = "issued_at"
    valid_at: str = "valid_at"
    expires_at: str = "expires_at"

    temperature_c: str = "temperature_c"
    humidity_percent: str = "humidity_percent"

    rainfall_mm: str = "rainfall_mm"
    rainfall_period_hours: str = "rainfall_period_hours"

    wind_speed_kmph: str = "wind_speed_kmph"
    wind_direction_degrees: str = "wind_direction_degrees"

    pressure_hpa: str = "pressure_hpa"

    condition: str = "condition"

    warning_level: str = "warning_level"
    warning_headline: str = "headline"
    warning_description: str = "description"
    warning_instruction: str = "instruction"

    source_url: str = "source_url"


# ============================================================
# MODELS
# ============================================================

@dataclass(frozen=True)
class WeatherObservation:
    observation_id: str

    station_id: str
    station_name: str

    district: str

    observed_at: datetime

    temperature_c: float | None
    humidity_percent: float | None

    rainfall_mm: float | None
    rainfall_period_hours: float | None

    wind_speed_kmph: float | None
    wind_direction_degrees: float | None

    pressure_hpa: float | None

    condition: str | None

    data_status: WeatherDataStatus

    verification: WeatherVerification

    source_uri: str

    retrieved_at: datetime

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class WeatherForecast:
    forecast_id: str

    district: str

    issued_at: datetime
    valid_at: datetime

    temperature_c: float | None
    rainfall_mm: float | None

    rainfall_period_hours: float | None

    wind_speed_kmph: float | None

    condition: str | None

    data_status: WeatherDataStatus

    verification: WeatherVerification

    source_uri: str

    retrieved_at: datetime


@dataclass(frozen=True)
class RainfallRecord:
    rainfall_id: str

    district: str

    observed_at: datetime

    rainfall_mm: float

    accumulation_hours: float

    station_id: str | None

    data_status: WeatherDataStatus

    verification: WeatherVerification

    source_uri: str

    retrieved_at: datetime


@dataclass(frozen=True)
class WeatherWarning:
    warning_id: str

    district: str

    level: WeatherWarningLevel

    headline: str
    description: str
    instruction: str

    issued_at: datetime

    valid_at: datetime
    expires_at: datetime

    data_status: WeatherDataStatus

    verification: WeatherVerification

    source_uri: str

    retrieved_at: datetime

    requires_human_review: bool = True


@dataclass(frozen=True)
class WeatherAuditEvent:
    event_id: str

    action: str

    record_type: WeatherRecordType

    timestamp: datetime

    record_count: int

    status: str


@dataclass(frozen=True)
class RainfallSummary:
    district: str

    station_count: int

    total_records: int

    mean_station_rainfall_mm: float | None

    maximum_station_rainfall_mm: float | None

    accumulation_hours: float

    indicator: RainfallIndicator

    requires_human_review: bool = True


# ============================================================
# SOURCE AND FRESHNESS
# ============================================================

class WeatherSourcePolicy:
    def __init__(
        self,
        config: WeatherAdapterConfig,
    ) -> None:
        self.config = config

    @property
    def verification(self) -> WeatherVerification:
        return (
            WeatherVerification.VERIFIED
            if self.config.source_verified
            else WeatherVerification.UNVERIFIED
        )

    def require_verified(self) -> None:
        if (
            self.config.require_verified_source
            and not self.config.source_verified
        ):
            raise WeatherSourceError(
                "Weather source not independently verified"
            )

    @staticmethod
    def freshness(
        timestamp: datetime,
        *,
        now: datetime,
        max_age_minutes: int,
    ) -> WeatherDataStatus:
        age = now - timestamp

        if age < timedelta(0):
            return WeatherDataStatus.FUTURE_TIMESTAMP

        if age > timedelta(
            minutes=max_age_minutes
        ):
            return WeatherDataStatus.STALE

        return WeatherDataStatus.CURRENT


# ============================================================
# HTTP CLIENT
# ============================================================

class WeatherHTTPClient:
    RETRYABLE_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        config: WeatherAdapterConfig,
    ) -> None:
        if httpx is None:
            raise WeatherConfigurationError(
                "Install httpx: pip install httpx"
            )

        self.config = config

        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": (
                "KeralaFloodReliefCoordinationAgent/1.0"
            ),
        }

        if (
            self.config.auth_type
            == WeatherAuthType.API_KEY
        ):
            headers[
                self.config.api_key_header
            ] = self.config.api_key or ""

        elif (
            self.config.auth_type
            == WeatherAuthType.BEARER
        ):
            headers["Authorization"] = (
                f"Bearer {self.config.bearer_token}"
            )

        return headers

    async def _get_client(
        self,
    ) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers=self._headers(),
                follow_redirects=False,
                trust_env=False,
            )

        return self._client

    async def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        client = await self._get_client()

        url = (
            self.config.base_url.rstrip("/")
            + endpoint
        )

        last_error: Exception | None = None

        for attempt in range(
            self.config.max_retries + 1
        ):
            try:
                async with client.stream(
                    "GET",
                    url,
                    params=params,
                ) as response:
                    if response.status_code in (401, 403):
                        raise WeatherAuthenticationError(
                            "Weather API authentication failed"
                        )

                    if (
                        response.status_code
                        in self.RETRYABLE_CODES
                    ):
                        raise WeatherRequestError(
                            "Temporary weather API failure"
                        )

                    response.raise_for_status()

                    chunks = []
                    total = 0

                    async for chunk in response.aiter_bytes():
                        total += len(chunk)

                        if total > (
                            self.config.max_response_bytes
                        ):
                            raise WeatherResponseError(
                                "Weather response too large"
                            )

                        chunks.append(chunk)

                return json.loads(
                    b"".join(chunks)
                )

            except (
                WeatherAuthenticationError,
                WeatherResponseError,
            ):
                raise

            except (
                WeatherRequestError,
                httpx.TimeoutException,
                httpx.TransportError,
            ) as exc:
                last_error = exc

                if attempt >= self.config.max_retries:
                    break

                delay = (
                    self.config.retry_base_seconds
                    * (2 ** attempt)
                )

                delay += random.uniform(
                    0,
                    min(1.0, delay * 0.2),
                )

                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as exc:
                raise WeatherRequestError(
                    f"Weather API HTTP "
                    f"{exc.response.status_code}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise WeatherResponseError(
                    "Invalid weather JSON"
                ) from exc

        raise WeatherRequestError(
            "Weather API request failed"
        ) from last_error

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ============================================================
# RESPONSE PARSER
# ============================================================

class WeatherResponseParser:
    def __init__(
        self,
        config: WeatherAdapterConfig,
        mapping: WeatherFieldMapping,
    ) -> None:
        self.config = config
        self.mapping = mapping

        self.policy = WeatherSourcePolicy(
            config
        )

    def _records(
        self,
        payload: Any,
    ) -> list[Mapping[str, Any]]:
        if isinstance(payload, list):
            records = payload

        elif isinstance(payload, Mapping):
            records = nested_get(
                payload,
                self.mapping.records_path,
            )

        else:
            raise WeatherResponseError(
                "Invalid weather response"
            )

        if not isinstance(records, list):
            raise WeatherResponseError(
                "Expected weather records list"
            )

        if len(records) > self.config.max_records:
            raise WeatherResponseError(
                "Weather record limit exceeded"
            )

        if not all(
            isinstance(item, Mapping)
            for item in records
        ):
            raise WeatherResponseError(
                "Invalid weather record structure"
            )

        return records

    def _district(
        self,
        item: Mapping[str, Any],
    ) -> str:
        return normalize_district(
            str(
                nested_get(
                    item,
                    self.mapping.district,
                    "",
                )
            )
        )

    def _source_uri(
        self,
        item: Mapping[str, Any],
    ) -> str:
        candidate = nested_get(
            item,
            self.mapping.source_url,
        )

        if isinstance(candidate, str):
            parsed = urlparse(candidate)

            if (
                parsed.scheme == "https"
                and parsed.hostname
                == urlparse(
                    self.config.base_url
                ).hostname
            ):
                return candidate

        return self.config.base_url

    def parse_observations(
        self,
        payload: Any,
    ) -> list[WeatherObservation]:
        m = self.mapping
        now = utc_now()

        output = []

        for item in self._records(payload):
            observed_at = parse_timestamp(
                nested_get(
                    item,
                    m.observed_at,
                )
            )

            output.append(
                WeatherObservation(
                    observation_id=str(
                        nested_get(
                            item,
                            m.record_id,
                            new_id(),
                        )
                    ),
                    station_id=str(
                        nested_get(
                            item,
                            m.station_id,
                            "",
                        )
                    ),
                    station_name=str(
                        nested_get(
                            item,
                            m.station_name,
                            "",
                        )
                    ),
                    district=self._district(item),
                    observed_at=observed_at,
                    temperature_c=optional_number(
                        nested_get(
                            item,
                            m.temperature_c,
                        ),
                        "temperature",
                    ),
                    humidity_percent=optional_number(
                        nested_get(
                            item,
                            m.humidity_percent,
                        ),
                        "humidity",
                        minimum=0,
                        maximum=100,
                    ),
                    rainfall_mm=optional_number(
                        nested_get(
                            item,
                            m.rainfall_mm,
                        ),
                        "rainfall",
                        minimum=0,
                    ),
                    rainfall_period_hours=optional_number(
                        nested_get(
                            item,
                            m.rainfall_period_hours,
                        ),
                        "rainfall period",
                        minimum=0,
                    ),
                    wind_speed_kmph=optional_number(
                        nested_get(
                            item,
                            m.wind_speed_kmph,
                        ),
                        "wind speed",
                        minimum=0,
                    ),
                    wind_direction_degrees=optional_number(
                        nested_get(
                            item,
                            m.wind_direction_degrees,
                        ),
                        "wind direction",
                        minimum=0,
                        maximum=360,
                    ),
                    pressure_hpa=optional_number(
                        nested_get(
                            item,
                            m.pressure_hpa,
                        ),
                        "pressure",
                        minimum=0,
                    ),
                    condition=nested_get(
                        item,
                        m.condition,
                    ),
                    data_status=self.policy.freshness(
                        observed_at,
                        now=now,
                        max_age_minutes=(
                            self.config
                            .observation_max_age_minutes
                        ),
                    ),
                    verification=(
                        self.policy.verification
                    ),
                    source_uri=self._source_uri(
                        item
                    ),
                    retrieved_at=now,
                )
            )

        return output

    def parse_forecasts(
        self,
        payload: Any,
    ) -> list[WeatherForecast]:
        m = self.mapping
        now = utc_now()

        output = []

        for item in self._records(payload):
            issued_at = parse_timestamp(
                nested_get(
                    item,
                    m.issued_at,
                )
            )

            valid_at = parse_timestamp(
                nested_get(
                    item,
                    m.valid_at,
                )
            )

            output.append(
                WeatherForecast(
                    forecast_id=str(
                        nested_get(
                            item,
                            m.record_id,
                            new_id(),
                        )
                    ),
                    district=self._district(item),
                    issued_at=issued_at,
                    valid_at=valid_at,
                    temperature_c=optional_number(
                        nested_get(
                            item,
                            m.temperature_c,
                        ),
                        "temperature",
                    ),
                    rainfall_mm=optional_number(
                        nested_get(
                            item,
                            m.rainfall_mm,
                        ),
                        "forecast rainfall",
                        minimum=0,
                    ),
                    rainfall_period_hours=optional_number(
                        nested_get(
                            item,
                            m.rainfall_period_hours,
                        ),
                        "forecast period",
                        minimum=0,
                    ),
                    wind_speed_kmph=optional_number(
                        nested_get(
                            item,
                            m.wind_speed_kmph,
                        ),
                        "wind speed",
                        minimum=0,
                    ),
                    condition=nested_get(
                        item,
                        m.condition,
                    ),
                    data_status=self.policy.freshness(
                        issued_at,
                        now=now,
                        max_age_minutes=(
                            self.config
                            .forecast_max_age_minutes
                        ),
                    ),
                    verification=(
                        self.policy.verification
                    ),
                    source_uri=self._source_uri(
                        item
                    ),
                    retrieved_at=now,
                )
            )

        return output

    def parse_rainfall(
        self,
        payload: Any,
    ) -> list[RainfallRecord]:
        m = self.mapping
        now = utc_now()

        output = []

        for item in self._records(payload):
            observed_at = parse_timestamp(
                nested_get(
                    item,
                    m.observed_at,
                )
            )

            rainfall = optional_number(
                nested_get(
                    item,
                    m.rainfall_mm,
                ),
                "rainfall",
                minimum=0,
            )

            hours = optional_number(
                nested_get(
                    item,
                    m.rainfall_period_hours,
                ),
                "accumulation period",
                minimum=0.001,
            )

            if rainfall is None or hours is None:
                raise WeatherResponseError(
                    "Rainfall amount and accumulation "
                    "period are required"
                )

            output.append(
                RainfallRecord(
                    rainfall_id=str(
                        nested_get(
                            item,
                            m.record_id,
                            new_id(),
                        )
                    ),
                    district=self._district(item),
                    observed_at=observed_at,
                    rainfall_mm=rainfall,
                    accumulation_hours=hours,
                    station_id=nested_get(
                        item,
                        m.station_id,
                    ),
                    data_status=self.policy.freshness(
                        observed_at,
                        now=now,
                        max_age_minutes=(
                            self.config
                            .observation_max_age_minutes
                        ),
                    ),
                    verification=(
                        self.policy.verification
                    ),
                    source_uri=self._source_uri(
                        item
                    ),
                    retrieved_at=now,
                )
            )

        return output

    def parse_warnings(
        self,
        payload: Any,
    ) -> list[WeatherWarning]:
        m = self.mapping
        now = utc_now()

        output = []

        for item in self._records(payload):
            issued_at = parse_timestamp(
                nested_get(
                    item,
                    m.issued_at,
                )
            )

            valid_at = parse_timestamp(
                nested_get(
                    item,
                    m.valid_at,
                )
            )

            expires_at = parse_timestamp(
                nested_get(
                    item,
                    m.expires_at,
                )
            )

            if expires_at <= valid_at:
                raise WeatherValidationError(
                    "Invalid warning validity period"
                )

            raw_level = str(
                nested_get(
                    item,
                    m.warning_level,
                    "unknown",
                )
            ).casefold()

            try:
                level = WeatherWarningLevel(
                    raw_level
                )
            except ValueError:
                level = WeatherWarningLevel.UNKNOWN

            output.append(
                WeatherWarning(
                    warning_id=str(
                        nested_get(
                            item,
                            m.record_id,
                            new_id(),
                        )
                    ),
                    district=self._district(item),
                    level=level,
                    headline=str(
                        nested_get(
                            item,
                            m.warning_headline,
                            "",
                        )
                    ),
                    description=str(
                        nested_get(
                            item,
                            m.warning_description,
                            "",
                        )
                    ),
                    instruction=str(
                        nested_get(
                            item,
                            m.warning_instruction,
                            "",
                        )
                    ),
                    issued_at=issued_at,
                    valid_at=valid_at,
                    expires_at=expires_at,
                    data_status=self.policy.freshness(
                        issued_at,
                        now=now,
                        max_age_minutes=(
                            self.config
                            .warning_max_age_minutes
                        ),
                    ),
                    verification=(
                        self.policy.verification
                    ),
                    source_uri=self._source_uri(
                        item
                    ),
                    retrieved_at=now,
                    requires_human_review=True,
                )
            )

        return output


# ============================================================
# RAINFALL ANALYTICS
# ============================================================

class RainfallAnalytics:
    """
    Advisory analytics only.

    These configurable thresholds are not
    official IMD alert thresholds.

    Rainfall from different stations must not
    be added together to estimate district
    rainfall accumulation.
    """

    def __init__(
        self,
        *,
        elevated_mm_24h: float = 30.0,
        high_mm_24h: float = 60.0,
        very_high_mm_24h: float = 120.0,
    ) -> None:
        if not (
            0
            < elevated_mm_24h
            < high_mm_24h
            < very_high_mm_24h
        ):
            raise WeatherValidationError(
                "Invalid rainfall thresholds"
            )

        self.elevated = elevated_mm_24h
        self.high = high_mm_24h
        self.very_high = very_high_mm_24h

    def indicator(
        self,
        rainfall_mm: float,
        *,
        accumulation_hours: float,
    ) -> RainfallIndicator:
        if accumulation_hours != 24:
            return RainfallIndicator.UNKNOWN

        if rainfall_mm >= self.very_high:
            return RainfallIndicator.VERY_HIGH

        if rainfall_mm >= self.high:
            return RainfallIndicator.HIGH

        if rainfall_mm >= self.elevated:
            return RainfallIndicator.ELEVATED

        return RainfallIndicator.LOW

    def summarize_district(
        self,
        records: Sequence[RainfallRecord],
        *,
        district: str,
        accumulation_hours: float = 24,
    ) -> RainfallSummary:
        district = normalize_district(
            district
        )

        eligible = [
            record
            for record in records
            if (
                record.district == district
                and record.accumulation_hours
                == accumulation_hours
                and record.data_status
                == WeatherDataStatus.CURRENT
                and record.verification
                == WeatherVerification.VERIFIED
            )
        ]

        # Retain latest record per station.
        latest: dict[str, RainfallRecord] = {}

        for record in eligible:
            station = record.station_id

            if not station:
                continue

            existing = latest.get(station)

            if (
                existing is None
                or record.observed_at
                > existing.observed_at
            ):
                latest[station] = record

        amounts = [
            record.rainfall_mm
            for record in latest.values()
        ]

        if not amounts:
            return RainfallSummary(
                district=district,
                station_count=0,
                total_records=len(eligible),
                mean_station_rainfall_mm=None,
                maximum_station_rainfall_mm=None,
                accumulation_hours=accumulation_hours,
                indicator=RainfallIndicator.UNKNOWN,
            )

        mean_rainfall = (
            sum(amounts) / len(amounts)
        )

        maximum = max(amounts)

        return RainfallSummary(
            district=district,
            station_count=len(amounts),
            total_records=len(eligible),
            mean_station_rainfall_mm=mean_rainfall,
            maximum_station_rainfall_mm=maximum,
            accumulation_hours=accumulation_hours,
            indicator=self.indicator(
                maximum,
                accumulation_hours=accumulation_hours,
            ),
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class WeatherAuditLogger:
    def __init__(
        self,
        max_events: int = 1000,
    ) -> None:
        self.max_events = max_events

        self._events: list[
            WeatherAuditEvent
        ] = []

    def record(
        self,
        *,
        action: str,
        record_type: WeatherRecordType,
        record_count: int,
        status: str,
    ) -> None:
        self._events.append(
            WeatherAuditEvent(
                event_id=new_id(),
                action=action,
                record_type=record_type,
                timestamp=utc_now(),
                record_count=record_count,
                status=status,
            )
        )

        if len(self._events) > self.max_events:
            self._events = self._events[
                -self.max_events:
            ]

    def events(self) -> list[WeatherAuditEvent]:
        return list(self._events)


# ============================================================
# MAIN WEATHER ADAPTER
# ============================================================

class WeatherAdapter:
    def __init__(
        self,
        config: WeatherAdapterConfig,
        *,
        mapping: WeatherFieldMapping | None = None,
    ) -> None:
        self.config = config

        self.mapping = (
            mapping or WeatherFieldMapping()
        )

        self.http_client = WeatherHTTPClient(
            config
        )

        self.parser = WeatherResponseParser(
            config,
            self.mapping,
        )

        self.source_policy = WeatherSourcePolicy(
            config
        )

        self.rainfall_analytics = RainfallAnalytics()

        self.audit = WeatherAuditLogger()

    async def get_observations(
        self,
        *,
        district: str | None = None,
        current_only: bool = True,
    ) -> list[WeatherObservation]:
        self.source_policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.observations_endpoint,
            params=params,
        )

        records = self.parser.parse_observations(
            payload
        )

        if district:
            selected = normalize_district(district)

            records = [
                item
                for item in records
                if item.district == selected
            ]

        if current_only:
            records = [
                item
                for item in records
                if item.data_status
                == WeatherDataStatus.CURRENT
            ]

        self.audit.record(
            action="get_observations",
            record_type=WeatherRecordType.OBSERVATION,
            record_count=len(records),
            status="completed",
        )

        return records

    async def get_forecasts(
        self,
        *,
        district: str | None = None,
        current_only: bool = True,
    ) -> list[WeatherForecast]:
        self.source_policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.forecasts_endpoint,
            params=params,
        )

        records = self.parser.parse_forecasts(
            payload
        )

        if district:
            selected = normalize_district(district)

            records = [
                item
                for item in records
                if item.district == selected
            ]

        if current_only:
            records = [
                item
                for item in records
                if item.data_status
                == WeatherDataStatus.CURRENT
            ]

        self.audit.record(
            action="get_forecasts",
            record_type=WeatherRecordType.FORECAST,
            record_count=len(records),
            status="completed",
        )

        return records

    async def get_rainfall(
        self,
        *,
        district: str | None = None,
        current_only: bool = True,
    ) -> list[RainfallRecord]:
        self.source_policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.rainfall_endpoint,
            params=params,
        )

        records = self.parser.parse_rainfall(
            payload
        )

        if district:
            selected = normalize_district(district)

            records = [
                item
                for item in records
                if item.district == selected
            ]

        if current_only:
            records = [
                item
                for item in records
                if item.data_status
                == WeatherDataStatus.CURRENT
            ]

        self.audit.record(
            action="get_rainfall",
            record_type=WeatherRecordType.RAINFALL,
            record_count=len(records),
            status="completed",
        )

        return records

    async def get_warnings(
        self,
        *,
        district: str | None = None,
        active_only: bool = True,
    ) -> list[WeatherWarning]:
        self.source_policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.warnings_endpoint,
            params=params,
        )

        records = self.parser.parse_warnings(
            payload
        )

        if district:
            selected = normalize_district(district)

            records = [
                item
                for item in records
                if item.district == selected
            ]

        if active_only:
            now = utc_now()

            records = [
                item
                for item in records
                if (
                    item.valid_at <= now < item.expires_at
                    and item.data_status
                    == WeatherDataStatus.CURRENT
                    and item.verification
                    == WeatherVerification.VERIFIED
                )
            ]

        self.audit.record(
            action="get_warnings",
            record_type=WeatherRecordType.WARNING,
            record_count=len(records),
            status="completed",
        )

        return records

    async def get_district_snapshot(
        self,
        district: str,
    ) -> dict[str, Any]:
        selected = normalize_district(
            district
        )

        observations = await self.get_observations(
            district=selected
        )

        forecasts = await self.get_forecasts(
            district=selected
        )

        rainfall = await self.get_rainfall(
            district=selected
        )

        warnings = await self.get_warnings(
            district=selected
        )

        rainfall_summary = (
            self.rainfall_analytics.summarize_district(
                rainfall,
                district=selected,
            )
        )

        return {
            "district": selected,
            "retrieved_at": utc_now().isoformat(),
            "observations": json_safe(observations),
            "forecasts": json_safe(forecasts),
            "rainfall": json_safe(rainfall),
            "rainfall_summary": json_safe(
                rainfall_summary
            ),
            "warnings": json_safe(warnings),
            "source_verified": (
                self.config.source_verified
            ),
            "advisory_only": True,
            "requires_human_review": True,
        }

    async def close(self) -> None:
        await self.http_client.close()

    async def __aenter__(
        self,
    ) -> "WeatherAdapter":
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.close()


# ============================================================
# RAG INTEGRATION
# ============================================================

class WeatherRAGAdapter:
    @staticmethod
    def observations_to_records(
        observations: Sequence[WeatherObservation],
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        output = []

        for observation in observations:
            if (
                observation.verification
                != WeatherVerification.VERIFIED
                or observation.data_status
                != WeatherDataStatus.CURRENT
            ):
                continue

            output.append({
                "id": (
                    f"weather:{observation.observation_id}"
                ),
                "content": (
                    f"Weather observation for "
                    f"{observation.district}. "
                    f"Station: {observation.station_name}. "
                    f"Temperature: "
                    f"{observation.temperature_c} C. "
                    f"Rainfall: "
                    f"{observation.rainfall_mm} mm "
                    f"over "
                    f"{observation.rainfall_period_hours} "
                    f"hours. "
                    f"Humidity: "
                    f"{observation.humidity_percent}%. "
                    f"Observed at: "
                    f"{observation.observed_at.isoformat()}."
                ),
                "metadata": {
                    "tenant_id": tenant_id,
                    "source": "weather",
                    "source_uri": (
                        observation.source_uri
                    ),
                    "district": (
                        observation.district
                    ),
                    "station_id": (
                        observation.station_id
                    ),
                    "observed_at": (
                        observation.observed_at.isoformat()
                    ),
                    "verification": "verified",
                    "requires_human_review": True,
                },
            })

        return output

    @staticmethod
    def warnings_to_records(
        warnings: Sequence[WeatherWarning],
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        output = []
        now = utc_now()

        for warning in warnings:
            if not (
                warning.verification
                == WeatherVerification.VERIFIED
                and warning.data_status
                == WeatherDataStatus.CURRENT
                and warning.valid_at
                <= now
                < warning.expires_at
            ):
                continue

            output.append({
                "id": (
                    f"weather-warning:{warning.warning_id}"
                ),
                "content": (
                    f"Weather warning for "
                    f"{warning.district}. "
                    f"Level: {warning.level.value}. "
                    f"Headline: {warning.headline}. "
                    f"Description: "
                    f"{warning.description}. "
                    f"Instructions: "
                    f"{warning.instruction}. "
                    f"Expires: "
                    f"{warning.expires_at.isoformat()}."
                ),
                "metadata": {
                    "tenant_id": tenant_id,
                    "source": "weather_warning",
                    "source_uri": warning.source_uri,
                    "district": warning.district,
                    "warning_level": (
                        warning.level.value
                    ),
                    "expires_at": (
                        warning.expires_at.isoformat()
                    ),
                    "verification": "verified",
                    "requires_human_review": True,
                },
            })

        return output


# ============================================================
# HUMAN REVIEW ADAPTER
# ============================================================

class WeatherNotificationReviewAdapter:
    @staticmethod
    def prepare_reviews(
        warnings: Sequence[WeatherWarning],
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        output = []
        now = utc_now()

        for warning in warnings:
            if not (
                warning.verification
                == WeatherVerification.VERIFIED
                and warning.data_status
                == WeatherDataStatus.CURRENT
                and warning.valid_at
                <= now
                < warning.expires_at
            ):
                continue

            output.append({
                "review_id": new_id(),
                "tenant_id": tenant_id,
                "district": warning.district,
                "warning_id": warning.warning_id,
                "warning_level": (
                    warning.level.value
                ),
                "headline": warning.headline,
                "description": warning.description,
                "instruction": warning.instruction,
                "source_uri": warning.source_uri,
                "review_status": "pending",
                "requires_human_approval": True,
                "automatically_publish": False,
                "automatically_dispatch": False,
            })

        return output


# ============================================================
# FACTORY
# ============================================================

def create_weather_adapter(
    config: WeatherAdapterConfig,
    *,
    mapping: WeatherFieldMapping | None = None,
) -> WeatherAdapter:
    return WeatherAdapter(
        config=config,
        mapping=mapping,
    )


# ============================================================
# SYNTHETIC DEMONSTRATION
# ============================================================

def demo() -> None:
    """
    Synthetic example.
    No live weather API requests.
    """

    config = WeatherAdapterConfig(
        base_url="https://example.invalid",
        source_verified=False,
        require_verified_source=False,
    )

    parser = WeatherResponseParser(
        config,
        WeatherFieldMapping(),
    )

    now = utc_now()

    payload = {
        "data": [
            {
                "id": "demo-obs-001",
                "station_id": "DEMO-THR-001",
                "station_name": "Demo Station",
                "district": "Thrissur",
                "observed_at": now.isoformat(),
                "temperature_c": 27.5,
                "humidity_percent": 88,
                "rainfall_mm": 42.0,
                "rainfall_period_hours": 24,
                "wind_speed_kmph": 12.0,
                "wind_direction_degrees": 180,
                "pressure_hpa": 1008,
                "condition": "Synthetic rain",
            }
        ]
    }

    observations = parser.parse_observations(
        payload
    )

    print(
        json.dumps(
            json_safe(observations),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    demo()


__all__ = [
    "WeatherAdapterError",
    "WeatherConfigurationError",
    "WeatherValidationError",
    "WeatherAuthenticationError",
    "WeatherRequestError",
    "WeatherResponseError",
    "WeatherSourceError",
    "WeatherAuthType",
    "WeatherRecordType",
    "WeatherDataStatus",
    "WeatherVerification",
    "WeatherWarningLevel",
    "RainfallIndicator",
    "KERALA_DISTRICTS",
    "WeatherAdapterConfig",
    "WeatherFieldMapping",
    "WeatherObservation",
    "WeatherForecast",
    "RainfallRecord",
    "WeatherWarning",
    "WeatherAuditEvent",
    "RainfallSummary",
    "WeatherSourcePolicy",
    "WeatherHTTPClient",
    "WeatherResponseParser",
    "RainfallAnalytics",
    "WeatherAuditLogger",
    "WeatherAdapter",
    "WeatherRAGAdapter",
    "WeatherNotificationReviewAdapter",
    "create_weather_adapter",
]
