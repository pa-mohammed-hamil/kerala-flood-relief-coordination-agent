"""Placeholder module: integrations/imd_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
India Meteorological Department (IMD) Adapter

File: backend/integrations/imd_adapter.py

Python 3.11+

Dependencies:
    pip install httpx

Features:
    - Configurable authorized IMD API integration
    - Weather observations
    - Rainfall observations
    - District weather forecasts
    - Official warning ingestion
    - Kerala district filtering
    - English/Malayalam text preservation
    - API key and bearer authentication
    - HTTPS transport
    - Retry and timeout handling
    - Configurable JSON field mappings
    - UTC timestamp normalization
    - Rainfall unit conversion
    - Freshness and validity checks
    - Warning severity normalization
    - Duplicate removal
    - Batch district retrieval
    - RAG ingestion adapter
    - ML feature adapter
    - Structured audit records
    - Async API
    - JSON export
    - Human-review requirements

IMPORTANT:

This module does not assume that IMD exposes
a universal public JSON API with these routes.

The endpoint paths and response fields below
are configurable examples, not verified
official IMD endpoint specifications.

Use documented, authorized IMD data sources.

Never generate an official IMD warning from
rainfall measurements or model predictions.

Official warning status, severity, source,
issue time and validity must come from an
authenticated authoritative source.

Emergency decisions require human review.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import unicodedata
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

class IMDAdapterError(Exception):
    pass


class IMDConfigurationError(IMDAdapterError):
    pass


class IMDAuthenticationError(IMDAdapterError):
    pass


class IMDRequestError(IMDAdapterError):
    pass


class IMDResponseError(IMDAdapterError):
    pass


class IMDValidationError(IMDAdapterError):
    pass


class IMDDataUnavailableError(IMDAdapterError):
    pass


class IMDSourceVerificationError(IMDAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class IMDAuthenticationType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"


class IMDWarningSeverity(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    UNKNOWN = "unknown"


class IMDWarningType(str, Enum):
    HEAVY_RAIN = "heavy_rain"
    EXTREME_RAIN = "extreme_rain"
    THUNDERSTORM = "thunderstorm"
    LIGHTNING = "lightning"
    STRONG_WIND = "strong_wind"
    CYCLONE = "cyclone"
    FLOOD_RELATED = "flood_related"
    OTHER = "other"


class IMDDataStatus(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    FUTURE_TIMESTAMP = "future_timestamp"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    UNAVAILABLE = "unavailable"


class IMDSourceVerification(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class IMDRainfallUnit(str, Enum):
    MILLIMETERS = "mm"
    CENTIMETERS = "cm"


# ============================================================
# KERALA DISTRICTS
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
    district.casefold(): district
    for district in KERALA_DISTRICTS
}


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise IMDValidationError(
            "Expected text"
        )

    return unicodedata.normalize(
        "NFC",
        value,
    ).strip()


def normalize_district(
    district: str,
) -> str:
    value = normalize_text(
        district
    ).casefold()

    result = DISTRICT_LOOKUP.get(
        value
    )

    if result is None:
        raise IMDValidationError(
            "Unsupported Kerala district"
        )

    return result


def finite_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise IMDValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)

    except (TypeError, ValueError) as exc:
        raise IMDValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise IMDValidationError(
            f"{name} must be finite"
        )

    return result


def parse_timestamp(
    value: Any,
) -> datetime:
    if isinstance(value, datetime):
        result = value

    elif isinstance(value, str):
        text = value.strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            result = datetime.fromisoformat(
                text
            )

        except ValueError as exc:
            raise IMDValidationError(
                "Invalid ISO timestamp"
            ) from exc

    else:
        raise IMDValidationError(
            "Unsupported timestamp type"
        )

    if (
        result.tzinfo is None
        or result.utcoffset() is None
    ):
        raise IMDValidationError(
            "Timestamp must include timezone"
        )

    return result.astimezone(
        timezone.utc
    )


def optional_timestamp(
    value: Any,
) -> datetime | None:
    if value is None or value == "":
        return None

    return parse_timestamp(
        value
    )


def nested_get(
    data: Mapping[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(
            current,
            Mapping,
        ):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def optional_number(
    value: Any,
) -> float | None:
    if value is None or value == "":
        return None

    return finite_number(
        value,
        "numeric field",
    )


def rainfall_to_mm(
    value: float,
    unit: IMDRainfallUnit,
) -> float:
    if unit == IMDRainfallUnit.MILLIMETERS:
        return value

    if unit == IMDRainfallUnit.CENTIMETERS:
        return value * 10.0

    raise IMDValidationError(
        "Unsupported rainfall unit"
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return json_safe(
            asdict(value)
        )

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
class IMDAdapterConfig:
    base_url: str

    observations_endpoint: str = (
        "/weather/observations"
    )

    rainfall_endpoint: str = (
        "/weather/rainfall"
    )

    forecast_endpoint: str = (
        "/weather/forecast"
    )

    warnings_endpoint: str = (
        "/weather/warnings"
    )

    authentication_type: IMDAuthenticationType = (
        IMDAuthenticationType.NONE
    )

    api_key: str | None = None
    api_key_header: str = "X-API-Key"

    bearer_token: str | None = None

    timeout_seconds: float = 20.0

    max_retries: int = 3

    retry_base_seconds: float = 0.5

    max_response_bytes: int = 5_000_000

    max_records: int = 10_000

    max_batch_districts: int = 14

    observation_stale_minutes: int = 180

    forecast_max_age_hours: int = 24

    require_verified_source: bool = True

    source_verified: bool = False

    allowed_hosts: tuple[str, ...] = ()

    allow_local_http_testing: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(
            self.base_url
        )

        if not parsed.hostname:
            raise IMDConfigurationError(
                "Valid base_url required"
            )

        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise IMDConfigurationError(
                "Credentials, query and fragment "
                "are not allowed in base_url"
            )

        if parsed.scheme != "https":
            local_allowed = (
                self.allow_local_http_testing
                and parsed.scheme == "http"
                and parsed.hostname in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
            )

            if not local_allowed:
                raise IMDConfigurationError(
                    "HTTPS is required"
                )

        if (
            self.allowed_hosts
            and parsed.hostname
            not in self.allowed_hosts
        ):
            raise IMDConfigurationError(
                "API host is not allowlisted"
            )

        for endpoint in (
            self.observations_endpoint,
            self.rainfall_endpoint,
            self.forecast_endpoint,
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
                raise IMDConfigurationError(
                    "Invalid API endpoint path"
                )

        if (
            self.authentication_type
            == IMDAuthenticationType.API_KEY
            and not self.api_key
        ):
            raise IMDConfigurationError(
                "API key required"
            )

        if (
            self.authentication_type
            == IMDAuthenticationType.BEARER
            and not self.bearer_token
        ):
            raise IMDConfigurationError(
                "Bearer token required"
            )

        if self.timeout_seconds <= 0:
            raise IMDConfigurationError(
                "Invalid timeout"
            )

        if self.max_retries < 0:
            raise IMDConfigurationError(
                "Invalid retry count"
            )

        for name in (
            "max_response_bytes",
            "max_records",
            "max_batch_districts",
            "observation_stale_minutes",
            "forecast_max_age_hours",
        ):
            if getattr(self, name) <= 0:
                raise IMDConfigurationError(
                    f"{name} must be positive"
                )


# ============================================================
# FIELD MAPPING
# ============================================================

@dataclass(frozen=True)
class IMDFieldMapping:
    observations_list_path: str = "data"
    rainfall_list_path: str = "data"
    forecasts_list_path: str = "data"
    warnings_list_path: str = "data"

    station_id: str = "station_id"
    station_name: str = "station_name"

    district: str = "district"

    observed_at: str = "observed_at"

    temperature_c: str = "temperature_c"
    humidity_percent: str = "humidity_percent"
    wind_speed_kmh: str = "wind_speed_kmh"

    rainfall_mm: str = "rainfall_mm"
    rainfall_period_hours: str = "period_hours"

    forecast_issued_at: str = "issued_at"
    forecast_valid_from: str = "valid_from"
    forecast_valid_until: str = "valid_until"

    forecast_rainfall_mm: str = "rainfall_mm"
    forecast_description: str = "description"

    warning_id: str = "warning_id"
    warning_type: str = "warning_type"
    warning_severity: str = "severity"
    warning_title: str = "title"
    warning_message: str = "message"

    warning_issued_at: str = "issued_at"
    warning_valid_from: str = "valid_from"
    warning_valid_until: str = "valid_until"

    rainfall_unit: IMDRainfallUnit = (
        IMDRainfallUnit.MILLIMETERS
    )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class IMDWeatherObservation:
    station_id: str
    station_name: str | None

    district: str

    observed_at: datetime

    temperature_c: float | None
    humidity_percent: float | None
    wind_speed_kmh: float | None

    data_status: IMDDataStatus

    source_uri: str

    verification: IMDSourceVerification

    retrieved_at: datetime

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class IMDRainfallObservation:
    station_id: str
    station_name: str | None

    district: str

    rainfall_mm: float

    period_hours: float

    observed_at: datetime

    data_status: IMDDataStatus

    source_uri: str

    verification: IMDSourceVerification

    retrieved_at: datetime

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class IMDDistrictForecast:
    district: str

    issued_at: datetime

    valid_from: datetime
    valid_until: datetime

    forecast_rainfall_mm: float | None

    description: str

    data_status: IMDDataStatus

    source_uri: str

    verification: IMDSourceVerification

    retrieved_at: datetime

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class IMDOfficialWarning:
    warning_id: str

    district: str

    warning_type: IMDWarningType

    severity: IMDWarningSeverity

    title: str
    message: str

    issued_at: datetime

    valid_from: datetime
    valid_until: datetime

    data_status: IMDDataStatus

    source_uri: str

    verification: IMDSourceVerification

    retrieved_at: datetime

    requires_human_review: bool = True

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class IMDKeralaWeatherSnapshot:
    observations: tuple[
        IMDWeatherObservation,
        ...
    ]

    rainfall: tuple[
        IMDRainfallObservation,
        ...
    ]

    forecasts: tuple[
        IMDDistrictForecast,
        ...
    ]

    warnings: tuple[
        IMDOfficialWarning,
        ...
    ]

    retrieved_at: datetime

    source_verified: bool

    warnings_text: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# DATA FRESHNESS
# ============================================================

class IMDDataFreshnessChecker:
    def __init__(
        self,
        config: IMDAdapterConfig,
    ) -> None:
        self.config = config

    def observation_status(
        self,
        observed_at: datetime,
        *,
        now: datetime,
    ) -> IMDDataStatus:
        age = now - observed_at

        if age < timedelta(0):
            return IMDDataStatus.FUTURE_TIMESTAMP

        if age > timedelta(
            minutes=self.config.observation_stale_minutes
        ):
            return IMDDataStatus.STALE

        return IMDDataStatus.CURRENT

    def forecast_status(
        self,
        issued_at: datetime,
        valid_from: datetime,
        valid_until: datetime,
        *,
        now: datetime,
    ) -> IMDDataStatus:
        if issued_at > now:
            return IMDDataStatus.FUTURE_TIMESTAMP

        if now < valid_from:
            return IMDDataStatus.NOT_YET_VALID

        if now > valid_until:
            return IMDDataStatus.EXPIRED

        if now - issued_at > timedelta(
            hours=self.config.forecast_max_age_hours
        ):
            return IMDDataStatus.STALE

        return IMDDataStatus.CURRENT

    @staticmethod
    def warning_status(
        issued_at: datetime,
        valid_from: datetime,
        valid_until: datetime,
        *,
        now: datetime,
    ) -> IMDDataStatus:
        if issued_at > now:
            return IMDDataStatus.FUTURE_TIMESTAMP

        if now < valid_from:
            return IMDDataStatus.NOT_YET_VALID

        if now > valid_until:
            return IMDDataStatus.EXPIRED

        return IMDDataStatus.CURRENT


# ============================================================
# HTTP CLIENT
# ============================================================

class IMDHTTPClient:
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        config: IMDAdapterConfig,
    ) -> None:
        if httpx is None:
            raise IMDConfigurationError(
                "Install httpx: pip install httpx"
            )

        self.config = config

        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": (
                "KeralaFloodReliefAgent/1.0"
            ),
        }

        if (
            self.config.authentication_type
            == IMDAuthenticationType.API_KEY
        ):
            headers[
                self.config.api_key_header
            ] = self.config.api_key or ""

        elif (
            self.config.authentication_type
            == IMDAuthenticationType.BEARER
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
    ) -> Mapping[str, Any] | list[Any]:
        approved_endpoints = {
            self.config.observations_endpoint,
            self.config.rainfall_endpoint,
            self.config.forecast_endpoint,
            self.config.warnings_endpoint,
        }

        if endpoint not in approved_endpoints:
            raise IMDConfigurationError(
                "Unapproved API endpoint"
            )

        url = (
            self.config.base_url.rstrip("/")
            + endpoint
        )

        client = await self._get_client()

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
                    if response.status_code in (
                        401,
                        403,
                    ):
                        raise IMDAuthenticationError(
                            "IMD API authentication failed"
                        )

                    if response.status_code in (
                        self.RETRYABLE_STATUS_CODES
                    ):
                        raise IMDRequestError(
                            "Temporary upstream failure"
                        )

                    response.raise_for_status()

                    content_length = (
                        response.headers.get(
                            "Content-Length"
                        )
                    )

                    if content_length is not None:
                        try:
                            declared_size = int(
                                content_length
                            )
                        except ValueError as exc:
                            raise IMDResponseError(
                                "Invalid Content-Length"
                            ) from exc

                        if declared_size > (
                            self.config.max_response_bytes
                        ):
                            raise IMDResponseError(
                                "Response size limit exceeded"
                            )

                    chunks = []
                    total_size = 0

                    async for chunk in response.aiter_bytes():
                        total_size += len(chunk)

                        if total_size > (
                            self.config.max_response_bytes
                        ):
                            raise IMDResponseError(
                                "Response size limit exceeded"
                            )

                        chunks.append(
                            chunk
                        )

                payload = json.loads(
                    b"".join(chunks)
                )

                if not isinstance(
                    payload,
                    (dict, list),
                ):
                    raise IMDResponseError(
                        "Unexpected JSON response root"
                    )

                return payload

            except (
                IMDAuthenticationError,
                IMDResponseError,
            ):
                raise

            except (
                httpx.TimeoutException,
                httpx.TransportError,
                IMDRequestError,
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

                await asyncio.sleep(
                    delay
                )

            except httpx.HTTPStatusError as exc:
                raise IMDRequestError(
                    f"Upstream HTTP status: "
                    f"{exc.response.status_code}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise IMDResponseError(
                    "Invalid JSON response"
                ) from exc

        raise IMDRequestError(
            "IMD request failed after retries"
        ) from last_error

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(
        self,
    ) -> "IMDHTTPClient":
        await self._get_client()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.close()


# ============================================================
# RESPONSE PARSER
# ============================================================

class IMDResponseParser:
    def __init__(
        self,
        config: IMDAdapterConfig,
        mapping: IMDFieldMapping,
    ) -> None:
        self.config = config
        self.mapping = mapping

        self.freshness = (
            IMDDataFreshnessChecker(config)
        )

    @property
    def verification(
        self,
    ) -> IMDSourceVerification:
        return (
            IMDSourceVerification.VERIFIED
            if self.config.source_verified
            else IMDSourceVerification.UNVERIFIED
        )

    def _records(
        self,
        payload: Mapping[str, Any] | list[Any],
        path: str,
    ) -> list[Mapping[str, Any]]:
        if isinstance(payload, list):
            records = payload

        else:
            records = nested_get(
                payload,
                path,
            )

        if not isinstance(records, list):
            raise IMDResponseError(
                "Expected list at configured JSON path"
            )

        if len(records) > self.config.max_records:
            raise IMDResponseError(
                "Response record limit exceeded"
            )

        if not all(
            isinstance(record, Mapping)
            for record in records
        ):
            raise IMDResponseError(
                "Invalid record structure"
            )

        return records

    def parse_observations(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[IMDWeatherObservation]:
        now = now or utc_now()

        records = self._records(
            payload,
            self.mapping.observations_list_path,
        )

        output = []

        for item in records:
            station_id = nested_get(
                item,
                self.mapping.station_id,
            )

            district = nested_get(
                item,
                self.mapping.district,
            )

            observed_at = nested_get(
                item,
                self.mapping.observed_at,
            )

            if (
                station_id is None
                or district is None
                or observed_at is None
            ):
                raise IMDResponseError(
                    "Required observation field missing"
                )

            observed_at = parse_timestamp(
                observed_at
            )

            humidity = optional_number(
                nested_get(
                    item,
                    self.mapping.humidity_percent,
                )
            )

            if humidity is not None and not (
                0 <= humidity <= 100
            ):
                raise IMDValidationError(
                    "Humidity must be 0..100"
                )

            wind_speed = optional_number(
                nested_get(
                    item,
                    self.mapping.wind_speed_kmh,
                )
            )

            if (
                wind_speed is not None
                and wind_speed < 0
            ):
                raise IMDValidationError(
                    "Wind speed cannot be negative"
                )

            output.append(
                IMDWeatherObservation(
                    station_id=str(station_id),
                    station_name=nested_get(
                        item,
                        self.mapping.station_name,
                    ),
                    district=normalize_district(
                        str(district)
                    ),
                    observed_at=observed_at,
                    temperature_c=optional_number(
                        nested_get(
                            item,
                            self.mapping.temperature_c,
                        )
                    ),
                    humidity_percent=humidity,
                    wind_speed_kmh=wind_speed,
                    data_status=(
                        self.freshness.observation_status(
                            observed_at,
                            now=now,
                        )
                    ),
                    source_uri=self.config.base_url,
                    verification=self.verification,
                    retrieved_at=now,
                )
            )

        return output

    def parse_rainfall(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[IMDRainfallObservation]:
        now = now or utc_now()

        records = self._records(
            payload,
            self.mapping.rainfall_list_path,
        )

        output = []

        for item in records:
            station_id = nested_get(
                item,
                self.mapping.station_id,
            )

            district = nested_get(
                item,
                self.mapping.district,
            )

            rainfall = nested_get(
                item,
                self.mapping.rainfall_mm,
            )

            period_hours = nested_get(
                item,
                self.mapping.rainfall_period_hours,
            )

            observed_at = nested_get(
                item,
                self.mapping.observed_at,
            )

            if any(
                value is None
                for value in (
                    station_id,
                    district,
                    rainfall,
                    period_hours,
                    observed_at,
                )
            ):
                raise IMDResponseError(
                    "Required rainfall field missing"
                )

            rainfall_mm = rainfall_to_mm(
                finite_number(
                    rainfall,
                    "rainfall",
                ),
                self.mapping.rainfall_unit,
            )

            duration = finite_number(
                period_hours,
                "period_hours",
            )

            if rainfall_mm < 0 or duration <= 0:
                raise IMDValidationError(
                    "Invalid rainfall measurement"
                )

            timestamp = parse_timestamp(
                observed_at
            )

            output.append(
                IMDRainfallObservation(
                    station_id=str(station_id),
                    station_name=nested_get(
                        item,
                        self.mapping.station_name,
                    ),
                    district=normalize_district(
                        str(district)
                    ),
                    rainfall_mm=rainfall_mm,
                    period_hours=duration,
                    observed_at=timestamp,
                    data_status=(
                        self.freshness.observation_status(
                            timestamp,
                            now=now,
                        )
                    ),
                    source_uri=self.config.base_url,
                    verification=self.verification,
                    retrieved_at=now,
                )
            )

        return output

    def parse_forecasts(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[IMDDistrictForecast]:
        now = now or utc_now()

        records = self._records(
            payload,
            self.mapping.forecasts_list_path,
        )

        output = []

        for item in records:
            district = normalize_district(
                str(
                    nested_get(
                        item,
                        self.mapping.district,
                        "",
                    )
                )
            )

            issued_at = parse_timestamp(
                nested_get(
                    item,
                    self.mapping.forecast_issued_at,
                )
            )

            valid_from = parse_timestamp(
                nested_get(
                    item,
                    self.mapping.forecast_valid_from,
                )
            )

            valid_until = parse_timestamp(
                nested_get(
                    item,
                    self.mapping.forecast_valid_until,
                )
            )

            if valid_until <= valid_from:
                raise IMDValidationError(
                    "Invalid forecast validity period"
                )

            raw_rainfall = optional_number(
                nested_get(
                    item,
                    self.mapping.forecast_rainfall_mm,
                )
            )

            rainfall_mm = (
                rainfall_to_mm(
                    raw_rainfall,
                    self.mapping.rainfall_unit,
                )
                if raw_rainfall is not None
                else None
            )

            if (
                rainfall_mm is not None
                and rainfall_mm < 0
            ):
                raise IMDValidationError(
                    "Forecast rainfall cannot be negative"
                )

            output.append(
                IMDDistrictForecast(
                    district=district,
                    issued_at=issued_at,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    forecast_rainfall_mm=rainfall_mm,
                    description=str(
                        nested_get(
                            item,
                            self.mapping.forecast_description,
                            "",
                        )
                    ),
                    data_status=(
                        self.freshness.forecast_status(
                            issued_at,
                            valid_from,
                            valid_until,
                            now=now,
                        )
                    ),
                    source_uri=self.config.base_url,
                    verification=self.verification,
                    retrieved_at=now,
                )
            )

        return output

    def parse_warnings(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[IMDOfficialWarning]:
        now = now or utc_now()

        records = self._records(
            payload,
            self.mapping.warnings_list_path,
        )

        output = []

        for item in records:
            warning_id = nested_get(
                item,
                self.mapping.warning_id,
            )

            district = nested_get(
                item,
                self.mapping.district,
            )

            if not warning_id or not district:
                raise IMDResponseError(
                    "Warning ID and district required"
                )

            issued_at = parse_timestamp(
                nested_get(
                    item,
                    self.mapping.warning_issued_at,
                )
            )

            valid_from = parse_timestamp(
                nested_get(
                    item,
                    self.mapping.warning_valid_from,
                )
            )

            valid_until = parse_timestamp(
                nested_get(
                    item,
                    self.mapping.warning_valid_until,
                )
            )

            if valid_until <= valid_from:
                raise IMDValidationError(
                    "Invalid warning validity period"
                )

            severity_raw = str(
                nested_get(
                    item,
                    self.mapping.warning_severity,
                    "unknown",
                )
            ).casefold()

            try:
                severity = IMDWarningSeverity(
                    severity_raw
                )
            except ValueError:
                severity = (
                    IMDWarningSeverity.UNKNOWN
                )

            warning_type_raw = str(
                nested_get(
                    item,
                    self.mapping.warning_type,
                    "other",
                )
            ).casefold()

            try:
                warning_type = IMDWarningType(
                    warning_type_raw
                )
            except ValueError:
                warning_type = IMDWarningType.OTHER

            output.append(
                IMDOfficialWarning(
                    warning_id=str(warning_id),
                    district=normalize_district(
                        str(district)
                    ),
                    warning_type=warning_type,
                    severity=severity,
                    title=normalize_text(
                        str(
                            nested_get(
                                item,
                                self.mapping.warning_title,
                                "",
                            )
                        )
                    ),
                    message=normalize_text(
                        str(
                            nested_get(
                                item,
                                self.mapping.warning_message,
                                "",
                            )
                        )
                    ),
                    issued_at=issued_at,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    data_status=(
                        self.freshness.warning_status(
                            issued_at,
                            valid_from,
                            valid_until,
                            now=now,
                        )
                    ),
                    source_uri=self.config.base_url,
                    verification=self.verification,
                    retrieved_at=now,
                    requires_human_review=True,
                )
            )

        return output


# ============================================================
# DEDUPLICATION
# ============================================================

class IMDRecordDeduplicator:
    @staticmethod
    def observations(
        records: Sequence[IMDWeatherObservation],
    ) -> list[IMDWeatherObservation]:
        unique = {}

        for record in records:
            key = (
                record.station_id,
                record.observed_at,
            )

            unique[key] = record

        return sorted(
            unique.values(),
            key=lambda item: (
                item.station_id,
                item.observed_at,
            ),
        )

    @staticmethod
    def rainfall(
        records: Sequence[IMDRainfallObservation],
    ) -> list[IMDRainfallObservation]:
        unique = {}

        for record in records:
            key = (
                record.station_id,
                record.observed_at,
                record.period_hours,
            )

            unique[key] = record

        return sorted(
            unique.values(),
            key=lambda item: (
                item.station_id,
                item.observed_at,
            ),
        )

    @staticmethod
    def warnings(
        records: Sequence[IMDOfficialWarning],
    ) -> list[IMDOfficialWarning]:
        unique = {}

        for record in records:
            key = (
                record.warning_id,
                record.district,
                record.issued_at,
            )

            unique[key] = record

        return sorted(
            unique.values(),
            key=lambda item: (
                item.district,
                item.issued_at,
            ),
        )


# ============================================================
# MAIN IMD ADAPTER
# ============================================================

class IMDAdapter:
    def __init__(
        self,
        config: IMDAdapterConfig,
        *,
        mapping: IMDFieldMapping | None = None,
        http_client: IMDHTTPClient | None = None,
    ) -> None:
        self.config = config

        self.mapping = (
            mapping or IMDFieldMapping()
        )

        self.http_client = (
            http_client
            or IMDHTTPClient(config)
        )

        self.parser = IMDResponseParser(
            config,
            self.mapping,
        )

        self.deduplicator = (
            IMDRecordDeduplicator()
        )

    def _verify_source_policy(self) -> None:
        if (
            self.config.require_verified_source
            and not self.config.source_verified
        ):
            raise IMDSourceVerificationError(
                "Source must be independently "
                "verified before operational use"
            )

    async def get_observations(
        self,
        *,
        district: str | None = None,
    ) -> list[IMDWeatherObservation]:
        self._verify_source_policy()

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
            selected = normalize_district(
                district
            )

            records = [
                record
                for record in records
                if record.district == selected
            ]

        return self.deduplicator.observations(
            records
        )

    async def get_rainfall(
        self,
        *,
        district: str | None = None,
    ) -> list[IMDRainfallObservation]:
        self._verify_source_policy()

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
            selected = normalize_district(
                district
            )

            records = [
                record
                for record in records
                if record.district == selected
            ]

        return self.deduplicator.rainfall(
            records
        )

    async def get_forecasts(
        self,
        *,
        district: str | None = None,
    ) -> list[IMDDistrictForecast]:
        self._verify_source_policy()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.forecast_endpoint,
            params=params,
        )

        records = self.parser.parse_forecasts(
            payload
        )

        if district:
            selected = normalize_district(
                district
            )

            records = [
                record
                for record in records
                if record.district == selected
            ]

        return records

    async def get_official_warnings(
        self,
        *,
        district: str | None = None,
        active_only: bool = True,
    ) -> list[IMDOfficialWarning]:
        self._verify_source_policy()

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
            selected = normalize_district(
                district
            )

            records = [
                record
                for record in records
                if record.district == selected
            ]

        if active_only:
            records = [
                record
                for record in records
                if record.data_status
                == IMDDataStatus.CURRENT
            ]

        return self.deduplicator.warnings(
            records
        )

    async def get_district_snapshot(
        self,
        district: str,
    ) -> IMDKeralaWeatherSnapshot:
        district = normalize_district(
            district
        )

        (
            observations,
            rainfall,
            forecasts,
            warnings,
        ) = await asyncio.gather(
            self.get_observations(
                district=district
            ),
            self.get_rainfall(
                district=district
            ),
            self.get_forecasts(
                district=district
            ),
            self.get_official_warnings(
                district=district
            ),
        )

        warnings_text = []

        if any(
            item.data_status != IMDDataStatus.CURRENT
            for item in observations
        ):
            warnings_text.append(
                "Some weather observations are stale "
                "or have invalid timestamps"
            )

        if any(
            item.data_status != IMDDataStatus.CURRENT
            for item in rainfall
        ):
            warnings_text.append(
                "Some rainfall observations are stale"
            )

        if not self.config.source_verified:
            warnings_text.append(
                "Source has not been independently verified"
            )

        return IMDKeralaWeatherSnapshot(
            observations=tuple(observations),
            rainfall=tuple(rainfall),
            forecasts=tuple(forecasts),
            warnings=tuple(warnings),
            retrieved_at=utc_now(),
            source_verified=self.config.source_verified,
            warnings_text=tuple(warnings_text),
        )

    async def get_kerala_snapshot(
        self,
        *,
        districts: Sequence[str] | None = None,
    ) -> list[IMDKeralaWeatherSnapshot]:
        selected = (
            tuple(districts)
            if districts is not None
            else KERALA_DISTRICTS
        )

        if len(selected) > (
            self.config.max_batch_districts
        ):
            raise IMDValidationError(
                "District batch limit exceeded"
            )

        normalized = tuple(
            dict.fromkeys(
                normalize_district(district)
                for district in selected
            )
        )

        return list(
            await asyncio.gather(
                *(
                    self.get_district_snapshot(
                        district
                    )
                    for district in normalized
                )
            )
        )

    async def close(self) -> None:
        await self.http_client.close()

    async def __aenter__(
        self,
    ) -> "IMDAdapter":
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.close()


# ============================================================
# RAG ADAPTER
# ============================================================

class IMDRAGAdapter:
    """
    Converts verified IMD weather records
    into RAG ingestion documents.

    Only current, verified records are
    included by default.
    """

    @staticmethod
    def to_rag_records(
        snapshot: IMDKeralaWeatherSnapshot,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        if not tenant_id.strip():
            raise IMDValidationError(
                "tenant_id required"
            )

        if not snapshot.source_verified:
            return []

        records = []

        for observation in snapshot.rainfall:
            if (
                observation.verification
                != IMDSourceVerification.VERIFIED
                or observation.data_status
                != IMDDataStatus.CURRENT
            ):
                continue

            records.append({
                "id": (
                    f"imd:rainfall:"
                    f"{observation.station_id}:"
                    f"{observation.observed_at.isoformat()}"
                ),
                "content": (
                    f"IMD rainfall observation. "
                    f"District: {observation.district}. "
                    f"Station: {observation.station_id}. "
                    f"Rainfall: {observation.rainfall_mm} mm "
                    f"over {observation.period_hours} hours. "
                    f"Observed at: "
                    f"{observation.observed_at.isoformat()}."
                ),
                "metadata": {
                    "tenant_id": tenant_id,
                    "source": "imd",
                    "source_uri": observation.source_uri,
                    "district": observation.district,
                    "station_id": observation.station_id,
                    "observed_at": (
                        observation.observed_at.isoformat()
                    ),
                    "verification": "verified",
                    "requires_human_review": True,
                },
            })

        for warning in snapshot.warnings:
            if (
                warning.verification
                != IMDSourceVerification.VERIFIED
                or warning.data_status
                != IMDDataStatus.CURRENT
            ):
                continue

            records.append({
                "id": (
                    f"imd:warning:"
                    f"{warning.warning_id}:"
                    f"{warning.issued_at.isoformat()}"
                ),
                "content": (
                    f"Official IMD warning record. "
                    f"District: {warning.district}. "
                    f"Severity: {warning.severity.value}. "
                    f"Title: {warning.title}. "
                    f"Message: {warning.message}. "
                    f"Valid until: "
                    f"{warning.valid_until.isoformat()}."
                ),
                "metadata": {
                    "tenant_id": tenant_id,
                    "source": "imd",
                    "source_uri": warning.source_uri,
                    "district": warning.district,
                    "warning_id": warning.warning_id,
                    "severity": warning.severity.value,
                    "issued_at": (
                        warning.issued_at.isoformat()
                    ),
                    "valid_until": (
                        warning.valid_until.isoformat()
                    ),
                    "verification": "verified",
                    "requires_human_review": True,
                },
            })

        return records


# ============================================================
# ML FEATURE ADAPTER
# ============================================================

class IMDMLFeatureAdapter:
    """
    Creates model input features.

    These are measurements, not forecasts
    or independently verified flood risks.
    """

    @staticmethod
    def rainfall_features(
        records: Sequence[IMDRainfallObservation],
    ) -> list[dict[str, Any]]:
        features = []

        for record in records:
            if (
                record.verification
                != IMDSourceVerification.VERIFIED
                or record.data_status
                != IMDDataStatus.CURRENT
            ):
                continue

            features.append({
                "station_id": record.station_id,
                "district": record.district,
                "rainfall_mm": record.rainfall_mm,
                "period_hours": record.period_hours,
                "average_rainfall_rate_mm_per_hour": (
                    record.rainfall_mm
                    / record.period_hours
                ),
                "observed_at": (
                    record.observed_at.isoformat()
                ),
                "source": "imd",
            })

        return features

    @staticmethod
    def weather_features(
        records: Sequence[IMDWeatherObservation],
    ) -> list[dict[str, Any]]:
        return [
            {
                "station_id": record.station_id,
                "district": record.district,
                "temperature_c": record.temperature_c,
                "humidity_percent": record.humidity_percent,
                "wind_speed_kmh": record.wind_speed_kmh,
                "observed_at": (
                    record.observed_at.isoformat()
                ),
            }
            for record in records
            if (
                record.verification
                == IMDSourceVerification.VERIFIED
                and record.data_status
                == IMDDataStatus.CURRENT
            )
        ]


# ============================================================
# EXPORTER
# ============================================================

class IMDResponseExporter:
    @staticmethod
    def to_json(
        snapshot: IMDKeralaWeatherSnapshot,
    ) -> str:
        return json.dumps(
            snapshot.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


# ============================================================
# FACTORY
# ============================================================

def create_imd_adapter(
    config: IMDAdapterConfig,
    *,
    mapping: IMDFieldMapping | None = None,
) -> IMDAdapter:
    return IMDAdapter(
        config=config,
        mapping=mapping,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def demo() -> None:
    """
    Synthetic parser demonstration only.

    Does not contact IMD.
    Does not represent real rainfall,
    weather forecasts or warnings.
    """

    config = IMDAdapterConfig(
        base_url="https://example.invalid",
        require_verified_source=False,
        source_verified=False,
    )

    parser = IMDResponseParser(
        config,
        IMDFieldMapping(),
    )

    payload = {
        "data": [
            {
                "station_id": "DEMO-001",
                "station_name": "Demo Station",
                "district": "Thrissur",
                "rainfall_mm": 25.0,
                "period_hours": 24,
                "observed_at": (
                    utc_now().isoformat()
                ),
            }
        ]
    }

    rainfall = parser.parse_rainfall(
        payload
    )

    print(
        json.dumps(
            json_safe(rainfall),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    demo()


__all__ = [
    "IMDAdapterError",
    "IMDConfigurationError",
    "IMDAuthenticationError",
    "IMDRequestError",
    "IMDResponseError",
    "IMDValidationError",
    "IMDDataUnavailableError",
    "IMDSourceVerificationError",
    "IMDAuthenticationType",
    "IMDWarningSeverity",
    "IMDWarningType",
    "IMDDataStatus",
    "IMDSourceVerification",
    "IMDRainfallUnit",
    "KERALA_DISTRICTS",
    "IMDAdapterConfig",
    "IMDFieldMapping",
    "IMDWeatherObservation",
    "IMDRainfallObservation",
    "IMDDistrictForecast",
    "IMDOfficialWarning",
    "IMDKeralaWeatherSnapshot",
    "IMDDataFreshnessChecker",
    "IMDHTTPClient",
    "IMDResponseParser",
    "IMDRecordDeduplicator",
    "IMDAdapter",
    "IMDRAGAdapter",
    "IMDMLFeatureAdapter",
    "IMDResponseExporter",
    "create_imd_adapter",
]
