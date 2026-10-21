"""Placeholder module: integrations/cwc_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Central Water Commission (CWC) Adapter

File: backend/integrations/cwc_adapter.py

Python 3.11+
Dependencies:
    pip install httpx

Purpose:
    Integrate officially authorized Central Water
    Commission river-level and flood-monitoring data
    into the Kerala Flood Relief Coordination Agent.

Features:
    - Configurable official API endpoint
    - API-key or bearer-token authentication
    - Asynchronous HTTP requests
    - Retry with exponential backoff
    - Request timeout handling
    - River station metadata
    - Water-level observations
    - Historical observations
    - Configurable JSON field mapping
    - UTC timestamp normalization
    - Unit conversion
    - Flood threshold comparison
    - Stale-data detection
    - Duplicate observation removal
    - Station filtering
    - Batch station retrieval
    - Structured integration results
    - Audit-safe logging
    - FastAPI-compatible service interface
    - JSON export

IMPORTANT:
    This adapter does not assume that CWC provides
    a particular public JSON API or response schema.

    Configure an authorized endpoint and field mapping
    using the actual CWC API documentation or agreement.

    CWC flood-monitoring data may also be distributed
    through official dashboards, bulletins or datasets
    rather than an unrestricted JSON API.

    No live readings or official warnings are fabricated.

    Threshold classifications are local comparisons
    against configured, verified station thresholds.
    They are not official CWC flood forecasts, warnings,
    evacuation orders or operational instructions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import threading
import unicodedata

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

class CWCAdapterError(Exception):
    pass


class CWCConfigurationError(CWCAdapterError):
    pass


class CWCAuthenticationError(CWCAdapterError):
    pass


class CWCRequestError(CWCAdapterError):
    pass


class CWCResponseError(CWCAdapterError):
    pass


class CWCValidationError(CWCAdapterError):
    pass


class CWCDataUnavailableError(CWCAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CWCAuthenticationType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"


class CWCWaterLevelUnit(str, Enum):
    METERS = "m"
    CENTIMETERS = "cm"
    MILLIMETERS = "mm"


class CWCThresholdStatus(str, Enum):
    BELOW_WARNING = "below_warning"
    WARNING_THRESHOLD = "warning_threshold"
    DANGER_THRESHOLD = "danger_threshold"
    HIGH_FLOOD_THRESHOLD = "high_flood_threshold"
    UNKNOWN = "unknown"


class CWCDataStatus(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    FUTURE_TIMESTAMP = "future_timestamp"
    UNAVAILABLE = "unavailable"


class CWCSourceVerification(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise CWCValidationError(
            "Expected text"
        )

    return unicodedata.normalize(
        "NFC",
        value,
    ).strip()


def ensure_aware_datetime(
    value: datetime,
) -> datetime:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise CWCValidationError(
            "Timestamp must include timezone"
        )

    return value.astimezone(
        timezone.utc
    )


def parse_timestamp(
    value: Any,
) -> datetime:
    if isinstance(value, datetime):
        return ensure_aware_datetime(value)

    if not isinstance(value, str):
        raise CWCValidationError(
            "Unsupported timestamp value"
        )

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CWCValidationError(
            "Invalid timestamp format"
        ) from exc

    return ensure_aware_datetime(
        parsed
    )


def finite_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise CWCValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CWCValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise CWCValidationError(
            f"{name} must be finite"
        )

    return result


def to_meters(
    value: float,
    unit: CWCWaterLevelUnit,
) -> float:
    if unit == CWCWaterLevelUnit.METERS:
        return value

    if unit == CWCWaterLevelUnit.CENTIMETERS:
        return value / 100.0

    if unit == CWCWaterLevelUnit.MILLIMETERS:
        return value / 1000.0

    raise CWCValidationError(
        "Unsupported water-level unit"
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


def nested_get(
    data: Mapping[str, Any],
    path: str,
    *,
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


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CWCAdapterConfig:
    base_url: str

    station_endpoint: str = "/stations"
    observation_endpoint: str = "/observations"
    history_endpoint: str = "/observations/history"

    authentication_type: CWCAuthenticationType = (
        CWCAuthenticationType.NONE
    )

    api_key: str | None = None

    api_key_header: str = "X-API-Key"

    bearer_token: str | None = None

    timeout_seconds: float = 20.0

    max_retries: int = 3

    retry_base_seconds: float = 0.5

    max_response_bytes: int = 5_000_000

    max_stations_per_batch: int = 100

    max_history_records: int = 10_000

    stale_after_minutes: int = 180

    source_verified: bool = False

    allow_http_for_local_testing: bool = False

    allowed_hosts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parsed = urlparse(
            self.base_url
        )

        if not parsed.hostname:
            raise CWCConfigurationError(
                "Valid base_url required"
            )

        if parsed.username or parsed.password:
            raise CWCConfigurationError(
                "Credentials must not be embedded in URL"
            )

        if parsed.query or parsed.fragment:
            raise CWCConfigurationError(
                "base_url must not contain query or fragment"
            )

        if parsed.scheme != "https":
            if not (
                self.allow_http_for_local_testing
                and parsed.scheme == "http"
                and parsed.hostname in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
            ):
                raise CWCConfigurationError(
                    "HTTPS required"
                )

        if (
            self.allowed_hosts
            and parsed.hostname not in self.allowed_hosts
        ):
            raise CWCConfigurationError(
                "Host not in configured allowlist"
            )

        for endpoint in (
            self.station_endpoint,
            self.observation_endpoint,
            self.history_endpoint,
        ):
            if (
                not endpoint.startswith("/")
                or endpoint.startswith("//")
                or "://" in endpoint
                or "?" in endpoint
                or "#" in endpoint
                or ".." in endpoint.split("/")
            ):
                raise CWCConfigurationError(
                    "Endpoints must be relative API paths"
                )

        if self.authentication_type == CWCAuthenticationType.API_KEY:
            if not self.api_key:
                raise CWCConfigurationError(
                    "API key required"
                )

        if self.authentication_type == CWCAuthenticationType.BEARER:
            if not self.bearer_token:
                raise CWCConfigurationError(
                    "Bearer token required"
                )

        if self.timeout_seconds <= 0:
            raise CWCConfigurationError(
                "timeout_seconds must be positive"
            )

        if self.max_retries < 0:
            raise CWCConfigurationError(
                "max_retries cannot be negative"
            )

        for name in (
            "max_response_bytes",
            "max_stations_per_batch",
            "max_history_records",
            "stale_after_minutes",
        ):
            if getattr(self, name) <= 0:
                raise CWCConfigurationError(
                    f"{name} must be positive"
                )


# ============================================================
# API RESPONSE FIELD MAPPING
# ============================================================

@dataclass(frozen=True)
class CWCFieldMapping:
    station_list_path: str = "data"
    observation_list_path: str = "data"

    station_id: str = "station_id"
    station_name: str = "station_name"

    river_name: str = "river_name"
    district: str = "district"
    state: str = "state"

    latitude: str = "latitude"
    longitude: str = "longitude"

    observation_station_id: str = "station_id"

    water_level: str = "water_level"
    observed_at: str = "observed_at"

    warning_level: str = "warning_level"
    danger_level: str = "danger_level"
    high_flood_level: str = "high_flood_level"

    water_level_unit: CWCWaterLevelUnit = (
        CWCWaterLevelUnit.METERS
    )

    """
    Field paths are examples only.

    Replace them with the actual paths
    documented by the authorized data source.
    """


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class CWCStation:
    station_id: str
    station_name: str

    river_name: str | None
    district: str | None
    state: str | None

    latitude: float | None
    longitude: float | None

    source_uri: str

    verification: CWCSourceVerification

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class CWCWaterLevelThresholds:
    warning_level_m: float | None = None
    danger_level_m: float | None = None
    high_flood_level_m: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "warning_level_m",
            "danger_level_m",
            "high_flood_level_m",
        ):
            value = getattr(self, name)

            if value is not None:
                finite_number(value, name)

        values = [
            value
            for value in (
                self.warning_level_m,
                self.danger_level_m,
                self.high_flood_level_m,
            )
            if value is not None
        ]

        if values != sorted(values):
            raise CWCValidationError(
                "Threshold levels must be ascending"
            )


@dataclass(frozen=True)
class CWCWaterLevelObservation:
    station_id: str

    water_level_m: float

    observed_at: datetime

    thresholds: CWCWaterLevelThresholds

    threshold_status: CWCThresholdStatus

    data_status: CWCDataStatus

    source_uri: str

    verification: CWCSourceVerification

    retrieved_at: datetime

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class CWCStationResult:
    station: CWCStation

    latest_observation: CWCWaterLevelObservation | None

    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CWCAdapterResponse:
    results: tuple[CWCStationResult, ...]

    source_verified: bool

    retrieved_at: datetime

    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# HTTP CLIENT
# ============================================================

class CWCHTTPClient:
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        config: CWCAdapterConfig,
    ) -> None:
        if httpx is None:
            raise CWCConfigurationError(
                "Install httpx: pip install httpx"
            )

        self.config = config

        self._client: httpx.AsyncClient | None = None

        self._lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": (
                "KeralaFloodReliefAgent/1.0"
            ),
        }

        if (
            self.config.authentication_type
            == CWCAuthenticationType.API_KEY
        ):
            headers[
                self.config.api_key_header
            ] = self.config.api_key or ""

        elif (
            self.config.authentication_type
            == CWCAuthenticationType.BEARER
        ):
            headers["Authorization"] = (
                f"Bearer {self.config.bearer_token}"
            )

        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self.config.base_url.rstrip("/") + "/",
                    timeout=self.config.timeout_seconds,
                    follow_redirects=False,
                    headers=self._headers(),
                    trust_env=False,
                )

            return self._client

    async def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | list[Any]:
        if endpoint not in {
            self.config.station_endpoint,
            self.config.observation_endpoint,
            self.config.history_endpoint,
        }:
            raise CWCConfigurationError(
                "Unapproved endpoint"
            )

        client = await self._get_client()

        # httpx base_url joins absolute-path inputs
        # relative to its configured base path.
        # We explicitly construct a fixed endpoint
        # beneath the trusted base URL.

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
                    if response.status_code in (
                        401,
                        403,
                    ):
                        raise CWCAuthenticationError(
                            "CWC API authentication failed"
                        )

                    if (
                        response.status_code
                        in self.RETRYABLE_STATUS_CODES
                    ):
                        raise CWCRequestError(
                            "Temporary upstream API failure"
                        )

                    response.raise_for_status()

                    content_length = response.headers.get(
                        "Content-Length"
                    )

                    if (
                        content_length is not None
                        and int(content_length)
                        > self.config.max_response_bytes
                    ):
                        raise CWCResponseError(
                            "Response exceeds configured size"
                        )

                    chunks = []
                    total_bytes = 0

                    async for chunk in response.aiter_bytes():
                        total_bytes += len(chunk)

                        if total_bytes > (
                            self.config.max_response_bytes
                        ):
                            raise CWCResponseError(
                                "Response size limit exceeded"
                            )

                        chunks.append(chunk)

                    raw = b"".join(chunks)

                payload = json.loads(
                    raw
                )

                if not isinstance(
                    payload,
                    (dict, list),
                ):
                    raise CWCResponseError(
                        "Unexpected JSON root"
                    )

                return payload

            except (
                CWCAuthenticationError,
                CWCResponseError,
            ):
                raise

            except (
                httpx.TimeoutException,
                httpx.TransportError,
                CWCRequestError,
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
                    min(delay * 0.2, 1.0),
                )

                await asyncio.sleep(
                    delay
                )

            except httpx.HTTPStatusError as exc:
                raise CWCRequestError(
                    f"Upstream returned HTTP "
                    f"{exc.response.status_code}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise CWCResponseError(
                    "Invalid JSON response"
                ) from exc

        raise CWCRequestError(
            "CWC API request failed after retries"
        ) from last_error

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def __aenter__(self) -> "CWCHTTPClient":
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

class CWCResponseParser:
    def __init__(
        self,
        mapping: CWCFieldMapping,
        *,
        source_uri: str,
        source_verified: bool,
        stale_after_minutes: int,
    ) -> None:
        self.mapping = mapping
        self.source_uri = source_uri
        self.source_verified = source_verified
        self.stale_after_minutes = stale_after_minutes

    @property
    def verification(self) -> CWCSourceVerification:
        return (
            CWCSourceVerification.VERIFIED
            if self.source_verified
            else CWCSourceVerification.UNVERIFIED
        )

    @staticmethod
    def _optional_number(
        value: Any,
    ) -> float | None:
        if value is None or value == "":
            return None

        return finite_number(
            value,
            "numeric field",
        )

    def _records(
        self,
        payload: Mapping[str, Any] | list[Any],
        path: str,
    ) -> list[Mapping[str, Any]]:
        if isinstance(payload, list):
            raw = payload

        else:
            raw = nested_get(
                payload,
                path,
            )

        if not isinstance(raw, list):
            raise CWCResponseError(
                "Expected an array at configured response path"
            )

        if not all(
            isinstance(item, Mapping)
            for item in raw
        ):
            raise CWCResponseError(
                "Response contains invalid records"
            )

        return raw

    def parse_stations(
        self,
        payload: Mapping[str, Any] | list[Any],
    ) -> list[CWCStation]:
        records = self._records(
            payload,
            self.mapping.station_list_path,
        )

        stations = []

        for item in records:
            station_id = nested_get(
                item,
                self.mapping.station_id,
            )

            station_name = nested_get(
                item,
                self.mapping.station_name,
            )

            if not station_id or not station_name:
                raise CWCResponseError(
                    "Station ID or name missing"
                )

            latitude = self._optional_number(
                nested_get(
                    item,
                    self.mapping.latitude,
                )
            )

            longitude = self._optional_number(
                nested_get(
                    item,
                    self.mapping.longitude,
                )
            )

            if latitude is not None and not (
                -90 <= latitude <= 90
            ):
                raise CWCResponseError(
                    "Invalid latitude"
                )

            if longitude is not None and not (
                -180 <= longitude <= 180
            ):
                raise CWCResponseError(
                    "Invalid longitude"
                )

            stations.append(
                CWCStation(
                    station_id=str(station_id),
                    station_name=normalize_text(
                        str(station_name)
                    ),
                    river_name=nested_get(
                        item,
                        self.mapping.river_name,
                    ),
                    district=nested_get(
                        item,
                        self.mapping.district,
                    ),
                    state=nested_get(
                        item,
                        self.mapping.state,
                    ),
                    latitude=latitude,
                    longitude=longitude,
                    source_uri=self.source_uri,
                    verification=self.verification,
                )
            )

        return stations

    def parse_observations(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[CWCWaterLevelObservation]:
        records = self._records(
            payload,
            self.mapping.observation_list_path,
        )

        now = (
            now or utc_now()
        )

        observations = []

        for item in records:
            station_id = nested_get(
                item,
                self.mapping.observation_station_id,
            )

            raw_level = nested_get(
                item,
                self.mapping.water_level,
            )

            raw_time = nested_get(
                item,
                self.mapping.observed_at,
            )

            if (
                station_id is None
                or raw_level is None
                or raw_time is None
            ):
                raise CWCResponseError(
                    "Required observation field missing"
                )

            water_level_m = to_meters(
                finite_number(
                    raw_level,
                    "water_level",
                ),
                self.mapping.water_level_unit,
            )

            observed_at = parse_timestamp(
                raw_time
            )

            thresholds = CWCWaterLevelThresholds(
                warning_level_m=self._threshold(
                    item,
                    self.mapping.warning_level,
                ),
                danger_level_m=self._threshold(
                    item,
                    self.mapping.danger_level,
                ),
                high_flood_level_m=self._threshold(
                    item,
                    self.mapping.high_flood_level,
                ),
            )

            data_status = self._data_status(
                observed_at,
                now=now,
            )

            threshold_status = (
                self._threshold_status(
                    water_level_m,
                    thresholds,
                )
            )

            observations.append(
                CWCWaterLevelObservation(
                    station_id=str(station_id),
                    water_level_m=water_level_m,
                    observed_at=observed_at,
                    thresholds=thresholds,
                    threshold_status=threshold_status,
                    data_status=data_status,
                    source_uri=self.source_uri,
                    verification=self.verification,
                    retrieved_at=now,
                )
            )

        return observations

    def _threshold(
        self,
        item: Mapping[str, Any],
        path: str,
    ) -> float | None:
        value = self._optional_number(
            nested_get(item, path)
        )

        if value is None:
            return None

        return to_meters(
            value,
            self.mapping.water_level_unit,
        )

    def _data_status(
        self,
        observed_at: datetime,
        *,
        now: datetime,
    ) -> CWCDataStatus:
        age = now - observed_at

        if age < timedelta(0):
            return CWCDataStatus.FUTURE_TIMESTAMP

        if age > timedelta(
            minutes=self.stale_after_minutes
        ):
            return CWCDataStatus.STALE

        return CWCDataStatus.CURRENT

    @staticmethod
    def _threshold_status(
        water_level_m: float,
        thresholds: CWCWaterLevelThresholds,
    ) -> CWCThresholdStatus:
        if (
            thresholds.high_flood_level_m is not None
            and water_level_m
            >= thresholds.high_flood_level_m
        ):
            return CWCThresholdStatus.HIGH_FLOOD_THRESHOLD

        if (
            thresholds.danger_level_m is not None
            and water_level_m
            >= thresholds.danger_level_m
        ):
            return CWCThresholdStatus.DANGER_THRESHOLD

        if (
            thresholds.warning_level_m is not None
            and water_level_m
            >= thresholds.warning_level_m
        ):
            return CWCThresholdStatus.WARNING_THRESHOLD

        if thresholds.warning_level_m is not None:
            return CWCThresholdStatus.BELOW_WARNING

        return CWCThresholdStatus.UNKNOWN


# ============================================================
# OBSERVATION DEDUPLICATION
# ============================================================

class CWCObservationDeduplicator:
    @staticmethod
    def deduplicate(
        observations: Sequence[CWCWaterLevelObservation],
    ) -> list[CWCWaterLevelObservation]:
        unique: dict[
            tuple[str, datetime],
            CWCWaterLevelObservation,
        ] = {}

        for observation in observations:
            key = (
                observation.station_id,
                observation.observed_at,
            )

            previous = unique.get(key)

            if (
                previous is None
                or observation.retrieved_at
                > previous.retrieved_at
            ):
                unique[key] = observation

        return sorted(
            unique.values(),
            key=lambda item: (
                item.station_id,
                item.observed_at,
            ),
        )


# ============================================================
# MAIN CWC ADAPTER
# ============================================================

class CWCAdapter:
    def __init__(
        self,
        config: CWCAdapterConfig,
        *,
        mapping: CWCFieldMapping | None = None,
        http_client: CWCHTTPClient | None = None,
    ) -> None:
        self.config = config

        self.mapping = (
            mapping or CWCFieldMapping()
        )

        self.http_client = (
            http_client
            or CWCHTTPClient(config)
        )

        self.parser = CWCResponseParser(
            self.mapping,
            source_uri=config.base_url,
            source_verified=config.source_verified,
            stale_after_minutes=config.stale_after_minutes,
        )

        self.deduplicator = (
            CWCObservationDeduplicator()
        )

    async def list_stations(
        self,
        *,
        state: str | None = None,
        district: str | None = None,
    ) -> list[CWCStation]:
        params: dict[str, str] = {}

        if state:
            params["state"] = state

        if district:
            params["district"] = district

        payload = await self.http_client.get_json(
            self.config.station_endpoint,
            params=params,
        )

        stations = self.parser.parse_stations(
            payload
        )

        if state:
            stations = [
                station
                for station in stations
                if station.state
                and station.state.casefold()
                == state.casefold()
            ]

        if district:
            stations = [
                station
                for station in stations
                if station.district
                and station.district.casefold()
                == district.casefold()
            ]

        return stations

    async def get_latest_observations(
        self,
        *,
        station_ids: Sequence[str] | None = None,
    ) -> list[CWCWaterLevelObservation]:
        params: dict[str, str] = {}

        if station_ids:
            if len(station_ids) > (
                self.config.max_stations_per_batch
            ):
                raise CWCValidationError(
                    "Station batch limit exceeded"
                )

            params["station_ids"] = ",".join(
                station_ids
            )

        payload = await self.http_client.get_json(
            self.config.observation_endpoint,
            params=params,
        )

        observations = (
            self.parser.parse_observations(
                payload
            )
        )

        observations = (
            self.deduplicator.deduplicate(
                observations
            )
        )

        if station_ids:
            allowed_ids = set(
                station_ids
            )

            observations = [
                observation
                for observation in observations
                if observation.station_id in allowed_ids
            ]

        latest: dict[
            str,
            CWCWaterLevelObservation,
        ] = {}

        for observation in observations:
            previous = latest.get(
                observation.station_id
            )

            if (
                previous is None
                or observation.observed_at
                > previous.observed_at
            ):
                latest[
                    observation.station_id
                ] = observation

        return list(
            latest.values()
        )

    async def get_station_history(
        self,
        station_id: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[CWCWaterLevelObservation]:
        start = ensure_aware_datetime(
            start
        )

        end = ensure_aware_datetime(
            end
        )

        if end <= start:
            raise CWCValidationError(
                "end must be after start"
            )

        if not station_id.strip():
            raise CWCValidationError(
                "station_id required"
            )

        payload = await self.http_client.get_json(
            self.config.history_endpoint,
            params={
                "station_id": station_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )

        observations = (
            self.parser.parse_observations(
                payload
            )
        )

        observations = [
            observation
            for observation in observations
            if observation.station_id == station_id
            and start
            <= observation.observed_at
            <= end
        ]

        if len(observations) > (
            self.config.max_history_records
        ):
            raise CWCValidationError(
                "History record limit exceeded"
            )

        return self.deduplicator.deduplicate(
            observations
        )

    async def get_kerala_snapshot(
        self,
        *,
        district: str | None = None,
    ) -> CWCAdapterResponse:
        stations = await self.list_stations(
            state="Kerala",
            district=district,
        )

        station_ids = [
            station.station_id
            for station in stations
        ]

        observations = []

        for offset in range(
            0,
            len(station_ids),
            self.config.max_stations_per_batch,
        ):
            batch = station_ids[
                offset:
                offset + self.config.max_stations_per_batch
            ]

            observations.extend(
                await self.get_latest_observations(
                    station_ids=batch
                )
            )

        observation_map = {
            observation.station_id: observation
            for observation in observations
        }

        results = []

        warnings = []

        for station in stations:
            observation = observation_map.get(
                station.station_id
            )

            station_warnings = []

            if observation is None:
                station_warnings.append(
                    "No water-level observation available"
                )

            else:
                if observation.data_status != CWCDataStatus.CURRENT:
                    station_warnings.append(
                        "Observation is not current"
                    )

                if (
                    observation.threshold_status
                    != CWCThresholdStatus.BELOW_WARNING
                    and observation.threshold_status
                    != CWCThresholdStatus.UNKNOWN
                ):
                    station_warnings.append(
                        "Configured threshold reached; "
                        "human verification required"
                    )

            results.append(
                CWCStationResult(
                    station=station,
                    latest_observation=observation,
                    warnings=tuple(station_warnings),
                )
            )

        if not self.config.source_verified:
            warnings.append(
                "Source authenticity has not been "
                "verified by the application"
            )

        return CWCAdapterResponse(
            results=tuple(results),
            source_verified=self.config.source_verified,
            retrieved_at=utc_now(),
            warnings=tuple(warnings),
        )

    async def close(self) -> None:
        await self.http_client.close()

    async def __aenter__(self) -> "CWCAdapter":
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

class CWCRAGAdapter:
    """
    Converts verified CWC observations into
    structured RAG ingestion records.

    Unverified or stale observations are
    excluded by default.
    """

    @staticmethod
    def to_rag_records(
        response: CWCAdapterResponse,
        *,
        tenant_id: str,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        if not tenant_id.strip():
            raise CWCValidationError(
                "tenant_id required"
            )

        records = []

        for result in response.results:
            observation = result.latest_observation

            if observation is None:
                continue

            if (
                observation.verification
                != CWCSourceVerification.VERIFIED
            ):
                continue

            if (
                not include_stale
                and observation.data_status
                != CWCDataStatus.CURRENT
            ):
                continue

            station = result.station

            text = (
                f"Station: {station.station_name}. "
                f"River: {station.river_name or 'unknown'}. "
                f"District: {station.district or 'unknown'}. "
                f"Observed water level: "
                f"{observation.water_level_m:.3f} metres. "
                f"Observation time: "
                f"{observation.observed_at.isoformat()}. "
                f"Configured threshold comparison: "
                f"{observation.threshold_status.value}."
            )

            records.append({
                "id": (
                    f"cwc:{station.station_id}:"
                    f"{observation.observed_at.isoformat()}"
                ),
                "content": text,
                "metadata": {
                    "tenant_id": tenant_id,
                    "station_id": station.station_id,
                    "station_name": station.station_name,
                    "river_name": station.river_name,
                    "district": station.district,
                    "state": station.state,
                    "water_level_m": observation.water_level_m,
                    "observed_at": (
                        observation.observed_at.isoformat()
                    ),
                    "threshold_status": (
                        observation.threshold_status.value
                    ),
                    "source_uri": observation.source_uri,
                    "verification": (
                        observation.verification.value
                    ),
                    "requires_human_review": True,
                },
            })

        return records


# ============================================================
# EXPORTER
# ============================================================

class CWCResponseExporter:
    @staticmethod
    def to_json(
        response: CWCAdapterResponse,
    ) -> str:
        return json.dumps(
            response.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


# ============================================================
# FACTORY
# ============================================================

def create_cwc_adapter(
    config: CWCAdapterConfig,
    *,
    mapping: CWCFieldMapping | None = None,
) -> CWCAdapter:
    return CWCAdapter(
        config=config,
        mapping=mapping,
    )


# ============================================================
# FICTIONAL PARSER DEMO
# ============================================================

def demo() -> None:
    """
    Synthetic data only.

    No network calls.
    No real CWC observations.
    """

    parser = CWCResponseParser(
        CWCFieldMapping(),
        source_uri="demo://cwc",
        source_verified=False,
        stale_after_minutes=180,
    )

    payload = {
        "data": [
            {
                "station_id": "DEMO-STATION-001",
                "water_level": 10.5,
                "observed_at": utc_now().isoformat(),
                "warning_level": 11.0,
                "danger_level": 12.0,
                "high_flood_level": 13.0,
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
    "CWCAdapterError",
    "CWCConfigurationError",
    "CWCAuthenticationError",
    "CWCRequestError",
    "CWCResponseError",
    "CWCValidationError",
    "CWCDataUnavailableError",
    "CWCAuthenticationType",
    "CWCWaterLevelUnit",
    "CWCThresholdStatus",
    "CWCDataStatus",
    "CWCSourceVerification",
    "CWCAdapterConfig",
    "CWCFieldMapping",
    "CWCStation",
    "CWCWaterLevelThresholds",
    "CWCWaterLevelObservation",
    "CWCStationResult",
    "CWCAdapterResponse",
    "CWCHTTPClient",
    "CWCResponseParser",
    "CWCObservationDeduplicator",
    "CWCAdapter",
    "CWCRAGAdapter",
    "CWCResponseExporter",
    "create_cwc_adapter",
]
