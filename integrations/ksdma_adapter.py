"""Placeholder module: integrations/ksdma_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Kerala State Disaster Management Authority Adapter

File: backend/integrations/ksdma_adapter.py

Python 3.11+
Dependency:
    pip install httpx

Features:
    - Authorized KSDMA API integration
    - Official disaster advisories
    - Flood and weather-related alerts
    - District emergency bulletins
    - Relief and shelter information
    - Emergency contact directories
    - Document and bulletin references
    - Kerala district filtering
    - Source verification controls
    - Alert validity and freshness checks
    - Human-review requirements
    - Configurable JSON response mapping
    - HTTPS, authentication and retries
    - Response size limits
    - Duplicate removal
    - Async requests
    - RAG ingestion adapter
    - Notification preparation adapter
    - Structured JSON export

IMPORTANT:
    The API endpoints and JSON fields in this
    module are examples, NOT confirmed official
    KSDMA API specifications.

    Configure documented, authorized sources.

    Never treat a configurable "verified" flag
    as independent evidence of authenticity.

    Do not automatically issue evacuation orders,
    dispatch rescue teams, publish warnings or
    modify shelter capacity.

    Emergency data requires authoritative
    verification and human oversight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence
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

class KSDMAAdapterError(Exception):
    pass


class KSDMAConfigurationError(KSDMAAdapterError):
    pass


class KSDMAAuthenticationError(KSDMAAdapterError):
    pass


class KSDMARequestError(KSDMAAdapterError):
    pass


class KSDMAResponseError(KSDMAAdapterError):
    pass


class KSDMAValidationError(KSDMAAdapterError):
    pass


class KSDMASourceVerificationError(KSDMAAdapterError):
    pass


class KSDMADataUnavailableError(KSDMAAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class KSDMAAuthenticationType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"


class KSDMAAlertSeverity(str, Enum):
    INFORMATION = "information"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class KSDMAHazardType(str, Enum):
    FLOOD = "flood"
    HEAVY_RAIN = "heavy_rain"
    LANDSLIDE = "landslide"
    CYCLONE = "cyclone"
    COASTAL_FLOODING = "coastal_flooding"
    DAM_RELATED = "dam_related"
    LIGHTNING = "lightning"
    STRONG_WIND = "strong_wind"
    EARTHQUAKE = "earthquake"
    FIRE = "fire"
    OTHER = "other"


class KSDMARecordStatus(str, Enum):
    ACTIVE = "active"
    UPCOMING = "upcoming"
    EXPIRED = "expired"
    STALE = "stale"
    FUTURE_ISSUE_TIME = "future_issue_time"
    UNKNOWN = "unknown"


class KSDMASourceVerification(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class KSDMAShelterStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    FULL = "full"
    UNKNOWN = "unknown"


class KSDMARecordType(str, Enum):
    ALERT = "alert"
    BULLETIN = "bulletin"
    SHELTER = "shelter"
    EMERGENCY_CONTACT = "emergency_contact"


# ============================================================
# CONSTANTS
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


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise KSDMAValidationError(
            "Expected a string"
        )

    return unicodedata.normalize(
        "NFC",
        value,
    ).strip()


def normalize_district(
    value: str,
) -> str:
    district = DISTRICT_LOOKUP.get(
        normalize_text(value).casefold()
    )

    if district is None:
        raise KSDMAValidationError(
            "Unsupported Kerala district"
        )

    return district


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value

    elif isinstance(value, str):
        raw = value.strip()

        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        try:
            result = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise KSDMAValidationError(
                "Invalid ISO timestamp"
            ) from exc

    else:
        raise KSDMAValidationError(
            "Unsupported timestamp type"
        )

    if (
        result.tzinfo is None
        or result.utcoffset() is None
    ):
        raise KSDMAValidationError(
            "Timestamp must contain timezone"
        )

    return result.astimezone(timezone.utc)


def optional_timestamp(
    value: Any,
) -> datetime | None:
    if value is None or value == "":
        return None

    return parse_timestamp(value)


def finite_number(
    value: Any,
    field_name: str,
) -> float:
    if isinstance(value, bool):
        raise KSDMAValidationError(
            f"{field_name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise KSDMAValidationError(
            f"{field_name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise KSDMAValidationError(
            f"{field_name} must be finite"
        )

    return number


def optional_number(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None or value == "":
        return None

    return finite_number(value, field_name)


def optional_integer(
    value: Any,
    field_name: str,
) -> int | None:
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        raise KSDMAValidationError(
            f"{field_name} must be an integer"
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise KSDMAValidationError(
            f"{field_name} must be an integer"
        ) from exc

    if str(result) != str(value).strip():
        raise KSDMAValidationError(
            f"{field_name} must be an exact integer"
        )

    return result


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


def enum_or_default(
    enum_type: type[Enum],
    value: Any,
    default: Enum,
) -> Enum:
    try:
        return enum_type(
            str(value).strip().casefold()
        )
    except (ValueError, TypeError):
        return default


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

    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]

    return value


def validate_reference_url(
    value: str,
    *,
    allowed_hosts: Sequence[str],
) -> str:
    parsed = urlparse(value)

    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise KSDMAValidationError(
            "Reference must use a valid HTTPS URL"
        )

    if (
        allowed_hosts
        and parsed.hostname not in allowed_hosts
    ):
        raise KSDMAValidationError(
            "Reference host is not allowlisted"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class KSDMAAdapterConfig:
    base_url: str

    alerts_endpoint: str = "/alerts"
    bulletins_endpoint: str = "/bulletins"
    shelters_endpoint: str = "/shelters"
    contacts_endpoint: str = "/emergency-contacts"

    authentication_type: KSDMAAuthenticationType = (
        KSDMAAuthenticationType.NONE
    )

    api_key: str | None = None
    api_key_header: str = "X-API-Key"
    bearer_token: str | None = None

    timeout_seconds: float = 20.0
    max_retries: int = 3
    retry_base_seconds: float = 0.5

    max_response_bytes: int = 5_000_000
    max_records_per_response: int = 10_000
    max_batch_districts: int = 14

    bulletin_stale_after_hours: int = 72
    shelter_stale_after_minutes: int = 60
    contacts_stale_after_days: int = 30

    source_verified: bool = False
    require_verified_source: bool = True

    allowed_hosts: tuple[str, ...] = ()
    allowed_reference_hosts: tuple[str, ...] = ()

    allow_local_http_testing: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)

        if not parsed.hostname:
            raise KSDMAConfigurationError(
                "Valid base_url required"
            )

        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise KSDMAConfigurationError(
                "Credentials and query parameters "
                "are not permitted in base_url"
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
                raise KSDMAConfigurationError(
                    "HTTPS required"
                )

        if (
            self.allowed_hosts
            and parsed.hostname not in self.allowed_hosts
        ):
            raise KSDMAConfigurationError(
                "Host not allowlisted"
            )

        for endpoint in (
            self.alerts_endpoint,
            self.bulletins_endpoint,
            self.shelters_endpoint,
            self.contacts_endpoint,
        ):
            if (
                not endpoint.startswith("/")
                or endpoint.startswith("//")
                or "://" in endpoint
                or "?" in endpoint
                or "#" in endpoint
                or ".." in endpoint.split("/")
            ):
                raise KSDMAConfigurationError(
                    "Invalid endpoint path"
                )

        if (
            self.authentication_type
            == KSDMAAuthenticationType.API_KEY
            and not self.api_key
        ):
            raise KSDMAConfigurationError(
                "API key required"
            )

        if (
            self.authentication_type
            == KSDMAAuthenticationType.BEARER
            and not self.bearer_token
        ):
            raise KSDMAConfigurationError(
                "Bearer token required"
            )

        if self.timeout_seconds <= 0:
            raise KSDMAConfigurationError(
                "Timeout must be positive"
            )

        if self.max_retries < 0:
            raise KSDMAConfigurationError(
                "Retry count cannot be negative"
            )

        for name in (
            "max_response_bytes",
            "max_records_per_response",
            "max_batch_districts",
            "bulletin_stale_after_hours",
            "shelter_stale_after_minutes",
            "contacts_stale_after_days",
        ):
            if getattr(self, name) <= 0:
                raise KSDMAConfigurationError(
                    f"{name} must be positive"
                )


# ============================================================
# JSON FIELD MAPPING
# ============================================================

@dataclass(frozen=True)
class KSDMAFieldMapping:
    alerts_list_path: str = "data"
    bulletins_list_path: str = "data"
    shelters_list_path: str = "data"
    contacts_list_path: str = "data"

    alert_id: str = "alert_id"
    alert_district: str = "district"
    alert_hazard: str = "hazard_type"
    alert_severity: str = "severity"
    alert_title: str = "title"
    alert_message: str = "message"
    alert_issued_at: str = "issued_at"
    alert_valid_from: str = "valid_from"
    alert_valid_until: str = "valid_until"
    alert_reference_url: str = "reference_url"

    bulletin_id: str = "bulletin_id"
    bulletin_district: str = "district"
    bulletin_title: str = "title"
    bulletin_summary: str = "summary"
    bulletin_published_at: str = "published_at"
    bulletin_document_url: str = "document_url"

    shelter_id: str = "shelter_id"
    shelter_name: str = "shelter_name"
    shelter_district: str = "district"
    shelter_address: str = "address"
    shelter_status: str = "status"
    shelter_capacity: str = "capacity"
    shelter_occupancy: str = "occupancy"
    shelter_updated_at: str = "updated_at"

    contact_id: str = "contact_id"
    contact_district: str = "district"
    contact_department: str = "department"
    contact_name: str = "name"
    contact_phone: str = "phone"
    contact_updated_at: str = "updated_at"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class KSDMAOfficialAlert:
    alert_id: str
    district: str

    hazard_type: KSDMAHazardType
    severity: KSDMAAlertSeverity

    title: str
    message: str

    issued_at: datetime
    valid_from: datetime
    valid_until: datetime

    status: KSDMARecordStatus

    source_uri: str
    reference_url: str | None

    verification: KSDMASourceVerification

    retrieved_at: datetime

    requires_human_review: bool = True

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class KSDMABulletin:
    bulletin_id: str
    district: str | None

    title: str
    summary: str

    published_at: datetime

    document_url: str | None

    status: KSDMARecordStatus

    source_uri: str
    verification: KSDMASourceVerification

    retrieved_at: datetime

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class KSDMAShelter:
    shelter_id: str
    name: str
    district: str

    address: str

    status: KSDMAShelterStatus

    capacity: int | None
    occupancy: int | None

    updated_at: datetime

    data_status: KSDMARecordStatus

    source_uri: str
    verification: KSDMASourceVerification

    retrieved_at: datetime

    requires_confirmation: bool = True

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def available_spaces(self) -> int | None:
        if (
            self.capacity is None
            or self.occupancy is None
        ):
            return None

        return max(
            self.capacity - self.occupancy,
            0,
        )


@dataclass(frozen=True)
class KSDMAEmergencyContact:
    contact_id: str
    district: str | None

    department: str
    name: str

    phone: str

    updated_at: datetime

    data_status: KSDMARecordStatus

    source_uri: str
    verification: KSDMASourceVerification

    retrieved_at: datetime


@dataclass(frozen=True)
class KSDMADistrictSnapshot:
    district: str

    alerts: tuple[KSDMAOfficialAlert, ...]
    bulletins: tuple[KSDMABulletin, ...]
    shelters: tuple[KSDMAShelter, ...]
    contacts: tuple[KSDMAEmergencyContact, ...]

    retrieved_at: datetime
    source_verified: bool

    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# SOURCE AND FRESHNESS POLICY
# ============================================================

class KSDMASourcePolicy:
    def __init__(
        self,
        config: KSDMAAdapterConfig,
    ) -> None:
        self.config = config

    @property
    def verification(self) -> KSDMASourceVerification:
        return (
            KSDMASourceVerification.VERIFIED
            if self.config.source_verified
            else KSDMASourceVerification.UNVERIFIED
        )

    def require_verified(self) -> None:
        if (
            self.config.require_verified_source
            and not self.config.source_verified
        ):
            raise KSDMASourceVerificationError(
                "Source has not been independently "
                "verified for operational use"
            )

    @staticmethod
    def alert_status(
        issued_at: datetime,
        valid_from: datetime,
        valid_until: datetime,
        *,
        now: datetime,
    ) -> KSDMARecordStatus:
        if issued_at > now:
            return KSDMARecordStatus.FUTURE_ISSUE_TIME

        if now < valid_from:
            return KSDMARecordStatus.UPCOMING

        if now > valid_until:
            return KSDMARecordStatus.EXPIRED

        return KSDMARecordStatus.ACTIVE

    @staticmethod
    def age_status(
        timestamp: datetime,
        *,
        now: datetime,
        maximum_age: timedelta,
    ) -> KSDMARecordStatus:
        age = now - timestamp

        if age < timedelta(0):
            return KSDMARecordStatus.FUTURE_ISSUE_TIME

        if age > maximum_age:
            return KSDMARecordStatus.STALE

        return KSDMARecordStatus.ACTIVE


# ============================================================
# HTTP CLIENT
# ============================================================

class KSDMAHTTPClient:
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        config: KSDMAAdapterConfig,
    ) -> None:
        if httpx is None:
            raise KSDMAConfigurationError(
                "Install httpx: pip install httpx"
            )

        self.config = config
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "KeralaFloodReliefAgent/1.0",
        }

        if (
            self.config.authentication_type
            == KSDMAAuthenticationType.API_KEY
        ):
            headers[self.config.api_key_header] = (
                self.config.api_key or ""
            )

        elif (
            self.config.authentication_type
            == KSDMAAuthenticationType.BEARER
        ):
            headers["Authorization"] = (
                f"Bearer {self.config.bearer_token}"
            )

        return headers

    async def _get_client(self) -> httpx.AsyncClient:
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
        allowed = {
            self.config.alerts_endpoint,
            self.config.bulletins_endpoint,
            self.config.shelters_endpoint,
            self.config.contacts_endpoint,
        }

        if endpoint not in allowed:
            raise KSDMAConfigurationError(
                "Unapproved endpoint"
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
                    if response.status_code in (401, 403):
                        raise KSDMAAuthenticationError(
                            "Authentication failed"
                        )

                    if (
                        response.status_code
                        in self.RETRYABLE_STATUS_CODES
                    ):
                        raise KSDMARequestError(
                            "Temporary upstream failure"
                        )

                    response.raise_for_status()

                    raw_length = response.headers.get(
                        "Content-Length"
                    )

                    if raw_length is not None:
                        try:
                            declared_size = int(raw_length)
                        except ValueError as exc:
                            raise KSDMAResponseError(
                                "Invalid Content-Length"
                            ) from exc

                        if declared_size > (
                            self.config.max_response_bytes
                        ):
                            raise KSDMAResponseError(
                                "Response too large"
                            )

                    chunks = []
                    total_size = 0

                    async for chunk in response.aiter_bytes():
                        total_size += len(chunk)

                        if total_size > (
                            self.config.max_response_bytes
                        ):
                            raise KSDMAResponseError(
                                "Response size limit exceeded"
                            )

                        chunks.append(chunk)

                payload = json.loads(
                    b"".join(chunks)
                )

                if not isinstance(
                    payload,
                    (dict, list),
                ):
                    raise KSDMAResponseError(
                        "Invalid JSON root"
                    )

                return payload

            except (
                KSDMAAuthenticationError,
                KSDMAResponseError,
            ):
                raise

            except (
                httpx.TimeoutException,
                httpx.TransportError,
                KSDMARequestError,
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
                raise KSDMARequestError(
                    f"Upstream HTTP "
                    f"{exc.response.status_code}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise KSDMAResponseError(
                    "Invalid JSON response"
                ) from exc

        raise KSDMARequestError(
            "Request failed after retries"
        ) from last_error

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(
        self,
    ) -> "KSDMAHTTPClient":
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

class KSDMAResponseParser:
    def __init__(
        self,
        config: KSDMAAdapterConfig,
        mapping: KSDMAFieldMapping,
    ) -> None:
        self.config = config
        self.mapping = mapping
        self.policy = KSDMASourcePolicy(config)

    def _records(
        self,
        payload: Mapping[str, Any] | list[Any],
        path: str,
    ) -> list[Mapping[str, Any]]:
        if isinstance(payload, list):
            records = payload
        else:
            records = nested_get(payload, path)

        if not isinstance(records, list):
            raise KSDMAResponseError(
                "Expected JSON record array"
            )

        if len(records) > (
            self.config.max_records_per_response
        ):
            raise KSDMAResponseError(
                "Too many records"
            )

        if not all(
            isinstance(record, Mapping)
            for record in records
        ):
            raise KSDMAResponseError(
                "Invalid record structure"
            )

        return records

    def _reference(
        self,
        value: Any,
    ) -> str | None:
        if not value:
            return None

        return validate_reference_url(
            str(value),
            allowed_hosts=(
                self.config.allowed_reference_hosts
                or self.config.allowed_hosts
            ),
        )

    def parse_alerts(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[KSDMAOfficialAlert]:
        now = now or utc_now()
        m = self.mapping

        output = []

        for item in self._records(
            payload,
            m.alerts_list_path,
        ):
            alert_id = nested_get(item, m.alert_id)
            district = nested_get(item, m.alert_district)

            if not alert_id or not district:
                raise KSDMAResponseError(
                    "Alert ID and district required"
                )

            issued_at = parse_timestamp(
                nested_get(item, m.alert_issued_at)
            )
            valid_from = parse_timestamp(
                nested_get(item, m.alert_valid_from)
            )
            valid_until = parse_timestamp(
                nested_get(item, m.alert_valid_until)
            )

            if valid_until <= valid_from:
                raise KSDMAValidationError(
                    "Invalid alert validity period"
                )

            severity = enum_or_default(
                KSDMAAlertSeverity,
                nested_get(item, m.alert_severity),
                KSDMAAlertSeverity.UNKNOWN,
            )

            hazard = enum_or_default(
                KSDMAHazardType,
                nested_get(item, m.alert_hazard),
                KSDMAHazardType.OTHER,
            )

            output.append(
                KSDMAOfficialAlert(
                    alert_id=str(alert_id),
                    district=normalize_district(
                        str(district)
                    ),
                    hazard_type=hazard,
                    severity=severity,
                    title=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.alert_title,
                                "",
                            )
                        )
                    ),
                    message=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.alert_message,
                                "",
                            )
                        )
                    ),
                    issued_at=issued_at,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    status=self.policy.alert_status(
                        issued_at,
                        valid_from,
                        valid_until,
                        now=now,
                    ),
                    source_uri=self.config.base_url,
                    reference_url=self._reference(
                        nested_get(
                            item,
                            m.alert_reference_url,
                        )
                    ),
                    verification=self.policy.verification,
                    retrieved_at=now,
                )
            )

        return output

    def parse_bulletins(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[KSDMABulletin]:
        now = now or utc_now()
        m = self.mapping

        output = []

        for item in self._records(
            payload,
            m.bulletins_list_path,
        ):
            bulletin_id = nested_get(
                item,
                m.bulletin_id,
            )

            if not bulletin_id:
                raise KSDMAResponseError(
                    "Bulletin ID required"
                )

            raw_district = nested_get(
                item,
                m.bulletin_district,
            )

            published_at = parse_timestamp(
                nested_get(
                    item,
                    m.bulletin_published_at,
                )
            )

            output.append(
                KSDMABulletin(
                    bulletin_id=str(bulletin_id),
                    district=(
                        normalize_district(
                            str(raw_district)
                        )
                        if raw_district
                        else None
                    ),
                    title=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.bulletin_title,
                                "",
                            )
                        )
                    ),
                    summary=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.bulletin_summary,
                                "",
                            )
                        )
                    ),
                    published_at=published_at,
                    document_url=self._reference(
                        nested_get(
                            item,
                            m.bulletin_document_url,
                        )
                    ),
                    status=self.policy.age_status(
                        published_at,
                        now=now,
                        maximum_age=timedelta(
                            hours=(
                                self.config
                                .bulletin_stale_after_hours
                            )
                        ),
                    ),
                    source_uri=self.config.base_url,
                    verification=self.policy.verification,
                    retrieved_at=now,
                )
            )

        return output

    def parse_shelters(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[KSDMAShelter]:
        now = now or utc_now()
        m = self.mapping

        output = []

        for item in self._records(
            payload,
            m.shelters_list_path,
        ):
            shelter_id = nested_get(
                item,
                m.shelter_id,
            )

            district = nested_get(
                item,
                m.shelter_district,
            )

            if not shelter_id or not district:
                raise KSDMAResponseError(
                    "Shelter ID and district required"
                )

            capacity = optional_integer(
                nested_get(
                    item,
                    m.shelter_capacity,
                ),
                "capacity",
            )

            occupancy = optional_integer(
                nested_get(
                    item,
                    m.shelter_occupancy,
                ),
                "occupancy",
            )

            if capacity is not None and capacity < 0:
                raise KSDMAValidationError(
                    "Capacity cannot be negative"
                )

            if occupancy is not None and occupancy < 0:
                raise KSDMAValidationError(
                    "Occupancy cannot be negative"
                )

            updated_at = parse_timestamp(
                nested_get(
                    item,
                    m.shelter_updated_at,
                )
            )

            output.append(
                KSDMAShelter(
                    shelter_id=str(shelter_id),
                    name=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.shelter_name,
                                "",
                            )
                        )
                    ),
                    district=normalize_district(
                        str(district)
                    ),
                    address=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.shelter_address,
                                "",
                            )
                        )
                    ),
                    status=enum_or_default(
                        KSDMAShelterStatus,
                        nested_get(
                            item,
                            m.shelter_status,
                        ),
                        KSDMAShelterStatus.UNKNOWN,
                    ),
                    capacity=capacity,
                    occupancy=occupancy,
                    updated_at=updated_at,
                    data_status=self.policy.age_status(
                        updated_at,
                        now=now,
                        maximum_age=timedelta(
                            minutes=(
                                self.config
                                .shelter_stale_after_minutes
                            )
                        ),
                    ),
                    source_uri=self.config.base_url,
                    verification=self.policy.verification,
                    retrieved_at=now,
                )
            )

        return output

    def parse_contacts(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        now: datetime | None = None,
    ) -> list[KSDMAEmergencyContact]:
        now = now or utc_now()
        m = self.mapping

        output = []

        for item in self._records(
            payload,
            m.contacts_list_path,
        ):
            contact_id = nested_get(
                item,
                m.contact_id,
            )

            if not contact_id:
                raise KSDMAResponseError(
                    "Contact ID required"
                )

            raw_district = nested_get(
                item,
                m.contact_district,
            )

            phone = normalize_text(
                str(
                    nested_get(
                        item,
                        m.contact_phone,
                        "",
                    )
                )
            )

            if not re.fullmatch(
                r"[+0-9()\-\s]{5,30}",
                phone,
            ):
                raise KSDMAValidationError(
                    "Invalid contact phone format"
                )

            updated_at = parse_timestamp(
                nested_get(
                    item,
                    m.contact_updated_at,
                )
            )

            output.append(
                KSDMAEmergencyContact(
                    contact_id=str(contact_id),
                    district=(
                        normalize_district(
                            str(raw_district)
                        )
                        if raw_district
                        else None
                    ),
                    department=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.contact_department,
                                "",
                            )
                        )
                    ),
                    name=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.contact_name,
                                "",
                            )
                        )
                    ),
                    phone=phone,
                    updated_at=updated_at,
                    data_status=self.policy.age_status(
                        updated_at,
                        now=now,
                        maximum_age=timedelta(
                            days=(
                                self.config
                                .contacts_stale_after_days
                            )
                        ),
                    ),
                    source_uri=self.config.base_url,
                    verification=self.policy.verification,
                    retrieved_at=now,
                )
            )

        return output


# ============================================================
# DEDUPLICATION
# ============================================================

class KSDMARecordDeduplicator:
    @staticmethod
    def alerts(
        records: Sequence[KSDMAOfficialAlert],
    ) -> list[KSDMAOfficialAlert]:
        unique = {}

        for record in records:
            key = (
                record.alert_id,
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

    @staticmethod
    def bulletins(
        records: Sequence[KSDMABulletin],
    ) -> list[KSDMABulletin]:
        unique = {}

        for record in records:
            key = (
                record.bulletin_id,
                record.published_at,
            )
            unique[key] = record

        return list(unique.values())

    @staticmethod
    def shelters(
        records: Sequence[KSDMAShelter],
    ) -> list[KSDMAShelter]:
        unique = {}

        for record in records:
            previous = unique.get(
                record.shelter_id
            )

            if (
                previous is None
                or record.updated_at
                > previous.updated_at
            ):
                unique[
                    record.shelter_id
                ] = record

        return list(unique.values())

    @staticmethod
    def contacts(
        records: Sequence[KSDMAEmergencyContact],
    ) -> list[KSDMAEmergencyContact]:
        unique = {}

        for record in records:
            previous = unique.get(
                record.contact_id
            )

            if (
                previous is None
                or record.updated_at
                > previous.updated_at
            ):
                unique[
                    record.contact_id
                ] = record

        return list(unique.values())


# ============================================================
# MAIN KSDMA ADAPTER
# ============================================================

class KSDMAAdapter:
    def __init__(
        self,
        config: KSDMAAdapterConfig,
        *,
        mapping: KSDMAFieldMapping | None = None,
        http_client: KSDMAHTTPClient | None = None,
    ) -> None:
        self.config = config
        self.mapping = mapping or KSDMAFieldMapping()

        self.http_client = (
            http_client
            or KSDMAHTTPClient(config)
        )

        self.policy = KSDMASourcePolicy(config)

        self.parser = KSDMAResponseParser(
            config,
            self.mapping,
        )

        self.deduplicator = (
            KSDMARecordDeduplicator()
        )

    async def get_alerts(
        self,
        *,
        district: str | None = None,
        active_only: bool = True,
    ) -> list[KSDMAOfficialAlert]:
        self.policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.alerts_endpoint,
            params=params,
        )

        records = self.parser.parse_alerts(
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
                if record.status
                == KSDMARecordStatus.ACTIVE
            ]

        return self.deduplicator.alerts(
            records
        )

    async def get_bulletins(
        self,
        *,
        district: str | None = None,
    ) -> list[KSDMABulletin]:
        self.policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.bulletins_endpoint,
            params=params,
        )

        records = self.parser.parse_bulletins(
            payload
        )

        if district:
            selected = normalize_district(
                district
            )

            records = [
                record
                for record in records
                if record.district in (
                    None,
                    selected,
                )
            ]

        return self.deduplicator.bulletins(
            records
        )

    async def get_shelters(
        self,
        *,
        district: str | None = None,
    ) -> list[KSDMAShelter]:
        self.policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.shelters_endpoint,
            params=params,
        )

        records = self.parser.parse_shelters(
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

        return self.deduplicator.shelters(
            records
        )

    async def get_emergency_contacts(
        self,
        *,
        district: str | None = None,
    ) -> list[KSDMAEmergencyContact]:
        self.policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_json(
            self.config.contacts_endpoint,
            params=params,
        )

        records = self.parser.parse_contacts(
            payload
        )

        if district:
            selected = normalize_district(
                district
            )

            records = [
                record
                for record in records
                if record.district in (
                    None,
                    selected,
                )
            ]

        return self.deduplicator.contacts(
            records
        )

    async def get_district_snapshot(
        self,
        district: str,
    ) -> KSDMADistrictSnapshot:
        district = normalize_district(
            district
        )

        (
            alerts,
            bulletins,
            shelters,
            contacts,
        ) = await asyncio.gather(
            self.get_alerts(
                district=district
            ),
            self.get_bulletins(
                district=district
            ),
            self.get_shelters(
                district=district
            ),
            self.get_emergency_contacts(
                district=district
            ),
        )

        warnings = []

        if any(
            shelter.data_status
            != KSDMARecordStatus.ACTIVE
            for shelter in shelters
        ):
            warnings.append(
                "Some shelter records are stale "
                "or have invalid timestamps"
            )

        if any(
            contact.data_status
            != KSDMARecordStatus.ACTIVE
            for contact in contacts
        ):
            warnings.append(
                "Some emergency contacts require "
                "freshness verification"
            )

        if not self.config.source_verified:
            warnings.append(
                "Source has not been independently verified"
            )

        return KSDMADistrictSnapshot(
            district=district,
            alerts=tuple(alerts),
            bulletins=tuple(bulletins),
            shelters=tuple(shelters),
            contacts=tuple(contacts),
            retrieved_at=utc_now(),
            source_verified=self.config.source_verified,
            warnings=tuple(warnings),
        )

    async def get_kerala_snapshots(
        self,
        *,
        districts: Sequence[str] | None = None,
    ) -> list[KSDMADistrictSnapshot]:
        selected = (
            tuple(districts)
            if districts is not None
            else KERALA_DISTRICTS
        )

        if len(selected) > (
            self.config.max_batch_districts
        ):
            raise KSDMAValidationError(
                "District batch limit exceeded"
            )

        unique = tuple(
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
                    for district in unique
                )
            )
        )

    async def close(self) -> None:
        await self.http_client.close()

    async def __aenter__(
        self,
    ) -> "KSDMAAdapter":
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

class KSDMARAGAdapter:
    """
    Converts authenticated, active records
    into evidence for RAG ingestion.

    The output remains untrusted text until
    the downstream retrieval and grounding
    pipeline independently checks provenance.
    """

    @staticmethod
    def to_rag_records(
        snapshot: KSDMADistrictSnapshot,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        if not tenant_id.strip():
            raise KSDMAValidationError(
                "tenant_id required"
            )

        if not snapshot.source_verified:
            return []

        output = []

        for alert in snapshot.alerts:
            if (
                alert.verification
                != KSDMASourceVerification.VERIFIED
                or alert.status
                != KSDMARecordStatus.ACTIVE
            ):
                continue

            output.append({
                "id": (
                    f"ksdma:alert:{alert.alert_id}:"
                    f"{alert.issued_at.isoformat()}"
                ),
                "content": (
                    f"KSDMA alert record. "
                    f"District: {alert.district}. "
                    f"Hazard: {alert.hazard_type.value}. "
                    f"Severity: {alert.severity.value}. "
                    f"Title: {alert.title}. "
                    f"Message: {alert.message}. "
                    f"Valid until: "
                    f"{alert.valid_until.isoformat()}."
                ),
                "metadata": {
                    "tenant_id": tenant_id,
                    "source": "ksdma",
                    "source_uri": alert.source_uri,
                    "reference_url": alert.reference_url,
                    "district": alert.district,
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "hazard_type": alert.hazard_type.value,
                    "observed_at": (
                        alert.issued_at.isoformat()
                    ),
                    "valid_until": (
                        alert.valid_until.isoformat()
                    ),
                    "verification": "verified",
                    "requires_human_review": True,
                },
            })

        for bulletin in snapshot.bulletins:
            if (
                bulletin.verification
                != KSDMASourceVerification.VERIFIED
                or bulletin.status
                != KSDMARecordStatus.ACTIVE
            ):
                continue

            output.append({
                "id": (
                    f"ksdma:bulletin:"
                    f"{bulletin.bulletin_id}"
                ),
                "content": (
                    f"KSDMA bulletin. "
                    f"Title: {bulletin.title}. "
                    f"Summary: {bulletin.summary}. "
                    f"Published: "
                    f"{bulletin.published_at.isoformat()}."
                ),
                "metadata": {
                    "tenant_id": tenant_id,
                    "source": "ksdma",
                    "source_uri": bulletin.source_uri,
                    "document_url": bulletin.document_url,
                    "district": bulletin.district,
                    "observed_at": (
                        bulletin.published_at.isoformat()
                    ),
                    "verification": "verified",
                    "requires_human_review": True,
                },
            })

        return output


# ============================================================
# NOTIFICATION PREPARATION ADAPTER
# ============================================================

class KSDMANotificationAdapter:
    """
    Prepares internal review requests.

    Does not send email, SMS or public alerts.
    """

    @staticmethod
    def prepare_alert_reviews(
        alerts: Sequence[KSDMAOfficialAlert],
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        reviews = []

        for alert in alerts:
            if (
                alert.verification
                != KSDMASourceVerification.VERIFIED
                or alert.status
                != KSDMARecordStatus.ACTIVE
            ):
                continue

            reviews.append({
                "tenant_id": tenant_id,
                "alert_id": alert.alert_id,
                "district": alert.district,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "source_uri": alert.source_uri,
                "reference_url": alert.reference_url,
                "valid_until": (
                    alert.valid_until.isoformat()
                ),
                "review_status": "pending",
                "requires_human_approval": True,
                "automatically_publish": False,
                "automatically_dispatch": False,
            })

        return reviews


# ============================================================
# SHELTER SERVICE ADAPTER
# ============================================================

class KSDMAShelterServiceAdapter:
    """
    Produces advisory shelter records.

    Shelter availability must be confirmed
    by authorized shelter personnel before
    allocation or public recommendations.
    """

    @staticmethod
    def to_shelter_records(
        shelters: Sequence[KSDMAShelter],
    ) -> list[dict[str, Any]]:
        output = []

        for shelter in shelters:
            if (
                shelter.verification
                != KSDMASourceVerification.VERIFIED
                or shelter.data_status
                != KSDMARecordStatus.ACTIVE
            ):
                continue

            output.append({
                "shelter_id": shelter.shelter_id,
                "name": shelter.name,
                "district": shelter.district,
                "address": shelter.address,
                "status": shelter.status.value,
                "capacity": shelter.capacity,
                "occupancy": shelter.occupancy,
                "available_spaces": (
                    shelter.available_spaces
                ),
                "updated_at": (
                    shelter.updated_at.isoformat()
                ),
                "requires_live_confirmation": True,
                "allocation_authorized": False,
            })

        return output


# ============================================================
# EXPORTER
# ============================================================

class KSDMAResponseExporter:
    @staticmethod
    def to_json(
        snapshot: KSDMADistrictSnapshot,
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

def create_ksdma_adapter(
    config: KSDMAAdapterConfig,
    *,
    mapping: KSDMAFieldMapping | None = None,
) -> KSDMAAdapter:
    return KSDMAAdapter(
        config=config,
        mapping=mapping,
    )


# ============================================================
# FICTIONAL DEMONSTRATION
# ============================================================

def demo() -> None:
    """
    Synthetic parser test.

    No network requests.
    No real disaster alerts.
    """

    config = KSDMAAdapterConfig(
        base_url="https://example.invalid",
        require_verified_source=False,
        source_verified=False,
    )

    parser = KSDMAResponseParser(
        config,
        KSDMAFieldMapping(),
    )

    now = utc_now()

    payload = {
        "data": [
            {
                "alert_id": "DEMO-ALERT-001",
                "district": "Thrissur",
                "hazard_type": "flood",
                "severity": "moderate",
                "title": "Synthetic flood alert",
                "message": (
                    "Demonstration only. "
                    "Not an actual KSDMA warning."
                ),
                "issued_at": now.isoformat(),
                "valid_from": (
                    now - timedelta(minutes=10)
                ).isoformat(),
                "valid_until": (
                    now + timedelta(hours=2)
                ).isoformat(),
                "reference_url": None,
            }
        ]
    }

    alerts = parser.parse_alerts(
        payload
    )

    print(
        json.dumps(
            json_safe(alerts),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    demo()


__all__ = [
    "KSDMAAdapterError",
    "KSDMAConfigurationError",
    "KSDMAAuthenticationError",
    "KSDMARequestError",
    "KSDMAResponseError",
    "KSDMAValidationError",
    "KSDMASourceVerificationError",
    "KSDMADataUnavailableError",
    "KSDMAAuthenticationType",
    "KSDMAAlertSeverity",
    "KSDMAHazardType",
    "KSDMARecordStatus",
    "KSDMASourceVerification",
    "KSDMAShelterStatus",
    "KSDMARecordType",
    "KERALA_DISTRICTS",
    "KSDMAAdapterConfig",
    "KSDMAFieldMapping",
    "KSDMAOfficialAlert",
    "KSDMABulletin",
    "KSDMAShelter",
    "KSDMAEmergencyContact",
    "KSDMADistrictSnapshot",
    "KSDMASourcePolicy",
    "KSDMAHTTPClient",
    "KSDMAResponseParser",
    "KSDMARecordDeduplicator",
    "KSDMAAdapter",
    "KSDMARAGAdapter",
    "KSDMANotificationAdapter",
    "KSDMAShelterServiceAdapter",
    "KSDMAResponseExporter",
    "create_ksdma_adapter",
]
