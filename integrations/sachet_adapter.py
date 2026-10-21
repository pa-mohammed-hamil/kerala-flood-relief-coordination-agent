"""Placeholder module: integrations/sachet_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
SACHET Disaster Alert Integration Adapter

File:
    backend/integrations/sachet_adapter.py

Python:
    3.11+

Dependencies:
    pip install httpx

Capabilities:
    - Authorized SACHET alert ingestion
    - CAP-style disaster alert normalization
    - Flood, rainfall, landslide and cyclone alerts
    - Kerala district filtering
    - Multi-language alert preservation
    - Alert severity and urgency classification
    - Alert lifecycle and cancellation tracking
    - Alert validity and expiry checks
    - Duplicate detection
    - Source provenance metadata
    - Configurable authentication
    - HTTPS and request timeouts
    - Retry with exponential backoff
    - Response size limits
    - JSON response mapping
    - Human review queue preparation
    - RAG ingestion records
    - Structured audit events
    - Async operations
    - JSON export

IMPORTANT:
    SACHET integration endpoints, authentication,
    data contracts and access permissions must be
    obtained from an authorized provider.

    The endpoint paths and JSON structures here
    are configurable examples. They are NOT
    confirmed official SACHET API specifications.

    A configuration flag does not independently
    establish source authenticity.

    This adapter never automatically:
        - publishes disaster warnings
        - sends mass notifications
        - dispatches rescue teams
        - issues evacuation orders
        - changes infrastructure controls

    Emergency alerts must be authenticated and
    reviewed by authorized personnel.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
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

class SachetAdapterError(Exception):
    pass


class SachetConfigurationError(SachetAdapterError):
    pass


class SachetAuthenticationError(SachetAdapterError):
    pass


class SachetRequestError(SachetAdapterError):
    pass


class SachetResponseError(SachetAdapterError):
    pass


class SachetValidationError(SachetAdapterError):
    pass


class SachetAuthorizationError(SachetAdapterError):
    pass


class SachetSourceVerificationError(SachetAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SachetAuthType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"


class SachetSeverity(str, Enum):
    EXTREME = "extreme"
    SEVERE = "severe"
    MODERATE = "moderate"
    MINOR = "minor"
    UNKNOWN = "unknown"


class SachetUrgency(str, Enum):
    IMMEDIATE = "immediate"
    EXPECTED = "expected"
    FUTURE = "future"
    PAST = "past"
    UNKNOWN = "unknown"


class SachetCertainty(str, Enum):
    OBSERVED = "observed"
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNLIKELY = "unlikely"
    UNKNOWN = "unknown"


class SachetMessageType(str, Enum):
    ALERT = "alert"
    UPDATE = "update"
    CANCEL = "cancel"
    ACK = "ack"
    ERROR = "error"


class SachetStatus(str, Enum):
    ACTUAL = "actual"
    EXERCISE = "exercise"
    SYSTEM = "system"
    TEST = "test"
    DRAFT = "draft"


class SachetHazardType(str, Enum):
    FLOOD = "flood"
    HEAVY_RAIN = "heavy_rain"
    LANDSLIDE = "landslide"
    CYCLONE = "cyclone"
    COASTAL_FLOODING = "coastal_flooding"
    LIGHTNING = "lightning"
    STRONG_WIND = "strong_wind"
    TSUNAMI = "tsunami"
    EARTHQUAKE = "earthquake"
    FIRE = "fire"
    OTHER = "other"


class SachetLifecycle(str, Enum):
    ACTIVE = "active"
    UPCOMING = "upcoming"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    FUTURE_TIMESTAMP = "future_timestamp"
    UNKNOWN = "unknown"


class SachetSourceVerification(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


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
        raise SachetValidationError(
            "Expected string"
        )

    return value.strip()


def normalize_district(
    value: str,
) -> str:
    district = DISTRICT_LOOKUP.get(
        normalize_text(value).casefold()
    )

    if district is None:
        raise SachetValidationError(
            "Unsupported Kerala district"
        )

    return district


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
            raise SachetValidationError(
                "Invalid timestamp"
            ) from exc

    else:
        raise SachetValidationError(
            "Unsupported timestamp"
        )

    if (
        result.tzinfo is None
        or result.utcoffset() is None
    ):
        raise SachetValidationError(
            "Timestamp requires timezone"
        )

    return result.astimezone(
        timezone.utc
    )


def optional_timestamp(
    value: Any,
) -> datetime | None:
    if value is None or value == "":
        return None

    return parse_timestamp(value)


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


def stable_hash(
    *values: str,
) -> str:
    content = "|".join(values)

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SachetAdapterConfig:
    base_url: str

    alerts_endpoint: str = "/alerts"

    auth_type: SachetAuthType = (
        SachetAuthType.NONE
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

    max_alert_age_hours: int = 72

    source_verified: bool = False

    require_verified_source: bool = True

    allowed_hosts: tuple[str, ...] = ()

    allow_local_http_testing: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(
            self.base_url
        )

        if not parsed.hostname:
            raise SachetConfigurationError(
                "Valid base_url required"
            )

        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise SachetConfigurationError(
                "Invalid base_url components"
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
                raise SachetConfigurationError(
                    "HTTPS required"
                )

        if (
            self.allowed_hosts
            and parsed.hostname not in self.allowed_hosts
        ):
            raise SachetConfigurationError(
                "Host not allowlisted"
            )

        if (
            not self.alerts_endpoint.startswith("/")
            or self.alerts_endpoint.startswith("//")
            or "://" in self.alerts_endpoint
            or "?" in self.alerts_endpoint
            or "#" in self.alerts_endpoint
            or ".." in self.alerts_endpoint.split("/")
        ):
            raise SachetConfigurationError(
                "Invalid alerts endpoint"
            )

        if (
            self.auth_type == SachetAuthType.API_KEY
            and not self.api_key
        ):
            raise SachetConfigurationError(
                "API key required"
            )

        if (
            self.auth_type == SachetAuthType.BEARER
            and not self.bearer_token
        ):
            raise SachetConfigurationError(
                "Bearer token required"
            )

        if self.timeout_seconds <= 0:
            raise SachetConfigurationError(
                "Timeout must be positive"
            )

        if self.max_retries < 0:
            raise SachetConfigurationError(
                "Retries cannot be negative"
            )

        for name in (
            "max_response_bytes",
            "max_records",
            "max_batch_districts",
            "max_alert_age_hours",
        ):
            if getattr(self, name) <= 0:
                raise SachetConfigurationError(
                    f"{name} must be positive"
                )


# ============================================================
# JSON FIELD MAPPING
# ============================================================

@dataclass(frozen=True)
class SachetFieldMapping:
    alerts_list_path: str = "data"

    identifier: str = "identifier"
    sender: str = "sender"
    sent: str = "sent"

    status: str = "status"
    message_type: str = "msgType"

    references: str = "references"

    district: str = "district"

    event: str = "event"
    hazard_type: str = "hazard_type"

    severity: str = "severity"
    urgency: str = "urgency"
    certainty: str = "certainty"

    headline: str = "headline"
    description: str = "description"
    instruction: str = "instruction"

    language: str = "language"

    effective: str = "effective"
    expires: str = "expires"

    source_url: str = "source_url"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class SachetAlert:
    identifier: str
    sender: str

    sent_at: datetime

    district: str

    event: str

    hazard_type: SachetHazardType

    severity: SachetSeverity
    urgency: SachetUrgency
    certainty: SachetCertainty

    headline: str
    description: str
    instruction: str

    language: str

    effective_at: datetime
    expires_at: datetime

    message_type: SachetMessageType

    cap_status: SachetStatus

    references: tuple[str, ...]

    lifecycle: SachetLifecycle

    source_uri: str

    verification: SachetSourceVerification

    retrieved_at: datetime

    requires_human_review: bool = True

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def alert_key(self) -> str:
        return stable_hash(
            self.sender,
            self.identifier,
            self.sent_at.isoformat(),
        )


@dataclass(frozen=True)
class SachetAlertBatch:
    alerts: tuple[SachetAlert, ...]

    retrieved_at: datetime

    source_verified: bool

    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class SachetAuditEvent:
    event_id: str
    action: str
    timestamp: datetime

    record_count: int

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# SOURCE POLICY
# ============================================================

class SachetSourcePolicy:
    def __init__(
        self,
        config: SachetAdapterConfig,
    ) -> None:
        self.config = config

    @property
    def verification(
        self,
    ) -> SachetSourceVerification:
        return (
            SachetSourceVerification.VERIFIED
            if self.config.source_verified
            else SachetSourceVerification.UNVERIFIED
        )

    def require_verified(self) -> None:
        if (
            self.config.require_verified_source
            and not self.config.source_verified
        ):
            raise SachetSourceVerificationError(
                "SACHET source has not been "
                "independently verified"
            )


# ============================================================
# ALERT LIFECYCLE
# ============================================================

class SachetLifecycleEvaluator:
    def __init__(
        self,
        config: SachetAdapterConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        sent_at: datetime,
        effective_at: datetime,
        expires_at: datetime,
        message_type: SachetMessageType,
        cap_status: SachetStatus,
        now: datetime,
    ) -> SachetLifecycle:
        if cap_status != SachetStatus.ACTUAL:
            return SachetLifecycle.UNKNOWN

        if message_type == SachetMessageType.CANCEL:
            return SachetLifecycle.CANCELLED

        if sent_at > now:
            return SachetLifecycle.FUTURE_TIMESTAMP

        if now < effective_at:
            return SachetLifecycle.UPCOMING

        if now > expires_at:
            return SachetLifecycle.EXPIRED

        if now - sent_at > timedelta(
            hours=self.config.max_alert_age_hours
        ):
            return SachetLifecycle.UNKNOWN

        return SachetLifecycle.ACTIVE


# ============================================================
# HTTP CLIENT
# ============================================================

class SachetHTTPClient:
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        config: SachetAdapterConfig,
    ) -> None:
        if httpx is None:
            raise SachetConfigurationError(
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
            self.config.auth_type
            == SachetAuthType.API_KEY
        ):
            headers[
                self.config.api_key_header
            ] = self.config.api_key or ""

        elif (
            self.config.auth_type
            == SachetAuthType.BEARER
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

    async def get_alerts_json(
        self,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        url = (
            self.config.base_url.rstrip("/")
            + self.config.alerts_endpoint
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
                        raise SachetAuthenticationError(
                            "SACHET authentication failed"
                        )

                    if (
                        response.status_code
                        in self.RETRYABLE_STATUS_CODES
                    ):
                        raise SachetRequestError(
                            "Temporary upstream failure"
                        )

                    response.raise_for_status()

                    raw_length = response.headers.get(
                        "Content-Length"
                    )

                    if raw_length is not None:
                        try:
                            declared_size = int(
                                raw_length
                            )
                        except ValueError as exc:
                            raise SachetResponseError(
                                "Invalid Content-Length"
                            ) from exc

                        if declared_size > (
                            self.config.max_response_bytes
                        ):
                            raise SachetResponseError(
                                "Response too large"
                            )

                    chunks = []
                    total = 0

                    async for chunk in response.aiter_bytes():
                        total += len(chunk)

                        if total > (
                            self.config.max_response_bytes
                        ):
                            raise SachetResponseError(
                                "Response size limit exceeded"
                            )

                        chunks.append(chunk)

                return json.loads(
                    b"".join(chunks)
                )

            except (
                SachetAuthenticationError,
                SachetResponseError,
            ):
                raise

            except (
                SachetRequestError,
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
                raise SachetRequestError(
                    f"Upstream HTTP "
                    f"{exc.response.status_code}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise SachetResponseError(
                    "Invalid JSON response"
                ) from exc

        raise SachetRequestError(
            "Request failed after retries"
        ) from last_error

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ============================================================
# RESPONSE PARSER
# ============================================================

class SachetResponseParser:
    def __init__(
        self,
        config: SachetAdapterConfig,
        mapping: SachetFieldMapping,
    ) -> None:
        self.config = config
        self.mapping = mapping

        self.source_policy = SachetSourcePolicy(
            config
        )

        self.lifecycle_evaluator = (
            SachetLifecycleEvaluator(config)
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
                self.mapping.alerts_list_path,
            )

        else:
            raise SachetResponseError(
                "Invalid response root"
            )

        if not isinstance(records, list):
            raise SachetResponseError(
                "Expected alert list"
            )

        if len(records) > self.config.max_records:
            raise SachetResponseError(
                "Too many alerts"
            )

        if not all(
            isinstance(item, Mapping)
            for item in records
        ):
            raise SachetResponseError(
                "Invalid alert structure"
            )

        return records

    @staticmethod
    def _parse_references(
        raw: Any,
    ) -> tuple[str, ...]:
        if raw is None:
            return ()

        if isinstance(raw, str):
            return tuple(
                part.strip()
                for part in raw.split()
                if part.strip()
            )

        if isinstance(raw, list):
            return tuple(
                str(part).strip()
                for part in raw
                if str(part).strip()
            )

        raise SachetValidationError(
            "Invalid CAP references"
        )

    def parse(
        self,
        payload: Any,
        *,
        now: datetime | None = None,
    ) -> list[SachetAlert]:
        now = now or utc_now()
        m = self.mapping

        output = []

        for item in self._records(payload):
            identifier = nested_get(
                item,
                m.identifier,
            )

            sender = nested_get(
                item,
                m.sender,
            )

            district = nested_get(
                item,
                m.district,
            )

            if (
                not identifier
                or not sender
                or not district
            ):
                raise SachetResponseError(
                    "Missing identifier, sender "
                    "or district"
                )

            sent_at = parse_timestamp(
                nested_get(
                    item,
                    m.sent,
                )
            )

            effective_at = parse_timestamp(
                nested_get(
                    item,
                    m.effective,
                )
            )

            expires_at = parse_timestamp(
                nested_get(
                    item,
                    m.expires,
                )
            )

            if expires_at <= effective_at:
                raise SachetValidationError(
                    "Invalid alert validity period"
                )

            message_type = enum_or_default(
                SachetMessageType,
                nested_get(
                    item,
                    m.message_type,
                ),
                SachetMessageType.ERROR,
            )

            cap_status = enum_or_default(
                SachetStatus,
                nested_get(
                    item,
                    m.status,
                ),
                SachetStatus.DRAFT,
            )

            severity = enum_or_default(
                SachetSeverity,
                nested_get(
                    item,
                    m.severity,
                ),
                SachetSeverity.UNKNOWN,
            )

            urgency = enum_or_default(
                SachetUrgency,
                nested_get(
                    item,
                    m.urgency,
                ),
                SachetUrgency.UNKNOWN,
            )

            certainty = enum_or_default(
                SachetCertainty,
                nested_get(
                    item,
                    m.certainty,
                ),
                SachetCertainty.UNKNOWN,
            )

            hazard = enum_or_default(
                SachetHazardType,
                nested_get(
                    item,
                    m.hazard_type,
                ),
                SachetHazardType.OTHER,
            )

            lifecycle = (
                self.lifecycle_evaluator.evaluate(
                    sent_at=sent_at,
                    effective_at=effective_at,
                    expires_at=expires_at,
                    message_type=message_type,
                    cap_status=cap_status,
                    now=now,
                )
            )

            output.append(
                SachetAlert(
                    identifier=str(identifier),
                    sender=str(sender),
                    sent_at=sent_at,
                    district=normalize_district(
                        str(district)
                    ),
                    event=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.event,
                                "",
                            )
                        )
                    ),
                    hazard_type=hazard,
                    severity=severity,
                    urgency=urgency,
                    certainty=certainty,
                    headline=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.headline,
                                "",
                            )
                        )
                    ),
                    description=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.description,
                                "",
                            )
                        )
                    ),
                    instruction=normalize_text(
                        str(
                            nested_get(
                                item,
                                m.instruction,
                                "",
                            )
                        )
                    ),
                    language=str(
                        nested_get(
                            item,
                            m.language,
                            "en",
                        )
                    ),
                    effective_at=effective_at,
                    expires_at=expires_at,
                    message_type=message_type,
                    cap_status=cap_status,
                    references=self._parse_references(
                        nested_get(
                            item,
                            m.references,
                        )
                    ),
                    lifecycle=lifecycle,
                    source_uri=self.config.base_url,
                    verification=(
                        self.source_policy.verification
                    ),
                    retrieved_at=now,
                    requires_human_review=True,
                )
            )

        return output


# ============================================================
# DEDUPLICATION
# ============================================================

class SachetAlertDeduplicator:
    @staticmethod
    def deduplicate(
        alerts: Sequence[SachetAlert],
    ) -> list[SachetAlert]:
        unique = {}

        for alert in alerts:
            unique[
                alert.alert_key
            ] = alert

        return sorted(
            unique.values(),
            key=lambda item: (
                item.district,
                item.sent_at,
            ),
        )


# ============================================================
# ALERT LIFECYCLE STORE
# ============================================================

class SachetAlertLifecycleStore:
    """
    In-memory CAP-style lifecycle tracking.

    Production deployments should use
    a persistent database and authenticated
    alert references.

    Cancellation and update references
    must be validated against original
    sender and alert identity.
    """

    def __init__(self) -> None:
        self._alerts: dict[
            str,
            SachetAlert,
        ] = {}

        self._cancelled: set[str] = set()

        self._superseded: set[str] = set()

    def ingest(
        self,
        alerts: Sequence[SachetAlert],
    ) -> None:
        for alert in alerts:
            self._alerts[
                alert.alert_key
            ] = alert

        for alert in alerts:
            if (
                alert.verification
                != SachetSourceVerification.VERIFIED
                or alert.cap_status
                != SachetStatus.ACTUAL
            ):
                continue

            if alert.message_type not in (
                SachetMessageType.CANCEL,
                SachetMessageType.UPDATE,
            ):
                continue

            for reference in alert.references:
                for existing in self._alerts.values():
                    if existing.sender != alert.sender:
                        continue

                    reference_matches = (
                        reference == existing.identifier
                        or reference == (
                            f"{existing.sender},"
                            f"{existing.identifier},"
                            f"{existing.sent_at.isoformat()}"
                        )
                    )

                    if not reference_matches:
                        continue

                    if (
                        alert.sent_at
                        <= existing.sent_at
                    ):
                        continue

                    if (
                        alert.message_type
                        == SachetMessageType.CANCEL
                    ):
                        self._cancelled.add(
                            existing.alert_key
                        )

                    else:
                        self._superseded.add(
                            existing.alert_key
                        )

    def effective_lifecycle(
        self,
        alert: SachetAlert,
    ) -> SachetLifecycle:
        if alert.alert_key in self._cancelled:
            return SachetLifecycle.CANCELLED

        if alert.alert_key in self._superseded:
            return SachetLifecycle.SUPERSEDED

        return alert.lifecycle

    def active_alerts(
        self,
    ) -> list[SachetAlert]:
        return [
            alert
            for alert in self._alerts.values()
            if (
                alert.verification
                == SachetSourceVerification.VERIFIED
                and alert.message_type
                in (
                    SachetMessageType.ALERT,
                    SachetMessageType.UPDATE,
                )
                and self.effective_lifecycle(alert)
                == SachetLifecycle.ACTIVE
            )
        ]


# ============================================================
# MAIN SACHET ADAPTER
# ============================================================

class SachetAdapter:
    def __init__(
        self,
        config: SachetAdapterConfig,
        *,
        mapping: SachetFieldMapping | None = None,
        http_client: SachetHTTPClient | None = None,
    ) -> None:
        self.config = config

        self.mapping = (
            mapping or SachetFieldMapping()
        )

        self.http_client = (
            http_client
            or SachetHTTPClient(config)
        )

        self.parser = SachetResponseParser(
            config,
            self.mapping,
        )

        self.source_policy = SachetSourcePolicy(
            config
        )

        self.deduplicator = (
            SachetAlertDeduplicator()
        )

        self.lifecycle_store = (
            SachetAlertLifecycleStore()
        )

        self._audit: list[SachetAuditEvent] = []

    def _audit_event(
        self,
        action: str,
        record_count: int,
    ) -> None:
        self._audit.append(
            SachetAuditEvent(
                event_id=generate_id(),
                action=action,
                timestamp=utc_now(),
                record_count=record_count,
            )
        )

        if len(self._audit) > 1000:
            self._audit = self._audit[-1000:]

    async def get_alerts(
        self,
        *,
        district: str | None = None,
        active_only: bool = True,
    ) -> list[SachetAlert]:
        self.source_policy.require_verified()

        params = {}

        if district:
            params["district"] = normalize_district(
                district
            )

        payload = await self.http_client.get_alerts_json(
            params=params
        )

        alerts = self.parser.parse(
            payload
        )

        if district:
            selected = normalize_district(
                district
            )

            alerts = [
                alert
                for alert in alerts
                if alert.district == selected
            ]

        alerts = self.deduplicator.deduplicate(
            alerts
        )

        self.lifecycle_store.ingest(
            alerts
        )

        if active_only:
            alerts = [
                alert
                for alert in alerts
                if (
                    alert.verification
                    == SachetSourceVerification.VERIFIED
                    and alert.message_type
                    in (
                        SachetMessageType.ALERT,
                        SachetMessageType.UPDATE,
                    )
                    and (
                        self.lifecycle_store
                        .effective_lifecycle(alert)
                        == SachetLifecycle.ACTIVE
                    )
                )
            ]

        self._audit_event(
            "get_alerts",
            len(alerts),
        )

        return alerts

    async def get_kerala_alerts(
        self,
        *,
        active_only: bool = True,
    ) -> list[SachetAlert]:
        alerts = await self.get_alerts(
            active_only=active_only
        )

        return [
            alert
            for alert in alerts
            if alert.district in KERALA_DISTRICTS
        ]

    async def get_district_alerts(
        self,
        districts: Sequence[str],
        *,
        active_only: bool = True,
    ) -> list[SachetAlert]:
        if len(districts) > (
            self.config.max_batch_districts
        ):
            raise SachetValidationError(
                "District batch limit exceeded"
            )

        selected = tuple(
            dict.fromkeys(
                normalize_district(district)
                for district in districts
            )
        )

        # Fetch once so lifecycle updates/cancellations
        # can be processed across district records.
        alerts = await self.get_alerts(
            active_only=active_only
        )

        return [
            alert
            for alert in alerts
            if alert.district in selected
        ]

    async def get_alert_batch(
        self,
        *,
        district: str | None = None,
    ) -> SachetAlertBatch:
        alerts = await self.get_alerts(
            district=district,
            active_only=True,
        )

        warnings = []

        if not self.config.source_verified:
            warnings.append(
                "Source is not independently verified"
            )

        return SachetAlertBatch(
            alerts=tuple(alerts),
            retrieved_at=utc_now(),
            source_verified=(
                self.config.source_verified
            ),
            warnings=tuple(warnings),
        )

    def audit_events(
        self,
    ) -> list[SachetAuditEvent]:
        return list(
            self._audit
        )

    async def close(self) -> None:
        await self.http_client.close()

    async def __aenter__(
        self,
    ) -> "SachetAdapter":
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

class SachetRAGAdapter:
    @staticmethod
    def to_rag_records(
        alerts: Sequence[SachetAlert],
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        if not tenant_id.strip():
            raise SachetValidationError(
                "tenant_id required"
            )

        output = []

        for alert in alerts:
            if (
                alert.verification
                != SachetSourceVerification.VERIFIED
                or alert.lifecycle
                != SachetLifecycle.ACTIVE
                or alert.cap_status
                != SachetStatus.ACTUAL
                or alert.message_type
                not in (
                    SachetMessageType.ALERT,
                    SachetMessageType.UPDATE,
                )
            ):
                continue

            output.append({
                "id": (
                    f"sachet:{alert.alert_key}"
                ),
                "content": (
                    f"SACHET disaster alert. "
                    f"District: {alert.district}. "
                    f"Event: {alert.event}. "
                    f"Severity: {alert.severity.value}. "
                    f"Urgency: {alert.urgency.value}. "
                    f"Headline: {alert.headline}. "
                    f"Description: {alert.description}. "
                    f"Official instructions: "
                    f"{alert.instruction}. "
                    f"Expires: "
                    f"{alert.expires_at.isoformat()}."
                ),
                "metadata": {
                    "tenant_id": tenant_id,
                    "source": "sachet",
                    "source_uri": alert.source_uri,
                    "district": alert.district,
                    "identifier": alert.identifier,
                    "sender": alert.sender,
                    "severity": alert.severity.value,
                    "urgency": alert.urgency.value,
                    "language": alert.language,
                    "observed_at": (
                        alert.sent_at.isoformat()
                    ),
                    "expires_at": (
                        alert.expires_at.isoformat()
                    ),
                    "verification": "verified",
                    "requires_human_review": True,
                },
            })

        return output


# ============================================================
# NOTIFICATION REVIEW ADAPTER
# ============================================================

class SachetNotificationAdapter:
    """
    Creates internal approval tasks.

    Does not publish or send notifications.
    """

    @staticmethod
    def prepare_reviews(
        alerts: Sequence[SachetAlert],
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        output = []

        for alert in alerts:
            if (
                alert.verification
                != SachetSourceVerification.VERIFIED
                or alert.lifecycle
                != SachetLifecycle.ACTIVE
                or alert.cap_status
                != SachetStatus.ACTUAL
            ):
                continue

            output.append({
                "review_id": generate_id(),
                "tenant_id": tenant_id,
                "source": "sachet",
                "alert_identifier": alert.identifier,
                "district": alert.district,
                "headline": alert.headline,
                "description": alert.description,
                "instruction": alert.instruction,
                "severity": alert.severity.value,
                "urgency": alert.urgency.value,
                "expires_at": (
                    alert.expires_at.isoformat()
                ),
                "review_status": "pending",
                "requires_human_approval": True,
                "automatically_publish": False,
                "automatically_dispatch": False,
            })

        return output


# ============================================================
# EXPORTER
# ============================================================

class SachetResponseExporter:
    @staticmethod
    def to_json(
        batch: SachetAlertBatch,
    ) -> str:
        return json.dumps(
            batch.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


# ============================================================
# FACTORY
# ============================================================

def create_sachet_adapter(
    config: SachetAdapterConfig,
    *,
    mapping: SachetFieldMapping | None = None,
) -> SachetAdapter:
    return SachetAdapter(
        config=config,
        mapping=mapping,
    )


# ============================================================
# SYNTHETIC DEMONSTRATION
# ============================================================

def demo() -> None:
    """
    Synthetic example only.

    No network calls.
    No real disaster alerts.
    """

    config = SachetAdapterConfig(
        base_url="https://example.invalid",
        require_verified_source=False,
        source_verified=False,
    )

    parser = SachetResponseParser(
        config,
        SachetFieldMapping(),
    )

    now = utc_now()

    payload = {
        "data": [
            {
                "identifier": "DEMO-SACHET-001",
                "sender": "demo.invalid",
                "sent": now.isoformat(),
                "status": "Actual",
                "msgType": "Alert",
                "district": "Thrissur",
                "event": "Synthetic Flood Alert",
                "hazard_type": "flood",
                "severity": "Moderate",
                "urgency": "Expected",
                "certainty": "Possible",
                "headline": (
                    "Demonstration alert"
                ),
                "description": (
                    "This is fictional test data."
                ),
                "instruction": (
                    "No operational action."
                ),
                "language": "en",
                "effective": (
                    now - timedelta(minutes=5)
                ).isoformat(),
                "expires": (
                    now + timedelta(hours=2)
                ).isoformat(),
                "references": [],
            }
        ]
    }

    alerts = parser.parse(
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
    "SachetAdapterError",
    "SachetConfigurationError",
    "SachetAuthenticationError",
    "SachetRequestError",
    "SachetResponseError",
    "SachetValidationError",
    "SachetAuthorizationError",
    "SachetSourceVerificationError",
    "SachetAuthType",
    "SachetSeverity",
    "SachetUrgency",
    "SachetCertainty",
    "SachetMessageType",
    "SachetStatus",
    "SachetHazardType",
    "SachetLifecycle",
    "SachetSourceVerification",
    "KERALA_DISTRICTS",
    "SachetAdapterConfig",
    "SachetFieldMapping",
    "SachetAlert",
    "SachetAlertBatch",
    "SachetAuditEvent",
    "SachetSourcePolicy",
    "SachetLifecycleEvaluator",
    "SachetHTTPClient",
    "SachetResponseParser",
    "SachetAlertDeduplicator",
    "SachetAlertLifecycleStore",
    "SachetAdapter",
    "SachetRAGAdapter",
    "SachetNotificationAdapter",
    "SachetResponseExporter",
    "create_sachet_adapter",
]
