"""Placeholder module: integrations/sms_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
SMS Integration Adapter

File: backend/integrations/sms_adapter.py
Python: 3.11+

Dependencies:
    pip install httpx

Features:
    - Pluggable SMS provider
    - Configurable REST API integration
    - SMS message templates
    - English and Malayalam Unicode
    - Indian phone number validation
    - SOS acknowledgment messages
    - Rescue coordination messages
    - Shelter information messages
    - Relief delivery notifications
    - Medical coordination messages
    - Human-approved emergency alerts
    - Idempotency protection
    - Retry and timeout handling
    - Provider delivery status
    - Webhook signature verification
    - Message audit logging
    - PII masking
    - Tenant isolation
    - Consent and purpose validation
    - Rate limiting
    - Dry-run mode
    - Async operations

IMPORTANT:
    No provider is configured by default.

    The REST request and response mappings are
    generic examples, not specifications for
    Twilio, MSG91, AWS SNS or another provider.

    For Indian production SMS delivery, configure
    applicable DLT registration, approved headers,
    templates and consent requirements.

    Emergency warnings and operational dispatch
    require appropriate human authorization.

    This adapter never sends messages automatically
    when an SMS request is merely constructed.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol
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

class SMSAdapterError(Exception):
    pass


class SMSConfigurationError(SMSAdapterError):
    pass


class SMSValidationError(SMSAdapterError):
    pass


class SMSAuthorizationError(SMSAdapterError):
    pass


class SMSProviderError(SMSAdapterError):
    pass


class SMSRateLimitError(SMSAdapterError):
    pass


class SMSWebhookError(SMSAdapterError):
    pass


class SMSIdempotencyError(SMSAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SMSPurpose(str, Enum):
    SOS_ACKNOWLEDGMENT = "sos_acknowledgment"
    RESCUE_COORDINATION = "rescue_coordination"
    SHELTER_INFORMATION = "shelter_information"
    RELIEF_DELIVERY = "relief_delivery"
    MEDICAL_COORDINATION = "medical_coordination"
    VOLUNTEER_COORDINATION = "volunteer_coordination"
    EMERGENCY_WARNING = "emergency_warning"
    GENERAL_NOTIFICATION = "general_notification"


class SMSPriority(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SMSLanguage(str, Enum):
    ENGLISH = "en"
    MALAYALAM = "ml"


class SMSStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    QUEUED = "queued"
    SENDING = "sending"
    ACCEPTED = "accepted"
    DELIVERED = "delivered"
    FAILED = "failed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    DRY_RUN = "dry_run"


class SMSApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def mask_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)

    if len(digits) < 4:
        return "****"

    return "*" * max(
        0,
        len(digits) - 4,
    ) + digits[-4:]


def normalize_indian_phone(
    value: str,
) -> str:
    if not isinstance(value, str):
        raise SMSValidationError(
            "Phone number must be a string"
        )

    normalized = re.sub(
        r"[\s\-()]",
        "",
        value.strip(),
    )

    if normalized.startswith("0091"):
        normalized = "+91" + normalized[4:]

    elif normalized.startswith("91") and len(
        normalized
    ) == 12:
        normalized = "+" + normalized

    elif re.fullmatch(r"[6-9]\d{9}", normalized):
        normalized = "+91" + normalized

    if not re.fullmatch(
        r"\+91[6-9]\d{9}",
        normalized,
    ):
        raise SMSValidationError(
            "Expected valid Indian mobile number"
        )

    return normalized


def normalize_text(
    value: str,
    *,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise SMSValidationError(
            "Message must be text"
        )

    result = value.strip()

    if not result:
        raise SMSValidationError(
            "Message cannot be empty"
        )

    if len(result) > max_length:
        raise SMSValidationError(
            "Message exceeds length limit"
        )

    if any(
        ord(char) < 32
        and char not in "\n\t"
        for char in result
    ):
        raise SMSValidationError(
            "Message contains control characters"
        )

    return result


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SMSAdapterConfig:
    provider_name: str = "generic"

    api_base_url: str | None = None

    send_endpoint: str = "/messages"

    api_token: str | None = None

    sender_id: str | None = None

    timeout_seconds: float = 15.0

    max_retries: int = 2

    retry_base_seconds: float = 0.5

    max_message_length: int = 1000

    max_recipients_per_request: int = 1

    max_messages_per_minute: int = 30

    max_response_bytes: int = 100_000

    dry_run: bool = True

    allowed_hosts: tuple[str, ...] = ()

    allow_local_http_testing: bool = False

    webhook_secret: str | None = None

    webhook_max_age_seconds: int = 300

    require_approval_for_all: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise SMSConfigurationError(
                "Timeout must be positive"
            )

        if self.max_retries < 0:
            raise SMSConfigurationError(
                "Invalid retry count"
            )

        for name in (
            "max_message_length",
            "max_messages_per_minute",
            "max_response_bytes",
            "webhook_max_age_seconds",
        ):
            if getattr(self, name) <= 0:
                raise SMSConfigurationError(
                    f"{name} must be positive"
                )

        if self.max_recipients_per_request != 1:
            raise SMSConfigurationError(
                "This adapter supports one recipient "
                "per request"
            )

        if not self.dry_run:
            if not self.api_base_url:
                raise SMSConfigurationError(
                    "SMS provider URL required"
                )

            if not self.api_token:
                raise SMSConfigurationError(
                    "SMS provider token required"
                )

        if self.api_base_url:
            parsed = urlparse(
                self.api_base_url
            )

            if not parsed.hostname:
                raise SMSConfigurationError(
                    "Invalid provider URL"
                )

            if (
                parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise SMSConfigurationError(
                    "Invalid provider URL components"
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
                    raise SMSConfigurationError(
                        "HTTPS required"
                    )

            if (
                self.allowed_hosts
                and parsed.hostname
                not in self.allowed_hosts
            ):
                raise SMSConfigurationError(
                    "Provider host not allowlisted"
                )

        if (
            not self.send_endpoint.startswith("/")
            or self.send_endpoint.startswith("//")
            or "://" in self.send_endpoint
            or "?" in self.send_endpoint
            or "#" in self.send_endpoint
            or ".." in self.send_endpoint.split("/")
        ):
            raise SMSConfigurationError(
                "Invalid send endpoint"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class SMSPrincipal:
    user_id: str
    tenant_id: str

    roles: frozenset[str]

    def __post_init__(self) -> None:
        if (
            not self.user_id.strip()
            or not self.tenant_id.strip()
        ):
            raise SMSValidationError(
                "Principal identity required"
            )


@dataclass(frozen=True)
class SMSApproval:
    approval_id: str

    request_id: str

    tenant_id: str

    approved_by: str

    approved_at: datetime

    status: SMSApprovalStatus

    reason: str = ""


@dataclass(frozen=True)
class SMSRequest:
    recipient: str

    message: str

    purpose: SMSPurpose

    tenant_id: str

    idempotency_key: str

    priority: SMSPriority = SMSPriority.NORMAL

    language: SMSLanguage = SMSLanguage.ENGLISH

    consent_confirmed: bool = False

    lawful_emergency_basis_confirmed: bool = False

    reference_id: str | None = None

    request_id: str = field(
        default_factory=new_id
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class SMSResult:
    request_id: str

    tenant_id: str

    provider_message_id: str | None

    status: SMSStatus

    recipient_masked: str

    purpose: SMSPurpose

    submitted_at: datetime

    error_code: str | None = None

    approval_id: str | None = None

    dry_run: bool = False


@dataclass(frozen=True)
class SMSDeliveryEvent:
    event_id: str

    provider_message_id: str

    status: SMSStatus

    received_at: datetime

    provider_timestamp: datetime | None = None


@dataclass(frozen=True)
class SMSAuditEvent:
    audit_id: str

    request_id: str

    tenant_id: str

    action: str

    status: str

    timestamp: datetime

    recipient_masked: str


# ============================================================
# AUTHORIZATION
# ============================================================

class SMSAuthorizationPolicy:
    ALLOWED_ROLES = frozenset({
        "admin",
        "incident_commander",
        "emergency_operator",
        "communications_officer",
        "rescue_coordinator",
    })

    CRITICAL_PURPOSES = frozenset({
        SMSPurpose.EMERGENCY_WARNING,
        SMSPurpose.RESCUE_COORDINATION,
        SMSPurpose.MEDICAL_COORDINATION,
    })

    def __init__(
        self,
        config: SMSAdapterConfig,
    ) -> None:
        self.config = config

    def authorize(
        self,
        request: SMSRequest,
        principal: SMSPrincipal,
        approval: SMSApproval | None,
    ) -> None:
        if request.tenant_id != principal.tenant_id:
            raise SMSAuthorizationError(
                "Tenant mismatch"
            )

        if not (
            principal.roles
            & self.ALLOWED_ROLES
        ):
            raise SMSAuthorizationError(
                "Insufficient permissions"
            )

        if not (
            request.consent_confirmed
            or request.lawful_emergency_basis_confirmed
        ):
            raise SMSAuthorizationError(
                "Consent or lawful emergency "
                "basis required"
            )

        approval_required = (
            self.config.require_approval_for_all
            or request.purpose
            in self.CRITICAL_PURPOSES
        )

        if approval_required:
            if approval is None:
                raise SMSAuthorizationError(
                    "Human approval required"
                )

            if (
                approval.status
                != SMSApprovalStatus.APPROVED
                or approval.request_id
                != request.request_id
                or approval.tenant_id
                != request.tenant_id
                or approval.approved_by
                == principal.user_id
            ):
                raise SMSAuthorizationError(
                    "Invalid independent approval"
                )


# ============================================================
# MESSAGE TEMPLATES
# ============================================================

class SMSTemplateEngine:
    """
    These are draft templates.

    Production text should be reviewed,
    translated and DLT-approved as needed.
    """

    TEMPLATES: dict[
        tuple[SMSPurpose, SMSLanguage],
        str,
    ] = {
        (
            SMSPurpose.SOS_ACKNOWLEDGMENT,
            SMSLanguage.ENGLISH,
        ): (
            "Your SOS request {reference_id} has been "
            "received for review. This does not "
            "confirm rescue dispatch. If in immediate "
            "danger, contact official emergency services."
        ),

        (
            SMSPurpose.RESCUE_COORDINATION,
            SMSLanguage.ENGLISH,
        ): (
            "Rescue coordination update for "
            "{reference_id}: {update}. "
            "Follow instructions from authorized "
            "emergency personnel."
        ),

        (
            SMSPurpose.SHELTER_INFORMATION,
            SMSLanguage.ENGLISH,
        ): (
            "Shelter information for {reference_id}: "
            "{update}. Confirm availability with "
            "the authorized shelter team."
        ),

        (
            SMSPurpose.RELIEF_DELIVERY,
            SMSLanguage.ENGLISH,
        ): (
            "Relief delivery update for "
            "{reference_id}: {update}."
        ),

        (
            SMSPurpose.MEDICAL_COORDINATION,
            SMSLanguage.ENGLISH,
        ): (
            "Medical coordination update for "
            "{reference_id}: {update}. "
            "For emergencies, contact official "
            "emergency services."
        ),

        (
            SMSPurpose.VOLUNTEER_COORDINATION,
            SMSLanguage.ENGLISH,
        ): (
            "Volunteer assignment update "
            "{reference_id}: {update}. "
            "Proceed only after authorization."
        ),

        (
            SMSPurpose.EMERGENCY_WARNING,
            SMSLanguage.ENGLISH,
        ): (
            "AUTHORIZED EMERGENCY ALERT: "
            "{update}. Follow verified instructions "
            "from the responsible authorities."
        ),

        (
            SMSPurpose.GENERAL_NOTIFICATION,
            SMSLanguage.ENGLISH,
        ): (
            "Kerala Flood Relief update "
            "{reference_id}: {update}."
        ),

        (
            SMSPurpose.SOS_ACKNOWLEDGMENT,
            SMSLanguage.MALAYALAM,
        ): (
            "നിങ്ങളുടെ SOS അപേക്ഷ {reference_id} "
            "പരിശോധനയ്ക്കായി ലഭിച്ചിട്ടുണ്ട്. "
            "രക്ഷാപ്രവർത്തനം ആരംഭിച്ചതായി "
            "ഇത് സ്ഥിരീകരിക്കുന്നില്ല."
        ),

        (
            SMSPurpose.SHELTER_INFORMATION,
            SMSLanguage.MALAYALAM,
        ): (
            "ദുരിതാശ്വാസ ക്യാമ്പ് വിവരം "
            "{reference_id}: {update}. "
            "ലഭ്യത അധികൃതരുമായി സ്ഥിരീകരിക്കുക."
        ),
    }

    @classmethod
    def render(
        cls,
        purpose: SMSPurpose,
        language: SMSLanguage,
        values: Mapping[str, str],
    ) -> str:
        template = cls.TEMPLATES.get(
            (purpose, language)
        )

        if template is None:
            raise SMSValidationError(
                "No approved draft template "
                "for this purpose and language"
            )

        try:
            return template.format_map(
                dict(values)
            )

        except KeyError as exc:
            raise SMSValidationError(
                f"Missing template field: {exc}"
            ) from exc


# ============================================================
# RATE LIMITER
# ============================================================

class SMSRateLimiter:
    def __init__(
        self,
        max_per_minute: int,
    ) -> None:
        self.max_per_minute = max_per_minute

        self._timestamps: dict[
            str,
            list[float],
        ] = {}

        self._lock = asyncio.Lock()

    async def acquire(
        self,
        tenant_id: str,
    ) -> None:
        async with self._lock:
            now = time.monotonic()

            history = [
                value
                for value in self._timestamps.get(
                    tenant_id,
                    [],
                )
                if now - value < 60
            ]

            if len(history) >= self.max_per_minute:
                raise SMSRateLimitError(
                    "SMS rate limit exceeded"
                )

            history.append(now)

            self._timestamps[tenant_id] = history


# ============================================================
# IDEMPOTENCY STORE
# ============================================================

class SMSIdempotencyStore:
    """
    In-memory duplicate protection.

    Production requires a persistent,
    transactional idempotency database.
    """

    def __init__(self) -> None:
        self._results: dict[
            tuple[str, str],
            tuple[str, SMSResult],
        ] = {}

        self._lock = asyncio.Lock()

    async def get(
        self,
        tenant_id: str,
        key: str,
        fingerprint: str,
    ) -> SMSResult | None:
        async with self._lock:
            entry = self._results.get(
                (tenant_id, key)
            )

            if entry is None:
                return None

            stored_fingerprint, result = entry

            if stored_fingerprint != fingerprint:
                raise SMSIdempotencyError(
                    "Idempotency key reused "
                    "with different content"
                )

            return result

    async def save(
        self,
        tenant_id: str,
        key: str,
        fingerprint: str,
        result: SMSResult,
    ) -> None:
        async with self._lock:
            self._results[
                (tenant_id, key)
            ] = (
                fingerprint,
                result,
            )


# ============================================================
# PROVIDER PROTOCOL
# ============================================================

class SMSProvider(Protocol):
    async def send(
        self,
        *,
        recipient: str,
        message: str,
        idempotency_key: str,
    ) -> tuple[str, SMSStatus]:
        ...

    async def close(self) -> None:
        ...


# ============================================================
# DRY-RUN PROVIDER
# ============================================================

class DryRunSMSProvider:
    async def send(
        self,
        *,
        recipient: str,
        message: str,
        idempotency_key: str,
    ) -> tuple[str, SMSStatus]:
        return (
            "dry-run-" + new_id(),
            SMSStatus.DRY_RUN,
        )

    async def close(self) -> None:
        return None


# ============================================================
# GENERIC REST PROVIDER
# ============================================================

class GenericRESTSMSProvider:
    """
    Generic provider implementation.

    Configure request/response mapping to
    match your actual authorized provider.

    The response must contain a stable
    provider message ID and accepted status.
    """

    def __init__(
        self,
        config: SMSAdapterConfig,
    ) -> None:
        if httpx is None:
            raise SMSConfigurationError(
                "Install httpx: pip install httpx"
            )

        if not config.api_base_url:
            raise SMSConfigurationError(
                "Provider URL required"
            )

        self.config = config

        self._client = httpx.AsyncClient(
            timeout=config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    async def send(
        self,
        *,
        recipient: str,
        message: str,
        idempotency_key: str,
    ) -> tuple[str, SMSStatus]:
        url = (
            self.config.api_base_url.rstrip("/")
            + self.config.send_endpoint
        )

        headers = {
            "Authorization": (
                f"Bearer {self.config.api_token}"
            ),
            "Idempotency-Key": idempotency_key,
            "Accept": "application/json",
        }

        payload = {
            "to": recipient,
            "message": message,
            "sender_id": self.config.sender_id,
        }

        last_error: Exception | None = None

        for attempt in range(
            self.config.max_retries + 1
        ):
            try:
                async with self._client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code in (401, 403):
                        raise SMSAuthorizationError(
                            "Provider authentication failed"
                        )

                    if response.status_code >= 400:
                        # Do not retry ambiguous provider
                        # failures unless documented safe.
                        raise SMSProviderError(
                            f"Provider HTTP "
                            f"{response.status_code}"
                        )

                    chunks = []
                    total = 0

                    async for chunk in response.aiter_bytes():
                        total += len(chunk)

                        if total > (
                            self.config.max_response_bytes
                        ):
                            raise SMSProviderError(
                                "Provider response too large"
                            )

                        chunks.append(chunk)

                data = json.loads(
                    b"".join(chunks)
                )

                if not isinstance(data, Mapping):
                    raise SMSProviderError(
                        "Invalid provider response"
                    )

                message_id = data.get(
                    "message_id"
                )

                if not isinstance(
                    message_id,
                    str,
                ) or not message_id.strip():
                    raise SMSProviderError(
                        "Missing provider message ID"
                    )

                status = str(
                    data.get("status", "")
                ).casefold()

                if status not in (
                    "accepted",
                    "queued",
                ):
                    raise SMSProviderError(
                        "Provider did not confirm acceptance"
                    )

                return (
                    message_id,
                    (
                        SMSStatus.ACCEPTED
                        if status == "accepted"
                        else SMSStatus.QUEUED
                    ),
                )

            except (
                SMSProviderError,
                SMSAuthorizationError,
            ):
                raise

            except (
                httpx.TimeoutException,
                httpx.TransportError,
            ) as exc:
                last_error = exc

                # Retries can cause duplicate SMS if
                # provider idempotency is unsupported.
                if attempt >= self.config.max_retries:
                    break

                delay = (
                    self.config.retry_base_seconds
                    * (2 ** attempt)
                )

                await asyncio.sleep(delay)

            except json.JSONDecodeError as exc:
                raise SMSProviderError(
                    "Invalid provider JSON"
                ) from exc

        raise SMSProviderError(
            "SMS submission outcome unknown; "
            "reconcile provider state before retry"
        ) from last_error

    async def close(self) -> None:
        await self._client.aclose()


# ============================================================
# WEBHOOK VERIFIER
# ============================================================

class SMSWebhookVerifier:
    """
    Example HMAC-SHA256 webhook verification.

    Use only if the actual provider documents
    this exact signature format:

        HMAC_SHA256(
            secret,
            timestamp + "." + raw_body
        )

    Otherwise replace with the provider's
    official signature verification algorithm.
    """

    def __init__(
        self,
        secret: str,
        *,
        max_age_seconds: int = 300,
    ) -> None:
        if not secret:
            raise SMSConfigurationError(
                "Webhook secret required"
            )

        self.secret = secret.encode(
            "utf-8"
        )

        self.max_age_seconds = (
            max_age_seconds
        )

    def verify(
        self,
        *,
        body: bytes,
        timestamp: str,
        signature: str,
    ) -> None:
        try:
            sent_at = int(timestamp)
        except ValueError as exc:
            raise SMSWebhookError(
                "Invalid webhook timestamp"
            ) from exc

        now = int(
            utc_now().timestamp()
        )

        if abs(now - sent_at) > self.max_age_seconds:
            raise SMSWebhookError(
                "Webhook timestamp expired"
            )

        signed_content = (
            timestamp.encode("utf-8")
            + b"."
            + body
        )

        expected = hmac.new(
            self.secret,
            signed_content,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(
            expected,
            signature.strip().lower(),
        ):
            raise SMSWebhookError(
                "Invalid webhook signature"
            )


# ============================================================
# DELIVERY EVENT PARSER
# ============================================================

class SMSDeliveryEventParser:
    ALLOWED_STATUSES = {
        "queued": SMSStatus.QUEUED,
        "accepted": SMSStatus.ACCEPTED,
        "delivered": SMSStatus.DELIVERED,
        "failed": SMSStatus.FAILED,
        "rejected": SMSStatus.REJECTED,
    }

    @classmethod
    def parse(
        cls,
        payload: Mapping[str, Any],
    ) -> SMSDeliveryEvent:
        message_id = payload.get(
            "message_id"
        )

        status_value = str(
            payload.get("status", "")
        ).casefold()

        if not isinstance(
            message_id,
            str,
        ) or not message_id.strip():
            raise SMSWebhookError(
                "Missing provider message ID"
            )

        status = cls.ALLOWED_STATUSES.get(
            status_value
        )

        if status is None:
            raise SMSWebhookError(
                "Unknown delivery status"
            )

        return SMSDeliveryEvent(
            event_id=str(
                payload.get(
                    "event_id",
                    new_id(),
                )
            ),
            provider_message_id=message_id,
            status=status,
            received_at=utc_now(),
        )


# ============================================================
# SMS AUDIT LOGGER
# ============================================================

class SMSAuditLogger:
    def __init__(
        self,
        max_events: int = 5000,
    ) -> None:
        self.max_events = max_events

        self._events: list[
            SMSAuditEvent
        ] = []

    def record(
        self,
        request: SMSRequest,
        *,
        action: str,
        status: str,
    ) -> None:
        self._events.append(
            SMSAuditEvent(
                audit_id=new_id(),
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                action=action,
                status=status,
                timestamp=utc_now(),
                recipient_masked=mask_phone(
                    request.recipient
                ),
            )
        )

        if len(self._events) > self.max_events:
            self._events = self._events[
                -self.max_events:
            ]

    def list_events(
        self,
        tenant_id: str,
    ) -> list[SMSAuditEvent]:
        return [
            event
            for event in self._events
            if event.tenant_id == tenant_id
        ]


# ============================================================
# MAIN SMS ADAPTER
# ============================================================

class SMSAdapter:
    def __init__(
        self,
        config: SMSAdapterConfig,
        *,
        provider: SMSProvider | None = None,
    ) -> None:
        self.config = config

        if provider is not None:
            self.provider = provider

        elif config.dry_run:
            self.provider = DryRunSMSProvider()

        else:
            self.provider = GenericRESTSMSProvider(
                config
            )

        self.authorization = (
            SMSAuthorizationPolicy(config)
        )

        self.rate_limiter = SMSRateLimiter(
            config.max_messages_per_minute
        )

        self.idempotency = SMSIdempotencyStore()

        self.audit = SMSAuditLogger()

        # Prevent concurrent duplicate submissions
        # within this adapter instance.
        self._send_lock = asyncio.Lock()

    @staticmethod
    def _fingerprint(
        request: SMSRequest,
        normalized_recipient: str,
    ) -> str:
        payload = {
            "recipient": normalized_recipient,
            "message": request.message,
            "purpose": request.purpose.value,
            "tenant_id": request.tenant_id,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    async def send(
        self,
        request: SMSRequest,
        *,
        principal: SMSPrincipal,
        approval: SMSApproval | None = None,
    ) -> SMSResult:
        recipient = normalize_indian_phone(
            request.recipient
        )

        message = normalize_text(
            request.message,
            max_length=(
                self.config.max_message_length
            ),
        )

        if not request.idempotency_key.strip():
            raise SMSValidationError(
                "Idempotency key required"
            )

        normalized_request = replace(
            request,
            recipient=recipient,
            message=message,
        )

        self.authorization.authorize(
            normalized_request,
            principal,
            approval,
        )

        fingerprint = self._fingerprint(
            normalized_request,
            recipient,
        )

        async with self._send_lock:
            existing = await self.idempotency.get(
                request.tenant_id,
                request.idempotency_key,
                fingerprint,
            )

            if existing is not None:
                return existing

            await self.rate_limiter.acquire(
                request.tenant_id
            )

            self.audit.record(
                normalized_request,
                action="sms_submission_attempt",
                status="authorized",
            )

            try:
                provider_id, status = (
                    await self.provider.send(
                        recipient=recipient,
                        message=message,
                        idempotency_key=(
                            request.idempotency_key
                        ),
                    )
                )

            except Exception:
                self.audit.record(
                    normalized_request,
                    action="sms_submission",
                    status="failed_or_unknown",
                )
                raise

            result = SMSResult(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                provider_message_id=provider_id,
                status=status,
                recipient_masked=mask_phone(
                    recipient
                ),
                purpose=request.purpose,
                submitted_at=utc_now(),
                approval_id=(
                    approval.approval_id
                    if approval
                    else None
                ),
                dry_run=self.config.dry_run,
            )

            await self.idempotency.save(
                request.tenant_id,
                request.idempotency_key,
                fingerprint,
                result,
            )

            self.audit.record(
                normalized_request,
                action="sms_submission",
                status=status.value,
            )

            return result

    async def send_template(
        self,
        *,
        recipient: str,
        purpose: SMSPurpose,
        language: SMSLanguage,
        values: Mapping[str, str],
        tenant_id: str,
        idempotency_key: str,
        principal: SMSPrincipal,
        approval: SMSApproval | None = None,
        consent_confirmed: bool = False,
        lawful_emergency_basis_confirmed: bool = False,
        reference_id: str | None = None,
    ) -> SMSResult:
        message = SMSTemplateEngine.render(
            purpose,
            language,
            values,
        )

        request = SMSRequest(
            recipient=recipient,
            message=message,
            purpose=purpose,
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            language=language,
            consent_confirmed=consent_confirmed,
            lawful_emergency_basis_confirmed=(
                lawful_emergency_basis_confirmed
            ),
            reference_id=reference_id,
        )

        return await self.send(
            request,
            principal=principal,
            approval=approval,
        )

    async def close(self) -> None:
        await self.provider.close()

    async def __aenter__(
        self,
    ) -> "SMSAdapter":
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.close()


# ============================================================
# FACTORY
# ============================================================

def create_sms_adapter(
    config: SMSAdapterConfig | None = None,
    *,
    provider: SMSProvider | None = None,
) -> SMSAdapter:
    return SMSAdapter(
        config or SMSAdapterConfig(),
        provider=provider,
    )


# ============================================================
# SYNTHETIC DEMONSTRATION
# ============================================================

async def demo() -> None:
    """
    Dry-run demonstration only.

    No SMS is transmitted.
    The phone number is fictional.
    """

    config = SMSAdapterConfig(
        dry_run=True,
        require_approval_for_all=True,
    )

    principal = SMSPrincipal(
        user_id="demo-operator",
        tenant_id="demo-tenant",
        roles=frozenset({
            "emergency_operator"
        }),
    )

    request = SMSRequest(
        recipient="+919876543210",
        message=(
            "DEMO ONLY: Your SOS request has "
            "been received for human review."
        ),
        purpose=(
            SMSPurpose.SOS_ACKNOWLEDGMENT
        ),
        tenant_id="demo-tenant",
        idempotency_key="demo-sos-001",
        consent_confirmed=True,
    )

    approval = SMSApproval(
        approval_id="demo-approval",
        request_id=request.request_id,
        tenant_id="demo-tenant",
        approved_by="demo-supervisor",
        approved_at=utc_now(),
        status=SMSApprovalStatus.APPROVED,
    )

    async with SMSAdapter(config) as adapter:
        result = await adapter.send(
            request,
            principal=principal,
            approval=approval,
        )

        print({
            "request_id": result.request_id,
            "status": result.status.value,
            "recipient": result.recipient_masked,
            "dry_run": result.dry_run,
        })


if __name__ == "__main__":
    asyncio.run(demo())


__all__ = [
    "SMSAdapterError",
    "SMSConfigurationError",
    "SMSValidationError",
    "SMSAuthorizationError",
    "SMSProviderError",
    "SMSRateLimitError",
    "SMSWebhookError",
    "SMSIdempotencyError",
    "SMSPurpose",
    "SMSPriority",
    "SMSLanguage",
    "SMSStatus",
    "SMSApprovalStatus",
    "SMSAdapterConfig",
    "SMSPrincipal",
    "SMSApproval",
    "SMSRequest",
    "SMSResult",
    "SMSDeliveryEvent",
    "SMSAuditEvent",
    "SMSAuthorizationPolicy",
    "SMSTemplateEngine",
    "SMSRateLimiter",
    "SMSIdempotencyStore",
    "SMSProvider",
    "DryRunSMSProvider",
    "GenericRESTSMSProvider",
    "SMSWebhookVerifier",
    "SMSDeliveryEventParser",
    "SMSAuditLogger",
    "SMSAdapter",
    "create_sms_adapter",
]
