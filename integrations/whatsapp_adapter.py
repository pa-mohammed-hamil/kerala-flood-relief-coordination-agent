"""Placeholder module: integrations/whatsapp_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
WhatsApp Business Cloud API Adapter

File: backend/integrations/whatsapp_adapter.py

Python: 3.11+
Dependency:
    pip install httpx

Environment variables:
    WHATSAPP_ACCESS_TOKEN
    WHATSAPP_PHONE_NUMBER_ID
    WHATSAPP_APP_SECRET
    WHATSAPP_VERIFY_TOKEN
    WHATSAPP_GRAPH_API_VERSION

Features:
    - Meta WhatsApp Business Cloud API
    - Approved message templates
    - English and Malayalam templates
    - 24-hour customer service window checks
    - SOS acknowledgment preparation
    - Rescue and shelter notifications
    - Relief delivery updates
    - Human-approved emergency communications
    - Role-based authorization
    - Consent validation
    - Tenant isolation
    - Async HTTP transport
    - Request validation
    - Rate limiting
    - Idempotency handling
    - Webhook signature verification
    - Webhook challenge verification
    - Delivery status parsing
    - Privacy-aware audit logging
    - Dry-run mode
    - No automatic rescue dispatch

IMPORTANT:
    Requires an authorized WhatsApp Business
    Platform account and valid Meta credentials.

    Template names must correspond to actual
    approved WhatsApp message templates.

    WhatsApp is not a guaranteed emergency
    delivery channel. Critical SOS requests
    must also reach authorized responders
    through reliable emergency workflows.

    A caller-provided approval object is NOT
    proof of approval. Production must verify
    approvals against a trusted approval store.

    Production idempotency and rate limiting
    must use shared persistent infrastructure.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol

try:
    import httpx
except ImportError:
    httpx = None


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================
# EXCEPTIONS
# ============================================================

class WhatsAppAdapterError(Exception):
    pass


class WhatsAppConfigurationError(WhatsAppAdapterError):
    pass


class WhatsAppValidationError(WhatsAppAdapterError):
    pass


class WhatsAppAuthorizationError(WhatsAppAdapterError):
    pass


class WhatsAppProviderError(WhatsAppAdapterError):
    pass


class WhatsAppRateLimitError(WhatsAppAdapterError):
    pass


class WhatsAppWebhookError(WhatsAppAdapterError):
    pass


class WhatsAppIdempotencyError(WhatsAppAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class WhatsAppMessageType(str, Enum):
    TEXT = "text"
    TEMPLATE = "template"


class WhatsAppPurpose(str, Enum):
    SOS_ACKNOWLEDGMENT = "sos_acknowledgment"
    RESCUE_UPDATE = "rescue_update"
    SHELTER_UPDATE = "shelter_update"
    RELIEF_DELIVERY = "relief_delivery"
    MEDICAL_COORDINATION = "medical_coordination"
    VOLUNTEER_COORDINATION = "volunteer_coordination"
    EMERGENCY_WARNING = "emergency_warning"
    GENERAL = "general"


class WhatsAppLanguage(str, Enum):
    ENGLISH = "en"
    MALAYALAM = "ml"


class WhatsAppStatus(str, Enum):
    DRY_RUN = "dry_run"
    ACCEPTED = "accepted"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    UNKNOWN = "unknown"


class WhatsAppPriority(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def normalize_phone(phone: str) -> str:
    """
    Normalize a phone number to international
    digits-only format for WhatsApp Cloud API.

    Indian 10-digit numbers are converted
    to 91XXXXXXXXXX.

    Other international numbers must already
    include a country code.
    """
    if not isinstance(phone, str):
        raise WhatsAppValidationError(
            "Phone number must be a string"
        )

    value = re.sub(
        r"[\s()\-]",
        "",
        phone.strip(),
    )

    if value.startswith("00"):
        value = value[2:]

    elif value.startswith("+"):
        value = value[1:]

    if re.fullmatch(r"[6-9]\d{9}", value):
        value = "91" + value

    if not re.fullmatch(r"[1-9]\d{7,14}", value):
        raise WhatsAppValidationError(
            "Invalid international phone number"
        )

    return value


def mask_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)

    if len(digits) <= 4:
        return "****"

    return "*" * (len(digits) - 4) + digits[-4:]


def validate_message_text(
    text: str,
    *,
    max_length: int = 4096,
) -> str:
    if not isinstance(text, str):
        raise WhatsAppValidationError(
            "Message must be text"
        )

    value = text.strip()

    if not value:
        raise WhatsAppValidationError(
            "Message cannot be empty"
        )

    if len(value) > max_length:
        raise WhatsAppValidationError(
            "Message exceeds maximum length"
        )

    return value


def validate_template_name(name: str) -> str:
    if not re.fullmatch(
        r"[a-z0-9_]{1,512}",
        name,
    ):
        raise WhatsAppValidationError(
            "Invalid WhatsApp template name"
        )

    return name


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class WhatsAppAdapterConfig:
    access_token: str | None = None

    phone_number_id: str | None = None

    app_secret: str | None = None

    verify_token: str | None = None

    graph_api_version: str = "v23.0"

    timeout_seconds: float = 20.0

    max_message_length: int = 4096

    max_messages_per_minute: int = 30

    max_response_bytes: int = 100_000

    dry_run: bool = True

    require_approval_for_all: bool = True

    require_approved_templates: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(
            r"v\d+\.\d+",
            self.graph_api_version,
        ):
            raise WhatsAppConfigurationError(
                "Invalid Graph API version"
            )

        if self.timeout_seconds <= 0:
            raise WhatsAppConfigurationError(
                "Timeout must be positive"
            )

        if self.max_message_length <= 0:
            raise WhatsAppConfigurationError(
                "Message length must be positive"
            )

        if self.max_messages_per_minute <= 0:
            raise WhatsAppConfigurationError(
                "Rate limit must be positive"
            )

        if self.max_response_bytes <= 0:
            raise WhatsAppConfigurationError(
                "Response size limit must be positive"
            )

        if not self.dry_run:
            if not self.access_token:
                raise WhatsAppConfigurationError(
                    "WhatsApp access token required"
                )

            if not self.phone_number_id:
                raise WhatsAppConfigurationError(
                    "WhatsApp phone number ID required"
                )

            if not self.phone_number_id.isdigit():
                raise WhatsAppConfigurationError(
                    "Invalid WhatsApp phone number ID"
                )

    @classmethod
    def from_env(
        cls,
        *,
        dry_run: bool = True,
    ) -> "WhatsAppAdapterConfig":
        return cls(
            access_token=os.getenv(
                "WHATSAPP_ACCESS_TOKEN"
            ),
            phone_number_id=os.getenv(
                "WHATSAPP_PHONE_NUMBER_ID"
            ),
            app_secret=os.getenv(
                "WHATSAPP_APP_SECRET"
            ),
            verify_token=os.getenv(
                "WHATSAPP_VERIFY_TOKEN"
            ),
            graph_api_version=os.getenv(
                "WHATSAPP_GRAPH_API_VERSION",
                "v23.0",
            ),
            dry_run=dry_run,
        )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class WhatsAppPrincipal:
    user_id: str
    tenant_id: str
    roles: frozenset[str]


@dataclass(frozen=True)
class WhatsAppApproval:
    approval_id: str
    request_id: str
    tenant_id: str
    approved_by: str
    approved_at: datetime
    approved: bool


@dataclass(frozen=True)
class WhatsAppTemplate:
    name: str
    language: WhatsAppLanguage

    body_parameters: tuple[str, ...] = ()

    approved: bool = False


@dataclass(frozen=True)
class WhatsAppRequest:
    recipient: str

    purpose: WhatsAppPurpose

    tenant_id: str

    idempotency_key: str

    message_type: WhatsAppMessageType

    text: str | None = None

    template: WhatsAppTemplate | None = None

    priority: WhatsAppPriority = (
        WhatsAppPriority.NORMAL
    )

    consent_confirmed: bool = False

    lawful_emergency_basis_confirmed: bool = False

    customer_service_window_confirmed: bool = False

    reference_id: str | None = None

    request_id: str = field(
        default_factory=new_id
    )


@dataclass(frozen=True)
class WhatsAppResult:
    request_id: str

    tenant_id: str

    provider_message_id: str | None

    recipient_masked: str

    purpose: WhatsAppPurpose

    status: WhatsAppStatus

    submitted_at: datetime

    approval_id: str | None

    dry_run: bool


@dataclass(frozen=True)
class WhatsAppDeliveryEvent:
    provider_message_id: str

    status: WhatsAppStatus

    recipient_masked: str | None

    provider_timestamp: datetime | None

    received_at: datetime


@dataclass(frozen=True)
class WhatsAppAuditEvent:
    audit_id: str

    request_id: str

    tenant_id: str

    action: str

    status: str

    recipient_masked: str

    timestamp: datetime


# ============================================================
# APPROVAL VERIFICATION INTERFACE
# ============================================================

class WhatsAppApprovalVerifier(Protocol):
    async def verify(
        self,
        request: WhatsAppRequest,
        approval: WhatsAppApproval,
    ) -> bool:
        """
        Verify approval using trusted storage.

        Implement using the application's
        authenticated human approval service.
        """
        ...


# ============================================================
# AUTHORIZATION POLICY
# ============================================================

class WhatsAppAuthorizationPolicy:
    ALLOWED_ROLES = frozenset({
        "admin",
        "emergency_operator",
        "incident_commander",
        "communications_officer",
        "rescue_coordinator",
    })

    CRITICAL_PURPOSES = frozenset({
        WhatsAppPurpose.EMERGENCY_WARNING,
        WhatsAppPurpose.RESCUE_UPDATE,
        WhatsAppPurpose.MEDICAL_COORDINATION,
    })

    def __init__(
        self,
        config: WhatsAppAdapterConfig,
        approval_verifier: (
            WhatsAppApprovalVerifier | None
        ) = None,
    ) -> None:
        self.config = config
        self.approval_verifier = approval_verifier

    async def authorize(
        self,
        request: WhatsAppRequest,
        principal: WhatsAppPrincipal,
        approval: WhatsAppApproval | None,
    ) -> None:
        if request.tenant_id != principal.tenant_id:
            raise WhatsAppAuthorizationError(
                "Tenant mismatch"
            )

        if not (
            principal.roles & self.ALLOWED_ROLES
        ):
            raise WhatsAppAuthorizationError(
                "Insufficient permissions"
            )

        if not (
            request.consent_confirmed
            or request.lawful_emergency_basis_confirmed
        ):
            raise WhatsAppAuthorizationError(
                "Consent or lawful emergency "
                "messaging basis required"
            )

        approval_required = (
            self.config.require_approval_for_all
            or request.purpose
            in self.CRITICAL_PURPOSES
        )

        if not approval_required:
            return

        if approval is None or not approval.approved:
            raise WhatsAppAuthorizationError(
                "Human approval required"
            )

        if (
            approval.request_id != request.request_id
            or approval.tenant_id != request.tenant_id
            or approval.approved_by == principal.user_id
        ):
            raise WhatsAppAuthorizationError(
                "Invalid independent approval"
            )

        if self.approval_verifier is None:
            raise WhatsAppAuthorizationError(
                "Trusted approval verifier not configured"
            )

        verified = await self.approval_verifier.verify(
            request,
            approval,
        )

        if not verified:
            raise WhatsAppAuthorizationError(
                "Approval verification failed"
            )


# ============================================================
# TEMPLATE REGISTRY
# ============================================================

class WhatsAppTemplateRegistry:
    """
    Template definitions must match the
    templates approved in WhatsApp Manager.

    This registry is not proof of Meta approval.
    """

    def __init__(self) -> None:
        self._templates: dict[
            tuple[str, str],
            WhatsAppTemplate,
        ] = {}

    def register(
        self,
        template: WhatsAppTemplate,
    ) -> None:
        validate_template_name(
            template.name
        )

        key = (
            template.name,
            template.language.value,
        )

        self._templates[key] = template

    def get(
        self,
        name: str,
        language: WhatsAppLanguage,
    ) -> WhatsAppTemplate | None:
        return self._templates.get(
            (name, language.value)
        )

    def validate(
        self,
        template: WhatsAppTemplate,
        *,
        require_approved: bool,
    ) -> None:
        registered = self.get(
            template.name,
            template.language,
        )

        if registered is None:
            raise WhatsAppValidationError(
                "Template not registered"
            )

        if registered != template:
            raise WhatsAppValidationError(
                "Template definition mismatch"
            )

        if require_approved and not registered.approved:
            raise WhatsAppValidationError(
                "Template not marked approved"
            )


# ============================================================
# MESSAGE BUILDER
# ============================================================

class WhatsAppMessageBuilder:
    def __init__(
        self,
        config: WhatsAppAdapterConfig,
        registry: WhatsAppTemplateRegistry,
    ) -> None:
        self.config = config
        self.registry = registry

    def build(
        self,
        request: WhatsAppRequest,
    ) -> dict[str, Any]:
        recipient = normalize_phone(
            request.recipient
        )

        base = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
        }

        if (
            request.message_type
            == WhatsAppMessageType.TEXT
        ):
            if not (
                request.customer_service_window_confirmed
            ):
                raise WhatsAppValidationError(
                    "Free-form text requires a verified "
                    "open customer service window"
                )

            if request.text is None:
                raise WhatsAppValidationError(
                    "Text message required"
                )

            text = validate_message_text(
                request.text,
                max_length=(
                    self.config.max_message_length
                ),
            )

            base.update({
                "type": "text",
                "text": {
                    "preview_url": False,
                    "body": text,
                },
            })

            return base

        if (
            request.message_type
            == WhatsAppMessageType.TEMPLATE
        ):
            template = request.template

            if template is None:
                raise WhatsAppValidationError(
                    "WhatsApp template required"
                )

            self.registry.validate(
                template,
                require_approved=(
                    self.config.require_approved_templates
                ),
            )

            parameters = [
                {
                    "type": "text",
                    "text": validate_message_text(
                        value,
                        max_length=1024,
                    ),
                }
                for value in template.body_parameters
            ]

            template_payload: dict[str, Any] = {
                "name": template.name,
                "language": {
                    "code": template.language.value
                },
            }

            if parameters:
                template_payload["components"] = [
                    {
                        "type": "body",
                        "parameters": parameters,
                    }
                ]

            base.update({
                "type": "template",
                "template": template_payload,
            })

            return base

        raise WhatsAppValidationError(
            "Unsupported message type"
        )


# ============================================================
# RATE LIMITER
# ============================================================

class WhatsAppRateLimiter:
    def __init__(
        self,
        max_per_minute: int,
    ) -> None:
        self.max_per_minute = max_per_minute

        self._history: dict[
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

            timestamps = [
                value
                for value in self._history.get(
                    tenant_id,
                    [],
                )
                if now - value < 60
            ]

            if len(timestamps) >= self.max_per_minute:
                raise WhatsAppRateLimitError(
                    "WhatsApp rate limit exceeded"
                )

            timestamps.append(now)

            self._history[tenant_id] = timestamps


# ============================================================
# IDEMPOTENCY STORE
# ============================================================

class WhatsAppIdempotencyStore:
    """
    Process-local demonstration store.

    Use Redis or PostgreSQL with atomic
    reservation and transactional outbox
    for production.
    """

    def __init__(self) -> None:
        self._records: dict[
            tuple[str, str],
            tuple[str, WhatsAppResult],
        ] = {}

        self._lock = asyncio.Lock()

    async def get(
        self,
        tenant_id: str,
        key: str,
        fingerprint: str,
    ) -> WhatsAppResult | None:
        async with self._lock:
            existing = self._records.get(
                (tenant_id, key)
            )

            if existing is None:
                return None

            stored_fingerprint, result = existing

            if stored_fingerprint != fingerprint:
                raise WhatsAppIdempotencyError(
                    "Idempotency key reused "
                    "with different message"
                )

            return result

    async def save(
        self,
        tenant_id: str,
        key: str,
        fingerprint: str,
        result: WhatsAppResult,
    ) -> None:
        async with self._lock:
            self._records[
                (tenant_id, key)
            ] = (
                fingerprint,
                result,
            )


# ============================================================
# TRANSPORT INTERFACE
# ============================================================

class WhatsAppTransport(Protocol):
    async def send(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[str, WhatsAppStatus]:
        ...

    async def close(self) -> None:
        ...


# ============================================================
# DRY-RUN TRANSPORT
# ============================================================

class DryRunWhatsAppTransport:
    async def send(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[str, WhatsAppStatus]:
        return (
            "dry-run-" + new_id(),
            WhatsAppStatus.DRY_RUN,
        )

    async def close(self) -> None:
        return None


# ============================================================
# META CLOUD API TRANSPORT
# ============================================================

class MetaWhatsAppCloudTransport:
    BASE_URL = "https://graph.facebook.com"

    def __init__(
        self,
        config: WhatsAppAdapterConfig,
    ) -> None:
        if httpx is None:
            raise WhatsAppConfigurationError(
                "Install httpx: pip install httpx"
            )

        if not config.access_token:
            raise WhatsAppConfigurationError(
                "Access token required"
            )

        if not config.phone_number_id:
            raise WhatsAppConfigurationError(
                "Phone number ID required"
            )

        self.config = config

        self._client = httpx.AsyncClient(
            timeout=config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    async def send(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[str, WhatsAppStatus]:
        url = (
            f"{self.BASE_URL}/"
            f"{self.config.graph_api_version}/"
            f"{self.config.phone_number_id}/messages"
        )

        headers = {
            "Authorization": (
                f"Bearer {self.config.access_token}"
            ),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=dict(payload),
            ) as response:
                if response.status_code in (401, 403):
                    raise WhatsAppAuthorizationError(
                        "Meta authentication failed"
                    )

                if response.status_code >= 400:
                    raise WhatsAppProviderError(
                        f"Meta API HTTP "
                        f"{response.status_code}"
                    )

                chunks = []
                total = 0

                async for chunk in response.aiter_bytes():
                    total += len(chunk)

                    if total > (
                        self.config.max_response_bytes
                    ):
                        raise WhatsAppProviderError(
                            "Provider response too large"
                        )

                    chunks.append(chunk)

            data = json.loads(
                b"".join(chunks)
            )

            if not isinstance(data, Mapping):
                raise WhatsAppProviderError(
                    "Invalid Meta API response"
                )

            messages = data.get(
                "messages"
            )

            if (
                not isinstance(messages, list)
                or not messages
                or not isinstance(
                    messages[0],
                    Mapping,
                )
            ):
                raise WhatsAppProviderError(
                    "Missing Meta message response"
                )

            message_id = messages[0].get(
                "id"
            )

            if not isinstance(
                message_id,
                str,
            ) or not message_id:
                raise WhatsAppProviderError(
                    "Missing WhatsApp message ID"
                )

            # API acceptance does not prove delivery.
            return (
                message_id,
                WhatsAppStatus.ACCEPTED,
            )

        except (
            WhatsAppProviderError,
            WhatsAppAuthorizationError,
        ):
            raise

        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            raise WhatsAppProviderError(
                "WhatsApp submission outcome unknown. "
                "Reconcile provider state before retry."
            ) from exc

        except json.JSONDecodeError as exc:
            raise WhatsAppProviderError(
                "Invalid Meta response JSON"
            ) from exc

    async def close(self) -> None:
        await self._client.aclose()


# ============================================================
# WEBHOOK VERIFICATION
# ============================================================

class WhatsAppWebhookVerifier:
    """
    Meta webhook signature:
        X-Hub-Signature-256:
        sha256=<hex digest>

    HMAC-SHA256 is computed over raw body
    using the Meta App Secret.
    """

    def __init__(
        self,
        *,
        app_secret: str,
        verify_token: str,
    ) -> None:
        if not app_secret or not verify_token:
            raise WhatsAppConfigurationError(
                "Webhook credentials required"
            )

        self.app_secret = app_secret.encode(
            "utf-8"
        )

        self.verify_token = verify_token

    def verify_challenge(
        self,
        *,
        mode: str,
        token: str,
        challenge: str,
    ) -> str:
        if mode != "subscribe":
            raise WhatsAppWebhookError(
                "Invalid webhook mode"
            )

        if not hmac.compare_digest(
            token,
            self.verify_token,
        ):
            raise WhatsAppWebhookError(
                "Invalid verification token"
            )

        return challenge

    def verify_signature(
        self,
        *,
        raw_body: bytes,
        signature_header: str,
    ) -> None:
        if not signature_header.startswith(
            "sha256="
        ):
            raise WhatsAppWebhookError(
                "Invalid signature format"
            )

        received = signature_header[
            len("sha256="):
        ]

        expected = hmac.new(
            self.app_secret,
            raw_body,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(
            received.lower(),
            expected,
        ):
            raise WhatsAppWebhookError(
                "Webhook signature verification failed"
            )


# ============================================================
# DELIVERY STATUS PARSER
# ============================================================

class WhatsAppWebhookParser:
    STATUS_MAPPING = {
        "sent": WhatsAppStatus.SENT,
        "delivered": WhatsAppStatus.DELIVERED,
        "read": WhatsAppStatus.READ,
        "failed": WhatsAppStatus.FAILED,
    }

    @classmethod
    def parse_delivery_events(
        cls,
        payload: Mapping[str, Any],
    ) -> list[WhatsAppDeliveryEvent]:
        events = []

        entries = payload.get(
            "entry",
            [],
        )

        if not isinstance(entries, list):
            raise WhatsAppWebhookError(
                "Invalid webhook entries"
            )

        for entry in entries:
            if not isinstance(entry, Mapping):
                continue

            changes = entry.get(
                "changes",
                [],
            )

            if not isinstance(changes, list):
                continue

            for change in changes:
                if not isinstance(change, Mapping):
                    continue

                value = change.get(
                    "value",
                    {},
                )

                if not isinstance(value, Mapping):
                    continue

                statuses = value.get(
                    "statuses",
                    [],
                )

                if not isinstance(statuses, list):
                    continue

                for item in statuses:
                    if not isinstance(item, Mapping):
                        continue

                    message_id = item.get(
                        "id"
                    )

                    if not isinstance(
                        message_id,
                        str,
                    ) or not message_id:
                        continue

                    raw_status = str(
                        item.get("status", "")
                    ).casefold()

                    status = cls.STATUS_MAPPING.get(
                        raw_status,
                        WhatsAppStatus.UNKNOWN,
                    )

                    timestamp = None

                    raw_timestamp = item.get(
                        "timestamp"
                    )

                    if raw_timestamp is not None:
                        try:
                            timestamp = (
                                datetime.fromtimestamp(
                                    int(raw_timestamp),
                                    tz=timezone.utc,
                                )
                            )
                        except (
                            TypeError,
                            ValueError,
                            OverflowError,
                        ):
                            timestamp = None

                    recipient = item.get(
                        "recipient_id"
                    )

                    events.append(
                        WhatsAppDeliveryEvent(
                            provider_message_id=message_id,
                            status=status,
                            recipient_masked=(
                                mask_phone(str(recipient))
                                if recipient
                                else None
                            ),
                            provider_timestamp=timestamp,
                            received_at=utc_now(),
                        )
                    )

        return events


# ============================================================
# AUDIT LOGGER
# ============================================================

class WhatsAppAuditLogger:
    def __init__(
        self,
        max_events: int = 5000,
    ) -> None:
        self.max_events = max_events

        self._events: list[
            WhatsAppAuditEvent
        ] = []

    def record(
        self,
        request: WhatsAppRequest,
        *,
        action: str,
        status: str,
    ) -> None:
        self._events.append(
            WhatsAppAuditEvent(
                audit_id=new_id(),
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                action=action,
                status=status,
                recipient_masked=mask_phone(
                    request.recipient
                ),
                timestamp=utc_now(),
            )
        )

        if len(self._events) > self.max_events:
            self._events = self._events[
                -self.max_events:
            ]

    def events(
        self,
        tenant_id: str,
    ) -> list[WhatsAppAuditEvent]:
        return [
            event
            for event in self._events
            if event.tenant_id == tenant_id
        ]


# ============================================================
# MAIN WHATSAPP ADAPTER
# ============================================================

class WhatsAppAdapter:
    def __init__(
        self,
        config: WhatsAppAdapterConfig,
        *,
        transport: WhatsAppTransport | None = None,
        approval_verifier: (
            WhatsAppApprovalVerifier | None
        ) = None,
        template_registry: (
            WhatsAppTemplateRegistry | None
        ) = None,
    ) -> None:
        self.config = config

        self.registry = (
            template_registry
            or WhatsAppTemplateRegistry()
        )

        self.builder = WhatsAppMessageBuilder(
            config,
            self.registry,
        )

        self.authorization = (
            WhatsAppAuthorizationPolicy(
                config,
                approval_verifier,
            )
        )

        self.rate_limiter = WhatsAppRateLimiter(
            config.max_messages_per_minute
        )

        self.idempotency = (
            WhatsAppIdempotencyStore()
        )

        self.audit = WhatsAppAuditLogger()

        if transport is not None:
            self.transport = transport

        elif config.dry_run:
            self.transport = DryRunWhatsAppTransport()

        else:
            self.transport = (
                MetaWhatsAppCloudTransport(config)
            )

        self._send_lock = asyncio.Lock()

    @staticmethod
    def _fingerprint(
        request: WhatsAppRequest,
        payload: Mapping[str, Any],
    ) -> str:
        serialized = json.dumps(
            {
                "tenant_id": request.tenant_id,
                "purpose": request.purpose.value,
                "payload": payload,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    async def send(
        self,
        request: WhatsAppRequest,
        *,
        principal: WhatsAppPrincipal,
        approval: WhatsAppApproval | None = None,
    ) -> WhatsAppResult:
        if not request.idempotency_key.strip():
            raise WhatsAppValidationError(
                "Idempotency key required"
            )

        await self.authorization.authorize(
            request,
            principal,
            approval,
        )

        payload = self.builder.build(
            request
        )

        fingerprint = self._fingerprint(
            request,
            payload,
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
                request,
                action="submission_attempt",
                status="authorized",
            )

            try:
                message_id, status = (
                    await self.transport.send(
                        payload
                    )
                )

            except Exception:
                self.audit.record(
                    request,
                    action="submission",
                    status="failed_or_unknown",
                )
                raise

            result = WhatsAppResult(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                provider_message_id=message_id,
                recipient_masked=mask_phone(
                    request.recipient
                ),
                purpose=request.purpose,
                status=status,
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
                request,
                action="submission",
                status=status.value,
            )

            return result

    async def close(self) -> None:
        await self.transport.close()

    async def __aenter__(
        self,
    ) -> "WhatsAppAdapter":
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.close()


# ============================================================
# NOTIFICATION FACTORY
# ============================================================

class FloodWhatsAppNotificationFactory:
    """
    Creates requests only.
    Does not transmit messages.
    """

    @staticmethod
    def sos_acknowledgment(
        *,
        recipient: str,
        tenant_id: str,
        sos_reference: str,
        template_name: str,
        language: WhatsAppLanguage,
        consent_confirmed: bool,
    ) -> WhatsAppRequest:
        return WhatsAppRequest(
            recipient=recipient,
            purpose=(
                WhatsAppPurpose.SOS_ACKNOWLEDGMENT
            ),
            tenant_id=tenant_id,
            idempotency_key=(
                f"sos-ack:{tenant_id}:{sos_reference}"
            ),
            message_type=(
                WhatsAppMessageType.TEMPLATE
            ),
            template=WhatsAppTemplate(
                name=template_name,
                language=language,
                body_parameters=(
                    sos_reference,
                ),
                approved=True,
            ),
            priority=WhatsAppPriority.HIGH,
            consent_confirmed=consent_confirmed,
            reference_id=sos_reference,
        )

    @staticmethod
    def shelter_update(
        *,
        recipient: str,
        tenant_id: str,
        shelter_reference: str,
        update: str,
        template_name: str,
        language: WhatsAppLanguage,
        consent_confirmed: bool,
    ) -> WhatsAppRequest:
        return WhatsAppRequest(
            recipient=recipient,
            purpose=WhatsAppPurpose.SHELTER_UPDATE,
            tenant_id=tenant_id,
            idempotency_key=(
                f"shelter:{tenant_id}:"
                f"{shelter_reference}:"
                f"{hashlib.sha256(update.encode()).hexdigest()}"
            ),
            message_type=(
                WhatsAppMessageType.TEMPLATE
            ),
            template=WhatsAppTemplate(
                name=template_name,
                language=language,
                body_parameters=(
                    shelter_reference,
                    update,
                ),
                approved=True,
            ),
            consent_confirmed=consent_confirmed,
            reference_id=shelter_reference,
        )

    @staticmethod
    def rescue_update(
        *,
        recipient: str,
        tenant_id: str,
        rescue_reference: str,
        update: str,
        template_name: str,
        language: WhatsAppLanguage,
        lawful_emergency_basis_confirmed: bool,
    ) -> WhatsAppRequest:
        return WhatsAppRequest(
            recipient=recipient,
            purpose=WhatsAppPurpose.RESCUE_UPDATE,
            tenant_id=tenant_id,
            idempotency_key=(
                f"rescue:{tenant_id}:"
                f"{rescue_reference}:"
                f"{hashlib.sha256(update.encode()).hexdigest()}"
            ),
            message_type=(
                WhatsAppMessageType.TEMPLATE
            ),
            template=WhatsAppTemplate(
                name=template_name,
                language=language,
                body_parameters=(
                    rescue_reference,
                    update,
                ),
                approved=True,
            ),
            priority=WhatsAppPriority.CRITICAL,
            lawful_emergency_basis_confirmed=(
                lawful_emergency_basis_confirmed
            ),
            reference_id=rescue_reference,
        )


# ============================================================
# FACTORY
# ============================================================

def create_whatsapp_adapter(
    config: WhatsAppAdapterConfig | None = None,
    *,
    transport: WhatsAppTransport | None = None,
    approval_verifier: (
        WhatsAppApprovalVerifier | None
    ) = None,
    template_registry: (
        WhatsAppTemplateRegistry | None
    ) = None,
) -> WhatsAppAdapter:
    return WhatsAppAdapter(
        config or WhatsAppAdapterConfig(),
        transport=transport,
        approval_verifier=approval_verifier,
        template_registry=template_registry,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def demo() -> None:
    """
    Synthetic request construction only.

    No WhatsApp message is sent.
    """

    registry = WhatsAppTemplateRegistry()

    registry.register(
        WhatsAppTemplate(
            name="flood_sos_acknowledgment",
            language=WhatsAppLanguage.ENGLISH,
            body_parameters=("DEMO-SOS-001",),
            approved=True,
        )
    )

    config = WhatsAppAdapterConfig(
        dry_run=True,
    )

    request = (
        FloodWhatsAppNotificationFactory
        .sos_acknowledgment(
            recipient="+919876543210",
            tenant_id="demo",
            sos_reference="DEMO-SOS-001",
            template_name=(
                "flood_sos_acknowledgment"
            ),
            language=WhatsAppLanguage.ENGLISH,
            consent_confirmed=True,
        )
    )

    builder = WhatsAppMessageBuilder(
        config,
        registry,
    )

    payload = builder.build(request)

    # Mask recipient before printing.
    payload["to"] = mask_phone(
        payload["to"]
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    demo()


__all__ = [
    "WhatsAppAdapterError",
    "WhatsAppConfigurationError",
    "WhatsAppValidationError",
    "WhatsAppAuthorizationError",
    "WhatsAppProviderError",
    "WhatsAppRateLimitError",
    "WhatsAppWebhookError",
    "WhatsAppIdempotencyError",
    "WhatsAppMessageType",
    "WhatsAppPurpose",
    "WhatsAppLanguage",
    "WhatsAppStatus",
    "WhatsAppPriority",
    "WhatsAppAdapterConfig",
    "WhatsAppPrincipal",
    "WhatsAppApproval",
    "WhatsAppTemplate",
    "WhatsAppRequest",
    "WhatsAppResult",
    "WhatsAppDeliveryEvent",
    "WhatsAppAuditEvent",
    "WhatsAppApprovalVerifier",
    "WhatsAppAuthorizationPolicy",
    "WhatsAppTemplateRegistry",
    "WhatsAppMessageBuilder",
    "WhatsAppRateLimiter",
    "WhatsAppIdempotencyStore",
    "WhatsAppTransport",
    "DryRunWhatsAppTransport",
    "MetaWhatsAppCloudTransport",
    "WhatsAppWebhookVerifier",
    "WhatsAppWebhookParser",
    "WhatsAppAuditLogger",
    "WhatsAppAdapter",
    "FloodWhatsAppNotificationFactory",
    "create_whatsapp_adapter",
]
