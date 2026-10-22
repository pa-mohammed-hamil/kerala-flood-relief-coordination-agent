from __future__ import annotations

"""
backend/services/notification_service.py

Kerala Flood Relief Coordination Agent
---------------------------------------
Production-oriented notification coordination service.

Features:
- SMS, email, push, WhatsApp and in-app notifications
- Emergency SOS and flood-alert notifications
- Rescue, evacuation, medical and logistics notifications
- Malayalam and English templates
- Priority-based notification handling
- Recipient consent and channel preferences
- Role-based authorization
- Notification scheduling and expiration
- Idempotency and duplicate prevention
- Retry policies with exponential backoff
- Provider delivery status tracking
- Delivery receipts and acknowledgements
- Notification cancellation
- Batch notification creation
- District notification statistics
- Audit logging and structured reporting

Architecture:
    NotificationService
        -> NotificationRepository
        -> NotificationPreferenceGateway
        -> NotificationAuthorizationGateway
        -> NotificationProvider
        -> NotificationAuditGateway

IMPORTANT:
- Creating a notification does not guarantee delivery.
- Providers must be integrated and configured separately.
- Emergency notifications must use approved operational
  procedures and verified recipient directories.
- No external notification is sent during database
  transactions.
- Production systems should use a transactional outbox,
  queue workers, provider idempotency and reconciliation.
- Do not place patient details, passwords or sensitive
  personal information in notification payloads.
"""


import logging
import random
import re
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from string import Formatter
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class NotificationChannel(str, Enum):
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    WHATSAPP = "whatsapp"
    IN_APP = "in_app"


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    PROCESSING = "processing"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    ACKNOWLEDGED = "acknowledged"
    RETRY_PENDING = "retry_pending"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class NotificationType(str, Enum):
    FLOOD_ALERT = "flood_alert"
    WEATHER_WARNING = "weather_warning"
    RIVER_WARNING = "river_warning"
    DAM_WARNING = "dam_warning"

    SOS_RECEIVED = "sos_received"
    SOS_ACKNOWLEDGED = "sos_acknowledged"
    SOS_ESCALATED = "sos_escalated"

    INCIDENT_CREATED = "incident_created"
    INCIDENT_ESCALATED = "incident_escalated"

    RESCUE_ASSIGNED = "rescue_assigned"
    RESCUE_DISPATCHED = "rescue_dispatched"
    RESCUE_STATUS = "rescue_status"

    EVACUATION_APPROVED = "evacuation_approved"
    EVACUATION_NOTICE = "evacuation_notice"

    SHELTER_ASSIGNED = "shelter_assigned"
    SHELTER_CAPACITY = "shelter_capacity"

    RELIEF_DISPATCHED = "relief_dispatched"
    RELIEF_DELIVERED = "relief_delivered"
    INVENTORY_LOW = "inventory_low"

    MEDICAL_TRANSPORT = "medical_transport"
    MEDICAL_ESCALATION = "medical_escalation"

    VOLUNTEER_ASSIGNED = "volunteer_assigned"

    MISSING_PERSON_UPDATE = "missing_person_update"

    SYSTEM_ALERT = "system_alert"
    GENERAL = "general"


class NotificationLanguage(str, Enum):
    ENGLISH = "en"
    MALAYALAM = "ml"


class DeliveryFailureType(str, Enum):
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    PERMANENT = "permanent"
    INVALID_RECIPIENT = "invalid_recipient"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


# ============================================================
# EXCEPTIONS
# ============================================================

class NotificationServiceError(Exception):
    pass


class NotificationValidationError(NotificationServiceError):
    pass


class NotificationNotFoundError(NotificationServiceError):
    pass


class NotificationConflictError(NotificationServiceError):
    pass


class NotificationAuthorizationError(NotificationServiceError):
    pass


class NotificationProviderError(NotificationServiceError):

    def __init__(
        self,
        message: str,
        *,
        failure_type: DeliveryFailureType = (
            DeliveryFailureType.TRANSIENT
        ),
        retry_after_seconds: int | None = None,
    ):
        super().__init__(message)

        self.failure_type = failure_type
        self.retry_after_seconds = retry_after_seconds


class NotificationTemplateError(NotificationServiceError):
    pass


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise NotificationValidationError(
            f"{name} must be timezone-aware"
        )


def require_text(
    value: str,
    name: str,
    *,
    max_length: int = 10000,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NotificationValidationError(
            f"{name} is required"
        )

    cleaned = value.strip()

    if len(cleaned) > max_length:
        raise NotificationValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return cleaned


def normalize_phone(value: str) -> str:
    cleaned = re.sub(r"[\s()-]", "", value)

    if not re.fullmatch(r"\+[1-9]\d{7,14}", cleaned):
        raise NotificationValidationError(
            "Phone number must use international E.164 format"
        )

    return cleaned


def validate_email(value: str) -> str:
    cleaned = value.strip().lower()

    if (
        len(cleaned) > 254
        or not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            cleaned,
        )
    ):
        raise NotificationValidationError(
            "Invalid email address"
        )

    return cleaned


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class NotificationRecipient:
    recipient_id: str
    district_id: int

    phone_number: str | None = None
    email_address: str | None = None
    push_token: str | None = None

    preferred_language: NotificationLanguage = (
        NotificationLanguage.ENGLISH
    )

    verified: bool = False

    def __post_init__(self) -> None:
        require_text(
            self.recipient_id,
            "Recipient ID",
            max_length=200,
        )

        if self.district_id <= 0:
            raise NotificationValidationError(
                "Invalid district ID"
            )


@dataclass(frozen=True)
class NotificationCreateRequest:
    recipient: NotificationRecipient

    notification_type: NotificationType
    channel: NotificationChannel
    priority: NotificationPriority

    title: str
    message: str

    reference_id: str | None = None
    idempotency_key: str | None = None

    scheduled_at: datetime | None = None
    expires_at: datetime | None = None

    requires_acknowledgement: bool = False

    language: NotificationLanguage | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Notification:
    id: str
    district_id: int

    recipient_id: str

    notification_type: NotificationType
    channel: NotificationChannel
    priority: NotificationPriority

    title: str
    message: str

    language: NotificationLanguage

    status: NotificationStatus

    destination: str

    reference_id: str | None = None
    idempotency_key: str | None = None

    requires_acknowledgement: bool = False

    created_by: int | None = None

    provider_message_id: str | None = None

    attempts: int = 0
    max_attempts: int = 5

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    scheduled_at: datetime | None = None
    next_attempt_at: datetime | None = None
    expires_at: datetime | None = None

    sent_at: datetime | None = None
    delivered_at: datetime | None = None
    read_at: datetime | None = None
    acknowledged_at: datetime | None = None

    last_error: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ProviderSendResult:
    accepted: bool

    provider_message_id: str | None = None

    delivered: bool = False

    failure_type: DeliveryFailureType | None = None
    error_message: str | None = None

    retry_after_seconds: int | None = None


@dataclass(frozen=True)
class NotificationTemplate:
    template_id: str
    language: NotificationLanguage

    title_template: str
    body_template: str


@dataclass(frozen=True)
class NotificationDeliveryReceipt:
    notification_id: str
    provider_message_id: str

    delivered: bool
    occurred_at: datetime

    provider_event_id: str | None = None

    error_message: str | None = None

    def __post_init__(self) -> None:
        validate_datetime(
            self.occurred_at,
            "occurred_at",
        )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class NotificationRepository(Protocol):

    async def create(
        self,
        notification: Notification,
    ) -> Notification:
        ...

    async def get_by_id(
        self,
        notification_id: str,
    ) -> Notification | None:
        ...

    async def update(
        self,
        notification: Notification,
        *,
        expected_version: int,
    ) -> Notification:
        """
        Atomically compare stored version and update.

        Raise NotificationConflictError on mismatch.
        """
        ...

    async def find_by_idempotency_key(
        self,
        *,
        district_id: int,
        idempotency_key: str,
    ) -> Notification | None:
        ...

    async def list_due(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[Notification]:
        ...

    async def list_by_recipient(
        self,
        recipient_id: str,
    ) -> Sequence[Notification]:
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[Notification]:
        ...

    async def find_by_provider_message_id(
        self,
        provider_message_id: str,
    ) -> Notification | None:
        ...


class NotificationPreferenceGateway(Protocol):

    async def is_allowed(
        self,
        *,
        recipient_id: str,
        channel: NotificationChannel,
        notification_type: NotificationType,
    ) -> bool:
        """
        Enforce verified consent, opt-out preferences,
        channel restrictions and applicable regulations.

        Mandatory official emergency messages require
        a separate legally authorized policy.
        """
        ...


class NotificationAuthorizationGateway(Protocol):

    async def can_create(
        self,
        *,
        user_id: int,
        district_id: int,
        notification_type: NotificationType,
    ) -> bool:
        ...

    async def can_dispatch(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_manage(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class NotificationAuditGateway(Protocol):

    async def record(
        self,
        action: str,
        *,
        actor_id: int | None,
        entity_type: str,
        entity_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...


class NotificationProvider(Protocol):

    async def send(
        self,
        *,
        destination: str,
        title: str,
        message: str,
        idempotency_key: str,
        metadata: Mapping[str, Any],
    ) -> ProviderSendResult:
        ...


# ============================================================
# TEMPLATE ENGINE
# ============================================================

class NotificationTemplateEngine:
    """
    Simple allowlisted template interpolation.

    Rejects attribute access, index access and
    unsupported conversion/format expressions.
    """

    def __init__(
        self,
        templates: Sequence[NotificationTemplate] = (),
    ):
        self._templates = {
            (
                template.template_id,
                template.language,
            ): template
            for template in templates
        }

    def register(
        self,
        template: NotificationTemplate,
    ) -> None:
        self._templates[
            (
                template.template_id,
                template.language,
            )
        ] = template

    @staticmethod
    def _render_string(
        template: str,
        values: Mapping[str, str],
    ) -> str:

        formatter = Formatter()

        for _, field_name, format_spec, conversion in (
            formatter.parse(template)
        ):
            if field_name is None:
                continue

            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*",
                field_name,
            ):
                raise NotificationTemplateError(
                    "Unsafe template field"
                )

            if format_spec or conversion:
                raise NotificationTemplateError(
                    "Template formatting is not allowed"
                )

            if field_name not in values:
                raise NotificationTemplateError(
                    f"Missing template variable: {field_name}"
                )

        try:
            return template.format_map(values)
        except (KeyError, ValueError) as exc:
            raise NotificationTemplateError(
                "Template rendering failed"
            ) from exc

    def render(
        self,
        *,
        template_id: str,
        language: NotificationLanguage,
        values: Mapping[str, str],
    ) -> tuple[str, str]:

        template = self._templates.get(
            (template_id, language)
        )

        if template is None:
            raise NotificationTemplateError(
                "Notification template not found"
            )

        safe_values = {
            key: str(value)
            for key, value in values.items()
        }

        title = self._render_string(
            template.title_template,
            safe_values,
        )

        message = self._render_string(
            template.body_template,
            safe_values,
        )

        return title, message


# ============================================================
# DEFAULT TEMPLATES
# ============================================================

DEFAULT_TEMPLATES = (
    NotificationTemplate(
        template_id="sos_received",
        language=NotificationLanguage.ENGLISH,
        title_template="SOS Request Received",
        body_template=(
            "Your emergency request {reference} has been "
            "recorded. This is not confirmation that a "
            "rescue team has been dispatched."
        ),
    ),
    NotificationTemplate(
        template_id="sos_received",
        language=NotificationLanguage.MALAYALAM,
        title_template="അടിയന്തര സഹായ അഭ്യർത്ഥന ലഭിച്ചു",
        body_template=(
            "നിങ്ങളുടെ അടിയന്തര സഹായ അഭ്യർത്ഥന "
            "{reference} രേഖപ്പെടുത്തി. "
            "രക്ഷാസംഘത്തെ അയച്ചതായി ഇത് സ്ഥിരീകരിക്കുന്നില്ല."
        ),
    ),
    NotificationTemplate(
        template_id="rescue_assigned",
        language=NotificationLanguage.ENGLISH,
        title_template="Rescue Coordination Update",
        body_template=(
            "Rescue coordination reference {reference} "
            "has been assigned for further action. "
            "Follow instructions from authorized responders."
        ),
    ),
    NotificationTemplate(
        template_id="relief_delivered",
        language=NotificationLanguage.ENGLISH,
        title_template="Relief Delivery Update",
        body_template=(
            "Relief shipment {reference} has been "
            "recorded as delivered."
        ),
    ),
    NotificationTemplate(
        template_id="medical_transport",
        language=NotificationLanguage.ENGLISH,
        title_template="Medical Transport Coordination",
        body_template=(
            "Medical transport case {reference} "
            "has a coordination update. "
            "Contact the authorized medical coordinator "
            "for confirmed arrangements."
        ),
    ),
)


# ============================================================
# RETRY POLICY
# ============================================================

class NotificationRetryPolicy:

    RETRYABLE_FAILURES = frozenset({
        DeliveryFailureType.TRANSIENT,
        DeliveryFailureType.RATE_LIMITED,
        DeliveryFailureType.PROVIDER_UNAVAILABLE,
    })

    def __init__(
        self,
        *,
        base_delay_seconds: int = 30,
        maximum_delay_seconds: int = 3600,
        jitter_ratio: float = 0.20,
    ):
        if base_delay_seconds <= 0:
            raise ValueError(
                "Base retry delay must be positive"
            )

        if maximum_delay_seconds < base_delay_seconds:
            raise ValueError(
                "Maximum retry delay is too small"
            )

        if not 0 <= jitter_ratio <= 1:
            raise ValueError(
                "Jitter ratio must be between 0 and 1"
            )

        self.base_delay_seconds = base_delay_seconds
        self.maximum_delay_seconds = (
            maximum_delay_seconds
        )
        self.jitter_ratio = jitter_ratio

    def should_retry(
        self,
        *,
        failure_type: DeliveryFailureType,
        attempts: int,
        max_attempts: int,
    ) -> bool:
        return (
            failure_type in self.RETRYABLE_FAILURES
            and attempts < max_attempts
        )

    def next_attempt(
        self,
        *,
        attempts: int,
        retry_after_seconds: int | None = None,
        now: datetime | None = None,
    ) -> datetime:

        now = now or utc_now()

        exponential_delay = min(
            self.maximum_delay_seconds,
            self.base_delay_seconds
            * (2 ** max(0, attempts - 1)),
        )

        jitter = random.uniform(
            0,
            exponential_delay * self.jitter_ratio,
        )

        delay = exponential_delay + jitter

        if retry_after_seconds is not None:
            delay = max(
                delay,
                retry_after_seconds,
            )

        return now + timedelta(
            seconds=delay
        )


# ============================================================
# NOTIFICATION SERVICE
# ============================================================

class NotificationService:

    TERMINAL_STATUSES = frozenset({
        NotificationStatus.DELIVERED,
        NotificationStatus.READ,
        NotificationStatus.ACKNOWLEDGED,
        NotificationStatus.FAILED,
        NotificationStatus.CANCELLED,
        NotificationStatus.EXPIRED,
    })

    def __init__(
        self,
        *,
        repository: NotificationRepository,
        preference_gateway: NotificationPreferenceGateway,
        authorization_gateway: NotificationAuthorizationGateway,
        providers: Mapping[
            NotificationChannel,
            NotificationProvider,
        ],
        audit_gateway: NotificationAuditGateway | None = None,
        template_engine: NotificationTemplateEngine | None = None,
        retry_policy: NotificationRetryPolicy | None = None,
    ):
        self.repository = repository
        self.preferences = preference_gateway
        self.authorization = authorization_gateway
        self.providers = dict(providers)
        self.audit = audit_gateway

        self.templates = (
            template_engine
            or NotificationTemplateEngine(
                DEFAULT_TEMPLATES
            )
        )

        self.retry_policy = (
            retry_policy
            or NotificationRetryPolicy()
        )

    async def _audit(
        self,
        action: str,
        notification: Notification,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        # Do not log destination addresses,
        # message contents or sensitive metadata.
        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="notification",
            entity_id=notification.id,
            metadata={
                "district_id": notification.district_id,
                "channel": notification.channel.value,
                "type": notification.notification_type.value,
                "status": notification.status.value,
                "attempts": notification.attempts,
            },
        )

    @staticmethod
    def _resolve_destination(
        recipient: NotificationRecipient,
        channel: NotificationChannel,
    ) -> str:

        if channel in {
            NotificationChannel.SMS,
            NotificationChannel.WHATSAPP,
        }:
            if not recipient.phone_number:
                raise NotificationValidationError(
                    "Verified phone number is required"
                )

            return normalize_phone(
                recipient.phone_number
            )

        if channel == NotificationChannel.EMAIL:
            if not recipient.email_address:
                raise NotificationValidationError(
                    "Email address is required"
                )

            return validate_email(
                recipient.email_address
            )

        if channel == NotificationChannel.PUSH:
            return require_text(
                recipient.push_token or "",
                "Push token",
                max_length=2048,
            )

        if channel == NotificationChannel.IN_APP:
            return recipient.recipient_id

        raise NotificationValidationError(
            "Unsupported notification channel"
        )

    async def get_notification(
        self,
        notification_id: str,
    ) -> Notification:

        notification = (
            await self.repository.get_by_id(
                notification_id
            )
        )

        if notification is None:
            raise NotificationNotFoundError(
                "Notification not found"
            )

        return notification

    async def create_notification(
        self,
        request: NotificationCreateRequest,
        *,
        actor_id: int,
    ) -> Notification:

        recipient = request.recipient

        if not await self.authorization.can_create(
            user_id=actor_id,
            district_id=recipient.district_id,
            notification_type=request.notification_type,
        ):
            raise NotificationAuthorizationError(
                "Notification creation permission denied"
            )

        title = require_text(
            request.title,
            "Title",
            max_length=200,
        )

        message = require_text(
            request.message,
            "Message",
            max_length=10000,
        )

        if not recipient.verified:
            raise NotificationValidationError(
                "Recipient must be verified"
            )

        if request.scheduled_at is not None:
            validate_datetime(
                request.scheduled_at,
                "scheduled_at",
            )

        if request.expires_at is not None:
            validate_datetime(
                request.expires_at,
                "expires_at",
            )

            if request.expires_at <= utc_now():
                raise NotificationValidationError(
                    "Notification expiry must be in the future"
                )

            if (
                request.scheduled_at is not None
                and request.expires_at
                <= request.scheduled_at
            ):
                raise NotificationValidationError(
                    "Expiry must be later than scheduled time"
                )

        if request.idempotency_key:
            existing = (
                await self.repository
                .find_by_idempotency_key(
                    district_id=recipient.district_id,
                    idempotency_key=request.idempotency_key,
                )
            )

            if existing is not None:
                if (
                    existing.recipient_id != recipient.recipient_id
                    or existing.channel != request.channel
                    or existing.notification_type
                    != request.notification_type
                    or existing.title != title
                    or existing.message != message
                ):
                    raise NotificationConflictError(
                        "Idempotency key already used "
                        "for another notification"
                    )

                return existing

        allowed = await self.preferences.is_allowed(
            recipient_id=recipient.recipient_id,
            channel=request.channel,
            notification_type=request.notification_type,
        )

        if not allowed:
            raise NotificationAuthorizationError(
                "Recipient preference or consent "
                "does not permit this notification"
            )

        destination = self._resolve_destination(
            recipient,
            request.channel,
        )

        now = utc_now()

        scheduled_at = request.scheduled_at

        if (
            scheduled_at is not None
            and scheduled_at > now
        ):
            status = NotificationStatus.SCHEDULED
        else:
            status = NotificationStatus.QUEUED

        notification = Notification(
            id=str(uuid.uuid4()),
            district_id=recipient.district_id,
            recipient_id=recipient.recipient_id,
            notification_type=request.notification_type,
            channel=request.channel,
            priority=request.priority,
            title=title,
            message=message,
            language=(
                request.language
                or recipient.preferred_language
            ),
            status=status,
            destination=destination,
            reference_id=request.reference_id,
            idempotency_key=request.idempotency_key,
            requires_acknowledgement=(
                request.requires_acknowledgement
            ),
            created_by=actor_id,
            scheduled_at=scheduled_at,
            next_attempt_at=(
                scheduled_at
                if status == NotificationStatus.SCHEDULED
                else now
            ),
            expires_at=request.expires_at,
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            notification
        )

        await self._audit(
            "notification.created",
            created,
            actor_id,
        )

        return created

    async def create_from_template(
        self,
        *,
        recipient: NotificationRecipient,
        notification_type: NotificationType,
        channel: NotificationChannel,
        priority: NotificationPriority,
        template_id: str,
        values: Mapping[str, str],
        actor_id: int,
        reference_id: str | None = None,
        idempotency_key: str | None = None,
        expires_at: datetime | None = None,
        requires_acknowledgement: bool = False,
    ) -> Notification:

        title, message = self.templates.render(
            template_id=template_id,
            language=recipient.preferred_language,
            values=values,
        )

        request = NotificationCreateRequest(
            recipient=recipient,
            notification_type=notification_type,
            channel=channel,
            priority=priority,
            title=title,
            message=message,
            reference_id=reference_id,
            idempotency_key=idempotency_key,
            expires_at=expires_at,
            requires_acknowledgement=(
                requires_acknowledgement
            ),
        )

        return await self.create_notification(
            request,
            actor_id=actor_id,
        )

    async def create_batch(
        self,
        requests: Sequence[NotificationCreateRequest],
        *,
        actor_id: int,
        max_batch_size: int = 500,
    ) -> list[Notification]:

        if not requests:
            return []

        if len(requests) > max_batch_size:
            raise NotificationValidationError(
                "Notification batch exceeds allowed size"
            )

        results: list[Notification] = []

        # Sequential creation avoids uncontrolled
        # provider or database concurrency.
        #
        # Production bulk operations should use
        # a transactional batch or outbox.
        for request in requests:
            created = await self.create_notification(
                request,
                actor_id=actor_id,
            )

            results.append(created)

        return results

    async def process_notification(
        self,
        notification_id: str,
        *,
        actor_id: int,
    ) -> Notification:

        notification = await self.get_notification(
            notification_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=notification.district_id,
        ):
            raise NotificationAuthorizationError(
                "Notification dispatch permission denied"
            )

        now = utc_now()

        if notification.status not in {
            NotificationStatus.QUEUED,
            NotificationStatus.SCHEDULED,
            NotificationStatus.RETRY_PENDING,
        }:
            raise NotificationConflictError(
                "Notification is not ready for dispatch"
            )

        if (
            notification.next_attempt_at is not None
            and notification.next_attempt_at > now
        ):
            raise NotificationConflictError(
                "Notification is not due yet"
            )

        if (
            notification.expires_at is not None
            and notification.expires_at <= now
        ):
            expired = replace(
                notification,
                status=NotificationStatus.EXPIRED,
                updated_at=now,
                version=notification.version + 1,
            )

            saved = await self.repository.update(
                expired,
                expected_version=notification.version,
            )

            await self._audit(
                "notification.expired",
                saved,
                actor_id,
            )

            return saved

        # Recheck consent at delivery time.
        allowed = await self.preferences.is_allowed(
            recipient_id=notification.recipient_id,
            channel=notification.channel,
            notification_type=notification.notification_type,
        )

        if not allowed:
            cancelled = replace(
                notification,
                status=NotificationStatus.CANCELLED,
                updated_at=now,
                version=notification.version + 1,
            )

            saved = await self.repository.update(
                cancelled,
                expected_version=notification.version,
            )

            await self._audit(
                "notification.cancelled_by_preference",
                saved,
                actor_id,
            )

            return saved

        provider = self.providers.get(
            notification.channel
        )

        if provider is None:
            raise NotificationProviderError(
                "Notification provider is not configured",
                failure_type=(
                    DeliveryFailureType.PROVIDER_UNAVAILABLE
                ),
            )

        # Optimistic update claims the notification.
        # Production queue workers should use leases
        # and recover abandoned PROCESSING records.
        processing = replace(
            notification,
            status=NotificationStatus.PROCESSING,
            attempts=notification.attempts + 1,
            updated_at=now,
            version=notification.version + 1,
        )

        claimed = await self.repository.update(
            processing,
            expected_version=notification.version,
        )

        try:
            result = await provider.send(
                destination=claimed.destination,
                title=claimed.title,
                message=claimed.message,
                idempotency_key=claimed.id,
                metadata={
                    "notification_id": claimed.id,
                    "notification_type": (
                        claimed.notification_type.value
                    ),
                    "priority": claimed.priority.value,
                    "language": claimed.language.value,
                },
            )

        except NotificationProviderError as exc:
            result = ProviderSendResult(
                accepted=False,
                failure_type=exc.failure_type,
                error_message=str(exc),
                retry_after_seconds=(
                    exc.retry_after_seconds
                ),
            )

        except Exception:
            logger.exception(
                "Notification provider failed: %s",
                claimed.id,
            )

            result = ProviderSendResult(
                accepted=False,
                failure_type=(
                    DeliveryFailureType.TRANSIENT
                ),
                error_message=(
                    "Unexpected provider failure"
                ),
            )

        finished_at = utc_now()

        if result.accepted:
            new_status = (
                NotificationStatus.DELIVERED
                if result.delivered
                else NotificationStatus.SENT
            )

            updated = replace(
                claimed,
                status=new_status,
                provider_message_id=(
                    result.provider_message_id
                ),
                sent_at=finished_at,
                delivered_at=(
                    finished_at
                    if result.delivered
                    else None
                ),
                next_attempt_at=None,
                last_error=None,
                updated_at=finished_at,
                version=claimed.version + 1,
            )

        else:
            failure_type = (
                result.failure_type
                or DeliveryFailureType.TRANSIENT
            )

            should_retry = (
                self.retry_policy.should_retry(
                    failure_type=failure_type,
                    attempts=claimed.attempts,
                    max_attempts=claimed.max_attempts,
                )
            )

            next_attempt = (
                self.retry_policy.next_attempt(
                    attempts=claimed.attempts,
                    retry_after_seconds=(
                        result.retry_after_seconds
                    ),
                    now=finished_at,
                )
                if should_retry
                else None
            )

            if (
                next_attempt is not None
                and claimed.expires_at is not None
                and next_attempt >= claimed.expires_at
            ):
                should_retry = False
                next_attempt = None

            updated = replace(
                claimed,
                status=(
                    NotificationStatus.RETRY_PENDING
                    if should_retry
                    else NotificationStatus.FAILED
                ),
                next_attempt_at=next_attempt,
                last_error=(
                    result.error_message
                    or failure_type.value
                )[:500],
                updated_at=finished_at,
                version=claimed.version + 1,
            )

        saved = await self.repository.update(
            updated,
            expected_version=claimed.version,
        )

        await self._audit(
            f"notification.{saved.status.value}",
            saved,
            actor_id,
        )

        return saved

    async def process_due_notifications(
        self,
        *,
        actor_id: int,
        limit: int = 100,
    ) -> dict[str, int]:

        if not 1 <= limit <= 1000:
            raise NotificationValidationError(
                "Limit must be between 1 and 1000"
            )

        due = await self.repository.list_due(
            now=utc_now(),
            limit=limit,
        )

        results = {
            "processed": 0,
            "sent": 0,
            "delivered": 0,
            "retry_pending": 0,
            "failed": 0,
            "expired": 0,
            "skipped": 0,
        }

        for notification in due:
            try:
                updated = await self.process_notification(
                    notification.id,
                    actor_id=actor_id,
                )

                results["processed"] += 1

                if updated.status == NotificationStatus.SENT:
                    results["sent"] += 1

                elif (
                    updated.status
                    == NotificationStatus.DELIVERED
                ):
                    results["delivered"] += 1

                elif (
                    updated.status
                    == NotificationStatus.RETRY_PENDING
                ):
                    results["retry_pending"] += 1

                elif (
                    updated.status
                    == NotificationStatus.FAILED
                ):
                    results["failed"] += 1

                elif (
                    updated.status
                    == NotificationStatus.EXPIRED
                ):
                    results["expired"] += 1

            except (
                NotificationConflictError,
                NotificationAuthorizationError,
            ):
                results["skipped"] += 1

            except Exception:
                logger.exception(
                    "Notification processing error: %s",
                    notification.id,
                )

                results["failed"] += 1

        return results

    async def record_delivery_receipt(
        self,
        receipt: NotificationDeliveryReceipt,
    ) -> Notification:

        notification = await self.get_notification(
            receipt.notification_id
        )

        if (
            notification.provider_message_id
            != receipt.provider_message_id
        ):
            raise NotificationConflictError(
                "Provider message ID mismatch"
            )

        # IMPORTANT:
        # Provider webhooks must be authenticated and
        # replay-protected at the API/integration layer.

        if notification.status in {
            NotificationStatus.CANCELLED,
            NotificationStatus.EXPIRED,
        }:
            raise NotificationConflictError(
                "Notification is no longer active"
            )

        if notification.status in {
            NotificationStatus.DELIVERED,
            NotificationStatus.READ,
            NotificationStatus.ACKNOWLEDGED,
        }:
            return notification

        if notification.status != NotificationStatus.SENT:
            raise NotificationConflictError(
                "Notification has not been recorded as sent"
            )

        updated = replace(
            notification,
            status=(
                NotificationStatus.DELIVERED
                if receipt.delivered
                else NotificationStatus.FAILED
            ),
            delivered_at=(
                receipt.occurred_at
                if receipt.delivered
                else None
            ),
            last_error=(
                None
                if receipt.delivered
                else receipt.error_message
            ),
            updated_at=utc_now(),
            version=notification.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=notification.version,
        )

        await self._audit(
            "notification.delivery_receipt",
            saved,
            None,
        )

        return saved

    async def mark_as_read(
        self,
        notification_id: str,
        *,
        recipient_id: str,
    ) -> Notification:

        notification = await self.get_notification(
            notification_id
        )

        if notification.recipient_id != recipient_id:
            raise NotificationAuthorizationError(
                "Recipient mismatch"
            )

        if notification.status in {
            NotificationStatus.READ,
            NotificationStatus.ACKNOWLEDGED,
        }:
            return notification

        if notification.status != NotificationStatus.DELIVERED:
            raise NotificationConflictError(
                "Notification is not delivered"
            )

        updated = replace(
            notification,
            status=NotificationStatus.READ,
            read_at=utc_now(),
            updated_at=utc_now(),
            version=notification.version + 1,
        )

        return await self.repository.update(
            updated,
            expected_version=notification.version,
        )

    async def acknowledge_notification(
        self,
        notification_id: str,
        *,
        recipient_id: str,
    ) -> Notification:

        notification = await self.get_notification(
            notification_id
        )

        if notification.recipient_id != recipient_id:
            raise NotificationAuthorizationError(
                "Recipient mismatch"
            )

        if not notification.requires_acknowledgement:
            raise NotificationConflictError(
                "Acknowledgement is not required"
            )

        if (
            notification.status
            == NotificationStatus.ACKNOWLEDGED
        ):
            return notification

        if notification.status not in {
            NotificationStatus.DELIVERED,
            NotificationStatus.READ,
        }:
            raise NotificationConflictError(
                "Notification cannot be acknowledged yet"
            )

        updated = replace(
            notification,
            status=NotificationStatus.ACKNOWLEDGED,
            acknowledged_at=utc_now(),
            updated_at=utc_now(),
            version=notification.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=notification.version,
        )

        await self._audit(
            "notification.acknowledged",
            saved,
            None,
        )

        return saved

    async def cancel_notification(
        self,
        notification_id: str,
        *,
        actor_id: int,
    ) -> Notification:

        notification = await self.get_notification(
            notification_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=notification.district_id,
        ):
            raise NotificationAuthorizationError(
                "Notification cancellation permission denied"
            )

        if notification.status not in {
            NotificationStatus.DRAFT,
            NotificationStatus.SCHEDULED,
            NotificationStatus.QUEUED,
            NotificationStatus.RETRY_PENDING,
        }:
            raise NotificationConflictError(
                "Notification can no longer be cancelled"
            )

        updated = replace(
            notification,
            status=NotificationStatus.CANCELLED,
            next_attempt_at=None,
            updated_at=utc_now(),
            version=notification.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=notification.version,
        )

        await self._audit(
            "notification.cancelled",
            saved,
            actor_id,
        )

        return saved

    async def list_recipient_notifications(
        self,
        recipient_id: str,
    ) -> Sequence[Notification]:

        require_text(
            recipient_id,
            "Recipient ID",
        )

        # API layer must authenticate the recipient
        # or authorize privileged access.
        return await self.repository.list_by_recipient(
            recipient_id
        )

    async def list_district_notifications(
        self,
        district_id: int,
    ) -> Sequence[Notification]:

        if district_id <= 0:
            raise NotificationValidationError(
                "Invalid district ID"
            )

        # API layer must enforce district-scoped access.
        return await self.repository.list_by_district(
            district_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        notifications = (
            await self.list_district_notifications(
                district_id
            )
        )

        status_counts = {
            status.value: 0
            for status in NotificationStatus
        }

        channel_counts = {
            channel.value: 0
            for channel in NotificationChannel
        }

        priority_counts = {
            priority.value: 0
            for priority in NotificationPriority
        }

        for notification in notifications:
            status_counts[
                notification.status.value
            ] += 1

            channel_counts[
                notification.channel.value
            ] += 1

            priority_counts[
                notification.priority.value
            ] += 1

        return {
            "district_id": district_id,
            "total_notifications": len(notifications),
            "status_counts": status_counts,
            "channel_counts": channel_counts,
            "priority_counts": priority_counts,
            "total_attempts": sum(
                notification.attempts
                for notification in notifications
            ),
            "generated_at": utc_now().isoformat(),
        }

    async def generate_notification_report(
        self,
        notification_id: str,
    ) -> dict[str, Any]:

        notification = await self.get_notification(
            notification_id
        )

        return {
            "notification_id": notification.id,
            "district_id": notification.district_id,
            "notification_type": (
                notification.notification_type.value
            ),
            "channel": notification.channel.value,
            "priority": notification.priority.value,
            "language": notification.language.value,
            "status": notification.status.value,
            "attempts": notification.attempts,
            "max_attempts": notification.max_attempts,
            "requires_acknowledgement": (
                notification.requires_acknowledgement
            ),
            "reference_id": notification.reference_id,
            "created_at": (
                notification.created_at.isoformat()
            ),
            "sent_at": (
                notification.sent_at.isoformat()
                if notification.sent_at
                else None
            ),
            "delivered_at": (
                notification.delivered_at.isoformat()
                if notification.delivered_at
                else None
            ),
            "acknowledged_at": (
                notification.acknowledged_at.isoformat()
                if notification.acknowledged_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Provider acceptance is not proof of "
                "recipient delivery or acknowledgement. "
                "Operational emergency communication "
                "requires verified delivery monitoring "
                "and authorized escalation procedures."
            ),
        }


# ============================================================
# OPTIONAL PROVIDER ADAPTER EXAMPLE
# ============================================================

class InAppNotificationProvider:
    """
    Minimal in-app provider adapter.

    This adapter acknowledges acceptance only.
    Actual in-app persistence and delivery must
    be implemented in the notification repository
    or a dedicated inbox gateway.
    """

    async def send(
        self,
        *,
        destination: str,
        title: str,
        message: str,
        idempotency_key: str,
        metadata: Mapping[str, Any],
    ) -> ProviderSendResult:

        require_text(
            destination,
            "Recipient ID",
        )

        require_text(
            title,
            "Title",
        )

        require_text(
            message,
            "Message",
        )

        return ProviderSendResult(
            accepted=True,
            provider_message_id=(
                f"inapp-{idempotency_key}"
            ),
            delivered=False,
        )
