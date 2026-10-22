from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/workers/notification_worker.py

Notification background worker.

Features:
- Email notifications (SMTP)
- SMS notifications (provider adapter)
- WhatsApp notifications (provider adapter)
- Push notifications (provider adapter)
- Flood warning notifications
- SOS acknowledgments
- Rescue assignment notifications
- District-targeted notifications
- Celery asynchronous processing
- Redis idempotency and tracking
- Retry with exponential backoff
- Delivery status and audit events
- Human authorization for public alerts

IMPORTANT:
Provider acceptance is not proof of recipient delivery.
Official public emergency warnings must be authorized
and sent only through approved operational channels.
"""


import json
import logging
import os
import smtplib
import ssl
import uuid

from datetime import datetime, timezone
from email.message import EmailMessage
from enum import Enum
from typing import Any

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

celery_app = Celery(
    "kerala_flood_notification_worker",
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
)

redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)

NOTIFICATION_TTL = 60 * 60 * 24 * 30
MAX_RETRIES = 3


# ============================================================
# ENUMS
# ============================================================

class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    PUSH = "push"


class NotificationType(str, Enum):
    FLOOD_ALERT = "flood_alert"
    SOS_ACKNOWLEDGMENT = "sos_acknowledgment"
    RESCUE_ASSIGNMENT = "rescue_assignment"
    RESCUE_STATUS = "rescue_status"
    EVACUATION_NOTICE = "evacuation_notice"
    SHELTER_UPDATE = "shelter_update"
    MEDICAL_EMERGENCY = "medical_emergency"
    VOLUNTEER_ASSIGNMENT = "volunteer_assignment"
    RELIEF_DISTRIBUTION = "relief_distribution"
    SYSTEM = "system"


class NotificationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    DELIVERED = "delivered"
    FAILED = "failed"
    BLOCKED = "blocked"


# ============================================================
# SCHEMAS
# ============================================================

class NotificationRequest(BaseModel):
    notification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    notification_type: NotificationType
    channel: NotificationChannel

    recipient: str = Field(min_length=1)
    subject: str = Field(default="", max_length=200)
    message: str = Field(min_length=1, max_length=5000)

    district: str | None = None
    reference_id: str | None = None

    is_public_warning: bool = False

    approved_by: str | None = None
    approval_reference: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("recipient")
    @classmethod
    def clean_recipient(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def check_public_warning(self):
        if self.is_public_warning:
            if not self.approved_by or not self.approval_reference:
                raise ValueError(
                    "Public warnings require verified "
                    "human authorization and approval reference"
                )

        return self


class DeliveryResult(BaseModel):
    notification_id: str
    channel: NotificationChannel
    status: NotificationStatus

    provider: str | None = None
    provider_message_id: str | None = None

    error: str | None = None

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ============================================================
# UTILITIES
# ============================================================

def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def notification_key(notification_id: str) -> str:
    return f"notifications:item:{notification_id}"


def lock_key(notification_id: str) -> str:
    return f"notifications:lock:{notification_id}"


def save_notification(
    notification: NotificationRequest,
    status: NotificationStatus,
) -> None:
    payload = notification.model_dump(mode="json")
    payload["status"] = status.value
    payload["updated_at"] = utc_iso()

    redis_client.set(
        notification_key(notification.notification_id),
        json.dumps(payload),
        ex=NOTIFICATION_TTL,
    )


def get_notification(
    notification_id: str,
) -> dict[str, Any] | None:
    raw = redis_client.get(
        notification_key(notification_id)
    )

    return json.loads(raw) if raw else None


def update_notification(
    notification_id: str,
    status: NotificationStatus,
    **extra: Any,
) -> None:
    key = notification_key(notification_id)
    raw = redis_client.get(key)

    if not raw:
        raise ValueError(
            f"Notification not found: {notification_id}"
        )

    payload = json.loads(raw)
    payload["status"] = status.value
    payload["updated_at"] = utc_iso()
    payload.update(extra)

    redis_client.set(
        key,
        json.dumps(payload, default=str),
        ex=NOTIFICATION_TTL,
    )


# ============================================================
# AUDIT LOGGING
# ============================================================

def audit_event(
    notification_id: str,
    event: str,
    details: dict[str, Any] | None = None,
) -> None:
    record = {
        "notification_id": notification_id,
        "event": event,
        "timestamp": utc_iso(),
        "details": details or {},
    }

    redis_client.xadd(
        "notifications:audit",
        {"record": json.dumps(record, default=str)},
        maxlen=100000,
        approximate=True,
    )

    logger.info(
        "Notification audit: id=%s event=%s",
        notification_id,
        event,
    )


# ============================================================
# PROVIDER ERRORS
# ============================================================

class ProviderUnavailableError(RuntimeError):
    pass


class TemporaryDeliveryError(RuntimeError):
    pass


# ============================================================
# EMAIL PROVIDER
# ============================================================

def send_email(
    notification: NotificationRequest,
) -> DeliveryResult:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "465"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM")

    if not all([
        host,
        username,
        password,
        sender,
    ]):
        raise ProviderUnavailableError(
            "SMTP configuration is incomplete"
        )

    if "@" not in notification.recipient:
        raise ValueError(
            "Invalid email recipient"
        )

    message = EmailMessage()

    message["From"] = sender
    message["To"] = notification.recipient
    message["Subject"] = (
        notification.subject or "Kerala Flood Relief Update"
    )

    message.set_content(notification.message)

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL(
            host,
            port,
            context=context,
            timeout=20,
        ) as server:
            server.login(username, password)
            server.send_message(message)

    except (
        smtplib.SMTPException,
        OSError,
    ) as exc:
        raise TemporaryDeliveryError(
            "SMTP delivery attempt failed"
        ) from exc

    return DeliveryResult(
        notification_id=notification.notification_id,
        channel=NotificationChannel.EMAIL,
        status=NotificationStatus.ACCEPTED,
        provider="smtp",
    )


# ============================================================
# SMS PROVIDER
# ============================================================

def send_sms(
    notification: NotificationRequest,
) -> DeliveryResult:
    """
    Integrate a verified SMS provider in:
        integrations/sms_adapter.py

    Examples:
        - Authorized SMS gateway
        - Registered enterprise SMS provider

    Production deployments must comply with
    applicable Indian telecom requirements.
    """

    raise ProviderUnavailableError(
        "SMS adapter is not configured"
    )


# ============================================================
# WHATSAPP PROVIDER
# ============================================================

def send_whatsapp(
    notification: NotificationRequest,
) -> DeliveryResult:
    """
    Integrate an authorized WhatsApp Business
    messaging provider with approved templates
    and recipient permissions.
    """

    raise ProviderUnavailableError(
        "WhatsApp adapter is not configured"
    )


# ============================================================
# PUSH PROVIDER
# ============================================================

def send_push(
    notification: NotificationRequest,
) -> DeliveryResult:
    """
    Integrate Firebase Cloud Messaging or another
    authenticated push notification provider.
    """

    raise ProviderUnavailableError(
        "Push notification adapter is not configured"
    )


# ============================================================
# CHANNEL ROUTER
# ============================================================

def deliver_notification(
    notification: NotificationRequest,
) -> DeliveryResult:

    handlers = {
        NotificationChannel.EMAIL: send_email,
        NotificationChannel.SMS: send_sms,
        NotificationChannel.WHATSAPP: send_whatsapp,
        NotificationChannel.PUSH: send_push,
    }

    handler = handlers.get(notification.channel)

    if handler is None:
        raise ProviderUnavailableError(
            "Unsupported notification channel"
        )

    return handler(notification)


# ============================================================
# CELERY SEND TASK
# ============================================================

@celery_app.task(
    bind=True,
    name="notifications.send",
    max_retries=MAX_RETRIES,
    default_retry_delay=30,
)
def send_notification(
    self,
    notification_data: dict[str, Any],
) -> dict[str, Any]:

    notification = NotificationRequest.model_validate(
        notification_data
    )

    notification_id = notification.notification_id

    lock = redis_client.lock(
        lock_key(notification_id),
        timeout=90,
        blocking_timeout=5,
    )

    acquired = False

    try:
        acquired = lock.acquire(blocking=True)

        if not acquired:
            raise TemporaryDeliveryError(
                "Notification is currently locked"
            )

        existing = get_notification(notification_id)

        if existing and existing.get("status") in {
            NotificationStatus.ACCEPTED.value,
            NotificationStatus.DELIVERED.value,
        }:
            return {
                "notification_id": notification_id,
                "status": existing["status"],
                "duplicate": True,
            }

        # An approval reference in the request is
        # not sufficient evidence of authorization.
        # The trusted API must verify authorization
        # against a durable approval record before
        # enqueuing any public warning.
        if notification.is_public_warning:
            raise PermissionError(
                "Public warning delivery requires "
                "server-side approval verification"
            )

        if existing is None:
            save_notification(
                notification,
                NotificationStatus.PENDING,
            )

        update_notification(
            notification_id,
            NotificationStatus.PROCESSING,
        )

        result = deliver_notification(notification)

        update_notification(
            notification_id,
            result.status,
            provider=result.provider,
            provider_message_id=result.provider_message_id,
        )

        audit_event(
            notification_id,
            "provider_accepted",
            {
                "channel": notification.channel.value,
                "provider": result.provider,
            },
        )

        return result.model_dump(mode="json")

    except ProviderUnavailableError as exc:
        logger.error(
            "Provider unavailable: %s",
            exc,
        )

        if get_notification(notification_id):
            update_notification(
                notification_id,
                NotificationStatus.BLOCKED,
                error=str(exc),
            )

        return {
            "notification_id": notification_id,
            "status": NotificationStatus.BLOCKED.value,
            "error": str(exc),
        }

    except PermissionError as exc:
        if get_notification(notification_id):
            update_notification(
                notification_id,
                NotificationStatus.BLOCKED,
                error=str(exc),
            )

        return {
            "notification_id": notification_id,
            "status": NotificationStatus.BLOCKED.value,
            "error": str(exc),
        }

    except (
        TemporaryDeliveryError,
        RedisError,
    ) as exc:
        logger.exception(
            "Temporary notification failure"
        )

        countdown = min(
            30 * (2 ** self.request.retries),
            300,
        )

        raise self.retry(
            exc=exc,
            countdown=countdown,
        )

    finally:
        if acquired:
            try:
                lock.release()
            except RedisError:
                logger.exception(
                    "Failed to release notification lock"
                )


# ============================================================
# SOS ACKNOWLEDGMENT
# ============================================================

@celery_app.task(
    name="notifications.sos_acknowledgment",
)
def sos_acknowledgment(
    request_id: str,
    recipient: str,
    channel: str = "sms",
) -> dict[str, Any]:

    notification = NotificationRequest(
        notification_type=(
            NotificationType.SOS_ACKNOWLEDGMENT
        ),
        channel=NotificationChannel(channel),
        recipient=recipient,
        subject="SOS Request Received",
        message=(
            f"Your SOS request {request_id} has been "
            "recorded in the system. This does not "
            "confirm that a rescue team has been "
            "assigned or dispatched."
        ),
        reference_id=request_id,
    )

    task = send_notification.apply_async(
        args=[notification.model_dump(mode="json")]
    )

    return {
        "notification_id": notification.notification_id,
        "task_id": task.id,
        "status": "queued",
    }


# ============================================================
# RESCUE STATUS NOTIFICATION
# ============================================================

@celery_app.task(
    name="notifications.rescue_status",
)
def rescue_status_notification(
    request_id: str,
    recipient: str,
    status_text: str,
    channel: str = "sms",
) -> dict[str, Any]:

    notification = NotificationRequest(
        notification_type=NotificationType.RESCUE_STATUS,
        channel=NotificationChannel(channel),
        recipient=recipient,
        subject="Rescue Request Update",
        message=(
            f"Rescue request {request_id}: "
            f"{status_text}"
        ),
        reference_id=request_id,
    )

    task = send_notification.apply_async(
        args=[notification.model_dump(mode="json")]
    )

    return {
        "notification_id": notification.notification_id,
        "task_id": task.id,
        "status": "queued",
    }


# ============================================================
# FLOOD ALERT NOTIFICATION
# ============================================================

@celery_app.task(
    name="notifications.flood_alert",
)
def flood_alert_notification(
    alert_id: str,
    district: str,
    recipient: str,
    message: str,
    approved_by: str,
    approval_reference: str,
    channel: str = "sms",
) -> dict[str, Any]:
    """
    Prepare an authorized public-warning request.

    The send task deliberately blocks public warnings
    until a trusted approval-verification integration
    has been implemented.
    """

    notification = NotificationRequest(
        notification_type=NotificationType.FLOOD_ALERT,
        channel=NotificationChannel(channel),
        recipient=recipient,
        subject=f"Flood Alert - {district}",
        message=message,
        district=district,
        reference_id=alert_id,
        is_public_warning=True,
        approved_by=approved_by,
        approval_reference=approval_reference,
    )

    task = send_notification.apply_async(
        args=[notification.model_dump(mode="json")]
    )

    return {
        "notification_id": notification.notification_id,
        "task_id": task.id,
        "status": "queued_for_verification",
        "delivery_enabled": False,
    }


# ============================================================
# DELIVERY STATUS
# ============================================================

@celery_app.task(
    name="notifications.get_status",
)
def get_notification_status(
    notification_id: str,
) -> dict[str, Any]:

    record = get_notification(notification_id)

    if record is None:
        return {
            "notification_id": notification_id,
            "status": "not_found",
        }

    return {
        "notification_id": notification_id,
        "status": record["status"],
        "channel": record["channel"],
        "updated_at": record.get("updated_at"),
        "provider": record.get("provider"),
    }


# ============================================================
# WORKER HEALTH
# ============================================================

@celery_app.task(
    name="notifications.worker_health",
)
def worker_health() -> dict[str, Any]:

    redis_client.ping()

    return {
        "worker": "notification_worker",
        "status": "healthy",
        "timestamp": utc_iso(),
        "channels": [
            channel.value
            for channel in NotificationChannel
        ],
    }
