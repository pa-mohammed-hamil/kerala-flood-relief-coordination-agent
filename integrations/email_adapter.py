"""Placeholder module: integrations/email_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Email Notification Integration

File: backend/integrations/email_adapter.py

Python 3.11+
Dependencies: Python standard library only.

Features:
    - SMTP email delivery
    - SMTP SSL and STARTTLS
    - Username/password authentication
    - Plain-text and HTML emails
    - Emergency SOS notifications
    - Flood alert notifications
    - Rescue coordination updates
    - Shelter status notifications
    - Volunteer notifications
    - Relief inventory alerts
    - Missing-person case notifications
    - Delivery retries
    - Exponential backoff
    - Recipient validation
    - Email templates
    - Batch notifications
    - Async sending
    - Attachment support
    - Delivery audit events
    - Idempotency keys
    - Rate limiting
    - Safe error reporting
    - Human approval for emergency broadcasts

Safety:
    SMTP acceptance is not proof of inbox delivery.

    SOS notifications must never depend solely
    on email. Emergency workflows require
    approved real-time escalation channels.

    Emergency broadcasts require explicit
    human approval.

    Avoid including unnecessary sensitive
    personal or medical information in emails.
"""

from __future__ import annotations

import asyncio
import html
import logging
import mimetypes
import random
import re
import smtplib
import ssl
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================
# EXCEPTIONS
# ============================================================

class EmailAdapterError(Exception):
    pass


class EmailConfigurationError(EmailAdapterError):
    pass


class EmailValidationError(EmailAdapterError):
    pass


class EmailAuthenticationError(EmailAdapterError):
    pass


class EmailDeliveryError(EmailAdapterError):
    pass


class EmailRateLimitError(EmailAdapterError):
    pass


class EmailApprovalError(EmailAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class EmailSecurityMode(str, Enum):
    SSL = "ssl"
    STARTTLS = "starttls"


class EmailPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EmailNotificationType(str, Enum):
    GENERAL = "general"
    FLOOD_ALERT = "flood_alert"
    SOS_REQUEST = "sos_request"
    RESCUE_UPDATE = "rescue_update"
    SHELTER_UPDATE = "shelter_update"
    VOLUNTEER_UPDATE = "volunteer_update"
    RELIEF_INVENTORY = "relief_inventory"
    MISSING_PERSON = "missing_person"
    MEDICAL_COORDINATION = "medical_coordination"
    RECOVERY_UPDATE = "recovery_update"
    SYSTEM_ALERT = "system_alert"


class EmailDeliveryStatus(str, Enum):
    ACCEPTED = "accepted"
    FAILED = "failed"
    SKIPPED_DUPLICATE = "skipped_duplicate"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+$"
)


def validate_email_address(
    address: str,
) -> str:
    if not isinstance(address, str):
        raise EmailValidationError(
            "Email address must be a string"
        )

    address = address.strip()

    display_name, parsed_address = parseaddr(
        address
    )

    if (
        display_name
        or parsed_address != address
        or len(address) > 254
        or not EMAIL_PATTERN.fullmatch(address)
        or "\r" in address
        or "\n" in address
    ):
        raise EmailValidationError(
            "Invalid email address"
        )

    return address


def validate_recipients(
    recipients: Sequence[str],
    *,
    maximum: int,
) -> tuple[str, ...]:
    if not recipients:
        raise EmailValidationError(
            "At least one recipient is required"
        )

    if len(recipients) > maximum:
        raise EmailValidationError(
            "Recipient limit exceeded"
        )

    unique = []
    seen = set()

    for recipient in recipients:
        validated = validate_email_address(
            recipient
        )

        key = validated.casefold()

        if key not in seen:
            seen.add(key)
            unique.append(validated)

    return tuple(unique)


def safe_subject(
    value: str,
    *,
    maximum: int = 200,
) -> str:
    if not isinstance(value, str):
        raise EmailValidationError(
            "Subject must be a string"
        )

    value = value.strip()

    if (
        not value
        or len(value) > maximum
        or "\r" in value
        or "\n" in value
    ):
        raise EmailValidationError(
            "Invalid email subject"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class EmailAdapterConfig:
    smtp_host: str
    smtp_port: int

    sender_email: str

    sender_name: str = (
        "Kerala Flood Relief Coordination"
    )

    security_mode: EmailSecurityMode = (
        EmailSecurityMode.STARTTLS
    )

    smtp_username: str | None = None
    smtp_password: str | None = None

    timeout_seconds: float = 15.0

    max_retries: int = 3
    retry_base_seconds: float = 1.0

    max_recipients: int = 50
    max_attachment_bytes: int = 5_000_000
    max_message_bytes: int = 10_000_000

    rate_limit_per_minute: int = 60

    require_approval_for_critical: bool = True

    require_approval_for_broadcast: bool = True

    allow_attachments: bool = False

    def __post_init__(self) -> None:
        if not self.smtp_host.strip():
            raise EmailConfigurationError(
                "smtp_host required"
            )

        if not 1 <= self.smtp_port <= 65535:
            raise EmailConfigurationError(
                "Invalid SMTP port"
            )

        validate_email_address(
            self.sender_email
        )

        if (
            self.smtp_username is None
        ) != (
            self.smtp_password is None
        ):
            raise EmailConfigurationError(
                "SMTP username and password "
                "must be configured together"
            )

        if self.timeout_seconds <= 0:
            raise EmailConfigurationError(
                "timeout_seconds must be positive"
            )

        if self.max_retries < 0:
            raise EmailConfigurationError(
                "max_retries cannot be negative"
            )

        for name in (
            "max_recipients",
            "max_attachment_bytes",
            "max_message_bytes",
            "rate_limit_per_minute",
        ):
            if getattr(self, name) <= 0:
                raise EmailConfigurationError(
                    f"{name} must be positive"
                )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class EmailAttachment:
    filename: str
    content: bytes
    content_type: str = (
        "application/octet-stream"
    )

    def __post_init__(self) -> None:
        if (
            not self.filename
            or Path(self.filename).name
            != self.filename
            or "\r" in self.filename
            or "\n" in self.filename
        ):
            raise EmailValidationError(
                "Invalid attachment filename"
            )

        if not isinstance(
            self.content,
            bytes,
        ):
            raise EmailValidationError(
                "Attachment content must be bytes"
            )


@dataclass(frozen=True)
class EmailApproval:
    approved: bool

    approved_by: str | None = None

    approval_reference: str | None = None

    approved_at: datetime | None = None


@dataclass(frozen=True)
class EmailRequest:
    recipients: tuple[str, ...]

    subject: str
    text_body: str

    html_body: str | None = None

    notification_type: EmailNotificationType = (
        EmailNotificationType.GENERAL
    )

    priority: EmailPriority = (
        EmailPriority.NORMAL
    )

    tenant_id: str = "default"

    idempotency_key: str | None = None

    approval: EmailApproval | None = None

    attachments: tuple[
        EmailAttachment,
        ...
    ] = ()

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    request_id: str = field(
        default_factory=generate_id
    )

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise EmailValidationError(
                "tenant_id required"
            )

        if not self.text_body.strip():
            raise EmailValidationError(
                "text_body required"
            )

        safe_subject(
            self.subject
        )


@dataclass(frozen=True)
class EmailDeliveryResult:
    request_id: str

    status: EmailDeliveryStatus

    accepted_recipients: tuple[str, ...]

    attempts: int

    message_id: str | None

    timestamp: datetime

    error_code: str | None = None

    idempotency_key: str | None = None


@dataclass(frozen=True)
class EmailAuditEvent:
    event_id: str
    request_id: str

    tenant_id: str

    event_type: str

    timestamp: datetime

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# AUDIT LOGGER
# ============================================================

class EmailAuditLogger:
    def __init__(self) -> None:
        self._events: list[
            EmailAuditEvent
        ] = []

        self._lock = threading.RLock()

    def record(
        self,
        request: EmailRequest,
        event_type: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        event = EmailAuditEvent(
            event_id=generate_id(),
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            event_type=event_type,
            timestamp=utc_now(),
            metadata=metadata or {},
        )

        with self._lock:
            self._events.append(
                event
            )

    def list_events(
        self,
        *,
        tenant_id: str,
    ) -> list[EmailAuditEvent]:
        with self._lock:
            return [
                event
                for event in self._events
                if event.tenant_id == tenant_id
            ]


# ============================================================
# RATE LIMITER
# ============================================================

class EmailRateLimiter:
    def __init__(
        self,
        limit_per_minute: int,
    ) -> None:
        self.limit = limit_per_minute

        self._timestamps: dict[
            str,
            deque[float],
        ] = {}

        self._lock = threading.RLock()

    def acquire(
        self,
        tenant_id: str,
    ) -> None:
        now = time.monotonic()

        with self._lock:
            queue = self._timestamps.setdefault(
                tenant_id,
                deque(),
            )

            while (
                queue
                and now - queue[0] >= 60
            ):
                queue.popleft()

            if len(queue) >= self.limit:
                raise EmailRateLimitError(
                    "Email rate limit exceeded"
                )

            queue.append(now)


# ============================================================
# IDEMPOTENCY STORE
# ============================================================

class EmailIdempotencyStore:
    """
    In-memory idempotency cache.

    For multi-worker deployments, replace
    with a transactional shared datastore.
    """

    def __init__(self) -> None:
        self._results: dict[
            tuple[str, str],
            EmailDeliveryResult,
        ] = {}

        self._lock = threading.RLock()

    def get(
        self,
        tenant_id: str,
        key: str,
    ) -> EmailDeliveryResult | None:
        with self._lock:
            return self._results.get(
                (
                    tenant_id,
                    key,
                )
            )

    def save(
        self,
        tenant_id: str,
        key: str,
        result: EmailDeliveryResult,
    ) -> None:
        with self._lock:
            self._results[
                (
                    tenant_id,
                    key,
                )
            ] = result


# ============================================================
# APPROVAL POLICY
# ============================================================

class EmailApprovalPolicy:
    BROADCAST_TYPES = {
        EmailNotificationType.FLOOD_ALERT,
        EmailNotificationType.SOS_REQUEST,
    }

    def __init__(
        self,
        config: EmailAdapterConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        request: EmailRequest,
    ) -> None:
        requires_approval = False

        if (
            self.config.require_approval_for_critical
            and request.priority
            == EmailPriority.CRITICAL
        ):
            requires_approval = True

        if (
            self.config.require_approval_for_broadcast
            and request.notification_type
            in self.BROADCAST_TYPES
        ):
            requires_approval = True

        if not requires_approval:
            return

        approval = request.approval

        if (
            approval is None
            or not approval.approved
            or not approval.approved_by
            or not approval.approval_reference
            or approval.approved_at is None
        ):
            raise EmailApprovalError(
                "Verified human approval required"
            )

        if (
            approval.approved_at.tzinfo is None
            or approval.approved_at > utc_now()
        ):
            raise EmailApprovalError(
                "Invalid approval timestamp"
            )

        # Production integration must verify the
        # approval reference against a trusted,
        # tenant-scoped authorization service.
        # A supplied approval object alone is
        # not proof of authorization.


# ============================================================
# MESSAGE BUILDER
# ============================================================

class EmailMessageBuilder:
    def __init__(
        self,
        config: EmailAdapterConfig,
    ) -> None:
        self.config = config

    def build(
        self,
        request: EmailRequest,
    ) -> EmailMessage:
        recipients = validate_recipients(
            request.recipients,
            maximum=self.config.max_recipients,
        )

        message = EmailMessage()

        message["From"] = formataddr(
            (
                self.config.sender_name,
                self.config.sender_email,
            )
        )

        message["To"] = ", ".join(
            recipients
        )

        message["Subject"] = safe_subject(
            request.subject
        )

        message["Message-ID"] = (
            f"<{generate_id()}@"
            f"{self.config.sender_email.split('@')[1]}>"
        )

        message["X-Notification-Type"] = (
            request.notification_type.value
        )

        priority_headers = {
            EmailPriority.LOW: "5",
            EmailPriority.NORMAL: "3",
            EmailPriority.HIGH: "2",
            EmailPriority.CRITICAL: "1",
        }

        message["X-Priority"] = (
            priority_headers[
                request.priority
            ]
        )

        message.set_content(
            request.text_body
        )

        if request.html_body:
            message.add_alternative(
                request.html_body,
                subtype="html",
            )

        if request.attachments:
            if not self.config.allow_attachments:
                raise EmailValidationError(
                    "Attachments disabled"
                )

            for attachment in request.attachments:
                if len(attachment.content) > (
                    self.config.max_attachment_bytes
                ):
                    raise EmailValidationError(
                        "Attachment size limit exceeded"
                    )

                content_type = (
                    attachment.content_type
                )

                if "/" not in content_type:
                    raise EmailValidationError(
                        "Invalid attachment content type"
                    )

                maintype, subtype = (
                    content_type.split(
                        "/",
                        1,
                    )
                )

                message.add_attachment(
                    attachment.content,
                    maintype=maintype,
                    subtype=subtype,
                    filename=attachment.filename,
                )

        if len(message.as_bytes()) > (
            self.config.max_message_bytes
        ):
            raise EmailValidationError(
                "Email message size limit exceeded"
            )

        return message


# ============================================================
# SMTP TRANSPORT
# ============================================================

class SMTPEmailTransport:
    def __init__(
        self,
        config: EmailAdapterConfig,
    ) -> None:
        self.config = config

    def send(
        self,
        message: EmailMessage,
        recipients: Sequence[str],
    ) -> None:
        context = ssl.create_default_context()

        try:
            if (
                self.config.security_mode
                == EmailSecurityMode.SSL
            ):
                with smtplib.SMTP_SSL(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout_seconds,
                    context=context,
                ) as smtp:
                    self._authenticate(
                        smtp
                    )

                    refused = smtp.send_message(
                        message,
                        to_addrs=list(recipients),
                    )

            else:
                with smtplib.SMTP(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout_seconds,
                ) as smtp:
                    smtp.ehlo()

                    smtp.starttls(
                        context=context
                    )

                    smtp.ehlo()

                    self._authenticate(
                        smtp
                    )

                    refused = smtp.send_message(
                        message,
                        to_addrs=list(recipients),
                    )

            if refused:
                raise EmailDeliveryError(
                    "One or more recipients were refused"
                )

        except smtplib.SMTPAuthenticationError as exc:
            raise EmailAuthenticationError(
                "SMTP authentication failed"
            ) from exc

        except (
            smtplib.SMTPException,
            OSError,
        ) as exc:
            raise EmailDeliveryError(
                "SMTP transport failed"
            ) from exc

    def _authenticate(
        self,
        smtp: smtplib.SMTP,
    ) -> None:
        if (
            self.config.smtp_username
            and self.config.smtp_password
        ):
            smtp.login(
                self.config.smtp_username,
                self.config.smtp_password,
            )


# ============================================================
# EMAIL TEMPLATES
# ============================================================

class EmailTemplateEngine:
    @staticmethod
    def build(
        *,
        title: str,
        sections: Mapping[str, Any],
        footer: str = (
            "This message is for authorized "
            "coordination personnel. Verify "
            "critical information before action."
        ),
    ) -> tuple[str, str]:
        lines = [
            title,
            "=" * len(title),
            "",
        ]

        html_sections = []

        for label, value in sections.items():
            label_text = str(label)
            value_text = str(value)

            lines.append(
                f"{label_text}: {value_text}"
            )

            html_sections.append(
                "<p><strong>"
                + html.escape(label_text)
                + ":</strong> "
                + html.escape(value_text)
                + "</p>"
            )

        lines.extend([
            "",
            footer,
        ])

        text_body = "\n".join(
            lines
        )

        html_body = (
            "<!DOCTYPE html>"
            "<html><body>"
            f"<h2>{html.escape(title)}</h2>"
            + "".join(html_sections)
            + "<hr><p>"
            + html.escape(footer)
            + "</p></body></html>"
        )

        return (
            text_body,
            html_body,
        )


# ============================================================
# NOTIFICATION FACTORY
# ============================================================

class FloodEmailNotificationFactory:
    @staticmethod
    def create(
        *,
        recipients: Sequence[str],
        tenant_id: str,
        notification_type: EmailNotificationType,
        title: str,
        details: Mapping[str, Any],
        priority: EmailPriority = EmailPriority.NORMAL,
        approval: EmailApproval | None = None,
        idempotency_key: str | None = None,
    ) -> EmailRequest:
        text_body, html_body = (
            EmailTemplateEngine.build(
                title=title,
                sections=details,
            )
        )

        return EmailRequest(
            recipients=tuple(recipients),
            subject=title,
            text_body=text_body,
            html_body=html_body,
            tenant_id=tenant_id,
            notification_type=notification_type,
            priority=priority,
            approval=approval,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def flood_alert(
        *,
        recipients: Sequence[str],
        tenant_id: str,
        district: str,
        alert_reference: str,
        message: str,
        source: str,
        approval: EmailApproval,
    ) -> EmailRequest:
        return FloodEmailNotificationFactory.create(
            recipients=recipients,
            tenant_id=tenant_id,
            notification_type=(
                EmailNotificationType.FLOOD_ALERT
            ),
            title=(
                f"Flood Coordination Alert - {district}"
            ),
            details={
                "District": district,
                "Alert reference": alert_reference,
                "Message": message,
                "Source": source,
                "Status": (
                    "Human verification required"
                ),
            },
            priority=EmailPriority.CRITICAL,
            approval=approval,
            idempotency_key=(
                f"flood-alert:{alert_reference}"
            ),
        )

    @staticmethod
    def sos_notification(
        *,
        recipients: Sequence[str],
        tenant_id: str,
        sos_reference: str,
        district: str,
        location_description: str,
        priority_description: str,
        approval: EmailApproval,
    ) -> EmailRequest:
        return FloodEmailNotificationFactory.create(
            recipients=recipients,
            tenant_id=tenant_id,
            notification_type=(
                EmailNotificationType.SOS_REQUEST
            ),
            title=(
                f"SOS Coordination Notice - {sos_reference}"
            ),
            details={
                "SOS reference": sos_reference,
                "District": district,
                "Location": location_description,
                "Priority": priority_description,
                "Action": (
                    "Escalate through approved "
                    "emergency response channels"
                ),
            },
            priority=EmailPriority.CRITICAL,
            approval=approval,
            idempotency_key=(
                f"sos:{sos_reference}"
            ),
        )

    @staticmethod
    def rescue_update(
        *,
        recipients: Sequence[str],
        tenant_id: str,
        mission_reference: str,
        status: str,
        district: str,
    ) -> EmailRequest:
        return FloodEmailNotificationFactory.create(
            recipients=recipients,
            tenant_id=tenant_id,
            notification_type=(
                EmailNotificationType.RESCUE_UPDATE
            ),
            title=(
                f"Rescue Mission Update - {mission_reference}"
            ),
            details={
                "Mission": mission_reference,
                "District": district,
                "Status": status,
            },
            priority=EmailPriority.HIGH,
            idempotency_key=(
                f"rescue:{mission_reference}:{status}"
            ),
        )

    @staticmethod
    def shelter_update(
        *,
        recipients: Sequence[str],
        tenant_id: str,
        shelter_reference: str,
        occupancy: int,
        capacity: int,
    ) -> EmailRequest:
        if (
            occupancy < 0
            or capacity < 0
            or occupancy > capacity
        ):
            raise EmailValidationError(
                "Invalid shelter occupancy"
            )

        return FloodEmailNotificationFactory.create(
            recipients=recipients,
            tenant_id=tenant_id,
            notification_type=(
                EmailNotificationType.SHELTER_UPDATE
            ),
            title=(
                f"Shelter Capacity Update - {shelter_reference}"
            ),
            details={
                "Shelter": shelter_reference,
                "Occupancy": occupancy,
                "Capacity": capacity,
                "Available spaces": (
                    capacity - occupancy
                ),
                "Note": (
                    "Confirm live capacity with "
                    "authorized shelter personnel"
                ),
            },
            priority=EmailPriority.NORMAL,
        )

    @staticmethod
    def inventory_alert(
        *,
        recipients: Sequence[str],
        tenant_id: str,
        item_name: str,
        available_quantity: float,
        minimum_quantity: float,
        warehouse_reference: str,
    ) -> EmailRequest:
        return FloodEmailNotificationFactory.create(
            recipients=recipients,
            tenant_id=tenant_id,
            notification_type=(
                EmailNotificationType.RELIEF_INVENTORY
            ),
            title=(
                f"Relief Inventory Alert - {item_name}"
            ),
            details={
                "Warehouse": warehouse_reference,
                "Item": item_name,
                "Available": available_quantity,
                "Minimum threshold": minimum_quantity,
                "Action": (
                    "Review replenishment requirement"
                ),
            },
            priority=EmailPriority.HIGH,
        )


# ============================================================
# MAIN EMAIL ADAPTER
# ============================================================

class EmailAdapter:
    def __init__(
        self,
        config: EmailAdapterConfig,
        *,
        transport: SMTPEmailTransport | None = None,
        audit_logger: EmailAuditLogger | None = None,
        rate_limiter: EmailRateLimiter | None = None,
        idempotency_store: EmailIdempotencyStore | None = None,
    ) -> None:
        self.config = config

        self.transport = (
            transport
            or SMTPEmailTransport(config)
        )

        self.audit_logger = (
            audit_logger
            or EmailAuditLogger()
        )

        self.rate_limiter = (
            rate_limiter
            or EmailRateLimiter(
                config.rate_limit_per_minute
            )
        )

        self.idempotency_store = (
            idempotency_store
            or EmailIdempotencyStore()
        )

        self.approval_policy = (
            EmailApprovalPolicy(config)
        )

        self.message_builder = (
            EmailMessageBuilder(config)
        )

    def send(
        self,
        request: EmailRequest,
    ) -> EmailDeliveryResult:
        recipients = validate_recipients(
            request.recipients,
            maximum=self.config.max_recipients,
        )

        self.approval_policy.validate(
            request
        )

        if request.idempotency_key:
            previous = self.idempotency_store.get(
                request.tenant_id,
                request.idempotency_key,
            )

            if previous is not None:
                return EmailDeliveryResult(
                    request_id=request.request_id,
                    status=(
                        EmailDeliveryStatus.SKIPPED_DUPLICATE
                    ),
                    accepted_recipients=(
                        previous.accepted_recipients
                    ),
                    attempts=0,
                    message_id=previous.message_id,
                    timestamp=utc_now(),
                    idempotency_key=(
                        request.idempotency_key
                    ),
                )

        self.rate_limiter.acquire(
            request.tenant_id
        )

        message = self.message_builder.build(
            request
        )

        message_id = message["Message-ID"]

        self.audit_logger.record(
            request,
            "delivery_started",
            metadata={
                "notification_type": (
                    request.notification_type.value
                ),
                "recipient_count": len(recipients),
            },
        )

        last_error: Exception | None = None

        attempts = 0

        for attempt in range(
            self.config.max_retries + 1
        ):
            attempts += 1

            try:
                self.transport.send(
                    message,
                    recipients,
                )

                result = EmailDeliveryResult(
                    request_id=request.request_id,
                    status=EmailDeliveryStatus.ACCEPTED,
                    accepted_recipients=recipients,
                    attempts=attempts,
                    message_id=message_id,
                    timestamp=utc_now(),
                    idempotency_key=(
                        request.idempotency_key
                    ),
                )

                if request.idempotency_key:
                    self.idempotency_store.save(
                        request.tenant_id,
                        request.idempotency_key,
                        result,
                    )

                self.audit_logger.record(
                    request,
                    "smtp_accepted",
                    metadata={
                        "attempts": attempts,
                    },
                )

                return result

            except EmailAuthenticationError:
                self.audit_logger.record(
                    request,
                    "authentication_failed",
                )
                raise

            except EmailDeliveryError as exc:
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

                time.sleep(
                    delay
                )

        self.audit_logger.record(
            request,
            "delivery_failed",
            metadata={
                "attempts": attempts,
            },
        )

        raise EmailDeliveryError(
            "Email delivery failed after retries; "
            "the remote server may have accepted "
            "an earlier attempt"
        ) from last_error

    async def asend(
        self,
        request: EmailRequest,
    ) -> EmailDeliveryResult:
        return await asyncio.to_thread(
            self.send,
            request,
        )

    def send_batch(
        self,
        requests: Sequence[EmailRequest],
    ) -> list[EmailDeliveryResult]:
        results = []

        for request in requests:
            results.append(
                self.send(request)
            )

        return results

    async def asend_batch(
        self,
        requests: Sequence[EmailRequest],
    ) -> list[EmailDeliveryResult]:
        results = []

        for request in requests:
            results.append(
                await self.asend(request)
            )

        return results


# ============================================================
# NOTIFICATION SERVICE ADAPTER
# ============================================================

class EmailNotificationServiceAdapter:
    """
    Integrates with backend notification services.

    Requires a trusted approval workflow and
    verified recipient directory.
    """

    def __init__(
        self,
        email_adapter: EmailAdapter,
    ) -> None:
        self.email_adapter = email_adapter

    async def send_notification(
        self,
        *,
        recipients: Sequence[str],
        tenant_id: str,
        title: str,
        details: Mapping[str, Any],
        notification_type: EmailNotificationType,
        priority: EmailPriority = EmailPriority.NORMAL,
        approval: EmailApproval | None = None,
        idempotency_key: str | None = None,
    ) -> EmailDeliveryResult:
        request = (
            FloodEmailNotificationFactory.create(
                recipients=recipients,
                tenant_id=tenant_id,
                notification_type=notification_type,
                title=title,
                details=details,
                priority=priority,
                approval=approval,
                idempotency_key=idempotency_key,
            )
        )

        return await self.email_adapter.asend(
            request
        )


# ============================================================
# FACTORY
# ============================================================

def create_email_adapter(
    config: EmailAdapterConfig,
) -> EmailAdapter:
    return EmailAdapter(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def demo() -> None:
    """
    Demonstrates message construction only.

    No email is sent.
    """

    config = EmailAdapterConfig(
        smtp_host="smtp.example.invalid",
        smtp_port=587,
        sender_email="alerts@example.invalid",
    )

    request = (
        FloodEmailNotificationFactory.shelter_update(
            recipients=[
                "operator@example.invalid"
            ],
            tenant_id="demo_kerala",
            shelter_reference="DEMO-SHELTER-001",
            occupancy=25,
            capacity=100,
        )
    )

    message = EmailMessageBuilder(
        config
    ).build(request)

    print(
        message.as_string()
    )


if __name__ == "__main__":
    demo()


__all__ = [
    "EmailAdapterError",
    "EmailConfigurationError",
    "EmailValidationError",
    "EmailAuthenticationError",
    "EmailDeliveryError",
    "EmailRateLimitError",
    "EmailApprovalError",
    "EmailSecurityMode",
    "EmailPriority",
    "EmailNotificationType",
    "EmailDeliveryStatus",
    "EmailAdapterConfig",
    "EmailAttachment",
    "EmailApproval",
    "EmailRequest",
    "EmailDeliveryResult",
    "EmailAuditEvent",
    "EmailAuditLogger",
    "EmailRateLimiter",
    "EmailIdempotencyStore",
    "EmailApprovalPolicy",
    "EmailMessageBuilder",
    "SMTPEmailTransport",
    "EmailTemplateEngine",
    "FloodEmailNotificationFactory",
    "EmailAdapter",
    "EmailNotificationServiceAdapter",
    "create_email_adapter",
]
