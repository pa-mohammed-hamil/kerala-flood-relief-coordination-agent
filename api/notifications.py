from __future__ import annotations

"""
backend/api/notifications.py

Kerala Flood Relief Coordination Agent
Notification & Emergency Communication API

Python 3.11+ | FastAPI | Pydantic v2

Endpoints:
    GET    /notifications
    GET    /notifications/statistics
    GET    /notifications/preferences
    PUT    /notifications/preferences
    GET    /notifications/templates
    GET    /notifications/{notification_id}
    POST   /notifications/send
    POST   /notifications/broadcasts
    POST   /notifications/broadcasts/{broadcast_id}/approve
    POST   /notifications/broadcasts/{broadcast_id}/publish
    POST   /notifications/{notification_id}/read
    POST   /notifications/{notification_id}/acknowledge
    POST   /notifications/{notification_id}/retry
    GET    /notifications/{notification_id}/deliveries

Integration:
    Register a concrete notification service at:
        app.state.notification_service = service

Safety:
    - AI cannot independently publish official alerts.
    - Emergency broadcasts require human approval.
    - Broadcast publication requires a separate approver.
    - Recipient lists must be verified.
    - No fabricated delivery confirmations.
    - Phone numbers and email addresses are not returned.
    - Failed integrations return 503.
"""


import logging
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Protocol

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from backend.dependencies import get_current_user


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/notifications",
    tags=["Notifications"],
)

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# ROLES
# ============================================================

ADMIN_ROLES = {
    "admin",
    "super_admin",
}

COMMAND_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "incident_commander",
}

OPERATOR_ROLES = {
    "notification_operator",
    "communications_officer",
}

BROADCAST_ROLES = ADMIN_ROLES | COMMAND_ROLES

SEND_ROLES = (
    ADMIN_ROLES
    | COMMAND_ROLES
    | OPERATOR_ROLES
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class NotificationChannel(str, Enum):
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WHATSAPP = "whatsapp"


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationCategory(str, Enum):
    SYSTEM = "system"
    FLOOD_ALERT = "flood_alert"
    SOS = "sos"
    RESCUE = "rescue"
    EVACUATION = "evacuation"
    SHELTER = "shelter"
    HEALTHCARE = "healthcare"
    RELIEF = "relief"
    VOLUNTEER = "volunteer"
    MISSING_PERSON = "missing_person"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BroadcastStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    CANCELLED = "cancelled"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class NotificationSendRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    recipient_user_id: str

    title: str = Field(
        min_length=3,
        max_length=200,
    )

    message: str = Field(
        min_length=5,
        max_length=5000,
    )

    category: NotificationCategory

    priority: NotificationPriority = (
        NotificationPriority.NORMAL
    )

    channels: list[NotificationChannel] = Field(
        min_length=1,
        max_length=5,
    )

    reference_id: str | None = Field(
        default=None,
        max_length=150,
    )

    idempotency_key: str = Field(
        min_length=8,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_request(
        self,
    ) -> "NotificationSendRequest":
        self.recipient_user_id = str(
            uuid.UUID(self.recipient_user_id)
        )

        if len(set(self.channels)) != len(
            self.channels
        ):
            raise ValueError(
                "Duplicate notification channels"
            )

        return self


class NotificationPreferencesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email_enabled: bool = True
    sms_enabled: bool = True
    push_enabled: bool = True
    in_app_enabled: bool = True
    whatsapp_enabled: bool = False

    quiet_hours_enabled: bool = False

    quiet_hours_start: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
    )

    quiet_hours_end: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
    )

    @model_validator(mode="after")
    def validate_quiet_hours(
        self,
    ) -> "NotificationPreferencesUpdate":
        if self.quiet_hours_enabled:
            if not (
                self.quiet_hours_start
                and self.quiet_hours_end
            ):
                raise ValueError(
                    "Quiet hours require start and end times"
                )

        return self


class EmergencyBroadcastCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    district_id: int = Field(ge=1)

    title: str = Field(
        min_length=5,
        max_length=200,
    )

    message: str = Field(
        min_length=10,
        max_length=5000,
    )

    category: NotificationCategory

    priority: NotificationPriority = (
        NotificationPriority.CRITICAL
    )

    channels: list[NotificationChannel] = Field(
        min_length=1,
    )

    target_area_reference: str = Field(
        min_length=3,
        max_length=500,
    )

    source_alert_id: str | None = None

    source_verified: bool = False

    audience_verified: bool = False

    idempotency_key: str = Field(
        min_length=8,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_broadcast(
        self,
    ) -> "EmergencyBroadcastCreate":
        if self.source_alert_id is not None:
            self.source_alert_id = str(
                uuid.UUID(self.source_alert_id)
            )

        if len(set(self.channels)) != len(
            self.channels
        ):
            raise ValueError(
                "Duplicate broadcast channels"
            )

        return self


class BroadcastApprovalRequest(BaseModel):
    approved: bool

    review_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    source_verified: bool = False

    content_verified: bool = False

    target_area_verified: bool = False


class BroadcastPublishRequest(BaseModel):
    confirmation_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    authorized_for_publication: bool

    recipient_scope_verified: bool

    channels_verified: bool


class NotificationRetryRequest(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=2000,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class NotificationResponse(BaseModel):
    id: str

    title: str

    message: str

    category: str

    priority: str

    status: str

    is_read: bool = False

    acknowledged_at: datetime | None = None

    created_at: datetime


class NotificationListResponse(BaseModel):
    items: list[NotificationResponse]

    total: int

    limit: int

    offset: int


class NotificationStatisticsResponse(BaseModel):
    total: int

    unread: int

    acknowledged: int

    by_status: dict[str, int]

    generated_at: datetime


class NotificationPreferencesResponse(BaseModel):
    email_enabled: bool
    sms_enabled: bool
    push_enabled: bool
    in_app_enabled: bool
    whatsapp_enabled: bool
    quiet_hours_enabled: bool
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None


class NotificationDeliveryResponse(BaseModel):
    id: str

    notification_id: str

    channel: NotificationChannel

    status: DeliveryStatus

    attempted_at: datetime | None = None

    delivered_at: datetime | None = None


class NotificationTemplateResponse(BaseModel):
    id: str

    name: str

    category: NotificationCategory

    supported_channels: list[NotificationChannel]

    version: str

    active: bool


class BroadcastResponse(BaseModel):
    id: str

    district_id: int

    status: BroadcastStatus

    title: str

    created_at: datetime

    approved_at: datetime | None = None

    published_at: datetime | None = None


class OperationResponse(BaseModel):
    success: bool

    reference_id: str

    status: str

    message: str


# ============================================================
# SERVICE CONTRACT
# ============================================================

class NotificationService(Protocol):
    """
    Concrete service must enforce:

    - User ownership and district scoping
    - Permission checks
    - Consent and channel preferences
    - Verified recipients
    - Message sanitization
    - Idempotency
    - Rate limits
    - Delivery audit records
    - Provider callbacks
    - Safe retries
    - Human approval for broadcasts
    """

    async def list_notifications(
        self,
        user_id: str,
        limit: int,
        offset: int,
        unread_only: bool,
    ) -> NotificationListResponse:
        ...

    async def get_notification(
        self,
        notification_id: str,
        user_id: str,
    ) -> NotificationResponse | None:
        ...

    async def get_statistics(
        self,
        user_id: str,
    ) -> NotificationStatisticsResponse:
        ...

    async def get_preferences(
        self,
        user_id: str,
    ) -> NotificationPreferencesResponse:
        ...

    async def update_preferences(
        self,
        user_id: str,
        payload: NotificationPreferencesUpdate,
    ) -> NotificationPreferencesResponse:
        ...

    async def list_templates(
        self,
    ) -> list[NotificationTemplateResponse]:
        ...

    async def send_notification(
        self,
        payload: NotificationSendRequest,
        actor_id: str,
    ) -> OperationResponse:
        ...

    async def mark_read(
        self,
        notification_id: str,
        user_id: str,
    ) -> OperationResponse:
        ...

    async def acknowledge(
        self,
        notification_id: str,
        user_id: str,
    ) -> OperationResponse:
        ...

    async def list_deliveries(
        self,
        notification_id: str,
        actor_id: str,
    ) -> list[NotificationDeliveryResponse]:
        ...

    async def retry_notification(
        self,
        notification_id: str,
        actor_id: str,
        reason: str,
    ) -> OperationResponse:
        ...

    async def create_broadcast(
        self,
        payload: EmergencyBroadcastCreate,
        actor_id: str,
    ) -> BroadcastResponse:
        ...

    async def get_broadcast(
        self,
        broadcast_id: str,
        actor_id: str,
    ) -> BroadcastResponse | None:
        ...

    async def approve_broadcast(
        self,
        broadcast_id: str,
        actor_id: str,
        payload: BroadcastApprovalRequest,
    ) -> BroadcastResponse:
        ...

    async def publish_broadcast(
        self,
        broadcast_id: str,
        actor_id: str,
        payload: BroadcastPublishRequest,
    ) -> BroadcastResponse:
        ...


# ============================================================
# DEPENDENCIES
# ============================================================

def get_notification_service(
    request: Request,
) -> NotificationService:
    service = getattr(
        request.app.state,
        "notification_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Notification service is not configured",
        )

    return service


NotificationServiceDependency = Annotated[
    NotificationService,
    Depends(get_notification_service),
]


# ============================================================
# SECURITY HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid UUID",
        ) from None


def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return str(identifier)


def role_of(user: Any) -> str:
    value = getattr(user, "role", "")

    if isinstance(value, Enum):
        value = value.value

    return str(value).lower()


def require_role(
    user: Any,
    allowed_roles: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions",
        )


def require_district_access(
    user: Any,
    district_id: int,
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) in ADMIN_ROLES:
        return

    assigned_district = getattr(
        user,
        "district_id",
        None,
    )

    if (
        assigned_district is None
        or int(assigned_district) != district_id
    ):
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )


# ============================================================
# NOTIFICATION LIST
# ============================================================

@router.get(
    "",
    response_model=NotificationListResponse,
)
async def list_notifications(
    current_user: CurrentUser,
    service: NotificationServiceDependency,
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    unread_only: bool = False,
) -> NotificationListResponse:
    return await service.list_notifications(
        user_id=actor_id(current_user),
        limit=limit,
        offset=offset,
        unread_only=unread_only,
    )


# ============================================================
# NOTIFICATION STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=NotificationStatisticsResponse,
)
async def notification_statistics(
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> NotificationStatisticsResponse:
    return await service.get_statistics(
        user_id=actor_id(current_user),
    )


# ============================================================
# USER PREFERENCES
# ============================================================

@router.get(
    "/preferences",
    response_model=NotificationPreferencesResponse,
)
async def get_notification_preferences(
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> NotificationPreferencesResponse:
    return await service.get_preferences(
        user_id=actor_id(current_user),
    )


@router.put(
    "/preferences",
    response_model=NotificationPreferencesResponse,
)
async def update_notification_preferences(
    payload: NotificationPreferencesUpdate,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> NotificationPreferencesResponse:
    return await service.update_preferences(
        user_id=actor_id(current_user),
        payload=payload,
    )


# ============================================================
# TEMPLATES
# ============================================================

@router.get(
    "/templates",
    response_model=list[NotificationTemplateResponse],
)
async def list_notification_templates(
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> list[NotificationTemplateResponse]:
    require_role(current_user, SEND_ROLES)

    return await service.list_templates()


# ============================================================
# SEND DIRECT NOTIFICATION
# ============================================================

@router.post(
    "/send",
    response_model=OperationResponse,
    status_code=202,
)
async def send_notification(
    payload: NotificationSendRequest,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> OperationResponse:
    require_role(current_user, SEND_ROLES)

    # Critical flood alerts should use the separately
    # approved broadcast workflow, not direct sending.
    if payload.category == NotificationCategory.FLOOD_ALERT:
        raise HTTPException(
            status_code=409,
            detail=(
                "Official flood alerts must use "
                "the approved broadcast workflow"
            ),
        )

    return await service.send_notification(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# CREATE EMERGENCY BROADCAST
# ============================================================

@router.post(
    "/broadcasts",
    response_model=BroadcastResponse,
    status_code=201,
)
async def create_emergency_broadcast(
    payload: EmergencyBroadcastCreate,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> BroadcastResponse:
    require_role(current_user, BROADCAST_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    if not (
        payload.source_verified
        and payload.audience_verified
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Source and target audience "
                "must be verified"
            ),
        )

    return await service.create_broadcast(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# APPROVE BROADCAST
# ============================================================

@router.post(
    "/broadcasts/{broadcast_id}/approve",
    response_model=BroadcastResponse,
)
async def approve_emergency_broadcast(
    broadcast_id: str,
    payload: BroadcastApprovalRequest,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> BroadcastResponse:
    require_role(current_user, BROADCAST_ROLES)

    identifier = validate_uuid(broadcast_id)

    broadcast = await service.get_broadcast(
        broadcast_id=identifier,
        actor_id=actor_id(current_user),
    )

    if broadcast is None:
        raise HTTPException(
            status_code=404,
            detail="Broadcast not found",
        )

    require_district_access(
        current_user,
        broadcast.district_id,
    )

    if payload.approved:
        if not all(
            [
                payload.source_verified,
                payload.content_verified,
                payload.target_area_verified,
            ]
        ):
            raise HTTPException(
                status_code=422,
                detail="Broadcast approval checks incomplete",
            )

    return await service.approve_broadcast(
        broadcast_id=identifier,
        actor_id=actor_id(current_user),
        payload=payload,
    )


# ============================================================
# PUBLISH BROADCAST
# ============================================================

@router.post(
    "/broadcasts/{broadcast_id}/publish",
    response_model=BroadcastResponse,
)
async def publish_emergency_broadcast(
    broadcast_id: str,
    payload: BroadcastPublishRequest,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> BroadcastResponse:
    require_role(current_user, BROADCAST_ROLES)

    identifier = validate_uuid(broadcast_id)

    broadcast = await service.get_broadcast(
        broadcast_id=identifier,
        actor_id=actor_id(current_user),
    )

    if broadcast is None:
        raise HTTPException(
            status_code=404,
            detail="Broadcast not found",
        )

    require_district_access(
        current_user,
        broadcast.district_id,
    )

    if broadcast.status != BroadcastStatus.APPROVED:
        raise HTTPException(
            status_code=409,
            detail="Broadcast is not approved",
        )

    if not all(
        [
            payload.authorized_for_publication,
            payload.recipient_scope_verified,
            payload.channels_verified,
        ]
    ):
        raise HTTPException(
            status_code=422,
            detail="Publication safety checks incomplete",
        )

    # Service must verify a distinct authorized approver,
    # current approval validity, idempotency and an
    # atomic audited publication transaction.
    return await service.publish_broadcast(
        broadcast_id=identifier,
        actor_id=actor_id(current_user),
        payload=payload,
    )


# ============================================================
# NOTIFICATION DETAILS
# ============================================================

@router.get(
    "/{notification_id}",
    response_model=NotificationResponse,
)
async def get_notification(
    notification_id: str,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> NotificationResponse:
    notification = await service.get_notification(
        notification_id=validate_uuid(
            notification_id
        ),
        user_id=actor_id(current_user),
    )

    if notification is None:
        raise HTTPException(
            status_code=404,
            detail="Notification not found",
        )

    return notification


# ============================================================
# MARK AS READ
# ============================================================

@router.post(
    "/{notification_id}/read",
    response_model=OperationResponse,
)
async def mark_notification_read(
    notification_id: str,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> OperationResponse:
    return await service.mark_read(
        notification_id=validate_uuid(
            notification_id
        ),
        user_id=actor_id(current_user),
    )


# ============================================================
# ACKNOWLEDGE NOTIFICATION
# ============================================================

@router.post(
    "/{notification_id}/acknowledge",
    response_model=OperationResponse,
)
async def acknowledge_notification(
    notification_id: str,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> OperationResponse:
    return await service.acknowledge(
        notification_id=validate_uuid(
            notification_id
        ),
        user_id=actor_id(current_user),
    )


# ============================================================
# DELIVERY HISTORY
# ============================================================

@router.get(
    "/{notification_id}/deliveries",
    response_model=list[NotificationDeliveryResponse],
)
async def notification_deliveries(
    notification_id: str,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> list[NotificationDeliveryResponse]:
    require_role(current_user, SEND_ROLES)

    return await service.list_deliveries(
        notification_id=validate_uuid(
            notification_id
        ),
        actor_id=actor_id(current_user),
    )


# ============================================================
# RETRY FAILED NOTIFICATION
# ============================================================

@router.post(
    "/{notification_id}/retry",
    response_model=OperationResponse,
    status_code=202,
)
async def retry_failed_notification(
    notification_id: str,
    payload: NotificationRetryRequest,
    current_user: CurrentUser,
    service: NotificationServiceDependency,
) -> OperationResponse:
    require_role(current_user, SEND_ROLES)

    # Service must ensure that:
    # - Notification belongs to an authorized scope
    # - Delivery actually failed
    # - Retries are idempotent
    # - Provider rate limits are respected
    # - Official alerts cannot bypass approval
    return await service.retry_notification(
        notification_id=validate_uuid(
            notification_id
        ),
        actor_id=actor_id(current_user),
        reason=payload.reason,
    )


__all__ = [
    "router",
    "NotificationService",
    "NotificationChannel",
    "NotificationPriority",
    "NotificationCategory",
    "DeliveryStatus",
    "BroadcastStatus",
    "NotificationSendRequest",
    "NotificationPreferencesUpdate",
    "EmergencyBroadcastCreate",
    "BroadcastApprovalRequest",
    "BroadcastPublishRequest",
    "NotificationRetryRequest",
    "NotificationResponse",
    "NotificationListResponse",
    "NotificationStatisticsResponse",
    "NotificationPreferencesResponse",
    "NotificationDeliveryResponse",
    "NotificationTemplateResponse",
    "BroadcastResponse",
    "OperationResponse",
]
