from __future__ import annotations

"""
backend/api/flood_alerts.py

Kerala Flood Relief Coordination Agent
Flood Alert Management API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Endpoints:
    GET    /flood-alerts
    GET    /flood-alerts/statistics
    POST   /flood-alerts
    GET    /flood-alerts/{alert_id}
    PATCH  /flood-alerts/{alert_id}
    POST   /flood-alerts/{alert_id}/submit
    POST   /flood-alerts/{alert_id}/review
    POST   /flood-alerts/{alert_id}/publish
    POST   /flood-alerts/{alert_id}/resolve
    POST   /flood-alerts/{alert_id}/cancel
    GET    /flood-alerts/{alert_id}/evidence
    POST   /flood-alerts/{alert_id}/evidence
    GET    /flood-alerts/{alert_id}/deliveries
    GET    /flood-alerts/{alert_id}/acknowledgments
    POST   /flood-alerts/{alert_id}/acknowledgments
    GET    /flood-alerts/{alert_id}/events

Safety:
    AI-generated alerts are advisory until verified.
    Official publication requires human authorization.
    Public warning delivery requires an audited service.
    This API does not issue physical dam-control commands.
"""


import logging
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.flood_alert import (
    FloodAlert,
    FloodAlertEvidence,
    FloodAlertDelivery,
    FloodAlertAcknowledgment,
    FloodAlertEvent,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/flood-alerts",
    tags=["Flood Alerts"],
)

DbSession = Annotated[
    AsyncSession,
    Depends(get_db),
]

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# AUTHORIZATION
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

ANALYST_ROLES = {
    "hydrologist",
    "meteorologist",
    "risk_analyst",
}

OPERATOR_ROLES = {
    "operator",
    "field_officer",
}

READ_ROLES = (
    ADMIN_ROLES
    | COMMAND_ROLES
    | ANALYST_ROLES
    | OPERATOR_ROLES
    | {"reviewer"}
)

CREATE_ROLES = (
    ADMIN_ROLES
    | COMMAND_ROLES
    | ANALYST_ROLES
)

APPROVAL_ROLES = (
    ADMIN_ROLES
    | COMMAND_ROLES
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class FloodAlertSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"


class FloodAlertCategory(str, Enum):
    HEAVY_RAINFALL = "heavy_rainfall"
    RIVER_FLOODING = "river_flooding"
    FLASH_FLOOD = "flash_flood"
    URBAN_FLOODING = "urban_flooding"
    DAM_RELEASE = "dam_release"
    LANDSLIDE_RISK = "landslide_risk"
    COASTAL_FLOODING = "coastal_flooding"
    GENERAL_FLOOD = "general_flood"


class FloodAlertLifecycle(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    APPROVED = "approved"
    PUBLISHED = "published"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AlertReviewAction(str, Enum):
    VERIFY = "verify"
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class AlertDeliveryChannel(str, Enum):
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    WEB = "web"
    WHATSAPP = "whatsapp"
    SIREN = "siren"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class FloodAlertCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    district_id: int = Field(ge=1)

    title: str = Field(
        min_length=5,
        max_length=255,
    )

    description: str = Field(
        min_length=10,
        max_length=10000,
    )

    severity: FloodAlertSeverity

    category: FloodAlertCategory

    affected_area: str | None = Field(
        default=None,
        max_length=1000,
    )

    valid_from: datetime | None = None

    valid_until: datetime | None = None

    recommended_actions: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    source_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    ai_generated: bool = False

    model_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    confidence_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    @field_validator("recommended_actions")
    @classmethod
    def validate_actions(
        cls,
        value: list[str],
    ) -> list[str]:
        for action in value:
            if not action.strip():
                raise ValueError(
                    "Recommended actions cannot be empty"
                )

            if len(action) > 1000:
                raise ValueError(
                    "Recommended action exceeds length limit"
                )

        return value


class FloodAlertUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None,
        min_length=5,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
        min_length=10,
        max_length=10000,
    )

    severity: FloodAlertSeverity | None = None

    affected_area: str | None = Field(
        default=None,
        max_length=1000,
    )

    valid_until: datetime | None = None


class FloodAlertReviewRequest(BaseModel):
    action: AlertReviewAction

    notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    evidence_verified: bool = False


class FloodAlertPublishRequest(BaseModel):
    confirmation_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    channels: list[AlertDeliveryChannel] = Field(
        min_length=1,
    )

    target_area_verified: bool

    content_verified: bool

    authorized_for_publication: bool


class FloodAlertResolveRequest(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=5000,
    )


class FloodAlertCancelRequest(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=5000,
    )


class FloodAlertEvidenceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_type: str = Field(
        min_length=2,
        max_length=100,
    )

    source_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    observed_at: datetime | None = None


class FloodAlertAcknowledgmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery_id: str | None = None

    acknowledgment_type: str = Field(
        min_length=2,
        max_length=80,
    )

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )

    @field_validator("delivery_id")
    @classmethod
    def validate_delivery_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return str(uuid.UUID(value))


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class FloodAlertResponse(BaseModel):
    id: str
    district_id: int
    status: str
    severity: str | None = None
    title: str | None = None
    created_at: datetime | None = None


class FloodAlertListResponse(BaseModel):
    items: list[FloodAlertResponse]
    total: int
    limit: int
    offset: int


class FloodAlertStatisticsResponse(BaseModel):
    total_alerts: int
    by_status: dict[str, int]
    by_severity: dict[str, int]
    generated_at: datetime


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid UUID",
        ) from None


def role_of(user: Any) -> str:
    return enum_value(
        getattr(user, "role", "")
    ).lower()


def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return str(identifier)


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


def scoped_district(
    user: Any,
    district_id: int | None,
) -> int | None:
    require_role(user, READ_ROLES)

    if (
        getattr(user, "is_superuser", False)
        or role_of(user) in ADMIN_ROLES
    ):
        return district_id

    assigned_district = getattr(
        user,
        "district_id",
        None,
    )

    if assigned_district is None:
        raise HTTPException(
            status_code=403,
            detail="District assignment required",
        )

    if (
        district_id is not None
        and district_id != int(assigned_district)
    ):
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )

    return int(assigned_district)


async def get_alert_or_404(
    db: AsyncSession,
    alert_id: str,
    user: Any,
) -> FloodAlert:
    identifier = validate_uuid(alert_id)

    result = await db.execute(
        select(FloodAlert).where(
            FloodAlert.id == identifier
        )
    )

    alert = result.scalar_one_or_none()

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Flood alert not found",
        )

    require_district_access(
        user,
        alert.district_id,
    )

    return alert


def alert_response(
    alert: FloodAlert,
) -> FloodAlertResponse:
    return FloodAlertResponse(
        id=str(alert.id),
        district_id=alert.district_id,
        status=enum_value(alert.status),
        severity=enum_value(
            getattr(
                alert,
                "severity",
                None,
            )
        ) or None,
        title=getattr(
            alert,
            "title",
            None,
        ),
        created_at=getattr(
            alert,
            "created_at",
            None,
        ),
    )


def require_editable(alert: FloodAlert) -> None:
    if enum_value(alert.status) not in {
        "draft",
        "returned_for_changes",
    }:
        raise HTTPException(
            status_code=409,
            detail="Flood alert cannot be edited",
        )


# ============================================================
# LIST ALERTS
# ============================================================

@router.get(
    "",
    response_model=FloodAlertListResponse,
)
async def list_flood_alerts(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> FloodAlertListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(FloodAlert)

    count_query = (
        select(func.count())
        .select_from(FloodAlert)
    )

    if district_id is not None:
        query = query.where(
            FloodAlert.district_id == district_id
        )

        count_query = count_query.where(
            FloodAlert.district_id == district_id
        )

    total_result = await db.execute(
        count_query
    )

    result = await db.execute(
        query.order_by(
            FloodAlert.created_at.desc()
        )
        .offset(offset)
        .limit(limit)
    )

    return FloodAlertListResponse(
        items=[
            alert_response(alert)
            for alert in result.scalars().all()
        ],
        total=int(total_result.scalar_one()),
        limit=limit,
        offset=offset,
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=FloodAlertStatisticsResponse,
)
async def flood_alert_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> FloodAlertStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(
        FloodAlert.status,
        FloodAlert.severity,
        func.count(FloodAlert.id),
    )

    if district_id is not None:
        query = query.where(
            FloodAlert.district_id == district_id
        )

    query = query.group_by(
        FloodAlert.status,
        FloodAlert.severity,
    )

    result = await db.execute(query)

    by_status: dict[str, int] = {}
    by_severity: dict[str, int] = {}

    total = 0

    for row in result.all():
        current_status = enum_value(row[0])
        severity = enum_value(row[1])
        count = int(row[2])

        by_status[current_status] = (
            by_status.get(current_status, 0) + count
        )

        by_severity[severity] = (
            by_severity.get(severity, 0) + count
        )

        total += count

    return FloodAlertStatisticsResponse(
        total_alerts=total,
        by_status=by_status,
        by_severity=by_severity,
        generated_at=utc_now(),
    )


# ============================================================
# CREATE ALERT
# ============================================================

@router.post(
    "",
    response_model=FloodAlertResponse,
    status_code=201,
)
async def create_flood_alert(
    payload: FloodAlertCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> FloodAlertResponse:
    require_role(current_user, CREATE_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    # Requires domain service integration:
    # - source validation
    # - correct model field mapping
    # - duplicate alert detection
    # - draft status enforcement
    # - audit event creation
    raise HTTPException(
        status_code=501,
        detail="Flood alert creation service is not configured",
    )


# ============================================================
# GET ALERT
# ============================================================

@router.get(
    "/{alert_id}",
    response_model=FloodAlertResponse,
)
async def get_flood_alert(
    alert_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> FloodAlertResponse:
    require_role(current_user, READ_ROLES)

    alert = await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    return alert_response(alert)


# ============================================================
# UPDATE ALERT
# ============================================================

@router.patch(
    "/{alert_id}",
    response_model=FloodAlertResponse,
)
async def update_flood_alert(
    alert_id: str,
    payload: FloodAlertUpdate,
    db: DbSession,
    current_user: CurrentUser,
) -> FloodAlertResponse:
    require_role(current_user, CREATE_ROLES)

    alert = await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    require_editable(alert)

    raise HTTPException(
        status_code=501,
        detail="Flood alert update service is not configured",
    )


# ============================================================
# SUBMIT FOR REVIEW
# ============================================================

@router.post("/{alert_id}/submit")
async def submit_flood_alert(
    alert_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, CREATE_ROLES)

    alert = await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    require_editable(alert)

    raise HTTPException(
        status_code=501,
        detail="Flood alert review submission is not configured",
    )


# ============================================================
# HUMAN REVIEW
# ============================================================

@router.post("/{alert_id}/review")
async def review_flood_alert(
    alert_id: str,
    payload: FloodAlertReviewRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, APPROVAL_ROLES)

    alert = await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    if enum_value(alert.status) not in {
        "pending_review",
        "verified",
    }:
        raise HTTPException(
            status_code=409,
            detail="Alert is not awaiting review",
        )

    if (
        payload.action
        in {
            AlertReviewAction.VERIFY,
            AlertReviewAction.APPROVE,
        }
        and not payload.evidence_verified
    ):
        raise HTTPException(
            status_code=422,
            detail="Evidence verification required",
        )

    raise HTTPException(
        status_code=501,
        detail="Human alert review workflow is not configured",
    )


# ============================================================
# PUBLICATION
# ============================================================

@router.post("/{alert_id}/publish")
async def publish_flood_alert(
    alert_id: str,
    payload: FloodAlertPublishRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, APPROVAL_ROLES)

    alert = await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    if enum_value(alert.status) != "approved":
        raise HTTPException(
            status_code=409,
            detail="Alert must be approved before publication",
        )

    if not all(
        [
            payload.target_area_verified,
            payload.content_verified,
            payload.authorized_for_publication,
        ]
    ):
        raise HTTPException(
            status_code=422,
            detail="Publication safety checks incomplete",
        )

    # An approved alert must be published through an
    # authorized notification/outbox service with:
    # - channel allowlists
    # - recipient scope validation
    # - idempotency
    # - delivery logging
    # - immutable audit records
    # - human authorization verification
    raise HTTPException(
        status_code=501,
        detail="Official alert publication service is not configured",
    )


# ============================================================
# RESOLUTION
# ============================================================

@router.post("/{alert_id}/resolve")
async def resolve_flood_alert(
    alert_id: str,
    payload: FloodAlertResolveRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, APPROVAL_ROLES)

    alert = await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    if enum_value(alert.status) != "published":
        raise HTTPException(
            status_code=409,
            detail="Only published alerts can be resolved",
        )

    raise HTTPException(
        status_code=501,
        detail="Flood alert resolution service is not configured",
    )


# ============================================================
# CANCELLATION
# ============================================================

@router.post("/{alert_id}/cancel")
async def cancel_flood_alert(
    alert_id: str,
    payload: FloodAlertCancelRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, APPROVAL_ROLES)

    alert = await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    if enum_value(alert.status) in {
        "resolved",
        "cancelled",
    }:
        raise HTTPException(
            status_code=409,
            detail="Alert is already finalized",
        )

    raise HTTPException(
        status_code=501,
        detail="Flood alert cancellation service is not configured",
    )


# ============================================================
# EVIDENCE
# ============================================================

@router.get("/{alert_id}/evidence")
async def list_alert_evidence(
    alert_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    result = await db.execute(
        select(FloodAlertEvidence).where(
            FloodAlertEvidence.alert_id
            == validate_uuid(alert_id)
        )
    )

    return [
        {
            "id": str(item.id),
            "alert_id": str(item.alert_id),
        }
        for item in result.scalars().all()
    ]


@router.post(
    "/{alert_id}/evidence",
    status_code=201,
)
async def add_alert_evidence(
    alert_id: str,
    payload: FloodAlertEvidenceCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, CREATE_ROLES)

    await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    raise HTTPException(
        status_code=501,
        detail="Alert evidence ingestion is not configured",
    )


# ============================================================
# DELIVERIES
# ============================================================

@router.get("/{alert_id}/deliveries")
async def list_alert_deliveries(
    alert_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    result = await db.execute(
        select(FloodAlertDelivery)
        .where(
            FloodAlertDelivery.alert_id
            == validate_uuid(alert_id)
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "alert_id": str(item.alert_id),
            "status": enum_value(
                getattr(
                    item,
                    "status",
                    None,
                )
            ),
        }
        for item in result.scalars().all()
    ]


# ============================================================
# ACKNOWLEDGMENTS
# ============================================================

@router.get("/{alert_id}/acknowledgments")
async def list_alert_acknowledgments(
    alert_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    result = await db.execute(
        select(FloodAlertAcknowledgment)
        .where(
            FloodAlertAcknowledgment.alert_id
            == validate_uuid(alert_id)
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "alert_id": str(item.alert_id),
        }
        for item in result.scalars().all()
    ]


@router.post(
    "/{alert_id}/acknowledgments",
    status_code=201,
)
async def acknowledge_flood_alert(
    alert_id: str,
    payload: FloodAlertAcknowledgmentCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, READ_ROLES)

    await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    raise HTTPException(
        status_code=501,
        detail="Alert acknowledgment service is not configured",
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get("/{alert_id}/events")
async def list_alert_events(
    alert_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_alert_or_404(
        db,
        alert_id,
        current_user,
    )

    result = await db.execute(
        select(FloodAlertEvent)
        .where(
            FloodAlertEvent.alert_id
            == validate_uuid(alert_id)
        )
        .order_by(
            FloodAlertEvent.created_at.desc()
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "alert_id": str(item.alert_id),
            "created_at": item.created_at,
        }
        for item in result.scalars().all()
    ]


__all__ = [
    "router",
    "FloodAlertCreate",
    "FloodAlertUpdate",
    "FloodAlertReviewRequest",
    "FloodAlertPublishRequest",
    "FloodAlertResolveRequest",
    "FloodAlertCancelRequest",
    "FloodAlertEvidenceCreate",
    "FloodAlertAcknowledgmentCreate",
    "FloodAlertResponse",
    "FloodAlertListResponse",
    "FloodAlertStatisticsResponse",
]
