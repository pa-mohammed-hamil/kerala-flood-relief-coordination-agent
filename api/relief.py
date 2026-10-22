
"""
backend/api/relief.py

Kerala Flood Relief Coordination Agent
Relief Distribution & Humanitarian Logistics API

Python 3.11+ | FastAPI | Pydantic v2

Endpoints:
    GET    /relief/health
    GET    /relief/deliveries
    GET    /relief/statistics
    POST   /relief/deliveries
    GET    /relief/deliveries/{delivery_id}
    PATCH  /relief/deliveries/{delivery_id}
    GET    /relief/deliveries/{delivery_id}/items
    POST   /relief/deliveries/{delivery_id}/items
    GET    /relief/deliveries/{delivery_id}/checkpoints
    POST   /relief/deliveries/{delivery_id}/checkpoints
    GET    /relief/deliveries/{delivery_id}/proofs
    POST   /relief/deliveries/{delivery_id}/proofs
    GET    /relief/deliveries/{delivery_id}/exceptions
    POST   /relief/deliveries/{delivery_id}/exceptions
    POST   /relief/deliveries/{delivery_id}/submit
    POST   /relief/deliveries/{delivery_id}/approve
    POST   /relief/deliveries/{delivery_id}/dispatch
    POST   /relief/deliveries/{delivery_id}/complete
    POST   /relief/deliveries/{delivery_id}/cancel
    GET    /relief/deliveries/{delivery_id}/events

Safety:
    - AI recommendations do not authorize allocation.
    - Inventory must be verified before dispatch.
    - Dispatch requires human authorization.
    - Proof of delivery must be verified.
    - Personal beneficiary information is restricted.
    - Operational transitions require audit logging.
"""

from __future__ import annotations

import logging
import uuid

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Protocol

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.dependencies import get_current_user


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/relief",
    tags=["Relief Distribution"],
)

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# ROLE CONFIGURATION
# ============================================================

ADMIN_ROLES = {
    "admin",
    "super_admin",
}

RELIEF_ROLES = {
    "relief_coordinator",
    "district_officer",
    "logistics_coordinator",
    "warehouse_manager",
}

FIELD_ROLES = {
    "field_officer",
    "delivery_coordinator",
    "relief_volunteer",
}

APPROVAL_ROLES = {
    "district_officer",
    "relief_coordinator",
    "incident_commander",
}

READ_ROLES = (
    ADMIN_ROLES
    | RELIEF_ROLES
    | FIELD_ROLES
    | APPROVAL_ROLES
    | {"auditor", "emergency_coordinator"}
)

WRITE_ROLES = (
    ADMIN_ROLES
    | RELIEF_ROLES
    | {"delivery_coordinator"}
)

APPROVER_ROLES = (
    ADMIN_ROLES
    | APPROVAL_ROLES
)

DISPATCH_ROLES = (
    ADMIN_ROLES
    | {
        "relief_coordinator",
        "logistics_coordinator",
        "district_officer",
    }
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class ReliefCategory(str, Enum):
    FOOD = "food"
    DRINKING_WATER = "drinking_water"
    MEDICINE = "medicine"
    HYGIENE = "hygiene"
    CLOTHING = "clothing"
    BLANKETS = "blankets"
    SHELTER_KITS = "shelter_kits"
    BABY_SUPPLIES = "baby_supplies"
    SANITATION = "sanitation"
    EMERGENCY_EQUIPMENT = "emergency_equipment"
    OTHER = "other"


class ReliefPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReliefDeliveryStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    PREPARING = "preparing"
    DISPATCHED = "dispatched"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    VERIFIED = "verified"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CheckpointStatus(str, Enum):
    ARRIVED = "arrived"
    DEPARTED = "departed"
    DELAYED = "delayed"
    BLOCKED = "blocked"
    REROUTED = "rerouted"


class ReliefExceptionType(str, Enum):
    STOCK_SHORTAGE = "stock_shortage"
    ROAD_BLOCKAGE = "road_blockage"
    VEHICLE_BREAKDOWN = "vehicle_breakdown"
    WEATHER_DISRUPTION = "weather_disruption"
    DELIVERY_MISMATCH = "delivery_mismatch"
    DAMAGED_GOODS = "damaged_goods"
    MISSING_GOODS = "missing_goods"
    OTHER = "other"


class ReliefReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ReliefDeliveryCreate(BaseModel):
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
        max_length=5000,
    )

    priority: ReliefPriority = ReliefPriority.MEDIUM

    source_location_id: str

    destination_location_id: str

    beneficiary_group_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    expected_beneficiaries: int = Field(
        default=0,
        ge=0,
    )

    requested_delivery_at: datetime | None = None

    incident_id: str | None = None

    @field_validator(
        "source_location_id",
        "destination_location_id",
    )
    @classmethod
    def validate_location_id(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))

    @field_validator("incident_id")
    @classmethod
    def validate_incident_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return str(uuid.UUID(value))

    @model_validator(mode="after")
    def validate_locations(
        self,
    ) -> "ReliefDeliveryCreate":
        if (
            self.source_location_id
            == self.destination_location_id
        ):
            raise ValueError(
                "Source and destination must differ"
            )

        return self


class ReliefDeliveryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None,
        min_length=5,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
        min_length=10,
        max_length=5000,
    )

    priority: ReliefPriority | None = None

    expected_beneficiaries: int | None = Field(
        default=None,
        ge=0,
    )

    requested_delivery_at: datetime | None = None


class ReliefDeliveryItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inventory_item_id: str

    category: ReliefCategory

    quantity: Decimal = Field(gt=0)

    unit: str = Field(
        min_length=1,
        max_length=50,
    )

    notes: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator("inventory_item_id")
    @classmethod
    def validate_inventory_id(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))


class ReliefCheckpointCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    status: CheckpointStatus

    observed_at: datetime

    notes: str = Field(
        min_length=5,
        max_length=3000,
    )

    @field_validator("observed_at")
    @classmethod
    def validate_time(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "Timezone-aware timestamp required"
            )

        if value > datetime.now(timezone.utc):
            raise ValueError(
                "Checkpoint timestamp cannot be in the future"
            )

        return value


class ReliefDeliveryProofCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    received_by_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    received_at: datetime

    quantity_verified: bool = False

    recipient_verified: bool = False

    notes: str | None = Field(
        default=None,
        max_length=3000,
    )


class ReliefExceptionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exception_type: ReliefExceptionType

    description: str = Field(
        min_length=10,
        max_length=5000,
    )

    severity: ReliefPriority

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )


class ReliefSubmissionRequest(BaseModel):
    notes: str = Field(
        min_length=10,
        max_length=3000,
    )

    inventory_checked: bool = False

    destination_verified: bool = False


class ReliefApprovalRequest(BaseModel):
    action: ReliefReviewAction

    review_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    inventory_verified: bool = False

    beneficiary_need_verified: bool = False

    destination_verified: bool = False


class ReliefDispatchRequest(BaseModel):
    transport_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    driver_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    dispatch_notes: str = Field(
        min_length=10,
        max_length=3000,
    )

    route_verified: bool = False

    inventory_reserved: bool = False

    transport_ready: bool = False


class ReliefCompletionRequest(BaseModel):
    completed_at: datetime

    completion_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    delivery_proof_verified: bool = False

    recipient_verified: bool = False

    quantity_verified: bool = False


class ReliefCancellationRequest(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=5000,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ReliefDeliveryResponse(BaseModel):
    id: str

    district_id: int

    title: str

    priority: ReliefPriority

    status: ReliefDeliveryStatus

    created_at: datetime

    updated_at: datetime | None = None


class ReliefDeliveryListResponse(BaseModel):
    items: list[ReliefDeliveryResponse]

    total: int

    limit: int

    offset: int


class ReliefDeliveryItemResponse(BaseModel):
    id: str

    delivery_id: str

    category: ReliefCategory

    quantity: Decimal

    unit: str


class ReliefCheckpointResponse(BaseModel):
    id: str

    delivery_id: str

    status: CheckpointStatus

    observed_at: datetime


class ReliefProofResponse(BaseModel):
    id: str

    delivery_id: str

    verified: bool

    created_at: datetime


class ReliefExceptionResponse(BaseModel):
    id: str

    delivery_id: str

    exception_type: ReliefExceptionType

    resolved: bool

    created_at: datetime


class ReliefEventResponse(BaseModel):
    id: str

    delivery_id: str

    event_type: str

    created_at: datetime


class ReliefStatisticsResponse(BaseModel):
    total_deliveries: int

    by_status: dict[str, int]

    by_priority: dict[str, int]

    completed_deliveries: int

    generated_at: datetime


class ReliefHealthResponse(BaseModel):
    service: str

    status: str

    timestamp: datetime


# ============================================================
# SERVICE CONTRACT
# ============================================================

class ReliefService(Protocol):
    """
    Required service responsibilities:

    - Persistent relief deliveries
    - Verified warehouse inventory
    - Inventory reservations
    - District and beneficiary authorization
    - Delivery scheduling
    - Human approval workflows
    - Verified dispatch and routing
    - Checkpoint ingestion
    - Proof-of-delivery validation
    - Exception escalation
    - Stock reconciliation
    - Immutable audit events
    """

    async def health(
        self,
    ) -> ReliefHealthResponse:
        ...

    async def list_deliveries(
        self,
        actor_id: str,
        district_id: int | None,
        limit: int,
        offset: int,
    ) -> ReliefDeliveryListResponse:
        ...

    async def statistics(
        self,
        actor_id: str,
        district_id: int | None,
    ) -> ReliefStatisticsResponse:
        ...

    async def create_delivery(
        self,
        payload: ReliefDeliveryCreate,
        actor_id: str,
    ) -> ReliefDeliveryResponse:
        ...

    async def get_delivery(
        self,
        delivery_id: str,
        actor_id: str,
    ) -> ReliefDeliveryResponse | None:
        ...

    async def update_delivery(
        self,
        delivery_id: str,
        payload: ReliefDeliveryUpdate,
        actor_id: str,
    ) -> ReliefDeliveryResponse:
        ...

    async def list_items(
        self,
        delivery_id: str,
        actor_id: str,
    ) -> list[ReliefDeliveryItemResponse]:
        ...

    async def add_item(
        self,
        delivery_id: str,
        payload: ReliefDeliveryItemCreate,
        actor_id: str,
    ) -> ReliefDeliveryItemResponse:
        ...

    async def list_checkpoints(
        self,
        delivery_id: str,
        actor_id: str,
    ) -> list[ReliefCheckpointResponse]:
        ...

    async def add_checkpoint(
        self,
        delivery_id: str,
        payload: ReliefCheckpointCreate,
        actor_id: str,
    ) -> ReliefCheckpointResponse:
        ...

    async def list_proofs(
        self,
        delivery_id: str,
        actor_id: str,
    ) -> list[ReliefProofResponse]:
        ...

    async def add_proof(
        self,
        delivery_id: str,
        payload: ReliefDeliveryProofCreate,
        actor_id: str,
    ) -> ReliefProofResponse:
        ...

    async def list_exceptions(
        self,
        delivery_id: str,
        actor_id: str,
    ) -> list[ReliefExceptionResponse]:
        ...

    async def add_exception(
        self,
        delivery_id: str,
        payload: ReliefExceptionCreate,
        actor_id: str,
    ) -> ReliefExceptionResponse:
        ...

    async def submit_delivery(
        self,
        delivery_id: str,
        payload: ReliefSubmissionRequest,
        actor_id: str,
    ) -> ReliefDeliveryResponse:
        ...

    async def approve_delivery(
        self,
        delivery_id: str,
        payload: ReliefApprovalRequest,
        actor_id: str,
    ) -> ReliefDeliveryResponse:
        ...

    async def dispatch_delivery(
        self,
        delivery_id: str,
        payload: ReliefDispatchRequest,
        actor_id: str,
    ) -> ReliefDeliveryResponse:
        ...

    async def complete_delivery(
        self,
        delivery_id: str,
        payload: ReliefCompletionRequest,
        actor_id: str,
    ) -> ReliefDeliveryResponse:
        ...

    async def cancel_delivery(
        self,
        delivery_id: str,
        payload: ReliefCancellationRequest,
        actor_id: str,
    ) -> ReliefDeliveryResponse:
        ...

    async def list_events(
        self,
        delivery_id: str,
        actor_id: str,
    ) -> list[ReliefEventResponse]:
        ...


# ============================================================
# DEPENDENCY INJECTION
# ============================================================

def get_relief_service(
    request: Request,
) -> ReliefService:
    service = getattr(
        request.app.state,
        "relief_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Relief service is not configured",
        )

    return service


ReliefServiceDependency = Annotated[
    ReliefService,
    Depends(get_relief_service),
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


async def require_delivery_access(
    service: ReliefService,
    delivery_id: str,
    user: Any,
) -> ReliefDeliveryResponse:
    delivery = await service.get_delivery(
        delivery_id=delivery_id,
        actor_id=actor_id(user),
    )

    if delivery is None:
        raise HTTPException(
            status_code=404,
            detail="Relief delivery not found",
        )

    require_district_access(
        user,
        delivery.district_id,
    )

    return delivery


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=ReliefHealthResponse,
)
async def relief_health(
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefHealthResponse:
    require_role(current_user, READ_ROLES)

    return await service.health()


# ============================================================
# LIST DELIVERIES
# ============================================================

@router.get(
    "/deliveries",
    response_model=ReliefDeliveryListResponse,
)
async def list_relief_deliveries(
    current_user: CurrentUser,
    service: ReliefServiceDependency,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> ReliefDeliveryListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.list_deliveries(
        actor_id=actor_id(current_user),
        district_id=district_id,
        limit=limit,
        offset=offset,
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=ReliefStatisticsResponse,
)
async def relief_statistics(
    current_user: CurrentUser,
    service: ReliefServiceDependency,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> ReliefStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.statistics(
        actor_id=actor_id(current_user),
        district_id=district_id,
    )


# ============================================================
# CREATE DELIVERY
# ============================================================

@router.post(
    "/deliveries",
    response_model=ReliefDeliveryResponse,
    status_code=201,
)
async def create_relief_delivery(
    payload: ReliefDeliveryCreate,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryResponse:
    require_role(current_user, WRITE_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    return await service.create_delivery(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DELIVERY DETAILS
# ============================================================

@router.get(
    "/deliveries/{delivery_id}",
    response_model=ReliefDeliveryResponse,
)
async def get_relief_delivery(
    delivery_id: str,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryResponse:
    require_role(current_user, READ_ROLES)

    return await require_delivery_access(
        service,
        validate_uuid(delivery_id),
        current_user,
    )


# ============================================================
# UPDATE DELIVERY
# ============================================================

@router.patch(
    "/deliveries/{delivery_id}",
    response_model=ReliefDeliveryResponse,
)
async def update_relief_delivery(
    delivery_id: str,
    payload: ReliefDeliveryUpdate,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(delivery_id)

    delivery = await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    if delivery.status not in {
        ReliefDeliveryStatus.DRAFT,
        ReliefDeliveryStatus.PENDING_APPROVAL,
    }:
        raise HTTPException(
            status_code=409,
            detail="Delivery cannot be edited in current state",
        )

    return await service.update_delivery(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DELIVERY ITEMS
# ============================================================

@router.get(
    "/deliveries/{delivery_id}/items",
    response_model=list[ReliefDeliveryItemResponse],
)
async def list_relief_items(
    delivery_id: str,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> list[ReliefDeliveryItemResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(delivery_id)

    await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_items(
        delivery_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/deliveries/{delivery_id}/items",
    response_model=ReliefDeliveryItemResponse,
    status_code=201,
)
async def add_relief_item(
    delivery_id: str,
    payload: ReliefDeliveryItemCreate,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryItemResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(delivery_id)

    delivery = await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    if delivery.status != ReliefDeliveryStatus.DRAFT:
        raise HTTPException(
            status_code=409,
            detail="Items can only be added to draft deliveries",
        )

    return await service.add_item(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DELIVERY CHECKPOINTS
# ============================================================

@router.get(
    "/deliveries/{delivery_id}/checkpoints",
    response_model=list[ReliefCheckpointResponse],
)
async def list_delivery_checkpoints(
    delivery_id: str,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> list[ReliefCheckpointResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(delivery_id)

    await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_checkpoints(
        delivery_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/deliveries/{delivery_id}/checkpoints",
    response_model=ReliefCheckpointResponse,
    status_code=201,
)
async def add_delivery_checkpoint(
    delivery_id: str,
    payload: ReliefCheckpointCreate,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefCheckpointResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(delivery_id)

    delivery = await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    if delivery.status not in {
        ReliefDeliveryStatus.DISPATCHED,
        ReliefDeliveryStatus.IN_TRANSIT,
    }:
        raise HTTPException(
            status_code=409,
            detail="Delivery is not in transit",
        )

    return await service.add_checkpoint(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DELIVERY PROOF
# ============================================================

@router.get(
    "/deliveries/{delivery_id}/proofs",
    response_model=list[ReliefProofResponse],
)
async def list_delivery_proofs(
    delivery_id: str,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> list[ReliefProofResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(delivery_id)

    await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_proofs(
        delivery_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/deliveries/{delivery_id}/proofs",
    response_model=ReliefProofResponse,
    status_code=201,
)
async def add_delivery_proof(
    delivery_id: str,
    payload: ReliefDeliveryProofCreate,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefProofResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(delivery_id)

    await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    return await service.add_proof(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DELIVERY EXCEPTIONS
# ============================================================

@router.get(
    "/deliveries/{delivery_id}/exceptions",
    response_model=list[ReliefExceptionResponse],
)
async def list_delivery_exceptions(
    delivery_id: str,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> list[ReliefExceptionResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(delivery_id)

    await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_exceptions(
        delivery_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/deliveries/{delivery_id}/exceptions",
    response_model=ReliefExceptionResponse,
    status_code=201,
)
async def report_delivery_exception(
    delivery_id: str,
    payload: ReliefExceptionCreate,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefExceptionResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(delivery_id)

    await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    return await service.add_exception(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# SUBMIT DELIVERY
# ============================================================

@router.post(
    "/deliveries/{delivery_id}/submit",
    response_model=ReliefDeliveryResponse,
)
async def submit_relief_delivery(
    delivery_id: str,
    payload: ReliefSubmissionRequest,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(delivery_id)

    delivery = await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    if delivery.status != ReliefDeliveryStatus.DRAFT:
        raise HTTPException(
            status_code=409,
            detail="Only draft deliveries can be submitted",
        )

    if not (
        payload.inventory_checked
        and payload.destination_verified
    ):
        raise HTTPException(
            status_code=422,
            detail="Submission checks incomplete",
        )

    return await service.submit_delivery(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# APPROVE DELIVERY
# ============================================================

@router.post(
    "/deliveries/{delivery_id}/approve",
    response_model=ReliefDeliveryResponse,
)
async def approve_relief_delivery(
    delivery_id: str,
    payload: ReliefApprovalRequest,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryResponse:
    require_role(current_user, APPROVER_ROLES)

    identifier = validate_uuid(delivery_id)

    delivery = await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    if delivery.status != ReliefDeliveryStatus.PENDING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail="Delivery is not pending approval",
        )

    if payload.action == ReliefReviewAction.APPROVE:
        if not all(
            [
                payload.inventory_verified,
                payload.beneficiary_need_verified,
                payload.destination_verified,
            ]
        ):
            raise HTTPException(
                status_code=422,
                detail="Approval verification checks incomplete",
            )

    # Service must enforce independent human approval.
    return await service.approve_delivery(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DISPATCH DELIVERY
# ============================================================

@router.post(
    "/deliveries/{delivery_id}/dispatch",
    response_model=ReliefDeliveryResponse,
)
async def dispatch_relief_delivery(
    delivery_id: str,
    payload: ReliefDispatchRequest,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryResponse:
    require_role(current_user, DISPATCH_ROLES)

    identifier = validate_uuid(delivery_id)

    delivery = await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    if delivery.status not in {
        ReliefDeliveryStatus.APPROVED,
        ReliefDeliveryStatus.PREPARING,
    }:
        raise HTTPException(
            status_code=409,
            detail="Delivery is not approved for dispatch",
        )

    if not all(
        [
            payload.route_verified,
            payload.inventory_reserved,
            payload.transport_ready,
        ]
    ):
        raise HTTPException(
            status_code=422,
            detail="Dispatch readiness checks incomplete",
        )

    return await service.dispatch_delivery(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# COMPLETE DELIVERY
# ============================================================

@router.post(
    "/deliveries/{delivery_id}/complete",
    response_model=ReliefDeliveryResponse,
)
async def complete_relief_delivery(
    delivery_id: str,
    payload: ReliefCompletionRequest,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryResponse:
    require_role(current_user, APPROVER_ROLES)

    identifier = validate_uuid(delivery_id)

    delivery = await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    if delivery.status not in {
        ReliefDeliveryStatus.DELIVERED,
        ReliefDeliveryStatus.VERIFIED,
    }:
        raise HTTPException(
            status_code=409,
            detail="Delivery is not ready for completion",
        )

    if not all(
        [
            payload.delivery_proof_verified,
            payload.recipient_verified,
            payload.quantity_verified,
        ]
    ):
        raise HTTPException(
            status_code=422,
            detail="Delivery verification checks incomplete",
        )

    return await service.complete_delivery(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# CANCEL DELIVERY
# ============================================================

@router.post(
    "/deliveries/{delivery_id}/cancel",
    response_model=ReliefDeliveryResponse,
)
async def cancel_relief_delivery(
    delivery_id: str,
    payload: ReliefCancellationRequest,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> ReliefDeliveryResponse:
    require_role(current_user, APPROVER_ROLES)

    identifier = validate_uuid(delivery_id)

    delivery = await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    if delivery.status in {
        ReliefDeliveryStatus.COMPLETED,
        ReliefDeliveryStatus.CANCELLED,
    }:
        raise HTTPException(
            status_code=409,
            detail="Delivery is already finalized",
        )

    # Service must reconcile reserved inventory.
    return await service.cancel_delivery(
        delivery_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get(
    "/deliveries/{delivery_id}/events",
    response_model=list[ReliefEventResponse],
)
async def list_relief_events(
    delivery_id: str,
    current_user: CurrentUser,
    service: ReliefServiceDependency,
) -> list[ReliefEventResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(delivery_id)

    await require_delivery_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_events(
        delivery_id=identifier,
        actor_id=actor_id(current_user),
    )


__all__ = [
    "router",
    "ReliefService",
    "ReliefCategory",
    "ReliefPriority",
    "ReliefDeliveryStatus",
    "CheckpointStatus",
    "ReliefExceptionType",
    "ReliefReviewAction",
    "ReliefDeliveryCreate",
    "ReliefDeliveryUpdate",
    "ReliefDeliveryItemCreate",
    "ReliefCheckpointCreate",
    "ReliefDeliveryProofCreate",
    "ReliefExceptionCreate",
    "ReliefSubmissionRequest",
    "ReliefApprovalRequest",
    "ReliefDispatchRequest",
    "ReliefCompletionRequest",
    "ReliefCancellationRequest",
    "ReliefDeliveryResponse",
    "ReliefDeliveryListResponse",
    "ReliefStatisticsResponse",
]
