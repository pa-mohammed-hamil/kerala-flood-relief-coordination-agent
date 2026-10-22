from __future__ import annotations



"""
backend/api/recovery.py

Kerala Flood Relief Coordination Agent
Post-Flood Recovery & Rehabilitation API

Python 3.11+ | FastAPI | Pydantic v2

Endpoints:
    GET    /recovery/health
    GET    /recovery
    GET    /recovery/statistics
    POST   /recovery
    GET    /recovery/{case_id}
    PATCH  /recovery/{case_id}
    GET    /recovery/{case_id}/assessments
    POST   /recovery/{case_id}/assessments
    GET    /recovery/{case_id}/cost-estimates
    POST   /recovery/{case_id}/cost-estimates
    GET    /recovery/{case_id}/assistance
    POST   /recovery/{case_id}/assistance
    GET    /recovery/{case_id}/milestones
    POST   /recovery/{case_id}/milestones
    GET    /recovery/{case_id}/progress
    POST   /recovery/{case_id}/progress
    POST   /recovery/{case_id}/submit
    POST   /recovery/{case_id}/review
    POST   /recovery/{case_id}/complete
    GET    /recovery/{case_id}/events

Safety:
    - AI cost estimates are advisory.
    - Compensation is never automatically approved.
    - Financial assistance requires authorized review.
    - Recovery completion requires verified evidence.
    - Personal and financial information is restricted.
    - Every operational change must be audited.
"""


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
    status,
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
    prefix="/recovery",
    tags=["Flood Recovery & Rehabilitation"],
)

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# CONFIGURATION
# ============================================================

ADMIN_ROLES = {
    "admin",
    "super_admin",
}

RECOVERY_ROLES = {
    "recovery_officer",
    "district_officer",
    "rehabilitation_coordinator",
    "damage_assessor",
}

FINANCE_ROLES = {
    "finance_officer",
    "compensation_reviewer",
}

ENGINEERING_ROLES = {
    "civil_engineer",
    "infrastructure_inspector",
}

READ_ROLES = (
    ADMIN_ROLES
    | RECOVERY_ROLES
    | FINANCE_ROLES
    | ENGINEERING_ROLES
    | {"emergency_coordinator", "auditor"}
)

WRITE_ROLES = (
    ADMIN_ROLES
    | RECOVERY_ROLES
    | ENGINEERING_ROLES
)

REVIEW_ROLES = (
    ADMIN_ROLES
    | {
        "district_officer",
        "rehabilitation_coordinator",
        "compensation_reviewer",
    }
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class RecoveryCategory(str, Enum):
    RESIDENTIAL_BUILDING = "residential_building"
    COMMERCIAL_BUILDING = "commercial_building"
    ROAD = "road"
    BRIDGE = "bridge"
    ELECTRICITY = "electricity"
    WATER_SUPPLY = "water_supply"
    SANITATION = "sanitation"
    HEALTHCARE_FACILITY = "healthcare_facility"
    SCHOOL = "school"
    AGRICULTURE = "agriculture"
    LIVELIHOOD = "livelihood"
    PUBLIC_INFRASTRUCTURE = "public_infrastructure"
    OTHER = "other"


class RecoveryPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStatus(str, Enum):
    REPORTED = "reported"
    UNDER_ASSESSMENT = "under_assessment"
    DRAFT_PLAN = "draft_plan"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    VERIFICATION_PENDING = "verification_pending"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class DamageSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"
    TOTAL_LOSS = "total_loss"


class AssistanceType(str, Enum):
    FINANCIAL = "financial"
    TEMPORARY_HOUSING = "temporary_housing"
    MATERIALS = "materials"
    FOOD = "food"
    MEDICAL = "medical"
    LIVELIHOOD = "livelihood"
    RECONSTRUCTION = "reconstruction"
    OTHER = "other"


class ReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class CostCategory(str, Enum):
    MATERIALS = "materials"
    LABOR = "labor"
    EQUIPMENT = "equipment"
    TRANSPORT = "transport"
    DEBRIS_REMOVAL = "debris_removal"
    REPLACEMENT = "replacement"
    OTHER = "other"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class RecoveryCaseCreate(BaseModel):
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

    category: RecoveryCategory

    priority: RecoveryPriority = (
        RecoveryPriority.MEDIUM
    )

    damage_report_id: str | None = None

    affected_households: int = Field(
        default=0,
        ge=0,
    )

    affected_people: int = Field(
        default=0,
        ge=0,
    )

    location_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    source_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    @field_validator("damage_report_id")
    @classmethod
    def validate_damage_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        return str(uuid.UUID(value))


class RecoveryCaseUpdate(BaseModel):
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

    priority: RecoveryPriority | None = None

    affected_households: int | None = Field(
        default=None,
        ge=0,
    )

    affected_people: int | None = Field(
        default=None,
        ge=0,
    )


class RecoveryAssessmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: DamageSeverity

    findings: str = Field(
        min_length=10,
        max_length=10000,
    )

    recommended_actions: str = Field(
        min_length=10,
        max_length=5000,
    )

    safety_risk_identified: bool = False

    requires_engineering_review: bool = False

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    ai_assisted: bool = False

    model_reference: str | None = Field(
        default=None,
        max_length=150,
    )


class RecoveryCostItem(BaseModel):
    category: CostCategory

    description: str = Field(
        min_length=3,
        max_length=500,
    )

    quantity: Decimal = Field(gt=0)

    unit_cost_inr: Decimal = Field(ge=0)

    @property
    def total_cost(self) -> Decimal:
        return self.quantity * self.unit_cost_inr


class RecoveryCostEstimateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RecoveryCostItem] = Field(
        min_length=1,
        max_length=200,
    )

    contingency_percent: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        le=50,
    )

    justification: str = Field(
        min_length=10,
        max_length=5000,
    )

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    ai_generated: bool = False


class RecoveryAssistanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistance_type: AssistanceType

    description: str = Field(
        min_length=10,
        max_length=5000,
    )

    requested_amount_inr: Decimal | None = Field(
        default=None,
        ge=0,
    )

    beneficiary_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    justification: str = Field(
        min_length=10,
        max_length=5000,
    )


class RecoveryMilestoneCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=5,
        max_length=200,
    )

    description: str = Field(
        min_length=10,
        max_length=3000,
    )

    target_date: datetime

    responsible_team_reference: str | None = Field(
        default=None,
        max_length=150,
    )


class RecoveryProgressUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    progress_percent: float = Field(
        ge=0,
        le=100,
    )

    progress_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )


class RecoverySubmissionRequest(BaseModel):
    submission_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    evidence_verified: bool = False


class RecoveryReviewRequest(BaseModel):
    action: ReviewAction

    review_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    assessment_verified: bool = False

    cost_estimate_verified: bool = False

    beneficiary_eligibility_verified: bool = False


class RecoveryCompletionRequest(BaseModel):
    completed_at: datetime

    completion_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    work_verified: bool = False

    safety_verified: bool = False

    evidence_verified: bool = False


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class RecoveryCaseResponse(BaseModel):
    id: str

    district_id: int

    title: str

    category: RecoveryCategory

    priority: RecoveryPriority

    status: RecoveryStatus

    progress_percent: float = Field(
        default=0,
        ge=0,
        le=100,
    )

    created_at: datetime

    updated_at: datetime | None = None


class RecoveryCaseListResponse(BaseModel):
    items: list[RecoveryCaseResponse]

    total: int

    limit: int

    offset: int


class RecoveryAssessmentResponse(BaseModel):
    id: str

    recovery_case_id: str

    severity: DamageSeverity

    created_at: datetime


class RecoveryCostEstimateResponse(BaseModel):
    id: str

    recovery_case_id: str

    subtotal_inr: Decimal

    contingency_inr: Decimal

    total_estimated_inr: Decimal

    approved: bool = False

    created_at: datetime


class RecoveryAssistanceResponse(BaseModel):
    id: str

    recovery_case_id: str

    assistance_type: AssistanceType

    status: str

    created_at: datetime


class RecoveryMilestoneResponse(BaseModel):
    id: str

    recovery_case_id: str

    title: str

    status: str

    target_date: datetime


class RecoveryProgressResponse(BaseModel):
    id: str

    recovery_case_id: str

    progress_percent: float

    created_at: datetime


class RecoveryEventResponse(BaseModel):
    id: str

    recovery_case_id: str

    event_type: str

    created_at: datetime


class RecoveryStatisticsResponse(BaseModel):
    total_cases: int

    by_status: dict[str, int]

    by_category: dict[str, int]

    completed_cases: int

    average_progress_percent: float

    generated_at: datetime


class RecoveryHealthResponse(BaseModel):
    service: str

    status: str

    timestamp: datetime


# ============================================================
# SERVICE CONTRACT
# ============================================================

class RecoveryService(Protocol):
    """
    Domain service requirements:

    - Persist cases and assessments
    - Verify district and asset ownership
    - Prevent duplicate compensation claims
    - Validate beneficiary eligibility
    - Maintain versioned cost estimates
    - Track milestones and progress
    - Enforce approved state transitions
    - Require independent reviewers
    - Protect financial and personal data
    - Create immutable audit events
    """

    async def health(
        self,
    ) -> RecoveryHealthResponse:
        ...

    async def list_cases(
        self,
        actor_id: str,
        district_id: int | None,
        limit: int,
        offset: int,
    ) -> RecoveryCaseListResponse:
        ...

    async def get_statistics(
        self,
        actor_id: str,
        district_id: int | None,
    ) -> RecoveryStatisticsResponse:
        ...

    async def create_case(
        self,
        payload: RecoveryCaseCreate,
        actor_id: str,
    ) -> RecoveryCaseResponse:
        ...

    async def get_case(
        self,
        case_id: str,
        actor_id: str,
    ) -> RecoveryCaseResponse | None:
        ...

    async def update_case(
        self,
        case_id: str,
        payload: RecoveryCaseUpdate,
        actor_id: str,
    ) -> RecoveryCaseResponse:
        ...

    async def list_assessments(
        self,
        case_id: str,
        actor_id: str,
    ) -> list[RecoveryAssessmentResponse]:
        ...

    async def create_assessment(
        self,
        case_id: str,
        payload: RecoveryAssessmentCreate,
        actor_id: str,
    ) -> RecoveryAssessmentResponse:
        ...

    async def list_cost_estimates(
        self,
        case_id: str,
        actor_id: str,
    ) -> list[RecoveryCostEstimateResponse]:
        ...

    async def create_cost_estimate(
        self,
        case_id: str,
        payload: RecoveryCostEstimateCreate,
        actor_id: str,
    ) -> RecoveryCostEstimateResponse:
        ...

    async def list_assistance(
        self,
        case_id: str,
        actor_id: str,
    ) -> list[RecoveryAssistanceResponse]:
        ...

    async def request_assistance(
        self,
        case_id: str,
        payload: RecoveryAssistanceRequest,
        actor_id: str,
    ) -> RecoveryAssistanceResponse:
        ...

    async def list_milestones(
        self,
        case_id: str,
        actor_id: str,
    ) -> list[RecoveryMilestoneResponse]:
        ...

    async def create_milestone(
        self,
        case_id: str,
        payload: RecoveryMilestoneCreate,
        actor_id: str,
    ) -> RecoveryMilestoneResponse:
        ...

    async def list_progress(
        self,
        case_id: str,
        actor_id: str,
    ) -> list[RecoveryProgressResponse]:
        ...

    async def update_progress(
        self,
        case_id: str,
        payload: RecoveryProgressUpdate,
        actor_id: str,
    ) -> RecoveryProgressResponse:
        ...

    async def submit_case(
        self,
        case_id: str,
        payload: RecoverySubmissionRequest,
        actor_id: str,
    ) -> RecoveryCaseResponse:
        ...

    async def review_case(
        self,
        case_id: str,
        payload: RecoveryReviewRequest,
        actor_id: str,
    ) -> RecoveryCaseResponse:
        ...

    async def complete_case(
        self,
        case_id: str,
        payload: RecoveryCompletionRequest,
        actor_id: str,
    ) -> RecoveryCaseResponse:
        ...

    async def list_events(
        self,
        case_id: str,
        actor_id: str,
    ) -> list[RecoveryEventResponse]:
        ...


# ============================================================
# DEPENDENCY INJECTION
# ============================================================

def get_recovery_service(
    request: Request,
) -> RecoveryService:
    service = getattr(
        request.app.state,
        "recovery_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recovery service is not configured",
        )

    return service


RecoveryServiceDependency = Annotated[
    RecoveryService,
    Depends(get_recovery_service),
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


async def require_case_access(
    service: RecoveryService,
    case_id: str,
    user: Any,
) -> RecoveryCaseResponse:
    case = await service.get_case(
        case_id=case_id,
        actor_id=actor_id(user),
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Recovery case not found",
        )

    require_district_access(
        user,
        case.district_id,
    )

    return case


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=RecoveryHealthResponse,
)
async def recovery_health(
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryHealthResponse:
    require_role(current_user, READ_ROLES)

    return await service.health()


# ============================================================
# LIST RECOVERY CASES
# ============================================================

@router.get(
    "",
    response_model=RecoveryCaseListResponse,
)
async def list_recovery_cases(
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
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
) -> RecoveryCaseListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.list_cases(
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
    response_model=RecoveryStatisticsResponse,
)
async def recovery_statistics(
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> RecoveryStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    return await service.get_statistics(
        actor_id=actor_id(current_user),
        district_id=district_id,
    )


# ============================================================
# CREATE RECOVERY CASE
# ============================================================

@router.post(
    "",
    response_model=RecoveryCaseResponse,
    status_code=201,
)
async def create_recovery_case(
    payload: RecoveryCaseCreate,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryCaseResponse:
    require_role(current_user, WRITE_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    return await service.create_case(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# RECOVERY CASE DETAILS
# ============================================================

@router.get(
    "/{case_id}",
    response_model=RecoveryCaseResponse,
)
async def get_recovery_case(
    case_id: str,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryCaseResponse:
    require_role(current_user, READ_ROLES)

    return await require_case_access(
        service,
        validate_uuid(case_id),
        current_user,
    )


# ============================================================
# UPDATE RECOVERY CASE
# ============================================================

@router.patch(
    "/{case_id}",
    response_model=RecoveryCaseResponse,
)
async def update_recovery_case(
    case_id: str,
    payload: RecoveryCaseUpdate,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryCaseResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(case_id)

    case = await require_case_access(
        service,
        identifier,
        current_user,
    )

    if case.status in {
        RecoveryStatus.COMPLETED,
        RecoveryStatus.CANCELLED,
        RecoveryStatus.REJECTED,
    }:
        raise HTTPException(
            status_code=409,
            detail="Finalized recovery case cannot be edited",
        )

    return await service.update_case(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# ASSESSMENTS
# ============================================================

@router.get(
    "/{case_id}/assessments",
    response_model=list[RecoveryAssessmentResponse],
)
async def list_recovery_assessments(
    case_id: str,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> list[RecoveryAssessmentResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_assessments(
        case_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{case_id}/assessments",
    response_model=RecoveryAssessmentResponse,
    status_code=201,
)
async def create_recovery_assessment(
    case_id: str,
    payload: RecoveryAssessmentCreate,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryAssessmentResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    return await service.create_assessment(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# COST ESTIMATES
# ============================================================

@router.get(
    "/{case_id}/cost-estimates",
    response_model=list[RecoveryCostEstimateResponse],
)
async def list_recovery_cost_estimates(
    case_id: str,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> list[RecoveryCostEstimateResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_cost_estimates(
        case_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{case_id}/cost-estimates",
    response_model=RecoveryCostEstimateResponse,
    status_code=201,
)
async def create_recovery_cost_estimate(
    case_id: str,
    payload: RecoveryCostEstimateCreate,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryCostEstimateResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    # A cost estimate does not authorize payment.
    return await service.create_cost_estimate(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# ASSISTANCE
# ============================================================

@router.get(
    "/{case_id}/assistance",
    response_model=list[RecoveryAssistanceResponse],
)
async def list_recovery_assistance(
    case_id: str,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> list[RecoveryAssistanceResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_assistance(
        case_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{case_id}/assistance",
    response_model=RecoveryAssistanceResponse,
    status_code=201,
)
async def request_recovery_assistance(
    case_id: str,
    payload: RecoveryAssistanceRequest,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryAssistanceResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    # Request creation must not disburse funds.
    return await service.request_assistance(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# MILESTONES
# ============================================================

@router.get(
    "/{case_id}/milestones",
    response_model=list[RecoveryMilestoneResponse],
)
async def list_recovery_milestones(
    case_id: str,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> list[RecoveryMilestoneResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_milestones(
        case_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{case_id}/milestones",
    response_model=RecoveryMilestoneResponse,
    status_code=201,
)
async def create_recovery_milestone(
    case_id: str,
    payload: RecoveryMilestoneCreate,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryMilestoneResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    return await service.create_milestone(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# PROGRESS TRACKING
# ============================================================

@router.get(
    "/{case_id}/progress",
    response_model=list[RecoveryProgressResponse],
)
async def list_recovery_progress(
    case_id: str,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> list[RecoveryProgressResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_progress(
        case_id=identifier,
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{case_id}/progress",
    response_model=RecoveryProgressResponse,
    status_code=201,
)
async def update_recovery_progress(
    case_id: str,
    payload: RecoveryProgressUpdate,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryProgressResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(case_id)

    case = await require_case_access(
        service,
        identifier,
        current_user,
    )

    if case.status not in {
        RecoveryStatus.APPROVED,
        RecoveryStatus.IN_PROGRESS,
    }:
        raise HTTPException(
            status_code=409,
            detail="Recovery case is not approved for work",
        )

    return await service.update_progress(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# SUBMIT FOR REVIEW
# ============================================================

@router.post(
    "/{case_id}/submit",
    response_model=RecoveryCaseResponse,
)
async def submit_recovery_case(
    case_id: str,
    payload: RecoverySubmissionRequest,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryCaseResponse:
    require_role(current_user, WRITE_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    if not payload.evidence_verified:
        raise HTTPException(
            status_code=422,
            detail="Supporting evidence must be verified",
        )

    return await service.submit_case(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# REVIEW RECOVERY CASE
# ============================================================

@router.post(
    "/{case_id}/review",
    response_model=RecoveryCaseResponse,
)
async def review_recovery_case(
    case_id: str,
    payload: RecoveryReviewRequest,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryCaseResponse:
    require_role(current_user, REVIEW_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    if payload.action == ReviewAction.APPROVE:
        if not all(
            [
                payload.assessment_verified,
                payload.cost_estimate_verified,
                payload.beneficiary_eligibility_verified,
            ]
        ):
            raise HTTPException(
                status_code=422,
                detail="Recovery approval checks incomplete",
            )

    # Service must enforce reviewer independence.
    return await service.review_case(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# COMPLETE RECOVERY CASE
# ============================================================

@router.post(
    "/{case_id}/complete",
    response_model=RecoveryCaseResponse,
)
async def complete_recovery_case(
    case_id: str,
    payload: RecoveryCompletionRequest,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> RecoveryCaseResponse:
    require_role(current_user, REVIEW_ROLES)

    identifier = validate_uuid(case_id)

    case = await require_case_access(
        service,
        identifier,
        current_user,
    )

    if case.status not in {
        RecoveryStatus.IN_PROGRESS,
        RecoveryStatus.VERIFICATION_PENDING,
    }:
        raise HTTPException(
            status_code=409,
            detail="Recovery case is not ready for completion",
        )

    if not all(
        [
            payload.work_verified,
            payload.safety_verified,
            payload.evidence_verified,
        ]
    ):
        raise HTTPException(
            status_code=422,
            detail="Completion verification checks incomplete",
        )

    return await service.complete_case(
        case_id=identifier,
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get(
    "/{case_id}/events",
    response_model=list[RecoveryEventResponse],
)
async def list_recovery_events(
    case_id: str,
    current_user: CurrentUser,
    service: RecoveryServiceDependency,
) -> list[RecoveryEventResponse]:
    require_role(current_user, READ_ROLES)

    identifier = validate_uuid(case_id)

    await require_case_access(
        service,
        identifier,
        current_user,
    )

    return await service.list_events(
        case_id=identifier,
        actor_id=actor_id(current_user),
    )


__all__ = [
    "router",
    "RecoveryService",
    "RecoveryCategory",
    "RecoveryPriority",
    "RecoveryStatus",
    "DamageSeverity",
    "AssistanceType",
    "ReviewAction",
    "CostCategory",
    "RecoveryCaseCreate",
    "RecoveryCaseUpdate",
    "RecoveryAssessmentCreate",
    "RecoveryCostEstimateCreate",
    "RecoveryAssistanceRequest",
    "RecoveryMilestoneCreate",
    "RecoveryProgressUpdate",
    "RecoverySubmissionRequest",
    "RecoveryReviewRequest",
    "RecoveryCompletionRequest",
    "RecoveryCaseResponse",
    "RecoveryCaseListResponse",
    "RecoveryStatisticsResponse",
]
