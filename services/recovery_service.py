from __future__ import annotations

"""
backend/services/recovery_service.py

Kerala Flood Relief Coordination Agent
=======================================

Flood Recovery, Rehabilitation and Reconstruction Service.

Features
--------
- Post-flood recovery case registration
- Building and infrastructure damage assessment linkage
- Household rehabilitation coordination
- Recovery priority evaluation
- Repair and reconstruction cost estimation
- Material and workforce requirement tracking
- Contractor and field-team assignment
- Recovery project planning
- Human approval workflow
- Budget and funding allocation
- Milestone tracking
- Progress verification
- Evidence-based completion
- Recovery risk and delay monitoring
- District recovery statistics
- Audit logging
- Structured recovery reports

SAFETY
------
- Estimates are indicative, not official compensation values.
- Structural safety requires qualified professional inspection.
- No automatic fund transfer, demolition or construction order.
- Approval and completion require authorized human review.
- Actual funding and resource reservations require transactional
  integrations and reconciliation.
- Sensitive household data should remain in protected systems.
"""


import logging
import math
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)

MONEY_QUANTUM = Decimal("0.01")


# ============================================================
# ENUMERATIONS
# ============================================================

class RecoveryCategory(str, Enum):
    RESIDENTIAL_BUILDING = "residential_building"
    COMMERCIAL_BUILDING = "commercial_building"
    PUBLIC_BUILDING = "public_building"
    ROAD = "road"
    BRIDGE = "bridge"
    ELECTRICITY = "electricity"
    WATER_SUPPLY = "water_supply"
    SANITATION = "sanitation"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    AGRICULTURE = "agriculture"
    LIVELIHOOD = "livelihood"
    ENVIRONMENT = "environment"
    OTHER = "other"


class RecoveryPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStatus(str, Enum):
    DRAFT = "draft"
    ASSESSMENT_PENDING = "assessment_pending"
    ASSESSED = "assessed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    FUNDED = "funded"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    PENDING_VERIFICATION = "pending_verification"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DamageLevel(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    DESTROYED = "destroyed"


class RecoveryWorkType(str, Enum):
    CLEANUP = "cleanup"
    DEBRIS_REMOVAL = "debris_removal"
    REPAIR = "repair"
    RECONSTRUCTION = "reconstruction"
    REPLACEMENT = "replacement"
    RESTORATION = "restoration"
    REHABILITATION = "rehabilitation"
    SAFETY_INSPECTION = "safety_inspection"


class MilestoneStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    VERIFIED = "verified"


class FundingStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    RESERVED = "reserved"
    RELEASED = "released"
    REJECTED = "rejected"


class RecoveryRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# EXCEPTIONS
# ============================================================

class RecoveryServiceError(Exception):
    pass


class RecoveryValidationError(RecoveryServiceError):
    pass


class RecoveryNotFoundError(RecoveryServiceError):
    pass


class RecoveryConflictError(RecoveryServiceError):
    pass


class RecoveryAuthorizationError(RecoveryServiceError):
    pass


class RecoveryFundingError(RecoveryServiceError):
    pass


class RecoveryVerificationError(RecoveryServiceError):
    pass


class RecoveryIntegrationError(RecoveryServiceError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_text(
    value: str,
    name: str,
    *,
    max_length: int = 1000,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryValidationError(
            f"{name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise RecoveryValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RecoveryValidationError(
            f"{name} must be timezone-aware"
        )


def money(
    value: Decimal | int | float | str,
) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RecoveryValidationError(
            "Invalid monetary value"
        ) from exc

    if not result.is_finite() or result < 0:
        raise RecoveryValidationError(
            "Monetary value must be finite and non-negative"
        )

    return result.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def validate_ratio(
    value: float,
    name: str,
) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RecoveryValidationError(
            f"{name} must be between 0 and 1"
        )


def validate_progress(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise RecoveryValidationError(
            "Progress must be between 0 and 100"
        )


# ============================================================
# DOMAIN MODELS
# ============================================================

@dataclass(frozen=True)
class RecoveryLocation:
    district_id: int
    latitude: float
    longitude: float
    address: str = ""
    panchayat: str = ""
    village: str = ""

    def __post_init__(self) -> None:
        if self.district_id <= 0:
            raise RecoveryValidationError(
                "Invalid district ID"
            )

        if (
            not math.isfinite(self.latitude)
            or not -90 <= self.latitude <= 90
        ):
            raise RecoveryValidationError(
                "Invalid latitude"
            )

        if (
            not math.isfinite(self.longitude)
            or not -180 <= self.longitude <= 180
        ):
            raise RecoveryValidationError(
                "Invalid longitude"
            )


@dataclass(frozen=True)
class RecoveryCostItem:
    description: str
    quantity: Decimal
    unit: str
    unit_cost_inr: Decimal

    category: str = "general"

    def __post_init__(self) -> None:
        require_text(
            self.description,
            "Cost item description",
        )

        require_text(
            self.unit,
            "Cost item unit",
        )

        if (
            not self.quantity.is_finite()
            or self.quantity <= 0
        ):
            raise RecoveryValidationError(
                "Cost item quantity must be positive"
            )

        money(self.unit_cost_inr)

    @property
    def total_cost_inr(self) -> Decimal:
        return money(
            self.quantity * self.unit_cost_inr
        )


@dataclass(frozen=True)
class RecoveryAssessment:
    assessment_id: str

    damage_level: DamageLevel
    assessed_by: int
    assessed_at: datetime

    structural_safety_verified: bool = False
    safe_for_occupancy: bool | None = None

    affected_people: int = 0
    displaced_people: int = 0

    essential_service_disruption: bool = False
    critical_infrastructure: bool = False

    verified: bool = False

    evidence_references: tuple[str, ...] = ()
    findings: str = ""

    def __post_init__(self) -> None:
        validate_datetime(
            self.assessed_at,
            "assessed_at",
        )

        if self.affected_people < 0:
            raise RecoveryValidationError(
                "Affected population cannot be negative"
            )

        if (
            self.displaced_people < 0
            or self.displaced_people > self.affected_people
        ):
            raise RecoveryValidationError(
                "Invalid displaced population"
            )


@dataclass(frozen=True)
class RecoveryCreateRequest:
    incident_id: str
    location: RecoveryLocation

    title: str
    description: str

    category: RecoveryCategory
    work_type: RecoveryWorkType

    damage_report_id: str | None = None

    beneficiary_reference: str | None = None

    estimated_start_date: datetime | None = None
    estimated_end_date: datetime | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RecoveryCostEstimate:
    direct_cost_inr: Decimal
    contingency_inr: Decimal
    total_estimated_cost_inr: Decimal

    contingency_rate: Decimal

    currency: str = "INR"

    items: tuple[RecoveryCostItem, ...] = ()

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class RecoveryPriorityAssessment:
    priority: RecoveryPriority
    risk_level: RecoveryRiskLevel

    score: float

    reasons: tuple[str, ...]

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass
class RecoveryMilestone:
    id: str
    title: str
    description: str

    target_date: datetime

    status: MilestoneStatus = (
        MilestoneStatus.NOT_STARTED
    )

    progress_percent: float = 0.0

    assigned_team_id: str | None = None

    completed_at: datetime | None = None
    verified_at: datetime | None = None

    verified_by: int | None = None

    evidence_references: list[str] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class RecoveryProgressUpdate:
    milestone_id: str
    progress_percent: float

    notes: str = ""

    evidence_references: tuple[str, ...] = ()

    reported_at: datetime = field(
        default_factory=utc_now
    )


@dataclass
class RecoveryProject:
    id: str

    district_id: int
    incident_id: str

    location: RecoveryLocation

    title: str
    description: str

    category: RecoveryCategory
    work_type: RecoveryWorkType

    status: RecoveryStatus
    priority: RecoveryPriority

    created_by: int

    damage_report_id: str | None = None
    beneficiary_reference: str | None = None

    assessment: RecoveryAssessment | None = None
    priority_assessment: (
        RecoveryPriorityAssessment | None
    ) = None

    cost_estimate: RecoveryCostEstimate | None = None

    funding_status: FundingStatus = (
        FundingStatus.NOT_REQUESTED
    )

    reserved_funding_inr: Decimal = Decimal("0.00")

    assigned_team_id: str | None = None

    milestones: list[RecoveryMilestone] = field(
        default_factory=list
    )

    progress_percent: float = 0.0

    approved_by: int | None = None
    approval_reference: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    approved_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None

    estimated_start_date: datetime | None = None
    estimated_end_date: datetime | None = None

    cancellation_reason: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class RecoveryRepository(Protocol):

    async def create(
        self,
        project: RecoveryProject,
    ) -> RecoveryProject:
        ...

    async def get_by_id(
        self,
        project_id: str,
    ) -> RecoveryProject | None:
        ...

    async def update(
        self,
        project: RecoveryProject,
        *,
        expected_version: int,
    ) -> RecoveryProject:
        """
        Must atomically enforce optimistic concurrency.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[RecoveryProject]:
        ...

    async def list_by_incident(
        self,
        incident_id: str,
    ) -> Sequence[RecoveryProject]:
        ...


class RecoveryFundingGateway(Protocol):

    async def check_available_budget(
        self,
        *,
        district_id: int,
        amount_inr: Decimal,
    ) -> bool:
        ...

    async def reserve_budget(
        self,
        *,
        project_id: str,
        district_id: int,
        amount_inr: Decimal,
    ) -> bool:
        """
        Must atomically reserve budget and enforce
        idempotency using project_id.
        """
        ...

    async def release_budget(
        self,
        *,
        project_id: str,
    ) -> None:
        ...


class RecoveryTeamGateway(Protocol):

    async def verify_team(
        self,
        *,
        team_id: str,
        district_id: int,
        work_type: RecoveryWorkType,
    ) -> bool:
        ...

    async def reserve_team(
        self,
        *,
        team_id: str,
        project_id: str,
    ) -> bool:
        ...

    async def release_team(
        self,
        *,
        team_id: str,
        project_id: str,
    ) -> None:
        ...


class RecoveryAuthorizationGateway(Protocol):

    async def can_create(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_assess(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_approve(
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

    async def can_verify(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class RecoveryAuditGateway(Protocol):

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


# ============================================================
# COST ESTIMATOR
# ============================================================

class RecoveryCostEstimator:
    """
    Uses supplied cost items.

    No government compensation schedule or market
    price is assumed.
    """

    def __init__(
        self,
        *,
        default_contingency_rate: Decimal = (
            Decimal("0.10")
        ),
    ):
        if not (
            Decimal("0")
            <= default_contingency_rate
            <= Decimal("0.50")
        ):
            raise RecoveryValidationError(
                "Invalid contingency rate"
            )

        self.default_contingency_rate = (
            default_contingency_rate
        )

    def estimate(
        self,
        items: Sequence[RecoveryCostItem],
        *,
        contingency_rate: Decimal | None = None,
    ) -> RecoveryCostEstimate:

        if not items:
            raise RecoveryValidationError(
                "At least one cost item is required"
            )

        rate = (
            contingency_rate
            if contingency_rate is not None
            else self.default_contingency_rate
        )

        if not Decimal("0") <= rate <= Decimal("0.50"):
            raise RecoveryValidationError(
                "Contingency rate must be between 0 and 0.50"
            )

        direct_cost = money(
            sum(
                (
                    item.total_cost_inr
                    for item in items
                ),
                Decimal("0"),
            )
        )

        contingency = money(
            direct_cost * rate
        )

        total = money(
            direct_cost + contingency
        )

        return RecoveryCostEstimate(
            direct_cost_inr=direct_cost,
            contingency_inr=contingency,
            total_estimated_cost_inr=total,
            contingency_rate=rate,
            items=tuple(items),
        )


# ============================================================
# PRIORITY EVALUATOR
# ============================================================

class RecoveryPriorityEvaluator:
    """
    Illustrative recovery prioritization model.

    Scores are planning aids, not official
    eligibility or compensation criteria.
    """

    DAMAGE_SCORES = {
        DamageLevel.NONE: 0.0,
        DamageLevel.MINOR: 0.20,
        DamageLevel.MODERATE: 0.45,
        DamageLevel.SEVERE: 0.80,
        DamageLevel.DESTROYED: 1.0,
    }

    def evaluate(
        self,
        assessment: RecoveryAssessment,
    ) -> RecoveryPriorityAssessment:

        damage_score = self.DAMAGE_SCORES[
            assessment.damage_level
        ]

        population_score = min(
            1.0,
            assessment.affected_people / 100.0,
        )

        displacement_score = min(
            1.0,
            assessment.displaced_people / 50.0,
        )

        infrastructure_score = (
            1.0
            if assessment.critical_infrastructure
            else 0.0
        )

        service_score = (
            1.0
            if assessment.essential_service_disruption
            else 0.0
        )

        score = (
            damage_score * 0.30
            + population_score * 0.15
            + displacement_score * 0.20
            + infrastructure_score * 0.20
            + service_score * 0.15
        )

        reasons: list[str] = []

        if assessment.damage_level in {
            DamageLevel.SEVERE,
            DamageLevel.DESTROYED,
        }:
            reasons.append(
                "Severe physical damage"
            )

        if assessment.displaced_people > 0:
            reasons.append(
                "Population displacement"
            )

        if assessment.critical_infrastructure:
            reasons.append(
                "Critical infrastructure affected"
            )

        if assessment.essential_service_disruption:
            reasons.append(
                "Essential services disrupted"
            )

        if not assessment.structural_safety_verified:
            reasons.append(
                "Structural safety verification pending"
            )

        if score >= 0.75:
            priority = RecoveryPriority.CRITICAL
            risk = RecoveryRiskLevel.CRITICAL

        elif score >= 0.50:
            priority = RecoveryPriority.HIGH
            risk = RecoveryRiskLevel.HIGH

        elif score >= 0.25:
            priority = RecoveryPriority.MEDIUM
            risk = RecoveryRiskLevel.MEDIUM

        else:
            priority = RecoveryPriority.LOW
            risk = RecoveryRiskLevel.LOW

        return RecoveryPriorityAssessment(
            priority=priority,
            risk_level=risk,
            score=round(score, 4),
            reasons=tuple(reasons),
        )


# ============================================================
# RECOVERY SERVICE
# ============================================================

class RecoveryService:

    TERMINAL_STATUSES = frozenset({
        RecoveryStatus.COMPLETED,
        RecoveryStatus.CANCELLED,
    })

    def __init__(
        self,
        *,
        repository: RecoveryRepository,
        funding_gateway: RecoveryFundingGateway,
        team_gateway: RecoveryTeamGateway,
        authorization_gateway: RecoveryAuthorizationGateway,
        audit_gateway: RecoveryAuditGateway | None = None,
        cost_estimator: RecoveryCostEstimator | None = None,
        priority_evaluator: RecoveryPriorityEvaluator | None = None,
    ):
        self.repository = repository
        self.funding = funding_gateway
        self.teams = team_gateway
        self.authorization = authorization_gateway
        self.audit = audit_gateway

        self.cost_estimator = (
            cost_estimator
            or RecoveryCostEstimator()
        )

        self.priority_evaluator = (
            priority_evaluator
            or RecoveryPriorityEvaluator()
        )

    async def _audit(
        self,
        action: str,
        project: RecoveryProject,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="recovery_project",
            entity_id=project.id,
            metadata={
                "district_id": project.district_id,
                "status": project.status.value,
                "priority": project.priority.value,
                "progress_percent": (
                    project.progress_percent
                ),
            },
        )

    async def _save(
        self,
        previous: RecoveryProject,
        updated: RecoveryProject,
        *,
        action: str,
        actor_id: int,
    ) -> RecoveryProject:

        saved = await self.repository.update(
            updated,
            expected_version=previous.version,
        )

        await self._audit(
            action,
            saved,
            actor_id,
        )

        return saved

    async def get_project(
        self,
        project_id: str,
    ) -> RecoveryProject:

        project = await self.repository.get_by_id(
            project_id
        )

        if project is None:
            raise RecoveryNotFoundError(
                "Recovery project not found"
            )

        return project

    async def create_project(
        self,
        request: RecoveryCreateRequest,
        *,
        actor_id: int,
    ) -> RecoveryProject:

        district_id = request.location.district_id

        if not await self.authorization.can_create(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise RecoveryAuthorizationError(
                "Recovery project creation denied"
            )

        title = require_text(
            request.title,
            "Project title",
            max_length=200,
        )

        description = require_text(
            request.description,
            "Project description",
            max_length=5000,
        )

        require_text(
            request.incident_id,
            "Incident ID",
        )

        if request.estimated_start_date is not None:
            validate_datetime(
                request.estimated_start_date,
                "estimated_start_date",
            )

        if request.estimated_end_date is not None:
            validate_datetime(
                request.estimated_end_date,
                "estimated_end_date",
            )

        if (
            request.estimated_start_date is not None
            and request.estimated_end_date is not None
            and request.estimated_end_date
            <= request.estimated_start_date
        ):
            raise RecoveryValidationError(
                "End date must be later than start date"
            )

        project = RecoveryProject(
            id=str(uuid.uuid4()),
            district_id=district_id,
            incident_id=request.incident_id,
            location=request.location,
            title=title,
            description=description,
            category=request.category,
            work_type=request.work_type,
            status=RecoveryStatus.DRAFT,
            priority=RecoveryPriority.MEDIUM,
            created_by=actor_id,
            damage_report_id=request.damage_report_id,
            beneficiary_reference=(
                request.beneficiary_reference
            ),
            estimated_start_date=(
                request.estimated_start_date
            ),
            estimated_end_date=(
                request.estimated_end_date
            ),
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            project
        )

        await self._audit(
            "recovery.project_created",
            created,
            actor_id,
        )

        return created

    async def request_assessment(
        self,
        project_id: str,
        *,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Assessment request permission denied"
            )

        if project.status != RecoveryStatus.DRAFT:
            raise RecoveryConflictError(
                "Only draft projects can request assessment"
            )

        updated = replace(
            project,
            status=RecoveryStatus.ASSESSMENT_PENDING,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.assessment_requested",
            actor_id=actor_id,
        )

    async def record_assessment(
        self,
        project_id: str,
        *,
        assessment: RecoveryAssessment,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_assess(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Recovery assessment permission denied"
            )

        if (
            project.status
            != RecoveryStatus.ASSESSMENT_PENDING
        ):
            raise RecoveryConflictError(
                "Project is not awaiting assessment"
            )

        if assessment.assessed_by != actor_id:
            raise RecoveryValidationError(
                "Assessment actor mismatch"
            )

        if not assessment.verified:
            raise RecoveryVerificationError(
                "Verified field assessment is required"
            )

        if not assessment.evidence_references:
            raise RecoveryVerificationError(
                "Assessment evidence is required"
            )

        priority_assessment = (
            self.priority_evaluator.evaluate(
                assessment
            )
        )

        updated = replace(
            project,
            assessment=assessment,
            priority_assessment=priority_assessment,
            priority=priority_assessment.priority,
            status=RecoveryStatus.ASSESSED,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.assessment_recorded",
            actor_id=actor_id,
        )

    async def estimate_cost(
        self,
        project_id: str,
        *,
        items: Sequence[RecoveryCostItem],
        actor_id: int,
        contingency_rate: Decimal | None = None,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Recovery cost estimation denied"
            )

        if project.status != RecoveryStatus.ASSESSED:
            raise RecoveryConflictError(
                "Verified assessment is required"
            )

        estimate = self.cost_estimator.estimate(
            items,
            contingency_rate=contingency_rate,
        )

        updated = replace(
            project,
            cost_estimate=estimate,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.cost_estimated",
            actor_id=actor_id,
        )

    async def add_milestone(
        self,
        project_id: str,
        *,
        title: str,
        description: str,
        target_date: datetime,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Milestone creation denied"
            )

        if project.status not in {
            RecoveryStatus.ASSESSED,
            RecoveryStatus.APPROVED,
        }:
            raise RecoveryConflictError(
                "Milestones can only be planned "
                "before project execution"
            )

        validate_datetime(
            target_date,
            "target_date",
        )

        milestone = RecoveryMilestone(
            id=str(uuid.uuid4()),
            title=require_text(
                title,
                "Milestone title",
                max_length=200,
            ),
            description=description.strip(),
            target_date=target_date,
        )

        updated = replace(
            project,
            milestones=[
                *project.milestones,
                milestone,
            ],
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.milestone_added",
            actor_id=actor_id,
        )

    async def submit_for_approval(
        self,
        project_id: str,
        *,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Project submission denied"
            )

        if project.status != RecoveryStatus.ASSESSED:
            raise RecoveryConflictError(
                "Project must be assessed first"
            )

        if project.cost_estimate is None:
            raise RecoveryValidationError(
                "Cost estimate is required"
            )

        if not project.milestones:
            raise RecoveryValidationError(
                "At least one milestone is required"
            )

        updated = replace(
            project,
            status=RecoveryStatus.PENDING_APPROVAL,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.submitted_for_approval",
            actor_id=actor_id,
        )

    async def approve_project(
        self,
        project_id: str,
        *,
        reviewer_id: int,
        approval_reference: str,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Recovery approval denied"
            )

        if (
            project.status
            != RecoveryStatus.PENDING_APPROVAL
        ):
            raise RecoveryConflictError(
                "Project is not awaiting approval"
            )

        if reviewer_id == project.created_by:
            raise RecoveryAuthorizationError(
                "Independent reviewer required"
            )

        require_text(
            approval_reference,
            "Approval reference",
        )

        updated = replace(
            project,
            status=RecoveryStatus.APPROVED,
            approved_by=reviewer_id,
            approval_reference=approval_reference.strip(),
            approved_at=utc_now(),
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.project_approved",
            actor_id=reviewer_id,
        )

    async def reserve_funding(
        self,
        project_id: str,
        *,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Funding reservation denied"
            )

        if project.status != RecoveryStatus.APPROVED:
            raise RecoveryConflictError(
                "Project must be approved before funding"
            )

        if project.cost_estimate is None:
            raise RecoveryValidationError(
                "Cost estimate is missing"
            )

        amount = (
            project.cost_estimate.total_estimated_cost_inr
        )

        available = (
            await self.funding.check_available_budget(
                district_id=project.district_id,
                amount_inr=amount,
            )
        )

        if not available:
            raise RecoveryFundingError(
                "Insufficient recovery budget"
            )

        reserved = await self.funding.reserve_budget(
            project_id=project.id,
            district_id=project.district_id,
            amount_inr=amount,
        )

        if not reserved:
            raise RecoveryFundingError(
                "Budget reservation failed"
            )

        updated = replace(
            project,
            status=RecoveryStatus.FUNDED,
            funding_status=FundingStatus.RESERVED,
            reserved_funding_inr=amount,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        try:
            saved = await self._save(
                project,
                updated,
                action="recovery.funding_reserved",
                actor_id=actor_id,
            )
        except Exception:
            # Production implementations should use
            # durable compensation and reconciliation.
            await self.funding.release_budget(
                project_id=project.id
            )
            raise

        return saved

    async def assign_team(
        self,
        project_id: str,
        *,
        team_id: str,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Team assignment denied"
            )

        if project.status != RecoveryStatus.FUNDED:
            raise RecoveryConflictError(
                "Project must be funded before team assignment"
            )

        if project.assigned_team_id is not None:
            raise RecoveryConflictError(
                "Project already has an assigned team"
            )

        require_text(
            team_id,
            "Team ID",
        )

        verified = await self.teams.verify_team(
            team_id=team_id,
            district_id=project.district_id,
            work_type=project.work_type,
        )

        if not verified:
            raise RecoveryValidationError(
                "Team is not verified for this work"
            )

        reserved = await self.teams.reserve_team(
            team_id=team_id,
            project_id=project.id,
        )

        if not reserved:
            raise RecoveryConflictError(
                "Team reservation failed"
            )

        updated = replace(
            project,
            assigned_team_id=team_id,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        try:
            return await self._save(
                project,
                updated,
                action="recovery.team_assigned",
                actor_id=actor_id,
            )
        except Exception:
            await self.teams.release_team(
                team_id=team_id,
                project_id=project.id,
            )
            raise

    async def start_project(
        self,
        project_id: str,
        *,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Project start denied"
            )

        if project.status != RecoveryStatus.FUNDED:
            raise RecoveryConflictError(
                "Project is not ready to start"
            )

        if project.assigned_team_id is None:
            raise RecoveryValidationError(
                "Verified recovery team is required"
            )

        if project.approved_by is None:
            raise RecoveryAuthorizationError(
                "Human approval is required"
            )

        if (
            project.assessment is None
            or not project.assessment.verified
        ):
            raise RecoveryVerificationError(
                "Verified assessment is required"
            )

        # A professional structural inspection must
        # precede work where structural hazards exist.
        if (
            project.work_type
            in {
                RecoveryWorkType.REPAIR,
                RecoveryWorkType.RECONSTRUCTION,
            }
            and project.category
            in {
                RecoveryCategory.RESIDENTIAL_BUILDING,
                RecoveryCategory.COMMERCIAL_BUILDING,
                RecoveryCategory.PUBLIC_BUILDING,
                RecoveryCategory.BRIDGE,
            }
            and not project.assessment.structural_safety_verified
        ):
            raise RecoveryVerificationError(
                "Professional structural safety verification "
                "is required before construction work"
            )

        updated = replace(
            project,
            status=RecoveryStatus.IN_PROGRESS,
            started_at=utc_now(),
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.project_started",
            actor_id=actor_id,
        )

    async def update_milestone_progress(
        self,
        project_id: str,
        *,
        update: RecoveryProgressUpdate,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Progress update denied"
            )

        if project.status != RecoveryStatus.IN_PROGRESS:
            raise RecoveryConflictError(
                "Project is not in progress"
            )

        validate_progress(
            update.progress_percent
        )

        validate_datetime(
            update.reported_at,
            "reported_at",
        )

        if update.reported_at > utc_now():
            raise RecoveryValidationError(
                "Progress report cannot be future-dated"
            )

        milestones = [
            replace(milestone)
            for milestone in project.milestones
        ]

        target = next(
            (
                milestone
                for milestone in milestones
                if milestone.id == update.milestone_id
            ),
            None,
        )

        if target is None:
            raise RecoveryNotFoundError(
                "Recovery milestone not found"
            )

        if target.status == MilestoneStatus.VERIFIED:
            raise RecoveryConflictError(
                "Verified milestones cannot be modified"
            )

        if (
            update.progress_percent
            < target.progress_percent
        ):
            raise RecoveryConflictError(
                "Milestone progress cannot decrease"
            )

        target.progress_percent = (
            update.progress_percent
        )

        target.evidence_references = list(
            dict.fromkeys([
                *target.evidence_references,
                *update.evidence_references,
            ])
        )

        if update.progress_percent == 100:
            target.status = MilestoneStatus.COMPLETED
            target.completed_at = update.reported_at

        elif update.progress_percent > 0:
            target.status = MilestoneStatus.IN_PROGRESS

        overall_progress = (
            sum(
                milestone.progress_percent
                for milestone in milestones
            )
            / len(milestones)
            if milestones
            else 0.0
        )

        updated = replace(
            project,
            milestones=milestones,
            progress_percent=round(
                overall_progress,
                2,
            ),
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.milestone_progress_updated",
            actor_id=actor_id,
        )

    async def verify_milestone(
        self,
        project_id: str,
        *,
        milestone_id: str,
        reviewer_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_verify(
            user_id=reviewer_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Milestone verification denied"
            )

        if project.status != RecoveryStatus.IN_PROGRESS:
            raise RecoveryConflictError(
                "Project is not in progress"
            )

        milestones = [
            replace(milestone)
            for milestone in project.milestones
        ]

        target = next(
            (
                milestone
                for milestone in milestones
                if milestone.id == milestone_id
            ),
            None,
        )

        if target is None:
            raise RecoveryNotFoundError(
                "Milestone not found"
            )

        if target.status != MilestoneStatus.COMPLETED:
            raise RecoveryConflictError(
                "Milestone must be completed first"
            )

        if not target.evidence_references:
            raise RecoveryVerificationError(
                "Completion evidence is required"
            )

        target.status = MilestoneStatus.VERIFIED
        target.verified_by = reviewer_id
        target.verified_at = utc_now()

        updated = replace(
            project,
            milestones=milestones,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.milestone_verified",
            actor_id=reviewer_id,
        )

    async def request_completion(
        self,
        project_id: str,
        *,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Completion request denied"
            )

        if project.status != RecoveryStatus.IN_PROGRESS:
            raise RecoveryConflictError(
                "Project is not in progress"
            )

        if not project.milestones:
            raise RecoveryValidationError(
                "Project has no milestones"
            )

        if not all(
            milestone.status == MilestoneStatus.VERIFIED
            for milestone in project.milestones
        ):
            raise RecoveryVerificationError(
                "All milestones must be verified"
            )

        updated = replace(
            project,
            status=RecoveryStatus.PENDING_VERIFICATION,
            progress_percent=100.0,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.completion_requested",
            actor_id=actor_id,
        )

    async def complete_project(
        self,
        project_id: str,
        *,
        reviewer_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_verify(
            user_id=reviewer_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Project completion verification denied"
            )

        if (
            project.status
            != RecoveryStatus.PENDING_VERIFICATION
        ):
            raise RecoveryConflictError(
                "Project is not awaiting final verification"
            )

        if reviewer_id == project.created_by:
            raise RecoveryAuthorizationError(
                "Independent completion reviewer required"
            )

        if not all(
            milestone.status == MilestoneStatus.VERIFIED
            for milestone in project.milestones
        ):
            raise RecoveryVerificationError(
                "Unverified milestones remain"
            )

        updated = replace(
            project,
            status=RecoveryStatus.COMPLETED,
            completed_at=utc_now(),
            updated_at=utc_now(),
            version=project.version + 1,
        )

        saved = await self._save(
            project,
            updated,
            action="recovery.project_completed",
            actor_id=reviewer_id,
        )

        if project.assigned_team_id:
            try:
                await self.teams.release_team(
                    team_id=project.assigned_team_id,
                    project_id=project.id,
                )
            except Exception:
                logger.exception(
                    "Recovery team release failed: %s",
                    project.id,
                )

        # Funding is not automatically released.
        # Finance reconciliation and settlement
        # must be handled by a separate workflow.

        return saved

    async def put_on_hold(
        self,
        project_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Project hold permission denied"
            )

        if project.status != RecoveryStatus.IN_PROGRESS:
            raise RecoveryConflictError(
                "Only active projects can be put on hold"
            )

        reason = require_text(
            reason,
            "Hold reason",
        )

        updated = replace(
            project,
            status=RecoveryStatus.ON_HOLD,
            updated_at=utc_now(),
            version=project.version + 1,
            metadata={
                **project.metadata,
                "hold_reason": reason,
                "hold_at": utc_now().isoformat(),
            },
        )

        return await self._save(
            project,
            updated,
            action="recovery.project_on_hold",
            actor_id=actor_id,
        )

    async def resume_project(
        self,
        project_id: str,
        *,
        actor_id: int,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Project resume permission denied"
            )

        if project.status != RecoveryStatus.ON_HOLD:
            raise RecoveryConflictError(
                "Project is not on hold"
            )

        if project.assigned_team_id is None:
            raise RecoveryValidationError(
                "Recovery team must be assigned"
            )

        verified = await self.teams.verify_team(
            team_id=project.assigned_team_id,
            district_id=project.district_id,
            work_type=project.work_type,
        )

        if not verified:
            raise RecoveryVerificationError(
                "Assigned team is no longer verified"
            )

        updated = replace(
            project,
            status=RecoveryStatus.IN_PROGRESS,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        return await self._save(
            project,
            updated,
            action="recovery.project_resumed",
            actor_id=actor_id,
        )

    async def cancel_project(
        self,
        project_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> RecoveryProject:

        project = await self.get_project(
            project_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=project.district_id,
        ):
            raise RecoveryAuthorizationError(
                "Project cancellation denied"
            )

        if project.status in self.TERMINAL_STATUSES:
            raise RecoveryConflictError(
                "Project is already terminal"
            )

        if project.status in {
            RecoveryStatus.IN_PROGRESS,
            RecoveryStatus.ON_HOLD,
            RecoveryStatus.PENDING_VERIFICATION,
        }:
            raise RecoveryConflictError(
                "Started projects require a supervised "
                "closure and financial reconciliation workflow"
            )

        reason = require_text(
            reason,
            "Cancellation reason",
        )

        updated = replace(
            project,
            status=RecoveryStatus.CANCELLED,
            cancelled_at=utc_now(),
            cancellation_reason=reason,
            updated_at=utc_now(),
            version=project.version + 1,
        )

        saved = await self._save(
            project,
            updated,
            action="recovery.project_cancelled",
            actor_id=actor_id,
        )

        if project.funding_status == FundingStatus.RESERVED:
            try:
                await self.funding.release_budget(
                    project_id=project.id
                )
            except Exception:
                logger.exception(
                    "Recovery budget release failed: %s",
                    project.id,
                )

        if project.assigned_team_id:
            try:
                await self.teams.release_team(
                    team_id=project.assigned_team_id,
                    project_id=project.id,
                )
            except Exception:
                logger.exception(
                    "Recovery team release failed: %s",
                    project.id,
                )

        return saved

    async def list_district_projects(
        self,
        district_id: int,
    ) -> Sequence[RecoveryProject]:

        if district_id <= 0:
            raise RecoveryValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_incident_projects(
        self,
        incident_id: str,
    ) -> Sequence[RecoveryProject]:

        require_text(
            incident_id,
            "Incident ID",
        )

        return await self.repository.list_by_incident(
            incident_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        projects = (
            await self.list_district_projects(
                district_id
            )
        )

        status_counts = {
            status.value: 0
            for status in RecoveryStatus
        }

        priority_counts = {
            priority.value: 0
            for priority in RecoveryPriority
        }

        category_counts = {
            category.value: 0
            for category in RecoveryCategory
        }

        for project in projects:
            status_counts[
                project.status.value
            ] += 1

            priority_counts[
                project.priority.value
            ] += 1

            category_counts[
                project.category.value
            ] += 1

        estimated_budget = sum(
            (
                project.cost_estimate.total_estimated_cost_inr
                for project in projects
                if project.cost_estimate is not None
            ),
            Decimal("0"),
        )

        reserved_budget = sum(
            (
                project.reserved_funding_inr
                for project in projects
            ),
            Decimal("0"),
        )

        average_progress = (
            sum(
                project.progress_percent
                for project in projects
            )
            / len(projects)
            if projects
            else 0.0
        )

        return {
            "district_id": district_id,
            "total_projects": len(projects),
            "active_projects": sum(
                project.status not in self.TERMINAL_STATUSES
                for project in projects
            ),
            "completed_projects": status_counts[
                RecoveryStatus.COMPLETED.value
            ],
            "critical_projects": priority_counts[
                RecoveryPriority.CRITICAL.value
            ],
            "average_progress_percent": round(
                average_progress,
                2,
            ),
            "estimated_budget_inr": str(
                money(estimated_budget)
            ),
            "reserved_budget_inr": str(
                money(reserved_budget)
            ),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "category_counts": category_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_project_report(
        self,
        project_id: str,
    ) -> dict[str, Any]:

        project = await self.get_project(
            project_id
        )

        return {
            "project_id": project.id,
            "district_id": project.district_id,
            "incident_id": project.incident_id,
            "title": project.title,
            "description": project.description,
            "category": project.category.value,
            "work_type": project.work_type.value,
            "status": project.status.value,
            "priority": project.priority.value,
            "progress_percent": project.progress_percent,
            "damage_report_id": project.damage_report_id,
            "assessment": (
                {
                    "assessment_id": (
                        project.assessment.assessment_id
                    ),
                    "damage_level": (
                        project.assessment.damage_level.value
                    ),
                    "verified": project.assessment.verified,
                    "structural_safety_verified": (
                        project.assessment
                        .structural_safety_verified
                    ),
                    "affected_people": (
                        project.assessment.affected_people
                    ),
                    "displaced_people": (
                        project.assessment.displaced_people
                    ),
                }
                if project.assessment
                else None
            ),
            "priority_assessment": (
                {
                    "score": (
                        project.priority_assessment.score
                    ),
                    "risk_level": (
                        project.priority_assessment
                        .risk_level.value
                    ),
                    "reasons": list(
                        project.priority_assessment.reasons
                    ),
                }
                if project.priority_assessment
                else None
            ),
            "cost_estimate": (
                {
                    "direct_cost_inr": str(
                        project.cost_estimate.direct_cost_inr
                    ),
                    "contingency_inr": str(
                        project.cost_estimate.contingency_inr
                    ),
                    "total_estimated_cost_inr": str(
                        project.cost_estimate
                        .total_estimated_cost_inr
                    ),
                }
                if project.cost_estimate
                else None
            ),
            "funding": {
                "status": project.funding_status.value,
                "reserved_amount_inr": str(
                    project.reserved_funding_inr
                ),
            },
            "milestones": [
                {
                    "id": milestone.id,
                    "title": milestone.title,
                    "status": milestone.status.value,
                    "progress_percent": (
                        milestone.progress_percent
                    ),
                    "target_date": (
                        milestone.target_date.isoformat()
                    ),
                    "verified_by": milestone.verified_by,
                    "verified_at": (
                        milestone.verified_at.isoformat()
                        if milestone.verified_at
                        else None
                    ),
                }
                for milestone in project.milestones
            ],
            "approval": {
                "approved_by": project.approved_by,
                "approval_reference": (
                    project.approval_reference
                ),
                "approved_at": (
                    project.approved_at.isoformat()
                    if project.approved_at
                    else None
                ),
            },
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
            "completed_at": (
                project.completed_at.isoformat()
                if project.completed_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Recovery estimates and priorities are "
                "planning aids only. Structural safety, "
                "project approval, funding eligibility, "
                "construction completion and compensation "
                "require verification by authorized "
                "professionals and government processes."
            ),
        }
