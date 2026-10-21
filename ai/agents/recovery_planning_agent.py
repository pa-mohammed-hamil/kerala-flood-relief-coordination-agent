"""Placeholder module: ai/agents/recovery_planning_agent.py; implementation required."""

"""
backend/ai/agents/recovery_planning_agent.py

Kerala Flood Relief Coordination Agent
Post-Flood Recovery Planning Agent

Python 3.11+
Standard library only.

IMPORTANT:
- All cost estimates are preliminary planning estimates.
- Damage reports require qualified human verification.
- Priority scores are planning heuristics, not validated
  safety or engineering assessments.
- No reconstruction, procurement, fund transfer,
  demolition or infrastructure operation is authorized.
- Human approval is mandatory.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class RecoveryPlanningError(Exception):
    pass


class RecoveryValidationError(RecoveryPlanningError):
    pass


class RecoveryApprovalError(RecoveryPlanningError):
    pass


class RecoveryResourceError(RecoveryPlanningError):
    pass


class RecoveryExportError(RecoveryPlanningError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RecoverySector(str, Enum):
    HOUSING = "housing"
    ROADS = "roads"
    BRIDGES = "bridges"
    ELECTRICITY = "electricity"
    WATER_SUPPLY = "water_supply"
    SANITATION = "sanitation"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    AGRICULTURE = "agriculture"
    TELECOMMUNICATION = "telecommunication"
    PUBLIC_INFRASTRUCTURE = "public_infrastructure"
    LIVELIHOODS = "livelihoods"


class DamageSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RecoveryPriority(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryPhase(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class RecoveryTaskStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED_FOR_PLANNING = "approved_for_planning"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    COMPLETED_REPORTED = "completed_reported"


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"


class RecoveryResourceType(str, Enum):
    ENGINEERING_TEAM = "engineering_team"
    CONSTRUCTION_CREW = "construction_crew"
    ELECTRICAL_CREW = "electrical_crew"
    WATER_SERVICE_CREW = "water_service_crew"
    EQUIPMENT = "equipment"
    MATERIALS = "materials"
    TRANSPORT = "transport"
    MEDICAL_SUPPORT = "medical_support"
    OTHER = "other"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def validate_datetime(value: datetime) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise RecoveryValidationError(
            "Timezone-aware datetime required"
        )


def validate_nonnegative(
    value: int | float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RecoveryValidationError(
            f"{name} must be nonnegative and finite"
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    validate_nonnegative(value, name)

    if value > 1:
        raise RecoveryValidationError(
            f"{name} must not exceed 1"
        )


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(k): serialize(v)
            for k, v in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [serialize(v) for v in value]

    return value


def stable_hash(data: Mapping[str, Any]) -> str:
    payload = json.dumps(
        serialize(data),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class RecoveryPlanningConfig:
    max_assets: int = 5000
    max_tasks: int = 10000

    min_evidence_reliability: float = 0.70

    critical_priority_threshold: float = 0.80
    high_priority_threshold: float = 0.60
    moderate_priority_threshold: float = 0.35

    immediate_phase_threshold: float = 0.80
    short_term_phase_threshold: float = 0.60
    medium_term_phase_threshold: float = 0.35

    default_contingency_rate: float = 0.15

    require_verified_evidence: bool = True
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        for name in ("max_assets", "max_tasks"):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RecoveryValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "min_evidence_reliability",
            "critical_priority_threshold",
            "high_priority_threshold",
            "moderate_priority_threshold",
            "immediate_phase_threshold",
            "short_term_phase_threshold",
            "medium_term_phase_threshold",
            "default_contingency_rate",
        ):
            validate_probability(
                getattr(self, name),
                name,
            )

        if not (
            self.moderate_priority_threshold
            <= self.high_priority_threshold
            <= self.critical_priority_threshold
        ):
            raise RecoveryValidationError(
                "Priority thresholds must be ascending"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class RecoveryEvidence:
    evidence_id: str
    source_id: str

    status: EvidenceStatus
    reliability: float

    observed_at: datetime
    description: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.source_id:
            raise RecoveryValidationError(
                "Evidence identifiers required"
            )

        validate_datetime(self.observed_at)

        validate_probability(
            self.reliability,
            "reliability",
        )


@dataclass(frozen=True)
class DamagedAsset:
    asset_id: str

    name: str
    district: str
    location: str

    sector: RecoverySector
    severity: DamageSeverity

    affected_population: int

    essential_service: bool
    safety_concern: bool

    estimated_repair_cost_inr: float

    evidence: tuple[RecoveryEvidence, ...]

    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "asset_id",
            "name",
            "district",
            "location",
        ):
            if not getattr(self, name).strip():
                raise RecoveryValidationError(
                    f"{name} required"
                )

        validate_nonnegative(
            self.affected_population,
            "affected_population",
        )

        if not isinstance(
            self.affected_population,
            int,
        ):
            raise RecoveryValidationError(
                "affected_population must be integer"
            )

        validate_nonnegative(
            self.estimated_repair_cost_inr,
            "estimated_repair_cost_inr",
        )

        if self.asset_id in self.dependencies:
            raise RecoveryValidationError(
                "Asset cannot depend on itself"
            )


@dataclass(frozen=True)
class RecoveryResource:
    resource_id: str

    resource_type: RecoveryResourceType

    district: str

    available_units: int

    verified: bool

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise RecoveryValidationError(
                "resource_id required"
            )

        validate_nonnegative(
            self.available_units,
            "available_units",
        )

        if not isinstance(
            self.available_units,
            int,
        ):
            raise RecoveryValidationError(
                "available_units must be integer"
            )


@dataclass(frozen=True)
class RecoveryPlanningRequest:
    request_id: str

    district: str

    assets: tuple[DamagedAsset, ...]
    resources: tuple[RecoveryResource, ...]

    available_budget_inr: float

    created_at: datetime

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise RecoveryValidationError(
                "request_id required"
            )

        if not self.district.strip():
            raise RecoveryValidationError(
                "district required"
            )

        validate_nonnegative(
            self.available_budget_inr,
            "available_budget_inr",
        )

        validate_datetime(self.created_at)


@dataclass(frozen=True)
class RecoveryPriorityAssessment:
    asset_id: str

    score: float
    priority: RecoveryPriority
    phase: RecoveryPhase

    factors: Mapping[str, float]

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryCostEstimate:
    asset_id: str

    base_cost_inr: float
    contingency_inr: float

    total_estimated_cost_inr: float

    requires_engineer_validation: bool


@dataclass(frozen=True)
class RecoveryResourceRequirement:
    resource_type: RecoveryResourceType
    required_units: int


@dataclass(frozen=True)
class RecoveryTask:
    task_id: str
    asset_id: str

    title: str

    priority: RecoveryPriority
    phase: RecoveryPhase

    estimated_cost_inr: float

    required_resources: tuple[
        RecoveryResourceRequirement,
        ...
    ]

    dependencies: tuple[str, ...]

    status: RecoveryTaskStatus

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryResourceGap:
    resource_type: RecoveryResourceType

    required_units: int
    available_units: int

    shortage_units: int


@dataclass(frozen=True)
class RecoveryBudgetAssessment:
    total_estimated_cost_inr: float
    available_budget_inr: float

    funding_gap_inr: float
    estimated_surplus_inr: float

    funding_sufficient: bool


@dataclass(frozen=True)
class RecoveryPlan:
    plan_id: str
    request_id: str

    district: str

    assessments: tuple[
        RecoveryPriorityAssessment,
        ...
    ]

    cost_estimates: tuple[
        RecoveryCostEstimate,
        ...
    ]

    tasks: tuple[RecoveryTask, ...]

    resource_gaps: tuple[
        RecoveryResourceGap,
        ...
    ]

    budget: RecoveryBudgetAssessment

    proposed_task_order: tuple[str, ...]

    warnings: tuple[str, ...]

    human_approval_required: bool

    approved_for_planning: bool

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RecoveryApprovalRecord:
    approval_id: str
    plan_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ApprovalDecision

    reason: str
    reviewed_at: datetime


@dataclass(frozen=True)
class RecoveryAuditRecord:
    audit_id: str
    plan_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# EVIDENCE VALIDATOR
# ============================================================

class RecoveryEvidenceValidator:
    def __init__(
        self,
        config: RecoveryPlanningConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        asset: DamagedAsset,
        *,
        now: datetime,
    ) -> tuple[bool, tuple[str, ...]]:
        warnings = []
        verified = False
        seen_ids = set()

        for item in asset.evidence:
            if item.evidence_id in seen_ids:
                warnings.append(
                    "Duplicate evidence identifier"
                )
                continue

            seen_ids.add(item.evidence_id)

            if item.observed_at > now:
                warnings.append(
                    f"{item.evidence_id}: future timestamp"
                )
                continue

            if item.status != EvidenceStatus.VERIFIED:
                warnings.append(
                    f"{item.evidence_id}: unverified evidence"
                )
                continue

            if item.reliability < (
                self.config.min_evidence_reliability
            ):
                warnings.append(
                    f"{item.evidence_id}: insufficient reliability"
                )
                continue

            verified = True

        if not verified:
            warnings.append(
                "No sufficiently reliable verified evidence"
            )

        return verified, tuple(warnings)


# ============================================================
# PRIORITY SCORING ENGINE
# ============================================================

class RecoveryPriorityEngine:
    SEVERITY_SCORE = {
        DamageSeverity.MINOR: 0.20,
        DamageSeverity.MODERATE: 0.45,
        DamageSeverity.SEVERE: 0.75,
        DamageSeverity.CRITICAL: 1.00,
        DamageSeverity.UNKNOWN: 0.50,
    }

    def __init__(
        self,
        config: RecoveryPlanningConfig,
    ) -> None:
        self.config = config

    def assess(
        self,
        asset: DamagedAsset,
        *,
        evidence_verified: bool,
    ) -> RecoveryPriorityAssessment:
        severity = self.SEVERITY_SCORE[
            asset.severity
        ]

        # Heuristic population normalization.
        population_factor = min(
            asset.affected_population / 1000,
            1.0,
        )

        service_factor = float(
            asset.essential_service
        )

        safety_factor = float(
            asset.safety_concern
        )

        factors = {
            "severity": severity,
            "population": population_factor,
            "essential_service": service_factor,
            "safety_concern": safety_factor,
        }

        score = (
            0.30 * severity
            + 0.20 * population_factor
            + 0.25 * service_factor
            + 0.25 * safety_factor
        )

        score = round(
            min(1.0, max(0.0, score)),
            4,
        )

        if score >= (
            self.config.critical_priority_threshold
        ):
            priority = RecoveryPriority.CRITICAL

        elif score >= (
            self.config.high_priority_threshold
        ):
            priority = RecoveryPriority.HIGH

        elif score >= (
            self.config.moderate_priority_threshold
        ):
            priority = RecoveryPriority.MODERATE

        else:
            priority = RecoveryPriority.LOW

        if score >= (
            self.config.immediate_phase_threshold
        ):
            phase = RecoveryPhase.IMMEDIATE

        elif score >= (
            self.config.short_term_phase_threshold
        ):
            phase = RecoveryPhase.SHORT_TERM

        elif score >= (
            self.config.medium_term_phase_threshold
        ):
            phase = RecoveryPhase.MEDIUM_TERM

        else:
            phase = RecoveryPhase.LONG_TERM

        warnings = [
            "Priority score is a planning heuristic",
            "Engineering and disaster-management "
            "review is required",
        ]

        if not evidence_verified:
            warnings.append(
                "Priority assessment uses unverified damage"
            )

        if asset.severity == DamageSeverity.UNKNOWN:
            warnings.append(
                "Damage severity is unknown"
            )

        return RecoveryPriorityAssessment(
            asset_id=asset.asset_id,
            score=score,
            priority=priority,
            phase=phase,
            factors=factors,
            warnings=tuple(warnings),
        )


# ============================================================
# COST ESTIMATION ENGINE
# ============================================================

class RecoveryCostEstimationEngine:
    def __init__(
        self,
        config: RecoveryPlanningConfig,
    ) -> None:
        self.config = config

    def estimate(
        self,
        asset: DamagedAsset,
    ) -> RecoveryCostEstimate:
        base_cost = (
            asset.estimated_repair_cost_inr
        )

        contingency = (
            base_cost
            * self.config.default_contingency_rate
        )

        total = base_cost + contingency

        return RecoveryCostEstimate(
            asset_id=asset.asset_id,
            base_cost_inr=round(base_cost, 2),
            contingency_inr=round(
                contingency,
                2,
            ),
            total_estimated_cost_inr=round(
                total,
                2,
            ),
            requires_engineer_validation=True,
        )


# ============================================================
# RESOURCE REQUIREMENT ENGINE
# ============================================================

class RecoveryResourceRequirementEngine:
    """
    Preliminary resource planning assumptions.

    These quantities are illustrative defaults,
    not engineering staffing specifications.
    """

    SECTOR_REQUIREMENTS = {
        RecoverySector.HOUSING: (
            RecoveryResourceType.CONSTRUCTION_CREW,
            RecoveryResourceType.MATERIALS,
        ),
        RecoverySector.ROADS: (
            RecoveryResourceType.ENGINEERING_TEAM,
            RecoveryResourceType.EQUIPMENT,
        ),
        RecoverySector.BRIDGES: (
            RecoveryResourceType.ENGINEERING_TEAM,
            RecoveryResourceType.EQUIPMENT,
        ),
        RecoverySector.ELECTRICITY: (
            RecoveryResourceType.ELECTRICAL_CREW,
        ),
        RecoverySector.WATER_SUPPLY: (
            RecoveryResourceType.WATER_SERVICE_CREW,
        ),
        RecoverySector.SANITATION: (
            RecoveryResourceType.WATER_SERVICE_CREW,
        ),
        RecoverySector.HEALTHCARE: (
            RecoveryResourceType.ENGINEERING_TEAM,
        ),
        RecoverySector.EDUCATION: (
            RecoveryResourceType.CONSTRUCTION_CREW,
        ),
        RecoverySector.AGRICULTURE: (
            RecoveryResourceType.EQUIPMENT,
        ),
        RecoverySector.TELECOMMUNICATION: (
            RecoveryResourceType.ENGINEERING_TEAM,
        ),
        RecoverySector.PUBLIC_INFRASTRUCTURE: (
            RecoveryResourceType.ENGINEERING_TEAM,
            RecoveryResourceType.CONSTRUCTION_CREW,
        ),
        RecoverySector.LIVELIHOODS: (
            RecoveryResourceType.OTHER,
        ),
    }

    @classmethod
    def estimate(
        cls,
        asset: DamagedAsset,
    ) -> tuple[RecoveryResourceRequirement, ...]:
        types = cls.SECTOR_REQUIREMENTS.get(
            asset.sector,
            (RecoveryResourceType.OTHER,),
        )

        return tuple(
            RecoveryResourceRequirement(
                resource_type=resource_type,
                required_units=1,
            )
            for resource_type in types
        )


# ============================================================
# TASK GENERATOR
# ============================================================

class RecoveryTaskGenerator:
    def __init__(
        self,
        resource_engine: RecoveryResourceRequirementEngine,
    ) -> None:
        self.resource_engine = resource_engine

    def generate(
        self,
        asset: DamagedAsset,
        assessment: RecoveryPriorityAssessment,
        cost: RecoveryCostEstimate,
        *,
        evidence_verified: bool,
    ) -> RecoveryTask:
        warnings = [
            "Task is a proposal, not a work order",
            "Site safety and technical feasibility "
            "must be verified",
        ]

        if not evidence_verified:
            warnings.append(
                "Verified damage evidence is missing"
            )

        return RecoveryTask(
            task_id=f"RECOVERY-{asset.asset_id}",
            asset_id=asset.asset_id,
            title=f"Recovery planning: {asset.name}",
            priority=assessment.priority,
            phase=assessment.phase,
            estimated_cost_inr=(
                cost.total_estimated_cost_inr
            ),
            required_resources=(
                self.resource_engine.estimate(
                    asset
                )
            ),
            dependencies=asset.dependencies,
            status=RecoveryTaskStatus.PENDING_APPROVAL,
            warnings=tuple(warnings),
        )


# ============================================================
# RESOURCE GAP ENGINE
# ============================================================

class RecoveryResourceAllocationEngine:
    @staticmethod
    def calculate_gaps(
        tasks: Sequence[RecoveryTask],
        resources: Sequence[RecoveryResource],
        *,
        district: str,
    ) -> tuple[RecoveryResourceGap, ...]:
        required = {
            resource_type: 0
            for resource_type in RecoveryResourceType
        }

        available = {
            resource_type: 0
            for resource_type in RecoveryResourceType
        }

        for task in tasks:
            for item in task.required_resources:
                required[item.resource_type] += (
                    item.required_units
                )

        seen_resource_ids = set()

        for resource in resources:
            if resource.resource_id in seen_resource_ids:
                raise RecoveryResourceError(
                    "Duplicate resource ID"
                )

            seen_resource_ids.add(
                resource.resource_id
            )

            if (
                resource.verified
                and resource.district.casefold()
                == district.casefold()
            ):
                available[resource.resource_type] += (
                    resource.available_units
                )

        return tuple(
            RecoveryResourceGap(
                resource_type=resource_type,
                required_units=required[resource_type],
                available_units=available[resource_type],
                shortage_units=max(
                    0,
                    required[resource_type]
                    - available[resource_type],
                ),
            )
            for resource_type in RecoveryResourceType
            if required[resource_type] > 0
        )


# ============================================================
# BUDGET ENGINE
# ============================================================

class RecoveryBudgetEngine:
    @staticmethod
    def assess(
        tasks: Sequence[RecoveryTask],
        available_budget_inr: float,
    ) -> RecoveryBudgetAssessment:
        total = sum(
            task.estimated_cost_inr
            for task in tasks
        )

        gap = max(
            0.0,
            total - available_budget_inr,
        )

        surplus = max(
            0.0,
            available_budget_inr - total,
        )

        return RecoveryBudgetAssessment(
            total_estimated_cost_inr=round(total, 2),
            available_budget_inr=round(
                available_budget_inr,
                2,
            ),
            funding_gap_inr=round(gap, 2),
            estimated_surplus_inr=round(
                surplus,
                2,
            ),
            funding_sufficient=(
                available_budget_inr >= total
            ),
        )


# ============================================================
# DEPENDENCY PLANNER
# ============================================================

class RecoveryDependencyPlanner:
    """
    Returns dependency-respecting task order.

    Task dependencies refer to asset IDs.
    """

    PRIORITY_ORDER = {
        RecoveryPriority.CRITICAL: 0,
        RecoveryPriority.HIGH: 1,
        RecoveryPriority.MODERATE: 2,
        RecoveryPriority.LOW: 3,
    }

    @classmethod
    def order(
        cls,
        tasks: Sequence[RecoveryTask],
    ) -> tuple[str, ...]:
        asset_to_task = {
            task.asset_id: task
            for task in tasks
        }

        if len(asset_to_task) != len(tasks):
            raise RecoveryValidationError(
                "Duplicate asset tasks"
            )

        dependencies = {}

        for task in tasks:
            missing = [
                dependency
                for dependency in task.dependencies
                if dependency not in asset_to_task
            ]

            if missing:
                raise RecoveryValidationError(
                    f"Unknown dependencies for "
                    f"{task.asset_id}: {missing}"
                )

            dependencies[task.asset_id] = set(
                task.dependencies
            )

        completed = set()
        ordered = []

        while len(completed) < len(tasks):
            ready = [
                task
                for task in tasks
                if task.asset_id not in completed
                and dependencies[
                    task.asset_id
                ].issubset(completed)
            ]

            if not ready:
                raise RecoveryValidationError(
                    "Circular recovery dependencies detected"
                )

            ready.sort(
                key=lambda task: (
                    cls.PRIORITY_ORDER[
                        task.priority
                    ],
                    task.asset_id,
                )
            )

            selected = ready[0]

            ordered.append(
                selected.task_id
            )

            completed.add(
                selected.asset_id
            )

        return tuple(ordered)


# ============================================================
# AUDIT LOGGER
# ============================================================

class RecoveryAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            RecoveryAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        plan_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> RecoveryAuditRecord:
        record = RecoveryAuditRecord(
            audit_id=new_id(),
            plan_id=plan_id,
            actor_id=actor_id,
            action=action,
            timestamp=utc_now(),
            details=dict(details or {}),
        )

        with self._lock:
            self._records.append(
                record
            )

        return record

    def list_records(
        self,
        plan_id: str | None = None,
    ) -> tuple[RecoveryAuditRecord, ...]:
        with self._lock:
            records = tuple(
                self._records
            )

        if plan_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.plan_id == plan_id
        )


# ============================================================
# HUMAN APPROVAL MANAGER
# ============================================================

class RecoveryHumanApprovalManager:
    ALLOWED_ROLES = {
        "district_official",
        "disaster_management_officer",
        "recovery_coordinator",
        "qualified_engineer",
    }

    def __init__(
        self,
        audit_logger: RecoveryAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._approvals: dict[
            str,
            RecoveryApprovalRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        plan: RecoveryPlan,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ApprovalDecision,
        reason: str,
    ) -> RecoveryApprovalRecord:
        if not reviewer_id.strip():
            raise RecoveryApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise RecoveryApprovalError(
                "Reviewer role not permitted"
            )

        if not reason.strip():
            raise RecoveryApprovalError(
                "Review reason required"
            )

        with self._lock:
            if plan.plan_id in self._approvals:
                raise RecoveryApprovalError(
                    "Plan already reviewed"
                )

            approval = RecoveryApprovalRecord(
                approval_id=new_id(),
                plan_id=plan.plan_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._approvals[
                plan.plan_id
            ] = approval

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id=reviewer_id,
            action=f"review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return approval


# ============================================================
# MAIN RECOVERY PLANNING AGENT
# ============================================================

class RecoveryPlanningAgent:
    def __init__(
        self,
        config: RecoveryPlanningConfig | None = None,
    ) -> None:
        self.config = (
            config or RecoveryPlanningConfig()
        )

        self.evidence_validator = (
            RecoveryEvidenceValidator(
                self.config
            )
        )

        self.priority_engine = (
            RecoveryPriorityEngine(
                self.config
            )
        )

        self.cost_engine = (
            RecoveryCostEstimationEngine(
                self.config
            )
        )

        self.resource_requirement_engine = (
            RecoveryResourceRequirementEngine()
        )

        self.task_generator = (
            RecoveryTaskGenerator(
                self.resource_requirement_engine
            )
        )

        self.resource_allocation_engine = (
            RecoveryResourceAllocationEngine()
        )

        self.budget_engine = (
            RecoveryBudgetEngine()
        )

        self.dependency_planner = (
            RecoveryDependencyPlanner()
        )

        self.audit_logger = (
            RecoveryAuditLogger()
        )

        self.approval_manager = (
            RecoveryHumanApprovalManager(
                self.audit_logger
            )
        )

        self._plans: dict[
            str,
            RecoveryPlan,
        ] = {}

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # CREATE RECOVERY PLAN
    # --------------------------------------------------------

    def run(
        self,
        request: RecoveryPlanningRequest,
        *,
        now: datetime | None = None,
    ) -> RecoveryPlan:
        now = now or utc_now()

        validate_datetime(now)

        if request.created_at > now:
            raise RecoveryValidationError(
                "Planning request timestamp is in the future"
            )

        if len(request.assets) > self.config.max_assets:
            raise RecoveryValidationError(
                "Asset limit exceeded"
            )

        asset_ids = [
            asset.asset_id
            for asset in request.assets
        ]

        if len(asset_ids) != len(set(asset_ids)):
            raise RecoveryValidationError(
                "Duplicate asset IDs"
            )

        assessments = []
        costs = []
        tasks = []
        warnings = []

        for asset in request.assets:
            if (
                asset.district.casefold()
                != request.district.casefold()
            ):
                raise RecoveryValidationError(
                    "Asset district mismatch"
                )

            evidence_verified, evidence_warnings = (
                self.evidence_validator.validate(
                    asset,
                    now=now,
                )
            )

            warnings.extend(
                f"{asset.asset_id}: {warning}"
                for warning in evidence_warnings
            )

            assessment = (
                self.priority_engine.assess(
                    asset,
                    evidence_verified=evidence_verified,
                )
            )

            cost = self.cost_engine.estimate(
                asset
            )

            task = self.task_generator.generate(
                asset,
                assessment,
                cost,
                evidence_verified=evidence_verified,
            )

            assessments.append(
                assessment
            )

            costs.append(
                cost
            )

            tasks.append(
                task
            )

        if len(tasks) > self.config.max_tasks:
            raise RecoveryValidationError(
                "Task limit exceeded"
            )

        task_order = self.dependency_planner.order(
            tasks
        )

        resource_gaps = (
            self.resource_allocation_engine.calculate_gaps(
                tasks,
                request.resources,
                district=request.district,
            )
        )

        budget = self.budget_engine.assess(
            tasks,
            request.available_budget_inr,
        )

        if not budget.funding_sufficient:
            warnings.append(
                "Estimated recovery costs exceed "
                "available planning budget"
            )

        if any(
            gap.shortage_units > 0
            for gap in resource_gaps
        ):
            warnings.append(
                "Potential resource shortages detected"
            )

        warnings.extend([
            "Recovery plan is advisory only",
            "No work order has been issued",
            "No procurement has been authorized",
            "No funds have been transferred",
            "No construction has been initiated",
            "Damage and cost estimates require expert validation",
            "Task order is not a calendar schedule",
            "Resource quantities are preliminary assumptions",
            "Human approval is mandatory",
        ])

        plan = RecoveryPlan(
            plan_id=new_id(),
            request_id=request.request_id,
            district=request.district,
            assessments=tuple(assessments),
            cost_estimates=tuple(costs),
            tasks=tuple(tasks),
            resource_gaps=resource_gaps,
            budget=budget,
            proposed_task_order=task_order,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            human_approval_required=True,
            approved_for_planning=False,
            created_at=now,
        )

        with self._lock:
            self._plans[
                plan.plan_id
            ] = plan

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id="recovery_planning_agent",
            action="recovery_plan_created",
            details={
                "district": request.district,
                "asset_count": str(
                    len(request.assets)
                ),
                "task_count": str(
                    len(tasks)
                ),
            },
        )

        return plan

    # --------------------------------------------------------
    # HUMAN REVIEW
    # --------------------------------------------------------

    def review_plan(
        self,
        plan_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ApprovalDecision,
        reason: str,
    ) -> RecoveryPlan:
        with self._lock:
            plan = self._plans.get(
                plan_id
            )

            if plan is None:
                raise RecoveryApprovalError(
                    "Recovery plan not found"
                )

            self.approval_manager.review(
                plan,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            approved = (
                decision == ApprovalDecision.APPROVE
            )

            updated_tasks = tuple(
                replace(
                    task,
                    status=(
                        RecoveryTaskStatus.APPROVED_FOR_PLANNING
                        if approved
                        else RecoveryTaskStatus.REJECTED
                        if decision == ApprovalDecision.REJECT
                        else RecoveryTaskStatus.PROPOSED
                    ),
                )
                for task in plan.tasks
            )

            updated = replace(
                plan,
                tasks=updated_tasks,
                approved_for_planning=approved,
            )

            self._plans[
                plan_id
            ] = updated

            return updated

    # --------------------------------------------------------
    # GET PLAN
    # --------------------------------------------------------

    def get_plan(
        self,
        plan_id: str,
    ) -> RecoveryPlan | None:
        with self._lock:
            return self._plans.get(
                plan_id
            )

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def get_audit_history(
        self,
        plan_id: str,
    ) -> tuple[RecoveryAuditRecord, ...]:
        return self.audit_logger.list_records(
            plan_id
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class RecoveryPlanningReportExporter:
    @staticmethod
    def export_json(
        plan: RecoveryPlan,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise RecoveryExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                plan.to_dict(),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_recovery_planning_agent(
    config: RecoveryPlanningConfig | None = None,
) -> RecoveryPlanningAgent:
    return RecoveryPlanningAgent(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional demonstration data.
    evidence = RecoveryEvidence(
        evidence_id="DEMO-EVIDENCE-001",
        source_id="DEMO-INSPECTION-001",
        status=EvidenceStatus.VERIFIED,
        reliability=0.90,
        observed_at=now,
        description="Illustrative inspection record",
    )

    assets = (
        DamagedAsset(
            asset_id="BRIDGE-001",
            name="Illustrative Bridge",
            district="Thrissur",
            location="Fictional Zone A",
            sector=RecoverySector.BRIDGES,
            severity=DamageSeverity.SEVERE,
            affected_population=2500,
            essential_service=True,
            safety_concern=True,
            estimated_repair_cost_inr=2500000,
            evidence=(evidence,),
        ),
        DamagedAsset(
            asset_id="ROAD-001",
            name="Illustrative Road",
            district="Thrissur",
            location="Fictional Zone B",
            sector=RecoverySector.ROADS,
            severity=DamageSeverity.MODERATE,
            affected_population=1200,
            essential_service=True,
            safety_concern=False,
            estimated_repair_cost_inr=800000,
            evidence=(evidence,),
            dependencies=("BRIDGE-001",),
        ),
    )

    resources = (
        RecoveryResource(
            resource_id="ENGINEERS-001",
            resource_type=(
                RecoveryResourceType.ENGINEERING_TEAM
            ),
            district="Thrissur",
            available_units=2,
            verified=True,
        ),
        RecoveryResource(
            resource_id="EQUIPMENT-001",
            resource_type=RecoveryResourceType.EQUIPMENT,
            district="Thrissur",
            available_units=1,
            verified=True,
        ),
    )

    request = RecoveryPlanningRequest(
        request_id="DEMO-RECOVERY-001",
        district="Thrissur",
        assets=assets,
        resources=resources,
        available_budget_inr=3000000,
        created_at=now,
    )

    agent = create_recovery_planning_agent()

    plan = agent.run(
        request,
        now=now,
    )

    print(
        json.dumps(
            plan.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nProposed recovery task order:",
        plan.proposed_task_order,
    )

    print(
        "Estimated funding gap (INR):",
        plan.budget.funding_gap_inr,
    )

    reviewed = agent.review_plan(
        plan.plan_id,
        reviewer_id="DEMO-OFFICIAL-001",
        reviewer_role="district_official",
        decision=ApprovalDecision.APPROVE,
        reason="Illustrative planning review",
    )

    print(
        "Approved for planning:",
        reviewed.approved_for_planning,
    )

    RecoveryPlanningReportExporter.export_json(
        reviewed,
        "outputs/agents/recovery_planning_report.json",
    )

    # Approval here is for planning review only.
    # No real-world reconstruction is authorized.


if __name__ == "__main__":
    main()


__all__ = [
    "RecoveryPlanningError",
    "RecoveryValidationError",
    "RecoveryApprovalError",
    "RecoveryResourceError",
    "RecoveryExportError",
    "RecoverySector",
    "DamageSeverity",
    "RecoveryPriority",
    "RecoveryPhase",
    "RecoveryTaskStatus",
    "EvidenceStatus",
    "ApprovalDecision",
    "RecoveryResourceType",
    "RecoveryPlanningConfig",
    "RecoveryEvidence",
    "DamagedAsset",
    "RecoveryResource",
    "RecoveryPlanningRequest",
    "RecoveryPriorityAssessment",
    "RecoveryCostEstimate",
    "RecoveryResourceRequirement",
    "RecoveryTask",
    "RecoveryResourceGap",
    "RecoveryBudgetAssessment",
    "RecoveryPlan",
    "RecoveryApprovalRecord",
    "RecoveryAuditRecord",
    "RecoveryEvidenceValidator",
    "RecoveryPriorityEngine",
    "RecoveryCostEstimationEngine",
    "RecoveryResourceRequirementEngine",
    "RecoveryTaskGenerator",
    "RecoveryResourceAllocationEngine",
    "RecoveryBudgetEngine",
    "RecoveryDependencyPlanner",
    "RecoveryAuditLogger",
    "RecoveryHumanApprovalManager",
    "RecoveryPlanningAgent",
    "RecoveryPlanningReportExporter",
    "create_recovery_planning_agent",
]
