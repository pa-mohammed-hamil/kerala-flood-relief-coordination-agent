"""Placeholder module: ai/agents/human_approval_agent.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
backend/ai/agents/human_approval_agent.py

Human-in-the-Loop Approval & Oversight Agent
===========================================

Features:
    - Centralized approval for AI agent recommendations
    - Role-based access control (RBAC)
    - Multi-level approval workflows
    - Risk-based escalation
    - Separation of duties
    - Emergency action safeguards
    - Evidence verification
    - Approval expiration
    - Decision revision and rejection
    - Audit logging
    - Workflow status tracking
    - Idempotent decision handling
    - JSON report export

Integrates conceptually with:
    - alert_intelligence_agent.py
    - communication_agent.py
    - damage_assessment_agent.py
    - evacuation_agent.py
    - flood_prediction_agent.py
    - rescue_agent.py
    - shelter_agent.py
    - relief_agent.py

IMPORTANT:
    This module records approval decisions.
    It does NOT execute approved actions.

    No automatic:
        - Evacuation orders
        - Public emergency alerts
        - Rescue dispatch
        - Dam or infrastructure control
        - Medical treatment decisions
        - Public communication delivery

    Identity verification, authentication,
    durable persistence, official authority
    checks and execution controls must be
    implemented by the production platform.

Python 3.11+
Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class HumanApprovalError(Exception):
    """Base approval workflow error."""


class ApprovalValidationError(HumanApprovalError):
    """Invalid approval request or decision."""


class ApprovalPermissionError(HumanApprovalError):
    """Reviewer lacks required permission."""


class ApprovalStateError(HumanApprovalError):
    """Invalid workflow state transition."""


class ApprovalNotFoundError(HumanApprovalError):
    """Requested approval does not exist."""


class ApprovalExportError(HumanApprovalError):
    """Report export failed."""


# ============================================================
# ENUMS
# ============================================================

class ApprovalActionType(str, Enum):
    FLOOD_ALERT = "flood_alert"
    PUBLIC_COMMUNICATION = "public_communication"
    EVACUATION_PLANNING = "evacuation_planning"
    EVACUATION_ORDER = "evacuation_order"
    RESCUE_DISPATCH = "rescue_dispatch"
    SHELTER_ALLOCATION = "shelter_allocation"
    RELIEF_DISTRIBUTION = "relief_distribution"
    DAMAGE_ASSESSMENT = "damage_assessment"
    FLOOD_FORECAST = "flood_forecast"
    MEDICAL_COORDINATION = "medical_coordination"
    INFRASTRUCTURE_CONTROL = "infrastructure_control"


class ApprovalRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    ESCALATED = "escalated"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalDecisionType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class ReviewerRole(str, Enum):
    DISTRICT_OFFICIAL = "district_official"
    DISASTER_MANAGEMENT_OFFICER = (
        "disaster_management_officer"
    )
    EMERGENCY_COORDINATOR = "emergency_coordinator"
    RESCUE_COMMANDER = "rescue_commander"
    MEDICAL_OFFICER = "medical_officer"
    HYDROLOGY_SPECIALIST = "hydrology_specialist"
    QUALIFIED_ENGINEER = "qualified_engineer"
    SHELTER_COORDINATOR = "shelter_coordinator"
    RELIEF_COORDINATOR = "relief_coordinator"
    COMMUNICATION_SUPERVISOR = (
        "communication_supervisor"
    )
    SYSTEM_AUDITOR = "system_auditor"


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    STALE = "stale"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class HumanApprovalConfig:
    default_expiry_minutes: int = 60
    critical_expiry_minutes: int = 15

    maximum_evidence_age_minutes: int = 180

    minimum_evidence_reliability: float = 0.70

    require_verified_evidence: bool = True

    require_separation_of_duties: bool = True

    require_two_person_critical_approval: bool = True

    maximum_evidence_items: int = 50

    maximum_decisions_per_request: int = 20

    def __post_init__(self) -> None:
        for name in (
            "default_expiry_minutes",
            "critical_expiry_minutes",
            "maximum_evidence_age_minutes",
            "maximum_evidence_items",
            "maximum_decisions_per_request",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ApprovalValidationError(
                    f"{name} must be a positive integer"
                )

        require_probability(
            self.minimum_evidence_reliability,
            "minimum_evidence_reliability",
        )


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def require_aware_datetime(
    value: datetime,
) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ApprovalValidationError(
            "Datetime must be timezone-aware"
        )


def require_probability(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ApprovalValidationError(
            f"{name} must be in [0,1]"
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
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            serialize(item)
            for item in value
        ]

    return value


def canonical_hash(
    payload: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        serialize(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ApprovalEvidence:
    evidence_id: str
    source_id: str

    observed_at: datetime

    status: EvidenceStatus
    reliability: float

    summary: str

    reference_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ApprovalValidationError(
                "evidence_id is required"
            )

        if not self.source_id.strip():
            raise ApprovalValidationError(
                "source_id is required"
            )

        require_aware_datetime(
            self.observed_at
        )

        require_probability(
            self.reliability,
            "reliability",
        )


@dataclass(frozen=True)
class ApprovalReviewer:
    reviewer_id: str
    roles: frozenset[ReviewerRole]

    active: bool = True

    districts: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.reviewer_id.strip():
            raise ApprovalValidationError(
                "reviewer_id is required"
            )

        if not self.roles:
            raise ApprovalValidationError(
                "Reviewer must have at least one role"
            )


@dataclass(frozen=True)
class ApprovalRequest:
    request_id: str

    source_agent: str
    source_record_id: str

    action_type: ApprovalActionType
    risk_level: ApprovalRiskLevel

    district: str

    title: str
    description: str

    proposed_action: Mapping[str, Any]

    evidence: tuple[ApprovalEvidence, ...]

    requested_by: str
    requested_at: datetime

    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "source_agent",
            "source_record_id",
            "district",
            "title",
            "description",
            "requested_by",
        ):
            if not getattr(self, name).strip():
                raise ApprovalValidationError(
                    f"{name} is required"
                )

        require_aware_datetime(
            self.requested_at
        )

        if self.expires_at is not None:
            require_aware_datetime(
                self.expires_at
            )

            if self.expires_at <= self.requested_at:
                raise ApprovalValidationError(
                    "Expiry must follow request time"
                )


@dataclass(frozen=True)
class ApprovalPolicy:
    action_type: ApprovalActionType

    minimum_risk_level: ApprovalRiskLevel

    required_roles: frozenset[ReviewerRole]

    minimum_approvals: int

    require_verified_evidence: bool = True

    allow_approval: bool = True


@dataclass(frozen=True)
class ApprovalDecision:
    decision_id: str
    request_id: str

    reviewer_id: str
    reviewer_role: ReviewerRole

    decision: ApprovalDecisionType

    reason: str

    decided_at: datetime

    payload_hash: str


@dataclass(frozen=True)
class ApprovalWorkflow:
    workflow_id: str
    request: ApprovalRequest

    status: ApprovalStatus

    required_approvals: int

    required_roles: frozenset[ReviewerRole]

    decisions: tuple[ApprovalDecision, ...]

    verified_evidence_ids: tuple[str, ...]

    warnings: tuple[str, ...]

    payload_hash: str

    created_at: datetime
    expires_at: datetime

    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ApprovalAuditRecord:
    audit_id: str
    workflow_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


@dataclass(frozen=True)
class ApprovalSummary:
    workflow_id: str
    status: ApprovalStatus

    required_approvals: int
    approvals_received: int

    remaining_approvals: int

    expires_at: datetime

    approved_for_execution: bool

    warnings: tuple[str, ...]


# ============================================================
# RISK LEVEL UTILITIES
# ============================================================

class ApprovalRiskPolicy:
    RISK_ORDER = {
        ApprovalRiskLevel.LOW: 0,
        ApprovalRiskLevel.MODERATE: 1,
        ApprovalRiskLevel.HIGH: 2,
        ApprovalRiskLevel.CRITICAL: 3,
    }

    @classmethod
    def is_at_least(
        cls,
        actual: ApprovalRiskLevel,
        minimum: ApprovalRiskLevel,
    ) -> bool:
        return (
            cls.RISK_ORDER[actual]
            >= cls.RISK_ORDER[minimum]
        )


# ============================================================
# RBAC POLICY ENGINE
# ============================================================

class ApprovalRBACEngine:
    """
    Application-level role checks.

    This does not authenticate identities.
    Production callers must use trusted
    identity-provider claims, not client-
    supplied reviewer objects.
    """

    ACTION_ROLES = {
        ApprovalActionType.FLOOD_ALERT: frozenset({
            ReviewerRole.DISTRICT_OFFICIAL,
            ReviewerRole.DISASTER_MANAGEMENT_OFFICER,
        }),

        ApprovalActionType.PUBLIC_COMMUNICATION: frozenset({
            ReviewerRole.DISTRICT_OFFICIAL,
            ReviewerRole.COMMUNICATION_SUPERVISOR,
        }),

        ApprovalActionType.EVACUATION_PLANNING: frozenset({
            ReviewerRole.DISTRICT_OFFICIAL,
            ReviewerRole.EMERGENCY_COORDINATOR,
        }),

        ApprovalActionType.EVACUATION_ORDER: frozenset({
            ReviewerRole.DISTRICT_OFFICIAL,
            ReviewerRole.DISASTER_MANAGEMENT_OFFICER,
        }),

        ApprovalActionType.RESCUE_DISPATCH: frozenset({
            ReviewerRole.RESCUE_COMMANDER,
            ReviewerRole.EMERGENCY_COORDINATOR,
        }),

        ApprovalActionType.SHELTER_ALLOCATION: frozenset({
            ReviewerRole.SHELTER_COORDINATOR,
            ReviewerRole.DISTRICT_OFFICIAL,
        }),

        ApprovalActionType.RELIEF_DISTRIBUTION: frozenset({
            ReviewerRole.RELIEF_COORDINATOR,
            ReviewerRole.DISTRICT_OFFICIAL,
        }),

        ApprovalActionType.DAMAGE_ASSESSMENT: frozenset({
            ReviewerRole.QUALIFIED_ENGINEER,
            ReviewerRole.DISTRICT_OFFICIAL,
        }),

        ApprovalActionType.FLOOD_FORECAST: frozenset({
            ReviewerRole.HYDROLOGY_SPECIALIST,
            ReviewerRole.DISASTER_MANAGEMENT_OFFICER,
        }),

        ApprovalActionType.MEDICAL_COORDINATION: frozenset({
            ReviewerRole.MEDICAL_OFFICER,
            ReviewerRole.EMERGENCY_COORDINATOR,
        }),

        ApprovalActionType.INFRASTRUCTURE_CONTROL: frozenset({
            ReviewerRole.QUALIFIED_ENGINEER,
            ReviewerRole.DISTRICT_OFFICIAL,
        }),
    }

    # These workflows may be reviewed as plans,
    # but this module never authorizes physical
    # infrastructure operations.
    NON_EXECUTABLE_ACTIONS = frozenset({
        ApprovalActionType.INFRASTRUCTURE_CONTROL,
        ApprovalActionType.EVACUATION_ORDER,
        ApprovalActionType.RESCUE_DISPATCH,
        ApprovalActionType.FLOOD_ALERT,
        ApprovalActionType.PUBLIC_COMMUNICATION,
        ApprovalActionType.MEDICAL_COORDINATION,
    })

    def authorize(
        self,
        *,
        reviewer: ApprovalReviewer,
        workflow: ApprovalWorkflow,
    ) -> ReviewerRole:
        if not reviewer.active:
            raise ApprovalPermissionError(
                "Reviewer is inactive"
            )

        if (
            reviewer.districts
            and workflow.request.district.casefold()
            not in {
                district.casefold()
                for district in reviewer.districts
            }
        ):
            raise ApprovalPermissionError(
                "Reviewer is not assigned to this district"
            )

        permitted_roles = (
            self.ACTION_ROLES[
                workflow.request.action_type
            ]
        )

        matched = (
            reviewer.roles
            & permitted_roles
            & workflow.required_roles
        )

        if not matched:
            raise ApprovalPermissionError(
                "Reviewer lacks an authorized role"
            )

        return sorted(
            matched,
            key=lambda role: role.value,
        )[0]


# ============================================================
# APPROVAL POLICY ENGINE
# ============================================================

class ApprovalPolicyEngine:
    def __init__(
        self,
        config: HumanApprovalConfig,
    ) -> None:
        self.config = config

    def resolve(
        self,
        request: ApprovalRequest,
    ) -> ApprovalPolicy:
        roles = ApprovalRBACEngine.ACTION_ROLES[
            request.action_type
        ]

        minimum_approvals = 1

        if (
            request.risk_level
            == ApprovalRiskLevel.CRITICAL
            and self.config.require_two_person_critical_approval
        ):
            minimum_approvals = 2

        # Higher-impact actions always need
        # at least two independent reviewers.
        if request.action_type in (
            ApprovalActionType.EVACUATION_ORDER,
            ApprovalActionType.INFRASTRUCTURE_CONTROL,
            ApprovalActionType.RESCUE_DISPATCH,
            ApprovalActionType.FLOOD_ALERT,
        ):
            minimum_approvals = max(
                minimum_approvals,
                2,
            )

        return ApprovalPolicy(
            action_type=request.action_type,
            minimum_risk_level=request.risk_level,
            required_roles=roles,
            minimum_approvals=minimum_approvals,
            require_verified_evidence=(
                self.config.require_verified_evidence
            ),
            allow_approval=True,
        )


# ============================================================
# EVIDENCE VERIFICATION ENGINE
# ============================================================

class ApprovalEvidenceVerifier:
    def __init__(
        self,
        config: HumanApprovalConfig,
    ) -> None:
        self.config = config

    def verify(
        self,
        evidence: Sequence[ApprovalEvidence],
        *,
        now: datetime,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
    ]:
        if len(evidence) > (
            self.config.maximum_evidence_items
        ):
            raise ApprovalValidationError(
                "Evidence limit exceeded"
            )

        verified_ids = []
        warnings = []
        seen = set()

        for item in evidence:
            reason = None

            if item.evidence_id in seen:
                reason = "duplicate evidence ID"

            elif item.status != EvidenceStatus.VERIFIED:
                reason = "evidence is not verified"

            elif item.reliability < (
                self.config.minimum_evidence_reliability
            ):
                reason = "low evidence reliability"

            elif item.observed_at > (
                now + timedelta(minutes=5)
            ):
                reason = "future evidence timestamp"

            elif (
                now - item.observed_at
            ) > timedelta(
                minutes=(
                    self.config.maximum_evidence_age_minutes
                )
            ):
                reason = "stale evidence"

            seen.add(item.evidence_id)

            if reason:
                warnings.append(
                    f"{item.evidence_id}: {reason}"
                )

            else:
                verified_ids.append(
                    item.evidence_id
                )

        return (
            tuple(verified_ids),
            tuple(warnings),
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class HumanApprovalAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            ApprovalAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        workflow_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> ApprovalAuditRecord:
        record = ApprovalAuditRecord(
            audit_id=generate_id(),
            workflow_id=workflow_id,
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
        workflow_id: str | None = None,
    ) -> tuple[ApprovalAuditRecord, ...]:
        with self._lock:
            records = tuple(
                self._records
            )

        if workflow_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.workflow_id == workflow_id
        )


# ============================================================
# HUMAN APPROVAL AGENT
# ============================================================

class HumanApprovalAgent:
    def __init__(
        self,
        config: HumanApprovalConfig | None = None,
    ) -> None:
        self.config = (
            config or HumanApprovalConfig()
        )

        self.policy_engine = (
            ApprovalPolicyEngine(
                self.config
            )
        )

        self.rbac_engine = (
            ApprovalRBACEngine()
        )

        self.evidence_verifier = (
            ApprovalEvidenceVerifier(
                self.config
            )
        )

        self.audit_logger = (
            HumanApprovalAuditLogger()
        )

        self._workflows: dict[
            str,
            ApprovalWorkflow,
        ] = {}

        self._request_index: dict[
            str,
            str,
        ] = {}

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # CREATE WORKFLOW
    # --------------------------------------------------------

    def submit(
        self,
        request: ApprovalRequest,
        *,
        now: datetime | None = None,
    ) -> ApprovalWorkflow:
        now = now or utc_now()
        require_aware_datetime(now)

        if request.requested_at > (
            now + timedelta(minutes=5)
        ):
            raise ApprovalValidationError(
                "Request timestamp is in the future"
            )

        payload_hash = canonical_hash({
            "source_agent": request.source_agent,
            "source_record_id": request.source_record_id,
            "action_type": request.action_type,
            "risk_level": request.risk_level,
            "district": request.district,
            "proposed_action": request.proposed_action,
            "evidence_ids": sorted(
                item.evidence_id
                for item in request.evidence
            ),
        })

        with self._lock:
            existing_id = self._request_index.get(
                request.request_id
            )

            if existing_id is not None:
                existing = self._workflows[
                    existing_id
                ]

                if existing.payload_hash != payload_hash:
                    raise ApprovalValidationError(
                        "Request ID reused with different payload"
                    )

                return existing

            policy = self.policy_engine.resolve(
                request
            )

            (
                verified_ids,
                evidence_warnings,
            ) = self.evidence_verifier.verify(
                request.evidence,
                now=now,
            )

            if (
                policy.require_verified_evidence
                and not verified_ids
            ):
                raise ApprovalValidationError(
                    "At least one current verified "
                    "evidence item is required"
                )

            expiry_minutes = (
                self.config.critical_expiry_minutes
                if request.risk_level
                == ApprovalRiskLevel.CRITICAL
                else self.config.default_expiry_minutes
            )

            expires_at = (
                request.expires_at
                or now + timedelta(
                    minutes=expiry_minutes
                )
            )

            if expires_at <= now:
                raise ApprovalValidationError(
                    "Request has already expired"
                )

            warnings = list(
                evidence_warnings
            )

            warnings.extend([
                "Approval records review decisions only",
                "No real-world action is executed",
                "Reviewer identity must be authenticated externally",
                "Authorization must be checked again "
                "by any downstream execution service",
            ])

            if request.action_type in (
                self.rbac_engine.NON_EXECUTABLE_ACTIONS
            ):
                warnings.append(
                    "Approval is not operational authorization "
                    "for this high-impact action"
                )

            workflow = ApprovalWorkflow(
                workflow_id=generate_id(),
                request=request,
                status=ApprovalStatus.PENDING,
                required_approvals=(
                    policy.minimum_approvals
                ),
                required_roles=(
                    policy.required_roles
                ),
                decisions=(),
                verified_evidence_ids=verified_ids,
                warnings=tuple(warnings),
                payload_hash=payload_hash,
                created_at=now,
                expires_at=expires_at,
                updated_at=now,
            )

            self._workflows[
                workflow.workflow_id
            ] = workflow

            self._request_index[
                request.request_id
            ] = workflow.workflow_id

            self.audit_logger.record(
                workflow_id=workflow.workflow_id,
                actor_id=request.requested_by,
                action="approval_submitted",
                details={
                    "action_type": request.action_type.value,
                    "risk_level": request.risk_level.value,
                    "required_approvals": str(
                        policy.minimum_approvals
                    ),
                    "payload_hash": payload_hash,
                },
            )

            return workflow

    # --------------------------------------------------------
    # GET WORKFLOW
    # --------------------------------------------------------

    def get_workflow(
        self,
        workflow_id: str,
        *,
        now: datetime | None = None,
    ) -> ApprovalWorkflow:
        now = now or utc_now()
        require_aware_datetime(now)

        with self._lock:
            workflow = self._workflows.get(
                workflow_id
            )

            if workflow is None:
                raise ApprovalNotFoundError(
                    "Workflow not found"
                )

            return self._expire_if_needed(
                workflow,
                now=now,
            )

    # --------------------------------------------------------
    # EXPIRATION
    # --------------------------------------------------------

    def _expire_if_needed(
        self,
        workflow: ApprovalWorkflow,
        *,
        now: datetime,
    ) -> ApprovalWorkflow:
        if (
            workflow.status in (
                ApprovalStatus.PENDING,
                ApprovalStatus.ESCALATED,
            )
            and now >= workflow.expires_at
        ):
            workflow = replace(
                workflow,
                status=ApprovalStatus.EXPIRED,
                updated_at=now,
            )

            self._workflows[
                workflow.workflow_id
            ] = workflow

            self.audit_logger.record(
                workflow_id=workflow.workflow_id,
                actor_id="human_approval_agent",
                action="workflow_expired",
            )

        return workflow

    # --------------------------------------------------------
    # RECORD HUMAN DECISION
    # --------------------------------------------------------

    def decide(
        self,
        workflow_id: str,
        *,
        reviewer: ApprovalReviewer,
        decision: ApprovalDecisionType,
        reason: str,
        now: datetime | None = None,
    ) -> ApprovalWorkflow:
        now = now or utc_now()
        require_aware_datetime(now)

        if not reason.strip():
            raise ApprovalValidationError(
                "Decision reason is required"
            )

        with self._lock:
            workflow = self.get_workflow(
                workflow_id,
                now=now,
            )

            if workflow.status not in (
                ApprovalStatus.PENDING,
                ApprovalStatus.ESCALATED,
            ):
                raise ApprovalStateError(
                    "Workflow is not awaiting decisions"
                )

            if (
                self.config.require_separation_of_duties
                and reviewer.reviewer_id
                == workflow.request.requested_by
            ):
                raise ApprovalPermissionError(
                    "Requester cannot approve their own request"
                )

            if any(
                item.reviewer_id == reviewer.reviewer_id
                for item in workflow.decisions
            ):
                raise ApprovalStateError(
                    "Reviewer has already submitted a decision"
                )

            if len(workflow.decisions) >= (
                self.config.maximum_decisions_per_request
            ):
                raise ApprovalStateError(
                    "Maximum decision count exceeded"
                )

            reviewer_role = (
                self.rbac_engine.authorize(
                    reviewer=reviewer,
                    workflow=workflow,
                )
            )

            decision_record = ApprovalDecision(
                decision_id=generate_id(),
                request_id=workflow.request.request_id,
                reviewer_id=reviewer.reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                decided_at=now,
                payload_hash=workflow.payload_hash,
            )

            decisions = (
                workflow.decisions
                + (decision_record,)
            )

            if decision == ApprovalDecisionType.REJECT:
                new_status = ApprovalStatus.REJECTED

            elif decision == (
                ApprovalDecisionType.REQUEST_CHANGES
            ):
                new_status = (
                    ApprovalStatus.CHANGES_REQUESTED
                )

            else:
                approval_count = sum(
                    item.decision
                    == ApprovalDecisionType.APPROVE
                    for item in decisions
                )

                if approval_count >= (
                    workflow.required_approvals
                ):
                    new_status = ApprovalStatus.APPROVED

                else:
                    new_status = ApprovalStatus.ESCALATED

            updated = replace(
                workflow,
                decisions=decisions,
                status=new_status,
                updated_at=now,
            )

            self._workflows[
                workflow_id
            ] = updated

            self.audit_logger.record(
                workflow_id=workflow_id,
                actor_id=reviewer.reviewer_id,
                action=f"decision_{decision.value}",
                details={
                    "reviewer_role": reviewer_role.value,
                    "reason": reason,
                    "resulting_status": new_status.value,
                },
            )

            return updated

    # --------------------------------------------------------
    # CANCEL WORKFLOW
    # --------------------------------------------------------

    def cancel(
        self,
        workflow_id: str,
        *,
        actor_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> ApprovalWorkflow:
        now = now or utc_now()

        if not actor_id.strip() or not reason.strip():
            raise ApprovalValidationError(
                "Actor and reason are required"
            )

        with self._lock:
            workflow = self.get_workflow(
                workflow_id,
                now=now,
            )

            if workflow.status not in (
                ApprovalStatus.PENDING,
                ApprovalStatus.ESCALATED,
            ):
                raise ApprovalStateError(
                    "Only pending workflows can be cancelled"
                )

            if actor_id != workflow.request.requested_by:
                raise ApprovalPermissionError(
                    "Only the original requester may cancel "
                    "through this interface"
                )

            updated = replace(
                workflow,
                status=ApprovalStatus.CANCELLED,
                updated_at=now,
            )

            self._workflows[
                workflow_id
            ] = updated

            self.audit_logger.record(
                workflow_id=workflow_id,
                actor_id=actor_id,
                action="workflow_cancelled",
                details={"reason": reason},
            )

            return updated

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summarize(
        self,
        workflow_id: str,
        *,
        now: datetime | None = None,
    ) -> ApprovalSummary:
        workflow = self.get_workflow(
            workflow_id,
            now=now,
        )

        approvals = sum(
            decision.decision
            == ApprovalDecisionType.APPROVE
            for decision in workflow.decisions
        )

        # Deliberately false: workflow approval
        # never constitutes execution authorization.
        approved_for_execution = False

        return ApprovalSummary(
            workflow_id=workflow.workflow_id,
            status=workflow.status,
            required_approvals=workflow.required_approvals,
            approvals_received=approvals,
            remaining_approvals=max(
                0,
                workflow.required_approvals - approvals,
            ),
            expires_at=workflow.expires_at,
            approved_for_execution=approved_for_execution,
            warnings=workflow.warnings,
        )

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def get_audit_history(
        self,
        workflow_id: str,
    ) -> tuple[ApprovalAuditRecord, ...]:
        return self.audit_logger.list_records(
            workflow_id
        )


# ============================================================
# JSON REPORT EXPORTER
# ============================================================

class HumanApprovalReportExporter:
    @staticmethod
    def export_json(
        workflow: ApprovalWorkflow,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise ApprovalExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                workflow.to_dict(),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_human_approval_agent(
    config: HumanApprovalConfig | None = None,
) -> HumanApprovalAgent:
    return HumanApprovalAgent(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    now = utc_now()

    agent = create_human_approval_agent()

    request = ApprovalRequest(
        request_id=generate_id(),
        source_agent="evacuation_agent",
        source_record_id="PLAN-001",
        action_type=(
            ApprovalActionType.EVACUATION_PLANNING
        ),
        risk_level=ApprovalRiskLevel.CRITICAL,
        district="Thrissur",
        title="Review evacuation planning recommendation",
        description=(
            "Illustrative flood-risk evacuation "
            "planning review."
        ),
        proposed_action={
            "planning_only": True,
            "affected_area": "Illustrative Zone A",
            "estimated_population": 500,
        },
        evidence=(
            ApprovalEvidence(
                evidence_id="EVIDENCE-001",
                source_id="RIVER-STATION-001",
                observed_at=now - timedelta(minutes=10),
                status=EvidenceStatus.VERIFIED,
                reliability=0.95,
                summary="Illustrative river observation",
            ),
            ApprovalEvidence(
                evidence_id="EVIDENCE-002",
                source_id="FORECAST-SERVICE-001",
                observed_at=now - timedelta(minutes=5),
                status=EvidenceStatus.VERIFIED,
                reliability=0.90,
                summary="Illustrative flood forecast",
            ),
        ),
        requested_by="evacuation_agent_service",
        requested_at=now,
    )

    workflow = agent.submit(
        request,
        now=now,
    )

    reviewer_1 = ApprovalReviewer(
        reviewer_id="official-001",
        roles=frozenset({
            ReviewerRole.DISTRICT_OFFICIAL,
        }),
        districts=frozenset({
            "Thrissur",
        }),
    )

    reviewer_2 = ApprovalReviewer(
        reviewer_id="coordinator-001",
        roles=frozenset({
            ReviewerRole.EMERGENCY_COORDINATOR,
        }),
        districts=frozenset({
            "Thrissur",
        }),
    )

    workflow = agent.decide(
        workflow.workflow_id,
        reviewer=reviewer_1,
        decision=ApprovalDecisionType.APPROVE,
        reason="Approved for further planning review",
        now=now,
    )

    workflow = agent.decide(
        workflow.workflow_id,
        reviewer=reviewer_2,
        decision=ApprovalDecisionType.APPROVE,
        reason="Planning documentation reviewed",
        now=now,
    )

    print(
        json.dumps(
            workflow.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    summary = agent.summarize(
        workflow.workflow_id,
        now=now,
    )

    print(
        json.dumps(
            serialize(summary),
            indent=2,
            ensure_ascii=False,
        )
    )

    HumanApprovalReportExporter.export_json(
        workflow,
        "outputs/agents/human_approval_report.json",
    )

    # APPROVED means the proposed document was reviewed.
    # It does not trigger evacuation or any real-world action.


if __name__ == "__main__":
    main()


__all__ = [
    "HumanApprovalError",
    "ApprovalValidationError",
    "ApprovalPermissionError",
    "ApprovalStateError",
    "ApprovalNotFoundError",
    "ApprovalExportError",
    "ApprovalActionType",
    "ApprovalRiskLevel",
    "ApprovalStatus",
    "ApprovalDecisionType",
    "ReviewerRole",
    "EvidenceStatus",
    "HumanApprovalConfig",
    "ApprovalEvidence",
    "ApprovalReviewer",
    "ApprovalRequest",
    "ApprovalPolicy",
    "ApprovalDecision",
    "ApprovalWorkflow",
    "ApprovalAuditRecord",
    "ApprovalSummary",
    "ApprovalRiskPolicy",
    "ApprovalRBACEngine",
    "ApprovalPolicyEngine",
    "ApprovalEvidenceVerifier",
    "HumanApprovalAuditLogger",
    "HumanApprovalAgent",
    "HumanApprovalReportExporter",
    "create_human_approval_agent",
]
