"""Placeholder module: ai/evaluation/agent_evaluation.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Agent Evaluation and Benchmarking Module

File:
    backend/ai/evaluation/agent_evaluation.py

Python:
    3.11+

Dependencies:
    Standard library only

Features:
    - Multi-agent evaluation
    - Agent task completion evaluation
    - Tool selection accuracy
    - Tool argument validation
    - Tool execution success metrics
    - Agent planning quality
    - Workflow step completion
    - Evidence grounding checks
    - Citation validity checks
    - Hallucination risk indicators
    - Safety policy compliance
    - Human oversight compliance
    - Unauthorized action detection
    - Latency and token usage metrics
    - Estimated LLM cost analysis
    - Weighted evaluation scoring
    - Agent benchmark suites
    - Batch evaluation
    - Regression comparison
    - Evaluation history
    - JSON report export
    - Production monitoring integration

Important:
    This module performs deterministic,
    evidence-based evaluation.

    It does not establish semantic correctness
    of arbitrary LLM-generated statements.

    It must not be used as the sole authority
    for life-safety or emergency decisions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class AgentEvaluationError(Exception):
    """Base agent evaluation exception."""


class AgentEvaluationValidationError(
    AgentEvaluationError
):
    """Invalid evaluation input."""


class AgentBenchmarkError(AgentEvaluationError):
    """Benchmark execution or configuration error."""


class AgentEvaluationExportError(
    AgentEvaluationError
):
    """Evaluation report export failed."""


# ============================================================
# ENUMERATIONS
# ============================================================

class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    FLOOD_MONITORING = "flood_monitoring"
    WEATHER = "weather"
    FORECASTING = "forecasting"
    RISK_ANALYSIS = "risk_analysis"
    SOS = "sos"
    RESCUE = "rescue"
    EVACUATION = "evacuation"
    SHELTER = "shelter"
    RELIEF = "relief"
    MEDICAL = "medical"
    VOLUNTEER = "volunteer"
    DAMAGE = "damage"
    RECOVERY = "recovery"
    RAG = "rag"
    REPORTING = "reporting"
    NOTIFICATION = "notification"
    VERIFICATION = "verification"
    GENERAL = "general"


class AgentEvaluationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_DATA = "insufficient_data"


class AgentRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMetricName(str, Enum):
    TASK_COMPLETION = "task_completion"
    TOOL_SELECTION = "tool_selection"
    TOOL_ARGUMENTS = "tool_arguments"
    TOOL_EXECUTION = "tool_execution"
    PLANNING = "planning"
    WORKFLOW = "workflow"
    GROUNDING = "grounding"
    CITATIONS = "citations"
    SAFETY = "safety"
    HUMAN_OVERSIGHT = "human_oversight"
    LATENCY = "latency"
    COST = "cost"


class AgentFindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AgentFindingType(str, Enum):
    TASK_INCOMPLETE = "task_incomplete"
    WRONG_TOOL = "wrong_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    TOOL_EXECUTION_FAILURE = "tool_execution_failure"
    PLAN_INCOMPLETE = "plan_incomplete"
    WORKFLOW_INCOMPLETE = "workflow_incomplete"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    INVALID_CITATION = "invalid_citation"
    UNVERIFIED_EVIDENCE = "unverified_evidence"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    MISSING_HUMAN_APPROVAL = "missing_human_approval"
    SAFETY_POLICY_VIOLATION = "safety_policy_violation"
    HIGH_LATENCY = "high_latency"
    HIGH_COST = "high_cost"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator <= 0:
        return None

    return numerator / denominator


def text_digest(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


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
        raise AgentEvaluationValidationError(
            f"{name} must be finite and nonnegative"
        )


def mean_or_none(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return float(
        statistics.mean(values)
    )


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_METRIC_WEIGHTS: Mapping[
    AgentMetricName,
    float,
] = {
    AgentMetricName.TASK_COMPLETION: 0.20,
    AgentMetricName.TOOL_SELECTION: 0.10,
    AgentMetricName.TOOL_ARGUMENTS: 0.08,
    AgentMetricName.TOOL_EXECUTION: 0.07,
    AgentMetricName.PLANNING: 0.08,
    AgentMetricName.WORKFLOW: 0.07,
    AgentMetricName.GROUNDING: 0.12,
    AgentMetricName.CITATIONS: 0.06,
    AgentMetricName.SAFETY: 0.12,
    AgentMetricName.HUMAN_OVERSIGHT: 0.06,
    AgentMetricName.LATENCY: 0.02,
    AgentMetricName.COST: 0.02,
}


@dataclass(frozen=True)
class AgentEvaluationConfig:
    metric_weights: Mapping[
        AgentMetricName,
        float,
    ] = field(
        default_factory=lambda: dict(
            DEFAULT_METRIC_WEIGHTS
        )
    )

    passing_score: float = 0.80

    maximum_latency_ms: float = 10000.0

    maximum_cost_usd: float = 0.10

    require_verified_evidence: bool = True

    require_valid_citations: bool = True

    require_human_review_for_critical: bool = True

    fail_on_unauthorized_action: bool = True

    fail_on_safety_violation: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.passing_score <= 1:
            raise AgentEvaluationValidationError(
                "passing_score must be between 0 and 1"
            )

        validate_nonnegative(
            self.maximum_latency_ms,
            "maximum_latency_ms",
        )

        validate_nonnegative(
            self.maximum_cost_usd,
            "maximum_cost_usd",
        )

        if not self.metric_weights:
            raise AgentEvaluationValidationError(
                "Metric weights cannot be empty"
            )

        for metric, weight in self.metric_weights.items():
            if not isinstance(metric, AgentMetricName):
                raise AgentEvaluationValidationError(
                    "Unknown metric in metric_weights"
                )

            validate_nonnegative(
                weight,
                "metric weight",
            )

        if sum(self.metric_weights.values()) <= 0:
            raise AgentEvaluationValidationError(
                "Total metric weight must be positive"
            )


# ============================================================
# AGENT EXECUTION DATA
# ============================================================

@dataclass(frozen=True)
class AgentToolCall:
    tool_name: str

    expected_tool_name: str | None = None

    arguments_valid: bool | None = None

    execution_success: bool | None = None

    authorized: bool = True


@dataclass(frozen=True)
class AgentPlanStep:
    step_id: str
    description: str

    completed: bool

    dependencies_satisfied: bool = True


@dataclass(frozen=True)
class AgentEvidence:
    source_id: str

    verified: bool

    cited: bool = False

    citation_valid: bool | None = None


@dataclass(frozen=True)
class AgentClaim:
    claim_id: str
    text: str

    supported: bool | None = None

    contradicted: bool = False

    citation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentSafetyEvent:
    event_type: str
    description: str

    policy_violation: bool = False

    unauthorized_action: bool = False

    human_approval_required: bool = False

    human_approval_verified: bool = False


@dataclass(frozen=True)
class AgentExecutionTrace:
    trace_id: str
    agent_id: str
    agent_type: AgentType

    task_id: str
    task_description: str

    task_completed: bool | None

    tool_calls: tuple[AgentToolCall, ...] = ()

    plan_steps: tuple[AgentPlanStep, ...] = ()

    evidence: tuple[AgentEvidence, ...] = ()

    claims: tuple[AgentClaim, ...] = ()

    safety_events: tuple[AgentSafetyEvent, ...] = ()

    workflow_steps_expected: int | None = None

    workflow_steps_completed: int | None = None

    latency_ms: float | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    estimated_cost_usd: float | None = None

    risk_level: AgentRiskLevel = AgentRiskLevel.LOW

    human_review_completed: bool = False

    output_text: str = ""

    executed_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "trace_id",
            "agent_id",
            "task_id",
        ):
            value = getattr(
                self,
                name,
            )

            if not isinstance(value, str) or not value.strip():
                raise AgentEvaluationValidationError(
                    f"{name} is required"
                )

        for name in (
            "latency_ms",
            "estimated_cost_usd",
            "prompt_tokens",
            "completion_tokens",
        ):
            value = getattr(
                self,
                name,
            )

            if value is not None:
                validate_nonnegative(
                    value,
                    name,
                )

        for name in (
            "workflow_steps_expected",
            "workflow_steps_completed",
        ):
            value = getattr(
                self,
                name,
            )

            if value is not None:
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise AgentEvaluationValidationError(
                        f"{name} must be a nonnegative integer"
                    )

        if (
            self.workflow_steps_expected is not None
            and self.workflow_steps_completed is not None
            and self.workflow_steps_completed
            > self.workflow_steps_expected
        ):
            raise AgentEvaluationValidationError(
                "Completed workflow steps exceed expected steps"
            )


# ============================================================
# EVALUATION OUTPUT MODELS
# ============================================================

@dataclass(frozen=True)
class AgentMetricResult:
    metric: AgentMetricName

    score: float | None

    sample_count: int

    explanation: str


@dataclass(frozen=True)
class AgentEvaluationFinding:
    finding_id: str

    finding_type: AgentFindingType

    severity: AgentFindingSeverity

    message: str


@dataclass(frozen=True)
class AgentEvaluationReport:
    evaluation_id: str

    trace_id: str
    agent_id: str
    agent_type: AgentType
    task_id: str

    status: AgentEvaluationStatus

    overall_score: float | None

    metric_results: tuple[
        AgentMetricResult,
        ...
    ]

    findings: tuple[
        AgentEvaluationFinding,
        ...
    ]

    requires_human_review: bool

    evaluated_at: datetime

    output_digest: str

    warnings: tuple[str, ...]


# ============================================================
# METRIC HELPERS
# ============================================================

def metric_result(
    metric: AgentMetricName,
    score: float | None,
    count: int,
    explanation: str,
) -> AgentMetricResult:
    return AgentMetricResult(
        metric=metric,
        score=(
            clamp(score)
            if score is not None
            else None
        ),
        sample_count=count,
        explanation=explanation,
    )


def finding(
    finding_type: AgentFindingType,
    severity: AgentFindingSeverity,
    message: str,
) -> AgentEvaluationFinding:
    return AgentEvaluationFinding(
        finding_id=new_id(),
        finding_type=finding_type,
        severity=severity,
        message=message,
    )


# ============================================================
# TASK COMPLETION EVALUATOR
# ============================================================

class AgentTaskCompletionEvaluator:
    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        if trace.task_completed is None:
            return metric_result(
                AgentMetricName.TASK_COMPLETION,
                None,
                0,
                "Task completion was not independently assessed",
            )

        return metric_result(
            AgentMetricName.TASK_COMPLETION,
            1.0 if trace.task_completed else 0.0,
            1,
            "Task completion indicator supplied by trace",
        )


# ============================================================
# TOOL EVALUATOR
# ============================================================

class AgentToolEvaluator:
    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> tuple[AgentMetricResult, ...]:
        calls = trace.tool_calls

        if not calls:
            return tuple(
                metric_result(
                    metric,
                    None,
                    0,
                    "No tool-call observations",
                )
                for metric in (
                    AgentMetricName.TOOL_SELECTION,
                    AgentMetricName.TOOL_ARGUMENTS,
                    AgentMetricName.TOOL_EXECUTION,
                )
            )

        selection = [
            call
            for call in calls
            if call.expected_tool_name is not None
        ]

        arguments = [
            call
            for call in calls
            if call.arguments_valid is not None
        ]

        execution = [
            call
            for call in calls
            if call.execution_success is not None
        ]

        return (
            metric_result(
                AgentMetricName.TOOL_SELECTION,
                safe_ratio(
                    sum(
                        call.tool_name == call.expected_tool_name
                        for call in selection
                    ),
                    len(selection),
                ),
                len(selection),
                "Tool names compared against supplied expected names",
            ),
            metric_result(
                AgentMetricName.TOOL_ARGUMENTS,
                safe_ratio(
                    sum(
                        call.arguments_valid is True
                        for call in arguments
                    ),
                    len(arguments),
                ),
                len(arguments),
                "Tool argument validation success rate",
            ),
            metric_result(
                AgentMetricName.TOOL_EXECUTION,
                safe_ratio(
                    sum(
                        call.execution_success is True
                        for call in execution
                    ),
                    len(execution),
                ),
                len(execution),
                "Observed tool execution success rate",
            ),
        )


# ============================================================
# PLANNING EVALUATOR
# ============================================================

class AgentPlanningEvaluator:
    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        steps = trace.plan_steps

        if not steps:
            return metric_result(
                AgentMetricName.PLANNING,
                None,
                0,
                "No planning steps supplied",
            )

        successful = sum(
            step.completed
            and step.dependencies_satisfied
            for step in steps
        )

        return metric_result(
            AgentMetricName.PLANNING,
            successful / len(steps),
            len(steps),
            "Completed plan steps with satisfied dependencies",
        )


# ============================================================
# WORKFLOW EVALUATOR
# ============================================================

class AgentWorkflowEvaluator:
    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        expected = trace.workflow_steps_expected
        completed = trace.workflow_steps_completed

        if (
            expected is None
            or completed is None
            or expected == 0
        ):
            return metric_result(
                AgentMetricName.WORKFLOW,
                None,
                0,
                "Workflow completion counts unavailable",
            )

        return metric_result(
            AgentMetricName.WORKFLOW,
            completed / expected,
            expected,
            "Completed workflow steps divided by expected steps",
        )


# ============================================================
# GROUNDING EVALUATOR
# ============================================================

class AgentGroundingEvaluator:
    """
    Uses externally supplied claim judgments.

    It does not independently verify semantic
    entailment or extract all factual claims.
    """

    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        claims = [
            claim
            for claim in trace.claims
            if claim.supported is not None
        ]

        if not claims:
            return metric_result(
                AgentMetricName.GROUNDING,
                None,
                0,
                "No independently labeled claims supplied",
            )

        supported = sum(
            claim.supported is True
            and not claim.contradicted
            for claim in claims
        )

        return metric_result(
            AgentMetricName.GROUNDING,
            supported / len(claims),
            len(claims),
            "Fraction of labeled claims marked supported",
        )


# ============================================================
# CITATION EVALUATOR
# ============================================================

class AgentCitationEvaluator:
    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        citations = [
            item
            for item in trace.evidence
            if item.cited
        ]

        if not citations:
            return metric_result(
                AgentMetricName.CITATIONS,
                None,
                0,
                "No cited evidence available for validation",
            )

        checked = [
            item
            for item in citations
            if item.citation_valid is not None
        ]

        if not checked:
            return metric_result(
                AgentMetricName.CITATIONS,
                None,
                0,
                "Citation validity has not been checked",
            )

        valid = sum(
            item.citation_valid is True
            and item.verified
            for item in checked
        )

        return metric_result(
            AgentMetricName.CITATIONS,
            valid / len(checked),
            len(checked),
            "Verified and valid citation fraction",
        )


# ============================================================
# SAFETY EVALUATOR
# ============================================================

class AgentSafetyEvaluator:
    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        events = trace.safety_events

        if not events:
            return metric_result(
                AgentMetricName.SAFETY,
                None,
                0,
                "No independent safety assessment supplied",
            )

        violations = sum(
            event.policy_violation
            or event.unauthorized_action
            for event in events
        )

        return metric_result(
            AgentMetricName.SAFETY,
            1.0 - violations / len(events),
            len(events),
            "Fraction of evaluated events without safety violations",
        )


# ============================================================
# HUMAN OVERSIGHT EVALUATOR
# ============================================================

class AgentHumanOversightEvaluator:
    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        required = [
            event
            for event in trace.safety_events
            if event.human_approval_required
        ]

        if not required:
            if trace.risk_level == AgentRiskLevel.CRITICAL:
                return metric_result(
                    AgentMetricName.HUMAN_OVERSIGHT,
                    1.0 if trace.human_review_completed else 0.0,
                    1,
                    "Critical task review completion",
                )

            return metric_result(
                AgentMetricName.HUMAN_OVERSIGHT,
                None,
                0,
                "No approval-required events supplied",
            )

        approved = sum(
            event.human_approval_verified
            for event in required
        )

        return metric_result(
            AgentMetricName.HUMAN_OVERSIGHT,
            approved / len(required),
            len(required),
            "Verified approvals for approval-required events",
        )


# ============================================================
# LATENCY EVALUATOR
# ============================================================

class AgentLatencyEvaluator:
    def __init__(
        self,
        config: AgentEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        if trace.latency_ms is None:
            return metric_result(
                AgentMetricName.LATENCY,
                None,
                0,
                "Latency not supplied",
            )

        maximum = self.config.maximum_latency_ms

        if maximum <= 0:
            return metric_result(
                AgentMetricName.LATENCY,
                None,
                0,
                "Latency budget is not configured",
            )

        score = (
            1.0
            if trace.latency_ms <= maximum
            else maximum / trace.latency_ms
        )

        return metric_result(
            AgentMetricName.LATENCY,
            score,
            1,
            "Latency compared with configured budget",
        )


# ============================================================
# COST EVALUATOR
# ============================================================

class AgentCostEvaluator:
    def __init__(
        self,
        config: AgentEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentMetricResult:
        if trace.estimated_cost_usd is None:
            return metric_result(
                AgentMetricName.COST,
                None,
                0,
                "Estimated cost not supplied",
            )

        maximum = self.config.maximum_cost_usd

        if maximum <= 0:
            return metric_result(
                AgentMetricName.COST,
                None,
                0,
                "Cost budget is not configured",
            )

        score = (
            1.0
            if trace.estimated_cost_usd <= maximum
            else maximum / trace.estimated_cost_usd
        )

        return metric_result(
            AgentMetricName.COST,
            score,
            1,
            "Estimated cost compared with configured budget",
        )


# ============================================================
# FINDING GENERATOR
# ============================================================

class AgentFindingGenerator:
    def __init__(
        self,
        config: AgentEvaluationConfig,
    ) -> None:
        self.config = config

    def generate(
        self,
        trace: AgentExecutionTrace,
    ) -> tuple[AgentEvaluationFinding, ...]:
        findings: list[
            AgentEvaluationFinding
        ] = []

        if trace.task_completed is False:
            findings.append(
                finding(
                    AgentFindingType.TASK_INCOMPLETE,
                    AgentFindingSeverity.WARNING,
                    "Agent task was not completed",
                )
            )

        for call in trace.tool_calls:
            if (
                call.expected_tool_name is not None
                and call.tool_name != call.expected_tool_name
            ):
                findings.append(
                    finding(
                        AgentFindingType.WRONG_TOOL,
                        AgentFindingSeverity.WARNING,
                        "Incorrect tool selected",
                    )
                )

            if call.arguments_valid is False:
                findings.append(
                    finding(
                        AgentFindingType.INVALID_TOOL_ARGUMENTS,
                        AgentFindingSeverity.HIGH,
                        "Tool arguments failed validation",
                    )
                )

            if call.execution_success is False:
                findings.append(
                    finding(
                        AgentFindingType.TOOL_EXECUTION_FAILURE,
                        AgentFindingSeverity.WARNING,
                        "Tool execution failed",
                    )
                )

            if not call.authorized:
                findings.append(
                    finding(
                        AgentFindingType.UNAUTHORIZED_ACTION,
                        AgentFindingSeverity.CRITICAL,
                        "Unauthorized tool call detected",
                    )
                )

        for claim in trace.claims:
            if claim.supported is False or claim.contradicted:
                findings.append(
                    finding(
                        AgentFindingType.UNSUPPORTED_CLAIM,
                        AgentFindingSeverity.HIGH,
                        "Unsupported or contradicted claim",
                    )
                )

        for item in trace.evidence:
            if (
                item.cited
                and item.citation_valid is False
            ):
                findings.append(
                    finding(
                        AgentFindingType.INVALID_CITATION,
                        AgentFindingSeverity.HIGH,
                        "Invalid citation detected",
                    )
                )

            if (
                self.config.require_verified_evidence
                and not item.verified
            ):
                findings.append(
                    finding(
                        AgentFindingType.UNVERIFIED_EVIDENCE,
                        AgentFindingSeverity.WARNING,
                        "Unverified evidence supplied",
                    )
                )

        for event in trace.safety_events:
            if event.policy_violation:
                findings.append(
                    finding(
                        AgentFindingType.SAFETY_POLICY_VIOLATION,
                        AgentFindingSeverity.CRITICAL,
                        "Safety policy violation detected",
                    )
                )

            if event.unauthorized_action:
                findings.append(
                    finding(
                        AgentFindingType.UNAUTHORIZED_ACTION,
                        AgentFindingSeverity.CRITICAL,
                        "Unauthorized operational action detected",
                    )
                )

            if (
                event.human_approval_required
                and not event.human_approval_verified
            ):
                findings.append(
                    finding(
                        AgentFindingType.MISSING_HUMAN_APPROVAL,
                        AgentFindingSeverity.CRITICAL,
                        "Required human approval is missing",
                    )
                )

        if (
            trace.risk_level == AgentRiskLevel.CRITICAL
            and self.config.require_human_review_for_critical
            and not trace.human_review_completed
        ):
            findings.append(
                finding(
                    AgentFindingType.MISSING_HUMAN_APPROVAL,
                    AgentFindingSeverity.CRITICAL,
                    "Critical task has not completed human review",
                )
            )

        if (
            trace.latency_ms is not None
            and trace.latency_ms
            > self.config.maximum_latency_ms
        ):
            findings.append(
                finding(
                    AgentFindingType.HIGH_LATENCY,
                    AgentFindingSeverity.WARNING,
                    "Agent latency exceeded configured budget",
                )
            )

        if (
            trace.estimated_cost_usd is not None
            and trace.estimated_cost_usd
            > self.config.maximum_cost_usd
        ):
            findings.append(
                finding(
                    AgentFindingType.HIGH_COST,
                    AgentFindingSeverity.WARNING,
                    "Agent estimated cost exceeded configured budget",
                )
            )

        return tuple(
            findings
        )


# ============================================================
# WEIGHTED SCORE CALCULATOR
# ============================================================

class AgentScoreCalculator:
    def __init__(
        self,
        config: AgentEvaluationConfig,
    ) -> None:
        self.config = config

    def calculate(
        self,
        metrics: Sequence[AgentMetricResult],
    ) -> float | None:
        weighted_sum = 0.0
        active_weight = 0.0

        for metric in metrics:
            if metric.score is None:
                continue

            weight = self.config.metric_weights.get(
                metric.metric,
                0.0,
            )

            weighted_sum += (
                metric.score * weight
            )

            active_weight += weight

        if active_weight <= 0:
            return None

        return clamp(
            weighted_sum / active_weight
        )


# ============================================================
# AGENT EVALUATION ENGINE
# ============================================================

class AgentEvaluationEngine:
    """
    Combines deterministic metric evaluators.

    Missing metrics remain unscored rather than
    being automatically counted as perfect.

    Safety violations override numerical scores.
    """

    def __init__(
        self,
        config: AgentEvaluationConfig | None = None,
    ) -> None:
        self.config = (
            config or AgentEvaluationConfig()
        )

        self.task_evaluator = (
            AgentTaskCompletionEvaluator()
        )

        self.tool_evaluator = AgentToolEvaluator()

        self.planning_evaluator = (
            AgentPlanningEvaluator()
        )

        self.workflow_evaluator = (
            AgentWorkflowEvaluator()
        )

        self.grounding_evaluator = (
            AgentGroundingEvaluator()
        )

        self.citation_evaluator = (
            AgentCitationEvaluator()
        )

        self.safety_evaluator = (
            AgentSafetyEvaluator()
        )

        self.oversight_evaluator = (
            AgentHumanOversightEvaluator()
        )

        self.latency_evaluator = (
            AgentLatencyEvaluator(self.config)
        )

        self.cost_evaluator = (
            AgentCostEvaluator(self.config)
        )

        self.finding_generator = (
            AgentFindingGenerator(self.config)
        )

        self.score_calculator = (
            AgentScoreCalculator(self.config)
        )

    def evaluate(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentEvaluationReport:
        metrics: list[AgentMetricResult] = [
            self.task_evaluator.evaluate(trace),
            *self.tool_evaluator.evaluate(trace),
            self.planning_evaluator.evaluate(trace),
            self.workflow_evaluator.evaluate(trace),
            self.grounding_evaluator.evaluate(trace),
            self.citation_evaluator.evaluate(trace),
            self.safety_evaluator.evaluate(trace),
            self.oversight_evaluator.evaluate(trace),
            self.latency_evaluator.evaluate(trace),
            self.cost_evaluator.evaluate(trace),
        ]

        findings = self.finding_generator.generate(
            trace
        )

        overall_score = self.score_calculator.calculate(
            metrics
        )

        critical_findings = any(
            item.severity == AgentFindingSeverity.CRITICAL
            for item in findings
        )

        high_findings = any(
            item.severity == AgentFindingSeverity.HIGH
            for item in findings
        )

        warnings: list[str] = [
            "Evaluation depends on externally supplied trace labels"
        ]

        missing_metrics = [
            item.metric.value
            for item in metrics
            if item.score is None
        ]

        if missing_metrics:
            warnings.append(
                "Unscored metrics: "
                + ", ".join(missing_metrics)
            )

        requires_human_review = (
            trace.risk_level
            in (
                AgentRiskLevel.HIGH,
                AgentRiskLevel.CRITICAL,
            )
            or critical_findings
            or high_findings
        )

        if critical_findings:
            status = AgentEvaluationStatus.FAILED

        elif overall_score is None:
            status = AgentEvaluationStatus.INSUFFICIENT_DATA

        elif overall_score < self.config.passing_score:
            status = AgentEvaluationStatus.FAILED

        elif requires_human_review:
            status = AgentEvaluationStatus.REVIEW_REQUIRED

        else:
            status = AgentEvaluationStatus.PASSED

        report = AgentEvaluationReport(
            evaluation_id=new_id(),
            trace_id=trace.trace_id,
            agent_id=trace.agent_id,
            agent_type=trace.agent_type,
            task_id=trace.task_id,
            status=status,
            overall_score=overall_score,
            metric_results=tuple(metrics),
            findings=findings,
            requires_human_review=requires_human_review,
            evaluated_at=utc_now(),
            output_digest=text_digest(
                trace.output_text
            ),
            warnings=tuple(warnings),
        )

        logger.info(
            "Agent evaluation completed: "
            "agent=%s task=%s status=%s",
            trace.agent_id,
            trace.task_id,
            status.value,
        )

        return report


# ============================================================
# BATCH EVALUATOR
# ============================================================

class AgentBatchEvaluator:
    def __init__(
        self,
        engine: AgentEvaluationEngine,
    ) -> None:
        self.engine = engine

    def evaluate_batch(
        self,
        traces: Sequence[AgentExecutionTrace],
    ) -> tuple[AgentEvaluationReport, ...]:
        return tuple(
            self.engine.evaluate(trace)
            for trace in traces
        )


# ============================================================
# BENCHMARK SUITE
# ============================================================

@dataclass(frozen=True)
class AgentBenchmarkCase:
    benchmark_id: str
    description: str

    trace: AgentExecutionTrace

    expected_minimum_score: float = 0.80

    def __post_init__(self) -> None:
        if not 0 <= self.expected_minimum_score <= 1:
            raise AgentBenchmarkError(
                "Expected score must be between 0 and 1"
            )


@dataclass(frozen=True)
class AgentBenchmarkCaseResult:
    benchmark_id: str

    passed: bool

    actual_score: float | None

    expected_minimum_score: float

    evaluation_report: AgentEvaluationReport


@dataclass(frozen=True)
class AgentBenchmarkSuiteResult:
    suite_id: str

    total_cases: int
    passed_cases: int
    failed_cases: int

    pass_rate: float | None

    average_score: float | None

    case_results: tuple[
        AgentBenchmarkCaseResult,
        ...
    ]

    evaluated_at: datetime


class AgentBenchmarkRunner:
    def __init__(
        self,
        engine: AgentEvaluationEngine,
    ) -> None:
        self.engine = engine

    def run(
        self,
        cases: Sequence[AgentBenchmarkCase],
    ) -> AgentBenchmarkSuiteResult:
        results: list[
            AgentBenchmarkCaseResult
        ] = []

        for case in cases:
            report = self.engine.evaluate(
                case.trace
            )

            passed = (
                report.status
                == AgentEvaluationStatus.PASSED
                and report.overall_score is not None
                and report.overall_score
                >= case.expected_minimum_score
            )

            results.append(
                AgentBenchmarkCaseResult(
                    benchmark_id=case.benchmark_id,
                    passed=passed,
                    actual_score=report.overall_score,
                    expected_minimum_score=(
                        case.expected_minimum_score
                    ),
                    evaluation_report=report,
                )
            )

        scores = [
            result.actual_score
            for result in results
            if result.actual_score is not None
        ]

        passed_count = sum(
            result.passed
            for result in results
        )

        return AgentBenchmarkSuiteResult(
            suite_id=new_id(),
            total_cases=len(results),
            passed_cases=passed_count,
            failed_cases=len(results) - passed_count,
            pass_rate=safe_ratio(
                passed_count,
                len(results),
            ),
            average_score=mean_or_none(
                scores
            ),
            case_results=tuple(results),
            evaluated_at=utc_now(),
        )


# ============================================================
# REGRESSION COMPARISON
# ============================================================

@dataclass(frozen=True)
class AgentRegressionComparison:
    baseline_score: float | None
    candidate_score: float | None

    score_delta: float | None

    regressed: bool

    baseline_status: AgentEvaluationStatus
    candidate_status: AgentEvaluationStatus


class AgentRegressionEvaluator:
    def compare(
        self,
        baseline: AgentEvaluationReport,
        candidate: AgentEvaluationReport,
        *,
        tolerance: float = 0.02,
    ) -> AgentRegressionComparison:
        validate_nonnegative(
            tolerance,
            "tolerance",
        )

        if (
            baseline.overall_score is None
            or candidate.overall_score is None
        ):
            delta = None
            regressed = (
                candidate.status
                == AgentEvaluationStatus.FAILED
                and baseline.status
                != AgentEvaluationStatus.FAILED
            )

        else:
            delta = (
                candidate.overall_score
                - baseline.overall_score
            )

            regressed = (
                delta < -tolerance
                or (
                    candidate.status
                    == AgentEvaluationStatus.FAILED
                    and baseline.status
                    != AgentEvaluationStatus.FAILED
                )
            )

        return AgentRegressionComparison(
            baseline_score=baseline.overall_score,
            candidate_score=candidate.overall_score,
            score_delta=delta,
            regressed=regressed,
            baseline_status=baseline.status,
            candidate_status=candidate.status,
        )


# ============================================================
# EVALUATION HISTORY
# ============================================================

class AgentEvaluationHistory:
    """
    Thread-safe in-memory report history.

    Replace with a persistent database for
    production evaluation monitoring.
    """

    def __init__(self) -> None:
        self._reports: list[
            AgentEvaluationReport
        ] = []

        self._lock = threading.RLock()

    def add(
        self,
        report: AgentEvaluationReport,
    ) -> None:
        with self._lock:
            self._reports.append(
                report
            )

    def list_reports(
        self,
        *,
        agent_id: str | None = None,
    ) -> tuple[AgentEvaluationReport, ...]:
        with self._lock:
            reports = tuple(
                self._reports
            )

        if agent_id is None:
            return reports

        return tuple(
            report
            for report in reports
            if report.agent_id == agent_id
        )

    def average_score(
        self,
        *,
        agent_id: str | None = None,
    ) -> float | None:
        scores = [
            report.overall_score
            for report in self.list_reports(
                agent_id=agent_id
            )
            if report.overall_score is not None
        ]

        return mean_or_none(
            scores
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class AgentEvaluationReportExporter:
    @staticmethod
    def to_dict(
        report: AgentEvaluationReport,
    ) -> dict[str, Any]:
        return asdict(
            report
        )

    @staticmethod
    def to_json(
        report: AgentEvaluationReport,
        *,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            asdict(report),
            ensure_ascii=False,
            indent=indent,
            default=lambda value: (
                value.value
                if isinstance(value, Enum)
                else value.isoformat()
                if isinstance(value, datetime)
                else str(value)
            ),
        )

    @classmethod
    def save_json(
        cls,
        report: AgentEvaluationReport,
        destination: str | Path,
    ) -> Path:
        path = Path(
            destination
        )

        if path.suffix.lower() != ".json":
            raise AgentEvaluationExportError(
                "Report filename must end in .json"
            )

        try:
            with path.open(
                "x",
                encoding="utf-8",
            ) as file:
                file.write(
                    cls.to_json(report)
                )

        except OSError as exc:
            raise AgentEvaluationExportError(
                "Unable to export evaluation report"
            ) from exc

        return path


# ============================================================
# SERVICE LAYER
# ============================================================

class AgentEvaluationService:
    """
    Application-facing agent evaluation service.

    Intended integrations:
        backend/ai/agents/
        backend/ai/orchestration/
        backend/ai/llm/tool_calling.py
        backend/ai/evaluation/hallucination_checks.py
        backend/ai/safety/human_oversight.py
        backend/ai/safety/decision_audit.py
        backend/ai/llm/model_router.py
        backend/mlops/
    """

    def __init__(
        self,
        engine: AgentEvaluationEngine | None = None,
        history: AgentEvaluationHistory | None = None,
    ) -> None:
        self.engine = (
            engine or AgentEvaluationEngine()
        )

        self.history = (
            history or AgentEvaluationHistory()
        )

        self.batch_evaluator = AgentBatchEvaluator(
            self.engine
        )

        self.benchmark_runner = AgentBenchmarkRunner(
            self.engine
        )

        self.regression_evaluator = (
            AgentRegressionEvaluator()
        )

    def evaluate_agent(
        self,
        trace: AgentExecutionTrace,
    ) -> AgentEvaluationReport:
        report = self.engine.evaluate(
            trace
        )

        self.history.add(
            report
        )

        return report

    def evaluate_batch(
        self,
        traces: Sequence[AgentExecutionTrace],
    ) -> tuple[AgentEvaluationReport, ...]:
        reports = self.batch_evaluator.evaluate_batch(
            traces
        )

        for report in reports:
            self.history.add(
                report
            )

        return reports

    def run_benchmarks(
        self,
        cases: Sequence[AgentBenchmarkCase],
    ) -> AgentBenchmarkSuiteResult:
        return self.benchmark_runner.run(
            cases
        )

    def compare_versions(
        self,
        baseline: AgentEvaluationReport,
        candidate: AgentEvaluationReport,
    ) -> AgentRegressionComparison:
        return self.regression_evaluator.compare(
            baseline,
            candidate,
        )

    def agent_average_score(
        self,
        agent_id: str,
    ) -> float | None:
        return self.history.average_score(
            agent_id=agent_id
        )


# ============================================================
# FACTORIES
# ============================================================

def create_agent_evaluation_engine(
    config: AgentEvaluationConfig | None = None,
) -> AgentEvaluationEngine:
    return AgentEvaluationEngine(
        config=config
    )


def create_agent_evaluation_service(
    config: AgentEvaluationConfig | None = None,
) -> AgentEvaluationService:
    return AgentEvaluationService(
        engine=create_agent_evaluation_engine(
            config=config
        )
    )


__all__ = [
    "AgentEvaluationError",
    "AgentEvaluationValidationError",
    "AgentBenchmarkError",
    "AgentEvaluationExportError",
    "AgentType",
    "AgentEvaluationStatus",
    "AgentRiskLevel",
    "AgentMetricName",
    "AgentFindingSeverity",
    "AgentFindingType",
    "AgentEvaluationConfig",
    "AgentToolCall",
    "AgentPlanStep",
    "AgentEvidence",
    "AgentClaim",
    "AgentSafetyEvent",
    "AgentExecutionTrace",
    "AgentMetricResult",
    "AgentEvaluationFinding",
    "AgentEvaluationReport",
    "AgentTaskCompletionEvaluator",
    "AgentToolEvaluator",
    "AgentPlanningEvaluator",
    "AgentWorkflowEvaluator",
    "AgentGroundingEvaluator",
    "AgentCitationEvaluator",
    "AgentSafetyEvaluator",
    "AgentHumanOversightEvaluator",
    "AgentLatencyEvaluator",
    "AgentCostEvaluator",
    "AgentFindingGenerator",
    "AgentScoreCalculator",
    "AgentEvaluationEngine",
    "AgentBatchEvaluator",
    "AgentBenchmarkCase",
    "AgentBenchmarkCaseResult",
    "AgentBenchmarkSuiteResult",
    "AgentBenchmarkRunner",
    "AgentRegressionComparison",
    "AgentRegressionEvaluator",
    "AgentEvaluationHistory",
    "AgentEvaluationReportExporter",
    "AgentEvaluationService",
    "create_agent_evaluation_engine",
    "create_agent_evaluation_service",
]
