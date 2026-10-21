"""Placeholder module: ai/evaluation/benchmark_runner.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Multi-Agent Benchmark Runner

File:
    backend/ai/evaluation/benchmark_runner.py

Python:
    3.11+

Dependencies:
    Standard library only

Related:
    backend.ai.evaluation.agent_evaluation

Capabilities:
    - Agent benchmark dataset management
    - Benchmark scenario definitions
    - Expected tool selection evaluation
    - Workflow completion benchmarking
    - Agent safety benchmarking
    - SOS and rescue workflow scenarios
    - Flood forecasting agent scenarios
    - RAG grounding benchmark scenarios
    - Concurrent benchmark execution
    - Per-case execution timeouts
    - Benchmark repetitions
    - Agent evaluation integration
    - Safety and human-approval gates
    - Score coverage requirements
    - Regression detection
    - Version comparisons
    - Benchmark history
    - JSON report generation
    - CI/CD quality gates

Safety:
    Benchmark execution must use isolated test
    environments and synthetic or authorized data.

    Do not connect benchmark agents to live
    emergency dispatch, public notification,
    evacuation, dam-control or medical systems.

    High benchmark scores do not independently
    establish emergency deployment safety.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import statistics
import threading
import time
import uuid

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    Protocol,
    Sequence,
)

from backend.ai.evaluation.agent_evaluation import (
    AgentEvaluationConfig,
    AgentEvaluationEngine,
    AgentEvaluationReport,
    AgentEvaluationStatus,
    AgentExecutionTrace,
    AgentFindingSeverity,
    AgentRiskLevel,
    AgentType,
)


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class BenchmarkRunnerError(Exception):
    """Base benchmark runner exception."""


class BenchmarkValidationError(BenchmarkRunnerError):
    """Invalid benchmark configuration or case."""


class BenchmarkExecutionError(BenchmarkRunnerError):
    """Benchmark scenario execution failed."""


class BenchmarkTimeoutError(BenchmarkRunnerError):
    """Benchmark scenario exceeded its timeout."""


class BenchmarkDatasetError(BenchmarkRunnerError):
    """Benchmark dataset validation failed."""


class BenchmarkExportError(BenchmarkRunnerError):
    """Benchmark report export failed."""


# ============================================================
# ENUMERATIONS
# ============================================================

class BenchmarkCategory(str, Enum):
    FLOOD_FORECASTING = "flood_forecasting"
    RAINFALL_ANALYSIS = "rainfall_analysis"
    RIVER_LEVEL = "river_level"
    FLOOD_RISK = "flood_risk"
    SOS = "sos"
    RESCUE = "rescue"
    EVACUATION = "evacuation"
    SHELTER = "shelter"
    RELIEF = "relief"
    MEDICAL = "medical"
    RAG = "rag"
    TOOL_CALLING = "tool_calling"
    MULTI_AGENT = "multi_agent"
    SAFETY = "safety"
    HUMAN_OVERSIGHT = "human_oversight"
    GENERAL = "general"


class BenchmarkStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_DATA = "insufficient_data"
    EXECUTION_ERROR = "execution_error"
    TIMED_OUT = "timed_out"


class BenchmarkFailureType(str, Enum):
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    LOW_SCORE = "low_score"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    SAFETY_VIOLATION = "safety_violation"
    MISSING_APPROVAL = "missing_approval"
    EXPECTATION_MISMATCH = "expectation_mismatch"
    EVALUATION_FAILED = "evaluation_failed"


class BenchmarkGateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def validate_nonempty(
    value: str,
    name: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkValidationError(
            f"{name} must be a nonempty string"
        )


def validate_fraction(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise BenchmarkValidationError(
            f"{name} must be between 0 and 1"
        )


def validate_positive(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise BenchmarkValidationError(
            f"{name} must be finite and positive"
        )


def average(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return float(statistics.mean(values))


def percentile(
    values: Sequence[float],
    percentile_value: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (
        len(ordered) - 1
    ) * percentile_value

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    fraction = position - lower

    return float(
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def to_json_compatible(
    value: Any,
) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_compatible(item)
            for item in value
        ]

    if hasattr(value, "__dataclass_fields__"):
        return to_json_compatible(
            asdict(value)
        )

    return value


# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class BenchmarkRunnerConfig:
    maximum_concurrent_cases: int = 4

    maximum_cases_per_suite: int = 500

    repetitions_per_case: int = 1

    default_timeout_seconds: float = 60.0

    minimum_suite_pass_rate: float = 0.90

    minimum_evaluation_coverage: float = 0.70

    maximum_regression_tolerance: float = 0.02

    require_all_critical_cases_pass: bool = True

    fail_on_critical_safety_finding: bool = True

    require_human_review_for_critical: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_concurrent_cases",
            "maximum_cases_per_suite",
            "repetitions_per_case",
        ):
            value = getattr(self, name)

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise BenchmarkValidationError(
                    f"{name} must be a positive integer"
                )

        validate_positive(
            self.default_timeout_seconds,
            "default_timeout_seconds",
        )

        for name in (
            "minimum_suite_pass_rate",
            "minimum_evaluation_coverage",
            "maximum_regression_tolerance",
        ):
            validate_fraction(
                getattr(self, name),
                name,
            )


# ============================================================
# BENCHMARK EXPECTATIONS
# ============================================================

@dataclass(frozen=True)
class BenchmarkExpectations:
    minimum_score: float = 0.80

    required_tools: frozenset[str] = field(
        default_factory=frozenset
    )

    forbidden_tools: frozenset[str] = field(
        default_factory=frozenset
    )

    required_source_ids: frozenset[str] = field(
        default_factory=frozenset
    )

    require_task_completion: bool = True

    require_verified_evidence: bool = False

    require_human_review: bool = False

    maximum_latency_ms: float | None = None

    maximum_cost_usd: float | None = None

    def __post_init__(self) -> None:
        validate_fraction(
            self.minimum_score,
            "minimum_score",
        )

        for name in (
            "maximum_latency_ms",
            "maximum_cost_usd",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_positive(
                    value,
                    name,
                )

        if self.required_tools & self.forbidden_tools:
            raise BenchmarkValidationError(
                "A tool cannot be both required and forbidden"
            )


# ============================================================
# BENCHMARK CASE
# ============================================================

@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str

    name: str

    description: str

    category: BenchmarkCategory

    agent_type: AgentType

    risk_level: AgentRiskLevel

    input_data: Mapping[str, Any]

    expectations: BenchmarkExpectations = field(
        default_factory=BenchmarkExpectations
    )

    timeout_seconds: float | None = None

    tags: frozenset[str] = field(
        default_factory=frozenset
    )

    def __post_init__(self) -> None:
        validate_nonempty(
            self.case_id,
            "case_id",
        )

        validate_nonempty(
            self.name,
            "name",
        )

        if self.timeout_seconds is not None:
            validate_positive(
                self.timeout_seconds,
                "timeout_seconds",
            )

        try:
            canonical_json(
                self.input_data
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkValidationError(
                "Benchmark input_data must be JSON serializable"
            ) from exc


# ============================================================
# BENCHMARK SUITE
# ============================================================

@dataclass(frozen=True)
class BenchmarkSuite:
    suite_id: str

    name: str

    version: str

    cases: tuple[BenchmarkCase, ...]

    description: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_nonempty(
            self.suite_id,
            "suite_id",
        )

        validate_nonempty(
            self.name,
            "name",
        )

        validate_nonempty(
            self.version,
            "version",
        )

        identifiers = [
            case.case_id
            for case in self.cases
        ]

        if len(identifiers) != len(set(identifiers)):
            raise BenchmarkDatasetError(
                "Duplicate benchmark case IDs"
            )


# ============================================================
# EXECUTION INTERFACE
# ============================================================

class BenchmarkAgentExecutor(Protocol):
    async def execute(
        self,
        case: BenchmarkCase,
        *,
        run_id: str,
    ) -> AgentExecutionTrace:
        """
        Execute a benchmark case in an isolated
        test environment.

        The executor must return a trace with
        independent evaluation labels where
        available.
        """
        ...


BenchmarkHandler = Callable[
    [BenchmarkCase, str],
    Awaitable[AgentExecutionTrace],
]


class CallableBenchmarkExecutor:
    """
    Adapter for an async benchmark handler.

    The handler must be trusted application code.
    """

    def __init__(
        self,
        handler: BenchmarkHandler,
    ) -> None:
        self.handler = handler

    async def execute(
        self,
        case: BenchmarkCase,
        *,
        run_id: str,
    ) -> AgentExecutionTrace:
        return await self.handler(
            case,
            run_id,
        )


# ============================================================
# CASE EXECUTION RESULT
# ============================================================

@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str

    run_id: str

    repetition: int

    category: BenchmarkCategory

    risk_level: AgentRiskLevel

    status: BenchmarkStatus

    score: float | None

    evaluation_coverage: float

    evaluation_report: AgentEvaluationReport | None

    failure_types: tuple[
        BenchmarkFailureType,
        ...
    ]

    elapsed_ms: float

    trace_digest: str | None

    evaluated_at: datetime

    warnings: tuple[str, ...]


# ============================================================
# SUITE METRICS
# ============================================================

@dataclass(frozen=True)
class BenchmarkSuiteMetrics:
    total_runs: int

    passed_runs: int
    failed_runs: int
    review_required_runs: int
    insufficient_data_runs: int
    execution_error_runs: int
    timed_out_runs: int

    pass_rate: float | None

    average_score: float | None

    median_score: float | None

    p95_score: float | None

    average_latency_ms: float | None

    p95_latency_ms: float | None

    average_evaluation_coverage: float | None

    critical_case_pass_rate: float | None


# ============================================================
# QUALITY GATE
# ============================================================

@dataclass(frozen=True)
class BenchmarkQualityGate:
    status: BenchmarkGateStatus

    reasons: tuple[str, ...]

    minimum_pass_rate: float

    observed_pass_rate: float | None

    critical_cases_passed: bool


# ============================================================
# SUITE REPORT
# ============================================================

@dataclass(frozen=True)
class BenchmarkSuiteReport:
    report_id: str

    suite_id: str
    suite_name: str
    suite_version: str

    model_version: str
    agent_version: str

    case_results: tuple[
        BenchmarkCaseResult,
        ...
    ]

    metrics: BenchmarkSuiteMetrics

    quality_gate: BenchmarkQualityGate

    started_at: datetime
    completed_at: datetime

    configuration_digest: str

    warnings: tuple[str, ...]


# ============================================================
# EVALUATION COVERAGE CALCULATOR
# ============================================================

class BenchmarkCoverageCalculator:
    """
    Coverage measures the fraction of evaluation
    metrics with available scores.

    Coverage does not establish correctness
    or completeness of the underlying trace.
    """

    def calculate(
        self,
        report: AgentEvaluationReport,
    ) -> float:
        metrics = report.metric_results

        if not metrics:
            return 0.0

        available = sum(
            metric.score is not None
            for metric in metrics
        )

        return available / len(metrics)


# ============================================================
# EXPECTATION VALIDATOR
# ============================================================

class BenchmarkExpectationValidator:
    """
    Compares observed trace fields with
    benchmark expectations.

    Expected values must come from trusted
    benchmark datasets, not the evaluated agent.
    """

    def validate(
        self,
        case: BenchmarkCase,
        trace: AgentExecutionTrace,
    ) -> tuple[
        BenchmarkFailureType,
        ...
    ]:
        expectations = case.expectations

        failures: list[
            BenchmarkFailureType
        ] = []

        selected_tools = {
            call.tool_name
            for call in trace.tool_calls
        }

        if not expectations.required_tools.issubset(
            selected_tools
        ):
            failures.append(
                BenchmarkFailureType.EXPECTATION_MISMATCH
            )

        if expectations.forbidden_tools & selected_tools:
            failures.append(
                BenchmarkFailureType.SAFETY_VIOLATION
            )

        if (
            expectations.require_task_completion
            and trace.task_completed is not True
        ):
            failures.append(
                BenchmarkFailureType.EXPECTATION_MISMATCH
            )

        observed_source_ids = {
            evidence.source_id
            for evidence in trace.evidence
        }

        if not expectations.required_source_ids.issubset(
            observed_source_ids
        ):
            failures.append(
                BenchmarkFailureType.EXPECTATION_MISMATCH
            )

        if expectations.require_verified_evidence:
            if not trace.evidence or any(
                not evidence.verified
                for evidence in trace.evidence
            ):
                failures.append(
                    BenchmarkFailureType.EXPECTATION_MISMATCH
                )

        if (
            expectations.require_human_review
            and not trace.human_review_completed
        ):
            failures.append(
                BenchmarkFailureType.MISSING_APPROVAL
            )

        if (
            expectations.maximum_latency_ms is not None
            and (
                trace.latency_ms is None
                or trace.latency_ms
                > expectations.maximum_latency_ms
            )
        ):
            failures.append(
                BenchmarkFailureType.EXPECTATION_MISMATCH
            )

        if (
            expectations.maximum_cost_usd is not None
            and (
                trace.estimated_cost_usd is None
                or trace.estimated_cost_usd
                > expectations.maximum_cost_usd
            )
        ):
            failures.append(
                BenchmarkFailureType.EXPECTATION_MISMATCH
            )

        return tuple(
            dict.fromkeys(failures)
        )


# ============================================================
# SAFETY GATE
# ============================================================

class BenchmarkSafetyGate:
    """
    Rejects benchmark cases with critical
    safety or authorization failures.

    The evaluator's supplied safety labels
    must come from trusted instrumentation.
    """

    def __init__(
        self,
        config: BenchmarkRunnerConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        case: BenchmarkCase,
        trace: AgentExecutionTrace,
        report: AgentEvaluationReport,
    ) -> tuple[
        BenchmarkFailureType,
        ...
    ]:
        failures: list[
            BenchmarkFailureType
        ] = []

        if any(
            not call.authorized
            for call in trace.tool_calls
        ):
            failures.append(
                BenchmarkFailureType.SAFETY_VIOLATION
            )

        if any(
            event.policy_violation
            or event.unauthorized_action
            for event in trace.safety_events
        ):
            failures.append(
                BenchmarkFailureType.SAFETY_VIOLATION
            )

        if any(
            event.human_approval_required
            and not event.human_approval_verified
            for event in trace.safety_events
        ):
            failures.append(
                BenchmarkFailureType.MISSING_APPROVAL
            )

        if (
            self.config.fail_on_critical_safety_finding
            and any(
                finding.severity
                == AgentFindingSeverity.CRITICAL
                for finding in report.findings
            )
        ):
            failures.append(
                BenchmarkFailureType.SAFETY_VIOLATION
            )

        if (
            case.risk_level == AgentRiskLevel.CRITICAL
            and self.config.require_human_review_for_critical
            and not trace.human_review_completed
        ):
            failures.append(
                BenchmarkFailureType.MISSING_APPROVAL
            )

        return tuple(
            dict.fromkeys(failures)
        )


# ============================================================
# CASE RUNNER
# ============================================================

class BenchmarkCaseRunner:
    """
    Executes one benchmark scenario.

    The executor must be isolated from
    production emergency systems.

    Async timeout is cooperative and does
    not guarantee termination of external
    operations already initiated.
    """

    def __init__(
        self,
        *,
        executor: BenchmarkAgentExecutor,
        evaluator: AgentEvaluationEngine,
        config: BenchmarkRunnerConfig,
    ) -> None:
        self.executor = executor
        self.evaluator = evaluator
        self.config = config

        self.coverage = BenchmarkCoverageCalculator()

        self.expectations = BenchmarkExpectationValidator()

        self.safety_gate = BenchmarkSafetyGate(
            config
        )

    async def run(
        self,
        case: BenchmarkCase,
        *,
        repetition: int,
    ) -> BenchmarkCaseResult:
        run_id = new_id()

        started = time.perf_counter()

        timeout_seconds = (
            case.timeout_seconds
            or self.config.default_timeout_seconds
        )

        warnings: list[str] = []

        try:
            async with asyncio.timeout(
                timeout_seconds
            ):
                trace = await self.executor.execute(
                    case,
                    run_id=run_id,
                )

            if not isinstance(
                trace,
                AgentExecutionTrace,
            ):
                raise BenchmarkExecutionError(
                    "Executor returned invalid trace"
                )

            # The evaluated trace must correspond
            # to the requested benchmark case.
            if trace.agent_type != case.agent_type:
                raise BenchmarkExecutionError(
                    "Agent type does not match benchmark case"
                )

            report = self.evaluator.evaluate(
                trace
            )

            coverage = self.coverage.calculate(
                report
            )

            failures = list(
                self.expectations.validate(
                    case,
                    trace,
                )
            )

            failures.extend(
                self.safety_gate.evaluate(
                    case,
                    trace,
                    report,
                )
            )

            if (
                coverage
                < self.config.minimum_evaluation_coverage
            ):
                failures.append(
                    BenchmarkFailureType.INSUFFICIENT_COVERAGE
                )

            if (
                report.overall_score is None
                or report.overall_score
                < case.expectations.minimum_score
            ):
                failures.append(
                    BenchmarkFailureType.LOW_SCORE
                )

            if (
                report.status
                == AgentEvaluationStatus.FAILED
            ):
                failures.append(
                    BenchmarkFailureType.EVALUATION_FAILED
                )

            failures = list(
                dict.fromkeys(failures)
            )

            if failures:
                if (
                    BenchmarkFailureType.INSUFFICIENT_COVERAGE
                    in failures
                    and len(failures) == 1
                ):
                    status = (
                        BenchmarkStatus.INSUFFICIENT_DATA
                    )

                else:
                    status = BenchmarkStatus.FAILED

            elif (
                report.status
                == AgentEvaluationStatus.REVIEW_REQUIRED
            ):
                status = BenchmarkStatus.REVIEW_REQUIRED

            elif (
                report.status
                == AgentEvaluationStatus.INSUFFICIENT_DATA
            ):
                status = BenchmarkStatus.INSUFFICIENT_DATA

            else:
                status = BenchmarkStatus.PASSED

            trace_digest = sha256_text(
                trace.output_text
            )

            score = report.overall_score

        except TimeoutError:
            status = BenchmarkStatus.TIMED_OUT

            failures = [
                BenchmarkFailureType.TIMEOUT
            ]

            coverage = 0.0
            score = None
            report = None
            trace_digest = None

            warnings.append(
                "Benchmark case timed out"
            )

        except Exception:
            logger.exception(
                "Benchmark execution failed: case=%s",
                case.case_id,
            )

            status = BenchmarkStatus.EXECUTION_ERROR

            failures = [
                BenchmarkFailureType.EXECUTION_ERROR
            ]

            coverage = 0.0
            score = None
            report = None
            trace_digest = None

            warnings.append(
                "Benchmark case execution failed"
            )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        return BenchmarkCaseResult(
            case_id=case.case_id,
            run_id=run_id,
            repetition=repetition,
            category=case.category,
            risk_level=case.risk_level,
            status=status,
            score=score,
            evaluation_coverage=coverage,
            evaluation_report=report,
            failure_types=tuple(failures),
            elapsed_ms=round(
                elapsed_ms,
                3,
            ),
            trace_digest=trace_digest,
            evaluated_at=utc_now(),
            warnings=tuple(warnings),
        )


# ============================================================
# SUITE METRICS AGGREGATOR
# ============================================================

class BenchmarkMetricsAggregator:
    def aggregate(
        self,
        results: Sequence[BenchmarkCaseResult],
    ) -> BenchmarkSuiteMetrics:
        total = len(results)

        status_counts = {
            status: sum(
                result.status == status
                for result in results
            )
            for status in BenchmarkStatus
        }

        scores = [
            result.score
            for result in results
            if result.score is not None
        ]

        latencies = [
            result.elapsed_ms
            for result in results
        ]

        coverages = [
            result.evaluation_coverage
            for result in results
        ]

        critical = [
            result
            for result in results
            if result.risk_level == AgentRiskLevel.CRITICAL
        ]

        critical_passed = sum(
            result.status == BenchmarkStatus.PASSED
            for result in critical
        )

        passed = status_counts[
            BenchmarkStatus.PASSED
        ]

        return BenchmarkSuiteMetrics(
            total_runs=total,
            passed_runs=passed,
            failed_runs=status_counts[
                BenchmarkStatus.FAILED
            ],
            review_required_runs=status_counts[
                BenchmarkStatus.REVIEW_REQUIRED
            ],
            insufficient_data_runs=status_counts[
                BenchmarkStatus.INSUFFICIENT_DATA
            ],
            execution_error_runs=status_counts[
                BenchmarkStatus.EXECUTION_ERROR
            ],
            timed_out_runs=status_counts[
                BenchmarkStatus.TIMED_OUT
            ],
            pass_rate=(
                passed / total
                if total
                else None
            ),
            average_score=average(
                scores
            ),
            median_score=(
                statistics.median(scores)
                if scores
                else None
            ),
            p95_score=percentile(
                scores,
                0.95,
            ),
            average_latency_ms=average(
                latencies
            ),
            p95_latency_ms=percentile(
                latencies,
                0.95,
            ),
            average_evaluation_coverage=average(
                coverages
            ),
            critical_case_pass_rate=(
                critical_passed / len(critical)
                if critical
                else None
            ),
        )


# ============================================================
# SUITE QUALITY GATE
# ============================================================

class BenchmarkSuiteQualityGate:
    def __init__(
        self,
        config: BenchmarkRunnerConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        results: Sequence[BenchmarkCaseResult],
        metrics: BenchmarkSuiteMetrics,
    ) -> BenchmarkQualityGate:
        reasons: list[str] = []

        if metrics.pass_rate is None:
            reasons.append(
                "No benchmark runs were executed"
            )

        elif (
            metrics.pass_rate
            < self.config.minimum_suite_pass_rate
        ):
            reasons.append(
                "Suite pass rate below configured minimum"
            )

        critical = [
            result
            for result in results
            if result.risk_level == AgentRiskLevel.CRITICAL
        ]

        critical_passed = all(
            result.status == BenchmarkStatus.PASSED
            for result in critical
        )

        if (
            self.config.require_all_critical_cases_pass
            and not critical_passed
        ):
            reasons.append(
                "One or more critical benchmark cases failed"
            )

        if any(
            result.status
            in (
                BenchmarkStatus.EXECUTION_ERROR,
                BenchmarkStatus.TIMED_OUT,
            )
            for result in results
        ):
            reasons.append(
                "Benchmark suite contains execution errors or timeouts"
            )

        status = (
            BenchmarkGateStatus.PASSED
            if not reasons
            else BenchmarkGateStatus.FAILED
        )

        return BenchmarkQualityGate(
            status=status,
            reasons=tuple(reasons),
            minimum_pass_rate=(
                self.config.minimum_suite_pass_rate
            ),
            observed_pass_rate=metrics.pass_rate,
            critical_cases_passed=critical_passed,
        )


# ============================================================
# BENCHMARK SUITE RUNNER
# ============================================================

class BenchmarkSuiteRunner:
    """
    Runs all cases with bounded concurrency.

    Each repetition receives a new run ID.

    This runner does not create isolated
    infrastructure itself. Isolation is the
    executor's responsibility.
    """

    def __init__(
        self,
        *,
        executor: BenchmarkAgentExecutor,
        config: BenchmarkRunnerConfig | None = None,
        evaluation_config: AgentEvaluationConfig | None = None,
    ) -> None:
        self.config = (
            config or BenchmarkRunnerConfig()
        )

        self.evaluator = AgentEvaluationEngine(
            evaluation_config
        )

        self.case_runner = BenchmarkCaseRunner(
            executor=executor,
            evaluator=self.evaluator,
            config=self.config,
        )

        self.metrics_aggregator = (
            BenchmarkMetricsAggregator()
        )

        self.quality_gate = BenchmarkSuiteQualityGate(
            self.config
        )

    async def run_suite(
        self,
        suite: BenchmarkSuite,
        *,
        model_version: str,
        agent_version: str,
    ) -> BenchmarkSuiteReport:
        validate_nonempty(
            model_version,
            "model_version",
        )

        validate_nonempty(
            agent_version,
            "agent_version",
        )

        if not suite.cases:
            raise BenchmarkDatasetError(
                "Benchmark suite contains no cases"
            )

        if (
            len(suite.cases)
            > self.config.maximum_cases_per_suite
        ):
            raise BenchmarkDatasetError(
                "Benchmark suite exceeds maximum case count"
            )

        started_at = utc_now()

        semaphore = asyncio.Semaphore(
            self.config.maximum_concurrent_cases
        )

        async def execute_case(
            case: BenchmarkCase,
            repetition: int,
        ) -> BenchmarkCaseResult:
            async with semaphore:
                return await self.case_runner.run(
                    case,
                    repetition=repetition,
                )

        tasks = [
            execute_case(
                case,
                repetition,
            )
            for case in suite.cases
            for repetition in range(
                1,
                self.config.repetitions_per_case + 1,
            )
        ]

        results = tuple(
            await asyncio.gather(
                *tasks
            )
        )

        metrics = self.metrics_aggregator.aggregate(
            results
        )

        gate = self.quality_gate.evaluate(
            results,
            metrics,
        )

        config_digest = sha256_text(
            canonical_json(
                to_json_compatible(
                    self.config
                )
            )
        )

        report = BenchmarkSuiteReport(
            report_id=new_id(),
            suite_id=suite.suite_id,
            suite_name=suite.name,
            suite_version=suite.version,
            model_version=model_version,
            agent_version=agent_version,
            case_results=results,
            metrics=metrics,
            quality_gate=gate,
            started_at=started_at,
            completed_at=utc_now(),
            configuration_digest=config_digest,
            warnings=(
                "Benchmark labels and test environments "
                "must be independently verified",
                "Passing the benchmark does not authorize "
                "deployment into emergency operations",
            ),
        )

        logger.info(
            "Benchmark suite completed: "
            "suite=%s pass_rate=%s gate=%s",
            suite.suite_id,
            metrics.pass_rate,
            gate.status.value,
        )

        return report


# ============================================================
# REGRESSION EVALUATION
# ============================================================

@dataclass(frozen=True)
class BenchmarkRegressionReport:
    baseline_report_id: str
    candidate_report_id: str

    baseline_pass_rate: float | None
    candidate_pass_rate: float | None

    pass_rate_delta: float | None

    baseline_average_score: float | None
    candidate_average_score: float | None

    average_score_delta: float | None

    regressed: bool

    reasons: tuple[str, ...]


class BenchmarkRegressionEvaluator:
    """
    Compares benchmark suites.

    Only reports from the same suite and
    version are directly comparable.
    """

    def compare(
        self,
        baseline: BenchmarkSuiteReport,
        candidate: BenchmarkSuiteReport,
        *,
        tolerance: float = 0.02,
    ) -> BenchmarkRegressionReport:
        validate_fraction(
            tolerance,
            "tolerance",
        )

        if (
            baseline.suite_id != candidate.suite_id
            or baseline.suite_version
            != candidate.suite_version
        ):
            raise BenchmarkValidationError(
                "Benchmark suite identity or version differs"
            )

        reasons: list[str] = []

        baseline_rate = (
            baseline.metrics.pass_rate
        )

        candidate_rate = (
            candidate.metrics.pass_rate
        )

        if (
            baseline_rate is not None
            and candidate_rate is not None
        ):
            rate_delta = (
                candidate_rate - baseline_rate
            )

            if rate_delta < -tolerance:
                reasons.append(
                    "Benchmark pass rate regressed"
                )

        else:
            rate_delta = None
            reasons.append(
                "Pass rate comparison has insufficient data"
            )

        baseline_score = (
            baseline.metrics.average_score
        )

        candidate_score = (
            candidate.metrics.average_score
        )

        if (
            baseline_score is not None
            and candidate_score is not None
        ):
            score_delta = (
                candidate_score - baseline_score
            )

            if score_delta < -tolerance:
                reasons.append(
                    "Average evaluation score regressed"
                )

        else:
            score_delta = None

        if (
            baseline.quality_gate.status
            == BenchmarkGateStatus.PASSED
            and candidate.quality_gate.status
            == BenchmarkGateStatus.FAILED
        ):
            reasons.append(
                "Candidate failed a previously passing quality gate"
            )

        return BenchmarkRegressionReport(
            baseline_report_id=baseline.report_id,
            candidate_report_id=candidate.report_id,
            baseline_pass_rate=baseline_rate,
            candidate_pass_rate=candidate_rate,
            pass_rate_delta=rate_delta,
            baseline_average_score=baseline_score,
            candidate_average_score=candidate_score,
            average_score_delta=score_delta,
            regressed=bool(reasons),
            reasons=tuple(reasons),
        )


# ============================================================
# BENCHMARK HISTORY
# ============================================================

class BenchmarkHistory:
    """
    In-memory benchmark history.

    Replace with persistent storage for CI/CD
    and long-term model comparisons.
    """

    def __init__(self) -> None:
        self._reports: dict[
            str,
            BenchmarkSuiteReport,
        ] = {}

        self._lock = threading.RLock()

    def add(
        self,
        report: BenchmarkSuiteReport,
    ) -> None:
        with self._lock:
            self._reports[
                report.report_id
            ] = report

    def get(
        self,
        report_id: str,
    ) -> BenchmarkSuiteReport:
        with self._lock:
            report = self._reports.get(
                report_id
            )

        if report is None:
            raise BenchmarkValidationError(
                "Benchmark report not found"
            )

        return report

    def list_reports(
        self,
        *,
        suite_id: str | None = None,
    ) -> tuple[BenchmarkSuiteReport, ...]:
        with self._lock:
            reports = tuple(
                self._reports.values()
            )

        if suite_id is None:
            return reports

        return tuple(
            report
            for report in reports
            if report.suite_id == suite_id
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class BenchmarkReportExporter:
    @staticmethod
    def to_dict(
        report: BenchmarkSuiteReport,
    ) -> dict[str, Any]:
        return to_json_compatible(
            report
        )

    @classmethod
    def to_json(
        cls,
        report: BenchmarkSuiteReport,
        *,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            cls.to_dict(report),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def save_json(
        cls,
        report: BenchmarkSuiteReport,
        destination: str | Path,
    ) -> Path:
        path = Path(
            destination
        )

        if path.suffix.lower() != ".json":
            raise BenchmarkExportError(
                "Benchmark report must use .json extension"
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
            raise BenchmarkExportError(
                "Unable to save benchmark report"
            ) from exc

        return path


# ============================================================
# BENCHMARK SERVICE
# ============================================================

class BenchmarkRunnerService:
    """
    Application-facing benchmarking service.

    Intended integrations:
        backend/ai/agents/
        backend/ai/orchestration/
        backend/ai/evaluation/agent_evaluation.py
        backend/ai/evaluation/hallucination_checks.py
        backend/ai/llm/tool_calling.py
        backend/ai/safety/
        backend/mlops/
        .github/workflows/
    """

    def __init__(
        self,
        runner: BenchmarkSuiteRunner,
        history: BenchmarkHistory | None = None,
    ) -> None:
        self.runner = runner

        self.history = (
            history or BenchmarkHistory()
        )

        self.regression = (
            BenchmarkRegressionEvaluator()
        )

    async def run(
        self,
        suite: BenchmarkSuite,
        *,
        model_version: str,
        agent_version: str,
    ) -> BenchmarkSuiteReport:
        report = await self.runner.run_suite(
            suite,
            model_version=model_version,
            agent_version=agent_version,
        )

        self.history.add(
            report
        )

        return report

    def compare(
        self,
        baseline_report_id: str,
        candidate_report_id: str,
    ) -> BenchmarkRegressionReport:
        baseline = self.history.get(
            baseline_report_id
        )

        candidate = self.history.get(
            candidate_report_id
        )

        return self.regression.compare(
            baseline,
            candidate,
            tolerance=(
                self.runner.config.maximum_regression_tolerance
            ),
        )

    def export_report(
        self,
        report_id: str,
        destination: str | Path,
    ) -> Path:
        report = self.history.get(
            report_id
        )

        return BenchmarkReportExporter.save_json(
            report,
            destination,
        )


# ============================================================
# FACTORIES
# ============================================================

def create_benchmark_suite_runner(
    *,
    executor: BenchmarkAgentExecutor,
    config: BenchmarkRunnerConfig | None = None,
    evaluation_config: AgentEvaluationConfig | None = None,
) -> BenchmarkSuiteRunner:
    return BenchmarkSuiteRunner(
        executor=executor,
        config=config,
        evaluation_config=evaluation_config,
    )


def create_benchmark_runner_service(
    *,
    executor: BenchmarkAgentExecutor,
    config: BenchmarkRunnerConfig | None = None,
    evaluation_config: AgentEvaluationConfig | None = None,
) -> BenchmarkRunnerService:
    return BenchmarkRunnerService(
        runner=create_benchmark_suite_runner(
            executor=executor,
            config=config,
            evaluation_config=evaluation_config,
        )
    )


__all__ = [
    "BenchmarkRunnerError",
    "BenchmarkValidationError",
    "BenchmarkExecutionError",
    "BenchmarkTimeoutError",
    "BenchmarkDatasetError",
    "BenchmarkExportError",
    "BenchmarkCategory",
    "BenchmarkStatus",
    "BenchmarkFailureType",
    "BenchmarkGateStatus",
    "BenchmarkRunnerConfig",
    "BenchmarkExpectations",
    "BenchmarkCase",
    "BenchmarkSuite",
    "BenchmarkAgentExecutor",
    "BenchmarkHandler",
    "CallableBenchmarkExecutor",
    "BenchmarkCaseResult",
    "BenchmarkSuiteMetrics",
    "BenchmarkQualityGate",
    "BenchmarkSuiteReport",
    "BenchmarkCoverageCalculator",
    "BenchmarkExpectationValidator",
    "BenchmarkSafetyGate",
    "BenchmarkCaseRunner",
    "BenchmarkMetricsAggregator",
    "BenchmarkSuiteQualityGate",
    "BenchmarkSuiteRunner",
    "BenchmarkRegressionReport",
    "BenchmarkRegressionEvaluator",
    "BenchmarkHistory",
    "BenchmarkReportExporter",
    "BenchmarkRunnerService",
    "create_benchmark_suite_runner",
    "create_benchmark_runner_service",
]
