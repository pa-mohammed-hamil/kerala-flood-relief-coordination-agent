"""Placeholder module: ai/agents/supervisor_agent.py; implementation required."""

"""
backend/ai/agents/supervisor_agent.py

Kerala Flood Relief Coordination Agent
Multi-Agent Supervisor & Workflow Orchestration

Python 3.11+
Standard library only.

FEATURES
--------
1. Central multi-agent supervisor
2. Agent registration and capability routing
3. Flood-response task classification
4. Dependency-aware workflow execution
5. Task priority scheduling
6. Agent health and availability checks
7. Bounded retries and failure handling
8. Timeouts and circuit-breaker protection
9. Human escalation for critical decisions
10. Agent result validation
11. Workflow state tracking
12. Audit logging
13. JSON report export

SAFETY
------
All outputs are advisory.

This supervisor does not automatically:
- Dispatch rescue teams or ambulances
- Issue public flood alerts
- Order evacuations
- Move people into shelters
- Execute vehicle or infrastructure controls
- Approve medical decisions
- Authorize spending or procurement

Human approval is required for operational
decisions. An approved advisory workflow
does not itself execute any physical action.

Registered agents are local Python callables.
Real integrations require explicit adapters,
authentication, authorization and monitoring.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import threading
import time
import uuid

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class SupervisorError(Exception):
    pass


class SupervisorValidationError(SupervisorError):
    pass


class AgentRegistrationError(SupervisorError):
    pass


class AgentExecutionError(SupervisorError):
    pass


class WorkflowNotFoundError(SupervisorError):
    pass


class SupervisorApprovalError(SupervisorError):
    pass


class SupervisorExportError(SupervisorError):
    pass


# ============================================================
# ENUMS
# ============================================================

class AgentCapability(str, Enum):
    FLOOD_PREDICTION = "flood_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    ALERT_INTELLIGENCE = "alert_intelligence"
    COMMUNICATION = "communication"

    SOS_TRIAGE = "sos_triage"
    RESCUE_COORDINATION = "rescue_coordination"
    ROUTE_PLANNING = "route_planning"
    EVACUATION = "evacuation"
    SHELTER_ALLOCATION = "shelter_allocation"
    MEDICAL_RESPONSE = "medical_response"

    RELIEF_LOGISTICS = "relief_logistics"
    MISSING_PERSON = "missing_person"
    DAMAGE_ASSESSMENT = "damage_assessment"
    RECOVERY_PLANNING = "recovery_planning"

    HUMAN_APPROVAL = "human_approval"


class TaskPriority(str, Enum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED_PENDING_REVIEW = (
        "completed_pending_review"
    )
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    APPROVED_FOR_COORDINATION = (
        "approved_for_coordination"
    )
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"


class AgentHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ReviewDecision(str, Enum):
    APPROVE_FOR_COORDINATION = (
        "approve_for_coordination"
    )
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_datetime(value: datetime) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise SupervisorValidationError(
            "Timezone-aware datetime required"
        )


def validate_positive_int(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise SupervisorValidationError(
            f"{name} must be a positive integer"
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, float) and math.isfinite(value):
        return value

    raise SupervisorValidationError(
        f"Non-JSON-safe value: {type(value).__name__}"
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SupervisorConfig:
    max_agents: int = 100
    max_tasks_per_workflow: int = 100

    max_retries: int = 2
    task_timeout_seconds: float = 30.0

    circuit_breaker_failure_threshold: int = 3

    require_human_approval: bool = True
    allow_automatic_physical_actions: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_agents",
            "max_tasks_per_workflow",
            "circuit_breaker_failure_threshold",
        ):
            validate_positive_int(
                getattr(self, name),
                name,
            )

        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise SupervisorValidationError(
                "max_retries must be nonnegative integer"
            )

        if (
            not math.isfinite(self.task_timeout_seconds)
            or self.task_timeout_seconds <= 0
        ):
            raise SupervisorValidationError(
                "task_timeout_seconds must be positive"
            )

        if self.allow_automatic_physical_actions:
            raise SupervisorValidationError(
                "Automatic physical actions are prohibited"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class AgentDescriptor:
    agent_id: str
    agent_name: str

    capabilities: tuple[AgentCapability, ...]

    health: AgentHealth = AgentHealth.HEALTHY

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise SupervisorValidationError(
                "agent_id required"
            )

        if not self.agent_name.strip():
            raise SupervisorValidationError(
                "agent_name required"
            )

        if not self.capabilities:
            raise SupervisorValidationError(
                "At least one capability required"
            )


@dataclass(frozen=True)
class SupervisorTask:
    task_id: str
    capability: AgentCapability

    priority: TaskPriority

    payload: Mapping[str, Any]

    dependencies: tuple[str, ...] = ()

    requires_human_review: bool = True

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise SupervisorValidationError(
                "task_id required"
            )

        if self.task_id in self.dependencies:
            raise SupervisorValidationError(
                "Task cannot depend on itself"
            )


@dataclass(frozen=True)
class WorkflowRequest:
    workflow_id: str

    district: str
    incident_reference: str

    tasks: tuple[SupervisorTask, ...]

    created_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "workflow_id",
            "district",
            "incident_reference",
        ):
            if not getattr(self, name).strip():
                raise SupervisorValidationError(
                    f"{name} required"
                )

        validate_datetime(
            self.created_at
        )


@dataclass(frozen=True)
class AgentExecutionResult:
    task_id: str
    agent_id: str

    status: TaskStatus

    output: Mapping[str, Any]

    attempts: int
    duration_seconds: float

    warnings: tuple[str, ...] = ()

    error: str | None = None


@dataclass(frozen=True)
class SupervisorWorkflowResult:
    workflow_id: str

    district: str
    incident_reference: str

    task_results: tuple[AgentExecutionResult, ...]

    status: WorkflowStatus

    human_approval_required: bool

    warnings: tuple[str, ...]

    started_at: datetime
    completed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class SupervisorReviewRecord:
    review_id: str
    workflow_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ReviewDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class SupervisorAuditRecord:
    audit_id: str
    workflow_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# AGENT REGISTRY
# ============================================================

AgentHandler = Callable[
    [Mapping[str, Any]],
    Mapping[str, Any],
]


@dataclass
class RegisteredAgent:
    descriptor: AgentDescriptor
    handler: AgentHandler

    consecutive_failures: int = 0


class SupervisorAgentRegistry:
    def __init__(
        self,
        config: SupervisorConfig,
    ) -> None:
        self.config = config

        self._agents: dict[
            str,
            RegisteredAgent,
        ] = {}

        self._lock = threading.RLock()

    def register(
        self,
        descriptor: AgentDescriptor,
        handler: AgentHandler,
    ) -> None:
        if not callable(handler):
            raise AgentRegistrationError(
                "Agent handler must be callable"
            )

        with self._lock:
            if descriptor.agent_id in self._agents:
                raise AgentRegistrationError(
                    "Agent ID already registered"
                )

            if len(self._agents) >= self.config.max_agents:
                raise AgentRegistrationError(
                    "Agent registry capacity exceeded"
                )

            self._agents[
                descriptor.agent_id
            ] = RegisteredAgent(
                descriptor=descriptor,
                handler=handler,
            )

    def select(
        self,
        capability: AgentCapability,
    ) -> RegisteredAgent | None:
        with self._lock:
            candidates = [
                agent
                for agent in self._agents.values()
                if (
                    capability in agent.descriptor.capabilities
                    and agent.descriptor.health
                    == AgentHealth.HEALTHY
                    and agent.consecutive_failures
                    < self.config.circuit_breaker_failure_threshold
                )
            ]

            if not candidates:
                return None

            candidates.sort(
                key=lambda agent: (
                    agent.consecutive_failures,
                    agent.descriptor.agent_id,
                )
            )

            return candidates[0]

    def record_success(
        self,
        agent_id: str,
    ) -> None:
        with self._lock:
            agent = self._agents.get(
                agent_id
            )

            if agent:
                agent.consecutive_failures = 0

    def record_failure(
        self,
        agent_id: str,
    ) -> None:
        with self._lock:
            agent = self._agents.get(
                agent_id
            )

            if agent:
                agent.consecutive_failures += 1

    def list_agents(
        self,
    ) -> tuple[AgentDescriptor, ...]:
        with self._lock:
            return tuple(
                agent.descriptor
                for agent in self._agents.values()
            )


# ============================================================
# WORKFLOW VALIDATOR
# ============================================================

class SupervisorWorkflowValidator:
    def __init__(
        self,
        config: SupervisorConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        request: WorkflowRequest,
    ) -> None:
        if len(request.tasks) > (
            self.config.max_tasks_per_workflow
        ):
            raise SupervisorValidationError(
                "Workflow task limit exceeded"
            )

        task_ids = [
            task.task_id
            for task in request.tasks
        ]

        if len(task_ids) != len(set(task_ids)):
            raise SupervisorValidationError(
                "Duplicate task ID"
            )

        task_set = set(
            task_ids
        )

        for task in request.tasks:
            missing = set(
                task.dependencies
            ) - task_set

            if missing:
                raise SupervisorValidationError(
                    f"Unknown dependencies for {task.task_id}: "
                    f"{sorted(missing)}"
                )

            json_safe(
                task.payload
            )

        self._validate_acyclic(
            request.tasks
        )

    @staticmethod
    def _validate_acyclic(
        tasks: Sequence[SupervisorTask],
    ) -> None:
        dependency_map = {
            task.task_id: set(task.dependencies)
            for task in tasks
        }

        completed = set()

        while len(completed) < len(tasks):
            ready = [
                task_id
                for task_id, dependencies
                in dependency_map.items()
                if (
                    task_id not in completed
                    and dependencies.issubset(completed)
                )
            ]

            if not ready:
                raise SupervisorValidationError(
                    "Workflow contains dependency cycle"
                )

            completed.update(
                ready
            )


# ============================================================
# TASK SCHEDULER
# ============================================================

class SupervisorTaskScheduler:
    PRIORITY_ORDER = {
        TaskPriority.CRITICAL: 0,
        TaskPriority.HIGH: 1,
        TaskPriority.ELEVATED: 2,
        TaskPriority.ROUTINE: 3,
    }

    @classmethod
    def order(
        cls,
        tasks: Sequence[SupervisorTask],
    ) -> tuple[SupervisorTask, ...]:
        remaining = {
            task.task_id: task
            for task in tasks
        }

        completed = set()
        ordered = []

        while remaining:
            ready = [
                task
                for task in remaining.values()
                if set(
                    task.dependencies
                ).issubset(completed)
            ]

            if not ready:
                raise SupervisorValidationError(
                    "Unresolvable workflow dependencies"
                )

            ready.sort(
                key=lambda task: (
                    cls.PRIORITY_ORDER[
                        task.priority
                    ],
                    task.task_id,
                )
            )

            selected = ready[0]

            ordered.append(
                selected
            )

            completed.add(
                selected.task_id
            )

            del remaining[
                selected.task_id
            ]

        return tuple(
            ordered
        )


# ============================================================
# AGENT EXECUTION ENGINE
# ============================================================

class SupervisorExecutionEngine:
    """
    Executes local agent handlers.

    Important limitation:
    Python thread timeouts cannot forcibly stop
    a handler already running. Handlers must be
    side-effect-free and must implement their
    own cancellation/deadline checks.

    For production isolation, use supervised
    worker processes or external task queues.
    """

    def __init__(
        self,
        config: SupervisorConfig,
        registry: SupervisorAgentRegistry,
    ) -> None:
        self.config = config
        self.registry = registry

    def execute(
        self,
        task: SupervisorTask,
    ) -> AgentExecutionResult:
        agent = self.registry.select(
            task.capability
        )

        if agent is None:
            return AgentExecutionResult(
                task_id=task.task_id,
                agent_id="unassigned",
                status=TaskStatus.FAILED,
                output={},
                attempts=0,
                duration_seconds=0.0,
                warnings=(
                    "No healthy registered agent supports capability",
                    "Manual coordination required",
                ),
                error="No eligible agent",
            )

        started = time.monotonic()

        last_error = None
        attempts = 0

        for attempt in range(
            self.config.max_retries + 1
        ):
            attempts = attempt + 1

            executor = (
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=1
                )
            )

            future = executor.submit(
                agent.handler,
                dict(task.payload),
            )

            try:
                output = future.result(
                    timeout=(
                        self.config.task_timeout_seconds
                    )
                )

                if not isinstance(output, Mapping):
                    raise AgentExecutionError(
                        "Agent output must be a mapping"
                    )

                normalized_output = json_safe(
                    output
                )

                self.registry.record_success(
                    agent.descriptor.agent_id
                )

                return AgentExecutionResult(
                    task_id=task.task_id,
                    agent_id=agent.descriptor.agent_id,
                    status=TaskStatus.COMPLETED,
                    output=normalized_output,
                    attempts=attempts,
                    duration_seconds=round(
                        time.monotonic() - started,
                        4,
                    ),
                    warnings=(
                        "Agent output is advisory only",
                        "Human verification required",
                    ),
                )

            except concurrent.futures.TimeoutError:
                last_error = (
                    "Agent execution timed out"
                )

                future.cancel()

                self.registry.record_failure(
                    agent.descriptor.agent_id
                )

                # A running thread may continue.
                # Do not retry a potentially live
                # timed-out handler.
                break

            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: {exc}"
                )

                self.registry.record_failure(
                    agent.descriptor.agent_id
                )

            finally:
                executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )

        return AgentExecutionResult(
            task_id=task.task_id,
            agent_id=agent.descriptor.agent_id,
            status=TaskStatus.FAILED,
            output={},
            attempts=attempts,
            duration_seconds=round(
                time.monotonic() - started,
                4,
            ),
            warnings=(
                "Agent execution failed",
                "Manual review required",
                "Timed-out handlers may still be running",
            ),
            error=last_error,
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class SupervisorAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            SupervisorAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        workflow_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> SupervisorAuditRecord:
        record = SupervisorAuditRecord(
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
    ) -> tuple[SupervisorAuditRecord, ...]:
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
# HUMAN APPROVAL MANAGER
# ============================================================

class SupervisorHumanApprovalManager:
    ALLOWED_ROLES = {
        "incident_commander",
        "district_disaster_management_officer",
        "authorized_emergency_coordinator",
    }

    def __init__(
        self,
        audit_logger: SupervisorAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            SupervisorReviewRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        workflow: SupervisorWorkflowResult,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> SupervisorReviewRecord:
        if not reviewer_id.strip():
            raise SupervisorApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise SupervisorApprovalError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise SupervisorApprovalError(
                "Review reason required"
            )

        if (
            decision
            == ReviewDecision.APPROVE_FOR_COORDINATION
            and workflow.status
            != WorkflowStatus.COMPLETED_PENDING_REVIEW
        ):
            raise SupervisorApprovalError(
                "Only fully completed advisory workflows "
                "may be approved for coordination"
            )

        with self._lock:
            if workflow.workflow_id in self._reviews:
                raise SupervisorApprovalError(
                    "Workflow already reviewed"
                )

            record = SupervisorReviewRecord(
                review_id=generate_id(),
                workflow_id=workflow.workflow_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                workflow.workflow_id
            ] = record

        self.audit_logger.record(
            workflow_id=workflow.workflow_id,
            actor_id=reviewer_id,
            action=f"workflow_review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return record


# ============================================================
# MAIN SUPERVISOR AGENT
# ============================================================

class SupervisorAgent:
    def __init__(
        self,
        config: SupervisorConfig | None = None,
    ) -> None:
        self.config = (
            config or SupervisorConfig()
        )

        self.registry = (
            SupervisorAgentRegistry(
                self.config
            )
        )

        self.validator = (
            SupervisorWorkflowValidator(
                self.config
            )
        )

        self.scheduler = (
            SupervisorTaskScheduler()
        )

        self.execution_engine = (
            SupervisorExecutionEngine(
                self.config,
                self.registry,
            )
        )

        self.audit_logger = (
            SupervisorAuditLogger()
        )

        self.approval_manager = (
            SupervisorHumanApprovalManager(
                self.audit_logger
            )
        )

        self._workflows: dict[
            str,
            SupervisorWorkflowResult,
        ] = {}

        self._lock = threading.RLock()

    def register_agent(
        self,
        descriptor: AgentDescriptor,
        handler: AgentHandler,
    ) -> None:
        self.registry.register(
            descriptor,
            handler,
        )

    def run(
        self,
        request: WorkflowRequest,
    ) -> SupervisorWorkflowResult:
        self.validator.validate(
            request
        )

        started = utc_now()

        if request.created_at > started:
            raise SupervisorValidationError(
                "Workflow timestamp is in the future"
            )

        with self._lock:
            if request.workflow_id in self._workflows:
                raise SupervisorValidationError(
                    "Workflow ID already exists"
                )

            # Reserve ID before execution.
            self._workflows[
                request.workflow_id
            ] = SupervisorWorkflowResult(
                workflow_id=request.workflow_id,
                district=request.district,
                incident_reference=request.incident_reference,
                task_results=(),
                status=WorkflowStatus.RUNNING,
                human_approval_required=True,
                warnings=(),
                started_at=started,
                completed_at=started,
            )

        ordered_tasks = self.scheduler.order(
            request.tasks
        )

        results = []
        result_by_task = {}

        for task in ordered_tasks:
            dependencies_succeeded = all(
                result_by_task[dependency].status
                == TaskStatus.COMPLETED
                for dependency in task.dependencies
            )

            if not dependencies_succeeded:
                result = AgentExecutionResult(
                    task_id=task.task_id,
                    agent_id="unassigned",
                    status=TaskStatus.BLOCKED,
                    output={},
                    attempts=0,
                    duration_seconds=0.0,
                    warnings=(
                        "Dependency did not complete successfully",
                    ),
                    error="Blocked by dependency",
                )

            else:
                result = (
                    self.execution_engine.execute(
                        task
                    )
                )

            results.append(
                result
            )

            result_by_task[
                task.task_id
            ] = result

            self.audit_logger.record(
                workflow_id=request.workflow_id,
                actor_id=(
                    result.agent_id
                ),
                action=f"task_{result.status.value}",
                details={
                    "task_id": task.task_id,
                    "capability": task.capability.value,
                    "attempts": str(result.attempts),
                },
            )

        completed_count = sum(
            result.status == TaskStatus.COMPLETED
            for result in results
        )

        failed_count = sum(
            result.status == TaskStatus.FAILED
            for result in results
        )

        blocked_count = sum(
            result.status == TaskStatus.BLOCKED
            for result in results
        )

        if completed_count == len(results):
            workflow_status = (
                WorkflowStatus.COMPLETED_PENDING_REVIEW
            )

        elif completed_count > 0:
            workflow_status = (
                WorkflowStatus.PARTIALLY_COMPLETED
            )

        else:
            workflow_status = WorkflowStatus.FAILED

        warnings = [
            "Workflow outputs are advisory",
            "No physical action has been executed",
            "Human approval is mandatory",
        ]

        if failed_count or blocked_count:
            warnings.append(
                "Some agent tasks failed or were blocked"
            )

        workflow = SupervisorWorkflowResult(
            workflow_id=request.workflow_id,
            district=request.district,
            incident_reference=request.incident_reference,
            task_results=tuple(results),
            status=workflow_status,
            human_approval_required=True,
            warnings=tuple(warnings),
            started_at=started,
            completed_at=utc_now(),
        )

        with self._lock:
            self._workflows[
                workflow.workflow_id
            ] = workflow

        self.audit_logger.record(
            workflow_id=workflow.workflow_id,
            actor_id="supervisor_agent",
            action="workflow_completed",
            details={
                "status": workflow.status.value,
                "completed_tasks": str(completed_count),
                "failed_tasks": str(failed_count),
                "blocked_tasks": str(blocked_count),
            },
        )

        return workflow

    def review_workflow(
        self,
        workflow_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> SupervisorWorkflowResult:
        with self._lock:
            workflow = self._workflows.get(
                workflow_id
            )

            if workflow is None:
                raise WorkflowNotFoundError(
                    "Workflow not found"
                )

            if workflow.status not in {
                WorkflowStatus.COMPLETED_PENDING_REVIEW,
                WorkflowStatus.PARTIALLY_COMPLETED,
                WorkflowStatus.FAILED,
            }:
                raise SupervisorApprovalError(
                    "Workflow is not reviewable"
                )

            self.approval_manager.review(
                workflow,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            status_map = {
                ReviewDecision.APPROVE_FOR_COORDINATION: (
                    WorkflowStatus.APPROVED_FOR_COORDINATION
                ),
                ReviewDecision.REJECT: (
                    WorkflowStatus.REJECTED
                ),
                ReviewDecision.REQUEST_REVISION: (
                    WorkflowStatus.REVISION_REQUIRED
                ),
            }

            updated = replace(
                workflow,
                status=status_map[
                    decision
                ],
            )

            self._workflows[
                workflow_id
            ] = updated

            return updated

    def get_workflow(
        self,
        workflow_id: str,
    ) -> SupervisorWorkflowResult | None:
        with self._lock:
            return self._workflows.get(
                workflow_id
            )

    def get_audit_history(
        self,
        workflow_id: str,
    ) -> tuple[SupervisorAuditRecord, ...]:
        return self.audit_logger.list_records(
            workflow_id
        )

    def list_registered_agents(
        self,
    ) -> tuple[AgentDescriptor, ...]:
        return self.registry.list_agents()


# ============================================================
# JSON EXPORT
# ============================================================

class SupervisorReportExporter:
    @staticmethod
    def export_json(
        workflow: SupervisorWorkflowResult,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise SupervisorExportError(
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

def create_supervisor_agent(
    config: SupervisorConfig | None = None,
) -> SupervisorAgent:
    return SupervisorAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    supervisor = create_supervisor_agent()

    # Fictional handlers.
    # Replace with explicit adapters to real agents.

    def mock_sos_triage(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "sos_reference": payload.get(
                "sos_reference"
            ),
            "urgency": "immediate_human_escalation",
            "human_review_required": True,
            "advisory_only": True,
        }

    def mock_risk_assessment(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "risk_level": "high",
            "source": "fictional_demo",
            "human_review_required": True,
            "advisory_only": True,
        }

    def mock_rescue_coordination(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "recommendation": (
                "Request authorized rescue coordinator review"
            ),
            "dispatch_performed": False,
            "human_review_required": True,
            "advisory_only": True,
        }

    supervisor.register_agent(
        AgentDescriptor(
            agent_id="sos_triage_agent",
            agent_name="SOS Triage Agent",
            capabilities=(
                AgentCapability.SOS_TRIAGE,
            ),
        ),
        mock_sos_triage,
    )

    supervisor.register_agent(
        AgentDescriptor(
            agent_id="risk_assessment_agent",
            agent_name="Risk Assessment Agent",
            capabilities=(
                AgentCapability.RISK_ASSESSMENT,
            ),
        ),
        mock_risk_assessment,
    )

    supervisor.register_agent(
        AgentDescriptor(
            agent_id="rescue_coordinator_agent",
            agent_name="Rescue Coordinator Agent",
            capabilities=(
                AgentCapability.RESCUE_COORDINATION,
            ),
        ),
        mock_rescue_coordination,
    )

    request = WorkflowRequest(
        workflow_id="DEMO-WORKFLOW-001",
        district="Thrissur",
        incident_reference="DEMO-INCIDENT-001",
        created_at=utc_now(),
        tasks=(
            SupervisorTask(
                task_id="triage",
                capability=AgentCapability.SOS_TRIAGE,
                priority=TaskPriority.CRITICAL,
                payload={
                    "sos_reference": "DEMO-SOS-001"
                },
            ),
            SupervisorTask(
                task_id="risk",
                capability=AgentCapability.RISK_ASSESSMENT,
                priority=TaskPriority.HIGH,
                payload={
                    "district": "Thrissur"
                },
            ),
            SupervisorTask(
                task_id="rescue",
                capability=AgentCapability.RESCUE_COORDINATION,
                priority=TaskPriority.CRITICAL,
                payload={
                    "incident_reference": "DEMO-INCIDENT-001"
                },
                dependencies=(
                    "triage",
                    "risk",
                ),
            ),
        ),
    )

    result = supervisor.run(
        request
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    if result.status == (
        WorkflowStatus.COMPLETED_PENDING_REVIEW
    ):
        reviewed = supervisor.review_workflow(
            result.workflow_id,
            reviewer_id="DEMO-INCIDENT-COMMANDER",
            reviewer_role="incident_commander",
            decision=(
                ReviewDecision.APPROVE_FOR_COORDINATION
            ),
            reason="Fictional workflow review",
        )

        print(
            "Workflow status:",
            reviewed.status.value,
        )

        SupervisorReportExporter.export_json(
            reviewed,
            "outputs/agents/supervisor_workflow_report.json",
        )

    print(
        "No rescue, evacuation, alert, "
        "or other physical action was executed."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SupervisorError",
    "SupervisorValidationError",
    "AgentRegistrationError",
    "AgentExecutionError",
    "WorkflowNotFoundError",
    "SupervisorApprovalError",
    "SupervisorExportError",
    "AgentCapability",
    "TaskPriority",
    "TaskStatus",
    "WorkflowStatus",
    "AgentHealth",
    "ReviewDecision",
    "SupervisorConfig",
    "AgentDescriptor",
    "SupervisorTask",
    "WorkflowRequest",
    "AgentExecutionResult",
    "SupervisorWorkflowResult",
    "SupervisorReviewRecord",
    "SupervisorAuditRecord",
    "AgentHandler",
    "RegisteredAgent",
    "SupervisorAgentRegistry",
    "SupervisorWorkflowValidator",
    "SupervisorTaskScheduler",
    "SupervisorExecutionEngine",
    "SupervisorAuditLogger",
    "SupervisorHumanApprovalManager",
    "SupervisorAgent",
    "SupervisorReportExporter",
    "create_supervisor_agent",
]
