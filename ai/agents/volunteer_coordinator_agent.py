"""Placeholder module: ai/agents/volunteer_coordinator_agent.py; implementation required."""

"""
backend/ai/agents/volunteer_coordinator_agent.py

Kerala Flood Relief Coordination Agent
Volunteer Coordination & Safety Intelligence

Python 3.11+
Standard library only.

FEATURES
--------
1. Volunteer registration and eligibility screening
2. Skill and training verification
3. Volunteer availability and shift matching
4. Disaster-relief task prioritization
5. Location and district matching
6. Safety-risk screening
7. Medical and rescue qualification checks
8. Volunteer fatigue screening
9. Multi-volunteer task allocation proposals
10. Unfilled task escalation
11. Human approval workflow
12. Audit logging and JSON export

SAFETY
------
All assignments are advisory proposals.

This module does NOT:
- Dispatch volunteers
- Authorize entry into floodwater
- Certify rescue or medical qualifications
- Confirm safe travel routes
- Send volunteer notifications
- Execute emergency operations

High-risk tasks require authorized
professional supervision and separate
operational authorization.

No automated assignment is permitted
for water rescue, active search and rescue,
medical treatment, or hazardous operations.

Illustrative rules are not certified
emergency-management procedures.
"""

from __future__ import annotations

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

class VolunteerCoordinatorError(Exception):
    pass


class VolunteerValidationError(VolunteerCoordinatorError):
    pass


class VolunteerEligibilityError(VolunteerCoordinatorError):
    pass


class VolunteerApprovalError(VolunteerCoordinatorError):
    pass


class VolunteerExportError(VolunteerCoordinatorError):
    pass


# ============================================================
# ENUMS
# ============================================================

class VolunteerStatus(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    RESTING = "resting"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class VolunteerSkill(str, Enum):
    RELIEF_DISTRIBUTION = "relief_distribution"
    SHELTER_SUPPORT = "shelter_support"
    FOOD_PREPARATION = "food_preparation"
    INVENTORY_MANAGEMENT = "inventory_management"
    TRANSLATION = "translation"
    COMMUNICATION = "communication"
    DATA_ENTRY = "data_entry"
    LOGISTICS = "logistics"
    DRIVER = "driver"
    FIRST_AID = "first_aid"
    MEDICAL_PROFESSIONAL = "medical_professional"
    SWIFT_WATER_RESCUE = "swift_water_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    ELECTRICAL_SAFETY = "electrical_safety"
    STRUCTURAL_INSPECTION = "structural_inspection"


class VolunteerTaskType(str, Enum):
    RELIEF_PACKING = "relief_packing"
    RELIEF_DISTRIBUTION = "relief_distribution"
    SHELTER_SUPPORT = "shelter_support"
    FOOD_PREPARATION = "food_preparation"
    INVENTORY_SUPPORT = "inventory_support"
    TRANSLATION = "translation"
    INFORMATION_DESK = "information_desk"
    DATA_ENTRY = "data_entry"
    LOGISTICS_SUPPORT = "logistics_support"
    TRANSPORT_SUPPORT = "transport_support"
    FIRST_AID_SUPPORT = "first_aid_support"
    MEDICAL_RESPONSE = "medical_response"
    WATER_RESCUE = "water_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    ELECTRICAL_INSPECTION = "electrical_inspection"
    STRUCTURAL_INSPECTION = "structural_inspection"


class TaskPriority(str, Enum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class TaskRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class QualificationStatus(str, Enum):
    VERIFIED = "verified"
    PENDING_VERIFICATION = "pending_verification"
    EXPIRED = "expired"
    REJECTED = "rejected"


class AssignmentStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED_FOR_COORDINATION = "approved_for_coordination"
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"


class VolunteerPlanStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED_FOR_COORDINATION = "approved_for_coordination"
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
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
        raise VolunteerValidationError(
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
        raise VolunteerValidationError(
            f"{name} must be nonnegative and finite"
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    validate_nonnegative(value, name)

    if value > 1:
        raise VolunteerValidationError(
            f"{name} must be between 0 and 1"
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


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class VolunteerCoordinatorConfig:
    max_volunteers: int = 10000
    max_tasks: int = 5000

    max_status_age_minutes: int = 60

    maximum_shift_hours: float = 8.0
    maximum_recent_work_hours: float = 10.0

    minimum_readiness_score: float = 0.70

    require_verified_qualifications: bool = True
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_volunteers",
            "max_tasks",
            "max_status_age_minutes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise VolunteerValidationError(
                    f"{name} must be positive integer"
                )

        for name in (
            "maximum_shift_hours",
            "maximum_recent_work_hours",
        ):
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or value <= 0
            ):
                raise VolunteerValidationError(
                    f"{name} must be positive"
                )

        validate_probability(
            self.minimum_readiness_score,
            "minimum_readiness_score",
        )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class VolunteerQualification:
    skill: VolunteerSkill

    status: QualificationStatus

    verified_by: str | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.expires_at is not None:
            validate_datetime(
                self.expires_at
            )


@dataclass(frozen=True)
class VolunteerProfile:
    volunteer_id: str

    district: str
    locality: str

    status: VolunteerStatus

    qualifications: tuple[
        VolunteerQualification,
        ...
    ]

    available_from: datetime
    available_until: datetime

    recent_work_hours: float

    readiness_score: float

    status_verified_at: datetime

    willing_for_field_work: bool = False

    def __post_init__(self) -> None:
        for name in (
            "volunteer_id",
            "district",
            "locality",
        ):
            if not getattr(self, name).strip():
                raise VolunteerValidationError(
                    f"{name} required"
                )

        for value in (
            self.available_from,
            self.available_until,
            self.status_verified_at,
        ):
            validate_datetime(value)

        if self.available_until <= self.available_from:
            raise VolunteerValidationError(
                "Invalid availability window"
            )

        validate_nonnegative(
            self.recent_work_hours,
            "recent_work_hours",
        )

        validate_probability(
            self.readiness_score,
            "readiness_score",
        )


@dataclass(frozen=True)
class VolunteerTask:
    task_id: str

    district: str
    locality: str

    task_type: VolunteerTaskType

    priority: TaskPriority
    risk_level: TaskRiskLevel

    required_skills: tuple[
        VolunteerSkill,
        ...
    ]

    required_volunteers: int

    shift_start: datetime
    shift_end: datetime

    authorized_supervisor_id: str | None = None

    field_work_required: bool = False

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "district",
            "locality",
        ):
            if not getattr(self, name).strip():
                raise VolunteerValidationError(
                    f"{name} required"
                )

        if (
            isinstance(self.required_volunteers, bool)
            or not isinstance(self.required_volunteers, int)
            or self.required_volunteers <= 0
        ):
            raise VolunteerValidationError(
                "required_volunteers must be positive integer"
            )

        validate_datetime(
            self.shift_start
        )

        validate_datetime(
            self.shift_end
        )

        if self.shift_end <= self.shift_start:
            raise VolunteerValidationError(
                "Invalid task shift window"
            )


@dataclass(frozen=True)
class VolunteerCoordinationRequest:
    request_id: str

    district: str

    volunteers: tuple[
        VolunteerProfile,
        ...
    ]

    tasks: tuple[
        VolunteerTask,
        ...
    ]

    created_at: datetime

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise VolunteerValidationError(
                "request_id required"
            )

        if not self.district.strip():
            raise VolunteerValidationError(
                "district required"
            )

        validate_datetime(
            self.created_at
        )


@dataclass(frozen=True)
class VolunteerEligibilityResult:
    volunteer_id: str
    task_id: str

    eligible: bool

    reasons: tuple[str, ...]


@dataclass(frozen=True)
class VolunteerAssignmentProposal:
    assignment_id: str

    task_id: str

    volunteer_ids: tuple[str, ...]

    requested_volunteers: int
    proposed_volunteers: int

    status: AssignmentStatus

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class UnfilledVolunteerTask:
    task_id: str

    required_volunteers: int
    proposed_volunteers: int

    shortage: int

    reason: str


@dataclass(frozen=True)
class VolunteerCoordinationPlan:
    plan_id: str
    request_id: str

    district: str

    assignments: tuple[
        VolunteerAssignmentProposal,
        ...
    ]

    unfilled_tasks: tuple[
        UnfilledVolunteerTask,
        ...
    ]

    excluded_tasks: Mapping[str, str]

    warnings: tuple[str, ...]

    status: VolunteerPlanStatus

    human_approval_required: bool

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class VolunteerReviewRecord:
    review_id: str
    plan_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ReviewDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class VolunteerAuditRecord:
    audit_id: str
    plan_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# TASK SAFETY ENGINE
# ============================================================

class VolunteerTaskSafetyEngine:
    """
    Excludes hazardous professional-response tasks
    from automated volunteer matching.

    Excluded tasks require separate specialist
    coordination outside this agent.
    """

    RESTRICTED_TASKS = {
        VolunteerTaskType.MEDICAL_RESPONSE,
        VolunteerTaskType.WATER_RESCUE,
        VolunteerTaskType.SEARCH_AND_RESCUE,
        VolunteerTaskType.ELECTRICAL_INSPECTION,
        VolunteerTaskType.STRUCTURAL_INSPECTION,
    }

    @classmethod
    def screen(
        cls,
        tasks: Sequence[VolunteerTask],
    ) -> tuple[
        tuple[VolunteerTask, ...],
        dict[str, str],
    ]:
        eligible = []
        excluded = {}

        for task in tasks:
            if task.task_type in cls.RESTRICTED_TASKS:
                excluded[
                    task.task_id
                ] = (
                    "Specialist emergency operation; "
                    "separate authorized coordination required"
                )

                continue

            if task.risk_level in {
                TaskRiskLevel.HIGH,
                TaskRiskLevel.EXTREME,
            }:
                excluded[
                    task.task_id
                ] = (
                    "Hazard level too high for "
                    "general volunteer matching"
                )

                continue

            if (
                task.risk_level == TaskRiskLevel.MODERATE
                and not task.authorized_supervisor_id
            ):
                excluded[
                    task.task_id
                ] = (
                    "Moderate-risk task requires "
                    "named authorized supervisor"
                )

                continue

            eligible.append(
                task
            )

        return tuple(eligible), excluded


# ============================================================
# VOLUNTEER ELIGIBILITY ENGINE
# ============================================================

class VolunteerEligibilityEngine:
    def __init__(
        self,
        config: VolunteerCoordinatorConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        volunteer: VolunteerProfile,
        task: VolunteerTask,
        *,
        now: datetime,
    ) -> VolunteerEligibilityResult:
        reasons = []

        if volunteer.status != VolunteerStatus.AVAILABLE:
            reasons.append(
                "Volunteer not available"
            )

        if (
            volunteer.district.casefold()
            != task.district.casefold()
        ):
            reasons.append(
                "District mismatch"
            )

        if volunteer.status_verified_at > now:
            reasons.append(
                "Future status verification timestamp"
            )

        elif (
            now - volunteer.status_verified_at
        ) > timedelta(
            minutes=self.config.max_status_age_minutes
        ):
            reasons.append(
                "Volunteer status is stale"
            )

        if volunteer.readiness_score < (
            self.config.minimum_readiness_score
        ):
            reasons.append(
                "Readiness below configured threshold"
            )

        if volunteer.recent_work_hours >= (
            self.config.maximum_recent_work_hours
        ):
            reasons.append(
                "Recent work hours exceed fatigue threshold"
            )

        if (
            volunteer.available_from > task.shift_start
            or volunteer.available_until < task.shift_end
        ):
            reasons.append(
                "Availability does not cover task shift"
            )

        shift_hours = (
            task.shift_end - task.shift_start
        ).total_seconds() / 3600

        if shift_hours > self.config.maximum_shift_hours:
            reasons.append(
                "Task shift exceeds maximum duration"
            )

        if (
            task.field_work_required
            and not volunteer.willing_for_field_work
        ):
            reasons.append(
                "Volunteer has not opted into field work"
            )

        verified_skills = {
            qualification.skill
            for qualification in volunteer.qualifications
            if (
                qualification.status
                == QualificationStatus.VERIFIED
                and bool(qualification.verified_by)
                and (
                    qualification.expires_at is None
                    or qualification.expires_at > now
                )
            )
        }

        missing_skills = (
            set(task.required_skills)
            - verified_skills
        )

        if missing_skills:
            reasons.append(
                "Missing verified required skills: "
                + ", ".join(
                    sorted(
                        skill.value
                        for skill in missing_skills
                    )
                )
            )

        return VolunteerEligibilityResult(
            volunteer_id=volunteer.volunteer_id,
            task_id=task.task_id,
            eligible=not reasons,
            reasons=tuple(reasons),
        )


# ============================================================
# TASK PRIORITY SCHEDULER
# ============================================================

class VolunteerTaskScheduler:
    PRIORITY_ORDER = {
        TaskPriority.CRITICAL: 0,
        TaskPriority.HIGH: 1,
        TaskPriority.ELEVATED: 2,
        TaskPriority.ROUTINE: 3,
    }

    @classmethod
    def order(
        cls,
        tasks: Sequence[VolunteerTask],
    ) -> tuple[VolunteerTask, ...]:
        return tuple(
            sorted(
                tasks,
                key=lambda task: (
                    cls.PRIORITY_ORDER[
                        task.priority
                    ],
                    task.shift_start,
                    task.task_id,
                ),
            )
        )


# ============================================================
# ASSIGNMENT ENGINE
# ============================================================

class VolunteerAssignmentEngine:
    def __init__(
        self,
        eligibility_engine: VolunteerEligibilityEngine,
    ) -> None:
        self.eligibility_engine = eligibility_engine

    @staticmethod
    def overlaps(
        first: VolunteerTask,
        second: VolunteerTask,
    ) -> bool:
        return (
            first.shift_start < second.shift_end
            and second.shift_start < first.shift_end
        )

    def allocate(
        self,
        volunteers: Sequence[VolunteerProfile],
        tasks: Sequence[VolunteerTask],
        *,
        now: datetime,
    ) -> tuple[
        tuple[VolunteerAssignmentProposal, ...],
        tuple[UnfilledVolunteerTask, ...],
    ]:
        assignments = []
        shortages = []

        # Tracks proposed task IDs for each volunteer.
        # Prevents overlapping shifts within this plan.
        proposed_tasks: dict[
            str,
            list[VolunteerTask],
        ] = {
            volunteer.volunteer_id: []
            for volunteer in volunteers
        }

        for task in tasks:
            candidates = []

            for volunteer in volunteers:
                eligibility = (
                    self.eligibility_engine.evaluate(
                        volunteer,
                        task,
                        now=now,
                    )
                )

                if not eligibility.eligible:
                    continue

                has_conflict = any(
                    self.overlaps(
                        task,
                        existing_task,
                    )
                    for existing_task in proposed_tasks[
                        volunteer.volunteer_id
                    ]
                )

                if has_conflict:
                    continue

                locality_match = (
                    volunteer.locality.casefold()
                    == task.locality.casefold()
                )

                candidates.append(
                    (
                        not locality_match,
                        -volunteer.readiness_score,
                        volunteer.recent_work_hours,
                        volunteer.volunteer_id,
                        volunteer,
                    )
                )

            candidates.sort(
                key=lambda item: item[:4]
            )

            selected = [
                item[4]
                for item in candidates[
                    :task.required_volunteers
                ]
            ]

            selected_ids = tuple(
                volunteer.volunteer_id
                for volunteer in selected
            )

            for volunteer in selected:
                proposed_tasks[
                    volunteer.volunteer_id
                ].append(
                    task
                )

            if selected_ids:
                assignments.append(
                    VolunteerAssignmentProposal(
                        assignment_id=generate_id(),
                        task_id=task.task_id,
                        volunteer_ids=selected_ids,
                        requested_volunteers=(
                            task.required_volunteers
                        ),
                        proposed_volunteers=len(
                            selected_ids
                        ),
                        status=AssignmentStatus.PROPOSED,
                        warnings=(
                            "Assignment is advisory only",
                            "Volunteer consent must be reconfirmed",
                            "Travel safety must be checked separately",
                            "No notification or dispatch performed",
                        ),
                    )
                )

            shortage = (
                task.required_volunteers
                - len(selected_ids)
            )

            if shortage > 0:
                shortages.append(
                    UnfilledVolunteerTask(
                        task_id=task.task_id,
                        required_volunteers=(
                            task.required_volunteers
                        ),
                        proposed_volunteers=len(
                            selected_ids
                        ),
                        shortage=shortage,
                        reason=(
                            "Insufficient eligible volunteers "
                            "for the requested shift"
                        ),
                    )
                )

        return (
            tuple(assignments),
            tuple(shortages),
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class VolunteerCoordinationAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            VolunteerAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        plan_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> VolunteerAuditRecord:
        record = VolunteerAuditRecord(
            audit_id=generate_id(),
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
    ) -> tuple[VolunteerAuditRecord, ...]:
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

class VolunteerHumanApprovalManager:
    ALLOWED_ROLES = {
        "district_disaster_management_officer",
        "authorized_volunteer_coordinator",
        "incident_commander",
        "relief_operations_manager",
    }

    def __init__(
        self,
        audit_logger: VolunteerCoordinationAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            VolunteerReviewRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        plan: VolunteerCoordinationPlan,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> VolunteerReviewRecord:
        if not reviewer_id.strip():
            raise VolunteerApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise VolunteerApprovalError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise VolunteerApprovalError(
                "Review reason required"
            )

        with self._lock:
            if plan.plan_id in self._reviews:
                raise VolunteerApprovalError(
                    "Plan already reviewed"
                )

            review = VolunteerReviewRecord(
                review_id=generate_id(),
                plan_id=plan.plan_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                plan.plan_id
            ] = review

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id=reviewer_id,
            action=f"human_review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN VOLUNTEER COORDINATOR AGENT
# ============================================================

class VolunteerCoordinatorAgent:
    def __init__(
        self,
        config: VolunteerCoordinatorConfig | None = None,
    ) -> None:
        self.config = (
            config or VolunteerCoordinatorConfig()
        )

        self.safety_engine = (
            VolunteerTaskSafetyEngine()
        )

        self.eligibility_engine = (
            VolunteerEligibilityEngine(
                self.config
            )
        )

        self.scheduler = (
            VolunteerTaskScheduler()
        )

        self.assignment_engine = (
            VolunteerAssignmentEngine(
                self.eligibility_engine
            )
        )

        self.audit_logger = (
            VolunteerCoordinationAuditLogger()
        )

        self.approval_manager = (
            VolunteerHumanApprovalManager(
                self.audit_logger
            )
        )

        self._plans: dict[
            str,
            VolunteerCoordinationPlan,
        ] = {}

        self._lock = threading.RLock()

    def run(
        self,
        request: VolunteerCoordinationRequest,
        *,
        now: datetime | None = None,
    ) -> VolunteerCoordinationPlan:
        now = now or utc_now()

        validate_datetime(now)

        if request.created_at > now:
            raise VolunteerValidationError(
                "Request timestamp is in the future"
            )

        if len(request.volunteers) > (
            self.config.max_volunteers
        ):
            raise VolunteerValidationError(
                "Volunteer limit exceeded"
            )

        if len(request.tasks) > (
            self.config.max_tasks
        ):
            raise VolunteerValidationError(
                "Task limit exceeded"
            )

        volunteer_ids = [
            volunteer.volunteer_id
            for volunteer in request.volunteers
        ]

        task_ids = [
            task.task_id
            for task in request.tasks
        ]

        if len(volunteer_ids) != len(
            set(volunteer_ids)
        ):
            raise VolunteerValidationError(
                "Duplicate volunteer ID"
            )

        if len(task_ids) != len(
            set(task_ids)
        ):
            raise VolunteerValidationError(
                "Duplicate task ID"
            )

        for task in request.tasks:
            if (
                task.district.casefold()
                != request.district.casefold()
            ):
                raise VolunteerValidationError(
                    "Task district mismatch"
                )

            if task.shift_start < now:
                raise VolunteerValidationError(
                    "Task shift cannot start in the past"
                )

        eligible_tasks, excluded = (
            self.safety_engine.screen(
                request.tasks
            )
        )

        ordered_tasks = self.scheduler.order(
            eligible_tasks
        )

        assignments, shortages = (
            self.assignment_engine.allocate(
                request.volunteers,
                ordered_tasks,
                now=now,
            )
        )

        warnings = [
            "Volunteer assignments are proposals only",
            "No volunteer has been dispatched",
            "Volunteer consent must be reconfirmed",
            "Travel and site safety require separate verification",
            "High-risk operations require specialist coordination",
            "Human approval is mandatory",
        ]

        if excluded:
            warnings.append(
                "Some hazardous tasks were excluded "
                "from general volunteer matching"
            )

        if shortages:
            warnings.append(
                "Some volunteer tasks remain unfilled"
            )

        plan = VolunteerCoordinationPlan(
            plan_id=generate_id(),
            request_id=request.request_id,
            district=request.district,
            assignments=assignments,
            unfilled_tasks=shortages,
            excluded_tasks=excluded,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            status=VolunteerPlanStatus.PENDING_REVIEW,
            human_approval_required=True,
            created_at=now,
        )

        with self._lock:
            self._plans[
                plan.plan_id
            ] = plan

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id="volunteer_coordinator_agent",
            action="volunteer_plan_created",
            details={
                "district": request.district,
                "volunteers": str(
                    len(request.volunteers)
                ),
                "tasks": str(
                    len(request.tasks)
                ),
                "assignments": str(
                    len(assignments)
                ),
                "excluded_tasks": str(
                    len(excluded)
                ),
            },
        )

        return plan

    def review_plan(
        self,
        plan_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> VolunteerCoordinationPlan:
        with self._lock:
            plan = self._plans.get(
                plan_id
            )

            if plan is None:
                raise VolunteerApprovalError(
                    "Volunteer plan not found"
                )

            if plan.status != (
                VolunteerPlanStatus.PENDING_REVIEW
            ):
                raise VolunteerApprovalError(
                    "Plan is not pending review"
                )

            self.approval_manager.review(
                plan,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            if decision == ReviewDecision.APPROVE:
                plan_status = (
                    VolunteerPlanStatus.APPROVED_FOR_COORDINATION
                )

                assignment_status = (
                    AssignmentStatus.APPROVED_FOR_COORDINATION
                )

            elif decision == ReviewDecision.REJECT:
                plan_status = (
                    VolunteerPlanStatus.REJECTED
                )

                assignment_status = (
                    AssignmentStatus.REJECTED
                )

            else:
                plan_status = (
                    VolunteerPlanStatus.REVISION_REQUIRED
                )

                assignment_status = (
                    AssignmentStatus.REVISION_REQUIRED
                )

            updated_assignments = tuple(
                replace(
                    assignment,
                    status=assignment_status,
                )
                for assignment in plan.assignments
            )

            updated = replace(
                plan,
                status=plan_status,
                assignments=updated_assignments,
            )

            self._plans[
                plan_id
            ] = updated

            return updated

    def get_plan(
        self,
        plan_id: str,
    ) -> VolunteerCoordinationPlan | None:
        with self._lock:
            return self._plans.get(
                plan_id
            )

    def get_audit_history(
        self,
        plan_id: str,
    ) -> tuple[VolunteerAuditRecord, ...]:
        return self.audit_logger.list_records(
            plan_id
        )


# ============================================================
# JSON EXPORTER
# ============================================================

class VolunteerCoordinationReportExporter:
    @staticmethod
    def export_json(
        plan: VolunteerCoordinationPlan,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise VolunteerExportError(
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

def create_volunteer_coordinator_agent(
    config: VolunteerCoordinatorConfig | None = None,
) -> VolunteerCoordinatorAgent:
    return VolunteerCoordinatorAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional demonstration only.
    # No real volunteer or task is registered.

    volunteers = (
        VolunteerProfile(
            volunteer_id="DEMO-VOL-001",
            district="Thrissur",
            locality="Demo Area A",
            status=VolunteerStatus.AVAILABLE,
            qualifications=(
                VolunteerQualification(
                    skill=VolunteerSkill.RELIEF_DISTRIBUTION,
                    status=QualificationStatus.VERIFIED,
                    verified_by="DEMO-COORDINATOR",
                ),
                VolunteerQualification(
                    skill=VolunteerSkill.LOGISTICS,
                    status=QualificationStatus.VERIFIED,
                    verified_by="DEMO-COORDINATOR",
                ),
            ),
            available_from=now,
            available_until=now + timedelta(hours=8),
            recent_work_hours=2.0,
            readiness_score=0.95,
            status_verified_at=now,
            willing_for_field_work=True,
        ),
        VolunteerProfile(
            volunteer_id="DEMO-VOL-002",
            district="Thrissur",
            locality="Demo Area B",
            status=VolunteerStatus.AVAILABLE,
            qualifications=(
                VolunteerQualification(
                    skill=VolunteerSkill.RELIEF_DISTRIBUTION,
                    status=QualificationStatus.VERIFIED,
                    verified_by="DEMO-COORDINATOR",
                ),
            ),
            available_from=now,
            available_until=now + timedelta(hours=8),
            recent_work_hours=1.0,
            readiness_score=0.90,
            status_verified_at=now,
            willing_for_field_work=True,
        ),
    )

    tasks = (
        VolunteerTask(
            task_id="DEMO-TASK-001",
            district="Thrissur",
            locality="Demo Area A",
            task_type=VolunteerTaskType.RELIEF_DISTRIBUTION,
            priority=TaskPriority.HIGH,
            risk_level=TaskRiskLevel.LOW,
            required_skills=(
                VolunteerSkill.RELIEF_DISTRIBUTION,
            ),
            required_volunteers=2,
            shift_start=now + timedelta(minutes=30),
            shift_end=now + timedelta(hours=4),
            field_work_required=True,
        ),
        VolunteerTask(
            task_id="DEMO-TASK-002",
            district="Thrissur",
            locality="Demo Flood Zone",
            task_type=VolunteerTaskType.WATER_RESCUE,
            priority=TaskPriority.CRITICAL,
            risk_level=TaskRiskLevel.EXTREME,
            required_skills=(
                VolunteerSkill.SWIFT_WATER_RESCUE,
            ),
            required_volunteers=3,
            shift_start=now + timedelta(minutes=30),
            shift_end=now + timedelta(hours=3),
            field_work_required=True,
        ),
    )

    request = VolunteerCoordinationRequest(
        request_id="DEMO-VOLUNTEER-REQUEST-001",
        district="Thrissur",
        volunteers=volunteers,
        tasks=tasks,
        created_at=now,
    )

    agent = create_volunteer_coordinator_agent()

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

    reviewed = agent.review_plan(
        plan.plan_id,
        reviewer_id="DEMO-VOLUNTEER-COORDINATOR",
        reviewer_role="authorized_volunteer_coordinator",
        decision=ReviewDecision.APPROVE,
        reason="Fictional coordination review",
    )

    print(
        "Plan status:",
        reviewed.status.value,
    )

    VolunteerCoordinationReportExporter.export_json(
        reviewed,
        "outputs/agents/volunteer_coordination_report.json",
    )

    print(
        "No volunteer dispatch or "
        "hazardous operation occurred."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "VolunteerCoordinatorError",
    "VolunteerValidationError",
    "VolunteerEligibilityError",
    "VolunteerApprovalError",
    "VolunteerExportError",
    "VolunteerStatus",
    "VolunteerSkill",
    "VolunteerTaskType",
    "TaskPriority",
    "TaskRiskLevel",
    "QualificationStatus",
    "AssignmentStatus",
    "VolunteerPlanStatus",
    "ReviewDecision",
    "VolunteerCoordinatorConfig",
    "VolunteerQualification",
    "VolunteerProfile",
    "VolunteerTask",
    "VolunteerCoordinationRequest",
    "VolunteerEligibilityResult",
    "VolunteerAssignmentProposal",
    "UnfilledVolunteerTask",
    "VolunteerCoordinationPlan",
    "VolunteerReviewRecord",
    "VolunteerAuditRecord",
    "VolunteerTaskSafetyEngine",
    "VolunteerEligibilityEngine",
    "VolunteerTaskScheduler",
    "VolunteerAssignmentEngine",
    "VolunteerCoordinationAuditLogger",
    "VolunteerHumanApprovalManager",
    "VolunteerCoordinatorAgent",
    "VolunteerCoordinationReportExporter",
    "create_volunteer_coordinator_agent",
]
