
"""
backend/ai/agents/shelter_allocation_agent.py

Kerala Flood Relief Coordination Agent
Emergency Shelter Allocation Agent

Python 3.11+
Standard library only.

FEATURES
--------
1. Displaced household intake
2. Vulnerability-based prioritization
3. Verified shelter availability screening
4. Shelter capacity management
5. Family/household group preservation
6. Accessibility and medical support matching
7. Pet accommodation matching
8. Shelter resource requirement screening
9. Provisional household-to-shelter allocation
10. Alternative shelter recommendations
11. Unallocated household escalation
12. Human approval workflow
13. Audit logging
14. JSON report export

SAFETY
------
All allocations are advisory proposals.

- No shelter reservation is made.
- No transport or evacuation is dispatched.
- No household is automatically relocated.
- No medical suitability is clinically certified.
- Shelter safety and operating status require
  verification by authorized personnel.
- Human authorization is mandatory before
  implementing any allocation.
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

class ShelterAllocationError(Exception):
    pass


class ShelterValidationError(ShelterAllocationError):
    pass


class ShelterCapacityError(ShelterAllocationError):
    pass


class ShelterApprovalError(ShelterAllocationError):
    pass


class ShelterExportError(ShelterAllocationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ShelterStatus(str, Enum):
    VERIFIED_OPERATIONAL = "verified_operational"
    LIMITED_CAPACITY = "limited_capacity"
    UNAVAILABLE = "unavailable"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class HouseholdPriority(str, Enum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    HIGH = "high"
    URGENT = "urgent"


class AccessibilityRequirement(str, Enum):
    NONE = "none"
    WHEELCHAIR_ACCESS = "wheelchair_access"
    STEP_FREE_ACCESS = "step_free_access"
    ACCESSIBLE_TOILET = "accessible_toilet"


class MedicalSupportRequirement(str, Enum):
    NONE = "none"
    BASIC_FIRST_AID = "basic_first_aid"
    MEDICAL_STAFF_ACCESS = "medical_staff_access"
    ELECTRICITY_DEPENDENT_EQUIPMENT = (
        "electricity_dependent_equipment"
    )


class ShelterFacility(str, Enum):
    DRINKING_WATER = "drinking_water"
    SANITATION = "sanitation"
    ELECTRICITY = "electricity"
    BACKUP_POWER = "backup_power"
    FOOD_SERVICE = "food_service"
    FIRST_AID = "first_aid"
    MEDICAL_STAFF = "medical_staff"
    ACCESSIBLE_TOILET = "accessible_toilet"
    STEP_FREE_ACCESS = "step_free_access"
    WHEELCHAIR_ACCESS = "wheelchair_access"
    PET_AREA = "pet_area"
    CHILD_FRIENDLY_SPACE = "child_friendly_space"


class AllocationStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED_FOR_COORDINATION = "approved_for_coordination"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"


class ShelterPlanStatus(str, Enum):
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
        raise ShelterValidationError(
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
        raise ShelterValidationError(
            f"{name} must be nonnegative and finite"
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    validate_nonnegative(value, name)

    if value > 1:
        raise ShelterValidationError(
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

    if isinstance(value, (list, tuple)):
        return [
            serialize(item)
            for item in value
        ]

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ShelterAllocationConfig:
    max_households: int = 10000
    max_shelters: int = 1000

    max_shelter_status_age_minutes: int = 60

    maximum_target_utilization: float = 0.90

    minimum_shelter_reliability: float = 0.70

    max_alternative_shelters: int = 3

    require_verified_operational_status: bool = True
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_households",
            "max_shelters",
            "max_shelter_status_age_minutes",
            "max_alternative_shelters",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ShelterValidationError(
                    f"{name} must be positive integer"
                )

        validate_probability(
            self.maximum_target_utilization,
            "maximum_target_utilization",
        )

        validate_probability(
            self.minimum_shelter_reliability,
            "minimum_shelter_reliability",
        )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class DisplacedHousehold:
    household_id: str

    district: str
    current_location: str

    family_size: int

    children_count: int = 0
    older_adults_count: int = 0
    medically_vulnerable_count: int = 0

    accessibility_requirements: tuple[
        AccessibilityRequirement,
        ...
    ] = ()

    medical_requirements: tuple[
        MedicalSupportRequirement,
        ...
    ] = ()

    pets_count: int = 0

    immediate_shelter_need: bool = False

    requested_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "household_id",
            "district",
            "current_location",
        ):
            if not getattr(self, name).strip():
                raise ShelterValidationError(
                    f"{name} required"
                )

        if (
            isinstance(self.family_size, bool)
            or not isinstance(self.family_size, int)
            or self.family_size <= 0
        ):
            raise ShelterValidationError(
                "family_size must be positive integer"
            )

        for name in (
            "children_count",
            "older_adults_count",
            "medically_vulnerable_count",
            "pets_count",
        ):
            value = getattr(self, name)

            validate_nonnegative(value, name)

            if not isinstance(value, int):
                raise ShelterValidationError(
                    f"{name} must be integer"
                )

        for name in (
            "children_count",
            "older_adults_count",
            "medically_vulnerable_count",
        ):
            if getattr(self, name) > self.family_size:
                raise ShelterValidationError(
                    f"{name} exceeds family size"
                )

        if self.requested_at is not None:
            validate_datetime(
                self.requested_at
            )


@dataclass(frozen=True)
class EmergencyShelter:
    shelter_id: str
    shelter_name: str

    district: str
    location_description: str

    status: ShelterStatus

    total_capacity: int
    current_occupancy: int

    facilities: tuple[
        ShelterFacility,
        ...
    ]

    reliability: float

    status_verified_at: datetime | None

    def __post_init__(self) -> None:
        for name in (
            "shelter_id",
            "shelter_name",
            "district",
            "location_description",
        ):
            if not getattr(self, name).strip():
                raise ShelterValidationError(
                    f"{name} required"
                )

        for name in (
            "total_capacity",
            "current_occupancy",
        ):
            value = getattr(self, name)

            validate_nonnegative(value, name)

            if not isinstance(value, int):
                raise ShelterValidationError(
                    f"{name} must be integer"
                )

        if self.total_capacity <= 0:
            raise ShelterValidationError(
                "total_capacity must be positive"
            )

        if self.current_occupancy > self.total_capacity:
            raise ShelterValidationError(
                "current_occupancy exceeds capacity"
            )

        validate_probability(
            self.reliability,
            "reliability",
        )

        if self.status_verified_at is not None:
            validate_datetime(
                self.status_verified_at
            )


@dataclass(frozen=True)
class ShelterAllocationRequest:
    request_id: str

    district: str

    households: tuple[
        DisplacedHousehold,
        ...
    ]

    shelters: tuple[
        EmergencyShelter,
        ...
    ]

    created_at: datetime

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ShelterValidationError(
                "request_id required"
            )

        if not self.district.strip():
            raise ShelterValidationError(
                "district required"
            )

        validate_datetime(
            self.created_at
        )


@dataclass(frozen=True)
class HouseholdPriorityAssessment:
    household_id: str

    priority: HouseholdPriority
    priority_score: float

    factors: Mapping[str, float]

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ShelterScreeningResult:
    eligible_shelter_ids: tuple[str, ...]
    rejected_shelter_ids: tuple[str, ...]

    rejection_reasons: Mapping[str, str]

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ShelterAllocationProposal:
    allocation_id: str

    household_id: str
    shelter_id: str

    allocated_people: int

    priority: HouseholdPriority

    projected_occupancy: int
    projected_utilization: float

    status: AllocationStatus

    alternative_shelter_ids: tuple[str, ...]

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class UnallocatedHousehold:
    household_id: str

    priority: HouseholdPriority

    people_affected: int

    reason: str


@dataclass(frozen=True)
class ShelterCapacitySummary:
    shelter_id: str

    total_capacity: int
    current_occupancy: int

    proposed_additional_occupancy: int

    projected_occupancy: int
    projected_utilization: float

    remaining_planning_capacity: int


@dataclass(frozen=True)
class ShelterAllocationPlan:
    plan_id: str
    request_id: str

    district: str

    priority_assessments: tuple[
        HouseholdPriorityAssessment,
        ...
    ]

    shelter_screening: ShelterScreeningResult

    allocations: tuple[
        ShelterAllocationProposal,
        ...
    ]

    unallocated_households: tuple[
        UnallocatedHousehold,
        ...
    ]

    capacity_summaries: tuple[
        ShelterCapacitySummary,
        ...
    ]

    warnings: tuple[str, ...]

    status: ShelterPlanStatus

    human_approval_required: bool

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ShelterApprovalRecord:
    approval_id: str
    plan_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ReviewDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class ShelterAuditRecord:
    audit_id: str
    plan_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# HOUSEHOLD PRIORITY ENGINE
# ============================================================

class HouseholdPriorityEngine:
    @staticmethod
    def assess(
        household: DisplacedHousehold,
    ) -> HouseholdPriorityAssessment:
        vulnerable_count = min(
            household.family_size,
            household.children_count
            + household.older_adults_count
            + household.medically_vulnerable_count,
        )

        vulnerability = (
            vulnerable_count
            / household.family_size
        )

        medical = float(
            bool(household.medical_requirements)
        )

        accessibility = float(
            bool(household.accessibility_requirements)
        )

        immediate = float(
            household.immediate_shelter_need
        )

        score = (
            0.25 * vulnerability
            + 0.20 * medical
            + 0.15 * accessibility
            + 0.40 * immediate
        )

        score = round(
            min(score, 1.0),
            4,
        )

        if household.immediate_shelter_need:
            priority = HouseholdPriority.URGENT

        elif score >= 0.70:
            priority = HouseholdPriority.URGENT

        elif score >= 0.50:
            priority = HouseholdPriority.HIGH

        elif score >= 0.25:
            priority = HouseholdPriority.ELEVATED

        else:
            priority = HouseholdPriority.ROUTINE

        return HouseholdPriorityAssessment(
            household_id=household.household_id,
            priority=priority,
            priority_score=score,
            factors={
                "vulnerability": vulnerability,
                "medical": medical,
                "accessibility": accessibility,
                "immediate_need": immediate,
            },
            warnings=(
                "Priority is a planning heuristic",
                "Household needs require human verification",
            ),
        )


# ============================================================
# SHELTER ELIGIBILITY ENGINE
# ============================================================

class ShelterEligibilityEngine:
    def __init__(
        self,
        config: ShelterAllocationConfig,
    ) -> None:
        self.config = config

    def screen(
        self,
        shelters: Sequence[EmergencyShelter],
        district: str,
        *,
        now: datetime,
    ) -> tuple[
        tuple[EmergencyShelter, ...],
        ShelterScreeningResult,
    ]:
        eligible = []
        rejected = {}
        warnings = []

        for shelter in shelters:
            reason = None

            if (
                shelter.district.casefold()
                != district.casefold()
            ):
                reason = "District mismatch"

            elif shelter.status not in {
                ShelterStatus.VERIFIED_OPERATIONAL,
                ShelterStatus.LIMITED_CAPACITY,
            }:
                reason = "Shelter not verified operational"

            elif shelter.status_verified_at is None:
                reason = "Missing verification timestamp"

            elif shelter.status_verified_at > now:
                reason = "Future verification timestamp"

            elif (
                now - shelter.status_verified_at
            ) > timedelta(
                minutes=(
                    self.config.max_shelter_status_age_minutes
                )
            ):
                reason = "Stale shelter status"

            elif shelter.reliability < (
                self.config.minimum_shelter_reliability
            ):
                reason = "Insufficient source reliability"

            elif shelter.current_occupancy >= (
                shelter.total_capacity
            ):
                reason = "Shelter is full"

            if reason is not None:
                rejected[
                    shelter.shelter_id
                ] = reason

                continue

            eligible.append(
                shelter
            )

        if rejected:
            warnings.append(
                f"{len(rejected)} shelters excluded"
            )

        warnings.append(
            "Shelter status must be reverified before admission"
        )

        return (
            tuple(eligible),
            ShelterScreeningResult(
                eligible_shelter_ids=tuple(
                    shelter.shelter_id
                    for shelter in eligible
                ),
                rejected_shelter_ids=tuple(
                    rejected
                ),
                rejection_reasons=rejected,
                warnings=tuple(warnings),
            ),
        )


# ============================================================
# FACILITY MATCHING ENGINE
# ============================================================

class ShelterFacilityMatchingEngine:
    ACCESSIBILITY_MAP = {
        AccessibilityRequirement.WHEELCHAIR_ACCESS: (
            ShelterFacility.WHEELCHAIR_ACCESS
        ),
        AccessibilityRequirement.STEP_FREE_ACCESS: (
            ShelterFacility.STEP_FREE_ACCESS
        ),
        AccessibilityRequirement.ACCESSIBLE_TOILET: (
            ShelterFacility.ACCESSIBLE_TOILET
        ),
    }

    MEDICAL_MAP = {
        MedicalSupportRequirement.BASIC_FIRST_AID: (
            ShelterFacility.FIRST_AID,
        ),
        MedicalSupportRequirement.MEDICAL_STAFF_ACCESS: (
            ShelterFacility.MEDICAL_STAFF,
        ),
        MedicalSupportRequirement.ELECTRICITY_DEPENDENT_EQUIPMENT: (
            ShelterFacility.ELECTRICITY,
            ShelterFacility.BACKUP_POWER,
        ),
    }

    BASELINE_FACILITIES = {
        ShelterFacility.DRINKING_WATER,
        ShelterFacility.SANITATION,
    }

    @classmethod
    def required_facilities(
        cls,
        household: DisplacedHousehold,
    ) -> set[ShelterFacility]:
        required = set(
            cls.BASELINE_FACILITIES
        )

        for requirement in (
            household.accessibility_requirements
        ):
            if requirement == AccessibilityRequirement.NONE:
                continue

            required.add(
                cls.ACCESSIBILITY_MAP[
                    requirement
                ]
            )

        for requirement in (
            household.medical_requirements
        ):
            if requirement == MedicalSupportRequirement.NONE:
                continue

            required.update(
                cls.MEDICAL_MAP[
                    requirement
                ]
            )

        if household.pets_count > 0:
            required.add(
                ShelterFacility.PET_AREA
            )

        return required

    @classmethod
    def matches(
        cls,
        household: DisplacedHousehold,
        shelter: EmergencyShelter,
    ) -> bool:
        required = cls.required_facilities(
            household
        )

        available = set(
            shelter.facilities
        )

        return required.issubset(
            available
        )


# ============================================================
# CAPACITY ENGINE
# ============================================================

class ShelterCapacityEngine:
    def __init__(
        self,
        config: ShelterAllocationConfig,
    ) -> None:
        self.config = config

    def planning_capacity(
        self,
        shelter: EmergencyShelter,
    ) -> int:
        target_occupancy = math.floor(
            shelter.total_capacity
            * self.config.maximum_target_utilization
        )

        return max(
            0,
            target_occupancy
            - shelter.current_occupancy,
        )

    def can_allocate(
        self,
        shelter: EmergencyShelter,
        household: DisplacedHousehold,
        already_proposed: int,
    ) -> bool:
        return (
            already_proposed
            + household.family_size
            <= self.planning_capacity(
                shelter
            )
        )


# ============================================================
# ALLOCATION ENGINE
# ============================================================

class ShelterAllocationEngine:
    def __init__(
        self,
        config: ShelterAllocationConfig,
        capacity_engine: ShelterCapacityEngine,
        facility_engine: ShelterFacilityMatchingEngine,
    ) -> None:
        self.config = config
        self.capacity_engine = capacity_engine
        self.facility_engine = facility_engine

    def allocate(
        self,
        households: Sequence[DisplacedHousehold],
        shelters: Sequence[EmergencyShelter],
        priorities: Mapping[
            str,
            HouseholdPriorityAssessment,
        ],
    ) -> tuple[
        tuple[ShelterAllocationProposal, ...],
        tuple[UnallocatedHousehold, ...],
        tuple[ShelterCapacitySummary, ...],
    ]:
        proposed_occupancy = {
            shelter.shelter_id: 0
            for shelter in shelters
        }

        allocations = []
        unallocated = []

        ordered_households = sorted(
            households,
            key=lambda household: (
                -priorities[
                    household.household_id
                ].priority_score,
                household.requested_at
                or datetime.max.replace(
                    tzinfo=timezone.utc
                ),
                household.household_id,
            ),
        )

        for household in ordered_households:
            candidates = []

            for shelter in shelters:
                if not self.facility_engine.matches(
                    household,
                    shelter,
                ):
                    continue

                if not self.capacity_engine.can_allocate(
                    shelter,
                    household,
                    proposed_occupancy[
                        shelter.shelter_id
                    ],
                ):
                    continue

                remaining_after = (
                    self.capacity_engine.planning_capacity(
                        shelter
                    )
                    - proposed_occupancy[
                        shelter.shelter_id
                    ]
                    - household.family_size
                )

                projected_utilization = (
                    shelter.current_occupancy
                    + proposed_occupancy[
                        shelter.shelter_id
                    ]
                    + household.family_size
                ) / shelter.total_capacity

                candidates.append(
                    (
                        projected_utilization,
                        -remaining_after,
                        shelter.shelter_id,
                        shelter,
                    )
                )

            if not candidates:
                unallocated.append(
                    UnallocatedHousehold(
                        household_id=household.household_id,
                        priority=priorities[
                            household.household_id
                        ].priority,
                        people_affected=household.family_size,
                        reason=(
                            "No verified shelter satisfies "
                            "capacity and required facilities"
                        ),
                    )
                )

                continue

            candidates.sort(
                key=lambda item: item[:3]
            )

            selected = candidates[0][3]

            alternatives = tuple(
                item[3].shelter_id
                for item in candidates[1:]
            )[:self.config.max_alternative_shelters]

            proposed_occupancy[
                selected.shelter_id
            ] += household.family_size

            projected_occupancy = (
                selected.current_occupancy
                + proposed_occupancy[
                    selected.shelter_id
                ]
            )

            utilization = (
                projected_occupancy
                / selected.total_capacity
            )

            allocations.append(
                ShelterAllocationProposal(
                    allocation_id=generate_id(),
                    household_id=household.household_id,
                    shelter_id=selected.shelter_id,
                    allocated_people=household.family_size,
                    priority=priorities[
                        household.household_id
                    ].priority,
                    projected_occupancy=projected_occupancy,
                    projected_utilization=round(
                        utilization,
                        4,
                    ),
                    status=AllocationStatus.PROPOSED,
                    alternative_shelter_ids=alternatives,
                    warnings=(
                        "Provisional allocation only",
                        "No physical shelter reservation made",
                        "Reverify capacity before admission",
                        "Human coordination approval required",
                    ),
                )
            )

        summaries = []

        for shelter in shelters:
            additional = proposed_occupancy[
                shelter.shelter_id
            ]

            projected = (
                shelter.current_occupancy
                + additional
            )

            remaining = max(
                0,
                self.capacity_engine.planning_capacity(
                    shelter
                )
                - additional,
            )

            summaries.append(
                ShelterCapacitySummary(
                    shelter_id=shelter.shelter_id,
                    total_capacity=shelter.total_capacity,
                    current_occupancy=shelter.current_occupancy,
                    proposed_additional_occupancy=additional,
                    projected_occupancy=projected,
                    projected_utilization=round(
                        projected
                        / shelter.total_capacity,
                        4,
                    ),
                    remaining_planning_capacity=remaining,
                )
            )

        return (
            tuple(allocations),
            tuple(unallocated),
            tuple(summaries),
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class ShelterAllocationAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            ShelterAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        plan_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> ShelterAuditRecord:
        record = ShelterAuditRecord(
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
    ) -> tuple[ShelterAuditRecord, ...]:
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

class ShelterHumanApprovalManager:
    ALLOWED_ROLES = {
        "district_disaster_management_officer",
        "shelter_coordinator",
        "authorized_relief_officer",
        "incident_commander",
    }

    def __init__(
        self,
        audit_logger: ShelterAllocationAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            ShelterApprovalRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        plan: ShelterAllocationPlan,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> ShelterApprovalRecord:
        if not reviewer_id.strip():
            raise ShelterApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise ShelterApprovalError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise ShelterApprovalError(
                "Review reason required"
            )

        with self._lock:
            if plan.plan_id in self._reviews:
                raise ShelterApprovalError(
                    "Plan already reviewed"
                )

            record = ShelterApprovalRecord(
                approval_id=generate_id(),
                plan_id=plan.plan_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                plan.plan_id
            ] = record

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id=reviewer_id,
            action=f"shelter_review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return record


# ============================================================
# MAIN SHELTER ALLOCATION AGENT
# ============================================================

class ShelterAllocationAgent:
    def __init__(
        self,
        config: ShelterAllocationConfig | None = None,
    ) -> None:
        self.config = (
            config or ShelterAllocationConfig()
        )

        self.priority_engine = (
            HouseholdPriorityEngine()
        )

        self.eligibility_engine = (
            ShelterEligibilityEngine(
                self.config
            )
        )

        self.facility_engine = (
            ShelterFacilityMatchingEngine()
        )

        self.capacity_engine = (
            ShelterCapacityEngine(
                self.config
            )
        )

        self.allocation_engine = (
            ShelterAllocationEngine(
                self.config,
                self.capacity_engine,
                self.facility_engine,
            )
        )

        self.audit_logger = (
            ShelterAllocationAuditLogger()
        )

        self.approval_manager = (
            ShelterHumanApprovalManager(
                self.audit_logger
            )
        )

        self._plans: dict[
            str,
            ShelterAllocationPlan,
        ] = {}

        self._lock = threading.RLock()

    def run(
        self,
        request: ShelterAllocationRequest,
        *,
        now: datetime | None = None,
    ) -> ShelterAllocationPlan:
        now = now or utc_now()

        validate_datetime(now)

        if request.created_at > now:
            raise ShelterValidationError(
                "Request timestamp is in the future"
            )

        if len(request.households) > (
            self.config.max_households
        ):
            raise ShelterValidationError(
                "Household limit exceeded"
            )

        if len(request.shelters) > (
            self.config.max_shelters
        ):
            raise ShelterValidationError(
                "Shelter limit exceeded"
            )

        household_ids = [
            household.household_id
            for household in request.households
        ]

        shelter_ids = [
            shelter.shelter_id
            for shelter in request.shelters
        ]

        if len(household_ids) != len(
            set(household_ids)
        ):
            raise ShelterValidationError(
                "Duplicate household ID"
            )

        if len(shelter_ids) != len(
            set(shelter_ids)
        ):
            raise ShelterValidationError(
                "Duplicate shelter ID"
            )

        priorities = {}

        for household in request.households:
            if (
                household.district.casefold()
                != request.district.casefold()
            ):
                raise ShelterValidationError(
                    "Household district mismatch"
                )

            priorities[
                household.household_id
            ] = self.priority_engine.assess(
                household
            )

        eligible_shelters, screening = (
            self.eligibility_engine.screen(
                request.shelters,
                request.district,
                now=now,
            )
        )

        allocations, unallocated, summaries = (
            self.allocation_engine.allocate(
                request.households,
                eligible_shelters,
                priorities,
            )
        )

        warnings = list(
            screening.warnings
        )

        if unallocated:
            warnings.append(
                "Some households require urgent "
                "manual shelter coordination"
            )

        warnings.extend([
            "All shelter allocations are provisional",
            "No shelter beds or spaces have been reserved",
            "Actual admission requires authorized confirmation",
            "Travel route and transport safety are not assessed",
            "Medical needs require qualified clinical review",
            "Shelter suitability requires on-site verification",
            "Human approval is mandatory",
        ])

        plan = ShelterAllocationPlan(
            plan_id=generate_id(),
            request_id=request.request_id,
            district=request.district,
            priority_assessments=tuple(
                priorities.values()
            ),
            shelter_screening=screening,
            allocations=allocations,
            unallocated_households=unallocated,
            capacity_summaries=summaries,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            status=ShelterPlanStatus.PENDING_REVIEW,
            human_approval_required=True,
            created_at=now,
        )

        with self._lock:
            self._plans[
                plan.plan_id
            ] = plan

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id="shelter_allocation_agent",
            action="allocation_plan_created",
            details={
                "district": request.district,
                "household_count": str(
                    len(request.households)
                ),
                "allocation_count": str(
                    len(allocations)
                ),
                "unallocated_count": str(
                    len(unallocated)
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
    ) -> ShelterAllocationPlan:
        with self._lock:
            plan = self._plans.get(
                plan_id
            )

            if plan is None:
                raise ShelterApprovalError(
                    "Allocation plan not found"
                )

            if plan.status != (
                ShelterPlanStatus.PENDING_REVIEW
            ):
                raise ShelterApprovalError(
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
                    ShelterPlanStatus.APPROVED_FOR_COORDINATION
                )

                allocation_status = (
                    AllocationStatus.APPROVED_FOR_COORDINATION
                )

            elif decision == ReviewDecision.REJECT:
                plan_status = (
                    ShelterPlanStatus.REJECTED
                )

                allocation_status = (
                    AllocationStatus.REJECTED
                )

            else:
                plan_status = (
                    ShelterPlanStatus.REVISION_REQUIRED
                )

                allocation_status = (
                    AllocationStatus.REQUIRES_REVISION
                )

            updated_allocations = tuple(
                replace(
                    allocation,
                    status=allocation_status,
                )
                for allocation in plan.allocations
            )

            updated = replace(
                plan,
                status=plan_status,
                allocations=updated_allocations,
            )

            self._plans[
                plan_id
            ] = updated

            return updated

    def get_plan(
        self,
        plan_id: str,
    ) -> ShelterAllocationPlan | None:
        with self._lock:
            return self._plans.get(
                plan_id
            )

    def get_audit_history(
        self,
        plan_id: str,
    ) -> tuple[ShelterAuditRecord, ...]:
        return self.audit_logger.list_records(
            plan_id
        )


# ============================================================
# JSON EXPORTER
# ============================================================

class ShelterAllocationReportExporter:
    @staticmethod
    def export_json(
        plan: ShelterAllocationPlan,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise ShelterExportError(
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

def create_shelter_allocation_agent(
    config: ShelterAllocationConfig | None = None,
) -> ShelterAllocationAgent:
    return ShelterAllocationAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional data for demonstration only.
    # Not verified Kerala shelter information.

    households = (
        DisplacedHousehold(
            household_id="DEMO-HOUSEHOLD-001",
            district="Thrissur",
            current_location="Illustrative Flood Zone A",
            family_size=4,
            children_count=2,
            older_adults_count=1,
            immediate_shelter_need=True,
            requested_at=now - timedelta(minutes=15),
        ),
        DisplacedHousehold(
            household_id="DEMO-HOUSEHOLD-002",
            district="Thrissur",
            current_location="Illustrative Flood Zone B",
            family_size=3,
            medically_vulnerable_count=1,
            medical_requirements=(
                MedicalSupportRequirement.BASIC_FIRST_AID,
            ),
            requested_at=now - timedelta(minutes=10),
        ),
    )

    shelters = (
        EmergencyShelter(
            shelter_id="DEMO-SHELTER-001",
            shelter_name="Illustrative Relief Camp A",
            district="Thrissur",
            location_description="Demo Location A",
            status=ShelterStatus.VERIFIED_OPERATIONAL,
            total_capacity=100,
            current_occupancy=50,
            facilities=(
                ShelterFacility.DRINKING_WATER,
                ShelterFacility.SANITATION,
                ShelterFacility.ELECTRICITY,
                ShelterFacility.FOOD_SERVICE,
                ShelterFacility.FIRST_AID,
            ),
            reliability=0.95,
            status_verified_at=now - timedelta(minutes=10),
        ),
        EmergencyShelter(
            shelter_id="DEMO-SHELTER-002",
            shelter_name="Illustrative Relief Camp B",
            district="Thrissur",
            location_description="Demo Location B",
            status=ShelterStatus.VERIFIED_OPERATIONAL,
            total_capacity=80,
            current_occupancy=30,
            facilities=(
                ShelterFacility.DRINKING_WATER,
                ShelterFacility.SANITATION,
                ShelterFacility.ELECTRICITY,
                ShelterFacility.FOOD_SERVICE,
                ShelterFacility.FIRST_AID,
                ShelterFacility.MEDICAL_STAFF,
                ShelterFacility.STEP_FREE_ACCESS,
            ),
            reliability=0.93,
            status_verified_at=now - timedelta(minutes=8),
        ),
    )

    request = ShelterAllocationRequest(
        request_id="DEMO-SHELTER-REQUEST-001",
        district="Thrissur",
        households=households,
        shelters=shelters,
        created_at=now,
    )

    agent = create_shelter_allocation_agent()

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
        reviewer_id="DEMO-SHELTER-COORDINATOR",
        reviewer_role="shelter_coordinator",
        decision=ReviewDecision.APPROVE,
        reason="Illustrative coordination review",
    )

    print(
        "Plan status:",
        reviewed.status.value,
    )

    ShelterAllocationReportExporter.export_json(
        reviewed,
        "outputs/agents/shelter_allocation_report.json",
    )

    print(
        "No real shelter reservation or "
        "household relocation occurred."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ShelterAllocationError",
    "ShelterValidationError",
    "ShelterCapacityError",
    "ShelterApprovalError",
    "ShelterExportError",
    "ShelterStatus",
    "HouseholdPriority",
    "AccessibilityRequirement",
    "MedicalSupportRequirement",
    "ShelterFacility",
    "AllocationStatus",
    "ShelterPlanStatus",
    "ReviewDecision",
    "ShelterAllocationConfig",
    "DisplacedHousehold",
    "EmergencyShelter",
    "ShelterAllocationRequest",
    "HouseholdPriorityAssessment",
    "ShelterScreeningResult",
    "ShelterAllocationProposal",
    "UnallocatedHousehold",
    "ShelterCapacitySummary",
    "ShelterAllocationPlan",
    "ShelterApprovalRecord",
    "ShelterAuditRecord",
    "HouseholdPriorityEngine",
    "ShelterEligibilityEngine",
    "ShelterFacilityMatchingEngine",
    "ShelterCapacityEngine",
    "ShelterAllocationEngine",
    "ShelterAllocationAuditLogger",
    "ShelterHumanApprovalManager",
    "ShelterAllocationAgent",
    "ShelterAllocationReportExporter",
    "create_shelter_allocation_agent",
]
