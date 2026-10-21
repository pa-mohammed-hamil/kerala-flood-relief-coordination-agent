"""Placeholder module: ai/agents/rescue_coordinator_agent.py; implementation required."""

"""
backend/ai/agents/rescue_coordinator_agent.py

Kerala Flood Relief Coordination Agent
Rescue Coordination & Mission Planning Agent

Python 3.11+
Standard library only.

FEATURES
--------
1. SOS incident validation
2. Flood rescue priority assessment
3. Vulnerable-person prioritization
4. Rescue-team capability matching
5. Team availability verification
6. Route safety evidence validation
7. Provisional rescue mission planning
8. Prevention of duplicate team assignment
9. Unassigned incident reporting
10. Human authorization workflow
11. Mission approval audit trail
12. JSON report export

IMPORTANT SAFETY RESTRICTIONS
-----------------------------
- This agent does not dispatch rescue teams.
- This agent does not send SOS messages.
- This agent does not operate rescue vehicles.
- This agent does not establish that a route is safe.
- This agent does not approve entry into floodwater.
- All assignments are advisory proposals.
- A qualified incident commander must authorize
  operational dispatch.
- Live conditions must be checked immediately
  before any real-world movement.
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

class RescueCoordinatorError(Exception):
    pass


class RescueValidationError(RescueCoordinatorError):
    pass


class RescueAssignmentError(RescueCoordinatorError):
    pass


class RescueApprovalError(RescueCoordinatorError):
    pass


class RescueExportError(RescueCoordinatorError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RescueIncidentType(str, Enum):
    FLOOD_ENTRAPMENT = "flood_entrapment"
    ROOFTOP_RESCUE = "rooftop_rescue"
    WATER_RESCUE = "water_rescue"
    EVACUATION_ASSISTANCE = "evacuation_assistance"
    MEDICAL_EVACUATION = "medical_evacuation"
    LANDSLIDE_RELATED = "landslide_related"
    MISSING_PERSON_SEARCH = "missing_person_search"
    OTHER = "other"


class RescuePriority(str, Enum):
    ROUTINE_REVIEW = "routine_review"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentVerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"


class RescueTeamStatus(str, Enum):
    VERIFIED_AVAILABLE = "verified_available"
    ASSIGNED = "assigned"
    OFFLINE = "offline"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class RescueCapability(str, Enum):
    BOAT_RESCUE = "boat_rescue"
    SWIFT_WATER_RESCUE = "swift_water_rescue"
    MEDICAL_RESPONSE = "medical_response"
    LAND_RESCUE = "land_rescue"
    HIGH_ANGLE_RESCUE = "high_angle_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    ACCESSIBLE_EVACUATION = "accessible_evacuation"


class RescueRouteStatus(str, Enum):
    VERIFIED_PASSABLE = "verified_passable"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class RescueMissionStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_HUMAN_APPROVAL = "pending_human_approval"
    APPROVED_FOR_COORDINATION = "approved_for_coordination"
    REJECTED = "rejected"
    REQUIRES_REASSESSMENT = "requires_reassessment"


class RescuePlanStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED_FOR_COORDINATION = "approved_for_coordination"
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"


class ApprovalDecision(str, Enum):
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
        raise RescueValidationError(
            "Timezone-aware datetime required"
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise RescueValidationError(
            f"{name} must be between 0 and 1"
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
        raise RescueValidationError(
            f"{name} must be nonnegative"
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
        return [serialize(item) for item in value]

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RescueCoordinatorConfig:
    max_incidents: int = 10000
    max_teams: int = 5000
    max_routes: int = 20000

    max_team_status_age_minutes: int = 30
    max_route_verification_age_minutes: int = 20
    max_incident_verification_age_minutes: int = 60

    minimum_team_readiness: float = 0.70

    require_verified_routes: bool = True
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_incidents",
            "max_teams",
            "max_routes",
            "max_team_status_age_minutes",
            "max_route_verification_age_minutes",
            "max_incident_verification_age_minutes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RescueValidationError(
                    f"{name} must be positive integer"
                )

        validate_probability(
            self.minimum_team_readiness,
            "minimum_team_readiness",
        )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class RescueIncident:
    incident_id: str

    district: str
    location_description: str

    incident_type: RescueIncidentType

    people_at_risk: int

    children_count: int = 0
    older_adults_count: int = 0
    medically_vulnerable_count: int = 0

    immediate_life_threat: bool = False

    verification_status: IncidentVerificationStatus = (
        IncidentVerificationStatus.UNVERIFIED
    )

    verified_at: datetime | None = None
    reported_at: datetime | None = None

    required_capabilities: tuple[
        RescueCapability,
        ...
    ] = ()

    def __post_init__(self) -> None:
        for name in (
            "incident_id",
            "district",
            "location_description",
        ):
            if not getattr(self, name).strip():
                raise RescueValidationError(
                    f"{name} required"
                )

        if (
            isinstance(self.people_at_risk, bool)
            or not isinstance(self.people_at_risk, int)
            or self.people_at_risk <= 0
        ):
            raise RescueValidationError(
                "people_at_risk must be positive integer"
            )

        for name in (
            "children_count",
            "older_adults_count",
            "medically_vulnerable_count",
        ):
            value = getattr(self, name)

            validate_nonnegative(
                value,
                name,
            )

            if (
                not isinstance(value, int)
                or value > self.people_at_risk
            ):
                raise RescueValidationError(
                    f"Invalid {name}"
                )

        for timestamp in (
            self.verified_at,
            self.reported_at,
        ):
            if timestamp is not None:
                validate_datetime(timestamp)


@dataclass(frozen=True)
class RescueTeam:
    team_id: str
    team_name: str

    district: str

    status: RescueTeamStatus

    capabilities: tuple[
        RescueCapability,
        ...
    ]

    maximum_person_capacity: int

    readiness_score: float

    status_verified_at: datetime | None

    def __post_init__(self) -> None:
        if not self.team_id.strip():
            raise RescueValidationError(
                "team_id required"
            )

        if not self.district.strip():
            raise RescueValidationError(
                "district required"
            )

        if (
            isinstance(self.maximum_person_capacity, bool)
            or not isinstance(self.maximum_person_capacity, int)
            or self.maximum_person_capacity <= 0
        ):
            raise RescueValidationError(
                "maximum_person_capacity must be positive"
            )

        validate_probability(
            self.readiness_score,
            "readiness_score",
        )

        if self.status_verified_at is not None:
            validate_datetime(
                self.status_verified_at
            )


@dataclass(frozen=True)
class RescueRoute:
    route_id: str

    team_id: str
    incident_id: str

    status: RescueRouteStatus

    estimated_travel_minutes: float

    verified_at: datetime | None

    def __post_init__(self) -> None:
        for name in (
            "route_id",
            "team_id",
            "incident_id",
        ):
            if not getattr(self, name).strip():
                raise RescueValidationError(
                    f"{name} required"
                )

        validate_nonnegative(
            self.estimated_travel_minutes,
            "estimated_travel_minutes",
        )

        if self.verified_at is not None:
            validate_datetime(
                self.verified_at
            )


@dataclass(frozen=True)
class RescueCoordinationRequest:
    request_id: str

    district: str

    incidents: tuple[RescueIncident, ...]
    teams: tuple[RescueTeam, ...]
    routes: tuple[RescueRoute, ...]

    created_at: datetime

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise RescueValidationError(
                "request_id required"
            )

        if not self.district.strip():
            raise RescueValidationError(
                "district required"
            )

        validate_datetime(
            self.created_at
        )


@dataclass(frozen=True)
class RescuePriorityAssessment:
    incident_id: str

    score: float
    priority: RescuePriority

    factors: Mapping[str, float]

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RescueMissionProposal:
    mission_id: str

    incident_id: str
    team_id: str
    route_id: str

    priority: RescuePriority

    estimated_travel_minutes: float

    status: RescueMissionStatus

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class UnassignedRescueIncident:
    incident_id: str

    priority: RescuePriority

    reason: str


@dataclass(frozen=True)
class RescueCoordinationPlan:
    plan_id: str
    request_id: str

    district: str

    priority_assessments: tuple[
        RescuePriorityAssessment,
        ...
    ]

    mission_proposals: tuple[
        RescueMissionProposal,
        ...
    ]

    unassigned_incidents: tuple[
        UnassignedRescueIncident,
        ...
    ]

    warnings: tuple[str, ...]

    status: RescuePlanStatus

    human_approval_required: bool

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RescueApprovalRecord:
    approval_id: str
    plan_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ApprovalDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class RescueAuditRecord:
    audit_id: str
    plan_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# INCIDENT PRIORITY ENGINE
# ============================================================

class RescuePriorityEngine:
    INCIDENT_RISK = {
        RescueIncidentType.FLOOD_ENTRAPMENT: 0.90,
        RescueIncidentType.ROOFTOP_RESCUE: 0.95,
        RescueIncidentType.WATER_RESCUE: 1.00,
        RescueIncidentType.EVACUATION_ASSISTANCE: 0.55,
        RescueIncidentType.MEDICAL_EVACUATION: 0.90,
        RescueIncidentType.LANDSLIDE_RELATED: 0.90,
        RescueIncidentType.MISSING_PERSON_SEARCH: 0.75,
        RescueIncidentType.OTHER: 0.50,
    }

    @classmethod
    def assess(
        cls,
        incident: RescueIncident,
    ) -> RescuePriorityAssessment:
        risk = cls.INCIDENT_RISK[
            incident.incident_type
        ]

        population = min(
            incident.people_at_risk / 100,
            1.0,
        )

        vulnerable_count = min(
            incident.people_at_risk,
            incident.children_count
            + incident.older_adults_count
            + incident.medically_vulnerable_count,
        )

        vulnerability = (
            vulnerable_count
            / incident.people_at_risk
        )

        life_threat = float(
            incident.immediate_life_threat
        )

        score = (
            0.25 * risk
            + 0.15 * population
            + 0.20 * vulnerability
            + 0.40 * life_threat
        )

        score = round(
            min(score, 1.0),
            4,
        )

        if incident.immediate_life_threat:
            priority = RescuePriority.CRITICAL
        elif score >= 0.75:
            priority = RescuePriority.CRITICAL
        elif score >= 0.55:
            priority = RescuePriority.HIGH
        elif score >= 0.35:
            priority = RescuePriority.ELEVATED
        else:
            priority = RescuePriority.ROUTINE_REVIEW

        warnings = [
            "Priority is a heuristic triage recommendation",
            "Incident commander review is required",
        ]

        if incident.verification_status != (
            IncidentVerificationStatus.VERIFIED
        ):
            warnings.append(
                "Incident information is not fully verified"
            )

        return RescuePriorityAssessment(
            incident_id=incident.incident_id,
            score=score,
            priority=priority,
            factors={
                "incident_risk": risk,
                "population": population,
                "vulnerability": vulnerability,
                "immediate_life_threat": life_threat,
            },
            warnings=tuple(warnings),
        )


# ============================================================
# TEAM ELIGIBILITY ENGINE
# ============================================================

class RescueTeamEligibilityEngine:
    def __init__(
        self,
        config: RescueCoordinatorConfig,
    ) -> None:
        self.config = config

    def eligible(
        self,
        team: RescueTeam,
        incident: RescueIncident,
        *,
        now: datetime,
    ) -> bool:
        if team.status != (
            RescueTeamStatus.VERIFIED_AVAILABLE
        ):
            return False

        if team.status_verified_at is None:
            return False

        if team.status_verified_at > now:
            return False

        if (
            now - team.status_verified_at
        ) > timedelta(
            minutes=(
                self.config.max_team_status_age_minutes
            )
        ):
            return False

        if team.readiness_score < (
            self.config.minimum_team_readiness
        ):
            return False

        if (
            team.district.casefold()
            != incident.district.casefold()
        ):
            return False

        if not set(
            incident.required_capabilities
        ).issubset(set(team.capabilities)):
            return False

        # A mission may require multiple trips
        # or multiple teams. This basic planner
        # accepts only single-team capacity fits.
        if (
            team.maximum_person_capacity
            < incident.people_at_risk
        ):
            return False

        return True


# ============================================================
# ROUTE VALIDATION ENGINE
# ============================================================

class RescueRouteValidationEngine:
    def __init__(
        self,
        config: RescueCoordinatorConfig,
    ) -> None:
        self.config = config

    def eligible(
        self,
        route: RescueRoute,
        *,
        now: datetime,
    ) -> bool:
        if route.status != (
            RescueRouteStatus.VERIFIED_PASSABLE
        ):
            return False

        if route.verified_at is None:
            return False

        if route.verified_at > now:
            return False

        if (
            now - route.verified_at
        ) > timedelta(
            minutes=(
                self.config.max_route_verification_age_minutes
            )
        ):
            return False

        return True


# ============================================================
# RESCUE ASSIGNMENT ENGINE
# ============================================================

class RescueAssignmentEngine:
    """
    Greedy one-team-per-incident planning.

    Does not:
        - Perform multi-team rescue optimization
        - Schedule sequential missions
        - Validate water conditions
        - Dispatch personnel

    Each team is used at most once per plan.
    """

    def __init__(
        self,
        team_engine: RescueTeamEligibilityEngine,
        route_engine: RescueRouteValidationEngine,
    ) -> None:
        self.team_engine = team_engine
        self.route_engine = route_engine

    def assign(
        self,
        *,
        incidents: Sequence[RescueIncident],
        teams: Sequence[RescueTeam],
        routes: Sequence[RescueRoute],
        priorities: Mapping[
            str,
            RescuePriorityAssessment,
        ],
        now: datetime,
    ) -> tuple[
        tuple[RescueMissionProposal, ...],
        tuple[UnassignedRescueIncident, ...],
    ]:
        available_teams = {
            team.team_id: team
            for team in teams
        }

        route_map = {}

        for route in routes:
            route_map.setdefault(
                (
                    route.team_id,
                    route.incident_id,
                ),
                [],
            ).append(route)

        ordered_incidents = sorted(
            incidents,
            key=lambda incident: (
                -priorities[
                    incident.incident_id
                ].score,
                incident.reported_at or now,
                incident.incident_id,
            ),
        )

        missions = []
        unassigned = []

        for incident in ordered_incidents:
            candidates = []

            for team in available_teams.values():
                if not self.team_engine.eligible(
                    team,
                    incident,
                    now=now,
                ):
                    continue

                matching_routes = route_map.get(
                    (
                        team.team_id,
                        incident.incident_id,
                    ),
                    [],
                )

                for route in matching_routes:
                    if not self.route_engine.eligible(
                        route,
                        now=now,
                    ):
                        continue

                    candidates.append(
                        (
                            route.estimated_travel_minutes,
                            -team.readiness_score,
                            team.team_id,
                            route.route_id,
                            team,
                            route,
                        )
                    )

            if not candidates:
                unassigned.append(
                    UnassignedRescueIncident(
                        incident_id=incident.incident_id,
                        priority=priorities[
                            incident.incident_id
                        ].priority,
                        reason=(
                            "No available team satisfies "
                            "capability, capacity, readiness "
                            "and verified-route requirements"
                        ),
                    )
                )
                continue

            candidates.sort(
                key=lambda item: item[:4]
            )

            _, _, _, _, selected_team, selected_route = (
                candidates[0]
            )

            missions.append(
                RescueMissionProposal(
                    mission_id=generate_id(),
                    incident_id=incident.incident_id,
                    team_id=selected_team.team_id,
                    route_id=selected_route.route_id,
                    priority=priorities[
                        incident.incident_id
                    ].priority,
                    estimated_travel_minutes=(
                        selected_route.estimated_travel_minutes
                    ),
                    status=(
                        RescueMissionStatus.PENDING_HUMAN_APPROVAL
                    ),
                    warnings=(
                        "Advisory assignment only",
                        "No dispatch has occurred",
                        "Recheck conditions before movement",
                        "Incident commander authorization required",
                    ),
                )
            )

            del available_teams[
                selected_team.team_id
            ]

        return (
            tuple(missions),
            tuple(unassigned),
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class RescueCoordinationAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            RescueAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        plan_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> RescueAuditRecord:
        record = RescueAuditRecord(
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
    ) -> tuple[RescueAuditRecord, ...]:
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

class RescueHumanApprovalManager:
    ALLOWED_ROLES = {
        "incident_commander",
        "district_disaster_management_officer",
        "authorized_rescue_coordinator",
    }

    def __init__(
        self,
        audit_logger: RescueCoordinationAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            RescueApprovalRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        plan: RescueCoordinationPlan,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ApprovalDecision,
        reason: str,
    ) -> RescueApprovalRecord:
        if not reviewer_id.strip():
            raise RescueApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise RescueApprovalError(
                "Reviewer is not authorized"
            )

        if not reason.strip():
            raise RescueApprovalError(
                "Review reason required"
            )

        with self._lock:
            if plan.plan_id in self._reviews:
                raise RescueApprovalError(
                    "Plan already reviewed"
                )

            review = RescueApprovalRecord(
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
            ] = review

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id=reviewer_id,
            action=f"plan_review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN RESCUE COORDINATOR AGENT
# ============================================================

class RescueCoordinatorAgent:
    def __init__(
        self,
        config: RescueCoordinatorConfig | None = None,
    ) -> None:
        self.config = (
            config or RescueCoordinatorConfig()
        )

        self.priority_engine = (
            RescuePriorityEngine()
        )

        self.team_engine = (
            RescueTeamEligibilityEngine(
                self.config
            )
        )

        self.route_engine = (
            RescueRouteValidationEngine(
                self.config
            )
        )

        self.assignment_engine = (
            RescueAssignmentEngine(
                self.team_engine,
                self.route_engine,
            )
        )

        self.audit_logger = (
            RescueCoordinationAuditLogger()
        )

        self.approval_manager = (
            RescueHumanApprovalManager(
                self.audit_logger
            )
        )

        self._plans: dict[
            str,
            RescueCoordinationPlan,
        ] = {}

        self._lock = threading.RLock()

    def run(
        self,
        request: RescueCoordinationRequest,
        *,
        now: datetime | None = None,
    ) -> RescueCoordinationPlan:
        now = now or utc_now()

        validate_datetime(now)

        if request.created_at > now:
            raise RescueValidationError(
                "Request timestamp is in the future"
            )

        if len(request.incidents) > (
            self.config.max_incidents
        ):
            raise RescueValidationError(
                "Incident limit exceeded"
            )

        if len(request.teams) > (
            self.config.max_teams
        ):
            raise RescueValidationError(
                "Team limit exceeded"
            )

        if len(request.routes) > (
            self.config.max_routes
        ):
            raise RescueValidationError(
                "Route limit exceeded"
            )

        for collection, attribute in (
            (request.incidents, "incident_id"),
            (request.teams, "team_id"),
            (request.routes, "route_id"),
        ):
            ids = [
                getattr(item, attribute)
                for item in collection
            ]

            if len(ids) != len(set(ids)):
                raise RescueValidationError(
                    f"Duplicate {attribute}"
                )

        priorities = {}

        warnings = []

        for incident in request.incidents:
            if (
                incident.district.casefold()
                != request.district.casefold()
            ):
                raise RescueValidationError(
                    "Incident district mismatch"
                )

            assessment = (
                self.priority_engine.assess(
                    incident
                )
            )

            priorities[
                incident.incident_id
            ] = assessment

            if incident.immediate_life_threat:
                warnings.append(
                    f"Incident {incident.incident_id} "
                    "requires immediate human emergency escalation"
                )

            if (
                incident.verification_status
                != IncidentVerificationStatus.VERIFIED
            ):
                warnings.append(
                    f"Incident {incident.incident_id} "
                    "requires source verification"
                )

        missions, unassigned = (
            self.assignment_engine.assign(
                incidents=request.incidents,
                teams=request.teams,
                routes=request.routes,
                priorities=priorities,
                now=now,
            )
        )

        if unassigned:
            warnings.append(
                "Some incidents have no suitable provisional assignment"
            )

        warnings.extend([
            "This is an advisory coordination plan",
            "No rescue team has been dispatched",
            "No route is guaranteed safe",
            "Travel times are supplied estimates",
            "Operational conditions must be reverified",
            "Human incident-command approval is mandatory",
            "Mission approval does not automatically authorize dispatch",
        ])

        plan = RescueCoordinationPlan(
            plan_id=generate_id(),
            request_id=request.request_id,
            district=request.district,
            priority_assessments=tuple(
                priorities.values()
            ),
            mission_proposals=missions,
            unassigned_incidents=unassigned,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            status=RescuePlanStatus.PENDING_APPROVAL,
            human_approval_required=True,
            created_at=now,
        )

        with self._lock:
            self._plans[
                plan.plan_id
            ] = plan

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id="rescue_coordinator_agent",
            action="coordination_plan_created",
            details={
                "district": request.district,
                "incident_count": str(
                    len(request.incidents)
                ),
                "mission_count": str(
                    len(missions)
                ),
                "unassigned_count": str(
                    len(unassigned)
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
        decision: ApprovalDecision,
        reason: str,
    ) -> RescueCoordinationPlan:
        with self._lock:
            plan = self._plans.get(
                plan_id
            )

            if plan is None:
                raise RescueApprovalError(
                    "Plan not found"
                )

            if plan.status != (
                RescuePlanStatus.PENDING_APPROVAL
            ):
                raise RescueApprovalError(
                    "Plan is not pending review"
                )

            self.approval_manager.review(
                plan,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            if decision == ApprovalDecision.APPROVE:
                plan_status = (
                    RescuePlanStatus.APPROVED_FOR_COORDINATION
                )

                mission_status = (
                    RescueMissionStatus.APPROVED_FOR_COORDINATION
                )

            elif decision == ApprovalDecision.REJECT:
                plan_status = (
                    RescuePlanStatus.REJECTED
                )

                mission_status = (
                    RescueMissionStatus.REJECTED
                )

            else:
                plan_status = (
                    RescuePlanStatus.REVISION_REQUIRED
                )

                mission_status = (
                    RescueMissionStatus.REQUIRES_REASSESSMENT
                )

            updated_missions = tuple(
                replace(
                    mission,
                    status=mission_status,
                )
                for mission in plan.mission_proposals
            )

            updated = replace(
                plan,
                status=plan_status,
                mission_proposals=updated_missions,
            )

            self._plans[
                plan_id
            ] = updated

            return updated

    def get_plan(
        self,
        plan_id: str,
    ) -> RescueCoordinationPlan | None:
        with self._lock:
            return self._plans.get(
                plan_id
            )

    def get_audit_history(
        self,
        plan_id: str,
    ) -> tuple[RescueAuditRecord, ...]:
        return self.audit_logger.list_records(
            plan_id
        )


# ============================================================
# REPORT EXPORT
# ============================================================

class RescueCoordinationReportExporter:
    @staticmethod
    def export_json(
        plan: RescueCoordinationPlan,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise RescueExportError(
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

def create_rescue_coordinator_agent(
    config: RescueCoordinatorConfig | None = None,
) -> RescueCoordinatorAgent:
    return RescueCoordinatorAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional data only.
    incident = RescueIncident(
        incident_id="DEMO-SOS-001",
        district="Thrissur",
        location_description="Illustrative flood-affected area",
        incident_type=RescueIncidentType.FLOOD_ENTRAPMENT,
        people_at_risk=4,
        children_count=1,
        older_adults_count=1,
        immediate_life_threat=True,
        verification_status=(
            IncidentVerificationStatus.VERIFIED
        ),
        verified_at=now - timedelta(minutes=5),
        reported_at=now - timedelta(minutes=10),
        required_capabilities=(
            RescueCapability.BOAT_RESCUE,
        ),
    )

    team = RescueTeam(
        team_id="DEMO-TEAM-001",
        team_name="Illustrative Rescue Team",
        district="Thrissur",
        status=RescueTeamStatus.VERIFIED_AVAILABLE,
        capabilities=(
            RescueCapability.BOAT_RESCUE,
            RescueCapability.MEDICAL_RESPONSE,
        ),
        maximum_person_capacity=6,
        readiness_score=0.95,
        status_verified_at=now - timedelta(minutes=3),
    )

    route = RescueRoute(
        route_id="DEMO-ROUTE-001",
        team_id=team.team_id,
        incident_id=incident.incident_id,
        status=RescueRouteStatus.VERIFIED_PASSABLE,
        estimated_travel_minutes=18,
        verified_at=now - timedelta(minutes=2),
    )

    request = RescueCoordinationRequest(
        request_id="DEMO-RESCUE-PLAN-001",
        district="Thrissur",
        incidents=(incident,),
        teams=(team,),
        routes=(route,),
        created_at=now,
    )

    agent = create_rescue_coordinator_agent()

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
        reviewer_id="DEMO-COMMANDER-001",
        reviewer_role="incident_commander",
        decision=ApprovalDecision.APPROVE,
        reason="Illustrative coordination review",
    )

    print(
        "Plan status:",
        reviewed.status.value,
    )

    RescueCoordinationReportExporter.export_json(
        reviewed,
        "outputs/agents/rescue_coordination_report.json",
    )

    print(
        "No actual rescue dispatch was performed."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RescueCoordinatorError",
    "RescueValidationError",
    "RescueAssignmentError",
    "RescueApprovalError",
    "RescueExportError",
    "RescueIncidentType",
    "RescuePriority",
    "IncidentVerificationStatus",
    "RescueTeamStatus",
    "RescueCapability",
    "RescueRouteStatus",
    "RescueMissionStatus",
    "RescuePlanStatus",
    "ApprovalDecision",
    "RescueCoordinatorConfig",
    "RescueIncident",
    "RescueTeam",
    "RescueRoute",
    "RescueCoordinationRequest",
    "RescuePriorityAssessment",
    "RescueMissionProposal",
    "UnassignedRescueIncident",
    "RescueCoordinationPlan",
    "RescueApprovalRecord",
    "RescueAuditRecord",
    "RescuePriorityEngine",
    "RescueTeamEligibilityEngine",
    "RescueRouteValidationEngine",
    "RescueAssignmentEngine",
    "RescueCoordinationAuditLogger",
    "RescueHumanApprovalManager",
    "RescueCoordinatorAgent",
    "RescueCoordinationReportExporter",
    "create_rescue_coordinator_agent",
]
