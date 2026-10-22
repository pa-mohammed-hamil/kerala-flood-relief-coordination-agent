from __future__ import annotations

"""
backend/services/rescue_service.py

Kerala Flood Relief Coordination Agent
=======================================

Rescue Operations Management Service

Features:
- Rescue request registration
- SOS and incident linkage
- Rescue priority and risk assessment
- Rescue team eligibility checks
- Rescue mission planning
- Human approval workflow
- Rescue team assignment
- Dispatch coordination
- Mission acknowledgement
- Real-time mission progress updates
- People rescued and evacuation tracking
- Rescue safety checks
- Mission escalation
- Mission pause and resumption
- Rescue completion verification
- Cancellation and audit logging
- District rescue statistics
- Structured rescue reports

IMPORTANT:
- AI recommendations are advisory only.
- Dispatch requires authorization and human approval.
- A mission marked DISPATCHED does not prove arrival.
- Rescue counts must come from verified field reports.
- This module does not directly contact emergency services.
- Production requires durable transactions, queue workers,
  GPS integrations, idempotency and field verification.
"""


import logging
import math
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class RescuePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RescueStatus(str, Enum):
    REQUESTED = "requested"
    TRIAGED = "triaged"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    TEAM_ASSIGNED = "team_assigned"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    PENDING_VERIFICATION = "pending_verification"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RescueType(str, Enum):
    FLOOD_RESCUE = "flood_rescue"
    WATER_RESCUE = "water_rescue"
    BOAT_RESCUE = "boat_rescue"
    ROOFTOP_RESCUE = "rooftop_rescue"
    BUILDING_EVACUATION = "building_evacuation"
    MEDICAL_EVACUATION = "medical_evacuation"
    VEHICLE_RESCUE = "vehicle_rescue"
    LANDSLIDE_RESCUE = "landslide_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    OTHER = "other"


class RescueRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class RescueTeamCapability(str, Enum):
    BOAT = "boat"
    SWIFT_WATER = "swift_water"
    DIVING = "diving"
    ROPE_RESCUE = "rope_rescue"
    MEDICAL = "medical"
    AMBULANCE = "ambulance"
    SEARCH = "search"
    HEAVY_EQUIPMENT = "heavy_equipment"
    LAND_TRANSPORT = "land_transport"
    NIGHT_OPERATION = "night_operation"


class RescueTeamStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    DEPLOYED = "deployed"
    UNAVAILABLE = "unavailable"
    OFF_DUTY = "off_duty"


class RescueEventType(str, Enum):
    CREATED = "created"
    TRIAGED = "triaged"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    TEAM_ASSIGNED = "team_assigned"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    EN_ROUTE = "en_route"
    ARRIVED = "arrived"
    STARTED = "started"
    PROGRESS_UPDATED = "progress_updated"
    PAUSED = "paused"
    RESUMED = "resumed"
    ESCALATED = "escalated"
    COMPLETION_REQUESTED = "completion_requested"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ============================================================
# EXCEPTIONS
# ============================================================

class RescueServiceError(Exception):
    pass


class RescueValidationError(RescueServiceError):
    pass


class RescueNotFoundError(RescueServiceError):
    pass


class RescueConflictError(RescueServiceError):
    pass


class RescueAuthorizationError(RescueServiceError):
    pass


class RescueSafetyError(RescueServiceError):
    pass


class RescueIntegrationError(RescueServiceError):
    pass


class RescueVerificationError(RescueServiceError):
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
    max_length: int = 2000,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RescueValidationError(
            f"{name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise RescueValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RescueValidationError(
            f"{name} must be timezone-aware"
        )


def validate_non_negative(
    value: int,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RescueValidationError(
            f"{name} must be an integer"
        )

    if value < 0:
        raise RescueValidationError(
            f"{name} cannot be negative"
        )


def validate_progress(value: float) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 100
    ):
        raise RescueValidationError(
            "Progress must be between 0 and 100"
        )


# ============================================================
# DOMAIN MODELS
# ============================================================

@dataclass(frozen=True)
class RescueLocation:
    district_id: int
    latitude: float
    longitude: float

    address: str = ""
    landmark: str = ""
    panchayat: str = ""

    def __post_init__(self) -> None:
        if self.district_id <= 0:
            raise RescueValidationError(
                "Invalid district ID"
            )

        if (
            not math.isfinite(self.latitude)
            or not -90 <= self.latitude <= 90
        ):
            raise RescueValidationError(
                "Invalid latitude"
            )

        if (
            not math.isfinite(self.longitude)
            or not -180 <= self.longitude <= 180
        ):
            raise RescueValidationError(
                "Invalid longitude"
            )


@dataclass(frozen=True)
class RescuePopulation:
    total_people: int

    children: int = 0
    elderly: int = 0
    pregnant_people: int = 0
    disabled_people: int = 0

    injured_people: int = 0
    critical_patients: int = 0

    def __post_init__(self) -> None:
        for name in (
            "total_people",
            "children",
            "elderly",
            "pregnant_people",
            "disabled_people",
            "injured_people",
            "critical_patients",
        ):
            validate_non_negative(
                getattr(self, name),
                name,
            )

        if self.total_people <= 0:
            raise RescueValidationError(
                "At least one person must need rescue"
            )

        # Categories may overlap; therefore their
        # sum is not used as a population total.
        for name in (
            "children",
            "elderly",
            "pregnant_people",
            "disabled_people",
            "injured_people",
            "critical_patients",
        ):
            if getattr(self, name) > self.total_people:
                raise RescueValidationError(
                    f"{name} exceeds total_people"
                )


@dataclass(frozen=True)
class RescueCreateRequest:
    district_id: int

    rescue_type: RescueType
    location: RescueLocation
    population: RescuePopulation

    description: str

    incident_id: str | None = None
    sos_request_id: str | None = None

    water_depth_meters: float | None = None
    water_current_risk: bool = False

    building_instability: bool = False
    electrical_hazard: bool = False
    landslide_risk: bool = False

    medical_emergency: bool = False

    reported_by: int | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RescueRiskAssessment:
    priority: RescuePriority
    risk_level: RescueRiskLevel

    score: float

    required_capabilities: tuple[
        RescueTeamCapability,
        ...
    ]

    reasons: tuple[str, ...]

    requires_specialist_review: bool = False

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class RescueTeam:
    id: str
    district_id: int
    name: str

    status: RescueTeamStatus

    capabilities: frozenset[
        RescueTeamCapability
    ]

    member_count: int

    verified: bool = False

    equipment_ready: bool = False

    current_location: RescueLocation | None = None

    last_status_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class RescueSafetyAssessment:
    route_verified: bool
    weather_reviewed: bool
    equipment_verified: bool
    communication_verified: bool
    medical_support_reviewed: bool

    approved_by: int
    assessed_at: datetime

    notes: str = ""

    def __post_init__(self) -> None:
        validate_datetime(
            self.assessed_at,
            "assessed_at",
        )

    @property
    def ready(self) -> bool:
        return all((
            self.route_verified,
            self.weather_reviewed,
            self.equipment_verified,
            self.communication_verified,
            self.medical_support_reviewed,
        ))


@dataclass(frozen=True)
class RescueProgressUpdate:
    progress_percent: float

    people_rescued: int
    people_remaining: int

    injured_people_transferred: int = 0

    notes: str = ""

    evidence_references: tuple[str, ...] = ()

    reported_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_progress(
            self.progress_percent
        )

        for name in (
            "people_rescued",
            "people_remaining",
            "injured_people_transferred",
        ):
            validate_non_negative(
                getattr(self, name),
                name,
            )

        validate_datetime(
            self.reported_at,
            "reported_at",
        )


@dataclass(frozen=True)
class RescueTimelineEvent:
    id: str
    event_type: RescueEventType

    actor_id: int | None
    occurred_at: datetime

    description: str = ""

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class RescueMission:
    id: str
    district_id: int

    rescue_type: RescueType
    location: RescueLocation
    population: RescuePopulation

    description: str

    status: RescueStatus
    priority: RescuePriority

    created_by: int

    incident_id: str | None = None
    sos_request_id: str | None = None

    risk_assessment: RescueRiskAssessment | None = None
    safety_assessment: RescueSafetyAssessment | None = None

    assigned_team_id: str | None = None

    approved_by: int | None = None
    approval_reference: str | None = None

    progress_percent: float = 0.0

    people_rescued: int = 0
    people_remaining: int = 0

    injured_people_transferred: int = 0

    evidence_references: list[str] = field(
        default_factory=list
    )

    timeline: list[RescueTimelineEvent] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    approved_at: datetime | None = None
    dispatched_at: datetime | None = None
    acknowledged_at: datetime | None = None
    arrived_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None

    cancellation_reason: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class RescueRepository(Protocol):

    async def create(
        self,
        mission: RescueMission,
    ) -> RescueMission:
        ...

    async def get_by_id(
        self,
        mission_id: str,
    ) -> RescueMission | None:
        ...

    async def update(
        self,
        mission: RescueMission,
        *,
        expected_version: int,
    ) -> RescueMission:
        """
        Must enforce optimistic concurrency atomically.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[RescueMission]:
        ...

    async def list_by_incident(
        self,
        incident_id: str,
    ) -> Sequence[RescueMission]:
        ...

    async def find_by_sos_request(
        self,
        sos_request_id: str,
    ) -> RescueMission | None:
        ...


class RescueTeamGateway(Protocol):

    async def get_team(
        self,
        team_id: str,
    ) -> RescueTeam | None:
        ...

    async def list_available_teams(
        self,
        district_id: int,
    ) -> Sequence[RescueTeam]:
        ...

    async def reserve_team(
        self,
        *,
        team_id: str,
        mission_id: str,
    ) -> bool:
        """
        Atomically reserve an available team.
        """
        ...

    async def release_team(
        self,
        *,
        team_id: str,
        mission_id: str,
    ) -> None:
        ...


class RescueDispatchGateway(Protocol):

    async def dispatch(
        self,
        *,
        mission_id: str,
        team_id: str,
        approval_reference: str,
    ) -> bool:
        """
        Submit a dispatch command to an authorized
        dispatch system.

        True means accepted, not arrived or completed.
        Must be idempotent by mission_id.
        """
        ...


class RescueAuthorizationGateway(Protocol):

    async def can_create(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_triage(
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

    async def can_dispatch(
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


class RescueAuditGateway(Protocol):

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
# RESCUE RISK EVALUATOR
# ============================================================

class RescueRiskEvaluator:
    """
    Illustrative prioritization model.

    It must not replace emergency dispatch triage,
    qualified rescue leadership or official guidance.
    """

    def evaluate(
        self,
        request: RescueCreateRequest,
    ) -> RescueRiskAssessment:

        score = 0.0
        reasons: list[str] = []

        required: set[RescueTeamCapability] = {
            RescueTeamCapability.SEARCH,
        }

        population = request.population

        score += min(
            population.total_people / 30,
            1.0,
        ) * 0.15

        if population.children:
            score += 0.08
            reasons.append(
                "Children require assistance"
            )

        if population.elderly:
            score += 0.08
            reasons.append(
                "Elderly people require assistance"
            )

        if population.disabled_people:
            score += 0.08
            reasons.append(
                "People with disabilities require assistance"
            )

        if population.critical_patients:
            score += 0.20
            required.add(
                RescueTeamCapability.MEDICAL
            )
            reasons.append(
                "Critical medical patients reported"
            )

        elif population.injured_people:
            score += 0.10
            required.add(
                RescueTeamCapability.MEDICAL
            )
            reasons.append(
                "Injured people reported"
            )

        if request.medical_emergency:
            score += 0.12
            required.add(
                RescueTeamCapability.MEDICAL
            )
            reasons.append(
                "Medical emergency reported"
            )

        if request.water_current_risk:
            score += 0.15
            required.add(
                RescueTeamCapability.SWIFT_WATER
            )
            reasons.append(
                "Dangerous water current reported"
            )

        if request.building_instability:
            score += 0.15
            reasons.append(
                "Possible structural instability"
            )

        if request.electrical_hazard:
            score += 0.15
            reasons.append(
                "Electrical hazard reported"
            )

        if request.landslide_risk:
            score += 0.15
            reasons.append(
                "Landslide hazard reported"
            )

        if request.rescue_type in {
            RescueType.BOAT_RESCUE,
            RescueType.WATER_RESCUE,
            RescueType.ROOFTOP_RESCUE,
        }:
            required.add(
                RescueTeamCapability.BOAT
            )

        if request.rescue_type == RescueType.MEDICAL_EVACUATION:
            required.add(
                RescueTeamCapability.MEDICAL
            )

        if request.water_depth_meters is not None:
            if (
                not math.isfinite(request.water_depth_meters)
                or request.water_depth_meters < 0
            ):
                raise RescueValidationError(
                    "Invalid water depth"
                )

            if request.water_depth_meters >= 1.5:
                score += 0.10
                reasons.append(
                    "Significant reported water depth"
                )

        score = min(
            score,
            1.0,
        )

        if score >= 0.75:
            priority = RescuePriority.CRITICAL
            risk = RescueRiskLevel.EXTREME

        elif score >= 0.50:
            priority = RescuePriority.HIGH
            risk = RescueRiskLevel.HIGH

        elif score >= 0.25:
            priority = RescuePriority.MEDIUM
            risk = RescueRiskLevel.MODERATE

        else:
            priority = RescuePriority.LOW
            risk = RescueRiskLevel.LOW

        specialist_review = any((
            request.water_current_risk,
            request.building_instability,
            request.electrical_hazard,
            request.landslide_risk,
            population.critical_patients > 0,
        ))

        return RescueRiskAssessment(
            priority=priority,
            risk_level=risk,
            score=round(score, 4),
            required_capabilities=tuple(
                sorted(
                    required,
                    key=lambda capability: capability.value,
                )
            ),
            reasons=tuple(reasons),
            requires_specialist_review=specialist_review,
        )


# ============================================================
# TEAM MATCHER
# ============================================================

class RescueTeamMatcher:

    def eligible_teams(
        self,
        *,
        mission: RescueMission,
        teams: Sequence[RescueTeam],
    ) -> list[RescueTeam]:

        if mission.risk_assessment is None:
            return []

        required = set(
            mission.risk_assessment.required_capabilities
        )

        eligible = []

        for team in teams:
            if team.district_id != mission.district_id:
                continue

            if team.status != RescueTeamStatus.AVAILABLE:
                continue

            if not team.verified:
                continue

            if not team.equipment_ready:
                continue

            if not required.issubset(
                team.capabilities
            ):
                continue

            if team.member_count <= 0:
                continue

            eligible.append(team)

        return sorted(
            eligible,
            key=lambda team: (
                -len(team.capabilities),
                -team.member_count,
                team.id,
            ),
        )


# ============================================================
# RESCUE SERVICE
# ============================================================

class RescueService:

    TERMINAL_STATUSES = frozenset({
        RescueStatus.COMPLETED,
        RescueStatus.CANCELLED,
        RescueStatus.REJECTED,
    })

    def __init__(
        self,
        *,
        repository: RescueRepository,
        team_gateway: RescueTeamGateway,
        dispatch_gateway: RescueDispatchGateway,
        authorization_gateway: RescueAuthorizationGateway,
        audit_gateway: RescueAuditGateway | None = None,
        risk_evaluator: RescueRiskEvaluator | None = None,
        team_matcher: RescueTeamMatcher | None = None,
    ):
        self.repository = repository
        self.teams = team_gateway
        self.dispatch = dispatch_gateway
        self.authorization = authorization_gateway
        self.audit = audit_gateway

        self.risk_evaluator = (
            risk_evaluator
            or RescueRiskEvaluator()
        )

        self.team_matcher = (
            team_matcher
            or RescueTeamMatcher()
        )

    @staticmethod
    def _event(
        event_type: RescueEventType,
        *,
        actor_id: int | None,
        description: str = "",
    ) -> RescueTimelineEvent:

        return RescueTimelineEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            actor_id=actor_id,
            occurred_at=utc_now(),
            description=description,
        )

    async def _audit(
        self,
        action: str,
        mission: RescueMission,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="rescue_mission",
            entity_id=mission.id,
            metadata={
                "district_id": mission.district_id,
                "status": mission.status.value,
                "priority": mission.priority.value,
                "version": mission.version,
            },
        )

    async def _save(
        self,
        original: RescueMission,
        updated: RescueMission,
        *,
        actor_id: int,
        action: str,
    ) -> RescueMission:

        saved = await self.repository.update(
            updated,
            expected_version=original.version,
        )

        await self._audit(
            action,
            saved,
            actor_id,
        )

        return saved

    async def get_mission(
        self,
        mission_id: str,
    ) -> RescueMission:

        mission = await self.repository.get_by_id(
            mission_id
        )

        if mission is None:
            raise RescueNotFoundError(
                "Rescue mission not found"
            )

        return mission

    async def create_mission(
        self,
        request: RescueCreateRequest,
        *,
        actor_id: int,
    ) -> RescueMission:

        if request.district_id != request.location.district_id:
            raise RescueValidationError(
                "District and location mismatch"
            )

        if not await self.authorization.can_create(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue creation permission denied"
            )

        description = require_text(
            request.description,
            "Rescue description",
        )

        if request.sos_request_id:
            existing = (
                await self.repository.find_by_sos_request(
                    request.sos_request_id
                )
            )

            if existing is not None:
                raise RescueConflictError(
                    "SOS request already linked "
                    "to a rescue mission"
                )

        mission = RescueMission(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            rescue_type=request.rescue_type,
            location=request.location,
            population=request.population,
            description=description,
            status=RescueStatus.REQUESTED,
            priority=RescuePriority.MEDIUM,
            created_by=actor_id,
            incident_id=request.incident_id,
            sos_request_id=request.sos_request_id,
            people_remaining=(
                request.population.total_people
            ),
            timeline=[
                self._event(
                    RescueEventType.CREATED,
                    actor_id=actor_id,
                )
            ],
            metadata={
                **request.metadata,
                "reported_water_depth_meters": (
                    request.water_depth_meters
                ),
                "water_current_risk": (
                    request.water_current_risk
                ),
                "building_instability": (
                    request.building_instability
                ),
                "electrical_hazard": (
                    request.electrical_hazard
                ),
                "landslide_risk": (
                    request.landslide_risk
                ),
                "medical_emergency": (
                    request.medical_emergency
                ),
            },
        )

        created = await self.repository.create(
            mission
        )

        await self._audit(
            "rescue.created",
            created,
            actor_id,
        )

        return created

    async def triage_mission(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_triage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue triage permission denied"
            )

        if mission.status != RescueStatus.REQUESTED:
            raise RescueConflictError(
                "Mission is not awaiting triage"
            )

        request = RescueCreateRequest(
            district_id=mission.district_id,
            rescue_type=mission.rescue_type,
            location=mission.location,
            population=mission.population,
            description=mission.description,
            incident_id=mission.incident_id,
            sos_request_id=mission.sos_request_id,
            water_depth_meters=mission.metadata.get(
                "reported_water_depth_meters"
            ),
            water_current_risk=bool(
                mission.metadata.get(
                    "water_current_risk",
                    False,
                )
            ),
            building_instability=bool(
                mission.metadata.get(
                    "building_instability",
                    False,
                )
            ),
            electrical_hazard=bool(
                mission.metadata.get(
                    "electrical_hazard",
                    False,
                )
            ),
            landslide_risk=bool(
                mission.metadata.get(
                    "landslide_risk",
                    False,
                )
            ),
            medical_emergency=bool(
                mission.metadata.get(
                    "medical_emergency",
                    False,
                )
            ),
        )

        assessment = self.risk_evaluator.evaluate(
            request
        )

        updated = replace(
            mission,
            status=RescueStatus.TRIAGED,
            priority=assessment.priority,
            risk_assessment=assessment,
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.TRIAGED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.triaged",
        )

    async def submit_for_approval(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue submission permission denied"
            )

        if mission.status != RescueStatus.TRIAGED:
            raise RescueConflictError(
                "Mission must be triaged first"
            )

        updated = replace(
            mission,
            status=RescueStatus.PENDING_APPROVAL,
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.SUBMITTED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.submitted",
        )

    async def approve_mission(
        self,
        mission_id: str,
        *,
        reviewer_id: int,
        approval_reference: str,
        safety_assessment: RescueSafetyAssessment,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue approval permission denied"
            )

        if mission.status != RescueStatus.PENDING_APPROVAL:
            raise RescueConflictError(
                "Mission is not awaiting approval"
            )

        if reviewer_id == mission.created_by:
            raise RescueAuthorizationError(
                "Independent rescue approval required"
            )

        if safety_assessment.approved_by != reviewer_id:
            raise RescueValidationError(
                "Safety assessment reviewer mismatch"
            )

        if not safety_assessment.ready:
            raise RescueSafetyError(
                "Rescue safety checks are incomplete"
            )

        approval_reference = require_text(
            approval_reference,
            "Approval reference",
        )

        updated = replace(
            mission,
            status=RescueStatus.APPROVED,
            approved_by=reviewer_id,
            approval_reference=approval_reference,
            safety_assessment=safety_assessment,
            approved_at=utc_now(),
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.APPROVED,
                    actor_id=reviewer_id,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=reviewer_id,
            action="rescue.approved",
        )

    async def get_eligible_teams(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> list[RescueTeam]:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Team matching permission denied"
            )

        teams = await self.teams.list_available_teams(
            mission.district_id
        )

        return self.team_matcher.eligible_teams(
            mission=mission,
            teams=teams,
        )

    async def assign_team(
        self,
        mission_id: str,
        *,
        team_id: str,
        actor_id: int,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Team assignment permission denied"
            )

        if mission.status != RescueStatus.APPROVED:
            raise RescueConflictError(
                "Mission must be approved first"
            )

        team = await self.teams.get_team(
            team_id
        )

        if team is None:
            raise RescueNotFoundError(
                "Rescue team not found"
            )

        eligible = self.team_matcher.eligible_teams(
            mission=mission,
            teams=[team],
        )

        if not eligible:
            raise RescueSafetyError(
                "Team does not meet mission requirements"
            )

        reserved = await self.teams.reserve_team(
            team_id=team_id,
            mission_id=mission.id,
        )

        if not reserved:
            raise RescueConflictError(
                "Rescue team is unavailable"
            )

        updated = replace(
            mission,
            status=RescueStatus.TEAM_ASSIGNED,
            assigned_team_id=team_id,
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.TEAM_ASSIGNED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        try:
            return await self._save(
                mission,
                updated,
                actor_id=actor_id,
                action="rescue.team_assigned",
            )
        except Exception:
            await self.teams.release_team(
                team_id=team_id,
                mission_id=mission.id,
            )
            raise

    async def dispatch_mission(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue dispatch permission denied"
            )

        if mission.status != RescueStatus.TEAM_ASSIGNED:
            raise RescueConflictError(
                "Mission is not ready for dispatch"
            )

        if (
            mission.assigned_team_id is None
            or mission.approval_reference is None
            or mission.safety_assessment is None
        ):
            raise RescueSafetyError(
                "Dispatch prerequisites are incomplete"
            )

        if not mission.safety_assessment.ready:
            raise RescueSafetyError(
                "Safety approval is incomplete"
            )

        # This gateway must implement durable
        # idempotency and dispatch reconciliation.
        accepted = await self.dispatch.dispatch(
            mission_id=mission.id,
            team_id=mission.assigned_team_id,
            approval_reference=mission.approval_reference,
        )

        if not accepted:
            raise RescueIntegrationError(
                "Dispatch system did not accept mission"
            )

        updated = replace(
            mission,
            status=RescueStatus.DISPATCHED,
            dispatched_at=utc_now(),
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.DISPATCHED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.dispatched",
        )

    async def acknowledge_dispatch(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> RescueMission:

        return await self._transition(
            mission_id,
            actor_id=actor_id,
            expected=RescueStatus.DISPATCHED,
            target=RescueStatus.ACKNOWLEDGED,
            event_type=RescueEventType.ACKNOWLEDGED,
            timestamp_field="acknowledged_at",
        )

    async def mark_en_route(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> RescueMission:

        return await self._transition(
            mission_id,
            actor_id=actor_id,
            expected=RescueStatus.ACKNOWLEDGED,
            target=RescueStatus.EN_ROUTE,
            event_type=RescueEventType.EN_ROUTE,
        )

    async def mark_on_scene(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> RescueMission:

        return await self._transition(
            mission_id,
            actor_id=actor_id,
            expected=RescueStatus.EN_ROUTE,
            target=RescueStatus.ON_SCENE,
            event_type=RescueEventType.ARRIVED,
            timestamp_field="arrived_at",
        )

    async def start_rescue(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> RescueMission:

        return await self._transition(
            mission_id,
            actor_id=actor_id,
            expected=RescueStatus.ON_SCENE,
            target=RescueStatus.IN_PROGRESS,
            event_type=RescueEventType.STARTED,
            timestamp_field="started_at",
        )

    async def _transition(
        self,
        mission_id: str,
        *,
        actor_id: int,
        expected: RescueStatus,
        target: RescueStatus,
        event_type: RescueEventType,
        timestamp_field: str | None = None,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue status update denied"
            )

        if mission.status != expected:
            raise RescueConflictError(
                f"Expected status {expected.value}; "
                f"found {mission.status.value}"
            )

        changes: dict[str, Any] = {
            "status": target,
            "updated_at": utc_now(),
            "version": mission.version + 1,
            "timeline": [
                *mission.timeline,
                self._event(
                    event_type,
                    actor_id=actor_id,
                ),
            ],
        }

        if timestamp_field is not None:
            changes[timestamp_field] = utc_now()

        updated = replace(
            mission,
            **changes,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action=f"rescue.{target.value}",
        )

    async def update_progress(
        self,
        mission_id: str,
        *,
        update: RescueProgressUpdate,
        actor_id: int,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue progress update denied"
            )

        if mission.status != RescueStatus.IN_PROGRESS:
            raise RescueConflictError(
                "Rescue mission is not in progress"
            )

        if (
            update.progress_percent
            < mission.progress_percent
        ):
            raise RescueConflictError(
                "Progress cannot decrease"
            )

        if (
            update.people_rescued
            < mission.people_rescued
        ):
            raise RescueConflictError(
                "Confirmed rescue count cannot decrease"
            )

        if (
            update.people_rescued
            + update.people_remaining
            != mission.population.total_people
        ):
            raise RescueValidationError(
                "Rescued and remaining counts must match "
                "the recorded population"
            )

        if (
            update.injured_people_transferred
            > update.people_rescued
        ):
            raise RescueValidationError(
                "Transferred people exceed rescued people"
            )

        if (
            update.people_rescued > 0
            and not update.evidence_references
        ):
            raise RescueVerificationError(
                "Rescue evidence is required"
            )

        updated = replace(
            mission,
            progress_percent=update.progress_percent,
            people_rescued=update.people_rescued,
            people_remaining=update.people_remaining,
            injured_people_transferred=(
                update.injured_people_transferred
            ),
            evidence_references=list(
                dict.fromkeys([
                    *mission.evidence_references,
                    *update.evidence_references,
                ])
            ),
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.PROGRESS_UPDATED,
                    actor_id=actor_id,
                    description=update.notes[:500],
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.progress_updated",
        )

    async def pause_mission(
        self,
        mission_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue pause permission denied"
            )

        if mission.status != RescueStatus.IN_PROGRESS:
            raise RescueConflictError(
                "Only active missions can be paused"
            )

        reason = require_text(
            reason,
            "Pause reason",
        )

        updated = replace(
            mission,
            status=RescueStatus.PAUSED,
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.PAUSED,
                    actor_id=actor_id,
                    description=reason,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.paused",
        )

    async def resume_mission(
        self,
        mission_id: str,
        *,
        actor_id: int,
        safety_assessment: RescueSafetyAssessment,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_approve(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue resume approval denied"
            )

        if mission.status != RescueStatus.PAUSED:
            raise RescueConflictError(
                "Mission is not paused"
            )

        if not safety_assessment.ready:
            raise RescueSafetyError(
                "Updated safety assessment is incomplete"
            )

        if safety_assessment.approved_by != actor_id:
            raise RescueAuthorizationError(
                "Safety reviewer mismatch"
            )

        updated = replace(
            mission,
            status=RescueStatus.IN_PROGRESS,
            safety_assessment=safety_assessment,
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.RESUMED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.resumed",
        )

    async def escalate_mission(
        self,
        mission_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue escalation denied"
            )

        if mission.status in self.TERMINAL_STATUSES:
            raise RescueConflictError(
                "Cannot escalate a terminal mission"
            )

        reason = require_text(
            reason,
            "Escalation reason",
        )

        updated = replace(
            mission,
            priority=RescuePriority.CRITICAL,
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.ESCALATED,
                    actor_id=actor_id,
                    description=reason,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.escalated",
        )

    async def request_completion(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Completion request denied"
            )

        if mission.status != RescueStatus.IN_PROGRESS:
            raise RescueConflictError(
                "Mission is not in progress"
            )

        if mission.people_remaining != 0:
            raise RescueVerificationError(
                "People remain unaccounted for"
            )

        if (
            mission.people_rescued
            != mission.population.total_people
        ):
            raise RescueVerificationError(
                "Rescue count does not match population"
            )

        if not mission.evidence_references:
            raise RescueVerificationError(
                "Rescue completion evidence is required"
            )

        updated = replace(
            mission,
            status=RescueStatus.PENDING_VERIFICATION,
            progress_percent=100.0,
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.COMPLETION_REQUESTED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        return await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.completion_requested",
        )

    async def complete_mission(
        self,
        mission_id: str,
        *,
        reviewer_id: int,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_verify(
            user_id=reviewer_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue completion verification denied"
            )

        if (
            mission.status
            != RescueStatus.PENDING_VERIFICATION
        ):
            raise RescueConflictError(
                "Mission is not awaiting verification"
            )

        if reviewer_id == mission.created_by:
            raise RescueAuthorizationError(
                "Independent verification required"
            )

        updated = replace(
            mission,
            status=RescueStatus.COMPLETED,
            completed_at=utc_now(),
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.COMPLETED,
                    actor_id=reviewer_id,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        saved = await self._save(
            mission,
            updated,
            actor_id=reviewer_id,
            action="rescue.completed",
        )

        if saved.assigned_team_id:
            try:
                await self.teams.release_team(
                    team_id=saved.assigned_team_id,
                    mission_id=saved.id,
                )
            except Exception:
                logger.exception(
                    "Rescue team release failed: %s",
                    saved.id,
                )

        return saved

    async def cancel_mission(
        self,
        mission_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> RescueMission:

        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_approve(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise RescueAuthorizationError(
                "Rescue cancellation denied"
            )

        if mission.status in self.TERMINAL_STATUSES:
            raise RescueConflictError(
                "Mission is already terminal"
            )

        if mission.status in {
            RescueStatus.DISPATCHED,
            RescueStatus.ACKNOWLEDGED,
            RescueStatus.EN_ROUTE,
            RescueStatus.ON_SCENE,
            RescueStatus.IN_PROGRESS,
            RescueStatus.PAUSED,
            RescueStatus.PENDING_VERIFICATION,
        }:
            raise RescueConflictError(
                "Dispatched missions require an authorized "
                "operational stand-down and acknowledgement "
                "workflow before cancellation"
            )

        reason = require_text(
            reason,
            "Cancellation reason",
        )

        updated = replace(
            mission,
            status=RescueStatus.CANCELLED,
            cancelled_at=utc_now(),
            cancellation_reason=reason,
            timeline=[
                *mission.timeline,
                self._event(
                    RescueEventType.CANCELLED,
                    actor_id=actor_id,
                    description=reason,
                ),
            ],
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        saved = await self._save(
            mission,
            updated,
            actor_id=actor_id,
            action="rescue.cancelled",
        )

        if saved.assigned_team_id:
            try:
                await self.teams.release_team(
                    team_id=saved.assigned_team_id,
                    mission_id=saved.id,
                )
            except Exception:
                logger.exception(
                    "Rescue team release failed: %s",
                    saved.id,
                )

        return saved

    async def list_district_missions(
        self,
        district_id: int,
    ) -> Sequence[RescueMission]:

        if district_id <= 0:
            raise RescueValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_incident_missions(
        self,
        incident_id: str,
    ) -> Sequence[RescueMission]:

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

        missions = (
            await self.list_district_missions(
                district_id
            )
        )

        status_counts = {
            status.value: 0
            for status in RescueStatus
        }

        priority_counts = {
            priority.value: 0
            for priority in RescuePriority
        }

        type_counts = {
            rescue_type.value: 0
            for rescue_type in RescueType
        }

        for mission in missions:
            status_counts[
                mission.status.value
            ] += 1

            priority_counts[
                mission.priority.value
            ] += 1

            type_counts[
                mission.rescue_type.value
            ] += 1

        completed = [
            mission
            for mission in missions
            if mission.status == RescueStatus.COMPLETED
        ]

        return {
            "district_id": district_id,
            "total_missions": len(missions),
            "completed_missions": len(completed),
            "active_missions": sum(
                mission.status not in self.TERMINAL_STATUSES
                for mission in missions
            ),
            "critical_missions": priority_counts[
                RescuePriority.CRITICAL.value
            ],
            "verified_people_rescued": sum(
                mission.people_rescued
                for mission in completed
            ),
            "reported_people_rescued_all_missions": sum(
                mission.people_rescued
                for mission in missions
            ),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "type_counts": type_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_mission_report(
        self,
        mission_id: str,
    ) -> dict[str, Any]:

        mission = await self.get_mission(
            mission_id
        )

        return {
            "mission_id": mission.id,
            "district_id": mission.district_id,
            "incident_id": mission.incident_id,
            "sos_request_id": mission.sos_request_id,
            "rescue_type": mission.rescue_type.value,
            "status": mission.status.value,
            "priority": mission.priority.value,
            "population": {
                "total_people": (
                    mission.population.total_people
                ),
                "children": mission.population.children,
                "elderly": mission.population.elderly,
                "injured_people": (
                    mission.population.injured_people
                ),
                "critical_patients": (
                    mission.population.critical_patients
                ),
            },
            "risk_assessment": (
                {
                    "risk_level": (
                        mission.risk_assessment.risk_level.value
                    ),
                    "score": mission.risk_assessment.score,
                    "reasons": list(
                        mission.risk_assessment.reasons
                    ),
                    "required_capabilities": [
                        capability.value
                        for capability in (
                            mission.risk_assessment
                            .required_capabilities
                        )
                    ],
                }
                if mission.risk_assessment
                else None
            ),
            "assigned_team_id": mission.assigned_team_id,
            "progress_percent": mission.progress_percent,
            "people_rescued": mission.people_rescued,
            "people_remaining": mission.people_remaining,
            "injured_people_transferred": (
                mission.injured_people_transferred
            ),
            "approval": {
                "approved_by": mission.approved_by,
                "approval_reference": (
                    mission.approval_reference
                ),
                "approved_at": (
                    mission.approved_at.isoformat()
                    if mission.approved_at
                    else None
                ),
            },
            "timeline": [
                {
                    "event_id": event.id,
                    "event_type": event.event_type.value,
                    "actor_id": event.actor_id,
                    "occurred_at": (
                        event.occurred_at.isoformat()
                    ),
                }
                for event in mission.timeline
            ],
            "created_at": mission.created_at.isoformat(),
            "dispatched_at": (
                mission.dispatched_at.isoformat()
                if mission.dispatched_at
                else None
            ),
            "completed_at": (
                mission.completed_at.isoformat()
                if mission.completed_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Rescue recommendations are advisory. "
                "Dispatch, arrival, field outcomes and "
                "completion must be verified by authorized "
                "rescue personnel. This report does not "
                "replace official emergency records."
            ),
        }
