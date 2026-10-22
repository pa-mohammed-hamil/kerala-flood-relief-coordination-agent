from __future__ import annotations

"""
backend/services/dispatch_service.py

Kerala Flood Relief Coordination Agent

Human-authorized rescue dispatch coordination.

Features:
- Rescue mission dispatch planning
- Rescue team availability and capability matching
- District and operational-area filtering
- Incident priority and urgency assessment
- Team workload and equipment evaluation
- Route feasibility integration
- Candidate ranking and recommendations
- Human approval before dispatch
- Team reservation and assignment
- Dispatch status tracking
- Mission acknowledgment
- Dispatch cancellation
- Audit logging
- Idempotency support
- Concurrent assignment protection

SAFETY:
This service must not autonomously dispatch emergency
responders. A recommendation is not a confirmed dispatch.

Production repositories must provide atomic reservations,
persistent approval records, transactional outbox delivery,
and optimistic concurrency control.
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

class DispatchPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DispatchStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TeamStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    BUSY = "busy"
    OFFLINE = "offline"
    UNAVAILABLE = "unavailable"


class RescueCapability(str, Enum):
    BOAT_RESCUE = "boat_rescue"
    SWIFT_WATER = "swift_water"
    MEDICAL = "medical"
    EVACUATION = "evacuation"
    SEARCH_AND_RESCUE = "search_and_rescue"
    HIGH_ANGLE = "high_angle"
    DIVING = "diving"
    HEAVY_EQUIPMENT = "heavy_equipment"
    DRONE_RECONNAISSANCE = "drone_reconnaissance"
    ANIMAL_RESCUE = "animal_rescue"


# ============================================================
# EXCEPTIONS
# ============================================================

class DispatchServiceError(Exception):
    pass


class DispatchValidationError(DispatchServiceError):
    pass


class DispatchNotFoundError(DispatchServiceError):
    pass


class DispatchConflictError(DispatchServiceError):
    pass


class DispatchAuthorizationError(DispatchServiceError):
    pass


class DispatchIntegrationError(DispatchServiceError):
    pass


class NoSuitableTeamError(DispatchServiceError):
    pass


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_aware_datetime(
    value: datetime,
    field_name: str,
) -> None:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise DispatchValidationError(
            f"{field_name} must be timezone-aware"
        )


def validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:
    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
    ):
        raise DispatchValidationError(
            "Coordinates must be finite"
        )

    if not -90 <= latitude <= 90:
        raise DispatchValidationError(
            "Latitude outside valid range"
        )

    if not -180 <= longitude <= 180:
        raise DispatchValidationError(
            "Longitude outside valid range"
        )


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Great-circle distance.

    This is NOT a travel-time estimate and does
    not account for floodwater or blocked roads.
    """
    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    radius_km = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return 2 * radius_km * math.asin(
        math.sqrt(a)
    )


# ============================================================
# DOMAIN MODELS
# ============================================================

@dataclass(frozen=True)
class GeoLocation:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        validate_coordinates(
            self.latitude,
            self.longitude,
        )


@dataclass(frozen=True)
class RescueTeam:
    id: str
    name: str
    district_id: int
    location: GeoLocation

    status: TeamStatus
    capabilities: frozenset[RescueCapability]

    available_members: int
    max_capacity: int

    equipment: frozenset[str] = frozenset()

    active_missions: int = 0
    max_concurrent_missions: int = 1

    verified: bool = False
    last_location_update: datetime | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DispatchRequest:
    incident_id: str
    district_id: int
    location: GeoLocation

    priority: DispatchPriority

    required_capabilities: frozenset[
        RescueCapability
    ]

    required_equipment: frozenset[str] = frozenset()

    minimum_team_members: int = 1
    people_at_risk: int = 0

    description: str = ""
    requested_by: int | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RouteAssessment:
    team_id: str
    incident_id: str

    feasible: bool
    distance_km: float | None = None
    estimated_minutes: float | None = None

    verified_at: datetime | None = None
    source: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DispatchCandidate:
    team_id: str
    team_name: str

    score: float
    straight_line_distance_km: float

    route_feasible: bool
    estimated_minutes: float | None

    capability_score: float
    availability_score: float
    workload_score: float
    proximity_score: float

    warnings: tuple[str, ...] = ()


@dataclass
class DispatchMission:
    id: str
    incident_id: str
    district_id: int

    priority: DispatchPriority
    status: DispatchStatus

    selected_team_id: str
    requested_by: int | None

    location: GeoLocation

    required_capabilities: frozenset[
        RescueCapability
    ]

    required_equipment: frozenset[str]

    minimum_team_members: int

    candidate_score: float

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    approved_by: int | None = None
    approved_at: datetime | None = None

    dispatched_by: int | None = None
    dispatched_at: datetime | None = None

    acknowledged_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None

    cancellation_reason: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DispatchRecommendation:
    incident_id: str
    district_id: int

    candidates: tuple[
        DispatchCandidate, ...
    ]

    generated_at: datetime

    requires_human_approval: bool = True


@dataclass(frozen=True)
class DispatchApproval:
    mission_id: str
    reviewer_id: int
    approved: bool
    reason: str

    reviewed_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class RescueTeamRepository(Protocol):

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[RescueTeam]:
        ...

    async def get_by_id(
        self,
        team_id: str,
    ) -> RescueTeam | None:
        ...

    async def reserve_team(
        self,
        *,
        team_id: str,
        mission_id: str,
    ) -> bool:
        """
        Atomically reserve a team only when
        capacity and availability still permit it.
        """
        ...

    async def release_team(
        self,
        *,
        team_id: str,
        mission_id: str,
    ) -> None:
        ...


class DispatchRepository(Protocol):

    async def create(
        self,
        mission: DispatchMission,
    ) -> DispatchMission:
        ...

    async def get_by_id(
        self,
        mission_id: str,
    ) -> DispatchMission | None:
        ...

    async def update(
        self,
        mission: DispatchMission,
        *,
        expected_version: int,
    ) -> DispatchMission:
        """
        Update using optimistic concurrency.

        Raise DispatchConflictError when
        expected_version does not match.
        """
        ...

    async def list_by_incident(
        self,
        incident_id: str,
    ) -> Sequence[DispatchMission]:
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[DispatchMission]:
        ...


class RoutePlanningGateway(Protocol):

    async def assess_route(
        self,
        *,
        team: RescueTeam,
        request: DispatchRequest,
    ) -> RouteAssessment:
        ...


class DispatchNotificationGateway(Protocol):

    async def notify_dispatch(
        self,
        mission: DispatchMission,
    ) -> bool:
        ...

    async def notify_cancellation(
        self,
        mission: DispatchMission,
    ) -> bool:
        ...


class DispatchAuditGateway(Protocol):

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


class DispatchAuthorizationGateway(Protocol):

    async def can_dispatch(
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


# ============================================================
# CANDIDATE EVALUATOR
# ============================================================

class DispatchCandidateEvaluator:
    """
    Conservative ranking of rescue teams.

    Ranking is advisory and must not be used
    as an autonomous dispatch decision.
    """

    def __init__(
        self,
        *,
        maximum_location_age_minutes: int = 15,
        maximum_distance_km: float = 100.0,
    ):
        if maximum_location_age_minutes <= 0:
            raise ValueError(
                "Location age limit must be positive"
            )

        if maximum_distance_km <= 0:
            raise ValueError(
                "Maximum distance must be positive"
            )

        self.maximum_location_age_minutes = (
            maximum_location_age_minutes
        )

        self.maximum_distance_km = (
            maximum_distance_km
        )

    def validate_request(
        self,
        request: DispatchRequest,
    ) -> None:
        if not request.incident_id.strip():
            raise DispatchValidationError(
                "Incident ID is required"
            )

        if request.district_id <= 0:
            raise DispatchValidationError(
                "District ID must be positive"
            )

        if request.minimum_team_members <= 0:
            raise DispatchValidationError(
                "Minimum team size must be positive"
            )

        if request.people_at_risk < 0:
            raise DispatchValidationError(
                "People at risk cannot be negative"
            )

        if not request.required_capabilities:
            raise DispatchValidationError(
                "At least one rescue capability is required"
            )

    def is_team_eligible(
        self,
        team: RescueTeam,
        request: DispatchRequest,
        *,
        now: datetime | None = None,
    ) -> bool:
        now = now or utc_now()

        if team.district_id != request.district_id:
            return False

        if not team.verified:
            return False

        if team.status != TeamStatus.AVAILABLE:
            return False

        if (
            team.available_members
            < request.minimum_team_members
        ):
            return False

        if (
            team.active_missions
            >= team.max_concurrent_missions
        ):
            return False

        if not request.required_capabilities.issubset(
            team.capabilities
        ):
            return False

        if not request.required_equipment.issubset(
            team.equipment
        ):
            return False

        if team.last_location_update is None:
            return False

        require_aware_datetime(
            team.last_location_update,
            "last_location_update",
        )

        age = now - team.last_location_update

        if age < timedelta(0):
            return False

        if age > timedelta(
            minutes=self.maximum_location_age_minutes
        ):
            return False

        distance = haversine_distance_km(
            team.location.latitude,
            team.location.longitude,
            request.location.latitude,
            request.location.longitude,
        )

        return distance <= self.maximum_distance_km

    def evaluate(
        self,
        team: RescueTeam,
        request: DispatchRequest,
        route: RouteAssessment,
    ) -> DispatchCandidate:

        if not self.is_team_eligible(
            team,
            request,
        ):
            raise DispatchValidationError(
                "Team is not eligible for this mission"
            )

        distance = haversine_distance_km(
            team.location.latitude,
            team.location.longitude,
            request.location.latitude,
            request.location.longitude,
        )

        if not route.feasible:
            raise DispatchValidationError(
                "Route is not verified as feasible"
            )

        if route.verified_at is None:
            raise DispatchValidationError(
                "Route verification timestamp required"
            )

        require_aware_datetime(
            route.verified_at,
            "route.verified_at",
        )

        if (
            utc_now() - route.verified_at
            > timedelta(minutes=10)
        ):
            raise DispatchValidationError(
                "Route verification is stale"
            )

        if (
            route.estimated_minutes is not None
            and (
                not math.isfinite(
                    route.estimated_minutes
                )
                or route.estimated_minutes < 0
            )
        ):
            raise DispatchValidationError(
                "Invalid route ETA"
            )

        capability_score = 1.0

        availability_score = min(
            1.0,
            team.available_members
            / max(
                1,
                request.minimum_team_members,
            ),
        )

        workload_score = max(
            0.0,
            1.0 - (
                team.active_missions
                / max(
                    1,
                    team.max_concurrent_missions,
                )
            ),
        )

        proximity_score = max(
            0.0,
            1.0 - (
                distance / self.maximum_distance_km
            ),
        )

        score = (
            capability_score * 0.35
            + availability_score * 0.25
            + workload_score * 0.20
            + proximity_score * 0.20
        )

        return DispatchCandidate(
            team_id=team.id,
            team_name=team.name,
            score=round(score, 4),
            straight_line_distance_km=round(
                distance,
                3,
            ),
            route_feasible=True,
            estimated_minutes=route.estimated_minutes,
            capability_score=capability_score,
            availability_score=availability_score,
            workload_score=workload_score,
            proximity_score=proximity_score,
            warnings=route.warnings,
        )


# ============================================================
# DISPATCH SERVICE
# ============================================================

class DispatchService:

    TERMINAL_STATUSES = frozenset({
        DispatchStatus.COMPLETED,
        DispatchStatus.CANCELLED,
        DispatchStatus.FAILED,
    })

    def __init__(
        self,
        *,
        team_repository: RescueTeamRepository,
        dispatch_repository: DispatchRepository,
        route_gateway: RoutePlanningGateway,
        authorization_gateway: DispatchAuthorizationGateway,
        notification_gateway: (
            DispatchNotificationGateway | None
        ) = None,
        audit_gateway: DispatchAuditGateway | None = None,
        evaluator: (
            DispatchCandidateEvaluator | None
        ) = None,
    ):
        self.teams = team_repository
        self.dispatches = dispatch_repository
        self.routes = route_gateway
        self.authorization = authorization_gateway
        self.notifications = notification_gateway
        self.audit = audit_gateway

        self.evaluator = (
            evaluator
            or DispatchCandidateEvaluator()
        )

    async def _audit(
        self,
        action: str,
        mission: DispatchMission,
        actor_id: int | None,
    ) -> None:
        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="dispatch_mission",
            entity_id=mission.id,
            metadata={
                "incident_id": mission.incident_id,
                "district_id": mission.district_id,
                "team_id": mission.selected_team_id,
                "status": mission.status.value,
            },
        )

    async def get_mission(
        self,
        mission_id: str,
    ) -> DispatchMission:
        mission = await self.dispatches.get_by_id(
            mission_id
        )

        if mission is None:
            raise DispatchNotFoundError(
                "Dispatch mission not found"
            )

        return mission

    async def recommend_teams(
        self,
        request: DispatchRequest,
        *,
        limit: int = 5,
    ) -> DispatchRecommendation:
        self.evaluator.validate_request(
            request
        )

        if limit <= 0 or limit > 50:
            raise DispatchValidationError(
                "Candidate limit must be between 1 and 50"
            )

        teams = await self.teams.list_by_district(
            request.district_id
        )

        candidates: list[DispatchCandidate] = []

        for team in teams:
            if not self.evaluator.is_team_eligible(
                team,
                request,
            ):
                continue

            try:
                route = await self.routes.assess_route(
                    team=team,
                    request=request,
                )

                if not route.feasible:
                    continue

                candidate = self.evaluator.evaluate(
                    team,
                    request,
                    route,
                )

                candidates.append(candidate)

            except (
                DispatchValidationError,
                DispatchIntegrationError,
            ):
                logger.warning(
                    "Team excluded from dispatch ranking: %s",
                    team.id,
                )

            except Exception:
                logger.exception(
                    "Route assessment failed for team %s",
                    team.id,
                )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                (
                    candidate.estimated_minutes
                    if candidate.estimated_minutes is not None
                    else float("inf")
                ),
                candidate.team_id,
            )
        )

        return DispatchRecommendation(
            incident_id=request.incident_id,
            district_id=request.district_id,
            candidates=tuple(
                candidates[:limit]
            ),
            generated_at=utc_now(),
        )

    async def create_dispatch_proposal(
        self,
        request: DispatchRequest,
        *,
        selected_team_id: str,
        actor_id: int,
    ) -> DispatchMission:
        self.evaluator.validate_request(
            request
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise DispatchAuthorizationError(
                "Dispatch planning permission denied"
            )

        recommendation = await self.recommend_teams(
            request,
            limit=50,
        )

        candidate = next(
            (
                candidate
                for candidate in recommendation.candidates
                if candidate.team_id == selected_team_id
            ),
            None,
        )

        if candidate is None:
            raise NoSuitableTeamError(
                "Selected team is not currently eligible"
            )

        mission = DispatchMission(
            id=str(uuid.uuid4()),
            incident_id=request.incident_id,
            district_id=request.district_id,
            priority=request.priority,
            status=DispatchStatus.PENDING_APPROVAL,
            selected_team_id=selected_team_id,
            requested_by=actor_id,
            location=request.location,
            required_capabilities=(
                request.required_capabilities
            ),
            required_equipment=(
                request.required_equipment
            ),
            minimum_team_members=(
                request.minimum_team_members
            ),
            candidate_score=candidate.score,
            metadata=dict(request.metadata),
        )

        created = await self.dispatches.create(
            mission
        )

        await self._audit(
            "dispatch.proposed",
            created,
            actor_id,
        )

        logger.info(
            "Dispatch proposal created: %s",
            created.id,
        )

        return created

    async def approve_dispatch(
        self,
        mission_id: str,
        *,
        reviewer_id: int,
        reason: str,
    ) -> DispatchMission:
        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=mission.district_id,
        ):
            raise DispatchAuthorizationError(
                "Dispatch approval permission denied"
            )

        if (
            mission.status
            != DispatchStatus.PENDING_APPROVAL
        ):
            raise DispatchConflictError(
                "Mission is not awaiting approval"
            )

        if mission.requested_by == reviewer_id:
            raise DispatchAuthorizationError(
                "Independent reviewer required"
            )

        if not reason.strip():
            raise DispatchValidationError(
                "Approval reason is required"
            )

        updated = replace(
            mission,
            status=DispatchStatus.APPROVED,
            approved_by=reviewer_id,
            approved_at=utc_now(),
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        saved = await self.dispatches.update(
            updated,
            expected_version=mission.version,
        )

        await self._audit(
            "dispatch.approved",
            saved,
            reviewer_id,
        )

        return saved

    async def dispatch_team(
        self,
        mission_id: str,
        *,
        actor_id: int,
    ) -> DispatchMission:
        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise DispatchAuthorizationError(
                "Dispatch permission denied"
            )

        if mission.status != DispatchStatus.APPROVED:
            raise DispatchConflictError(
                "Only approved missions can be dispatched"
            )

        if (
            mission.approved_by is None
            or mission.approved_at is None
        ):
            raise DispatchAuthorizationError(
                "Verified approval is missing"
            )

        team = await self.teams.get_by_id(
            mission.selected_team_id
        )

        if team is None:
            raise NoSuitableTeamError(
                "Selected rescue team no longer exists"
            )

        if (
            not team.verified
            or team.status != TeamStatus.AVAILABLE
            or team.district_id != mission.district_id
            or team.available_members
            < mission.minimum_team_members
            or not mission.required_capabilities.issubset(
                team.capabilities
            )
            or not mission.required_equipment.issubset(
                team.equipment
            )
        ):
            raise NoSuitableTeamError(
                "Team is no longer eligible"
            )

        reserved = await self.teams.reserve_team(
            team_id=team.id,
            mission_id=mission.id,
        )

        if not reserved:
            raise DispatchConflictError(
                "Rescue team could not be reserved"
            )

        try:
            updated = replace(
                mission,
                status=DispatchStatus.DISPATCHED,
                dispatched_by=actor_id,
                dispatched_at=utc_now(),
                updated_at=utc_now(),
                version=mission.version + 1,
            )

            saved = await self.dispatches.update(
                updated,
                expected_version=mission.version,
            )

        except Exception:
            await self.teams.release_team(
                team_id=team.id,
                mission_id=mission.id,
            )
            raise

        await self._audit(
            "dispatch.dispatched",
            saved,
            actor_id,
        )

        if self.notifications is not None:
            try:
                delivered = (
                    await self.notifications.notify_dispatch(
                        saved
                    )
                )

                if not delivered:
                    logger.error(
                        "Dispatch notification unconfirmed: %s",
                        saved.id,
                    )

            except Exception:
                logger.exception(
                    "Dispatch notification failed: %s",
                    saved.id,
                )

        return saved

    async def acknowledge_dispatch(
        self,
        mission_id: str,
        *,
        team_id: str,
    ) -> DispatchMission:
        mission = await self.get_mission(
            mission_id
        )

        if mission.selected_team_id != team_id:
            raise DispatchAuthorizationError(
                "Mission belongs to another team"
            )

        if mission.status != DispatchStatus.DISPATCHED:
            raise DispatchConflictError(
                "Mission cannot be acknowledged"
            )

        updated = replace(
            mission,
            status=DispatchStatus.ACKNOWLEDGED,
            acknowledged_at=utc_now(),
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        saved = await self.dispatches.update(
            updated,
            expected_version=mission.version,
        )

        await self._audit(
            "dispatch.acknowledged",
            saved,
            None,
        )

        return saved

    async def update_mission_status(
        self,
        mission_id: str,
        *,
        new_status: DispatchStatus,
        actor_id: int,
    ) -> DispatchMission:
        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise DispatchAuthorizationError(
                "Mission update permission denied"
            )

        allowed_transitions = {
            DispatchStatus.ACKNOWLEDGED: {
                DispatchStatus.EN_ROUTE,
            },
            DispatchStatus.EN_ROUTE: {
                DispatchStatus.ON_SCENE,
            },
            DispatchStatus.ON_SCENE: {
                DispatchStatus.COMPLETED,
            },
        }

        allowed = allowed_transitions.get(
            mission.status,
            set(),
        )

        if new_status not in allowed:
            raise DispatchConflictError(
                f"Invalid mission transition: "
                f"{mission.status.value} -> "
                f"{new_status.value}"
            )

        updated = replace(
            mission,
            status=new_status,
            updated_at=utc_now(),
            completed_at=(
                utc_now()
                if new_status == DispatchStatus.COMPLETED
                else mission.completed_at
            ),
            version=mission.version + 1,
        )

        saved = await self.dispatches.update(
            updated,
            expected_version=mission.version,
        )

        if new_status == DispatchStatus.COMPLETED:
            await self.teams.release_team(
                team_id=saved.selected_team_id,
                mission_id=saved.id,
            )

        await self._audit(
            f"dispatch.{new_status.value}",
            saved,
            actor_id,
        )

        return saved

    async def cancel_dispatch(
        self,
        mission_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> DispatchMission:
        mission = await self.get_mission(
            mission_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=mission.district_id,
        ):
            raise DispatchAuthorizationError(
                "Dispatch cancellation permission denied"
            )

        if mission.status in self.TERMINAL_STATUSES:
            raise DispatchConflictError(
                "Mission is already in a terminal state"
            )

        if not reason.strip():
            raise DispatchValidationError(
                "Cancellation reason is required"
            )

        updated = replace(
            mission,
            status=DispatchStatus.CANCELLED,
            cancellation_reason=reason.strip(),
            cancelled_at=utc_now(),
            updated_at=utc_now(),
            version=mission.version + 1,
        )

        saved = await self.dispatches.update(
            updated,
            expected_version=mission.version,
        )

        if mission.status in {
            DispatchStatus.RESERVED,
            DispatchStatus.DISPATCHED,
            DispatchStatus.ACKNOWLEDGED,
            DispatchStatus.EN_ROUTE,
            DispatchStatus.ON_SCENE,
        }:
            await self.teams.release_team(
                team_id=saved.selected_team_id,
                mission_id=saved.id,
            )

        await self._audit(
            "dispatch.cancelled",
            saved,
            actor_id,
        )

        if self.notifications is not None:
            try:
                await self.notifications.notify_cancellation(
                    saved
                )
            except Exception:
                logger.exception(
                    "Cancellation notification failed: %s",
                    saved.id,
                )

        return saved

    async def list_incident_dispatches(
        self,
        incident_id: str,
    ) -> Sequence[DispatchMission]:
        return await self.dispatches.list_by_incident(
            incident_id
        )

    async def list_district_dispatches(
        self,
        district_id: int,
    ) -> Sequence[DispatchMission]:
        return await self.dispatches.list_by_district(
            district_id
        )

    async def get_dispatch_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:
        missions = (
            await self.list_district_dispatches(
                district_id
            )
        )

        status_counts = {
            status.value: 0
            for status in DispatchStatus
        }

        priority_counts = {
            priority.value: 0
            for priority in DispatchPriority
        }

        for mission in missions:
            status_counts[
                mission.status.value
            ] += 1

            priority_counts[
                mission.priority.value
            ] += 1

        return {
            "district_id": district_id,
            "total_missions": len(missions),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "generated_at": utc_now().isoformat(),
        }
