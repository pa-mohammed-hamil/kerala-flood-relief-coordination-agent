
"""
backend/services/evacuation_service.py

Kerala Flood Relief Coordination Agent

Human-supervised evacuation planning and coordination.

Features:
- District-level evacuation planning
- Flood-risk and evacuation-priority assessment
- Vulnerable population prioritization
- Shelter capacity and accessibility matching
- Evacuation route verification
- Transport resource requirements
- Evacuation plan creation and updates
- Independent human approval
- Evacuation activation and coordination
- Evacuation progress tracking
- Shelter arrival confirmation
- Plan cancellation and completion
- Audit logging and statistics

SAFETY:
This module produces planning recommendations.

It does not autonomously order evacuations,
control vehicles, dispatch responders, or issue
official public warnings.

Actual evacuation instructions require authorized
officials, current local conditions, and verified
communication channels.

All repository transitions must use transactional
persistence and optimistic concurrency.
"""

from __future__ import annotations

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

class EvacuationPriority(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EvacuationStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class EvacuationTransport(str, Enum):
    WALKING = "walking"
    BUS = "bus"
    VAN = "van"
    AMBULANCE = "ambulance"
    BOAT = "boat"
    RESCUE_VEHICLE = "rescue_vehicle"
    ACCESSIBLE_VEHICLE = "accessible_vehicle"


class EvacuationRiskSource(str, Enum):
    FLOOD_FORECAST = "flood_forecast"
    RIVER_LEVEL = "river_level"
    DAM_WARNING = "dam_warning"
    RAINFALL = "rainfall"
    FIELD_REPORT = "field_report"
    OFFICIAL_ORDER = "official_order"


# ============================================================
# EXCEPTIONS
# ============================================================

class EvacuationServiceError(Exception):
    pass


class EvacuationValidationError(EvacuationServiceError):
    pass


class EvacuationNotFoundError(EvacuationServiceError):
    pass


class EvacuationConflictError(EvacuationServiceError):
    pass


class EvacuationAuthorizationError(EvacuationServiceError):
    pass


class EvacuationCapacityError(EvacuationServiceError):
    pass


class EvacuationRouteError(EvacuationServiceError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_ratio(
    value: float,
    name: str,
) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise EvacuationValidationError(
            f"{name} must be a finite value between 0 and 1"
        )
    return value


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EvacuationValidationError(
            f"{name} must be timezone-aware"
        )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class EvacuationLocation:
    latitude: float
    longitude: float
    name: str = ""

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.latitude)
            and -90 <= self.latitude <= 90
        ):
            raise EvacuationValidationError(
                "Invalid latitude"
            )

        if not (
            math.isfinite(self.longitude)
            and -180 <= self.longitude <= 180
        ):
            raise EvacuationValidationError(
                "Invalid longitude"
            )


@dataclass(frozen=True)
class EvacuationPopulation:
    total_people: int
    children: int = 0
    elderly: int = 0
    people_with_disabilities: int = 0
    medical_support_required: int = 0
    mobility_assistance_required: int = 0

    def __post_init__(self) -> None:
        values = (
            self.total_people,
            self.children,
            self.elderly,
            self.people_with_disabilities,
            self.medical_support_required,
            self.mobility_assistance_required,
        )

        if any(value < 0 for value in values):
            raise EvacuationValidationError(
                "Population counts cannot be negative"
            )

        if self.total_people <= 0:
            raise EvacuationValidationError(
                "Total population must be positive"
            )

        if any(
            value > self.total_people
            for value in values[1:]
        ):
            raise EvacuationValidationError(
                "Population subgroup exceeds total population"
            )


@dataclass(frozen=True)
class EvacuationRisk:
    flood_probability: float
    flood_severity: float
    exposure: float
    urgency: float
    source: EvacuationRiskSource
    source_reference: str | None = None
    observed_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "flood_probability",
            "flood_severity",
            "exposure",
            "urgency",
        ):
            validate_ratio(
                getattr(self, name),
                name,
            )

        if self.observed_at is not None:
            validate_datetime(
                self.observed_at,
                "observed_at",
            )


@dataclass(frozen=True)
class EvacuationRequest:
    district_id: int
    incident_id: str
    origin: EvacuationLocation
    population: EvacuationPopulation
    risk: EvacuationRisk

    affected_areas: tuple[str, ...] = ()
    required_transport: frozenset[
        EvacuationTransport
    ] = frozenset()

    preferred_shelter_id: str | None = None
    special_requirements: tuple[str, ...] = ()
    requested_by: int | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ShelterOption:
    id: str
    name: str
    district_id: int
    location: EvacuationLocation

    total_capacity: int
    occupied_capacity: int
    reserved_capacity: int = 0

    operational: bool = True
    verified: bool = False

    accessible: bool = False
    medical_support: bool = False

    @property
    def available_capacity(self) -> int:
        return max(
            0,
            self.total_capacity
            - self.occupied_capacity
            - self.reserved_capacity,
        )


@dataclass(frozen=True)
class EvacuationRoute:
    route_id: str
    origin: EvacuationLocation
    destination: EvacuationLocation

    feasible: bool
    verified: bool

    distance_km: float | None = None
    estimated_minutes: float | None = None

    verified_at: datetime | None = None

    hazards: tuple[str, ...] = ()
    blocked_segments: tuple[str, ...] = ()

    source: str = ""


@dataclass(frozen=True)
class ShelterRecommendation:
    shelter_id: str
    shelter_name: str

    available_capacity: int
    capacity_sufficient: bool

    accessible: bool
    medical_support: bool

    route: EvacuationRoute
    score: float

    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvacuationRiskAssessment:
    risk_score: float
    priority: EvacuationPriority
    factors: Mapping[str, float]
    requires_human_review: bool = True


@dataclass
class EvacuationPlan:
    id: str
    district_id: int
    incident_id: str

    origin: EvacuationLocation
    population: EvacuationPopulation

    priority: EvacuationPriority
    risk_score: float

    shelter_id: str
    route_id: str

    transport_modes: frozenset[
        EvacuationTransport
    ]

    status: EvacuationStatus = EvacuationStatus.DRAFT

    evacuated_people: int = 0
    arrived_people: int = 0

    requested_by: int | None = None
    approved_by: int | None = None
    activated_by: int | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )
    updated_at: datetime = field(
        default_factory=utc_now
    )

    approved_at: datetime | None = None
    activated_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None

    cancellation_reason: str | None = None
    approval_reference: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class EvacuationProgress:
    plan_id: str
    evacuated_people: int
    arrived_people: int
    updated_by: int
    recorded_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class EvacuationRepository(Protocol):

    async def create(
        self,
        plan: EvacuationPlan,
    ) -> EvacuationPlan:
        ...

    async def get_by_id(
        self,
        plan_id: str,
    ) -> EvacuationPlan | None:
        ...

    async def update(
        self,
        plan: EvacuationPlan,
        *,
        expected_version: int,
    ) -> EvacuationPlan:
        """
        Must atomically compare expected_version
        against the stored version.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[EvacuationPlan]:
        ...

    async def list_by_incident(
        self,
        incident_id: str,
    ) -> Sequence[EvacuationPlan]:
        ...


class ShelterGateway(Protocol):

    async def list_available(
        self,
        district_id: int,
    ) -> Sequence[ShelterOption]:
        ...

    async def get_shelter(
        self,
        shelter_id: str,
    ) -> ShelterOption | None:
        ...

    async def reserve_capacity(
        self,
        *,
        shelter_id: str,
        plan_id: str,
        people: int,
    ) -> bool:
        """
        Atomically reserve shelter capacity.

        Must be idempotent for a given plan_id.
        """
        ...

    async def release_capacity(
        self,
        *,
        shelter_id: str,
        plan_id: str,
    ) -> None:
        ...


class EvacuationRouteGateway(Protocol):

    async def assess_route(
        self,
        *,
        origin: EvacuationLocation,
        destination: EvacuationLocation,
    ) -> EvacuationRoute:
        ...


class EvacuationAuthorizationGateway(Protocol):

    async def can_plan(
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

    async def can_activate(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class EvacuationAuditGateway(Protocol):

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


class EvacuationNotificationGateway(Protocol):

    async def notify_activation(
        self,
        plan: EvacuationPlan,
    ) -> bool:
        ...

    async def notify_cancellation(
        self,
        plan: EvacuationPlan,
    ) -> bool:
        ...


# ============================================================
# RISK ASSESSMENT
# ============================================================

class EvacuationRiskEvaluator:
    """
    Illustrative prioritization model.

    Weights are not official evacuation thresholds.
    Operational decisions must follow authoritative
    local emergency-management guidance.
    """

    WEIGHTS = {
        "flood_probability": 0.25,
        "flood_severity": 0.30,
        "exposure": 0.20,
        "urgency": 0.25,
    }

    def assess(
        self,
        risk: EvacuationRisk,
    ) -> EvacuationRiskAssessment:

        factors = {
            name: getattr(risk, name)
            for name in self.WEIGHTS
        }

        score = sum(
            factors[name] * weight
            for name, weight in self.WEIGHTS.items()
        )

        if score >= 0.85:
            priority = EvacuationPriority.CRITICAL
        elif score >= 0.65:
            priority = EvacuationPriority.HIGH
        elif score >= 0.35:
            priority = EvacuationPriority.MODERATE
        else:
            priority = EvacuationPriority.LOW

        return EvacuationRiskAssessment(
            risk_score=round(score, 4),
            priority=priority,
            factors=factors,
        )


# ============================================================
# SHELTER MATCHING
# ============================================================

class ShelterMatcher:

    def __init__(
        self,
        *,
        maximum_route_age_minutes: int = 10,
    ):
        if maximum_route_age_minutes <= 0:
            raise ValueError(
                "Maximum route age must be positive"
            )

        self.maximum_route_age_minutes = (
            maximum_route_age_minutes
        )

    def validate_route(
        self,
        route: EvacuationRoute,
    ) -> bool:
        if not route.feasible or not route.verified:
            return False

        if route.blocked_segments:
            return False

        if route.verified_at is None:
            return False

        validate_datetime(
            route.verified_at,
            "route.verified_at",
        )

        age = utc_now() - route.verified_at

        if age < timedelta(0):
            return False

        if age > timedelta(
            minutes=self.maximum_route_age_minutes
        ):
            return False

        if route.estimated_minutes is not None:
            if (
                not math.isfinite(route.estimated_minutes)
                or route.estimated_minutes < 0
            ):
                return False

        return True

    def evaluate(
        self,
        *,
        shelter: ShelterOption,
        request: EvacuationRequest,
        route: EvacuationRoute,
    ) -> ShelterRecommendation | None:

        if shelter.district_id != request.district_id:
            return None

        if not shelter.operational or not shelter.verified:
            return None

        if not self.validate_route(route):
            return None

        population = request.population

        if (
            population.mobility_assistance_required > 0
            or population.people_with_disabilities > 0
        ) and not shelter.accessible:
            return None

        if (
            population.medical_support_required > 0
            and not shelter.medical_support
        ):
            return None

        capacity_sufficient = (
            shelter.available_capacity
            >= population.total_people
        )

        if not capacity_sufficient:
            return None

        capacity_score = min(
            1.0,
            shelter.available_capacity
            / max(1, population.total_people),
        )

        if route.estimated_minutes is not None:
            travel_score = 1.0 / (
                1.0 + route.estimated_minutes / 60.0
            )
        else:
            travel_score = 0.5

        accessibility_score = (
            1.0 if shelter.accessible else 0.5
        )

        medical_score = (
            1.0 if shelter.medical_support else 0.5
        )

        score = (
            capacity_score * 0.35
            + travel_score * 0.35
            + accessibility_score * 0.15
            + medical_score * 0.15
        )

        return ShelterRecommendation(
            shelter_id=shelter.id,
            shelter_name=shelter.name,
            available_capacity=shelter.available_capacity,
            capacity_sufficient=True,
            accessible=shelter.accessible,
            medical_support=shelter.medical_support,
            route=route,
            score=round(score, 4),
            warnings=route.hazards,
        )


# ============================================================
# EVACUATION SERVICE
# ============================================================

class EvacuationService:

    TERMINAL_STATUSES = frozenset({
        EvacuationStatus.COMPLETED,
        EvacuationStatus.CANCELLED,
    })

    def __init__(
        self,
        *,
        repository: EvacuationRepository,
        shelter_gateway: ShelterGateway,
        route_gateway: EvacuationRouteGateway,
        authorization_gateway: EvacuationAuthorizationGateway,
        audit_gateway: EvacuationAuditGateway | None = None,
        notification_gateway: (
            EvacuationNotificationGateway | None
        ) = None,
        risk_evaluator: (
            EvacuationRiskEvaluator | None
        ) = None,
        shelter_matcher: ShelterMatcher | None = None,
    ):
        self.repository = repository
        self.shelters = shelter_gateway
        self.routes = route_gateway
        self.authorization = authorization_gateway
        self.audit = audit_gateway
        self.notifications = notification_gateway

        self.risk_evaluator = (
            risk_evaluator
            or EvacuationRiskEvaluator()
        )

        self.shelter_matcher = (
            shelter_matcher
            or ShelterMatcher()
        )

    async def _audit(
        self,
        action: str,
        plan: EvacuationPlan,
        actor_id: int | None,
    ) -> None:
        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="evacuation_plan",
            entity_id=plan.id,
            metadata={
                "district_id": plan.district_id,
                "incident_id": plan.incident_id,
                "status": plan.status.value,
                "priority": plan.priority.value,
            },
        )

    async def get_plan(
        self,
        plan_id: str,
    ) -> EvacuationPlan:

        plan = await self.repository.get_by_id(
            plan_id
        )

        if plan is None:
            raise EvacuationNotFoundError(
                "Evacuation plan not found"
            )

        return plan

    def _validate_request(
        self,
        request: EvacuationRequest,
    ) -> None:

        if request.district_id <= 0:
            raise EvacuationValidationError(
                "Invalid district identifier"
            )

        if not request.incident_id.strip():
            raise EvacuationValidationError(
                "Incident ID is required"
            )

    async def recommend_shelters(
        self,
        request: EvacuationRequest,
        *,
        limit: int = 5,
    ) -> list[ShelterRecommendation]:

        self._validate_request(request)

        if not 1 <= limit <= 50:
            raise EvacuationValidationError(
                "Limit must be between 1 and 50"
            )

        shelters = await self.shelters.list_available(
            request.district_id
        )

        recommendations: list[
            ShelterRecommendation
        ] = []

        for shelter in shelters:
            if (
                request.preferred_shelter_id is not None
                and shelter.id
                != request.preferred_shelter_id
            ):
                continue

            if (
                shelter.available_capacity
                < request.population.total_people
            ):
                continue

            try:
                route = await self.routes.assess_route(
                    origin=request.origin,
                    destination=shelter.location,
                )

                recommendation = (
                    self.shelter_matcher.evaluate(
                        shelter=shelter,
                        request=request,
                        route=route,
                    )
                )

                if recommendation is not None:
                    recommendations.append(
                        recommendation
                    )

            except Exception:
                logger.exception(
                    "Shelter route assessment failed: %s",
                    shelter.id,
                )

        recommendations.sort(
            key=lambda item: (
                -item.score,
                item.shelter_id,
            )
        )

        return recommendations[:limit]

    async def create_plan(
        self,
        request: EvacuationRequest,
        *,
        shelter_id: str,
        actor_id: int,
    ) -> EvacuationPlan:

        self._validate_request(request)

        authorized = await self.authorization.can_plan(
            user_id=actor_id,
            district_id=request.district_id,
        )

        if not authorized:
            raise EvacuationAuthorizationError(
                "Evacuation planning permission denied"
            )

        recommendations = (
            await self.recommend_shelters(
                request,
                limit=50,
            )
        )

        selected = next(
            (
                item
                for item in recommendations
                if item.shelter_id == shelter_id
            ),
            None,
        )

        if selected is None:
            raise EvacuationCapacityError(
                "Selected shelter is not currently suitable"
            )

        risk_assessment = self.risk_evaluator.assess(
            request.risk
        )

        plan = EvacuationPlan(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            incident_id=request.incident_id,
            origin=request.origin,
            population=request.population,
            priority=risk_assessment.priority,
            risk_score=risk_assessment.risk_score,
            shelter_id=selected.shelter_id,
            route_id=selected.route.route_id,
            transport_modes=request.required_transport,
            requested_by=actor_id,
            status=EvacuationStatus.DRAFT,
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            plan
        )

        await self._audit(
            "evacuation.plan_created",
            created,
            actor_id,
        )

        return created

    async def submit_for_approval(
        self,
        plan_id: str,
        *,
        actor_id: int,
    ) -> EvacuationPlan:

        plan = await self.get_plan(
            plan_id
        )

        if not await self.authorization.can_plan(
            user_id=actor_id,
            district_id=plan.district_id,
        ):
            raise EvacuationAuthorizationError(
                "Planning permission denied"
            )

        if plan.status != EvacuationStatus.DRAFT:
            raise EvacuationConflictError(
                "Only draft plans can be submitted"
            )

        updated = replace(
            plan,
            status=EvacuationStatus.PENDING_APPROVAL,
            updated_at=utc_now(),
            version=plan.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=plan.version,
        )

        await self._audit(
            "evacuation.plan_submitted",
            saved,
            actor_id,
        )

        return saved

    async def approve_plan(
        self,
        plan_id: str,
        *,
        reviewer_id: int,
        approval_reference: str,
    ) -> EvacuationPlan:

        plan = await self.get_plan(
            plan_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=plan.district_id,
        ):
            raise EvacuationAuthorizationError(
                "Evacuation approval permission denied"
            )

        if (
            plan.status
            != EvacuationStatus.PENDING_APPROVAL
        ):
            raise EvacuationConflictError(
                "Plan is not pending approval"
            )

        if plan.requested_by == reviewer_id:
            raise EvacuationAuthorizationError(
                "Independent reviewer required"
            )

        if not approval_reference.strip():
            raise EvacuationValidationError(
                "Approval reference is required"
            )

        updated = replace(
            plan,
            status=EvacuationStatus.APPROVED,
            approved_by=reviewer_id,
            approved_at=utc_now(),
            approval_reference=approval_reference.strip(),
            updated_at=utc_now(),
            version=plan.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=plan.version,
        )

        await self._audit(
            "evacuation.plan_approved",
            saved,
            reviewer_id,
        )

        return saved

    async def activate_plan(
        self,
        plan_id: str,
        *,
        actor_id: int,
    ) -> EvacuationPlan:

        plan = await self.get_plan(
            plan_id
        )

        if not await self.authorization.can_activate(
            user_id=actor_id,
            district_id=plan.district_id,
        ):
            raise EvacuationAuthorizationError(
                "Evacuation activation permission denied"
            )

        if plan.status != EvacuationStatus.APPROVED:
            raise EvacuationConflictError(
                "Only approved plans can be activated"
            )

        if (
            plan.approved_by is None
            or plan.approved_at is None
            or not plan.approval_reference
        ):
            raise EvacuationAuthorizationError(
                "Verified human approval is missing"
            )

        shelter = await self.shelters.get_shelter(
            plan.shelter_id
        )

        if shelter is None:
            raise EvacuationCapacityError(
                "Shelter no longer exists"
            )

        if (
            not shelter.operational
            or not shelter.verified
            or shelter.district_id != plan.district_id
        ):
            raise EvacuationCapacityError(
                "Shelter is not operational or verified"
            )

        if (
            shelter.available_capacity
            < plan.population.total_people
        ):
            raise EvacuationCapacityError(
                "Insufficient shelter capacity"
            )

        # Verify the route again before activation.
        route = await self.routes.assess_route(
            origin=plan.origin,
            destination=shelter.location,
        )

        if not self.shelter_matcher.validate_route(
            route
        ):
            raise EvacuationRouteError(
                "Evacuation route is not currently verified"
            )

        reserved = await self.shelters.reserve_capacity(
            shelter_id=shelter.id,
            plan_id=plan.id,
            people=plan.population.total_people,
        )

        if not reserved:
            raise EvacuationCapacityError(
                "Shelter reservation failed"
            )

        try:
            updated = replace(
                plan,
                status=EvacuationStatus.ACTIVE,
                route_id=route.route_id,
                activated_by=actor_id,
                activated_at=utc_now(),
                updated_at=utc_now(),
                version=plan.version + 1,
            )

            saved = await self.repository.update(
                updated,
                expected_version=plan.version,
            )

        except Exception:
            await self.shelters.release_capacity(
                shelter_id=plan.shelter_id,
                plan_id=plan.id,
            )
            raise

        await self._audit(
            "evacuation.plan_activated",
            saved,
            actor_id,
        )

        if self.notifications is not None:
            try:
                delivered = (
                    await self.notifications.notify_activation(
                        saved
                    )
                )

                if not delivered:
                    logger.error(
                        "Evacuation activation notification "
                        "not confirmed for plan %s",
                        saved.id,
                    )

            except Exception:
                logger.exception(
                    "Evacuation notification failed: %s",
                    saved.id,
                )

        return saved

    async def update_progress(
        self,
        progress: EvacuationProgress,
    ) -> EvacuationPlan:

        plan = await self.get_plan(
            progress.plan_id
        )

        if plan.status not in {
            EvacuationStatus.ACTIVE,
            EvacuationStatus.PAUSED,
        }:
            raise EvacuationConflictError(
                "Progress updates require an active plan"
            )

        if not await self.authorization.can_activate(
            user_id=progress.updated_by,
            district_id=plan.district_id,
        ):
            raise EvacuationAuthorizationError(
                "Progress update permission denied"
            )

        if (
            progress.evacuated_people < 0
            or progress.arrived_people < 0
        ):
            raise EvacuationValidationError(
                "Progress counts cannot be negative"
            )

        if (
            progress.arrived_people
            > progress.evacuated_people
        ):
            raise EvacuationValidationError(
                "Arrivals cannot exceed evacuated people"
            )

        if (
            progress.evacuated_people
            > plan.population.total_people
        ):
            raise EvacuationValidationError(
                "Evacuated count exceeds planned population"
            )

        if (
            progress.evacuated_people
            < plan.evacuated_people
            or progress.arrived_people
            < plan.arrived_people
        ):
            raise EvacuationConflictError(
                "Progress counts cannot decrease"
            )

        updated = replace(
            plan,
            evacuated_people=progress.evacuated_people,
            arrived_people=progress.arrived_people,
            updated_at=utc_now(),
            version=plan.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=plan.version,
        )

        await self._audit(
            "evacuation.progress_updated",
            saved,
            progress.updated_by,
        )

        return saved

    async def pause_plan(
        self,
        plan_id: str,
        *,
        actor_id: int,
    ) -> EvacuationPlan:

        plan = await self.get_plan(
            plan_id
        )

        if not await self.authorization.can_activate(
            user_id=actor_id,
            district_id=plan.district_id,
        ):
            raise EvacuationAuthorizationError(
                "Evacuation control permission denied"
            )

        if plan.status != EvacuationStatus.ACTIVE:
            raise EvacuationConflictError(
                "Only active plans can be paused"
            )

        updated = replace(
            plan,
            status=EvacuationStatus.PAUSED,
            updated_at=utc_now(),
            version=plan.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=plan.version,
        )

        await self._audit(
            "evacuation.plan_paused",
            saved,
            actor_id,
        )

        return saved

    async def resume_plan(
        self,
        plan_id: str,
        *,
        actor_id: int,
    ) -> EvacuationPlan:

        plan = await self.get_plan(
            plan_id
        )

        if not await self.authorization.can_activate(
            user_id=actor_id,
            district_id=plan.district_id,
        ):
            raise EvacuationAuthorizationError(
                "Evacuation control permission denied"
            )

        if plan.status != EvacuationStatus.PAUSED:
            raise EvacuationConflictError(
                "Only paused plans can be resumed"
            )

        shelter = await self.shelters.get_shelter(
            plan.shelter_id
        )

        if shelter is None or not shelter.operational:
            raise EvacuationCapacityError(
                "Shelter is unavailable"
            )

        route = await self.routes.assess_route(
            origin=plan.origin,
            destination=shelter.location,
        )

        if not self.shelter_matcher.validate_route(
            route
        ):
            raise EvacuationRouteError(
                "Route verification failed"
            )

        updated = replace(
            plan,
            status=EvacuationStatus.ACTIVE,
            route_id=route.route_id,
            updated_at=utc_now(),
            version=plan.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=plan.version,
        )

        await self._audit(
            "evacuation.plan_resumed",
            saved,
            actor_id,
        )

        return saved

    async def complete_plan(
        self,
        plan_id: str,
        *,
        actor_id: int,
    ) -> EvacuationPlan:

        plan = await self.get_plan(
            plan_id
        )

        if not await self.authorization.can_activate(
            user_id=actor_id,
            district_id=plan.district_id,
        ):
            raise EvacuationAuthorizationError(
                "Evacuation completion permission denied"
            )

        if plan.status not in {
            EvacuationStatus.ACTIVE,
            EvacuationStatus.PAUSED,
        }:
            raise EvacuationConflictError(
                "Plan is not active or paused"
            )

        if (
            plan.arrived_people
            != plan.population.total_people
        ):
            raise EvacuationConflictError(
                "All planned evacuees must be accounted for "
                "before normal completion"
            )

        updated = replace(
            plan,
            status=EvacuationStatus.COMPLETED,
            completed_at=utc_now(),
            updated_at=utc_now(),
            version=plan.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=plan.version,
        )

        # Shelter reservation must be reconciled with
        # actual occupancy before releasing capacity.
        # Do not automatically release occupied beds.

        await self._audit(
            "evacuation.plan_completed",
            saved,
            actor_id,
        )

        return saved

    async def cancel_plan(
        self,
        plan_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> EvacuationPlan:

        plan = await self.get_plan(
            plan_id
        )

        if not await self.authorization.can_activate(
            user_id=actor_id,
            district_id=plan.district_id,
        ):
            raise EvacuationAuthorizationError(
                "Evacuation cancellation permission denied"
            )

        if plan.status in self.TERMINAL_STATUSES:
            raise EvacuationConflictError(
                "Plan is already in a terminal state"
            )

        if not reason.strip():
            raise EvacuationValidationError(
                "Cancellation reason is required"
            )

        updated = replace(
            plan,
            status=EvacuationStatus.CANCELLED,
            cancelled_at=utc_now(),
            cancellation_reason=reason.strip(),
            updated_at=utc_now(),
            version=plan.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=plan.version,
        )

        # Release only unused reserved capacity.
        # The shelter integration must preserve beds
        # already occupied by actual evacuees.
        if plan.status in {
            EvacuationStatus.ACTIVE,
            EvacuationStatus.PAUSED,
        }:
            await self.shelters.release_capacity(
                shelter_id=plan.shelter_id,
                plan_id=plan.id,
            )

        await self._audit(
            "evacuation.plan_cancelled",
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

    async def list_district_plans(
        self,
        district_id: int,
    ) -> Sequence[EvacuationPlan]:

        if district_id <= 0:
            raise EvacuationValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_incident_plans(
        self,
        incident_id: str,
    ) -> Sequence[EvacuationPlan]:
        return await self.repository.list_by_incident(
            incident_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        plans = await self.list_district_plans(
            district_id
        )

        status_counts = {
            status.value: 0
            for status in EvacuationStatus
        }

        priority_counts = {
            priority.value: 0
            for priority in EvacuationPriority
        }

        for plan in plans:
            status_counts[
                plan.status.value
            ] += 1

            priority_counts[
                plan.priority.value
            ] += 1

        return {
            "district_id": district_id,
            "total_plans": len(plans),
            "planned_people": sum(
                plan.population.total_people
                for plan in plans
                if plan.status
                != EvacuationStatus.CANCELLED
            ),
            "evacuated_people": sum(
                plan.evacuated_people
                for plan in plans
            ),
            "arrived_people": sum(
                plan.arrived_people
                for plan in plans
            ),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "generated_at": utc_now().isoformat(),
        }
