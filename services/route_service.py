
"""
backend/services/route_service.py

Kerala Flood Relief Coordination Agent
=======================================

Flood-Aware Route Planning and Safety Service.

Features:
- Rescue and evacuation route planning
- Ambulance and medical transport routes
- Relief logistics and volunteer routes
- Flood-affected road detection
- Road closure and blockage checks
- Bridge and landslide hazard assessment
- Vehicle compatibility verification
- Multi-route candidate evaluation
- Distance and travel-time estimation
- Route risk scoring and ranking
- Live route revalidation
- Human approval workflow
- Route expiry and invalidation
- Audit logging and district reporting

IMPORTANT:
- This module does not provide live road data itself.
- Route providers must supply verified geometry and
  current road/hazard information.
- AI route recommendations are advisory only.
- Unknown road safety must never mean "safe".
- High-risk or unverified routes cannot be approved.
- Emergency dispatch requires separate authorization.
- Travel times are provider estimates, not guarantees.
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

class RoutePurpose(str, Enum):
    RESCUE = "rescue"
    EVACUATION = "evacuation"
    AMBULANCE = "ambulance"
    MEDICAL_SUPPLY = "medical_supply"
    RELIEF_DELIVERY = "relief_delivery"
    VOLUNTEER_TRANSPORT = "volunteer_transport"
    SHELTER_TRANSFER = "shelter_transfer"
    FIELD_INSPECTION = "field_inspection"


class RouteStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class RouteSafetyLevel(str, Enum):
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    EXTREME_RISK = "extreme_risk"
    UNKNOWN = "unknown"


class RoadCondition(str, Enum):
    OPEN = "open"
    RESTRICTED = "restricted"
    FLOODED = "flooded"
    BLOCKED = "blocked"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class RouteHazardType(str, Enum):
    FLOOD_WATER = "flood_water"
    STRONG_CURRENT = "strong_current"
    LANDSLIDE = "landslide"
    BRIDGE_DAMAGE = "bridge_damage"
    ROAD_COLLAPSE = "road_collapse"
    FALLEN_TREE = "fallen_tree"
    ELECTRICAL_HAZARD = "electrical_hazard"
    DEBRIS = "debris"
    LOW_VISIBILITY = "low_visibility"
    OTHER = "other"


class RouteVehicleType(str, Enum):
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    VAN = "van"
    AMBULANCE = "ambulance"
    PICKUP = "pickup"
    TRUCK = "truck"
    RESCUE_VEHICLE = "rescue_vehicle"
    BUS = "bus"
    BOAT = "boat"


class RouteTravelMode(str, Enum):
    DRIVING = "driving"
    WALKING = "walking"
    BOAT = "boat"


class RouteVerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    STALE = "stale"


# ============================================================
# EXCEPTIONS
# ============================================================

class RouteServiceError(Exception):
    pass


class RouteValidationError(RouteServiceError):
    pass


class RouteNotFoundError(RouteServiceError):
    pass


class RouteConflictError(RouteServiceError):
    pass


class RouteAuthorizationError(RouteServiceError):
    pass


class RouteSafetyError(RouteServiceError):
    pass


class RouteIntegrationError(RouteServiceError):
    pass


class RouteDataError(RouteServiceError):
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
        raise RouteValidationError(
            f"{name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise RouteValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if not isinstance(value, datetime):
        raise RouteValidationError(
            f"{name} must be a datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise RouteValidationError(
            f"{name} must be timezone-aware"
        )


def validate_non_negative(
    value: float | int,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RouteValidationError(
            f"{name} must be a non-negative finite number"
        )

    return float(value)


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    radius_km = 6371.0088

    lat1 = math.radians(latitude_1)
    lon1 = math.radians(longitude_1)
    lat2 = math.radians(latitude_2)
    lon2 = math.radians(longitude_2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = clamp(a)

    return (
        2
        * radius_km
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a),
        )
    )


# ============================================================
# LOCATION MODELS
# ============================================================

@dataclass(frozen=True)
class RouteLocation:
    latitude: float
    longitude: float

    district_id: int | None = None

    name: str = ""
    address: str = ""

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("latitude", self.latitude, -90, 90),
            ("longitude", self.longitude, -180, 180),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not minimum <= value <= maximum
            ):
                raise RouteValidationError(
                    f"Invalid {name}"
                )

        if (
            self.district_id is not None
            and self.district_id <= 0
        ):
            raise RouteValidationError(
                "Invalid district ID"
            )


@dataclass(frozen=True)
class RouteWaypoint:
    latitude: float
    longitude: float

    sequence: int

    name: str = ""

    def __post_init__(self) -> None:
        RouteLocation(
            latitude=self.latitude,
            longitude=self.longitude,
        )

        if self.sequence < 0:
            raise RouteValidationError(
                "Waypoint sequence cannot be negative"
            )


# ============================================================
# ROAD AND HAZARD MODELS
# ============================================================

@dataclass(frozen=True)
class RoadSegment:
    segment_id: str

    condition: RoadCondition

    length_km: float

    verified: bool = False

    observed_at: datetime | None = None

    allowed_vehicle_types: frozenset[
        RouteVehicleType
    ] = field(
        default_factory=frozenset
    )

    notes: str = ""

    def __post_init__(self) -> None:
        require_text(
            self.segment_id,
            "segment_id",
        )

        validate_non_negative(
            self.length_km,
            "length_km",
        )

        if self.observed_at is not None:
            validate_datetime(
                self.observed_at,
                "observed_at",
            )


@dataclass(frozen=True)
class RouteHazard:
    id: str

    hazard_type: RouteHazardType

    severity: float

    segment_id: str | None = None

    verified: bool = False

    observed_at: datetime = field(
        default_factory=utc_now
    )

    description: str = ""

    def __post_init__(self) -> None:
        require_text(
            self.id,
            "hazard ID",
        )

        if (
            isinstance(self.severity, bool)
            or not isinstance(self.severity, (int, float))
            or not math.isfinite(self.severity)
            or not 0 <= self.severity <= 1
        ):
            raise RouteValidationError(
                "Hazard severity must be between 0 and 1"
            )

        validate_datetime(
            self.observed_at,
            "observed_at",
        )


# ============================================================
# ROUTE REQUEST
# ============================================================

@dataclass(frozen=True)
class RoutePlanRequest:
    district_id: int

    purpose: RoutePurpose

    origin: RouteLocation
    destination: RouteLocation

    vehicle_type: RouteVehicleType

    travel_mode: RouteTravelMode = (
        RouteTravelMode.DRIVING
    )

    incident_id: str | None = None
    rescue_mission_id: str | None = None
    evacuation_id: str | None = None
    shipment_id: str | None = None

    maximum_candidates: int = 5

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.district_id <= 0:
            raise RouteValidationError(
                "Invalid district ID"
            )

        if not 1 <= self.maximum_candidates <= 10:
            raise RouteValidationError(
                "maximum_candidates must be between 1 and 10"
            )

        if (
            self.origin.district_id is not None
            and self.origin.district_id != self.district_id
        ):
            raise RouteValidationError(
                "Origin district mismatch"
            )


# ============================================================
# ROUTE CANDIDATES
# ============================================================

@dataclass(frozen=True)
class RouteCandidate:
    provider_route_id: str

    distance_km: float
    estimated_duration_minutes: float

    waypoints: tuple[RouteWaypoint, ...]

    road_segments: tuple[RoadSegment, ...]

    hazards: tuple[RouteHazard, ...] = ()

    provider_name: str = ""

    generated_at: datetime = field(
        default_factory=utc_now
    )

    provider_verified: bool = False

    def __post_init__(self) -> None:
        require_text(
            self.provider_route_id,
            "provider_route_id",
        )

        validate_non_negative(
            self.distance_km,
            "distance_km",
        )

        validate_non_negative(
            self.estimated_duration_minutes,
            "estimated_duration_minutes",
        )

        validate_datetime(
            self.generated_at,
            "generated_at",
        )


@dataclass(frozen=True)
class RouteSafetyAssessment:
    safety_level: RouteSafetyLevel

    risk_score: float

    feasible: bool

    verification_status: RouteVerificationStatus

    blocked_segments: tuple[str, ...] = ()

    restricted_segments: tuple[str, ...] = ()

    warnings: tuple[str, ...] = ()

    assessed_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class RouteRecommendation:
    provider_route_id: str

    distance_km: float
    estimated_duration_minutes: float

    safety: RouteSafetyAssessment

    ranking_score: float

    recommended: bool

    explanation: str


@dataclass
class PlannedRoute:
    id: str
    district_id: int

    purpose: RoutePurpose

    origin: RouteLocation
    destination: RouteLocation

    vehicle_type: RouteVehicleType
    travel_mode: RouteTravelMode

    candidate: RouteCandidate

    safety: RouteSafetyAssessment

    status: RouteStatus

    created_by: int

    incident_id: str | None = None
    rescue_mission_id: str | None = None
    evacuation_id: str | None = None
    shipment_id: str | None = None

    reviewed_by: int | None = None
    reviewed_at: datetime | None = None

    approved_by: int | None = None
    approved_at: datetime | None = None

    expires_at: datetime | None = None

    rejection_reason: str | None = None
    invalidation_reason: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# INTEGRATION CONTRACTS
# ============================================================

class RouteProviderGateway(Protocol):

    async def get_candidate_routes(
        self,
        request: RoutePlanRequest,
    ) -> Sequence[RouteCandidate]:
        """
        Return routes using a real mapping provider.

        Must not invent route geometry or travel times.
        """
        ...


class RouteHazardGateway(Protocol):

    async def get_current_segments(
        self,
        candidate: RouteCandidate,
    ) -> Sequence[RoadSegment]:
        ...

    async def get_current_hazards(
        self,
        candidate: RouteCandidate,
    ) -> Sequence[RouteHazard]:
        ...


class RouteRepository(Protocol):

    async def create(
        self,
        route: PlannedRoute,
    ) -> PlannedRoute:
        ...

    async def get_by_id(
        self,
        route_id: str,
    ) -> PlannedRoute | None:
        ...

    async def update(
        self,
        route: PlannedRoute,
        *,
        expected_version: int,
    ) -> PlannedRoute:
        """
        Must implement atomic optimistic concurrency.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[PlannedRoute]:
        ...

    async def list_by_mission(
        self,
        rescue_mission_id: str,
    ) -> Sequence[PlannedRoute]:
        ...


class RouteAuthorizationGateway(Protocol):

    async def can_plan(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_review(
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

    async def can_view(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class RouteAuditGateway(Protocol):

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
# ROUTE SAFETY CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RouteSafetyConfig:
    maximum_road_data_age: timedelta = timedelta(
        minutes=15
    )

    maximum_hazard_data_age: timedelta = timedelta(
        minutes=15
    )

    maximum_route_age: timedelta = timedelta(
        minutes=10
    )

    approved_route_validity: timedelta = timedelta(
        minutes=15
    )

    maximum_acceptable_risk: float = 0.35

    def __post_init__(self) -> None:
        for name in (
            "maximum_road_data_age",
            "maximum_hazard_data_age",
            "maximum_route_age",
            "approved_route_validity",
        ):
            if getattr(self, name) <= timedelta(0):
                raise RouteValidationError(
                    f"{name} must be positive"
                )

        if not 0 <= self.maximum_acceptable_risk <= 1:
            raise RouteValidationError(
                "Invalid maximum acceptable risk"
            )


# ============================================================
# ROUTE SAFETY EVALUATOR
# ============================================================

class RouteSafetyEvaluator:

    BLOCKED_CONDITIONS = frozenset({
        RoadCondition.FLOODED,
        RoadCondition.BLOCKED,
        RoadCondition.CLOSED,
    })

    CRITICAL_HAZARDS = frozenset({
        RouteHazardType.STRONG_CURRENT,
        RouteHazardType.BRIDGE_DAMAGE,
        RouteHazardType.ROAD_COLLAPSE,
        RouteHazardType.ELECTRICAL_HAZARD,
    })

    def __init__(
        self,
        config: RouteSafetyConfig | None = None,
    ):
        self.config = (
            config
            or RouteSafetyConfig()
        )

    def evaluate(
        self,
        candidate: RouteCandidate,
        *,
        vehicle_type: RouteVehicleType,
        now: datetime | None = None,
    ) -> RouteSafetyAssessment:

        now = now or utc_now()

        validate_datetime(
            now,
            "now",
        )

        warnings: list[str] = []
        blocked: list[str] = []
        restricted: list[str] = []

        verified = candidate.provider_verified

        risk = 0.0
        critical_hazard = False

        if (
            now - candidate.generated_at
            > self.config.maximum_route_age
        ):
            verified = False
            warnings.append(
                "Route geometry is stale"
            )

        if not candidate.road_segments:
            verified = False
            warnings.append(
                "No road segment verification data"
            )

        for segment in candidate.road_segments:
            if segment.condition in self.BLOCKED_CONDITIONS:
                blocked.append(
                    segment.segment_id
                )
                risk = max(risk, 1.0)

            elif segment.condition == RoadCondition.RESTRICTED:
                restricted.append(
                    segment.segment_id
                )
                risk = max(risk, 0.5)

            elif segment.condition == RoadCondition.UNKNOWN:
                verified = False
                warnings.append(
                    f"Unknown road condition: "
                    f"{segment.segment_id}"
                )

            if (
                segment.allowed_vehicle_types
                and vehicle_type
                not in segment.allowed_vehicle_types
            ):
                blocked.append(
                    segment.segment_id
                )
                warnings.append(
                    "Vehicle not permitted on segment "
                    f"{segment.segment_id}"
                )

            if not segment.verified:
                verified = False

            if segment.observed_at is None:
                verified = False

            else:
                age = now - segment.observed_at

                if (
                    age > self.config.maximum_road_data_age
                    or age < -timedelta(minutes=5)
                ):
                    verified = False
                    warnings.append(
                        f"Stale road data: {segment.segment_id}"
                    )

        for hazard in candidate.hazards:
            risk = max(
                risk,
                hazard.severity,
            )

            if (
                hazard.hazard_type in self.CRITICAL_HAZARDS
                and hazard.severity >= 0.5
            ):
                critical_hazard = True

            if not hazard.verified:
                verified = False

            age = now - hazard.observed_at

            if (
                age > self.config.maximum_hazard_data_age
                or age < -timedelta(minutes=5)
            ):
                verified = False
                warnings.append(
                    f"Stale hazard data: {hazard.id}"
                )

        if blocked:
            safety_level = RouteSafetyLevel.EXTREME_RISK

        elif critical_hazard or risk >= 0.75:
            safety_level = RouteSafetyLevel.EXTREME_RISK

        elif risk >= 0.5:
            safety_level = RouteSafetyLevel.HIGH_RISK

        elif risk >= 0.25:
            safety_level = RouteSafetyLevel.MODERATE_RISK

        else:
            safety_level = RouteSafetyLevel.LOW_RISK

        if not verified:
            safety_level = RouteSafetyLevel.UNKNOWN

        feasible = (
            verified
            and not blocked
            and not critical_hazard
            and risk <= self.config.maximum_acceptable_risk
        )

        verification_status = (
            RouteVerificationStatus.VERIFIED
            if verified
            else RouteVerificationStatus.UNVERIFIED
        )

        if any(
            "stale" in warning.lower()
            for warning in warnings
        ):
            verification_status = (
                RouteVerificationStatus.STALE
            )

        return RouteSafetyAssessment(
            safety_level=safety_level,
            risk_score=round(risk, 4),
            feasible=feasible,
            verification_status=verification_status,
            blocked_segments=tuple(
                dict.fromkeys(blocked)
            ),
            restricted_segments=tuple(
                dict.fromkeys(restricted)
            ),
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            assessed_at=now,
        )


# ============================================================
# ROUTE RANKING ENGINE
# ============================================================

class RouteRankingEngine:

    def rank(
        self,
        candidates: Sequence[RouteCandidate],
        assessments: Mapping[
            str,
            RouteSafetyAssessment,
        ],
    ) -> list[RouteRecommendation]:

        if not candidates:
            return []

        max_distance = max(
            max(candidate.distance_km, 0.001)
            for candidate in candidates
        )

        max_duration = max(
            max(candidate.estimated_duration_minutes, 0.001)
            for candidate in candidates
        )

        recommendations: list[RouteRecommendation] = []

        for candidate in candidates:
            safety = assessments[
                candidate.provider_route_id
            ]

            safety_score = (
                1.0 - safety.risk_score
            )

            distance_score = (
                1.0
                - candidate.distance_km / max_distance
            )

            duration_score = (
                1.0
                - candidate.estimated_duration_minutes
                / max_duration
            )

            score = (
                0.65 * safety_score
                + 0.15 * distance_score
                + 0.20 * duration_score
            )

            if not safety.feasible:
                score = 0.0

            recommendations.append(
                RouteRecommendation(
                    provider_route_id=(
                        candidate.provider_route_id
                    ),
                    distance_km=candidate.distance_km,
                    estimated_duration_minutes=(
                        candidate.estimated_duration_minutes
                    ),
                    safety=safety,
                    ranking_score=round(
                        clamp(score),
                        4,
                    ),
                    recommended=False,
                    explanation=(
                        "Verified candidate evaluated for "
                        "road safety, distance and "
                        "estimated travel duration."
                        if safety.feasible
                        else
                        "Route is not currently eligible "
                        "for operational approval."
                    ),
                )
            )

        recommendations.sort(
            key=lambda item: (
                not item.safety.feasible,
                -item.ranking_score,
                item.estimated_duration_minutes,
            )
        )

        for index, item in enumerate(recommendations):
            if index == 0 and item.safety.feasible:
                recommendations[index] = replace(
                    item,
                    recommended=True,
                )

        return recommendations


# ============================================================
# ROUTE SERVICE
# ============================================================

class RouteService:

    def __init__(
        self,
        *,
        repository: RouteRepository,
        provider_gateway: RouteProviderGateway,
        hazard_gateway: RouteHazardGateway,
        authorization_gateway: RouteAuthorizationGateway,
        audit_gateway: RouteAuditGateway | None = None,
        safety_evaluator: RouteSafetyEvaluator | None = None,
        ranking_engine: RouteRankingEngine | None = None,
    ):
        self.repository = repository
        self.provider = provider_gateway
        self.hazards = hazard_gateway
        self.authorization = authorization_gateway
        self.audit = audit_gateway

        self.safety_evaluator = (
            safety_evaluator
            or RouteSafetyEvaluator()
        )

        self.ranking_engine = (
            ranking_engine
            or RouteRankingEngine()
        )

    async def _audit(
        self,
        action: str,
        route: PlannedRoute,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="planned_route",
            entity_id=route.id,
            metadata={
                "district_id": route.district_id,
                "status": route.status.value,
                "purpose": route.purpose.value,
                "version": route.version,
            },
        )

    async def _save(
        self,
        original: PlannedRoute,
        updated: PlannedRoute,
        *,
        action: str,
        actor_id: int,
    ) -> PlannedRoute:

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

    async def get_route(
        self,
        route_id: str,
    ) -> PlannedRoute:

        route = await self.repository.get_by_id(
            route_id
        )

        if route is None:
            raise RouteNotFoundError(
                "Route not found"
            )

        return route

    async def _refresh_candidate(
        self,
        candidate: RouteCandidate,
    ) -> RouteCandidate:

        segments = (
            await self.hazards.get_current_segments(
                candidate
            )
        )

        hazards = (
            await self.hazards.get_current_hazards(
                candidate
            )
        )

        # Empty segment data must remain unverified;
        # the service does not fall back to assuming
        # roads are open.
        return replace(
            candidate,
            road_segments=tuple(segments),
            hazards=tuple(hazards),
        )

    async def recommend_routes(
        self,
        request: RoutePlanRequest,
        *,
        actor_id: int,
    ) -> list[RouteRecommendation]:

        if not await self.authorization.can_plan(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise RouteAuthorizationError(
                "Route planning permission denied"
            )

        candidates = await self.provider.get_candidate_routes(
            request
        )

        candidates = list(candidates)[
            :request.maximum_candidates
        ]

        if not candidates:
            return []

        assessments: dict[
            str,
            RouteSafetyAssessment,
        ] = {}

        refreshed_candidates: list[
            RouteCandidate
        ] = []

        for candidate in candidates:
            refreshed = await self._refresh_candidate(
                candidate
            )

            refreshed_candidates.append(
                refreshed
            )

            assessments[
                refreshed.provider_route_id
            ] = self.safety_evaluator.evaluate(
                refreshed,
                vehicle_type=request.vehicle_type,
            )

        return self.ranking_engine.rank(
            refreshed_candidates,
            assessments,
        )

    async def create_route(
        self,
        request: RoutePlanRequest,
        *,
        provider_route_id: str,
        actor_id: int,
    ) -> PlannedRoute:

        if not await self.authorization.can_plan(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise RouteAuthorizationError(
                "Route creation permission denied"
            )

        provider_route_id = require_text(
            provider_route_id,
            "provider_route_id",
        )

        candidates = await self.provider.get_candidate_routes(
            request
        )

        candidate = next(
            (
                item
                for item in candidates
                if item.provider_route_id
                == provider_route_id
            ),
            None,
        )

        if candidate is None:
            raise RouteNotFoundError(
                "Provider route not found"
            )

        candidate = await self._refresh_candidate(
            candidate
        )

        safety = self.safety_evaluator.evaluate(
            candidate,
            vehicle_type=request.vehicle_type,
        )

        route = PlannedRoute(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            purpose=request.purpose,
            origin=request.origin,
            destination=request.destination,
            vehicle_type=request.vehicle_type,
            travel_mode=request.travel_mode,
            candidate=candidate,
            safety=safety,
            status=RouteStatus.DRAFT,
            created_by=actor_id,
            incident_id=request.incident_id,
            rescue_mission_id=request.rescue_mission_id,
            evacuation_id=request.evacuation_id,
            shipment_id=request.shipment_id,
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            route
        )

        await self._audit(
            "route.created",
            created,
            actor_id,
        )

        return created

    async def submit_for_review(
        self,
        route_id: str,
        *,
        actor_id: int,
    ) -> PlannedRoute:

        route = await self.get_route(
            route_id
        )

        if not await self.authorization.can_plan(
            user_id=actor_id,
            district_id=route.district_id,
        ):
            raise RouteAuthorizationError(
                "Route submission denied"
            )

        if route.status != RouteStatus.DRAFT:
            raise RouteConflictError(
                "Only draft routes can be submitted"
            )

        updated = replace(
            route,
            status=RouteStatus.PENDING_REVIEW,
            updated_at=utc_now(),
            version=route.version + 1,
        )

        return await self._save(
            route,
            updated,
            action="route.submitted",
            actor_id=actor_id,
        )

    async def approve_route(
        self,
        route_id: str,
        *,
        reviewer_id: int,
    ) -> PlannedRoute:

        route = await self.get_route(
            route_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=route.district_id,
        ):
            raise RouteAuthorizationError(
                "Route approval permission denied"
            )

        if route.status != RouteStatus.PENDING_REVIEW:
            raise RouteConflictError(
                "Route is not awaiting review"
            )

        if reviewer_id == route.created_by:
            raise RouteAuthorizationError(
                "Independent route reviewer required"
            )

        refreshed = await self._refresh_candidate(
            route.candidate
        )

        safety = self.safety_evaluator.evaluate(
            refreshed,
            vehicle_type=route.vehicle_type,
        )

        if not safety.feasible:
            raise RouteSafetyError(
                "Route is not verified as safe for approval"
            )

        now = utc_now()

        updated = replace(
            route,
            candidate=refreshed,
            safety=safety,
            status=RouteStatus.APPROVED,
            reviewed_by=reviewer_id,
            reviewed_at=now,
            approved_by=reviewer_id,
            approved_at=now,
            expires_at=(
                now
                + self.safety_evaluator
                .config.approved_route_validity
            ),
            updated_at=now,
            version=route.version + 1,
        )

        return await self._save(
            route,
            updated,
            action="route.approved",
            actor_id=reviewer_id,
        )

    async def reject_route(
        self,
        route_id: str,
        *,
        reviewer_id: int,
        reason: str,
    ) -> PlannedRoute:

        route = await self.get_route(
            route_id
        )

        if not await self.authorization.can_review(
            user_id=reviewer_id,
            district_id=route.district_id,
        ):
            raise RouteAuthorizationError(
                "Route rejection permission denied"
            )

        if route.status != RouteStatus.PENDING_REVIEW:
            raise RouteConflictError(
                "Route is not awaiting review"
            )

        updated = replace(
            route,
            status=RouteStatus.REJECTED,
            reviewed_by=reviewer_id,
            reviewed_at=utc_now(),
            rejection_reason=require_text(
                reason,
                "Rejection reason",
            ),
            updated_at=utc_now(),
            version=route.version + 1,
        )

        return await self._save(
            route,
            updated,
            action="route.rejected",
            actor_id=reviewer_id,
        )

    async def revalidate_route(
        self,
        route_id: str,
        *,
        actor_id: int,
    ) -> PlannedRoute:

        route = await self.get_route(
            route_id
        )

        if not await self.authorization.can_plan(
            user_id=actor_id,
            district_id=route.district_id,
        ):
            raise RouteAuthorizationError(
                "Route revalidation denied"
            )

        if route.status in {
            RouteStatus.REJECTED,
            RouteStatus.INVALIDATED,
            RouteStatus.EXPIRED,
        }:
            raise RouteConflictError(
                "Cannot revalidate a terminal route"
            )

        refreshed = await self._refresh_candidate(
            route.candidate
        )

        safety = self.safety_evaluator.evaluate(
            refreshed,
            vehicle_type=route.vehicle_type,
        )

        now = utc_now()

        new_status = route.status

        if route.status == RouteStatus.APPROVED:
            if not safety.feasible:
                new_status = RouteStatus.INVALIDATED

            elif (
                route.expires_at is not None
                and now >= route.expires_at
            ):
                new_status = RouteStatus.EXPIRED

        updated = replace(
            route,
            candidate=refreshed,
            safety=safety,
            status=new_status,
            invalidation_reason=(
                "Route failed live safety revalidation"
                if new_status == RouteStatus.INVALIDATED
                else route.invalidation_reason
            ),
            updated_at=now,
            version=route.version + 1,
        )

        return await self._save(
            route,
            updated,
            action="route.revalidated",
            actor_id=actor_id,
        )

    async def invalidate_route(
        self,
        route_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> PlannedRoute:

        route = await self.get_route(
            route_id
        )

        if not await self.authorization.can_review(
            user_id=actor_id,
            district_id=route.district_id,
        ):
            raise RouteAuthorizationError(
                "Route invalidation denied"
            )

        if route.status in {
            RouteStatus.INVALIDATED,
            RouteStatus.EXPIRED,
            RouteStatus.REJECTED,
        }:
            raise RouteConflictError(
                "Route is already terminal"
            )

        updated = replace(
            route,
            status=RouteStatus.INVALIDATED,
            invalidation_reason=require_text(
                reason,
                "Invalidation reason",
            ),
            updated_at=utc_now(),
            version=route.version + 1,
        )

        return await self._save(
            route,
            updated,
            action="route.invalidated",
            actor_id=actor_id,
        )

    async def is_route_usable(
        self,
        route_id: str,
        *,
        actor_id: int,
    ) -> bool:

        route = await self.get_route(
            route_id
        )

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=route.district_id,
        ):
            raise RouteAuthorizationError(
                "Route access denied"
            )

        if route.status != RouteStatus.APPROVED:
            return False

        if route.expires_at is None:
            return False

        if utc_now() >= route.expires_at:
            return False

        # A route must be revalidated before use.
        # Approval alone does not guarantee that
        # road conditions remain unchanged.
        refreshed = await self._refresh_candidate(
            route.candidate
        )

        safety = self.safety_evaluator.evaluate(
            refreshed,
            vehicle_type=route.vehicle_type,
        )

        return safety.feasible

    async def list_district_routes(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> Sequence[PlannedRoute]:

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise RouteAuthorizationError(
                "District route access denied"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_rescue_routes(
        self,
        rescue_mission_id: str,
        *,
        actor_id: int,
    ) -> list[PlannedRoute]:

        routes = await self.repository.list_by_mission(
            rescue_mission_id
        )

        visible = []

        for route in routes:
            if await self.authorization.can_view(
                user_id=actor_id,
                district_id=route.district_id,
            ):
                visible.append(route)

        return visible

    async def get_district_statistics(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        routes = await self.list_district_routes(
            district_id,
            actor_id=actor_id,
        )

        status_counts = {
            status.value: 0
            for status in RouteStatus
        }

        safety_counts = {
            level.value: 0
            for level in RouteSafetyLevel
        }

        purpose_counts = {
            purpose.value: 0
            for purpose in RoutePurpose
        }

        for route in routes:
            status_counts[
                route.status.value
            ] += 1

            safety_counts[
                route.safety.safety_level.value
            ] += 1

            purpose_counts[
                route.purpose.value
            ] += 1

        now = utc_now()

        valid_approved = sum(
            route.status == RouteStatus.APPROVED
            and route.expires_at is not None
            and now < route.expires_at
            for route in routes
        )

        return {
            "district_id": district_id,
            "total_routes": len(routes),
            "approved_not_expired_routes": (
                valid_approved
            ),
            "status_counts": status_counts,
            "safety_counts": safety_counts,
            "purpose_counts": purpose_counts,
            "generated_at": now.isoformat(),
            "warning": (
                "Approved-not-expired counts do not "
                "guarantee current road safety. "
                "Each route requires live revalidation."
            ),
        }

    async def generate_route_report(
        self,
        route_id: str,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        route = await self.get_route(
            route_id
        )

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=route.district_id,
        ):
            raise RouteAuthorizationError(
                "Route report access denied"
            )

        return {
            "route_id": route.id,
            "provider_route_id": (
                route.candidate.provider_route_id
            ),
            "district_id": route.district_id,
            "purpose": route.purpose.value,
            "status": route.status.value,
            "vehicle_type": route.vehicle_type.value,
            "travel_mode": route.travel_mode.value,
            "origin": {
                "latitude": route.origin.latitude,
                "longitude": route.origin.longitude,
                "name": route.origin.name,
            },
            "destination": {
                "latitude": route.destination.latitude,
                "longitude": route.destination.longitude,
                "name": route.destination.name,
            },
            "distance_km": route.candidate.distance_km,
            "estimated_duration_minutes": (
                route.candidate.estimated_duration_minutes
            ),
            "safety": {
                "level": route.safety.safety_level.value,
                "risk_score": route.safety.risk_score,
                "feasible": route.safety.feasible,
                "verification_status": (
                    route.safety.verification_status.value
                ),
                "blocked_segments": list(
                    route.safety.blocked_segments
                ),
                "restricted_segments": list(
                    route.safety.restricted_segments
                ),
                "warnings": list(
                    route.safety.warnings
                ),
                "assessed_at": (
                    route.safety.assessed_at.isoformat()
                ),
            },
            "hazards": [
                {
                    "hazard_id": hazard.id,
                    "hazard_type": hazard.hazard_type.value,
                    "severity": hazard.severity,
                    "verified": hazard.verified,
                }
                for hazard in route.candidate.hazards
            ],
            "approved_by": route.approved_by,
            "approved_at": (
                route.approved_at.isoformat()
                if route.approved_at
                else None
            ),
            "expires_at": (
                route.expires_at.isoformat()
                if route.expires_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Route recommendations are advisory. "
                "Current road and flood conditions must "
                "be verified before travel. Never enter "
                "floodwater or closed roads based on "
                "this report."
            ),
        }
