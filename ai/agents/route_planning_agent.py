
"""
backend/ai/agents/route_planning_agent.py

Kerala Flood Relief Coordination Agent
Emergency Route Planning & Safety Intelligence

Python 3.11+
Standard library only.

FEATURES
--------
1. Flood-aware road network representation
2. Emergency vehicle capability matching
3. Verified road and bridge restrictions
4. Flood depth and road blockage screening
5. Road observation freshness validation
6. Route planning using Dijkstra's algorithm
7. Alternative route generation
8. Distance, duration and risk estimation
9. Route evidence provenance
10. Human approval workflow
11. Route audit logging
12. JSON report export

SAFETY
------
This module provides advisory route candidates only.

A computed route is NOT proof of safe passage.

- No navigation commands are transmitted.
- No rescue or medical vehicle is dispatched.
- No road closure is authorized.
- No floodwater crossing is authorized.
- Unknown or stale road conditions are excluded.
- Flood-depth limits are illustrative screening
  parameters, not vehicle operating permissions.
- Route approval does not authorize vehicle movement.
- Field verification and incident-command approval
  are mandatory before operational use.
"""

from __future__ import annotations

import heapq
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

class RoutePlanningError(Exception):
    pass


class RouteValidationError(RoutePlanningError):
    pass


class RouteNotFoundError(RoutePlanningError):
    pass


class RouteApprovalError(RoutePlanningError):
    pass


class RouteExportError(RoutePlanningError):
    pass


# ============================================================
# ENUMS
# ============================================================

class EmergencyMissionType(str, Enum):
    RESCUE = "rescue"
    AMBULANCE = "ambulance"
    EVACUATION = "evacuation"
    RELIEF_DELIVERY = "relief_delivery"
    MEDICAL_SUPPLY = "medical_supply"
    INFRASTRUCTURE_INSPECTION = "infrastructure_inspection"


class VehicleType(str, Enum):
    AMBULANCE = "ambulance"
    RESCUE_VAN = "rescue_van"
    RESCUE_TRUCK = "rescue_truck"
    EVACUATION_BUS = "evacuation_bus"
    RELIEF_TRUCK = "relief_truck"
    FOUR_WHEEL_DRIVE = "four_wheel_drive"
    MOTORCYCLE = "motorcycle"


class RoadStatus(str, Enum):
    VERIFIED_OPEN = "verified_open"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class RoadSurface(str, Enum):
    PAVED = "paved"
    GRAVEL = "gravel"
    UNPAVED = "unpaved"
    UNKNOWN = "unknown"


class RouteRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNDETERMINED = "undetermined"


class RoutePlanStatus(str, Enum):
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
        raise RouteValidationError(
            "Timezone-aware datetime required"
        )


def validate_nonnegative(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RouteValidationError(
            f"{name} must be nonnegative and finite"
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    validate_nonnegative(
        value,
        name,
    )

    if value > 1:
        raise RouteValidationError(
            f"{name} must be <= 1"
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
class RoutePlanningConfig:
    max_nodes: int = 100000
    max_edges: int = 500000

    max_alternative_routes: int = 3

    max_road_observation_age_minutes: int = 20

    minimum_evidence_reliability: float = 0.70

    # Conservative default: do not plan through
    # reported standing floodwater.
    maximum_screened_flood_depth_m: float = 0.0

    risk_weight: float = 2.0

    require_verified_open_roads: bool = True
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_nodes",
            "max_edges",
            "max_alternative_routes",
            "max_road_observation_age_minutes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RouteValidationError(
                    f"{name} must be positive integer"
                )

        validate_probability(
            self.minimum_evidence_reliability,
            "minimum_evidence_reliability",
        )

        validate_nonnegative(
            self.maximum_screened_flood_depth_m,
            "maximum_screened_flood_depth_m",
        )

        validate_nonnegative(
            self.risk_weight,
            "risk_weight",
        )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class RoadNode:
    node_id: str
    name: str

    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise RouteValidationError(
                "node_id required"
            )

        if self.latitude is not None:
            if (
                not math.isfinite(self.latitude)
                or not -90 <= self.latitude <= 90
            ):
                raise RouteValidationError(
                    "Invalid latitude"
                )

        if self.longitude is not None:
            if (
                not math.isfinite(self.longitude)
                or not -180 <= self.longitude <= 180
            ):
                raise RouteValidationError(
                    "Invalid longitude"
                )


@dataclass(frozen=True)
class RoadEdge:
    edge_id: str

    source_node_id: str
    target_node_id: str

    distance_km: float
    estimated_travel_minutes: float

    road_status: RoadStatus
    road_surface: RoadSurface

    flood_depth_m: float

    risk_score: float

    bridge_present: bool = False
    bridge_verified_safe: bool = False

    max_vehicle_weight_tonnes: float | None = None

    allowed_vehicle_types: tuple[
        VehicleType,
        ...
    ] = ()

    evidence_source_id: str = ""
    evidence_reliability: float = 0.0
    verified_at: datetime | None = None

    bidirectional: bool = True

    def __post_init__(self) -> None:
        for name in (
            "edge_id",
            "source_node_id",
            "target_node_id",
        ):
            if not getattr(self, name).strip():
                raise RouteValidationError(
                    f"{name} required"
                )

        if self.source_node_id == self.target_node_id:
            raise RouteValidationError(
                "Self-loop road edges are not supported"
            )

        for name in (
            "distance_km",
            "estimated_travel_minutes",
            "flood_depth_m",
        ):
            validate_nonnegative(
                getattr(self, name),
                name,
            )

        if self.distance_km <= 0:
            raise RouteValidationError(
                "Road distance must be positive"
            )

        if self.estimated_travel_minutes <= 0:
            raise RouteValidationError(
                "Travel time must be positive"
            )

        validate_probability(
            self.risk_score,
            "risk_score",
        )

        validate_probability(
            self.evidence_reliability,
            "evidence_reliability",
        )

        if self.max_vehicle_weight_tonnes is not None:
            validate_nonnegative(
                self.max_vehicle_weight_tonnes,
                "max_vehicle_weight_tonnes",
            )

        if self.verified_at is not None:
            validate_datetime(
                self.verified_at
            )


@dataclass(frozen=True)
class EmergencyVehicleProfile:
    vehicle_id: str
    vehicle_type: VehicleType

    gross_weight_tonnes: float

    def __post_init__(self) -> None:
        if not self.vehicle_id.strip():
            raise RouteValidationError(
                "vehicle_id required"
            )

        validate_nonnegative(
            self.gross_weight_tonnes,
            "gross_weight_tonnes",
        )


@dataclass(frozen=True)
class RoutePlanningRequest:
    request_id: str

    district: str

    mission_type: EmergencyMissionType

    origin_node_id: str
    destination_node_id: str

    vehicle: EmergencyVehicleProfile

    nodes: tuple[RoadNode, ...]
    edges: tuple[RoadEdge, ...]

    created_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "district",
            "origin_node_id",
            "destination_node_id",
        ):
            if not getattr(self, name).strip():
                raise RouteValidationError(
                    f"{name} required"
                )

        validate_datetime(
            self.created_at
        )

        if (
            self.origin_node_id
            == self.destination_node_id
        ):
            raise RouteValidationError(
                "Origin and destination must differ"
            )


@dataclass(frozen=True)
class RoadScreeningResult:
    eligible_edge_ids: tuple[str, ...]
    rejected_edge_ids: tuple[str, ...]

    rejection_reasons: Mapping[str, str]

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PlannedRoute:
    route_id: str

    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]

    distance_km: float
    estimated_travel_minutes: float

    cumulative_risk_score: float

    risk_level: RouteRiskLevel

    evidence_source_ids: tuple[str, ...]

    advisory_only: bool = True


@dataclass(frozen=True)
class RoutePlanningResult:
    plan_id: str
    request_id: str

    district: str

    mission_type: EmergencyMissionType
    vehicle_id: str

    origin_node_id: str
    destination_node_id: str

    recommended_route: PlannedRoute | None

    alternative_routes: tuple[PlannedRoute, ...]

    road_screening: RoadScreeningResult

    warnings: tuple[str, ...]

    status: RoutePlanStatus

    human_approval_required: bool

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RouteApprovalRecord:
    approval_id: str
    plan_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ApprovalDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class RouteAuditRecord:
    audit_id: str
    plan_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# ROAD SAFETY SCREENING ENGINE
# ============================================================

class RoadSafetyScreeningEngine:
    def __init__(
        self,
        config: RoutePlanningConfig,
    ) -> None:
        self.config = config

    def screen(
        self,
        edges: Sequence[RoadEdge],
        vehicle: EmergencyVehicleProfile,
        *,
        now: datetime,
    ) -> tuple[
        tuple[RoadEdge, ...],
        RoadScreeningResult,
    ]:
        eligible = []
        rejected = {}
        warnings = []

        for edge in edges:
            reason = None

            if edge.road_status != RoadStatus.VERIFIED_OPEN:
                reason = "Road not verified open"

            elif not edge.evidence_source_id.strip():
                reason = "Missing evidence source"

            elif edge.verified_at is None:
                reason = "Missing verification timestamp"

            elif edge.verified_at > now:
                reason = "Future verification timestamp"

            elif (
                now - edge.verified_at
            ) > timedelta(
                minutes=(
                    self.config.max_road_observation_age_minutes
                )
            ):
                reason = "Stale road observation"

            elif edge.evidence_reliability < (
                self.config.minimum_evidence_reliability
            ):
                reason = "Insufficient evidence reliability"

            elif edge.flood_depth_m > (
                self.config.maximum_screened_flood_depth_m
            ):
                reason = "Flood depth exceeds screening limit"

            elif (
                edge.bridge_present
                and not edge.bridge_verified_safe
            ):
                reason = "Bridge safety not verified"

            elif (
                edge.max_vehicle_weight_tonnes is not None
                and vehicle.gross_weight_tonnes
                > edge.max_vehicle_weight_tonnes
            ):
                reason = "Vehicle exceeds road weight limit"

            elif (
                edge.allowed_vehicle_types
                and vehicle.vehicle_type
                not in edge.allowed_vehicle_types
            ):
                reason = "Vehicle type not permitted"

            if reason is not None:
                rejected[
                    edge.edge_id
                ] = reason
                continue

            eligible.append(edge)

        if rejected:
            warnings.append(
                f"{len(rejected)} road edges excluded"
            )

        warnings.append(
            "Road eligibility does not establish "
            "actual operational safety"
        )

        screening = RoadScreeningResult(
            eligible_edge_ids=tuple(
                edge.edge_id
                for edge in eligible
            ),
            rejected_edge_ids=tuple(
                rejected
            ),
            rejection_reasons=rejected,
            warnings=tuple(warnings),
        )

        return tuple(eligible), screening


# ============================================================
# ROAD GRAPH
# ============================================================

class RoadGraph:
    def __init__(
        self,
        nodes: Sequence[RoadNode],
        edges: Sequence[RoadEdge],
    ) -> None:
        self.nodes = {
            node.node_id: node
            for node in nodes
        }

        self.edges = {
            edge.edge_id: edge
            for edge in edges
        }

        self.adjacency: dict[
            str,
            list[tuple[str, RoadEdge]],
        ] = {
            node_id: []
            for node_id in self.nodes
        }

        for edge in edges:
            if (
                edge.source_node_id not in self.nodes
                or edge.target_node_id not in self.nodes
            ):
                raise RouteValidationError(
                    "Road edge references unknown node"
                )

            self.adjacency[
                edge.source_node_id
            ].append(
                (
                    edge.target_node_id,
                    edge,
                )
            )

            if edge.bidirectional:
                self.adjacency[
                    edge.target_node_id
                ].append(
                    (
                        edge.source_node_id,
                        edge,
                    )
                )


# ============================================================
# DIJKSTRA ROUTE PLANNER
# ============================================================

class DijkstraRoutePlanner:
    def __init__(
        self,
        config: RoutePlanningConfig,
    ) -> None:
        self.config = config

    def edge_cost(
        self,
        edge: RoadEdge,
    ) -> float:
        return (
            edge.estimated_travel_minutes
            * (
                1
                + self.config.risk_weight
                * edge.risk_score
            )
        )

    def find_route(
        self,
        graph: RoadGraph,
        origin: str,
        destination: str,
        *,
        excluded_edges: set[str] | None = None,
    ) -> PlannedRoute | None:
        excluded_edges = (
            excluded_edges or set()
        )

        queue = [
            (
                0.0,
                origin,
            )
        ]

        distances = {
            origin: 0.0
        }

        previous: dict[
            str,
            tuple[str, str],
        ] = {}

        while queue:
            cost, node = heapq.heappop(
                queue
            )

            if cost > distances.get(
                node,
                math.inf,
            ):
                continue

            if node == destination:
                break

            for neighbor, edge in (
                graph.adjacency.get(
                    node,
                    [],
                )
            ):
                if edge.edge_id in excluded_edges:
                    continue

                new_cost = (
                    cost
                    + self.edge_cost(edge)
                )

                if new_cost < distances.get(
                    neighbor,
                    math.inf,
                ):
                    distances[
                        neighbor
                    ] = new_cost

                    previous[
                        neighbor
                    ] = (
                        node,
                        edge.edge_id,
                    )

                    heapq.heappush(
                        queue,
                        (
                            new_cost,
                            neighbor,
                        ),
                    )

        if destination not in distances:
            return None

        node_path = [
            destination
        ]

        edge_path = []

        current = destination

        while current != origin:
            if current not in previous:
                return None

            previous_node, edge_id = (
                previous[current]
            )

            edge_path.append(
                edge_id
            )

            node_path.append(
                previous_node
            )

            current = previous_node

        node_path.reverse()
        edge_path.reverse()

        selected_edges = [
            graph.edges[edge_id]
            for edge_id in edge_path
        ]

        total_distance = sum(
            edge.distance_km
            for edge in selected_edges
        )

        total_duration = sum(
            edge.estimated_travel_minutes
            for edge in selected_edges
        )

        # This is a cumulative heuristic, not
        # a calibrated probability of failure.
        cumulative_risk = (
            1.0
            - math.prod(
                1.0 - edge.risk_score
                for edge in selected_edges
            )
        )

        if cumulative_risk >= 0.75:
            risk_level = RouteRiskLevel.HIGH

        elif cumulative_risk >= 0.40:
            risk_level = RouteRiskLevel.MODERATE

        else:
            risk_level = RouteRiskLevel.LOW

        return PlannedRoute(
            route_id=generate_id(),
            node_ids=tuple(node_path),
            edge_ids=tuple(edge_path),
            distance_km=round(
                total_distance,
                3,
            ),
            estimated_travel_minutes=round(
                total_duration,
                2,
            ),
            cumulative_risk_score=round(
                cumulative_risk,
                4,
            ),
            risk_level=risk_level,
            evidence_source_ids=tuple(
                sorted({
                    edge.evidence_source_id
                    for edge in selected_edges
                })
            ),
        )


# ============================================================
# ALTERNATIVE ROUTE ENGINE
# ============================================================

class AlternativeRouteEngine:
    """
    Generates alternative candidate paths by
    excluding individual edges from the primary
    route.

    This is a simple alternative generator,
    not a full Yen's K-shortest-path implementation.
    """

    def __init__(
        self,
        planner: DijkstraRoutePlanner,
        config: RoutePlanningConfig,
    ) -> None:
        self.planner = planner
        self.config = config

    def generate(
        self,
        graph: RoadGraph,
        origin: str,
        destination: str,
        primary: PlannedRoute,
    ) -> tuple[PlannedRoute, ...]:
        candidates = {}

        for edge_id in primary.edge_ids:
            alternative = (
                self.planner.find_route(
                    graph,
                    origin,
                    destination,
                    excluded_edges={
                        edge_id
                    },
                )
            )

            if alternative is None:
                continue

            if alternative.edge_ids == (
                primary.edge_ids
            ):
                continue

            candidates[
                alternative.edge_ids
            ] = alternative

        alternatives = sorted(
            candidates.values(),
            key=lambda route: (
                route.estimated_travel_minutes
                * (
                    1
                    + self.config.risk_weight
                    * route.cumulative_risk_score
                ),
                route.edge_ids,
            ),
        )

        return tuple(
            alternatives[
                :self.config.max_alternative_routes
            ]
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class RoutePlanningAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            RouteAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        plan_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> RouteAuditRecord:
        record = RouteAuditRecord(
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
    ) -> tuple[RouteAuditRecord, ...]:
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
# HUMAN APPROVAL
# ============================================================

class RouteHumanApprovalManager:
    ALLOWED_ROLES = {
        "incident_commander",
        "district_disaster_management_officer",
        "authorized_route_coordinator",
        "transport_safety_officer",
    }

    def __init__(
        self,
        audit_logger: RoutePlanningAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            RouteApprovalRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        plan: RoutePlanningResult,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ApprovalDecision,
        reason: str,
    ) -> RouteApprovalRecord:
        if not reviewer_id.strip():
            raise RouteApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise RouteApprovalError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise RouteApprovalError(
                "Review reason required"
            )

        if (
            decision == ApprovalDecision.APPROVE
            and plan.recommended_route is None
        ):
            raise RouteApprovalError(
                "Cannot approve plan without route"
            )

        with self._lock:
            if plan.plan_id in self._reviews:
                raise RouteApprovalError(
                    "Plan already reviewed"
                )

            record = RouteApprovalRecord(
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
            action=f"route_review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return record


# ============================================================
# MAIN ROUTE PLANNING AGENT
# ============================================================

class RoutePlanningAgent:
    def __init__(
        self,
        config: RoutePlanningConfig | None = None,
    ) -> None:
        self.config = (
            config or RoutePlanningConfig()
        )

        self.safety_engine = (
            RoadSafetyScreeningEngine(
                self.config
            )
        )

        self.route_planner = (
            DijkstraRoutePlanner(
                self.config
            )
        )

        self.alternative_engine = (
            AlternativeRouteEngine(
                self.route_planner,
                self.config,
            )
        )

        self.audit_logger = (
            RoutePlanningAuditLogger()
        )

        self.approval_manager = (
            RouteHumanApprovalManager(
                self.audit_logger
            )
        )

        self._plans: dict[
            str,
            RoutePlanningResult,
        ] = {}

        self._lock = threading.RLock()

    def run(
        self,
        request: RoutePlanningRequest,
        *,
        now: datetime | None = None,
    ) -> RoutePlanningResult:
        now = now or utc_now()

        validate_datetime(now)

        if request.created_at > now:
            raise RouteValidationError(
                "Request timestamp is in the future"
            )

        if len(request.nodes) > (
            self.config.max_nodes
        ):
            raise RouteValidationError(
                "Node limit exceeded"
            )

        if len(request.edges) > (
            self.config.max_edges
        ):
            raise RouteValidationError(
                "Edge limit exceeded"
            )

        node_ids = [
            node.node_id
            for node in request.nodes
        ]

        edge_ids = [
            edge.edge_id
            for edge in request.edges
        ]

        if len(node_ids) != len(set(node_ids)):
            raise RouteValidationError(
                "Duplicate node ID"
            )

        if len(edge_ids) != len(set(edge_ids)):
            raise RouteValidationError(
                "Duplicate edge ID"
            )

        node_set = set(
            node_ids
        )

        if request.origin_node_id not in node_set:
            raise RouteValidationError(
                "Origin node not found"
            )

        if request.destination_node_id not in node_set:
            raise RouteValidationError(
                "Destination node not found"
            )

        # Validate the complete graph before
        # removing ineligible road segments.
        RoadGraph(
            request.nodes,
            request.edges,
        )

        eligible_edges, screening = (
            self.safety_engine.screen(
                request.edges,
                request.vehicle,
                now=now,
            )
        )

        graph = RoadGraph(
            request.nodes,
            eligible_edges,
        )

        primary = (
            self.route_planner.find_route(
                graph,
                request.origin_node_id,
                request.destination_node_id,
            )
        )

        alternatives = ()

        warnings = list(
            screening.warnings
        )

        if primary is not None:
            alternatives = (
                self.alternative_engine.generate(
                    graph,
                    request.origin_node_id,
                    request.destination_node_id,
                    primary,
                )
            )

        else:
            warnings.append(
                "No route satisfies the configured "
                "road-evidence screening criteria"
            )

            warnings.append(
                "Escalate for human route and mission review"
            )

        warnings.extend([
            "Routes are advisory only",
            "A verified-open record does not guarantee safety",
            "Floodwater crossing is not authorized",
            "Estimated travel times may be inaccurate",
            "Road conditions may change rapidly",
            "Field verification is required before movement",
            "Human approval does not constitute dispatch",
        ])

        result = RoutePlanningResult(
            plan_id=generate_id(),
            request_id=request.request_id,
            district=request.district,
            mission_type=request.mission_type,
            vehicle_id=request.vehicle.vehicle_id,
            origin_node_id=request.origin_node_id,
            destination_node_id=request.destination_node_id,
            recommended_route=primary,
            alternative_routes=alternatives,
            road_screening=screening,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            status=RoutePlanStatus.PENDING_APPROVAL,
            human_approval_required=True,
            created_at=now,
        )

        with self._lock:
            self._plans[
                result.plan_id
            ] = result

        self.audit_logger.record(
            plan_id=result.plan_id,
            actor_id="route_planning_agent",
            action="route_plan_created",
            details={
                "district": request.district,
                "mission_type": request.mission_type.value,
                "route_found": str(
                    primary is not None
                ),
                "alternative_count": str(
                    len(alternatives)
                ),
            },
        )

        return result

    def review_plan(
        self,
        plan_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ApprovalDecision,
        reason: str,
    ) -> RoutePlanningResult:
        with self._lock:
            plan = self._plans.get(
                plan_id
            )

            if plan is None:
                raise RouteApprovalError(
                    "Route plan not found"
                )

            if plan.status != (
                RoutePlanStatus.PENDING_APPROVAL
            ):
                raise RouteApprovalError(
                    "Route plan is not pending review"
                )

            self.approval_manager.review(
                plan,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            if decision == ApprovalDecision.APPROVE:
                status = (
                    RoutePlanStatus.APPROVED_FOR_COORDINATION
                )

            elif decision == ApprovalDecision.REJECT:
                status = (
                    RoutePlanStatus.REJECTED
                )

            else:
                status = (
                    RoutePlanStatus.REVISION_REQUIRED
                )

            updated = replace(
                plan,
                status=status,
            )

            self._plans[
                plan_id
            ] = updated

            return updated

    def get_plan(
        self,
        plan_id: str,
    ) -> RoutePlanningResult | None:
        with self._lock:
            return self._plans.get(
                plan_id
            )

    def get_audit_history(
        self,
        plan_id: str,
    ) -> tuple[RouteAuditRecord, ...]:
        return self.audit_logger.list_records(
            plan_id
        )


# ============================================================
# JSON EXPORTER
# ============================================================

class RoutePlanningReportExporter:
    @staticmethod
    def export_json(
        result: RoutePlanningResult,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise RouteExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                result.to_dict(),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_route_planning_agent(
    config: RoutePlanningConfig | None = None,
) -> RoutePlanningAgent:
    return RoutePlanningAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional road network.
    # Coordinates, conditions and times are
    # illustrative and not live navigation data.

    nodes = (
        RoadNode(
            node_id="A",
            name="Demo Rescue Base",
        ),
        RoadNode(
            node_id="B",
            name="Demo Junction",
        ),
        RoadNode(
            node_id="C",
            name="Demo Bridge",
        ),
        RoadNode(
            node_id="D",
            name="Demo Relief Shelter",
        ),
    )

    edges = (
        RoadEdge(
            edge_id="ROAD-AB",
            source_node_id="A",
            target_node_id="B",
            distance_km=3.0,
            estimated_travel_minutes=8,
            road_status=RoadStatus.VERIFIED_OPEN,
            road_surface=RoadSurface.PAVED,
            flood_depth_m=0.0,
            risk_score=0.10,
            evidence_source_id="DEMO-ROAD-OFFICER-1",
            evidence_reliability=0.95,
            verified_at=now - timedelta(minutes=5),
        ),
        RoadEdge(
            edge_id="ROAD-BD",
            source_node_id="B",
            target_node_id="D",
            distance_km=4.0,
            estimated_travel_minutes=10,
            road_status=RoadStatus.VERIFIED_OPEN,
            road_surface=RoadSurface.PAVED,
            flood_depth_m=0.0,
            risk_score=0.15,
            evidence_source_id="DEMO-ROAD-OFFICER-2",
            evidence_reliability=0.92,
            verified_at=now - timedelta(minutes=5),
        ),
        RoadEdge(
            edge_id="ROAD-AC",
            source_node_id="A",
            target_node_id="C",
            distance_km=2.0,
            estimated_travel_minutes=6,
            road_status=RoadStatus.VERIFIED_OPEN,
            road_surface=RoadSurface.PAVED,
            flood_depth_m=0.0,
            risk_score=0.08,
            evidence_source_id="DEMO-ROAD-OFFICER-3",
            evidence_reliability=0.94,
            verified_at=now - timedelta(minutes=5),
        ),
        RoadEdge(
            edge_id="ROAD-CD",
            source_node_id="C",
            target_node_id="D",
            distance_km=5.0,
            estimated_travel_minutes=12,
            road_status=RoadStatus.VERIFIED_OPEN,
            road_surface=RoadSurface.PAVED,
            flood_depth_m=0.0,
            risk_score=0.12,
            bridge_present=True,
            bridge_verified_safe=True,
            max_vehicle_weight_tonnes=10.0,
            evidence_source_id="DEMO-BRIDGE-INSPECTION",
            evidence_reliability=0.95,
            verified_at=now - timedelta(minutes=5),
        ),
    )

    vehicle = EmergencyVehicleProfile(
        vehicle_id="DEMO-AMBULANCE-001",
        vehicle_type=VehicleType.AMBULANCE,
        gross_weight_tonnes=3.5,
    )

    request = RoutePlanningRequest(
        request_id="DEMO-ROUTE-REQUEST-001",
        district="Thrissur",
        mission_type=EmergencyMissionType.AMBULANCE,
        origin_node_id="A",
        destination_node_id="D",
        vehicle=vehicle,
        nodes=nodes,
        edges=edges,
        created_at=now,
    )

    agent = create_route_planning_agent()

    result = agent.run(
        request,
        now=now,
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    if result.recommended_route:
        reviewed = agent.review_plan(
            result.plan_id,
            reviewer_id="DEMO-ROUTE-OFFICER",
            reviewer_role="transport_safety_officer",
            decision=ApprovalDecision.APPROVE,
            reason="Illustrative coordination review",
        )

        print(
            "Plan status:",
            reviewed.status.value,
        )

        RoutePlanningReportExporter.export_json(
            reviewed,
            "outputs/agents/route_planning_report.json",
        )

    print(
        "No vehicle movement or dispatch occurred."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RoutePlanningError",
    "RouteValidationError",
    "RouteNotFoundError",
    "RouteApprovalError",
    "RouteExportError",
    "EmergencyMissionType",
    "VehicleType",
    "RoadStatus",
    "RoadSurface",
    "RouteRiskLevel",
    "RoutePlanStatus",
    "ApprovalDecision",
    "RoutePlanningConfig",
    "RoadNode",
    "RoadEdge",
    "EmergencyVehicleProfile",
    "RoutePlanningRequest",
    "RoadScreeningResult",
    "PlannedRoute",
    "RoutePlanningResult",
    "RouteApprovalRecord",
    "RouteAuditRecord",
    "RoadSafetyScreeningEngine",
    "RoadGraph",
    "DijkstraRoutePlanner",
    "AlternativeRouteEngine",
    "RoutePlanningAuditLogger",
    "RouteHumanApprovalManager",
    "RoutePlanningAgent",
    "RoutePlanningReportExporter",
    "create_route_planning_agent",
]

"""
backend/ai/agents/sos_triage_agent.py

Kerala Flood Relief Coordination Agent
SOS Triage & Emergency Escalation Agent

Python 3.11+
Standard library only.

Features
--------
- Flood-related SOS intake
- SOS validation
- Emergency urgency flagging
- Vulnerability-aware prioritization
- Life-threat keyword detection
- Missing-person and medical escalation flags
- Duplicate-report screening
- Evidence verification and freshness checks
- Response capability recommendations
- Human review and escalation
- Audit trail
- Privacy-conscious JSON export

SAFETY
------
This is an advisory decision-support component.

It does NOT:
- Diagnose medical conditions
- Automatically dispatch emergency responders
- Automatically send emergency notifications
- Confirm a person is safe or rescued
- Automatically close an SOS case
- Replace emergency operators

Immediate danger must be surfaced for human
emergency escalation even when the report
is incomplete or unverified.

Scoring weights and thresholds are illustrative,
not clinically or operationally validated.
"""

from __future__ import annotations

import hashlib
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

class SOSTriageError(Exception):
    pass


class SOSValidationError(SOSTriageError):
    pass


class SOSPermissionError(SOSTriageError):
    pass


class SOSStateError(SOSTriageError):
    pass


class SOSNotFoundError(SOSTriageError):
    pass


class SOSExportError(SOSTriageError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SOSIncidentType(str, Enum):
    FLOOD_ENTRAPMENT = "flood_entrapment"
    ROOFTOP_ENTRAPMENT = "rooftop_entrapment"
    WATER_RESCUE = "water_rescue"
    MEDICAL_EMERGENCY = "medical_emergency"
    EVACUATION_ASSISTANCE = "evacuation_assistance"
    MISSING_PERSON = "missing_person"
    LANDSLIDE = "landslide"
    BUILDING_COLLAPSE = "building_collapse"
    FIRE_OR_ELECTRICAL = "fire_or_electrical"
    FOOD_WATER_SHORTAGE = "food_water_shortage"
    OTHER = "other"


class SOSUrgency(str, Enum):
    ROUTINE_REVIEW = "routine_review"
    ELEVATED = "elevated"
    HIGH = "high"
    IMMEDIATE_HUMAN_ESCALATION = (
        "immediate_human_escalation"
    )


class SOSStatus(str, Enum):
    RECEIVED = "received"
    TRIAGED = "triaged"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    REVIEWED_FOR_COORDINATION = "reviewed_for_coordination"
    ESCALATED_TO_HUMAN = "escalated_to_human"
    REQUIRES_MORE_INFORMATION = "requires_more_information"
    REJECTED_AS_INVALID = "rejected_as_invalid"


class EvidenceVerification(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class SOSSourceType(str, Enum):
    PHONE_CALL = "phone_call"
    MOBILE_APP = "mobile_app"
    WEB_PORTAL = "web_portal"
    SMS = "sms"
    FIELD_RESPONDER = "field_responder"
    SHELTER_OFFICER = "shelter_officer"
    AUTHORIZED_OFFICIAL = "authorized_official"


class ResponseCapability(str, Enum):
    RESCUE_COORDINATION = "rescue_coordination"
    BOAT_RESCUE = "boat_rescue"
    MEDICAL_RESPONSE = "medical_response"
    EVACUATION_SUPPORT = "evacuation_support"
    MISSING_PERSON_COORDINATION = (
        "missing_person_coordination"
    )
    INFRASTRUCTURE_SAFETY = "infrastructure_safety"
    FIRE_RESPONSE = "fire_response"
    RELIEF_LOGISTICS = "relief_logistics"


class ReviewDecision(str, Enum):
    CONFIRM_FOR_COORDINATION = "confirm_for_coordination"
    ESCALATE = "escalate"
    REQUEST_MORE_INFORMATION = "request_more_information"
    REJECT_INVALID = "reject_invalid"


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
        raise SOSValidationError(
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
        raise SOSValidationError(
            f"{name} must be nonnegative and finite"
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    validate_nonnegative(value, name)

    if value > 1:
        raise SOSValidationError(
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


def normalize_text(value: str) -> str:
    return " ".join(
        value.casefold().split()
    )


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SOSTriageConfig:
    max_description_length: int = 5000
    max_people_at_risk: int = 100000

    maximum_report_age_hours: int = 72
    maximum_evidence_age_minutes: int = 180

    duplicate_window_hours: int = 24

    minimum_evidence_reliability: float = 0.65

    elevated_threshold: float = 0.30
    high_threshold: float = 0.55
    immediate_threshold: float = 0.80

    require_human_review: bool = True
    allow_automatic_dispatch: bool = False
    allow_public_personal_data: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_description_length",
            "max_people_at_risk",
            "maximum_report_age_hours",
            "maximum_evidence_age_minutes",
            "duplicate_window_hours",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SOSValidationError(
                    f"{name} must be positive integer"
                )

        for name in (
            "minimum_evidence_reliability",
            "elevated_threshold",
            "high_threshold",
            "immediate_threshold",
        ):
            validate_probability(
                getattr(self, name),
                name,
            )

        if not (
            self.elevated_threshold
            <= self.high_threshold
            <= self.immediate_threshold
        ):
            raise SOSValidationError(
                "Urgency thresholds must be ascending"
            )

        if self.allow_automatic_dispatch:
            raise SOSValidationError(
                "Automatic dispatch is not supported"
            )

        if self.allow_public_personal_data:
            raise SOSValidationError(
                "Public personal-data disclosure is disabled"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class SOSReporter:
    reporter_id: str
    reporter_name: str | None = None
    phone_number: str | None = None

    def __post_init__(self) -> None:
        if not self.reporter_id.strip():
            raise SOSValidationError(
                "reporter_id required"
            )


@dataclass(frozen=True)
class SOSLocation:
    district: str
    locality: str
    description: str

    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self) -> None:
        if not self.district.strip():
            raise SOSValidationError(
                "district required"
            )

        if not self.locality.strip():
            raise SOSValidationError(
                "locality required"
            )

        if not self.description.strip():
            raise SOSValidationError(
                "location description required"
            )

        if self.latitude is not None:
            if (
                not math.isfinite(self.latitude)
                or not -90 <= self.latitude <= 90
            ):
                raise SOSValidationError(
                    "Invalid latitude"
                )

        if self.longitude is not None:
            if (
                not math.isfinite(self.longitude)
                or not -180 <= self.longitude <= 180
            ):
                raise SOSValidationError(
                    "Invalid longitude"
                )


@dataclass(frozen=True)
class SOSEvidence:
    evidence_id: str
    source_id: str

    verification: EvidenceVerification

    reliability: float
    observed_at: datetime

    description: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise SOSValidationError(
                "evidence_id required"
            )

        if not self.source_id.strip():
            raise SOSValidationError(
                "source_id required"
            )

        validate_probability(
            self.reliability,
            "reliability",
        )

        validate_datetime(
            self.observed_at
        )


@dataclass(frozen=True)
class SOSReport:
    sos_id: str

    reporter: SOSReporter
    source_type: SOSSourceType

    location: SOSLocation
    incident_type: SOSIncidentType

    description: str

    people_at_risk: int

    children_count: int = 0
    older_adults_count: int = 0
    medically_vulnerable_count: int = 0

    trapped: bool = False
    immediate_life_threat: bool = False
    serious_injury_reported: bool = False
    rising_water_reported: bool = False

    evidence: tuple[SOSEvidence, ...] = ()

    reported_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.sos_id.strip():
            raise SOSValidationError(
                "sos_id required"
            )

        if not self.description.strip():
            raise SOSValidationError(
                "SOS description required"
            )

        if (
            isinstance(self.people_at_risk, bool)
            or not isinstance(self.people_at_risk, int)
            or self.people_at_risk <= 0
        ):
            raise SOSValidationError(
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
                raise SOSValidationError(
                    f"Invalid {name}"
                )

        if self.reported_at is not None:
            validate_datetime(
                self.reported_at
            )


@dataclass(frozen=True)
class SOSEvidenceAssessment:
    accepted_evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]

    verified_source_count: int

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SOSUrgencyAssessment:
    sos_id: str

    urgency: SOSUrgency
    score: float

    factors: Mapping[str, float]

    immediate_danger_detected: bool

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SOSDuplicateCandidate:
    existing_sos_id: str
    new_sos_id: str

    similarity_score: float

    reason: str

    human_review_required: bool = True


@dataclass(frozen=True)
class SOSTriageResult:
    triage_id: str
    sos_id: str

    district: str
    locality: str

    urgency_assessment: SOSUrgencyAssessment
    evidence_assessment: SOSEvidenceAssessment

    required_capabilities: tuple[
        ResponseCapability,
        ...
    ]

    possible_duplicates: tuple[
        SOSDuplicateCandidate,
        ...
    ]

    recommended_actions: tuple[str, ...]

    warnings: tuple[str, ...]

    status: SOSStatus

    requires_human_review: bool
    requires_immediate_escalation: bool

    created_at: datetime

    def to_dict(
        self,
        *,
        redact_location: bool = True,
    ) -> dict[str, Any]:
        data = serialize(self)

        if redact_location:
            data["locality"] = "[REDACTED]"

        return data


@dataclass(frozen=True)
class SOSHumanReview:
    review_id: str
    triage_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ReviewDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class SOSAuditRecord:
    audit_id: str
    triage_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# REPORT VALIDATION ENGINE
# ============================================================

class SOSReportValidator:
    def __init__(
        self,
        config: SOSTriageConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        report: SOSReport,
        *,
        now: datetime,
    ) -> tuple[str, ...]:
        warnings = []

        if len(report.description) > (
            self.config.max_description_length
        ):
            raise SOSValidationError(
                "SOS description exceeds maximum length"
            )

        if report.people_at_risk > (
            self.config.max_people_at_risk
        ):
            raise SOSValidationError(
                "People-at-risk count exceeds configured limit"
            )

        if report.reported_at is None:
            warnings.append(
                "Report timestamp is missing"
            )

        else:
            if report.reported_at > now:
                raise SOSValidationError(
                    "SOS timestamp is in the future"
                )

            if (
                now - report.reported_at
            ) > timedelta(
                hours=(
                    self.config.maximum_report_age_hours
                )
            ):
                warnings.append(
                    "SOS report may be stale; "
                    "human re-verification required"
                )

        if not report.evidence:
            warnings.append(
                "No supporting evidence provided"
            )

        if (
            report.location.latitude is None
            or report.location.longitude is None
        ):
            warnings.append(
                "Exact coordinates unavailable; "
                "location confirmation required"
            )

        return tuple(warnings)


# ============================================================
# EVIDENCE ENGINE
# ============================================================

class SOSEvidenceValidator:
    def __init__(
        self,
        config: SOSTriageConfig,
    ) -> None:
        self.config = config

    def assess(
        self,
        evidence: Sequence[SOSEvidence],
        *,
        now: datetime,
    ) -> SOSEvidenceAssessment:
        accepted = []
        rejected = []
        warnings = []
        seen_ids = set()

        for item in evidence:
            reason = None

            if item.evidence_id in seen_ids:
                reason = "duplicate evidence ID"

            seen_ids.add(
                item.evidence_id
            )

            if reason is None and (
                item.verification
                != EvidenceVerification.VERIFIED
            ):
                reason = "evidence not verified"

            if reason is None and (
                item.reliability
                < self.config.minimum_evidence_reliability
            ):
                reason = "insufficient reliability"

            if reason is None and (
                item.observed_at > now
            ):
                reason = "future observation timestamp"

            if reason is None and (
                now - item.observed_at
            ) > timedelta(
                minutes=(
                    self.config.maximum_evidence_age_minutes
                )
            ):
                reason = "stale evidence"

            if reason is not None:
                rejected.append(
                    item.evidence_id
                )

                warnings.append(
                    f"{item.evidence_id}: {reason}"
                )

                continue

            accepted.append(item)

        source_count = len({
            item.source_id
            for item in accepted
        })

        if source_count == 0:
            warnings.append(
                "No verified fresh evidence available"
            )

        return SOSEvidenceAssessment(
            accepted_evidence_ids=tuple(
                item.evidence_id
                for item in accepted
            ),
            rejected_evidence_ids=tuple(
                rejected
            ),
            verified_source_count=source_count,
            warnings=tuple(warnings),
        )


# ============================================================
# IMMEDIATE DANGER DETECTION
# ============================================================

class SOSDangerSignalEngine:
    """
    Simple text signal detector.

    Keyword matches are advisory only.
    They are not reliable language understanding
    or medical diagnosis.

    Explicit structured danger flags are always
    considered independently of text matches.
    """

    DANGER_KEYWORDS = (
        "drowning",
        "cannot breathe",
        "can't breathe",
        "unconscious",
        "trapped",
        "roof",
        "rising water",
        "swept away",
        "severe bleeding",
        "electric shock",
        "building collapse",
        "help immediately",
        "മുങ്ങുന്നു",
        "കുടുങ്ങി",
        "വെള്ളം ഉയരുന്നു",
        "ശ്വാസം കിട്ടുന്നില്ല",
    )

    @classmethod
    def detect(
        cls,
        report: SOSReport,
    ) -> tuple[
        bool,
        tuple[str, ...],
    ]:
        text = normalize_text(
            report.description
        )

        matched = tuple(
            keyword
            for keyword in cls.DANGER_KEYWORDS
            if keyword in text
        )

        structured_danger = any([
            report.immediate_life_threat,
            report.serious_injury_reported,
            report.trapped,
        ])

        return (
            structured_danger or bool(matched),
            matched,
        )


# ============================================================
# URGENCY ENGINE
# ============================================================

class SOSUrgencyEngine:
    INCIDENT_RISK = {
        SOSIncidentType.FLOOD_ENTRAPMENT: 0.90,
        SOSIncidentType.ROOFTOP_ENTRAPMENT: 0.95,
        SOSIncidentType.WATER_RESCUE: 1.00,
        SOSIncidentType.MEDICAL_EMERGENCY: 0.95,
        SOSIncidentType.EVACUATION_ASSISTANCE: 0.60,
        SOSIncidentType.MISSING_PERSON: 0.75,
        SOSIncidentType.LANDSLIDE: 0.90,
        SOSIncidentType.BUILDING_COLLAPSE: 1.00,
        SOSIncidentType.FIRE_OR_ELECTRICAL: 0.95,
        SOSIncidentType.FOOD_WATER_SHORTAGE: 0.55,
        SOSIncidentType.OTHER: 0.40,
    }

    def __init__(
        self,
        config: SOSTriageConfig,
    ) -> None:
        self.config = config

    def assess(
        self,
        report: SOSReport,
    ) -> SOSUrgencyAssessment:
        immediate_danger, keywords = (
            SOSDangerSignalEngine.detect(
                report
            )
        )

        incident_risk = self.INCIDENT_RISK[
            report.incident_type
        ]

        population = min(
            report.people_at_risk / 100,
            1.0,
        )

        vulnerable = min(
            report.people_at_risk,
            report.children_count
            + report.older_adults_count
            + report.medically_vulnerable_count,
        )

        vulnerability = (
            vulnerable
            / report.people_at_risk
        )

        rising_water = float(
            report.rising_water_reported
        )

        danger_factor = float(
            immediate_danger
        )

        score = (
            0.25 * incident_risk
            + 0.10 * population
            + 0.15 * vulnerability
            + 0.10 * rising_water
            + 0.40 * danger_factor
        )

        score = round(
            min(score, 1.0),
            4,
        )

        if (
            immediate_danger
            or score >= self.config.immediate_threshold
        ):
            urgency = (
                SOSUrgency.IMMEDIATE_HUMAN_ESCALATION
            )

        elif score >= self.config.high_threshold:
            urgency = SOSUrgency.HIGH

        elif score >= self.config.elevated_threshold:
            urgency = SOSUrgency.ELEVATED

        else:
            urgency = SOSUrgency.ROUTINE_REVIEW

        warnings = [
            "Urgency is an advisory heuristic",
            "A low score does not establish safety",
        ]

        if keywords:
            warnings.append(
                "Potential danger keywords detected; "
                "human confirmation required"
            )

        return SOSUrgencyAssessment(
            sos_id=report.sos_id,
            urgency=urgency,
            score=score,
            factors={
                "incident_risk": incident_risk,
                "population": population,
                "vulnerability": vulnerability,
                "rising_water": rising_water,
                "immediate_danger": danger_factor,
            },
            immediate_danger_detected=(
                immediate_danger
            ),
            warnings=tuple(warnings),
        )


# ============================================================
# DUPLICATE SCREENING ENGINE
# ============================================================

class SOSDuplicateScreeningEngine:
    """
    Screens reports for possible duplication.

    Never automatically merges or suppresses
    SOS incidents.

    A duplicate report can contain new
    life-threatening information.
    """

    def __init__(
        self,
        config: SOSTriageConfig,
    ) -> None:
        self.config = config

    def screen(
        self,
        new_report: SOSReport,
        existing_reports: Sequence[SOSReport],
        *,
        now: datetime,
    ) -> tuple[SOSDuplicateCandidate, ...]:
        candidates = []

        for existing in existing_reports:
            if existing.sos_id == new_report.sos_id:
                continue

            if (
                existing.location.district.casefold()
                != new_report.location.district.casefold()
            ):
                continue

            if (
                existing.reported_at is not None
                and (
                    existing.reported_at > now
                    or now - existing.reported_at
                    > timedelta(
                        hours=(
                            self.config.duplicate_window_hours
                        )
                    )
                )
            ):
                continue

            score = 0.0
            reasons = []

            if (
                normalize_text(
                    existing.location.locality
                )
                == normalize_text(
                    new_report.location.locality
                )
            ):
                score += 0.30
                reasons.append(
                    "same locality"
                )

            if (
                existing.incident_type
                == new_report.incident_type
            ):
                score += 0.25
                reasons.append(
                    "same incident type"
                )

            if (
                existing.reporter.reporter_id
                == new_report.reporter.reporter_id
            ):
                score += 0.25
                reasons.append(
                    "same reporter"
                )

            if (
                normalize_text(
                    existing.description
                )
                == normalize_text(
                    new_report.description
                )
            ):
                score += 0.20
                reasons.append(
                    "same description"
                )

            if score >= 0.65:
                candidates.append(
                    SOSDuplicateCandidate(
                        existing_sos_id=existing.sos_id,
                        new_sos_id=new_report.sos_id,
                        similarity_score=round(
                            score,
                            4,
                        ),
                        reason=", ".join(reasons),
                    )
                )

        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.similarity_score,
                    item.existing_sos_id,
                ),
            )
        )


# ============================================================
# RESPONSE CAPABILITY ENGINE
# ============================================================

class SOSResponseCapabilityEngine:
    CAPABILITY_MAP = {
        SOSIncidentType.FLOOD_ENTRAPMENT: (
            ResponseCapability.RESCUE_COORDINATION,
            ResponseCapability.BOAT_RESCUE,
        ),
        SOSIncidentType.ROOFTOP_ENTRAPMENT: (
            ResponseCapability.RESCUE_COORDINATION,
        ),
        SOSIncidentType.WATER_RESCUE: (
            ResponseCapability.RESCUE_COORDINATION,
            ResponseCapability.BOAT_RESCUE,
        ),
        SOSIncidentType.MEDICAL_EMERGENCY: (
            ResponseCapability.MEDICAL_RESPONSE,
        ),
        SOSIncidentType.EVACUATION_ASSISTANCE: (
            ResponseCapability.EVACUATION_SUPPORT,
        ),
        SOSIncidentType.MISSING_PERSON: (
            ResponseCapability.MISSING_PERSON_COORDINATION,
        ),
        SOSIncidentType.LANDSLIDE: (
            ResponseCapability.RESCUE_COORDINATION,
            ResponseCapability.INFRASTRUCTURE_SAFETY,
        ),
        SOSIncidentType.BUILDING_COLLAPSE: (
            ResponseCapability.RESCUE_COORDINATION,
            ResponseCapability.INFRASTRUCTURE_SAFETY,
        ),
        SOSIncidentType.FIRE_OR_ELECTRICAL: (
            ResponseCapability.FIRE_RESPONSE,
        ),
        SOSIncidentType.FOOD_WATER_SHORTAGE: (
            ResponseCapability.RELIEF_LOGISTICS,
        ),
        SOSIncidentType.OTHER: (
            ResponseCapability.RESCUE_COORDINATION,
        ),
    }

    @classmethod
    def recommend(
        cls,
        report: SOSReport,
    ) -> tuple[ResponseCapability, ...]:
        capabilities = list(
            cls.CAPABILITY_MAP[
                report.incident_type
            ]
        )

        if (
            report.serious_injury_reported
            or report.medically_vulnerable_count > 0
        ):
            capabilities.append(
                ResponseCapability.MEDICAL_RESPONSE
            )

        if report.trapped:
            capabilities.append(
                ResponseCapability.RESCUE_COORDINATION
            )

        return tuple(
            dict.fromkeys(capabilities)
        )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class SOSRecommendationEngine:
    @staticmethod
    def generate(
        urgency: SOSUrgencyAssessment,
        evidence: SOSEvidenceAssessment,
        duplicates: Sequence[SOSDuplicateCandidate],
    ) -> tuple[str, ...]:
        actions = [
            "Present SOS report to an authorized operator",
            "Verify caller and location details where feasible",
            "Maintain an auditable case record",
        ]

        if urgency.urgency == (
            SOSUrgency.IMMEDIATE_HUMAN_ESCALATION
        ):
            actions.insert(
                0,
                "Immediately surface SOS to the "
                "authorized emergency coordination desk"
            )

            actions.append(
                "Request urgent human assessment "
                "of rescue and medical response needs"
            )

        elif urgency.urgency == SOSUrgency.HIGH:
            actions.append(
                "Prioritize prompt incident-commander review"
            )

        if evidence.verified_source_count == 0:
            actions.append(
                "Seek additional evidence without delaying "
                "human escalation of possible life threats"
            )

        if duplicates:
            actions.append(
                "Review possible duplicate reports manually; "
                "do not suppress the new SOS"
            )

        actions.append(
            "Do not dispatch resources automatically"
        )

        return tuple(
            dict.fromkeys(actions)
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class SOSTriageAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            SOSAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        triage_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> SOSAuditRecord:
        record = SOSAuditRecord(
            audit_id=generate_id(),
            triage_id=triage_id,
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
        triage_id: str | None = None,
    ) -> tuple[SOSAuditRecord, ...]:
        with self._lock:
            records = tuple(
                self._records
            )

        if triage_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.triage_id == triage_id
        )


# ============================================================
# HUMAN REVIEW MANAGER
# ============================================================

class SOSHumanReviewManager:
    ALLOWED_ROLES = {
        "emergency_operator",
        "incident_commander",
        "authorized_rescue_coordinator",
        "district_disaster_management_officer",
    }

    def __init__(
        self,
        audit_logger: SOSTriageAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            SOSHumanReview,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        triage: SOSTriageResult,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> SOSHumanReview:
        if not reviewer_id.strip():
            raise SOSPermissionError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise SOSPermissionError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise SOSValidationError(
                "Review reason required"
            )

        if (
            triage.requires_immediate_escalation
            and decision == ReviewDecision.REJECT_INVALID
        ):
            raise SOSStateError(
                "Immediate-danger SOS cannot be rejected "
                "through this review workflow"
            )

        with self._lock:
            if triage.triage_id in self._reviews:
                raise SOSStateError(
                    "Triage already reviewed"
                )

            review = SOSHumanReview(
                review_id=generate_id(),
                triage_id=triage.triage_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                triage.triage_id
            ] = review

        self.audit_logger.record(
            triage_id=triage.triage_id,
            actor_id=reviewer_id,
            action=f"human_review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN SOS TRIAGE AGENT
# ============================================================

class SOSTriageAgent:
    def __init__(
        self,
        config: SOSTriageConfig | None = None,
    ) -> None:
        self.config = (
            config or SOSTriageConfig()
        )

        self.report_validator = (
            SOSReportValidator(
                self.config
            )
        )

        self.evidence_validator = (
            SOSEvidenceValidator(
                self.config
            )
        )

        self.urgency_engine = (
            SOSUrgencyEngine(
                self.config
            )
        )

        self.duplicate_engine = (
            SOSDuplicateScreeningEngine(
                self.config
            )
        )

        self.capability_engine = (
            SOSResponseCapabilityEngine()
        )

        self.recommendation_engine = (
            SOSRecommendationEngine()
        )

        self.audit_logger = (
            SOSTriageAuditLogger()
        )

        self.review_manager = (
            SOSHumanReviewManager(
                self.audit_logger
            )
        )

        self._reports: dict[
            str,
            SOSReport,
        ] = {}

        self._triages: dict[
            str,
            SOSTriageResult,
        ] = {}

        self._lock = threading.RLock()

    def triage(
        self,
        report: SOSReport,
        *,
        now: datetime | None = None,
    ) -> SOSTriageResult:
        now = now or utc_now()

        validate_datetime(now)

        validation_warnings = (
            self.report_validator.validate(
                report,
                now=now,
            )
        )

        evidence = (
            self.evidence_validator.assess(
                report.evidence,
                now=now,
            )
        )

        urgency = (
            self.urgency_engine.assess(
                report
            )
        )

        with self._lock:
            if report.sos_id in self._reports:
                raise SOSStateError(
                    "SOS ID already registered"
                )

            duplicates = (
                self.duplicate_engine.screen(
                    report,
                    tuple(self._reports.values()),
                    now=now,
                )
            )

            capabilities = (
                self.capability_engine.recommend(
                    report
                )
            )

            actions = (
                self.recommendation_engine.generate(
                    urgency,
                    evidence,
                    duplicates,
                )
            )

            warnings = list(
                validation_warnings
            )

            warnings.extend(
                evidence.warnings
            )

            warnings.extend(
                urgency.warnings
            )

            if duplicates:
                warnings.append(
                    "Possible duplicate SOS reports detected"
                )

            if urgency.immediate_danger_detected:
                warnings.append(
                    "Possible immediate danger: "
                    "urgent human attention required"
                )

            warnings.extend([
                "No emergency dispatch has occurred",
                "No automated notification has been sent",
                "Human review is required",
                "SOS data contains potentially sensitive information",
            ])

            result = SOSTriageResult(
                triage_id=generate_id(),
                sos_id=report.sos_id,
                district=report.location.district,
                locality=report.location.locality,
                urgency_assessment=urgency,
                evidence_assessment=evidence,
                required_capabilities=capabilities,
                possible_duplicates=duplicates,
                recommended_actions=actions,
                warnings=tuple(
                    dict.fromkeys(warnings)
                ),
                status=SOSStatus.AWAITING_HUMAN_REVIEW,
                requires_human_review=True,
                requires_immediate_escalation=(
                    urgency.urgency
                    == SOSUrgency.IMMEDIATE_HUMAN_ESCALATION
                ),
                created_at=now,
            )

            self._reports[
                report.sos_id
            ] = report

            self._triages[
                result.triage_id
            ] = result

            self.audit_logger.record(
                triage_id=result.triage_id,
                actor_id="sos_triage_agent",
                action="sos_triaged",
                details={
                    "sos_id": report.sos_id,
                    "urgency": urgency.urgency.value,
                    "immediate_escalation": str(
                        result.requires_immediate_escalation
                    ),
                    "duplicate_candidates": str(
                        len(duplicates)
                    ),
                },
            )

            return result

    def review_triage(
        self,
        triage_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> SOSTriageResult:
        with self._lock:
            triage = self._triages.get(
                triage_id
            )

            if triage is None:
                raise SOSNotFoundError(
                    "Triage result not found"
                )

            if triage.status != (
                SOSStatus.AWAITING_HUMAN_REVIEW
            ):
                raise SOSStateError(
                    "Triage is not pending review"
                )

            self.review_manager.review(
                triage,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            status_map = {
                ReviewDecision.CONFIRM_FOR_COORDINATION: (
                    SOSStatus.REVIEWED_FOR_COORDINATION
                ),
                ReviewDecision.ESCALATE: (
                    SOSStatus.ESCALATED_TO_HUMAN
                ),
                ReviewDecision.REQUEST_MORE_INFORMATION: (
                    SOSStatus.REQUIRES_MORE_INFORMATION
                ),
                ReviewDecision.REJECT_INVALID: (
                    SOSStatus.REJECTED_AS_INVALID
                ),
            }

            updated = replace(
                triage,
                status=status_map[
                    decision
                ],
            )

            self._triages[
                triage_id
            ] = updated

            return updated

    def get_triage(
        self,
        triage_id: str,
    ) -> SOSTriageResult | None:
        with self._lock:
            return self._triages.get(
                triage_id
            )

    def get_pending_immediate_cases(
        self,
    ) -> tuple[SOSTriageResult, ...]:
        with self._lock:
            return tuple(
                result
                for result in self._triages.values()
                if (
                    result.requires_immediate_escalation
                    and result.status
                    == SOSStatus.AWAITING_HUMAN_REVIEW
                )
            )

    def get_audit_history(
        self,
        triage_id: str,
    ) -> tuple[SOSAuditRecord, ...]:
        return self.audit_logger.list_records(
            triage_id
        )


# ============================================================
# JSON EXPORTER
# ============================================================

class SOSTriageReportExporter:
    @staticmethod
    def export_json(
        result: SOSTriageResult,
        path: str | Path,
        *,
        redact_location: bool = True,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise SOSExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                result.to_dict(
                    redact_location=redact_location
                ),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_sos_triage_agent(
    config: SOSTriageConfig | None = None,
) -> SOSTriageAgent:
    return SOSTriageAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional example only.
    # No actual emergency report is registered
    # with a government or emergency service.

    report = SOSReport(
        sos_id="DEMO-SOS-001",
        reporter=SOSReporter(
            reporter_id="DEMO-REPORTER-001",
        ),
        source_type=SOSSourceType.WEB_PORTAL,
        location=SOSLocation(
            district="Thrissur",
            locality="Illustrative Flood Zone",
            description="Fictional residential area",
        ),
        incident_type=SOSIncidentType.FLOOD_ENTRAPMENT,
        description=(
            "Four people trapped in a building. "
            "Water is rising."
        ),
        people_at_risk=4,
        children_count=1,
        older_adults_count=1,
        trapped=True,
        immediate_life_threat=True,
        rising_water_reported=True,
        evidence=(
            SOSEvidence(
                evidence_id="DEMO-EVIDENCE-001",
                source_id="DEMO-FIELD-SOURCE",
                verification=EvidenceVerification.VERIFIED,
                reliability=0.90,
                observed_at=now - timedelta(minutes=5),
            ),
        ),
        reported_at=now,
    )

    agent = create_sos_triage_agent()

    result = agent.triage(
        report,
        now=now,
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "Urgency:",
        result.urgency_assessment.urgency.value,
    )

    print(
        "Immediate human escalation:",
        result.requires_immediate_escalation,
    )

    reviewed = agent.review_triage(
        result.triage_id,
        reviewer_id="DEMO-OPERATOR-001",
        reviewer_role="emergency_operator",
        decision=ReviewDecision.ESCALATE,
        reason="Fictional immediate-danger review",
    )

    print(
        "Review status:",
        reviewed.status.value,
    )

    SOSTriageReportExporter.export_json(
        reviewed,
        "outputs/agents/sos_triage_report.json",
    )

    print(
        "No emergency response was automatically dispatched."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SOSTriageError",
    "SOSValidationError",
    "SOSPermissionError",
    "SOSStateError",
    "SOSNotFoundError",
    "SOSExportError",
    "SOSIncidentType",
    "SOSUrgency",
    "SOSStatus",
    "EvidenceVerification",
    "SOSSourceType",
    "ResponseCapability",
    "ReviewDecision",
    "SOSTriageConfig",
    "SOSReporter",
    "SOSLocation",
    "SOSEvidence",
    "SOSReport",
    "SOSEvidenceAssessment",
    "SOSUrgencyAssessment",
    "SOSDuplicateCandidate",
    "SOSTriageResult",
    "SOSHumanReview",
    "SOSAuditRecord",
    "SOSReportValidator",
    "SOSEvidenceValidator",
    "SOSDangerSignalEngine",
    "SOSUrgencyEngine",
    "SOSDuplicateScreeningEngine",
    "SOSResponseCapabilityEngine",
    "SOSRecommendationEngine",
    "SOSTriageAuditLogger",
    "SOSHumanReviewManager",
    "SOSTriageAgent",
    "SOSTriageReportExporter",
    "create_sos_triage_agent",
]
