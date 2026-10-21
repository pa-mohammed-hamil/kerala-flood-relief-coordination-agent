"""Placeholder module: ai/agents/sos_triage_agent.py; implementation required."""

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
