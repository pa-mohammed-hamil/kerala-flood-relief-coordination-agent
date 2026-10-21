"""Placeholder module: ai/agents/relief_logistics_agent.py; implementation required."""

"""
backend/ai/agents/relief_logistics_agent.py

Kerala Flood Relief Coordination Agent
Relief Logistics & Supply Distribution Agent

Python 3.11+
Standard library only.

Features
--------
- Relief demand registration
- Food, water, medicine and essential supply tracking
- Warehouse stock management
- Shelter and community distribution planning
- Vulnerability-aware priority scoring
- Expiry and inventory verification
- Allocation without double counting stock
- Vehicle capacity and availability checks
- Preliminary delivery scheduling
- Supply shortage and unmet-demand reporting
- Human approval workflow
- Audit logging
- JSON report export

SAFETY
------
This module generates advisory logistics plans only.

It does NOT:
- Dispatch vehicles
- Transfer inventory
- Purchase supplies
- Authorize spending
- Confirm road safety
- Confirm medical suitability
- Execute deliveries

All operational actions require authorized human review.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ReliefLogisticsError(Exception):
    pass


class ReliefValidationError(ReliefLogisticsError):
    pass


class ReliefInventoryError(ReliefLogisticsError):
    pass


class ReliefAllocationError(ReliefLogisticsError):
    pass


class ReliefApprovalError(ReliefLogisticsError):
    pass


class ReliefExportError(ReliefLogisticsError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ReliefItemCategory(str, Enum):
    DRINKING_WATER = "drinking_water"
    FOOD = "food"
    READY_TO_EAT = "ready_to_eat"
    BABY_FOOD = "baby_food"
    MEDICAL_SUPPLIES = "medical_supplies"
    HYGIENE_KITS = "hygiene_kits"
    SANITATION = "sanitation"
    CLOTHING = "clothing"
    BLANKETS = "blankets"
    SHELTER_MATERIALS = "shelter_materials"
    OTHER = "other"


class ReliefPriority(str, Enum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class ReliefRequestStatus(str, Enum):
    PENDING = "pending"
    PARTIALLY_PLANNED = "partially_planned"
    FULLY_PLANNED = "fully_planned"
    REQUIRES_REVIEW = "requires_review"


class WarehouseStatus(str, Enum):
    VERIFIED_OPERATIONAL = "verified_operational"
    LIMITED = "limited"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class VehicleStatus(str, Enum):
    VERIFIED_AVAILABLE = "verified_available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class RoadAccessStatus(str, Enum):
    VERIFIED_ACCESSIBLE = "verified_accessible"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class LogisticsPlanStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED_FOR_PLANNING = "approved_for_planning"
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
        raise ReliefValidationError(
            "Timezone-aware datetime required"
        )


def validate_positive_integer(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ReliefValidationError(
            f"{name} must be a positive integer"
        )


def validate_nonnegative_integer(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ReliefValidationError(
            f"{name} must be a nonnegative integer"
        )


def validate_nonnegative_number(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ReliefValidationError(
            f"{name} must be nonnegative and finite"
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
class ReliefLogisticsConfig:
    max_requests: int = 10000
    max_warehouses: int = 1000
    max_inventory_batches: int = 50000
    max_vehicles: int = 5000

    maximum_inventory_age_minutes: int = 120
    maximum_warehouse_age_minutes: int = 120
    maximum_vehicle_age_minutes: int = 60
    maximum_road_age_minutes: int = 60

    max_vehicle_load_kg: float = 20000.0

    require_verified_inventory: bool = True
    require_verified_road_access: bool = True
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_requests",
            "max_warehouses",
            "max_inventory_batches",
            "max_vehicles",
            "maximum_inventory_age_minutes",
            "maximum_warehouse_age_minutes",
            "maximum_vehicle_age_minutes",
            "maximum_road_age_minutes",
        ):
            validate_positive_integer(
                getattr(self, name),
                name,
            )

        validate_nonnegative_number(
            self.max_vehicle_load_kg,
            "max_vehicle_load_kg",
        )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ReliefDemandItem:
    item_id: str
    category: ReliefItemCategory
    quantity: int
    unit: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ReliefValidationError(
                "item_id required"
            )

        if not self.unit.strip():
            raise ReliefValidationError(
                "unit required"
            )

        validate_positive_integer(
            self.quantity,
            "quantity",
        )


@dataclass(frozen=True)
class ReliefRequest:
    request_id: str

    district: str
    destination_id: str
    destination_name: str

    affected_people: int
    children_count: int = 0
    older_adults_count: int = 0
    medically_vulnerable_count: int = 0

    items: tuple[ReliefDemandItem, ...] = ()

    requested_at: datetime | None = None

    urgent_need: bool = False

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "district",
            "destination_id",
            "destination_name",
        ):
            if not getattr(self, name).strip():
                raise ReliefValidationError(
                    f"{name} required"
                )

        validate_positive_integer(
            self.affected_people,
            "affected_people",
        )

        for name in (
            "children_count",
            "older_adults_count",
            "medically_vulnerable_count",
        ):
            value = getattr(self, name)

            validate_nonnegative_integer(
                value,
                name,
            )

            if value > self.affected_people:
                raise ReliefValidationError(
                    f"{name} exceeds affected_people"
                )

        if self.requested_at is not None:
            validate_datetime(
                self.requested_at
            )

        if not self.items:
            raise ReliefValidationError(
                "At least one relief item required"
            )

        item_ids = [
            item.item_id
            for item in self.items
        ]

        if len(item_ids) != len(set(item_ids)):
            raise ReliefValidationError(
                "Duplicate demand item ID"
            )


@dataclass(frozen=True)
class Warehouse:
    warehouse_id: str
    warehouse_name: str

    district: str

    status: WarehouseStatus
    verified_at: datetime | None

    def __post_init__(self) -> None:
        if not self.warehouse_id.strip():
            raise ReliefValidationError(
                "warehouse_id required"
            )

        if not self.district.strip():
            raise ReliefValidationError(
                "district required"
            )

        if self.verified_at is not None:
            validate_datetime(
                self.verified_at
            )


@dataclass(frozen=True)
class InventoryBatch:
    batch_id: str
    warehouse_id: str

    category: ReliefItemCategory

    quantity_available: int
    unit: str

    unit_weight_kg: float

    verified: bool
    verified_at: datetime | None

    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.batch_id.strip():
            raise ReliefValidationError(
                "batch_id required"
            )

        if not self.warehouse_id.strip():
            raise ReliefValidationError(
                "warehouse_id required"
            )

        if not self.unit.strip():
            raise ReliefValidationError(
                "unit required"
            )

        validate_nonnegative_integer(
            self.quantity_available,
            "quantity_available",
        )

        validate_nonnegative_number(
            self.unit_weight_kg,
            "unit_weight_kg",
        )

        if self.verified_at is not None:
            validate_datetime(
                self.verified_at
            )

        if self.expires_at is not None:
            validate_datetime(
                self.expires_at
            )


@dataclass(frozen=True)
class ReliefVehicle:
    vehicle_id: str
    district: str

    status: VehicleStatus

    capacity_kg: float

    verified_at: datetime | None

    def __post_init__(self) -> None:
        if not self.vehicle_id.strip():
            raise ReliefValidationError(
                "vehicle_id required"
            )

        validate_nonnegative_number(
            self.capacity_kg,
            "capacity_kg",
        )

        if self.verified_at is not None:
            validate_datetime(
                self.verified_at
            )


@dataclass(frozen=True)
class ReliefRoadAccess:
    warehouse_id: str
    destination_id: str

    status: RoadAccessStatus

    verified_at: datetime | None

    def __post_init__(self) -> None:
        if not self.warehouse_id.strip():
            raise ReliefValidationError(
                "warehouse_id required"
            )

        if not self.destination_id.strip():
            raise ReliefValidationError(
                "destination_id required"
            )

        if self.verified_at is not None:
            validate_datetime(
                self.verified_at
            )


@dataclass(frozen=True)
class ReliefLogisticsPlanningRequest:
    planning_request_id: str
    district: str

    relief_requests: tuple[ReliefRequest, ...]
    warehouses: tuple[Warehouse, ...]
    inventory_batches: tuple[InventoryBatch, ...]
    vehicles: tuple[ReliefVehicle, ...]
    road_access: tuple[ReliefRoadAccess, ...]

    created_at: datetime


@dataclass(frozen=True)
class ReliefPriorityAssessment:
    request_id: str

    score: float
    priority: ReliefPriority

    factors: Mapping[str, float]


@dataclass(frozen=True)
class ReliefAllocation:
    allocation_id: str

    request_id: str
    item_id: str

    warehouse_id: str
    batch_id: str

    category: ReliefItemCategory

    quantity: int
    unit: str

    total_weight_kg: float


@dataclass(frozen=True)
class ReliefUnmetDemand:
    request_id: str
    item_id: str

    category: ReliefItemCategory

    requested_quantity: int
    planned_quantity: int
    shortage_quantity: int

    unit: str


@dataclass(frozen=True)
class ReliefVehicleAssignment:
    assignment_id: str
    vehicle_id: str

    request_id: str
    warehouse_id: str
    destination_id: str

    allocation_ids: tuple[str, ...]

    total_load_kg: float

    advisory_only: bool = True


@dataclass(frozen=True)
class ReliefRequestPlanningResult:
    request_id: str

    priority: ReliefPriority
    status: ReliefRequestStatus

    allocated_item_count: int
    unmet_item_count: int

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ReliefLogisticsPlan:
    plan_id: str
    planning_request_id: str

    district: str

    priority_assessments: tuple[
        ReliefPriorityAssessment,
        ...
    ]

    allocations: tuple[
        ReliefAllocation,
        ...
    ]

    unmet_demand: tuple[
        ReliefUnmetDemand,
        ...
    ]

    vehicle_assignments: tuple[
        ReliefVehicleAssignment,
        ...
    ]

    request_results: tuple[
        ReliefRequestPlanningResult,
        ...
    ]

    warnings: tuple[str, ...]

    status: LogisticsPlanStatus

    requires_human_approval: bool

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ReliefApprovalRecord:
    approval_id: str
    plan_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ApprovalDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class ReliefAuditRecord:
    audit_id: str
    plan_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# PRIORITY ENGINE
# ============================================================

class ReliefPriorityEngine:
    @staticmethod
    def assess(
        request: ReliefRequest,
    ) -> ReliefPriorityAssessment:
        population_factor = min(
            request.affected_people / 1000,
            1.0,
        )

        vulnerable_count = min(
            request.affected_people,
            request.children_count
            + request.older_adults_count
            + request.medically_vulnerable_count,
        )

        vulnerability_factor = (
            vulnerable_count
            / request.affected_people
        )

        urgent_factor = float(
            request.urgent_need
        )

        essential_categories = {
            ReliefItemCategory.DRINKING_WATER,
            ReliefItemCategory.FOOD,
            ReliefItemCategory.READY_TO_EAT,
            ReliefItemCategory.BABY_FOOD,
            ReliefItemCategory.MEDICAL_SUPPLIES,
        }

        essential_factor = float(
            any(
                item.category in essential_categories
                for item in request.items
            )
        )

        score = (
            0.20 * population_factor
            + 0.25 * vulnerability_factor
            + 0.35 * urgent_factor
            + 0.20 * essential_factor
        )

        score = round(
            min(score, 1.0),
            4,
        )

        if score >= 0.80:
            priority = ReliefPriority.CRITICAL
        elif score >= 0.60:
            priority = ReliefPriority.HIGH
        elif score >= 0.35:
            priority = ReliefPriority.ELEVATED
        else:
            priority = ReliefPriority.ROUTINE

        return ReliefPriorityAssessment(
            request_id=request.request_id,
            score=score,
            priority=priority,
            factors={
                "population": population_factor,
                "vulnerability": vulnerability_factor,
                "urgent_need": urgent_factor,
                "essential_supplies": essential_factor,
            },
        )


# ============================================================
# INVENTORY VALIDATION ENGINE
# ============================================================

class ReliefInventoryValidator:
    def __init__(
        self,
        config: ReliefLogisticsConfig,
    ) -> None:
        self.config = config

    def warehouse_eligible(
        self,
        warehouse: Warehouse,
        *,
        now: datetime,
    ) -> bool:
        if warehouse.status != (
            WarehouseStatus.VERIFIED_OPERATIONAL
        ):
            return False

        if warehouse.verified_at is None:
            return False

        if warehouse.verified_at > now:
            return False

        return (
            now - warehouse.verified_at
            <= timedelta(
                minutes=(
                    self.config.maximum_warehouse_age_minutes
                )
            )
        )

    def batch_eligible(
        self,
        batch: InventoryBatch,
        *,
        now: datetime,
    ) -> bool:
        if batch.quantity_available <= 0:
            return False

        if not batch.verified:
            return False

        if batch.verified_at is None:
            return False

        if batch.verified_at > now:
            return False

        if (
            now - batch.verified_at
        ) > timedelta(
            minutes=(
                self.config.maximum_inventory_age_minutes
            )
        ):
            return False

        if (
            batch.expires_at is not None
            and batch.expires_at <= now
        ):
            return False

        return True


# ============================================================
# ROAD ACCESS ENGINE
# ============================================================

class ReliefRoadAccessEngine:
    def __init__(
        self,
        config: ReliefLogisticsConfig,
    ) -> None:
        self.config = config

    def accessible(
        self,
        *,
        warehouse_id: str,
        destination_id: str,
        road_access: Sequence[ReliefRoadAccess],
        now: datetime,
    ) -> bool:
        for road in road_access:
            if (
                road.warehouse_id != warehouse_id
                or road.destination_id != destination_id
            ):
                continue

            if road.status != (
                RoadAccessStatus.VERIFIED_ACCESSIBLE
            ):
                continue

            if road.verified_at is None:
                continue

            if road.verified_at > now:
                continue

            if (
                now - road.verified_at
            ) <= timedelta(
                minutes=(
                    self.config.maximum_road_age_minutes
                )
            ):
                return True

        return False


# ============================================================
# ALLOCATION ENGINE
# ============================================================

class ReliefAllocationEngine:
    """
    Creates tentative allocations.

    Inventory balances are local to a planning run.
    No persistent inventory is modified.

    Matching requires:
        - Same category
        - Same unit
        - Eligible warehouse
        - Eligible inventory batch
        - Verified road access

    Vehicle capacity is handled later.
    """

    def __init__(
        self,
        inventory_validator: ReliefInventoryValidator,
        road_engine: ReliefRoadAccessEngine,
    ) -> None:
        self.inventory_validator = (
            inventory_validator
        )

        self.road_engine = road_engine

    def allocate(
        self,
        *,
        requests: Sequence[ReliefRequest],
        priorities: Mapping[
            str,
            ReliefPriorityAssessment,
        ],
        warehouses: Sequence[Warehouse],
        batches: Sequence[InventoryBatch],
        road_access: Sequence[ReliefRoadAccess],
        now: datetime,
    ) -> tuple[
        tuple[ReliefAllocation, ...],
        tuple[ReliefUnmetDemand, ...],
    ]:
        warehouse_map = {
            warehouse.warehouse_id: warehouse
            for warehouse in warehouses
        }

        balances = {
            batch.batch_id: batch.quantity_available
            for batch in batches
        }

        allocations = []
        unmet = []

        ordered_requests = sorted(
            requests,
            key=lambda request: (
                -priorities[request.request_id].score,
                request.requested_at or now,
                request.request_id,
            ),
        )

        for request in ordered_requests:
            for demand in request.items:
                remaining = demand.quantity

                eligible_batches = []

                for batch in batches:
                    warehouse = warehouse_map.get(
                        batch.warehouse_id
                    )

                    if warehouse is None:
                        continue

                    if batch.category != demand.category:
                        continue

                    if batch.unit != demand.unit:
                        continue

                    if (
                        warehouse.district.casefold()
                        != request.district.casefold()
                    ):
                        continue

                    if not self.inventory_validator.warehouse_eligible(
                        warehouse,
                        now=now,
                    ):
                        continue

                    if not self.inventory_validator.batch_eligible(
                        batch,
                        now=now,
                    ):
                        continue

                    if not self.road_engine.accessible(
                        warehouse_id=batch.warehouse_id,
                        destination_id=request.destination_id,
                        road_access=road_access,
                        now=now,
                    ):
                        continue

                    eligible_batches.append(batch)

                # First-expiring-first-out where
                # expiry information is available.
                eligible_batches.sort(
                    key=lambda batch: (
                        batch.expires_at
                        or datetime.max.replace(
                            tzinfo=timezone.utc
                        ),
                        batch.batch_id,
                    )
                )

                for batch in eligible_batches:
                    if remaining <= 0:
                        break

                    available = balances[
                        batch.batch_id
                    ]

                    if available <= 0:
                        continue

                    quantity = min(
                        available,
                        remaining,
                    )

                    allocations.append(
                        ReliefAllocation(
                            allocation_id=generate_id(),
                            request_id=request.request_id,
                            item_id=demand.item_id,
                            warehouse_id=batch.warehouse_id,
                            batch_id=batch.batch_id,
                            category=demand.category,
                            quantity=quantity,
                            unit=demand.unit,
                            total_weight_kg=round(
                                quantity
                                * batch.unit_weight_kg,
                                4,
                            ),
                        )
                    )

                    balances[
                        batch.batch_id
                    ] -= quantity

                    remaining -= quantity

                unmet.append(
                    ReliefUnmetDemand(
                        request_id=request.request_id,
                        item_id=demand.item_id,
                        category=demand.category,
                        requested_quantity=demand.quantity,
                        planned_quantity=(
                            demand.quantity - remaining
                        ),
                        shortage_quantity=remaining,
                        unit=demand.unit,
                    )
                )

        return (
            tuple(allocations),
            tuple(unmet),
        )


# ============================================================
# VEHICLE PLANNING ENGINE
# ============================================================

class ReliefVehiclePlanningEngine:
    def __init__(
        self,
        config: ReliefLogisticsConfig,
    ) -> None:
        self.config = config

    def eligible_vehicle(
        self,
        vehicle: ReliefVehicle,
        *,
        now: datetime,
    ) -> bool:
        if vehicle.status != (
            VehicleStatus.VERIFIED_AVAILABLE
        ):
            return False

        if vehicle.verified_at is None:
            return False

        if vehicle.verified_at > now:
            return False

        if (
            now - vehicle.verified_at
        ) > timedelta(
            minutes=(
                self.config.maximum_vehicle_age_minutes
            )
        ):
            return False

        return (
            0 < vehicle.capacity_kg
            <= self.config.max_vehicle_load_kg
        )

    def assign(
        self,
        *,
        allocations: Sequence[ReliefAllocation],
        vehicles: Sequence[ReliefVehicle],
        district: str,
        now: datetime,
    ) -> tuple[
        tuple[ReliefVehicleAssignment, ...],
        tuple[str, ...],
    ]:
        eligible = [
            vehicle
            for vehicle in vehicles
            if (
                vehicle.district.casefold()
                == district.casefold()
                and self.eligible_vehicle(
                    vehicle,
                    now=now,
                )
            )
        ]

        eligible.sort(
            key=lambda vehicle: (
                -vehicle.capacity_kg,
                vehicle.vehicle_id,
            )
        )

        # Group by destination and source warehouse.
        groups = defaultdict(list)

        for allocation in allocations:
            groups[
                (
                    allocation.request_id,
                    allocation.warehouse_id,
                )
            ].append(allocation)

        assignments = []
        warnings = []

        unused_vehicles = {
            vehicle.vehicle_id: vehicle
            for vehicle in eligible
        }

        for (
            request_id,
            warehouse_id,
        ), group in sorted(groups.items()):
            total_weight = sum(
                item.total_weight_kg
                for item in group
            )

            # This simplified planner requires
            # a single vehicle per grouped shipment.
            # It does not split allocations.
            candidates = [
                vehicle
                for vehicle in unused_vehicles.values()
                if vehicle.capacity_kg >= total_weight
            ]

            if not candidates:
                warnings.append(
                    "No verified single vehicle has "
                    f"sufficient capacity for request "
                    f"{request_id}, warehouse {warehouse_id}"
                )
                continue

            selected = min(
                candidates,
                key=lambda vehicle: (
                    vehicle.capacity_kg,
                    vehicle.vehicle_id,
                ),
            )

            assignments.append(
                ReliefVehicleAssignment(
                    assignment_id=generate_id(),
                    vehicle_id=selected.vehicle_id,
                    request_id=request_id,
                    warehouse_id=warehouse_id,
                    destination_id=request_id,
                    allocation_ids=tuple(
                        item.allocation_id
                        for item in group
                    ),
                    total_load_kg=round(
                        total_weight,
                        4,
                    ),
                )
            )

            del unused_vehicles[
                selected.vehicle_id
            ]

        return (
            tuple(assignments),
            tuple(warnings),
        )


# ============================================================
# REQUEST STATUS ENGINE
# ============================================================

class ReliefRequestStatusEngine:
    @staticmethod
    def summarize(
        *,
        requests: Sequence[ReliefRequest],
        priorities: Mapping[
            str,
            ReliefPriorityAssessment,
        ],
        unmet: Sequence[ReliefUnmetDemand],
    ) -> tuple[
        ReliefRequestPlanningResult,
        ...
    ]:
        unmet_by_request = defaultdict(list)

        for item in unmet:
            unmet_by_request[
                item.request_id
            ].append(item)

        results = []

        for request in requests:
            items = unmet_by_request[
                request.request_id
            ]

            fully_planned = sum(
                item.shortage_quantity == 0
                for item in items
            )

            partially_planned = sum(
                item.planned_quantity > 0
                for item in items
            )

            if fully_planned == len(request.items):
                status = (
                    ReliefRequestStatus.FULLY_PLANNED
                )

            elif partially_planned > 0:
                status = (
                    ReliefRequestStatus.PARTIALLY_PLANNED
                )

            else:
                status = (
                    ReliefRequestStatus.REQUIRES_REVIEW
                )

            warnings = []

            if status != (
                ReliefRequestStatus.FULLY_PLANNED
            ):
                warnings.append(
                    "Some requested supplies remain unallocated"
                )

            results.append(
                ReliefRequestPlanningResult(
                    request_id=request.request_id,
                    priority=priorities[
                        request.request_id
                    ].priority,
                    status=status,
                    allocated_item_count=fully_planned,
                    unmet_item_count=sum(
                        item.shortage_quantity > 0
                        for item in items
                    ),
                    warnings=tuple(warnings),
                )
            )

        return tuple(results)


# ============================================================
# AUDIT LOGGER
# ============================================================

class ReliefLogisticsAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            ReliefAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        plan_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> ReliefAuditRecord:
        record = ReliefAuditRecord(
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
    ) -> tuple[ReliefAuditRecord, ...]:
        with self._lock:
            records = tuple(self._records)

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

class ReliefLogisticsApprovalManager:
    ALLOWED_ROLES = {
        "district_official",
        "relief_coordinator",
        "logistics_manager",
        "disaster_management_officer",
    }

    def __init__(
        self,
        audit_logger: ReliefLogisticsAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            ReliefApprovalRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        plan: ReliefLogisticsPlan,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ApprovalDecision,
        reason: str,
    ) -> ReliefApprovalRecord:
        if not reviewer_id.strip():
            raise ReliefApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise ReliefApprovalError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise ReliefApprovalError(
                "Review reason required"
            )

        with self._lock:
            if plan.plan_id in self._reviews:
                raise ReliefApprovalError(
                    "Plan already reviewed"
                )

            review = ReliefApprovalRecord(
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
# MAIN RELIEF LOGISTICS AGENT
# ============================================================

class ReliefLogisticsAgent:
    def __init__(
        self,
        config: ReliefLogisticsConfig | None = None,
    ) -> None:
        self.config = (
            config or ReliefLogisticsConfig()
        )

        self.priority_engine = (
            ReliefPriorityEngine()
        )

        self.inventory_validator = (
            ReliefInventoryValidator(
                self.config
            )
        )

        self.road_engine = (
            ReliefRoadAccessEngine(
                self.config
            )
        )

        self.allocation_engine = (
            ReliefAllocationEngine(
                self.inventory_validator,
                self.road_engine,
            )
        )

        self.vehicle_engine = (
            ReliefVehiclePlanningEngine(
                self.config
            )
        )

        self.status_engine = (
            ReliefRequestStatusEngine()
        )

        self.audit_logger = (
            ReliefLogisticsAuditLogger()
        )

        self.approval_manager = (
            ReliefLogisticsApprovalManager(
                self.audit_logger
            )
        )

        self._plans: dict[
            str,
            ReliefLogisticsPlan,
        ] = {}

        self._lock = threading.RLock()

    def run(
        self,
        request: ReliefLogisticsPlanningRequest,
        *,
        now: datetime | None = None,
    ) -> ReliefLogisticsPlan:
        now = now or utc_now()

        validate_datetime(now)
        validate_datetime(request.created_at)

        if request.created_at > now:
            raise ReliefValidationError(
                "Planning request timestamp is in the future"
            )

        if not request.planning_request_id.strip():
            raise ReliefValidationError(
                "Planning request ID required"
            )

        if not request.district.strip():
            raise ReliefValidationError(
                "District required"
            )

        if len(request.relief_requests) > (
            self.config.max_requests
        ):
            raise ReliefValidationError(
                "Request limit exceeded"
            )

        if len(request.warehouses) > (
            self.config.max_warehouses
        ):
            raise ReliefValidationError(
                "Warehouse limit exceeded"
            )

        if len(request.inventory_batches) > (
            self.config.max_inventory_batches
        ):
            raise ReliefValidationError(
                "Inventory batch limit exceeded"
            )

        if len(request.vehicles) > (
            self.config.max_vehicles
        ):
            raise ReliefValidationError(
                "Vehicle limit exceeded"
            )

        for collection, attribute in (
            (request.relief_requests, "request_id"),
            (request.warehouses, "warehouse_id"),
            (request.inventory_batches, "batch_id"),
            (request.vehicles, "vehicle_id"),
        ):
            ids = [
                getattr(item, attribute)
                for item in collection
            ]

            if len(ids) != len(set(ids)):
                raise ReliefValidationError(
                    f"Duplicate {attribute}"
                )

        priorities = {}

        for relief_request in request.relief_requests:
            if (
                relief_request.district.casefold()
                != request.district.casefold()
            ):
                raise ReliefValidationError(
                    "Relief request district mismatch"
                )

            priorities[
                relief_request.request_id
            ] = self.priority_engine.assess(
                relief_request
            )

        allocations, unmet = (
            self.allocation_engine.allocate(
                requests=request.relief_requests,
                priorities=priorities,
                warehouses=request.warehouses,
                batches=request.inventory_batches,
                road_access=request.road_access,
                now=now,
            )
        )

        assignments, vehicle_warnings = (
            self.vehicle_engine.assign(
                allocations=allocations,
                vehicles=request.vehicles,
                district=request.district,
                now=now,
            )
        )

        request_results = (
            self.status_engine.summarize(
                requests=request.relief_requests,
                priorities=priorities,
                unmet=unmet,
            )
        )

        warnings = list(
            vehicle_warnings
        )

        if any(
            item.shortage_quantity > 0
            for item in unmet
        ):
            warnings.append(
                "Unmet relief supply demand detected"
            )

        warnings.extend([
            "All allocations are provisional",
            "Inventory has not been reserved or deducted",
            "Vehicles have not been dispatched",
            "No delivery time or route is guaranteed",
            "Road accessibility requires operational rechecking",
            "Vehicle assignments are preliminary",
            "Medical supplies require qualified review",
            "Food and water safety must be independently verified",
            "Human approval is required before execution",
        ])

        plan = ReliefLogisticsPlan(
            plan_id=generate_id(),
            planning_request_id=(
                request.planning_request_id
            ),
            district=request.district,
            priority_assessments=tuple(
                priorities.values()
            ),
            allocations=allocations,
            unmet_demand=unmet,
            vehicle_assignments=assignments,
            request_results=request_results,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            status=LogisticsPlanStatus.PENDING_APPROVAL,
            requires_human_approval=True,
            created_at=now,
        )

        with self._lock:
            self._plans[
                plan.plan_id
            ] = plan

        self.audit_logger.record(
            plan_id=plan.plan_id,
            actor_id="relief_logistics_agent",
            action="logistics_plan_created",
            details={
                "district": request.district,
                "request_count": str(
                    len(request.relief_requests)
                ),
                "allocation_count": str(
                    len(allocations)
                ),
                "vehicle_assignment_count": str(
                    len(assignments)
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
    ) -> ReliefLogisticsPlan:
        with self._lock:
            plan = self._plans.get(
                plan_id
            )

            if plan is None:
                raise ReliefApprovalError(
                    "Plan not found"
                )

            if plan.status != (
                LogisticsPlanStatus.PENDING_APPROVAL
            ):
                raise ReliefApprovalError(
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
                status = (
                    LogisticsPlanStatus.APPROVED_FOR_PLANNING
                )

            elif decision == ApprovalDecision.REJECT:
                status = (
                    LogisticsPlanStatus.REJECTED
                )

            else:
                status = (
                    LogisticsPlanStatus.REVISION_REQUIRED
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
    ) -> ReliefLogisticsPlan | None:
        with self._lock:
            return self._plans.get(
                plan_id
            )

    def get_audit_history(
        self,
        plan_id: str,
    ) -> tuple[ReliefAuditRecord, ...]:
        return self.audit_logger.list_records(
            plan_id
        )


# ============================================================
# JSON EXPORTER
# ============================================================

class ReliefLogisticsReportExporter:
    @staticmethod
    def export_json(
        plan: ReliefLogisticsPlan,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise ReliefExportError(
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

def create_relief_logistics_agent(
    config: ReliefLogisticsConfig | None = None,
) -> ReliefLogisticsAgent:
    return ReliefLogisticsAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # All IDs, inventory and resources below
    # are fictional demonstration data.

    relief_request = ReliefRequest(
        request_id="DEMO-RELIEF-001",
        district="Thrissur",
        destination_id="DEMO-SHELTER-001",
        destination_name="Illustrative Relief Shelter",
        affected_people=300,
        children_count=60,
        older_adults_count=35,
        medically_vulnerable_count=20,
        urgent_need=True,
        items=(
            ReliefDemandItem(
                item_id="WATER-001",
                category=ReliefItemCategory.DRINKING_WATER,
                quantity=500,
                unit="bottle",
            ),
            ReliefDemandItem(
                item_id="FOOD-001",
                category=ReliefItemCategory.FOOD,
                quantity=300,
                unit="packet",
            ),
        ),
        requested_at=now,
    )

    warehouse = Warehouse(
        warehouse_id="DEMO-WAREHOUSE-001",
        warehouse_name="Illustrative District Warehouse",
        district="Thrissur",
        status=WarehouseStatus.VERIFIED_OPERATIONAL,
        verified_at=now - timedelta(minutes=10),
    )

    inventory = (
        InventoryBatch(
            batch_id="WATER-BATCH-001",
            warehouse_id=warehouse.warehouse_id,
            category=ReliefItemCategory.DRINKING_WATER,
            quantity_available=700,
            unit="bottle",
            unit_weight_kg=1.0,
            verified=True,
            verified_at=now - timedelta(minutes=10),
            expires_at=now + timedelta(days=120),
        ),
        InventoryBatch(
            batch_id="FOOD-BATCH-001",
            warehouse_id=warehouse.warehouse_id,
            category=ReliefItemCategory.FOOD,
            quantity_available=250,
            unit="packet",
            unit_weight_kg=0.5,
            verified=True,
            verified_at=now - timedelta(minutes=10),
            expires_at=now + timedelta(days=30),
        ),
    )

    vehicle = ReliefVehicle(
        vehicle_id="DEMO-TRUCK-001",
        district="Thrissur",
        status=VehicleStatus.VERIFIED_AVAILABLE,
        capacity_kg=2000,
        verified_at=now - timedelta(minutes=5),
    )

    road = ReliefRoadAccess(
        warehouse_id=warehouse.warehouse_id,
        destination_id=relief_request.destination_id,
        status=RoadAccessStatus.VERIFIED_ACCESSIBLE,
        verified_at=now - timedelta(minutes=5),
    )

    planning_request = ReliefLogisticsPlanningRequest(
        planning_request_id="DEMO-PLAN-REQUEST-001",
        district="Thrissur",
        relief_requests=(relief_request,),
        warehouses=(warehouse,),
        inventory_batches=inventory,
        vehicles=(vehicle,),
        road_access=(road,),
        created_at=now,
    )

    agent = create_relief_logistics_agent()

    plan = agent.run(
        planning_request,
        now=now,
    )

    print(
        json.dumps(
            plan.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nTotal provisional allocations:",
        len(plan.allocations),
    )

    print(
        "Vehicle assignments:",
        len(plan.vehicle_assignments),
    )

    reviewed = agent.review_plan(
        plan.plan_id,
        reviewer_id="DEMO-LOGISTICS-OFFICER",
        reviewer_role="logistics_manager",
        decision=ApprovalDecision.APPROVE,
        reason="Illustrative planning approval",
    )

    print(
        "Plan status:",
        reviewed.status.value,
    )

    ReliefLogisticsReportExporter.export_json(
        reviewed,
        "outputs/agents/relief_logistics_report.json",
    )

    print(
        "No supplies were transferred "
        "and no vehicles were dispatched."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ReliefLogisticsError",
    "ReliefValidationError",
    "ReliefInventoryError",
    "ReliefAllocationError",
    "ReliefApprovalError",
    "ReliefExportError",
    "ReliefItemCategory",
    "ReliefPriority",
    "ReliefRequestStatus",
    "WarehouseStatus",
    "VehicleStatus",
    "RoadAccessStatus",
    "LogisticsPlanStatus",
    "ApprovalDecision",
    "ReliefLogisticsConfig",
    "ReliefDemandItem",
    "ReliefRequest",
    "Warehouse",
    "InventoryBatch",
    "ReliefVehicle",
    "ReliefRoadAccess",
    "ReliefLogisticsPlanningRequest",
    "ReliefPriorityAssessment",
    "ReliefAllocation",
    "ReliefUnmetDemand",
    "ReliefVehicleAssignment",
    "ReliefRequestPlanningResult",
    "ReliefLogisticsPlan",
    "ReliefApprovalRecord",
    "ReliefAuditRecord",
    "ReliefPriorityEngine",
    "ReliefInventoryValidator",
    "ReliefRoadAccessEngine",
    "ReliefAllocationEngine",
    "ReliefVehiclePlanningEngine",
    "ReliefRequestStatusEngine",
    "ReliefLogisticsAuditLogger",
    "ReliefLogisticsApprovalManager",
    "ReliefLogisticsAgent",
    "ReliefLogisticsReportExporter",
    "create_relief_logistics_agent",
]
