from __future__ import annotations

"""
backend/services/logistics_service.py

Kerala Flood Relief Coordination Agent
Relief Logistics and Supply Chain Coordination Service

Features:
- Relief shipment planning
- Warehouse-to-shelter deliveries
- Inventory allocation recommendations
- Vehicle and driver matching
- Vehicle capacity validation
- Route feasibility and flood hazard checks
- Shipment priority assessment
- Human approval workflow
- Vehicle reservation
- Shipment dispatch
- Real-time shipment status updates
- Delivery proof and confirmation
- Failed delivery handling
- Shipment cancellation
- Audit logging and reporting
- District logistics statistics

SAFETY:
This service coordinates logistics records only.

It does not autonomously dispatch vehicles,
reserve physical inventory, or control drivers.

Production integrations must implement:
- Atomic vehicle and stock reservations
- Verified driver credentials
- Live route safety validation
- Transactional persistence
- Idempotent delivery operations
- Reliable notifications and audit logs
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

class LogisticsPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ShipmentStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    IN_TRANSIT = "in_transit"
    ARRIVED = "arrived"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VehicleType(str, Enum):
    MOTORCYCLE = "motorcycle"
    VAN = "van"
    TRUCK = "truck"
    PICKUP = "pickup"
    AMBULANCE = "ambulance"
    BOAT = "boat"
    RESCUE_VEHICLE = "rescue_vehicle"
    REFRIGERATED_TRUCK = "refrigerated_truck"
    OTHER = "other"


class VehicleStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class DeliveryDestinationType(str, Enum):
    SHELTER = "shelter"
    HOSPITAL = "hospital"
    RELIEF_CAMP = "relief_camp"
    WAREHOUSE = "warehouse"
    DISTRIBUTION_CENTER = "distribution_center"
    FIELD_STATION = "field_station"


class ShipmentItemCategory(str, Enum):
    FOOD = "food"
    WATER = "water"
    MEDICINE = "medicine"
    FIRST_AID = "first_aid"
    CLOTHING = "clothing"
    HYGIENE = "hygiene"
    SHELTER = "shelter"
    RESCUE_EQUIPMENT = "rescue_equipment"
    MEDICAL_EQUIPMENT = "medical_equipment"
    OTHER = "other"


# ============================================================
# EXCEPTIONS
# ============================================================

class LogisticsServiceError(Exception):
    pass


class LogisticsValidationError(LogisticsServiceError):
    pass


class LogisticsNotFoundError(LogisticsServiceError):
    pass


class LogisticsConflictError(LogisticsServiceError):
    pass


class LogisticsAuthorizationError(LogisticsServiceError):
    pass


class LogisticsCapacityError(LogisticsServiceError):
    pass


class LogisticsRouteError(LogisticsServiceError):
    pass


class LogisticsIntegrationError(LogisticsServiceError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LogisticsValidationError(
            f"{name} must be timezone-aware"
        )


def validate_positive(
    value: float,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    if not math.isfinite(value):
        raise LogisticsValidationError(
            f"{name} must be finite"
        )

    if allow_zero:
        valid = value >= 0
    else:
        valid = value > 0

    if not valid:
        raise LogisticsValidationError(
            f"{name} must be "
            + ("non-negative" if allow_zero else "positive")
        )


def require_text(
    value: str,
    name: str,
) -> str:
    if not value or not value.strip():
        raise LogisticsValidationError(
            f"{name} is required"
        )

    return value.strip()


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_km = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return (
        2
        * radius_km
        * math.asin(math.sqrt(min(1.0, max(0.0, a))))
    )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class LogisticsLocation:
    latitude: float
    longitude: float
    name: str = ""
    address: str = ""

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.latitude)
            or not -90 <= self.latitude <= 90
        ):
            raise LogisticsValidationError(
                "Invalid latitude"
            )

        if (
            not math.isfinite(self.longitude)
            or not -180 <= self.longitude <= 180
        ):
            raise LogisticsValidationError(
                "Invalid longitude"
            )


@dataclass(frozen=True)
class ShipmentItem:
    inventory_id: str
    item_name: str
    category: ShipmentItemCategory

    quantity: int
    unit: str

    weight_kg_per_unit: float = 0.0
    volume_m3_per_unit: float = 0.0

    requires_cold_storage: bool = False

    def __post_init__(self) -> None:
        require_text(
            self.inventory_id,
            "Inventory ID",
        )

        require_text(
            self.item_name,
            "Item name",
        )

        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise LogisticsValidationError(
                "Shipment quantity must be a positive integer"
            )

        validate_positive(
            self.weight_kg_per_unit,
            "Unit weight",
            allow_zero=True,
        )

        validate_positive(
            self.volume_m3_per_unit,
            "Unit volume",
            allow_zero=True,
        )

    @property
    def total_weight_kg(self) -> float:
        return self.quantity * self.weight_kg_per_unit

    @property
    def total_volume_m3(self) -> float:
        return self.quantity * self.volume_m3_per_unit


@dataclass(frozen=True)
class ShipmentCreateRequest:
    district_id: int

    source_warehouse_id: str
    destination_id: str
    destination_type: DeliveryDestinationType

    origin: LogisticsLocation
    destination: LogisticsLocation

    items: tuple[ShipmentItem, ...]

    priority: LogisticsPriority = (
        LogisticsPriority.NORMAL
    )

    requested_vehicle_type: VehicleType | None = None

    incident_id: str | None = None
    relief_request_id: str | None = None

    special_instructions: str = ""

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class LogisticsVehicle:
    id: str
    vehicle_type: VehicleType
    district_id: int

    location: LogisticsLocation

    max_payload_kg: float
    max_volume_m3: float

    status: VehicleStatus

    driver_id: str | None = None
    driver_verified: bool = False

    operational: bool = True
    verified: bool = False

    supports_cold_storage: bool = False

    last_location_update: datetime | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class LogisticsRoute:
    route_id: str
    feasible: bool
    verified: bool

    distance_km: float | None = None
    estimated_minutes: float | None = None

    verified_at: datetime | None = None

    flood_affected: bool = False

    blocked_segments: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VehicleRecommendation:
    vehicle_id: str
    vehicle_type: VehicleType

    score: float

    capacity_score: float
    distance_score: float
    route_score: float

    route: LogisticsRoute

    warnings: tuple[str, ...] = ()


@dataclass
class ReliefShipment:
    id: str
    district_id: int

    source_warehouse_id: str
    destination_id: str
    destination_type: DeliveryDestinationType

    origin: LogisticsLocation
    destination: LogisticsLocation

    items: list[ShipmentItem]

    priority: LogisticsPriority
    status: ShipmentStatus

    selected_vehicle_id: str
    route_id: str

    total_weight_kg: float
    total_volume_m3: float

    requested_by: int

    approved_by: int | None = None
    dispatched_by: int | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    approved_at: datetime | None = None
    dispatched_at: datetime | None = None
    delivered_at: datetime | None = None
    cancelled_at: datetime | None = None

    delivery_reference: str | None = None
    received_by: str | None = None

    cancellation_reason: str | None = None
    failure_reason: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DeliveryConfirmation:
    shipment_id: str
    delivery_reference: str

    received_by: str
    confirmed_by: int

    received_at: datetime

    proof_references: tuple[str, ...] = ()
    notes: str = ""


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class LogisticsRepository(Protocol):

    async def create(
        self,
        shipment: ReliefShipment,
    ) -> ReliefShipment:
        ...

    async def get_by_id(
        self,
        shipment_id: str,
    ) -> ReliefShipment | None:
        ...

    async def update(
        self,
        shipment: ReliefShipment,
        *,
        expected_version: int,
    ) -> ReliefShipment:
        """
        Must use optimistic concurrency.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[ReliefShipment]:
        ...

    async def list_by_destination(
        self,
        destination_id: str,
    ) -> Sequence[ReliefShipment]:
        ...


class LogisticsVehicleGateway(Protocol):

    async def list_available(
        self,
        district_id: int,
    ) -> Sequence[LogisticsVehicle]:
        ...

    async def get_vehicle(
        self,
        vehicle_id: str,
    ) -> LogisticsVehicle | None:
        ...

    async def reserve_vehicle(
        self,
        *,
        vehicle_id: str,
        shipment_id: str,
    ) -> bool:
        """
        Atomically reserve vehicle availability.
        Must be idempotent by shipment_id.
        """
        ...

    async def release_vehicle(
        self,
        *,
        vehicle_id: str,
        shipment_id: str,
    ) -> None:
        ...


class LogisticsRouteGateway(Protocol):

    async def assess_route(
        self,
        *,
        vehicle: LogisticsVehicle,
        origin: LogisticsLocation,
        destination: LogisticsLocation,
    ) -> LogisticsRoute:
        ...


class LogisticsInventoryGateway(Protocol):

    async def check_availability(
        self,
        *,
        warehouse_id: str,
        items: Sequence[ShipmentItem],
    ) -> bool:
        ...

    async def reserve_items(
        self,
        *,
        shipment_id: str,
        warehouse_id: str,
        items: Sequence[ShipmentItem],
    ) -> bool:
        """
        Atomically reserve all requested stock
        or fail without partial reservations.
        """
        ...

    async def release_items(
        self,
        *,
        shipment_id: str,
    ) -> None:
        ...

    async def commit_issue(
        self,
        *,
        shipment_id: str,
    ) -> bool:
        """
        Atomically convert stock reservations
        into inventory issue movements.
        """
        ...


class LogisticsAuthorizationGateway(Protocol):

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

    async def can_dispatch(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class LogisticsAuditGateway(Protocol):

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


class LogisticsNotificationGateway(Protocol):

    async def notify_dispatch(
        self,
        shipment: ReliefShipment,
    ) -> bool:
        ...

    async def notify_delivery(
        self,
        shipment: ReliefShipment,
    ) -> bool:
        ...

    async def notify_failure(
        self,
        shipment: ReliefShipment,
    ) -> bool:
        ...


# ============================================================
# VEHICLE MATCHING
# ============================================================

class LogisticsVehicleMatcher:

    def __init__(
        self,
        *,
        maximum_location_age_minutes: int = 20,
        maximum_route_age_minutes: int = 10,
    ):
        self.maximum_location_age_minutes = (
            maximum_location_age_minutes
        )

        self.maximum_route_age_minutes = (
            maximum_route_age_minutes
        )

    def validate_vehicle(
        self,
        vehicle: LogisticsVehicle,
        request: ShipmentCreateRequest,
    ) -> bool:

        if vehicle.district_id != request.district_id:
            return False

        if not vehicle.operational or not vehicle.verified:
            return False

        if vehicle.status != VehicleStatus.AVAILABLE:
            return False

        if not vehicle.driver_id or not vehicle.driver_verified:
            return False

        if (
            request.requested_vehicle_type is not None
            and vehicle.vehicle_type
            != request.requested_vehicle_type
        ):
            return False

        if vehicle.last_location_update is None:
            return False

        validate_datetime(
            vehicle.last_location_update,
            "last_location_update",
        )

        location_age = (
            utc_now() - vehicle.last_location_update
        )

        if (
            location_age < timedelta(0)
            or location_age > timedelta(
                minutes=self.maximum_location_age_minutes
            )
        ):
            return False

        total_weight = sum(
            item.total_weight_kg
            for item in request.items
        )

        total_volume = sum(
            item.total_volume_m3
            for item in request.items
        )

        if total_weight > vehicle.max_payload_kg:
            return False

        if total_volume > vehicle.max_volume_m3:
            return False

        if any(
            item.requires_cold_storage
            for item in request.items
        ) and not vehicle.supports_cold_storage:
            return False

        return True

    def validate_route(
        self,
        route: LogisticsRoute,
    ) -> bool:

        if not route.feasible or not route.verified:
            return False

        if route.blocked_segments:
            return False

        if route.flood_affected:
            return False

        if route.verified_at is None:
            return False

        validate_datetime(
            route.verified_at,
            "route.verified_at",
        )

        age = utc_now() - route.verified_at

        if (
            age < timedelta(0)
            or age > timedelta(
                minutes=self.maximum_route_age_minutes
            )
        ):
            return False

        return True

    def evaluate(
        self,
        vehicle: LogisticsVehicle,
        request: ShipmentCreateRequest,
        route: LogisticsRoute,
    ) -> VehicleRecommendation | None:

        if not self.validate_vehicle(
            vehicle,
            request,
        ):
            return None

        if not self.validate_route(
            route
        ):
            return None

        total_weight = sum(
            item.total_weight_kg
            for item in request.items
        )

        total_volume = sum(
            item.total_volume_m3
            for item in request.items
        )

        weight_utilization = (
            total_weight / vehicle.max_payload_kg
            if vehicle.max_payload_kg > 0
            else 0.0
        )

        volume_utilization = (
            total_volume / vehicle.max_volume_m3
            if vehicle.max_volume_m3 > 0
            else 0.0
        )

        capacity_score = max(
            0.0,
            1.0 - abs(
                max(
                    weight_utilization,
                    volume_utilization,
                ) - 0.75
            ),
        )

        distance = haversine_distance_km(
            vehicle.location.latitude,
            vehicle.location.longitude,
            request.origin.latitude,
            request.origin.longitude,
        )

        distance_score = 1.0 / (
            1.0 + distance / 20.0
        )

        if route.estimated_minutes is not None:
            route_score = 1.0 / (
                1.0 + route.estimated_minutes / 60.0
            )
        else:
            route_score = 0.5

        score = (
            capacity_score * 0.30
            + distance_score * 0.25
            + route_score * 0.45
        )

        return VehicleRecommendation(
            vehicle_id=vehicle.id,
            vehicle_type=vehicle.vehicle_type,
            score=round(score, 4),
            capacity_score=round(
                capacity_score,
                4,
            ),
            distance_score=round(
                distance_score,
                4,
            ),
            route_score=round(
                route_score,
                4,
            ),
            route=route,
            warnings=route.warnings,
        )


# ============================================================
# LOGISTICS SERVICE
# ============================================================

class LogisticsService:

    TERMINAL_STATUSES = frozenset({
        ShipmentStatus.DELIVERED,
        ShipmentStatus.FAILED,
        ShipmentStatus.CANCELLED,
    })

    def __init__(
        self,
        *,
        repository: LogisticsRepository,
        vehicle_gateway: LogisticsVehicleGateway,
        route_gateway: LogisticsRouteGateway,
        inventory_gateway: LogisticsInventoryGateway,
        authorization_gateway: LogisticsAuthorizationGateway,
        audit_gateway: LogisticsAuditGateway | None = None,
        notification_gateway: (
            LogisticsNotificationGateway | None
        ) = None,
        matcher: LogisticsVehicleMatcher | None = None,
    ):
        self.repository = repository
        self.vehicles = vehicle_gateway
        self.routes = route_gateway
        self.inventory = inventory_gateway
        self.authorization = authorization_gateway
        self.audit = audit_gateway
        self.notifications = notification_gateway

        self.matcher = (
            matcher
            or LogisticsVehicleMatcher()
        )

    async def _audit(
        self,
        action: str,
        shipment: ReliefShipment,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="relief_shipment",
            entity_id=shipment.id,
            metadata={
                "district_id": shipment.district_id,
                "vehicle_id": shipment.selected_vehicle_id,
                "destination_id": shipment.destination_id,
                "status": shipment.status.value,
            },
        )

    async def get_shipment(
        self,
        shipment_id: str,
    ) -> ReliefShipment:

        shipment = await self.repository.get_by_id(
            shipment_id
        )

        if shipment is None:
            raise LogisticsNotFoundError(
                "Relief shipment not found"
            )

        return shipment

    def _validate_request(
        self,
        request: ShipmentCreateRequest,
    ) -> None:

        if request.district_id <= 0:
            raise LogisticsValidationError(
                "Invalid district ID"
            )

        require_text(
            request.source_warehouse_id,
            "Source warehouse ID",
        )

        require_text(
            request.destination_id,
            "Destination ID",
        )

        if not request.items:
            raise LogisticsValidationError(
                "Shipment must contain at least one item"
            )

        inventory_ids = [
            item.inventory_id
            for item in request.items
        ]

        if len(inventory_ids) != len(set(inventory_ids)):
            raise LogisticsValidationError(
                "Duplicate inventory batches in shipment"
            )

    async def recommend_vehicles(
        self,
        request: ShipmentCreateRequest,
        *,
        limit: int = 5,
    ) -> list[VehicleRecommendation]:

        self._validate_request(
            request
        )

        if not 1 <= limit <= 50:
            raise LogisticsValidationError(
                "Limit must be between 1 and 50"
            )

        vehicles = await self.vehicles.list_available(
            request.district_id
        )

        recommendations: list[
            VehicleRecommendation
        ] = []

        for vehicle in vehicles:
            if not self.matcher.validate_vehicle(
                vehicle,
                request,
            ):
                continue

            try:
                route = await self.routes.assess_route(
                    vehicle=vehicle,
                    origin=request.origin,
                    destination=request.destination,
                )

                recommendation = self.matcher.evaluate(
                    vehicle,
                    request,
                    route,
                )

                if recommendation is not None:
                    recommendations.append(
                        recommendation
                    )

            except Exception:
                logger.exception(
                    "Vehicle route evaluation failed: %s",
                    vehicle.id,
                )

        recommendations.sort(
            key=lambda item: (
                -item.score,
                item.vehicle_id,
            )
        )

        return recommendations[:limit]

    async def create_shipment(
        self,
        request: ShipmentCreateRequest,
        *,
        vehicle_id: str,
        actor_id: int,
    ) -> ReliefShipment:

        self._validate_request(
            request
        )

        if not await self.authorization.can_plan(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise LogisticsAuthorizationError(
                "Logistics planning permission denied"
            )

        available = await self.inventory.check_availability(
            warehouse_id=request.source_warehouse_id,
            items=request.items,
        )

        if not available:
            raise LogisticsCapacityError(
                "Requested relief inventory is unavailable"
            )

        recommendations = (
            await self.recommend_vehicles(
                request,
                limit=50,
            )
        )

        selected = next(
            (
                item
                for item in recommendations
                if item.vehicle_id == vehicle_id
            ),
            None,
        )

        if selected is None:
            raise LogisticsCapacityError(
                "Selected vehicle is not suitable"
            )

        shipment = ReliefShipment(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            source_warehouse_id=(
                request.source_warehouse_id
            ),
            destination_id=request.destination_id,
            destination_type=request.destination_type,
            origin=request.origin,
            destination=request.destination,
            items=list(request.items),
            priority=request.priority,
            status=ShipmentStatus.DRAFT,
            selected_vehicle_id=vehicle_id,
            route_id=selected.route.route_id,
            total_weight_kg=sum(
                item.total_weight_kg
                for item in request.items
            ),
            total_volume_m3=sum(
                item.total_volume_m3
                for item in request.items
            ),
            requested_by=actor_id,
            metadata={
                **dict(request.metadata),
                "incident_id": request.incident_id,
                "relief_request_id": (
                    request.relief_request_id
                ),
                "special_instructions": (
                    request.special_instructions
                ),
            },
        )

        created = await self.repository.create(
            shipment
        )

        await self._audit(
            "logistics.shipment_created",
            created,
            actor_id,
        )

        return created

    async def submit_for_approval(
        self,
        shipment_id: str,
        *,
        actor_id: int,
    ) -> ReliefShipment:

        shipment = await self.get_shipment(
            shipment_id
        )

        if not await self.authorization.can_plan(
            user_id=actor_id,
            district_id=shipment.district_id,
        ):
            raise LogisticsAuthorizationError(
                "Shipment submission permission denied"
            )

        if shipment.status != ShipmentStatus.DRAFT:
            raise LogisticsConflictError(
                "Only draft shipments can be submitted"
            )

        updated = replace(
            shipment,
            status=ShipmentStatus.PENDING_APPROVAL,
            updated_at=utc_now(),
            version=shipment.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=shipment.version,
        )

        await self._audit(
            "logistics.shipment_submitted",
            saved,
            actor_id,
        )

        return saved

    async def approve_shipment(
        self,
        shipment_id: str,
        *,
        reviewer_id: int,
    ) -> ReliefShipment:

        shipment = await self.get_shipment(
            shipment_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=shipment.district_id,
        ):
            raise LogisticsAuthorizationError(
                "Shipment approval permission denied"
            )

        if (
            shipment.status
            != ShipmentStatus.PENDING_APPROVAL
        ):
            raise LogisticsConflictError(
                "Shipment is not awaiting approval"
            )

        if shipment.requested_by == reviewer_id:
            raise LogisticsAuthorizationError(
                "Independent reviewer required"
            )

        updated = replace(
            shipment,
            status=ShipmentStatus.APPROVED,
            approved_by=reviewer_id,
            approved_at=utc_now(),
            updated_at=utc_now(),
            version=shipment.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=shipment.version,
        )

        await self._audit(
            "logistics.shipment_approved",
            saved,
            reviewer_id,
        )

        return saved

    async def dispatch_shipment(
        self,
        shipment_id: str,
        *,
        actor_id: int,
    ) -> ReliefShipment:

        shipment = await self.get_shipment(
            shipment_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=shipment.district_id,
        ):
            raise LogisticsAuthorizationError(
                "Shipment dispatch permission denied"
            )

        if shipment.status != ShipmentStatus.APPROVED:
            raise LogisticsConflictError(
                "Only approved shipments can be dispatched"
            )

        if (
            shipment.approved_by is None
            or shipment.approved_at is None
        ):
            raise LogisticsAuthorizationError(
                "Human approval is missing"
            )

        vehicle = await self.vehicles.get_vehicle(
            shipment.selected_vehicle_id
        )

        if vehicle is None:
            raise LogisticsCapacityError(
                "Assigned vehicle not found"
            )

        if (
            not vehicle.operational
            or not vehicle.verified
            or not vehicle.driver_verified
            or vehicle.status != VehicleStatus.AVAILABLE
        ):
            raise LogisticsCapacityError(
                "Vehicle or driver is unavailable"
            )

        if (
            shipment.total_weight_kg
            > vehicle.max_payload_kg
            or shipment.total_volume_m3
            > vehicle.max_volume_m3
        ):
            raise LogisticsCapacityError(
                "Vehicle capacity is insufficient"
            )

        route = await self.routes.assess_route(
            vehicle=vehicle,
            origin=shipment.origin,
            destination=shipment.destination,
        )

        if not self.matcher.validate_route(
            route
        ):
            raise LogisticsRouteError(
                "Route is not currently verified as safe"
            )

        vehicle_reserved = (
            await self.vehicles.reserve_vehicle(
                vehicle_id=vehicle.id,
                shipment_id=shipment.id,
            )
        )

        if not vehicle_reserved:
            raise LogisticsConflictError(
                "Vehicle reservation failed"
            )

        try:
            inventory_reserved = (
                await self.inventory.reserve_items(
                    shipment_id=shipment.id,
                    warehouse_id=(
                        shipment.source_warehouse_id
                    ),
                    items=shipment.items,
                )
            )

            if not inventory_reserved:
                raise LogisticsCapacityError(
                    "Inventory reservation failed"
                )

            updated = replace(
                shipment,
                status=ShipmentStatus.RESERVED,
                route_id=route.route_id,
                updated_at=utc_now(),
                version=shipment.version + 1,
            )

            saved = await self.repository.update(
                updated,
                expected_version=shipment.version,
            )

        except Exception:
            await self.inventory.release_items(
                shipment_id=shipment.id,
            )

            await self.vehicles.release_vehicle(
                vehicle_id=vehicle.id,
                shipment_id=shipment.id,
            )

            raise

        await self._audit(
            "logistics.shipment_reserved",
            saved,
            actor_id,
        )

        # The inventory gateway must guarantee an
        # idempotent issue and persistent movement
        # before physical goods leave the warehouse.
        issued = await self.inventory.commit_issue(
            shipment_id=shipment.id
        )

        if not issued:
            raise LogisticsIntegrationError(
                "Inventory issue was not confirmed; "
                "shipment remains reserved for reconciliation"
            )

        dispatched = replace(
            saved,
            status=ShipmentStatus.DISPATCHED,
            dispatched_by=actor_id,
            dispatched_at=utc_now(),
            updated_at=utc_now(),
            version=saved.version + 1,
        )

        final = await self.repository.update(
            dispatched,
            expected_version=saved.version,
        )

        await self._audit(
            "logistics.shipment_dispatched",
            final,
            actor_id,
        )

        if self.notifications is not None:
            try:
                delivered = (
                    await self.notifications.notify_dispatch(
                        final
                    )
                )

                if not delivered:
                    logger.error(
                        "Shipment dispatch notification "
                        "unconfirmed: %s",
                        final.id,
                    )

            except Exception:
                logger.exception(
                    "Shipment dispatch notification failed: %s",
                    final.id,
                )

        return final

    async def update_shipment_status(
        self,
        shipment_id: str,
        *,
        new_status: ShipmentStatus,
        actor_id: int,
    ) -> ReliefShipment:

        shipment = await self.get_shipment(
            shipment_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=shipment.district_id,
        ):
            raise LogisticsAuthorizationError(
                "Shipment update permission denied"
            )

        transitions = {
            ShipmentStatus.DISPATCHED: {
                ShipmentStatus.IN_TRANSIT,
            },
            ShipmentStatus.IN_TRANSIT: {
                ShipmentStatus.ARRIVED,
            },
        }

        allowed = transitions.get(
            shipment.status,
            set(),
        )

        if new_status not in allowed:
            raise LogisticsConflictError(
                f"Invalid shipment transition: "
                f"{shipment.status.value} -> "
                f"{new_status.value}"
            )

        updated = replace(
            shipment,
            status=new_status,
            updated_at=utc_now(),
            version=shipment.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=shipment.version,
        )

        await self._audit(
            f"logistics.{new_status.value}",
            saved,
            actor_id,
        )

        return saved

    async def confirm_delivery(
        self,
        confirmation: DeliveryConfirmation,
    ) -> ReliefShipment:

        shipment = await self.get_shipment(
            confirmation.shipment_id
        )

        if not await self.authorization.can_dispatch(
            user_id=confirmation.confirmed_by,
            district_id=shipment.district_id,
        ):
            raise LogisticsAuthorizationError(
                "Delivery confirmation permission denied"
            )

        if shipment.status != ShipmentStatus.ARRIVED:
            raise LogisticsConflictError(
                "Shipment must arrive before delivery confirmation"
            )

        require_text(
            confirmation.delivery_reference,
            "Delivery reference",
        )

        require_text(
            confirmation.received_by,
            "Receiver",
        )

        validate_datetime(
            confirmation.received_at,
            "received_at",
        )

        if confirmation.received_at > utc_now():
            raise LogisticsValidationError(
                "Delivery time cannot be in the future"
            )

        updated = replace(
            shipment,
            status=ShipmentStatus.DELIVERED,
            delivery_reference=(
                confirmation.delivery_reference
            ),
            received_by=confirmation.received_by,
            delivered_at=confirmation.received_at,
            updated_at=utc_now(),
            version=shipment.version + 1,
            metadata={
                **shipment.metadata,
                "delivery_proof_references": list(
                    confirmation.proof_references
                ),
                "delivery_notes": confirmation.notes,
            },
        )

        saved = await self.repository.update(
            updated,
            expected_version=shipment.version,
        )

        await self.vehicles.release_vehicle(
            vehicle_id=shipment.selected_vehicle_id,
            shipment_id=shipment.id,
        )

        await self._audit(
            "logistics.shipment_delivered",
            saved,
            confirmation.confirmed_by,
        )

        if self.notifications is not None:
            try:
                await self.notifications.notify_delivery(
                    saved
                )
            except Exception:
                logger.exception(
                    "Delivery notification failed: %s",
                    saved.id,
                )

        return saved

    async def fail_shipment(
        self,
        shipment_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> ReliefShipment:

        shipment = await self.get_shipment(
            shipment_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=shipment.district_id,
        ):
            raise LogisticsAuthorizationError(
                "Shipment failure permission denied"
            )

        if shipment.status not in {
            ShipmentStatus.DISPATCHED,
            ShipmentStatus.IN_TRANSIT,
            ShipmentStatus.ARRIVED,
        }:
            raise LogisticsConflictError(
                "Shipment cannot be marked failed"
            )

        require_text(
            reason,
            "Failure reason",
        )

        updated = replace(
            shipment,
            status=ShipmentStatus.FAILED,
            failure_reason=reason.strip(),
            updated_at=utc_now(),
            version=shipment.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=shipment.version,
        )

        # Inventory already issued must be reconciled.
        # Do not automatically return goods to stock.

        await self._audit(
            "logistics.shipment_failed",
            saved,
            actor_id,
        )

        if self.notifications is not None:
            try:
                await self.notifications.notify_failure(
                    saved
                )
            except Exception:
                logger.exception(
                    "Shipment failure notification failed: %s",
                    saved.id,
                )

        return saved

    async def cancel_shipment(
        self,
        shipment_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> ReliefShipment:

        shipment = await self.get_shipment(
            shipment_id
        )

        if not await self.authorization.can_dispatch(
            user_id=actor_id,
            district_id=shipment.district_id,
        ):
            raise LogisticsAuthorizationError(
                "Shipment cancellation permission denied"
            )

        if shipment.status in self.TERMINAL_STATUSES:
            raise LogisticsConflictError(
                "Shipment is already terminal"
            )

        if shipment.status in {
            ShipmentStatus.DISPATCHED,
            ShipmentStatus.IN_TRANSIT,
            ShipmentStatus.ARRIVED,
        }:
            raise LogisticsConflictError(
                "Dispatched shipments require a "
                "return or failure workflow"
            )

        require_text(
            reason,
            "Cancellation reason",
        )

        updated = replace(
            shipment,
            status=ShipmentStatus.CANCELLED,
            cancelled_at=utc_now(),
            cancellation_reason=reason.strip(),
            updated_at=utc_now(),
            version=shipment.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=shipment.version,
        )

        if shipment.status == ShipmentStatus.RESERVED:
            await self.inventory.release_items(
                shipment_id=shipment.id,
            )

            await self.vehicles.release_vehicle(
                vehicle_id=shipment.selected_vehicle_id,
                shipment_id=shipment.id,
            )

        await self._audit(
            "logistics.shipment_cancelled",
            saved,
            actor_id,
        )

        return saved

    async def list_district_shipments(
        self,
        district_id: int,
    ) -> Sequence[ReliefShipment]:

        if district_id <= 0:
            raise LogisticsValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_destination_shipments(
        self,
        destination_id: str,
    ) -> Sequence[ReliefShipment]:

        require_text(
            destination_id,
            "Destination ID",
        )

        return await self.repository.list_by_destination(
            destination_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        shipments = (
            await self.list_district_shipments(
                district_id
            )
        )

        status_counts = {
            status.value: 0
            for status in ShipmentStatus
        }

        priority_counts = {
            priority.value: 0
            for priority in LogisticsPriority
        }

        for shipment in shipments:
            status_counts[
                shipment.status.value
            ] += 1

            priority_counts[
                shipment.priority.value
            ] += 1

        return {
            "district_id": district_id,
            "total_shipments": len(shipments),
            "delivered_shipments": status_counts[
                ShipmentStatus.DELIVERED.value
            ],
            "failed_shipments": status_counts[
                ShipmentStatus.FAILED.value
            ],
            "in_transit_shipments": status_counts[
                ShipmentStatus.IN_TRANSIT.value
            ],
            "total_planned_weight_kg": round(
                sum(
                    shipment.total_weight_kg
                    for shipment in shipments
                ),
                3,
            ),
            "total_planned_volume_m3": round(
                sum(
                    shipment.total_volume_m3
                    for shipment in shipments
                ),
                3,
            ),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_shipment_report(
        self,
        shipment_id: str,
    ) -> dict[str, Any]:

        shipment = await self.get_shipment(
            shipment_id
        )

        return {
            "shipment_id": shipment.id,
            "district_id": shipment.district_id,
            "source_warehouse_id": (
                shipment.source_warehouse_id
            ),
            "destination_id": shipment.destination_id,
            "destination_type": (
                shipment.destination_type.value
            ),
            "priority": shipment.priority.value,
            "status": shipment.status.value,
            "vehicle_id": shipment.selected_vehicle_id,
            "route_id": shipment.route_id,
            "total_weight_kg": (
                shipment.total_weight_kg
            ),
            "total_volume_m3": (
                shipment.total_volume_m3
            ),
            "items": [
                {
                    "inventory_id": item.inventory_id,
                    "item_name": item.item_name,
                    "category": item.category.value,
                    "quantity": item.quantity,
                    "unit": item.unit,
                    "weight_kg": item.total_weight_kg,
                    "volume_m3": item.total_volume_m3,
                    "requires_cold_storage": (
                        item.requires_cold_storage
                    ),
                }
                for item in shipment.items
            ],
            "approval": {
                "approved_by": shipment.approved_by,
                "approved_at": (
                    shipment.approved_at.isoformat()
                    if shipment.approved_at
                    else None
                ),
            },
            "delivery": {
                "delivery_reference": (
                    shipment.delivery_reference
                ),
                "received_by": shipment.received_by,
                "delivered_at": (
                    shipment.delivered_at.isoformat()
                    if shipment.delivered_at
                    else None
                ),
            },
            "created_at": shipment.created_at.isoformat(),
            "updated_at": shipment.updated_at.isoformat(),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Shipment information reflects recorded "
                "coordination status. Vehicle movements, "
                "route safety, stock availability and "
                "delivery confirmation require verified "
                "operational integrations."
            ),
        }
