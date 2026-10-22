from __future__ import annotations

"""
backend/services/inventory_service.py

Kerala Flood Relief Coordination Agent
Relief Inventory Management Service

Features:
- Relief inventory registration
- Food, water, medicine, clothing and equipment tracking
- District and warehouse inventory management
- Stock receiving and issuing
- Atomic stock reservations
- Reservation fulfillment and cancellation
- Low-stock and out-of-stock detection
- Expiry and batch tracking
- FEFO stock allocation
- Inventory transfers
- Donation inventory tracking
- Stock adjustment approval
- Inventory movement history
- Stock reconciliation
- Inventory statistics
- Audit logging
- Optimistic concurrency support

SAFETY:
Medical inventory records are not a substitute
for pharmaceutical handling procedures.

Expired, recalled, quarantined or damaged stock
must never be allocated for distribution.

All stock mutations must be atomic in the
persistent repository. In-memory calculations
alone cannot prevent concurrent overselling.
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

class InventoryCategory(str, Enum):
    FOOD = "food"
    DRINKING_WATER = "drinking_water"
    MEDICINE = "medicine"
    FIRST_AID = "first_aid"
    CLOTHING = "clothing"
    BLANKETS = "blankets"
    HYGIENE = "hygiene"
    SANITATION = "sanitation"
    BABY_SUPPLIES = "baby_supplies"
    WOMEN_ESSENTIALS = "women_essentials"
    SHELTER_MATERIAL = "shelter_material"
    RESCUE_EQUIPMENT = "rescue_equipment"
    MEDICAL_EQUIPMENT = "medical_equipment"
    FUEL = "fuel"
    POWER_SUPPLIES = "power_supplies"
    COMMUNICATION = "communication"
    OTHER = "other"


class InventoryUnit(str, Enum):
    PIECE = "piece"
    PACK = "pack"
    BOX = "box"
    KIT = "kit"
    BOTTLE = "bottle"
    LITER = "liter"
    KILOGRAM = "kilogram"
    GRAM = "gram"
    BAG = "bag"
    CARTON = "carton"
    SET = "set"


class StockStatus(str, Enum):
    AVAILABLE = "available"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"
    RECALLED = "recalled"
    DAMAGED = "damaged"


class StockMovementType(str, Enum):
    RECEIPT = "receipt"
    ISSUE = "issue"
    RESERVATION = "reservation"
    RESERVATION_RELEASE = "reservation_release"
    TRANSFER_OUT = "transfer_out"
    TRANSFER_IN = "transfer_in"
    ADJUSTMENT_IN = "adjustment_in"
    ADJUSTMENT_OUT = "adjustment_out"
    DISPOSAL = "disposal"


class ReservationStatus(str, Enum):
    ACTIVE = "active"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TransferStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    IN_TRANSIT = "in_transit"
    RECEIVED = "received"
    CANCELLED = "cancelled"


class StockSource(str, Enum):
    GOVERNMENT = "government"
    DONATION = "donation"
    NGO = "ngo"
    PURCHASE = "purchase"
    TRANSFER = "transfer"
    OTHER = "other"


# ============================================================
# EXCEPTIONS
# ============================================================

class InventoryServiceError(Exception):
    pass


class InventoryValidationError(InventoryServiceError):
    pass


class InventoryNotFoundError(InventoryServiceError):
    pass


class InsufficientStockError(InventoryServiceError):
    pass


class InventoryConflictError(InventoryServiceError):
    pass


class InventoryAuthorizationError(InventoryServiceError):
    pass


class InventoryExpiredError(InventoryServiceError):
    pass


class InventoryIntegrationError(InventoryServiceError):
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
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise InventoryValidationError(
            f"{name} must be timezone-aware"
        )


def validate_quantity(
    value: int,
    *,
    allow_zero: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InventoryValidationError(
            "Quantity must be an integer"
        )

    if allow_zero:
        valid = value >= 0
    else:
        valid = value > 0

    if not valid:
        raise InventoryValidationError(
            "Quantity must be positive"
            if not allow_zero
            else "Quantity cannot be negative"
        )

    return value


def require_text(
    value: str,
    name: str,
) -> str:
    if not value or not value.strip():
        raise InventoryValidationError(
            f"{name} is required"
        )

    return value.strip()


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class InventoryCreateRequest:
    district_id: int
    warehouse_id: str

    item_name: str
    category: InventoryCategory
    unit: InventoryUnit

    sku: str
    batch_number: str

    initial_quantity: int

    minimum_stock_level: int = 0
    reorder_level: int = 0

    source: StockSource = StockSource.OTHER

    supplier_reference: str | None = None
    donation_reference: str | None = None

    manufacture_date: datetime | None = None
    expiry_date: datetime | None = None

    requires_cold_storage: bool = False

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class InventoryItem:
    id: str
    district_id: int
    warehouse_id: str

    item_name: str
    category: InventoryCategory
    unit: InventoryUnit

    sku: str
    batch_number: str

    quantity_on_hand: int
    quantity_reserved: int

    minimum_stock_level: int
    reorder_level: int

    source: StockSource

    supplier_reference: str | None = None
    donation_reference: str | None = None

    manufacture_date: datetime | None = None
    expiry_date: datetime | None = None

    requires_cold_storage: bool = False

    quarantined: bool = False
    recalled: bool = False
    damaged: bool = False

    created_by: int | None = None

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

    @property
    def available_quantity(self) -> int:
        return max(
            0,
            self.quantity_on_hand
            - self.quantity_reserved,
        )

    @property
    def is_expired(self) -> bool:
        return (
            self.expiry_date is not None
            and self.expiry_date <= utc_now()
        )

    @property
    def is_distributable(self) -> bool:
        return not (
            self.is_expired
            or self.quarantined
            or self.recalled
            or self.damaged
        )


@dataclass(frozen=True)
class StockMovement:
    id: str
    inventory_id: str
    district_id: int
    warehouse_id: str

    movement_type: StockMovementType
    quantity: int

    reference_id: str
    performed_by: int

    quantity_before: int
    quantity_after: int

    created_at: datetime = field(
        default_factory=utc_now
    )

    notes: str = ""


@dataclass
class StockReservation:
    id: str
    inventory_id: str

    quantity: int
    reference_id: str

    reserved_by: int

    status: ReservationStatus = (
        ReservationStatus.ACTIVE
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    expires_at: datetime | None = None
    fulfilled_at: datetime | None = None
    cancelled_at: datetime | None = None


@dataclass
class InventoryTransfer:
    id: str

    inventory_id: str
    source_warehouse_id: str
    destination_warehouse_id: str

    district_id: int
    quantity: int

    status: TransferStatus

    requested_by: int
    approved_by: int | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    approved_at: datetime | None = None
    dispatched_at: datetime | None = None
    received_at: datetime | None = None

    notes: str = ""


@dataclass(frozen=True)
class StockAlert:
    inventory_id: str
    district_id: int
    warehouse_id: str

    item_name: str
    status: StockStatus

    quantity_on_hand: int
    available_quantity: int

    minimum_stock_level: int
    reorder_level: int

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class StockAllocation:
    inventory_id: str
    batch_number: str
    quantity: int
    expiry_date: datetime | None


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class InventoryRepository(Protocol):

    async def create(
        self,
        item: InventoryItem,
    ) -> InventoryItem:
        ...

    async def get_by_id(
        self,
        inventory_id: str,
    ) -> InventoryItem | None:
        ...

    async def update(
        self,
        item: InventoryItem,
        *,
        expected_version: int,
    ) -> InventoryItem:
        """
        Must enforce optimistic concurrency.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[InventoryItem]:
        ...

    async def list_by_warehouse(
        self,
        warehouse_id: str,
    ) -> Sequence[InventoryItem]:
        ...

    async def list_by_sku(
        self,
        *,
        district_id: int,
        sku: str,
    ) -> Sequence[InventoryItem]:
        ...

    async def find_batch(
        self,
        *,
        warehouse_id: str,
        sku: str,
        batch_number: str,
    ) -> InventoryItem | None:
        ...

    async def reserve_stock(
        self,
        *,
        reservation: StockReservation,
    ) -> bool:
        """
        Atomically check available stock,
        increment reserved quantity,
        and create the reservation.

        Must enforce unique reservation IDs
        and idempotency for reference IDs.
        """
        ...

    async def fulfill_reservation(
        self,
        *,
        reservation_id: str,
        actor_id: int,
    ) -> bool:
        """
        Atomically decrement on-hand and reserved
        quantities, mark the reservation fulfilled,
        and record the stock movement.
        """
        ...

    async def cancel_reservation(
        self,
        *,
        reservation_id: str,
        actor_id: int,
    ) -> bool:
        """
        Atomically release reserved quantity
        and update the reservation status.
        """
        ...


class ReservationRepository(Protocol):

    async def get_by_id(
        self,
        reservation_id: str,
    ) -> StockReservation | None:
        ...

    async def find_by_reference(
        self,
        *,
        inventory_id: str,
        reference_id: str,
    ) -> StockReservation | None:
        ...

    async def list_expired(
        self,
        now: datetime,
    ) -> Sequence[StockReservation]:
        ...


class StockMovementRepository(Protocol):

    async def create(
        self,
        movement: StockMovement,
    ) -> StockMovement:
        ...

    async def list_by_inventory(
        self,
        inventory_id: str,
    ) -> Sequence[StockMovement]:
        ...


class InventoryTransferRepository(Protocol):

    async def create(
        self,
        transfer: InventoryTransfer,
    ) -> InventoryTransfer:
        ...

    async def get_by_id(
        self,
        transfer_id: str,
    ) -> InventoryTransfer | None:
        ...

    async def update(
        self,
        transfer: InventoryTransfer,
    ) -> InventoryTransfer:
        ...


class InventoryAuthorizationGateway(Protocol):

    async def can_manage(
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

    async def can_distribute(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class InventoryAuditGateway(Protocol):

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
# STOCK STATUS EVALUATOR
# ============================================================

class StockStatusEvaluator:

    @staticmethod
    def evaluate(
        item: InventoryItem,
    ) -> StockStatus:

        if item.recalled:
            return StockStatus.RECALLED

        if item.quarantined:
            return StockStatus.QUARANTINED

        if item.damaged:
            return StockStatus.DAMAGED

        if item.is_expired:
            return StockStatus.EXPIRED

        if item.available_quantity == 0:
            return StockStatus.OUT_OF_STOCK

        threshold = max(
            item.minimum_stock_level,
            item.reorder_level,
        )

        if item.available_quantity <= threshold:
            return StockStatus.LOW_STOCK

        return StockStatus.AVAILABLE


# ============================================================
# FEFO ALLOCATION
# ============================================================

class InventoryAllocator:
    """
    First-Expire-First-Out allocation.

    Items without expiry dates are considered
    after dated inventory.

    Allocation is advisory. Reservations must
    be committed atomically in the database.
    """

    def allocate(
        self,
        items: Sequence[InventoryItem],
        *,
        quantity: int,
    ) -> list[StockAllocation]:

        validate_quantity(quantity)

        eligible = [
            item
            for item in items
            if (
                item.is_distributable
                and item.available_quantity > 0
            )
        ]

        eligible.sort(
            key=lambda item: (
                item.expiry_date is None,
                item.expiry_date
                or datetime.max.replace(
                    tzinfo=timezone.utc
                ),
                item.created_at,
                item.id,
            )
        )

        remaining = quantity
        allocations: list[StockAllocation] = []

        for item in eligible:
            if remaining <= 0:
                break

            allocated = min(
                remaining,
                item.available_quantity,
            )

            allocations.append(
                StockAllocation(
                    inventory_id=item.id,
                    batch_number=item.batch_number,
                    quantity=allocated,
                    expiry_date=item.expiry_date,
                )
            )

            remaining -= allocated

        if remaining > 0:
            raise InsufficientStockError(
                f"Insufficient distributable stock: "
                f"{remaining} units unavailable"
            )

        return allocations


# ============================================================
# INVENTORY SERVICE
# ============================================================

class InventoryService:

    def __init__(
        self,
        *,
        repository: InventoryRepository,
        reservation_repository: ReservationRepository,
        movement_repository: StockMovementRepository,
        authorization_gateway: InventoryAuthorizationGateway,
        audit_gateway: InventoryAuditGateway | None = None,
        allocator: InventoryAllocator | None = None,
        status_evaluator: StockStatusEvaluator | None = None,
    ):
        self.repository = repository
        self.reservations = reservation_repository
        self.movements = movement_repository
        self.authorization = authorization_gateway
        self.audit = audit_gateway

        self.allocator = (
            allocator or InventoryAllocator()
        )

        self.status_evaluator = (
            status_evaluator
            or StockStatusEvaluator()
        )

    async def _audit(
        self,
        action: str,
        item: InventoryItem,
        actor_id: int | None,
        *,
        reference_id: str | None = None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="inventory_item",
            entity_id=item.id,
            metadata={
                "district_id": item.district_id,
                "warehouse_id": item.warehouse_id,
                "sku": item.sku,
                "batch_number": item.batch_number,
                "reference_id": reference_id,
            },
        )

    async def _authorize_manage(
        self,
        actor_id: int,
        district_id: int,
    ) -> None:

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise InventoryAuthorizationError(
                "Inventory management permission denied"
            )

    async def _authorize_distribute(
        self,
        actor_id: int,
        district_id: int,
    ) -> None:

        if not await self.authorization.can_distribute(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise InventoryAuthorizationError(
                "Inventory distribution permission denied"
            )

    async def get_item(
        self,
        inventory_id: str,
    ) -> InventoryItem:

        item = await self.repository.get_by_id(
            inventory_id
        )

        if item is None:
            raise InventoryNotFoundError(
                "Inventory item not found"
            )

        return item

    def _validate_create_request(
        self,
        request: InventoryCreateRequest,
    ) -> None:

        if request.district_id <= 0:
            raise InventoryValidationError(
                "Invalid district ID"
            )

        require_text(
            request.warehouse_id,
            "Warehouse ID",
        )

        require_text(
            request.item_name,
            "Item name",
        )

        require_text(
            request.sku,
            "SKU",
        )

        require_text(
            request.batch_number,
            "Batch number",
        )

        validate_quantity(
            request.initial_quantity,
            allow_zero=True,
        )

        validate_quantity(
            request.minimum_stock_level,
            allow_zero=True,
        )

        validate_quantity(
            request.reorder_level,
            allow_zero=True,
        )

        if request.manufacture_date is not None:
            validate_datetime(
                request.manufacture_date,
                "manufacture_date",
            )

        if request.expiry_date is not None:
            validate_datetime(
                request.expiry_date,
                "expiry_date",
            )

            if request.expiry_date <= utc_now():
                raise InventoryExpiredError(
                    "Cannot register expired stock as available"
                )

        if (
            request.manufacture_date is not None
            and request.expiry_date is not None
            and request.manufacture_date
            >= request.expiry_date
        ):
            raise InventoryValidationError(
                "Manufacture date must precede expiry date"
            )

    async def create_item(
        self,
        request: InventoryCreateRequest,
        *,
        actor_id: int,
    ) -> InventoryItem:

        self._validate_create_request(
            request
        )

        await self._authorize_manage(
            actor_id,
            request.district_id,
        )

        existing = await self.repository.find_batch(
            warehouse_id=request.warehouse_id,
            sku=request.sku,
            batch_number=request.batch_number,
        )

        if existing is not None:
            raise InventoryConflictError(
                "Inventory batch already exists"
            )

        item = InventoryItem(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            warehouse_id=request.warehouse_id,
            item_name=request.item_name.strip(),
            category=request.category,
            unit=request.unit,
            sku=request.sku.strip(),
            batch_number=request.batch_number.strip(),
            quantity_on_hand=request.initial_quantity,
            quantity_reserved=0,
            minimum_stock_level=(
                request.minimum_stock_level
            ),
            reorder_level=request.reorder_level,
            source=request.source,
            supplier_reference=(
                request.supplier_reference
            ),
            donation_reference=(
                request.donation_reference
            ),
            manufacture_date=(
                request.manufacture_date
            ),
            expiry_date=request.expiry_date,
            requires_cold_storage=(
                request.requires_cold_storage
            ),
            created_by=actor_id,
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            item
        )

        if request.initial_quantity > 0:
            await self.movements.create(
                StockMovement(
                    id=str(uuid.uuid4()),
                    inventory_id=created.id,
                    district_id=created.district_id,
                    warehouse_id=created.warehouse_id,
                    movement_type=(
                        StockMovementType.RECEIPT
                    ),
                    quantity=request.initial_quantity,
                    reference_id=created.id,
                    performed_by=actor_id,
                    quantity_before=0,
                    quantity_after=(
                        request.initial_quantity
                    ),
                    notes="Initial inventory receipt",
                )
            )

        await self._audit(
            "inventory.created",
            created,
            actor_id,
        )

        logger.info(
            "Inventory item created: %s",
            created.id,
        )

        return created

    async def receive_stock(
        self,
        inventory_id: str,
        *,
        quantity: int,
        reference_id: str,
        actor_id: int,
        notes: str = "",
    ) -> InventoryItem:

        validate_quantity(quantity)
        require_text(reference_id, "Reference ID")

        item = await self.get_item(
            inventory_id
        )

        await self._authorize_manage(
            actor_id,
            item.district_id,
        )

        if not item.is_distributable:
            raise InventoryConflictError(
                "Stock is expired, recalled, damaged "
                "or quarantined"
            )

        before = item.quantity_on_hand

        updated = replace(
            item,
            quantity_on_hand=before + quantity,
            updated_at=utc_now(),
            version=item.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=item.version,
        )

        await self.movements.create(
            StockMovement(
                id=str(uuid.uuid4()),
                inventory_id=item.id,
                district_id=item.district_id,
                warehouse_id=item.warehouse_id,
                movement_type=StockMovementType.RECEIPT,
                quantity=quantity,
                reference_id=reference_id,
                performed_by=actor_id,
                quantity_before=before,
                quantity_after=saved.quantity_on_hand,
                notes=notes,
            )
        )

        await self._audit(
            "inventory.stock_received",
            saved,
            actor_id,
            reference_id=reference_id,
        )

        return saved

    async def reserve_stock(
        self,
        inventory_id: str,
        *,
        quantity: int,
        reference_id: str,
        actor_id: int,
        expires_in_minutes: int = 120,
    ) -> StockReservation:

        validate_quantity(quantity)
        require_text(reference_id, "Reference ID")

        if expires_in_minutes <= 0:
            raise InventoryValidationError(
                "Reservation duration must be positive"
            )

        item = await self.get_item(
            inventory_id
        )

        await self._authorize_distribute(
            actor_id,
            item.district_id,
        )

        existing = (
            await self.reservations.find_by_reference(
                inventory_id=inventory_id,
                reference_id=reference_id,
            )
        )

        if existing is not None:
            if (
                existing.quantity != quantity
                or existing.reserved_by != actor_id
            ):
                raise InventoryConflictError(
                    "Reference ID already used for "
                    "a different reservation"
                )

            return existing

        if not item.is_distributable:
            raise InventoryConflictError(
                "Inventory batch is not distributable"
            )

        if item.available_quantity < quantity:
            raise InsufficientStockError(
                "Insufficient available inventory"
            )

        reservation = StockReservation(
            id=str(uuid.uuid4()),
            inventory_id=item.id,
            quantity=quantity,
            reference_id=reference_id,
            reserved_by=actor_id,
            expires_at=utc_now() + timedelta(
                minutes=expires_in_minutes
            ),
        )

        reserved = await self.repository.reserve_stock(
            reservation=reservation
        )

        if not reserved:
            raise InsufficientStockError(
                "Atomic stock reservation failed"
            )

        await self._audit(
            "inventory.stock_reserved",
            item,
            actor_id,
            reference_id=reference_id,
        )

        return reservation

    async def fulfill_reservation(
        self,
        reservation_id: str,
        *,
        actor_id: int,
    ) -> StockReservation:

        reservation = await self.reservations.get_by_id(
            reservation_id
        )

        if reservation is None:
            raise InventoryNotFoundError(
                "Stock reservation not found"
            )

        item = await self.get_item(
            reservation.inventory_id
        )

        await self._authorize_distribute(
            actor_id,
            item.district_id,
        )

        if reservation.status == ReservationStatus.FULFILLED:
            return reservation

        if reservation.status != ReservationStatus.ACTIVE:
            raise InventoryConflictError(
                "Reservation is not active"
            )

        if (
            reservation.expires_at is not None
            and reservation.expires_at <= utc_now()
        ):
            raise InventoryConflictError(
                "Reservation has expired"
            )

        if not item.is_distributable:
            raise InventoryConflictError(
                "Stock is not distributable"
            )

        fulfilled = (
            await self.repository.fulfill_reservation(
                reservation_id=reservation_id,
                actor_id=actor_id,
            )
        )

        if not fulfilled:
            raise InventoryConflictError(
                "Atomic reservation fulfillment failed"
            )

        updated_reservation = (
            await self.reservations.get_by_id(
                reservation_id
            )
        )

        if updated_reservation is None:
            raise InventoryIntegrationError(
                "Fulfilled reservation could not be loaded"
            )

        await self._audit(
            "inventory.reservation_fulfilled",
            item,
            actor_id,
            reference_id=reservation.reference_id,
        )

        return updated_reservation

    async def cancel_reservation(
        self,
        reservation_id: str,
        *,
        actor_id: int,
    ) -> StockReservation:

        reservation = await self.reservations.get_by_id(
            reservation_id
        )

        if reservation is None:
            raise InventoryNotFoundError(
                "Reservation not found"
            )

        item = await self.get_item(
            reservation.inventory_id
        )

        await self._authorize_manage(
            actor_id,
            item.district_id,
        )

        if reservation.status == ReservationStatus.CANCELLED:
            return reservation

        if reservation.status != ReservationStatus.ACTIVE:
            raise InventoryConflictError(
                "Reservation cannot be cancelled"
            )

        cancelled = (
            await self.repository.cancel_reservation(
                reservation_id=reservation_id,
                actor_id=actor_id,
            )
        )

        if not cancelled:
            raise InventoryConflictError(
                "Reservation cancellation failed"
            )

        updated = await self.reservations.get_by_id(
            reservation_id
        )

        if updated is None:
            raise InventoryIntegrationError(
                "Cancelled reservation could not be loaded"
            )

        await self._audit(
            "inventory.reservation_cancelled",
            item,
            actor_id,
            reference_id=reservation.reference_id,
        )

        return updated

    async def allocate_stock(
        self,
        *,
        district_id: int,
        sku: str,
        quantity: int,
        actor_id: int,
    ) -> list[StockAllocation]:

        validate_quantity(quantity)

        await self._authorize_distribute(
            actor_id,
            district_id,
        )

        items = await self.repository.list_by_sku(
            district_id=district_id,
            sku=sku,
        )

        return self.allocator.allocate(
            items,
            quantity=quantity,
        )

    async def get_stock_status(
        self,
        inventory_id: str,
    ) -> StockStatus:

        item = await self.get_item(
            inventory_id
        )

        return self.status_evaluator.evaluate(
            item
        )

    async def get_low_stock_alerts(
        self,
        district_id: int,
    ) -> list[StockAlert]:

        items = await self.repository.list_by_district(
            district_id
        )

        alerts: list[StockAlert] = []

        for item in items:
            status = self.status_evaluator.evaluate(
                item
            )

            if status == StockStatus.AVAILABLE:
                continue

            alerts.append(
                StockAlert(
                    inventory_id=item.id,
                    district_id=item.district_id,
                    warehouse_id=item.warehouse_id,
                    item_name=item.item_name,
                    status=status,
                    quantity_on_hand=item.quantity_on_hand,
                    available_quantity=(
                        item.available_quantity
                    ),
                    minimum_stock_level=(
                        item.minimum_stock_level
                    ),
                    reorder_level=item.reorder_level,
                )
            )

        return alerts

    async def list_district_inventory(
        self,
        district_id: int,
    ) -> Sequence[InventoryItem]:

        if district_id <= 0:
            raise InventoryValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_warehouse_inventory(
        self,
        warehouse_id: str,
    ) -> Sequence[InventoryItem]:

        require_text(
            warehouse_id,
            "Warehouse ID",
        )

        return await self.repository.list_by_warehouse(
            warehouse_id
        )

    async def get_inventory_movements(
        self,
        inventory_id: str,
    ) -> Sequence[StockMovement]:

        await self.get_item(
            inventory_id
        )

        return await self.movements.list_by_inventory(
            inventory_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        items = await self.list_district_inventory(
            district_id
        )

        category_counts = {
            category.value: 0
            for category in InventoryCategory
        }

        status_counts = {
            status.value: 0
            for status in StockStatus
        }

        for item in items:
            category_counts[
                item.category.value
            ] += 1

            status = self.status_evaluator.evaluate(
                item
            )

            status_counts[
                status.value
            ] += 1

        return {
            "district_id": district_id,
            "total_inventory_batches": len(items),
            "total_quantity_on_hand": sum(
                item.quantity_on_hand
                for item in items
            ),
            "total_reserved_quantity": sum(
                item.quantity_reserved
                for item in items
            ),
            "total_available_quantity": sum(
                item.available_quantity
                for item in items
                if item.is_distributable
            ),
            "category_counts": category_counts,
            "status_counts": status_counts,
            "generated_at": utc_now().isoformat(),
            "note": (
                "Quantities across different units "
                "are not directly comparable."
            ),
        }

    async def generate_inventory_report(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        items = await self.list_district_inventory(
            district_id
        )

        alerts = await self.get_low_stock_alerts(
            district_id
        )

        return {
            "report_id": str(uuid.uuid4()),
            "district_id": district_id,
            "generated_at": utc_now().isoformat(),
            "inventory": [
                {
                    "inventory_id": item.id,
                    "warehouse_id": item.warehouse_id,
                    "item_name": item.item_name,
                    "sku": item.sku,
                    "batch_number": item.batch_number,
                    "category": item.category.value,
                    "unit": item.unit.value,
                    "quantity_on_hand": (
                        item.quantity_on_hand
                    ),
                    "quantity_reserved": (
                        item.quantity_reserved
                    ),
                    "available_quantity": (
                        item.available_quantity
                        if item.is_distributable
                        else 0
                    ),
                    "status": (
                        self.status_evaluator
                        .evaluate(item)
                        .value
                    ),
                    "expiry_date": (
                        item.expiry_date.isoformat()
                        if item.expiry_date
                        else None
                    ),
                }
                for item in items
            ],
            "alerts": [
                {
                    "inventory_id": alert.inventory_id,
                    "item_name": alert.item_name,
                    "status": alert.status.value,
                    "available_quantity": (
                        alert.available_quantity
                    ),
                }
                for alert in alerts
            ],
            "disclaimer": (
                "Inventory availability depends on "
                "verified warehouse records, stock "
                "quality, reservations and physical "
                "reconciliation. Medical supplies "
                "require appropriate storage and "
                "authorized distribution."
            ),
        }
