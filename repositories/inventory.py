from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/repositories/inventory.py

Async inventory repository using SQLAlchemy 2.x.

Features:
- Inventory creation and retrieval
- Warehouse and shelter stock tracking
- Stock receipt and issue
- Atomic stock reservations
- Reservation release and fulfillment
- Warehouse-to-warehouse transfers
- Stock adjustments
- Low-stock and shortage detection
- Expiry tracking
- Inventory transaction history
- District-based inventory filtering
- Row-level locking for concurrency safety
- Pagination and aggregate statistics

Expected ORM models:
    backend.models.inventory.Inventory
    backend.models.inventory.InventoryMovement
    backend.models.inventory.InventoryReservation

Expected Inventory columns:
    id
    item_name
    category
    sku
    warehouse_id
    shelter_id
    district
    quantity_available
    quantity_reserved
    minimum_stock
    unit
    expiry_date
    created_at
    updated_at
    is_deleted

Expected InventoryMovement columns:
    id
    inventory_id
    movement_type
    quantity
    reference_id
    notes
    performed_by
    created_at

Expected InventoryReservation columns:
    id
    inventory_id
    quantity
    reference_id
    status
    created_at
    updated_at

Important:
- ORM models must be implemented separately.
- Inventory writes must run inside database transactions.
- Never report a stock movement as completed before commit.
- Never allocate more stock than is available.
- AI allocation suggestions do not reserve physical stock.
- All stock adjustments must be authorized and audited.
"""


import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import (
    and_,
    func,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.inventory import (
    Inventory,
    InventoryMovement,
    InventoryReservation,
)


logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

KERALA_DISTRICTS = {
    "thiruvananthapuram",
    "kollam",
    "pathanamthitta",
    "alappuzha",
    "kottayam",
    "idukki",
    "ernakulam",
    "thrissur",
    "palakkad",
    "malappuram",
    "kozhikode",
    "wayanad",
    "kannur",
    "kasaragod",
}


class InventoryError(Exception):
    """Base inventory exception."""


class InventoryNotFoundError(InventoryError):
    """Inventory record was not found."""


class InsufficientStockError(InventoryError):
    """Insufficient unreserved stock."""


class InvalidStockOperationError(InventoryError):
    """Invalid inventory transaction."""


class InventoryConflictError(InventoryError):
    """Conflicting or duplicate inventory operation."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_district(value: str) -> str:
    district = value.strip().lower()

    if district not in KERALA_DISTRICTS:
        raise ValueError(
            f"Invalid Kerala district: {value}"
        )

    return district


def to_quantity(
    value: Decimal | int | float | str,
    *,
    allow_zero: bool = False,
) -> Decimal:
    """
    Normalize a stock quantity to Decimal.

    Floats are converted using their string
    representation to avoid binary artifacts.
    """

    try:
        quantity = Decimal(str(value))
    except Exception as exc:
        raise ValueError(
            "Invalid inventory quantity"
        ) from exc

    if not quantity.is_finite():
        raise ValueError(
            "Quantity must be finite"
        )

    if quantity < 0:
        raise ValueError(
            "Quantity cannot be negative"
        )

    if not allow_zero and quantity == 0:
        raise ValueError(
            "Quantity must be greater than zero"
        )

    return quantity


def validate_pagination(
    limit: int,
    offset: int,
) -> None:
    if not 1 <= limit <= 500:
        raise ValueError(
            "limit must be between 1 and 500"
        )

    if offset < 0:
        raise ValueError(
            "offset cannot be negative"
        )


# ============================================================
# INVENTORY REPOSITORY
# ============================================================

class InventoryRepository:
    """
    Inventory persistence and stock operations.

    Transaction ownership:
        The calling service controls commit/rollback.

    Example:

        async with transaction() as session:
            repository = InventoryRepository(session)

            inventory = await repository.receive_stock(
                inventory_id=inventory_id,
                quantity=100,
                reference_id="DONATION-123",
                performed_by=user_id,
            )

    Important:
        Never share one AsyncSession across
        concurrent requests or worker tasks.
    """

    ACTIVE_RESERVATION_STATUS = "active"

    RESERVATION_FULFILLED = "fulfilled"

    RESERVATION_RELEASED = "released"

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _not_deleted():
        return Inventory.is_deleted.is_(False)

    @staticmethod
    def _free_stock(
        inventory: Inventory,
    ) -> Decimal:
        available = to_quantity(
            inventory.quantity_available,
            allow_zero=True,
        )

        reserved = to_quantity(
            inventory.quantity_reserved,
            allow_zero=True,
        )

        if reserved > available:
            raise InventoryConflictError(
                "Reserved stock exceeds physical stock"
            )

        return available - reserved

    async def _get_locked(
        self,
        inventory_id: UUID,
    ) -> Inventory:
        """
        Lock an inventory row until transaction completion.

        Requires PostgreSQL transaction semantics.
        """

        statement = (
            select(Inventory)
            .where(
                Inventory.id == inventory_id,
                self._not_deleted(),
            )
            .with_for_update()
        )

        result = await self.session.execute(
            statement
        )

        inventory = result.scalar_one_or_none()

        if inventory is None:
            raise InventoryNotFoundError(
                f"Inventory {inventory_id} not found"
            )

        return inventory

    async def _record_movement(
        self,
        *,
        inventory_id: UUID,
        movement_type: str,
        quantity: Decimal,
        reference_id: str,
        performed_by: UUID,
        notes: str | None = None,
    ) -> InventoryMovement:

        movement = InventoryMovement(
            inventory_id=inventory_id,
            movement_type=movement_type,
            quantity=quantity,
            reference_id=reference_id,
            performed_by=performed_by,
            notes=notes,
        )

        self.session.add(movement)

        await self.session.flush()

        return movement

    # ========================================================
    # CREATE INVENTORY
    # ========================================================

    async def create(
        self,
        *,
        item_name: str,
        category: str,
        sku: str,
        district: str,
        unit: str,
        warehouse_id: UUID | None = None,
        shelter_id: UUID | None = None,
        minimum_stock: Decimal | int | float = 0,
        expiry_date: date | None = None,
    ) -> Inventory:
        """
        Create an empty inventory record.

        Initial stock must be received through
        receive_stock() to preserve movement history.
        """

        if not item_name.strip():
            raise ValueError(
                "Item name is required"
            )

        if not sku.strip():
            raise ValueError(
                "SKU is required"
            )

        if not unit.strip():
            raise ValueError(
                "Unit is required"
            )

        if (
            warehouse_id is None
        ) == (
            shelter_id is None
        ):
            raise ValueError(
                "Exactly one warehouse_id or shelter_id "
                "must be supplied"
            )

        inventory = Inventory(
            item_name=item_name.strip(),
            category=category,
            sku=sku.strip(),
            district=normalize_district(district),
            unit=unit.strip(),
            warehouse_id=warehouse_id,
            shelter_id=shelter_id,
            quantity_available=Decimal("0"),
            quantity_reserved=Decimal("0"),
            minimum_stock=to_quantity(
                minimum_stock,
                allow_zero=True,
            ),
            expiry_date=expiry_date,
            is_deleted=False,
        )

        self.session.add(inventory)

        await self.session.flush()

        return inventory

    # ========================================================
    # GET INVENTORY
    # ========================================================

    async def get_by_id(
        self,
        inventory_id: UUID,
    ) -> Inventory | None:

        statement = select(Inventory).where(
            Inventory.id == inventory_id,
            self._not_deleted(),
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # LIST INVENTORY
    # ========================================================

    async def list_inventory(
        self,
        *,
        district: str | None = None,
        warehouse_id: UUID | None = None,
        shelter_id: UUID | None = None,
        category: str | None = None,
        sku: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Inventory]:

        validate_pagination(limit, offset)

        statement = select(Inventory).where(
            self._not_deleted()
        )

        if district is not None:
            statement = statement.where(
                Inventory.district
                == normalize_district(district)
            )

        if warehouse_id is not None:
            statement = statement.where(
                Inventory.warehouse_id == warehouse_id
            )

        if shelter_id is not None:
            statement = statement.where(
                Inventory.shelter_id == shelter_id
            )

        if category is not None:
            statement = statement.where(
                Inventory.category == category
            )

        if sku is not None:
            statement = statement.where(
                Inventory.sku == sku
            )

        statement = (
            statement
            .order_by(
                Inventory.created_at.desc(),
                Inventory.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # RECEIVE STOCK
    # ========================================================

    async def receive_stock(
        self,
        *,
        inventory_id: UUID,
        quantity: Decimal | int | float,
        reference_id: str,
        performed_by: UUID,
        notes: str | None = None,
    ) -> Inventory:
        """
        Receive stock into an inventory location.

        Uses row-level locking.
        """

        amount = to_quantity(quantity)

        inventory = await self._get_locked(
            inventory_id
        )

        inventory.quantity_available = (
            to_quantity(
                inventory.quantity_available,
                allow_zero=True,
            )
            + amount
        )

        await self._record_movement(
            inventory_id=inventory.id,
            movement_type="stock_in",
            quantity=amount,
            reference_id=reference_id,
            performed_by=performed_by,
            notes=notes,
        )

        await self.session.flush()

        return inventory

    # ========================================================
    # ISSUE STOCK
    # ========================================================

    async def issue_stock(
        self,
        *,
        inventory_id: UUID,
        quantity: Decimal | int | float,
        reference_id: str,
        performed_by: UUID,
        notes: str | None = None,
    ) -> Inventory:
        """
        Issue unreserved inventory stock.

        Does not permit issuing stock held by
        active reservations.
        """

        amount = to_quantity(quantity)

        inventory = await self._get_locked(
            inventory_id
        )

        free_stock = self._free_stock(
            inventory
        )

        if amount > free_stock:
            raise InsufficientStockError(
                "Insufficient unreserved inventory"
            )

        inventory.quantity_available = (
            to_quantity(
                inventory.quantity_available,
                allow_zero=True,
            )
            - amount
        )

        await self._record_movement(
            inventory_id=inventory.id,
            movement_type="stock_out",
            quantity=amount,
            reference_id=reference_id,
            performed_by=performed_by,
            notes=notes,
        )

        await self.session.flush()

        return inventory

    # ========================================================
    # RESERVE STOCK
    # ========================================================

    async def reserve_stock(
        self,
        *,
        inventory_id: UUID,
        quantity: Decimal | int | float,
        reference_id: str,
    ) -> InventoryReservation:
        """
        Reserve stock without physically issuing it.

        Reservation is not delivery confirmation.
        """

        amount = to_quantity(quantity)

        inventory = await self._get_locked(
            inventory_id
        )

        if amount > self._free_stock(inventory):
            raise InsufficientStockError(
                "Not enough stock for reservation"
            )

        inventory.quantity_reserved = (
            to_quantity(
                inventory.quantity_reserved,
                allow_zero=True,
            )
            + amount
        )

        reservation = InventoryReservation(
            inventory_id=inventory_id,
            quantity=amount,
            reference_id=reference_id,
            status=self.ACTIVE_RESERVATION_STATUS,
        )

        self.session.add(reservation)

        await self.session.flush()

        return reservation

    # ========================================================
    # RELEASE RESERVATION
    # ========================================================

    async def release_reservation(
        self,
        *,
        reservation_id: UUID,
    ) -> InventoryReservation:
        """
        Release reserved stock.

        Lock inventory before reservation to preserve
        consistent lock ordering.
        """

        lookup = await self.session.execute(
            select(
                InventoryReservation.inventory_id
            ).where(
                InventoryReservation.id
                == reservation_id
            )
        )

        inventory_id = lookup.scalar_one_or_none()

        if inventory_id is None:
            raise InventoryNotFoundError(
                "Reservation not found"
            )

        inventory = await self._get_locked(
            inventory_id
        )

        result = await self.session.execute(
            select(InventoryReservation)
            .where(
                InventoryReservation.id
                == reservation_id
            )
            .with_for_update()
        )

        reservation = result.scalar_one_or_none()

        if reservation is None:
            raise InventoryNotFoundError(
                "Reservation not found"
            )

        if (
            reservation.status
            != self.ACTIVE_RESERVATION_STATUS
        ):
            raise InvalidStockOperationError(
                "Reservation is not active"
            )

        amount = to_quantity(
            reservation.quantity
        )

        reserved = to_quantity(
            inventory.quantity_reserved,
            allow_zero=True,
        )

        if amount > reserved:
            raise InventoryConflictError(
                "Reservation exceeds reserved stock"
            )

        inventory.quantity_reserved = (
            reserved - amount
        )

        reservation.status = (
            self.RESERVATION_RELEASED
        )

        await self.session.flush()

        return reservation

    # ========================================================
    # FULFILL RESERVATION
    # ========================================================

    async def fulfill_reservation(
        self,
        *,
        reservation_id: UUID,
        reference_id: str,
        performed_by: UUID,
    ) -> InventoryReservation:
        """
        Convert a reservation into an actual stock issue.

        This confirms warehouse stock release,
        not delivery to beneficiaries.
        """

        lookup = await self.session.execute(
            select(
                InventoryReservation.inventory_id
            ).where(
                InventoryReservation.id
                == reservation_id
            )
        )

        inventory_id = lookup.scalar_one_or_none()

        if inventory_id is None:
            raise InventoryNotFoundError(
                "Reservation not found"
            )

        inventory = await self._get_locked(
            inventory_id
        )

        result = await self.session.execute(
            select(InventoryReservation)
            .where(
                InventoryReservation.id
                == reservation_id
            )
            .with_for_update()
        )

        reservation = result.scalar_one_or_none()

        if reservation is None:
            raise InventoryNotFoundError(
                "Reservation not found"
            )

        if (
            reservation.status
            != self.ACTIVE_RESERVATION_STATUS
        ):
            raise InvalidStockOperationError(
                "Reservation is not active"
            )

        amount = to_quantity(
            reservation.quantity
        )

        physical = to_quantity(
            inventory.quantity_available,
            allow_zero=True,
        )

        reserved = to_quantity(
            inventory.quantity_reserved,
            allow_zero=True,
        )

        if amount > physical or amount > reserved:
            raise InventoryConflictError(
                "Inventory reservation balance is inconsistent"
            )

        inventory.quantity_available = (
            physical - amount
        )

        inventory.quantity_reserved = (
            reserved - amount
        )

        reservation.status = (
            self.RESERVATION_FULFILLED
        )

        await self._record_movement(
            inventory_id=inventory.id,
            movement_type="reservation_fulfilled",
            quantity=amount,
            reference_id=reference_id,
            performed_by=performed_by,
        )

        await self.session.flush()

        return reservation

    # ========================================================
    # TRANSFER STOCK
    # ========================================================

    async def transfer_stock(
        self,
        *,
        source_inventory_id: UUID,
        destination_inventory_id: UUID,
        quantity: Decimal | int | float,
        reference_id: str,
        performed_by: UUID,
    ) -> dict[str, Any]:
        """
        Atomically transfer stock between two inventories.

        Locks rows in deterministic UUID order to
        reduce deadlock risk.

        Both inventory records must represent the
        same SKU and unit.

        This operation models an immediate confirmed
        stock transfer. Physical shipments should use
        a separate in-transit workflow.
        """

        if source_inventory_id == destination_inventory_id:
            raise InvalidStockOperationError(
                "Source and destination must differ"
            )

        amount = to_quantity(quantity)

        ids = sorted(
            [
                source_inventory_id,
                destination_inventory_id,
            ],
            key=str,
        )

        locked = {}

        for inventory_id in ids:
            locked[inventory_id] = (
                await self._get_locked(
                    inventory_id
                )
            )

        source = locked[source_inventory_id]
        destination = locked[destination_inventory_id]

        if source.sku != destination.sku:
            raise InvalidStockOperationError(
                "Cannot transfer different SKUs"
            )

        if source.unit != destination.unit:
            raise InvalidStockOperationError(
                "Cannot transfer different units"
            )

        if amount > self._free_stock(source):
            raise InsufficientStockError(
                "Insufficient source inventory"
            )

        source.quantity_available = (
            to_quantity(
                source.quantity_available,
                allow_zero=True,
            )
            - amount
        )

        destination.quantity_available = (
            to_quantity(
                destination.quantity_available,
                allow_zero=True,
            )
            + amount
        )

        outgoing = await self._record_movement(
            inventory_id=source.id,
            movement_type="transfer_out",
            quantity=amount,
            reference_id=reference_id,
            performed_by=performed_by,
        )

        incoming = await self._record_movement(
            inventory_id=destination.id,
            movement_type="transfer_in",
            quantity=amount,
            reference_id=reference_id,
            performed_by=performed_by,
        )

        await self.session.flush()

        return {
            "source_inventory_id": source.id,
            "destination_inventory_id": destination.id,
            "quantity": amount,
            "reference_id": reference_id,
            "outgoing_movement_id": outgoing.id,
            "incoming_movement_id": incoming.id,
        }

    # ========================================================
    # ADJUST STOCK
    # ========================================================

    async def adjust_stock(
        self,
        *,
        inventory_id: UUID,
        new_quantity: Decimal | int | float,
        reason: str,
        reference_id: str,
        performed_by: UUID,
    ) -> Inventory:
        """
        Correct physical stock after verified stocktake.

        This must only be invoked by an authorized
        inventory operator.
        """

        target = to_quantity(
            new_quantity,
            allow_zero=True,
        )

        if not reason.strip():
            raise ValueError(
                "Adjustment reason is required"
            )

        inventory = await self._get_locked(
            inventory_id
        )

        previous = to_quantity(
            inventory.quantity_available,
            allow_zero=True,
        )

        reserved = to_quantity(
            inventory.quantity_reserved,
            allow_zero=True,
        )

        if target < reserved:
            raise InventoryConflictError(
                "Adjustment cannot reduce stock "
                "below active reservations"
            )

        difference = target - previous

        if difference == 0:
            return inventory

        inventory.quantity_available = target

        await self._record_movement(
            inventory_id=inventory.id,
            movement_type=(
                "adjustment_in"
                if difference > 0
                else "adjustment_out"
            ),
            quantity=abs(difference),
            reference_id=reference_id,
            performed_by=performed_by,
            notes=reason,
        )

        await self.session.flush()

        return inventory

    # ========================================================
    # LOW STOCK
    # ========================================================

    async def get_low_stock(
        self,
        *,
        district: str | None = None,
        limit: int = 100,
    ) -> Sequence[Inventory]:

        validate_pagination(limit, 0)

        free_stock = (
            Inventory.quantity_available
            - Inventory.quantity_reserved
        )

        statement = select(Inventory).where(
            self._not_deleted(),
            free_stock <= Inventory.minimum_stock,
        )

        if district is not None:
            statement = statement.where(
                Inventory.district
                == normalize_district(district)
            )

        statement = (
            statement
            .order_by(
                free_stock.asc(),
                Inventory.id.asc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # OUT OF STOCK
    # ========================================================

    async def get_out_of_stock(
        self,
        *,
        district: str | None = None,
        limit: int = 100,
    ) -> Sequence[Inventory]:

        validate_pagination(limit, 0)

        statement = select(Inventory).where(
            self._not_deleted(),
            (
                Inventory.quantity_available
                - Inventory.quantity_reserved
            ) <= 0,
        )

        if district is not None:
            statement = statement.where(
                Inventory.district
                == normalize_district(district)
            )

        statement = statement.limit(limit)

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # EXPIRING INVENTORY
    # ========================================================

    async def get_expiring_inventory(
        self,
        *,
        days: int = 30,
        district: str | None = None,
        limit: int = 100,
    ) -> Sequence[Inventory]:

        if days < 0:
            raise ValueError(
                "days cannot be negative"
            )

        validate_pagination(limit, 0)

        deadline = (
            utc_now().date()
            + timedelta(days=days)
        )

        statement = select(Inventory).where(
            self._not_deleted(),
            Inventory.expiry_date.is_not(None),
            Inventory.expiry_date <= deadline,
        )

        if district is not None:
            statement = statement.where(
                Inventory.district
                == normalize_district(district)
            )

        statement = (
            statement
            .order_by(
                Inventory.expiry_date.asc(),
                Inventory.id.asc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # INVENTORY MOVEMENT HISTORY
    # ========================================================

    async def get_movements(
        self,
        *,
        inventory_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[InventoryMovement]:

        validate_pagination(limit, offset)

        statement = (
            select(InventoryMovement)
            .where(
                InventoryMovement.inventory_id
                == inventory_id
            )
            .order_by(
                InventoryMovement.created_at.desc(),
                InventoryMovement.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # RESERVATION HISTORY
    # ========================================================

    async def get_reservations(
        self,
        *,
        inventory_id: UUID,
        status: str | None = None,
        limit: int = 100,
    ) -> Sequence[InventoryReservation]:

        validate_pagination(limit, 0)

        statement = select(
            InventoryReservation
        ).where(
            InventoryReservation.inventory_id
            == inventory_id
        )

        if status is not None:
            statement = statement.where(
                InventoryReservation.status == status
            )

        statement = (
            statement
            .order_by(
                InventoryReservation.created_at.desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # INVENTORY STATISTICS
    # ========================================================

    async def get_statistics(
        self,
        *,
        district: str | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate stock counts.

        Quantity totals are grouped by unit because
        kilograms, litres and pieces cannot be added
        into one meaningful total.
        """

        statement = select(
            Inventory.unit,
            func.count(Inventory.id).label(
                "item_records"
            ),
            func.sum(
                Inventory.quantity_available
            ).label("physical_quantity"),
            func.sum(
                Inventory.quantity_reserved
            ).label("reserved_quantity"),
            func.sum(
                Inventory.quantity_available
                - Inventory.quantity_reserved
            ).label("free_quantity"),
        ).where(
            self._not_deleted()
        )

        if district is not None:
            statement = statement.where(
                Inventory.district
                == normalize_district(district)
            )

        statement = statement.group_by(
            Inventory.unit
        )

        result = await self.session.execute(
            statement
        )

        units = []

        for row in result.all():
            units.append(
                {
                    "unit": row.unit,
                    "item_records": row.item_records,
                    "physical_quantity": str(
                        row.physical_quantity or 0
                    ),
                    "reserved_quantity": str(
                        row.reserved_quantity or 0
                    ),
                    "free_quantity": str(
                        row.free_quantity or 0
                    ),
                }
            )

        return {
            "district": district,
            "units": units,
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "InventoryRepository",
    "InventoryError",
    "InventoryNotFoundError",
    "InsufficientStockError",
    "InvalidStockOperationError",
    "InventoryConflictError",
    "normalize_district",
    "to_quantity",
    "validate_pagination",
    "utc_now",
]
