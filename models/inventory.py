from __future__ import annotations

"""
backend/models/inventory.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x inventory and relief-supply models.

Features:
- Relief inventory item catalog
- Food, water, medicine and emergency supplies
- District and shelter stock locations
- Batch and expiry tracking
- Inventory stock snapshots
- Stock movement history
- Donations and supply provenance
- Relief supply requests
- Inventory reservations
- Low-stock and expiry alerts
- Human-approved allocation workflows
- Audit and traceability
- Optimistic concurrency control

Safety:
- Inventory records do not prove physical availability.
- Expired, recalled or quarantined supplies must not
  be allocated for distribution.
- AI-generated allocation recommendations are advisory.
- Operational releases require authorization.
- Stock changes must use transactional service logic.
"""


import enum
import math
import uuid

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
)

from sqlalchemy.ext.mutable import MutableDict

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from backend.database.base import Base


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(
    enum_class: type[enum.Enum],
) -> SQLEnum:
    return SQLEnum(
        enum_class,
        values_callable=lambda members: [
            member.value for member in members
        ],
        native_enum=False,
        validate_strings=True,
        length=80,
    )


def require_text(
    value: str,
    field_name: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} is required"
        )

    result = value.strip()

    if len(result) > maximum:
        raise ValueError(
            f"{field_name} exceeds {maximum} characters"
        )

    return result


def validate_quantity(
    value: Decimal | int | float | str | None,
    field_name: str,
    *,
    allow_none: bool = False,
    positive: bool = False,
) -> Decimal | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required")

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(
            f"{field_name} must be numeric"
        ) from exc

    if not result.is_finite():
        raise ValueError(
            f"{field_name} must be finite"
        )

    if positive and result <= 0:
        raise ValueError(
            f"{field_name} must be positive"
        )

    if not positive and result < 0:
        raise ValueError(
            f"{field_name} must be non-negative"
        )

    if result.as_tuple().exponent < -3:
        raise ValueError(
            f"{field_name} supports at most 3 decimals"
        )

    if result >= Decimal("1000000000000"):
        raise ValueError(
            f"{field_name} exceeds storage capacity"
        )

    return result


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class InventoryCategory(str, enum.Enum):
    FOOD = "food"
    DRINKING_WATER = "drinking_water"
    MEDICINE = "medicine"
    FIRST_AID = "first_aid"
    HYGIENE = "hygiene"
    SANITATION = "sanitation"
    CLOTHING = "clothing"
    BLANKETS = "blankets"
    BEDDING = "bedding"
    BABY_CARE = "baby_care"
    MATERNITY_CARE = "maternity_care"
    ELDERLY_CARE = "elderly_care"
    DISABILITY_SUPPORT = "disability_support"
    SHELTER_MATERIAL = "shelter_material"
    RESCUE_EQUIPMENT = "rescue_equipment"
    MEDICAL_EQUIPMENT = "medical_equipment"
    POWER_EQUIPMENT = "power_equipment"
    COMMUNICATION_EQUIPMENT = "communication_equipment"
    CLEANING_SUPPLIES = "cleaning_supplies"
    FUEL = "fuel"
    OTHER = "other"


class InventoryUnit(str, enum.Enum):
    PIECE = "piece"
    PACK = "pack"
    BOX = "box"
    CARTON = "carton"
    KIT = "kit"
    BOTTLE = "bottle"
    BAG = "bag"
    KILOGRAM = "kilogram"
    GRAM = "gram"
    LITRE = "litre"
    MILLILITRE = "millilitre"
    METRE = "metre"
    SET = "set"
    PAIR = "pair"


class InventoryLocationType(str, enum.Enum):
    CENTRAL_WAREHOUSE = "central_warehouse"
    DISTRICT_WAREHOUSE = "district_warehouse"
    TALUK_STORE = "taluk_store"
    RELIEF_CAMP = "relief_camp"
    SHELTER = "shelter"
    HOSPITAL = "hospital"
    DISTRIBUTION_CENTRE = "distribution_centre"
    MOBILE_STORAGE = "mobile_storage"
    OTHER = "other"


class InventoryLocationStatus(str, enum.Enum):
    ACTIVE = "active"
    LIMITED_ACCESS = "limited_access"
    FLOOD_AFFECTED = "flood_affected"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class InventoryBatchStatus(str, enum.Enum):
    AVAILABLE = "available"
    QUARANTINED = "quarantined"
    RECALLED = "recalled"
    EXPIRED = "expired"
    DEPLETED = "depleted"
    DISPOSED = "disposed"


class InventoryMovementType(str, enum.Enum):
    RECEIPT = "receipt"
    DONATION = "donation"
    PURCHASE = "purchase"
    TRANSFER_OUT = "transfer_out"
    TRANSFER_IN = "transfer_in"
    DISTRIBUTION = "distribution"
    RETURN = "return"
    ADJUSTMENT_IN = "adjustment_in"
    ADJUSTMENT_OUT = "adjustment_out"
    DAMAGE = "damage"
    EXPIRY = "expiry"
    DISPOSAL = "disposal"


class InventoryMovementStatus(str, enum.Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    POSTED = "posted"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class InventoryRequestStatus(str, enum.Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PARTIALLY_ALLOCATED = "partially_allocated"
    ALLOCATED = "allocated"
    PARTIALLY_FULFILLED = "partially_fulfilled"
    FULFILLED = "fulfilled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class InventoryPriority(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class InventoryReservationStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    RELEASED = "released"
    FULFILLED = "fulfilled"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class InventoryAlertType(str, enum.Enum):
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    RECALL = "recall"
    DAMAGED_STOCK = "damaged_stock"
    STORAGE_RISK = "storage_risk"
    DISCREPANCY = "discrepancy"


class InventoryAlertStatus(str, enum.Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


# ============================================================
# INVENTORY ITEM CATALOG
# ============================================================

class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    item_code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    name_malayalam: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    category: Mapped[InventoryCategory] = mapped_column(
        enum_type(InventoryCategory),
        nullable=False,
        index=True,
    )

    unit: Mapped[InventoryUnit] = mapped_column(
        enum_type(InventoryUnit),
        nullable=False,
    )

    sku: Mapped[str | None] = mapped_column(
        String(150),
        unique=True,
        nullable=True,
    )

    manufacturer: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    requires_expiry_tracking: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_batch_tracking: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    requires_cold_storage: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_special_handling: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_medical_item: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_perishable: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    minimum_stock_level: Mapped[Decimal] = mapped_column(
        Numeric(15, 3),
        default=Decimal("0"),
        nullable=False,
    )

    reorder_level: Mapped[Decimal] = mapped_column(
        Numeric(15, 3),
        default=Decimal("0"),
        nullable=False,
    )

    estimated_unit_cost_inr: Mapped[
        Decimal | None
    ] = mapped_column(
        Numeric(15, 2),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    batches: Mapped[list["InventoryBatch"]] = relationship(
        "InventoryBatch",
        back_populates="item",
    )

    stocks: Mapped[list["InventoryStock"]] = relationship(
        "InventoryStock",
        back_populates="item",
    )

    __table_args__ = (
        CheckConstraint(
            "minimum_stock_level >= 0",
            name="ck_inventory_item_min_stock",
        ),
        CheckConstraint(
            "reorder_level >= 0",
            name="ck_inventory_item_reorder",
        ),
        CheckConstraint(
            "estimated_unit_cost_inr IS NULL OR "
            "estimated_unit_cost_inr >= 0",
            name="ck_inventory_item_cost",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_inventory_item_version",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("item_code", "name")
    def validate_required(
        self,
        key: str,
        value: str,
    ) -> str:
        return require_text(
            value,
            key,
            100 if key == "item_code" else 255,
        )

    @validates(
        "minimum_stock_level",
        "reorder_level",
        "estimated_unit_cost_inr",
    )
    def validate_amount(
        self,
        key: str,
        value: Any,
    ) -> Decimal | None:
        return validate_quantity(
            value,
            key,
            allow_none=(
                key == "estimated_unit_cost_inr"
            ),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "item_code": self.item_code,
            "name": self.name,
            "category": self.category.value,
            "unit": self.unit.value,
            "requires_expiry_tracking": (
                self.requires_expiry_tracking
            ),
            "requires_cold_storage": (
                self.requires_cold_storage
            ),
            "is_active": self.is_active,
        }


# ============================================================
# INVENTORY STORAGE LOCATION
# ============================================================

class InventoryLocation(Base):
    __tablename__ = "inventory_locations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    location_code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    location_type: Mapped[
        InventoryLocationType
    ] = mapped_column(
        enum_type(InventoryLocationType),
        nullable=False,
    )

    status: Mapped[
        InventoryLocationStatus
    ] = mapped_column(
        enum_type(InventoryLocationStatus),
        default=InventoryLocationStatus.UNKNOWN,
        nullable=False,
    )

    district_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "districts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    shelter_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    hospital_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    address: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    latitude: Mapped[float | None] = mapped_column(
        Numeric(10, 7),
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Numeric(10, 7),
        nullable=True,
    )

    contact_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    manager_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    has_cold_storage: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    last_verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    stocks: Mapped[list["InventoryStock"]] = relationship(
        "InventoryStock",
        back_populates="location",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_inventory_location_district",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_inventory_location_lat",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_inventory_location_lon",
        ),
        Index(
            "ix_inventory_location_district_status",
            "district_id",
            "status",
        ),
    )


# ============================================================
# INVENTORY BATCH
# ============================================================

class InventoryBatch(Base):
    __tablename__ = "inventory_batches"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    item_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_items.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    batch_number: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    lot_reference: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    supplier_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    donation_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    manufacturer: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    manufactured_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    expires_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    status: Mapped[InventoryBatchStatus] = (
        mapped_column(
            enum_type(InventoryBatchStatus),
            default=InventoryBatchStatus.QUARANTINED,
            nullable=False,
            index=True,
        )
    )

    quality_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    quality_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    quality_verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    recall_reference: Mapped[str | None] = (
        mapped_column(
            String(200),
            nullable=True,
        )
    )

    storage_instructions: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    item: Mapped["InventoryItem"] = relationship(
        "InventoryItem",
        back_populates="batches",
    )

    stocks: Mapped[list["InventoryStock"]] = relationship(
        "InventoryStock",
        back_populates="batch",
    )

    __table_args__ = (
        UniqueConstraint(
            "item_id",
            "batch_number",
            name="uq_inventory_item_batch",
        ),
        Index(
            "ix_inventory_batch_expiry",
            "status",
            "expires_at",
        ),
    )

    def is_expired(
        self,
        now: datetime | None = None,
    ) -> bool:
        if self.expires_at is None:
            return False

        return as_utc(self.expires_at) <= (
            now or utc_now()
        )

    @property
    def can_be_distributed(self) -> bool:
        if self.item.requires_expiry_tracking:
            if self.expires_at is None:
                return False

        return (
            self.status == InventoryBatchStatus.AVAILABLE
            and self.quality_verified
            and not self.is_expired()
        )


# ============================================================
# INVENTORY STOCK
# ============================================================

class InventoryStock(Base):
    __tablename__ = "inventory_stocks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    item_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_items.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    batch_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_batches.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    location_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_locations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    quantity_on_hand: Mapped[Decimal] = mapped_column(
        Numeric(15, 3),
        default=Decimal("0"),
        nullable=False,
    )

    quantity_reserved: Mapped[Decimal] = mapped_column(
        Numeric(15, 3),
        default=Decimal("0"),
        nullable=False,
    )

    quantity_quarantined: Mapped[Decimal] = (
        mapped_column(
            Numeric(15, 3),
            default=Decimal("0"),
            nullable=False,
        )
    )

    last_counted_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    last_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    item: Mapped["InventoryItem"] = relationship(
        "InventoryItem",
        back_populates="stocks",
    )

    batch: Mapped["InventoryBatch"] = relationship(
        "InventoryBatch",
        back_populates="stocks",
    )

    location: Mapped["InventoryLocation"] = relationship(
        "InventoryLocation",
        back_populates="stocks",
    )

    __table_args__ = (
        UniqueConstraint(
            "batch_id",
            "location_id",
            name="uq_inventory_batch_location",
        ),
        CheckConstraint(
            "quantity_on_hand >= 0",
            name="ck_inventory_stock_on_hand",
        ),
        CheckConstraint(
            "quantity_reserved >= 0",
            name="ck_inventory_stock_reserved",
        ),
        CheckConstraint(
            "quantity_quarantined >= 0",
            name="ck_inventory_stock_quarantined",
        ),
        CheckConstraint(
            "quantity_reserved + quantity_quarantined "
            "<= quantity_on_hand",
            name="ck_inventory_stock_balance",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_inventory_stock_version",
        ),
        Index(
            "ix_inventory_stock_location_item",
            "location_id",
            "item_id",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "quantity_on_hand",
        "quantity_reserved",
        "quantity_quarantined",
    )
    def validate_stock_quantity(
        self,
        key: str,
        value: Any,
    ) -> Decimal:
        result = validate_quantity(value, key)
        assert result is not None
        return result

    @property
    def available_quantity(self) -> Decimal:
        return max(
            self.quantity_on_hand
            - self.quantity_reserved
            - self.quantity_quarantined,
            Decimal("0"),
        )

    @property
    def is_distributable(self) -> bool:
        return (
            self.item.is_active
            and self.location.is_active
            and self.location.is_verified
            and self.location.status
            == InventoryLocationStatus.ACTIVE
            and self.batch.can_be_distributed
            and (
                not self.item.requires_cold_storage
                or self.location.has_cold_storage
            )
            and self.available_quantity > 0
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "item_id": self.item_id,
            "batch_id": self.batch_id,
            "location_id": self.location_id,
            "quantity_on_hand": str(
                self.quantity_on_hand
            ),
            "quantity_reserved": str(
                self.quantity_reserved
            ),
            "quantity_quarantined": str(
                self.quantity_quarantined
            ),
            "available_quantity": str(
                self.available_quantity
            ),
        }


# ============================================================
# INVENTORY MOVEMENT
# ============================================================

class InventoryMovement(Base):
    __tablename__ = "inventory_movements"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    movement_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    movement_type: Mapped[
        InventoryMovementType
    ] = mapped_column(
        enum_type(InventoryMovementType),
        nullable=False,
    )

    status: Mapped[
        InventoryMovementStatus
    ] = mapped_column(
        enum_type(InventoryMovementStatus),
        default=InventoryMovementStatus.PROPOSED,
        nullable=False,
    )

    item_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_items.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    batch_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_batches.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    source_location_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_locations.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    destination_location_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_locations.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    quantity: Mapped[Decimal] = mapped_column(
        Numeric(15, 3),
        nullable=False,
    )

    reference_type: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    reference_id: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    justification: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    requested_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    posted_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    posted_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    idempotency_key: Mapped[str] = mapped_column(
        String(150),
        unique=True,
        nullable=False,
    )

    audit_reference: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "quantity > 0",
            name="ck_inventory_movement_quantity",
        ),
        CheckConstraint(
            "source_location_id IS NULL OR "
            "destination_location_id IS NULL OR "
            "source_location_id != destination_location_id",
            name="ck_inventory_movement_locations",
        ),
        Index(
            "ix_inventory_movement_item_time",
            "item_id",
            "created_at",
        ),
        Index(
            "ix_inventory_movement_status",
            "status",
            "created_at",
        ),
    )

    @validates("quantity")
    def validate_movement_quantity(
        self,
        key: str,
        value: Any,
    ) -> Decimal:
        result = validate_quantity(
            value, key, positive=True
        )
        assert result is not None
        return result

    def submit_for_approval(self) -> None:
        if self.status != InventoryMovementStatus.PROPOSED:
            raise ValueError(
                "Movement is not proposed"
            )

        self.status = (
            InventoryMovementStatus.PENDING_APPROVAL
        )

    def approve(self, approver_id: int) -> None:
        if self.status != (
            InventoryMovementStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Movement is not pending approval"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if (
            self.requested_by_user_id is not None
            and approver_id == self.requested_by_user_id
        ):
            raise ValueError(
                "Independent approval required"
            )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = InventoryMovementStatus.APPROVED

    def mark_posted(self, posted_by_user_id: int) -> None:
        """
        Call only after the inventory service has
        atomically updated stock and written the
        corresponding audit record.
        """
        if self.status != InventoryMovementStatus.APPROVED:
            raise ValueError(
                "Movement must be approved"
            )

        if posted_by_user_id <= 0:
            raise ValueError(
                "Invalid posting user"
            )

        self.posted_by_user_id = posted_by_user_id
        self.posted_at = utc_now()
        self.status = InventoryMovementStatus.POSTED


# ============================================================
# INVENTORY SUPPLY REQUEST
# ============================================================

class InventoryRequest(Base):
    __tablename__ = "inventory_requests"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    request_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    district_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "districts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    incident_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    shelter_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    destination_location_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_locations.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    item_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_items.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    requested_quantity: Mapped[
        Decimal
    ] = mapped_column(
        Numeric(15, 3),
        nullable=False,
    )

    approved_quantity: Mapped[
        Decimal
    ] = mapped_column(
        Numeric(15, 3),
        default=Decimal("0"),
        nullable=False,
    )

    allocated_quantity: Mapped[
        Decimal
    ] = mapped_column(
        Numeric(15, 3),
        default=Decimal("0"),
        nullable=False,
    )

    fulfilled_quantity: Mapped[
        Decimal
    ] = mapped_column(
        Numeric(15, 3),
        default=Decimal("0"),
        nullable=False,
    )

    priority: Mapped[
        InventoryPriority
    ] = mapped_column(
        enum_type(InventoryPriority),
        default=InventoryPriority.NORMAL,
        nullable=False,
    )

    status: Mapped[
        InventoryRequestStatus
    ] = mapped_column(
        enum_type(InventoryRequestStatus),
        default=InventoryRequestStatus.DRAFT,
        nullable=False,
        index=True,
    )

    justification: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    requested_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    needed_by: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    reservations: Mapped[
        list["InventoryReservation"]
    ] = relationship(
        "InventoryReservation",
        back_populates="request",
    )

    __table_args__ = (
        CheckConstraint(
            "requested_quantity > 0",
            name="ck_inventory_request_quantity",
        ),
        CheckConstraint(
            "approved_quantity >= 0",
            name="ck_inventory_request_approved",
        ),
        CheckConstraint(
            "allocated_quantity >= 0",
            name="ck_inventory_request_allocated",
        ),
        CheckConstraint(
            "fulfilled_quantity >= 0",
            name="ck_inventory_request_fulfilled",
        ),
        CheckConstraint(
            "approved_quantity <= requested_quantity",
            name="ck_inventory_request_approval_limit",
        ),
        CheckConstraint(
            "allocated_quantity <= approved_quantity",
            name="ck_inventory_request_allocation_limit",
        ),
        CheckConstraint(
            "fulfilled_quantity <= allocated_quantity",
            name="ck_inventory_request_fulfillment_limit",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_inventory_request_version",
        ),
        Index(
            "ix_inventory_request_district_status",
            "district_id",
            "status",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    def submit(self) -> None:
        if self.status != InventoryRequestStatus.DRAFT:
            raise ValueError(
                "Request is not a draft"
            )

        self.status = InventoryRequestStatus.SUBMITTED

    def approve(
        self,
        approver_id: int,
        quantity: Decimal,
    ) -> None:
        if self.status != (
            InventoryRequestStatus.UNDER_REVIEW
        ):
            raise ValueError(
                "Request is not under review"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if (
            self.requested_by_user_id is not None
            and approver_id == self.requested_by_user_id
        ):
            raise ValueError(
                "Independent approval required"
            )

        approved = validate_quantity(
            quantity,
            "approved_quantity",
            positive=True,
        )

        if approved > self.requested_quantity:
            raise ValueError(
                "Approved quantity exceeds request"
            )

        self.approved_quantity = approved
        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = InventoryRequestStatus.APPROVED


# ============================================================
# INVENTORY RESERVATION
# ============================================================

class InventoryReservation(Base):
    __tablename__ = "inventory_reservations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    reservation_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_requests.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    stock_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_stocks.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    quantity: Mapped[Decimal] = mapped_column(
        Numeric(15, 3),
        nullable=False,
    )

    status: Mapped[
        InventoryReservationStatus
    ] = mapped_column(
        enum_type(InventoryReservationStatus),
        default=InventoryReservationStatus.PENDING,
        nullable=False,
    )

    reserved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    expires_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    confirmed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    released_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    fulfilled_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    request: Mapped["InventoryRequest"] = relationship(
        "InventoryRequest",
        back_populates="reservations",
    )

    stock: Mapped["InventoryStock"] = relationship(
        "InventoryStock",
    )

    __table_args__ = (
        CheckConstraint(
            "quantity > 0",
            name="ck_inventory_reservation_quantity",
        ),
        Index(
            "ix_inventory_reservation_stock_status",
            "stock_id",
            "status",
        ),
    )


# ============================================================
# INVENTORY ALERT
# ============================================================

class InventoryAlert(Base):
    __tablename__ = "inventory_alerts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    item_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_items.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    location_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            ForeignKey(
                "inventory_locations.id",
                ondelete="RESTRICT",
            ),
            nullable=True,
        )
    )

    batch_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            ForeignKey(
                "inventory_batches.id",
                ondelete="RESTRICT",
            ),
            nullable=True,
        )
    )

    alert_type: Mapped[
        InventoryAlertType
    ] = mapped_column(
        enum_type(InventoryAlertType),
        nullable=False,
    )

    status: Mapped[
        InventoryAlertStatus
    ] = mapped_column(
        enum_type(InventoryAlertStatus),
        default=InventoryAlertStatus.OPEN,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    threshold_quantity: Mapped[
        Decimal | None
    ] = mapped_column(
        Numeric(15, 3),
        nullable=True,
    )

    observed_quantity: Mapped[
        Decimal | None
    ] = mapped_column(
        Numeric(15, 3),
        nullable=True,
    )

    is_ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    acknowledged_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    resolved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    acknowledged_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    resolved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "threshold_quantity IS NULL OR "
            "threshold_quantity >= 0",
            name="ck_inventory_alert_threshold",
        ),
        CheckConstraint(
            "observed_quantity IS NULL OR "
            "observed_quantity >= 0",
            name="ck_inventory_alert_observed",
        ),
        Index(
            "ix_inventory_alert_status_type",
            "status",
            "alert_type",
        ),
    )


# ============================================================
# TIMESTAMP EVENTS
# ============================================================

@event.listens_for(InventoryItem, "before_update")
def update_inventory_item_timestamp(
    mapper: Any,
    connection: Any,
    target: InventoryItem,
) -> None:
    target.updated_at = utc_now()


@event.listens_for(InventoryLocation, "before_update")
def update_inventory_location_timestamp(
    mapper: Any,
    connection: Any,
    target: InventoryLocation,
) -> None:
    target.updated_at = utc_now()


@event.listens_for(InventoryStock, "before_update")
def update_inventory_stock_timestamp(
    mapper: Any,
    connection: Any,
    target: InventoryStock,
) -> None:
    target.updated_at = utc_now()


@event.listens_for(InventoryRequest, "before_update")
def update_inventory_request_timestamp(
    mapper: Any,
    connection: Any,
    target: InventoryRequest,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

INVENTORY_MODELS = (
    InventoryItem,
    InventoryLocation,
    InventoryBatch,
    InventoryStock,
    InventoryMovement,
    InventoryRequest,
    InventoryReservation,
    InventoryAlert,
)


__all__ = [
    "InventoryItem",
    "InventoryLocation",
    "InventoryBatch",
    "InventoryStock",
    "InventoryMovement",
    "InventoryRequest",
    "InventoryReservation",
    "InventoryAlert",
    "InventoryCategory",
    "InventoryUnit",
    "InventoryLocationType",
    "InventoryLocationStatus",
    "InventoryBatchStatus",
    "InventoryMovementType",
    "InventoryMovementStatus",
    "InventoryRequestStatus",
    "InventoryPriority",
    "InventoryReservationStatus",
    "InventoryAlertType",
    "InventoryAlertStatus",
    "INVENTORY_MODELS",
]
