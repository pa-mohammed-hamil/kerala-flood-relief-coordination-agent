from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/relief.py

Pydantic v2 schemas for flood relief coordination.

Modules:
- Relief requests
- Relief inventory
- Warehouses and distribution centers
- Supply allocation
- Relief packages
- Distribution planning
- Vehicle and shipment tracking
- Delivery confirmation
- Stock transfers
- Donations
- Supply shortages
- Relief analytics
- Human approval and audit records

Safety:
- AI allocation recommendations are advisory.
- Physical dispatch requires authorized confirmation.
- Inventory must be based on verified records.
- Never report a delivery as completed without evidence.
- Do not expose beneficiary contact information publicly.
"""


from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def normalize_district(value: str) -> str:
    district = value.strip().lower()

    if district not in KERALA_DISTRICTS:
        raise ValueError(
            f"Invalid Kerala district: {value}"
        )

    return district


def validate_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Datetime must include timezone information"
        )

    return value


# ============================================================
# BASE SCHEMA
# ============================================================

class ReliefBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ============================================================
# ENUMS
# ============================================================

class ReliefCategory(str, Enum):
    FOOD = "food"
    DRINKING_WATER = "drinking_water"
    CLOTHING = "clothing"
    BLANKETS = "blankets"
    HYGIENE = "hygiene"
    SANITATION = "sanitation"
    BABY_SUPPLIES = "baby_supplies"
    MEDICAL_SUPPLIES = "medical_supplies"
    SHELTER_MATERIALS = "shelter_materials"
    CLEANING_SUPPLIES = "cleaning_supplies"
    EMERGENCY_EQUIPMENT = "emergency_equipment"
    OTHER = "other"


class ReliefPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReliefRequestStatus(str, Enum):
    RECEIVED = "received"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ALLOCATED = "allocated"
    PARTIALLY_FULFILLED = "partially_fulfilled"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"


class InventoryStatus(str, Enum):
    AVAILABLE = "available"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    RESERVED = "reserved"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class WarehouseStatus(str, Enum):
    OPERATIONAL = "operational"
    LIMITED = "limited"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class AllocationStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    RESERVED = "reserved"
    RELEASED = "released"
    CANCELLED = "cancelled"


class DeliveryStatus(str, Enum):
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    DISPATCHED = "dispatched"
    IN_TRANSIT = "in_transit"
    ARRIVED = "arrived"
    PARTIALLY_DELIVERED = "partially_delivered"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VehicleType(str, Enum):
    TRUCK = "truck"
    VAN = "van"
    JEEP = "jeep"
    BOAT = "boat"
    MOTORCYCLE = "motorcycle"
    OTHER = "other"


class VehicleStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    ASSIGNED = "assigned"
    IN_TRANSIT = "in_transit"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"


class DonationStatus(str, Enum):
    PLEDGED = "pledged"
    RECEIVED = "received"
    INSPECTED = "inspected"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DISTRIBUTED = "distributed"


class StockMovementType(str, Enum):
    RECEIPT = "receipt"
    ISSUE = "issue"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    ADJUSTMENT = "adjustment"
    RETURN = "return"
    DISPOSAL = "disposal"


# ============================================================
# LOCATION
# ============================================================

class ReliefLocation(ReliefBase):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    district: str

    address: str | None = None
    landmark: str | None = None
    panchayat: str | None = None
    ward: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# RELIEF ITEMS
# ============================================================

class ReliefItemCreate(ReliefBase):
    name: str = Field(
        min_length=2,
        max_length=200,
    )

    category: ReliefCategory

    description: str | None = None

    unit: str = Field(
        min_length=1,
        max_length=50,
    )

    perishable: bool = False

    requires_special_storage: bool = False


class ReliefItemResponse(ReliefItemCreate):
    item_id: UUID = Field(default_factory=uuid4)

    created_at: datetime = Field(
        default_factory=utc_now
    )

    active: bool = True


# ============================================================
# RELIEF REQUEST ITEMS
# ============================================================

class ReliefRequestedItem(ReliefBase):
    item_id: UUID

    quantity: float = Field(gt=0)

    unit: str = Field(
        min_length=1,
        max_length=50,
    )

    notes: str | None = None


# ============================================================
# RELIEF REQUEST
# ============================================================

class ReliefRequestCreate(ReliefBase):
    location: ReliefLocation

    requested_by: str | None = None

    contact_phone: str | None = None

    beneficiary_count: int = Field(
        ge=1,
    )

    children_count: int = Field(
        default=0,
        ge=0,
    )

    elderly_count: int = Field(
        default=0,
        ge=0,
    )

    items: list[ReliefRequestedItem] = Field(
        min_length=1,
    )

    description: str | None = Field(
        default=None,
        max_length=3000,
    )

    priority: ReliefPriority = ReliefPriority.MEDIUM

    shelter_id: UUID | None = None

    sos_request_id: UUID | None = None

    @model_validator(mode="after")
    def validate_counts(self):
        if self.children_count > self.beneficiary_count:
            raise ValueError(
                "Children count exceeds beneficiary count"
            )

        if self.elderly_count > self.beneficiary_count:
            raise ValueError(
                "Elderly count exceeds beneficiary count"
            )

        return self


class ReliefRequestResponse(ReliefRequestCreate):
    request_id: UUID = Field(
        default_factory=uuid4
    )

    status: ReliefRequestStatus = (
        ReliefRequestStatus.RECEIVED
    )

    approved_by: str | None = None

    approved_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# WAREHOUSE
# ============================================================

class WarehouseCreate(ReliefBase):
    name: str = Field(
        min_length=2,
        max_length=200,
    )

    location: ReliefLocation

    manager_name: str | None = None

    contact_phone: str | None = None

    capacity_description: str | None = None

    supports_cold_storage: bool = False

    supports_medical_storage: bool = False


class WarehouseResponse(WarehouseCreate):
    warehouse_id: UUID = Field(
        default_factory=uuid4
    )

    status: WarehouseStatus = (
        WarehouseStatus.UNKNOWN
    )

    verified: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# INVENTORY
# ============================================================

class InventoryCreate(ReliefBase):
    warehouse_id: UUID
    item_id: UUID

    quantity_on_hand: float = Field(ge=0)

    quantity_reserved: float = Field(
        default=0,
        ge=0,
    )

    minimum_stock_level: float = Field(
        default=0,
        ge=0,
    )

    unit: str

    batch_number: str | None = None

    expiry_date: datetime | None = None

    @model_validator(mode="after")
    def validate_quantities(self):
        if self.quantity_reserved > self.quantity_on_hand:
            raise ValueError(
                "Reserved quantity cannot exceed on-hand stock"
            )

        return self


class InventoryResponse(InventoryCreate):
    inventory_id: UUID = Field(
        default_factory=uuid4
    )

    status: InventoryStatus = (
        InventoryStatus.AVAILABLE
    )

    verified: bool = False

    last_counted_at: datetime | None = None

    updated_at: datetime = Field(
        default_factory=utc_now
    )


class InventoryAvailability(ReliefBase):
    warehouse_id: UUID
    item_id: UUID

    quantity_on_hand: float = Field(ge=0)
    quantity_reserved: float = Field(ge=0)
    quantity_available: float = Field(ge=0)

    unit: str

    @model_validator(mode="after")
    def validate_balance(self):
        if abs(
            self.quantity_available
            - (
                self.quantity_on_hand
                - self.quantity_reserved
            )
        ) > 0.000001:
            raise ValueError(
                "Available stock does not match inventory balance"
            )

        return self


# ============================================================
# STOCK MOVEMENT
# ============================================================

class StockMovementCreate(ReliefBase):
    warehouse_id: UUID
    item_id: UUID

    movement_type: StockMovementType

    quantity: float = Field(gt=0)

    unit: str

    reference_id: UUID | None = None

    destination_warehouse_id: UUID | None = None

    reason: str | None = None


class StockMovementResponse(StockMovementCreate):
    movement_id: UUID = Field(
        default_factory=uuid4
    )

    performed_by: str

    performed_at: datetime = Field(
        default_factory=utc_now
    )

    confirmed: bool = False


# ============================================================
# ALLOCATION
# ============================================================

class ReliefAllocationItem(ReliefBase):
    item_id: UUID
    warehouse_id: UUID

    quantity: float = Field(gt=0)

    unit: str


class ReliefAllocationCreate(ReliefBase):
    request_id: UUID

    items: list[ReliefAllocationItem] = Field(
        min_length=1,
    )

    recommendation_reason: str | None = None

    generated_by_ai: bool = False


class ReliefAllocationResponse(ReliefAllocationCreate):
    allocation_id: UUID = Field(
        default_factory=uuid4
    )

    status: AllocationStatus = (
        AllocationStatus.PENDING_APPROVAL
    )

    approved_by: str | None = None

    approved_at: datetime | None = None

    reserved_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# AI ALLOCATION RECOMMENDATION
# ============================================================

class AIAllocationRecommendation(ReliefBase):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    request_id: UUID

    suggested_allocations: list[
        ReliefAllocationItem
    ] = Field(default_factory=list)

    estimated_delivery_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_transport_cost_inr: float | None = Field(
        default=None,
        ge=0,
    )

    reasoning_summary: str

    constraints: list[str] = Field(
        default_factory=list
    )

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    human_approval_required: bool = True

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RELIEF PACKAGE
# ============================================================

class ReliefPackageItem(ReliefBase):
    item_id: UUID

    quantity: float = Field(gt=0)

    unit: str


class ReliefPackageCreate(ReliefBase):
    package_name: str = Field(
        min_length=2,
        max_length=200,
    )

    description: str | None = None

    items: list[ReliefPackageItem] = Field(
        min_length=1,
    )

    intended_beneficiaries: int | None = Field(
        default=None,
        ge=1,
    )


class ReliefPackageResponse(ReliefPackageCreate):
    package_id: UUID = Field(
        default_factory=uuid4
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# TRANSPORT VEHICLE
# ============================================================

class ReliefVehicleCreate(ReliefBase):
    registration_number: str = Field(
        min_length=3,
        max_length=30,
    )

    vehicle_type: VehicleType

    operator_name: str

    district: str

    load_capacity_kg: float | None = Field(
        default=None,
        gt=0,
    )

    contact_phone: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class ReliefVehicleResponse(ReliefVehicleCreate):
    vehicle_id: UUID = Field(
        default_factory=uuid4
    )

    status: VehicleStatus = (
        VehicleStatus.UNAVAILABLE
    )

    current_location: ReliefLocation | None = None

    verified: bool = False

    last_updated: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RELIEF DELIVERY
# ============================================================

class ReliefDeliveryCreate(ReliefBase):
    allocation_id: UUID

    origin_warehouse_id: UUID

    destination: ReliefLocation

    vehicle_id: UUID | None = None

    assigned_driver_id: UUID | None = None

    planned_departure: datetime | None = None

    estimated_arrival: datetime | None = None

    delivery_notes: str | None = None

    @model_validator(mode="after")
    def validate_schedule(self):
        if self.planned_departure is not None:
            validate_datetime(self.planned_departure)

        if self.estimated_arrival is not None:
            validate_datetime(self.estimated_arrival)

        if (
            self.planned_departure is not None
            and self.estimated_arrival is not None
            and self.estimated_arrival
            < self.planned_departure
        ):
            raise ValueError(
                "Estimated arrival precedes departure"
            )

        return self


class ReliefDeliveryResponse(ReliefDeliveryCreate):
    delivery_id: UUID = Field(
        default_factory=uuid4
    )

    status: DeliveryStatus = (
        DeliveryStatus.PLANNED
    )

    approved_by: str | None = None
    approved_at: datetime | None = None

    dispatch_confirmed_by: str | None = None
    dispatched_at: datetime | None = None

    delivered_at: datetime | None = None

    proof_of_delivery_id: UUID | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_delivery_status(self):
        if self.status == DeliveryStatus.DELIVERED:
            if (
                self.delivered_at is None
                or self.proof_of_delivery_id is None
            ):
                raise ValueError(
                    "Completed delivery requires confirmation evidence"
                )

        return self


# ============================================================
# DELIVERY TRACKING
# ============================================================

class DeliveryTrackingUpdate(ReliefBase):
    delivery_id: UUID

    location: ReliefLocation

    status: DeliveryStatus

    notes: str | None = None

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# PROOF OF DELIVERY
# ============================================================

class ProofOfDelivery(ReliefBase):
    proof_id: UUID = Field(
        default_factory=uuid4
    )

    delivery_id: UUID

    received_by: str

    confirmed_item_quantities: list[
        ReliefPackageItem
    ] = Field(min_length=1)

    confirmation_reference: str | None = None

    evidence_file_reference: str | None = None

    notes: str | None = None

    confirmed_at: datetime = Field(
        default_factory=utc_now
    )

    verified: bool = False


# ============================================================
# DONATIONS
# ============================================================

class ReliefDonationCreate(ReliefBase):
    donor_name: str | None = None

    organization_name: str | None = None

    contact_phone: str | None = None

    items: list[ReliefPackageItem] = Field(
        min_length=1,
    )

    destination_warehouse_id: UUID | None = None

    notes: str | None = None


class ReliefDonationResponse(ReliefDonationCreate):
    donation_id: UUID = Field(
        default_factory=uuid4
    )

    status: DonationStatus = (
        DonationStatus.PLEDGED
    )

    received_at: datetime | None = None

    inspected_by: str | None = None

    verified: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# SUPPLY SHORTAGES
# ============================================================

class SupplyShortage(ReliefBase):
    shortage_id: UUID = Field(
        default_factory=uuid4
    )

    warehouse_id: UUID
    item_id: UUID

    quantity_required: float = Field(gt=0)

    quantity_available: float = Field(ge=0)

    shortage_quantity: float = Field(gt=0)

    priority: ReliefPriority = ReliefPriority.MEDIUM

    detected_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_shortage(self):
        expected = max(
            self.quantity_required
            - self.quantity_available,
            0,
        )

        if abs(
            expected - self.shortage_quantity
        ) > 0.000001:
            raise ValueError(
                "Shortage quantity does not match stock deficit"
            )

        return self


# ============================================================
# HUMAN APPROVAL
# ============================================================

class ReliefApprovalRequest(ReliefBase):
    resource_id: UUID

    resource_type: str

    approve: bool

    reason: str = Field(
        min_length=5,
        max_length=2000,
    )


class ReliefApprovalResponse(ReliefBase):
    resource_id: UUID

    approved: bool

    reviewed_by: str

    reviewed_at: datetime = Field(
        default_factory=utc_now
    )

    authorization_verified: bool = False


# ============================================================
# AUDIT LOG
# ============================================================

class ReliefAuditEvent(ReliefBase):
    event_id: UUID = Field(
        default_factory=uuid4
    )

    entity_id: UUID

    entity_type: str

    action: str

    performed_by: str

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RELIEF DASHBOARD
# ============================================================

class ReliefDashboard(ReliefBase):
    district: str | None = None

    active_relief_requests: int | None = Field(
        default=None,
        ge=0,
    )

    pending_allocations: int | None = Field(
        default=None,
        ge=0,
    )

    operational_warehouses: int | None = Field(
        default=None,
        ge=0,
    )

    low_stock_items: int | None = Field(
        default=None,
        ge=0,
    )

    available_vehicles: int | None = Field(
        default=None,
        ge=0,
    )

    active_deliveries: int | None = Field(
        default=None,
        ge=0,
    )

    completed_deliveries: int | None = Field(
        default=None,
        ge=0,
    )

    verified_beneficiaries_served: int | None = Field(
        default=None,
        ge=0,
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    data_verified: bool = False


# ============================================================
# GENERIC API RESPONSE
# ============================================================

class ReliefAPIResponse(ReliefBase):
    success: bool

    message: str

    data: dict[str, Any] | None = None

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "ReliefCategory",
    "ReliefPriority",
    "ReliefRequestStatus",
    "InventoryStatus",
    "WarehouseStatus",
    "AllocationStatus",
    "DeliveryStatus",
    "VehicleType",
    "VehicleStatus",
    "DonationStatus",
    "StockMovementType",
    "ReliefLocation",
    "ReliefItemCreate",
    "ReliefItemResponse",
    "ReliefRequestedItem",
    "ReliefRequestCreate",
    "ReliefRequestResponse",
    "WarehouseCreate",
    "WarehouseResponse",
    "InventoryCreate",
    "InventoryResponse",
    "InventoryAvailability",
    "StockMovementCreate",
    "StockMovementResponse",
    "ReliefAllocationItem",
    "ReliefAllocationCreate",
    "ReliefAllocationResponse",
    "AIAllocationRecommendation",
    "ReliefPackageItem",
    "ReliefPackageCreate",
    "ReliefPackageResponse",
    "ReliefVehicleCreate",
    "ReliefVehicleResponse",
    "ReliefDeliveryCreate",
    "ReliefDeliveryResponse",
    "DeliveryTrackingUpdate",
    "ProofOfDelivery",
    "ReliefDonationCreate",
    "ReliefDonationResponse",
    "SupplyShortage",
    "ReliefApprovalRequest",
    "ReliefApprovalResponse",
    "ReliefAuditEvent",
    "ReliefDashboard",
    "ReliefAPIResponse",
    "normalize_district",
    "validate_datetime",
]
