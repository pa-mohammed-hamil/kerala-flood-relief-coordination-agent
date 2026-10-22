from __future__ import annotations

"""
backend/models/relief_delivery.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x ORM models for relief delivery,
distribution, shipment tracking, and verification.

Features:
- Relief delivery registration
- District, shelter, and incident coordination
- Food, water, medicine, and emergency supplies
- Delivery items and batch traceability
- Warehouse dispatch coordination
- Vehicle and driver references
- Delivery route and checkpoint tracking
- Proof of delivery
- Recipient acknowledgment
- Partial and failed deliveries
- Damage and shortage reporting
- Human approval and verification
- AI-assisted logistics recommendations
- Delivery event history
- Audit references and optimistic locking

SAFETY:
- AI recommendations are advisory.
- Dispatch requires authorized human approval.
- Delivery confirmation requires evidence.
- Inventory deductions must be transactional.
- Expired or recalled supplies must not be shipped.
- Sensitive recipient information must not be
  exposed through public APIs.
"""


import enum
import math
import uuid

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    Float,
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
# HELPERS
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
    positive: bool = False,
    allow_none: bool = False,
) -> Decimal | None:
    if value is None:
        if allow_none:
            return None

        raise ValueError(
            f"{field_name} is required"
        )

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
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
            f"{field_name} exceeds supported precision"
        )

    return result


def validate_coordinate(
    value: float | None,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"Invalid {field_name}"
        )

    return float(value)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class ReliefDeliveryStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    PLANNING = "planning"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    READY_FOR_DISPATCH = "ready_for_dispatch"
    DISPATCHED = "dispatched"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    ARRIVED = "arrived"
    PARTIALLY_DELIVERED = "partially_delivered"
    DELIVERED = "delivered"
    DELIVERY_VERIFIED = "delivery_verified"
    COMPLETED = "completed"
    FAILED = "failed"
    RETURNED = "returned"
    CANCELLED = "cancelled"


class ReliefDeliveryPriority(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ReliefDeliveryType(str, enum.Enum):
    WAREHOUSE_TO_SHELTER = "warehouse_to_shelter"
    WAREHOUSE_TO_HOSPITAL = "warehouse_to_hospital"
    WAREHOUSE_TO_DISTRIBUTION_CENTRE = (
        "warehouse_to_distribution_centre"
    )
    SHELTER_TO_SHELTER = "shelter_to_shelter"
    DIRECT_COMMUNITY_DISTRIBUTION = (
        "direct_community_distribution"
    )
    EMERGENCY_MEDICAL_SUPPLY = (
        "emergency_medical_supply"
    )
    RESCUE_TEAM_SUPPLY = "rescue_team_supply"
    INTER_DISTRICT_TRANSFER = "inter_district_transfer"
    DONATION_DELIVERY = "donation_delivery"
    OTHER = "other"


class ReliefTransportMode(str, enum.Enum):
    TRUCK = "truck"
    VAN = "van"
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BOAT = "boat"
    RESCUE_BOAT = "rescue_boat"
    HELICOPTER = "helicopter"
    DRONE = "drone"
    ON_FOOT = "on_foot"
    MIXED = "mixed"
    OTHER = "other"


class ReliefRouteStatus(str, enum.Enum):
    NOT_ASSESSED = "not_assessed"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED_SAFE = "verified_safe"
    RESTRICTED = "restricted"
    UNSAFE = "unsafe"
    EXPIRED = "expired"


class ReliefCheckpointStatus(str, enum.Enum):
    PENDING = "pending"
    REACHED = "reached"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class ReliefDeliveryItemStatus(str, enum.Enum):
    PLANNED = "planned"
    ALLOCATED = "allocated"
    PACKED = "packed"
    DISPATCHED = "dispatched"
    PARTIALLY_RECEIVED = "partially_received"
    RECEIVED = "received"
    DAMAGED = "damaged"
    REJECTED = "rejected"
    RETURNED = "returned"


class ReliefProofType(str, enum.Enum):
    SIGNATURE = "signature"
    PHOTO = "photo"
    DOCUMENT = "document"
    QR_CODE = "qr_code"
    OTP = "otp"
    RECEIPT = "receipt"
    OFFICIAL_CONFIRMATION = "official_confirmation"
    OTHER = "other"


class ReliefProofStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    REJECTED = "rejected"


class ReliefDeliveryExceptionType(str, enum.Enum):
    ROAD_BLOCKED = "road_blocked"
    FLOOD_RISK = "flood_risk"
    VEHICLE_BREAKDOWN = "vehicle_breakdown"
    WEATHER_DELAY = "weather_delay"
    ROUTE_UNSAFE = "route_unsafe"
    RECIPIENT_UNAVAILABLE = "recipient_unavailable"
    SHORTAGE = "shortage"
    DAMAGED_GOODS = "damaged_goods"
    EXPIRED_GOODS = "expired_goods"
    COLD_CHAIN_FAILURE = "cold_chain_failure"
    DOCUMENT_MISMATCH = "document_mismatch"
    SECURITY_INCIDENT = "security_incident"
    OTHER = "other"


class ReliefDeliveryEventType(str, enum.Enum):
    CREATED = "created"
    VERIFIED = "verified"
    PLAN_UPDATED = "plan_updated"
    ROUTE_VERIFIED = "route_verified"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    ITEMS_ALLOCATED = "items_allocated"
    PACKED = "packed"
    DISPATCHED = "dispatched"
    CHECKPOINT_REACHED = "checkpoint_reached"
    DELAY_REPORTED = "delay_reported"
    ROUTE_CHANGED = "route_changed"
    ARRIVED = "arrived"
    PARTIAL_DELIVERY = "partial_delivery"
    DELIVERY_RECORDED = "delivery_recorded"
    PROOF_SUBMITTED = "proof_submitted"
    PROOF_VERIFIED = "proof_verified"
    COMPLETED = "completed"
    FAILED = "failed"
    RETURNED = "returned"
    CANCELLED = "cancelled"
    EXCEPTION_REPORTED = "exception_reported"
    NOTE_ADDED = "note_added"


# ============================================================
# RELIEF DELIVERY
# ============================================================

class ReliefDelivery(Base):
    __tablename__ = "relief_deliveries"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    delivery_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
        index=True,
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

    destination_district_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            ForeignKey(
                "districts.id",
                ondelete="RESTRICT",
            ),
            nullable=True,
        )
    )

    incident_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    inventory_request_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_requests.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    delivery_type: Mapped[
        ReliefDeliveryType
    ] = mapped_column(
        enum_type(ReliefDeliveryType),
        nullable=False,
    )

    priority: Mapped[
        ReliefDeliveryPriority
    ] = mapped_column(
        enum_type(ReliefDeliveryPriority),
        default=ReliefDeliveryPriority.NORMAL,
        nullable=False,
        index=True,
    )

    status: Mapped[
        ReliefDeliveryStatus
    ] = mapped_column(
        enum_type(ReliefDeliveryStatus),
        default=ReliefDeliveryStatus.DRAFT,
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source_location_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_locations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
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

    destination_shelter_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    destination_hospital_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    destination_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    destination_address: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    destination_latitude: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    destination_longitude: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    recipient_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    recipient_contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    expected_beneficiary_count: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    transport_mode: Mapped[
        ReliefTransportMode
    ] = mapped_column(
        enum_type(ReliefTransportMode),
        nullable=False,
    )

    vehicle_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    driver_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    driver_contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    logistics_provider_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    route_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    route_status: Mapped[
        ReliefRouteStatus
    ] = mapped_column(
        enum_type(ReliefRouteStatus),
        default=ReliefRouteStatus.NOT_ASSESSED,
        nullable=False,
    )

    route_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    route_verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    route_valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    requires_cold_chain: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    cold_chain_verified: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    requires_special_handling: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_ai_planned: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    ai_plan_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    created_by_user_id: Mapped[
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

    approval_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    dispatched_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    received_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    delivery_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    delivery_verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    planned_dispatch_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    estimated_arrival_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    dispatched_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    arrived_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    delivered_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    completed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    cancellation_reason: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    failure_reason: Mapped[str | None] = (
        mapped_column(
            Text,
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

    items: Mapped[
        list["ReliefDeliveryItem"]
    ] = relationship(
        "ReliefDeliveryItem",
        back_populates="delivery",
        cascade="save-update, merge",
    )

    checkpoints: Mapped[
        list["ReliefDeliveryCheckpoint"]
    ] = relationship(
        "ReliefDeliveryCheckpoint",
        back_populates="delivery",
        cascade="save-update, merge",
    )

    proofs: Mapped[
        list["ReliefDeliveryProof"]
    ] = relationship(
        "ReliefDeliveryProof",
        back_populates="delivery",
        cascade="save-update, merge",
    )

    exceptions: Mapped[
        list["ReliefDeliveryException"]
    ] = relationship(
        "ReliefDeliveryException",
        back_populates="delivery",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["ReliefDeliveryEvent"]
    ] = relationship(
        "ReliefDeliveryEvent",
        back_populates="delivery",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_relief_delivery_district",
        ),
        CheckConstraint(
            "expected_beneficiary_count IS NULL OR "
            "expected_beneficiary_count >= 0",
            name="ck_relief_delivery_beneficiaries",
        ),
        CheckConstraint(
            "destination_latitude IS NULL OR "
            "(destination_latitude >= -90 "
            "AND destination_latitude <= 90)",
            name="ck_relief_delivery_lat",
        ),
        CheckConstraint(
            "destination_longitude IS NULL OR "
            "(destination_longitude >= -180 "
            "AND destination_longitude <= 180)",
            name="ck_relief_delivery_lon",
        ),
        CheckConstraint(
            "destination_location_id IS NULL OR "
            "source_location_id != destination_location_id",
            name="ck_relief_delivery_locations",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_relief_delivery_version",
        ),
        Index(
            "ix_relief_delivery_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_relief_delivery_priority_status",
            "priority",
            "status",
        ),
        Index(
            "ix_relief_delivery_eta",
            "status",
            "estimated_arrival_at",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "delivery_number",
        "title",
        "destination_name",
        "idempotency_key",
    )
    def validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "delivery_number": 120,
            "title": 255,
            "destination_name": 255,
            "idempotency_key": 150,
        }

        return require_text(
            value,
            key,
            limits[key],
        )

    @validates("destination_latitude")
    def validate_latitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -90, 90
        )

    @validates("destination_longitude")
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -180, 180
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            ReliefDeliveryStatus.COMPLETED,
            ReliefDeliveryStatus.FAILED,
            ReliefDeliveryStatus.RETURNED,
            ReliefDeliveryStatus.CANCELLED,
        }

    @property
    def is_human_approved(self) -> bool:
        return (
            self.approved_by_user_id is not None
            and self.approved_at is not None
            and bool(self.approval_reference)
        )

    @property
    def has_valid_route(self) -> bool:
        if (
            self.route_status
            != ReliefRouteStatus.VERIFIED_SAFE
        ):
            return False

        if self.route_verified_by_user_id is None:
            return False

        if self.route_valid_until is None:
            return False

        return as_utc(self.route_valid_until) > utc_now()

    @property
    def can_dispatch(self) -> bool:
        return (
            self.status
            == ReliefDeliveryStatus.READY_FOR_DISPATCH
            and self.is_verified
            and self.is_human_approved
            and self.has_valid_route
            and (
                not self.requires_cold_chain
                or self.cold_chain_verified
            )
            and bool(self.items)
            and all(
                item.status
                in {
                    ReliefDeliveryItemStatus.ALLOCATED,
                    ReliefDeliveryItemStatus.PACKED,
                }
                for item in self.items
            )
        )

    def submit_for_verification(self) -> None:
        if self.status != ReliefDeliveryStatus.DRAFT:
            raise ValueError(
                "Only draft deliveries can be submitted"
            )

        self.status = (
            ReliefDeliveryStatus.PENDING_VERIFICATION
        )

    def verify(self, verifier_id: int) -> None:
        if self.status != (
            ReliefDeliveryStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Delivery is not pending verification"
            )

        if verifier_id <= 0:
            raise ValueError("Invalid verifier")

        if (
            self.created_by_user_id is not None
            and verifier_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent verification required"
            )

        self.is_verified = True
        self.verified_by_user_id = verifier_id
        self.verified_at = utc_now()
        self.status = ReliefDeliveryStatus.VERIFIED

    def begin_planning(self) -> None:
        if self.status != ReliefDeliveryStatus.VERIFIED:
            raise ValueError(
                "Delivery must be verified"
            )

        self.status = ReliefDeliveryStatus.PLANNING

    def request_approval(self) -> None:
        if self.status != ReliefDeliveryStatus.PLANNING:
            raise ValueError(
                "Delivery is not in planning"
            )

        self.status = (
            ReliefDeliveryStatus.PENDING_APPROVAL
        )

    def approve(
        self,
        *,
        approver_id: int,
        approval_reference: str,
    ) -> None:
        if self.status != (
            ReliefDeliveryStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Delivery is not pending approval"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if approver_id in {
            self.created_by_user_id,
            self.verified_by_user_id,
        }:
            raise ValueError(
                "Independent approval required"
            )

        self.approval_reference = require_text(
            approval_reference,
            "approval_reference",
            500,
        )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = ReliefDeliveryStatus.APPROVED

    def mark_ready(self) -> None:
        if self.status != ReliefDeliveryStatus.APPROVED:
            raise ValueError(
                "Delivery is not approved"
            )

        self.status = (
            ReliefDeliveryStatus.READY_FOR_DISPATCH
        )

    def dispatch(self, dispatcher_id: int) -> None:
        """
        Invoke only after the logistics service has
        atomically posted stock movements and confirmed
        vehicle, driver, route and batch safety.
        """
        if not self.can_dispatch:
            raise ValueError(
                "Delivery does not meet dispatch requirements"
            )

        if dispatcher_id <= 0:
            raise ValueError("Invalid dispatcher")

        self.dispatched_by_user_id = dispatcher_id
        self.dispatched_at = utc_now()
        self.status = ReliefDeliveryStatus.DISPATCHED

    def mark_in_transit(self) -> None:
        if self.status != ReliefDeliveryStatus.DISPATCHED:
            raise ValueError(
                "Delivery has not been dispatched"
            )

        self.status = ReliefDeliveryStatus.IN_TRANSIT

    def mark_arrived(self) -> None:
        if self.status not in {
            ReliefDeliveryStatus.IN_TRANSIT,
            ReliefDeliveryStatus.DELAYED,
        }:
            raise ValueError(
                "Delivery is not in transit"
            )

        self.arrived_at = utc_now()
        self.status = ReliefDeliveryStatus.ARRIVED

    def record_delivery(
        self,
        *,
        received_by_user_id: int,
        partial: bool = False,
    ) -> None:
        if self.status not in {
            ReliefDeliveryStatus.ARRIVED,
            ReliefDeliveryStatus.PARTIALLY_DELIVERED,
        }:
            raise ValueError(
                "Delivery has not arrived"
            )

        if received_by_user_id <= 0:
            raise ValueError(
                "Invalid receiving user"
            )

        self.received_by_user_id = received_by_user_id
        self.delivered_at = utc_now()

        self.status = (
            ReliefDeliveryStatus.PARTIALLY_DELIVERED
            if partial
            else ReliefDeliveryStatus.DELIVERED
        )

    def verify_delivery(
        self,
        *,
        verifier_id: int,
    ) -> None:
        if self.status != ReliefDeliveryStatus.DELIVERED:
            raise ValueError(
                "Delivery is not fully recorded"
            )

        if verifier_id <= 0:
            raise ValueError("Invalid verifier")

        if verifier_id in {
            self.dispatched_by_user_id,
            self.received_by_user_id,
        }:
            raise ValueError(
                "Independent delivery verification required"
            )

        if not any(
            proof.status == ReliefProofStatus.VERIFIED
            for proof in self.proofs
        ):
            raise ValueError(
                "Verified proof of delivery required"
            )

        self.delivery_verified_by_user_id = verifier_id
        self.delivery_verified_at = utc_now()
        self.status = (
            ReliefDeliveryStatus.DELIVERY_VERIFIED
        )

    def complete(self) -> None:
        if self.status != (
            ReliefDeliveryStatus.DELIVERY_VERIFIED
        ):
            raise ValueError(
                "Delivery verification is incomplete"
            )

        self.status = ReliefDeliveryStatus.COMPLETED
        self.completed_at = utc_now()

    def cancel(self, reason: str) -> None:
        if self.is_terminal:
            raise ValueError(
                "Terminal delivery cannot be cancelled"
            )

        if self.status in {
            ReliefDeliveryStatus.DISPATCHED,
            ReliefDeliveryStatus.IN_TRANSIT,
            ReliefDeliveryStatus.DELAYED,
            ReliefDeliveryStatus.ARRIVED,
            ReliefDeliveryStatus.PARTIALLY_DELIVERED,
            ReliefDeliveryStatus.DELIVERED,
            ReliefDeliveryStatus.DELIVERY_VERIFIED,
        }:
            raise ValueError(
                "Dispatched deliveries require a "
                "controlled return or exception workflow"
            )

        self.cancellation_reason = require_text(
            reason,
            "reason",
            5000,
        )
        self.status = ReliefDeliveryStatus.CANCELLED

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "delivery_number": self.delivery_number,
            "district_id": self.district_id,
            "delivery_type": self.delivery_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "title": self.title,
            "destination_name": self.destination_name,
            "transport_mode": self.transport_mode.value,
            "route_status": self.route_status.value,
            "is_verified": self.is_verified,
            "is_human_approved": self.is_human_approved,
            "requires_cold_chain": self.requires_cold_chain,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }


# ============================================================
# RELIEF DELIVERY ITEM
# ============================================================

class ReliefDeliveryItem(Base):
    __tablename__ = "relief_delivery_items"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    delivery_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "relief_deliveries.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
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

    stock_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "inventory_stocks.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    reservation_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            ForeignKey(
                "inventory_reservations.id",
                ondelete="RESTRICT",
            ),
            nullable=True,
        )
    )

    requested_quantity: Mapped[Decimal] = (
        mapped_column(
            Numeric(15, 3),
            nullable=False,
        )
    )

    allocated_quantity: Mapped[Decimal] = (
        mapped_column(
            Numeric(15, 3),
            default=Decimal("0"),
            nullable=False,
        )
    )

    dispatched_quantity: Mapped[Decimal] = (
        mapped_column(
            Numeric(15, 3),
            default=Decimal("0"),
            nullable=False,
        )
    )

    received_quantity: Mapped[Decimal] = (
        mapped_column(
            Numeric(15, 3),
            default=Decimal("0"),
            nullable=False,
        )
    )

    damaged_quantity: Mapped[Decimal] = (
        mapped_column(
            Numeric(15, 3),
            default=Decimal("0"),
            nullable=False,
        )
    )

    rejected_quantity: Mapped[Decimal] = (
        mapped_column(
            Numeric(15, 3),
            default=Decimal("0"),
            nullable=False,
        )
    )

    status: Mapped[
        ReliefDeliveryItemStatus
    ] = mapped_column(
        enum_type(ReliefDeliveryItemStatus),
        default=ReliefDeliveryItemStatus.PLANNED,
        nullable=False,
    )

    requires_expiry_verification: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    batch_quality_verified: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    checked_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    checked_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    notes: Mapped[str | None] = mapped_column(
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

    delivery: Mapped["ReliefDelivery"] = relationship(
        "ReliefDelivery",
        back_populates="items",
    )

    __table_args__ = (
        UniqueConstraint(
            "delivery_id",
            "batch_id",
            name="uq_relief_delivery_batch",
        ),
        CheckConstraint(
            "requested_quantity > 0",
            name="ck_relief_item_requested",
        ),
        CheckConstraint(
            "allocated_quantity >= 0",
            name="ck_relief_item_allocated",
        ),
        CheckConstraint(
            "dispatched_quantity >= 0",
            name="ck_relief_item_dispatched",
        ),
        CheckConstraint(
            "received_quantity >= 0",
            name="ck_relief_item_received",
        ),
        CheckConstraint(
            "damaged_quantity >= 0",
            name="ck_relief_item_damaged",
        ),
        CheckConstraint(
            "rejected_quantity >= 0",
            name="ck_relief_item_rejected",
        ),
        CheckConstraint(
            "allocated_quantity <= requested_quantity",
            name="ck_relief_item_allocation_limit",
        ),
        CheckConstraint(
            "dispatched_quantity <= allocated_quantity",
            name="ck_relief_item_dispatch_limit",
        ),
        CheckConstraint(
            "received_quantity + damaged_quantity + "
            "rejected_quantity <= dispatched_quantity",
            name="ck_relief_item_receipt_limit",
        ),
        Index(
            "ix_relief_item_delivery_status",
            "delivery_id",
            "status",
        ),
    )

    @validates(
        "requested_quantity",
        "allocated_quantity",
        "dispatched_quantity",
        "received_quantity",
        "damaged_quantity",
        "rejected_quantity",
    )
    def validate_item_quantity(
        self,
        key: str,
        value: Any,
    ) -> Decimal:
        result = validate_quantity(
            value,
            key,
            positive=(key == "requested_quantity"),
        )
        assert result is not None
        return result

    @property
    def outstanding_quantity(self) -> Decimal:
        return max(
            self.dispatched_quantity
            - self.received_quantity
            - self.damaged_quantity
            - self.rejected_quantity,
            Decimal("0"),
        )


# ============================================================
# DELIVERY CHECKPOINT
# ============================================================

class ReliefDeliveryCheckpoint(Base):
    __tablename__ = "relief_delivery_checkpoints"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    delivery_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "relief_deliveries.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    checkpoint_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    status: Mapped[
        ReliefCheckpointStatus
    ] = mapped_column(
        enum_type(ReliefCheckpointStatus),
        default=ReliefCheckpointStatus.PENDING,
        nullable=False,
    )

    is_mandatory: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    expected_arrival_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    actual_arrival_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    recorded_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
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

    delivery: Mapped["ReliefDelivery"] = relationship(
        "ReliefDelivery",
        back_populates="checkpoints",
    )

    __table_args__ = (
        UniqueConstraint(
            "delivery_id",
            "checkpoint_number",
            name="uq_relief_checkpoint_number",
        ),
        CheckConstraint(
            "checkpoint_number > 0",
            name="ck_relief_checkpoint_positive",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_relief_checkpoint_lat",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_relief_checkpoint_lon",
        ),
    )


# ============================================================
# PROOF OF DELIVERY
# ============================================================

class ReliefDeliveryProof(Base):
    __tablename__ = "relief_delivery_proofs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    delivery_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "relief_deliveries.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    proof_type: Mapped[ReliefProofType] = mapped_column(
        enum_type(ReliefProofType),
        nullable=False,
    )

    status: Mapped[
        ReliefProofStatus
    ] = mapped_column(
        enum_type(ReliefProofStatus),
        default=ReliefProofStatus.SUBMITTED,
        nullable=False,
    )

    storage_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    content_sha256: Mapped[str | None] = (
        mapped_column(
            String(64),
            nullable=True,
        )
    )

    receipt_reference: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    recipient_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    submitted_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    review_notes: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    delivery: Mapped["ReliefDelivery"] = relationship(
        "ReliefDelivery",
        back_populates="proofs",
    )

    __table_args__ = (
        CheckConstraint(
            "content_sha256 IS NULL OR "
            "length(content_sha256) = 64",
            name="ck_relief_proof_sha256",
        ),
        Index(
            "ix_relief_proof_delivery_status",
            "delivery_id",
            "status",
        ),
    )

    def verify(self, verifier_id: int) -> None:
        if self.status not in {
            ReliefProofStatus.SUBMITTED,
            ReliefProofStatus.UNDER_REVIEW,
        }:
            raise ValueError(
                "Proof cannot be verified"
            )

        if verifier_id <= 0:
            raise ValueError("Invalid verifier")

        if (
            self.submitted_by_user_id is not None
            and verifier_id == self.submitted_by_user_id
        ):
            raise ValueError(
                "Independent proof verification required"
            )

        if not (
            self.storage_reference
            or self.receipt_reference
        ):
            raise ValueError(
                "Proof evidence reference is required"
            )

        self.verified_by_user_id = verifier_id
        self.verified_at = utc_now()
        self.status = ReliefProofStatus.VERIFIED


# ============================================================
# DELIVERY EXCEPTION
# ============================================================

class ReliefDeliveryException(Base):
    __tablename__ = "relief_delivery_exceptions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    delivery_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "relief_deliveries.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    exception_type: Mapped[
        ReliefDeliveryExceptionType
    ] = mapped_column(
        enum_type(ReliefDeliveryExceptionType),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    is_safety_critical: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    requires_human_review: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    is_resolved: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    reported_by_user_id: Mapped[
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

    resolution_notes: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    reported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    resolved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    delivery: Mapped["ReliefDelivery"] = relationship(
        "ReliefDelivery",
        back_populates="exceptions",
    )

    __table_args__ = (
        Index(
            "ix_relief_exception_type_time",
            "exception_type",
            "reported_at",
        ),
    )


# ============================================================
# DELIVERY EVENT HISTORY
# ============================================================

class ReliefDeliveryEvent(Base):
    __tablename__ = "relief_delivery_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    delivery_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "relief_deliveries.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        ReliefDeliveryEventType
    ] = mapped_column(
        enum_type(ReliefDeliveryEventType),
        nullable=False,
    )

    previous_status: Mapped[
        ReliefDeliveryStatus | None
    ] = mapped_column(
        enum_type(ReliefDeliveryStatus),
        nullable=True,
    )

    new_status: Mapped[
        ReliefDeliveryStatus | None
    ] = mapped_column(
        enum_type(ReliefDeliveryStatus),
        nullable=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    actor_agent_name: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    audit_reference: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    occurred_at: Mapped[datetime] = mapped_column(
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

    delivery: Mapped["ReliefDelivery"] = relationship(
        "ReliefDelivery",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_relief_event_delivery_time",
            "delivery_id",
            "occurred_at",
        ),
    )


# ============================================================
# ORM TIMESTAMP EVENT
# ============================================================

@event.listens_for(
    ReliefDelivery,
    "before_update",
)
def update_delivery_timestamp(
    mapper: Any,
    connection: Any,
    target: ReliefDelivery,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

RELIEF_DELIVERY_MODELS = (
    ReliefDelivery,
    ReliefDeliveryItem,
    ReliefDeliveryCheckpoint,
    ReliefDeliveryProof,
    ReliefDeliveryException,
    ReliefDeliveryEvent,
)


__all__ = [
    "ReliefDelivery",
    "ReliefDeliveryItem",
    "ReliefDeliveryCheckpoint",
    "ReliefDeliveryProof",
    "ReliefDeliveryException",
    "ReliefDeliveryEvent",
    "ReliefDeliveryStatus",
    "ReliefDeliveryPriority",
    "ReliefDeliveryType",
    "ReliefTransportMode",
    "ReliefRouteStatus",
    "ReliefCheckpointStatus",
    "ReliefDeliveryItemStatus",
    "ReliefProofType",
    "ReliefProofStatus",
    "ReliefDeliveryExceptionType",
    "ReliefDeliveryEventType",
    "RELIEF_DELIVERY_MODELS",
]
