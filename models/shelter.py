from __future__ import annotations

"""
Shelter management models for Kerala Flood Relief Coordination Agent.

SQLAlchemy 2.x | Python 3.11+

Features:
- Shelter registration and verification
- Shelter capacity and occupancy
- Room and bed management
- Accessibility and medical facilities
- Food, water, sanitation and power availability
- Emergency shelter operations
- Safety inspections
- Shelter resource requirements
- Occupancy snapshots
- Shelter alerts and incidents
- Human-approved activation and closure
- Audit events

AI recommendations are advisory. Official shelter activation,
evacuation, admission, and closure decisions require authorized
human review and application-level access control.
"""


import enum
import uuid

from datetime import datetime, timezone
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
    String,
    Text,
    UniqueConstraint,
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


def enum_type(cls: type[enum.Enum]) -> SQLEnum:
    return SQLEnum(
        cls,
        values_callable=lambda members: [
            member.value for member in members
        ],
        native_enum=False,
        validate_strings=True,
        length=80,
    )


def require_text(
    value: str,
    field: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")

    result = value.strip()

    if len(result) > maximum:
        raise ValueError(
            f"{field} exceeds {maximum} characters"
        )

    return result


def require_user_id(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError("Valid authorized user ID required")
    return value


# ============================================================
# ENUMS
# ============================================================

class ShelterType(str, enum.Enum):
    GOVERNMENT = "government"
    SCHOOL = "school"
    COLLEGE = "college"
    COMMUNITY_HALL = "community_hall"
    AUDITORIUM = "auditorium"
    RELIGIOUS_FACILITY = "religious_facility"
    SPORTS_COMPLEX = "sports_complex"
    TEMPORARY_CAMP = "temporary_camp"
    MEDICAL_SHELTER = "medical_shelter"
    NGO = "ngo"
    PRIVATE_AUTHORIZED = "private_authorized"
    OTHER = "other"


class ShelterStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    READY = "ready"
    OPEN = "open"
    FULL = "full"
    RESTRICTED = "restricted"
    SUSPENDED = "suspended"
    CLOSING = "closing"
    CLOSED = "closed"
    DECOMMISSIONED = "decommissioned"


class ShelterSafetyStatus(str, enum.Enum):
    NOT_INSPECTED = "not_inspected"
    SAFE = "safe"
    CONDITIONAL = "conditional"
    UNSAFE = "unsafe"
    REINSPECTION_REQUIRED = "reinspection_required"


class ShelterFacilityStatus(str, enum.Enum):
    AVAILABLE = "available"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"
    UNDER_REPAIR = "under_repair"
    UNKNOWN = "unknown"


class ShelterFacilityType(str, enum.Enum):
    DRINKING_WATER = "drinking_water"
    FOOD_KITCHEN = "food_kitchen"
    TOILET = "toilet"
    BATHROOM = "bathroom"
    HANDWASHING = "handwashing"
    ELECTRICITY = "electricity"
    GENERATOR = "generator"
    INTERNET = "internet"
    MOBILE_NETWORK = "mobile_network"
    MEDICAL_ROOM = "medical_room"
    FIRST_AID = "first_aid"
    ISOLATION_ROOM = "isolation_room"
    MATERNAL_CARE = "maternal_care"
    INFANT_CARE = "infant_care"
    ACCESSIBLE_TOILET = "accessible_toilet"
    WHEELCHAIR_ACCESS = "wheelchair_access"
    RAMP = "ramp"
    SECURITY = "security"
    FIRE_SAFETY = "fire_safety"
    WASTE_MANAGEMENT = "waste_management"
    LAUNDRY = "laundry"
    SLEEPING_AREA = "sleeping_area"
    OTHER = "other"


class ShelterRoomType(str, enum.Enum):
    GENERAL = "general"
    FAMILY = "family"
    WOMEN = "women"
    CHILDREN = "children"
    ELDERLY = "elderly"
    ACCESSIBLE = "accessible"
    MEDICAL = "medical"
    ISOLATION = "isolation"
    STAFF = "staff"
    STORAGE = "storage"
    OTHER = "other"


class ShelterRoomStatus(str, enum.Enum):
    AVAILABLE = "available"
    PARTIALLY_OCCUPIED = "partially_occupied"
    FULL = "full"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"
    CLOSED = "closed"


class ShelterBedStatus(str, enum.Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    CLEANING = "cleaning"
    DAMAGED = "damaged"
    UNAVAILABLE = "unavailable"


class ShelterResourcePriority(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ShelterResourceStatus(str, enum.Enum):
    REQUESTED = "requested"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    PARTIALLY_FULFILLED = "partially_fulfilled"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"


class ShelterAlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class ShelterAlertStatus(str, enum.Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ShelterInspectionResult(str, enum.Enum):
    PASS = "pass"
    CONDITIONAL = "conditional"
    FAIL = "fail"


class ShelterEventType(str, enum.Enum):
    CREATED = "created"
    VERIFIED = "verified"
    INSPECTED = "inspected"
    ACTIVATED = "activated"
    OPENED = "opened"
    CAPACITY_UPDATED = "capacity_updated"
    OCCUPANCY_UPDATED = "occupancy_updated"
    FACILITY_UPDATED = "facility_updated"
    RESOURCE_REQUESTED = "resource_requested"
    ALERT_CREATED = "alert_created"
    ALERT_RESOLVED = "alert_resolved"
    RESTRICTED = "restricted"
    SUSPENDED = "suspended"
    CLOSURE_REQUESTED = "closure_requested"
    CLOSED = "closed"
    REOPENED = "reopened"
    NOTE_ADDED = "note_added"


# ============================================================
# SHELTER
# ============================================================

class Shelter(Base):
    __tablename__ = "shelters"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    shelter_code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    shelter_type: Mapped[ShelterType] = mapped_column(
        enum_type(ShelterType),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
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

    taluk: Mapped[str | None] = mapped_column(
        String(150),
    )

    village: Mapped[str | None] = mapped_column(
        String(150),
    )

    panchayat: Mapped[str | None] = mapped_column(
        String(150),
    )

    ward: Mapped[str | None] = mapped_column(
        String(100),
    )

    address: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    pincode: Mapped[str | None] = mapped_column(
        String(10),
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
    )

    elevation_m: Mapped[float | None] = mapped_column(
        Float,
    )

    managing_organization: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    manager_user_id: Mapped[int | None] = mapped_column(
        Integer,
    )

    contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    emergency_contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    status: Mapped[ShelterStatus] = mapped_column(
        enum_type(ShelterStatus),
        default=ShelterStatus.DRAFT,
        nullable=False,
        index=True,
    )

    safety_status: Mapped[
        ShelterSafetyStatus
    ] = mapped_column(
        enum_type(ShelterSafetyStatus),
        default=ShelterSafetyStatus.NOT_INSPECTED,
        nullable=False,
    )

    maximum_capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    operational_capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    current_occupancy: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    reserved_capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_rooms: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_beds: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    available_beds: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    accessible_capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    medical_capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    has_drinking_water: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_food_service: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_electricity: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_sanitation: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_medical_support: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_accessibility_support: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_backup_power: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_emergency_exit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    flood_risk_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
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
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    safety_inspected_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    safety_inspected_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    safety_valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    activated_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    activated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    opened_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    closed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    closure_reason: Mapped[
        str | None
    ] = mapped_column(
        Text,
    )

    closed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    created_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
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

    rooms: Mapped[list["ShelterRoom"]] = relationship(
        "ShelterRoom",
        back_populates="shelter",
        cascade="save-update, merge",
    )

    facilities: Mapped[
        list["ShelterFacility"]
    ] = relationship(
        "ShelterFacility",
        back_populates="shelter",
        cascade="save-update, merge",
    )

    occupancy_snapshots: Mapped[
        list["ShelterOccupancySnapshot"]
    ] = relationship(
        "ShelterOccupancySnapshot",
        back_populates="shelter",
    )

    inspections: Mapped[
        list["ShelterSafetyInspection"]
    ] = relationship(
        "ShelterSafetyInspection",
        back_populates="shelter",
    )

    resource_requests: Mapped[
        list["ShelterResourceRequest"]
    ] = relationship(
        "ShelterResourceRequest",
        back_populates="shelter",
    )

    alerts: Mapped[list["ShelterAlert"]] = relationship(
        "ShelterAlert",
        back_populates="shelter",
    )

    events: Mapped[list["ShelterEvent"]] = relationship(
        "ShelterEvent",
        back_populates="shelter",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_shelter_district",
        ),
        CheckConstraint(
            "maximum_capacity >= 0",
            name="ck_shelter_maximum_capacity",
        ),
        CheckConstraint(
            "operational_capacity >= 0 AND "
            "operational_capacity <= maximum_capacity",
            name="ck_shelter_operational_capacity",
        ),
        CheckConstraint(
            "current_occupancy >= 0",
            name="ck_shelter_occupancy",
        ),
        CheckConstraint(
            "reserved_capacity >= 0",
            name="ck_shelter_reserved",
        ),
        CheckConstraint(
            "total_rooms >= 0 AND total_beds >= 0",
            name="ck_shelter_room_bed_counts",
        ),
        CheckConstraint(
            "available_beds >= 0 AND "
            "available_beds <= total_beds",
            name="ck_shelter_available_beds",
        ),
        CheckConstraint(
            "accessible_capacity >= 0 AND "
            "medical_capacity >= 0",
            name="ck_shelter_special_capacity",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude BETWEEN -90 AND 90)",
            name="ck_shelter_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude BETWEEN -180 AND 180)",
            name="ck_shelter_longitude",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_shelter_version",
        ),
        Index(
            "ix_shelter_district_status",
            "district_id",
            "status",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("shelter_code", "name", "address")
    def validate_required(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "shelter_code": 100,
            "name": 255,
            "address": 500,
        }
        return require_text(value, key, limits[key])

    @validates("latitude", "longitude")
    def validate_coordinates(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        if value is None:
            return None

        import math

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid {key}")

        lower, upper = (
            (-90, 90)
            if key == "latitude"
            else (-180, 180)
        )

        if not lower <= value <= upper:
            raise ValueError(f"Invalid {key}")

        return float(value)

    @property
    def available_capacity(self) -> int:
        return max(
            0,
            self.operational_capacity
            - self.current_occupancy
            - self.reserved_capacity,
        )

    @property
    def occupancy_percentage(self) -> float:
        if self.operational_capacity <= 0:
            return 0.0

        return round(
            (
                self.current_occupancy
                / self.operational_capacity
            ) * 100,
            2,
        )

    @property
    def has_valid_safety_clearance(self) -> bool:
        return (
            self.safety_status == ShelterSafetyStatus.SAFE
            and self.safety_valid_until is not None
            and (
                self.safety_valid_until.replace(
                    tzinfo=timezone.utc
                )
                if self.safety_valid_until.tzinfo is None
                else self.safety_valid_until
            ) > utc_now()
        )

    @property
    def can_accept_residents(self) -> bool:
        return (
            self.status == ShelterStatus.OPEN
            and self.is_verified
            and self.has_valid_safety_clearance
            and self.flood_risk_verified
            and self.available_capacity > 0
        )

    def submit_for_verification(self) -> None:
        if self.status != ShelterStatus.DRAFT:
            raise ValueError(
                "Only draft shelters can be submitted"
            )

        self.status = ShelterStatus.PENDING_VERIFICATION

    def verify(self, verifier_id: int) -> None:
        require_user_id(verifier_id)

        if self.status != (
            ShelterStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Shelter is not pending verification"
            )

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
        self.status = ShelterStatus.VERIFIED

    def activate(self, authorized_user_id: int) -> None:
        require_user_id(authorized_user_id)

        if self.status != ShelterStatus.VERIFIED:
            raise ValueError(
                "Shelter must be verified"
            )

        if not self.has_valid_safety_clearance:
            raise ValueError(
                "Valid safety clearance required"
            )

        if not self.flood_risk_verified:
            raise ValueError(
                "Flood risk verification required"
            )

        if self.operational_capacity <= 0:
            raise ValueError(
                "Operational capacity must be positive"
            )

        self.activated_by_user_id = authorized_user_id
        self.activated_at = utc_now()
        self.status = ShelterStatus.READY

    def open(self, authorized_user_id: int) -> None:
        require_user_id(authorized_user_id)

        if self.status != ShelterStatus.READY:
            raise ValueError(
                "Shelter is not ready"
            )

        if not self.has_valid_safety_clearance:
            raise ValueError(
                "Safety clearance expired"
            )

        self.status = ShelterStatus.OPEN
        self.opened_at = utc_now()

    def update_occupancy(
        self,
        occupancy: int,
    ) -> None:
        if occupancy < 0:
            raise ValueError(
                "Occupancy cannot be negative"
            )

        if (
            occupancy + self.reserved_capacity
            > self.operational_capacity
        ):
            raise ValueError(
                "Occupancy exceeds operational capacity"
            )

        self.current_occupancy = occupancy

        if self.status in {
            ShelterStatus.OPEN,
            ShelterStatus.FULL,
        }:
            self.status = (
                ShelterStatus.FULL
                if self.available_capacity == 0
                else ShelterStatus.OPEN
            )

    def request_closure(self) -> None:
        if self.status not in {
            ShelterStatus.OPEN,
            ShelterStatus.FULL,
            ShelterStatus.RESTRICTED,
        }:
            raise ValueError(
                "Shelter cannot enter closing state"
            )

        self.status = ShelterStatus.CLOSING

    def close(
        self,
        *,
        authorized_user_id: int,
        reason: str,
    ) -> None:
        require_user_id(authorized_user_id)

        if self.status != ShelterStatus.CLOSING:
            raise ValueError(
                "Closure must be requested first"
            )

        if self.current_occupancy > 0:
            raise ValueError(
                "Residents must be safely transferred "
                "or discharged before closure"
            )

        self.closure_reason = require_text(
            reason, "reason", 5000
        )
        self.closed_by_user_id = authorized_user_id
        self.closed_at = utc_now()
        self.status = ShelterStatus.CLOSED

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "shelter_code": self.shelter_code,
            "name": self.name,
            "shelter_type": self.shelter_type.value,
            "district_id": self.district_id,
            "status": self.status.value,
            "safety_status": self.safety_status.value,
            "operational_capacity": self.operational_capacity,
            "current_occupancy": self.current_occupancy,
            "available_capacity": self.available_capacity,
            "occupancy_percentage": self.occupancy_percentage,
            "can_accept_residents": self.can_accept_residents,
        }


# ============================================================
# SHELTER ROOM
# ============================================================

class ShelterRoom(Base):
    __tablename__ = "shelter_rooms"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelters.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    room_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    room_name: Mapped[str | None] = mapped_column(
        String(150),
    )

    room_type: Mapped[ShelterRoomType] = mapped_column(
        enum_type(ShelterRoomType),
        default=ShelterRoomType.GENERAL,
        nullable=False,
    )

    floor_number: Mapped[int | None] = mapped_column(
        Integer,
    )

    capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    current_occupancy: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    status: Mapped[ShelterRoomStatus] = mapped_column(
        enum_type(ShelterRoomStatus),
        default=ShelterRoomStatus.AVAILABLE,
        nullable=False,
    )

    is_accessible: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_ventilation: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
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

    shelter: Mapped["Shelter"] = relationship(
        "Shelter",
        back_populates="rooms",
    )

    beds: Mapped[list["ShelterBed"]] = relationship(
        "ShelterBed",
        back_populates="room",
    )

    __table_args__ = (
        UniqueConstraint(
            "shelter_id",
            "room_code",
            name="uq_shelter_room_code",
        ),
        CheckConstraint(
            "capacity >= 0",
            name="ck_shelter_room_capacity",
        ),
        CheckConstraint(
            "current_occupancy >= 0 AND "
            "current_occupancy <= capacity",
            name="ck_shelter_room_occupancy",
        ),
    )


# ============================================================
# SHELTER BED
# ============================================================

class ShelterBed(Base):
    __tablename__ = "shelter_beds"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    room_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelter_rooms.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    bed_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    status: Mapped[ShelterBedStatus] = mapped_column(
        enum_type(ShelterBedStatus),
        default=ShelterBedStatus.AVAILABLE,
        nullable=False,
    )

    is_accessible: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    resident_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    reserved_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    assigned_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    assigned_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
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

    room: Mapped["ShelterRoom"] = relationship(
        "ShelterRoom",
        back_populates="beds",
    )

    __table_args__ = (
        UniqueConstraint(
            "room_id",
            "bed_code",
            name="uq_shelter_bed_code",
        ),
    )

    def assign(
        self,
        resident_id: str,
        authorized_user_id: int,
    ) -> None:
        require_user_id(authorized_user_id)

        if self.status != ShelterBedStatus.AVAILABLE:
            raise ValueError("Bed is not available")

        self.resident_id = require_text(
            resident_id,
            "resident_id",
            36,
        )
        self.assigned_by_user_id = authorized_user_id
        self.assigned_at = utc_now()
        self.status = ShelterBedStatus.OCCUPIED

    def release(self) -> None:
        if self.status != ShelterBedStatus.OCCUPIED:
            raise ValueError("Bed is not occupied")

        self.resident_id = None
        self.status = ShelterBedStatus.CLEANING


# ============================================================
# SHELTER FACILITY
# ============================================================

class ShelterFacility(Base):
    __tablename__ = "shelter_facilities"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelters.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    facility_type: Mapped[
        ShelterFacilityType
    ] = mapped_column(
        enum_type(ShelterFacilityType),
        nullable=False,
    )

    facility_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    status: Mapped[
        ShelterFacilityStatus
    ] = mapped_column(
        enum_type(ShelterFacilityStatus),
        default=ShelterFacilityStatus.UNKNOWN,
        nullable=False,
    )

    total_units: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    operational_units: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    last_inspected_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
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

    shelter: Mapped["Shelter"] = relationship(
        "Shelter",
        back_populates="facilities",
    )

    __table_args__ = (
        CheckConstraint(
            "total_units >= 0",
            name="ck_shelter_facility_total",
        ),
        CheckConstraint(
            "operational_units >= 0 AND "
            "operational_units <= total_units",
            name="ck_shelter_facility_operational",
        ),
    )


# ============================================================
# OCCUPANCY SNAPSHOT
# ============================================================

class ShelterOccupancySnapshot(Base):
    __tablename__ = "shelter_occupancy_snapshots"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelters.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    operational_capacity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    total_residents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    adults: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    children: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    elderly: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    residents_needing_assistance: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    available_beds: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    recorded_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    recorded_at: Mapped[datetime] = mapped_column(
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

    shelter: Mapped["Shelter"] = relationship(
        "Shelter",
        back_populates="occupancy_snapshots",
    )

    __table_args__ = (
        CheckConstraint(
            "operational_capacity >= 0 AND "
            "total_residents >= 0 AND "
            "adults >= 0 AND children >= 0 AND "
            "elderly >= 0 AND "
            "residents_needing_assistance >= 0 AND "
            "available_beds >= 0",
            name="ck_shelter_snapshot_nonnegative",
        ),
        Index(
            "ix_shelter_snapshot_time",
            "shelter_id",
            "recorded_at",
        ),
    )


# ============================================================
# SAFETY INSPECTION
# ============================================================

class ShelterSafetyInspection(Base):
    __tablename__ = "shelter_safety_inspections"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelters.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    inspection_reference: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    result: Mapped[
        ShelterInspectionResult
    ] = mapped_column(
        enum_type(ShelterInspectionResult),
        nullable=False,
    )

    structural_safety_passed: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    flood_safety_passed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    fire_safety_passed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    sanitation_passed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    emergency_exit_passed: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    findings: Mapped[str | None] = mapped_column(
        Text,
    )

    corrective_actions: Mapped[
        str | None
    ] = mapped_column(
        Text,
    )

    inspector_user_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    inspected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    report_storage_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
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

    shelter: Mapped["Shelter"] = relationship(
        "Shelter",
        back_populates="inspections",
    )

    __table_args__ = (
        CheckConstraint(
            "inspector_user_id > 0",
            name="ck_shelter_inspector",
        ),
    )


# ============================================================
# RESOURCE REQUEST
# ============================================================

class ShelterResourceRequest(Base):
    __tablename__ = "shelter_resource_requests"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelters.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    request_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    resource_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    resource_category: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    quantity_requested: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    quantity_fulfilled: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    unit: Mapped[str] = mapped_column(
        String(50),
        default="units",
        nullable=False,
    )

    priority: Mapped[
        ShelterResourcePriority
    ] = mapped_column(
        enum_type(ShelterResourcePriority),
        default=ShelterResourcePriority.NORMAL,
        nullable=False,
    )

    status: Mapped[
        ShelterResourceStatus
    ] = mapped_column(
        enum_type(ShelterResourceStatus),
        default=ShelterResourceStatus.REQUESTED,
        nullable=False,
    )

    reason: Mapped[str | None] = mapped_column(
        Text,
    )

    inventory_request_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    requested_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    fulfilled_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    shelter: Mapped["Shelter"] = relationship(
        "Shelter",
        back_populates="resource_requests",
    )

    __table_args__ = (
        CheckConstraint(
            "quantity_requested > 0",
            name="ck_shelter_resource_requested",
        ),
        CheckConstraint(
            "quantity_fulfilled >= 0 AND "
            "quantity_fulfilled <= quantity_requested",
            name="ck_shelter_resource_fulfilled",
        ),
    )


# ============================================================
# SHELTER ALERT
# ============================================================

class ShelterAlert(Base):
    __tablename__ = "shelter_alerts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelters.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    alert_code: Mapped[str] = mapped_column(
        String(120),
        unique=True,
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

    severity: Mapped[
        ShelterAlertSeverity
    ] = mapped_column(
        enum_type(ShelterAlertSeverity),
        nullable=False,
    )

    status: Mapped[
        ShelterAlertStatus
    ] = mapped_column(
        enum_type(ShelterAlertStatus),
        default=ShelterAlertStatus.OPEN,
        nullable=False,
    )

    is_ai_detected: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_evacuation_review: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    reported_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    acknowledged_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    resolved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    acknowledged_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    resolved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    resolution_notes: Mapped[
        str | None
    ] = mapped_column(
        Text,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    shelter: Mapped["Shelter"] = relationship(
        "Shelter",
        back_populates="alerts",
    )

    __table_args__ = (
        Index(
            "ix_shelter_alert_status_severity",
            "status",
            "severity",
        ),
    )


# ============================================================
# SHELTER AUDIT EVENTS
# ============================================================

class ShelterEvent(Base):
    __tablename__ = "shelter_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelters.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        ShelterEventType
    ] = mapped_column(
        enum_type(ShelterEventType),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    previous_status: Mapped[
        ShelterStatus | None
    ] = mapped_column(
        enum_type(ShelterStatus),
    )

    new_status: Mapped[
        ShelterStatus | None
    ] = mapped_column(
        enum_type(ShelterStatus),
    )

    actor_user_id: Mapped[int | None] = mapped_column(
        Integer,
    )

    actor_agent_name: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
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

    shelter: Mapped["Shelter"] = relationship(
        "Shelter",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_shelter_event_time",
            "shelter_id",
            "occurred_at",
        ),
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

SHELTER_MODELS = (
    Shelter,
    ShelterRoom,
    ShelterBed,
    ShelterFacility,
    ShelterOccupancySnapshot,
    ShelterSafetyInspection,
    ShelterResourceRequest,
    ShelterAlert,
    ShelterEvent,
)


__all__ = [
    "Shelter",
    "ShelterRoom",
    "ShelterBed",
    "ShelterFacility",
    "ShelterOccupancySnapshot",
    "ShelterSafetyInspection",
    "ShelterResourceRequest",
    "ShelterAlert",
    "ShelterEvent",
    "ShelterType",
    "ShelterStatus",
    "ShelterSafetyStatus",
    "ShelterFacilityStatus",
    "ShelterFacilityType",
    "ShelterRoomType",
    "ShelterRoomStatus",
    "ShelterBedStatus",
    "ShelterResourcePriority",
    "ShelterResourceStatus",
    "ShelterAlertSeverity",
    "ShelterAlertStatus",
    "ShelterInspectionResult",
    "ShelterEventType",
    "SHELTER_MODELS",
]
