from __future__ import annotations

"""
backend/models/hospital.py

Kerala Flood Relief Coordination Agent

SQLAlchemy 2.x models for:
- Hospital registration and verification
- Emergency medical facility availability
- ICU, emergency and general bed capacity
- Ambulance availability
- Medical equipment and supply status
- Flood-related hospital operational risks
- Hospital capacity snapshots
- Patient transfer coordination
- Human-reviewed medical resource requests
- Audit-friendly status history

SAFETY:
This module stores operational information only.
It does not diagnose patients, prescribe treatment,
dispatch ambulances, or guarantee bed availability.

Capacity information may become outdated.
Patient transfers require receiving-facility
confirmation and authorized medical coordination.
"""


import enum
import math
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
    max_length: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise ValueError(
            f"{field_name} exceeds maximum length"
        )

    return result


def validate_non_negative(
    value: int | None,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a non-negative integer"
        )

    return value


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


def normalized_utc(value: datetime) -> datetime:
    """
    SQL databases such as SQLite may return naive
    timestamps even for timezone-aware columns.

    Application policy: naive database timestamps
    are interpreted as UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class HospitalType(str, enum.Enum):
    GOVERNMENT = "government"
    PRIVATE = "private"
    DISTRICT = "district"
    GENERAL = "general"
    MEDICAL_COLLEGE = "medical_college"
    COMMUNITY_HEALTH_CENTRE = "community_health_centre"
    PRIMARY_HEALTH_CENTRE = "primary_health_centre"
    SPECIALITY = "speciality"
    TEMPORARY_FIELD_HOSPITAL = "temporary_field_hospital"
    OTHER = "other"


class HospitalOperationalStatus(str, enum.Enum):
    OPERATIONAL = "operational"
    LIMITED_CAPACITY = "limited_capacity"
    EMERGENCY_ONLY = "emergency_only"
    EVACUATING = "evacuating"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class HospitalFloodRiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HospitalDataQuality(str, enum.Enum):
    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    SUSPECT = "suspect"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class HospitalBedType(str, enum.Enum):
    GENERAL = "general"
    EMERGENCY = "emergency"
    ICU = "icu"
    HDU = "hdu"
    PEDIATRIC = "pediatric"
    MATERNITY = "maternity"
    ISOLATION = "isolation"
    OBSERVATION = "observation"
    OTHER = "other"


class HospitalResourceType(str, enum.Enum):
    AMBULANCE = "ambulance"
    OXYGEN_CYLINDER = "oxygen_cylinder"
    OXYGEN_CONCENTRATOR = "oxygen_concentrator"
    VENTILATOR = "ventilator"
    DEFIBRILLATOR = "defibrillator"
    STRETCHER = "stretcher"
    WHEELCHAIR = "wheelchair"
    FIRST_AID_KIT = "first_aid_kit"
    PPE_KIT = "ppe_kit"
    IV_FLUID = "iv_fluid"
    ESSENTIAL_MEDICINE = "essential_medicine"
    GENERATOR = "generator"
    WATER_SUPPLY = "water_supply"
    MEDICAL_TEAM = "medical_team"
    OTHER = "other"


class HospitalTransferStatus(str, enum.Enum):
    REQUESTED = "requested"
    UNDER_REVIEW = "under_review"
    PENDING_ACCEPTANCE = "pending_acceptance"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TRANSPORT_ARRANGED = "transport_arranged"
    IN_TRANSIT = "in_transit"
    ARRIVED = "arrived"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class HospitalTransferPriority(str, enum.Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


class HospitalStatusSource(str, enum.Enum):
    HOSPITAL_ADMIN = "hospital_admin"
    DISTRICT_CONTROL_ROOM = "district_control_room"
    HEALTH_DEPARTMENT = "health_department"
    VERIFIED_FIELD_REPORT = "verified_field_report"
    INTEGRATED_SYSTEM = "integrated_system"
    MANUAL_ENTRY = "manual_entry"


# ============================================================
# HOSPITAL
# ============================================================

class Hospital(Base):
    __tablename__ = "hospitals"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    hospital_code: Mapped[str] = mapped_column(
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

    hospital_type: Mapped[HospitalType] = mapped_column(
        enum_type(HospitalType),
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

    taluk: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    panchayat: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    address: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    pincode: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    public_phone: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    emergency_phone: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    official_email: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    official_website: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    registration_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    operating_authority: Mapped[str | None] = (
        mapped_column(
            String(200),
            nullable=True,
        )
    )

    operational_status: Mapped[
        HospitalOperationalStatus
    ] = mapped_column(
        enum_type(HospitalOperationalStatus),
        default=HospitalOperationalStatus.UNKNOWN,
        nullable=False,
        index=True,
    )

    flood_risk_level: Mapped[
        HospitalFloodRiskLevel
    ] = mapped_column(
        enum_type(HospitalFloodRiskLevel),
        default=HospitalFloodRiskLevel.UNKNOWN,
        nullable=False,
    )

    emergency_department_available: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    trauma_care_available: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    icu_available: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    pediatric_care_available: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    maternity_care_available: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    dialysis_available: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    oxygen_supply_available: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    backup_power_available: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    safe_drinking_water_available: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    road_accessible: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    accepting_emergency_patients: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    verified_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    source_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    last_status_confirmed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    bed_snapshots: Mapped[
        list["HospitalBedSnapshot"]
    ] = relationship(
        "HospitalBedSnapshot",
        back_populates="hospital",
        cascade="save-update, merge",
    )

    resource_snapshots: Mapped[
        list["HospitalResourceSnapshot"]
    ] = relationship(
        "HospitalResourceSnapshot",
        back_populates="hospital",
        cascade="save-update, merge",
    )

    status_history: Mapped[
        list["HospitalStatusHistory"]
    ] = relationship(
        "HospitalStatusHistory",
        back_populates="hospital",
        cascade="save-update, merge",
    )

    outgoing_transfers: Mapped[
        list["HospitalPatientTransfer"]
    ] = relationship(
        "HospitalPatientTransfer",
        foreign_keys=(
            "HospitalPatientTransfer.source_hospital_id"
        ),
        back_populates="source_hospital",
    )

    incoming_transfers: Mapped[
        list["HospitalPatientTransfer"]
    ] = relationship(
        "HospitalPatientTransfer",
        foreign_keys=(
            "HospitalPatientTransfer.destination_hospital_id"
        ),
        back_populates="destination_hospital",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_hospital_district",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_hospital_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_hospital_longitude",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_hospital_version",
        ),
        Index(
            "ix_hospital_district_status",
            "district_id",
            "operational_status",
        ),
        Index(
            "ix_hospital_emergency",
            "district_id",
            "accepting_emergency_patients",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("name", "hospital_code", "address")
    def validate_required_field(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "name": 255,
            "hospital_code": 100,
            "address": 500,
        }

        return require_text(
            value,
            key,
            limits[key],
        )

    @validates("latitude")
    def validate_latitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -90, 90
        )

    @validates("longitude")
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -180, 180
        )

    @property
    def is_operational(self) -> bool:
        return (
            self.is_active
            and self.operational_status in {
                HospitalOperationalStatus.OPERATIONAL,
                HospitalOperationalStatus.LIMITED_CAPACITY,
                HospitalOperationalStatus.EMERGENCY_ONLY,
            }
        )

    def status_is_stale(
        self,
        max_age_minutes: int = 30,
    ) -> bool:
        if max_age_minutes <= 0:
            raise ValueError(
                "max_age_minutes must be positive"
            )

        if self.last_status_confirmed_at is None:
            return True

        age_seconds = (
            utc_now()
            - normalized_utc(
                self.last_status_confirmed_at
            )
        ).total_seconds()

        return age_seconds > max_age_minutes * 60

    def can_be_considered_for_emergency_referral(
        self,
    ) -> bool:
        return (
            self.is_verified
            and self.is_operational
            and self.accepting_emergency_patients
            and self.emergency_department_available
            and self.road_accessible is True
            and not self.status_is_stale()
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hospital_code": self.hospital_code,
            "name": self.name,
            "hospital_type": self.hospital_type.value,
            "district_id": self.district_id,
            "operational_status": (
                self.operational_status.value
            ),
            "flood_risk_level": (
                self.flood_risk_level.value
            ),
            "is_verified": self.is_verified,
            "is_active": self.is_active,
            "accepting_emergency_patients": (
                self.accepting_emergency_patients
            ),
            "last_status_confirmed_at": (
                self.last_status_confirmed_at.isoformat()
                if self.last_status_confirmed_at
                else None
            ),
        }


# ============================================================
# HOSPITAL BED CAPACITY SNAPSHOT
# ============================================================

class HospitalBedSnapshot(Base):
    __tablename__ = "hospital_bed_snapshots"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    hospital_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "hospitals.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    bed_type: Mapped[HospitalBedType] = mapped_column(
        enum_type(HospitalBedType),
        nullable=False,
    )

    total_beds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    occupied_beds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    reserved_beds: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    unavailable_beds: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    data_quality: Mapped[
        HospitalDataQuality
    ] = mapped_column(
        enum_type(HospitalDataQuality),
        default=HospitalDataQuality.UNKNOWN,
        nullable=False,
    )

    source_type: Mapped[
        HospitalStatusSource
    ] = mapped_column(
        enum_type(HospitalStatusSource),
        nullable=False,
    )

    source_reference: Mapped[str | None] = (
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

    verified_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
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

    hospital: Mapped["Hospital"] = relationship(
        "Hospital",
        back_populates="bed_snapshots",
    )

    __table_args__ = (
        CheckConstraint(
            "total_beds >= 0",
            name="ck_hospital_bed_total",
        ),
        CheckConstraint(
            "occupied_beds >= 0",
            name="ck_hospital_bed_occupied",
        ),
        CheckConstraint(
            "reserved_beds >= 0",
            name="ck_hospital_bed_reserved",
        ),
        CheckConstraint(
            "unavailable_beds >= 0",
            name="ck_hospital_bed_unavailable",
        ),
        CheckConstraint(
            "occupied_beds + reserved_beds + "
            "unavailable_beds <= total_beds",
            name="ck_hospital_bed_capacity",
        ),
        Index(
            "ix_hospital_bed_type_time",
            "hospital_id",
            "bed_type",
            "observed_at",
        ),
    )

    @validates(
        "total_beds",
        "occupied_beds",
        "reserved_beds",
        "unavailable_beds",
    )
    def validate_count(
        self,
        key: str,
        value: int,
    ) -> int:
        result = validate_non_negative(
            value, key
        )

        if result is None:
            raise ValueError(
                f"{key} is required"
            )

        return result

    @property
    def available_beds(self) -> int:
        return max(
            self.total_beds
            - self.occupied_beds
            - self.reserved_beds
            - self.unavailable_beds,
            0,
        )

    @property
    def occupancy_percentage(self) -> float:
        if self.total_beds == 0:
            return 0.0

        return round(
            self.occupied_beds
            / self.total_beds
            * 100,
            2,
        )

    def is_stale(
        self,
        max_age_minutes: int = 30,
    ) -> bool:
        if max_age_minutes <= 0:
            raise ValueError(
                "max_age_minutes must be positive"
            )

        age_seconds = (
            utc_now()
            - normalized_utc(self.observed_at)
        ).total_seconds()

        return age_seconds > max_age_minutes * 60

    @property
    def is_usable(self) -> bool:
        return (
            self.is_verified
            and self.data_quality
            == HospitalDataQuality.VERIFIED
            and not self.is_stale()
        )


# ============================================================
# HOSPITAL RESOURCE SNAPSHOT
# ============================================================

class HospitalResourceSnapshot(Base):
    __tablename__ = "hospital_resource_snapshots"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    hospital_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "hospitals.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    resource_type: Mapped[
        HospitalResourceType
    ] = mapped_column(
        enum_type(HospitalResourceType),
        nullable=False,
    )

    resource_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    total_quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    available_quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    allocated_quantity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    unavailable_quantity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    unit: Mapped[str] = mapped_column(
        String(50),
        default="units",
        nullable=False,
    )

    data_quality: Mapped[
        HospitalDataQuality
    ] = mapped_column(
        enum_type(HospitalDataQuality),
        default=HospitalDataQuality.UNKNOWN,
        nullable=False,
    )

    source_type: Mapped[
        HospitalStatusSource
    ] = mapped_column(
        enum_type(HospitalStatusSource),
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    observed_at: Mapped[datetime] = mapped_column(
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

    hospital: Mapped["Hospital"] = relationship(
        "Hospital",
        back_populates="resource_snapshots",
    )

    __table_args__ = (
        CheckConstraint(
            "total_quantity >= 0",
            name="ck_hospital_resource_total",
        ),
        CheckConstraint(
            "available_quantity >= 0",
            name="ck_hospital_resource_available",
        ),
        CheckConstraint(
            "allocated_quantity >= 0",
            name="ck_hospital_resource_allocated",
        ),
        CheckConstraint(
            "unavailable_quantity >= 0",
            name="ck_hospital_resource_unavailable",
        ),
        CheckConstraint(
            "available_quantity + allocated_quantity + "
            "unavailable_quantity <= total_quantity",
            name="ck_hospital_resource_capacity",
        ),
        Index(
            "ix_hospital_resource_type_time",
            "hospital_id",
            "resource_type",
            "observed_at",
        ),
    )


# ============================================================
# PATIENT TRANSFER COORDINATION
# ============================================================

class HospitalPatientTransfer(Base):
    __tablename__ = "hospital_patient_transfers"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    transfer_number: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    source_hospital_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            ForeignKey(
                "hospitals.id",
                ondelete="RESTRICT",
            ),
            nullable=True,
        )
    )

    destination_hospital_id: Mapped[str] = (
        mapped_column(
            String(36),
            ForeignKey(
                "hospitals.id",
                ondelete="RESTRICT",
            ),
            nullable=False,
            index=True,
        )
    )

    incident_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    rescue_mission_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            nullable=True,
        )
    )

    patient_reference: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    priority: Mapped[
        HospitalTransferPriority
    ] = mapped_column(
        enum_type(HospitalTransferPriority),
        nullable=False,
    )

    status: Mapped[
        HospitalTransferStatus
    ] = mapped_column(
        enum_type(HospitalTransferStatus),
        default=HospitalTransferStatus.REQUESTED,
        nullable=False,
        index=True,
    )

    requested_bed_type: Mapped[
        HospitalBedType | None
    ] = mapped_column(
        enum_type(HospitalBedType),
        nullable=True,
    )

    requires_oxygen: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_critical_care_transport: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requested_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    accepted_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    receiving_facility_confirmation: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    transport_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    clinical_handover_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    accepted_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    departed_at: Mapped[datetime | None] = (
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

    completed_at: Mapped[datetime | None] = (
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

    source_hospital: Mapped[
        "Hospital | None"
    ] = relationship(
        "Hospital",
        foreign_keys=[source_hospital_id],
        back_populates="outgoing_transfers",
    )

    destination_hospital: Mapped[
        "Hospital"
    ] = relationship(
        "Hospital",
        foreign_keys=[destination_hospital_id],
        back_populates="incoming_transfers",
    )

    __table_args__ = (
        CheckConstraint(
            "source_hospital_id IS NULL OR "
            "source_hospital_id != destination_hospital_id",
            name="ck_hospital_transfer_distinct",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_hospital_transfer_version",
        ),
        Index(
            "ix_hospital_transfer_status_priority",
            "status",
            "priority",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    def accept(
        self,
        *,
        accepting_user_id: int,
        confirmation_reference: str,
    ) -> None:
        if self.status not in {
            HospitalTransferStatus.UNDER_REVIEW,
            HospitalTransferStatus.PENDING_ACCEPTANCE,
        }:
            raise ValueError(
                "Transfer is not awaiting acceptance"
            )

        if accepting_user_id <= 0:
            raise ValueError(
                "Invalid accepting user"
            )

        self.receiving_facility_confirmation = (
            require_text(
                confirmation_reference,
                "confirmation_reference",
                500,
            )
        )

        self.accepted_by_user_id = accepting_user_id
        self.accepted_at = utc_now()
        self.status = HospitalTransferStatus.ACCEPTED

    def mark_transport_arranged(
        self,
        transport_reference: str,
    ) -> None:
        if self.status != HospitalTransferStatus.ACCEPTED:
            raise ValueError(
                "Transfer must be accepted first"
            )

        self.transport_reference = require_text(
            transport_reference,
            "transport_reference",
            150,
        )

        self.status = (
            HospitalTransferStatus.TRANSPORT_ARRANGED
        )

    def mark_in_transit(self) -> None:
        if self.status != (
            HospitalTransferStatus.TRANSPORT_ARRANGED
        ):
            raise ValueError(
                "Transport has not been arranged"
            )

        self.status = HospitalTransferStatus.IN_TRANSIT
        self.departed_at = utc_now()

    def mark_arrived(self) -> None:
        if self.status != HospitalTransferStatus.IN_TRANSIT:
            raise ValueError(
                "Transfer is not in transit"
            )

        self.status = HospitalTransferStatus.ARRIVED
        self.arrived_at = utc_now()

    def complete(
        self,
        handover_reference: str,
    ) -> None:
        if self.status != HospitalTransferStatus.ARRIVED:
            raise ValueError(
                "Patient arrival must be confirmed"
            )

        self.clinical_handover_reference = (
            require_text(
                handover_reference,
                "handover_reference",
                500,
            )
        )

        self.status = HospitalTransferStatus.COMPLETED
        self.completed_at = utc_now()


# ============================================================
# HOSPITAL STATUS HISTORY
# ============================================================

class HospitalStatusHistory(Base):
    __tablename__ = "hospital_status_history"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    hospital_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "hospitals.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    previous_status: Mapped[
        HospitalOperationalStatus | None
    ] = mapped_column(
        enum_type(HospitalOperationalStatus),
        nullable=True,
    )

    new_status: Mapped[
        HospitalOperationalStatus
    ] = mapped_column(
        enum_type(HospitalOperationalStatus),
        nullable=False,
    )

    previous_flood_risk: Mapped[
        HospitalFloodRiskLevel | None
    ] = mapped_column(
        enum_type(HospitalFloodRiskLevel),
        nullable=True,
    )

    new_flood_risk: Mapped[
        HospitalFloodRiskLevel
    ] = mapped_column(
        enum_type(HospitalFloodRiskLevel),
        nullable=False,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source_type: Mapped[
        HospitalStatusSource
    ] = mapped_column(
        enum_type(HospitalStatusSource),
        nullable=False,
    )

    source_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    changed_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
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

    hospital: Mapped["Hospital"] = relationship(
        "Hospital",
        back_populates="status_history",
    )

    __table_args__ = (
        Index(
            "ix_hospital_status_history_time",
            "hospital_id",
            "effective_at",
        ),
    )


# ============================================================
# TIMESTAMP EVENTS
# ============================================================

@event.listens_for(Hospital, "before_update")
def update_hospital_timestamp(
    mapper: Any,
    connection: Any,
    target: Hospital,
) -> None:
    target.updated_at = utc_now()


@event.listens_for(
    HospitalPatientTransfer,
    "before_update",
)
def update_transfer_timestamp(
    mapper: Any,
    connection: Any,
    target: HospitalPatientTransfer,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

HOSPITAL_MODELS = (
    Hospital,
    HospitalBedSnapshot,
    HospitalResourceSnapshot,
    HospitalPatientTransfer,
    HospitalStatusHistory,
)


__all__ = [
    "Hospital",
    "HospitalBedSnapshot",
    "HospitalResourceSnapshot",
    "HospitalPatientTransfer",
    "HospitalStatusHistory",
    "HospitalType",
    "HospitalOperationalStatus",
    "HospitalFloodRiskLevel",
    "HospitalDataQuality",
    "HospitalBedType",
    "HospitalResourceType",
    "HospitalTransferStatus",
    "HospitalTransferPriority",
    "HospitalStatusSource",
    "HOSPITAL_MODELS",
]
