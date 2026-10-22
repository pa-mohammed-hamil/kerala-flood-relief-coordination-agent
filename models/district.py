from __future__ import annotations

"""
backend/models/district.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x models for district-level flood
monitoring, emergency coordination and administration.

Features:
- Kerala's 14 district master records
- District codes and administrative information
- District Emergency Operations Centre (DEOC)
- Taluk and local administrative divisions
- District flood-risk profiles
- Population and vulnerability statistics
- Flood monitoring and operational status
- District resource and response capacity
- Emergency contact references
- Geographic coordinates and boundaries
- Official data provenance
- Human-verified district status updates
- Historical risk assessments
- Audit and optimistic concurrency support

SAFETY:
Flood risk, population exposure and resource
availability must come from verified sources.
Stored operational status is not an official warning.
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
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(enum_class: type[enum.Enum]) -> SQLEnum:
    return SQLEnum(
        enum_class,
        values_callable=lambda members: [
            member.value for member in members
        ],
        native_enum=False,
        validate_strings=True,
        length=64,
    )


def validate_non_negative(
    value: float | int | None,
    field_name: str,
) -> float | int | None:

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be finite and non-negative"
        )

    return value


def validate_score(
    value: float | None,
    field_name: str,
) -> float | None:

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )

    return float(value)


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


# ============================================================
# ENUMERATIONS
# ============================================================

class KeralaDistrictCode(str, enum.Enum):
    THIRUVANANTHAPURAM = "TVM"
    KOLLAM = "KLM"
    PATHANAMTHITTA = "PTA"
    ALAPPUZHA = "ALP"
    KOTTAYAM = "KTM"
    IDUKKI = "IDK"
    ERNAKULAM = "EKM"
    THRISSUR = "TSR"
    PALAKKAD = "PKD"
    MALAPPURAM = "MLP"
    KOZHIKODE = "KKD"
    WAYANAD = "WYD"
    KANNUR = "KNR"
    KASARAGOD = "KSD"


class DistrictOperationalStatus(str, enum.Enum):
    NORMAL = "normal"
    MONITORING = "monitoring"
    ELEVATED_RESPONSE = "elevated_response"
    EMERGENCY_RESPONSE = "emergency_response"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class DistrictFloodRiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class DistrictAlertLevel(str, enum.Enum):
    NONE = "none"
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    UNKNOWN = "unknown"


class DistrictDataQuality(str, enum.Enum):
    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    SUSPECT = "suspect"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class AdministrativeDivisionType(str, enum.Enum):
    TALUK = "taluk"
    VILLAGE = "village"
    PANCHAYAT = "panchayat"
    MUNICIPALITY = "municipality"
    CORPORATION = "corporation"
    WARD = "ward"
    OTHER = "other"


class DistrictResourceType(str, enum.Enum):
    RESCUE_TEAM = "rescue_team"
    RESCUE_BOAT = "rescue_boat"
    AMBULANCE = "ambulance"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    RELIEF_VEHICLE = "relief_vehicle"
    SHELTER = "shelter"
    HOSPITAL = "hospital"
    MEDICAL_TEAM = "medical_team"
    VOLUNTEER = "volunteer"
    FOOD_KIT = "food_kit"
    DRINKING_WATER = "drinking_water"
    GENERATOR = "generator"
    COMMUNICATION_EQUIPMENT = "communication_equipment"
    OTHER = "other"


class DistrictStatusSource(str, enum.Enum):
    OFFICIAL_AUTHORITY = "official_authority"
    DISTRICT_CONTROL_ROOM = "district_control_room"
    VERIFIED_FIELD_REPORT = "verified_field_report"
    SENSOR_NETWORK = "sensor_network"
    MODEL_ESTIMATE = "model_estimate"
    MANUAL_ENTRY = "manual_entry"


# ============================================================
# KERALA DISTRICT REFERENCE DATA
# ============================================================

KERALA_DISTRICT_NAMES = {
    KeralaDistrictCode.THIRUVANANTHAPURAM:
        "Thiruvananthapuram",
    KeralaDistrictCode.KOLLAM:
        "Kollam",
    KeralaDistrictCode.PATHANAMTHITTA:
        "Pathanamthitta",
    KeralaDistrictCode.ALAPPUZHA:
        "Alappuzha",
    KeralaDistrictCode.KOTTAYAM:
        "Kottayam",
    KeralaDistrictCode.IDUKKI:
        "Idukki",
    KeralaDistrictCode.ERNAKULAM:
        "Ernakulam",
    KeralaDistrictCode.THRISSUR:
        "Thrissur",
    KeralaDistrictCode.PALAKKAD:
        "Palakkad",
    KeralaDistrictCode.MALAPPURAM:
        "Malappuram",
    KeralaDistrictCode.KOZHIKODE:
        "Kozhikode",
    KeralaDistrictCode.WAYANAD:
        "Wayanad",
    KeralaDistrictCode.KANNUR:
        "Kannur",
    KeralaDistrictCode.KASARAGOD:
        "Kasaragod",
}


# ============================================================
# DISTRICT MASTER MODEL
# ============================================================

class District(Base):
    __tablename__ = "districts"

    # Integer ID supports the existing service layer.
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    code: Mapped[KeralaDistrictCode] = mapped_column(
        enum_type(KeralaDistrictCode),
        nullable=False,
        unique=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        unique=True,
    )

    name_malayalam: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    state_name: Mapped[str] = mapped_column(
        String(100),
        default="Kerala",
        nullable=False,
    )

    country_code: Mapped[str] = mapped_column(
        String(2),
        default="IN",
        nullable=False,
    )

    headquarters: Mapped[str | None] = mapped_column(
        String(150),
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

    area_sq_km: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    population: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    population_reference_year: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    population_source_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    estimated_households: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    elderly_population: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    child_population: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    disability_support_population: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    coastal_district: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    hilly_district: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    major_rivers: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )

    major_reservoirs: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )

    operational_status: Mapped[
        DistrictOperationalStatus
    ] = mapped_column(
        enum_type(DistrictOperationalStatus),
        default=DistrictOperationalStatus.UNKNOWN,
        nullable=False,
        index=True,
    )

    current_flood_risk: Mapped[
        DistrictFloodRiskLevel
    ] = mapped_column(
        enum_type(DistrictFloodRiskLevel),
        default=DistrictFloodRiskLevel.UNKNOWN,
        nullable=False,
        index=True,
    )

    current_alert_level: Mapped[
        DistrictAlertLevel
    ] = mapped_column(
        enum_type(DistrictAlertLevel),
        default=DistrictAlertLevel.UNKNOWN,
        nullable=False,
    )

    risk_last_updated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    emergency_operations_centre_name: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    emergency_operations_centre_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    district_control_room_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    district_collector_office_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    official_website: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    official_data_source: Mapped[
        str | None
    ] = mapped_column(
        String(500),
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

    divisions: Mapped[
        list["DistrictAdministrativeDivision"]
    ] = relationship(
        "DistrictAdministrativeDivision",
        back_populates="district",
    )

    risk_profiles: Mapped[
        list["DistrictFloodRiskProfile"]
    ] = relationship(
        "DistrictFloodRiskProfile",
        back_populates="district",
    )

    resource_snapshots: Mapped[
        list["DistrictResourceSnapshot"]
    ] = relationship(
        "DistrictResourceSnapshot",
        back_populates="district",
    )

    status_history: Mapped[
        list["DistrictStatusHistory"]
    ] = relationship(
        "DistrictStatusHistory",
        back_populates="district",
    )

    __table_args__ = (
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_district_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_district_longitude",
        ),
        CheckConstraint(
            "population IS NULL OR population >= 0",
            name="ck_district_population",
        ),
        CheckConstraint(
            "area_sq_km IS NULL OR area_sq_km >= 0",
            name="ck_district_area",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_district_version",
        ),
        Index(
            "ix_district_status_risk",
            "operational_status",
            "current_flood_risk",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

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

    @validates(
        "population",
        "estimated_households",
        "elderly_population",
        "child_population",
        "disability_support_population",
        "area_sq_km",
    )
    def validate_measurement(
        self,
        key: str,
        value: float | int | None,
    ) -> float | int | None:
        return validate_non_negative(
            value, key
        )

    @property
    def has_known_flood_risk(self) -> bool:
        return (
            self.current_flood_risk
            != DistrictFloodRiskLevel.UNKNOWN
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code.value,
            "name": self.name,
            "state": self.state_name,
            "headquarters": self.headquarters,
            "operational_status": (
                self.operational_status.value
            ),
            "current_flood_risk": (
                self.current_flood_risk.value
            ),
            "current_alert_level": (
                self.current_alert_level.value
            ),
            "risk_last_updated_at": (
                self.risk_last_updated_at.isoformat()
                if self.risk_last_updated_at
                else None
            ),
            "is_active": self.is_active,
        }

    def __repr__(self) -> str:
        return (
            f"<District("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"risk={self.current_flood_risk.value!r}"
            f")>"
        )


# ============================================================
# ADMINISTRATIVE DIVISIONS
# ============================================================

class DistrictAdministrativeDivision(Base):
    __tablename__ = "district_administrative_divisions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
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

    division_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    name_malayalam: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    division_type: Mapped[
        AdministrativeDivisionType
    ] = mapped_column(
        enum_type(AdministrativeDivisionType),
        nullable=False,
    )

    parent_division_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "district_administrative_divisions.id",
            ondelete="SET NULL",
        ),
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

    population: Mapped[int | None] = mapped_column(
        Integer,
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

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    district: Mapped["District"] = relationship(
        "District",
        back_populates="divisions",
    )

    parent_division: Mapped[
        "DistrictAdministrativeDivision | None"
    ] = relationship(
        "DistrictAdministrativeDivision",
        remote_side="DistrictAdministrativeDivision.id",
        back_populates="child_divisions",
    )

    child_divisions: Mapped[
        list["DistrictAdministrativeDivision"]
    ] = relationship(
        "DistrictAdministrativeDivision",
        back_populates="parent_division",
    )

    __table_args__ = (
        UniqueConstraint(
            "district_id",
            "division_type",
            "division_code",
            name="uq_district_division_code",
        ),
        CheckConstraint(
            "population IS NULL OR population >= 0",
            name="ck_district_division_population",
        ),
        Index(
            "ix_district_division_type",
            "district_id",
            "division_type",
        ),
    )


# ============================================================
# FLOOD RISK PROFILE
# ============================================================

class DistrictFloodRiskProfile(Base):
    __tablename__ = "district_flood_risk_profiles"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
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

    risk_level: Mapped[
        DistrictFloodRiskLevel
    ] = mapped_column(
        enum_type(DistrictFloodRiskLevel),
        default=DistrictFloodRiskLevel.UNKNOWN,
        nullable=False,
    )

    risk_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    rainfall_risk_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    river_risk_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    dam_risk_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    terrain_risk_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    population_exposure_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    estimated_exposed_population: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    estimated_exposed_households: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    confidence_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    data_quality: Mapped[
        DistrictDataQuality
    ] = mapped_column(
        enum_type(DistrictDataQuality),
        default=DistrictDataQuality.UNKNOWN,
        nullable=False,
    )

    source: Mapped[DistrictStatusSource] = mapped_column(
        enum_type(DistrictStatusSource),
        nullable=False,
    )

    source_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    model_reference: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
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

    verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    assessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    district: Mapped["District"] = relationship(
        "District",
        back_populates="risk_profiles",
    )

    __table_args__ = (
        CheckConstraint(
            "risk_score IS NULL OR "
            "(risk_score >= 0 AND risk_score <= 1)",
            name="ck_district_risk_score",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 "
            "AND confidence_score <= 1)",
            name="ck_district_risk_confidence",
        ),
        CheckConstraint(
            "estimated_exposed_population IS NULL OR "
            "estimated_exposed_population >= 0",
            name="ck_district_exposed_population",
        ),
        Index(
            "ix_district_risk_assessed",
            "district_id",
            "assessed_at",
        ),
        Index(
            "ix_district_risk_level_time",
            "risk_level",
            "assessed_at",
        ),
    )

    @validates(
        "risk_score",
        "rainfall_risk_score",
        "river_risk_score",
        "dam_risk_score",
        "terrain_risk_score",
        "population_exposure_score",
        "confidence_score",
    )
    def validate_risk_score(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_score(
            value, key
        )

    def is_expired(
        self,
        now: datetime | None = None,
    ) -> bool:

        if self.valid_until is None:
            return False

        now = now or utc_now()

        return self.valid_until <= now


# ============================================================
# RESOURCE CAPACITY SNAPSHOT
# ============================================================

class DistrictResourceSnapshot(Base):
    __tablename__ = "district_resource_snapshots"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
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

    resource_type: Mapped[
        DistrictResourceType
    ] = mapped_column(
        enum_type(DistrictResourceType),
        nullable=False,
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

    source_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
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

    district: Mapped["District"] = relationship(
        "District",
        back_populates="resource_snapshots",
    )

    __table_args__ = (
        CheckConstraint(
            "total_quantity >= 0",
            name="ck_district_resource_total",
        ),
        CheckConstraint(
            "available_quantity >= 0",
            name="ck_district_resource_available",
        ),
        CheckConstraint(
            "allocated_quantity >= 0",
            name="ck_district_resource_allocated",
        ),
        CheckConstraint(
            "unavailable_quantity >= 0",
            name="ck_district_resource_unavailable",
        ),
        CheckConstraint(
            "available_quantity + allocated_quantity "
            "+ unavailable_quantity <= total_quantity",
            name="ck_district_resource_capacity",
        ),
        Index(
            "ix_district_resource_type_time",
            "district_id",
            "resource_type",
            "snapshot_at",
        ),
    )

    @property
    def utilization_percentage(self) -> float:

        if self.total_quantity == 0:
            return 0.0

        return round(
            self.allocated_quantity
            / self.total_quantity
            * 100,
            2,
        )


# ============================================================
# DISTRICT OPERATIONAL STATUS HISTORY
# ============================================================

class DistrictStatusHistory(Base):
    __tablename__ = "district_status_history"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
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

    previous_status: Mapped[
        DistrictOperationalStatus
    ] = mapped_column(
        enum_type(DistrictOperationalStatus),
        nullable=False,
    )

    new_status: Mapped[
        DistrictOperationalStatus
    ] = mapped_column(
        enum_type(DistrictOperationalStatus),
        nullable=False,
    )

    previous_risk_level: Mapped[
        DistrictFloodRiskLevel
    ] = mapped_column(
        enum_type(DistrictFloodRiskLevel),
        nullable=False,
    )

    new_risk_level: Mapped[
        DistrictFloodRiskLevel
    ] = mapped_column(
        enum_type(DistrictFloodRiskLevel),
        nullable=False,
    )

    source: Mapped[DistrictStatusSource] = mapped_column(
        enum_type(DistrictStatusSource),
        nullable=False,
    )

    source_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    changed_by_user_id: Mapped[
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

    is_human_approved: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
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

    district: Mapped["District"] = relationship(
        "District",
        back_populates="status_history",
    )

    __table_args__ = (
        Index(
            "ix_district_status_history_time",
            "district_id",
            "effective_at",
        ),
    )


# ============================================================
# DISTRICT SEED DATA
# ============================================================

KERALA_DISTRICT_SEED_DATA = [
    {
        "code": code,
        "name": name,
        "state_name": "Kerala",
        "country_code": "IN",
        "operational_status": (
            DistrictOperationalStatus.UNKNOWN
        ),
        "current_flood_risk": (
            DistrictFloodRiskLevel.UNKNOWN
        ),
        "current_alert_level": (
            DistrictAlertLevel.UNKNOWN
        ),
        "is_active": True,
    }
    for code, name in KERALA_DISTRICT_NAMES.items()
]


def build_kerala_districts() -> list[District]:
    """
    Build district ORM objects for initial seeding.

    Does not insert records into the database.
    Does not populate unverified demographic data.
    """

    return [
        District(**record)
        for record in KERALA_DISTRICT_SEED_DATA
    ]


# ============================================================
# TIMESTAMP EVENTS
# ============================================================

@event.listens_for(
    District,
    "before_update",
)
def update_district_timestamp(
    mapper: Any,
    connection: Any,
    target: District,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

DISTRICT_MODELS = (
    District,
    DistrictAdministrativeDivision,
    DistrictFloodRiskProfile,
    DistrictResourceSnapshot,
    DistrictStatusHistory,
)


__all__ = [
    "District",
    "DistrictAdministrativeDivision",
    "DistrictFloodRiskProfile",
    "DistrictResourceSnapshot",
    "DistrictStatusHistory",
    "KeralaDistrictCode",
    "DistrictOperationalStatus",
    "DistrictFloodRiskLevel",
    "DistrictAlertLevel",
    "DistrictDataQuality",
    "AdministrativeDivisionType",
    "DistrictResourceType",
    "DistrictStatusSource",
    "KERALA_DISTRICT_NAMES",
    "KERALA_DISTRICT_SEED_DATA",
    "DISTRICT_MODELS",
    "build_kerala_districts",
]
