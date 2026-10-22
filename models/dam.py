from __future__ import annotations

"""
backend/models/dam.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x ORM models for dam monitoring,
reservoir telemetry, spillway gates, discharge
observations, safety alerts and operational records.

Features:
- Dam master records and geolocation
- Reservoir capacity and storage information
- Real-time and historical water-level readings
- Inflow, outflow and spillway discharge monitoring
- Gate configuration and reported gate status
- Dam safety observations and warning levels
- Sensor quality, freshness and verification
- Reservoir operational advisory records
- Human approval for operational recommendations
- Data provenance and audit references
- Time-series indexing
- PostgreSQL-compatible database models

SAFETY:
These models store observations and recommendations.
They do not control dam gates, change reservoir
discharge or issue official flood warnings.

Operational decisions require authorization by
the responsible dam authority. Telemetry thresholds
must be calibrated for each dam using official data.
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
            item.value for item in members
        ],
        native_enum=False,
        validate_strings=True,
        length=64,
    )


def validate_non_negative(
    value: float | None,
    field_name: str,
) -> float | None:

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a finite "
            "non-negative number"
        )

    return float(value)


def validate_percentage(
    value: float | None,
    field_name: str,
) -> float | None:

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 100
    ):
        raise ValueError(
            f"{field_name} must be between 0 and 100"
        )

    return float(value)


def validate_coordinate(
    value: float,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:

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

class DamType(str, enum.Enum):
    GRAVITY = "gravity"
    ARCH = "arch"
    EARTHFILL = "earthfill"
    ROCKFILL = "rockfill"
    CONCRETE = "concrete"
    COMPOSITE = "composite"
    BARRAGE = "barrage"
    WEIR = "weir"
    OTHER = "other"


class DamPurpose(str, enum.Enum):
    HYDROELECTRIC = "hydroelectric"
    IRRIGATION = "irrigation"
    WATER_SUPPLY = "water_supply"
    FLOOD_CONTROL = "flood_control"
    MULTIPURPOSE = "multipurpose"
    OTHER = "other"


class DamOperationalStatus(str, enum.Enum):
    OPERATIONAL = "operational"
    RESTRICTED = "restricted"
    UNDER_MAINTENANCE = "under_maintenance"
    OUT_OF_SERVICE = "out_of_service"
    UNKNOWN = "unknown"


class DamRiskLevel(str, enum.Enum):
    NORMAL = "normal"
    ADVISORY = "advisory"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DamSensorStatus(str, enum.Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class DamDataQuality(str, enum.Enum):
    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    SUSPECT = "suspect"
    INVALID = "invalid"
    MISSING = "missing"


class DamGateType(str, enum.Enum):
    RADIAL = "radial"
    VERTICAL_LIFT = "vertical_lift"
    SLUICE = "sluice"
    FLAP = "flap"
    OTHER = "other"


class DamGateStatus(str, enum.Enum):
    CLOSED = "closed"
    PARTIALLY_OPEN = "partially_open"
    OPEN = "open"
    MOVING = "moving"
    FAULT = "fault"
    UNKNOWN = "unknown"


class DamObservationSource(str, enum.Enum):
    SENSOR = "sensor"
    SCADA = "scada"
    OFFICIAL_API = "official_api"
    MANUAL = "manual"
    FIELD_INSPECTION = "field_inspection"
    IMPORT = "import"


class DamAdvisoryStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class DamAlertType(str, enum.Enum):
    HIGH_WATER_LEVEL = "high_water_level"
    RAPID_WATER_RISE = "rapid_water_rise"
    HIGH_INFLOW = "high_inflow"
    HIGH_DISCHARGE = "high_discharge"
    SENSOR_FAILURE = "sensor_failure"
    GATE_ANOMALY = "gate_anomaly"
    STRUCTURAL_CONCERN = "structural_concern"
    DATA_QUALITY = "data_quality"
    OTHER = "other"


class DamAlertStatus(str, enum.Enum):
    DETECTED = "detected"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    DISMISSED = "dismissed"
    RESOLVED = "resolved"


# ============================================================
# DAM MASTER MODEL
# ============================================================

class Dam(Base):
    __tablename__ = "dams"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    dam_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )

    alternate_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    district_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    river_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    basin_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    latitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    longitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    elevation_meters: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    dam_type: Mapped[DamType] = mapped_column(
        enum_type(DamType),
        default=DamType.OTHER,
        nullable=False,
    )

    purpose: Mapped[DamPurpose] = mapped_column(
        enum_type(DamPurpose),
        default=DamPurpose.MULTIPURPOSE,
        nullable=False,
    )

    operating_authority: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    official_source_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    year_completed: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    dam_height_meters: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    dam_length_meters: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    full_reservoir_level_meters: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    maximum_water_level_meters: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    minimum_drawdown_level_meters: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    total_storage_mcm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    live_storage_mcm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    dead_storage_mcm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    catchment_area_sq_km: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    designed_spillway_capacity_cumecs: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    operational_status: Mapped[
        DamOperationalStatus
    ] = mapped_column(
        enum_type(DamOperationalStatus),
        default=DamOperationalStatus.UNKNOWN,
        nullable=False,
    )

    current_risk_level: Mapped[
        DamRiskLevel
    ] = mapped_column(
        enum_type(DamRiskLevel),
        default=DamRiskLevel.UNKNOWN,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    last_inspection_at: Mapped[
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

    readings: Mapped[
        list["DamWaterLevelReading"]
    ] = relationship(
        "DamWaterLevelReading",
        back_populates="dam",
        cascade="save-update, merge",
    )

    gates: Mapped[
        list["DamGate"]
    ] = relationship(
        "DamGate",
        back_populates="dam",
        cascade="save-update, merge",
    )

    alerts: Mapped[
        list["DamAlert"]
    ] = relationship(
        "DamAlert",
        back_populates="dam",
        cascade="save-update, merge",
    )

    advisories: Mapped[
        list["DamOperationalAdvisory"]
    ] = relationship(
        "DamOperationalAdvisory",
        back_populates="dam",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_dam_district_positive",
        ),
        CheckConstraint(
            "latitude >= -90 AND latitude <= 90",
            name="ck_dam_latitude",
        ),
        CheckConstraint(
            "longitude >= -180 AND longitude <= 180",
            name="ck_dam_longitude",
        ),
        CheckConstraint(
            "total_storage_mcm IS NULL OR "
            "total_storage_mcm >= 0",
            name="ck_dam_total_storage",
        ),
        CheckConstraint(
            "live_storage_mcm IS NULL OR "
            "live_storage_mcm >= 0",
            name="ck_dam_live_storage",
        ),
        CheckConstraint(
            "dead_storage_mcm IS NULL OR "
            "dead_storage_mcm >= 0",
            name="ck_dam_dead_storage",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_dam_version",
        ),
        Index(
            "ix_dam_district_status",
            "district_id",
            "operational_status",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("latitude")
    def validate_latitude(
        self,
        key: str,
        value: float,
    ) -> float:
        return validate_coordinate(
            value,
            key,
            -90,
            90,
        )

    @validates("longitude")
    def validate_longitude(
        self,
        key: str,
        value: float,
    ) -> float:
        return validate_coordinate(
            value,
            key,
            -180,
            180,
        )

    @validates(
        "total_storage_mcm",
        "live_storage_mcm",
        "dead_storage_mcm",
        "catchment_area_sq_km",
        "designed_spillway_capacity_cumecs",
        "dam_height_meters",
        "dam_length_meters",
    )
    def validate_positive_measurement(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_non_negative(
            value,
            key,
        )

    @property
    def location(self) -> dict[str, float]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
        }

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dam_code": self.dam_code,
            "name": self.name,
            "district_id": self.district_id,
            "river_name": self.river_name,
            "basin_name": self.basin_name,
            "dam_type": self.dam_type.value,
            "purpose": self.purpose.value,
            "operational_status": (
                self.operational_status.value
            ),
            "current_risk_level": (
                self.current_risk_level.value
            ),
            "is_active": self.is_active,
        }

    def __repr__(self) -> str:
        return (
            f"<Dam("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"status={self.operational_status.value!r}"
            f")>"
        )


# ============================================================
# WATER LEVEL AND RESERVOIR READINGS
# ============================================================

class DamWaterLevelReading(Base):
    __tablename__ = "dam_water_level_readings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    dam_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "dams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    source: Mapped[DamObservationSource] = mapped_column(
        enum_type(DamObservationSource),
        nullable=False,
    )

    source_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    external_reading_id: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    water_level_meters: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    storage_mcm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    storage_percentage: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    inflow_cumecs: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    outflow_cumecs: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    spillway_discharge_cumecs: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    rainfall_mm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    water_level_change_m_per_hour: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    sensor_status: Mapped[DamSensorStatus] = (
        mapped_column(
            enum_type(DamSensorStatus),
            default=DamSensorStatus.UNKNOWN,
            nullable=False,
        )
    )

    data_quality: Mapped[DamDataQuality] = (
        mapped_column(
            enum_type(DamDataQuality),
            default=DamDataQuality.PROVISIONAL,
            nullable=False,
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

    verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    received_at: Mapped[datetime] = mapped_column(
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

    dam: Mapped["Dam"] = relationship(
        "Dam",
        back_populates="readings",
    )

    __table_args__ = (
        UniqueConstraint(
            "dam_id",
            "source",
            "external_reading_id",
            name="uq_dam_external_reading",
        ),
        CheckConstraint(
            "storage_percentage IS NULL OR "
            "(storage_percentage >= 0 "
            "AND storage_percentage <= 100)",
            name="ck_dam_reading_storage_percentage",
        ),
        CheckConstraint(
            "inflow_cumecs IS NULL OR inflow_cumecs >= 0",
            name="ck_dam_reading_inflow",
        ),
        CheckConstraint(
            "outflow_cumecs IS NULL OR outflow_cumecs >= 0",
            name="ck_dam_reading_outflow",
        ),
        CheckConstraint(
            "storage_mcm IS NULL OR storage_mcm >= 0",
            name="ck_dam_reading_storage",
        ),
        CheckConstraint(
            "spillway_discharge_cumecs IS NULL OR "
            "spillway_discharge_cumecs >= 0",
            name="ck_dam_reading_spillway",
        ),
        Index(
            "ix_dam_reading_dam_observed",
            "dam_id",
            "observed_at",
        ),
        Index(
            "ix_dam_reading_quality_time",
            "data_quality",
            "observed_at",
        ),
    )

    @validates(
        "storage_mcm",
        "inflow_cumecs",
        "outflow_cumecs",
        "spillway_discharge_cumecs",
        "rainfall_mm",
    )
    def validate_measurement(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_non_negative(
            value,
            key,
        )

    @validates("storage_percentage")
    def validate_storage_percentage(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_percentage(
            value,
            key,
        )

    @validates("water_level_change_m_per_hour")
    def validate_change_rate(
        self,
        key: str,
        value: float | None,
    ) -> float | None:

        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                "Invalid water-level change rate"
            )

        return float(value)

    def is_stale(
        self,
        *,
        maximum_age_seconds: int = 3600,
        now: datetime | None = None,
    ) -> bool:

        if maximum_age_seconds < 0:
            raise ValueError(
                "maximum_age_seconds must be non-negative"
            )

        now = now or utc_now()

        if (
            self.observed_at.tzinfo is None
            or now.tzinfo is None
        ):
            raise ValueError(
                "Timezone-aware timestamps required"
            )

        return (
            now - self.observed_at
        ).total_seconds() > maximum_age_seconds

    @property
    def is_usable_for_verified_reporting(self) -> bool:
        return (
            self.is_verified
            and self.data_quality == DamDataQuality.VERIFIED
            and self.sensor_status
            not in {
                DamSensorStatus.OFFLINE,
                DamSensorStatus.UNKNOWN,
            }
        )


# ============================================================
# DAM GATE CONFIGURATION
# ============================================================

class DamGate(Base):
    __tablename__ = "dam_gates"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    dam_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "dams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    gate_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    gate_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    gate_type: Mapped[DamGateType] = mapped_column(
        enum_type(DamGateType),
        default=DamGateType.OTHER,
        nullable=False,
    )

    maximum_opening_meters: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    gate_width_meters: Mapped[
        float | None
    ] = mapped_column(
        Float,
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

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    dam: Mapped["Dam"] = relationship(
        "Dam",
        back_populates="gates",
    )

    observations: Mapped[
        list["DamGateObservation"]
    ] = relationship(
        "DamGateObservation",
        back_populates="gate",
    )

    __table_args__ = (
        UniqueConstraint(
            "dam_id",
            "gate_code",
            name="uq_dam_gate_code",
        ),
        CheckConstraint(
            "maximum_opening_meters IS NULL OR "
            "maximum_opening_meters >= 0",
            name="ck_dam_gate_max_opening",
        ),
        CheckConstraint(
            "gate_width_meters IS NULL OR "
            "gate_width_meters >= 0",
            name="ck_dam_gate_width",
        ),
    )


# ============================================================
# GATE STATUS OBSERVATIONS
# ============================================================

class DamGateObservation(Base):
    __tablename__ = "dam_gate_observations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    gate_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "dam_gates.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    status: Mapped[DamGateStatus] = mapped_column(
        enum_type(DamGateStatus),
        nullable=False,
    )

    opening_meters: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    opening_percentage: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    estimated_discharge_cumecs: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    source: Mapped[DamObservationSource] = mapped_column(
        enum_type(DamObservationSource),
        nullable=False,
    )

    source_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    data_quality: Mapped[DamDataQuality] = (
        mapped_column(
            enum_type(DamDataQuality),
            default=DamDataQuality.PROVISIONAL,
            nullable=False,
        )
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    received_at: Mapped[datetime] = mapped_column(
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

    gate: Mapped["DamGate"] = relationship(
        "DamGate",
        back_populates="observations",
    )

    __table_args__ = (
        CheckConstraint(
            "opening_meters IS NULL OR opening_meters >= 0",
            name="ck_dam_gate_observation_opening",
        ),
        CheckConstraint(
            "opening_percentage IS NULL OR "
            "(opening_percentage >= 0 "
            "AND opening_percentage <= 100)",
            name="ck_dam_gate_observation_percentage",
        ),
        CheckConstraint(
            "estimated_discharge_cumecs IS NULL OR "
            "estimated_discharge_cumecs >= 0",
            name="ck_dam_gate_observation_discharge",
        ),
        Index(
            "ix_dam_gate_observation_time",
            "gate_id",
            "observed_at",
        ),
    )

    @validates(
        "opening_meters",
        "estimated_discharge_cumecs",
    )
    def validate_measurement(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_non_negative(
            value,
            key,
        )

    @validates("opening_percentage")
    def validate_opening_percentage(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_percentage(
            value,
            key,
        )


# ============================================================
# DAM SAFETY ALERTS
# ============================================================

class DamAlert(Base):
    __tablename__ = "dam_alerts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    dam_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "dams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    alert_type: Mapped[DamAlertType] = mapped_column(
        enum_type(DamAlertType),
        nullable=False,
    )

    risk_level: Mapped[DamRiskLevel] = mapped_column(
        enum_type(DamRiskLevel),
        default=DamRiskLevel.UNKNOWN,
        nullable=False,
    )

    status: Mapped[DamAlertStatus] = mapped_column(
        enum_type(DamAlertStatus),
        default=DamAlertStatus.DETECTED,
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

    observation_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "dam_water_level_readings.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    source_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    requires_human_review: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=True,
            nullable=False,
        )
    )

    reviewed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    review_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reviewed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    resolved_at: Mapped[
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

    dam: Mapped["Dam"] = relationship(
        "Dam",
        back_populates="alerts",
    )

    __table_args__ = (
        Index(
            "ix_dam_alert_dam_status",
            "dam_id",
            "status",
        ),
        Index(
            "ix_dam_alert_risk_detected",
            "risk_level",
            "detected_at",
        ),
    )


# ============================================================
# OPERATIONAL ADVISORY MODEL
# ============================================================

class DamOperationalAdvisory(Base):
    __tablename__ = "dam_operational_advisories"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    dam_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "dams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    recommendation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    reasoning_summary: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    status: Mapped[DamAdvisoryStatus] = mapped_column(
        enum_type(DamAdvisoryStatus),
        default=DamAdvisoryStatus.DRAFT,
        nullable=False,
    )

    risk_level: Mapped[DamRiskLevel] = mapped_column(
        enum_type(DamRiskLevel),
        default=DamRiskLevel.UNKNOWN,
        nullable=False,
    )

    confidence_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    generated_by_agent: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    model_reference: Mapped[str | None] = (
        mapped_column(
            String(200),
            nullable=True,
        )
    )

    source_observation_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "dam_water_level_readings.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    created_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    reviewed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    review_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    approval_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    reviewed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    expires_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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

    dam: Mapped["Dam"] = relationship(
        "Dam",
        back_populates="advisories",
    )

    __table_args__ = (
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 "
            "AND confidence_score <= 1)",
            name="ck_dam_advisory_confidence",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_dam_advisory_version",
        ),
        Index(
            "ix_dam_advisory_dam_status",
            "dam_id",
            "status",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("confidence_score")
    def validate_confidence(
        self,
        key: str,
        value: float | None,
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
                "Confidence must be between 0 and 1"
            )

        return float(value)

    @property
    def is_approved(self) -> bool:
        return (
            self.status == DamAdvisoryStatus.APPROVED
            and self.reviewed_by_user_id is not None
            and bool(self.approval_reference)
        )

    def is_expired(
        self,
        now: datetime | None = None,
    ) -> bool:

        if self.expires_at is None:
            return False

        now = now or utc_now()

        return self.expires_at <= now

    def submit_for_review(self) -> None:

        if self.status != DamAdvisoryStatus.DRAFT:
            raise ValueError(
                "Only draft advisories can be submitted"
            )

        self.status = DamAdvisoryStatus.PENDING_REVIEW
        self.updated_at = utc_now()

    def approve(
        self,
        *,
        reviewer_id: int,
        approval_reference: str,
        notes: str,
    ) -> None:

        if self.status != DamAdvisoryStatus.PENDING_REVIEW:
            raise ValueError(
                "Advisory is not pending review"
            )

        if self.is_expired():
            raise ValueError(
                "Expired advisory cannot be approved"
            )

        if reviewer_id <= 0:
            raise ValueError(
                "Invalid reviewer"
            )

        if (
            self.created_by_user_id is not None
            and reviewer_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent human approval required"
            )

        if not approval_reference.strip():
            raise ValueError(
                "Official approval reference required"
            )

        if not notes.strip():
            raise ValueError(
                "Review notes required"
            )

        self.status = DamAdvisoryStatus.APPROVED
        self.reviewed_by_user_id = reviewer_id
        self.approval_reference = approval_reference.strip()
        self.review_notes = notes.strip()
        self.reviewed_at = utc_now()
        self.updated_at = utc_now()

    def reject(
        self,
        *,
        reviewer_id: int,
        reason: str,
    ) -> None:

        if self.status != DamAdvisoryStatus.PENDING_REVIEW:
            raise ValueError(
                "Advisory is not pending review"
            )

        if reviewer_id <= 0:
            raise ValueError(
                "Invalid reviewer"
            )

        if not reason.strip():
            raise ValueError(
                "Rejection reason required"
            )

        self.status = DamAdvisoryStatus.REJECTED
        self.reviewed_by_user_id = reviewer_id
        self.review_notes = reason.strip()
        self.reviewed_at = utc_now()
        self.updated_at = utc_now()


# ============================================================
# ORM TIMESTAMP EVENTS
# ============================================================

@event.listens_for(
    Dam,
    "before_update",
)
def update_dam_timestamp(
    mapper: Any,
    connection: Any,
    target: Dam,
) -> None:
    target.updated_at = utc_now()


@event.listens_for(
    DamOperationalAdvisory,
    "before_update",
)
def update_advisory_timestamp(
    mapper: Any,
    connection: Any,
    target: DamOperationalAdvisory,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

DAM_MODELS = (
    Dam,
    DamWaterLevelReading,
    DamGate,
    DamGateObservation,
    DamAlert,
    DamOperationalAdvisory,
)


__all__ = [
    "Dam",
    "DamWaterLevelReading",
    "DamGate",
    "DamGateObservation",
    "DamAlert",
    "DamOperationalAdvisory",
    "DamType",
    "DamPurpose",
    "DamOperationalStatus",
    "DamRiskLevel",
    "DamSensorStatus",
    "DamDataQuality",
    "DamGateType",
    "DamGateStatus",
    "DamObservationSource",
    "DamAdvisoryStatus",
    "DamAlertType",
    "DamAlertStatus",
    "DAM_MODELS",
]
