from __future__ import annotations

"""
backend/models/river_station.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x ORM models for river monitoring.

Features:
- River and water-level monitoring stations
- River basin and district information
- Real-time water-level observations
- Water discharge and flow measurements
- Rainfall and environmental telemetry
- Warning and danger thresholds
- Sensor health and calibration
- Station operational status
- AI flood-level predictions
- Forecast verification
- Data quality and anomaly detection
- Human-reviewed river advisories
- Station maintenance history
- Monitoring audit events

Safety:
- AI forecasts are advisory.
- River readings may be delayed or inaccurate.
- Thresholds require authoritative verification.
- Official flood warnings require authorized approval.
- Sensor failures must not be interpreted as safe conditions.
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
            f"{field_name} exceeds maximum length"
        )

    return result


def validate_finite(
    value: float | int | None,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise ValueError(
            f"{field_name} is below minimum"
        )

    if maximum is not None and result > maximum:
        raise ValueError(
            f"{field_name} exceeds maximum"
        )

    return result


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class RiverStationType(str, enum.Enum):
    WATER_LEVEL = "water_level"
    DISCHARGE = "discharge"
    HYDROLOGICAL = "hydrological"
    RAINFALL = "rainfall"
    MULTIPARAMETER = "multiparameter"
    MANUAL_GAUGE = "manual_gauge"
    AUTOMATIC_GAUGE = "automatic_gauge"
    TELEMETRY = "telemetry"
    OTHER = "other"


class RiverStationStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_VERIFICATION = "pending_verification"
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    DECOMMISSIONED = "decommissioned"


class RiverStationSource(str, enum.Enum):
    CWC = "cwc"
    KSDMA = "ksdma"
    STATE_WATER_RESOURCES = "state_water_resources"
    IRRIGATION_DEPARTMENT = "irrigation_department"
    IMD = "imd"
    LOCAL_AUTHORITY = "local_authority"
    RESEARCH_INSTITUTION = "research_institution"
    VERIFIED_SENSOR = "verified_sensor"
    MANUAL_OBSERVATION = "manual_observation"
    SIMULATION = "simulation"
    OTHER = "other"


class RiverWaterLevelStatus(str, enum.Enum):
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    DANGER = "danger"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class RiverReadingQuality(str, enum.Enum):
    RAW = "raw"
    PENDING_VALIDATION = "pending_validation"
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"
    MISSING = "missing"
    CORRECTED = "corrected"


class RiverSensorStatus(str, enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CALIBRATION_REQUIRED = "calibration_required"
    FAULTY = "faulty"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class RiverSensorType(str, enum.Enum):
    WATER_LEVEL = "water_level"
    PRESSURE = "pressure"
    ULTRASONIC = "ultrasonic"
    RADAR = "radar"
    FLOW_METER = "flow_meter"
    RAIN_GAUGE = "rain_gauge"
    TEMPERATURE = "temperature"
    TURBIDITY = "turbidity"
    BATTERY = "battery"
    COMMUNICATION = "communication"
    OTHER = "other"


class RiverForecastStatus(str, enum.Enum):
    GENERATED = "generated"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RiverAdvisoryStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class RiverMaintenanceType(str, enum.Enum):
    ROUTINE_INSPECTION = "routine_inspection"
    SENSOR_CALIBRATION = "sensor_calibration"
    SENSOR_REPLACEMENT = "sensor_replacement"
    BATTERY_REPLACEMENT = "battery_replacement"
    COMMUNICATION_REPAIR = "communication_repair"
    CLEANING = "cleaning"
    DATA_QUALITY_INVESTIGATION = (
        "data_quality_investigation"
    )
    EMERGENCY_REPAIR = "emergency_repair"
    OTHER = "other"


class RiverStationEventType(str, enum.Enum):
    CREATED = "created"
    VERIFIED = "verified"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    READING_RECEIVED = "reading_received"
    READING_VALIDATED = "reading_validated"
    ANOMALY_DETECTED = "anomaly_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    THRESHOLD_UPDATED = "threshold_updated"
    SENSOR_FAILURE = "sensor_failure"
    SENSOR_RECOVERED = "sensor_recovered"
    FORECAST_GENERATED = "forecast_generated"
    ADVISORY_CREATED = "advisory_created"
    ADVISORY_APPROVED = "advisory_approved"
    ADVISORY_PUBLISHED = "advisory_published"
    MAINTENANCE_STARTED = "maintenance_started"
    MAINTENANCE_COMPLETED = "maintenance_completed"
    NOTE_ADDED = "note_added"


# ============================================================
# RIVER MONITORING STATION
# ============================================================

class RiverStation(Base):
    __tablename__ = "river_stations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    station_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    river_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    river_basin: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    tributary_name: Mapped[str | None] = (
        mapped_column(
            String(255),
            nullable=True,
        )
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

    location_name: Mapped[str | None] = (
        mapped_column(
            String(255),
            nullable=True,
        )
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    elevation_m: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    gauge_zero_elevation_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    station_type: Mapped[
        RiverStationType
    ] = mapped_column(
        enum_type(RiverStationType),
        nullable=False,
    )

    source: Mapped[RiverStationSource] = mapped_column(
        enum_type(RiverStationSource),
        nullable=False,
    )

    external_station_id: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    managing_authority: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    status: Mapped[
        RiverStationStatus
    ] = mapped_column(
        enum_type(RiverStationStatus),
        default=RiverStationStatus.DRAFT,
        nullable=False,
        index=True,
    )

    current_level_status: Mapped[
        RiverWaterLevelStatus
    ] = mapped_column(
        enum_type(RiverWaterLevelStatus),
        default=RiverWaterLevelStatus.UNKNOWN,
        nullable=False,
    )

    latest_water_level_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    latest_discharge_cumecs: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    latest_reading_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    latest_validated_reading_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    warning_level_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    danger_level_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    extreme_level_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    thresholds_verified: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    threshold_source_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    thresholds_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    thresholds_verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    data_reporting_interval_minutes: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    stale_after_minutes: Mapped[int] = (
        mapped_column(
            Integer,
            default=60,
            nullable=False,
        )
    )

    is_telemetry_enabled: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
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

    last_maintenance_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    next_maintenance_due_at: Mapped[
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
        list["RiverWaterLevelReading"]
    ] = relationship(
        "RiverWaterLevelReading",
        back_populates="station",
        cascade="save-update, merge",
    )

    sensors: Mapped[
        list["RiverStationSensor"]
    ] = relationship(
        "RiverStationSensor",
        back_populates="station",
        cascade="save-update, merge",
    )

    forecasts: Mapped[
        list["RiverLevelForecast"]
    ] = relationship(
        "RiverLevelForecast",
        back_populates="station",
        cascade="save-update, merge",
    )

    advisories: Mapped[
        list["RiverStationAdvisory"]
    ] = relationship(
        "RiverStationAdvisory",
        back_populates="station",
        cascade="save-update, merge",
    )

    maintenance_records: Mapped[
        list["RiverStationMaintenance"]
    ] = relationship(
        "RiverStationMaintenance",
        back_populates="station",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["RiverStationEvent"]
    ] = relationship(
        "RiverStationEvent",
        back_populates="station",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_river_station_district",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_river_station_lat",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_river_station_lon",
        ),
        CheckConstraint(
            "latest_discharge_cumecs IS NULL OR "
            "latest_discharge_cumecs >= 0",
            name="ck_river_station_discharge",
        ),
        CheckConstraint(
            "stale_after_minutes > 0",
            name="ck_river_station_stale_minutes",
        ),
        CheckConstraint(
            "data_reporting_interval_minutes IS NULL OR "
            "data_reporting_interval_minutes > 0",
            name="ck_river_station_reporting_interval",
        ),
        CheckConstraint(
            "warning_level_m IS NULL OR "
            "danger_level_m IS NULL OR "
            "warning_level_m < danger_level_m",
            name="ck_river_station_warning_danger",
        ),
        CheckConstraint(
            "danger_level_m IS NULL OR "
            "extreme_level_m IS NULL OR "
            "danger_level_m < extreme_level_m",
            name="ck_river_station_danger_extreme",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_river_station_version",
        ),
        Index(
            "ix_river_station_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_river_station_river_status",
            "river_name",
            "current_level_status",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "station_code",
        "station_name",
        "river_name",
    )
    def validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "station_code": 100,
            "station_name": 255,
            "river_name": 255,
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
        return validate_finite(
            value, key, -90, 90
        )

    @validates("longitude")
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_finite(
            value, key, -180, 180
        )

    @validates(
        "latest_water_level_m",
        "warning_level_m",
        "danger_level_m",
        "extreme_level_m",
        "elevation_m",
        "gauge_zero_elevation_m",
    )
    def validate_levels(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_finite(value, key)

    @validates("latest_discharge_cumecs")
    def validate_discharge(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_finite(
            value, key, minimum=0
        )

    @property
    def is_data_stale(self) -> bool:
        if self.latest_validated_reading_at is None:
            return True

        last = as_utc(
            self.latest_validated_reading_at
        )

        age_seconds = (
            utc_now() - last
        ).total_seconds()

        return (
            age_seconds < 0
            or age_seconds
            > self.stale_after_minutes * 60
        )

    @property
    def can_use_for_live_decisions(self) -> bool:
        return (
            self.status == RiverStationStatus.ACTIVE
            and self.is_verified
            and self.thresholds_verified
            and not self.is_data_stale
            and self.latest_water_level_m is not None
        )

    def submit_for_verification(self) -> None:
        if self.status != RiverStationStatus.DRAFT:
            raise ValueError(
                "Only draft stations can be submitted"
            )

        self.status = (
            RiverStationStatus.PENDING_VERIFICATION
        )

    def verify(self, verifier_id: int) -> None:
        if self.status != (
            RiverStationStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Station is not pending verification"
            )

        if verifier_id <= 0:
            raise ValueError(
                "Invalid verifier"
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
        self.status = RiverStationStatus.ACTIVE

    def classify_level(
        self,
        water_level_m: float,
    ) -> RiverWaterLevelStatus:
        """
        Classify a validated reading using the
        station's verified reference thresholds.

        Returns UNKNOWN if thresholds are unverified.
        This is not an official public flood warning.
        """
        level = validate_finite(
            water_level_m,
            "water_level_m",
        )

        if (
            level is None
            or not self.thresholds_verified
            or self.warning_level_m is None
            or self.danger_level_m is None
        ):
            return RiverWaterLevelStatus.UNKNOWN

        if (
            self.extreme_level_m is not None
            and level >= self.extreme_level_m
        ):
            return RiverWaterLevelStatus.EXTREME

        if level >= self.danger_level_m:
            return RiverWaterLevelStatus.DANGER

        if level >= self.warning_level_m:
            return RiverWaterLevelStatus.WARNING

        return RiverWaterLevelStatus.NORMAL

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "station_code": self.station_code,
            "station_name": self.station_name,
            "river_name": self.river_name,
            "district_id": self.district_id,
            "station_type": self.station_type.value,
            "source": self.source.value,
            "status": self.status.value,
            "current_level_status": (
                self.current_level_status.value
            ),
            "latest_water_level_m": (
                self.latest_water_level_m
            ),
            "latest_discharge_cumecs": (
                self.latest_discharge_cumecs
            ),
            "latest_reading_at": (
                self.latest_reading_at.isoformat()
                if self.latest_reading_at
                else None
            ),
            "is_data_stale": self.is_data_stale,
            "is_verified": self.is_verified,
            "thresholds_verified": (
                self.thresholds_verified
            ),
        }


# ============================================================
# WATER LEVEL READING
# ============================================================

class RiverWaterLevelReading(Base):
    __tablename__ = "river_water_level_readings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "river_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    external_reading_id: Mapped[
        str | None
    ] = mapped_column(
        String(150),
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

    water_level_m: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    discharge_cumecs: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    rainfall_mm: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    water_temperature_c: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    flow_velocity_mps: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    level_change_m_per_hour: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    level_status: Mapped[
        RiverWaterLevelStatus
    ] = mapped_column(
        enum_type(RiverWaterLevelStatus),
        default=RiverWaterLevelStatus.UNKNOWN,
        nullable=False,
    )

    quality: Mapped[
        RiverReadingQuality
    ] = mapped_column(
        enum_type(RiverReadingQuality),
        default=RiverReadingQuality.RAW,
        nullable=False,
        index=True,
    )

    is_anomaly: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    anomaly_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    sensor_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "river_station_sensors.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    source_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    validated_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    validated_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    quality_notes: Mapped[str | None] = (
        mapped_column(
            Text,
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

    station: Mapped["RiverStation"] = relationship(
        "RiverStation",
        back_populates="readings",
    )

    __table_args__ = (
        UniqueConstraint(
            "station_id",
            "external_reading_id",
            name="uq_river_external_reading",
        ),
        CheckConstraint(
            "discharge_cumecs IS NULL OR "
            "discharge_cumecs >= 0",
            name="ck_river_reading_discharge",
        ),
        CheckConstraint(
            "rainfall_mm IS NULL OR rainfall_mm >= 0",
            name="ck_river_reading_rainfall",
        ),
        CheckConstraint(
            "flow_velocity_mps IS NULL OR "
            "flow_velocity_mps >= 0",
            name="ck_river_reading_velocity",
        ),
        Index(
            "ix_river_reading_station_time",
            "station_id",
            "observed_at",
        ),
        Index(
            "ix_river_reading_station_quality",
            "station_id",
            "quality",
        ),
    )

    @validates(
        "water_level_m",
        "discharge_cumecs",
        "rainfall_mm",
        "water_temperature_c",
        "flow_velocity_mps",
        "level_change_m_per_hour",
        "anomaly_score",
    )
    def validate_numeric(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        nonnegative = {
            "discharge_cumecs",
            "rainfall_mm",
            "flow_velocity_mps",
        }

        return validate_finite(
            value,
            key,
            minimum=0 if key in nonnegative else None,
        )

    def validate_reading(
        self,
        *,
        verifier_id: int,
        station: RiverStation,
    ) -> None:
        if self.quality not in {
            RiverReadingQuality.RAW,
            RiverReadingQuality.PENDING_VALIDATION,
            RiverReadingQuality.SUSPECT,
        }:
            raise ValueError(
                "Reading is not awaiting validation"
            )

        if verifier_id <= 0:
            raise ValueError(
                "Invalid verifier"
            )

        if self.is_anomaly:
            raise ValueError(
                "Anomalous reading requires investigation"
            )

        if not station.is_verified:
            raise ValueError(
                "Station is not verified"
            )

        self.quality = RiverReadingQuality.VALID
        self.level_status = station.classify_level(
            self.water_level_m
        )
        self.validated_by_user_id = verifier_id
        self.validated_at = utc_now()


# ============================================================
# RIVER STATION SENSOR
# ============================================================

class RiverStationSensor(Base):
    __tablename__ = "river_station_sensors"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "river_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    sensor_code: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )

    sensor_type: Mapped[
        RiverSensorType
    ] = mapped_column(
        enum_type(RiverSensorType),
        nullable=False,
    )

    manufacturer: Mapped[str | None] = (
        mapped_column(
            String(255),
            nullable=True,
        )
    )

    model_name: Mapped[str | None] = (
        mapped_column(
            String(255),
            nullable=True,
        )
    )

    serial_number: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    status: Mapped[
        RiverSensorStatus
    ] = mapped_column(
        enum_type(RiverSensorStatus),
        default=RiverSensorStatus.UNKNOWN,
        nullable=False,
    )

    battery_percentage: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    signal_strength_dbm: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    last_seen_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    last_calibrated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    calibration_valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    calibration_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    installed_at: Mapped[datetime | None] = (
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

    station: Mapped["RiverStation"] = relationship(
        "RiverStation",
        back_populates="sensors",
    )

    __table_args__ = (
        UniqueConstraint(
            "station_id",
            "sensor_code",
            name="uq_river_station_sensor",
        ),
        CheckConstraint(
            "battery_percentage IS NULL OR "
            "(battery_percentage >= 0 "
            "AND battery_percentage <= 100)",
            name="ck_river_sensor_battery",
        ),
    )

    @property
    def is_healthy(self) -> bool:
        if self.status != RiverSensorStatus.HEALTHY:
            return False

        if self.calibration_valid_until is None:
            return False

        return (
            as_utc(self.calibration_valid_until)
            > utc_now()
        )


# ============================================================
# RIVER LEVEL FORECAST
# ============================================================

class RiverLevelForecast(Base):
    __tablename__ = "river_level_forecasts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "river_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    experiment_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    forecast_created_at: Mapped[
        datetime
    ] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    target_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    forecast_horizon_hours: Mapped[int] = (
        mapped_column(
            Integer,
            nullable=False,
        )
    )

    predicted_water_level_m: Mapped[
        float
    ] = mapped_column(
        Float,
        nullable=False,
    )

    predicted_discharge_cumecs: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    lower_bound_m: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    upper_bound_m: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    confidence_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    predicted_level_status: Mapped[
        RiverWaterLevelStatus
    ] = mapped_column(
        enum_type(RiverWaterLevelStatus),
        default=RiverWaterLevelStatus.UNKNOWN,
        nullable=False,
    )

    status: Mapped[
        RiverForecastStatus
    ] = mapped_column(
        enum_type(RiverForecastStatus),
        default=RiverForecastStatus.GENERATED,
        nullable=False,
    )

    input_data_quality_verified: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_out_of_distribution: Mapped[
        bool
    ] = mapped_column(
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

    actual_water_level_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    absolute_error_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    evaluation_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
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

    station: Mapped["RiverStation"] = relationship(
        "RiverStation",
        back_populates="forecasts",
    )

    __table_args__ = (
        CheckConstraint(
            "forecast_horizon_hours > 0",
            name="ck_river_forecast_horizon",
        ),
        CheckConstraint(
            "predicted_discharge_cumecs IS NULL OR "
            "predicted_discharge_cumecs >= 0",
            name="ck_river_forecast_discharge",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 "
            "AND confidence_score <= 1)",
            name="ck_river_forecast_confidence",
        ),
        CheckConstraint(
            "absolute_error_m IS NULL OR "
            "absolute_error_m >= 0",
            name="ck_river_forecast_error",
        ),
        CheckConstraint(
            "lower_bound_m IS NULL OR "
            "upper_bound_m IS NULL OR "
            "lower_bound_m <= upper_bound_m",
            name="ck_river_forecast_bounds",
        ),
        Index(
            "ix_river_forecast_station_target",
            "station_id",
            "target_time",
        ),
    )

    @validates(
        "predicted_water_level_m",
        "predicted_discharge_cumecs",
        "lower_bound_m",
        "upper_bound_m",
        "confidence_score",
        "actual_water_level_m",
        "absolute_error_m",
    )
    def validate_numeric(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        if key == "confidence_score":
            return validate_finite(
                value, key, 0, 1
            )

        if key in {
            "predicted_discharge_cumecs",
            "absolute_error_m",
        }:
            return validate_finite(
                value, key, minimum=0
            )

        return validate_finite(value, key)

    def record_actual(
        self,
        actual_level_m: float,
    ) -> None:
        actual = validate_finite(
            actual_level_m,
            "actual_level_m",
        )

        assert actual is not None

        self.actual_water_level_m = actual
        self.absolute_error_m = abs(
            actual - self.predicted_water_level_m
        )


# ============================================================
# RIVER STATION ADVISORY
# ============================================================

class RiverStationAdvisory(Base):
    __tablename__ = "river_station_advisories"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "river_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    advisory_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    level_status: Mapped[
        RiverWaterLevelStatus
    ] = mapped_column(
        enum_type(RiverWaterLevelStatus),
        nullable=False,
    )

    status: Mapped[
        RiverAdvisoryStatus
    ] = mapped_column(
        enum_type(RiverAdvisoryStatus),
        default=RiverAdvisoryStatus.DRAFT,
        nullable=False,
    )

    is_ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    supporting_reading_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "river_water_level_readings.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    supporting_forecast_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "river_level_forecasts.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
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

    published_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    published_at: Mapped[datetime | None] = (
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

    station: Mapped["RiverStation"] = relationship(
        "RiverStation",
        back_populates="advisories",
    )

    def request_review(self) -> None:
        if self.status != RiverAdvisoryStatus.DRAFT:
            raise ValueError(
                "Only draft advisories can be reviewed"
            )

        self.status = RiverAdvisoryStatus.PENDING_REVIEW

    def approve(
        self,
        *,
        approver_id: int,
        approval_reference: str,
    ) -> None:
        if self.status != (
            RiverAdvisoryStatus.PENDING_REVIEW
        ):
            raise ValueError(
                "Advisory is not pending review"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if (
            self.created_by_user_id is not None
            and approver_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent approval required"
            )

        if not (
            self.supporting_reading_id
            or self.supporting_forecast_id
        ):
            raise ValueError(
                "Supporting evidence is required"
            )

        self.approval_reference = require_text(
            approval_reference,
            "approval_reference",
            500,
        )
        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = RiverAdvisoryStatus.APPROVED

    def publish(self, publisher_id: int) -> None:
        """
        Call only from an authorized publication
        service after checking source validity,
        current conditions and official permissions.
        """
        if self.status != RiverAdvisoryStatus.APPROVED:
            raise ValueError(
                "Advisory is not approved"
            )

        if publisher_id <= 0:
            raise ValueError("Invalid publisher")

        if self.expires_at is not None:
            if as_utc(self.expires_at) <= utc_now():
                raise ValueError(
                    "Advisory has expired"
                )

        self.published_by_user_id = publisher_id
        self.published_at = utc_now()
        self.status = RiverAdvisoryStatus.PUBLISHED


# ============================================================
# STATION MAINTENANCE
# ============================================================

class RiverStationMaintenance(Base):
    __tablename__ = "river_station_maintenance"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "river_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    sensor_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "river_station_sensors.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    maintenance_type: Mapped[
        RiverMaintenanceType
    ] = mapped_column(
        enum_type(RiverMaintenanceType),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    performed_by_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    successful: Mapped[bool | None] = (
        mapped_column(
            Boolean,
            nullable=True,
        )
    )

    inspection_report_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    verified_by_user_id: Mapped[
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

    station: Mapped["RiverStation"] = relationship(
        "RiverStation",
        back_populates="maintenance_records",
    )

    __table_args__ = (
        Index(
            "ix_river_maintenance_station_time",
            "station_id",
            "started_at",
        ),
    )


# ============================================================
# STATION AUDIT EVENTS
# ============================================================

class RiverStationEvent(Base):
    __tablename__ = "river_station_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "river_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        RiverStationEventType
    ] = mapped_column(
        enum_type(RiverStationEventType),
        nullable=False,
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

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
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

    station: Mapped["RiverStation"] = relationship(
        "RiverStation",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_river_event_station_time",
            "station_id",
            "occurred_at",
        ),
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

RIVER_STATION_MODELS = (
    RiverStation,
    RiverWaterLevelReading,
    RiverStationSensor,
    RiverLevelForecast,
    RiverStationAdvisory,
    RiverStationMaintenance,
    RiverStationEvent,
)


__all__ = [
    "RiverStation",
    "RiverWaterLevelReading",
    "RiverStationSensor",
    "RiverLevelForecast",
    "RiverStationAdvisory",
    "RiverStationMaintenance",
    "RiverStationEvent",
    "RiverStationType",
    "RiverStationStatus",
    "RiverStationSource",
    "RiverWaterLevelStatus",
    "RiverReadingQuality",
    "RiverSensorStatus",
    "RiverSensorType",
    "RiverForecastStatus",
    "RiverAdvisoryStatus",
    "RiverMaintenanceType",
    "RiverStationEventType",
    "RIVER_STATION_MODELS",
]
