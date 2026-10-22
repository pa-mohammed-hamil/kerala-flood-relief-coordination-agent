
"""
backend/models/weather_station.py

Kerala Flood Relief Coordination Agent
Python 3.11+ | SQLAlchemy 2.x

Weather station and meteorological monitoring models.

Features:
- Automatic and manual weather stations
- Station registration, verification and activation
- Real-time meteorological observations
- Rainfall intensity and accumulated rainfall
- Temperature, humidity and atmospheric pressure
- Wind speed, direction and gust monitoring
- Weather sensor health and calibration
- Rainfall threshold configurations
- Extreme weather and flood-risk alerts
- Weather station maintenance
- Data quality checks and anomaly detection
- Forecast and ML feature references
- AI-assisted weather risk assessments
- Station communication and telemetry health
- Audit history

Safety:
- Sensor observations are not automatically treated as verified.
- AI predictions are advisory, not official warnings.
- Public alerts require authorized human approval.
- Sensor faults must not silently suppress warnings.
- Weather observations are stored with UTC timestamps.
"""

from __future__ import annotations

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


def required_text(
    value: str,
    field_name: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")

    result = value.strip()

    if len(result) > maximum:
        raise ValueError(
            f"{field_name} exceeds {maximum} characters"
        )

    return result


def finite_number(
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
            f"{field_name} must be a finite number"
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


def valid_actor_id(value: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        raise ValueError(
            "Valid actor UUID required"
        ) from None


def normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMS
# ============================================================

class WeatherStationType(str, enum.Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    RAIN_GAUGE = "rain_gauge"
    AGRO_METEOROLOGICAL = "agro_meteorological"
    HYDROMETEOROLOGICAL = "hydrometeorological"
    MOBILE = "mobile"
    RESEARCH = "research"
    IOT = "iot"
    SATELLITE_DERIVED = "satellite_derived"
    OTHER = "other"


class WeatherStationStatus(str, enum.Enum):
    REGISTERED = "registered"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    SUSPENDED = "suspended"
    DECOMMISSIONED = "decommissioned"


class WeatherStationOwner(str, enum.Enum):
    IMD = "imd"
    KSDMA = "ksdma"
    GOVERNMENT = "government"
    RESEARCH_INSTITUTE = "research_institute"
    UNIVERSITY = "university"
    PRIVATE = "private"
    COMMUNITY = "community"
    OTHER = "other"


class WeatherSensorType(str, enum.Enum):
    RAIN_GAUGE = "rain_gauge"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    SOLAR_RADIATION = "solar_radiation"
    UV_INDEX = "uv_index"
    SOIL_MOISTURE = "soil_moisture"
    SOIL_TEMPERATURE = "soil_temperature"
    VISIBILITY = "visibility"
    LIGHTNING = "lightning"
    WATER_LEVEL = "water_level"
    OTHER = "other"


class WeatherSensorStatus(str, enum.Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    CALIBRATION_DUE = "calibration_due"
    FAULTY = "faulty"
    OFFLINE = "offline"
    REPLACED = "replaced"
    RETIRED = "retired"


class WeatherDataQuality(str, enum.Enum):
    RAW = "raw"
    PENDING = "pending"
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"
    MISSING = "missing"
    CORRECTED = "corrected"


class WeatherObservationSource(str, enum.Enum):
    SENSOR = "sensor"
    MANUAL = "manual"
    API = "api"
    SATELLITE = "satellite"
    RADAR = "radar"
    IMPORT = "import"
    OTHER = "other"


class WeatherAlertType(str, enum.Enum):
    HEAVY_RAINFALL = "heavy_rainfall"
    EXTREME_RAINFALL = "extreme_rainfall"
    FLASH_FLOOD_RISK = "flash_flood_risk"
    FLOOD_RISK = "flood_risk"
    LANDSLIDE_RISK = "landslide_risk"
    STRONG_WIND = "strong_wind"
    THUNDERSTORM = "thunderstorm"
    LIGHTNING = "lightning"
    EXTREME_TEMPERATURE = "extreme_temperature"
    SENSOR_FAILURE = "sensor_failure"
    DATA_ANOMALY = "data_anomaly"
    OTHER = "other"


class WeatherAlertSeverity(str, enum.Enum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class WeatherAlertStatus(str, enum.Enum):
    DETECTED = "detected"
    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    APPROVED = "approved"
    PUBLISHED = "published"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class WeatherMaintenanceType(str, enum.Enum):
    INSPECTION = "inspection"
    CALIBRATION = "calibration"
    SENSOR_REPLACEMENT = "sensor_replacement"
    BATTERY_REPLACEMENT = "battery_replacement"
    NETWORK_REPAIR = "network_repair"
    FIRMWARE_UPDATE = "firmware_update"
    CLEANING = "cleaning"
    GENERAL_REPAIR = "general_repair"
    OTHER = "other"


class WeatherMaintenanceStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WeatherEventType(str, enum.Enum):
    REGISTERED = "registered"
    VERIFIED = "verified"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    STATUS_CHANGED = "status_changed"
    OBSERVATION_RECEIVED = "observation_received"
    DATA_VALIDATED = "data_validated"
    ANOMALY_DETECTED = "anomaly_detected"
    SENSOR_FAULT = "sensor_fault"
    SENSOR_RECOVERED = "sensor_recovered"
    CALIBRATION_COMPLETED = "calibration_completed"
    MAINTENANCE_STARTED = "maintenance_started"
    MAINTENANCE_COMPLETED = "maintenance_completed"
    ALERT_DETECTED = "alert_detected"
    ALERT_APPROVED = "alert_approved"
    ALERT_PUBLISHED = "alert_published"
    COMMUNICATION_LOST = "communication_lost"
    COMMUNICATION_RESTORED = "communication_restored"
    NOTE_ADDED = "note_added"


# ============================================================
# WEATHER STATION
# ============================================================

class WeatherStation(Base):
    __tablename__ = "weather_stations"

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

    station_type: Mapped[
        WeatherStationType
    ] = mapped_column(
        enum_type(WeatherStationType),
        nullable=False,
    )

    owner_type: Mapped[
        WeatherStationOwner
    ] = mapped_column(
        enum_type(WeatherStationOwner),
        nullable=False,
    )

    operator_name: Mapped[str | None] = mapped_column(
        String(255),
    )

    external_station_id: Mapped[
        str | None
    ] = mapped_column(
        String(150),
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

    location_description: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    latitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    longitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    elevation_m: Mapped[float | None] = mapped_column(
        Float,
    )

    catchment_name: Mapped[str | None] = mapped_column(
        String(255),
    )

    river_basin_name: Mapped[str | None] = mapped_column(
        String(255),
    )

    status: Mapped[
        WeatherStationStatus
    ] = mapped_column(
        enum_type(WeatherStationStatus),
        default=WeatherStationStatus.REGISTERED,
        nullable=False,
        index=True,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_public_data_source: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_flood_monitoring_enabled: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    observation_interval_minutes: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=15,
        nullable=False,
    )

    expected_heartbeat_minutes: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=30,
        nullable=False,
    )

    firmware_version: Mapped[
        str | None
    ] = mapped_column(
        String(100),
    )

    communication_protocol: Mapped[
        str | None
    ] = mapped_column(
        String(100),
    )

    power_source: Mapped[str | None] = mapped_column(
        String(100),
    )

    battery_percentage: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    signal_strength_dbm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    last_heartbeat_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    last_observation_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    last_maintenance_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    next_maintenance_due_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    installed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    verified_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    activated_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    created_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
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

    sensors: Mapped[
        list["WeatherSensor"]
    ] = relationship(
        "WeatherSensor",
        back_populates="station",
    )

    observations: Mapped[
        list["WeatherObservation"]
    ] = relationship(
        "WeatherObservation",
        back_populates="station",
    )

    rainfall_readings: Mapped[
        list["RainfallReading"]
    ] = relationship(
        "RainfallReading",
        back_populates="station",
    )

    thresholds: Mapped[
        list["WeatherThreshold"]
    ] = relationship(
        "WeatherThreshold",
        back_populates="station",
    )

    alerts: Mapped[
        list["WeatherStationAlert"]
    ] = relationship(
        "WeatherStationAlert",
        back_populates="station",
    )

    health_snapshots: Mapped[
        list["WeatherStationHealth"]
    ] = relationship(
        "WeatherStationHealth",
        back_populates="station",
    )

    maintenance_records: Mapped[
        list["WeatherStationMaintenance"]
    ] = relationship(
        "WeatherStationMaintenance",
        back_populates="station",
    )

    events: Mapped[
        list["WeatherStationEvent"]
    ] = relationship(
        "WeatherStationEvent",
        back_populates="station",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_weather_station_district",
        ),
        CheckConstraint(
            "latitude BETWEEN -90 AND 90",
            name="ck_weather_station_latitude",
        ),
        CheckConstraint(
            "longitude BETWEEN -180 AND 180",
            name="ck_weather_station_longitude",
        ),
        CheckConstraint(
            "observation_interval_minutes > 0",
            name="ck_weather_station_interval",
        ),
        CheckConstraint(
            "expected_heartbeat_minutes > 0",
            name="ck_weather_station_heartbeat",
        ),
        CheckConstraint(
            "battery_percentage IS NULL OR "
            "(battery_percentage >= 0 AND "
            "battery_percentage <= 100)",
            name="ck_weather_station_battery",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_weather_station_version",
        ),
        Index(
            "ix_weather_station_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_weather_station_location",
            "latitude",
            "longitude",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("station_code", "station_name")
    def validate_required(
        self,
        key: str,
        value: str,
    ) -> str:
        return required_text(
            value,
            key,
            100 if key == "station_code" else 255,
        )

    @validates(
        "latitude",
        "longitude",
        "battery_percentage",
    )
    def validate_numeric(
        self,
        key: str,
        value: float,
    ) -> float:
        bounds = {
            "latitude": (-90, 90),
            "longitude": (-180, 180),
            "battery_percentage": (0, 100),
        }

        lower, upper = bounds[key]

        result = finite_number(
            value,
            key,
            lower,
            upper,
        )

        if result is None and key != "battery_percentage":
            raise ValueError(f"{key} is required")

        return result

    @property
    def is_operational(self) -> bool:
        if not self.is_verified:
            return False

        if self.status != WeatherStationStatus.ACTIVE:
            return False

        if self.last_heartbeat_at is None:
            return False

        heartbeat_age = (
            utc_now()
            - normalize_datetime(self.last_heartbeat_at)
        ).total_seconds() / 60

        return (
            0 <= heartbeat_age
            <= self.expected_heartbeat_minutes
        )

    def request_verification(self) -> None:
        if self.status != WeatherStationStatus.REGISTERED:
            raise ValueError(
                "Station cannot request verification"
            )

        self.status = (
            WeatherStationStatus.PENDING_VERIFICATION
        )

    def verify(self, authorized_user_id: str) -> None:
        actor = valid_actor_id(authorized_user_id)

        if self.status != (
            WeatherStationStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Station is not pending verification"
            )

        if actor == self.created_by_user_id:
            raise ValueError(
                "Independent verification required"
            )

        self.is_verified = True
        self.verified_by_user_id = actor
        self.verified_at = utc_now()
        self.status = WeatherStationStatus.VERIFIED

    def activate(self, authorized_user_id: str) -> None:
        actor = valid_actor_id(authorized_user_id)

        if not self.is_verified:
            raise ValueError(
                "Station verification required"
            )

        if self.status not in {
            WeatherStationStatus.VERIFIED,
            WeatherStationStatus.OFFLINE,
            WeatherStationStatus.MAINTENANCE,
            WeatherStationStatus.DEGRADED,
        }:
            raise ValueError(
                "Station cannot be activated"
            )

        self.activated_by_user_id = actor
        self.activated_at = utc_now()
        self.status = WeatherStationStatus.ACTIVE

    def record_heartbeat(
        self,
        *,
        observed_at: datetime | None = None,
        battery_percentage: float | None = None,
        signal_strength_dbm: float | None = None,
    ) -> None:
        self.last_heartbeat_at = (
            normalize_datetime(observed_at)
            if observed_at is not None
            else utc_now()
        )

        if battery_percentage is not None:
            self.battery_percentage = battery_percentage

        if signal_strength_dbm is not None:
            self.signal_strength_dbm = finite_number(
                signal_strength_dbm,
                "signal_strength_dbm",
            )

    def mark_degraded(self) -> None:
        if self.status == WeatherStationStatus.ACTIVE:
            self.status = WeatherStationStatus.DEGRADED

    def mark_offline(self) -> None:
        if self.status != (
            WeatherStationStatus.DECOMMISSIONED
        ):
            self.status = WeatherStationStatus.OFFLINE

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "station_code": self.station_code,
            "station_name": self.station_name,
            "station_type": self.station_type.value,
            "district_id": self.district_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "status": self.status.value,
            "is_verified": self.is_verified,
            "is_operational": self.is_operational,
            "last_observation_at": (
                self.last_observation_at.isoformat()
                if self.last_observation_at
                else None
            ),
        }


# ============================================================
# WEATHER SENSOR
# ============================================================

class WeatherSensor(Base):
    __tablename__ = "weather_sensors"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "weather_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    sensor_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    sensor_type: Mapped[
        WeatherSensorType
    ] = mapped_column(
        enum_type(WeatherSensorType),
        nullable=False,
    )

    manufacturer: Mapped[str | None] = mapped_column(
        String(150),
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
    )

    serial_number: Mapped[str | None] = mapped_column(
        String(150),
    )

    measurement_unit: Mapped[
        str | None
    ] = mapped_column(
        String(50),
    )

    status: Mapped[
        WeatherSensorStatus
    ] = mapped_column(
        enum_type(WeatherSensorStatus),
        default=WeatherSensorStatus.ACTIVE,
        nullable=False,
    )

    minimum_value: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    maximum_value: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    accuracy_tolerance: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    calibration_offset: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    last_calibrated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    next_calibration_due_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    installed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    last_reading_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    is_primary_sensor: Mapped[bool] = mapped_column(
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

    station: Mapped["WeatherStation"] = relationship(
        "WeatherStation",
        back_populates="sensors",
    )

    __table_args__ = (
        UniqueConstraint(
            "station_id",
            "sensor_code",
            name="uq_weather_sensor_station_code",
        ),
        CheckConstraint(
            "accuracy_tolerance IS NULL OR "
            "accuracy_tolerance >= 0",
            name="ck_weather_sensor_tolerance",
        ),
        CheckConstraint(
            "minimum_value IS NULL OR "
            "maximum_value IS NULL OR "
            "minimum_value <= maximum_value",
            name="ck_weather_sensor_range",
        ),
    )


# ============================================================
# WEATHER OBSERVATION
# ============================================================

class WeatherObservation(Base):
    __tablename__ = "weather_observations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "weather_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    external_observation_id: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    source: Mapped[
        WeatherObservationSource
    ] = mapped_column(
        enum_type(WeatherObservationSource),
        default=WeatherObservationSource.SENSOR,
        nullable=False,
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    temperature_c: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    humidity_percent: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    pressure_hpa: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    wind_speed_mps: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    wind_gust_mps: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    wind_direction_degrees: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    rainfall_mm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    rainfall_intensity_mm_per_hour: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    solar_radiation_wm2: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    uv_index: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    visibility_m: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    soil_moisture_percent: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    soil_temperature_c: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    quality: Mapped[
        WeatherDataQuality
    ] = mapped_column(
        enum_type(WeatherDataQuality),
        default=WeatherDataQuality.RAW,
        nullable=False,
        index=True,
    )

    is_anomaly: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    anomaly_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    quality_notes: Mapped[
        str | None
    ] = mapped_column(
        Text,
    )

    validated_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    validated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    raw_payload_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
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

    station: Mapped["WeatherStation"] = relationship(
        "WeatherStation",
        back_populates="observations",
    )

    __table_args__ = (
        CheckConstraint(
            "humidity_percent IS NULL OR "
            "(humidity_percent BETWEEN 0 AND 100)",
            name="ck_weather_obs_humidity",
        ),
        CheckConstraint(
            "rainfall_mm IS NULL OR rainfall_mm >= 0",
            name="ck_weather_obs_rainfall",
        ),
        CheckConstraint(
            "rainfall_intensity_mm_per_hour IS NULL OR "
            "rainfall_intensity_mm_per_hour >= 0",
            name="ck_weather_obs_rainfall_intensity",
        ),
        CheckConstraint(
            "wind_speed_mps IS NULL OR wind_speed_mps >= 0",
            name="ck_weather_obs_wind_speed",
        ),
        CheckConstraint(
            "wind_gust_mps IS NULL OR wind_gust_mps >= 0",
            name="ck_weather_obs_wind_gust",
        ),
        CheckConstraint(
            "wind_direction_degrees IS NULL OR "
            "(wind_direction_degrees BETWEEN 0 AND 360)",
            name="ck_weather_obs_wind_direction",
        ),
        CheckConstraint(
            "pressure_hpa IS NULL OR pressure_hpa > 0",
            name="ck_weather_obs_pressure",
        ),
        CheckConstraint(
            "soil_moisture_percent IS NULL OR "
            "(soil_moisture_percent BETWEEN 0 AND 100)",
            name="ck_weather_obs_soil_moisture",
        ),
        CheckConstraint(
            "anomaly_score IS NULL OR "
            "(anomaly_score BETWEEN 0 AND 1)",
            name="ck_weather_obs_anomaly",
        ),
        Index(
            "ix_weather_obs_station_time",
            "station_id",
            "observed_at",
        ),
        Index(
            "ix_weather_obs_station_quality",
            "station_id",
            "quality",
        ),
    )

    @validates(
        "humidity_percent",
        "rainfall_mm",
        "rainfall_intensity_mm_per_hour",
        "wind_speed_mps",
        "wind_gust_mps",
        "wind_direction_degrees",
        "pressure_hpa",
        "soil_moisture_percent",
        "anomaly_score",
    )
    def validate_values(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        bounds = {
            "humidity_percent": (0, 100),
            "rainfall_mm": (0, None),
            "rainfall_intensity_mm_per_hour": (0, None),
            "wind_speed_mps": (0, None),
            "wind_gust_mps": (0, None),
            "wind_direction_degrees": (0, 360),
            "pressure_hpa": (0, None),
            "soil_moisture_percent": (0, 100),
            "anomaly_score": (0, 1),
        }

        lower, upper = bounds[key]

        return finite_number(
            value,
            key,
            lower,
            upper,
        )

    def validate_observation(
        self,
        validator_id: str,
    ) -> None:
        actor = valid_actor_id(validator_id)

        if self.quality == WeatherDataQuality.INVALID:
            raise ValueError(
                "Invalid observation requires correction"
            )

        if self.is_anomaly:
            raise ValueError(
                "Anomalous observation requires review"
            )

        self.quality = WeatherDataQuality.VALID
        self.validated_by_user_id = actor
        self.validated_at = utc_now()


# ============================================================
# RAINFALL READING
# ============================================================

class RainfallReading(Base):
    __tablename__ = "rainfall_readings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "weather_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    interval_minutes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    rainfall_mm: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    rainfall_intensity_mm_per_hour: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    cumulative_1h_mm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    cumulative_3h_mm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    cumulative_6h_mm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    cumulative_12h_mm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    cumulative_24h_mm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    cumulative_48h_mm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    cumulative_72h_mm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    quality: Mapped[
        WeatherDataQuality
    ] = mapped_column(
        enum_type(WeatherDataQuality),
        default=WeatherDataQuality.RAW,
        nullable=False,
    )

    is_anomaly: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    source_observation_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "weather_observations.id",
            ondelete="SET NULL",
        ),
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

    station: Mapped["WeatherStation"] = relationship(
        "WeatherStation",
        back_populates="rainfall_readings",
    )

    __table_args__ = (
        CheckConstraint(
            "interval_minutes > 0",
            name="ck_rainfall_interval",
        ),
        CheckConstraint(
            "rainfall_mm >= 0",
            name="ck_rainfall_amount",
        ),
        CheckConstraint(
            "rainfall_intensity_mm_per_hour IS NULL OR "
            "rainfall_intensity_mm_per_hour >= 0",
            name="ck_rainfall_intensity",
        ),
        CheckConstraint(
            "cumulative_1h_mm IS NULL OR cumulative_1h_mm >= 0",
            name="ck_rainfall_1h",
        ),
        CheckConstraint(
            "cumulative_3h_mm IS NULL OR cumulative_3h_mm >= 0",
            name="ck_rainfall_3h",
        ),
        CheckConstraint(
            "cumulative_6h_mm IS NULL OR cumulative_6h_mm >= 0",
            name="ck_rainfall_6h",
        ),
        CheckConstraint(
            "cumulative_12h_mm IS NULL OR cumulative_12h_mm >= 0",
            name="ck_rainfall_12h",
        ),
        CheckConstraint(
            "cumulative_24h_mm IS NULL OR cumulative_24h_mm >= 0",
            name="ck_rainfall_24h",
        ),
        CheckConstraint(
            "cumulative_48h_mm IS NULL OR cumulative_48h_mm >= 0",
            name="ck_rainfall_48h",
        ),
        CheckConstraint(
            "cumulative_72h_mm IS NULL OR cumulative_72h_mm >= 0",
            name="ck_rainfall_72h",
        ),
        Index(
            "ix_rainfall_station_time",
            "station_id",
            "observed_at",
        ),
    )


# ============================================================
# WEATHER THRESHOLD
# ============================================================

class WeatherThreshold(Base):
    __tablename__ = "weather_thresholds"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "weather_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    threshold_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    alert_type: Mapped[
        WeatherAlertType
    ] = mapped_column(
        enum_type(WeatherAlertType),
        nullable=False,
    )

    severity: Mapped[
        WeatherAlertSeverity
    ] = mapped_column(
        enum_type(WeatherAlertSeverity),
        nullable=False,
    )

    metric_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    operator: Mapped[str] = mapped_column(
        String(10),
        default=">=",
        nullable=False,
    )

    threshold_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    measurement_unit: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    evaluation_window_minutes: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=60,
        nullable=False,
    )

    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    requires_human_review: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    configured_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    approved_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    approved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    station: Mapped["WeatherStation"] = relationship(
        "WeatherStation",
        back_populates="thresholds",
    )

    __table_args__ = (
        CheckConstraint(
            "operator IN ('>', '>=', '<', '<=', '==')",
            name="ck_weather_threshold_operator",
        ),
        CheckConstraint(
            "evaluation_window_minutes > 0",
            name="ck_weather_threshold_window",
        ),
    )

    def evaluate(self, value: float) -> bool:
        value = finite_number(value, "value")

        if value is None:
            return False

        threshold = self.threshold_value

        operations = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
        }

        if self.operator not in operations:
            raise ValueError("Unsupported threshold operator")

        return operations[self.operator](value, threshold)


# ============================================================
# WEATHER STATION ALERT
# ============================================================

class WeatherStationAlert(Base):
    __tablename__ = "weather_station_alerts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "weather_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    threshold_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "weather_thresholds.id",
            ondelete="SET NULL",
        ),
    )

    alert_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    alert_type: Mapped[
        WeatherAlertType
    ] = mapped_column(
        enum_type(WeatherAlertType),
        nullable=False,
    )

    severity: Mapped[
        WeatherAlertSeverity
    ] = mapped_column(
        enum_type(WeatherAlertSeverity),
        nullable=False,
    )

    status: Mapped[
        WeatherAlertStatus
    ] = mapped_column(
        enum_type(WeatherAlertStatus),
        default=WeatherAlertStatus.DETECTED,
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    observed_value: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    threshold_value: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    is_ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    ai_risk_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    ai_model_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    evidence_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    verified_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    approved_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    approved_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    published_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    published_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    station: Mapped["WeatherStation"] = relationship(
        "WeatherStation",
        back_populates="alerts",
    )

    __table_args__ = (
        CheckConstraint(
            "ai_risk_score IS NULL OR "
            "(ai_risk_score BETWEEN 0 AND 1)",
            name="ck_weather_alert_ai_risk",
        ),
        Index(
            "ix_weather_alert_station_status",
            "station_id",
            "status",
        ),
    )

    def submit_for_review(self) -> None:
        if self.status != WeatherAlertStatus.DETECTED:
            raise ValueError(
                "Alert cannot be submitted for review"
            )

        self.status = WeatherAlertStatus.PENDING_REVIEW

    def verify(self, verifier_id: str) -> None:
        actor = valid_actor_id(verifier_id)

        if self.status != WeatherAlertStatus.PENDING_REVIEW:
            raise ValueError(
                "Alert is not pending review"
            )

        self.verified_by_user_id = actor
        self.verified_at = utc_now()
        self.status = WeatherAlertStatus.VERIFIED

    def approve(self, approver_id: str) -> None:
        actor = valid_actor_id(approver_id)

        if self.status != WeatherAlertStatus.VERIFIED:
            raise ValueError(
                "Alert must be verified"
            )

        if actor == self.verified_by_user_id:
            raise ValueError(
                "Independent alert approval required"
            )

        self.approved_by_user_id = actor
        self.approved_at = utc_now()
        self.status = WeatherAlertStatus.APPROVED

    def publish(self, publisher_id: str) -> None:
        actor = valid_actor_id(publisher_id)

        if self.status != WeatherAlertStatus.APPROVED:
            raise ValueError(
                "Human approval required before publication"
            )

        self.published_by_user_id = actor
        self.published_at = utc_now()
        self.status = WeatherAlertStatus.PUBLISHED

    def resolve(self) -> None:
        if self.status != WeatherAlertStatus.PUBLISHED:
            raise ValueError(
                "Only published alerts can be resolved"
            )

        self.resolved_at = utc_now()
        self.status = WeatherAlertStatus.RESOLVED


# ============================================================
# WEATHER STATION HEALTH
# ============================================================

class WeatherStationHealth(Base):
    __tablename__ = "weather_station_health"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "weather_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    battery_percentage: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    signal_strength_dbm: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    uptime_percentage: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    sensor_health_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    data_completeness_percentage: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    expected_readings: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    received_readings: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    missing_readings: Mapped[
        int | None
    ] = mapped_column(
        Integer,
    )

    anomaly_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    faulty_sensor_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    is_communication_healthy: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_maintenance: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    ai_failure_probability: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    ai_model_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
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

    station: Mapped["WeatherStation"] = relationship(
        "WeatherStation",
        back_populates="health_snapshots",
    )

    __table_args__ = (
        CheckConstraint(
            "battery_percentage IS NULL OR "
            "(battery_percentage BETWEEN 0 AND 100)",
            name="ck_weather_health_battery",
        ),
        CheckConstraint(
            "uptime_percentage IS NULL OR "
            "(uptime_percentage BETWEEN 0 AND 100)",
            name="ck_weather_health_uptime",
        ),
        CheckConstraint(
            "sensor_health_score IS NULL OR "
            "(sensor_health_score BETWEEN 0 AND 1)",
            name="ck_weather_health_sensor_score",
        ),
        CheckConstraint(
            "data_completeness_percentage IS NULL OR "
            "(data_completeness_percentage BETWEEN 0 AND 100)",
            name="ck_weather_health_completeness",
        ),
        CheckConstraint(
            "ai_failure_probability IS NULL OR "
            "(ai_failure_probability BETWEEN 0 AND 1)",
            name="ck_weather_health_failure",
        ),
        CheckConstraint(
            "anomaly_count >= 0 AND faulty_sensor_count >= 0",
            name="ck_weather_health_counts",
        ),
        CheckConstraint(
            "(expected_readings IS NULL OR expected_readings >= 0) "
            "AND (received_readings IS NULL OR received_readings >= 0) "
            "AND (missing_readings IS NULL OR missing_readings >= 0)",
            name="ck_weather_health_readings",
        ),
        Index(
            "ix_weather_health_station_time",
            "station_id",
            "recorded_at",
        ),
    )


# ============================================================
# WEATHER STATION MAINTENANCE
# ============================================================

class WeatherStationMaintenance(Base):
    __tablename__ = "weather_station_maintenance"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "weather_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    sensor_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "weather_sensors.id",
            ondelete="SET NULL",
        ),
    )

    maintenance_type: Mapped[
        WeatherMaintenanceType
    ] = mapped_column(
        enum_type(WeatherMaintenanceType),
        nullable=False,
    )

    status: Mapped[
        WeatherMaintenanceStatus
    ] = mapped_column(
        enum_type(WeatherMaintenanceStatus),
        default=WeatherMaintenanceStatus.SCHEDULED,
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    started_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    technician_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    performed_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    verified_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    findings: Mapped[str | None] = mapped_column(
        Text,
    )

    action_taken: Mapped[str | None] = mapped_column(
        Text,
    )

    cost_amount: Mapped[float | None] = mapped_column(
        Float,
    )

    next_maintenance_due_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    station: Mapped["WeatherStation"] = relationship(
        "WeatherStation",
        back_populates="maintenance_records",
    )

    __table_args__ = (
        CheckConstraint(
            "cost_amount IS NULL OR cost_amount >= 0",
            name="ck_weather_maintenance_cost",
        ),
        Index(
            "ix_weather_maintenance_station_status",
            "station_id",
            "status",
        ),
    )

    def start(self) -> None:
        if self.status != WeatherMaintenanceStatus.SCHEDULED:
            raise ValueError(
                "Maintenance is not scheduled"
            )

        self.started_at = utc_now()
        self.status = WeatherMaintenanceStatus.IN_PROGRESS

    def complete(
        self,
        *,
        technician_user_id: str,
        verifier_user_id: str,
        findings: str,
    ) -> None:
        technician = valid_actor_id(technician_user_id)
        verifier = valid_actor_id(verifier_user_id)

        if self.status != (
            WeatherMaintenanceStatus.IN_PROGRESS
        ):
            raise ValueError(
                "Maintenance is not in progress"
            )

        if technician == verifier:
            raise ValueError(
                "Independent maintenance verification required"
            )

        self.performed_by_user_id = technician
        self.verified_by_user_id = verifier
        self.findings = required_text(
            findings,
            "findings",
            10000,
        )
        self.completed_at = utc_now()
        self.status = WeatherMaintenanceStatus.COMPLETED


# ============================================================
# WEATHER STATION AUDIT EVENT
# ============================================================

class WeatherStationEvent(Base):
    __tablename__ = "weather_station_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    station_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "weather_stations.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        WeatherEventType
    ] = mapped_column(
        enum_type(WeatherEventType),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    previous_status: Mapped[
        WeatherStationStatus | None
    ] = mapped_column(
        enum_type(WeatherStationStatus),
    )

    new_status: Mapped[
        WeatherStationStatus | None
    ] = mapped_column(
        enum_type(WeatherStationStatus),
    )

    actor_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
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

    station: Mapped["WeatherStation"] = relationship(
        "WeatherStation",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_weather_event_station_time",
            "station_id",
            "occurred_at",
        ),
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

WEATHER_STATION_MODELS = (
    WeatherStation,
    WeatherSensor,
    WeatherObservation,
    RainfallReading,
    WeatherThreshold,
    WeatherStationAlert,
    WeatherStationHealth,
    WeatherStationMaintenance,
    WeatherStationEvent,
)


__all__ = [
    "WeatherStation",
    "WeatherSensor",
    "WeatherObservation",
    "RainfallReading",
    "WeatherThreshold",
    "WeatherStationAlert",
    "WeatherStationHealth",
    "WeatherStationMaintenance",
    "WeatherStationEvent",
    "WeatherStationType",
    "WeatherStationStatus",
    "WeatherStationOwner",
    "WeatherSensorType",
    "WeatherSensorStatus",
    "WeatherDataQuality",
    "WeatherObservationSource",
    "WeatherAlertType",
    "WeatherAlertSeverity",
    "WeatherAlertStatus",
    "WeatherMaintenanceType",
    "WeatherMaintenanceStatus",
    "WeatherEventType",
    "WEATHER_STATION_MODELS",
]
