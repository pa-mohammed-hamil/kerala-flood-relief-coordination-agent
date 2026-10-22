from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/flood.py

Pydantic v2 schemas for:
- Flood monitoring
- Rainfall observations and forecasts
- River-level monitoring
- Dam monitoring
- Flood-risk predictions
- Flood severity assessments
- Flood alerts
- Geographic flood-risk mapping
- Satellite flood observations
- Human-reviewed emergency warnings
- Model prediction metadata
- Data provenance and quality

IMPORTANT:
Predictions are advisory, not official flood warnings.
Only authorized personnel may approve public alerts.
Missing observations must never be treated as zero.
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
            f"Unsupported Kerala district: {value}"
        )

    return district


def validate_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Datetime must contain timezone information"
        )

    return value


# ============================================================
# BASE SCHEMA
# ============================================================

class FloodBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
    )


# ============================================================
# ENUMS
# ============================================================

class FloodRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class FloodSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class FloodAlertStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISPATCHED = "dispatched"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class FloodAlertType(str, Enum):
    HEAVY_RAINFALL = "heavy_rainfall"
    RIVER_FLOODING = "river_flooding"
    FLASH_FLOOD = "flash_flood"
    URBAN_FLOODING = "urban_flooding"
    DAM_RELATED = "dam_related"
    COASTAL_FLOODING = "coastal_flooding"
    GENERAL_FLOOD = "general_flood"


class DataSourceType(str, Enum):
    IMD = "imd"
    CWC = "cwc"
    KSDMA = "ksdma"
    SACHET = "sachet"
    DAM_OPERATOR = "dam_operator"
    WEATHER_STATION = "weather_station"
    RIVER_STATION = "river_station"
    SATELLITE = "satellite"
    FIELD_REPORT = "field_report"
    OPEN_METEO = "open_meteo"
    MODEL = "model"
    OTHER = "other"


class ObservationQuality(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    SUSPECT = "suspect"
    STALE = "stale"
    MISSING = "missing"


class TrendDirection(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    UNKNOWN = "unknown"


class PredictionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


class DamGateStatus(str, Enum):
    CLOSED = "closed"
    PARTIALLY_OPEN = "partially_open"
    OPEN = "open"
    UNKNOWN = "unknown"


# ============================================================
# GEOGRAPHIC SCHEMAS
# ============================================================

class GeoPoint(FloodBase):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class FloodLocation(GeoPoint):
    district: str

    place_name: str | None = None
    panchayat: str | None = None
    municipality: str | None = None
    ward: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class GeoBoundingBox(FloodBase):
    min_latitude: float = Field(ge=-90, le=90)
    min_longitude: float = Field(ge=-180, le=180)

    max_latitude: float = Field(ge=-90, le=90)
    max_longitude: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min_latitude >= self.max_latitude:
            raise ValueError(
                "Minimum latitude must be less than maximum"
            )

        if self.min_longitude >= self.max_longitude:
            raise ValueError(
                "Minimum longitude must be less than maximum"
            )

        return self


# ============================================================
# DATA PROVENANCE
# ============================================================

class FloodDataSource(FloodBase):
    source_type: DataSourceType

    source_name: str = Field(
        min_length=1,
        max_length=200,
    )

    reference_id: str | None = None
    source_url: str | None = None

    observed_at: datetime | None = None
    retrieved_at: datetime = Field(
        default_factory=utc_now
    )

    quality: ObservationQuality = (
        ObservationQuality.UNVERIFIED
    )

    verified: bool = False

    @field_validator("observed_at", "retrieved_at")
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is not None:
            validate_timezone(value)

        return value


# ============================================================
# RAINFALL OBSERVATIONS
# ============================================================

class RainfallObservation(FloodBase):
    observation_id: UUID = Field(
        default_factory=uuid4
    )

    station_id: str | None = None
    location: FloodLocation

    rainfall_mm: float = Field(ge=0)

    accumulation_hours: float = Field(
        gt=0,
        le=720,
    )

    observed_at: datetime

    source: FloodDataSource

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_timezone(value)


class RainfallForecast(FloodBase):
    forecast_id: UUID = Field(
        default_factory=uuid4
    )

    district: str

    predicted_rainfall_mm: float | None = Field(
        default=None,
        ge=0,
    )

    forecast_horizon_hours: int = Field(
        ge=1,
        le=240,
    )

    probability_of_precipitation: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    model_name: str
    model_version: str | None = None

    issued_at: datetime = Field(
        default_factory=utc_now
    )

    valid_until: datetime

    source: FloodDataSource | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @model_validator(mode="after")
    def validate_forecast_period(self):
        validate_timezone(self.issued_at)
        validate_timezone(self.valid_until)

        if self.valid_until <= self.issued_at:
            raise ValueError(
                "Forecast expiry must follow issue time"
            )

        return self


# ============================================================
# RIVER MONITORING
# ============================================================

class RiverLevelObservation(FloodBase):
    observation_id: UUID = Field(
        default_factory=uuid4
    )

    river_name: str
    station_id: str

    location: FloodLocation

    water_level_m: float

    warning_level_m: float | None = None
    danger_level_m: float | None = None

    discharge_m3s: float | None = Field(
        default=None,
        ge=0,
    )

    trend: TrendDirection = TrendDirection.UNKNOWN

    observed_at: datetime

    source: FloodDataSource

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_timezone(value)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if (
            self.warning_level_m is not None
            and self.danger_level_m is not None
            and self.warning_level_m > self.danger_level_m
        ):
            raise ValueError(
                "Warning level must not exceed danger level"
            )

        return self


class RiverLevelForecast(FloodBase):
    forecast_id: UUID = Field(
        default_factory=uuid4
    )

    river_name: str
    station_id: str

    predicted_level_m: float | None = None

    forecast_horizon_hours: int = Field(
        ge=1,
        le=240,
    )

    model_name: str
    model_version: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# DAM MONITORING
# ============================================================

class DamObservation(FloodBase):
    observation_id: UUID = Field(
        default_factory=uuid4
    )

    dam_id: str
    dam_name: str

    district: str

    water_level_m: float | None = None

    storage_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    inflow_m3s: float | None = Field(
        default=None,
        ge=0,
    )

    outflow_m3s: float | None = Field(
        default=None,
        ge=0,
    )

    gate_status: DamGateStatus = (
        DamGateStatus.UNKNOWN
    )

    observed_at: datetime
    source: FloodDataSource

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_timezone(value)


# ============================================================
# FLOOD PREDICTION REQUEST
# ============================================================

class FloodPredictionRequest(FloodBase):
    prediction_id: UUID = Field(
        default_factory=uuid4
    )

    district: str

    location: GeoPoint | None = None

    forecast_horizon_hours: int = Field(
        default=24,
        ge=1,
        le=240,
    )

    rainfall_mm: float | None = Field(
        default=None,
        ge=0,
    )

    rainfall_period_hours: float | None = Field(
        default=None,
        gt=0,
    )

    river_level_m: float | None = None

    soil_moisture_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    elevation_m: float | None = None

    additional_features: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @model_validator(mode="after")
    def validate_rainfall(self):
        if (
            self.rainfall_mm is not None
            and self.rainfall_period_hours is None
        ):
            raise ValueError(
                "Rainfall period is required with rainfall_mm"
            )

        return self


# ============================================================
# FLOOD PREDICTION RESULT
# ============================================================

class FloodPredictionResponse(FloodBase):
    prediction_id: UUID

    district: str

    status: PredictionStatus

    risk_level: FloodRiskLevel = (
        FloodRiskLevel.UNKNOWN
    )

    flood_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    predicted_water_depth_m: float | None = Field(
        default=None,
        ge=0,
    )

    affected_area_km2: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_affected_population: int | None = Field(
        default=None,
        ge=0,
    )

    model_name: str | None = None
    model_version: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    input_sources: list[FloodDataSource] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    human_review_required: bool = True

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# FLOOD SEVERITY ASSESSMENT
# ============================================================

class FloodSeverityAssessment(FloodBase):
    assessment_id: UUID = Field(
        default_factory=uuid4
    )

    district: str

    severity: FloodSeverity = FloodSeverity.UNKNOWN

    estimated_water_depth_m: float | None = Field(
        default=None,
        ge=0,
    )

    affected_households: int | None = Field(
        default=None,
        ge=0,
    )

    affected_population: int | None = Field(
        default=None,
        ge=0,
    )

    road_disruptions: int | None = Field(
        default=None,
        ge=0,
    )

    evidence: list[FloodDataSource] = Field(
        default_factory=list
    )

    assessment_notes: str | None = None

    verified_by: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# FLOOD ALERT CREATION
# ============================================================

class FloodAlertCreate(FloodBase):
    district: str

    alert_type: FloodAlertType

    risk_level: FloodRiskLevel

    title: str = Field(
        min_length=5,
        max_length=200,
    )

    message: str = Field(
        min_length=10,
        max_length=5000,
    )

    instructions: list[str] = Field(
        default_factory=list
    )

    affected_locations: list[str] = Field(
        default_factory=list
    )

    evidence: list[FloodDataSource] = Field(
        default_factory=list
    )

    valid_from: datetime
    valid_until: datetime

    prediction_id: UUID | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @model_validator(mode="after")
    def validate_alert_period(self):
        validate_timezone(self.valid_from)
        validate_timezone(self.valid_until)

        if self.valid_until <= self.valid_from:
            raise ValueError(
                "Alert expiry must follow start time"
            )

        return self


# ============================================================
# FLOOD ALERT RESPONSE
# ============================================================

class FloodAlertResponse(FloodAlertCreate):
    alert_id: UUID = Field(
        default_factory=uuid4
    )

    status: FloodAlertStatus = (
        FloodAlertStatus.PENDING_REVIEW
    )

    created_by: str | None = None

    approved_by: str | None = None
    approved_at: datetime | None = None

    dispatched_at: datetime | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    official_warning: bool = False


# ============================================================
# ALERT APPROVAL
# ============================================================

class FloodAlertApprovalRequest(FloodBase):
    alert_id: UUID

    approve: bool

    reason: str = Field(
        min_length=5,
        max_length=2000,
    )


class FloodAlertApprovalResponse(FloodBase):
    alert_id: UUID

    status: FloodAlertStatus

    reviewed_by: str

    reviewed_at: datetime = Field(
        default_factory=utc_now
    )

    authorization_verified: bool = False


# ============================================================
# SATELLITE FLOOD OBSERVATION
# ============================================================

class SatelliteFloodObservation(FloodBase):
    observation_id: UUID = Field(
        default_factory=uuid4
    )

    district: str

    satellite_name: str
    image_reference: str

    captured_at: datetime

    flood_extent_km2: float | None = Field(
        default=None,
        ge=0,
    )

    water_coverage_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    model_name: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    verified: bool = False

    source: FloodDataSource

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_timezone(value)


# ============================================================
# FLOOD MAP
# ============================================================

class FloodMapFeature(FloodBase):
    feature_id: UUID = Field(
        default_factory=uuid4
    )

    district: str

    risk_level: FloodRiskLevel = (
        FloodRiskLevel.UNKNOWN
    )

    geometry: dict[str, Any]

    properties: dict[str, Any] = Field(
        default_factory=dict
    )

    source: FloodDataSource | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @field_validator("geometry")
    @classmethod
    def validate_geometry(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        if value.get("type") not in {
            "Point",
            "MultiPoint",
            "LineString",
            "MultiLineString",
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError(
                "Unsupported GeoJSON geometry type"
            )

        if "coordinates" not in value:
            raise ValueError(
                "GeoJSON geometry requires coordinates"
            )

        return value


class FloodMapResponse(FloodBase):
    map_id: UUID = Field(
        default_factory=uuid4
    )

    features: list[FloodMapFeature] = Field(
        default_factory=list
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    verified: bool = False


# ============================================================
# DISTRICT FLOOD SUMMARY
# ============================================================

class DistrictFloodSummary(FloodBase):
    district: str

    risk_level: FloodRiskLevel = (
        FloodRiskLevel.UNKNOWN
    )

    rainfall_24h_mm: float | None = Field(
        default=None,
        ge=0,
    )

    river_level_m: float | None = None

    active_alert_count: int | None = Field(
        default=None,
        ge=0,
    )

    affected_population: int | None = Field(
        default=None,
        ge=0,
    )

    active_sos_count: int | None = Field(
        default=None,
        ge=0,
    )

    observation_quality: ObservationQuality = (
        ObservationQuality.MISSING
    )

    last_updated: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# FLOOD DASHBOARD
# ============================================================

class FloodDashboardResponse(FloodBase):
    districts: list[DistrictFloodSummary] = Field(
        default_factory=list
    )

    active_alerts: list[FloodAlertResponse] = Field(
        default_factory=list
    )

    latest_predictions: list[
        FloodPredictionResponse
    ] = Field(default_factory=list)

    rainfall_observations: list[
        RainfallObservation
    ] = Field(default_factory=list)

    river_observations: list[
        RiverLevelObservation
    ] = Field(default_factory=list)

    dam_observations: list[
        DamObservation
    ] = Field(default_factory=list)

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    data_complete: bool = False


# ============================================================
# API RESPONSE
# ============================================================

class FloodAPIResponse(FloodBase):
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
    "FloodRiskLevel",
    "FloodSeverity",
    "FloodAlertStatus",
    "FloodAlertType",
    "DataSourceType",
    "ObservationQuality",
    "TrendDirection",
    "PredictionStatus",
    "DamGateStatus",
    "GeoPoint",
    "FloodLocation",
    "GeoBoundingBox",
    "FloodDataSource",
    "RainfallObservation",
    "RainfallForecast",
    "RiverLevelObservation",
    "RiverLevelForecast",
    "DamObservation",
    "FloodPredictionRequest",
    "FloodPredictionResponse",
    "FloodSeverityAssessment",
    "FloodAlertCreate",
    "FloodAlertResponse",
    "FloodAlertApprovalRequest",
    "FloodAlertApprovalResponse",
    "SatelliteFloodObservation",
    "FloodMapFeature",
    "FloodMapResponse",
    "DistrictFloodSummary",
    "FloodDashboardResponse",
    "FloodAPIResponse",
    "normalize_district",
    "validate_timezone",
]
