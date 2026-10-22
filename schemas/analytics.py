from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/analytics.py

Pydantic schemas for disaster management analytics.

Modules:
- Executive dashboard
- Flood risk analytics
- Rainfall and river monitoring
- SOS and rescue analytics
- Evacuation analytics
- Shelter occupancy
- Relief and inventory analytics
- Medical response analytics
- Volunteer analytics
- Damage and recovery analytics
- Multi-agent performance
- District comparisons
- Time-series charts
- Geographic heatmaps
- KPI monitoring
- Report exports

All metrics must originate from verified data sources.
Missing observations must not be represented as zero.
"""


from datetime import date, datetime, timezone
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


# ============================================================
# BASE SCHEMA
# ============================================================

class AnalyticsBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


# ============================================================
# ENUMS
# ============================================================

class AnalyticsPeriod(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class AnalyticsCategory(str, Enum):
    OVERVIEW = "overview"
    FLOOD = "flood"
    RAINFALL = "rainfall"
    RIVER = "river"
    DAM = "dam"
    SOS = "sos"
    RESCUE = "rescue"
    EVACUATION = "evacuation"
    SHELTER = "shelter"
    RELIEF = "relief"
    MEDICAL = "medical"
    VOLUNTEER = "volunteer"
    DAMAGE = "damage"
    RECOVERY = "recovery"
    AGENTS = "agents"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    PIE = "pie"
    DONUT = "donut"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    TABLE = "table"


class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    XLSX = "xlsx"


# ============================================================
# KERALA DISTRICTS
# ============================================================

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


# ============================================================
# ANALYTICS FILTERS
# ============================================================

class AnalyticsFilter(AnalyticsBase):
    district: str | None = None

    start_date: datetime
    end_date: datetime

    period: AnalyticsPeriod = AnalyticsPeriod.DAILY

    category: AnalyticsCategory = (
        AnalyticsCategory.OVERVIEW
    )

    include_predictions: bool = False

    @field_validator("district")
    @classmethod
    def validate_district(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        return normalize_district(value)

    @model_validator(mode="after")
    def validate_dates(self):
        if (
            self.start_date.tzinfo is None
            or self.end_date.tzinfo is None
        ):
            raise ValueError(
                "Analytics dates must include timezone"
            )

        if self.start_date > self.end_date:
            raise ValueError(
                "start_date must not exceed end_date"
            )

        return self


# ============================================================
# METRIC / KPI
# ============================================================

class KPIMetric(AnalyticsBase):
    key: str
    label: str

    value: float | int | None = None
    unit: str | None = None

    previous_value: float | int | None = None

    change_percent: float | None = None

    trend: TrendDirection = TrendDirection.UNKNOWN

    description: str | None = None

    source: str | None = None
    verified: bool = False

    updated_at: datetime = Field(
        default_factory=utc_now
    )


class KPIGroup(AnalyticsBase):
    category: AnalyticsCategory

    metrics: list[KPIMetric] = Field(
        default_factory=list
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# FLOOD ANALYTICS
# ============================================================

class FloodRiskAnalytics(AnalyticsBase):
    district: str

    risk_level: RiskLevel = RiskLevel.UNKNOWN

    risk_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    affected_area_km2: float | None = Field(
        default=None,
        ge=0,
    )

    affected_population: int | None = Field(
        default=None,
        ge=0,
    )

    active_alerts: int | None = Field(
        default=None,
        ge=0,
    )

    forecast_horizon_hours: int | None = Field(
        default=None,
        ge=1,
    )

    model_name: str | None = None
    model_version: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# RAINFALL ANALYTICS
# ============================================================

class RainfallAnalytics(AnalyticsBase):
    district: str

    rainfall_1h_mm: float | None = Field(
        default=None,
        ge=0,
    )

    rainfall_24h_mm: float | None = Field(
        default=None,
        ge=0,
    )

    rainfall_7d_mm: float | None = Field(
        default=None,
        ge=0,
    )

    forecast_rainfall_mm: float | None = Field(
        default=None,
        ge=0,
    )

    forecast_horizon_hours: int | None = Field(
        default=None,
        ge=1,
    )

    observation_count: int | None = Field(
        default=None,
        ge=0,
    )

    trend: TrendDirection = TrendDirection.UNKNOWN

    source: str | None = None

    observed_at: datetime | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# RIVER ANALYTICS
# ============================================================

class RiverAnalytics(AnalyticsBase):
    river_name: str
    station_id: str

    district: str

    current_level_m: float | None = None
    warning_level_m: float | None = None
    danger_level_m: float | None = None

    discharge_m3s: float | None = Field(
        default=None,
        ge=0,
    )

    predicted_level_m: float | None = None

    trend: TrendDirection = TrendDirection.UNKNOWN

    observed_at: datetime | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# DAM ANALYTICS
# ============================================================

class DamAnalytics(AnalyticsBase):
    dam_id: str
    dam_name: str

    district: str

    current_storage_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    current_water_level_m: float | None = None

    inflow_m3s: float | None = Field(
        default=None,
        ge=0,
    )

    outflow_m3s: float | None = Field(
        default=None,
        ge=0,
    )

    gate_status: str | None = None

    source: str | None = None
    observed_at: datetime | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# SOS ANALYTICS
# ============================================================

class SOSAnalytics(AnalyticsBase):
    district: str | None = None

    total_requests: int | None = Field(
        default=None,
        ge=0,
    )

    pending_requests: int | None = Field(
        default=None,
        ge=0,
    )

    assigned_requests: int | None = Field(
        default=None,
        ge=0,
    )

    in_progress_requests: int | None = Field(
        default=None,
        ge=0,
    )

    resolved_requests: int | None = Field(
        default=None,
        ge=0,
    )

    critical_requests: int | None = Field(
        default=None,
        ge=0,
    )

    average_response_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    people_requesting_assistance: int | None = Field(
        default=None,
        ge=0,
    )

    @field_validator("district")
    @classmethod
    def validate_district(
        cls,
        value: str | None,
    ) -> str | None:
        return (
            normalize_district(value)
            if value is not None
            else None
        )


# ============================================================
# RESCUE ANALYTICS
# ============================================================

class RescueAnalytics(AnalyticsBase):
    district: str | None = None

    total_rescue_teams: int | None = Field(
        default=None,
        ge=0,
    )

    available_teams: int | None = Field(
        default=None,
        ge=0,
    )

    deployed_teams: int | None = Field(
        default=None,
        ge=0,
    )

    active_missions: int | None = Field(
        default=None,
        ge=0,
    )

    completed_missions: int | None = Field(
        default=None,
        ge=0,
    )

    confirmed_people_rescued: int | None = Field(
        default=None,
        ge=0,
    )

    average_dispatch_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    mission_completion_rate: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )


# ============================================================
# EVACUATION ANALYTICS
# ============================================================

class EvacuationAnalytics(AnalyticsBase):
    district: str | None = None

    evacuation_zones: int | None = Field(
        default=None,
        ge=0,
    )

    active_evacuation_operations: int | None = Field(
        default=None,
        ge=0,
    )

    confirmed_evacuated_people: int | None = Field(
        default=None,
        ge=0,
    )

    pending_assistance_requests: int | None = Field(
        default=None,
        ge=0,
    )

    available_transport_vehicles: int | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# SHELTER ANALYTICS
# ============================================================

class ShelterAnalytics(AnalyticsBase):
    district: str | None = None

    total_shelters: int | None = Field(
        default=None,
        ge=0,
    )

    active_shelters: int | None = Field(
        default=None,
        ge=0,
    )

    total_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    current_occupancy: int | None = Field(
        default=None,
        ge=0,
    )

    available_capacity: int | None = Field(
        default=None,
        ge=0,
    )

    occupancy_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    overcrowded_shelters: int | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# RELIEF ANALYTICS
# ============================================================

class ReliefAnalytics(AnalyticsBase):
    district: str | None = None

    inventory_items: int | None = Field(
        default=None,
        ge=0,
    )

    low_stock_items: int | None = Field(
        default=None,
        ge=0,
    )

    pending_deliveries: int | None = Field(
        default=None,
        ge=0,
    )

    completed_deliveries: int | None = Field(
        default=None,
        ge=0,
    )

    confirmed_beneficiaries: int | None = Field(
        default=None,
        ge=0,
    )

    delivery_completion_rate: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    estimated_supply_shortfall: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# MEDICAL ANALYTICS
# ============================================================

class MedicalAnalytics(AnalyticsBase):
    district: str | None = None

    active_medical_requests: int | None = Field(
        default=None,
        ge=0,
    )

    available_ambulances: int | None = Field(
        default=None,
        ge=0,
    )

    available_hospital_beds: int | None = Field(
        default=None,
        ge=0,
    )

    occupied_hospital_beds: int | None = Field(
        default=None,
        ge=0,
    )

    medical_teams_deployed: int | None = Field(
        default=None,
        ge=0,
    )

    confirmed_patients_assisted: int | None = Field(
        default=None,
        ge=0,
    )

    average_medical_response_minutes: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# VOLUNTEER ANALYTICS
# ============================================================

class VolunteerAnalytics(AnalyticsBase):
    district: str | None = None

    registered_volunteers: int | None = Field(
        default=None,
        ge=0,
    )

    available_volunteers: int | None = Field(
        default=None,
        ge=0,
    )

    assigned_volunteers: int | None = Field(
        default=None,
        ge=0,
    )

    completed_assignments: int | None = Field(
        default=None,
        ge=0,
    )

    total_verified_service_hours: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# DAMAGE ANALYTICS
# ============================================================

class DamageAnalytics(AnalyticsBase):
    district: str | None = None

    damage_reports: int | None = Field(
        default=None,
        ge=0,
    )

    verified_damage_reports: int | None = Field(
        default=None,
        ge=0,
    )

    damaged_buildings: int | None = Field(
        default=None,
        ge=0,
    )

    affected_roads: int | None = Field(
        default=None,
        ge=0,
    )

    damaged_bridges: int | None = Field(
        default=None,
        ge=0,
    )

    estimated_damage_inr: float | None = Field(
        default=None,
        ge=0,
    )

    estimation_method: str | None = None


# ============================================================
# RECOVERY ANALYTICS
# ============================================================

class RecoveryAnalytics(AnalyticsBase):
    district: str | None = None

    total_recovery_projects: int | None = Field(
        default=None,
        ge=0,
    )

    ongoing_projects: int | None = Field(
        default=None,
        ge=0,
    )

    completed_projects: int | None = Field(
        default=None,
        ge=0,
    )

    estimated_recovery_cost_inr: float | None = Field(
        default=None,
        ge=0,
    )

    verified_expenditure_inr: float | None = Field(
        default=None,
        ge=0,
    )

    completion_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )


# ============================================================
# AGENT ANALYTICS
# ============================================================

class AgentAnalytics(AnalyticsBase):
    agent_name: str
    agent_type: str

    total_tasks: int | None = Field(
        default=None,
        ge=0,
    )

    completed_tasks: int | None = Field(
        default=None,
        ge=0,
    )

    failed_tasks: int | None = Field(
        default=None,
        ge=0,
    )

    pending_tasks: int | None = Field(
        default=None,
        ge=0,
    )

    average_execution_ms: float | None = Field(
        default=None,
        ge=0,
    )

    tool_success_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    groundedness_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    human_approval_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


# ============================================================
# TIME SERIES
# ============================================================

class TimeSeriesPoint(AnalyticsBase):
    timestamp: datetime
    value: float | None = None

    verified: bool = False


class TimeSeriesAnalytics(AnalyticsBase):
    metric_name: str
    unit: str | None = None

    district: str | None = None

    period: AnalyticsPeriod

    points: list[TimeSeriesPoint] = Field(
        default_factory=list
    )

    source: str | None = None


# ============================================================
# CHART SCHEMAS
# ============================================================

class ChartDataset(AnalyticsBase):
    label: str

    data: list[float | int | None] = Field(
        default_factory=list
    )

    color: str | None = None


class ChartResponse(AnalyticsBase):
    chart_id: UUID = Field(default_factory=uuid4)

    title: str
    chart_type: ChartType

    labels: list[str] = Field(
        default_factory=list
    )

    datasets: list[ChartDataset] = Field(
        default_factory=list
    )

    unit: str | None = None

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_chart_lengths(self):
        for dataset in self.datasets:
            if len(dataset.data) != len(self.labels):
                raise ValueError(
                    "Dataset length must match chart labels"
                )
        return self


# ============================================================
# GEOGRAPHIC ANALYTICS
# ============================================================

class GeoAnalyticsPoint(AnalyticsBase):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    district: str | None = None

    metric_name: str

    value: float | None = None

    risk_level: RiskLevel = RiskLevel.UNKNOWN

    verified: bool = False


class GeoHeatmapResponse(AnalyticsBase):
    title: str

    points: list[GeoAnalyticsPoint] = Field(
        default_factory=list
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# DISTRICT COMPARISON
# ============================================================

class DistrictComparison(AnalyticsBase):
    district: str

    flood_risk: RiskLevel = RiskLevel.UNKNOWN

    rainfall_24h_mm: float | None = None

    active_sos: int | None = Field(
        default=None,
        ge=0,
    )

    active_rescue_missions: int | None = Field(
        default=None,
        ge=0,
    )

    shelter_occupancy_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    available_rescue_teams: int | None = Field(
        default=None,
        ge=0,
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class DistrictComparisonResponse(AnalyticsBase):
    districts: list[DistrictComparison] = Field(
        default_factory=list
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

class ExecutiveDashboard(AnalyticsBase):
    dashboard_id: UUID = Field(
        default_factory=uuid4
    )

    district: str | None = None

    flood: FloodRiskAnalytics | None = None
    rainfall: RainfallAnalytics | None = None

    sos: SOSAnalytics | None = None
    rescue: RescueAnalytics | None = None
    evacuation: EvacuationAnalytics | None = None
    shelter: ShelterAnalytics | None = None
    relief: ReliefAnalytics | None = None
    medical: MedicalAnalytics | None = None
    volunteer: VolunteerAnalytics | None = None
    damage: DamageAnalytics | None = None
    recovery: RecoveryAnalytics | None = None

    kpis: list[KPIMetric] = Field(
        default_factory=list
    )

    charts: list[ChartResponse] = Field(
        default_factory=list
    )

    data_sources: list[str] = Field(
        default_factory=list
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# ANALYTICS EXPORT
# ============================================================

class AnalyticsExportRequest(AnalyticsBase):
    export_id: UUID = Field(
        default_factory=uuid4
    )

    category: AnalyticsCategory
    export_format: ExportFormat

    filters: AnalyticsFilter

    include_charts: bool = True
    include_summary: bool = True

    requested_at: datetime = Field(
        default_factory=utc_now
    )


class AnalyticsExportResponse(AnalyticsBase):
    export_id: UUID

    status: str

    file_name: str | None = None
    download_url: str | None = None

    generated_at: datetime | None = None

    error_message: str | None = None


# ============================================================
# GENERIC ANALYTICS API RESPONSE
# ============================================================

class AnalyticsAPIResponse(AnalyticsBase):
    success: bool

    category: AnalyticsCategory

    data: dict[str, Any] = Field(
        default_factory=dict
    )

    source: str | None = None

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "AnalyticsPeriod",
    "AnalyticsCategory",
    "RiskLevel",
    "TrendDirection",
    "ChartType",
    "ExportFormat",
    "AnalyticsFilter",
    "KPIMetric",
    "KPIGroup",
    "FloodRiskAnalytics",
    "RainfallAnalytics",
    "RiverAnalytics",
    "DamAnalytics",
    "SOSAnalytics",
    "RescueAnalytics",
    "EvacuationAnalytics",
    "ShelterAnalytics",
    "ReliefAnalytics",
    "MedicalAnalytics",
    "VolunteerAnalytics",
    "DamageAnalytics",
    "RecoveryAnalytics",
    "AgentAnalytics",
    "TimeSeriesPoint",
    "TimeSeriesAnalytics",
    "ChartDataset",
    "ChartResponse",
    "GeoAnalyticsPoint",
    "GeoHeatmapResponse",
    "DistrictComparison",
    "DistrictComparisonResponse",
    "ExecutiveDashboard",
    "AnalyticsExportRequest",
    "AnalyticsExportResponse",
    "AnalyticsAPIResponse",
]
