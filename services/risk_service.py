from __future__ import annotations

"""
backend/services/risk_service.py

Kerala Flood Relief Coordination Agent
=======================================

Flood Risk Assessment and Intelligence Service.

Features:
- District and location-based flood risk assessment
- Rainfall intensity and cumulative rainfall analysis
- River water-level and rising-trend assessment
- Dam discharge and downstream exposure analysis
- Forecast risk integration
- Terrain and drainage vulnerability assessment
- Population and infrastructure exposure
- Historical flood vulnerability
- Multi-factor risk scoring
- Confidence and data-quality evaluation
- Missing/stale sensor detection
- Risk escalation and trend tracking
- Human verification and approval
- Risk assessment history and reporting
- Audit logging and role-based permissions

IMPORTANT:
- The scoring model is illustrative, not calibrated
  against verified Kerala flood-disaster datasets.
- Scores must not be used as official flood warnings.
- Missing data never implies that conditions are safe.
- High-risk signals require qualified human review.
- This service does not automatically order evacuation,
  publish public alerts or dispatch rescue teams.
- Production requires calibrated hydrological models,
  verified sources, sensor QA and official procedures.
"""


import logging
import math
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class FloodRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class RiskAssessmentStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class RiskTrend(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


class RiskSourceType(str, Enum):
    RAINFALL = "rainfall"
    RIVER = "river"
    DAM = "dam"
    FORECAST = "forecast"
    TERRAIN = "terrain"
    EXPOSURE = "exposure"
    HISTORICAL = "historical"


class RiskConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class RiskRecommendationType(str, Enum):
    MONITOR = "monitor"
    VERIFY_DATA = "verify_data"
    REVIEW_ALERT = "review_alert"
    REVIEW_EVACUATION = "review_evacuation"
    REVIEW_RESCUE_READINESS = "review_rescue_readiness"
    REVIEW_SHELTER_CAPACITY = "review_shelter_capacity"
    INSPECT_INFRASTRUCTURE = "inspect_infrastructure"


# ============================================================
# EXCEPTIONS
# ============================================================

class RiskServiceError(Exception):
    pass


class RiskValidationError(RiskServiceError):
    pass


class RiskNotFoundError(RiskServiceError):
    pass


class RiskConflictError(RiskServiceError):
    pass


class RiskAuthorizationError(RiskServiceError):
    pass


class RiskDataError(RiskServiceError):
    pass


class RiskVerificationError(RiskServiceError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_text(
    value: str,
    name: str,
    *,
    max_length: int = 2000,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RiskValidationError(
            f"{name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise RiskValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if not isinstance(value, datetime):
        raise RiskValidationError(
            f"{name} must be a datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise RiskValidationError(
            f"{name} must be timezone-aware"
        )


def validate_number(
    value: float | int,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RiskValidationError(
            f"{name} must be a finite number"
        )

    result = float(value)

    if result < minimum:
        raise RiskValidationError(
            f"{name} cannot be below {minimum}"
        )

    if maximum is not None and result > maximum:
        raise RiskValidationError(
            f"{name} cannot exceed {maximum}"
        )

    return result


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


def normalized_score(
    value: float,
    reference: float,
) -> float:
    if reference <= 0:
        raise RiskValidationError(
            "Normalization reference must be positive"
        )

    return clamp(value / reference)


# ============================================================
# LOCATION
# ============================================================

@dataclass(frozen=True)
class RiskLocation:
    district_id: int
    latitude: float
    longitude: float

    location_name: str = ""
    panchayat: str = ""
    taluk: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.district_id, bool)
            or self.district_id <= 0
        ):
            raise RiskValidationError(
                "Invalid district ID"
            )

        validate_number(
            self.latitude,
            "latitude",
            minimum=-90,
            maximum=90,
        )

        validate_number(
            self.longitude,
            "longitude",
            minimum=-180,
            maximum=180,
        )


# ============================================================
# SENSOR OBSERVATIONS
# ============================================================

@dataclass(frozen=True)
class RiskObservation:
    source_id: str
    source_type: RiskSourceType

    observed_at: datetime

    verified: bool = False
    quality_score: float = 1.0

    def __post_init__(self) -> None:
        require_text(
            self.source_id,
            "source_id",
        )

        validate_datetime(
            self.observed_at,
            "observed_at",
        )

        validate_number(
            self.quality_score,
            "quality_score",
            maximum=1.0,
        )


@dataclass(frozen=True)
class RainfallObservation(RiskObservation):
    rainfall_1h_mm: float = 0.0
    rainfall_24h_mm: float = 0.0
    rainfall_72h_mm: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        for name in (
            "rainfall_1h_mm",
            "rainfall_24h_mm",
            "rainfall_72h_mm",
        ):
            validate_number(
                getattr(self, name),
                name,
            )


@dataclass(frozen=True)
class RiverObservation(RiskObservation):
    water_level_m: float = 0.0

    warning_level_m: float | None = None
    danger_level_m: float | None = None

    rise_rate_m_per_hour: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        validate_number(
            self.water_level_m,
            "water_level_m",
        )

        for name in (
            "warning_level_m",
            "danger_level_m",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_number(
                    value,
                    name,
                )

        if (
            self.warning_level_m is not None
            and self.danger_level_m is not None
            and self.warning_level_m
            >= self.danger_level_m
        ):
            raise RiskValidationError(
                "Warning level must be below danger level"
            )

        if not math.isfinite(
            self.rise_rate_m_per_hour
        ):
            raise RiskValidationError(
                "Invalid river rise rate"
            )


@dataclass(frozen=True)
class DamObservation(RiskObservation):
    storage_percent: float = 0.0
    discharge_cumecs: float = 0.0

    downstream_exposure: float = 0.0
    discharge_change_cumecs_per_hour: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        validate_number(
            self.storage_percent,
            "storage_percent",
            maximum=100,
        )

        validate_number(
            self.discharge_cumecs,
            "discharge_cumecs",
        )

        validate_number(
            self.downstream_exposure,
            "downstream_exposure",
            maximum=1,
        )

        if not math.isfinite(
            self.discharge_change_cumecs_per_hour
        ):
            raise RiskValidationError(
                "Invalid discharge change rate"
            )


@dataclass(frozen=True)
class ForecastObservation(RiskObservation):
    predicted_rainfall_24h_mm: float = 0.0
    predicted_river_rise_m: float = 0.0

    model_confidence: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        validate_number(
            self.predicted_rainfall_24h_mm,
            "predicted_rainfall_24h_mm",
        )

        if not math.isfinite(
            self.predicted_river_rise_m
        ):
            raise RiskValidationError(
                "Invalid predicted river rise"
            )

        validate_number(
            self.model_confidence,
            "model_confidence",
            maximum=1.0,
        )


# ============================================================
# TERRAIN AND EXPOSURE
# ============================================================

@dataclass(frozen=True)
class TerrainRiskProfile:
    low_lying_area_score: float = 0.0
    drainage_vulnerability_score: float = 0.0
    landslide_susceptibility_score: float = 0.0
    historical_flood_score: float = 0.0

    verified: bool = False

    def __post_init__(self) -> None:
        for name in (
            "low_lying_area_score",
            "drainage_vulnerability_score",
            "landslide_susceptibility_score",
            "historical_flood_score",
        ):
            validate_number(
                getattr(self, name),
                name,
                maximum=1.0,
            )


@dataclass(frozen=True)
class PopulationExposure:
    population_count: int = 0

    vulnerable_population_count: int = 0

    hospitals_exposed: int = 0
    shelters_exposed: int = 0
    schools_exposed: int = 0

    critical_roads_exposed: int = 0
    power_facilities_exposed: int = 0

    verified: bool = False

    def __post_init__(self) -> None:
        for name in (
            "population_count",
            "vulnerable_population_count",
            "hospitals_exposed",
            "shelters_exposed",
            "schools_exposed",
            "critical_roads_exposed",
            "power_facilities_exposed",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise RiskValidationError(
                    f"{name} must be a non-negative integer"
                )

        if (
            self.vulnerable_population_count
            > self.population_count
        ):
            raise RiskValidationError(
                "Vulnerable population exceeds population"
            )


# ============================================================
# REQUEST AND RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class RiskAssessmentRequest:
    location: RiskLocation

    rainfall: RainfallObservation | None = None
    river: RiverObservation | None = None
    dam: DamObservation | None = None
    forecast: ForecastObservation | None = None

    terrain: TerrainRiskProfile | None = None
    exposure: PopulationExposure | None = None

    incident_id: str | None = None

    description: str = ""

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RiskFactor:
    name: str
    score: float
    weight: float

    contribution: float

    verified: bool

    explanation: str = ""


@dataclass(frozen=True)
class RiskDataQuality:
    confidence: RiskConfidence

    completeness_percent: float

    verified_sources: int
    total_sources: int

    stale_sources: tuple[str, ...] = ()
    missing_sources: tuple[str, ...] = ()
    unverified_sources: tuple[str, ...] = ()

    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RiskRecommendation:
    recommendation_type: RiskRecommendationType

    priority: FloodRiskLevel

    description: str

    requires_human_approval: bool = True


@dataclass(frozen=True)
class RiskAssessmentResult:
    score: float

    risk_level: FloodRiskLevel
    trend: RiskTrend

    factors: tuple[RiskFactor, ...]

    quality: RiskDataQuality

    recommendations: tuple[
        RiskRecommendation,
        ...
    ]

    reasoning: tuple[str, ...]

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass
class FloodRiskAssessment:
    id: str

    district_id: int
    location: RiskLocation

    status: RiskAssessmentStatus

    score: float
    risk_level: FloodRiskLevel
    trend: RiskTrend

    factors: list[RiskFactor]

    quality: RiskDataQuality

    recommendations: list[RiskRecommendation]

    reasoning: list[str]

    created_by: int

    incident_id: str | None = None

    description: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_by: int | None = None
    reviewed_at: datetime | None = None

    rejection_reason: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class RiskRepository(Protocol):

    async def create(
        self,
        assessment: FloodRiskAssessment,
    ) -> FloodRiskAssessment:
        ...

    async def get_by_id(
        self,
        assessment_id: str,
    ) -> FloodRiskAssessment | None:
        ...

    async def update(
        self,
        assessment: FloodRiskAssessment,
        *,
        expected_version: int,
    ) -> FloodRiskAssessment:
        """
        Must implement atomic optimistic concurrency.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[FloodRiskAssessment]:
        ...

    async def list_by_location(
        self,
        location: RiskLocation,
    ) -> Sequence[FloodRiskAssessment]:
        ...


class RiskAuthorizationGateway(Protocol):

    async def can_assess(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_review(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_view(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class RiskAuditGateway(Protocol):

    async def record(
        self,
        action: str,
        *,
        actor_id: int | None,
        entity_type: str,
        entity_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...


# ============================================================
# RISK CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RiskModelConfig:
    """
    Illustrative reference values only.

    These are not official Kerala flood warning
    thresholds and must be calibrated locally.
    """

    rainfall_1h_reference_mm: float = 60.0
    rainfall_24h_reference_mm: float = 150.0
    rainfall_72h_reference_mm: float = 300.0

    river_rise_reference_mph: float = 0.5

    dam_storage_reference_percent: float = 90.0
    dam_discharge_reference_cumecs: float = 1000.0

    forecast_rainfall_reference_mm: float = 150.0

    source_max_age: timedelta = timedelta(
        hours=3
    )

    forecast_max_age: timedelta = timedelta(
        hours=6
    )

    minimum_completeness: float = 50.0

    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "rainfall": 0.25,
            "river": 0.25,
            "dam": 0.10,
            "forecast": 0.15,
            "terrain": 0.15,
            "exposure": 0.10,
        }
    )

    def __post_init__(self) -> None:
        total = sum(
            self.weights.values()
        )

        if not math.isclose(
            total,
            1.0,
            abs_tol=1e-8,
        ):
            raise RiskValidationError(
                "Risk factor weights must total 1.0"
            )

        if any(
            not math.isfinite(weight) or weight < 0
            for weight in self.weights.values()
        ):
            raise RiskValidationError(
                "Risk weights must be non-negative"
            )

        if self.source_max_age <= timedelta(0):
            raise RiskValidationError(
                "source_max_age must be positive"
            )

        if self.forecast_max_age <= timedelta(0):
            raise RiskValidationError(
                "forecast_max_age must be positive"
            )

        validate_number(
            self.minimum_completeness,
            "minimum_completeness",
            maximum=100,
        )


# ============================================================
# RISK SCORING ENGINE
# ============================================================

class FloodRiskScoringEngine:

    def __init__(
        self,
        config: RiskModelConfig | None = None,
    ):
        self.config = (
            config
            or RiskModelConfig()
        )

    def rainfall_score(
        self,
        data: RainfallObservation,
    ) -> float:

        return clamp(
            0.35 * normalized_score(
                data.rainfall_1h_mm,
                self.config.rainfall_1h_reference_mm,
            )
            + 0.40 * normalized_score(
                data.rainfall_24h_mm,
                self.config.rainfall_24h_reference_mm,
            )
            + 0.25 * normalized_score(
                data.rainfall_72h_mm,
                self.config.rainfall_72h_reference_mm,
            )
        )

    def river_score(
        self,
        data: RiverObservation,
    ) -> float:

        if (
            data.danger_level_m is not None
            and data.water_level_m >= data.danger_level_m
        ):
            level_score = 1.0

        elif (
            data.warning_level_m is not None
            and data.water_level_m
            >= data.warning_level_m
        ):
            level_score = 0.75

        elif (
            data.warning_level_m is not None
            and data.warning_level_m > 0
        ):
            level_score = normalized_score(
                data.water_level_m,
                data.warning_level_m,
            ) * 0.7

        else:
            # Absolute river levels cannot be compared
            # safely without station-specific thresholds.
            level_score = 0.0

        rise_score = normalized_score(
            max(0.0, data.rise_rate_m_per_hour),
            self.config.river_rise_reference_mph,
        )

        return clamp(
            0.75 * level_score
            + 0.25 * rise_score
        )

    def dam_score(
        self,
        data: DamObservation,
    ) -> float:

        storage_score = normalized_score(
            data.storage_percent,
            self.config.dam_storage_reference_percent,
        )

        discharge_score = normalized_score(
            data.discharge_cumecs,
            self.config.dam_discharge_reference_cumecs,
        )

        return clamp(
            (
                0.35 * storage_score
                + 0.65 * discharge_score
            )
            * data.downstream_exposure
        )

    def forecast_score(
        self,
        data: ForecastObservation,
    ) -> float:

        rainfall_score = normalized_score(
            data.predicted_rainfall_24h_mm,
            self.config.forecast_rainfall_reference_mm,
        )

        river_score = normalized_score(
            max(0.0, data.predicted_river_rise_m),
            1.0,
        )

        return clamp(
            0.65 * rainfall_score
            + 0.35 * river_score
        )

    @staticmethod
    def terrain_score(
        data: TerrainRiskProfile,
    ) -> float:

        return clamp(
            0.35 * data.low_lying_area_score
            + 0.30 * data.drainage_vulnerability_score
            + 0.20 * data.landslide_susceptibility_score
            + 0.15 * data.historical_flood_score
        )

    @staticmethod
    def exposure_score(
        data: PopulationExposure,
    ) -> float:

        population_score = normalized_score(
            data.population_count,
            10000,
        )

        vulnerable_score = (
            data.vulnerable_population_count
            / data.population_count
            if data.population_count > 0
            else 0.0
        )

        infrastructure_score = normalized_score(
            (
                data.hospitals_exposed * 3
                + data.shelters_exposed * 2
                + data.schools_exposed
                + data.critical_roads_exposed * 2
                + data.power_facilities_exposed * 3
            ),
            30,
        )

        return clamp(
            0.35 * population_score
            + 0.35 * vulnerable_score
            + 0.30 * infrastructure_score
        )

    @staticmethod
    def classify(
        score: float,
    ) -> FloodRiskLevel:

        if score >= 0.85:
            return FloodRiskLevel.EXTREME

        if score >= 0.70:
            return FloodRiskLevel.SEVERE

        if score >= 0.50:
            return FloodRiskLevel.HIGH

        if score >= 0.30:
            return FloodRiskLevel.MODERATE

        return FloodRiskLevel.LOW


# ============================================================
# DATA QUALITY EVALUATOR
# ============================================================

class RiskDataQualityEvaluator:

    SOURCE_NAMES = (
        "rainfall",
        "river",
        "dam",
        "forecast",
        "terrain",
        "exposure",
    )

    def __init__(
        self,
        config: RiskModelConfig,
    ):
        self.config = config

    def evaluate(
        self,
        request: RiskAssessmentRequest,
        *,
        now: datetime,
    ) -> RiskDataQuality:

        missing: list[str] = []
        stale: list[str] = []
        unverified: list[str] = []
        warnings: list[str] = []

        verified_count = 0
        present_count = 0

        for name in self.SOURCE_NAMES:
            source = getattr(
                request,
                name,
            )

            if source is None:
                missing.append(name)
                continue

            present_count += 1

            if not source.verified:
                unverified.append(name)
            else:
                verified_count += 1

            if isinstance(source, RiskObservation):
                max_age = (
                    self.config.forecast_max_age
                    if name == "forecast"
                    else self.config.source_max_age
                )

                age = now - source.observed_at

                if (
                    age > max_age
                    or age < -timedelta(minutes=5)
                ):
                    stale.append(name)

                if source.quality_score < 0.6:
                    warnings.append(
                        f"Low sensor quality: {name}"
                    )

        completeness = (
            present_count
            / len(self.SOURCE_NAMES)
            * 100
        )

        if (
            completeness == 100
            and verified_count == present_count
            and not stale
            and not warnings
        ):
            confidence = RiskConfidence.HIGH

        elif (
            completeness >= 75
            and verified_count >= 3
            and not stale
        ):
            confidence = RiskConfidence.MEDIUM

        elif (
            completeness >= self.config.minimum_completeness
            and present_count > 0
        ):
            confidence = RiskConfidence.LOW

        else:
            confidence = RiskConfidence.INSUFFICIENT

        if missing:
            warnings.append(
                "Missing sources must not be interpreted "
                "as low flood risk"
            )

        if stale:
            warnings.append(
                "Stale sensor readings require revalidation"
            )

        return RiskDataQuality(
            confidence=confidence,
            completeness_percent=round(
                completeness,
                2,
            ),
            verified_sources=verified_count,
            total_sources=present_count,
            stale_sources=tuple(stale),
            missing_sources=tuple(missing),
            unverified_sources=tuple(unverified),
            warnings=tuple(warnings),
        )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class RiskRecommendationEngine:

    def generate(
        self,
        *,
        level: FloodRiskLevel,
        quality: RiskDataQuality,
        request: RiskAssessmentRequest,
    ) -> tuple[RiskRecommendation, ...]:

        recommendations: list[RiskRecommendation] = []

        if quality.confidence in {
            RiskConfidence.LOW,
            RiskConfidence.INSUFFICIENT,
        }:
            recommendations.append(
                RiskRecommendation(
                    recommendation_type=(
                        RiskRecommendationType.VERIFY_DATA
                    ),
                    priority=level,
                    description=(
                        "Verify missing, stale or unverified "
                        "risk data before operational decisions."
                    ),
                )
            )

        if level in {
            FloodRiskLevel.HIGH,
            FloodRiskLevel.SEVERE,
            FloodRiskLevel.EXTREME,
        }:
            recommendations.extend([
                RiskRecommendation(
                    recommendation_type=(
                        RiskRecommendationType.REVIEW_ALERT
                    ),
                    priority=level,
                    description=(
                        "Authorized officials should review "
                        "whether an official flood alert "
                        "is warranted."
                    ),
                ),
                RiskRecommendation(
                    recommendation_type=(
                        RiskRecommendationType.REVIEW_RESCUE_READINESS
                    ),
                    priority=level,
                    description=(
                        "Review rescue team availability, "
                        "equipment and safe access routes."
                    ),
                ),
                RiskRecommendation(
                    recommendation_type=(
                        RiskRecommendationType.REVIEW_SHELTER_CAPACITY
                    ),
                    priority=level,
                    description=(
                        "Review verified shelter capacity "
                        "and accessibility."
                    ),
                ),
            ])

        if level in {
            FloodRiskLevel.SEVERE,
            FloodRiskLevel.EXTREME,
        }:
            recommendations.append(
                RiskRecommendation(
                    recommendation_type=(
                        RiskRecommendationType.REVIEW_EVACUATION
                    ),
                    priority=level,
                    description=(
                        "Qualified authorities should assess "
                        "evacuation requirements using "
                        "current field conditions."
                    ),
                )
            )

        if request.terrain is not None:
            if (
                request.terrain
                .drainage_vulnerability_score >= 0.7
            ):
                recommendations.append(
                    RiskRecommendation(
                        recommendation_type=(
                            RiskRecommendationType
                            .INSPECT_INFRASTRUCTURE
                        ),
                        priority=level,
                        description=(
                            "Review drainage and vulnerable "
                            "infrastructure conditions."
                        ),
                    )
                )

        if not recommendations:
            recommendations.append(
                RiskRecommendation(
                    recommendation_type=(
                        RiskRecommendationType.MONITOR
                    ),
                    priority=level,
                    description=(
                        "Continue verified monitoring and "
                        "review changing conditions."
                    ),
                )
            )

        return tuple(recommendations)


# ============================================================
# RISK ASSESSMENT ENGINE
# ============================================================

class FloodRiskAssessmentEngine:

    def __init__(
        self,
        config: RiskModelConfig | None = None,
    ):
        self.config = config or RiskModelConfig()

        self.scoring = FloodRiskScoringEngine(
            self.config
        )

        self.quality_evaluator = (
            RiskDataQualityEvaluator(
                self.config
            )
        )

        self.recommendations = (
            RiskRecommendationEngine()
        )

    def evaluate(
        self,
        request: RiskAssessmentRequest,
        *,
        previous_score: float | None = None,
        now: datetime | None = None,
    ) -> RiskAssessmentResult:

        now = now or utc_now()

        validate_datetime(
            now,
            "now",
        )

        quality = self.quality_evaluator.evaluate(
            request,
            now=now,
        )

        factors: list[RiskFactor] = []
        reasoning: list[str] = []

        factor_definitions = (
            (
                "rainfall",
                request.rainfall,
                self.scoring.rainfall_score,
            ),
            (
                "river",
                request.river,
                self.scoring.river_score,
            ),
            (
                "dam",
                request.dam,
                self.scoring.dam_score,
            ),
            (
                "forecast",
                request.forecast,
                self.scoring.forecast_score,
            ),
            (
                "terrain",
                request.terrain,
                self.scoring.terrain_score,
            ),
            (
                "exposure",
                request.exposure,
                self.scoring.exposure_score,
            ),
        )

        # Exclude stale/future observations from the
        # numerical score. Their absence reduces
        # confidence rather than implying safety.
        stale = set(
            quality.stale_sources
        )

        for name, source, scorer in factor_definitions:
            if source is None or name in stale:
                continue

            score = scorer(source)

            weight = self.config.weights[name]

            contribution = score * weight

            factors.append(
                RiskFactor(
                    name=name,
                    score=round(score, 4),
                    weight=weight,
                    contribution=round(
                        contribution,
                        4,
                    ),
                    verified=source.verified,
                    explanation=(
                        f"Illustrative {name} risk factor"
                    ),
                )
            )

        # The model intentionally does not rescale
        # missing factors to 100% because doing so can
        # amplify a single unreliable observation.
        #
        # The result is marked UNKNOWN when required
        # source coverage is insufficient.
        raw_score = sum(
            factor.contribution
            for factor in factors
        )

        score = round(
            clamp(raw_score),
            4,
        )

        usable_weight = sum(
            factor.weight
            for factor in factors
        )

        if (
            quality.confidence
            == RiskConfidence.INSUFFICIENT
            or usable_weight < 0.5
        ):
            level = FloodRiskLevel.UNKNOWN

        else:
            level = self.scoring.classify(
                score
            )

        if previous_score is None:
            trend = RiskTrend.UNKNOWN

        else:
            validate_number(
                previous_score,
                "previous_score",
                maximum=1.0,
            )

            difference = score - previous_score

            if difference >= 0.05:
                trend = RiskTrend.INCREASING

            elif difference <= -0.05:
                trend = RiskTrend.DECREASING

            else:
                trend = RiskTrend.STABLE

        if level == FloodRiskLevel.UNKNOWN:
            trend = RiskTrend.UNKNOWN

        for factor in factors:
            reasoning.append(
                f"{factor.name}: score={factor.score}, "
                f"weight={factor.weight}, "
                f"contribution={factor.contribution}"
            )

        if quality.missing_sources:
            reasoning.append(
                "Missing data: "
                + ", ".join(
                    quality.missing_sources
                )
            )

        if quality.stale_sources:
            reasoning.append(
                "Stale data: "
                + ", ".join(
                    quality.stale_sources
                )
            )

        if level == FloodRiskLevel.UNKNOWN:
            reasoning.append(
                "Insufficient evidence for a reliable "
                "flood risk classification."
            )

        recommendations = (
            self.recommendations.generate(
                level=level,
                quality=quality,
                request=request,
            )
        )

        return RiskAssessmentResult(
            score=score,
            risk_level=level,
            trend=trend,
            factors=tuple(factors),
            quality=quality,
            recommendations=recommendations,
            reasoning=tuple(reasoning),
            evaluated_at=now,
        )


# ============================================================
# RISK SERVICE
# ============================================================

class RiskService:

    def __init__(
        self,
        *,
        repository: RiskRepository,
        authorization_gateway: RiskAuthorizationGateway,
        audit_gateway: RiskAuditGateway | None = None,
        assessment_engine: (
            FloodRiskAssessmentEngine | None
        ) = None,
    ):
        self.repository = repository
        self.authorization = authorization_gateway
        self.audit = audit_gateway

        self.engine = (
            assessment_engine
            or FloodRiskAssessmentEngine()
        )

    async def _audit(
        self,
        action: str,
        assessment: FloodRiskAssessment,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="flood_risk_assessment",
            entity_id=assessment.id,
            metadata={
                "district_id": assessment.district_id,
                "risk_level": assessment.risk_level.value,
                "status": assessment.status.value,
                "version": assessment.version,
            },
        )

    async def _save(
        self,
        original: FloodRiskAssessment,
        updated: FloodRiskAssessment,
        *,
        action: str,
        actor_id: int,
    ) -> FloodRiskAssessment:

        saved = await self.repository.update(
            updated,
            expected_version=original.version,
        )

        await self._audit(
            action,
            saved,
            actor_id,
        )

        return saved

    async def get_assessment(
        self,
        assessment_id: str,
    ) -> FloodRiskAssessment:

        assessment = await self.repository.get_by_id(
            assessment_id
        )

        if assessment is None:
            raise RiskNotFoundError(
                "Risk assessment not found"
            )

        return assessment

    async def create_assessment(
        self,
        request: RiskAssessmentRequest,
        *,
        actor_id: int,
    ) -> FloodRiskAssessment:

        district_id = request.location.district_id

        if not await self.authorization.can_assess(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise RiskAuthorizationError(
                "Risk assessment permission denied"
            )

        previous_assessments = (
            await self.repository.list_by_location(
                request.location
            )
        )

        previous_verified = [
            item
            for item in previous_assessments
            if (
                item.status
                == RiskAssessmentStatus.VERIFIED
                and item.risk_level
                != FloodRiskLevel.UNKNOWN
            )
        ]

        previous_verified.sort(
            key=lambda item: item.created_at,
            reverse=True,
        )

        previous_score = (
            previous_verified[0].score
            if previous_verified
            else None
        )

        result = self.engine.evaluate(
            request,
            previous_score=previous_score,
        )

        assessment = FloodRiskAssessment(
            id=str(uuid.uuid4()),
            district_id=district_id,
            location=request.location,
            status=RiskAssessmentStatus.DRAFT,
            score=result.score,
            risk_level=result.risk_level,
            trend=result.trend,
            factors=list(result.factors),
            quality=result.quality,
            recommendations=list(
                result.recommendations
            ),
            reasoning=list(result.reasoning),
            created_by=actor_id,
            incident_id=request.incident_id,
            description=request.description.strip(),
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            assessment
        )

        await self._audit(
            "risk.assessment_created",
            created,
            actor_id,
        )

        return created

    async def submit_for_review(
        self,
        assessment_id: str,
        *,
        actor_id: int,
    ) -> FloodRiskAssessment:

        assessment = await self.get_assessment(
            assessment_id
        )

        if not await self.authorization.can_assess(
            user_id=actor_id,
            district_id=assessment.district_id,
        ):
            raise RiskAuthorizationError(
                "Risk submission permission denied"
            )

        if (
            assessment.status
            != RiskAssessmentStatus.DRAFT
        ):
            raise RiskConflictError(
                "Only draft assessments can be submitted"
            )

        updated = replace(
            assessment,
            status=RiskAssessmentStatus.PENDING_REVIEW,
            updated_at=utc_now(),
            version=assessment.version + 1,
        )

        return await self._save(
            assessment,
            updated,
            action="risk.submitted_for_review",
            actor_id=actor_id,
        )

    async def verify_assessment(
        self,
        assessment_id: str,
        *,
        reviewer_id: int,
    ) -> FloodRiskAssessment:

        assessment = await self.get_assessment(
            assessment_id
        )

        if not await self.authorization.can_review(
            user_id=reviewer_id,
            district_id=assessment.district_id,
        ):
            raise RiskAuthorizationError(
                "Risk verification permission denied"
            )

        if (
            assessment.status
            != RiskAssessmentStatus.PENDING_REVIEW
        ):
            raise RiskConflictError(
                "Assessment is not awaiting review"
            )

        if reviewer_id == assessment.created_by:
            raise RiskAuthorizationError(
                "Independent reviewer required"
            )

        if (
            assessment.quality.confidence
            == RiskConfidence.INSUFFICIENT
        ):
            raise RiskVerificationError(
                "Insufficient source data for verification"
            )

        if assessment.quality.stale_sources:
            raise RiskVerificationError(
                "Stale source data must be refreshed"
            )

        if (
            assessment.risk_level
            == FloodRiskLevel.UNKNOWN
        ):
            raise RiskVerificationError(
                "Unknown risk cannot be verified "
                "as a definitive risk classification"
            )

        updated = replace(
            assessment,
            status=RiskAssessmentStatus.VERIFIED,
            reviewed_by=reviewer_id,
            reviewed_at=utc_now(),
            updated_at=utc_now(),
            version=assessment.version + 1,
        )

        return await self._save(
            assessment,
            updated,
            action="risk.assessment_verified",
            actor_id=reviewer_id,
        )

    async def reject_assessment(
        self,
        assessment_id: str,
        *,
        reviewer_id: int,
        reason: str,
    ) -> FloodRiskAssessment:

        assessment = await self.get_assessment(
            assessment_id
        )

        if not await self.authorization.can_review(
            user_id=reviewer_id,
            district_id=assessment.district_id,
        ):
            raise RiskAuthorizationError(
                "Risk rejection permission denied"
            )

        if (
            assessment.status
            != RiskAssessmentStatus.PENDING_REVIEW
        ):
            raise RiskConflictError(
                "Assessment is not awaiting review"
            )

        reason = require_text(
            reason,
            "Rejection reason",
        )

        updated = replace(
            assessment,
            status=RiskAssessmentStatus.REJECTED,
            reviewed_by=reviewer_id,
            reviewed_at=utc_now(),
            rejection_reason=reason,
            updated_at=utc_now(),
            version=assessment.version + 1,
        )

        return await self._save(
            assessment,
            updated,
            action="risk.assessment_rejected",
            actor_id=reviewer_id,
        )

    async def supersede_assessment(
        self,
        assessment_id: str,
        *,
        actor_id: int,
    ) -> FloodRiskAssessment:

        assessment = await self.get_assessment(
            assessment_id
        )

        if not await self.authorization.can_review(
            user_id=actor_id,
            district_id=assessment.district_id,
        ):
            raise RiskAuthorizationError(
                "Risk supersession permission denied"
            )

        if (
            assessment.status
            != RiskAssessmentStatus.VERIFIED
        ):
            raise RiskConflictError(
                "Only verified assessments can be superseded"
            )

        updated = replace(
            assessment,
            status=RiskAssessmentStatus.SUPERSEDED,
            updated_at=utc_now(),
            version=assessment.version + 1,
        )

        return await self._save(
            assessment,
            updated,
            action="risk.assessment_superseded",
            actor_id=actor_id,
        )

    async def get_latest_verified_risk(
        self,
        location: RiskLocation,
        *,
        actor_id: int,
    ) -> FloodRiskAssessment | None:

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=location.district_id,
        ):
            raise RiskAuthorizationError(
                "Risk data access denied"
            )

        assessments = (
            await self.repository.list_by_location(
                location
            )
        )

        verified = [
            assessment
            for assessment in assessments
            if (
                assessment.status
                == RiskAssessmentStatus.VERIFIED
            )
        ]

        if not verified:
            return None

        return max(
            verified,
            key=lambda assessment: (
                assessment.reviewed_at
                or assessment.created_at
            ),
        )

    async def list_district_assessments(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> Sequence[FloodRiskAssessment]:

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise RiskAuthorizationError(
                "District risk access denied"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        assessments = (
            await self.list_district_assessments(
                district_id,
                actor_id=actor_id,
            )
        )

        level_counts = {
            level.value: 0
            for level in FloodRiskLevel
        }

        status_counts = {
            status.value: 0
            for status in RiskAssessmentStatus
        }

        for assessment in assessments:
            level_counts[
                assessment.risk_level.value
            ] += 1

            status_counts[
                assessment.status.value
            ] += 1

        verified = [
            assessment
            for assessment in assessments
            if (
                assessment.status
                == RiskAssessmentStatus.VERIFIED
            )
        ]

        average_score = (
            sum(
                assessment.score
                for assessment in verified
            ) / len(verified)
            if verified
            else None
        )

        return {
            "district_id": district_id,
            "total_assessments": len(assessments),
            "verified_assessments": len(verified),
            "average_verified_score": (
                round(average_score, 4)
                if average_score is not None
                else None
            ),
            "level_counts": level_counts,
            "status_counts": status_counts,
            "generated_at": utc_now().isoformat(),
            "warning": (
                "Assessment counts are not counts "
                "of unique affected locations. "
                "Scores are illustrative and "
                "not official flood warnings."
            ),
        }

    async def generate_risk_report(
        self,
        assessment_id: str,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        assessment = await self.get_assessment(
            assessment_id
        )

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=assessment.district_id,
        ):
            raise RiskAuthorizationError(
                "Risk report access denied"
            )

        return {
            "assessment_id": assessment.id,
            "district_id": assessment.district_id,
            "incident_id": assessment.incident_id,
            "location": {
                "latitude": assessment.location.latitude,
                "longitude": assessment.location.longitude,
                "location_name": (
                    assessment.location.location_name
                ),
                "panchayat": (
                    assessment.location.panchayat
                ),
                "taluk": assessment.location.taluk,
            },
            "status": assessment.status.value,
            "risk_score": assessment.score,
            "risk_level": assessment.risk_level.value,
            "trend": assessment.trend.value,
            "confidence": (
                assessment.quality.confidence.value
            ),
            "data_quality": {
                "completeness_percent": (
                    assessment.quality.completeness_percent
                ),
                "verified_sources": (
                    assessment.quality.verified_sources
                ),
                "total_sources": (
                    assessment.quality.total_sources
                ),
                "missing_sources": list(
                    assessment.quality.missing_sources
                ),
                "stale_sources": list(
                    assessment.quality.stale_sources
                ),
                "unverified_sources": list(
                    assessment.quality.unverified_sources
                ),
                "warnings": list(
                    assessment.quality.warnings
                ),
            },
            "factors": [
                {
                    "name": factor.name,
                    "score": factor.score,
                    "weight": factor.weight,
                    "contribution": factor.contribution,
                    "verified": factor.verified,
                    "explanation": factor.explanation,
                }
                for factor in assessment.factors
            ],
            "recommendations": [
                {
                    "type": (
                        recommendation
                        .recommendation_type.value
                    ),
                    "priority": (
                        recommendation.priority.value
                    ),
                    "description": (
                        recommendation.description
                    ),
                    "requires_human_approval": (
                        recommendation.requires_human_approval
                    ),
                }
                for recommendation
                in assessment.recommendations
            ],
            "reasoning": list(
                assessment.reasoning
            ),
            "created_at": (
                assessment.created_at.isoformat()
            ),
            "reviewed_at": (
                assessment.reviewed_at.isoformat()
                if assessment.reviewed_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "This assessment uses illustrative "
                "risk scoring. It is not an official "
                "flood warning or evacuation order. "
                "Use verified observations and "
                "authorized disaster-management guidance."
            ),
        }
