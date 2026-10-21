"""Placeholder module: ai/agents/flood_prediction_agent.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
File: backend/ai/agents/flood_prediction_agent.py

Flood Prediction Intelligence Agent
===================================

Features:
    - Rainfall-based flood risk assessment
    - River water-level monitoring
    - River rise-rate analysis
    - Reservoir/dam monitoring
    - Weather forecast integration
    - ML/DL flood prediction integration
    - LSTM/Transformer/TFT forecast adapter support
    - Multi-model forecast aggregation
    - Forecast uncertainty estimation
    - Data freshness and quality validation
    - District-level flood-risk screening
    - Forecast horizon analysis
    - Model disagreement detection
    - Historical baseline comparison
    - Human-reviewed forecast recommendations
    - Prediction deduplication
    - Audit logging
    - JSON report export

Safety:
    Predictions are advisory only.

    This agent does NOT:
        - Issue public flood alerts
        - Order evacuations
        - Dispatch rescue teams
        - Control dam gates
        - Certify an area as safe

    Human review and official verification are required.

Python: 3.11+
Dependencies: Standard library only

Integration:
    ML models must be called through explicit adapters.
    Model predictions must include calibrated probabilities
    and independently validated model metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
import uuid

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class FloodPredictionAgentError(Exception):
    pass


class FloodPredictionValidationError(
    FloodPredictionAgentError
):
    pass


class FloodPredictionModelError(
    FloodPredictionAgentError
):
    pass


class FloodPredictionApprovalError(
    FloodPredictionAgentError
):
    pass


class FloodPredictionExportError(
    FloodPredictionAgentError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class FloodRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNDETERMINED = "undetermined"


class ForecastModelType(str, Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    TEMPORAL_FUSION = "temporal_fusion"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    PHYSICS_BASED = "physics_based"
    EXTERNAL = "external"


class ForecastStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED_FOR_ANALYSIS = "approved_for_analysis"
    REJECTED = "rejected"


class ForecastConfidence(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class ObservationType(str, Enum):
    RAINFALL = "rainfall"
    RIVER_LEVEL = "river_level"
    RIVER_RISE_RATE = "river_rise_rate"
    DAM_STORAGE = "dam_storage"
    DAM_RELEASE = "dam_release"
    WEATHER_FORECAST = "weather_forecast"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FloodPredictionAgentConfig:
    maximum_observation_age_minutes: int = 180

    maximum_model_prediction_age_minutes: int = 120

    minimum_model_reliability: float = 0.60

    minimum_independent_models: int = 2

    minimum_independent_observation_sources: int = 2

    moderate_risk_threshold: float = 0.40
    high_risk_threshold: float = 0.70
    critical_risk_threshold: float = 0.90

    maximum_model_disagreement: float = 0.25

    rainfall_24h_moderate_mm: float = 50.0
    rainfall_24h_high_mm: float = 115.0

    river_rise_high_m_per_hour: float = 0.40

    dam_storage_high_percent: float = 85.0

    forecast_horizons_hours: tuple[int, ...] = (
        6,
        12,
        24,
        48,
    )

    deduplication_window_minutes: int = 60

    maximum_observations: int = 1000
    maximum_predictions: int = 100

    require_human_review: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "maximum_observation_age_minutes",
            "maximum_model_prediction_age_minutes",
            "minimum_independent_models",
            "minimum_independent_observation_sources",
            "deduplication_window_minutes",
            "maximum_observations",
            "maximum_predictions",
        )

        for name in integer_fields:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise FloodPredictionValidationError(
                    f"{name} must be a positive integer"
                )

        probability_fields = (
            "minimum_model_reliability",
            "moderate_risk_threshold",
            "high_risk_threshold",
            "critical_risk_threshold",
            "maximum_model_disagreement",
        )

        for name in probability_fields:
            require_probability(
                getattr(self, name),
                name,
            )

        if not (
            self.moderate_risk_threshold
            <= self.high_risk_threshold
            <= self.critical_risk_threshold
        ):
            raise FloodPredictionValidationError(
                "Risk thresholds must be ascending"
            )

        if (
            not self.forecast_horizons_hours
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.forecast_horizons_hours
            )
            or len(set(self.forecast_horizons_hours))
            != len(self.forecast_horizons_hours)
        ):
            raise FloodPredictionValidationError(
                "Forecast horizons must be unique positive integers"
            )


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def require_aware_datetime(
    value: datetime,
) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise FloodPredictionValidationError(
            "Datetime must be timezone-aware"
        )


def require_probability(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise FloodPredictionValidationError(
            f"{name} must be in [0,1]"
        )


def require_finite(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise FloodPredictionValidationError(
            f"{name} must be finite"
        )


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            serialize(item)
            for item in value
        ]

    return value


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class FloodObservation:
    observation_id: str

    district: str
    station_id: str

    observation_type: ObservationType

    value: float
    unit: str

    observed_at: datetime

    verified: bool
    reliability: float

    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise FloodPredictionValidationError(
                "observation_id is required"
            )

        if not self.district.strip():
            raise FloodPredictionValidationError(
                "district is required"
            )

        if not self.station_id.strip():
            raise FloodPredictionValidationError(
                "station_id is required"
            )

        if not self.unit.strip():
            raise FloodPredictionValidationError(
                "unit is required"
            )

        require_finite(
            self.value,
            "observation value",
        )

        require_probability(
            self.reliability,
            "reliability",
        )

        require_aware_datetime(
            self.observed_at
        )


@dataclass(frozen=True)
class FloodModelPrediction:
    prediction_id: str

    model_id: str
    model_version: str
    model_type: ForecastModelType

    district: str

    forecast_horizon_hours: int

    flood_probability: float

    predicted_river_level_m: float | None = None
    predicted_rainfall_mm: float | None = None

    model_reliability: float = 0.80

    generated_at: datetime | None = None

    calibrated: bool = False
    validated: bool = False

    def __post_init__(self) -> None:
        for name in (
            "prediction_id",
            "model_id",
            "model_version",
            "district",
        ):
            if not getattr(self, name).strip():
                raise FloodPredictionValidationError(
                    f"{name} is required"
                )

        if (
            isinstance(self.forecast_horizon_hours, bool)
            or not isinstance(
                self.forecast_horizon_hours,
                int,
            )
            or self.forecast_horizon_hours <= 0
        ):
            raise FloodPredictionValidationError(
                "Invalid forecast horizon"
            )

        require_probability(
            self.flood_probability,
            "flood_probability",
        )

        require_probability(
            self.model_reliability,
            "model_reliability",
        )

        for name in (
            "predicted_river_level_m",
            "predicted_rainfall_mm",
        ):
            value = getattr(self, name)

            if value is not None:
                require_finite(value, name)

        if self.generated_at is not None:
            require_aware_datetime(
                self.generated_at
            )


@dataclass(frozen=True)
class FloodPredictionRequest:
    request_id: str
    district: str

    observations: tuple[
        FloodObservation,
        ...
    ]

    model_predictions: tuple[
        FloodModelPrediction,
        ...
    ]

    requested_at: datetime


@dataclass(frozen=True)
class ObservationQualityResult:
    accepted_observation_ids: tuple[str, ...]
    rejected_observation_ids: tuple[str, ...]

    independent_source_count: int

    rainfall_24h_mm: float | None
    river_level_m: float | None
    river_rise_m_per_hour: float | None
    dam_storage_percent: float | None

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class HorizonForecast:
    horizon_hours: int

    risk_level: FloodRiskLevel

    flood_probability: float | None

    confidence: ForecastConfidence

    model_count: int

    model_disagreement: float | None

    predicted_river_level_m: float | None
    predicted_rainfall_mm: float | None

    model_ids: tuple[str, ...]

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FloodPredictionAssessment:
    assessment_id: str

    district: str

    overall_risk: FloodRiskLevel
    overall_probability: float | None

    confidence: ForecastConfidence

    horizon_forecasts: tuple[
        HorizonForecast,
        ...
    ]

    observation_quality: ObservationQualityResult

    recommended_actions: tuple[str, ...]
    warnings: tuple[str, ...]

    assessed_at: datetime


@dataclass(frozen=True)
class FloodPredictionReport:
    report_id: str
    request_id: str

    assessment: FloodPredictionAssessment

    status: ForecastStatus

    requires_human_review: bool

    duplicate_detected: bool

    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class FloodPredictionReview:
    review_id: str
    report_id: str

    reviewer_id: str
    reviewer_role: str

    approved: bool
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class FloodPredictionAuditRecord:
    audit_id: str
    report_id: str

    action: str
    actor_id: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# MODEL ADAPTER INTERFACE
# ============================================================

class FloodForecastModelAdapter(Protocol):
    """
    Interface for external trained forecasting models.

    Implementations may wrap:
        - LSTM flood models
        - Transformer forecasting models
        - Temporal Fusion Transformers
        - XGBoost models
        - Physics-based flood models

    The adapter must return a validated
    FloodModelPrediction object.

    Do not mark predictions as calibrated or
    validated without supporting evaluation.
    """

    def predict(
        self,
        *,
        district: str,
        horizon_hours: int,
        observations: Sequence[FloodObservation],
    ) -> FloodModelPrediction:
        ...


class FloodModelAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[
            str,
            FloodForecastModelAdapter,
        ] = {}

    def register(
        self,
        model_id: str,
        adapter: FloodForecastModelAdapter,
    ) -> None:
        if not model_id.strip():
            raise FloodPredictionModelError(
                "model_id is required"
            )

        if model_id in self._adapters:
            raise FloodPredictionModelError(
                "Model adapter already registered"
            )

        self._adapters[model_id] = adapter

    def predict_all(
        self,
        *,
        district: str,
        horizons: Sequence[int],
        observations: Sequence[FloodObservation],
    ) -> tuple[FloodModelPrediction, ...]:
        predictions = []

        for model_id, adapter in self._adapters.items():
            for horizon in horizons:
                try:
                    prediction = adapter.predict(
                        district=district,
                        horizon_hours=horizon,
                        observations=observations,
                    )

                except Exception as exc:
                    raise FloodPredictionModelError(
                        f"Model {model_id} failed: {exc}"
                    ) from exc

                if prediction.model_id != model_id:
                    raise FloodPredictionModelError(
                        "Adapter returned incorrect model ID"
                    )

                if (
                    prediction.district.casefold()
                    != district.casefold()
                    or prediction.forecast_horizon_hours
                    != horizon
                ):
                    raise FloodPredictionModelError(
                        "Adapter returned mismatched prediction"
                    )

                predictions.append(
                    prediction
                )

        return tuple(predictions)


# ============================================================
# OBSERVATION QUALITY VALIDATOR
# ============================================================

class FloodObservationQualityEngine:
    EXPECTED_UNITS = {
        ObservationType.RAINFALL: "mm",
        ObservationType.RIVER_LEVEL: "m",
        ObservationType.RIVER_RISE_RATE: "m/hour",
        ObservationType.DAM_STORAGE: "%",
        ObservationType.DAM_RELEASE: "m3/s",
        ObservationType.WEATHER_FORECAST: "probability",
    }

    def __init__(
        self,
        config: FloodPredictionAgentConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        observations: Sequence[FloodObservation],
        *,
        district: str,
        now: datetime,
    ) -> ObservationQualityResult:
        if len(observations) > (
            self.config.maximum_observations
        ):
            raise FloodPredictionValidationError(
                "Observation limit exceeded"
            )

        accepted = []
        rejected = []
        warnings = []

        seen_ids = set()

        for observation in observations:
            reason = None

            if observation.observation_id in seen_ids:
                reason = "duplicate observation"

            elif (
                observation.district.casefold()
                != district.casefold()
            ):
                reason = "district mismatch"

            elif not observation.verified:
                reason = "unverified observation"

            elif observation.reliability < (
                self.config.minimum_model_reliability
            ):
                reason = "low source reliability"

            elif observation.observed_at > (
                now + timedelta(minutes=5)
            ):
                reason = "future timestamp"

            elif (
                now - observation.observed_at
            ) > timedelta(
                minutes=(
                    self.config.maximum_observation_age_minutes
                )
            ):
                reason = "stale observation"

            elif observation.unit != (
                self.EXPECTED_UNITS[
                    observation.observation_type
                ]
            ):
                reason = "unexpected measurement unit"

            elif (
                observation.observation_type in (
                    ObservationType.RAINFALL,
                    ObservationType.DAM_STORAGE,
                    ObservationType.DAM_RELEASE,
                )
                and observation.value < 0
            ):
                reason = "negative measurement"

            elif (
                observation.observation_type
                == ObservationType.DAM_STORAGE
                and observation.value > 100
            ):
                reason = "storage percentage exceeds 100"

            elif (
                observation.observation_type
                == ObservationType.WEATHER_FORECAST
                and not 0 <= observation.value <= 1
            ):
                reason = "forecast probability out of range"

            seen_ids.add(
                observation.observation_id
            )

            if reason:
                rejected.append(
                    observation.observation_id
                )

                warnings.append(
                    f"{observation.observation_id}: {reason}"
                )

            else:
                accepted.append(
                    observation
                )

        def latest_value(
            observation_type: ObservationType,
        ) -> float | None:
            candidates = [
                item
                for item in accepted
                if item.observation_type
                == observation_type
            ]

            if not candidates:
                return None

            latest = max(
                candidates,
                key=lambda item: item.observed_at,
            )

            return latest.value

        sources = {
            item.station_id
            for item in accepted
        }

        if len(sources) < (
            self.config.minimum_independent_observation_sources
        ):
            warnings.append(
                "Insufficient independent observation sources"
            )

        return ObservationQualityResult(
            accepted_observation_ids=tuple(
                item.observation_id
                for item in accepted
            ),
            rejected_observation_ids=tuple(
                rejected
            ),
            independent_source_count=len(sources),
            rainfall_24h_mm=latest_value(
                ObservationType.RAINFALL
            ),
            river_level_m=latest_value(
                ObservationType.RIVER_LEVEL
            ),
            river_rise_m_per_hour=latest_value(
                ObservationType.RIVER_RISE_RATE
            ),
            dam_storage_percent=latest_value(
                ObservationType.DAM_STORAGE
            ),
            warnings=tuple(warnings),
        )


# ============================================================
# MODEL PREDICTION VALIDATOR
# ============================================================

class FloodModelPredictionValidator:
    def __init__(
        self,
        config: FloodPredictionAgentConfig,
    ) -> None:
        self.config = config

    def filter(
        self,
        predictions: Sequence[FloodModelPrediction],
        *,
        district: str,
        now: datetime,
    ) -> tuple[
        tuple[FloodModelPrediction, ...],
        tuple[str, ...],
    ]:
        if len(predictions) > (
            self.config.maximum_predictions
        ):
            raise FloodPredictionValidationError(
                "Prediction limit exceeded"
            )

        accepted = []
        warnings = []
        seen_ids = set()

        for prediction in predictions:
            reason = None

            if prediction.prediction_id in seen_ids:
                reason = "duplicate prediction ID"

            elif (
                prediction.district.casefold()
                != district.casefold()
            ):
                reason = "district mismatch"

            elif prediction.forecast_horizon_hours not in (
                self.config.forecast_horizons_hours
            ):
                reason = "unsupported forecast horizon"

            elif not prediction.validated:
                reason = "model not independently validated"

            elif not prediction.calibrated:
                reason = "probabilities not calibrated"

            elif prediction.model_reliability < (
                self.config.minimum_model_reliability
            ):
                reason = "model reliability too low"

            elif prediction.generated_at is None:
                reason = "missing generation timestamp"

            elif prediction.generated_at > (
                now + timedelta(minutes=5)
            ):
                reason = "future timestamp"

            elif (
                now - prediction.generated_at
            ) > timedelta(
                minutes=(
                    self.config.maximum_model_prediction_age_minutes
                )
            ):
                reason = "stale prediction"

            seen_ids.add(
                prediction.prediction_id
            )

            if reason:
                warnings.append(
                    f"{prediction.prediction_id}: {reason}"
                )

            else:
                accepted.append(
                    prediction
                )

        return (
            tuple(accepted),
            tuple(warnings),
        )


# ============================================================
# MULTI-MODEL FORECAST FUSION
# ============================================================

class FloodForecastFusionEngine:
    def __init__(
        self,
        config: FloodPredictionAgentConfig,
    ) -> None:
        self.config = config

    def classify_risk(
        self,
        probability: float | None,
    ) -> FloodRiskLevel:
        if probability is None:
            return FloodRiskLevel.UNDETERMINED

        if probability >= (
            self.config.critical_risk_threshold
        ):
            return FloodRiskLevel.CRITICAL

        if probability >= (
            self.config.high_risk_threshold
        ):
            return FloodRiskLevel.HIGH

        if probability >= (
            self.config.moderate_risk_threshold
        ):
            return FloodRiskLevel.MODERATE

        return FloodRiskLevel.LOW

    @staticmethod
    def weighted_average(
        values: Sequence[tuple[float, float]],
    ) -> float | None:
        if not values:
            return None

        total_weight = sum(
            weight
            for _, weight in values
        )

        if total_weight <= 0:
            return None

        return sum(
            value * weight
            for value, weight in values
        ) / total_weight

    def aggregate_horizon(
        self,
        predictions: Sequence[FloodModelPrediction],
        *,
        horizon_hours: int,
    ) -> HorizonForecast:
        candidates = [
            prediction
            for prediction in predictions
            if prediction.forecast_horizon_hours
            == horizon_hours
        ]

        # Use one prediction per model.
        by_model: dict[
            str,
            FloodModelPrediction,
        ] = {}

        for prediction in candidates:
            previous = by_model.get(
                prediction.model_id
            )

            if (
                previous is None
                or (
                    prediction.generated_at is not None
                    and previous.generated_at is not None
                    and prediction.generated_at
                    > previous.generated_at
                )
            ):
                by_model[
                    prediction.model_id
                ] = prediction

        candidates = list(
            by_model.values()
        )

        model_count = len(
            candidates
        )

        warnings = []

        if model_count < (
            self.config.minimum_independent_models
        ):
            warnings.append(
                "Insufficient independent forecasting models"
            )

            return HorizonForecast(
                horizon_hours=horizon_hours,
                risk_level=FloodRiskLevel.UNDETERMINED,
                flood_probability=None,
                confidence=ForecastConfidence.INSUFFICIENT,
                model_count=model_count,
                model_disagreement=None,
                predicted_river_level_m=None,
                predicted_rainfall_mm=None,
                model_ids=tuple(
                    sorted(by_model)
                ),
                warnings=tuple(warnings),
            )

        probabilities = [
            prediction.flood_probability
            for prediction in candidates
        ]

        weighted_probability = (
            self.weighted_average([
                (
                    prediction.flood_probability,
                    prediction.model_reliability,
                )
                for prediction in candidates
            ])
        )

        disagreement = (
            max(probabilities)
            - min(probabilities)
        )

        if disagreement > (
            self.config.maximum_model_disagreement
        ):
            confidence = ForecastConfidence.LOW

            warnings.append(
                "Large disagreement between forecasting models"
            )

        elif disagreement > (
            self.config.maximum_model_disagreement / 2
        ):
            confidence = ForecastConfidence.MODERATE

        else:
            confidence = ForecastConfidence.HIGH

        river_level = self.weighted_average([
            (
                prediction.predicted_river_level_m,
                prediction.model_reliability,
            )
            for prediction in candidates
            if prediction.predicted_river_level_m
            is not None
        ])

        rainfall = self.weighted_average([
            (
                prediction.predicted_rainfall_mm,
                prediction.model_reliability,
            )
            for prediction in candidates
            if prediction.predicted_rainfall_mm
            is not None
        ])

        return HorizonForecast(
            horizon_hours=horizon_hours,
            risk_level=self.classify_risk(
                weighted_probability
            ),
            flood_probability=(
                round(weighted_probability, 4)
                if weighted_probability is not None
                else None
            ),
            confidence=confidence,
            model_count=model_count,
            model_disagreement=round(
                disagreement,
                4,
            ),
            predicted_river_level_m=(
                round(river_level, 3)
                if river_level is not None
                else None
            ),
            predicted_rainfall_mm=(
                round(rainfall, 2)
                if rainfall is not None
                else None
            ),
            model_ids=tuple(
                sorted(by_model)
            ),
            warnings=tuple(warnings),
        )


# ============================================================
# OBSERVATION-BASED RISK INDICATORS
# ============================================================

class FloodObservationRiskAnalyzer:
    def __init__(
        self,
        config: FloodPredictionAgentConfig,
    ) -> None:
        self.config = config

    def analyze(
        self,
        quality: ObservationQualityResult,
    ) -> tuple[str, ...]:
        findings = []

        rainfall = quality.rainfall_24h_mm

        if rainfall is not None:
            if rainfall >= (
                self.config.rainfall_24h_high_mm
            ):
                findings.append(
                    "High 24-hour rainfall threshold exceeded"
                )

            elif rainfall >= (
                self.config.rainfall_24h_moderate_mm
            ):
                findings.append(
                    "Moderate rainfall screening threshold exceeded"
                )

        rise_rate = quality.river_rise_m_per_hour

        if (
            rise_rate is not None
            and rise_rate >= (
                self.config.river_rise_high_m_per_hour
            )
        ):
            findings.append(
                "Rapid river-level rise reported"
            )

        storage = quality.dam_storage_percent

        if (
            storage is not None
            and storage >= (
                self.config.dam_storage_high_percent
            )
        ):
            findings.append(
                "High dam storage reported; "
                "downstream risk requires separate assessment"
            )

        return tuple(findings)


# ============================================================
# FORECAST RECOMMENDATION ENGINE
# ============================================================

class FloodPredictionRecommendationEngine:
    @staticmethod
    def generate(
        *,
        overall_risk: FloodRiskLevel,
        horizons: Sequence[HorizonForecast],
        observation_quality: ObservationQualityResult,
    ) -> tuple[str, ...]:
        actions = [
            "Verify latest official weather and river bulletins",
            "Review source freshness and forecast model validity",
        ]

        if overall_risk == FloodRiskLevel.CRITICAL:
            actions.append(
                "Escalate forecast assessment for immediate "
                "review by authorized disaster-management officials"
            )

        elif overall_risk == FloodRiskLevel.HIGH:
            actions.append(
                "Prioritize district-level flood preparedness review"
            )

        elif overall_risk == FloodRiskLevel.MODERATE:
            actions.append(
                "Increase monitoring of relevant river "
                "and rainfall stations"
            )

        elif overall_risk == FloodRiskLevel.UNDETERMINED:
            actions.append(
                "Collect sufficient validated model forecasts"
            )

        if any(
            horizon.confidence in (
                ForecastConfidence.LOW,
                ForecastConfidence.INSUFFICIENT,
            )
            for horizon in horizons
        ):
            actions.append(
                "Investigate model disagreement or missing forecasts"
            )

        if observation_quality.independent_source_count < 2:
            actions.append(
                "Obtain additional independent monitoring observations"
            )

        actions.append(
            "Do not automatically issue public alerts "
            "or initiate emergency operations"
        )

        return tuple(
            dict.fromkeys(actions)
        )


# ============================================================
# DEDUPLICATION ENGINE
# ============================================================

class FloodPredictionDeduplicationEngine:
    def __init__(
        self,
        config: FloodPredictionAgentConfig,
    ) -> None:
        self.config = config

        self._recent: dict[
            str,
            datetime,
        ] = {}

        self._lock = threading.Lock()

    def check(
        self,
        *,
        district: str,
        prediction_ids: Sequence[str],
        now: datetime,
    ) -> bool:
        content = (
            district.casefold()
            + "|"
            + "|".join(
                sorted(prediction_ids)
            )
        )

        key = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        cutoff = now - timedelta(
            minutes=(
                self.config.deduplication_window_minutes
            )
        )

        with self._lock:
            self._recent = {
                item_key: timestamp
                for item_key, timestamp in self._recent.items()
                if timestamp >= cutoff
            }

            previous = self._recent.get(key)

            duplicate = (
                previous is not None
                and previous >= cutoff
            )

            if not duplicate:
                self._recent[key] = now

        return duplicate


# ============================================================
# AUDIT LOGGER
# ============================================================

class FloodPredictionAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            FloodPredictionAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        report_id: str,
        action: str,
        actor_id: str,
        details: Mapping[str, str] | None = None,
    ) -> FloodPredictionAuditRecord:
        record = FloodPredictionAuditRecord(
            audit_id=generate_id(),
            report_id=report_id,
            action=action,
            actor_id=actor_id,
            timestamp=utc_now(),
            details=dict(
                details or {}
            ),
        )

        with self._lock:
            self._records.append(
                record
            )

        return record

    def list_records(
        self,
        report_id: str | None = None,
    ) -> tuple[
        FloodPredictionAuditRecord,
        ...
    ]:
        with self._lock:
            records = tuple(
                self._records
            )

        if report_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.report_id == report_id
        )


# ============================================================
# HUMAN REVIEW MANAGER
# ============================================================

class FloodPredictionReviewManager:
    ALLOWED_ROLES = {
        "district_emergency_official",
        "hydrology_specialist",
        "forecast_supervisor",
        "disaster_management_officer",
    }

    def __init__(
        self,
        audit_logger: FloodPredictionAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            FloodPredictionReview,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        report: FloodPredictionReport,
        *,
        reviewer_id: str,
        reviewer_role: str,
        approved: bool,
        reason: str,
    ) -> FloodPredictionReview:
        if not reviewer_id.strip():
            raise FloodPredictionApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise FloodPredictionApprovalError(
                "Reviewer role not permitted"
            )

        if not reason.strip():
            raise FloodPredictionApprovalError(
                "Review reason required"
            )

        if report.status != ForecastStatus.PENDING_REVIEW:
            raise FloodPredictionApprovalError(
                "Report is not pending review"
            )

        if report.duplicate_detected and approved:
            raise FloodPredictionApprovalError(
                "Duplicate report requires separate review"
            )

        with self._lock:
            if report.report_id in self._reviews:
                raise FloodPredictionApprovalError(
                    "Report already reviewed"
                )

            review = FloodPredictionReview(
                review_id=generate_id(),
                report_id=report.report_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                approved=approved,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[report.report_id] = review

        self.audit_logger.record(
            report_id=report.report_id,
            action=(
                "forecast_review_approved"
                if approved
                else "forecast_review_rejected"
            ),
            actor_id=reviewer_id,
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN FLOOD PREDICTION AGENT
# ============================================================

class FloodPredictionAgent:
    def __init__(
        self,
        config: FloodPredictionAgentConfig | None = None,
    ) -> None:
        self.config = (
            config or FloodPredictionAgentConfig()
        )

        self.observation_engine = (
            FloodObservationQualityEngine(
                self.config
            )
        )

        self.prediction_validator = (
            FloodModelPredictionValidator(
                self.config
            )
        )

        self.fusion_engine = (
            FloodForecastFusionEngine(
                self.config
            )
        )

        self.observation_risk_analyzer = (
            FloodObservationRiskAnalyzer(
                self.config
            )
        )

        self.recommendation_engine = (
            FloodPredictionRecommendationEngine()
        )

        self.deduplication_engine = (
            FloodPredictionDeduplicationEngine(
                self.config
            )
        )

        self.audit_logger = (
            FloodPredictionAuditLogger()
        )

        self.review_manager = (
            FloodPredictionReviewManager(
                self.audit_logger
            )
        )

        self.model_registry = (
            FloodModelAdapterRegistry()
        )

        self._reports: dict[
            str,
            FloodPredictionReport,
        ] = {}

        self._lock = threading.Lock()

    def register_model(
        self,
        model_id: str,
        adapter: FloodForecastModelAdapter,
    ) -> None:
        self.model_registry.register(
            model_id,
            adapter,
        )

    def run_registered_models(
        self,
        *,
        district: str,
        observations: Sequence[FloodObservation],
    ) -> tuple[FloodModelPrediction, ...]:
        return self.model_registry.predict_all(
            district=district,
            horizons=self.config.forecast_horizons_hours,
            observations=observations,
        )

    def run(
        self,
        request: FloodPredictionRequest,
        *,
        now: datetime | None = None,
    ) -> FloodPredictionReport:
        now = now or utc_now()

        require_aware_datetime(now)
        require_aware_datetime(
            request.requested_at
        )

        if not request.request_id.strip():
            raise FloodPredictionValidationError(
                "request_id is required"
            )

        if not request.district.strip():
            raise FloodPredictionValidationError(
                "district is required"
            )

        observation_quality = (
            self.observation_engine.evaluate(
                request.observations,
                district=request.district,
                now=now,
            )
        )

        (
            accepted_predictions,
            prediction_warnings,
        ) = self.prediction_validator.filter(
            request.model_predictions,
            district=request.district,
            now=now,
        )

        horizons = tuple(
            self.fusion_engine.aggregate_horizon(
                accepted_predictions,
                horizon_hours=horizon,
            )
            for horizon in (
                self.config.forecast_horizons_hours
            )
        )

        available_horizons = [
            horizon
            for horizon in horizons
            if horizon.flood_probability is not None
        ]

        if available_horizons:
            strongest = max(
                available_horizons,
                key=lambda item: item.flood_probability,
            )

            overall_probability = (
                strongest.flood_probability
            )

            overall_risk = (
                strongest.risk_level
            )

            if any(
                horizon.confidence
                == ForecastConfidence.LOW
                for horizon in available_horizons
            ):
                overall_confidence = (
                    ForecastConfidence.LOW
                )

            elif any(
                horizon.confidence
                == ForecastConfidence.MODERATE
                for horizon in available_horizons
            ):
                overall_confidence = (
                    ForecastConfidence.MODERATE
                )

            else:
                overall_confidence = (
                    ForecastConfidence.HIGH
                )

        else:
            overall_probability = None
            overall_risk = FloodRiskLevel.UNDETERMINED
            overall_confidence = (
                ForecastConfidence.INSUFFICIENT
            )

        observation_findings = (
            self.observation_risk_analyzer.analyze(
                observation_quality
            )
        )

        recommendations = (
            self.recommendation_engine.generate(
                overall_risk=overall_risk,
                horizons=horizons,
                observation_quality=observation_quality,
            )
        )

        warnings = list(
            observation_quality.warnings
        )

        warnings.extend(
            prediction_warnings
        )

        warnings.extend(
            observation_findings
        )

        warnings.extend([
            "Flood predictions are advisory estimates",
            "Forecast confidence labels are heuristic",
            "Forecast probabilities require independent calibration",
            "Risk level does not establish actual flood extent",
            "Official forecasts and field observations "
            "must be consulted",
            "No public warning or emergency action was issued",
            "Human review is required",
        ])

        assessment = FloodPredictionAssessment(
            assessment_id=generate_id(),
            district=request.district,
            overall_risk=overall_risk,
            overall_probability=overall_probability,
            confidence=overall_confidence,
            horizon_forecasts=horizons,
            observation_quality=observation_quality,
            recommended_actions=recommendations,
            warnings=tuple(warnings),
            assessed_at=now,
        )

        duplicate = (
            self.deduplication_engine.check(
                district=request.district,
                prediction_ids=[
                    item.prediction_id
                    for item in accepted_predictions
                ],
                now=now,
            )
        )

        report = FloodPredictionReport(
            report_id=generate_id(),
            request_id=request.request_id,
            assessment=assessment,
            status=ForecastStatus.PENDING_REVIEW,
            requires_human_review=True,
            duplicate_detected=duplicate,
            generated_at=now,
        )

        with self._lock:
            self._reports[
                report.report_id
            ] = report

        self.audit_logger.record(
            report_id=report.report_id,
            action="forecast_report_created",
            actor_id="flood_prediction_agent",
            details={
                "district": request.district,
                "overall_risk": overall_risk.value,
                "confidence": overall_confidence.value,
                "model_predictions": str(
                    len(accepted_predictions)
                ),
            },
        )

        return report

    def review_report(
        self,
        report_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        approved: bool,
        reason: str,
    ) -> FloodPredictionReport:
        with self._lock:
            report = self._reports.get(
                report_id
            )

            if report is None:
                raise FloodPredictionApprovalError(
                    "Report not found"
                )

            self.review_manager.review(
                report,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                approved=approved,
                reason=reason,
            )

            updated = replace(
                report,
                status=(
                    ForecastStatus.APPROVED_FOR_ANALYSIS
                    if approved
                    else ForecastStatus.REJECTED
                ),
            )

            self._reports[
                report_id
            ] = updated

            return updated

    def get_report(
        self,
        report_id: str,
    ) -> FloodPredictionReport | None:
        with self._lock:
            return self._reports.get(
                report_id
            )

    def get_audit_history(
        self,
        report_id: str,
    ) -> tuple[
        FloodPredictionAuditRecord,
        ...
    ]:
        return self.audit_logger.list_records(
            report_id
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class FloodPredictionReportExporter:
    @staticmethod
    def export_json(
        report: FloodPredictionReport,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise FloodPredictionExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                report.to_dict(),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_flood_prediction_agent(
    config: FloodPredictionAgentConfig | None = None,
) -> FloodPredictionAgent:
    return FloodPredictionAgent(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    now = utc_now()

    observations = (
        FloodObservation(
            observation_id="RAIN-001",
            district="Thrissur",
            station_id="RAIN-STATION-001",
            observation_type=ObservationType.RAINFALL,
            value=125.0,
            unit="mm",
            observed_at=now - timedelta(minutes=15),
            verified=True,
            reliability=0.95,
        ),
        FloodObservation(
            observation_id="RIVER-001",
            district="Thrissur",
            station_id="RIVER-STATION-001",
            observation_type=ObservationType.RIVER_RISE_RATE,
            value=0.45,
            unit="m/hour",
            observed_at=now - timedelta(minutes=10),
            verified=True,
            reliability=0.96,
        ),
    )

    # Illustrative predictions only.
    # Replace with outputs from trained,
    # validated and calibrated models.
    predictions = (
        FloodModelPrediction(
            prediction_id="PRED-LSTM-001",
            model_id="flood-lstm",
            model_version="1.0",
            model_type=ForecastModelType.LSTM,
            district="Thrissur",
            forecast_horizon_hours=6,
            flood_probability=0.78,
            predicted_river_level_m=8.5,
            predicted_rainfall_mm=65.0,
            model_reliability=0.85,
            generated_at=now,
            calibrated=True,
            validated=True,
        ),
        FloodModelPrediction(
            prediction_id="PRED-TFT-001",
            model_id="flood-tft",
            model_version="1.0",
            model_type=ForecastModelType.TEMPORAL_FUSION,
            district="Thrissur",
            forecast_horizon_hours=6,
            flood_probability=0.82,
            predicted_river_level_m=8.7,
            predicted_rainfall_mm=70.0,
            model_reliability=0.88,
            generated_at=now,
            calibrated=True,
            validated=True,
        ),
    )

    request = FloodPredictionRequest(
        request_id=generate_id(),
        district="Thrissur",
        observations=observations,
        model_predictions=predictions,
        requested_at=now,
    )

    agent = create_flood_prediction_agent()

    report = agent.run(
        request,
        now=now,
    )

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    FloodPredictionReportExporter.export_json(
        report,
        "outputs/agents/flood_prediction_report.json",
    )

    # Human review only.
    # Does not issue a flood alert.
    reviewed = agent.review_report(
        report.report_id,
        reviewer_id="hydrologist-001",
        reviewer_role="hydrology_specialist",
        approved=False,
        reason=(
            "Awaiting independent forecast "
            "and station verification"
        ),
    )

    print(
        "Review status:",
        reviewed.status.value,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FloodPredictionAgentError",
    "FloodPredictionValidationError",
    "FloodPredictionModelError",
    "FloodPredictionApprovalError",
    "FloodPredictionExportError",
    "FloodRiskLevel",
    "ForecastModelType",
    "ForecastStatus",
    "ForecastConfidence",
    "ObservationType",
    "FloodPredictionAgentConfig",
    "FloodObservation",
    "FloodModelPrediction",
    "FloodPredictionRequest",
    "ObservationQualityResult",
    "HorizonForecast",
    "FloodPredictionAssessment",
    "FloodPredictionReport",
    "FloodPredictionReview",
    "FloodPredictionAuditRecord",
    "FloodForecastModelAdapter",
    "FloodModelAdapterRegistry",
    "FloodObservationQualityEngine",
    "FloodModelPredictionValidator",
    "FloodForecastFusionEngine",
    "FloodObservationRiskAnalyzer",
    "FloodPredictionRecommendationEngine",
    "FloodPredictionDeduplicationEngine",
    "FloodPredictionAuditLogger",
    "FloodPredictionReviewManager",
    "FloodPredictionAgent",
    "FloodPredictionReportExporter",
    "create_flood_prediction_agent",
]
