from __future__ import annotations

"""
backend/services/forecast_service.py

Kerala Flood Relief Coordination Agent

Flood Forecasting Service

Features:
- Rainfall forecasting
- River water-level forecasting
- Flood probability prediction
- Flood severity classification
- District-level risk forecasting
- Multi-model forecast integration
- Forecast confidence and uncertainty
- Forecast horizon management
- Historical observation validation
- Model evaluation metrics
- Forecast persistence
- Model version tracking
- Human review workflow
- Forecast reporting
- Audit logging

SAFETY:
Predictions are decision-support information,
not official flood warnings.

Do not use unvalidated predictions to trigger
evacuations, dam operations, or emergency dispatch.

Operational forecasts require calibrated models,
verified data sources, monitoring, and human review.
"""


import asyncio
import logging
import math
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from statistics import mean
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class ForecastType(str, Enum):
    RAINFALL = "rainfall"
    RIVER_LEVEL = "river_level"
    FLOOD_PROBABILITY = "flood_probability"
    FLOOD_SEVERITY = "flood_severity"
    DISTRICT_RISK = "district_risk"


class ForecastStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ForecastRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class ForecastSource(str, Enum):
    ML_MODEL = "ml_model"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    HYDROLOGICAL_MODEL = "hydrological_model"
    OFFICIAL = "official"


# ============================================================
# EXCEPTIONS
# ============================================================

class ForecastServiceError(Exception):
    pass


class ForecastValidationError(ForecastServiceError):
    pass


class ForecastNotFoundError(ForecastServiceError):
    pass


class ForecastConflictError(ForecastServiceError):
    pass


class ForecastAuthorizationError(ForecastServiceError):
    pass


class ForecastModelError(ForecastServiceError):
    pass


class ForecastDataError(ForecastServiceError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ForecastValidationError(
            f"{name} must be timezone-aware"
        )


def validate_probability(
    value: float,
    name: str = "probability",
) -> float:
    if (
        not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ForecastValidationError(
            f"{name} must be between 0 and 1"
        )

    return value


def validate_finite(
    value: float,
    name: str,
) -> float:
    if not math.isfinite(value):
        raise ForecastValidationError(
            f"{name} must be finite"
        )

    return value


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class WeatherObservation:
    district_id: int
    station_id: str
    observed_at: datetime

    rainfall_mm: float
    temperature_c: float | None = None
    humidity_percent: float | None = None
    wind_speed_kmh: float | None = None

    source: str = ""
    quality_verified: bool = False

    def __post_init__(self) -> None:
        validate_datetime(
            self.observed_at,
            "observed_at",
        )

        validate_finite(
            self.rainfall_mm,
            "rainfall_mm",
        )

        if self.rainfall_mm < 0:
            raise ForecastValidationError(
                "Rainfall cannot be negative"
            )

        if self.humidity_percent is not None:
            if not 0 <= self.humidity_percent <= 100:
                raise ForecastValidationError(
                    "Humidity must be between 0 and 100"
                )


@dataclass(frozen=True)
class RiverObservation:
    district_id: int
    station_id: str
    observed_at: datetime

    water_level_m: float
    warning_level_m: float | None = None
    danger_level_m: float | None = None

    flow_rate_m3s: float | None = None

    source: str = ""
    quality_verified: bool = False

    def __post_init__(self) -> None:
        validate_datetime(
            self.observed_at,
            "observed_at",
        )

        validate_finite(
            self.water_level_m,
            "water_level_m",
        )

        if self.flow_rate_m3s is not None:
            validate_finite(
                self.flow_rate_m3s,
                "flow_rate_m3s",
            )

            if self.flow_rate_m3s < 0:
                raise ForecastValidationError(
                    "Flow rate cannot be negative"
                )


@dataclass(frozen=True)
class ForecastRequest:
    district_id: int
    forecast_type: ForecastType

    horizon_hours: int = 24

    station_id: str | None = None
    model_name: str | None = None

    requested_by: int | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ForecastPoint:
    timestamp: datetime
    predicted_value: float

    lower_bound: float | None = None
    upper_bound: float | None = None

    probability: float | None = None

    def __post_init__(self) -> None:
        validate_datetime(
            self.timestamp,
            "timestamp",
        )

        validate_finite(
            self.predicted_value,
            "predicted_value",
        )

        if self.lower_bound is not None:
            validate_finite(
                self.lower_bound,
                "lower_bound",
            )

        if self.upper_bound is not None:
            validate_finite(
                self.upper_bound,
                "upper_bound",
            )

        if (
            self.lower_bound is not None
            and self.upper_bound is not None
        ):
            if self.lower_bound > self.upper_bound:
                raise ForecastValidationError(
                    "Lower bound exceeds upper bound"
                )

            if not (
                self.lower_bound
                <= self.predicted_value
                <= self.upper_bound
            ):
                raise ForecastValidationError(
                    "Prediction lies outside interval"
                )

        if self.probability is not None:
            validate_probability(
                self.probability,
            )


@dataclass(frozen=True)
class ModelForecastResult:
    model_name: str
    model_version: str

    forecast_type: ForecastType
    source: ForecastSource

    points: tuple[ForecastPoint, ...]

    confidence: float | None = None
    uncertainty: float | None = None

    metrics: Mapping[str, float] = field(
        default_factory=dict
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class FloodForecast:
    id: str

    district_id: int
    forecast_type: ForecastType

    model_name: str
    model_version: str
    source: ForecastSource

    horizon_hours: int

    points: list[ForecastPoint]

    risk_level: ForecastRiskLevel

    confidence: float | None
    uncertainty: float | None

    status: ForecastStatus = ForecastStatus.DRAFT

    station_id: str | None = None

    created_by: int | None = None
    reviewed_by: int | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None
    expires_at: datetime | None = None

    review_notes: str | None = None

    version: int = 1

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ForecastEvaluation:
    sample_count: int
    mae: float
    mse: float
    rmse: float
    bias: float
    mape: float | None = None


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class ForecastRepository(Protocol):

    async def create(
        self,
        forecast: FloodForecast,
    ) -> FloodForecast:
        ...

    async def get_by_id(
        self,
        forecast_id: str,
    ) -> FloodForecast | None:
        ...

    async def update(
        self,
        forecast: FloodForecast,
        *,
        expected_version: int,
    ) -> FloodForecast:
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[FloodForecast]:
        ...

    async def list_active(
        self,
    ) -> Sequence[FloodForecast]:
        ...


class WeatherDataGateway(Protocol):

    async def get_observations(
        self,
        *,
        district_id: int,
        station_id: str | None,
        hours: int,
    ) -> Sequence[WeatherObservation]:
        ...


class RiverDataGateway(Protocol):

    async def get_observations(
        self,
        *,
        district_id: int,
        station_id: str | None,
        hours: int,
    ) -> Sequence[RiverObservation]:
        ...


class ForecastModelGateway(Protocol):

    async def predict(
        self,
        *,
        request: ForecastRequest,
        weather: Sequence[WeatherObservation],
        rivers: Sequence[RiverObservation],
    ) -> ModelForecastResult:
        ...


class ForecastAuthorizationGateway(Protocol):

    async def can_generate(
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


class ForecastAuditGateway(Protocol):

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
# RISK CLASSIFICATION
# ============================================================

class FloodRiskClassifier:
    """
    Classify calibrated flood probabilities.

    Risk bands are illustrative, not official
    flood warning or evacuation thresholds.

    Rainfall and river levels must not be treated
    as probabilities without a calibrated model.
    """

    @staticmethod
    def classify_probability(
        probability: float,
    ) -> ForecastRiskLevel:
        validate_probability(
            probability
        )

        if probability >= 0.90:
            return ForecastRiskLevel.EXTREME

        if probability >= 0.75:
            return ForecastRiskLevel.SEVERE

        if probability >= 0.55:
            return ForecastRiskLevel.HIGH

        if probability >= 0.30:
            return ForecastRiskLevel.MODERATE

        return ForecastRiskLevel.LOW

    def classify(
        self,
        forecast_type: ForecastType,
        points: Sequence[ForecastPoint],
    ) -> ForecastRiskLevel:

        if not points:
            return ForecastRiskLevel.UNKNOWN

        if forecast_type not in {
            ForecastType.FLOOD_PROBABILITY,
            ForecastType.DISTRICT_RISK,
        }:
            return ForecastRiskLevel.UNKNOWN

        probabilities = [
            point.probability
            if point.probability is not None
            else point.predicted_value
            for point in points
        ]

        for probability in probabilities:
            validate_probability(
                probability
            )

        return self.classify_probability(
            max(probabilities)
        )


# ============================================================
# FORECAST VALIDATION
# ============================================================

class ForecastValidator:

    MAX_HORIZON_HOURS = 168

    def validate_request(
        self,
        request: ForecastRequest,
    ) -> None:

        if request.district_id <= 0:
            raise ForecastValidationError(
                "Invalid district ID"
            )

        if not (
            1 <= request.horizon_hours
            <= self.MAX_HORIZON_HOURS
        ):
            raise ForecastValidationError(
                "Forecast horizon must be "
                "between 1 and 168 hours"
            )

        if (
            request.station_id is not None
            and not request.station_id.strip()
        ):
            raise ForecastValidationError(
                "Station ID cannot be empty"
            )

    def validate_observations(
        self,
        weather: Sequence[WeatherObservation],
        rivers: Sequence[RiverObservation],
        *,
        district_id: int,
        now: datetime,
    ) -> None:

        all_observations = [
            *weather,
            *rivers,
        ]

        if not all_observations:
            raise ForecastDataError(
                "No observations available"
            )

        for observation in all_observations:
            if observation.district_id != district_id:
                raise ForecastDataError(
                    "Observation district mismatch"
                )

            if not observation.quality_verified:
                raise ForecastDataError(
                    "Unverified observation received"
                )

            if observation.observed_at > now:
                raise ForecastDataError(
                    "Future-dated observation received"
                )

        latest = max(
            observation.observed_at
            for observation in all_observations
        )

        if now - latest > timedelta(hours=3):
            raise ForecastDataError(
                "Latest observation is too old"
            )

    def validate_model_result(
        self,
        result: ModelForecastResult,
        request: ForecastRequest,
        *,
        issued_at: datetime,
    ) -> None:

        if not result.model_name.strip():
            raise ForecastModelError(
                "Model name is required"
            )

        if not result.model_version.strip():
            raise ForecastModelError(
                "Model version is required"
            )

        if result.forecast_type != request.forecast_type:
            raise ForecastModelError(
                "Model returned wrong forecast type"
            )

        if not result.points:
            raise ForecastModelError(
                "Model returned no forecast points"
            )

        if result.confidence is not None:
            validate_probability(
                result.confidence,
                "confidence",
            )

        if result.uncertainty is not None:
            if (
                not math.isfinite(result.uncertainty)
                or result.uncertainty < 0
            ):
                raise ForecastModelError(
                    "Invalid uncertainty"
                )

        timestamps = [
            point.timestamp
            for point in result.points
        ]

        if timestamps != sorted(set(timestamps)):
            raise ForecastModelError(
                "Forecast timestamps must be "
                "unique and ascending"
            )

        max_time = issued_at + timedelta(
            hours=request.horizon_hours
        )

        for point in result.points:
            if not issued_at < point.timestamp <= max_time:
                raise ForecastModelError(
                    "Forecast point outside horizon"
                )

            if request.forecast_type == ForecastType.RAINFALL:
                if point.predicted_value < 0:
                    raise ForecastModelError(
                        "Rainfall forecast cannot be negative"
                    )

            if request.forecast_type in {
                ForecastType.FLOOD_PROBABILITY,
                ForecastType.DISTRICT_RISK,
            }:
                validate_probability(
                    point.predicted_value,
                    "predicted probability",
                )

            if point.probability is not None:
                validate_probability(
                    point.probability,
                )

        for metric_name, value in result.metrics.items():
            validate_finite(
                value,
                metric_name,
            )


# ============================================================
# FORECAST EVALUATION
# ============================================================

class ForecastEvaluator:

    @staticmethod
    def evaluate(
        actual: Sequence[float],
        predicted: Sequence[float],
    ) -> ForecastEvaluation:

        if len(actual) != len(predicted):
            raise ForecastValidationError(
                "Actual and predicted lengths differ"
            )

        if not actual:
            raise ForecastValidationError(
                "Evaluation data cannot be empty"
            )

        for value in [*actual, *predicted]:
            validate_finite(
                value,
                "evaluation value",
            )

        errors = [
            prediction - observation
            for observation, prediction
            in zip(actual, predicted)
        ]

        absolute_errors = [
            abs(error)
            for error in errors
        ]

        squared_errors = [
            error ** 2
            for error in errors
        ]

        mae = mean(absolute_errors)
        mse = mean(squared_errors)
        rmse = math.sqrt(mse)
        bias = mean(errors)

        percentage_errors = [
            abs(
                (prediction - observation)
                / observation
            ) * 100
            for observation, prediction
            in zip(actual, predicted)
            if observation != 0
        ]

        mape = (
            mean(percentage_errors)
            if percentage_errors
            else None
        )

        return ForecastEvaluation(
            sample_count=len(actual),
            mae=round(mae, 6),
            mse=round(mse, 6),
            rmse=round(rmse, 6),
            bias=round(bias, 6),
            mape=(
                round(mape, 6)
                if mape is not None
                else None
            ),
        )


# ============================================================
# FORECAST SERVICE
# ============================================================

class ForecastService:

    def __init__(
        self,
        *,
        repository: ForecastRepository,
        weather_gateway: WeatherDataGateway,
        river_gateway: RiverDataGateway,
        model_gateway: ForecastModelGateway,
        authorization_gateway: ForecastAuthorizationGateway,
        audit_gateway: ForecastAuditGateway | None = None,
        validator: ForecastValidator | None = None,
        classifier: FloodRiskClassifier | None = None,
        evaluator: ForecastEvaluator | None = None,
    ):
        self.repository = repository
        self.weather = weather_gateway
        self.rivers = river_gateway
        self.models = model_gateway
        self.authorization = authorization_gateway
        self.audit = audit_gateway

        self.validator = (
            validator or ForecastValidator()
        )

        self.classifier = (
            classifier or FloodRiskClassifier()
        )

        self.evaluator = (
            evaluator or ForecastEvaluator()
        )

    async def _audit(
        self,
        action: str,
        forecast: FloodForecast,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="flood_forecast",
            entity_id=forecast.id,
            metadata={
                "district_id": forecast.district_id,
                "forecast_type": (
                    forecast.forecast_type.value
                ),
                "model_name": forecast.model_name,
                "model_version": forecast.model_version,
                "status": forecast.status.value,
                "risk_level": forecast.risk_level.value,
            },
        )

    async def get_forecast(
        self,
        forecast_id: str,
    ) -> FloodForecast:

        forecast = await self.repository.get_by_id(
            forecast_id
        )

        if forecast is None:
            raise ForecastNotFoundError(
                "Forecast not found"
            )

        return forecast

    async def generate_forecast(
        self,
        request: ForecastRequest,
        *,
        actor_id: int,
    ) -> FloodForecast:

        self.validator.validate_request(
            request
        )

        authorized = (
            await self.authorization.can_generate(
                user_id=actor_id,
                district_id=request.district_id,
            )
        )

        if not authorized:
            raise ForecastAuthorizationError(
                "Forecast generation permission denied"
            )

        issued_at = utc_now()

        weather, rivers = await asyncio.gather(
            self.weather.get_observations(
                district_id=request.district_id,
                station_id=request.station_id,
                hours=72,
            ),
            self.rivers.get_observations(
                district_id=request.district_id,
                station_id=request.station_id,
                hours=72,
            ),
        )

        self.validator.validate_observations(
            weather,
            rivers,
            district_id=request.district_id,
            now=issued_at,
        )

        try:
            model_result = await self.models.predict(
                request=request,
                weather=weather,
                rivers=rivers,
            )

        except Exception as exc:
            logger.exception(
                "Forecast model failed for district %s",
                request.district_id,
            )

            raise ForecastModelError(
                "Forecast model execution failed"
            ) from exc

        self.validator.validate_model_result(
            model_result,
            request,
            issued_at=issued_at,
        )

        risk_level = self.classifier.classify(
            request.forecast_type,
            model_result.points,
        )

        forecast = FloodForecast(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            forecast_type=request.forecast_type,
            model_name=model_result.model_name,
            model_version=model_result.model_version,
            source=model_result.source,
            horizon_hours=request.horizon_hours,
            points=list(model_result.points),
            risk_level=risk_level,
            confidence=model_result.confidence,
            uncertainty=model_result.uncertainty,
            status=ForecastStatus.DRAFT,
            station_id=request.station_id,
            created_by=actor_id,
            created_at=issued_at,
            updated_at=issued_at,
            expires_at=issued_at + timedelta(
                hours=request.horizon_hours
            ),
            metrics=dict(model_result.metrics),
            metadata={
                **dict(request.metadata),
                "model_metadata": dict(
                    model_result.metadata
                ),
                "weather_observation_count": len(
                    weather
                ),
                "river_observation_count": len(
                    rivers
                ),
                "issued_at": issued_at.isoformat(),
            },
        )

        created = await self.repository.create(
            forecast
        )

        await self._audit(
            "forecast.generated",
            created,
            actor_id,
        )

        logger.info(
            "Forecast generated: %s",
            created.id,
        )

        return created

    async def submit_for_review(
        self,
        forecast_id: str,
        *,
        actor_id: int,
    ) -> FloodForecast:

        forecast = await self.get_forecast(
            forecast_id
        )

        if not await self.authorization.can_generate(
            user_id=actor_id,
            district_id=forecast.district_id,
        ):
            raise ForecastAuthorizationError(
                "Forecast submission permission denied"
            )

        if forecast.status != ForecastStatus.DRAFT:
            raise ForecastConflictError(
                "Only draft forecasts can be submitted"
            )

        if (
            forecast.expires_at is not None
            and forecast.expires_at <= utc_now()
        ):
            raise ForecastConflictError(
                "Expired forecast cannot be submitted"
            )

        updated = replace(
            forecast,
            status=ForecastStatus.PENDING_REVIEW,
            updated_at=utc_now(),
            version=forecast.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=forecast.version,
        )

        await self._audit(
            "forecast.submitted",
            saved,
            actor_id,
        )

        return saved

    async def verify_forecast(
        self,
        forecast_id: str,
        *,
        reviewer_id: int,
        review_notes: str,
    ) -> FloodForecast:

        forecast = await self.get_forecast(
            forecast_id
        )

        if not await self.authorization.can_review(
            user_id=reviewer_id,
            district_id=forecast.district_id,
        ):
            raise ForecastAuthorizationError(
                "Forecast review permission denied"
            )

        if forecast.status != ForecastStatus.PENDING_REVIEW:
            raise ForecastConflictError(
                "Forecast is not awaiting review"
            )

        if forecast.created_by == reviewer_id:
            raise ForecastAuthorizationError(
                "Independent reviewer required"
            )

        if not review_notes.strip():
            raise ForecastValidationError(
                "Review notes are required"
            )

        if (
            forecast.expires_at is not None
            and forecast.expires_at <= utc_now()
        ):
            raise ForecastConflictError(
                "Expired forecast cannot be verified"
            )

        updated = replace(
            forecast,
            status=ForecastStatus.VERIFIED,
            reviewed_by=reviewer_id,
            reviewed_at=utc_now(),
            review_notes=review_notes.strip(),
            updated_at=utc_now(),
            version=forecast.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=forecast.version,
        )

        await self._audit(
            "forecast.verified",
            saved,
            reviewer_id,
        )

        return saved

    async def reject_forecast(
        self,
        forecast_id: str,
        *,
        reviewer_id: int,
        reason: str,
    ) -> FloodForecast:

        forecast = await self.get_forecast(
            forecast_id
        )

        if not await self.authorization.can_review(
            user_id=reviewer_id,
            district_id=forecast.district_id,
        ):
            raise ForecastAuthorizationError(
                "Forecast review permission denied"
            )

        if forecast.status != ForecastStatus.PENDING_REVIEW:
            raise ForecastConflictError(
                "Forecast is not awaiting review"
            )

        if not reason.strip():
            raise ForecastValidationError(
                "Rejection reason is required"
            )

        updated = replace(
            forecast,
            status=ForecastStatus.REJECTED,
            reviewed_by=reviewer_id,
            reviewed_at=utc_now(),
            review_notes=reason.strip(),
            updated_at=utc_now(),
            version=forecast.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=forecast.version,
        )

        await self._audit(
            "forecast.rejected",
            saved,
            reviewer_id,
        )

        return saved

    async def expire_forecasts(
        self,
    ) -> int:

        active_forecasts = (
            await self.repository.list_active()
        )

        now = utc_now()
        expired_count = 0

        for forecast in active_forecasts:
            if forecast.status in {
                ForecastStatus.REJECTED,
                ForecastStatus.EXPIRED,
            }:
                continue

            if (
                forecast.expires_at is not None
                and forecast.expires_at <= now
            ):
                updated = replace(
                    forecast,
                    status=ForecastStatus.EXPIRED,
                    updated_at=now,
                    version=forecast.version + 1,
                )

                saved = await self.repository.update(
                    updated,
                    expected_version=forecast.version,
                )

                await self._audit(
                    "forecast.expired",
                    saved,
                    None,
                )

                expired_count += 1

        return expired_count

    async def list_district_forecasts(
        self,
        district_id: int,
    ) -> Sequence[FloodForecast]:

        if district_id <= 0:
            raise ForecastValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def get_latest_verified_forecast(
        self,
        district_id: int,
        forecast_type: ForecastType,
    ) -> FloodForecast | None:

        forecasts = await self.list_district_forecasts(
            district_id
        )

        now = utc_now()

        matching = [
            forecast
            for forecast in forecasts
            if (
                forecast.forecast_type == forecast_type
                and forecast.status
                == ForecastStatus.VERIFIED
                and (
                    forecast.expires_at is None
                    or forecast.expires_at > now
                )
            )
        ]

        if not matching:
            return None

        return max(
            matching,
            key=lambda item: item.created_at,
        )

    async def evaluate_predictions(
        self,
        *,
        actual: Sequence[float],
        predicted: Sequence[float],
    ) -> ForecastEvaluation:

        return self.evaluator.evaluate(
            actual,
            predicted,
        )

    async def get_district_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        forecasts = await self.list_district_forecasts(
            district_id
        )

        status_counts = {
            status.value: 0
            for status in ForecastStatus
        }

        type_counts = {
            forecast_type.value: 0
            for forecast_type in ForecastType
        }

        risk_counts = {
            risk.value: 0
            for risk in ForecastRiskLevel
        }

        for forecast in forecasts:
            status_counts[
                forecast.status.value
            ] += 1

            type_counts[
                forecast.forecast_type.value
            ] += 1

            risk_counts[
                forecast.risk_level.value
            ] += 1

        return {
            "district_id": district_id,
            "total_forecasts": len(forecasts),
            "status_counts": status_counts,
            "forecast_type_counts": type_counts,
            "risk_level_counts": risk_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_forecast_report(
        self,
        forecast_id: str,
    ) -> dict[str, Any]:

        forecast = await self.get_forecast(
            forecast_id
        )

        return {
            "forecast_id": forecast.id,
            "district_id": forecast.district_id,
            "forecast_type": forecast.forecast_type.value,
            "model": {
                "name": forecast.model_name,
                "version": forecast.model_version,
                "source": forecast.source.value,
            },
            "horizon_hours": forecast.horizon_hours,
            "risk_level": forecast.risk_level.value,
            "confidence": forecast.confidence,
            "uncertainty": forecast.uncertainty,
            "status": forecast.status.value,
            "forecast_points": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "predicted_value": point.predicted_value,
                    "lower_bound": point.lower_bound,
                    "upper_bound": point.upper_bound,
                    "probability": point.probability,
                }
                for point in forecast.points
            ],
            "metrics": forecast.metrics,
            "verification": {
                "reviewed_by": forecast.reviewed_by,
                "reviewed_at": (
                    forecast.reviewed_at.isoformat()
                    if forecast.reviewed_at
                    else None
                ),
                "review_notes": forecast.review_notes,
            },
            "created_at": forecast.created_at.isoformat(),
            "expires_at": (
                forecast.expires_at.isoformat()
                if forecast.expires_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Forecasts are model-based estimates, "
                "not official disaster warnings. "
                "Operational decisions require verified "
                "observations, calibrated models, "
                "and authorized human review."
            ),
        }
