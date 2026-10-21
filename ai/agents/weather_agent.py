"""Placeholder module: ai/agents/weather_agent.py; implementation required."""

"""
backend/ai/agents/weather_agent.py

Kerala Flood Relief Coordination Agent
Weather Intelligence & Rainfall Monitoring Agent

Python 3.11+
Standard library only.

CAPABILITIES
------------
1. Weather observation validation
2. Multi-source weather intelligence
3. Rainfall intensity classification
4. 24-hour rainfall accumulation
5. Forecast rainfall analysis
6. Wind and thunderstorm screening
7. Weather source reliability checks
8. Data freshness assessment
9. Conflicting observation detection
10. Rainfall trend analysis
11. Weather-related flood concern screening
12. Human review and audit logging
13. JSON report export
14. Supervisor integration adapter

SAFETY
------
This module provides advisory analysis only.

It does not:
- Issue official IMD warnings
- Declare flood emergencies
- Order evacuations
- Dispatch emergency services
- Control dams or infrastructure
- Send public alerts

Thresholds are illustrative configuration
values, not operational warning standards.

Weather observations and forecasts must come
from authenticated external integrations
before operational consideration.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import uuid

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class WeatherAgentError(Exception):
    pass


class WeatherValidationError(WeatherAgentError):
    pass


class WeatherEvidenceError(WeatherAgentError):
    pass


class WeatherApprovalError(WeatherAgentError):
    pass


class WeatherExportError(WeatherAgentError):
    pass


# ============================================================
# ENUMS
# ============================================================

class WeatherSourceType(str, Enum):
    WEATHER_STATION = "weather_station"
    RAIN_GAUGE = "rain_gauge"
    RADAR = "radar"
    SATELLITE = "satellite"
    FORECAST_MODEL = "forecast_model"
    AUTHORIZED_FIELD_REPORT = "authorized_field_report"


class SourceVerification(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class RainfallIntensity(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    VERY_HEAVY = "very_heavy"
    EXTREME = "extreme"
    UNDETERMINED = "undetermined"


class WeatherConcernLevel(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    SEVERE = "severe"
    UNDETERMINED = "undetermined"


class WeatherConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    INSUFFICIENT = "insufficient"


class WeatherReportStatus(str, Enum):
    PENDING_HUMAN_REVIEW = "pending_human_review"
    INSUFFICIENT_DATA = "insufficient_data"
    CONFLICTED_DATA = "conflicted_data"
    CONFIRMED_FOR_ANALYSIS = "confirmed_for_analysis"
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"


class ReviewDecision(str, Enum):
    CONFIRM_FOR_ANALYSIS = "confirm_for_analysis"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise WeatherValidationError(
            f"{name} must be timezone-aware"
        )


def validate_nonnegative(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise WeatherValidationError(
            f"{name} must be finite and nonnegative"
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    validate_nonnegative(value, name)

    if value > 1:
        raise WeatherValidationError(
            f"{name} must be between 0 and 1"
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

    if isinstance(value, (list, tuple)):
        return [
            serialize(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise WeatherValidationError(
                "Non-finite report value"
            )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class WeatherAgentConfig:
    max_observations: int = 10000
    max_forecasts: int = 5000

    observation_max_age_minutes: int = 120
    forecast_max_age_minutes: int = 180

    minimum_source_reliability: float = 0.70
    minimum_independent_sources: int = 2

    # Illustrative hourly rainfall bands.
    # Not official IMD classifications.
    light_rain_mm_h: float = 0.1
    moderate_rain_mm_h: float = 2.5
    heavy_rain_mm_h: float = 7.5
    very_heavy_rain_mm_h: float = 15.0
    extreme_rain_mm_h: float = 30.0

    # Illustrative 24-hour totals.
    elevated_24h_mm: float = 50.0
    high_24h_mm: float = 115.0
    severe_24h_mm: float = 200.0

    # Illustrative wind screening.
    strong_wind_kmh: float = 40.0
    severe_wind_kmh: float = 70.0

    # Relative disagreement threshold for
    # same-window observation comparison.
    disagreement_fraction: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "max_observations",
            "max_forecasts",
            "observation_max_age_minutes",
            "forecast_max_age_minutes",
            "minimum_independent_sources",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise WeatherValidationError(
                    f"{name} must be positive integer"
                )

        validate_probability(
            self.minimum_source_reliability,
            "minimum_source_reliability",
        )

        validate_probability(
            self.disagreement_fraction,
            "disagreement_fraction",
        )

        bands = (
            self.light_rain_mm_h,
            self.moderate_rain_mm_h,
            self.heavy_rain_mm_h,
            self.very_heavy_rain_mm_h,
            self.extreme_rain_mm_h,
        )

        if not (
            all(math.isfinite(x) and x >= 0 for x in bands)
            and list(bands) == sorted(set(bands))
        ):
            raise WeatherValidationError(
                "Rainfall thresholds must be strictly increasing"
            )

        totals = (
            self.elevated_24h_mm,
            self.high_24h_mm,
            self.severe_24h_mm,
        )

        if not (
            all(math.isfinite(x) and x >= 0 for x in totals)
            and totals[0] < totals[1] < totals[2]
        ):
            raise WeatherValidationError(
                "24-hour thresholds must increase"
            )

        if not (
            0 <= self.strong_wind_kmh
            < self.severe_wind_kmh
        ):
            raise WeatherValidationError(
                "Invalid wind thresholds"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class WeatherObservation:
    observation_id: str
    source_id: str
    source_type: WeatherSourceType

    district: str
    locality: str

    observed_at: datetime

    rainfall_mm: float
    accumulation_minutes: int

    wind_speed_kmh: float | None = None
    temperature_c: float | None = None
    humidity_percent: float | None = None

    thunderstorm_detected: bool = False

    verification: SourceVerification = (
        SourceVerification.UNVERIFIED
    )

    reliability: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "source_id",
            "district",
            "locality",
        ):
            if not getattr(self, name).strip():
                raise WeatherValidationError(
                    f"{name} required"
                )

        validate_datetime(
            self.observed_at,
            "observed_at",
        )

        validate_nonnegative(
            self.rainfall_mm,
            "rainfall_mm",
        )

        if (
            isinstance(self.accumulation_minutes, bool)
            or not isinstance(self.accumulation_minutes, int)
            or self.accumulation_minutes <= 0
        ):
            raise WeatherValidationError(
                "accumulation_minutes must be positive"
            )

        if self.wind_speed_kmh is not None:
            validate_nonnegative(
                self.wind_speed_kmh,
                "wind_speed_kmh",
            )

        if self.temperature_c is not None:
            if not math.isfinite(
                self.temperature_c
            ):
                raise WeatherValidationError(
                    "temperature_c must be finite"
                )

        if self.humidity_percent is not None:
            if not (
                0 <= self.humidity_percent <= 100
            ):
                raise WeatherValidationError(
                    "Invalid humidity percentage"
                )

        validate_probability(
            self.reliability,
            "reliability",
        )


@dataclass(frozen=True)
class WeatherForecast:
    forecast_id: str
    source_id: str

    district: str
    locality: str

    issued_at: datetime
    valid_from: datetime
    valid_until: datetime

    predicted_rainfall_mm: float
    precipitation_probability: float

    predicted_max_wind_kmh: float | None = None

    thunderstorm_probability: float | None = None

    verification: SourceVerification = (
        SourceVerification.UNVERIFIED
    )

    reliability: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "forecast_id",
            "source_id",
            "district",
            "locality",
        ):
            if not getattr(self, name).strip():
                raise WeatherValidationError(
                    f"{name} required"
                )

        for name in (
            "issued_at",
            "valid_from",
            "valid_until",
        ):
            validate_datetime(
                getattr(self, name),
                name,
            )

        if self.valid_until <= self.valid_from:
            raise WeatherValidationError(
                "Invalid forecast interval"
            )

        validate_nonnegative(
            self.predicted_rainfall_mm,
            "predicted_rainfall_mm",
        )

        validate_probability(
            self.precipitation_probability,
            "precipitation_probability",
        )

        validate_probability(
            self.reliability,
            "reliability",
        )

        if self.predicted_max_wind_kmh is not None:
            validate_nonnegative(
                self.predicted_max_wind_kmh,
                "predicted_max_wind_kmh",
            )

        if self.thunderstorm_probability is not None:
            validate_probability(
                self.thunderstorm_probability,
                "thunderstorm_probability",
            )


@dataclass(frozen=True)
class WeatherAnalysisRequest:
    request_id: str

    district: str
    locality: str

    observations: tuple[
        WeatherObservation,
        ...
    ]

    forecasts: tuple[
        WeatherForecast,
        ...
    ]

    created_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "district",
            "locality",
        ):
            if not getattr(self, name).strip():
                raise WeatherValidationError(
                    f"{name} required"
                )

        validate_datetime(
            self.created_at,
            "created_at",
        )


@dataclass(frozen=True)
class WeatherEvidenceQuality:
    accepted_observation_ids: tuple[str, ...]
    rejected_observation_ids: tuple[str, ...]

    accepted_forecast_ids: tuple[str, ...]
    rejected_forecast_ids: tuple[str, ...]

    independent_source_count: int
    sufficient_sources: bool

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RainfallAnalysis:
    latest_intensity: RainfallIntensity

    latest_rate_mm_h: float | None

    observed_24h_mm: float | None

    maximum_observed_rate_mm_h: float | None

    trend: str

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ForecastAnalysis:
    maximum_forecast_rainfall_mm: float | None

    maximum_forecast_wind_kmh: float | None

    maximum_thunderstorm_probability: float | None

    forecast_concern: WeatherConcernLevel

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class WeatherAgentReport:
    report_id: str
    request_id: str

    district: str
    locality: str

    evidence_quality: WeatherEvidenceQuality
    rainfall: RainfallAnalysis
    forecast: ForecastAnalysis

    concern_level: WeatherConcernLevel
    confidence: WeatherConfidence

    recommendations: tuple[str, ...]

    status: WeatherReportStatus
    requires_human_review: bool

    warnings: tuple[str, ...]

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class WeatherReviewRecord:
    review_id: str
    report_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ReviewDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class WeatherAuditRecord:
    audit_id: str
    report_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# EVIDENCE VALIDATION ENGINE
# ============================================================

class WeatherEvidenceValidator:
    def __init__(
        self,
        config: WeatherAgentConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def _location_matches(
        district: str,
        locality: str,
        expected_district: str,
        expected_locality: str,
    ) -> bool:
        return (
            district.casefold() == expected_district.casefold()
            and locality.casefold() == expected_locality.casefold()
        )

    def validate(
        self,
        request: WeatherAnalysisRequest,
        *,
        now: datetime,
    ) -> tuple[
        tuple[WeatherObservation, ...],
        tuple[WeatherForecast, ...],
        WeatherEvidenceQuality,
    ]:
        if len(request.observations) > (
            self.config.max_observations
        ):
            raise WeatherEvidenceError(
                "Observation limit exceeded"
            )

        if len(request.forecasts) > (
            self.config.max_forecasts
        ):
            raise WeatherEvidenceError(
                "Forecast limit exceeded"
            )

        observation_ids = [
            item.observation_id
            for item in request.observations
        ]

        forecast_ids = [
            item.forecast_id
            for item in request.forecasts
        ]

        if len(observation_ids) != len(set(observation_ids)):
            raise WeatherEvidenceError(
                "Duplicate observation ID"
            )

        if len(forecast_ids) != len(set(forecast_ids)):
            raise WeatherEvidenceError(
                "Duplicate forecast ID"
            )

        accepted_observations = []
        rejected_observations = []

        accepted_forecasts = []
        rejected_forecasts = []

        source_ids = set()

        for item in request.observations:
            valid = (
                item.verification
                == SourceVerification.VERIFIED
                and item.reliability
                >= self.config.minimum_source_reliability
                and item.observed_at <= now
                and (
                    now - item.observed_at
                ) <= timedelta(
                    minutes=self.config.observation_max_age_minutes
                )
                and self._location_matches(
                    item.district,
                    item.locality,
                    request.district,
                    request.locality,
                )
            )

            if valid:
                accepted_observations.append(
                    item
                )

                source_ids.add(
                    item.source_id
                )

            else:
                rejected_observations.append(
                    item.observation_id
                )

        for item in request.forecasts:
            valid = (
                item.verification
                == SourceVerification.VERIFIED
                and item.reliability
                >= self.config.minimum_source_reliability
                and item.issued_at <= now
                and (
                    now - item.issued_at
                ) <= timedelta(
                    minutes=self.config.forecast_max_age_minutes
                )
                and item.valid_until > now
                and self._location_matches(
                    item.district,
                    item.locality,
                    request.district,
                    request.locality,
                )
            )

            if valid:
                accepted_forecasts.append(
                    item
                )

                source_ids.add(
                    item.source_id
                )

            else:
                rejected_forecasts.append(
                    item.forecast_id
                )

        sufficient = (
            len(source_ids)
            >= self.config.minimum_independent_sources
        )

        warnings = []

        if not sufficient:
            warnings.append(
                "Insufficient distinct source IDs"
            )

        if rejected_observations:
            warnings.append(
                "Some weather observations were rejected"
            )

        if rejected_forecasts:
            warnings.append(
                "Some weather forecasts were rejected"
            )

        warnings.append(
            "Distinct source IDs do not guarantee "
            "true source independence"
        )

        quality = WeatherEvidenceQuality(
            accepted_observation_ids=tuple(
                item.observation_id
                for item in accepted_observations
            ),
            rejected_observation_ids=tuple(
                rejected_observations
            ),
            accepted_forecast_ids=tuple(
                item.forecast_id
                for item in accepted_forecasts
            ),
            rejected_forecast_ids=tuple(
                rejected_forecasts
            ),
            independent_source_count=len(
                source_ids
            ),
            sufficient_sources=sufficient,
            warnings=tuple(warnings),
        )

        return (
            tuple(accepted_observations),
            tuple(accepted_forecasts),
            quality,
        )


# ============================================================
# RAINFALL INTENSITY ENGINE
# ============================================================

class RainfallIntensityEngine:
    def __init__(
        self,
        config: WeatherAgentConfig,
    ) -> None:
        self.config = config

    def classify(
        self,
        rate_mm_h: float | None,
    ) -> RainfallIntensity:
        if rate_mm_h is None:
            return RainfallIntensity.UNDETERMINED

        if rate_mm_h >= self.config.extreme_rain_mm_h:
            return RainfallIntensity.EXTREME

        if rate_mm_h >= self.config.very_heavy_rain_mm_h:
            return RainfallIntensity.VERY_HEAVY

        if rate_mm_h >= self.config.heavy_rain_mm_h:
            return RainfallIntensity.HEAVY

        if rate_mm_h >= self.config.moderate_rain_mm_h:
            return RainfallIntensity.MODERATE

        if rate_mm_h >= self.config.light_rain_mm_h:
            return RainfallIntensity.LIGHT

        return RainfallIntensity.NONE

    @staticmethod
    def hourly_rate(
        observation: WeatherObservation,
    ) -> float:
        return (
            observation.rainfall_mm
            * 60.0
            / observation.accumulation_minutes
        )


# ============================================================
# RAINFALL ANALYSIS ENGINE
# ============================================================

class RainfallAnalysisEngine:
    """
    Rainfall observations represent accumulated
    amounts over a stated preceding time window.

    The module does not sum overlapping windows
    from different sources.

    It estimates a 24-hour total only when a
    single source provides contiguous, non-
    overlapping observations covering 24 hours.
    """

    def __init__(
        self,
        config: WeatherAgentConfig,
    ) -> None:
        self.config = config

        self.intensity_engine = (
            RainfallIntensityEngine(
                config
            )
        )

    @staticmethod
    def _window_start(
        observation: WeatherObservation,
    ) -> datetime:
        return (
            observation.observed_at
            - timedelta(
                minutes=observation.accumulation_minutes
            )
        )

    def _complete_24h_total(
        self,
        observations: Sequence[WeatherObservation],
        *,
        now: datetime,
    ) -> float | None:
        window_start = now - timedelta(hours=24)

        grouped: dict[
            str,
            list[WeatherObservation],
        ] = {}

        for item in observations:
            grouped.setdefault(
                item.source_id,
                [],
            ).append(
                item
            )

        totals = []

        for source_items in grouped.values():
            relevant = sorted(
                source_items,
                key=lambda item: self._window_start(item),
            )

            cursor = window_start
            total = 0.0
            complete = True

            for item in relevant:
                start = self._window_start(item)
                end = item.observed_at

                if end <= cursor:
                    continue

                if start > cursor:
                    complete = False
                    break

                if start < cursor:
                    # Overlapping observation windows
                    # cannot be safely added.
                    complete = False
                    break

                if end > now:
                    complete = False
                    break

                total += item.rainfall_mm
                cursor = end

                if cursor == now:
                    break

            if complete and cursor == now:
                totals.append(total)

        if not totals:
            return None

        return round(
            statistics.median(totals),
            3,
        )

    def analyze(
        self,
        observations: Sequence[WeatherObservation],
        *,
        now: datetime,
    ) -> RainfallAnalysis:
        if not observations:
            return RainfallAnalysis(
                latest_intensity=(
                    RainfallIntensity.UNDETERMINED
                ),
                latest_rate_mm_h=None,
                observed_24h_mm=None,
                maximum_observed_rate_mm_h=None,
                trend="undetermined",
                warnings=(
                    "No accepted rainfall observations",
                ),
            )

        ordered = sorted(
            observations,
            key=lambda item: item.observed_at,
        )

        latest_time = ordered[-1].observed_at

        latest_group = [
            item
            for item in ordered
            if item.observed_at == latest_time
        ]

        latest_rates = [
            self.intensity_engine.hourly_rate(item)
            for item in latest_group
        ]

        latest_rate = statistics.median(
            latest_rates
        )

        all_rates = [
            self.intensity_engine.hourly_rate(item)
            for item in ordered
        ]

        total_24h = self._complete_24h_total(
            ordered,
            now=now,
        )

        # Trend is meaningful only when a single
        # source has at least two sequential,
        # non-overlapping observation windows.
        trend = "undetermined"

        by_source: dict[
            str,
            list[WeatherObservation],
        ] = {}

        for item in ordered:
            by_source.setdefault(
                item.source_id,
                [],
            ).append(
                item
            )

        for source_items in by_source.values():
            if len(source_items) < 2:
                continue

            previous, latest = source_items[-2:]

            if previous.observed_at != (
                latest.observed_at
                - timedelta(
                    minutes=latest.accumulation_minutes
                )
            ):
                continue

            previous_rate = (
                self.intensity_engine.hourly_rate(
                    previous
                )
            )

            current_rate = (
                self.intensity_engine.hourly_rate(
                    latest
                )
            )

            if current_rate > previous_rate * 1.20:
                trend = "increasing"

            elif current_rate < previous_rate * 0.80:
                trend = "decreasing"

            else:
                trend = "stable"

            break

        warnings = []

        if total_24h is None:
            warnings.append(
                "Complete non-overlapping 24-hour "
                "rainfall coverage is unavailable"
            )

        if trend == "undetermined":
            warnings.append(
                "Insufficient comparable observations "
                "for a reliable rainfall trend"
            )

        return RainfallAnalysis(
            latest_intensity=(
                self.intensity_engine.classify(
                    latest_rate
                )
            ),
            latest_rate_mm_h=round(
                latest_rate,
                3,
            ),
            observed_24h_mm=total_24h,
            maximum_observed_rate_mm_h=round(
                max(all_rates),
                3,
            ),
            trend=trend,
            warnings=tuple(warnings),
        )


# ============================================================
# FORECAST ANALYSIS ENGINE
# ============================================================

class ForecastAnalysisEngine:
    def __init__(
        self,
        config: WeatherAgentConfig,
    ) -> None:
        self.config = config

    def analyze(
        self,
        forecasts: Sequence[WeatherForecast],
    ) -> ForecastAnalysis:
        if not forecasts:
            return ForecastAnalysis(
                maximum_forecast_rainfall_mm=None,
                maximum_forecast_wind_kmh=None,
                maximum_thunderstorm_probability=None,
                forecast_concern=(
                    WeatherConcernLevel.UNDETERMINED
                ),
                warnings=(
                    "No accepted weather forecasts",
                ),
            )

        rainfall_values = [
            item.predicted_rainfall_mm
            for item in forecasts
        ]

        wind_values = [
            item.predicted_max_wind_kmh
            for item in forecasts
            if item.predicted_max_wind_kmh is not None
        ]

        thunder_values = [
            item.thunderstorm_probability
            for item in forecasts
            if item.thunderstorm_probability is not None
        ]

        maximum_rainfall = max(
            rainfall_values
        )

        maximum_wind = (
            max(wind_values)
            if wind_values
            else None
        )

        maximum_thunder = (
            max(thunder_values)
            if thunder_values
            else None
        )

        # Forecast rainfall amounts may describe
        # different time horizons. Without a common
        # duration they must not be treated as 24h
        # accumulations.
        concern = WeatherConcernLevel.ELEVATED

        if (
            maximum_wind is not None
            and maximum_wind
            >= self.config.severe_wind_kmh
        ):
            concern = WeatherConcernLevel.SEVERE

        elif (
            maximum_wind is not None
            and maximum_wind
            >= self.config.strong_wind_kmh
        ):
            concern = WeatherConcernLevel.HIGH

        if (
            maximum_thunder is not None
            and maximum_thunder >= 0.80
            and concern != WeatherConcernLevel.SEVERE
        ):
            concern = WeatherConcernLevel.HIGH

        return ForecastAnalysis(
            maximum_forecast_rainfall_mm=round(
                maximum_rainfall,
                3,
            ),
            maximum_forecast_wind_kmh=maximum_wind,
            maximum_thunderstorm_probability=(
                maximum_thunder
            ),
            forecast_concern=concern,
            warnings=(
                "Forecast rainfall windows may differ",
                "Forecast maxima are not additive",
                "Forecast concern is heuristic, not an official warning",
            ),
        )


# ============================================================
# SOURCE CONSISTENCY ENGINE
# ============================================================

class WeatherConsistencyEngine:
    def __init__(
        self,
        config: WeatherAgentConfig,
    ) -> None:
        self.config = config

    def detect_conflicts(
        self,
        observations: Sequence[WeatherObservation],
    ) -> tuple[str, ...]:
        groups: dict[
            tuple[datetime, int],
            list[WeatherObservation],
        ] = {}

        for item in observations:
            key = (
                item.observed_at,
                item.accumulation_minutes,
            )

            groups.setdefault(
                key,
                [],
            ).append(
                item
            )

        warnings = []

        for (_, duration), group in groups.items():
            if len({
                item.source_id
                for item in group
            }) < 2:
                continue

            rates = [
                item.rainfall_mm
                for item in group
            ]

            maximum = max(rates)
            minimum = min(rates)

            if maximum <= 0:
                continue

            difference = (
                maximum - minimum
            ) / maximum

            if difference > self.config.disagreement_fraction:
                warnings.append(
                    "Rainfall observations disagree "
                    f"for {duration}-minute window"
                )

        return tuple(
            dict.fromkeys(warnings)
        )


# ============================================================
# WEATHER CONCERN FUSION
# ============================================================

class WeatherConcernEngine:
    INTENSITY_RANK = {
        RainfallIntensity.NONE: 0,
        RainfallIntensity.LIGHT: 0,
        RainfallIntensity.MODERATE: 1,
        RainfallIntensity.HEAVY: 2,
        RainfallIntensity.VERY_HEAVY: 3,
        RainfallIntensity.EXTREME: 3,
        RainfallIntensity.UNDETERMINED: -1,
    }

    CONCERN_RANK = {
        WeatherConcernLevel.LOW: 0,
        WeatherConcernLevel.ELEVATED: 1,
        WeatherConcernLevel.HIGH: 2,
        WeatherConcernLevel.SEVERE: 3,
        WeatherConcernLevel.UNDETERMINED: -1,
    }

    LEVELS = (
        WeatherConcernLevel.LOW,
        WeatherConcernLevel.ELEVATED,
        WeatherConcernLevel.HIGH,
        WeatherConcernLevel.SEVERE,
    )

    def __init__(
        self,
        config: WeatherAgentConfig,
    ) -> None:
        self.config = config

    def assess(
        self,
        rainfall: RainfallAnalysis,
        forecast: ForecastAnalysis,
        quality: WeatherEvidenceQuality,
        *,
        conflicts: Sequence[str],
    ) -> tuple[
        WeatherConcernLevel,
        WeatherConfidence,
    ]:
        if not quality.sufficient_sources:
            return (
                WeatherConcernLevel.UNDETERMINED,
                WeatherConfidence.INSUFFICIENT,
            )

        scores = []

        rainfall_rank = self.INTENSITY_RANK[
            rainfall.latest_intensity
        ]

        if rainfall_rank >= 0:
            scores.append(
                rainfall_rank
            )

        if rainfall.observed_24h_mm is not None:
            total = rainfall.observed_24h_mm

            if total >= self.config.severe_24h_mm:
                scores.append(3)

            elif total >= self.config.high_24h_mm:
                scores.append(2)

            elif total >= self.config.elevated_24h_mm:
                scores.append(1)

            else:
                scores.append(0)

        forecast_rank = self.CONCERN_RANK[
            forecast.forecast_concern
        ]

        if forecast_rank >= 0:
            scores.append(
                forecast_rank
            )

        if not scores:
            return (
                WeatherConcernLevel.UNDETERMINED,
                WeatherConfidence.INSUFFICIENT,
            )

        level = self.LEVELS[
            max(scores)
        ]

        if conflicts:
            confidence = WeatherConfidence.LOW

        elif quality.independent_source_count >= 3:
            confidence = WeatherConfidence.HIGH

        else:
            confidence = WeatherConfidence.MODERATE

        return level, confidence


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class WeatherRecommendationEngine:
    @staticmethod
    def generate(
        concern: WeatherConcernLevel,
        confidence: WeatherConfidence,
        *,
        conflicts: Sequence[str],
    ) -> tuple[str, ...]:
        recommendations = [
            "Consult current official IMD and authorized "
            "district disaster-management bulletins",
            "Forward validated weather evidence to the "
            "Flood Prediction and Risk Assessment agents",
            "Keep human operators informed of evidence "
            "freshness and forecast uncertainty",
        ]

        if concern == WeatherConcernLevel.SEVERE:
            recommendations.extend(
                [
                    "Request urgent review by authorized "
                    "weather and disaster-management personnel",
                    "Prepare an advisory briefing for "
                    "the incident command team",
                ]
            )

        elif concern == WeatherConcernLevel.HIGH:
            recommendations.append(
                "Prioritize authorized review of rainfall, "
                "river, drainage, and local hazard observations"
            )

        elif concern == WeatherConcernLevel.ELEVATED:
            recommendations.append(
                "Continue monitoring validated weather updates"
            )

        elif concern == WeatherConcernLevel.UNDETERMINED:
            recommendations.append(
                "Obtain additional verified weather data "
                "before making a weather concern assessment"
            )

        if confidence in {
            WeatherConfidence.LOW,
            WeatherConfidence.INSUFFICIENT,
        }:
            recommendations.append(
                "Escalate evidence uncertainty to human reviewers"
            )

        if conflicts:
            recommendations.append(
                "Resolve conflicting weather observations "
                "before operational interpretation"
            )

        return tuple(
            dict.fromkeys(recommendations)
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class WeatherAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            WeatherAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        report_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> WeatherAuditRecord:
        record = WeatherAuditRecord(
            audit_id=generate_id(),
            report_id=report_id,
            actor_id=actor_id,
            action=action,
            timestamp=utc_now(),
            details=dict(details or {}),
        )

        with self._lock:
            self._records.append(
                record
            )

        return record

    def list_records(
        self,
        report_id: str | None = None,
    ) -> tuple[WeatherAuditRecord, ...]:
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

class WeatherHumanReviewManager:
    ALLOWED_ROLES = {
        "district_disaster_management_officer",
        "authorized_meteorological_analyst",
        "incident_commander",
        "authorized_weather_officer",
    }

    def __init__(
        self,
        audit_logger: WeatherAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            WeatherReviewRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        report: WeatherAgentReport,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> WeatherReviewRecord:
        if not reviewer_id.strip():
            raise WeatherApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise WeatherApprovalError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise WeatherApprovalError(
                "Review reason required"
            )

        if (
            decision == ReviewDecision.CONFIRM_FOR_ANALYSIS
            and report.status
            != WeatherReportStatus.PENDING_HUMAN_REVIEW
        ):
            raise WeatherApprovalError(
                "Only reports pending human review "
                "can be confirmed for analysis"
            )

        with self._lock:
            if report.report_id in self._reviews:
                raise WeatherApprovalError(
                    "Report already reviewed"
                )

            review = WeatherReviewRecord(
                review_id=generate_id(),
                report_id=report.report_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                report.report_id
            ] = review

        self.audit_logger.record(
            report_id=report.report_id,
            actor_id=reviewer_id,
            action=f"human_review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN WEATHER AGENT
# ============================================================

class WeatherAgent:
    def __init__(
        self,
        config: WeatherAgentConfig | None = None,
    ) -> None:
        self.config = (
            config or WeatherAgentConfig()
        )

        self.evidence_validator = (
            WeatherEvidenceValidator(
                self.config
            )
        )

        self.rainfall_engine = (
            RainfallAnalysisEngine(
                self.config
            )
        )

        self.forecast_engine = (
            ForecastAnalysisEngine(
                self.config
            )
        )

        self.consistency_engine = (
            WeatherConsistencyEngine(
                self.config
            )
        )

        self.concern_engine = (
            WeatherConcernEngine(
                self.config
            )
        )

        self.recommendation_engine = (
            WeatherRecommendationEngine()
        )

        self.audit_logger = (
            WeatherAuditLogger()
        )

        self.review_manager = (
            WeatherHumanReviewManager(
                self.audit_logger
            )
        )

        self._reports: dict[
            str,
            WeatherAgentReport,
        ] = {}

        self._lock = threading.RLock()

    def run(
        self,
        request: WeatherAnalysisRequest,
        *,
        now: datetime | None = None,
    ) -> WeatherAgentReport:
        now = now or utc_now()

        validate_datetime(
            now,
            "now",
        )

        if request.created_at > now:
            raise WeatherValidationError(
                "Request timestamp is in the future"
            )

        (
            observations,
            forecasts,
            quality,
        ) = self.evidence_validator.validate(
            request,
            now=now,
        )

        rainfall = self.rainfall_engine.analyze(
            observations,
            now=now,
        )

        forecast = self.forecast_engine.analyze(
            forecasts
        )

        conflicts = (
            self.consistency_engine.detect_conflicts(
                observations
            )
        )

        concern, confidence = (
            self.concern_engine.assess(
                rainfall,
                forecast,
                quality,
                conflicts=conflicts,
            )
        )

        if not quality.sufficient_sources:
            status = (
                WeatherReportStatus.INSUFFICIENT_DATA
            )

        elif conflicts:
            status = (
                WeatherReportStatus.CONFLICTED_DATA
            )

        else:
            status = (
                WeatherReportStatus.PENDING_HUMAN_REVIEW
            )

        recommendations = (
            self.recommendation_engine.generate(
                concern,
                confidence,
                conflicts=conflicts,
            )
        )

        warnings = [
            "Weather analysis is advisory only",
            "Rainfall thresholds are illustrative",
            "No official IMD warning has been issued",
            "No public alert or emergency action was executed",
            "Human review is required",
        ]

        warnings.extend(
            quality.warnings
        )

        warnings.extend(
            rainfall.warnings
        )

        warnings.extend(
            forecast.warnings
        )

        warnings.extend(
            conflicts
        )

        report = WeatherAgentReport(
            report_id=generate_id(),
            request_id=request.request_id,
            district=request.district,
            locality=request.locality,
            evidence_quality=quality,
            rainfall=rainfall,
            forecast=forecast,
            concern_level=concern,
            confidence=confidence,
            recommendations=recommendations,
            status=status,
            requires_human_review=True,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            created_at=now,
        )

        with self._lock:
            self._reports[
                report.report_id
            ] = report

        self.audit_logger.record(
            report_id=report.report_id,
            actor_id="weather_agent",
            action="weather_analysis_completed",
            details={
                "district": request.district,
                "locality": request.locality,
                "concern_level": concern.value,
                "confidence": confidence.value,
                "status": status.value,
            },
        )

        return report

    def review_report(
        self,
        report_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> WeatherAgentReport:
        with self._lock:
            report = self._reports.get(
                report_id
            )

            if report is None:
                raise WeatherApprovalError(
                    "Weather report not found"
                )

            if report.status in {
                WeatherReportStatus.CONFIRMED_FOR_ANALYSIS,
                WeatherReportStatus.REJECTED,
                WeatherReportStatus.REVISION_REQUIRED,
            }:
                raise WeatherApprovalError(
                    "Report has already been reviewed"
                )

            self.review_manager.review(
                report,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            if decision == ReviewDecision.CONFIRM_FOR_ANALYSIS:
                new_status = (
                    WeatherReportStatus.CONFIRMED_FOR_ANALYSIS
                )

            elif decision == ReviewDecision.REJECT:
                new_status = (
                    WeatherReportStatus.REJECTED
                )

            else:
                new_status = (
                    WeatherReportStatus.REVISION_REQUIRED
                )

            updated = replace(
                report,
                status=new_status,
            )

            self._reports[
                report_id
            ] = updated

            return updated

    def get_report(
        self,
        report_id: str,
    ) -> WeatherAgentReport | None:
        with self._lock:
            return self._reports.get(
                report_id
            )

    def get_audit_history(
        self,
        report_id: str,
    ) -> tuple[WeatherAuditRecord, ...]:
        return self.audit_logger.list_records(
            report_id
        )


# ============================================================
# SUPERVISOR ADAPTER
# ============================================================

class WeatherSupervisorAdapter:
    """
    Adapter for the SupervisorAgent registry.

    The caller must provide a validated
    WeatherAnalysisRequest object.

    This adapter does not fetch weather data.
    """

    def __init__(
        self,
        weather_agent: WeatherAgent,
    ) -> None:
        self.weather_agent = weather_agent

    def __call__(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = payload.get(
            "weather_request"
        )

        if not isinstance(
            request,
            WeatherAnalysisRequest,
        ):
            raise WeatherValidationError(
                "weather_request must be "
                "WeatherAnalysisRequest"
            )

        report = self.weather_agent.run(
            request
        )

        return {
            "report_id": report.report_id,
            "district": report.district,
            "locality": report.locality,
            "concern_level": report.concern_level.value,
            "confidence": report.confidence.value,
            "status": report.status.value,
            "recommendations": list(
                report.recommendations
            ),
            "human_review_required": True,
            "advisory_only": True,
        }


# ============================================================
# JSON REPORT EXPORT
# ============================================================

class WeatherReportExporter:
    @staticmethod
    def export_json(
        report: WeatherAgentReport,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise WeatherExportError(
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

def create_weather_agent(
    config: WeatherAgentConfig | None = None,
) -> WeatherAgent:
    return WeatherAgent(
        config=config
    )


# ============================================================
# FICTIONAL DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional observations.
    # These are not real Kerala weather readings.

    observations = (
        WeatherObservation(
            observation_id="DEMO-OBS-001",
            source_id="DEMO-STATION-A",
            source_type=WeatherSourceType.RAIN_GAUGE,
            district="Thrissur",
            locality="Demo Zone",
            observed_at=now - timedelta(minutes=10),
            rainfall_mm=18.0,
            accumulation_minutes=60,
            wind_speed_kmh=32.0,
            thunderstorm_detected=True,
            verification=SourceVerification.VERIFIED,
            reliability=0.94,
        ),
        WeatherObservation(
            observation_id="DEMO-OBS-002",
            source_id="DEMO-STATION-B",
            source_type=WeatherSourceType.WEATHER_STATION,
            district="Thrissur",
            locality="Demo Zone",
            observed_at=now - timedelta(minutes=10),
            rainfall_mm=17.0,
            accumulation_minutes=60,
            wind_speed_kmh=35.0,
            thunderstorm_detected=True,
            verification=SourceVerification.VERIFIED,
            reliability=0.91,
        ),
    )

    forecasts = (
        WeatherForecast(
            forecast_id="DEMO-FORECAST-001",
            source_id="DEMO-FORECAST-SERVICE",
            district="Thrissur",
            locality="Demo Zone",
            issued_at=now - timedelta(minutes=30),
            valid_from=now,
            valid_until=now + timedelta(hours=6),
            predicted_rainfall_mm=45.0,
            precipitation_probability=0.85,
            predicted_max_wind_kmh=45.0,
            thunderstorm_probability=0.80,
            verification=SourceVerification.VERIFIED,
            reliability=0.90,
        ),
    )

    request = WeatherAnalysisRequest(
        request_id="DEMO-WEATHER-REQUEST-001",
        district="Thrissur",
        locality="Demo Zone",
        observations=observations,
        forecasts=forecasts,
        created_at=now,
    )

    agent = create_weather_agent()

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

    if report.status == (
        WeatherReportStatus.PENDING_HUMAN_REVIEW
    ):
        reviewed = agent.review_report(
            report.report_id,
            reviewer_id="DEMO-WEATHER-OFFICER",
            reviewer_role="authorized_weather_officer",
            decision=ReviewDecision.CONFIRM_FOR_ANALYSIS,
            reason="Fictional demonstration review",
        )

        print(
            "Review status:",
            reviewed.status.value,
        )

        WeatherReportExporter.export_json(
            reviewed,
            "outputs/agents/weather_report.json",
        )

    print(
        "No real-world weather warning, "
        "public alert, or emergency action was issued."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "WeatherAgentError",
    "WeatherValidationError",
    "WeatherEvidenceError",
    "WeatherApprovalError",
    "WeatherExportError",
    "WeatherSourceType",
    "SourceVerification",
    "RainfallIntensity",
    "WeatherConcernLevel",
    "WeatherConfidence",
    "WeatherReportStatus",
    "ReviewDecision",
    "WeatherAgentConfig",
    "WeatherObservation",
    "WeatherForecast",
    "WeatherAnalysisRequest",
    "WeatherEvidenceQuality",
    "RainfallAnalysis",
    "ForecastAnalysis",
    "WeatherAgentReport",
    "WeatherReviewRecord",
    "WeatherAuditRecord",
    "WeatherEvidenceValidator",
    "RainfallIntensityEngine",
    "RainfallAnalysisEngine",
    "ForecastAnalysisEngine",
    "WeatherConsistencyEngine",
    "WeatherConcernEngine",
    "WeatherRecommendationEngine",
    "WeatherAuditLogger",
    "WeatherHumanReviewManager",
    "WeatherAgent",
    "WeatherSupervisorAdapter",
    "WeatherReportExporter",
    "create_weather_agent",
]
