"""Placeholder module: ai/evaluation/forecasting_evaluation.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Forecasting Evaluation Module

File:
    backend/ai/evaluation/forecasting_evaluation.py

Python:
    3.11+

Dependencies:
    numpy>=1.26
    pandas>=2.1
    scikit-learn>=1.4

Evaluates:
    Rainfall forecasting
    River level prediction
    Flood risk classification
    Flood severity classification
    Shelter demand forecasting
    District-level forecasting
    Forecast horizon performance
    Prediction interval reliability
    Model regression

Important:
    Forecast observations and predictions
    must represent the same target, units,
    timestamp, station/district and horizon.

    All forecasts are advisory.
    This module does not issue emergency alerts,
    authorize evacuation or dispatch responders.
"""

from __future__ import annotations

import json
import logging
import math
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class ForecastEvaluationError(Exception):
    """Base forecasting evaluation error."""


class ForecastValidationError(ForecastEvaluationError):
    """Invalid forecast evaluation data."""


class ForecastAlignmentError(ForecastEvaluationError):
    """Forecast and observation alignment failed."""


class ForecastExportError(ForecastEvaluationError):
    """Evaluation report export failed."""


# ============================================================
# ENUMERATIONS
# ============================================================

class ForecastTask(str, Enum):
    RAINFALL = "rainfall"
    RIVER_LEVEL = "river_level"
    FLOOD_RISK = "flood_risk"
    FLOOD_SEVERITY = "flood_severity"
    SHELTER_DEMAND = "shelter_demand"


class ForecastEvaluationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_DATA = "insufficient_data"


class ForecastMetric(str, Enum):
    MAE = "mae"
    RMSE = "rmse"
    MSE = "mse"
    BIAS = "bias"
    WAPE = "wape"
    R2 = "r2"
    INTERVAL_COVERAGE = "interval_coverage"
    INTERVAL_WIDTH = "interval_width"
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    BRIER = "brier"
    LOG_LOSS = "log_loss"


class ForecastRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    result = float(value)

    return result if math.isfinite(result) else None


def validate_fraction(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ForecastValidationError(
            f"{name} must be between 0 and 1"
        )


def validate_numeric_array(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ForecastValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if result.ndim != 1:
        raise ForecastValidationError(
            f"{name} must be one-dimensional"
        )

    if result.size == 0:
        raise ForecastValidationError(
            f"{name} cannot be empty"
        )

    if not np.all(np.isfinite(result)):
        raise ForecastValidationError(
            f"{name} contains non-finite values"
        )

    return result


def validate_pair(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    y_true = validate_numeric_array(
        actual,
        "actual",
    )

    y_pred = validate_numeric_array(
        predicted,
        "predicted",
    )

    if len(y_true) != len(y_pred):
        raise ForecastAlignmentError(
            "Actual and predicted lengths differ"
        )

    return y_true, y_pred


def to_serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_serializable(item)
            for item in value
        ]

    if hasattr(value, "__dataclass_fields__"):
        return to_serializable(
            asdict(value)
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ForecastEvaluationConfig:
    minimum_samples: int = 30

    minimum_group_samples: int = 10

    interval_confidence: float = 0.90

    probability_threshold: float = 0.50

    maximum_mae: float | None = None

    maximum_rmse: float | None = None

    maximum_wape: float | None = None

    minimum_interval_coverage: float | None = None

    minimum_recall: float | None = None

    maximum_brier: float | None = None

    require_group_evaluation: bool = False

    def __post_init__(self) -> None:
        if self.minimum_samples < 2:
            raise ForecastValidationError(
                "minimum_samples must be at least 2"
            )

        if self.minimum_group_samples < 2:
            raise ForecastValidationError(
                "minimum_group_samples must be at least 2"
            )

        for name in (
            "interval_confidence",
            "probability_threshold",
        ):
            validate_fraction(
                getattr(self, name),
                name,
            )

        for name in (
            "minimum_interval_coverage",
            "minimum_recall",
            "maximum_brier",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_fraction(
                    value,
                    name,
                )

        for name in (
            "maximum_mae",
            "maximum_rmse",
            "maximum_wape",
        ):
            value = getattr(self, name)

            if value is not None:
                if (
                    not math.isfinite(value)
                    or value < 0
                ):
                    raise ForecastValidationError(
                        f"{name} must be nonnegative"
                    )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class RegressionForecastMetrics:
    sample_count: int

    mae: float
    mse: float
    rmse: float
    bias: float

    wape: float | None
    r2: float | None

    actual_mean: float
    predicted_mean: float

    actual_min: float
    actual_max: float


@dataclass(frozen=True)
class IntervalForecastMetrics:
    sample_count: int

    nominal_coverage: float

    empirical_coverage: float

    mean_interval_width: float

    median_interval_width: float

    lower_miss_rate: float
    upper_miss_rate: float

    coverage_gap: float


@dataclass(frozen=True)
class ClassificationForecastMetrics:
    sample_count: int

    accuracy: float
    balanced_accuracy: float

    precision: float
    recall: float
    f1: float

    roc_auc: float | None

    brier_score: float
    log_loss: float

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True)
class CalibrationBin:
    lower_bound: float
    upper_bound: float

    sample_count: int

    mean_predicted_probability: float | None

    observed_event_rate: float | None


@dataclass(frozen=True)
class CalibrationReport:
    bins: tuple[CalibrationBin, ...]

    expected_calibration_error: float


@dataclass(frozen=True)
class ForecastGroupResult:
    group_name: str

    sample_count: int

    regression_metrics: RegressionForecastMetrics | None

    classification_metrics: (
        ClassificationForecastMetrics | None
    )

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ForecastEvaluationReport:
    evaluation_id: str

    task: ForecastTask

    status: ForecastEvaluationStatus

    regression_metrics: RegressionForecastMetrics | None

    interval_metrics: IntervalForecastMetrics | None

    classification_metrics: (
        ClassificationForecastMetrics | None
    )

    calibration_report: CalibrationReport | None

    group_results: tuple[ForecastGroupResult, ...]

    passed_quality_gates: tuple[str, ...]
    failed_quality_gates: tuple[str, ...]

    warnings: tuple[str, ...]

    evaluated_at: datetime


# ============================================================
# REGRESSION METRIC EVALUATOR
# ============================================================

class RegressionForecastEvaluator:
    """
    Evaluates continuous targets:

        rainfall_mm
        river_level_m
        shelter_demand_people

    WAPE is undefined when sum(abs(actual)) = 0.
    R2 is undefined for fewer than two samples
    or a constant target.
    """

    def evaluate(
        self,
        actual: Sequence[float],
        predicted: Sequence[float],
    ) -> RegressionForecastMetrics:
        y_true, y_pred = validate_pair(
            actual,
            predicted,
        )

        errors = y_pred - y_true

        mae = float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        )

        mse = float(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        rmse = float(
            np.sqrt(mse)
        )

        bias = float(
            np.mean(errors)
        )

        denominator = float(
            np.sum(np.abs(y_true))
        )

        wape = (
            float(
                np.sum(np.abs(errors))
                / denominator
            )
            if denominator > 0
            else None
        )

        r2 = (
            float(r2_score(y_true, y_pred))
            if (
                len(y_true) >= 2
                and np.var(y_true) > 0
            )
            else None
        )

        return RegressionForecastMetrics(
            sample_count=len(y_true),
            mae=mae,
            mse=mse,
            rmse=rmse,
            bias=bias,
            wape=wape,
            r2=r2,
            actual_mean=float(
                np.mean(y_true)
            ),
            predicted_mean=float(
                np.mean(y_pred)
            ),
            actual_min=float(
                np.min(y_true)
            ),
            actual_max=float(
                np.max(y_true)
            ),
        )


# ============================================================
# PREDICTION INTERVAL EVALUATOR
# ============================================================

class PredictionIntervalEvaluator:
    """
    Evaluates whether observed values fall
    inside forecast prediction intervals.

    Coverage alone is insufficient:
    excessively wide intervals may cover
    nearly every observation.
    """

    def evaluate(
        self,
        actual: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        nominal_coverage: float = 0.90,
    ) -> IntervalForecastMetrics:
        validate_fraction(
            nominal_coverage,
            "nominal_coverage",
        )

        y_true = validate_numeric_array(
            actual,
            "actual",
        )

        lower_array = validate_numeric_array(
            lower,
            "lower",
        )

        upper_array = validate_numeric_array(
            upper,
            "upper",
        )

        if not (
            len(y_true)
            == len(lower_array)
            == len(upper_array)
        ):
            raise ForecastAlignmentError(
                "Prediction interval lengths differ"
            )

        if np.any(
            lower_array > upper_array
        ):
            raise ForecastValidationError(
                "Lower interval exceeds upper interval"
            )

        inside = (
            (y_true >= lower_array)
            & (y_true <= upper_array)
        )

        widths = (
            upper_array - lower_array
        )

        empirical = float(
            np.mean(inside)
        )

        return IntervalForecastMetrics(
            sample_count=len(y_true),
            nominal_coverage=nominal_coverage,
            empirical_coverage=empirical,
            mean_interval_width=float(
                np.mean(widths)
            ),
            median_interval_width=float(
                np.median(widths)
            ),
            lower_miss_rate=float(
                np.mean(y_true < lower_array)
            ),
            upper_miss_rate=float(
                np.mean(y_true > upper_array)
            ),
            coverage_gap=(
                empirical - nominal_coverage
            ),
        )


# ============================================================
# CLASSIFICATION EVALUATOR
# ============================================================

class FloodRiskClassificationEvaluator:
    """
    Binary event probability evaluation.

    Examples:
        flood event vs no flood event
        danger threshold exceeded vs not exceeded

    Multi-class flood severity should be
    evaluated with a separate multi-class
    evaluation configuration.
    """

    def evaluate(
        self,
        actual: Sequence[int],
        probabilities: Sequence[float],
        *,
        threshold: float = 0.50,
    ) -> ClassificationForecastMetrics:
        validate_fraction(
            threshold,
            "threshold",
        )

        y_true = validate_numeric_array(
            actual,
            "actual",
        )

        y_prob = validate_numeric_array(
            probabilities,
            "probabilities",
        )

        if len(y_true) != len(y_prob):
            raise ForecastAlignmentError(
                "Actual and probability lengths differ"
            )

        if not np.all(
            np.isin(y_true, [0, 1])
        ):
            raise ForecastValidationError(
                "Binary labels must be 0 or 1"
            )

        if np.any(
            (y_prob < 0) | (y_prob > 1)
        ):
            raise ForecastValidationError(
                "Probabilities must be between 0 and 1"
            )

        labels = y_true.astype(int)

        predictions = (
            y_prob >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()

        auc = None

        if len(np.unique(labels)) == 2:
            auc = float(
                roc_auc_score(
                    labels,
                    y_prob,
                )
            )

        clipped = np.clip(
            y_prob,
            1e-15,
            1 - 1e-15,
        )

        return ClassificationForecastMetrics(
            sample_count=len(labels),
            accuracy=float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),
            balanced_accuracy=float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),
            precision=float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            f1=float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            roc_auc=auc,
            brier_score=float(
                brier_score_loss(
                    labels,
                    y_prob,
                )
            ),
            log_loss=float(
                log_loss(
                    labels,
                    clipped,
                    labels=[0, 1],
                )
            ),
            true_positive=int(tp),
            false_positive=int(fp),
            true_negative=int(tn),
            false_negative=int(fn),
        )


# ============================================================
# PROBABILITY CALIBRATION
# ============================================================

class ForecastCalibrationEvaluator:
    """
    Reliability diagram data and ECE.

    ECE is a descriptive metric and depends
    on binning and sample size.
    """

    def evaluate(
        self,
        actual: Sequence[int],
        probabilities: Sequence[float],
        *,
        number_of_bins: int = 10,
    ) -> CalibrationReport:
        if number_of_bins < 2:
            raise ForecastValidationError(
                "number_of_bins must be at least 2"
            )

        labels = validate_numeric_array(
            actual,
            "actual",
        )

        probs = validate_numeric_array(
            probabilities,
            "probabilities",
        )

        if len(labels) != len(probs):
            raise ForecastAlignmentError(
                "Calibration arrays differ in length"
            )

        if not np.all(np.isin(labels, [0, 1])):
            raise ForecastValidationError(
                "Calibration labels must be binary"
            )

        if np.any((probs < 0) | (probs > 1)):
            raise ForecastValidationError(
                "Invalid calibration probabilities"
            )

        edges = np.linspace(
            0,
            1,
            number_of_bins + 1,
        )

        bins: list[CalibrationBin] = []

        weighted_error = 0.0

        for index in range(number_of_bins):
            lower = float(edges[index])
            upper = float(edges[index + 1])

            if index == number_of_bins - 1:
                mask = (
                    (probs >= lower)
                    & (probs <= upper)
                )
            else:
                mask = (
                    (probs >= lower)
                    & (probs < upper)
                )

            count = int(
                np.sum(mask)
            )

            if count == 0:
                mean_probability = None
                observed_rate = None

            else:
                mean_probability = float(
                    np.mean(probs[mask])
                )

                observed_rate = float(
                    np.mean(labels[mask])
                )

                weighted_error += (
                    count / len(labels)
                ) * abs(
                    mean_probability - observed_rate
                )

            bins.append(
                CalibrationBin(
                    lower_bound=lower,
                    upper_bound=upper,
                    sample_count=count,
                    mean_predicted_probability=(
                        mean_probability
                    ),
                    observed_event_rate=observed_rate,
                )
            )

        return CalibrationReport(
            bins=tuple(bins),
            expected_calibration_error=float(
                weighted_error
            ),
        )


# ============================================================
# FORECAST ALIGNMENT VALIDATOR
# ============================================================

class ForecastAlignmentValidator:
    """
    Aligns predictions with observations.

    Required columns:
        forecast_issued_at
        target_time
        entity_id
        horizon_hours
        actual
        predicted

    Evaluation data must already be joined
    against the correct observed target.

    Duplicate forecast identities are rejected.
    """

    REQUIRED_COLUMNS = frozenset({
        "forecast_issued_at",
        "target_time",
        "entity_id",
        "horizon_hours",
        "actual",
        "predicted",
    })

    KEY_COLUMNS = [
        "forecast_issued_at",
        "target_time",
        "entity_id",
        "horizon_hours",
    ]

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        missing = (
            self.REQUIRED_COLUMNS
            - set(dataframe.columns)
        )

        if missing:
            raise ForecastAlignmentError(
                "Missing columns: "
                + ", ".join(sorted(missing))
            )

        if dataframe.empty:
            raise ForecastAlignmentError(
                "Forecast dataset is empty"
            )

        frame = dataframe.copy()

        for column in (
            "forecast_issued_at",
            "target_time",
        ):
            frame[column] = pd.to_datetime(
                frame[column],
                utc=True,
                errors="raise",
            )

        for column in (
            "actual",
            "predicted",
            "horizon_hours",
        ):
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            )

        if frame[
            self.KEY_COLUMNS
        ].isna().any().any():
            raise ForecastAlignmentError(
                "Forecast identity contains missing values"
            )

        if frame[
            self.KEY_COLUMNS
        ].duplicated().any():
            raise ForecastAlignmentError(
                "Duplicate forecast identity detected"
            )

        if frame[
            ["actual", "predicted"]
        ].isna().any().any():
            raise ForecastAlignmentError(
                "Missing actual or predicted values"
            )

        if not np.all(
            np.isfinite(
                frame[
                    ["actual", "predicted", "horizon_hours"]
                ].to_numpy(dtype=float)
            )
        ):
            raise ForecastAlignmentError(
                "Non-finite forecast values"
            )

        if (
            frame["horizon_hours"] <= 0
        ).any():
            raise ForecastAlignmentError(
                "Forecast horizon must be positive"
            )

        if (
            frame["target_time"]
            <= frame["forecast_issued_at"]
        ).any():
            raise ForecastAlignmentError(
                "Forecast target must follow issue time"
            )

        actual_horizon = (
            frame["target_time"]
            - frame["forecast_issued_at"]
        ).dt.total_seconds() / 3600

        if not np.allclose(
            actual_horizon,
            frame["horizon_hours"],
            rtol=0,
            atol=1e-6,
        ):
            raise ForecastAlignmentError(
                "Forecast horizon does not match timestamps"
            )

        return frame.sort_values(
            self.KEY_COLUMNS
        ).reset_index(drop=True)


# ============================================================
# GROUP EVALUATOR
# ============================================================

class ForecastGroupEvaluator:
    """
    Evaluates model performance by:
        district
        station
        forecast horizon
        season
        rainfall intensity group

    Group labels are supplied by the caller.
    """

    def __init__(
        self,
        config: ForecastEvaluationConfig,
    ) -> None:
        self.config = config

        self.regression = (
            RegressionForecastEvaluator()
        )

        self.classification = (
            FloodRiskClassificationEvaluator()
        )

    def evaluate_regression_groups(
        self,
        dataframe: pd.DataFrame,
        *,
        group_column: str,
    ) -> tuple[ForecastGroupResult, ...]:
        if group_column not in dataframe.columns:
            raise ForecastValidationError(
                "Group column not found"
            )

        results: list[ForecastGroupResult] = []

        for group_name, group in dataframe.groupby(
            group_column,
            dropna=False,
        ):
            warnings: list[str] = []

            if (
                len(group)
                < self.config.minimum_group_samples
            ):
                warnings.append(
                    "Insufficient group samples"
                )

                metrics = None

            else:
                metrics = self.regression.evaluate(
                    group["actual"].to_numpy(),
                    group["predicted"].to_numpy(),
                )

            results.append(
                ForecastGroupResult(
                    group_name=str(group_name),
                    sample_count=len(group),
                    regression_metrics=metrics,
                    classification_metrics=None,
                    warnings=tuple(warnings),
                )
            )

        return tuple(results)

    def evaluate_classification_groups(
        self,
        dataframe: pd.DataFrame,
        *,
        group_column: str,
    ) -> tuple[ForecastGroupResult, ...]:
        if group_column not in dataframe.columns:
            raise ForecastValidationError(
                "Group column not found"
            )

        results: list[ForecastGroupResult] = []

        for group_name, group in dataframe.groupby(
            group_column,
            dropna=False,
        ):
            warnings: list[str] = []

            if (
                len(group)
                < self.config.minimum_group_samples
            ):
                warnings.append(
                    "Insufficient group samples"
                )

                metrics = None

            else:
                metrics = self.classification.evaluate(
                    group["actual"].to_numpy(),
                    group["predicted"].to_numpy(),
                    threshold=(
                        self.config.probability_threshold
                    ),
                )

            results.append(
                ForecastGroupResult(
                    group_name=str(group_name),
                    sample_count=len(group),
                    regression_metrics=None,
                    classification_metrics=metrics,
                    warnings=tuple(warnings),
                )
            )

        return tuple(results)


# ============================================================
# QUALITY GATE EVALUATOR
# ============================================================

class ForecastQualityGateEvaluator:
    """
    Thresholds must be established using
    domain-specific validation data.

    There are no universal acceptable MAE,
    RMSE or flood-warning recall thresholds.
    """

    def __init__(
        self,
        config: ForecastEvaluationConfig,
    ) -> None:
        self.config = config

    def regression_gates(
        self,
        metrics: RegressionForecastMetrics,
        interval: IntervalForecastMetrics | None = None,
    ) -> tuple[list[str], list[str]]:
        passed: list[str] = []
        failed: list[str] = []

        checks = (
            (
                "mae",
                metrics.mae,
                self.config.maximum_mae,
            ),
            (
                "rmse",
                metrics.rmse,
                self.config.maximum_rmse,
            ),
            (
                "wape",
                metrics.wape,
                self.config.maximum_wape,
            ),
        )

        for name, value, maximum in checks:
            if maximum is None:
                continue

            if value is None:
                failed.append(
                    f"{name}: metric unavailable"
                )

            elif value <= maximum:
                passed.append(name)

            else:
                failed.append(name)

        minimum_coverage = (
            self.config.minimum_interval_coverage
        )

        if minimum_coverage is not None:
            if interval is None:
                failed.append(
                    "interval_coverage: unavailable"
                )

            elif (
                interval.empirical_coverage
                >= minimum_coverage
            ):
                passed.append(
                    "interval_coverage"
                )

            else:
                failed.append(
                    "interval_coverage"
                )

        return passed, failed

    def classification_gates(
        self,
        metrics: ClassificationForecastMetrics,
    ) -> tuple[list[str], list[str]]:
        passed: list[str] = []
        failed: list[str] = []

        if self.config.minimum_recall is not None:
            if (
                metrics.recall
                >= self.config.minimum_recall
            ):
                passed.append("recall")
            else:
                failed.append("recall")

        if self.config.maximum_brier is not None:
            if (
                metrics.brier_score
                <= self.config.maximum_brier
            ):
                passed.append("brier")
            else:
                failed.append("brier")

        return passed, failed


# ============================================================
# MAIN FORECAST EVALUATION ENGINE
# ============================================================

class ForecastEvaluationEngine:
    """
    Evaluates aligned forecasting datasets.

    Regression:
        rainfall
        river level
        shelter demand

    Binary classification:
        flood risk
        binary flood severity event

    The caller must supply the correct
    ground-truth target definition.
    """

    REGRESSION_TASKS = frozenset({
        ForecastTask.RAINFALL,
        ForecastTask.RIVER_LEVEL,
        ForecastTask.SHELTER_DEMAND,
    })

    CLASSIFICATION_TASKS = frozenset({
        ForecastTask.FLOOD_RISK,
        ForecastTask.FLOOD_SEVERITY,
    })

    def __init__(
        self,
        config: ForecastEvaluationConfig | None = None,
    ) -> None:
        self.config = (
            config or ForecastEvaluationConfig()
        )

        self.alignment = (
            ForecastAlignmentValidator()
        )

        self.regression = (
            RegressionForecastEvaluator()
        )

        self.intervals = (
            PredictionIntervalEvaluator()
        )

        self.classification = (
            FloodRiskClassificationEvaluator()
        )

        self.calibration = (
            ForecastCalibrationEvaluator()
        )

        self.groups = ForecastGroupEvaluator(
            self.config
        )

        self.gates = ForecastQualityGateEvaluator(
            self.config
        )

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        task: ForecastTask,
        group_column: str | None = None,
    ) -> ForecastEvaluationReport:
        frame = self.alignment.validate(
            dataframe
        )

        warnings: list[str] = []

        if len(frame) < self.config.minimum_samples:
            warnings.append(
                "Evaluation dataset below minimum sample count"
            )

        regression_metrics = None
        interval_metrics = None
        classification_metrics = None
        calibration_report = None

        group_results: tuple[
            ForecastGroupResult,
            ...
        ] = ()

        passed_gates: list[str] = []
        failed_gates: list[str] = []

        if task in self.REGRESSION_TASKS:
            regression_metrics = self.regression.evaluate(
                frame["actual"].to_numpy(),
                frame["predicted"].to_numpy(),
            )

            has_lower = "lower" in frame.columns
            has_upper = "upper" in frame.columns

            if has_lower != has_upper:
                raise ForecastValidationError(
                    "Both lower and upper intervals are required"
                )

            if has_lower and has_upper:
                interval_metrics = self.intervals.evaluate(
                    frame["actual"].to_numpy(),
                    frame["lower"].to_numpy(),
                    frame["upper"].to_numpy(),
                    nominal_coverage=(
                        self.config.interval_confidence
                    ),
                )

            passed_gates, failed_gates = (
                self.gates.regression_gates(
                    regression_metrics,
                    interval_metrics,
                )
            )

            if group_column:
                group_results = (
                    self.groups.evaluate_regression_groups(
                        frame,
                        group_column=group_column,
                    )
                )

        elif task in self.CLASSIFICATION_TASKS:
            classification_metrics = (
                self.classification.evaluate(
                    frame["actual"].to_numpy(),
                    frame["predicted"].to_numpy(),
                    threshold=(
                        self.config.probability_threshold
                    ),
                )
            )

            calibration_report = (
                self.calibration.evaluate(
                    frame["actual"].to_numpy(),
                    frame["predicted"].to_numpy(),
                )
            )

            passed_gates, failed_gates = (
                self.gates.classification_gates(
                    classification_metrics
                )
            )

            if group_column:
                group_results = (
                    self.groups.evaluate_classification_groups(
                        frame,
                        group_column=group_column,
                    )
                )

        else:
            raise ForecastValidationError(
                f"Unsupported forecast task: {task}"
            )

        if (
            self.config.require_group_evaluation
            and not group_column
        ):
            failed_gates.append(
                "required group evaluation missing"
            )

        if failed_gates:
            status = ForecastEvaluationStatus.FAILED

        elif len(frame) < self.config.minimum_samples:
            status = (
                ForecastEvaluationStatus.INSUFFICIENT_DATA
            )

        elif not passed_gates:
            status = (
                ForecastEvaluationStatus.REVIEW_REQUIRED
            )

            warnings.append(
                "No configured performance thresholds "
                "were evaluated"
            )

        else:
            status = ForecastEvaluationStatus.PASSED

        return ForecastEvaluationReport(
            evaluation_id=new_id(),
            task=task,
            status=status,
            regression_metrics=regression_metrics,
            interval_metrics=interval_metrics,
            classification_metrics=classification_metrics,
            calibration_report=calibration_report,
            group_results=group_results,
            passed_quality_gates=tuple(passed_gates),
            failed_quality_gates=tuple(failed_gates),
            warnings=tuple(warnings),
            evaluated_at=utc_now(),
        )


# ============================================================
# CHRONOLOGICAL BACKTESTING
# ============================================================

@dataclass(frozen=True)
class BacktestFold:
    fold_id: int

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    train_end_time: datetime
    test_start_time: datetime


class ChronologicalBacktestSplitter:
    """
    Expanding-window time-series splitter.

    An embargo prevents training on rows
    whose target timestamps overlap the
    beginning of the test period.

    For multi-horizon forecasts, target-time
    leakage must be checked per fold.
    """

    def split(
        self,
        dataframe: pd.DataFrame,
        *,
        n_splits: int = 5,
        minimum_train_samples: int = 100,
        test_samples: int = 30,
        embargo_hours: float = 0,
    ) -> tuple[BacktestFold, ...]:
        if (
            n_splits < 1
            or minimum_train_samples < 1
            or test_samples < 1
            or embargo_hours < 0
        ):
            raise ForecastValidationError(
                "Invalid backtesting configuration"
            )

        required = {
            "forecast_issued_at",
            "target_time",
        }

        if not required.issubset(
            dataframe.columns
        ):
            raise ForecastValidationError(
                "Backtesting timestamps are missing"
            )

        frame = dataframe.copy()

        frame["forecast_issued_at"] = pd.to_datetime(
            frame["forecast_issued_at"],
            utc=True,
        )

        frame["target_time"] = pd.to_datetime(
            frame["target_time"],
            utc=True,
        )

        frame = frame.sort_values(
            "forecast_issued_at"
        ).reset_index(drop=True)

        total = len(frame)

        required_rows = (
            minimum_train_samples
            + n_splits * test_samples
        )

        if total < required_rows:
            raise ForecastValidationError(
                "Insufficient rows for requested backtest"
            )

        folds: list[BacktestFold] = []

        for fold_number in range(n_splits):
            test_start = (
                minimum_train_samples
                + fold_number * test_samples
            )

            test_end = (
                test_start + test_samples
            )

            test_frame = frame.iloc[
                test_start:test_end
            ]

            test_start_time = test_frame[
                "forecast_issued_at"
            ].min()

            embargo_cutoff = (
                test_start_time
                - pd.Timedelta(
                    hours=embargo_hours
                )
            )

            candidate_train = frame.iloc[
                :test_start
            ]

            valid_train = candidate_train[
                candidate_train["target_time"]
                < embargo_cutoff
            ]

            if valid_train.empty:
                raise ForecastValidationError(
                    "No valid training samples after embargo"
                )

            folds.append(
                BacktestFold(
                    fold_id=fold_number + 1,
                    train_indices=tuple(
                        valid_train.index.astype(int)
                    ),
                    test_indices=tuple(
                        range(test_start, test_end)
                    ),
                    train_end_time=(
                        valid_train["target_time"]
                        .max()
                        .to_pydatetime()
                    ),
                    test_start_time=(
                        test_start_time.to_pydatetime()
                    ),
                )
            )

        return tuple(folds)


# ============================================================
# MODEL REGRESSION COMPARISON
# ============================================================

@dataclass(frozen=True)
class ForecastRegressionComparison:
    metric_name: str

    baseline_value: float
    candidate_value: float

    delta: float

    lower_is_better: bool

    regressed: bool


class ForecastModelRegressionEvaluator:
    """
    Compares the same metric evaluated on
    identical observations and forecast keys.

    This class compares numeric metrics only.
    Dataset identity must be checked upstream.
    """

    def compare(
        self,
        *,
        metric_name: str,
        baseline_value: float,
        candidate_value: float,
        lower_is_better: bool = True,
        tolerance: float = 0.0,
    ) -> ForecastRegressionComparison:
        if tolerance < 0:
            raise ForecastValidationError(
                "tolerance cannot be negative"
            )

        for name, value in (
            ("baseline_value", baseline_value),
            ("candidate_value", candidate_value),
        ):
            if not math.isfinite(value):
                raise ForecastValidationError(
                    f"{name} must be finite"
                )

        delta = (
            candidate_value - baseline_value
        )

        regressed = (
            delta > tolerance
            if lower_is_better
            else delta < -tolerance
        )

        return ForecastRegressionComparison(
            metric_name=metric_name,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            delta=delta,
            lower_is_better=lower_is_better,
            regressed=regressed,
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class ForecastEvaluationReportExporter:
    @staticmethod
    def to_dict(
        report: ForecastEvaluationReport,
    ) -> dict[str, Any]:
        return to_serializable(
            report
        )

    @classmethod
    def to_json(
        cls,
        report: ForecastEvaluationReport,
        *,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            cls.to_dict(report),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def save_json(
        cls,
        report: ForecastEvaluationReport,
        destination: str | Path,
    ) -> Path:
        path = Path(
            destination
        )

        if path.suffix.lower() != ".json":
            raise ForecastExportError(
                "Evaluation report must use .json"
            )

        try:
            with path.open(
                "x",
                encoding="utf-8",
            ) as file:
                file.write(
                    cls.to_json(report)
                )

        except OSError as exc:
            raise ForecastExportError(
                "Unable to save evaluation report"
            ) from exc

        return path


# ============================================================
# SERVICE LAYER
# ============================================================

class ForecastEvaluationService:
    """
    Application-facing forecasting evaluation.

    Integrations:
        backend/ai/ml/rainfall_forecasting.py
        backend/ai/ml/river_level_prediction.py
        backend/ai/ml/flood_risk_prediction.py
        backend/ai/ml/flood_severity.py
        backend/ai/ml/shelter_demand.py
        backend/ai/ml/model_evaluation.py
        backend/ai/evaluation/benchmark_runner.py
        backend/mlops/
    """

    def __init__(
        self,
        engine: ForecastEvaluationEngine | None = None,
    ) -> None:
        self.engine = (
            engine or ForecastEvaluationEngine()
        )

        self.backtesting = (
            ChronologicalBacktestSplitter()
        )

        self.regression_comparison = (
            ForecastModelRegressionEvaluator()
        )

    def evaluate_forecast(
        self,
        dataframe: pd.DataFrame,
        *,
        task: ForecastTask,
        group_column: str | None = None,
    ) -> ForecastEvaluationReport:
        return self.engine.evaluate(
            dataframe,
            task=task,
            group_column=group_column,
        )

    def evaluate_rainfall(
        self,
        dataframe: pd.DataFrame,
        *,
        group_column: str | None = None,
    ) -> ForecastEvaluationReport:
        return self.evaluate_forecast(
            dataframe,
            task=ForecastTask.RAINFALL,
            group_column=group_column,
        )

    def evaluate_river_level(
        self,
        dataframe: pd.DataFrame,
        *,
        group_column: str | None = None,
    ) -> ForecastEvaluationReport:
        return self.evaluate_forecast(
            dataframe,
            task=ForecastTask.RIVER_LEVEL,
            group_column=group_column,
        )

    def evaluate_flood_risk(
        self,
        dataframe: pd.DataFrame,
        *,
        group_column: str | None = None,
    ) -> ForecastEvaluationReport:
        return self.evaluate_forecast(
            dataframe,
            task=ForecastTask.FLOOD_RISK,
            group_column=group_column,
        )

    def evaluate_shelter_demand(
        self,
        dataframe: pd.DataFrame,
        *,
        group_column: str | None = None,
    ) -> ForecastEvaluationReport:
        return self.evaluate_forecast(
            dataframe,
            task=ForecastTask.SHELTER_DEMAND,
            group_column=group_column,
        )


# ============================================================
# FACTORIES
# ============================================================

def create_forecast_evaluation_engine(
    config: ForecastEvaluationConfig | None = None,
) -> ForecastEvaluationEngine:
    return ForecastEvaluationEngine(
        config=config
    )


def create_forecast_evaluation_service(
    config: ForecastEvaluationConfig | None = None,
) -> ForecastEvaluationService:
    return ForecastEvaluationService(
        engine=create_forecast_evaluation_engine(
            config=config
        )
    )


__all__ = [
    "ForecastEvaluationError",
    "ForecastValidationError",
    "ForecastAlignmentError",
    "ForecastExportError",
    "ForecastTask",
    "ForecastEvaluationStatus",
    "ForecastMetric",
    "ForecastRiskLevel",
    "ForecastEvaluationConfig",
    "RegressionForecastMetrics",
    "IntervalForecastMetrics",
    "ClassificationForecastMetrics",
    "CalibrationBin",
    "CalibrationReport",
    "ForecastGroupResult",
    "ForecastEvaluationReport",
    "RegressionForecastEvaluator",
    "PredictionIntervalEvaluator",
    "FloodRiskClassificationEvaluator",
    "ForecastCalibrationEvaluator",
    "ForecastAlignmentValidator",
    "ForecastGroupEvaluator",
    "ForecastQualityGateEvaluator",
    "ForecastEvaluationEngine",
    "BacktestFold",
    "ChronologicalBacktestSplitter",
    "ForecastRegressionComparison",
    "ForecastModelRegressionEvaluator",
    "ForecastEvaluationReportExporter",
    "ForecastEvaluationService",
    "create_forecast_evaluation_engine",
    "create_forecast_evaluation_service",
]
