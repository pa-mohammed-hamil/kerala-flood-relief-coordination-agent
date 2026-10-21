"""Placeholder module: ai/deep_learning/satellite_image_model.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Multivariate LSTM Flood Forecasting Model

File:
    backend/ai/deep_learning/lstm_flood_model.py

Python:
    3.11+

Dependencies:
    numpy
    pandas
    scikit-learn
    torch

Capabilities:
    - Multivariate sensor time-series forecasting
    - LSTM and bidirectional LSTM
    - Rainfall / river / dam feature support
    - Configurable input sequence length
    - Multi-horizon point forecasting
    - Chronological dataset splitting
    - Training-only feature scaling
    - Missing-value handling
    - Gradient clipping
    - Early stopping
    - MAE / RMSE / R2 evaluation
    - Residual-based uncertainty intervals
    - Checkpoint save and load
    - CPU and CUDA inference
    - Stale-data rejection
    - Advisory-only flood forecasts

Safety:
    This model does not issue evacuation orders,
    dispatch responders, or operate dams.

    Forecasts must be validated independently
    before use in emergency decision support.
"""

from __future__ import annotations

import copy
import math
import random
import uuid

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# EXCEPTIONS
# ============================================================

class LSTMFloodError(Exception):
    """Base LSTM flood model exception."""


class LSTMFloodValidationError(LSTMFloodError):
    """Invalid time-series input."""


class LSTMFloodTrainingError(LSTMFloodError):
    """Training failure."""


class LSTMFloodNotReadyError(LSTMFloodError):
    """Model has not been trained or loaded."""


class LSTMFloodCheckpointError(LSTMFloodError):
    """Checkpoint failure."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if name not in ("cpu", "cuda"):
        raise LSTMFloodValidationError(
            "device must be auto, cpu, or cuda"
        )

    if name == "cuda" and not torch.cuda.is_available():
        raise LSTMFloodValidationError(
            "CUDA is unavailable"
        )

    return torch.device(name)


def validate_fraction(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        raise LSTMFloodValidationError(
            f"{name} must be between 0 and 1"
        )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class LSTMFloodConfig:
    timestamp_column: str = "timestamp"

    feature_columns: tuple[str, ...] = (
        "rainfall_mm",
        "river_level_m",
        "dam_level_percent",
        "temperature_c",
        "humidity_percent",
    )

    target_column: str = "river_level_m"

    sequence_length: int = 24

    forecast_horizon: int = 6

    hidden_size: int = 128

    number_of_layers: int = 2

    dropout: float = 0.20

    bidirectional: bool = False

    batch_size: int = 32

    epochs: int = 50

    learning_rate: float = 0.001

    weight_decay: float = 0.0001

    gradient_clip_norm: float = 1.0

    early_stopping_patience: int = 8

    train_fraction: float = 0.70

    validation_fraction: float = 0.15

    interval_coverage: float = 0.90

    expected_frequency_minutes: int = 60

    maximum_data_age_minutes: int = 120

    device: str = "auto"

    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise LSTMFloodValidationError(
                "feature_columns cannot be empty"
            )

        if len(set(self.feature_columns)) != len(
            self.feature_columns
        ):
            raise LSTMFloodValidationError(
                "Duplicate feature columns"
            )

        if self.target_column not in self.feature_columns:
            raise LSTMFloodValidationError(
                "target_column must be included "
                "in feature_columns"
            )

        for name in (
            "sequence_length",
            "forecast_horizon",
            "hidden_size",
            "number_of_layers",
            "batch_size",
            "epochs",
            "early_stopping_patience",
            "expected_frequency_minutes",
            "maximum_data_age_minutes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise LSTMFloodValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "train_fraction",
            "validation_fraction",
            "interval_coverage",
        ):
            validate_fraction(
                getattr(self, name),
                name,
            )

        if (
            self.train_fraction
            + self.validation_fraction
            >= 1
        ):
            raise LSTMFloodValidationError(
                "Train and validation fractions "
                "must sum to less than 1"
            )

        if not 0 <= self.dropout < 1:
            raise LSTMFloodValidationError(
                "dropout must be in [0, 1)"
            )

        for name in (
            "learning_rate",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise LSTMFloodValidationError(
                    f"{name} must be positive and finite"
                )

        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise LSTMFloodValidationError(
                "weight_decay must be nonnegative"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class LSTMFloodEpochMetrics:
    epoch: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class LSTMFloodTrainingResult:
    training_id: str
    epochs_completed: int
    best_validation_loss: float
    training_samples: int
    validation_samples: int
    test_samples: int
    history: tuple[LSTMFloodEpochMetrics, ...]
    trained_at: datetime


@dataclass(frozen=True)
class LSTMFloodEvaluationResult:
    sample_count: int
    horizon_steps: int
    mae: float
    rmse: float
    r2: float | None
    interval_coverage: float
    per_horizon_mae: tuple[float, ...]


@dataclass(frozen=True)
class LSTMFloodForecastPoint:
    timestamp: datetime
    horizon_step: int
    predicted_value: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class LSTMFloodForecast:
    forecast_id: str
    target_column: str
    observed_until: datetime
    points: tuple[LSTMFloodForecastPoint, ...]
    requires_human_review: bool
    generated_at: datetime
    warnings: tuple[str, ...]


# ============================================================
# LSTM NETWORK
# ============================================================

class FloodLSTMNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        config: LSTMFloodConfig,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.number_of_layers,
            dropout=(
                config.dropout
                if config.number_of_layers > 1
                else 0.0
            ),
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        output_size = (
            config.hidden_size
            * (2 if config.bidirectional else 1)
        )

        self.head = nn.Sequential(
            nn.Linear(output_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_size,
                config.forecast_horizon,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = self.lstm(inputs)

        final_step = output[:, -1, :]

        return self.head(final_step)


# ============================================================
# DATA VALIDATION
# ============================================================

class FloodTimeSeriesValidator:
    def __init__(
        self,
        config: LSTMFloodConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        required = {
            self.config.timestamp_column,
            *self.config.feature_columns,
        }

        missing = required - set(dataframe.columns)

        if missing:
            raise LSTMFloodValidationError(
                f"Missing columns: {sorted(missing)}"
            )

        frame = dataframe[
            [
                self.config.timestamp_column,
                *self.config.feature_columns,
            ]
        ].copy()

        frame[self.config.timestamp_column] = (
            pd.to_datetime(
                frame[self.config.timestamp_column],
                utc=True,
                errors="coerce",
            )
        )

        if frame[
            self.config.timestamp_column
        ].isna().any():
            raise LSTMFloodValidationError(
                "Invalid timestamps"
            )

        frame = frame.sort_values(
            self.config.timestamp_column
        ).reset_index(drop=True)

        if frame[
            self.config.timestamp_column
        ].duplicated().any():
            raise LSTMFloodValidationError(
                "Duplicate timestamps"
            )

        for column in self.config.feature_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            frame[column] = frame[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

        # Preserve chronology; missing values
        # must not be filled using future rows.
        frame[list(self.config.feature_columns)] = (
            frame[list(self.config.feature_columns)]
            .ffill()
        )

        if frame[
            list(self.config.feature_columns)
        ].isna().any().any():
            raise LSTMFloodValidationError(
                "Leading missing sensor values "
                "cannot be filled safely"
            )

        expected = pd.Timedelta(
            minutes=self.config.expected_frequency_minutes
        )

        differences = frame[
            self.config.timestamp_column
        ].diff().dropna()

        if not differences.eq(expected).all():
            raise LSTMFloodValidationError(
                "Time series must have a regular "
                "sampling interval; resample "
                "upstream with explicit gap rules"
            )

        minimum_rows = (
            self.config.sequence_length
            + self.config.forecast_horizon
            + 10
        )

        if len(frame) < minimum_rows:
            raise LSTMFloodValidationError(
                f"At least {minimum_rows} rows required"
            )

        return frame


# ============================================================
# SEQUENCE PREPROCESSOR
# ============================================================

class FloodSequencePreprocessor:
    """
    Fits scalers on training observations only.

    Targets:
        y[t+1], ..., y[t+H]

    Inputs:
        X[t-L+1], ..., X[t]

    No future observations enter the
    corresponding input window.
    """

    def __init__(
        self,
        config: LSTMFloodConfig,
    ) -> None:
        self.config = config

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.fitted = False

    def fit(
        self,
        training_frame: pd.DataFrame,
    ) -> None:
        features = training_frame[
            list(self.config.feature_columns)
        ].to_numpy(dtype=np.float64)

        targets = training_frame[
            [self.config.target_column]
        ].to_numpy(dtype=np.float64)

        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets)

        self.fitted = True

    def transform_features(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        if not self.fitted:
            raise LSTMFloodNotReadyError(
                "Preprocessor is not fitted"
            )

        return self.feature_scaler.transform(
            frame[list(self.config.feature_columns)]
            .to_numpy(dtype=np.float64)
        ).astype(np.float32)

    def transform_targets(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        if not self.fitted:
            raise LSTMFloodNotReadyError(
                "Preprocessor is not fitted"
            )

        return self.target_scaler.transform(
            frame[[self.config.target_column]]
            .to_numpy(dtype=np.float64)
        ).reshape(-1).astype(np.float32)

    def inverse_targets(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        shape = values.shape

        restored = (
            self.target_scaler.inverse_transform(
                values.reshape(-1, 1)
            )
        )

        return restored.reshape(shape)


# ============================================================
# CHRONOLOGICAL DATASET BUILDER
# ============================================================

@dataclass
class FloodDatasetSplits:
    train_x: np.ndarray
    train_y: np.ndarray

    val_x: np.ndarray
    val_y: np.ndarray

    test_x: np.ndarray
    test_y: np.ndarray


class FloodSequenceDatasetBuilder:
    def __init__(
        self,
        config: LSTMFloodConfig,
        preprocessor: FloodSequencePreprocessor,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor

    def build(
        self,
        frame: pd.DataFrame,
    ) -> FloodDatasetSplits:
        n = len(frame)

        train_end = int(
            n * self.config.train_fraction
        )

        validation_end = int(
            n * (
                self.config.train_fraction
                + self.config.validation_fraction
            )
        )

        self.preprocessor.fit(
            frame.iloc[:train_end]
        )

        features = (
            self.preprocessor.transform_features(
                frame
            )
        )

        targets = (
            self.preprocessor.transform_targets(
                frame
            )
        )

        sequence_length = (
            self.config.sequence_length
        )

        horizon = (
            self.config.forecast_horizon
        )

        buckets: dict[str, list] = {
            "train_x": [],
            "train_y": [],
            "val_x": [],
            "val_y": [],
            "test_x": [],
            "test_y": [],
        }

        for start in range(
            0,
            n - sequence_length - horizon + 1,
        ):
            input_end = (
                start + sequence_length
            )

            target_end = (
                input_end + horizon
            )

            x = features[
                start:input_end
            ]

            y = targets[
                input_end:target_end
            ]

            # Entire target horizon must
            # remain within one data split.
            if target_end <= train_end:
                split = "train"

            elif (
                input_end >= train_end
                and target_end <= validation_end
            ):
                split = "val"

            elif input_end >= validation_end:
                split = "test"

            else:
                continue

            buckets[f"{split}_x"].append(x)
            buckets[f"{split}_y"].append(y)

        arrays: dict[str, np.ndarray] = {}

        for key, values in buckets.items():
            if not values:
                raise LSTMFloodValidationError(
                    f"No samples for {key}; "
                    "increase dataset size"
                )

            arrays[key] = np.stack(
                values
            ).astype(np.float32)

        return FloodDatasetSplits(
            **arrays
        )


# ============================================================
# TRAINING ENGINE
# ============================================================

class LSTMFloodTrainer:
    def __init__(
        self,
        config: LSTMFloodConfig,
        model: FloodLSTMNetwork,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

    def _loader(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def _epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        training = optimizer is not None

        self.model.train(training)

        criterion = nn.MSELoss()

        total_loss = 0.0
        total_samples = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(training):
                predictions = self.model(x)

                loss = criterion(
                    predictions,
                    y,
                )

                if training:
                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                    optimizer.step()

            batch_size = x.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        if total_samples == 0:
            raise LSTMFloodTrainingError(
                "Empty training loader"
            )

        return total_loss / total_samples

    def train(
        self,
        splits: FloodDatasetSplits,
    ) -> LSTMFloodTrainingResult:
        train_loader = self._loader(
            splits.train_x,
            splits.train_y,
            shuffle=True,
        )

        val_loader = self._loader(
            splits.val_x,
            splits.val_y,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_loss = math.inf
        best_state = None
        patience = 0

        history = []

        for epoch in range(
            1,
            self.config.epochs + 1,
        ):
            train_loss = self._epoch(
                train_loader,
                optimizer,
            )

            with torch.no_grad():
                val_loss = self._epoch(
                    val_loader,
                    None,
                )

            history.append(
                LSTMFloodEpochMetrics(
                    epoch=epoch,
                    training_loss=train_loss,
                    validation_loss=val_loss,
                )
            )

            if val_loss < best_loss:
                best_loss = val_loss

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                patience = 0

            else:
                patience += 1

            if (
                patience
                >= self.config.early_stopping_patience
            ):
                break

        if best_state is None:
            raise LSTMFloodTrainingError(
                "No valid training checkpoint"
            )

        self.model.load_state_dict(
            best_state
        )

        self.model.eval()

        return LSTMFloodTrainingResult(
            training_id=new_id(),
            epochs_completed=len(history),
            best_validation_loss=best_loss,
            training_samples=len(splits.train_x),
            validation_samples=len(splits.val_x),
            test_samples=len(splits.test_x),
            history=tuple(history),
            trained_at=utc_now(),
        )


# ============================================================
# MODEL EVALUATOR
# ============================================================

class LSTMFloodEvaluator:
    def __init__(
        self,
        model: FloodLSTMNetwork,
        device: torch.device,
        preprocessor: FloodSequencePreprocessor,
        config: LSTMFloodConfig,
    ) -> None:
        self.model = model
        self.device = device
        self.preprocessor = preprocessor
        self.config = config

    @torch.inference_mode()
    def predict_array(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        self.model.eval()

        results = []

        for start in range(
            0,
            len(x),
            self.config.batch_size,
        ):
            batch = torch.from_numpy(
                x[
                    start:
                    start + self.config.batch_size
                ]
            ).to(self.device)

            prediction = self.model(
                batch
            )

            results.append(
                prediction.cpu().numpy()
            )

        if not results:
            raise LSTMFloodValidationError(
                "Prediction array is empty"
            )

        scaled = np.concatenate(
            results,
            axis=0,
        )

        return self.preprocessor.inverse_targets(
            scaled
        )

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        interval_half_widths: np.ndarray,
    ) -> LSTMFloodEvaluationResult:
        predictions = self.predict_array(x)

        actual = (
            self.preprocessor.inverse_targets(
                y
            )
        )

        errors = (
            actual - predictions
        )

        mae = mean_absolute_error(
            actual.reshape(-1),
            predictions.reshape(-1),
        )

        rmse = math.sqrt(
            mean_squared_error(
                actual.reshape(-1),
                predictions.reshape(-1),
            )
        )

        flat_actual = actual.reshape(-1)
        flat_predicted = predictions.reshape(-1)

        r2 = (
            float(r2_score(
                flat_actual,
                flat_predicted,
            ))
            if len(flat_actual) > 1
            else None
        )

        covered = (
            np.abs(errors)
            <= interval_half_widths[None, :]
        )

        per_horizon_mae = tuple(
            float(
                np.mean(
                    np.abs(
                        errors[:, step]
                    )
                )
            )
            for step in range(
                self.config.forecast_horizon
            )
        )

        return LSTMFloodEvaluationResult(
            sample_count=len(actual),
            horizon_steps=(
                self.config.forecast_horizon
            ),
            mae=float(mae),
            rmse=float(rmse),
            r2=r2,
            interval_coverage=float(
                np.mean(covered)
            ),
            per_horizon_mae=per_horizon_mae,
        )


# ============================================================
# FORECASTING SERVICE
# ============================================================

class LSTMFloodForecastingService:
    def __init__(
        self,
        config: LSTMFloodConfig | None = None,
    ) -> None:
        self.config = (
            config or LSTMFloodConfig()
        )

        set_seed(
            self.config.random_seed
        )

        self.device = get_device(
            self.config.device
        )

        self.validator = FloodTimeSeriesValidator(
            self.config
        )

        self.preprocessor = (
            FloodSequencePreprocessor(
                self.config
            )
        )

        self.model = FloodLSTMNetwork(
            input_size=len(
                self.config.feature_columns
            ),
            config=self.config,
        ).to(self.device)

        self.trainer = LSTMFloodTrainer(
            self.config,
            self.model,
            self.device,
        )

        self.evaluator = LSTMFloodEvaluator(
            self.model,
            self.device,
            self.preprocessor,
            self.config,
        )

        self.interval_half_widths: (
            np.ndarray | None
        ) = None

        self.trained = False

    def train(
        self,
        dataframe: pd.DataFrame,
    ) -> LSTMFloodTrainingResult:
        frame = self.validator.validate(
            dataframe
        )

        builder = FloodSequenceDatasetBuilder(
            self.config,
            self.preprocessor,
        )

        splits = builder.build(
            frame
        )

        result = self.trainer.train(
            splits
        )

        # Validation residuals provide
        # horizon-specific empirical intervals.
        validation_predictions = (
            self.evaluator.predict_array(
                splits.val_x
            )
        )

        validation_actual = (
            self.preprocessor.inverse_targets(
                splits.val_y
            )
        )

        residuals = np.abs(
            validation_actual
            - validation_predictions
        )

        self.interval_half_widths = np.quantile(
            residuals,
            self.config.interval_coverage,
            axis=0,
        )

        self.trained = True

        return result

    def evaluate(
        self,
        dataframe: pd.DataFrame,
    ) -> LSTMFloodEvaluationResult:
        if not self.trained:
            raise LSTMFloodNotReadyError(
                "Model is not trained"
            )

        if self.interval_half_widths is None:
            raise LSTMFloodNotReadyError(
                "Forecast intervals unavailable"
            )

        # Evaluation requires a held-out test
        # dataset. Do not re-fit scalers here.
        frame = self.validator.validate(
            dataframe
        )

        features = (
            self.preprocessor.transform_features(
                frame
            )
        )

        targets = (
            self.preprocessor.transform_targets(
                frame
            )
        )

        x = []
        y = []

        length = (
            self.config.sequence_length
        )

        horizon = (
            self.config.forecast_horizon
        )

        for start in range(
            len(frame) - length - horizon + 1
        ):
            end = start + length

            x.append(
                features[start:end]
            )

            y.append(
                targets[end:end + horizon]
            )

        if not x:
            raise LSTMFloodValidationError(
                "Not enough evaluation samples"
            )

        return self.evaluator.evaluate(
            np.stack(x).astype(np.float32),
            np.stack(y).astype(np.float32),
            self.interval_half_widths,
        )

    @torch.inference_mode()
    def forecast(
        self,
        dataframe: pd.DataFrame,
    ) -> LSTMFloodForecast:
        if not self.trained:
            raise LSTMFloodNotReadyError(
                "Train or load a model first"
            )

        if self.interval_half_widths is None:
            raise LSTMFloodNotReadyError(
                "Forecast intervals unavailable"
            )

        frame = self.validator.validate(
            dataframe
        )

        last_timestamp = (
            frame[
                self.config.timestamp_column
            ].iloc[-1].to_pydatetime()
        )

        age_minutes = (
            utc_now() - last_timestamp
        ).total_seconds() / 60

        if age_minutes < -1:
            raise LSTMFloodValidationError(
                "Latest observation is in the future"
            )

        if (
            age_minutes
            > self.config.maximum_data_age_minutes
        ):
            raise LSTMFloodValidationError(
                "Sensor observations are stale"
            )

        features = (
            self.preprocessor.transform_features(
                frame
            )
        )

        window = features[
            -self.config.sequence_length:
        ]

        tensor = torch.from_numpy(
            window[None, :, :]
        ).to(self.device)

        self.model.eval()

        scaled = (
            self.model(tensor)
            .cpu()
            .numpy()
        )

        predictions = (
            self.preprocessor.inverse_targets(
                scaled
            )[0]
        )

        points = []

        for step, value in enumerate(
            predictions,
            start=1,
        ):
            timestamp = (
                last_timestamp
                + pd.Timedelta(
                    minutes=(
                        step
                        * self.config.expected_frequency_minutes
                    )
                )
            )

            width = float(
                self.interval_half_widths[
                    step - 1
                ]
            )

            points.append(
                LSTMFloodForecastPoint(
                    timestamp=timestamp,
                    horizon_step=step,
                    predicted_value=float(value),
                    lower_bound=float(
                        value - width
                    ),
                    upper_bound=float(
                        value + width
                    ),
                )
            )

        return LSTMFloodForecast(
            forecast_id=new_id(),
            target_column=(
                self.config.target_column
            ),
            observed_until=last_timestamp,
            points=tuple(points),
            requires_human_review=True,
            generated_at=utc_now(),
            warnings=(
                "Advisory forecast only",
                "Intervals are empirical and not guaranteed",
                "Verify sensor data and official flood guidance",
            ),
        )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> Path:
        if not self.trained:
            raise LSTMFloodNotReadyError(
                "Model is not trained"
            )

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise LSTMFloodCheckpointError(
                "Checkpoint already exists"
            )

        payload = {
            "format_version": 1,
            "config": asdict(self.config),
            "model_state": self.model.state_dict(),
            "feature_mean": (
                self.preprocessor.feature_scaler.mean_
            ),
            "feature_scale": (
                self.preprocessor.feature_scaler.scale_
            ),
            "target_mean": (
                self.preprocessor.target_scaler.mean_
            ),
            "target_scale": (
                self.preprocessor.target_scaler.scale_
            ),
            "interval_half_widths": (
                self.interval_half_widths
            ),
        }

        torch.save(
            payload,
            destination,
        )

        return destination

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        checkpoint_path = Path(path)

        if not checkpoint_path.is_file():
            raise LSTMFloodCheckpointError(
                "Checkpoint does not exist"
            )

        payload = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        if payload.get("format_version") != 1:
            raise LSTMFloodCheckpointError(
                "Unsupported checkpoint version"
            )

        saved_config = payload.get(
            "config",
            {},
        )

        for key in (
            "feature_columns",
            "target_column",
            "sequence_length",
            "forecast_horizon",
            "hidden_size",
            "number_of_layers",
            "bidirectional",
        ):
            saved_value = saved_config.get(key)
            current_value = getattr(
                self.config,
                key,
            )

            if (
                key == "feature_columns"
                and saved_value is not None
            ):
                saved_value = tuple(saved_value)

            if saved_value != current_value:
                raise LSTMFloodCheckpointError(
                    f"Checkpoint config mismatch: {key}"
                )

        self.model.load_state_dict(
            payload["model_state"],
            strict=True,
        )

        feature_scaler = (
            self.preprocessor.feature_scaler
        )

        target_scaler = (
            self.preprocessor.target_scaler
        )

        feature_scaler.mean_ = np.asarray(
            payload["feature_mean"],
            dtype=np.float64,
        )

        feature_scaler.scale_ = np.asarray(
            payload["feature_scale"],
            dtype=np.float64,
        )

        feature_scaler.var_ = (
            feature_scaler.scale_ ** 2
        )

        feature_scaler.n_features_in_ = len(
            self.config.feature_columns
        )

        target_scaler.mean_ = np.asarray(
            payload["target_mean"],
            dtype=np.float64,
        )

        target_scaler.scale_ = np.asarray(
            payload["target_scale"],
            dtype=np.float64,
        )

        target_scaler.var_ = (
            target_scaler.scale_ ** 2
        )

        target_scaler.n_features_in_ = 1

        self.interval_half_widths = np.asarray(
            payload["interval_half_widths"],
            dtype=np.float64,
        )

        if (
            self.interval_half_widths.shape
            != (self.config.forecast_horizon,)
        ):
            raise LSTMFloodCheckpointError(
                "Invalid interval shape"
            )

        self.preprocessor.fitted = True

        self.model.to(self.device)
        self.model.eval()

        self.trained = True


# ============================================================
# FACTORY
# ============================================================

def create_lstm_flood_service(
    config: LSTMFloodConfig | None = None,
) -> LSTMFloodForecastingService:
    return LSTMFloodForecastingService(
        config=config
    )


__all__ = [
    "LSTMFloodError",
    "LSTMFloodValidationError",
    "LSTMFloodTrainingError",
    "LSTMFloodNotReadyError",
    "LSTMFloodCheckpointError",
    "LSTMFloodConfig",
    "LSTMFloodEpochMetrics",
    "LSTMFloodTrainingResult",
    "LSTMFloodEvaluationResult",
    "LSTMFloodForecastPoint",
    "LSTMFloodForecast",
    "FloodLSTMNetwork",
    "FloodTimeSeriesValidator",
    "FloodSequencePreprocessor",
    "FloodDatasetSplits",
    "FloodSequenceDatasetBuilder",
    "LSTMFloodTrainer",
    "LSTMFloodEvaluator",
    "LSTMFloodForecastingService",
    "create_lstm_flood_service",
]
