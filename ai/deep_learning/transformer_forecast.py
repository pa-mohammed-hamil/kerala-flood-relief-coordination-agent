"""Placeholder module: ai/deep_learning/transformer_forecast.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Transformer Flood Forecasting Model

File:
    backend/ai/deep_learning/transformer_forecast.py

Python:
    3.11+

Dependencies:
    torch >= 2.2
    numpy
    pandas

Features:
    - Transformer encoder forecasting
    - Multi-head self-attention
    - Sinusoidal positional encoding
    - Multivariate time-series input
    - Multi-horizon direct prediction
    - Chronological train/validation/test split
    - Training-only normalization
    - Gradient clipping
    - Early stopping
    - Validation residual prediction intervals
    - MAE / RMSE / R2 / interval coverage
    - Trusted checkpoint persistence
    - CPU / CUDA support
    - Stale observation detection

Safety:
    Forecasts are advisory only.

    No automatic rescue dispatch,
    evacuation orders, or dam operations.
"""

from __future__ import annotations

import copy
import math
import random
import uuid

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class TransformerForecastError(Exception):
    pass


class TransformerValidationError(
    TransformerForecastError
):
    pass


class TransformerNotReadyError(
    TransformerForecastError
):
    pass


class TransformerCheckpointError(
    TransformerForecastError
):
    pass


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


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if value not in ("cpu", "cuda"):
        raise TransformerValidationError(
            "device must be auto, cpu, or cuda"
        )

    if (
        value == "cuda"
        and not torch.cuda.is_available()
    ):
        raise TransformerValidationError(
            "CUDA is unavailable"
        )

    return torch.device(value)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TransformerForecastConfig:
    timestamp_column: str = "timestamp"

    feature_columns: tuple[str, ...] = (
        "rainfall_mm",
        "river_level_m",
        "dam_level_percent",
        "temperature_c",
        "humidity_percent",
    )

    target_column: str = "river_level_m"

    sequence_length: int = 48
    forecast_horizon: int = 12

    model_dimension: int = 128
    attention_heads: int = 8
    encoder_layers: int = 3
    feedforward_dimension: int = 256

    dropout: float = 0.10

    batch_size: int = 32
    epochs: int = 40

    learning_rate: float = 0.0005
    weight_decay: float = 0.0001

    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 7

    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    interval_coverage: float = 0.90

    expected_frequency_minutes: int = 60
    maximum_data_age_minutes: int = 120

    device: str = "auto"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise TransformerValidationError(
                "feature_columns cannot be empty"
            )

        if len(set(self.feature_columns)) != len(
            self.feature_columns
        ):
            raise TransformerValidationError(
                "Duplicate feature columns"
            )

        if self.target_column not in self.feature_columns:
            raise TransformerValidationError(
                "Target must be included in features"
            )

        for name in (
            "sequence_length",
            "forecast_horizon",
            "model_dimension",
            "attention_heads",
            "encoder_layers",
            "feedforward_dimension",
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
                raise TransformerValidationError(
                    f"{name} must be positive"
                )

        if (
            self.model_dimension
            % self.attention_heads
        ):
            raise TransformerValidationError(
                "model_dimension must be divisible "
                "by attention_heads"
            )

        if not 0 <= self.dropout < 1:
            raise TransformerValidationError(
                "dropout must be in [0,1)"
            )

        if not (
            0 < self.train_fraction < 1
            and 0 < self.validation_fraction < 1
            and self.train_fraction
            + self.validation_fraction < 1
        ):
            raise TransformerValidationError(
                "Invalid split fractions"
            )

        if not 0 < self.interval_coverage < 1:
            raise TransformerValidationError(
                "interval_coverage must be in (0,1)"
            )

        for name in (
            "learning_rate",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or value <= 0
            ):
                raise TransformerValidationError(
                    f"{name} must be positive and finite"
                )

        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise TransformerValidationError(
                "weight_decay must be nonnegative"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class TransformerEpochMetrics:
    epoch: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class TransformerTrainingResult:
    training_id: str
    epochs_completed: int
    best_validation_loss: float
    training_samples: int
    validation_samples: int
    test_samples: int
    history: tuple[TransformerEpochMetrics, ...]
    trained_at: datetime


@dataclass(frozen=True)
class TransformerEvaluationResult:
    sample_count: int
    mae: float
    rmse: float
    r2: float | None
    interval_coverage: float
    per_horizon_mae: tuple[float, ...]


@dataclass(frozen=True)
class TransformerForecastPoint:
    timestamp: datetime
    horizon_step: int
    predicted_value: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class TransformerForecast:
    forecast_id: str
    target_column: str
    observed_until: datetime
    points: tuple[TransformerForecastPoint, ...]
    requires_human_review: bool
    warnings: tuple[str, ...]
    generated_at: datetime


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        maximum_length: int = 4096,
    ) -> None:
        super().__init__()

        positions = torch.arange(
            maximum_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        dimensions = torch.arange(
            0,
            model_dimension,
            2,
            dtype=torch.float32,
        )

        div_term = torch.exp(
            dimensions
            * (
                -math.log(10000.0)
                / model_dimension
            )
        )

        encoding = torch.zeros(
            maximum_length,
            model_dimension,
        )

        encoding[:, 0::2] = torch.sin(
            positions * div_term
        )

        encoding[:, 1::2] = torch.cos(
            positions * div_term[
                :encoding[:, 1::2].shape[1]
            ]
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise TransformerValidationError(
                "Sequence exceeds positional limit"
            )

        return (
            x
            + self.encoding[
                :, :x.shape[1], :
            ].to(dtype=x.dtype)
        )


# ============================================================
# TRANSFORMER FORECAST NETWORK
# ============================================================

class FloodTransformerNetwork(nn.Module):
    """
    Direct multi-horizon Transformer model.

    Input:
        B x sequence_length x feature_count

    Output:
        B x forecast_horizon

    Only historical observations are used.
    No future targets enter the model.
    """

    def __init__(
        self,
        config: TransformerForecastConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.input_projection = nn.Linear(
            len(config.feature_columns),
            config.model_dimension,
        )

        self.positional_encoding = (
            SinusoidalPositionalEncoding(
                config.model_dimension,
                maximum_length=max(
                    4096,
                    config.sequence_length,
                ),
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=config.model_dimension,
                nhead=config.attention_heads,
                dim_feedforward=(
                    config.feedforward_dimension
                ),
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(
                config.model_dimension
            ),
            nn.Linear(
                config.model_dimension,
                config.model_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.model_dimension,
                config.forecast_horizon,
            ),
        )

    def forward(
        self,
        historical: torch.Tensor,
    ) -> torch.Tensor:
        if historical.ndim != 3:
            raise TransformerValidationError(
                "Input must have shape B,T,F"
            )

        if historical.shape[1] != (
            self.config.sequence_length
        ):
            raise TransformerValidationError(
                "Unexpected sequence length"
            )

        x = self.input_projection(
            historical
        )

        x = self.positional_encoding(
            x
        )

        encoded = self.encoder(
            x
        )

        # All input positions are historical.
        context = encoded[:, -1, :]

        return self.output_head(
            context
        )


# ============================================================
# DATA VALIDATOR
# ============================================================

class TransformerTimeSeriesValidator:
    def __init__(
        self,
        config: TransformerForecastConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        dataframe: pd.DataFrame,
        *,
        minimum_rows: int | None = None,
    ) -> pd.DataFrame:
        required = {
            self.config.timestamp_column,
            *self.config.feature_columns,
        }

        missing = required - set(dataframe.columns)

        if missing:
            raise TransformerValidationError(
                f"Missing columns: {sorted(missing)}"
            )

        frame = dataframe[
            [
                self.config.timestamp_column,
                *self.config.feature_columns,
            ]
        ].copy()

        column = self.config.timestamp_column

        frame[column] = pd.to_datetime(
            frame[column],
            utc=True,
            errors="coerce",
        )

        if frame[column].isna().any():
            raise TransformerValidationError(
                "Invalid timestamps"
            )

        frame = frame.sort_values(
            column
        ).reset_index(drop=True)

        if frame[column].duplicated().any():
            raise TransformerValidationError(
                "Duplicate timestamps"
            )

        for name in self.config.feature_columns:
            frame[name] = pd.to_numeric(
                frame[name],
                errors="coerce",
            )

        values = frame[
            list(self.config.feature_columns)
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise TransformerValidationError(
                "Missing or non-finite sensor values"
            )

        expected = pd.Timedelta(
            minutes=(
                self.config.expected_frequency_minutes
            )
        )

        intervals = frame[column].diff().dropna()

        if not intervals.eq(expected).all():
            raise TransformerValidationError(
                "Irregular sampling interval; "
                "resample upstream with "
                "documented gap handling"
            )

        minimum = (
            minimum_rows
            if minimum_rows is not None
            else self.config.sequence_length
        )

        if len(frame) < minimum:
            raise TransformerValidationError(
                f"At least {minimum} rows required"
            )

        return frame


# ============================================================
# STANDARDIZATION
# ============================================================

class TransformerFeatureScaler:
    """
    Feature and target statistics are fitted
    exclusively on training observations.
    """

    def __init__(
        self,
        config: TransformerForecastConfig,
    ) -> None:
        self.config = config

        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

        self.target_mean: float | None = None
        self.target_std: float | None = None

        self.fitted = False

    def fit(
        self,
        training_frame: pd.DataFrame,
    ) -> None:
        features = training_frame[
            list(self.config.feature_columns)
        ].to_numpy(dtype=np.float64)

        target = training_frame[
            self.config.target_column
        ].to_numpy(dtype=np.float64)

        self.feature_mean = features.mean(axis=0)

        self.feature_std = np.maximum(
            features.std(axis=0),
            1e-6,
        )

        self.target_mean = float(
            target.mean()
        )

        self.target_std = float(
            max(target.std(), 1e-6)
        )

        self.fitted = True

    def _require_ready(self) -> None:
        if not self.fitted:
            raise TransformerNotReadyError(
                "Feature scaler is not fitted"
            )

    def transform_features(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        self._require_ready()

        return (
            (values - self.feature_mean)
            / self.feature_std
        ).astype(np.float32)

    def transform_target(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        self._require_ready()

        return (
            (values - self.target_mean)
            / self.target_std
        ).astype(np.float32)

    def inverse_target(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        self._require_ready()

        return (
            values * self.target_std
            + self.target_mean
        )


# ============================================================
# CHRONOLOGICAL DATASET BUILDER
# ============================================================

@dataclass
class TransformerDatasetSplits:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


class TransformerDatasetBuilder:
    def __init__(
        self,
        config: TransformerForecastConfig,
        scaler: TransformerFeatureScaler,
    ) -> None:
        self.config = config
        self.scaler = scaler

    def build(
        self,
        frame: pd.DataFrame,
    ) -> TransformerDatasetSplits:
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

        self.scaler.fit(
            frame.iloc[:train_end]
        )

        raw_features = frame[
            list(self.config.feature_columns)
        ].to_numpy(dtype=np.float64)

        raw_target = frame[
            self.config.target_column
        ].to_numpy(dtype=np.float64)

        features = (
            self.scaler.transform_features(
                raw_features
            )
        )

        target = (
            self.scaler.transform_target(
                raw_target
            )
        )

        length = self.config.sequence_length
        horizon = self.config.forecast_horizon

        buckets = {
            name: {
                "x": [],
                "y": [],
            }
            for name in (
                "train",
                "validation",
                "test",
            )
        }

        for start in range(
            n - length - horizon + 1
        ):
            input_end = start + length
            target_end = input_end + horizon

            if target_end <= train_end:
                split = "train"

            elif (
                input_end >= train_end
                and target_end <= validation_end
            ):
                split = "validation"

            elif input_end >= validation_end:
                split = "test"

            else:
                continue

            buckets[split]["x"].append(
                features[start:input_end]
            )

            buckets[split]["y"].append(
                target[input_end:target_end]
            )

        def make_dataset(
            name: str,
        ) -> TensorDataset:
            x = buckets[name]["x"]
            y = buckets[name]["y"]

            if not x:
                raise TransformerValidationError(
                    f"No samples in {name} split"
                )

            return TensorDataset(
                torch.from_numpy(
                    np.stack(x)
                ),
                torch.from_numpy(
                    np.stack(y)
                ),
            )

        return TransformerDatasetSplits(
            train=make_dataset("train"),
            validation=make_dataset("validation"),
            test=make_dataset("test"),
        )


# ============================================================
# TRAINING ENGINE
# ============================================================

class TransformerForecastTrainer:
    def __init__(
        self,
        config: TransformerForecastConfig,
        model: FloodTransformerNetwork,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.criterion = nn.MSELoss()

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        training = optimizer is not None

        self.model.train(training)

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

                loss = self.criterion(
                    predictions,
                    y,
                )

                if not torch.isfinite(loss):
                    raise TransformerForecastError(
                        "Non-finite training loss"
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
                float(loss.item())
                * batch_size
            )

            total_samples += batch_size

        if total_samples == 0:
            raise TransformerValidationError(
                "Empty training dataset"
            )

        return total_loss / total_samples

    def train(
        self,
        splits: TransformerDatasetSplits,
    ) -> TransformerTrainingResult:
        training_loader = DataLoader(
            splits.train,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        validation_loader = DataLoader(
            splits.validation,
            batch_size=self.config.batch_size,
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
            training_loss = self._run_epoch(
                training_loader,
                optimizer,
            )

            with torch.no_grad():
                validation_loss = self._run_epoch(
                    validation_loader,
                    None,
                )

            history.append(
                TransformerEpochMetrics(
                    epoch=epoch,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                )
            )

            if validation_loss < best_loss:
                best_loss = validation_loss

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
            raise TransformerForecastError(
                "Training produced no valid checkpoint"
            )

        self.model.load_state_dict(
            best_state
        )

        self.model.eval()

        return TransformerTrainingResult(
            training_id=new_id(),
            epochs_completed=len(history),
            best_validation_loss=best_loss,
            training_samples=len(splits.train),
            validation_samples=len(splits.validation),
            test_samples=len(splits.test),
            history=tuple(history),
            trained_at=utc_now(),
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

class TransformerForecastEvaluator:
    def __init__(
        self,
        config: TransformerForecastConfig,
        model: FloodTransformerNetwork,
        scaler: TransformerFeatureScaler,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.scaler = scaler
        self.device = device

    @torch.inference_mode()
    def predict_dataset(
        self,
        dataset: TensorDataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        predictions = []
        targets = []

        self.model.eval()

        for x, y in loader:
            output = self.model(
                x.to(self.device)
            )

            predictions.append(
                output.cpu().numpy()
            )

            targets.append(
                y.numpy()
            )

        if not predictions:
            raise TransformerValidationError(
                "Evaluation dataset is empty"
            )

        predicted = np.concatenate(
            predictions
        )

        actual = np.concatenate(
            targets
        )

        return (
            self.scaler.inverse_target(
                predicted
            ),
            self.scaler.inverse_target(
                actual
            ),
        )

    def evaluate(
        self,
        dataset: TensorDataset,
        interval_widths: np.ndarray,
    ) -> TransformerEvaluationResult:
        predicted, actual = (
            self.predict_dataset(
                dataset
            )
        )

        errors = actual - predicted

        mae = float(
            np.mean(
                np.abs(errors)
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(errors ** 2)
            )
        )

        actual_flat = actual.reshape(-1)
        predicted_flat = predicted.reshape(-1)

        denominator = np.sum(
            (actual_flat - actual_flat.mean()) ** 2
        )

        r2 = (
            float(
                1 - (
                    np.sum(
                        (actual_flat - predicted_flat) ** 2
                    )
                    / denominator
                )
            )
            if denominator > 0
            else None
        )

        coverage = float(
            np.mean(
                np.abs(errors)
                <= interval_widths[None, :]
            )
        )

        per_horizon_mae = tuple(
            float(
                np.mean(
                    np.abs(errors[:, index])
                )
            )
            for index in range(
                self.config.forecast_horizon
            )
        )

        return TransformerEvaluationResult(
            sample_count=len(actual),
            mae=mae,
            rmse=rmse,
            r2=r2,
            interval_coverage=coverage,
            per_horizon_mae=per_horizon_mae,
        )


# ============================================================
# MAIN FORECASTING SERVICE
# ============================================================

class TransformerFloodForecastService:
    def __init__(
        self,
        config: TransformerForecastConfig | None = None,
    ) -> None:
        self.config = (
            config or TransformerForecastConfig()
        )

        set_seed(
            self.config.random_seed
        )

        self.device = resolve_device(
            self.config.device
        )

        self.validator = (
            TransformerTimeSeriesValidator(
                self.config
            )
        )

        self.scaler = TransformerFeatureScaler(
            self.config
        )

        self.model = FloodTransformerNetwork(
            self.config
        ).to(self.device)

        self.trainer = TransformerForecastTrainer(
            self.config,
            self.model,
            self.device,
        )

        self.evaluator = TransformerForecastEvaluator(
            self.config,
            self.model,
            self.scaler,
            self.device,
        )

        self.interval_widths: np.ndarray | None = None

        self.trained = False

    def train(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        TransformerTrainingResult,
        TransformerEvaluationResult,
    ]:
        frame = self.validator.validate(
            dataframe,
            minimum_rows=(
                self.config.sequence_length
                + self.config.forecast_horizon
                + 20
            ),
        )

        builder = TransformerDatasetBuilder(
            self.config,
            self.scaler,
        )

        splits = builder.build(
            frame
        )

        training = self.trainer.train(
            splits
        )

        validation_predictions, validation_actual = (
            self.evaluator.predict_dataset(
                splits.validation
            )
        )

        residuals = np.abs(
            validation_actual
            - validation_predictions
        )

        self.interval_widths = np.quantile(
            residuals,
            self.config.interval_coverage,
            axis=0,
        ).astype(np.float64)

        self.trained = True

        evaluation = self.evaluator.evaluate(
            splits.test,
            self.interval_widths,
        )

        return training, evaluation

    @torch.inference_mode()
    def forecast(
        self,
        recent_data: pd.DataFrame,
    ) -> TransformerForecast:
        if not self.trained:
            raise TransformerNotReadyError(
                "Train or load model first"
            )

        if self.interval_widths is None:
            raise TransformerNotReadyError(
                "Prediction intervals unavailable"
            )

        frame = self.validator.validate(
            recent_data,
            minimum_rows=(
                self.config.sequence_length
            ),
        )

        last_timestamp = (
            frame[
                self.config.timestamp_column
            ].iloc[-1].to_pydatetime()
        )

        age_minutes = (
            utc_now() - last_timestamp
        ).total_seconds() / 60

        if (
            age_minutes < -1
            or age_minutes
            > self.config.maximum_data_age_minutes
        ):
            raise TransformerValidationError(
                "Sensor data is stale or future-dated"
            )

        values = frame[
            list(self.config.feature_columns)
        ].to_numpy(dtype=np.float64)

        window = values[
            -self.config.sequence_length:
        ]

        scaled = self.scaler.transform_features(
            window
        )

        tensor = torch.from_numpy(
            scaled[None, :, :]
        ).to(self.device)

        self.model.eval()

        output = self.model(
            tensor
        )[0].cpu().numpy()

        predicted = (
            self.scaler.inverse_target(
                output
            )
        )

        points = []

        for index, value in enumerate(
            predicted
        ):
            step = index + 1

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
                self.interval_widths[index]
            )

            points.append(
                TransformerForecastPoint(
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

        return TransformerForecast(
            forecast_id=new_id(),
            target_column=(
                self.config.target_column
            ),
            observed_until=last_timestamp,
            points=tuple(points),
            requires_human_review=True,
            warnings=(
                "Advisory-only flood forecast",
                "Prediction intervals are empirical",
                "Verify with official hydrological data",
                "Human approval required for emergency actions",
            ),
            generated_at=utc_now(),
        )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> Path:
        if not self.trained:
            raise TransformerNotReadyError(
                "Cannot save untrained model"
            )

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise TransformerCheckpointError(
                "Checkpoint already exists"
            )

        torch.save(
            {
                "format_version": 1,
                "config": asdict(self.config),
                "model_state": self.model.state_dict(),
                "feature_mean": self.scaler.feature_mean,
                "feature_std": self.scaler.feature_std,
                "target_mean": self.scaler.target_mean,
                "target_std": self.scaler.target_std,
                "interval_widths": self.interval_widths,
            },
            destination,
        )

        return destination

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        checkpoint_path = Path(path)

        if not checkpoint_path.is_file():
            raise TransformerCheckpointError(
                "Checkpoint does not exist"
            )

        payload = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        if payload.get("format_version") != 1:
            raise TransformerCheckpointError(
                "Unsupported checkpoint version"
            )

        saved_config = payload.get(
            "config",
            {},
        )

        for name in (
            "feature_columns",
            "target_column",
            "sequence_length",
            "forecast_horizon",
            "model_dimension",
            "attention_heads",
            "encoder_layers",
            "feedforward_dimension",
        ):
            saved = saved_config.get(name)
            current = getattr(
                self.config,
                name,
            )

            if name == "feature_columns":
                saved = tuple(saved)

            if saved != current:
                raise TransformerCheckpointError(
                    f"Checkpoint config mismatch: {name}"
                )

        self.model.load_state_dict(
            payload["model_state"],
            strict=True,
        )

        self.scaler.feature_mean = np.asarray(
            payload["feature_mean"],
            dtype=np.float64,
        )

        self.scaler.feature_std = np.asarray(
            payload["feature_std"],
            dtype=np.float64,
        )

        self.scaler.target_mean = float(
            payload["target_mean"]
        )

        self.scaler.target_std = float(
            payload["target_std"]
        )

        self.interval_widths = np.asarray(
            payload["interval_widths"],
            dtype=np.float64,
        )

        if (
            self.interval_widths.shape
            != (self.config.forecast_horizon,)
        ):
            raise TransformerCheckpointError(
                "Invalid interval shape"
            )

        self.scaler.fitted = True

        self.model.to(self.device)
        self.model.eval()

        self.trained = True


# ============================================================
# FACTORY
# ============================================================

def create_transformer_forecast_service(
    config: TransformerForecastConfig | None = None,
) -> TransformerFloodForecastService:
    return TransformerFloodForecastService(
        config=config
    )


__all__ = [
    "TransformerForecastError",
    "TransformerValidationError",
    "TransformerNotReadyError",
    "TransformerCheckpointError",
    "TransformerForecastConfig",
    "TransformerEpochMetrics",
    "TransformerTrainingResult",
    "TransformerEvaluationResult",
    "TransformerForecastPoint",
    "TransformerForecast",
    "SinusoidalPositionalEncoding",
    "FloodTransformerNetwork",
    "TransformerTimeSeriesValidator",
    "TransformerFeatureScaler",
    "TransformerDatasetSplits",
    "TransformerDatasetBuilder",
    "TransformerForecastTrainer",
    "TransformerForecastEvaluator",
    "TransformerFloodForecastService",
    "create_transformer_forecast_service",
]
