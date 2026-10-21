"""Placeholder module: ai/deep_learning/temporal_fusion.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Temporal Fusion Forecasting Model

File:
    backend/ai/deep_learning/temporal_fusion.py

Python: 3.11+

Dependencies:
    torch
    numpy
    pandas

Architecture:
    TFT-inspired temporal fusion network

Components:
    - Feature projection
    - Gated variable selection
    - Historical LSTM encoder
    - Future-known feature encoder
    - Multi-head temporal attention
    - Gated residual networks
    - Quantile prediction heads

Targets:
    - River level
    - Rainfall
    - Shelter demand
    - Other continuous flood-related signals

Forecasts:
    - Multiple future time steps
    - P10 / P50 / P90 quantiles

Safety:
    - Advisory-only forecasts
    - No autonomous emergency actions
    - Missing/stale data rejected
    - Human review for uncertain predictions

This is a compact TFT-inspired implementation,
not an exact reproduction of the original TFT.
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

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# EXCEPTIONS
# ============================================================

class TemporalFusionError(Exception):
    pass


class TemporalFusionValidationError(TemporalFusionError):
    pass


class TemporalFusionNotReadyError(TemporalFusionError):
    pass


class TemporalFusionCheckpointError(TemporalFusionError):
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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if name not in {"cpu", "cuda"}:
        raise TemporalFusionValidationError(
            "device must be auto, cpu, or cuda"
        )

    if name == "cuda" and not torch.cuda.is_available():
        raise TemporalFusionValidationError(
            "CUDA is unavailable"
        )

    return torch.device(name)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TemporalFusionConfig:
    timestamp_column: str = "timestamp"

    historical_features: tuple[str, ...] = (
        "rainfall_mm",
        "river_level_m",
        "dam_level_percent",
        "temperature_c",
        "humidity_percent",
    )

    known_future_features: tuple[str, ...] = (
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    )

    target_column: str = "river_level_m"

    encoder_length: int = 48
    forecast_horizon: int = 12

    hidden_size: int = 64
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.10

    quantiles: tuple[float, ...] = (
        0.10,
        0.50,
        0.90,
    )

    batch_size: int = 32
    epochs: int = 40

    learning_rate: float = 0.0005
    weight_decay: float = 0.0001

    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 7

    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    expected_frequency_minutes: int = 60
    maximum_data_age_minutes: int = 120

    maximum_interval_width: float = 3.0

    device: str = "auto"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.historical_features:
            raise TemporalFusionValidationError(
                "Historical features are required"
            )

        if not self.known_future_features:
            raise TemporalFusionValidationError(
                "Future-known features are required"
            )

        if len(set(self.historical_features)) != len(
            self.historical_features
        ):
            raise TemporalFusionValidationError(
                "Duplicate historical features"
            )

        if len(set(self.known_future_features)) != len(
            self.known_future_features
        ):
            raise TemporalFusionValidationError(
                "Duplicate future features"
            )

        if self.target_column not in self.historical_features:
            raise TemporalFusionValidationError(
                "Target must be in historical features"
            )

        for name in (
            "encoder_length",
            "forecast_horizon",
            "hidden_size",
            "lstm_layers",
            "attention_heads",
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
                raise TemporalFusionValidationError(
                    f"{name} must be a positive integer"
                )

        if self.hidden_size % self.attention_heads:
            raise TemporalFusionValidationError(
                "hidden_size must be divisible "
                "by attention_heads"
            )

        if not 0 <= self.dropout < 1:
            raise TemporalFusionValidationError(
                "dropout must be in [0,1)"
            )

        if (
            not 0 < self.train_fraction < 1
            or not 0 < self.validation_fraction < 1
            or self.train_fraction
            + self.validation_fraction >= 1
        ):
            raise TemporalFusionValidationError(
                "Invalid chronological split fractions"
            )

        if (
            len(self.quantiles) < 3
            or tuple(sorted(self.quantiles)) != self.quantiles
            or len(set(self.quantiles)) != len(self.quantiles)
            or not all(0 < q < 1 for q in self.quantiles)
            or 0.5 not in self.quantiles
        ):
            raise TemporalFusionValidationError(
                "Quantiles must be sorted, unique, "
                "contain 0.5, and lie within (0,1)"
            )

        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "maximum_interval_width",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise TemporalFusionValidationError(
                    f"{name} must be positive and finite"
                )

        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise TemporalFusionValidationError(
                "weight_decay must be nonnegative"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class TemporalFusionEpoch:
    epoch: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class TemporalFusionTrainingResult:
    training_id: str
    epochs_completed: int
    best_validation_loss: float
    history: tuple[TemporalFusionEpoch, ...]
    trained_at: datetime


@dataclass(frozen=True)
class TemporalFusionEvaluation:
    sample_count: int
    mae: float
    rmse: float
    interval_coverage: float
    average_interval_width: float


@dataclass(frozen=True)
class TemporalFusionForecastPoint:
    timestamp: datetime
    horizon_step: int
    lower: float
    median: float
    upper: float


@dataclass(frozen=True)
class TemporalFusionForecast:
    forecast_id: str
    target_column: str
    observed_until: datetime
    points: tuple[TemporalFusionForecastPoint, ...]
    requires_human_review: bool
    warnings: tuple[str, ...]
    generated_at: datetime


# ============================================================
# GATED RESIDUAL NETWORK
# ============================================================

class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size,
        )

        self.fc2 = nn.Linear(
            hidden_size,
            output_size,
        )

        self.gate = nn.Linear(
            output_size,
            output_size,
        )

        self.skip = (
            nn.Linear(input_size, output_size)
            if input_size != output_size
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.nn.functional.elu(
            self.fc1(x)
        )

        hidden = self.dropout(
            self.fc2(hidden)
        )

        gated = (
            torch.sigmoid(
                self.gate(hidden)
            )
            * hidden
        )

        return self.norm(
            gated + self.skip(x)
        )


# ============================================================
# VARIABLE SELECTION NETWORK
# ============================================================

class VariableSelectionNetwork(nn.Module):
    """
    Learns feature weights at each time step.

    Input:
        B x T x F

    Output:
        B x T x H
    """

    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.feature_count = feature_count

        self.feature_projections = nn.ModuleList([
            GatedResidualNetwork(
                input_size=1,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for _ in range(feature_count)
        ])

        self.selection_network = nn.Sequential(
            nn.Linear(
                feature_count,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                feature_count,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if x.shape[-1] != self.feature_count:
            raise TemporalFusionValidationError(
                "Variable count mismatch"
            )

        weights = torch.softmax(
            self.selection_network(x),
            dim=-1,
        )

        projected = torch.stack(
            [
                layer(x[..., index:index + 1])
                for index, layer
                in enumerate(self.feature_projections)
            ],
            dim=-2,
        )

        combined = (
            projected
            * weights.unsqueeze(-1)
        ).sum(dim=-2)

        return combined, weights


# ============================================================
# TEMPORAL FUSION NETWORK
# ============================================================

class TemporalFusionNetwork(nn.Module):
    """
    Compact TFT-inspired network.

    Historical encoder:
        LSTM over observed measurements.

    Future decoder:
        Known future features + repeated
        historical context.

    Temporal attention:
        Decoder queries attend to historical
        states and permitted future-known
        representations.

    No future target observations are used.
    """

    def __init__(
        self,
        config: TemporalFusionConfig,
    ) -> None:
        super().__init__()

        self.config = config

        hidden = config.hidden_size

        self.historical_selection = (
            VariableSelectionNetwork(
                len(config.historical_features),
                hidden,
                config.dropout,
            )
        )

        self.future_selection = (
            VariableSelectionNetwork(
                len(config.known_future_features),
                hidden,
                config.dropout,
            )
        )

        self.encoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.lstm_layers > 1
                else 0
            ),
        )

        self.decoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.lstm_layers > 1
                else 0
            ),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.fusion = GatedResidualNetwork(
            input_size=hidden * 2,
            hidden_size=hidden,
            output_size=hidden,
            dropout=config.dropout,
        )

        self.output_head = nn.Linear(
            hidden,
            len(config.quantiles),
        )

    def forward(
        self,
        historical: torch.Tensor,
        future_known: torch.Tensor,
    ) -> torch.Tensor:
        history, _ = (
            self.historical_selection(
                historical
            )
        )

        future, _ = (
            self.future_selection(
                future_known
            )
        )

        encoded, state = self.encoder(
            history
        )

        decoded, _ = self.decoder(
            future,
            state,
        )

        combined = torch.cat(
            [encoded, decoded],
            dim=1,
        )

        # Decoder queries can attend to
        # historical states and earlier/equal
        # future-known positions.
        horizon = decoded.shape[1]
        history_length = encoded.shape[1]

        future_positions = torch.arange(
            horizon,
            device=decoded.device,
        )

        key_positions = torch.arange(
            history_length + horizon,
            device=decoded.device,
        )

        # True means disallowed.
        attention_mask = (
            key_positions.unsqueeze(0)
            >
            (
                history_length
                + future_positions.unsqueeze(1)
            )
        )

        attended, _ = self.attention(
            query=decoded,
            key=combined,
            value=combined,
            attn_mask=attention_mask,
            need_weights=False,
        )

        fused = self.fusion(
            torch.cat(
                [decoded, attended],
                dim=-1,
            )
        )

        raw_quantiles = self.output_head(
            fused
        )

        # Enforce monotonic quantiles:
        # first output is the lower quantile;
        # successive quantiles are increments.
        first = raw_quantiles[..., :1]

        increments = (
            torch.nn.functional.softplus(
                raw_quantiles[..., 1:]
            )
        )

        ordered = torch.cat(
            [
                first,
                first + torch.cumsum(
                    increments,
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        return ordered


# ============================================================
# QUANTILE LOSS
# ============================================================

class QuantileLoss(nn.Module):
    def __init__(
        self,
        quantiles: Sequence[float],
    ) -> None:
        super().__init__()

        self.register_buffer(
            "quantiles",
            torch.tensor(
                quantiles,
                dtype=torch.float32,
            ).view(1, 1, -1),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        errors = (
            targets.unsqueeze(-1)
            - predictions
        )

        losses = torch.maximum(
            self.quantiles * errors,
            (self.quantiles - 1) * errors,
        )

        return losses.mean()


# ============================================================
# DATA PREPROCESSING
# ============================================================

class TemporalFusionPreprocessor:
    """
    Uses training-only means and standard
    deviations for historical features,
    future-known features, and targets.
    """

    def __init__(
        self,
        config: TemporalFusionConfig,
    ) -> None:
        self.config = config

        self.history_mean = None
        self.history_std = None

        self.future_mean = None
        self.future_std = None

        self.target_mean = None
        self.target_std = None

        self.fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
    ) -> None:
        historical = frame[
            list(self.config.historical_features)
        ].to_numpy(dtype=np.float64)

        future = frame[
            list(self.config.known_future_features)
        ].to_numpy(dtype=np.float64)

        target = frame[
            self.config.target_column
        ].to_numpy(dtype=np.float64)

        self.history_mean = historical.mean(axis=0)
        self.history_std = np.maximum(
            historical.std(axis=0),
            1e-6,
        )

        self.future_mean = future.mean(axis=0)
        self.future_std = np.maximum(
            future.std(axis=0),
            1e-6,
        )

        self.target_mean = float(
            target.mean()
        )

        self.target_std = float(
            max(target.std(), 1e-6)
        )

        self.fitted = True

    def transform_history(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        self._require_fitted()

        return (
            (values - self.history_mean)
            / self.history_std
        ).astype(np.float32)

    def transform_future(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        self._require_fitted()

        return (
            (values - self.future_mean)
            / self.future_std
        ).astype(np.float32)

    def transform_target(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        self._require_fitted()

        return (
            (values - self.target_mean)
            / self.target_std
        ).astype(np.float32)

    def inverse_target(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        self._require_fitted()

        return (
            values * self.target_std
            + self.target_mean
        )

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise TemporalFusionNotReadyError(
                "Preprocessor is not fitted"
            )


# ============================================================
# TIME SERIES VALIDATION
# ============================================================

class TemporalFusionDataValidator:
    def __init__(
        self,
        config: TemporalFusionConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        required = {
            self.config.timestamp_column,
            *self.config.historical_features,
            *self.config.known_future_features,
        }

        missing = required - set(dataframe.columns)

        if missing:
            raise TemporalFusionValidationError(
                f"Missing columns: {sorted(missing)}"
            )

        frame = dataframe.copy()

        column = self.config.timestamp_column

        frame[column] = pd.to_datetime(
            frame[column],
            utc=True,
            errors="coerce",
        )

        if frame[column].isna().any():
            raise TemporalFusionValidationError(
                "Invalid timestamps"
            )

        frame = frame.sort_values(
            column
        ).reset_index(drop=True)

        if frame[column].duplicated().any():
            raise TemporalFusionValidationError(
                "Duplicate timestamps"
            )

        numeric_columns = set(
            self.config.historical_features
        ) | set(
            self.config.known_future_features
        )

        for name in numeric_columns:
            frame[name] = pd.to_numeric(
                frame[name],
                errors="coerce",
            )

        if not np.isfinite(
            frame[list(numeric_columns)]
            .to_numpy(dtype=np.float64)
        ).all():
            raise TemporalFusionValidationError(
                "Missing or non-finite values"
            )

        expected = pd.Timedelta(
            minutes=self.config.expected_frequency_minutes
        )

        differences = frame[column].diff().dropna()

        if not differences.eq(expected).all():
            raise TemporalFusionValidationError(
                "Irregular time series"
            )

        minimum = (
            self.config.encoder_length
            + self.config.forecast_horizon
            + 20
        )

        if len(frame) < minimum:
            raise TemporalFusionValidationError(
                f"At least {minimum} observations required"
            )

        return frame


# ============================================================
# CHRONOLOGICAL WINDOW BUILDER
# ============================================================

@dataclass
class TemporalFusionSplits:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


class TemporalFusionDatasetBuilder:
    def __init__(
        self,
        config: TemporalFusionConfig,
        preprocessor: TemporalFusionPreprocessor,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor

    def build(
        self,
        frame: pd.DataFrame,
    ) -> TemporalFusionSplits:
        n = len(frame)

        train_end = int(
            n * self.config.train_fraction
        )

        val_end = int(
            n * (
                self.config.train_fraction
                + self.config.validation_fraction
            )
        )

        self.preprocessor.fit(
            frame.iloc[:train_end]
        )

        historical = (
            self.preprocessor.transform_history(
                frame[
                    list(self.config.historical_features)
                ].to_numpy(dtype=np.float64)
            )
        )

        future = (
            self.preprocessor.transform_future(
                frame[
                    list(self.config.known_future_features)
                ].to_numpy(dtype=np.float64)
            )
        )

        target = (
            self.preprocessor.transform_target(
                frame[self.config.target_column]
                .to_numpy(dtype=np.float64)
            )
        )

        buckets: dict[str, dict[str, list]] = {
            name: {
                "history": [],
                "future": [],
                "target": [],
            }
            for name in (
                "train",
                "validation",
                "test",
            )
        }

        encoder_length = (
            self.config.encoder_length
        )

        horizon = (
            self.config.forecast_horizon
        )

        for start in range(
            n - encoder_length - horizon + 1
        ):
            input_end = (
                start + encoder_length
            )

            target_end = (
                input_end + horizon
            )

            if target_end <= train_end:
                split = "train"

            elif (
                input_end >= train_end
                and target_end <= val_end
            ):
                split = "validation"

            elif input_end >= val_end:
                split = "test"

            else:
                continue

            buckets[split]["history"].append(
                historical[start:input_end]
            )

            buckets[split]["future"].append(
                future[input_end:target_end]
            )

            buckets[split]["target"].append(
                target[input_end:target_end]
            )

        def make_dataset(name: str) -> TensorDataset:
            bucket = buckets[name]

            if not bucket["history"]:
                raise TemporalFusionValidationError(
                    f"No samples in {name} split"
                )

            return TensorDataset(
                torch.from_numpy(
                    np.stack(
                        bucket["history"]
                    )
                ),
                torch.from_numpy(
                    np.stack(
                        bucket["future"]
                    )
                ),
                torch.from_numpy(
                    np.stack(
                        bucket["target"]
                    )
                ),
            )

        return TemporalFusionSplits(
            train=make_dataset("train"),
            validation=make_dataset("validation"),
            test=make_dataset("test"),
        )


# ============================================================
# TRAINING ENGINE
# ============================================================

class TemporalFusionTrainer:
    def __init__(
        self,
        config: TemporalFusionConfig,
        model: TemporalFusionNetwork,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.criterion = QuantileLoss(
            config.quantiles
        ).to(device)

    def _epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        training = optimizer is not None

        self.model.train(training)

        total_loss = 0.0
        total_samples = 0

        for historical, future, target in loader:
            historical = historical.to(
                self.device
            )

            future = future.to(
                self.device
            )

            target = target.to(
                self.device
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(training):
                predictions = self.model(
                    historical,
                    future,
                )

                loss = self.criterion(
                    predictions,
                    target,
                )

                if training:
                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                    optimizer.step()

            batch_size = historical.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        if total_samples == 0:
            raise TemporalFusionValidationError(
                "Empty training dataset"
            )

        return total_loss / total_samples

    def train(
        self,
        splits: TemporalFusionSplits,
    ) -> TemporalFusionTrainingResult:
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
            train_loss = self._epoch(
                training_loader,
                optimizer,
            )

            with torch.no_grad():
                val_loss = self._epoch(
                    validation_loader,
                    None,
                )

            history.append(
                TemporalFusionEpoch(
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
            raise TemporalFusionError(
                "Training produced no valid checkpoint"
            )

        self.model.load_state_dict(
            best_state
        )

        self.model.eval()

        return TemporalFusionTrainingResult(
            training_id=new_id(),
            epochs_completed=len(history),
            best_validation_loss=best_loss,
            history=tuple(history),
            trained_at=utc_now(),
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

class TemporalFusionEvaluator:
    def __init__(
        self,
        config: TemporalFusionConfig,
        model: TemporalFusionNetwork,
        preprocessor: TemporalFusionPreprocessor,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.preprocessor = preprocessor
        self.device = device

    @torch.inference_mode()
    def evaluate(
        self,
        dataset: TensorDataset,
    ) -> TemporalFusionEvaluation:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        predicted = []
        actual = []

        self.model.eval()

        for history, future, target in loader:
            output = self.model(
                history.to(self.device),
                future.to(self.device),
            )

            predicted.append(
                output.cpu().numpy()
            )

            actual.append(
                target.numpy()
            )

        if not predicted:
            raise TemporalFusionValidationError(
                "Evaluation dataset is empty"
            )

        predicted = np.concatenate(
            predicted
        )

        actual = np.concatenate(
            actual
        )

        predicted = (
            self.preprocessor.inverse_target(
                predicted
            )
        )

        actual = (
            self.preprocessor.inverse_target(
                actual
            )
        )

        median_index = (
            self.config.quantiles.index(0.5)
        )

        median = predicted[
            ...,
            median_index
        ]

        lower = predicted[..., 0]
        upper = predicted[..., -1]

        errors = actual - median

        coverage = np.mean(
            (actual >= lower)
            & (actual <= upper)
        )

        return TemporalFusionEvaluation(
            sample_count=len(actual),
            mae=float(
                np.mean(
                    np.abs(errors)
                )
            ),
            rmse=float(
                np.sqrt(
                    np.mean(errors ** 2)
                )
            ),
            interval_coverage=float(coverage),
            average_interval_width=float(
                np.mean(upper - lower)
            ),
        )


# ============================================================
# MAIN SERVICE
# ============================================================

class TemporalFusionService:
    def __init__(
        self,
        config: TemporalFusionConfig | None = None,
    ) -> None:
        self.config = (
            config or TemporalFusionConfig()
        )

        set_seed(
            self.config.random_seed
        )

        self.device = resolve_device(
            self.config.device
        )

        self.validator = (
            TemporalFusionDataValidator(
                self.config
            )
        )

        self.preprocessor = (
            TemporalFusionPreprocessor(
                self.config
            )
        )

        self.model = TemporalFusionNetwork(
            self.config
        ).to(self.device)

        self.trainer = TemporalFusionTrainer(
            self.config,
            self.model,
            self.device,
        )

        self.evaluator = TemporalFusionEvaluator(
            self.config,
            self.model,
            self.preprocessor,
            self.device,
        )

        self.trained = False

    def train(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        TemporalFusionTrainingResult,
        TemporalFusionEvaluation,
    ]:
        frame = self.validator.validate(
            dataframe
        )

        builder = TemporalFusionDatasetBuilder(
            self.config,
            self.preprocessor,
        )

        splits = builder.build(
            frame
        )

        training = self.trainer.train(
            splits
        )

        self.trained = True

        evaluation = self.evaluator.evaluate(
            splits.test
        )

        return training, evaluation

    @torch.inference_mode()
    def forecast(
        self,
        historical_frame: pd.DataFrame,
        future_known_frame: pd.DataFrame,
    ) -> TemporalFusionForecast:
        if not self.trained:
            raise TemporalFusionNotReadyError(
                "Train or load model before forecasting"
            )

        historical = self.validator.validate(
            historical_frame
        )

        last_timestamp = (
            historical[
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
            raise TemporalFusionValidationError(
                "Historical sensor observations "
                "are stale or future-dated"
            )

        if len(future_known_frame) != (
            self.config.forecast_horizon
        ):
            raise TemporalFusionValidationError(
                "Future-known rows must equal horizon"
            )

        future = future_known_frame.copy()

        timestamp_column = (
            self.config.timestamp_column
        )

        required = {
            timestamp_column,
            *self.config.known_future_features,
        }

        missing = required - set(future.columns)

        if missing:
            raise TemporalFusionValidationError(
                f"Missing future columns: {sorted(missing)}"
            )

        future[timestamp_column] = pd.to_datetime(
            future[timestamp_column],
            utc=True,
            errors="coerce",
        )

        expected_times = pd.date_range(
            start=(
                last_timestamp
                + pd.Timedelta(
                    minutes=(
                        self.config.expected_frequency_minutes
                    )
                )
            ),
            periods=self.config.forecast_horizon,
            freq=pd.Timedelta(
                minutes=self.config.expected_frequency_minutes
            ),
        )

        if not future[timestamp_column].reset_index(
            drop=True
        ).equals(
            pd.Series(expected_times)
        ):
            raise TemporalFusionValidationError(
                "Future timestamps must match "
                "the forecast horizon exactly"
            )

        future_values = future[
            list(self.config.known_future_features)
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(future_values).all():
            raise TemporalFusionValidationError(
                "Future features contain invalid values"
            )

        historical_values = historical[
            list(self.config.historical_features)
        ].to_numpy(dtype=np.float64)

        historical_values = historical_values[
            -self.config.encoder_length:
        ]

        history_scaled = (
            self.preprocessor.transform_history(
                historical_values
            )
        )

        future_scaled = (
            self.preprocessor.transform_future(
                future_values
            )
        )

        history_tensor = torch.from_numpy(
            history_scaled[None, ...]
        ).to(self.device)

        future_tensor = torch.from_numpy(
            future_scaled[None, ...]
        ).to(self.device)

        self.model.eval()

        output = self.model(
            history_tensor,
            future_tensor,
        )[0].cpu().numpy()

        output = (
            self.preprocessor.inverse_target(
                output
            )
        )

        median_index = (
            self.config.quantiles.index(0.5)
        )

        points = []

        review_required = False

        for index, row in enumerate(
            output
        ):
            lower = float(row[0])
            median = float(
                row[median_index]
            )
            upper = float(row[-1])

            if (
                upper - lower
                > self.config.maximum_interval_width
            ):
                review_required = True

            points.append(
                TemporalFusionForecastPoint(
                    timestamp=(
                        expected_times[index]
                        .to_pydatetime()
                    ),
                    horizon_step=index + 1,
                    lower=lower,
                    median=median,
                    upper=upper,
                )
            )

        return TemporalFusionForecast(
            forecast_id=new_id(),
            target_column=self.config.target_column,
            observed_until=last_timestamp,
            points=tuple(points),
            requires_human_review=True,
            warnings=(
                "Advisory-only forecast",
                "Quantile coverage is not guaranteed",
                "Verify with official sensor observations",
                "Human oversight required for emergency decisions",
            ),
            generated_at=utc_now(),
        )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> Path:
        if not self.trained:
            raise TemporalFusionNotReadyError(
                "Cannot save an untrained model"
            )

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise TemporalFusionCheckpointError(
                "Checkpoint already exists"
            )

        torch.save(
            {
                "format_version": 1,
                "config": asdict(self.config),
                "model_state": self.model.state_dict(),
                "history_mean": (
                    self.preprocessor.history_mean
                ),
                "history_std": (
                    self.preprocessor.history_std
                ),
                "future_mean": (
                    self.preprocessor.future_mean
                ),
                "future_std": (
                    self.preprocessor.future_std
                ),
                "target_mean": (
                    self.preprocessor.target_mean
                ),
                "target_std": (
                    self.preprocessor.target_std
                ),
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
            raise TemporalFusionCheckpointError(
                "Checkpoint not found"
            )

        payload = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        if payload.get("format_version") != 1:
            raise TemporalFusionCheckpointError(
                "Unsupported checkpoint version"
            )

        saved_config = payload.get(
            "config",
            {},
        )

        for name in (
            "historical_features",
            "known_future_features",
            "target_column",
            "encoder_length",
            "forecast_horizon",
            "hidden_size",
            "lstm_layers",
            "attention_heads",
            "quantiles",
        ):
            saved = saved_config.get(name)
            current = getattr(
                self.config,
                name,
            )

            if isinstance(current, tuple):
                saved = tuple(saved)

            if saved != current:
                raise TemporalFusionCheckpointError(
                    f"Checkpoint mismatch: {name}"
                )

        self.model.load_state_dict(
            payload["model_state"],
            strict=True,
        )

        for name in (
            "history_mean",
            "history_std",
            "future_mean",
            "future_std",
            "target_mean",
            "target_std",
        ):
            value = payload[name]

            if name in (
                "target_mean",
                "target_std",
            ):
                value = float(value)
            else:
                value = np.asarray(
                    value,
                    dtype=np.float64,
                )

            setattr(
                self.preprocessor,
                name,
                value,
            )

        self.preprocessor.fitted = True

        self.model.to(self.device)
        self.model.eval()

        self.trained = True


# ============================================================
# FACTORY
# ============================================================

def create_temporal_fusion_service(
    config: TemporalFusionConfig | None = None,
) -> TemporalFusionService:
    return TemporalFusionService(
        config=config
    )


__all__ = [
    "TemporalFusionError",
    "TemporalFusionValidationError",
    "TemporalFusionNotReadyError",
    "TemporalFusionCheckpointError",
    "TemporalFusionConfig",
    "TemporalFusionEpoch",
    "TemporalFusionTrainingResult",
    "TemporalFusionEvaluation",
    "TemporalFusionForecastPoint",
    "TemporalFusionForecast",
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "TemporalFusionNetwork",
    "QuantileLoss",
    "TemporalFusionPreprocessor",
    "TemporalFusionDataValidator",
    "TemporalFusionSplits",
    "TemporalFusionDatasetBuilder",
    "TemporalFusionTrainer",
    "TemporalFusionEvaluator",
    "TemporalFusionService",
    "create_temporal_fusion_service",
]
