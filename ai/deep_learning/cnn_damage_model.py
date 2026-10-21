"""Placeholder module: ai/deep_learning/cnn_damage_model.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
CNN Flood Damage Classification Model

File:
    backend/ai/computer_vision/cnn_damage_model.py

Python:
    3.11+

Dependencies:
    torch>=2.2
    torchvision>=0.17
    Pillow>=10.0

Features:
    - EfficientNet-B0 transfer learning
    - ResNet18 alternative
    - Custom lightweight CNN
    - Flood damage image classification
    - Training and validation
    - Class-weighted loss
    - Early stopping
    - Evaluation and confusion matrix
    - Single and batch image inference
    - Confidence and uncertainty screening
    - Model checkpoint save/load
    - CPU and CUDA support
    - Human-review recommendations

SAFETY:
    Model predictions are advisory only.

    This module does not determine:
        - Structural stability
        - Building occupancy safety
        - Electrical safety
        - Water contamination
        - Emergency rescue priority
        - Insurance liability
        - Verified financial losses

    Critical assessments require trained
    human inspection and independent evidence.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import math
import random
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image, ImageOps, UnidentifiedImageError


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class DamageModelError(Exception):
    """Base CNN damage model exception."""


class DamageValidationError(DamageModelError):
    """Invalid model input or configuration."""


class DamageTrainingError(DamageModelError):
    """Model training failure."""


class DamageInferenceError(DamageModelError):
    """Damage inference failure."""


class DamageCheckpointError(DamageModelError):
    """Checkpoint loading or saving failure."""


# ============================================================
# ENUMERATIONS
# ============================================================

class DamageSeverity(str, Enum):
    NO_VISIBLE_DAMAGE = "no_visible_damage"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class CNNArchitecture(str, Enum):
    EFFICIENTNET_B0 = "efficientnet_b0"
    RESNET18 = "resnet18"
    CUSTOM_CNN = "custom_cnn"


class DamagePredictionStatus(str, Enum):
    PREDICTED = "predicted"
    REVIEW_REQUIRED = "review_required"


class DamageReviewReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    LOW_MARGIN = "low_margin"
    HIGH_ENTROPY = "high_entropy"
    SEVERE_DAMAGE = "severe_damage"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_probability(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise DamageValidationError(
            f"{name} must be between 0 and 1"
        )


def get_device(
    requested: str = "auto",
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested not in ("cpu", "cuda"):
        raise DamageValidationError(
            "device must be auto, cpu or cuda"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        raise DamageValidationError(
            "CUDA is not available"
        )

    return torch.device(requested)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CNNDamageConfig:
    architecture: CNNArchitecture = (
        CNNArchitecture.EFFICIENTNET_B0
    )

    class_names: tuple[str, ...] = (
        DamageSeverity.NO_VISIBLE_DAMAGE.value,
        DamageSeverity.MINOR.value,
        DamageSeverity.MODERATE.value,
        DamageSeverity.SEVERE.value,
    )

    image_size: int = 224

    batch_size: int = 16

    epochs: int = 20

    learning_rate: float = 0.0001

    weight_decay: float = 0.0001

    early_stopping_patience: int = 5

    num_workers: int = 0

    pretrained: bool = True

    freeze_backbone: bool = False

    minimum_confidence: float = 0.70

    minimum_probability_margin: float = 0.15

    maximum_normalized_entropy: float = 0.80

    require_review_for_severe: bool = True

    maximum_image_bytes: int = 15 * 1024 * 1024

    maximum_image_pixels: int = 25_000_000

    device: str = "auto"

    random_seed: int = 42

    def __post_init__(self) -> None:
        if len(self.class_names) < 2:
            raise DamageValidationError(
                "At least two classes are required"
            )

        if len(set(self.class_names)) != len(self.class_names):
            raise DamageValidationError(
                "Class names must be unique"
            )

        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.class_names
        ):
            raise DamageValidationError(
                "Class names must be nonempty strings"
            )

        for name in (
            "image_size",
            "batch_size",
            "epochs",
            "early_stopping_patience",
            "maximum_image_bytes",
            "maximum_image_pixels",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise DamageValidationError(
                    f"{name} must be a positive integer"
                )

        if self.num_workers < 0:
            raise DamageValidationError(
                "num_workers cannot be negative"
            )

        for name in (
            "learning_rate",
            "weight_decay",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value < 0:
                raise DamageValidationError(
                    f"{name} must be finite and nonnegative"
                )

        if self.learning_rate == 0:
            raise DamageValidationError(
                "learning_rate must be positive"
            )

        for name in (
            "minimum_confidence",
            "minimum_probability_margin",
            "maximum_normalized_entropy",
        ):
            validate_probability(
                getattr(self, name),
                name,
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class DamageClassProbability:
    class_name: str
    probability: float


@dataclass(frozen=True)
class DamagePrediction:
    prediction_id: str

    predicted_class: str

    confidence: float

    class_probabilities: tuple[
        DamageClassProbability,
        ...
    ]

    probability_margin: float

    normalized_entropy: float

    status: DamagePredictionStatus

    requires_human_review: bool

    review_reasons: tuple[
        DamageReviewReason,
        ...
    ]

    model_architecture: CNNArchitecture

    model_trained: bool

    inference_time_ms: float

    predicted_at: datetime

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DamageEpochMetrics:
    epoch: int

    train_loss: float
    train_accuracy: float

    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True)
class DamageTrainingResult:
    training_id: str

    architecture: CNNArchitecture

    epochs_completed: int

    best_validation_loss: float

    best_validation_accuracy: float

    history: tuple[
        DamageEpochMetrics,
        ...
    ]

    trained_at: datetime


@dataclass(frozen=True)
class DamageEvaluationResult:
    sample_count: int

    accuracy: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    confusion_matrix: tuple[
        tuple[int, ...],
        ...
    ]

    class_names: tuple[str, ...]


# ============================================================
# IMAGE VALIDATION
# ============================================================

class DamageImageValidator:
    SUPPORTED_FORMATS = frozenset({
        "JPEG",
        "PNG",
        "WEBP",
    })

    def __init__(
        self,
        config: CNNDamageConfig,
    ) -> None:
        self.config = config

    def load_bytes(
        self,
        image_bytes: bytes,
    ) -> Image.Image:
        if not isinstance(image_bytes, bytes):
            raise DamageValidationError(
                "Image input must be bytes"
            )

        if not image_bytes:
            raise DamageValidationError(
                "Image is empty"
            )

        if len(image_bytes) > self.config.maximum_image_bytes:
            raise DamageValidationError(
                "Image exceeds upload size limit"
            )

        try:
            with Image.open(
                io.BytesIO(image_bytes)
            ) as source:
                if source.format not in self.SUPPORTED_FORMATS:
                    raise DamageValidationError(
                        "Unsupported image format"
                    )

                width, height = source.size

                if width * height > self.config.maximum_image_pixels:
                    raise DamageValidationError(
                        "Image pixel limit exceeded"
                    )

                source.load()

                return ImageOps.exif_transpose(
                    source
                ).convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise DamageValidationError(
                "Unable to decode image"
            ) from exc

    def load_path(
        self,
        path: str | Path,
    ) -> Image.Image:
        image_path = Path(path)

        if not image_path.is_file():
            raise DamageValidationError(
                "Image file does not exist"
            )

        if (
            image_path.stat().st_size
            > self.config.maximum_image_bytes
        ):
            raise DamageValidationError(
                "Image exceeds size limit"
            )

        return self.load_bytes(
            image_path.read_bytes()
        )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

class DamageImageTransforms:
    def __init__(
        self,
        config: CNNDamageConfig,
    ) -> None:
        self.config = config

    def training(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(
                (
                    self.config.image_size,
                    self.config.image_size,
                )
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                degrees=10
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def evaluation(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(
                (
                    self.config.image_size,
                    self.config.image_size,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])


# ============================================================
# CUSTOM CNN
# ============================================================

class LightweightDamageCNN(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(
                256,
                number_of_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(
            self.features(x)
        )


# ============================================================
# MODEL FACTORY
# ============================================================

class DamageCNNModelFactory:
    @staticmethod
    def create(
        config: CNNDamageConfig,
    ) -> nn.Module:
        number_of_classes = len(
            config.class_names
        )

        if config.architecture == CNNArchitecture.CUSTOM_CNN:
            return LightweightDamageCNN(
                number_of_classes
            )

        if config.architecture == CNNArchitecture.RESNET18:
            weights = (
                models.ResNet18_Weights.DEFAULT
                if config.pretrained
                else None
            )

            model = models.resnet18(
                weights=weights
            )

            if config.freeze_backbone:
                for parameter in model.parameters():
                    parameter.requires_grad = False

            model.fc = nn.Linear(
                model.fc.in_features,
                number_of_classes,
            )

            return model

        if config.architecture == CNNArchitecture.EFFICIENTNET_B0:
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT
                if config.pretrained
                else None
            )

            model = models.efficientnet_b0(
                weights=weights
            )

            if config.freeze_backbone:
                for parameter in model.parameters():
                    parameter.requires_grad = False

            last_layer = model.classifier[-1]

            model.classifier[-1] = nn.Linear(
                last_layer.in_features,
                number_of_classes,
            )

            return model

        raise DamageValidationError(
            "Unsupported CNN architecture"
        )


# ============================================================
# DATASET LOADER
# ============================================================

class DamageDatasetLoader:
    """
    Expected directory structure:

    dataset/
        train/
            no_visible_damage/
            minor/
            moderate/
            severe/
        val/
            no_visible_damage/
            minor/
            moderate/
            severe/
        test/
            no_visible_damage/
            minor/
            moderate/
            severe/

    Ensure that photographs from the same
    property, incident, or near-duplicate
    group do not cross data splits.
    """

    def __init__(
        self,
        config: CNNDamageConfig,
    ) -> None:
        self.config = config

        self.image_transforms = (
            DamageImageTransforms(config)
        )

    def create_loader(
        self,
        directory: str | Path,
        *,
        training: bool,
    ) -> DataLoader:
        path = Path(directory)

        if not path.is_dir():
            raise DamageValidationError(
                f"Dataset directory not found: {path}"
            )

        transform = (
            self.image_transforms.training()
            if training
            else self.image_transforms.evaluation()
        )

        dataset = datasets.ImageFolder(
            root=str(path),
            transform=transform,
        )

        if tuple(dataset.classes) != tuple(
            sorted(self.config.class_names)
        ):
            raise DamageValidationError(
                "Dataset classes do not match "
                "configured class names"
            )

        # ImageFolder uses alphabetical class order.
        # The configured order must match its indices.
        if tuple(dataset.classes) != self.config.class_names:
            raise DamageValidationError(
                "Configure class_names in alphabetical "
                "ImageFolder order"
            )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=training,
            num_workers=self.config.num_workers,
            pin_memory=(
                torch.cuda.is_available()
            ),
        )


# ============================================================
# TRAINING ENGINE
# ============================================================

class DamageCNNTrainer:
    def __init__(
        self,
        config: CNNDamageConfig,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> tuple[float, float]:
        training = optimizer is not None

        self.model.train(training)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(training):
                logits = self.model(images)

                loss = criterion(
                    logits,
                    labels,
                )

                if training:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_correct += (
                logits.argmax(dim=1)
                == labels
            ).sum().item()

            total_samples += batch_size

        if total_samples == 0:
            raise DamageTrainingError(
                "Dataset loader is empty"
            )

        return (
            total_loss / total_samples,
            total_correct / total_samples,
        )

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        class_weights: Sequence[float] | None = None,
    ) -> DamageTrainingResult:
        if class_weights is not None:
            if len(class_weights) != len(
                self.config.class_names
            ):
                raise DamageValidationError(
                    "Incorrect class weight count"
                )

            if any(
                not math.isfinite(weight) or weight <= 0
                for weight in class_weights
            ):
                raise DamageValidationError(
                    "Class weights must be positive"
                )

            weight_tensor = torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=self.device,
            )

        else:
            weight_tensor = None

        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor
        )

        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_loss = math.inf
        best_accuracy = 0.0

        best_state = None

        patience_counter = 0

        history: list[
            DamageEpochMetrics
        ] = []

        for epoch in range(
            1,
            self.config.epochs + 1,
        ):
            train_loss, train_accuracy = (
                self._run_epoch(
                    train_loader,
                    criterion,
                    optimizer,
                )
            )

            with torch.no_grad():
                val_loss, val_accuracy = (
                    self._run_epoch(
                        validation_loader,
                        criterion,
                        None,
                    )
                )

            history.append(
                DamageEpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    validation_loss=val_loss,
                    validation_accuracy=val_accuracy,
                )
            )

            logger.info(
                "Epoch %d | train_loss=%.4f "
                "val_loss=%.4f val_acc=%.4f",
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_accuracy = val_accuracy

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                patience_counter = 0

            else:
                patience_counter += 1

            if (
                patience_counter
                >= self.config.early_stopping_patience
            ):
                logger.info(
                    "Early stopping at epoch %d",
                    epoch,
                )
                break

        if best_state is None:
            raise DamageTrainingError(
                "Training did not produce a checkpoint"
            )

        self.model.load_state_dict(
            best_state
        )

        self.model.eval()

        return DamageTrainingResult(
            training_id=new_id(),
            architecture=self.config.architecture,
            epochs_completed=len(history),
            best_validation_loss=best_loss,
            best_validation_accuracy=best_accuracy,
            history=tuple(history),
            trained_at=utc_now(),
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

class DamageCNNEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: tuple[str, ...],
    ) -> None:
        self.model = model
        self.device = device
        self.class_names = class_names

    @torch.inference_mode()
    def evaluate(
        self,
        loader: DataLoader,
    ) -> DamageEvaluationResult:
        self.model.eval()

        actual: list[int] = []
        predicted: list[int] = []

        for images, labels in loader:
            logits = self.model(
                images.to(self.device)
            )

            predictions = (
                logits.argmax(dim=1)
                .cpu()
                .tolist()
            )

            actual.extend(
                labels.tolist()
            )

            predicted.extend(
                predictions
            )

        if not actual:
            raise DamageValidationError(
                "Evaluation dataset is empty"
            )

        number_of_classes = len(
            self.class_names
        )

        matrix = [
            [0] * number_of_classes
            for _ in range(number_of_classes)
        ]

        for true_label, predicted_label in zip(
            actual,
            predicted,
        ):
            matrix[true_label][predicted_label] += 1

        precision_scores = []
        recall_scores = []
        f1_scores = []

        for class_index in range(number_of_classes):
            tp = matrix[
                class_index
            ][class_index]

            fp = sum(
                matrix[row][class_index]
                for row in range(number_of_classes)
                if row != class_index
            )

            fn = sum(
                matrix[class_index][column]
                for column in range(number_of_classes)
                if column != class_index
            )

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn
                else 0.0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall
                else 0.0
            )

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        accuracy = (
            sum(
                actual_label == predicted_label
                for actual_label, predicted_label
                in zip(actual, predicted)
            ) / len(actual)
        )

        return DamageEvaluationResult(
            sample_count=len(actual),
            accuracy=accuracy,
            macro_precision=(
                sum(precision_scores)
                / number_of_classes
            ),
            macro_recall=(
                sum(recall_scores)
                / number_of_classes
            ),
            macro_f1=(
                sum(f1_scores)
                / number_of_classes
            ),
            confusion_matrix=tuple(
                tuple(row)
                for row in matrix
            ),
            class_names=self.class_names,
        )


# ============================================================
# UNCERTAINTY SCREENING
# ============================================================

class DamageUncertaintyEvaluator:
    def __init__(
        self,
        config: CNNDamageConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        probabilities: Sequence[float],
        predicted_class: str,
    ) -> tuple[
        float,
        float,
        bool,
        tuple[DamageReviewReason, ...],
    ]:
        ordered = sorted(
            probabilities,
            reverse=True,
        )

        confidence = ordered[0]

        margin = (
            ordered[0] - ordered[1]
        )

        epsilon = 1e-12

        entropy = -sum(
            probability
            * math.log(probability + epsilon)
            for probability in probabilities
        )

        normalized_entropy = (
            entropy / math.log(len(probabilities))
        )

        reasons: list[
            DamageReviewReason
        ] = []

        if confidence < self.config.minimum_confidence:
            reasons.append(
                DamageReviewReason.LOW_CONFIDENCE
            )

        if margin < self.config.minimum_probability_margin:
            reasons.append(
                DamageReviewReason.LOW_MARGIN
            )

        if (
            normalized_entropy
            > self.config.maximum_normalized_entropy
        ):
            reasons.append(
                DamageReviewReason.HIGH_ENTROPY
            )

        if (
            self.config.require_review_for_severe
            and predicted_class
            == DamageSeverity.SEVERE.value
        ):
            reasons.append(
                DamageReviewReason.SEVERE_DAMAGE
            )

        return (
            margin,
            normalized_entropy,
            bool(reasons),
            tuple(reasons),
        )


# ============================================================
# INFERENCE ENGINE
# ============================================================

class CNNDamageInferenceEngine:
    def __init__(
        self,
        config: CNNDamageConfig,
        model: nn.Module,
        device: torch.device,
        *,
        model_trained: bool = False,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.model_trained = model_trained

        self.validator = DamageImageValidator(
            config
        )

        self.transform = (
            DamageImageTransforms(config)
            .evaluation()
        )

        self.uncertainty = (
            DamageUncertaintyEvaluator(
                config
            )
        )

    @torch.inference_mode()
    def predict_image(
        self,
        image: Image.Image,
    ) -> DamagePrediction:
        start = time.perf_counter()

        self.model.eval()

        tensor = self.transform(
            image.convert("RGB")
        ).unsqueeze(0).to(
            self.device
        )

        logits = self.model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0].cpu().tolist()

        predicted_index = max(
            range(len(probabilities)),
            key=probabilities.__getitem__,
        )

        predicted_class = (
            self.config.class_names[
                predicted_index
            ]
        )

        confidence = probabilities[
            predicted_index
        ]

        margin, entropy, review, reasons = (
            self.uncertainty.evaluate(
                probabilities,
                predicted_class,
            )
        )

        # An untrained model must never
        # be represented as validated.
        if not self.model_trained:
            review = True

        warnings = [
            "Image classification does not establish "
            "structural or electrical safety."
        ]

        if not self.model_trained:
            warnings.append(
                "Model weights are not marked as trained; "
                "prediction is not reliable."
            )

        if review:
            warnings.append(
                "Human review recommended."
            )

        return DamagePrediction(
            prediction_id=new_id(),
            predicted_class=predicted_class,
            confidence=float(confidence),
            class_probabilities=tuple(
                DamageClassProbability(
                    class_name=name,
                    probability=float(probability),
                )
                for name, probability in zip(
                    self.config.class_names,
                    probabilities,
                )
            ),
            probability_margin=float(margin),
            normalized_entropy=float(entropy),
            status=(
                DamagePredictionStatus.REVIEW_REQUIRED
                if review
                else DamagePredictionStatus.PREDICTED
            ),
            requires_human_review=review,
            review_reasons=reasons,
            model_architecture=self.config.architecture,
            model_trained=self.model_trained,
            inference_time_ms=(
                time.perf_counter() - start
            ) * 1000,
            predicted_at=utc_now(),
            warnings=tuple(warnings),
        )

    def predict_bytes(
        self,
        image_bytes: bytes,
    ) -> DamagePrediction:
        image = self.validator.load_bytes(
            image_bytes
        )

        return self.predict_image(
            image
        )

    def predict_path(
        self,
        path: str | Path,
    ) -> DamagePrediction:
        image = self.validator.load_path(
            path
        )

        return self.predict_image(
            image
        )

    def predict_batch(
        self,
        paths: Sequence[str | Path],
    ) -> tuple[DamagePrediction, ...]:
        return tuple(
            self.predict_path(path)
            for path in paths
        )


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

class DamageCheckpointManager:
    """
    Checkpoints must originate from trusted
    training infrastructure.

    Never load untrusted PyTorch checkpoint
    files supplied by arbitrary users.
    """

    @staticmethod
    def save(
        path: str | Path,
        *,
        model: nn.Module,
        config: CNNDamageConfig,
        training_result: DamageTrainingResult,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise DamageCheckpointError(
                "Checkpoint already exists"
            )

        payload = {
            "format_version": 1,
            "architecture": config.architecture.value,
            "class_names": list(
                config.class_names
            ),
            "image_size": config.image_size,
            "state_dict": model.state_dict(),
            "training_id": training_result.training_id,
            "trained_at": (
                training_result.trained_at.isoformat()
            ),
            "best_validation_accuracy": (
                training_result.best_validation_accuracy
            ),
        }

        try:
            torch.save(
                payload,
                destination,
            )
        except Exception as exc:
            raise DamageCheckpointError(
                "Checkpoint save failed"
            ) from exc

        return destination

    @staticmethod
    def load(
        path: str | Path,
        *,
        config: CNNDamageConfig,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        checkpoint_path = Path(path)

        if not checkpoint_path.is_file():
            raise DamageCheckpointError(
                "Checkpoint does not exist"
            )

        try:
            payload = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )

            if payload.get("format_version") != 1:
                raise DamageCheckpointError(
                    "Unsupported checkpoint format"
                )

            if payload.get("architecture") != (
                config.architecture.value
            ):
                raise DamageCheckpointError(
                    "Checkpoint architecture mismatch"
                )

            if tuple(payload.get("class_names", ())) != (
                config.class_names
            ):
                raise DamageCheckpointError(
                    "Checkpoint class mapping mismatch"
                )

            if payload.get("image_size") != config.image_size:
                raise DamageCheckpointError(
                    "Checkpoint image size mismatch"
                )

            model.load_state_dict(
                payload["state_dict"],
                strict=True,
            )

            model.to(device)
            model.eval()

        except DamageCheckpointError:
            raise

        except Exception as exc:
            raise DamageCheckpointError(
                "Checkpoint load failed"
            ) from exc


# ============================================================
# MAIN SERVICE
# ============================================================

class CNNDamageModelService:
    """
    Application-facing CNN damage service.

    Integrations:
        backend/api/damage.py
        backend/services/damage_service.py
        backend/ai/computer_vision/
        backend/ai/ml/damage_estimation.py
        backend/ai/evaluation/
        backend/mlops/
    """

    def __init__(
        self,
        config: CNNDamageConfig | None = None,
    ) -> None:
        self.config = (
            config or CNNDamageConfig()
        )

        set_random_seed(
            self.config.random_seed
        )

        self.device = get_device(
            self.config.device
        )

        self.model = DamageCNNModelFactory.create(
            self.config
        ).to(self.device)

        self.model_trained = False

        self.dataset_loader = DamageDatasetLoader(
            self.config
        )

        self.trainer = DamageCNNTrainer(
            self.config,
            self.model,
            self.device,
        )

        self.evaluator = DamageCNNEvaluator(
            self.model,
            self.device,
            self.config.class_names,
        )

        self.inference = CNNDamageInferenceEngine(
            self.config,
            self.model,
            self.device,
            model_trained=False,
        )

    def train(
        self,
        dataset_root: str | Path,
        *,
        class_weights: Sequence[float] | None = None,
    ) -> DamageTrainingResult:
        root = Path(dataset_root)

        train_loader = (
            self.dataset_loader.create_loader(
                root / "train",
                training=True,
            )
        )

        validation_loader = (
            self.dataset_loader.create_loader(
                root / "val",
                training=False,
            )
        )

        result = self.trainer.train(
            train_loader,
            validation_loader,
            class_weights=class_weights,
        )

        self.model_trained = True
        self.inference.model_trained = True

        return result

    def evaluate(
        self,
        dataset_root: str | Path,
    ) -> DamageEvaluationResult:
        if not self.model_trained:
            raise DamageValidationError(
                "Train or load a checkpoint before evaluation"
            )

        loader = self.dataset_loader.create_loader(
            Path(dataset_root) / "test",
            training=False,
        )

        return self.evaluator.evaluate(
            loader
        )

    def predict_bytes(
        self,
        image_bytes: bytes,
    ) -> DamagePrediction:
        return self.inference.predict_bytes(
            image_bytes
        )

    def predict_path(
        self,
        path: str | Path,
    ) -> DamagePrediction:
        return self.inference.predict_path(
            path
        )

    def predict_batch(
        self,
        paths: Sequence[str | Path],
    ) -> tuple[DamagePrediction, ...]:
        return self.inference.predict_batch(
            paths
        )

    def save_checkpoint(
        self,
        path: str | Path,
        training_result: DamageTrainingResult,
    ) -> Path:
        if not self.model_trained:
            raise DamageCheckpointError(
                "Cannot save an untrained model"
            )

        return DamageCheckpointManager.save(
            path,
            model=self.model,
            config=self.config,
            training_result=training_result,
        )

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        DamageCheckpointManager.load(
            path,
            config=self.config,
            model=self.model,
            device=self.device,
        )

        self.model_trained = True
        self.inference.model_trained = True


# ============================================================
# FACTORY
# ============================================================

def create_cnn_damage_model_service(
    config: CNNDamageConfig | None = None,
) -> CNNDamageModelService:
    return CNNDamageModelService(
        config=config
    )


__all__ = [
    "DamageModelError",
    "DamageValidationError",
    "DamageTrainingError",
    "DamageInferenceError",
    "DamageCheckpointError",
    "DamageSeverity",
    "CNNArchitecture",
    "DamagePredictionStatus",
    "DamageReviewReason",
    "CNNDamageConfig",
    "DamageClassProbability",
    "DamagePrediction",
    "DamageEpochMetrics",
    "DamageTrainingResult",
    "DamageEvaluationResult",
    "DamageImageValidator",
    "DamageImageTransforms",
    "LightweightDamageCNN",
    "DamageCNNModelFactory",
    "DamageDatasetLoader",
    "DamageCNNTrainer",
    "DamageCNNEvaluator",
    "DamageUncertaintyEvaluator",
    "CNNDamageInferenceEngine",
    "DamageCheckpointManager",
    "CNNDamageModelService",
    "create_cnn_damage_model_service",
]
