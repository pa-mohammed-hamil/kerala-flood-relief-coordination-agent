
"""
Kerala Flood Relief Coordination Agent
Building Damage Detection

File:
    backend/ai/computer_vision/building_damage_detection.py

Python:
    3.11+

Dependencies:
    torch >= 2.2
    torchvision >= 0.17
    Pillow >= 10
    numpy

Features:
    - Building image classification
    - CNN / EfficientNet-B0 architecture
    - Four damage severity classes
    - Image validation
    - Batch prediction
    - Confidence and entropy analysis
    - Human-review flags
    - Inspection priority recommendations
    - Model checkpoint persistence
    - JSON-compatible results

Important:
    This model classifies visible damage
    in an image. It does not perform
    structural engineering assessment.

    Outputs are advisory only.
"""

from __future__ import annotations

import hashlib
import io
import math
import uuid

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms


# ============================================================
# EXCEPTIONS
# ============================================================

class BuildingDamageError(Exception):
    pass


class BuildingDamageValidationError(
    BuildingDamageError
):
    pass


class BuildingDamageModelNotReadyError(
    BuildingDamageError
):
    pass


class BuildingDamageCheckpointError(
    BuildingDamageError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class BuildingDamageSeverity(str, Enum):
    NO_VISIBLE_DAMAGE = "no_visible_damage"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class InspectionPriority(str, Enum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    URGENT = "urgent"
    UNDETERMINED = "undetermined"


class DamagePredictionStatus(str, Enum):
    PREDICTED = "predicted"
    REVIEW_REQUIRED = "review_required"


class DamageReviewReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    LOW_MARGIN = "low_margin"
    HIGH_ENTROPY = "high_entropy"
    SEVERE_DAMAGE = "severe_damage"
    IMAGE_LIMITATION = "image_limitation"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if value not in ("cpu", "cuda"):
        raise BuildingDamageValidationError(
            "device must be auto, cpu, or cuda"
        )

    if (
        value == "cuda"
        and not torch.cuda.is_available()
    ):
        raise BuildingDamageValidationError(
            "CUDA is unavailable"
        )

    return torch.device(value)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class BuildingDamageConfig:
    class_names: tuple[str, ...] = (
        "minor",
        "moderate",
        "no_visible_damage",
        "severe",
    )

    image_size: int = 224

    minimum_confidence: float = 0.70
    minimum_probability_margin: float = 0.15
    maximum_normalized_entropy: float = 0.80

    require_review_for_severe: bool = True

    max_image_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 25_000_000

    max_batch_size: int = 16

    device: str = "auto"

    def __post_init__(self) -> None:
        expected = {
            item.value
            for item in BuildingDamageSeverity
        }

        if set(self.class_names) != expected:
            raise BuildingDamageValidationError(
                "Class names must match "
                "the four supported severities"
            )

        if len(self.class_names) != 4:
            raise BuildingDamageValidationError(
                "Exactly four classes required"
            )

        if self.image_size < 64:
            raise BuildingDamageValidationError(
                "image_size must be >= 64"
            )

        for name in (
            "minimum_confidence",
            "minimum_probability_margin",
            "maximum_normalized_entropy",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 1:
                raise BuildingDamageValidationError(
                    f"{name} must be in [0,1]"
                )

        if (
            self.max_image_bytes <= 0
            or self.max_image_pixels <= 0
            or self.max_batch_size <= 0
        ):
            raise BuildingDamageValidationError(
                "Image and batch limits must be positive"
            )


# ============================================================
# INPUT AND OUTPUT MODELS
# ============================================================

@dataclass(frozen=True)
class BuildingDamageRequest:
    image_bytes: bytes
    filename: str | None = None
    building_id: str | None = None
    district: str | None = None


@dataclass(frozen=True)
class BuildingDamagePrediction:
    prediction_id: str

    building_id: str | None
    district: str | None

    severity: BuildingDamageSeverity
    confidence: float

    class_probabilities: dict[str, float]

    probability_margin: float
    normalized_entropy: float

    inspection_priority: InspectionPriority
    status: DamagePredictionStatus

    review_reasons: tuple[DamageReviewReason, ...]
    warnings: tuple[str, ...]

    image_sha256: str

    predicted_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "building_id": self.building_id,
            "district": self.district,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "class_probabilities": (
                self.class_probabilities
            ),
            "probability_margin": (
                self.probability_margin
            ),
            "normalized_entropy": (
                self.normalized_entropy
            ),
            "inspection_priority": (
                self.inspection_priority.value
            ),
            "status": self.status.value,
            "review_reasons": [
                item.value
                for item in self.review_reasons
            ],
            "warnings": list(self.warnings),
            "image_sha256": self.image_sha256,
            "predicted_at": (
                self.predicted_at.isoformat()
            ),
        }


@dataclass(frozen=True)
class BuildingDamageBatchResult:
    predictions: tuple[
        BuildingDamagePrediction,
        ...
    ]

    total_images: int
    review_required_count: int


# ============================================================
# IMAGE VALIDATOR
# ============================================================

class BuildingImageValidator:
    def __init__(
        self,
        config: BuildingDamageConfig,
    ) -> None:
        self.config = config

    def load_bytes(
        self,
        data: bytes,
    ) -> Image.Image:
        if not data:
            raise BuildingDamageValidationError(
                "Image is empty"
            )

        if len(data) > self.config.max_image_bytes:
            raise BuildingDamageValidationError(
                "Image exceeds maximum byte size"
            )

        try:
            with Image.open(
                io.BytesIO(data)
            ) as source:
                if source.format not in (
                    "JPEG",
                    "PNG",
                    "WEBP",
                ):
                    raise BuildingDamageValidationError(
                        "Unsupported image format"
                    )

                if (
                    source.width * source.height
                    > self.config.max_image_pixels
                ):
                    raise BuildingDamageValidationError(
                        "Image exceeds pixel limit"
                    )

                image = ImageOps.exif_transpose(
                    source
                ).convert("RGB")

                return image.copy()

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            raise BuildingDamageValidationError(
                "Invalid or unsafe image"
            ) from exc

    def load_path(
        self,
        path: str | Path,
    ) -> Image.Image:
        source = Path(path)

        if not source.is_file():
            raise BuildingDamageValidationError(
                "Image file not found"
            )

        if (
            source.stat().st_size
            > self.config.max_image_bytes
        ):
            raise BuildingDamageValidationError(
                "Image exceeds maximum byte size"
            )

        return self.load_bytes(
            source.read_bytes()
        )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

class BuildingImagePreprocessor:
    def __init__(
        self,
        config: BuildingDamageConfig,
    ) -> None:
        self.config = config

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        config.image_size,
                        config.image_size,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(
                        0.485,
                        0.456,
                        0.406,
                    ),
                    std=(
                        0.229,
                        0.224,
                        0.225,
                    ),
                ),
            ]
        )

    def transform_image(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        return self.transform(
            image
        )


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class BuildingDamageCNN(nn.Module):
    """
    EfficientNet-B0 image classifier.

    Input:
        B x 3 x 224 x 224

    Output:
        B x 4 class logits

    Architecture only; weights must
    be trained on labeled building imagery.
    """

    def __init__(
        self,
        config: BuildingDamageConfig,
        *,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained_backbone
            else None
        )

        self.network = models.efficientnet_b0(
            weights=weights
        )

        input_features = (
            self.network.classifier[1].in_features
        )

        self.network.classifier[1] = nn.Linear(
            input_features,
            len(config.class_names),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            images
        )


# ============================================================
# UNCERTAINTY ANALYZER
# ============================================================

class BuildingDamageUncertaintyAnalyzer:
    def __init__(
        self,
        config: BuildingDamageConfig,
    ) -> None:
        self.config = config

    def analyze(
        self,
        probabilities: np.ndarray,
        severity: BuildingDamageSeverity,
    ) -> tuple[
        float,
        float,
        tuple[DamageReviewReason, ...],
    ]:
        sorted_probabilities = np.sort(
            probabilities
        )[::-1]

        margin = float(
            sorted_probabilities[0]
            - sorted_probabilities[1]
        )

        safe_probabilities = np.clip(
            probabilities,
            1e-12,
            1.0,
        )

        entropy = float(
            -np.sum(
                safe_probabilities
                * np.log(safe_probabilities)
            )
            / math.log(
                len(probabilities)
            )
        )

        reasons = []

        if (
            float(probabilities.max())
            < self.config.minimum_confidence
        ):
            reasons.append(
                DamageReviewReason.LOW_CONFIDENCE
            )

        if (
            margin
            < self.config.minimum_probability_margin
        ):
            reasons.append(
                DamageReviewReason.LOW_MARGIN
            )

        if (
            entropy
            > self.config.maximum_normalized_entropy
        ):
            reasons.append(
                DamageReviewReason.HIGH_ENTROPY
            )

        if (
            severity == BuildingDamageSeverity.SEVERE
            and self.config.require_review_for_severe
        ):
            reasons.append(
                DamageReviewReason.SEVERE_DAMAGE
            )

        return (
            margin,
            entropy,
            tuple(reasons),
        )


# ============================================================
# INSPECTION PRIORITY ADVISOR
# ============================================================

class BuildingInspectionAdvisor:
    """
    Provides an advisory inspection
    priority, not a safety clearance.

    No building is declared safe
    for occupation by this component.
    """

    @staticmethod
    def recommend(
        severity: BuildingDamageSeverity,
        *,
        uncertain: bool,
    ) -> InspectionPriority:
        if uncertain:
            return InspectionPriority.UNDETERMINED

        if severity == BuildingDamageSeverity.SEVERE:
            return InspectionPriority.URGENT

        if severity == BuildingDamageSeverity.MODERATE:
            return InspectionPriority.ELEVATED

        return InspectionPriority.ROUTINE


# ============================================================
# INFERENCE ENGINE
# ============================================================

class BuildingDamageInferenceEngine:
    def __init__(
        self,
        config: BuildingDamageConfig,
        model: BuildingDamageCNN,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.validator = BuildingImageValidator(
            config
        )

        self.preprocessor = BuildingImagePreprocessor(
            config
        )

        self.uncertainty = (
            BuildingDamageUncertaintyAnalyzer(
                config
            )
        )

        self.advisor = BuildingInspectionAdvisor()

    @torch.inference_mode()
    def predict(
        self,
        request: BuildingDamageRequest,
    ) -> BuildingDamagePrediction:
        image = self.validator.load_bytes(
            request.image_bytes
        )

        tensor = self.preprocessor.transform_image(
            image
        ).unsqueeze(0).to(
            self.device
        )

        self.model.eval()

        logits = self.model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0].cpu().numpy()

        index = int(
            probabilities.argmax()
        )

        severity = BuildingDamageSeverity(
            self.config.class_names[index]
        )

        confidence = float(
            probabilities[index]
        )

        margin, entropy, reasons = (
            self.uncertainty.analyze(
                probabilities,
                severity,
            )
        )

        status = (
            DamagePredictionStatus.REVIEW_REQUIRED
            if reasons
            else DamagePredictionStatus.PREDICTED
        )

        uncertain = any(
            reason in (
                DamageReviewReason.LOW_CONFIDENCE,
                DamageReviewReason.LOW_MARGIN,
                DamageReviewReason.HIGH_ENTROPY,
            )
            for reason in reasons
        )

        priority = self.advisor.recommend(
            severity,
            uncertain=uncertain,
        )

        warnings = [
            "AI image classification is advisory only",
            "Visible damage does not establish structural safety",
            "A qualified inspector must assess building stability",
        ]

        if reasons:
            warnings.append(
                "Prediction requires human review"
            )

        return BuildingDamagePrediction(
            prediction_id=new_id(),
            building_id=request.building_id,
            district=request.district,
            severity=severity,
            confidence=confidence,
            class_probabilities={
                name: float(probabilities[i])
                for i, name in enumerate(
                    self.config.class_names
                )
            },
            probability_margin=margin,
            normalized_entropy=entropy,
            inspection_priority=priority,
            status=status,
            review_reasons=reasons,
            warnings=tuple(warnings),
            image_sha256=hashlib.sha256(
                request.image_bytes
            ).hexdigest(),
            predicted_at=utc_now(),
        )


# ============================================================
# BATCH INFERENCE
# ============================================================

class BuildingDamageBatchInference:
    def __init__(
        self,
        engine: BuildingDamageInferenceEngine,
        config: BuildingDamageConfig,
    ) -> None:
        self.engine = engine
        self.config = config

    def predict_batch(
        self,
        requests: Sequence[BuildingDamageRequest],
    ) -> BuildingDamageBatchResult:
        if not requests:
            raise BuildingDamageValidationError(
                "Batch is empty"
            )

        if len(requests) > self.config.max_batch_size:
            raise BuildingDamageValidationError(
                "Batch exceeds maximum size"
            )

        predictions = tuple(
            self.engine.predict(request)
            for request in requests
        )

        review_count = sum(
            prediction.status
            == DamagePredictionStatus.REVIEW_REQUIRED
            for prediction in predictions
        )

        return BuildingDamageBatchResult(
            predictions=predictions,
            total_images=len(predictions),
            review_required_count=review_count,
        )


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class BuildingDamageCheckpointManager:
    @staticmethod
    def save(
        path: str | Path,
        *,
        model: BuildingDamageCNN,
        config: BuildingDamageConfig,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise BuildingDamageCheckpointError(
                "Checkpoint already exists"
            )

        with destination.open("xb") as stream:
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "saved_at": utc_now().isoformat(),
                },
                stream,
            )

        return destination

    @staticmethod
    def load(
        path: str | Path,
        *,
        model: BuildingDamageCNN,
        config: BuildingDamageConfig,
        device: torch.device,
    ) -> None:
        checkpoint = torch.load(
            Path(path),
            map_location=device,
            weights_only=True,
        )

        if checkpoint.get("format_version") != 1:
            raise BuildingDamageCheckpointError(
                "Unsupported checkpoint version"
            )

        saved_config = checkpoint.get(
            "config",
            {},
        )

        if tuple(
            saved_config.get(
                "class_names",
                (),
            )
        ) != config.class_names:
            raise BuildingDamageCheckpointError(
                "Class mapping mismatch"
            )

        if saved_config.get(
            "image_size"
        ) != config.image_size:
            raise BuildingDamageCheckpointError(
                "Image size mismatch"
            )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        model.to(device)
        model.eval()


# ============================================================
# APPLICATION SERVICE
# ============================================================

class BuildingDamageDetectionService:
    def __init__(
        self,
        config: BuildingDamageConfig | None = None,
    ) -> None:
        self.config = (
            config or BuildingDamageConfig()
        )

        self.device = resolve_device(
            self.config.device
        )

        self.model = BuildingDamageCNN(
            self.config,
            pretrained_backbone=False,
        ).to(self.device)

        self.engine = BuildingDamageInferenceEngine(
            self.config,
            self.model,
            self.device,
        )

        self.batch_engine = (
            BuildingDamageBatchInference(
                self.engine,
                self.config,
            )
        )

        self.trained = False

    def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        building_id: str | None = None,
        district: str | None = None,
        filename: str | None = None,
    ) -> BuildingDamagePrediction:
        if not self.trained:
            raise BuildingDamageModelNotReadyError(
                "Load a trained checkpoint first"
            )

        return self.engine.predict(
            BuildingDamageRequest(
                image_bytes=image_bytes,
                filename=filename,
                building_id=building_id,
                district=district,
            )
        )

    def predict_path(
        self,
        path: str | Path,
        *,
        building_id: str | None = None,
        district: str | None = None,
    ) -> BuildingDamagePrediction:
        source = Path(path)

        if not source.is_file():
            raise BuildingDamageValidationError(
                "Image file not found"
            )

        if source.stat().st_size > (
            self.config.max_image_bytes
        ):
            raise BuildingDamageValidationError(
                "Image exceeds maximum byte size"
            )

        return self.predict_bytes(
            source.read_bytes(),
            building_id=building_id,
            district=district,
            filename=source.name,
        )

    def predict_batch(
        self,
        requests: Sequence[BuildingDamageRequest],
    ) -> BuildingDamageBatchResult:
        if not self.trained:
            raise BuildingDamageModelNotReadyError(
                "Load a trained checkpoint first"
            )

        return self.batch_engine.predict_batch(
            requests
        )

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        BuildingDamageCheckpointManager.load(
            path,
            model=self.model,
            config=self.config,
            device=self.device,
        )

        self.trained = True

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> Path:
        if not self.trained:
            raise BuildingDamageModelNotReadyError(
                "Cannot save an untrained model"
            )

        return BuildingDamageCheckpointManager.save(
            path,
            model=self.model,
            config=self.config,
        )


# ============================================================
# FACTORY
# ============================================================

def create_building_damage_service(
    config: BuildingDamageConfig | None = None,
) -> BuildingDamageDetectionService:
    return BuildingDamageDetectionService(
        config=config
    )


__all__ = [
    "BuildingDamageError",
    "BuildingDamageValidationError",
    "BuildingDamageModelNotReadyError",
    "BuildingDamageCheckpointError",
    "BuildingDamageSeverity",
    "InspectionPriority",
    "DamagePredictionStatus",
    "DamageReviewReason",
    "BuildingDamageConfig",
    "BuildingDamageRequest",
    "BuildingDamagePrediction",
    "BuildingDamageBatchResult",
    "BuildingImageValidator",
    "BuildingImagePreprocessor",
    "BuildingDamageCNN",
    "BuildingDamageUncertaintyAnalyzer",
    "BuildingInspectionAdvisor",
    "BuildingDamageInferenceEngine",
    "BuildingDamageBatchInference",
    "BuildingDamageCheckpointManager",
    "BuildingDamageDetectionService",
    "create_building_damage_service",
]
