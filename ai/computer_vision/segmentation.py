
"""
Kerala Flood Relief Coordination Agent
File: backend/ai/computer_vision/segmentation.py

Reusable semantic segmentation inference service.

Supports:
    - Flood-water segmentation
    - Road segmentation
    - Building segmentation
    - Debris segmentation
    - Multi-class semantic segmentation
    - Binary segmentation
    - PyTorch segmentation models
    - Image preprocessing
    - Confidence and uncertainty analysis
    - Per-class pixel coverage
    - Segmentation overlays
    - Batch inference
    - Safe checkpoint loading
    - JSON report export
    - Human-review requirements

Dependencies:
    pip install torch numpy Pillow

Important:
    A trained model and a matching preprocessing
    configuration are required.

    Segmentation results are advisory only.
    No automatic road closures, rescue dispatch,
    evacuation or emergency alerts.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import uuid

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps


# ============================================================
# EXCEPTIONS
# ============================================================

class SegmentationError(Exception):
    pass


class SegmentationValidationError(
    SegmentationError
):
    pass


class SegmentationModelNotReadyError(
    SegmentationError
):
    pass


class SegmentationCheckpointError(
    SegmentationError
):
    pass


class SegmentationExportError(
    SegmentationError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class SegmentationTask(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"


class SegmentationStatus(str, Enum):
    COMPLETED = "completed"
    REVIEW_REQUIRED = "review_required"


class NormalizationMode(str, Enum):
    ZERO_ONE = "zero_one"
    IMAGENET = "imagenet"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SegmentationConfig:
    class_names: tuple[str, ...] = (
        "background",
        "flood_water",
        "road",
        "building",
        "debris",
    )

    task: SegmentationTask = (
        SegmentationTask.MULTICLASS
    )

    image_size: int = 512

    confidence_threshold: float = 0.70
    binary_threshold: float = 0.50

    maximum_uncertain_fraction: float = 0.30

    normalization: NormalizationMode = (
        NormalizationMode.ZERO_ONE
    )

    max_image_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    maximum_batch_size: int = 12

    device: str = "auto"

    require_human_review: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.image_size, int)
            or self.image_size < 32
            or self.image_size % 16 != 0
        ):
            raise SegmentationValidationError(
                "image_size must be >=32 "
                "and divisible by 16"
            )

        if not self.class_names:
            raise SegmentationValidationError(
                "class_names cannot be empty"
            )

        if len(set(self.class_names)) != len(
            self.class_names
        ):
            raise SegmentationValidationError(
                "Duplicate class names"
            )

        if self.task == SegmentationTask.BINARY:
            if len(self.class_names) != 2:
                raise SegmentationValidationError(
                    "Binary segmentation requires "
                    "two class names"
                )

        elif len(self.class_names) < 2:
            raise SegmentationValidationError(
                "Multiclass segmentation requires "
                "at least two classes"
            )

        for name in (
            "confidence_threshold",
            "binary_threshold",
            "maximum_uncertain_fraction",
        ):
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise SegmentationValidationError(
                    f"{name} must be in [0,1]"
                )

        for name in (
            "max_image_bytes",
            "max_image_pixels",
            "maximum_batch_size",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SegmentationValidationError(
                    f"{name} must be positive"
                )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class SegmentationClassResult:
    class_id: int
    class_name: str

    pixel_count: int
    coverage_percent: float

    mean_class_probability: float


@dataclass(frozen=True)
class SegmentationResult:
    analysis_id: str
    image_id: str

    filename: str | None

    image_width: int
    image_height: int

    task: SegmentationTask

    class_results: tuple[
        SegmentationClassResult,
        ...
    ]

    mean_pixel_confidence: float
    uncertain_pixel_fraction: float

    status: SegmentationStatus
    requires_human_review: bool

    image_sha256: str

    warnings: tuple[str, ...]

    analyzed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["task"] = self.task.value
        data["status"] = self.status.value

        data["analyzed_at"] = (
            self.analyzed_at.isoformat()
        )

        return data


@dataclass(frozen=True)
class SegmentationOutput:
    result: SegmentationResult

    class_mask: Image.Image
    color_mask: Image.Image
    overlay: Image.Image

    probability_maps: np.ndarray


@dataclass(frozen=True)
class SegmentationBatchResult:
    batch_id: str

    total_images: int
    review_required_count: int

    results: tuple[
        SegmentationResult,
        ...
    ]

    generated_at: datetime


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def resolve_device(
    value: str,
) -> torch.device:
    if value == "auto":
        value = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if value not in ("cpu", "cuda"):
        raise SegmentationValidationError(
            "device must be auto, cpu, or cuda"
        )

    if (
        value == "cuda"
        and not torch.cuda.is_available()
    ):
        raise SegmentationValidationError(
            "CUDA is unavailable"
        )

    return torch.device(value)


# ============================================================
# IMAGE VALIDATOR
# ============================================================

class SegmentationImageValidator:
    def __init__(
        self,
        config: SegmentationConfig,
    ) -> None:
        self.config = config

    def load_bytes(
        self,
        data: bytes,
    ) -> Image.Image:
        if not data:
            raise SegmentationValidationError(
                "Image is empty"
            )

        if (
            len(data)
            > self.config.max_image_bytes
        ):
            raise SegmentationValidationError(
                "Image exceeds byte limit"
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
                    raise SegmentationValidationError(
                        "Unsupported image format"
                    )

                if (
                    source.width * source.height
                    > self.config.max_image_pixels
                ):
                    raise SegmentationValidationError(
                        "Image exceeds pixel limit"
                    )

                return ImageOps.exif_transpose(
                    source
                ).convert("RGB").copy()

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            raise SegmentationValidationError(
                "Invalid image"
            ) from exc

    def load_path(
        self,
        path: str | Path,
    ) -> tuple[Image.Image, bytes]:
        source = Path(path)

        if not source.is_file():
            raise SegmentationValidationError(
                "Image not found"
            )

        if (
            source.stat().st_size
            > self.config.max_image_bytes
        ):
            raise SegmentationValidationError(
                "Image exceeds byte limit"
            )

        data = source.read_bytes()

        return self.load_bytes(data), data


# ============================================================
# IMAGE PREPROCESSOR
# ============================================================

class SegmentationPreprocessor:
    IMAGENET_MEAN = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    IMAGENET_STD = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    def __init__(
        self,
        config: SegmentationConfig,
    ) -> None:
        self.config = config

    def preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        resized = image.resize(
            (
                self.config.image_size,
                self.config.image_size,
            ),
            Image.Resampling.BILINEAR,
        )

        array = np.asarray(
            resized,
            dtype=np.float32,
        ) / 255.0

        if (
            self.config.normalization
            == NormalizationMode.IMAGENET
        ):
            array = (
                array - self.IMAGENET_MEAN
            ) / self.IMAGENET_STD

        tensor = torch.from_numpy(
            array.transpose(
                2,
                0,
                1,
            ).copy()
        )

        return tensor.unsqueeze(0)


# ============================================================
# GENERIC SEGMENTATION MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


class SegmentationUNet(nn.Module):
    def __init__(
        self,
        output_channels: int,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        b = base_channels

        self.enc1 = DoubleConv(3, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)

        self.bottleneck = DoubleConv(
            b * 8,
            b * 16,
        )

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(
            b * 16,
            b * 8,
            kernel_size=2,
            stride=2,
        )

        self.dec4 = DoubleConv(
            b * 16,
            b * 8,
        )

        self.up3 = nn.ConvTranspose2d(
            b * 8,
            b * 4,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            b * 8,
            b * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            b * 4,
            b * 2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            b * 4,
            b * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            b * 2,
            b,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            b * 2,
            b,
        )

        self.head = nn.Conv2d(
            b,
            output_channels,
            kernel_size=1,
        )

    @staticmethod
    def concatenate(
        upsampled: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return torch.cat(
            [skip, upsampled],
            dim=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        bottleneck = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.dec4(
            self.concatenate(
                self.up4(bottleneck),
                e4,
            )
        )

        d3 = self.dec3(
            self.concatenate(
                self.up3(d4),
                e3,
            )
        )

        d2 = self.dec2(
            self.concatenate(
                self.up2(d3),
                e2,
            )
        )

        d1 = self.dec1(
            self.concatenate(
                self.up1(d2),
                e1,
            )
        )

        return self.head(d1)


# ============================================================
# CLASS COLORS
# ============================================================

class SegmentationColorPalette:
    COLORS = (
        (0, 0, 0),
        (0, 130, 255),
        (255, 190, 0),
        (255, 70, 70),
        (160, 90, 220),
        (0, 200, 120),
        (255, 120, 0),
        (200, 200, 200),
    )

    @classmethod
    def color_for_class(
        cls,
        class_id: int,
    ) -> tuple[int, int, int]:
        if class_id < len(cls.COLORS):
            return cls.COLORS[class_id]

        # Deterministic fallback for extra classes.
        return (
            (class_id * 67) % 256,
            (class_id * 131) % 256,
            (class_id * 197) % 256,
        )


# ============================================================
# MASK RENDERER
# ============================================================

class SegmentationRenderer:
    @staticmethod
    def create_class_mask(
        class_map: np.ndarray,
    ) -> Image.Image:
        if (
            class_map.ndim != 2
            or np.any(class_map < 0)
            or np.any(class_map > 65535)
        ):
            raise SegmentationValidationError(
                "Invalid class map"
            )

        if int(class_map.max()) <= 255:
            return Image.fromarray(
                class_map.astype(np.uint8),
                mode="L",
            )

        return Image.fromarray(
            class_map.astype(np.uint16),
            mode="I;16",
        )

    @staticmethod
    def create_color_mask(
        class_map: np.ndarray,
        number_of_classes: int,
    ) -> Image.Image:
        height, width = class_map.shape

        output = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        for class_id in range(
            number_of_classes
        ):
            output[
                class_map == class_id
            ] = (
                SegmentationColorPalette
                .color_for_class(class_id)
            )

        return Image.fromarray(
            output,
            mode="RGB",
        )

    @staticmethod
    def create_overlay(
        image: Image.Image,
        class_map: np.ndarray,
        number_of_classes: int,
        *,
        opacity: float = 0.45,
    ) -> Image.Image:
        if not 0 <= opacity <= 1:
            raise SegmentationValidationError(
                "opacity must be in [0,1]"
            )

        original = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        ).copy()

        color_mask = np.asarray(
            SegmentationRenderer.create_color_mask(
                class_map,
                number_of_classes,
            ),
            dtype=np.uint8,
        )

        if original.shape != color_mask.shape:
            raise SegmentationValidationError(
                "Overlay shape mismatch"
            )

        # Preserve background pixels.
        foreground = (
            class_map != 0
        )

        original[foreground] = (
            (1 - opacity)
            * original[foreground].astype(
                np.float32
            )
            + opacity
            * color_mask[foreground].astype(
                np.float32
            )
        ).astype(np.uint8)

        return Image.fromarray(
            original,
            mode="RGB",
        )


# ============================================================
# METRICS ANALYZER
# ============================================================

class SegmentationMetricsAnalyzer:
    def __init__(
        self,
        config: SegmentationConfig,
    ) -> None:
        self.config = config

    def analyze(
        self,
        class_map: np.ndarray,
        probability_maps: np.ndarray,
    ) -> tuple[
        tuple[SegmentationClassResult, ...],
        float,
        float,
        bool,
    ]:
        if probability_maps.ndim != 3:
            raise SegmentationValidationError(
                "Probabilities must have shape C,H,W"
            )

        if not np.isfinite(
            probability_maps
        ).all():
            raise SegmentationValidationError(
                "Non-finite probabilities"
            )

        if (
            np.any(probability_maps < 0)
            or np.any(probability_maps > 1)
        ):
            raise SegmentationValidationError(
                "Probabilities must be in [0,1]"
            )

        confidence_map = np.max(
            probability_maps,
            axis=0,
        )

        mean_confidence = float(
            confidence_map.mean()
        )

        uncertain_fraction = float(
            (
                confidence_map
                < self.config.confidence_threshold
            ).mean()
        )

        total_pixels = int(
            class_map.size
        )

        class_results = []

        for class_id, class_name in enumerate(
            self.config.class_names
        ):
            mask = (
                class_map == class_id
            )

            pixel_count = int(
                mask.sum()
            )

            coverage = (
                100.0
                * pixel_count
                / total_pixels
            )

            class_probabilities = (
                probability_maps[class_id]
            )

            mean_probability = (
                float(
                    class_probabilities[mask].mean()
                )
                if pixel_count > 0
                else 0.0
            )

            class_results.append(
                SegmentationClassResult(
                    class_id=class_id,
                    class_name=class_name,
                    pixel_count=pixel_count,
                    coverage_percent=coverage,
                    mean_class_probability=(
                        mean_probability
                    ),
                )
            )

        requires_review = (
            mean_confidence
            < self.config.confidence_threshold
            or uncertain_fraction
            > self.config.maximum_uncertain_fraction
        )

        return (
            tuple(class_results),
            mean_confidence,
            uncertain_fraction,
            requires_review,
        )


# ============================================================
# INFERENCE ENGINE
# ============================================================

class SegmentationInferenceEngine:
    def __init__(
        self,
        config: SegmentationConfig,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.preprocessor = (
            SegmentationPreprocessor(
                config
            )
        )

        self.metrics = (
            SegmentationMetricsAnalyzer(
                config
            )
        )

        self.renderer = (
            SegmentationRenderer()
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        *,
        image_bytes: bytes,
        image_id: str | None = None,
        filename: str | None = None,
    ) -> SegmentationOutput:
        tensor = self.preprocessor.preprocess(
            image
        ).to(self.device)

        self.model.eval()

        logits = self.model(
            tensor
        )

        expected_channels = (
            1
            if self.config.task
            == SegmentationTask.BINARY
            else len(self.config.class_names)
        )

        if (
            logits.ndim != 4
            or logits.shape[0] != 1
            or logits.shape[1]
            != expected_channels
        ):
            raise SegmentationError(
                "Unexpected model output shape"
            )

        # Resize logits before activation
        # to restore original image size.
        logits = F.interpolate(
            logits,
            size=(
                image.height,
                image.width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        if (
            self.config.task
            == SegmentationTask.BINARY
        ):
            foreground_probability = (
                torch.sigmoid(
                    logits
                )[0, 0]
            )

            probability_maps = torch.stack(
                [
                    1.0 - foreground_probability,
                    foreground_probability,
                ],
                dim=0,
            )

            class_map = (
                foreground_probability
                >= self.config.binary_threshold
            ).long()

        else:
            probability_maps = (
                torch.softmax(
                    logits,
                    dim=1,
                )[0]
            )

            class_map = torch.argmax(
                probability_maps,
                dim=0,
            )

        probability_array = (
            probability_maps.cpu().numpy()
        )

        class_array = (
            class_map.cpu().numpy()
            .astype(np.int32)
        )

        (
            class_results,
            mean_confidence,
            uncertain_fraction,
            uncertainty_review,
        ) = self.metrics.analyze(
            class_array,
            probability_array,
        )

        requires_review = (
            self.config.require_human_review
            or uncertainty_review
        )

        status = (
            SegmentationStatus.REVIEW_REQUIRED
            if requires_review
            else SegmentationStatus.COMPLETED
        )

        warnings = [
            "Segmentation results are advisory only",
            "Pixel labels require model validation",
            "Occlusion, shadows and lighting may reduce accuracy",
            "Pixel coverage is not verified geographic area",
            "Human review is required for emergency decisions",
        ]

        if uncertainty_review:
            warnings.append(
                "Prediction uncertainty exceeds "
                "configured thresholds"
            )

        result = SegmentationResult(
            analysis_id=generate_id(),
            image_id=image_id or generate_id(),
            filename=filename,
            image_width=image.width,
            image_height=image.height,
            task=self.config.task,
            class_results=class_results,
            mean_pixel_confidence=mean_confidence,
            uncertain_pixel_fraction=(
                uncertain_fraction
            ),
            status=status,
            requires_human_review=requires_review,
            image_sha256=hashlib.sha256(
                image_bytes
            ).hexdigest(),
            warnings=tuple(warnings),
            analyzed_at=utc_now(),
        )

        class_mask = (
            self.renderer.create_class_mask(
                class_array
            )
        )

        color_mask = (
            self.renderer.create_color_mask(
                class_array,
                len(self.config.class_names),
            )
        )

        overlay = self.renderer.create_overlay(
            image,
            class_array,
            len(self.config.class_names),
        )

        return SegmentationOutput(
            result=result,
            class_mask=class_mask,
            color_mask=color_mask,
            overlay=overlay,
            probability_maps=probability_array,
        )


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class SegmentationCheckpointManager:
    @staticmethod
    def save(
        path: str | Path,
        *,
        model: nn.Module,
        config: SegmentationConfig,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise SegmentationCheckpointError(
                "Checkpoint already exists"
            )

        payload = {
            "model_state": model.state_dict(),
            "class_names": list(
                config.class_names
            ),
            "task": config.task.value,
            "image_size": config.image_size,
            "normalization": (
                config.normalization.value
            ),
        }

        torch.save(
            payload,
            destination,
        )

        return destination

    @staticmethod
    def load(
        path: str | Path,
        *,
        model: nn.Module,
        config: SegmentationConfig,
        device: torch.device,
    ) -> None:
        source = Path(path)

        if not source.is_file():
            raise SegmentationCheckpointError(
                "Checkpoint not found"
            )

        try:
            payload = torch.load(
                source,
                map_location=device,
                weights_only=True,
            )

            if not isinstance(payload, dict):
                raise SegmentationCheckpointError(
                    "Invalid checkpoint format"
                )

            if tuple(
                payload["class_names"]
            ) != config.class_names:
                raise SegmentationCheckpointError(
                    "Checkpoint class names differ"
                )

            if payload["task"] != config.task.value:
                raise SegmentationCheckpointError(
                    "Checkpoint task differs"
                )

            if (
                payload["image_size"]
                != config.image_size
            ):
                raise SegmentationCheckpointError(
                    "Checkpoint image size differs"
                )

            if (
                payload["normalization"]
                != config.normalization.value
            ):
                raise SegmentationCheckpointError(
                    "Checkpoint normalization differs"
                )

            model.load_state_dict(
                payload["model_state"],
                strict=True,
            )

            model.to(device)
            model.eval()

        except (
            RuntimeError,
            KeyError,
            ValueError,
            TypeError,
        ) as exc:
            raise SegmentationCheckpointError(
                "Unable to load checkpoint"
            ) from exc


# ============================================================
# SEGMENTATION SERVICE
# ============================================================

class SegmentationService:
    def __init__(
        self,
        config: SegmentationConfig | None = None,
        *,
        model: nn.Module | None = None,
    ) -> None:
        self.config = (
            config or SegmentationConfig()
        )

        self.device = resolve_device(
            self.config.device
        )

        output_channels = (
            1
            if self.config.task
            == SegmentationTask.BINARY
            else len(self.config.class_names)
        )

        self.model = (
            model
            if model is not None
            else SegmentationUNet(
                output_channels=output_channels
            )
        ).to(self.device)

        self.validator = SegmentationImageValidator(
            self.config
        )

        self.engine = SegmentationInferenceEngine(
            self.config,
            self.model,
            self.device,
        )

        self.model_ready = False

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        SegmentationCheckpointManager.load(
            path,
            model=self.model,
            config=self.config,
            device=self.device,
        )

        self.model_ready = True

    def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        image_id: str | None = None,
        filename: str | None = None,
    ) -> SegmentationOutput:
        if not self.model_ready:
            raise SegmentationModelNotReadyError(
                "Load a trained checkpoint first"
            )

        image = self.validator.load_bytes(
            image_bytes
        )

        return self.engine.predict(
            image,
            image_bytes=image_bytes,
            image_id=image_id,
            filename=filename,
        )

    def predict_path(
        self,
        path: str | Path,
        *,
        image_id: str | None = None,
    ) -> SegmentationOutput:
        if not self.model_ready:
            raise SegmentationModelNotReadyError(
                "Load a trained checkpoint first"
            )

        image, data = self.validator.load_path(
            path
        )

        return self.engine.predict(
            image,
            image_bytes=data,
            image_id=image_id,
            filename=Path(path).name,
        )

    def predict_batch(
        self,
        images: Sequence[tuple[str, bytes]],
    ) -> SegmentationBatchResult:
        if not images:
            raise SegmentationValidationError(
                "Batch cannot be empty"
            )

        if (
            len(images)
            > self.config.maximum_batch_size
        ):
            raise SegmentationValidationError(
                "Batch exceeds maximum size"
            )

        results = []

        for filename, image_bytes in images:
            output = self.predict_bytes(
                image_bytes,
                filename=filename,
            )

            results.append(
                output.result
            )

        return SegmentationBatchResult(
            batch_id=generate_id(),
            total_images=len(results),
            review_required_count=sum(
                item.requires_human_review
                for item in results
            ),
            results=tuple(results),
            generated_at=utc_now(),
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class SegmentationReportExporter:
    @staticmethod
    def export_json(
        result: SegmentationResult,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise SegmentationExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                result.to_dict(),
                stream,
                indent=2,
                allow_nan=False,
            )

        return destination

    @staticmethod
    def export_images(
        output: SegmentationOutput,
        directory: str | Path,
    ) -> dict[str, str]:
        destination = Path(directory)

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        analysis_id = (
            output.result.analysis_id
        )

        paths = {
            "class_mask": (
                destination
                / f"{analysis_id}_classes.png"
            ),
            "color_mask": (
                destination
                / f"{analysis_id}_colors.png"
            ),
            "overlay": (
                destination
                / f"{analysis_id}_overlay.png"
            ),
            "probabilities": (
                destination
                / f"{analysis_id}_probabilities.npy"
            ),
        }

        if any(
            path.exists()
            for path in paths.values()
        ):
            raise SegmentationExportError(
                "Output already exists"
            )

        output.class_mask.save(
            paths["class_mask"]
        )

        output.color_mask.save(
            paths["color_mask"]
        )

        output.overlay.save(
            paths["overlay"]
        )

        with paths["probabilities"].open("xb") as stream:
            np.save(
                stream,
                output.probability_maps,
                allow_pickle=False,
            )

        return {
            key: str(path)
            for key, path in paths.items()
        }


# ============================================================
# FACTORIES
# ============================================================

def create_segmentation_service(
    config: SegmentationConfig | None = None,
) -> SegmentationService:
    return SegmentationService(
        config=config
    )


def create_flood_segmentation_service() -> (
    SegmentationService
):
    return SegmentationService(
        SegmentationConfig(
            class_names=(
                "background",
                "flood_water",
            ),
            task=SegmentationTask.BINARY,
            image_size=256,
            normalization=NormalizationMode.ZERO_ONE,
        )
    )


def create_disaster_scene_segmentation_service() -> (
    SegmentationService
):
    return SegmentationService(
        SegmentationConfig(
            class_names=(
                "background",
                "flood_water",
                "road",
                "building",
                "debris",
            ),
            task=SegmentationTask.MULTICLASS,
            image_size=512,
        )
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    service = (
        create_disaster_scene_segmentation_service()
    )

    # Must be trained using the same U-Net
    # architecture and class ordering.
    service.load_checkpoint(
        "models/disaster_scene_unet.pt"
    )

    output = service.predict_path(
        "data/drone/flood_area_001.jpg"
    )

    print(
        json.dumps(
            output.result.to_dict(),
            indent=2,
        )
    )

    output_directory = Path(
        "outputs/segmentation"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    SegmentationReportExporter.export_json(
        output.result,
        output_directory / "report.json",
    )

    SegmentationReportExporter.export_images(
        output,
        output_directory,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SegmentationError",
    "SegmentationValidationError",
    "SegmentationModelNotReadyError",
    "SegmentationCheckpointError",
    "SegmentationExportError",
    "SegmentationTask",
    "SegmentationStatus",
    "NormalizationMode",
    "SegmentationConfig",
    "SegmentationClassResult",
    "SegmentationResult",
    "SegmentationOutput",
    "SegmentationBatchResult",
    "SegmentationImageValidator",
    "SegmentationPreprocessor",
    "DoubleConv",
    "SegmentationUNet",
    "SegmentationColorPalette",
    "SegmentationRenderer",
    "SegmentationMetricsAnalyzer",
    "SegmentationInferenceEngine",
    "SegmentationCheckpointManager",
    "SegmentationService",
    "SegmentationReportExporter",
    "create_segmentation_service",
    "create_flood_segmentation_service",
    "create_disaster_scene_segmentation_service",
]
