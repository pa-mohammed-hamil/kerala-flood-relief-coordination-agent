"""Placeholder module: ai/deep_learning/unet_flood_segmentation.py; implementation required."""
"""Placeholder module: ai/computer_vision/building_damage_detection.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
U-Net Flood Segmentation

File:
    backend/ai/deep_learning/unet_flood_segmentation.py

Python:
    3.11+

Dependencies:
    torch >= 2.2
    numpy
    Pillow

Purpose:
    Segment visible flood water in RGB
    satellite, drone, or aerial imagery.

Capabilities:
    - U-Net architecture
    - Binary water segmentation
    - Training and validation
    - BCE + Dice loss
    - IoU / Dice / precision / recall
    - Early stopping
    - Checkpoint persistence
    - Mask and overlay export
    - Uncertainty review flags

Limitations:
    - Pixel coverage is not geographic area.
    - Images must be appropriately georeferenced
      before estimating square kilometers.
    - Ordinary water bodies can be confused
      with floodwater.
    - Cloud, shadows, vegetation, and buildings
      can obscure flooding.
    - Outputs are advisory only.
"""

from __future__ import annotations

import copy
import io
import math
import random
import uuid

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# EXCEPTIONS
# ============================================================

class FloodSegmentationError(Exception):
    pass


class FloodSegmentationValidationError(
    FloodSegmentationError
):
    pass


class FloodSegmentationNotReadyError(
    FloodSegmentationError
):
    pass


class FloodSegmentationCheckpointError(
    FloodSegmentationError
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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if name not in ("cpu", "cuda"):
        raise FloodSegmentationValidationError(
            "device must be auto, cpu, or cuda"
        )

    if name == "cuda" and not torch.cuda.is_available():
        raise FloodSegmentationValidationError(
            "CUDA is unavailable"
        )

    return torch.device(name)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class UNetFloodConfig:
    image_size: int = 256

    input_channels: int = 3
    output_channels: int = 1

    base_channels: int = 32

    batch_size: int = 8
    epochs: int = 40

    learning_rate: float = 0.0001
    weight_decay: float = 0.0001

    dropout: float = 0.10

    segmentation_threshold: float = 0.50

    minimum_confidence: float = 0.70
    maximum_uncertain_pixel_fraction: float = 0.30

    early_stopping_patience: int = 7
    gradient_clip_norm: float = 1.0

    max_image_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 30_000_000

    device: str = "auto"
    random_seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "image_size",
            "input_channels",
            "output_channels",
            "base_channels",
            "batch_size",
            "epochs",
            "early_stopping_patience",
            "max_image_bytes",
            "max_image_pixels",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise FloodSegmentationValidationError(
                    f"{name} must be positive"
                )

        if self.image_size < 32 or self.image_size % 16:
            raise FloodSegmentationValidationError(
                "image_size must be >= 32 "
                "and divisible by 16"
            )

        if self.input_channels != 3:
            raise FloodSegmentationValidationError(
                "This implementation supports RGB only"
            )

        if self.output_channels != 1:
            raise FloodSegmentationValidationError(
                "Binary segmentation requires "
                "output_channels=1"
            )

        for name in (
            "segmentation_threshold",
            "minimum_confidence",
            "maximum_uncertain_pixel_fraction",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 1:
                raise FloodSegmentationValidationError(
                    f"{name} must be in [0,1]"
                )

        if not 0 <= self.dropout < 1:
            raise FloodSegmentationValidationError(
                "dropout must be in [0,1)"
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
                raise FloodSegmentationValidationError(
                    f"{name} must be positive and finite"
                )

        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise FloodSegmentationValidationError(
                "weight_decay must be nonnegative"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class SegmentationEpoch:
    epoch: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class SegmentationTrainingResult:
    training_id: str
    epochs_completed: int
    best_validation_loss: float
    history: tuple[SegmentationEpoch, ...]
    trained_at: datetime


@dataclass(frozen=True)
class SegmentationEvaluation:
    image_count: int
    pixel_accuracy: float
    precision: float
    recall: float
    dice_score: float
    intersection_over_union: float


@dataclass(frozen=True)
class FloodSegmentationResult:
    segmentation_id: str
    image_width: int
    image_height: int

    flood_pixel_count: int
    total_pixel_count: int

    flood_coverage_percent: float
    mean_pixel_confidence: float
    uncertain_pixel_fraction: float

    requires_human_review: bool
    warnings: tuple[str, ...]

    generated_at: datetime


# ============================================================
# CONVOLUTION BLOCK
# ============================================================

class DoubleConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                output_channels
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                output_channels
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


# ============================================================
# ENCODER BLOCK
# ============================================================

class DownBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                input_channels,
                output_channels,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


# ============================================================
# DECODER BLOCK
# ============================================================

class UpBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            output_channels + skip_channels,
            output_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        combined = torch.cat(
            [skip, x],
            dim=1,
        )

        return self.conv(combined)


# ============================================================
# U-NET MODEL
# ============================================================

class FloodUNet(nn.Module):
    """
    Input:
        B x 3 x H x W

    Output:
        B x 1 x H x W

    Output values are logits.
    """

    def __init__(
        self,
        config: UNetFloodConfig,
    ) -> None:
        super().__init__()

        base = config.base_channels

        self.encoder1 = DoubleConv(
            config.input_channels,
            base,
        )

        self.encoder2 = DownBlock(
            base,
            base * 2,
        )

        self.encoder3 = DownBlock(
            base * 2,
            base * 4,
        )

        self.encoder4 = DownBlock(
            base * 4,
            base * 8,
        )

        self.bottleneck = DownBlock(
            base * 8,
            base * 16,
        )

        self.dropout = nn.Dropout2d(
            config.dropout
        )

        self.decoder4 = UpBlock(
            base * 16,
            base * 8,
            base * 8,
        )

        self.decoder3 = UpBlock(
            base * 8,
            base * 4,
            base * 4,
        )

        self.decoder2 = UpBlock(
            base * 4,
            base * 2,
            base * 2,
        )

        self.decoder1 = UpBlock(
            base * 2,
            base,
            base,
        )

        self.output_layer = nn.Conv2d(
            base,
            config.output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        bottleneck = self.bottleneck(
            e4
        )

        bottleneck = self.dropout(
            bottleneck
        )

        d4 = self.decoder4(
            bottleneck,
            e4,
        )

        d3 = self.decoder3(
            d4,
            e3,
        )

        d2 = self.decoder2(
            d3,
            e2,
        )

        d1 = self.decoder1(
            d2,
            e1,
        )

        return self.output_layer(d1)


# ============================================================
# IMAGE VALIDATION
# ============================================================

class FloodImageValidator:
    def __init__(
        self,
        config: UNetFloodConfig,
    ) -> None:
        self.config = config

    def load_bytes(
        self,
        data: bytes,
    ) -> Image.Image:
        if not data:
            raise FloodSegmentationValidationError(
                "Empty image"
            )

        if len(data) > self.config.max_image_bytes:
            raise FloodSegmentationValidationError(
                "Image exceeds maximum byte size"
            )

        try:
            with Image.open(
                io.BytesIO(data)
            ) as source:
                if source.format not in (
                    "PNG",
                    "JPEG",
                    "WEBP",
                ):
                    raise FloodSegmentationValidationError(
                        "Unsupported image format"
                    )

                if (
                    source.width * source.height
                    > self.config.max_image_pixels
                ):
                    raise FloodSegmentationValidationError(
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
            raise FloodSegmentationValidationError(
                "Invalid or unsafe image"
            ) from exc

    def load_path(
        self,
        path: str | Path,
    ) -> Image.Image:
        source = Path(path)

        if not source.is_file():
            raise FloodSegmentationValidationError(
                "Image not found"
            )

        if source.stat().st_size > (
            self.config.max_image_bytes
        ):
            raise FloodSegmentationValidationError(
                "Image exceeds maximum byte size"
            )

        return self.load_bytes(
            source.read_bytes()
        )


# ============================================================
# DATASET
# ============================================================

class FloodSegmentationDataset(Dataset):
    """
    Dataset structure:

        images/
            image_001.png
            image_002.png

        masks/
            image_001.png
            image_002.png

    Mask convention:
        0   = non-water
        255 = flood water

    Training imagery must be trusted,
    and masks must be aligned with images.
    """

    def __init__(
        self,
        image_directory: str | Path,
        mask_directory: str | Path,
        config: UNetFloodConfig,
        *,
        augment: bool = False,
    ) -> None:
        self.config = config
        self.augment = augment

        self.image_directory = Path(
            image_directory
        )

        self.mask_directory = Path(
            mask_directory
        )

        extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }

        self.images = sorted(
            path
            for path in self.image_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in extensions
            )
        )

        if not self.images:
            raise FloodSegmentationValidationError(
                "No training images found"
            )

        for image_path in self.images:
            mask_path = (
                self.mask_directory
                / f"{image_path.stem}.png"
            )

            if not mask_path.is_file():
                raise FloodSegmentationValidationError(
                    f"Missing mask for {image_path.name}"
                )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        image_path = self.images[index]

        mask_path = (
            self.mask_directory
            / f"{image_path.stem}.png"
        )

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        with Image.open(mask_path) as source:
            mask = source.convert("L")

        if image.size != mask.size:
            raise FloodSegmentationValidationError(
                "Image and mask dimensions differ"
            )

        size = (
            self.config.image_size,
            self.config.image_size,
        )

        image = image.resize(
            size,
            Image.Resampling.BILINEAR,
        )

        mask = mask.resize(
            size,
            Image.Resampling.NEAREST,
        )

        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)

            if random.random() < 0.5:
                image = ImageOps.flip(image)
                mask = ImageOps.flip(mask)

            rotations = random.choice(
                (0, 90, 180, 270)
            )

            if rotations:
                image = image.rotate(
                    rotations
                )

                mask = mask.rotate(
                    rotations
                )

        image_array = (
            np.asarray(
                image,
                dtype=np.float32,
            )
            / 255.0
        )

        mask_array = (
            np.asarray(
                mask,
                dtype=np.float32,
            )
            / 255.0
        )

        mask_array = (
            mask_array >= 0.5
        ).astype(np.float32)

        image_tensor = torch.from_numpy(
            image_array.transpose(
                2,
                0,
                1,
            ).copy()
        )

        mask_tensor = torch.from_numpy(
            mask_array[None, ...].copy()
        )

        return (
            image_tensor,
            mask_tensor,
        )


# ============================================================
# BCE + DICE LOSS
# ============================================================

class FloodBCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        if (
            bce_weight < 0
            or dice_weight < 0
            or bce_weight + dice_weight <= 0
        ):
            raise FloodSegmentationValidationError(
                "Invalid loss weights"
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = self.bce(
            logits,
            targets,
        )

        probabilities = torch.sigmoid(
            logits
        )

        intersection = (
            probabilities * targets
        ).sum(dim=(1, 2, 3))

        denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            + targets.sum(
                dim=(1, 2, 3)
            )
        )

        dice = 1 - (
            (2 * intersection + 1)
            / (denominator + 1)
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice.mean()
        )


# ============================================================
# TRAINING ENGINE
# ============================================================

class FloodUNetTrainer:
    def __init__(
        self,
        config: UNetFloodConfig,
        model: FloodUNet,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.criterion = FloodBCEDiceLoss()

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        training = optimizer is not None

        self.model.train(training)

        total_loss = 0.0
        total_images = 0

        for images, masks in loader:
            images = images.to(
                self.device
            )

            masks = masks.to(
                self.device
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(training):
                logits = self.model(
                    images
                )

                loss = self.criterion(
                    logits,
                    masks,
                )

                if not torch.isfinite(loss):
                    raise FloodSegmentationError(
                        "Non-finite loss encountered"
                    )

                if training:
                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                    optimizer.step()

            batch_size = images.shape[0]

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_images += batch_size

        if total_images == 0:
            raise FloodSegmentationValidationError(
                "Empty DataLoader"
            )

        return total_loss / total_images

    def train(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> SegmentationTrainingResult:
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
                SegmentationEpoch(
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
            raise FloodSegmentationError(
                "No valid model state produced"
            )

        self.model.load_state_dict(
            best_state
        )

        self.model.eval()

        return SegmentationTrainingResult(
            training_id=new_id(),
            epochs_completed=len(history),
            best_validation_loss=best_loss,
            history=tuple(history),
            trained_at=utc_now(),
        )


# ============================================================
# SEGMENTATION EVALUATOR
# ============================================================

class FloodUNetEvaluator:
    def __init__(
        self,
        config: UNetFloodConfig,
        model: FloodUNet,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

    @torch.inference_mode()
    def evaluate(
        self,
        loader: DataLoader,
    ) -> SegmentationEvaluation:
        self.model.eval()

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        image_count = 0

        for images, masks in loader:
            images = images.to(
                self.device
            )

            masks = masks.to(
                self.device
            )

            logits = self.model(
                images
            )

            probabilities = torch.sigmoid(
                logits
            )

            predicted = (
                probabilities
                >= self.config.segmentation_threshold
            )

            actual = masks >= 0.5

            tp += int(
                (predicted & actual).sum().item()
            )

            fp += int(
                (predicted & ~actual).sum().item()
            )

            fn += int(
                (~predicted & actual).sum().item()
            )

            tn += int(
                (~predicted & ~actual).sum().item()
            )

            image_count += images.shape[0]

        if image_count == 0:
            raise FloodSegmentationValidationError(
                "Empty evaluation dataset"
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

        dice = (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn
            else 1.0
        )

        iou = (
            tp / (tp + fp + fn)
            if tp + fp + fn
            else 1.0
        )

        accuracy = (
            (tp + tn)
            / max(tp + tn + fp + fn, 1)
        )

        return SegmentationEvaluation(
            image_count=image_count,
            pixel_accuracy=accuracy,
            precision=precision,
            recall=recall,
            dice_score=dice,
            intersection_over_union=iou,
        )


# ============================================================
# INFERENCE ENGINE
# ============================================================

class FloodUNetInferenceEngine:
    def __init__(
        self,
        config: UNetFloodConfig,
        model: FloodUNet,
        device: torch.device,
        *,
        trained: bool = False,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device
        self.trained = trained

    @torch.inference_mode()
    def predict_image(
        self,
        image: Image.Image,
    ) -> tuple[
        FloodSegmentationResult,
        Image.Image,
        Image.Image,
    ]:
        if not self.trained:
            raise FloodSegmentationNotReadyError(
                "Model must be trained or loaded"
            )

        image = image.convert("RGB")

        original_width, original_height = (
            image.size
        )

        if (
            original_width * original_height
            > self.config.max_image_pixels
        ):
            raise FloodSegmentationValidationError(
                "Image exceeds pixel limit"
            )

        resized = image.resize(
            (
                self.config.image_size,
                self.config.image_size,
            ),
            Image.Resampling.BILINEAR,
        )

        array = (
            np.asarray(
                resized,
                dtype=np.float32,
            )
            / 255.0
        )

        tensor = torch.from_numpy(
            array.transpose(
                2,
                0,
                1,
            ).copy()
        ).unsqueeze(0).to(
            self.device
        )

        self.model.eval()

        logits = self.model(
            tensor
        )

        probabilities = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

        probability_image = Image.fromarray(
            probabilities.astype(
                np.float32
            ),
            mode="F",
        )

        probability_image = probability_image.resize(
            (
                original_width,
                original_height,
            ),
            Image.Resampling.BILINEAR,
        )

        full_probabilities = np.asarray(
            probability_image,
            dtype=np.float32,
        )

        binary_mask = (
            full_probabilities
            >= self.config.segmentation_threshold
        )

        mask_array = (
            binary_mask.astype(np.uint8)
            * 255
        )

        mask_image = Image.fromarray(
            mask_array,
            mode="L",
        )

        # Confidence in the predicted binary class.
        confidence = np.maximum(
            full_probabilities,
            1.0 - full_probabilities,
        )

        mean_confidence = float(
            confidence.mean()
        )

        uncertain_fraction = float(
            np.mean(
                confidence
                < self.config.minimum_confidence
            )
        )

        flood_pixels = int(
            binary_mask.sum()
        )

        total_pixels = (
            original_width
            * original_height
        )

        coverage = (
            100.0
            * flood_pixels
            / max(total_pixels, 1)
        )

        overlay_array = np.asarray(
            image,
            dtype=np.uint8,
        ).copy()

        overlay_array[
            binary_mask
        ] = (
            0.5 * overlay_array[binary_mask]
            + 0.5 * np.array(
                [0, 130, 255],
                dtype=np.float32,
            )
        ).astype(np.uint8)

        overlay_image = Image.fromarray(
            overlay_array,
            mode="RGB",
        )

        warnings = [
            "Advisory-only image segmentation",
            "Visible water may include permanent water bodies",
            "Pixel coverage is not geographic flood area",
        ]

        requires_review = (
            mean_confidence
            < self.config.minimum_confidence
            or uncertain_fraction
            > self.config.maximum_uncertain_pixel_fraction
        )

        if requires_review:
            warnings.append(
                "Low-confidence segmentation requires review"
            )

        result = FloodSegmentationResult(
            segmentation_id=new_id(),
            image_width=original_width,
            image_height=original_height,
            flood_pixel_count=flood_pixels,
            total_pixel_count=total_pixels,
            flood_coverage_percent=coverage,
            mean_pixel_confidence=mean_confidence,
            uncertain_pixel_fraction=uncertain_fraction,
            requires_human_review=requires_review,
            warnings=tuple(warnings),
            generated_at=utc_now(),
        )

        return (
            result,
            mask_image,
            overlay_image,
        )


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class FloodUNetCheckpointManager:
    @staticmethod
    def save(
        path: str | Path,
        *,
        model: FloodUNet,
        config: UNetFloodConfig,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise FloodSegmentationCheckpointError(
                "Checkpoint already exists"
            )

        with destination.open("xb") as stream:
            torch.save(
                {
                    "format_version": 1,
                    "config": asdict(config),
                    "model_state": model.state_dict(),
                    "saved_at": utc_now().isoformat(),
                },
                stream,
            )

        return destination

    @staticmethod
    def load(
        path: str | Path,
        *,
        model: FloodUNet,
        config: UNetFloodConfig,
        device: torch.device,
    ) -> None:
        checkpoint = torch.load(
            Path(path),
            map_location=device,
            weights_only=True,
        )

        if checkpoint.get("format_version") != 1:
            raise FloodSegmentationCheckpointError(
                "Unsupported checkpoint format"
            )

        saved_config = checkpoint.get(
            "config",
            {},
        )

        for name in (
            "image_size",
            "input_channels",
            "output_channels",
            "base_channels",
        ):
            if saved_config.get(name) != getattr(
                config,
                name,
            ):
                raise FloodSegmentationCheckpointError(
                    f"Checkpoint mismatch: {name}"
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

class FloodUNetSegmentationService:
    def __init__(
        self,
        config: UNetFloodConfig | None = None,
    ) -> None:
        self.config = (
            config or UNetFloodConfig()
        )

        set_seed(
            self.config.random_seed
        )

        self.device = resolve_device(
            self.config.device
        )

        self.validator = FloodImageValidator(
            self.config
        )

        self.model = FloodUNet(
            self.config
        ).to(self.device)

        self.trainer = FloodUNetTrainer(
            self.config,
            self.model,
            self.device,
        )

        self.evaluator = FloodUNetEvaluator(
            self.config,
            self.model,
            self.device,
        )

        self.trained = False

    def train(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> SegmentationTrainingResult:
        result = self.trainer.train(
            training_loader,
            validation_loader,
        )

        self.trained = True

        return result

    def evaluate(
        self,
        loader: DataLoader,
    ) -> SegmentationEvaluation:
        if not self.trained:
            raise FloodSegmentationNotReadyError(
                "Model is not trained"
            )

        return self.evaluator.evaluate(
            loader
        )

    def predict_image(
        self,
        image: Image.Image,
    ) -> tuple[
        FloodSegmentationResult,
        Image.Image,
        Image.Image,
    ]:
        engine = FloodUNetInferenceEngine(
            self.config,
            self.model,
            self.device,
            trained=self.trained,
        )

        return engine.predict_image(
            image
        )

    def predict_path(
        self,
        path: str | Path,
    ) -> tuple[
        FloodSegmentationResult,
        Image.Image,
        Image.Image,
    ]:
        image = self.validator.load_path(
            path
        )

        return self.predict_image(
            image
        )

    def predict_bytes(
        self,
        data: bytes,
    ) -> tuple[
        FloodSegmentationResult,
        Image.Image,
        Image.Image,
    ]:
        image = self.validator.load_bytes(
            data
        )

        return self.predict_image(
            image
        )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> Path:
        if not self.trained:
            raise FloodSegmentationNotReadyError(
                "Cannot save untrained model"
            )

        return FloodUNetCheckpointManager.save(
            path,
            model=self.model,
            config=self.config,
        )

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        FloodUNetCheckpointManager.load(
            path,
            model=self.model,
            config=self.config,
            device=self.device,
        )

        self.trained = True


# ============================================================
# FACTORY
# ============================================================

def create_flood_unet_service(
    config: UNetFloodConfig | None = None,
) -> FloodUNetSegmentationService:
    return FloodUNetSegmentationService(
        config=config
    )


__all__ = [
    "FloodSegmentationError",
    "FloodSegmentationValidationError",
    "FloodSegmentationNotReadyError",
    "FloodSegmentationCheckpointError",
    "UNetFloodConfig",
    "SegmentationEpoch",
    "SegmentationTrainingResult",
    "SegmentationEvaluation",
    "FloodSegmentationResult",
    "DoubleConv",
    "DownBlock",
    "UpBlock",
    "FloodUNet",
    "FloodImageValidator",
    "FloodSegmentationDataset",
    "FloodBCEDiceLoss",
    "FloodUNetTrainer",
    "FloodUNetEvaluator",
    "FloodUNetInferenceEngine",
    "FloodUNetCheckpointManager",
    "FloodUNetSegmentationService",
    "create_flood_unet_service",
]
