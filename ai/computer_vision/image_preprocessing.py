"""Placeholder module: ai/computer_vision/image_preprocessing.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
File: backend/ai/computer_vision/image_preprocessing.py

Production-oriented image preprocessing utilities for
flood-water segmentation, building damage classification,
satellite images, and drone image analysis.

Dependencies:
    pip install numpy Pillow torch

Features:
    - Safe JPEG / PNG / WebP image loading
    - EXIF orientation correction
    - RGB conversion
    - Image validation
    - Resize and letterbox preprocessing
    - ImageNet normalization
    - U-Net tensor preprocessing
    - CNN tensor preprocessing
    - Segmentation mask preprocessing
    - Paired image and mask augmentation
    - Contrast enhancement
    - Gaussian denoising
    - Blur and exposure quality analysis
    - Batch tensor preparation
    - Prediction mask restoration
    - Segmentation overlay generation
    - SHA-256 image fingerprinting

Security:
    - Reject oversized files
    - Reject excessive pixel counts
    - Reject unsupported formats
    - Validate tensor dimensions
    - Reject non-finite probability maps
    - Never execute image metadata

Important:
    - No automated emergency actions.
    - Geospatial rasters require a separate
      georeferenced processing pipeline.
    - Model-specific normalization must match
      the normalization used during training.
"""

from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    UnidentifiedImageError,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ImagePreprocessingError(Exception):
    pass


class ImageValidationError(ImagePreprocessingError):
    pass


class ImageDimensionError(ImagePreprocessingError):
    pass


class ImageMaskAlignmentError(ImagePreprocessingError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ResizeMode(str, Enum):
    STRETCH = "stretch"
    LETTERBOX = "letterbox"


class NormalizationMode(str, Enum):
    ZERO_ONE = "zero_one"
    IMAGENET = "imagenet"
    MINUS_ONE_ONE = "minus_one_one"


class ImageQualityStatus(str, Enum):
    GOOD = "good"
    REVIEW_REQUIRED = "review_required"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ImagePreprocessingConfig:
    image_width: int = 256
    image_height: int = 256

    resize_mode: ResizeMode = ResizeMode.STRETCH
    normalization: NormalizationMode = (
        NormalizationMode.ZERO_ONE
    )

    max_file_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    min_image_width: int = 32
    min_image_height: int = 32

    allowed_formats: tuple[str, ...] = (
        "JPEG",
        "PNG",
        "WEBP",
    )

    mask_threshold: float = 0.5

    letterbox_fill: tuple[int, int, int] = (
        0,
        0,
        0,
    )

    blur_threshold: float = 35.0

    minimum_brightness: float = 25.0
    maximum_brightness: float = 235.0

    random_seed: int = 42

    def __post_init__(self) -> None:
        integer_fields = (
            "image_width",
            "image_height",
            "max_file_bytes",
            "max_image_pixels",
            "min_image_width",
            "min_image_height",
        )

        for name in integer_fields:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ImageValidationError(
                    f"{name} must be a positive integer"
                )

        if not 0 <= self.mask_threshold <= 1:
            raise ImageValidationError(
                "mask_threshold must be in [0,1]"
            )

        if self.blur_threshold < 0:
            raise ImageValidationError(
                "blur_threshold cannot be negative"
            )

        if not (
            0 <= self.minimum_brightness
            < self.maximum_brightness
            <= 255
        ):
            raise ImageValidationError(
                "Invalid brightness thresholds"
            )

        if (
            len(self.letterbox_fill) != 3
            or any(
                not 0 <= value <= 255
                for value in self.letterbox_fill
            )
        ):
            raise ImageValidationError(
                "Invalid letterbox fill"
            )


# ============================================================
# METADATA
# ============================================================

@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    format: str | None
    mode: str
    sha256: str


@dataclass(frozen=True)
class ResizeMetadata:
    original_width: int
    original_height: int

    resized_width: int
    resized_height: int

    target_width: int
    target_height: int

    offset_x: int
    offset_y: int

    resize_mode: ResizeMode


@dataclass(frozen=True)
class ImageQualityReport:
    status: ImageQualityStatus

    mean_brightness: float
    blur_variance: float

    blurry: bool
    underexposed: bool
    overexposed: bool

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ProcessedImage:
    tensor: torch.Tensor
    metadata: ResizeMetadata


@dataclass(frozen=True)
class ProcessedImageMaskPair:
    image_tensor: torch.Tensor
    mask_tensor: torch.Tensor
    metadata: ResizeMetadata


# ============================================================
# IMAGE VALIDATOR
# ============================================================

class ImageValidator:
    def __init__(
        self,
        config: ImagePreprocessingConfig,
    ) -> None:
        self.config = config

    def validate_dimensions(
        self,
        width: int,
        height: int,
    ) -> None:
        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise ImageDimensionError(
                "Image resolution is too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise ImageDimensionError(
                "Image exceeds maximum pixel count"
            )

    def load_bytes(
        self,
        data: bytes,
    ) -> Image.Image:
        if not data:
            raise ImageValidationError(
                "Image data is empty"
            )

        if len(data) > self.config.max_file_bytes:
            raise ImageValidationError(
                "Image exceeds maximum file size"
            )

        try:
            with Image.open(
                io.BytesIO(data)
            ) as source:
                if source.format not in (
                    self.config.allowed_formats
                ):
                    raise ImageValidationError(
                        "Unsupported image format"
                    )

                self.validate_dimensions(
                    source.width,
                    source.height,
                )

                image = ImageOps.exif_transpose(
                    source
                ).convert("RGB")

                self.validate_dimensions(
                    image.width,
                    image.height,
                )

                return image.copy()

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as exc:
            raise ImageValidationError(
                "Invalid or unsafe image"
            ) from exc

    def load_path(
        self,
        path: str | Path,
    ) -> Image.Image:
        source = Path(path)

        if not source.is_file():
            raise ImageValidationError(
                "Image file not found"
            )

        if (
            source.stat().st_size
            > self.config.max_file_bytes
        ):
            raise ImageValidationError(
                "Image exceeds maximum file size"
            )

        return self.load_bytes(
            source.read_bytes()
        )

    def validate_pil(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise ImageValidationError(
                "Expected PIL Image"
            )

        self.validate_dimensions(
            image.width,
            image.height,
        )

        return ImageOps.exif_transpose(
            image
        ).convert("RGB")


# ============================================================
# IMAGE METADATA
# ============================================================

class ImageMetadataExtractor:
    @staticmethod
    def extract(
        image: Image.Image,
        original_bytes: bytes,
    ) -> ImageMetadata:
        return ImageMetadata(
            width=image.width,
            height=image.height,
            format=image.format,
            mode=image.mode,
            sha256=hashlib.sha256(
                original_bytes
            ).hexdigest(),
        )


# ============================================================
# IMAGE RESIZER
# ============================================================

class ImageResizer:
    def __init__(
        self,
        config: ImagePreprocessingConfig,
    ) -> None:
        self.config = config

    def resize(
        self,
        image: Image.Image,
        *,
        is_mask: bool = False,
    ) -> tuple[
        Image.Image,
        ResizeMetadata,
    ]:
        original_width, original_height = (
            image.size
        )

        target_width = self.config.image_width
        target_height = self.config.image_height

        interpolation = (
            Image.Resampling.NEAREST
            if is_mask
            else Image.Resampling.BILINEAR
        )

        if (
            self.config.resize_mode
            == ResizeMode.STRETCH
        ):
            resized = image.resize(
                (
                    target_width,
                    target_height,
                ),
                interpolation,
            )

            metadata = ResizeMetadata(
                original_width=original_width,
                original_height=original_height,
                resized_width=target_width,
                resized_height=target_height,
                target_width=target_width,
                target_height=target_height,
                offset_x=0,
                offset_y=0,
                resize_mode=ResizeMode.STRETCH,
            )

            return resized, metadata

        scale = min(
            target_width / original_width,
            target_height / original_height,
        )

        resized_width = max(
            1,
            min(
                target_width,
                int(round(original_width * scale)),
            ),
        )

        resized_height = max(
            1,
            min(
                target_height,
                int(round(original_height * scale)),
            ),
        )

        resized = image.resize(
            (
                resized_width,
                resized_height,
            ),
            interpolation,
        )

        offset_x = (
            target_width - resized_width
        ) // 2

        offset_y = (
            target_height - resized_height
        ) // 2

        fill = (
            0
            if is_mask
            else self.config.letterbox_fill
        )

        canvas = Image.new(
            "L" if is_mask else "RGB",
            (
                target_width,
                target_height,
            ),
            color=fill,
        )

        canvas.paste(
            resized,
            (
                offset_x,
                offset_y,
            ),
        )

        metadata = ResizeMetadata(
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
            target_width=target_width,
            target_height=target_height,
            offset_x=offset_x,
            offset_y=offset_y,
            resize_mode=ResizeMode.LETTERBOX,
        )

        return canvas, metadata


# ============================================================
# NORMALIZATION
# ============================================================

class ImageNormalizer:
    IMAGENET_MEAN = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    IMAGENET_STD = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    @classmethod
    def normalize(
        cls,
        image_array: np.ndarray,
        mode: NormalizationMode,
    ) -> np.ndarray:
        array = image_array.astype(
            np.float32
        ) / 255.0

        if mode == NormalizationMode.ZERO_ONE:
            return array

        if mode == NormalizationMode.MINUS_ONE_ONE:
            return array * 2.0 - 1.0

        if mode == NormalizationMode.IMAGENET:
            return (
                array - cls.IMAGENET_MEAN
            ) / cls.IMAGENET_STD

        raise ImageValidationError(
            "Unsupported normalization mode"
        )


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

class ImageEnhancementProcessor:
    @staticmethod
    def enhance_contrast(
        image: Image.Image,
        factor: float = 1.2,
    ) -> Image.Image:
        if (
            not math.isfinite(factor)
            or factor <= 0
        ):
            raise ImageValidationError(
                "Contrast factor must be positive"
            )

        return ImageEnhance.Contrast(
            image
        ).enhance(factor)

    @staticmethod
    def enhance_brightness(
        image: Image.Image,
        factor: float = 1.0,
    ) -> Image.Image:
        if (
            not math.isfinite(factor)
            or factor <= 0
        ):
            raise ImageValidationError(
                "Brightness factor must be positive"
            )

        return ImageEnhance.Brightness(
            image
        ).enhance(factor)

    @staticmethod
    def enhance_sharpness(
        image: Image.Image,
        factor: float = 1.1,
    ) -> Image.Image:
        if (
            not math.isfinite(factor)
            or factor < 0
        ):
            raise ImageValidationError(
                "Sharpness factor must be nonnegative"
            )

        return ImageEnhance.Sharpness(
            image
        ).enhance(factor)

    @staticmethod
    def gaussian_denoise(
        image: Image.Image,
        radius: float = 0.5,
    ) -> Image.Image:
        if (
            not math.isfinite(radius)
            or radius < 0
        ):
            raise ImageValidationError(
                "Gaussian radius must be nonnegative"
            )

        return image.filter(
            ImageFilter.GaussianBlur(
                radius=radius
            )
        )


# ============================================================
# IMAGE QUALITY ANALYSIS
# ============================================================

class ImageQualityAnalyzer:
    def __init__(
        self,
        config: ImagePreprocessingConfig,
    ) -> None:
        self.config = config

    def analyze(
        self,
        image: Image.Image,
    ) -> ImageQualityReport:
        grayscale = np.asarray(
            image.convert("L").resize(
                (256, 256),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

        brightness = float(
            grayscale.mean()
        )

        center = grayscale[
            1:-1,
            1:-1,
        ]

        laplacian = (
            -4.0 * center
            + grayscale[:-2, 1:-1]
            + grayscale[2:, 1:-1]
            + grayscale[1:-1, :-2]
            + grayscale[1:-1, 2:]
        )

        blur_variance = float(
            laplacian.var()
        )

        blurry = (
            blur_variance
            < self.config.blur_threshold
        )

        underexposed = (
            brightness
            < self.config.minimum_brightness
        )

        overexposed = (
            brightness
            > self.config.maximum_brightness
        )

        warnings = []

        if blurry:
            warnings.append(
                "Image may be blurry"
            )

        if underexposed:
            warnings.append(
                "Image may be underexposed"
            )

        if overexposed:
            warnings.append(
                "Image may be overexposed"
            )

        status = (
            ImageQualityStatus.REVIEW_REQUIRED
            if warnings
            else ImageQualityStatus.GOOD
        )

        return ImageQualityReport(
            status=status,
            mean_brightness=brightness,
            blur_variance=blur_variance,
            blurry=blurry,
            underexposed=underexposed,
            overexposed=overexposed,
            warnings=tuple(warnings),
        )


# ============================================================
# IMAGE TENSOR CONVERTER
# ============================================================

class ImageTensorConverter:
    @staticmethod
    def image_to_tensor(
        image: Image.Image,
        normalization: NormalizationMode,
    ) -> torch.Tensor:
        array = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

        normalized = ImageNormalizer.normalize(
            array,
            normalization,
        )

        tensor = torch.from_numpy(
            normalized.transpose(
                2,
                0,
                1,
            ).copy()
        )

        return tensor.float()

    @staticmethod
    def mask_to_tensor(
        mask: Image.Image,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        array = np.asarray(
            mask.convert("L"),
            dtype=np.float32,
        ) / 255.0

        binary = (
            array >= threshold
        ).astype(np.float32)

        return torch.from_numpy(
            binary[None, :, :].copy()
        )


# ============================================================
# PAIRED AUGMENTATION
# ============================================================

class PairedImageMaskAugmentor:
    """
    Applies identical spatial transforms
    to an image and its segmentation mask.

    Use only on training samples.
    """

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        self.random = random.Random(
            seed
        )

    def augment(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotate_90: bool = True,
    ) -> tuple[
        Image.Image,
        Image.Image,
    ]:
        if image.size != mask.size:
            raise ImageMaskAlignmentError(
                "Image and mask sizes differ"
            )

        if (
            horizontal_flip
            and self.random.random() < 0.5
        ):
            image = ImageOps.mirror(
                image
            )

            mask = ImageOps.mirror(
                mask
            )

        if (
            vertical_flip
            and self.random.random() < 0.5
        ):
            image = ImageOps.flip(
                image
            )

            mask = ImageOps.flip(
                mask
            )

        if rotate_90:
            angle = self.random.choice(
                (0, 90, 180, 270)
            )

            if angle:
                image = image.rotate(
                    angle,
                    resample=Image.Resampling.BILINEAR,
                    expand=False,
                )

                mask = mask.rotate(
                    angle,
                    resample=Image.Resampling.NEAREST,
                    expand=False,
                )

        return image, mask


# ============================================================
# PREDICTION RESTORATION
# ============================================================

class SegmentationPredictionRestorer:
    """
    Restores a model probability map
    to original image dimensions.

    For letterboxed inputs, removes
    padding before resizing.

    Expects probability maps with
    shape H x W and values in [0,1].
    """

    @staticmethod
    def restore_probabilities(
        probability_map: np.ndarray,
        metadata: ResizeMetadata,
    ) -> np.ndarray:
        probabilities = np.asarray(
            probability_map,
            dtype=np.float32,
        )

        if probabilities.shape != (
            metadata.target_height,
            metadata.target_width,
        ):
            raise ImageDimensionError(
                "Probability map shape mismatch"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise ImageValidationError(
                "Non-finite probability map"
            )

        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            raise ImageValidationError(
                "Probabilities must be in [0,1]"
            )

        if (
            metadata.resize_mode
            == ResizeMode.LETTERBOX
        ):
            x1 = metadata.offset_x
            y1 = metadata.offset_y

            x2 = (
                x1 + metadata.resized_width
            )

            y2 = (
                y1 + metadata.resized_height
            )

            probabilities = probabilities[
                y1:y2,
                x1:x2,
            ]

        image = Image.fromarray(
            probabilities,
            mode="F",
        )

        restored = image.resize(
            (
                metadata.original_width,
                metadata.original_height,
            ),
            Image.Resampling.BILINEAR,
        )

        return np.clip(
            np.asarray(
                restored,
                dtype=np.float32,
            ).copy(),
            0.0,
            1.0,
        )

    @staticmethod
    def probabilities_to_mask(
        probabilities: np.ndarray,
        threshold: float = 0.5,
    ) -> Image.Image:
        if not 0 <= threshold <= 1:
            raise ImageValidationError(
                "Invalid segmentation threshold"
            )

        array = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if (
            array.ndim != 2
            or array.size == 0
            or not np.isfinite(array).all()
            or np.any(array < 0)
            or np.any(array > 1)
        ):
            raise ImageValidationError(
                "Invalid probability map"
            )

        mask = (
            array >= threshold
        ).astype(np.uint8) * 255

        return Image.fromarray(
            mask,
            mode="L",
        )


# ============================================================
# SEGMENTATION OVERLAY
# ============================================================

class SegmentationOverlayGenerator:
    @staticmethod
    def create(
        image: Image.Image,
        mask: Image.Image,
        *,
        color: tuple[int, int, int] = (
            0,
            130,
            255,
        ),
        opacity: float = 0.5,
    ) -> Image.Image:
        if image.size != mask.size:
            raise ImageMaskAlignmentError(
                "Image and mask must align"
            )

        if not 0 <= opacity <= 1:
            raise ImageValidationError(
                "opacity must be in [0,1]"
            )

        if (
            len(color) != 3
            or any(
                not 0 <= component <= 255
                for component in color
            )
        ):
            raise ImageValidationError(
                "Invalid overlay color"
            )

        image_array = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        ).copy()

        mask_array = np.asarray(
            mask.convert("L"),
            dtype=np.uint8,
        ) >= 128

        overlay_color = np.asarray(
            color,
            dtype=np.float32,
        )

        image_array[mask_array] = (
            (1.0 - opacity)
            * image_array[
                mask_array
            ].astype(np.float32)
            + opacity * overlay_color
        ).astype(np.uint8)

        return Image.fromarray(
            image_array,
            mode="RGB",
        )


# ============================================================
# MAIN PREPROCESSING SERVICE
# ============================================================

class ImagePreprocessingService:
    def __init__(
        self,
        config: ImagePreprocessingConfig | None = None,
    ) -> None:
        self.config = (
            config or ImagePreprocessingConfig()
        )

        self.validator = ImageValidator(
            self.config
        )

        self.resizer = ImageResizer(
            self.config
        )

        self.quality_analyzer = ImageQualityAnalyzer(
            self.config
        )

        self.augmentor = (
            PairedImageMaskAugmentor(
                seed=self.config.random_seed
            )
        )

        self.restorer = (
            SegmentationPredictionRestorer()
        )

        self.overlay_generator = (
            SegmentationOverlayGenerator()
        )

    def preprocess_image(
        self,
        image: Image.Image,
    ) -> ProcessedImage:
        validated = self.validator.validate_pil(
            image
        )

        resized, metadata = self.resizer.resize(
            validated
        )

        tensor = ImageTensorConverter.image_to_tensor(
            resized,
            self.config.normalization,
        )

        return ProcessedImage(
            tensor=tensor,
            metadata=metadata,
        )

    def preprocess_bytes(
        self,
        image_bytes: bytes,
    ) -> ProcessedImage:
        image = self.validator.load_bytes(
            image_bytes
        )

        return self.preprocess_image(
            image
        )

    def preprocess_path(
        self,
        path: str | Path,
    ) -> ProcessedImage:
        image = self.validator.load_path(
            path
        )

        return self.preprocess_image(
            image
        )

    def preprocess_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        augment: bool = False,
    ) -> ProcessedImageMaskPair:
        validated_image = (
            self.validator.validate_pil(
                image
            )
        )

        validated_mask = mask.convert(
            "L"
        )

        if validated_image.size != (
            validated_mask.size
        ):
            raise ImageMaskAlignmentError(
                "Image and mask must have "
                "identical dimensions"
            )

        if augment:
            (
                validated_image,
                validated_mask,
            ) = self.augmentor.augment(
                validated_image,
                validated_mask,
            )

        resized_image, metadata = (
            self.resizer.resize(
                validated_image,
                is_mask=False,
            )
        )

        resized_mask, mask_metadata = (
            self.resizer.resize(
                validated_mask,
                is_mask=True,
            )
        )

        if (
            metadata.resized_width
            != mask_metadata.resized_width
            or metadata.resized_height
            != mask_metadata.resized_height
            or metadata.offset_x
            != mask_metadata.offset_x
            or metadata.offset_y
            != mask_metadata.offset_y
        ):
            raise ImageMaskAlignmentError(
                "Image and mask transforms differ"
            )

        image_tensor = (
            ImageTensorConverter.image_to_tensor(
                resized_image,
                self.config.normalization,
            )
        )

        mask_tensor = (
            ImageTensorConverter.mask_to_tensor(
                resized_mask,
                self.config.mask_threshold,
            )
        )

        return ProcessedImageMaskPair(
            image_tensor=image_tensor,
            mask_tensor=mask_tensor,
            metadata=metadata,
        )

    def preprocess_batch(
        self,
        images: Sequence[Image.Image],
    ) -> tuple[
        torch.Tensor,
        tuple[ResizeMetadata, ...],
    ]:
        if not images:
            raise ImageValidationError(
                "Image batch is empty"
            )

        processed = [
            self.preprocess_image(image)
            for image in images
        ]

        batch = torch.stack(
            [
                item.tensor
                for item in processed
            ],
            dim=0,
        )

        metadata = tuple(
            item.metadata
            for item in processed
        )

        return batch, metadata

    def analyze_quality(
        self,
        image: Image.Image,
    ) -> ImageQualityReport:
        validated = self.validator.validate_pil(
            image
        )

        return self.quality_analyzer.analyze(
            validated
        )

    def restore_prediction(
        self,
        probability_map: np.ndarray,
        metadata: ResizeMetadata,
        *,
        threshold: float | None = None,
    ) -> Image.Image:
        restored = (
            self.restorer.restore_probabilities(
                probability_map,
                metadata,
            )
        )

        return self.restorer.probabilities_to_mask(
            restored,
            threshold=(
                self.config.mask_threshold
                if threshold is None
                else threshold
            ),
        )

    def create_overlay(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        opacity: float = 0.5,
    ) -> Image.Image:
        return self.overlay_generator.create(
            image,
            mask,
            opacity=opacity,
        )


# ============================================================
# MODEL-SPECIFIC FACTORIES
# ============================================================

def create_unet_preprocessor(
    *,
    image_size: int = 256,
) -> ImagePreprocessingService:
    return ImagePreprocessingService(
        ImagePreprocessingConfig(
            image_width=image_size,
            image_height=image_size,
            resize_mode=ResizeMode.STRETCH,
            normalization=NormalizationMode.ZERO_ONE,
        )
    )


def create_cnn_damage_preprocessor(
    *,
    image_size: int = 224,
) -> ImagePreprocessingService:
    return ImagePreprocessingService(
        ImagePreprocessingConfig(
            image_width=image_size,
            image_height=image_size,
            resize_mode=ResizeMode.STRETCH,
            normalization=NormalizationMode.IMAGENET,
        )
    )


def create_drone_preprocessor(
    *,
    image_size: int = 512,
) -> ImagePreprocessingService:
    return ImagePreprocessingService(
        ImagePreprocessingConfig(
            image_width=image_size,
            image_height=image_size,
            resize_mode=ResizeMode.LETTERBOX,
            normalization=NormalizationMode.ZERO_ONE,
        )
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    service = create_unet_preprocessor(
        image_size=256
    )

    image_path = Path(
        "data/drone/flood_area_001.jpg"
    )

    image = service.validator.load_path(
        image_path
    )

    quality = service.analyze_quality(
        image
    )

    processed = service.preprocess_image(
        image
    )

    print(
        "Image tensor:",
        tuple(processed.tensor.shape),
    )

    print(
        "Image quality:",
        quality.status.value,
    )

    print(
        "Blur variance:",
        quality.blur_variance,
    )

    print(
        "Mean brightness:",
        quality.mean_brightness,
    )

    print(
        "Resize metadata:",
        processed.metadata,
    )

    # Example only:
    # Replace with probabilities from a trained U-Net.
    example_probabilities = np.zeros(
        (
            service.config.image_height,
            service.config.image_width,
        ),
        dtype=np.float32,
    )

    mask = service.restore_prediction(
        example_probabilities,
        processed.metadata,
    )

    overlay = service.create_overlay(
        image,
        mask,
    )

    output_dir = Path(
        "outputs/preprocessing"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask.save(
        output_dir / "mask.png"
    )

    overlay.save(
        output_dir / "overlay.png"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ImagePreprocessingError",
    "ImageValidationError",
    "ImageDimensionError",
    "ImageMaskAlignmentError",
    "ResizeMode",
    "NormalizationMode",
    "ImageQualityStatus",
    "ImagePreprocessingConfig",
    "ImageMetadata",
    "ResizeMetadata",
    "ImageQualityReport",
    "ProcessedImage",
    "ProcessedImageMaskPair",
    "ImageValidator",
    "ImageMetadataExtractor",
    "ImageResizer",
    "ImageNormalizer",
    "ImageEnhancementProcessor",
    "ImageQualityAnalyzer",
    "ImageTensorConverter",
    "PairedImageMaskAugmentor",
    "SegmentationPredictionRestorer",
    "SegmentationOverlayGenerator",
    "ImagePreprocessingService",
    "create_unet_preprocessor",
    "create_cnn_damage_preprocessor",
    "create_drone_preprocessor",
]
