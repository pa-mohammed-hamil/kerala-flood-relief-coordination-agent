"""Placeholder module: ai/computer_vision/object_detection.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
File: backend/ai/computer_vision/object_detection.py

Object detection for disaster-response imagery.

Dependencies:
    pip install torch torchvision Pillow numpy

Capabilities:
    - Faster R-CNN object detection
    - COCO-pretrained inference
    - Person, vehicle and boat screening
    - Bounding boxes and confidence scores
    - Confidence threshold filtering
    - Class-aware non-maximum suppression
    - Image validation
    - Single-image and batch inference
    - Annotated image generation
    - JSON export
    - Custom trained checkpoint loading
    - Human-review safeguards

Important:
    Standard COCO models cannot reliably detect
    flooding, trapped people, structural damage,
    emergency conditions or every boat type.

    A detected person is not evidence of distress.

    Outputs are advisory and must never trigger
    autonomous rescue dispatch or evacuation.
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

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps,
)

from torchvision import models
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class ObjectDetectionError(Exception):
    pass


class ObjectDetectionValidationError(
    ObjectDetectionError
):
    pass


class ObjectDetectionNotReadyError(
    ObjectDetectionError
):
    pass


class ObjectDetectionCheckpointError(
    ObjectDetectionError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class ObjectCategory(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    BOAT = "boat"
    ANIMAL = "animal"
    OTHER = "other"


class DetectionStatus(str, Enum):
    COMPLETED = "completed"
    REVIEW_REQUIRED = "review_required"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ObjectDetectionConfig:
    confidence_threshold: float = 0.50
    nms_iou_threshold: float = 0.45

    maximum_detections: int = 100
    maximum_batch_size: int = 8

    max_image_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 30_000_000

    minimum_image_width: int = 64
    minimum_image_height: int = 64

    max_inference_dimension: int = 2048

    device: str = "auto"

    pretrained: bool = True

    require_human_review: bool = True

    def __post_init__(self) -> None:
        for name in (
            "confidence_threshold",
            "nms_iou_threshold",
        ):
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ObjectDetectionValidationError(
                    f"{name} must be in [0,1]"
                )

        for name in (
            "maximum_detections",
            "maximum_batch_size",
            "max_image_bytes",
            "max_image_pixels",
            "minimum_image_width",
            "minimum_image_height",
            "max_inference_dimension",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ObjectDetectionValidationError(
                    f"{name} must be positive"
                )


# ============================================================
# COCO CLASS NAMES
# ============================================================

COCO_CLASSES = (
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(
            0.0,
            self.x_max - self.x_min,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.y_max - self.y_min,
        )

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class DetectedObject:
    detection_id: str

    class_id: int
    class_name: str

    category: ObjectCategory
    confidence: float

    bounding_box: BoundingBox

    requires_human_review: bool


@dataclass(frozen=True)
class ObjectDetectionResult:
    analysis_id: str

    image_id: str
    filename: str | None

    image_width: int
    image_height: int

    total_detections: int

    person_count: int
    vehicle_count: int
    boat_count: int
    animal_count: int

    objects: tuple[DetectedObject, ...]

    status: DetectionStatus
    requires_human_review: bool

    image_sha256: str

    warnings: tuple[str, ...]
    analyzed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["status"] = self.status.value

        result["analyzed_at"] = (
            self.analyzed_at.isoformat()
        )

        for item in result["objects"]:
            item["category"] = (
                item["category"].value
                if isinstance(
                    item["category"],
                    ObjectCategory,
                )
                else item["category"]
            )

        return result


@dataclass(frozen=True)
class ObjectDetectionOutput:
    result: ObjectDetectionResult
    annotated_image: Image.Image


@dataclass(frozen=True)
class ObjectDetectionBatchResult:
    batch_id: str
    total_images: int
    review_required_count: int

    results: tuple[
        ObjectDetectionResult,
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
        raise ObjectDetectionValidationError(
            "device must be auto, cpu, or cuda"
        )

    if (
        value == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ObjectDetectionValidationError(
            "CUDA unavailable"
        )

    return torch.device(value)


def classify_object(
    class_name: str,
) -> ObjectCategory:
    if class_name == "person":
        return ObjectCategory.PERSON

    if class_name in {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "train",
    }:
        return ObjectCategory.VEHICLE

    if class_name == "boat":
        return ObjectCategory.BOAT

    if class_name in {
        "dog",
        "cat",
        "cow",
        "horse",
        "sheep",
        "bird",
        "elephant",
    }:
        return ObjectCategory.ANIMAL

    return ObjectCategory.OTHER


# ============================================================
# IMAGE VALIDATION
# ============================================================

class DetectionImageValidator:
    def __init__(
        self,
        config: ObjectDetectionConfig,
    ) -> None:
        self.config = config

    def load_bytes(
        self,
        data: bytes,
    ) -> Image.Image:
        if not data:
            raise ObjectDetectionValidationError(
                "Empty image"
            )

        if (
            len(data)
            > self.config.max_image_bytes
        ):
            raise ObjectDetectionValidationError(
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
                    raise ObjectDetectionValidationError(
                        "Unsupported image format"
                    )

                width, height = source.size

                if (
                    width < self.config.minimum_image_width
                    or height < self.config.minimum_image_height
                ):
                    raise ObjectDetectionValidationError(
                        "Image resolution too small"
                    )

                if (
                    width * height
                    > self.config.max_image_pixels
                ):
                    raise ObjectDetectionValidationError(
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
            raise ObjectDetectionValidationError(
                "Invalid or unsafe image"
            ) from exc

    def load_path(
        self,
        path: str | Path,
    ) -> tuple[Image.Image, bytes]:
        source = Path(path)

        if not source.is_file():
            raise ObjectDetectionValidationError(
                "Image not found"
            )

        if (
            source.stat().st_size
            > self.config.max_image_bytes
        ):
            raise ObjectDetectionValidationError(
                "Image exceeds byte limit"
            )

        data = source.read_bytes()

        return self.load_bytes(data), data


# ============================================================
# MODEL FACTORY
# ============================================================

class ObjectDetectionModelFactory:
    @staticmethod
    def create(
        *,
        pretrained: bool = True,
    ) -> torch.nn.Module:
        weights = (
            models.detection
            .FasterRCNN_MobileNet_V3_Large_FPN_Weights
            .DEFAULT
            if pretrained
            else None
        )

        model = (
            models.detection
            .fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights,
                weights_backbone=None,
            )
        )

        model.eval()

        return model


# ============================================================
# IMAGE PREPROCESSOR
# ============================================================

class DetectionImagePreprocessor:
    def __init__(
        self,
        config: ObjectDetectionConfig,
    ) -> None:
        self.config = config

    def preprocess(
        self,
        image: Image.Image,
    ) -> tuple[
        torch.Tensor,
        float,
        float,
    ]:
        width, height = image.size

        longest_side = max(
            width,
            height,
        )

        scale = min(
            1.0,
            self.config.max_inference_dimension
            / longest_side,
        )

        resized_width = max(
            1,
            int(round(width * scale)),
        )

        resized_height = max(
            1,
            int(round(height * scale)),
        )

        resized = image.resize(
            (
                resized_width,
                resized_height,
            ),
            Image.Resampling.BILINEAR,
        )

        tensor = to_tensor(
            resized
        )

        scale_x = (
            width / resized_width
        )

        scale_y = (
            height / resized_height
        )

        return (
            tensor,
            scale_x,
            scale_y,
        )


# ============================================================
# PREDICTION FILTER
# ============================================================

class DetectionPredictionFilter:
    def __init__(
        self,
        config: ObjectDetectionConfig,
    ) -> None:
        self.config = config

    def filter(
        self,
        prediction: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]

        keep = (
            scores
            >= self.config.confidence_threshold
        )

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.numel() == 0:
            return boxes, scores, labels

        selected = []

        for label in labels.unique():
            indices = torch.where(
                labels == label
            )[0]

            retained = nms(
                boxes[indices],
                scores[indices],
                self.config.nms_iou_threshold,
            )

            selected.append(
                indices[retained]
            )

        selected = torch.cat(
            selected
        )

        selected = selected[
            torch.argsort(
                scores[selected],
                descending=True,
            )
        ]

        selected = selected[
            :self.config.maximum_detections
        ]

        return (
            boxes[selected],
            scores[selected],
            labels[selected],
        )


# ============================================================
# ANNOTATION RENDERER
# ============================================================

class ObjectDetectionRenderer:
    CATEGORY_COLORS = {
        ObjectCategory.PERSON: (
            255, 70, 70
        ),
        ObjectCategory.VEHICLE: (
            255, 190, 0
        ),
        ObjectCategory.BOAT: (
            0, 180, 255
        ),
        ObjectCategory.ANIMAL: (
            140, 220, 90
        ),
        ObjectCategory.OTHER: (
            190, 120, 255
        ),
    }

    @classmethod
    def render(
        cls,
        image: Image.Image,
        objects: Sequence[DetectedObject],
    ) -> Image.Image:
        output = image.convert(
            "RGB"
        ).copy()

        draw = ImageDraw.Draw(
            output
        )

        font = ImageFont.load_default()

        for detected in objects:
            box = detected.bounding_box

            color = cls.CATEGORY_COLORS[
                detected.category
            ]

            draw.rectangle(
                [
                    box.x_min,
                    box.y_min,
                    box.x_max,
                    box.y_max,
                ],
                outline=color,
                width=3,
            )

            label = (
                f"{detected.class_name} "
                f"{detected.confidence:.2f}"
            )

            text_bounds = draw.textbbox(
                (0, 0),
                label,
                font=font,
            )

            text_width = (
                text_bounds[2]
                - text_bounds[0]
            )

            text_height = (
                text_bounds[3]
                - text_bounds[1]
            )

            label_x = max(
                0,
                int(box.x_min),
            )

            label_y = max(
                0,
                int(box.y_min)
                - text_height
                - 8,
            )

            draw.rectangle(
                [
                    label_x,
                    label_y,
                    label_x + text_width + 8,
                    label_y + text_height + 8,
                ],
                fill=color,
            )

            draw.text(
                (
                    label_x + 4,
                    label_y + 4,
                ),
                label,
                fill=(0, 0, 0),
                font=font,
            )

        return output


# ============================================================
# INFERENCE ENGINE
# ============================================================

class ObjectDetectionInferenceEngine:
    def __init__(
        self,
        config: ObjectDetectionConfig,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device

        self.preprocessor = (
            DetectionImagePreprocessor(
                config
            )
        )

        self.filter = (
            DetectionPredictionFilter(
                config
            )
        )

        self.renderer = (
            ObjectDetectionRenderer()
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        *,
        image_bytes: bytes,
        image_id: str | None = None,
        filename: str | None = None,
    ) -> ObjectDetectionOutput:
        tensor, scale_x, scale_y = (
            self.preprocessor.preprocess(
                image
            )
        )

        self.model.eval()

        predictions = self.model(
            [
                tensor.to(
                    self.device
                )
            ]
        )

        if len(predictions) != 1:
            raise ObjectDetectionError(
                "Unexpected model output"
            )

        boxes, scores, labels = (
            self.filter.filter(
                predictions[0]
            )
        )

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        objects = []

        for box, score, label in zip(
            boxes,
            scores,
            labels,
        ):
            class_id = int(
                label
            )

            class_name = (
                COCO_CLASSES[class_id]
                if 0 <= class_id < len(COCO_CLASSES)
                else f"class_{class_id}"
            )

            category = classify_object(
                class_name
            )

            x_min = float(
                np.clip(
                    box[0] * scale_x,
                    0,
                    image.width,
                )
            )

            y_min = float(
                np.clip(
                    box[1] * scale_y,
                    0,
                    image.height,
                )
            )

            x_max = float(
                np.clip(
                    box[2] * scale_x,
                    0,
                    image.width,
                )
            )

            y_max = float(
                np.clip(
                    box[3] * scale_y,
                    0,
                    image.height,
                )
            )

            if (
                x_max <= x_min
                or y_max <= y_min
            ):
                continue

            detected = DetectedObject(
                detection_id=generate_id(),
                class_id=class_id,
                class_name=class_name,
                category=category,
                confidence=float(score),
                bounding_box=BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                ),
                requires_human_review=(
                    self.config.require_human_review
                ),
            )

            objects.append(
                detected
            )

        person_count = sum(
            item.category == ObjectCategory.PERSON
            for item in objects
        )

        vehicle_count = sum(
            item.category == ObjectCategory.VEHICLE
            for item in objects
        )

        boat_count = sum(
            item.category == ObjectCategory.BOAT
            for item in objects
        )

        animal_count = sum(
            item.category == ObjectCategory.ANIMAL
            for item in objects
        )

        warnings = [
            "Detections are advisory only",
            "Object detection does not establish distress",
            "COCO models are not specialized flood detectors",
            "Occlusion and aerial viewpoints can reduce accuracy",
            "Human verification required for rescue decisions",
        ]

        status = (
            DetectionStatus.REVIEW_REQUIRED
            if self.config.require_human_review
            else DetectionStatus.COMPLETED
        )

        result = ObjectDetectionResult(
            analysis_id=generate_id(),
            image_id=image_id or generate_id(),
            filename=filename,
            image_width=image.width,
            image_height=image.height,
            total_detections=len(objects),
            person_count=person_count,
            vehicle_count=vehicle_count,
            boat_count=boat_count,
            animal_count=animal_count,
            objects=tuple(objects),
            status=status,
            requires_human_review=(
                self.config.require_human_review
            ),
            image_sha256=hashlib.sha256(
                image_bytes
            ).hexdigest(),
            warnings=tuple(warnings),
            analyzed_at=utc_now(),
        )

        annotated = self.renderer.render(
            image,
            objects,
        )

        return ObjectDetectionOutput(
            result=result,
            annotated_image=annotated,
        )


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class ObjectDetectionCheckpointManager:
    @staticmethod
    def load(
        path: str | Path,
        *,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        source = Path(path)

        if not source.is_file():
            raise ObjectDetectionCheckpointError(
                "Checkpoint not found"
            )

        try:
            checkpoint = torch.load(
                source,
                map_location=device,
                weights_only=True,
            )

            if (
                isinstance(checkpoint, dict)
                and "model_state" in checkpoint
            ):
                state = checkpoint["model_state"]

            else:
                state = checkpoint

            model.load_state_dict(
                state,
                strict=True,
            )

            model.to(device)
            model.eval()

        except (
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            raise ObjectDetectionCheckpointError(
                "Invalid or incompatible checkpoint"
            ) from exc


# ============================================================
# APPLICATION SERVICE
# ============================================================

class ObjectDetectionService:
    def __init__(
        self,
        config: ObjectDetectionConfig | None = None,
        *,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.config = (
            config or ObjectDetectionConfig()
        )

        self.device = resolve_device(
            self.config.device
        )

        self.model = (
            model
            if model is not None
            else ObjectDetectionModelFactory.create(
                pretrained=self.config.pretrained
            )
        ).to(self.device)

        self.validator = DetectionImageValidator(
            self.config
        )

        self.engine = ObjectDetectionInferenceEngine(
            self.config,
            self.model,
            self.device,
        )

        # Caller-provided models require an explicit
        # trusted checkpoint before inference.
        self.model_ready = (
            model is None
            and self.config.pretrained
        )

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        ObjectDetectionCheckpointManager.load(
            path,
            model=self.model,
            device=self.device,
        )

        self.model_ready = True

    def detect_bytes(
        self,
        image_bytes: bytes,
        *,
        image_id: str | None = None,
        filename: str | None = None,
    ) -> ObjectDetectionOutput:
        if not self.model_ready:
            raise ObjectDetectionNotReadyError(
                "Load trained model weights first"
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

    def detect_path(
        self,
        path: str | Path,
        *,
        image_id: str | None = None,
    ) -> ObjectDetectionOutput:
        image, data = self.validator.load_path(
            path
        )

        if not self.model_ready:
            raise ObjectDetectionNotReadyError(
                "Load trained model weights first"
            )

        return self.engine.predict(
            image,
            image_bytes=data,
            image_id=image_id,
            filename=Path(path).name,
        )

    def detect_batch(
        self,
        images: Sequence[tuple[str, bytes]],
    ) -> ObjectDetectionBatchResult:
        if not images:
            raise ObjectDetectionValidationError(
                "Batch is empty"
            )

        if (
            len(images)
            > self.config.maximum_batch_size
        ):
            raise ObjectDetectionValidationError(
                "Batch exceeds maximum size"
            )

        results = []

        for filename, data in images:
            output = self.detect_bytes(
                data,
                filename=filename,
            )

            results.append(
                output.result
            )

        review_count = sum(
            result.requires_human_review
            for result in results
        )

        return ObjectDetectionBatchResult(
            batch_id=generate_id(),
            total_images=len(results),
            review_required_count=review_count,
            results=tuple(results),
            generated_at=utc_now(),
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class ObjectDetectionReportExporter:
    @staticmethod
    def export_json(
        result: ObjectDetectionResult,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise ObjectDetectionError(
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
    def export_annotated_image(
        output: ObjectDetectionOutput,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise ObjectDetectionError(
                "Annotated image already exists"
            )

        if destination.suffix.lower() != ".png":
            raise ObjectDetectionValidationError(
                "Annotated image must be PNG"
            )

        with destination.open("xb") as stream:
            output.annotated_image.save(
                stream,
                format="PNG",
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_object_detection_service(
    config: ObjectDetectionConfig | None = None,
) -> ObjectDetectionService:
    return ObjectDetectionService(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    service = create_object_detection_service(
        ObjectDetectionConfig(
            confidence_threshold=0.60,
            nms_iou_threshold=0.45,
            maximum_detections=100,
            pretrained=True,
            require_human_review=True,
        )
    )

    output = service.detect_path(
        "data/drone/flood_area_001.jpg"
    )

    print(
        json.dumps(
            output.result.to_dict(),
            indent=2,
        )
    )

    output_directory = Path(
        "outputs/object_detection"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    ObjectDetectionReportExporter.export_json(
        output.result,
        output_directory / "detections.json",
    )

    ObjectDetectionReportExporter.export_annotated_image(
        output,
        output_directory / "annotated.png",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ObjectDetectionError",
    "ObjectDetectionValidationError",
    "ObjectDetectionNotReadyError",
    "ObjectDetectionCheckpointError",
    "ObjectCategory",
    "DetectionStatus",
    "ObjectDetectionConfig",
    "BoundingBox",
    "DetectedObject",
    "ObjectDetectionResult",
    "ObjectDetectionOutput",
    "ObjectDetectionBatchResult",
    "DetectionImageValidator",
    "ObjectDetectionModelFactory",
    "DetectionImagePreprocessor",
    "DetectionPredictionFilter",
    "ObjectDetectionRenderer",
    "ObjectDetectionInferenceEngine",
    "ObjectDetectionCheckpointManager",
    "ObjectDetectionService",
    "ObjectDetectionReportExporter",
    "create_object_detection_service",
]
