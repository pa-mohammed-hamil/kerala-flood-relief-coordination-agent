"""Placeholder module: ai/computer_vision/visual_evaluation.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
File: backend/ai/computer_vision/visual_evaluation.py

Evaluation and benchmarking for computer vision models.

Supported tasks:
    - Flood-water segmentation
    - Road segmentation
    - Building segmentation
    - Debris segmentation
    - Object detection
    - Building damage classification
    - Road obstruction screening
    - Drone and satellite image evaluation

Metrics:
    Segmentation:
        Pixel accuracy
        Precision / Recall / F1
        IoU / Dice
        Per-class metrics
        Mean IoU
        Frequency-weighted IoU

    Object detection:
        IoU matching
        Precision / Recall / F1
        AP50
        mAP50
        Per-class AP50

    Classification:
        Accuracy
        Macro precision / recall / F1
        Confusion matrix
        Per-class metrics

    Reliability:
        Expected calibration error
        Brier score
        Abstention / review coverage

    Operations:
        Latency percentiles
        Evaluation thresholds
        JSON reports

Dependencies:
    pip install numpy

Notes:
    - AP50 uses greedy confidence-ordered matching.
    - mAP50 is not COCO mAP@[.50:.95].
    - Evaluation requires independent ground truth.
    - Use location/event-disjoint test splits.
    - Evaluation never authorizes emergency actions.
"""

from __future__ import annotations

import json
import math
import uuid

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class VisualEvaluationError(Exception):
    pass


class VisualEvaluationValidationError(
    VisualEvaluationError
):
    pass


class VisualEvaluationExportError(
    VisualEvaluationError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class VisualTask(str, Enum):
    SEGMENTATION = "segmentation"
    OBJECT_DETECTION = "object_detection"
    CLASSIFICATION = "classification"
    CALIBRATION = "calibration"


class EvaluationGateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class VisualEvaluationConfig:
    number_of_classes: int = 2

    ignore_index: int = 255

    detection_iou_threshold: float = 0.50

    confidence_threshold: float = 0.50

    calibration_bins: int = 15

    minimum_mean_iou: float = 0.65
    minimum_macro_f1: float = 0.70
    minimum_map50: float = 0.50

    maximum_ece: float = 0.15

    require_human_review: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.number_of_classes, int)
            or self.number_of_classes < 2
        ):
            raise VisualEvaluationValidationError(
                "number_of_classes must be >=2"
            )

        if (
            not isinstance(self.calibration_bins, int)
            or self.calibration_bins <= 0
        ):
            raise VisualEvaluationValidationError(
                "calibration_bins must be positive"
            )

        for name in (
            "detection_iou_threshold",
            "confidence_threshold",
            "minimum_mean_iou",
            "minimum_macro_f1",
            "minimum_map50",
            "maximum_ece",
        ):
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise VisualEvaluationValidationError(
                    f"{name} must be in [0,1]"
                )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    weighted_f1: float

    confusion_matrix: tuple[
        tuple[int, ...],
        ...
    ]

    per_class: tuple[
        dict[str, Any],
        ...
    ]


@dataclass(frozen=True)
class SegmentationMetrics:
    pixel_accuracy: float

    mean_iou: float
    mean_dice: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    frequency_weighted_iou: float

    confusion_matrix: tuple[
        tuple[int, ...],
        ...
    ]

    per_class: tuple[
        dict[str, Any],
        ...
    ]


@dataclass(frozen=True)
class DetectionBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    class_id: int

    confidence: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
            self.confidence,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise VisualEvaluationValidationError(
                "Detection values must be finite"
            )

        if (
            self.x_max <= self.x_min
            or self.y_max <= self.y_min
        ):
            raise VisualEvaluationValidationError(
                "Invalid bounding box"
            )

        if not 0 <= self.confidence <= 1:
            raise VisualEvaluationValidationError(
                "Confidence must be in [0,1]"
            )

        if self.class_id < 0:
            raise VisualEvaluationValidationError(
                "class_id must be nonnegative"
            )


@dataclass(frozen=True)
class DetectionSample:
    image_id: str

    ground_truth: tuple[
        DetectionBox,
        ...
    ]

    predictions: tuple[
        DetectionBox,
        ...
    ]


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float

    true_positives: int
    false_positives: int
    false_negatives: int

    map50: float

    per_class: tuple[
        dict[str, Any],
        ...
    ]


@dataclass(frozen=True)
class CalibrationMetrics:
    expected_calibration_error: float
    brier_score: float

    bin_results: tuple[
        dict[str, float | int],
        ...
    ]


@dataclass(frozen=True)
class LatencyMetrics:
    count: int

    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

    maximum_ms: float


@dataclass(frozen=True)
class EvaluationGate:
    metric_name: str
    observed_value: float | None
    threshold: float

    comparison: str
    status: EvaluationGateStatus


@dataclass(frozen=True)
class VisualEvaluationReport:
    report_id: str

    model_name: str
    model_version: str

    task: VisualTask

    metrics: dict[str, Any]

    gates: tuple[
        EvaluationGate,
        ...
    ]

    overall_status: EvaluationGateStatus

    dataset_name: str
    dataset_version: str

    sample_count: int

    requires_human_review: bool

    warnings: tuple[str, ...]

    evaluated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        def serialize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value

            if isinstance(value, datetime):
                return value.isoformat()

            if isinstance(value, dict):
                return {
                    key: serialize(item)
                    for key, item in value.items()
                }

            if isinstance(value, (list, tuple)):
                return [
                    serialize(item)
                    for item in value
                ]

            return value

        return serialize(asdict(self))


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def safe_ratio(
    numerator: float,
    denominator: float,
    *,
    empty_value: float = 0.0,
) -> float:
    if denominator == 0:
        return empty_value

    return float(
        numerator / denominator
    )


def harmonic_f1(
    precision: float,
    recall: float,
) -> float:
    return safe_ratio(
        2.0 * precision * recall,
        precision + recall,
    )


def mean_or_zero(
    values: Sequence[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

class ConfusionMatrixBuilder:
    @staticmethod
    def build(
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        *,
        number_of_classes: int,
        ignore_index: int | None = None,
    ) -> np.ndarray:
        truth = np.asarray(
            ground_truth
        ).reshape(-1)

        predicted = np.asarray(
            predictions
        ).reshape(-1)

        if truth.shape != predicted.shape:
            raise VisualEvaluationValidationError(
                "Ground truth and predictions differ"
            )

        if truth.size == 0:
            raise VisualEvaluationValidationError(
                "Evaluation arrays are empty"
            )

        if not (
            np.issubdtype(
                truth.dtype,
                np.integer,
            )
            and np.issubdtype(
                predicted.dtype,
                np.integer,
            )
        ):
            raise VisualEvaluationValidationError(
                "Class labels must be integers"
            )

        if ignore_index is not None:
            valid = (
                truth != ignore_index
            )

            truth = truth[valid]
            predicted = predicted[valid]

        if (
            np.any(truth < 0)
            or np.any(truth >= number_of_classes)
            or np.any(predicted < 0)
            or np.any(predicted >= number_of_classes)
        ):
            raise VisualEvaluationValidationError(
                "Class labels out of range"
            )

        encoded = (
            truth * number_of_classes
            + predicted
        )

        matrix = np.bincount(
            encoded,
            minlength=(
                number_of_classes ** 2
            ),
        )

        return matrix.reshape(
            number_of_classes,
            number_of_classes,
        ).astype(np.int64)


# ============================================================
# CLASSIFICATION EVALUATOR
# ============================================================

class ClassificationEvaluator:
    @staticmethod
    def evaluate(
        ground_truth: Sequence[int],
        predictions: Sequence[int],
        *,
        number_of_classes: int,
    ) -> ClassificationMetrics:
        matrix = ConfusionMatrixBuilder.build(
            np.asarray(
                ground_truth,
                dtype=np.int64,
            ),
            np.asarray(
                predictions,
                dtype=np.int64,
            ),
            number_of_classes=number_of_classes,
        )

        total = int(
            matrix.sum()
        )

        correct = int(
            np.trace(matrix)
        )

        per_class = []

        precision_values = []
        recall_values = []
        f1_values = []

        weighted_f1_sum = 0.0

        for class_id in range(
            number_of_classes
        ):
            tp = int(
                matrix[class_id, class_id]
            )

            fp = int(
                matrix[:, class_id].sum() - tp
            )

            fn = int(
                matrix[class_id, :].sum() - tp
            )

            support = int(
                matrix[class_id, :].sum()
            )

            precision = safe_ratio(
                tp,
                tp + fp,
            )

            recall = safe_ratio(
                tp,
                tp + fn,
            )

            f1 = harmonic_f1(
                precision,
                recall,
            )

            precision_values.append(
                precision
            )

            recall_values.append(
                recall
            )

            f1_values.append(
                f1
            )

            weighted_f1_sum += (
                f1 * support
            )

            per_class.append({
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            })

        return ClassificationMetrics(
            accuracy=safe_ratio(
                correct,
                total,
            ),
            macro_precision=mean_or_zero(
                precision_values
            ),
            macro_recall=mean_or_zero(
                recall_values
            ),
            macro_f1=mean_or_zero(
                f1_values
            ),
            weighted_f1=safe_ratio(
                weighted_f1_sum,
                total,
            ),
            confusion_matrix=tuple(
                tuple(int(v) for v in row)
                for row in matrix
            ),
            per_class=tuple(per_class),
        )


# ============================================================
# SEGMENTATION EVALUATOR
# ============================================================

class SegmentationEvaluator:
    @staticmethod
    def evaluate(
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        *,
        number_of_classes: int,
        ignore_index: int | None = 255,
    ) -> SegmentationMetrics:
        matrix = ConfusionMatrixBuilder.build(
            ground_truth,
            predictions,
            number_of_classes=number_of_classes,
            ignore_index=ignore_index,
        )

        total = int(
            matrix.sum()
        )

        if total == 0:
            raise VisualEvaluationValidationError(
                "No valid segmentation pixels"
            )

        per_class = []

        precision_values = []
        recall_values = []
        f1_values = []
        iou_values = []
        dice_values = []

        frequency_weighted_iou = 0.0

        for class_id in range(
            number_of_classes
        ):
            tp = int(
                matrix[class_id, class_id]
            )

            fp = int(
                matrix[:, class_id].sum() - tp
            )

            fn = int(
                matrix[class_id, :].sum() - tp
            )

            support = int(
                matrix[class_id, :].sum()
            )

            precision = safe_ratio(
                tp,
                tp + fp,
            )

            recall = safe_ratio(
                tp,
                tp + fn,
            )

            f1 = harmonic_f1(
                precision,
                recall,
            )

            iou = safe_ratio(
                tp,
                tp + fp + fn,
            )

            dice = safe_ratio(
                2 * tp,
                2 * tp + fp + fn,
            )

            frequency_weighted_iou += (
                support / total
            ) * iou

            precision_values.append(
                precision
            )

            recall_values.append(
                recall
            )

            f1_values.append(
                f1
            )

            iou_values.append(
                iou
            )

            dice_values.append(
                dice
            )

            per_class.append({
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "dice": dice,
                "support_pixels": support,
            })

        return SegmentationMetrics(
            pixel_accuracy=safe_ratio(
                np.trace(matrix),
                total,
            ),
            mean_iou=mean_or_zero(
                iou_values
            ),
            mean_dice=mean_or_zero(
                dice_values
            ),
            macro_precision=mean_or_zero(
                precision_values
            ),
            macro_recall=mean_or_zero(
                recall_values
            ),
            macro_f1=mean_or_zero(
                f1_values
            ),
            frequency_weighted_iou=float(
                frequency_weighted_iou
            ),
            confusion_matrix=tuple(
                tuple(int(v) for v in row)
                for row in matrix
            ),
            per_class=tuple(per_class),
        )


# ============================================================
# BOUNDING BOX IOU
# ============================================================

class BoundingBoxEvaluator:
    @staticmethod
    def calculate_iou(
        first: DetectionBox,
        second: DetectionBox,
    ) -> float:
        x1 = max(
            first.x_min,
            second.x_min,
        )

        y1 = max(
            first.y_min,
            second.y_min,
        )

        x2 = min(
            first.x_max,
            second.x_max,
        )

        y2 = min(
            first.y_max,
            second.y_max,
        )

        intersection_width = max(
            0.0,
            x2 - x1,
        )

        intersection_height = max(
            0.0,
            y2 - y1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        first_area = (
            (first.x_max - first.x_min)
            * (first.y_max - first.y_min)
        )

        second_area = (
            (second.x_max - second.x_min)
            * (second.y_max - second.y_min)
        )

        union = (
            first_area
            + second_area
            - intersection
        )

        return safe_ratio(
            intersection,
            union,
        )


# ============================================================
# AVERAGE PRECISION
# ============================================================

class AveragePrecisionCalculator:
    @staticmethod
    def calculate(
        recalls: np.ndarray,
        precisions: np.ndarray,
    ) -> float:
        if recalls.size == 0:
            return 0.0

        # Precision envelope integration.
        mrec = np.concatenate(
            (
                [0.0],
                recalls,
                [1.0],
            )
        )

        mpre = np.concatenate(
            (
                [0.0],
                precisions,
                [0.0],
            )
        )

        for index in range(
            mpre.size - 2,
            -1,
            -1,
        ):
            mpre[index] = max(
                mpre[index],
                mpre[index + 1],
            )

        changing_indices = np.where(
            mrec[1:] != mrec[:-1]
        )[0]

        return float(
            np.sum(
                (
                    mrec[changing_indices + 1]
                    - mrec[changing_indices]
                )
                * mpre[changing_indices + 1]
            )
        )


# ============================================================
# OBJECT DETECTION EVALUATOR
# ============================================================

class ObjectDetectionEvaluator:
    def __init__(
        self,
        config: VisualEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        samples: Sequence[DetectionSample],
    ) -> DetectionMetrics:
        if not samples:
            raise VisualEvaluationValidationError(
                "Detection samples are empty"
            )

        if len({
            sample.image_id
            for sample in samples
        }) != len(samples):
            raise VisualEvaluationValidationError(
                "Duplicate image IDs"
            )

        class_ids = set()

        for sample in samples:
            for box in (
                sample.ground_truth
                + sample.predictions
            ):
                if box.class_id >= (
                    self.config.number_of_classes
                ):
                    raise VisualEvaluationValidationError(
                        "Detection class out of range"
                    )

                class_ids.add(
                    box.class_id
                )

        per_class = []

        total_tp = 0
        total_fp = 0
        total_fn = 0

        ap_values = []

        for class_id in sorted(
            class_ids
        ):
            ground_truth_by_image = {}

            predictions = []

            total_gt = 0

            for sample in samples:
                ground_truth = [
                    box
                    for box in sample.ground_truth
                    if box.class_id == class_id
                ]

                ground_truth_by_image[
                    sample.image_id
                ] = {
                    "boxes": ground_truth,
                    "matched": set(),
                }

                total_gt += len(
                    ground_truth
                )

                for box in sample.predictions:
                    if box.class_id == class_id:
                        predictions.append(
                            (
                                sample.image_id,
                                box,
                            )
                        )

            predictions.sort(
                key=lambda item: (
                    -item[1].confidence,
                    item[0],
                )
            )

            tp_flags = []
            fp_flags = []

            for image_id, prediction in predictions:
                image_ground_truth = (
                    ground_truth_by_image[
                        image_id
                    ]
                )

                gt_boxes = image_ground_truth[
                    "boxes"
                ]

                matched = image_ground_truth[
                    "matched"
                ]

                best_iou = 0.0
                best_index = None

                for index, gt_box in enumerate(
                    gt_boxes
                ):
                    if index in matched:
                        continue

                    iou = (
                        BoundingBoxEvaluator
                        .calculate_iou(
                            prediction,
                            gt_box,
                        )
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_index = index

                if (
                    best_index is not None
                    and best_iou
                    >= self.config.detection_iou_threshold
                ):
                    matched.add(
                        best_index
                    )

                    tp_flags.append(1)
                    fp_flags.append(0)

                else:
                    tp_flags.append(0)
                    fp_flags.append(1)

            tp_array = np.cumsum(
                tp_flags,
                dtype=np.float64,
            )

            fp_array = np.cumsum(
                fp_flags,
                dtype=np.float64,
            )

            recalls = (
                tp_array / total_gt
                if total_gt > 0
                else np.zeros_like(
                    tp_array
                )
            )

            precisions = np.divide(
                tp_array,
                tp_array + fp_array,
                out=np.zeros_like(
                    tp_array
                ),
                where=(
                    tp_array + fp_array
                ) > 0,
            )

            ap50 = (
                AveragePrecisionCalculator
                .calculate(
                    recalls,
                    precisions,
                )
                if total_gt > 0
                else None
            )

            if ap50 is not None:
                ap_values.append(
                    ap50
                )

            # Thresholded detection metrics.
            thresholded_predictions = [
                (image_id, box)
                for image_id, box in predictions
                if box.confidence
                >= self.config.confidence_threshold
            ]

            # Because predictions are confidence-sorted,
            # thresholded predictions are a prefix.
            threshold_count = len(
                thresholded_predictions
            )

            class_tp = int(
                sum(
                    tp_flags[:threshold_count]
                )
            )

            class_fp = int(
                sum(
                    fp_flags[:threshold_count]
                )
            )

            class_fn = max(
                0,
                total_gt - class_tp,
            )

            total_tp += class_tp
            total_fp += class_fp
            total_fn += class_fn

            precision = safe_ratio(
                class_tp,
                class_tp + class_fp,
            )

            recall = safe_ratio(
                class_tp,
                class_tp + class_fn,
            )

            per_class.append({
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "f1": harmonic_f1(
                    precision,
                    recall,
                ),
                "ap50": ap50,
                "ground_truth_count": total_gt,
                "true_positives": class_tp,
                "false_positives": class_fp,
                "false_negatives": class_fn,
            })

        precision = safe_ratio(
            total_tp,
            total_tp + total_fp,
        )

        recall = safe_ratio(
            total_tp,
            total_tp + total_fn,
        )

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1=harmonic_f1(
                precision,
                recall,
            ),
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            map50=mean_or_zero(
                ap_values
            ),
            per_class=tuple(per_class),
        )


# ============================================================
# CALIBRATION EVALUATOR
# ============================================================

class CalibrationEvaluator:
    @staticmethod
    def evaluate(
        probabilities: np.ndarray,
        ground_truth: np.ndarray,
        *,
        bins: int = 15,
    ) -> CalibrationMetrics:
        """
        Binary probability calibration.

        probabilities:
            Shape (N,), positive-class probability.

        ground_truth:
            Shape (N,), labels 0 or 1.

        Brier score:
            Mean squared probability error.

        ECE:
            Confidence-binned difference between
            predicted positive probability and
            observed positive-class frequency.
        """
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        ).reshape(-1)

        truth = np.asarray(
            ground_truth
        ).reshape(-1)

        if (
            probabilities.size == 0
            or probabilities.shape != truth.shape
        ):
            raise VisualEvaluationValidationError(
                "Invalid calibration arrays"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise VisualEvaluationValidationError(
                "Probabilities must be finite"
            )

        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            raise VisualEvaluationValidationError(
                "Probabilities must be in [0,1]"
            )

        if not np.isin(
            truth,
            [0, 1],
        ).all():
            raise VisualEvaluationValidationError(
                "Ground truth must be binary"
            )

        if bins <= 0:
            raise VisualEvaluationValidationError(
                "bins must be positive"
            )

        brier_score = float(
            np.mean(
                (
                    probabilities - truth
                ) ** 2
            )
        )

        edges = np.linspace(
            0.0,
            1.0,
            bins + 1,
        )

        expected_calibration_error = 0.0

        bin_results = []

        for index in range(bins):
            lower = edges[index]
            upper = edges[index + 1]

            if index == bins - 1:
                selected = (
                    (probabilities >= lower)
                    & (probabilities <= upper)
                )
            else:
                selected = (
                    (probabilities >= lower)
                    & (probabilities < upper)
                )

            count = int(
                selected.sum()
            )

            if count == 0:
                continue

            mean_probability = float(
                probabilities[selected].mean()
            )

            observed_frequency = float(
                truth[selected].mean()
            )

            gap = abs(
                mean_probability
                - observed_frequency
            )

            expected_calibration_error += (
                count / probabilities.size
            ) * gap

            bin_results.append({
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": (
                    mean_probability
                ),
                "observed_frequency": (
                    observed_frequency
                ),
                "calibration_gap": gap,
            })

        return CalibrationMetrics(
            expected_calibration_error=float(
                expected_calibration_error
            ),
            brier_score=brier_score,
            bin_results=tuple(bin_results),
        )


# ============================================================
# LATENCY EVALUATOR
# ============================================================

class LatencyEvaluator:
    @staticmethod
    def evaluate(
        latency_ms: Sequence[float],
    ) -> LatencyMetrics:
        values = np.asarray(
            latency_ms,
            dtype=np.float64,
        )

        if values.size == 0:
            raise VisualEvaluationValidationError(
                "Latency values are empty"
            )

        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise VisualEvaluationValidationError(
                "Invalid latency values"
            )

        return LatencyMetrics(
            count=int(values.size),
            mean_ms=float(values.mean()),
            p50_ms=float(
                np.percentile(values, 50)
            ),
            p90_ms=float(
                np.percentile(values, 90)
            ),
            p95_ms=float(
                np.percentile(values, 95)
            ),
            p99_ms=float(
                np.percentile(values, 99)
            ),
            maximum_ms=float(
                values.max()
            ),
        )


# ============================================================
# EVALUATION GATES
# ============================================================

class VisualEvaluationGateEngine:
    @staticmethod
    def minimum_gate(
        metric_name: str,
        value: float | None,
        threshold: float,
    ) -> EvaluationGate:
        if value is None:
            status = (
                EvaluationGateStatus.NOT_EVALUATED
            )

        elif value >= threshold:
            status = (
                EvaluationGateStatus.PASSED
            )

        else:
            status = (
                EvaluationGateStatus.FAILED
            )

        return EvaluationGate(
            metric_name=metric_name,
            observed_value=value,
            threshold=threshold,
            comparison=">=",
            status=status,
        )

    @staticmethod
    def maximum_gate(
        metric_name: str,
        value: float | None,
        threshold: float,
    ) -> EvaluationGate:
        if value is None:
            status = (
                EvaluationGateStatus.NOT_EVALUATED
            )

        elif value <= threshold:
            status = (
                EvaluationGateStatus.PASSED
            )

        else:
            status = (
                EvaluationGateStatus.FAILED
            )

        return EvaluationGate(
            metric_name=metric_name,
            observed_value=value,
            threshold=threshold,
            comparison="<=",
            status=status,
        )

    @staticmethod
    def overall(
        gates: Sequence[EvaluationGate],
    ) -> EvaluationGateStatus:
        if not gates:
            return (
                EvaluationGateStatus.NOT_EVALUATED
            )

        if any(
            gate.status
            == EvaluationGateStatus.FAILED
            for gate in gates
        ):
            return EvaluationGateStatus.FAILED

        if any(
            gate.status
            == EvaluationGateStatus.NOT_EVALUATED
            for gate in gates
        ):
            return (
                EvaluationGateStatus.NOT_EVALUATED
            )

        return EvaluationGateStatus.PASSED


# ============================================================
# MAIN VISUAL EVALUATION SERVICE
# ============================================================

class VisualEvaluationService:
    def __init__(
        self,
        config: VisualEvaluationConfig | None = None,
    ) -> None:
        self.config = (
            config or VisualEvaluationConfig()
        )

        self.segmentation_evaluator = (
            SegmentationEvaluator()
        )

        self.classification_evaluator = (
            ClassificationEvaluator()
        )

        self.detection_evaluator = (
            ObjectDetectionEvaluator(
                self.config
            )
        )

        self.calibration_evaluator = (
            CalibrationEvaluator()
        )

        self.latency_evaluator = (
            LatencyEvaluator()
        )

        self.gate_engine = (
            VisualEvaluationGateEngine()
        )

    def evaluate_segmentation(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        *,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        sample_count: int,
    ) -> VisualEvaluationReport:
        metrics = (
            self.segmentation_evaluator.evaluate(
                ground_truth,
                predictions,
                number_of_classes=(
                    self.config.number_of_classes
                ),
                ignore_index=(
                    self.config.ignore_index
                ),
            )
        )

        gates = (
            self.gate_engine.minimum_gate(
                "mean_iou",
                metrics.mean_iou,
                self.config.minimum_mean_iou,
            ),
            self.gate_engine.minimum_gate(
                "macro_f1",
                metrics.macro_f1,
                self.config.minimum_macro_f1,
            ),
        )

        return self._build_report(
            task=VisualTask.SEGMENTATION,
            metrics=asdict(metrics),
            gates=gates,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            sample_count=sample_count,
        )

    def evaluate_classification(
        self,
        ground_truth: Sequence[int],
        predictions: Sequence[int],
        *,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
    ) -> VisualEvaluationReport:
        metrics = (
            self.classification_evaluator.evaluate(
                ground_truth,
                predictions,
                number_of_classes=(
                    self.config.number_of_classes
                ),
            )
        )

        gates = (
            self.gate_engine.minimum_gate(
                "macro_f1",
                metrics.macro_f1,
                self.config.minimum_macro_f1,
            ),
        )

        return self._build_report(
            task=VisualTask.CLASSIFICATION,
            metrics=asdict(metrics),
            gates=gates,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            sample_count=len(ground_truth),
        )

    def evaluate_detection(
        self,
        samples: Sequence[DetectionSample],
        *,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
    ) -> VisualEvaluationReport:
        metrics = (
            self.detection_evaluator.evaluate(
                samples
            )
        )

        gates = (
            self.gate_engine.minimum_gate(
                "map50",
                metrics.map50,
                self.config.minimum_map50,
            ),
        )

        return self._build_report(
            task=VisualTask.OBJECT_DETECTION,
            metrics=asdict(metrics),
            gates=gates,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            sample_count=len(samples),
        )

    def evaluate_calibration(
        self,
        probabilities: np.ndarray,
        ground_truth: np.ndarray,
        *,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
    ) -> VisualEvaluationReport:
        metrics = (
            self.calibration_evaluator.evaluate(
                probabilities,
                ground_truth,
                bins=self.config.calibration_bins,
            )
        )

        gates = (
            self.gate_engine.maximum_gate(
                "expected_calibration_error",
                metrics.expected_calibration_error,
                self.config.maximum_ece,
            ),
        )

        return self._build_report(
            task=VisualTask.CALIBRATION,
            metrics=asdict(metrics),
            gates=gates,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            sample_count=int(
                np.asarray(
                    ground_truth
                ).size
            ),
        )

    def _build_report(
        self,
        *,
        task: VisualTask,
        metrics: dict[str, Any],
        gates: Sequence[EvaluationGate],
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        sample_count: int,
    ) -> VisualEvaluationReport:
        if sample_count <= 0:
            raise VisualEvaluationValidationError(
                "sample_count must be positive"
            )

        overall_status = (
            self.gate_engine.overall(
                gates
            )
        )

        warnings = [
            "Evaluation scores are dataset-dependent",
            "Thresholds are configurable and not "
            "independently validated safety criteria",
            "Use geographically and temporally "
            "independent test data",
            "Passing metric gates does not establish "
            "operational safety",
            "Human approval is required for "
            "emergency-response decisions",
        ]

        if overall_status != (
            EvaluationGateStatus.PASSED
        ):
            warnings.append(
                "Evaluation quality gates "
                "have not all passed"
            )

        return VisualEvaluationReport(
            report_id=generate_id(),
            model_name=model_name,
            model_version=model_version,
            task=task,
            metrics=metrics,
            gates=tuple(gates),
            overall_status=overall_status,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            sample_count=sample_count,
            requires_human_review=True,
            warnings=tuple(warnings),
            evaluated_at=utc_now(),
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class VisualEvaluationReportExporter:
    @staticmethod
    def export_json(
        report: VisualEvaluationReport,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise VisualEvaluationExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                report.to_dict(),
                stream,
                indent=2,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_visual_evaluation_service(
    config: VisualEvaluationConfig | None = None,
) -> VisualEvaluationService:
    return VisualEvaluationService(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    service = create_visual_evaluation_service(
        VisualEvaluationConfig(
            number_of_classes=2,
            minimum_mean_iou=0.65,
            minimum_macro_f1=0.70,
        )
    )

    # Illustrative ground truth and predictions.
    # Replace with actual held-out dataset results.
    ground_truth = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int64,
    )

    predictions = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int64,
    )

    report = service.evaluate_segmentation(
        ground_truth,
        predictions,
        model_name="flood-water-unet",
        model_version="1.0.0",
        dataset_name="illustrative-flood-masks",
        dataset_version="example",
        sample_count=1,
    )

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
        )
    )

    VisualEvaluationReportExporter.export_json(
        report,
        "outputs/evaluation/visual_report.json",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "VisualEvaluationError",
    "VisualEvaluationValidationError",
    "VisualEvaluationExportError",
    "VisualTask",
    "EvaluationGateStatus",
    "VisualEvaluationConfig",
    "ClassificationMetrics",
    "SegmentationMetrics",
    "DetectionBox",
    "DetectionSample",
    "DetectionMetrics",
    "CalibrationMetrics",
    "LatencyMetrics",
    "EvaluationGate",
    "VisualEvaluationReport",
    "ConfusionMatrixBuilder",
    "ClassificationEvaluator",
    "SegmentationEvaluator",
    "BoundingBoxEvaluator",
    "AveragePrecisionCalculator",
    "ObjectDetectionEvaluator",
    "CalibrationEvaluator",
    "LatencyEvaluator",
    "VisualEvaluationGateEngine",
    "VisualEvaluationService",
    "VisualEvaluationReportExporter",
    "create_visual_evaluation_service",
]
