
"""
Kerala Flood Relief Coordination Agent
Computer Vision Inference Service

File:
    backend/ai/computer_vision/inference.py

Python:
    3.11+

Dependencies:
    torch
    torchvision
    Pillow

Features:
    - CNN damage classification inference
    - EfficientNet / ResNet / Custom CNN
    - Trusted checkpoint loading
    - Image bytes and file inference
    - Batch inference
    - Image upload validation
    - Model warmup
    - Confidence and uncertainty evaluation
    - Human-review escalation
    - Structured JSON-compatible results
    - Async integration with FastAPI
    - Inference health and statistics
    - No autonomous emergency actions

Safety:
    Image predictions are advisory only.
    They cannot establish structural safety,
    repair costs, or emergency response priority.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
import time
import uuid

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .cnn_damage_model import (
    CNNDamageConfig,
    CNNDamageModelService,
    DamagePrediction,
    DamagePredictionStatus,
    DamageValidationError,
    create_cnn_damage_model_service,
)


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class CVInferenceError(Exception):
    """Base inference error."""


class CVInferenceValidationError(CVInferenceError):
    """Invalid inference request."""


class CVModelNotReadyError(CVInferenceError):
    """Model has not been loaded or validated."""


class CVInferenceBatchError(CVInferenceError):
    """Batch inference failure."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_nonnegative(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise CVInferenceValidationError(
            f"{name} must be finite and nonnegative"
        )


def prediction_to_dict(
    prediction: DamagePrediction,
) -> dict[str, Any]:
    return {
        "prediction_id": prediction.prediction_id,
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "class_probabilities": [
            {
                "class_name": item.class_name,
                "probability": item.probability,
            }
            for item in prediction.class_probabilities
        ],
        "probability_margin": (
            prediction.probability_margin
        ),
        "normalized_entropy": (
            prediction.normalized_entropy
        ),
        "status": prediction.status.value,
        "requires_human_review": (
            prediction.requires_human_review
        ),
        "review_reasons": [
            reason.value
            for reason in prediction.review_reasons
        ],
        "model_architecture": (
            prediction.model_architecture.value
        ),
        "model_trained": prediction.model_trained,
        "inference_time_ms": (
            prediction.inference_time_ms
        ),
        "predicted_at": (
            prediction.predicted_at.isoformat()
        ),
        "warnings": list(
            prediction.warnings
        ),
    }


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CVInferenceConfig:
    checkpoint_path: str | None = None

    require_trained_model: bool = True

    maximum_batch_size: int = 16

    maximum_concurrent_requests: int = 4

    request_timeout_seconds: float = 30.0

    enable_image_digest: bool = True

    enable_statistics: bool = True

    warmup_on_load: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_batch_size, bool)
            or not isinstance(self.maximum_batch_size, int)
            or self.maximum_batch_size < 1
        ):
            raise CVInferenceValidationError(
                "maximum_batch_size must be positive"
            )

        if (
            isinstance(self.maximum_concurrent_requests, bool)
            or not isinstance(self.maximum_concurrent_requests, int)
            or self.maximum_concurrent_requests < 1
        ):
            raise CVInferenceValidationError(
                "maximum_concurrent_requests must be positive"
            )

        finite_nonnegative(
            self.request_timeout_seconds,
            "request_timeout_seconds",
        )

        if self.request_timeout_seconds == 0:
            raise CVInferenceValidationError(
                "request_timeout_seconds must be positive"
            )


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

@dataclass(frozen=True)
class CVInferenceRequest:
    image_bytes: bytes

    request_id: str = field(
        default_factory=new_id
    )

    filename: str | None = None

    content_type: str | None = None

    metadata: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.image_bytes, bytes):
            raise CVInferenceValidationError(
                "image_bytes must be bytes"
            )

        if not self.image_bytes:
            raise CVInferenceValidationError(
                "Image is empty"
            )

        if not self.request_id.strip():
            raise CVInferenceValidationError(
                "request_id is required"
            )


@dataclass(frozen=True)
class CVInferenceResponse:
    request_id: str

    prediction: DamagePrediction

    image_digest: str | None

    processing_time_ms: float

    completed_at: datetime

    advisory_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prediction": prediction_to_dict(
                self.prediction
            ),
            "image_digest": self.image_digest,
            "processing_time_ms": (
                self.processing_time_ms
            ),
            "completed_at": (
                self.completed_at.isoformat()
            ),
            "advisory_only": self.advisory_only,
        }


@dataclass(frozen=True)
class CVBatchInferenceResponse:
    batch_id: str

    total_requests: int

    successful_requests: int

    failed_requests: int

    results: tuple[
        CVInferenceResponse,
        ...
    ]

    errors: tuple[
        dict[str, str],
        ...
    ]

    processing_time_ms: float

    completed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "total_requests": self.total_requests,
            "successful_requests": (
                self.successful_requests
            ),
            "failed_requests": (
                self.failed_requests
            ),
            "results": [
                item.to_dict()
                for item in self.results
            ],
            "errors": list(self.errors),
            "processing_time_ms": (
                self.processing_time_ms
            ),
            "completed_at": (
                self.completed_at.isoformat()
            ),
        }


@dataclass(frozen=True)
class CVInferenceHealth:
    ready: bool

    model_loaded: bool

    model_architecture: str

    device: str

    checkpoint_configured: bool

    checked_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_loaded": self.model_loaded,
            "model_architecture": (
                self.model_architecture
            ),
            "device": self.device,
            "checkpoint_configured": (
                self.checkpoint_configured
            ),
            "checked_at": (
                self.checked_at.isoformat()
            ),
        }


@dataclass(frozen=True)
class CVInferenceStatistics:
    total_requests: int

    successful_requests: int

    failed_requests: int

    review_required_requests: int

    average_processing_time_ms: float | None

    predictions_by_class: dict[str, int]


# ============================================================
# THREAD-SAFE STATISTICS
# ============================================================

class CVInferenceStatisticsTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._total = 0
        self._successful = 0
        self._failed = 0
        self._review_required = 0

        self._total_duration_ms = 0.0

        self._classes: Counter[str] = Counter()

    def record_success(
        self,
        response: CVInferenceResponse,
    ) -> None:
        with self._lock:
            self._total += 1
            self._successful += 1

            self._total_duration_ms += (
                response.processing_time_ms
            )

            self._classes[
                response.prediction.predicted_class
            ] += 1

            if response.prediction.requires_human_review:
                self._review_required += 1

    def record_failure(self) -> None:
        with self._lock:
            self._total += 1
            self._failed += 1

    def snapshot(self) -> CVInferenceStatistics:
        with self._lock:
            average = (
                self._total_duration_ms
                / self._successful
                if self._successful
                else None
            )

            return CVInferenceStatistics(
                total_requests=self._total,
                successful_requests=self._successful,
                failed_requests=self._failed,
                review_required_requests=(
                    self._review_required
                ),
                average_processing_time_ms=average,
                predictions_by_class=dict(
                    self._classes
                ),
            )


# ============================================================
# MODEL MANAGER
# ============================================================

class CVModelManager:
    """
    Owns a single loaded CNN model.

    Model weights must come from trusted,
    versioned training infrastructure.
    """

    def __init__(
        self,
        model_config: CNNDamageConfig,
        inference_config: CVInferenceConfig,
    ) -> None:
        self.model_config = model_config
        self.inference_config = inference_config

        self._service: (
            CNNDamageModelService | None
        ) = None

        self._lock = threading.RLock()

    def load(self) -> None:
        with self._lock:
            if self._service is not None:
                return

            if (
                self.inference_config.require_trained_model
                and not self.inference_config.checkpoint_path
            ):
                raise CVModelNotReadyError(
                    "A trained checkpoint is required"
                )

            service = create_cnn_damage_model_service(
                self.model_config
            )

            checkpoint = (
                self.inference_config.checkpoint_path
            )

            if checkpoint:
                path = Path(checkpoint)

                if not path.is_file():
                    raise CVModelNotReadyError(
                        "Checkpoint file does not exist"
                    )

                service.load_checkpoint(
                    path
                )

            if self.inference_config.warmup_on_load:
                self._warmup(service)

            self._service = service

            logger.info(
                "CNN model loaded: architecture=%s device=%s",
                self.model_config.architecture.value,
                service.device,
            )

    def _warmup(
        self,
        service: CNNDamageModelService,
    ) -> None:
        service.model.eval()

        with torch.inference_mode():
            dummy = torch.zeros(
                (
                    1,
                    3,
                    self.model_config.image_size,
                    self.model_config.image_size,
                ),
                device=service.device,
            )

            output = service.model(
                dummy
            )

            expected = (
                1,
                len(self.model_config.class_names),
            )

            if tuple(output.shape) != expected:
                raise CVModelNotReadyError(
                    "Unexpected model output shape"
                )

    def get_service(
        self,
    ) -> CNNDamageModelService:
        with self._lock:
            if self._service is None:
                raise CVModelNotReadyError(
                    "Model is not loaded"
                )

            return self._service

    def is_ready(self) -> bool:
        with self._lock:
            return (
                self._service is not None
                and (
                    self._service.model_trained
                    or not self.inference_config.require_trained_model
                )
            )


# ============================================================
# INFERENCE ENGINE
# ============================================================

class CVInferenceEngine:
    """
    Synchronous CNN inference engine.

    Concurrency:
        A model-level lock prevents unsafe
        simultaneous lifecycle operations.

    For high-throughput deployments, use
    controlled inference workers or a
    dedicated model-serving process.
    """

    def __init__(
        self,
        model_manager: CVModelManager,
        inference_config: CVInferenceConfig,
    ) -> None:
        self.model_manager = model_manager
        self.config = inference_config

        self.statistics = (
            CVInferenceStatisticsTracker()
        )

        self._inference_lock = threading.RLock()

    def predict(
        self,
        request: CVInferenceRequest,
    ) -> CVInferenceResponse:
        start = time.perf_counter()

        try:
            service = (
                self.model_manager.get_service()
            )

            if (
                self.config.require_trained_model
                and not service.model_trained
            ):
                raise CVModelNotReadyError(
                    "CNN model is not trained"
                )

            with self._inference_lock:
                prediction = (
                    service.predict_bytes(
                        request.image_bytes
                    )
                )

            duration = (
                time.perf_counter() - start
            ) * 1000

            response = CVInferenceResponse(
                request_id=request.request_id,
                prediction=prediction,
                image_digest=(
                    sha256_bytes(
                        request.image_bytes
                    )
                    if self.config.enable_image_digest
                    else None
                ),
                processing_time_ms=duration,
                completed_at=utc_now(),
            )

            if self.config.enable_statistics:
                self.statistics.record_success(
                    response
                )

            return response

        except Exception:
            if self.config.enable_statistics:
                self.statistics.record_failure()

            logger.exception(
                "CNN inference failed for request %s",
                request.request_id,
            )

            raise

    def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> CVInferenceResponse:
        return self.predict(
            CVInferenceRequest(
                image_bytes=image_bytes,
                filename=filename,
                content_type=content_type,
            )
        )

    def predict_file(
        self,
        path: str | Path,
    ) -> CVInferenceResponse:
        service = (
            self.model_manager.get_service()
        )

        image = service.inference.validator.load_path(
            path
        )

        # Re-encode validated image to PNG so
        # downstream inference receives bytes.
        import io

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="PNG",
        )

        return self.predict_bytes(
            buffer.getvalue(),
            filename=Path(path).name,
            content_type="image/png",
        )

    def predict_batch(
        self,
        requests: Sequence[CVInferenceRequest],
    ) -> CVBatchInferenceResponse:
        if len(requests) > self.config.maximum_batch_size:
            raise CVInferenceBatchError(
                "Batch size exceeds configured limit"
            )

        start = time.perf_counter()

        results: list[
            CVInferenceResponse
        ] = []

        errors: list[
            dict[str, str]
        ] = []

        for request in requests:
            try:
                results.append(
                    self.predict(request)
                )

            except (
                DamageValidationError,
                CVInferenceError,
            ) as exc:
                errors.append({
                    "request_id": request.request_id,
                    "error": str(exc),
                })

            except Exception:
                errors.append({
                    "request_id": request.request_id,
                    "error": "Internal inference failure",
                })

        return CVBatchInferenceResponse(
            batch_id=new_id(),
            total_requests=len(requests),
            successful_requests=len(results),
            failed_requests=len(errors),
            results=tuple(results),
            errors=tuple(errors),
            processing_time_ms=(
                time.perf_counter() - start
            ) * 1000,
            completed_at=utc_now(),
        )

    def health(self) -> CVInferenceHealth:
        service = (
            self.model_manager._service
        )

        return CVInferenceHealth(
            ready=self.model_manager.is_ready(),
            model_loaded=service is not None,
            model_architecture=(
                self.model_manager.model_config
                .architecture.value
            ),
            device=(
                str(service.device)
                if service is not None
                else "uninitialized"
            ),
            checkpoint_configured=bool(
                self.config.checkpoint_path
            ),
            checked_at=utc_now(),
        )

    def get_statistics(
        self,
    ) -> CVInferenceStatistics:
        return self.statistics.snapshot()


# ============================================================
# ASYNC INFERENCE SERVICE
# ============================================================

class AsyncCVInferenceService:
    """
    Async wrapper for FastAPI.

    Uses asyncio.to_thread so image decoding
    and model execution do not block the
    event loop.

    Note:
        asyncio.wait_for does not forcibly
        terminate a running worker thread.
        Use process-level timeouts for hard
        execution limits.
    """

    def __init__(
        self,
        engine: CVInferenceEngine,
    ) -> None:
        self.engine = engine

        self._semaphore = asyncio.Semaphore(
            engine.config.maximum_concurrent_requests
        )

    async def predict(
        self,
        request: CVInferenceRequest,
    ) -> CVInferenceResponse:
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self.engine.predict,
                        request,
                    ),
                    timeout=(
                        self.engine.config
                        .request_timeout_seconds
                    ),
                )

            except asyncio.TimeoutError as exc:
                raise CVInferenceError(
                    "Inference request timed out"
                ) from exc

    async def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> CVInferenceResponse:
        request = CVInferenceRequest(
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
        )

        return await self.predict(
            request
        )

    async def predict_batch(
        self,
        requests: Sequence[CVInferenceRequest],
    ) -> CVBatchInferenceResponse:
        if len(requests) > self.engine.config.maximum_batch_size:
            raise CVInferenceBatchError(
                "Batch size exceeds configured limit"
            )

        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self.engine.predict_batch,
                        requests,
                    ),
                    timeout=(
                        self.engine.config
                        .request_timeout_seconds
                    ),
                )

            except asyncio.TimeoutError as exc:
                raise CVInferenceError(
                    "Batch inference timed out"
                ) from exc


# ============================================================
# APPLICATION SERVICE
# ============================================================

class ComputerVisionInferenceService:
    """
    Main application-facing service.

    Integrates with:
        backend/api/damage.py
        backend/services/damage_service.py
        backend/ai/computer_vision/cnn_damage_model.py
        backend/ai/evaluation/model_evaluation.py
        backend/ai/safety/human_oversight.py
        backend/mlops/model_registry.py
    """

    def __init__(
        self,
        model_config: CNNDamageConfig | None = None,
        inference_config: CVInferenceConfig | None = None,
    ) -> None:
        self.model_config = (
            model_config or CNNDamageConfig()
        )

        self.inference_config = (
            inference_config or CVInferenceConfig()
        )

        self.model_manager = CVModelManager(
            self.model_config,
            self.inference_config,
        )

        self.engine = CVInferenceEngine(
            self.model_manager,
            self.inference_config,
        )

        self.async_service = AsyncCVInferenceService(
            self.engine
        )

    def load_model(self) -> None:
        self.model_manager.load()

    def predict(
        self,
        image_bytes: bytes,
    ) -> CVInferenceResponse:
        return self.engine.predict_bytes(
            image_bytes
        )

    def predict_file(
        self,
        path: str | Path,
    ) -> CVInferenceResponse:
        return self.engine.predict_file(
            path
        )

    async def predict_async(
        self,
        image_bytes: bytes,
    ) -> CVInferenceResponse:
        return await self.async_service.predict_bytes(
            image_bytes
        )

    def predict_batch(
        self,
        requests: Sequence[CVInferenceRequest],
    ) -> CVBatchInferenceResponse:
        return self.engine.predict_batch(
            requests
        )

    def health(self) -> CVInferenceHealth:
        return self.engine.health()

    def statistics(
        self,
    ) -> CVInferenceStatistics:
        return self.engine.get_statistics()


# ============================================================
# FACTORY
# ============================================================

def create_cv_inference_service(
    *,
    model_config: CNNDamageConfig | None = None,
    inference_config: CVInferenceConfig | None = None,
    auto_load: bool = False,
) -> ComputerVisionInferenceService:
    service = ComputerVisionInferenceService(
        model_config=model_config,
        inference_config=inference_config,
    )

    if auto_load:
        service.load_model()

    return service


__all__ = [
    "CVInferenceError",
    "CVInferenceValidationError",
    "CVModelNotReadyError",
    "CVInferenceBatchError",
    "CVInferenceConfig",
    "CVInferenceRequest",
    "CVInferenceResponse",
    "CVBatchInferenceResponse",
    "CVInferenceHealth",
    "CVInferenceStatistics",
    "CVInferenceStatisticsTracker",
    "CVModelManager",
    "CVInferenceEngine",
    "AsyncCVInferenceService",
    "ComputerVisionInferenceService",
    "create_cv_inference_service",
]
