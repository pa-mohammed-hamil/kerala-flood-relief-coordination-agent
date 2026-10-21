"""Placeholder module: ai/computer_vision/satellite_processing.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
File: backend/ai/computer_vision/satellite_processing.py

Geospatial satellite imagery preprocessing and analysis.

Supported workflows:
    - GeoTIFF raster loading
    - CRS and georeferencing validation
    - Sentinel-1 SAR preprocessing
    - Sentinel-2 optical band processing
    - Cloud and invalid-pixel masking
    - NDWI and MNDWI calculation
    - Water candidate extraction
    - Pre/post flood comparison
    - Valid-pixel flood area estimation
    - Raster reprojection and alignment
    - Raster tiling
    - GeoTIFF export
    - JSON report export
    - Human-review safeguards

Dependencies:
    pip install numpy rasterio

Notes:
    - SAR backscatter values must have known units.
    - Sentinel-2 bands must be radiometrically
      compatible and spatially aligned.
    - Cloud masking requires valid quality masks.
    - Water detection is not the same as flood detection.
    - Thresholds are illustrative, not calibrated.
    - All disaster-response decisions require
      verification by authorized personnel.
"""

from __future__ import annotations

import json
import math
import uuid

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import (
    reproject,
    transform_bounds,
)
from rasterio.windows import Window


# ============================================================
# EXCEPTIONS
# ============================================================

class SatelliteProcessingError(Exception):
    pass


class SatelliteValidationError(
    SatelliteProcessingError
):
    pass


class SatelliteAlignmentError(
    SatelliteProcessingError
):
    pass


class SatelliteExportError(
    SatelliteProcessingError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class SatelliteSensor(str, Enum):
    SENTINEL_1 = "sentinel_1"
    SENTINEL_2 = "sentinel_2"
    LANDSAT = "landsat"
    OTHER = "other"


class SatelliteProduct(str, Enum):
    SAR = "sar"
    OPTICAL = "optical"


class WaterDetectionMethod(str, Enum):
    NDWI = "ndwi"
    MNDWI = "mndwi"
    SAR_CHANGE = "sar_change"


class SatelliteAnalysisStatus(str, Enum):
    REVIEW_REQUIRED = "review_required"
    INCOMPLETE = "incomplete"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SatelliteProcessingConfig:
    max_pixels: int = 100_000_000
    max_bands: int = 32

    tile_size: int = 512
    tile_overlap: int = 32

    ndwi_threshold: float = 0.0
    mndwi_threshold: float = 0.0

    # Negative SAR change indicates a decrease
    # in post-event backscatter.
    sar_change_threshold_db: float = -3.0

    minimum_valid_fraction: float = 0.50

    maximum_output_features: int = 10000

    require_projected_crs_for_area: bool = True
    require_human_review: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_pixels",
            "max_bands",
            "tile_size",
            "maximum_output_features",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SatelliteValidationError(
                    f"{name} must be positive"
                )

        if not (
            0 <= self.tile_overlap < self.tile_size
        ):
            raise SatelliteValidationError(
                "tile_overlap must be smaller "
                "than tile_size"
            )

        if not (
            0 <= self.minimum_valid_fraction <= 1
        ):
            raise SatelliteValidationError(
                "minimum_valid_fraction must be in [0,1]"
            )

        for name in (
            "ndwi_threshold",
            "mndwi_threshold",
            "sar_change_threshold_db",
        ):
            if not math.isfinite(
                getattr(self, name)
            ):
                raise SatelliteValidationError(
                    f"{name} must be finite"
                )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class SatelliteRasterMetadata:
    width: int
    height: int
    band_count: int

    crs: str
    transform: tuple[float, ...]

    bounds: tuple[float, float, float, float]

    nodata: float | None
    dtype: str

    pixel_area_m2: float | None


@dataclass(frozen=True)
class SatelliteRaster:
    data: np.ndarray
    valid_mask: np.ndarray

    metadata: SatelliteRasterMetadata

    affine_transform: Affine


@dataclass(frozen=True)
class SatelliteTile:
    tile_id: str

    row_offset: int
    column_offset: int

    width: int
    height: int

    data: np.ndarray
    valid_mask: np.ndarray

    transform: Affine


@dataclass(frozen=True)
class SatelliteWaterResult:
    analysis_id: str

    method: WaterDetectionMethod

    width: int
    height: int

    valid_pixels: int
    water_pixels: int

    water_fraction_valid: float
    valid_fraction_total: float

    estimated_water_area_m2: float | None

    status: SatelliteAnalysisStatus
    requires_human_review: bool

    warnings: tuple[str, ...]

    analyzed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["method"] = self.method.value
        result["status"] = self.status.value

        result["analyzed_at"] = (
            self.analyzed_at.isoformat()
        )

        return result


@dataclass(frozen=True)
class SatelliteChangeResult:
    analysis_id: str

    method: WaterDetectionMethod

    newly_detected_water_pixels: int
    persistent_water_pixels: int
    water_loss_pixels: int

    valid_comparison_pixels: int

    newly_detected_water_area_m2: float | None

    status: SatelliteAnalysisStatus
    requires_human_review: bool

    warnings: tuple[str, ...]

    analyzed_at: datetime


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    numerator = np.asarray(
        numerator,
        dtype=np.float32,
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float32,
    )

    output = np.zeros_like(
        numerator,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > epsilon)
    )

    np.divide(
        numerator,
        denominator,
        out=output,
        where=valid,
    )

    return output


def affine_pixel_area(
    transform: Affine,
) -> float:
    return abs(
        transform.a * transform.e
        - transform.b * transform.d
    )


# ============================================================
# RASTER READER
# ============================================================

class SatelliteRasterReader:
    def __init__(
        self,
        config: SatelliteProcessingConfig,
    ) -> None:
        self.config = config

    def read(
        self,
        path: str | Path,
        *,
        bands: tuple[int, ...] | None = None,
    ) -> SatelliteRaster:
        source = Path(path)

        if not source.is_file():
            raise SatelliteValidationError(
                "Raster file not found"
            )

        try:
            with rasterio.open(source) as dataset:
                if dataset.crs is None:
                    raise SatelliteValidationError(
                        "Raster CRS is missing"
                    )

                if dataset.transform.is_identity:
                    raise SatelliteValidationError(
                        "Raster georeferencing is missing"
                    )

                if (
                    dataset.width * dataset.height
                    > self.config.max_pixels
                ):
                    raise SatelliteValidationError(
                        "Raster exceeds pixel limit"
                    )

                selected_bands = (
                    bands
                    if bands is not None
                    else tuple(
                        range(
                            1,
                            dataset.count + 1,
                        )
                    )
                )

                if (
                    not selected_bands
                    or len(selected_bands)
                    > self.config.max_bands
                ):
                    raise SatelliteValidationError(
                        "Invalid number of bands"
                    )

                if any(
                    band < 1
                    or band > dataset.count
                    for band in selected_bands
                ):
                    raise SatelliteValidationError(
                        "Band index out of range"
                    )

                masked = dataset.read(
                    indexes=list(selected_bands),
                    masked=True,
                    out_dtype="float32",
                )

                data = np.asarray(
                    masked.filled(np.nan),
                    dtype=np.float32,
                )

                valid_mask = (
                    ~np.ma.getmaskarray(masked)
                ).all(axis=0)

                valid_mask &= np.isfinite(
                    data
                ).all(axis=0)

                pixel_area = None

                if dataset.crs.is_projected:
                    # Only convert projected units
                    # when their conversion factor
                    # to meters is available.
                    try:
                        factors = [
                            float(axis.unit_conversion_factor)
                            for axis in dataset.crs.axis_info[:2]
                        ]

                        if (
                            len(factors) == 2
                            and all(
                                math.isfinite(factor)
                                and factor > 0
                                for factor in factors
                            )
                        ):
                            pixel_area = (
                                affine_pixel_area(
                                    dataset.transform
                                )
                                * factors[0]
                                * factors[1]
                            )
                    except (
                        AttributeError,
                        TypeError,
                        ValueError,
                    ):
                        pixel_area = None

                metadata = SatelliteRasterMetadata(
                    width=dataset.width,
                    height=dataset.height,
                    band_count=len(selected_bands),
                    crs=str(dataset.crs),
                    transform=tuple(
                        dataset.transform
                    ),
                    bounds=tuple(
                        dataset.bounds
                    ),
                    nodata=dataset.nodata,
                    dtype=str(data.dtype),
                    pixel_area_m2=pixel_area,
                )

                return SatelliteRaster(
                    data=data,
                    valid_mask=valid_mask,
                    metadata=metadata,
                    affine_transform=(
                        dataset.transform
                    ),
                )

        except rasterio.errors.RasterioError as exc:
            raise SatelliteValidationError(
                "Unable to read raster"
            ) from exc


# ============================================================
# RASTER VALIDATOR
# ============================================================

class SatelliteRasterValidator:
    @staticmethod
    def validate(
        raster: SatelliteRaster,
    ) -> None:
        if raster.data.ndim != 3:
            raise SatelliteValidationError(
                "Raster must have shape C,H,W"
            )

        channels, height, width = (
            raster.data.shape
        )

        if (
            channels <= 0
            or height <= 0
            or width <= 0
        ):
            raise SatelliteValidationError(
                "Invalid raster dimensions"
            )

        if raster.valid_mask.shape != (
            height,
            width,
        ):
            raise SatelliteValidationError(
                "Invalid valid-pixel mask"
            )

        if not np.isfinite(
            raster.data[:, raster.valid_mask]
        ).all():
            raise SatelliteValidationError(
                "Valid raster pixels contain "
                "non-finite values"
            )


# ============================================================
# RASTER ALIGNMENT
# ============================================================

class SatelliteRasterAligner:
    @staticmethod
    def validate_alignment(
        first: SatelliteRaster,
        second: SatelliteRaster,
    ) -> None:
        if first.data.shape[1:] != (
            second.data.shape[1:]
        ):
            raise SatelliteAlignmentError(
                "Raster dimensions differ"
            )

        if first.metadata.crs != second.metadata.crs:
            raise SatelliteAlignmentError(
                "Raster CRS differs"
            )

        if not np.allclose(
            tuple(first.affine_transform),
            tuple(second.affine_transform),
            rtol=0,
            atol=1e-9,
        ):
            raise SatelliteAlignmentError(
                "Raster transforms differ"
            )

    @staticmethod
    def reproject_to_reference(
        source: SatelliteRaster,
        reference: SatelliteRaster,
        *,
        resampling: Resampling = (
            Resampling.bilinear
        ),
    ) -> SatelliteRaster:
        source_data = source.data

        target_height = (
            reference.metadata.height
        )

        target_width = (
            reference.metadata.width
        )

        destination = np.full(
            (
                source_data.shape[0],
                target_height,
                target_width,
            ),
            np.nan,
            dtype=np.float32,
        )

        for band_index in range(
            source_data.shape[0]
        ):
            reproject(
                source=source_data[band_index],
                destination=destination[band_index],
                src_transform=source.affine_transform,
                src_crs=source.metadata.crs,
                dst_transform=reference.affine_transform,
                dst_crs=reference.metadata.crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=resampling,
            )

        destination_valid = np.zeros(
            (
                target_height,
                target_width,
            ),
            dtype=np.uint8,
        )

        reproject(
            source=source.valid_mask.astype(
                np.uint8
            ),
            destination=destination_valid,
            src_transform=source.affine_transform,
            src_crs=source.metadata.crs,
            dst_transform=reference.affine_transform,
            dst_crs=reference.metadata.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        valid_mask = (
            destination_valid > 0
        )

        valid_mask &= np.isfinite(
            destination
        ).all(axis=0)

        destination[
            :,
            ~valid_mask,
        ] = np.nan

        metadata = replace(
            source.metadata,
            width=target_width,
            height=target_height,
            crs=reference.metadata.crs,
            transform=tuple(
                reference.affine_transform
            ),
            bounds=reference.metadata.bounds,
            pixel_area_m2=(
                reference.metadata.pixel_area_m2
            ),
        )

        return SatelliteRaster(
            data=destination,
            valid_mask=valid_mask,
            metadata=metadata,
            affine_transform=(
                reference.affine_transform
            ),
        )


# ============================================================
# SPECTRAL INDICES
# ============================================================

class SatelliteSpectralIndices:
    @staticmethod
    def ndwi(
        green: np.ndarray,
        nir: np.ndarray,
    ) -> np.ndarray:
        """
        Normalized Difference Water Index.

        NDWI = (Green - NIR) / (Green + NIR)
        """
        return safe_divide(
            green - nir,
            green + nir,
        )

    @staticmethod
    def mndwi(
        green: np.ndarray,
        swir: np.ndarray,
    ) -> np.ndarray:
        """
        Modified Normalized Difference Water Index.

        MNDWI = (Green - SWIR) / (Green + SWIR)
        """
        return safe_divide(
            green - swir,
            green + swir,
        )

    @staticmethod
    def ndvi(
        red: np.ndarray,
        nir: np.ndarray,
    ) -> np.ndarray:
        """
        Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)
        """
        return safe_divide(
            nir - red,
            nir + red,
        )


# ============================================================
# SENTINEL-2 PROCESSOR
# ============================================================

class Sentinel2Processor:
    """
    Expects already aligned optical bands.

    Sentinel-2:
        B03 = Green
        B04 = Red
        B08 = NIR
        B11 = SWIR1

    B11 is normally 20 m resolution and
    must be aligned to the chosen grid.
    """

    @staticmethod
    def calculate_indices(
        *,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir: np.ndarray,
        valid_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        shapes = {
            green.shape,
            red.shape,
            nir.shape,
            swir.shape,
            valid_mask.shape,
        }

        if len(shapes) != 1:
            raise SatelliteAlignmentError(
                "Optical band shapes differ"
            )

        valid = (
            valid_mask.astype(bool)
            & np.isfinite(green)
            & np.isfinite(red)
            & np.isfinite(nir)
            & np.isfinite(swir)
        )

        ndwi = SatelliteSpectralIndices.ndwi(
            green,
            nir,
        )

        mndwi = SatelliteSpectralIndices.mndwi(
            green,
            swir,
        )

        ndvi = SatelliteSpectralIndices.ndvi(
            red,
            nir,
        )

        for index in (
            ndwi,
            mndwi,
            ndvi,
        ):
            index[~valid] = np.nan

        return {
            "ndwi": ndwi,
            "mndwi": mndwi,
            "ndvi": ndvi,
            "valid_mask": valid,
        }


# ============================================================
# SENTINEL-1 SAR PROCESSOR
# ============================================================

class Sentinel1SARProcessor:
    @staticmethod
    def linear_to_db(
        backscatter_linear: np.ndarray,
        *,
        epsilon: float = 1e-10,
    ) -> np.ndarray:
        values = np.asarray(
            backscatter_linear,
            dtype=np.float32,
        )

        result = np.full_like(
            values,
            np.nan,
        )

        valid = (
            np.isfinite(values)
            & (values > 0)
        )

        result[valid] = (
            10.0
            * np.log10(
                np.maximum(
                    values[valid],
                    epsilon,
                )
            )
        )

        return result

    @staticmethod
    def calculate_change_db(
        before_db: np.ndarray,
        after_db: np.ndarray,
    ) -> np.ndarray:
        if before_db.shape != after_db.shape:
            raise SatelliteAlignmentError(
                "SAR image shapes differ"
            )

        before = np.asarray(
            before_db,
            dtype=np.float32,
        )

        after = np.asarray(
            after_db,
            dtype=np.float32,
        )

        change = after - before

        change[
            ~(
                np.isfinite(before)
                & np.isfinite(after)
            )
        ] = np.nan

        return change


# ============================================================
# CLOUD MASK PROCESSOR
# ============================================================

class SatelliteCloudMaskProcessor:
    @staticmethod
    def sentinel2_scl_valid_mask(
        scl: np.ndarray,
        *,
        exclude_classes: tuple[int, ...] = (
            0,   # No data
            1,   # Saturated / defective
            3,   # Cloud shadows
            8,   # Medium cloud probability
            9,   # High cloud probability
            10,  # Thin cirrus
            11,  # Snow / ice
        ),
    ) -> np.ndarray:
        """
        Returns True for pixels not in the
        excluded Sentinel-2 SCL categories.

        SCL must already be aligned with
        the imagery.
        """
        if scl.ndim != 2:
            raise SatelliteValidationError(
                "SCL must be a 2D array"
            )

        return (
            np.isfinite(scl)
            & ~np.isin(
                scl,
                exclude_classes,
            )
        )


# ============================================================
# WATER MASK DETECTOR
# ============================================================

class SatelliteWaterMaskDetector:
    def __init__(
        self,
        config: SatelliteProcessingConfig,
    ) -> None:
        self.config = config

    def from_ndwi(
        self,
        ndwi: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        return (
            valid_mask.astype(bool)
            & np.isfinite(ndwi)
            & (
                ndwi
                > self.config.ndwi_threshold
            )
        )

    def from_mndwi(
        self,
        mndwi: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        return (
            valid_mask.astype(bool)
            & np.isfinite(mndwi)
            & (
                mndwi
                > self.config.mndwi_threshold
            )
        )

    def from_sar_change(
        self,
        change_db: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        SAR change is only a candidate
        screening signal.

        Decreased backscatter can have
        multiple causes besides flooding.
        """
        return (
            valid_mask.astype(bool)
            & np.isfinite(change_db)
            & (
                change_db
                <= self.config.sar_change_threshold_db
            )
        )


# ============================================================
# FLOOD CHANGE DETECTOR
# ============================================================

class SatelliteFloodChangeDetector:
    @staticmethod
    def compare(
        before_water: np.ndarray,
        after_water: np.ndarray,
        *,
        before_valid: np.ndarray,
        after_valid: np.ndarray,
    ) -> dict[str, np.ndarray]:
        shapes = {
            before_water.shape,
            after_water.shape,
            before_valid.shape,
            after_valid.shape,
        }

        if len(shapes) != 1:
            raise SatelliteAlignmentError(
                "Before and after masks differ"
            )

        valid = (
            before_valid.astype(bool)
            & after_valid.astype(bool)
        )

        before = (
            before_water.astype(bool)
            & valid
        )

        after = (
            after_water.astype(bool)
            & valid
        )

        newly_detected = (
            ~before
            & after
            & valid
        )

        persistent = (
            before
            & after
            & valid
        )

        water_loss = (
            before
            & ~after
            & valid
        )

        return {
            "newly_detected_water": newly_detected,
            "persistent_water": persistent,
            "water_loss": water_loss,
            "valid_mask": valid,
        }


# ============================================================
# AREA ESTIMATION
# ============================================================

class SatelliteAreaEstimator:
    @staticmethod
    def estimate_area_m2(
        mask: np.ndarray,
        *,
        pixel_area_m2: float | None,
    ) -> float | None:
        if pixel_area_m2 is None:
            return None

        if (
            not math.isfinite(pixel_area_m2)
            or pixel_area_m2 <= 0
        ):
            raise SatelliteValidationError(
                "Invalid pixel area"
            )

        return (
            int(np.count_nonzero(mask))
            * pixel_area_m2
        )


# ============================================================
# RASTER TILING
# ============================================================

class SatelliteRasterTiler:
    def __init__(
        self,
        config: SatelliteProcessingConfig,
    ) -> None:
        self.config = config

    def iter_tiles(
        self,
        raster: SatelliteRaster,
    ) -> Iterator[SatelliteTile]:
        SatelliteRasterValidator.validate(
            raster
        )

        height = raster.metadata.height
        width = raster.metadata.width

        step = (
            self.config.tile_size
            - self.config.tile_overlap
        )

        for row in range(
            0,
            height,
            step,
        ):
            for col in range(
                0,
                width,
                step,
            ):
                tile_height = min(
                    self.config.tile_size,
                    height - row,
                )

                tile_width = min(
                    self.config.tile_size,
                    width - col,
                )

                data = raster.data[
                    :,
                    row:row + tile_height,
                    col:col + tile_width,
                ].copy()

                valid = raster.valid_mask[
                    row:row + tile_height,
                    col:col + tile_width,
                ].copy()

                transform = (
                    raster.affine_transform
                    * Affine.translation(
                        col,
                        row,
                    )
                )

                yield SatelliteTile(
                    tile_id=generate_id(),
                    row_offset=row,
                    column_offset=col,
                    width=tile_width,
                    height=tile_height,
                    data=data,
                    valid_mask=valid,
                    transform=transform,
                )


# ============================================================
# ANALYSIS SERVICE
# ============================================================

class SatelliteProcessingService:
    def __init__(
        self,
        config: SatelliteProcessingConfig | None = None,
    ) -> None:
        self.config = (
            config or SatelliteProcessingConfig()
        )

        self.reader = SatelliteRasterReader(
            self.config
        )

        self.aligner = SatelliteRasterAligner()

        self.mask_detector = (
            SatelliteWaterMaskDetector(
                self.config
            )
        )

        self.change_detector = (
            SatelliteFloodChangeDetector()
        )

        self.area_estimator = (
            SatelliteAreaEstimator()
        )

        self.tiler = SatelliteRasterTiler(
            self.config
        )

    def analyze_water_index(
        self,
        index: np.ndarray,
        valid_mask: np.ndarray,
        *,
        method: WaterDetectionMethod,
        pixel_area_m2: float | None = None,
    ) -> tuple[
        SatelliteWaterResult,
        np.ndarray,
    ]:
        if index.shape != valid_mask.shape:
            raise SatelliteAlignmentError(
                "Index and validity mask differ"
            )

        valid = (
            valid_mask.astype(bool)
            & np.isfinite(index)
        )

        if method == WaterDetectionMethod.NDWI:
            water_mask = (
                self.mask_detector.from_ndwi(
                    index,
                    valid,
                )
            )

        elif method == WaterDetectionMethod.MNDWI:
            water_mask = (
                self.mask_detector.from_mndwi(
                    index,
                    valid,
                )
            )

        elif method == WaterDetectionMethod.SAR_CHANGE:
            water_mask = (
                self.mask_detector.from_sar_change(
                    index,
                    valid,
                )
            )

        else:
            raise SatelliteValidationError(
                "Unsupported detection method"
            )

        total_pixels = int(
            index.size
        )

        valid_pixels = int(
            valid.sum()
        )

        water_pixels = int(
            water_mask.sum()
        )

        valid_fraction = (
            valid_pixels / total_pixels
            if total_pixels
            else 0.0
        )

        water_fraction = (
            water_pixels / valid_pixels
            if valid_pixels
            else 0.0
        )

        area = self.area_estimator.estimate_area_m2(
            water_mask,
            pixel_area_m2=pixel_area_m2,
        )

        warnings = [
            "Water classification is a screening result",
            "Permanent water may be included",
            "Thresholds require local calibration",
            "Human verification required",
        ]

        if valid_fraction < (
            self.config.minimum_valid_fraction
        ):
            warnings.append(
                "Low valid-pixel coverage"
            )

        status = (
            SatelliteAnalysisStatus.REVIEW_REQUIRED
            if (
                valid_pixels > 0
                and valid_fraction
                >= self.config.minimum_valid_fraction
            )
            else SatelliteAnalysisStatus.INCOMPLETE
        )

        result = SatelliteWaterResult(
            analysis_id=generate_id(),
            method=method,
            width=index.shape[1],
            height=index.shape[0],
            valid_pixels=valid_pixels,
            water_pixels=water_pixels,
            water_fraction_valid=water_fraction,
            valid_fraction_total=valid_fraction,
            estimated_water_area_m2=area,
            status=status,
            requires_human_review=True,
            warnings=tuple(warnings),
            analyzed_at=utc_now(),
        )

        return result, water_mask

    def analyze_flood_change(
        self,
        before_water: np.ndarray,
        after_water: np.ndarray,
        *,
        before_valid: np.ndarray,
        after_valid: np.ndarray,
        method: WaterDetectionMethod,
        pixel_area_m2: float | None = None,
    ) -> tuple[
        SatelliteChangeResult,
        dict[str, np.ndarray],
    ]:
        masks = self.change_detector.compare(
            before_water,
            after_water,
            before_valid=before_valid,
            after_valid=after_valid,
        )

        newly_detected = masks[
            "newly_detected_water"
        ]

        area = self.area_estimator.estimate_area_m2(
            newly_detected,
            pixel_area_m2=pixel_area_m2,
        )

        valid_count = int(
            masks["valid_mask"].sum()
        )

        result = SatelliteChangeResult(
            analysis_id=generate_id(),
            method=method,
            newly_detected_water_pixels=int(
                newly_detected.sum()
            ),
            persistent_water_pixels=int(
                masks["persistent_water"].sum()
            ),
            water_loss_pixels=int(
                masks["water_loss"].sum()
            ),
            valid_comparison_pixels=valid_count,
            newly_detected_water_area_m2=area,
            status=(
                SatelliteAnalysisStatus.REVIEW_REQUIRED
                if valid_count > 0
                else SatelliteAnalysisStatus.INCOMPLETE
            ),
            requires_human_review=True,
            warnings=(
                "Newly detected water is not a verified flood extent",
                "Registration errors can produce false changes",
                "Clouds, shadows, vegetation and SAR effects "
                "can change water classifications",
                "Human verification required",
            ),
            analyzed_at=utc_now(),
        )

        return result, masks


# ============================================================
# GEOTIFF EXPORTER
# ============================================================

class SatelliteGeoTIFFExporter:
    @staticmethod
    def export_mask(
        mask: np.ndarray,
        path: str | Path,
        *,
        transform: Affine,
        crs: str,
    ) -> Path:
        if mask.ndim != 2:
            raise SatelliteExportError(
                "Mask must be two-dimensional"
            )

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise SatelliteExportError(
                "Output already exists"
            )

        try:
            with rasterio.open(
                destination,
                "w",
                driver="GTiff",
                width=mask.shape[1],
                height=mask.shape[0],
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
                compress="deflate",
            ) as dataset:
                dataset.write(
                    mask.astype(
                        np.uint8
                    ),
                    1,
                )

        except rasterio.errors.RasterioError as exc:
            raise SatelliteExportError(
                "Unable to export GeoTIFF"
            ) from exc

        return destination


# ============================================================
# JSON EXPORTER
# ============================================================

class SatelliteReportExporter:
    @staticmethod
    def export_json(
        result: SatelliteWaterResult,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise SatelliteExportError(
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


# ============================================================
# FACTORY
# ============================================================

def create_satellite_processing_service(
    config: SatelliteProcessingConfig | None = None,
) -> SatelliteProcessingService:
    return SatelliteProcessingService(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    service = create_satellite_processing_service(
        SatelliteProcessingConfig(
            ndwi_threshold=0.0,
            mndwi_threshold=0.0,
            tile_size=512,
            tile_overlap=32,
        )
    )

    # Example input must contain aligned:
    # band 1: Sentinel-2 Green (B03)
    # band 2: Sentinel-2 Red   (B04)
    # band 3: Sentinel-2 NIR   (B08)
    # band 4: Sentinel-2 SWIR  (B11)
    #
    # Cloud/quality masking should be applied
    # before operational interpretation.

    raster = service.reader.read(
        "data/satellite/sentinel2_aligned.tif",
        bands=(1, 2, 3, 4),
    )

    green = raster.data[0]
    red = raster.data[1]
    nir = raster.data[2]
    swir = raster.data[3]

    indices = Sentinel2Processor.calculate_indices(
        green=green,
        red=red,
        nir=nir,
        swir=swir,
        valid_mask=raster.valid_mask,
    )

    result, water_mask = (
        service.analyze_water_index(
            indices["mndwi"],
            indices["valid_mask"],
            method=WaterDetectionMethod.MNDWI,
            pixel_area_m2=(
                raster.metadata.pixel_area_m2
            ),
        )
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    SatelliteGeoTIFFExporter.export_mask(
        water_mask,
        "outputs/satellite/water_mask.tif",
        transform=raster.affine_transform,
        crs=raster.metadata.crs,
    )

    SatelliteReportExporter.export_json(
        result,
        "outputs/satellite/water_report.json",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SatelliteProcessingError",
    "SatelliteValidationError",
    "SatelliteAlignmentError",
    "SatelliteExportError",
    "SatelliteSensor",
    "SatelliteProduct",
    "WaterDetectionMethod",
    "SatelliteAnalysisStatus",
    "SatelliteProcessingConfig",
    "SatelliteRasterMetadata",
    "SatelliteRaster",
    "SatelliteTile",
    "SatelliteWaterResult",
    "SatelliteChangeResult",
    "SatelliteRasterReader",
    "SatelliteRasterValidator",
    "SatelliteRasterAligner",
    "SatelliteSpectralIndices",
    "Sentinel2Processor",
    "Sentinel1SARProcessor",
    "SatelliteCloudMaskProcessor",
    "SatelliteWaterMaskDetector",
    "SatelliteFloodChangeDetector",
    "SatelliteAreaEstimator",
    "SatelliteRasterTiler",
    "SatelliteProcessingService",
    "SatelliteGeoTIFFExporter",
    "SatelliteReportExporter",
    "create_satellite_processing_service",
]
