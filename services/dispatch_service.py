from __future__ import annotations

"""
backend/services/damage_service.py

Kerala Flood Relief Coordination Agent

Flood Damage Assessment Service

Features:
- Building and infrastructure damage assessment
- Residential and commercial property damage
- Road and bridge damage
- Agricultural and crop loss assessment
- Vehicle and equipment damage
- Public utility damage
- Flood depth and exposure analysis
- Damage severity classification
- Financial loss estimation
- Repair and replacement cost calculation
- AI/CV damage assessment integration
- Damage report generation
- Human verification and approval
- District-level damage analytics
- Audit logging

IMPORTANT:
All financial estimates are preliminary.
Official compensation, insurance decisions,
and government damage certification require
authorized human verification.
"""


import logging
import math
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)

MONEY_QUANTUM = Decimal("0.01")
MAX_AMOUNT = Decimal("1000000000000")


# ============================================================
# ENUMERATIONS
# ============================================================

class DamageCategory(str, Enum):
    RESIDENTIAL_BUILDING = "residential_building"
    COMMERCIAL_BUILDING = "commercial_building"
    GOVERNMENT_BUILDING = "government_building"
    ROAD = "road"
    BRIDGE = "bridge"
    AGRICULTURE = "agriculture"
    CROP = "crop"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    ELECTRICITY = "electricity"
    WATER_SUPPLY = "water_supply"
    TELECOMMUNICATION = "telecommunication"
    HEALTHCARE_FACILITY = "healthcare_facility"
    EDUCATIONAL_FACILITY = "educational_facility"
    OTHER = "other"


class DamageSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"
    DESTROYED = "destroyed"


class AssessmentStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AssessmentSource(str, Enum):
    MANUAL = "manual"
    FIELD_SURVEY = "field_survey"
    DRONE = "drone"
    SATELLITE = "satellite"
    IMAGE_AI = "image_ai"
    SENSOR = "sensor"
    HYBRID = "hybrid"


# ============================================================
# EXCEPTIONS
# ============================================================

class DamageServiceError(Exception):
    pass


class DamageValidationError(DamageServiceError):
    pass


class DamageNotFoundError(DamageServiceError):
    pass


class DamageConflictError(DamageServiceError):
    pass


class DamageAuthorizationError(DamageServiceError):
    pass


class DamageIntegrationError(DamageServiceError):
    pass


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_decimal(
    value: Decimal | int | float | str,
) -> Decimal:
    try:
        amount = Decimal(str(value))
    except Exception as exc:
        raise DamageValidationError(
            "Invalid monetary amount"
        ) from exc

    if not amount.is_finite():
        raise DamageValidationError(
            "Monetary amount must be finite"
        )

    if amount < 0 or amount > MAX_AMOUNT:
        raise DamageValidationError(
            "Monetary amount outside allowed range"
        )

    return amount


def round_money(
    value: Decimal | int | float | str,
) -> Decimal:
    return to_decimal(value).quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def validate_ratio(
    value: float,
    name: str,
) -> float:
    if not math.isfinite(value):
        raise DamageValidationError(
            f"{name} must be finite"
        )

    if not 0 <= value <= 1:
        raise DamageValidationError(
            f"{name} must be between 0 and 1"
        )

    return value


def validate_aware_datetime(
    value: datetime | None,
    name: str,
) -> None:
    if value is not None and (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise DamageValidationError(
            f"{name} must be timezone-aware"
        )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class DamageCostItem:
    description: str
    quantity: Decimal
    unit_cost: Decimal
    unit: str = "unit"

    @property
    def total(self) -> Decimal:
        return round_money(
            self.quantity * self.unit_cost
        )


@dataclass(frozen=True)
class DamageAssessmentRequest:
    district_id: int
    asset_reference: str
    category: DamageCategory
    source: AssessmentSource

    description: str

    estimated_asset_value: Decimal

    damage_ratio: float | None = None
    flood_depth_m: float | None = None
    exposure_hours: float | None = None

    affected_area_sqm: float | None = None
    affected_area_hectares: float | None = None

    repair_items: tuple[DamageCostItem, ...] = ()

    replacement_cost: Decimal | None = None
    debris_removal_cost: Decimal = Decimal("0")
    cleanup_cost: Decimal = Decimal("0")
    downtime_cost: Decimal = Decimal("0")

    image_references: tuple[str, ...] = ()
    evidence_references: tuple[str, ...] = ()

    confidence: float | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class DamageAssessment:
    id: str
    district_id: int
    asset_reference: str

    category: DamageCategory
    source: AssessmentSource

    description: str
    severity: DamageSeverity

    estimated_asset_value: Decimal
    damage_ratio: float

    repair_cost: Decimal
    replacement_cost: Decimal
    cleanup_cost: Decimal
    debris_removal_cost: Decimal
    downtime_cost: Decimal

    estimated_total_loss: Decimal

    status: AssessmentStatus = AssessmentStatus.DRAFT

    flood_depth_m: float | None = None
    exposure_hours: float | None = None

    affected_area_sqm: float | None = None
    affected_area_hectares: float | None = None

    confidence: float | None = None

    image_references: list[str] = field(
        default_factory=list
    )

    evidence_references: list[str] = field(
        default_factory=list
    )

    repair_items: list[DamageCostItem] = field(
        default_factory=list
    )

    created_by: int | None = None
    reviewed_by: int | None = None

    review_notes: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DamageEstimate:
    severity: DamageSeverity
    damage_ratio: float
    repair_cost: Decimal
    replacement_cost: Decimal
    direct_loss: Decimal
    indirect_loss: Decimal
    total_loss: Decimal
    currency: str = "INR"
    preliminary: bool = True


@dataclass(frozen=True)
class VisionDamageResult:
    damage_ratio: float
    confidence: float
    detected_damage: tuple[str, ...] = ()
    model_name: str = ""
    model_version: str = ""


# ============================================================
# REPOSITORY CONTRACT
# ============================================================

class DamageRepository(Protocol):

    async def create(
        self,
        assessment: DamageAssessment,
    ) -> DamageAssessment:
        ...

    async def get_by_id(
        self,
        assessment_id: str,
    ) -> DamageAssessment | None:
        ...

    async def update(
        self,
        assessment: DamageAssessment,
    ) -> DamageAssessment:
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[DamageAssessment]:
        ...

    async def list_all(
        self,
    ) -> Sequence[DamageAssessment]:
        ...


# ============================================================
# AI VISION CONTRACT
# ============================================================

class DamageVisionGateway(Protocol):

    async def analyze(
        self,
        *,
        image_references: Sequence[str],
        category: DamageCategory,
    ) -> VisionDamageResult:
        ...


# ============================================================
# AUDIT CONTRACT
# ============================================================

class DamageAuditGateway(Protocol):

    async def record(
        self,
        action: str,
        *,
        actor_id: int | None,
        entity_type: str,
        entity_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...


# ============================================================
# DAMAGE SEVERITY CLASSIFIER
# ============================================================

class DamageSeverityClassifier:
    """
    Illustrative damage-ratio bands.

    These bands are configurable examples and
    are not official government certification rules.
    """

    @staticmethod
    def classify(
        damage_ratio: float,
    ) -> DamageSeverity:
        validate_ratio(
            damage_ratio,
            "damage_ratio",
        )

        if damage_ratio == 0:
            return DamageSeverity.NONE

        if damage_ratio < 0.10:
            return DamageSeverity.MINOR

        if damage_ratio < 0.30:
            return DamageSeverity.MODERATE

        if damage_ratio < 0.60:
            return DamageSeverity.MAJOR

        if damage_ratio < 0.90:
            return DamageSeverity.SEVERE

        return DamageSeverity.DESTROYED


# ============================================================
# FLOOD EXPOSURE ASSESSMENT
# ============================================================

class FloodExposureAnalyzer:
    """
    Screening-level flood exposure indicator.

    The result is NOT a calibrated engineering
    damage function or structural safety rating.
    """

    CATEGORY_SENSITIVITY = {
        DamageCategory.RESIDENTIAL_BUILDING: 0.70,
        DamageCategory.COMMERCIAL_BUILDING: 0.75,
        DamageCategory.GOVERNMENT_BUILDING: 0.65,
        DamageCategory.ROAD: 0.60,
        DamageCategory.BRIDGE: 0.80,
        DamageCategory.AGRICULTURE: 0.85,
        DamageCategory.CROP: 0.95,
        DamageCategory.VEHICLE: 0.80,
        DamageCategory.EQUIPMENT: 0.85,
        DamageCategory.ELECTRICITY: 0.90,
        DamageCategory.WATER_SUPPLY: 0.80,
        DamageCategory.TELECOMMUNICATION: 0.85,
        DamageCategory.HEALTHCARE_FACILITY: 0.80,
        DamageCategory.EDUCATIONAL_FACILITY: 0.70,
        DamageCategory.OTHER: 0.70,
    }

    def estimate_screening_ratio(
        self,
        *,
        category: DamageCategory,
        flood_depth_m: float,
        exposure_hours: float,
    ) -> float:
        if not math.isfinite(flood_depth_m):
            raise DamageValidationError(
                "Invalid flood depth"
            )

        if not math.isfinite(exposure_hours):
            raise DamageValidationError(
                "Invalid exposure duration"
            )

        if flood_depth_m < 0:
            raise DamageValidationError(
                "Flood depth cannot be negative"
            )

        if exposure_hours < 0:
            raise DamageValidationError(
                "Exposure duration cannot be negative"
            )

        if flood_depth_m == 0 or exposure_hours == 0:
            return 0.0

        depth_factor = min(
            flood_depth_m / 3.0,
            1.0,
        )

        duration_factor = min(
            exposure_hours / 72.0,
            1.0,
        )

        sensitivity = self.CATEGORY_SENSITIVITY[
            category
        ]

        ratio = (
            depth_factor * 0.65
            + duration_factor * 0.35
        ) * sensitivity

        return round(
            min(max(ratio, 0.0), 1.0),
            4,
        )


# ============================================================
# FINANCIAL DAMAGE ESTIMATION
# ============================================================

class DamageCostEstimator:
    """
    Estimate financial losses in INR.

    No fabricated market prices are used.
    Costs must come from supplied estimates,
    quotations, or validated pricing sources.
    """

    def __init__(self):
        self.classifier = DamageSeverityClassifier()

    def estimate(
        self,
        *,
        asset_value: Decimal,
        damage_ratio: float,
        repair_items: Sequence[DamageCostItem] = (),
        replacement_cost: Decimal | None = None,
        cleanup_cost: Decimal = Decimal("0"),
        debris_removal_cost: Decimal = Decimal("0"),
        downtime_cost: Decimal = Decimal("0"),
    ) -> DamageEstimate:

        validate_ratio(
            damage_ratio,
            "damage_ratio",
        )

        asset_value = round_money(
            asset_value
        )

        cleanup_cost = round_money(
            cleanup_cost
        )

        debris_removal_cost = round_money(
            debris_removal_cost
        )

        downtime_cost = round_money(
            downtime_cost
        )

        for item in repair_items:
            if not item.description.strip():
                raise DamageValidationError(
                    "Repair item description is required"
                )

            to_decimal(item.quantity)
            to_decimal(item.unit_cost)

        repair_cost = round_money(
            sum(
                (
                    item.total
                    for item in repair_items
                ),
                Decimal("0"),
            )
        )

        if replacement_cost is None:
            replacement = asset_value
        else:
            replacement = round_money(
                replacement_cost
            )

        severity = self.classifier.classify(
            damage_ratio
        )

        depreciated_damage = round_money(
            asset_value
            * Decimal(str(damage_ratio))
        )

        if severity == DamageSeverity.NONE:
            direct_loss = Decimal("0.00")

        elif repair_items:
            direct_loss = min(
                repair_cost,
                replacement,
            )

        else:
            direct_loss = min(
                depreciated_damage,
                replacement,
            )

        indirect_loss = round_money(
            cleanup_cost
            + debris_removal_cost
            + downtime_cost
        )

        total_loss = round_money(
            direct_loss + indirect_loss
        )

        return DamageEstimate(
            severity=severity,
            damage_ratio=damage_ratio,
            repair_cost=repair_cost,
            replacement_cost=replacement,
            direct_loss=direct_loss,
            indirect_loss=indirect_loss,
            total_loss=total_loss,
        )


# ============================================================
# DAMAGE ASSESSMENT SERVICE
# ============================================================

class DamageService:

    def __init__(
        self,
        repository: DamageRepository,
        *,
        vision_gateway: DamageVisionGateway | None = None,
        audit_gateway: DamageAuditGateway | None = None,
        estimator: DamageCostEstimator | None = None,
        exposure_analyzer: FloodExposureAnalyzer | None = None,
    ):
        self.repository = repository
        self.vision_gateway = vision_gateway
        self.audit_gateway = audit_gateway

        self.estimator = (
            estimator
            or DamageCostEstimator()
        )

        self.exposure_analyzer = (
            exposure_analyzer
            or FloodExposureAnalyzer()
        )

    async def _audit(
        self,
        action: str,
        assessment: DamageAssessment,
        actor_id: int | None,
    ) -> None:
        if self.audit_gateway is None:
            return

        await self.audit_gateway.record(
            action,
            actor_id=actor_id,
            entity_type="damage_assessment",
            entity_id=assessment.id,
            metadata={
                "district_id": assessment.district_id,
                "category": assessment.category.value,
                "status": assessment.status.value,
                "severity": assessment.severity.value,
            },
        )

    async def get_assessment(
        self,
        assessment_id: str,
    ) -> DamageAssessment:

        assessment = await self.repository.get_by_id(
            assessment_id
        )

        if assessment is None:
            raise DamageNotFoundError(
                "Damage assessment not found"
            )

        return assessment

    def _validate_request(
        self,
        request: DamageAssessmentRequest,
    ) -> None:

        if request.district_id <= 0:
            raise DamageValidationError(
                "Invalid district ID"
            )

        if not request.asset_reference.strip():
            raise DamageValidationError(
                "Asset reference is required"
            )

        if not request.description.strip():
            raise DamageValidationError(
                "Damage description is required"
            )

        to_decimal(
            request.estimated_asset_value
        )

        if request.damage_ratio is not None:
            validate_ratio(
                request.damage_ratio,
                "damage_ratio",
            )

        if request.confidence is not None:
            validate_ratio(
                request.confidence,
                "confidence",
            )

        for name, value in (
            ("flood_depth_m", request.flood_depth_m),
            ("exposure_hours", request.exposure_hours),
            ("affected_area_sqm", request.affected_area_sqm),
            (
                "affected_area_hectares",
                request.affected_area_hectares,
            ),
        ):
            if value is not None:
                if not math.isfinite(value) or value < 0:
                    raise DamageValidationError(
                        f"{name} must be finite and nonnegative"
                    )

    async def create_assessment(
        self,
        request: DamageAssessmentRequest,
        *,
        actor_id: int,
    ) -> DamageAssessment:

        self._validate_request(request)

        if request.damage_ratio is not None:
            damage_ratio = request.damage_ratio

        elif (
            request.flood_depth_m is not None
            and request.exposure_hours is not None
        ):
            damage_ratio = (
                self.exposure_analyzer
                .estimate_screening_ratio(
                    category=request.category,
                    flood_depth_m=request.flood_depth_m,
                    exposure_hours=request.exposure_hours,
                )
            )

        else:
            raise DamageValidationError(
                "Damage ratio or flood exposure "
                "measurements are required"
            )

        estimate = self.estimator.estimate(
            asset_value=request.estimated_asset_value,
            damage_ratio=damage_ratio,
            repair_items=request.repair_items,
            replacement_cost=request.replacement_cost,
            cleanup_cost=request.cleanup_cost,
            debris_removal_cost=request.debris_removal_cost,
            downtime_cost=request.downtime_cost,
        )

        assessment = DamageAssessment(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            asset_reference=request.asset_reference.strip(),
            category=request.category,
            source=request.source,
            description=request.description.strip(),
            severity=estimate.severity,
            estimated_asset_value=round_money(
                request.estimated_asset_value
            ),
            damage_ratio=estimate.damage_ratio,
            repair_cost=estimate.repair_cost,
            replacement_cost=estimate.replacement_cost,
            cleanup_cost=round_money(
                request.cleanup_cost
            ),
            debris_removal_cost=round_money(
                request.debris_removal_cost
            ),
            downtime_cost=round_money(
                request.downtime_cost
            ),
            estimated_total_loss=estimate.total_loss,
            flood_depth_m=request.flood_depth_m,
            exposure_hours=request.exposure_hours,
            affected_area_sqm=request.affected_area_sqm,
            affected_area_hectares=(
                request.affected_area_hectares
            ),
            confidence=request.confidence,
            image_references=list(
                request.image_references
            ),
            evidence_references=list(
                request.evidence_references
            ),
            repair_items=list(
                request.repair_items
            ),
            created_by=actor_id,
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            assessment
        )

        await self._audit(
            "damage.assessment_created",
            created,
            actor_id,
        )

        logger.info(
            "Damage assessment created: %s",
            created.id,
        )

        return created

    async def assess_images(
        self,
        *,
        image_references: Sequence[str],
        category: DamageCategory,
    ) -> VisionDamageResult:

        if self.vision_gateway is None:
            raise DamageIntegrationError(
                "Damage vision model is not configured"
            )

        if not image_references:
            raise DamageValidationError(
                "At least one image is required"
            )

        result = await self.vision_gateway.analyze(
            image_references=image_references,
            category=category,
        )

        validate_ratio(
            result.damage_ratio,
            "AI damage ratio",
        )

        validate_ratio(
            result.confidence,
            "AI confidence",
        )

        return result

    async def submit_for_review(
        self,
        assessment_id: str,
        *,
        actor_id: int,
    ) -> DamageAssessment:

        assessment = await self.get_assessment(
            assessment_id
        )

        if assessment.status != AssessmentStatus.DRAFT:
            raise DamageConflictError(
                "Only draft assessments can be submitted"
            )

        assessment.status = (
            AssessmentStatus.PENDING_REVIEW
        )

        assessment.updated_at = utc_now()

        updated = await self.repository.update(
            assessment
        )

        await self._audit(
            "damage.assessment_submitted",
            updated,
            actor_id,
        )

        return updated

    async def verify_assessment(
        self,
        assessment_id: str,
        *,
        reviewer_id: int,
        authorized: bool,
        review_notes: str,
    ) -> DamageAssessment:

        if not authorized:
            raise DamageAuthorizationError(
                "Damage verification permission required"
            )

        assessment = await self.get_assessment(
            assessment_id
        )

        if (
            assessment.status
            != AssessmentStatus.PENDING_REVIEW
        ):
            raise DamageConflictError(
                "Assessment is not pending review"
            )

        if assessment.created_by == reviewer_id:
            raise DamageAuthorizationError(
                "Independent reviewer required"
            )

        if not review_notes.strip():
            raise DamageValidationError(
                "Review notes are required"
            )

        assessment.status = AssessmentStatus.VERIFIED
        assessment.reviewed_by = reviewer_id
        assessment.review_notes = review_notes.strip()
        assessment.reviewed_at = utc_now()
        assessment.updated_at = utc_now()

        updated = await self.repository.update(
            assessment
        )

        await self._audit(
            "damage.assessment_verified",
            updated,
            reviewer_id,
        )

        return updated

    async def reject_assessment(
        self,
        assessment_id: str,
        *,
        reviewer_id: int,
        authorized: bool,
        reason: str,
    ) -> DamageAssessment:

        if not authorized:
            raise DamageAuthorizationError(
                "Damage review permission required"
            )

        if not reason.strip():
            raise DamageValidationError(
                "Rejection reason is required"
            )

        assessment = await self.get_assessment(
            assessment_id
        )

        if (
            assessment.status
            != AssessmentStatus.PENDING_REVIEW
        ):
            raise DamageConflictError(
                "Assessment is not pending review"
            )

        assessment.status = AssessmentStatus.REJECTED
        assessment.reviewed_by = reviewer_id
        assessment.review_notes = reason.strip()
        assessment.reviewed_at = utc_now()
        assessment.updated_at = utc_now()

        updated = await self.repository.update(
            assessment
        )

        await self._audit(
            "damage.assessment_rejected",
            updated,
            reviewer_id,
        )

        return updated

    async def list_district_assessments(
        self,
        district_id: int,
    ) -> Sequence[DamageAssessment]:

        if district_id <= 0:
            raise DamageValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
        *,
        verified_only: bool = True,
    ) -> dict[str, Any]:

        assessments = list(
            await self.list_district_assessments(
                district_id
            )
        )

        included = [
            assessment
            for assessment in assessments
            if (
                not verified_only
                or assessment.status
                == AssessmentStatus.VERIFIED
            )
        ]

        total_loss = sum(
            (
                assessment.estimated_total_loss
                for assessment in included
            ),
            Decimal("0"),
        )

        category_losses: dict[str, Decimal] = {}

        severity_counts = {
            severity.value: 0
            for severity in DamageSeverity
        }

        for assessment in included:
            category_key = assessment.category.value

            category_losses[category_key] = (
                category_losses.get(
                    category_key,
                    Decimal("0"),
                )
                + assessment.estimated_total_loss
            )

            severity_counts[
                assessment.severity.value
            ] += 1

        return {
            "district_id": district_id,
            "currency": "INR",
            "verified_only": verified_only,
            "total_assessments": len(assessments),
            "included_assessments": len(included),
            "estimated_total_loss": str(
                round_money(total_loss)
            ),
            "category_losses": {
                category: str(
                    round_money(amount)
                )
                for category, amount
                in category_losses.items()
            },
            "severity_counts": severity_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_report(
        self,
        assessment_id: str,
    ) -> dict[str, Any]:

        assessment = await self.get_assessment(
            assessment_id
        )

        return {
            "report_id": str(uuid.uuid4()),
            "assessment_id": assessment.id,
            "district_id": assessment.district_id,
            "asset_reference": (
                assessment.asset_reference
            ),
            "category": assessment.category.value,
            "source": assessment.source.value,
            "description": assessment.description,
            "severity": assessment.severity.value,
            "damage_ratio": assessment.damage_ratio,
            "confidence": assessment.confidence,
            "financial_estimate": {
                "currency": "INR",
                "asset_value": str(
                    assessment.estimated_asset_value
                ),
                "repair_cost": str(
                    assessment.repair_cost
                ),
                "replacement_cost": str(
                    assessment.replacement_cost
                ),
                "cleanup_cost": str(
                    assessment.cleanup_cost
                ),
                "debris_removal_cost": str(
                    assessment.debris_removal_cost
                ),
                "downtime_cost": str(
                    assessment.downtime_cost
                ),
                "estimated_total_loss": str(
                    assessment.estimated_total_loss
                ),
            },
            "flood_exposure": {
                "depth_m": assessment.flood_depth_m,
                "duration_hours": (
                    assessment.exposure_hours
                ),
                "affected_area_sqm": (
                    assessment.affected_area_sqm
                ),
                "affected_area_hectares": (
                    assessment.affected_area_hectares
                ),
            },
            "evidence": {
                "images": assessment.image_references,
                "documents": (
                    assessment.evidence_references
                ),
            },
            "verification": {
                "status": assessment.status.value,
                "reviewed_by": assessment.reviewed_by,
                "review_notes": (
                    assessment.review_notes
                ),
                "reviewed_at": (
                    assessment.reviewed_at.isoformat()
                    if assessment.reviewed_at
                    else None
                ),
            },
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "This report provides a preliminary "
                "flood damage estimate. It does not "
                "constitute official damage certification, "
                "structural safety clearance, insurance "
                "approval, or government compensation "
                "authorization."
            ),
        }
