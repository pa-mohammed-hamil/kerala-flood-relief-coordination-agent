"""Placeholder module: ai/agents/damage_assessment_agent.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
backend/ai/agents/damage_assessment_agent.py

Evidence-grounded, human-reviewed disaster damage assessment.

Capabilities:
- Building, road, bridge, vehicle and infrastructure assessment
- Flood and debris observations
- Drone/satellite/computer-vision evidence integration
- Before/after comparison evidence
- Damage severity screening
- Repair-cost estimation from verified unit costs
- Assessment confidence and uncertainty
- Missing evidence identification
- Priority-based inspection recommendations
- Duplicate assessment detection
- Human approval and audit trail
- JSON report export

No automatic:
- Structural safety certification
- Road closure or reopening
- Evacuation
- Rescue dispatch
- Compensation or insurance approval

All damage estimates are preliminary and advisory.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class DamageAssessmentError(Exception):
    pass


class DamageValidationError(DamageAssessmentError):
    pass


class DamageEvidenceError(DamageAssessmentError):
    pass


class DamageApprovalError(DamageAssessmentError):
    pass


class DamageExportError(DamageAssessmentError):
    pass


# ============================================================
# ENUMS
# ============================================================

class AssetType(str, Enum):
    BUILDING = "building"
    ROAD = "road"
    BRIDGE = "bridge"
    VEHICLE = "vehicle"
    ELECTRICAL = "electrical"
    WATER_SUPPLY = "water_supply"
    HEALTHCARE_FACILITY = "healthcare_facility"
    SCHOOL = "school"
    OTHER = "other"


class DamageSeverity(str, Enum):
    NONE_VISIBLE = "none_visible"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    FIELD_INSPECTION = "field_inspection"
    DRONE_IMAGE = "drone_image"
    SATELLITE_IMAGE = "satellite_image"
    COMPUTER_VISION = "computer_vision"
    SENSOR = "sensor"
    DOCUMENT = "document"
    BEFORE_AFTER = "before_after"


class AssessmentStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class InspectionPriority(str, Enum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    HIGH = "high"
    URGENT = "urgent"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DamageAssessmentConfig:
    maximum_evidence_age_days: int = 90
    minimum_reliability: float = 0.60
    minimum_confidence: float = 0.70
    minimum_independent_sources: int = 2

    maximum_cost_items: int = 100
    maximum_evidence_items: int = 50

    deduplication_minutes: int = 60

    require_human_review: bool = True

    currency: str = "INR"

    def __post_init__(self) -> None:
        for name in (
            "maximum_evidence_age_days",
            "minimum_independent_sources",
            "maximum_cost_items",
            "maximum_evidence_items",
            "deduplication_minutes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DamageValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "minimum_reliability",
            "minimum_confidence",
        ):
            value = getattr(self, name)

            if not (
                math.isfinite(value)
                and 0 <= value <= 1
            ):
                raise DamageValidationError(
                    f"{name} must be in [0,1]"
                )

        if not self.currency.strip():
            raise DamageValidationError(
                "Currency is required"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class DamageEvidence:
    evidence_id: str
    source_id: str
    evidence_type: EvidenceType

    observed_at: datetime

    verified: bool
    reliability: float

    asset_id: str

    damage_probabilities: Mapping[
        DamageSeverity,
        float,
    ]

    observations: tuple[str, ...] = ()

    reference_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise DamageValidationError(
                "evidence_id is required"
            )

        if not self.source_id.strip():
            raise DamageValidationError(
                "source_id is required"
            )

        if not self.asset_id.strip():
            raise DamageValidationError(
                "asset_id is required"
            )

        require_aware_datetime(
            self.observed_at
        )

        require_probability(
            self.reliability,
            "reliability",
        )

        if not self.damage_probabilities:
            raise DamageValidationError(
                "Damage probabilities are required"
            )

        total = 0.0

        for severity, probability in (
            self.damage_probabilities.items()
        ):
            if not isinstance(
                severity,
                DamageSeverity,
            ):
                raise DamageValidationError(
                    "Invalid severity key"
                )

            require_probability(
                probability,
                "damage_probability",
            )

            total += probability

        if not math.isclose(
            total,
            1.0,
            abs_tol=0.001,
        ):
            raise DamageValidationError(
                "Damage probabilities must sum to 1"
            )


@dataclass(frozen=True)
class DamageAsset:
    asset_id: str
    asset_type: AssetType

    district: str
    location_description: str

    latitude: float | None = None
    longitude: float | None = None

    critical_infrastructure: bool = False

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise DamageValidationError(
                "asset_id is required"
            )

        if not self.district.strip():
            raise DamageValidationError(
                "district is required"
            )

        if (
            self.latitude is None
        ) != (
            self.longitude is None
        ):
            raise DamageValidationError(
                "Latitude and longitude must be supplied together"
            )

        if self.latitude is not None:
            if not (
                math.isfinite(self.latitude)
                and -90 <= self.latitude <= 90
                and math.isfinite(self.longitude)
                and -180 <= self.longitude <= 180
            ):
                raise DamageValidationError(
                    "Invalid coordinates"
                )


@dataclass(frozen=True)
class DamageCostItem:
    item_id: str
    description: str

    quantity: float
    unit: str
    unit_cost: float

    verified_price_source: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise DamageValidationError(
                "item_id is required"
            )

        if not self.description.strip():
            raise DamageValidationError(
                "description is required"
            )

        if not self.unit.strip():
            raise DamageValidationError(
                "unit is required"
            )

        if not self.verified_price_source.strip():
            raise DamageValidationError(
                "Verified price source is required"
            )

        for name in (
            "quantity",
            "unit_cost",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise DamageValidationError(
                    f"{name} must be finite and nonnegative"
                )

    @property
    def total_cost(self) -> float:
        return round(
            self.quantity * self.unit_cost,
            2,
        )


@dataclass(frozen=True)
class DamageAssessmentRequest:
    request_id: str
    asset: DamageAsset

    evidence: tuple[DamageEvidence, ...]
    cost_items: tuple[DamageCostItem, ...]

    requested_at: datetime


@dataclass(frozen=True)
class DamageCostEstimate:
    currency: str
    direct_cost: float

    lower_estimate: float
    upper_estimate: float

    items: tuple[dict[str, Any], ...]

    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class DamageAssessmentResult:
    assessment_id: str
    request_id: str

    asset: DamageAsset

    severity: DamageSeverity
    severity_probabilities: Mapping[str, float]

    confidence: float

    inspection_priority: InspectionPriority

    accepted_evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]

    independent_source_count: int

    cost_estimate: DamageCostEstimate | None

    findings: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    warnings: tuple[str, ...]

    status: AssessmentStatus

    requires_human_review: bool
    duplicate_detected: bool

    assessed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class DamageAssessmentReview:
    review_id: str
    assessment_id: str

    reviewer_id: str
    reviewer_role: str

    approved: bool
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class DamageAuditRecord:
    audit_id: str
    assessment_id: str

    actor_id: str
    action: str

    timestamp: datetime
    details: Mapping[str, str]


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def require_aware_datetime(
    value: datetime,
) -> None:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise DamageValidationError(
            "Datetime must be timezone-aware"
        )


def require_probability(
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
            f"{name} must be in [0,1]"
        )


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(
            asdict(value)
        )

    if isinstance(value, Mapping):
        return {
            str(
                key.value
                if isinstance(key, Enum)
                else key
            ): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            serialize(item)
            for item in value
        ]

    return value


# ============================================================
# EVIDENCE VALIDATOR
# ============================================================

class DamageEvidenceValidator:
    def __init__(
        self,
        config: DamageAssessmentConfig,
    ) -> None:
        self.config = config

    def filter(
        self,
        request: DamageAssessmentRequest,
        *,
        now: datetime,
    ) -> tuple[
        tuple[DamageEvidence, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        require_aware_datetime(now)

        if len(request.evidence) > (
            self.config.maximum_evidence_items
        ):
            raise DamageEvidenceError(
                "Evidence limit exceeded"
            )

        accepted = []
        rejected = []
        warnings = []

        seen = set()

        for item in request.evidence:
            reason = None

            if item.evidence_id in seen:
                reason = "duplicate evidence ID"

            elif item.asset_id != (
                request.asset.asset_id
            ):
                reason = "asset ID mismatch"

            elif not item.verified:
                reason = "unverified evidence"

            elif item.reliability < (
                self.config.minimum_reliability
            ):
                reason = "insufficient reliability"

            elif item.observed_at > (
                now + timedelta(minutes=5)
            ):
                reason = "future timestamp"

            elif (
                now - item.observed_at
            ) > timedelta(
                days=(
                    self.config.maximum_evidence_age_days
                )
            ):
                reason = "stale evidence"

            seen.add(
                item.evidence_id
            )

            if reason:
                rejected.append(
                    item.evidence_id
                )

                warnings.append(
                    f"{item.evidence_id}: {reason}"
                )

            else:
                accepted.append(
                    item
                )

        return (
            tuple(accepted),
            tuple(rejected),
            tuple(warnings),
        )


# ============================================================
# SEVERITY FUSION ENGINE
# ============================================================

class DamageSeverityFusionEngine:
    SEVERITY_ORDER = (
        DamageSeverity.NONE_VISIBLE,
        DamageSeverity.MINOR,
        DamageSeverity.MODERATE,
        DamageSeverity.SEVERE,
        DamageSeverity.UNKNOWN,
    )

    @classmethod
    def aggregate(
        cls,
        evidence: Sequence[DamageEvidence],
    ) -> tuple[
        DamageSeverity,
        dict[str, float],
        float,
    ]:
        if not evidence:
            return (
                DamageSeverity.UNKNOWN,
                {
                    severity.value: (
                        1.0
                        if severity == DamageSeverity.UNKNOWN
                        else 0.0
                    )
                    for severity in cls.SEVERITY_ORDER
                },
                0.0,
            )

        # Limit correlated evidence:
        # one highest-reliability observation per source.
        by_source: dict[
            str,
            DamageEvidence,
        ] = {}

        for item in evidence:
            previous = by_source.get(
                item.source_id
            )

            if (
                previous is None
                or item.reliability
                > previous.reliability
            ):
                by_source[
                    item.source_id
                ] = item

        weighted = {
            severity: 0.0
            for severity in cls.SEVERITY_ORDER
        }

        total_weight = 0.0

        for item in by_source.values():
            weight = item.reliability

            total_weight += weight

            for severity in cls.SEVERITY_ORDER:
                weighted[severity] += (
                    weight
                    * item.damage_probabilities.get(
                        severity,
                        0.0,
                    )
                )

        if total_weight <= 0:
            raise DamageEvidenceError(
                "Invalid evidence weights"
            )

        probabilities = {
            severity.value: (
                weighted[severity] / total_weight
            )
            for severity in cls.SEVERITY_ORDER
        }

        severity = max(
            cls.SEVERITY_ORDER,
            key=lambda value: (
                probabilities[value.value]
            ),
        )

        confidence = (
            probabilities[severity.value]
        )

        return (
            severity,
            probabilities,
            confidence,
        )


# ============================================================
# INSPECTION PRIORITY ENGINE
# ============================================================

class DamageInspectionPriorityEngine:
    @staticmethod
    def determine(
        *,
        asset: DamageAsset,
        severity: DamageSeverity,
        confidence: float,
        minimum_confidence: float,
    ) -> InspectionPriority:
        if (
            asset.critical_infrastructure
            and severity in (
                DamageSeverity.SEVERE,
                DamageSeverity.UNKNOWN,
            )
        ):
            return InspectionPriority.URGENT

        if severity == DamageSeverity.SEVERE:
            return InspectionPriority.HIGH

        if (
            asset.critical_infrastructure
            or severity == DamageSeverity.MODERATE
            or confidence < minimum_confidence
        ):
            return InspectionPriority.ELEVATED

        return InspectionPriority.ROUTINE


# ============================================================
# COST ESTIMATION ENGINE
# ============================================================

class DamageCostEstimationEngine:
    def __init__(
        self,
        config: DamageAssessmentConfig,
    ) -> None:
        self.config = config

    def estimate(
        self,
        items: Sequence[DamageCostItem],
    ) -> DamageCostEstimate | None:
        if not items:
            return None

        if len(items) > (
            self.config.maximum_cost_items
        ):
            raise DamageValidationError(
                "Cost item limit exceeded"
            )

        if len({
            item.item_id
            for item in items
        }) != len(items):
            raise DamageValidationError(
                "Duplicate cost item IDs"
            )

        direct_cost = round(
            sum(
                item.total_cost
                for item in items
            ),
            2,
        )

        # Illustrative ±20% scenario bounds.
        # These are NOT statistically calibrated
        # confidence intervals.
        lower = round(
            direct_cost * 0.80,
            2,
        )

        upper = round(
            direct_cost * 1.20,
            2,
        )

        item_details = tuple(
            {
                "item_id": item.item_id,
                "description": item.description,
                "quantity": item.quantity,
                "unit": item.unit,
                "unit_cost": item.unit_cost,
                "total_cost": item.total_cost,
                "price_source": (
                    item.verified_price_source
                ),
            }
            for item in items
        )

        return DamageCostEstimate(
            currency=self.config.currency,
            direct_cost=direct_cost,
            lower_estimate=lower,
            upper_estimate=upper,
            items=item_details,
            assumptions=(
                "Quantities are supplied estimates",
                "Unit prices require current verification",
                "Scenario bounds are illustrative ±20%",
                "No engineering survey has been performed",
                "Indirect losses and taxes may be excluded",
                "This is not an approved compensation amount",
            ),
        )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class DamageRecommendationEngine:
    @staticmethod
    def generate(
        *,
        asset: DamageAsset,
        severity: DamageSeverity,
        priority: InspectionPriority,
        confidence: float,
        evidence_count: int,
    ) -> tuple[str, ...]:
        recommendations = []

        if evidence_count == 0:
            recommendations.append(
                "Obtain verified inspection evidence"
            )

        if confidence < 0.70:
            recommendations.append(
                "Request additional independent evidence"
            )

        if severity == DamageSeverity.UNKNOWN:
            recommendations.append(
                "Arrange professional assessment "
                "before making a damage determination"
            )

        elif severity == DamageSeverity.SEVERE:
            recommendations.append(
                "Request urgent qualified inspection"
            )

            recommendations.append(
                "Do not infer structural safety "
                "from image predictions"
            )

        elif severity == DamageSeverity.MODERATE:
            recommendations.append(
                "Arrange detailed on-site inspection"
            )

        elif severity == DamageSeverity.MINOR:
            recommendations.append(
                "Document visible damage and "
                "confirm through field inspection"
            )

        else:
            recommendations.append(
                "Continue monitoring; absence of "
                "visible damage does not establish safety"
            )

        if asset.asset_type == AssetType.ROAD:
            recommendations.append(
                "Verify road accessibility through "
                "authorized field personnel"
            )

        elif asset.asset_type == AssetType.BRIDGE:
            recommendations.append(
                "Request qualified bridge engineering review"
            )

        elif asset.asset_type == AssetType.ELECTRICAL:
            recommendations.append(
                "Request authorized electrical safety inspection"
            )

        elif asset.asset_type == (
            AssetType.HEALTHCARE_FACILITY
        ):
            recommendations.append(
                "Coordinate facility status review "
                "with authorized healthcare officials"
            )

        if asset.critical_infrastructure:
            recommendations.append(
                "Escalate inspection recommendation "
                "to the responsible infrastructure authority"
            )

        if priority == InspectionPriority.URGENT:
            recommendations.append(
                "Prioritize human inspection scheduling"
            )

        return tuple(
            dict.fromkeys(recommendations)
        )


# ============================================================
# DEDUPLICATION ENGINE
# ============================================================

class DamageDeduplicationEngine:
    def __init__(
        self,
        config: DamageAssessmentConfig,
    ) -> None:
        self.config = config

        self._recent: dict[
            str,
            datetime,
        ] = {}

        self._lock = threading.Lock()

    def check(
        self,
        *,
        asset_id: str,
        evidence_ids: Sequence[str],
        now: datetime,
    ) -> bool:
        content = (
            asset_id
            + "|"
            + "|".join(
                sorted(evidence_ids)
            )
        )

        key = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        cutoff = (
            now - timedelta(
                minutes=(
                    self.config.deduplication_minutes
                )
            )
        )

        with self._lock:
            self._recent = {
                key: timestamp
                for key, timestamp in self._recent.items()
                if timestamp >= cutoff
            }

            previous = self._recent.get(
                key
            )

            duplicate = (
                previous is not None
                and previous >= cutoff
            )

            if not duplicate:
                self._recent[
                    key
                ] = now

        return duplicate


# ============================================================
# AUDIT LOGGER
# ============================================================

class DamageAssessmentAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            DamageAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        assessment_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> DamageAuditRecord:
        record = DamageAuditRecord(
            audit_id=generate_id(),
            assessment_id=assessment_id,
            actor_id=actor_id,
            action=action,
            timestamp=utc_now(),
            details=dict(
                details or {}
            ),
        )

        with self._lock:
            self._records.append(
                record
            )

        return record

    def list_records(
        self,
        assessment_id: str | None = None,
    ) -> tuple[DamageAuditRecord, ...]:
        with self._lock:
            records = tuple(
                self._records
            )

        if assessment_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.assessment_id == assessment_id
        )


# ============================================================
# HUMAN REVIEW MANAGER
# ============================================================

class DamageAssessmentReviewManager:
    ALLOWED_ROLES = {
        "damage_assessment_officer",
        "district_official",
        "qualified_engineer",
        "assessment_supervisor",
    }

    def __init__(
        self,
        audit_logger: DamageAssessmentAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            DamageAssessmentReview,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        result: DamageAssessmentResult,
        *,
        reviewer_id: str,
        reviewer_role: str,
        approved: bool,
        reason: str,
    ) -> DamageAssessmentReview:
        if not reviewer_id.strip():
            raise DamageApprovalError(
                "Reviewer identity required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise DamageApprovalError(
                "Reviewer role is not permitted"
            )

        if not reason.strip():
            raise DamageApprovalError(
                "Review reason required"
            )

        if result.status != (
            AssessmentStatus.PENDING_REVIEW
        ):
            raise DamageApprovalError(
                "Assessment is not pending review"
            )

        if (
            result.duplicate_detected
            and approved
        ):
            raise DamageApprovalError(
                "Duplicate assessment cannot be "
                "approved through this workflow"
            )

        with self._lock:
            if result.assessment_id in self._reviews:
                raise DamageApprovalError(
                    "Assessment already reviewed"
                )

            review = DamageAssessmentReview(
                review_id=generate_id(),
                assessment_id=result.assessment_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                approved=approved,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                result.assessment_id
            ] = review

        self.audit_logger.record(
            assessment_id=result.assessment_id,
            actor_id=reviewer_id,
            action=(
                "assessment_approved"
                if approved
                else "assessment_rejected"
            ),
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN DAMAGE ASSESSMENT AGENT
# ============================================================

class DamageAssessmentAgent:
    def __init__(
        self,
        config: DamageAssessmentConfig | None = None,
    ) -> None:
        self.config = (
            config or DamageAssessmentConfig()
        )

        self.evidence_validator = (
            DamageEvidenceValidator(
                self.config
            )
        )

        self.severity_engine = (
            DamageSeverityFusionEngine()
        )

        self.priority_engine = (
            DamageInspectionPriorityEngine()
        )

        self.cost_engine = (
            DamageCostEstimationEngine(
                self.config
            )
        )

        self.recommendation_engine = (
            DamageRecommendationEngine()
        )

        self.deduplication_engine = (
            DamageDeduplicationEngine(
                self.config
            )
        )

        self.audit_logger = (
            DamageAssessmentAuditLogger()
        )

        self.review_manager = (
            DamageAssessmentReviewManager(
                self.audit_logger
            )
        )

        self._assessments: dict[
            str,
            DamageAssessmentResult,
        ] = {}

        self._lock = threading.Lock()

    def run(
        self,
        request: DamageAssessmentRequest,
        *,
        now: datetime | None = None,
    ) -> DamageAssessmentResult:
        now = now or utc_now()

        require_aware_datetime(now)

        require_aware_datetime(
            request.requested_at
        )

        if not request.request_id.strip():
            raise DamageValidationError(
                "request_id is required"
            )

        (
            accepted,
            rejected,
            evidence_warnings,
        ) = self.evidence_validator.filter(
            request,
            now=now,
        )

        independent_sources = len({
            item.source_id
            for item in accepted
        })

        sufficient_evidence = (
            independent_sources
            >= self.config.minimum_independent_sources
        )

        (
            severity,
            probabilities,
            confidence,
        ) = self.severity_engine.aggregate(
            accepted
        )

        warnings = list(
            evidence_warnings
        )

        if not sufficient_evidence:
            severity = DamageSeverity.UNKNOWN
            confidence = 0.0

            probabilities = {
                item.value: (
                    1.0
                    if item == DamageSeverity.UNKNOWN
                    else 0.0
                )
                for item in DamageSeverity
            }

            warnings.append(
                "Insufficient independent verified evidence"
            )

        if confidence < (
            self.config.minimum_confidence
        ):
            warnings.append(
                "Assessment confidence below threshold"
            )

        priority = (
            self.priority_engine.determine(
                asset=request.asset,
                severity=severity,
                confidence=confidence,
                minimum_confidence=(
                    self.config.minimum_confidence
                ),
            )
        )

        cost_estimate = (
            self.cost_engine.estimate(
                request.cost_items
            )
        )

        findings = []

        for item in accepted:
            for observation in item.observations:
                cleaned = " ".join(
                    observation.split()
                )

                if cleaned:
                    findings.append(
                        cleaned[:500]
                    )

        findings = list(
            dict.fromkeys(findings)
        )

        recommendations = (
            self.recommendation_engine.generate(
                asset=request.asset,
                severity=severity,
                priority=priority,
                confidence=confidence,
                evidence_count=len(accepted),
            )
        )

        duplicate = (
            self.deduplication_engine.check(
                asset_id=request.asset.asset_id,
                evidence_ids=[
                    item.evidence_id
                    for item in accepted
                ],
                now=now,
            )
        )

        if duplicate:
            warnings.append(
                "Duplicate assessment evidence detected"
            )

        warnings.extend([
            "Severity is a preliminary screening result",
            "Confidence is not a calibrated probability "
            "of structural safety",
            "Image-based assessment may miss hidden damage",
            "Repair estimates require professional verification",
            "No operational or emergency action was executed",
            "Human review is mandatory",
        ])

        result = DamageAssessmentResult(
            assessment_id=generate_id(),
            request_id=request.request_id,
            asset=request.asset,
            severity=severity,
            severity_probabilities=probabilities,
            confidence=confidence,
            inspection_priority=priority,
            accepted_evidence_ids=tuple(
                item.evidence_id
                for item in accepted
            ),
            rejected_evidence_ids=rejected,
            independent_source_count=(
                independent_sources
            ),
            cost_estimate=cost_estimate,
            findings=tuple(findings),
            recommended_actions=recommendations,
            warnings=tuple(warnings),
            status=AssessmentStatus.PENDING_REVIEW,
            requires_human_review=True,
            duplicate_detected=duplicate,
            assessed_at=now,
        )

        with self._lock:
            self._assessments[
                result.assessment_id
            ] = result

        self.audit_logger.record(
            assessment_id=result.assessment_id,
            actor_id="damage_assessment_agent",
            action="assessment_created",
            details={
                "asset_id": request.asset.asset_id,
                "severity": severity.value,
                "priority": priority.value,
                "source_count": str(
                    independent_sources
                ),
            },
        )

        return result

    def review_assessment(
        self,
        assessment_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        approved: bool,
        reason: str,
    ) -> DamageAssessmentResult:
        with self._lock:
            result = self._assessments.get(
                assessment_id
            )

            if result is None:
                raise DamageApprovalError(
                    "Assessment not found"
                )

            self.review_manager.review(
                result,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                approved=approved,
                reason=reason,
            )

            updated = replace(
                result,
                status=(
                    AssessmentStatus.APPROVED
                    if approved
                    else AssessmentStatus.REJECTED
                ),
            )

            self._assessments[
                assessment_id
            ] = updated

            return updated

    def get_assessment(
        self,
        assessment_id: str,
    ) -> DamageAssessmentResult | None:
        with self._lock:
            return self._assessments.get(
                assessment_id
            )

    def get_audit_history(
        self,
        assessment_id: str,
    ) -> tuple[DamageAuditRecord, ...]:
        return self.audit_logger.list_records(
            assessment_id
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class DamageAssessmentReportExporter:
    @staticmethod
    def export_json(
        result: DamageAssessmentResult,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise DamageExportError(
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
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_damage_assessment_agent(
    config: DamageAssessmentConfig | None = None,
) -> DamageAssessmentAgent:
    return DamageAssessmentAgent(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    now = utc_now()

    asset = DamageAsset(
        asset_id="BUILDING-001",
        asset_type=AssetType.BUILDING,
        district="Thrissur",
        location_description=(
            "Illustrative flood-affected building"
        ),
        critical_infrastructure=False,
    )

    field_evidence = DamageEvidence(
        evidence_id="FIELD-001",
        source_id="INSPECTOR-001",
        evidence_type=EvidenceType.FIELD_INSPECTION,
        observed_at=now - timedelta(hours=2),
        verified=True,
        reliability=0.95,
        asset_id=asset.asset_id,
        damage_probabilities={
            DamageSeverity.NONE_VISIBLE: 0.05,
            DamageSeverity.MINOR: 0.10,
            DamageSeverity.MODERATE: 0.70,
            DamageSeverity.SEVERE: 0.10,
            DamageSeverity.UNKNOWN: 0.05,
        },
        observations=(
            "Illustrative observation: wall damage",
            "Illustrative observation: water marks",
        ),
    )

    vision_evidence = DamageEvidence(
        evidence_id="VISION-001",
        source_id="CV-MODEL-001",
        evidence_type=EvidenceType.COMPUTER_VISION,
        observed_at=now - timedelta(hours=1),
        verified=True,
        reliability=0.80,
        asset_id=asset.asset_id,
        damage_probabilities={
            DamageSeverity.NONE_VISIBLE: 0.05,
            DamageSeverity.MINOR: 0.10,
            DamageSeverity.MODERATE: 0.65,
            DamageSeverity.SEVERE: 0.15,
            DamageSeverity.UNKNOWN: 0.05,
        },
        observations=(
            "Illustrative CV prediction: moderate damage",
        ),
    )

    cost_items = (
        DamageCostItem(
            item_id="COST-001",
            description="Illustrative wall repair",
            quantity=20.0,
            unit="square_meter",
            unit_cost=1500.0,
            verified_price_source=(
                "EXAMPLE-PRICE-LIST-NOT-FOR-PRODUCTION"
            ),
        ),
        DamageCostItem(
            item_id="COST-002",
            description="Illustrative cleaning",
            quantity=1.0,
            unit="job",
            unit_cost=5000.0,
            verified_price_source=(
                "EXAMPLE-PRICE-LIST-NOT-FOR-PRODUCTION"
            ),
        ),
    )

    request = DamageAssessmentRequest(
        request_id=generate_id(),
        asset=asset,
        evidence=(
            field_evidence,
            vision_evidence,
        ),
        cost_items=cost_items,
        requested_at=now,
    )

    agent = create_damage_assessment_agent()

    result = agent.run(
        request,
        now=now,
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    DamageAssessmentReportExporter.export_json(
        result,
        "outputs/agents/damage_assessment_report.json",
    )

    # Human review demonstration.
    # Does not certify building safety.
    updated = agent.review_assessment(
        result.assessment_id,
        reviewer_id="engineer-001",
        reviewer_role="qualified_engineer",
        approved=False,
        reason="Physical inspection is still required",
    )

    print(
        "Review status:",
        updated.status.value,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DamageAssessmentError",
    "DamageValidationError",
    "DamageEvidenceError",
    "DamageApprovalError",
    "DamageExportError",
    "AssetType",
    "DamageSeverity",
    "EvidenceType",
    "AssessmentStatus",
    "InspectionPriority",
    "DamageAssessmentConfig",
    "DamageEvidence",
    "DamageAsset",
    "DamageCostItem",
    "DamageAssessmentRequest",
    "DamageCostEstimate",
    "DamageAssessmentResult",
    "DamageAssessmentReview",
    "DamageAuditRecord",
    "DamageEvidenceValidator",
    "DamageSeverityFusionEngine",
    "DamageInspectionPriorityEngine",
    "DamageCostEstimationEngine",
    "DamageRecommendationEngine",
    "DamageDeduplicationEngine",
    "DamageAssessmentAuditLogger",
    "DamageAssessmentReviewManager",
    "DamageAssessmentAgent",
    "DamageAssessmentReportExporter",
    "create_damage_assessment_agent",
]
