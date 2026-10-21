"""Placeholder module: ai/agents/risk_assessment_agent.py; implementation required."""

"""
backend/ai/agents/risk_assessment_agent.py

Kerala Flood Relief Coordination Agent
Multi-Hazard Risk Assessment Agent

Python 3.11+
Standard library only.

Capabilities
------------
- District and locality risk assessment
- Rainfall and river risk indicators
- Dam storage and release indicators
- Flood forecast probability integration
- Population and infrastructure exposure
- Vulnerability assessment
- Source verification and freshness checks
- Multi-source evidence fusion
- Risk confidence and uncertainty reporting
- Human review workflow
- Audit logging
- JSON report export

IMPORTANT:
This module is a decision-support prototype.

The included thresholds and scoring weights are
illustrative heuristics, NOT validated Kerala flood
warning thresholds or official emergency criteria.

No public warning, evacuation order, dam operation,
rescue dispatch or road closure is executed.

Qualified authorities must independently verify
evidence and authorize operational decisions.
"""

from __future__ import annotations

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

class RiskAssessmentError(Exception):
    pass


class RiskValidationError(RiskAssessmentError):
    pass


class RiskEvidenceError(RiskAssessmentError):
    pass


class RiskApprovalError(RiskAssessmentError):
    pass


class RiskExportError(RiskAssessmentError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNDETERMINED = "undetermined"


class HazardType(str, Enum):
    HEAVY_RAINFALL = "heavy_rainfall"
    RIVER_FLOODING = "river_flooding"
    DAM_RELATED_FLOODING = "dam_related_flooding"
    URBAN_FLOODING = "urban_flooding"
    FLASH_FLOODING = "flash_flooding"
    LANDSLIDE = "landslide"
    COASTAL_FLOODING = "coastal_flooding"
    COMPOUND_FLOODING = "compound_flooding"


class EvidenceType(str, Enum):
    RAINFALL = "rainfall"
    RIVER_LEVEL = "river_level"
    RIVER_RISE_RATE = "river_rise_rate"
    DAM_STORAGE = "dam_storage"
    DAM_RELEASE = "dam_release"
    FLOOD_FORECAST = "flood_forecast"
    FIELD_REPORT = "field_report"
    SATELLITE_OBSERVATION = "satellite_observation"
    INFRASTRUCTURE_REPORT = "infrastructure_report"


class EvidenceVerification(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class AssessmentConfidence(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class AssessmentStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED_FOR_ANALYSIS = "approved_for_analysis"
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_datetime(value: datetime) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise RiskValidationError(
            "Timezone-aware datetime required"
        )


def validate_probability(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise RiskValidationError(
            f"{name} must be between 0 and 1"
        )


def validate_nonnegative(
    value: int | float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RiskValidationError(
            f"{name} must be nonnegative and finite"
        )


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize(item)
            for item in value
        ]

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RiskAssessmentConfig:
    max_evidence_age_minutes: int = 180
    max_forecast_age_minutes: int = 120

    min_source_reliability: float = 0.65
    min_independent_sources: int = 2

    moderate_threshold: float = 0.35
    high_threshold: float = 0.60
    critical_threshold: float = 0.85

    rainfall_moderate_mm_24h: float = 50.0
    rainfall_high_mm_24h: float = 115.0
    rainfall_critical_mm_24h: float = 204.5

    river_rise_high_m_per_hour: float = 0.40

    dam_storage_high_percent: float = 85.0
    dam_storage_critical_percent: float = 95.0

    max_evidence_items: int = 5000

    require_verified_evidence: bool = True
    require_human_review: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_evidence_age_minutes",
            "max_forecast_age_minutes",
            "min_independent_sources",
            "max_evidence_items",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RiskValidationError(
                    f"{name} must be positive integer"
                )

        for name in (
            "min_source_reliability",
            "moderate_threshold",
            "high_threshold",
            "critical_threshold",
        ):
            validate_probability(
                getattr(self, name),
                name,
            )

        if not (
            self.moderate_threshold
            <= self.high_threshold
            <= self.critical_threshold
        ):
            raise RiskValidationError(
                "Risk thresholds must be ascending"
            )

        for name in (
            "rainfall_moderate_mm_24h",
            "rainfall_high_mm_24h",
            "rainfall_critical_mm_24h",
            "river_rise_high_m_per_hour",
            "dam_storage_high_percent",
            "dam_storage_critical_percent",
        ):
            validate_nonnegative(
                getattr(self, name),
                name,
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class RiskEvidence:
    evidence_id: str
    source_id: str

    district: str
    locality: str

    evidence_type: EvidenceType

    value: float
    unit: str

    verification: EvidenceVerification
    reliability: float

    observed_at: datetime

    description: str = ""

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "source_id",
            "district",
            "locality",
            "unit",
        ):
            if not getattr(self, name).strip():
                raise RiskValidationError(
                    f"{name} required"
                )

        validate_nonnegative(
            self.value,
            "value",
        )

        validate_probability(
            self.reliability,
            "reliability",
        )

        validate_datetime(
            self.observed_at
        )


@dataclass(frozen=True)
class PopulationExposure:
    total_population: int

    exposed_population: int

    children_exposed: int = 0
    older_adults_exposed: int = 0
    medically_vulnerable_exposed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "total_population",
            "exposed_population",
            "children_exposed",
            "older_adults_exposed",
            "medically_vulnerable_exposed",
        ):
            value = getattr(self, name)

            validate_nonnegative(
                value,
                name,
            )

            if not isinstance(value, int):
                raise RiskValidationError(
                    f"{name} must be integer"
                )

        if (
            self.exposed_population
            > self.total_population
        ):
            raise RiskValidationError(
                "Exposed population exceeds total population"
            )

        for name in (
            "children_exposed",
            "older_adults_exposed",
            "medically_vulnerable_exposed",
        ):
            if getattr(self, name) > (
                self.exposed_population
            ):
                raise RiskValidationError(
                    f"{name} exceeds exposed population"
                )


@dataclass(frozen=True)
class InfrastructureExposure:
    hospitals_at_risk: int = 0
    shelters_at_risk: int = 0
    roads_at_risk: int = 0
    bridges_at_risk: int = 0
    power_facilities_at_risk: int = 0
    water_facilities_at_risk: int = 0

    def __post_init__(self) -> None:
        for name in (
            "hospitals_at_risk",
            "shelters_at_risk",
            "roads_at_risk",
            "bridges_at_risk",
            "power_facilities_at_risk",
            "water_facilities_at_risk",
        ):
            value = getattr(self, name)

            validate_nonnegative(
                value,
                name,
            )

            if not isinstance(value, int):
                raise RiskValidationError(
                    f"{name} must be integer"
                )


@dataclass(frozen=True)
class RiskAssessmentRequest:
    request_id: str

    district: str
    locality: str

    hazards: tuple[HazardType, ...]

    evidence: tuple[RiskEvidence, ...]

    population: PopulationExposure
    infrastructure: InfrastructureExposure

    created_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "district",
            "locality",
        ):
            if not getattr(self, name).strip():
                raise RiskValidationError(
                    f"{name} required"
                )

        validate_datetime(
            self.created_at
        )

        if not self.hazards:
            raise RiskValidationError(
                "At least one hazard required"
            )


@dataclass(frozen=True)
class RiskEvidenceQuality:
    accepted_evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]

    independent_source_count: int

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RiskIndicatorAssessment:
    indicator_name: str

    score: float

    evidence_ids: tuple[str, ...]

    explanation: str


@dataclass(frozen=True)
class RiskExposureAssessment:
    population_score: float
    vulnerability_score: float
    infrastructure_score: float

    combined_exposure_score: float


@dataclass(frozen=True)
class RiskAssessmentResult:
    assessment_id: str
    request_id: str

    district: str
    locality: str

    hazards: tuple[HazardType, ...]

    risk_score: float | None
    risk_level: RiskLevel

    confidence: AssessmentConfidence

    evidence_quality: RiskEvidenceQuality

    indicators: tuple[
        RiskIndicatorAssessment,
        ...
    ]

    exposure: RiskExposureAssessment

    recommendations: tuple[str, ...]

    warnings: tuple[str, ...]

    status: AssessmentStatus

    human_review_required: bool

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RiskAssessmentReview:
    review_id: str
    assessment_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ReviewDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class RiskAuditRecord:
    audit_id: str
    assessment_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# EVIDENCE VALIDATION ENGINE
# ============================================================

class RiskEvidenceValidator:
    EXPECTED_UNITS = {
        EvidenceType.RAINFALL: "mm_24h",
        EvidenceType.RIVER_LEVEL: "m",
        EvidenceType.RIVER_RISE_RATE: "m_per_hour",
        EvidenceType.DAM_STORAGE: "percent",
        EvidenceType.DAM_RELEASE: "m3_per_second",
        EvidenceType.FLOOD_FORECAST: "probability",
        EvidenceType.FIELD_REPORT: "probability",
        EvidenceType.SATELLITE_OBSERVATION: "probability",
        EvidenceType.INFRASTRUCTURE_REPORT: "probability",
    }

    def __init__(
        self,
        config: RiskAssessmentConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        request: RiskAssessmentRequest,
        *,
        now: datetime,
    ) -> tuple[
        tuple[RiskEvidence, ...],
        RiskEvidenceQuality,
    ]:
        accepted = []
        rejected = []
        warnings = []
        seen_ids = set()

        if len(request.evidence) > (
            self.config.max_evidence_items
        ):
            raise RiskEvidenceError(
                "Evidence limit exceeded"
            )

        for item in request.evidence:
            reason = None

            if item.evidence_id in seen_ids:
                reason = "duplicate evidence ID"

            seen_ids.add(
                item.evidence_id
            )

            if reason is None and (
                item.district.casefold()
                != request.district.casefold()
                or item.locality.casefold()
                != request.locality.casefold()
            ):
                reason = "location mismatch"

            if reason is None and (
                item.verification
                != EvidenceVerification.VERIFIED
            ):
                reason = "not verified"

            if reason is None and (
                item.reliability
                < self.config.min_source_reliability
            ):
                reason = "insufficient source reliability"

            if reason is None and (
                item.observed_at > now
            ):
                reason = "future observation timestamp"

            max_age = (
                self.config.max_forecast_age_minutes
                if item.evidence_type
                == EvidenceType.FLOOD_FORECAST
                else self.config.max_evidence_age_minutes
            )

            if reason is None and (
                now - item.observed_at
                > timedelta(minutes=max_age)
            ):
                reason = "stale observation"

            expected_unit = self.EXPECTED_UNITS[
                item.evidence_type
            ]

            if reason is None and (
                item.unit != expected_unit
            ):
                reason = (
                    f"unit mismatch: expected {expected_unit}"
                )

            if reason is None and (
                item.evidence_type
                in {
                    EvidenceType.FLOOD_FORECAST,
                    EvidenceType.FIELD_REPORT,
                    EvidenceType.SATELLITE_OBSERVATION,
                    EvidenceType.INFRASTRUCTURE_REPORT,
                }
                and item.value > 1
            ):
                reason = "probability exceeds 1"

            if reason is None and (
                item.evidence_type
                == EvidenceType.DAM_STORAGE
                and item.value > 100
            ):
                reason = "dam storage exceeds 100 percent"

            if reason is not None:
                rejected.append(
                    item.evidence_id
                )

                warnings.append(
                    f"{item.evidence_id}: {reason}"
                )

                continue

            accepted.append(item)

        independent_sources = len({
            item.source_id
            for item in accepted
        })

        if independent_sources < (
            self.config.min_independent_sources
        ):
            warnings.append(
                "Insufficient independent evidence sources"
            )

        quality = RiskEvidenceQuality(
            accepted_evidence_ids=tuple(
                item.evidence_id
                for item in accepted
            ),
            rejected_evidence_ids=tuple(
                rejected
            ),
            independent_source_count=(
                independent_sources
            ),
            warnings=tuple(warnings),
        )

        return tuple(accepted), quality


# ============================================================
# HAZARD INDICATOR ENGINE
# ============================================================

class RiskIndicatorEngine:
    def __init__(
        self,
        config: RiskAssessmentConfig,
    ) -> None:
        self.config = config

    def score_evidence(
        self,
        evidence: RiskEvidence,
    ) -> tuple[float | None, str]:
        value = evidence.value

        if evidence.evidence_type == EvidenceType.RAINFALL:
            if value >= (
                self.config.rainfall_critical_mm_24h
            ):
                return 1.0, "Very high 24-hour rainfall indicator"

            if value >= (
                self.config.rainfall_high_mm_24h
            ):
                return 0.75, "High 24-hour rainfall indicator"

            if value >= (
                self.config.rainfall_moderate_mm_24h
            ):
                return 0.45, "Elevated rainfall indicator"

            return 0.15, "Lower rainfall indicator"

        if evidence.evidence_type == (
            EvidenceType.RIVER_RISE_RATE
        ):
            if value >= (
                self.config.river_rise_high_m_per_hour
            ):
                return 0.85, "Rapid river-level rise indicator"

            if value >= 0.15:
                return 0.45, "Elevated river-rise indicator"

            return 0.15, "Lower river-rise indicator"

        if evidence.evidence_type == (
            EvidenceType.DAM_STORAGE
        ):
            if value >= (
                self.config.dam_storage_critical_percent
            ):
                return 0.90, "Very high reservoir storage indicator"

            if value >= (
                self.config.dam_storage_high_percent
            ):
                return 0.65, "High reservoir storage indicator"

            return 0.20, "Lower reservoir storage indicator"

        if evidence.evidence_type in {
            EvidenceType.FLOOD_FORECAST,
            EvidenceType.FIELD_REPORT,
            EvidenceType.SATELLITE_OBSERVATION,
            EvidenceType.INFRASTRUCTURE_REPORT,
        }:
            return (
                value,
                "Source-reported probability indicator",
            )

        # Absolute river level and dam release
        # cannot be interpreted safely without
        # location-specific reference thresholds.
        return (
            None,
            "Requires site-specific reference thresholds",
        )

    def assess(
        self,
        evidence: Sequence[RiskEvidence],
    ) -> tuple[RiskIndicatorAssessment, ...]:
        indicators = []

        # Use the latest observation per source/type
        # to avoid counting repeated measurements
        # from one source as independent evidence.
        latest = {}

        for item in evidence:
            key = (
                item.source_id,
                item.evidence_type,
            )

            previous = latest.get(key)

            if (
                previous is None
                or item.observed_at > previous.observed_at
            ):
                latest[key] = item

        for item in latest.values():
            score, explanation = (
                self.score_evidence(item)
            )

            if score is None:
                continue

            indicators.append(
                RiskIndicatorAssessment(
                    indicator_name=(
                        item.evidence_type.value
                    ),
                    score=round(score, 4),
                    evidence_ids=(
                        item.evidence_id,
                    ),
                    explanation=explanation,
                )
            )

        return tuple(indicators)


# ============================================================
# EXPOSURE ENGINE
# ============================================================

class RiskExposureEngine:
    @staticmethod
    def assess(
        population: PopulationExposure,
        infrastructure: InfrastructureExposure,
    ) -> RiskExposureAssessment:
        if population.total_population > 0:
            population_score = (
                population.exposed_population
                / population.total_population
            )
        else:
            population_score = 0.0

        if population.exposed_population > 0:
            vulnerable = min(
                population.exposed_population,
                population.children_exposed
                + population.older_adults_exposed
                + population.medically_vulnerable_exposed,
            )

            vulnerability_score = (
                vulnerable
                / population.exposed_population
            )
        else:
            vulnerability_score = 0.0

        # Infrastructure score is a planning heuristic.
        infrastructure_score = min(
            (
                3 * infrastructure.hospitals_at_risk
                + 2 * infrastructure.shelters_at_risk
                + infrastructure.roads_at_risk
                + 2 * infrastructure.bridges_at_risk
                + 2 * infrastructure.power_facilities_at_risk
                + 2 * infrastructure.water_facilities_at_risk
            ) / 30,
            1.0,
        )

        combined = (
            0.40 * population_score
            + 0.30 * vulnerability_score
            + 0.30 * infrastructure_score
        )

        return RiskExposureAssessment(
            population_score=round(
                population_score, 4
            ),
            vulnerability_score=round(
                vulnerability_score, 4
            ),
            infrastructure_score=round(
                infrastructure_score, 4
            ),
            combined_exposure_score=round(
                combined, 4
            ),
        )


# ============================================================
# RISK FUSION ENGINE
# ============================================================

class RiskFusionEngine:
    def __init__(
        self,
        config: RiskAssessmentConfig,
    ) -> None:
        self.config = config

    def fuse(
        self,
        indicators: Sequence[RiskIndicatorAssessment],
        exposure: RiskExposureAssessment,
        quality: RiskEvidenceQuality,
    ) -> tuple[
        float | None,
        RiskLevel,
        AssessmentConfidence,
    ]:
        if not indicators:
            return (
                None,
                RiskLevel.UNDETERMINED,
                AssessmentConfidence.INSUFFICIENT,
            )

        if quality.independent_source_count < (
            self.config.min_independent_sources
        ):
            return (
                None,
                RiskLevel.UNDETERMINED,
                AssessmentConfidence.INSUFFICIENT,
            )

        # Maximum indicator preserves a strong
        # single hazard signal rather than averaging
        # it away with unrelated low indicators.
        hazard_score = max(
            item.score
            for item in indicators
        )

        combined = (
            0.75 * hazard_score
            + 0.25 * exposure.combined_exposure_score
        )

        combined = round(
            min(max(combined, 0.0), 1.0),
            4,
        )

        if combined >= (
            self.config.critical_threshold
        ):
            level = RiskLevel.CRITICAL

        elif combined >= (
            self.config.high_threshold
        ):
            level = RiskLevel.HIGH

        elif combined >= (
            self.config.moderate_threshold
        ):
            level = RiskLevel.MODERATE

        else:
            level = RiskLevel.LOW

        source_count = (
            quality.independent_source_count
        )

        if source_count >= 4:
            confidence = AssessmentConfidence.HIGH

        elif source_count >= 3:
            confidence = AssessmentConfidence.MODERATE

        else:
            confidence = AssessmentConfidence.LOW

        return (
            combined,
            level,
            confidence,
        )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class RiskRecommendationEngine:
    @staticmethod
    def generate(
        risk_level: RiskLevel,
        exposure: RiskExposureAssessment,
    ) -> tuple[str, ...]:
        recommendations = [
            "Verify observations with authorized data providers",
            "Continue monitoring rainfall, rivers and forecasts",
            "Obtain human disaster-management review",
        ]

        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Immediately escalate the assessment "
                "to the responsible emergency authority",
                "Request operational verification of "
                "rescue-team and shelter readiness",
                "Request qualified review of evacuation needs",
            ])

        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Escalate to district disaster-management personnel",
                "Review emergency resources and response readiness",
                "Verify vulnerable settlements and access routes",
            ])

        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "Increase observation and situation-report frequency",
                "Review shelter capacity and local response resources",
            ])

        elif risk_level == RiskLevel.LOW:
            recommendations.append(
                "Maintain routine preparedness and monitoring"
            )

        else:
            recommendations.extend([
                "Obtain additional independent evidence",
                "Do not interpret insufficient data as low risk",
            ])

        if exposure.vulnerability_score >= 0.5:
            recommendations.append(
                "Prioritize human review of vulnerable populations"
            )

        if exposure.infrastructure_score >= 0.5:
            recommendations.append(
                "Request infrastructure-specific safety assessments"
            )

        return tuple(
            dict.fromkeys(recommendations)
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class RiskAssessmentAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            RiskAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        assessment_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> RiskAuditRecord:
        record = RiskAuditRecord(
            audit_id=generate_id(),
            assessment_id=assessment_id,
            actor_id=actor_id,
            action=action,
            timestamp=utc_now(),
            details=dict(details or {}),
        )

        with self._lock:
            self._records.append(
                record
            )

        return record

    def list_records(
        self,
        assessment_id: str | None = None,
    ) -> tuple[RiskAuditRecord, ...]:
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

class RiskAssessmentReviewManager:
    ALLOWED_ROLES = {
        "district_disaster_management_officer",
        "authorized_risk_analyst",
        "incident_commander",
        "hydrology_officer",
    }

    def __init__(
        self,
        audit_logger: RiskAssessmentAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            RiskAssessmentReview,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        assessment: RiskAssessmentResult,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> RiskAssessmentReview:
        if not reviewer_id.strip():
            raise RiskApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise RiskApprovalError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise RiskApprovalError(
                "Review reason required"
            )

        with self._lock:
            if assessment.assessment_id in self._reviews:
                raise RiskApprovalError(
                    "Assessment already reviewed"
                )

            review = RiskAssessmentReview(
                review_id=generate_id(),
                assessment_id=assessment.assessment_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                assessment.assessment_id
            ] = review

        self.audit_logger.record(
            assessment_id=assessment.assessment_id,
            actor_id=reviewer_id,
            action=f"review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN RISK ASSESSMENT AGENT
# ============================================================

class RiskAssessmentAgent:
    def __init__(
        self,
        config: RiskAssessmentConfig | None = None,
    ) -> None:
        self.config = (
            config or RiskAssessmentConfig()
        )

        self.evidence_validator = (
            RiskEvidenceValidator(
                self.config
            )
        )

        self.indicator_engine = (
            RiskIndicatorEngine(
                self.config
            )
        )

        self.exposure_engine = (
            RiskExposureEngine()
        )

        self.fusion_engine = (
            RiskFusionEngine(
                self.config
            )
        )

        self.recommendation_engine = (
            RiskRecommendationEngine()
        )

        self.audit_logger = (
            RiskAssessmentAuditLogger()
        )

        self.review_manager = (
            RiskAssessmentReviewManager(
                self.audit_logger
            )
        )

        self._assessments: dict[
            str,
            RiskAssessmentResult,
        ] = {}

        self._lock = threading.RLock()

    def run(
        self,
        request: RiskAssessmentRequest,
        *,
        now: datetime | None = None,
    ) -> RiskAssessmentResult:
        now = now or utc_now()

        validate_datetime(now)

        if request.created_at > now:
            raise RiskValidationError(
                "Request timestamp is in the future"
            )

        accepted, quality = (
            self.evidence_validator.validate(
                request,
                now=now,
            )
        )

        indicators = (
            self.indicator_engine.assess(
                accepted
            )
        )

        exposure = (
            self.exposure_engine.assess(
                request.population,
                request.infrastructure,
            )
        )

        score, level, confidence = (
            self.fusion_engine.fuse(
                indicators,
                exposure,
                quality,
            )
        )

        recommendations = (
            self.recommendation_engine.generate(
                level,
                exposure,
            )
        )

        warnings = list(
            quality.warnings
        )

        if level == RiskLevel.UNDETERMINED:
            warnings.append(
                "Risk cannot be reliably classified "
                "with the available evidence"
            )

        if any(
            item.evidence_type in {
                EvidenceType.RIVER_LEVEL,
                EvidenceType.DAM_RELEASE,
            }
            for item in accepted
        ):
            warnings.append(
                "Absolute river levels and dam release "
                "rates require site-specific thresholds"
            )

        warnings.extend([
            "Risk scores are heuristic, not calibrated probabilities",
            "Hazard-specific models require independent validation",
            "Evidence source independence is not established "
            "solely by unique source IDs",
            "Population vulnerability categories may overlap",
            "This result is not an official flood warning",
            "Human authorization is required for operational action",
        ])

        result = RiskAssessmentResult(
            assessment_id=generate_id(),
            request_id=request.request_id,
            district=request.district,
            locality=request.locality,
            hazards=request.hazards,
            risk_score=score,
            risk_level=level,
            confidence=confidence,
            evidence_quality=quality,
            indicators=indicators,
            exposure=exposure,
            recommendations=recommendations,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            status=AssessmentStatus.PENDING_REVIEW,
            human_review_required=True,
            created_at=now,
        )

        with self._lock:
            self._assessments[
                result.assessment_id
            ] = result

        self.audit_logger.record(
            assessment_id=result.assessment_id,
            actor_id="risk_assessment_agent",
            action="risk_assessment_created",
            details={
                "district": request.district,
                "locality": request.locality,
                "risk_level": level.value,
                "confidence": confidence.value,
                "accepted_evidence": str(
                    len(accepted)
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
        decision: ReviewDecision,
        reason: str,
    ) -> RiskAssessmentResult:
        with self._lock:
            assessment = self._assessments.get(
                assessment_id
            )

            if assessment is None:
                raise RiskApprovalError(
                    "Assessment not found"
                )

            if assessment.status != (
                AssessmentStatus.PENDING_REVIEW
            ):
                raise RiskApprovalError(
                    "Assessment is not pending review"
                )

            self.review_manager.review(
                assessment,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            if decision == ReviewDecision.APPROVE:
                status = (
                    AssessmentStatus.APPROVED_FOR_ANALYSIS
                )

            elif decision == ReviewDecision.REJECT:
                status = (
                    AssessmentStatus.REJECTED
                )

            else:
                status = (
                    AssessmentStatus.REVISION_REQUIRED
                )

            updated = replace(
                assessment,
                status=status,
            )

            self._assessments[
                assessment_id
            ] = updated

            return updated

    def get_assessment(
        self,
        assessment_id: str,
    ) -> RiskAssessmentResult | None:
        with self._lock:
            return self._assessments.get(
                assessment_id
            )

    def get_audit_history(
        self,
        assessment_id: str,
    ) -> tuple[RiskAuditRecord, ...]:
        return self.audit_logger.list_records(
            assessment_id
        )


# ============================================================
# JSON EXPORTER
# ============================================================

class RiskAssessmentReportExporter:
    @staticmethod
    def export_json(
        assessment: RiskAssessmentResult,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise RiskExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                assessment.to_dict(),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_risk_assessment_agent(
    config: RiskAssessmentConfig | None = None,
) -> RiskAssessmentAgent:
    return RiskAssessmentAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional example observations.
    # These are not official Kerala measurements.

    evidence = (
        RiskEvidence(
            evidence_id="DEMO-RAIN-001",
            source_id="DEMO-RAIN-STATION",
            district="Thrissur",
            locality="Illustrative Zone A",
            evidence_type=EvidenceType.RAINFALL,
            value=135.0,
            unit="mm_24h",
            verification=EvidenceVerification.VERIFIED,
            reliability=0.92,
            observed_at=now - timedelta(minutes=20),
            description="Fictional rainfall observation",
        ),
        RiskEvidence(
            evidence_id="DEMO-RIVER-001",
            source_id="DEMO-RIVER-STATION",
            district="Thrissur",
            locality="Illustrative Zone A",
            evidence_type=EvidenceType.RIVER_RISE_RATE,
            value=0.48,
            unit="m_per_hour",
            verification=EvidenceVerification.VERIFIED,
            reliability=0.90,
            observed_at=now - timedelta(minutes=15),
            description="Fictional river-rise observation",
        ),
        RiskEvidence(
            evidence_id="DEMO-FORECAST-001",
            source_id="DEMO-FORECAST-MODEL",
            district="Thrissur",
            locality="Illustrative Zone A",
            evidence_type=EvidenceType.FLOOD_FORECAST,
            value=0.82,
            unit="probability",
            verification=EvidenceVerification.VERIFIED,
            reliability=0.88,
            observed_at=now - timedelta(minutes=10),
            description="Fictional model output",
        ),
    )

    request = RiskAssessmentRequest(
        request_id="DEMO-RISK-001",
        district="Thrissur",
        locality="Illustrative Zone A",
        hazards=(
            HazardType.HEAVY_RAINFALL,
            HazardType.RIVER_FLOODING,
        ),
        evidence=evidence,
        population=PopulationExposure(
            total_population=10000,
            exposed_population=2500,
            children_exposed=400,
            older_adults_exposed=300,
            medically_vulnerable_exposed=100,
        ),
        infrastructure=InfrastructureExposure(
            hospitals_at_risk=1,
            shelters_at_risk=2,
            roads_at_risk=4,
            bridges_at_risk=1,
            power_facilities_at_risk=1,
        ),
        created_at=now,
    )

    agent = create_risk_assessment_agent()

    assessment = agent.run(
        request,
        now=now,
    )

    print(
        json.dumps(
            assessment.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    reviewed = agent.review_assessment(
        assessment.assessment_id,
        reviewer_id="DEMO-RISK-OFFICER",
        reviewer_role="authorized_risk_analyst",
        decision=ReviewDecision.APPROVE,
        reason="Illustrative analysis review",
    )

    print(
        "Review status:",
        reviewed.status.value,
    )

    RiskAssessmentReportExporter.export_json(
        reviewed,
        "outputs/agents/risk_assessment_report.json",
    )

    print(
        "No public warning or operational "
        "response was triggered."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RiskAssessmentError",
    "RiskValidationError",
    "RiskEvidenceError",
    "RiskApprovalError",
    "RiskExportError",
    "RiskLevel",
    "HazardType",
    "EvidenceType",
    "EvidenceVerification",
    "AssessmentConfidence",
    "AssessmentStatus",
    "ReviewDecision",
    "RiskAssessmentConfig",
    "RiskEvidence",
    "PopulationExposure",
    "InfrastructureExposure",
    "RiskAssessmentRequest",
    "RiskEvidenceQuality",
    "RiskIndicatorAssessment",
    "RiskExposureAssessment",
    "RiskAssessmentResult",
    "RiskAssessmentReview",
    "RiskAuditRecord",
    "RiskEvidenceValidator",
    "RiskIndicatorEngine",
    "RiskExposureEngine",
    "RiskFusionEngine",
    "RiskRecommendationEngine",
    "RiskAssessmentAuditLogger",
    "RiskAssessmentReviewManager",
    "RiskAssessmentAgent",
    "RiskAssessmentReportExporter",
    "create_risk_assessment_agent",
]
