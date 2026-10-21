"""Placeholder module: ai/agents/alert_intelligence_agent.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
File: backend/ai/agents/alert_intelligence_agent.py

Multi-source flood alert intelligence agent.

Features:
    - Rainfall monitoring and anomaly detection
    - River-level and rise-rate assessment
    - Dam-level and release information
    - Weather forecast risk signals
    - Flood ML prediction integration
    - Satellite and drone evidence integration
    - Source freshness and reliability checks
    - District-level risk assessment
    - Evidence-grounded alert recommendations
    - Alert deduplication
    - Human approval workflow
    - Audit trail and JSON reporting

IMPORTANT:
    This module generates advisory recommendations only.

    It must NOT autonomously:
        - Issue public emergency alerts
        - Order evacuations
        - Dispatch rescue personnel
        - Operate dam gates or infrastructure
        - Send SMS, calls or push notifications

    Emergency actions require verified data,
    authorized personnel and approved integrations.

Dependencies:
    Python 3.11+

No external libraries required.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class AlertIntelligenceError(Exception):
    pass


class AlertValidationError(AlertIntelligenceError):
    pass


class AlertSourceError(AlertIntelligenceError):
    pass


class AlertApprovalError(AlertIntelligenceError):
    pass


class AlertExportError(AlertIntelligenceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class AlertSourceType(str, Enum):
    RAINFALL = "rainfall"
    RIVER = "river"
    DAM = "dam"
    WEATHER_FORECAST = "weather_forecast"
    ML_FORECAST = "ml_forecast"
    SATELLITE = "satellite"
    DRONE = "drone"
    FIELD_REPORT = "field_report"


class AlertRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNDETERMINED = "undetermined"


class AlertRecommendationStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class AlertSignalType(str, Enum):
    HEAVY_RAINFALL = "heavy_rainfall"
    RIVER_RISE = "river_rise"
    HIGH_RIVER_LEVEL = "high_river_level"
    HIGH_DAM_STORAGE = "high_dam_storage"
    DAM_RELEASE = "dam_release"
    FORECAST_RISK = "forecast_risk"
    VISUAL_FLOOD_EVIDENCE = "visual_flood_evidence"
    FIELD_FLOOD_REPORT = "field_flood_report"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AlertIntelligenceConfig:
    maximum_source_age_minutes: int = 180

    rainfall_24h_moderate_mm: float = 50.0
    rainfall_24h_high_mm: float = 115.0
    rainfall_24h_critical_mm: float = 204.5

    river_rise_moderate_m_per_hour: float = 0.15
    river_rise_high_m_per_hour: float = 0.40

    dam_storage_high_percent: float = 85.0
    dam_storage_critical_percent: float = 95.0

    forecast_high_probability: float = 0.70
    forecast_critical_probability: float = 0.90

    minimum_source_reliability: float = 0.60
    minimum_independent_sources: int = 2

    deduplication_window_minutes: int = 60

    require_human_review: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_source_age_minutes",
            "minimum_independent_sources",
            "deduplication_window_minutes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise AlertValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "forecast_high_probability",
            "forecast_critical_probability",
            "minimum_source_reliability",
        ):
            value = getattr(self, name)

            if not (
                math.isfinite(value)
                and 0 <= value <= 1
            ):
                raise AlertValidationError(
                    f"{name} must be in [0,1]"
                )

        if (
            self.forecast_high_probability
            > self.forecast_critical_probability
        ):
            raise AlertValidationError(
                "Forecast thresholds are out of order"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class AlertEvidence:
    evidence_id: str

    source_type: AlertSourceType
    source_id: str

    district: str
    observed_at: datetime

    measurements: Mapping[str, float]

    reliability: float = 1.0

    verified: bool = False

    reference_url: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise AlertValidationError(
                "evidence_id is required"
            )

        if not self.source_id.strip():
            raise AlertValidationError(
                "source_id is required"
            )

        if not self.district.strip():
            raise AlertValidationError(
                "district is required"
            )

        if (
            self.observed_at.tzinfo is None
            or self.observed_at.utcoffset() is None
        ):
            raise AlertValidationError(
                "observed_at must be timezone-aware"
            )

        if not (
            math.isfinite(self.reliability)
            and 0 <= self.reliability <= 1
        ):
            raise AlertValidationError(
                "reliability must be in [0,1]"
            )

        for key, value in self.measurements.items():
            if (
                not key
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise AlertValidationError(
                    "Measurements must be finite numbers"
                )


@dataclass(frozen=True)
class AlertSignal:
    signal_id: str

    signal_type: AlertSignalType
    source_id: str

    evidence_id: str

    severity: AlertRiskLevel
    strength: float

    explanation: str


@dataclass(frozen=True)
class AlertRiskAssessment:
    assessment_id: str

    district: str

    risk_level: AlertRiskLevel
    risk_score: float | None

    signals: tuple[AlertSignal, ...]

    evidence_ids: tuple[str, ...]

    independent_source_count: int

    sufficient_evidence: bool

    requires_human_review: bool

    warnings: tuple[str, ...]

    assessed_at: datetime


@dataclass(frozen=True)
class AlertRecommendation:
    recommendation_id: str

    district: str

    risk_level: AlertRiskLevel
    risk_score: float | None

    title: str
    summary: str

    recommended_actions: tuple[str, ...]

    evidence_ids: tuple[str, ...]

    status: AlertRecommendationStatus

    requires_human_review: bool

    created_at: datetime

    assessment_id: str

    deduplication_key: str


@dataclass(frozen=True)
class AlertApprovalRecord:
    approval_id: str

    recommendation_id: str

    reviewer_id: str

    approved: bool

    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class AlertAgentOutput:
    run_id: str

    assessment: AlertRiskAssessment
    recommendation: AlertRecommendation

    duplicate_recommendation: bool

    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize_dataclass(self)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def serialize_dataclass(
    value: Any,
) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: serialize_dataclass(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            key: serialize_dataclass(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize_dataclass(item)
            for item in value
        ]

    return value


# ============================================================
# SOURCE VALIDATOR
# ============================================================

class AlertEvidenceValidator:
    def __init__(
        self,
        config: AlertIntelligenceConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        evidence: AlertEvidence,
        *,
        now: datetime,
    ) -> tuple[bool, str | None]:
        if evidence.observed_at > (
            now + timedelta(minutes=5)
        ):
            return (
                False,
                "Evidence timestamp is in the future",
            )

        age_minutes = (
            now - evidence.observed_at
        ).total_seconds() / 60.0

        if age_minutes > (
            self.config.maximum_source_age_minutes
        ):
            return (
                False,
                "Evidence is stale",
            )

        if evidence.reliability < (
            self.config.minimum_source_reliability
        ):
            return (
                False,
                "Evidence reliability is too low",
            )

        if not evidence.verified:
            return (
                False,
                "Evidence is not verified",
            )

        return True, None


# ============================================================
# SIGNAL EXTRACTOR
# ============================================================

class AlertSignalExtractor:
    def __init__(
        self,
        config: AlertIntelligenceConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def _signal(
        evidence: AlertEvidence,
        signal_type: AlertSignalType,
        severity: AlertRiskLevel,
        strength: float,
        explanation: str,
    ) -> AlertSignal:
        return AlertSignal(
            signal_id=generate_id(),
            signal_type=signal_type,
            source_id=evidence.source_id,
            evidence_id=evidence.evidence_id,
            severity=severity,
            strength=min(
                1.0,
                max(0.0, strength),
            ),
            explanation=explanation,
        )

    def extract(
        self,
        evidence: AlertEvidence,
    ) -> tuple[AlertSignal, ...]:
        signals = []

        measurements = evidence.measurements

        if evidence.source_type == (
            AlertSourceType.RAINFALL
        ):
            rainfall = measurements.get(
                "rainfall_24h_mm"
            )

            if rainfall is not None:
                if rainfall >= (
                    self.config.rainfall_24h_critical_mm
                ):
                    severity = AlertRiskLevel.CRITICAL

                elif rainfall >= (
                    self.config.rainfall_24h_high_mm
                ):
                    severity = AlertRiskLevel.HIGH

                elif rainfall >= (
                    self.config.rainfall_24h_moderate_mm
                ):
                    severity = AlertRiskLevel.MODERATE

                else:
                    severity = AlertRiskLevel.LOW

                signals.append(
                    self._signal(
                        evidence,
                        AlertSignalType.HEAVY_RAINFALL,
                        severity,
                        min(
                            1.0,
                            rainfall / max(
                                self.config.rainfall_24h_critical_mm,
                                1.0,
                            ),
                        ),
                        (
                            f"24-hour rainfall: "
                            f"{rainfall:.1f} mm"
                        ),
                    )
                )

        elif evidence.source_type == (
            AlertSourceType.RIVER
        ):
            rise_rate = measurements.get(
                "rise_rate_m_per_hour"
            )

            if rise_rate is not None:
                if rise_rate >= (
                    self.config.river_rise_high_m_per_hour
                ):
                    severity = AlertRiskLevel.HIGH

                elif rise_rate >= (
                    self.config.river_rise_moderate_m_per_hour
                ):
                    severity = AlertRiskLevel.MODERATE

                else:
                    severity = AlertRiskLevel.LOW

                signals.append(
                    self._signal(
                        evidence,
                        AlertSignalType.RIVER_RISE,
                        severity,
                        min(
                            1.0,
                            max(
                                0.0,
                                rise_rate
                                / self.config.river_rise_high_m_per_hour,
                            ),
                        ),
                        (
                            f"River rise rate: "
                            f"{rise_rate:.2f} m/hour"
                        ),
                    )
                )

            # Station-specific warning levels are
            # required to interpret river height.
            river_level = measurements.get(
                "river_level_m"
            )

            warning_level = measurements.get(
                "warning_level_m"
            )

            if (
                river_level is not None
                and warning_level is not None
                and warning_level > 0
            ):
                ratio = (
                    river_level / warning_level
                )

                if ratio >= 1.0:
                    severity = AlertRiskLevel.HIGH

                elif ratio >= 0.90:
                    severity = AlertRiskLevel.MODERATE

                else:
                    severity = AlertRiskLevel.LOW

                signals.append(
                    self._signal(
                        evidence,
                        AlertSignalType.HIGH_RIVER_LEVEL,
                        severity,
                        min(
                            1.0,
                            max(0.0, ratio),
                        ),
                        (
                            f"River level: {river_level:.2f} m; "
                            f"station warning level: "
                            f"{warning_level:.2f} m"
                        ),
                    )
                )

        elif evidence.source_type == (
            AlertSourceType.DAM
        ):
            storage = measurements.get(
                "storage_percent"
            )

            if storage is not None:
                if storage >= (
                    self.config.dam_storage_critical_percent
                ):
                    severity = AlertRiskLevel.CRITICAL

                elif storage >= (
                    self.config.dam_storage_high_percent
                ):
                    severity = AlertRiskLevel.HIGH

                else:
                    severity = AlertRiskLevel.LOW

                signals.append(
                    self._signal(
                        evidence,
                        AlertSignalType.HIGH_DAM_STORAGE,
                        severity,
                        min(
                            1.0,
                            max(
                                0.0,
                                storage / 100.0,
                            ),
                        ),
                        (
                            f"Reported dam storage: "
                            f"{storage:.1f}%"
                        ),
                    )
                )

            release = measurements.get(
                "release_cumecs"
            )

            if (
                release is not None
                and release > 0
            ):
                # A release volume alone does not
                # establish downstream flood severity.
                signals.append(
                    self._signal(
                        evidence,
                        AlertSignalType.DAM_RELEASE,
                        AlertRiskLevel.MODERATE,
                        0.50,
                        (
                            f"Reported dam release: "
                            f"{release:.1f} m³/s; "
                            "downstream impact unverified"
                        ),
                    )
                )

        elif evidence.source_type in (
            AlertSourceType.WEATHER_FORECAST,
            AlertSourceType.ML_FORECAST,
        ):
            probability = measurements.get(
                "flood_probability"
            )

            if probability is not None:
                if probability >= (
                    self.config.forecast_critical_probability
                ):
                    severity = AlertRiskLevel.CRITICAL

                elif probability >= (
                    self.config.forecast_high_probability
                ):
                    severity = AlertRiskLevel.HIGH

                elif probability >= 0.40:
                    severity = AlertRiskLevel.MODERATE

                else:
                    severity = AlertRiskLevel.LOW

                signals.append(
                    self._signal(
                        evidence,
                        AlertSignalType.FORECAST_RISK,
                        severity,
                        probability,
                        (
                            "Reported forecast flood "
                            f"probability: {probability:.1%}"
                        ),
                    )
                )

        elif evidence.source_type in (
            AlertSourceType.SATELLITE,
            AlertSourceType.DRONE,
        ):
            visual_fraction = measurements.get(
                "flood_coverage_fraction"
            )

            if visual_fraction is not None:
                if visual_fraction >= 0.60:
                    severity = AlertRiskLevel.HIGH

                elif visual_fraction >= 0.25:
                    severity = AlertRiskLevel.MODERATE

                else:
                    severity = AlertRiskLevel.LOW

                signals.append(
                    self._signal(
                        evidence,
                        AlertSignalType.VISUAL_FLOOD_EVIDENCE,
                        severity,
                        visual_fraction,
                        (
                            "Reported image flood coverage: "
                            f"{visual_fraction:.1%}; "
                            "not geographic flood extent"
                        ),
                    )
                )

        elif evidence.source_type == (
            AlertSourceType.FIELD_REPORT
        ):
            observed = measurements.get(
                "flood_observed"
            )

            if observed == 1.0:
                signals.append(
                    self._signal(
                        evidence,
                        AlertSignalType.FIELD_FLOOD_REPORT,
                        AlertRiskLevel.HIGH,
                        0.85,
                        "Verified field report indicates flooding",
                    )
                )

        return tuple(signals)


# ============================================================
# RISK SCORING
# ============================================================

class AlertRiskScoringEngine:
    SEVERITY_SCORE = {
        AlertRiskLevel.LOW: 0.15,
        AlertRiskLevel.MODERATE: 0.45,
        AlertRiskLevel.HIGH: 0.75,
        AlertRiskLevel.CRITICAL: 1.00,
        AlertRiskLevel.UNDETERMINED: 0.0,
    }

    @classmethod
    def score(
        cls,
        signals: Sequence[AlertSignal],
    ) -> float | None:
        if not signals:
            return None

        # Use the strongest signal per source to
        # avoid overcounting correlated indicators
        # from the same monitoring station.
        source_scores = {}

        for signal in signals:
            severity_score = (
                cls.SEVERITY_SCORE[
                    signal.severity
                ]
            )

            score = (
                0.75 * severity_score
                + 0.25 * signal.strength
            )

            source_scores[
                signal.source_id
            ] = max(
                source_scores.get(
                    signal.source_id,
                    0.0,
                ),
                score,
            )

        values = sorted(
            source_scores.values(),
            reverse=True,
        )

        # Conservative advisory aggregation.
        strongest = values[0]

        supporting = (
            sum(values[1:])
            / len(values[1:])
            if len(values) > 1
            else 0.0
        )

        result = (
            0.75 * strongest
            + 0.25 * supporting
        )

        return round(
            min(1.0, max(0.0, result)),
            4,
        )

    @staticmethod
    def classify(
        score: float | None,
    ) -> AlertRiskLevel:
        if score is None:
            return AlertRiskLevel.UNDETERMINED

        if score >= 0.85:
            return AlertRiskLevel.CRITICAL

        if score >= 0.65:
            return AlertRiskLevel.HIGH

        if score >= 0.35:
            return AlertRiskLevel.MODERATE

        return AlertRiskLevel.LOW


# ============================================================
# DISTRICT ASSESSMENT
# ============================================================

class DistrictAlertAssessmentEngine:
    def __init__(
        self,
        config: AlertIntelligenceConfig,
    ) -> None:
        self.config = config

        self.validator = (
            AlertEvidenceValidator(
                config
            )
        )

        self.signal_extractor = (
            AlertSignalExtractor(
                config
            )
        )

        self.scoring_engine = (
            AlertRiskScoringEngine()
        )

    def assess(
        self,
        district: str,
        evidence: Sequence[AlertEvidence],
        *,
        now: datetime | None = None,
    ) -> AlertRiskAssessment:
        if not district.strip():
            raise AlertValidationError(
                "District is required"
            )

        now = now or utc_now()

        if (
            now.tzinfo is None
            or now.utcoffset() is None
        ):
            raise AlertValidationError(
                "now must be timezone-aware"
            )

        accepted_evidence = []
        warnings = []

        for item in evidence:
            if (
                item.district.casefold()
                != district.casefold()
            ):
                continue

            accepted, reason = (
                self.validator.validate(
                    item,
                    now=now,
                )
            )

            if not accepted:
                warnings.append(
                    f"Evidence {item.evidence_id}: "
                    f"{reason}"
                )

                continue

            accepted_evidence.append(
                item
            )

        signals = []

        for item in accepted_evidence:
            signals.extend(
                self.signal_extractor.extract(
                    item
                )
            )

        independent_sources = {
            item.source_id
            for item in accepted_evidence
        }

        sufficient_evidence = (
            len(independent_sources)
            >= self.config.minimum_independent_sources
            and len(signals) > 0
        )

        if sufficient_evidence:
            score = self.scoring_engine.score(
                signals
            )

            risk_level = (
                self.scoring_engine.classify(
                    score
                )
            )

        else:
            score = None

            risk_level = (
                AlertRiskLevel.UNDETERMINED
            )

            warnings.append(
                "Insufficient independent verified sources"
            )

        warnings.extend([
            "Risk score is a screening heuristic, "
            "not a calibrated flood probability",
            "District-level scoring can hide "
            "local differences",
            "Human verification required before "
            "public alerts or emergency actions",
        ])

        return AlertRiskAssessment(
            assessment_id=generate_id(),
            district=district,
            risk_level=risk_level,
            risk_score=score,
            signals=tuple(signals),
            evidence_ids=tuple(
                item.evidence_id
                for item in accepted_evidence
            ),
            independent_source_count=len(
                independent_sources
            ),
            sufficient_evidence=sufficient_evidence,
            requires_human_review=True,
            warnings=tuple(warnings),
            assessed_at=now,
        )


# ============================================================
# RECOMMENDATION GENERATOR
# ============================================================

class AlertRecommendationGenerator:
    @staticmethod
    def generate(
        assessment: AlertRiskAssessment,
    ) -> AlertRecommendation:
        risk_level = assessment.risk_level

        actions_by_level = {
            AlertRiskLevel.LOW: (
                "Continue routine monitoring",
                "Check sensor availability and freshness",
            ),
            AlertRiskLevel.MODERATE: (
                "Request review by district monitoring staff",
                "Cross-check official rainfall and river bulletins",
                "Inspect affected locations where safe",
            ),
            AlertRiskLevel.HIGH: (
                "Prioritize human verification of high-risk locations",
                "Review official flood and weather warnings",
                "Prepare response resources subject to authorization",
            ),
            AlertRiskLevel.CRITICAL: (
                "Escalate immediately to authorized emergency officials",
                "Verify current observations and official warnings",
                "Request human assessment of response readiness",
            ),
            AlertRiskLevel.UNDETERMINED: (
                "Collect additional verified evidence",
                "Investigate unavailable or stale sources",
                "Do not infer that conditions are safe",
            ),
        }

        title = (
            f"{assessment.district}: "
            f"{risk_level.value.upper()} "
            "flood-risk review"
        )

        score_text = (
            f"{assessment.risk_score:.2f}"
            if assessment.risk_score is not None
            else "unavailable"
        )

        summary = (
            f"Advisory flood-risk screening for "
            f"{assessment.district}. "
            f"Risk level: {risk_level.value}. "
            f"Heuristic score: {score_text}. "
            f"Evidence sources: "
            f"{assessment.independent_source_count}. "
            "No public warning has been issued."
        )

        deduplication_key = hashlib.sha256(
            (
                assessment.district.casefold()
                + "|"
                + risk_level.value
            ).encode("utf-8")
        ).hexdigest()

        return AlertRecommendation(
            recommendation_id=generate_id(),
            district=assessment.district,
            risk_level=risk_level,
            risk_score=assessment.risk_score,
            title=title,
            summary=summary,
            recommended_actions=(
                actions_by_level[risk_level]
            ),
            evidence_ids=assessment.evidence_ids,
            status=(
                AlertRecommendationStatus.PENDING_REVIEW
            ),
            requires_human_review=True,
            created_at=utc_now(),
            assessment_id=assessment.assessment_id,
            deduplication_key=deduplication_key,
        )


# ============================================================
# DEDUPLICATION
# ============================================================

class AlertDeduplicationEngine:
    def __init__(
        self,
        config: AlertIntelligenceConfig,
    ) -> None:
        self.config = config

        self._recent: dict[
            str,
            datetime,
        ] = {}

    def is_duplicate(
        self,
        recommendation: AlertRecommendation,
        *,
        now: datetime | None = None,
    ) -> bool:
        now = now or utc_now()

        cutoff = (
            now
            - timedelta(
                minutes=(
                    self.config.deduplication_window_minutes
                )
            )
        )

        self._recent = {
            key: timestamp
            for key, timestamp in self._recent.items()
            if timestamp >= cutoff
        }

        previous = self._recent.get(
            recommendation.deduplication_key
        )

        duplicate = (
            previous is not None
            and previous >= cutoff
        )

        if not duplicate:
            self._recent[
                recommendation.deduplication_key
            ] = now

        return duplicate


# ============================================================
# HUMAN APPROVAL WORKFLOW
# ============================================================

class AlertHumanReviewManager:
    def __init__(self) -> None:
        self._reviews: dict[
            str,
            AlertApprovalRecord,
        ] = {}

    def review(
        self,
        recommendation: AlertRecommendation,
        *,
        reviewer_id: str,
        approved: bool,
        reason: str,
    ) -> AlertApprovalRecord:
        if not reviewer_id.strip():
            raise AlertApprovalError(
                "Reviewer ID is required"
            )

        if not reason.strip():
            raise AlertApprovalError(
                "Review reason is required"
            )

        if recommendation.recommendation_id in (
            self._reviews
        ):
            raise AlertApprovalError(
                "Recommendation already reviewed"
            )

        record = AlertApprovalRecord(
            approval_id=generate_id(),
            recommendation_id=(
                recommendation.recommendation_id
            ),
            reviewer_id=reviewer_id,
            approved=approved,
            reason=reason,
            reviewed_at=utc_now(),
        )

        self._reviews[
            recommendation.recommendation_id
        ] = record

        return record

    def get_review(
        self,
        recommendation_id: str,
    ) -> AlertApprovalRecord | None:
        return self._reviews.get(
            recommendation_id
        )


# ============================================================
# MAIN AGENT
# ============================================================

class AlertIntelligenceAgent:
    def __init__(
        self,
        config: AlertIntelligenceConfig | None = None,
    ) -> None:
        self.config = (
            config or AlertIntelligenceConfig()
        )

        self.assessment_engine = (
            DistrictAlertAssessmentEngine(
                self.config
            )
        )

        self.recommendation_generator = (
            AlertRecommendationGenerator()
        )

        self.deduplication_engine = (
            AlertDeduplicationEngine(
                self.config
            )
        )

        self.review_manager = (
            AlertHumanReviewManager()
        )

    def run(
        self,
        *,
        district: str,
        evidence: Sequence[AlertEvidence],
        now: datetime | None = None,
    ) -> AlertAgentOutput:
        assessment = (
            self.assessment_engine.assess(
                district,
                evidence,
                now=now,
            )
        )

        recommendation = (
            self.recommendation_generator.generate(
                assessment
            )
        )

        duplicate = (
            self.deduplication_engine.is_duplicate(
                recommendation,
                now=now,
            )
        )

        return AlertAgentOutput(
            run_id=generate_id(),
            assessment=assessment,
            recommendation=recommendation,
            duplicate_recommendation=duplicate,
            generated_at=utc_now(),
        )

    def run_multiple_districts(
        self,
        evidence: Sequence[AlertEvidence],
        *,
        now: datetime | None = None,
    ) -> dict[str, AlertAgentOutput]:
        districts = sorted({
            item.district
            for item in evidence
        })

        return {
            district: self.run(
                district=district,
                evidence=evidence,
                now=now,
            )
            for district in districts
        }


# ============================================================
# JSON REPORT EXPORTER
# ============================================================

class AlertIntelligenceReportExporter:
    @staticmethod
    def export_json(
        output: AlertAgentOutput,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise AlertExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                output.to_dict(),
                stream,
                indent=2,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_alert_intelligence_agent(
    config: AlertIntelligenceConfig | None = None,
) -> AlertIntelligenceAgent:
    return AlertIntelligenceAgent(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    now = utc_now()

    evidence = [
        AlertEvidence(
            evidence_id="RAIN-001",
            source_type=AlertSourceType.RAINFALL,
            source_id="STATION-RAIN-001",
            district="Thrissur",
            observed_at=now - timedelta(
                minutes=15
            ),
            measurements={
                "rainfall_24h_mm": 125.0,
            },
            reliability=0.95,
            verified=True,
        ),
        AlertEvidence(
            evidence_id="RIVER-001",
            source_type=AlertSourceType.RIVER,
            source_id="STATION-RIVER-001",
            district="Thrissur",
            observed_at=now - timedelta(
                minutes=10
            ),
            measurements={
                "river_level_m": 8.2,
                "warning_level_m": 8.0,
                "rise_rate_m_per_hour": 0.45,
            },
            reliability=0.96,
            verified=True,
        ),
        AlertEvidence(
            evidence_id="FORECAST-001",
            source_type=(
                AlertSourceType.ML_FORECAST
            ),
            source_id="MODEL-FLOOD-001",
            district="Thrissur",
            observed_at=now - timedelta(
                minutes=5
            ),
            measurements={
                "flood_probability": 0.86,
            },
            reliability=0.85,
            verified=True,
        ),
    ]

    agent = create_alert_intelligence_agent()

    output = agent.run(
        district="Thrissur",
        evidence=evidence,
        now=now,
    )

    print(
        json.dumps(
            output.to_dict(),
            indent=2,
        )
    )

    AlertIntelligenceReportExporter.export_json(
        output,
        "outputs/agents/alert_intelligence_report.json",
    )

    # Example human review record.
    # This does NOT publish any emergency alert.
    review = agent.review_manager.review(
        output.recommendation,
        reviewer_id="authorized-reviewer-001",
        approved=False,
        reason=(
            "Awaiting independent official "
            "confirmation"
        ),
    )

    print(
        json.dumps(
            serialize_dataclass(review),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "AlertIntelligenceError",
    "AlertValidationError",
    "AlertSourceError",
    "AlertApprovalError",
    "AlertExportError",
    "AlertSourceType",
    "AlertRiskLevel",
    "AlertRecommendationStatus",
    "AlertSignalType",
    "AlertIntelligenceConfig",
    "AlertEvidence",
    "AlertSignal",
    "AlertRiskAssessment",
    "AlertRecommendation",
    "AlertApprovalRecord",
    "AlertAgentOutput",
    "AlertEvidenceValidator",
    "AlertSignalExtractor",
    "AlertRiskScoringEngine",
    "DistrictAlertAssessmentEngine",
    "AlertRecommendationGenerator",
    "AlertDeduplicationEngine",
    "AlertHumanReviewManager",
    "AlertIntelligenceAgent",
    "AlertIntelligenceReportExporter",
    "create_alert_intelligence_agent",
]
