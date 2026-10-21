"""Placeholder module: ai/agents/verification_agent.py; implementation required."""

"""
backend/ai/agents/verification_agent.py

Kerala Flood Relief Coordination Agent
Multi-Agent Verification & Safety Intelligence

Python 3.11+
Standard library only.

FEATURES
--------
1. Multi-agent output verification
2. Evidence provenance and reliability checks
3. Source freshness validation
4. Claim-to-evidence grounding
5. Cross-agent contradiction detection
6. Rule-based hallucination-risk screening
7. Confidence and uncertainty assessment
8. Safety-critical action detection
9. Human verification workflow
10. Approval and rejection audit trail
11. Privacy-conscious report export
12. Supervisor integration support

IMPORTANT
---------
This module is advisory decision support.

A verified result means the configured checks
passed. It does NOT prove that the underlying
real-world facts are correct.

No automated rescue dispatch, evacuation order,
medical treatment, public warning, infrastructure
control, shelter admission, or resource allocation
is authorized by this module.

All safety-critical decisions require authorized
human review.

This is a reference implementation, not a
certified emergency-management system.
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

class VerificationError(Exception):
    pass


class VerificationValidationError(VerificationError):
    pass


class VerificationPermissionError(VerificationError):
    pass


class VerificationStateError(VerificationError):
    pass


class VerificationNotFoundError(VerificationError):
    pass


class VerificationExportError(VerificationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class VerificationStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    PASSED_CHECKS = "passed_checks"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    CONFLICT_DETECTED = "conflict_detected"
    SAFETY_BLOCKED = "safety_blocked"
    HUMAN_CONFIRMED = "human_confirmed"
    HUMAN_REJECTED = "human_rejected"


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class ClaimCriticality(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    LIFE_CRITICAL = "life_critical"


class ClaimType(str, Enum):
    FLOOD_RISK = "flood_risk"
    FLOOD_FORECAST = "flood_forecast"
    SOS_URGENCY = "sos_urgency"
    RESCUE_REQUIREMENT = "rescue_requirement"
    ROUTE_SAFETY = "route_safety"
    SHELTER_CAPACITY = "shelter_capacity"
    MEDICAL_NEED = "medical_need"
    RESOURCE_AVAILABILITY = "resource_availability"
    DAMAGE_ASSESSMENT = "damage_assessment"
    MISSING_PERSON_STATUS = "missing_person_status"
    RECOVERY_COST = "recovery_cost"
    GENERAL = "general"


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewDecision(str, Enum):
    CONFIRM_CHECKS = "confirm_checks"
    REJECT = "reject"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"


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
        raise VerificationValidationError(
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
        raise VerificationValidationError(
            f"{name} must be between 0 and 1"
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

    if isinstance(value, (tuple, list)):
        return [
            serialize(item)
            for item in value
        ]

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, float) and math.isfinite(value):
        return value

    raise VerificationValidationError(
        "Unsupported report value"
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class VerificationConfig:
    max_claims: int = 1000
    max_evidence: int = 5000

    max_evidence_age_minutes: int = 120
    max_agent_output_age_minutes: int = 60

    minimum_source_reliability: float = 0.70
    minimum_grounding_score: float = 0.70

    require_independent_sources_for_critical: int = 2

    require_human_review: bool = True
    allow_automatic_operational_approval: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_claims",
            "max_evidence",
            "max_evidence_age_minutes",
            "max_agent_output_age_minutes",
            "require_independent_sources_for_critical",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise VerificationValidationError(
                    f"{name} must be positive integer"
                )

        validate_probability(
            self.minimum_source_reliability,
            "minimum_source_reliability",
        )

        validate_probability(
            self.minimum_grounding_score,
            "minimum_grounding_score",
        )

        if self.allow_automatic_operational_approval:
            raise VerificationValidationError(
                "Automatic operational approval is prohibited"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class VerificationEvidence:
    evidence_id: str
    source_id: str

    description: str

    status: EvidenceStatus
    reliability: float

    observed_at: datetime

    supported_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise VerificationValidationError(
                "evidence_id required"
            )

        if not self.source_id.strip():
            raise VerificationValidationError(
                "source_id required"
            )

        validate_probability(
            self.reliability,
            "reliability",
        )

        validate_datetime(
            self.observed_at
        )


@dataclass(frozen=True)
class VerificationClaim:
    claim_id: str
    agent_id: str

    claim_type: ClaimType
    statement: str

    criticality: ClaimCriticality

    confidence: float

    evidence_ids: tuple[str, ...]

    # Shared key identifying the same factual question
    # across different agents.
    consistency_key: str | None = None

    # A normalized categorical value, e.g.
    # "open", "closed", "unknown".
    normalized_value: str | None = None

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise VerificationValidationError(
                "claim_id required"
            )

        if not self.agent_id.strip():
            raise VerificationValidationError(
                "agent_id required"
            )

        if not self.statement.strip():
            raise VerificationValidationError(
                "claim statement required"
            )

        validate_probability(
            self.confidence,
            "confidence",
        )


@dataclass(frozen=True)
class AgentOutputRecord:
    agent_id: str
    task_id: str

    generated_at: datetime

    claims: tuple[VerificationClaim, ...]

    advisory_only: bool = True

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise VerificationValidationError(
                "agent_id required"
            )

        if not self.task_id.strip():
            raise VerificationValidationError(
                "task_id required"
            )

        validate_datetime(
            self.generated_at
        )


@dataclass(frozen=True)
class VerificationRequest:
    request_id: str
    workflow_id: str

    district: str

    agent_outputs: tuple[AgentOutputRecord, ...]
    evidence: tuple[VerificationEvidence, ...]

    created_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "workflow_id",
            "district",
        ):
            if not getattr(self, name).strip():
                raise VerificationValidationError(
                    f"{name} required"
                )

        validate_datetime(
            self.created_at
        )


@dataclass(frozen=True)
class VerificationFinding:
    finding_id: str

    claim_id: str | None

    severity: FindingSeverity
    category: str

    message: str


@dataclass(frozen=True)
class ClaimVerificationResult:
    claim_id: str
    agent_id: str

    grounding_score: float

    accepted_evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]

    independent_source_count: int

    passed_grounding_checks: bool

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class VerificationReport:
    report_id: str

    request_id: str
    workflow_id: str

    district: str

    claim_results: tuple[
        ClaimVerificationResult,
        ...
    ]

    findings: tuple[VerificationFinding, ...]

    status: VerificationStatus

    passed_claim_count: int
    failed_claim_count: int

    requires_human_review: bool

    warnings: tuple[str, ...]

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class VerificationReviewRecord:
    review_id: str
    report_id: str

    reviewer_id: str
    reviewer_role: str

    decision: ReviewDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class VerificationAuditRecord:
    audit_id: str
    report_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# REQUEST VALIDATOR
# ============================================================

class VerificationRequestValidator:
    def __init__(
        self,
        config: VerificationConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        request: VerificationRequest,
        *,
        now: datetime,
    ) -> None:
        if request.created_at > now:
            raise VerificationValidationError(
                "Request timestamp is in the future"
            )

        claims = [
            claim
            for output in request.agent_outputs
            for claim in output.claims
        ]

        if len(claims) > self.config.max_claims:
            raise VerificationValidationError(
                "Claim limit exceeded"
            )

        if len(request.evidence) > (
            self.config.max_evidence
        ):
            raise VerificationValidationError(
                "Evidence limit exceeded"
            )

        claim_ids = [
            claim.claim_id
            for claim in claims
        ]

        evidence_ids = [
            item.evidence_id
            for item in request.evidence
        ]

        if len(claim_ids) != len(set(claim_ids)):
            raise VerificationValidationError(
                "Duplicate claim ID"
            )

        if len(evidence_ids) != len(set(evidence_ids)):
            raise VerificationValidationError(
                "Duplicate evidence ID"
            )

        for output in request.agent_outputs:
            if output.generated_at > now:
                raise VerificationValidationError(
                    "Future agent output timestamp"
                )

            for claim in output.claims:
                if claim.agent_id != output.agent_id:
                    raise VerificationValidationError(
                        "Claim agent ID mismatch"
                    )


# ============================================================
# EVIDENCE VALIDATOR
# ============================================================

class VerificationEvidenceEngine:
    def __init__(
        self,
        config: VerificationConfig,
    ) -> None:
        self.config = config

    def accepted_evidence(
        self,
        evidence: Sequence[VerificationEvidence],
        *,
        now: datetime,
    ) -> dict[str, VerificationEvidence]:
        accepted = {}

        for item in evidence:
            if item.status != EvidenceStatus.VERIFIED:
                continue

            if (
                item.reliability
                < self.config.minimum_source_reliability
            ):
                continue

            if item.observed_at > now:
                continue

            if (
                now - item.observed_at
            ) > timedelta(
                minutes=self.config.max_evidence_age_minutes
            ):
                continue

            accepted[
                item.evidence_id
            ] = item

        return accepted


# ============================================================
# CLAIM GROUNDING ENGINE
# ============================================================

class ClaimGroundingEngine:
    """
    Structural grounding checks only.

    Evidence IDs and support declarations are
    supplied by upstream systems and are not
    independently authenticated here.

    A passing score is NOT proof that the
    evidence semantically entails the claim.
    """

    def __init__(
        self,
        config: VerificationConfig,
    ) -> None:
        self.config = config

    def assess(
        self,
        claim: VerificationClaim,
        accepted: Mapping[
            str,
            VerificationEvidence,
        ],
    ) -> ClaimVerificationResult:
        valid = []
        invalid = []
        source_ids = set()

        for evidence_id in dict.fromkeys(
            claim.evidence_ids
        ):
            item = accepted.get(
                evidence_id
            )

            if (
                item is None
                or claim.claim_id
                not in item.supported_claim_ids
            ):
                invalid.append(
                    evidence_id
                )

                continue

            valid.append(
                evidence_id
            )

            source_ids.add(
                item.source_id
            )

        total = len(
            set(claim.evidence_ids)
        )

        grounding_score = (
            len(valid) / total
            if total > 0
            else 0.0
        )

        is_critical = claim.criticality in {
            ClaimCriticality.HIGH,
            ClaimCriticality.LIFE_CRITICAL,
        }

        source_requirement = (
            self.config.require_independent_sources_for_critical
            if is_critical
            else 1
        )

        passed = (
            grounding_score
            >= self.config.minimum_grounding_score
            and len(source_ids)
            >= source_requirement
        )

        warnings = []

        if not valid:
            warnings.append(
                "No accepted evidence supports this claim"
            )

        if invalid:
            warnings.append(
                "Some evidence references are invalid, "
                "stale, unverified, or do not declare support"
            )

        if len(source_ids) < source_requirement:
            warnings.append(
                "Insufficient independent source IDs"
            )

        warnings.append(
            "Evidence support has not been "
            "semantically or independently verified"
        )

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            agent_id=claim.agent_id,
            grounding_score=round(
                grounding_score,
                4,
            ),
            accepted_evidence_ids=tuple(valid),
            rejected_evidence_ids=tuple(invalid),
            independent_source_count=len(
                source_ids
            ),
            passed_grounding_checks=passed,
            warnings=tuple(warnings),
        )


# ============================================================
# CROSS-AGENT CONSISTENCY ENGINE
# ============================================================

class CrossAgentConsistencyEngine:
    """
    Detects contradictory normalized values
    sharing the same consistency key.

    This is exact-value comparison, not semantic
    contradiction detection.
    """

    @staticmethod
    def check(
        claims: Sequence[VerificationClaim],
    ) -> tuple[VerificationFinding, ...]:
        groups: dict[
            str,
            list[VerificationClaim],
        ] = {}

        for claim in claims:
            if (
                not claim.consistency_key
                or claim.normalized_value is None
            ):
                continue

            groups.setdefault(
                claim.consistency_key,
                [],
            ).append(
                claim
            )

        findings = []

        for key, group in groups.items():
            values = {
                claim.normalized_value.strip().casefold()
                for claim in group
            }

            if len(values) <= 1:
                continue

            agents = {
                claim.agent_id
                for claim in group
            }

            if len(agents) < 2:
                continue

            findings.append(
                VerificationFinding(
                    finding_id=generate_id(),
                    claim_id=None,
                    severity=FindingSeverity.HIGH,
                    category="cross_agent_conflict",
                    message=(
                        f"Conflicting normalized values "
                        f"for consistency key '{key}'"
                    ),
                )
            )

        return tuple(findings)


# ============================================================
# SAFETY RULE ENGINE
# ============================================================

class VerificationSafetyEngine:
    """
    Detects certain prohibited operational claims.

    Rules are deliberately conservative.
    These checks do not replace policy review,
    clinical review, or incident command.
    """

    UNSAFE_PHRASES = (
        "automatically dispatch",
        "dispatch completed",
        "evacuation order issued",
        "automatically evacuate",
        "administer medication",
        "treatment approved",
        "shelter admission confirmed",
        "route guaranteed safe",
        "person confirmed deceased",
        "person confirmed rescued",
        "automatically release funds",
        "open dam gates",
        "close dam gates",
    )

    @classmethod
    def check(
        cls,
        claims: Sequence[VerificationClaim],
    ) -> tuple[VerificationFinding, ...]:
        findings = []

        for claim in claims:
            text = claim.statement.casefold()

            matched = [
                phrase
                for phrase in cls.UNSAFE_PHRASES
                if phrase in text
            ]

            if not matched:
                continue

            findings.append(
                VerificationFinding(
                    finding_id=generate_id(),
                    claim_id=claim.claim_id,
                    severity=FindingSeverity.CRITICAL,
                    category="operational_safety",
                    message=(
                        "Claim contains a possible "
                        "unauthorized operational action; "
                        "human investigation required"
                    ),
                )
            )

        return tuple(findings)


# ============================================================
# HALLUCINATION-RISK SCREENING
# ============================================================

class HallucinationRiskEngine:
    """
    Flags unsupported or overconfident claims.

    Does not calculate actual hallucination
    probability or establish factual truth.
    """

    @staticmethod
    def check(
        claims: Sequence[VerificationClaim],
        results: Mapping[
            str,
            ClaimVerificationResult,
        ],
    ) -> tuple[VerificationFinding, ...]:
        findings = []

        for claim in claims:
            result = results[
                claim.claim_id
            ]

            if (
                claim.confidence >= 0.90
                and not result.passed_grounding_checks
            ):
                findings.append(
                    VerificationFinding(
                        finding_id=generate_id(),
                        claim_id=claim.claim_id,
                        severity=FindingSeverity.HIGH,
                        category="overconfidence",
                        message=(
                            "High-confidence claim lacks "
                            "sufficient grounding"
                        ),
                    )
                )

            if (
                claim.criticality
                == ClaimCriticality.LIFE_CRITICAL
                and not result.passed_grounding_checks
            ):
                findings.append(
                    VerificationFinding(
                        finding_id=generate_id(),
                        claim_id=claim.claim_id,
                        severity=FindingSeverity.CRITICAL,
                        category="unsupported_life_critical_claim",
                        message=(
                            "Life-critical claim requires "
                            "additional independent verification"
                        ),
                    )
                )

        return tuple(findings)


# ============================================================
# AUDIT LOGGER
# ============================================================

class VerificationAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            VerificationAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        report_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> VerificationAuditRecord:
        record = VerificationAuditRecord(
            audit_id=generate_id(),
            report_id=report_id,
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
        report_id: str | None = None,
    ) -> tuple[VerificationAuditRecord, ...]:
        with self._lock:
            records = tuple(
                self._records
            )

        if report_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.report_id == report_id
        )


# ============================================================
# HUMAN REVIEW MANAGER
# ============================================================

class VerificationHumanReviewManager:
    ALLOWED_ROLES = {
        "incident_commander",
        "district_disaster_management_officer",
        "authorized_verification_officer",
        "safety_review_officer",
    }

    def __init__(
        self,
        audit_logger: VerificationAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            VerificationReviewRecord,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        report: VerificationReport,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> VerificationReviewRecord:
        if not reviewer_id.strip():
            raise VerificationPermissionError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise VerificationPermissionError(
                "Unauthorized reviewer role"
            )

        if not reason.strip():
            raise VerificationValidationError(
                "Review reason required"
            )

        if (
            decision == ReviewDecision.CONFIRM_CHECKS
            and report.status
            != VerificationStatus.PASSED_CHECKS
        ):
            raise VerificationStateError(
                "Only reports passing configured checks "
                "can be confirmed through this workflow"
            )

        with self._lock:
            if report.report_id in self._reviews:
                raise VerificationStateError(
                    "Verification report already reviewed"
                )

            review = VerificationReviewRecord(
                review_id=generate_id(),
                report_id=report.report_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                report.report_id
            ] = review

        self.audit_logger.record(
            report_id=report.report_id,
            actor_id=reviewer_id,
            action=f"human_review_{decision.value}",
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN VERIFICATION AGENT
# ============================================================

class VerificationAgent:
    def __init__(
        self,
        config: VerificationConfig | None = None,
    ) -> None:
        self.config = (
            config or VerificationConfig()
        )

        self.request_validator = (
            VerificationRequestValidator(
                self.config
            )
        )

        self.evidence_engine = (
            VerificationEvidenceEngine(
                self.config
            )
        )

        self.grounding_engine = (
            ClaimGroundingEngine(
                self.config
            )
        )

        self.consistency_engine = (
            CrossAgentConsistencyEngine()
        )

        self.safety_engine = (
            VerificationSafetyEngine()
        )

        self.hallucination_engine = (
            HallucinationRiskEngine()
        )

        self.audit_logger = (
            VerificationAuditLogger()
        )

        self.review_manager = (
            VerificationHumanReviewManager(
                self.audit_logger
            )
        )

        self._reports: dict[
            str,
            VerificationReport,
        ] = {}

        self._lock = threading.RLock()

    def verify(
        self,
        request: VerificationRequest,
        *,
        now: datetime | None = None,
    ) -> VerificationReport:
        now = now or utc_now()

        validate_datetime(now)

        self.request_validator.validate(
            request,
            now=now,
        )

        claims = [
            claim
            for output in request.agent_outputs
            for claim in output.claims
        ]

        accepted = (
            self.evidence_engine.accepted_evidence(
                request.evidence,
                now=now,
            )
        )

        results = {}

        for claim in claims:
            results[
                claim.claim_id
            ] = self.grounding_engine.assess(
                claim,
                accepted,
            )

        findings = []

        for output in request.agent_outputs:
            if not output.advisory_only:
                findings.append(
                    VerificationFinding(
                        finding_id=generate_id(),
                        claim_id=None,
                        severity=FindingSeverity.CRITICAL,
                        category="non_advisory_output",
                        message=(
                            "Agent output is marked as "
                            "non-advisory and requires "
                            "operational safety investigation"
                        ),
                    )
                )

            if (
                now - output.generated_at
            ) > timedelta(
                minutes=(
                    self.config.max_agent_output_age_minutes
                )
            ):
                findings.append(
                    VerificationFinding(
                        finding_id=generate_id(),
                        claim_id=None,
                        severity=FindingSeverity.HIGH,
                        category="stale_agent_output",
                        message=(
                            f"Agent output '{output.agent_id}' "
                            "is stale"
                        ),
                    )
                )

        findings.extend(
            self.consistency_engine.check(
                claims
            )
        )

        findings.extend(
            self.safety_engine.check(
                claims
            )
        )

        findings.extend(
            self.hallucination_engine.check(
                claims,
                results,
            )
        )

        failed_count = sum(
            not result.passed_grounding_checks
            for result in results.values()
        )

        passed_count = (
            len(results) - failed_count
        )

        has_critical = any(
            finding.severity == FindingSeverity.CRITICAL
            for finding in findings
        )

        has_conflict = any(
            finding.category == "cross_agent_conflict"
            for finding in findings
        )

        if has_critical:
            status = VerificationStatus.SAFETY_BLOCKED

        elif has_conflict:
            status = VerificationStatus.CONFLICT_DETECTED

        elif (
            failed_count > 0
            or any(
                finding.severity == FindingSeverity.HIGH
                for finding in findings
            )
            or not claims
        ):
            status = VerificationStatus.NEEDS_MORE_EVIDENCE

        else:
            status = VerificationStatus.PASSED_CHECKS

        warnings = [
            "Passing checks does not establish factual truth",
            "Evidence provenance is not independently authenticated",
            "Grounding checks are structural, not semantic",
            "Human review is required",
            "No operational action has been authorized",
        ]

        if failed_count:
            warnings.append(
                "Some claims lack sufficient accepted evidence"
            )

        if has_conflict:
            warnings.append(
                "Cross-agent contradictions require resolution"
            )

        if has_critical:
            warnings.append(
                "Safety-critical finding blocks confirmation"
            )

        report = VerificationReport(
            report_id=generate_id(),
            request_id=request.request_id,
            workflow_id=request.workflow_id,
            district=request.district,
            claim_results=tuple(
                results.values()
            ),
            findings=tuple(findings),
            status=status,
            passed_claim_count=passed_count,
            failed_claim_count=failed_count,
            requires_human_review=True,
            warnings=tuple(
                dict.fromkeys(warnings)
            ),
            created_at=now,
        )

        with self._lock:
            self._reports[
                report.report_id
            ] = report

        self.audit_logger.record(
            report_id=report.report_id,
            actor_id="verification_agent",
            action="verification_completed",
            details={
                "workflow_id": request.workflow_id,
                "status": status.value,
                "passed_claims": str(passed_count),
                "failed_claims": str(failed_count),
                "finding_count": str(len(findings)),
            },
        )

        return report

    def review_report(
        self,
        report_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: ReviewDecision,
        reason: str,
    ) -> VerificationReport:
        with self._lock:
            report = self._reports.get(
                report_id
            )

            if report is None:
                raise VerificationNotFoundError(
                    "Verification report not found"
                )

            if report.status in {
                VerificationStatus.HUMAN_CONFIRMED,
                VerificationStatus.HUMAN_REJECTED,
            }:
                raise VerificationStateError(
                    "Report has already received final review"
                )

            self.review_manager.review(
                report,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            if decision == ReviewDecision.CONFIRM_CHECKS:
                status = VerificationStatus.HUMAN_CONFIRMED

            elif decision == ReviewDecision.REJECT:
                status = VerificationStatus.HUMAN_REJECTED

            else:
                status = VerificationStatus.NEEDS_MORE_EVIDENCE

            updated = replace(
                report,
                status=status,
            )

            self._reports[
                report_id
            ] = updated

            return updated

    def get_report(
        self,
        report_id: str,
    ) -> VerificationReport | None:
        with self._lock:
            return self._reports.get(
                report_id
            )

    def get_audit_history(
        self,
        report_id: str,
    ) -> tuple[VerificationAuditRecord, ...]:
        return self.audit_logger.list_records(
            report_id
        )


# ============================================================
# JSON EXPORTER
# ============================================================

class VerificationReportExporter:
    @staticmethod
    def export_json(
        report: VerificationReport,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise VerificationExportError(
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
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_verification_agent(
    config: VerificationConfig | None = None,
) -> VerificationAgent:
    return VerificationAgent(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    now = utc_now()

    # Fictional demonstration data.
    # These are not verified flood observations.

    claim = VerificationClaim(
        claim_id="DEMO-CLAIM-001",
        agent_id="risk_assessment_agent",
        claim_type=ClaimType.FLOOD_RISK,
        statement=(
            "Illustrative zone has elevated flood risk"
        ),
        criticality=ClaimCriticality.HIGH,
        confidence=0.82,
        evidence_ids=(
            "DEMO-EVIDENCE-001",
            "DEMO-EVIDENCE-002",
        ),
        consistency_key="demo_zone_flood_risk",
        normalized_value="high",
    )

    request = VerificationRequest(
        request_id="DEMO-VERIFY-REQUEST-001",
        workflow_id="DEMO-WORKFLOW-001",
        district="Thrissur",
        agent_outputs=(
            AgentOutputRecord(
                agent_id="risk_assessment_agent",
                task_id="risk_assessment",
                generated_at=now,
                claims=(claim,),
            ),
        ),
        evidence=(
            VerificationEvidence(
                evidence_id="DEMO-EVIDENCE-001",
                source_id="DEMO-SOURCE-A",
                description="Illustrative rainfall reading",
                status=EvidenceStatus.VERIFIED,
                reliability=0.90,
                observed_at=now - timedelta(minutes=10),
                supported_claim_ids=(
                    "DEMO-CLAIM-001",
                ),
            ),
            VerificationEvidence(
                evidence_id="DEMO-EVIDENCE-002",
                source_id="DEMO-SOURCE-B",
                description="Illustrative river observation",
                status=EvidenceStatus.VERIFIED,
                reliability=0.92,
                observed_at=now - timedelta(minutes=8),
                supported_claim_ids=(
                    "DEMO-CLAIM-001",
                ),
            ),
        ),
        created_at=now,
    )

    agent = create_verification_agent()

    report = agent.verify(
        request,
        now=now,
    )

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    if report.status == VerificationStatus.PASSED_CHECKS:
        reviewed = agent.review_report(
            report.report_id,
            reviewer_id="DEMO-VERIFICATION-OFFICER",
            reviewer_role="authorized_verification_officer",
            decision=ReviewDecision.CONFIRM_CHECKS,
            reason="Fictional verification demonstration",
        )

        print(
            "Review status:",
            reviewed.status.value,
        )

        VerificationReportExporter.export_json(
            reviewed,
            "outputs/agents/verification_report.json",
        )

    print(
        "No real-world flood decision or "
        "emergency action was authorized."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "VerificationError",
    "VerificationValidationError",
    "VerificationPermissionError",
    "VerificationStateError",
    "VerificationNotFoundError",
    "VerificationExportError",
    "VerificationStatus",
    "EvidenceStatus",
    "ClaimCriticality",
    "ClaimType",
    "FindingSeverity",
    "ReviewDecision",
    "VerificationConfig",
    "VerificationEvidence",
    "VerificationClaim",
    "AgentOutputRecord",
    "VerificationRequest",
    "VerificationFinding",
    "ClaimVerificationResult",
    "VerificationReport",
    "VerificationReviewRecord",
    "VerificationAuditRecord",
    "VerificationRequestValidator",
    "VerificationEvidenceEngine",
    "ClaimGroundingEngine",
    "CrossAgentConsistencyEngine",
    "VerificationSafetyEngine",
    "HallucinationRiskEngine",
    "VerificationAuditLogger",
    "VerificationHumanReviewManager",
    "VerificationAgent",
    "VerificationReportExporter",
    "create_verification_agent",
]
