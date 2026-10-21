"""Placeholder module: ai/agents/missing_person_agent.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
backend/ai/agents/missing_person_agent.py

Missing Person Search & Reunification Coordination Agent
========================================================

Features:
    - Missing-person case registration
    - Flood-related missing-person reporting
    - Last-known-location tracking
    - Witness and sighting report intake
    - Shelter and hospital inquiry coordination
    - Possible identity-match screening
    - Duplicate case detection
    - Search priority assessment
    - Vulnerable-person support
    - Family reunification workflow
    - Evidence verification
    - Privacy and consent safeguards
    - Human-reviewed case matching
    - Case status management
    - Audit logging
    - JSON report export

Safety and privacy:
    - No facial recognition or biometric matching.
    - No automated identity confirmation.
    - No automatic publication of personal information.
    - No automatic rescue dispatch.
    - No automatic disclosure of a person's location.
    - No automatic family reunification confirmation.
    - Possible matches require authorized human verification.
    - An adult's location must not be disclosed without
      an appropriate lawful basis and safeguarding review.
    - Child protection and vulnerable-person procedures
      must be followed.

Python 3.11+
Standard library only.
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

class MissingPersonAgentError(Exception):
    pass


class MissingPersonValidationError(MissingPersonAgentError):
    pass


class MissingPersonPermissionError(MissingPersonAgentError):
    pass


class MissingPersonStateError(MissingPersonAgentError):
    pass


class MissingPersonNotFoundError(MissingPersonAgentError):
    pass


class MissingPersonExportError(MissingPersonAgentError):
    pass


# ============================================================
# ENUMS
# ============================================================

class MissingPersonStatus(str, Enum):
    REPORTED = "reported"
    UNDER_REVIEW = "under_review"
    SEARCH_COORDINATION = "search_coordination"
    POSSIBLE_MATCH = "possible_match"
    LOCATED_PENDING_VERIFICATION = (
        "located_pending_verification"
    )
    LOCATED_CONFIRMED = "located_confirmed"
    REUNIFICATION_PENDING = "reunification_pending"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class MissingPersonPriority(str, Enum):
    ROUTINE_REVIEW = "routine_review"
    ELEVATED_REVIEW = "elevated_review"
    HIGH_PRIORITY_REVIEW = "high_priority_review"
    URGENT_HUMAN_ESCALATION = (
        "urgent_human_escalation"
    )


class SightingSourceType(str, Enum):
    WITNESS = "witness"
    SHELTER = "shelter"
    HOSPITAL = "hospital"
    RESCUE_TEAM = "rescue_team"
    AUTHORIZED_OFFICIAL = "authorized_official"


class EvidenceVerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class MatchReviewStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ReunificationStatus(str, Enum):
    NOT_STARTED = "not_started"
    PENDING_SAFEGUARDING_REVIEW = (
        "pending_safeguarding_review"
    )
    APPROVED_FOR_COORDINATION = (
        "approved_for_coordination"
    )
    COMPLETED = "completed"
    BLOCKED = "blocked"


class AuthorizedRole(str, Enum):
    MISSING_PERSON_COORDINATOR = (
        "missing_person_coordinator"
    )
    DISTRICT_OFFICIAL = "district_official"
    RESCUE_COORDINATOR = "rescue_coordinator"
    CHILD_PROTECTION_OFFICER = (
        "child_protection_officer"
    )
    SAFEGUARDING_OFFICER = "safeguarding_officer"
    HEALTHCARE_LIAISON = "healthcare_liaison"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MissingPersonAgentConfig:
    maximum_sighting_age_days: int = 30

    minimum_sighting_reliability: float = 0.65

    maximum_sightings_per_case: int = 100

    maximum_match_candidates: int = 100

    duplicate_window_hours: int = 72

    maximum_age_years: int = 125

    require_human_match_verification: bool = True

    require_safeguarding_review: bool = True

    allow_public_location_disclosure: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_sighting_age_days",
            "maximum_sightings_per_case",
            "maximum_match_candidates",
            "duplicate_window_hours",
            "maximum_age_years",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise MissingPersonValidationError(
                    f"{name} must be a positive integer"
                )

        require_probability(
            self.minimum_sighting_reliability,
            "minimum_sighting_reliability",
        )

        if self.allow_public_location_disclosure:
            raise MissingPersonValidationError(
                "Automatic public location disclosure is prohibited"
            )


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
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise MissingPersonValidationError(
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
        raise MissingPersonValidationError(
            f"{name} must be in [0,1]"
        )


def normalize_text(value: str) -> str:
    return " ".join(
        value.casefold().split()
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

    return value


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class MissingPersonProfile:
    person_id: str

    full_name: str

    age_years: int | None

    district: str

    last_known_location: str

    last_seen_at: datetime

    is_child: bool = False

    requires_medical_support: bool = False

    mobility_support_required: bool = False

    description: str = ""

    def __post_init__(self) -> None:
        if not self.person_id.strip():
            raise MissingPersonValidationError(
                "person_id is required"
            )

        if not self.full_name.strip():
            raise MissingPersonValidationError(
                "full_name is required"
            )

        if not self.district.strip():
            raise MissingPersonValidationError(
                "district is required"
            )

        if not self.last_known_location.strip():
            raise MissingPersonValidationError(
                "last_known_location is required"
            )

        require_aware_datetime(
            self.last_seen_at
        )

        if self.age_years is not None:
            if (
                isinstance(self.age_years, bool)
                or not isinstance(self.age_years, int)
                or not 0 <= self.age_years <= 125
            ):
                raise MissingPersonValidationError(
                    "Invalid age"
                )


@dataclass(frozen=True)
class MissingPersonReport:
    report_id: str

    person: MissingPersonProfile

    reporter_reference: str

    relationship_to_person: str

    incident_description: str

    reported_at: datetime

    def __post_init__(self) -> None:
        if not self.report_id.strip():
            raise MissingPersonValidationError(
                "report_id is required"
            )

        if not self.reporter_reference.strip():
            raise MissingPersonValidationError(
                "reporter_reference is required"
            )

        require_aware_datetime(
            self.reported_at
        )


@dataclass(frozen=True)
class MissingPersonSighting:
    sighting_id: str
    case_id: str

    source_id: str
    source_type: SightingSourceType

    observed_at: datetime

    location_description: str

    verification_status: EvidenceVerificationStatus

    reliability: float

    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "sighting_id",
            "case_id",
            "source_id",
            "location_description",
        ):
            if not getattr(self, name).strip():
                raise MissingPersonValidationError(
                    f"{name} is required"
                )

        require_aware_datetime(
            self.observed_at
        )

        require_probability(
            self.reliability,
            "reliability",
        )


@dataclass(frozen=True)
class MissingPersonMatchCandidate:
    candidate_id: str

    source_reference: str

    reported_name: str

    reported_age_years: int | None

    district: str

    location_reference: str

    verified_source: bool

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "source_reference",
            "reported_name",
            "district",
            "location_reference",
        ):
            if not getattr(self, name).strip():
                raise MissingPersonValidationError(
                    f"{name} is required"
                )


@dataclass(frozen=True)
class MissingPersonMatchResult:
    match_id: str
    case_id: str
    candidate_id: str

    name_consistent: bool
    age_consistent: bool
    district_consistent: bool

    screening_score: float

    status: MatchReviewStatus

    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        require_probability(
            self.screening_score,
            "screening_score",
        )


@dataclass(frozen=True)
class MissingPersonCase:
    case_id: str

    report: MissingPersonReport

    status: MissingPersonStatus

    priority: MissingPersonPriority

    sightings: tuple[MissingPersonSighting, ...]

    match_results: tuple[MissingPersonMatchResult, ...]

    confirmed_candidate_id: str | None

    reunification_status: ReunificationStatus

    recommendations: tuple[str, ...]

    warnings: tuple[str, ...]

    duplicate_detected: bool

    created_at: datetime
    updated_at: datetime

    def to_dict(
        self,
        *,
        include_sensitive: bool = False,
    ) -> dict[str, Any]:
        if include_sensitive:
            return serialize(self)

        # Default output excludes name, precise
        # location, reporter information and
        # individual sighting details.
        return {
            "case_id": self.case_id,
            "district": self.report.person.district,
            "status": self.status.value,
            "priority": self.priority.value,
            "sighting_count": len(self.sightings),
            "possible_match_count": len(
                self.match_results
            ),
            "reunification_status": (
                self.reunification_status.value
            ),
            "recommendations": list(
                self.recommendations
            ),
            "warnings": list(
                self.warnings
            ),
            "duplicate_detected": (
                self.duplicate_detected
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class MissingPersonAuditRecord:
    audit_id: str
    case_id: str

    actor_id: str
    action: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# SEARCH PRIORITY ENGINE
# ============================================================

class MissingPersonPriorityEngine:
    @staticmethod
    def assess(
        person: MissingPersonProfile,
        *,
        now: datetime,
    ) -> MissingPersonPriority:
        require_aware_datetime(now)

        if person.last_seen_at > now:
            raise MissingPersonValidationError(
                "Last-seen timestamp cannot be in the future"
            )

        if (
            person.is_child
            or (
                person.age_years is not None
                and person.age_years < 18
            )
            or person.requires_medical_support
        ):
            return (
                MissingPersonPriority.URGENT_HUMAN_ESCALATION
            )

        elapsed_hours = (
            now - person.last_seen_at
        ).total_seconds() / 3600

        if (
            person.mobility_support_required
            or elapsed_hours >= 24
        ):
            return (
                MissingPersonPriority.HIGH_PRIORITY_REVIEW
            )

        if elapsed_hours >= 6:
            return (
                MissingPersonPriority.ELEVATED_REVIEW
            )

        return (
            MissingPersonPriority.ROUTINE_REVIEW
        )


# ============================================================
# SIGHTING VERIFICATION ENGINE
# ============================================================

class MissingPersonSightingValidator:
    def __init__(
        self,
        config: MissingPersonAgentConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        sighting: MissingPersonSighting,
        *,
        case_id: str,
        now: datetime,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        if sighting.case_id != case_id:
            reasons.append(
                "Case ID mismatch"
            )

        if sighting.verification_status != (
            EvidenceVerificationStatus.VERIFIED
        ):
            reasons.append(
                "Sighting is not verified"
            )

        if sighting.reliability < (
            self.config.minimum_sighting_reliability
        ):
            reasons.append(
                "Sighting reliability below threshold"
            )

        if sighting.observed_at > (
            now + timedelta(minutes=5)
        ):
            reasons.append(
                "Sighting timestamp is in the future"
            )

        if (
            now - sighting.observed_at
        ) > timedelta(
            days=self.config.maximum_sighting_age_days
        ):
            reasons.append(
                "Sighting is stale"
            )

        return (
            len(reasons) == 0,
            tuple(reasons),
        )


# ============================================================
# POSSIBLE MATCH SCREENING
# ============================================================

class MissingPersonMatchEngine:
    """
    Conservative demographic consistency screening.

    Does not perform:
        - Face recognition
        - Biometric matching
        - Identity confirmation

    Scores are heuristic similarity indicators,
    not probabilities that two records refer
    to the same person.
    """

    @staticmethod
    def screen(
        case: MissingPersonCase,
        candidate: MissingPersonMatchCandidate,
    ) -> MissingPersonMatchResult:
        person = case.report.person

        name_consistent = (
            normalize_text(person.full_name)
            == normalize_text(candidate.reported_name)
        )

        age_consistent = (
            person.age_years is not None
            and candidate.reported_age_years is not None
            and abs(
                person.age_years
                - candidate.reported_age_years
            ) <= 2
        )

        district_consistent = (
            normalize_text(person.district)
            == normalize_text(candidate.district)
        )

        score = (
            0.55 * float(name_consistent)
            + 0.25 * float(age_consistent)
            + 0.20 * float(district_consistent)
        )

        warnings = [
            "Candidate match is unconfirmed",
            "Demographic similarity is not identity proof",
            "Do not disclose candidate location",
            "Authorized identity and safeguarding "
            "verification is mandatory",
        ]

        if not candidate.verified_source:
            warnings.append(
                "Candidate source is unverified"
            )

        return MissingPersonMatchResult(
            match_id=generate_id(),
            case_id=case.case_id,
            candidate_id=candidate.candidate_id,
            name_consistent=name_consistent,
            age_consistent=age_consistent,
            district_consistent=district_consistent,
            screening_score=round(score, 4),
            status=MatchReviewStatus.PENDING,
            warnings=tuple(warnings),
        )


# ============================================================
# DUPLICATE CASE DETECTION
# ============================================================

class MissingPersonDeduplicationEngine:
    def __init__(
        self,
        config: MissingPersonAgentConfig,
    ) -> None:
        self.config = config

        self._recent: dict[
            str,
            datetime,
        ] = {}

        self._lock = threading.Lock()

    def check(
        self,
        report: MissingPersonReport,
        *,
        now: datetime,
    ) -> bool:
        # Exact person-reference deduplication only.
        # Similar names must not cause distinct
        # people to be merged.
        content = (
            report.person.person_id
            + "|"
            + report.report_id
        )

        key = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        cutoff = now - timedelta(
            hours=self.config.duplicate_window_hours
        )

        with self._lock:
            self._recent = {
                item_key: timestamp
                for item_key, timestamp in self._recent.items()
                if timestamp >= cutoff
            }

            previous = self._recent.get(key)

            duplicate = (
                previous is not None
                and previous >= cutoff
            )

            if not duplicate:
                self._recent[key] = now

        return duplicate


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class MissingPersonRecommendationEngine:
    @staticmethod
    def generate(
        *,
        priority: MissingPersonPriority,
        person: MissingPersonProfile,
    ) -> tuple[str, ...]:
        recommendations = [
            "Notify the authorized missing-person "
            "coordination team through established channels",
            "Verify last-known information with the reporter",
            "Coordinate authorized shelter and "
            "healthcare facility inquiries",
            "Protect the missing person's identity "
            "and location information",
        ]

        if priority == (
            MissingPersonPriority.URGENT_HUMAN_ESCALATION
        ):
            recommendations.append(
                "Escalate immediately to authorized "
                "emergency and safeguarding personnel"
            )

        if (
            person.is_child
            or (
                person.age_years is not None
                and person.age_years < 18
            )
        ):
            recommendations.append(
                "Apply child-protection procedures "
                "and involve an authorized child-protection officer"
            )

        if person.requires_medical_support:
            recommendations.append(
                "Coordinate with authorized medical personnel"
            )

        if person.mobility_support_required:
            recommendations.append(
                "Review accessible search and transport needs"
            )

        recommendations.extend([
            "Verify any possible identity match manually",
            "Do not publish sightings or locations automatically",
            "Do not dispatch search teams automatically",
            "Reunification requires consent and safeguarding review",
        ])

        return tuple(
            dict.fromkeys(recommendations)
        )


# ============================================================
# AUDIT LOGGER
# ============================================================

class MissingPersonAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            MissingPersonAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        case_id: str,
        actor_id: str,
        action: str,
        details: Mapping[str, str] | None = None,
    ) -> MissingPersonAuditRecord:
        record = MissingPersonAuditRecord(
            audit_id=generate_id(),
            case_id=case_id,
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
        case_id: str | None = None,
    ) -> tuple[MissingPersonAuditRecord, ...]:
        with self._lock:
            records = tuple(self._records)

        if case_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.case_id == case_id
        )


# ============================================================
# HUMAN REVIEW MANAGER
# ============================================================

class MissingPersonHumanReviewManager:
    MATCH_REVIEW_ROLES = {
        AuthorizedRole.MISSING_PERSON_COORDINATOR,
        AuthorizedRole.DISTRICT_OFFICIAL,
        AuthorizedRole.SAFEGUARDING_OFFICER,
    }

    CHILD_REVIEW_ROLES = {
        AuthorizedRole.CHILD_PROTECTION_OFFICER,
        AuthorizedRole.SAFEGUARDING_OFFICER,
    }

    @classmethod
    def authorize_match_review(
        cls,
        *,
        role: AuthorizedRole,
        person: MissingPersonProfile,
    ) -> None:
        if role not in cls.MATCH_REVIEW_ROLES:
            raise MissingPersonPermissionError(
                "Reviewer lacks match-verification permission"
            )

        is_child = (
            person.is_child
            or (
                person.age_years is not None
                and person.age_years < 18
            )
        )

        if is_child and role not in (
            cls.CHILD_REVIEW_ROLES
        ):
            raise MissingPersonPermissionError(
                "Child-related identity review requires "
                "a safeguarding or child-protection officer"
            )


# ============================================================
# MAIN MISSING PERSON AGENT
# ============================================================

class MissingPersonAgent:
    def __init__(
        self,
        config: MissingPersonAgentConfig | None = None,
    ) -> None:
        self.config = (
            config or MissingPersonAgentConfig()
        )

        self.priority_engine = (
            MissingPersonPriorityEngine()
        )

        self.sighting_validator = (
            MissingPersonSightingValidator(
                self.config
            )
        )

        self.match_engine = (
            MissingPersonMatchEngine()
        )

        self.deduplication_engine = (
            MissingPersonDeduplicationEngine(
                self.config
            )
        )

        self.recommendation_engine = (
            MissingPersonRecommendationEngine()
        )

        self.audit_logger = (
            MissingPersonAuditLogger()
        )

        self.review_manager = (
            MissingPersonHumanReviewManager()
        )

        self._cases: dict[
            str,
            MissingPersonCase,
        ] = {}

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # REGISTER CASE
    # --------------------------------------------------------

    def register_case(
        self,
        report: MissingPersonReport,
        *,
        now: datetime | None = None,
    ) -> MissingPersonCase:
        now = now or utc_now()
        require_aware_datetime(now)

        if report.reported_at > (
            now + timedelta(minutes=5)
        ):
            raise MissingPersonValidationError(
                "Report timestamp is in the future"
            )

        priority = self.priority_engine.assess(
            report.person,
            now=now,
        )

        duplicate = (
            self.deduplication_engine.check(
                report,
                now=now,
            )
        )

        recommendations = (
            self.recommendation_engine.generate(
                priority=priority,
                person=report.person,
            )
        )

        warnings = [
            "Missing-person information is sensitive",
            "No search team has been dispatched",
            "No location has been disclosed",
            "No identity match has been confirmed",
            "All operational actions require human authorization",
        ]

        if duplicate:
            warnings.append(
                "Repeated case submission detected"
            )

        case = MissingPersonCase(
            case_id=generate_id(),
            report=report,
            status=MissingPersonStatus.REPORTED,
            priority=priority,
            sightings=(),
            match_results=(),
            confirmed_candidate_id=None,
            reunification_status=(
                ReunificationStatus.NOT_STARTED
            ),
            recommendations=recommendations,
            warnings=tuple(warnings),
            duplicate_detected=duplicate,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._cases[
                case.case_id
            ] = case

        self.audit_logger.record(
            case_id=case.case_id,
            actor_id="missing_person_agent",
            action="case_registered",
            details={
                "priority": priority.value,
                "district": report.person.district,
            },
        )

        return case

    # --------------------------------------------------------
    # GET CASE
    # --------------------------------------------------------

    def get_case(
        self,
        case_id: str,
    ) -> MissingPersonCase:
        with self._lock:
            case = self._cases.get(
                case_id
            )

        if case is None:
            raise MissingPersonNotFoundError(
                "Case not found"
            )

        return case

    # --------------------------------------------------------
    # ADD SIGHTING
    # --------------------------------------------------------

    def add_sighting(
        self,
        case_id: str,
        sighting: MissingPersonSighting,
        *,
        actor_id: str,
        now: datetime | None = None,
    ) -> MissingPersonCase:
        now = now or utc_now()

        if not actor_id.strip():
            raise MissingPersonValidationError(
                "Actor ID required"
            )

        with self._lock:
            case = self.get_case(
                case_id
            )

            if case.status in (
                MissingPersonStatus.CLOSED,
                MissingPersonStatus.CANCELLED,
            ):
                raise MissingPersonStateError(
                    "Cannot add sightings to a closed case"
                )

            if len(case.sightings) >= (
                self.config.maximum_sightings_per_case
            ):
                raise MissingPersonValidationError(
                    "Sighting limit exceeded"
                )

            if any(
                item.sighting_id == sighting.sighting_id
                for item in case.sightings
            ):
                raise MissingPersonValidationError(
                    "Duplicate sighting ID"
                )

            valid, reasons = (
                self.sighting_validator.validate(
                    sighting,
                    case_id=case_id,
                    now=now,
                )
            )

            warnings = list(
                case.warnings
            )

            if not valid:
                warnings.extend(
                    reasons
                )

            updated = replace(
                case,
                sightings=(
                    case.sightings + (sighting,)
                ),
                status=(
                    MissingPersonStatus.SEARCH_COORDINATION
                    if valid
                    else MissingPersonStatus.UNDER_REVIEW
                ),
                warnings=tuple(warnings),
                updated_at=now,
            )

            self._cases[
                case_id
            ] = updated

            # Sighting location is intentionally
            # excluded from audit metadata.
            self.audit_logger.record(
                case_id=case_id,
                actor_id=actor_id,
                action="sighting_recorded",
                details={
                    "sighting_id": sighting.sighting_id,
                    "verified": str(valid),
                },
            )

            return updated

    # --------------------------------------------------------
    # SCREEN POSSIBLE MATCH
    # --------------------------------------------------------

    def screen_candidate(
        self,
        case_id: str,
        candidate: MissingPersonMatchCandidate,
        *,
        actor_id: str,
        now: datetime | None = None,
    ) -> MissingPersonMatchResult:
        now = now or utc_now()

        if not actor_id.strip():
            raise MissingPersonValidationError(
                "Actor ID required"
            )

        with self._lock:
            case = self.get_case(
                case_id
            )

            if case.status in (
                MissingPersonStatus.CLOSED,
                MissingPersonStatus.CANCELLED,
            ):
                raise MissingPersonStateError(
                    "Case is closed"
                )

            if len(case.match_results) >= (
                self.config.maximum_match_candidates
            ):
                raise MissingPersonValidationError(
                    "Candidate limit exceeded"
                )

            if any(
                item.candidate_id == candidate.candidate_id
                for item in case.match_results
            ):
                raise MissingPersonValidationError(
                    "Candidate already screened"
                )

            result = self.match_engine.screen(
                case,
                candidate,
            )

            updated = replace(
                case,
                match_results=(
                    case.match_results + (result,)
                ),
                status=MissingPersonStatus.POSSIBLE_MATCH,
                updated_at=now,
            )

            self._cases[
                case_id
            ] = updated

            self.audit_logger.record(
                case_id=case_id,
                actor_id=actor_id,
                action="candidate_screened",
                details={
                    "candidate_id": candidate.candidate_id,
                    "match_id": result.match_id,
                },
            )

            return result

    # --------------------------------------------------------
    # REVIEW MATCH
    # --------------------------------------------------------

    def review_match(
        self,
        case_id: str,
        match_id: str,
        *,
        reviewer_id: str,
        reviewer_role: AuthorizedRole,
        confirmed: bool,
        reason: str,
        now: datetime | None = None,
    ) -> MissingPersonCase:
        now = now or utc_now()

        if not reviewer_id.strip() or not reason.strip():
            raise MissingPersonValidationError(
                "Reviewer and reason required"
            )

        with self._lock:
            case = self.get_case(
                case_id
            )

            if case.status in (
                MissingPersonStatus.CLOSED,
                MissingPersonStatus.CANCELLED,
            ):
                raise MissingPersonStateError(
                    "Case is closed"
                )

            self.review_manager.authorize_match_review(
                role=reviewer_role,
                person=case.report.person,
            )

            target = next(
                (
                    item
                    for item in case.match_results
                    if item.match_id == match_id
                ),
                None,
            )

            if target is None:
                raise MissingPersonNotFoundError(
                    "Match not found"
                )

            if target.status != MatchReviewStatus.PENDING:
                raise MissingPersonStateError(
                    "Match already reviewed"
                )

            new_results = tuple(
                replace(
                    item,
                    status=(
                        MatchReviewStatus.CONFIRMED
                        if confirmed
                        else MatchReviewStatus.REJECTED
                    ),
                )
                if item.match_id == match_id
                else item
                for item in case.match_results
            )

            updated = replace(
                case,
                match_results=new_results,
                confirmed_candidate_id=(
                    target.candidate_id
                    if confirmed
                    else case.confirmed_candidate_id
                ),
                status=(
                    MissingPersonStatus.LOCATED_PENDING_VERIFICATION
                    if confirmed
                    else MissingPersonStatus.SEARCH_COORDINATION
                ),
                updated_at=now,
            )

            self._cases[
                case_id
            ] = updated

            self.audit_logger.record(
                case_id=case_id,
                actor_id=reviewer_id,
                action=(
                    "candidate_review_confirmed"
                    if confirmed
                    else "candidate_review_rejected"
                ),
                details={
                    "match_id": match_id,
                    "reviewer_role": reviewer_role.value,
                    "reason": reason,
                },
            )

            return updated

    # --------------------------------------------------------
    # REQUEST REUNIFICATION REVIEW
    # --------------------------------------------------------

    def request_reunification_review(
        self,
        case_id: str,
        *,
        actor_id: str,
        now: datetime | None = None,
    ) -> MissingPersonCase:
        now = now or utc_now()

        if not actor_id.strip():
            raise MissingPersonValidationError(
                "Actor ID required"
            )

        with self._lock:
            case = self.get_case(
                case_id
            )

            if case.status != (
                MissingPersonStatus.LOCATED_CONFIRMED
            ):
                raise MissingPersonStateError(
                    "Independent location and identity "
                    "verification is required first"
                )

            updated = replace(
                case,
                status=MissingPersonStatus.REUNIFICATION_PENDING,
                reunification_status=(
                    ReunificationStatus.PENDING_SAFEGUARDING_REVIEW
                ),
                updated_at=now,
            )

            self._cases[
                case_id
            ] = updated

            self.audit_logger.record(
                case_id=case_id,
                actor_id=actor_id,
                action="reunification_review_requested",
            )

            return updated

    # --------------------------------------------------------
    # SAFEGUARDING REVIEW
    # --------------------------------------------------------

    def review_reunification(
        self,
        case_id: str,
        *,
        reviewer_id: str,
        reviewer_role: AuthorizedRole,
        approved: bool,
        consent_verified: bool,
        identity_verified: bool,
        reason: str,
        now: datetime | None = None,
    ) -> MissingPersonCase:
        now = now or utc_now()

        if reviewer_role not in {
            AuthorizedRole.SAFEGUARDING_OFFICER,
            AuthorizedRole.CHILD_PROTECTION_OFFICER,
        }:
            raise MissingPersonPermissionError(
                "Safeguarding reviewer required"
            )

        if not reviewer_id.strip() or not reason.strip():
            raise MissingPersonValidationError(
                "Reviewer and reason required"
            )

        with self._lock:
            case = self.get_case(
                case_id
            )

            if case.reunification_status != (
                ReunificationStatus.PENDING_SAFEGUARDING_REVIEW
            ):
                raise MissingPersonStateError(
                    "Reunification review not pending"
                )

            if (
                approved
                and not (
                    consent_verified
                    and identity_verified
                )
            ):
                raise MissingPersonPermissionError(
                    "Identity and applicable consent "
                    "must be verified before approval"
                )

            updated = replace(
                case,
                reunification_status=(
                    ReunificationStatus.APPROVED_FOR_COORDINATION
                    if approved
                    else ReunificationStatus.BLOCKED
                ),
                updated_at=now,
            )

            self._cases[
                case_id
            ] = updated

            self.audit_logger.record(
                case_id=case_id,
                actor_id=reviewer_id,
                action=(
                    "reunification_review_approved"
                    if approved
                    else "reunification_review_blocked"
                ),
                details={
                    "reviewer_role": reviewer_role.value,
                    "reason": reason,
                },
            )

            return updated

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def get_audit_history(
        self,
        case_id: str,
    ) -> tuple[MissingPersonAuditRecord, ...]:
        return self.audit_logger.list_records(
            case_id
        )


# ============================================================
# PRIVACY-AWARE JSON EXPORT
# ============================================================

class MissingPersonReportExporter:
    @staticmethod
    def export_json(
        case: MissingPersonCase,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise MissingPersonExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                case.to_dict(
                    include_sensitive=False
                ),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_missing_person_agent(
    config: MissingPersonAgentConfig | None = None,
) -> MissingPersonAgent:
    return MissingPersonAgent(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    now = utc_now()

    # All data below is fictional.
    person = MissingPersonProfile(
        person_id="DEMO-PERSON-001",
        full_name="Example Person",
        age_years=16,
        district="Thrissur",
        last_known_location="Illustrative flood zone",
        last_seen_at=now - timedelta(hours=3),
        is_child=True,
    )

    report = MissingPersonReport(
        report_id="DEMO-REPORT-001",
        person=person,
        reporter_reference="DEMO-REPORTER-001",
        relationship_to_person="guardian",
        incident_description=(
            "Fictional missing-person report "
            "for system demonstration."
        ),
        reported_at=now,
    )

    agent = create_missing_person_agent()

    case = agent.register_case(
        report,
        now=now,
    )

    print(
        json.dumps(
            case.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    sighting = MissingPersonSighting(
        sighting_id="DEMO-SIGHTING-001",
        case_id=case.case_id,
        source_id="DEMO-SHELTER-001",
        source_type=SightingSourceType.SHELTER,
        observed_at=now - timedelta(minutes=20),
        location_description="Fictional shelter",
        verification_status=(
            EvidenceVerificationStatus.UNVERIFIED
        ),
        reliability=0.60,
        notes="Unconfirmed illustrative sighting",
    )

    case = agent.add_sighting(
        case.case_id,
        sighting,
        actor_id="demo-coordinator",
        now=now,
    )

    candidate = MissingPersonMatchCandidate(
        candidate_id="DEMO-CANDIDATE-001",
        source_reference="DEMO-SHELTER-001",
        reported_name="Example Person",
        reported_age_years=16,
        district="Thrissur",
        location_reference="PRIVATE-DEMO-LOCATION",
        verified_source=False,
    )

    match = agent.screen_candidate(
        case.case_id,
        candidate,
        actor_id="demo-coordinator",
        now=now,
    )

    print(
        "Possible match screening score:",
        match.screening_score,
    )

    print(
        "Match status:",
        match.status.value,
    )

    MissingPersonReportExporter.export_json(
        case,
        "outputs/agents/missing_person_report.json",
    )

    print(
        "No identity was confirmed, "
        "no location was disclosed, "
        "and no rescue action was executed."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "MissingPersonAgentError",
    "MissingPersonValidationError",
    "MissingPersonPermissionError",
    "MissingPersonStateError",
    "MissingPersonNotFoundError",
    "MissingPersonExportError",
    "MissingPersonStatus",
    "MissingPersonPriority",
    "SightingSourceType",
    "EvidenceVerificationStatus",
    "MatchReviewStatus",
    "ReunificationStatus",
    "AuthorizedRole",
    "MissingPersonAgentConfig",
    "MissingPersonProfile",
    "MissingPersonReport",
    "MissingPersonSighting",
    "MissingPersonMatchCandidate",
    "MissingPersonMatchResult",
    "MissingPersonCase",
    "MissingPersonAuditRecord",
    "MissingPersonPriorityEngine",
    "MissingPersonSightingValidator",
    "MissingPersonMatchEngine",
    "MissingPersonDeduplicationEngine",
    "MissingPersonRecommendationEngine",
    "MissingPersonAuditLogger",
    "MissingPersonHumanReviewManager",
    "MissingPersonAgent",
    "MissingPersonReportExporter",
    "create_missing_person_agent",
]
