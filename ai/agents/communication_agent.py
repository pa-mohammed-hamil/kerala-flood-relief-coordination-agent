"""Placeholder module: ai/agents/communication_agent.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
File: backend/ai/agents/communication_agent.py

Multilingual, evidence-grounded emergency communication
drafting and human-review agent.

Features:
    - English and Malayalam message drafting
    - Flood situation updates
    - Shelter information
    - Rescue-request acknowledgments
    - Volunteer coordination
    - Relief distribution updates
    - Healthcare coordination drafts
    - SMS, push, email and dashboard formats
    - Verified-source references
    - Template-based message generation
    - Sensitive-information protection
    - Message deduplication
    - Human approval workflow
    - Audit records
    - JSON export

Security:
    - No autonomous message delivery
    - No automatic public warnings
    - No automatic rescue dispatch
    - No fabricated emergency contact details
    - No unverified evacuation instructions
    - No exposure of private medical information
    - Approval is not equivalent to publication

Dependencies:
    Python 3.11+

This module is deliberately transport-independent.
Authorized communication integrations must be
implemented separately with access control,
consent checks and delivery auditing.
"""

from __future__ import annotations

import hashlib
import json
import re
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

class CommunicationAgentError(Exception):
    pass


class CommunicationValidationError(
    CommunicationAgentError
):
    pass


class CommunicationEvidenceError(
    CommunicationAgentError
):
    pass


class CommunicationApprovalError(
    CommunicationAgentError
):
    pass


class CommunicationExportError(
    CommunicationAgentError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class CommunicationLanguage(str, Enum):
    ENGLISH = "en"
    MALAYALAM = "ml"


class CommunicationChannel(str, Enum):
    SMS = "sms"
    PUSH = "push"
    EMAIL = "email"
    DASHBOARD = "dashboard"


class CommunicationAudience(str, Enum):
    PUBLIC = "public"
    RESCUE_TEAM = "rescue_team"
    VOLUNTEER = "volunteer"
    SHELTER_STAFF = "shelter_staff"
    HEALTHCARE_STAFF = "healthcare_staff"
    INTERNAL_OFFICIAL = "internal_official"


class CommunicationType(str, Enum):
    FLOOD_UPDATE = "flood_update"
    SHELTER_UPDATE = "shelter_update"
    RESCUE_ACKNOWLEDGMENT = "rescue_acknowledgment"
    VOLUNTEER_COORDINATION = "volunteer_coordination"
    RELIEF_DISTRIBUTION = "relief_distribution"
    HEALTHCARE_COORDINATION = "healthcare_coordination"
    GENERAL_INFORMATION = "general_information"


class CommunicationPriority(str, Enum):
    ROUTINE = "routine"
    IMPORTANT = "important"
    URGENT = "urgent"


class CommunicationStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class EvidenceVerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CommunicationAgentConfig:
    max_message_characters: int = 4000
    sms_character_limit: int = 320

    maximum_evidence_age_minutes: int = 180

    deduplication_window_minutes: int = 60

    maximum_evidence_items: int = 20

    require_verified_evidence: bool = True
    require_human_approval: bool = True

    allow_unverified_general_information: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_message_characters",
            "sms_character_limit",
            "maximum_evidence_age_minutes",
            "deduplication_window_minutes",
            "maximum_evidence_items",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CommunicationValidationError(
                    f"{name} must be positive"
                )

        if (
            self.sms_character_limit
            > self.max_message_characters
        ):
            raise CommunicationValidationError(
                "SMS limit exceeds maximum message length"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class CommunicationEvidence:
    evidence_id: str
    source_id: str
    source_name: str

    observed_at: datetime

    verification_status: EvidenceVerificationStatus

    facts: Mapping[str, str]

    reference_url: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "source_id",
            "source_name",
        ):
            if not getattr(self, name).strip():
                raise CommunicationValidationError(
                    f"{name} is required"
                )

        if (
            self.observed_at.tzinfo is None
            or self.observed_at.utcoffset() is None
        ):
            raise CommunicationValidationError(
                "Evidence timestamp must be timezone-aware"
            )

        if not isinstance(self.facts, Mapping):
            raise CommunicationValidationError(
                "facts must be a mapping"
            )

        for key, value in self.facts.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
                or not key.strip()
            ):
                raise CommunicationValidationError(
                    "Evidence facts must be string pairs"
                )


@dataclass(frozen=True)
class CommunicationRequest:
    request_id: str

    communication_type: CommunicationType
    audience: CommunicationAudience
    channel: CommunicationChannel
    language: CommunicationLanguage

    district: str

    priority: CommunicationPriority

    evidence: tuple[CommunicationEvidence, ...]

    context: Mapping[str, str]

    created_at: datetime


@dataclass(frozen=True)
class CommunicationDraft:
    draft_id: str
    request_id: str

    communication_type: CommunicationType
    audience: CommunicationAudience
    channel: CommunicationChannel
    language: CommunicationLanguage

    district: str
    priority: CommunicationPriority

    subject: str
    body: str

    evidence_ids: tuple[str, ...]

    status: CommunicationStatus

    requires_human_approval: bool

    duplicate_detected: bool

    warnings: tuple[str, ...]

    created_at: datetime

    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return serialize_value(self)


@dataclass(frozen=True)
class CommunicationReview:
    review_id: str
    draft_id: str

    reviewer_id: str
    reviewer_role: str

    approved: bool
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class CommunicationAuditRecord:
    audit_id: str
    draft_id: str

    action: str
    actor_id: str

    timestamp: datetime

    details: Mapping[str, str]


@dataclass(frozen=True)
class CommunicationAgentOutput:
    run_id: str
    draft: CommunicationDraft

    accepted_evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]

    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize_value(self)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize_value(
            asdict(value)
        )

    if isinstance(value, Mapping):
        return {
            str(key): serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            serialize_value(item)
            for item in value
        ]

    return value


def normalize_whitespace(
    value: str,
) -> str:
    return " ".join(
        value.split()
    )


def ensure_aware_datetime(
    value: datetime,
) -> None:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise CommunicationValidationError(
            "Datetime must be timezone-aware"
        )


# ============================================================
# REQUEST VALIDATOR
# ============================================================

class CommunicationRequestValidator:
    @staticmethod
    def validate(
        request: CommunicationRequest,
    ) -> None:
        if not request.request_id.strip():
            raise CommunicationValidationError(
                "request_id is required"
            )

        if not request.district.strip():
            raise CommunicationValidationError(
                "district is required"
            )

        ensure_aware_datetime(
            request.created_at
        )

        if not isinstance(
            request.context,
            Mapping,
        ):
            raise CommunicationValidationError(
                "context must be a mapping"
            )

        for key, value in request.context.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
            ):
                raise CommunicationValidationError(
                    "Context must contain string pairs"
                )


# ============================================================
# EVIDENCE VALIDATOR
# ============================================================

class CommunicationEvidenceValidator:
    def __init__(
        self,
        config: CommunicationAgentConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        evidence: CommunicationEvidence,
        *,
        now: datetime,
    ) -> tuple[bool, str | None]:
        ensure_aware_datetime(now)

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
            self.config.maximum_evidence_age_minutes
        ):
            return (
                False,
                "Evidence is stale",
            )

        if (
            self.config.require_verified_evidence
            and evidence.verification_status
            != EvidenceVerificationStatus.VERIFIED
        ):
            return (
                False,
                "Evidence is unverified",
            )

        if not evidence.facts:
            return (
                False,
                "Evidence has no facts",
            )

        return True, None

    def filter_evidence(
        self,
        evidence: Sequence[CommunicationEvidence],
        *,
        now: datetime,
    ) -> tuple[
        tuple[CommunicationEvidence, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        if len(evidence) > (
            self.config.maximum_evidence_items
        ):
            raise CommunicationEvidenceError(
                "Too many evidence items"
            )

        accepted = []
        rejected = []
        warnings = []

        seen_ids = set()

        for item in evidence:
            if item.evidence_id in seen_ids:
                rejected.append(
                    item.evidence_id
                )

                warnings.append(
                    "Duplicate evidence ID rejected"
                )

                continue

            seen_ids.add(
                item.evidence_id
            )

            valid, reason = self.validate(
                item,
                now=now,
            )

            if valid:
                accepted.append(item)

            else:
                rejected.append(
                    item.evidence_id
                )

                warnings.append(
                    f"Evidence {item.evidence_id}: {reason}"
                )

        return (
            tuple(accepted),
            tuple(rejected),
            tuple(warnings),
        )


# ============================================================
# PRIVACY PROTECTION
# ============================================================

class CommunicationPrivacyGuard:
    """
    Conservative redaction of common identifiers.

    Regex-based redaction is not a complete
    PII detection solution. Public-facing
    drafts require privacy review.
    """

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    PHONE_PATTERN = re.compile(
        r"(?<!\d)(?:\+91[\s-]?)?"
        r"[6-9]\d{9}(?!\d)"
    )

    @classmethod
    def redact(
        cls,
        text: str,
    ) -> str:
        text = cls.EMAIL_PATTERN.sub(
            "[REDACTED_EMAIL]",
            text,
        )

        text = cls.PHONE_PATTERN.sub(
            "[REDACTED_PHONE]",
            text,
        )

        return text

    @classmethod
    def sanitize_context(
        cls,
        context: Mapping[str, str],
    ) -> dict[str, str]:
        return {
            key: cls.redact(
                normalize_whitespace(value)
            )
            for key, value in context.items()
        }


# ============================================================
# MULTILINGUAL TEMPLATE ENGINE
# ============================================================

class CommunicationTemplateEngine:
    """
    Deterministic templates.

    No LLM-generated emergency instructions
    are introduced by this component.
    """

    TEMPLATES = {
        CommunicationLanguage.ENGLISH: {
            CommunicationType.FLOOD_UPDATE: (
                "Flood Situation Update",
                (
                    "Flood-related conditions in {district} "
                    "are under review. {details} "
                    "Please refer to verified updates "
                    "from authorized authorities."
                ),
            ),

            CommunicationType.SHELTER_UPDATE: (
                "Shelter Information",
                (
                    "Shelter information for {district}: "
                    "{details} "
                    "Availability and access must be "
                    "confirmed by authorized shelter staff."
                ),
            ),

            CommunicationType.RESCUE_ACKNOWLEDGMENT: (
                "Rescue Request Acknowledgment",
                (
                    "Your rescue-related request for "
                    "{district} has been recorded for "
                    "review. {details} "
                    "This message does not confirm "
                    "that a rescue team has been dispatched."
                ),
            ),

            CommunicationType.VOLUNTEER_COORDINATION: (
                "Volunteer Coordination",
                (
                    "Volunteer coordination update for "
                    "{district}: {details} "
                    "Please follow instructions from "
                    "authorized coordinators."
                ),
            ),

            CommunicationType.RELIEF_DISTRIBUTION: (
                "Relief Distribution Information",
                (
                    "Relief distribution information "
                    "for {district}: {details} "
                    "Distribution details must be "
                    "confirmed with authorized personnel."
                ),
            ),

            CommunicationType.HEALTHCARE_COORDINATION: (
                "Healthcare Coordination",
                (
                    "Healthcare coordination update "
                    "for {district}: {details} "
                    "Do not include private patient "
                    "information in public communications."
                ),
            ),

            CommunicationType.GENERAL_INFORMATION: (
                "Information Update",
                (
                    "Information for {district}: "
                    "{details}"
                ),
            ),
        },

        CommunicationLanguage.MALAYALAM: {
            CommunicationType.FLOOD_UPDATE: (
                "വെള്ളപ്പൊക്ക സ്ഥിതിവിവര അറിയിപ്പ്",
                (
                    "{district} ജില്ലയിലെ വെള്ളപ്പൊക്കവുമായി "
                    "ബന്ധപ്പെട്ട സാഹചര്യം പരിശോധിച്ചുവരുന്നു. "
                    "{details} "
                    "അധികൃതരുടെ സ്ഥിരീകരിച്ച അറിയിപ്പുകൾ "
                    "പരിശോധിക്കുക."
                ),
            ),

            CommunicationType.SHELTER_UPDATE: (
                "ദുരിതാശ്വാസ ക്യാമ്പ് വിവരം",
                (
                    "{district} ജില്ലയിലെ ദുരിതാശ്വാസ "
                    "ക്യാമ്പുകളെക്കുറിച്ചുള്ള വിവരം: "
                    "{details} "
                    "ലഭ്യതയും പ്രവേശനവും ബന്ധപ്പെട്ട "
                    "അധികൃതരുമായി സ്ഥിരീകരിക്കുക."
                ),
            ),

            CommunicationType.RESCUE_ACKNOWLEDGMENT: (
                "രക്ഷാപ്രവർത്തന അഭ്യർത്ഥന",
                (
                    "{district} ജില്ലയുമായി ബന്ധപ്പെട്ട "
                    "രക്ഷാപ്രവർത്തന അഭ്യർത്ഥന "
                    "പരിശോധനയ്ക്കായി രേഖപ്പെടുത്തിയിട്ടുണ്ട്. "
                    "{details} "
                    "രക്ഷാസംഘത്തെ അയച്ചതായി ഈ സന്ദേശം "
                    "സ്ഥിരീകരിക്കുന്നില്ല."
                ),
            ),

            CommunicationType.VOLUNTEER_COORDINATION: (
                "സന്നദ്ധപ്രവർത്തക ഏകോപനം",
                (
                    "{district} ജില്ലയിലെ സന്നദ്ധപ്രവർത്തക "
                    "ഏകോപന വിവരം: {details} "
                    "അംഗീകൃത ഏകോപകരുടെ നിർദേശങ്ങൾ "
                    "പാലിക്കുക."
                ),
            ),

            CommunicationType.RELIEF_DISTRIBUTION: (
                "ദുരിതാശ്വാസ വിതരണം",
                (
                    "{district} ജില്ലയിലെ ദുരിതാശ്വാസ "
                    "വിതരണ വിവരം: {details} "
                    "വിതരണ വിവരങ്ങൾ ബന്ധപ്പെട്ട "
                    "അധികൃതരുമായി സ്ഥിരീകരിക്കുക."
                ),
            ),

            CommunicationType.HEALTHCARE_COORDINATION: (
                "ആരോഗ്യസേവന ഏകോപനം",
                (
                    "{district} ജില്ലയിലെ ആരോഗ്യസേവന "
                    "ഏകോപന വിവരം: {details} "
                    "രോഗികളുടെ സ്വകാര്യ വിവരങ്ങൾ "
                    "പൊതു സന്ദേശങ്ങളിൽ ഉൾപ്പെടുത്തരുത്."
                ),
            ),

            CommunicationType.GENERAL_INFORMATION: (
                "വിവര അറിയിപ്പ്",
                (
                    "{district} ജില്ലയ്ക്കുള്ള വിവരം: "
                    "{details}"
                ),
            ),
        },
    }

    @classmethod
    def render(
        cls,
        *,
        language: CommunicationLanguage,
        communication_type: CommunicationType,
        district: str,
        details: str,
    ) -> tuple[str, str]:
        try:
            subject_template, body_template = (
                cls.TEMPLATES[
                    language
                ][communication_type]
            )

        except KeyError as exc:
            raise CommunicationValidationError(
                "Unsupported communication template"
            ) from exc

        values = {
            "district": district,
            "details": details,
        }

        return (
            subject_template.format_map(values),
            body_template.format_map(values),
        )


# ============================================================
# EVIDENCE-GROUNDED CONTENT BUILDER
# ============================================================

class CommunicationContentBuilder:
    """
    Only approved structured fact keys are
    eligible for inclusion in message bodies.

    Source text is data, not instructions.
    """

    ALLOWED_FACT_KEYS = {
        CommunicationType.FLOOD_UPDATE: {
            "rainfall_summary",
            "river_summary",
            "official_status",
            "affected_area_summary",
        },

        CommunicationType.SHELTER_UPDATE: {
            "shelter_name",
            "shelter_location",
            "availability_status",
            "facility_summary",
        },

        CommunicationType.RESCUE_ACKNOWLEDGMENT: {
            "request_reference",
            "request_status",
        },

        CommunicationType.VOLUNTEER_COORDINATION: {
            "coordination_location",
            "volunteer_activity",
            "reporting_status",
        },

        CommunicationType.RELIEF_DISTRIBUTION: {
            "distribution_location",
            "distribution_status",
            "item_summary",
        },

        CommunicationType.HEALTHCARE_COORDINATION: {
            "facility_name",
            "facility_status",
            "service_summary",
        },

        CommunicationType.GENERAL_INFORMATION: {
            "official_status",
            "general_summary",
        },
    }

    @classmethod
    def build_details(
        cls,
        communication_type: CommunicationType,
        evidence: Sequence[CommunicationEvidence],
    ) -> str:
        allowed_keys = cls.ALLOWED_FACT_KEYS[
            communication_type
        ]

        details = []

        for item in evidence:
            for key in sorted(item.facts):
                if key not in allowed_keys:
                    continue

                value = normalize_whitespace(
                    item.facts[key]
                )

                if not value:
                    continue

                value = (
                    CommunicationPrivacyGuard.redact(
                        value
                    )
                )

                details.append(
                    f"{key.replace('_', ' ')}: {value}"
                )

        return "; ".join(
            details
        )


# ============================================================
# MESSAGE VALIDATOR
# ============================================================

class CommunicationMessageValidator:
    def __init__(
        self,
        config: CommunicationAgentConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        *,
        subject: str,
        body: str,
        channel: CommunicationChannel,
    ) -> None:
        if not subject.strip():
            raise CommunicationValidationError(
                "Message subject is empty"
            )

        if not body.strip():
            raise CommunicationValidationError(
                "Message body is empty"
            )

        if len(body) > (
            self.config.max_message_characters
        ):
            raise CommunicationValidationError(
                "Message exceeds maximum length"
            )

        if (
            channel == CommunicationChannel.SMS
            and len(body)
            > self.config.sms_character_limit
        ):
            raise CommunicationValidationError(
                "SMS draft exceeds configured limit"
            )


# ============================================================
# DEDUPLICATION ENGINE
# ============================================================

class CommunicationDeduplicationEngine:
    def __init__(
        self,
        config: CommunicationAgentConfig,
    ) -> None:
        self.config = config

        self._recent: dict[
            str,
            datetime,
        ] = {}

        self._lock = threading.Lock()

    def check_and_record(
        self,
        content_hash: str,
        *,
        now: datetime,
    ) -> bool:
        ensure_aware_datetime(now)

        cutoff = (
            now - timedelta(
                minutes=(
                    self.config.deduplication_window_minutes
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
                content_hash
            )

            duplicate = (
                previous is not None
                and previous >= cutoff
            )

            if not duplicate:
                self._recent[
                    content_hash
                ] = now

            return duplicate


# ============================================================
# AUDIT LOGGER
# ============================================================

class CommunicationAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            CommunicationAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        draft_id: str,
        action: str,
        actor_id: str,
        details: Mapping[str, str] | None = None,
    ) -> CommunicationAuditRecord:
        record = CommunicationAuditRecord(
            audit_id=generate_id(),
            draft_id=draft_id,
            action=action,
            actor_id=actor_id,
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
        draft_id: str | None = None,
    ) -> tuple[
        CommunicationAuditRecord,
        ...
    ]:
        with self._lock:
            records = tuple(
                self._records
            )

        if draft_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.draft_id == draft_id
        )


# ============================================================
# HUMAN REVIEW MANAGER
# ============================================================

class CommunicationReviewManager:
    """
    Stores review decisions.

    In-memory storage is for local development.
    Production systems require durable storage,
    authenticated reviewer identities, RBAC,
    audit retention and concurrency controls.
    """

    ALLOWED_REVIEWER_ROLES = {
        "district_official",
        "emergency_coordinator",
        "communication_supervisor",
    }

    def __init__(
        self,
        audit_logger: CommunicationAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            CommunicationReview,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        draft: CommunicationDraft,
        *,
        reviewer_id: str,
        reviewer_role: str,
        approved: bool,
        reason: str,
    ) -> CommunicationReview:
        if not reviewer_id.strip():
            raise CommunicationApprovalError(
                "Reviewer identity required"
            )

        if reviewer_role not in (
            self.ALLOWED_REVIEWER_ROLES
        ):
            raise CommunicationApprovalError(
                "Reviewer role not permitted"
            )

        if not reason.strip():
            raise CommunicationApprovalError(
                "Review reason required"
            )

        if draft.status != (
            CommunicationStatus.PENDING_REVIEW
        ):
            raise CommunicationApprovalError(
                "Draft is not pending review"
            )

        if draft.duplicate_detected and approved:
            raise CommunicationApprovalError(
                "Duplicate draft requires a new "
                "explicit review workflow"
            )

        with self._lock:
            if draft.draft_id in self._reviews:
                raise CommunicationApprovalError(
                    "Draft already reviewed"
                )

            review = CommunicationReview(
                review_id=generate_id(),
                draft_id=draft.draft_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                approved=approved,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                draft.draft_id
            ] = review

        self.audit_logger.record(
            draft_id=draft.draft_id,
            action=(
                "draft_approved"
                if approved
                else "draft_rejected"
            ),
            actor_id=reviewer_id,
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review

    def get_review(
        self,
        draft_id: str,
    ) -> CommunicationReview | None:
        with self._lock:
            return self._reviews.get(
                draft_id
            )


# ============================================================
# MAIN COMMUNICATION AGENT
# ============================================================

class CommunicationAgent:
    def __init__(
        self,
        config: CommunicationAgentConfig | None = None,
    ) -> None:
        self.config = (
            config or CommunicationAgentConfig()
        )

        self.request_validator = (
            CommunicationRequestValidator()
        )

        self.evidence_validator = (
            CommunicationEvidenceValidator(
                self.config
            )
        )

        self.content_builder = (
            CommunicationContentBuilder()
        )

        self.template_engine = (
            CommunicationTemplateEngine()
        )

        self.message_validator = (
            CommunicationMessageValidator(
                self.config
            )
        )

        self.deduplication_engine = (
            CommunicationDeduplicationEngine(
                self.config
            )
        )

        self.audit_logger = (
            CommunicationAuditLogger()
        )

        self.review_manager = (
            CommunicationReviewManager(
                self.audit_logger
            )
        )

        self._drafts: dict[
            str,
            CommunicationDraft,
        ] = {}

        self._lock = threading.Lock()

    def run(
        self,
        request: CommunicationRequest,
        *,
        now: datetime | None = None,
    ) -> CommunicationAgentOutput:
        now = now or utc_now()

        ensure_aware_datetime(now)

        self.request_validator.validate(
            request
        )

        (
            accepted_evidence,
            rejected_ids,
            evidence_warnings,
        ) = self.evidence_validator.filter_evidence(
            request.evidence,
            now=now,
        )

        if (
            not accepted_evidence
            and self.config.require_verified_evidence
        ):
            raise CommunicationEvidenceError(
                "No fresh verified evidence available"
            )

        details = (
            self.content_builder.build_details(
                request.communication_type,
                accepted_evidence,
            )
        )

        if not details:
            if (
                request.communication_type
                == CommunicationType.GENERAL_INFORMATION
                and self.config.allow_unverified_general_information
            ):
                details = (
                    "No verified operational details available."
                )

            else:
                raise CommunicationEvidenceError(
                    "No approved facts for this message type"
                )

        district = (
            CommunicationPrivacyGuard.redact(
                normalize_whitespace(
                    request.district
                )
            )
        )

        subject, body = (
            self.template_engine.render(
                language=request.language,
                communication_type=(
                    request.communication_type
                ),
                district=district,
                details=details,
            )
        )

        subject = (
            CommunicationPrivacyGuard.redact(
                subject
            )
        )

        body = (
            CommunicationPrivacyGuard.redact(
                body
            )
        )

        self.message_validator.validate(
            subject=subject,
            body=body,
            channel=request.channel,
        )

        content_hash = hashlib.sha256(
            (
                request.communication_type.value
                + "|"
                + request.audience.value
                + "|"
                + request.channel.value
                + "|"
                + request.language.value
                + "|"
                + district.casefold()
                + "|"
                + subject
                + "|"
                + body
            ).encode("utf-8")
        ).hexdigest()

        duplicate = (
            self.deduplication_engine.check_and_record(
                content_hash,
                now=now,
            )
        )

        warnings = list(
            evidence_warnings
        )

        warnings.extend([
            "Draft only: message has not been delivered",
            "Verify facts and intended recipients before approval",
            "Human review is required",
            "Automated privacy redaction is not comprehensive",
        ])

        if duplicate:
            warnings.append(
                "Similar message was recently drafted"
            )

        draft = CommunicationDraft(
            draft_id=generate_id(),
            request_id=request.request_id,
            communication_type=(
                request.communication_type
            ),
            audience=request.audience,
            channel=request.channel,
            language=request.language,
            district=district,
            priority=request.priority,
            subject=subject,
            body=body,
            evidence_ids=tuple(
                item.evidence_id
                for item in accepted_evidence
            ),
            status=(
                CommunicationStatus.PENDING_REVIEW
            ),
            requires_human_approval=True,
            duplicate_detected=duplicate,
            warnings=tuple(warnings),
            created_at=now,
            content_hash=content_hash,
        )

        with self._lock:
            self._drafts[
                draft.draft_id
            ] = draft

        self.audit_logger.record(
            draft_id=draft.draft_id,
            action="draft_created",
            actor_id="communication_agent",
            details={
                "request_id": request.request_id,
                "communication_type": (
                    request.communication_type.value
                ),
                "channel": request.channel.value,
                "language": request.language.value,
                "duplicate": str(duplicate),
            },
        )

        return CommunicationAgentOutput(
            run_id=generate_id(),
            draft=draft,
            accepted_evidence_ids=tuple(
                item.evidence_id
                for item in accepted_evidence
            ),
            rejected_evidence_ids=rejected_ids,
            generated_at=now,
        )

    def review_draft(
        self,
        draft_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        approved: bool,
        reason: str,
    ) -> CommunicationDraft:
        with self._lock:
            draft = self._drafts.get(
                draft_id
            )

            if draft is None:
                raise CommunicationApprovalError(
                    "Draft not found"
                )

            self.review_manager.review(
                draft,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                approved=approved,
                reason=reason,
            )

            updated = replace(
                draft,
                status=(
                    CommunicationStatus.APPROVED
                    if approved
                    else CommunicationStatus.REJECTED
                ),
            )

            self._drafts[
                draft_id
            ] = updated

            return updated

    def get_draft(
        self,
        draft_id: str,
    ) -> CommunicationDraft | None:
        with self._lock:
            return self._drafts.get(
                draft_id
            )

    def get_audit_history(
        self,
        draft_id: str,
    ) -> tuple[
        CommunicationAuditRecord,
        ...
    ]:
        return self.audit_logger.list_records(
            draft_id
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class CommunicationReportExporter:
    @staticmethod
    def export_json(
        output: CommunicationAgentOutput,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise CommunicationExportError(
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
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_communication_agent(
    config: CommunicationAgentConfig | None = None,
) -> CommunicationAgent:
    return CommunicationAgent(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    now = utc_now()

    evidence = CommunicationEvidence(
        evidence_id="EVIDENCE-001",
        source_id="DISTRICT-CONTROL-ROOM",
        source_name="Verified District Information",
        observed_at=now - timedelta(
            minutes=10
        ),
        verification_status=(
            EvidenceVerificationStatus.VERIFIED
        ),
        facts={
            "official_status": (
                "Flood-related conditions are "
                "being monitored by district officials."
            ),
            "rainfall_summary": (
                "Recent heavy rainfall has been reported."
            ),
        },
    )

    request = CommunicationRequest(
        request_id=generate_id(),
        communication_type=(
            CommunicationType.FLOOD_UPDATE
        ),
        audience=CommunicationAudience.PUBLIC,
        channel=CommunicationChannel.DASHBOARD,
        language=CommunicationLanguage.ENGLISH,
        district="Thrissur",
        priority=CommunicationPriority.IMPORTANT,
        evidence=(evidence,),
        context={},
        created_at=now,
    )

    agent = create_communication_agent()

    output = agent.run(
        request,
        now=now,
    )

    print(
        json.dumps(
            output.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    CommunicationReportExporter.export_json(
        output,
        "outputs/agents/communication_report.json",
    )

    # Demonstration of a review decision.
    # This does not send the message.
    updated_draft = agent.review_draft(
        output.draft.draft_id,
        reviewer_id="reviewer-001",
        reviewer_role="district_official",
        approved=False,
        reason="Awaiting updated official confirmation",
    )

    print(
        json.dumps(
            updated_draft.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CommunicationAgentError",
    "CommunicationValidationError",
    "CommunicationEvidenceError",
    "CommunicationApprovalError",
    "CommunicationExportError",
    "CommunicationLanguage",
    "CommunicationChannel",
    "CommunicationAudience",
    "CommunicationType",
    "CommunicationPriority",
    "CommunicationStatus",
    "EvidenceVerificationStatus",
    "CommunicationAgentConfig",
    "CommunicationEvidence",
    "CommunicationRequest",
    "CommunicationDraft",
    "CommunicationReview",
    "CommunicationAuditRecord",
    "CommunicationAgentOutput",
    "CommunicationRequestValidator",
    "CommunicationEvidenceValidator",
    "CommunicationPrivacyGuard",
    "CommunicationTemplateEngine",
    "CommunicationContentBuilder",
    "CommunicationMessageValidator",
    "CommunicationDeduplicationEngine",
    "CommunicationAuditLogger",
    "CommunicationReviewManager",
    "CommunicationAgent",
    "CommunicationReportExporter",
    "create_communication_agent",
]
