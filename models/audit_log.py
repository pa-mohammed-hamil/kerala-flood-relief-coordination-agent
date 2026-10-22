from __future__ import annotations

"""
backend/models/audit_log.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x audit logging models.

Features:
- User, administrator, AI agent and system activity
- Authentication and authorization audit events
- SOS, rescue, evacuation and shelter audit events
- Volunteer, medical and relief coordination events
- AI decision and human approval tracking
- Before/after change summaries
- Security incident and access-denial tracking
- Request, trace and correlation identifiers
- Timestamped append-only audit records
- Tamper-evident SHA-256 hash chains
- JSON metadata and contextual references
- Actor, action, resource and district indexes
- PostgreSQL-compatible database schema
- Sensitive-data minimization
- Audit event validation

SECURITY NOTES:
- Application users must not update or delete audit rows.
- Hash chains detect changes only when trusted hash
  checkpoints are stored independently.
- Use database permissions, append-only triggers,
  external logging and secure backups in production.
- Do not store passwords, tokens, private medical
  records or unnecessary personal information.
- Audit writes should participate in the same
  database transaction as the operation being audited.
"""


import enum
import hashlib
import json
import math
import re
import uuid

from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    event,
    inspect,
)

from sqlalchemy.orm import (
    Mapped,
    Session,
    mapped_column,
    relationship,
    validates,
)

from sqlalchemy.ext.mutable import (
    MutableDict,
)

from backend.database.base import Base


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(enum_class: type[enum.Enum]) -> SQLEnum:
    return SQLEnum(
        enum_class,
        values_callable=lambda members: [
            member.value for member in members
        ],
        native_enum=False,
        validate_strings=True,
        length=80,
    )


def require_text(
    value: str,
    name: str,
    *,
    max_length: int = 500,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")

    result = value.strip()

    if len(result) > max_length:
        raise ValueError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def normalize_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Audit timestamps must be timezone-aware"
        )

    return value.astimezone(
        timezone.utc
    ).isoformat(
        timespec="microseconds"
    )


# ============================================================
# ENUMERATIONS
# ============================================================

class AuditActorType(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"
    VOLUNTEER = "volunteer"
    RESCUE_OPERATOR = "rescue_operator"
    MEDICAL_OPERATOR = "medical_operator"
    DISTRICT_OFFICER = "district_officer"
    STATE_OFFICER = "state_officer"
    AI_AGENT = "ai_agent"
    SYSTEM = "system"
    EXTERNAL_SERVICE = "external_service"
    ANONYMOUS = "anonymous"


class AuditAction(str, enum.Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"

    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"

    REGISTER = "register"
    VERIFY = "verify"

    APPROVE = "approve"
    REJECT = "reject"

    ASSIGN = "assign"
    UNASSIGN = "unassign"

    DISPATCH_REQUEST = "dispatch_request"
    DISPATCH_APPROVE = "dispatch_approve"

    ESCALATE = "escalate"
    ACKNOWLEDGE = "acknowledge"

    START = "start"
    PAUSE = "pause"
    RESUME = "resume"

    COMPLETE = "complete"
    CANCEL = "cancel"

    NOTIFY = "notify"
    EXPORT = "export"
    IMPORT = "import"

    GENERATE = "generate"
    PREDICT = "predict"
    EVALUATE = "evaluate"

    MODEL_INFERENCE = "model_inference"
    AGENT_DECISION = "agent_decision"

    HUMAN_REVIEW = "human_review"
    SAFETY_OVERRIDE = "safety_override"

    SECURITY_ALERT = "security_alert"
    CONFIGURATION_CHANGE = "configuration_change"

    DATA_ACCESS = "data_access"
    DATA_REDACTION = "data_redaction"

    OTHER = "other"


class AuditResourceType(str, enum.Enum):
    USER = "user"
    DISTRICT = "district"
    FLOOD_ALERT = "flood_alert"
    FLOOD_FORECAST = "flood_forecast"
    WEATHER_STATION = "weather_station"
    RIVER_STATION = "river_station"
    DAM = "dam"

    SOS_REQUEST = "sos_request"
    INCIDENT = "incident"

    RESCUE_TEAM = "rescue_team"
    RESCUE_MISSION = "rescue_mission"

    EVACUATION = "evacuation"
    SHELTER = "shelter"
    SHELTER_RESIDENT = "shelter_resident"

    INVENTORY = "inventory"
    RELIEF_DELIVERY = "relief_delivery"

    HOSPITAL = "hospital"
    MEDICAL_REQUEST = "medical_request"

    VOLUNTEER = "volunteer"
    VOLUNTEER_TASK = "volunteer_task"
    VOLUNTEER_ASSIGNMENT = "volunteer_assignment"

    MISSING_PERSON = "missing_person"
    DAMAGE_REPORT = "damage_report"
    RECOVERY_PLAN = "recovery_plan"

    AGENT_DECISION = "agent_decision"
    AGENT_EXECUTION = "agent_execution"

    NOTIFICATION = "notification"
    REPORT = "report"

    SECURITY = "security"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    OTHER = "other"


class AuditSeverity(str, enum.Enum):
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditOutcome(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    PARTIAL = "partial"
    PENDING = "pending"


class AuditReviewStatus(str, enum.Enum):
    UNREVIEWED = "unreviewed"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    ESCALATED = "escalated"


# ============================================================
# SENSITIVE DATA PROTECTION
# ============================================================

SENSITIVE_FIELD_NAMES = frozenset({
    "password",
    "password_hash",
    "secret",
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
    "session_token",
    "private_key",
    "otp",
    "pin",
    "credit_card",
    "card_number",
    "medical_history",
    "diagnosis",
    "patient_notes",
    "aadhaar",
    "aadhaar_number",
})


def sanitize_audit_value(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """
    Redact known sensitive fields.

    Only JSON-compatible values should be submitted
    for audit storage. Unknown objects are rejected
    rather than stringified.
    """

    if depth > 8:
        raise ValueError(
            "Audit metadata nesting exceeds limit"
        )

    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        if isinstance(value, str):
            return value[:2000]
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Non-finite audit metadata number"
            )
        return value

    if isinstance(value, datetime):
        return normalize_datetime(value)

    if isinstance(value, enum.Enum):
        return value.value

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    "Audit metadata keys must be strings"
                )

            normalized_key = re.sub(
                r"[^a-z0-9]+",
                "_",
                key.lower(),
            ).strip("_")

            if normalized_key in SENSITIVE_FIELD_NAMES:
                result[key] = "[REDACTED]"
            else:
                result[key] = sanitize_audit_value(
                    item,
                    depth=depth + 1,
                )

        return result

    if isinstance(value, (list, tuple)):
        return [
            sanitize_audit_value(
                item,
                depth=depth + 1,
            )
            for item in value
        ]

    raise ValueError(
        f"Unsupported audit metadata type: "
        f"{type(value).__name__}"
    )


# ============================================================
# HASHING
# ============================================================

AUDIT_HASH_VERSION = 1

GENESIS_HASH = "0" * 64


def canonical_json(
    payload: Mapping[str, Any],
) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def calculate_audit_hash(
    payload: Mapping[str, Any],
    *,
    previous_hash: str,
) -> str:
    """
    Calculate a tamper-evident hash for one audit entry.

    The caller must serialize writes to a hash chain
    and persist sequence/previous_hash atomically.
    """

    if not re.fullmatch(
        r"[a-fA-F0-9]{64}",
        previous_hash,
    ):
        raise ValueError(
            "Invalid previous audit hash"
        )

    message = canonical_json({
        "hash_version": AUDIT_HASH_VERSION,
        "previous_hash": previous_hash.lower(),
        "payload": payload,
    })

    return hashlib.sha256(
        message.encode("utf-8")
    ).hexdigest()


# ============================================================
# AUDIT LOG MODEL
# ============================================================

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    # Global sequence allocated transactionally by
    # the audit repository or database.
    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        unique=True,
    )

    actor_type: Mapped[AuditActorType] = mapped_column(
        enum_type(AuditActorType),
        nullable=False,
        index=True,
    )

    actor_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    actor_display_reference: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    agent_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    action: Mapped[AuditAction] = mapped_column(
        enum_type(AuditAction),
        nullable=False,
        index=True,
    )

    action_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    resource_type: Mapped[AuditResourceType] = mapped_column(
        enum_type(AuditResourceType),
        nullable=False,
        index=True,
    )

    resource_id: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    district_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )

    incident_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    sos_request_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    agent_decision_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    request_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    trace_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    session_reference: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    source_service: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="backend",
    )

    source_module: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    ip_address_reference: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    user_agent_summary: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

    severity: Mapped[AuditSeverity] = mapped_column(
        enum_type(AuditSeverity),
        default=AuditSeverity.INFO,
        nullable=False,
        index=True,
    )

    outcome: Mapped[AuditOutcome] = mapped_column(
        enum_type(AuditOutcome),
        default=AuditOutcome.SUCCESS,
        nullable=False,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reason_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    before_state: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    after_state: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    changes: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    requires_review: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    hash_version: Mapped[int] = mapped_column(
        Integer,
        default=AUDIT_HASH_VERSION,
        nullable=False,
    )

    previous_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    entry_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )

    __table_args__ = (
        CheckConstraint(
            "sequence_number > 0",
            name="ck_audit_sequence_positive",
        ),
        CheckConstraint(
            "district_id IS NULL OR district_id > 0",
            name="ck_audit_district_positive",
        ),
        CheckConstraint(
            "hash_version > 0",
            name="ck_audit_hash_version_positive",
        ),
        Index(
            "ix_audit_district_occurred",
            "district_id",
            "occurred_at",
        ),
        Index(
            "ix_audit_resource_occurred",
            "resource_type",
            "resource_id",
            "occurred_at",
        ),
        Index(
            "ix_audit_actor_occurred",
            "actor_type",
            "actor_id",
            "occurred_at",
        ),
        Index(
            "ix_audit_severity_outcome",
            "severity",
            "outcome",
        ),
    )

    @validates("action_name")
    def validate_action_name(
        self,
        key: str,
        value: str,
    ) -> str:
        return require_text(
            value,
            key,
            max_length=150,
        )

    @validates("source_service")
    def validate_source_service(
        self,
        key: str,
        value: str,
    ) -> str:
        return require_text(
            value,
            key,
            max_length=150,
        )

    def hash_payload(self) -> dict[str, Any]:
        """
        Canonical set of audit fields included in
        the integrity hash.

        Do not change this schema without incrementing
        hash_version and supporting historical hashes.
        """

        return {
            "id": self.id,
            "sequence_number": self.sequence_number,
            "actor_type": self.actor_type.value,
            "actor_id": self.actor_id,
            "actor_display_reference": (
                self.actor_display_reference
            ),
            "agent_name": self.agent_name,
            "action": self.action.value,
            "action_name": self.action_name,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "district_id": self.district_id,
            "incident_id": self.incident_id,
            "sos_request_id": self.sos_request_id,
            "agent_decision_id": self.agent_decision_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "session_reference": self.session_reference,
            "source_service": self.source_service,
            "source_module": self.source_module,
            "ip_address_reference": (
                self.ip_address_reference
            ),
            "user_agent_summary": (
                self.user_agent_summary
            ),
            "severity": self.severity.value,
            "outcome": self.outcome.value,
            "description": self.description,
            "reason_code": self.reason_code,
            "error_code": self.error_code,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "changes": self.changes,
            "metadata": self.metadata_json,
            "requires_review": self.requires_review,
            "occurred_at": normalize_datetime(
                self.occurred_at
            ),
            "recorded_at": normalize_datetime(
                self.recorded_at
            ),
        }

    def calculate_hash(self) -> str:
        return calculate_audit_hash(
            self.hash_payload(),
            previous_hash=self.previous_hash,
        )

    def verify_hash(self) -> bool:
        if self.hash_version != AUDIT_HASH_VERSION:
            return False

        return (
            self.calculate_hash()
            == self.entry_hash
        )

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Minimal safe representation for dashboards.
        """

        return {
            "id": self.id,
            "sequence_number": self.sequence_number,
            "actor_type": self.actor_type.value,
            "action": self.action.value,
            "action_name": self.action_name,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "district_id": self.district_id,
            "severity": self.severity.value,
            "outcome": self.outcome.value,
            "requires_review": self.requires_review,
            "occurred_at": (
                self.occurred_at.isoformat()
                if self.occurred_at
                else None
            ),
            "trace_id": self.trace_id,
        }

    def __repr__(self) -> str:
        return (
            f"<AuditLog("
            f"id={self.id!r}, "
            f"sequence={self.sequence_number!r}, "
            f"action={self.action.value!r}, "
            f"outcome={self.outcome.value!r}"
            f")>"
        )


# ============================================================
# AUDIT REVIEW MODEL
# ============================================================

class AuditLogReview(Base):
    __tablename__ = "audit_log_reviews"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    audit_log_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "audit_logs.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    reviewer_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    review_status: Mapped[AuditReviewStatus] = mapped_column(
        enum_type(AuditReviewStatus),
        default=AuditReviewStatus.UNREVIEWED,
        nullable=False,
    )

    notes: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    evidence_reference: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    audit_log: Mapped["AuditLog"] = relationship(
        "AuditLog",
    )

    __table_args__ = (
        CheckConstraint(
            "reviewer_id > 0",
            name="ck_audit_review_reviewer_positive",
        ),
        Index(
            "ix_audit_review_log_time",
            "audit_log_id",
            "reviewed_at",
        ),
    )


# ============================================================
# AUDIT INTEGRITY CHECKPOINT
# ============================================================

class AuditIntegrityCheckpoint(Base):
    __tablename__ = "audit_integrity_checkpoints"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        unique=True,
    )

    entry_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    verified_by_reference: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    verified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    external_anchor_reference: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    verification_details: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "sequence_number > 0",
            name="ck_audit_checkpoint_sequence_positive",
        ),
    )


# ============================================================
# AUDIT LOG FACTORY
# ============================================================

def create_audit_log(
    *,
    sequence_number: int,
    previous_hash: str,
    actor_type: AuditActorType,
    action: AuditAction,
    resource_type: AuditResourceType,
    action_name: str,
    source_service: str,
    actor_id: str | None = None,
    resource_id: str | None = None,
    district_id: int | None = None,
    severity: AuditSeverity = AuditSeverity.INFO,
    outcome: AuditOutcome = AuditOutcome.SUCCESS,
    description: str | None = None,
    before_state: Mapping[str, Any] | None = None,
    after_state: Mapping[str, Any] | None = None,
    changes: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
    correlation_id: str | None = None,
    requires_review: bool = False,
    occurred_at: datetime | None = None,
) -> AuditLog:
    """
    Create a validated, hash-linked audit record.

    Sequence allocation and previous-hash selection
    must occur inside a serialized database transaction.
    """

    if (
        isinstance(sequence_number, bool)
        or not isinstance(sequence_number, int)
        or sequence_number <= 0
    ):
        raise ValueError(
            "sequence_number must be positive"
        )

    if (
        district_id is not None
        and (
            isinstance(district_id, bool)
            or not isinstance(district_id, int)
            or district_id <= 0
        )
    ):
        raise ValueError(
            "Invalid district_id"
        )

    occurred_at = occurred_at or utc_now()

    normalize_datetime(occurred_at)

    if not re.fullmatch(
        r"[a-fA-F0-9]{64}",
        previous_hash,
    ):
        raise ValueError(
            "Invalid previous_hash"
        )

    log = AuditLog(
        id=new_uuid(),
        sequence_number=sequence_number,
        previous_hash=previous_hash.lower(),
        entry_hash=GENESIS_HASH,
        hash_version=AUDIT_HASH_VERSION,
        actor_type=actor_type,
        actor_id=actor_id,
        action=action,
        action_name=require_text(
            action_name,
            "action_name",
            max_length=150,
        ),
        resource_type=resource_type,
        resource_id=resource_id,
        district_id=district_id,
        source_service=require_text(
            source_service,
            "source_service",
            max_length=150,
        ),
        severity=severity,
        outcome=outcome,
        description=(
            description[:2000]
            if description is not None
            else None
        ),
        before_state=sanitize_audit_value(
            before_state or {}
        ),
        after_state=sanitize_audit_value(
            after_state or {}
        ),
        changes=sanitize_audit_value(
            changes or {}
        ),
        metadata_json=sanitize_audit_value(
            metadata or {}
        ),
        request_id=request_id,
        trace_id=trace_id,
        correlation_id=correlation_id,
        requires_review=requires_review,
        occurred_at=occurred_at,
        recorded_at=utc_now(),
    )

    log.entry_hash = log.calculate_hash()

    return log


# ============================================================
# AUDIT CHAIN VERIFICATION
# ============================================================

def verify_audit_chain(
    entries: list[AuditLog],
    *,
    expected_previous_hash: str = GENESIS_HASH,
    expected_start_sequence: int = 1,
) -> dict[str, Any]:
    """
    Verify a contiguous sequence of audit records.

    For partial chains, supply the independently
    verified preceding hash and start sequence.
    """

    ordered = sorted(
        entries,
        key=lambda item: item.sequence_number,
    )

    previous_hash = expected_previous_hash.lower()
    expected_sequence = expected_start_sequence

    errors: list[str] = []

    for entry in ordered:
        if entry.sequence_number != expected_sequence:
            errors.append(
                f"Sequence mismatch at {entry.id}"
            )

        if entry.previous_hash != previous_hash:
            errors.append(
                f"Previous hash mismatch at {entry.id}"
            )

        try:
            if not entry.verify_hash():
                errors.append(
                    f"Entry hash mismatch at {entry.id}"
                )
        except (ValueError, TypeError, AttributeError):
            errors.append(
                f"Invalid hash payload at {entry.id}"
            )

        previous_hash = entry.entry_hash
        expected_sequence += 1

    return {
        "valid": len(errors) == 0,
        "checked_entries": len(ordered),
        "start_sequence": expected_start_sequence,
        "last_sequence": (
            ordered[-1].sequence_number
            if ordered
            else None
        ),
        "last_hash": previous_hash,
        "errors": errors,
        "verified_at": utc_now().isoformat(),
    }


# ============================================================
# APPEND-ONLY ORM GUARDS
# ============================================================

@event.listens_for(
    AuditLog,
    "before_update",
)
def prevent_audit_update(
    mapper: Any,
    connection: Any,
    target: AuditLog,
) -> None:
    raise RuntimeError(
        "Audit logs are append-only and cannot be updated"
    )


@event.listens_for(
    AuditLog,
    "before_delete",
)
def prevent_audit_delete(
    mapper: Any,
    connection: Any,
    target: AuditLog,
) -> None:
    raise RuntimeError(
        "Audit logs are append-only and cannot be deleted"
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

AUDIT_MODELS = (
    AuditLog,
    AuditLogReview,
    AuditIntegrityCheckpoint,
)


__all__ = [
    "AuditActorType",
    "AuditAction",
    "AuditResourceType",
    "AuditSeverity",
    "AuditOutcome",
    "AuditReviewStatus",
    "AuditLog",
    "AuditLogReview",
    "AuditIntegrityCheckpoint",
    "AUDIT_MODELS",
    "GENESIS_HASH",
    "AUDIT_HASH_VERSION",
    "create_audit_log",
    "verify_audit_chain",
    "calculate_audit_hash",
    "sanitize_audit_value",
    "utc_now",
]
