from __future__ import annotations

"""
backend/models/agent_decision.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy ORM models for AI agent decisions.

Features:
- Multi-agent decision records
- AI-generated recommendations
- Human approval and rejection
- Decision confidence and uncertainty
- Evidence and RAG citation references
- Model and prompt version tracking
- Safety and risk assessment
- Agent execution metadata
- Human review audit trail
- Decision lifecycle management
- Expiration and supersession
- Idempotency and trace correlation
- Optimistic concurrency
- PostgreSQL-compatible JSON storage

AI recommendations are advisory.
A stored or approved decision does not itself execute
any emergency operation, contact responders, or
authorize autonomous rescue dispatch.
"""


import enum
import uuid

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    Float,
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
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from sqlalchemy.ext.mutable import MutableDict, MutableList


# ============================================================
# DATABASE BASE
# ============================================================

# Replace this import with your project's shared Base
# if backend/database/base.py already defines it.
#
# from backend.database.base import Base


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(enum_class: type[enum.Enum]) -> SQLEnum:
    return SQLEnum(
        enum_class,
        values_callable=lambda values: [
            item.value for item in values
        ],
        native_enum=False,
        validate_strings=True,
        length=64,
    )


# ============================================================
# ENUMS
# ============================================================

class AgentType(str, enum.Enum):
    SUPERVISOR = "supervisor"
    WEATHER = "weather"
    FLOOD_PREDICTION = "flood_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    ALERT_INTELLIGENCE = "alert_intelligence"
    SOS_TRIAGE = "sos_triage"
    RESCUE_COORDINATOR = "rescue_coordinator"
    ROUTE_PLANNING = "route_planning"
    EVACUATION = "evacuation"
    SHELTER_ALLOCATION = "shelter_allocation"
    RELIEF_LOGISTICS = "relief_logistics"
    MEDICAL_RESPONSE = "medical_response"
    VOLUNTEER_COORDINATOR = "volunteer_coordinator"
    MISSING_PERSON = "missing_person"
    DAMAGE_ASSESSMENT = "damage_assessment"
    RECOVERY_PLANNING = "recovery_planning"
    COMMUNICATION = "communication"
    VERIFICATION = "verification"
    HUMAN_APPROVAL = "human_approval"


class DecisionType(str, enum.Enum):
    FLOOD_FORECAST = "flood_forecast"
    FLOOD_RISK_ASSESSMENT = "flood_risk_assessment"
    EARLY_WARNING = "early_warning"
    SOS_PRIORITY = "sos_priority"
    INCIDENT_PRIORITY = "incident_priority"
    RESCUE_RECOMMENDATION = "rescue_recommendation"
    RESCUE_TEAM_MATCHING = "rescue_team_matching"
    ROUTE_RECOMMENDATION = "route_recommendation"
    EVACUATION_RECOMMENDATION = "evacuation_recommendation"
    SHELTER_RECOMMENDATION = "shelter_recommendation"
    RESOURCE_ALLOCATION = "resource_allocation"
    RELIEF_DISTRIBUTION = "relief_distribution"
    MEDICAL_COORDINATION = "medical_coordination"
    VOLUNTEER_MATCHING = "volunteer_matching"
    MISSING_PERSON_SEARCH = "missing_person_search"
    DAMAGE_ASSESSMENT = "damage_assessment"
    RECOVERY_PRIORITIZATION = "recovery_prioritization"
    COMMUNICATION_DRAFT = "communication_draft"
    DATA_VERIFICATION = "data_verification"
    GENERAL_RECOMMENDATION = "general_recommendation"


class DecisionStatus(str, enum.Enum):
    DRAFT = "draft"
    GENERATED = "generated"
    VALIDATING = "validating"
    VALIDATED = "validated"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"


class DecisionPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionRiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DecisionReviewOutcome(str, enum.Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


class DecisionSourceType(str, enum.Enum):
    SENSOR = "sensor"
    WEATHER_API = "weather_api"
    OFFICIAL_ALERT = "official_alert"
    SATELLITE = "satellite"
    DRONE = "drone"
    USER_REPORT = "user_report"
    FIELD_REPORT = "field_report"
    DATABASE = "database"
    RAG_DOCUMENT = "rag_document"
    ML_MODEL = "ml_model"
    HUMAN_REVIEW = "human_review"
    OTHER = "other"


class AgentExecutionStatus(str, enum.Enum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


# ============================================================
# AGENT DECISION MODEL
# ============================================================

class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    district_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
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
        index=True,
    )

    rescue_mission_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    workflow_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    execution_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "agent_executions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    agent_type: Mapped[AgentType] = mapped_column(
        enum_type(AgentType),
        nullable=False,
        index=True,
    )

    agent_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    decision_type: Mapped[DecisionType] = mapped_column(
        enum_type(DecisionType),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    recommendation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    reasoning_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[DecisionStatus] = mapped_column(
        enum_type(DecisionStatus),
        default=DecisionStatus.DRAFT,
        nullable=False,
        index=True,
    )

    priority: Mapped[DecisionPriority] = mapped_column(
        enum_type(DecisionPriority),
        default=DecisionPriority.MEDIUM,
        nullable=False,
    )

    risk_level: Mapped[DecisionRiskLevel] = mapped_column(
        enum_type(DecisionRiskLevel),
        default=DecisionRiskLevel.UNKNOWN,
        nullable=False,
    )

    confidence_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    uncertainty_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    safety_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    evidence_quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    requires_human_approval: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    is_safety_critical: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    validation_passed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    validation_errors: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )

    safety_flags: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )

    input_summary: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    structured_recommendation: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    alternatives: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )

    constraints: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    assumptions: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )

    model_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    prompt_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    agent_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
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

    idempotency_key: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        unique=True,
    )

    parent_decision_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "agent_decisions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    reviewed_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    review_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    rejected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    execution: Mapped["AgentExecution | None"] = relationship(
        "AgentExecution",
        back_populates="decisions",
    )

    evidence: Mapped[list["AgentDecisionEvidence"]] = relationship(
        "AgentDecisionEvidence",
        back_populates="decision",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    reviews: Mapped[list["AgentDecisionReview"]] = relationship(
        "AgentDecisionReview",
        back_populates="decision",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    parent_decision: Mapped["AgentDecision | None"] = relationship(
        "AgentDecision",
        remote_side="AgentDecision.id",
        back_populates="child_decisions",
    )

    child_decisions: Mapped[list["AgentDecision"]] = relationship(
        "AgentDecision",
        back_populates="parent_decision",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_agent_decision_district_positive",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_agent_decision_version_positive",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 AND confidence_score <= 1)",
            name="ck_agent_decision_confidence_range",
        ),
        CheckConstraint(
            "uncertainty_score IS NULL OR "
            "(uncertainty_score >= 0 AND uncertainty_score <= 1)",
            name="ck_agent_decision_uncertainty_range",
        ),
        CheckConstraint(
            "safety_score IS NULL OR "
            "(safety_score >= 0 AND safety_score <= 1)",
            name="ck_agent_decision_safety_range",
        ),
        CheckConstraint(
            "evidence_quality_score IS NULL OR "
            "(evidence_quality_score >= 0 "
            "AND evidence_quality_score <= 1)",
            name="ck_agent_decision_evidence_quality_range",
        ),
        Index(
            "ix_agent_decision_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_agent_decision_agent_created",
            "agent_type",
            "created_at",
        ),
        Index(
            "ix_agent_decision_priority_status",
            "priority",
            "status",
        ),
    )

    @validates(
        "confidence_score",
        "uncertainty_score",
        "safety_score",
        "evidence_quality_score",
    )
    def validate_score(
        self,
        key: str,
        value: float | None,
    ) -> float | None:

        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math_is_finite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"{key} must be between 0 and 1"
            )

        return float(value)

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            DecisionStatus.REJECTED,
            DecisionStatus.EXPIRED,
            DecisionStatus.SUPERSEDED,
            DecisionStatus.CANCELLED,
        }

    @property
    def is_approved(self) -> bool:
        return (
            self.status == DecisionStatus.APPROVED
            and self.approved_at is not None
        )

    @property
    def needs_review(self) -> bool:
        return (
            self.requires_human_approval
            and self.status == DecisionStatus.PENDING_REVIEW
        )

    def is_expired(
        self,
        now: datetime | None = None,
    ) -> bool:

        if self.expires_at is None:
            return False

        now = now or utc_now()

        return self.expires_at <= now

    def can_be_approved(self) -> bool:
        return (
            self.status == DecisionStatus.PENDING_REVIEW
            and self.validation_passed
            and self.is_verified
            and not self.is_expired()
        )

    def submit_for_review(self) -> None:

        if self.status != DecisionStatus.VALIDATED:
            raise ValueError(
                "Only validated decisions can be "
                "submitted for review"
            )

        if not self.validation_passed:
            raise ValueError(
                "Decision validation has not passed"
            )

        self.status = DecisionStatus.PENDING_REVIEW
        self.updated_at = utc_now()

    def approve(
        self,
        *,
        reviewer_id: int,
        notes: str,
    ) -> None:

        if not self.can_be_approved():
            raise ValueError(
                "Decision is not eligible for approval"
            )

        if (
            self.created_by_user_id is not None
            and reviewer_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent review is required"
            )

        if reviewer_id <= 0:
            raise ValueError(
                "Invalid reviewer ID"
            )

        if not notes.strip():
            raise ValueError(
                "Approval notes are required"
            )

        self.status = DecisionStatus.APPROVED
        self.reviewed_by_user_id = reviewer_id
        self.review_notes = notes.strip()
        self.approved_at = utc_now()
        self.updated_at = utc_now()

    def reject(
        self,
        *,
        reviewer_id: int,
        reason: str,
    ) -> None:

        if self.status != DecisionStatus.PENDING_REVIEW:
            raise ValueError(
                "Decision is not pending review"
            )

        if reviewer_id <= 0:
            raise ValueError(
                "Invalid reviewer ID"
            )

        if not reason.strip():
            raise ValueError(
                "Rejection reason is required"
            )

        self.status = DecisionStatus.REJECTED
        self.reviewed_by_user_id = reviewer_id
        self.review_notes = reason.strip()
        self.rejected_at = utc_now()
        self.updated_at = utc_now()

    def supersede(self) -> None:

        if self.is_terminal:
            raise ValueError(
                "Terminal decision cannot be superseded"
            )

        self.status = DecisionStatus.SUPERSEDED
        self.superseded_at = utc_now()
        self.updated_at = utc_now()

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Return operational metadata without raw
        personal information or internal model inputs.
        """

        return {
            "id": self.id,
            "district_id": self.district_id,
            "incident_id": self.incident_id,
            "agent_type": self.agent_type.value,
            "decision_type": self.decision_type.value,
            "title": self.title,
            "status": self.status.value,
            "priority": self.priority.value,
            "risk_level": self.risk_level.value,
            "confidence_score": self.confidence_score,
            "uncertainty_score": self.uncertainty_score,
            "requires_human_approval": (
                self.requires_human_approval
            ),
            "is_safety_critical": self.is_safety_critical,
            "is_verified": self.is_verified,
            "validation_passed": self.validation_passed,
            "is_approved": self.is_approved,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
            "approved_at": (
                self.approved_at.isoformat()
                if self.approved_at
                else None
            ),
            "expires_at": (
                self.expires_at.isoformat()
                if self.expires_at
                else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<AgentDecision("
            f"id={self.id!r}, "
            f"agent={self.agent_type.value!r}, "
            f"status={self.status.value!r}"
            f")>"
        )


# ============================================================
# DECISION EVIDENCE
# ============================================================

class AgentDecisionEvidence(Base):
    __tablename__ = "agent_decision_evidence"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    decision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "agent_decisions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    source_type: Mapped[DecisionSourceType] = mapped_column(
        enum_type(DecisionSourceType),
        nullable=False,
    )

    source_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    source_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    citation_reference: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    relevance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    reliability_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    verified_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
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

    decision: Mapped["AgentDecision"] = relationship(
        "AgentDecision",
        back_populates="evidence",
    )

    __table_args__ = (
        CheckConstraint(
            "relevance_score IS NULL OR "
            "(relevance_score >= 0 AND relevance_score <= 1)",
            name="ck_agent_evidence_relevance_range",
        ),
        CheckConstraint(
            "reliability_score IS NULL OR "
            "(reliability_score >= 0 AND reliability_score <= 1)",
            name="ck_agent_evidence_reliability_range",
        ),
        Index(
            "ix_agent_evidence_decision_source",
            "decision_id",
            "source_type",
        ),
    )

    @validates(
        "relevance_score",
        "reliability_score",
    )
    def validate_score(
        self,
        key: str,
        value: float | None,
    ) -> float | None:

        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math_is_finite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"{key} must be between 0 and 1"
            )

        return float(value)


# ============================================================
# HUMAN REVIEW HISTORY
# ============================================================

class AgentDecisionReview(Base):
    __tablename__ = "agent_decision_reviews"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    decision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "agent_decisions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    reviewer_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    outcome: Mapped[DecisionReviewOutcome] = mapped_column(
        enum_type(DecisionReviewOutcome),
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

    decision: Mapped["AgentDecision"] = relationship(
        "AgentDecision",
        back_populates="reviews",
    )

    __table_args__ = (
        CheckConstraint(
            "reviewer_id > 0",
            name="ck_agent_review_reviewer_positive",
        ),
        Index(
            "ix_agent_review_decision_time",
            "decision_id",
            "reviewed_at",
        ),
    )


# ============================================================
# AGENT EXECUTION MODEL
# ============================================================

class AgentExecution(Base):
    __tablename__ = "agent_executions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    district_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    workflow_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    agent_type: Mapped[AgentType] = mapped_column(
        enum_type(AgentType),
        nullable=False,
    )

    agent_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    status: Mapped[AgentExecutionStatus] = mapped_column(
        enum_type(AgentExecutionStatus),
        default=AgentExecutionStatus.STARTED,
        nullable=False,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    trace_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    input_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    estimated_cost_usd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    decisions: Mapped[list["AgentDecision"]] = relationship(
        "AgentDecision",
        back_populates="execution",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_agent_execution_district_positive",
        ),
        CheckConstraint(
            "input_tokens >= 0",
            name="ck_agent_execution_input_tokens",
        ),
        CheckConstraint(
            "output_tokens >= 0",
            name="ck_agent_execution_output_tokens",
        ),
        CheckConstraint(
            "retry_count >= 0",
            name="ck_agent_execution_retry_count",
        ),
        CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="ck_agent_execution_latency",
        ),
        CheckConstraint(
            "estimated_cost_usd IS NULL OR "
            "estimated_cost_usd >= 0",
            name="ck_agent_execution_cost",
        ),
    )


# ============================================================
# HELPER VALIDATION
# ============================================================

def math_is_finite(value: float) -> bool:
    import math
    return math.isfinite(value)


# ============================================================
# OPTIMISTIC LOCKING
# ============================================================

AgentDecision.__mapper_args__ = {
    "version_id_col": AgentDecision.__table__.c.version,
}


# ============================================================
# TIMESTAMP HOOKS
# ============================================================

@event.listens_for(
    AgentDecision,
    "before_update",
)
def update_decision_timestamp(
    mapper: Any,
    connection: Any,
    target: AgentDecision,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

AGENT_DECISION_MODELS = (
    AgentDecision,
    AgentDecisionEvidence,
    AgentDecisionReview,
    AgentExecution,
)
