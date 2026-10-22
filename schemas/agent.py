from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/agent.py

Pydantic schemas for multi-agent AI coordination.

Supports:
- Supervisor Agent
- Weather Agent
- Flood Prediction Agent
- Risk Assessment Agent
- SOS Triage Agent
- Rescue Coordinator Agent
- Route Planning Agent
- Evacuation Agent
- Shelter Allocation Agent
- Relief Logistics Agent
- Medical Response Agent
- Volunteer Coordinator Agent
- Damage Assessment Agent
- Recovery Planning Agent
- Communication Agent
- Verification Agent
- Human Approval Agent

These schemas describe proposed agent workflows.
They do not independently execute rescue operations.
"""


from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# ENUMS
# ============================================================

class AgentType(str, Enum):
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


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OFFLINE = "offline"


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionType(str, Enum):
    FLOOD_RISK_ASSESSMENT = "flood_risk_assessment"
    FLOOD_ALERT_RECOMMENDATION = "flood_alert_recommendation"
    SOS_TRIAGE = "sos_triage"
    RESCUE_ASSIGNMENT = "rescue_assignment"
    RESCUE_ROUTE = "rescue_route"
    EVACUATION_PLAN = "evacuation_plan"
    SHELTER_ALLOCATION = "shelter_allocation"
    RELIEF_DISTRIBUTION = "relief_distribution"
    MEDICAL_RESPONSE = "medical_response"
    VOLUNTEER_ASSIGNMENT = "volunteer_assignment"
    DAMAGE_ASSESSMENT = "damage_assessment"
    RECOVERY_PLAN = "recovery_plan"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    WEATHER = "weather"
    RIVER_LEVEL = "river_level"
    DAM_LEVEL = "dam_level"
    SATELLITE = "satellite"
    SOS_REPORT = "sos_report"
    FIELD_REPORT = "field_report"
    DOCUMENT = "document"
    MODEL_PREDICTION = "model_prediction"
    OTHER = "other"


# ============================================================
# BASE SCHEMA
# ============================================================

class AgentBaseSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )


# ============================================================
# LOCATION
# ============================================================

class GeoLocation(AgentBaseSchema):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    district: str | None = None
    address: str | None = None


# ============================================================
# AGENT INFORMATION
# ============================================================

class AgentInfo(AgentBaseSchema):
    agent_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=120)
    agent_type: AgentType

    description: str | None = None

    status: AgentStatus = AgentStatus.IDLE

    capabilities: list[str] = Field(
        default_factory=list
    )

    model_name: str | None = None
    model_version: str | None = None

    enabled: bool = True

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


class AgentHealth(AgentBaseSchema):
    agent_id: UUID
    status: AgentStatus

    healthy: bool

    last_heartbeat: datetime | None = None

    active_tasks: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)

    average_latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    error_message: str | None = None


# ============================================================
# EVIDENCE AND CITATIONS
# ============================================================

class AgentEvidence(AgentBaseSchema):
    evidence_id: UUID = Field(
        default_factory=uuid4
    )

    evidence_type: EvidenceType

    source_name: str = Field(
        min_length=1,
        max_length=200,
    )

    source_reference: str | None = None

    observed_at: datetime | None = None
    retrieved_at: datetime = Field(
        default_factory=utc_now
    )

    verified: bool = False

    summary: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# AGENT TASKS
# ============================================================

class AgentTaskCreate(AgentBaseSchema):
    task_id: UUID = Field(default_factory=uuid4)

    task_name: str = Field(
        min_length=1,
        max_length=200,
    )

    description: str = Field(
        min_length=1,
        max_length=5000,
    )

    assigned_agent: AgentType

    priority: TaskPriority = TaskPriority.MEDIUM

    district: str | None = None
    location: GeoLocation | None = None

    input_data: dict[str, Any] = Field(
        default_factory=dict
    )

    parent_task_id: UUID | None = None

    dependencies: list[UUID] = Field(
        default_factory=list
    )

    requires_human_approval: bool = False

    created_by: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )


class AgentTaskResponse(AgentTaskCreate):
    status: TaskStatus = TaskStatus.PENDING

    output_data: dict[str, Any] | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None

    execution_time_ms: float | None = Field(
        default=None,
        ge=0,
    )

    error_message: str | None = None

    retry_count: int = Field(default=0, ge=0)

    evidence: list[AgentEvidence] = Field(
        default_factory=list
    )


# ============================================================
# AGENT EXECUTION
# ============================================================

class AgentExecutionRequest(AgentBaseSchema):
    execution_id: UUID = Field(
        default_factory=uuid4
    )

    agent_type: AgentType

    task_id: UUID | None = None

    instruction: str = Field(
        min_length=1,
        max_length=10000,
    )

    context: dict[str, Any] = Field(
        default_factory=dict
    )

    tools_allowed: list[str] = Field(
        default_factory=list
    )

    timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=3600,
    )

    dry_run: bool = True


class AgentExecutionResponse(AgentBaseSchema):
    execution_id: UUID

    agent_type: AgentType
    status: AgentStatus

    result: dict[str, Any] = Field(
        default_factory=dict
    )

    reasoning_summary: str | None = None

    evidence: list[AgentEvidence] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    error_message: str | None = None

    external_actions_executed: bool = False


# ============================================================
# AGENT DECISIONS
# ============================================================

class AgentDecision(AgentBaseSchema):
    decision_id: UUID = Field(
        default_factory=uuid4
    )

    task_id: UUID | None = None

    agent_type: AgentType
    decision_type: DecisionType

    recommendation: str = Field(
        min_length=1,
        max_length=5000,
    )

    risk_level: RiskLevel = RiskLevel.UNKNOWN

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    uncertainty_notes: str | None = None

    alternatives: list[str] = Field(
        default_factory=list
    )

    evidence: list[AgentEvidence] = Field(
        default_factory=list
    )

    requires_human_approval: bool = True

    approved: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# HUMAN APPROVAL
# ============================================================

class HumanApprovalRequest(AgentBaseSchema):
    approval_id: UUID = Field(
        default_factory=uuid4
    )

    decision_id: UUID

    requested_by_agent: AgentType

    action_summary: str = Field(
        min_length=1,
        max_length=3000,
    )

    reason: str = Field(
        min_length=1,
        max_length=3000,
    )

    priority: TaskPriority = TaskPriority.MEDIUM

    status: ApprovalStatus = ApprovalStatus.PENDING

    requested_at: datetime = Field(
        default_factory=utc_now
    )

    expires_at: datetime | None = None

    @model_validator(mode="after")
    def validate_expiry(self):
        if (
            self.expires_at is not None
            and self.expires_at <= self.requested_at
        ):
            raise ValueError(
                "Approval expiry must follow request time"
            )
        return self


class HumanApprovalDecision(AgentBaseSchema):
    approval_id: UUID

    status: ApprovalStatus

    comments: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(
        cls,
        value: ApprovalStatus,
    ) -> ApprovalStatus:
        if value not in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
        }:
            raise ValueError(
                "Decision must be approved or rejected"
            )

        return value


class HumanApprovalResponse(HumanApprovalRequest):
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    comments: str | None = None

    authorization_verified: bool = False


# ============================================================
# MULTI-AGENT WORKFLOW
# ============================================================

class AgentWorkflowCreate(AgentBaseSchema):
    workflow_id: UUID = Field(
        default_factory=uuid4
    )

    name: str = Field(
        min_length=1,
        max_length=200,
    )

    description: str | None = None

    supervisor_agent: AgentType = (
        AgentType.SUPERVISOR
    )

    tasks: list[AgentTaskCreate] = Field(
        default_factory=list
    )

    context: dict[str, Any] = Field(
        default_factory=dict
    )

    requires_approval: bool = True

    created_at: datetime = Field(
        default_factory=utc_now
    )


class AgentWorkflowResponse(AgentBaseSchema):
    workflow_id: UUID
    name: str

    status: TaskStatus = TaskStatus.PENDING

    total_tasks: int = Field(ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)

    started_at: datetime | None = None
    completed_at: datetime | None = None

    results: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_counts(self):
        if (
            self.completed_tasks + self.failed_tasks
            > self.total_tasks
        ):
            raise ValueError(
                "Completed and failed tasks exceed total"
            )
        return self


# ============================================================
# AGENT MESSAGES
# ============================================================

class AgentMessage(AgentBaseSchema):
    message_id: UUID = Field(
        default_factory=uuid4
    )

    sender: AgentType
    receiver: AgentType

    task_id: UUID | None = None

    message_type: str = "task"

    content: dict[str, Any] = Field(
        default_factory=dict
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    correlation_id: UUID | None = None


# ============================================================
# AGENT MONITORING
# ============================================================

class AgentMetrics(AgentBaseSchema):
    agent_type: AgentType

    total_executions: int = Field(default=0, ge=0)
    successful_executions: int = Field(default=0, ge=0)
    failed_executions: int = Field(default=0, ge=0)

    average_latency_ms: float = Field(
        default=0,
        ge=0,
    )

    average_confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    tool_success_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    last_updated: datetime = Field(
        default_factory=utc_now
    )


class MultiAgentDashboard(AgentBaseSchema):
    total_agents: int = Field(default=0, ge=0)
    active_agents: int = Field(default=0, ge=0)

    pending_tasks: int = Field(default=0, ge=0)
    running_tasks: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)

    pending_approvals: int = Field(default=0, ge=0)

    agents: list[AgentInfo] = Field(
        default_factory=list
    )

    metrics: list[AgentMetrics] = Field(
        default_factory=list
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# API RESPONSE
# ============================================================

class AgentAPIResponse(AgentBaseSchema):
    success: bool
    message: str

    data: dict[str, Any] | None = None

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "AgentType",
    "AgentStatus",
    "TaskStatus",
    "TaskPriority",
    "DecisionType",
    "ApprovalStatus",
    "RiskLevel",
    "EvidenceType",
    "GeoLocation",
    "AgentInfo",
    "AgentHealth",
    "AgentEvidence",
    "AgentTaskCreate",
    "AgentTaskResponse",
    "AgentExecutionRequest",
    "AgentExecutionResponse",
    "AgentDecision",
    "HumanApprovalRequest",
    "HumanApprovalDecision",
    "HumanApprovalResponse",
    "AgentWorkflowCreate",
    "AgentWorkflowResponse",
    "AgentMessage",
    "AgentMetrics",
    "MultiAgentDashboard",
    "AgentAPIResponse",
]
