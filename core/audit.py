from __future__ import annotations

"""
backend/core/audit.py

Audit logging for the Kerala Flood Relief Coordination Agent.

Features:
- Structured audit events
- User and administrator actions
- SOS and rescue activity
- Volunteer assignment tracking
- AI agent decisions
- Human approval tracking
- Security events
- Async SQLAlchemy persistence
- Sensitive-data redaction
- Correlation and request IDs
"""


import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.audit_log import AuditLog


logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"

    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"

    SOS_CREATED = "sos.created"
    SOS_UPDATED = "sos.updated"
    SOS_CANCELLED = "sos.cancelled"
    SOS_ESCALATED = "sos.escalated"

    INCIDENT_CREATED = "incident.created"
    INCIDENT_UPDATED = "incident.updated"

    RESCUE_ASSIGNED = "rescue.assigned"
    RESCUE_DISPATCHED = "rescue.dispatched"
    RESCUE_COMPLETED = "rescue.completed"

    VOLUNTEER_REGISTERED = "volunteer.registered"
    VOLUNTEER_ASSIGNED = "volunteer.assigned"
    VOLUNTEER_RELEASED = "volunteer.released"

    SHELTER_CREATED = "shelter.created"
    SHELTER_ALLOCATED = "shelter.allocated"

    RELIEF_DISTRIBUTED = "relief.distributed"
    INVENTORY_UPDATED = "inventory.updated"

    MEDICAL_REQUEST = "medical.request"
    AMBULANCE_DISPATCH = "medical.ambulance_dispatch"

    FLOOD_ALERT_CREATED = "flood.alert_created"
    FLOOD_ALERT_PUBLISHED = "flood.alert_published"

    AGENT_DECISION = "agent.decision"
    AGENT_EXECUTION = "agent.execution"
    AGENT_FAILED = "agent.failed"

    HUMAN_APPROVED = "approval.approved"
    HUMAN_REJECTED = "approval.rejected"

    PERMISSION_DENIED = "security.permission_denied"
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"

    DATA_EXPORTED = "data.exported"
    DATA_DELETED = "data.deleted"


class AuditSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


SENSITIVE_KEYS = {
    "password",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "api_key",
    "secret",
    "private_key",
    "phone",
    "phone_number",
    "email",
    "address",
    "latitude",
    "longitude",
    "gps",
    "medical_history",
    "patient_name",
    "aadhaar",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def redact_sensitive(
    value: Any,
    depth: int = 0,
) -> Any:
    """Remove sensitive fields from audit metadata."""

    if depth > 10:
        return "[MAX_DEPTH]"

    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in SENSITIVE_KEYS
                else redact_sensitive(item, depth + 1)
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            redact_sensitive(item, depth + 1)
            for item in value[:100]
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    if value is None:
        return None

    return str(type(value).__name__)


class AuditService:
    """Writes structured audit events to the database."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def record(
        self,
        action: AuditAction | str,
        *,
        actor_id: int | None = None,
        entity_type: str | None = None,
        entity_id: str | int | None = None,
        description: str | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        ip_address: str | None = None,
        success: bool = True,
    ) -> AuditLog:
        """
        Stage an audit event in the active transaction.

        The caller controls commit and rollback.
        """

        action_value = (
            action.value
            if isinstance(action, AuditAction)
            else str(action)
        )

        audit = AuditLog(
            actor_id=actor_id,
            action=action_value,
            entity_type=entity_type,
            entity_id=(
                str(entity_id)
                if entity_id is not None
                else None
            ),
            description=description,
            severity=severity.value,
            metadata_json=redact_sensitive(metadata or {}),
            request_id=request_id or str(uuid.uuid4()),
            ip_address=ip_address,
            success=success,
            created_at=utc_now(),
        )

        self.db.add(audit)
        await self.db.flush()
        return audit

    async def record_sos(
        self,
        sos_id: int,
        actor_id: int | None = None,
        district_id: int | None = None,
    ) -> AuditLog:
        return await self.record(
            AuditAction.SOS_CREATED,
            actor_id=actor_id,
            entity_type="sos",
            entity_id=sos_id,
            description="Emergency SOS request created",
            severity=AuditSeverity.CRITICAL,
            metadata={"district_id": district_id},
        )

    async def record_rescue_dispatch(
        self,
        mission_id: int,
        team_id: int,
        actor_id: int | None = None,
    ) -> AuditLog:
        return await self.record(
            AuditAction.RESCUE_DISPATCHED,
            actor_id=actor_id,
            entity_type="rescue_mission",
            entity_id=mission_id,
            description="Rescue team dispatched",
            severity=AuditSeverity.CRITICAL,
            metadata={"team_id": team_id},
        )

    async def record_volunteer_assignment(
        self,
        volunteer_id: int,
        mission_id: int,
        actor_id: int | None = None,
    ) -> AuditLog:
        return await self.record(
            AuditAction.VOLUNTEER_ASSIGNED,
            actor_id=actor_id,
            entity_type="volunteer",
            entity_id=volunteer_id,
            description="Volunteer assigned to rescue mission",
            metadata={"mission_id": mission_id},
        )

    async def record_agent_decision(
        self,
        agent_name: str,
        decision_id: str,
        *,
        confidence: float | None = None,
        human_review_required: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

        return await self.record(
            AuditAction.AGENT_DECISION,
            entity_type="agent_decision",
            entity_id=decision_id,
            description=f"Decision recorded for agent {agent_name}",
            metadata={
                "agent_name": agent_name,
                "confidence": confidence,
                "human_review_required": human_review_required,
                **(metadata or {}),
            },
        )

    async def record_human_approval(
        self,
        decision_id: str,
        reviewer_id: int,
        approved: bool,
        reason: str | None = None,
    ) -> AuditLog:
        return await self.record(
            (
                AuditAction.HUMAN_APPROVED
                if approved
                else AuditAction.HUMAN_REJECTED
            ),
            actor_id=reviewer_id,
            entity_type="agent_decision",
            entity_id=decision_id,
            description="Human review completed",
            metadata={
                "approved": approved,
                "reason": reason,
            },
        )

    async def record_security_event(
        self,
        action: AuditAction,
        *,
        actor_id: int | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        return await self.record(
            action,
            actor_id=actor_id,
            entity_type="security",
            description=description,
            severity=AuditSeverity.WARNING,
            metadata=metadata,
            success=False,
        )


async def audit_event(
    db: AsyncSession,
    action: AuditAction | str,
    **kwargs: Any,
) -> AuditLog:
    """Convenience function for recording an audit event."""
    service = AuditService(db)
    return await service.record(action, **kwargs)
