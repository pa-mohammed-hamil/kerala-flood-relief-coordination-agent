from __future__ import annotations

"""
backend/services/alert_service.py

Kerala Flood Relief Coordination Agent

Flood alert management service.

Features:
- Flood alert creation and validation
- Rainfall, river and dam risk assessment
- District-level flood alerts
- Severity classification
- Alert approval and publication
- Alert acknowledgment and cancellation
- Alert expiration
- Duplicate alert prevention
- Notification dispatch integration
- Audit logging
- Human oversight for public warnings
- Structured logging

This service does not autonomously publish
official disaster warnings.

All public alerts require authorization and
a verified human approval workflow.
"""


import logging
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class AlertSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"


class AlertStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    PUBLISHED = "published"
    ACKNOWLEDGED = "acknowledged"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class AlertSource(str, Enum):
    RAINFALL = "rainfall"
    RIVER = "river"
    DAM = "dam"
    WEATHER = "weather"
    SATELLITE = "satellite"
    FLOOD_MODEL = "flood_model"
    MANUAL = "manual"
    OFFICIAL = "official"


class AlertChannel(str, Enum):
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    WHATSAPP = "whatsapp"
    DASHBOARD = "dashboard"


# ============================================================
# EXCEPTIONS
# ============================================================

class AlertServiceError(Exception):
    pass


class AlertNotFoundError(AlertServiceError):
    pass


class AlertValidationError(AlertServiceError):
    pass


class AlertConflictError(AlertServiceError):
    pass


class AlertAuthorizationError(AlertServiceError):
    pass


class AlertDeliveryError(AlertServiceError):
    pass


# ============================================================
# DATA MODELS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class FloodAlert:
    id: str
    district_id: int
    title: str
    message: str
    severity: AlertSeverity
    source: AlertSource

    status: AlertStatus = AlertStatus.DRAFT

    risk_score: float | None = None
    confidence: float | None = None

    affected_areas: list[str] = field(
        default_factory=list
    )

    recommended_actions: list[str] = field(
        default_factory=list
    )

    source_reference: str | None = None

    created_by: int | None = None
    approved_by: int | None = None
    published_by: int | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    approved_at: datetime | None = None
    published_at: datetime | None = None
    expires_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class AlertCreateRequest:
    district_id: int
    title: str
    message: str
    severity: AlertSeverity
    source: AlertSource

    risk_score: float | None = None
    confidence: float | None = None

    affected_areas: tuple[str, ...] = ()
    recommended_actions: tuple[str, ...] = ()

    source_reference: str | None = None
    expires_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class AlertAssessment:
    severity: AlertSeverity
    risk_score: float
    factors: dict[str, float]
    requires_review: bool = True


@dataclass(frozen=True)
class AlertDeliveryResult:
    alert_id: str
    channel: AlertChannel
    success: bool
    recipient_count: int = 0
    error: str | None = None


# ============================================================
# REPOSITORY CONTRACT
# ============================================================

class AlertRepository(Protocol):

    async def create(
        self,
        alert: FloodAlert,
    ) -> FloodAlert:
        ...

    async def get_by_id(
        self,
        alert_id: str,
    ) -> FloodAlert | None:
        ...

    async def update(
        self,
        alert: FloodAlert,
    ) -> FloodAlert:
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[FloodAlert]:
        ...

    async def list_active(
        self,
    ) -> Sequence[FloodAlert]:
        ...


# ============================================================
# NOTIFICATION CONTRACT
# ============================================================

class AlertNotificationGateway(Protocol):

    async def send_alert(
        self,
        alert: FloodAlert,
        channel: AlertChannel,
    ) -> AlertDeliveryResult:
        ...


# ============================================================
# AUDIT CONTRACT
# ============================================================

class AlertAuditGateway(Protocol):

    async def record(
        self,
        action: str,
        *,
        actor_id: int | None,
        entity_type: str,
        entity_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...


# ============================================================
# RISK ASSESSMENT
# ============================================================

class FloodRiskAssessment:
    """
    Configurable demonstration risk assessment.

    These are illustrative thresholds, not official
    IMD, CWC, KSDMA or dam operating thresholds.

    Operational thresholds must come from verified,
    location-specific authoritative sources.
    """

    def __init__(
        self,
        *,
        rainfall_reference_mm: float = 200.0,
        river_reference_ratio: float = 1.0,
        dam_reference_ratio: float = 1.0,
    ):
        if min(
            rainfall_reference_mm,
            river_reference_ratio,
            dam_reference_ratio,
        ) <= 0:
            raise ValueError(
                "Risk reference values must be positive"
            )

        self.rainfall_reference_mm = (
            rainfall_reference_mm
        )
        self.river_reference_ratio = (
            river_reference_ratio
        )
        self.dam_reference_ratio = (
            dam_reference_ratio
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def assess(
        self,
        *,
        rainfall_mm: float | None = None,
        river_level_ratio: float | None = None,
        dam_level_ratio: float | None = None,
        model_risk: float | None = None,
    ) -> AlertAssessment:

        factors: dict[str, float] = {}

        if rainfall_mm is not None:
            if rainfall_mm < 0:
                raise AlertValidationError(
                    "Rainfall cannot be negative"
                )

            factors["rainfall"] = self._clamp(
                rainfall_mm
                / self.rainfall_reference_mm
            )

        if river_level_ratio is not None:
            if river_level_ratio < 0:
                raise AlertValidationError(
                    "Invalid river level ratio"
                )

            factors["river"] = self._clamp(
                river_level_ratio
                / self.river_reference_ratio
            )

        if dam_level_ratio is not None:
            if dam_level_ratio < 0:
                raise AlertValidationError(
                    "Invalid dam level ratio"
                )

            factors["dam"] = self._clamp(
                dam_level_ratio
                / self.dam_reference_ratio
            )

        if model_risk is not None:
            if not 0 <= model_risk <= 1:
                raise AlertValidationError(
                    "Model risk must be between 0 and 1"
                )

            factors["model"] = model_risk

        if not factors:
            raise AlertValidationError(
                "At least one risk factor is required"
            )

        weights = {
            "rainfall": 0.25,
            "river": 0.30,
            "dam": 0.20,
            "model": 0.25,
        }

        total_weight = sum(
            weights[name]
            for name in factors
        )

        risk_score = sum(
            factors[name] * weights[name]
            for name in factors
        ) / total_weight

        if risk_score >= 0.90:
            severity = AlertSeverity.EXTREME
        elif risk_score >= 0.75:
            severity = AlertSeverity.SEVERE
        elif risk_score >= 0.55:
            severity = AlertSeverity.HIGH
        elif risk_score >= 0.30:
            severity = AlertSeverity.MODERATE
        else:
            severity = AlertSeverity.LOW

        return AlertAssessment(
            severity=severity,
            risk_score=round(risk_score, 4),
            factors=factors,
            requires_review=True,
        )


# ============================================================
# ALERT SERVICE
# ============================================================

class AlertService:

    ACTIVE_STATUSES = frozenset({
        AlertStatus.DRAFT,
        AlertStatus.PENDING_APPROVAL,
        AlertStatus.APPROVED,
        AlertStatus.PUBLISHED,
        AlertStatus.ACKNOWLEDGED,
    })

    def __init__(
        self,
        repository: AlertRepository,
        *,
        notifications: AlertNotificationGateway | None = None,
        audit: AlertAuditGateway | None = None,
        risk_assessment: FloodRiskAssessment | None = None,
    ):
        self.repository = repository
        self.notifications = notifications
        self.audit = audit
        self.risk_assessment = (
            risk_assessment
            or FloodRiskAssessment()
        )

    async def _audit(
        self,
        action: str,
        alert: FloodAlert,
        actor_id: int | None,
    ) -> None:
        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="flood_alert",
            entity_id=alert.id,
            metadata={
                "district_id": alert.district_id,
                "status": alert.status.value,
                "severity": alert.severity.value,
            },
        )

    async def get_alert(
        self,
        alert_id: str,
    ) -> FloodAlert:
        alert = await self.repository.get_by_id(
            alert_id
        )

        if alert is None:
            raise AlertNotFoundError(
                "Flood alert not found"
            )

        return alert

    async def create_alert(
        self,
        request: AlertCreateRequest,
        *,
        actor_id: int,
    ) -> FloodAlert:

        if request.district_id <= 0:
            raise AlertValidationError(
                "Invalid district identifier"
            )

        if not request.title.strip():
            raise AlertValidationError(
                "Alert title is required"
            )

        if not request.message.strip():
            raise AlertValidationError(
                "Alert message is required"
            )

        if (
            request.risk_score is not None
            and not 0 <= request.risk_score <= 1
        ):
            raise AlertValidationError(
                "Risk score must be between 0 and 1"
            )

        if (
            request.confidence is not None
            and not 0 <= request.confidence <= 1
        ):
            raise AlertValidationError(
                "Confidence must be between 0 and 1"
            )

        if (
            request.expires_at is not None
            and request.expires_at <= utc_now()
        ):
            raise AlertValidationError(
                "Expiration must be in the future"
            )

        existing_alerts = (
            await self.repository.list_by_district(
                request.district_id
            )
        )

        for existing in existing_alerts:
            if (
                existing.status in self.ACTIVE_STATUSES
                and existing.title.strip().casefold()
                == request.title.strip().casefold()
                and existing.source == request.source
            ):
                raise AlertConflictError(
                    "A matching active alert already exists"
                )

        alert = FloodAlert(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            title=request.title.strip(),
            message=request.message.strip(),
            severity=request.severity,
            source=request.source,
            risk_score=request.risk_score,
            confidence=request.confidence,
            affected_areas=list(
                request.affected_areas
            ),
            recommended_actions=list(
                request.recommended_actions
            ),
            source_reference=request.source_reference,
            expires_at=request.expires_at,
            created_by=actor_id,
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            alert
        )

        await self._audit(
            "flood.alert_created",
            created,
            actor_id,
        )

        logger.info(
            "Flood alert created: %s",
            created.id,
        )

        return created

    async def assess_flood_risk(
        self,
        *,
        rainfall_mm: float | None = None,
        river_level_ratio: float | None = None,
        dam_level_ratio: float | None = None,
        model_risk: float | None = None,
    ) -> AlertAssessment:
        return self.risk_assessment.assess(
            rainfall_mm=rainfall_mm,
            river_level_ratio=river_level_ratio,
            dam_level_ratio=dam_level_ratio,
            model_risk=model_risk,
        )

    async def submit_for_approval(
        self,
        alert_id: str,
        *,
        actor_id: int,
    ) -> FloodAlert:
        alert = await self.get_alert(alert_id)

        if alert.status != AlertStatus.DRAFT:
            raise AlertConflictError(
                "Only draft alerts can be submitted"
            )

        alert.status = AlertStatus.PENDING_APPROVAL
        alert.updated_at = utc_now()

        updated = await self.repository.update(
            alert
        )

        await self._audit(
            "flood.alert_submitted",
            updated,
            actor_id,
        )

        return updated

    async def approve_alert(
        self,
        alert_id: str,
        *,
        reviewer_id: int,
        authorized: bool,
    ) -> FloodAlert:
        if not authorized:
            raise AlertAuthorizationError(
                "Alert approval permission required"
            )

        alert = await self.get_alert(alert_id)

        if alert.status != AlertStatus.PENDING_APPROVAL:
            raise AlertConflictError(
                "Alert is not awaiting approval"
            )

        if alert.created_by == reviewer_id:
            raise AlertAuthorizationError(
                "Independent reviewer required"
            )

        alert.status = AlertStatus.APPROVED
        alert.approved_by = reviewer_id
        alert.approved_at = utc_now()
        alert.updated_at = utc_now()

        updated = await self.repository.update(
            alert
        )

        await self._audit(
            "flood.alert_approved",
            updated,
            reviewer_id,
        )

        return updated

    async def publish_alert(
        self,
        alert_id: str,
        *,
        actor_id: int,
        authorized: bool,
        channels: Sequence[AlertChannel] = (
            AlertChannel.DASHBOARD,
        ),
    ) -> tuple[
        FloodAlert,
        list[AlertDeliveryResult],
    ]:
        if not authorized:
            raise AlertAuthorizationError(
                "Alert publication permission required"
            )

        alert = await self.get_alert(alert_id)

        if alert.status != AlertStatus.APPROVED:
            raise AlertConflictError(
                "Only approved alerts can be published"
            )

        if (
            alert.expires_at is not None
            and alert.expires_at <= utc_now()
        ):
            raise AlertConflictError(
                "Expired alerts cannot be published"
            )

        if alert.approved_by is None:
            raise AlertAuthorizationError(
                "Verified human approval required"
            )

        alert.status = AlertStatus.PUBLISHED
        alert.published_by = actor_id
        alert.published_at = utc_now()
        alert.updated_at = utc_now()

        updated = await self.repository.update(
            alert
        )

        await self._audit(
            "flood.alert_published",
            updated,
            actor_id,
        )

        delivery_results: list[
            AlertDeliveryResult
        ] = []

        if self.notifications is not None:
            for channel in dict.fromkeys(channels):
                try:
                    result = (
                        await self.notifications.send_alert(
                            updated,
                            channel,
                        )
                    )

                except Exception:
                    logger.exception(
                        "Alert delivery failed for %s",
                        channel.value,
                    )

                    result = AlertDeliveryResult(
                        alert_id=updated.id,
                        channel=channel,
                        success=False,
                        error="Delivery failed",
                    )

                delivery_results.append(result)

        return updated, delivery_results

    async def acknowledge_alert(
        self,
        alert_id: str,
        *,
        actor_id: int,
    ) -> FloodAlert:
        alert = await self.get_alert(alert_id)

        if alert.status != AlertStatus.PUBLISHED:
            raise AlertConflictError(
                "Only published alerts can be acknowledged"
            )

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.updated_at = utc_now()

        updated = await self.repository.update(
            alert
        )

        await self._audit(
            "flood.alert_acknowledged",
            updated,
            actor_id,
        )

        return updated

    async def cancel_alert(
        self,
        alert_id: str,
        *,
        actor_id: int,
        authorized: bool,
    ) -> FloodAlert:
        if not authorized:
            raise AlertAuthorizationError(
                "Alert cancellation permission required"
            )

        alert = await self.get_alert(alert_id)

        if alert.status in {
            AlertStatus.CANCELLED,
            AlertStatus.EXPIRED,
        }:
            raise AlertConflictError(
                "Alert is already inactive"
            )

        alert.status = AlertStatus.CANCELLED
        alert.updated_at = utc_now()

        updated = await self.repository.update(
            alert
        )

        await self._audit(
            "flood.alert_cancelled",
            updated,
            actor_id,
        )

        return updated

    async def expire_alerts(
        self,
    ) -> int:
        active_alerts = (
            await self.repository.list_active()
        )

        now = utc_now()
        expired_count = 0

        for alert in active_alerts:
            if (
                alert.expires_at is not None
                and alert.expires_at <= now
            ):
                alert.status = AlertStatus.EXPIRED
                alert.updated_at = now

                await self.repository.update(
                    alert
                )

                await self._audit(
                    "flood.alert_expired",
                    alert,
                    None,
                )

                expired_count += 1

        return expired_count

    async def list_district_alerts(
        self,
        district_id: int,
    ) -> Sequence[FloodAlert]:
        return await self.repository.list_by_district(
            district_id
        )

    async def list_active_alerts(
        self,
    ) -> Sequence[FloodAlert]:
        return await self.repository.list_active()

    async def get_alert_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:
        alerts = await self.list_district_alerts(
            district_id
        )

        return {
            "district_id": district_id,
            "total_alerts": len(alerts),
            "published": sum(
                alert.status == AlertStatus.PUBLISHED
                for alert in alerts
            ),
            "pending_approval": sum(
                alert.status == AlertStatus.PENDING_APPROVAL
                for alert in alerts
            ),
            "high_priority": sum(
                alert.severity in {
                    AlertSeverity.HIGH,
                    AlertSeverity.SEVERE,
                    AlertSeverity.EXTREME,
                }
                for alert in alerts
            ),
            "cancelled": sum(
                alert.status == AlertStatus.CANCELLED
                for alert in alerts
            ),
            "expired": sum(
                alert.status == AlertStatus.EXPIRED
                for alert in alerts
            ),
        }
