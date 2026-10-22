
"""
Kerala Flood Relief Coordination Agent
File: backend/workers/alert_worker.py

Flood alert background worker.

Features:
- Flood-risk alert processing
- Rainfall and river-level threshold monitoring
- District-specific alerts
- Alert severity classification
- Duplicate alert prevention
- Celery background execution
- Redis-based locking and deduplication
- Notification delivery
- Retry handling
- Alert audit logging
- Human approval for public emergency warnings

IMPORTANT:
This worker must not independently issue official
evacuation orders or emergency instructions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from celery import Celery
from celery.exceptions import MaxRetriesExceededError
from pydantic import BaseModel, Field, ValidationError
from redis import Redis
from redis.exceptions import RedisError

from backend.core.config import get_settings


settings = get_settings()

logger = logging.getLogger(__name__)


# =====================================================
# CELERY CONFIGURATION
# =====================================================

celery_app = Celery(
    "kerala_flood_alert_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
)


# =====================================================
# REDIS
# =====================================================

redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


# =====================================================
# ALERT TYPES
# =====================================================

class AlertSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    FLOOD_RISK = "flood_risk"
    HEAVY_RAINFALL = "heavy_rainfall"
    RIVER_LEVEL = "river_level"
    DAM_LEVEL = "dam_level"
    LANDSLIDE_RISK = "landslide_risk"
    RESCUE_EMERGENCY = "rescue_emergency"


class AlertStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISPATCHED = "dispatched"
    FAILED = "failed"


# =====================================================
# ALERT SCHEMA
# =====================================================

class FloodAlert(BaseModel):
    alert_id: str = Field(min_length=1)
    district: str = Field(min_length=1)

    alert_type: AlertType
    severity: AlertSeverity

    title: str = Field(min_length=1)
    message: str = Field(min_length=1)

    source: str = Field(min_length=1)

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    requires_approval: bool = True

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# =====================================================
# DISTRICT CONFIGURATION
# =====================================================

KERALA_DISTRICTS = {
    "thiruvananthapuram",
    "kollam",
    "pathanamthitta",
    "alappuzha",
    "kottayam",
    "idukki",
    "ernakulam",
    "thrissur",
    "palakkad",
    "malappuram",
    "kozhikode",
    "wayanad",
    "kannur",
    "kasaragod",
}


def normalize_district(district: str) -> str:
    value = district.strip().lower()

    if value not in KERALA_DISTRICTS:
        raise ValueError(
            f"Unsupported Kerala district: {district}"
        )

    return value


# =====================================================
# TIME UTILITIES
# =====================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


# =====================================================
# ALERT IDENTIFIERS
# =====================================================

def build_alert_fingerprint(
    alert: FloodAlert,
) -> str:
    """
    Generate a stable fingerprint for duplicate detection.

    The alert_id itself is excluded so duplicate
    content can be detected across separate requests.
    """

    data = {
        "district": alert.district.lower(),
        "type": alert.alert_type.value,
        "severity": alert.severity.value,
        "title": alert.title.strip(),
        "message": alert.message.strip(),
        "source": alert.source.strip(),
    }

    encoded = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# =====================================================
# REDIS KEYS
# =====================================================

def alert_key(alert_id: str) -> str:
    return f"flood:alert:{alert_id}"


def dedup_key(fingerprint: str) -> str:
    return f"flood:alert:dedup:{fingerprint}"


def lock_key(alert_id: str) -> str:
    return f"flood:alert:lock:{alert_id}"


# =====================================================
# ALERT STORAGE
# =====================================================

def save_alert(
    alert: FloodAlert,
    status: AlertStatus,
) -> None:

    payload = alert.model_dump(mode="json")

    payload["status"] = status.value
    payload["updated_at"] = utc_iso()

    redis_client.set(
        alert_key(alert.alert_id),
        json.dumps(payload),
        ex=60 * 60 * 24 * 30,
    )

    logger.info(
        "Alert saved: id=%s district=%s status=%s",
        alert.alert_id,
        alert.district,
        status.value,
    )


def get_alert(alert_id: str) -> dict | None:
    raw = redis_client.get(alert_key(alert_id))

    if not raw:
        return None

    return json.loads(raw)


def update_alert_status(
    alert_id: str,
    status: AlertStatus,
    **fields: Any,
) -> None:

    key = alert_key(alert_id)

    raw = redis_client.get(key)

    if not raw:
        raise ValueError(
            f"Alert not found: {alert_id}"
        )

    payload = json.loads(raw)

    payload["status"] = status.value
    payload["updated_at"] = utc_iso()
    payload.update(fields)

    redis_client.set(
        key,
        json.dumps(payload),
        ex=60 * 60 * 24 * 30,
    )


# =====================================================
# DEDUPLICATION
# =====================================================

def reserve_alert_fingerprint(
    alert: FloodAlert,
    ttl_seconds: int = 900,
) -> bool:

    fingerprint = build_alert_fingerprint(alert)

    return bool(
        redis_client.set(
            dedup_key(fingerprint),
            alert.alert_id,
            nx=True,
            ex=ttl_seconds,
        )
    )


# =====================================================
# ALERT CLASSIFICATION
# =====================================================

def classify_flood_risk(
    risk_score: float,
) -> AlertSeverity:

    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(
            "Risk score must be between 0 and 1"
        )

    if risk_score >= 0.90:
        return AlertSeverity.CRITICAL

    if risk_score >= 0.75:
        return AlertSeverity.HIGH

    if risk_score >= 0.50:
        return AlertSeverity.MODERATE

    return AlertSeverity.LOW


def classify_rainfall(
    rainfall_mm: float,
    thresholds: dict[str, float],
) -> AlertSeverity:
    """
    Thresholds must be supplied by a validated
    source for the correct measurement period.
    """

    if rainfall_mm < 0:
        raise ValueError(
            "Rainfall cannot be negative"
        )

    if rainfall_mm >= thresholds["critical"]:
        return AlertSeverity.CRITICAL

    if rainfall_mm >= thresholds["high"]:
        return AlertSeverity.HIGH

    if rainfall_mm >= thresholds["moderate"]:
        return AlertSeverity.MODERATE

    return AlertSeverity.LOW


def classify_river_level(
    water_level_m: float,
    warning_level_m: float,
    danger_level_m: float,
) -> AlertSeverity:

    if danger_level_m <= warning_level_m:
        raise ValueError(
            "Danger level must exceed warning level"
        )

    if water_level_m >= danger_level_m:
        return AlertSeverity.CRITICAL

    if water_level_m >= warning_level_m:
        return AlertSeverity.HIGH

    return AlertSeverity.LOW


# =====================================================
# ALERT AUDIT
# =====================================================

def write_audit_event(
    alert_id: str,
    event: str,
    details: dict[str, Any] | None = None,
) -> None:

    record = {
        "alert_id": alert_id,
        "event": event,
        "timestamp": utc_iso(),
        "details": details or {},
    }

    redis_client.xadd(
        "flood:alert:audit",
        {"record": json.dumps(record)},
        maxlen=100000,
        approximate=True,
    )

    logger.info(
        "Alert audit: %s %s",
        alert_id,
        event,
    )


# =====================================================
# NOTIFICATION ADAPTER
# =====================================================

def deliver_alert(
    alert: FloodAlert,
) -> dict[str, Any]:
    """
    Integration point for the configured notification
    service (SMS, email, push or dashboard).

    Fail closed until a real delivery provider is
    configured. Never report simulated delivery
    as successful.
    """

    raise RuntimeError(
        "Notification provider not configured. "
        "Connect backend/services/notification_service.py"
    )


# =====================================================
# ALERT PROCESSING TASK
# =====================================================

@celery_app.task(
    bind=True,
    name="alerts.process_flood_alert",
    max_retries=3,
    default_retry_delay=30,
)
def process_flood_alert(
    self,
    alert_data: dict[str, Any],
) -> dict[str, Any]:

    try:
        alert = FloodAlert.model_validate(alert_data)

        alert.district = normalize_district(
            alert.district
        )

        existing = get_alert(alert.alert_id)

        if existing:
            return {
                "alert_id": alert.alert_id,
                "status": existing["status"],
                "duplicate": True,
            }

        if not reserve_alert_fingerprint(alert):
            return {
                "alert_id": alert.alert_id,
                "status": "duplicate",
                "duplicate": True,
            }

        # Public warnings always require approval.
        save_alert(
            alert,
            AlertStatus.PENDING_REVIEW,
        )

        write_audit_event(
            alert.alert_id,
            "alert_created",
            {
                "district": alert.district,
                "severity": alert.severity.value,
                "source": alert.source,
            },
        )

        return {
            "alert_id": alert.alert_id,
            "district": alert.district,
            "severity": alert.severity.value,
            "status": AlertStatus.PENDING_REVIEW.value,
            "requires_approval": True,
        }

    except ValidationError as exc:
        logger.warning(
            "Invalid flood alert: %s",
            exc,
        )
        raise

    except (RedisError, ConnectionError) as exc:
        logger.exception(
            "Temporary alert processing failure"
        )

        raise self.retry(exc=exc)


# =====================================================
# APPROVAL AND DISPATCH
# =====================================================

@celery_app.task(
    bind=True,
    name="alerts.dispatch_approved_alert",
    max_retries=3,
    default_retry_delay=60,
)
def dispatch_approved_alert(
    self,
    alert_id: str,
) -> dict[str, Any]:

    lock = redis_client.lock(
        lock_key(alert_id),
        timeout=120,
        blocking_timeout=5,
    )

    acquired = False

    try:
        acquired = lock.acquire(blocking=True)

        if not acquired:
            raise RuntimeError(
                "Could not acquire alert dispatch lock"
            )

        record = get_alert(alert_id)

        if record is None:
            raise ValueError(
                f"Alert not found: {alert_id}"
            )

        if record["status"] == AlertStatus.DISPATCHED.value:
            return {
                "alert_id": alert_id,
                "status": "already_dispatched",
            }

        if record["status"] != AlertStatus.APPROVED.value:
            raise PermissionError(
                "Only approved alerts can be dispatched"
            )

        alert = FloodAlert.model_validate(record)

        result = deliver_alert(alert)

        update_alert_status(
            alert_id,
            AlertStatus.DISPATCHED,
            delivery=result,
            dispatched_at=utc_iso(),
        )

        write_audit_event(
            alert_id,
            "alert_dispatched",
            result,
        )

        return {
            "alert_id": alert_id,
            "status": AlertStatus.DISPATCHED.value,
            "delivery": result,
        }

    except (RedisError, ConnectionError) as exc:
        logger.exception(
            "Temporary alert dispatch failure"
        )
        raise self.retry(exc=exc)

    except RuntimeError as exc:
        logger.exception(
            "Alert dispatch unavailable"
        )

        try:
            raise self.retry(exc=exc)
        except MaxRetriesExceededError:
            update_alert_status(
                alert_id,
                AlertStatus.FAILED,
                failure_reason="Delivery unavailable",
            )
            raise

    finally:
        if acquired:
            try:
                lock.release()
            except RedisError:
                logger.exception(
                    "Could not release alert lock"
                )


# =====================================================
# FLOOD RISK MONITORING
# =====================================================

@celery_app.task(
    name="alerts.evaluate_flood_risk",
)
def evaluate_flood_risk(
    district: str,
    risk_score: float,
    source: str,
) -> dict[str, Any]:

    district = normalize_district(district)

    severity = classify_flood_risk(risk_score)

    if severity == AlertSeverity.LOW:
        return {
            "district": district,
            "risk_score": risk_score,
            "alert_created": False,
        }

    alert = FloodAlert(
        alert_id=hashlib.sha256(
            f"{district}:{source}:{risk_score}:{utc_iso()}"
            .encode()
        ).hexdigest()[:24],
        district=district,
        alert_type=AlertType.FLOOD_RISK,
        severity=severity,
        title=f"Flood risk assessment: {district.title()}",
        message=(
            f"Model-estimated flood risk score: "
            f"{risk_score:.2f}. "
            "This is an unverified AI assessment "
            "requiring authorized review."
        ),
        source=source,
        confidence=risk_score,
        requires_approval=True,
    )

    return process_flood_alert.apply_async(
        args=[alert.model_dump(mode="json")]
    ).id


# =====================================================
# WORKER HEALTH
# =====================================================

@celery_app.task(
    name="alerts.worker_health",
)
def worker_health() -> dict[str, Any]:

    redis_client.ping()

    return {
        "worker": "alert_worker",
        "status": "healthy",
        "timestamp": utc_iso(),
    }
