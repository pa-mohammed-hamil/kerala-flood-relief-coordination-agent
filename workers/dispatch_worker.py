from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/workers/dispatch_worker.py

Rescue Dispatch Worker

Features:
- SOS rescue request processing
- District-based rescue-team matching
- Emergency priority scoring
- Rescue-team availability checking
- Team capability matching
- Assignment reservations
- Human approval before dispatch
- Redis distributed locking
- Celery background tasks
- Dispatch audit logging
- Retry handling
- Worker health monitoring

Safety:
AI recommendations are advisory only.
No rescue team is dispatched automatically.
An authorized human must approve assignments.
"""


import json
import logging
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from celery import Celery
from pydantic import BaseModel, Field, field_validator
from redis import Redis
from redis.exceptions import RedisError

from backend.core.config import get_settings


settings = get_settings()

logger = logging.getLogger(__name__)


# =====================================================
# CELERY CONFIGURATION
# =====================================================

celery_app = Celery(
    "kerala_flood_dispatch_worker",
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
# REDIS CONNECTION
# =====================================================

redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


# =====================================================
# ENUMS
# =====================================================

class RescuePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RescueStatus(str, Enum):
    PENDING = "pending"
    TRIAGED = "triaged"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TeamStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    DEPLOYED = "deployed"
    OFFLINE = "offline"


class RescueCapability(str, Enum):
    BOAT = "boat"
    SWIMMING = "swimming"
    MEDICAL = "medical"
    EVACUATION = "evacuation"
    SEARCH_RESCUE = "search_rescue"
    HIGH_WATER = "high_water"


# =====================================================
# KERALA DISTRICTS
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


def normalize_district(value: str) -> str:
    district = value.strip().lower()

    if district not in KERALA_DISTRICTS:
        raise ValueError(
            f"Unsupported Kerala district: {value}"
        )

    return district


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


# =====================================================
# REQUEST SCHEMAS
# =====================================================

class GeoLocation(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class SOSRequest(BaseModel):
    request_id: str = Field(min_length=1)
    district: str
    location: GeoLocation

    people_count: int = Field(default=1, ge=1)
    children_count: int = Field(default=0, ge=0)
    elderly_count: int = Field(default=0, ge=0)
    injured_count: int = Field(default=0, ge=0)

    water_level_m: float = Field(default=0, ge=0)

    trapped: bool = False
    medical_emergency: bool = False
    immediate_danger: bool = False

    required_capabilities: set[RescueCapability] = Field(
        default_factory=set
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class RescueTeam(BaseModel):
    team_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    district: str
    location: GeoLocation

    status: TeamStatus = TeamStatus.AVAILABLE

    capabilities: set[RescueCapability] = Field(
        default_factory=set
    )

    capacity: int = Field(default=1, ge=1)

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class DispatchRecommendation(BaseModel):
    recommendation_id: str
    request_id: str
    team_id: str

    priority: RescuePriority
    priority_score: float

    reasons: list[str]

    status: RescueStatus = (
        RescueStatus.AWAITING_APPROVAL
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )


# =====================================================
# REDIS KEYS
# =====================================================

def sos_key(request_id: str) -> str:
    return f"rescue:sos:{request_id}"


def team_key(team_id: str) -> str:
    return f"rescue:team:{team_id}"


def recommendation_key(
    recommendation_id: str,
) -> str:
    return f"rescue:recommendation:{recommendation_id}"


def request_recommendation_key(
    request_id: str,
) -> str:
    return f"rescue:request:recommendation:{request_id}"


def team_lock_key(team_id: str) -> str:
    return f"rescue:team:lock:{team_id}"


# =====================================================
# STORAGE HELPERS
# =====================================================

def save_json(
    key: str,
    payload: dict[str, Any],
) -> None:
    redis_client.set(
        key,
        json.dumps(payload, default=str),
    )


def load_json(
    key: str,
) -> dict[str, Any] | None:
    value = redis_client.get(key)

    if not value:
        return None

    return json.loads(value)


def save_sos_request(
    request: SOSRequest,
) -> None:
    save_json(
        sos_key(request.request_id),
        request.model_dump(mode="json"),
    )


def save_rescue_team(
    team: RescueTeam,
) -> None:
    save_json(
        team_key(team.team_id),
        team.model_dump(mode="json"),
    )


def save_recommendation(
    recommendation: DispatchRecommendation,
) -> None:
    save_json(
        recommendation_key(
            recommendation.recommendation_id
        ),
        recommendation.model_dump(mode="json"),
    )


# =====================================================
# AUDIT LOGGING
# =====================================================

def audit_event(
    event: str,
    request_id: str,
    details: dict[str, Any] | None = None,
) -> None:

    payload = {
        "event": event,
        "request_id": request_id,
        "details": details or {},
        "timestamp": utc_iso(),
    }

    redis_client.xadd(
        "rescue:dispatch:audit",
        {"record": json.dumps(payload)},
        maxlen=100000,
        approximate=True,
    )

    logger.info(
        "Dispatch audit: event=%s request=%s",
        event,
        request_id,
    )


# =====================================================
# PRIORITY SCORING
# =====================================================

def calculate_priority(
    request: SOSRequest,
) -> tuple[RescuePriority, float, list[str]]:
    """
    Heuristic for operator triage support only.

    Scores must not be used to automatically deny,
    delay or deprioritize real emergency requests.
    """

    score = 0.0
    reasons: list[str] = []

    if request.immediate_danger:
        score += 40
        reasons.append("Immediate danger reported")

    if request.trapped:
        score += 25
        reasons.append("People reportedly trapped")

    if request.medical_emergency:
        score += 30
        reasons.append("Medical emergency reported")

    if request.injured_count > 0:
        score += min(request.injured_count * 8, 24)
        reasons.append("Injuries reported")

    if request.children_count > 0:
        score += 10
        reasons.append("Children present")

    if request.elderly_count > 0:
        score += 10
        reasons.append("Elderly people present")

    if request.water_level_m >= 1.5:
        score += 20
        reasons.append("High reported water level")

    if request.people_count >= 10:
        score += 10
        reasons.append("Multiple people need assistance")

    score = min(score, 100.0)

    if score >= 80:
        priority = RescuePriority.CRITICAL
    elif score >= 55:
        priority = RescuePriority.HIGH
    elif score >= 25:
        priority = RescuePriority.MEDIUM
    else:
        priority = RescuePriority.LOW

    return priority, score, reasons


# =====================================================
# TEAM MATCHING
# =====================================================

def match_rescue_teams(
    request: SOSRequest,
    teams: list[RescueTeam],
) -> list[RescueTeam]:
    """
    Candidate filtering only.

    No straight-line distance or ETA is assumed
    to represent a safe, passable flood route.
    """

    matches: list[RescueTeam] = []

    for team in teams:
        if team.status != TeamStatus.AVAILABLE:
            continue

        if team.district != request.district:
            continue

        if team.capacity < request.people_count:
            continue

        if not request.required_capabilities.issubset(
            team.capabilities
        ):
            continue

        matches.append(team)

    return sorted(
        matches,
        key=lambda team: (
            -team.capacity,
            team.team_id,
        ),
    )


def get_available_teams(
    district: str,
) -> list[RescueTeam]:
    """
    Demo Redis-backed team registry.

    Production should use a database query/index
    rather than scanning Redis keys.
    """

    teams: list[RescueTeam] = []

    for key in redis_client.scan_iter(
        match="rescue:team:*"
    ):
        data = load_json(key)

        if not data:
            continue

        try:
            team = RescueTeam.model_validate(data)
        except Exception:
            logger.exception(
                "Invalid rescue team record: %s",
                key,
            )
            continue

        if (
            team.district == district
            and team.status == TeamStatus.AVAILABLE
        ):
            teams.append(team)

    return teams


# =====================================================
# GENERATE DISPATCH RECOMMENDATION
# =====================================================

@celery_app.task(
    bind=True,
    name="dispatch.recommend_rescue_team",
    max_retries=3,
    default_retry_delay=15,
)
def recommend_rescue_team(
    self,
    request_data: dict[str, Any],
) -> dict[str, Any]:

    try:
        request = SOSRequest.model_validate(
            request_data
        )

        save_sos_request(request)

        existing_id = redis_client.get(
            request_recommendation_key(
                request.request_id
            )
        )

        if existing_id:
            existing = load_json(
                recommendation_key(existing_id)
            )

            if existing:
                return existing

        priority, score, reasons = (
            calculate_priority(request)
        )

        teams = get_available_teams(
            request.district
        )

        matches = match_rescue_teams(
            request,
            teams,
        )

        if not matches:
            audit_event(
                "no_team_available",
                request.request_id,
                {
                    "district": request.district,
                    "priority": priority.value,
                },
            )

            return {
                "request_id": request.request_id,
                "status": "manual_coordination_required",
                "priority": priority.value,
                "priority_score": score,
                "message": (
                    "No matching team found. "
                    "Escalate to the district control room."
                ),
            }

        selected_team = matches[0]

        recommendation = DispatchRecommendation(
            recommendation_id=str(uuid.uuid4()),
            request_id=request.request_id,
            team_id=selected_team.team_id,
            priority=priority,
            priority_score=score,
            reasons=reasons,
        )

        save_recommendation(recommendation)

        redis_client.set(
            request_recommendation_key(
                request.request_id
            ),
            recommendation.recommendation_id,
        )

        audit_event(
            "recommendation_created",
            request.request_id,
            {
                "team_id": selected_team.team_id,
                "priority": priority.value,
                "score": score,
            },
        )

        return recommendation.model_dump(
            mode="json"
        )

    except RedisError as exc:
        logger.exception(
            "Redis unavailable during recommendation"
        )
        raise self.retry(exc=exc)


# =====================================================
# HUMAN APPROVAL
# =====================================================

def approve_recommendation(
    recommendation_id: str,
    approver_id: str,
) -> dict[str, Any]:
    """
    Must only be invoked by an authenticated,
    authorized dispatcher through the API.

    This function does not authenticate the caller.
    """

    if not approver_id.strip():
        raise ValueError(
            "Approver identifier is required"
        )

    key = recommendation_key(recommendation_id)

    record = load_json(key)

    if not record:
        raise ValueError(
            "Recommendation not found"
        )

    if record["status"] != (
        RescueStatus.AWAITING_APPROVAL.value
    ):
        raise ValueError(
            "Recommendation is not awaiting approval"
        )

    record["status"] = RescueStatus.APPROVED.value
    record["approved_by"] = approver_id
    record["approved_at"] = utc_iso()

    save_json(key, record)

    audit_event(
        "recommendation_approved",
        record["request_id"],
        {
            "recommendation_id": recommendation_id,
            "approver_id": approver_id,
        },
    )

    return record


# =====================================================
# RESCUE DISPATCH TASK
# =====================================================

@celery_app.task(
    bind=True,
    name="dispatch.execute_approved_dispatch",
    max_retries=3,
    default_retry_delay=30,
)
def execute_approved_dispatch(
    self,
    recommendation_id: str,
) -> dict[str, Any]:
    """
    Reserve the approved team for operator dispatch.

    This does not transmit a real dispatch order.
    A separate verified integration is required.
    """

    record = load_json(
        recommendation_key(recommendation_id)
    )

    if not record:
        raise ValueError(
            "Recommendation not found"
        )

    if record["status"] != RescueStatus.APPROVED.value:
        raise PermissionError(
            "Human approval is required"
        )

    team_id = record["team_id"]

    lock = redis_client.lock(
        team_lock_key(team_id),
        timeout=60,
        blocking_timeout=5,
    )

    acquired = False

    try:
        acquired = lock.acquire(blocking=True)

        if not acquired:
            raise RuntimeError(
                "Team reservation lock unavailable"
            )

        # Recheck recommendation under lock.
        record = load_json(
            recommendation_key(recommendation_id)
        )

        if not record or record["status"] != (
            RescueStatus.APPROVED.value
        ):
            raise PermissionError(
                "Recommendation no longer approved"
            )

        team_data = load_json(team_key(team_id))

        if not team_data:
            raise ValueError(
                "Rescue team not found"
            )

        team = RescueTeam.model_validate(
            team_data
        )

        if team.status != TeamStatus.AVAILABLE:
            return {
                "recommendation_id": recommendation_id,
                "status": "team_unavailable",
                "team_id": team_id,
            }

        team.status = TeamStatus.RESERVED

        save_rescue_team(team)

        record["status"] = "team_reserved"
        record["reserved_at"] = utc_iso()

        save_json(
            recommendation_key(recommendation_id),
            record,
        )

        audit_event(
            "team_reserved",
            record["request_id"],
            {
                "team_id": team_id,
                "recommendation_id": recommendation_id,
            },
        )

        return {
            "recommendation_id": recommendation_id,
            "request_id": record["request_id"],
            "team_id": team_id,
            "status": "team_reserved",
            "requires_dispatch_confirmation": True,
        }

    except RedisError as exc:
        logger.exception(
            "Redis failure during team reservation"
        )
        raise self.retry(exc=exc)

    finally:
        if acquired:
            try:
                lock.release()
            except RedisError:
                logger.exception(
                    "Failed to release team lock"
                )


# =====================================================
# CONFIRM OPERATIONAL DISPATCH
# =====================================================

def confirm_dispatch(
    recommendation_id: str,
    dispatcher_id: str,
    dispatch_reference: str,
) -> dict[str, Any]:
    """
    Record verified dispatch confirmation.

    Call only after the authorized operator
    confirms the dispatch through an actual
    operational channel.
    """

    if not dispatcher_id.strip():
        raise ValueError(
            "Dispatcher identifier required"
        )

    if not dispatch_reference.strip():
        raise ValueError(
            "Dispatch reference required"
        )

    key = recommendation_key(recommendation_id)
    record = load_json(key)

    if not record:
        raise ValueError(
            "Recommendation not found"
        )

    if record["status"] != "team_reserved":
        raise ValueError(
            "Team must be reserved before confirmation"
        )

    team_id = record["team_id"]

    record["status"] = RescueStatus.DISPATCHED.value
    record["dispatched_at"] = utc_iso()
    record["dispatched_by"] = dispatcher_id
    record["dispatch_reference"] = dispatch_reference

    save_json(key, record)

    team_data = load_json(team_key(team_id))

    if team_data:
        team = RescueTeam.model_validate(team_data)
        team.status = TeamStatus.DEPLOYED
        save_rescue_team(team)

    audit_event(
        "dispatch_confirmed",
        record["request_id"],
        {
            "team_id": team_id,
            "dispatcher_id": dispatcher_id,
            "dispatch_reference": dispatch_reference,
        },
    )

    return record


# =====================================================
# WORKER HEALTH
# =====================================================

@celery_app.task(
    name="dispatch.worker_health",
)
def worker_health() -> dict[str, Any]:

    redis_client.ping()

    return {
        "worker": "dispatch_worker",
        "status": "healthy",
        "timestamp": utc_iso(),
    }
