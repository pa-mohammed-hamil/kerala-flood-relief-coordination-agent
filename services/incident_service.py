from __future__ import annotations

"""
backend/services/incident_service.py

Kerala Flood Relief Coordination Agent

Incident Management Service

Features:
- Flood and disaster incident registration
- Incident classification and prioritization
- Incident location and district tracking
- Duplicate incident detection
- Incident verification and review
- Human-authorized incident escalation
- Incident assignment and reassignment
- Incident lifecycle management
- SOS and rescue mission references
- Casualty and affected-population tracking
- Incident evidence management
- Incident timeline and audit logging
- District-level incident statistics
- Incident report generation
- Optimistic concurrency control

SAFETY:
This service is a coordination component, not an
operational emergency-response system.

Creating an incident does not contact emergency
services. Escalation does not automatically
dispatch rescue teams or issue public alerts.

Emergency actions require authorized personnel,
verified information, and operational integrations.
"""


import logging
import math
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class IncidentType(str, Enum):
    FLOOD = "flood"
    FLASH_FLOOD = "flash_flood"
    LANDSLIDE = "landslide"
    WATERLOGGING = "waterlogging"
    RIVER_OVERFLOW = "river_overflow"
    DAM_RELATED = "dam_related"
    ROAD_BLOCKAGE = "road_blockage"
    BRIDGE_DAMAGE = "bridge_damage"
    BUILDING_DAMAGE = "building_damage"
    POWER_FAILURE = "power_failure"
    WATER_CONTAMINATION = "water_contamination"
    MEDICAL_EMERGENCY = "medical_emergency"
    MISSING_PERSON = "missing_person"
    EVACUATION_REQUIRED = "evacuation_required"
    RESCUE_REQUIRED = "rescue_required"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    OTHER = "other"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    REPORTED = "reported"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class IncidentSource(str, Enum):
    CITIZEN = "citizen"
    VOLUNTEER = "volunteer"
    FIELD_OFFICER = "field_officer"
    RESCUE_TEAM = "rescue_team"
    SENSOR = "sensor"
    SATELLITE = "satellite"
    DRONE = "drone"
    AI_DETECTION = "ai_detection"
    OFFICIAL = "official"
    SOS = "sos"


class IncidentPriority(str, Enum):
    P1_CRITICAL = "p1_critical"
    P2_HIGH = "p2_high"
    P3_MODERATE = "p3_moderate"
    P4_LOW = "p4_low"


# ============================================================
# EXCEPTIONS
# ============================================================

class IncidentServiceError(Exception):
    pass


class IncidentValidationError(IncidentServiceError):
    pass


class IncidentNotFoundError(IncidentServiceError):
    pass


class IncidentConflictError(IncidentServiceError):
    pass


class IncidentAuthorizationError(IncidentServiceError):
    pass


class IncidentDuplicateError(IncidentServiceError):
    pass


class IncidentIntegrationError(IncidentServiceError):
    pass


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_datetime(
    value: datetime,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise IncidentValidationError(
            f"{field_name} must be timezone-aware"
        )


def validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:
    if not math.isfinite(latitude):
        raise IncidentValidationError(
            "Latitude must be finite"
        )

    if not math.isfinite(longitude):
        raise IncidentValidationError(
            "Longitude must be finite"
        )

    if not -90 <= latitude <= 90:
        raise IncidentValidationError(
            "Latitude outside valid range"
        )

    if not -180 <= longitude <= 180:
        raise IncidentValidationError(
            "Longitude outside valid range"
        )


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    radius_km = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return (
        2
        * radius_km
        * math.asin(math.sqrt(min(1.0, max(0.0, a))))
    )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class IncidentLocation:
    latitude: float
    longitude: float

    address: str = ""
    landmark: str = ""
    panchayat: str = ""
    village: str = ""

    def __post_init__(self) -> None:
        validate_coordinates(
            self.latitude,
            self.longitude,
        )


@dataclass(frozen=True)
class IncidentImpact:
    affected_people: int = 0
    people_at_risk: int = 0
    injured_people: int = 0
    missing_people: int = 0
    confirmed_deaths: int = 0
    displaced_people: int = 0
    affected_households: int = 0

    infrastructure_affected: bool = False
    critical_services_disrupted: bool = False

    def __post_init__(self) -> None:
        counts = (
            self.affected_people,
            self.people_at_risk,
            self.injured_people,
            self.missing_people,
            self.confirmed_deaths,
            self.displaced_people,
            self.affected_households,
        )

        if any(value < 0 for value in counts):
            raise IncidentValidationError(
                "Impact counts cannot be negative"
            )


@dataclass(frozen=True)
class IncidentCreateRequest:
    district_id: int
    incident_type: IncidentType
    source: IncidentSource

    title: str
    description: str
    location: IncidentLocation

    impact: IncidentImpact = field(
        default_factory=IncidentImpact
    )

    reported_severity: IncidentSeverity | None = None

    sos_request_id: str | None = None
    external_reference: str | None = None

    evidence_references: tuple[str, ...] = ()

    occurred_at: datetime | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class IncidentTimelineEvent:
    id: str
    incident_id: str

    action: str
    description: str

    actor_id: int | None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Incident:
    id: str
    district_id: int

    incident_type: IncidentType
    source: IncidentSource

    title: str
    description: str

    location: IncidentLocation
    impact: IncidentImpact

    severity: IncidentSeverity
    priority: IncidentPriority

    status: IncidentStatus = IncidentStatus.REPORTED

    reported_by: int | None = None
    verified_by: int | None = None

    assigned_team_id: str | None = None
    assigned_by: int | None = None

    sos_request_id: str | None = None
    external_reference: str | None = None

    evidence_references: list[str] = field(
        default_factory=list
    )

    rescue_mission_ids: list[str] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    occurred_at: datetime | None = None
    verified_at: datetime | None = None
    assigned_at: datetime | None = None
    resolved_at: datetime | None = None
    closed_at: datetime | None = None

    review_notes: str | None = None
    resolution_notes: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class IncidentAssessment:
    severity: IncidentSeverity
    priority: IncidentPriority
    score: float
    reasons: tuple[str, ...]
    requires_human_review: bool = True


@dataclass(frozen=True)
class DuplicateIncidentMatch:
    incident_id: str
    distance_km: float
    time_difference_minutes: float
    similarity_score: float


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class IncidentRepository(Protocol):

    async def create(
        self,
        incident: Incident,
    ) -> Incident:
        ...

    async def get_by_id(
        self,
        incident_id: str,
    ) -> Incident | None:
        ...

    async def update(
        self,
        incident: Incident,
        *,
        expected_version: int,
    ) -> Incident:
        """
        Must enforce optimistic concurrency.

        Raise IncidentConflictError when the
        stored version differs from expected_version.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[Incident]:
        ...

    async def list_by_status(
        self,
        status: IncidentStatus,
    ) -> Sequence[Incident]:
        ...

    async def find_nearby(
        self,
        *,
        district_id: int,
        latitude: float,
        longitude: float,
        radius_km: float,
        since: datetime,
    ) -> Sequence[Incident]:
        ...

    async def find_by_external_reference(
        self,
        *,
        source: IncidentSource,
        external_reference: str,
    ) -> Incident | None:
        ...


class IncidentTimelineRepository(Protocol):

    async def append(
        self,
        event: IncidentTimelineEvent,
    ) -> None:
        ...

    async def list_by_incident(
        self,
        incident_id: str,
    ) -> Sequence[IncidentTimelineEvent]:
        ...


class IncidentAuthorizationGateway(Protocol):

    async def can_create(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_review(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_assign(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_manage(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class IncidentAuditGateway(Protocol):

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


class IncidentNotificationGateway(Protocol):

    async def notify_escalation(
        self,
        incident: Incident,
    ) -> bool:
        ...

    async def notify_assignment(
        self,
        incident: Incident,
    ) -> bool:
        ...


# ============================================================
# INCIDENT PRIORITY ENGINE
# ============================================================

class IncidentPriorityEngine:
    """
    Advisory incident prioritization.

    The scoring rules below are illustrative
    and must be calibrated against operational
    emergency-management procedures.
    """

    SEVERITY_BASE_SCORE = {
        IncidentSeverity.LOW: 0.10,
        IncidentSeverity.MODERATE: 0.30,
        IncidentSeverity.HIGH: 0.55,
        IncidentSeverity.SEVERE: 0.75,
        IncidentSeverity.CRITICAL: 0.95,
    }

    HIGH_RISK_TYPES = {
        IncidentType.FLASH_FLOOD,
        IncidentType.LANDSLIDE,
        IncidentType.MEDICAL_EMERGENCY,
        IncidentType.RESCUE_REQUIRED,
        IncidentType.EVACUATION_REQUIRED,
        IncidentType.BRIDGE_DAMAGE,
    }

    def assess(
        self,
        *,
        incident_type: IncidentType,
        impact: IncidentImpact,
        reported_severity: IncidentSeverity | None,
    ) -> IncidentAssessment:

        reasons: list[str] = []

        score = (
            self.SEVERITY_BASE_SCORE[
                reported_severity
            ]
            if reported_severity is not None
            else 0.15
        )

        if incident_type in self.HIGH_RISK_TYPES:
            score += 0.15
            reasons.append(
                "Potentially life-threatening incident category"
            )

        if impact.people_at_risk > 0:
            score += min(
                0.20,
                impact.people_at_risk / 1000,
            )
            reasons.append(
                "People reported at risk"
            )

        if impact.injured_people > 0:
            score += 0.20
            reasons.append(
                "Injuries reported"
            )

        if impact.missing_people > 0:
            score += 0.20
            reasons.append(
                "Missing people reported"
            )

        if impact.confirmed_deaths > 0:
            score += 0.25
            reasons.append(
                "Fatalities reported"
            )

        if impact.critical_services_disrupted:
            score += 0.15
            reasons.append(
                "Critical services disrupted"
            )

        if impact.displaced_people > 0:
            score += min(
                0.10,
                impact.displaced_people / 2000,
            )
            reasons.append(
                "Population displacement reported"
            )

        score = min(1.0, score)

        if score >= 0.85:
            severity = IncidentSeverity.CRITICAL
            priority = IncidentPriority.P1_CRITICAL

        elif score >= 0.65:
            severity = IncidentSeverity.SEVERE
            priority = IncidentPriority.P2_HIGH

        elif score >= 0.45:
            severity = IncidentSeverity.HIGH
            priority = IncidentPriority.P2_HIGH

        elif score >= 0.25:
            severity = IncidentSeverity.MODERATE
            priority = IncidentPriority.P3_MODERATE

        else:
            severity = IncidentSeverity.LOW
            priority = IncidentPriority.P4_LOW

        if not reasons:
            reasons.append(
                "Priority based on reported incident severity"
            )

        return IncidentAssessment(
            severity=severity,
            priority=priority,
            score=round(score, 4),
            reasons=tuple(reasons),
        )


# ============================================================
# DUPLICATE INCIDENT DETECTION
# ============================================================

class IncidentDuplicateDetector:

    def __init__(
        self,
        *,
        radius_km: float = 0.5,
        time_window_minutes: int = 120,
    ):
        if radius_km <= 0:
            raise ValueError(
                "Duplicate radius must be positive"
            )

        if time_window_minutes <= 0:
            raise ValueError(
                "Duplicate time window must be positive"
            )

        self.radius_km = radius_km
        self.time_window_minutes = (
            time_window_minutes
        )

    def detect(
        self,
        request: IncidentCreateRequest,
        candidates: Sequence[Incident],
        *,
        now: datetime | None = None,
    ) -> list[DuplicateIncidentMatch]:

        now = now or utc_now()

        reference_time = (
            request.occurred_at or now
        )

        matches: list[DuplicateIncidentMatch] = []

        for candidate in candidates:
            if candidate.status in {
                IncidentStatus.REJECTED,
                IncidentStatus.CANCELLED,
            }:
                continue

            if candidate.incident_type != request.incident_type:
                continue

            candidate_time = (
                candidate.occurred_at
                or candidate.created_at
            )

            difference_minutes = abs(
                (
                    reference_time - candidate_time
                ).total_seconds()
            ) / 60

            if (
                difference_minutes
                > self.time_window_minutes
            ):
                continue

            distance = haversine_km(
                request.location.latitude,
                request.location.longitude,
                candidate.location.latitude,
                candidate.location.longitude,
            )

            if distance > self.radius_km:
                continue

            distance_score = (
                1.0 - distance / self.radius_km
            )

            time_score = (
                1.0
                - difference_minutes
                / self.time_window_minutes
            )

            similarity = (
                distance_score * 0.65
                + time_score * 0.35
            )

            matches.append(
                DuplicateIncidentMatch(
                    incident_id=candidate.id,
                    distance_km=round(distance, 4),
                    time_difference_minutes=round(
                        difference_minutes,
                        2,
                    ),
                    similarity_score=round(
                        similarity,
                        4,
                    ),
                )
            )

        matches.sort(
            key=lambda match: -match.similarity_score
        )

        return matches


# ============================================================
# INCIDENT SERVICE
# ============================================================

class IncidentService:

    TERMINAL_STATUSES = frozenset({
        IncidentStatus.CLOSED,
        IncidentStatus.REJECTED,
        IncidentStatus.CANCELLED,
    })

    def __init__(
        self,
        *,
        repository: IncidentRepository,
        timeline_repository: IncidentTimelineRepository,
        authorization_gateway: IncidentAuthorizationGateway,
        audit_gateway: IncidentAuditGateway | None = None,
        notification_gateway: (
            IncidentNotificationGateway | None
        ) = None,
        priority_engine: IncidentPriorityEngine | None = None,
        duplicate_detector: (
            IncidentDuplicateDetector | None
        ) = None,
    ):
        self.repository = repository
        self.timeline = timeline_repository
        self.authorization = authorization_gateway
        self.audit = audit_gateway
        self.notifications = notification_gateway

        self.priority_engine = (
            priority_engine
            or IncidentPriorityEngine()
        )

        self.duplicate_detector = (
            duplicate_detector
            or IncidentDuplicateDetector()
        )

    async def _audit(
        self,
        action: str,
        incident: Incident,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="incident",
            entity_id=incident.id,
            metadata={
                "district_id": incident.district_id,
                "incident_type": incident.incident_type.value,
                "severity": incident.severity.value,
                "priority": incident.priority.value,
                "status": incident.status.value,
            },
        )

    async def _timeline(
        self,
        incident: Incident,
        *,
        action: str,
        description: str,
        actor_id: int | None,
    ) -> None:

        event = IncidentTimelineEvent(
            id=str(uuid.uuid4()),
            incident_id=incident.id,
            action=action,
            description=description,
            actor_id=actor_id,
        )

        await self.timeline.append(
            event
        )

    async def get_incident(
        self,
        incident_id: str,
    ) -> Incident:

        incident = await self.repository.get_by_id(
            incident_id
        )

        if incident is None:
            raise IncidentNotFoundError(
                "Incident not found"
            )

        return incident

    def _validate_request(
        self,
        request: IncidentCreateRequest,
    ) -> None:

        if request.district_id <= 0:
            raise IncidentValidationError(
                "District ID must be positive"
            )

        if not request.title.strip():
            raise IncidentValidationError(
                "Incident title is required"
            )

        if not request.description.strip():
            raise IncidentValidationError(
                "Incident description is required"
            )

        if len(request.title) > 200:
            raise IncidentValidationError(
                "Incident title exceeds 200 characters"
            )

        if request.occurred_at is not None:
            validate_datetime(
                request.occurred_at,
                "occurred_at",
            )

            if request.occurred_at > utc_now():
                raise IncidentValidationError(
                    "Incident occurrence cannot be in the future"
                )

    async def find_duplicates(
        self,
        request: IncidentCreateRequest,
    ) -> list[DuplicateIncidentMatch]:

        reference_time = (
            request.occurred_at or utc_now()
        )

        since = reference_time - timedelta(
            minutes=self.duplicate_detector.time_window_minutes
        )

        candidates = await self.repository.find_nearby(
            district_id=request.district_id,
            latitude=request.location.latitude,
            longitude=request.location.longitude,
            radius_km=self.duplicate_detector.radius_km,
            since=since,
        )

        return self.duplicate_detector.detect(
            request,
            candidates,
        )

    async def create_incident(
        self,
        request: IncidentCreateRequest,
        *,
        actor_id: int,
        allow_possible_duplicate: bool = False,
    ) -> Incident:

        self._validate_request(request)

        authorized = await self.authorization.can_create(
            user_id=actor_id,
            district_id=request.district_id,
        )

        if not authorized:
            raise IncidentAuthorizationError(
                "Incident creation permission denied"
            )

        if request.external_reference:
            existing = (
                await self.repository
                .find_by_external_reference(
                    source=request.source,
                    external_reference=(
                        request.external_reference
                    ),
                )
            )

            if existing is not None:
                return existing

        duplicates = await self.find_duplicates(
            request
        )

        if duplicates and not allow_possible_duplicate:
            raise IncidentDuplicateError(
                "Possible duplicate incident detected: "
                + ", ".join(
                    match.incident_id
                    for match in duplicates[:3]
                )
            )

        assessment = self.priority_engine.assess(
            incident_type=request.incident_type,
            impact=request.impact,
            reported_severity=request.reported_severity,
        )

        incident = Incident(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            incident_type=request.incident_type,
            source=request.source,
            title=request.title.strip(),
            description=request.description.strip(),
            location=request.location,
            impact=request.impact,
            severity=assessment.severity,
            priority=assessment.priority,
            status=IncidentStatus.REPORTED,
            reported_by=actor_id,
            sos_request_id=request.sos_request_id,
            external_reference=request.external_reference,
            evidence_references=list(
                request.evidence_references
            ),
            occurred_at=request.occurred_at,
            metadata={
                **dict(request.metadata),
                "priority_score": assessment.score,
                "priority_reasons": list(
                    assessment.reasons
                ),
                "possible_duplicate_ids": [
                    match.incident_id
                    for match in duplicates
                ],
            },
        )

        created = await self.repository.create(
            incident
        )

        await self._timeline(
            created,
            action="incident.created",
            description="Incident report registered",
            actor_id=actor_id,
        )

        await self._audit(
            "incident.created",
            created,
            actor_id,
        )

        logger.info(
            "Incident created: %s",
            created.id,
        )

        return created

    async def submit_for_review(
        self,
        incident_id: str,
        *,
        actor_id: int,
    ) -> Incident:

        incident = await self.get_incident(
            incident_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=incident.district_id,
        ):
            raise IncidentAuthorizationError(
                "Incident management permission denied"
            )

        if incident.status != IncidentStatus.REPORTED:
            raise IncidentConflictError(
                "Only reported incidents can enter review"
            )

        updated = replace(
            incident,
            status=IncidentStatus.UNDER_REVIEW,
            updated_at=utc_now(),
            version=incident.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=incident.version,
        )

        await self._timeline(
            saved,
            action="incident.review_started",
            description="Incident submitted for verification",
            actor_id=actor_id,
        )

        await self._audit(
            "incident.review_started",
            saved,
            actor_id,
        )

        return saved

    async def verify_incident(
        self,
        incident_id: str,
        *,
        reviewer_id: int,
        review_notes: str,
    ) -> Incident:

        incident = await self.get_incident(
            incident_id
        )

        if not await self.authorization.can_review(
            user_id=reviewer_id,
            district_id=incident.district_id,
        ):
            raise IncidentAuthorizationError(
                "Incident verification permission denied"
            )

        if incident.status != IncidentStatus.UNDER_REVIEW:
            raise IncidentConflictError(
                "Incident is not under review"
            )

        if incident.reported_by == reviewer_id:
            raise IncidentAuthorizationError(
                "Independent reviewer required"
            )

        if not review_notes.strip():
            raise IncidentValidationError(
                "Verification notes are required"
            )

        updated = replace(
            incident,
            status=IncidentStatus.VERIFIED,
            verified_by=reviewer_id,
            verified_at=utc_now(),
            review_notes=review_notes.strip(),
            updated_at=utc_now(),
            version=incident.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=incident.version,
        )

        await self._timeline(
            saved,
            action="incident.verified",
            description="Incident verified by authorized reviewer",
            actor_id=reviewer_id,
        )

        await self._audit(
            "incident.verified",
            saved,
            reviewer_id,
        )

        return saved

    async def reject_incident(
        self,
        incident_id: str,
        *,
        reviewer_id: int,
        reason: str,
    ) -> Incident:

        incident = await self.get_incident(
            incident_id
        )

        if not await self.authorization.can_review(
            user_id=reviewer_id,
            district_id=incident.district_id,
        ):
            raise IncidentAuthorizationError(
                "Incident review permission denied"
            )

        if incident.status != IncidentStatus.UNDER_REVIEW:
            raise IncidentConflictError(
                "Incident is not under review"
            )

        if not reason.strip():
            raise IncidentValidationError(
                "Rejection reason is required"
            )

        updated = replace(
            incident,
            status=IncidentStatus.REJECTED,
            verified_by=reviewer_id,
            review_notes=reason.strip(),
            updated_at=utc_now(),
            version=incident.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=incident.version,
        )

        await self._timeline(
            saved,
            action="incident.rejected",
            description="Incident rejected after review",
            actor_id=reviewer_id,
        )

        await self._audit(
            "incident.rejected",
            saved,
            reviewer_id,
        )

        return saved

    async def assign_team(
        self,
        incident_id: str,
        *,
        team_id: str,
        actor_id: int,
    ) -> Incident:

        incident = await self.get_incident(
            incident_id
        )

        if not await self.authorization.can_assign(
            user_id=actor_id,
            district_id=incident.district_id,
        ):
            raise IncidentAuthorizationError(
                "Incident assignment permission denied"
            )

        if incident.status != IncidentStatus.VERIFIED:
            raise IncidentConflictError(
                "Only verified incidents can be assigned"
            )

        if not team_id.strip():
            raise IncidentValidationError(
                "Team ID is required"
            )

        # Assignment is an administrative association.
        # Operational dispatch must be handled by
        # dispatch_service.py with separate approval.

        updated = replace(
            incident,
            status=IncidentStatus.ASSIGNED,
            assigned_team_id=team_id,
            assigned_by=actor_id,
            assigned_at=utc_now(),
            updated_at=utc_now(),
            version=incident.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=incident.version,
        )

        await self._timeline(
            saved,
            action="incident.assigned",
            description="Incident assigned for coordination",
            actor_id=actor_id,
        )

        await self._audit(
            "incident.assigned",
            saved,
            actor_id,
        )

        if self.notifications is not None:
            try:
                await self.notifications.notify_assignment(
                    saved
                )
            except Exception:
                logger.exception(
                    "Incident assignment notification failed: %s",
                    saved.id,
                )

        return saved

    async def update_status(
        self,
        incident_id: str,
        *,
        new_status: IncidentStatus,
        actor_id: int,
    ) -> Incident:

        incident = await self.get_incident(
            incident_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=incident.district_id,
        ):
            raise IncidentAuthorizationError(
                "Incident status update permission denied"
            )

        transitions = {
            IncidentStatus.ASSIGNED: {
                IncidentStatus.IN_PROGRESS,
            },
            IncidentStatus.IN_PROGRESS: {
                IncidentStatus.CONTAINED,
                IncidentStatus.RESOLVED,
            },
            IncidentStatus.CONTAINED: {
                IncidentStatus.RESOLVED,
            },
            IncidentStatus.RESOLVED: {
                IncidentStatus.CLOSED,
            },
        }

        allowed = transitions.get(
            incident.status,
            set(),
        )

        if new_status not in allowed:
            raise IncidentConflictError(
                f"Invalid transition: "
                f"{incident.status.value} -> "
                f"{new_status.value}"
            )

        now = utc_now()

        updated = replace(
            incident,
            status=new_status,
            resolved_at=(
                now
                if new_status == IncidentStatus.RESOLVED
                else incident.resolved_at
            ),
            closed_at=(
                now
                if new_status == IncidentStatus.CLOSED
                else incident.closed_at
            ),
            updated_at=now,
            version=incident.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=incident.version,
        )

        await self._timeline(
            saved,
            action=f"incident.{new_status.value}",
            description=(
                f"Incident status changed to "
                f"{new_status.value}"
            ),
            actor_id=actor_id,
        )

        await self._audit(
            f"incident.{new_status.value}",
            saved,
            actor_id,
        )

        return saved

    async def update_impact(
        self,
        incident_id: str,
        *,
        impact: IncidentImpact,
        actor_id: int,
    ) -> Incident:

        incident = await self.get_incident(
            incident_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=incident.district_id,
        ):
            raise IncidentAuthorizationError(
                "Incident impact update permission denied"
            )

        if incident.status in self.TERMINAL_STATUSES:
            raise IncidentConflictError(
                "Cannot update a terminal incident"
            )

        assessment = self.priority_engine.assess(
            incident_type=incident.incident_type,
            impact=impact,
            reported_severity=incident.severity,
        )

        updated = replace(
            incident,
            impact=impact,
            severity=assessment.severity,
            priority=assessment.priority,
            updated_at=utc_now(),
            version=incident.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=incident.version,
        )

        await self._timeline(
            saved,
            action="incident.impact_updated",
            description="Incident impact assessment updated",
            actor_id=actor_id,
        )

        await self._audit(
            "incident.impact_updated",
            saved,
            actor_id,
        )

        return saved

    async def escalate_incident(
        self,
        incident_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> Incident:

        incident = await self.get_incident(
            incident_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=incident.district_id,
        ):
            raise IncidentAuthorizationError(
                "Incident escalation permission denied"
            )

        if incident.status in self.TERMINAL_STATUSES:
            raise IncidentConflictError(
                "Cannot escalate a terminal incident"
            )

        if not reason.strip():
            raise IncidentValidationError(
                "Escalation reason is required"
            )

        updated = replace(
            incident,
            severity=IncidentSeverity.CRITICAL,
            priority=IncidentPriority.P1_CRITICAL,
            updated_at=utc_now(),
            version=incident.version + 1,
            metadata={
                **incident.metadata,
                "last_escalation_reason": reason.strip(),
                "last_escalated_by": actor_id,
            },
        )

        saved = await self.repository.update(
            updated,
            expected_version=incident.version,
        )

        await self._timeline(
            saved,
            action="incident.escalated",
            description=reason.strip(),
            actor_id=actor_id,
        )

        await self._audit(
            "incident.escalated",
            saved,
            actor_id,
        )

        if self.notifications is not None:
            try:
                delivered = (
                    await self.notifications.notify_escalation(
                        saved
                    )
                )

                if not delivered:
                    logger.error(
                        "Escalation notification unconfirmed: %s",
                        saved.id,
                    )

            except Exception:
                logger.exception(
                    "Escalation notification failed: %s",
                    saved.id,
                )

        return saved

    async def add_rescue_mission(
        self,
        incident_id: str,
        *,
        rescue_mission_id: str,
        actor_id: int,
    ) -> Incident:

        incident = await self.get_incident(
            incident_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=incident.district_id,
        ):
            raise IncidentAuthorizationError(
                "Rescue mission linking permission denied"
            )

        if incident.status in self.TERMINAL_STATUSES:
            raise IncidentConflictError(
                "Cannot link missions to terminal incidents"
            )

        if not rescue_mission_id.strip():
            raise IncidentValidationError(
                "Rescue mission ID is required"
            )

        if rescue_mission_id in incident.rescue_mission_ids:
            return incident

        updated = replace(
            incident,
            rescue_mission_ids=[
                *incident.rescue_mission_ids,
                rescue_mission_id,
            ],
            updated_at=utc_now(),
            version=incident.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=incident.version,
        )

        await self._audit(
            "incident.rescue_mission_linked",
            saved,
            actor_id,
        )

        return saved

    async def list_district_incidents(
        self,
        district_id: int,
    ) -> Sequence[Incident]:

        if district_id <= 0:
            raise IncidentValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_active_incidents(
        self,
        district_id: int,
    ) -> list[Incident]:

        incidents = await self.list_district_incidents(
            district_id
        )

        return [
            incident
            for incident in incidents
            if incident.status not in {
                IncidentStatus.RESOLVED,
                IncidentStatus.CLOSED,
                IncidentStatus.REJECTED,
                IncidentStatus.CANCELLED,
            }
        ]

    async def get_incident_timeline(
        self,
        incident_id: str,
    ) -> Sequence[IncidentTimelineEvent]:

        await self.get_incident(
            incident_id
        )

        return await self.timeline.list_by_incident(
            incident_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        incidents = await self.list_district_incidents(
            district_id
        )

        status_counts = {
            status.value: 0
            for status in IncidentStatus
        }

        severity_counts = {
            severity.value: 0
            for severity in IncidentSeverity
        }

        type_counts = {
            incident_type.value: 0
            for incident_type in IncidentType
        }

        for incident in incidents:
            status_counts[
                incident.status.value
            ] += 1

            severity_counts[
                incident.severity.value
            ] += 1

            type_counts[
                incident.incident_type.value
            ] += 1

        return {
            "district_id": district_id,
            "total_incidents": len(incidents),
            "active_incidents": sum(
                incident.status not in {
                    IncidentStatus.RESOLVED,
                    IncidentStatus.CLOSED,
                    IncidentStatus.REJECTED,
                    IncidentStatus.CANCELLED,
                }
                for incident in incidents
            ),
            "total_affected_people": sum(
                incident.impact.affected_people
                for incident in incidents
            ),
            "total_people_at_risk": sum(
                incident.impact.people_at_risk
                for incident in incidents
            ),
            "total_injured": sum(
                incident.impact.injured_people
                for incident in incidents
            ),
            "total_missing": sum(
                incident.impact.missing_people
                for incident in incidents
            ),
            "status_counts": status_counts,
            "severity_counts": severity_counts,
            "incident_type_counts": type_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_incident_report(
        self,
        incident_id: str,
    ) -> dict[str, Any]:

        incident = await self.get_incident(
            incident_id
        )

        timeline = await self.get_incident_timeline(
            incident_id
        )

        return {
            "incident_id": incident.id,
            "district_id": incident.district_id,
            "incident_type": incident.incident_type.value,
            "source": incident.source.value,
            "title": incident.title,
            "description": incident.description,
            "severity": incident.severity.value,
            "priority": incident.priority.value,
            "status": incident.status.value,
            "location": {
                "latitude": incident.location.latitude,
                "longitude": incident.location.longitude,
                "address": incident.location.address,
                "landmark": incident.location.landmark,
                "panchayat": incident.location.panchayat,
                "village": incident.location.village,
            },
            "impact": {
                "affected_people": (
                    incident.impact.affected_people
                ),
                "people_at_risk": (
                    incident.impact.people_at_risk
                ),
                "injured_people": (
                    incident.impact.injured_people
                ),
                "missing_people": (
                    incident.impact.missing_people
                ),
                "confirmed_deaths": (
                    incident.impact.confirmed_deaths
                ),
                "displaced_people": (
                    incident.impact.displaced_people
                ),
                "affected_households": (
                    incident.impact.affected_households
                ),
            },
            "coordination": {
                "assigned_team_id": (
                    incident.assigned_team_id
                ),
                "rescue_mission_ids": (
                    incident.rescue_mission_ids
                ),
                "sos_request_id": (
                    incident.sos_request_id
                ),
            },
            "verification": {
                "verified_by": incident.verified_by,
                "verified_at": (
                    incident.verified_at.isoformat()
                    if incident.verified_at
                    else None
                ),
                "review_notes": incident.review_notes,
            },
            "timeline": [
                {
                    "event_id": event.id,
                    "action": event.action,
                    "description": event.description,
                    "actor_id": event.actor_id,
                    "timestamp": event.timestamp.isoformat(),
                }
                for event in timeline
            ],
            "created_at": incident.created_at.isoformat(),
            "updated_at": incident.updated_at.isoformat(),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Incident records are coordination data. "
                "Reports may contain unverified information. "
                "Official emergency response decisions require "
                "authorized human assessment."
            ),
        }
