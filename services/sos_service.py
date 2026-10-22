from __future__ import annotations

"""
backend/services/sos_service.py

Kerala Flood Relief Coordination Agent
=======================================

Emergency SOS Request Management Service.

Features:
- Emergency SOS registration
- Citizen and anonymous SOS requests
- GPS-based emergency location
- Flood, medical, rescue and shelter emergencies
- Duplicate SOS detection
- Human-reviewed emergency triage
- Vulnerable-person information
- SOS verification and escalation
- Rescue incident coordination
- Human-approved dispatch handoff
- Emergency contact notification requests
- SOS status and progress tracking
- Cancellation and false-alarm handling
- Priority queue and district statistics
- Audit logging and permission checks
- Idempotent external request handling
- Structured emergency reports

IMPORTANT:
- This service is a coordination component, not
  a replacement for official emergency services.
- A successfully created SOS record does not mean
  emergency responders have been contacted.
- No automatic rescue dispatch is performed.
- AI-derived urgency is advisory only.
- Critical cases require immediate human review.
- Production deployments require durable storage,
  transactional outboxes, secure integrations,
  retries, monitoring and operational testing.
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

class SOSPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SOSStatus(str, Enum):
    RECEIVED = "received"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    TRIAGED = "triaged"
    PENDING_DISPATCH_APPROVAL = "pending_dispatch_approval"
    DISPATCH_APPROVED = "dispatch_approved"
    DISPATCH_REQUESTED = "dispatch_requested"
    RESPONDER_ACKNOWLEDGED = "responder_acknowledged"
    RESPONDER_EN_ROUTE = "responder_en_route"
    RESPONDER_ON_SCENE = "responder_on_scene"
    RESCUE_IN_PROGRESS = "rescue_in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    DUPLICATE = "duplicate"
    FALSE_ALARM = "false_alarm"


class SOSEmergencyType(str, Enum):
    FLOOD_TRAPPED = "flood_trapped"
    WATER_RESCUE = "water_rescue"
    MEDICAL_EMERGENCY = "medical_emergency"
    BUILDING_COLLAPSE = "building_collapse"
    LANDSLIDE = "landslide"
    MISSING_PERSON = "missing_person"
    STRANDED_VEHICLE = "stranded_vehicle"
    EVACUATION_REQUIRED = "evacuation_required"
    SHELTER_REQUIRED = "shelter_required"
    FOOD_WATER_REQUIRED = "food_water_required"
    ELECTRICAL_HAZARD = "electrical_hazard"
    FIRE = "fire"
    OTHER = "other"


class SOSSource(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB = "web"
    PHONE_OPERATOR = "phone_operator"
    FIELD_VOLUNTEER = "field_volunteer"
    GOVERNMENT_CONTROL_ROOM = "government_control_room"
    INTEGRATION = "integration"


class SOSVerificationMethod(str, Enum):
    CALLBACK = "callback"
    FIELD_CONFIRMATION = "field_confirmation"
    AUTHORIZED_OPERATOR = "authorized_operator"
    OFFICIAL_SOURCE = "official_source"
    MULTI_SOURCE = "multi_source"


class SOSEscalationLevel(str, Enum):
    NONE = "none"
    DISTRICT_CONTROL_ROOM = "district_control_room"
    DISTRICT_EMERGENCY_COORDINATOR = (
        "district_emergency_coordinator"
    )
    STATE_EMERGENCY_COORDINATOR = (
        "state_emergency_coordinator"
    )


class SOSNotificationChannel(str, Enum):
    SMS = "sms"
    PUSH = "push"
    EMAIL = "email"
    VOICE = "voice"


class SOSEventType(str, Enum):
    CREATED = "created"
    DUPLICATE_DETECTED = "duplicate_detected"
    VERIFIED = "verified"
    TRIAGED = "triaged"
    ESCALATED = "escalated"
    DISPATCH_SUBMITTED = "dispatch_submitted"
    DISPATCH_APPROVED = "dispatch_approved"
    DISPATCH_REQUESTED = "dispatch_requested"
    RESPONDER_ACKNOWLEDGED = "responder_acknowledged"
    RESPONDER_EN_ROUTE = "responder_en_route"
    RESPONDER_ON_SCENE = "responder_on_scene"
    RESCUE_STARTED = "rescue_started"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    FALSE_ALARM = "false_alarm"
    STATUS_NOTE = "status_note"


# ============================================================
# EXCEPTIONS
# ============================================================

class SOSServiceError(Exception):
    pass


class SOSValidationError(SOSServiceError):
    pass


class SOSNotFoundError(SOSServiceError):
    pass


class SOSConflictError(SOSServiceError):
    pass


class SOSAuthorizationError(SOSServiceError):
    pass


class SOSVerificationError(SOSServiceError):
    pass


class SOSDispatchError(SOSServiceError):
    pass


class SOSIntegrationError(SOSServiceError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_text(
    value: str,
    name: str,
    *,
    max_length: int = 2000,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SOSValidationError(
            f"{name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise SOSValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if not isinstance(value, datetime):
        raise SOSValidationError(
            f"{name} must be a datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise SOSValidationError(
            f"{name} must be timezone-aware"
        )


def validate_non_negative_integer(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise SOSValidationError(
            f"{name} must be a non-negative integer"
        )


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
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

    a = max(
        0.0,
        min(1.0, a),
    )

    return (
        2
        * radius_km
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a),
        )
    )


# ============================================================
# LOCATION
# ============================================================

@dataclass(frozen=True)
class SOSLocation:
    district_id: int

    latitude: float
    longitude: float

    address: str = ""
    landmark: str = ""
    panchayat: str = ""
    taluk: str = ""

    accuracy_meters: float | None = None

    captured_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.district_id, bool)
            or not isinstance(self.district_id, int)
            or self.district_id <= 0
        ):
            raise SOSValidationError(
                "Invalid district ID"
            )

        for name, value, minimum, maximum in (
            ("latitude", self.latitude, -90, 90),
            ("longitude", self.longitude, -180, 180),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not minimum <= value <= maximum
            ):
                raise SOSValidationError(
                    f"Invalid {name}"
                )

        if self.accuracy_meters is not None:
            if (
                not math.isfinite(self.accuracy_meters)
                or self.accuracy_meters < 0
            ):
                raise SOSValidationError(
                    "Invalid GPS accuracy"
                )

        validate_datetime(
            self.captured_at,
            "captured_at",
        )


# ============================================================
# PEOPLE AND EMERGENCY DETAILS
# ============================================================

@dataclass(frozen=True)
class SOSPeopleInformation:
    people_affected: int

    children: int = 0
    elderly: int = 0

    disabled_people: int = 0
    pregnant_people: int = 0

    injured_people: int = 0
    critical_patients: int = 0

    missing_people: int = 0

    def __post_init__(self) -> None:
        for name in (
            "people_affected",
            "children",
            "elderly",
            "disabled_people",
            "pregnant_people",
            "injured_people",
            "critical_patients",
            "missing_people",
        ):
            validate_non_negative_integer(
                getattr(self, name),
                name,
            )

        if self.people_affected <= 0:
            raise SOSValidationError(
                "At least one affected person is required"
            )

        for name in (
            "children",
            "elderly",
            "disabled_people",
            "pregnant_people",
            "injured_people",
            "critical_patients",
            "missing_people",
        ):
            if getattr(self, name) > self.people_affected:
                raise SOSValidationError(
                    f"{name} exceeds affected population"
                )


@dataclass(frozen=True)
class SOSHazardInformation:
    rising_water: bool = False
    strong_current: bool = False

    building_instability: bool = False
    electrical_hazard: bool = False

    landslide_risk: bool = False
    fire_or_smoke: bool = False

    blocked_exit: bool = False

    water_depth_meters: float | None = None

    def __post_init__(self) -> None:
        if self.water_depth_meters is not None:
            if (
                isinstance(self.water_depth_meters, bool)
                or not isinstance(
                    self.water_depth_meters,
                    (int, float),
                )
                or not math.isfinite(
                    self.water_depth_meters
                )
                or self.water_depth_meters < 0
            ):
                raise SOSValidationError(
                    "Invalid reported water depth"
                )


@dataclass(frozen=True)
class SOSContact:
    contact_reference: str | None = None

    anonymous: bool = False

    preferred_language: str = "ml"

    emergency_contact_reference: str | None = None

    notification_consent: bool = False


# ============================================================
# SOS REQUEST MODELS
# ============================================================

@dataclass(frozen=True)
class SOSCreateRequest:
    emergency_type: SOSEmergencyType

    source: SOSSource

    location: SOSLocation

    people: SOSPeopleInformation

    hazards: SOSHazardInformation

    description: str

    contact: SOSContact = field(
        default_factory=SOSContact
    )

    incident_id: str | None = None

    external_reference: str | None = None

    media_references: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class SOSTriageAssessment:
    recommended_priority: SOSPriority

    risk_score: float

    reasons: tuple[str, ...]

    medical_review_required: bool

    rescue_review_required: bool

    specialist_review_required: bool

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class SOSVerification:
    method: SOSVerificationMethod

    verified_by: int

    verified_at: datetime

    evidence_reference: str

    notes: str = ""

    def __post_init__(self) -> None:
        validate_datetime(
            self.verified_at,
            "verified_at",
        )

        require_text(
            self.evidence_reference,
            "evidence_reference",
        )


@dataclass(frozen=True)
class SOSEvent:
    id: str

    event_type: SOSEventType

    actor_id: int | None

    occurred_at: datetime

    description: str = ""

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class SOSRequest:
    id: str

    district_id: int

    emergency_type: SOSEmergencyType
    source: SOSSource

    location: SOSLocation

    people: SOSPeopleInformation
    hazards: SOSHazardInformation

    description: str

    contact: SOSContact

    status: SOSStatus

    priority: SOSPriority

    created_by: int | None = None

    incident_id: str | None = None

    external_reference: str | None = None

    duplicate_of: str | None = None

    triage: SOSTriageAssessment | None = None
    verification: SOSVerification | None = None

    escalation_level: SOSEscalationLevel = (
        SOSEscalationLevel.NONE
    )

    rescue_mission_id: str | None = None

    dispatch_approved_by: int | None = None
    dispatch_approval_reference: str | None = None

    media_references: list[str] = field(
        default_factory=list
    )

    events: list[SOSEvent] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    verified_at: datetime | None = None
    triaged_at: datetime | None = None

    dispatch_requested_at: datetime | None = None
    resolved_at: datetime | None = None
    cancelled_at: datetime | None = None

    resolution_notes: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class SOSRepository(Protocol):

    async def create(
        self,
        request: SOSRequest,
    ) -> SOSRequest:
        ...

    async def get_by_id(
        self,
        sos_id: str,
    ) -> SOSRequest | None:
        ...

    async def get_by_external_reference(
        self,
        external_reference: str,
    ) -> SOSRequest | None:
        ...

    async def update(
        self,
        request: SOSRequest,
        *,
        expected_version: int,
    ) -> SOSRequest:
        """
        Must enforce atomic optimistic concurrency.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[SOSRequest]:
        ...

    async def list_recent_by_district(
        self,
        district_id: int,
        *,
        since: datetime,
    ) -> Sequence[SOSRequest]:
        ...

    async def list_by_incident(
        self,
        incident_id: str,
    ) -> Sequence[SOSRequest]:
        ...


class SOSAuthorizationGateway(Protocol):

    async def can_create(
        self,
        *,
        user_id: int | None,
        district_id: int,
        source: SOSSource,
    ) -> bool:
        ...

    async def can_verify(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_triage(
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

    async def can_approve_dispatch(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_view(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class SOSAuditGateway(Protocol):

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


class SOSRescueGateway(Protocol):

    async def request_rescue_coordination(
        self,
        *,
        sos_id: str,
        district_id: int,
        emergency_type: SOSEmergencyType,
        priority: SOSPriority,
        approval_reference: str,
    ) -> str:
        """
        Register a coordination request.

        Returning an ID does not prove a rescue team
        was dispatched.
        """
        ...


class SOSNotificationGateway(Protocol):

    async def request_notification(
        self,
        *,
        sos_id: str,
        contact_reference: str,
        channel: SOSNotificationChannel,
        message: str,
        idempotency_key: str,
    ) -> str:
        """
        Queue an authorized notification.

        Returning a reference means accepted for
        processing, not delivered.
        """
        ...


# ============================================================
# SOS TRIAGE ENGINE
# ============================================================

class SOSTriageEngine:
    """
    Advisory risk scoring.

    The illustrative score is not a validated
    emergency triage algorithm.
    """

    def evaluate(
        self,
        request: SOSRequest,
    ) -> SOSTriageAssessment:

        score = 0.0

        reasons: list[str] = []

        people = request.people
        hazards = request.hazards

        medical_review = False
        rescue_review = False
        specialist_review = False

        score += min(
            people.people_affected / 25,
            1.0,
        ) * 0.10

        if people.children > 0:
            score += 0.06
            reasons.append(
                "Children are affected"
            )

        if people.elderly > 0:
            score += 0.06
            reasons.append(
                "Elderly people are affected"
            )

        if people.disabled_people > 0:
            score += 0.06
            reasons.append(
                "People with disabilities are affected"
            )

        if people.pregnant_people > 0:
            score += 0.06
            reasons.append(
                "Pregnant people are affected"
            )

        if people.injured_people > 0:
            score += 0.12
            medical_review = True
            reasons.append(
                "Injured people reported"
            )

        if people.critical_patients > 0:
            score += 0.25
            medical_review = True
            specialist_review = True
            reasons.append(
                "Critical patients reported"
            )

        if people.missing_people > 0:
            score += 0.12
            rescue_review = True
            reasons.append(
                "Missing people reported"
            )

        if hazards.rising_water:
            score += 0.10
            rescue_review = True
            reasons.append(
                "Rising floodwater reported"
            )

        if hazards.strong_current:
            score += 0.15
            rescue_review = True
            specialist_review = True
            reasons.append(
                "Strong water current reported"
            )

        if hazards.building_instability:
            score += 0.15
            specialist_review = True
            reasons.append(
                "Building instability reported"
            )

        if hazards.electrical_hazard:
            score += 0.15
            specialist_review = True
            reasons.append(
                "Electrical hazard reported"
            )

        if hazards.landslide_risk:
            score += 0.15
            specialist_review = True
            reasons.append(
                "Landslide hazard reported"
            )

        if hazards.fire_or_smoke:
            score += 0.15
            specialist_review = True
            reasons.append(
                "Fire or smoke reported"
            )

        if hazards.blocked_exit:
            score += 0.15
            rescue_review = True
            reasons.append(
                "Exit is blocked"
            )

        if request.emergency_type in {
            SOSEmergencyType.FLOOD_TRAPPED,
            SOSEmergencyType.WATER_RESCUE,
            SOSEmergencyType.BUILDING_COLLAPSE,
            SOSEmergencyType.LANDSLIDE,
        }:
            score += 0.15
            rescue_review = True

        if (
            request.emergency_type
            == SOSEmergencyType.MEDICAL_EMERGENCY
        ):
            score += 0.20
            medical_review = True

        score = min(
            score,
            1.0,
        )

        if score >= 0.75:
            priority = SOSPriority.CRITICAL

        elif score >= 0.50:
            priority = SOSPriority.HIGH

        elif score >= 0.25:
            priority = SOSPriority.MEDIUM

        else:
            priority = SOSPriority.LOW

        return SOSTriageAssessment(
            recommended_priority=priority,
            risk_score=round(score, 4),
            reasons=tuple(reasons),
            medical_review_required=medical_review,
            rescue_review_required=rescue_review,
            specialist_review_required=specialist_review,
        )


# ============================================================
# DUPLICATE DETECTOR
# ============================================================

class SOSDuplicateDetector:

    ACTIVE_STATUSES = frozenset({
        SOSStatus.RECEIVED,
        SOSStatus.PENDING_VERIFICATION,
        SOSStatus.VERIFIED,
        SOSStatus.TRIAGED,
        SOSStatus.PENDING_DISPATCH_APPROVAL,
        SOSStatus.DISPATCH_APPROVED,
        SOSStatus.DISPATCH_REQUESTED,
        SOSStatus.RESPONDER_ACKNOWLEDGED,
        SOSStatus.RESPONDER_EN_ROUTE,
        SOSStatus.RESPONDER_ON_SCENE,
        SOSStatus.RESCUE_IN_PROGRESS,
    })

    def __init__(
        self,
        *,
        maximum_distance_km: float = 0.15,
        maximum_age: timedelta = timedelta(
            minutes=20
        ),
    ):
        self.maximum_distance_km = (
            maximum_distance_km
        )

        self.maximum_age = maximum_age

    def find_possible_duplicate(
        self,
        request: SOSCreateRequest,
        recent: Sequence[SOSRequest],
        *,
        now: datetime | None = None,
    ) -> SOSRequest | None:

        now = now or utc_now()

        for existing in recent:
            if existing.status not in self.ACTIVE_STATUSES:
                continue

            if existing.emergency_type != request.emergency_type:
                continue

            if (
                now - existing.created_at
                > self.maximum_age
            ):
                continue

            distance = haversine_distance_km(
                request.location.latitude,
                request.location.longitude,
                existing.location.latitude,
                existing.location.longitude,
            )

            if distance > self.maximum_distance_km:
                continue

            # Geographical proximity is not sufficient
            # to automatically merge emergencies.
            #
            # A possible duplicate is only returned
            # when a shared external/household reference
            # also supports the match.
            if (
                request.external_reference
                and request.external_reference
                == existing.external_reference
            ):
                return existing

            if (
                request.contact.contact_reference
                and request.contact.contact_reference
                == existing.contact.contact_reference
            ):
                return existing

        return None


# ============================================================
# SOS SERVICE
# ============================================================

class SOSService:

    TERMINAL_STATUSES = frozenset({
        SOSStatus.RESOLVED,
        SOSStatus.CANCELLED,
        SOSStatus.DUPLICATE,
        SOSStatus.FALSE_ALARM,
    })

    def __init__(
        self,
        *,
        repository: SOSRepository,
        authorization_gateway: SOSAuthorizationGateway,
        rescue_gateway: SOSRescueGateway,
        notification_gateway: SOSNotificationGateway | None = None,
        audit_gateway: SOSAuditGateway | None = None,
        triage_engine: SOSTriageEngine | None = None,
        duplicate_detector: SOSDuplicateDetector | None = None,
    ):
        self.repository = repository
        self.authorization = authorization_gateway
        self.rescue = rescue_gateway
        self.notifications = notification_gateway
        self.audit = audit_gateway

        self.triage_engine = (
            triage_engine
            or SOSTriageEngine()
        )

        self.duplicate_detector = (
            duplicate_detector
            or SOSDuplicateDetector()
        )

    @staticmethod
    def _event(
        event_type: SOSEventType,
        *,
        actor_id: int | None,
        description: str = "",
    ) -> SOSEvent:

        return SOSEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            actor_id=actor_id,
            occurred_at=utc_now(),
            description=description,
        )

    async def _audit(
        self,
        action: str,
        sos: SOSRequest,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="sos_request",
            entity_id=sos.id,
            metadata={
                "district_id": sos.district_id,
                "status": sos.status.value,
                "priority": sos.priority.value,
                "version": sos.version,
            },
        )

    async def _save(
        self,
        original: SOSRequest,
        updated: SOSRequest,
        *,
        actor_id: int,
        action: str,
    ) -> SOSRequest:

        saved = await self.repository.update(
            updated,
            expected_version=original.version,
        )

        await self._audit(
            action,
            saved,
            actor_id,
        )

        return saved

    async def get_sos(
        self,
        sos_id: str,
    ) -> SOSRequest:

        sos = await self.repository.get_by_id(
            sos_id
        )

        if sos is None:
            raise SOSNotFoundError(
                "SOS request not found"
            )

        return sos

    async def create_sos(
        self,
        request: SOSCreateRequest,
        *,
        actor_id: int | None = None,
    ) -> SOSRequest:

        if not await self.authorization.can_create(
            user_id=actor_id,
            district_id=request.location.district_id,
            source=request.source,
        ):
            raise SOSAuthorizationError(
                "SOS registration denied"
            )

        description = require_text(
            request.description,
            "Emergency description",
        )

        if request.external_reference:
            existing = (
                await self.repository
                .get_by_external_reference(
                    request.external_reference
                )
            )

            if existing is not None:
                return existing

        recent = (
            await self.repository.list_recent_by_district(
                request.location.district_id,
                since=(
                    utc_now()
                    - timedelta(minutes=20)
                ),
            )
        )

        possible_duplicate = (
            self.duplicate_detector
            .find_possible_duplicate(
                request,
                recent,
            )
        )

        sos = SOSRequest(
            id=str(uuid.uuid4()),
            district_id=request.location.district_id,
            emergency_type=request.emergency_type,
            source=request.source,
            location=request.location,
            people=request.people,
            hazards=request.hazards,
            description=description,
            contact=request.contact,
            status=SOSStatus.RECEIVED,
            priority=SOSPriority.MEDIUM,
            created_by=actor_id,
            incident_id=request.incident_id,
            external_reference=request.external_reference,
            duplicate_of=(
                possible_duplicate.id
                if possible_duplicate
                else None
            ),
            media_references=list(
                request.media_references
            ),
            events=[
                self._event(
                    SOSEventType.CREATED,
                    actor_id=actor_id,
                )
            ],
            metadata=dict(request.metadata),
        )

        if possible_duplicate:
            sos.events.append(
                self._event(
                    SOSEventType.DUPLICATE_DETECTED,
                    actor_id=actor_id,
                    description=(
                        "Possible duplicate requires "
                        "human verification"
                    ),
                )
            )

        created = await self.repository.create(
            sos
        )

        await self._audit(
            "sos.created",
            created,
            actor_id,
        )

        return created

    async def verify_sos(
        self,
        sos_id: str,
        *,
        reviewer_id: int,
        verification: SOSVerification,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_verify(
            user_id=reviewer_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "SOS verification denied"
            )

        if sos.status not in {
            SOSStatus.RECEIVED,
            SOSStatus.PENDING_VERIFICATION,
        }:
            raise SOSConflictError(
                "SOS cannot be verified in current state"
            )

        if verification.verified_by != reviewer_id:
            raise SOSVerificationError(
                "Verification reviewer mismatch"
            )

        updated = replace(
            sos,
            status=SOSStatus.VERIFIED,
            verification=verification,
            verified_at=utc_now(),
            events=[
                *sos.events,
                self._event(
                    SOSEventType.VERIFIED,
                    actor_id=reviewer_id,
                ),
            ],
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=reviewer_id,
            action="sos.verified",
        )

    async def triage_sos(
        self,
        sos_id: str,
        *,
        actor_id: int,
        confirmed_priority: SOSPriority | None = None,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_triage(
            user_id=actor_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "SOS triage denied"
            )

        if sos.status != SOSStatus.VERIFIED:
            raise SOSConflictError(
                "SOS must be verified before triage"
            )

        assessment = self.triage_engine.evaluate(
            sos
        )

        priority = (
            confirmed_priority
            or assessment.recommended_priority
        )

        updated = replace(
            sos,
            status=SOSStatus.TRIAGED,
            priority=priority,
            triage=assessment,
            triaged_at=utc_now(),
            events=[
                *sos.events,
                self._event(
                    SOSEventType.TRIAGED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=actor_id,
            action="sos.triaged",
        )

    async def escalate_sos(
        self,
        sos_id: str,
        *,
        actor_id: int,
        level: SOSEscalationLevel,
        reason: str,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "SOS escalation denied"
            )

        if sos.status in self.TERMINAL_STATUSES:
            raise SOSConflictError(
                "Cannot escalate terminal SOS"
            )

        reason = require_text(
            reason,
            "Escalation reason",
        )

        updated = replace(
            sos,
            priority=SOSPriority.CRITICAL,
            escalation_level=level,
            events=[
                *sos.events,
                self._event(
                    SOSEventType.ESCALATED,
                    actor_id=actor_id,
                    description=reason,
                ),
            ],
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=actor_id,
            action="sos.escalated",
        )

    async def submit_for_dispatch_approval(
        self,
        sos_id: str,
        *,
        actor_id: int,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "Dispatch submission denied"
            )

        if sos.status != SOSStatus.TRIAGED:
            raise SOSConflictError(
                "SOS must be triaged first"
            )

        updated = replace(
            sos,
            status=SOSStatus.PENDING_DISPATCH_APPROVAL,
            events=[
                *sos.events,
                self._event(
                    SOSEventType.DISPATCH_SUBMITTED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=actor_id,
            action="sos.dispatch_submitted",
        )

    async def approve_dispatch(
        self,
        sos_id: str,
        *,
        reviewer_id: int,
        approval_reference: str,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_approve_dispatch(
            user_id=reviewer_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "Dispatch approval denied"
            )

        if (
            sos.status
            != SOSStatus.PENDING_DISPATCH_APPROVAL
        ):
            raise SOSConflictError(
                "SOS is not awaiting dispatch approval"
            )

        if reviewer_id == sos.created_by:
            raise SOSAuthorizationError(
                "Independent dispatch approval required"
            )

        approval_reference = require_text(
            approval_reference,
            "Approval reference",
        )

        updated = replace(
            sos,
            status=SOSStatus.DISPATCH_APPROVED,
            dispatch_approved_by=reviewer_id,
            dispatch_approval_reference=approval_reference,
            events=[
                *sos.events,
                self._event(
                    SOSEventType.DISPATCH_APPROVED,
                    actor_id=reviewer_id,
                ),
            ],
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=reviewer_id,
            action="sos.dispatch_approved",
        )

    async def request_rescue_coordination(
        self,
        sos_id: str,
        *,
        actor_id: int,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "Rescue coordination denied"
            )

        if sos.status != SOSStatus.DISPATCH_APPROVED:
            raise SOSConflictError(
                "Dispatch approval is required"
            )

        if not sos.dispatch_approval_reference:
            raise SOSDispatchError(
                "Dispatch approval reference missing"
            )

        mission_id = (
            await self.rescue.request_rescue_coordination(
                sos_id=sos.id,
                district_id=sos.district_id,
                emergency_type=sos.emergency_type,
                priority=sos.priority,
                approval_reference=(
                    sos.dispatch_approval_reference
                ),
            )
        )

        updated = replace(
            sos,
            status=SOSStatus.DISPATCH_REQUESTED,
            rescue_mission_id=mission_id,
            dispatch_requested_at=utc_now(),
            events=[
                *sos.events,
                self._event(
                    SOSEventType.DISPATCH_REQUESTED,
                    actor_id=actor_id,
                ),
            ],
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=actor_id,
            action="sos.rescue_coordination_requested",
        )

    async def update_responder_status(
        self,
        sos_id: str,
        *,
        actor_id: int,
        new_status: SOSStatus,
        evidence_reference: str,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "Responder status update denied"
            )

        transitions = {
            SOSStatus.DISPATCH_REQUESTED: {
                SOSStatus.RESPONDER_ACKNOWLEDGED,
            },
            SOSStatus.RESPONDER_ACKNOWLEDGED: {
                SOSStatus.RESPONDER_EN_ROUTE,
            },
            SOSStatus.RESPONDER_EN_ROUTE: {
                SOSStatus.RESPONDER_ON_SCENE,
            },
            SOSStatus.RESPONDER_ON_SCENE: {
                SOSStatus.RESCUE_IN_PROGRESS,
            },
        }

        allowed = transitions.get(
            sos.status,
            set(),
        )

        if new_status not in allowed:
            raise SOSConflictError(
                "Invalid responder status transition"
            )

        evidence_reference = require_text(
            evidence_reference,
            "Evidence reference",
        )

        event_mapping = {
            SOSStatus.RESPONDER_ACKNOWLEDGED: (
                SOSEventType.RESPONDER_ACKNOWLEDGED
            ),
            SOSStatus.RESPONDER_EN_ROUTE: (
                SOSEventType.RESPONDER_EN_ROUTE
            ),
            SOSStatus.RESPONDER_ON_SCENE: (
                SOSEventType.RESPONDER_ON_SCENE
            ),
            SOSStatus.RESCUE_IN_PROGRESS: (
                SOSEventType.RESCUE_STARTED
            ),
        }

        updated = replace(
            sos,
            status=new_status,
            events=[
                *sos.events,
                self._event(
                    event_mapping[new_status],
                    actor_id=actor_id,
                    description=(
                        "Verified field status update"
                    ),
                ),
            ],
            metadata={
                **sos.metadata,
                "latest_responder_evidence": (
                    evidence_reference
                ),
            },
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=actor_id,
            action="sos.responder_status_updated",
        )

    async def resolve_sos(
        self,
        sos_id: str,
        *,
        reviewer_id: int,
        resolution_notes: str,
        evidence_reference: str,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_verify(
            user_id=reviewer_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "SOS resolution denied"
            )

        if sos.status not in {
            SOSStatus.RESPONDER_ON_SCENE,
            SOSStatus.RESCUE_IN_PROGRESS,
        }:
            raise SOSConflictError(
                "SOS cannot be resolved in current state"
            )

        resolution_notes = require_text(
            resolution_notes,
            "Resolution notes",
        )

        evidence_reference = require_text(
            evidence_reference,
            "Resolution evidence",
        )

        updated = replace(
            sos,
            status=SOSStatus.RESOLVED,
            resolved_at=utc_now(),
            resolution_notes=resolution_notes,
            metadata={
                **sos.metadata,
                "resolution_evidence": evidence_reference,
            },
            events=[
                *sos.events,
                self._event(
                    SOSEventType.RESOLVED,
                    actor_id=reviewer_id,
                ),
            ],
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=reviewer_id,
            action="sos.resolved",
        )

    async def cancel_sos(
        self,
        sos_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "SOS cancellation denied"
            )

        if sos.status in self.TERMINAL_STATUSES:
            raise SOSConflictError(
                "SOS is already terminal"
            )

        if sos.status in {
            SOSStatus.DISPATCH_REQUESTED,
            SOSStatus.RESPONDER_ACKNOWLEDGED,
            SOSStatus.RESPONDER_EN_ROUTE,
            SOSStatus.RESPONDER_ON_SCENE,
            SOSStatus.RESCUE_IN_PROGRESS,
        }:
            raise SOSConflictError(
                "Active rescue coordination requires "
                "authorized stand-down confirmation"
            )

        reason = require_text(
            reason,
            "Cancellation reason",
        )

        updated = replace(
            sos,
            status=SOSStatus.CANCELLED,
            cancelled_at=utc_now(),
            resolution_notes=reason,
            events=[
                *sos.events,
                self._event(
                    SOSEventType.CANCELLED,
                    actor_id=actor_id,
                    description=reason,
                ),
            ],
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=actor_id,
            action="sos.cancelled",
        )

    async def mark_false_alarm(
        self,
        sos_id: str,
        *,
        reviewer_id: int,
        evidence_reference: str,
        reason: str,
    ) -> SOSRequest:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_verify(
            user_id=reviewer_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "False alarm verification denied"
            )

        if sos.status in self.TERMINAL_STATUSES:
            raise SOSConflictError(
                "SOS is already terminal"
            )

        if sos.status in {
            SOSStatus.DISPATCH_REQUESTED,
            SOSStatus.RESPONDER_ACKNOWLEDGED,
            SOSStatus.RESPONDER_EN_ROUTE,
            SOSStatus.RESPONDER_ON_SCENE,
            SOSStatus.RESCUE_IN_PROGRESS,
        }:
            raise SOSConflictError(
                "Active responder workflow requires "
                "stand-down confirmation"
            )

        evidence_reference = require_text(
            evidence_reference,
            "Evidence reference",
        )

        reason = require_text(
            reason,
            "False alarm reason",
        )

        updated = replace(
            sos,
            status=SOSStatus.FALSE_ALARM,
            resolution_notes=reason,
            events=[
                *sos.events,
                self._event(
                    SOSEventType.FALSE_ALARM,
                    actor_id=reviewer_id,
                    description=reason,
                ),
            ],
            metadata={
                **sos.metadata,
                "false_alarm_evidence": (
                    evidence_reference
                ),
            },
            updated_at=utc_now(),
            version=sos.version + 1,
        )

        return await self._save(
            sos,
            updated,
            actor_id=reviewer_id,
            action="sos.false_alarm_verified",
        )

    async def request_contact_notification(
        self,
        sos_id: str,
        *,
        actor_id: int,
        channel: SOSNotificationChannel,
        message: str,
    ) -> str:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "Notification permission denied"
            )

        if self.notifications is None:
            raise SOSIntegrationError(
                "Notification gateway is not configured"
            )

        if not sos.contact.notification_consent:
            raise SOSAuthorizationError(
                "Notification consent is missing"
            )

        if not sos.contact.contact_reference:
            raise SOSValidationError(
                "Contact reference is unavailable"
            )

        message = require_text(
            message,
            "Notification message",
            max_length=1000,
        )

        reference = (
            await self.notifications.request_notification(
                sos_id=sos.id,
                contact_reference=(
                    sos.contact.contact_reference
                ),
                channel=channel,
                message=message,
                idempotency_key=(
                    f"sos:{sos.id}:"
                    f"{sos.status.value}:"
                    f"{channel.value}"
                ),
            )
        )

        await self._audit(
            "sos.notification_requested",
            sos,
            actor_id,
        )

        return reference

    async def list_district_sos(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> Sequence[SOSRequest]:

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise SOSAuthorizationError(
                "District SOS access denied"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_incident_sos(
        self,
        incident_id: str,
        *,
        actor_id: int,
    ) -> list[SOSRequest]:

        require_text(
            incident_id,
            "Incident ID",
        )

        requests = await self.repository.list_by_incident(
            incident_id
        )

        visible: list[SOSRequest] = []

        for sos in requests:
            if await self.authorization.can_view(
                user_id=actor_id,
                district_id=sos.district_id,
            ):
                visible.append(sos)

        return visible

    async def get_priority_queue(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> list[SOSRequest]:

        requests = await self.list_district_sos(
            district_id,
            actor_id=actor_id,
        )

        active = [
            sos
            for sos in requests
            if sos.status not in self.TERMINAL_STATUSES
        ]

        priority_order = {
            SOSPriority.CRITICAL: 0,
            SOSPriority.HIGH: 1,
            SOSPriority.MEDIUM: 2,
            SOSPriority.LOW: 3,
        }

        active.sort(
            key=lambda sos: (
                priority_order[sos.priority],
                sos.created_at,
            )
        )

        return active

    async def get_district_statistics(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        requests = await self.list_district_sos(
            district_id,
            actor_id=actor_id,
        )

        status_counts = {
            status.value: 0
            for status in SOSStatus
        }

        priority_counts = {
            priority.value: 0
            for priority in SOSPriority
        }

        emergency_counts = {
            emergency.value: 0
            for emergency in SOSEmergencyType
        }

        for sos in requests:
            status_counts[
                sos.status.value
            ] += 1

            priority_counts[
                sos.priority.value
            ] += 1

            emergency_counts[
                sos.emergency_type.value
            ] += 1

        active = [
            sos
            for sos in requests
            if sos.status not in self.TERMINAL_STATUSES
        ]

        return {
            "district_id": district_id,
            "total_sos_requests": len(requests),
            "active_sos_requests": len(active),
            "critical_active_requests": sum(
                sos.priority == SOSPriority.CRITICAL
                for sos in active
            ),
            "resolved_requests": status_counts[
                SOSStatus.RESOLVED.value
            ],
            "possible_duplicates": sum(
                sos.duplicate_of is not None
                for sos in requests
            ),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "emergency_type_counts": emergency_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_sos_report(
        self,
        sos_id: str,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        sos = await self.get_sos(
            sos_id
        )

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=sos.district_id,
        ):
            raise SOSAuthorizationError(
                "SOS report access denied"
            )

        # Contact references are intentionally omitted.
        # Exact coordinates should be exposed only through
        # separately authorized operational endpoints.

        return {
            "sos_id": sos.id,
            "district_id": sos.district_id,
            "incident_id": sos.incident_id,
            "emergency_type": sos.emergency_type.value,
            "source": sos.source.value,
            "status": sos.status.value,
            "priority": sos.priority.value,
            "people": {
                "affected": sos.people.people_affected,
                "children": sos.people.children,
                "elderly": sos.people.elderly,
                "disabled": sos.people.disabled_people,
                "injured": sos.people.injured_people,
                "critical_patients": (
                    sos.people.critical_patients
                ),
                "missing": sos.people.missing_people,
            },
            "triage": (
                {
                    "recommended_priority": (
                        sos.triage.recommended_priority.value
                    ),
                    "risk_score": sos.triage.risk_score,
                    "reasons": list(
                        sos.triage.reasons
                    ),
                    "medical_review_required": (
                        sos.triage.medical_review_required
                    ),
                    "rescue_review_required": (
                        sos.triage.rescue_review_required
                    ),
                }
                if sos.triage
                else None
            ),
            "verified": sos.verification is not None,
            "duplicate_of": sos.duplicate_of,
            "escalation_level": (
                sos.escalation_level.value
            ),
            "rescue_mission_id": (
                sos.rescue_mission_id
            ),
            "dispatch_approval_reference": (
                sos.dispatch_approval_reference
            ),
            "events": [
                {
                    "event_type": event.event_type.value,
                    "occurred_at": (
                        event.occurred_at.isoformat()
                    ),
                }
                for event in sos.events
            ],
            "created_at": sos.created_at.isoformat(),
            "resolved_at": (
                sos.resolved_at.isoformat()
                if sos.resolved_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "SOS registration and coordination records "
                "do not confirm emergency-service dispatch "
                "or response. All operational outcomes "
                "require authorized field verification."
            ),
        }
