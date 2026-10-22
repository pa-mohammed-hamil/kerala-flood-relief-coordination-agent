from __future__ import annotations

"""
backend/services/volunteer_service.py

Kerala Flood Relief Coordination Agent
=======================================

Volunteer Registration, Verification, Assignment,
Safety and Coordination Service.

Features:
- Volunteer registration and profile management
- Identity and skill verification
- Volunteer availability and duty status
- District-based volunteer discovery
- Skills-based task matching
- Medical and rescue qualification checks
- Human-approved volunteer assignments
- Shift scheduling and conflict prevention
- Check-in, check-out and task completion
- Volunteer safety and emergency escalation
- Volunteer workload and fatigue monitoring
- Assignment cancellation and reassignment
- Volunteer performance and activity reports
- Role-based access control
- Audit logging
- Repository abstraction
- Optimistic concurrency

IMPORTANT:
- Volunteers must not be dispatched automatically.
- Rescue and medical duties require verified credentials.
- Assignment approval does not confirm physical arrival.
- Unsafe or unverified routes must not be used.
- Emergency work must follow official coordination.
- Volunteer personal data must be protected.
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

class VolunteerStatus(str, Enum):
    REGISTERED = "registered"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    ACTIVE = "active"
    ON_DUTY = "on_duty"
    UNAVAILABLE = "unavailable"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"


class VolunteerSkill(str, Enum):
    GENERAL_SUPPORT = "general_support"
    RESCUE = "rescue"
    SWIMMING = "swimming"
    BOAT_OPERATION = "boat_operation"
    FIRST_AID = "first_aid"
    PARAMEDIC = "paramedic"
    DOCTOR = "doctor"
    NURSE = "nurse"
    AMBULANCE_DRIVER = "ambulance_driver"
    VEHICLE_DRIVER = "vehicle_driver"
    LOGISTICS = "logistics"
    INVENTORY = "inventory"
    FOOD_DISTRIBUTION = "food_distribution"
    SHELTER_MANAGEMENT = "shelter_management"
    COMMUNICATION = "communication"
    TRANSLATION = "translation"
    DATA_ENTRY = "data_entry"
    GIS_MAPPING = "gis_mapping"
    DRONE_OPERATION = "drone_operation"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    CONSTRUCTION = "construction"
    COUNSELLING = "counselling"
    ANIMAL_RESCUE = "animal_rescue"


class VolunteerTaskType(str, Enum):
    RELIEF_DISTRIBUTION = "relief_distribution"
    FOOD_DISTRIBUTION = "food_distribution"
    WATER_DISTRIBUTION = "water_distribution"
    SHELTER_SUPPORT = "shelter_support"
    INVENTORY_SUPPORT = "inventory_support"
    LOGISTICS_SUPPORT = "logistics_support"
    COMMUNICATION_SUPPORT = "communication_support"
    DATA_COLLECTION = "data_collection"
    FIELD_ASSESSMENT = "field_assessment"
    MISSING_PERSON_SUPPORT = "missing_person_support"
    MEDICAL_SUPPORT = "medical_support"
    FIRST_AID = "first_aid"
    AMBULANCE_SUPPORT = "ambulance_support"
    WATER_RESCUE = "water_rescue"
    BOAT_RESCUE = "boat_rescue"
    EVACUATION_SUPPORT = "evacuation_support"
    CLEANUP = "cleanup"
    RECOVERY_SUPPORT = "recovery_support"


class VolunteerTaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VolunteerTaskStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class VolunteerAssignmentStatus(str, Enum):
    RESERVED = "reserved"
    APPROVED = "approved"
    ACCEPTED = "accepted"
    CHECKED_IN = "checked_in"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DECLINED = "declined"
    NO_SHOW = "no_show"


class VolunteerSafetyStatus(str, Enum):
    SAFE = "safe"
    NEEDS_ASSISTANCE = "needs_assistance"
    IN_DANGER = "in_danger"
    UNKNOWN = "unknown"


class VolunteerVerificationType(str, Enum):
    IDENTITY = "identity"
    SKILL = "skill"
    PROFESSIONAL_LICENSE = "professional_license"
    TRAINING = "training"
    ORGANIZATION = "organization"


# ============================================================
# EXCEPTIONS
# ============================================================

class VolunteerServiceError(Exception):
    pass


class VolunteerValidationError(VolunteerServiceError):
    pass


class VolunteerNotFoundError(VolunteerServiceError):
    pass


class VolunteerConflictError(VolunteerServiceError):
    pass


class VolunteerAuthorizationError(VolunteerServiceError):
    pass


class VolunteerVerificationError(VolunteerServiceError):
    pass


class VolunteerAvailabilityError(VolunteerServiceError):
    pass


class VolunteerSafetyError(VolunteerServiceError):
    pass


class VolunteerIntegrationError(VolunteerServiceError):
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
        raise VolunteerValidationError(
            f"{name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise VolunteerValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if not isinstance(value, datetime):
        raise VolunteerValidationError(
            f"{name} must be a datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise VolunteerValidationError(
            f"{name} must be timezone-aware"
        )


def validate_time_window(
    start: datetime,
    end: datetime,
) -> None:
    validate_datetime(start, "start")
    validate_datetime(end, "end")

    if end <= start:
        raise VolunteerValidationError(
            "End time must be after start time"
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

    a = max(0.0, min(1.0, a))

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
class VolunteerLocation:
    district_id: int
    latitude: float
    longitude: float

    panchayat: str = ""
    taluk: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.district_id, bool)
            or not isinstance(self.district_id, int)
            or self.district_id <= 0
        ):
            raise VolunteerValidationError(
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
                raise VolunteerValidationError(
                    f"Invalid {name}"
                )


# ============================================================
# VOLUNTEER MODELS
# ============================================================

@dataclass(frozen=True)
class VolunteerRegistrationRequest:
    full_name: str

    location: VolunteerLocation

    contact_reference: str

    skills: frozenset[VolunteerSkill]

    organization_reference: str | None = None

    emergency_contact_reference: str | None = None

    available_from: datetime | None = None
    available_until: datetime | None = None

    consent_to_contact: bool = False
    consent_to_data_processing: bool = False

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class VolunteerVerification:
    verification_type: VolunteerVerificationType

    verified_by: int
    verified_at: datetime

    evidence_reference: str

    skill: VolunteerSkill | None = None

    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_datetime(
            self.verified_at,
            "verified_at",
        )

        require_text(
            self.evidence_reference,
            "evidence_reference",
        )

        if self.expires_at is not None:
            validate_datetime(
                self.expires_at,
                "expires_at",
            )

            if self.expires_at <= self.verified_at:
                raise VolunteerValidationError(
                    "Verification expiry must be later "
                    "than verification date"
                )


@dataclass
class Volunteer:
    id: str

    district_id: int

    full_name: str

    location: VolunteerLocation

    contact_reference: str

    skills: frozenset[VolunteerSkill]

    status: VolunteerStatus

    created_by: int | None = None

    organization_reference: str | None = None
    emergency_contact_reference: str | None = None

    available_from: datetime | None = None
    available_until: datetime | None = None

    consent_to_contact: bool = False
    consent_to_data_processing: bool = False

    identity_verified: bool = False

    verified_skills: frozenset[
        VolunteerSkill
    ] = field(
        default_factory=frozenset
    )

    verifications: list[
        VolunteerVerification
    ] = field(
        default_factory=list
    )

    safety_status: VolunteerSafetyStatus = (
        VolunteerSafetyStatus.UNKNOWN
    )

    total_completed_tasks: int = 0

    total_service_minutes: int = 0

    last_check_in_at: datetime | None = None
    last_check_out_at: datetime | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# TASK MODELS
# ============================================================

@dataclass(frozen=True)
class VolunteerTaskCreateRequest:
    district_id: int

    task_type: VolunteerTaskType

    title: str
    description: str

    location: VolunteerLocation

    priority: VolunteerTaskPriority

    required_skills: frozenset[
        VolunteerSkill
    ]

    required_volunteers: int

    start_time: datetime
    end_time: datetime

    incident_id: str | None = None
    shelter_id: str | None = None
    rescue_mission_id: str | None = None

    route_reference: str | None = None

    safety_briefing_reference: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.district_id <= 0:
            raise VolunteerValidationError(
                "Invalid district ID"
            )

        if (
            isinstance(self.required_volunteers, bool)
            or not isinstance(self.required_volunteers, int)
            or self.required_volunteers <= 0
        ):
            raise VolunteerValidationError(
                "required_volunteers must be positive"
            )

        validate_time_window(
            self.start_time,
            self.end_time,
        )

        if self.location.district_id != self.district_id:
            raise VolunteerValidationError(
                "Task location district mismatch"
            )


@dataclass
class VolunteerTask:
    id: str

    district_id: int

    task_type: VolunteerTaskType

    title: str
    description: str

    location: VolunteerLocation

    priority: VolunteerTaskPriority

    required_skills: frozenset[
        VolunteerSkill
    ]

    required_volunteers: int

    start_time: datetime
    end_time: datetime

    status: VolunteerTaskStatus

    created_by: int

    incident_id: str | None = None
    shelter_id: str | None = None
    rescue_mission_id: str | None = None

    route_reference: str | None = None
    safety_briefing_reference: str | None = None

    approved_by: int | None = None
    approved_at: datetime | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class VolunteerAssignment:
    id: str

    volunteer_id: str
    task_id: str

    district_id: int

    status: VolunteerAssignmentStatus

    reserved_at: datetime

    approved_by: int | None = None
    approved_at: datetime | None = None

    accepted_at: datetime | None = None

    checked_in_at: datetime | None = None
    checked_out_at: datetime | None = None

    completed_at: datetime | None = None

    completion_notes: str | None = None
    completion_evidence_reference: str | None = None

    cancellation_reason: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    version: int = 1


@dataclass(frozen=True)
class VolunteerMatch:
    volunteer_id: str

    verified_skill_match: bool

    available: bool

    route_verified: bool

    distance_km: float

    suitability_score: float

    eligible: bool

    reasons: tuple[str, ...] = ()


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class VolunteerRepository(Protocol):

    async def create(
        self,
        volunteer: Volunteer,
    ) -> Volunteer:
        ...

    async def get_by_id(
        self,
        volunteer_id: str,
    ) -> Volunteer | None:
        ...

    async def update(
        self,
        volunteer: Volunteer,
        *,
        expected_version: int,
    ) -> Volunteer:
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[Volunteer]:
        ...

    async def list_available(
        self,
        district_id: int,
    ) -> Sequence[Volunteer]:
        ...


class VolunteerTaskRepository(Protocol):

    async def create(
        self,
        task: VolunteerTask,
    ) -> VolunteerTask:
        ...

    async def get_by_id(
        self,
        task_id: str,
    ) -> VolunteerTask | None:
        ...

    async def update(
        self,
        task: VolunteerTask,
        *,
        expected_version: int,
    ) -> VolunteerTask:
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[VolunteerTask]:
        ...


class VolunteerAssignmentRepository(Protocol):

    async def reserve(
        self,
        *,
        volunteer_id: str,
        task_id: str,
        district_id: int,
        start_time: datetime,
        end_time: datetime,
        maximum_task_assignments: int,
    ) -> VolunteerAssignment:
        """
        Atomically verify:
        - No overlapping volunteer assignments
        - Task assignment capacity
        - No duplicate active assignment

        Must use durable database constraints.
        """
        ...

    async def get_by_id(
        self,
        assignment_id: str,
    ) -> VolunteerAssignment | None:
        ...

    async def update(
        self,
        assignment: VolunteerAssignment,
        *,
        expected_version: int,
    ) -> VolunteerAssignment:
        ...

    async def list_by_task(
        self,
        task_id: str,
    ) -> Sequence[VolunteerAssignment]:
        ...

    async def list_by_volunteer(
        self,
        volunteer_id: str,
    ) -> Sequence[VolunteerAssignment]:
        ...


# ============================================================
# EXTERNAL GATEWAYS
# ============================================================

class VolunteerAuthorizationGateway(Protocol):

    async def can_register(
        self,
        *,
        user_id: int | None,
        district_id: int,
    ) -> bool:
        ...

    async def can_verify(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_create_task(
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

    async def can_approve(
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

    async def can_view(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class VolunteerRouteGateway(Protocol):

    async def get_verified_route(
        self,
        *,
        origin: VolunteerLocation,
        destination: VolunteerLocation,
    ) -> tuple[float, float] | None:
        """
        Return (distance_km, duration_minutes)
        only for a verified safe route.
        """
        ...


class VolunteerAuditGateway(Protocol):

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


class VolunteerSafetyGateway(Protocol):

    async def request_safety_escalation(
        self,
        *,
        volunteer_id: str,
        assignment_id: str | None,
        district_id: int,
        safety_status: VolunteerSafetyStatus,
        description: str,
        idempotency_key: str,
    ) -> str:
        """
        Create an escalation request.

        A returned reference does not prove that
        responders have been notified or dispatched.
        """
        ...


# ============================================================
# TASK SAFETY POLICY
# ============================================================

class VolunteerSafetyPolicy:

    SPECIALIST_SKILLS = frozenset({
        VolunteerSkill.RESCUE,
        VolunteerSkill.BOAT_OPERATION,
        VolunteerSkill.PARAMEDIC,
        VolunteerSkill.DOCTOR,
        VolunteerSkill.NURSE,
        VolunteerSkill.AMBULANCE_DRIVER,
        VolunteerSkill.DRONE_OPERATION,
        VolunteerSkill.ELECTRICAL,
    })

    SPECIALIST_TASKS = frozenset({
        VolunteerTaskType.WATER_RESCUE,
        VolunteerTaskType.BOAT_RESCUE,
        VolunteerTaskType.MEDICAL_SUPPORT,
        VolunteerTaskType.FIRST_AID,
        VolunteerTaskType.AMBULANCE_SUPPORT,
    })

    TASK_REQUIRED_SKILLS = {
        VolunteerTaskType.WATER_RESCUE: frozenset({
            VolunteerSkill.RESCUE,
        }),
        VolunteerTaskType.BOAT_RESCUE: frozenset({
            VolunteerSkill.BOAT_OPERATION,
            VolunteerSkill.RESCUE,
        }),
        VolunteerTaskType.FIRST_AID: frozenset({
            VolunteerSkill.FIRST_AID,
        }),
        VolunteerTaskType.AMBULANCE_SUPPORT: frozenset({
            VolunteerSkill.AMBULANCE_DRIVER,
        }),
    }

    MAX_SHIFT_DURATION = timedelta(
        hours=8
    )

    MINIMUM_REST = timedelta(
        hours=8
    )

    def validate_task(
        self,
        task: VolunteerTask,
    ) -> None:

        validate_time_window(
            task.start_time,
            task.end_time,
        )

        if (
            task.end_time - task.start_time
            > self.MAX_SHIFT_DURATION
        ):
            raise VolunteerSafetyError(
                "Shift exceeds configured safety limit"
            )

        mandatory = self.TASK_REQUIRED_SKILLS.get(
            task.task_type,
            frozenset(),
        )

        if not mandatory.issubset(
            task.required_skills
        ):
            raise VolunteerSafetyError(
                "Specialist task is missing "
                "mandatory skill requirements"
            )

        if task.task_type in self.SPECIALIST_TASKS:
            if not task.safety_briefing_reference:
                raise VolunteerSafetyError(
                    "Specialist task requires a "
                    "verified safety briefing"
                )

    def has_valid_verification(
        self,
        volunteer: Volunteer,
        skill: VolunteerSkill,
        *,
        now: datetime | None = None,
    ) -> bool:

        now = now or utc_now()

        for verification in volunteer.verifications:
            if verification.skill != skill:
                continue

            if verification.verification_type not in {
                VolunteerVerificationType.SKILL,
                VolunteerVerificationType.PROFESSIONAL_LICENSE,
                VolunteerVerificationType.TRAINING,
            }:
                continue

            if (
                verification.expires_at is not None
                and verification.expires_at <= now
            ):
                continue

            return True

        return False

    def validate_volunteer_for_task(
        self,
        volunteer: Volunteer,
        task: VolunteerTask,
    ) -> None:

        if not volunteer.identity_verified:
            raise VolunteerVerificationError(
                "Volunteer identity is not verified"
            )

        if volunteer.status not in {
            VolunteerStatus.VERIFIED,
            VolunteerStatus.ACTIVE,
        }:
            raise VolunteerAvailabilityError(
                "Volunteer is not available for assignment"
            )

        if volunteer.safety_status in {
            VolunteerSafetyStatus.IN_DANGER,
            VolunteerSafetyStatus.NEEDS_ASSISTANCE,
        }:
            raise VolunteerSafetyError(
                "Volunteer has an active safety concern"
            )

        if not task.required_skills.issubset(
            volunteer.skills
        ):
            raise VolunteerVerificationError(
                "Volunteer lacks required skills"
            )

        for skill in task.required_skills:
            if skill in self.SPECIALIST_SKILLS:
                if not self.has_valid_verification(
                    volunteer,
                    skill,
                ):
                    raise VolunteerVerificationError(
                        f"Specialist skill not verified: "
                        f"{skill.value}"
                    )

        if (
            volunteer.available_from is not None
            and task.start_time < volunteer.available_from
        ):
            raise VolunteerAvailabilityError(
                "Task begins before volunteer availability"
            )

        if (
            volunteer.available_until is not None
            and task.end_time > volunteer.available_until
        ):
            raise VolunteerAvailabilityError(
                "Task exceeds volunteer availability"
            )

        self.validate_task(task)


# ============================================================
# VOLUNTEER MATCHING
# ============================================================

class VolunteerMatchingEngine:

    def __init__(
        self,
        safety_policy: VolunteerSafetyPolicy | None = None,
    ):
        self.safety_policy = (
            safety_policy
            or VolunteerSafetyPolicy()
        )

    def evaluate(
        self,
        *,
        volunteer: Volunteer,
        task: VolunteerTask,
        distance_km: float,
        route_verified: bool,
    ) -> VolunteerMatch:

        reasons: list[str] = []

        available = True
        verified_skill_match = True

        try:
            self.safety_policy.validate_volunteer_for_task(
                volunteer,
                task,
            )

        except VolunteerVerificationError as exc:
            verified_skill_match = False
            reasons.append(str(exc))

        except (
            VolunteerAvailabilityError,
            VolunteerSafetyError,
        ) as exc:
            available = False
            reasons.append(str(exc))

        if not route_verified:
            reasons.append(
                "No verified safe travel route"
            )

        eligible = (
            available
            and verified_skill_match
            and route_verified
        )

        distance_score = (
            1.0
            / (1.0 + distance_km / 10.0)
        )

        score = (
            0.70
            + 0.30 * distance_score
        ) if eligible else 0.0

        return VolunteerMatch(
            volunteer_id=volunteer.id,
            verified_skill_match=verified_skill_match,
            available=available,
            route_verified=route_verified,
            distance_km=round(distance_km, 2),
            suitability_score=round(score, 4),
            eligible=eligible,
            reasons=tuple(reasons),
        )


# ============================================================
# VOLUNTEER SERVICE
# ============================================================

class VolunteerService:

    def __init__(
        self,
        *,
        volunteer_repository: VolunteerRepository,
        task_repository: VolunteerTaskRepository,
        assignment_repository: VolunteerAssignmentRepository,
        authorization_gateway: VolunteerAuthorizationGateway,
        route_gateway: VolunteerRouteGateway,
        audit_gateway: VolunteerAuditGateway | None = None,
        safety_gateway: VolunteerSafetyGateway | None = None,
        safety_policy: VolunteerSafetyPolicy | None = None,
        matching_engine: VolunteerMatchingEngine | None = None,
    ):
        self.volunteers = volunteer_repository
        self.tasks = task_repository
        self.assignments = assignment_repository

        self.authorization = authorization_gateway
        self.routes = route_gateway
        self.audit = audit_gateway
        self.safety_gateway = safety_gateway

        self.safety_policy = (
            safety_policy
            or VolunteerSafetyPolicy()
        )

        self.matching_engine = (
            matching_engine
            or VolunteerMatchingEngine(
                self.safety_policy
            )
        )

    async def _audit(
        self,
        action: str,
        *,
        actor_id: int | None,
        entity_type: str,
        entity_id: str,
        district_id: int,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type=entity_type,
            entity_id=entity_id,
            metadata={
                "district_id": district_id,
            },
        )

    async def get_volunteer(
        self,
        volunteer_id: str,
    ) -> Volunteer:

        volunteer = await self.volunteers.get_by_id(
            volunteer_id
        )

        if volunteer is None:
            raise VolunteerNotFoundError(
                "Volunteer not found"
            )

        return volunteer

    async def get_task(
        self,
        task_id: str,
    ) -> VolunteerTask:

        task = await self.tasks.get_by_id(
            task_id
        )

        if task is None:
            raise VolunteerNotFoundError(
                "Volunteer task not found"
            )

        return task

    async def get_assignment(
        self,
        assignment_id: str,
    ) -> VolunteerAssignment:

        assignment = await self.assignments.get_by_id(
            assignment_id
        )

        if assignment is None:
            raise VolunteerNotFoundError(
                "Volunteer assignment not found"
            )

        return assignment

    async def register_volunteer(
        self,
        request: VolunteerRegistrationRequest,
        *,
        actor_id: int | None = None,
    ) -> Volunteer:

        district_id = request.location.district_id

        if not await self.authorization.can_register(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer registration denied"
            )

        if not request.consent_to_data_processing:
            raise VolunteerValidationError(
                "Data processing consent is required"
            )

        if (
            request.available_from is not None
            and request.available_until is not None
        ):
            validate_time_window(
                request.available_from,
                request.available_until,
            )

        volunteer = Volunteer(
            id=str(uuid.uuid4()),
            district_id=district_id,
            full_name=require_text(
                request.full_name,
                "Volunteer name",
                max_length=200,
            ),
            location=request.location,
            contact_reference=require_text(
                request.contact_reference,
                "Contact reference",
                max_length=500,
            ),
            skills=request.skills,
            status=VolunteerStatus.REGISTERED,
            created_by=actor_id,
            organization_reference=(
                request.organization_reference
            ),
            emergency_contact_reference=(
                request.emergency_contact_reference
            ),
            available_from=request.available_from,
            available_until=request.available_until,
            consent_to_contact=(
                request.consent_to_contact
            ),
            consent_to_data_processing=(
                request.consent_to_data_processing
            ),
            metadata=dict(request.metadata),
        )

        created = await self.volunteers.create(
            volunteer
        )

        await self._audit(
            "volunteer.registered",
            actor_id=actor_id,
            entity_type="volunteer",
            entity_id=created.id,
            district_id=district_id,
        )

        return created

    async def submit_for_verification(
        self,
        volunteer_id: str,
        *,
        actor_id: int,
    ) -> Volunteer:

        volunteer = await self.get_volunteer(
            volunteer_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=volunteer.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer verification submission denied"
            )

        if volunteer.status != VolunteerStatus.REGISTERED:
            raise VolunteerConflictError(
                "Volunteer is not awaiting submission"
            )

        updated = replace(
            volunteer,
            status=VolunteerStatus.PENDING_VERIFICATION,
            updated_at=utc_now(),
            version=volunteer.version + 1,
        )

        saved = await self.volunteers.update(
            updated,
            expected_version=volunteer.version,
        )

        await self._audit(
            "volunteer.verification_requested",
            actor_id=actor_id,
            entity_type="volunteer",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def verify_volunteer(
        self,
        volunteer_id: str,
        *,
        reviewer_id: int,
        verifications: Sequence[VolunteerVerification],
    ) -> Volunteer:

        volunteer = await self.get_volunteer(
            volunteer_id
        )

        if not await self.authorization.can_verify(
            user_id=reviewer_id,
            district_id=volunteer.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer verification denied"
            )

        if (
            volunteer.status
            != VolunteerStatus.PENDING_VERIFICATION
        ):
            raise VolunteerConflictError(
                "Volunteer is not awaiting verification"
            )

        if (
            volunteer.created_by is not None
            and reviewer_id == volunteer.created_by
        ):
            raise VolunteerAuthorizationError(
                "Independent verification required"
            )

        if not verifications:
            raise VolunteerVerificationError(
                "Verification evidence is required"
            )

        now = utc_now()

        for verification in verifications:
            if verification.verified_by != reviewer_id:
                raise VolunteerVerificationError(
                    "Verification reviewer mismatch"
                )

            if verification.verified_at > now:
                raise VolunteerVerificationError(
                    "Verification timestamp is in the future"
                )

        identity_verified = any(
            verification.verification_type
            == VolunteerVerificationType.IDENTITY
            and (
                verification.expires_at is None
                or verification.expires_at > now
            )
            for verification in verifications
        )

        if not identity_verified:
            raise VolunteerVerificationError(
                "Valid identity verification is required"
            )

        verified_skills = frozenset(
            verification.skill
            for verification in verifications
            if verification.skill is not None
            and verification.verification_type in {
                VolunteerVerificationType.SKILL,
                VolunteerVerificationType.TRAINING,
                VolunteerVerificationType.PROFESSIONAL_LICENSE,
            }
            and (
                verification.expires_at is None
                or verification.expires_at > now
            )
            and verification.skill in volunteer.skills
        )

        updated = replace(
            volunteer,
            status=VolunteerStatus.VERIFIED,
            identity_verified=True,
            verified_skills=verified_skills,
            verifications=list(verifications),
            updated_at=now,
            version=volunteer.version + 1,
        )

        saved = await self.volunteers.update(
            updated,
            expected_version=volunteer.version,
        )

        await self._audit(
            "volunteer.verified",
            actor_id=reviewer_id,
            entity_type="volunteer",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def activate_volunteer(
        self,
        volunteer_id: str,
        *,
        actor_id: int,
    ) -> Volunteer:

        volunteer = await self.get_volunteer(
            volunteer_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=volunteer.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer activation denied"
            )

        if volunteer.status not in {
            VolunteerStatus.VERIFIED,
            VolunteerStatus.UNAVAILABLE,
            VolunteerStatus.INACTIVE,
        }:
            raise VolunteerConflictError(
                "Volunteer cannot be activated"
            )

        if not volunteer.identity_verified:
            raise VolunteerVerificationError(
                "Volunteer identity is not verified"
            )

        updated = replace(
            volunteer,
            status=VolunteerStatus.ACTIVE,
            updated_at=utc_now(),
            version=volunteer.version + 1,
        )

        saved = await self.volunteers.update(
            updated,
            expected_version=volunteer.version,
        )

        await self._audit(
            "volunteer.activated",
            actor_id=actor_id,
            entity_type="volunteer",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def create_task(
        self,
        request: VolunteerTaskCreateRequest,
        *,
        actor_id: int,
    ) -> VolunteerTask:

        if not await self.authorization.can_create_task(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer task creation denied"
            )

        task = VolunteerTask(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            task_type=request.task_type,
            title=require_text(
                request.title,
                "Task title",
                max_length=200,
            ),
            description=require_text(
                request.description,
                "Task description",
            ),
            location=request.location,
            priority=request.priority,
            required_skills=request.required_skills,
            required_volunteers=request.required_volunteers,
            start_time=request.start_time,
            end_time=request.end_time,
            status=VolunteerTaskStatus.DRAFT,
            created_by=actor_id,
            incident_id=request.incident_id,
            shelter_id=request.shelter_id,
            rescue_mission_id=request.rescue_mission_id,
            route_reference=request.route_reference,
            safety_briefing_reference=(
                request.safety_briefing_reference
            ),
            metadata=dict(request.metadata),
        )

        self.safety_policy.validate_task(
            task
        )

        created = await self.tasks.create(
            task
        )

        await self._audit(
            "volunteer_task.created",
            actor_id=actor_id,
            entity_type="volunteer_task",
            entity_id=created.id,
            district_id=created.district_id,
        )

        return created

    async def approve_task(
        self,
        task_id: str,
        *,
        reviewer_id: int,
    ) -> VolunteerTask:

        task = await self.get_task(
            task_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=task.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Task approval denied"
            )

        if task.status not in {
            VolunteerTaskStatus.DRAFT,
            VolunteerTaskStatus.PENDING_APPROVAL,
        }:
            raise VolunteerConflictError(
                "Task cannot be approved"
            )

        if reviewer_id == task.created_by:
            raise VolunteerAuthorizationError(
                "Independent task approval required"
            )

        self.safety_policy.validate_task(
            task
        )

        updated = replace(
            task,
            status=VolunteerTaskStatus.OPEN,
            approved_by=reviewer_id,
            approved_at=utc_now(),
            updated_at=utc_now(),
            version=task.version + 1,
        )

        saved = await self.tasks.update(
            updated,
            expected_version=task.version,
        )

        await self._audit(
            "volunteer_task.approved",
            actor_id=reviewer_id,
            entity_type="volunteer_task",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def recommend_volunteers(
        self,
        task_id: str,
        *,
        actor_id: int,
    ) -> list[VolunteerMatch]:

        task = await self.get_task(
            task_id
        )

        if not await self.authorization.can_assign(
            user_id=actor_id,
            district_id=task.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer matching denied"
            )

        if task.status != VolunteerTaskStatus.OPEN:
            raise VolunteerConflictError(
                "Task is not open"
            )

        volunteers = await self.volunteers.list_available(
            task.district_id
        )

        recommendations: list[
            VolunteerMatch
        ] = []

        for volunteer in volunteers:
            route = await self.routes.get_verified_route(
                origin=volunteer.location,
                destination=task.location,
            )

            route_verified = route is not None

            if route is not None:
                distance_km, duration_minutes = route

                if (
                    not math.isfinite(distance_km)
                    or distance_km < 0
                    or not math.isfinite(duration_minutes)
                    or duration_minutes < 0
                ):
                    raise VolunteerIntegrationError(
                        "Route provider returned invalid values"
                    )

            else:
                distance_km = haversine_distance_km(
                    volunteer.location.latitude,
                    volunteer.location.longitude,
                    task.location.latitude,
                    task.location.longitude,
                )

            recommendation = (
                self.matching_engine.evaluate(
                    volunteer=volunteer,
                    task=task,
                    distance_km=distance_km,
                    route_verified=route_verified,
                )
            )

            recommendations.append(
                recommendation
            )

        recommendations.sort(
            key=lambda item: (
                not item.eligible,
                -item.suitability_score,
                item.distance_km,
            )
        )

        return recommendations

    async def reserve_assignment(
        self,
        task_id: str,
        volunteer_id: str,
        *,
        actor_id: int,
    ) -> VolunteerAssignment:

        task = await self.get_task(
            task_id
        )

        volunteer = await self.get_volunteer(
            volunteer_id
        )

        if not await self.authorization.can_assign(
            user_id=actor_id,
            district_id=task.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer assignment denied"
            )

        if task.status != VolunteerTaskStatus.OPEN:
            raise VolunteerConflictError(
                "Task is not open"
            )

        if volunteer.district_id != task.district_id:
            raise VolunteerValidationError(
                "Cross-district assignment requires "
                "a separate authorization workflow"
            )

        self.safety_policy.validate_volunteer_for_task(
            volunteer,
            task,
        )

        route = await self.routes.get_verified_route(
            origin=volunteer.location,
            destination=task.location,
        )

        if route is None:
            raise VolunteerSafetyError(
                "No verified safe route to task location"
            )

        assignment = await self.assignments.reserve(
            volunteer_id=volunteer.id,
            task_id=task.id,
            district_id=task.district_id,
            start_time=task.start_time,
            end_time=task.end_time,
            maximum_task_assignments=(
                task.required_volunteers
            ),
        )

        await self._audit(
            "volunteer_assignment.reserved",
            actor_id=actor_id,
            entity_type="volunteer_assignment",
            entity_id=assignment.id,
            district_id=assignment.district_id,
        )

        return assignment

    async def approve_assignment(
        self,
        assignment_id: str,
        *,
        reviewer_id: int,
    ) -> VolunteerAssignment:

        assignment = await self.get_assignment(
            assignment_id
        )

        task = await self.get_task(
            assignment.task_id
        )

        volunteer = await self.get_volunteer(
            assignment.volunteer_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=assignment.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Assignment approval denied"
            )

        if (
            assignment.status
            != VolunteerAssignmentStatus.RESERVED
        ):
            raise VolunteerConflictError(
                "Assignment is not reserved"
            )

        if reviewer_id == task.created_by:
            raise VolunteerAuthorizationError(
                "Independent assignment approval required"
            )

        self.safety_policy.validate_volunteer_for_task(
            volunteer,
            task,
        )

        route = await self.routes.get_verified_route(
            origin=volunteer.location,
            destination=task.location,
        )

        if route is None:
            raise VolunteerSafetyError(
                "Verified safe route is unavailable"
            )

        now = utc_now()

        updated = replace(
            assignment,
            status=VolunteerAssignmentStatus.APPROVED,
            approved_by=reviewer_id,
            approved_at=now,
            updated_at=now,
            version=assignment.version + 1,
        )

        saved = await self.assignments.update(
            updated,
            expected_version=assignment.version,
        )

        await self._audit(
            "volunteer_assignment.approved",
            actor_id=reviewer_id,
            entity_type="volunteer_assignment",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def accept_assignment(
        self,
        assignment_id: str,
        *,
        actor_id: int,
    ) -> VolunteerAssignment:

        assignment = await self.get_assignment(
            assignment_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=assignment.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Assignment acceptance denied"
            )

        if (
            assignment.status
            != VolunteerAssignmentStatus.APPROVED
        ):
            raise VolunteerConflictError(
                "Assignment is not approved"
            )

        updated = replace(
            assignment,
            status=VolunteerAssignmentStatus.ACCEPTED,
            accepted_at=utc_now(),
            updated_at=utc_now(),
            version=assignment.version + 1,
        )

        saved = await self.assignments.update(
            updated,
            expected_version=assignment.version,
        )

        await self._audit(
            "volunteer_assignment.accepted",
            actor_id=actor_id,
            entity_type="volunteer_assignment",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def check_in_volunteer(
        self,
        assignment_id: str,
        *,
        actor_id: int,
        safety_briefing_confirmed: bool,
    ) -> VolunteerAssignment:

        assignment = await self.get_assignment(
            assignment_id
        )

        volunteer = await self.get_volunteer(
            assignment.volunteer_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=assignment.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer check-in denied"
            )

        if (
            assignment.status
            != VolunteerAssignmentStatus.ACCEPTED
        ):
            raise VolunteerConflictError(
                "Assignment is not accepted"
            )

        if not safety_briefing_confirmed:
            raise VolunteerSafetyError(
                "Safety briefing confirmation required"
            )

        if volunteer.safety_status in {
            VolunteerSafetyStatus.IN_DANGER,
            VolunteerSafetyStatus.NEEDS_ASSISTANCE,
        }:
            raise VolunteerSafetyError(
                "Volunteer has unresolved safety concerns"
            )

        now = utc_now()

        updated = replace(
            assignment,
            status=VolunteerAssignmentStatus.CHECKED_IN,
            checked_in_at=now,
            updated_at=now,
            version=assignment.version + 1,
        )

        saved = await self.assignments.update(
            updated,
            expected_version=assignment.version,
        )

        # Production implementation should atomically
        # synchronize volunteer duty state and assignment.
        volunteer_updated = replace(
            volunteer,
            status=VolunteerStatus.ON_DUTY,
            safety_status=VolunteerSafetyStatus.SAFE,
            last_check_in_at=now,
            updated_at=now,
            version=volunteer.version + 1,
        )

        await self.volunteers.update(
            volunteer_updated,
            expected_version=volunteer.version,
        )

        await self._audit(
            "volunteer.checked_in",
            actor_id=actor_id,
            entity_type="volunteer_assignment",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def complete_assignment(
        self,
        assignment_id: str,
        *,
        actor_id: int,
        completion_notes: str,
        evidence_reference: str,
    ) -> VolunteerAssignment:

        assignment = await self.get_assignment(
            assignment_id
        )

        volunteer = await self.get_volunteer(
            assignment.volunteer_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=assignment.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Assignment completion denied"
            )

        if (
            assignment.status
            != VolunteerAssignmentStatus.CHECKED_IN
        ):
            raise VolunteerConflictError(
                "Assignment is not in progress"
            )

        completion_notes = require_text(
            completion_notes,
            "Completion notes",
        )

        evidence_reference = require_text(
            evidence_reference,
            "Completion evidence",
        )

        now = utc_now()

        service_minutes = 0

        if assignment.checked_in_at:
            service_minutes = max(
                0,
                int(
                    (
                        now - assignment.checked_in_at
                    ).total_seconds()
                    // 60
                ),
            )

        updated = replace(
            assignment,
            status=VolunteerAssignmentStatus.COMPLETED,
            completed_at=now,
            checked_out_at=now,
            completion_notes=completion_notes,
            completion_evidence_reference=(
                evidence_reference
            ),
            updated_at=now,
            version=assignment.version + 1,
        )

        saved = await self.assignments.update(
            updated,
            expected_version=assignment.version,
        )

        volunteer_updated = replace(
            volunteer,
            status=VolunteerStatus.ACTIVE,
            total_completed_tasks=(
                volunteer.total_completed_tasks + 1
            ),
            total_service_minutes=(
                volunteer.total_service_minutes
                + service_minutes
            ),
            last_check_out_at=now,
            updated_at=now,
            version=volunteer.version + 1,
        )

        await self.volunteers.update(
            volunteer_updated,
            expected_version=volunteer.version,
        )

        await self._audit(
            "volunteer_assignment.completed",
            actor_id=actor_id,
            entity_type="volunteer_assignment",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def cancel_assignment(
        self,
        assignment_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> VolunteerAssignment:

        assignment = await self.get_assignment(
            assignment_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=assignment.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Assignment cancellation denied"
            )

        if assignment.status in {
            VolunteerAssignmentStatus.COMPLETED,
            VolunteerAssignmentStatus.CANCELLED,
            VolunteerAssignmentStatus.CHECKED_IN,
        }:
            raise VolunteerConflictError(
                "Assignment requires a different "
                "completion or stand-down workflow"
            )

        updated = replace(
            assignment,
            status=VolunteerAssignmentStatus.CANCELLED,
            cancellation_reason=require_text(
                reason,
                "Cancellation reason",
            ),
            updated_at=utc_now(),
            version=assignment.version + 1,
        )

        saved = await self.assignments.update(
            updated,
            expected_version=assignment.version,
        )

        await self._audit(
            "volunteer_assignment.cancelled",
            actor_id=actor_id,
            entity_type="volunteer_assignment",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def report_volunteer_safety(
        self,
        volunteer_id: str,
        *,
        actor_id: int,
        safety_status: VolunteerSafetyStatus,
        description: str,
        assignment_id: str | None = None,
    ) -> Volunteer:

        volunteer = await self.get_volunteer(
            volunteer_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=volunteer.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer safety update denied"
            )

        description = require_text(
            description,
            "Safety report description",
        )

        if assignment_id is not None:
            assignment = await self.get_assignment(
                assignment_id
            )

            if assignment.volunteer_id != volunteer.id:
                raise VolunteerValidationError(
                    "Assignment does not belong to volunteer"
                )

        updated = replace(
            volunteer,
            safety_status=safety_status,
            updated_at=utc_now(),
            version=volunteer.version + 1,
        )

        saved = await self.volunteers.update(
            updated,
            expected_version=volunteer.version,
        )

        await self._audit(
            "volunteer.safety_reported",
            actor_id=actor_id,
            entity_type="volunteer",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        if safety_status in {
            VolunteerSafetyStatus.NEEDS_ASSISTANCE,
            VolunteerSafetyStatus.IN_DANGER,
        }:
            if self.safety_gateway is None:
                logger.error(
                    "Safety escalation gateway missing "
                    "for volunteer %s",
                    volunteer.id,
                )

                raise VolunteerIntegrationError(
                    "Safety status recorded but escalation "
                    "gateway is not configured"
                )

            try:
                await self.safety_gateway.request_safety_escalation(
                    volunteer_id=volunteer.id,
                    assignment_id=assignment_id,
                    district_id=volunteer.district_id,
                    safety_status=safety_status,
                    description=description,
                    idempotency_key=(
                        f"volunteer-safety:"
                        f"{volunteer.id}:"
                        f"{saved.version}"
                    ),
                )

            except Exception as exc:
                logger.exception(
                    "Volunteer safety escalation failed"
                )

                raise VolunteerIntegrationError(
                    "Safety report was saved, but "
                    "escalation could not be confirmed"
                ) from exc

        return saved

    async def list_district_volunteers(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> Sequence[Volunteer]:

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise VolunteerAuthorizationError(
                "District volunteer access denied"
            )

        return await self.volunteers.list_by_district(
            district_id
        )

    async def list_district_tasks(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> Sequence[VolunteerTask]:

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise VolunteerAuthorizationError(
                "District task access denied"
            )

        return await self.tasks.list_by_district(
            district_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        volunteers = await self.list_district_volunteers(
            district_id,
            actor_id=actor_id,
        )

        tasks = await self.list_district_tasks(
            district_id,
            actor_id=actor_id,
        )

        volunteer_status_counts = {
            status.value: 0
            for status in VolunteerStatus
        }

        task_status_counts = {
            status.value: 0
            for status in VolunteerTaskStatus
        }

        for volunteer in volunteers:
            volunteer_status_counts[
                volunteer.status.value
            ] += 1

        for task in tasks:
            task_status_counts[
                task.status.value
            ] += 1

        return {
            "district_id": district_id,
            "total_volunteers": len(volunteers),
            "identity_verified_volunteers": sum(
                volunteer.identity_verified
                for volunteer in volunteers
            ),
            "active_volunteers": sum(
                volunteer.status == VolunteerStatus.ACTIVE
                for volunteer in volunteers
            ),
            "on_duty_volunteers": sum(
                volunteer.status == VolunteerStatus.ON_DUTY
                for volunteer in volunteers
            ),
            "volunteers_needing_assistance": sum(
                volunteer.safety_status
                in {
                    VolunteerSafetyStatus.NEEDS_ASSISTANCE,
                    VolunteerSafetyStatus.IN_DANGER,
                }
                for volunteer in volunteers
            ),
            "total_tasks": len(tasks),
            "volunteer_status_counts": (
                volunteer_status_counts
            ),
            "task_status_counts": (
                task_status_counts
            ),
            "total_completed_assignments_reported": sum(
                volunteer.total_completed_tasks
                for volunteer in volunteers
            ),
            "total_service_hours_reported": round(
                sum(
                    volunteer.total_service_minutes
                    for volunteer in volunteers
                ) / 60,
                2,
            ),
            "generated_at": utc_now().isoformat(),
        }

    async def generate_volunteer_report(
        self,
        volunteer_id: str,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        volunteer = await self.get_volunteer(
            volunteer_id
        )

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=volunteer.district_id,
        ):
            raise VolunteerAuthorizationError(
                "Volunteer report access denied"
            )

        assignments = (
            await self.assignments.list_by_volunteer(
                volunteer.id
            )
        )

        assignment_counts = {
            status.value: 0
            for status in VolunteerAssignmentStatus
        }

        for assignment in assignments:
            assignment_counts[
                assignment.status.value
            ] += 1

        return {
            "volunteer_id": volunteer.id,
            "district_id": volunteer.district_id,
            "status": volunteer.status.value,
            "skills": sorted(
                skill.value
                for skill in volunteer.skills
            ),
            "verified_skills": sorted(
                skill.value
                for skill in volunteer.verified_skills
            ),
            "identity_verified": (
                volunteer.identity_verified
            ),
            "safety_status": (
                volunteer.safety_status.value
            ),
            "completed_tasks": (
                volunteer.total_completed_tasks
            ),
            "service_hours": round(
                volunteer.total_service_minutes / 60,
                2,
            ),
            "assignment_counts": assignment_counts,
            "last_check_in_at": (
                volunteer.last_check_in_at.isoformat()
                if volunteer.last_check_in_at
                else None
            ),
            "last_check_out_at": (
                volunteer.last_check_out_at.isoformat()
                if volunteer.last_check_out_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Volunteer records are operational "
                "coordination data. Professional "
                "credentials, availability, safe routes "
                "and field conditions must be verified "
                "before assignment or deployment."
            ),
        }
