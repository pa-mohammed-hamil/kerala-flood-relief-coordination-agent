from __future__ import annotations

"""
backend/services/medical_service.py

Kerala Flood Relief Coordination Agent
Medical Emergency Coordination Service

Features:
- Medical emergency registration
- Patient triage information recording
- Emergency severity and priority assessment
- Hospital availability matching
- Hospital bed and specialty verification
- Ambulance availability matching
- Verified medical transport route assessment
- Human-reviewed medical coordination
- Hospital and ambulance reservations
- Medical transfer tracking
- Patient arrival confirmation
- Medical incident escalation
- Medical resource statistics
- Audit logging and reporting
- Optimistic concurrency control

SAFETY:
This service supports emergency coordination.
It is not a medical diagnosis or treatment system.

Automated triage is advisory and must not replace
qualified medical assessment.

The service does not autonomously dispatch ambulances,
admit patients, or contact emergency authorities.

Patient information is sensitive. Production systems
must enforce authentication, access control, encryption,
retention policies and access auditing.

Operational deployment requires verified integrations,
atomic resource reservations and reliable notifications.
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

class MedicalPriority(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


class MedicalCaseStatus(str, Enum):
    REPORTED = "reported"
    UNDER_TRIAGE = "under_triage"
    TRIAGED = "triaged"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    RESOURCES_RESERVED = "resources_reserved"
    TRANSPORT_REQUESTED = "transport_requested"
    EN_ROUTE = "en_route"
    ARRIVED = "arrived"
    HANDED_OVER = "handed_over"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MedicalSpecialty(str, Enum):
    GENERAL = "general"
    EMERGENCY_MEDICINE = "emergency_medicine"
    TRAUMA = "trauma"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    PEDIATRICS = "pediatrics"
    OBSTETRICS = "obstetrics"
    ORTHOPEDICS = "orthopedics"
    CRITICAL_CARE = "critical_care"
    DIALYSIS = "dialysis"
    RESPIRATORY = "respiratory"
    OTHER = "other"


class BedType(str, Enum):
    GENERAL = "general"
    EMERGENCY = "emergency"
    ICU = "icu"
    PEDIATRIC = "pediatric"
    MATERNITY = "maternity"
    ISOLATION = "isolation"


class AmbulanceType(str, Enum):
    BASIC_LIFE_SUPPORT = "basic_life_support"
    ADVANCED_LIFE_SUPPORT = "advanced_life_support"
    PATIENT_TRANSPORT = "patient_transport"
    BOAT_AMBULANCE = "boat_ambulance"
    OTHER = "other"


class AmbulanceStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class MedicalCaseSource(str, Enum):
    SOS = "sos"
    CITIZEN = "citizen"
    VOLUNTEER = "volunteer"
    RESCUE_TEAM = "rescue_team"
    HEALTH_WORKER = "health_worker"
    HOSPITAL = "hospital"
    OFFICIAL = "official"


# ============================================================
# EXCEPTIONS
# ============================================================

class MedicalServiceError(Exception):
    pass


class MedicalValidationError(MedicalServiceError):
    pass


class MedicalNotFoundError(MedicalServiceError):
    pass


class MedicalConflictError(MedicalServiceError):
    pass


class MedicalAuthorizationError(MedicalServiceError):
    pass


class MedicalCapacityError(MedicalServiceError):
    pass


class MedicalRouteError(MedicalServiceError):
    pass


class MedicalIntegrationError(MedicalServiceError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise MedicalValidationError(
            f"{name} must be timezone-aware"
        )


def validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:
    if (
        not math.isfinite(latitude)
        or not -90 <= latitude <= 90
    ):
        raise MedicalValidationError(
            "Invalid latitude"
        )

    if (
        not math.isfinite(longitude)
        or not -180 <= longitude <= 180
    ):
        raise MedicalValidationError(
            "Invalid longitude"
        )


def require_text(
    value: str,
    name: str,
) -> str:
    if not value or not value.strip():
        raise MedicalValidationError(
            f"{name} is required"
        )

    return value.strip()


def haversine_distance_km(
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
class MedicalLocation:
    latitude: float
    longitude: float
    address: str = ""
    landmark: str = ""

    def __post_init__(self) -> None:
        validate_coordinates(
            self.latitude,
            self.longitude,
        )


@dataclass(frozen=True)
class PatientInformation:
    patient_reference: str
    age: int | None = None

    requires_mobility_assistance: bool = False
    requires_accessible_transport: bool = False

    notes: str = ""

    def __post_init__(self) -> None:
        require_text(
            self.patient_reference,
            "Patient reference",
        )

        if self.age is not None:
            if not 0 <= self.age <= 125:
                raise MedicalValidationError(
                    "Invalid patient age"
                )


@dataclass(frozen=True)
class MedicalTriageInformation:
    """
    Structured information supplied by a
    qualified medical professional.

    This object does not independently
    diagnose or classify a patient.
    """

    clinician_priority: MedicalPriority | None = None

    specialty_required: MedicalSpecialty = (
        MedicalSpecialty.EMERGENCY_MEDICINE
    )

    bed_type_required: BedType = BedType.EMERGENCY

    ambulance_type_required: AmbulanceType = (
        AmbulanceType.BASIC_LIFE_SUPPORT
    )

    requires_oxygen: bool = False
    requires_monitoring: bool = False
    suspected_infectious_risk: bool = False

    clinician_reference: str | None = None
    assessment_notes: str = ""

    assessed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.assessed_at is not None:
            validate_datetime(
                self.assessed_at,
                "assessed_at",
            )


@dataclass(frozen=True)
class MedicalCaseCreateRequest:
    district_id: int
    incident_id: str

    source: MedicalCaseSource
    location: MedicalLocation

    patient: PatientInformation

    description: str

    triage: MedicalTriageInformation | None = None

    sos_request_id: str | None = None

    external_reference: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class HospitalOption:
    id: str
    name: str
    district_id: int
    location: MedicalLocation

    operational: bool
    verified: bool

    specialties: frozenset[MedicalSpecialty]

    available_beds: Mapping[BedType, int]

    oxygen_available: bool = False
    patient_monitoring_available: bool = False

    infectious_isolation_available: bool = False

    last_capacity_update: datetime | None = None


@dataclass(frozen=True)
class AmbulanceOption:
    id: str
    district_id: int

    ambulance_type: AmbulanceType
    location: MedicalLocation

    status: AmbulanceStatus

    operational: bool = True
    verified: bool = False

    driver_verified: bool = False
    medical_crew_verified: bool = False

    oxygen_available: bool = False
    monitoring_available: bool = False

    last_location_update: datetime | None = None


@dataclass(frozen=True)
class MedicalRoute:
    route_id: str

    feasible: bool
    verified: bool

    distance_km: float | None = None
    estimated_minutes: float | None = None

    verified_at: datetime | None = None

    flood_affected: bool = False

    blocked_segments: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class HospitalRecommendation:
    hospital_id: str
    hospital_name: str

    available_beds: int

    specialty_supported: bool
    required_equipment_available: bool

    route: MedicalRoute

    score: float

    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AmbulanceRecommendation:
    ambulance_id: str
    ambulance_type: AmbulanceType

    score: float
    distance_km: float

    warnings: tuple[str, ...] = ()


@dataclass
class MedicalCase:
    id: str
    district_id: int
    incident_id: str

    source: MedicalCaseSource

    location: MedicalLocation
    patient: PatientInformation

    description: str

    priority: MedicalPriority

    triage: MedicalTriageInformation | None

    status: MedicalCaseStatus = (
        MedicalCaseStatus.REPORTED
    )

    hospital_id: str | None = None
    ambulance_id: str | None = None
    route_id: str | None = None

    sos_request_id: str | None = None
    external_reference: str | None = None

    reported_by: int | None = None
    triaged_by: int | None = None
    approved_by: int | None = None
    coordinated_by: int | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    triaged_at: datetime | None = None
    approved_at: datetime | None = None
    reserved_at: datetime | None = None
    transport_requested_at: datetime | None = None
    arrived_at: datetime | None = None
    handed_over_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None

    approval_reference: str | None = None
    handover_reference: str | None = None
    cancellation_reason: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class MedicalCaseRepository(Protocol):

    async def create(
        self,
        case: MedicalCase,
    ) -> MedicalCase:
        ...

    async def get_by_id(
        self,
        case_id: str,
    ) -> MedicalCase | None:
        ...

    async def update(
        self,
        case: MedicalCase,
        *,
        expected_version: int,
    ) -> MedicalCase:
        """
        Must enforce optimistic concurrency.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[MedicalCase]:
        ...

    async def list_by_incident(
        self,
        incident_id: str,
    ) -> Sequence[MedicalCase]:
        ...

    async def find_by_external_reference(
        self,
        external_reference: str,
    ) -> MedicalCase | None:
        ...


class HospitalGateway(Protocol):

    async def list_available(
        self,
        district_id: int,
    ) -> Sequence[HospitalOption]:
        ...

    async def get_hospital(
        self,
        hospital_id: str,
    ) -> HospitalOption | None:
        ...

    async def reserve_bed(
        self,
        *,
        hospital_id: str,
        case_id: str,
        bed_type: BedType,
    ) -> bool:
        """
        Atomically reserve one verified bed.
        Must be idempotent by case_id.
        """
        ...

    async def release_bed(
        self,
        *,
        hospital_id: str,
        case_id: str,
    ) -> None:
        ...


class AmbulanceGateway(Protocol):

    async def list_available(
        self,
        district_id: int,
    ) -> Sequence[AmbulanceOption]:
        ...

    async def get_ambulance(
        self,
        ambulance_id: str,
    ) -> AmbulanceOption | None:
        ...

    async def reserve_ambulance(
        self,
        *,
        ambulance_id: str,
        case_id: str,
    ) -> bool:
        """
        Atomically reserve an ambulance.
        Must be idempotent by case_id.
        """
        ...

    async def release_ambulance(
        self,
        *,
        ambulance_id: str,
        case_id: str,
    ) -> None:
        ...


class MedicalRouteGateway(Protocol):

    async def assess_route(
        self,
        *,
        origin: MedicalLocation,
        destination: MedicalLocation,
        ambulance_type: AmbulanceType,
    ) -> MedicalRoute:
        ...


class MedicalAuthorizationGateway(Protocol):

    async def can_register(
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

    async def can_approve(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_coordinate(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...


class MedicalAuditGateway(Protocol):

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


class MedicalNotificationGateway(Protocol):

    async def notify_transport_request(
        self,
        case: MedicalCase,
    ) -> bool:
        ...

    async def notify_escalation(
        self,
        case: MedicalCase,
    ) -> bool:
        ...

    async def notify_cancellation(
        self,
        case: MedicalCase,
    ) -> bool:
        ...


# ============================================================
# TRIAGE VALIDATOR
# ============================================================

class MedicalTriageValidator:
    """
    Validates clinician-provided triage information.

    No algorithmic diagnosis or automatic
    medical-priority inference is performed.
    """

    def validate(
        self,
        triage: MedicalTriageInformation,
    ) -> MedicalPriority:

        if triage.clinician_priority is None:
            raise MedicalValidationError(
                "Qualified clinician priority is required"
            )

        if not triage.clinician_reference:
            raise MedicalValidationError(
                "Clinician reference is required"
            )

        if triage.assessed_at is None:
            raise MedicalValidationError(
                "Clinical assessment time is required"
            )

        if triage.assessed_at > utc_now():
            raise MedicalValidationError(
                "Clinical assessment cannot be future-dated"
            )

        return triage.clinician_priority


# ============================================================
# HOSPITAL MATCHING
# ============================================================

class HospitalMatcher:

    def __init__(
        self,
        *,
        maximum_capacity_age_minutes: int = 15,
        maximum_route_age_minutes: int = 10,
    ):
        self.maximum_capacity_age_minutes = (
            maximum_capacity_age_minutes
        )

        self.maximum_route_age_minutes = (
            maximum_route_age_minutes
        )

    def validate_route(
        self,
        route: MedicalRoute,
    ) -> bool:

        if not route.feasible or not route.verified:
            return False

        if route.flood_affected or route.blocked_segments:
            return False

        if route.verified_at is None:
            return False

        validate_datetime(
            route.verified_at,
            "route.verified_at",
        )

        age = utc_now() - route.verified_at

        if age < timedelta(0):
            return False

        if age > timedelta(
            minutes=self.maximum_route_age_minutes
        ):
            return False

        return True

    def evaluate(
        self,
        *,
        hospital: HospitalOption,
        triage: MedicalTriageInformation,
        route: MedicalRoute,
    ) -> HospitalRecommendation | None:

        if not hospital.operational or not hospital.verified:
            return None

        if hospital.last_capacity_update is None:
            return None

        validate_datetime(
            hospital.last_capacity_update,
            "last_capacity_update",
        )

        capacity_age = (
            utc_now() - hospital.last_capacity_update
        )

        if (
            capacity_age < timedelta(0)
            or capacity_age > timedelta(
                minutes=self.maximum_capacity_age_minutes
            )
        ):
            return None

        specialty_supported = (
            triage.specialty_required
            in hospital.specialties
        )

        if not specialty_supported:
            return None

        available_beds = hospital.available_beds.get(
            triage.bed_type_required,
            0,
        )

        if available_beds <= 0:
            return None

        if (
            triage.requires_oxygen
            and not hospital.oxygen_available
        ):
            return None

        if (
            triage.requires_monitoring
            and not hospital.patient_monitoring_available
        ):
            return None

        if (
            triage.suspected_infectious_risk
            and not hospital.infectious_isolation_available
        ):
            return None

        if not self.validate_route(
            route
        ):
            return None

        if route.estimated_minutes is not None:
            travel_score = 1.0 / (
                1.0 + route.estimated_minutes / 30.0
            )
        else:
            travel_score = 0.5

        capacity_score = min(
            1.0,
            available_beds / 10.0,
        )

        score = (
            travel_score * 0.60
            + capacity_score * 0.25
            + 0.15
        )

        return HospitalRecommendation(
            hospital_id=hospital.id,
            hospital_name=hospital.name,
            available_beds=available_beds,
            specialty_supported=True,
            required_equipment_available=True,
            route=route,
            score=round(score, 4),
            warnings=route.warnings,
        )


# ============================================================
# AMBULANCE MATCHING
# ============================================================

class AmbulanceMatcher:

    def __init__(
        self,
        *,
        maximum_location_age_minutes: int = 15,
    ):
        self.maximum_location_age_minutes = (
            maximum_location_age_minutes
        )

    def evaluate(
        self,
        *,
        ambulance: AmbulanceOption,
        case: MedicalCase,
    ) -> AmbulanceRecommendation | None:

        if case.triage is None:
            return None

        if ambulance.district_id != case.district_id:
            return None

        if ambulance.status != AmbulanceStatus.AVAILABLE:
            return None

        if not ambulance.operational or not ambulance.verified:
            return None

        if (
            not ambulance.driver_verified
            or not ambulance.medical_crew_verified
        ):
            return None

        if (
            ambulance.ambulance_type
            != case.triage.ambulance_type_required
        ):
            return None

        if (
            case.triage.requires_oxygen
            and not ambulance.oxygen_available
        ):
            return None

        if (
            case.triage.requires_monitoring
            and not ambulance.monitoring_available
        ):
            return None

        if ambulance.last_location_update is None:
            return None

        validate_datetime(
            ambulance.last_location_update,
            "last_location_update",
        )

        age = utc_now() - ambulance.last_location_update

        if (
            age < timedelta(0)
            or age > timedelta(
                minutes=self.maximum_location_age_minutes
            )
        ):
            return None

        distance = haversine_distance_km(
            ambulance.location.latitude,
            ambulance.location.longitude,
            case.location.latitude,
            case.location.longitude,
        )

        score = 1.0 / (
            1.0 + distance / 10.0
        )

        return AmbulanceRecommendation(
            ambulance_id=ambulance.id,
            ambulance_type=ambulance.ambulance_type,
            score=round(score, 4),
            distance_km=round(distance, 3),
        )


# ============================================================
# MEDICAL SERVICE
# ============================================================

class MedicalService:

    TERMINAL_STATUSES = frozenset({
        MedicalCaseStatus.COMPLETED,
        MedicalCaseStatus.CANCELLED,
    })

    def __init__(
        self,
        *,
        repository: MedicalCaseRepository,
        hospital_gateway: HospitalGateway,
        ambulance_gateway: AmbulanceGateway,
        route_gateway: MedicalRouteGateway,
        authorization_gateway: MedicalAuthorizationGateway,
        audit_gateway: MedicalAuditGateway | None = None,
        notification_gateway: (
            MedicalNotificationGateway | None
        ) = None,
        triage_validator: MedicalTriageValidator | None = None,
        hospital_matcher: HospitalMatcher | None = None,
        ambulance_matcher: AmbulanceMatcher | None = None,
    ):
        self.repository = repository
        self.hospitals = hospital_gateway
        self.ambulances = ambulance_gateway
        self.routes = route_gateway
        self.authorization = authorization_gateway
        self.audit = audit_gateway
        self.notifications = notification_gateway

        self.triage_validator = (
            triage_validator
            or MedicalTriageValidator()
        )

        self.hospital_matcher = (
            hospital_matcher
            or HospitalMatcher()
        )

        self.ambulance_matcher = (
            ambulance_matcher
            or AmbulanceMatcher()
        )

    async def _audit(
        self,
        action: str,
        case: MedicalCase,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        # Deliberately exclude patient identifiers
        # and clinical details from audit metadata.
        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="medical_case",
            entity_id=case.id,
            metadata={
                "district_id": case.district_id,
                "status": case.status.value,
                "priority": case.priority.value,
            },
        )

    async def get_case(
        self,
        case_id: str,
    ) -> MedicalCase:

        case = await self.repository.get_by_id(
            case_id
        )

        if case is None:
            raise MedicalNotFoundError(
                "Medical case not found"
            )

        return case

    async def register_case(
        self,
        request: MedicalCaseCreateRequest,
        *,
        actor_id: int,
    ) -> MedicalCase:

        if request.district_id <= 0:
            raise MedicalValidationError(
                "Invalid district ID"
            )

        require_text(
            request.incident_id,
            "Incident ID",
        )

        require_text(
            request.description,
            "Description",
        )

        if not await self.authorization.can_register(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical case registration permission denied"
            )

        if request.external_reference:
            existing = (
                await self.repository.find_by_external_reference(
                    request.external_reference
                )
            )

            if existing is not None:
                return existing

        case = MedicalCase(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            incident_id=request.incident_id,
            source=request.source,
            location=request.location,
            patient=request.patient,
            description=request.description.strip(),
            priority=MedicalPriority.URGENT,
            triage=None,
            status=MedicalCaseStatus.REPORTED,
            sos_request_id=request.sos_request_id,
            external_reference=request.external_reference,
            reported_by=actor_id,
            metadata=dict(request.metadata),
        )

        created = await self.repository.create(
            case
        )

        await self._audit(
            "medical.case_registered",
            created,
            actor_id,
        )

        return created

    async def start_triage(
        self,
        case_id: str,
        *,
        actor_id: int,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_triage(
            user_id=actor_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical triage permission denied"
            )

        if case.status != MedicalCaseStatus.REPORTED:
            raise MedicalConflictError(
                "Only reported cases can enter triage"
            )

        updated = replace(
            case,
            status=MedicalCaseStatus.UNDER_TRIAGE,
            updated_at=utc_now(),
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        await self._audit(
            "medical.triage_started",
            saved,
            actor_id,
        )

        return saved

    async def record_triage(
        self,
        case_id: str,
        *,
        triage: MedicalTriageInformation,
        clinician_user_id: int,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_triage(
            user_id=clinician_user_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical triage permission denied"
            )

        if case.status != MedicalCaseStatus.UNDER_TRIAGE:
            raise MedicalConflictError(
                "Case is not under triage"
            )

        priority = self.triage_validator.validate(
            triage
        )

        updated = replace(
            case,
            triage=triage,
            priority=priority,
            status=MedicalCaseStatus.TRIAGED,
            triaged_by=clinician_user_id,
            triaged_at=utc_now(),
            updated_at=utc_now(),
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        await self._audit(
            "medical.triage_recorded",
            saved,
            clinician_user_id,
        )

        return saved

    async def recommend_hospitals(
        self,
        case_id: str,
        *,
        ambulance_type: AmbulanceType | None = None,
        limit: int = 5,
    ) -> list[HospitalRecommendation]:

        case = await self.get_case(
            case_id
        )

        if case.triage is None:
            raise MedicalConflictError(
                "Clinical triage is required"
            )

        if not 1 <= limit <= 50:
            raise MedicalValidationError(
                "Limit must be between 1 and 50"
            )

        hospitals = await self.hospitals.list_available(
            case.district_id
        )

        recommendations: list[
            HospitalRecommendation
        ] = []

        for hospital in hospitals:
            try:
                route = await self.routes.assess_route(
                    origin=case.location,
                    destination=hospital.location,
                    ambulance_type=(
                        ambulance_type
                        or case.triage.ambulance_type_required
                    ),
                )

                recommendation = (
                    self.hospital_matcher.evaluate(
                        hospital=hospital,
                        triage=case.triage,
                        route=route,
                    )
                )

                if recommendation is not None:
                    recommendations.append(
                        recommendation
                    )

            except Exception:
                logger.exception(
                    "Hospital route assessment failed: %s",
                    hospital.id,
                )

        recommendations.sort(
            key=lambda item: (
                -item.score,
                item.hospital_id,
            )
        )

        return recommendations[:limit]

    async def recommend_ambulances(
        self,
        case_id: str,
        *,
        limit: int = 5,
    ) -> list[AmbulanceRecommendation]:

        case = await self.get_case(
            case_id
        )

        if case.triage is None:
            raise MedicalConflictError(
                "Clinical triage is required"
            )

        ambulances = (
            await self.ambulances.list_available(
                case.district_id
            )
        )

        recommendations = []

        for ambulance in ambulances:
            recommendation = (
                self.ambulance_matcher.evaluate(
                    ambulance=ambulance,
                    case=case,
                )
            )

            if recommendation is not None:
                recommendations.append(
                    recommendation
                )

        recommendations.sort(
            key=lambda item: (
                -item.score,
                item.ambulance_id,
            )
        )

        return recommendations[:limit]

    async def submit_for_approval(
        self,
        case_id: str,
        *,
        actor_id: int,
        hospital_id: str,
        ambulance_id: str,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_coordinate(
            user_id=actor_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical coordination permission denied"
            )

        if case.status != MedicalCaseStatus.TRIAGED:
            raise MedicalConflictError(
                "Only triaged cases can be submitted"
            )

        require_text(
            hospital_id,
            "Hospital ID",
        )

        require_text(
            ambulance_id,
            "Ambulance ID",
        )

        hospital_options = (
            await self.recommend_hospitals(
                case_id,
                limit=50,
            )
        )

        ambulance_options = (
            await self.recommend_ambulances(
                case_id,
                limit=50,
            )
        )

        if not any(
            option.hospital_id == hospital_id
            for option in hospital_options
        ):
            raise MedicalCapacityError(
                "Selected hospital is unsuitable"
            )

        if not any(
            option.ambulance_id == ambulance_id
            for option in ambulance_options
        ):
            raise MedicalCapacityError(
                "Selected ambulance is unsuitable"
            )

        updated = replace(
            case,
            hospital_id=hospital_id,
            ambulance_id=ambulance_id,
            status=MedicalCaseStatus.PENDING_APPROVAL,
            coordinated_by=actor_id,
            updated_at=utc_now(),
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        await self._audit(
            "medical.approval_requested",
            saved,
            actor_id,
        )

        return saved

    async def approve_transfer(
        self,
        case_id: str,
        *,
        reviewer_id: int,
        approval_reference: str,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical transfer approval denied"
            )

        if (
            case.status
            != MedicalCaseStatus.PENDING_APPROVAL
        ):
            raise MedicalConflictError(
                "Case is not awaiting approval"
            )

        if case.coordinated_by == reviewer_id:
            raise MedicalAuthorizationError(
                "Independent reviewer required"
            )

        require_text(
            approval_reference,
            "Approval reference",
        )

        updated = replace(
            case,
            status=MedicalCaseStatus.APPROVED,
            approved_by=reviewer_id,
            approved_at=utc_now(),
            approval_reference=approval_reference.strip(),
            updated_at=utc_now(),
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        await self._audit(
            "medical.transfer_approved",
            saved,
            reviewer_id,
        )

        return saved

    async def reserve_resources(
        self,
        case_id: str,
        *,
        actor_id: int,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_coordinate(
            user_id=actor_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical resource reservation denied"
            )

        if case.status != MedicalCaseStatus.APPROVED:
            raise MedicalConflictError(
                "Only approved cases can reserve resources"
            )

        if (
            case.triage is None
            or case.hospital_id is None
            or case.ambulance_id is None
            or case.approved_by is None
        ):
            raise MedicalConflictError(
                "Incomplete approved transfer"
            )

        hospital = await self.hospitals.get_hospital(
            case.hospital_id
        )

        ambulance = await self.ambulances.get_ambulance(
            case.ambulance_id
        )

        if hospital is None or ambulance is None:
            raise MedicalCapacityError(
                "Selected resources are unavailable"
            )

        if not hospital.operational or not hospital.verified:
            raise MedicalCapacityError(
                "Hospital is not verified and operational"
            )

        if (
            ambulance.status != AmbulanceStatus.AVAILABLE
            or not ambulance.operational
            or not ambulance.verified
            or not ambulance.driver_verified
            or not ambulance.medical_crew_verified
        ):
            raise MedicalCapacityError(
                "Ambulance is unavailable or unverified"
            )

        if (
            ambulance.ambulance_type
            != case.triage.ambulance_type_required
        ):
            raise MedicalCapacityError(
                "Ambulance type does not meet clinical requirements"
            )

        route = await self.routes.assess_route(
            origin=case.location,
            destination=hospital.location,
            ambulance_type=ambulance.ambulance_type,
        )

        if not self.hospital_matcher.validate_route(
            route
        ):
            raise MedicalRouteError(
                "Medical transport route is not verified"
            )

        bed_reserved = await self.hospitals.reserve_bed(
            hospital_id=hospital.id,
            case_id=case.id,
            bed_type=case.triage.bed_type_required,
        )

        if not bed_reserved:
            raise MedicalCapacityError(
                "Hospital bed reservation failed"
            )

        try:
            ambulance_reserved = (
                await self.ambulances.reserve_ambulance(
                    ambulance_id=ambulance.id,
                    case_id=case.id,
                )
            )

            if not ambulance_reserved:
                raise MedicalCapacityError(
                    "Ambulance reservation failed"
                )

            updated = replace(
                case,
                status=MedicalCaseStatus.RESOURCES_RESERVED,
                route_id=route.route_id,
                reserved_at=utc_now(),
                updated_at=utc_now(),
                version=case.version + 1,
            )

            saved = await self.repository.update(
                updated,
                expected_version=case.version,
            )

        except Exception:
            # Production: use durable compensation or
            # a reservation saga to avoid orphaned resources.
            await self.ambulances.release_ambulance(
                ambulance_id=ambulance.id,
                case_id=case.id,
            )

            await self.hospitals.release_bed(
                hospital_id=hospital.id,
                case_id=case.id,
            )

            raise

        await self._audit(
            "medical.resources_reserved",
            saved,
            actor_id,
        )

        return saved

    async def request_transport(
        self,
        case_id: str,
        *,
        actor_id: int,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_coordinate(
            user_id=actor_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical transport permission denied"
            )

        if (
            case.status
            != MedicalCaseStatus.RESOURCES_RESERVED
        ):
            raise MedicalConflictError(
                "Resources must be reserved first"
            )

        updated = replace(
            case,
            status=MedicalCaseStatus.TRANSPORT_REQUESTED,
            transport_requested_at=utc_now(),
            updated_at=utc_now(),
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        await self._audit(
            "medical.transport_requested",
            saved,
            actor_id,
        )

        if self.notifications is not None:
            try:
                delivered = (
                    await self.notifications
                    .notify_transport_request(saved)
                )

                if not delivered:
                    logger.error(
                        "Transport notification unconfirmed: %s",
                        saved.id,
                    )

            except Exception:
                logger.exception(
                    "Transport notification failed: %s",
                    saved.id,
                )

        return saved

    async def update_transport_status(
        self,
        case_id: str,
        *,
        new_status: MedicalCaseStatus,
        actor_id: int,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_coordinate(
            user_id=actor_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical transport update denied"
            )

        transitions = {
            MedicalCaseStatus.TRANSPORT_REQUESTED: {
                MedicalCaseStatus.EN_ROUTE,
            },
            MedicalCaseStatus.EN_ROUTE: {
                MedicalCaseStatus.ARRIVED,
            },
        }

        if new_status not in transitions.get(
            case.status,
            set(),
        ):
            raise MedicalConflictError(
                "Invalid medical transport status transition"
            )

        now = utc_now()

        updated = replace(
            case,
            status=new_status,
            arrived_at=(
                now
                if new_status == MedicalCaseStatus.ARRIVED
                else case.arrived_at
            ),
            updated_at=now,
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        await self._audit(
            f"medical.{new_status.value}",
            saved,
            actor_id,
        )

        return saved

    async def confirm_handover(
        self,
        case_id: str,
        *,
        actor_id: int,
        handover_reference: str,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_coordinate(
            user_id=actor_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical handover permission denied"
            )

        if case.status != MedicalCaseStatus.ARRIVED:
            raise MedicalConflictError(
                "Patient must arrive before handover"
            )

        require_text(
            handover_reference,
            "Handover reference",
        )

        updated = replace(
            case,
            status=MedicalCaseStatus.HANDED_OVER,
            handover_reference=handover_reference.strip(),
            handed_over_at=utc_now(),
            updated_at=utc_now(),
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        # Hospital reservation must be converted
        # to actual occupancy before releasing it.

        await self._audit(
            "medical.patient_handed_over",
            saved,
            actor_id,
        )

        return saved

    async def complete_case(
        self,
        case_id: str,
        *,
        actor_id: int,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_coordinate(
            user_id=actor_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical completion permission denied"
            )

        if case.status != MedicalCaseStatus.HANDED_OVER:
            raise MedicalConflictError(
                "Patient handover is required"
            )

        updated = replace(
            case,
            status=MedicalCaseStatus.COMPLETED,
            completed_at=utc_now(),
            updated_at=utc_now(),
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        if case.ambulance_id is not None:
            await self.ambulances.release_ambulance(
                ambulance_id=case.ambulance_id,
                case_id=case.id,
            )

        await self._audit(
            "medical.case_completed",
            saved,
            actor_id,
        )

        return saved

    async def cancel_case(
        self,
        case_id: str,
        *,
        actor_id: int,
        reason: str,
    ) -> MedicalCase:

        case = await self.get_case(
            case_id
        )

        if not await self.authorization.can_coordinate(
            user_id=actor_id,
            district_id=case.district_id,
        ):
            raise MedicalAuthorizationError(
                "Medical case cancellation denied"
            )

        if case.status in self.TERMINAL_STATUSES:
            raise MedicalConflictError(
                "Medical case is already terminal"
            )

        if case.status in {
            MedicalCaseStatus.EN_ROUTE,
            MedicalCaseStatus.ARRIVED,
            MedicalCaseStatus.HANDED_OVER,
        }:
            raise MedicalConflictError(
                "Active patient transfers require "
                "a supervised diversion or handover workflow"
            )

        require_text(
            reason,
            "Cancellation reason",
        )

        updated = replace(
            case,
            status=MedicalCaseStatus.CANCELLED,
            cancelled_at=utc_now(),
            cancellation_reason=reason.strip(),
            updated_at=utc_now(),
            version=case.version + 1,
        )

        saved = await self.repository.update(
            updated,
            expected_version=case.version,
        )

        if case.status in {
            MedicalCaseStatus.RESOURCES_RESERVED,
            MedicalCaseStatus.TRANSPORT_REQUESTED,
        }:
            if case.ambulance_id is not None:
                await self.ambulances.release_ambulance(
                    ambulance_id=case.ambulance_id,
                    case_id=case.id,
                )

            if case.hospital_id is not None:
                await self.hospitals.release_bed(
                    hospital_id=case.hospital_id,
                    case_id=case.id,
                )

        await self._audit(
            "medical.case_cancelled",
            saved,
            actor_id,
        )

        if self.notifications is not None:
            try:
                await self.notifications.notify_cancellation(
                    saved
                )
            except Exception:
                logger.exception(
                    "Medical cancellation notification failed: %s",
                    saved.id,
                )

        return saved

    async def list_district_cases(
        self,
        district_id: int,
    ) -> Sequence[MedicalCase]:

        if district_id <= 0:
            raise MedicalValidationError(
                "Invalid district ID"
            )

        return await self.repository.list_by_district(
            district_id
        )

    async def list_incident_cases(
        self,
        incident_id: str,
    ) -> Sequence[MedicalCase]:

        require_text(
            incident_id,
            "Incident ID",
        )

        return await self.repository.list_by_incident(
            incident_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
    ) -> dict[str, Any]:

        cases = await self.list_district_cases(
            district_id
        )

        status_counts = {
            status.value: 0
            for status in MedicalCaseStatus
        }

        priority_counts = {
            priority.value: 0
            for priority in MedicalPriority
        }

        for case in cases:
            status_counts[
                case.status.value
            ] += 1

            priority_counts[
                case.priority.value
            ] += 1

        return {
            "district_id": district_id,
            "total_cases": len(cases),
            "active_cases": sum(
                case.status not in self.TERMINAL_STATUSES
                for case in cases
            ),
            "critical_cases": priority_counts[
                MedicalPriority.CRITICAL.value
            ],
            "completed_transfers": status_counts[
                MedicalCaseStatus.COMPLETED.value
            ],
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_case_report(
        self,
        case_id: str,
    ) -> dict[str, Any]:

        case = await self.get_case(
            case_id
        )

        return {
            "case_id": case.id,
            "district_id": case.district_id,
            "incident_id": case.incident_id,
            "source": case.source.value,
            "priority": case.priority.value,
            "status": case.status.value,
            "hospital_id": case.hospital_id,
            "ambulance_id": case.ambulance_id,
            "route_id": case.route_id,
            "triage": {
                "specialty_required": (
                    case.triage.specialty_required.value
                    if case.triage
                    else None
                ),
                "bed_type_required": (
                    case.triage.bed_type_required.value
                    if case.triage
                    else None
                ),
                "ambulance_type_required": (
                    case.triage.ambulance_type_required.value
                    if case.triage
                    else None
                ),
            },
            "approval": {
                "approved_by": case.approved_by,
                "approved_at": (
                    case.approved_at.isoformat()
                    if case.approved_at
                    else None
                ),
                "approval_reference": (
                    case.approval_reference
                ),
            },
            "handover": {
                "handover_reference": (
                    case.handover_reference
                ),
                "handed_over_at": (
                    case.handed_over_at.isoformat()
                    if case.handed_over_at
                    else None
                ),
            },
            "created_at": case.created_at.isoformat(),
            "updated_at": case.updated_at.isoformat(),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "This is an operational coordination report, "
                "not a clinical record or medical diagnosis. "
                "Patient-sensitive information is excluded "
                "from this general report. Clinical decisions "
                "require qualified healthcare professionals."
            ),
        }
