from __future__ import annotations

"""
backend/services/shelter_service.py

Kerala Flood Relief Coordination Agent
=======================================

Emergency Shelter Management and Allocation Service.

Features:
- Shelter registration and verification
- Shelter capacity and occupancy management
- Evacuee registration and household tracking
- Vulnerable-person accessibility requirements
- Shelter suitability and allocation recommendations
- Human-approved shelter allocation
- Atomic capacity reservations
- Evacuee check-in and check-out
- Shelter-to-shelter transfer
- Shelter safety and operational status
- Food, water, sanitation and medical readiness
- Shelter inventory and supply requests
- Capacity and overcrowding alerts
- Occupancy reconciliation
- District-level statistics and reports
- Role-based access and audit logging

SAFETY:
- No autonomous evacuation or transport dispatch.
- Unverified shelters cannot receive allocations.
- Unknown shelter safety is not treated as safe.
- Reservations require atomic repository operations.
- Shelter suitability requires field verification.
- Sensitive resident details must be access-controlled.
- Production deployments require durable transactions,
  idempotency, encryption and operational monitoring.
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

class ShelterStatus(str, Enum):
    DRAFT = "draft"
    PENDING_VERIFICATION = "pending_verification"
    OPEN = "open"
    AT_CAPACITY = "at_capacity"
    TEMPORARILY_CLOSED = "temporarily_closed"
    CLOSED = "closed"
    UNSAFE = "unsafe"


class ShelterType(str, Enum):
    SCHOOL = "school"
    COMMUNITY_HALL = "community_hall"
    GOVERNMENT_BUILDING = "government_building"
    AUDITORIUM = "auditorium"
    RELIEF_CAMP = "relief_camp"
    HOSPITAL_SUPPORT = "hospital_support"
    TEMPORARY_STRUCTURE = "temporary_structure"
    OTHER = "other"


class ShelterSafetyLevel(str, Enum):
    VERIFIED_SAFE = "verified_safe"
    REQUIRES_REVIEW = "requires_review"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class ShelterFacility(str, Enum):
    DRINKING_WATER = "drinking_water"
    FOOD = "food"
    TOILETS = "toilets"
    BATHING = "bathing"
    ELECTRICITY = "electricity"
    GENERATOR = "generator"
    MEDICAL_SUPPORT = "medical_support"
    FIRST_AID = "first_aid"
    WHEELCHAIR_ACCESS = "wheelchair_access"
    ACCESSIBLE_TOILET = "accessible_toilet"
    CHILD_FRIENDLY_SPACE = "child_friendly_space"
    WOMEN_SAFE_SPACE = "women_safe_space"
    PRIVATE_SLEEPING_AREA = "private_sleeping_area"
    INFANT_CARE = "infant_care"
    PET_FRIENDLY = "pet_friendly"
    COMMUNICATION = "communication"
    SECURITY = "security"


class EvacueeStatus(str, Enum):
    REGISTERED = "registered"
    RESERVED = "reserved"
    CHECKED_IN = "checked_in"
    CHECKED_OUT = "checked_out"
    TRANSFER_PENDING = "transfer_pending"
    TRANSFERRED = "transferred"
    CANCELLED = "cancelled"


class ShelterReservationStatus(str, Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    RELEASED = "released"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ShelterAlertType(str, Enum):
    HIGH_OCCUPANCY = "high_occupancy"
    FULL_CAPACITY = "full_capacity"
    UNSAFE_SHELTER = "unsafe_shelter"
    WATER_SHORTAGE = "water_shortage"
    FOOD_SHORTAGE = "food_shortage"
    SANITATION_ISSUE = "sanitation_issue"
    MEDICAL_SUPPORT_REQUIRED = "medical_support_required"
    STAFF_SHORTAGE = "staff_shortage"
    DATA_STALE = "data_stale"


class ShelterSupplyCategory(str, Enum):
    FOOD = "food"
    DRINKING_WATER = "drinking_water"
    MEDICINE = "medicine"
    HYGIENE = "hygiene"
    BEDDING = "bedding"
    CLOTHING = "clothing"
    BABY_SUPPLIES = "baby_supplies"
    SANITATION = "sanitation"
    POWER = "power"
    OTHER = "other"


# ============================================================
# EXCEPTIONS
# ============================================================

class ShelterServiceError(Exception):
    pass


class ShelterValidationError(ShelterServiceError):
    pass


class ShelterNotFoundError(ShelterServiceError):
    pass


class ShelterConflictError(ShelterServiceError):
    pass


class ShelterAuthorizationError(ShelterServiceError):
    pass


class ShelterCapacityError(ShelterServiceError):
    pass


class ShelterSafetyError(ShelterServiceError):
    pass


class ShelterIntegrationError(ShelterServiceError):
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
        raise ShelterValidationError(
            f"{name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise ShelterValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if not isinstance(value, datetime):
        raise ShelterValidationError(
            f"{name} must be a datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ShelterValidationError(
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
        raise ShelterValidationError(
            f"{name} must be a non-negative integer"
        )


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius = 6371.0088

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
        * radius
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a),
        )
    )


# ============================================================
# LOCATION
# ============================================================

@dataclass(frozen=True)
class ShelterLocation:
    district_id: int

    latitude: float
    longitude: float

    address: str = ""
    panchayat: str = ""
    taluk: str = ""

    def __post_init__(self) -> None:
        if self.district_id <= 0:
            raise ShelterValidationError(
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
                raise ShelterValidationError(
                    f"Invalid {name}"
                )


# ============================================================
# SHELTER FACILITY AND SAFETY
# ============================================================

@dataclass(frozen=True)
class ShelterFacilityStatus:
    available_facilities: frozenset[
        ShelterFacility
    ] = field(
        default_factory=frozenset
    )

    toilets_count: int = 0
    beds_count: int = 0

    drinking_water_available: bool = False
    food_available: bool = False

    medical_staff_available: bool = False

    verified: bool = False

    verified_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_non_negative_integer(
            self.toilets_count,
            "toilets_count",
        )

        validate_non_negative_integer(
            self.beds_count,
            "beds_count",
        )

        if self.verified_at is not None:
            validate_datetime(
                self.verified_at,
                "verified_at",
            )


@dataclass(frozen=True)
class ShelterSafetyAssessment:
    safety_level: ShelterSafetyLevel

    structural_integrity_verified: bool
    flood_exposure_reviewed: bool
    electrical_safety_verified: bool
    sanitation_verified: bool
    fire_safety_verified: bool
    emergency_access_verified: bool

    assessed_by: int

    assessed_at: datetime

    notes: str = ""

    def __post_init__(self) -> None:
        validate_datetime(
            self.assessed_at,
            "assessed_at",
        )

    @property
    def safe_for_occupancy(self) -> bool:
        return (
            self.safety_level
            == ShelterSafetyLevel.VERIFIED_SAFE
            and self.structural_integrity_verified
            and self.flood_exposure_reviewed
            and self.electrical_safety_verified
            and self.sanitation_verified
            and self.fire_safety_verified
            and self.emergency_access_verified
        )


# ============================================================
# SHELTER MODELS
# ============================================================

@dataclass(frozen=True)
class ShelterCreateRequest:
    name: str
    shelter_type: ShelterType

    location: ShelterLocation

    maximum_capacity: int

    facilities: ShelterFacilityStatus

    contact_reference: str | None = None

    description: str = ""

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Shelter:
    id: str
    district_id: int

    name: str
    shelter_type: ShelterType

    location: ShelterLocation

    status: ShelterStatus

    maximum_capacity: int

    current_occupancy: int = 0
    reserved_capacity: int = 0

    facilities: ShelterFacilityStatus = field(
        default_factory=ShelterFacilityStatus
    )

    safety_assessment: ShelterSafetyAssessment | None = None

    created_by: int = 0

    verified_by: int | None = None
    verified_at: datetime | None = None

    contact_reference: str | None = None

    description: str = ""

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

    @property
    def available_capacity(self) -> int:
        return max(
            0,
            self.maximum_capacity
            - self.current_occupancy
            - self.reserved_capacity,
        )

    @property
    def occupancy_percent(self) -> float:
        if self.maximum_capacity <= 0:
            return 0.0

        return round(
            self.current_occupancy
            / self.maximum_capacity
            * 100,
            2,
        )


# ============================================================
# EVACUEE AND RESERVATION MODELS
# ============================================================

@dataclass(frozen=True)
class EvacueeGroupRequest:
    district_id: int

    group_size: int

    origin: ShelterLocation

    household_reference: str | None = None
    evacuation_id: str | None = None
    incident_id: str | None = None

    children_count: int = 0
    elderly_count: int = 0

    disabled_people_count: int = 0
    pregnant_people_count: int = 0

    medical_support_required: bool = False
    wheelchair_access_required: bool = False

    required_facilities: frozenset[
        ShelterFacility
    ] = field(
        default_factory=frozenset
    )

    notes: str = ""

    def __post_init__(self) -> None:
        if self.district_id <= 0:
            raise ShelterValidationError(
                "Invalid district ID"
            )

        validate_non_negative_integer(
            self.group_size,
            "group_size",
        )

        if self.group_size == 0:
            raise ShelterValidationError(
                "Group size must be positive"
            )

        for name in (
            "children_count",
            "elderly_count",
            "disabled_people_count",
            "pregnant_people_count",
        ):
            value = getattr(self, name)

            validate_non_negative_integer(
                value,
                name,
            )

            if value > self.group_size:
                raise ShelterValidationError(
                    f"{name} exceeds group size"
                )


@dataclass
class EvacueeGroup:
    id: str

    district_id: int

    group_size: int

    status: EvacueeStatus

    origin: ShelterLocation

    created_by: int

    household_reference: str | None = None
    evacuation_id: str | None = None
    incident_id: str | None = None

    assigned_shelter_id: str | None = None
    reservation_id: str | None = None

    children_count: int = 0
    elderly_count: int = 0
    disabled_people_count: int = 0
    pregnant_people_count: int = 0

    medical_support_required: bool = False
    wheelchair_access_required: bool = False

    required_facilities: frozenset[
        ShelterFacility
    ] = field(
        default_factory=frozenset
    )

    checked_in_at: datetime | None = None
    checked_out_at: datetime | None = None

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


@dataclass(frozen=True)
class ShelterReservation:
    id: str

    shelter_id: str
    evacuee_group_id: str

    reserved_people: int

    status: ShelterReservationStatus

    created_at: datetime

    expires_at: datetime

    confirmed_at: datetime | None = None


@dataclass(frozen=True)
class ShelterRecommendation:
    shelter_id: str
    shelter_name: str

    available_capacity: int

    distance_km: float

    suitability_score: float

    required_facilities_met: bool

    eligible: bool

    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShelterSupplyRequest:
    shelter_id: str

    category: ShelterSupplyCategory

    quantity: float
    unit: str

    reason: str

    urgent: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, (int, float))
            or not math.isfinite(self.quantity)
            or self.quantity <= 0
        ):
            raise ShelterValidationError(
                "Supply quantity must be positive"
            )

        require_text(
            self.unit,
            "Supply unit",
        )

        require_text(
            self.reason,
            "Supply reason",
        )


# ============================================================
# REPOSITORY CONTRACTS
# ============================================================

class ShelterRepository(Protocol):

    async def create(
        self,
        shelter: Shelter,
    ) -> Shelter:
        ...

    async def get_by_id(
        self,
        shelter_id: str,
    ) -> Shelter | None:
        ...

    async def update(
        self,
        shelter: Shelter,
        *,
        expected_version: int,
    ) -> Shelter:
        """
        Must enforce optimistic concurrency atomically.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[Shelter]:
        ...

    async def list_open_shelters(
        self,
        district_id: int,
    ) -> Sequence[Shelter]:
        ...

    async def reserve_capacity(
        self,
        *,
        shelter_id: str,
        group_id: str,
        people: int,
        expires_at: datetime,
    ) -> ShelterReservation:
        """
        Atomically verify availability and reserve capacity.

        Must enforce:
            occupancy + reservations + people <= capacity

        Must be idempotent for group_id.
        """
        ...

    async def release_reservation(
        self,
        reservation_id: str,
    ) -> None:
        ...

    async def confirm_check_in(
        self,
        *,
        reservation_id: str,
        group_id: str,
    ) -> None:
        """
        Atomically convert a valid reservation
        into actual occupancy.
        """
        ...

    async def confirm_check_out(
        self,
        *,
        shelter_id: str,
        group_id: str,
        people: int,
    ) -> None:
        """
        Atomically decrease occupancy once.
        Must be idempotent for group_id.
        """
        ...


class EvacueeGroupRepository(Protocol):

    async def create(
        self,
        group: EvacueeGroup,
    ) -> EvacueeGroup:
        ...

    async def get_by_id(
        self,
        group_id: str,
    ) -> EvacueeGroup | None:
        ...

    async def update(
        self,
        group: EvacueeGroup,
        *,
        expected_version: int,
    ) -> EvacueeGroup:
        ...

    async def list_by_shelter(
        self,
        shelter_id: str,
    ) -> Sequence[EvacueeGroup]:
        ...


class ShelterAuthorizationGateway(Protocol):

    async def can_create(
        self,
        *,
        user_id: int,
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

    async def can_allocate(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_manage_residents(
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


class ShelterAuditGateway(Protocol):

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


class ShelterRouteGateway(Protocol):

    async def get_verified_route(
        self,
        *,
        origin: ShelterLocation,
        destination: ShelterLocation,
    ) -> tuple[float, float] | None:
        """
        Return verified (distance_km, travel_minutes).

        None means that no verified safe route exists.
        """
        ...


class ShelterSupplyGateway(Protocol):

    async def submit_supply_request(
        self,
        request: ShelterSupplyRequest,
        *,
        requested_by: int,
    ) -> str:
        """
        Submit request to inventory/logistics workflow.

        A returned ID confirms request registration,
        not delivery or allocation.
        """
        ...


# ============================================================
# SHELTER MATCHER
# ============================================================

class ShelterMatcher:

    def required_facilities(
        self,
        request: EvacueeGroupRequest,
    ) -> frozenset[ShelterFacility]:

        required = set(
            request.required_facilities
        )

        if request.wheelchair_access_required:
            required.update({
                ShelterFacility.WHEELCHAIR_ACCESS,
                ShelterFacility.ACCESSIBLE_TOILET,
            })

        if request.medical_support_required:
            required.add(
                ShelterFacility.MEDICAL_SUPPORT
            )

        if request.children_count > 0:
            required.add(
                ShelterFacility.CHILD_FRIENDLY_SPACE
            )

        return frozenset(required)

    def evaluate(
        self,
        *,
        shelter: Shelter,
        request: EvacueeGroupRequest,
        distance_km: float,
        verified_route: bool,
    ) -> ShelterRecommendation:

        reasons: list[str] = []

        required = self.required_facilities(
            request
        )

        available = (
            shelter.facilities.available_facilities
        )

        facilities_met = required.issubset(
            available
        )

        if not facilities_met:
            missing = required - available

            reasons.append(
                "Missing required facilities: "
                + ", ".join(
                    sorted(
                        facility.value
                        for facility in missing
                    )
                )
            )

        if shelter.status != ShelterStatus.OPEN:
            reasons.append(
                "Shelter is not open"
            )

        if (
            shelter.safety_assessment is None
            or not shelter.safety_assessment.safe_for_occupancy
        ):
            reasons.append(
                "Shelter safety is not verified"
            )

        if not shelter.facilities.verified:
            reasons.append(
                "Shelter facilities are not verified"
            )

        if (
            shelter.available_capacity
            < request.group_size
        ):
            reasons.append(
                "Insufficient available capacity"
            )

        if not verified_route:
            reasons.append(
                "No verified safe access route"
            )

        eligible = (
            shelter.status == ShelterStatus.OPEN
            and shelter.safety_assessment is not None
            and shelter.safety_assessment.safe_for_occupancy
            and shelter.facilities.verified
            and facilities_met
            and shelter.available_capacity
            >= request.group_size
            and verified_route
        )

        capacity_score = (
            shelter.available_capacity
            / max(
                shelter.maximum_capacity,
                1,
            )
        )

        distance_score = (
            1.0
            / (1.0 + distance_km / 10.0)
        )

        facility_score = (
            1.0
            if facilities_met
            else 0.0
        )

        suitability = (
            0.40 * facility_score
            + 0.30 * capacity_score
            + 0.30 * distance_score
        )

        if not eligible:
            suitability = 0.0

        return ShelterRecommendation(
            shelter_id=shelter.id,
            shelter_name=shelter.name,
            available_capacity=shelter.available_capacity,
            distance_km=round(distance_km, 2),
            suitability_score=round(
                suitability,
                4,
            ),
            required_facilities_met=facilities_met,
            eligible=eligible,
            reasons=tuple(reasons),
        )


# ============================================================
# SHELTER SERVICE
# ============================================================

class ShelterService:

    RESERVATION_DURATION = timedelta(
        minutes=45
    )

    def __init__(
        self,
        *,
        shelter_repository: ShelterRepository,
        evacuee_repository: EvacueeGroupRepository,
        authorization_gateway: ShelterAuthorizationGateway,
        route_gateway: ShelterRouteGateway,
        supply_gateway: ShelterSupplyGateway | None = None,
        audit_gateway: ShelterAuditGateway | None = None,
        matcher: ShelterMatcher | None = None,
    ):
        self.shelters = shelter_repository
        self.evacuees = evacuee_repository

        self.authorization = authorization_gateway
        self.routes = route_gateway
        self.supplies = supply_gateway
        self.audit = audit_gateway

        self.matcher = (
            matcher
            or ShelterMatcher()
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

    async def get_shelter(
        self,
        shelter_id: str,
    ) -> Shelter:

        shelter = await self.shelters.get_by_id(
            shelter_id
        )

        if shelter is None:
            raise ShelterNotFoundError(
                "Shelter not found"
            )

        return shelter

    async def get_evacuee_group(
        self,
        group_id: str,
    ) -> EvacueeGroup:

        group = await self.evacuees.get_by_id(
            group_id
        )

        if group is None:
            raise ShelterNotFoundError(
                "Evacuee group not found"
            )

        return group

    async def register_shelter(
        self,
        request: ShelterCreateRequest,
        *,
        actor_id: int,
    ) -> Shelter:

        district_id = request.location.district_id

        if not await self.authorization.can_create(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter registration denied"
            )

        validate_non_negative_integer(
            request.maximum_capacity,
            "maximum_capacity",
        )

        if request.maximum_capacity == 0:
            raise ShelterValidationError(
                "Shelter capacity must be positive"
            )

        shelter = Shelter(
            id=str(uuid.uuid4()),
            district_id=district_id,
            name=require_text(
                request.name,
                "Shelter name",
            ),
            shelter_type=request.shelter_type,
            location=request.location,
            status=ShelterStatus.DRAFT,
            maximum_capacity=request.maximum_capacity,
            facilities=request.facilities,
            created_by=actor_id,
            contact_reference=request.contact_reference,
            description=request.description.strip(),
            metadata=dict(request.metadata),
        )

        created = await self.shelters.create(
            shelter
        )

        await self._audit(
            "shelter.created",
            actor_id=actor_id,
            entity_type="shelter",
            entity_id=created.id,
            district_id=district_id,
        )

        return created

    async def submit_shelter_for_verification(
        self,
        shelter_id: str,
        *,
        actor_id: int,
    ) -> Shelter:

        shelter = await self.get_shelter(
            shelter_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=shelter.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter submission denied"
            )

        if shelter.status != ShelterStatus.DRAFT:
            raise ShelterConflictError(
                "Only draft shelters can be submitted"
            )

        updated = replace(
            shelter,
            status=ShelterStatus.PENDING_VERIFICATION,
            updated_at=utc_now(),
            version=shelter.version + 1,
        )

        saved = await self.shelters.update(
            updated,
            expected_version=shelter.version,
        )

        await self._audit(
            "shelter.submitted_for_verification",
            actor_id=actor_id,
            entity_type="shelter",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def verify_shelter(
        self,
        shelter_id: str,
        *,
        reviewer_id: int,
        safety_assessment: ShelterSafetyAssessment,
    ) -> Shelter:

        shelter = await self.get_shelter(
            shelter_id
        )

        if not await self.authorization.can_verify(
            user_id=reviewer_id,
            district_id=shelter.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter verification denied"
            )

        if (
            shelter.status
            != ShelterStatus.PENDING_VERIFICATION
        ):
            raise ShelterConflictError(
                "Shelter is not awaiting verification"
            )

        if reviewer_id == shelter.created_by:
            raise ShelterAuthorizationError(
                "Independent verification required"
            )

        if safety_assessment.assessed_by != reviewer_id:
            raise ShelterValidationError(
                "Safety assessor mismatch"
            )

        if not safety_assessment.safe_for_occupancy:
            raise ShelterSafetyError(
                "Shelter is not verified safe"
            )

        if not shelter.facilities.verified:
            raise ShelterSafetyError(
                "Facility verification is incomplete"
            )

        if (
            shelter.current_occupancy
            + shelter.reserved_capacity
            > shelter.maximum_capacity
        ):
            raise ShelterCapacityError(
                "Shelter exceeds verified capacity"
            )

        now = utc_now()

        updated = replace(
            shelter,
            status=ShelterStatus.OPEN,
            safety_assessment=safety_assessment,
            verified_by=reviewer_id,
            verified_at=now,
            updated_at=now,
            version=shelter.version + 1,
        )

        saved = await self.shelters.update(
            updated,
            expected_version=shelter.version,
        )

        await self._audit(
            "shelter.verified",
            actor_id=reviewer_id,
            entity_type="shelter",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def update_shelter_status(
        self,
        shelter_id: str,
        *,
        status: ShelterStatus,
        actor_id: int,
        reason: str,
    ) -> Shelter:

        shelter = await self.get_shelter(
            shelter_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=shelter.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter status update denied"
            )

        reason = require_text(
            reason,
            "Status change reason",
        )

        if status == ShelterStatus.OPEN:
            if (
                shelter.safety_assessment is None
                or not shelter.safety_assessment.safe_for_occupancy
                or not shelter.facilities.verified
            ):
                raise ShelterSafetyError(
                    "Shelter requires safety verification"
                )

        if status == ShelterStatus.CLOSED:
            if shelter.current_occupancy > 0:
                raise ShelterConflictError(
                    "Cannot close an occupied shelter "
                    "without resident transfer"
                )

        updated = replace(
            shelter,
            status=status,
            updated_at=utc_now(),
            version=shelter.version + 1,
            metadata={
                **shelter.metadata,
                "last_status_reason": reason,
            },
        )

        saved = await self.shelters.update(
            updated,
            expected_version=shelter.version,
        )

        await self._audit(
            "shelter.status_updated",
            actor_id=actor_id,
            entity_type="shelter",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def register_evacuee_group(
        self,
        request: EvacueeGroupRequest,
        *,
        actor_id: int,
    ) -> EvacueeGroup:

        if not await self.authorization.can_manage_residents(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise ShelterAuthorizationError(
                "Evacuee registration denied"
            )

        if request.origin.district_id != request.district_id:
            raise ShelterValidationError(
                "Evacuee district mismatch"
            )

        group = EvacueeGroup(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            group_size=request.group_size,
            status=EvacueeStatus.REGISTERED,
            origin=request.origin,
            created_by=actor_id,
            household_reference=request.household_reference,
            evacuation_id=request.evacuation_id,
            incident_id=request.incident_id,
            children_count=request.children_count,
            elderly_count=request.elderly_count,
            disabled_people_count=(
                request.disabled_people_count
            ),
            pregnant_people_count=(
                request.pregnant_people_count
            ),
            medical_support_required=(
                request.medical_support_required
            ),
            wheelchair_access_required=(
                request.wheelchair_access_required
            ),
            required_facilities=request.required_facilities,
        )

        created = await self.evacuees.create(
            group
        )

        await self._audit(
            "evacuee_group.registered",
            actor_id=actor_id,
            entity_type="evacuee_group",
            entity_id=created.id,
            district_id=created.district_id,
        )

        return created

    @staticmethod
    def _group_request(
        group: EvacueeGroup,
    ) -> EvacueeGroupRequest:

        return EvacueeGroupRequest(
            district_id=group.district_id,
            group_size=group.group_size,
            origin=group.origin,
            household_reference=group.household_reference,
            evacuation_id=group.evacuation_id,
            incident_id=group.incident_id,
            children_count=group.children_count,
            elderly_count=group.elderly_count,
            disabled_people_count=(
                group.disabled_people_count
            ),
            pregnant_people_count=(
                group.pregnant_people_count
            ),
            medical_support_required=(
                group.medical_support_required
            ),
            wheelchair_access_required=(
                group.wheelchair_access_required
            ),
            required_facilities=group.required_facilities,
        )

    async def recommend_shelters(
        self,
        group_id: str,
        *,
        actor_id: int,
    ) -> list[ShelterRecommendation]:

        group = await self.get_evacuee_group(
            group_id
        )

        if not await self.authorization.can_allocate(
            user_id=actor_id,
            district_id=group.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter matching denied"
            )

        shelters = await self.shelters.list_open_shelters(
            group.district_id
        )

        request = self._group_request(
            group
        )

        recommendations: list[
            ShelterRecommendation
        ] = []

        for shelter in shelters:
            route = await self.routes.get_verified_route(
                origin=group.origin,
                destination=shelter.location,
            )

            verified_route = route is not None

            if route is not None:
                distance_km, travel_minutes = route

                if (
                    not math.isfinite(distance_km)
                    or distance_km < 0
                    or not math.isfinite(travel_minutes)
                    or travel_minutes < 0
                ):
                    raise ShelterIntegrationError(
                        "Route provider returned invalid values"
                    )

            else:
                distance_km = haversine_distance_km(
                    group.origin.latitude,
                    group.origin.longitude,
                    shelter.location.latitude,
                    shelter.location.longitude,
                )

            recommendation = self.matcher.evaluate(
                shelter=shelter,
                request=request,
                distance_km=distance_km,
                verified_route=verified_route,
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

    async def reserve_shelter(
        self,
        group_id: str,
        *,
        shelter_id: str,
        actor_id: int,
    ) -> EvacueeGroup:

        group = await self.get_evacuee_group(
            group_id
        )

        shelter = await self.get_shelter(
            shelter_id
        )

        if not await self.authorization.can_allocate(
            user_id=actor_id,
            district_id=group.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter allocation denied"
            )

        if group.status != EvacueeStatus.REGISTERED:
            raise ShelterConflictError(
                "Group is not awaiting allocation"
            )

        if shelter.status != ShelterStatus.OPEN:
            raise ShelterSafetyError(
                "Shelter is not open"
            )

        if (
            shelter.safety_assessment is None
            or not shelter.safety_assessment.safe_for_occupancy
        ):
            raise ShelterSafetyError(
                "Shelter safety is not verified"
            )

        if shelter.available_capacity < group.group_size:
            raise ShelterCapacityError(
                "Insufficient shelter capacity"
            )

        required = self.matcher.required_facilities(
            self._group_request(group)
        )

        if not required.issubset(
            shelter.facilities.available_facilities
        ):
            raise ShelterSafetyError(
                "Shelter lacks required accessibility "
                "or support facilities"
            )

        route = await self.routes.get_verified_route(
            origin=group.origin,
            destination=shelter.location,
        )

        if route is None:
            raise ShelterSafetyError(
                "No verified safe route to shelter"
            )

        reservation = await self.shelters.reserve_capacity(
            shelter_id=shelter.id,
            group_id=group.id,
            people=group.group_size,
            expires_at=(
                utc_now()
                + self.RESERVATION_DURATION
            ),
        )

        updated = replace(
            group,
            status=EvacueeStatus.RESERVED,
            assigned_shelter_id=shelter.id,
            reservation_id=reservation.id,
            updated_at=utc_now(),
            version=group.version + 1,
        )

        try:
            saved = await self.evacuees.update(
                updated,
                expected_version=group.version,
            )

        except Exception:
            try:
                await self.shelters.release_reservation(
                    reservation.id
                )
            except Exception:
                logger.exception(
                    "Reservation compensation failed: %s",
                    reservation.id,
                )
            raise

        await self._audit(
            "shelter.capacity_reserved",
            actor_id=actor_id,
            entity_type="evacuee_group",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def check_in_group(
        self,
        group_id: str,
        *,
        actor_id: int,
    ) -> EvacueeGroup:

        group = await self.get_evacuee_group(
            group_id
        )

        if not await self.authorization.can_manage_residents(
            user_id=actor_id,
            district_id=group.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter check-in denied"
            )

        if group.status != EvacueeStatus.RESERVED:
            raise ShelterConflictError(
                "Group has no active reservation"
            )

        if (
            group.reservation_id is None
            or group.assigned_shelter_id is None
        ):
            raise ShelterConflictError(
                "Reservation details are incomplete"
            )

        shelter = await self.get_shelter(
            group.assigned_shelter_id
        )

        if shelter.status not in {
            ShelterStatus.OPEN,
            ShelterStatus.AT_CAPACITY,
        }:
            raise ShelterSafetyError(
                "Shelter is not accepting check-ins"
            )

        await self.shelters.confirm_check_in(
            reservation_id=group.reservation_id,
            group_id=group.id,
        )

        updated = replace(
            group,
            status=EvacueeStatus.CHECKED_IN,
            checked_in_at=utc_now(),
            updated_at=utc_now(),
            version=group.version + 1,
        )

        # Repository implementations should coordinate
        # occupancy and group-state updates through a
        # transaction or recoverable outbox workflow.
        saved = await self.evacuees.update(
            updated,
            expected_version=group.version,
        )

        await self._audit(
            "evacuee_group.checked_in",
            actor_id=actor_id,
            entity_type="evacuee_group",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def check_out_group(
        self,
        group_id: str,
        *,
        actor_id: int,
    ) -> EvacueeGroup:

        group = await self.get_evacuee_group(
            group_id
        )

        if not await self.authorization.can_manage_residents(
            user_id=actor_id,
            district_id=group.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter check-out denied"
            )

        if group.status != EvacueeStatus.CHECKED_IN:
            raise ShelterConflictError(
                "Group is not checked in"
            )

        if group.assigned_shelter_id is None:
            raise ShelterConflictError(
                "Assigned shelter is missing"
            )

        await self.shelters.confirm_check_out(
            shelter_id=group.assigned_shelter_id,
            group_id=group.id,
            people=group.group_size,
        )

        updated = replace(
            group,
            status=EvacueeStatus.CHECKED_OUT,
            checked_out_at=utc_now(),
            updated_at=utc_now(),
            version=group.version + 1,
        )

        saved = await self.evacuees.update(
            updated,
            expected_version=group.version,
        )

        await self._audit(
            "evacuee_group.checked_out",
            actor_id=actor_id,
            entity_type="evacuee_group",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def cancel_reservation(
        self,
        group_id: str,
        *,
        actor_id: int,
    ) -> EvacueeGroup:

        group = await self.get_evacuee_group(
            group_id
        )

        if not await self.authorization.can_allocate(
            user_id=actor_id,
            district_id=group.district_id,
        ):
            raise ShelterAuthorizationError(
                "Reservation cancellation denied"
            )

        if group.status != EvacueeStatus.RESERVED:
            raise ShelterConflictError(
                "Group has no cancellable reservation"
            )

        if group.reservation_id is None:
            raise ShelterConflictError(
                "Reservation ID is missing"
            )

        await self.shelters.release_reservation(
            group.reservation_id
        )

        updated = replace(
            group,
            status=EvacueeStatus.REGISTERED,
            assigned_shelter_id=None,
            reservation_id=None,
            updated_at=utc_now(),
            version=group.version + 1,
        )

        saved = await self.evacuees.update(
            updated,
            expected_version=group.version,
        )

        await self._audit(
            "shelter.reservation_cancelled",
            actor_id=actor_id,
            entity_type="evacuee_group",
            entity_id=saved.id,
            district_id=saved.district_id,
        )

        return saved

    async def request_supplies(
        self,
        request: ShelterSupplyRequest,
        *,
        actor_id: int,
    ) -> str:

        shelter = await self.get_shelter(
            request.shelter_id
        )

        if not await self.authorization.can_manage(
            user_id=actor_id,
            district_id=shelter.district_id,
        ):
            raise ShelterAuthorizationError(
                "Supply request permission denied"
            )

        if self.supplies is None:
            raise ShelterIntegrationError(
                "Supply workflow is not configured"
            )

        reference = (
            await self.supplies.submit_supply_request(
                request,
                requested_by=actor_id,
            )
        )

        await self._audit(
            "shelter.supplies_requested",
            actor_id=actor_id,
            entity_type="shelter",
            entity_id=shelter.id,
            district_id=shelter.district_id,
        )

        return reference

    async def get_shelter_alerts(
        self,
        shelter_id: str,
        *,
        actor_id: int,
    ) -> list[dict[str, Any]]:

        shelter = await self.get_shelter(
            shelter_id
        )

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=shelter.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter alert access denied"
            )

        alerts: list[dict[str, Any]] = []

        def add_alert(
            alert_type: ShelterAlertType,
            message: str,
        ) -> None:
            alerts.append({
                "type": alert_type.value,
                "message": message,
                "shelter_id": shelter.id,
                "evaluated_at": utc_now().isoformat(),
            })

        if shelter.status == ShelterStatus.UNSAFE:
            add_alert(
                ShelterAlertType.UNSAFE_SHELTER,
                "Shelter is marked unsafe",
            )

        if shelter.available_capacity == 0:
            add_alert(
                ShelterAlertType.FULL_CAPACITY,
                "No remaining shelter capacity",
            )

        elif shelter.occupancy_percent >= 85:
            add_alert(
                ShelterAlertType.HIGH_OCCUPANCY,
                "Shelter occupancy is above 85%",
            )

        if not shelter.facilities.drinking_water_available:
            add_alert(
                ShelterAlertType.WATER_SHORTAGE,
                "Drinking water availability requires review",
            )

        if not shelter.facilities.food_available:
            add_alert(
                ShelterAlertType.FOOD_SHORTAGE,
                "Food availability requires review",
            )

        if shelter.facilities.toilets_count == 0:
            add_alert(
                ShelterAlertType.SANITATION_ISSUE,
                "No verified toilets recorded",
            )

        if (
            shelter.facilities.verified_at is None
            or utc_now()
            - shelter.facilities.verified_at
            > timedelta(hours=12)
        ):
            add_alert(
                ShelterAlertType.DATA_STALE,
                "Facility information requires re-verification",
            )

        return alerts

    async def list_district_shelters(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> Sequence[Shelter]:

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=district_id,
        ):
            raise ShelterAuthorizationError(
                "District shelter access denied"
            )

        return await self.shelters.list_by_district(
            district_id
        )

    async def get_district_statistics(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        shelters = await self.list_district_shelters(
            district_id,
            actor_id=actor_id,
        )

        status_counts = {
            status.value: 0
            for status in ShelterStatus
        }

        for shelter in shelters:
            status_counts[
                shelter.status.value
            ] += 1

        operational = [
            shelter
            for shelter in shelters
            if shelter.status in {
                ShelterStatus.OPEN,
                ShelterStatus.AT_CAPACITY,
            }
        ]

        return {
            "district_id": district_id,
            "total_shelters": len(shelters),
            "operational_shelters": len(operational),
            "total_capacity": sum(
                shelter.maximum_capacity
                for shelter in operational
            ),
            "current_occupancy": sum(
                shelter.current_occupancy
                for shelter in operational
            ),
            "reserved_capacity": sum(
                shelter.reserved_capacity
                for shelter in operational
            ),
            "available_capacity": sum(
                shelter.available_capacity
                for shelter in operational
            ),
            "status_counts": status_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_shelter_report(
        self,
        shelter_id: str,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        shelter = await self.get_shelter(
            shelter_id
        )

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=shelter.district_id,
        ):
            raise ShelterAuthorizationError(
                "Shelter report access denied"
            )

        return {
            "shelter_id": shelter.id,
            "district_id": shelter.district_id,
            "name": shelter.name,
            "type": shelter.shelter_type.value,
            "status": shelter.status.value,
            "location": {
                "latitude": shelter.location.latitude,
                "longitude": shelter.location.longitude,
                "address": shelter.location.address,
                "panchayat": shelter.location.panchayat,
                "taluk": shelter.location.taluk,
            },
            "capacity": {
                "maximum": shelter.maximum_capacity,
                "occupied": shelter.current_occupancy,
                "reserved": shelter.reserved_capacity,
                "available": shelter.available_capacity,
                "occupancy_percent": (
                    shelter.occupancy_percent
                ),
            },
            "facilities": {
                "available": sorted(
                    facility.value
                    for facility in (
                        shelter.facilities.available_facilities
                    )
                ),
                "toilets_count": (
                    shelter.facilities.toilets_count
                ),
                "beds_count": (
                    shelter.facilities.beds_count
                ),
                "drinking_water_available": (
                    shelter.facilities.drinking_water_available
                ),
                "food_available": (
                    shelter.facilities.food_available
                ),
                "medical_staff_available": (
                    shelter.facilities.medical_staff_available
                ),
                "verified": shelter.facilities.verified,
            },
            "safety": (
                {
                    "level": (
                        shelter.safety_assessment
                        .safety_level.value
                    ),
                    "safe_for_occupancy": (
                        shelter.safety_assessment
                        .safe_for_occupancy
                    ),
                    "assessed_at": (
                        shelter.safety_assessment
                        .assessed_at.isoformat()
                    ),
                }
                if shelter.safety_assessment
                else None
            ),
            "verified_by": shelter.verified_by,
            "verified_at": (
                shelter.verified_at.isoformat()
                if shelter.verified_at
                else None
            ),
            "generated_at": utc_now().isoformat(),
            "disclaimer": (
                "Shelter status and capacity require "
                "current field verification. Allocation "
                "and evacuation decisions remain subject "
                "to authorized human oversight."
            ),
        }
