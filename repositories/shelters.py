from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/repositories/shelters.py

Async SQLAlchemy 2.x shelter repository.

Features:
- Shelter registration and retrieval
- Shelter verification and status management
- District-based shelter searches
- Capacity and occupancy management
- Atomic capacity reservations
- Resident registration
- Human-approved shelter assignments
- Resident check-in and check-out
- Resident transfer between shelters
- Available shelter recommendations
- Shelter occupancy analytics
- Row-level locking
- Transaction-safe database operations

Expected ORM models:
    backend.models.shelter.Shelter
    backend.models.shelter_resident.ShelterResident

Expected Shelter columns:
    id
    name
    district
    address
    latitude
    longitude
    status
    capacity
    occupied_count
    reserved_count
    facilities
    is_verified
    is_deleted
    created_at
    updated_at

Expected ShelterResident columns:
    id
    shelter_id
    full_name
    family_id
    status
    checked_in_at
    checked_out_at
    created_at
    updated_at

Additional required ORM model:
    backend.models.shelter.ShelterReservation

Expected ShelterReservation columns:
    id
    shelter_id
    resident_id
    family_id
    people_count
    status
    approved_by
    created_at
    updated_at

IMPORTANT:
- Capacity reservations are not physical check-ins.
- Shelter safety must be verified by authorized personnel.
- AI recommendations never approve assignments.
- Never exceed verified shelter capacity.
- Sensitive resident information must remain private.
- Repository methods flush but do not commit.
"""


import logging
import math
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.shelter import (
    Shelter,
    ShelterReservation,
)
from backend.models.shelter_resident import ShelterResident


logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

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

SHELTER_OPEN = "open"
SHELTER_FULL = "full"
SHELTER_CLOSED = "closed"

RESERVATION_ACTIVE = "active"
RESERVATION_FULFILLED = "fulfilled"
RESERVATION_CANCELLED = "cancelled"

RESIDENT_REGISTERED = "registered"
RESIDENT_CHECKED_IN = "checked_in"
RESIDENT_CHECKED_OUT = "checked_out"
RESIDENT_TRANSFERRED = "transferred"


# ============================================================
# EXCEPTIONS
# ============================================================

class ShelterRepositoryError(Exception):
    """Base shelter repository exception."""


class ShelterNotFoundError(ShelterRepositoryError):
    """Shelter or resident not found."""


class ShelterCapacityError(ShelterRepositoryError):
    """Insufficient shelter capacity."""


class ShelterConflictError(ShelterRepositoryError):
    """Conflicting shelter or resident operation."""


class ShelterAuthorizationError(ShelterRepositoryError):
    """Required human authorization is missing."""


class ShelterStateError(ShelterRepositoryError):
    """Invalid shelter state transition."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_district(value: str) -> str:
    district = value.strip().lower()

    if district not in KERALA_DISTRICTS:
        raise ValueError(
            f"Invalid Kerala district: {value}"
        )

    return district


def validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude")

    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude")


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Straight-line distance, not travel distance.
    """

    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

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

    a = min(1.0, max(0.0, a))

    return 2 * radius * math.asin(math.sqrt(a))


def validate_pagination(
    limit: int,
    offset: int,
) -> None:
    if not 1 <= limit <= 500:
        raise ValueError(
            "limit must be between 1 and 500"
        )

    if offset < 0:
        raise ValueError(
            "offset cannot be negative"
        )


# ============================================================
# SHELTER REPOSITORY
# ============================================================

class ShelterRepository:
    """
    Shelter persistence and capacity management.

    Transaction ownership:
        The service layer commits or rolls back.

    Example:

        async with transaction() as session:
            repo = ShelterRepository(session)

            reservation = await repo.reserve_capacity(
                shelter_id=shelter_id,
                people_count=4,
                approved_by=operator_id,
                authorization_verified=True,
                family_id=family_id,
            )

    Do not share AsyncSession across concurrent tasks.
    """

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _not_deleted():
        return Shelter.is_deleted.is_(False)

    @staticmethod
    def _free_capacity(
        shelter: Shelter,
    ) -> int:
        capacity = shelter.capacity
        occupied = shelter.occupied_count
        reserved = shelter.reserved_count

        if min(capacity, occupied, reserved) < 0:
            raise ShelterConflictError(
                "Negative shelter capacity values"
            )

        if occupied + reserved > capacity:
            raise ShelterConflictError(
                "Shelter occupancy exceeds capacity"
            )

        return capacity - occupied - reserved

    @staticmethod
    def _update_full_status(
        shelter: Shelter,
    ) -> None:
        """
        Update full/open status only.

        Never automatically reopen a closed or
        unsafe shelter.
        """

        if shelter.status == SHELTER_CLOSED:
            return

        if (
            shelter.occupied_count
            + shelter.reserved_count
            >= shelter.capacity
        ):
            shelter.status = SHELTER_FULL

        elif shelter.status == SHELTER_FULL:
            shelter.status = SHELTER_OPEN

    async def _get_locked_shelter(
        self,
        shelter_id: UUID,
    ) -> Shelter:

        statement = (
            select(Shelter)
            .where(
                Shelter.id == shelter_id,
                self._not_deleted(),
            )
            .with_for_update()
        )

        result = await self.session.execute(
            statement
        )

        shelter = result.scalar_one_or_none()

        if shelter is None:
            raise ShelterNotFoundError(
                "Shelter not found"
            )

        return shelter

    # ========================================================
    # CREATE SHELTER
    # ========================================================

    async def create(
        self,
        *,
        name: str,
        district: str,
        address: str,
        capacity: int,
        facilities: list[str] | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> Shelter:
        """
        Register a new shelter.

        New shelters remain unverified and closed
        until authorized safety inspection.
        """

        if not name.strip():
            raise ValueError(
                "Shelter name is required"
            )

        if capacity <= 0:
            raise ValueError(
                "Capacity must be positive"
            )

        if (
            latitude is None
        ) != (
            longitude is None
        ):
            raise ValueError(
                "Coordinates must be provided together"
            )

        if latitude is not None:
            validate_coordinates(
                latitude,
                longitude,
            )

        shelter = Shelter(
            name=name.strip(),
            district=normalize_district(district),
            address=address.strip(),
            latitude=latitude,
            longitude=longitude,
            capacity=capacity,
            occupied_count=0,
            reserved_count=0,
            facilities=facilities or [],
            status=SHELTER_CLOSED,
            is_verified=False,
            is_deleted=False,
        )

        self.session.add(shelter)

        await self.session.flush()

        return shelter

    # ========================================================
    # GET SHELTER
    # ========================================================

    async def get_by_id(
        self,
        shelter_id: UUID,
    ) -> Shelter | None:

        statement = select(Shelter).where(
            Shelter.id == shelter_id,
            self._not_deleted(),
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # LIST SHELTERS
    # ========================================================

    async def list_shelters(
        self,
        *,
        district: str | None = None,
        status: str | None = None,
        verified_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Shelter]:

        validate_pagination(limit, offset)

        statement = select(Shelter).where(
            self._not_deleted()
        )

        if district is not None:
            statement = statement.where(
                Shelter.district
                == normalize_district(district)
            )

        if status is not None:
            statement = statement.where(
                Shelter.status == status
            )

        if verified_only:
            statement = statement.where(
                Shelter.is_verified.is_(True)
            )

        statement = (
            statement
            .order_by(
                Shelter.name.asc(),
                Shelter.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # OPEN SHELTER
    # ========================================================

    async def open_shelter(
        self,
        *,
        shelter_id: UUID,
        authorized_by: UUID,
        safety_verified: bool,
    ) -> Shelter:
        """
        Open a shelter after authorized safety review.

        The service must independently verify
        operator permissions and inspection evidence.
        """

        if not safety_verified:
            raise ShelterAuthorizationError(
                "Safety verification is required"
            )

        shelter = await self._get_locked_shelter(
            shelter_id
        )

        if not shelter.is_verified:
            raise ShelterStateError(
                "Unverified shelter cannot open"
            )

        shelter.status = SHELTER_OPEN

        self._update_full_status(shelter)

        await self.session.flush()

        logger.info(
            "Shelter %s opened by %s",
            shelter_id,
            authorized_by,
        )

        return shelter

    # ========================================================
    # CLOSE SHELTER
    # ========================================================

    async def close_shelter(
        self,
        *,
        shelter_id: UUID,
        authorized_by: UUID,
        reason: str,
    ) -> Shelter:
        """
        Close shelter for new admissions.

        Existing residents require a separate,
        coordinated evacuation or transfer plan.
        """

        if not reason.strip():
            raise ValueError(
                "Closure reason is required"
            )

        shelter = await self._get_locked_shelter(
            shelter_id
        )

        shelter.status = SHELTER_CLOSED

        await self.session.flush()

        logger.warning(
            "Shelter %s closed by %s",
            shelter_id,
            authorized_by,
        )

        return shelter

    # ========================================================
    # UPDATE CAPACITY
    # ========================================================

    async def update_capacity(
        self,
        *,
        shelter_id: UUID,
        new_capacity: int,
        authorized_by: UUID,
        capacity_verified: bool,
    ) -> Shelter:
        """
        Update verified physical capacity.

        Cannot reduce capacity below occupied
        and reserved places.
        """

        if not capacity_verified:
            raise ShelterAuthorizationError(
                "Capacity verification required"
            )

        if new_capacity <= 0:
            raise ValueError(
                "Capacity must be positive"
            )

        shelter = await self._get_locked_shelter(
            shelter_id
        )

        committed_places = (
            shelter.occupied_count
            + shelter.reserved_count
        )

        if new_capacity < committed_places:
            raise ShelterCapacityError(
                "New capacity is below committed places"
            )

        shelter.capacity = new_capacity

        self._update_full_status(shelter)

        await self.session.flush()

        return shelter

    # ========================================================
    # AVAILABLE SHELTERS
    # ========================================================

    async def get_available_shelters(
        self,
        *,
        district: str | None = None,
        minimum_places: int = 1,
        required_facilities: list[str] | None = None,
        limit: int = 100,
    ) -> Sequence[Shelter]:
        """
        Return verified open shelters with capacity.

        Facilities are filtered in Python to support
        either ARRAY or JSON-backed ORM columns.
        """

        if minimum_places <= 0:
            raise ValueError(
                "minimum_places must be positive"
            )

        validate_pagination(limit, 0)

        free_capacity = (
            Shelter.capacity
            - Shelter.occupied_count
            - Shelter.reserved_count
        )

        statement = select(Shelter).where(
            self._not_deleted(),
            Shelter.is_verified.is_(True),
            Shelter.status == SHELTER_OPEN,
            free_capacity >= minimum_places,
        )

        if district is not None:
            statement = statement.where(
                Shelter.district
                == normalize_district(district)
            )

        result = await self.session.execute(
            statement.order_by(
                Shelter.id.asc()
            )
        )

        required = set(
            required_facilities or []
        )

        shelters = [
            shelter
            for shelter in result.scalars().all()
            if required.issubset(
                set(shelter.facilities or [])
            )
        ]

        return shelters[:limit]

    # ========================================================
    # NEAREST AVAILABLE SHELTERS
    # ========================================================

    async def find_nearest_shelters(
        self,
        *,
        latitude: float,
        longitude: float,
        district: str | None = None,
        people_count: int = 1,
        required_facilities: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Rank shelters by straight-line distance.

        Recommendations do not guarantee:
        - Safe travel route
        - Transport availability
        - Actual shelter admission
        """

        validate_coordinates(
            latitude,
            longitude,
        )

        validate_pagination(limit, 0)

        shelters = await self.get_available_shelters(
            district=district,
            minimum_places=people_count,
            required_facilities=required_facilities,
            limit=500,
        )

        candidates = []

        for shelter in shelters:
            if (
                shelter.latitude is None
                or shelter.longitude is None
            ):
                continue

            distance = haversine_distance_km(
                latitude,
                longitude,
                shelter.latitude,
                shelter.longitude,
            )

            candidates.append(
                {
                    "shelter": shelter,
                    "distance_km": round(
                        distance,
                        3,
                    ),
                    "free_places": self._free_capacity(
                        shelter
                    ),
                    "route_verified": False,
                    "assignment_approved": False,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["distance_km"],
                str(item["shelter"].id),
            )
        )

        return candidates[:limit]

    # ========================================================
    # RESERVE CAPACITY
    # ========================================================

    async def reserve_capacity(
        self,
        *,
        shelter_id: UUID,
        people_count: int,
        approved_by: UUID,
        authorization_verified: bool,
        resident_id: UUID | None = None,
        family_id: UUID | None = None,
    ) -> ShelterReservation:
        """
        Reserve shelter places.

        Human approval is mandatory.

        A reservation does not mean people
        have physically arrived.
        """

        if not authorization_verified:
            raise ShelterAuthorizationError(
                "Human approval required"
            )

        if people_count <= 0:
            raise ValueError(
                "people_count must be positive"
            )

        shelter = await self._get_locked_shelter(
            shelter_id
        )

        if (
            not shelter.is_verified
            or shelter.status != SHELTER_OPEN
        ):
            raise ShelterStateError(
                "Shelter is not verified and open"
            )

        if people_count > self._free_capacity(shelter):
            raise ShelterCapacityError(
                "Insufficient available shelter places"
            )

        shelter.reserved_count += people_count

        reservation = ShelterReservation(
            shelter_id=shelter_id,
            resident_id=resident_id,
            family_id=family_id,
            people_count=people_count,
            status=RESERVATION_ACTIVE,
            approved_by=approved_by,
        )

        self.session.add(reservation)

        self._update_full_status(shelter)

        await self.session.flush()

        return reservation

    # ========================================================
    # CANCEL RESERVATION
    # ========================================================

    async def cancel_reservation(
        self,
        *,
        reservation_id: UUID,
    ) -> ShelterReservation:
        """
        Release reserved places.

        Shelter is locked before the reservation
        to maintain consistent lock ordering.
        """

        lookup = await self.session.execute(
            select(
                ShelterReservation.shelter_id
            ).where(
                ShelterReservation.id == reservation_id
            )
        )

        shelter_id = lookup.scalar_one_or_none()

        if shelter_id is None:
            raise ShelterNotFoundError(
                "Reservation not found"
            )

        shelter = await self._get_locked_shelter(
            shelter_id
        )

        result = await self.session.execute(
            select(ShelterReservation)
            .where(
                ShelterReservation.id
                == reservation_id
            )
            .with_for_update()
        )

        reservation = result.scalar_one_or_none()

        if reservation is None:
            raise ShelterNotFoundError(
                "Reservation not found"
            )

        if reservation.status != RESERVATION_ACTIVE:
            raise ShelterStateError(
                "Reservation is not active"
            )

        if (
            shelter.reserved_count
            < reservation.people_count
        ):
            raise ShelterConflictError(
                "Reservation count is inconsistent"
            )

        shelter.reserved_count -= (
            reservation.people_count
        )

        reservation.status = (
            RESERVATION_CANCELLED
        )

        self._update_full_status(shelter)

        await self.session.flush()

        return reservation

    # ========================================================
    # REGISTER RESIDENT
    # ========================================================

    async def register_resident(
        self,
        *,
        shelter_id: UUID,
        full_name: str,
        family_id: UUID | None = None,
    ) -> ShelterResident:
        """
        Create a resident registration record.

        Registration alone does not consume a bed.
        """

        shelter = await self.get_by_id(
            shelter_id
        )

        if shelter is None:
            raise ShelterNotFoundError(
                "Shelter not found"
            )

        if not full_name.strip():
            raise ValueError(
                "Resident name is required"
            )

        resident = ShelterResident(
            shelter_id=shelter_id,
            full_name=full_name.strip(),
            family_id=family_id,
            status=RESIDENT_REGISTERED,
        )

        self.session.add(resident)

        await self.session.flush()

        return resident

    # ========================================================
    # GET RESIDENT
    # ========================================================

    async def get_resident(
        self,
        resident_id: UUID,
    ) -> ShelterResident | None:

        statement = select(
            ShelterResident
        ).where(
            ShelterResident.id == resident_id
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # RESIDENT CHECK-IN
    # ========================================================

    async def check_in_resident(
        self,
        *,
        resident_id: UUID,
        reservation_id: UUID | None = None,
    ) -> ShelterResident:
        """
        Confirm physical resident arrival.

        If using a reservation, it must be an
        active single-person reservation associated
        with this resident.

        Family reservations require a separate
        household check-in workflow.
        """

        resident_lookup = await self.get_resident(
            resident_id
        )

        if resident_lookup is None:
            raise ShelterNotFoundError(
                "Resident not found"
            )

        shelter = await self._get_locked_shelter(
            resident_lookup.shelter_id
        )

        resident_result = await self.session.execute(
            select(ShelterResident)
            .where(
                ShelterResident.id == resident_id
            )
            .with_for_update()
        )

        resident = resident_result.scalar_one()

        if resident.status == RESIDENT_CHECKED_IN:
            raise ShelterConflictError(
                "Resident is already checked in"
            )

        if resident.status not in {
            RESIDENT_REGISTERED,
        }:
            raise ShelterStateError(
                "Resident cannot check in from current state"
            )

        if reservation_id is not None:
            reservation_result = await self.session.execute(
                select(ShelterReservation)
                .where(
                    ShelterReservation.id
                    == reservation_id
                )
                .with_for_update()
            )

            reservation = (
                reservation_result.scalar_one_or_none()
            )

            if reservation is None:
                raise ShelterNotFoundError(
                    "Reservation not found"
                )

            if (
                reservation.shelter_id != shelter.id
                or reservation.resident_id != resident.id
                or reservation.people_count != 1
                or reservation.status != RESERVATION_ACTIVE
            ):
                raise ShelterConflictError(
                    "Reservation does not match resident"
                )

            if shelter.reserved_count < 1:
                raise ShelterConflictError(
                    "Reserved capacity is inconsistent"
                )

            shelter.reserved_count -= 1

            reservation.status = (
                RESERVATION_FULFILLED
            )

        else:
            if (
                not shelter.is_verified
                or shelter.status != SHELTER_OPEN
            ):
                raise ShelterStateError(
                    "Shelter is not open for new admissions"
                )

            if self._free_capacity(shelter) < 1:
                raise ShelterCapacityError(
                    "No unreserved places available"
                )

        shelter.occupied_count += 1

        resident.status = RESIDENT_CHECKED_IN
        resident.checked_in_at = utc_now()

        self._update_full_status(shelter)

        await self.session.flush()

        return resident

    # ========================================================
    # RESIDENT CHECK-OUT
    # ========================================================

    async def check_out_resident(
        self,
        *,
        resident_id: UUID,
    ) -> ShelterResident:

        resident_lookup = await self.get_resident(
            resident_id
        )

        if resident_lookup is None:
            raise ShelterNotFoundError(
                "Resident not found"
            )

        shelter = await self._get_locked_shelter(
            resident_lookup.shelter_id
        )

        result = await self.session.execute(
            select(ShelterResident)
            .where(
                ShelterResident.id == resident_id
            )
            .with_for_update()
        )

        resident = result.scalar_one()

        if resident.status != RESIDENT_CHECKED_IN:
            raise ShelterStateError(
                "Resident is not checked in"
            )

        if shelter.occupied_count <= 0:
            raise ShelterConflictError(
                "Shelter occupancy is inconsistent"
            )

        shelter.occupied_count -= 1

        resident.status = RESIDENT_CHECKED_OUT
        resident.checked_out_at = utc_now()

        self._update_full_status(shelter)

        await self.session.flush()

        return resident

    # ========================================================
    # TRANSFER RESIDENT
    # ========================================================

    async def transfer_resident(
        self,
        *,
        resident_id: UUID,
        destination_shelter_id: UUID,
        approved_by: UUID,
        authorization_verified: bool,
        arrival_confirmed: bool,
    ) -> ShelterResident:
        """
        Transfer a resident between shelters.

        Requires authorization and confirmed arrival.

        For journeys in progress, use a separate
        transfer request and destination reservation
        instead of calling this method immediately.
        """

        if not authorization_verified:
            raise ShelterAuthorizationError(
                "Transfer approval required"
            )

        if not arrival_confirmed:
            raise ShelterStateError(
                "Destination arrival must be confirmed"
            )

        resident_lookup = await self.get_resident(
            resident_id
        )

        if resident_lookup is None:
            raise ShelterNotFoundError(
                "Resident not found"
            )

        source_id = resident_lookup.shelter_id

        if source_id == destination_shelter_id:
            raise ShelterConflictError(
                "Source and destination shelters are identical"
            )

        shelter_ids = sorted(
            [source_id, destination_shelter_id],
            key=str,
        )

        locked_shelters = {}

        for shelter_id in shelter_ids:
            locked_shelters[shelter_id] = (
                await self._get_locked_shelter(
                    shelter_id
                )
            )

        source = locked_shelters[source_id]
        destination = locked_shelters[
            destination_shelter_id
        ]

        resident_result = await self.session.execute(
            select(ShelterResident)
            .where(
                ShelterResident.id == resident_id
            )
            .with_for_update()
        )

        resident = resident_result.scalar_one()

        if (
            resident.status != RESIDENT_CHECKED_IN
            or resident.shelter_id != source_id
        ):
            raise ShelterStateError(
                "Resident is not currently checked in "
                "at the source shelter"
            )

        if (
            not destination.is_verified
            or destination.status != SHELTER_OPEN
        ):
            raise ShelterStateError(
                "Destination shelter is not open and verified"
            )

        if self._free_capacity(destination) < 1:
            raise ShelterCapacityError(
                "Destination has no available places"
            )

        if source.occupied_count <= 0:
            raise ShelterConflictError(
                "Source occupancy is inconsistent"
            )

        source.occupied_count -= 1
        destination.occupied_count += 1

        resident.shelter_id = destination.id
        resident.status = RESIDENT_CHECKED_IN
        resident.checked_in_at = utc_now()
        resident.checked_out_at = None

        self._update_full_status(source)
        self._update_full_status(destination)

        await self.session.flush()

        logger.info(
            "Resident %s transferred by operator %s",
            resident_id,
            approved_by,
        )

        return resident

    # ========================================================
    # LIST SHELTER RESIDENTS
    # ========================================================

    async def list_residents(
        self,
        *,
        shelter_id: UUID,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ShelterResident]:

        validate_pagination(limit, offset)

        statement = select(
            ShelterResident
        ).where(
            ShelterResident.shelter_id == shelter_id
        )

        if status is not None:
            statement = statement.where(
                ShelterResident.status == status
            )

        statement = (
            statement
            .order_by(
                ShelterResident.created_at.desc(),
                ShelterResident.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # SHELTER OCCUPANCY
    # ========================================================

    async def get_occupancy(
        self,
        *,
        shelter_id: UUID,
    ) -> dict[str, Any]:

        shelter = await self.get_by_id(
            shelter_id
        )

        if shelter is None:
            raise ShelterNotFoundError(
                "Shelter not found"
            )

        free_places = self._free_capacity(
            shelter
        )

        occupancy_percentage = (
            (
                shelter.occupied_count
                / shelter.capacity
            ) * 100
            if shelter.capacity > 0
            else None
        )

        return {
            "shelter_id": shelter.id,
            "capacity": shelter.capacity,
            "occupied": shelter.occupied_count,
            "reserved": shelter.reserved_count,
            "available": free_places,
            "occupancy_percentage": (
                round(occupancy_percentage, 2)
                if occupancy_percentage is not None
                else None
            ),
            "status": shelter.status,
            "is_verified": shelter.is_verified,
            "generated_at": utc_now().isoformat(),
        }

    # ========================================================
    # DISTRICT STATISTICS
    # ========================================================

    async def get_district_statistics(
        self,
        *,
        district: str,
    ) -> dict[str, Any]:

        district = normalize_district(district)

        statement = select(
            func.count(Shelter.id).label(
                "total_shelters"
            ),
            func.coalesce(
                func.sum(Shelter.capacity),
                0,
            ).label("total_capacity"),
            func.coalesce(
                func.sum(Shelter.occupied_count),
                0,
            ).label("occupied"),
            func.coalesce(
                func.sum(Shelter.reserved_count),
                0,
            ).label("reserved"),
        ).where(
            self._not_deleted(),
            Shelter.district == district,
        )

        result = await self.session.execute(
            statement
        )

        row = result.one()

        return {
            "district": district,
            "total_shelters": int(
                row.total_shelters
            ),
            "total_capacity": int(
                row.total_capacity
            ),
            "occupied_places": int(
                row.occupied
            ),
            "reserved_places": int(
                row.reserved
            ),
            "uncommitted_places": max(
                0,
                int(row.total_capacity)
                - int(row.occupied)
                - int(row.reserved),
            ),
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "ShelterRepository",
    "ShelterRepositoryError",
    "ShelterNotFoundError",
    "ShelterCapacityError",
    "ShelterConflictError",
    "ShelterAuthorizationError",
    "ShelterStateError",
    "normalize_district",
    "validate_coordinates",
    "haversine_distance_km",
    "validate_pagination",
    "utc_now",
]
