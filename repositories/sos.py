from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/repositories/sos.py

Async SQLAlchemy 2.x SOS repository.

Features:
- Emergency SOS registration
- SOS retrieval and pagination
- District and priority filtering
- Human-reviewed SOS triage
- Duplicate candidate detection
- Emergency escalation
- SOS acknowledgment
- Rescue mission association
- Location updates
- Status tracking
- Cancellation and resolution
- Emergency statistics
- Row-level locking
- Transaction-safe operations

Expected ORM model:
    backend.models.sos_request.SOSRequest

Required ORM columns:
    id
    tracking_reference
    sos_type
    description
    district
    latitude
    longitude
    address
    requester_name
    requester_phone
    people_count
    status
    priority
    source
    assigned_team_id
    rescue_mission_id
    acknowledged_at
    resolved_at
    cancelled_at
    cancellation_reason
    created_at
    updated_at

Recommended database constraints:
- Unique tracking_reference
- Nonnegative people_count
- Valid status and priority values
- Appropriate indexes on district/status/priority
- Index on created_at

IMPORTANT:
- Never automatically reject emergency reports.
- AI triage is advisory only.
- Acknowledgment is not rescue dispatch.
- SOS assignment requires an authorized workflow.
- Do not expose personal information publicly.
- Database writes require transaction commit.
"""


import logging
import math
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import (
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.sos_request import SOSRequest


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

VALID_PRIORITIES = {
    "unassessed",
    "low",
    "medium",
    "high",
    "critical",
}

VALID_STATUSES = {
    "received",
    "triaged",
    "awaiting_review",
    "acknowledged",
    "assigned",
    "dispatched",
    "in_progress",
    "resolved",
    "cancelled",
}

CLOSED_STATUSES = {
    "resolved",
    "cancelled",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class SOSRepositoryError(Exception):
    """Base SOS repository exception."""


class SOSNotFoundError(SOSRepositoryError):
    """SOS request not found."""


class SOSConflictError(SOSRepositoryError):
    """Conflicting SOS operation."""


class SOSAuthorizationError(SOSRepositoryError):
    """Required authorization is missing."""


class SOSStateError(SOSRepositoryError):
    """Invalid SOS state transition."""


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
    if not math.isfinite(latitude):
        raise ValueError("Latitude must be finite")

    if not math.isfinite(longitude):
        raise ValueError("Longitude must be finite")

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
    Approximate straight-line distance.

    This does not indicate road accessibility
    or a safe rescue route.
    """

    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    radius = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
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


def generate_tracking_reference() -> str:
    """
    Generate a human-readable SOS tracking reference.

    Example:
        SOS-20261014-4A7C91D2

    A database unique constraint must enforce uniqueness.
    """

    date_part = utc_now().strftime("%Y%m%d")

    random_part = secrets.token_hex(4).upper()

    return f"SOS-{date_part}-{random_part}"


# ============================================================
# SOS REPOSITORY
# ============================================================

class SOSRepository:
    """
    Async SOS persistence repository.

    Transaction ownership:
        - Repository flushes writes.
        - Calling service commits or rolls back.
        - Each request/task must use its own session.

    Example:

        async with transaction() as session:
            repository = SOSRepository(session)

            sos = await repository.create(
                sos_type="flood_rescue",
                description="Family trapped in flooded house",
                district="thrissur",
                requester_name="Resident",
                requester_phone="...",
                people_count=4,
            )
    """

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    async def _get_locked(
        self,
        sos_id: UUID,
    ) -> SOSRequest:

        statement = (
            select(SOSRequest)
            .where(
                SOSRequest.id == sos_id
            )
            .with_for_update()
        )

        result = await self.session.execute(
            statement
        )

        sos = result.scalar_one_or_none()

        if sos is None:
            raise SOSNotFoundError(
                "SOS request not found"
            )

        return sos

    @staticmethod
    def _ensure_open(
        sos: SOSRequest,
    ) -> None:
        if sos.status in CLOSED_STATUSES:
            raise SOSStateError(
                "SOS request is already closed"
            )

    # ========================================================
    # CREATE SOS
    # ========================================================

    async def create(
        self,
        *,
        sos_type: str,
        description: str,
        district: str,
        requester_name: str | None = None,
        requester_phone: str | None = None,
        people_count: int = 1,
        latitude: float | None = None,
        longitude: float | None = None,
        address: str | None = None,
        source: str = "web",
    ) -> SOSRequest:
        """
        Register an emergency SOS.

        This method must not reject a report based
        on AI duplicate scores or predicted priority.

        A created SOS means the record is pending
        operational processing, not that rescue
        has been dispatched.
        """

        if not description.strip():
            raise ValueError(
                "Emergency description is required"
            )

        if people_count < 1:
            raise ValueError(
                "people_count must be at least 1"
            )

        if (
            latitude is None
        ) != (
            longitude is None
        ):
            raise ValueError(
                "Latitude and longitude must be provided together"
            )

        if latitude is not None:
            validate_coordinates(
                latitude,
                longitude,
            )

        if (
            latitude is None
            and not (address or "").strip()
        ):
            raise ValueError(
                "GPS coordinates or an address are required"
            )

        sos = SOSRequest(
            tracking_reference=generate_tracking_reference(),
            sos_type=sos_type,
            description=description.strip(),
            district=normalize_district(district),
            requester_name=requester_name,
            requester_phone=requester_phone,
            people_count=people_count,
            latitude=latitude,
            longitude=longitude,
            address=address,
            source=source,
            status="received",
            priority="unassessed",
        )

        self.session.add(sos)

        await self.session.flush()

        return sos

    # ========================================================
    # GET SOS BY ID
    # ========================================================

    async def get_by_id(
        self,
        sos_id: UUID,
    ) -> SOSRequest | None:

        statement = select(SOSRequest).where(
            SOSRequest.id == sos_id
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # GET SOS BY TRACKING REFERENCE
    # ========================================================

    async def get_by_tracking_reference(
        self,
        tracking_reference: str,
    ) -> SOSRequest | None:

        statement = select(SOSRequest).where(
            SOSRequest.tracking_reference
            == tracking_reference.strip().upper()
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # LIST SOS REQUESTS
    # ========================================================

    async def list_requests(
        self,
        *,
        district: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        sos_type: str | None = None,
        source: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[SOSRequest]:

        validate_pagination(limit, offset)

        statement = select(SOSRequest)

        if district is not None:
            statement = statement.where(
                SOSRequest.district
                == normalize_district(district)
            )

        if status is not None:
            statement = statement.where(
                SOSRequest.status == status
            )

        if priority is not None:
            statement = statement.where(
                SOSRequest.priority == priority
            )

        if sos_type is not None:
            statement = statement.where(
                SOSRequest.sos_type == sos_type
            )

        if source is not None:
            statement = statement.where(
                SOSRequest.source == source
            )

        statement = (
            statement
            .order_by(
                SOSRequest.created_at.asc(),
                SOSRequest.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # COUNT SOS REQUESTS
    # ========================================================

    async def count(
        self,
        *,
        district: str | None = None,
        status: str | None = None,
        priority: str | None = None,
    ) -> int:

        statement = select(
            func.count(SOSRequest.id)
        )

        if district is not None:
            statement = statement.where(
                SOSRequest.district
                == normalize_district(district)
            )

        if status is not None:
            statement = statement.where(
                SOSRequest.status == status
            )

        if priority is not None:
            statement = statement.where(
                SOSRequest.priority == priority
            )

        result = await self.session.execute(
            statement
        )

        return int(result.scalar_one())

    # ========================================================
    # ACKNOWLEDGE SOS
    # ========================================================

    async def acknowledge(
        self,
        *,
        sos_id: UUID,
        acknowledged_by: UUID,
    ) -> SOSRequest:
        """
        Record operator acknowledgment.

        Acknowledgment is not dispatch.
        """

        sos = await self._get_locked(sos_id)

        self._ensure_open(sos)

        if sos.acknowledged_at is not None:
            return sos

        if sos.status not in {
            "received",
            "triaged",
            "awaiting_review",
        }:
            raise SOSStateError(
                "SOS cannot be acknowledged "
                "from its current state"
            )

        sos.status = "acknowledged"
        sos.acknowledged_at = utc_now()

        await self.session.flush()

        logger.info(
            "SOS %s acknowledged by %s",
            sos_id,
            acknowledged_by,
        )

        return sos

    # ========================================================
    # UPDATE PRIORITY
    # ========================================================

    async def update_priority(
        self,
        *,
        sos_id: UUID,
        priority: str,
        reviewed_by: UUID,
        human_review_verified: bool,
    ) -> SOSRequest:
        """
        Apply a human-reviewed priority.

        AI priority recommendations must not
        be applied without qualified review.
        """

        if priority not in VALID_PRIORITIES:
            raise ValueError(
                "Invalid SOS priority"
            )

        if not human_review_verified:
            raise SOSAuthorizationError(
                "Human review is required"
            )

        sos = await self._get_locked(sos_id)

        self._ensure_open(sos)

        sos.priority = priority

        if sos.status in {
            "received",
            "awaiting_review",
        }:
            sos.status = "triaged"

        await self.session.flush()

        logger.info(
            "SOS %s priority reviewed by %s",
            sos_id,
            reviewed_by,
        )

        return sos

    # ========================================================
    # ESCALATE SOS
    # ========================================================

    async def escalate(
        self,
        *,
        sos_id: UUID,
        escalated_by: UUID,
        reason: str,
        new_priority: str = "critical",
    ) -> SOSRequest:
        """
        Escalate an active SOS.

        Escalation does not automatically dispatch
        a rescue team.
        """

        if new_priority not in {
            "high",
            "critical",
        }:
            raise ValueError(
                "Escalation priority must be high or critical"
            )

        if not reason.strip():
            raise ValueError(
                "Escalation reason is required"
            )

        sos = await self._get_locked(sos_id)

        self._ensure_open(sos)

        sos.priority = new_priority

        if sos.status in {
            "received",
            "triaged",
        }:
            sos.status = "awaiting_review"

        await self.session.flush()

        logger.warning(
            "SOS %s escalated by %s: %s",
            sos_id,
            escalated_by,
            reason,
        )

        return sos

    # ========================================================
    # ASSIGN RESCUE MISSION
    # ========================================================

    async def assign_rescue_mission(
        self,
        *,
        sos_id: UUID,
        team_id: UUID,
        mission_id: UUID,
        assigned_by: UUID,
        authorization_verified: bool,
    ) -> SOSRequest:
        """
        Associate an approved rescue assignment.

        Does not confirm physical dispatch.
        """

        if not authorization_verified:
            raise SOSAuthorizationError(
                "Authorized assignment approval required"
            )

        sos = await self._get_locked(sos_id)

        self._ensure_open(sos)

        if sos.rescue_mission_id is not None:
            raise SOSConflictError(
                "SOS already has an assigned rescue mission"
            )

        sos.assigned_team_id = team_id
        sos.rescue_mission_id = mission_id
        sos.status = "assigned"

        await self.session.flush()

        logger.info(
            "SOS %s assigned by %s",
            sos_id,
            assigned_by,
        )

        return sos

    # ========================================================
    # CONFIRM SOS DISPATCH STATUS
    # ========================================================

    async def mark_dispatched(
        self,
        *,
        sos_id: UUID,
        confirmed_by: UUID,
        dispatch_record_verified: bool,
    ) -> SOSRequest:
        """
        Update SOS status after the corresponding
        rescue mission dispatch is confirmed.

        Caller must verify the committed dispatch
        record and responder acknowledgment.
        """

        if not dispatch_record_verified:
            raise SOSAuthorizationError(
                "Verified dispatch record required"
            )

        sos = await self._get_locked(sos_id)

        if sos.status != "assigned":
            raise SOSStateError(
                "Only assigned SOS requests "
                "can be marked dispatched"
            )

        if (
            sos.rescue_mission_id is None
            or sos.assigned_team_id is None
        ):
            raise SOSConflictError(
                "SOS has no rescue mission assignment"
            )

        sos.status = "dispatched"

        await self.session.flush()

        return sos

    # ========================================================
    # MARK IN PROGRESS
    # ========================================================

    async def mark_in_progress(
        self,
        *,
        sos_id: UUID,
        confirmed_by: UUID,
    ) -> SOSRequest:

        sos = await self._get_locked(sos_id)

        if sos.status != "dispatched":
            raise SOSStateError(
                "SOS must be dispatched first"
            )

        sos.status = "in_progress"

        await self.session.flush()

        return sos

    # ========================================================
    # RESOLVE SOS
    # ========================================================

    async def resolve(
        self,
        *,
        sos_id: UUID,
        resolved_by: UUID,
        field_confirmation_verified: bool,
    ) -> SOSRequest:
        """
        Resolve an SOS after verified field confirmation.

        Never automatically resolve based on
        AI predictions or elapsed time.
        """

        if not field_confirmation_verified:
            raise SOSAuthorizationError(
                "Verified field confirmation required"
            )

        sos = await self._get_locked(sos_id)

        self._ensure_open(sos)

        sos.status = "resolved"
        sos.resolved_at = utc_now()

        await self.session.flush()

        logger.info(
            "SOS %s resolved by %s",
            sos_id,
            resolved_by,
        )

        return sos

    # ========================================================
    # CANCEL SOS
    # ========================================================

    async def cancel(
        self,
        *,
        sos_id: UUID,
        cancelled_by: UUID,
        identity_verified: bool,
        reason: str,
    ) -> SOSRequest:
        """
        Cancel an SOS after identity verification.

        If rescue resources are already deployed,
        the service must coordinate cancellation
        with the active rescue mission.
        """

        if not identity_verified:
            raise SOSAuthorizationError(
                "Identity verification required"
            )

        if not reason.strip():
            raise ValueError(
                "Cancellation reason is required"
            )

        sos = await self._get_locked(sos_id)

        self._ensure_open(sos)

        if sos.status in {
            "dispatched",
            "in_progress",
        }:
            raise SOSStateError(
                "Active rescue missions require "
                "coordinated cancellation"
            )

        sos.status = "cancelled"
        sos.cancelled_at = utc_now()
        sos.cancellation_reason = reason.strip()

        await self.session.flush()

        logger.info(
            "SOS %s cancelled by %s",
            sos_id,
            cancelled_by,
        )

        return sos

    # ========================================================
    # UPDATE SOS LOCATION
    # ========================================================

    async def update_location(
        self,
        *,
        sos_id: UUID,
        latitude: float,
        longitude: float,
        updated_by: UUID,
    ) -> SOSRequest:
        """
        Update SOS GPS coordinates.

        Caller must authenticate the requester
        or an authorized emergency operator.
        """

        validate_coordinates(
            latitude,
            longitude,
        )

        sos = await self._get_locked(sos_id)

        self._ensure_open(sos)

        sos.latitude = latitude
        sos.longitude = longitude

        await self.session.flush()

        return sos

    # ========================================================
    # OPEN SOS REQUESTS
    # ========================================================

    async def get_open_requests(
        self,
        *,
        district: str | None = None,
        limit: int = 100,
    ) -> Sequence[SOSRequest]:

        validate_pagination(limit, 0)

        statement = select(SOSRequest).where(
            SOSRequest.status.notin_(
                CLOSED_STATUSES
            )
        )

        if district is not None:
            statement = statement.where(
                SOSRequest.district
                == normalize_district(district)
            )

        statement = (
            statement
            .order_by(
                SOSRequest.created_at.asc(),
                SOSRequest.id.asc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # CRITICAL SOS REQUESTS
    # ========================================================

    async def get_critical_requests(
        self,
        *,
        district: str | None = None,
        limit: int = 100,
    ) -> Sequence[SOSRequest]:

        validate_pagination(limit, 0)

        statement = select(SOSRequest).where(
            SOSRequest.priority == "critical",
            SOSRequest.status.notin_(
                CLOSED_STATUSES
            ),
        )

        if district is not None:
            statement = statement.where(
                SOSRequest.district
                == normalize_district(district)
            )

        statement = (
            statement
            .order_by(
                SOSRequest.created_at.asc(),
                SOSRequest.id.asc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # NEARBY SOS REQUESTS
    # ========================================================

    async def find_nearby_requests(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0,
        district: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Find geographically nearby SOS reports.

        This method does not classify reports
        as duplicates or reject them.
        """

        validate_coordinates(
            latitude,
            longitude,
        )

        if radius_km <= 0:
            raise ValueError(
                "radius_km must be positive"
            )

        validate_pagination(limit, 0)

        lat_delta = radius_km / 111.32

        cos_lat = abs(
            math.cos(math.radians(latitude))
        )

        lon_delta = min(
            180.0,
            radius_km / (
                111.32 * max(cos_lat, 0.000001)
            ),
        )

        statement = select(SOSRequest).where(
            SOSRequest.latitude.is_not(None),
            SOSRequest.longitude.is_not(None),
            SOSRequest.latitude.between(
                max(-90, latitude - lat_delta),
                min(90, latitude + lat_delta),
            ),
        )

        if lon_delta < 180:
            minimum = longitude - lon_delta
            maximum = longitude + lon_delta

            if minimum < -180:
                from sqlalchemy import or_

                statement = statement.where(
                    or_(
                        SOSRequest.longitude
                        >= minimum + 360,
                        SOSRequest.longitude
                        <= maximum,
                    )
                )

            elif maximum > 180:
                from sqlalchemy import or_

                statement = statement.where(
                    or_(
                        SOSRequest.longitude >= minimum,
                        SOSRequest.longitude
                        <= maximum - 360,
                    )
                )

            else:
                statement = statement.where(
                    SOSRequest.longitude.between(
                        minimum,
                        maximum,
                    )
                )

        if district is not None:
            statement = statement.where(
                SOSRequest.district
                == normalize_district(district)
            )

        result = await self.session.execute(
            statement
        )

        matches = []

        for sos in result.scalars().all():
            distance = haversine_distance_km(
                latitude,
                longitude,
                sos.latitude,
                sos.longitude,
            )

            if distance <= radius_km:
                matches.append(
                    {
                        "sos": sos,
                        "distance_km": round(
                            distance,
                            3,
                        ),
                    }
                )

        matches.sort(
            key=lambda item: (
                item["distance_km"],
                str(item["sos"].id),
            )
        )

        return matches[:limit]

    # ========================================================
    # DUPLICATE CANDIDATES
    # ========================================================

    async def find_duplicate_candidates(
        self,
        *,
        sos_type: str,
        district: str,
        latitude: float,
        longitude: float,
        radius_km: float = 1.0,
        hours: int = 6,
    ) -> list[dict[str, Any]]:
        """
        Identify possible duplicate SOS reports.

        Results require human review.

        Multiple SOS reports from the same location
        may represent different people or emergencies.
        """

        if hours <= 0:
            raise ValueError(
                "hours must be positive"
            )

        candidates = await self.find_nearby_requests(
            latitude=latitude,
            longitude=longitude,
            district=district,
            radius_km=radius_km,
            limit=500,
        )

        cutoff = utc_now() - timedelta(
            hours=hours
        )

        return [
            candidate
            for candidate in candidates
            if (
                candidate["sos"].sos_type == sos_type
                and candidate["sos"].created_at >= cutoff
            )
        ]

    # ========================================================
    # SOS STATISTICS
    # ========================================================

    async def get_statistics(
        self,
        *,
        district: str | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate actual SOS database records.

        Does not invent rescue completion
        or emergency response statistics.
        """

        base_filter = []

        if district is not None:
            district = normalize_district(district)

            base_filter.append(
                SOSRequest.district == district
            )

        status_statement = (
            select(
                SOSRequest.status,
                func.count(SOSRequest.id),
            )
            .where(*base_filter)
            .group_by(SOSRequest.status)
        )

        priority_statement = (
            select(
                SOSRequest.priority,
                func.count(SOSRequest.id),
            )
            .where(*base_filter)
            .group_by(SOSRequest.priority)
        )

        status_result = await self.session.execute(
            status_statement
        )

        priority_result = await self.session.execute(
            priority_statement
        )

        status_counts = {
            str(status): int(count)
            for status, count in status_result.all()
        }

        priority_counts = {
            str(priority): int(count)
            for priority, count in priority_result.all()
        }

        total = sum(
            status_counts.values()
        )

        open_count = sum(
            count
            for status, count in status_counts.items()
            if status not in CLOSED_STATUSES
        )

        return {
            "district": district,
            "total_sos": total,
            "open_sos": open_count,
            "critical_sos": priority_counts.get(
                "critical",
                0,
            ),
            "resolved_sos": status_counts.get(
                "resolved",
                0,
            ),
            "cancelled_sos": status_counts.get(
                "cancelled",
                0,
            ),
            "by_status": status_counts,
            "by_priority": priority_counts,
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "SOSRepository",
    "SOSRepositoryError",
    "SOSNotFoundError",
    "SOSConflictError",
    "SOSAuthorizationError",
    "SOSStateError",
    "normalize_district",
    "validate_coordinates",
    "haversine_distance_km",
    "validate_pagination",
    "generate_tracking_reference",
    "utc_now",
]
