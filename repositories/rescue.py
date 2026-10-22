from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/repositories/rescue.py

Async SQLAlchemy 2.x rescue repository.

Features:
- Rescue request retrieval and management
- Rescue team registration and tracking
- Team availability and capability filtering
- District-based rescue coordination
- Rescue team assignment reservations
- Human-approved dispatch recording
- Rescue mission management
- Mission progress and completion
- Rescue statistics
- Row-level locking
- Audit-friendly transaction handling

Expected ORM models in backend/models/:
    rescue_team.py:
        RescueTeam

    rescue_mission.py:
        RescueMission

    sos_request.py:
        SOSRequest

Required RescueTeam columns:
    id
    name
    district
    status
    capabilities
    latitude
    longitude
    is_verified
    is_deleted
    created_at
    updated_at

Required RescueMission columns:
    id
    sos_request_id
    team_id
    status
    priority
    assigned_by
    assigned_at
    dispatched_by
    dispatched_at
    started_at
    completed_at
    completion_notes
    created_at
    updated_at

Required SOSRequest columns:
    id
    district
    status
    priority
    latitude
    longitude
    created_at

IMPORTANT:
- AI recommendations never authorize rescue dispatch.
- Team assignment is not dispatch confirmation.
- Real rescue actions require authorized human approval.
- Rescue completion requires verified field confirmation.
- Never invent live team locations or response times.
- All writes require transaction commit by the service.
"""


import logging
import math
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import (
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.rescue_team import RescueTeam
from backend.models.rescue_mission import RescueMission
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

ACTIVE_MISSION_STATUSES = {
    "assigned",
    "dispatched",
    "in_progress",
}

TERMINAL_MISSION_STATUSES = {
    "completed",
    "cancelled",
}

RESCUE_TEAM_AVAILABLE = "available"
RESCUE_TEAM_ASSIGNED = "assigned"
RESCUE_TEAM_DEPLOYED = "deployed"


# ============================================================
# EXCEPTIONS
# ============================================================

class RescueRepositoryError(Exception):
    """Base rescue repository exception."""


class RescueNotFoundError(RescueRepositoryError):
    """Requested rescue resource was not found."""


class RescueConflictError(RescueRepositoryError):
    """Conflicting rescue assignment or state."""


class RescueAuthorizationError(RescueRepositoryError):
    """Required authorization was not supplied."""


class RescueStateError(RescueRepositoryError):
    """Invalid rescue mission state transition."""


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
        raise ValueError(
            "Invalid latitude"
        )

    if not -180 <= longitude <= 180:
        raise ValueError(
            "Invalid longitude"
        )


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Approximate straight-line distance.

    Not a driving distance or verified rescue route.
    """

    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    radius = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2 * radius * math.asin(math.sqrt(a))
    )


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
# RESCUE REPOSITORY
# ============================================================

class RescueRepository:
    """
    Rescue request, team and mission persistence.

    Example:

        async with transaction() as session:
            repository = RescueRepository(session)

            mission = await repository.reserve_team(
                sos_request_id=sos_id,
                team_id=team_id,
                assigned_by=operator_id,
                priority="critical",
            )

    The calling service must authenticate and authorize
    the operator before invoking privileged methods.

    Repository operations flush but do not commit.
    """

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session

    # ========================================================
    # GET SOS REQUEST
    # ========================================================

    async def get_sos_request(
        self,
        sos_request_id: UUID,
    ) -> SOSRequest | None:

        statement = select(SOSRequest).where(
            SOSRequest.id == sos_request_id
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # LIST RESCUE REQUESTS
    # ========================================================

    async def list_rescue_requests(
        self,
        *,
        district: str | None = None,
        status: str | None = None,
        priority: str | None = None,
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
    # GET RESCUE TEAM
    # ========================================================

    async def get_team(
        self,
        team_id: UUID,
    ) -> RescueTeam | None:

        statement = select(RescueTeam).where(
            RescueTeam.id == team_id,
            RescueTeam.is_deleted.is_(False),
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # REGISTER RESCUE TEAM
    # ========================================================

    async def create_team(
        self,
        *,
        name: str,
        district: str,
        capabilities: list[str],
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> RescueTeam:
        """
        Register a rescue team.

        New teams remain unverified and unavailable
        until an authorized operator verifies them.
        """

        if not name.strip():
            raise ValueError(
                "Team name is required"
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

        team = RescueTeam(
            name=name.strip(),
            district=normalize_district(district),
            capabilities=capabilities,
            latitude=latitude,
            longitude=longitude,
            status="unavailable",
            is_verified=False,
            is_deleted=False,
        )

        self.session.add(team)

        await self.session.flush()

        return team

    # ========================================================
    # LIST RESCUE TEAMS
    # ========================================================

    async def list_teams(
        self,
        *,
        district: str | None = None,
        status: str | None = None,
        verified_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[RescueTeam]:

        validate_pagination(limit, offset)

        statement = select(RescueTeam).where(
            RescueTeam.is_deleted.is_(False)
        )

        if verified_only:
            statement = statement.where(
                RescueTeam.is_verified.is_(True)
            )

        if district is not None:
            statement = statement.where(
                RescueTeam.district
                == normalize_district(district)
            )

        if status is not None:
            statement = statement.where(
                RescueTeam.status == status
            )

        statement = (
            statement
            .order_by(
                RescueTeam.name.asc(),
                RescueTeam.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # AVAILABLE RESCUE TEAMS
    # ========================================================

    async def get_available_teams(
        self,
        *,
        district: str | None = None,
        required_capabilities: list[str] | None = None,
        limit: int = 100,
    ) -> Sequence[RescueTeam]:
        """
        Return verified, available rescue teams.

        Capabilities are assumed to be stored as
        PostgreSQL ARRAY or compatible JSONB.

        This implementation filters capabilities
        in Python for schema portability.
        """

        validate_pagination(limit, 0)

        statement = select(RescueTeam).where(
            RescueTeam.is_deleted.is_(False),
            RescueTeam.is_verified.is_(True),
            RescueTeam.status
            == RESCUE_TEAM_AVAILABLE,
        )

        if district is not None:
            statement = statement.where(
                RescueTeam.district
                == normalize_district(district)
            )

        result = await self.session.execute(
            statement.order_by(
                RescueTeam.id.asc()
            )
        )

        teams = result.scalars().all()

        required = set(
            required_capabilities or []
        )

        matches = [
            team
            for team in teams
            if required.issubset(
                set(team.capabilities or [])
            )
        ]

        return matches[:limit]

    # ========================================================
    # NEAREST AVAILABLE TEAMS
    # ========================================================

    async def find_nearest_teams(
        self,
        *,
        latitude: float,
        longitude: float,
        district: str | None = None,
        required_capabilities: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Rank available teams by straight-line distance.

        This is advisory only.

        A real dispatch requires:
        - Verified team status
        - Verified availability
        - Safe route confirmation
        - Human authorization
        """

        validate_coordinates(
            latitude,
            longitude,
        )

        validate_pagination(limit, 0)

        teams = await self.get_available_teams(
            district=district,
            required_capabilities=required_capabilities,
            limit=500,
        )

        candidates = []

        for team in teams:
            if (
                team.latitude is None
                or team.longitude is None
            ):
                continue

            distance = haversine_distance_km(
                latitude,
                longitude,
                team.latitude,
                team.longitude,
            )

            candidates.append(
                {
                    "team": team,
                    "distance_km": round(
                        distance,
                        3,
                    ),
                    "route_verified": False,
                    "dispatch_authorized": False,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["distance_km"],
                str(item["team"].id),
            )
        )

        return candidates[:limit]

    # ========================================================
    # UPDATE TEAM STATUS
    # ========================================================

    async def update_team_status(
        self,
        *,
        team_id: UUID,
        status: str,
    ) -> RescueTeam:
        """
        Update a rescue team's operational status.

        Service layer must enforce authorized
        state transitions and operator permissions.
        """

        statement = (
            select(RescueTeam)
            .where(
                RescueTeam.id == team_id,
                RescueTeam.is_deleted.is_(False),
            )
            .with_for_update()
        )

        result = await self.session.execute(
            statement
        )

        team = result.scalar_one_or_none()

        if team is None:
            raise RescueNotFoundError(
                "Rescue team not found"
            )

        if (
            status == RESCUE_TEAM_AVAILABLE
            and not team.is_verified
        ):
            raise RescueStateError(
                "Unverified rescue team cannot "
                "be marked available"
            )

        team.status = status

        await self.session.flush()

        return team

    # ========================================================
    # UPDATE TEAM LOCATION
    # ========================================================

    async def update_team_location(
        self,
        *,
        team_id: UUID,
        latitude: float,
        longitude: float,
    ) -> RescueTeam:
        """
        Store a verified team location update.

        Caller must authenticate the reporting
        responder or trusted tracking device.
        """

        validate_coordinates(
            latitude,
            longitude,
        )

        team = await self.get_team(team_id)

        if team is None:
            raise RescueNotFoundError(
                "Rescue team not found"
            )

        team.latitude = latitude
        team.longitude = longitude

        await self.session.flush()

        return team

    # ========================================================
    # GET RESCUE MISSION
    # ========================================================

    async def get_mission(
        self,
        mission_id: UUID,
    ) -> RescueMission | None:

        statement = select(RescueMission).where(
            RescueMission.id == mission_id
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # LIST RESCUE MISSIONS
    # ========================================================

    async def list_missions(
        self,
        *,
        team_id: UUID | None = None,
        sos_request_id: UUID | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[RescueMission]:

        validate_pagination(limit, offset)

        statement = select(RescueMission)

        if team_id is not None:
            statement = statement.where(
                RescueMission.team_id == team_id
            )

        if sos_request_id is not None:
            statement = statement.where(
                RescueMission.sos_request_id
                == sos_request_id
            )

        if status is not None:
            statement = statement.where(
                RescueMission.status == status
            )

        statement = (
            statement
            .order_by(
                RescueMission.created_at.desc(),
                RescueMission.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # RESERVE RESCUE TEAM
    # ========================================================

    async def reserve_team(
        self,
        *,
        sos_request_id: UUID,
        team_id: UUID,
        assigned_by: UUID,
        priority: str,
    ) -> RescueMission:
        """
        Reserve a verified, available team for an SOS.

        This does not dispatch the rescue team.

        The team row is locked to prevent competing
        reservations within the same database.
        """

        team_statement = (
            select(RescueTeam)
            .where(
                RescueTeam.id == team_id,
                RescueTeam.is_deleted.is_(False),
            )
            .with_for_update()
        )

        team_result = await self.session.execute(
            team_statement
        )

        team = team_result.scalar_one_or_none()

        if team is None:
            raise RescueNotFoundError(
                "Rescue team not found"
            )

        if not team.is_verified:
            raise RescueStateError(
                "Rescue team is not verified"
            )

        if team.status != RESCUE_TEAM_AVAILABLE:
            raise RescueConflictError(
                "Rescue team is not available"
            )

        sos_statement = (
            select(SOSRequest)
            .where(
                SOSRequest.id == sos_request_id
            )
            .with_for_update()
        )

        sos_result = await self.session.execute(
            sos_statement
        )

        sos_request = sos_result.scalar_one_or_none()

        if sos_request is None:
            raise RescueNotFoundError(
                "SOS request not found"
            )

        if sos_request.status in {
            "resolved",
            "cancelled",
        }:
            raise RescueStateError(
                "Cannot assign team to a closed SOS"
            )

        existing_statement = select(
            RescueMission
        ).where(
            RescueMission.sos_request_id
            == sos_request_id,
            RescueMission.status.in_(
                ACTIVE_MISSION_STATUSES
            ),
        )

        existing_result = await self.session.execute(
            existing_statement
        )

        if existing_result.scalars().first():
            raise RescueConflictError(
                "SOS already has an active rescue mission"
            )

        mission = RescueMission(
            sos_request_id=sos_request_id,
            team_id=team_id,
            status="assigned",
            priority=priority,
            assigned_by=assigned_by,
            assigned_at=utc_now(),
        )

        team.status = RESCUE_TEAM_ASSIGNED

        self.session.add(mission)

        await self.session.flush()

        return mission

    # ========================================================
    # CONFIRM DISPATCH
    # ========================================================

    async def confirm_dispatch(
        self,
        *,
        mission_id: UUID,
        dispatched_by: UUID,
        authorization_verified: bool,
        team_acknowledged: bool,
    ) -> RescueMission:
        """
        Record authorized rescue dispatch.

        A successful repository call means the database
        dispatch record is ready for commit.

        It does not prove a vehicle physically departed.
        """

        if not authorization_verified:
            raise RescueAuthorizationError(
                "Authorized dispatch approval required"
            )

        if not team_acknowledged:
            raise RescueStateError(
                "Rescue team acknowledgment required"
            )

        mission_statement = (
            select(RescueMission)
            .where(
                RescueMission.id == mission_id
            )
            .with_for_update()
        )

        mission_result = await self.session.execute(
            mission_statement
        )

        mission = mission_result.scalar_one_or_none()

        if mission is None:
            raise RescueNotFoundError(
                "Rescue mission not found"
            )

        if mission.status != "assigned":
            raise RescueStateError(
                "Only assigned missions can be dispatched"
            )

        team_statement = (
            select(RescueTeam)
            .where(
                RescueTeam.id == mission.team_id
            )
            .with_for_update()
        )

        team_result = await self.session.execute(
            team_statement
        )

        team = team_result.scalar_one_or_none()

        if team is None:
            raise RescueNotFoundError(
                "Assigned rescue team not found"
            )

        if not team.is_verified:
            raise RescueStateError(
                "Rescue team verification is invalid"
            )

        if team.status != RESCUE_TEAM_ASSIGNED:
            raise RescueConflictError(
                "Team is not reserved for dispatch"
            )

        mission.status = "dispatched"
        mission.dispatched_by = dispatched_by
        mission.dispatched_at = utc_now()

        team.status = RESCUE_TEAM_DEPLOYED

        await self.session.flush()

        return mission

    # ========================================================
    # START MISSION
    # ========================================================

    async def start_mission(
        self,
        *,
        mission_id: UUID,
    ) -> RescueMission:

        statement = (
            select(RescueMission)
            .where(
                RescueMission.id == mission_id
            )
            .with_for_update()
        )

        result = await self.session.execute(
            statement
        )

        mission = result.scalar_one_or_none()

        if mission is None:
            raise RescueNotFoundError(
                "Rescue mission not found"
            )

        if mission.status != "dispatched":
            raise RescueStateError(
                "Only dispatched missions can start"
            )

        mission.status = "in_progress"
        mission.started_at = utc_now()

        await self.session.flush()

        return mission

    # ========================================================
    # COMPLETE MISSION
    # ========================================================

    async def complete_mission(
        self,
        *,
        mission_id: UUID,
        completion_notes: str,
        field_confirmation_verified: bool,
    ) -> RescueMission:
        """
        Complete a mission after field confirmation.

        Do not infer completion from AI predictions,
        GPS arrival or elapsed time.
        """

        if not field_confirmation_verified:
            raise RescueAuthorizationError(
                "Verified field confirmation is required"
            )

        if not completion_notes.strip():
            raise ValueError(
                "Completion notes are required"
            )

        statement = (
            select(RescueMission)
            .where(
                RescueMission.id == mission_id
            )
            .with_for_update()
        )

        result = await self.session.execute(
            statement
        )

        mission = result.scalar_one_or_none()

        if mission is None:
            raise RescueNotFoundError(
                "Rescue mission not found"
            )

        if mission.status != "in_progress":
            raise RescueStateError(
                "Only in-progress missions can complete"
            )

        team_statement = (
            select(RescueTeam)
            .where(
                RescueTeam.id == mission.team_id
            )
            .with_for_update()
        )

        team_result = await self.session.execute(
            team_statement
        )

        team = team_result.scalar_one_or_none()

        mission.status = "completed"
        mission.completed_at = utc_now()
        mission.completion_notes = (
            completion_notes.strip()
        )

        if team is not None:
            # Availability must be confirmed separately
            # after safety, equipment and readiness checks.
            team.status = "unavailable"

        await self.session.flush()

        return mission

    # ========================================================
    # CANCEL MISSION
    # ========================================================

    async def cancel_mission(
        self,
        *,
        mission_id: UUID,
        cancellation_reason: str,
        authorized_by: UUID,
    ) -> RescueMission:
        """
        Cancel a mission through an authorized workflow.

        The service must arrange reassignment or escalation
        if the SOS still requires assistance.
        """

        if not cancellation_reason.strip():
            raise ValueError(
                "Cancellation reason is required"
            )

        statement = (
            select(RescueMission)
            .where(
                RescueMission.id == mission_id
            )
            .with_for_update()
        )

        result = await self.session.execute(
            statement
        )

        mission = result.scalar_one_or_none()

        if mission is None:
            raise RescueNotFoundError(
                "Rescue mission not found"
            )

        if mission.status in TERMINAL_MISSION_STATUSES:
            raise RescueStateError(
                "Mission is already closed"
            )

        team_statement = (
            select(RescueTeam)
            .where(
                RescueTeam.id == mission.team_id
            )
            .with_for_update()
        )

        team_result = await self.session.execute(
            team_statement
        )

        team = team_result.scalar_one_or_none()

        mission.status = "cancelled"
        mission.completion_notes = (
            "Cancelled by authorized operator: "
            + cancellation_reason.strip()
        )

        if team is not None:
            team.status = "unavailable"

        logger.info(
            "Rescue mission %s cancelled by operator %s",
            mission_id,
            authorized_by,
        )

        await self.session.flush()

        return mission

    # ========================================================
    # ACTIVE MISSIONS
    # ========================================================

    async def get_active_missions(
        self,
        *,
        team_id: UUID | None = None,
        limit: int = 100,
    ) -> Sequence[RescueMission]:

        validate_pagination(limit, 0)

        statement = select(RescueMission).where(
            RescueMission.status.in_(
                ACTIVE_MISSION_STATUSES
            )
        )

        if team_id is not None:
            statement = statement.where(
                RescueMission.team_id == team_id
            )

        statement = (
            statement
            .order_by(
                RescueMission.created_at.asc(),
                RescueMission.id.asc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # RESCUE STATISTICS
    # ========================================================

    async def get_statistics(
        self,
    ) -> dict[str, Any]:
        """
        Aggregate real database records.

        Unknown metrics are not fabricated.
        """

        team_query = select(
            RescueTeam.status,
            func.count(RescueTeam.id),
        ).where(
            RescueTeam.is_deleted.is_(False)
        ).group_by(
            RescueTeam.status
        )

        mission_query = select(
            RescueMission.status,
            func.count(RescueMission.id),
        ).group_by(
            RescueMission.status
        )

        team_result = await self.session.execute(
            team_query
        )

        mission_result = await self.session.execute(
            mission_query
        )

        team_counts = {
            str(status): int(count)
            for status, count in team_result.all()
        }

        mission_counts = {
            str(status): int(count)
            for status, count in mission_result.all()
        }

        return {
            "teams": team_counts,
            "missions": mission_counts,
            "available_teams": team_counts.get(
                "available",
                0,
            ),
            "active_missions": sum(
                mission_counts.get(status, 0)
                for status in ACTIVE_MISSION_STATUSES
            ),
            "completed_missions": mission_counts.get(
                "completed",
                0,
            ),
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RescueRepository",
    "RescueRepositoryError",
    "RescueNotFoundError",
    "RescueConflictError",
    "RescueAuthorizationError",
    "RescueStateError",
    "normalize_district",
    "validate_coordinates",
    "haversine_distance_km",
    "validate_pagination",
    "utc_now",
]
