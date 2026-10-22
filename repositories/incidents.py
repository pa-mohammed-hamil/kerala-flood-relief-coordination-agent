from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/repositories/incidents.py

Async SQLAlchemy 2.x incident repository.

Features:
- Create, retrieve and update incidents
- District-based incident searches
- Incident type and severity filtering
- Status and priority management
- Incident assignment
- Duplicate candidate discovery
- GPS-based incident proximity search
- Pagination and sorting
- Incident statistics
- Soft deletion
- Optimistic concurrency support
- Database transaction integration

Required ORM model:
    backend.models.incident.Incident

Expected ORM columns:
    id
    title
    description
    incident_type
    status
    priority
    severity
    district
    latitude
    longitude
    reported_at
    assigned_to
    resolved_at
    created_at
    updated_at
    is_deleted

The ORM model should inherit suitable mixins from:
    backend.database.base

Notes:
- This repository does not perform rescue dispatch.
- Emergency incidents must not be discarded by AI scoring.
- Database writes require transaction commit.
- Authorization belongs in the service/API layer.
"""


import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import (
    and_,
    case,
    func,
    or_,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.incident import Incident


logger = logging.getLogger(__name__)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
            f"Invalid Kerala district: {value}"
        )

    return district


def validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError(
            "Latitude must be between -90 and 90"
        )

    if not -180 <= longitude <= 180:
        raise ValueError(
            "Longitude must be between -180 and 180"
        )


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate approximate great-circle distance.

    This is not a driving distance or safe rescue route.
    """

    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    earth_radius_km = 6371.0088

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

    return (
        2
        * earth_radius_km
        * math.asin(math.sqrt(a))
    )


# ============================================================
# REPOSITORY
# ============================================================

class IncidentRepository:
    """
    Async repository for incident persistence.

    Transaction ownership:
        - Repository methods flush changes.
        - Calling service commits or rolls back.
        - Repository methods never commit automatically.

    Example:

        async with transaction() as session:
            repo = IncidentRepository(session)
            incident = await repo.create(
                title="Flooded road",
                description="Road blocked by floodwater",
                incident_type="road_blockage",
                district="thrissur",
            )
    """

    ALLOWED_SORT_FIELDS = {
        "created_at",
        "updated_at",
        "reported_at",
        "priority",
        "severity",
        "status",
    }

    ALLOWED_UPDATE_FIELDS = {
        "title",
        "description",
        "incident_type",
        "status",
        "priority",
        "severity",
        "district",
        "latitude",
        "longitude",
        "assigned_to",
        "resolved_at",
    }

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
        return Incident.is_deleted.is_(False)

    @staticmethod
    def _validate_pagination(
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

    # ========================================================
    # CREATE INCIDENT
    # ========================================================

    async def create(
        self,
        *,
        title: str,
        description: str,
        incident_type: str,
        district: str,
        status: str = "reported",
        priority: str = "unassessed",
        severity: str = "unknown",
        latitude: float | None = None,
        longitude: float | None = None,
        reported_at: datetime | None = None,
        assigned_to: UUID | None = None,
    ) -> Incident:
        """
        Create a new incident.

        The caller must commit the transaction.
        """

        district = normalize_district(district)

        if not title.strip():
            raise ValueError(
                "Incident title is required"
            )

        if not description.strip():
            raise ValueError(
                "Incident description is required"
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

        incident = Incident(
            title=title.strip(),
            description=description.strip(),
            incident_type=incident_type,
            district=district,
            status=status,
            priority=priority,
            severity=severity,
            latitude=latitude,
            longitude=longitude,
            reported_at=reported_at or utc_now(),
            assigned_to=assigned_to,
            is_deleted=False,
        )

        self.session.add(incident)

        await self.session.flush()

        return incident

    # ========================================================
    # GET INCIDENT
    # ========================================================

    async def get_by_id(
        self,
        incident_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> Incident | None:

        query = select(Incident).where(
            Incident.id == incident_id
        )

        if not include_deleted:
            query = query.where(
                self._not_deleted()
            )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    # ========================================================
    # LIST INCIDENTS
    # ========================================================

    async def list_incidents(
        self,
        *,
        district: str | None = None,
        incident_type: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        severity: str | None = None,
        assigned_to: UUID | None = None,
        reported_after: datetime | None = None,
        reported_before: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created_at",
        descending: bool = True,
    ) -> Sequence[Incident]:

        self._validate_pagination(
            limit,
            offset,
        )

        if sort_by not in self.ALLOWED_SORT_FIELDS:
            raise ValueError(
                "Unsupported sorting field"
            )

        query = select(Incident).where(
            self._not_deleted()
        )

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        if incident_type is not None:
            query = query.where(
                Incident.incident_type == incident_type
            )

        if status is not None:
            query = query.where(
                Incident.status == status
            )

        if priority is not None:
            query = query.where(
                Incident.priority == priority
            )

        if severity is not None:
            query = query.where(
                Incident.severity == severity
            )

        if assigned_to is not None:
            query = query.where(
                Incident.assigned_to == assigned_to
            )

        if reported_after is not None:
            query = query.where(
                Incident.reported_at >= reported_after
            )

        if reported_before is not None:
            query = query.where(
                Incident.reported_at <= reported_before
            )

        sort_column = getattr(
            Incident,
            sort_by,
        )

        query = query.order_by(
            sort_column.desc()
            if descending
            else sort_column.asc(),
            Incident.id.asc(),
        )

        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)

        return result.scalars().all()

    # ========================================================
    # COUNT INCIDENTS
    # ========================================================

    async def count(
        self,
        *,
        district: str | None = None,
        status: str | None = None,
        priority: str | None = None,
    ) -> int:

        query = select(
            func.count(Incident.id)
        ).where(
            self._not_deleted()
        )

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        if status is not None:
            query = query.where(
                Incident.status == status
            )

        if priority is not None:
            query = query.where(
                Incident.priority == priority
            )

        result = await self.session.execute(query)

        return int(result.scalar_one())

    # ========================================================
    # UPDATE INCIDENT
    # ========================================================

    async def update_incident(
        self,
        incident_id: UUID,
        *,
        changes: dict[str, Any],
    ) -> Incident | None:

        incident = await self.get_by_id(
            incident_id
        )

        if incident is None:
            return None

        invalid_fields = (
            set(changes)
            - self.ALLOWED_UPDATE_FIELDS
        )

        if invalid_fields:
            raise ValueError(
                "Unsupported incident update fields: "
                + ", ".join(sorted(invalid_fields))
            )

        if "district" in changes:
            changes["district"] = normalize_district(
                changes["district"]
            )

        latitude = changes.get(
            "latitude",
            incident.latitude,
        )

        longitude = changes.get(
            "longitude",
            incident.longitude,
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

        for field, value in changes.items():
            setattr(
                incident,
                field,
                value,
            )

        await self.session.flush()

        return incident

    # ========================================================
    # UPDATE STATUS
    # ========================================================

    async def update_status(
        self,
        incident_id: UUID,
        *,
        status: str,
        resolved_by: UUID | None = None,
    ) -> Incident | None:

        incident = await self.get_by_id(
            incident_id
        )

        if incident is None:
            return None

        incident.status = status

        if status == "resolved":
            if resolved_by is None:
                raise ValueError(
                    "Resolving an incident requires "
                    "an authenticated resolver"
                )

            incident.resolved_at = utc_now()

        await self.session.flush()

        return incident

    # ========================================================
    # UPDATE PRIORITY
    # ========================================================

    async def update_priority(
        self,
        incident_id: UUID,
        *,
        priority: str,
    ) -> Incident | None:

        incident = await self.get_by_id(
            incident_id
        )

        if incident is None:
            return None

        incident.priority = priority

        await self.session.flush()

        return incident

    # ========================================================
    # ASSIGN INCIDENT
    # ========================================================

    async def assign_incident(
        self,
        incident_id: UUID,
        *,
        assigned_to: UUID,
    ) -> Incident | None:
        """
        Assign an incident to a coordinator.

        This is not rescue-team dispatch.
        """

        incident = await self.get_by_id(
            incident_id
        )

        if incident is None:
            return None

        incident.assigned_to = assigned_to

        await self.session.flush()

        return incident

    # ========================================================
    # GET OPEN INCIDENTS
    # ========================================================

    async def get_open_incidents(
        self,
        *,
        district: str | None = None,
        limit: int = 100,
    ) -> Sequence[Incident]:

        self._validate_pagination(limit, 0)

        query = select(Incident).where(
            self._not_deleted(),
            Incident.status.notin_(
                [
                    "resolved",
                    "closed",
                    "cancelled",
                ]
            ),
        )

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        query = query.order_by(
            Incident.created_at.asc(),
            Incident.id.asc(),
        ).limit(limit)

        result = await self.session.execute(query)

        return result.scalars().all()

    # ========================================================
    # GET CRITICAL INCIDENTS
    # ========================================================

    async def get_critical_incidents(
        self,
        *,
        district: str | None = None,
        limit: int = 100,
    ) -> Sequence[Incident]:

        self._validate_pagination(limit, 0)

        query = select(Incident).where(
            self._not_deleted(),
            Incident.priority == "critical",
            Incident.status.notin_(
                [
                    "resolved",
                    "closed",
                    "cancelled",
                ]
            ),
        )

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        query = query.order_by(
            Incident.reported_at.asc(),
            Incident.id.asc(),
        ).limit(limit)

        result = await self.session.execute(query)

        return result.scalars().all()

    # ========================================================
    # RECENT INCIDENTS
    # ========================================================

    async def get_recent_incidents(
        self,
        *,
        hours: int = 24,
        district: str | None = None,
        limit: int = 100,
    ) -> Sequence[Incident]:

        if hours <= 0:
            raise ValueError(
                "hours must be positive"
            )

        self._validate_pagination(limit, 0)

        cutoff = utc_now() - timedelta(
            hours=hours
        )

        query = select(Incident).where(
            self._not_deleted(),
            Incident.reported_at >= cutoff,
        )

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        query = query.order_by(
            Incident.reported_at.desc(),
            Incident.id.asc(),
        ).limit(limit)

        result = await self.session.execute(query)

        return result.scalars().all()

    # ========================================================
    # NEARBY INCIDENTS
    # ========================================================

    async def find_nearby_incidents(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
        district: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Find geographically nearby incidents.

        Uses a latitude/longitude bounding box for
        initial filtering, followed by Haversine distance.

        Suitable for bounded district-scale datasets.

        For large-scale deployment, use PostGIS.

        Distance is straight-line, not route distance.
        """

        validate_coordinates(
            latitude,
            longitude,
        )

        if radius_km <= 0:
            raise ValueError(
                "radius_km must be positive"
            )

        self._validate_pagination(limit, 0)

        latitude_delta = radius_km / 111.32

        cos_lat = abs(
            math.cos(math.radians(latitude))
        )

        longitude_delta = min(
            180.0,
            radius_km / (
                111.32 * max(cos_lat, 0.000001)
            ),
        )

        query = select(Incident).where(
            self._not_deleted(),
            Incident.latitude.is_not(None),
            Incident.longitude.is_not(None),
            Incident.latitude.between(
                max(-90, latitude - latitude_delta),
                min(90, latitude + latitude_delta),
            ),
        )

        if longitude_delta < 180:
            min_lon = longitude - longitude_delta
            max_lon = longitude + longitude_delta

            if min_lon < -180:
                longitude_filter = or_(
                    Incident.longitude >= min_lon + 360,
                    Incident.longitude <= max_lon,
                )
            elif max_lon > 180:
                longitude_filter = or_(
                    Incident.longitude >= min_lon,
                    Incident.longitude <= max_lon - 360,
                )
            else:
                longitude_filter = (
                    Incident.longitude.between(
                        min_lon,
                        max_lon,
                    )
                )

            query = query.where(longitude_filter)

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        result = await self.session.execute(query)

        matches = []

        for incident in result.scalars().all():
            distance = haversine_distance_km(
                latitude,
                longitude,
                incident.latitude,
                incident.longitude,
            )

            if distance <= radius_km:
                matches.append(
                    {
                        "incident": incident,
                        "distance_km": round(
                            distance,
                            3,
                        ),
                    }
                )

        matches.sort(
            key=lambda item: (
                item["distance_km"],
                str(item["incident"].id),
            )
        )

        return matches[:limit]

    # ========================================================
    # DUPLICATE CANDIDATES
    # ========================================================

    async def find_duplicate_candidates(
        self,
        *,
        incident_type: str,
        district: str,
        latitude: float,
        longitude: float,
        radius_km: float = 1.0,
        hours: int = 6,
    ) -> list[dict[str, Any]]:
        """
        Return possible duplicate incidents.

        Never automatically delete or reject an
        emergency report based on this result.
        """

        candidates = await self.find_nearby_incidents(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            district=district,
            limit=500,
        )

        cutoff = utc_now() - timedelta(
            hours=hours
        )

        return [
            candidate
            for candidate in candidates
            if (
                candidate["incident"].incident_type
                == incident_type
                and candidate["incident"].reported_at
                >= cutoff
            )
        ]

    # ========================================================
    # DISTRICT STATISTICS
    # ========================================================

    async def get_district_statistics(
        self,
        *,
        district: str,
    ) -> dict[str, Any]:

        district = normalize_district(district)

        query = select(
            func.count(Incident.id).label(
                "total"
            ),
            func.count(
                case(
                    (
                        Incident.priority == "critical",
                        1,
                    ),
                )
            ).label("critical"),
            func.count(
                case(
                    (
                        Incident.status == "resolved",
                        1,
                    ),
                )
            ).label("resolved"),
            func.count(
                case(
                    (
                        Incident.status.notin_(
                            [
                                "resolved",
                                "closed",
                                "cancelled",
                            ]
                        ),
                        1,
                    ),
                )
            ).label("open"),
        ).where(
            self._not_deleted(),
            Incident.district == district,
        )

        result = await self.session.execute(query)

        row = result.one()

        return {
            "district": district,
            "total_incidents": row.total,
            "critical_incidents": row.critical,
            "resolved_incidents": row.resolved,
            "open_incidents": row.open,
            "generated_at": utc_now().isoformat(),
        }

    # ========================================================
    # STATUS COUNTS
    # ========================================================

    async def count_by_status(
        self,
        *,
        district: str | None = None,
    ) -> dict[str, int]:

        query = select(
            Incident.status,
            func.count(Incident.id),
        ).where(
            self._not_deleted()
        )

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        query = query.group_by(
            Incident.status
        )

        result = await self.session.execute(query)

        return {
            str(status): count
            for status, count in result.all()
        }

    # ========================================================
    # PRIORITY COUNTS
    # ========================================================

    async def count_by_priority(
        self,
        *,
        district: str | None = None,
    ) -> dict[str, int]:

        query = select(
            Incident.priority,
            func.count(Incident.id),
        ).where(
            self._not_deleted()
        )

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        query = query.group_by(
            Incident.priority
        )

        result = await self.session.execute(query)

        return {
            str(priority): count
            for priority, count in result.all()
        }

    # ========================================================
    # INCIDENT TYPE COUNTS
    # ========================================================

    async def count_by_type(
        self,
        *,
        district: str | None = None,
    ) -> dict[str, int]:

        query = select(
            Incident.incident_type,
            func.count(Incident.id),
        ).where(
            self._not_deleted()
        )

        if district is not None:
            query = query.where(
                Incident.district
                == normalize_district(district)
            )

        query = query.group_by(
            Incident.incident_type
        )

        result = await self.session.execute(query)

        return {
            str(incident_type): count
            for incident_type, count in result.all()
        }

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        incident_id: UUID,
        *,
        deleted_by: UUID,
    ) -> bool:
        """
        Mark an incident as deleted.

        Emergency records should normally be retained
        under a documented retention policy.

        This method must be restricted to authorized
        administrators and audited by the service layer.
        """

        incident = await self.get_by_id(
            incident_id
        )

        if incident is None:
            return False

        incident.is_deleted = True
        incident.deleted_at = utc_now()
        incident.deleted_by = deleted_by

        await self.session.flush()

        return True

    # ========================================================
    # RESTORE INCIDENT
    # ========================================================

    async def restore(
        self,
        incident_id: UUID,
    ) -> Incident | None:

        incident = await self.get_by_id(
            incident_id,
            include_deleted=True,
        )

        if incident is None:
            return None

        incident.is_deleted = False
        incident.deleted_at = None
        incident.deleted_by = None

        await self.session.flush()

        return incident


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "IncidentRepository",
    "utc_now",
    "normalize_district",
    "validate_coordinates",
    "haversine_distance_km",
]
