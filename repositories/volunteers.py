from __future__ import annotations
"""Placeholder module: backend/repositories/volunteers.py; implementation required."""


from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.volunteer import Volunteer


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VolunteerRepository:
    """Database operations for flood-relief volunteers."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: dict[str, Any]) -> Volunteer:
        volunteer = Volunteer(**data)
        self.db.add(volunteer)
        await self.db.flush()
        await self.db.refresh(volunteer)
        return volunteer

    async def get_by_id(
        self, volunteer_id: int
    ) -> Optional[Volunteer]:
        return await self.db.get(Volunteer, volunteer_id)

    async def get_by_user_id(
        self, user_id: int
    ) -> Optional[Volunteer]:
        result = await self.db.execute(
            select(Volunteer).where(Volunteer.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def list_volunteers(
        self,
        district_id: Optional[int] = None,
        status: Optional[str] = None,
        available_only: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[Volunteer]:
        query = select(Volunteer)

        if district_id is not None:
            query = query.where(
                Volunteer.district_id == district_id
            )

        if status is not None:
            query = query.where(Volunteer.status == status)

        if available_only:
            query = query.where(
                Volunteer.is_available.is_(True)
            )

        query = (
            query.order_by(Volunteer.id.desc())
            .offset(max(skip, 0))
            .limit(min(max(limit, 1), 500))
        )

        result = await self.db.execute(query)
        return result.scalars().all()

    async def update(
        self,
        volunteer_id: int,
        data: dict[str, Any],
    ) -> Optional[Volunteer]:
        volunteer = await self.get_by_id(volunteer_id)

        if volunteer is None:
            return None

        protected = {"id", "user_id", "created_at"}

        for key, value in data.items():
            if key not in protected and hasattr(volunteer, key):
                setattr(volunteer, key, value)

        if hasattr(volunteer, "updated_at"):
            volunteer.updated_at = utc_now()

        await self.db.flush()
        await self.db.refresh(volunteer)
        return volunteer

    async def set_availability(
        self,
        volunteer_id: int,
        available: bool,
    ) -> Optional[Volunteer]:
        return await self.update(
            volunteer_id,
            {"is_available": available},
        )

    async def update_status(
        self,
        volunteer_id: int,
        status: str,
    ) -> Optional[Volunteer]:
        return await self.update(
            volunteer_id,
            {"status": status},
        )

    async def assign_mission(
        self,
        volunteer_id: int,
        mission_id: int,
    ) -> Optional[Volunteer]:
        volunteer = await self.get_by_id(volunteer_id)

        if volunteer is None or not volunteer.is_available:
            return None

        return await self.update(
            volunteer_id,
            {
                "assigned_mission_id": mission_id,
                "is_available": False,
                "status": "assigned",
            },
        )

    async def release_mission(
        self,
        volunteer_id: int,
    ) -> Optional[Volunteer]:
        return await self.update(
            volunteer_id,
            {
                "assigned_mission_id": None,
                "is_available": True,
                "status": "available",
            },
        )

    async def update_location(
        self,
        volunteer_id: int,
        latitude: float,
        longitude: float,
    ) -> Optional[Volunteer]:
        if not (-90 <= latitude <= 90):
            raise ValueError("Invalid latitude")

        if not (-180 <= longitude <= 180):
            raise ValueError("Invalid longitude")

        return await self.update(
            volunteer_id,
            {
                "latitude": latitude,
                "longitude": longitude,
                "last_location_update": utc_now(),
            },
        )

    async def count(
        self,
        district_id: Optional[int] = None,
    ) -> int:
        query = select(func.count(Volunteer.id))

        if district_id is not None:
            query = query.where(
                Volunteer.district_id == district_id
            )

        result = await self.db.execute(query)
        return result.scalar_one()

    async def count_available(
        self,
        district_id: Optional[int] = None,
    ) -> int:
        query = select(func.count(Volunteer.id)).where(
            Volunteer.is_available.is_(True)
        )

        if district_id is not None:
            query = query.where(
                Volunteer.district_id == district_id
            )

        result = await self.db.execute(query)
        return result.scalar_one()

    async def get_statistics(
        self,
        district_id: Optional[int] = None,
    ) -> dict[str, int]:
        total = await self.count(district_id)
        available = await self.count_available(district_id)

        return {
            "total_volunteers": total,
            "available_volunteers": available,
            "unavailable_volunteers": total - available,
        }

    async def delete(self, volunteer_id: int) -> bool:
        volunteer = await self.get_by_id(volunteer_id)

        if volunteer is None:
            return False

        await self.db.delete(volunteer)
        await self.db.flush()
        return True
