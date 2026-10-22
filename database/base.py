
"""
Kerala Flood Relief Coordination Agent
File: backend/database/base.py

SQLAlchemy 2.x database foundation.

Features:
- Declarative ORM base
- UUID primary keys
- UTC timestamps
- Automatic created_at / updated_at
- Soft deletion
- Optimistic locking
- Audit metadata
- PostgreSQL-compatible models
- Naming conventions for Alembic
- Common database mixins

Dependencies:
    pip install sqlalchemy>=2.0 asyncpg alembic

Note:
    Database connections belong in backend/database/session.py.
    ORM models inherit from Base and the required mixins.
"""


from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    MetaData,
    String,
    event,
    func,
    inspect,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
)


# ============================================================
# UTC UTILITIES
# ============================================================

def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# ============================================================
# NAMING CONVENTIONS
# ============================================================

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": (
        "fk_%(table_name)s_"
        "%(column_0_name)s_"
        "%(referred_table_name)s"
    ),
    "pk": "pk_%(table_name)s",
}


metadata = MetaData(
    naming_convention=NAMING_CONVENTION
)


# ============================================================
# DECLARATIVE BASE
# ============================================================

class Base(DeclarativeBase):
    """
    Shared declarative base for all SQLAlchemy ORM models.

    Example:

        class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
            __tablename__ = "users"

            email: Mapped[str] = mapped_column(
                String(255),
                unique=True,
                nullable=False,
            )
    """

    metadata = metadata

    type_annotation_map = {
        UUID: PG_UUID(as_uuid=True),
    }

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Generate a default snake_case table name.

        Explicit __tablename__ declarations override this.
        """
        import re

        name = cls.__name__

        name = re.sub(
            r"(.)([A-Z][a-z]+)",
            r"\1_\2",
            name,
        )

        name = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            name,
        )

        return name.lower()

    def to_dict(
        self,
        *,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Convert mapped columns to a dictionary.

        WARNING:
            Internal use only.
            Do not return directly from public API endpoints.
            Models may contain passwords, tokens, patient
            information, private locations or other PII.
        """
        excluded = exclude or set()

        mapper = inspect(self).mapper

        return {
            column.key: getattr(self, column.key)
            for column in mapper.column_attrs
            if column.key not in excluded
        }


# ============================================================
# UUID PRIMARY KEY MIXIN
# ============================================================

class UUIDPrimaryKeyMixin:
    """
    UUID primary key shared by database models.
    """

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )


# ============================================================
# TIMESTAMP MIXIN
# ============================================================

class TimestampMixin:
    """
    Automatic creation and update timestamps.

    Uses database server timestamps.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ============================================================
# SOFT DELETE MIXIN
# ============================================================

class SoftDeleteMixin:
    """
    Soft deletion support.

    IMPORTANT:
        Soft-deleted rows are not automatically excluded
        from SQLAlchemy queries.

        Repository methods must explicitly filter
        is_deleted=False or apply a centralized
        loader-criteria policy.
    """

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
        index=True,
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deleted_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )

    def soft_delete(
        self,
        *,
        deleted_by: UUID | None = None,
    ) -> None:
        self.is_deleted = True
        self.deleted_at = utc_now()
        self.deleted_by = deleted_by

    def restore(self) -> None:
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None


# ============================================================
# AUDIT METADATA MIXIN
# ============================================================

class AuditMixin:
    """
    Store the identity of the actor responsible for changes.

    Set these fields from authenticated application context,
    not directly from untrusted client input.
    """

    created_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )

    updated_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )


# ============================================================
# VERIFICATION MIXIN
# ============================================================

class VerificationMixin:
    """
    Verification metadata for rescue teams,
    shelters, volunteers and other resources.

    Verification must be performed by an authorized actor.
    """

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
        index=True,
    )

    verified_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def mark_verified(
        self,
        *,
        verified_by: UUID,
    ) -> None:
        self.is_verified = True
        self.verified_by = verified_by
        self.verified_at = utc_now()

    def revoke_verification(self) -> None:
        self.is_verified = False
        self.verified_by = None
        self.verified_at = None


# ============================================================
# ACTIVE STATUS MIXIN
# ============================================================

class ActiveMixin:
    """
    Generic activation/deactivation support.
    """

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="true",
        nullable=False,
        index=True,
    )


# ============================================================
# OPTIMISTIC LOCKING MIXIN
# ============================================================

class VersionMixin:
    """
    Optimistic concurrency control.

    Useful for:
    - Shelter capacity
    - Rescue assignments
    - Relief inventory
    - Ambulance reservations
    - Volunteer scheduling

    Prevents silent overwrites of concurrent ORM updates.

    For stock/capacity reservations, use this together
    with database transactions and appropriate row locks.
    """

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )

    @declared_attr.directive
    def __mapper_args__(cls):
        return {
            "version_id_col": cls.version,
        }


# ============================================================
# EXTERNAL SOURCE MIXIN
# ============================================================

class ExternalSourceMixin:
    """
    Provenance for imported records.

    Example providers:
    - IMD
    - CWC
    - KSDMA
    - SACHET
    - Open-Meteo
    - Authorized field data sources
    """

    source_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    source_reference: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    source_observed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    source_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
    )


# ============================================================
# OPTIONAL COMMON ENTITY BASE
# ============================================================

class BaseEntity(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Abstract base for ordinary entities.

    Models can inherit from BaseEntity to obtain:
    - UUID primary key
    - created_at
    - updated_at

    Example:

        class District(BaseEntity):
            __tablename__ = "districts"

            name: Mapped[str] = mapped_column(
                String(100),
                unique=True,
                nullable=False,
            )
    """

    __abstract__ = True


# ============================================================
# OPTIONAL AUDITED ENTITY BASE
# ============================================================

class AuditedEntity(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditMixin,
    SoftDeleteMixin,
    Base,
):
    """
    Abstract entity with timestamps, audit metadata
    and soft-deletion support.
    """

    __abstract__ = True


# ============================================================
# OPTIONAL CONCURRENT ENTITY BASE
# ============================================================

class ConcurrentEntity(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    VersionMixin,
    Base,
):
    """
    Abstract entity with optimistic locking.

    Recommended for capacity and inventory records.
    """

    __abstract__ = True


# ============================================================
# DATABASE METADATA HELPERS
# ============================================================

def get_metadata() -> MetaData:
    """
    Return shared SQLAlchemy metadata.

    Used by Alembic migrations.
    """
    return Base.metadata


def get_registered_tables() -> list[str]:
    """
    Return names of models imported into Base.metadata.

    All ORM modules must be imported before invoking
    this helper for complete results.
    """
    return sorted(Base.metadata.tables.keys())


def get_table_count() -> int:
    return len(Base.metadata.tables)


# ============================================================
# MODEL IMPORT REGISTRATION
# ============================================================

def import_all_models() -> None:
    """
    Import every ORM model module.

    Call this from Alembic env.py before setting
    target_metadata, or during application startup.

    Missing modules are not silently ignored:
    import errors should fail migrations clearly.
    """
    from importlib import import_module

    model_modules = [
        "backend.models.user",
        "backend.models.district",
        "backend.models.flood_alert",
        "backend.models.weather_station",
        "backend.models.river_station",
        "backend.models.dam",
        "backend.models.sos_request",
        "backend.models.incident",
        "backend.models.rescue_team",
        "backend.models.rescue_mission",
        "backend.models.evacuation",
        "backend.models.shelter",
        "backend.models.shelter_resident",
        "backend.models.inventory",
        "backend.models.relief_delivery",
        "backend.models.hospital",
        "backend.models.volunteer",
        "backend.models.missing_person",
        "backend.models.damage_report",
        "backend.models.agent_decision",
        "backend.models.audit_log",
    ]

    for module_name in model_modules:
        import_module(module_name)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "Base",
    "BaseEntity",
    "AuditedEntity",
    "ConcurrentEntity",
    "UUIDPrimaryKeyMixin",
    "TimestampMixin",
    "SoftDeleteMixin",
    "AuditMixin",
    "VerificationMixin",
    "ActiveMixin",
    "VersionMixin",
    "ExternalSourceMixin",
    "NAMING_CONVENTION",
    "metadata",
    "utc_now",
    "get_metadata",
    "get_registered_tables",
    "get_table_count",
    "import_all_models",
]
