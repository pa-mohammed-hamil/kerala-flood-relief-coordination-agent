from __future__ import annotations

"""
backend/models/missing_person.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x models for missing-person reporting,
search coordination, sightings, and reunification.

Features:
- Missing-person case registration
- District and last-known-location tracking
- Flood-related disappearance details
- Physical descriptions and clothing information
- Vulnerability and assistance indicators
- Privacy-aware reporter and contact references
- Verified sightings and evidence references
- Search and rescue coordination
- Human-reviewed potential matches
- Family notification and reunification records
- Case status history
- Audit references and optimistic concurrency

SAFETY AND PRIVACY:
- AI matches are leads, not identity confirmations.
- Do not publicly expose personal contact details,
  medical information, or exact locations of children.
- Sensitive records require role-based authorization.
- Identity and reunification require human verification.
- Emergency rescue dispatch is handled by authorized
  services, not by these database models.
"""


import enum
import math
import uuid

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    event,
)

from sqlalchemy.ext.mutable import MutableDict

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from backend.database.base import Base


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(
    enum_class: type[enum.Enum],
) -> SQLEnum:
    return SQLEnum(
        enum_class,
        values_callable=lambda members: [
            member.value for member in members
        ],
        native_enum=False,
        validate_strings=True,
        length=80,
    )


def require_text(
    value: str,
    field_name: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} is required"
        )

    result = value.strip()

    if len(result) > maximum:
        raise ValueError(
            f"{field_name} exceeds {maximum} characters"
        )

    return result


def validate_probability(
    value: float | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )

    return float(value)


def validate_coordinate(
    value: float | None,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"Invalid {field_name}"
        )

    return float(value)


def validate_non_negative_int(
    value: int | None,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be non-negative"
        )

    return value


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class MissingPersonStatus(str, enum.Enum):
    REPORTED = "reported"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    SEARCH_ACTIVE = "search_active"
    POTENTIAL_MATCH = "potential_match"
    LOCATED = "located"
    REUNIFICATION_PENDING = "reunification_pending"
    REUNITED = "reunited"
    TRANSFERRED_TO_AUTHORITY = "transferred_to_authority"
    CLOSED = "closed"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class MissingPersonPriority(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MissingPersonIncidentType(str, enum.Enum):
    FLOOD = "flood"
    FLASH_FLOOD = "flash_flood"
    LANDSLIDE = "landslide"
    EVACUATION = "evacuation"
    RESCUE_OPERATION = "rescue_operation"
    SHELTER_DISPLACEMENT = "shelter_displacement"
    ROAD_INCIDENT = "road_incident"
    OTHER = "other"
    UNKNOWN = "unknown"


class MissingPersonAgeGroup(str, enum.Enum):
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    OLDER_ADULT = "older_adult"
    UNKNOWN = "unknown"


class MissingPersonVerificationStatus(str, enum.Enum):
    UNVERIFIED = "unverified"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    REJECTED = "rejected"


class MissingPersonSightingStatus(str, enum.Enum):
    REPORTED = "reported"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class MissingPersonMatchStatus(str, enum.Enum):
    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    POSSIBLE_MATCH = "possible_match"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class MissingPersonMatchMethod(str, enum.Enum):
    MANUAL = "manual"
    NAME_SIMILARITY = "name_similarity"
    DESCRIPTION_SIMILARITY = "description_similarity"
    IMAGE_SIMILARITY = "image_similarity"
    LOCATION_CORRELATION = "location_correlation"
    MULTIMODAL_AI = "multimodal_ai"
    OFFICIAL_RECORD = "official_record"
    OTHER = "other"


class MissingPersonSearchStatus(str, enum.Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ASSIGNED = "assigned"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MissingPersonReunificationStatus(str, enum.Enum):
    PENDING_IDENTITY_VERIFICATION = (
        "pending_identity_verification"
    )
    IDENTITY_VERIFIED = "identity_verified"
    GUARDIAN_VERIFICATION_PENDING = (
        "guardian_verification_pending"
    )
    APPROVED = "approved"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class MissingPersonEventType(str, enum.Enum):
    CREATED = "created"
    UPDATED = "updated"
    VERIFICATION_REQUESTED = "verification_requested"
    VERIFIED = "verified"
    REJECTED = "rejected"
    SEARCH_REQUESTED = "search_requested"
    SEARCH_APPROVED = "search_approved"
    SEARCH_STARTED = "search_started"
    SIGHTING_REPORTED = "sighting_reported"
    SIGHTING_VERIFIED = "sighting_verified"
    MATCH_PROPOSED = "match_proposed"
    MATCH_CONFIRMED = "match_confirmed"
    LOCATED = "located"
    IDENTITY_VERIFIED = "identity_verified"
    REUNIFICATION_REQUESTED = "reunification_requested"
    REUNIFICATION_APPROVED = "reunification_approved"
    REUNITED = "reunited"
    CLOSED = "closed"
    DUPLICATE = "duplicate"
    CANCELLED = "cancelled"
    NOTE_ADDED = "note_added"


# ============================================================
# MISSING PERSON CASE
# ============================================================

class MissingPerson(Base):
    __tablename__ = "missing_persons"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    case_number: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    district_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "districts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    incident_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    sos_request_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    evacuation_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    rescue_mission_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            nullable=True,
        )
    )

    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    full_name_malayalam: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    preferred_name: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    approximate_age: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    age_group: Mapped[
        MissingPersonAgeGroup
    ] = mapped_column(
        enum_type(MissingPersonAgeGroup),
        default=MissingPersonAgeGroup.UNKNOWN,
        nullable=False,
    )

    gender_description: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    physical_description: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    clothing_description: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    identifying_features: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    photo_storage_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    photo_consent_record_reference: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    status: Mapped[
        MissingPersonStatus
    ] = mapped_column(
        enum_type(MissingPersonStatus),
        default=MissingPersonStatus.REPORTED,
        nullable=False,
        index=True,
    )

    priority: Mapped[
        MissingPersonPriority
    ] = mapped_column(
        enum_type(MissingPersonPriority),
        default=MissingPersonPriority.NORMAL,
        nullable=False,
        index=True,
    )

    incident_type: Mapped[
        MissingPersonIncidentType
    ] = mapped_column(
        enum_type(MissingPersonIncidentType),
        default=MissingPersonIncidentType.UNKNOWN,
        nullable=False,
    )

    verification_status: Mapped[
        MissingPersonVerificationStatus
    ] = mapped_column(
        enum_type(MissingPersonVerificationStatus),
        default=MissingPersonVerificationStatus.UNVERIFIED,
        nullable=False,
    )

    last_seen_location_name: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    last_seen_address: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    last_seen_taluk: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    last_seen_panchayat: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    last_seen_latitude: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    last_seen_longitude: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    last_seen_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    disappearance_description: Mapped[
        str
    ] = mapped_column(
        Text,
        nullable=False,
    )

    possible_destination: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    requires_medical_assistance: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    mobility_assistance_required: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    communication_assistance_required: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_minor: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_protected_handling: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    reporter_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    reporter_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    reporter_relationship: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    reporter_contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    emergency_contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    guardian_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    police_report_reference: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    official_case_reference: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    assigned_coordinator_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    located_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    located_location_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    located_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    resolution_summary: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    closure_approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    closed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    duplicate_of_case_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "missing_persons.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    reported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    audit_reference: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    sightings: Mapped[
        list["MissingPersonSighting"]
    ] = relationship(
        "MissingPersonSighting",
        back_populates="missing_person",
        cascade="save-update, merge",
    )

    matches: Mapped[
        list["MissingPersonMatch"]
    ] = relationship(
        "MissingPersonMatch",
        back_populates="missing_person",
        cascade="save-update, merge",
    )

    search_assignments: Mapped[
        list["MissingPersonSearchAssignment"]
    ] = relationship(
        "MissingPersonSearchAssignment",
        back_populates="missing_person",
        cascade="save-update, merge",
    )

    reunifications: Mapped[
        list["MissingPersonReunification"]
    ] = relationship(
        "MissingPersonReunification",
        back_populates="missing_person",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["MissingPersonEvent"]
    ] = relationship(
        "MissingPersonEvent",
        back_populates="missing_person",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_missing_person_district",
        ),
        CheckConstraint(
            "approximate_age IS NULL OR "
            "(approximate_age >= 0 "
            "AND approximate_age <= 125)",
            name="ck_missing_person_age",
        ),
        CheckConstraint(
            "last_seen_latitude IS NULL OR "
            "(last_seen_latitude >= -90 "
            "AND last_seen_latitude <= 90)",
            name="ck_missing_person_lat",
        ),
        CheckConstraint(
            "last_seen_longitude IS NULL OR "
            "(last_seen_longitude >= -180 "
            "AND last_seen_longitude <= 180)",
            name="ck_missing_person_lon",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_missing_person_version",
        ),
        Index(
            "ix_missing_person_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_missing_person_priority_status",
            "priority",
            "status",
        ),
        Index(
            "ix_missing_person_last_seen",
            "district_id",
            "last_seen_at",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "case_number",
        "full_name",
        "disappearance_description",
    )
    def validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "case_number": 100,
            "full_name": 255,
            "disappearance_description": 10000,
        }

        return require_text(
            value,
            key,
            limits[key],
        )

    @validates("approximate_age")
    def validate_age(
        self,
        key: str,
        value: int | None,
    ) -> int | None:
        result = validate_non_negative_int(
            value, key
        )

        if result is not None and result > 125:
            raise ValueError(
                "Age exceeds supported range"
            )

        return result

    @validates("last_seen_latitude")
    def validate_latitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -90, 90
        )

    @validates("last_seen_longitude")
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -180, 180
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            MissingPersonStatus.CLOSED,
            MissingPersonStatus.DUPLICATE,
            MissingPersonStatus.REJECTED,
            MissingPersonStatus.CANCELLED,
        }

    @property
    def requires_high_privacy(self) -> bool:
        return (
            self.is_minor
            or self.requires_protected_handling
            or self.age_group in {
                MissingPersonAgeGroup.INFANT,
                MissingPersonAgeGroup.CHILD,
                MissingPersonAgeGroup.ADOLESCENT,
            }
        )

    def request_verification(self) -> None:
        if self.status != MissingPersonStatus.REPORTED:
            raise ValueError(
                "Case is not in reported state"
            )

        self.status = (
            MissingPersonStatus.PENDING_VERIFICATION
        )
        self.verification_status = (
            MissingPersonVerificationStatus.UNDER_REVIEW
        )

    def verify(
        self,
        *,
        verifier_id: int,
    ) -> None:
        if self.status != (
            MissingPersonStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Case is not pending verification"
            )

        if verifier_id <= 0:
            raise ValueError(
                "Invalid verifier"
            )

        if (
            self.reporter_user_id is not None
            and verifier_id == self.reporter_user_id
        ):
            raise ValueError(
                "Independent verification required"
            )

        self.verified_by_user_id = verifier_id
        self.verified_at = utc_now()
        self.verification_status = (
            MissingPersonVerificationStatus.VERIFIED
        )
        self.status = MissingPersonStatus.VERIFIED

    def activate_search(self) -> None:
        if self.status != MissingPersonStatus.VERIFIED:
            raise ValueError(
                "Case must be verified first"
            )

        self.status = MissingPersonStatus.SEARCH_ACTIVE

    def flag_potential_match(self) -> None:
        if self.status not in {
            MissingPersonStatus.VERIFIED,
            MissingPersonStatus.SEARCH_ACTIVE,
        }:
            raise ValueError(
                "Case cannot enter potential-match state"
            )

        self.status = MissingPersonStatus.POTENTIAL_MATCH

    def confirm_located(
        self,
        *,
        verifier_id: int,
        location_reference: str,
    ) -> None:
        if self.status not in {
            MissingPersonStatus.SEARCH_ACTIVE,
            MissingPersonStatus.POTENTIAL_MATCH,
            MissingPersonStatus.VERIFIED,
        }:
            raise ValueError(
                "Case cannot be marked located"
            )

        if verifier_id <= 0:
            raise ValueError(
                "Invalid verifier"
            )

        self.located_location_reference = (
            require_text(
                location_reference,
                "location_reference",
                500,
            )
        )

        self.located_verified_by_user_id = verifier_id
        self.located_at = utc_now()
        self.status = MissingPersonStatus.LOCATED

    def begin_reunification(self) -> None:
        if self.status != MissingPersonStatus.LOCATED:
            raise ValueError(
                "Person must be located first"
            )

        self.status = (
            MissingPersonStatus.REUNIFICATION_PENDING
        )

    def mark_reunited(
        self,
        *,
        approved_reunification: (
            "MissingPersonReunification"
        ),
    ) -> None:
        if self.status != (
            MissingPersonStatus.REUNIFICATION_PENDING
        ):
            raise ValueError(
                "Reunification is not pending"
            )

        if (
            approved_reunification.missing_person_id
            != self.id
        ):
            raise ValueError(
                "Reunification belongs to another case"
            )

        if approved_reunification.status != (
            MissingPersonReunificationStatus.COMPLETED
        ):
            raise ValueError(
                "Reunification has not been completed"
            )

        self.status = MissingPersonStatus.REUNITED

    def close(
        self,
        *,
        approver_id: int,
        resolution_summary: str,
    ) -> None:
        if self.status not in {
            MissingPersonStatus.REUNITED,
            MissingPersonStatus.TRANSFERRED_TO_AUTHORITY,
        }:
            raise ValueError(
                "Case is not eligible for closure"
            )

        if approver_id <= 0:
            raise ValueError(
                "Invalid closure approver"
            )

        if approver_id == (
            self.located_verified_by_user_id
        ):
            raise ValueError(
                "Independent closure approval required"
            )

        self.resolution_summary = require_text(
            resolution_summary,
            "resolution_summary",
            10000,
        )

        self.closure_approved_by_user_id = approver_id
        self.closed_at = utc_now()
        self.status = MissingPersonStatus.CLOSED

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Minimal privacy-preserving representation.

        This is not a public missing-person bulletin.
        """
        return {
            "id": self.id,
            "case_number": self.case_number,
            "district_id": self.district_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "age_group": self.age_group.value,
            "incident_type": self.incident_type.value,
            "verification_status": (
                self.verification_status.value
            ),
            "requires_high_privacy": (
                self.requires_high_privacy
            ),
            "reported_at": (
                self.reported_at.isoformat()
                if self.reported_at
                else None
            ),
        }


# ============================================================
# MISSING PERSON SIGHTING
# ============================================================

class MissingPersonSighting(Base):
    __tablename__ = "missing_person_sightings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    missing_person_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "missing_persons.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    sighting_number: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    location_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    sighted_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    reporter_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    reporter_contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    evidence_storage_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    status: Mapped[
        MissingPersonSightingStatus
    ] = mapped_column(
        enum_type(MissingPersonSightingStatus),
        default=MissingPersonSightingStatus.REPORTED,
        nullable=False,
    )

    confidence_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    is_ai_assisted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    review_notes: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    missing_person: Mapped["MissingPerson"] = (
        relationship(
            "MissingPerson",
            back_populates="sightings",
        )
    )

    __table_args__ = (
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 "
            "AND confidence_score <= 1)",
            name="ck_missing_sighting_confidence",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_missing_sighting_lat",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_missing_sighting_lon",
        ),
        Index(
            "ix_missing_sighting_status_time",
            "status",
            "created_at",
        ),
    )

    @validates("confidence_score")
    def validate_confidence(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_probability(value, key)

    @validates("latitude")
    def validate_latitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -90, 90
        )

    @validates("longitude")
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -180, 180
        )


# ============================================================
# POTENTIAL MATCH
# ============================================================

class MissingPersonMatch(Base):
    __tablename__ = "missing_person_matches"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    missing_person_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "missing_persons.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    sighting_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "missing_person_sightings.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    candidate_reference: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    method: Mapped[
        MissingPersonMatchMethod
    ] = mapped_column(
        enum_type(MissingPersonMatchMethod),
        nullable=False,
    )

    status: Mapped[
        MissingPersonMatchStatus
    ] = mapped_column(
        enum_type(MissingPersonMatchStatus),
        default=MissingPersonMatchStatus.PROPOSED,
        nullable=False,
    )

    confidence_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    evidence_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    reasoning_summary: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    limitations: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    requires_human_review: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    reviewed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    reviewed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    identity_verification_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    review_notes: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    missing_person: Mapped["MissingPerson"] = (
        relationship(
            "MissingPerson",
            back_populates="matches",
        )
    )

    __table_args__ = (
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 "
            "AND confidence_score <= 1)",
            name="ck_missing_match_confidence",
        ),
        Index(
            "ix_missing_match_status",
            "missing_person_id",
            "status",
        ),
    )

    @validates("confidence_score")
    def validate_confidence(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_probability(value, key)

    def confirm(
        self,
        *,
        reviewer_id: int,
        identity_verification_reference: str,
    ) -> None:
        if self.status not in {
            MissingPersonMatchStatus.UNDER_REVIEW,
            MissingPersonMatchStatus.POSSIBLE_MATCH,
        }:
            raise ValueError(
                "Match is not awaiting confirmation"
            )

        if reviewer_id <= 0:
            raise ValueError("Invalid reviewer")

        self.identity_verification_reference = (
            require_text(
                identity_verification_reference,
                "identity_verification_reference",
                500,
            )
        )

        self.reviewed_by_user_id = reviewer_id
        self.reviewed_at = utc_now()
        self.status = MissingPersonMatchStatus.CONFIRMED


# ============================================================
# SEARCH ASSIGNMENT
# ============================================================

class MissingPersonSearchAssignment(Base):
    __tablename__ = "missing_person_search_assignments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    missing_person_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "missing_persons.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    rescue_team_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            nullable=True,
        )
    )

    volunteer_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            nullable=True,
        )
    )

    assigned_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    search_area_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    instructions: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[
        MissingPersonSearchStatus
    ] = mapped_column(
        enum_type(MissingPersonSearchStatus),
        default=MissingPersonSearchStatus.PROPOSED,
        nullable=False,
    )

    requested_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    started_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    completed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    result_summary: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    missing_person: Mapped["MissingPerson"] = (
        relationship(
            "MissingPerson",
            back_populates="search_assignments",
        )
    )

    __table_args__ = (
        Index(
            "ix_missing_search_status",
            "missing_person_id",
            "status",
        ),
    )

    def approve(self, approver_id: int) -> None:
        if self.status != (
            MissingPersonSearchStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Search is not pending approval"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if (
            self.requested_by_user_id is not None
            and approver_id == self.requested_by_user_id
        ):
            raise ValueError(
                "Independent approval required"
            )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = MissingPersonSearchStatus.APPROVED


# ============================================================
# REUNIFICATION RECORD
# ============================================================

class MissingPersonReunification(Base):
    __tablename__ = "missing_person_reunifications"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    missing_person_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "missing_persons.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    recipient_reference: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    relationship_description: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    recipient_contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    status: Mapped[
        MissingPersonReunificationStatus
    ] = mapped_column(
        enum_type(MissingPersonReunificationStatus),
        default=(
            MissingPersonReunificationStatus
            .PENDING_IDENTITY_VERIFICATION
        ),
        nullable=False,
    )

    person_identity_verified: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    recipient_identity_verified: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    guardian_authority_verified: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    safeguarding_review_completed: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    person_identity_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    recipient_identity_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    verification_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    completed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    completed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    handover_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    missing_person: Mapped["MissingPerson"] = (
        relationship(
            "MissingPerson",
            back_populates="reunifications",
        )
    )

    __table_args__ = (
        Index(
            "ix_missing_reunification_status",
            "missing_person_id",
            "status",
        ),
    )

    def approve(
        self,
        *,
        approver_id: int,
    ) -> None:
        if self.status not in {
            MissingPersonReunificationStatus.IDENTITY_VERIFIED,
            MissingPersonReunificationStatus
            .GUARDIAN_VERIFICATION_PENDING,
        }:
            raise ValueError(
                "Reunification is not ready for approval"
            )

        if not (
            self.person_identity_verified
            and self.recipient_identity_verified
            and self.safeguarding_review_completed
        ):
            raise ValueError(
                "Required verification is incomplete"
            )

        if (
            self.missing_person.requires_high_privacy
            and not self.guardian_authority_verified
        ):
            raise ValueError(
                "Protected-person authorization required"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if approver_id == self.verification_by_user_id:
            raise ValueError(
                "Independent approval required"
            )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = (
            MissingPersonReunificationStatus.APPROVED
        )

    def complete(
        self,
        *,
        completed_by_user_id: int,
        handover_reference: str,
    ) -> None:
        if self.status != (
            MissingPersonReunificationStatus.APPROVED
        ):
            raise ValueError(
                "Reunification must be approved"
            )

        if completed_by_user_id <= 0:
            raise ValueError(
                "Invalid completion user"
            )

        self.handover_reference = require_text(
            handover_reference,
            "handover_reference",
            500,
        )

        self.completed_by_user_id = (
            completed_by_user_id
        )
        self.completed_at = utc_now()
        self.status = (
            MissingPersonReunificationStatus.COMPLETED
        )


# ============================================================
# CASE EVENT HISTORY
# ============================================================

class MissingPersonEvent(Base):
    __tablename__ = "missing_person_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    missing_person_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "missing_persons.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        MissingPersonEventType
    ] = mapped_column(
        enum_type(MissingPersonEventType),
        nullable=False,
    )

    previous_status: Mapped[
        MissingPersonStatus | None
    ] = mapped_column(
        enum_type(MissingPersonStatus),
        nullable=True,
    )

    new_status: Mapped[
        MissingPersonStatus | None
    ] = mapped_column(
        enum_type(MissingPersonStatus),
        nullable=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    actor_agent_name: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    missing_person: Mapped["MissingPerson"] = (
        relationship(
            "MissingPerson",
            back_populates="events",
        )
    )

    __table_args__ = (
        Index(
            "ix_missing_person_event_time",
            "missing_person_id",
            "occurred_at",
        ),
    )


# ============================================================
# ORM EVENTS
# ============================================================

@event.listens_for(
    MissingPerson,
    "before_update",
)
def update_missing_person_timestamp(
    mapper: Any,
    connection: Any,
    target: MissingPerson,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

MISSING_PERSON_MODELS = (
    MissingPerson,
    MissingPersonSighting,
    MissingPersonMatch,
    MissingPersonSearchAssignment,
    MissingPersonReunification,
    MissingPersonEvent,
)


__all__ = [
    "MissingPerson",
    "MissingPersonSighting",
    "MissingPersonMatch",
    "MissingPersonSearchAssignment",
    "MissingPersonReunification",
    "MissingPersonEvent",
    "MissingPersonStatus",
    "MissingPersonPriority",
    "MissingPersonIncidentType",
    "MissingPersonAgeGroup",
    "MissingPersonVerificationStatus",
    "MissingPersonSightingStatus",
    "MissingPersonMatchStatus",
    "MissingPersonMatchMethod",
    "MissingPersonSearchStatus",
    "MissingPersonReunificationStatus",
    "MissingPersonEventType",
    "MISSING_PERSON_MODELS",
]
