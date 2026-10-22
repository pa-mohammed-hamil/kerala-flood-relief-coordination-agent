from __future__ import annotations

"""
backend/models/volunteer.py

Kerala Flood Relief Coordination Agent
Python 3.11+ | SQLAlchemy 2.x

Volunteer management:
- Registration and verification
- Skills, qualifications and certifications
- Availability and duty scheduling
- District and organization assignments
- Rescue and relief task assignments
- Training and safety readiness
- Check-in/check-out and location updates
- Equipment allocation
- Welfare and incident reporting
- Performance and contribution metrics
- Human-approved emergency deployment
- Audit event history

Safety:
- Volunteers cannot self-authorize rescue deployments.
- High-risk tasks require verified qualifications.
- AI recommendations are advisory only.
- Sensitive contact information is stored by encrypted reference.
- Live location data requires access control.
- Emergency decisions require authorized human oversight.
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


def required_text(
    value: str,
    field_name: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")

    value = value.strip()

    if len(value) > maximum:
        raise ValueError(
            f"{field_name} exceeds {maximum} characters"
        )

    return value


def valid_actor_id(value: str | None) -> str:
    if not value:
        raise ValueError("Authorized actor ID required")

    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError):
        raise ValueError(
            "Invalid authorized actor UUID"
        ) from None


def valid_coordinate(
    value: float | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Invalid {field_name}")

    lower, upper = (
        (-90, 90)
        if field_name == "latitude"
        else (-180, 180)
    )

    if not lower <= value <= upper:
        raise ValueError(f"Invalid {field_name}")

    return float(value)


def normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class VolunteerStatus(str, enum.Enum):
    REGISTERED = "registered"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ON_DUTY = "on_duty"
    DEPLOYED = "deployed"
    RESTING = "resting"
    SUSPENDED = "suspended"
    BLOCKED = "blocked"
    WITHDRAWN = "withdrawn"


class VolunteerType(str, enum.Enum):
    GENERAL = "general"
    COMMUNITY = "community"
    NGO = "ngo"
    STUDENT = "student"
    PROFESSIONAL = "professional"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    LOGISTICS = "logistics"
    RESCUE_SUPPORT = "rescue_support"
    REMOTE = "remote"
    OTHER = "other"


class VolunteerVerificationStatus(str, enum.Enum):
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class VolunteerAvailabilityStatus(str, enum.Enum):
    AVAILABLE = "available"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"
    ON_DUTY = "on_duty"
    RESTING = "resting"
    UNKNOWN = "unknown"


class VolunteerSkillType(str, enum.Enum):
    FIRST_AID = "first_aid"
    CPR = "cpr"
    MEDICAL_SUPPORT = "medical_support"
    NURSING = "nursing"
    SEARCH_AND_RESCUE = "search_and_rescue"
    WATER_RESCUE = "water_rescue"
    BOAT_OPERATION = "boat_operation"
    SWIMMING = "swimming"
    DRIVING = "driving"
    HEAVY_VEHICLE = "heavy_vehicle"
    LOGISTICS = "logistics"
    INVENTORY = "inventory"
    FOOD_DISTRIBUTION = "food_distribution"
    SHELTER_MANAGEMENT = "shelter_management"
    COMMUNICATION = "communication"
    TRANSLATION = "translation"
    COUNSELLING_SUPPORT = "counselling_support"
    DATA_ENTRY = "data_entry"
    GIS_MAPPING = "gis_mapping"
    DRONE_OPERATION = "drone_operation"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    CONSTRUCTION = "construction"
    IT_SUPPORT = "it_support"
    CHILD_SAFEGUARDING = "child_safeguarding"
    ACCESSIBILITY_SUPPORT = "accessibility_support"
    OTHER = "other"


class VolunteerSkillLevel(str, enum.Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"
    CERTIFIED = "certified"


class VolunteerTrainingStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    ENROLLED = "enrolled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class VolunteerAssignmentType(str, enum.Enum):
    RESCUE_SUPPORT = "rescue_support"
    EVACUATION_SUPPORT = "evacuation_support"
    SHELTER_SUPPORT = "shelter_support"
    FOOD_DISTRIBUTION = "food_distribution"
    WATER_DISTRIBUTION = "water_distribution"
    MEDICAL_SUPPORT = "medical_support"
    LOGISTICS = "logistics"
    INVENTORY = "inventory"
    TRANSPORT = "transport"
    COMMUNICATION = "communication"
    DAMAGE_ASSESSMENT = "damage_assessment"
    CLEANUP = "cleanup"
    RECOVERY = "recovery"
    DATA_COLLECTION = "data_collection"
    OTHER = "other"


class VolunteerAssignmentStatus(str, enum.Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class VolunteerRiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class VolunteerShiftStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    ON_BREAK = "on_break"
    CHECKED_OUT = "checked_out"
    MISSED = "missed"
    CANCELLED = "cancelled"


class VolunteerWelfareStatus(str, enum.Enum):
    SAFE = "safe"
    NEEDS_REST = "needs_rest"
    NEEDS_ASSISTANCE = "needs_assistance"
    INJURED = "injured"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


class VolunteerEquipmentStatus(str, enum.Enum):
    REQUESTED = "requested"
    ISSUED = "issued"
    RETURNED = "returned"
    DAMAGED = "damaged"
    LOST = "lost"


class VolunteerEventType(str, enum.Enum):
    REGISTERED = "registered"
    VERIFICATION_REQUESTED = "verification_requested"
    VERIFIED = "verified"
    ACTIVATED = "activated"
    SUSPENDED = "suspended"
    PROFILE_UPDATED = "profile_updated"
    SKILL_ADDED = "skill_added"
    TRAINING_COMPLETED = "training_completed"
    AVAILABILITY_UPDATED = "availability_updated"
    SHIFT_SCHEDULED = "shift_scheduled"
    CHECKED_IN = "checked_in"
    CHECKED_OUT = "checked_out"
    ASSIGNMENT_PROPOSED = "assignment_proposed"
    ASSIGNMENT_APPROVED = "assignment_approved"
    ASSIGNMENT_STARTED = "assignment_started"
    ASSIGNMENT_COMPLETED = "assignment_completed"
    LOCATION_UPDATED = "location_updated"
    EQUIPMENT_ISSUED = "equipment_issued"
    EQUIPMENT_RETURNED = "equipment_returned"
    WELFARE_CHECK = "welfare_check"
    SAFETY_INCIDENT = "safety_incident"
    NOTE_ADDED = "note_added"


# ============================================================
# VOLUNTEER
# ============================================================

class Volunteer(Base):
    __tablename__ = "volunteers"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
        index=True,
    )

    user_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        unique=True,
    )

    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    volunteer_type: Mapped[
        VolunteerType
    ] = mapped_column(
        enum_type(VolunteerType),
        default=VolunteerType.GENERAL,
        nullable=False,
    )

    status: Mapped[
        VolunteerStatus
    ] = mapped_column(
        enum_type(VolunteerStatus),
        default=VolunteerStatus.REGISTERED,
        nullable=False,
        index=True,
    )

    verification_status: Mapped[
        VolunteerVerificationStatus
    ] = mapped_column(
        enum_type(VolunteerVerificationStatus),
        default=VolunteerVerificationStatus.UNVERIFIED,
        nullable=False,
    )

    availability_status: Mapped[
        VolunteerAvailabilityStatus
    ] = mapped_column(
        enum_type(VolunteerAvailabilityStatus),
        default=VolunteerAvailabilityStatus.UNKNOWN,
        nullable=False,
    )

    welfare_status: Mapped[
        VolunteerWelfareStatus
    ] = mapped_column(
        enum_type(VolunteerWelfareStatus),
        default=VolunteerWelfareStatus.UNKNOWN,
        nullable=False,
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

    organization_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    assigned_district_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        ForeignKey(
            "districts.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    encrypted_phone_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    encrypted_email_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    encrypted_address_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    encrypted_identity_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    encrypted_emergency_contact_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    preferred_language: Mapped[str] = mapped_column(
        String(20),
        default="ml",
        nullable=False,
    )

    has_transport: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_driving_license: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_medical_training: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_rescue_training: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_available_for_deployment: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    accepts_high_risk_tasks: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_safety_orientation: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_required_ppe: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    consent_for_location_tracking: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    consent_for_emergency_contact: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    privacy_notice_acknowledged: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    verified_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    activated_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    last_check_in_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    last_check_out_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    total_completed_assignments: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_service_minutes: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    registered_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
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

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    skills: Mapped[
        list["VolunteerSkill"]
    ] = relationship(
        "VolunteerSkill",
        back_populates="volunteer",
    )

    trainings: Mapped[
        list["VolunteerTraining"]
    ] = relationship(
        "VolunteerTraining",
        back_populates="volunteer",
    )

    availabilities: Mapped[
        list["VolunteerAvailability"]
    ] = relationship(
        "VolunteerAvailability",
        back_populates="volunteer",
    )

    assignments: Mapped[
        list["VolunteerAssignment"]
    ] = relationship(
        "VolunteerAssignment",
        back_populates="volunteer",
    )

    shifts: Mapped[
        list["VolunteerShift"]
    ] = relationship(
        "VolunteerShift",
        back_populates="volunteer",
    )

    locations: Mapped[
        list["VolunteerLocation"]
    ] = relationship(
        "VolunteerLocation",
        back_populates="volunteer",
    )

    equipment: Mapped[
        list["VolunteerEquipment"]
    ] = relationship(
        "VolunteerEquipment",
        back_populates="volunteer",
    )

    welfare_checks: Mapped[
        list["VolunteerWelfareCheck"]
    ] = relationship(
        "VolunteerWelfareCheck",
        back_populates="volunteer",
    )

    events: Mapped[
        list["VolunteerEvent"]
    ] = relationship(
        "VolunteerEvent",
        back_populates="volunteer",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_volunteer_district",
        ),
        CheckConstraint(
            "total_completed_assignments >= 0",
            name="ck_volunteer_completed_assignments",
        ),
        CheckConstraint(
            "total_service_minutes >= 0",
            name="ck_volunteer_service_minutes",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_volunteer_version",
        ),
        Index(
            "ix_volunteer_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_volunteer_availability",
            "availability_status",
            "is_available_for_deployment",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("volunteer_number", "display_name")
    def validate_text(
        self,
        key: str,
        value: str,
    ) -> str:
        return required_text(
            value,
            key,
            120 if key == "volunteer_number" else 255,
        )

    @property
    def is_verified(self) -> bool:
        return (
            self.verification_status
            == VolunteerVerificationStatus.VERIFIED
        )

    @property
    def can_be_deployed(self) -> bool:
        return (
            self.is_verified
            and self.status in {
                VolunteerStatus.ACTIVE,
                VolunteerStatus.VERIFIED,
            }
            and self.availability_status
            == VolunteerAvailabilityStatus.AVAILABLE
            and self.is_available_for_deployment
            and self.has_safety_orientation
            and self.welfare_status
            not in {
                VolunteerWelfareStatus.INJURED,
                VolunteerWelfareStatus.UNREACHABLE,
                VolunteerWelfareStatus.NEEDS_REST,
            }
        )

    def request_verification(self) -> None:
        if self.status != VolunteerStatus.REGISTERED:
            raise ValueError(
                "Volunteer cannot request verification"
            )

        self.status = VolunteerStatus.PENDING_VERIFICATION
        self.verification_status = (
            VolunteerVerificationStatus.PENDING
        )

    def verify(
        self,
        authorized_user_id: str,
    ) -> None:
        actor = valid_actor_id(authorized_user_id)

        if self.status != (
            VolunteerStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Volunteer is not pending verification"
            )

        if self.registered_by_user_id == actor:
            raise ValueError(
                "Independent verification required"
            )

        self.verified_by_user_id = actor
        self.verified_at = utc_now()
        self.verification_status = (
            VolunteerVerificationStatus.VERIFIED
        )
        self.status = VolunteerStatus.VERIFIED

    def activate(
        self,
        authorized_user_id: str,
    ) -> None:
        actor = valid_actor_id(authorized_user_id)

        if not self.is_verified:
            raise ValueError(
                "Volunteer verification required"
            )

        if not self.has_safety_orientation:
            raise ValueError(
                "Safety orientation required"
            )

        if self.status not in {
            VolunteerStatus.VERIFIED,
            VolunteerStatus.INACTIVE,
        }:
            raise ValueError(
                "Volunteer cannot be activated"
            )

        self.activated_by_user_id = actor
        self.activated_at = utc_now()
        self.status = VolunteerStatus.ACTIVE

    def suspend(self) -> None:
        if self.status in {
            VolunteerStatus.BLOCKED,
            VolunteerStatus.WITHDRAWN,
        }:
            raise ValueError(
                "Volunteer cannot be suspended"
            )

        self.status = VolunteerStatus.SUSPENDED
        self.is_available_for_deployment = False

    def withdraw(self) -> None:
        self.status = VolunteerStatus.WITHDRAWN
        self.is_available_for_deployment = False

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "volunteer_number": self.volunteer_number,
            "display_name": self.display_name,
            "volunteer_type": self.volunteer_type.value,
            "status": self.status.value,
            "verification_status": (
                self.verification_status.value
            ),
            "availability_status": (
                self.availability_status.value
            ),
            "district_id": self.district_id,
            "can_be_deployed": self.can_be_deployed,
            "total_completed_assignments": (
                self.total_completed_assignments
            ),
        }


# ============================================================
# VOLUNTEER SKILL
# ============================================================

class VolunteerSkill(Base):
    __tablename__ = "volunteer_skills"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    skill_type: Mapped[VolunteerSkillType] = mapped_column(
        enum_type(VolunteerSkillType),
        nullable=False,
    )

    skill_level: Mapped[
        VolunteerSkillLevel
    ] = mapped_column(
        enum_type(VolunteerSkillLevel),
        default=VolunteerSkillLevel.BEGINNER,
        nullable=False,
    )

    years_experience: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    certification_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    certification_expires_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    verified_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="skills",
    )

    __table_args__ = (
        UniqueConstraint(
            "volunteer_id",
            "skill_type",
            name="uq_volunteer_skill",
        ),
        CheckConstraint(
            "years_experience >= 0",
            name="ck_volunteer_skill_experience",
        ),
    )

    @property
    def is_currently_verified(self) -> bool:
        if not self.is_verified:
            return False

        if self.certification_expires_at is None:
            return True

        return (
            normalize_datetime(
                self.certification_expires_at
            ) > utc_now()
        )


# ============================================================
# VOLUNTEER TRAINING
# ============================================================

class VolunteerTraining(Base):
    __tablename__ = "volunteer_trainings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    training_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    training_provider: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    status: Mapped[
        VolunteerTrainingStatus
    ] = mapped_column(
        enum_type(VolunteerTrainingStatus),
        default=VolunteerTrainingStatus.NOT_STARTED,
        nullable=False,
    )

    duration_hours: Mapped[float | None] = mapped_column(
        Float,
    )

    certificate_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    verified_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="trainings",
    )

    __table_args__ = (
        CheckConstraint(
            "duration_hours IS NULL OR duration_hours >= 0",
            name="ck_volunteer_training_duration",
        ),
    )


# ============================================================
# VOLUNTEER AVAILABILITY
# ============================================================

class VolunteerAvailability(Base):
    __tablename__ = "volunteer_availabilities"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    available_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    available_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    status: Mapped[
        VolunteerAvailabilityStatus
    ] = mapped_column(
        enum_type(VolunteerAvailabilityStatus),
        default=VolunteerAvailabilityStatus.AVAILABLE,
        nullable=False,
    )

    district_id: Mapped[int | None] = mapped_column(
        Integer,
    )

    maximum_travel_distance_km: Mapped[
        float | None
    ] = mapped_column(
        Float,
    )

    remote_only: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="availabilities",
    )

    __table_args__ = (
        CheckConstraint(
            "available_until > available_from",
            name="ck_volunteer_availability_window",
        ),
        CheckConstraint(
            "maximum_travel_distance_km IS NULL OR "
            "maximum_travel_distance_km >= 0",
            name="ck_volunteer_travel_distance",
        ),
    )


# ============================================================
# VOLUNTEER ASSIGNMENT
# ============================================================

class VolunteerAssignment(Base):
    __tablename__ = "volunteer_assignments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    assignment_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    assignment_type: Mapped[
        VolunteerAssignmentType
    ] = mapped_column(
        enum_type(VolunteerAssignmentType),
        nullable=False,
    )

    status: Mapped[
        VolunteerAssignmentStatus
    ] = mapped_column(
        enum_type(VolunteerAssignmentStatus),
        default=VolunteerAssignmentStatus.PROPOSED,
        nullable=False,
        index=True,
    )

    risk_level: Mapped[
        VolunteerRiskLevel
    ] = mapped_column(
        enum_type(VolunteerRiskLevel),
        default=VolunteerRiskLevel.LOW,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    district_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "districts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    incident_id: Mapped[str | None] = mapped_column(
        String(36),
    )

    rescue_mission_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    shelter_id: Mapped[str | None] = mapped_column(
        String(36),
    )

    relief_delivery_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    requires_certification: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    required_skill: Mapped[
        VolunteerSkillType | None
    ] = mapped_column(
        enum_type(VolunteerSkillType),
    )

    qualifications_verified: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    safety_briefing_completed: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    ppe_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_ai_recommended: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    ai_recommendation_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    proposed_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    approved_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    scheduled_start_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    scheduled_end_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    completion_notes: Mapped[
        str | None
    ] = mapped_column(
        Text,
    )

    verified_completion_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
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

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="assignments",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_volunteer_assignment_district",
        ),
        CheckConstraint(
            "scheduled_end_at IS NULL OR "
            "scheduled_start_at IS NULL OR "
            "scheduled_end_at > scheduled_start_at",
            name="ck_volunteer_assignment_schedule",
        ),
        Index(
            "ix_volunteer_assignment_status",
            "volunteer_id",
            "status",
        ),
    )

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level in {
            VolunteerRiskLevel.HIGH,
            VolunteerRiskLevel.CRITICAL,
        }

    @property
    def can_start(self) -> bool:
        return (
            self.status == VolunteerAssignmentStatus.ACCEPTED
            and self.approved_by_user_id is not None
            and self.safety_briefing_completed
            and (
                not self.requires_certification
                or self.qualifications_verified
            )
            and (
                not self.is_high_risk
                or self.ppe_verified
            )
        )

    def request_approval(self) -> None:
        if self.status != (
            VolunteerAssignmentStatus.PROPOSED
        ):
            raise ValueError(
                "Assignment is not proposed"
            )

        self.status = (
            VolunteerAssignmentStatus.PENDING_APPROVAL
        )

    def approve(self, approver_id: str) -> None:
        actor = valid_actor_id(approver_id)

        if self.status != (
            VolunteerAssignmentStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Assignment is not pending approval"
            )

        if self.proposed_by_user_id == actor:
            raise ValueError(
                "Independent approval required"
            )

        if (
            self.requires_certification
            and not self.qualifications_verified
        ):
            raise ValueError(
                "Required qualifications not verified"
            )

        self.approved_by_user_id = actor
        self.approved_at = utc_now()
        self.status = VolunteerAssignmentStatus.APPROVED

    def accept(self) -> None:
        if self.status != (
            VolunteerAssignmentStatus.APPROVED
        ):
            raise ValueError(
                "Assignment must be approved"
            )

        self.accepted_at = utc_now()
        self.status = VolunteerAssignmentStatus.ACCEPTED

    def start(self) -> None:
        if not self.can_start:
            raise ValueError(
                "Assignment safety requirements not met"
            )

        self.started_at = utc_now()
        self.status = VolunteerAssignmentStatus.IN_PROGRESS

    def complete(
        self,
        *,
        notes: str,
        verifier_id: str,
    ) -> None:
        actor = valid_actor_id(verifier_id)

        if self.status != (
            VolunteerAssignmentStatus.IN_PROGRESS
        ):
            raise ValueError(
                "Assignment is not in progress"
            )

        self.completion_notes = required_text(
            notes,
            "notes",
            10000,
        )
        self.verified_completion_by_user_id = actor
        self.completed_at = utc_now()
        self.status = VolunteerAssignmentStatus.COMPLETED


# ============================================================
# VOLUNTEER SHIFT
# ============================================================

class VolunteerShift(Base):
    __tablename__ = "volunteer_shifts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    assignment_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "volunteer_assignments.id",
            ondelete="SET NULL",
        ),
    )

    status: Mapped[
        VolunteerShiftStatus
    ] = mapped_column(
        enum_type(VolunteerShiftStatus),
        default=VolunteerShiftStatus.SCHEDULED,
        nullable=False,
    )

    scheduled_start_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    scheduled_end_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    checked_in_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    checked_out_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    checked_in_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    checked_out_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    total_break_minutes: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="shifts",
    )

    __table_args__ = (
        CheckConstraint(
            "scheduled_end_at > scheduled_start_at",
            name="ck_volunteer_shift_schedule",
        ),
        CheckConstraint(
            "total_break_minutes >= 0",
            name="ck_volunteer_shift_break",
        ),
    )

    def check_in(self, actor_id: str) -> None:
        actor = valid_actor_id(actor_id)

        if self.status not in {
            VolunteerShiftStatus.SCHEDULED,
            VolunteerShiftStatus.CONFIRMED,
        }:
            raise ValueError(
                "Shift cannot be checked in"
            )

        self.checked_in_by_user_id = actor
        self.checked_in_at = utc_now()
        self.status = VolunteerShiftStatus.CHECKED_IN

    def check_out(self, actor_id: str) -> None:
        actor = valid_actor_id(actor_id)

        if self.status not in {
            VolunteerShiftStatus.CHECKED_IN,
            VolunteerShiftStatus.ON_BREAK,
        }:
            raise ValueError(
                "Shift is not active"
            )

        self.checked_out_by_user_id = actor
        self.checked_out_at = utc_now()
        self.status = VolunteerShiftStatus.CHECKED_OUT

    @property
    def service_minutes(self) -> int:
        if not self.checked_in_at or not self.checked_out_at:
            return 0

        elapsed = (
            normalize_datetime(self.checked_out_at)
            - normalize_datetime(self.checked_in_at)
        ).total_seconds()

        return max(
            0,
            int(elapsed // 60) - self.total_break_minutes,
        )


# ============================================================
# VOLUNTEER LOCATION
# ============================================================

class VolunteerLocation(Base):
    __tablename__ = "volunteer_locations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    latitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    longitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    accuracy_m: Mapped[float | None] = mapped_column(
        Float,
    )

    is_on_duty: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(100),
        default="mobile_app",
        nullable=False,
    )

    recorded_at: Mapped[datetime] = mapped_column(
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

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="locations",
    )

    __table_args__ = (
        CheckConstraint(
            "latitude BETWEEN -90 AND 90",
            name="ck_volunteer_location_lat",
        ),
        CheckConstraint(
            "longitude BETWEEN -180 AND 180",
            name="ck_volunteer_location_lon",
        ),
        CheckConstraint(
            "accuracy_m IS NULL OR accuracy_m >= 0",
            name="ck_volunteer_location_accuracy",
        ),
        Index(
            "ix_volunteer_location_time",
            "volunteer_id",
            "recorded_at",
        ),
    )

    @validates("latitude", "longitude")
    def validate_coordinates(
        self,
        key: str,
        value: float,
    ) -> float:
        result = valid_coordinate(value, key)

        if result is None:
            raise ValueError(f"{key} is required")

        return result


# ============================================================
# VOLUNTEER EQUIPMENT
# ============================================================

class VolunteerEquipment(Base):
    __tablename__ = "volunteer_equipment"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    equipment_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    equipment_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    inventory_item_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    quantity: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    status: Mapped[
        VolunteerEquipmentStatus
    ] = mapped_column(
        enum_type(VolunteerEquipmentStatus),
        default=VolunteerEquipmentStatus.REQUESTED,
        nullable=False,
    )

    issued_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    issued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    returned_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    received_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="equipment",
    )

    __table_args__ = (
        CheckConstraint(
            "quantity > 0",
            name="ck_volunteer_equipment_quantity",
        ),
    )


# ============================================================
# VOLUNTEER WELFARE CHECK
# ============================================================

class VolunteerWelfareCheck(Base):
    __tablename__ = "volunteer_welfare_checks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    welfare_status: Mapped[
        VolunteerWelfareStatus
    ] = mapped_column(
        enum_type(VolunteerWelfareStatus),
        nullable=False,
    )

    requires_medical_assistance: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_rest: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_emergency_response: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    private_notes_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    checked_by_user_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    follow_up_due_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="welfare_checks",
    )

    __table_args__ = (
        Index(
            "ix_volunteer_welfare_time",
            "volunteer_id",
            "checked_at",
        ),
    )


# ============================================================
# VOLUNTEER AUDIT EVENT
# ============================================================

class VolunteerEvent(Base):
    __tablename__ = "volunteer_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    volunteer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "volunteers.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        VolunteerEventType
    ] = mapped_column(
        enum_type(VolunteerEventType),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    previous_status: Mapped[
        VolunteerStatus | None
    ] = mapped_column(
        enum_type(VolunteerStatus),
    )

    new_status: Mapped[
        VolunteerStatus | None
    ] = mapped_column(
        enum_type(VolunteerStatus),
    )

    actor_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    actor_agent_name: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
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

    volunteer: Mapped["Volunteer"] = relationship(
        "Volunteer",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_volunteer_event_time",
            "volunteer_id",
            "occurred_at",
        ),
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

VOLUNTEER_MODELS = (
    Volunteer,
    VolunteerSkill,
    VolunteerTraining,
    VolunteerAvailability,
    VolunteerAssignment,
    VolunteerShift,
    VolunteerLocation,
    VolunteerEquipment,
    VolunteerWelfareCheck,
    VolunteerEvent,
)


__all__ = [
    "Volunteer",
    "VolunteerSkill",
    "VolunteerTraining",
    "VolunteerAvailability",
    "VolunteerAssignment",
    "VolunteerShift",
    "VolunteerLocation",
    "VolunteerEquipment",
    "VolunteerWelfareCheck",
    "VolunteerEvent",
    "VolunteerStatus",
    "VolunteerType",
    "VolunteerVerificationStatus",
    "VolunteerAvailabilityStatus",
    "VolunteerSkillType",
    "VolunteerSkillLevel",
    "VolunteerTrainingStatus",
    "VolunteerAssignmentType",
    "VolunteerAssignmentStatus",
    "VolunteerRiskLevel",
    "VolunteerShiftStatus",
    "VolunteerWelfareStatus",
    "VolunteerEquipmentStatus",
    "VolunteerEventType",
    "VOLUNTEER_MODELS",
]
