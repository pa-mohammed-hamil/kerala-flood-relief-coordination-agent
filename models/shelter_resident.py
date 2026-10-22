from __future__ import annotations

"""
backend/models/shelter_resident.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x models for shelter resident management.

Features:
- Flood shelter resident registration
- Family and household grouping
- Shelter admission and check-in
- Bed and room allocation
- Resident transfers and discharge
- Missing-family-member reporting references
- Vulnerable resident assistance
- Medical assistance requests
- Food, water and essential supply requirements
- Emergency contact references
- Consent and privacy controls
- Welfare checks and safeguarding
- Family reunification tracking
- Resident movement history
- Human-approved shelter transfers
- Audit-friendly operational records

Safety:
- Resident identity and medical details are sensitive.
- Emergency contacts must be access-controlled.
- AI recommendations cannot approve transfers.
- Medical decisions require qualified professionals.
- Children and vulnerable residents require safeguarding.
- Public API responses must exclude private identifiers.
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


def validate_nonnegative(
    value: int,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a non-negative integer"
        )

    return value


def validate_score(
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


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class ShelterResidentStatus(str, enum.Enum):
    REGISTERED = "registered"
    PENDING_ADMISSION = "pending_admission"
    ADMITTED = "admitted"
    TEMPORARILY_AWAY = "temporarily_away"
    TRANSFER_PENDING = "transfer_pending"
    TRANSFERRED = "transferred"
    DISCHARGED = "discharged"
    LEFT_WITHOUT_NOTICE = "left_without_notice"
    DECEASED = "deceased"
    RECORD_CANCELLED = "record_cancelled"


class ShelterResidentCategory(str, enum.Enum):
    ADULT = "adult"
    CHILD = "child"
    INFANT = "infant"
    ELDERLY = "elderly"
    PREGNANT_PERSON = "pregnant_person"
    PERSON_WITH_DISABILITY = "person_with_disability"
    OTHER = "other"
    UNSPECIFIED = "unspecified"


class ShelterAdmissionSource(str, enum.Enum):
    SELF_REGISTRATION = "self_registration"
    WALK_IN = "walk_in"
    RESCUE_TEAM = "rescue_team"
    EVACUATION_MISSION = "evacuation_mission"
    HOSPITAL_REFERRAL = "hospital_referral"
    GOVERNMENT_REFERRAL = "government_referral"
    OTHER_SHELTER = "other_shelter"
    VOLUNTEER_REFERRAL = "volunteer_referral"
    OTHER = "other"


class ShelterIdentityVerification(str, enum.Enum):
    NOT_REQUIRED = "not_required"
    NOT_VERIFIED = "not_verified"
    PENDING = "pending"
    VERIFIED = "verified"
    UNABLE_TO_VERIFY = "unable_to_verify"


class ShelterResidentPriority(str, enum.Enum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    HIGH = "high"
    URGENT = "urgent"


class ShelterAssistanceType(str, enum.Enum):
    FOOD = "food"
    DRINKING_WATER = "drinking_water"
    CLOTHING = "clothing"
    HYGIENE = "hygiene"
    SANITATION = "sanitation"
    MEDICAL = "medical"
    MEDICATION = "medication"
    MATERNAL_CARE = "maternal_care"
    INFANT_CARE = "infant_care"
    MOBILITY_SUPPORT = "mobility_support"
    ACCESSIBILITY = "accessibility"
    PSYCHOSOCIAL_SUPPORT = "psychosocial_support"
    FAMILY_REUNIFICATION = "family_reunification"
    TRANSLATION = "translation"
    TRANSPORT = "transport"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class ShelterAssistanceStatus(str, enum.Enum):
    REQUESTED = "requested"
    TRIAGED = "triaged"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    FULFILLED = "fulfilled"
    PARTIALLY_FULFILLED = "partially_fulfilled"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"


class ShelterTransferStatus(str, enum.Enum):
    REQUESTED = "requested"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DESTINATION_CONFIRMED = "destination_confirmed"
    IN_TRANSIT = "in_transit"
    ARRIVED = "arrived"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ShelterWelfareStatus(str, enum.Enum):
    NOT_ASSESSED = "not_assessed"
    STABLE = "stable"
    NEEDS_ASSISTANCE = "needs_assistance"
    URGENT_ATTENTION = "urgent_attention"
    REFERRED_TO_PROFESSIONAL = (
        "referred_to_professional"
    )
    UNABLE_TO_CONTACT = "unable_to_contact"


class ShelterWelfareCheckType(str, enum.Enum):
    ROUTINE = "routine"
    ADMISSION = "admission"
    FOLLOW_UP = "follow_up"
    SAFEGUARDING = "safeguarding"
    MEDICAL_REFERRAL = "medical_referral"
    DISCHARGE = "discharge"
    OTHER = "other"


class ShelterMovementType(str, enum.Enum):
    ADMISSION = "admission"
    CHECK_IN = "check_in"
    TEMPORARY_EXIT = "temporary_exit"
    RETURN = "return"
    ROOM_CHANGE = "room_change"
    TRANSFER_OUT = "transfer_out"
    TRANSFER_IN = "transfer_in"
    DISCHARGE = "discharge"
    OTHER = "other"


class ShelterReunificationStatus(str, enum.Enum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    SEARCHING = "searching"
    POSSIBLE_MATCH = "possible_match"
    IDENTITY_VERIFICATION = "identity_verification"
    VERIFIED_MATCH = "verified_match"
    REUNIFIED = "reunified"
    CLOSED_UNRESOLVED = "closed_unresolved"
    CANCELLED = "cancelled"


class ShelterResidentEventType(str, enum.Enum):
    REGISTERED = "registered"
    IDENTITY_VERIFIED = "identity_verified"
    ADMITTED = "admitted"
    BED_ASSIGNED = "bed_assigned"
    BED_RELEASED = "bed_released"
    PROFILE_UPDATED = "profile_updated"
    ASSISTANCE_REQUESTED = "assistance_requested"
    ASSISTANCE_FULFILLED = "assistance_fulfilled"
    WELFARE_CHECK = "welfare_check"
    MEDICAL_REFERRAL = "medical_referral"
    SAFEGUARDING_ESCALATION = "safeguarding_escalation"
    TRANSFER_REQUESTED = "transfer_requested"
    TRANSFER_APPROVED = "transfer_approved"
    TRANSFER_COMPLETED = "transfer_completed"
    DISCHARGED = "discharged"
    TEMPORARY_EXIT = "temporary_exit"
    RETURNED = "returned"
    REUNIFICATION_REQUESTED = "reunification_requested"
    REUNIFIED = "reunified"
    RECORD_CORRECTED = "record_corrected"
    NOTE_ADDED = "note_added"


# ============================================================
# SHELTER HOUSEHOLD
# ============================================================

class ShelterHousehold(Base):
    __tablename__ = "shelter_households"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    household_reference: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
        index=True,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )

    household_display_name: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    primary_contact_resident_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    expected_members: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    registered_members: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    requires_family_co_location: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    contains_unaccompanied_minor: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    safeguarding_review_required: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    created_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
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

    residents: Mapped[
        list["ShelterResident"]
    ] = relationship(
        "ShelterResident",
        back_populates="household",
    )

    __table_args__ = (
        CheckConstraint(
            "expected_members >= 0",
            name="ck_shelter_household_expected",
        ),
        CheckConstraint(
            "registered_members >= 0",
            name="ck_shelter_household_registered",
        ),
    )


# ============================================================
# SHELTER RESIDENT
# ============================================================

class ShelterResident(Base):
    __tablename__ = "shelter_residents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    resident_reference: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
        index=True,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
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

    household_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "shelter_households.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    encrypted_identity_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    encrypted_contact_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    encrypted_emergency_contact_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    age_years: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    category: Mapped[
        ShelterResidentCategory
    ] = mapped_column(
        enum_type(ShelterResidentCategory),
        default=ShelterResidentCategory.UNSPECIFIED,
        nullable=False,
    )

    status: Mapped[
        ShelterResidentStatus
    ] = mapped_column(
        enum_type(ShelterResidentStatus),
        default=ShelterResidentStatus.REGISTERED,
        nullable=False,
        index=True,
    )

    priority: Mapped[
        ShelterResidentPriority
    ] = mapped_column(
        enum_type(ShelterResidentPriority),
        default=ShelterResidentPriority.ROUTINE,
        nullable=False,
    )

    admission_source: Mapped[
        ShelterAdmissionSource
    ] = mapped_column(
        enum_type(ShelterAdmissionSource),
        nullable=False,
    )

    identity_verification: Mapped[
        ShelterIdentityVerification
    ] = mapped_column(
        enum_type(ShelterIdentityVerification),
        default=ShelterIdentityVerification.NOT_VERIFIED,
        nullable=False,
    )

    identity_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    identity_verified_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    source_incident_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    source_rescue_mission_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    source_evacuation_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    original_district_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    original_location_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    room_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    bed_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    is_unaccompanied_minor: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_accessibility_support: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_medical_assistance: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_mobility_assistance: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_special_diet: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_safeguarding_review: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    encrypted_medical_information_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    welfare_status: Mapped[
        ShelterWelfareStatus
    ] = mapped_column(
        enum_type(ShelterWelfareStatus),
        default=ShelterWelfareStatus.NOT_ASSESSED,
        nullable=False,
    )

    last_welfare_check_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    consent_for_contact: Mapped[
        bool | None
    ] = mapped_column(
        Boolean,
        nullable=True,
    )

    consent_for_family_reunification: Mapped[
        bool | None
    ] = mapped_column(
        Boolean,
        nullable=True,
    )

    privacy_notice_acknowledged: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    registered_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    admitted_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    admitted_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    discharged_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    discharged_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    discharge_reason: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    audit_reference: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
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

    household: Mapped[
        "ShelterHousehold | None"
    ] = relationship(
        "ShelterHousehold",
        back_populates="residents",
    )

    assistance_requests: Mapped[
        list["ShelterResidentAssistance"]
    ] = relationship(
        "ShelterResidentAssistance",
        back_populates="resident",
        cascade="save-update, merge",
    )

    welfare_checks: Mapped[
        list["ShelterResidentWelfareCheck"]
    ] = relationship(
        "ShelterResidentWelfareCheck",
        back_populates="resident",
        cascade="save-update, merge",
    )

    transfers: Mapped[
        list["ShelterResidentTransfer"]
    ] = relationship(
        "ShelterResidentTransfer",
        back_populates="resident",
        cascade="save-update, merge",
    )

    movements: Mapped[
        list["ShelterResidentMovement"]
    ] = relationship(
        "ShelterResidentMovement",
        back_populates="resident",
        cascade="save-update, merge",
    )

    reunification_cases: Mapped[
        list["ShelterResidentReunification"]
    ] = relationship(
        "ShelterResidentReunification",
        back_populates="resident",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["ShelterResidentEvent"]
    ] = relationship(
        "ShelterResidentEvent",
        back_populates="resident",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_shelter_resident_district",
        ),
        CheckConstraint(
            "age_years IS NULL OR "
            "(age_years >= 0 AND age_years <= 125)",
            name="ck_shelter_resident_age",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_shelter_resident_version",
        ),
        Index(
            "ix_shelter_resident_shelter_status",
            "shelter_id",
            "status",
        ),
        Index(
            "ix_shelter_resident_district_priority",
            "district_id",
            "priority",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "resident_reference",
        "display_name",
    )
    def validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        return require_text(
            value,
            key,
            120 if key == "resident_reference" else 255,
        )

    @validates("age_years")
    def validate_age(
        self,
        key: str,
        value: int | None,
    ) -> int | None:
        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 125
        ):
            raise ValueError(
                "age_years must be between 0 and 125"
            )

        return value

    @property
    def is_current_resident(self) -> bool:
        return self.status in {
            ShelterResidentStatus.ADMITTED,
            ShelterResidentStatus.TEMPORARILY_AWAY,
            ShelterResidentStatus.TRANSFER_PENDING,
        }

    @property
    def needs_priority_attention(self) -> bool:
        return (
            self.priority in {
                ShelterResidentPriority.HIGH,
                ShelterResidentPriority.URGENT,
            }
            or self.is_unaccompanied_minor
            or self.requires_safeguarding_review
            or self.welfare_status
            == ShelterWelfareStatus.URGENT_ATTENTION
        )

    def request_admission(self) -> None:
        if self.status != (
            ShelterResidentStatus.REGISTERED
        ):
            raise ValueError(
                "Resident is not in registered state"
            )

        self.status = (
            ShelterResidentStatus.PENDING_ADMISSION
        )

    def admit(
        self,
        *,
        authorized_user_id: int,
        room_reference: str | None = None,
        bed_reference: str | None = None,
    ) -> None:
        """
        Shelter capacity, bed availability and
        safeguarding checks belong in the service layer.
        Identity documents must not be mandatory
        prerequisites for emergency assistance.
        """
        if self.status not in {
            ShelterResidentStatus.REGISTERED,
            ShelterResidentStatus.PENDING_ADMISSION,
        }:
            raise ValueError(
                "Resident cannot be admitted"
            )

        if authorized_user_id <= 0:
            raise ValueError(
                "Invalid authorized user"
            )

        if room_reference is not None:
            self.room_reference = require_text(
                room_reference,
                "room_reference",
                150,
            )

        if bed_reference is not None:
            self.bed_reference = require_text(
                bed_reference,
                "bed_reference",
                150,
            )

        self.admitted_by_user_id = authorized_user_id
        self.admitted_at = utc_now()
        self.status = ShelterResidentStatus.ADMITTED

    def mark_temporarily_away(self) -> None:
        if self.status != ShelterResidentStatus.ADMITTED:
            raise ValueError(
                "Resident is not admitted"
            )

        self.status = (
            ShelterResidentStatus.TEMPORARILY_AWAY
        )

    def mark_returned(self) -> None:
        if self.status != (
            ShelterResidentStatus.TEMPORARILY_AWAY
        ):
            raise ValueError(
                "Resident is not marked temporarily away"
            )

        self.status = ShelterResidentStatus.ADMITTED

    def discharge(
        self,
        *,
        authorized_user_id: int,
        reason: str,
    ) -> None:
        if self.status not in {
            ShelterResidentStatus.ADMITTED,
            ShelterResidentStatus.TEMPORARILY_AWAY,
        }:
            raise ValueError(
                "Resident cannot be discharged"
            )

        if authorized_user_id <= 0:
            raise ValueError(
                "Invalid authorized user"
            )

        if self.is_unaccompanied_minor:
            raise ValueError(
                "Unaccompanied minor discharge requires "
                "a separate safeguarding workflow"
            )

        self.discharge_reason = require_text(
            reason,
            "reason",
            5000,
        )
        self.discharged_by_user_id = authorized_user_id
        self.discharged_at = utc_now()
        self.status = ShelterResidentStatus.DISCHARGED
        self.room_reference = None
        self.bed_reference = None

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Minimal operational representation.
        Additional role-based redaction is required
        before exposing resident-level data.
        """
        return {
            "id": self.id,
            "resident_reference": self.resident_reference,
            "shelter_id": self.shelter_id,
            "district_id": self.district_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "is_current_resident": self.is_current_resident,
            "needs_priority_attention": (
                self.needs_priority_attention
            ),
            "admitted_at": (
                self.admitted_at.isoformat()
                if self.admitted_at
                else None
            ),
        }


# ============================================================
# RESIDENT ASSISTANCE
# ============================================================

class ShelterResidentAssistance(Base):
    __tablename__ = "shelter_resident_assistance"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    resident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelter_residents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    assistance_type: Mapped[
        ShelterAssistanceType
    ] = mapped_column(
        enum_type(ShelterAssistanceType),
        nullable=False,
    )

    priority: Mapped[
        ShelterResidentPriority
    ] = mapped_column(
        enum_type(ShelterResidentPriority),
        default=ShelterResidentPriority.ROUTINE,
        nullable=False,
    )

    status: Mapped[
        ShelterAssistanceStatus
    ] = mapped_column(
        enum_type(ShelterAssistanceStatus),
        default=ShelterAssistanceStatus.REQUESTED,
        nullable=False,
        index=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    requested_quantity: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    fulfilled_quantity: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
            nullable=False,
        )
    )

    inventory_request_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    medical_referral_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    requested_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    assigned_to_user_id: Mapped[
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

    fulfilled_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    fulfilled_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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

    resident: Mapped["ShelterResident"] = relationship(
        "ShelterResident",
        back_populates="assistance_requests",
    )

    __table_args__ = (
        CheckConstraint(
            "requested_quantity IS NULL OR "
            "requested_quantity > 0",
            name="ck_shelter_assistance_requested",
        ),
        CheckConstraint(
            "fulfilled_quantity >= 0",
            name="ck_shelter_assistance_fulfilled",
        ),
        CheckConstraint(
            "requested_quantity IS NULL OR "
            "fulfilled_quantity <= requested_quantity",
            name="ck_shelter_assistance_quantity_limit",
        ),
        Index(
            "ix_shelter_assistance_resident_status",
            "resident_id",
            "status",
        ),
    )


# ============================================================
# WELFARE CHECKS
# ============================================================

class ShelterResidentWelfareCheck(Base):
    __tablename__ = "shelter_resident_welfare_checks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    resident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelter_residents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    check_type: Mapped[
        ShelterWelfareCheckType
    ] = mapped_column(
        enum_type(ShelterWelfareCheckType),
        nullable=False,
    )

    welfare_status: Mapped[
        ShelterWelfareStatus
    ] = mapped_column(
        enum_type(ShelterWelfareStatus),
        nullable=False,
    )

    check_summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    private_notes_encrypted_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    requires_follow_up: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    requires_medical_referral: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_safeguarding_escalation: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    follow_up_due_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    checked_by_user_id: Mapped[int] = (
        mapped_column(
            Integer,
            nullable=False,
        )
    )

    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
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

    resident: Mapped["ShelterResident"] = relationship(
        "ShelterResident",
        back_populates="welfare_checks",
    )

    __table_args__ = (
        CheckConstraint(
            "checked_by_user_id > 0",
            name="ck_shelter_welfare_checker",
        ),
        Index(
            "ix_shelter_welfare_resident_time",
            "resident_id",
            "checked_at",
        ),
    )


# ============================================================
# RESIDENT TRANSFER
# ============================================================

class ShelterResidentTransfer(Base):
    __tablename__ = "shelter_resident_transfers"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    transfer_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    resident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelter_residents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    source_shelter_id: Mapped[str] = (
        mapped_column(
            String(36),
            nullable=False,
        )
    )

    destination_shelter_id: Mapped[
        str
    ] = mapped_column(
        String(36),
        nullable=False,
    )

    status: Mapped[
        ShelterTransferStatus
    ] = mapped_column(
        enum_type(ShelterTransferStatus),
        default=ShelterTransferStatus.REQUESTED,
        nullable=False,
        index=True,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    transport_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    escort_required: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    medical_transport_required: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
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

    destination_confirmed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    destination_confirmed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    departed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    arrived_at: Mapped[datetime | None] = (
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

    receiving_officer_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(100),
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

    resident: Mapped["ShelterResident"] = relationship(
        "ShelterResident",
        back_populates="transfers",
    )

    __table_args__ = (
        CheckConstraint(
            "source_shelter_id <> destination_shelter_id",
            name="ck_shelter_transfer_different",
        ),
        Index(
            "ix_shelter_transfer_resident_status",
            "resident_id",
            "status",
        ),
    )

    def request_approval(self) -> None:
        if self.status != ShelterTransferStatus.REQUESTED:
            raise ValueError(
                "Transfer is not in requested state"
            )

        self.status = (
            ShelterTransferStatus.PENDING_APPROVAL
        )

    def approve(self, approver_id: int) -> None:
        if self.status != (
            ShelterTransferStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Transfer is not pending approval"
            )

        if approver_id <= 0:
            raise ValueError(
                "Invalid approver"
            )

        if (
            self.requested_by_user_id is not None
            and approver_id == self.requested_by_user_id
        ):
            raise ValueError(
                "Independent approval required"
            )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = ShelterTransferStatus.APPROVED

    def confirm_destination(
        self,
        receiving_officer_id: int,
    ) -> None:
        if self.status != ShelterTransferStatus.APPROVED:
            raise ValueError(
                "Transfer must be approved"
            )

        if receiving_officer_id <= 0:
            raise ValueError(
                "Invalid receiving officer"
            )

        self.destination_confirmed_by_user_id = (
            receiving_officer_id
        )
        self.destination_confirmed_at = utc_now()
        self.status = (
            ShelterTransferStatus.DESTINATION_CONFIRMED
        )

    def mark_in_transit(self) -> None:
        if self.status != (
            ShelterTransferStatus.DESTINATION_CONFIRMED
        ):
            raise ValueError(
                "Destination must be confirmed"
            )

        self.departed_at = utc_now()
        self.status = ShelterTransferStatus.IN_TRANSIT

    def mark_arrived(
        self,
        receiving_officer_id: int,
    ) -> None:
        if self.status != ShelterTransferStatus.IN_TRANSIT:
            raise ValueError(
                "Transfer is not in transit"
            )

        if receiving_officer_id <= 0:
            raise ValueError(
                "Invalid receiving officer"
            )

        self.receiving_officer_user_id = (
            receiving_officer_id
        )
        self.arrived_at = utc_now()
        self.status = ShelterTransferStatus.ARRIVED

    def complete(self) -> None:
        """
        Service layer must atomically update:
        - Resident's shelter_id
        - Resident's room/bed allocation
        - Source and destination occupancy
        - Household membership if applicable
        - Audit events and movement records
        """
        if self.status != ShelterTransferStatus.ARRIVED:
            raise ValueError(
                "Transfer arrival is not confirmed"
            )

        if self.receiving_officer_user_id is None:
            raise ValueError(
                "Receiving officer is required"
            )

        self.completed_at = utc_now()
        self.status = ShelterTransferStatus.COMPLETED


# ============================================================
# RESIDENT MOVEMENT
# ============================================================

class ShelterResidentMovement(Base):
    __tablename__ = "shelter_resident_movements"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    resident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelter_residents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    shelter_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    movement_type: Mapped[
        ShelterMovementType
    ] = mapped_column(
        enum_type(ShelterMovementType),
        nullable=False,
    )

    from_location_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    to_location_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    transfer_id: Mapped[str | None] = (
        mapped_column(
            String(36),
            ForeignKey(
                "shelter_resident_transfers.id",
                ondelete="SET NULL",
            ),
            nullable=True,
        )
    )

    recorded_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
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

    resident: Mapped["ShelterResident"] = relationship(
        "ShelterResident",
        back_populates="movements",
    )

    __table_args__ = (
        Index(
            "ix_shelter_movement_resident_time",
            "resident_id",
            "occurred_at",
        ),
    )


# ============================================================
# FAMILY REUNIFICATION
# ============================================================

class ShelterResidentReunification(Base):
    __tablename__ = "shelter_resident_reunifications"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    case_number: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        nullable=False,
    )

    resident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelter_residents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    missing_person_report_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    encrypted_family_contact_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    relationship_description: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    status: Mapped[
        ShelterReunificationStatus
    ] = mapped_column(
        enum_type(ShelterReunificationStatus),
        default=ShelterReunificationStatus.REQUESTED,
        nullable=False,
        index=True,
    )

    is_minor_safeguarding_case: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    identity_match_confidence: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    identity_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    safeguarding_approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    reunited_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    completed_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    private_case_notes_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
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

    resident: Mapped["ShelterResident"] = relationship(
        "ShelterResident",
        back_populates="reunification_cases",
    )

    __table_args__ = (
        CheckConstraint(
            "identity_match_confidence IS NULL OR "
            "(identity_match_confidence >= 0 "
            "AND identity_match_confidence <= 1)",
            name="ck_shelter_reunification_confidence",
        ),
    )

    @validates("identity_match_confidence")
    def validate_confidence(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_score(value, key)

    def mark_reunified(
        self,
        *,
        authorized_user_id: int,
    ) -> None:
        if self.status != (
            ShelterReunificationStatus.VERIFIED_MATCH
        ):
            raise ValueError(
                "Family match must be verified"
            )

        if self.identity_verified_by_user_id is None:
            raise ValueError(
                "Human identity verification required"
            )

        if (
            self.is_minor_safeguarding_case
            and self.safeguarding_approved_by_user_id
            is None
        ):
            raise ValueError(
                "Minor safeguarding approval required"
            )

        if authorized_user_id <= 0:
            raise ValueError(
                "Invalid authorized user"
            )

        self.completed_by_user_id = authorized_user_id
        self.reunited_at = utc_now()
        self.status = (
            ShelterReunificationStatus.REUNIFIED
        )


# ============================================================
# RESIDENT AUDIT EVENTS
# ============================================================

class ShelterResidentEvent(Base):
    __tablename__ = "shelter_resident_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    resident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "shelter_residents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        ShelterResidentEventType
    ] = mapped_column(
        enum_type(ShelterResidentEventType),
        nullable=False,
    )

    previous_status: Mapped[
        ShelterResidentStatus | None
    ] = mapped_column(
        enum_type(ShelterResidentStatus),
        nullable=True,
    )

    new_status: Mapped[
        ShelterResidentStatus | None
    ] = mapped_column(
        enum_type(ShelterResidentStatus),
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

    resident: Mapped["ShelterResident"] = relationship(
        "ShelterResident",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_shelter_resident_event_time",
            "resident_id",
            "occurred_at",
        ),
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

SHELTER_RESIDENT_MODELS = (
    ShelterHousehold,
    ShelterResident,
    ShelterResidentAssistance,
    ShelterResidentWelfareCheck,
    ShelterResidentTransfer,
    ShelterResidentMovement,
    ShelterResidentReunification,
    ShelterResidentEvent,
)


__all__ = [
    "ShelterHousehold",
    "ShelterResident",
    "ShelterResidentAssistance",
    "ShelterResidentWelfareCheck",
    "ShelterResidentTransfer",
    "ShelterResidentMovement",
    "ShelterResidentReunification",
    "ShelterResidentEvent",
    "ShelterResidentStatus",
    "ShelterResidentCategory",
    "ShelterAdmissionSource",
    "ShelterIdentityVerification",
    "ShelterResidentPriority",
    "ShelterAssistanceType",
    "ShelterAssistanceStatus",
    "ShelterTransferStatus",
    "ShelterWelfareStatus",
    "ShelterWelfareCheckType",
    "ShelterMovementType",
    "ShelterReunificationStatus",
    "ShelterResidentEventType",
    "SHELTER_RESIDENT_MODELS",
]
