from __future__ import annotations

"""
backend/models/rescue_team.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x models for rescue team management.

Includes:
- Rescue team registration
- NDRF, SDRF, Fire & Rescue, police and volunteers
- Team members and professional qualifications
- Team availability and operational readiness
- Rescue vehicles and boats
- Equipment and safety inspections
- Team location and telemetry
- District assignments
- Deployment and shift tracking
- Certification expiry tracking
- Human verification and authorization
- Audit history

Safety:
- AI cannot independently activate rescue teams.
- Qualifications and readiness require verification.
- Dispatch authorization belongs to designated officials.
- Location and personnel information are access-controlled.
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

    value = value.strip()

    if len(value) > maximum:
        raise ValueError(
            f"{field_name} exceeds maximum length"
        )

    return value


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


def is_future(value: datetime | None) -> bool:
    if value is None:
        return False

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc) > utc_now()


# ============================================================
# ENUMERATIONS
# ============================================================

class RescueTeamType(str, enum.Enum):
    NDRF = "ndrf"
    SDRF = "sdrf"
    FIRE_AND_RESCUE = "fire_and_rescue"
    POLICE = "police"
    COAST_GUARD = "coast_guard"
    NAVY = "navy"
    ARMY = "army"
    CIVIL_DEFENCE = "civil_defence"
    MEDICAL_RESCUE = "medical_rescue"
    AMBULANCE = "ambulance"
    WATER_RESCUE = "water_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    MOUNTAIN_RESCUE = "mountain_rescue"
    VOLUNTEER = "volunteer"
    NGO = "ngo"
    LOCAL_GOVERNMENT = "local_government"
    PRIVATE_AUTHORIZED = "private_authorized"
    OTHER = "other"


class RescueTeamStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DECOMMISSIONED = "decommissioned"


class RescueTeamAvailability(str, enum.Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    ON_MISSION = "on_mission"
    RETURNING = "returning"
    RESTING = "resting"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class RescueTeamReadiness(str, enum.Enum):
    READY = "ready"
    LIMITED = "limited"
    NOT_READY = "not_ready"
    PENDING_INSPECTION = "pending_inspection"
    UNKNOWN = "unknown"


class RescueMemberRole(str, enum.Enum):
    COMMANDER = "commander"
    TEAM_LEADER = "team_leader"
    DEPUTY_LEADER = "deputy_leader"
    RESCUER = "rescuer"
    BOAT_OPERATOR = "boat_operator"
    DRIVER = "driver"
    PARAMEDIC = "paramedic"
    DOCTOR = "doctor"
    NURSE = "nurse"
    DIVER = "diver"
    SWIMMER = "swimmer"
    ROPE_RESCUE_SPECIALIST = "rope_rescue_specialist"
    TECHNICAL_RESCUE_SPECIALIST = (
        "technical_rescue_specialist"
    )
    COMMUNICATION_OFFICER = "communication_officer"
    LOGISTICS_OFFICER = "logistics_officer"
    SAFETY_OFFICER = "safety_officer"
    DRONE_OPERATOR = "drone_operator"
    VOLUNTEER = "volunteer"
    OTHER = "other"


class RescueMemberStatus(str, enum.Enum):
    ACTIVE = "active"
    OFF_DUTY = "off_duty"
    ON_LEAVE = "on_leave"
    INJURED = "injured"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"


class RescueCertificationType(str, enum.Enum):
    BASIC_FIRST_AID = "basic_first_aid"
    ADVANCED_FIRST_AID = "advanced_first_aid"
    CPR = "cpr"
    BLS = "bls"
    ALS = "als"
    SWIFT_WATER_RESCUE = "swift_water_rescue"
    FLOOD_RESCUE = "flood_rescue"
    BOAT_OPERATION = "boat_operation"
    DIVING = "diving"
    ROPE_RESCUE = "rope_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    DISASTER_MANAGEMENT = "disaster_management"
    VEHICLE_OPERATION = "vehicle_operation"
    DRONE_OPERATION = "drone_operation"
    RADIO_COMMUNICATION = "radio_communication"
    OTHER = "other"


class RescueVehicleType(str, enum.Enum):
    RESCUE_BOAT = "rescue_boat"
    INFLATABLE_BOAT = "inflatable_boat"
    MOTORBOAT = "motorboat"
    AMBULANCE = "ambulance"
    RESCUE_TRUCK = "rescue_truck"
    FOUR_WHEEL_DRIVE = "four_wheel_drive"
    VAN = "van"
    JEEP = "jeep"
    HELICOPTER = "helicopter"
    OTHER = "other"


class RescueVehicleStatus(str, enum.Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    DEPLOYED = "deployed"
    UNDER_MAINTENANCE = "under_maintenance"
    OUT_OF_SERVICE = "out_of_service"
    UNKNOWN = "unknown"


class RescueEquipmentType(str, enum.Enum):
    LIFE_JACKET = "life_jacket"
    RESCUE_ROPE = "rescue_rope"
    RESCUE_RING = "rescue_ring"
    HELMET = "helmet"
    PPE = "ppe"
    STRETCHER = "stretcher"
    FIRST_AID_KIT = "first_aid_kit"
    MEDICAL_KIT = "medical_kit"
    OXYGEN = "oxygen"
    RADIO = "radio"
    SATELLITE_PHONE = "satellite_phone"
    GPS_DEVICE = "gps_device"
    FLASHLIGHT = "flashlight"
    THERMAL_CAMERA = "thermal_camera"
    DRONE = "drone"
    WATER_PUMP = "water_pump"
    GENERATOR = "generator"
    OTHER = "other"


class RescueEquipmentCondition(str, enum.Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    DAMAGED = "damaged"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class RescueShiftStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RescueTeamEventType(str, enum.Enum):
    CREATED = "created"
    VERIFIED = "verified"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    SUSPENDED = "suspended"
    READINESS_UPDATED = "readiness_updated"
    AVAILABILITY_UPDATED = "availability_updated"
    MEMBER_ADDED = "member_added"
    MEMBER_REMOVED = "member_removed"
    CERTIFICATION_UPDATED = "certification_updated"
    VEHICLE_ASSIGNED = "vehicle_assigned"
    EQUIPMENT_UPDATED = "equipment_updated"
    LOCATION_UPDATED = "location_updated"
    SHIFT_STARTED = "shift_started"
    SHIFT_COMPLETED = "shift_completed"
    MISSION_ASSIGNED = "mission_assigned"
    MISSION_COMPLETED = "mission_completed"
    SAFETY_ALERT = "safety_alert"
    NOTE_ADDED = "note_added"


# ============================================================
# RESCUE TEAM
# ============================================================

class RescueTeam(Base):
    __tablename__ = "rescue_teams"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    team_code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    team_type: Mapped[RescueTeamType] = mapped_column(
        enum_type(RescueTeamType),
        nullable=False,
    )

    organization_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    organization_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
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

    base_location_name: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    base_address: Mapped[str | None] = (
        mapped_column(
            String(500),
            nullable=True,
        )
    )

    base_latitude: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    base_longitude: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    current_latitude: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    current_longitude: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    location_updated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    status: Mapped[RescueTeamStatus] = mapped_column(
        enum_type(RescueTeamStatus),
        default=RescueTeamStatus.DRAFT,
        nullable=False,
        index=True,
    )

    availability: Mapped[
        RescueTeamAvailability
    ] = mapped_column(
        enum_type(RescueTeamAvailability),
        default=RescueTeamAvailability.UNKNOWN,
        nullable=False,
        index=True,
    )

    readiness: Mapped[
        RescueTeamReadiness
    ] = mapped_column(
        enum_type(RescueTeamReadiness),
        default=RescueTeamReadiness.UNKNOWN,
        nullable=False,
    )

    total_members: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    available_members: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    maximum_rescue_capacity: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
            nullable=False,
        )
    )

    has_water_rescue_capability: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_medical_capability: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_technical_rescue_capability: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_night_operation_capability: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    has_communication_equipment: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    command_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    emergency_contact_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    communication_channel_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    last_readiness_check_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    readiness_valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    readiness_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    is_verified: Mapped[bool] = mapped_column(
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

    created_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    activated_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    activated_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    deactivated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    members: Mapped[
        list["RescueTeamMember"]
    ] = relationship(
        "RescueTeamMember",
        back_populates="team",
        cascade="save-update, merge",
    )

    vehicles: Mapped[
        list["RescueTeamVehicle"]
    ] = relationship(
        "RescueTeamVehicle",
        back_populates="team",
        cascade="save-update, merge",
    )

    equipment: Mapped[
        list["RescueTeamEquipment"]
    ] = relationship(
        "RescueTeamEquipment",
        back_populates="team",
        cascade="save-update, merge",
    )

    shifts: Mapped[
        list["RescueTeamShift"]
    ] = relationship(
        "RescueTeamShift",
        back_populates="team",
        cascade="save-update, merge",
    )

    location_history: Mapped[
        list["RescueTeamLocation"]
    ] = relationship(
        "RescueTeamLocation",
        back_populates="team",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["RescueTeamEvent"]
    ] = relationship(
        "RescueTeamEvent",
        back_populates="team",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_rescue_team_district",
        ),
        CheckConstraint(
            "total_members >= 0",
            name="ck_rescue_team_total_members",
        ),
        CheckConstraint(
            "available_members >= 0",
            name="ck_rescue_team_available_members",
        ),
        CheckConstraint(
            "available_members <= total_members",
            name="ck_rescue_team_member_capacity",
        ),
        CheckConstraint(
            "maximum_rescue_capacity >= 0",
            name="ck_rescue_team_rescue_capacity",
        ),
        CheckConstraint(
            "base_latitude IS NULL OR "
            "(base_latitude >= -90 AND base_latitude <= 90)",
            name="ck_rescue_team_base_lat",
        ),
        CheckConstraint(
            "base_longitude IS NULL OR "
            "(base_longitude >= -180 "
            "AND base_longitude <= 180)",
            name="ck_rescue_team_base_lon",
        ),
        CheckConstraint(
            "current_latitude IS NULL OR "
            "(current_latitude >= -90 "
            "AND current_latitude <= 90)",
            name="ck_rescue_team_current_lat",
        ),
        CheckConstraint(
            "current_longitude IS NULL OR "
            "(current_longitude >= -180 "
            "AND current_longitude <= 180)",
            name="ck_rescue_team_current_lon",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_rescue_team_version",
        ),
        Index(
            "ix_rescue_team_district_availability",
            "district_id",
            "availability",
        ),
        Index(
            "ix_rescue_team_type_status",
            "team_type",
            "status",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "team_code",
        "name",
        "organization_name",
    )
    def validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        limits = {
            "team_code": 100,
            "name": 255,
            "organization_name": 255,
        }

        return require_text(
            value,
            key,
            limits[key],
        )

    @validates(
        "base_latitude",
        "current_latitude",
    )
    def validate_latitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -90, 90
        )

    @validates(
        "base_longitude",
        "current_longitude",
    )
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_coordinate(
            value, key, -180, 180
        )

    @validates(
        "total_members",
        "available_members",
        "maximum_rescue_capacity",
    )
    def validate_counts(
        self,
        key: str,
        value: int,
    ) -> int:
        return validate_nonnegative(value, key)

    @property
    def is_operational(self) -> bool:
        return (
            self.status == RescueTeamStatus.ACTIVE
            and self.is_verified
            and self.availability
            == RescueTeamAvailability.AVAILABLE
            and self.readiness
            == RescueTeamReadiness.READY
            and self.available_members > 0
            and self.maximum_rescue_capacity > 0
            and self.readiness_verified_by_user_id
            is not None
            and is_future(self.readiness_valid_until)
        )

    def submit_for_verification(self) -> None:
        if self.status != RescueTeamStatus.DRAFT:
            raise ValueError(
                "Only draft teams can be submitted"
            )

        self.status = (
            RescueTeamStatus.PENDING_VERIFICATION
        )

    def verify(self, verifier_id: int) -> None:
        if self.status != (
            RescueTeamStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Team is not pending verification"
            )

        if verifier_id <= 0:
            raise ValueError("Invalid verifier")

        if (
            self.created_by_user_id is not None
            and verifier_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent verification required"
            )

        self.is_verified = True
        self.verified_by_user_id = verifier_id
        self.verified_at = utc_now()
        self.status = RescueTeamStatus.VERIFIED

    def activate(self, authorized_user_id: int) -> None:
        if self.status not in {
            RescueTeamStatus.VERIFIED,
            RescueTeamStatus.INACTIVE,
        }:
            raise ValueError(
                "Team cannot be activated"
            )

        if not self.is_verified:
            raise ValueError(
                "Team verification is required"
            )

        if authorized_user_id <= 0:
            raise ValueError(
                "Invalid authorized user"
            )

        self.status = RescueTeamStatus.ACTIVE
        self.activated_by_user_id = authorized_user_id
        self.activated_at = utc_now()

    def update_readiness(
        self,
        *,
        readiness: RescueTeamReadiness,
        verifier_id: int,
        valid_until: datetime,
    ) -> None:
        if verifier_id <= 0:
            raise ValueError(
                "Invalid readiness verifier"
            )

        if not is_future(valid_until):
            raise ValueError(
                "Readiness validity must be in the future"
            )

        self.readiness = readiness
        self.readiness_verified_by_user_id = verifier_id
        self.last_readiness_check_at = utc_now()
        self.readiness_valid_until = valid_until

    def update_location(
        self,
        *,
        latitude: float,
        longitude: float,
    ) -> None:
        self.current_latitude = validate_coordinate(
            latitude, "latitude", -90, 90
        )
        self.current_longitude = validate_coordinate(
            longitude, "longitude", -180, 180
        )
        self.location_updated_at = utc_now()

    def deactivate(self) -> None:
        if self.availability in {
            RescueTeamAvailability.DISPATCHED,
            RescueTeamAvailability.ON_MISSION,
        }:
            raise ValueError(
                "Active deployments require command review"
            )

        self.status = RescueTeamStatus.INACTIVE
        self.availability = (
            RescueTeamAvailability.UNAVAILABLE
        )
        self.deactivated_at = utc_now()

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team_code": self.team_code,
            "name": self.name,
            "team_type": self.team_type.value,
            "organization_name": self.organization_name,
            "district_id": self.district_id,
            "status": self.status.value,
            "availability": self.availability.value,
            "readiness": self.readiness.value,
            "total_members": self.total_members,
            "available_members": self.available_members,
            "maximum_rescue_capacity": (
                self.maximum_rescue_capacity
            ),
            "is_verified": self.is_verified,
            "is_operational": self.is_operational,
        }


# ============================================================
# TEAM MEMBERS
# ============================================================

class RescueTeamMember(Base):
    __tablename__ = "rescue_team_members"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    team_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_teams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    personnel_reference: Mapped[str] = (
        mapped_column(
            String(150),
            nullable=False,
        )
    )

    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[RescueMemberRole] = mapped_column(
        enum_type(RescueMemberRole),
        nullable=False,
    )

    status: Mapped[RescueMemberStatus] = mapped_column(
        enum_type(RescueMemberStatus),
        default=RescueMemberStatus.ACTIVE,
        nullable=False,
    )

    is_team_leader: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_medically_fit_for_duty: Mapped[
        bool | None
    ] = mapped_column(
        Boolean,
        nullable=True,
    )

    medical_fitness_valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    qualification_verified: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    qualification_verified_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    left_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    team: Mapped["RescueTeam"] = relationship(
        "RescueTeam",
        back_populates="members",
    )

    certifications: Mapped[
        list["RescueTeamCertification"]
    ] = relationship(
        "RescueTeamCertification",
        back_populates="member",
        cascade="save-update, merge",
    )

    __table_args__ = (
        UniqueConstraint(
            "team_id",
            "personnel_reference",
            name="uq_rescue_team_personnel",
        ),
        Index(
            "ix_rescue_member_team_status",
            "team_id",
            "status",
        ),
    )

    @property
    def is_available_for_duty(self) -> bool:
        return (
            self.status == RescueMemberStatus.ACTIVE
            and self.qualification_verified
            and self.is_medically_fit_for_duty is True
            and is_future(
                self.medical_fitness_valid_until
            )
        )


# ============================================================
# MEMBER CERTIFICATIONS
# ============================================================

class RescueTeamCertification(Base):
    __tablename__ = "rescue_team_certifications"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    member_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_team_members.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    certification_type: Mapped[
        RescueCertificationType
    ] = mapped_column(
        enum_type(RescueCertificationType),
        nullable=False,
    )

    certification_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    issuing_organization: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    certificate_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    issued_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    expires_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    is_verified: Mapped[bool] = mapped_column(
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

    evidence_storage_reference: Mapped[
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

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    member: Mapped["RescueTeamMember"] = relationship(
        "RescueTeamMember",
        back_populates="certifications",
    )

    @property
    def is_valid(self) -> bool:
        return (
            self.is_verified
            and (
                self.expires_at is None
                or is_future(self.expires_at)
            )
        )


# ============================================================
# TEAM VEHICLES
# ============================================================

class RescueTeamVehicle(Base):
    __tablename__ = "rescue_team_vehicles"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    team_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_teams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    vehicle_reference: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    vehicle_type: Mapped[
        RescueVehicleType
    ] = mapped_column(
        enum_type(RescueVehicleType),
        nullable=False,
    )

    status: Mapped[
        RescueVehicleStatus
    ] = mapped_column(
        enum_type(RescueVehicleStatus),
        default=RescueVehicleStatus.UNKNOWN,
        nullable=False,
    )

    passenger_capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    rescue_capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    operator_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    is_safety_inspected: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    inspection_valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    inspected_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    last_maintenance_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    next_maintenance_due_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
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

    team: Mapped["RescueTeam"] = relationship(
        "RescueTeam",
        back_populates="vehicles",
    )

    __table_args__ = (
        UniqueConstraint(
            "team_id",
            "vehicle_reference",
            name="uq_rescue_team_vehicle",
        ),
        CheckConstraint(
            "passenger_capacity >= 0",
            name="ck_rescue_vehicle_passengers",
        ),
        CheckConstraint(
            "rescue_capacity >= 0",
            name="ck_rescue_vehicle_capacity",
        ),
    )

    @property
    def is_operational(self) -> bool:
        return (
            self.status == RescueVehicleStatus.AVAILABLE
            and self.is_safety_inspected
            and is_future(self.inspection_valid_until)
        )


# ============================================================
# TEAM EQUIPMENT
# ============================================================

class RescueTeamEquipment(Base):
    __tablename__ = "rescue_team_equipment"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    team_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_teams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    equipment_type: Mapped[
        RescueEquipmentType
    ] = mapped_column(
        enum_type(RescueEquipmentType),
        nullable=False,
    )

    equipment_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    inventory_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    total_quantity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    available_quantity: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
            nullable=False,
        )
    )

    condition: Mapped[
        RescueEquipmentCondition
    ] = mapped_column(
        enum_type(RescueEquipmentCondition),
        default=RescueEquipmentCondition.UNKNOWN,
        nullable=False,
    )

    is_safety_critical: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    is_inspected: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    inspection_valid_until: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    inspected_by_user_id: Mapped[
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

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    team: Mapped["RescueTeam"] = relationship(
        "RescueTeam",
        back_populates="equipment",
    )

    __table_args__ = (
        CheckConstraint(
            "total_quantity >= 0",
            name="ck_rescue_team_equipment_total",
        ),
        CheckConstraint(
            "available_quantity >= 0",
            name="ck_rescue_team_equipment_available",
        ),
        CheckConstraint(
            "available_quantity <= total_quantity",
            name="ck_rescue_team_equipment_balance",
        ),
    )

    @property
    def is_usable(self) -> bool:
        return (
            self.available_quantity > 0
            and self.condition in {
                RescueEquipmentCondition.EXCELLENT,
                RescueEquipmentCondition.GOOD,
            }
            and self.is_inspected
            and is_future(self.inspection_valid_until)
        )


# ============================================================
# TEAM SHIFTS
# ============================================================

class RescueTeamShift(Base):
    __tablename__ = "rescue_team_shifts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    team_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_teams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    shift_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    status: Mapped[
        RescueShiftStatus
    ] = mapped_column(
        enum_type(RescueShiftStatus),
        default=RescueShiftStatus.SCHEDULED,
        nullable=False,
    )

    scheduled_start_at: Mapped[datetime] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=False,
        )
    )

    scheduled_end_at: Mapped[datetime] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=False,
        )
    )

    actual_start_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    actual_end_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    member_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    supervisor_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
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

    team: Mapped["RescueTeam"] = relationship(
        "RescueTeam",
        back_populates="shifts",
    )

    __table_args__ = (
        CheckConstraint(
            "member_count >= 0",
            name="ck_rescue_shift_members",
        ),
        Index(
            "ix_rescue_shift_team_schedule",
            "team_id",
            "scheduled_start_at",
        ),
    )


# ============================================================
# LOCATION HISTORY
# ============================================================

class RescueTeamLocation(Base):
    __tablename__ = "rescue_team_locations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    team_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_teams.id",
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

    accuracy_meters: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
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

    team: Mapped["RescueTeam"] = relationship(
        "RescueTeam",
        back_populates="location_history",
    )

    __table_args__ = (
        CheckConstraint(
            "latitude >= -90 AND latitude <= 90",
            name="ck_rescue_location_lat",
        ),
        CheckConstraint(
            "longitude >= -180 AND longitude <= 180",
            name="ck_rescue_location_lon",
        ),
        CheckConstraint(
            "accuracy_meters IS NULL OR accuracy_meters >= 0",
            name="ck_rescue_location_accuracy",
        ),
        Index(
            "ix_rescue_location_team_time",
            "team_id",
            "recorded_at",
        ),
    )

    @validates("latitude")
    def validate_latitude(
        self,
        key: str,
        value: float,
    ) -> float:
        result = validate_coordinate(
            value, key, -90, 90
        )
        assert result is not None
        return result

    @validates("longitude")
    def validate_longitude(
        self,
        key: str,
        value: float,
    ) -> float:
        result = validate_coordinate(
            value, key, -180, 180
        )
        assert result is not None
        return result


# ============================================================
# TEAM AUDIT EVENTS
# ============================================================

class RescueTeamEvent(Base):
    __tablename__ = "rescue_team_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    team_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "rescue_teams.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        RescueTeamEventType
    ] = mapped_column(
        enum_type(RescueTeamEventType),
        nullable=False,
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

    team: Mapped["RescueTeam"] = relationship(
        "RescueTeam",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_rescue_team_event_time",
            "team_id",
            "occurred_at",
        ),
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

RESCUE_TEAM_MODELS = (
    RescueTeam,
    RescueTeamMember,
    RescueTeamCertification,
    RescueTeamVehicle,
    RescueTeamEquipment,
    RescueTeamShift,
    RescueTeamLocation,
    RescueTeamEvent,
)


__all__ = [
    "RescueTeam",
    "RescueTeamMember",
    "RescueTeamCertification",
    "RescueTeamVehicle",
    "RescueTeamEquipment",
    "RescueTeamShift",
    "RescueTeamLocation",
    "RescueTeamEvent",
    "RescueTeamType",
    "RescueTeamStatus",
    "RescueTeamAvailability",
    "RescueTeamReadiness",
    "RescueMemberRole",
    "RescueMemberStatus",
    "RescueCertificationType",
    "RescueVehicleType",
    "RescueVehicleStatus",
    "RescueEquipmentType",
    "RescueEquipmentCondition",
    "RescueShiftStatus",
    "RescueTeamEventType",
    "RESCUE_TEAM_MODELS",
]
