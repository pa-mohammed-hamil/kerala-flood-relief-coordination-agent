from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/rescue.py

Pydantic v2 schemas for rescue operations.

Modules:
- Rescue requests and emergency triage
- Rescue teams and personnel
- Rescue vehicles and equipment
- Rescue team GPS tracking
- Rescue mission management
- AI rescue assignment recommendations
- Rescue dispatch and human approval
- Route planning and navigation
- Rescue mission progress
- Rescue completion and verification
- Rescue analytics and audit logging

SAFETY:
- AI recommendations never authorize physical dispatch.
- Dispatch requires verified human authorization.
- A reserved team is not a dispatched team.
- Never report a rescue as completed without confirmation.
- Missing availability information is not availability.
- Emergency requests must not be discarded by AI scoring.
"""


from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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


def validate_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Datetime must include timezone information"
        )

    return value


# ============================================================
# BASE SCHEMA
# ============================================================

class RescueBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# ============================================================
# ENUMS
# ============================================================

class RescuePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RescueStatus(str, Enum):
    RECEIVED = "received"
    AWAITING_REVIEW = "awaiting_review"
    TRIAGED = "triaged"
    AWAITING_ASSIGNMENT = "awaiting_assignment"
    ASSIGNED = "assigned"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    RESCUED = "rescued"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RescueType(str, Enum):
    FLOOD_RESCUE = "flood_rescue"
    BOAT_RESCUE = "boat_rescue"
    WATER_RESCUE = "water_rescue"
    BUILDING_EVACUATION = "building_evacuation"
    MEDICAL_RESCUE = "medical_rescue"
    TRAPPED_PERSON = "trapped_person"
    VEHICLE_RESCUE = "vehicle_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    OTHER = "other"


class RescueTeamStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    ASSIGNED = "assigned"
    DEPLOYED = "deployed"
    ON_SCENE = "on_scene"
    RETURNING = "returning"
    RESTING = "resting"
    UNAVAILABLE = "unavailable"
    OFFLINE = "offline"


class RescueCapability(str, Enum):
    WATER_RESCUE = "water_rescue"
    BOAT_OPERATION = "boat_operation"
    SWIMMING = "swimming"
    DIVING = "diving"
    FIRST_AID = "first_aid"
    ADVANCED_MEDICAL = "advanced_medical"
    ROPE_RESCUE = "rope_rescue"
    HIGH_ANGLE_RESCUE = "high_angle_rescue"
    SEARCH_AND_RESCUE = "search_and_rescue"
    NIGHT_OPERATIONS = "night_operations"
    EVACUATION = "evacuation"
    HEAVY_EQUIPMENT = "heavy_equipment"
    OTHER = "other"


class RescueVehicleType(str, Enum):
    RESCUE_BOAT = "rescue_boat"
    INFLATABLE_BOAT = "inflatable_boat"
    AMBULANCE = "ambulance"
    JEEP = "jeep"
    TRUCK = "truck"
    HELICOPTER = "helicopter"
    OTHER = "other"


class RescueVehicleStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    DEPLOYED = "deployed"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"


class DispatchStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    TEAM_RESERVED = "team_reserved"
    DISPATCH_CONFIRMED = "dispatch_confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RescueMissionStatus(str, Enum):
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    RESCUE_IN_PROGRESS = "rescue_in_progress"
    RETURNING = "returning"
    COMPLETED = "completed"
    ABORTED = "aborted"
    CANCELLED = "cancelled"


class RouteStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    BLOCKED = "blocked"
    UNSAFE = "unsafe"
    UNAVAILABLE = "unavailable"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


# ============================================================
# LOCATION
# ============================================================

class RescueLocation(RescueBase):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    district: str

    address: str | None = None
    landmark: str | None = None
    panchayat: str | None = None
    ward: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


# ============================================================
# RESCUE REQUEST
# ============================================================

class RescueRequestCreate(RescueBase):
    sos_request_id: UUID | None = None

    rescue_type: RescueType

    location: RescueLocation

    description: str = Field(
        min_length=5,
        max_length=5000,
    )

    contact_phone: str | None = None

    people_count: int = Field(
        default=1,
        ge=1,
    )

    children_count: int = Field(
        default=0,
        ge=0,
    )

    elderly_count: int = Field(
        default=0,
        ge=0,
    )

    injured_count: int = Field(
        default=0,
        ge=0,
    )

    trapped: bool = False
    immediate_danger: bool = False
    medical_emergency: bool = False

    reported_water_depth_m: float | None = Field(
        default=None,
        ge=0,
    )

    required_capabilities: list[
        RescueCapability
    ] = Field(default_factory=list)

    reported_by: str | None = None

    reported_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_people_counts(self):
        if (
            self.children_count > self.people_count
            or self.elderly_count > self.people_count
            or self.injured_count > self.people_count
        ):
            raise ValueError(
                "Subgroup counts cannot exceed people_count"
            )

        return self


class RescueRequestResponse(RescueRequestCreate):
    request_id: UUID = Field(
        default_factory=uuid4
    )

    status: RescueStatus = RescueStatus.RECEIVED

    priority: RescuePriority | None = None

    assigned_team_id: UUID | None = None

    assigned_mission_id: UUID | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RESCUE TRIAGE
# ============================================================

class RescueTriageAssessment(RescueBase):
    assessment_id: UUID = Field(
        default_factory=uuid4
    )

    request_id: UUID

    priority: RescuePriority

    assessment_notes: str = Field(
        min_length=1,
        max_length=3000,
    )

    assessed_by: str

    assessed_at: datetime = Field(
        default_factory=utc_now
    )


class AIRescueTriageRecommendation(RescueBase):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    request_id: UUID

    suggested_priority: RescuePriority

    priority_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    reasons: list[str] = Field(
        default_factory=list
    )

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    human_review_required: bool = True

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RESCUE TEAM PERSONNEL
# ============================================================

class RescuePersonnel(RescueBase):
    personnel_id: UUID = Field(
        default_factory=uuid4
    )

    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    role: str

    capabilities: list[RescueCapability] = Field(
        default_factory=list
    )

    contact_phone: str | None = None

    verified: bool = False


# ============================================================
# RESCUE TEAM
# ============================================================

class RescueTeamCreate(RescueBase):
    team_name: str = Field(
        min_length=2,
        max_length=200,
    )

    organization: str

    district: str

    capabilities: list[RescueCapability] = Field(
        default_factory=list
    )

    personnel: list[RescuePersonnel] = Field(
        default_factory=list
    )

    team_capacity: int = Field(
        ge=1,
    )

    contact_phone: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class RescueTeamResponse(RescueTeamCreate):
    team_id: UUID = Field(
        default_factory=uuid4
    )

    status: RescueTeamStatus = (
        RescueTeamStatus.UNAVAILABLE
    )

    current_location: RescueLocation | None = None

    assigned_mission_id: UUID | None = None

    verified: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RESCUE VEHICLE
# ============================================================

class RescueVehicleCreate(RescueBase):
    registration_number: str = Field(
        min_length=3,
        max_length=30,
    )

    vehicle_type: RescueVehicleType

    team_id: UUID | None = None

    passenger_capacity: int = Field(
        ge=1,
    )

    equipment: list[str] = Field(
        default_factory=list
    )

    district: str

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)


class RescueVehicleResponse(RescueVehicleCreate):
    vehicle_id: UUID = Field(
        default_factory=uuid4
    )

    status: RescueVehicleStatus = (
        RescueVehicleStatus.UNAVAILABLE
    )

    current_location: RescueLocation | None = None

    verified: bool = False

    updated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RESCUE EQUIPMENT
# ============================================================

class RescueEquipment(RescueBase):
    equipment_id: UUID = Field(
        default_factory=uuid4
    )

    name: str

    category: str

    team_id: UUID | None = None

    quantity: int = Field(
        default=1,
        ge=0,
    )

    operational: bool = False

    last_inspection_at: datetime | None = None


# ============================================================
# GPS TRACKING
# ============================================================

class RescueTeamLocationUpdate(RescueBase):
    team_id: UUID

    location: RescueLocation

    accuracy_m: float | None = Field(
        default=None,
        ge=0,
    )

    speed_kmh: float | None = Field(
        default=None,
        ge=0,
    )

    heading_degrees: float | None = Field(
        default=None,
        ge=0,
        le=360,
    )

    recorded_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_datetime(value)


# ============================================================
# RESCUE TEAM MATCHING
# ============================================================

class RescueTeamMatch(RescueBase):
    team_id: UUID

    request_id: UUID

    match_score: float = Field(
        ge=0,
        le=100,
    )

    distance_km: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_travel_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    matched_capabilities: list[
        RescueCapability
    ] = Field(default_factory=list)

    missing_capabilities: list[
        RescueCapability
    ] = Field(default_factory=list)

    team_availability_verified: bool = False

    route_safety_verified: bool = False

    reasons: list[str] = Field(
        default_factory=list
    )


class RescueAssignmentRecommendation(RescueBase):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    request_id: UUID

    recommended_team_id: UUID | None = None

    alternatives: list[RescueTeamMatch] = Field(
        default_factory=list
    )

    reasoning_summary: str

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    human_approval_required: bool = True

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RESCUE DISPATCH
# ============================================================

class RescueDispatchCreate(RescueBase):
    request_id: UUID
    team_id: UUID

    vehicle_id: UUID | None = None

    pickup_location: RescueLocation

    destination: RescueLocation | None = None

    recommendation_id: UUID | None = None

    dispatch_notes: str | None = None


class RescueDispatchResponse(RescueDispatchCreate):
    dispatch_id: UUID = Field(
        default_factory=uuid4
    )

    status: DispatchStatus = (
        DispatchStatus.PENDING_APPROVAL
    )

    approved_by: str | None = None
    approved_at: datetime | None = None

    team_reserved_at: datetime | None = None

    dispatch_confirmed_by: str | None = None
    dispatch_confirmed_at: datetime | None = None

    dispatch_reference: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_dispatch_confirmation(self):
        if self.status == DispatchStatus.DISPATCH_CONFIRMED:
            if not (
                self.approved_by
                and self.approved_at
                and self.dispatch_confirmed_by
                and self.dispatch_confirmed_at
                and self.dispatch_reference
            ):
                raise ValueError(
                    "Confirmed dispatch requires approval "
                    "and operational confirmation"
                )

        return self


# ============================================================
# ROUTE PLANNING
# ============================================================

class RescueRouteRequest(RescueBase):
    request_id: UUID

    origin: RescueLocation
    destination: RescueLocation

    vehicle_type: RescueVehicleType

    avoid_flooded_roads: bool = True
    avoid_blocked_roads: bool = True

    include_alternatives: bool = True


class RescueRoutePoint(RescueBase):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class RescueRouteResponse(RescueBase):
    route_id: UUID = Field(
        default_factory=uuid4
    )

    request_id: UUID

    status: RouteStatus = RouteStatus.PROPOSED

    distance_km: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_duration_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    route_points: list[RescueRoutePoint] = Field(
        default_factory=list
    )

    blocked_segments: list[str] = Field(
        default_factory=list
    )

    safety_warnings: list[str] = Field(
        default_factory=list
    )

    source: str | None = None

    verified_by: str | None = None
    verified_at: datetime | None = None

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_route_verification(self):
        if self.status == RouteStatus.VERIFIED:
            if not self.verified_by or not self.verified_at:
                raise ValueError(
                    "Verified route requires reviewer information"
                )

        return self


# ============================================================
# RESCUE MISSION
# ============================================================

class RescueMissionCreate(RescueBase):
    request_id: UUID
    team_id: UUID

    vehicle_id: UUID | None = None
    dispatch_id: UUID | None = None

    mission_name: str = Field(
        min_length=2,
        max_length=200,
    )

    objective: str = Field(
        min_length=5,
        max_length=3000,
    )

    target_location: RescueLocation

    expected_people_count: int = Field(
        ge=1,
    )

    planned_route_id: UUID | None = None

    safety_notes: str | None = None


class RescueMissionResponse(RescueMissionCreate):
    mission_id: UUID = Field(
        default_factory=uuid4
    )

    status: RescueMissionStatus = (
        RescueMissionStatus.PLANNED
    )

    approved_by: str | None = None
    approved_at: datetime | None = None

    dispatched_at: datetime | None = None

    arrived_at: datetime | None = None

    completed_at: datetime | None = None

    confirmed_people_rescued: int | None = Field(
        default=None,
        ge=0,
    )

    completion_verified_by: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_mission_completion(self):
        if self.status == RescueMissionStatus.COMPLETED:
            if (
                self.completed_at is None
                or not self.completion_verified_by
                or self.confirmed_people_rescued is None
            ):
                raise ValueError(
                    "Completed mission requires verified outcome"
                )

        return self


# ============================================================
# MISSION PROGRESS
# ============================================================

class RescueMissionProgress(RescueBase):
    mission_id: UUID

    status: RescueMissionStatus

    location: RescueLocation | None = None

    people_rescued_so_far: int | None = Field(
        default=None,
        ge=0,
    )

    issues: list[str] = Field(
        default_factory=list
    )

    notes: str | None = None

    reported_by: str

    reported_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RESCUE COMPLETION
# ============================================================

class RescueCompletionReport(RescueBase):
    mission_id: UUID

    confirmed_people_rescued: int = Field(
        ge=0,
    )

    people_transferred_to_shelter: int = Field(
        default=0,
        ge=0,
    )

    people_transferred_to_hospital: int = Field(
        default=0,
        ge=0,
    )

    unresolved_cases: int = Field(
        default=0,
        ge=0,
    )

    completion_notes: str

    evidence_references: list[str] = Field(
        default_factory=list
    )

    verified_by: str

    completed_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_outcomes(self):
        if (
            self.people_transferred_to_shelter
            + self.people_transferred_to_hospital
            > self.confirmed_people_rescued
        ):
            raise ValueError(
                "Transfers cannot exceed confirmed rescues"
            )

        return self


# ============================================================
# HUMAN APPROVAL
# ============================================================

class RescueApprovalRequest(RescueBase):
    approval_id: UUID = Field(
        default_factory=uuid4
    )

    dispatch_id: UUID

    action_summary: str

    reason: str

    status: ApprovalStatus = ApprovalStatus.PENDING

    requested_at: datetime = Field(
        default_factory=utc_now
    )


class RescueApprovalDecision(RescueBase):
    approval_id: UUID

    approve: bool

    reason: str = Field(
        min_length=5,
        max_length=2000,
    )


class RescueApprovalResponse(RescueBase):
    approval_id: UUID

    status: ApprovalStatus

    reviewed_by: str

    reviewed_at: datetime = Field(
        default_factory=utc_now
    )

    authorization_verified: bool = False


# ============================================================
# AUDIT LOG
# ============================================================

class RescueAuditEvent(RescueBase):
    event_id: UUID = Field(
        default_factory=uuid4
    )

    entity_id: UUID

    entity_type: str

    action: str

    performed_by: str

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RESCUE DASHBOARD
# ============================================================

class RescueDashboard(RescueBase):
    district: str | None = None

    active_rescue_requests: int | None = Field(
        default=None,
        ge=0,
    )

    critical_requests: int | None = Field(
        default=None,
        ge=0,
    )

    available_teams: int | None = Field(
        default=None,
        ge=0,
    )

    deployed_teams: int | None = Field(
        default=None,
        ge=0,
    )

    available_vehicles: int | None = Field(
        default=None,
        ge=0,
    )

    active_missions: int | None = Field(
        default=None,
        ge=0,
    )

    completed_missions: int | None = Field(
        default=None,
        ge=0,
    )

    confirmed_people_rescued: int | None = Field(
        default=None,
        ge=0,
    )

    average_dispatch_minutes: float | None = Field(
        default=None,
        ge=0,
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )

    data_verified: bool = False


# ============================================================
# API RESPONSE
# ============================================================

class RescueAPIResponse(RescueBase):
    success: bool

    message: str

    data: dict[str, Any] | None = None

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RescuePriority",
    "RescueStatus",
    "RescueType",
    "RescueTeamStatus",
    "RescueCapability",
    "RescueVehicleType",
    "RescueVehicleStatus",
    "DispatchStatus",
    "RescueMissionStatus",
    "RouteStatus",
    "ApprovalStatus",
    "RescueLocation",
    "RescueRequestCreate",
    "RescueRequestResponse",
    "RescueTriageAssessment",
    "AIRescueTriageRecommendation",
    "RescuePersonnel",
    "RescueTeamCreate",
    "RescueTeamResponse",
    "RescueVehicleCreate",
    "RescueVehicleResponse",
    "RescueEquipment",
    "RescueTeamLocationUpdate",
    "RescueTeamMatch",
    "RescueAssignmentRecommendation",
    "RescueDispatchCreate",
    "RescueDispatchResponse",
    "RescueRouteRequest",
    "RescueRoutePoint",
    "RescueRouteResponse",
    "RescueMissionCreate",
    "RescueMissionResponse",
    "RescueMissionProgress",
    "RescueCompletionReport",
    "RescueApprovalRequest",
    "RescueApprovalDecision",
    "RescueApprovalResponse",
    "RescueAuditEvent",
    "RescueDashboard",
    "RescueAPIResponse",
    "normalize_district",
    "validate_datetime",
]
