from __future__ import annotations

"""
backend/models/damage_report.py

Kerala Flood Relief Coordination Agent

SQLAlchemy 2.x models for flood damage assessment.

Features:
- Flood damage report registration
- Building, road, bridge, vehicle and infrastructure damage
- Geographic location and district tracking
- AI/ML/CV damage assessment results
- Before/after image and satellite evidence
- Structural safety assessment
- Repair and replacement cost estimates
- Human inspection and verification
- Insurance and recovery references
- Report lifecycle and approval
- Historical assessments and audit references
- Optimistic concurrency control

SAFETY:
AI damage assessments are preliminary estimates.
They are not structural engineering certifications.
Unsafe buildings and infrastructure require inspection
by qualified personnel. Reports do not automatically
authorize repairs, demolitions or compensation.
"""


import enum
import math
import uuid

from datetime import datetime, timezone
from decimal import Decimal
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
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
)

from sqlalchemy.ext.mutable import (
    MutableDict,
    MutableList,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from backend.database.base import Base


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(enum_class: type[enum.Enum]) -> SQLEnum:
    return SQLEnum(
        enum_class,
        values_callable=lambda members: [
            member.value for member in members
        ],
        native_enum=False,
        validate_strings=True,
        length=64,
    )


def validate_non_negative(
    value: float | int | None,
    field_name: str,
) -> float | int | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a finite "
            "non-negative number"
        )

    return value


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


def validate_money(
    value: Decimal | int | float | str | None,
    field_name: str,
) -> Decimal | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"Invalid {field_name}")

    try:
        amount = Decimal(str(value))
    except (ValueError, ArithmeticError) as exc:
        raise ValueError(
            f"Invalid {field_name}"
        ) from exc

    if not amount.is_finite() or amount < 0:
        raise ValueError(
            f"{field_name} must be non-negative"
        )

    if amount > Decimal("999999999999.99"):
        raise ValueError(
            f"{field_name} exceeds supported range"
        )

    return amount


# ============================================================
# ENUMS
# ============================================================

class DamageAssetType(str, enum.Enum):
    RESIDENTIAL_BUILDING = "residential_building"
    COMMERCIAL_BUILDING = "commercial_building"
    PUBLIC_BUILDING = "public_building"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    ROAD = "road"
    BRIDGE = "bridge"
    RAILWAY = "railway"
    ELECTRICITY_INFRASTRUCTURE = "electricity_infrastructure"
    WATER_SUPPLY = "water_supply"
    TELECOMMUNICATION = "telecommunication"
    DRAINAGE = "drainage"
    DAM_INFRASTRUCTURE = "dam_infrastructure"
    AGRICULTURAL_LAND = "agricultural_land"
    CROP = "crop"
    LIVESTOCK_FACILITY = "livestock_facility"
    VEHICLE = "vehicle"
    MACHINERY = "machinery"
    OTHER = "other"


class DamageCause(str, enum.Enum):
    FLOOD_WATER = "flood_water"
    FLASH_FLOOD = "flash_flood"
    LANDSLIDE = "landslide"
    EROSION = "erosion"
    DEBRIS_IMPACT = "debris_impact"
    WATERLOGGING = "waterlogging"
    STRUCTURAL_FAILURE = "structural_failure"
    STORM = "storm"
    UNKNOWN = "unknown"
    OTHER = "other"


class DamageSeverity(str, enum.Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"
    DESTROYED = "destroyed"
    UNKNOWN = "unknown"


class DamageReportStatus(str, enum.Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    PENDING_INSPECTION = "pending_inspection"
    UNDER_ASSESSMENT = "under_assessment"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class StructuralSafetyStatus(str, enum.Enum):
    NOT_ASSESSED = "not_assessed"
    SUSPECTED_UNSAFE = "suspected_unsafe"
    RESTRICTED_ACCESS = "restricted_access"
    SAFE_BY_QUALIFIED_INSPECTOR = (
        "safe_by_qualified_inspector"
    )
    UNSAFE_BY_QUALIFIED_INSPECTOR = (
        "unsafe_by_qualified_inspector"
    )
    NOT_APPLICABLE = "not_applicable"


class DamageAssessmentMethod(str, enum.Enum):
    FIELD_INSPECTION = "field_inspection"
    DRONE_IMAGE = "drone_image"
    SATELLITE_IMAGE = "satellite_image"
    USER_UPLOADED_IMAGE = "user_uploaded_image"
    COMPUTER_VISION = "computer_vision"
    ML_PREDICTION = "ml_prediction"
    REMOTE_SENSING = "remote_sensing"
    DOCUMENT_REVIEW = "document_review"
    COMBINED = "combined"


class DamageEvidenceType(str, enum.Enum):
    BEFORE_IMAGE = "before_image"
    AFTER_IMAGE = "after_image"
    DRONE_IMAGE = "drone_image"
    SATELLITE_IMAGE = "satellite_image"
    VIDEO = "video"
    INSPECTION_DOCUMENT = "inspection_document"
    COST_ESTIMATE = "cost_estimate"
    SENSOR_DATA = "sensor_data"
    GEO_SPATIAL_DATA = "geo_spatial_data"
    OTHER = "other"


class DamageEvidenceVerification(str, enum.Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"


class DamageCostCategory(str, enum.Enum):
    MATERIALS = "materials"
    LABOUR = "labour"
    EQUIPMENT = "equipment"
    TRANSPORT = "transport"
    DEBRIS_REMOVAL = "debris_removal"
    STRUCTURAL_REPAIR = "structural_repair"
    ELECTRICAL_REPAIR = "electrical_repair"
    WATER_DAMAGE_REPAIR = "water_damage_repair"
    REPLACEMENT = "replacement"
    TEMPORARY_RELOCATION = "temporary_relocation"
    ENVIRONMENTAL_REMEDIATION = (
        "environmental_remediation"
    )
    OTHER = "other"


class DamageReviewOutcome(str, enum.Enum):
    VERIFIED = "verified"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


# ============================================================
# DAMAGE REPORT
# ============================================================

class DamageReport(Base):
    __tablename__ = "damage_reports"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    report_number: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
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

    evacuation_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    recovery_project_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    asset_type: Mapped[DamageAssetType] = mapped_column(
        enum_type(DamageAssetType),
        nullable=False,
        index=True,
    )

    asset_reference: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    asset_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    damage_cause: Mapped[DamageCause] = mapped_column(
        enum_type(DamageCause),
        default=DamageCause.UNKNOWN,
        nullable=False,
    )

    severity: Mapped[DamageSeverity] = mapped_column(
        enum_type(DamageSeverity),
        default=DamageSeverity.UNKNOWN,
        nullable=False,
        index=True,
    )

    status: Mapped[DamageReportStatus] = mapped_column(
        enum_type(DamageReportStatus),
        default=DamageReportStatus.DRAFT,
        nullable=False,
        index=True,
    )

    structural_safety_status: Mapped[
        StructuralSafetyStatus
    ] = mapped_column(
        enum_type(StructuralSafetyStatus),
        default=StructuralSafetyStatus.NOT_ASSESSED,
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

    address: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    taluk: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    panchayat: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    ward: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    flood_depth_meters: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    affected_area_sq_meters: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    estimated_affected_people: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    estimated_displaced_people: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    estimated_repair_cost_inr: Mapped[
        Decimal | None
    ] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    estimated_replacement_cost_inr: Mapped[
        Decimal | None
    ] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    approved_cost_inr: Mapped[
        Decimal | None
    ] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    currency_code: Mapped[str] = mapped_column(
        String(3),
        default="INR",
        nullable=False,
    )

    requires_emergency_inspection: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    access_restricted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_ai_assisted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_human_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    created_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    submitted_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    verified_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    approved_by_user_id: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    inspection_notes: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    rejection_reason: Mapped[str | None] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    insurance_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    audit_reference: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    damaged_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    inspected_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    submitted_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    verified_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    approved_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    closed_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
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

    assessments: Mapped[
        list["DamageAssessment"]
    ] = relationship(
        "DamageAssessment",
        back_populates="report",
        cascade="all, delete-orphan",
    )

    evidence: Mapped[
        list["DamageEvidence"]
    ] = relationship(
        "DamageEvidence",
        back_populates="report",
        cascade="all, delete-orphan",
    )

    cost_items: Mapped[
        list["DamageCostItem"]
    ] = relationship(
        "DamageCostItem",
        back_populates="report",
        cascade="all, delete-orphan",
    )

    reviews: Mapped[
        list["DamageReportReview"]
    ] = relationship(
        "DamageReportReview",
        back_populates="report",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_damage_report_district",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_damage_report_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_damage_report_longitude",
        ),
        CheckConstraint(
            "flood_depth_meters IS NULL OR "
            "flood_depth_meters >= 0",
            name="ck_damage_report_flood_depth",
        ),
        CheckConstraint(
            "affected_area_sq_meters IS NULL OR "
            "affected_area_sq_meters >= 0",
            name="ck_damage_report_area",
        ),
        CheckConstraint(
            "estimated_affected_people IS NULL OR "
            "estimated_affected_people >= 0",
            name="ck_damage_report_affected_people",
        ),
        CheckConstraint(
            "estimated_displaced_people IS NULL OR "
            "estimated_displaced_people >= 0",
            name="ck_damage_report_displaced_people",
        ),
        CheckConstraint(
            "estimated_repair_cost_inr IS NULL OR "
            "estimated_repair_cost_inr >= 0",
            name="ck_damage_report_repair_cost",
        ),
        CheckConstraint(
            "estimated_replacement_cost_inr IS NULL OR "
            "estimated_replacement_cost_inr >= 0",
            name="ck_damage_report_replacement_cost",
        ),
        CheckConstraint(
            "approved_cost_inr IS NULL OR "
            "approved_cost_inr >= 0",
            name="ck_damage_report_approved_cost",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_damage_report_version",
        ),
        Index(
            "ix_damage_report_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_damage_report_asset_severity",
            "asset_type",
            "severity",
        ),
        Index(
            "ix_damage_report_created",
            "created_at",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("latitude")
    def validate_latitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -90 <= value <= 90
        ):
            raise ValueError("Invalid latitude")

        return float(value)

    @validates("longitude")
    def validate_longitude(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -180 <= value <= 180
        ):
            raise ValueError("Invalid longitude")

        return float(value)

    @validates(
        "flood_depth_meters",
        "affected_area_sq_meters",
        "estimated_affected_people",
        "estimated_displaced_people",
    )
    def validate_measurement(
        self,
        key: str,
        value: float | int | None,
    ) -> float | int | None:
        return validate_non_negative(value, key)

    @validates(
        "estimated_repair_cost_inr",
        "estimated_replacement_cost_inr",
        "approved_cost_inr",
    )
    def validate_cost(
        self,
        key: str,
        value: Decimal | int | float | str | None,
    ) -> Decimal | None:
        return validate_money(value, key)

    @property
    def is_finalized(self) -> bool:
        return self.status in {
            DamageReportStatus.APPROVED,
            DamageReportStatus.CLOSED,
        }

    @property
    def requires_human_review(self) -> bool:
        return (
            self.status in {
                DamageReportStatus.PENDING_VERIFICATION,
                DamageReportStatus.VERIFIED,
            }
            or self.requires_emergency_inspection
            or self.structural_safety_status
            == StructuralSafetyStatus.SUSPECTED_UNSAFE
        )

    def submit(self, user_id: int) -> None:
        if self.status not in {
            DamageReportStatus.DRAFT,
            DamageReportStatus.NEEDS_REVISION,
        }:
            raise ValueError(
                "Report cannot be submitted "
                "from its current status"
            )

        if user_id <= 0:
            raise ValueError("Invalid user ID")

        self.status = DamageReportStatus.SUBMITTED
        self.submitted_by_user_id = user_id
        self.submitted_at = utc_now()

    def mark_pending_inspection(self) -> None:
        if self.status != DamageReportStatus.SUBMITTED:
            raise ValueError(
                "Only submitted reports can "
                "await inspection"
            )

        self.status = (
            DamageReportStatus.PENDING_INSPECTION
        )

    def start_assessment(self) -> None:
        if self.status not in {
            DamageReportStatus.SUBMITTED,
            DamageReportStatus.PENDING_INSPECTION,
        }:
            raise ValueError(
                "Report is not ready for assessment"
            )

        self.status = (
            DamageReportStatus.UNDER_ASSESSMENT
        )

    def submit_for_verification(self) -> None:
        if self.status != (
            DamageReportStatus.UNDER_ASSESSMENT
        ):
            raise ValueError(
                "Report must be under assessment"
            )

        self.status = (
            DamageReportStatus.PENDING_VERIFICATION
        )

    def verify(
        self,
        *,
        reviewer_id: int,
        notes: str,
    ) -> None:
        if self.status != (
            DamageReportStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Report is not pending verification"
            )

        if reviewer_id <= 0:
            raise ValueError("Invalid reviewer")

        if (
            self.created_by_user_id is not None
            and reviewer_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent verification required"
            )

        if not notes.strip():
            raise ValueError(
                "Verification notes required"
            )

        self.status = DamageReportStatus.VERIFIED
        self.is_human_verified = True
        self.verified_by_user_id = reviewer_id
        self.verified_at = utc_now()
        self.inspection_notes = notes.strip()

    def approve(
        self,
        *,
        approver_id: int,
    ) -> None:
        if self.status != DamageReportStatus.VERIFIED:
            raise ValueError(
                "Only verified reports can be approved"
            )

        if approver_id <= 0:
            raise ValueError("Invalid approver")

        if (
            self.created_by_user_id is not None
            and approver_id == self.created_by_user_id
        ):
            raise ValueError(
                "Independent approval required"
            )

        self.status = DamageReportStatus.APPROVED
        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()

    def reject(
        self,
        *,
        reason: str,
    ) -> None:
        if self.status in {
            DamageReportStatus.APPROVED,
            DamageReportStatus.CLOSED,
            DamageReportStatus.CANCELLED,
        }:
            raise ValueError(
                "Finalized report cannot be rejected"
            )

        if not reason.strip():
            raise ValueError(
                "Rejection reason required"
            )

        self.status = DamageReportStatus.REJECTED
        self.rejection_reason = reason.strip()

    def close(self) -> None:
        if self.status != DamageReportStatus.APPROVED:
            raise ValueError(
                "Only approved reports can be closed"
            )

        self.status = DamageReportStatus.CLOSED
        self.closed_at = utc_now()

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "report_number": self.report_number,
            "district_id": self.district_id,
            "asset_type": self.asset_type.value,
            "severity": self.severity.value,
            "damage_cause": self.damage_cause.value,
            "status": self.status.value,
            "structural_safety_status": (
                self.structural_safety_status.value
            ),
            "is_ai_assisted": self.is_ai_assisted,
            "is_human_verified": self.is_human_verified,
            "requires_emergency_inspection": (
                self.requires_emergency_inspection
            ),
            "access_restricted": self.access_restricted,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<DamageReport("
            f"id={self.id!r}, "
            f"number={self.report_number!r}, "
            f"status={self.status.value!r}"
            f")>"
        )


# ============================================================
# DAMAGE ASSESSMENT
# ============================================================

class DamageAssessment(Base):
    __tablename__ = "damage_assessments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "damage_reports.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    method: Mapped[
        DamageAssessmentMethod
    ] = mapped_column(
        enum_type(DamageAssessmentMethod),
        nullable=False,
    )

    predicted_severity: Mapped[
        DamageSeverity
    ] = mapped_column(
        enum_type(DamageSeverity),
        default=DamageSeverity.UNKNOWN,
        nullable=False,
    )

    confidence_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    uncertainty_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    damage_percentage: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    model_run_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    predicted_damage_classes: Mapped[list] = (
        mapped_column(
            MutableList.as_mutable(JSON),
            default=list,
            nullable=False,
        )
    )

    bounding_boxes: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )

    segmentation_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    findings: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    limitations: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )

    is_human_reviewed: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
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

    assessment_notes: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    assessed_at: Mapped[datetime] = mapped_column(
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

    report: Mapped["DamageReport"] = relationship(
        "DamageReport",
        back_populates="assessments",
    )

    __table_args__ = (
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 "
            "AND confidence_score <= 1)",
            name="ck_damage_assessment_confidence",
        ),
        CheckConstraint(
            "uncertainty_score IS NULL OR "
            "(uncertainty_score >= 0 "
            "AND uncertainty_score <= 1)",
            name="ck_damage_assessment_uncertainty",
        ),
        CheckConstraint(
            "damage_percentage IS NULL OR "
            "(damage_percentage >= 0 "
            "AND damage_percentage <= 100)",
            name="ck_damage_assessment_percentage",
        ),
        Index(
            "ix_damage_assessment_report_time",
            "report_id",
            "assessed_at",
        ),
    )

    @validates(
        "confidence_score",
        "uncertainty_score",
    )
    def validate_score(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_probability(value, key)

    @validates("damage_percentage")
    def validate_damage_percentage(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise ValueError(
                "Damage percentage must be 0–100"
            )

        return float(value)


# ============================================================
# DAMAGE EVIDENCE
# ============================================================

class DamageEvidence(Base):
    __tablename__ = "damage_evidence"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "damage_reports.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    evidence_type: Mapped[
        DamageEvidenceType
    ] = mapped_column(
        enum_type(DamageEvidenceType),
        nullable=False,
    )

    file_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    original_filename: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    file_size_bytes: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    sha256_hash: Mapped[str | None] = (
        mapped_column(
            String(64),
            nullable=True,
        )
    )

    caption: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    captured_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    verification_status: Mapped[
        DamageEvidenceVerification
    ] = mapped_column(
        enum_type(DamageEvidenceVerification),
        default=DamageEvidenceVerification.UNVERIFIED,
        nullable=False,
    )

    malware_scan_passed: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    uploaded_by_user_id: Mapped[
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

    report: Mapped["DamageReport"] = relationship(
        "DamageReport",
        back_populates="evidence",
    )

    __table_args__ = (
        CheckConstraint(
            "file_size_bytes IS NULL OR "
            "file_size_bytes >= 0",
            name="ck_damage_evidence_file_size",
        ),
        Index(
            "ix_damage_evidence_report_type",
            "report_id",
            "evidence_type",
        ),
    )

    @property
    def is_trusted_evidence(self) -> bool:
        return (
            self.verification_status
            == DamageEvidenceVerification.VERIFIED
            and self.malware_scan_passed
            and bool(self.sha256_hash)
        )


# ============================================================
# REPAIR AND REPLACEMENT COST ITEMS
# ============================================================

class DamageCostItem(Base):
    __tablename__ = "damage_cost_items"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "damage_reports.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    category: Mapped[DamageCostCategory] = (
        mapped_column(
            enum_type(DamageCostCategory),
            nullable=False,
        )
    )

    description: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    quantity: Mapped[Decimal] = mapped_column(
        Numeric(12, 3),
        nullable=False,
    )

    unit: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    unit_cost_inr: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
    )

    estimated_total_inr: Mapped[Decimal] = (
        mapped_column(
            Numeric(14, 2),
            nullable=False,
        )
    )

    source_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
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

    report: Mapped["DamageReport"] = relationship(
        "DamageReport",
        back_populates="cost_items",
    )

    __table_args__ = (
        CheckConstraint(
            "quantity > 0",
            name="ck_damage_cost_quantity",
        ),
        CheckConstraint(
            "unit_cost_inr >= 0",
            name="ck_damage_cost_unit",
        ),
        CheckConstraint(
            "estimated_total_inr >= 0",
            name="ck_damage_cost_total",
        ),
    )

    @validates(
        "unit_cost_inr",
        "estimated_total_inr",
    )
    def validate_cost(
        self,
        key: str,
        value: Decimal | int | float | str,
    ) -> Decimal:
        amount = validate_money(value, key)

        if amount is None:
            raise ValueError(
                f"{key} is required"
            )

        return amount

    @validates("quantity")
    def validate_quantity(
        self,
        key: str,
        value: Decimal | int | float | str,
    ) -> Decimal:
        if isinstance(value, bool):
            raise ValueError("Invalid quantity")

        try:
            quantity = Decimal(str(value))
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(
                "Invalid quantity"
            ) from exc

        if (
            not quantity.is_finite()
            or quantity <= 0
            or quantity > Decimal("999999999.999")
        ):
            raise ValueError(
                "Quantity must be positive"
            )

        return quantity


# ============================================================
# HUMAN REVIEW HISTORY
# ============================================================

class DamageReportReview(Base):
    __tablename__ = "damage_report_reviews"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "damage_reports.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    reviewer_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    outcome: Mapped[
        DamageReviewOutcome
    ] = mapped_column(
        enum_type(DamageReviewOutcome),
        nullable=False,
    )

    notes: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    inspection_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    qualification_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    reviewed_at: Mapped[datetime] = mapped_column(
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

    report: Mapped["DamageReport"] = relationship(
        "DamageReport",
        back_populates="reviews",
    )

    __table_args__ = (
        CheckConstraint(
            "reviewer_id > 0",
            name="ck_damage_review_reviewer",
        ),
        Index(
            "ix_damage_review_report_time",
            "report_id",
            "reviewed_at",
        ),
    )


# ============================================================
# ORM TIMESTAMP EVENTS
# ============================================================

@event.listens_for(
    DamageReport,
    "before_update",
)
def update_damage_report_timestamp(
    mapper: Any,
    connection: Any,
    target: DamageReport,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

DAMAGE_REPORT_MODELS = (
    DamageReport,
    DamageAssessment,
    DamageEvidence,
    DamageCostItem,
    DamageReportReview,
)


__all__ = [
    "DamageReport",
    "DamageAssessment",
    "DamageEvidence",
    "DamageCostItem",
    "DamageReportReview",
    "DamageAssetType",
    "DamageCause",
    "DamageSeverity",
    "DamageReportStatus",
    "StructuralSafetyStatus",
    "DamageAssessmentMethod",
    "DamageEvidenceType",
    "DamageEvidenceVerification",
    "DamageCostCategory",
    "DamageReviewOutcome",
    "DAMAGE_REPORT_MODELS",
]
