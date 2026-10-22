from __future__ import annotations

"""
backend/models/flood_alert.py

Kerala Flood Relief Coordination Agent
=======================================

SQLAlchemy 2.x ORM models for flood alert management.

Features:
- District-level flood alerts
- Rainfall, river, reservoir and flash-flood alerts
- Landslide and infrastructure-related flood warnings
- Official alerts and AI-generated risk advisories
- Source provenance and evidence references
- Alert severity, confidence and uncertainty
- Human verification and approval workflows
- Geographic targeting
- Multilingual alert content
- Alert publication and expiry
- Notification delivery tracking
- Acknowledgment and escalation history
- Duplicate prevention and version control
- Audit-friendly event records

SAFETY:
An AI prediction is not an official emergency warning.

Public dissemination of operational warnings must be
authorized by the responsible authority. This module
does not send notifications or initiate evacuation.
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
    maximum: int = 1000,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string"
        )

    result = value.strip()

    if not result:
        raise ValueError(
            f"{field_name} is required"
        )

    if len(result) > maximum:
        raise ValueError(
            f"{field_name} exceeds {maximum} characters"
        )

    return result


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


def validate_aware_datetime(
    value: datetime | None,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None

    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# ENUMERATIONS
# ============================================================

class FloodAlertType(str, enum.Enum):
    HEAVY_RAINFALL = "heavy_rainfall"
    EXTREME_RAINFALL = "extreme_rainfall"
    RIVER_LEVEL_RISE = "river_level_rise"
    RIVER_OVERFLOW = "river_overflow"
    RESERVOIR_HIGH_LEVEL = "reservoir_high_level"
    DAM_RELEASE = "dam_release"
    FLOOD_RISK = "flood_risk"
    FLASH_FLOOD = "flash_flood"
    URBAN_FLOODING = "urban_flooding"
    COASTAL_FLOODING = "coastal_flooding"
    LANDSLIDE_RISK = "landslide_risk"
    ROAD_INUNDATION = "road_inundation"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    EVACUATION_ADVISORY = "evacuation_advisory"
    ALL_CLEAR = "all_clear"
    OTHER = "other"


class FloodAlertSeverity(str, enum.Enum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class FloodAlertStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    UPDATED = "updated"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class FloodAlertSourceType(str, enum.Enum):
    IMD = "imd"
    CWC = "cwc"
    KSDMA = "ksdma"
    DISTRICT_AUTHORITY = "district_authority"
    DAM_AUTHORITY = "dam_authority"
    WEATHER_STATION = "weather_station"
    RIVER_STATION = "river_station"
    SATELLITE = "satellite"
    FIELD_REPORT = "field_report"
    ML_MODEL = "ml_model"
    AI_AGENT = "ai_agent"
    MANUAL = "manual"
    OTHER = "other"


class FloodAlertVerificationStatus(str, enum.Enum):
    UNVERIFIED = "unverified"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    REJECTED = "rejected"


class FloodAlertChannel(str, enum.Enum):
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    WHATSAPP = "whatsapp"
    VOICE_CALL = "voice_call"
    WEB = "web"
    DASHBOARD = "dashboard"
    PUBLIC_ADDRESS = "public_address"
    OFFICIAL_API = "official_api"


class FloodAlertDeliveryStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FloodAlertEventType(str, enum.Enum):
    CREATED = "created"
    EVIDENCE_ADDED = "evidence_added"
    VERIFICATION_REQUESTED = "verification_requested"
    VERIFIED = "verified"
    REJECTED = "rejected"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    UPDATED = "updated"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


# ============================================================
# FLOOD ALERT MODEL
# ============================================================

class FloodAlert(Base):
    __tablename__ = "flood_alerts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    alert_code: Mapped[str] = mapped_column(
        String(120),
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

    forecast_reference: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    weather_station_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    river_station_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    dam_reference: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    alert_type: Mapped[
        FloodAlertType
    ] = mapped_column(
        enum_type(FloodAlertType),
        nullable=False,
        index=True,
    )

    severity: Mapped[
        FloodAlertSeverity
    ] = mapped_column(
        enum_type(FloodAlertSeverity),
        default=FloodAlertSeverity.UNKNOWN,
        nullable=False,
        index=True,
    )

    status: Mapped[
        FloodAlertStatus
    ] = mapped_column(
        enum_type(FloodAlertStatus),
        default=FloodAlertStatus.DRAFT,
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    instructions: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    title_malayalam: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    message_malayalam: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    instructions_malayalam: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    source_type: Mapped[
        FloodAlertSourceType
    ] = mapped_column(
        enum_type(FloodAlertSourceType),
        nullable=False,
    )

    source_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    source_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    source_alert_id: Mapped[str | None] = (
        mapped_column(
            String(150),
            nullable=True,
        )
    )

    official_warning_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    is_official_alert: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    agent_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    confidence_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    uncertainty_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    affected_area_name: Mapped[
        str | None
    ] = mapped_column(
        String(255),
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

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    radius_km: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    geographic_area_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    estimated_affected_population: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    verification_status: Mapped[
        FloodAlertVerificationStatus
    ] = mapped_column(
        enum_type(FloodAlertVerificationStatus),
        default=FloodAlertVerificationStatus.UNVERIFIED,
        nullable=False,
    )

    requires_human_approval: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_by_user_id: Mapped[
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

    approved_by_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    issuing_authority: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    approval_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
        nullable=True,
    )

    review_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
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

    issued_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    effective_at: Mapped[datetime | None] = (
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

    published_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    cancelled_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    cancellation_reason: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    supersedes_alert_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        ForeignKey(
            "flood_alerts.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    correlation_id: Mapped[str | None] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    idempotency_key: Mapped[str | None] = (
        mapped_column(
            String(150),
            unique=True,
            nullable=True,
        )
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

    evidence: Mapped[
        list["FloodAlertEvidence"]
    ] = relationship(
        "FloodAlertEvidence",
        back_populates="alert",
        cascade="save-update, merge",
    )

    deliveries: Mapped[
        list["FloodAlertDelivery"]
    ] = relationship(
        "FloodAlertDelivery",
        back_populates="alert",
        cascade="save-update, merge",
    )

    acknowledgments: Mapped[
        list["FloodAlertAcknowledgment"]
    ] = relationship(
        "FloodAlertAcknowledgment",
        back_populates="alert",
        cascade="save-update, merge",
    )

    events: Mapped[
        list["FloodAlertEvent"]
    ] = relationship(
        "FloodAlertEvent",
        back_populates="alert",
        cascade="save-update, merge",
    )

    __table_args__ = (
        CheckConstraint(
            "district_id > 0",
            name="ck_flood_alert_district",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 "
            "AND confidence_score <= 1)",
            name="ck_flood_alert_confidence",
        ),
        CheckConstraint(
            "uncertainty_score IS NULL OR "
            "(uncertainty_score >= 0 "
            "AND uncertainty_score <= 1)",
            name="ck_flood_alert_uncertainty",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_flood_alert_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_flood_alert_longitude",
        ),
        CheckConstraint(
            "radius_km IS NULL OR radius_km >= 0",
            name="ck_flood_alert_radius",
        ),
        CheckConstraint(
            "estimated_affected_population IS NULL OR "
            "estimated_affected_population >= 0",
            name="ck_flood_alert_population",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_flood_alert_version",
        ),
        Index(
            "ix_flood_alert_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_flood_alert_severity_time",
            "severity",
            "created_at",
        ),
        Index(
            "ix_flood_alert_type_effective",
            "alert_type",
            "effective_at",
        ),
        Index(
            "ix_flood_alert_source",
            "source_type",
            "source_alert_id",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates(
        "confidence_score",
        "uncertainty_score",
    )
    def validate_probability(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_score(value, key)

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

    @validates("radius_km")
    def validate_radius(
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
            or value < 0
        ):
            raise ValueError(
                "radius_km must be non-negative"
            )

        return float(value)

    @validates(
        "effective_at",
        "expires_at",
        "issued_at",
    )
    def validate_timestamp(
        self,
        key: str,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware_datetime(
            value, key
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            FloodAlertStatus.EXPIRED,
            FloodAlertStatus.CANCELLED,
            FloodAlertStatus.REJECTED,
        }

    @property
    def is_human_approved(self) -> bool:
        return (
            self.approved_by_user_id is not None
            and self.approved_at is not None
            and bool(self.approval_reference)
            and bool(self.issuing_authority)
        )

    def is_expired(
        self,
        now: datetime | None = None,
    ) -> bool:
        if self.expires_at is None:
            return False

        now = validate_aware_datetime(
            now or utc_now(), "now"
        )

        return self.expires_at <= now

    @property
    def can_publish(self) -> bool:
        if self.status not in {
            FloodAlertStatus.APPROVED,
            FloodAlertStatus.SCHEDULED,
        }:
            return False

        if (
            self.verification_status
            != FloodAlertVerificationStatus.VERIFIED
        ):
            return False

        if not self.is_human_approved:
            return False

        if self.is_expired():
            return False

        if self.effective_at is None:
            return False

        if self.expires_at is None:
            return False

        if self.effective_at >= self.expires_at:
            return False

        return True

    def submit_for_verification(self) -> None:
        if self.status != FloodAlertStatus.DRAFT:
            raise ValueError(
                "Only draft alerts can be submitted"
            )

        self.status = (
            FloodAlertStatus.PENDING_VERIFICATION
        )
        self.verification_status = (
            FloodAlertVerificationStatus.UNDER_REVIEW
        )

    def verify(
        self,
        *,
        verifier_id: int,
        notes: str,
    ) -> None:
        if self.status != (
            FloodAlertStatus.PENDING_VERIFICATION
        ):
            raise ValueError(
                "Alert is not pending verification"
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

        self.review_notes = require_text(
            notes, "notes", 5000
        )
        self.verified_by_user_id = verifier_id
        self.verified_at = utc_now()
        self.verification_status = (
            FloodAlertVerificationStatus.VERIFIED
        )
        self.status = FloodAlertStatus.VERIFIED

    def request_approval(self) -> None:
        if self.status != FloodAlertStatus.VERIFIED:
            raise ValueError(
                "Alert must be verified"
            )

        self.status = (
            FloodAlertStatus.PENDING_APPROVAL
        )

    def approve(
        self,
        *,
        approver_id: int,
        issuing_authority: str,
        approval_reference: str,
    ) -> None:
        if self.status != (
            FloodAlertStatus.PENDING_APPROVAL
        ):
            raise ValueError(
                "Alert is not pending approval"
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

        if approver_id == self.verified_by_user_id:
            raise ValueError(
                "Approver must differ from verifier"
            )

        self.issuing_authority = require_text(
            issuing_authority,
            "issuing_authority",
            200,
        )
        self.approval_reference = require_text(
            approval_reference,
            "approval_reference",
            500,
        )

        self.approved_by_user_id = approver_id
        self.approved_at = utc_now()
        self.status = FloodAlertStatus.APPROVED

    def publish(
        self,
        *,
        published_at: datetime | None = None,
    ) -> None:
        if not self.can_publish:
            raise ValueError(
                "Alert is not authorized for publication"
            )

        now = validate_aware_datetime(
            published_at or utc_now(),
            "published_at",
        )

        if now < self.effective_at:
            raise ValueError(
                "Alert effective time has not arrived"
            )

        if now >= self.expires_at:
            raise ValueError(
                "Alert has expired"
            )

        self.status = FloodAlertStatus.PUBLISHED
        self.published_at = now

    def cancel(
        self,
        *,
        reason: str,
    ) -> None:
        if self.is_terminal:
            raise ValueError(
                "Alert is already terminal"
            )

        self.cancellation_reason = require_text(
            reason, "reason", 5000
        )
        self.status = FloodAlertStatus.CANCELLED
        self.cancelled_at = utc_now()

    def mark_expired(self) -> None:
        if self.is_terminal:
            raise ValueError(
                "Alert is already terminal"
            )

        if not self.is_expired():
            raise ValueError(
                "Alert has not reached its expiry time"
            )

        self.status = FloodAlertStatus.EXPIRED

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "alert_code": self.alert_code,
            "district_id": self.district_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "message": self.message,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "is_official_alert": self.is_official_alert,
            "is_ai_generated": self.is_ai_generated,
            "verification_status": (
                self.verification_status.value
            ),
            "is_human_approved": self.is_human_approved,
            "effective_at": (
                self.effective_at.isoformat()
                if self.effective_at
                else None
            ),
            "expires_at": (
                self.expires_at.isoformat()
                if self.expires_at
                else None
            ),
        }


# ============================================================
# FLOOD ALERT EVIDENCE
# ============================================================

class FloodAlertEvidence(Base):
    __tablename__ = "flood_alert_evidence"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    alert_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "flood_alerts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    source_type: Mapped[
        FloodAlertSourceType
    ] = mapped_column(
        enum_type(FloodAlertSourceType),
        nullable=False,
    )

    source_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    relevance_score: Mapped[float | None] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    reliability_score: Mapped[float | None] = (
        mapped_column(
            Float,
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

    observed_at: Mapped[datetime | None] = (
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

    alert: Mapped["FloodAlert"] = relationship(
        "FloodAlert",
        back_populates="evidence",
    )

    __table_args__ = (
        CheckConstraint(
            "relevance_score IS NULL OR "
            "(relevance_score >= 0 "
            "AND relevance_score <= 1)",
            name="ck_flood_evidence_relevance",
        ),
        CheckConstraint(
            "reliability_score IS NULL OR "
            "(reliability_score >= 0 "
            "AND reliability_score <= 1)",
            name="ck_flood_evidence_reliability",
        ),
        Index(
            "ix_flood_evidence_alert_source",
            "alert_id",
            "source_type",
        ),
    )

    @validates(
        "relevance_score",
        "reliability_score",
    )
    def validate_evidence_score(
        self,
        key: str,
        value: float | None,
    ) -> float | None:
        return validate_score(value, key)


# ============================================================
# FLOOD ALERT DELIVERY
# ============================================================

class FloodAlertDelivery(Base):
    __tablename__ = "flood_alert_deliveries"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    alert_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "flood_alerts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    channel: Mapped[
        FloodAlertChannel
    ] = mapped_column(
        enum_type(FloodAlertChannel),
        nullable=False,
    )

    recipient_reference: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    status: Mapped[
        FloodAlertDeliveryStatus
    ] = mapped_column(
        enum_type(FloodAlertDeliveryStatus),
        default=FloodAlertDeliveryStatus.PENDING,
        nullable=False,
    )

    provider_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    provider_message_id: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    idempotency_key: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        unique=True,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    last_error_code: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    queued_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    sent_at: Mapped[datetime | None] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    delivered_at: Mapped[datetime | None] = (
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

    alert: Mapped["FloodAlert"] = relationship(
        "FloodAlert",
        back_populates="deliveries",
    )

    __table_args__ = (
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_flood_delivery_attempts",
        ),
        Index(
            "ix_flood_delivery_alert_status",
            "alert_id",
            "status",
        ),
    )


# ============================================================
# ALERT ACKNOWLEDGMENT
# ============================================================

class FloodAlertAcknowledgment(Base):
    __tablename__ = "flood_alert_acknowledgments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    alert_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "flood_alerts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    actor_user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    actor_reference: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    channel: Mapped[
        FloodAlertChannel
    ] = mapped_column(
        enum_type(FloodAlertChannel),
        nullable=False,
    )

    acknowledgment_reference: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    acknowledged_at: Mapped[datetime] = mapped_column(
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

    alert: Mapped["FloodAlert"] = relationship(
        "FloodAlert",
        back_populates="acknowledgments",
    )

    __table_args__ = (
        UniqueConstraint(
            "alert_id",
            "actor_reference",
            "channel",
            name="uq_flood_alert_ack_actor_channel",
        ),
        Index(
            "ix_flood_alert_ack_time",
            "alert_id",
            "acknowledged_at",
        ),
    )


# ============================================================
# ALERT EVENT HISTORY
# ============================================================

class FloodAlertEvent(Base):
    __tablename__ = "flood_alert_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    alert_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "flood_alerts.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[
        FloodAlertEventType
    ] = mapped_column(
        enum_type(FloodAlertEventType),
        nullable=False,
    )

    previous_status: Mapped[
        FloodAlertStatus | None
    ] = mapped_column(
        enum_type(FloodAlertStatus),
        nullable=True,
    )

    new_status: Mapped[
        FloodAlertStatus | None
    ] = mapped_column(
        enum_type(FloodAlertStatus),
        nullable=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_user_id: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    actor_agent_name: Mapped[
        str | None
    ] = mapped_column(
        String(150),
        nullable=True,
    )

    correlation_id: Mapped[
        str | None
    ] = mapped_column(
        String(100),
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

    alert: Mapped["FloodAlert"] = relationship(
        "FloodAlert",
        back_populates="events",
    )

    __table_args__ = (
        Index(
            "ix_flood_alert_event_time",
            "alert_id",
            "occurred_at",
        ),
    )


# ============================================================
# ORM EVENTS
# ============================================================

@event.listens_for(
    FloodAlert,
    "before_update",
)
def update_flood_alert_timestamp(
    mapper: Any,
    connection: Any,
    target: FloodAlert,
) -> None:
    target.updated_at = utc_now()


# ============================================================
# MODEL REGISTRY
# ============================================================

FLOOD_ALERT_MODELS = (
    FloodAlert,
    FloodAlertEvidence,
    FloodAlertDelivery,
    FloodAlertAcknowledgment,
    FloodAlertEvent,
)


__all__ = [
    "FloodAlert",
    "FloodAlertEvidence",
    "FloodAlertDelivery",
    "FloodAlertAcknowledgment",
    "FloodAlertEvent",
    "FloodAlertType",
    "FloodAlertSeverity",
    "FloodAlertStatus",
    "FloodAlertSourceType",
    "FloodAlertVerificationStatus",
    "FloodAlertChannel",
    "FloodAlertDeliveryStatus",
    "FloodAlertEventType",
    "FLOOD_ALERT_MODELS",
]
