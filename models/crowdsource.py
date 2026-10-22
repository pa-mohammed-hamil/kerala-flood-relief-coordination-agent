"""
backend/models/crowdsource.py

Crowdsourced Reporting Models
Kerala Flood Relief Coordination Agent

Database models for citizen flood observations:
    - Reports with photos
    - AI analysis results
    - Quality metrics
    - Verification workflow
    - Community voting
"""

from __future__ import annotations

import enum
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
)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(cls: type[enum.Enum]) -> SQLEnum:
    return SQLEnum(
        cls,
        values_callable=lambda members: [member.value for member in members],
        native_enum=False,
        validate_strings=True,
        length=80,
    )


# ============================================================
# ENUMERATIONS
# ============================================================

class ObservationType(str, enum.Enum):
    """Type of observation"""
    WATER_LEVEL = "water_level"
    FLOODING = "flooding"
    ROAD_BLOCKAGE = "road_blockage"
    DAMAGE = "damage"
    RESCUE_NEEDED = "rescue_needed"
    INFRASTRUCTURE = "infrastructure"
    WEATHER = "weather"
    GENERAL = "general"


class WaterLevelCategory(str, enum.Enum):
    """Water level categories"""
    DRY = "dry"
    ANKLE_DEEP = "ankle_deep"
    KNEE_DEEP = "knee_deep"
    WAIST_DEEP = "waist_deep"
    CHEST_DEEP = "chest_deep"
    OVERHEAD = "overhead"
    UNKNOWN = "unknown"


class DamageSeverity(str, enum.Enum):
    """Damage severity"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ReportStatus(str, enum.Enum):
    """Report status"""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    DUPLICATE = "duplicate"


class VerificationSource(str, enum.Enum):
    """Verification source"""
    AI_ANALYSIS = "ai_analysis"
    HUMAN_REVIEW = "human_review"
    CROSS_VALIDATION = "cross_validation"
    OFFICIAL_SOURCE = "official_source"


# ============================================================
# CROWDSOURCED REPORT MODEL
# ============================================================

class CrowdsourcedReport(Base):
    """
    Citizen-submitted flood observation report.
    
    Includes AI photo analysis, quality scoring, and verification workflow.
    """
    
    __tablename__ = "crowdsourced_reports"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    # Reporter information
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
    )

    is_anonymous: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Observation details
    observation_type: Mapped[ObservationType] = mapped_column(
        enum_type(ObservationType),
        nullable=False,
        index=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Location
    latitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    longitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    location_accuracy_m: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    district: Mapped[str | None] = mapped_column(
        String(100),
        index=True,
    )

    taluk: Mapped[str | None] = mapped_column(
        String(100),
    )

    village: Mapped[str | None] = mapped_column(
        String(100),
    )

    landmark: Mapped[str | None] = mapped_column(
        String(200),
    )

    address: Mapped[str | None] = mapped_column(
        Text,
    )

    # Status and verification
    status: Mapped[ReportStatus] = mapped_column(
        enum_type(ReportStatus),
        default=ReportStatus.PENDING,
        nullable=False,
        index=True,
    )

    verification_source: Mapped[VerificationSource | None] = mapped_column(
        enum_type(VerificationSource),
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    verified_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
    )

    rejection_reason: Mapped[str | None] = mapped_column(
        Text,
    )

    # Quality metrics
    quality_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    trust_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    timeliness_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    completeness_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    consistency_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    overall_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        index=True,
    )

    # Community engagement
    upvotes: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    downvotes: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    views: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Tags and metadata
    tags: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # Relationships
    photos: Mapped[list["CrowdsourcedPhoto"]] = relationship(
        "CrowdsourcedPhoto",
        back_populates="report",
        cascade="all, delete-orphan",
    )

    votes: Mapped[list["CrowdsourcedVote"]] = relationship(
        "CrowdsourcedVote",
        back_populates="report",
        cascade="all, delete-orphan",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "latitude >= -90 AND latitude <= 90",
            name="ck_crowdsource_lat",
        ),
        CheckConstraint(
            "longitude >= -180 AND longitude <= 180",
            name="ck_crowdsource_lon",
        ),
        CheckConstraint(
            "location_accuracy_m >= 0",
            name="ck_crowdsource_accuracy",
        ),
        CheckConstraint(
            "quality_score >= 0 AND quality_score <= 1",
            name="ck_crowdsource_quality",
        ),
        CheckConstraint(
            "overall_score >= 0 AND overall_score <= 1",
            name="ck_crowdsource_overall",
        ),
        CheckConstraint(
            "upvotes >= 0 AND downvotes >= 0",
            name="ck_crowdsource_votes",
        ),
        Index(
            "ix_crowdsource_location",
            "latitude",
            "longitude",
        ),
        Index(
            "ix_crowdsource_district_type",
            "district",
            "observation_type",
        ),
        Index(
            "ix_crowdsource_status_score",
            "status",
            "overall_score",
        ),
    )


# ============================================================
# CROWDSOURCED PHOTO MODEL
# ============================================================

class CrowdsourcedPhoto(Base):
    """
    Photo attached to a crowdsourced report with AI analysis.
    """
    
    __tablename__ = "crowdsourced_photos"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("crowdsourced_reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Photo metadata
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    storage_url: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    mime_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    width_px: Mapped[int | None] = mapped_column(
        Integer,
    )

    height_px: Mapped[int | None] = mapped_column(
        Integer,
    )

    # AI Analysis results
    analyzed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    contains_water: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    water_level_category: Mapped[WaterLevelCategory | None] = mapped_column(
        enum_type(WaterLevelCategory),
    )

    damage_severity: Mapped[DamageSeverity | None] = mapped_column(
        enum_type(DamageSeverity),
    )

    is_outdoor: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    weather_condition: Mapped[str | None] = mapped_column(
        String(50),
    )

    ai_confidence_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    # Detected objects/features
    detected_objects: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    analysis_metadata: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    # Timestamps
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    analyzed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    # Relationship
    report: Mapped["CrowdsourcedReport"] = relationship(
        "CrowdsourcedReport",
        back_populates="photos",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "size_bytes > 0",
            name="ck_photo_size",
        ),
        CheckConstraint(
            "ai_confidence_score >= 0 AND ai_confidence_score <= 1",
            name="ck_photo_confidence",
        ),
    )


# ============================================================
# CROWDSOURCED VOTE MODEL
# ============================================================

class CrowdsourcedVote(Base):
    """
    Community votes on report reliability.
    """
    
    __tablename__ = "crowdsourced_votes"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("crowdsourced_reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    is_upvote: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationship
    report: Mapped["CrowdsourcedReport"] = relationship(
        "CrowdsourcedReport",
        back_populates="votes",
    )

    # Constraints
    __table_args__ = (
        Index(
            "ix_vote_unique",
            "report_id",
            "user_id",
            unique=True,
        ),
    )


__all__ = [
    "CrowdsourcedReport",
    "CrowdsourcedPhoto",
    "CrowdsourcedVote",
    "ObservationType",
    "WaterLevelCategory",
    "DamageSeverity",
    "ReportStatus",
    "VerificationSource",
]
