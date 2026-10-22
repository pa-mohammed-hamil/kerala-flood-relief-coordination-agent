"""
backend/models/voice_sos.py

Voice SOS Request Models
Kerala Flood Relief Coordination Agent

Database models for voice-based emergency SOS requests:
    - Voice recording metadata
    - Transcription results
    - Intent analysis results
    - Confidence scores
    - Language detection
    - Processing status tracking
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

class VoiceLanguage(str, enum.Enum):
    """Language of voice recording"""
    AUTO = "auto"
    MALAYALAM = "malayalam"
    ENGLISH = "english"
    HINDI = "hindi"
    TAMIL = "tamil"
    UNKNOWN = "unknown"


class VoiceCommandType(str, enum.Enum):
    """Type of voice command"""
    EMERGENCY_SOS = "emergency_sos"
    LOCATION_UPDATE = "location_update"
    STATUS_UPDATE = "status_update"
    HELP_REQUEST = "help_request"
    CANCEL_REQUEST = "cancel_request"
    CONFIRMATION = "confirmation"
    QUERY = "query"
    UNKNOWN = "unknown"


class VoiceEmergencyType(str, enum.Enum):
    """Emergency type from voice"""
    FLOOD_TRAPPED = "flood_trapped"
    WATER_RESCUE = "water_rescue"
    MEDICAL_EMERGENCY = "medical_emergency"
    BUILDING_COLLAPSE = "building_collapse"
    LANDSLIDE = "landslide"
    STRANDED = "stranded"
    FOOD_WATER = "food_water"
    SHELTER_NEEDED = "shelter_needed"
    EVACUATION_HELP = "evacuation_help"
    UNKNOWN = "unknown"


class VoiceUrgencyLevel(str, enum.Enum):
    """Urgency level from voice analysis"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VoiceProcessingStatus(str, enum.Enum):
    """Voice processing status"""
    RECEIVED = "received"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    EXTRACTING_INTENT = "extracting_intent"
    VALIDATING = "validating"
    CREATING_SOS = "creating_sos"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_CLARIFICATION = "requires_clarification"
    CANCELLED = "cancelled"


class TranscriptionConfidence(str, enum.Enum):
    """Transcription quality confidence"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# ============================================================
# VOICE SOS REQUEST MODEL
# ============================================================

class VoiceSOSRequest(Base):
    """
    Voice-based SOS request with transcription and intent analysis.
    
    Tracks the complete lifecycle from audio recording through
    transcription, intent extraction, and SOS creation.
    """
    
    __tablename__ = "voice_sos_requests"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    # User information
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
    )

    # Audio metadata
    audio_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    audio_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    audio_duration_seconds: Mapped[float | None] = mapped_column(
        Float,
    )

    audio_format: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    audio_sha256: Mapped[str | None] = mapped_column(
        String(64),
        index=True,
    )

    # Audio storage reference (S3, local path, etc.)
    audio_storage_reference: Mapped[str | None] = mapped_column(
        String(500),
    )

    # Language
    requested_language: Mapped[VoiceLanguage] = mapped_column(
        enum_type(VoiceLanguage),
        default=VoiceLanguage.AUTO,
        nullable=False,
    )

    detected_language: Mapped[VoiceLanguage] = mapped_column(
        enum_type(VoiceLanguage),
        default=VoiceLanguage.UNKNOWN,
        nullable=False,
    )

    # Transcription
    transcript: Mapped[str | None] = mapped_column(
        Text,
    )

    transcription_confidence: Mapped[TranscriptionConfidence] = mapped_column(
        enum_type(TranscriptionConfidence),
        default=TranscriptionConfidence.UNKNOWN,
        nullable=False,
    )

    requires_human_review: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Intent analysis
    command_type: Mapped[VoiceCommandType] = mapped_column(
        enum_type(VoiceCommandType),
        default=VoiceCommandType.UNKNOWN,
        nullable=False,
    )

    emergency_type: Mapped[VoiceEmergencyType | None] = mapped_column(
        enum_type(VoiceEmergencyType),
    )

    urgency_level: Mapped[VoiceUrgencyLevel] = mapped_column(
        enum_type(VoiceUrgencyLevel),
        default=VoiceUrgencyLevel.MEDIUM,
        nullable=False,
    )

    intent_confidence: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    # Extracted details
    people_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    children_present: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    elderly_present: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    disabled_present: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    medical_needed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    trapped: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    immediate_danger: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Location information from voice
    location_description: Mapped[str | None] = mapped_column(
        Text,
    )

    location_latitude: Mapped[float | None] = mapped_column(
        Float,
    )

    location_longitude: Mapped[float | None] = mapped_column(
        Float,
    )

    location_accuracy_m: Mapped[float | None] = mapped_column(
        Float,
    )

    location_confidence: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    # Processing status
    processing_status: Mapped[VoiceProcessingStatus] = mapped_column(
        enum_type(VoiceProcessingStatus),
        default=VoiceProcessingStatus.RECEIVED,
        nullable=False,
        index=True,
    )

    # Related SOS request
    sos_request_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("sos_requests.id", ondelete="SET NULL"),
        index=True,
    )

    # Confirmation
    requires_confirmation: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    confirmed_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
    )

    # Metadata
    keywords: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    entities: Mapped[dict] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    warnings: Mapped[dict] = mapped_column(
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

    transcribed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    analyzed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "audio_size_bytes > 0",
            name="ck_voice_sos_audio_size",
        ),
        CheckConstraint(
            "audio_duration_seconds IS NULL OR audio_duration_seconds > 0",
            name="ck_voice_sos_duration",
        ),
        CheckConstraint(
            "people_count >= 1",
            name="ck_voice_sos_people",
        ),
        CheckConstraint(
            "intent_confidence >= 0 AND intent_confidence <= 1",
            name="ck_voice_sos_confidence",
        ),
        CheckConstraint(
            "location_confidence >= 0 AND location_confidence <= 1",
            name="ck_voice_sos_location_conf",
        ),
        CheckConstraint(
            "location_latitude IS NULL OR (location_latitude >= -90 AND location_latitude <= 90)",
            name="ck_voice_sos_lat",
        ),
        CheckConstraint(
            "location_longitude IS NULL OR (location_longitude >= -180 AND location_longitude <= 180)",
            name="ck_voice_sos_lon",
        ),
        Index(
            "ix_voice_sos_status_created",
            "processing_status",
            "created_at",
        ),
        Index(
            "ix_voice_sos_user_created",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_voice_sos_language",
            "detected_language",
            "created_at",
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "audio_filename": self.audio_filename,
            "audio_duration_seconds": self.audio_duration_seconds,
            "requested_language": self.requested_language.value,
            "detected_language": self.detected_language.value,
            "transcript": self.transcript,
            "transcription_confidence": self.transcription_confidence.value,
            "command_type": self.command_type.value,
            "emergency_type": self.emergency_type.value if self.emergency_type else None,
            "urgency_level": self.urgency_level.value,
            "intent_confidence": self.intent_confidence,
            "people_count": self.people_count,
            "processing_status": self.processing_status.value,
            "sos_request_id": self.sos_request_id,
            "requires_confirmation": self.requires_confirmation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# VOICE TRANSCRIPTION SEGMENT
# ============================================================

class VoiceTranscriptionSegment(Base):
    """
    Time-aligned transcription segments with word timestamps.
    
    Enables replay and detailed analysis of voice content.
    """
    
    __tablename__ = "voice_transcription_segments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    voice_sos_request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("voice_sos_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    segment_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    start_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    end_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "start_seconds >= 0",
            name="ck_voice_segment_start",
        ),
        CheckConstraint(
            "end_seconds >= start_seconds",
            name="ck_voice_segment_end",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_voice_segment_confidence",
        ),
        Index(
            "ix_voice_segment_request_index",
            "voice_sos_request_id",
            "segment_index",
        ),
    )


__all__ = [
    "VoiceSOSRequest",
    "VoiceTranscriptionSegment",
    "VoiceLanguage",
    "VoiceCommandType",
    "VoiceEmergencyType",
    "VoiceUrgencyLevel",
    "VoiceProcessingStatus",
    "TranscriptionConfidence",
]
