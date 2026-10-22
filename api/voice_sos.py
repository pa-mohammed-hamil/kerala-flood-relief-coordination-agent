"""
backend/api/voice_sos.py

Voice-based SOS API
Kerala Flood Relief Coordination Agent

RESTful API endpoints for voice command processing:
    - Multi-language voice SOS submission
    - Real-time transcription status
    - Voice intent confirmation
    - Accessibility-focused voice interface
    
Features:
    - Audio file upload (WAV, MP3, M4A, WEBM, OGG)
    - Streaming audio support
    - Language auto-detection
    - Voice command history
    - Emergency priority handling
"""

from __future__ import annotations

import io
from datetime import datetime
from enum import Enum
from typing import Annotated, Any
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from backend.dependencies import get_current_user
from backend.services.voice_sos_service import (
    VoiceCommandType,
    VoiceEmergencyType,
    VoiceProcessingStatus,
    VoiceSOSRequest,
    VoiceSOSService,
    VoiceUrgencyLevel,
)


router = APIRouter(
    prefix="/voice-sos",
    tags=["Voice SOS"],
)

CurrentUser = Annotated[Any, Depends(get_current_user)]


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class VoiceLanguage(str, Enum):
    """Supported voice languages"""
    AUTO = "auto"
    ENGLISH = "en"
    MALAYALAM = "ml"
    HINDI = "hi"
    TAMIL = "ta"


class VoiceSOSSubmit(BaseModel):
    """Voice SOS submission metadata"""
    model_config = ConfigDict(extra="forbid")
    
    language: VoiceLanguage = VoiceLanguage.AUTO
    auto_create: bool = Field(
        default=False,
        description="Auto-create SOS without confirmation for high-confidence requests"
    )
    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        description="Optional GPS latitude"
    )
    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        description="Optional GPS longitude"
    )
    accuracy_m: float | None = Field(
        default=None,
        ge=0,
        description="GPS accuracy in meters"
    )


class VoiceSOSConfirmation(BaseModel):
    """Confirmation of voice SOS with optional corrections"""
    model_config = ConfigDict(extra="forbid")
    
    confirmed: bool
    corrections: dict[str, Any] | None = Field(
        default=None,
        description="User corrections to extracted data"
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class VoiceLocationResponse(BaseModel):
    """Location information from voice"""
    description: str
    latitude: float | None = None
    longitude: float | None = None
    accuracy: float | None = None
    confidence: float
    landmarks: list[str] | None = None
    district: str | None = None


class VoiceEmergencyResponse(BaseModel):
    """Emergency details from voice"""
    emergency_type: VoiceEmergencyType
    description: str
    people_count: int
    children_present: bool
    elderly_present: bool
    disabled_present: bool
    medical_needed: bool
    trapped: bool
    immediate_danger: bool
    confidence: float


class VoiceIntentResponse(BaseModel):
    """Voice intent analysis result"""
    command_type: VoiceCommandType
    emergency_details: VoiceEmergencyResponse | None = None
    location: VoiceLocationResponse | None = None
    urgency: VoiceUrgencyLevel
    confidence: float
    keywords: list[str] | None = None
    requires_confirmation: bool


class VoiceSOSResponse(BaseModel):
    """Voice SOS request response"""
    request_id: str
    transcript: str
    language: str
    intent: VoiceIntentResponse
    processing_status: VoiceProcessingStatus
    sos_id: str | None = None
    created_at: datetime
    warnings: list[str] | None = None
    
    # Transcription metadata
    transcription_confidence: str | None = None
    requires_human_review: bool = False


class VoiceSOSHealthResponse(BaseModel):
    """Service health status"""
    service: str = "voice_sos"
    status: str = "healthy"
    supported_languages: list[str] = ["malayalam", "english", "hindi", "tamil"]
    max_audio_size_mb: int = 25
    supported_formats: list[str] = ["wav", "mp3", "m4a", "webm", "ogg", "flac"]
    timestamp: datetime


class VoiceSOSStatistics(BaseModel):
    """Voice SOS usage statistics"""
    total_requests: int
    by_language: dict[str, int]
    by_status: dict[str, int]
    by_urgency: dict[str, int]
    average_confidence: float
    generated_at: datetime


# ============================================================
# SERVICE DEPENDENCY
# ============================================================

def get_voice_sos_service(request: Request) -> VoiceSOSService:
    """Get voice SOS service from app state"""
    service = getattr(request.app.state, "voice_sos_service", None)
    
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Voice SOS service is not configured",
        )
    
    return service


VoiceSOSServiceDep = Annotated[VoiceSOSService, Depends(get_voice_sos_service)]


# ============================================================
# AUTHORIZATION HELPERS
# ============================================================

def get_user_id(user: Any) -> str | None:
    """Extract user ID from authenticated user"""
    if user is None:
        return None
    
    user_id = getattr(user, "id", None)
    return str(user_id) if user_id is not None else None


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/health", response_model=VoiceSOSHealthResponse)
async def voice_sos_health() -> VoiceSOSHealthResponse:
    """
    Voice SOS service health check.
    
    Returns service status and capabilities.
    """
    return VoiceSOSHealthResponse(
        timestamp=datetime.utcnow(),
    )


@router.post(
    "/submit",
    response_model=VoiceSOSResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_voice_sos(
    audio: UploadFile = File(
        ...,
        description="Audio file containing voice SOS (WAV, MP3, M4A, WEBM, OGG, FLAC)"
    ),
    language: VoiceLanguage = Form(VoiceLanguage.AUTO),
    auto_create: bool = Form(False),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    accuracy_m: float | None = Form(None),
    current_user: CurrentUser = Depends(get_current_user),
    service: VoiceSOSServiceDep = Depends(get_voice_sos_service),
) -> VoiceSOSResponse:
    """
    Submit a voice-based SOS request.
    
    Supports multiple languages with auto-detection:
    - Malayalam (ml)
    - English (en)
    - Hindi (hi)
    - Tamil (ta)
    
    The system will:
    1. Transcribe the audio to text
    2. Analyze the content to extract emergency details
    3. Extract location information
    4. Assess urgency level
    5. Optionally create an SOS request automatically
    
    For critical emergencies with high confidence, an SOS will be created
    immediately. Otherwise, the request will be held for confirmation.
    
    Maximum audio size: 25 MB
    Maximum duration: 30 minutes
    """
    
    # Validate audio file
    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio filename is required",
        )
    
    # Check file extension
    allowed_extensions = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}
    file_ext = audio.filename.lower().split(".")[-1]
    if f".{file_ext}" not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_ext}. "
                   f"Supported formats: {', '.join(allowed_extensions)}",
        )
    
    # Read audio data
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read audio file: {str(e)}",
        )
    
    # Check size
    if len(audio_bytes) > 25 * 1024 * 1024:  # 25 MB
        raise HTTPException(
            status_code=413,
            detail="Audio file too large. Maximum size is 25 MB.",
        )
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty",
        )
    
    # Map language enum to SpeechLanguage
    from ai.llm.speech_to_text import SpeechLanguage
    
    language_map = {
        VoiceLanguage.AUTO: SpeechLanguage.AUTO,
        VoiceLanguage.ENGLISH: SpeechLanguage.ENGLISH,
        VoiceLanguage.MALAYALAM: SpeechLanguage.MALAYALAM,
        VoiceLanguage.HINDI: SpeechLanguage.ENGLISH,  # Fallback to English for now
        VoiceLanguage.TAMIL: SpeechLanguage.ENGLISH,  # Fallback to English for now
    }
    
    speech_language = language_map.get(language, SpeechLanguage.AUTO)
    user_id = get_user_id(current_user)
    
    # Process voice SOS
    try:
        voice_request = await service.process_voice_sos(
            audio_bytes=audio_bytes,
            filename=audio.filename,
            language=speech_language,
            user_id=user_id,
            auto_create_sos=auto_create,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process voice SOS: {str(e)}",
        )
    
    # Convert to response
    return _to_response(voice_request)


@router.get(
    "/requests/{request_id}",
    response_model=VoiceSOSResponse,
)
async def get_voice_request(
    request_id: str,
    current_user: CurrentUser,
    service: VoiceSOSServiceDep,
) -> VoiceSOSResponse:
    """
    Get details of a voice SOS request.
    
    Retrieves the transcription, intent analysis, and processing status
    of a previously submitted voice request.
    """
    
    voice_request = await service.get_pending_request(request_id)
    
    if voice_request is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice request {request_id} not found or already processed",
        )
    
    return _to_response(voice_request)


@router.post(
    "/requests/{request_id}/confirm",
    response_model=dict[str, str],
)
async def confirm_voice_request(
    request_id: str,
    confirmation: VoiceSOSConfirmation,
    current_user: CurrentUser,
    service: VoiceSOSServiceDep,
) -> dict[str, str]:
    """
    Confirm a pending voice SOS request.
    
    After reviewing the transcription and extracted emergency details,
    confirm to create an actual SOS request in the system.
    
    Optionally provide corrections to the extracted data.
    """
    
    if not confirmation.confirmed:
        # Cancel the request
        await service.cancel_voice_request(request_id)
        return {
            "status": "cancelled",
            "message": "Voice SOS request cancelled",
        }
    
    user_id = get_user_id(current_user)
    
    try:
        sos_id = await service.confirm_voice_sos(
            request_id=request_id,
            user_id=user_id,
            modifications=confirmation.corrections,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create SOS: {str(e)}",
        )
    
    return {
        "status": "confirmed",
        "sos_id": sos_id,
        "message": "SOS request created successfully",
    }


@router.delete(
    "/requests/{request_id}",
    response_model=dict[str, str],
)
async def cancel_voice_request(
    request_id: str,
    current_user: CurrentUser,
    service: VoiceSOSServiceDep,
) -> dict[str, str]:
    """
    Cancel a pending voice SOS request.
    
    Removes the request from the pending queue without creating an SOS.
    """
    
    await service.cancel_voice_request(request_id)
    
    return {
        "status": "cancelled",
        "request_id": request_id,
        "message": "Voice request cancelled successfully",
    }


# ============================================================
# RESPONSE CONVERTERS
# ============================================================

def _to_response(voice_request: VoiceSOSRequest) -> VoiceSOSResponse:
    """Convert VoiceSOSRequest to API response"""
    
    # Convert emergency details
    emergency_response = None
    if voice_request.intent.emergency_details:
        ed = voice_request.intent.emergency_details
        emergency_response = VoiceEmergencyResponse(
            emergency_type=ed.emergency_type,
            description=ed.description,
            people_count=ed.people_count,
            children_present=ed.children_present,
            elderly_present=ed.elderly_present,
            disabled_present=ed.disabled_present,
            medical_needed=ed.medical_needed,
            trapped=ed.trapped,
            immediate_danger=ed.immediate_danger,
            confidence=ed.confidence,
        )
    
    # Convert location
    location_response = None
    if voice_request.intent.location:
        loc = voice_request.intent.location
        location_response = VoiceLocationResponse(
            description=loc.description,
            latitude=loc.latitude,
            longitude=loc.longitude,
            accuracy=loc.accuracy,
            confidence=loc.confidence,
            landmarks=loc.landmarks,
            district=loc.district,
        )
    
    # Convert intent
    intent_response = VoiceIntentResponse(
        command_type=voice_request.intent.command_type,
        emergency_details=emergency_response,
        location=location_response,
        urgency=voice_request.intent.urgency,
        confidence=voice_request.intent.confidence,
        keywords=voice_request.intent.keywords,
        requires_confirmation=voice_request.intent.requires_confirmation,
    )
    
    # Get transcription metadata
    transcription_confidence = None
    requires_review = False
    if voice_request.transcription_result:
        transcription_confidence = voice_request.transcription_result.quality.confidence_level.value
        requires_review = voice_request.transcription_result.requires_human_review
    
    return VoiceSOSResponse(
        request_id=voice_request.request_id,
        transcript=voice_request.transcript,
        language=voice_request.language,
        intent=intent_response,
        processing_status=voice_request.processing_status,
        sos_id=voice_request.sos_id,
        created_at=voice_request.created_at or datetime.utcnow(),
        warnings=voice_request.warnings,
        transcription_confidence=transcription_confidence,
        requires_human_review=requires_review,
    )
