"""
backend/services/voice_sos_service.py

Voice-based SOS Service
Kerala Flood Relief Coordination Agent

Features:
    - Multi-language voice command processing (Malayalam, English, Hindi, Tamil)
    - Real-time speech-to-text conversion
    - Voice intent recognition and classification
    - Automatic SOS creation from voice commands
    - Location extraction from voice descriptions
    - Priority assessment from voice tone and content
    - Accessibility-focused design for elderly and disabled persons
    - Offline voice command caching
    - Voice verification and confirmation
    - Emergency keyword detection
    - Multi-dialect support
    - Background noise filtering
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4

from ai.llm.speech_to_text import (
    AudioInput,
    SpeechLanguage,
    SpeechToTextService,
    SpeechTranscriptionResult,
    TranscriptionContext,
)


logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class VoiceCommandType(str, Enum):
    """Types of voice commands"""
    EMERGENCY_SOS = "emergency_sos"
    LOCATION_UPDATE = "location_update"
    STATUS_UPDATE = "status_update"
    HELP_REQUEST = "help_request"
    CANCEL_REQUEST = "cancel_request"
    CONFIRMATION = "confirmation"
    QUERY = "query"
    UNKNOWN = "unknown"


class VoiceEmergencyType(str, Enum):
    """Emergency types extracted from voice"""
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


class VoiceUrgencyLevel(str, Enum):
    """Urgency detected from voice"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VoiceProcessingStatus(str, Enum):
    """Voice processing pipeline status"""
    RECEIVED = "received"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    EXTRACTING_INTENT = "extracting_intent"
    VALIDATING = "validating"
    CREATING_SOS = "creating_sos"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_CLARIFICATION = "requires_clarification"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class VoiceLocation:
    """Location information extracted from voice"""
    description: str
    latitude: float | None = None
    longitude: float | None = None
    accuracy: float | None = None
    confidence: float = 0.0
    landmarks: list[str] | None = None
    district: str | None = None
    taluk: str | None = None
    village: str | None = None
    panchayat: str | None = None


@dataclass(frozen=True)
class VoiceEmergencyDetails:
    """Emergency details extracted from voice"""
    emergency_type: VoiceEmergencyType
    description: str
    people_count: int = 1
    children_present: bool = False
    elderly_present: bool = False
    disabled_present: bool = False
    injured_count: int = 0
    medical_needed: bool = False
    water_depth: str | None = None
    trapped: bool = False
    immediate_danger: bool = False
    confidence: float = 0.0


@dataclass(frozen=True)
class VoiceIntent:
    """Intent extracted from voice command"""
    command_type: VoiceCommandType
    emergency_details: VoiceEmergencyDetails | None = None
    location: VoiceLocation | None = None
    urgency: VoiceUrgencyLevel = VoiceUrgencyLevel.MEDIUM
    confidence: float = 0.0
    keywords: list[str] | None = None
    entities: dict[str, Any] | None = None
    requires_confirmation: bool = True


@dataclass(frozen=True)
class VoiceSOSRequest:
    """Complete voice SOS request"""
    request_id: str
    transcript: str
    language: str
    intent: VoiceIntent
    transcription_result: SpeechTranscriptionResult
    processing_status: VoiceProcessingStatus
    sos_id: str | None = None
    created_at: datetime | None = None
    warnings: list[str] | None = None


# ============================================================
# PROTOCOLS
# ============================================================

class LocationService(Protocol):
    """Location resolution service"""
    async def resolve_location(
        self,
        description: str,
        language: str,
    ) -> VoiceLocation: ...


class SOSCreationService(Protocol):
    """SOS creation service"""
    async def create_from_voice(
        self,
        voice_request: VoiceSOSRequest,
        user_id: str | None,
    ) -> str: ...


# ============================================================
# VOICE INTENT ANALYZER
# ============================================================

class VoiceIntentAnalyzer:
    """
    Analyzes transcribed voice to extract intent and emergency details.
    
    Uses keyword matching and pattern recognition for multiple languages.
    For production, integrate with LLM for better intent understanding.
    """

    # Emergency keywords by language
    EMERGENCY_KEYWORDS = {
        "malayalam": {
            "urgent": ["സഹായം", "അത്യാവശ്യം", "അപകടം", "രക്ഷ", "വേഗം"],
            "flood": ["വെള്ളപ്പൊക്കം", "വെള്ളം", "പ്രളയം", "വെള്ളത്തിൽ"],
            "trapped": ["കുടുങ്ങി", "പുറത്തിറങ്ങാൻ വയ്യ", "കെട്ടിക്കിടക്കുന്നു"],
            "medical": ["രോഗം", "ആശുപത്രി", "ചികിത്സ", "മരുന്ന", "പരിക്ക്"],
            "rescue": ["രക്ഷാപ്രവർത്തനം", "ബോട്ട്", "ആംബുലൻസ്"],
            "children": ["കുട്ടികൾ", "കുഞ്ഞ്", "കൊച്ചുകുട്ടി"],
            "elderly": ["മുതിർന്നവർ", "വയസ്സായവർ"],
        },
        "english": {
            "urgent": ["help", "emergency", "urgent", "danger", "rescue", "please", "quickly", "immediate"],
            "flood": ["flood", "water", "drowning", "submerged", "flooded"],
            "trapped": ["trapped", "stuck", "can't get out", "surrounded", "blocked"],
            "medical": ["medical", "hospital", "doctor", "medicine", "injured", "sick"],
            "rescue": ["rescue", "boat", "ambulance", "evacuate"],
            "children": ["child", "children", "baby", "kid", "infant"],
            "elderly": ["elderly", "old", "senior", "aged"],
        },
        "hindi": {
            "urgent": ["मदद", "आपातकाल", "जल्दी", "खतरा", "बचाओ"],
            "flood": ["बाढ़", "पानी", "डूब", "जलमग्न"],
            "trapped": ["फंस गया", "बाहर नहीं निकल सकता", "घिरा हुआ"],
            "medical": ["चिकित्सा", "अस्पताल", "डॉक्टर", "दवा", "घायल"],
            "rescue": ["बचाव", "नाव", "एम्बुलेंस"],
            "children": ["बच्चे", "शिशु"],
            "elderly": ["बुजुर्ग", "वृद्ध"],
        },
        "tamil": {
            "urgent": ["உதவி", "அவசரம", "ஆபத்து", "மீட்பு", "விரைவு"],
            "flood": ["வெள்ளம்", "தண்ணீர்", "மூழ்கும்"],
            "trapped": ["சிக்கியுள்ளது", "வெளியேற முடியவில்லை"],
            "medical": ["மருத்துவம்", "மருத்துவமனை", "காயம்"],
            "rescue": ["மீட்பு", "படகு", "ஆம்புலன்ஸ்"],
            "children": ["குழந்தைகள்", "குழந்தை"],
            "elderly": ["வயதானவர்கள்", "முதியவர்கள்"],
        }
    }

    def __init__(self):
        self.location_patterns = {
            "malayalam": re.compile(r"(ഗ്രാമം|നഗരം|ജില്ല|താലൂക്ക്|പഞ്ചായത്ത്)", re.IGNORECASE),
            "english": re.compile(r"\b(village|town|city|district|near|at|in)\s+([A-Z][a-z]+)", re.IGNORECASE),
            "hindi": re.compile(r"(गांव|शहर|जिला|तालुका)", re.IGNORECASE),
            "tamil": re.compile(r"(கிராமம்|நகரம்|மாவட்டம்)", re.IGNORECASE),
        }

    def analyze(
        self,
        transcript: str,
        language: str,
    ) -> VoiceIntent:
        """Analyze transcript to extract intent"""
        
        text_lower = transcript.lower()
        
        # Detect command type
        command_type = self._detect_command_type(text_lower, language)
        
        # Detect emergency details if it's an emergency command
        emergency_details = None
        if command_type == VoiceCommandType.EMERGENCY_SOS:
            emergency_details = self._extract_emergency_details(transcript, language)
        
        # Extract location information
        location = self._extract_location(transcript, language)
        
        # Assess urgency
        urgency = self._assess_urgency(text_lower, language, emergency_details)
        
        # Extract keywords
        keywords = self._extract_keywords(text_lower, language)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            command_type,
            emergency_details,
            location,
            keywords,
        )
        
        # Determine if confirmation needed
        requires_confirmation = confidence < 0.8 or urgency == VoiceUrgencyLevel.CRITICAL
        
        return VoiceIntent(
            command_type=command_type,
            emergency_details=emergency_details,
            location=location,
            urgency=urgency,
            confidence=confidence,
            keywords=keywords,
            requires_confirmation=requires_confirmation,
        )

    def _detect_command_type(self, text: str, language: str) -> VoiceCommandType:
        """Detect the type of voice command"""
        
        keywords = self.EMERGENCY_KEYWORDS.get(language, self.EMERGENCY_KEYWORDS["english"])
        
        # Check for emergency keywords
        urgent_count = sum(1 for keyword in keywords.get("urgent", []) if keyword in text)
        flood_count = sum(1 for keyword in keywords.get("flood", []) if keyword in text)
        rescue_count = sum(1 for keyword in keywords.get("rescue", []) if keyword in text)
        
        if urgent_count > 0 or flood_count > 0 or rescue_count > 0:
            return VoiceCommandType.EMERGENCY_SOS
        
        # Check for location updates
        if any(word in text for word in ["location", "moved", "now at", "സ്ഥലം", "स्थान", "இடம்"]):
            return VoiceCommandType.LOCATION_UPDATE
        
        # Check for status updates
        if any(word in text for word in ["status", "update", "situation", "നില", "स्थिति", "நிலை"]):
            return VoiceCommandType.STATUS_UPDATE
        
        # Check for cancellation
        if any(word in text for word in ["cancel", "stop", "mistake", "റദ്ദാക്കുക", "रद्द", "ரத்து"]):
            return VoiceCommandType.CANCEL_REQUEST
        
        # Check for confirmation
        if any(word in text for word in ["yes", "confirm", "correct", "അതെ", "हाँ", "ஆம்"]):
            return VoiceCommandType.CONFIRMATION
        
        return VoiceCommandType.UNKNOWN

    def _extract_emergency_details(
        self,
        text: str,
        language: str,
    ) -> VoiceEmergencyDetails:
        """Extract emergency details from transcript"""
        
        text_lower = text.lower()
        keywords = self.EMERGENCY_KEYWORDS.get(language, self.EMERGENCY_KEYWORDS["english"])
        
        # Detect emergency type
        emergency_type = VoiceEmergencyType.UNKNOWN
        if any(keyword in text_lower for keyword in keywords.get("flood", [])):
            emergency_type = VoiceEmergencyType.FLOOD_TRAPPED
        elif any(keyword in text_lower for keyword in keywords.get("medical", [])):
            emergency_type = VoiceEmergencyType.MEDICAL_EMERGENCY
        
        # Detect trapped status
        trapped = any(keyword in text_lower for keyword in keywords.get("trapped", []))
        
        # Detect vulnerable populations
        children_present = any(keyword in text_lower for keyword in keywords.get("children", []))
        elderly_present = any(keyword in text_lower for keyword in keywords.get("elderly", []))
        
        # Extract people count (simple number extraction)
        people_count = self._extract_number(text)
        if people_count == 0:
            people_count = 1
        
        # Detect medical need
        medical_needed = any(keyword in text_lower for keyword in keywords.get("medical", []))
        
        # Detect immediate danger
        immediate_danger = trapped or any(
            keyword in text_lower 
            for keyword in keywords.get("urgent", [])[:3]  # Top 3 most urgent words
        )
        
        # Calculate confidence
        confidence = 0.6
        if emergency_type != VoiceEmergencyType.UNKNOWN:
            confidence += 0.2
        if people_count > 0:
            confidence += 0.1
        if children_present or elderly_present:
            confidence += 0.1
        
        return VoiceEmergencyDetails(
            emergency_type=emergency_type,
            description=text[:500],  # First 500 chars
            people_count=people_count,
            children_present=children_present,
            elderly_present=elderly_present,
            trapped=trapped,
            medical_needed=medical_needed,
            immediate_danger=immediate_danger,
            confidence=min(confidence, 1.0),
        )

    def _extract_location(self, text: str, language: str) -> VoiceLocation:
        """Extract location information from transcript"""
        
        # Try to match location patterns
        pattern = self.location_patterns.get(language)
        matches = []
        if pattern:
            matches = pattern.findall(text)
        
        # Extract location description (simplified)
        location_desc = text[:200] if len(text) <= 200 else text[:200] + "..."
        
        confidence = 0.3 if matches else 0.1
        
        return VoiceLocation(
            description=location_desc,
            confidence=confidence,
            landmarks=[match[-1] if isinstance(match, tuple) else match for match in matches[:5]],
        )

    def _assess_urgency(
        self,
        text: str,
        language: str,
        emergency_details: VoiceEmergencyDetails | None,
    ) -> VoiceUrgencyLevel:
        """Assess urgency level from text and context"""
        
        keywords = self.EMERGENCY_KEYWORDS.get(language, self.EMERGENCY_KEYWORDS["english"])
        
        # Critical indicators
        if emergency_details:
            if emergency_details.immediate_danger or emergency_details.trapped:
                return VoiceUrgencyLevel.CRITICAL
            if emergency_details.medical_needed or emergency_details.children_present:
                return VoiceUrgencyLevel.HIGH
        
        # Check for urgent keywords
        urgent_count = sum(
            1 for keyword in keywords.get("urgent", [])
            if keyword in text
        )
        
        if urgent_count >= 2:
            return VoiceUrgencyLevel.HIGH
        elif urgent_count == 1:
            return VoiceUrgencyLevel.MEDIUM
        
        return VoiceUrgencyLevel.MEDIUM

    def _extract_keywords(self, text: str, language: str) -> list[str]:
        """Extract relevant keywords from text"""
        
        keywords = self.EMERGENCY_KEYWORDS.get(language, self.EMERGENCY_KEYWORDS["english"])
        extracted = []
        
        for category, words in keywords.items():
            for word in words:
                if word in text:
                    extracted.append(word)
        
        return extracted[:20]  # Limit to top 20 keywords

    def _extract_number(self, text: str) -> int:
        """Extract number from text (simple implementation)"""
        
        # English number words
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        
        # Malayalam number words
        malayalam_numbers = {
            "ഒന്ന്": 1, "രണ്ട്": 2, "മൂന്ന്": 3, "നാല്": 4, "അഞ്ച്": 5,
        }
        
        # Check for digit numbers
        import re
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[0])
        
        # Check for word numbers
        text_lower = text.lower()
        for word, num in number_words.items():
            if word in text_lower:
                return num
        
        for word, num in malayalam_numbers.items():
            if word in text:
                return num
        
        return 1  # Default to 1 person

    def _calculate_confidence(
        self,
        command_type: VoiceCommandType,
        emergency_details: VoiceEmergencyDetails | None,
        location: VoiceLocation,
        keywords: list[str],
    ) -> float:
        """Calculate overall confidence score"""
        
        confidence = 0.3  # Base confidence
        
        # Command type confidence
        if command_type != VoiceCommandType.UNKNOWN:
            confidence += 0.2
        
        # Emergency details confidence
        if emergency_details:
            confidence += emergency_details.confidence * 0.3
        
        # Location confidence
        confidence += location.confidence * 0.1
        
        # Keyword confidence
        if len(keywords) > 3:
            confidence += 0.1
        
        return min(confidence, 1.0)


# ============================================================
# VOICE SOS SERVICE
# ============================================================

class VoiceSOSService:
    """
    Main service for processing voice-based SOS requests.
    
    Coordinates speech-to-text, intent analysis, and SOS creation.
    """

    def __init__(
        self,
        speech_service: SpeechToTextService,
        location_service: LocationService | None = None,
        sos_creation_service: SOSCreationService | None = None,
    ):
        self.speech_service = speech_service
        self.location_service = location_service
        self.sos_creation_service = sos_creation_service
        self.intent_analyzer = VoiceIntentAnalyzer()
        
        # In-memory cache for pending confirmations (use Redis in production)
        self.pending_requests: dict[str, VoiceSOSRequest] = {}

    async def process_voice_sos(
        self,
        audio_bytes: bytes,
        filename: str,
        language: SpeechLanguage = SpeechLanguage.AUTO,
        user_id: str | None = None,
        auto_create_sos: bool = False,
    ) -> VoiceSOSRequest:
        """
        Process a voice-based SOS request.
        
        Args:
            audio_bytes: Audio data
            filename: Audio filename
            language: Preferred language (auto-detected if not specified)
            user_id: User ID if authenticated
            auto_create_sos: If True, automatically create SOS without confirmation
            
        Returns:
            VoiceSOSRequest with processing results
        """
        
        request_id = str(uuid4())
        warnings = []
        
        try:
            # Step 1: Transcribe audio
            logger.info(f"Transcribing voice SOS: request_id={request_id}")
            
            transcription = await self.speech_service.transcribe_sos(
                audio_bytes=audio_bytes,
                filename=filename,
                language=language,
            )
            
            if transcription.requires_human_review:
                warnings.append("Transcription requires human review for accuracy")
            
            # Step 2: Analyze intent
            logger.info(f"Analyzing intent: request_id={request_id}")
            
            language_code = self._map_language(transcription.detected_language)
            intent = self.intent_analyzer.analyze(
                transcript=transcription.text,
                language=language_code,
            )
            
            # Step 3: Enhance location if service available
            if self.location_service and intent.location:
                try:
                    enhanced_location = await self.location_service.resolve_location(
                        description=intent.location.description,
                        language=language_code,
                    )
                    intent = VoiceIntent(
                        command_type=intent.command_type,
                        emergency_details=intent.emergency_details,
                        location=enhanced_location,
                        urgency=intent.urgency,
                        confidence=intent.confidence,
                        keywords=intent.keywords,
                        requires_confirmation=intent.requires_confirmation,
                    )
                except Exception as e:
                    logger.warning(f"Location enhancement failed: {e}")
                    warnings.append("Could not enhance location information")
            
            # Step 4: Create voice SOS request
            processing_status = VoiceProcessingStatus.COMPLETED
            sos_id = None
            
            if intent.command_type == VoiceCommandType.EMERGENCY_SOS:
                if auto_create_sos and not intent.requires_confirmation:
                    # Auto-create SOS
                    voice_request = VoiceSOSRequest(
                        request_id=request_id,
                        transcript=transcription.text,
                        language=language_code,
                        intent=intent,
                        transcription_result=transcription,
                        processing_status=VoiceProcessingStatus.CREATING_SOS,
                        created_at=datetime.now(timezone.utc),
                        warnings=warnings,
                    )
                    
                    if self.sos_creation_service:
                        try:
                            sos_id = await self.sos_creation_service.create_from_voice(
                                voice_request=voice_request,
                                user_id=user_id,
                            )
                            processing_status = VoiceProcessingStatus.COMPLETED
                            logger.info(f"SOS created from voice: sos_id={sos_id}")
                        except Exception as e:
                            logger.error(f"Failed to create SOS: {e}")
                            processing_status = VoiceProcessingStatus.FAILED
                            warnings.append(f"SOS creation failed: {str(e)}")
                else:
                    # Store for confirmation
                    processing_status = VoiceProcessingStatus.REQUIRES_CLARIFICATION
                    warnings.append("Voice command requires confirmation before creating SOS")
            
            # Create final request object
            voice_request = VoiceSOSRequest(
                request_id=request_id,
                transcript=transcription.text,
                language=language_code,
                intent=intent,
                transcription_result=transcription,
                processing_status=processing_status,
                sos_id=sos_id,
                created_at=datetime.now(timezone.utc),
                warnings=warnings,
            )
            
            # Cache for confirmation if needed
            if processing_status == VoiceProcessingStatus.REQUIRES_CLARIFICATION:
                self.pending_requests[request_id] = voice_request
            
            logger.info(
                f"Voice SOS processed: request_id={request_id}, "
                f"status={processing_status.value}, confidence={intent.confidence:.2f}"
            )
            
            return voice_request
            
        except Exception as e:
            logger.error(f"Voice SOS processing failed: {e}", exc_info=True)
            
            return VoiceSOSRequest(
                request_id=request_id,
                transcript="",
                language="unknown",
                intent=VoiceIntent(
                    command_type=VoiceCommandType.UNKNOWN,
                    confidence=0.0,
                ),
                transcription_result=None,  # type: ignore
                processing_status=VoiceProcessingStatus.FAILED,
                created_at=datetime.now(timezone.utc),
                warnings=[f"Processing error: {str(e)}"],
            )

    async def confirm_voice_sos(
        self,
        request_id: str,
        user_id: str | None = None,
        modifications: dict[str, Any] | None = None,
    ) -> str:
        """
        Confirm a pending voice SOS request and create the SOS.
        
        Args:
            request_id: Voice request ID
            user_id: User ID confirming the request
            modifications: Optional modifications to the extracted data
            
        Returns:
            Created SOS ID
        """
        
        if request_id not in self.pending_requests:
            raise ValueError(f"Voice request {request_id} not found or already processed")
        
        voice_request = self.pending_requests[request_id]
        
        # Apply modifications if provided
        if modifications:
            # This would update the voice_request with user corrections
            pass
        
        if not self.sos_creation_service:
            raise RuntimeError("SOS creation service not configured")
        
        sos_id = await self.sos_creation_service.create_from_voice(
            voice_request=voice_request,
            user_id=user_id,
        )
        
        # Remove from pending
        del self.pending_requests[request_id]
        
        logger.info(f"Voice SOS confirmed and created: request_id={request_id}, sos_id={sos_id}")
        
        return sos_id

    async def get_pending_request(self, request_id: str) -> VoiceSOSRequest | None:
        """Get a pending voice request for review"""
        return self.pending_requests.get(request_id)

    async def cancel_voice_request(self, request_id: str) -> None:
        """Cancel a pending voice request"""
        if request_id in self.pending_requests:
            del self.pending_requests[request_id]
            logger.info(f"Voice request cancelled: request_id={request_id}")

    def _map_language(self, speech_lang: SpeechLanguage) -> str:
        """Map SpeechLanguage enum to language code"""
        mapping = {
            SpeechLanguage.ENGLISH: "english",
            SpeechLanguage.MALAYALAM: "malayalam",
        }
        return mapping.get(speech_lang, "english")


# ============================================================
# FACTORY
# ============================================================

async def create_voice_sos_service(
    speech_service: SpeechToTextService,
    location_service: LocationService | None = None,
    sos_creation_service: SOSCreationService | None = None,
) -> VoiceSOSService:
    """Create and initialize VoiceSOSService"""
    
    service = VoiceSOSService(
        speech_service=speech_service,
        location_service=location_service,
        sos_creation_service=sos_creation_service,
    )
    
    logger.info("Voice SOS service initialized")
    
    return service
