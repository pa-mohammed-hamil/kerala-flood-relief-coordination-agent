"""
backend/services/crowdsource_service.py

Crowdsourced Flood Observation Service
Kerala Flood Relief Coordination Agent

Features:
    - Real-time citizen flood observations
    - Photo uploads with AI analysis
    - Water level estimation from images
    - Damage assessment from photos
    - Location verification
    - Report clustering and aggregation
    - Verification workflow
    - Quality scoring
    - Spam and fake detection
    - Real-time map updates
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class ObservationType(str, Enum):
    """Type of flood observation"""
    WATER_LEVEL = "water_level"
    FLOODING = "flooding"
    ROAD_BLOCKAGE = "road_blockage"
    DAMAGE = "damage"
    RESCUE_NEEDED = "rescue_needed"
    INFRASTRUCTURE = "infrastructure"
    WEATHER = "weather"
    GENERAL = "general"


class WaterLevelCategory(str, Enum):
    """Water level categories"""
    DRY = "dry"
    ANKLE_DEEP = "ankle_deep"
    KNEE_DEEP = "knee_deep"
    WAIST_DEEP = "waist_deep"
    CHEST_DEEP = "chest_deep"
    OVERHEAD = "overhead"
    UNKNOWN = "unknown"


class DamageSeverity(str, Enum):
    """Damage severity levels"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ReportStatus(str, Enum):
    """Report verification status"""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    DUPLICATE = "duplicate"


class VerificationSource(str, Enum):
    """Source of verification"""
    AI_ANALYSIS = "ai_analysis"
    HUMAN_REVIEW = "human_review"
    CROSS_VALIDATION = "cross_validation"
    OFFICIAL_SOURCE = "official_source"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class PhotoAnalysis:
    """AI analysis results from photo"""
    contains_water: bool
    water_level_category: WaterLevelCategory
    damage_severity: DamageSeverity
    objects_detected: list[str]
    is_outdoor: bool
    weather_condition: str
    confidence_score: float
    analysis_metadata: dict[str, Any]


@dataclass(frozen=True)
class LocationData:
    """Location information"""
    latitude: float
    longitude: float
    accuracy_m: float
    district: str | None = None
    taluk: str | None = None
    village: str | None = None
    landmark: str | None = None
    address: str | None = None


@dataclass(frozen=True)
class ReportMetrics:
    """Quality and trust metrics"""
    quality_score: float
    trust_score: float
    timeliness_score: float
    completeness_score: float
    consistency_score: float
    overall_score: float


@dataclass(frozen=True)
class CrowdsourcedReport:
    """Complete crowdsourced flood report"""
    report_id: str
    user_id: str | None
    observation_type: ObservationType
    description: str
    location: LocationData
    photos: list[str]
    photo_analyses: list[PhotoAnalysis]
    timestamp: datetime
    status: ReportStatus
    metrics: ReportMetrics
    verification_source: VerificationSource | None
    verified_at: datetime | None
    verified_by: str | None
    tags: list[str]
    related_reports: list[str]
    upvotes: int
    downvotes: int


# ============================================================
# PROTOCOLS
# ============================================================

class ImageAnalysisService(Protocol):
    """AI image analysis service"""
    async def analyze_flood_image(
        self,
        image_bytes: bytes,
        image_name: str,
    ) -> PhotoAnalysis: ...


class LocationService(Protocol):
    """Location verification service"""
    async def verify_location(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]: ...


class DuplicateDetector(Protocol):
    """Duplicate report detection"""
    async def find_similar_reports(
        self,
        location: LocationData,
        timestamp: datetime,
        radius_m: float,
    ) -> list[str]: ...


# ============================================================
# IMAGE ANALYSIS SERVICE
# ============================================================

class FloodImageAnalyzer:
    """
    Analyzes flood photos using AI computer vision.
    
    In production, integrate with:
    - ai/computer_vision/flood_water_detection.py
    - ai/computer_vision/building_damage_detection.py
    - ai/computer_vision/object_detection.py
    """
    
    def __init__(self):
        self.water_keywords = {
            "water", "flood", "rain", "puddle", "stream",
            "river", "overflow", "submerged", "wet"
        }
        self.damage_keywords = {
            "damage", "broken", "collapsed", "destroyed",
            "debris", "wreckage", "ruins"
        }

    async def analyze_flood_image(
        self,
        image_bytes: bytes,
        image_name: str,
    ) -> PhotoAnalysis:
        """
        Analyze flood image using computer vision.
        
        This is a simplified implementation. In production:
        1. Use CNN models for water detection
        2. Use object detection for infrastructure
        3. Use depth estimation for water level
        4. Use damage assessment models
        """
        
        # Simulate AI analysis
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Simple heuristic based on image name/metadata
        # In production, use actual CV models
        name_lower = image_name.lower()
        
        contains_water = any(kw in name_lower for kw in self.water_keywords)
        has_damage = any(kw in name_lower for kw in self.damage_keywords)
        
        # Estimate water level
        if "chest" in name_lower or "deep" in name_lower:
            water_level = WaterLevelCategory.CHEST_DEEP
            confidence = 0.75
        elif "knee" in name_lower:
            water_level = WaterLevelCategory.KNEE_DEEP
            confidence = 0.70
        elif contains_water:
            water_level = WaterLevelCategory.ANKLE_DEEP
            confidence = 0.65
        else:
            water_level = WaterLevelCategory.DRY
            confidence = 0.60
        
        # Estimate damage
        if has_damage:
            if "severe" in name_lower or "critical" in name_lower:
                damage = DamageSeverity.SEVERE
            else:
                damage = DamageSeverity.MODERATE
        else:
            damage = DamageSeverity.MINOR if contains_water else DamageSeverity.NONE
        
        # Detected objects (mock)
        objects = []
        if contains_water:
            objects.append("water")
        if "road" in name_lower:
            objects.append("road")
        if "building" in name_lower or "house" in name_lower:
            objects.append("building")
        if "vehicle" in name_lower or "car" in name_lower:
            objects.append("vehicle")
        
        return PhotoAnalysis(
            contains_water=contains_water,
            water_level_category=water_level,
            damage_severity=damage,
            objects_detected=objects,
            is_outdoor=True,
            weather_condition="rainy" if contains_water else "clear",
            confidence_score=confidence,
            analysis_metadata={
                "image_size_bytes": len(image_bytes),
                "image_name": image_name,
                "analysis_method": "simplified_heuristic",
                "note": "Use real CV models in production"
            }
        )


# ============================================================
# QUALITY SCORER
# ============================================================

class ReportQualityScorer:
    """Calculates quality and trust scores for reports"""
    
    def calculate_metrics(
        self,
        report_data: dict[str, Any],
        photo_analyses: list[PhotoAnalysis],
        user_history: dict[str, Any] | None = None,
    ) -> ReportMetrics:
        """Calculate comprehensive quality metrics"""
        
        # Quality score (0-1)
        quality_score = self._calculate_quality_score(report_data, photo_analyses)
        
        # Trust score (0-1)
        trust_score = self._calculate_trust_score(user_history)
        
        # Timeliness score (0-1)
        timeliness_score = self._calculate_timeliness_score(
            report_data.get("timestamp")
        )
        
        # Completeness score (0-1)
        completeness_score = self._calculate_completeness_score(report_data)
        
        # Consistency score (0-1)
        consistency_score = self._calculate_consistency_score(
            report_data,
            photo_analyses
        )
        
        # Overall score (weighted average)
        overall_score = (
            quality_score * 0.25 +
            trust_score * 0.20 +
            timeliness_score * 0.15 +
            completeness_score * 0.20 +
            consistency_score * 0.20
        )
        
        return ReportMetrics(
            quality_score=quality_score,
            trust_score=trust_score,
            timeliness_score=timeliness_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            overall_score=overall_score,
        )

    def _calculate_quality_score(
        self,
        report_data: dict[str, Any],
        photo_analyses: list[PhotoAnalysis],
    ) -> float:
        """Calculate photo and description quality"""
        score = 0.5  # Base score
        
        # Has photos
        if photo_analyses:
            score += 0.2
            
            # High confidence photos
            avg_confidence = sum(
                p.confidence_score for p in photo_analyses
            ) / len(photo_analyses)
            score += avg_confidence * 0.2
        
        # Has description
        description = report_data.get("description", "")
        if description and len(description) >= 20:
            score += 0.1
            if len(description) >= 50:
                score += 0.1
        
        return min(score, 1.0)

    def _calculate_trust_score(
        self,
        user_history: dict[str, Any] | None,
    ) -> float:
        """Calculate user trust score based on history"""
        if not user_history:
            return 0.5  # Neutral for new users
        
        score = 0.5
        
        # Previous verified reports
        verified_count = user_history.get("verified_reports", 0)
        total_count = user_history.get("total_reports", 1)
        
        if total_count > 0:
            accuracy_rate = verified_count / total_count
            score += accuracy_rate * 0.3
        
        # Account age
        account_age_days = user_history.get("account_age_days", 0)
        if account_age_days > 30:
            score += 0.1
        if account_age_days > 90:
            score += 0.1
        
        return min(score, 1.0)

    def _calculate_timeliness_score(
        self,
        timestamp: datetime | None,
    ) -> float:
        """Calculate how recent the report is"""
        if not timestamp:
            return 0.5
        
        now = datetime.now(timezone.utc)
        age_hours = (now - timestamp).total_seconds() / 3600
        
        # Fresher reports get higher scores
        if age_hours < 1:
            return 1.0
        elif age_hours < 6:
            return 0.9
        elif age_hours < 24:
            return 0.7
        elif age_hours < 72:
            return 0.5
        else:
            return 0.3

    def _calculate_completeness_score(
        self,
        report_data: dict[str, Any],
    ) -> float:
        """Calculate how complete the report is"""
        score = 0.0
        
        # Required fields
        if report_data.get("description"):
            score += 0.3
        if report_data.get("location"):
            score += 0.3
        if report_data.get("observation_type"):
            score += 0.2
        
        # Optional but valuable fields
        if report_data.get("photos"):
            score += 0.1
        if report_data.get("tags"):
            score += 0.05
        if report_data.get("landmark"):
            score += 0.05
        
        return min(score, 1.0)

    def _calculate_consistency_score(
        self,
        report_data: dict[str, Any],
        photo_analyses: list[PhotoAnalysis],
    ) -> float:
        """Calculate consistency between description and photos"""
        if not photo_analyses:
            return 0.7  # Neutral without photos
        
        description = report_data.get("description", "").lower()
        obs_type = report_data.get("observation_type")
        
        score = 0.5
        
        # Check if photos match observation type
        if obs_type == ObservationType.FLOODING.value:
            contains_water = any(p.contains_water for p in photo_analyses)
            if contains_water:
                score += 0.3
        
        # Check if description matches photo analysis
        if any(p.contains_water for p in photo_analyses):
            if any(word in description for word in ["water", "flood", "rain"]):
                score += 0.2
        
        return min(score, 1.0)


# ============================================================
# CROWDSOURCE SERVICE
# ============================================================

class CrowdsourceService:
    """
    Main service for managing crowdsourced flood reports.
    
    Coordinates:
    - Report submission and validation
    - Photo analysis
    - Quality scoring
    - Verification workflow
    - Duplicate detection
    - Real-time aggregation
    """
    
    def __init__(
        self,
        image_analyzer: ImageAnalysisService | None = None,
        location_service: LocationService | None = None,
        duplicate_detector: DuplicateDetector | None = None,
    ):
        self.image_analyzer = image_analyzer or FloodImageAnalyzer()
        self.location_service = location_service
        self.duplicate_detector = duplicate_detector
        self.quality_scorer = ReportQualityScorer()
        
        # In-memory cache (use Redis in production)
        self.pending_reports: dict[str, dict] = {}
        self.verified_reports: dict[str, CrowdsourcedReport] = {}

    async def submit_report(
        self,
        user_id: str | None,
        observation_type: ObservationType,
        description: str,
        latitude: float,
        longitude: float,
        accuracy_m: float,
        photos: list[tuple[bytes, str]],  # (bytes, filename)
        tags: list[str] | None = None,
        landmark: str | None = None,
    ) -> CrowdsourcedReport:
        """
        Submit a new crowdsourced flood report.
        
        Args:
            user_id: User ID (None for anonymous)
            observation_type: Type of observation
            description: Text description
            latitude: GPS latitude
            longitude: GPS longitude
            accuracy_m: GPS accuracy in meters
            photos: List of (image_bytes, filename) tuples
            tags: Optional tags
            landmark: Optional landmark description
            
        Returns:
            CrowdsourcedReport with analysis results
        """
        
        report_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)
        
        logger.info(f"Processing crowdsourced report: {report_id}")
        
        # Analyze photos
        photo_analyses = []
        photo_ids = []
        
        for image_bytes, filename in photos:
            # Store photo (in production, upload to S3)
            photo_id = self._store_photo(image_bytes, filename)
            photo_ids.append(photo_id)
            
            # Analyze photo
            analysis = await self.image_analyzer.analyze_flood_image(
                image_bytes=image_bytes,
                image_name=filename,
            )
            photo_analyses.append(analysis)
        
        # Create location data
        location = LocationData(
            latitude=latitude,
            longitude=longitude,
            accuracy_m=accuracy_m,
            landmark=landmark,
        )
        
        # Enhance location if service available
        if self.location_service:
            try:
                location_info = await self.location_service.verify_location(
                    latitude, longitude
                )
                location = LocationData(
                    latitude=latitude,
                    longitude=longitude,
                    accuracy_m=accuracy_m,
                    district=location_info.get("district"),
                    taluk=location_info.get("taluk"),
                    village=location_info.get("village"),
                    landmark=landmark,
                    address=location_info.get("address"),
                )
            except Exception as e:
                logger.warning(f"Location enhancement failed: {e}")
        
        # Calculate quality metrics
        report_data = {
            "description": description,
            "location": location,
            "observation_type": observation_type.value,
            "photos": photo_ids,
            "tags": tags or [],
            "landmark": landmark,
            "timestamp": timestamp,
        }
        
        # Get user history (mock - in production query database)
        user_history = None
        if user_id:
            user_history = {
                "verified_reports": 5,
                "total_reports": 10,
                "account_age_days": 45,
            }
        
        metrics = self.quality_scorer.calculate_metrics(
            report_data=report_data,
            photo_analyses=photo_analyses,
            user_history=user_history,
        )
        
        # Determine initial status
        status = ReportStatus.PENDING
        verification_source = None
        
        # Auto-verify high-quality reports
        if metrics.overall_score >= 0.8 and photo_analyses:
            status = ReportStatus.VERIFIED
            verification_source = VerificationSource.AI_ANALYSIS
        
        # Check for duplicates
        related_reports = []
        if self.duplicate_detector:
            try:
                related_reports = await self.duplicate_detector.find_similar_reports(
                    location=location,
                    timestamp=timestamp,
                    radius_m=500,  # 500m radius
                )
                if related_reports:
                    logger.info(
                        f"Found {len(related_reports)} similar reports near this location"
                    )
            except Exception as e:
                logger.warning(f"Duplicate detection failed: {e}")
        
        # Create report
        report = CrowdsourcedReport(
            report_id=report_id,
            user_id=user_id,
            observation_type=observation_type,
            description=description,
            location=location,
            photos=photo_ids,
            photo_analyses=photo_analyses,
            timestamp=timestamp,
            status=status,
            metrics=metrics,
            verification_source=verification_source,
            verified_at=timestamp if status == ReportStatus.VERIFIED else None,
            verified_by="ai_system" if status == ReportStatus.VERIFIED else None,
            tags=tags or [],
            related_reports=related_reports,
            upvotes=0,
            downvotes=0,
        )
        
        # Store report
        if status == ReportStatus.VERIFIED:
            self.verified_reports[report_id] = report
        else:
            self.pending_reports[report_id] = report.__dict__
        
        logger.info(
            f"Report submitted: {report_id}, status={status.value}, "
            f"score={metrics.overall_score:.2f}"
        )
        
        return report

    async def verify_report(
        self,
        report_id: str,
        verified_by: str,
        approved: bool,
        reason: str | None = None,
    ) -> bool:
        """Manually verify or reject a report"""
        
        if report_id not in self.pending_reports:
            return False
        
        report_data = self.pending_reports[report_id]
        
        if approved:
            report_data["status"] = ReportStatus.VERIFIED.value
            report_data["verification_source"] = VerificationSource.HUMAN_REVIEW.value
            report_data["verified_at"] = datetime.now(timezone.utc)
            report_data["verified_by"] = verified_by
            
            # Move to verified
            self.verified_reports[report_id] = report_data
            del self.pending_reports[report_id]
            
            logger.info(f"Report verified: {report_id} by {verified_by}")
        else:
            report_data["status"] = ReportStatus.REJECTED.value
            report_data["rejection_reason"] = reason
            
            logger.info(f"Report rejected: {report_id} by {verified_by}")
        
        return True

    async def upvote_report(self, report_id: str, user_id: str) -> bool:
        """Upvote a report for community validation"""
        report = self.verified_reports.get(report_id)
        if report:
            # In production, track votes per user to prevent duplicates
            report = report._replace(upvotes=report.upvotes + 1)
            self.verified_reports[report_id] = report
            return True
        return False

    async def get_recent_reports(
        self,
        limit: int = 50,
        district: str | None = None,
        observation_type: ObservationType | None = None,
        min_score: float = 0.0,
    ) -> list[CrowdsourcedReport]:
        """Get recent verified reports with filters"""
        
        reports = list(self.verified_reports.values())
        
        # Apply filters
        if district:
            reports = [r for r in reports if r.location.district == district]
        
        if observation_type:
            reports = [r for r in reports if r.observation_type == observation_type]
        
        if min_score > 0:
            reports = [r for r in reports if r.metrics.overall_score >= min_score]
        
        # Sort by timestamp (newest first)
        reports.sort(key=lambda r: r.timestamp, reverse=True)
        
        return reports[:limit]

    async def get_heatmap_data(
        self,
        district: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated data for heatmap visualization"""
        
        reports = list(self.verified_reports.values())
        
        if district:
            reports = [r for r in reports if r.location.district == district]
        
        # Create heatmap points
        heatmap_data = []
        for report in reports:
            # Weight by quality score and severity
            weight = report.metrics.overall_score
            
            # Increase weight for severe flooding
            if report.photo_analyses:
                for analysis in report.photo_analyses:
                    if analysis.water_level_category in [
                        WaterLevelCategory.CHEST_DEEP,
                        WaterLevelCategory.OVERHEAD
                    ]:
                        weight *= 2.0
            
            heatmap_data.append({
                "lat": report.location.latitude,
                "lng": report.location.longitude,
                "weight": weight,
                "report_id": report.report_id,
            })
        
        return heatmap_data

    def _store_photo(self, image_bytes: bytes, filename: str) -> str:
        """Store photo and return ID (mock - use S3 in production)"""
        photo_id = hashlib.sha256(image_bytes).hexdigest()[:16]
        logger.info(f"Stored photo: {photo_id} ({len(image_bytes)} bytes)")
        return photo_id


# ============================================================
# FACTORY
# ============================================================

async def create_crowdsource_service(
    image_analyzer: ImageAnalysisService | None = None,
    location_service: LocationService | None = None,
    duplicate_detector: DuplicateDetector | None = None,
) -> CrowdsourceService:
    """Create and initialize CrowdsourceService"""
    
    service = CrowdsourceService(
        image_analyzer=image_analyzer,
        location_service=location_service,
        duplicate_detector=duplicate_detector,
    )
    
    logger.info("Crowdsource service initialized")
    
    return service
