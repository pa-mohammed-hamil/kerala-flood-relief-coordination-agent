"""
backend/api/crowdsource.py

Crowdsourced Flood Reporting API
Kerala Flood Relief Coordination Agent

RESTful API endpoints for citizen flood observations:
    - Submit flood reports with photos
    - Real-time photo analysis
    - Report verification workflow
    - Community voting
    - Report browsing and filtering
    - Heatmap data aggregation
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from backend.dependencies import get_current_user
from backend.services.crowdsource_service import (
    CrowdsourceService,
    CrowdsourcedReport,
    DamageSeverity,
    ObservationType,
    ReportStatus,
    WaterLevelCategory,
)


router = APIRouter(
    prefix="/crowdsource",
    tags=["Crowdsourced Reporting"],
)

CurrentUser = Annotated[Any, Depends(get_current_user)]


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ObservationTypeEnum(str, Enum):
    """Observation types"""
    WATER_LEVEL = "water_level"
    FLOODING = "flooding"
    ROAD_BLOCKAGE = "road_blockage"
    DAMAGE = "damage"
    RESCUE_NEEDED = "rescue_needed"
    INFRASTRUCTURE = "infrastructure"
    WEATHER = "weather"
    GENERAL = "general"


class ReportVerificationRequest(BaseModel):
    """Verify or reject a report"""
    model_config = ConfigDict(extra="forbid")
    
    approved: bool
    reason: str | None = Field(
        default=None,
        max_length=500,
        description="Reason for rejection (if not approved)"
    )


class ReportFilterParams(BaseModel):
    """Query parameters for filtering reports"""
    district: str | None = None
    observation_type: ObservationTypeEnum | None = None
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=50, ge=1, le=200)


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class PhotoAnalysisResponse(BaseModel):
    """Photo AI analysis result"""
    contains_water: bool
    water_level_category: str
    damage_severity: str
    objects_detected: list[str]
    is_outdoor: bool
    weather_condition: str
    confidence_score: float


class LocationResponse(BaseModel):
    """Location information"""
    latitude: float
    longitude: float
    accuracy_m: float
    district: str | None = None
    taluk: str | None = None
    village: str | None = None
    landmark: str | None = None
    address: str | None = None


class MetricsResponse(BaseModel):
    """Quality metrics"""
    quality_score: float
    trust_score: float
    timeliness_score: float
    completeness_score: float
    consistency_score: float
    overall_score: float


class ReportResponse(BaseModel):
    """Crowdsourced report response"""
    report_id: str
    user_id: str | None
    observation_type: str
    description: str
    location: LocationResponse
    photos: list[str]
    photo_analyses: list[PhotoAnalysisResponse]
    timestamp: datetime
    status: str
    metrics: MetricsResponse
    verification_source: str | None
    verified_at: datetime | None
    verified_by: str | None
    tags: list[str]
    related_reports: list[str]
    upvotes: int
    downvotes: int


class ReportListResponse(BaseModel):
    """List of reports"""
    reports: list[ReportResponse]
    total: int
    filters_applied: dict[str, Any]


class HeatmapDataResponse(BaseModel):
    """Heatmap visualization data"""
    points: list[dict[str, Any]]
    total_points: int
    timestamp: datetime


class ReportStatisticsResponse(BaseModel):
    """Crowdsource statistics"""
    total_reports: int
    verified_reports: int
    pending_reports: int
    by_observation_type: dict[str, int]
    by_water_level: dict[str, int]
    average_quality_score: float
    reports_last_24h: int
    active_reporters: int
    generated_at: datetime


class CrowdsourceHealthResponse(BaseModel):
    """Service health"""
    service: str = "crowdsource"
    status: str = "healthy"
    features: list[str] = [
        "photo_upload",
        "ai_analysis",
        "verification",
        "heatmap",
        "voting"
    ]
    max_photos_per_report: int = 5
    max_photo_size_mb: int = 10
    supported_formats: list[str] = ["jpg", "jpeg", "png", "webp"]
    timestamp: datetime


# ============================================================
# SERVICE DEPENDENCY
# ============================================================

def get_crowdsource_service(request: Request) -> CrowdsourceService:
    """Get crowdsource service from app state"""
    service = getattr(request.app.state, "crowdsource_service", None)
    
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Crowdsource service is not configured",
        )
    
    return service


CrowdsourceServiceDep = Annotated[
    CrowdsourceService,
    Depends(get_crowdsource_service)
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_user_id(user: Any) -> str | None:
    """Extract user ID"""
    if user is None:
        return None
    user_id = getattr(user, "id", None)
    return str(user_id) if user_id is not None else None


def to_report_response(report: CrowdsourcedReport) -> ReportResponse:
    """Convert report to API response"""
    
    photo_analyses = [
        PhotoAnalysisResponse(
            contains_water=pa.contains_water,
            water_level_category=pa.water_level_category.value,
            damage_severity=pa.damage_severity.value,
            objects_detected=pa.objects_detected,
            is_outdoor=pa.is_outdoor,
            weather_condition=pa.weather_condition,
            confidence_score=pa.confidence_score,
        )
        for pa in report.photo_analyses
    ]
    
    location = LocationResponse(
        latitude=report.location.latitude,
        longitude=report.location.longitude,
        accuracy_m=report.location.accuracy_m,
        district=report.location.district,
        taluk=report.location.taluk,
        village=report.location.village,
        landmark=report.location.landmark,
        address=report.location.address,
    )
    
    metrics = MetricsResponse(
        quality_score=report.metrics.quality_score,
        trust_score=report.metrics.trust_score,
        timeliness_score=report.metrics.timeliness_score,
        completeness_score=report.metrics.completeness_score,
        consistency_score=report.metrics.consistency_score,
        overall_score=report.metrics.overall_score,
    )
    
    return ReportResponse(
        report_id=report.report_id,
        user_id=report.user_id,
        observation_type=report.observation_type.value,
        description=report.description,
        location=location,
        photos=report.photos,
        photo_analyses=photo_analyses,
        timestamp=report.timestamp,
        status=report.status.value,
        metrics=metrics,
        verification_source=report.verification_source.value if report.verification_source else None,
        verified_at=report.verified_at,
        verified_by=report.verified_by,
        tags=report.tags,
        related_reports=report.related_reports,
        upvotes=report.upvotes,
        downvotes=report.downvotes,
    )


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/health", response_model=CrowdsourceHealthResponse)
async def crowdsource_health() -> CrowdsourceHealthResponse:
    """
    Crowdsource service health check.
    
    Returns service status and capabilities.
    """
    return CrowdsourceHealthResponse(
        timestamp=datetime.utcnow(),
    )


@router.post(
    "/reports",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_report(
    observation_type: ObservationTypeEnum = Form(...),
    description: str = Form(..., min_length=10, max_length=2000),
    latitude: float = Form(..., ge=-90, le=90),
    longitude: float = Form(..., ge=-180, le=180),
    accuracy_m: float = Form(..., ge=0),
    photos: list[UploadFile] = File(..., max_length=5),
    tags: str | None = Form(None),
    landmark: str | None = Form(None, max_length=200),
    current_user: CurrentUser = Depends(get_current_user),
    service: CrowdsourceServiceDep = Depends(get_crowdsource_service),
) -> ReportResponse:
    """
    Submit a crowdsourced flood observation report.
    
    Citizens can report:
    - Flooding and water levels
    - Road blockages
    - Infrastructure damage
    - Weather conditions
    - Rescue needs
    
    Include up to 5 photos for AI analysis:
    - Water level detection
    - Damage assessment
    - Object recognition
    
    Reports are automatically scored for quality and may be
    auto-verified for high-quality submissions.
    
    Maximum photo size: 10 MB per photo
    Supported formats: JPG, JPEG, PNG, WEBP
    """
    
    # Validate photos
    if len(photos) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 photos per report"
        )
    
    photo_data = []
    for photo in photos:
        if not photo.filename:
            raise HTTPException(
                status_code=400,
                detail="Photo filename is required"
            )
        
        # Check file extension
        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        file_ext = photo.filename.lower().split(".")[-1]
        if f".{file_ext}" not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported photo format: {file_ext}. "
                       f"Supported: {', '.join(allowed_extensions)}"
            )
        
        # Read photo data
        try:
            photo_bytes = await photo.read()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read photo: {str(e)}"
            )
        
        # Check size
        if len(photo_bytes) > 10 * 1024 * 1024:  # 10 MB
            raise HTTPException(
                status_code=413,
                detail=f"Photo {photo.filename} too large. Maximum size is 10 MB."
            )
        
        if len(photo_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Photo {photo.filename} is empty"
            )
        
        photo_data.append((photo_bytes, photo.filename))
    
    # Parse tags
    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    
    # Get user ID
    user_id = get_user_id(current_user)
    
    # Map observation type
    obs_type_map = {
        ObservationTypeEnum.WATER_LEVEL: ObservationType.WATER_LEVEL,
        ObservationTypeEnum.FLOODING: ObservationType.FLOODING,
        ObservationTypeEnum.ROAD_BLOCKAGE: ObservationType.ROAD_BLOCKAGE,
        ObservationTypeEnum.DAMAGE: ObservationType.DAMAGE,
        ObservationTypeEnum.RESCUE_NEEDED: ObservationType.RESCUE_NEEDED,
        ObservationTypeEnum.INFRASTRUCTURE: ObservationType.INFRASTRUCTURE,
        ObservationTypeEnum.WEATHER: ObservationType.WEATHER,
        ObservationTypeEnum.GENERAL: ObservationType.GENERAL,
    }
    
    try:
        report = await service.submit_report(
            user_id=user_id,
            observation_type=obs_type_map[observation_type],
            description=description,
            latitude=latitude,
            longitude=longitude,
            accuracy_m=accuracy_m,
            photos=photo_data,
            tags=tag_list,
            landmark=landmark,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process report: {str(e)}"
        )
    
    return to_report_response(report)


@router.get(
    "/reports",
    response_model=ReportListResponse,
)
async def list_reports(
    district: str | None = Query(None),
    observation_type: ObservationTypeEnum | None = Query(None),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    current_user: CurrentUser = Depends(get_current_user),
    service: CrowdsourceServiceDep = Depends(get_crowdsource_service),
) -> ReportListResponse:
    """
    List verified crowdsourced reports with filtering.
    
    Filter by:
    - District
    - Observation type
    - Minimum quality score
    
    Results are sorted by timestamp (newest first).
    """
    
    # Map observation type
    obs_type = None
    if observation_type:
        obs_type_map = {
            ObservationTypeEnum.WATER_LEVEL: ObservationType.WATER_LEVEL,
            ObservationTypeEnum.FLOODING: ObservationType.FLOODING,
            ObservationTypeEnum.ROAD_BLOCKAGE: ObservationType.ROAD_BLOCKAGE,
            ObservationTypeEnum.DAMAGE: ObservationType.DAMAGE,
            ObservationTypeEnum.RESCUE_NEEDED: ObservationType.RESCUE_NEEDED,
            ObservationTypeEnum.INFRASTRUCTURE: ObservationType.INFRASTRUCTURE,
            ObservationTypeEnum.WEATHER: ObservationType.WEATHER,
            ObservationTypeEnum.GENERAL: ObservationType.GENERAL,
        }
        obs_type = obs_type_map[observation_type]
    
    reports = await service.get_recent_reports(
        limit=limit,
        district=district,
        observation_type=obs_type,
        min_score=min_score,
    )
    
    return ReportListResponse(
        reports=[to_report_response(r) for r in reports],
        total=len(reports),
        filters_applied={
            "district": district,
            "observation_type": observation_type.value if observation_type else None,
            "min_score": min_score,
        }
    )


@router.get(
    "/reports/{report_id}",
    response_model=ReportResponse,
)
async def get_report(
    report_id: str,
    current_user: CurrentUser,
    service: CrowdsourceServiceDep,
) -> ReportResponse:
    """Get detailed information about a specific report"""
    
    # In production, query from database
    # For now, check in-memory caches
    if report_id in service.verified_reports:
        report = service.verified_reports[report_id]
        return to_report_response(report)
    
    if report_id in service.pending_reports:
        report_data = service.pending_reports[report_id]
        # Convert dict back to report object (simplified)
        raise HTTPException(
            status_code=200,
            detail="Report is pending verification"
        )
    
    raise HTTPException(
        status_code=404,
        detail=f"Report {report_id} not found"
    )


@router.post(
    "/reports/{report_id}/verify",
    response_model=dict[str, str],
)
async def verify_report(
    report_id: str,
    verification: ReportVerificationRequest,
    current_user: CurrentUser,
    service: CrowdsourceServiceDep,
) -> dict[str, str]:
    """
    Verify or reject a pending report.
    
    Requires coordinator or admin role.
    """
    
    # Check permissions (simplified - add proper role check)
    user_id = get_user_id(current_user)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    success = await service.verify_report(
        report_id=report_id,
        verified_by=user_id,
        approved=verification.approved,
        reason=verification.reason,
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Report {report_id} not found or already processed"
        )
    
    return {
        "status": "verified" if verification.approved else "rejected",
        "report_id": report_id,
        "message": "Report processed successfully"
    }


@router.post(
    "/reports/{report_id}/upvote",
    response_model=dict[str, Any],
)
async def upvote_report(
    report_id: str,
    current_user: CurrentUser,
    service: CrowdsourceServiceDep,
) -> dict[str, Any]:
    """
    Upvote a report for community validation.
    
    Helps identify reliable reports through community consensus.
    """
    
    user_id = get_user_id(current_user)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for voting"
        )
    
    success = await service.upvote_report(report_id, user_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Report {report_id} not found"
        )
    
    return {
        "status": "success",
        "report_id": report_id,
        "message": "Vote recorded"
    }


@router.get(
    "/heatmap",
    response_model=HeatmapDataResponse,
)
async def get_heatmap_data(
    district: str | None = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
    service: CrowdsourceServiceDep = Depends(get_crowdsource_service),
) -> HeatmapDataResponse:
    """
    Get aggregated data for heatmap visualization.
    
    Returns location points weighted by:
    - Report quality score
    - Water level severity
    - Damage severity
    
    Use this data to display flood intensity on a map.
    """
    
    points = await service.get_heatmap_data(district=district)
    
    return HeatmapDataResponse(
        points=points,
        total_points=len(points),
        timestamp=datetime.utcnow(),
    )


# ============================================================
# CONVERTERS
# ============================================================

__all__ = [
    "router",
]
