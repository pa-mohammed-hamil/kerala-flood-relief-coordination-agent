"""
Family Safety API
REST endpoints for emergency contact network and family safety check-ins
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models.user import User
from backend.models.family_safety import (
    FamilyCircle, FamilyMember, SafetyCheckIn, MemberLocation,
    CheckInStatus, MemberRole, PrivacyLevel
)
from backend.services.family_safety_service import (
    FamilyCircleService, CheckInService, NotificationOrchestrator, AutomationEngine
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/family-safety", tags=["Family Safety"])


# ========== Request/Response Schemas ==========

class CreateCircleRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    home_latitude: Optional[float] = None
    home_longitude: Optional[float] = None
    home_address: Optional[str] = None
    check_in_frequency_hours: int = Field(default=12, ge=1, le=168)


class AddMemberRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    role: MemberRole = MemberRole.MEMBER
    user_id: Optional[int] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    privacy_level: PrivacyLevel = PrivacyLevel.CONTACTS_ONLY
    has_mobility_issues: bool = False
    has_medical_conditions: bool = False
    medical_notes: Optional[str] = None


class UpdateMemberRequest(BaseModel):
    name: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    privacy_level: Optional[PrivacyLevel] = None
    notify_sms: Optional[bool] = None
    notify_whatsapp: Optional[bool] = None
    notify_email: Optional[bool] = None
    notify_push: Optional[bool] = None


class SubmitCheckInRequest(BaseModel):
    status: CheckInStatus
    message: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_accuracy: Optional[float] = None
    is_automated: bool = False


class UpdateLocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = Field(None, ge=0, lt=360)
    battery_level: Optional[int] = Field(None, ge=0, le=100)


class CircleResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    is_active: bool
    member_count: int
    check_in_frequency_hours: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class MemberResponse(BaseModel):
    id: int
    circle_id: int
    name: str
    role: MemberRole
    phone_number: Optional[str]
    email: Optional[str]
    privacy_level: PrivacyLevel
    is_active: bool
    last_check_in_time: Optional[datetime]
    joined_at: datetime
    
    class Config:
        from_attributes = True


class CheckInResponse(BaseModel):
    id: int
    member_id: int
    member_name: str
    status: CheckInStatus
    message: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    check_in_time: datetime
    is_automated: bool
    escalated: bool
    
    class Config:
        from_attributes = True


# ========== Circle Management Endpoints ==========

@router.post("/circles", response_model=CircleResponse)
def create_family_circle(
    request: CreateCircleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new family circle"""
    service = FamilyCircleService(db)
    
    circle = service.create_circle(
        name=request.name,
        created_by_user_id=current_user.id,
        description=request.description,
        home_latitude=request.home_latitude,
        home_longitude=request.home_longitude,
        home_address=request.home_address,
        check_in_frequency_hours=request.check_in_frequency_hours
    )
    
    members = service.get_circle_members(circle.id)
    
    return CircleResponse(
        id=circle.id,
        name=circle.name,
        description=circle.description,
        is_active=circle.is_active,
        member_count=len(members),
        check_in_frequency_hours=circle.check_in_frequency_hours,
        created_at=circle.created_at
    )


@router.get("/circles", response_model=List[CircleResponse])
def get_my_circles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all circles the current user belongs to"""
    service = FamilyCircleService(db)
    circles = service.get_member_circles(current_user.id)
    
    response = []
    for circle in circles:
        members = service.get_circle_members(circle.id)
        response.append(CircleResponse(
            id=circle.id,
            name=circle.name,
            description=circle.description,
            is_active=circle.is_active,
            member_count=len(members),
            check_in_frequency_hours=circle.check_in_frequency_hours,
            created_at=circle.created_at
        ))
    
    return response


@router.get("/circles/{circle_id}")
def get_circle_details(
    circle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a circle"""
    circle = db.query(FamilyCircle).filter(FamilyCircle.id == circle_id).first()
    if not circle:
        raise HTTPException(status_code=404, detail="Circle not found")
    
    # Verify user is member
    member = db.query(FamilyMember).filter(
        FamilyMember.circle_id == circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.is_active == True
    ).first()
    
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this circle")
    
    service = FamilyCircleService(db)
    members = service.get_circle_members(circle_id)
    
    return {
        "id": circle.id,
        "name": circle.name,
        "description": circle.description,
        "home_latitude": circle.home_latitude,
        "home_longitude": circle.home_longitude,
        "home_address": circle.home_address,
        "check_in_frequency_hours": circle.check_in_frequency_hours,
        "auto_check_in_enabled": circle.auto_check_in_enabled,
        "is_active": circle.is_active,
        "member_count": len(members),
        "created_at": circle.created_at
    }


@router.patch("/circles/{circle_id}")
def update_circle(
    circle_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    check_in_frequency_hours: Optional[int] = None,
    auto_check_in_enabled: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update circle settings (admin only)"""
    circle = db.query(FamilyCircle).filter(FamilyCircle.id == circle_id).first()
    if not circle:
        raise HTTPException(status_code=404, detail="Circle not found")
    
    # Verify user is admin
    member = db.query(FamilyMember).filter(
        FamilyMember.circle_id == circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.role == MemberRole.ADMIN
    ).first()
    
    if not member:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if name:
        circle.name = name
    if description is not None:
        circle.description = description
    if check_in_frequency_hours:
        circle.check_in_frequency_hours = check_in_frequency_hours
    if auto_check_in_enabled is not None:
        circle.auto_check_in_enabled = auto_check_in_enabled
    
    circle.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Circle updated successfully"}


@router.delete("/circles/{circle_id}")
def deactivate_circle(
    circle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deactivate a circle (admin only)"""
    circle = db.query(FamilyCircle).filter(FamilyCircle.id == circle_id).first()
    if not circle:
        raise HTTPException(status_code=404, detail="Circle not found")
    
    # Verify user is admin
    member = db.query(FamilyMember).filter(
        FamilyMember.circle_id == circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.role == MemberRole.ADMIN
    ).first()
    
    if not member:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    circle.is_active = False
    circle.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Circle deactivated"}


# ========== Member Management Endpoints ==========

@router.post("/circles/{circle_id}/members", response_model=MemberResponse)
def add_circle_member(
    circle_id: int,
    request: AddMemberRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a member to the circle (admin only)"""
    circle = db.query(FamilyCircle).filter(FamilyCircle.id == circle_id).first()
    if not circle:
        raise HTTPException(status_code=404, detail="Circle not found")
    
    # Verify user is admin
    admin = db.query(FamilyMember).filter(
        FamilyMember.circle_id == circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.role == MemberRole.ADMIN
    ).first()
    
    if not admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    service = FamilyCircleService(db)
    member = service.add_member(
        circle_id=circle_id,
        name=request.name,
        role=request.role,
        user_id=request.user_id,
        phone_number=request.phone_number,
        email=request.email,
        privacy_level=request.privacy_level,
        has_mobility_issues=request.has_mobility_issues,
        has_medical_conditions=request.has_medical_conditions,
        medical_notes=request.medical_notes
    )
    
    return MemberResponse(
        id=member.id,
        circle_id=member.circle_id,
        name=member.name,
        role=member.role,
        phone_number=member.phone_number,
        email=member.email,
        privacy_level=member.privacy_level,
        is_active=member.is_active,
        last_check_in_time=None,
        joined_at=member.joined_at
    )


@router.get("/circles/{circle_id}/members", response_model=List[MemberResponse])
def get_circle_members(
    circle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all members of a circle"""
    # Verify user is member
    member = db.query(FamilyMember).filter(
        FamilyMember.circle_id == circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.is_active == True
    ).first()
    
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this circle")
    
    service = FamilyCircleService(db)
    members = service.get_circle_members(circle_id)
    
    response = []
    for m in members:
        last_check_in = db.query(SafetyCheckIn).filter(
            SafetyCheckIn.member_id == m.id
        ).order_by(SafetyCheckIn.check_in_time.desc()).first()
        
        response.append(MemberResponse(
            id=m.id,
            circle_id=m.circle_id,
            name=m.name,
            role=m.role,
            phone_number=m.phone_number if m.privacy_level != PrivacyLevel.PRIVATE else None,
            email=m.email if m.privacy_level != PrivacyLevel.PRIVATE else None,
            privacy_level=m.privacy_level,
            is_active=m.is_active,
            last_check_in_time=last_check_in.check_in_time if last_check_in else None,
            joined_at=m.joined_at
        ))
    
    return response


@router.patch("/members/{member_id}")
def update_member(
    member_id: int,
    request: UpdateMemberRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update member information"""
    member = db.query(FamilyMember).filter(FamilyMember.id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    # Verify user is the member or admin
    is_self = member.user_id == current_user.id
    is_admin = db.query(FamilyMember).filter(
        FamilyMember.circle_id == member.circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.role == MemberRole.ADMIN
    ).first() is not None
    
    if not (is_self or is_admin):
        raise HTTPException(status_code=403, detail="Permission denied")
    
    if request.name:
        member.name = request.name
    if request.phone_number is not None:
        member.phone_number = request.phone_number
    if request.email is not None:
        member.email = request.email
    if request.privacy_level:
        member.privacy_level = request.privacy_level
    if request.notify_sms is not None:
        member.notify_sms = request.notify_sms
    if request.notify_whatsapp is not None:
        member.notify_whatsapp = request.notify_whatsapp
    if request.notify_email is not None:
        member.notify_email = request.notify_email
    if request.notify_push is not None:
        member.notify_push = request.notify_push
    
    member.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Member updated successfully"}


@router.delete("/members/{member_id}")
def remove_member(
    member_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove member from circle"""
    service = FamilyCircleService(db)
    success = service.remove_member(member_id, current_user.id)
    
    if not success:
        raise HTTPException(status_code=403, detail="Permission denied or member not found")
    
    return {"message": "Member removed successfully"}


# ========== Check-In Endpoints ==========

@router.post("/members/{member_id}/check-in", response_model=CheckInResponse)
def submit_check_in(
    member_id: int,
    request: SubmitCheckInRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit a safety check-in"""
    member = db.query(FamilyMember).filter(FamilyMember.id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    # Verify user is the member or admin
    is_self = member.user_id == current_user.id
    is_admin = db.query(FamilyMember).filter(
        FamilyMember.circle_id == member.circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.role == MemberRole.ADMIN
    ).first() is not None
    
    if not (is_self or is_admin):
        raise HTTPException(status_code=403, detail="Permission denied")
    
    service = CheckInService(db)
    check_in = service.submit_check_in(
        member_id=member_id,
        status=request.status,
        message=request.message,
        latitude=request.latitude,
        longitude=request.longitude,
        location_accuracy=request.location_accuracy,
        is_automated=request.is_automated
    )
    
    return CheckInResponse(
        id=check_in.id,
        member_id=check_in.member_id,
        member_name=member.name,
        status=check_in.status,
        message=check_in.message,
        latitude=check_in.latitude,
        longitude=check_in.longitude,
        check_in_time=check_in.check_in_time,
        is_automated=check_in.is_automated,
        escalated=check_in.escalated
    )


@router.post("/members/{member_id}/location")
def update_member_location(
    member_id: int,
    request: UpdateLocationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update real-time location for member"""
    member = db.query(FamilyMember).filter(FamilyMember.id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    # Verify user is the member
    if member.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Can only update own location")
    
    service = CheckInService(db)
    location = service.update_member_location(
        member_id=member_id,
        latitude=request.latitude,
        longitude=request.longitude,
        accuracy=request.accuracy,
        speed=request.speed,
        heading=request.heading,
        battery_level=request.battery_level
    )
    
    return {
        "message": "Location updated",
        "timestamp": location.timestamp
    }


@router.get("/circles/{circle_id}/status")
def get_circle_status(
    circle_id: int,
    hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get status overview for entire circle"""
    # Verify user is member
    member = db.query(FamilyMember).filter(
        FamilyMember.circle_id == circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.is_active == True
    ).first()
    
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this circle")
    
    service = CheckInService(db)
    status = service.get_circle_status(circle_id, hours)
    
    return status


@router.get("/members/{member_id}/history", response_model=List[CheckInResponse])
def get_member_check_in_history(
    member_id: int,
    days: int = Query(default=7, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get check-in history for a member"""
    member = db.query(FamilyMember).filter(FamilyMember.id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    # Verify user is circle member
    requester = db.query(FamilyMember).filter(
        FamilyMember.circle_id == member.circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.is_active == True
    ).first()
    
    if not requester:
        raise HTTPException(status_code=403, detail="Not a member of this circle")
    
    service = CheckInService(db)
    check_ins = service.get_member_history(member_id, days)
    
    return [
        CheckInResponse(
            id=c.id,
            member_id=c.member_id,
            member_name=member.name,
            status=c.status,
            message=c.message,
            latitude=c.latitude,
            longitude=c.longitude,
            check_in_time=c.check_in_time,
            is_automated=c.is_automated,
            escalated=c.escalated
        )
        for c in check_ins
    ]


@router.get("/circles/{circle_id}/risk-assessment")
def assess_circle_risk(
    circle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get risk assessment for family circle"""
    # Verify user is member
    member = db.query(FamilyMember).filter(
        FamilyMember.circle_id == circle_id,
        FamilyMember.user_id == current_user.id,
        FamilyMember.is_active == True
    ).first()
    
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this circle")
    
    engine = AutomationEngine(db)
    risk_assessment = engine.assess_circle_risk(circle_id)
    
    return risk_assessment


# ========== Admin/System Endpoints ==========

@router.post("/admin/send-check-in-requests")
def trigger_check_in_requests(
    alert_id: Optional[int] = None,
    affected_districts: List[str] = Query(default=[]),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Manually trigger check-in requests (admin only)"""
    # Verify admin role
    if current_user.role not in ["admin", "coordinator"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    engine = AutomationEngine(db)
    
    if alert_id:
        sent_count = engine.process_alert_triggered_check_ins(alert_id, affected_districts)
    else:
        sent_count = engine.process_scheduled_check_ins()
    
    return {
        "message": f"Sent {sent_count} check-in requests",
        "sent_count": sent_count
    }


@router.post("/admin/process-overdue")
def process_overdue_check_ins(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Process overdue check-ins and escalations (admin only)"""
    if current_user.role not in ["admin", "coordinator"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    engine = AutomationEngine(db)
    escalated_count = engine.process_overdue_check_ins()
    
    return {
        "message": f"Processed overdue check-ins, escalated {escalated_count}",
        "escalated_count": escalated_count
    }
