"""
Family Safety and Emergency Contact Network Models
Database models for family circles, member tracking, and safety check-ins
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from backend.database import Base
import enum


class PrivacyLevel(str, enum.Enum):
    """Privacy settings for location sharing"""
    PRIVATE = "private"  # Only circle admins
    CONTACTS_ONLY = "contacts_only"  # All circle members
    PUBLIC = "public"  # Public emergency responders


class MemberRole(str, enum.Enum):
    """Role within family circle"""
    ADMIN = "admin"  # Can manage circle
    MEMBER = "member"  # Regular member
    CHILD = "child"  # Dependent (special handling)
    ELDERLY = "elderly"  # Senior (special handling)
    CAREGIVER = "caregiver"  # Designated helper


class CheckInStatus(str, enum.Enum):
    """Status of safety check-in"""
    SAFE = "safe"
    UNSAFE = "unsafe"
    NEEDS_HELP = "needs_help"
    EVACUATING = "evacuating"
    AT_SHELTER = "at_shelter"
    NO_RESPONSE = "no_response"


class NotificationChannel(str, enum.Enum):
    """Notification delivery methods"""
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    PUSH = "push"
    VOICE_CALL = "voice_call"


class FamilyCircle(Base):
    """Family or community circle for emergency coordination"""
    __tablename__ = "family_circles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)  # "Kumar Family", "Apartment A-Block"
    description = Column(Text, nullable=True)
    
    # Admin/creator
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_by = relationship("User", foreign_keys=[created_by_user_id])
    
    # Circle settings
    is_active = Column(Boolean, default=True)
    auto_check_in_enabled = Column(Boolean, default=True)
    check_in_frequency_hours = Column(Integer, default=12)  # How often to prompt
    
    # Location tracking
    home_latitude = Column(Float, nullable=True)  # Primary address
    home_longitude = Column(Float, nullable=True)
    home_address = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    members = relationship("FamilyMember", back_populates="circle", cascade="all, delete-orphan")
    check_ins = relationship("SafetyCheckIn", back_populates="circle", cascade="all, delete-orphan")


class FamilyMember(Base):
    """Individual member of a family circle"""
    __tablename__ = "family_members"

    id = Column(Integer, primary_key=True, index=True)
    circle_id = Column(Integer, ForeignKey("family_circles.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Null for children without accounts
    
    # Member info
    name = Column(String(200), nullable=False)
    role = Column(SQLEnum(MemberRole), default=MemberRole.MEMBER)
    
    # Contact information
    phone_number = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True)
    alternate_phone = Column(String(20), nullable=True)
    
    # Privacy and notification preferences
    privacy_level = Column(SQLEnum(PrivacyLevel), default=PrivacyLevel.CONTACTS_ONLY)
    notify_sms = Column(Boolean, default=True)
    notify_whatsapp = Column(Boolean, default=True)
    notify_email = Column(Boolean, default=False)
    notify_push = Column(Boolean, default=True)
    
    # Health/special needs (for evacuation planning)
    has_mobility_issues = Column(Boolean, default=False)
    has_medical_conditions = Column(Boolean, default=False)
    medical_notes = Column(Text, nullable=True)  # Encrypted in production
    
    # Status
    is_active = Column(Boolean, default=True)
    last_check_in_id = Column(Integer, ForeignKey("safety_check_ins.id"), nullable=True)
    
    # Timestamps
    joined_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    circle = relationship("FamilyCircle", back_populates="members")
    user = relationship("User", foreign_keys=[user_id])
    check_ins = relationship("SafetyCheckIn", back_populates="member", foreign_keys="SafetyCheckIn.member_id")
    locations = relationship("MemberLocation", back_populates="member", cascade="all, delete-orphan")


class SafetyCheckIn(Base):
    """Individual safety check-in record"""
    __tablename__ = "safety_check_ins"

    id = Column(Integer, primary_key=True, index=True)
    circle_id = Column(Integer, ForeignKey("family_circles.id"), nullable=False)
    member_id = Column(Integer, ForeignKey("family_members.id"), nullable=False)
    
    # Check-in details
    status = Column(SQLEnum(CheckInStatus), nullable=False)
    message = Column(Text, nullable=True)  # Optional text message
    
    # Location at check-in
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_accuracy = Column(Float, nullable=True)  # meters
    location_address = Column(Text, nullable=True)  # Reverse geocoded
    
    # Context
    is_automated = Column(Boolean, default=False)  # Auto-prompted vs manual
    triggered_by_alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    
    # Response tracking
    response_time_seconds = Column(Integer, nullable=True)  # Time to respond to prompt
    reminder_count = Column(Integer, default=0)  # How many reminders sent
    escalated = Column(Boolean, default=False)  # Escalated to emergency services
    
    # Verification
    verified_by_admin = Column(Boolean, default=False)
    verified_at = Column(DateTime, nullable=True)
    
    # Timestamps
    check_in_time = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    circle = relationship("FamilyCircle", back_populates="check_ins")
    member = relationship("FamilyMember", back_populates="check_ins", foreign_keys=[member_id])


class MemberLocation(Base):
    """Real-time location tracking for family members"""
    __tablename__ = "member_locations"

    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey("family_members.id"), nullable=False)
    
    # Location data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=True)  # meters
    altitude = Column(Float, nullable=True)
    
    # Movement tracking
    speed = Column(Float, nullable=True)  # km/h
    heading = Column(Float, nullable=True)  # degrees
    
    # Device info
    battery_level = Column(Integer, nullable=True)  # percentage
    device_id = Column(String(100), nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    member = relationship("FamilyMember", back_populates="locations")


class CheckInReminder(Base):
    """Scheduled and sent check-in reminders"""
    __tablename__ = "check_in_reminders"

    id = Column(Integer, primary_key=True, index=True)
    circle_id = Column(Integer, ForeignKey("family_circles.id"), nullable=False)
    member_id = Column(Integer, ForeignKey("family_members.id"), nullable=False)
    
    # Reminder details
    reminder_type = Column(String(50))  # "scheduled", "alert_triggered", "overdue_followup"
    triggered_by_alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    
    # Delivery
    channel = Column(SQLEnum(NotificationChannel), nullable=False)
    phone_number = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True)
    message_body = Column(Text)
    
    # Status tracking
    sent_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    responded_at = Column(DateTime, nullable=True)
    response_check_in_id = Column(Integer, ForeignKey("safety_check_ins.id"), nullable=True)
    
    # Escalation
    is_escalated = Column(Boolean, default=False)
    escalation_reason = Column(String(200), nullable=True)
    
    # Timestamps
    scheduled_for = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class EmergencyContact(Base):
    """External emergency contacts (outside circle)"""
    __tablename__ = "emergency_contacts"

    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey("family_members.id"), nullable=False)
    
    # Contact info
    name = Column(String(200), nullable=False)
    relationship = Column(String(100))  # "neighbor", "friend", "doctor"
    phone_number = Column(String(20), nullable=False)
    email = Column(String(255), nullable=True)
    
    # Priority
    priority_order = Column(Integer, default=1)  # 1 = first to contact
    notify_automatically = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
