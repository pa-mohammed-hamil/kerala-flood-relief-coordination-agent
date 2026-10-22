"""
Community Alert Models
Database models for hyperlocal alert system
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum as SQLEnum, JSON
from sqlalchemy.orm import relationship
from backend.database import Base
import enum


class AlertSeverity(str, enum.Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertCategory(str, enum.Enum):
    """Alert categories"""
    FLOOD = "flood"
    WEATHER = "weather"
    EVACUATION = "evacuation"
    RESCUE = "rescue"
    SHELTER = "shelter"
    ALL_CLEAR = "all_clear"
    GENERAL = "general"


class DeliveryStatus(str, enum.Enum):
    """Message delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


class SubscriptionPreference(str, enum.Enum):
    """Channel preferences"""
    SMS_ONLY = "sms_only"
    WHATSAPP_ONLY = "whatsapp_only"
    BOTH = "both"


class CommunityAlert(Base):
    """Community alert broadcast"""
    __tablename__ = "community_alerts"

    id = Column(Integer, primary_key=True, index=True)
    
    # Content
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(SQLEnum(AlertSeverity), nullable=False, index=True)
    category = Column(SQLEnum(AlertCategory), nullable=False, index=True)
    
    # Creator
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_by = relationship("User", foreign_keys=[created_by_user_id])
    
    # Targeting
    location_name = Column(String(500), nullable=False)  # "Aluva, Ernakulam"
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    radius_km = Column(Float, nullable=True)  # For radius-based alerts
    zone_ids = Column(JSON, nullable=True)  # List of predefined zone IDs
    
    # Template support
    template_type = Column(String(100), nullable=True)  # "flood_warning", "evacuation_order"
    template_params = Column(JSON, nullable=True)  # Template parameters
    
    # Metadata
    expires_at = Column(DateTime, nullable=True, index=True)
    action_url = Column(String(500), nullable=True)  # Link for more info
    is_active = Column(Boolean, default=True, index=True)
    
    # Delivery tracking
    sent_at = Column(DateTime, nullable=True)
    recipient_count = Column(Integer, default=0)
    delivery_success_count = Column(Integer, default=0)
    delivery_failed_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    deliveries = relationship("AlertDelivery", back_populates="alert", cascade="all, delete-orphan")


class AlertSubscription(Base):
    """User subscription to community alerts"""
    __tablename__ = "alert_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    
    # Contact info
    phone_number = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(255), nullable=True)
    
    # Location (for geofencing)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    address = Column(Text, nullable=True)
    
    # Preferences
    language = Column(String(10), default="en")  # en, ml, hi, ta
    categories = Column(JSON, nullable=False)  # List of AlertCategory values
    preference = Column(SQLEnum(SubscriptionPreference), default=SubscriptionPreference.BOTH)
    
    # Notification schedule
    quiet_hours_start = Column(Integer, nullable=True)  # 22 (10 PM)
    quiet_hours_end = Column(Integer, nullable=True)  # 6 (6 AM)
    emergency_override_quiet = Column(Boolean, default=True)  # Send critical even during quiet hours
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    verified = Column(Boolean, default=False)
    verification_code = Column(String(10), nullable=True)
    
    # Timestamps
    subscribed_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_alert_at = Column(DateTime, nullable=True)
    
    # Relationships
    deliveries = relationship("AlertDelivery", back_populates="subscription", cascade="all, delete-orphan")


class AlertDelivery(Base):
    """Individual alert delivery record"""
    __tablename__ = "alert_deliveries"

    id = Column(Integer, primary_key=True, index=True)
    
    # Links
    alert_id = Column(Integer, ForeignKey("community_alerts.id"), nullable=False, index=True)
    subscription_id = Column(Integer, ForeignKey("alert_subscriptions.id"), nullable=False, index=True)
    
    # Delivery details
    phone_number = Column(String(20), nullable=False)
    channel = Column(String(20), nullable=False)  # "sms", "whatsapp"
    message_content = Column(Text, nullable=False)
    
    # Status tracking
    status = Column(SQLEnum(DeliveryStatus), default=DeliveryStatus.PENDING, index=True)
    sent_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    read_at = Column(DateTime, nullable=True)
    
    # External tracking
    external_message_id = Column(String(200), nullable=True)  # Twilio/WhatsApp message ID
    failure_reason = Column(Text, nullable=True)
    
    # Cost tracking
    cost_units = Column(Float, nullable=True)  # SMS credits or API calls
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    alert = relationship("CommunityAlert", back_populates="deliveries")
    subscription = relationship("AlertSubscription", back_populates="deliveries")


class AlertZone(Base):
    """Predefined geographic zones for alerts"""
    __tablename__ = "alert_zones"

    id = Column(Integer, primary_key=True, index=True)
    
    # Zone definition
    name = Column(String(200), nullable=False, index=True)  # "Aluva Urban Area"
    description = Column(Text, nullable=True)
    district = Column(String(100), nullable=False, index=True)
    
    # Geofence
    geofence_type = Column(String(50), nullable=False)  # "radius", "polygon"
    center_latitude = Column(Float, nullable=True)  # For radius
    center_longitude = Column(Float, nullable=True)  # For radius
    radius_km = Column(Float, nullable=True)  # For radius
    polygon_coordinates = Column(Text, nullable=True)  # JSON array of [lat, lng] for polygon
    
    # Metadata
    population = Column(Integer, nullable=True)
    subscriber_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AlertFeedback(Base):
    """User feedback on alerts"""
    __tablename__ = "alert_feedback"

    id = Column(Integer, primary_key=True, index=True)
    
    # Links
    alert_id = Column(Integer, ForeignKey("community_alerts.id"), nullable=False)
    subscription_id = Column(Integer, ForeignKey("alert_subscriptions.id"), nullable=False)
    
    # Feedback
    rating = Column(Integer, nullable=True)  # 1-5 stars
    was_helpful = Column(Boolean, nullable=True)
    was_timely = Column(Boolean, nullable=True)
    was_accurate = Column(Boolean, nullable=True)
    comment = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
