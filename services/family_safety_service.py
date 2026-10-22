"""
Family Safety Service
Core business logic for emergency contact network and family safety check-ins
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import logging
from geopy.distance import geodesic

from backend.models.family_safety import (
    FamilyCircle, FamilyMember, SafetyCheckIn, MemberLocation,
    CheckInReminder, EmergencyContact,
    PrivacyLevel, MemberRole, CheckInStatus, NotificationChannel
)
from integrations.sms_adapter import send_sms
from integrations.whatsapp_adapter import send_whatsapp_message
from integrations.email_adapter import send_email

logger = logging.getLogger(__name__)


class FamilyCircleService:
    """Manages family circles and member relationships"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_circle(
        self,
        name: str,
        created_by_user_id: int,
        description: Optional[str] = None,
        home_latitude: Optional[float] = None,
        home_longitude: Optional[float] = None,
        home_address: Optional[str] = None,
        check_in_frequency_hours: int = 12
    ) -> FamilyCircle:
        """Create a new family circle"""
        circle = FamilyCircle(
            name=name,
            description=description,
            created_by_user_id=created_by_user_id,
            home_latitude=home_latitude,
            home_longitude=home_longitude,
            home_address=home_address,
            check_in_frequency_hours=check_in_frequency_hours,
            auto_check_in_enabled=True,
            is_active=True
        )
        self.db.add(circle)
        self.db.commit()
        self.db.refresh(circle)
        
        # Add creator as admin member
        self.add_member(
            circle_id=circle.id,
            user_id=created_by_user_id,
            name="Circle Admin",
            role=MemberRole.ADMIN
        )
        
        logger.info(f"Created family circle {circle.id}: {name}")
        return circle
    
    def add_member(
        self,
        circle_id: int,
        name: str,
        role: MemberRole = MemberRole.MEMBER,
        user_id: Optional[int] = None,
        phone_number: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs
    ) -> FamilyMember:
        """Add member to circle"""
        member = FamilyMember(
            circle_id=circle_id,
            user_id=user_id,
            name=name,
            role=role,
            phone_number=phone_number,
            email=email,
            **kwargs
        )
        self.db.add(member)
        self.db.commit()
        self.db.refresh(member)
        
        logger.info(f"Added member {member.id} to circle {circle_id}")
        return member
    
    def get_circle_members(
        self,
        circle_id: int,
        active_only: bool = True
    ) -> List[FamilyMember]:
        """Get all members of a circle"""
        query = self.db.query(FamilyMember).filter(
            FamilyMember.circle_id == circle_id
        )
        if active_only:
            query = query.filter(FamilyMember.is_active == True)
        
        return query.all()
    
    def get_member_circles(self, user_id: int) -> List[FamilyCircle]:
        """Get all circles a user belongs to"""
        members = self.db.query(FamilyMember).filter(
            FamilyMember.user_id == user_id,
            FamilyMember.is_active == True
        ).all()
        
        circle_ids = [m.circle_id for m in members]
        circles = self.db.query(FamilyCircle).filter(
            FamilyCircle.id.in_(circle_ids),
            FamilyCircle.is_active == True
        ).all()
        
        return circles
    
    def update_member_role(
        self,
        member_id: int,
        new_role: MemberRole,
        requester_user_id: int
    ) -> bool:
        """Update member role (admin only)"""
        # Verify requester is admin
        member = self.db.query(FamilyMember).get(member_id)
        if not member:
            return False
        
        requester = self.db.query(FamilyMember).filter(
            FamilyMember.circle_id == member.circle_id,
            FamilyMember.user_id == requester_user_id,
            FamilyMember.role == MemberRole.ADMIN
        ).first()
        
        if not requester:
            logger.warning(f"Unauthorized role update attempt by user {requester_user_id}")
            return False
        
        member.role = new_role
        member.updated_at = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"Updated member {member_id} role to {new_role}")
        return True
    
    def remove_member(
        self,
        member_id: int,
        requester_user_id: int
    ) -> bool:
        """Remove member from circle (soft delete)"""
        member = self.db.query(FamilyMember).get(member_id)
        if not member:
            return False
        
        # Verify requester is admin or the member themselves
        requester = self.db.query(FamilyMember).filter(
            FamilyMember.circle_id == member.circle_id,
            FamilyMember.user_id == requester_user_id,
            or_(
                FamilyMember.role == MemberRole.ADMIN,
                FamilyMember.id == member_id
            )
        ).first()
        
        if not requester:
            return False
        
        member.is_active = False
        member.updated_at = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"Removed member {member_id} from circle")
        return True


class CheckInService:
    """Handles safety check-ins and location updates"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def submit_check_in(
        self,
        member_id: int,
        status: CheckInStatus,
        message: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_accuracy: Optional[float] = None,
        is_automated: bool = False,
        triggered_by_alert_id: Optional[int] = None
    ) -> SafetyCheckIn:
        """Submit a safety check-in"""
        member = self.db.query(FamilyMember).get(member_id)
        if not member:
            raise ValueError(f"Member {member_id} not found")
        
        check_in = SafetyCheckIn(
            circle_id=member.circle_id,
            member_id=member_id,
            status=status,
            message=message,
            latitude=latitude,
            longitude=longitude,
            location_accuracy=location_accuracy,
            is_automated=is_automated,
            triggered_by_alert_id=triggered_by_alert_id,
            check_in_time=datetime.utcnow()
        )
        self.db.add(check_in)
        
        # Update member's last check-in
        member.last_check_in_id = check_in.id
        member.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(check_in)
        
        # Update location if provided
        if latitude and longitude:
            self.update_member_location(
                member_id=member_id,
                latitude=latitude,
                longitude=longitude,
                accuracy=location_accuracy
            )
        
        # Check if escalation needed
        if status in [CheckInStatus.UNSAFE, CheckInStatus.NEEDS_HELP]:
            self._escalate_check_in(check_in)
        
        logger.info(f"Check-in submitted for member {member_id}: {status}")
        return check_in
    
    def update_member_location(
        self,
        member_id: int,
        latitude: float,
        longitude: float,
        accuracy: Optional[float] = None,
        speed: Optional[float] = None,
        heading: Optional[float] = None,
        battery_level: Optional[int] = None
    ) -> MemberLocation:
        """Update real-time location for member"""
        location = MemberLocation(
            member_id=member_id,
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            speed=speed,
            heading=heading,
            battery_level=battery_level,
            timestamp=datetime.utcnow()
        )
        self.db.add(location)
        self.db.commit()
        self.db.refresh(location)
        
        return location
    
    def get_circle_status(
        self,
        circle_id: int,
        hours: int = 24
    ) -> Dict:
        """Get status overview for entire circle"""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        members = self.db.query(FamilyMember).filter(
            FamilyMember.circle_id == circle_id,
            FamilyMember.is_active == True
        ).all()
        
        member_statuses = []
        for member in members:
            # Get latest check-in
            latest_check_in = self.db.query(SafetyCheckIn).filter(
                SafetyCheckIn.member_id == member.id,
                SafetyCheckIn.check_in_time >= since
            ).order_by(desc(SafetyCheckIn.check_in_time)).first()
            
            # Get latest location
            latest_location = self.db.query(MemberLocation).filter(
                MemberLocation.member_id == member.id
            ).order_by(desc(MemberLocation.timestamp)).first()
            
            member_statuses.append({
                "member_id": member.id,
                "name": member.name,
                "role": member.role,
                "last_check_in": latest_check_in.status if latest_check_in else None,
                "last_check_in_time": latest_check_in.check_in_time if latest_check_in else None,
                "last_location": {
                    "latitude": latest_location.latitude if latest_location else None,
                    "longitude": latest_location.longitude if latest_location else None,
                    "timestamp": latest_location.timestamp if latest_location else None
                } if latest_location else None,
                "needs_check_in": not latest_check_in or (
                    datetime.utcnow() - latest_check_in.check_in_time
                ).total_seconds() / 3600 > 12
            })
        
        return {
            "circle_id": circle_id,
            "total_members": len(members),
            "member_statuses": member_statuses,
            "checked_in_24h": sum(1 for m in member_statuses if m["last_check_in"]),
            "needs_check_in": sum(1 for m in member_statuses if m["needs_check_in"]),
            "at_risk": sum(
                1 for m in member_statuses
                if m["last_check_in"] in [CheckInStatus.UNSAFE, CheckInStatus.NEEDS_HELP]
            )
        }
    
    def get_member_history(
        self,
        member_id: int,
        days: int = 7
    ) -> List[SafetyCheckIn]:
        """Get check-in history for member"""
        since = datetime.utcnow() - timedelta(days=days)
        
        check_ins = self.db.query(SafetyCheckIn).filter(
            SafetyCheckIn.member_id == member_id,
            SafetyCheckIn.check_in_time >= since
        ).order_by(desc(SafetyCheckIn.check_in_time)).all()
        
        return check_ins
    
    def _escalate_check_in(self, check_in: SafetyCheckIn):
        """Escalate unsafe check-in to emergency services"""
        check_in.escalated = True
        self.db.commit()
        
        # TODO: Create SOS request or notify emergency coordinator
        logger.warning(
            f"ESCALATION: Member {check_in.member_id} reported {check_in.status}"
        )


class NotificationOrchestrator:
    """Manages multi-channel notifications to family members"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def send_check_in_request(
        self,
        member_id: int,
        reminder_type: str = "scheduled",
        triggered_by_alert_id: Optional[int] = None
    ) -> CheckInReminder:
        """Send check-in request via preferred channels"""
        member = self.db.query(FamilyMember).get(member_id)
        if not member:
            raise ValueError(f"Member {member_id} not found")
        
        # Determine channels to use
        channels = []
        if member.notify_sms and member.phone_number:
            channels.append((NotificationChannel.SMS, member.phone_number))
        if member.notify_whatsapp and member.phone_number:
            channels.append((NotificationChannel.WHATSAPP, member.phone_number))
        if member.notify_email and member.email:
            channels.append((NotificationChannel.EMAIL, member.email))
        
        if not channels:
            logger.warning(f"No notification channels available for member {member_id}")
            return None
        
        # Send via primary channel (SMS preferred for critical)
        primary_channel, contact = channels[0]
        message = self._generate_check_in_message(member, reminder_type)
        
        reminder = CheckInReminder(
            circle_id=member.circle_id,
            member_id=member_id,
            reminder_type=reminder_type,
            triggered_by_alert_id=triggered_by_alert_id,
            channel=primary_channel,
            phone_number=member.phone_number,
            email=member.email,
            message_body=message,
            scheduled_for=datetime.utcnow()
        )
        self.db.add(reminder)
        self.db.commit()
        self.db.refresh(reminder)
        
        # Send notification
        try:
            if primary_channel == NotificationChannel.SMS:
                send_sms(member.phone_number, message)
            elif primary_channel == NotificationChannel.WHATSAPP:
                send_whatsapp_message(member.phone_number, message)
            elif primary_channel == NotificationChannel.EMAIL:
                send_email(member.email, "Safety Check-In Request", message)
            
            reminder.sent_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Sent check-in request to member {member_id} via {primary_channel}")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
        
        return reminder
    
    def send_circle_update(
        self,
        circle_id: int,
        subject: str,
        message: str,
        exclude_member_ids: Optional[List[int]] = None
    ):
        """Send update to all circle members"""
        members = self.db.query(FamilyMember).filter(
            FamilyMember.circle_id == circle_id,
            FamilyMember.is_active == True
        ).all()
        
        if exclude_member_ids:
            members = [m for m in members if m.id not in exclude_member_ids]
        
        for member in members:
            # WhatsApp for updates (less intrusive than SMS)
            if member.notify_whatsapp and member.phone_number:
                try:
                    send_whatsapp_message(member.phone_number, message)
                except Exception as e:
                    logger.error(f"Failed WhatsApp to member {member.id}: {e}")
            
            # Email for detailed updates
            if member.notify_email and member.email:
                try:
                    send_email(member.email, subject, message)
                except Exception as e:
                    logger.error(f"Failed email to member {member.id}: {e}")
        
        logger.info(f"Sent circle update to {len(members)} members")
    
    def _generate_check_in_message(
        self,
        member: FamilyMember,
        reminder_type: str
    ) -> str:
        """Generate localized check-in message"""
        # Malayalam support
        if reminder_type == "alert_triggered":
            return (
                f"Hi {member.name}, flood alert issued for your area. "
                f"Please confirm your safety status: Reply SAFE, UNSAFE, or HELP. "
                f"Kerala Flood Relief"
            )
        else:
            return (
                f"Hi {member.name}, scheduled safety check-in. "
                f"Reply: SAFE, UNSAFE, EVACUATING, or HELP. "
                f"Kerala Flood Relief"
            )


class AutomationEngine:
    """Automated workflows for family safety"""
    
    def __init__(self, db: Session):
        self.db = db
        self.notification_service = NotificationOrchestrator(db)
        self.check_in_service = CheckInService(db)
    
    def process_alert_triggered_check_ins(self, alert_id: int, affected_districts: List[str]):
        """Send check-in requests to all families in affected areas"""
        # Find circles in affected areas
        # For now, send to all active circles (would filter by district in production)
        circles = self.db.query(FamilyCircle).filter(
            FamilyCircle.is_active == True,
            FamilyCircle.auto_check_in_enabled == True
        ).all()
        
        sent_count = 0
        for circle in circles:
            members = self.db.query(FamilyMember).filter(
                FamilyMember.circle_id == circle.id,
                FamilyMember.is_active == True
            ).all()
            
            for member in members:
                try:
                    self.notification_service.send_check_in_request(
                        member_id=member.id,
                        reminder_type="alert_triggered",
                        triggered_by_alert_id=alert_id
                    )
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send alert check-in to member {member.id}: {e}")
        
        logger.info(f"Sent {sent_count} alert-triggered check-in requests")
        return sent_count
    
    def process_scheduled_check_ins(self):
        """Send scheduled check-in reminders"""
        circles = self.db.query(FamilyCircle).filter(
            FamilyCircle.is_active == True,
            FamilyCircle.auto_check_in_enabled == True
        ).all()
        
        sent_count = 0
        for circle in circles:
            members = self.db.query(FamilyMember).filter(
                FamilyMember.circle_id == circle.id,
                FamilyMember.is_active == True
            ).all()
            
            for member in members:
                # Check if check-in needed
                if self._needs_check_in(member, circle.check_in_frequency_hours):
                    try:
                        self.notification_service.send_check_in_request(
                            member_id=member.id,
                            reminder_type="scheduled"
                        )
                        sent_count += 1
                    except Exception as e:
                        logger.error(f"Failed scheduled check-in for member {member.id}: {e}")
        
        logger.info(f"Sent {sent_count} scheduled check-in reminders")
        return sent_count
    
    def process_overdue_check_ins(self):
        """Detect and escalate overdue check-ins"""
        threshold = datetime.utcnow() - timedelta(hours=24)
        
        # Find reminders sent but not responded to
        overdue_reminders = self.db.query(CheckInReminder).filter(
            CheckInReminder.sent_at < threshold,
            CheckInReminder.responded_at == None,
            CheckInReminder.is_escalated == False
        ).all()
        
        escalated_count = 0
        for reminder in overdue_reminders:
            if reminder.reminder_count >= 3:
                # Escalate after 3 failed attempts
                self._escalate_overdue_member(reminder)
                reminder.is_escalated = True
                reminder.escalation_reason = "No response after 3 reminders"
                escalated_count += 1
            else:
                # Send another reminder
                try:
                    self.notification_service.send_check_in_request(
                        member_id=reminder.member_id,
                        reminder_type="overdue_followup"
                    )
                    reminder.reminder_count += 1
                except Exception as e:
                    logger.error(f"Failed overdue reminder for member {reminder.member_id}: {e}")
        
        self.db.commit()
        logger.info(f"Processed {len(overdue_reminders)} overdue check-ins, escalated {escalated_count}")
        return escalated_count
    
    def assess_circle_risk(self, circle_id: int) -> Dict:
        """Assess overall risk level for family circle"""
        status = self.check_in_service.get_circle_status(circle_id, hours=24)
        
        risk_score = 0
        risk_factors = []
        
        # No recent check-ins
        if status["needs_check_in"] > status["total_members"] * 0.5:
            risk_score += 30
            risk_factors.append(f"{status['needs_check_in']} members need check-in")
        
        # Members reporting unsafe
        if status["at_risk"] > 0:
            risk_score += 50
            risk_factors.append(f"{status['at_risk']} members at risk")
        
        # Low check-in rate
        if status["checked_in_24h"] < status["total_members"] * 0.3:
            risk_score += 20
            risk_factors.append("Low check-in response rate")
        
        risk_level = "LOW"
        if risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        
        return {
            "circle_id": circle_id,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "status": status
        }
    
    def _needs_check_in(self, member: FamilyMember, frequency_hours: int) -> bool:
        """Check if member needs check-in reminder"""
        if not member.last_check_in_id:
            return True
        
        last_check_in = self.db.query(SafetyCheckIn).get(member.last_check_in_id)
        if not last_check_in:
            return True
        
        hours_since = (datetime.utcnow() - last_check_in.check_in_time).total_seconds() / 3600
        return hours_since >= frequency_hours
    
    def _escalate_overdue_member(self, reminder: CheckInReminder):
        """Escalate overdue member to circle admins and emergency contacts"""
        member = self.db.query(FamilyMember).get(reminder.member_id)
        if not member:
            return
        
        # Notify circle admins
        admins = self.db.query(FamilyMember).filter(
            FamilyMember.circle_id == member.circle_id,
            FamilyMember.role == MemberRole.ADMIN,
            FamilyMember.is_active == True
        ).all()
        
        message = (
            f"URGENT: {member.name} has not responded to 3 safety check-in requests. "
            f"Last known status: {reminder.reminder_type}. Please verify their safety."
        )
        
        for admin in admins:
            if admin.notify_sms and admin.phone_number:
                try:
                    send_sms(admin.phone_number, message)
                except Exception as e:
                    logger.error(f"Failed escalation SMS to admin: {e}")
        
        logger.warning(f"Escalated overdue member {member.id} to admins")
