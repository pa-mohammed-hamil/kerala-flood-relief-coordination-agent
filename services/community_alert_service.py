"""
Community Alert Service
Hyperlocal alert system with geofencing and multi-channel delivery
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import logging
from geopy.distance import geodesic
import json

from backend.models.community_alert import (
    CommunityAlert, AlertSubscription, AlertDelivery, AlertZone,
    AlertSeverity, AlertCategory, DeliveryStatus, SubscriptionPreference
)
from integrations.sms_adapter import send_sms, send_bulk_sms
from integrations.whatsapp_adapter import send_whatsapp_message, send_bulk_whatsapp
from integrations.maps_adapter import reverse_geocode

logger = logging.getLogger(__name__)


class GeofencingService:
    """Handle geographic area calculations and subscriber filtering"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_subscribers_in_radius(
        self,
        center_lat: float,
        center_lng: float,
        radius_km: float,
        category: Optional[AlertCategory] = None
    ) -> List[AlertSubscription]:
        """
        Get all active subscribers within radius of center point
        
        Args:
            center_lat: Center latitude
            center_lng: Center longitude
            radius_km: Radius in kilometers
            category: Optional category filter
        
        Returns:
            List of subscriptions within radius
        """
        # Get all active subscriptions
        query = self.db.query(AlertSubscription).filter(
            AlertSubscription.is_active == True
        )
        
        if category:
            query = query.filter(AlertSubscription.categories.contains([category]))
        
        subscriptions = query.all()
        
        # Filter by distance
        nearby_subscriptions = []
        center_point = (center_lat, center_lng)
        
        for sub in subscriptions:
            if sub.latitude and sub.longitude:
                sub_point = (sub.latitude, sub.longitude)
                distance = geodesic(center_point, sub_point).kilometers
                
                if distance <= radius_km:
                    nearby_subscriptions.append(sub)
        
        logger.info(
            f"Found {len(nearby_subscriptions)} subscribers within {radius_km}km "
            f"of ({center_lat}, {center_lng})"
        )
        
        return nearby_subscriptions
    
    def get_subscribers_in_polygon(
        self,
        polygon_coords: List[Tuple[float, float]],
        category: Optional[AlertCategory] = None
    ) -> List[AlertSubscription]:
        """
        Get subscribers within a polygon area (e.g., district boundaries)
        
        Args:
            polygon_coords: List of (lat, lng) tuples defining polygon
            category: Optional category filter
        
        Returns:
            List of subscriptions within polygon
        """
        from shapely.geometry import Point, Polygon
        
        query = self.db.query(AlertSubscription).filter(
            AlertSubscription.is_active == True
        )
        
        if category:
            query = query.filter(AlertSubscription.categories.contains([category]))
        
        subscriptions = query.all()
        
        # Create polygon
        polygon = Polygon(polygon_coords)
        
        # Filter by polygon containment
        inside_subscriptions = []
        for sub in subscriptions:
            if sub.latitude and sub.longitude:
                point = Point(sub.latitude, sub.longitude)
                if polygon.contains(point):
                    inside_subscriptions.append(sub)
        
        logger.info(
            f"Found {len(inside_subscriptions)} subscribers within polygon area"
        )
        
        return inside_subscriptions
    
    def get_subscribers_in_zone(
        self,
        zone_id: int,
        category: Optional[AlertCategory] = None
    ) -> List[AlertSubscription]:
        """Get subscribers in a predefined alert zone"""
        zone = self.db.query(AlertZone).filter(AlertZone.id == zone_id).first()
        if not zone:
            return []
        
        if zone.geofence_type == "radius":
            return self.get_subscribers_in_radius(
                zone.center_latitude,
                zone.center_longitude,
                zone.radius_km,
                category
            )
        elif zone.geofence_type == "polygon":
            polygon_coords = json.loads(zone.polygon_coordinates)
            return self.get_subscribers_in_polygon(polygon_coords, category)
        
        return []
    
    def get_zones_for_location(
        self,
        latitude: float,
        longitude: float
    ) -> List[AlertZone]:
        """Get all zones that contain a specific location"""
        zones = self.db.query(AlertZone).filter(AlertZone.is_active == True).all()
        
        matching_zones = []
        point = (latitude, longitude)
        
        for zone in zones:
            if zone.geofence_type == "radius":
                center = (zone.center_latitude, zone.center_longitude)
                distance = geodesic(point, center).kilometers
                if distance <= zone.radius_km:
                    matching_zones.append(zone)
            
            elif zone.geofence_type == "polygon":
                from shapely.geometry import Point, Polygon
                polygon_coords = json.loads(zone.polygon_coordinates)
                polygon = Polygon(polygon_coords)
                if polygon.contains(Point(latitude, longitude)):
                    matching_zones.append(zone)
        
        return matching_zones


class AlertTemplateService:
    """Manage alert message templates with localization"""
    
    TEMPLATES = {
        "flood_warning": {
            "en": "🚨 FLOOD WARNING: {severity} flood alert for {location}. Water level rising. {action}. Kerala Flood Relief - {timestamp}",
            "ml": "🚨 വെള്ളപ്പൊക്ക മുന്നറിയിപ്പ്: {location}-ൽ {severity} വെള്ളപ്പൊക്കം. ജലനിരപ്പ് ഉയരുന്നു. {action}. കേരള വെള്ളപ്പൊക്ക റിലീഫ് - {timestamp}",
            "hi": "🚨 बाढ़ चेतावनी: {location} में {severity} बाढ़ अलर्ट। जल स्तर बढ़ रहा है। {action}। केरल बाढ़ राहत - {timestamp}",
            "ta": "🚨 வெள்ளப்பெருக்கு எச்சரிக்கை: {location}-ல் {severity} வெள்ளப்பெருக்கு. நீர் மட்டம் உயர்கிறது. {action}. கேரள வெள்ளப்பெருக்கு நிவாரணம் - {timestamp}"
        },
        "evacuation_order": {
            "en": "⚠️ EVACUATION ORDER: Immediate evacuation required from {location}. Proceed to {shelter}. Bring essentials only. Help available at 1077.",
            "ml": "⚠️ ഒഴിപ്പിക്കൽ ഉത്തരവ്: {location}-ൽ നിന്ന് ഉടൻ ഒഴിയേണ്ടതാണ്. {shelter}-ലേക്ക് പോകുക. അത്യാവശ്യ സാധനങ്ങൾ മാത്രം കൊണ്ടുവരിക. സഹായം 1077.",
            "hi": "⚠️ निकासी आदेश: {location} से तुरंत निकासी आवश्यक। {shelter} की ओर आगे बढ़ें। केवल आवश्यक सामान लाएं। 1077 पर सहायता।",
            "ta": "⚠️ வெளியேற்ற உத்தரவு: {location}-இலிருந்து உடனடி வெளியேற்றம் தேவை. {shelter}-க்கு செல்லுங்கள். அத்தியாவசியங்களை மட்டும் கொண்டு வாருங்கள். 1077-ல் உதவி."
        },
        "shelter_opening": {
            "en": "🏠 SHELTER OPEN: {shelter} is now open in {location}. Capacity: {capacity}. Medical aid, food & water available. GPS: {gps}",
            "ml": "🏠 ഷെൽട്ടർ തുറന്നു: {location}-ൽ {shelter} ഇപ്പോൾ തുറന്നിരിക്കുന്നു. ശേഷി: {capacity}. വൈദ്യസഹായം, ഭക്ഷണം, വെള്ളം ലഭ്യമാണ്. GPS: {gps}",
            "hi": "🏠 आश्रय खुला: {location} में {shelter} अब खुला है। क्षमता: {capacity}। चिकित्सा सहायता, भोजन और पानी उपलब्ध। GPS: {gps}",
            "ta": "🏠 தங்குமிடம் திறப்பு: {location}-ல் {shelter} இப்போது திறந்துள்ளது. திறன்: {capacity}. மருத்துவ உதவி, உணவு மற்றும் தண்ணீர் கிடைக்கும். GPS: {gps}"
        },
        "rescue_alert": {
            "en": "🚁 RESCUE OPS: Rescue teams active in {location}. If trapped, call 1077 or reply HELP. Battery low? Send location SMS.",
            "ml": "🚁 രക്ഷാ പ്രവർത്തനം: {location}-ൽ രക്ഷാ സംഘങ്ങൾ സജീവമാണ്. കുടുങ്ങിയാൽ 1077 വിളിക്കുക അല്ലെങ്കിൽ HELP എന്ന് മറുപടി അയക്കുക. ബാറ്ററി കുറവാണോ? ലൊക്കേഷൻ SMS അയക്കുക.",
            "hi": "🚁 बचाव अभियान: {location} में बचाव दल सक्रिय। फंसे हैं तो 1077 कॉल करें या HELP उत्तर दें। बैटरी कम? स्थान SMS भेजें।",
            "ta": "🚁 மீட்பு நடவடிக்கை: {location}-ல் மீட்புக் குழுக்கள் செயல்பாட்டில். சிக்கிக்கொண்டால் 1077 அழைக்கவும் அல்லது HELP என்று பதிலளிக்கவும். பேட்டரி குறைவு? இருப்பிட SMS அனுப்பவும்."
        },
        "weather_warning": {
            "en": "🌧️ WEATHER ALERT: Heavy rainfall expected in {location} - {rainfall}mm. Stay indoors. Avoid travel. Monitor updates.",
            "ml": "🌧️ കാലാവസ്ഥ മുന്നറിയിപ്പ്: {location}-ൽ കനത്ത മഴ പ്രതീക്ഷിക്കുന്നു - {rainfall}mm. വീടിനുള്ളിൽ തന്നെ ഇരിക്കുക. യാത്ര ഒഴിവാക്കുക. അപ്ഡേറ്റുകൾ നിരീക്ഷിക്കുക.",
            "hi": "🌧️ मौसम अलर्ट: {location} में भारी बारिश की उम्मीद - {rainfall}mm। घर के अंदर रहें। यात्रा से बचें। अपडेट देखें।",
            "ta": "🌧️ வானிலை எச்சரிக்கை: {location}-ல் கனமழை எதிர்பார்க்கப்படுகிறது - {rainfall}mm. வீட்டிற்குள் இருங்கள். பயணத்தைத் தவிர்க்கவும். புதுப்பிப்புகளைக் கண்காணிக்கவும்."
        },
        "all_clear": {
            "en": "✅ ALL CLEAR: Flood alert lifted for {location}. Water receding. Safe to return. Exercise caution. Check home before entering.",
            "ml": "✅ അപകടം ഒഴിഞ്ഞു: {location}-ന്റെ വെള്ളപ്പൊക്ക അലേർട്ട് പിൻവലിച്ചു. വെള്ളം കുറയുന്നു. മടങ്ങാൻ സുരക്ഷിതം. ജാഗ്രത പാലിക്കുക. വീട് പരിശോധിച്ചതിനുശേഷം പ്രവേശിക്കുക.",
            "hi": "✅ सब ठीक: {location} के लिए बाढ़ अलर्ट हटा। पानी कम हो रहा। वापस लौटना सुरक्षित। सावधानी बरतें। प्रवेश से पहले घर जांचें।",
            "ta": "✅ எல்லாம் சரி: {location}-க்கான வெள்ளப்பெருக்கு எச்சரிக்கை நீக்கப்பட்டது. நீர் குறைகிறது. திரும்ப பாதுகாப்பானது. எச்சரிக்கையாக இருங்கள். உள்ளே செல்வதற்கு முன் வீட்டைச் சரிபார்க்கவும்."
        }
    }
    
    def generate_message(
        self,
        template_type: str,
        language: str,
        **kwargs
    ) -> str:
        """Generate localized message from template"""
        if template_type not in self.TEMPLATES:
            raise ValueError(f"Unknown template type: {template_type}")
        
        templates = self.TEMPLATES[template_type]
        template = templates.get(language, templates["en"])  # Fallback to English
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template parameter: {e}")
            return template


class CommunityAlertService:
    """Main service for creating and sending community alerts"""
    
    def __init__(self, db: Session):
        self.db = db
        self.geofencing = GeofencingService(db)
        self.templates = AlertTemplateService()
    
    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        category: AlertCategory,
        created_by_user_id: int,
        location_name: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: Optional[float] = None,
        zone_ids: Optional[List[int]] = None,
        template_type: Optional[str] = None,
        template_params: Optional[Dict] = None,
        expires_at: Optional[datetime] = None,
        action_url: Optional[str] = None
    ) -> CommunityAlert:
        """
        Create a new community alert
        
        Args:
            title: Alert title
            message: Alert message (can be template)
            severity: LOW, MEDIUM, HIGH, CRITICAL
            category: FLOOD, WEATHER, EVACUATION, RESCUE, SHELTER, ALL_CLEAR
            created_by_user_id: Creator user ID
            location_name: Human-readable location
            latitude: Center latitude (for radius alerts)
            longitude: Center longitude (for radius alerts)
            radius_km: Radius in km (for radius alerts)
            zone_ids: List of predefined zone IDs
            template_type: Optional template identifier
            template_params: Parameters for template
            expires_at: Optional expiration time
            action_url: Optional action link
        
        Returns:
            Created CommunityAlert
        """
        alert = CommunityAlert(
            title=title,
            message=message,
            severity=severity,
            category=category,
            created_by_user_id=created_by_user_id,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            zone_ids=zone_ids or [],
            template_type=template_type,
            template_params=template_params,
            expires_at=expires_at,
            action_url=action_url,
            is_active=True
        )
        
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        
        logger.info(f"Created alert {alert.id}: {title} ({severity})")
        
        return alert
    
    def send_alert(
        self,
        alert_id: int,
        dry_run: bool = False
    ) -> Dict:
        """
        Send alert to all subscribers in target area
        
        Args:
            alert_id: Alert ID to send
            dry_run: If True, don't actually send, just count recipients
        
        Returns:
            Delivery statistics
        """
        alert = self.db.query(CommunityAlert).filter(
            CommunityAlert.id == alert_id
        ).first()
        
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")
        
        # Get target subscribers
        subscribers = self._get_target_subscribers(alert)
        
        if dry_run:
            return {
                "alert_id": alert_id,
                "dry_run": True,
                "total_subscribers": len(subscribers),
                "by_channel": self._count_by_channel(subscribers)
            }
        
        # Send to subscribers
        stats = {
            "total_subscribers": len(subscribers),
            "sms_sent": 0,
            "whatsapp_sent": 0,
            "failed": 0,
            "deliveries": []
        }
        
        for subscription in subscribers:
            delivery_result = self._send_to_subscriber(alert, subscription)
            stats["deliveries"].append(delivery_result)
            
            if delivery_result["status"] == "sent":
                if delivery_result["channel"] == "sms":
                    stats["sms_sent"] += 1
                elif delivery_result["channel"] == "whatsapp":
                    stats["whatsapp_sent"] += 1
            else:
                stats["failed"] += 1
        
        # Update alert
        alert.sent_at = datetime.utcnow()
        alert.recipient_count = len(subscribers)
        alert.delivery_success_count = stats["sms_sent"] + stats["whatsapp_sent"]
        alert.delivery_failed_count = stats["failed"]
        self.db.commit()
        
        logger.info(
            f"Sent alert {alert_id} to {len(subscribers)} subscribers: "
            f"{stats['sms_sent']} SMS, {stats['whatsapp_sent']} WhatsApp, "
            f"{stats['failed']} failed"
        )
        
        return stats
    
    def _get_target_subscribers(
        self,
        alert: CommunityAlert
    ) -> List[AlertSubscription]:
        """Get all subscribers who should receive this alert"""
        all_subscribers = []
        
        # Radius-based targeting
        if alert.latitude and alert.longitude and alert.radius_km:
            subscribers = self.geofencing.get_subscribers_in_radius(
                alert.latitude,
                alert.longitude,
                alert.radius_km,
                alert.category
            )
            all_subscribers.extend(subscribers)
        
        # Zone-based targeting
        if alert.zone_ids:
            for zone_id in alert.zone_ids:
                subscribers = self.geofencing.get_subscribers_in_zone(
                    zone_id,
                    alert.category
                )
                all_subscribers.extend(subscribers)
        
        # Deduplicate
        unique_subscribers = list({sub.id: sub for sub in all_subscribers}.values())
        
        return unique_subscribers
    
    def _send_to_subscriber(
        self,
        alert: CommunityAlert,
        subscription: AlertSubscription
    ) -> Dict:
        """Send alert to individual subscriber"""
        # Generate localized message
        if alert.template_type:
            message = self.templates.generate_message(
                alert.template_type,
                subscription.language,
                **(alert.template_params or {})
            )
        else:
            message = alert.message
        
        # Determine channel
        channel = self._get_preferred_channel(subscription, alert.severity)
        
        # Create delivery record
        delivery = AlertDelivery(
            alert_id=alert.id,
            subscription_id=subscription.id,
            phone_number=subscription.phone_number,
            channel=channel,
            message_content=message,
            status=DeliveryStatus.PENDING
        )
        self.db.add(delivery)
        self.db.commit()
        self.db.refresh(delivery)
        
        # Send message
        try:
            if channel == "sms":
                result = send_sms(subscription.phone_number, message)
                delivery.status = DeliveryStatus.SENT
                delivery.sent_at = datetime.utcnow()
                delivery.external_message_id = result.get("message_id")
            
            elif channel == "whatsapp":
                result = send_whatsapp_message(subscription.phone_number, message)
                delivery.status = DeliveryStatus.SENT
                delivery.sent_at = datetime.utcnow()
                delivery.external_message_id = result.get("message_id")
            
            self.db.commit()
            
            return {
                "delivery_id": delivery.id,
                "subscription_id": subscription.id,
                "channel": channel,
                "status": "sent"
            }
            
        except Exception as e:
            delivery.status = DeliveryStatus.FAILED
            delivery.failure_reason = str(e)
            self.db.commit()
            
            logger.error(f"Failed to send alert to {subscription.phone_number}: {e}")
            
            return {
                "delivery_id": delivery.id,
                "subscription_id": subscription.id,
                "channel": channel,
                "status": "failed",
                "error": str(e)
            }
    
    def _get_preferred_channel(
        self,
        subscription: AlertSubscription,
        severity: AlertSeverity
    ) -> str:
        """Determine best channel based on preferences and severity"""
        # Critical alerts always SMS (more reliable)
        if severity == AlertSeverity.CRITICAL:
            return "sms"
        
        # Use subscriber preference
        if subscription.preference == SubscriptionPreference.SMS_ONLY:
            return "sms"
        elif subscription.preference == SubscriptionPreference.WHATSAPP_ONLY:
            return "whatsapp"
        elif subscription.preference == SubscriptionPreference.BOTH:
            # Default to WhatsApp for non-critical (cheaper)
            return "whatsapp" if severity in [AlertSeverity.LOW, AlertSeverity.MEDIUM] else "sms"
        
        return "sms"  # Default fallback
    
    def _count_by_channel(self, subscribers: List[AlertSubscription]) -> Dict:
        """Count subscribers by preferred channel"""
        counts = {"sms": 0, "whatsapp": 0, "both": 0}
        for sub in subscribers:
            if sub.preference == SubscriptionPreference.SMS_ONLY:
                counts["sms"] += 1
            elif sub.preference == SubscriptionPreference.WHATSAPP_ONLY:
                counts["whatsapp"] += 1
            elif sub.preference == SubscriptionPreference.BOTH:
                counts["both"] += 1
        return counts
    
    def subscribe_user(
        self,
        phone_number: str,
        name: str,
        latitude: float,
        longitude: float,
        language: str = "en",
        categories: Optional[List[AlertCategory]] = None,
        preference: SubscriptionPreference = SubscriptionPreference.BOTH
    ) -> AlertSubscription:
        """Subscribe user to community alerts"""
        # Check if already subscribed
        existing = self.db.query(AlertSubscription).filter(
            AlertSubscription.phone_number == phone_number
        ).first()
        
        if existing:
            # Update existing subscription
            existing.name = name
            existing.latitude = latitude
            existing.longitude = longitude
            existing.language = language
            existing.categories = categories or [AlertCategory.FLOOD]
            existing.preference = preference
            existing.is_active = True
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            
            logger.info(f"Updated subscription for {phone_number}")
            return existing
        
        # Create new subscription
        subscription = AlertSubscription(
            phone_number=phone_number,
            name=name,
            latitude=latitude,
            longitude=longitude,
            language=language,
            categories=categories or [AlertCategory.FLOOD],
            preference=preference,
            is_active=True
        )
        self.db.add(subscription)
        self.db.commit()
        self.db.refresh(subscription)
        
        logger.info(f"Created subscription for {phone_number}")
        
        # Send welcome message
        welcome_msg = self.templates.generate_message(
            "subscription_confirmed",
            language,
            name=name,
            categories=", ".join(subscription.categories)
        ) if "subscription_confirmed" in self.templates.TEMPLATES else \
            f"Welcome {name}! You're subscribed to Kerala Flood Relief alerts."
        
        try:
            send_sms(phone_number, welcome_msg)
        except Exception as e:
            logger.error(f"Failed to send welcome SMS: {e}")
        
        return subscription
    
    def unsubscribe_user(self, phone_number: str) -> bool:
        """Unsubscribe user from alerts"""
        subscription = self.db.query(AlertSubscription).filter(
            AlertSubscription.phone_number == phone_number
        ).first()
        
        if subscription:
            subscription.is_active = False
            subscription.updated_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Unsubscribed {phone_number}")
            return True
        
        return False
    
    def get_alert_history(
        self,
        phone_number: str,
        days: int = 7
    ) -> List[Dict]:
        """Get alert history for a subscriber"""
        subscription = self.db.query(AlertSubscription).filter(
            AlertSubscription.phone_number == phone_number
        ).first()
        
        if not subscription:
            return []
        
        since = datetime.utcnow() - timedelta(days=days)
        
        deliveries = self.db.query(AlertDelivery).filter(
            AlertDelivery.subscription_id == subscription.id,
            AlertDelivery.sent_at >= since
        ).order_by(AlertDelivery.sent_at.desc()).all()
        
        history = []
        for delivery in deliveries:
            alert = self.db.query(CommunityAlert).get(delivery.alert_id)
            history.append({
                "alert_id": alert.id,
                "title": alert.title,
                "category": alert.category,
                "severity": alert.severity,
                "sent_at": delivery.sent_at,
                "channel": delivery.channel,
                "status": delivery.status
            })
        
        return history
