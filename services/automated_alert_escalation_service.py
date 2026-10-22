"""
Automated Alert Escalation Service
Real-time monitoring with color-coded alerts (Yellow → Orange → Red) and automated escalation
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import logging
from enum import Enum

from backend.models.flood_alert import FloodAlert
from backend.services.community_alert_service import CommunityAlertService, AlertSeverity, AlertCategory
from integrations.imd_adapter import get_rainfall_forecast, get_current_rainfall
from integrations.cwc_adapter import get_river_level, get_dam_level
from integrations.sms_adapter import send_sms, send_bulk_sms
from integrations.whatsapp_adapter import send_whatsapp_message
from backend.services.notification_service import NotificationService

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Color-coded alert levels"""
    GREEN = "green"      # Normal - No alert
    YELLOW = "yellow"    # Watch - Be alert
    ORANGE = "orange"    # Warning - Prepare to evacuate
    RED = "red"          # Critical - Evacuate immediately


class AlertEscalationEngine:
    """
    Monitors real-time conditions and automatically escalates alerts
    """
    
    # Thresholds for automatic escalation
    RAINFALL_THRESHOLDS = {
        AlertLevel.YELLOW: 50,   # 50mm in 24h
        AlertLevel.ORANGE: 100,  # 100mm in 24h
        AlertLevel.RED: 150      # 150mm in 24h
    }
    
    RIVER_LEVEL_THRESHOLDS = {
        AlertLevel.YELLOW: 0.7,   # 70% of danger level
        AlertLevel.ORANGE: 0.85,  # 85% of danger level
        AlertLevel.RED: 1.0       # At or above danger level
    }
    
    DAM_LEVEL_THRESHOLDS = {
        AlertLevel.YELLOW: 0.8,   # 80% of capacity
        AlertLevel.ORANGE: 0.9,   # 90% of capacity
        AlertLevel.RED: 0.95      # 95% of capacity (imminent release)
    }
    
    def __init__(self, db: Session):
        self.db = db
        self.community_alert_service = CommunityAlertService(db)
        self.notification_service = NotificationService(db)
        self.active_alerts = {}  # Track active alerts by district
    
    def monitor_and_escalate(self, district: str) -> Dict:
        """
        Main monitoring function - checks all parameters and escalates if needed
        
        Returns:
            Status and actions taken
        """
        logger.info(f"Starting automated monitoring for {district}")
        
        # Collect real-time data
        rainfall_data = self._get_rainfall_status(district)
        river_data = self._get_river_status(district)
        dam_data = self._get_dam_status(district)
        
        # Determine alert level
        current_level = self._calculate_alert_level(
            rainfall_data, river_data, dam_data
        )
        
        # Get previous alert level
        previous_level = self.active_alerts.get(district, AlertLevel.GREEN)
        
        # Check if escalation needed
        if self._should_escalate(previous_level, current_level):
            return self._escalate_alert(
                district, previous_level, current_level,
                rainfall_data, river_data, dam_data
            )
        
        return {
            "district": district,
            "current_level": current_level,
            "previous_level": previous_level,
            "escalated": False,
            "data": {
                "rainfall": rainfall_data,
                "rivers": river_data,
                "dams": dam_data
            }
        }
    
    def _get_rainfall_status(self, district: str) -> Dict:
        """Get current and forecast rainfall"""
        try:
            # Get current rainfall (last 24h)
            current = get_current_rainfall(district, hours=24)
            
            # Get forecast (next 24h)
            forecast = get_rainfall_forecast(district, hours=24)
            
            return {
                "current_24h": current.get("total_mm", 0),
                "forecast_24h": forecast.get("total_mm", 0),
                "intensity": current.get("intensity", "light"),
                "status": self._assess_rainfall_level(current.get("total_mm", 0))
            }
        except Exception as e:
            logger.error(f"Error getting rainfall data: {e}")
            return {"current_24h": 0, "forecast_24h": 0, "status": AlertLevel.GREEN}
    
    def _get_river_status(self, district: str) -> List[Dict]:
        """Get status of all rivers in district"""
        try:
            # Get rivers in district
            rivers = self._get_district_rivers(district)
            
            river_statuses = []
            for river in rivers:
                level_data = get_river_level(river["station_id"])
                
                danger_level = river.get("danger_level", 100)
                current_level = level_data.get("water_level", 0)
                percentage = (current_level / danger_level) * 100
                
                river_statuses.append({
                    "name": river["name"],
                    "station": river["station_id"],
                    "current_level": current_level,
                    "danger_level": danger_level,
                    "percentage": percentage,
                    "status": self._assess_river_level(percentage / 100),
                    "trend": level_data.get("trend", "stable")
                })
            
            return river_statuses
        except Exception as e:
            logger.error(f"Error getting river data: {e}")
            return []
    
    def _get_dam_status(self, district: str) -> List[Dict]:
        """Get status of all dams in district"""
        try:
            dams = self._get_district_dams(district)
            
            dam_statuses = []
            for dam in dams:
                level_data = get_dam_level(dam["dam_id"])
                
                capacity = dam.get("capacity", 100)
                current_level = level_data.get("storage", 0)
                percentage = (current_level / capacity) * 100
                
                dam_statuses.append({
                    "name": dam["name"],
                    "current_storage": current_level,
                    "capacity": capacity,
                    "percentage": percentage,
                    "status": self._assess_dam_level(percentage / 100),
                    "gates_open": level_data.get("gates_open", 0),
                    "discharge_rate": level_data.get("discharge_rate", 0)
                })
            
            return dam_statuses
        except Exception as e:
            logger.error(f"Error getting dam data: {e}")
            return []
    
    def _calculate_alert_level(
        self,
        rainfall_data: Dict,
        river_data: List[Dict],
        dam_data: List[Dict]
    ) -> AlertLevel:
        """Calculate overall alert level based on all parameters"""
        levels = []
        
        # Rainfall contribution
        levels.append(rainfall_data.get("status", AlertLevel.GREEN))
        
        # River contribution (worst case)
        if river_data:
            river_levels = [r["status"] for r in river_data]
            levels.append(max(river_levels, key=lambda x: self._level_to_int(x)))
        
        # Dam contribution (worst case)
        if dam_data:
            dam_levels = [d["status"] for d in dam_data]
            levels.append(max(dam_levels, key=lambda x: self._level_to_int(x)))
        
        # Return highest alert level
        return max(levels, key=lambda x: self._level_to_int(x))
    
    def _assess_rainfall_level(self, rainfall_mm: float) -> AlertLevel:
        """Assess alert level based on rainfall"""
        if rainfall_mm >= self.RAINFALL_THRESHOLDS[AlertLevel.RED]:
            return AlertLevel.RED
        elif rainfall_mm >= self.RAINFALL_THRESHOLDS[AlertLevel.ORANGE]:
            return AlertLevel.ORANGE
        elif rainfall_mm >= self.RAINFALL_THRESHOLDS[AlertLevel.YELLOW]:
            return AlertLevel.YELLOW
        return AlertLevel.GREEN
    
    def _assess_river_level(self, percentage: float) -> AlertLevel:
        """Assess alert level based on river level"""
        if percentage >= self.RIVER_LEVEL_THRESHOLDS[AlertLevel.RED]:
            return AlertLevel.RED
        elif percentage >= self.RIVER_LEVEL_THRESHOLDS[AlertLevel.ORANGE]:
            return AlertLevel.ORANGE
        elif percentage >= self.RIVER_LEVEL_THRESHOLDS[AlertLevel.YELLOW]:
            return AlertLevel.YELLOW
        return AlertLevel.GREEN
    
    def _assess_dam_level(self, percentage: float) -> AlertLevel:
        """Assess alert level based on dam level"""
        if percentage >= self.DAM_LEVEL_THRESHOLDS[AlertLevel.RED]:
            return AlertLevel.RED
        elif percentage >= self.DAM_LEVEL_THRESHOLDS[AlertLevel.ORANGE]:
            return AlertLevel.ORANGE
        elif percentage >= self.DAM_LEVEL_THRESHOLDS[AlertLevel.YELLOW]:
            return AlertLevel.YELLOW
        return AlertLevel.GREEN
    
    def _should_escalate(
        self,
        previous: AlertLevel,
        current: AlertLevel
    ) -> bool:
        """Check if alert should be escalated"""
        return self._level_to_int(current) > self._level_to_int(previous)
    
    def _level_to_int(self, level: AlertLevel) -> int:
        """Convert alert level to integer for comparison"""
        mapping = {
            AlertLevel.GREEN: 0,
            AlertLevel.YELLOW: 1,
            AlertLevel.ORANGE: 2,
            AlertLevel.RED: 3
        }
        return mapping.get(level, 0)
    
    def _escalate_alert(
        self,
        district: str,
        previous_level: AlertLevel,
        current_level: AlertLevel,
        rainfall_data: Dict,
        river_data: List[Dict],
        dam_data: List[Dict]
    ) -> Dict:
        """Execute alert escalation"""
        logger.warning(
            f"ESCALATING ALERT for {district}: {previous_level} → {current_level}"
        )
        
        # Update active alerts tracking
        self.active_alerts[district] = current_level
        
        # Generate alert message
        message = self._generate_alert_message(
            district, current_level, rainfall_data, river_data, dam_data
        )
        
        # Map to severity
        severity_mapping = {
            AlertLevel.YELLOW: AlertSeverity.LOW,
            AlertLevel.ORANGE: AlertSeverity.HIGH,
            AlertLevel.RED: AlertSeverity.CRITICAL
        }
        severity = severity_mapping.get(current_level, AlertSeverity.MEDIUM)
        
        # Create and send community alert
        alert = self.community_alert_service.create_alert(
            title=f"{current_level.upper()} ALERT: {district}",
            message=message,
            severity=severity,
            category=AlertCategory.FLOOD,
            created_by_user_id=1,  # System user
            location_name=district,
            template_type="flood_warning"
        )
        
        # Send to subscribers
        delivery_stats = self.community_alert_service.send_alert(alert.id)
        
        # Automated deep calls for RED alerts
        if current_level == AlertLevel.RED:
            self._trigger_deep_calls(district, message)
        
        # Notify emergency coordinators
        self._notify_coordinators(district, current_level, message)
        
        # Trigger automated actions
        automated_actions = self._trigger_automated_actions(
            district, current_level, rainfall_data, river_data, dam_data
        )
        
        return {
            "district": district,
            "escalated": True,
            "previous_level": previous_level,
            "current_level": current_level,
            "alert_id": alert.id,
            "delivery_stats": delivery_stats,
            "automated_actions": automated_actions,
            "timestamp": datetime.utcnow()
        }
    
    def _generate_alert_message(
        self,
        district: str,
        level: AlertLevel,
        rainfall_data: Dict,
        river_data: List[Dict],
        dam_data: List[Dict]
    ) -> str:
        """Generate detailed alert message"""
        messages = {
            AlertLevel.YELLOW: f"⚠️ YELLOW ALERT for {district}\n",
            AlertLevel.ORANGE: f"🟠 ORANGE WARNING for {district}\n",
            AlertLevel.RED: f"🔴 RED ALERT - CRITICAL for {district}\n"
        }
        
        message = messages[level]
        
        # Rainfall info
        if rainfall_data["current_24h"] > 0:
            message += f"Rainfall: {rainfall_data['current_24h']:.1f}mm (24h)\n"
        
        # River info
        critical_rivers = [r for r in river_data if r["status"] in [AlertLevel.ORANGE, AlertLevel.RED]]
        if critical_rivers:
            message += f"Rivers above danger: {', '.join([r['name'] for r in critical_rivers])}\n"
        
        # Dam info
        critical_dams = [d for d in dam_data if d["status"] in [AlertLevel.ORANGE, AlertLevel.RED]]
        if critical_dams:
            message += f"Dams at high level: {', '.join([d['name'] for d in critical_dams])}\n"
        
        # Action instructions
        if level == AlertLevel.RED:
            message += "\n🚨 EVACUATE IMMEDIATELY\nMove to high ground. Do not wait."
        elif level == AlertLevel.ORANGE:
            message += "\n⚠️ PREPARE TO EVACUATE\nPack essentials. Stay alert."
        elif level == AlertLevel.YELLOW:
            message += "\n⚠️ BE ALERT\nMonitor updates. Avoid low-lying areas."
        
        message += f"\n\nCall 1077 for help\nTime: {datetime.now().strftime('%d-%b %I:%M %p')}"
        
        return message
    
    def _trigger_deep_calls(self, district: str, message: str):
        """
        Automated voice calls for RED alerts
        Using Text-to-Speech for automated calls
        """
        logger.critical(f"Triggering automated deep calls for {district}")
        
        try:
            # Get high-priority contacts in district
            priority_contacts = self._get_priority_contacts(district)
            
            # Voice call message (shorter for calls)
            voice_message = (
                f"This is Kerala Flood Relief automated alert. "
                f"Red alert issued for {district}. "
                f"Immediate evacuation required. "
                f"Move to high ground now. "
                f"Call 1077 for emergency assistance."
            )
            
            for contact in priority_contacts:
                try:
                    # Would use Twilio Voice API or similar
                    # from twilio.rest import Client
                    # client.calls.create(
                    #     to=contact['phone'],
                    #     from_='+1234567890',
                    #     twiml=f'<Response><Say voice="alice" language="en-IN">{voice_message}</Say></Response>'
                    # )
                    
                    # For now, log
                    logger.info(f"AUTO CALL scheduled to {contact['phone']}: {contact['name']}")
                    
                except Exception as e:
                    logger.error(f"Failed to place call to {contact['phone']}: {e}")
            
            logger.info(f"Automated calls initiated for {len(priority_contacts)} contacts")
            
        except Exception as e:
            logger.error(f"Error in deep call automation: {e}")
    
    def _notify_coordinators(self, district: str, level: AlertLevel, message: str):
        """Notify emergency coordinators via multiple channels"""
        try:
            coordinators = self._get_district_coordinators(district)
            
            for coordinator in coordinators:
                # SMS for immediate notification
                send_sms(
                    coordinator["phone"],
                    f"SYSTEM ALERT: {level.upper()} escalation for {district}. Check dashboard immediately."
                )
                
                # WhatsApp for detailed info
                send_whatsapp_message(
                    coordinator["phone"],
                    f"🚨 AUTOMATED ESCALATION\n\n{message}\n\nAction required on dashboard."
                )
            
            logger.info(f"Notified {len(coordinators)} coordinators for {district}")
            
        except Exception as e:
            logger.error(f"Error notifying coordinators: {e}")
    
    def _trigger_automated_actions(
        self,
        district: str,
        level: AlertLevel,
        rainfall_data: Dict,
        river_data: List[Dict],
        dam_data: List[Dict]
    ) -> List[str]:
        """Trigger automated response actions"""
        actions = []
        
        if level == AlertLevel.YELLOW:
            # Yellow: Monitor and prepare
            actions.extend([
                "activate_monitoring_stations",
                "alert_rescue_teams_standby",
                "prepare_shelters"
            ])
        
        elif level == AlertLevel.ORANGE:
            # Orange: Prepare for evacuation
            actions.extend([
                "activate_all_monitoring",
                "deploy_rescue_teams",
                "open_emergency_shelters",
                "activate_relief_supplies",
                "alert_hospitals"
            ])
        
        elif level == AlertLevel.RED:
            # Red: Immediate action
            actions.extend([
                "full_emergency_activation",
                "evacuate_low_lying_areas",
                "all_shelters_emergency_mode",
                "all_rescue_teams_deployed",
                "request_military_assistance",
                "activate_disaster_response_fund"
            ])
        
        # Execute actions
        for action in actions:
            try:
                self._execute_action(district, action)
                logger.info(f"Executed automated action: {action} for {district}")
            except Exception as e:
                logger.error(f"Failed to execute {action}: {e}")
        
        return actions
    
    def _execute_action(self, district: str, action: str):
        """Execute a specific automated action"""
        # Placeholder - would integrate with various systems
        action_handlers = {
            "activate_monitoring_stations": lambda: logger.info("Monitoring stations activated"),
            "alert_rescue_teams_standby": lambda: logger.info("Rescue teams on standby"),
            "open_emergency_shelters": lambda: logger.info("Emergency shelters opened"),
            "full_emergency_activation": lambda: logger.critical("FULL EMERGENCY MODE ACTIVATED"),
            # ... other actions
        }
        
        handler = action_handlers.get(action, lambda: logger.info(f"Action: {action}"))
        handler()
    
    def _get_priority_contacts(self, district: str) -> List[Dict]:
        """Get priority contacts for automated calls"""
        # Placeholder - would query database
        return [
            {"name": "District Collector", "phone": "+911234567890", "priority": 1},
            {"name": "Police Commissioner", "phone": "+911234567891", "priority": 2},
            {"name": "Fire Chief", "phone": "+911234567892", "priority": 3}
        ]
    
    def _get_district_coordinators(self, district: str) -> List[Dict]:
        """Get emergency coordinators for district"""
        # Placeholder - would query database
        return [
            {"name": "Emergency Coordinator", "phone": "+911234567890"},
            {"name": "Relief Officer", "phone": "+911234567891"}
        ]
    
    def _get_district_rivers(self, district: str) -> List[Dict]:
        """Get rivers in district"""
        # Placeholder - would query database
        district_rivers = {
            "Ernakulam": [
                {"name": "Periyar", "station_id": "PER-001", "danger_level": 5.5},
                {"name": "Muvattupuzha", "station_id": "MUV-001", "danger_level": 4.0}
            ],
            "Thrissur": [
                {"name": "Bharathapuzha", "station_id": "BHA-001", "danger_level": 6.0}
            ]
        }
        return district_rivers.get(district, [])
    
    def _get_district_dams(self, district: str) -> List[Dict]:
        """Get dams in district"""
        # Placeholder - would query database
        district_dams = {
            "Idukki": [
                {"name": "Idukki Dam", "dam_id": "IDK-001", "capacity": 1996.0},
                {"name": "Mullaperiyar", "dam_id": "MUL-001", "capacity": 443.0}
            ]
        }
        return district_dams.get(district, [])


class ContinuousMonitoringService:
    """
    Background service that continuously monitors all districts
    Runs every 15 minutes
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.escalation_engine = AlertEscalationEngine(db)
    
    def monitor_all_districts(self) -> Dict:
        """Monitor all Kerala districts"""
        kerala_districts = [
            "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha",
            "Kottayam", "Idukki", "Ernakulam", "Thrissur",
            "Palakkad", "Malappuram", "Kozhikode", "Wayanad",
            "Kannur", "Kasaragod"
        ]
        
        results = {
            "timestamp": datetime.utcnow(),
            "districts_monitored": len(kerala_districts),
            "alerts_by_level": {
                "green": 0,
                "yellow": 0,
                "orange": 0,
                "red": 0
            },
            "escalations": [],
            "district_status": []
        }
        
        for district in kerala_districts:
            try:
                status = self.escalation_engine.monitor_and_escalate(district)
                
                results["district_status"].append(status)
                
                # Count by level
                level = status["current_level"]
                results["alerts_by_level"][level] += 1
                
                # Track escalations
                if status.get("escalated"):
                    results["escalations"].append({
                        "district": district,
                        "level": status["current_level"],
                        "alert_id": status.get("alert_id")
                    })
                
            except Exception as e:
                logger.error(f"Error monitoring {district}: {e}")
        
        logger.info(
            f"Monitoring complete: {results['alerts_by_level']['red']} RED, "
            f"{results['alerts_by_level']['orange']} ORANGE, "
            f"{results['alerts_by_level']['yellow']} YELLOW alerts"
        )
        
        return results
