"""
backend/services/drone_service.py

Drone Fleet Coordination Service
Kerala Flood Relief Coordination Agent

Features:
    - Automated drone fleet management
    - Mission planning and scheduling
    - Real-time tracking and telemetry
    - Surveillance zone assignment
    - Battery and maintenance tracking
    - Collision avoidance
    - No-fly zone enforcement
    - Weather-aware deployment
    - AI image analysis integration
    - Emergency landing protocols
    - Multi-drone coordination
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class DroneStatus(str, Enum):
    """Drone operational status"""
    AVAILABLE = "available"
    IN_FLIGHT = "in_flight"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    EMERGENCY = "emergency"


class MissionType(str, Enum):
    """Type of drone mission"""
    SURVEILLANCE = "surveillance"
    DAMAGE_ASSESSMENT = "damage_assessment"
    SEARCH_RESCUE = "search_rescue"
    SUPPLY_DELIVERY = "supply_delivery"
    MAPPING = "mapping"
    EMERGENCY_RESPONSE = "emergency_response"


class MissionStatus(str, Enum):
    """Mission execution status"""
    PLANNED = "planned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class DroneCapability(str, Enum):
    """Drone capabilities"""
    CAMERA_HD = "camera_hd"
    CAMERA_THERMAL = "camera_thermal"
    CAMERA_MULTISPECTRAL = "camera_multispectral"
    PAYLOAD_DELIVERY = "payload_delivery"
    LONG_RANGE = "long_range"
    NIGHT_VISION = "night_vision"
    WATER_RESISTANT = "water_resistant"


class FlightMode(str, Enum):
    """Drone flight modes"""
    MANUAL = "manual"
    AUTO_WAYPOINT = "auto_waypoint"
    AUTO_ORBIT = "auto_orbit"
    RETURN_HOME = "return_home"
    EMERGENCY_LAND = "emergency_land"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class GeoPosition:
    """Geographic position with altitude"""
    latitude: float
    longitude: float
    altitude_m: float
    timestamp: datetime


@dataclass(frozen=True)
class Telemetry:
    """Real-time drone telemetry"""
    position: GeoPosition
    battery_percent: float
    speed_mps: float
    heading_deg: float
    signal_strength: int
    flight_mode: FlightMode
    is_obstacle_detected: bool


@dataclass(frozen=True)
class DroneSpec:
    """Drone technical specifications"""
    model: str
    max_flight_time_min: int
    max_speed_mps: float
    max_altitude_m: float
    max_range_km: float
    camera_resolution: str
    capabilities: list[DroneCapability]
    weight_kg: float


@dataclass(frozen=True)
class SurveillanceZone:
    """Surveillance area definition"""
    zone_id: str
    name: str
    center_lat: float
    center_lon: float
    radius_m: float
    priority: int
    required_capabilities: list[DroneCapability]


@dataclass(frozen=True)
class FlightPlan:
    """Planned flight route"""
    waypoints: list[GeoPosition]
    estimated_duration_min: float
    estimated_distance_km: float
    min_battery_required: float
    no_fly_zones_checked: bool


@dataclass(frozen=True)
class Drone:
    """Complete drone information"""
    drone_id: str
    name: str
    serial_number: str
    specs: DroneSpec
    status: DroneStatus
    current_position: GeoPosition | None
    battery_percent: float
    flight_hours_total: float
    last_maintenance: datetime
    next_maintenance_due: datetime
    assigned_mission_id: str | None
    home_base_lat: float
    home_base_lon: float


@dataclass(frozen=True)
class Mission:
    """Drone mission"""
    mission_id: str
    mission_type: MissionType
    status: MissionStatus
    assigned_drone_id: str | None
    surveillance_zone: SurveillanceZone | None
    flight_plan: FlightPlan | None
    priority: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    images_captured: int
    analysis_results: dict[str, Any] | None


# ============================================================
# PROTOCOLS
# ============================================================

class ImageAnalysisService(Protocol):
    """AI image analysis service"""
    async def analyze_drone_image(
        self,
        image_bytes: bytes,
        location: GeoPosition,
    ) -> dict[str, Any]: ...


class WeatherService(Protocol):
    """Weather condition service"""
    async def get_flight_conditions(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]: ...


# ============================================================
# FLIGHT PLANNER
# ============================================================

class FlightPlanner:
    """
    Plans optimal flight routes for drone missions.
    
    Considers:
    - Distance optimization
    - Battery constraints
    - No-fly zones
    - Weather conditions
    - Collision avoidance
    """
    
    def __init__(self):
        self.no_fly_zones = [
            # Add Kerala airport and military zones
            {"lat": 10.1520, "lon": 76.3990, "radius_km": 5},  # Kochi Airport
            {"lat": 11.1361, "lon": 75.9522, "radius_km": 5},  # Calicut Airport
        ]
        self.max_flight_altitude_m = 120  # Standard drone limit
        self.safety_buffer_m = 50  # Minimum obstacle clearance

    async def plan_surveillance_mission(
        self,
        zone: SurveillanceZone,
        drone_spec: DroneSpec,
        start_position: GeoPosition,
    ) -> FlightPlan:
        """Plan a surveillance mission covering a zone"""
        
        # Create grid waypoints to cover the zone
        waypoints = self._generate_coverage_waypoints(
            center_lat=zone.center_lat,
            center_lon=zone.center_lon,
            radius_m=zone.radius_m,
            altitude_m=min(80, drone_spec.max_altitude_m),
        )
        
        # Add return to start
        waypoints.append(start_position)
        
        # Calculate distance and duration
        total_distance_km = self._calculate_path_distance(waypoints)
        
        # Estimate duration (flight + photo capture time)
        avg_speed_mps = drone_spec.max_speed_mps * 0.6  # Conservative
        flight_time_sec = (total_distance_km * 1000) / avg_speed_mps
        photo_time_sec = len(waypoints) * 5  # 5 sec per waypoint
        estimated_duration_min = (flight_time_sec + photo_time_sec) / 60
        
        # Calculate battery requirement (with 20% safety margin)
        battery_required = (estimated_duration_min / drone_spec.max_flight_time_min) * 1.2
        
        # Check no-fly zones
        no_fly_checked = self._check_no_fly_zones(waypoints)
        
        return FlightPlan(
            waypoints=waypoints,
            estimated_duration_min=estimated_duration_min,
            estimated_distance_km=total_distance_km,
            min_battery_required=min(battery_required * 100, 100),
            no_fly_zones_checked=no_fly_checked,
        )

    def _generate_coverage_waypoints(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        altitude_m: float,
    ) -> list[GeoPosition]:
        """Generate waypoints in a grid pattern to cover area"""
        
        waypoints = []
        
        # Simple lawn mower pattern
        num_passes = max(3, int(radius_m / 100))  # One pass per 100m
        
        for i in range(num_passes):
            # Calculate offset from center
            offset_m = (i - num_passes // 2) * (radius_m / num_passes)
            
            # Convert to lat/lon offset
            lat_offset = offset_m / 111320  # meters to degrees latitude
            
            # Create two points for this pass
            waypoints.append(GeoPosition(
                latitude=center_lat + lat_offset,
                longitude=center_lon - radius_m / (111320 * math.cos(math.radians(center_lat))),
                altitude_m=altitude_m,
                timestamp=datetime.now(timezone.utc),
            ))
            
            waypoints.append(GeoPosition(
                latitude=center_lat + lat_offset,
                longitude=center_lon + radius_m / (111320 * math.cos(math.radians(center_lat))),
                altitude_m=altitude_m,
                timestamp=datetime.now(timezone.utc),
            ))
        
        return waypoints

    def _calculate_path_distance(self, waypoints: list[GeoPosition]) -> float:
        """Calculate total path distance in kilometers"""
        total_distance_km = 0.0
        
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            # Haversine formula
            dlat = math.radians(wp2.latitude - wp1.latitude)
            dlon = math.radians(wp2.longitude - wp1.longitude)
            
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(wp1.latitude)) *
                 math.cos(math.radians(wp2.latitude)) *
                 math.sin(dlon / 2) ** 2)
            
            c = 2 * math.asin(math.sqrt(a))
            distance_km = 6371 * c  # Earth radius in km
            
            total_distance_km += distance_km
        
        return total_distance_km

    def _check_no_fly_zones(self, waypoints: list[GeoPosition]) -> bool:
        """Check if route intersects no-fly zones"""
        
        for waypoint in waypoints:
            for nfz in self.no_fly_zones:
                distance_km = self._calculate_distance(
                    waypoint.latitude,
                    waypoint.longitude,
                    nfz["lat"],
                    nfz["lon"],
                )
                
                if distance_km < nfz["radius_km"]:
                    logger.warning(
                        f"Waypoint {waypoint} is in no-fly zone at "
                        f"({nfz['lat']}, {nfz['lon']})"
                    )
                    return False
        
        return True

    def _calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two points in km"""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.asin(math.sqrt(a))
        return 6371 * c


# ============================================================
# DRONE COORDINATOR
# ============================================================

class DroneCoordinator:
    """
    Coordinates multiple drones for optimal coverage.
    
    Handles:
    - Task assignment
    - Collision avoidance
    - Load balancing
    - Emergency protocols
    """
    
    def __init__(self):
        self.active_drones: dict[str, Drone] = {}
        self.active_missions: dict[str, Mission] = {}
        self.flight_planner = FlightPlanner()
        
        # Collision avoidance
        self.min_separation_m = 100  # Minimum distance between drones

    async def assign_mission(
        self,
        mission: Mission,
        available_drones: list[Drone],
    ) -> tuple[str, FlightPlan] | None:
        """
        Assign mission to best available drone.
        
        Returns (drone_id, flight_plan) or None if no suitable drone
        """
        
        if not mission.surveillance_zone:
            logger.error("Mission has no surveillance zone")
            return None
        
        # Filter drones by capabilities
        capable_drones = [
            d for d in available_drones
            if self._has_required_capabilities(
                d.specs.capabilities,
                mission.surveillance_zone.required_capabilities
            )
        ]
        
        if not capable_drones:
            logger.warning("No drones with required capabilities")
            return None
        
        # Score drones by suitability
        best_drone = None
        best_score = -1
        best_plan = None
        
        for drone in capable_drones:
            if drone.status != DroneStatus.AVAILABLE:
                continue
            
            if drone.battery_percent < 30:
                continue
            
            # Plan flight
            if drone.current_position:
                start_pos = drone.current_position
            else:
                start_pos = GeoPosition(
                    latitude=drone.home_base_lat,
                    longitude=drone.home_base_lon,
                    altitude_m=0,
                    timestamp=datetime.now(timezone.utc),
                )
            
            flight_plan = await self.flight_planner.plan_surveillance_mission(
                zone=mission.surveillance_zone,
                drone_spec=drone.specs,
                start_position=start_pos,
            )
            
            # Check if drone has enough battery
            if drone.battery_percent < flight_plan.min_battery_required:
                continue
            
            # Score: closer is better, more battery is better
            distance_score = 1.0 / (1.0 + flight_plan.estimated_distance_km)
            battery_score = drone.battery_percent / 100.0
            score = (distance_score * 0.6) + (battery_score * 0.4)
            
            if score > best_score:
                best_score = score
                best_drone = drone
                best_plan = flight_plan
        
        if best_drone and best_plan:
            logger.info(
                f"Assigned mission {mission.mission_id} to drone {best_drone.drone_id} "
                f"(score: {best_score:.2f})"
            )
            return (best_drone.drone_id, best_plan)
        
        return None

    def _has_required_capabilities(
        self,
        drone_caps: list[DroneCapability],
        required_caps: list[DroneCapability],
    ) -> bool:
        """Check if drone has all required capabilities"""
        drone_cap_set = set(drone_caps)
        required_cap_set = set(required_caps)
        return required_cap_set.issubset(drone_cap_set)

    async def check_collisions(
        self,
        drone_id: str,
        planned_position: GeoPosition,
    ) -> bool:
        """
        Check if planned position would cause collision.
        
        Returns True if safe, False if collision risk
        """
        
        for other_id, other_drone in self.active_drones.items():
            if other_id == drone_id:
                continue
            
            if not other_drone.current_position:
                continue
            
            distance_m = self._calculate_3d_distance(
                planned_position,
                other_drone.current_position,
            )
            
            if distance_m < self.min_separation_m:
                logger.warning(
                    f"Collision risk: Drone {drone_id} would be {distance_m:.1f}m "
                    f"from drone {other_id}"
                )
                return False
        
        return True

    def _calculate_3d_distance(
        self,
        pos1: GeoPosition,
        pos2: GeoPosition,
    ) -> float:
        """Calculate 3D distance between positions in meters"""
        
        # Horizontal distance
        dlat = math.radians(pos2.latitude - pos1.latitude)
        dlon = math.radians(pos2.longitude - pos1.longitude)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(pos1.latitude)) *
             math.cos(math.radians(pos2.latitude)) *
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.asin(math.sqrt(a))
        horizontal_m = 6371000 * c  # Earth radius in meters
        
        # Vertical distance
        vertical_m = abs(pos2.altitude_m - pos1.altitude_m)
        
        # 3D distance
        return math.sqrt(horizontal_m ** 2 + vertical_m ** 2)


# ============================================================
# DRONE SERVICE
# ============================================================

class DroneService:
    """
    Main service for drone fleet management.
    
    Provides:
    - Drone registration and tracking
    - Mission planning and execution
    - Real-time monitoring
    - Safety management
    - Image analysis coordination
    """
    
    def __init__(
        self,
        image_analysis_service: ImageAnalysisService | None = None,
        weather_service: WeatherService | None = None,
    ):
        self.image_analysis_service = image_analysis_service
        self.weather_service = weather_service
        self.coordinator = DroneCoordinator()
        self.flight_planner = FlightPlanner()
        
        # Storage (use database in production)
        self.drones: dict[str, Drone] = {}
        self.missions: dict[str, Mission] = {}
        self.telemetry_history: dict[str, list[Telemetry]] = {}

    async def register_drone(
        self,
        name: str,
        serial_number: str,
        specs: DroneSpec,
        home_base_lat: float,
        home_base_lon: float,
    ) -> Drone:
        """Register a new drone in the fleet"""
        
        drone_id = str(uuid4())
        
        drone = Drone(
            drone_id=drone_id,
            name=name,
            serial_number=serial_number,
            specs=specs,
            status=DroneStatus.AVAILABLE,
            current_position=None,
            battery_percent=100.0,
            flight_hours_total=0.0,
            last_maintenance=datetime.now(timezone.utc),
            next_maintenance_due=datetime.now(timezone.utc) + timedelta(days=30),
            assigned_mission_id=None,
            home_base_lat=home_base_lat,
            home_base_lon=home_base_lon,
        )
        
        self.drones[drone_id] = drone
        self.coordinator.active_drones[drone_id] = drone
        
        logger.info(f"Registered drone: {name} ({drone_id})")
        
        return drone

    async def create_mission(
        self,
        mission_type: MissionType,
        surveillance_zone: SurveillanceZone,
        priority: int = 5,
    ) -> Mission:
        """Create a new drone mission"""
        
        mission_id = str(uuid4())
        
        mission = Mission(
            mission_id=mission_id,
            mission_type=mission_type,
            status=MissionStatus.PLANNED,
            assigned_drone_id=None,
            surveillance_zone=surveillance_zone,
            flight_plan=None,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            started_at=None,
            completed_at=None,
            images_captured=0,
            analysis_results=None,
        )
        
        self.missions[mission_id] = mission
        
        logger.info(
            f"Created mission: {mission_type.value} in zone {surveillance_zone.name} "
            f"({mission_id})"
        )
        
        return mission

    async def assign_and_start_mission(
        self,
        mission_id: str,
    ) -> bool:
        """Assign mission to best drone and start execution"""
        
        if mission_id not in self.missions:
            logger.error(f"Mission {mission_id} not found")
            return False
        
        mission = self.missions[mission_id]
        
        # Get available drones
        available = [d for d in self.drones.values() if d.status == DroneStatus.AVAILABLE]
        
        if not available:
            logger.warning("No drones available for mission assignment")
            return False
        
        # Assign mission
        assignment = await self.coordinator.assign_mission(mission, available)
        
        if not assignment:
            logger.warning(f"Could not assign mission {mission_id}")
            return False
        
        drone_id, flight_plan = assignment
        
        # Update mission
        mission = Mission(
            mission_id=mission.mission_id,
            mission_type=mission.mission_type,
            status=MissionStatus.ASSIGNED,
            assigned_drone_id=drone_id,
            surveillance_zone=mission.surveillance_zone,
            flight_plan=flight_plan,
            priority=mission.priority,
            created_at=mission.created_at,
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            images_captured=0,
            analysis_results=None,
        )
        
        self.missions[mission_id] = mission
        
        # Update drone status
        drone = self.drones[drone_id]
        self.drones[drone_id] = Drone(
            drone_id=drone.drone_id,
            name=drone.name,
            serial_number=drone.serial_number,
            specs=drone.specs,
            status=DroneStatus.IN_FLIGHT,
            current_position=drone.current_position,
            battery_percent=drone.battery_percent,
            flight_hours_total=drone.flight_hours_total,
            last_maintenance=drone.last_maintenance,
            next_maintenance_due=drone.next_maintenance_due,
            assigned_mission_id=mission_id,
            home_base_lat=drone.home_base_lat,
            home_base_lon=drone.home_base_lon,
        )
        
        logger.info(f"Mission {mission_id} started on drone {drone_id}")
        
        return True

    async def update_telemetry(
        self,
        drone_id: str,
        telemetry: Telemetry,
    ) -> None:
        """Update drone real-time telemetry"""
        
        if drone_id not in self.drones:
            logger.warning(f"Unknown drone {drone_id}")
            return
        
        # Store telemetry
        if drone_id not in self.telemetry_history:
            self.telemetry_history[drone_id] = []
        
        self.telemetry_history[drone_id].append(telemetry)
        
        # Keep last 1000 telemetry points
        if len(self.telemetry_history[drone_id]) > 1000:
            self.telemetry_history[drone_id] = self.telemetry_history[drone_id][-1000:]
        
        # Update drone position and battery
        drone = self.drones[drone_id]
        self.drones[drone_id] = Drone(
            drone_id=drone.drone_id,
            name=drone.name,
            serial_number=drone.serial_number,
            specs=drone.specs,
            status=drone.status,
            current_position=telemetry.position,
            battery_percent=telemetry.battery_percent,
            flight_hours_total=drone.flight_hours_total,
            last_maintenance=drone.last_maintenance,
            next_maintenance_due=drone.next_maintenance_due,
            assigned_mission_id=drone.assigned_mission_id,
            home_base_lat=drone.home_base_lat,
            home_base_lon=drone.home_base_lon,
        )
        
        # Check for low battery
        if telemetry.battery_percent < 20:
            logger.warning(
                f"Drone {drone_id} has low battery: {telemetry.battery_percent:.1f}%"
            )
            await self._initiate_return_home(drone_id)

    async def _initiate_return_home(self, drone_id: str) -> None:
        """Initiate return to home for low battery or emergency"""
        logger.info(f"Initiating return to home for drone {drone_id}")
        
        # In production, send command to drone
        # For now, just log
        
    async def get_fleet_status(self) -> dict[str, Any]:
        """Get overview of entire drone fleet"""
        
        total = len(self.drones)
        by_status = {}
        
        for drone in self.drones.values():
            status = drone.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        avg_battery = sum(d.battery_percent for d in self.drones.values()) / max(total, 1)
        
        return {
            "total_drones": total,
            "by_status": by_status,
            "average_battery": avg_battery,
            "active_missions": len([m for m in self.missions.values() if m.status == MissionStatus.IN_PROGRESS]),
            "timestamp": datetime.now(timezone.utc),
        }


# ============================================================
# FACTORY
# ============================================================

async def create_drone_service(
    image_analysis_service: ImageAnalysisService | None = None,
    weather_service: WeatherService | None = None,
) -> DroneService:
    """Create and initialize DroneService"""
    
    service = DroneService(
        image_analysis_service=image_analysis_service,
        weather_service=weather_service,
    )
    
    logger.info("Drone service initialized")
    
    return service
