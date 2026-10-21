"""Placeholder module: integrations/maps_adapter.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Maps and Geospatial Integration Adapter

File: backend/integrations/maps_adapter.py

Python 3.11+

Install:
    pip install httpx

Capabilities:
    - Geocoding and reverse geocoding
    - Coordinate validation
    - Haversine distance
    - OSRM-compatible route retrieval
    - Route alternatives
    - Route geometry decoding
    - Road-closure intersection checks
    - Flood-zone intersection checks
    - Rescue and shelter location models
    - Emergency route risk assessment
    - GeoJSON export
    - Async batch operations
    - Rate limiting
    - Retry and timeout handling
    - Human approval requirements

IMPORTANT:
    - No automatic rescue dispatch.
    - No automatic evacuation orders.
    - No automatic navigation instructions
      for emergency responders.
    - Route availability does not prove safety.
    - Flood/road data must be authenticated.
    - Routing engine does not automatically
      know about locally reported closures.
    - Hazard geometry intersection is advisory.
    - Emergency routes require human review.
    - Never send sensitive SOS coordinates
      to unauthorized third-party services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    httpx = None


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================
# EXCEPTIONS
# ============================================================

class MapsAdapterError(Exception):
    pass


class MapsConfigurationError(MapsAdapterError):
    pass


class MapsValidationError(MapsAdapterError):
    pass


class MapsRequestError(MapsAdapterError):
    pass


class MapsResponseError(MapsAdapterError):
    pass


class MapsAuthorizationError(MapsAdapterError):
    pass


class MapsRouteUnavailableError(MapsAdapterError):
    pass


class MapsSafetyError(MapsAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class TravelMode(str, Enum):
    DRIVING = "driving"
    WALKING = "walking"
    CYCLING = "cycling"


class MapLocationType(str, Enum):
    SOS = "sos"
    INCIDENT = "incident"
    SHELTER = "shelter"
    HOSPITAL = "hospital"
    RESCUE_TEAM = "rescue_team"
    RELIEF_CENTER = "relief_center"
    WAREHOUSE = "warehouse"
    ROAD_CLOSURE = "road_closure"
    FLOOD_ZONE = "flood_zone"
    OTHER = "other"


class HazardSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SourceVerification(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class RouteSafetyStatus(str, Enum):
    NO_KNOWN_INTERSECTION = "no_known_intersection"
    POTENTIAL_HAZARD = "potential_hazard"
    BLOCKED = "blocked"
    INSUFFICIENT_DATA = "insufficient_data"


class RouteApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================
# UTILITIES
# ============================================================

EARTH_RADIUS_KM = 6371.0088


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def finite_number(
    value: Any,
    field_name: str,
) -> float:
    if isinstance(value, bool):
        raise MapsValidationError(
            f"{field_name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MapsValidationError(
            f"{field_name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise MapsValidationError(
            f"{field_name} must be finite"
        )

    return number


def parse_timestamp(
    value: str | datetime,
) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        text = value.strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise MapsValidationError(
                "Invalid timestamp"
            ) from exc
    else:
        raise MapsValidationError(
            "Unsupported timestamp"
        )

    if result.tzinfo is None:
        raise MapsValidationError(
            "Timestamp must include timezone"
        )

    return result.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value


# ============================================================
# GEOSPATIAL MODELS
# ============================================================

@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        latitude = finite_number(
            self.latitude,
            "latitude",
        )

        longitude = finite_number(
            self.longitude,
            "longitude",
        )

        if not -90 <= latitude <= 90:
            raise MapsValidationError(
                "Latitude out of range"
            )

        if not -180 <= longitude <= 180:
            raise MapsValidationError(
                "Longitude out of range"
            )

    def to_lon_lat(self) -> list[float]:
        return [
            self.longitude,
            self.latitude,
        ]


@dataclass(frozen=True)
class GeoBoundingBox:
    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if not (
            -90 <= self.south <= self.north <= 90
        ):
            raise MapsValidationError(
                "Invalid latitude bounds"
            )

        if not (
            -180 <= self.west <= self.east <= 180
        ):
            raise MapsValidationError(
                "Invalid longitude bounds"
            )

    def contains(
        self,
        point: GeoPoint,
    ) -> bool:
        return (
            self.south <= point.latitude <= self.north
            and self.west <= point.longitude <= self.east
        )


@dataclass(frozen=True)
class MapLocation:
    location_id: str
    name: str
    point: GeoPoint

    location_type: MapLocationType

    district: str | None = None
    address: str | None = None

    tenant_id: str = "default"

    source_verification: SourceVerification = (
        SourceVerification.UNVERIFIED
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class GeocodingResult:
    display_name: str
    point: GeoPoint

    provider: str

    place_id: str | None = None
    bounding_box: GeoBoundingBox | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# ROUTING MODELS
# ============================================================

@dataclass(frozen=True)
class RouteRequest:
    origin: GeoPoint
    destination: GeoPoint

    travel_mode: TravelMode = TravelMode.DRIVING

    alternatives: bool = True

    tenant_id: str = "default"

    emergency_use: bool = False

    request_id: str = field(
        default_factory=generate_id
    )


@dataclass(frozen=True)
class MapRoute:
    route_id: str

    origin: GeoPoint
    destination: GeoPoint

    distance_meters: float
    duration_seconds: float

    geometry: tuple[GeoPoint, ...]

    provider: str

    travel_mode: TravelMode

    retrieved_at: datetime

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def distance_km(self) -> float:
        return self.distance_meters / 1000.0

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0


@dataclass(frozen=True)
class RouteHazardAssessment:
    route_id: str

    status: RouteSafetyStatus

    intersecting_flood_zone_ids: tuple[str, ...]

    intersecting_road_closure_ids: tuple[str, ...]

    warnings: tuple[str, ...]

    requires_human_review: bool = True

    assessed_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class RouteRecommendation:
    route: MapRoute

    assessment: RouteHazardAssessment

    approval_status: RouteApprovalStatus = (
        RouteApprovalStatus.PENDING
    )

    advisory_only: bool = True


# ============================================================
# HAZARD MODELS
# ============================================================

@dataclass(frozen=True)
class FloodHazardZone:
    zone_id: str

    name: str

    polygon: tuple[GeoPoint, ...]

    severity: HazardSeverity

    observed_at: datetime

    source_verification: SourceVerification

    tenant_id: str

    source_uri: str | None = None

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise MapsValidationError(
                "Flood polygon requires 3+ points"
            )


@dataclass(frozen=True)
class RoadClosure:
    closure_id: str

    name: str

    geometry: tuple[GeoPoint, ...]

    observed_at: datetime

    source_verification: SourceVerification

    tenant_id: str

    is_closed: bool = True

    source_uri: str | None = None

    def __post_init__(self) -> None:
        if len(self.geometry) < 2:
            raise MapsValidationError(
                "Road closure requires 2+ points"
            )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MapsAdapterConfig:
    geocoding_base_url: str
    routing_base_url: str

    user_agent: str = (
        "KeralaFloodReliefCoordinationAgent/1.0 "
        "(configure-valid-contact)"
    )

    timeout_seconds: float = 20.0

    max_retries: int = 2

    retry_base_seconds: float = 0.5

    max_response_bytes: int = 5_000_000

    max_geocoding_results: int = 5

    max_route_alternatives: int = 3

    max_route_geometry_points: int = 20_000

    request_interval_seconds: float = 1.0

    hazard_max_age_minutes: int = 120

    require_verified_hazards: bool = True

    require_human_route_approval: bool = True

    allowed_geocoding_hosts: tuple[str, ...] = ()

    allowed_routing_hosts: tuple[str, ...] = ()

    allow_local_http_testing: bool = False

    def __post_init__(self) -> None:
        self._validate_url(
            self.geocoding_base_url,
            self.allowed_geocoding_hosts,
        )

        self._validate_url(
            self.routing_base_url,
            self.allowed_routing_hosts,
        )

        if self.timeout_seconds <= 0:
            raise MapsConfigurationError(
                "Timeout must be positive"
            )

        if self.max_retries < 0:
            raise MapsConfigurationError(
                "Retries cannot be negative"
            )

        if self.request_interval_seconds < 0:
            raise MapsConfigurationError(
                "Invalid request interval"
            )

        for name in (
            "max_response_bytes",
            "max_geocoding_results",
            "max_route_alternatives",
            "max_route_geometry_points",
            "hazard_max_age_minutes",
        ):
            if getattr(self, name) <= 0:
                raise MapsConfigurationError(
                    f"{name} must be positive"
                )

    def _validate_url(
        self,
        value: str,
        allowed_hosts: Sequence[str],
    ) -> None:
        parsed = urlparse(value)

        if not parsed.hostname:
            raise MapsConfigurationError(
                "Invalid provider URL"
            )

        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise MapsConfigurationError(
                "Provider URL must not contain "
                "credentials or query parameters"
            )

        if parsed.scheme != "https":
            local_testing = (
                self.allow_local_http_testing
                and parsed.scheme == "http"
                and parsed.hostname in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
            )

            if not local_testing:
                raise MapsConfigurationError(
                    "HTTPS required"
                )

        if (
            allowed_hosts
            and parsed.hostname not in allowed_hosts
        ):
            raise MapsConfigurationError(
                "Provider host not allowlisted"
            )


# ============================================================
# DISTANCE CALCULATOR
# ============================================================

class GeoDistanceCalculator:
    @staticmethod
    def haversine_km(
        point_a: GeoPoint,
        point_b: GeoPoint,
    ) -> float:
        lat1 = math.radians(point_a.latitude)
        lat2 = math.radians(point_b.latitude)

        delta_lat = math.radians(
            point_b.latitude - point_a.latitude
        )

        delta_lon = math.radians(
            point_b.longitude - point_a.longitude
        )

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(delta_lon / 2) ** 2
        )

        return (
            2
            * EARTH_RADIUS_KM
            * math.asin(
                min(1.0, math.sqrt(a))
            )
        )

    @classmethod
    def nearest_locations(
        cls,
        origin: GeoPoint,
        locations: Sequence[MapLocation],
        *,
        top_k: int = 5,
    ) -> list[tuple[MapLocation, float]]:
        if top_k <= 0:
            raise MapsValidationError(
                "top_k must be positive"
            )

        results = [
            (
                location,
                cls.haversine_km(
                    origin,
                    location.point,
                ),
            )
            for location in locations
        ]

        results.sort(
            key=lambda item: item[1]
        )

        return results[:top_k]


# ============================================================
# HTTP TRANSPORT
# ============================================================

class MapsHTTPClient:
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        config: MapsAdapterConfig,
    ) -> None:
        if httpx is None:
            raise MapsConfigurationError(
                "Install httpx: pip install httpx"
            )

        self.config = config

        self._client: httpx.AsyncClient | None = None

        self._request_lock = asyncio.Lock()

        self._last_request_time = 0.0

    async def _get_client(
        self,
    ) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers={
                    "User-Agent": self.config.user_agent,
                    "Accept": "application/json",
                },
                follow_redirects=False,
                trust_env=False,
            )

        return self._client

    async def _throttle(self) -> None:
        async with self._request_lock:
            now = time.monotonic()

            elapsed = (
                now - self._last_request_time
            )

            wait_seconds = max(
                0.0,
                self.config.request_interval_seconds
                - elapsed,
            )

            if wait_seconds > 0:
                await asyncio.sleep(
                    wait_seconds
                )

            self._last_request_time = (
                time.monotonic()
            )

    async def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        allowed_base_url: str,
    ) -> Any:
        base = urlparse(allowed_base_url)
        target = urlparse(url)

        if (
            target.scheme != base.scheme
            or target.hostname != base.hostname
            or target.port != base.port
            or not target.path.startswith(
                base.path.rstrip("/") + "/"
            )
            or target.username
            or target.password
        ):
            raise MapsAuthorizationError(
                "Request URL outside authorized provider"
            )

        client = await self._get_client()

        last_error: Exception | None = None

        for attempt in range(
            self.config.max_retries + 1
        ):
            await self._throttle()

            try:
                async with client.stream(
                    "GET",
                    url,
                    params=params,
                ) as response:
                    if (
                        response.status_code
                        in self.RETRYABLE_STATUS_CODES
                    ):
                        raise MapsRequestError(
                            "Temporary provider failure"
                        )

                    response.raise_for_status()

                    content_length = (
                        response.headers.get(
                            "Content-Length"
                        )
                    )

                    if content_length:
                        try:
                            declared_size = int(
                                content_length
                            )
                        except ValueError as exc:
                            raise MapsResponseError(
                                "Invalid Content-Length"
                            ) from exc

                        if declared_size > (
                            self.config.max_response_bytes
                        ):
                            raise MapsResponseError(
                                "Response too large"
                            )

                    chunks = []
                    total = 0

                    async for chunk in response.aiter_bytes():
                        total += len(chunk)

                        if total > (
                            self.config.max_response_bytes
                        ):
                            raise MapsResponseError(
                                "Response size limit exceeded"
                            )

                        chunks.append(chunk)

                return json.loads(
                    b"".join(chunks)
                )

            except MapsResponseError:
                raise

            except (
                MapsRequestError,
                httpx.TimeoutException,
                httpx.TransportError,
            ) as exc:
                last_error = exc

                if attempt >= self.config.max_retries:
                    break

                delay = (
                    self.config.retry_base_seconds
                    * (2 ** attempt)
                )

                delay += random.uniform(
                    0,
                    min(1.0, delay * 0.2),
                )

                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as exc:
                raise MapsRequestError(
                    f"Provider returned HTTP "
                    f"{exc.response.status_code}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise MapsResponseError(
                    "Invalid provider JSON"
                ) from exc

        raise MapsRequestError(
            "Provider request failed"
        ) from last_error

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ============================================================
# GEOCODING
# ============================================================

class MapsGeocodingAdapter:
    def __init__(
        self,
        config: MapsAdapterConfig,
        http_client: MapsHTTPClient,
    ) -> None:
        self.config = config
        self.http_client = http_client

    async def geocode(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[GeocodingResult]:
        query = query.strip()

        if not query or len(query) > 500:
            raise MapsValidationError(
                "Invalid geocoding query"
            )

        limit = min(
            max(1, limit),
            self.config.max_geocoding_results,
        )

        base = (
            self.config.geocoding_base_url.rstrip("/")
        )

        payload = await self.http_client.get_json(
            f"{base}/search",
            params={
                "q": query,
                "format": "json",
                "limit": limit,
                "addressdetails": 1,
            },
            allowed_base_url=(
                self.config.geocoding_base_url
            ),
        )

        if not isinstance(payload, list):
            raise MapsResponseError(
                "Unexpected geocoding response"
            )

        output = []

        for item in payload[:limit]:
            if not isinstance(item, Mapping):
                raise MapsResponseError(
                    "Invalid geocoding result"
                )

            point = GeoPoint(
                latitude=finite_number(
                    item.get("lat"),
                    "latitude",
                ),
                longitude=finite_number(
                    item.get("lon"),
                    "longitude",
                ),
            )

            output.append(
                GeocodingResult(
                    display_name=str(
                        item.get("display_name", "")
                    ),
                    point=point,
                    provider="nominatim-compatible",
                    place_id=str(
                        item.get("place_id", "")
                    ),
                    metadata={
                        "type": item.get("type"),
                        "class": item.get("class"),
                    },
                )
            )

        return output

    async def reverse_geocode(
        self,
        point: GeoPoint,
    ) -> GeocodingResult:
        base = (
            self.config.geocoding_base_url.rstrip("/")
        )

        payload = await self.http_client.get_json(
            f"{base}/reverse",
            params={
                "lat": point.latitude,
                "lon": point.longitude,
                "format": "json",
                "zoom": 16,
            },
            allowed_base_url=(
                self.config.geocoding_base_url
            ),
        )

        if not isinstance(payload, Mapping):
            raise MapsResponseError(
                "Unexpected reverse geocoding response"
            )

        return GeocodingResult(
            display_name=str(
                payload.get("display_name", "")
            ),
            point=GeoPoint(
                latitude=finite_number(
                    payload.get(
                        "lat",
                        point.latitude,
                    ),
                    "latitude",
                ),
                longitude=finite_number(
                    payload.get(
                        "lon",
                        point.longitude,
                    ),
                    "longitude",
                ),
            ),
            provider="nominatim-compatible",
            place_id=str(
                payload.get("place_id", "")
            ),
        )


# ============================================================
# ROUTING
# ============================================================

class OSRMRoutingAdapter:
    """
    OSRM-compatible driving route integration.

    Walking/cycling require a separately
    configured routing profile/service.
    """

    def __init__(
        self,
        config: MapsAdapterConfig,
        http_client: MapsHTTPClient,
    ) -> None:
        self.config = config
        self.http_client = http_client

    async def get_routes(
        self,
        request: RouteRequest,
    ) -> list[MapRoute]:
        if request.travel_mode != TravelMode.DRIVING:
            raise MapsValidationError(
                "This OSRM adapter supports driving "
                "only; configure another routing "
                "profile for walking/cycling"
            )

        origin = request.origin
        destination = request.destination

        coordinates = (
            f"{origin.longitude},{origin.latitude};"
            f"{destination.longitude},"
            f"{destination.latitude}"
        )

        base = (
            self.config.routing_base_url.rstrip("/")
        )

        url = (
            f"{base}/route/v1/driving/"
            f"{coordinates}"
        )

        payload = await self.http_client.get_json(
            url,
            params={
                "alternatives": (
                    "true"
                    if request.alternatives
                    else "false"
                ),
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
            },
            allowed_base_url=(
                self.config.routing_base_url
            ),
        )

        if not isinstance(payload, Mapping):
            raise MapsResponseError(
                "Invalid routing response"
            )

        if payload.get("code") != "Ok":
            raise MapsRouteUnavailableError(
                "Routing provider could not find a route"
            )

        raw_routes = payload.get(
            "routes",
            [],
        )

        if not isinstance(raw_routes, list):
            raise MapsResponseError(
                "Invalid route list"
            )

        results = []

        for item in raw_routes[
            :self.config.max_route_alternatives
        ]:
            if not isinstance(item, Mapping):
                raise MapsResponseError(
                    "Invalid route record"
                )

            geometry = item.get(
                "geometry",
                {},
            )

            if not isinstance(
                geometry,
                Mapping,
            ):
                raise MapsResponseError(
                    "Invalid route geometry"
                )

            raw_coordinates = geometry.get(
                "coordinates",
                [],
            )

            if not isinstance(
                raw_coordinates,
                list,
            ):
                raise MapsResponseError(
                    "Invalid coordinate list"
                )

            if len(raw_coordinates) > (
                self.config.max_route_geometry_points
            ):
                raise MapsResponseError(
                    "Route geometry exceeds limit"
                )

            points = []

            for coordinate in raw_coordinates:
                if (
                    not isinstance(
                        coordinate,
                        (list, tuple),
                    )
                    or len(coordinate) < 2
                ):
                    raise MapsResponseError(
                        "Invalid geometry coordinate"
                    )

                points.append(
                    GeoPoint(
                        latitude=finite_number(
                            coordinate[1],
                            "latitude",
                        ),
                        longitude=finite_number(
                            coordinate[0],
                            "longitude",
                        ),
                    )
                )

            if len(points) < 2:
                raise MapsResponseError(
                    "Route geometry incomplete"
                )

            distance = finite_number(
                item.get("distance"),
                "distance",
            )

            duration = finite_number(
                item.get("duration"),
                "duration",
            )

            if distance < 0 or duration < 0:
                raise MapsResponseError(
                    "Negative route metric"
                )

            results.append(
                MapRoute(
                    route_id=generate_id(),
                    origin=origin,
                    destination=destination,
                    distance_meters=distance,
                    duration_seconds=duration,
                    geometry=tuple(points),
                    provider="osrm-compatible",
                    travel_mode=request.travel_mode,
                    retrieved_at=utc_now(),
                    metadata={
                        "request_id": request.request_id,
                        "emergency_use": request.emergency_use,
                    },
                )
            )

        if not results:
            raise MapsRouteUnavailableError(
                "No routes returned"
            )

        return results


# ============================================================
# GEOMETRY ENGINE
# ============================================================

class GeoGeometryEngine:
    @staticmethod
    def orientation(
        a: GeoPoint,
        b: GeoPoint,
        c: GeoPoint,
    ) -> float:
        return (
            (b.longitude - a.longitude)
            * (c.latitude - a.latitude)
            - (b.latitude - a.latitude)
            * (c.longitude - a.longitude)
        )

    @classmethod
    def on_segment(
        cls,
        a: GeoPoint,
        b: GeoPoint,
        p: GeoPoint,
    ) -> bool:
        epsilon = 1e-10

        if abs(
            cls.orientation(a, b, p)
        ) > epsilon:
            return False

        return (
            min(a.longitude, b.longitude) - epsilon
            <= p.longitude
            <= max(a.longitude, b.longitude) + epsilon
            and
            min(a.latitude, b.latitude) - epsilon
            <= p.latitude
            <= max(a.latitude, b.latitude) + epsilon
        )

    @classmethod
    def segments_intersect(
        cls,
        a: GeoPoint,
        b: GeoPoint,
        c: GeoPoint,
        d: GeoPoint,
    ) -> bool:
        o1 = cls.orientation(a, b, c)
        o2 = cls.orientation(a, b, d)
        o3 = cls.orientation(c, d, a)
        o4 = cls.orientation(c, d, b)

        if (
            (o1 > 0 > o2 or o2 > 0 > o1)
            and
            (o3 > 0 > o4 or o4 > 0 > o3)
        ):
            return True

        return any([
            cls.on_segment(a, b, c),
            cls.on_segment(a, b, d),
            cls.on_segment(c, d, a),
            cls.on_segment(c, d, b),
        ])

    @classmethod
    def point_in_polygon(
        cls,
        point: GeoPoint,
        polygon: Sequence[GeoPoint],
    ) -> bool:
        inside = False
        n = len(polygon)

        if n < 3:
            return False

        j = n - 1

        for i in range(n):
            a = polygon[i]
            b = polygon[j]

            if cls.on_segment(a, b, point):
                return True

            intersects = (
                (a.latitude > point.latitude)
                != (b.latitude > point.latitude)
            )

            if intersects:
                denominator = (
                    b.latitude - a.latitude
                )

                x_intersection = (
                    (b.longitude - a.longitude)
                    * (point.latitude - a.latitude)
                    / denominator
                    + a.longitude
                )

                if point.longitude < x_intersection:
                    inside = not inside

            j = i

        return inside

    @classmethod
    def route_intersects_polygon(
        cls,
        route: Sequence[GeoPoint],
        polygon: Sequence[GeoPoint],
    ) -> bool:
        if len(route) < 2 or len(polygon) < 3:
            return False

        if any(
            cls.point_in_polygon(point, polygon)
            for point in route
        ):
            return True

        for i in range(len(route) - 1):
            for j in range(len(polygon)):
                a = route[i]
                b = route[i + 1]

                c = polygon[j]
                d = polygon[
                    (j + 1) % len(polygon)
                ]

                if cls.segments_intersect(
                    a,
                    b,
                    c,
                    d,
                ):
                    return True

        return False

    @classmethod
    def polylines_intersect(
        cls,
        first: Sequence[GeoPoint],
        second: Sequence[GeoPoint],
    ) -> bool:
        if len(first) < 2 or len(second) < 2:
            return False

        for i in range(len(first) - 1):
            for j in range(len(second) - 1):
                if cls.segments_intersect(
                    first[i],
                    first[i + 1],
                    second[j],
                    second[j + 1],
                ):
                    return True

        return False


# ============================================================
# ROUTE HAZARD ASSESSMENT
# ============================================================

class RouteHazardEvaluator:
    def __init__(
        self,
        config: MapsAdapterConfig,
    ) -> None:
        self.config = config
        self.geometry = GeoGeometryEngine()

    def _eligible(
        self,
        observed_at: datetime,
        verification: SourceVerification,
        *,
        now: datetime,
    ) -> bool:
        if (
            self.config.require_verified_hazards
            and verification
            != SourceVerification.VERIFIED
        ):
            return False

        age = now - observed_at

        return (
            timedelta(0)
            <= age
            <= timedelta(
                minutes=self.config.hazard_max_age_minutes
            )
        )

    def evaluate(
        self,
        route: MapRoute,
        *,
        tenant_id: str,
        flood_zones: Sequence[FloodHazardZone],
        road_closures: Sequence[RoadClosure],
        hazard_dataset_complete: bool = False,
    ) -> RouteHazardAssessment:
        now = utc_now()

        flood_ids = []
        closure_ids = []
        warnings = []

        excluded_records = 0

        for zone in flood_zones:
            if zone.tenant_id != tenant_id:
                continue

            if not self._eligible(
                zone.observed_at,
                zone.source_verification,
                now=now,
            ):
                excluded_records += 1
                continue

            if self.geometry.route_intersects_polygon(
                route.geometry,
                zone.polygon,
            ):
                flood_ids.append(
                    zone.zone_id
                )

        for closure in road_closures:
            if closure.tenant_id != tenant_id:
                continue

            if not closure.is_closed:
                continue

            if not self._eligible(
                closure.observed_at,
                closure.source_verification,
                now=now,
            ):
                excluded_records += 1
                continue

            if self.geometry.polylines_intersect(
                route.geometry,
                closure.geometry,
            ):
                closure_ids.append(
                    closure.closure_id
                )

        if excluded_records:
            warnings.append(
                "Some hazard records were excluded "
                "because they were stale or unverified"
            )

        if closure_ids:
            status = RouteSafetyStatus.BLOCKED

            warnings.append(
                "Route intersects a known road closure"
            )

        elif flood_ids:
            status = RouteSafetyStatus.POTENTIAL_HAZARD

            warnings.append(
                "Route intersects a mapped flood hazard"
            )

        elif not hazard_dataset_complete:
            status = RouteSafetyStatus.INSUFFICIENT_DATA

            warnings.append(
                "Hazard coverage is incomplete; "
                "route safety cannot be established"
            )

        elif excluded_records:
            status = RouteSafetyStatus.INSUFFICIENT_DATA

        else:
            status = (
                RouteSafetyStatus.NO_KNOWN_INTERSECTION
            )

        warnings.append(
            "Geometry intersection is approximate "
            "and does not establish road passability"
        )

        return RouteHazardAssessment(
            route_id=route.route_id,
            status=status,
            intersecting_flood_zone_ids=tuple(
                flood_ids
            ),
            intersecting_road_closure_ids=tuple(
                closure_ids
            ),
            warnings=tuple(warnings),
            requires_human_review=True,
            assessed_at=now,
        )


# ============================================================
# ROUTE PLANNING SERVICE
# ============================================================

class MapsRoutePlanningService:
    def __init__(
        self,
        routing_adapter: OSRMRoutingAdapter,
        hazard_evaluator: RouteHazardEvaluator,
    ) -> None:
        self.routing_adapter = routing_adapter
        self.hazard_evaluator = hazard_evaluator

    async def plan_routes(
        self,
        request: RouteRequest,
        *,
        flood_zones: Sequence[FloodHazardZone] = (),
        road_closures: Sequence[RoadClosure] = (),
        hazard_dataset_complete: bool = False,
    ) -> list[RouteRecommendation]:
        routes = await self.routing_adapter.get_routes(
            request
        )

        recommendations = []

        for route in routes:
            assessment = self.hazard_evaluator.evaluate(
                route,
                tenant_id=request.tenant_id,
                flood_zones=flood_zones,
                road_closures=road_closures,
                hazard_dataset_complete=(
                    hazard_dataset_complete
                ),
            )

            recommendations.append(
                RouteRecommendation(
                    route=route,
                    assessment=assessment,
                    approval_status=(
                        RouteApprovalStatus.PENDING
                    ),
                    advisory_only=True,
                )
            )

        priority = {
            RouteSafetyStatus.NO_KNOWN_INTERSECTION: 0,
            RouteSafetyStatus.INSUFFICIENT_DATA: 1,
            RouteSafetyStatus.POTENTIAL_HAZARD: 2,
            RouteSafetyStatus.BLOCKED: 3,
        }

        recommendations.sort(
            key=lambda item: (
                priority[
                    item.assessment.status
                ],
                item.route.duration_seconds,
            )
        )

        return recommendations


# ============================================================
# MAP LOCATION SERVICE
# ============================================================

class MapLocationService:
    def __init__(self) -> None:
        self._locations: dict[
            tuple[str, str],
            MapLocation,
        ] = {}

    def add_location(
        self,
        location: MapLocation,
    ) -> None:
        self._locations[
            (
                location.tenant_id,
                location.location_id,
            )
        ] = location

    def get_location(
        self,
        *,
        tenant_id: str,
        location_id: str,
    ) -> MapLocation | None:
        return self._locations.get(
            (
                tenant_id,
                location_id,
            )
        )

    def list_locations(
        self,
        *,
        tenant_id: str,
        location_type: MapLocationType | None = None,
    ) -> list[MapLocation]:
        return [
            location
            for (tenant, _), location
            in self._locations.items()
            if tenant == tenant_id
            and (
                location_type is None
                or location.location_type
                == location_type
            )
        ]

    def nearest(
        self,
        *,
        tenant_id: str,
        origin: GeoPoint,
        location_type: MapLocationType,
        top_k: int = 5,
    ) -> list[tuple[MapLocation, float]]:
        locations = self.list_locations(
            tenant_id=tenant_id,
            location_type=location_type,
        )

        return GeoDistanceCalculator.nearest_locations(
            origin,
            locations,
            top_k=top_k,
        )


# ============================================================
# GEOJSON EXPORTER
# ============================================================

class MapsGeoJSONExporter:
    @staticmethod
    def point_feature(
        location: MapLocation,
    ) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": (
                    location.point.to_lon_lat()
                ),
            },
            "properties": {
                "location_id": location.location_id,
                "name": location.name,
                "location_type": (
                    location.location_type.value
                ),
                "district": location.district,
                "source_verification": (
                    location.source_verification.value
                ),
            },
        }

    @staticmethod
    def route_feature(
        route: MapRoute,
    ) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    point.to_lon_lat()
                    for point in route.geometry
                ],
            },
            "properties": {
                "route_id": route.route_id,
                "distance_km": route.distance_km,
                "duration_minutes": (
                    route.duration_minutes
                ),
                "provider": route.provider,
                "advisory_only": True,
            },
        }

    @staticmethod
    def flood_zone_feature(
        zone: FloodHazardZone,
    ) -> dict[str, Any]:
        coordinates = [
            point.to_lon_lat()
            for point in zone.polygon
        ]

        if coordinates[0] != coordinates[-1]:
            coordinates.append(
                coordinates[0]
            )

        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    coordinates
                ],
            },
            "properties": {
                "zone_id": zone.zone_id,
                "name": zone.name,
                "severity": zone.severity.value,
                "verification": (
                    zone.source_verification.value
                ),
                "observed_at": (
                    zone.observed_at.isoformat()
                ),
            },
        }

    @staticmethod
    def feature_collection(
        features: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": list(features),
        }

    @staticmethod
    def to_json(
        features: Sequence[Mapping[str, Any]],
    ) -> str:
        return json.dumps(
            MapsGeoJSONExporter.feature_collection(
                features
            ),
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN MAPS ADAPTER
# ============================================================

class MapsAdapter:
    def __init__(
        self,
        config: MapsAdapterConfig,
    ) -> None:
        self.config = config

        self.http_client = MapsHTTPClient(
            config
        )

        self.geocoding = MapsGeocodingAdapter(
            config,
            self.http_client,
        )

        self.routing = OSRMRoutingAdapter(
            config,
            self.http_client,
        )

        self.hazard_evaluator = (
            RouteHazardEvaluator(config)
        )

        self.route_planning = (
            MapsRoutePlanningService(
                self.routing,
                self.hazard_evaluator,
            )
        )

        self.locations = MapLocationService()

    async def geocode(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[GeocodingResult]:
        return await self.geocoding.geocode(
            query,
            limit=limit,
        )

    async def reverse_geocode(
        self,
        point: GeoPoint,
    ) -> GeocodingResult:
        return await self.geocoding.reverse_geocode(
            point
        )

    async def get_routes(
        self,
        request: RouteRequest,
    ) -> list[MapRoute]:
        return await self.routing.get_routes(
            request
        )

    async def plan_safe_routes(
        self,
        request: RouteRequest,
        *,
        flood_zones: Sequence[FloodHazardZone] = (),
        road_closures: Sequence[RoadClosure] = (),
        hazard_dataset_complete: bool = False,
    ) -> list[RouteRecommendation]:
        return await self.route_planning.plan_routes(
            request,
            flood_zones=flood_zones,
            road_closures=road_closures,
            hazard_dataset_complete=(
                hazard_dataset_complete
            ),
        )

    def distance_km(
        self,
        origin: GeoPoint,
        destination: GeoPoint,
    ) -> float:
        return GeoDistanceCalculator.haversine_km(
            origin,
            destination,
        )

    async def close(self) -> None:
        await self.http_client.close()

    async def __aenter__(
        self,
    ) -> "MapsAdapter":
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        await self.close()


# ============================================================
# FACTORY
# ============================================================

def create_maps_adapter(
    config: MapsAdapterConfig,
) -> MapsAdapter:
    return MapsAdapter(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def demo() -> None:
    """
    Synthetic coordinates only.
    No real route or hazard information.
    No network requests.
    """

    origin = GeoPoint(
        latitude=10.52,
        longitude=76.21,
    )

    destination = GeoPoint(
        latitude=10.53,
        longitude=76.23,
    )

    distance = (
        GeoDistanceCalculator.haversine_km(
            origin,
            destination,
        )
    )

    print(
        json.dumps(
            {
                "demo": True,
                "straight_line_distance_km": (
                    round(distance, 3)
                ),
                "note": (
                    "Not a road distance or "
                    "emergency travel estimate"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    demo()


__all__ = [
    "MapsAdapterError",
    "MapsConfigurationError",
    "MapsValidationError",
    "MapsRequestError",
    "MapsResponseError",
    "MapsAuthorizationError",
    "MapsRouteUnavailableError",
    "MapsSafetyError",
    "TravelMode",
    "MapLocationType",
    "HazardSeverity",
    "SourceVerification",
    "RouteSafetyStatus",
    "RouteApprovalStatus",
    "GeoPoint",
    "GeoBoundingBox",
    "MapLocation",
    "GeocodingResult",
    "RouteRequest",
    "MapRoute",
    "RouteHazardAssessment",
    "RouteRecommendation",
    "FloodHazardZone",
    "RoadClosure",
    "MapsAdapterConfig",
    "GeoDistanceCalculator",
    "MapsHTTPClient",
    "MapsGeocodingAdapter",
    "OSRMRoutingAdapter",
    "GeoGeometryEngine",
    "RouteHazardEvaluator",
    "MapsRoutePlanningService",
    "MapLocationService",
    "MapsGeoJSONExporter",
    "MapsAdapter",
    "create_maps_adapter",
]
