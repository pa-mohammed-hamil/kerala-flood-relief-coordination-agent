from __future__ import annotations

"""
backend/api/websocket.py

Kerala Flood Relief Coordination Agent
Real-Time WebSocket Communication API

Python 3.11+
FastAPI / Starlette
Pydantic v2

Features:
    - Authenticated WebSocket connections
    - Multi-client connection management
    - Topic-based subscriptions
    - District-level event isolation
    - User-specific notifications
    - Heartbeat / ping / pong
    - Event broadcasting
    - Event schema validation
    - Connection limits
    - Message size limits
    - Subscription limits
    - Permission-aware event delivery
    - Disconnect cleanup
    - Structured logging

Supported topics:
    flood_alerts
    rainfall
    rivers
    dams
    sos
    incidents
    rescue
    evacuation
    shelters
    relief
    healthcare
    volunteers
    missing_persons
    damage
    recovery
    notifications
    agents
    analytics

Security:
    - Never trust client-provided user IDs.
    - Never trust client-provided district IDs.
    - Authenticate before accepting connections.
    - Enforce topic permissions.
    - Restrict sensitive events to authorized users.
    - Client messages cannot directly dispatch rescue
      teams, publish official alerts, or control dams.
    - WebSocket events are informational; operational
      changes must use authorized domain services.

Production requirements:
    - Redis Pub/Sub or equivalent for multi-worker fanout
    - Distributed connection/rate limiting
    - Origin validation
    - TLS/WSS
    - Secure authentication and token revocation
    - Centralized audit and observability
"""


import asyncio
import json
import logging
import secrets

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

from fastapi import APIRouter, WebSocket
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)
from starlette.websockets import WebSocketDisconnect


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Real-Time WebSocket"],
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CONNECTIONS_PER_USER = 5
MAX_SUBSCRIPTIONS_PER_CONNECTION = 20
MAX_MESSAGE_BYTES = 16 * 1024
MAX_EVENT_BYTES = 64 * 1024

RECEIVE_TIMEOUT_SECONDS = 60
MAX_IDLE_SECONDS = 180

WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_FORBIDDEN = 4403
WS_CLOSE_TOO_MANY = 4429
WS_CLOSE_INVALID = 4400


# ============================================================
# TOPICS
# ============================================================

class WSTopic(str, Enum):
    FLOOD_ALERTS = "flood_alerts"
    RAINFALL = "rainfall"
    RIVERS = "rivers"
    DAMS = "dams"

    SOS = "sos"
    INCIDENTS = "incidents"
    RESCUE = "rescue"
    EVACUATION = "evacuation"

    SHELTERS = "shelters"
    RELIEF = "relief"
    HEALTHCARE = "healthcare"
    VOLUNTEERS = "volunteers"

    MISSING_PERSONS = "missing_persons"
    DAMAGE = "damage"
    RECOVERY = "recovery"

    NOTIFICATIONS = "notifications"
    AGENTS = "agents"
    ANALYTICS = "analytics"


class WSMessageType(str, Enum):
    CONNECTED = "connected"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    EVENT = "event"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


# ============================================================
# DATA MODELS
# ============================================================

class WSClientMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: WSMessageType

    topic: WSTopic | None = None

    request_id: str | None = Field(
        default=None,
        max_length=100,
    )


class WSEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    topic: WSTopic

    event_type: str = Field(
        min_length=1,
        max_length=150,
    )

    district_id: int | None = Field(
        default=None,
        ge=1,
    )

    user_id: str | None = None

    payload: dict[str, Any] = Field(
        default_factory=dict,
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


class WSError(BaseModel):
    type: str = "error"
    code: str
    message: str
    request_id: str | None = None


# ============================================================
# AUTHENTICATED PRINCIPAL
# ============================================================

@dataclass(frozen=True)
class WSPrincipal:
    user_id: str

    district_ids: frozenset[int]

    permissions: frozenset[str]

    is_superuser: bool = False

    session_id: str | None = None


class WSAuthenticator(Protocol):
    """
    Application-supplied authentication provider.

    Implement using your existing auth/session system.

    Requirements:
        - Validate signed session or short-lived ticket
        - Verify account is active
        - Verify session is not revoked
        - Resolve current DB-backed permissions
        - Resolve authorized district memberships
        - Never accept permissions from the client
    """

    async def authenticate(
        self,
        websocket: WebSocket,
    ) -> WSPrincipal | None:
        ...


class WSPermissionChecker(Protocol):
    async def can_subscribe(
        self,
        principal: WSPrincipal,
        topic: WSTopic,
    ) -> bool:
        ...


# ============================================================
# TOPIC PERMISSION MAPPING
# ============================================================

TOPIC_PERMISSIONS: dict[WSTopic, str] = {
    WSTopic.FLOOD_ALERTS: "flood_alerts:read",
    WSTopic.RAINFALL: "rainfall:read",
    WSTopic.RIVERS: "rivers:read",
    WSTopic.DAMS: "dams:read",

    WSTopic.SOS: "sos:read",
    WSTopic.INCIDENTS: "incidents:read",
    WSTopic.RESCUE: "rescue:read",
    WSTopic.EVACUATION: "evacuation:read",

    WSTopic.SHELTERS: "shelters:read",
    WSTopic.RELIEF: "relief:read",
    WSTopic.HEALTHCARE: "healthcare:read",
    WSTopic.VOLUNTEERS: "volunteers:read",

    WSTopic.MISSING_PERSONS: "missing_persons:read",
    WSTopic.DAMAGE: "damage:read",
    WSTopic.RECOVERY: "recovery:read",

    WSTopic.NOTIFICATIONS: "notifications:read",
    WSTopic.AGENTS: "agents:read",
    WSTopic.ANALYTICS: "analytics:read",
}


class DefaultWSPermissionChecker:
    """
    Uses verified permissions in WSPrincipal.

    The authentication provider must resolve these
    permissions from the database or a trusted
    authorization service.
    """

    async def can_subscribe(
        self,
        principal: WSPrincipal,
        topic: WSTopic,
    ) -> bool:
        if principal.is_superuser:
            return True

        required = TOPIC_PERMISSIONS.get(topic)

        return (
            required is not None
            and required in principal.permissions
        )


# ============================================================
# CONNECTION STATE
# ============================================================

@dataclass
class WSConnection:
    connection_id: str

    websocket: WebSocket

    principal: WSPrincipal

    subscriptions: set[WSTopic] = field(
        default_factory=set
    )

    connected_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    last_seen_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    send_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )


# ============================================================
# CONNECTION MANAGER
# ============================================================

class WebSocketManager:
    """
    In-memory single-process connection manager.

    Use Redis or another message broker for
    multi-worker / multi-instance deployments.
    """

    def __init__(self) -> None:
        self._connections: dict[
            str,
            WSConnection,
        ] = {}

        self._user_connections: dict[
            str,
            set[str],
        ] = defaultdict(set)

        self._topic_connections: dict[
            WSTopic,
            set[str],
        ] = defaultdict(set)

        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        principal: WSPrincipal,
    ) -> WSConnection | None:
        async with self._lock:
            user_connections = self._user_connections[
                principal.user_id
            ]

            if (
                len(user_connections)
                >= MAX_CONNECTIONS_PER_USER
            ):
                return None

            connection_id = str(uuid4())

            connection = WSConnection(
                connection_id=connection_id,
                websocket=websocket,
                principal=principal,
            )

            self._connections[
                connection_id
            ] = connection

            user_connections.add(
                connection_id
            )

            return connection

    async def disconnect(
        self,
        connection_id: str,
    ) -> None:
        async with self._lock:
            connection = self._connections.pop(
                connection_id,
                None,
            )

            if connection is None:
                return

            user_id = connection.principal.user_id

            self._user_connections[
                user_id
            ].discard(connection_id)

            if not self._user_connections[user_id]:
                self._user_connections.pop(
                    user_id,
                    None,
                )

            for topic in connection.subscriptions:
                self._topic_connections[
                    topic
                ].discard(connection_id)

    async def subscribe(
        self,
        connection_id: str,
        topic: WSTopic,
    ) -> bool:
        async with self._lock:
            connection = self._connections.get(
                connection_id
            )

            if connection is None:
                return False

            if topic in connection.subscriptions:
                return True

            if (
                len(connection.subscriptions)
                >= MAX_SUBSCRIPTIONS_PER_CONNECTION
            ):
                return False

            connection.subscriptions.add(topic)

            self._topic_connections[
                topic
            ].add(connection_id)

            return True

    async def unsubscribe(
        self,
        connection_id: str,
        topic: WSTopic,
    ) -> bool:
        async with self._lock:
            connection = self._connections.get(
                connection_id
            )

            if connection is None:
                return False

            connection.subscriptions.discard(topic)

            self._topic_connections[
                topic
            ].discard(connection_id)

            return True

    async def send_json(
        self,
        connection: WSConnection,
        data: dict[str, Any],
    ) -> bool:
        try:
            async with connection.send_lock:
                await connection.websocket.send_json(
                    data
                )
            return True

        except (
            WebSocketDisconnect,
            RuntimeError,
            OSError,
        ):
            await self.disconnect(
                connection.connection_id
            )
            return False

    async def broadcast(
        self,
        event: WSEvent,
    ) -> int:
        """
        Broadcast a validated event.

        District-scoped events go only to users
        with verified access to that district.

        User-targeted events go only to the
        specified user.

        Sensitive events must be sanitized
        before being published.
        """

        serialized = event.model_dump(
            mode="json"
        )

        encoded = json.dumps(
            serialized,
            separators=(",", ":"),
        ).encode("utf-8")

        if len(encoded) > MAX_EVENT_BYTES:
            raise ValueError(
                "WebSocket event exceeds size limit"
            )

        async with self._lock:
            ids = tuple(
                self._topic_connections.get(
                    event.topic,
                    set(),
                )
            )

            connections = [
                self._connections[cid]
                for cid in ids
                if cid in self._connections
            ]

        delivered = 0

        for connection in connections:
            principal = connection.principal

            if (
                event.user_id is not None
                and event.user_id != principal.user_id
            ):
                continue

            if event.district_id is not None:
                if (
                    not principal.is_superuser
                    and event.district_id
                    not in principal.district_ids
                ):
                    continue

            if await self.send_json(
                connection,
                {
                    "type": WSMessageType.EVENT.value,
                    "data": serialized,
                },
            ):
                delivered += 1

        return delivered

    async def send_to_user(
        self,
        user_id: str,
        event: WSEvent,
    ) -> int:
        """
        User-specific delivery.

        This does not represent confirmed receipt
        or acknowledgment by the recipient.
        """

        if event.user_id != user_id:
            raise ValueError(
                "Event user_id must match target user"
            )

        return await self.broadcast(event)

    async def connection_count(self) -> int:
        async with self._lock:
            return len(self._connections)

    async def topic_counts(
        self,
    ) -> dict[str, int]:
        async with self._lock:
            return {
                topic.value: len(ids)
                for topic, ids
                in self._topic_connections.items()
            }


# ============================================================
# APP STATE DEPENDENCIES
# ============================================================

def get_ws_manager(
    websocket: WebSocket,
) -> WebSocketManager:
    manager = getattr(
        websocket.app.state,
        "websocket_manager",
        None,
    )

    if manager is None:
        raise RuntimeError(
            "WebSocket manager is not configured"
        )

    return manager


def get_ws_authenticator(
    websocket: WebSocket,
) -> WSAuthenticator:
    authenticator = getattr(
        websocket.app.state,
        "websocket_authenticator",
        None,
    )

    if authenticator is None:
        raise RuntimeError(
            "WebSocket authenticator is not configured"
        )

    return authenticator


def get_ws_permission_checker(
    websocket: WebSocket,
) -> WSPermissionChecker:
    checker = getattr(
        websocket.app.state,
        "websocket_permission_checker",
        None,
    )

    if checker is None:
        raise RuntimeError(
            "WebSocket permission checker is not configured"
        )

    return checker


# ============================================================
# MESSAGE HELPERS
# ============================================================

async def send_error(
    manager: WebSocketManager,
    connection: WSConnection,
    code: str,
    message: str,
    request_id: str | None = None,
) -> None:
    await manager.send_json(
        connection,
        WSError(
            code=code,
            message=message,
            request_id=request_id,
        ).model_dump(),
    )


async def handle_client_message(
    manager: WebSocketManager,
    checker: WSPermissionChecker,
    connection: WSConnection,
    message: WSClientMessage,
) -> None:
    if message.type == WSMessageType.PING:
        await manager.send_json(
            connection,
            {
                "type": WSMessageType.PONG.value,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "request_id": message.request_id,
            },
        )
        return

    if message.type == WSMessageType.PONG:
        return

    if message.type == WSMessageType.SUBSCRIBE:
        if message.topic is None:
            await send_error(
                manager,
                connection,
                "TOPIC_REQUIRED",
                "Subscription topic is required",
                message.request_id,
            )
            return

        allowed = await checker.can_subscribe(
            connection.principal,
            message.topic,
        )

        if not allowed:
            await send_error(
                manager,
                connection,
                "SUBSCRIPTION_FORBIDDEN",
                "Not authorized for this topic",
                message.request_id,
            )
            return

        subscribed = await manager.subscribe(
            connection.connection_id,
            message.topic,
        )

        if not subscribed:
            await send_error(
                manager,
                connection,
                "SUBSCRIPTION_LIMIT",
                "Subscription limit reached",
                message.request_id,
            )
            return

        await manager.send_json(
            connection,
            {
                "type": WSMessageType.SUBSCRIBED.value,
                "topic": message.topic.value,
                "request_id": message.request_id,
            },
        )
        return

    if message.type == WSMessageType.UNSUBSCRIBE:
        if message.topic is None:
            await send_error(
                manager,
                connection,
                "TOPIC_REQUIRED",
                "Unsubscription topic is required",
                message.request_id,
            )
            return

        await manager.unsubscribe(
            connection.connection_id,
            message.topic,
        )

        await manager.send_json(
            connection,
            {
                "type": WSMessageType.UNSUBSCRIBED.value,
                "topic": message.topic.value,
                "request_id": message.request_id,
            },
        )
        return

    await send_error(
        manager,
        connection,
        "UNSUPPORTED_MESSAGE",
        "Only ping, pong, subscribe and unsubscribe "
        "messages are accepted from clients",
        message.request_id,
    )


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
) -> None:
    """
    Real-time event stream.

    Example:
        ws://localhost:8000/api/v1/ws

    Authentication:
        The application-supplied authenticator must
        validate a secure cookie or short-lived
        WebSocket ticket before accepting.

    Never use a long-lived access token in a URL.
    """

    manager: WebSocketManager | None = None
    connection: WSConnection | None = None

    try:
        manager = get_ws_manager(websocket)
        authenticator = get_ws_authenticator(
            websocket
        )
        checker = get_ws_permission_checker(
            websocket
        )

    except RuntimeError:
        logger.exception(
            "WebSocket dependencies unavailable"
        )

        await websocket.close(
            code=1013,
            reason="Service unavailable",
        )
        return

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    try:
        principal = await authenticator.authenticate(
            websocket
        )

    except Exception:
        logger.exception(
            "WebSocket authentication failure"
        )

        await websocket.close(
            code=1011,
            reason="Authentication service failure",
        )
        return

    if principal is None:
        await websocket.close(
            code=WS_CLOSE_UNAUTHORIZED,
            reason="Authentication required",
        )
        return

    # --------------------------------------------------------
    # CONNECTION REGISTRATION
    # --------------------------------------------------------

    connection = await manager.connect(
        websocket,
        principal,
    )

    if connection is None:
        await websocket.close(
            code=WS_CLOSE_TOO_MANY,
            reason="Connection limit reached",
        )
        return

    try:
        await websocket.accept()

        await manager.send_json(
            connection,
            {
                "type": WSMessageType.CONNECTED.value,
                "connection_id": connection.connection_id,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "supported_topics": [
                    topic.value
                    for topic in WSTopic
                ],
            },
        )

        logger.info(
            "WebSocket connected: connection_id=%s",
            connection.connection_id,
        )

        # ----------------------------------------------------
        # RECEIVE LOOP
        # ----------------------------------------------------

        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=RECEIVE_TIMEOUT_SECONDS,
                )

            except asyncio.TimeoutError:
                now = datetime.now(timezone.utc)

                idle_seconds = (
                    now - connection.last_seen_at
                ).total_seconds()

                if idle_seconds >= MAX_IDLE_SECONDS:
                    await websocket.close(
                        code=1001,
                        reason="Connection idle",
                    )
                    break

                sent = await manager.send_json(
                    connection,
                    {
                        "type": WSMessageType.HEARTBEAT.value,
                        "timestamp": now.isoformat(),
                    },
                )

                if not sent:
                    break

                continue

            if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
                await send_error(
                    manager,
                    connection,
                    "MESSAGE_TOO_LARGE",
                    "Message exceeds allowed size",
                )

                await websocket.close(
                    code=1009,
                    reason="Message too large",
                )
                break

            connection.last_seen_at = datetime.now(
                timezone.utc
            )

            try:
                message = WSClientMessage.model_validate_json(
                    raw
                )

            except ValidationError:
                await send_error(
                    manager,
                    connection,
                    "INVALID_MESSAGE",
                    "Invalid WebSocket message",
                )
                continue

            await handle_client_message(
                manager,
                checker,
                connection,
                message,
            )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected: connection_id=%s",
            connection.connection_id,
        )

    except Exception:
        logger.exception(
            "WebSocket connection error"
        )

        try:
            await websocket.close(
                code=1011,
                reason="Internal WebSocket error",
            )
        except (RuntimeError, OSError):
            pass

    finally:
        if connection is not None:
            await manager.disconnect(
                connection.connection_id
            )


# ============================================================
# INTERNAL EVENT PUBLISHING
# ============================================================

async def publish_realtime_event(
    manager: WebSocketManager,
    *,
    topic: WSTopic,
    event_type: str,
    payload: dict[str, Any],
    district_id: int | None = None,
    user_id: str | None = None,
) -> int:
    """
    Publish a domain event to connected clients.

    Call only from trusted backend services.

    Do not pass unfiltered ORM objects, passwords,
    medical records, private phone numbers, or
    other sensitive information.

    Returns:
        Number of WebSocket connections to which
        sending succeeded, not user acknowledgments.
    """

    event = WSEvent(
        topic=topic,
        event_type=event_type,
        payload=payload,
        district_id=district_id,
        user_id=user_id,
    )

    return await manager.broadcast(event)


# ============================================================
# DOMAIN EVENT HELPERS
# ============================================================

async def publish_flood_alert(
    manager: WebSocketManager,
    *,
    district_id: int,
    alert_id: str,
    severity: str,
    status: str,
) -> int:
    """
    Publish only an alert whose operational status
    has already been verified by the alert service.
    """

    return await publish_realtime_event(
        manager,
        topic=WSTopic.FLOOD_ALERTS,
        event_type="flood_alert.updated",
        district_id=district_id,
        payload={
            "alert_id": alert_id,
            "severity": severity,
            "status": status,
        },
    )


async def publish_sos_update(
    manager: WebSocketManager,
    *,
    district_id: int,
    sos_id: str,
    status: str,
    priority: str,
) -> int:
    return await publish_realtime_event(
        manager,
        topic=WSTopic.SOS,
        event_type="sos.updated",
        district_id=district_id,
        payload={
            "sos_id": sos_id,
            "status": status,
            "priority": priority,
        },
    )


async def publish_rescue_update(
    manager: WebSocketManager,
    *,
    district_id: int,
    mission_id: str,
    status: str,
) -> int:
    return await publish_realtime_event(
        manager,
        topic=WSTopic.RESCUE,
        event_type="rescue.mission_updated",
        district_id=district_id,
        payload={
            "mission_id": mission_id,
            "status": status,
        },
    )


async def publish_shelter_update(
    manager: WebSocketManager,
    *,
    district_id: int,
    shelter_id: str,
    occupied: int,
    available: int,
) -> int:
    return await publish_realtime_event(
        manager,
        topic=WSTopic.SHELTERS,
        event_type="shelter.occupancy_updated",
        district_id=district_id,
        payload={
            "shelter_id": shelter_id,
            "occupied": occupied,
            "available": available,
        },
    )


async def publish_rainfall_update(
    manager: WebSocketManager,
    *,
    district_id: int,
    station_id: str,
    rainfall_mm: float,
    observed_at: datetime,
) -> int:
    return await publish_realtime_event(
        manager,
        topic=WSTopic.RAINFALL,
        event_type="rainfall.observed",
        district_id=district_id,
        payload={
            "station_id": station_id,
            "rainfall_mm": rainfall_mm,
            "observed_at": observed_at.isoformat(),
        },
    )


async def publish_river_update(
    manager: WebSocketManager,
    *,
    district_id: int,
    station_id: str,
    water_level_m: float,
    observed_at: datetime,
) -> int:
    return await publish_realtime_event(
        manager,
        topic=WSTopic.RIVERS,
        event_type="river.level_updated",
        district_id=district_id,
        payload={
            "station_id": station_id,
            "water_level_m": water_level_m,
            "observed_at": observed_at.isoformat(),
        },
    )


async def publish_volunteer_update(
    manager: WebSocketManager,
    *,
    district_id: int,
    volunteer_id: str,
    assignment_id: str,
    status: str,
) -> int:
    return await publish_realtime_event(
        manager,
        topic=WSTopic.VOLUNTEERS,
        event_type="volunteer.assignment_updated",
        district_id=district_id,
        payload={
            "volunteer_id": volunteer_id,
            "assignment_id": assignment_id,
            "status": status,
        },
    )


async def publish_user_notification(
    manager: WebSocketManager,
    *,
    user_id: str,
    notification_id: str,
    category: str,
    title: str,
) -> int:
    return await publish_realtime_event(
        manager,
        topic=WSTopic.NOTIFICATIONS,
        event_type="notification.created",
        user_id=user_id,
        payload={
            "notification_id": notification_id,
            "category": category,
            "title": title,
        },
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "router",
    "WSTopic",
    "WSMessageType",
    "WSPrincipal",
    "WSClientMessage",
    "WSEvent",
    "WSAuthenticator",
    "WSPermissionChecker",
    "DefaultWSPermissionChecker",
    "WSConnection",
    "WebSocketManager",
    "publish_realtime_event",
    "publish_flood_alert",
    "publish_sos_update",
    "publish_rescue_update",
    "publish_shelter_update",
    "publish_rainfall_update",
    "publish_river_update",
    "publish_volunteer_update",
    "publish_user_notification",
]
