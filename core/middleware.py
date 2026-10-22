from __future__ import annotations

"""
backend/core/middleware.py

Kerala Flood Relief Coordination Agent

FastAPI middleware for:
- Request ID and correlation ID
- Request processing time
- Structured access logging
- Security response headers
- Request body size limits
- Trusted host validation
- CORS configuration
- HTTP request error handling
- Sensitive endpoint cache protection
- Maintenance mode
- Health check exclusions
"""


import logging
import os
import time
import uuid

from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.core.logging import (
    correlation_id_var,
    request_id_var,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MiddlewareSettings:
    environment: str = "development"

    allowed_origins: tuple[str, ...] = field(
        default_factory=tuple
    )

    allowed_hosts: tuple[str, ...] = (
        "localhost",
        "127.0.0.1",
    )

    allow_credentials: bool = False

    max_request_bytes: int = 10 * 1024 * 1024

    maintenance_mode: bool = False

    enable_hsts: bool = False

    hsts_max_age: int = 31536000

    @classmethod
    def from_environment(cls) -> "MiddlewareSettings":
        environment = os.getenv(
            "APP_ENV",
            "development",
        ).lower()

        origins = tuple(
            value.strip()
            for value in os.getenv(
                "CORS_ALLOWED_ORIGINS",
                "",
            ).split(",")
            if value.strip()
        )

        hosts = tuple(
            value.strip()
            for value in os.getenv(
                "ALLOWED_HOSTS",
                "localhost,127.0.0.1",
            ).split(",")
            if value.strip()
        )

        credentials = (
            os.getenv(
                "CORS_ALLOW_CREDENTIALS",
                "false",
            ).lower()
            == "true"
        )

        maintenance = (
            os.getenv(
                "MAINTENANCE_MODE",
                "false",
            ).lower()
            == "true"
        )

        hsts = (
            os.getenv(
                "ENABLE_HSTS",
                "false",
            ).lower()
            == "true"
        )

        max_bytes = int(
            os.getenv(
                "MAX_REQUEST_BYTES",
                str(10 * 1024 * 1024),
            )
        )

        if max_bytes <= 0:
            raise ValueError(
                "MAX_REQUEST_BYTES must be positive"
            )

        if credentials and "*" in origins:
            raise ValueError(
                "Wildcard CORS origin cannot be used "
                "with credentials."
            )

        if environment == "production":
            if not origins:
                logger.warning(
                    "No production CORS origins configured"
                )

            if "*" in hosts:
                raise ValueError(
                    "Wildcard hosts are not allowed "
                    "in production."
                )

        return cls(
            environment=environment,
            allowed_origins=origins,
            allowed_hosts=hosts,
            allow_credentials=credentials,
            max_request_bytes=max_bytes,
            maintenance_mode=maintenance,
            enable_hsts=hsts,
        )


# ============================================================
# RESPONSE HELPERS
# ============================================================

def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": message,
            },
            "request_id": request_id,
        },
    )

    if request_id:
        response.headers["X-Request-ID"] = request_id

    return response


# ============================================================
# REQUEST ID MIDDLEWARE
# ============================================================

class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Generate trusted server-side request identifiers.

    Client-supplied request IDs are not trusted.
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        request_id = str(uuid.uuid4())

        correlation_id = request_id

        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        request_token = request_id_var.set(
            request_id
        )

        correlation_token = correlation_id_var.set(
            correlation_id
        )

        try:
            response = await call_next(request)

            response.headers[
                "X-Request-ID"
            ] = request_id

            return response

        finally:
            request_id_var.reset(request_token)
            correlation_id_var.reset(
                correlation_token
            )


# ============================================================
# REQUEST TIMING MIDDLEWARE
# ============================================================

class RequestTimingMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        response.headers[
            "X-Response-Time"
        ] = f"{elapsed_ms:.2f}ms"

        return response


# ============================================================
# STRUCTURED ACCESS LOGGING
# ============================================================

class AccessLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log request metadata without sensitive payloads.

    Does not log:
    - Request bodies
    - Query strings
    - Authorization headers
    - Cookies
    - GPS coordinates
    - Patient or SOS personal information
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        start = time.perf_counter()

        try:
            response = await call_next(request)

            elapsed_ms = round(
                (
                    time.perf_counter() - start
                ) * 1000,
                2,
            )

            route = request.scope.get("route")

            route_name = (
                getattr(route, "path", None)
                or "unmatched"
            )

            logger.info(
                "HTTP request completed",
                extra={
                    "structured_data": {
                        "event": "http.request",
                        "method": request.method,
                        "route": route_name,
                        "status_code": response.status_code,
                        "duration_ms": elapsed_ms,
                    }
                },
            )

            return response

        except Exception:
            logger.error(
                "HTTP request failed",
                extra={
                    "structured_data": {
                        "event": "http.request_failed",
                        "method": request.method,
                    }
                },
            )

            raise


# ============================================================
# SECURITY HEADERS
# ============================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):

    def __init__(
        self,
        app,
        *,
        enable_hsts: bool = False,
        hsts_max_age: int = 31536000,
    ):
        super().__init__(app)

        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        response = await call_next(request)

        response.headers[
            "X-Content-Type-Options"
        ] = "nosniff"

        response.headers[
            "X-Frame-Options"
        ] = "DENY"

        response.headers[
            "Referrer-Policy"
        ] = "no-referrer"

        response.headers[
            "Permissions-Policy"
        ] = (
            "camera=(), microphone=(), "
            "geolocation=(self)"
        )

        response.headers[
            "Content-Security-Policy"
        ] = (
            "default-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'none'; "
            "form-action 'self'"
        )

        if self.enable_hsts:
            response.headers[
                "Strict-Transport-Security"
            ] = (
                f"max-age={self.hsts_max_age}; "
                "includeSubDomains"
            )

        return response


# ============================================================
# REQUEST BODY SIZE LIMIT
# ============================================================

class RequestSizeLimitMiddleware:
    """
    ASGI middleware that limits incoming request bodies.

    Enforces limits for both Content-Length and
    streamed/chunked request bodies.
    """

    def __init__(
        self,
        app,
        *,
        max_request_bytes: int,
    ):
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(
        self,
        scope,
        receive,
        send,
    ):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get(
            "content-length"
        )

        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                response = error_response(
                    400,
                    "INVALID_CONTENT_LENGTH",
                    "Invalid Content-Length header.",
                )
                await response(scope, receive, send)
                return

            if (
                declared_size < 0
                or declared_size > self.max_request_bytes
            ):
                response = error_response(
                    413,
                    "REQUEST_TOO_LARGE",
                    "Request body exceeds size limit.",
                )
                await response(scope, receive, send)
                return

        received_bytes = 0
        exceeded = False
        response_started = False

        async def limited_receive():
            nonlocal received_bytes, exceeded

            message = await receive()

            if message["type"] == "http.request":
                received_bytes += len(
                    message.get("body", b"")
                )

                if received_bytes > self.max_request_bytes:
                    exceeded = True

                    return {
                        "type": "http.disconnect"
                    }

            return message

        async def tracked_send(message):
            nonlocal response_started

            if message["type"] == "http.response.start":
                response_started = True

            await send(message)

        await self.app(
            scope,
            limited_receive,
            tracked_send,
        )

        if exceeded:
            logger.warning(
                "Streaming request exceeded size limit"
            )
            # Once an ASGI response has started, a
            # replacement 413 response cannot be sent.
            # The connection should be terminated by
            # the ASGI server or reverse proxy.


# ============================================================
# SENSITIVE ENDPOINT CACHE PROTECTION
# ============================================================

SENSITIVE_PREFIXES = (
    "/api/auth",
    "/api/sos",
    "/api/rescue",
    "/api/healthcare",
    "/api/volunteers",
    "/api/missing-persons",
    "/api/agents",
    "/api/audit",
    "/api/users",
)


class SensitiveCacheMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        response = await call_next(request)

        path = request.url.path

        if any(
            path == prefix
            or path.startswith(prefix + "/")
            for prefix in SENSITIVE_PREFIXES
        ):
            response.headers[
                "Cache-Control"
            ] = "no-store"

            response.headers[
                "Pragma"
            ] = "no-cache"

        return response


# ============================================================
# MAINTENANCE MODE
# ============================================================

MAINTENANCE_EXCLUDED_PATHS = {
    "/health",
    "/health/live",
    "/health/ready",
}

EMERGENCY_PRIORITY_PREFIXES = (
    "/api/sos",
    "/api/rescue",
    "/api/healthcare",
)


class MaintenanceMiddleware(BaseHTTPMiddleware):
    """
    Allow emergency operations during maintenance.

    Actual emergency service availability must be
    independently monitored.
    """

    def __init__(
        self,
        app,
        *,
        enabled: bool = False,
    ):
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        if not self.enabled:
            return await call_next(request)

        path = request.url.path

        if path in MAINTENANCE_EXCLUDED_PATHS:
            return await call_next(request)

        if any(
            path == prefix
            or path.startswith(prefix + "/")
            for prefix in EMERGENCY_PRIORITY_PREFIXES
        ):
            return await call_next(request)

        return error_response(
            503,
            "MAINTENANCE_MODE",
            "Service temporarily unavailable.",
            getattr(
                request.state,
                "request_id",
                None,
            ),
        )


# ============================================================
# MIDDLEWARE REGISTRATION
# ============================================================

def register_middlewares(
    app: FastAPI,
    settings: MiddlewareSettings | None = None,
) -> None:
    """
    Register middleware during application startup.

    Starlette executes the last added middleware
    first on incoming requests.

    Do not also install RequestLoggingMiddleware
    from core/logging.py, because it duplicates
    request context and access logging.
    """

    config = (
        settings
        or MiddlewareSettings.from_environment()
    )

    app.add_middleware(
        AccessLoggingMiddleware
    )

    app.add_middleware(
        RequestTimingMiddleware
    )

    app.add_middleware(
        SensitiveCacheMiddleware
    )

    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=config.enable_hsts,
        hsts_max_age=config.hsts_max_age,
    )

    app.add_middleware(
        MaintenanceMiddleware,
        enabled=config.maintenance_mode,
    )

    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_request_bytes=config.max_request_bytes,
    )

    app.add_middleware(
        RequestContextMiddleware
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(
            config.allowed_hosts
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            config.allowed_origins
        ),
        allow_credentials=config.allow_credentials,
        allow_methods=[
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
        ],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-CSRF-Token",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time",
        ],
        max_age=600,
    )
