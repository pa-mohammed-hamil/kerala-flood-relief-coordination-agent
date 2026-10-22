from __future__ import annotations

"""
backend/core/rate_limiter.py

Kerala Flood Relief Coordination Agent

Distributed FastAPI rate limiting using Redis.

Features:
- Redis-backed atomic sliding-window rate limiting
- Per-user and per-client limits
- Route-specific rate limits
- Emergency SOS and rescue endpoint exemptions
- Authentication brute-force protection
- Trusted client identity handling
- RateLimit response headers
- Retry-After support
- Fail-closed policy for protected endpoints
- Configurable fail-open policy for ordinary endpoints
- Async Redis integration

Dependency:
    pip install redis fastapi

Environment:
    REDIS_URL=redis://localhost:6379/0
    RATE_LIMIT_ENABLED=true
    RATE_LIMIT_DEFAULT=100
    RATE_LIMIT_WINDOW=60
    RATE_LIMIT_AUTH=10
    RATE_LIMIT_AUTH_WINDOW=60
"""


import hashlib
import hmac
import ipaddress
import logging
import math
import os
import secrets
import time

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RateLimitSettings:
    enabled: bool = True

    redis_url: str = "redis://localhost:6379/0"

    default_limit: int = 100
    default_window: int = 60

    auth_limit: int = 10
    auth_window: int = 60

    sos_limit: int = 30
    sos_window: int = 60

    ai_limit: int = 30
    ai_window: int = 60

    forecast_limit: int = 60
    forecast_window: int = 60

    fail_open: bool = False

    key_prefix: str = "kerala_flood:ratelimit"

    @classmethod
    def from_environment(cls) -> "RateLimitSettings":
        def env_bool(name: str, default: bool) -> bool:
            value = os.getenv(
                name,
                str(default),
            ).lower()

            return value in {
                "true",
                "1",
                "yes",
                "on",
            }

        config = cls(
            enabled=env_bool(
                "RATE_LIMIT_ENABLED",
                True,
            ),
            redis_url=os.getenv(
                "REDIS_URL",
                "redis://localhost:6379/0",
            ),
            default_limit=int(
                os.getenv(
                    "RATE_LIMIT_DEFAULT",
                    "100",
                )
            ),
            default_window=int(
                os.getenv(
                    "RATE_LIMIT_WINDOW",
                    "60",
                )
            ),
            auth_limit=int(
                os.getenv(
                    "RATE_LIMIT_AUTH",
                    "10",
                )
            ),
            auth_window=int(
                os.getenv(
                    "RATE_LIMIT_AUTH_WINDOW",
                    "60",
                )
            ),
            fail_open=env_bool(
                "RATE_LIMIT_FAIL_OPEN",
                False,
            ),
        )

        for name in (
            "default_limit",
            "default_window",
            "auth_limit",
            "auth_window",
            "sos_limit",
            "sos_window",
            "ai_limit",
            "ai_window",
            "forecast_limit",
            "forecast_window",
        ):
            if getattr(config, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        return config


# ============================================================
# ROUTE POLICIES
# ============================================================

@dataclass(frozen=True)
class RateLimitPolicy:
    limit: int
    window_seconds: int
    fail_open: bool = False


EMERGENCY_EXEMPT_PATHS = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
})

EMERGENCY_PRIORITY_PREFIXES = (
    "/api/sos",
    "/api/rescue",
    "/api/healthcare/emergency",
    "/api/ambulance",
)

AUTH_PREFIXES = (
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/refresh",
    "/api/auth/password-reset",
)

AI_PREFIXES = (
    "/api/agents",
    "/api/ai",
    "/api/rag",
)

FORECAST_PREFIXES = (
    "/api/flood-forecast",
    "/api/flood/forecast",
)


def path_matches(
    path: str,
    prefix: str,
) -> bool:
    return (
        path == prefix
        or path.startswith(prefix + "/")
    )


def is_emergency_path(path: str) -> bool:
    return any(
        path_matches(path, prefix)
        for prefix in EMERGENCY_PRIORITY_PREFIXES
    )


def get_rate_limit_policy(
    path: str,
    settings: RateLimitSettings,
) -> RateLimitPolicy | None:
    if path in EMERGENCY_EXEMPT_PATHS:
        return None

    if is_emergency_path(path):
        return RateLimitPolicy(
            limit=settings.sos_limit,
            window_seconds=settings.sos_window,
            fail_open=True,
        )

    if any(
        path_matches(path, prefix)
        for prefix in AUTH_PREFIXES
    ):
        return RateLimitPolicy(
            limit=settings.auth_limit,
            window_seconds=settings.auth_window,
            fail_open=False,
        )

    if any(
        path_matches(path, prefix)
        for prefix in AI_PREFIXES
    ):
        return RateLimitPolicy(
            limit=settings.ai_limit,
            window_seconds=settings.ai_window,
            fail_open=settings.fail_open,
        )

    if any(
        path_matches(path, prefix)
        for prefix in FORECAST_PREFIXES
    ):
        return RateLimitPolicy(
            limit=settings.forecast_limit,
            window_seconds=settings.forecast_window,
            fail_open=settings.fail_open,
        )

    return RateLimitPolicy(
        limit=settings.default_limit,
        window_seconds=settings.default_window,
        fail_open=settings.fail_open,
    )


# ============================================================
# CLIENT IDENTITY
# ============================================================

def normalize_ip(value: str) -> str:
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return "unknown"


def get_client_identity(
    request: Request,
) -> str:
    """
    Use the ASGI client address.

    Proxy headers must only be processed by a
    correctly configured trusted proxy layer.
    """

    user = getattr(
        request.state,
        "authenticated_user",
        None,
    )

    if user is not None:
        user_id = getattr(user, "id", None)

        if user_id is not None:
            return f"user:{user_id}"

    client = request.client

    if client is None:
        return "ip:unknown"

    return f"ip:{normalize_ip(client.host)}"


def hash_identity(
    identity: str,
) -> str:
    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


# ============================================================
# ATOMIC REDIS SLIDING WINDOW
# ============================================================

SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]

local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local member = ARGV[3]

local current_time = redis.call('TIME')
local now_ms = (
    tonumber(current_time[1]) * 1000
    + math.floor(tonumber(current_time[2]) / 1000)
)

local cutoff = now_ms - window_ms

redis.call(
    'ZREMRANGEBYSCORE',
    key,
    '-inf',
    cutoff
)

local count = redis.call(
    'ZCARD',
    key
)

if count >= limit then
    local oldest = redis.call(
        'ZRANGE',
        key,
        0,
        0,
        'WITHSCORES'
    )

    local retry_ms = window_ms

    if #oldest >= 2 then
        retry_ms = math.max(
            1,
            tonumber(oldest[2]) + window_ms - now_ms
        )
    end

    return {
        0,
        count,
        retry_ms
    }
end

redis.call(
    'ZADD',
    key,
    now_ms,
    member
)

redis.call(
    'PEXPIRE',
    key,
    window_ms + 1000
)

return {
    1,
    count + 1,
    0
}
"""


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int
    window_seconds: int


# ============================================================
# REDIS RATE LIMITER
# ============================================================

class RedisRateLimiter:
    def __init__(
        self,
        redis: Redis,
        settings: RateLimitSettings,
    ):
        self.redis = redis
        self.settings = settings

    def build_key(
        self,
        identity: str,
        scope: str,
    ) -> str:
        identity_hash = hash_identity(identity)

        scope_hash = hashlib.sha256(
            scope.encode("utf-8")
        ).hexdigest()[:20]

        return (
            f"{self.settings.key_prefix}:"
            f"{scope_hash}:{identity_hash}"
        )

    async def check(
        self,
        *,
        identity: str,
        scope: str,
        policy: RateLimitPolicy,
    ) -> RateLimitResult:
        key = self.build_key(
            identity,
            scope,
        )

        member = (
            f"{time.time_ns()}:"
            f"{secrets.token_hex(12)}"
        )

        result = await self.redis.eval(
            SLIDING_WINDOW_SCRIPT,
            1,
            key,
            policy.limit,
            policy.window_seconds * 1000,
            member,
        )

        allowed = bool(int(result[0]))
        count = int(result[1])
        retry_ms = int(result[2])

        return RateLimitResult(
            allowed=allowed,
            limit=policy.limit,
            remaining=max(
                0,
                policy.limit - count,
            ),
            retry_after=(
                math.ceil(retry_ms / 1000)
                if not allowed
                else 0
            ),
            window_seconds=policy.window_seconds,
        )


# ============================================================
# HTTP RESPONSE HELPERS
# ============================================================

def rate_limit_headers(
    result: RateLimitResult,
) -> dict[str, str]:
    headers = {
        "RateLimit-Limit": str(result.limit),
        "RateLimit-Remaining": str(
            result.remaining
        ),
        "RateLimit-Reset": str(
            result.retry_after
        ),
    }

    if not result.allowed:
        headers["Retry-After"] = str(
            max(1, result.retry_after)
        )

    return headers


def rate_limit_exceeded_response(
    result: RateLimitResult,
    request_id: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": (
                    "Too many requests. "
                    "Please retry later."
                ),
            },
            "request_id": request_id,
            "retry_after": result.retry_after,
        },
        headers=rate_limit_headers(result),
    )


def limiter_unavailable_response(
    request_id: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "error": {
                "code": "RATE_LIMITER_UNAVAILABLE",
                "message": (
                    "Request protection is temporarily "
                    "unavailable."
                ),
            },
            "request_id": request_id,
        },
        headers={
            "Retry-After": "5",
        },
    )


# ============================================================
# FASTAPI MIDDLEWARE
# ============================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        redis: Redis,
        settings: RateLimitSettings,
    ):
        super().__init__(app)

        self.settings = settings

        self.limiter = RedisRateLimiter(
            redis,
            settings,
        )

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        if not self.settings.enabled:
            return await call_next(request)

        path = request.url.path

        policy = get_rate_limit_policy(
            path,
            self.settings,
        )

        if policy is None:
            return await call_next(request)

        identity = get_client_identity(
            request
        )

        scope = (
            f"{request.method}:"
            f"{path}"
        )

        try:
            result = await self.limiter.check(
                identity=identity,
                scope=scope,
                policy=policy,
            )

        except RedisError:
            logger.exception(
                "Redis rate limiter unavailable"
            )

            if policy.fail_open:
                return await call_next(request)

            return limiter_unavailable_response(
                getattr(
                    request.state,
                    "request_id",
                    None,
                )
            )

        if not result.allowed:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "structured_data": {
                        "event": "rate_limit.exceeded",
                        "scope_hash": hashlib.sha256(
                            scope.encode("utf-8")
                        ).hexdigest()[:16],
                    }
                },
            )

            return rate_limit_exceeded_response(
                result,
                getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            )

        response = await call_next(request)

        for name, value in rate_limit_headers(
            result
        ).items():
            response.headers[name] = value

        return response


# ============================================================
# APPLICATION REGISTRATION
# ============================================================

def register_rate_limiter(
    app: FastAPI,
    *,
    redis: Redis,
    settings: RateLimitSettings | None = None,
) -> None:
    """
    Register rate limiting during FastAPI startup.

    Redis must be created and closed by the
    application's lifespan manager.

    Example:

        redis = Redis.from_url(
            "redis://localhost:6379/0",
            decode_responses=False,
        )

        register_rate_limiter(
            app,
            redis=redis,
        )
    """

    config = (
        settings
        or RateLimitSettings.from_environment()
    )

    app.add_middleware(
        RateLimitMiddleware,
        redis=redis,
        settings=config,
    )


# ============================================================
# OPTIONAL MANUAL RATE LIMIT CHECK
# ============================================================

async def enforce_rate_limit(
    request: Request,
    *,
    limiter: RedisRateLimiter,
    limit: int,
    window_seconds: int,
    scope: str,
) -> RateLimitResult:
    policy = RateLimitPolicy(
        limit=limit,
        window_seconds=window_seconds,
        fail_open=False,
    )

    identity = get_client_identity(request)

    result = await limiter.check(
        identity=identity,
        scope=scope,
        policy=policy,
    )

    if not result.allowed:
        from backend.core.exceptions import (
            RateLimitExceededError,
        )

        raise RateLimitExceededError(
            "Rate limit exceeded.",
            details={
                "retry_after": result.retry_after,
            },
        )

    return result
