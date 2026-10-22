from __future__ import annotations

"""
backend/core/security.py

Kerala Flood Relief Coordination Agent

Security features:
- Argon2id password hashing
- Password verification
- JWT access and refresh tokens
- Token expiration and validation
- OAuth2 Bearer authentication
- Current user authentication
- Role-based access integration
- Account status verification
- Password strength validation
- Secure random token generation
- Refresh token rotation support
- Token revocation using Redis
- Authentication error handling

Dependencies:
    pip install fastapi sqlalchemy asyncpg
    pip install "pwdlib[argon2]" PyJWT redis
    pip install pydantic-settings email-validator

Environment:
    JWT_SECRET_KEY=<secure-random-secret>
    JWT_ALGORITHM=HS256
    JWT_ISSUER=kerala-flood-relief
    JWT_AUDIENCE=kerala-flood-api
    ACCESS_TOKEN_EXPIRE_MINUTES=30
    REFRESH_TOKEN_EXPIRE_DAYS=7
"""


import hashlib
import logging
import os
import re
import secrets
import uuid

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Annotated

import jwt

from fastapi import Depends, Request
from fastapi.security import (
    OAuth2PasswordBearer,
)
from jwt import (
    ExpiredSignatureError,
    InvalidTokenError as JWTInvalidTokenError,
)
from pwdlib import PasswordHash
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    InvalidTokenError,
    TokenExpiredError,
    PermissionDeniedError,
    AccountDisabledError,
    ValidationError,
)
from backend.database.session import get_db
from backend.models.user import User


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SecuritySettings:
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"

    jwt_issuer: str = "kerala-flood-relief"
    jwt_audience: str = "kerala-flood-api"

    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    password_min_length: int = 12
    password_max_length: int = 128

    @classmethod
    def from_environment(cls) -> "SecuritySettings":
        secret = os.getenv("JWT_SECRET_KEY")

        if not secret or len(secret) < 32:
            raise RuntimeError(
                "JWT_SECRET_KEY must contain at least "
                "32 characters."
            )

        access_minutes = int(
            os.getenv(
                "ACCESS_TOKEN_EXPIRE_MINUTES",
                "30",
            )
        )

        refresh_days = int(
            os.getenv(
                "REFRESH_TOKEN_EXPIRE_DAYS",
                "7",
            )
        )

        if access_minutes <= 0 or refresh_days <= 0:
            raise ValueError(
                "Token expiration must be positive."
            )

        return cls(
            jwt_secret_key=secret,
            jwt_algorithm="HS256",
            jwt_issuer=os.getenv(
                "JWT_ISSUER",
                "kerala-flood-relief",
            ),
            jwt_audience=os.getenv(
                "JWT_AUDIENCE",
                "kerala-flood-api",
            ),
            access_token_expire_minutes=access_minutes,
            refresh_token_expire_days=refresh_days,
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# PASSWORD HASHING
# ============================================================

password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2id.
    """

    if not isinstance(password, str):
        raise ValidationError(
            "Password must be a string."
        )

    return password_hasher.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a password against its stored hash.
    """

    if not plain_password or not hashed_password:
        return False

    try:
        return password_hasher.verify(
            plain_password,
            hashed_password,
        )
    except (ValueError, TypeError):
        return False


def validate_password_strength(
    password: str,
    *,
    min_length: int = 12,
    max_length: int = 128,
) -> None:
    """
    Validate password length and basic strength.

    Use breached-password screening separately.
    """

    if not isinstance(password, str):
        raise ValidationError(
            "Password must be a string."
        )

    if len(password) < min_length:
        raise ValidationError(
            f"Password must contain at least "
            f"{min_length} characters."
        )

    if len(password) > max_length:
        raise ValidationError(
            "Password exceeds maximum length."
        )

    if password.isspace():
        raise ValidationError(
            "Password cannot contain only spaces."
        )


def needs_password_rehash(
    hashed_password: str,
) -> bool:
    return password_hasher.check_needs_rehash(
        hashed_password
    )


# ============================================================
# SECURE TOKEN GENERATION
# ============================================================

def generate_secure_token(
    length: int = 32,
) -> str:
    if length < 16:
        raise ValueError(
            "Token entropy length is too small."
        )

    return secrets.token_urlsafe(length)


def generate_token_id() -> str:
    return str(uuid.uuid4())


def hash_token(token: str) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


# ============================================================
# JWT TOKEN MANAGEMENT
# ============================================================

class TokenManager:
    """
    Create and validate signed JWT tokens.
    """

    def __init__(
        self,
        settings: SecuritySettings,
    ):
        self.settings = settings

    def _create_token(
        self,
        *,
        user_id: int,
        token_type: str,
        expires_delta: timedelta,
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        now = utc_now()

        expires_at = now + expires_delta

        payload: dict[str, Any] = {
            "sub": str(user_id),
            "type": token_type,
            "jti": generate_token_id(),
            "iss": self.settings.jwt_issuer,
            "aud": self.settings.jwt_audience,
            "iat": now,
            "nbf": now,
            "exp": expires_at,
        }

        if additional_claims:
            reserved = {
                "sub",
                "type",
                "jti",
                "iss",
                "aud",
                "iat",
                "nbf",
                "exp",
            }

            if reserved.intersection(
                additional_claims
            ):
                raise ValueError(
                    "Reserved JWT claims cannot be overridden."
                )

            payload.update(additional_claims)

        return jwt.encode(
            payload,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
        )

    def create_access_token(
        self,
        user_id: int,
        *,
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        return self._create_token(
            user_id=user_id,
            token_type="access",
            expires_delta=timedelta(
                minutes=(
                    self.settings
                    .access_token_expire_minutes
                )
            ),
            additional_claims=additional_claims,
        )

    def create_refresh_token(
        self,
        user_id: int,
    ) -> str:
        return self._create_token(
            user_id=user_id,
            token_type="refresh",
            expires_delta=timedelta(
                days=(
                    self.settings
                    .refresh_token_expire_days
                )
            ),
        )

    def decode_token(
        self,
        token: str,
        *,
        expected_type: str,
    ) -> dict[str, Any]:
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[
                    self.settings.jwt_algorithm
                ],
                issuer=self.settings.jwt_issuer,
                audience=self.settings.jwt_audience,
                options={
                    "require": [
                        "sub",
                        "type",
                        "jti",
                        "iss",
                        "aud",
                        "iat",
                        "nbf",
                        "exp",
                    ],
                },
                leeway=5,
            )

        except ExpiredSignatureError as exc:
            raise TokenExpiredError() from exc

        except JWTInvalidTokenError as exc:
            raise InvalidTokenError() from exc

        if payload.get("type") != expected_type:
            raise InvalidTokenError(
                "Invalid token type."
            )

        subject = payload.get("sub")

        if (
            not isinstance(subject, str)
            or not subject.isdigit()
        ):
            raise InvalidTokenError(
                "Invalid token subject."
            )

        if not isinstance(
            payload.get("jti"),
            str,
        ):
            raise InvalidTokenError(
                "Invalid token identifier."
            )

        return payload


# ============================================================
# TOKEN REVOCATION
# ============================================================

class TokenRevocationService:
    """
    Redis-backed token revocation.

    Revoked tokens remain blocked until expiration.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str = "kerala_flood:revoked",
    ):
        self.redis = redis
        self.prefix = prefix

    def _key(self, jti: str) -> str:
        return (
            f"{self.prefix}:"
            f"{hash_token(jti)}"
        )

    async def revoke(
        self,
        payload: dict[str, Any],
    ) -> None:
        jti = payload.get("jti")
        exp = payload.get("exp")

        if not isinstance(jti, str):
            raise InvalidTokenError()

        if not isinstance(exp, (int, float)):
            raise InvalidTokenError()

        remaining = int(
            exp - utc_now().timestamp()
        )

        if remaining <= 0:
            return

        await self.redis.set(
            self._key(jti),
            "1",
            ex=remaining + 10,
        )

    async def is_revoked(
        self,
        jti: str,
    ) -> bool:
        result = await self.redis.exists(
            self._key(jti)
        )

        return bool(result)


# ============================================================
# REFRESH TOKEN ROTATION
# ============================================================

class RefreshTokenService:
    """
    Refresh token rotation with replay detection.

    Redis SET NX prevents two concurrent requests
    from successfully consuming the same token.
    """

    def __init__(
        self,
        token_manager: TokenManager,
        redis: Redis,
    ):
        self.token_manager = token_manager
        self.redis = redis

    async def rotate(
        self,
        refresh_token: str,
    ) -> dict[str, str]:
        payload = self.token_manager.decode_token(
            refresh_token,
            expected_type="refresh",
        )

        jti = payload["jti"]
        user_id = int(payload["sub"])

        remaining = int(
            payload["exp"]
            - utc_now().timestamp()
        )

        if remaining <= 0:
            raise TokenExpiredError()

        key = (
            "kerala_flood:refresh_used:"
            f"{hash_token(jti)}"
        )

        consumed = await self.redis.set(
            key,
            "1",
            nx=True,
            ex=remaining + 10,
        )

        if not consumed:
            raise InvalidTokenError(
                "Refresh token has already been used."
            )

        return {
            "access_token": (
                self.token_manager.create_access_token(
                    user_id
                )
            ),
            "refresh_token": (
                self.token_manager.create_refresh_token(
                    user_id
                )
            ),
            "token_type": "bearer",
        }


# ============================================================
# OAUTH2 AUTHENTICATION
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    auto_error=False,
)


def get_token_manager() -> TokenManager:
    """
    Override or cache this dependency during startup.
    """
    return TokenManager(
        SecuritySettings.from_environment()
    )


def get_redis_client(
    request: Request,
) -> Redis:
    redis = getattr(
        request.app.state,
        "redis",
        None,
    )

    if redis is None:
        raise RuntimeError(
            "Redis authentication service is not configured."
        )

    return redis


async def get_current_user(
    token: Annotated[
        str | None,
        Depends(oauth2_scheme),
    ],
    db: Annotated[
        AsyncSession,
        Depends(get_db),
    ],
    token_manager: Annotated[
        TokenManager,
        Depends(get_token_manager),
    ],
    redis: Annotated[
        Redis,
        Depends(get_redis_client),
    ],
) -> User:
    """
    Authenticate a user using a JWT access token.

    Never trust client-supplied roles or user IDs.
    """

    if not token:
        raise AuthenticationError()

    payload = token_manager.decode_token(
        token,
        expected_type="access",
    )

    revocation = TokenRevocationService(redis)

    if await revocation.is_revoked(
        payload["jti"]
    ):
        raise InvalidTokenError(
            "Token has been revoked."
        )

    user_id = int(payload["sub"])

    result = await db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    user = result.scalar_one_or_none()

    if user is None:
        raise InvalidCredentialsError()

    if not getattr(
        user,
        "is_active",
        False,
    ):
        raise AccountDisabledError()

    return user


async def get_current_verified_user(
    current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> User:
    if not getattr(
        current_user,
        "is_verified",
        False,
    ):
        raise PermissionDeniedError(
            "Account verification required."
        )

    return current_user


# ============================================================
# OPTIONAL AUTHENTICATION
# ============================================================

async def get_optional_current_user(
    token: Annotated[
        str | None,
        Depends(oauth2_scheme),
    ],
    db: Annotated[
        AsyncSession,
        Depends(get_db),
    ],
    token_manager: Annotated[
        TokenManager,
        Depends(get_token_manager),
    ],
    redis: Annotated[
        Redis,
        Depends(get_redis_client),
    ],
) -> User | None:
    if not token:
        return None

    payload = token_manager.decode_token(
        token,
        expected_type="access",
    )

    revocation = TokenRevocationService(redis)

    if await revocation.is_revoked(
        payload["jti"]
    ):
        raise InvalidTokenError()

    user_id = int(payload["sub"])

    user = await db.get(
        User,
        user_id,
    )

    if user is None:
        raise InvalidCredentialsError()

    if not getattr(
        user,
        "is_active",
        False,
    ):
        raise AccountDisabledError()

    return user


# ============================================================
# ROLE AND PERMISSION INTEGRATION
# ============================================================

def get_user_roles(
    user: User,
) -> set[str]:
    raw_roles = getattr(
        user,
        "roles",
        None,
    )

    if raw_roles is None:
        role = getattr(
            user,
            "role",
            None,
        )
        raw_roles = (
            [role]
            if role is not None
            else []
        )

    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]

    return {
        str(getattr(role, "value", role))
        for role in raw_roles
    }


def user_has_role(
    user: User,
    role: str,
) -> bool:
    return role in get_user_roles(user)


def require_user_role(
    user: User,
    allowed_roles: set[str],
) -> None:
    if not get_user_roles(user).intersection(
        allowed_roles
    ):
        raise PermissionDeniedError(
            "Insufficient role privileges."
        )


# ============================================================
# LOGIN CREDENTIAL VERIFICATION
# ============================================================

async def authenticate_user(
    db: AsyncSession,
    *,
    email: str,
    password: str,
) -> User:
    """
    Verify login credentials.

    Apply per-account and per-IP rate limiting
    before calling this function.
    """

    normalized_email = email.strip().lower()

    result = await db.execute(
        select(User).where(
            User.email == normalized_email
        )
    )

    user = result.scalar_one_or_none()

    if user is None:
        # Perform a dummy hash verification to reduce
        # timing differences for unknown accounts.
        verify_password(
            password,
            DUMMY_PASSWORD_HASH,
        )

        raise InvalidCredentialsError()

    if not verify_password(
        password,
        user.hashed_password,
    ):
        raise InvalidCredentialsError()

    if not getattr(
        user,
        "is_active",
        False,
    ):
        raise AccountDisabledError()

    return user


DUMMY_PASSWORD_HASH = hash_password(
    secrets.token_urlsafe(32)
)


# ============================================================
# SECURITY UTILITIES
# ============================================================

def secure_compare(
    first: str,
    second: str,
) -> bool:
    return secrets.compare_digest(
        first.encode("utf-8"),
        second.encode("utf-8"),
    )


def sanitize_filename(
    filename: str,
) -> str:
    """
    Produce a safe display/storage basename.

    Generate a separate random storage filename
    for uploaded files in production.
    """

    filename = filename.replace(
        "\\",
        "/",
    ).split("/")[-1]

    filename = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        filename,
    )

    filename = filename.lstrip(".")

    return filename[:255] or "unnamed_file"


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def verify_csrf_token(
    submitted_token: str,
    expected_token: str,
) -> bool:
    if not submitted_token or not expected_token:
        return False

    return secure_compare(
        submitted_token,
        expected_token,
    )


# ============================================================
# SECURITY HEADERS
# ============================================================

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def get_security_headers() -> dict[str, str]:
    return SECURITY_HEADERS.copy()


# ============================================================
# FASTAPI USAGE
# ============================================================

"""
from fastapi import APIRouter, Depends

from backend.core.security import (
    get_current_user,
    get_current_verified_user,
)

router = APIRouter(
    prefix="/api/users",
    tags=["Users"],
)


@router.get("/me")
async def read_profile(
    user: User = Depends(get_current_user),
):
    return {
        "id": user.id,
        "email": user.email,
        "is_verified": user.is_verified,
    }
"""
