from __future__ import annotations

"""
backend/api/auth.py

Kerala Flood Relief Coordination Agent
Authentication and account security API.

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Endpoints:
    POST /auth/register
    POST /auth/login
    POST /auth/token
    POST /auth/refresh
    POST /auth/logout
    POST /auth/logout-all
    GET  /auth/me
    POST /auth/change-password
    POST /auth/forgot-password
    POST /auth/reset-password
    GET  /auth/sessions
    DELETE /auth/sessions/{session_id}

Required integrations:
    backend.core.security
    backend.services.auth_service
    backend.repositories.user_repository
    backend.database.session
    backend.dependencies

Security:
    - Password hashing
    - Short-lived JWT access tokens
    - Rotating refresh tokens
    - Hashed refresh-token storage
    - Session revocation
    - Login rate limiting
    - Account lockout
    - Password reset tokens
    - Role-based access control
    - Audit logging
    - Generic credential errors

Important:
    Registration never permits users to assign themselves
    administrative or emergency-response privileges.
"""


import logging
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user
from backend.services.auth_service import AuthService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

DbSession = Annotated[
    AsyncSession,
    Depends(get_db),
]

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# CONSTANTS
# ============================================================

MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128

GENERIC_AUTH_ERROR = (
    "Invalid credentials or account unavailable"
)

GENERIC_RESET_MESSAGE = (
    "If the account exists, password reset "
    "instructions will be sent."
)


# ============================================================
# ENUMERATIONS
# ============================================================

class RegistrationAccountType(str, Enum):
    CITIZEN = "citizen"
    VOLUNTEER = "volunteer"


class AuthenticationMethod(str, Enum):
    PASSWORD = "password"
    REFRESH_TOKEN = "refresh_token"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class RegisterRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    email: EmailStr

    password: SecretStr

    account_type: RegistrationAccountType = (
        RegistrationAccountType.CITIZEN
    )

    district_id: int | None = Field(
        default=None,
        ge=1,
    )

    preferred_language: str = Field(
        default="ml",
        min_length=2,
        max_length=10,
    )

    accept_terms: bool

    accept_privacy_policy: bool

    @field_validator("password")
    @classmethod
    def validate_password(
        cls,
        value: SecretStr,
    ) -> SecretStr:
        password = value.get_secret_value()

        if not (
            MIN_PASSWORD_LENGTH
            <= len(password)
            <= MAX_PASSWORD_LENGTH
        ):
            raise ValueError(
                "Password must contain 12 to 128 characters"
            )

        if not any(char.isalpha() for char in password):
            raise ValueError(
                "Password must contain a letter"
            )

        if not any(char.isdigit() for char in password):
            raise ValueError(
                "Password must contain a number"
            )

        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr

    password: SecretStr

    remember_me: bool = False


class RefreshTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: SecretStr


class LogoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: SecretStr


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: SecretStr

    new_password: SecretStr

    @field_validator("new_password")
    @classmethod
    def validate_new_password(
        cls,
        value: SecretStr,
    ) -> SecretStr:
        return RegisterRequest.validate_password(value)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    reset_token: SecretStr

    new_password: SecretStr

    @field_validator("new_password")
    @classmethod
    def validate_reset_password(
        cls,
        value: SecretStr,
    ) -> SecretStr:
        return RegisterRequest.validate_password(value)


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class MessageResponse(BaseModel):
    message: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int


class UserResponse(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    role: str
    district_id: int | None = None
    is_active: bool
    is_verified: bool
    created_at: datetime | None = None


class RegistrationResponse(BaseModel):
    message: str
    user: UserResponse


class SessionResponse(BaseModel):
    id: str
    created_at: datetime
    expires_at: datetime
    last_used_at: datetime | None = None
    is_current: bool = False
    device_name: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def user_id_of(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return str(identifier)


def serialize_user(user: Any) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        full_name=str(user.full_name),
        email=str(user.email),
        role=enum_value(user.role),
        district_id=getattr(
            user,
            "district_id",
            None,
        ),
        is_active=bool(user.is_active),
        is_verified=bool(
            getattr(user, "is_verified", False)
        ),
        created_at=getattr(
            user,
            "created_at",
            None,
        ),
    )


def client_ip(request: Request) -> str | None:
    """
    Use the direct peer address.

    Forwarded headers must only be interpreted by a
    correctly configured trusted-proxy middleware.
    """
    if request.client is None:
        return None

    return request.client.host


def user_agent(request: Request) -> str | None:
    value = request.headers.get("user-agent")

    if value is None:
        return None

    return value[:500]


def get_auth_service(
    db: DbSession,
) -> AuthService:
    return AuthService(db)


AuthServiceDependency = Annotated[
    AuthService,
    Depends(get_auth_service),
]


def token_response(
    result: Any,
) -> TokenResponse:
    """
    AuthService must return a validated token result
    containing these fields.
    """
    return TokenResponse(
        access_token=result.access_token,
        refresh_token=result.refresh_token,
        expires_in=result.expires_in,
        refresh_expires_in=result.refresh_expires_in,
    )


# ============================================================
# REGISTER
# ============================================================

@router.post(
    "/register",
    response_model=RegistrationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    request: Request,
    service: AuthServiceDependency,
) -> RegistrationResponse:
    if not payload.accept_terms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Terms acceptance required",
        )

    if not payload.accept_privacy_policy:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Privacy policy acceptance required",
        )

    # The service must enforce uniqueness, password hashing,
    # email verification and database transaction handling.
    #
    # Public registration creates only citizen accounts
    # or unverified volunteer applicants.
    # Volunteer operational permissions require separate
    # human verification and authorization.

    user = await service.register_user(
        full_name=payload.full_name,
        email=str(payload.email).lower(),
        password=payload.password.get_secret_value(),
        account_type=payload.account_type.value,
        district_id=payload.district_id,
        preferred_language=payload.preferred_language,
        ip_address=client_ip(request),
    )

    return RegistrationResponse(
        message=(
            "Registration successful. "
            "Complete account verification before "
            "accessing restricted services."
        ),
        user=serialize_user(user),
    )


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    service: AuthServiceDependency,
) -> TokenResponse:
    """
    Authenticate using email and password.

    AuthService must enforce:
    - Distributed rate limiting
    - Account lockout
    - Constant-time password verification
    - Disabled account checks
    - Refresh session creation
    - Audit logging
    """

    try:
        result = await service.login(
            email=str(payload.email).lower(),
            password=payload.password.get_secret_value(),
            remember_me=payload.remember_me,
            ip_address=client_ip(request),
            user_agent=user_agent(request),
        )

    except service.InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=GENERIC_AUTH_ERROR,
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from None

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    return token_response(result)


# ============================================================
# OAUTH2 PASSWORD TOKEN
# ============================================================

@router.post(
    "/token",
    response_model=TokenResponse,
)
async def oauth2_token(
    form: Annotated[
        OAuth2PasswordRequestForm,
        Depends(),
    ],
    request: Request,
    response: Response,
    service: AuthServiceDependency,
) -> TokenResponse:
    """
    OAuth2 password form endpoint for Swagger UI.

    Username is interpreted as the account email.
    """

    try:
        result = await service.login(
            email=form.username.lower(),
            password=form.password,
            remember_me=False,
            ip_address=client_ip(request),
            user_agent=user_agent(request),
        )

    except service.InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=GENERIC_AUTH_ERROR,
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from None

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    return token_response(result)


# ============================================================
# REFRESH TOKEN
# ============================================================

@router.post(
    "/refresh",
    response_model=TokenResponse,
)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    request: Request,
    response: Response,
    service: AuthServiceDependency,
) -> TokenResponse:
    """
    Rotate a refresh token.

    The service must atomically:
    1. Validate the stored token hash.
    2. Check expiry and revocation.
    3. Revoke the previous refresh token.
    4. Create a replacement session token.
    5. Detect refresh-token reuse.
    """

    try:
        result = await service.refresh_tokens(
            refresh_token=(
                payload.refresh_token.get_secret_value()
            ),
            ip_address=client_ip(request),
            user_agent=user_agent(request),
        )

    except service.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        ) from None

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    return token_response(result)


# ============================================================
# LOGOUT
# ============================================================

@router.post(
    "/logout",
    response_model=MessageResponse,
)
async def logout(
    payload: LogoutRequest,
    current_user: CurrentUser,
    service: AuthServiceDependency,
) -> MessageResponse:
    await service.logout(
        user_id=user_id_of(current_user),
        refresh_token=(
            payload.refresh_token.get_secret_value()
        ),
    )

    return MessageResponse(
        message="Session revoked successfully",
    )


@router.post(
    "/logout-all",
    response_model=MessageResponse,
)
async def logout_all(
    current_user: CurrentUser,
    service: AuthServiceDependency,
) -> MessageResponse:
    await service.revoke_all_sessions(
        user_id=user_id_of(current_user),
    )

    return MessageResponse(
        message="All refresh sessions revoked",
    )


# ============================================================
# CURRENT USER
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
)
async def get_my_profile(
    current_user: CurrentUser,
) -> UserResponse:
    return serialize_user(current_user)


# ============================================================
# CHANGE PASSWORD
# ============================================================

@router.post(
    "/change-password",
    response_model=MessageResponse,
)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: CurrentUser,
    service: AuthServiceDependency,
) -> MessageResponse:
    current_password = (
        payload.current_password.get_secret_value()
    )

    new_password = (
        payload.new_password.get_secret_value()
    )

    if current_password == new_password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="New password must be different",
        )

    try:
        await service.change_password(
            user_id=user_id_of(current_user),
            current_password=current_password,
            new_password=new_password,
            revoke_other_sessions=True,
        )

    except service.InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        ) from None

    return MessageResponse(
        message="Password changed successfully",
    )


# ============================================================
# FORGOT PASSWORD
# ============================================================

@router.post(
    "/forgot-password",
    response_model=MessageResponse,
)
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    service: AuthServiceDependency,
) -> MessageResponse:
    """
    Do not reveal whether an email address exists.

    AuthService must generate a cryptographically secure,
    short-lived, single-use reset token, store only its hash,
    and deliver it through an approved notification channel.
    """

    await service.request_password_reset(
        email=str(payload.email).lower(),
        ip_address=client_ip(request),
    )

    return MessageResponse(
        message=GENERIC_RESET_MESSAGE,
    )


# ============================================================
# RESET PASSWORD
# ============================================================

@router.post(
    "/reset-password",
    response_model=MessageResponse,
)
async def reset_password(
    payload: ResetPasswordRequest,
    service: AuthServiceDependency,
) -> MessageResponse:
    try:
        await service.reset_password(
            reset_token=(
                payload.reset_token.get_secret_value()
            ),
            new_password=(
                payload.new_password.get_secret_value()
            ),
            revoke_all_sessions=True,
        )

    except service.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        ) from None

    return MessageResponse(
        message=(
            "Password reset successful. "
            "Sign in using your new password."
        ),
    )


# ============================================================
# SESSION MANAGEMENT
# ============================================================

@router.get(
    "/sessions",
    response_model=SessionListResponse,
)
async def list_sessions(
    current_user: CurrentUser,
    service: AuthServiceDependency,
) -> SessionListResponse:
    sessions = await service.list_sessions(
        user_id=user_id_of(current_user),
    )

    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=str(item.id),
                created_at=item.created_at,
                expires_at=item.expires_at,
                last_used_at=item.last_used_at,
                is_current=getattr(
                    item,
                    "is_current",
                    False,
                ),
                device_name=getattr(
                    item,
                    "device_name",
                    None,
                ),
            )
            for item in sessions
        ],
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=MessageResponse,
)
async def revoke_session(
    session_id: str,
    current_user: CurrentUser,
    service: AuthServiceDependency,
) -> MessageResponse:
    try:
        validated_session_id = str(
            uuid.UUID(session_id)
        )

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid session identifier",
        ) from None

    await service.revoke_session(
        user_id=user_id_of(current_user),
        session_id=validated_session_id,
    )

    return MessageResponse(
        message="Session revoked",
    )


__all__ = [
    "router",
    "RegisterRequest",
    "LoginRequest",
    "RefreshTokenRequest",
    "LogoutRequest",
    "ChangePasswordRequest",
    "ForgotPasswordRequest",
    "ResetPasswordRequest",
    "TokenResponse",
    "UserResponse",
    "RegistrationResponse",
    "SessionResponse",
    "SessionListResponse",
]
