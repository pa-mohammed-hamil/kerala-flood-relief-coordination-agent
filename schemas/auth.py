from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/schemas/auth.py

Authentication and Authorization Schemas

Features:
- User registration
- Secure login
- JWT access and refresh tokens
- Password reset
- Email verification
- Role-based access control (RBAC)
- District-level permissions
- Emergency responder verification
- Multi-factor authentication
- Session management
- Account security
- Audit metadata

Pydantic v2 compatible.

NOTE:
These are validation schemas only.
Authentication, password hashing, JWT verification,
MFA, and authorization must be implemented in
backend/core/security.py and backend/api/auth.py.
"""


import re

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# BASE SCHEMA
# ============================================================

class AuthBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        str_strip_whitespace=True,
    )


# ============================================================
# ENUMS
# ============================================================

class UserRole(str, Enum):
    PUBLIC = "public"
    VOLUNTEER = "volunteer"
    RESCUE_TEAM = "rescue_team"
    MEDICAL_STAFF = "medical_staff"
    SHELTER_MANAGER = "shelter_manager"
    RELIEF_COORDINATOR = "relief_coordinator"
    DISTRICT_OFFICER = "district_officer"
    STATE_ADMIN = "state_admin"
    SYSTEM_ADMIN = "system_admin"
    AUDITOR = "auditor"


class UserStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISABLED = "disabled"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFICATION = "email_verification"
    MFA_CHALLENGE = "mfa_challenge"


class MFAType(str, Enum):
    TOTP = "totp"
    EMAIL_OTP = "email_otp"
    SMS_OTP = "sms_otp"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


class Permission(str, Enum):
    VIEW_PUBLIC_DASHBOARD = "view_public_dashboard"
    CREATE_SOS = "create_sos"
    VIEW_OWN_SOS = "view_own_sos"
    VIEW_DISTRICT_SOS = "view_district_sos"
    MANAGE_SOS = "manage_sos"
    VIEW_FLOOD_DATA = "view_flood_data"
    MANAGE_FLOOD_ALERTS = "manage_flood_alerts"
    APPROVE_PUBLIC_ALERTS = "approve_public_alerts"
    VIEW_RESCUE_OPERATIONS = "view_rescue_operations"
    MANAGE_RESCUE_TEAMS = "manage_rescue_teams"
    APPROVE_RESCUE_DISPATCH = "approve_rescue_dispatch"
    MANAGE_EVACUATION = "manage_evacuation"
    MANAGE_SHELTERS = "manage_shelters"
    MANAGE_RELIEF = "manage_relief"
    MANAGE_MEDICAL = "manage_medical"
    MANAGE_VOLUNTEERS = "manage_volunteers"
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_REPORTS = "export_reports"
    MANAGE_AGENTS = "manage_agents"
    APPROVE_AGENT_ACTIONS = "approve_agent_actions"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"


# ============================================================
# KERALA DISTRICTS
# ============================================================

KERALA_DISTRICTS = {
    "thiruvananthapuram",
    "kollam",
    "pathanamthitta",
    "alappuzha",
    "kottayam",
    "idukki",
    "ernakulam",
    "thrissur",
    "palakkad",
    "malappuram",
    "kozhikode",
    "wayanad",
    "kannur",
    "kasaragod",
}


def normalize_district(value: str) -> str:
    district = value.strip().lower()

    if district not in KERALA_DISTRICTS:
        raise ValueError(
            f"Invalid Kerala district: {value}"
        )

    return district


# ============================================================
# PASSWORD VALIDATION
# ============================================================

def validate_password_strength(
    password: str,
) -> str:
    """
    Basic password validation.

    Production authentication should also enforce
    breached-password checks, rate limiting,
    secure hashing and MFA where appropriate.
    """

    if len(password) < 12:
        raise ValueError(
            "Password must contain at least 12 characters"
        )

    if len(password) > 128:
        raise ValueError(
            "Password must not exceed 128 characters"
        )

    if not re.search(r"[A-Z]", password):
        raise ValueError(
            "Password must contain an uppercase letter"
        )

    if not re.search(r"[a-z]", password):
        raise ValueError(
            "Password must contain a lowercase letter"
        )

    if not re.search(r"\d", password):
        raise ValueError(
            "Password must contain a number"
        )

    if not re.search(r"[^A-Za-z0-9]", password):
        raise ValueError(
            "Password must contain a special character"
        )

    return password


# ============================================================
# PHONE VALIDATION
# ============================================================

def validate_indian_phone(
    phone: str,
) -> str:
    """
    Accept Indian mobile numbers in E.164 format.

    Example: +919876543210
    """

    normalized = phone.replace(" ", "").replace("-", "")

    if not re.fullmatch(
        r"\+91[6-9]\d{9}",
        normalized,
    ):
        raise ValueError(
            "Phone must be a valid Indian mobile number "
            "in +91XXXXXXXXXX format"
        )

    return normalized


# ============================================================
# USER REGISTRATION
# ============================================================

class UserRegistration(AuthBase):
    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    email: EmailStr

    phone: str | None = None

    password: SecretStr

    confirm_password: SecretStr

    district: str | None = None

    preferred_language: str = "en"

    accept_terms: bool

    @field_validator("phone")
    @classmethod
    def validate_phone(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return validate_indian_phone(value)

    @field_validator("district")
    @classmethod
    def validate_district(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return normalize_district(value)

    @field_validator("preferred_language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in {"en", "ml"}:
            raise ValueError(
                "Supported languages: en, ml"
            )
        return value

    @model_validator(mode="after")
    def validate_registration(self):
        password = self.password.get_secret_value()

        confirmation = (
            self.confirm_password.get_secret_value()
        )

        validate_password_strength(password)

        if password != confirmation:
            raise ValueError(
                "Passwords do not match"
            )

        if not self.accept_terms:
            raise ValueError(
                "Terms must be accepted"
            )

        return self


# ============================================================
# LOGIN
# ============================================================

class LoginRequest(AuthBase):
    email: EmailStr
    password: SecretStr

    remember_me: bool = False


class LoginResponse(AuthBase):
    access_token: str
    refresh_token: str | None = None

    token_type: str = "bearer"

    expires_in: int = Field(gt=0)

    user_id: UUID

    role: UserRole

    mfa_required: bool = False


# ============================================================
# TOKEN SCHEMAS
# ============================================================

class TokenPayload(AuthBase):
    sub: str = Field(min_length=1)

    jti: str = Field(min_length=1)

    token_type: TokenType

    iat: int = Field(ge=0)
    exp: int = Field(ge=0)

    iss: str | None = None
    aud: str | None = None

    @model_validator(mode="after")
    def validate_expiration(self):
        if self.exp <= self.iat:
            raise ValueError(
                "Token expiration must follow issue time"
            )
        return self


class RefreshTokenRequest(AuthBase):
    refresh_token: str = Field(min_length=1)


class RefreshTokenResponse(AuthBase):
    access_token: str
    refresh_token: str | None = None

    token_type: str = "bearer"

    expires_in: int = Field(gt=0)


class TokenRevocationRequest(AuthBase):
    refresh_token: str = Field(min_length=1)


# ============================================================
# USER PROFILE
# ============================================================

class UserProfile(AuthBase):
    user_id: UUID

    full_name: str
    email: EmailStr

    phone: str | None = None

    role: UserRole = UserRole.PUBLIC

    status: UserStatus = UserStatus.PENDING

    district: str | None = None

    preferred_language: str = "en"

    email_verified: bool = False
    phone_verified: bool = False

    mfa_enabled: bool = False

    created_at: datetime
    updated_at: datetime

    last_login_at: datetime | None = None


class UserProfileUpdate(AuthBase):
    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    phone: str | None = None

    district: str | None = None

    preferred_language: str | None = None

    @field_validator("phone")
    @classmethod
    def validate_phone(
        cls,
        value: str | None,
    ) -> str | None:
        return (
            validate_indian_phone(value)
            if value is not None
            else None
        )

    @field_validator("district")
    @classmethod
    def validate_district(
        cls,
        value: str | None,
    ) -> str | None:
        return (
            normalize_district(value)
            if value is not None
            else None
        )


# ============================================================
# CURRENT USER CONTEXT
# ============================================================

class CurrentUser(AuthBase):
    user_id: UUID
    email: EmailStr

    role: UserRole
    status: UserStatus

    district: str | None = None

    permissions: set[Permission] = Field(
        default_factory=set
    )

    mfa_verified: bool = False


# ============================================================
# PASSWORD MANAGEMENT
# ============================================================

class ChangePasswordRequest(AuthBase):
    current_password: SecretStr
    new_password: SecretStr
    confirm_new_password: SecretStr

    @model_validator(mode="after")
    def validate_change(self):
        current = self.current_password.get_secret_value()
        new = self.new_password.get_secret_value()
        confirm = (
            self.confirm_new_password.get_secret_value()
        )

        validate_password_strength(new)

        if current == new:
            raise ValueError(
                "New password must differ from current password"
            )

        if new != confirm:
            raise ValueError(
                "New passwords do not match"
            )

        return self


class ForgotPasswordRequest(AuthBase):
    email: EmailStr


class PasswordResetRequest(AuthBase):
    reset_token: str = Field(min_length=1)

    new_password: SecretStr
    confirm_password: SecretStr

    @model_validator(mode="after")
    def validate_reset(self):
        new = self.new_password.get_secret_value()
        confirm = self.confirm_password.get_secret_value()

        validate_password_strength(new)

        if new != confirm:
            raise ValueError(
                "Passwords do not match"
            )

        return self


# ============================================================
# EMAIL VERIFICATION
# ============================================================

class EmailVerificationRequest(AuthBase):
    verification_token: str = Field(
        min_length=1
    )


class ResendVerificationRequest(AuthBase):
    email: EmailStr


class VerificationResponse(AuthBase):
    success: bool
    verification_status: VerificationStatus
    message: str


# ============================================================
# MULTI-FACTOR AUTHENTICATION
# ============================================================

class MFASetupRequest(AuthBase):
    mfa_type: MFAType = MFAType.TOTP


class MFASetupResponse(AuthBase):
    mfa_type: MFAType

    # Enrollment secret is sensitive.
    # Return only from a protected setup endpoint.
    secret: SecretStr | None = None

    provisioning_uri: SecretStr | None = None

    expires_at: datetime | None = None


class MFAVerifyRequest(AuthBase):
    challenge_id: UUID | None = None

    code: str = Field(
        min_length=6,
        max_length=10,
    )


class MFAChallengeResponse(AuthBase):
    challenge_id: UUID = Field(
        default_factory=uuid4
    )

    mfa_type: MFAType

    expires_at: datetime

    message: str


class MFADisableRequest(AuthBase):
    password: SecretStr
    verification_code: str = Field(
        min_length=6,
        max_length=10,
    )


# ============================================================
# SESSION MANAGEMENT
# ============================================================

class UserSession(AuthBase):
    session_id: UUID = Field(
        default_factory=uuid4
    )

    user_id: UUID

    status: SessionStatus = SessionStatus.ACTIVE

    device_name: str | None = None

    ip_address: str | None = None

    user_agent: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    expires_at: datetime

    last_activity_at: datetime | None = None

    revoked_at: datetime | None = None


class LogoutRequest(AuthBase):
    all_devices: bool = False


class RevokeSessionRequest(AuthBase):
    session_id: UUID


# ============================================================
# ROLE AND PERMISSION MANAGEMENT
# ============================================================

class RoleAssignmentRequest(AuthBase):
    target_user_id: UUID

    role: UserRole

    district: str | None = None

    reason: str = Field(
        min_length=5,
        max_length=1000,
    )

    @field_validator("district")
    @classmethod
    def validate_district(
        cls,
        value: str | None,
    ) -> str | None:
        return (
            normalize_district(value)
            if value is not None
            else None
        )


class PermissionCheckRequest(AuthBase):
    permission: Permission

    district: str | None = None

    resource_id: str | None = None


class PermissionCheckResponse(AuthBase):
    allowed: bool

    permission: Permission

    reason: str | None = None


# ============================================================
# RESPONDER VERIFICATION
# ============================================================

class ResponderVerificationRequest(AuthBase):
    user_id: UUID

    organization_name: str = Field(
        min_length=2,
        max_length=200,
    )

    organization_id: str | None = None

    requested_role: UserRole

    district: str

    credential_reference: str | None = None

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        return normalize_district(value)

    @field_validator("requested_role")
    @classmethod
    def validate_responder_role(
        cls,
        value: UserRole,
    ) -> UserRole:

        permitted_roles = {
            UserRole.VOLUNTEER,
            UserRole.RESCUE_TEAM,
            UserRole.MEDICAL_STAFF,
            UserRole.SHELTER_MANAGER,
            UserRole.RELIEF_COORDINATOR,
            UserRole.DISTRICT_OFFICER,
        }

        if value not in permitted_roles:
            raise ValueError(
                "Requested role is not eligible for "
                "responder verification"
            )

        return value


class ResponderVerificationResponse(AuthBase):
    verification_id: UUID = Field(
        default_factory=uuid4
    )

    user_id: UUID

    status: VerificationStatus = (
        VerificationStatus.PENDING
    )

    reviewed_by: UUID | None = None

    reviewed_at: datetime | None = None

    notes: str | None = None


# ============================================================
# ACCOUNT SECURITY
# ============================================================

class AccountSecurityStatus(AuthBase):
    user_id: UUID

    email_verified: bool
    phone_verified: bool

    mfa_enabled: bool

    active_sessions: int = Field(ge=0)

    failed_login_attempts: int = Field(ge=0)

    last_password_change_at: datetime | None = None

    account_locked_until: datetime | None = None


# ============================================================
# AUTHENTICATION AUDIT
# ============================================================

class AuthenticationAuditEvent(AuthBase):
    event_id: UUID = Field(
        default_factory=uuid4
    )

    user_id: UUID | None = None

    event_type: str

    success: bool

    ip_address: str | None = None
    user_agent: str | None = None

    reason: str | None = None

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# API RESPONSE
# ============================================================

class AuthAPIResponse(AuthBase):
    success: bool

    message: str

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "UserRole",
    "UserStatus",
    "VerificationStatus",
    "TokenType",
    "MFAType",
    "SessionStatus",
    "Permission",
    "UserRegistration",
    "LoginRequest",
    "LoginResponse",
    "TokenPayload",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "TokenRevocationRequest",
    "UserProfile",
    "UserProfileUpdate",
    "CurrentUser",
    "ChangePasswordRequest",
    "ForgotPasswordRequest",
    "PasswordResetRequest",
    "EmailVerificationRequest",
    "ResendVerificationRequest",
    "VerificationResponse",
    "MFASetupRequest",
    "MFASetupResponse",
    "MFAChallengeResponse",
    "MFAVerifyRequest",
    "MFADisableRequest",
    "UserSession",
    "LogoutRequest",
    "RevokeSessionRequest",
    "RoleAssignmentRequest",
    "PermissionCheckRequest",
    "PermissionCheckResponse",
    "ResponderVerificationRequest",
    "ResponderVerificationResponse",
    "AccountSecurityStatus",
    "AuthenticationAuditEvent",
    "AuthAPIResponse",
    "validate_password_strength",
    "validate_indian_phone",
    "normalize_district",
]
