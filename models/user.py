from __future__ import annotations

"""
User and access management models.

Project: Kerala Flood Relief Coordination Agent
Python: 3.11+
ORM: SQLAlchemy 2.x

Includes:
- User accounts
- Role-based access control (RBAC)
- Permissions
- User-role assignments
- Role-permission assignments
- User profiles
- Organization membership
- Authentication sessions
- API keys
- MFA credentials
- Password reset tokens
- Email/phone verification
- Login attempts
- User activity history
- Account security and audit tracking

Security requirements:
- Never store plaintext passwords or API keys.
- Hash passwords using Argon2id or an approved equivalent.
- Store only hashed session and recovery tokens.
- Encrypt sensitive contact and identity references.
- Enforce permissions in the service/API layer.
- AI agents cannot grant themselves permissions.
"""


import enum
import uuid

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from backend.database.base import Base


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


def enum_type(
    enum_class: type[enum.Enum],
) -> SQLEnum:
    return SQLEnum(
        enum_class,
        values_callable=lambda members: [
            member.value for member in members
        ],
        native_enum=False,
        validate_strings=True,
        length=80,
    )


def required_text(
    value: str,
    field_name: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} is required"
        )

    value = value.strip()

    if len(value) > maximum:
        raise ValueError(
            f"{field_name} exceeds {maximum} characters"
        )

    return value


def normalize_email(value: str) -> str:
    value = required_text(
        value,
        "email",
        320,
    ).lower()

    if (
        value.count("@") != 1
        or not value.split("@")[0]
        or "." not in value.split("@")[1]
        or any(character.isspace() for character in value)
    ):
        raise ValueError("Invalid email address")

    return value


def require_user_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise ValueError(
            "Invalid user UUID"
        ) from None


# ============================================================
# ENUMS
# ============================================================

class UserStatus(str, enum.Enum):
    PENDING_VERIFICATION = "pending_verification"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    DISABLED = "disabled"
    DEACTIVATED = "deactivated"


class UserType(str, enum.Enum):
    CITIZEN = "citizen"
    VOLUNTEER = "volunteer"
    RESCUE_WORKER = "rescue_worker"
    MEDICAL_WORKER = "medical_worker"
    SHELTER_MANAGER = "shelter_manager"
    DISTRICT_OFFICER = "district_officer"
    STATE_OFFICER = "state_officer"
    NGO_STAFF = "ngo_staff"
    DISPATCHER = "dispatcher"
    ANALYST = "analyst"
    ADMINISTRATOR = "administrator"
    SYSTEM_SERVICE = "system_service"


class RoleScope(str, enum.Enum):
    GLOBAL = "global"
    STATE = "state"
    DISTRICT = "district"
    ORGANIZATION = "organization"
    SHELTER = "shelter"
    TEAM = "team"


class OrganizationMemberStatus(str, enum.Enum):
    INVITED = "invited"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class AuthProvider(str, enum.Enum):
    LOCAL = "local"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GOVERNMENT_SSO = "government_sso"
    OIDC = "oidc"
    SAML = "saml"


class MFAMethod(str, enum.Enum):
    TOTP = "totp"
    WEBAUTHN = "webauthn"
    HARDWARE_KEY = "hardware_key"
    RECOVERY_CODE = "recovery_code"


class VerificationType(str, enum.Enum):
    EMAIL = "email"
    PHONE = "phone"
    ACCOUNT_RECOVERY = "account_recovery"


class VerificationStatus(str, enum.Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    EXPIRED = "expired"
    REVOKED = "revoked"


class LoginAttemptResult(str, enum.Enum):
    SUCCESS = "success"
    INVALID_CREDENTIALS = "invalid_credentials"
    MFA_REQUIRED = "mfa_required"
    MFA_FAILED = "mfa_failed"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_DISABLED = "account_disabled"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"


class UserActivityType(str, enum.Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    PASSWORD_CHANGED = "password_changed"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    PROFILE_UPDATED = "profile_updated"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    ACCOUNT_ACTIVATED = "account_activated"
    ACCOUNT_SUSPENDED = "account_suspended"
    ACCOUNT_UNLOCKED = "account_unlocked"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    SESSION_REVOKED = "session_revoked"
    PERMISSION_DENIED = "permission_denied"
    SECURITY_ALERT = "security_alert"
    OTHER = "other"


# ============================================================
# USER
# ============================================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    email: Mapped[str] = mapped_column(
        String(320),
        unique=True,
        nullable=False,
        index=True,
    )

    password_hash: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    auth_provider: Mapped[AuthProvider] = mapped_column(
        enum_type(AuthProvider),
        default=AuthProvider.LOCAL,
        nullable=False,
    )

    external_subject_id: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    user_type: Mapped[UserType] = mapped_column(
        enum_type(UserType),
        default=UserType.CITIZEN,
        nullable=False,
    )

    status: Mapped[UserStatus] = mapped_column(
        enum_type(UserStatus),
        default=UserStatus.PENDING_VERIFICATION,
        nullable=False,
        index=True,
    )

    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    district_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(
            "districts.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    is_email_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_phone_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_mfa_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    must_change_password: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    password_changed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_active_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deactivated_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    profile: Mapped["UserProfile | None"] = relationship(
        "UserProfile",
        back_populates="user",
        uselist=False,
    )

    role_assignments: Mapped[
        list["UserRole"]
    ] = relationship(
        "UserRole",
        back_populates="user",
        foreign_keys="UserRole.user_id",
    )

    organization_memberships: Mapped[
        list["UserOrganizationMembership"]
    ] = relationship(
        "UserOrganizationMembership",
        back_populates="user",
    )

    sessions: Mapped[
        list["UserSession"]
    ] = relationship(
        "UserSession",
        back_populates="user",
    )

    api_keys: Mapped[
        list["UserAPIKey"]
    ] = relationship(
        "UserAPIKey",
        back_populates="user",
    )

    mfa_credentials: Mapped[
        list["UserMFACredential"]
    ] = relationship(
        "UserMFACredential",
        back_populates="user",
    )

    password_resets: Mapped[
        list["UserPasswordReset"]
    ] = relationship(
        "UserPasswordReset",
        back_populates="user",
    )

    verifications: Mapped[
        list["UserVerification"]
    ] = relationship(
        "UserVerification",
        back_populates="user",
    )

    login_attempts: Mapped[
        list["UserLoginAttempt"]
    ] = relationship(
        "UserLoginAttempt",
        back_populates="user",
    )

    activities: Mapped[
        list["UserActivity"]
    ] = relationship(
        "UserActivity",
        back_populates="user",
    )

    __table_args__ = (
        CheckConstraint(
            "failed_login_attempts >= 0",
            name="ck_user_failed_login_attempts",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_user_version",
        ),
        Index(
            "ix_user_district_status",
            "district_id",
            "status",
        ),
        Index(
            "ix_user_type_status",
            "user_type",
            "status",
        ),
        Index(
            "ix_user_auth_external_subject",
            "auth_provider",
            "external_subject_id",
        ),
    )

    __mapper_args__ = {
        "version_id_col": version,
    }

    @validates("username", "display_name")
    def validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        return required_text(
            value,
            key,
            100 if key == "username" else 255,
        )

    @validates("email")
    def validate_email(
        self,
        key: str,
        value: str,
    ) -> str:
        return normalize_email(value)

    @property
    def is_active(self) -> bool:
        return self.status == UserStatus.ACTIVE

    @property
    def is_locked(self) -> bool:
        if self.status == UserStatus.LOCKED:
            return True

        if self.locked_until is None:
            return False

        locked_until = self.locked_until

        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(
                tzinfo=timezone.utc
            )

        return locked_until > utc_now()

    @property
    def can_authenticate(self) -> bool:
        return self.is_active and not self.is_locked

    def record_successful_login(self) -> None:
        if not self.can_authenticate:
            raise ValueError(
                "User is not permitted to authenticate"
            )

        now = utc_now()

        self.last_login_at = now
        self.last_active_at = now
        self.failed_login_attempts = 0
        self.locked_until = None

    def record_failed_login(
        self,
        *,
        max_attempts: int = 5,
        lock_until: datetime | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be positive"
            )

        self.failed_login_attempts += 1

        if self.failed_login_attempts >= max_attempts:
            if lock_until is None:
                raise ValueError(
                    "A lock expiration is required"
                )

            self.locked_until = lock_until

    def activate(self) -> None:
        if not self.is_email_verified:
            raise ValueError(
                "Email verification required"
            )

        if self.status not in {
            UserStatus.PENDING_VERIFICATION,
            UserStatus.INACTIVE,
        }:
            raise ValueError(
                "User cannot be activated"
            )

        self.status = UserStatus.ACTIVE
        self.activated_at = utc_now()

    def suspend(self) -> None:
        if self.status != UserStatus.ACTIVE:
            raise ValueError(
                "Only active users can be suspended"
            )

        self.status = UserStatus.SUSPENDED

    def deactivate(self) -> None:
        self.status = UserStatus.DEACTIVATED
        self.deactivated_at = utc_now()

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "user_type": self.user_type.value,
            "status": self.status.value,
            "district_id": self.district_id,
            "is_active": self.is_active,
            "is_mfa_enabled": self.is_mfa_enabled,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }


# ============================================================
# USER PROFILE
# ============================================================

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        unique=True,
        nullable=False,
    )

    encrypted_phone_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    encrypted_address_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    encrypted_emergency_contact_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    encrypted_identity_document_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    preferred_language: Mapped[str] = mapped_column(
        String(20),
        default="ml",
        nullable=False,
    )

    timezone_name: Mapped[str] = mapped_column(
        String(100),
        default="Asia/Kolkata",
        nullable=False,
    )

    accessibility_preferences_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    notification_preferences: Mapped[dict] = (
        mapped_column(
            MutableDict.as_mutable(JSON),
            default=dict,
            nullable=False,
        )
    )

    consent_for_notifications: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    privacy_notice_acknowledged: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="profile",
    )


# ============================================================
# PERMISSION
# ============================================================

class Permission(Base):
    __tablename__ = "permissions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    code: Mapped[str] = mapped_column(
        String(150),
        unique=True,
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
    )

    resource: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    is_sensitive: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    requires_human_approval: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    role_permissions: Mapped[
        list["RolePermission"]
    ] = relationship(
        "RolePermission",
        back_populates="permission",
    )

    __table_args__ = (
        UniqueConstraint(
            "resource",
            "action",
            name="uq_permission_resource_action",
        ),
    )


# ============================================================
# ROLE
# ============================================================

class Role(Base):
    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
    )

    scope: Mapped[RoleScope] = mapped_column(
        enum_type(RoleScope),
        default=RoleScope.ORGANIZATION,
        nullable=False,
    )

    is_system_role: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user_roles: Mapped[
        list["UserRole"]
    ] = relationship(
        "UserRole",
        back_populates="role",
    )

    role_permissions: Mapped[
        list["RolePermission"]
    ] = relationship(
        "RolePermission",
        back_populates="role",
    )


# ============================================================
# ROLE PERMISSION
# ============================================================

class RolePermission(Base):
    __tablename__ = "role_permissions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    role_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "roles.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    permission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "permissions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    granted_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    role: Mapped["Role"] = relationship(
        "Role",
        back_populates="role_permissions",
    )

    permission: Mapped["Permission"] = relationship(
        "Permission",
        back_populates="role_permissions",
    )

    __table_args__ = (
        UniqueConstraint(
            "role_id",
            "permission_id",
            name="uq_role_permission",
        ),
    )


# ============================================================
# USER ROLE
# ============================================================

class UserRole(Base):
    __tablename__ = "user_roles"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    role_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "roles.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    scope_type: Mapped[RoleScope] = mapped_column(
        enum_type(RoleScope),
        nullable=False,
    )

    scope_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    assigned_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    revoked_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="role_assignments",
        foreign_keys=[user_id],
    )

    role: Mapped["Role"] = relationship(
        "Role",
        back_populates="user_roles",
    )

    __table_args__ = (
        Index(
            "ix_user_role_scope",
            "user_id",
            "scope_type",
            "scope_reference",
        ),
    )

    @property
    def is_effective(self) -> bool:
        if not self.is_active or self.revoked_at is not None:
            return False

        if self.expires_at is None:
            return True

        expiry = self.expires_at

        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        return expiry > utc_now()


# ============================================================
# ORGANIZATION MEMBERSHIP
# ============================================================

class UserOrganizationMembership(Base):
    __tablename__ = "user_organization_memberships"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )

    status: Mapped[
        OrganizationMemberStatus
    ] = mapped_column(
        enum_type(OrganizationMemberStatus),
        default=OrganizationMemberStatus.INVITED,
        nullable=False,
    )

    job_title: Mapped[str | None] = mapped_column(
        String(150),
    )

    is_primary_organization: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    invited_by_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    joined_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="organization_memberships",
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "organization_id",
            name="uq_user_organization_membership",
        ),
    )


# ============================================================
# AUTHENTICATION SESSION
# ============================================================

class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    refresh_token_hash: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    device_identifier_hash: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    ip_address_encrypted_ref: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(500),
    )

    is_mfa_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_revoked: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    revocation_reason: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="sessions",
    )

    def revoke(self, reason: str) -> None:
        self.is_revoked = True
        self.revoked_at = utc_now()
        self.revocation_reason = required_text(
            reason,
            "reason",
        )


# ============================================================
# USER API KEY
# ============================================================

class UserAPIKey(Base):
    __tablename__ = "user_api_keys"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    key_prefix: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    key_hash: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    scopes_json: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="api_keys",
    )

    def revoke(self) -> None:
        self.is_active = False
        self.revoked_at = utc_now()


# ============================================================
# MFA CREDENTIAL
# ============================================================

class UserMFACredential(Base):
    __tablename__ = "user_mfa_credentials"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    method: Mapped[MFAMethod] = mapped_column(
        enum_type(MFAMethod),
        nullable=False,
    )

    credential_identifier: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    encrypted_secret_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    public_key_reference: Mapped[
        str | None
    ] = mapped_column(
        String(500),
    )

    recovery_code_hash: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="mfa_credentials",
    )


# ============================================================
# PASSWORD RESET
# ============================================================

class UserPasswordReset(Base):
    __tablename__ = "user_password_resets"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    token_hash: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    is_used: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="password_resets",
    )

    def mark_used(self) -> None:
        if self.is_used:
            raise ValueError(
                "Password reset token already used"
            )

        expiry = self.expires_at

        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        if expiry <= utc_now():
            raise ValueError(
                "Password reset token expired"
            )

        self.is_used = True
        self.used_at = utc_now()


# ============================================================
# USER VERIFICATION
# ============================================================

class UserVerification(Base):
    __tablename__ = "user_verifications"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    verification_type: Mapped[
        VerificationType
    ] = mapped_column(
        enum_type(VerificationType),
        nullable=False,
    )

    status: Mapped[
        VerificationStatus
    ] = mapped_column(
        enum_type(VerificationStatus),
        default=VerificationStatus.PENDING,
        nullable=False,
    )

    token_hash: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="verifications",
    )

    __table_args__ = (
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_user_verification_attempts",
        ),
    )


# ============================================================
# LOGIN ATTEMPT
# ============================================================

class UserLoginAttempt(Base):
    __tablename__ = "user_login_attempts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    login_identifier_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    ip_address_hash: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(500),
    )

    result: Mapped[
        LoginAttemptResult
    ] = mapped_column(
        enum_type(LoginAttemptResult),
        nullable=False,
    )

    failure_reason_code: Mapped[
        str | None
    ] = mapped_column(
        String(100),
    )

    attempted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user: Mapped["User | None"] = relationship(
        "User",
        back_populates="login_attempts",
    )

    __table_args__ = (
        Index(
            "ix_login_attempt_identifier_time",
            "login_identifier_hash",
            "attempted_at",
        ),
    )


# ============================================================
# USER ACTIVITY
# ============================================================

class UserActivity(Base):
    __tablename__ = "user_activities"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    activity_type: Mapped[
        UserActivityType
    ] = mapped_column(
        enum_type(UserActivityType),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_user_id: Mapped[
        str | None
    ] = mapped_column(
        String(36),
    )

    ip_address_hash: Mapped[
        str | None
    ] = mapped_column(
        String(255),
    )

    audit_reference: Mapped[
        str | None
    ] = mapped_column(
        String(150),
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="activities",
    )

    __table_args__ = (
        Index(
            "ix_user_activity_user_time",
            "user_id",
            "occurred_at",
        ),
    )


# ============================================================
# DEFAULT ROLE DEFINITIONS
# ============================================================

DEFAULT_ROLE_CODES = (
    "super_admin",
    "state_admin",
    "district_admin",
    "incident_commander",
    "emergency_dispatcher",
    "rescue_team_leader",
    "rescue_worker",
    "medical_coordinator",
    "medical_worker",
    "shelter_manager",
    "shelter_staff",
    "logistics_coordinator",
    "inventory_manager",
    "volunteer_coordinator",
    "volunteer",
    "ngo_coordinator",
    "data_analyst",
    "auditor",
    "citizen",
    "read_only",
    "service_account",
)


# ============================================================
# DEFAULT PERMISSION DEFINITIONS
# ============================================================

DEFAULT_PERMISSION_CODES = (
    "users.read",
    "users.create",
    "users.update",
    "users.suspend",
    "users.assign_roles",
    "roles.read",
    "roles.manage",
    "permissions.read",
    "permissions.manage",
    "districts.read",
    "districts.manage",
    "flood_alerts.read",
    "flood_alerts.create",
    "flood_alerts.verify",
    "flood_alerts.publish",
    "sos.read",
    "sos.create",
    "sos.verify",
    "sos.triage",
    "sos.dispatch_approve",
    "sos.close",
    "incidents.read",
    "incidents.create",
    "incidents.manage",
    "rescue.read",
    "rescue.assign",
    "rescue.approve",
    "rescue.manage",
    "evacuation.read",
    "evacuation.plan",
    "evacuation.approve",
    "evacuation.manage",
    "shelters.read",
    "shelters.create",
    "shelters.verify",
    "shelters.activate",
    "shelters.manage",
    "shelters.close",
    "residents.read",
    "residents.register",
    "residents.manage",
    "residents.transfer",
    "residents.discharge",
    "inventory.read",
    "inventory.manage",
    "relief.read",
    "relief.request",
    "relief.approve",
    "relief.dispatch",
    "healthcare.read",
    "healthcare.manage",
    "volunteers.read",
    "volunteers.manage",
    "damage.read",
    "damage.assess",
    "damage.approve",
    "recovery.read",
    "recovery.manage",
    "analytics.read",
    "reports.generate",
    "agents.read",
    "agents.execute",
    "agents.approve_action",
    "models.read",
    "models.manage",
    "evaluations.read",
    "evaluations.run",
    "audit.read",
    "audit.export",
    "settings.read",
    "settings.manage",
)


# ============================================================
# MODEL REGISTRY
# ============================================================

USER_MODELS = (
    User,
    UserProfile,
    Permission,
    Role,
    RolePermission,
    UserRole,
    UserOrganizationMembership,
    UserSession,
    UserAPIKey,
    UserMFACredential,
    UserPasswordReset,
    UserVerification,
    UserLoginAttempt,
    UserActivity,
)


__all__ = [
    "User",
    "UserProfile",
    "Permission",
    "Role",
    "RolePermission",
    "UserRole",
    "UserOrganizationMembership",
    "UserSession",
    "UserAPIKey",
    "UserMFACredential",
    "UserPasswordReset",
    "UserVerification",
    "UserLoginAttempt",
    "UserActivity",
    "UserStatus",
    "UserType",
    "RoleScope",
    "OrganizationMemberStatus",
    "AuthProvider",
    "MFAMethod",
    "VerificationType",
    "VerificationStatus",
    "LoginAttemptResult",
    "UserActivityType",
    "DEFAULT_ROLE_CODES",
    "DEFAULT_PERMISSION_CODES",
    "USER_MODELS",
]
