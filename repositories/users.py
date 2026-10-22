from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/repositories/users.py

Async SQLAlchemy 2.x user repository.

Features:
- User account creation
- User retrieval by ID, email and phone
- Email and phone uniqueness checks
- User profile updates
- Password hash updates
- Email verification
- Phone verification
- Responder verification
- Role management
- Account activation and deactivation
- Failed login tracking
- Account lockout tracking
- Last login updates
- District-based user searches
- Pagination
- Soft deletion
- User statistics
- Transaction-safe database operations

Expected ORM model:
    backend.models.user.User

Expected User columns:
    id
    full_name
    email
    phone
    password_hash
    role
    status
    district
    is_active
    is_verified
    email_verified
    phone_verified
    failed_login_attempts
    locked_until
    last_login_at
    created_at
    updated_at
    is_deleted
    deleted_at
    deleted_by

Recommended additional columns:
    verified_by
    verified_at

IMPORTANT:
- Password hashing belongs in the authentication service.
- Never store plaintext passwords.
- Never expose password_hash in API responses.
- Never grant privileged roles through public registration.
- Never trust caller-provided authorization flags alone.
- Repository methods flush but do not commit.
- Use unique database constraints for email and phone.
"""


import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.user import User


logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
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

USER_ROLES = {
    "citizen",
    "volunteer",
    "rescue_operator",
    "rescue_coordinator",
    "medical_operator",
    "shelter_manager",
    "relief_coordinator",
    "district_officer",
    "state_officer",
    "admin",
    "super_admin",
}

PRIVILEGED_ROLES = {
    "rescue_operator",
    "rescue_coordinator",
    "medical_operator",
    "shelter_manager",
    "relief_coordinator",
    "district_officer",
    "state_officer",
    "admin",
    "super_admin",
}

USER_STATUSES = {
    "pending",
    "active",
    "suspended",
    "disabled",
}

MAX_FAILED_LOGIN_ATTEMPTS = 5

ACCOUNT_LOCK_MINUTES = 15


# ============================================================
# EXCEPTIONS
# ============================================================

class UserRepositoryError(Exception):
    """Base user repository exception."""


class UserNotFoundError(UserRepositoryError):
    """User account not found."""


class UserAlreadyExistsError(UserRepositoryError):
    """Email or phone is already registered."""


class UserAuthorizationError(UserRepositoryError):
    """Operation requires additional authorization."""


class UserStateError(UserRepositoryError):
    """Invalid user account state."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    normalized = email.strip().lower()

    if not normalized:
        raise ValueError("Email is required")

    if len(normalized) > 254:
        raise ValueError("Email is too long")

    if not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+",
        normalized,
    ):
        raise ValueError("Invalid email format")

    return normalized


def normalize_phone(phone: str) -> str:
    """
    Normalize Indian mobile numbers to +91XXXXXXXXXX.

    This is basic formatting validation, not
    proof of ownership or number reachability.
    """

    cleaned = re.sub(
        r"[\s()-]",
        "",
        phone.strip(),
    )

    if cleaned.startswith("0091"):
        cleaned = "+" + cleaned[2:]

    elif cleaned.startswith("91") and len(cleaned) == 12:
        cleaned = "+" + cleaned

    elif len(cleaned) == 10:
        cleaned = "+91" + cleaned

    if not re.fullmatch(
        r"\+91[6-9]\d{9}",
        cleaned,
    ):
        raise ValueError(
            "Invalid Indian mobile number"
        )

    return cleaned


def normalize_district(
    district: str | None,
) -> str | None:
    if district is None:
        return None

    normalized = district.strip().lower()

    if normalized not in KERALA_DISTRICTS:
        raise ValueError(
            f"Invalid Kerala district: {district}"
        )

    return normalized


def validate_pagination(
    limit: int,
    offset: int,
) -> None:
    if not 1 <= limit <= 500:
        raise ValueError(
            "limit must be between 1 and 500"
        )

    if offset < 0:
        raise ValueError(
            "offset cannot be negative"
        )


# ============================================================
# USER REPOSITORY
# ============================================================

class UserRepository:
    """
    Async repository for user persistence.

    Transaction ownership:
        - Repository methods flush changes.
        - Service layer commits or rolls back.
        - Each request receives its own AsyncSession.

    Example:

        async with transaction() as session:
            repository = UserRepository(session)

            user = await repository.create(
                full_name="Example User",
                email="user@example.com",
                phone="+919876543210",
                password_hash=hashed_password,
                district="thrissur",
            )

    Security:
        Authentication and authorization must be
        enforced by the service/API layer.
    """

    PROFILE_UPDATE_FIELDS = {
        "full_name",
        "phone",
        "district",
    }

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _not_deleted():
        return User.is_deleted.is_(False)

    async def _get_locked(
        self,
        user_id: UUID,
    ) -> User:

        statement = (
            select(User)
            .where(
                User.id == user_id,
                self._not_deleted(),
            )
            .with_for_update()
        )

        result = await self.session.execute(
            statement
        )

        user = result.scalar_one_or_none()

        if user is None:
            raise UserNotFoundError(
                "User account not found"
            )

        return user

    # ========================================================
    # CREATE USER
    # ========================================================

    async def create(
        self,
        *,
        full_name: str,
        email: str,
        phone: str,
        password_hash: str,
        district: str | None = None,
    ) -> User:
        """
        Register a public user account.

        Public registration always creates a citizen
        account. Privileged roles must be assigned
        through a separate authorized workflow.

        Password hash must be generated using an
        approved password hashing algorithm such as
        Argon2id before calling this method.
        """

        if not full_name.strip():
            raise ValueError(
                "Full name is required"
            )

        if not password_hash:
            raise ValueError(
                "Password hash is required"
            )

        email = normalize_email(email)
        phone = normalize_phone(phone)

        existing_email = await self.get_by_email(
            email
        )

        if existing_email is not None:
            raise UserAlreadyExistsError(
                "Email is already registered"
            )

        existing_phone = await self.get_by_phone(
            phone
        )

        if existing_phone is not None:
            raise UserAlreadyExistsError(
                "Phone is already registered"
            )

        user = User(
            full_name=full_name.strip(),
            email=email,
            phone=phone,
            password_hash=password_hash,
            role="citizen",
            status="pending",
            district=normalize_district(district),
            is_active=False,
            is_verified=False,
            email_verified=False,
            phone_verified=False,
            failed_login_attempts=0,
            is_deleted=False,
        )

        self.session.add(user)

        # A unique DB constraint is still required to
        # protect against concurrent registrations.
        await self.session.flush()

        return user

    # ========================================================
    # GET USER BY ID
    # ========================================================

    async def get_by_id(
        self,
        user_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> User | None:

        statement = select(User).where(
            User.id == user_id
        )

        if not include_deleted:
            statement = statement.where(
                self._not_deleted()
            )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # GET USER BY EMAIL
    # ========================================================

    async def get_by_email(
        self,
        email: str,
    ) -> User | None:

        email = normalize_email(email)

        statement = select(User).where(
            User.email == email,
            self._not_deleted(),
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # GET USER BY PHONE
    # ========================================================

    async def get_by_phone(
        self,
        phone: str,
    ) -> User | None:

        phone = normalize_phone(phone)

        statement = select(User).where(
            User.phone == phone,
            self._not_deleted(),
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # EMAIL EXISTS
    # ========================================================

    async def email_exists(
        self,
        email: str,
    ) -> bool:

        email = normalize_email(email)

        statement = select(
            User.id
        ).where(
            User.email == email,
            self._not_deleted(),
        ).limit(1)

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none() is not None

    # ========================================================
    # PHONE EXISTS
    # ========================================================

    async def phone_exists(
        self,
        phone: str,
    ) -> bool:

        phone = normalize_phone(phone)

        statement = select(
            User.id
        ).where(
            User.phone == phone,
            self._not_deleted(),
        ).limit(1)

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none() is not None

    # ========================================================
    # LIST USERS
    # ========================================================

    async def list_users(
        self,
        *,
        district: str | None = None,
        role: str | None = None,
        status: str | None = None,
        active_only: bool = False,
        verified_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[User]:

        validate_pagination(limit, offset)

        statement = select(User).where(
            self._not_deleted()
        )

        if district is not None:
            statement = statement.where(
                User.district
                == normalize_district(district)
            )

        if role is not None:
            statement = statement.where(
                User.role == role
            )

        if status is not None:
            statement = statement.where(
                User.status == status
            )

        if active_only:
            statement = statement.where(
                User.is_active.is_(True)
            )

        if verified_only:
            statement = statement.where(
                User.is_verified.is_(True)
            )

        statement = (
            statement
            .order_by(
                User.created_at.desc(),
                User.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(
            statement
        )

        return result.scalars().all()

    # ========================================================
    # COUNT USERS
    # ========================================================

    async def count(
        self,
        *,
        district: str | None = None,
        role: str | None = None,
        status: str | None = None,
    ) -> int:

        statement = select(
            func.count(User.id)
        ).where(
            self._not_deleted()
        )

        if district is not None:
            statement = statement.where(
                User.district
                == normalize_district(district)
            )

        if role is not None:
            statement = statement.where(
                User.role == role
            )

        if status is not None:
            statement = statement.where(
                User.status == status
            )

        result = await self.session.execute(
            statement
        )

        return int(result.scalar_one())

    # ========================================================
    # UPDATE USER PROFILE
    # ========================================================

    async def update_profile(
        self,
        *,
        user_id: UUID,
        changes: dict[str, Any],
    ) -> User:
        """
        Update allowed profile fields.

        Email, role, verification and password
        are handled by separate secure workflows.
        """

        invalid_fields = (
            set(changes)
            - self.PROFILE_UPDATE_FIELDS
        )

        if invalid_fields:
            raise ValueError(
                "Unsupported profile fields: "
                + ", ".join(sorted(invalid_fields))
            )

        user = await self._get_locked(
            user_id
        )

        if "full_name" in changes:
            full_name = changes["full_name"].strip()

            if not full_name:
                raise ValueError(
                    "Full name cannot be empty"
                )

            user.full_name = full_name

        if "phone" in changes:
            new_phone = normalize_phone(
                changes["phone"]
            )

            if new_phone != user.phone:
                existing = await self.get_by_phone(
                    new_phone
                )

                if existing is not None:
                    raise UserAlreadyExistsError(
                        "Phone already registered"
                    )

                user.phone = new_phone
                user.phone_verified = False

        if "district" in changes:
            user.district = normalize_district(
                changes["district"]
            )

        await self.session.flush()

        return user

    # ========================================================
    # UPDATE PASSWORD HASH
    # ========================================================

    async def update_password_hash(
        self,
        *,
        user_id: UUID,
        password_hash: str,
    ) -> User:
        """
        Replace a stored password hash.

        The service must verify password-reset
        authorization or the current password.

        The service must also revoke old sessions
        and refresh tokens as appropriate.
        """

        if not password_hash:
            raise ValueError(
                "Password hash cannot be empty"
            )

        user = await self._get_locked(
            user_id
        )

        user.password_hash = password_hash
        user.failed_login_attempts = 0
        user.locked_until = None

        await self.session.flush()

        return user

    # ========================================================
    # VERIFY EMAIL
    # ========================================================

    async def verify_email(
        self,
        *,
        user_id: UUID,
        verification_confirmed: bool,
    ) -> User:
        """
        Mark email verified after a valid token/OTP.

        Token validation belongs in the auth service.
        """

        if not verification_confirmed:
            raise UserAuthorizationError(
                "Email verification proof required"
            )

        user = await self._get_locked(
            user_id
        )

        user.email_verified = True

        await self.session.flush()

        return user

    # ========================================================
    # VERIFY PHONE
    # ========================================================

    async def verify_phone(
        self,
        *,
        user_id: UUID,
        verification_confirmed: bool,
    ) -> User:
        """
        Mark phone verified after OTP validation.
        """

        if not verification_confirmed:
            raise UserAuthorizationError(
                "Phone verification proof required"
            )

        user = await self._get_locked(
            user_id
        )

        user.phone_verified = True

        await self.session.flush()

        return user

    # ========================================================
    # ACTIVATE ACCOUNT
    # ========================================================

    async def activate(
        self,
        *,
        user_id: UUID,
        authorized_by: UUID,
    ) -> User:
        """
        Activate an account.

        Caller must verify operator authorization.
        """

        user = await self._get_locked(
            user_id
        )

        if user.status == "suspended":
            raise UserStateError(
                "Suspended accounts require a separate "
                "reinstatement review"
            )

        if not user.email_verified:
            raise UserStateError(
                "Email verification is required"
            )

        user.is_active = True
        user.status = "active"

        await self.session.flush()

        logger.info(
            "User %s activated by %s",
            user_id,
            authorized_by,
        )

        return user

    # ========================================================
    # DEACTIVATE ACCOUNT
    # ========================================================

    async def deactivate(
        self,
        *,
        user_id: UUID,
        authorized_by: UUID,
    ) -> User:
        """
        Disable user login access.

        Caller must revoke active sessions/tokens.
        """

        user = await self._get_locked(
            user_id
        )

        user.is_active = False
        user.status = "disabled"

        await self.session.flush()

        logger.info(
            "User %s deactivated by %s",
            user_id,
            authorized_by,
        )

        return user

    # ========================================================
    # SUSPEND ACCOUNT
    # ========================================================

    async def suspend(
        self,
        *,
        user_id: UUID,
        authorized_by: UUID,
        reason: str,
    ) -> User:

        if not reason.strip():
            raise ValueError(
                "Suspension reason required"
            )

        user = await self._get_locked(
            user_id
        )

        user.is_active = False
        user.status = "suspended"

        await self.session.flush()

        logger.warning(
            "User %s suspended by %s",
            user_id,
            authorized_by,
        )

        return user

    # ========================================================
    # UPDATE USER ROLE
    # ========================================================

    async def update_role(
        self,
        *,
        user_id: UUID,
        new_role: str,
        authorized_by: UUID,
        authorization_verified: bool,
    ) -> User:
        """
        Update user role after an authorized review.

        Never call this directly from a public
        registration or profile endpoint.
        """

        if new_role not in USER_ROLES:
            raise ValueError(
                "Invalid user role"
            )

        if not authorization_verified:
            raise UserAuthorizationError(
                "Role change authorization required"
            )

        user = await self._get_locked(
            user_id
        )

        user.role = new_role

        await self.session.flush()

        logger.warning(
            "User %s role changed by %s",
            user_id,
            authorized_by,
        )

        return user

    # ========================================================
    # VERIFY RESPONDER
    # ========================================================

    async def verify_responder(
        self,
        *,
        user_id: UUID,
        verified_by: UUID,
        verification_approved: bool,
    ) -> User:
        """
        Verify a responder or operational user.

        Identity, credentials, training and organization
        membership must be checked by the service layer.
        """

        if not verification_approved:
            raise UserAuthorizationError(
                "Responder verification approval required"
            )

        user = await self._get_locked(
            user_id
        )

        if user.role not in PRIVILEGED_ROLES:
            raise UserStateError(
                "User does not have a responder "
                "or operational role"
            )

        user.is_verified = True

        # These fields are optional in the ORM.
        if hasattr(user, "verified_by"):
            user.verified_by = verified_by

        if hasattr(user, "verified_at"):
            user.verified_at = utc_now()

        await self.session.flush()

        return user

    # ========================================================
    # RECORD SUCCESSFUL LOGIN
    # ========================================================

    async def record_successful_login(
        self,
        *,
        user_id: UUID,
    ) -> User:
        """
        Update login state after successful authentication.

        Password verification and MFA must be completed
        before this method is called.
        """

        user = await self._get_locked(
            user_id
        )

        if (
            not user.is_active
            or user.status != "active"
        ):
            raise UserStateError(
                "Account is not active"
            )

        if (
            user.locked_until is not None
            and user.locked_until > utc_now()
        ):
            raise UserStateError(
                "Account is temporarily locked"
            )

        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = utc_now()

        await self.session.flush()

        return user

    # ========================================================
    # RECORD FAILED LOGIN
    # ========================================================

    async def record_failed_login(
        self,
        *,
        user_id: UUID,
    ) -> User:
        """
        Record a failed login attempt.

        The authentication service should also apply
        IP/device-based throttling and generic error
        responses to prevent account enumeration.
        """

        user = await self._get_locked(
            user_id
        )

        user.failed_login_attempts += 1

        if (
            user.failed_login_attempts
            >= MAX_FAILED_LOGIN_ATTEMPTS
        ):
            user.locked_until = (
                utc_now()
                + timedelta(
                    minutes=ACCOUNT_LOCK_MINUTES
                )
            )

        await self.session.flush()

        return user

    # ========================================================
    # CHECK ACCOUNT LOCK
    # ========================================================

    async def is_account_locked(
        self,
        *,
        user_id: UUID,
    ) -> bool:

        user = await self.get_by_id(
            user_id
        )

        if user is None:
            raise UserNotFoundError(
                "User not found"
            )

        if user.locked_until is None:
            return False

        return user.locked_until > utc_now()

    # ========================================================
    # SOFT DELETE USER
    # ========================================================

    async def soft_delete(
        self,
        *,
        user_id: UUID,
        deleted_by: UUID,
    ) -> User:
        """
        Soft-delete a user account.

        Caller must enforce retention policy,
        authorization and session revocation.
        """

        user = await self._get_locked(
            user_id
        )

        user.is_deleted = True
        user.is_active = False
        user.status = "disabled"
        user.deleted_at = utc_now()
        user.deleted_by = deleted_by

        await self.session.flush()

        logger.warning(
            "User %s soft-deleted by %s",
            user_id,
            deleted_by,
        )

        return user

    # ========================================================
    # USER STATISTICS
    # ========================================================

    async def get_statistics(
        self,
        *,
        district: str | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate actual user records.

        Results should only be exposed to authorized
        administrative or analytics endpoints.
        """

        filters = [
            self._not_deleted()
        ]

        if district is not None:
            district = normalize_district(
                district
            )

            filters.append(
                User.district == district
            )

        role_statement = (
            select(
                User.role,
                func.count(User.id),
            )
            .where(*filters)
            .group_by(User.role)
        )

        status_statement = (
            select(
                User.status,
                func.count(User.id),
            )
            .where(*filters)
            .group_by(User.status)
        )

        role_result = await self.session.execute(
            role_statement
        )

        status_result = await self.session.execute(
            status_statement
        )

        roles = {
            str(role): int(count)
            for role, count in role_result.all()
        }

        statuses = {
            str(status): int(count)
            for status, count in status_result.all()
        }

        return {
            "district": district,
            "total_users": sum(
                roles.values()
            ),
            "active_users": statuses.get(
                "active",
                0,
            ),
            "pending_users": statuses.get(
                "pending",
                0,
            ),
            "suspended_users": statuses.get(
                "suspended",
                0,
            ),
            "by_role": roles,
            "by_status": statuses,
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "UserRepository",
    "UserRepositoryError",
    "UserNotFoundError",
    "UserAlreadyExistsError",
    "UserAuthorizationError",
    "UserStateError",
    "USER_ROLES",
    "PRIVILEGED_ROLES",
    "USER_STATUSES",
    "normalize_email",
    "normalize_phone",
    "normalize_district",
    "validate_pagination",
    "utc_now",
]
