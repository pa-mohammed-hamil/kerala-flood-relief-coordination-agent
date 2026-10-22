from __future__ import annotations

"""
backend/api/users.py

Kerala Flood Relief Coordination Agent
User Management API

Python 3.11+ | FastAPI | Pydantic v2

Routes:
    GET    /users/me
    PATCH  /users/me
    GET    /users/me/sessions
    DELETE /users/me/sessions/{session_id}

    GET    /users
    GET    /users/statistics
    POST   /users
    GET    /users/{user_id}
    PATCH  /users/{user_id}
    POST   /users/{user_id}/activate
    POST   /users/{user_id}/deactivate
    POST   /users/{user_id}/district
    GET    /users/{user_id}/roles
    POST   /users/{user_id}/roles
    DELETE /users/{user_id}/roles/{role_id}
    GET    /users/{user_id}/permissions
    GET    /users/{user_id}/sessions
    POST   /users/{user_id}/revoke-sessions
    GET    /users/{user_id}/activity

Security:
    - Authentication required
    - RBAC permissions
    - District-based access control
    - No password/hash disclosure
    - No arbitrary privilege escalation
    - Audit-sensitive operations
    - Service-enforced session revocation
"""


from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Protocol
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    model_validator,
)

from backend.dependencies import get_current_user


router = APIRouter(
    prefix="/users",
    tags=["User Management"],
)

CurrentUser = Annotated[
    Any,
    Depends(get_current_user),
]


# ============================================================
# PERMISSION CONSTANTS
# ============================================================

USER_READ = "users:read"
USER_CREATE = "users:create"
USER_UPDATE = "users:update"
USER_ACTIVATE = "users:activate"
USER_DEACTIVATE = "users:deactivate"

USER_ROLE_READ = "users:roles:read"
USER_ROLE_ASSIGN = "users:roles:assign"
USER_ROLE_REVOKE = "users:roles:revoke"

USER_DISTRICT_ASSIGN = "users:district:assign"

USER_SESSION_READ = "users:sessions:read"
USER_SESSION_REVOKE = "users:sessions:revoke"

USER_ACTIVITY_READ = "users:activity:read"

USER_STATISTICS_READ = "users:statistics:read"


# ============================================================
# ENUMERATIONS
# ============================================================

class UserStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"


class UserType(str, Enum):
    CITIZEN = "citizen"
    VOLUNTEER = "volunteer"
    RESCUE_PERSONNEL = "rescue_personnel"
    MEDICAL_PERSONNEL = "medical_personnel"
    GOVERNMENT_OFFICER = "government_officer"
    SHELTER_STAFF = "shelter_staff"
    SYSTEM_OPERATOR = "system_operator"


class UserSortBy(str, Enum):
    CREATED_AT = "created_at"
    EMAIL = "email"
    STATUS = "status"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class UserProfileUpdate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    phone_number: str | None = Field(
        default=None,
        min_length=7,
        max_length=20,
    )

    preferred_language: str | None = Field(
        default=None,
        max_length=20,
    )

    emergency_contact_reference: str | None = Field(
        default=None,
        max_length=150,
    )


class UserCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    email: EmailStr

    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    user_type: UserType

    district_id: int | None = Field(
        default=None,
        ge=1,
    )

    phone_number: str | None = Field(
        default=None,
        min_length=7,
        max_length=20,
    )

    invitation_required: bool = True


class UserAdminUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    phone_number: str | None = Field(
        default=None,
        min_length=7,
        max_length=20,
    )

    user_type: UserType | None = None


class UserDistrictAssignment(BaseModel):
    district_id: int = Field(ge=1)

    reason: str = Field(
        min_length=10,
        max_length=3000,
    )

    assignment_approved: bool = False


class UserRoleAssignment(BaseModel):
    role_id: UUID

    reason: str = Field(
        min_length=10,
        max_length=3000,
    )

    approval_reference: str = Field(
        min_length=5,
        max_length=150,
    )


class UserRoleRevocation(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=3000,
    )


class UserStatusChange(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=3000,
    )


class UserSessionRevocation(BaseModel):
    reason: str = Field(
        min_length=10,
        max_length=3000,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str | None = None
    user_type: str | None = None
    status: str | None = None
    district_id: int | None = None
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UserListResponse(BaseModel):
    items: list[UserResponse]
    total: int
    limit: int
    offset: int


class UserStatisticsResponse(BaseModel):
    total_users: int
    active_users: int
    inactive_users: int
    by_type: dict[str, int]
    generated_at: datetime


class UserRoleResponse(BaseModel):
    id: UUID
    name: str
    description: str | None = None


class UserPermissionResponse(BaseModel):
    id: UUID
    code: str
    description: str | None = None


class UserSessionResponse(BaseModel):
    id: UUID
    created_at: datetime
    last_seen_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None


class UserActivityResponse(BaseModel):
    id: UUID
    action: str
    created_at: datetime
    resource_type: str | None = None


class UserOperationResponse(BaseModel):
    success: bool
    message: str
    user_id: UUID


# ============================================================
# SERVICE CONTRACT
# ============================================================

class UserService(Protocol):
    """
    Implement in backend/services/user_service.py.

    The service must:
      - Enforce normalized RBAC permissions
      - Verify district/organization memberships
      - Avoid exposing sensitive profile fields
      - Protect the last active administrator
      - Prevent self-escalation
      - Validate role assignment approvals
      - Enforce account and session revocation
      - Record immutable audit events
      - Use transactions for administrative changes
    """

    async def get_user(
        self,
        user_id: str,
        actor_id: str,
    ) -> UserResponse | None: ...

    async def update_profile(
        self,
        user_id: str,
        payload: UserProfileUpdate,
        actor_id: str,
    ) -> UserResponse: ...

    async def list_users(
        self,
        actor_id: str,
        district_id: int | None,
        limit: int,
        offset: int,
        search: str | None,
        active_only: bool | None,
    ) -> UserListResponse: ...

    async def statistics(
        self,
        actor_id: str,
        district_id: int | None,
    ) -> UserStatisticsResponse: ...

    async def create_user(
        self,
        payload: UserCreate,
        actor_id: str,
    ) -> UserResponse: ...

    async def admin_update_user(
        self,
        user_id: str,
        payload: UserAdminUpdate,
        actor_id: str,
    ) -> UserResponse: ...

    async def activate_user(
        self,
        user_id: str,
        payload: UserStatusChange,
        actor_id: str,
    ) -> UserOperationResponse: ...

    async def deactivate_user(
        self,
        user_id: str,
        payload: UserStatusChange,
        actor_id: str,
    ) -> UserOperationResponse: ...

    async def assign_district(
        self,
        user_id: str,
        payload: UserDistrictAssignment,
        actor_id: str,
    ) -> UserResponse: ...

    async def list_roles(
        self,
        user_id: str,
        actor_id: str,
    ) -> list[UserRoleResponse]: ...

    async def assign_role(
        self,
        user_id: str,
        payload: UserRoleAssignment,
        actor_id: str,
    ) -> UserRoleResponse: ...

    async def revoke_role(
        self,
        user_id: str,
        role_id: str,
        payload: UserRoleRevocation,
        actor_id: str,
    ) -> UserOperationResponse: ...

    async def list_permissions(
        self,
        user_id: str,
        actor_id: str,
    ) -> list[UserPermissionResponse]: ...

    async def list_sessions(
        self,
        user_id: str,
        actor_id: str,
    ) -> list[UserSessionResponse]: ...

    async def revoke_session(
        self,
        user_id: str,
        session_id: str,
        actor_id: str,
    ) -> UserOperationResponse: ...

    async def revoke_all_sessions(
        self,
        user_id: str,
        reason: str,
        actor_id: str,
    ) -> UserOperationResponse: ...

    async def list_activity(
        self,
        user_id: str,
        actor_id: str,
        limit: int,
    ) -> list[UserActivityResponse]: ...


# ============================================================
# DEPENDENCIES
# ============================================================

def get_user_service(request: Request) -> UserService:
    service = getattr(
        request.app.state,
        "user_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="User service is not configured",
        )

    return service


UserServiceDependency = Annotated[
    UserService,
    Depends(get_user_service),
]


def get_permission_checker(request: Request):
    """
    Required application dependency:

        app.state.permission_checker

    Expected callable:

        await permission_checker(
            user=current_user,
            permission="users:read",
        )

    It must query verified roles and permissions.
    """
    checker = getattr(
        request.app.state,
        "permission_checker",
        None,
    )

    if checker is None:
        raise HTTPException(
            status_code=503,
            detail="Permission checker is not configured",
        )

    return checker


PermissionChecker = Annotated[
    Any,
    Depends(get_permission_checker),
]


# ============================================================
# SECURITY HELPERS
# ============================================================

def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return str(identifier)


async def require_permission(
    checker: Any,
    user: Any,
    permission: str,
) -> None:
    allowed = await checker(
        user=user,
        permission=permission,
    )

    if not allowed:
        raise HTTPException(
            status_code=403,
            detail="Permission denied",
        )


async def require_target_access(
    service: UserService,
    target_id: UUID,
    current_user: Any,
) -> UserResponse:
    target = await service.get_user(
        user_id=str(target_id),
        actor_id=actor_id(current_user),
    )

    if target is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return target


def reject_self_privilege_change(
    target_id: UUID,
    current_user: Any,
) -> None:
    if str(target_id) == actor_id(current_user):
        raise HTTPException(
            status_code=403,
            detail="Self-privilege changes are prohibited",
        )


# ============================================================
# CURRENT USER PROFILE
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
)
async def get_my_profile(
    current_user: CurrentUser,
    service: UserServiceDependency,
) -> UserResponse:
    user = await service.get_user(
        user_id=actor_id(current_user),
        actor_id=actor_id(current_user),
    )

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User profile not found",
        )

    return user


@router.patch(
    "/me",
    response_model=UserResponse,
)
async def update_my_profile(
    payload: UserProfileUpdate,
    current_user: CurrentUser,
    service: UserServiceDependency,
) -> UserResponse:
    return await service.update_profile(
        user_id=actor_id(current_user),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# CURRENT USER SESSIONS
# ============================================================

@router.get(
    "/me/sessions",
    response_model=list[UserSessionResponse],
)
async def get_my_sessions(
    current_user: CurrentUser,
    service: UserServiceDependency,
) -> list[UserSessionResponse]:
    identifier = actor_id(current_user)

    return await service.list_sessions(
        user_id=identifier,
        actor_id=identifier,
    )


@router.delete(
    "/me/sessions/{session_id}",
    response_model=UserOperationResponse,
)
async def revoke_my_session(
    session_id: UUID,
    current_user: CurrentUser,
    service: UserServiceDependency,
) -> UserOperationResponse:
    identifier = actor_id(current_user)

    return await service.revoke_session(
        user_id=identifier,
        session_id=str(session_id),
        actor_id=identifier,
    )


# ============================================================
# LIST USERS
# ============================================================

@router.get(
    "",
    response_model=UserListResponse,
)
async def list_users(
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
    district_id: int | None = Query(default=None, ge=1),
    search: str | None = Query(
        default=None,
        max_length=150,
    ),
    active_only: bool | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> UserListResponse:
    await require_permission(
        checker,
        current_user,
        USER_READ,
    )

    return await service.list_users(
        actor_id=actor_id(current_user),
        district_id=district_id,
        limit=limit,
        offset=offset,
        search=search,
        active_only=active_only,
    )


# ============================================================
# USER STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=UserStatisticsResponse,
)
async def user_statistics(
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
    district_id: int | None = Query(default=None, ge=1),
) -> UserStatisticsResponse:
    await require_permission(
        checker,
        current_user,
        USER_STATISTICS_READ,
    )

    return await service.statistics(
        actor_id=actor_id(current_user),
        district_id=district_id,
    )


# ============================================================
# CREATE USER
# ============================================================

@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    payload: UserCreate,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserResponse:
    await require_permission(
        checker,
        current_user,
        USER_CREATE,
    )

    return await service.create_user(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# GET USER
# ============================================================

@router.get(
    "/{user_id}",
    response_model=UserResponse,
)
async def get_user(
    user_id: UUID,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserResponse:
    await require_permission(
        checker,
        current_user,
        USER_READ,
    )

    return await require_target_access(
        service,
        user_id,
        current_user,
    )


# ============================================================
# UPDATE USER
# ============================================================

@router.patch(
    "/{user_id}",
    response_model=UserResponse,
)
async def update_user(
    user_id: UUID,
    payload: UserAdminUpdate,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserResponse:
    await require_permission(
        checker,
        current_user,
        USER_UPDATE,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.admin_update_user(
        user_id=str(user_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# ACTIVATE USER
# ============================================================

@router.post(
    "/{user_id}/activate",
    response_model=UserOperationResponse,
)
async def activate_user(
    user_id: UUID,
    payload: UserStatusChange,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserOperationResponse:
    await require_permission(
        checker,
        current_user,
        USER_ACTIVATE,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.activate_user(
        user_id=str(user_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DEACTIVATE USER
# ============================================================

@router.post(
    "/{user_id}/deactivate",
    response_model=UserOperationResponse,
)
async def deactivate_user(
    user_id: UUID,
    payload: UserStatusChange,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserOperationResponse:
    await require_permission(
        checker,
        current_user,
        USER_DEACTIVATE,
    )

    reject_self_privilege_change(
        user_id,
        current_user,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.deactivate_user(
        user_id=str(user_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DISTRICT ASSIGNMENT
# ============================================================

@router.post(
    "/{user_id}/district",
    response_model=UserResponse,
)
async def assign_user_district(
    user_id: UUID,
    payload: UserDistrictAssignment,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserResponse:
    await require_permission(
        checker,
        current_user,
        USER_DISTRICT_ASSIGN,
    )

    reject_self_privilege_change(
        user_id,
        current_user,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    if not payload.assignment_approved:
        raise HTTPException(
            status_code=422,
            detail="District assignment approval required",
        )

    return await service.assign_district(
        user_id=str(user_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# USER ROLES
# ============================================================

@router.get(
    "/{user_id}/roles",
    response_model=list[UserRoleResponse],
)
async def list_user_roles(
    user_id: UUID,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> list[UserRoleResponse]:
    await require_permission(
        checker,
        current_user,
        USER_ROLE_READ,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.list_roles(
        user_id=str(user_id),
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{user_id}/roles",
    response_model=UserRoleResponse,
    status_code=201,
)
async def assign_user_role(
    user_id: UUID,
    payload: UserRoleAssignment,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserRoleResponse:
    await require_permission(
        checker,
        current_user,
        USER_ROLE_ASSIGN,
    )

    reject_self_privilege_change(
        user_id,
        current_user,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.assign_role(
        user_id=str(user_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


@router.delete(
    "/{user_id}/roles/{role_id}",
    response_model=UserOperationResponse,
)
async def revoke_user_role(
    user_id: UUID,
    role_id: UUID,
    payload: UserRoleRevocation,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserOperationResponse:
    await require_permission(
        checker,
        current_user,
        USER_ROLE_REVOKE,
    )

    reject_self_privilege_change(
        user_id,
        current_user,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.revoke_role(
        user_id=str(user_id),
        role_id=str(role_id),
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# USER PERMISSIONS
# ============================================================

@router.get(
    "/{user_id}/permissions",
    response_model=list[UserPermissionResponse],
)
async def list_user_permissions(
    user_id: UUID,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> list[UserPermissionResponse]:
    await require_permission(
        checker,
        current_user,
        USER_ROLE_READ,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.list_permissions(
        user_id=str(user_id),
        actor_id=actor_id(current_user),
    )


# ============================================================
# ADMIN SESSION MANAGEMENT
# ============================================================

@router.get(
    "/{user_id}/sessions",
    response_model=list[UserSessionResponse],
)
async def list_user_sessions(
    user_id: UUID,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> list[UserSessionResponse]:
    await require_permission(
        checker,
        current_user,
        USER_SESSION_READ,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.list_sessions(
        user_id=str(user_id),
        actor_id=actor_id(current_user),
    )


@router.post(
    "/{user_id}/revoke-sessions",
    response_model=UserOperationResponse,
)
async def revoke_user_sessions(
    user_id: UUID,
    payload: UserSessionRevocation,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
) -> UserOperationResponse:
    await require_permission(
        checker,
        current_user,
        USER_SESSION_REVOKE,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.revoke_all_sessions(
        user_id=str(user_id),
        reason=payload.reason,
        actor_id=actor_id(current_user),
    )


# ============================================================
# USER ACTIVITY
# ============================================================

@router.get(
    "/{user_id}/activity",
    response_model=list[UserActivityResponse],
)
async def list_user_activity(
    user_id: UUID,
    current_user: CurrentUser,
    service: UserServiceDependency,
    checker: PermissionChecker,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[UserActivityResponse]:
    await require_permission(
        checker,
        current_user,
        USER_ACTIVITY_READ,
    )

    await require_target_access(
        service,
        user_id,
        current_user,
    )

    return await service.list_activity(
        user_id=str(user_id),
        actor_id=actor_id(current_user),
        limit=limit,
    )


__all__ = [
    "router",
    "UserService",
    "UserResponse",
    "UserListResponse",
    "UserStatisticsResponse",
    "UserRoleResponse",
    "UserPermissionResponse",
    "UserSessionResponse",
    "UserActivityResponse",
    "UserCreate",
    "UserProfileUpdate",
    "UserAdminUpdate",
    "UserRoleAssignment",
    "UserRoleRevocation",
    "UserDistrictAssignment",
]
