from __future__ import annotations

"""
backend/core/permissions.py

Kerala Flood Relief Coordination Agent

Role-Based Access Control (RBAC) for FastAPI.

Features:
- Role and permission definitions
- Centralized permission matrix
- FastAPI dependency authorization
- District-level access restrictions
- Resource ownership validation
- Human approval for critical actions
- Emergency dispatch authorization
- AI agent execution restrictions
- Least-privilege access
- Explicit denial by default

Integrate get_current_user with your existing
backend/core/security.py authentication system.
"""


from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from fastapi import Depends, HTTPException, status

from backend.core.exceptions import (
    AuthenticationError,
    PermissionDeniedError,
    HumanApprovalRequiredError,
)


# ============================================================
# ROLES
# ============================================================

class Role(str, Enum):
    SUPER_ADMIN = "super_admin"
    STATE_ADMIN = "state_admin"
    DISTRICT_ADMIN = "district_admin"
    DISASTER_OFFICER = "disaster_officer"
    RESCUE_COORDINATOR = "rescue_coordinator"
    RESCUE_TEAM = "rescue_team"
    MEDICAL_OFFICER = "medical_officer"
    HOSPITAL_STAFF = "hospital_staff"
    SHELTER_MANAGER = "shelter_manager"
    RELIEF_COORDINATOR = "relief_coordinator"
    VOLUNTEER_COORDINATOR = "volunteer_coordinator"
    VOLUNTEER = "volunteer"
    AI_AGENT = "ai_agent"
    AUDITOR = "auditor"
    CITIZEN = "citizen"


# ============================================================
# PERMISSIONS
# ============================================================

class Permission(str, Enum):

    # Users
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_MANAGE_ROLES = "user:manage_roles"

    # Districts
    DISTRICT_READ = "district:read"
    DISTRICT_MANAGE = "district:manage"

    # Flood monitoring
    FLOOD_READ = "flood:read"
    FLOOD_FORECAST = "flood:forecast"
    FLOOD_ALERT_CREATE = "flood:alert_create"
    FLOOD_ALERT_APPROVE = "flood:alert_approve"
    FLOOD_ALERT_PUBLISH = "flood:alert_publish"

    # SOS
    SOS_CREATE = "sos:create"
    SOS_READ = "sos:read"
    SOS_READ_ALL = "sos:read_all"
    SOS_UPDATE = "sos:update"
    SOS_TRIAGE = "sos:triage"
    SOS_ESCALATE = "sos:escalate"
    SOS_CANCEL = "sos:cancel"

    # Incidents
    INCIDENT_READ = "incident:read"
    INCIDENT_CREATE = "incident:create"
    INCIDENT_UPDATE = "incident:update"
    INCIDENT_CLOSE = "incident:close"

    # Rescue
    RESCUE_READ = "rescue:read"
    RESCUE_CREATE = "rescue:create"
    RESCUE_ASSIGN = "rescue:assign"
    RESCUE_DISPATCH = "rescue:dispatch"
    RESCUE_UPDATE = "rescue:update"
    RESCUE_COMPLETE = "rescue:complete"

    # Evacuation
    EVACUATION_READ = "evacuation:read"
    EVACUATION_PLAN = "evacuation:plan"
    EVACUATION_APPROVE = "evacuation:approve"
    EVACUATION_EXECUTE = "evacuation:execute"

    # Shelters
    SHELTER_READ = "shelter:read"
    SHELTER_CREATE = "shelter:create"
    SHELTER_UPDATE = "shelter:update"
    SHELTER_ALLOCATE = "shelter:allocate"
    SHELTER_MANAGE_CAPACITY = "shelter:manage_capacity"

    # Relief and inventory
    RELIEF_READ = "relief:read"
    RELIEF_CREATE = "relief:create"
    RELIEF_DISTRIBUTE = "relief:distribute"
    INVENTORY_READ = "inventory:read"
    INVENTORY_UPDATE = "inventory:update"

    # Healthcare
    HEALTHCARE_READ = "healthcare:read"
    HEALTHCARE_UPDATE = "healthcare:update"
    MEDICAL_TRIAGE = "medical:triage"
    AMBULANCE_REQUEST = "ambulance:request"
    AMBULANCE_DISPATCH = "ambulance:dispatch"

    # Volunteers
    VOLUNTEER_READ = "volunteer:read"
    VOLUNTEER_REGISTER = "volunteer:register"
    VOLUNTEER_UPDATE = "volunteer:update"
    VOLUNTEER_VERIFY = "volunteer:verify"
    VOLUNTEER_ASSIGN = "volunteer:assign"

    # Missing persons
    MISSING_PERSON_READ = "missing_person:read"
    MISSING_PERSON_CREATE = "missing_person:create"
    MISSING_PERSON_UPDATE = "missing_person:update"

    # AI agents
    AGENT_READ = "agent:read"
    AGENT_EXECUTE = "agent:execute"
    AGENT_MANAGE = "agent:manage"
    AGENT_APPROVE = "agent:approve"

    # RAG
    RAG_QUERY = "rag:query"
    RAG_DOCUMENT_UPLOAD = "rag:document_upload"
    RAG_DOCUMENT_DELETE = "rag:document_delete"

    # Analytics and reports
    ANALYTICS_READ = "analytics:read"
    REPORT_GENERATE = "report:generate"
    REPORT_EXPORT = "report:export"

    # Notifications
    NOTIFICATION_READ = "notification:read"
    NOTIFICATION_SEND = "notification:send"

    # Audit and system
    AUDIT_READ = "audit:read"
    SYSTEM_CONFIG_READ = "system:config_read"
    SYSTEM_CONFIG_UPDATE = "system:config_update"
    SYSTEM_MONITOR = "system:monitor"


# ============================================================
# ROLE PERMISSION MATRIX
# ============================================================

PUBLIC_PERMISSIONS = {
    Permission.FLOOD_READ,
    Permission.SHELTER_READ,
}

CITIZEN_PERMISSIONS = PUBLIC_PERMISSIONS | {
    Permission.SOS_CREATE,
    Permission.SOS_READ,
    Permission.SOS_CANCEL,
    Permission.VOLUNTEER_REGISTER,
    Permission.RAG_QUERY,
    Permission.NOTIFICATION_READ,
    Permission.MISSING_PERSON_CREATE,
}

RESCUE_PERMISSIONS = {
    Permission.FLOOD_READ,
    Permission.SOS_READ,
    Permission.INCIDENT_READ,
    Permission.RESCUE_READ,
    Permission.RESCUE_UPDATE,
    Permission.RESCUE_COMPLETE,
    Permission.EVACUATION_READ,
    Permission.SHELTER_READ,
}

MEDICAL_PERMISSIONS = {
    Permission.SOS_READ,
    Permission.HEALTHCARE_READ,
    Permission.HEALTHCARE_UPDATE,
    Permission.MEDICAL_TRIAGE,
    Permission.AMBULANCE_REQUEST,
    Permission.SHELTER_READ,
}

SHELTER_PERMISSIONS = {
    Permission.SHELTER_READ,
    Permission.SHELTER_UPDATE,
    Permission.SHELTER_ALLOCATE,
    Permission.SHELTER_MANAGE_CAPACITY,
    Permission.INVENTORY_READ,
}

VOLUNTEER_PERMISSIONS = {
    Permission.FLOOD_READ,
    Permission.SHELTER_READ,
    Permission.VOLUNTEER_READ,
    Permission.VOLUNTEER_UPDATE,
    Permission.RESCUE_READ,
}

COORDINATOR_PERMISSIONS = {
    Permission.SOS_READ_ALL,
    Permission.SOS_TRIAGE,
    Permission.SOS_ESCALATE,
    Permission.INCIDENT_READ,
    Permission.INCIDENT_CREATE,
    Permission.INCIDENT_UPDATE,
    Permission.RESCUE_READ,
    Permission.RESCUE_CREATE,
    Permission.RESCUE_ASSIGN,
    Permission.EVACUATION_READ,
    Permission.EVACUATION_PLAN,
    Permission.VOLUNTEER_READ,
    Permission.VOLUNTEER_ASSIGN,
    Permission.ANALYTICS_READ,
    Permission.REPORT_GENERATE,
}

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.SUPER_ADMIN: frozenset(Permission),

    Role.STATE_ADMIN: frozenset(Permission),

    Role.DISTRICT_ADMIN: frozenset(
        COORDINATOR_PERMISSIONS
        | {
            Permission.USER_READ,
            Permission.USER_CREATE,
            Permission.USER_UPDATE,
            Permission.DISTRICT_READ,
            Permission.FLOOD_READ,
            Permission.FLOOD_FORECAST,
            Permission.FLOOD_ALERT_CREATE,
            Permission.FLOOD_ALERT_APPROVE,
            Permission.FLOOD_ALERT_PUBLISH,
            Permission.RESCUE_DISPATCH,
            Permission.EVACUATION_APPROVE,
            Permission.EVACUATION_EXECUTE,
            Permission.SHELTER_READ,
            Permission.SHELTER_CREATE,
            Permission.SHELTER_UPDATE,
            Permission.RELIEF_READ,
            Permission.RELIEF_DISTRIBUTE,
            Permission.INVENTORY_READ,
            Permission.VOLUNTEER_VERIFY,
            Permission.AGENT_READ,
            Permission.AGENT_APPROVE,
            Permission.AUDIT_READ,
            Permission.REPORT_EXPORT,
            Permission.NOTIFICATION_SEND,
        }
    ),

    Role.DISASTER_OFFICER: frozenset(
        COORDINATOR_PERMISSIONS
        | {
            Permission.FLOOD_READ,
            Permission.FLOOD_FORECAST,
            Permission.FLOOD_ALERT_CREATE,
            Permission.FLOOD_ALERT_APPROVE,
            Permission.FLOOD_ALERT_PUBLISH,
            Permission.RESCUE_DISPATCH,
            Permission.EVACUATION_APPROVE,
            Permission.EVACUATION_EXECUTE,
            Permission.NOTIFICATION_SEND,
            Permission.AGENT_APPROVE,
        }
    ),

    Role.RESCUE_COORDINATOR: frozenset(
        RESCUE_PERMISSIONS
        | COORDINATOR_PERMISSIONS
        | {
            Permission.RESCUE_DISPATCH,
            Permission.EVACUATION_EXECUTE,
        }
    ),

    Role.RESCUE_TEAM: frozenset(
        RESCUE_PERMISSIONS
    ),

    Role.MEDICAL_OFFICER: frozenset(
        MEDICAL_PERMISSIONS
        | {
            Permission.AMBULANCE_DISPATCH,
            Permission.SOS_TRIAGE,
        }
    ),

    Role.HOSPITAL_STAFF: frozenset(
        MEDICAL_PERMISSIONS
    ),

    Role.SHELTER_MANAGER: frozenset(
        SHELTER_PERMISSIONS
    ),

    Role.RELIEF_COORDINATOR: frozenset({
        Permission.RELIEF_READ,
        Permission.RELIEF_CREATE,
        Permission.RELIEF_DISTRIBUTE,
        Permission.INVENTORY_READ,
        Permission.INVENTORY_UPDATE,
        Permission.SHELTER_READ,
        Permission.ANALYTICS_READ,
    }),

    Role.VOLUNTEER_COORDINATOR: frozenset({
        Permission.VOLUNTEER_READ,
        Permission.VOLUNTEER_REGISTER,
        Permission.VOLUNTEER_UPDATE,
        Permission.VOLUNTEER_VERIFY,
        Permission.VOLUNTEER_ASSIGN,
        Permission.RESCUE_READ,
        Permission.SHELTER_READ,
    }),

    Role.VOLUNTEER: frozenset(
        VOLUNTEER_PERMISSIONS
    ),

    Role.AI_AGENT: frozenset({
        Permission.FLOOD_READ,
        Permission.FLOOD_FORECAST,
        Permission.SOS_TRIAGE,
        Permission.INCIDENT_READ,
        Permission.RESCUE_READ,
        Permission.EVACUATION_READ,
        Permission.SHELTER_READ,
        Permission.RELIEF_READ,
        Permission.INVENTORY_READ,
        Permission.AGENT_READ,
        Permission.AGENT_EXECUTE,
        Permission.RAG_QUERY,
        Permission.ANALYTICS_READ,
    }),

    Role.AUDITOR: frozenset({
        Permission.AUDIT_READ,
        Permission.ANALYTICS_READ,
        Permission.REPORT_GENERATE,
        Permission.SYSTEM_MONITOR,
    }),

    Role.CITIZEN: frozenset(
        CITIZEN_PERMISSIONS
    ),
}


# ============================================================
# USER AUTHORIZATION CONTEXT
# ============================================================

@dataclass(frozen=True)
class AuthorizationContext:
    user_id: int
    roles: frozenset[Role]
    district_ids: frozenset[int] = field(
        default_factory=frozenset
    )
    is_active: bool = True
    is_verified: bool = False

    @classmethod
    def from_user(
        cls,
        user: Any,
    ) -> "AuthorizationContext":
        if user is None:
            raise AuthenticationError()

        raw_roles = getattr(user, "roles", None)

        if raw_roles is None:
            raw_role = getattr(user, "role", None)
            raw_roles = (
                [raw_role]
                if raw_role is not None
                else []
            )

        if isinstance(raw_roles, (str, Role)):
            raw_roles = [raw_roles]

        try:
            roles = frozenset(
                Role(
                    getattr(role, "value", role)
                )
                for role in raw_roles
            )
        except (TypeError, ValueError) as exc:
            raise PermissionDeniedError(
                "Invalid role assignment."
            ) from exc

        raw_districts = getattr(
            user,
            "district_ids",
            None,
        )

        if raw_districts is None:
            district_id = getattr(
                user,
                "district_id",
                None,
            )
            raw_districts = (
                [district_id]
                if district_id is not None
                else []
            )

        try:
            district_ids = frozenset(
                int(district_id)
                for district_id in raw_districts
            )
        except (TypeError, ValueError) as exc:
            raise PermissionDeniedError(
                "Invalid district assignment."
            ) from exc

        user_id = getattr(user, "id", None)

        if user_id is None:
            raise AuthenticationError()

        return cls(
            user_id=int(user_id),
            roles=roles,
            district_ids=district_ids,
            is_active=bool(
                getattr(user, "is_active", False)
            ),
            is_verified=bool(
                getattr(user, "is_verified", False)
            ),
        )


# ============================================================
# PERMISSION CHECKING
# ============================================================

def get_role_permissions(
    role: Role,
) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(
        role,
        frozenset(),
    )


def get_user_permissions(
    context: AuthorizationContext,
) -> frozenset[Permission]:
    permissions: set[Permission] = set()

    for role in context.roles:
        permissions.update(
            get_role_permissions(role)
        )

    return frozenset(permissions)


def has_permission(
    context: AuthorizationContext,
    permission: Permission,
) -> bool:
    if not context.is_active:
        return False

    return permission in get_user_permissions(
        context
    )


def has_any_permission(
    context: AuthorizationContext,
    permissions: Iterable[Permission],
) -> bool:
    return any(
        has_permission(context, permission)
        for permission in permissions
    )


def has_all_permissions(
    context: AuthorizationContext,
    permissions: Iterable[Permission],
) -> bool:
    return all(
        has_permission(context, permission)
        for permission in permissions
    )


def require_permission(
    context: AuthorizationContext,
    permission: Permission,
) -> None:
    if not has_permission(
        context,
        permission,
    ):
        raise PermissionDeniedError(
            "Required permission is missing."
        )


def require_any_permission(
    context: AuthorizationContext,
    permissions: Iterable[Permission],
) -> None:
    if not has_any_permission(
        context,
        permissions,
    ):
        raise PermissionDeniedError(
            "Insufficient permissions."
        )


def require_all_permissions(
    context: AuthorizationContext,
    permissions: Iterable[Permission],
) -> None:
    if not has_all_permissions(
        context,
        permissions,
    ):
        raise PermissionDeniedError(
            "Insufficient permissions."
        )


# ============================================================
# DISTRICT ACCESS CONTROL
# ============================================================

STATEWIDE_ROLES = {
    Role.SUPER_ADMIN,
    Role.STATE_ADMIN,
}


def can_access_district(
    context: AuthorizationContext,
    district_id: int,
) -> bool:
    if not context.is_active:
        return False

    if context.roles.intersection(
        STATEWIDE_ROLES
    ):
        return True

    return district_id in context.district_ids


def require_district_access(
    context: AuthorizationContext,
    district_id: int,
) -> None:
    if not can_access_district(
        context,
        district_id,
    ):
        raise PermissionDeniedError(
            "Access to this district is denied."
        )


# ============================================================
# RESOURCE OWNERSHIP
# ============================================================

def is_resource_owner(
    context: AuthorizationContext,
    owner_id: int | None,
) -> bool:
    return (
        context.is_active
        and owner_id is not None
        and context.user_id == owner_id
    )


def require_resource_owner(
    context: AuthorizationContext,
    owner_id: int | None,
) -> None:
    if not is_resource_owner(
        context,
        owner_id,
    ):
        raise PermissionDeniedError(
            "Resource ownership verification failed."
        )


def authorize_resource(
    context: AuthorizationContext,
    *,
    permission: Permission,
    district_id: int | None = None,
    owner_id: int | None = None,
    allow_owner: bool = False,
) -> None:
    if allow_owner and is_resource_owner(
        context,
        owner_id,
    ):
        return

    require_permission(
        context,
        permission,
    )

    if district_id is not None:
        require_district_access(
            context,
            district_id,
        )


# ============================================================
# CRITICAL OPERATIONS
# ============================================================

CRITICAL_PERMISSIONS = frozenset({
    Permission.FLOOD_ALERT_PUBLISH,
    Permission.RESCUE_DISPATCH,
    Permission.EVACUATION_EXECUTE,
    Permission.AMBULANCE_DISPATCH,
    Permission.AGENT_APPROVE,
})


def require_human_approval(
    *,
    permission: Permission,
    approved: bool,
    reviewer_id: int | None,
    actor_id: int | None = None,
) -> None:
    if permission not in CRITICAL_PERMISSIONS:
        return

    if not approved or reviewer_id is None:
        raise HumanApprovalRequiredError()

    if (
        actor_id is not None
        and reviewer_id == actor_id
    ):
        raise PermissionDeniedError(
            "Independent approval is required."
        )


def authorize_critical_action(
    context: AuthorizationContext,
    *,
    permission: Permission,
    district_id: int,
    approved: bool,
    reviewer_id: int | None,
) -> None:
    require_permission(
        context,
        permission,
    )

    require_district_access(
        context,
        district_id,
    )

    require_human_approval(
        permission=permission,
        approved=approved,
        reviewer_id=reviewer_id,
        actor_id=context.user_id,
    )


# ============================================================
# AI AGENT RESTRICTIONS
# ============================================================

AI_PROHIBITED_ACTIONS = frozenset({
    Permission.USER_DELETE,
    Permission.USER_MANAGE_ROLES,
    Permission.FLOOD_ALERT_PUBLISH,
    Permission.RESCUE_DISPATCH,
    Permission.EVACUATION_EXECUTE,
    Permission.AMBULANCE_DISPATCH,
    Permission.RELIEF_DISTRIBUTE,
    Permission.SYSTEM_CONFIG_UPDATE,
})


def authorize_agent_action(
    context: AuthorizationContext,
    permission: Permission,
    *,
    district_id: int | None = None,
) -> None:
    if Role.AI_AGENT not in context.roles:
        raise PermissionDeniedError(
            "AI agent identity required."
        )

    if permission in AI_PROHIBITED_ACTIONS:
        raise PermissionDeniedError(
            "AI agents cannot independently execute "
            "this critical operation."
        )

    require_permission(
        context,
        permission,
    )

    if district_id is not None:
        require_district_access(
            context,
            district_id,
        )


# ============================================================
# FASTAPI DEPENDENCIES
# ============================================================

async def get_current_user():
    """
    Authentication integration placeholder.

    Replace this function or override its dependency
    with the verified authentication dependency from
    backend/core/security.py.

    Never trust client-supplied roles or user IDs.
    """
    raise AuthenticationError(
        "Authentication dependency is not configured."
    )


async def get_authorization_context(
    user: Any = Depends(get_current_user),
) -> AuthorizationContext:
    return AuthorizationContext.from_user(
        user
    )


def permission_required(
    permission: Permission,
) -> Callable:
    async def dependency(
        context: AuthorizationContext = Depends(
            get_authorization_context
        ),
    ) -> AuthorizationContext:
        require_permission(
            context,
            permission,
        )

        return context

    return dependency


def any_permission_required(
    *permissions: Permission,
) -> Callable:
    async def dependency(
        context: AuthorizationContext = Depends(
            get_authorization_context
        ),
    ) -> AuthorizationContext:
        require_any_permission(
            context,
            permissions,
        )

        return context

    return dependency


def all_permissions_required(
    *permissions: Permission,
) -> Callable:
    async def dependency(
        context: AuthorizationContext = Depends(
            get_authorization_context
        ),
    ) -> AuthorizationContext:
        require_all_permissions(
            context,
            permissions,
        )

        return context

    return dependency


def role_required(
    *allowed_roles: Role,
) -> Callable:
    async def dependency(
        context: AuthorizationContext = Depends(
            get_authorization_context
        ),
    ) -> AuthorizationContext:
        if not context.is_active:
            raise PermissionDeniedError(
                "Account is inactive."
            )

        if not context.roles.intersection(
            allowed_roles
        ):
            raise PermissionDeniedError(
                "Required role is missing."
            )

        return context

    return dependency


# ============================================================
# FASTAPI USAGE EXAMPLE
# ============================================================

"""
from fastapi import APIRouter, Depends

from backend.core.permissions import (
    Permission,
    AuthorizationContext,
    permission_required,
    require_district_access,
)

router = APIRouter(prefix="/rescue", tags=["Rescue"])


@router.get("/missions")
async def list_rescue_missions(
    context: AuthorizationContext = Depends(
        permission_required(Permission.RESCUE_READ)
    ),
):
    return {
        "message": "Authorized rescue mission access",
        "user_id": context.user_id,
    }


@router.get("/district/{district_id}")
async def district_rescue_missions(
    district_id: int,
    context: AuthorizationContext = Depends(
        permission_required(Permission.RESCUE_READ)
    ),
):
    require_district_access(
        context,
        district_id,
    )

    return {
        "district_id": district_id,
        "authorized": True,
    }
"""
