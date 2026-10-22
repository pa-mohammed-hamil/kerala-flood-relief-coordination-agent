from __future__ import annotations

"""
backend/api/missing_persons.py

Kerala Flood Relief Coordination Agent
Missing Persons, Search & Family Reunification API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Endpoints:
    GET    /missing-persons
    GET    /missing-persons/statistics
    POST   /missing-persons
    GET    /missing-persons/{person_id}
    PATCH  /missing-persons/{person_id}
    GET    /missing-persons/{person_id}/sightings
    POST   /missing-persons/{person_id}/sightings
    GET    /missing-persons/{person_id}/matches
    POST   /missing-persons/{person_id}/matches
    POST   /missing-persons/{person_id}/matches/{match_id}/review
    GET    /missing-persons/{person_id}/assignments
    POST   /missing-persons/{person_id}/assignments
    GET    /missing-persons/{person_id}/reunifications
    POST   /missing-persons/{person_id}/reunifications
    POST   /missing-persons/{person_id}/status
    GET    /missing-persons/{person_id}/events

Safety:
    - Missing-person data is highly sensitive.
    - Public search results must not expose private records.
    - AI matches are candidates, not identity confirmation.
    - Reunification requires authorized human verification.
    - Child safeguarding must be enforced by domain services.
    - Exact locations and contact information are restricted.
    - Operational changes require immutable audit records.
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
    Query,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.missing_person import (
    MissingPerson,
    MissingPersonSighting,
    MissingPersonMatch,
    MissingPersonSearchAssignment,
    MissingPersonReunification,
    MissingPersonEvent,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/missing-persons",
    tags=["Missing Persons & Reunification"],
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
# ROLE CONFIGURATION
# ============================================================

ADMIN_ROLES = {
    "admin",
    "super_admin",
}

COORDINATION_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "incident_commander",
}

SEARCH_ROLES = {
    "search_coordinator",
    "rescue_coordinator",
    "authorized_case_worker",
    "police_liaison",
}

REUNIFICATION_ROLES = {
    "authorized_case_worker",
    "reunification_officer",
    "child_protection_officer",
}

READ_ROLES = (
    ADMIN_ROLES
    | COORDINATION_ROLES
    | SEARCH_ROLES
    | REUNIFICATION_ROLES
)

CASE_WRITE_ROLES = (
    ADMIN_ROLES
    | COORDINATION_ROLES
    | SEARCH_ROLES
)

REVIEW_ROLES = (
    ADMIN_ROLES
    | COORDINATION_ROLES
    | REUNIFICATION_ROLES
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class MissingPersonStatus(str, Enum):
    REPORTED = "reported"
    UNDER_REVIEW = "under_review"
    SEARCH_ACTIVE = "search_active"
    POSSIBLE_MATCH = "possible_match"
    LOCATED = "located"
    REUNIFICATION_PENDING = "reunification_pending"
    REUNITED = "reunited"
    CLOSED = "closed"


class PersonAgeGroup(str, Enum):
    CHILD = "child"
    ADULT = "adult"
    SENIOR = "senior"
    UNKNOWN = "unknown"


class CasePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SightingConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class MatchReviewAction(str, Enum):
    CONFIRM_CANDIDATE = "confirm_candidate"
    REJECT = "reject"
    REQUEST_VERIFICATION = "request_verification"


class ReunificationStatus(str, Enum):
    PENDING = "pending"
    UNDER_VERIFICATION = "under_verification"
    APPROVED = "approved"
    COMPLETED = "completed"
    REJECTED = "rejected"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class MissingPersonCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    district_id: int = Field(ge=1)

    person_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    age_group: PersonAgeGroup

    estimated_age: int | None = Field(
        default=None,
        ge=0,
        le=120,
    )

    description: str = Field(
        min_length=10,
        max_length=5000,
    )

    last_seen_at: datetime | None = None

    last_seen_location_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    reporter_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    priority: CasePriority = CasePriority.MEDIUM

    safeguarding_required: bool = False

    consent_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_safeguarding(
        self,
    ) -> "MissingPersonCreate":
        if self.age_group == PersonAgeGroup.CHILD:
            self.safeguarding_required = True

        if (
            self.last_seen_at is not None
            and self.last_seen_at > datetime.now(
                timezone.utc
            )
        ):
            raise ValueError(
                "Last-seen time cannot be in the future"
            )

        return self


class MissingPersonUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(
        default=None,
        min_length=10,
        max_length=5000,
    )

    priority: CasePriority | None = None

    last_seen_at: datetime | None = None

    last_seen_location_reference: str | None = Field(
        default=None,
        max_length=500,
    )


class MissingPersonSightingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime

    location_reference: str = Field(
        min_length=3,
        max_length=500,
    )

    description: str = Field(
        min_length=10,
        max_length=5000,
    )

    confidence: SightingConfidence = (
        SightingConfidence.LOW
    )

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    reporter_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "Timezone-aware timestamp required"
            )

        if value > datetime.now(timezone.utc):
            raise ValueError(
                "Sighting time cannot be in the future"
            )

        return value


class MissingPersonMatchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    match_score: float = Field(
        ge=0,
        le=1,
    )

    matching_method: str = Field(
        min_length=3,
        max_length=100,
    )

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    ai_generated: bool = False

    model_reference: str | None = Field(
        default=None,
        max_length=150,
    )


class MissingPersonMatchReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: MatchReviewAction

    review_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    identity_evidence_verified: bool = False


class MissingPersonAssignmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assigned_team_id: str

    assignment_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    priority: CasePriority = CasePriority.HIGH

    @field_validator("assigned_team_id")
    @classmethod
    def validate_team_id(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))


class MissingPersonReunificationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    relationship: str = Field(
        min_length=3,
        max_length=100,
    )

    identity_verification_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    safeguarding_review_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    verification_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    guardian_authorization_verified: bool = False


class MissingPersonStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: MissingPersonStatus

    reason: str = Field(
        min_length=10,
        max_length=5000,
    )

    evidence_reference: str | None = Field(
        default=None,
        max_length=500,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class MissingPersonResponse(BaseModel):
    id: str
    district_id: int
    status: str
    priority: str | None = None
    created_at: datetime | None = None


class MissingPersonListResponse(BaseModel):
    items: list[MissingPersonResponse]
    total: int
    limit: int
    offset: int


class MissingPersonSightingResponse(BaseModel):
    id: str
    missing_person_id: str
    status: str | None = None


class MissingPersonMatchResponse(BaseModel):
    id: str
    missing_person_id: str
    status: str | None = None


class MissingPersonAssignmentResponse(BaseModel):
    id: str
    missing_person_id: str
    status: str | None = None


class MissingPersonReunificationResponse(BaseModel):
    id: str
    missing_person_id: str
    status: str | None = None


class MissingPersonStatisticsResponse(BaseModel):
    total_cases: int
    by_status: dict[str, int]
    generated_at: datetime


# ============================================================
# SECURITY HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid UUID",
        ) from None


def role_of(user: Any) -> str:
    return enum_value(
        getattr(user, "role", "")
    ).lower()


def require_role(
    user: Any,
    allowed_roles: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions",
        )


def require_district_access(
    user: Any,
    district_id: int,
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if role_of(user) in ADMIN_ROLES:
        return

    assigned_district = getattr(
        user,
        "district_id",
        None,
    )

    if (
        assigned_district is None
        or int(assigned_district) != district_id
    ):
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )


def scoped_district(
    user: Any,
    district_id: int | None,
) -> int | None:
    require_role(user, READ_ROLES)

    if (
        getattr(user, "is_superuser", False)
        or role_of(user) in ADMIN_ROLES
    ):
        return district_id

    assigned_district = getattr(
        user,
        "district_id",
        None,
    )

    if assigned_district is None:
        raise HTTPException(
            status_code=403,
            detail="District assignment required",
        )

    if (
        district_id is not None
        and district_id != int(assigned_district)
    ):
        raise HTTPException(
            status_code=403,
            detail="District access denied",
        )

    return int(assigned_district)


async def get_person_or_404(
    db: AsyncSession,
    person_id: str,
    user: Any,
) -> MissingPerson:
    identifier = validate_uuid(person_id)

    result = await db.execute(
        select(MissingPerson).where(
            MissingPerson.id == identifier
        )
    )

    person = result.scalar_one_or_none()

    if person is None:
        raise HTTPException(
            status_code=404,
            detail="Missing-person case not found",
        )

    require_district_access(
        user,
        person.district_id,
    )

    return person


def person_response(
    person: MissingPerson,
) -> MissingPersonResponse:
    return MissingPersonResponse(
        id=str(person.id),
        district_id=person.district_id,
        status=enum_value(person.status),
        priority=enum_value(
            getattr(person, "priority", None)
        ) or None,
        created_at=getattr(
            person,
            "created_at",
            None,
        ),
    )


def require_open_case(
    person: MissingPerson,
) -> None:
    if enum_value(person.status) in {
        "reunited",
        "closed",
    }:
        raise HTTPException(
            status_code=409,
            detail="Case is already finalized",
        )


# ============================================================
# LIST MISSING PERSONS
# ============================================================

@router.get(
    "",
    response_model=MissingPersonListResponse,
)
async def list_missing_persons(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> MissingPersonListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(MissingPerson)

    count_query = (
        select(func.count())
        .select_from(MissingPerson)
    )

    if district_id is not None:
        query = query.where(
            MissingPerson.district_id == district_id
        )

        count_query = count_query.where(
            MissingPerson.district_id == district_id
        )

    total_result = await db.execute(
        count_query
    )

    result = await db.execute(
        query.order_by(
            MissingPerson.created_at.desc()
        )
        .offset(offset)
        .limit(limit)
    )

    return MissingPersonListResponse(
        items=[
            person_response(person)
            for person in result.scalars().all()
        ],
        total=int(total_result.scalar_one()),
        limit=limit,
        offset=offset,
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=MissingPersonStatisticsResponse,
)
async def missing_person_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> MissingPersonStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(
        MissingPerson.status,
        func.count(MissingPerson.id),
    )

    if district_id is not None:
        query = query.where(
            MissingPerson.district_id == district_id
        )

    query = query.group_by(MissingPerson.status)

    result = await db.execute(query)

    by_status = {
        enum_value(row[0]): int(row[1])
        for row in result.all()
    }

    return MissingPersonStatisticsResponse(
        total_cases=sum(by_status.values()),
        by_status=by_status,
        generated_at=utc_now(),
    )


# ============================================================
# CREATE CASE
# ============================================================

@router.post(
    "",
    response_model=MissingPersonResponse,
    status_code=201,
)
async def create_missing_person_case(
    payload: MissingPersonCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> MissingPersonResponse:
    require_role(current_user, CASE_WRITE_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    # Requires a dedicated case service for:
    # - Sensitive data encryption
    # - Duplicate case detection
    # - Reporter verification
    # - Child safeguarding
    # - Consent/legal-basis checks
    # - Immutable audit logging
    raise HTTPException(
        status_code=501,
        detail="Missing-person case creation service is not configured",
    )


# ============================================================
# CASE DETAILS
# ============================================================

@router.get(
    "/{person_id}",
    response_model=MissingPersonResponse,
)
async def get_missing_person(
    person_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> MissingPersonResponse:
    require_role(current_user, READ_ROLES)

    person = await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    return person_response(person)


# ============================================================
# UPDATE CASE
# ============================================================

@router.patch(
    "/{person_id}",
    response_model=MissingPersonResponse,
)
async def update_missing_person(
    person_id: str,
    payload: MissingPersonUpdate,
    db: DbSession,
    current_user: CurrentUser,
) -> MissingPersonResponse:
    require_role(current_user, CASE_WRITE_ROLES)

    person = await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    require_open_case(person)

    raise HTTPException(
        status_code=501,
        detail="Missing-person case update service is not configured",
    )


# ============================================================
# SIGHTINGS
# ============================================================

@router.get(
    "/{person_id}/sightings",
)
async def list_missing_person_sightings(
    person_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[MissingPersonSightingResponse]:
    require_role(current_user, READ_ROLES)

    await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    result = await db.execute(
        select(MissingPersonSighting)
        .where(
            MissingPersonSighting.missing_person_id
            == validate_uuid(person_id)
        )
        .limit(limit)
    )

    return [
        MissingPersonSightingResponse(
            id=str(item.id),
            missing_person_id=str(
                item.missing_person_id
            ),
            status=enum_value(
                getattr(item, "status", None)
            ) or None,
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/{person_id}/sightings",
    status_code=201,
)
async def create_missing_person_sighting(
    person_id: str,
    payload: MissingPersonSightingCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, CASE_WRITE_ROLES)

    person = await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    require_open_case(person)

    raise HTTPException(
        status_code=501,
        detail="Sighting verification service is not configured",
    )


# ============================================================
# POTENTIAL MATCHES
# ============================================================

@router.get(
    "/{person_id}/matches",
)
async def list_missing_person_matches(
    person_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[MissingPersonMatchResponse]:
    require_role(current_user, REVIEW_ROLES)

    await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    result = await db.execute(
        select(MissingPersonMatch)
        .where(
            MissingPersonMatch.missing_person_id
            == validate_uuid(person_id)
        )
        .limit(limit)
    )

    return [
        MissingPersonMatchResponse(
            id=str(item.id),
            missing_person_id=str(
                item.missing_person_id
            ),
            status=enum_value(
                getattr(item, "status", None)
            ) or None,
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/{person_id}/matches",
    status_code=201,
)
async def create_missing_person_match(
    person_id: str,
    payload: MissingPersonMatchCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, REVIEW_ROLES)

    person = await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    require_open_case(person)

    # AI similarity scores are never identity proof.
    raise HTTPException(
        status_code=501,
        detail="Potential match service is not configured",
    )


@router.post(
    "/{person_id}/matches/{match_id}/review",
)
async def review_missing_person_match(
    person_id: str,
    match_id: str,
    payload: MissingPersonMatchReview,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, REVIEW_ROLES)

    await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    validate_uuid(match_id)

    if (
        payload.action
        == MatchReviewAction.CONFIRM_CANDIDATE
        and not payload.identity_evidence_verified
    ):
        raise HTTPException(
            status_code=422,
            detail="Independent identity verification required",
        )

    raise HTTPException(
        status_code=501,
        detail="Match review workflow is not configured",
    )


# ============================================================
# SEARCH ASSIGNMENTS
# ============================================================

@router.get(
    "/{person_id}/assignments",
)
async def list_search_assignments(
    person_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[MissingPersonAssignmentResponse]:
    require_role(current_user, READ_ROLES)

    await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    result = await db.execute(
        select(MissingPersonSearchAssignment)
        .where(
            MissingPersonSearchAssignment.missing_person_id
            == validate_uuid(person_id)
        )
    )

    return [
        MissingPersonAssignmentResponse(
            id=str(item.id),
            missing_person_id=str(
                item.missing_person_id
            ),
            status=enum_value(
                getattr(item, "status", None)
            ) or None,
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/{person_id}/assignments",
    status_code=201,
)
async def create_search_assignment(
    person_id: str,
    payload: MissingPersonAssignmentCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, CASE_WRITE_ROLES)

    person = await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    require_open_case(person)

    # Must validate team jurisdiction, availability,
    # safeguarding clearance and command approval.
    raise HTTPException(
        status_code=501,
        detail="Search assignment service is not configured",
    )


# ============================================================
# FAMILY REUNIFICATION
# ============================================================

@router.get(
    "/{person_id}/reunifications",
)
async def list_reunifications(
    person_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[MissingPersonReunificationResponse]:
    require_role(current_user, REVIEW_ROLES)

    await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    result = await db.execute(
        select(MissingPersonReunification)
        .where(
            MissingPersonReunification.missing_person_id
            == validate_uuid(person_id)
        )
    )

    return [
        MissingPersonReunificationResponse(
            id=str(item.id),
            missing_person_id=str(
                item.missing_person_id
            ),
            status=enum_value(
                getattr(item, "status", None)
            ) or None,
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/{person_id}/reunifications",
    status_code=201,
)
async def create_reunification(
    person_id: str,
    payload: MissingPersonReunificationCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, REVIEW_ROLES)

    person = await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    require_open_case(person)

    # Children require additional safeguarding review.
    # Family relationships and legal authority must be
    # independently verified.
    raise HTTPException(
        status_code=501,
        detail="Family reunification service is not configured",
    )


# ============================================================
# CASE STATUS
# ============================================================

@router.post(
    "/{person_id}/status",
)
async def update_missing_person_status(
    person_id: str,
    payload: MissingPersonStatusUpdate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, REVIEW_ROLES)

    person = await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    require_open_case(person)

    # A case must not be marked reunited solely from
    # an AI match or unverified sighting.
    raise HTTPException(
        status_code=501,
        detail="Missing-person status workflow is not configured",
    )


# ============================================================
# AUDIT EVENTS
# ============================================================

@router.get(
    "/{person_id}/events",
)
async def missing_person_events(
    person_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> list[dict[str, Any]]:
    require_role(current_user, REVIEW_ROLES)

    await get_person_or_404(
        db,
        person_id,
        current_user,
    )

    result = await db.execute(
        select(MissingPersonEvent)
        .where(
            MissingPersonEvent.missing_person_id
            == validate_uuid(person_id)
        )
        .limit(limit)
    )

    return [
        {
            "id": str(item.id),
            "missing_person_id": str(
                item.missing_person_id
            ),
        }
        for item in result.scalars().all()
    ]


__all__ = [
    "router",
    "MissingPersonCreate",
    "MissingPersonUpdate",
    "MissingPersonSightingCreate",
    "MissingPersonMatchCreate",
    "MissingPersonMatchReview",
    "MissingPersonAssignmentCreate",
    "MissingPersonReunificationCreate",
    "MissingPersonStatusUpdate",
    "MissingPersonResponse",
    "MissingPersonListResponse",
    "MissingPersonStatisticsResponse",
]
