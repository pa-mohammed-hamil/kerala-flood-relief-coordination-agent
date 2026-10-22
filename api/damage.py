from __future__ import annotations

"""
backend/api/damage.py

Kerala Flood Relief Coordination Agent
Flood Damage Assessment API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Endpoints:
    POST   /damage/reports
    GET    /damage/reports
    GET    /damage/reports/{report_id}
    PATCH  /damage/reports/{report_id}
    POST   /damage/reports/{report_id}/submit
    POST   /damage/reports/{report_id}/assessments
    GET    /damage/reports/{report_id}/assessments
    POST   /damage/reports/{report_id}/evidence
    GET    /damage/reports/{report_id}/evidence
    POST   /damage/reports/{report_id}/cost-items
    GET    /damage/reports/{report_id}/cost-items
    GET    /damage/reports/{report_id}/cost-summary
    POST   /damage/reports/{report_id}/reviews
    GET    /damage/reports/{report_id}/reviews
    GET    /damage/statistics

Security:
    JWT authentication
    District-based authorization
    Human approval for consequential assessments
    Private evidence references
    AI-generated estimates are advisory only
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
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.damage_report import (
    DamageReport,
    DamageAssessment,
    DamageEvidence,
    DamageCostItem,
    DamageReportReview,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/damage",
    tags=["Flood Damage Assessment"],
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
# CONFIGURATION
# ============================================================

ADMIN_ROLES = {
    "admin",
    "super_admin",
}

ASSESSOR_ROLES = {
    "damage_assessor",
    "district_officer",
    "emergency_coordinator",
}

REVIEWER_ROLES = {
    "damage_reviewer",
    "district_officer",
    "emergency_coordinator",
}

OPERATOR_ROLES = {
    "operator",
    "field_officer",
}

READ_ROLES = (
    ADMIN_ROLES
    | ASSESSOR_ROLES
    | REVIEWER_ROLES
    | OPERATOR_ROLES
)

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


# ============================================================
# SCHEMAS
# ============================================================

class DamageReportCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    district_id: int = Field(ge=1)

    title: str = Field(
        min_length=5,
        max_length=255,
    )

    description: str = Field(
        min_length=10,
        max_length=10000,
    )

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
    )

    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
    )

    incident_id: str | None = None

    property_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    @field_validator("incident_id")
    @classmethod
    def validate_incident_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return str(uuid.UUID(value))


class DamageReportUpdate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    title: str | None = Field(
        default=None,
        min_length=5,
        max_length=255,
    )

    description: str | None = Field(
        default=None,
        min_length=10,
        max_length=10000,
    )


class DamageAssessmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: str = Field(
        min_length=10,
        max_length=10000,
    )

    estimated_damage_percentage: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    ai_generated: bool = False

    model_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    confidence_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class DamageEvidenceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_type: str = Field(
        min_length=2,
        max_length=100,
    )

    storage_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    sha256_checksum: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
    )


class DamageCostItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        min_length=3,
        max_length=500,
    )

    quantity: float = Field(gt=0)

    unit_cost: float = Field(ge=0)

    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    is_ai_estimate: bool = False


class DamageReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class DamageReviewRequest(BaseModel):
    action: DamageReviewAction

    notes: str = Field(
        min_length=10,
        max_length=5000,
    )


class DamageReportResponse(BaseModel):
    id: str
    district_id: int
    status: str
    created_at: datetime | None = None


class DamageAssessmentResponse(BaseModel):
    id: str
    report_id: str
    created_at: datetime | None = None


class DamageEvidenceResponse(BaseModel):
    id: str
    report_id: str


class DamageCostItemResponse(BaseModel):
    id: str
    report_id: str


class DamageCostSummary(BaseModel):
    report_id: str
    currency: str
    total_estimated_cost: float
    item_count: int
    human_verified: bool = False


class DamageStatisticsResponse(BaseModel):
    total_reports: int
    by_status: dict[str, int]
    generated_at: datetime


class PaginatedDamageReports(BaseModel):
    items: list[DamageReportResponse]
    total: int
    limit: int
    offset: int


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid UUID",
        ) from None


def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return str(identifier)


def user_role(user: Any) -> str:
    return enum_value(
        getattr(user, "role", "")
    ).lower()


def require_role(
    user: Any,
    allowed_roles: set[str],
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if user_role(user) not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )


def require_district_access(
    user: Any,
    district_id: int,
) -> None:
    if getattr(user, "is_superuser", False):
        return

    if user_role(user) in ADMIN_ROLES:
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
            status_code=status.HTTP_403_FORBIDDEN,
            detail="District access denied",
        )


async def get_report_or_404(
    db: AsyncSession,
    report_id: str,
    current_user: Any,
) -> DamageReport:
    identifier = validate_uuid(report_id)

    result = await db.execute(
        select(DamageReport).where(
            DamageReport.id == identifier
        )
    )

    report = result.scalar_one_or_none()

    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Damage report not found",
        )

    require_district_access(
        current_user,
        report.district_id,
    )

    return report


def report_response(
    report: DamageReport,
) -> DamageReportResponse:
    return DamageReportResponse(
        id=str(report.id),
        district_id=report.district_id,
        status=enum_value(report.status),
        created_at=report.created_at,
    )


def ensure_editable(report: DamageReport) -> None:
    if enum_value(report.status) not in {
        "draft",
        "returned_for_changes",
        "needs_revision",
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Report is not editable",
        )


# ============================================================
# CREATE REPORT
# ============================================================

@router.post(
    "/reports",
    response_model=DamageReportResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_damage_report(
    payload: DamageReportCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> DamageReportResponse:
    require_role(
        current_user,
        READ_ROLES,
    )

    require_district_access(
        current_user,
        payload.district_id,
    )

    # Field mappings must match the actual DamageReport model.
    # Domain creation is delegated to its model/service
    # to preserve required workflow invariants.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Damage report creation requires mapping "
            "to the DamageReport domain model"
        ),
    )


# ============================================================
# LIST REPORTS
# ============================================================

@router.get(
    "/reports",
    response_model=PaginatedDamageReports,
)
async def list_damage_reports(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
    limit: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> PaginatedDamageReports:
    require_role(
        current_user,
        READ_ROLES,
    )

    query = select(DamageReport)
    count_query = select(func.count()).select_from(
        DamageReport
    )

    if district_id is not None:
        require_district_access(
            current_user,
            district_id,
        )

        query = query.where(
            DamageReport.district_id == district_id
        )

        count_query = count_query.where(
            DamageReport.district_id == district_id
        )

    elif (
        not getattr(current_user, "is_superuser", False)
        and user_role(current_user) not in ADMIN_ROLES
    ):
        assigned_district = getattr(
            current_user,
            "district_id",
            None,
        )

        if assigned_district is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="District assignment required",
            )

        query = query.where(
            DamageReport.district_id
            == int(assigned_district)
        )

        count_query = count_query.where(
            DamageReport.district_id
            == int(assigned_district)
        )

    total_result = await db.execute(count_query)

    result = await db.execute(
        query.order_by(
            DamageReport.created_at.desc()
        )
        .offset(offset)
        .limit(limit)
    )

    return PaginatedDamageReports(
        items=[
            report_response(report)
            for report in result.scalars().all()
        ],
        total=int(total_result.scalar_one()),
        limit=limit,
        offset=offset,
    )


# ============================================================
# GET REPORT
# ============================================================

@router.get(
    "/reports/{report_id}",
    response_model=DamageReportResponse,
)
async def get_damage_report(
    report_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> DamageReportResponse:
    require_role(current_user, READ_ROLES)

    report = await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    return report_response(report)


# ============================================================
# UPDATE REPORT
# ============================================================

@router.patch(
    "/reports/{report_id}",
    response_model=DamageReportResponse,
)
async def update_damage_report(
    report_id: str,
    payload: DamageReportUpdate,
    db: DbSession,
    current_user: CurrentUser,
) -> DamageReportResponse:
    require_role(
        current_user,
        READ_ROLES,
    )

    report = await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    ensure_editable(report)

    # Actual domain-field mapping is required.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Damage report update is not configured",
    )


# ============================================================
# SUBMIT REPORT
# ============================================================

@router.post(
    "/reports/{report_id}/submit",
)
async def submit_damage_report(
    report_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, READ_ROLES)

    report = await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    ensure_editable(report)

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Report submission workflow is not configured",
    )


# ============================================================
# ASSESSMENTS
# ============================================================

@router.post(
    "/reports/{report_id}/assessments",
    status_code=status.HTTP_201_CREATED,
)
async def create_damage_assessment(
    report_id: str,
    payload: DamageAssessmentCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        ADMIN_ROLES | ASSESSOR_ROLES,
    )

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Assessment creation requires integration "
            "with the DamageAssessment domain model"
        ),
    )


@router.get(
    "/reports/{report_id}/assessments",
)
async def list_damage_assessments(
    report_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    result = await db.execute(
        select(DamageAssessment).where(
            DamageAssessment.report_id
            == validate_uuid(report_id)
        )
    )

    return [
        {
            "id": str(item.id),
            "report_id": str(item.report_id),
        }
        for item in result.scalars().all()
    ]


# ============================================================
# EVIDENCE
# ============================================================

@router.post(
    "/reports/{report_id}/evidence",
    status_code=status.HTTP_201_CREATED,
)
async def add_damage_evidence(
    report_id: str,
    payload: DamageEvidenceCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, READ_ROLES)

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    # Upload validation, malware scanning, ownership
    # verification and immutable evidence storage
    # must happen before an evidence record is created.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Secure evidence ingestion is not configured",
    )


@router.get(
    "/reports/{report_id}/evidence",
)
async def list_damage_evidence(
    report_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    result = await db.execute(
        select(DamageEvidence).where(
            DamageEvidence.report_id
            == validate_uuid(report_id)
        )
    )

    return [
        {
            "id": str(item.id),
            "report_id": str(item.report_id),
        }
        for item in result.scalars().all()
    ]


# ============================================================
# COST ITEMS
# ============================================================

@router.post(
    "/reports/{report_id}/cost-items",
    status_code=status.HTTP_201_CREATED,
)
async def add_damage_cost_item(
    report_id: str,
    payload: DamageCostItemCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        ADMIN_ROLES | ASSESSOR_ROLES,
    )

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Cost-item domain mapping is not configured",
    )


@router.get(
    "/reports/{report_id}/cost-items",
)
async def list_damage_cost_items(
    report_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    result = await db.execute(
        select(DamageCostItem).where(
            DamageCostItem.report_id
            == validate_uuid(report_id)
        )
    )

    return [
        {
            "id": str(item.id),
            "report_id": str(item.report_id),
        }
        for item in result.scalars().all()
    ]


@router.get(
    "/reports/{report_id}/cost-summary",
    response_model=DamageCostSummary,
)
async def damage_cost_summary(
    report_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> DamageCostSummary:
    require_role(current_user, READ_ROLES)

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    # Monetary calculations must use Decimal and the
    # exact quantity/price fields defined by the model.
    # No fabricated or silently rounded financial totals.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Verified cost calculation is not configured",
    )


# ============================================================
# HUMAN REVIEW
# ============================================================

@router.post(
    "/reports/{report_id}/reviews",
    status_code=status.HTTP_201_CREATED,
)
async def review_damage_report(
    report_id: str,
    payload: DamageReviewRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        ADMIN_ROLES | REVIEWER_ROLES,
    )

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Human review requires the verified "
            "DamageReport workflow implementation"
        ),
    )


@router.get(
    "/reports/{report_id}/reviews",
)
async def list_damage_reviews(
    report_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    require_role(current_user, READ_ROLES)

    await get_report_or_404(
        db,
        report_id,
        current_user,
    )

    result = await db.execute(
        select(DamageReportReview).where(
            DamageReportReview.report_id
            == validate_uuid(report_id)
        )
    )

    return [
        {
            "id": str(item.id),
            "report_id": str(item.report_id),
        }
        for item in result.scalars().all()
    ]


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=DamageStatisticsResponse,
)
async def damage_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> DamageStatisticsResponse:
    require_role(current_user, READ_ROLES)

    if district_id is not None:
        require_district_access(
            current_user,
            district_id,
        )

    elif (
        not getattr(current_user, "is_superuser", False)
        and user_role(current_user) not in ADMIN_ROLES
    ):
        assigned_district = getattr(
            current_user,
            "district_id",
            None,
        )

        if assigned_district is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="District assignment required",
            )

        district_id = int(assigned_district)

    query = select(
        DamageReport.status,
        func.count(DamageReport.id),
    )

    if district_id is not None:
        query = query.where(
            DamageReport.district_id == district_id
        )

    query = query.group_by(
        DamageReport.status
    )

    result = await db.execute(query)

    by_status = {
        enum_value(row[0]): int(row[1])
        for row in result.all()
    }

    return DamageStatisticsResponse(
        total_reports=sum(by_status.values()),
        by_status=by_status,
        generated_at=datetime.now(timezone.utc),
    )


__all__ = [
    "router",
    "DamageReportCreate",
    "DamageReportUpdate",
    "DamageAssessmentCreate",
    "DamageEvidenceCreate",
    "DamageCostItemCreate",
    "DamageReviewRequest",
    "DamageReportResponse",
    "DamageStatisticsResponse",
]
