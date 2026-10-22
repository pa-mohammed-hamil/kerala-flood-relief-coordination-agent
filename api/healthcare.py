from __future__ import annotations

"""
backend/api/healthcare.py

Kerala Flood Relief Coordination Agent
Emergency Healthcare & Hospital Coordination API

Python 3.11+ | FastAPI | SQLAlchemy 2.x

Endpoints:
    GET  /healthcare/hospitals
    GET  /healthcare/statistics
    GET  /healthcare/hospitals/{hospital_id}
    GET  /healthcare/hospitals/{hospital_id}/beds
    POST /healthcare/hospitals/{hospital_id}/beds
    GET  /healthcare/hospitals/{hospital_id}/resources
    POST /healthcare/hospitals/{hospital_id}/resources
    GET  /healthcare/hospitals/{hospital_id}/status-history
    POST /healthcare/hospitals/{hospital_id}/status
    GET  /healthcare/transfers
    POST /healthcare/transfers
    GET  /healthcare/transfers/{transfer_id}
    POST /healthcare/transfers/{transfer_id}/review
    POST /healthcare/transfers/{transfer_id}/complete

Safety:
    - No autonomous medical diagnosis or treatment.
    - No automated ambulance dispatch.
    - No automatic transfer of patients.
    - Patient details are restricted.
    - Bed counts must come from verified hospital records.
    - Medical transfers require authorized human review.
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

from sqlalchemy import (
    func,
    select,
)

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session import get_db
from backend.dependencies import get_current_user

from backend.models.hospital import (
    Hospital,
    HospitalBedSnapshot,
    HospitalResourceSnapshot,
    HospitalPatientTransfer,
    HospitalStatusHistory,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/healthcare",
    tags=["Emergency Healthcare"],
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

MEDICAL_ROLES = {
    "medical_officer",
    "hospital_admin",
    "hospital_coordinator",
    "emergency_physician",
}

COORDINATION_ROLES = {
    "district_officer",
    "emergency_coordinator",
    "incident_commander",
}

READ_ROLES = (
    ADMIN_ROLES
    | MEDICAL_ROLES
    | COORDINATION_ROLES
    | {"operator", "healthcare_analyst"}
)

WRITE_ROLES = (
    ADMIN_ROLES
    | MEDICAL_ROLES
)

TRANSFER_REVIEW_ROLES = (
    ADMIN_ROLES
    | {
        "medical_officer",
        "hospital_coordinator",
        "emergency_physician",
    }
)

MAX_PAGE_SIZE = 100


# ============================================================
# ENUMERATIONS
# ============================================================

class HospitalOperationalStatus(str, Enum):
    OPERATIONAL = "operational"
    LIMITED_CAPACITY = "limited_capacity"
    OVER_CAPACITY = "over_capacity"
    EMERGENCY_ONLY = "emergency_only"
    EVACUATING = "evacuating"
    CLOSED = "closed"


class PatientTransferPriority(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


class PatientTransferStatus(str, Enum):
    REQUESTED = "requested"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_TRANSIT = "in_transit"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TransferReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class HospitalBedSnapshotCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime

    total_beds: int = Field(ge=0)

    occupied_beds: int = Field(ge=0)

    icu_total: int = Field(ge=0)

    icu_occupied: int = Field(ge=0)

    emergency_total: int = Field(ge=0)

    emergency_occupied: int = Field(ge=0)

    ventilators_available: int = Field(ge=0)

    source_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_capacity(
        self,
    ) -> "HospitalBedSnapshotCreate":
        if self.occupied_beds > self.total_beds:
            raise ValueError(
                "Occupied beds cannot exceed total beds"
            )

        if self.icu_occupied > self.icu_total:
            raise ValueError(
                "Occupied ICU beds cannot exceed ICU capacity"
            )

        if self.emergency_occupied > self.emergency_total:
            raise ValueError(
                "Occupied emergency beds exceed capacity"
            )

        return self


class HospitalResourceSnapshotCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime

    oxygen_cylinders_available: int = Field(ge=0)

    ventilators_available: int = Field(ge=0)

    ambulances_available: int = Field(ge=0)

    doctors_available: int = Field(ge=0)

    nurses_available: int = Field(ge=0)

    emergency_kits_available: int = Field(ge=0)

    drinking_water_liters: float = Field(ge=0)

    backup_power_available: bool

    source_reference: str | None = Field(
        default=None,
        max_length=500,
    )


class HospitalStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: HospitalOperationalStatus

    reason: str = Field(
        min_length=10,
        max_length=5000,
    )

    effective_at: datetime | None = None


class PatientTransferCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_hospital_id: str

    destination_hospital_id: str

    patient_reference: str = Field(
        min_length=5,
        max_length=150,
    )

    priority: PatientTransferPriority

    medical_requirements: str = Field(
        min_length=10,
        max_length=5000,
    )

    transport_requirements: str | None = Field(
        default=None,
        max_length=3000,
    )

    requested_at: datetime | None = None

    @field_validator(
        "source_hospital_id",
        "destination_hospital_id",
    )
    @classmethod
    def validate_hospital_uuid(
        cls,
        value: str,
    ) -> str:
        return str(uuid.UUID(value))

    @model_validator(mode="after")
    def validate_hospitals(
        self,
    ) -> "PatientTransferCreate":
        if (
            self.source_hospital_id
            == self.destination_hospital_id
        ):
            raise ValueError(
                "Source and destination hospitals must differ"
            )

        return self


class PatientTransferReviewRequest(BaseModel):
    action: TransferReviewAction

    review_notes: str = Field(
        min_length=10,
        max_length=5000,
    )

    destination_capacity_verified: bool = False

    medical_team_confirmed: bool = False

    transport_availability_verified: bool = False


class PatientTransferCompletionRequest(BaseModel):
    completed_at: datetime

    receiving_staff_reference: str = Field(
        min_length=3,
        max_length=150,
    )

    completion_notes: str = Field(
        min_length=10,
        max_length=5000,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class HospitalResponse(BaseModel):
    id: str
    district_id: int
    name: str | None = None
    status: str | None = None


class HospitalListResponse(BaseModel):
    items: list[HospitalResponse]
    total: int
    limit: int
    offset: int


class HospitalBedResponse(BaseModel):
    id: str
    hospital_id: str
    observed_at: datetime | None = None


class HospitalResourceResponse(BaseModel):
    id: str
    hospital_id: str
    observed_at: datetime | None = None


class HospitalStatusHistoryResponse(BaseModel):
    id: str
    hospital_id: str
    status: str | None = None


class PatientTransferResponse(BaseModel):
    id: str
    source_hospital_id: str | None = None
    destination_hospital_id: str | None = None
    status: str | None = None


class HospitalStatisticsResponse(BaseModel):
    total_hospitals: int
    by_status: dict[str, int]
    generated_at: datetime


# ============================================================
# HELPERS
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


async def get_hospital_or_404(
    db: AsyncSession,
    hospital_id: str,
    user: Any,
) -> Hospital:
    identifier = validate_uuid(hospital_id)

    result = await db.execute(
        select(Hospital).where(
            Hospital.id == identifier
        )
    )

    hospital = result.scalar_one_or_none()

    if hospital is None:
        raise HTTPException(
            status_code=404,
            detail="Hospital not found",
        )

    require_district_access(
        user,
        hospital.district_id,
    )

    return hospital


def hospital_response(
    hospital: Hospital,
) -> HospitalResponse:
    return HospitalResponse(
        id=str(hospital.id),
        district_id=hospital.district_id,
        name=getattr(
            hospital,
            "name",
            None,
        ),
        status=enum_value(
            getattr(
                hospital,
                "status",
                None,
            )
        ) or None,
    )


def transfer_response(
    transfer: HospitalPatientTransfer,
) -> PatientTransferResponse:
    return PatientTransferResponse(
        id=str(transfer.id),
        source_hospital_id=(
            str(transfer.source_hospital_id)
            if getattr(
                transfer,
                "source_hospital_id",
                None,
            )
            else None
        ),
        destination_hospital_id=(
            str(transfer.destination_hospital_id)
            if getattr(
                transfer,
                "destination_hospital_id",
                None,
            )
            else None
        ),
        status=enum_value(
            getattr(
                transfer,
                "status",
                None,
            )
        ) or None,
    )


# ============================================================
# LIST HOSPITALS
# ============================================================

@router.get(
    "/hospitals",
    response_model=HospitalListResponse,
)
async def list_hospitals(
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
) -> HospitalListResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(Hospital)

    count_query = (
        select(func.count())
        .select_from(Hospital)
    )

    if district_id is not None:
        query = query.where(
            Hospital.district_id == district_id
        )

        count_query = count_query.where(
            Hospital.district_id == district_id
        )

    total_result = await db.execute(
        count_query
    )

    result = await db.execute(
        query.order_by(Hospital.id)
        .offset(offset)
        .limit(limit)
    )

    return HospitalListResponse(
        items=[
            hospital_response(hospital)
            for hospital in result.scalars().all()
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
    response_model=HospitalStatisticsResponse,
)
async def hospital_statistics(
    db: DbSession,
    current_user: CurrentUser,
    district_id: int | None = Query(
        default=None,
        ge=1,
    ),
) -> HospitalStatisticsResponse:
    district_id = scoped_district(
        current_user,
        district_id,
    )

    query = select(
        Hospital.status,
        func.count(Hospital.id),
    )

    if district_id is not None:
        query = query.where(
            Hospital.district_id == district_id
        )

    query = query.group_by(Hospital.status)

    result = await db.execute(query)

    by_status = {
        enum_value(row[0]): int(row[1])
        for row in result.all()
    }

    return HospitalStatisticsResponse(
        total_hospitals=sum(by_status.values()),
        by_status=by_status,
        generated_at=utc_now(),
    )


# ============================================================
# HOSPITAL DETAILS
# ============================================================

@router.get(
    "/hospitals/{hospital_id}",
    response_model=HospitalResponse,
)
async def get_hospital(
    hospital_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> HospitalResponse:
    require_role(current_user, READ_ROLES)

    hospital = await get_hospital_or_404(
        db,
        hospital_id,
        current_user,
    )

    return hospital_response(hospital)


# ============================================================
# BED CAPACITY
# ============================================================

@router.get(
    "/hospitals/{hospital_id}/beds",
)
async def hospital_bed_history(
    hospital_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=30,
        ge=1,
        le=200,
    ),
) -> list[HospitalBedResponse]:
    require_role(current_user, READ_ROLES)

    await get_hospital_or_404(
        db,
        hospital_id,
        current_user,
    )

    result = await db.execute(
        select(HospitalBedSnapshot)
        .where(
            HospitalBedSnapshot.hospital_id
            == validate_uuid(hospital_id)
        )
        .order_by(
            HospitalBedSnapshot.observed_at.desc()
        )
        .limit(limit)
    )

    return [
        HospitalBedResponse(
            id=str(item.id),
            hospital_id=str(item.hospital_id),
            observed_at=getattr(
                item,
                "observed_at",
                None,
            ),
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/hospitals/{hospital_id}/beds",
    status_code=201,
)
async def create_bed_snapshot(
    hospital_id: str,
    payload: HospitalBedSnapshotCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_hospital_or_404(
        db,
        hospital_id,
        current_user,
    )

    # Validate source authority, snapshot freshness,
    # hospital ownership and ORM field mapping.
    raise HTTPException(
        status_code=501,
        detail="Hospital bed ingestion service is not configured",
    )


# ============================================================
# MEDICAL RESOURCES
# ============================================================

@router.get(
    "/hospitals/{hospital_id}/resources",
)
async def hospital_resource_history(
    hospital_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=30,
        ge=1,
        le=200,
    ),
) -> list[HospitalResourceResponse]:
    require_role(current_user, READ_ROLES)

    await get_hospital_or_404(
        db,
        hospital_id,
        current_user,
    )

    result = await db.execute(
        select(HospitalResourceSnapshot)
        .where(
            HospitalResourceSnapshot.hospital_id
            == validate_uuid(hospital_id)
        )
        .order_by(
            HospitalResourceSnapshot.observed_at.desc()
        )
        .limit(limit)
    )

    return [
        HospitalResourceResponse(
            id=str(item.id),
            hospital_id=str(item.hospital_id),
            observed_at=getattr(
                item,
                "observed_at",
                None,
            ),
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/hospitals/{hospital_id}/resources",
    status_code=201,
)
async def create_resource_snapshot(
    hospital_id: str,
    payload: HospitalResourceSnapshotCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_hospital_or_404(
        db,
        hospital_id,
        current_user,
    )

    raise HTTPException(
        status_code=501,
        detail="Medical resource ingestion service is not configured",
    )


# ============================================================
# HOSPITAL STATUS
# ============================================================

@router.get(
    "/hospitals/{hospital_id}/status-history",
)
async def hospital_status_history(
    hospital_id: str,
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=30,
        ge=1,
        le=200,
    ),
) -> list[HospitalStatusHistoryResponse]:
    require_role(current_user, READ_ROLES)

    await get_hospital_or_404(
        db,
        hospital_id,
        current_user,
    )

    result = await db.execute(
        select(HospitalStatusHistory)
        .where(
            HospitalStatusHistory.hospital_id
            == validate_uuid(hospital_id)
        )
        .limit(limit)
    )

    return [
        HospitalStatusHistoryResponse(
            id=str(item.id),
            hospital_id=str(item.hospital_id),
            status=enum_value(
                getattr(
                    item,
                    "status",
                    None,
                )
            ) or None,
        )
        for item in result.scalars().all()
    ]


@router.post(
    "/hospitals/{hospital_id}/status",
)
async def update_hospital_status(
    hospital_id: str,
    payload: HospitalStatusUpdate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(current_user, WRITE_ROLES)

    await get_hospital_or_404(
        db,
        hospital_id,
        current_user,
    )

    raise HTTPException(
        status_code=501,
        detail="Hospital status workflow is not configured",
    )


# ============================================================
# PATIENT TRANSFERS
# ============================================================

@router.get("/transfers")
async def list_patient_transfers(
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> list[PatientTransferResponse]:
    require_role(
        current_user,
        TRANSFER_REVIEW_ROLES,
    )

    # Transfers contain sensitive medical information.
    # Restrict this list to an authorized hospital/district
    # scope before enabling the endpoint.
    raise HTTPException(
        status_code=501,
        detail="Scoped patient transfer listing is not configured",
    )


@router.post(
    "/transfers",
    status_code=201,
)
async def create_patient_transfer(
    payload: PatientTransferCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        TRANSFER_REVIEW_ROLES,
    )

    # Must verify:
    # - Authorized source hospital
    # - Receiving hospital consent
    # - Available capacity
    # - Clinical review
    # - Transport readiness
    # - Patient privacy
    # - Audit logging
    raise HTTPException(
        status_code=501,
        detail="Patient transfer service is not configured",
    )


@router.get(
    "/transfers/{transfer_id}",
)
async def get_patient_transfer(
    transfer_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> PatientTransferResponse:
    require_role(
        current_user,
        TRANSFER_REVIEW_ROLES,
    )

    # Row-level hospital and patient authorization must
    # be implemented before revealing transfer details.
    raise HTTPException(
        status_code=501,
        detail="Patient transfer access policy is not configured",
    )


@router.post(
    "/transfers/{transfer_id}/review",
)
async def review_patient_transfer(
    transfer_id: str,
    payload: PatientTransferReviewRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        TRANSFER_REVIEW_ROLES,
    )

    if payload.action == TransferReviewAction.APPROVE:
        if not all(
            [
                payload.destination_capacity_verified,
                payload.medical_team_confirmed,
                payload.transport_availability_verified,
            ]
        ):
            raise HTTPException(
                status_code=422,
                detail="Transfer readiness checks incomplete",
            )

    raise HTTPException(
        status_code=501,
        detail="Medical transfer review workflow is not configured",
    )


@router.post(
    "/transfers/{transfer_id}/complete",
)
async def complete_patient_transfer(
    transfer_id: str,
    payload: PatientTransferCompletionRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    require_role(
        current_user,
        TRANSFER_REVIEW_ROLES,
    )

    raise HTTPException(
        status_code=501,
        detail="Medical transfer completion workflow is not configured",
    )


__all__ = [
    "router",
    "HospitalResponse",
    "HospitalListResponse",
    "HospitalBedSnapshotCreate",
    "HospitalResourceSnapshotCreate",
    "HospitalStatusUpdate",
    "PatientTransferCreate",
    "PatientTransferReviewRequest",
    "PatientTransferCompletionRequest",
    "PatientTransferResponse",
    "HospitalStatisticsResponse",
]
