from __future__ import annotations

"""
backend/services/report_service.py

Kerala Flood Relief Coordination Agent
=======================================

Reporting and Analytics Service

Features:
- District flood situation reports
- Incident and SOS reporting
- Rescue and evacuation reporting
- Shelter occupancy reporting
- Relief inventory and distribution reporting
- Medical emergency coordination reporting
- Volunteer activity reporting
- Damage and recovery reporting
- Executive situation summaries
- Daily and custom date-range reports
- KPI calculations
- Data completeness and quality checks
- Human review and approval
- Role-based report access
- Sensitive-data minimization
- JSON and CSV export
- Report integrity verification
- Audit logging
- Version-controlled report storage

SAFETY:
- Reports contain operational records, not verified
  government statistics unless source verification exists.
- Never infer completed rescue or delivery from assignments.
- Public reports must use approved aggregated datasets.
- No patient names, personal contact details, exact SOS
  coordinates or sensitive case descriptions are exported.
- Production deployments require tenant/district isolation,
  transactional snapshots, encrypted storage, retention
  controls and verified data-source integrations.
"""


import csv
import hashlib
import io
import json
import logging
import math
import uuid

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class ReportType(str, Enum):
    DAILY_SITUATION = "daily_situation"
    DISTRICT_OVERVIEW = "district_overview"
    INCIDENT = "incident"
    SOS = "sos"
    FLOOD_FORECAST = "flood_forecast"
    RESCUE = "rescue"
    EVACUATION = "evacuation"
    SHELTER = "shelter"
    RELIEF = "relief"
    INVENTORY = "inventory"
    MEDICAL = "medical"
    VOLUNTEER = "volunteer"
    DAMAGE = "damage"
    RECOVERY = "recovery"
    COMPREHENSIVE = "comprehensive"


class ReportStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ReportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"


class ReportVisibility(str, Enum):
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    PUBLIC = "public"


class ReportPriority(str, Enum):
    ROUTINE = "routine"
    HIGH = "high"
    CRITICAL = "critical"


class DataQualityLevel(str, Enum):
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"
    UNKNOWN = "unknown"


# ============================================================
# EXCEPTIONS
# ============================================================

class ReportServiceError(Exception):
    pass


class ReportValidationError(ReportServiceError):
    pass


class ReportNotFoundError(ReportServiceError):
    pass


class ReportConflictError(ReportServiceError):
    pass


class ReportAuthorizationError(ReportServiceError):
    pass


class ReportDataError(ReportServiceError):
    pass


class ReportExportError(ReportServiceError):
    pass


class ReportIntegrityError(ReportServiceError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_text(
    value: str,
    name: str,
    *,
    max_length: int = 5000,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportValidationError(
            f"{name} is required"
        )

    result = value.strip()

    if len(result) > max_length:
        raise ReportValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ReportValidationError(
            f"{name} must be timezone-aware"
        )


def safe_percentage(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator <= 0:
        return None

    return round(
        numerator / denominator * 100,
        2,
    )


def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator <= 0:
        return None

    return round(
        numerator / denominator,
        4,
    )


def canonical_json(
    data: Mapping[str, Any],
) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=json_serializer,
    )


def json_serializer(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    raise TypeError(
        f"Unsupported JSON type: {type(value).__name__}"
    )


def content_sha256(
    data: Mapping[str, Any],
) -> str:
    encoded = canonical_json(data).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def sanitize_csv_cell(value: Any) -> str:
    """
    Prevent spreadsheet formula injection when CSV
    files are opened in spreadsheet applications.
    """
    if value is None:
        return ""

    result = str(value)

    if result.lstrip().startswith(
        ("=", "+", "-", "@", "\t", "\r", "\n")
    ):
        result = "'" + result

    return result


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ReportPeriod:
    start_at: datetime
    end_at: datetime

    def __post_init__(self) -> None:
        validate_datetime(
            self.start_at,
            "start_at",
        )

        validate_datetime(
            self.end_at,
            "end_at",
        )

        if self.end_at <= self.start_at:
            raise ReportValidationError(
                "Report end time must follow start time"
            )

        if (
            self.end_at - self.start_at
            > timedelta(days=366)
        ):
            raise ReportValidationError(
                "Report period cannot exceed 366 days"
            )


@dataclass(frozen=True)
class ReportCreateRequest:
    district_id: int
    report_type: ReportType

    period: ReportPeriod

    title: str

    visibility: ReportVisibility = (
        ReportVisibility.INTERNAL
    )

    priority: ReportPriority = (
        ReportPriority.ROUTINE
    )

    include_sections: tuple[str, ...] = ()

    description: str = ""

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ReportMetric:
    name: str
    value: int | float | str | None

    unit: str = ""
    description: str = ""

    source: str = ""
    verified: bool = False


@dataclass(frozen=True)
class ReportSection:
    section_id: str
    title: str

    metrics: tuple[ReportMetric, ...] = ()

    summary: str = ""

    data: Mapping[str, Any] = field(
        default_factory=dict
    )

    source_name: str = ""
    source_verified: bool = False

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class DataQualityAssessment:
    level: DataQualityLevel

    completeness_percent: float

    missing_sections: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass
class GeneratedReport:
    id: str
    district_id: int

    report_type: ReportType
    title: str
    description: str

    period: ReportPeriod

    visibility: ReportVisibility
    priority: ReportPriority

    status: ReportStatus

    created_by: int

    sections: list[ReportSection] = field(
        default_factory=list
    )

    quality: DataQualityAssessment | None = None

    executive_summary: str = ""

    content_hash: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    generated_at: datetime | None = None

    reviewed_by: int | None = None
    reviewed_at: datetime | None = None

    approved_by: int | None = None
    approved_at: datetime | None = None

    rejection_reason: str | None = None

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ReportExport:
    report_id: str
    format: ReportFormat

    filename: str
    content_type: str

    content: bytes

    checksum_sha256: str

    exported_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# DATA SOURCE CONTRACTS
# ============================================================

class ReportDataGateway(Protocol):
    """
    Each source must return aggregated, district-scoped
    information for the requested time interval.

    The gateway is responsible for:
    - Enforcing district isolation
    - Applying event-time filters
    - Avoiding duplicate counts
    - Returning source verification metadata
    - Excluding direct personal identifiers

    Expected payload:
    {
        "metrics": {
            "total_incidents": 10,
            "resolved_incidents": 4
        },
        "verified": True,
        "source": "incident_repository",
        "summary": "..."
    }
    """

    async def get_section_data(
        self,
        *,
        district_id: int,
        section_id: str,
        period: ReportPeriod,
    ) -> Mapping[str, Any]:
        ...


class ReportRepository(Protocol):

    async def create(
        self,
        report: GeneratedReport,
    ) -> GeneratedReport:
        ...

    async def get_by_id(
        self,
        report_id: str,
    ) -> GeneratedReport | None:
        ...

    async def update(
        self,
        report: GeneratedReport,
        *,
        expected_version: int,
    ) -> GeneratedReport:
        """
        Must enforce atomic optimistic concurrency.
        """
        ...

    async def list_by_district(
        self,
        district_id: int,
    ) -> Sequence[GeneratedReport]:
        ...

    async def list_by_type(
        self,
        *,
        district_id: int,
        report_type: ReportType,
    ) -> Sequence[GeneratedReport]:
        ...


class ReportAuthorizationGateway(Protocol):

    async def can_create(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_review(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_approve(
        self,
        *,
        user_id: int,
        district_id: int,
    ) -> bool:
        ...

    async def can_view(
        self,
        *,
        user_id: int,
        district_id: int,
        visibility: ReportVisibility,
    ) -> bool:
        ...

    async def can_export(
        self,
        *,
        user_id: int,
        district_id: int,
        visibility: ReportVisibility,
    ) -> bool:
        ...


class ReportAuditGateway(Protocol):

    async def record(
        self,
        action: str,
        *,
        actor_id: int | None,
        entity_type: str,
        entity_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...


# ============================================================
# SECTION REGISTRY
# ============================================================

class ReportSectionRegistry:

    SECTION_TITLES = {
        "incidents": "Flood Incidents",
        "sos": "Emergency SOS Requests",
        "forecasts": "Flood Forecasts",
        "rescue": "Rescue Operations",
        "evacuation": "Evacuation Operations",
        "shelters": "Shelter Management",
        "relief": "Relief Distribution",
        "inventory": "Relief Inventory",
        "medical": "Medical Coordination",
        "volunteers": "Volunteer Operations",
        "damage": "Damage Assessment",
        "recovery": "Recovery and Rehabilitation",
    }

    TYPE_SECTIONS = {
        ReportType.DAILY_SITUATION: (
            "incidents",
            "sos",
            "forecasts",
            "rescue",
            "evacuation",
            "shelters",
            "relief",
            "medical",
        ),
        ReportType.DISTRICT_OVERVIEW: (
            "incidents",
            "sos",
            "rescue",
            "evacuation",
            "shelters",
            "relief",
            "inventory",
            "medical",
            "volunteers",
            "damage",
            "recovery",
        ),
        ReportType.INCIDENT: ("incidents",),
        ReportType.SOS: ("sos",),
        ReportType.FLOOD_FORECAST: ("forecasts",),
        ReportType.RESCUE: ("rescue",),
        ReportType.EVACUATION: ("evacuation",),
        ReportType.SHELTER: ("shelters",),
        ReportType.RELIEF: ("relief",),
        ReportType.INVENTORY: ("inventory",),
        ReportType.MEDICAL: ("medical",),
        ReportType.VOLUNTEER: ("volunteers",),
        ReportType.DAMAGE: ("damage",),
        ReportType.RECOVERY: ("recovery",),
        ReportType.COMPREHENSIVE: tuple(
            SECTION_TITLES.keys()
        ) if False else (
            "incidents",
            "sos",
            "forecasts",
            "rescue",
            "evacuation",
            "shelters",
            "relief",
            "inventory",
            "medical",
            "volunteers",
            "damage",
            "recovery",
        ),
    }

    def get_sections(
        self,
        report_type: ReportType,
        custom_sections: Sequence[str] = (),
    ) -> tuple[str, ...]:

        if custom_sections:
            invalid = [
                section
                for section in custom_sections
                if section not in self.SECTION_TITLES
            ]

            if invalid:
                raise ReportValidationError(
                    f"Unknown report sections: {invalid}"
                )

            return tuple(
                dict.fromkeys(custom_sections)
            )

        return self.TYPE_SECTIONS[report_type]

    def get_title(
        self,
        section_id: str,
    ) -> str:
        return self.SECTION_TITLES.get(
            section_id,
            section_id.replace("_", " ").title(),
        )


# ============================================================
# DATA QUALITY EVALUATOR
# ============================================================

class ReportDataQualityEvaluator:

    def evaluate(
        self,
        *,
        expected_sections: Sequence[str],
        actual_sections: Sequence[ReportSection],
    ) -> DataQualityAssessment:

        actual_map = {
            section.section_id: section
            for section in actual_sections
        }

        missing = [
            section_id
            for section_id in expected_sections
            if section_id not in actual_map
        ]

        warnings: list[str] = []

        for section in actual_sections:
            if not section.source_verified:
                warnings.append(
                    f"Unverified source: {section.section_id}"
                )

            if not section.metrics:
                warnings.append(
                    f"No metrics: {section.section_id}"
                )

        completeness = (
            (
                len(expected_sections) - len(missing)
            )
            / len(expected_sections)
            * 100
            if expected_sections
            else 100.0
        )

        verified_count = sum(
            section.source_verified
            for section in actual_sections
        )

        if not actual_sections:
            level = DataQualityLevel.UNKNOWN

        elif (
            not missing
            and verified_count == len(actual_sections)
            and all(
                section.metrics
                for section in actual_sections
            )
        ):
            level = DataQualityLevel.GOOD

        elif completeness >= 60:
            level = DataQualityLevel.PARTIAL

        else:
            level = DataQualityLevel.POOR

        return DataQualityAssessment(
            level=level,
            completeness_percent=round(
                completeness,
                2,
            ),
            missing_sections=tuple(missing),
            warnings=tuple(warnings),
        )


# ============================================================
# KPI ENGINE
# ============================================================

class ReportKPIEngine:
    """
    Computes KPIs from explicit source-provided counts.

    Does not assume that assignments represent
    successful field operations.
    """

    @staticmethod
    def _count(
        metrics: Mapping[str, Any],
        name: str,
    ) -> int | None:
        value = metrics.get(name)

        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            return None

        return value

    def calculate(
        self,
        section_id: str,
        metrics: Mapping[str, Any],
    ) -> tuple[ReportMetric, ...]:

        results: list[ReportMetric] = []

        def add_count(
            name: str,
            description: str,
        ) -> None:
            value = self._count(
                metrics,
                name,
            )

            if value is not None:
                results.append(
                    ReportMetric(
                        name=name,
                        value=value,
                        unit="count",
                        description=description,
                    )
                )

        def add_percentage(
            name: str,
            numerator_key: str,
            denominator_key: str,
        ) -> None:
            numerator = self._count(
                metrics,
                numerator_key,
            )

            denominator = self._count(
                metrics,
                denominator_key,
            )

            if (
                numerator is None
                or denominator is None
                or numerator > denominator
            ):
                return

            value = safe_percentage(
                numerator,
                denominator,
            )

            if value is not None:
                results.append(
                    ReportMetric(
                        name=name,
                        value=value,
                        unit="percent",
                        description=(
                            f"{numerator_key} / "
                            f"{denominator_key}"
                        ),
                    )
                )

        if section_id == "incidents":
            add_count(
                "total_incidents",
                "Reported incidents",
            )
            add_count(
                "active_incidents",
                "Active incidents",
            )
            add_count(
                "resolved_incidents",
                "Resolved incidents",
            )

            add_percentage(
                "incident_resolution_rate",
                "resolved_incidents",
                "total_incidents",
            )

        elif section_id == "sos":
            add_count(
                "total_sos_requests",
                "SOS requests received",
            )
            add_count(
                "acknowledged_sos_requests",
                "Acknowledged SOS requests",
            )
            add_count(
                "resolved_sos_requests",
                "Resolved SOS requests",
            )

            add_percentage(
                "sos_acknowledgement_rate",
                "acknowledged_sos_requests",
                "total_sos_requests",
            )

        elif section_id == "rescue":
            add_count(
                "total_rescue_missions",
                "Recorded rescue missions",
            )
            add_count(
                "completed_rescue_missions",
                "Completed rescue missions",
            )
            add_count(
                "people_rescued",
                "People confirmed rescued",
            )

            add_percentage(
                "rescue_completion_rate",
                "completed_rescue_missions",
                "total_rescue_missions",
            )

        elif section_id == "evacuation":
            add_count(
                "planned_evacuations",
                "Planned evacuations",
            )
            add_count(
                "completed_evacuations",
                "Completed evacuations",
            )
            add_count(
                "people_evacuated",
                "People confirmed evacuated",
            )

        elif section_id == "shelters":
            add_count(
                "total_shelters",
                "Registered shelters",
            )
            add_count(
                "operational_shelters",
                "Operational shelters",
            )
            add_count(
                "total_capacity",
                "Verified shelter capacity",
            )
            add_count(
                "current_occupancy",
                "Current verified occupancy",
            )

            add_percentage(
                "shelter_occupancy_rate",
                "current_occupancy",
                "total_capacity",
            )

        elif section_id == "relief":
            add_count(
                "total_deliveries",
                "Relief deliveries recorded",
            )
            add_count(
                "completed_deliveries",
                "Confirmed relief deliveries",
            )

            add_percentage(
                "relief_delivery_completion_rate",
                "completed_deliveries",
                "total_deliveries",
            )

        elif section_id == "inventory":
            add_count(
                "inventory_items",
                "Tracked inventory item types",
            )
            add_count(
                "low_stock_items",
                "Low-stock item types",
            )

        elif section_id == "medical":
            add_count(
                "total_medical_cases",
                "Medical coordination cases",
            )
            add_count(
                "completed_transfers",
                "Completed medical transfers",
            )
            add_count(
                "critical_medical_cases",
                "Critical medical cases",
            )

        elif section_id == "volunteers":
            add_count(
                "total_volunteers",
                "Registered volunteers",
            )
            add_count(
                "active_volunteers",
                "Active volunteers",
            )

        elif section_id == "damage":
            add_count(
                "total_damage_reports",
                "Damage reports received",
            )
            add_count(
                "verified_damage_reports",
                "Verified damage reports",
            )

        elif section_id == "recovery":
            add_count(
                "total_recovery_projects",
                "Recovery projects",
            )
            add_count(
                "completed_recovery_projects",
                "Completed recovery projects",
            )

            add_percentage(
                "recovery_completion_rate",
                "completed_recovery_projects",
                "total_recovery_projects",
            )

        elif section_id == "forecasts":
            add_count(
                "total_forecasts",
                "Generated flood forecasts",
            )
            add_count(
                "verified_forecasts",
                "Verified flood forecasts",
            )

        return tuple(results)


# ============================================================
# EXECUTIVE SUMMARY GENERATOR
# ============================================================

class ExecutiveSummaryGenerator:
    """
    Deterministic summary generator.

    No LLM-generated facts or unsupported
    causal conclusions are introduced.
    """

    def generate(
        self,
        *,
        district_id: int,
        sections: Sequence[ReportSection],
        quality: DataQualityAssessment,
    ) -> str:

        parts = [
            f"District {district_id} operational report."
        ]

        for section in sections:
            metric_parts = [
                f"{metric.name.replace('_', ' ')}: "
                f"{metric.value}"
                for metric in section.metrics
                if metric.value is not None
            ]

            if metric_parts:
                parts.append(
                    f"{section.title}: "
                    + "; ".join(metric_parts[:5])
                    + "."
                )

        parts.append(
            "Data quality: "
            f"{quality.level.value}; "
            "section completeness: "
            f"{quality.completeness_percent}%."
        )

        if quality.missing_sections:
            parts.append(
                "Missing sections: "
                + ", ".join(
                    quality.missing_sections
                )
                + "."
            )

        if quality.warnings:
            parts.append(
                "Some source data requires verification."
            )

        return " ".join(parts)


# ============================================================
# REPORT SERVICE
# ============================================================

class ReportService:

    def __init__(
        self,
        *,
        repository: ReportRepository,
        data_gateway: ReportDataGateway,
        authorization_gateway: ReportAuthorizationGateway,
        audit_gateway: ReportAuditGateway | None = None,
        section_registry: ReportSectionRegistry | None = None,
        quality_evaluator: (
            ReportDataQualityEvaluator | None
        ) = None,
        kpi_engine: ReportKPIEngine | None = None,
        summary_generator: (
            ExecutiveSummaryGenerator | None
        ) = None,
    ):
        self.repository = repository
        self.data_gateway = data_gateway
        self.authorization = authorization_gateway
        self.audit = audit_gateway

        self.registry = (
            section_registry
            or ReportSectionRegistry()
        )

        self.quality_evaluator = (
            quality_evaluator
            or ReportDataQualityEvaluator()
        )

        self.kpi_engine = (
            kpi_engine
            or ReportKPIEngine()
        )

        self.summary_generator = (
            summary_generator
            or ExecutiveSummaryGenerator()
        )

    async def _audit(
        self,
        action: str,
        report: GeneratedReport,
        actor_id: int | None,
    ) -> None:

        if self.audit is None:
            return

        await self.audit.record(
            action,
            actor_id=actor_id,
            entity_type="generated_report",
            entity_id=report.id,
            metadata={
                "district_id": report.district_id,
                "report_type": report.report_type.value,
                "status": report.status.value,
                "visibility": report.visibility.value,
                "version": report.version,
            },
        )

    async def _save(
        self,
        original: GeneratedReport,
        updated: GeneratedReport,
        *,
        action: str,
        actor_id: int,
    ) -> GeneratedReport:

        saved = await self.repository.update(
            updated,
            expected_version=original.version,
        )

        await self._audit(
            action,
            saved,
            actor_id,
        )

        return saved

    async def get_report(
        self,
        report_id: str,
    ) -> GeneratedReport:

        report = await self.repository.get_by_id(
            report_id
        )

        if report is None:
            raise ReportNotFoundError(
                "Report not found"
            )

        return report

    async def create_report(
        self,
        request: ReportCreateRequest,
        *,
        actor_id: int,
    ) -> GeneratedReport:

        if request.district_id <= 0:
            raise ReportValidationError(
                "Invalid district ID"
            )

        if not await self.authorization.can_create(
            user_id=actor_id,
            district_id=request.district_id,
        ):
            raise ReportAuthorizationError(
                "Report creation permission denied"
            )

        title = require_text(
            request.title,
            "Report title",
            max_length=200,
        )

        sections = self.registry.get_sections(
            request.report_type,
            request.include_sections,
        )

        report = GeneratedReport(
            id=str(uuid.uuid4()),
            district_id=request.district_id,
            report_type=request.report_type,
            title=title,
            description=request.description.strip(),
            period=request.period,
            visibility=request.visibility,
            priority=request.priority,
            status=ReportStatus.DRAFT,
            created_by=actor_id,
            metadata={
                **request.metadata,
                "requested_sections": list(sections),
            },
        )

        created = await self.repository.create(
            report
        )

        await self._audit(
            "report.created",
            created,
            actor_id,
        )

        return created

    async def _generate_section(
        self,
        *,
        report: GeneratedReport,
        section_id: str,
    ) -> ReportSection:

        raw = await self.data_gateway.get_section_data(
            district_id=report.district_id,
            section_id=section_id,
            period=report.period,
        )

        if not isinstance(raw, Mapping):
            raise ReportDataError(
                f"Invalid source payload: {section_id}"
            )

        metrics_data = raw.get(
            "metrics",
            {},
        )

        if not isinstance(metrics_data, Mapping):
            raise ReportDataError(
                f"Invalid metrics payload: {section_id}"
            )

        source_verified = (
            raw.get("verified") is True
        )

        source_name = str(
            raw.get("source", "")
        )

        calculated_metrics = (
            self.kpi_engine.calculate(
                section_id,
                metrics_data,
            )
        )

        metrics = tuple(
            replace(
                metric,
                source=source_name,
                verified=source_verified,
            )
            for metric in calculated_metrics
        )

        return ReportSection(
            section_id=section_id,
            title=self.registry.get_title(
                section_id
            ),
            metrics=metrics,
            summary=str(
                raw.get("summary", "")
            )[:3000],
            data={
                "metrics": dict(metrics_data),
            },
            source_name=source_name,
            source_verified=source_verified,
        )

    @staticmethod
    def _content_payload(
        report: GeneratedReport,
    ) -> dict[str, Any]:

        return {
            "report_id": report.id,
            "district_id": report.district_id,
            "report_type": report.report_type.value,
            "title": report.title,
            "period": {
                "start_at": (
                    report.period.start_at.isoformat()
                ),
                "end_at": (
                    report.period.end_at.isoformat()
                ),
            },
            "executive_summary": (
                report.executive_summary
            ),
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "summary": section.summary,
                    "source_name": section.source_name,
                    "source_verified": (
                        section.source_verified
                    ),
                    "metrics": [
                        {
                            "name": metric.name,
                            "value": metric.value,
                            "unit": metric.unit,
                            "source": metric.source,
                            "verified": metric.verified,
                        }
                        for metric in section.metrics
                    ],
                }
                for section in report.sections
            ],
            "quality": (
                {
                    "level": report.quality.level.value,
                    "completeness_percent": (
                        report.quality
                        .completeness_percent
                    ),
                    "missing_sections": list(
                        report.quality.missing_sections
                    ),
                    "warnings": list(
                        report.quality.warnings
                    ),
                }
                if report.quality
                else None
            ),
        }

    async def generate_report(
        self,
        report_id: str,
        *,
        actor_id: int,
    ) -> GeneratedReport:

        report = await self.get_report(
            report_id
        )

        if not await self.authorization.can_create(
            user_id=actor_id,
            district_id=report.district_id,
        ):
            raise ReportAuthorizationError(
                "Report generation permission denied"
            )

        if report.status != ReportStatus.DRAFT:
            raise ReportConflictError(
                "Only draft reports can be generated"
            )

        expected_sections = (
            self.registry.get_sections(
                report.report_type,
                report.metadata.get(
                    "requested_sections",
                    (),
                ),
            )
        )

        sections: list[ReportSection] = []

        for section_id in expected_sections:
            try:
                section = await self._generate_section(
                    report=report,
                    section_id=section_id,
                )

                sections.append(
                    section
                )

            except Exception:
                logger.exception(
                    "Report section generation failed: %s",
                    section_id,
                )

        quality = self.quality_evaluator.evaluate(
            expected_sections=expected_sections,
            actual_sections=sections,
        )

        summary = self.summary_generator.generate(
            district_id=report.district_id,
            sections=sections,
            quality=quality,
        )

        updated = replace(
            report,
            sections=sections,
            quality=quality,
            executive_summary=summary,
            generated_at=utc_now(),
            updated_at=utc_now(),
            version=report.version + 1,
        )

        updated.content_hash = content_sha256(
            self._content_payload(updated)
        )

        return await self._save(
            report,
            updated,
            action="report.generated",
            actor_id=actor_id,
        )

    async def verify_integrity(
        self,
        report_id: str,
    ) -> bool:

        report = await self.get_report(
            report_id
        )

        if not report.content_hash:
            return False

        actual_hash = content_sha256(
            self._content_payload(report)
        )

        return actual_hash == report.content_hash

    async def submit_for_review(
        self,
        report_id: str,
        *,
        actor_id: int,
    ) -> GeneratedReport:

        report = await self.get_report(
            report_id
        )

        if not await self.authorization.can_create(
            user_id=actor_id,
            district_id=report.district_id,
        ):
            raise ReportAuthorizationError(
                "Report submission denied"
            )

        if report.status != ReportStatus.DRAFT:
            raise ReportConflictError(
                "Only draft reports can be submitted"
            )

        if not report.sections:
            raise ReportValidationError(
                "Report has no generated sections"
            )

        if not await self.verify_integrity(
            report_id
        ):
            raise ReportIntegrityError(
                "Report integrity verification failed"
            )

        updated = replace(
            report,
            status=ReportStatus.PENDING_REVIEW,
            updated_at=utc_now(),
            version=report.version + 1,
        )

        return await self._save(
            report,
            updated,
            action="report.submitted_for_review",
            actor_id=actor_id,
        )

    async def approve_report(
        self,
        report_id: str,
        *,
        reviewer_id: int,
    ) -> GeneratedReport:

        report = await self.get_report(
            report_id
        )

        if not await self.authorization.can_approve(
            user_id=reviewer_id,
            district_id=report.district_id,
        ):
            raise ReportAuthorizationError(
                "Report approval denied"
            )

        if report.status != ReportStatus.PENDING_REVIEW:
            raise ReportConflictError(
                "Report is not awaiting review"
            )

        if reviewer_id == report.created_by:
            raise ReportAuthorizationError(
                "Independent reviewer required"
            )

        if not await self.verify_integrity(
            report_id
        ):
            raise ReportIntegrityError(
                "Report content has changed"
            )

        if report.quality is None:
            raise ReportValidationError(
                "Report quality assessment missing"
            )

        if report.quality.level in {
            DataQualityLevel.POOR,
            DataQualityLevel.UNKNOWN,
        }:
            raise ReportDataError(
                "Report data quality is insufficient"
            )

        if report.visibility == ReportVisibility.PUBLIC:
            if report.quality.level != DataQualityLevel.GOOD:
                raise ReportDataError(
                    "Public reports require complete, "
                    "verified source data"
                )

            # A separate privacy disclosure review
            # is required before public publication.
            if report.metadata.get(
                "privacy_review_approved"
            ) is not True:
                raise ReportAuthorizationError(
                    "Public privacy review is required"
                )

        updated = replace(
            report,
            status=ReportStatus.APPROVED,
            reviewed_by=reviewer_id,
            reviewed_at=utc_now(),
            approved_by=reviewer_id,
            approved_at=utc_now(),
            updated_at=utc_now(),
            version=report.version + 1,
        )

        return await self._save(
            report,
            updated,
            action="report.approved",
            actor_id=reviewer_id,
        )

    async def reject_report(
        self,
        report_id: str,
        *,
        reviewer_id: int,
        reason: str,
    ) -> GeneratedReport:

        report = await self.get_report(
            report_id
        )

        if not await self.authorization.can_review(
            user_id=reviewer_id,
            district_id=report.district_id,
        ):
            raise ReportAuthorizationError(
                "Report rejection permission denied"
            )

        if report.status != ReportStatus.PENDING_REVIEW:
            raise ReportConflictError(
                "Report is not awaiting review"
            )

        updated = replace(
            report,
            status=ReportStatus.REJECTED,
            reviewed_by=reviewer_id,
            reviewed_at=utc_now(),
            rejection_reason=require_text(
                reason,
                "Rejection reason",
            ),
            updated_at=utc_now(),
            version=report.version + 1,
        )

        return await self._save(
            report,
            updated,
            action="report.rejected",
            actor_id=reviewer_id,
        )

    async def archive_report(
        self,
        report_id: str,
        *,
        actor_id: int,
    ) -> GeneratedReport:

        report = await self.get_report(
            report_id
        )

        if not await self.authorization.can_approve(
            user_id=actor_id,
            district_id=report.district_id,
        ):
            raise ReportAuthorizationError(
                "Report archive permission denied"
            )

        if report.status != ReportStatus.APPROVED:
            raise ReportConflictError(
                "Only approved reports can be archived"
            )

        updated = replace(
            report,
            status=ReportStatus.ARCHIVED,
            updated_at=utc_now(),
            version=report.version + 1,
        )

        return await self._save(
            report,
            updated,
            action="report.archived",
            actor_id=actor_id,
        )

    async def view_report(
        self,
        report_id: str,
        *,
        actor_id: int,
    ) -> GeneratedReport:

        report = await self.get_report(
            report_id
        )

        if not await self.authorization.can_view(
            user_id=actor_id,
            district_id=report.district_id,
            visibility=report.visibility,
        ):
            raise ReportAuthorizationError(
                "Report access denied"
            )

        return report

    async def export_report(
        self,
        report_id: str,
        *,
        actor_id: int,
        format: ReportFormat = ReportFormat.JSON,
    ) -> ReportExport:

        report = await self.get_report(
            report_id
        )

        if not await self.authorization.can_export(
            user_id=actor_id,
            district_id=report.district_id,
            visibility=report.visibility,
        ):
            raise ReportAuthorizationError(
                "Report export permission denied"
            )

        if report.status not in {
            ReportStatus.APPROVED,
            ReportStatus.ARCHIVED,
        }:
            raise ReportConflictError(
                "Only approved reports can be exported"
            )

        if not await self.verify_integrity(
            report_id
        ):
            raise ReportIntegrityError(
                "Report integrity verification failed"
            )

        payload = self._content_payload(
            report
        )

        if format == ReportFormat.JSON:
            content = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=json_serializer,
            ).encode("utf-8")

            filename = f"report_{report.id}.json"
            content_type = "application/json"

        elif format == ReportFormat.CSV:
            buffer = io.StringIO(
                newline=""
            )

            writer = csv.writer(
                buffer
            )

            writer.writerow([
                "report_id",
                "district_id",
                "section",
                "metric",
                "value",
                "unit",
                "source",
                "verified",
            ])

            for section in report.sections:
                for metric in section.metrics:
                    writer.writerow([
                        sanitize_csv_cell(report.id),
                        report.district_id,
                        sanitize_csv_cell(
                            section.section_id
                        ),
                        sanitize_csv_cell(
                            metric.name
                        ),
                        sanitize_csv_cell(
                            metric.value
                        ),
                        sanitize_csv_cell(
                            metric.unit
                        ),
                        sanitize_csv_cell(
                            metric.source
                        ),
                        metric.verified,
                    ])

            content = buffer.getvalue().encode(
                "utf-8-sig"
            )

            filename = f"report_{report.id}.csv"
            content_type = "text/csv"

        else:
            raise ReportExportError(
                "Unsupported report format"
            )

        checksum = hashlib.sha256(
            content
        ).hexdigest()

        exported = ReportExport(
            report_id=report.id,
            format=format,
            filename=filename,
            content_type=content_type,
            content=content,
            checksum_sha256=checksum,
        )

        await self._audit(
            "report.exported",
            report,
            actor_id,
        )

        return exported

    async def list_district_reports(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> list[GeneratedReport]:

        if district_id <= 0:
            raise ReportValidationError(
                "Invalid district ID"
            )

        reports = await self.repository.list_by_district(
            district_id
        )

        visible: list[GeneratedReport] = []

        for report in reports:
            if await self.authorization.can_view(
                user_id=actor_id,
                district_id=report.district_id,
                visibility=report.visibility,
            ):
                visible.append(report)

        return visible

    async def get_report_statistics(
        self,
        district_id: int,
        *,
        actor_id: int,
    ) -> dict[str, Any]:

        reports = await self.list_district_reports(
            district_id,
            actor_id=actor_id,
        )

        status_counts = {
            status.value: 0
            for status in ReportStatus
        }

        type_counts = {
            report_type.value: 0
            for report_type in ReportType
        }

        quality_counts = {
            quality.value: 0
            for quality in DataQualityLevel
        }

        for report in reports:
            status_counts[
                report.status.value
            ] += 1

            type_counts[
                report.report_type.value
            ] += 1

            if report.quality is not None:
                quality_counts[
                    report.quality.level.value
                ] += 1

        return {
            "district_id": district_id,
            "visible_reports": len(reports),
            "approved_reports": status_counts[
                ReportStatus.APPROVED.value
            ],
            "pending_review": status_counts[
                ReportStatus.PENDING_REVIEW.value
            ],
            "status_counts": status_counts,
            "type_counts": type_counts,
            "quality_counts": quality_counts,
            "generated_at": utc_now().isoformat(),
        }

    async def generate_daily_report(
        self,
        *,
        district_id: int,
        report_date: date,
        actor_id: int,
        timezone_offset_minutes: int = 330,
    ) -> GeneratedReport:
        """
        Creates and generates a daily situation report.

        Default timezone offset is IST (+05:30).
        The report interval is converted to UTC.

        Does not automatically approve or publish.
        """

        if not -720 <= timezone_offset_minutes <= 840:
            raise ReportValidationError(
                "Invalid timezone offset"
            )

        local_timezone = timezone(
            timedelta(
                minutes=timezone_offset_minutes
            )
        )

        start_local = datetime.combine(
            report_date,
            time.min,
            tzinfo=local_timezone,
        )

        end_local = start_local + timedelta(
            days=1
        )

        period = ReportPeriod(
            start_at=start_local.astimezone(
                timezone.utc
            ),
            end_at=end_local.astimezone(
                timezone.utc
            ),
        )

        request = ReportCreateRequest(
            district_id=district_id,
            report_type=ReportType.DAILY_SITUATION,
            period=period,
            title=(
                f"Daily Flood Situation Report - "
                f"{report_date.isoformat()}"
            ),
            visibility=ReportVisibility.INTERNAL,
            priority=ReportPriority.HIGH,
        )

        created = await self.create_report(
            request,
            actor_id=actor_id,
        )

        return await self.generate_report(
            created.id,
            actor_id=actor_id,
        )
