from __future__ import annotations

"""
Kerala Flood Relief Coordination Agent
File: backend/workers/report_worker.py

Background report generation worker.

Features:
- District flood situation reports
- SOS and rescue operation reports
- Shelter and relief reports
- Medical and volunteer reports
- Summary and operational analytics
- JSON, CSV and PDF export
- Celery background tasks
- Redis job status tracking
- Report validation
- Audit logging
- Safe file paths
- No fabricated operational statistics

Production requirements:
- Authenticated report requests
- Authorized data-source integration
- Privacy and access controls
- Durable audit records
- Secure report downloads
"""


import csv
import io
import json
import logging
import os
import uuid

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from celery import Celery
from pydantic import BaseModel, Field, field_validator
from redis import Redis
from redis.exceptions import RedisError

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from backend.core.config import get_settings


# ============================================================
# CONFIGURATION
# ============================================================

settings = get_settings()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = BASE_DIR / "storage" / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_TTL = 60 * 60 * 24 * 30


# ============================================================
# CELERY
# ============================================================

celery_app = Celery(
    "kerala_flood_report_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    task_soft_time_limit=240,
    task_time_limit=300,
)


# ============================================================
# REDIS
# ============================================================

redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


# ============================================================
# ENUMS
# ============================================================

class ReportType(str, Enum):
    FLOOD_SITUATION = "flood_situation"
    DISTRICT_SUMMARY = "district_summary"
    SOS = "sos"
    RESCUE = "rescue"
    SHELTER = "shelter"
    RELIEF = "relief"
    MEDICAL = "medical"
    VOLUNTEER = "volunteer"
    DAMAGE = "damage"
    RECOVERY = "recovery"
    OPERATIONS = "operations"


class ReportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"


class ReportStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# DISTRICTS
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


# ============================================================
# SCHEMAS
# ============================================================

class ReportRequest(BaseModel):
    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    report_type: ReportType
    report_format: ReportFormat = ReportFormat.PDF

    district: str | None = None

    start_date: datetime
    end_date: datetime

    requested_by: str = Field(min_length=1)

    title: str | None = None

    filters: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("district")
    @classmethod
    def validate_district(
        cls,
        value: str | None,
    ) -> str | None:

        if value is None:
            return None

        district = value.strip().lower()

        if district not in KERALA_DISTRICTS:
            raise ValueError(
                f"Invalid Kerala district: {value}"
            )

        return district


class ReportResult(BaseModel):
    report_id: str
    report_type: ReportType
    report_format: ReportFormat

    status: ReportStatus

    file_name: str | None = None
    file_path: str | None = None

    generated_at: datetime | None = None

    error: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# TIME UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


# ============================================================
# REDIS STATUS
# ============================================================

def report_status_key(report_id: str) -> str:
    return f"reports:status:{report_id}"


def save_report_status(
    report_id: str,
    status: ReportStatus,
    **fields: Any,
) -> None:

    key = report_status_key(report_id)

    raw = redis_client.get(key)

    payload = json.loads(raw) if raw else {}

    payload.update({
        "report_id": report_id,
        "status": status.value,
        "updated_at": utc_iso(),
        **fields,
    })

    redis_client.set(
        key,
        json.dumps(payload, default=str),
        ex=REPORT_TTL,
    )


def get_report_status(
    report_id: str,
) -> dict[str, Any] | None:

    raw = redis_client.get(
        report_status_key(report_id)
    )

    return json.loads(raw) if raw else None


# ============================================================
# AUDIT LOGGING
# ============================================================

def audit_event(
    report_id: str,
    event: str,
    details: dict[str, Any] | None = None,
) -> None:

    record = {
        "report_id": report_id,
        "event": event,
        "timestamp": utc_iso(),
        "details": details or {},
    }

    redis_client.xadd(
        "reports:audit",
        {
            "record": json.dumps(
                record,
                default=str,
            )
        },
        maxlen=100000,
        approximate=True,
    )

    logger.info(
        "Report audit: id=%s event=%s",
        report_id,
        event,
    )


# ============================================================
# DATA PROVIDER INTERFACE
# ============================================================

class ReportDataUnavailableError(RuntimeError):
    pass


def fetch_report_data(
    request: ReportRequest,
) -> dict[str, Any]:
    """
    Integrate with backend/services/report_service.py.

    The production implementation should query
    authorized, verified database records.

    Never substitute invented counts for missing
    rescue, casualty or flood observations.
    """

    raise ReportDataUnavailableError(
        "Report data provider is not configured. "
        "Connect backend/services/report_service.py"
    )


# ============================================================
# REPORT VALIDATION
# ============================================================

def validate_report_request(
    request: ReportRequest,
) -> None:

    if (
        request.start_date.tzinfo is None
        or request.end_date.tzinfo is None
    ):
        raise ValueError(
            "Report dates must include timezone"
        )

    if request.start_date > request.end_date:
        raise ValueError(
            "Start date must not exceed end date"
        )

    if request.end_date > utc_now():
        raise ValueError(
            "End date cannot be in the future"
        )


# ============================================================
# REPORT DATA FLATTENING
# ============================================================

def flatten_dict(
    data: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:

    result = {}

    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(
                flatten_dict(value, name)
            )
        elif isinstance(value, list):
            result[name] = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        else:
            result[name] = value

    return result


# ============================================================
# JSON EXPORT
# ============================================================

def export_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# CSV EXPORT
# ============================================================

def export_csv(
    data: dict[str, Any],
    output_path: Path,
) -> None:

    flattened = flatten_dict(data)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Metric",
            "Value",
        ])

        for key, value in flattened.items():
            writer.writerow([
                key,
                value,
            ])


# ============================================================
# PDF EXPORT
# ============================================================

def export_pdf(
    request: ReportRequest,
    data: dict[str, Any],
    output_path: Path,
) -> None:

    from xml.sax.saxutils import escape

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=17,
        leading=22,
        spaceAfter=12,
    )

    body_style = styles["BodyText"]

    elements = []

    title = (
        request.title
        or request.report_type.value.replace(
            "_",
            " ",
        ).title()
    )

    elements.append(
        Paragraph(
            escape(title),
            title_style,
        )
    )

    elements.append(
        Paragraph(
            "Kerala Flood Relief Coordination Agent",
            body_style,
        )
    )

    elements.append(Spacer(1, 10))

    metadata = [
        ["Field", "Value"],
        ["Report ID", request.report_id],
        ["Type", request.report_type.value],
        ["District", request.district or "All authorized districts"],
        ["Start", request.start_date.isoformat()],
        ["End", request.end_date.isoformat()],
        ["Generated", utc_iso()],
    ]

    table = Table(
        [
            [
                Paragraph(escape(str(cell)), body_style)
                for cell in row
            ]
            for row in metadata
        ],
        colWidths=[45 * mm, 125 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 18))

    elements.append(
        Paragraph(
            "Report Data",
            styles["Heading2"],
        )
    )

    flattened = flatten_dict(data)

    for key, value in flattened.items():
        elements.append(
            Paragraph(
                f"<b>{escape(str(key))}:</b> "
                f"{escape(str(value))}",
                body_style,
            )
        )

        elements.append(Spacer(1, 5))

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "Data must be verified before operational use. "
            "This report is not an official flood warning.",
            body_style,
        )
    )

    document.build(elements)


# ============================================================
# EXPORT ROUTER
# ============================================================

def export_report(
    request: ReportRequest,
    data: dict[str, Any],
) -> Path:

    filename = (
        f"{request.report_type.value}_"
        f"{request.report_id}."
        f"{request.report_format.value}"
    )

    output_path = REPORT_DIR / filename

    if request.report_format == ReportFormat.JSON:
        export_json(data, output_path)

    elif request.report_format == ReportFormat.CSV:
        export_csv(data, output_path)

    elif request.report_format == ReportFormat.PDF:
        export_pdf(
            request,
            data,
            output_path,
        )

    else:
        raise ValueError(
            "Unsupported report format"
        )

    return output_path


# ============================================================
# REPORT GENERATION TASK
# ============================================================

@celery_app.task(
    bind=True,
    name="reports.generate",
    max_retries=3,
    default_retry_delay=30,
)
def generate_report(
    self,
    request_data: dict[str, Any],
) -> dict[str, Any]:

    request = ReportRequest.model_validate(
        request_data
    )

    validate_report_request(request)

    report_id = request.report_id

    existing = get_report_status(report_id)

    if existing and existing.get("status") == (
        ReportStatus.COMPLETED.value
    ):
        return existing

    save_report_status(
        report_id,
        ReportStatus.PROCESSING,
        report_type=request.report_type.value,
        report_format=request.report_format.value,
    )

    try:
        data = fetch_report_data(request)

        if not isinstance(data, dict):
            raise ValueError(
                "Report data provider returned invalid data"
            )

        output_path = export_report(
            request,
            data,
        )

        result = ReportResult(
            report_id=report_id,
            report_type=request.report_type,
            report_format=request.report_format,
            status=ReportStatus.COMPLETED,
            file_name=output_path.name,
            file_path=str(output_path),
            generated_at=utc_now(),
            metadata={
                "district": request.district,
                "requested_by": request.requested_by,
                "file_size_bytes": output_path.stat().st_size,
            },
        )

        payload = result.model_dump(mode="json")

        save_report_status(
            report_id,
            ReportStatus.COMPLETED,
            **{
                key: value
                for key, value in payload.items()
                if key not in {"report_id", "status"}
            },
        )

        audit_event(
            report_id,
            "report_generated",
            {
                "type": request.report_type.value,
                "format": request.report_format.value,
            },
        )

        return payload

    except ReportDataUnavailableError as exc:
        logger.error(
            "Report data unavailable: %s",
            exc,
        )

        save_report_status(
            report_id,
            ReportStatus.FAILED,
            error=str(exc),
        )

        return {
            "report_id": report_id,
            "status": ReportStatus.FAILED.value,
            "error": str(exc),
        }

    except (RedisError, OSError) as exc:
        logger.exception(
            "Temporary report worker failure"
        )

        raise self.retry(exc=exc)

    except Exception as exc:
        logger.exception(
            "Report generation failed"
        )

        save_report_status(
            report_id,
            ReportStatus.FAILED,
            error="Report generation failed",
        )

        raise


# ============================================================
# REPORT STATUS TASK
# ============================================================

@celery_app.task(
    name="reports.status",
)
def report_status(
    report_id: str,
) -> dict[str, Any]:

    record = get_report_status(report_id)

    if record is None:
        return {
            "report_id": report_id,
            "status": "not_found",
        }

    return record


# ============================================================
# WORKER HEALTH
# ============================================================

@celery_app.task(
    name="reports.worker_health",
)
def worker_health() -> dict[str, Any]:

    redis_client.ping()

    return {
        "worker": "report_worker",
        "status": "healthy",
        "timestamp": utc_iso(),
        "report_formats": [
            format_.value
            for format_ in ReportFormat
        ],
    }
