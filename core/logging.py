from __future__ import annotations

"""
backend/core/logging.py

Kerala Flood Relief Coordination Agent

Production-oriented structured logging for:
- FastAPI backend
- Authentication and security
- SOS and emergency response
- Flood monitoring and prediction
- Rescue and evacuation
- Shelter and relief operations
- Volunteer management
- Healthcare operations
- AI agents, RAG, and ML models
- Background workers
- External integrations

Features:
- Structured JSON logging
- Console and rotating file handlers
- UTC timestamps
- Request and correlation IDs
- Context-aware logging
- Sensitive-data redaction
- Exception logging
- Log-level configuration
- Request duration tracking
- Optional access logging middleware

No external logging dependency required.
"""


import contextvars
import json
import logging
import logging.config
import os
import re
import sys
import time
import uuid

from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "kerala-flood-relief-coordination-agent"

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "json",
).lower()

LOG_DIRECTORY = Path(
    os.getenv("LOG_DIRECTORY", "logs")
)

LOG_FILE = os.getenv(
    "LOG_FILE",
    "application.log",
)

MAX_LOG_SIZE = int(
    os.getenv(
        "MAX_LOG_SIZE",
        str(10 * 1024 * 1024),
    )
)

LOG_BACKUP_COUNT = int(
    os.getenv("LOG_BACKUP_COUNT", "5")
)


# ============================================================
# REQUEST CONTEXT
# ============================================================

request_id_var = contextvars.ContextVar(
    "request_id",
    default=None,
)

correlation_id_var = contextvars.ContextVar(
    "correlation_id",
    default=None,
)

user_id_var = contextvars.ContextVar(
    "user_id",
    default=None,
)

district_id_var = contextvars.ContextVar(
    "district_id",
    default=None,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_request_id() -> str:
    return str(uuid.uuid4())


def set_request_context(
    *,
    request_id: str | None = None,
    correlation_id: str | None = None,
    user_id: str | int | None = None,
    district_id: str | int | None = None,
) -> None:
    request_id_var.set(request_id)
    correlation_id_var.set(correlation_id)
    user_id_var.set(user_id)
    district_id_var.set(district_id)


def clear_request_context() -> None:
    request_id_var.set(None)
    correlation_id_var.set(None)
    user_id_var.set(None)
    district_id_var.set(None)


def get_request_context() -> dict[str, Any]:
    return {
        "request_id": request_id_var.get(),
        "correlation_id": correlation_id_var.get(),
        "user_id": user_id_var.get(),
        "district_id": district_id_var.get(),
    }


# ============================================================
# SENSITIVE DATA PROTECTION
# ============================================================

SENSITIVE_FIELDS = {
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "api_key",
    "apikey",
    "private_key",
    "client_secret",
    "cookie",
    "set_cookie",
    "session",
    "session_id",
    "otp",
    "aadhaar",
    "aadhaar_number",
    "phone",
    "phone_number",
    "mobile",
    "email",
    "address",
    "latitude",
    "longitude",
    "gps",
    "coordinates",
    "location",
    "medical_history",
    "medical_record",
    "patient_name",
    "emergency_contact",
    "contact_number",
    "date_of_birth",
    "dob",
    "bank_account",
    "account_number",
}

REDACTED = "[REDACTED]"

MAX_REDACTION_DEPTH = 10

SECRET_PATTERNS = [
    re.compile(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-=]+"
    ),
    re.compile(
        r"(?i)\b(?:password|token|api_key|secret)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
]


def redact_string(value: str) -> str:
    result = value

    for pattern in SECRET_PATTERNS:
        result = pattern.sub(REDACTED, result)

    return result


def sanitize_log_data(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    if depth > MAX_REDACTION_DEPTH:
        return "[MAX_DEPTH]"

    if isinstance(value, Mapping):
        sanitized = {}

        for key, item in value.items():
            key_string = str(key)
            normalized_key = (
                key_string.lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized_key in SENSITIVE_FIELDS:
                sanitized[key_string] = REDACTED
            else:
                sanitized[key_string] = sanitize_log_data(
                    item,
                    depth=depth + 1,
                )

        return sanitized

    if isinstance(value, (list, tuple, set)):
        return [
            sanitize_log_data(
                item,
                depth=depth + 1,
            )
            for item in list(value)[:100]
        ]

    if isinstance(value, str):
        return redact_string(value[:10000])

    if isinstance(value, (int, float, bool)):
        return value

    if value is None:
        return None

    return f"[{type(value).__name__}]"


# ============================================================
# LOG RECORD FILTER
# ============================================================

class ContextFilter(logging.Filter):
    """
    Attach request context to every log record.
    """

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        context = get_request_context()

        record.request_id = context["request_id"]
        record.correlation_id = context["correlation_id"]
        record.user_id = context["user_id"]
        record.district_id = context["district_id"]

        return True


# ============================================================
# JSON FORMATTER
# ============================================================

class JSONFormatter(logging.Formatter):
    """
    Convert Python log records into structured JSON.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        timestamp = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc,
        ).isoformat()

        payload = {
            "timestamp": timestamp,
            "application": APP_NAME,
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_log_data(
                record.getMessage()
            ),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
            "request_id": getattr(
                record,
                "request_id",
                None,
            ),
            "correlation_id": getattr(
                record,
                "correlation_id",
                None,
            ),
            "user_id": getattr(
                record,
                "user_id",
                None,
            ),
            "district_id": getattr(
                record,
                "district_id",
                None,
            ),
        }

        extra_data = getattr(
            record,
            "structured_data",
            None,
        )

        if extra_data is not None:
            payload["data"] = sanitize_log_data(
                extra_data
            )

        if record.exc_info:
            exception = record.exc_info[1]

            payload["exception"] = {
                "type": (
                    type(exception).__name__
                    if exception
                    else "Unknown"
                ),
                "message": (
                    "Exception details withheld"
                    if exception
                    else None
                ),
            }

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# TEXT FORMATTER
# ============================================================

class SafeTextFormatter(logging.Formatter):
    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_log_data(
                record.getMessage()
            ),
            "request_id": getattr(
                record,
                "request_id",
                None,
            ),
        }

        return " | ".join(
            f"{key}={value}"
            for key, value in payload.items()
        )


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(
    *,
    level: str | None = None,
    log_format: str | None = None,
    enable_file_logging: bool = True,
) -> None:
    """
    Initialize application-wide logging.

    Call once during application startup.
    """

    selected_level = (
        level or LOG_LEVEL
    ).upper()

    selected_format = (
        log_format or LOG_FORMAT
    ).lower()

    numeric_level = getattr(
        logging,
        selected_level,
        None,
    )

    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid logging level: {selected_level}"
        )

    formatter = (
        JSONFormatter()
        if selected_format == "json"
        else SafeTextFormatter()
    )

    context_filter = ContextFilter()

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)

    root_logger.addHandler(console_handler)

    if enable_file_logging:
        LOG_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = RotatingFileHandler(
            filename=LOG_DIRECTORY / LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )

        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)

        root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(
        logging.WARNING
    )

    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.WARNING
    )

    logging.getLogger(__name__).info(
        "Application logging initialized"
    )


# ============================================================
# LOGGER FACTORY
# ============================================================

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class StructuredLogger:
    """
    Convenience wrapper for structured application logs.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        message: str,
        **data: Any,
    ) -> None:
        self.logger.log(
            level,
            message,
            extra={
                "structured_data": data,
            },
            stacklevel=3,
        )

    def debug(
        self,
        message: str,
        **data: Any,
    ) -> None:
        self._log(
            logging.DEBUG,
            message,
            **data,
        )

    def info(
        self,
        message: str,
        **data: Any,
    ) -> None:
        self._log(
            logging.INFO,
            message,
            **data,
        )

    def warning(
        self,
        message: str,
        **data: Any,
    ) -> None:
        self._log(
            logging.WARNING,
            message,
            **data,
        )

    def error(
        self,
        message: str,
        **data: Any,
    ) -> None:
        self._log(
            logging.ERROR,
            message,
            **data,
        )

    def critical(
        self,
        message: str,
        **data: Any,
    ) -> None:
        self._log(
            logging.CRITICAL,
            message,
            **data,
        )

    def exception(
        self,
        message: str,
        **data: Any,
    ) -> None:
        self.logger.error(
            message,
            extra={
                "structured_data": data,
            },
            exc_info=True,
        )


def get_structured_logger(
    name: str,
) -> StructuredLogger:
    return StructuredLogger(name)


# ============================================================
# DOMAIN LOGGING
# ============================================================

class FloodReliefLogger:
    """
    Domain-specific logging helpers.
    """

    def __init__(self):
        self.logger = get_structured_logger(
            "flood_relief"
        )

    def log_sos_created(
        self,
        sos_id: int,
        district_id: int,
        priority: str,
    ) -> None:
        self.logger.critical(
            "Emergency SOS request created",
            event="sos.created",
            sos_id=sos_id,
            district_id=district_id,
            priority=priority,
        )

    def log_sos_resolved(
        self,
        sos_id: int,
    ) -> None:
        self.logger.info(
            "SOS request resolved",
            event="sos.resolved",
            sos_id=sos_id,
        )

    def log_flood_alert(
        self,
        alert_id: int,
        district_id: int,
        severity: str,
    ) -> None:
        self.logger.warning(
            "Flood alert generated",
            event="flood.alert",
            alert_id=alert_id,
            district_id=district_id,
            severity=severity,
        )

    def log_flood_prediction(
        self,
        district_id: int,
        risk_level: str,
        model_version: str,
    ) -> None:
        self.logger.info(
            "Flood prediction generated",
            event="flood.prediction",
            district_id=district_id,
            risk_level=risk_level,
            model_version=model_version,
        )

    def log_rescue_dispatch(
        self,
        mission_id: int,
        team_id: int,
    ) -> None:
        self.logger.critical(
            "Rescue team dispatched",
            event="rescue.dispatched",
            mission_id=mission_id,
            team_id=team_id,
        )

    def log_rescue_completed(
        self,
        mission_id: int,
    ) -> None:
        self.logger.info(
            "Rescue mission completed",
            event="rescue.completed",
            mission_id=mission_id,
        )

    def log_volunteer_assignment(
        self,
        volunteer_id: int,
        mission_id: int,
    ) -> None:
        self.logger.info(
            "Volunteer assigned",
            event="volunteer.assigned",
            volunteer_id=volunteer_id,
            mission_id=mission_id,
        )

    def log_shelter_allocation(
        self,
        shelter_id: int,
        allocation_id: int,
    ) -> None:
        self.logger.info(
            "Shelter allocation completed",
            event="shelter.allocated",
            shelter_id=shelter_id,
            allocation_id=allocation_id,
        )

    def log_relief_distribution(
        self,
        delivery_id: int,
        district_id: int,
    ) -> None:
        self.logger.info(
            "Relief distribution recorded",
            event="relief.distributed",
            delivery_id=delivery_id,
            district_id=district_id,
        )

    def log_agent_decision(
        self,
        agent_name: str,
        decision_id: str,
        confidence: float | None = None,
    ) -> None:
        self.logger.info(
            "AI agent decision generated",
            event="agent.decision",
            agent_name=agent_name,
            decision_id=decision_id,
            confidence=confidence,
        )

    def log_human_approval(
        self,
        decision_id: str,
        approved: bool,
    ) -> None:
        self.logger.info(
            "Human review completed",
            event="agent.human_review",
            decision_id=decision_id,
            approved=approved,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str = "warning",
    ) -> None:
        self.logger.warning(
            "Security event detected",
            event="security.event",
            event_type=event_type,
            severity=severity,
        )

    def log_worker_event(
        self,
        worker_name: str,
        status: str,
        duration_ms: float | None = None,
    ) -> None:
        self.logger.info(
            "Background worker event",
            event="worker.execution",
            worker_name=worker_name,
            status=status,
            duration_ms=duration_ms,
        )


flood_logger = FloodReliefLogger()


# ============================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Track API request duration and correlation IDs.

    Does not log request bodies, query strings,
    authorization headers, or precise locations.
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        start_time = time.perf_counter()

        request_id = generate_request_id()

        request.state.request_id = request_id

        request_id_token = request_id_var.set(
            request_id
        )

        correlation_id = request_id

        correlation_token = correlation_id_var.set(
            correlation_id
        )

        access_logger = get_structured_logger(
            "http.access"
        )

        try:
            response = await call_next(request)

            duration_ms = round(
                (time.perf_counter() - start_time)
                * 1000,
                2,
            )

            response.headers[
                "X-Request-ID"
            ] = request_id

            response.headers[
                "X-Response-Time"
            ] = f"{duration_ms}ms"

            access_logger.info(
                "HTTP request completed",
                event="http.request",
                method=request.method,
                route=(
                    request.scope.get("route").path
                    if request.scope.get("route")
                    else "unmatched"
                ),
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

            return response

        except Exception:
            duration_ms = round(
                (time.perf_counter() - start_time)
                * 1000,
                2,
            )

            access_logger.error(
                "HTTP request failed",
                event="http.request_failed",
                method=request.method,
                duration_ms=duration_ms,
            )

            raise

        finally:
            request_id_var.reset(
                request_id_token
            )

            correlation_id_var.reset(
                correlation_token
            )


# ============================================================
# FASTAPI INTEGRATION
# ============================================================

def configure_application_logging(
    app: FastAPI,
    *,
    level: str | None = None,
    enable_file_logging: bool = True,
) -> None:
    setup_logging(
        level=level,
        enable_file_logging=enable_file_logging,
    )

    app.add_middleware(
        RequestLoggingMiddleware
    )


# ============================================================
# BACKGROUND TASK LOGGING
# ============================================================

def log_background_task(
    task_name: str,
    *,
    task_id: str | None = None,
):
    """
    Decorator for synchronous background tasks.
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            logger = get_structured_logger(
                f"worker.{task_name}"
            )

            current_task_id = (
                task_id or str(uuid.uuid4())
            )

            start = time.perf_counter()

            logger.info(
                "Background task started",
                task_name=task_name,
                task_id=current_task_id,
            )

            try:
                result = func(*args, **kwargs)

                duration = round(
                    (time.perf_counter() - start)
                    * 1000,
                    2,
                )

                logger.info(
                    "Background task completed",
                    task_name=task_name,
                    task_id=current_task_id,
                    duration_ms=duration,
                )

                return result

            except Exception:
                logger.error(
                    "Background task failed",
                    task_name=task_name,
                    task_id=current_task_id,
                )

                raise

        return wrapper

    return decorator


# ============================================================
# ASYNC BACKGROUND TASK LOGGING
# ============================================================

def log_async_background_task(
    task_name: str,
):
    """
    Decorator for asynchronous workers.
    """

    def decorator(func):

        async def wrapper(*args, **kwargs):
            logger = get_structured_logger(
                f"worker.{task_name}"
            )

            task_id = str(uuid.uuid4())
            start = time.perf_counter()

            logger.info(
                "Async background task started",
                task_name=task_name,
                task_id=task_id,
            )

            try:
                result = await func(
                    *args,
                    **kwargs,
                )

                duration = round(
                    (time.perf_counter() - start)
                    * 1000,
                    2,
                )

                logger.info(
                    "Async background task completed",
                    task_name=task_name,
                    task_id=task_id,
                    duration_ms=duration,
                )

                return result

            except Exception:
                logger.error(
                    "Async background task failed",
                    task_name=task_name,
                    task_id=task_id,
                )

                raise

        return wrapper

    return decorator


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":
    setup_logging()

    app_logger = get_structured_logger(
        "backend.core.logging"
    )

    app_logger.info(
        "Flood relief logging test",
        environment="development",
        district="Thrissur",
        system_status="initialized",
    )

    flood_logger.log_flood_alert(
        alert_id=101,
        district_id=7,
        severity="high",
    )

    flood_logger.log_rescue_dispatch(
        mission_id=1001,
        team_id=25,
    )

    flood_logger.log_agent_decision(
        agent_name="rescue_coordinator_agent",
        decision_id="decision-1001",
        confidence=0.94,
    )
