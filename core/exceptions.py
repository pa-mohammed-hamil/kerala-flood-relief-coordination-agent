from __future__ import annotations

"""
backend/core/exceptions.py

Kerala Flood Relief Coordination Agent
Centralized exception handling for FastAPI.
"""


import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "APPLICATION_ERROR"
    default_message = "An application error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Any = None,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        self.message = message or self.default_message
        self.details = details

        if status_code is not None:
            self.status_code = status_code

        if error_code is not None:
            self.error_code = error_code

        super().__init__(self.message)


# --------------------------------------------------
# Authentication and authorization
# --------------------------------------------------

class AuthenticationError(AppException):
    status_code = 401
    error_code = "AUTHENTICATION_FAILED"
    default_message = "Authentication required."


class InvalidCredentialsError(AuthenticationError):
    error_code = "INVALID_CREDENTIALS"
    default_message = "Invalid credentials."


class TokenExpiredError(AuthenticationError):
    error_code = "TOKEN_EXPIRED"
    default_message = "Authentication token has expired."


class InvalidTokenError(AuthenticationError):
    error_code = "INVALID_TOKEN"
    default_message = "Invalid authentication token."


class PermissionDeniedError(AppException):
    status_code = 403
    error_code = "PERMISSION_DENIED"
    default_message = "You do not have permission."


class AccountDisabledError(AppException):
    status_code = 403
    error_code = "ACCOUNT_DISABLED"
    default_message = "Account is disabled."


# --------------------------------------------------
# Resource and validation errors
# --------------------------------------------------

class NotFoundError(AppException):
    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"
    default_message = "Requested resource was not found."


class ConflictError(AppException):
    status_code = 409
    error_code = "RESOURCE_CONFLICT"
    default_message = "Resource conflict detected."


class AlreadyExistsError(ConflictError):
    error_code = "RESOURCE_ALREADY_EXISTS"
    default_message = "Resource already exists."


class ValidationError(AppException):
    status_code = 422
    error_code = "VALIDATION_ERROR"
    default_message = "Invalid request data."


class BadRequestError(AppException):
    status_code = 400
    error_code = "BAD_REQUEST"
    default_message = "Invalid request."


class RateLimitExceededError(AppException):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    default_message = "Too many requests."


# --------------------------------------------------
# Flood monitoring and forecasting
# --------------------------------------------------

class FloodDataError(AppException):
    status_code = 422
    error_code = "FLOOD_DATA_ERROR"
    default_message = "Invalid flood monitoring data."


class WeatherDataUnavailableError(AppException):
    status_code = 503
    error_code = "WEATHER_DATA_UNAVAILABLE"
    default_message = "Weather data is unavailable."


class RiverMonitoringError(AppException):
    status_code = 503
    error_code = "RIVER_MONITORING_ERROR"
    default_message = "River monitoring service unavailable."


class DamMonitoringError(AppException):
    status_code = 503
    error_code = "DAM_MONITORING_ERROR"
    default_message = "Dam monitoring service unavailable."


class FloodPredictionError(AppException):
    status_code = 503
    error_code = "FLOOD_PREDICTION_ERROR"
    default_message = "Flood prediction unavailable."


class FloodAlertError(AppException):
    status_code = 503
    error_code = "FLOOD_ALERT_ERROR"
    default_message = "Flood alert operation failed."


# --------------------------------------------------
# SOS and emergency response
# --------------------------------------------------

class SOSError(AppException):
    status_code = 400
    error_code = "SOS_ERROR"
    default_message = "SOS request could not be processed."


class SOSNotFoundError(NotFoundError):
    error_code = "SOS_NOT_FOUND"
    default_message = "SOS request not found."


class SOSAlreadyResolvedError(ConflictError):
    error_code = "SOS_ALREADY_RESOLVED"
    default_message = "SOS request is already resolved."


class EmergencyDispatchError(AppException):
    status_code = 503
    error_code = "EMERGENCY_DISPATCH_ERROR"
    default_message = "Emergency dispatch is unavailable."


class RescueTeamUnavailableError(AppException):
    status_code = 409
    error_code = "RESCUE_TEAM_UNAVAILABLE"
    default_message = "No eligible rescue team is available."


class RescueMissionError(AppException):
    status_code = 409
    error_code = "RESCUE_MISSION_ERROR"
    default_message = "Rescue mission operation failed."


class RoutePlanningError(AppException):
    status_code = 503
    error_code = "ROUTE_PLANNING_ERROR"
    default_message = "Safe route planning unavailable."


class EvacuationError(AppException):
    status_code = 409
    error_code = "EVACUATION_ERROR"
    default_message = "Evacuation operation could not be completed."


# --------------------------------------------------
# Shelter and relief management
# --------------------------------------------------

class ShelterNotFoundError(NotFoundError):
    error_code = "SHELTER_NOT_FOUND"
    default_message = "Shelter not found."


class ShelterCapacityExceededError(ConflictError):
    error_code = "SHELTER_CAPACITY_EXCEEDED"
    default_message = "Shelter capacity exceeded."


class ShelterAllocationError(ConflictError):
    error_code = "SHELTER_ALLOCATION_ERROR"
    default_message = "Shelter allocation failed."


class InventoryNotFoundError(NotFoundError):
    error_code = "INVENTORY_NOT_FOUND"
    default_message = "Inventory item not found."


class InsufficientInventoryError(ConflictError):
    error_code = "INSUFFICIENT_INVENTORY"
    default_message = "Insufficient relief inventory."


class ReliefDistributionError(ConflictError):
    error_code = "RELIEF_DISTRIBUTION_ERROR"
    default_message = "Relief distribution failed."


# --------------------------------------------------
# Healthcare and volunteers
# --------------------------------------------------

class MedicalEmergencyError(AppException):
    status_code = 503
    error_code = "MEDICAL_EMERGENCY_ERROR"
    default_message = "Medical emergency service unavailable."


class AmbulanceUnavailableError(AppException):
    status_code = 503
    error_code = "AMBULANCE_UNAVAILABLE"
    default_message = "Ambulance availability could not be confirmed."


class HospitalCapacityError(ConflictError):
    error_code = "HOSPITAL_CAPACITY_ERROR"
    default_message = "Hospital capacity is insufficient."


class VolunteerNotFoundError(NotFoundError):
    error_code = "VOLUNTEER_NOT_FOUND"
    default_message = "Volunteer not found."


class VolunteerUnavailableError(ConflictError):
    error_code = "VOLUNTEER_UNAVAILABLE"
    default_message = "Volunteer is unavailable."


class VolunteerAssignmentError(ConflictError):
    error_code = "VOLUNTEER_ASSIGNMENT_ERROR"
    default_message = "Volunteer assignment failed."


# --------------------------------------------------
# AI, RAG and agent orchestration
# --------------------------------------------------

class AIModelError(AppException):
    status_code = 503
    error_code = "AI_MODEL_ERROR"
    default_message = "AI model operation failed."


class ModelUnavailableError(AIModelError):
    error_code = "MODEL_UNAVAILABLE"
    default_message = "Requested AI model is unavailable."


class PredictionError(AIModelError):
    error_code = "PREDICTION_FAILED"
    default_message = "Prediction could not be generated."


class AgentExecutionError(AppException):
    status_code = 503
    error_code = "AGENT_EXECUTION_ERROR"
    default_message = "AI agent execution failed."


class AgentTimeoutError(AppException):
    status_code = 504
    error_code = "AGENT_TIMEOUT"
    default_message = "AI agent execution timed out."


class AgentSafetyError(AppException):
    status_code = 403
    error_code = "AGENT_SAFETY_BLOCK"
    default_message = "AI action blocked by safety controls."


class HumanApprovalRequiredError(AppException):
    status_code = 409
    error_code = "HUMAN_APPROVAL_REQUIRED"
    default_message = "Human approval is required."


class RAGRetrievalError(AppException):
    status_code = 503
    error_code = "RAG_RETRIEVAL_ERROR"
    default_message = "Knowledge retrieval failed."


class KnowledgeBaseError(AppException):
    status_code = 503
    error_code = "KNOWLEDGE_BASE_ERROR"
    default_message = "Knowledge base unavailable."


class HallucinationDetectedError(AppException):
    status_code = 422
    error_code = "UNVERIFIED_AI_OUTPUT"
    default_message = "AI output failed verification."


# --------------------------------------------------
# External integrations
# --------------------------------------------------

class ExternalServiceError(AppException):
    status_code = 503
    error_code = "EXTERNAL_SERVICE_ERROR"
    default_message = "External service unavailable."


class SMSDeliveryError(ExternalServiceError):
    error_code = "SMS_DELIVERY_FAILED"
    default_message = "SMS delivery could not be confirmed."


class NotificationError(ExternalServiceError):
    error_code = "NOTIFICATION_ERROR"
    default_message = "Notification delivery failed."


class MapsServiceError(ExternalServiceError):
    error_code = "MAPS_SERVICE_ERROR"
    default_message = "Maps service unavailable."


# --------------------------------------------------
# Database, encryption and system errors
# --------------------------------------------------

class DatabaseError(AppException):
    status_code = 500
    error_code = "DATABASE_ERROR"
    default_message = "Database operation failed."


class EncryptionOperationError(AppException):
    status_code = 500
    error_code = "ENCRYPTION_ERROR"
    default_message = "Secure data operation failed."


class ConfigurationError(AppException):
    status_code = 500
    error_code = "CONFIGURATION_ERROR"
    default_message = "Application configuration error."


class ServiceUnavailableError(AppException):
    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"
    default_message = "Service temporarily unavailable."


# --------------------------------------------------
# Standardized error responses
# --------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_request_id(request: Request) -> str:
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    if request_id:
        return str(request_id)

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


def error_response(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = get_request_id(request)

    body = {
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
            "details": details,
        },
        "request_id": request_id,
        "timestamp": utc_now(),
    }

    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=response_headers,
    )


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    if exc.status_code >= 500:
        logger.error(
            "Application error: %s, request_id=%s",
            exc.error_code,
            get_request_id(request),
        )

    return error_response(
        request,
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=(
            exc.details
            if exc.status_code < 500
            else None
        ),
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    message = (
        str(exc.detail)
        if isinstance(exc.detail, str)
        else "HTTP request failed."
    )

    return error_response(
        request,
        status_code=exc.status_code,
        error_code="HTTP_ERROR",
        message=message,
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    safe_details = []

    for error in exc.errors():
        safe_details.append({
            "field": [
                str(item)
                for item in error.get("loc", [])
            ],
            "message": error.get(
                "msg",
                "Invalid input",
            ),
            "type": error.get(
                "type",
                "validation_error",
            ),
        })

    return error_response(
        request,
        status_code=422,
        error_code="REQUEST_VALIDATION_ERROR",
        message="Request validation failed.",
        details=safe_details,
    )


async def integrity_exception_handler(
    request: Request,
    exc: IntegrityError,
) -> JSONResponse:
    logger.error(
        "Database integrity error, request_id=%s",
        get_request_id(request),
    )

    return error_response(
        request,
        status_code=409,
        error_code="DATABASE_CONSTRAINT_ERROR",
        message="Database constraint violation.",
    )


async def database_exception_handler(
    request: Request,
    exc: SQLAlchemyError,
) -> JSONResponse:
    logger.error(
        "Database operation failed, request_id=%s",
        get_request_id(request),
    )

    return error_response(
        request,
        status_code=500,
        error_code="DATABASE_ERROR",
        message="Database operation failed.",
    )


async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled application exception, request_id=%s",
        get_request_id(request),
    )

    return error_response(
        request,
        status_code=500,
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred.",
    )


# --------------------------------------------------
# FastAPI registration
# --------------------------------------------------

def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all application exception handlers.

    Usage:
        from backend.core.exceptions import (
            register_exception_handlers,
        )

        app = FastAPI()
        register_exception_handlers(app)
    """

    app.add_exception_handler(
        AppException,
        app_exception_handler,
    )

    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )

    app.add_exception_handler(
        IntegrityError,
        integrity_exception_handler,
    )

    app.add_exception_handler(
        SQLAlchemyError,
        database_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unexpected_exception_handler,
    )
