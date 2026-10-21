"""Placeholder module: ai/agents/medical_response_agent.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
backend/ai/agents/medical_response_agent.py

Medical Response Coordination Agent
===================================

Features:
    - Medical assistance request intake
    - Flood-related health incident coordination
    - Emergency symptom flag detection
    - Human-reviewed urgency screening
    - Vulnerable population identification
    - Hospital and healthcare facility screening
    - Ambulance resource availability screening
    - Medical supply requirement tracking
    - Medical transport planning
    - Case deduplication
    - Privacy-aware reporting
    - Human medical review and approval
    - Audit logging
    - JSON export

Safety:
    This module does not diagnose patients,
    prescribe medicines, calculate drug doses,
    authorize treatment or dispatch ambulances.

    Potential emergencies are immediately
    marked for human escalation, but no
    automated real-world action is performed.

    Medical triage decisions must be made
    by appropriately qualified professionals.

Python 3.11+
Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class MedicalResponseError(Exception):
    pass


class MedicalValidationError(MedicalResponseError):
    pass


class MedicalResourceError(MedicalResponseError):
    pass


class MedicalApprovalError(MedicalResponseError):
    pass


class MedicalExportError(MedicalResponseError):
    pass


# ============================================================
# ENUMS
# ============================================================

class MedicalIncidentType(str, Enum):
    FLOOD_INJURY = "flood_injury"
    DROWNING_EXPOSURE = "drowning_exposure"
    RESPIRATORY_DISTRESS = "respiratory_distress"
    SUSPECTED_INFECTION = "suspected_infection"
    DEHYDRATION_CONCERN = "dehydration_concern"
    CHRONIC_CARE_INTERRUPTION = "chronic_care_interruption"
    MATERNAL_HEALTH_CONCERN = "maternal_health_concern"
    MENTAL_HEALTH_CONCERN = "mental_health_concern"
    MEDICATION_ACCESS = "medication_access"
    GENERAL_MEDICAL_REQUEST = "general_medical_request"


class MedicalUrgency(str, Enum):
    EMERGENCY_FLAG = "emergency_flag"
    PRIORITY_REVIEW = "priority_review"
    STANDARD_REVIEW = "standard_review"
    UNDETERMINED = "undetermined"


class MedicalCaseStatus(str, Enum):
    PENDING_MEDICAL_REVIEW = "pending_medical_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class MedicalFacilityStatus(str, Enum):
    VERIFIED_OPERATIONAL = "verified_operational"
    LIMITED_SERVICES = "limited_services"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class AmbulanceStatus(str, Enum):
    VERIFIED_AVAILABLE = "verified_available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class MedicalSupplyCategory(str, Enum):
    FIRST_AID = "first_aid"
    WOUND_CARE = "wound_care"
    HYDRATION = "hydration"
    SANITATION = "sanitation"
    PERSONAL_PROTECTIVE_EQUIPMENT = "ppe"
    CHRONIC_CARE = "chronic_care"
    OTHER = "other"


class MedicalReviewDecision(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    REQUEST_MORE_INFORMATION = "request_more_information"
    REJECT_AS_INVALID = "reject_as_invalid"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MedicalResponseConfig:
    maximum_facility_age_minutes: int = 60
    maximum_ambulance_age_minutes: int = 30

    maximum_facilities: int = 100
    maximum_ambulances: int = 100
    maximum_supply_items: int = 100

    deduplication_window_minutes: int = 60

    require_human_medical_review: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_facility_age_minutes",
            "maximum_ambulance_age_minutes",
            "maximum_facilities",
            "maximum_ambulances",
            "maximum_supply_items",
            "deduplication_window_minutes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise MedicalValidationError(
                    f"{name} must be a positive integer"
                )


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def require_aware_datetime(
    value: datetime,
) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise MedicalValidationError(
            "Datetime must be timezone-aware"
        )


def require_nonnegative_integer(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise MedicalValidationError(
            f"{name} must be a nonnegative integer"
        )


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            serialize(item)
            for item in value
        ]

    return value


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class MedicalAssistanceRequest:
    request_id: str

    district: str
    location_description: str

    incident_type: MedicalIncidentType

    symptoms: tuple[str, ...]

    affected_people: int

    children_count: int = 0
    older_adults_count: int = 0
    pregnant_people_count: int = 0
    people_with_disabilities_count: int = 0

    requires_accessible_transport: bool = False

    requested_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise MedicalValidationError(
                "request_id is required"
            )

        if not self.district.strip():
            raise MedicalValidationError(
                "district is required"
            )

        if not self.location_description.strip():
            raise MedicalValidationError(
                "location_description is required"
            )

        require_nonnegative_integer(
            self.affected_people,
            "affected_people",
        )

        if self.affected_people == 0:
            raise MedicalValidationError(
                "affected_people must be greater than zero"
            )

        for name in (
            "children_count",
            "older_adults_count",
            "pregnant_people_count",
            "people_with_disabilities_count",
        ):
            value = getattr(self, name)

            require_nonnegative_integer(
                value,
                name,
            )

            if value > self.affected_people:
                raise MedicalValidationError(
                    f"{name} exceeds affected_people"
                )

        if self.requested_at is not None:
            require_aware_datetime(
                self.requested_at
            )


@dataclass(frozen=True)
class MedicalFacility:
    facility_id: str
    facility_name: str

    district: str
    status: MedicalFacilityStatus

    verified_at: datetime | None

    emergency_services: bool = False
    inpatient_services: bool = False
    maternal_services: bool = False

    accessible_facilities: bool = False

    available_beds: int = 0

    def __post_init__(self) -> None:
        if not self.facility_id.strip():
            raise MedicalValidationError(
                "facility_id is required"
            )

        if not self.facility_name.strip():
            raise MedicalValidationError(
                "facility_name is required"
            )

        if not self.district.strip():
            raise MedicalValidationError(
                "district is required"
            )

        require_nonnegative_integer(
            self.available_beds,
            "available_beds",
        )

        if self.verified_at is not None:
            require_aware_datetime(
                self.verified_at
            )


@dataclass(frozen=True)
class AmbulanceResource:
    ambulance_id: str

    district: str
    status: AmbulanceStatus

    verified_at: datetime | None

    accessible_transport: bool = False
    advanced_life_support: bool = False

    def __post_init__(self) -> None:
        if not self.ambulance_id.strip():
            raise MedicalValidationError(
                "ambulance_id is required"
            )

        if not self.district.strip():
            raise MedicalValidationError(
                "district is required"
            )

        if self.verified_at is not None:
            require_aware_datetime(
                self.verified_at
            )


@dataclass(frozen=True)
class MedicalSupplyRequirement:
    item_id: str
    category: MedicalSupplyCategory

    description: str
    requested_quantity: int

    available_quantity: int

    unit: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise MedicalValidationError(
                "item_id is required"
            )

        if not self.description.strip():
            raise MedicalValidationError(
                "description is required"
            )

        if not self.unit.strip():
            raise MedicalValidationError(
                "unit is required"
            )

        require_nonnegative_integer(
            self.requested_quantity,
            "requested_quantity",
        )

        require_nonnegative_integer(
            self.available_quantity,
            "available_quantity",
        )


@dataclass(frozen=True)
class MedicalResponsePlanningRequest:
    request_id: str

    assistance_request: MedicalAssistanceRequest

    facilities: tuple[MedicalFacility, ...]
    ambulances: tuple[AmbulanceResource, ...]
    supply_requirements: tuple[
        MedicalSupplyRequirement,
        ...
    ]

    created_at: datetime


@dataclass(frozen=True)
class MedicalUrgencyAssessment:
    urgency: MedicalUrgency

    detected_emergency_flags: tuple[str, ...]
    vulnerability_flags: tuple[str, ...]

    requires_immediate_human_escalation: bool

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MedicalFacilityScreeningResult:
    facility_id: str
    facility_name: str

    eligible_for_consideration: bool

    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AmbulanceScreeningResult:
    ambulance_id: str

    eligible_for_consideration: bool

    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MedicalSupplyGap:
    item_id: str
    category: MedicalSupplyCategory

    requested_quantity: int
    available_quantity: int

    shortage_quantity: int

    unit: str


@dataclass(frozen=True)
class MedicalResponsePlan:
    plan_id: str
    request_id: str

    district: str
    incident_type: MedicalIncidentType

    urgency_assessment: MedicalUrgencyAssessment

    facility_screening: tuple[
        MedicalFacilityScreeningResult,
        ...
    ]

    ambulance_screening: tuple[
        AmbulanceScreeningResult,
        ...
    ]

    supply_gaps: tuple[
        MedicalSupplyGap,
        ...
    ]

    recommendations: tuple[str, ...]
    missing_information: tuple[str, ...]
    warnings: tuple[str, ...]

    status: MedicalCaseStatus

    requires_human_medical_review: bool

    duplicate_detected: bool

    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class MedicalResponseReview:
    review_id: str
    plan_id: str

    reviewer_id: str
    reviewer_role: str

    decision: MedicalReviewDecision
    reason: str

    reviewed_at: datetime


@dataclass(frozen=True)
class MedicalAuditRecord:
    audit_id: str
    plan_id: str

    action: str
    actor_id: str

    timestamp: datetime

    details: Mapping[str, str]


# ============================================================
# EMERGENCY SYMPTOM FLAG ENGINE
# ============================================================

class MedicalUrgencyScreeningEngine:
    """
    Keyword-based emergency flag detection.

    This is NOT a medical triage algorithm.
    It is designed to flag obvious high-risk
    descriptions for rapid human review.

    Absence of a detected flag must never
    be interpreted as absence of emergency.
    """

    EMERGENCY_KEYWORDS = {
        "unconscious": (
            "unconscious",
            "unresponsive",
            "not responding",
        ),

        "breathing_difficulty": (
            "difficulty breathing",
            "cannot breathe",
            "not breathing",
            "severe breathlessness",
        ),

        "severe_bleeding": (
            "severe bleeding",
            "uncontrolled bleeding",
            "heavy bleeding",
        ),

        "chest_pain": (
            "chest pain",
            "chest pressure",
        ),

        "seizure": (
            "seizure",
            "convulsions",
        ),

        "drowning": (
            "drowning",
            "near drowning",
            "submersion",
        ),

        "stroke_signs": (
            "face drooping",
            "slurred speech",
            "sudden one-sided weakness",
        ),
    }

    HIGH_CONCERN_INCIDENTS = {
        MedicalIncidentType.DROWNING_EXPOSURE,
        MedicalIncidentType.RESPIRATORY_DISTRESS,
    }

    @classmethod
    def assess(
        cls,
        request: MedicalAssistanceRequest,
    ) -> MedicalUrgencyAssessment:
        text = " ".join(
            request.symptoms
        ).casefold()

        emergency_flags = []

        for flag, keywords in (
            cls.EMERGENCY_KEYWORDS.items()
        ):
            if any(
                keyword in text
                for keyword in keywords
            ):
                emergency_flags.append(
                    flag
                )

        vulnerability_flags = []

        if request.children_count > 0:
            vulnerability_flags.append(
                "children_present"
            )

        if request.older_adults_count > 0:
            vulnerability_flags.append(
                "older_adults_present"
            )

        if request.pregnant_people_count > 0:
            vulnerability_flags.append(
                "pregnancy_related_support"
            )

        if request.people_with_disabilities_count > 0:
            vulnerability_flags.append(
                "disability_access_support"
            )

        if request.requires_accessible_transport:
            vulnerability_flags.append(
                "accessible_transport_required"
            )

        emergency = (
            bool(emergency_flags)
            or request.incident_type
            in cls.HIGH_CONCERN_INCIDENTS
        )

        if emergency:
            urgency = MedicalUrgency.EMERGENCY_FLAG

        elif (
            vulnerability_flags
            or request.incident_type in {
                MedicalIncidentType.MATERNAL_HEALTH_CONCERN,
                MedicalIncidentType.CHRONIC_CARE_INTERRUPTION,
                MedicalIncidentType.FLOOD_INJURY,
            }
        ):
            urgency = MedicalUrgency.PRIORITY_REVIEW

        elif request.symptoms:
            urgency = MedicalUrgency.STANDARD_REVIEW

        else:
            urgency = MedicalUrgency.UNDETERMINED

        warnings = [
            "Automated symptom screening is not medical diagnosis",
            "Urgency labels require qualified clinical review",
            "Missing keywords do not rule out an emergency",
        ]

        if emergency:
            warnings.append(
                "Potential emergency: promptly escalate "
                "to authorized emergency medical personnel"
            )

        return MedicalUrgencyAssessment(
            urgency=urgency,
            detected_emergency_flags=tuple(
                emergency_flags
            ),
            vulnerability_flags=tuple(
                vulnerability_flags
            ),
            requires_immediate_human_escalation=emergency,
            warnings=tuple(warnings),
        )


# ============================================================
# HEALTHCARE FACILITY SCREENING
# ============================================================

class MedicalFacilityScreeningEngine:
    def __init__(
        self,
        config: MedicalResponseConfig,
    ) -> None:
        self.config = config

    def screen(
        self,
        *,
        request: MedicalAssistanceRequest,
        facilities: Sequence[MedicalFacility],
        now: datetime,
    ) -> tuple[
        MedicalFacilityScreeningResult,
        ...
    ]:
        if len(facilities) > (
            self.config.maximum_facilities
        ):
            raise MedicalResourceError(
                "Facility limit exceeded"
            )

        results = []

        for facility in facilities:
            reasons = []

            if (
                facility.district.casefold()
                != request.district.casefold()
            ):
                reasons.append(
                    "Facility outside requested district"
                )

            if facility.status != (
                MedicalFacilityStatus.VERIFIED_OPERATIONAL
            ):
                reasons.append(
                    "Facility operational status not fully verified"
                )

            if facility.verified_at is None:
                reasons.append(
                    "Missing facility verification timestamp"
                )

            elif (
                facility.verified_at > now
                or now - facility.verified_at
                > timedelta(
                    minutes=(
                        self.config.maximum_facility_age_minutes
                    )
                )
            ):
                reasons.append(
                    "Facility verification stale or invalid"
                )

            if (
                request.incident_type
                == MedicalIncidentType.MATERNAL_HEALTH_CONCERN
                and not facility.maternal_services
            ):
                reasons.append(
                    "Maternal services not confirmed"
                )

            if (
                request.requires_accessible_transport
                and not facility.accessible_facilities
            ):
                reasons.append(
                    "Accessible facilities not confirmed"
                )

            if (
                request.incident_type in {
                    MedicalIncidentType.DROWNING_EXPOSURE,
                    MedicalIncidentType.RESPIRATORY_DISTRESS,
                }
                and not facility.emergency_services
            ):
                reasons.append(
                    "Emergency services not confirmed"
                )

            results.append(
                MedicalFacilityScreeningResult(
                    facility_id=facility.facility_id,
                    facility_name=facility.facility_name,
                    eligible_for_consideration=(
                        len(reasons) == 0
                    ),
                    reasons=tuple(reasons),
                )
            )

        return tuple(results)


# ============================================================
# AMBULANCE SCREENING
# ============================================================

class MedicalAmbulanceScreeningEngine:
    def __init__(
        self,
        config: MedicalResponseConfig,
    ) -> None:
        self.config = config

    def screen(
        self,
        *,
        request: MedicalAssistanceRequest,
        ambulances: Sequence[AmbulanceResource],
        now: datetime,
    ) -> tuple[
        AmbulanceScreeningResult,
        ...
    ]:
        if len(ambulances) > (
            self.config.maximum_ambulances
        ):
            raise MedicalResourceError(
                "Ambulance limit exceeded"
            )

        results = []

        for ambulance in ambulances:
            reasons = []

            if (
                ambulance.district.casefold()
                != request.district.casefold()
            ):
                reasons.append(
                    "Ambulance outside requested district"
                )

            if ambulance.status != (
                AmbulanceStatus.VERIFIED_AVAILABLE
            ):
                reasons.append(
                    "Ambulance availability not verified"
                )

            if ambulance.verified_at is None:
                reasons.append(
                    "Missing ambulance verification timestamp"
                )

            elif (
                ambulance.verified_at > now
                or now - ambulance.verified_at
                > timedelta(
                    minutes=(
                        self.config.maximum_ambulance_age_minutes
                    )
                )
            ):
                reasons.append(
                    "Ambulance verification stale or invalid"
                )

            if (
                request.requires_accessible_transport
                and not ambulance.accessible_transport
            ):
                reasons.append(
                    "Accessible transport not confirmed"
                )

            results.append(
                AmbulanceScreeningResult(
                    ambulance_id=ambulance.ambulance_id,
                    eligible_for_consideration=(
                        len(reasons) == 0
                    ),
                    reasons=tuple(reasons),
                )
            )

        return tuple(results)


# ============================================================
# MEDICAL SUPPLY GAP ENGINE
# ============================================================

class MedicalSupplyGapEngine:
    def __init__(
        self,
        config: MedicalResponseConfig,
    ) -> None:
        self.config = config

    def calculate(
        self,
        items: Sequence[MedicalSupplyRequirement],
    ) -> tuple[MedicalSupplyGap, ...]:
        if len(items) > (
            self.config.maximum_supply_items
        ):
            raise MedicalResourceError(
                "Supply item limit exceeded"
            )

        seen = set()
        results = []

        for item in items:
            if item.item_id in seen:
                raise MedicalValidationError(
                    "Duplicate medical supply item ID"
                )

            seen.add(item.item_id)

            results.append(
                MedicalSupplyGap(
                    item_id=item.item_id,
                    category=item.category,
                    requested_quantity=(
                        item.requested_quantity
                    ),
                    available_quantity=(
                        item.available_quantity
                    ),
                    shortage_quantity=max(
                        0,
                        item.requested_quantity
                        - item.available_quantity,
                    ),
                    unit=item.unit,
                )
            )

        return tuple(results)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class MedicalResponseRecommendationEngine:
    @staticmethod
    def generate(
        *,
        urgency: MedicalUrgencyAssessment,
        facilities: Sequence[
            MedicalFacilityScreeningResult
        ],
        ambulances: Sequence[
            AmbulanceScreeningResult
        ],
        supply_gaps: Sequence[
            MedicalSupplyGap
        ],
    ) -> tuple[str, ...]:
        recommendations = []

        if urgency.requires_immediate_human_escalation:
            recommendations.append(
                "Immediately flag this request to an "
                "authorized emergency medical coordinator"
            )

            recommendations.append(
                "Use established emergency-service channels "
                "without waiting for automated assessment"
            )

        elif urgency.urgency == (
            MedicalUrgency.PRIORITY_REVIEW
        ):
            recommendations.append(
                "Prioritize qualified medical review"
            )

        else:
            recommendations.append(
                "Arrange appropriate medical intake review"
            )

        if not any(
            item.eligible_for_consideration
            for item in facilities
        ):
            recommendations.append(
                "Verify healthcare facility availability "
                "through authorized personnel"
            )

        if not any(
            item.eligible_for_consideration
            for item in ambulances
        ):
            recommendations.append(
                "Verify emergency transport availability "
                "with the responsible dispatch authority"
            )

        if any(
            item.shortage_quantity > 0
            for item in supply_gaps
        ):
            recommendations.append(
                "Review medical supply shortages "
                "with authorized logistics staff"
            )

        recommendations.extend([
            "Protect patient identity and health information",
            "Do not prescribe treatment from this assessment",
            "Do not dispatch medical resources automatically",
        ])

        return tuple(
            dict.fromkeys(recommendations)
        )


# ============================================================
# DEDUPLICATION
# ============================================================

class MedicalRequestDeduplicationEngine:
    def __init__(
        self,
        config: MedicalResponseConfig,
    ) -> None:
        self.config = config

        self._recent: dict[
            str,
            datetime,
        ] = {}

        self._lock = threading.Lock()

    def check(
        self,
        *,
        request: MedicalAssistanceRequest,
        now: datetime,
    ) -> bool:
        # Deduplication uses request ID only.
        # Similar symptoms or locations must not
        # cause separate patients to be merged.
        key = hashlib.sha256(
            request.request_id.encode("utf-8")
        ).hexdigest()

        cutoff = now - timedelta(
            minutes=(
                self.config.deduplication_window_minutes
            )
        )

        with self._lock:
            self._recent = {
                item_key: timestamp
                for item_key, timestamp in self._recent.items()
                if timestamp >= cutoff
            }

            previous = self._recent.get(key)

            duplicate = (
                previous is not None
                and previous >= cutoff
            )

            if not duplicate:
                self._recent[key] = now

        return duplicate


# ============================================================
# AUDIT LOGGER
# ============================================================

class MedicalResponseAuditLogger:
    def __init__(self) -> None:
        self._records: list[
            MedicalAuditRecord
        ] = []

        self._lock = threading.Lock()

    def record(
        self,
        *,
        plan_id: str,
        action: str,
        actor_id: str,
        details: Mapping[str, str] | None = None,
    ) -> MedicalAuditRecord:
        record = MedicalAuditRecord(
            audit_id=generate_id(),
            plan_id=plan_id,
            action=action,
            actor_id=actor_id,
            timestamp=utc_now(),
            details=dict(details or {}),
        )

        with self._lock:
            self._records.append(record)

        return record

    def list_records(
        self,
        plan_id: str | None = None,
    ) -> tuple[MedicalAuditRecord, ...]:
        with self._lock:
            records = tuple(self._records)

        if plan_id is None:
            return records

        return tuple(
            record
            for record in records
            if record.plan_id == plan_id
        )


# ============================================================
# HUMAN MEDICAL REVIEW
# ============================================================

class MedicalResponseReviewManager:
    ALLOWED_ROLES = {
        "medical_officer",
        "emergency_medical_coordinator",
        "authorized_clinician",
    }

    def __init__(
        self,
        audit_logger: MedicalResponseAuditLogger,
    ) -> None:
        self.audit_logger = audit_logger

        self._reviews: dict[
            str,
            MedicalResponseReview,
        ] = {}

        self._lock = threading.Lock()

    def review(
        self,
        plan: MedicalResponsePlan,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: MedicalReviewDecision,
        reason: str,
    ) -> MedicalResponseReview:
        if not reviewer_id.strip():
            raise MedicalApprovalError(
                "Reviewer ID required"
            )

        if reviewer_role not in self.ALLOWED_ROLES:
            raise MedicalApprovalError(
                "Reviewer role not permitted"
            )

        if not reason.strip():
            raise MedicalApprovalError(
                "Review reason required"
            )

        if plan.status != (
            MedicalCaseStatus.PENDING_MEDICAL_REVIEW
        ):
            raise MedicalApprovalError(
                "Plan is not pending medical review"
            )

        with self._lock:
            if plan.plan_id in self._reviews:
                raise MedicalApprovalError(
                    "Plan already reviewed"
                )

            review = MedicalResponseReview(
                review_id=generate_id(),
                plan_id=plan.plan_id,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
                reviewed_at=utc_now(),
            )

            self._reviews[
                plan.plan_id
            ] = review

        self.audit_logger.record(
            plan_id=plan.plan_id,
            action=f"medical_review_{decision.value}",
            actor_id=reviewer_id,
            details={
                "reviewer_role": reviewer_role,
                "reason": reason,
            },
        )

        return review


# ============================================================
# MAIN MEDICAL RESPONSE AGENT
# ============================================================

class MedicalResponseAgent:
    def __init__(
        self,
        config: MedicalResponseConfig | None = None,
    ) -> None:
        self.config = (
            config or MedicalResponseConfig()
        )

        self.urgency_engine = (
            MedicalUrgencyScreeningEngine()
        )

        self.facility_engine = (
            MedicalFacilityScreeningEngine(
                self.config
            )
        )

        self.ambulance_engine = (
            MedicalAmbulanceScreeningEngine(
                self.config
            )
        )

        self.supply_engine = (
            MedicalSupplyGapEngine(
                self.config
            )
        )

        self.recommendation_engine = (
            MedicalResponseRecommendationEngine()
        )

        self.deduplication_engine = (
            MedicalRequestDeduplicationEngine(
                self.config
            )
        )

        self.audit_logger = (
            MedicalResponseAuditLogger()
        )

        self.review_manager = (
            MedicalResponseReviewManager(
                self.audit_logger
            )
        )

        self._plans: dict[
            str,
            MedicalResponsePlan,
        ] = {}

        self._lock = threading.Lock()

    def run(
        self,
        request: MedicalResponsePlanningRequest,
        *,
        now: datetime | None = None,
    ) -> MedicalResponsePlan:
        now = now or utc_now()

        require_aware_datetime(now)
        require_aware_datetime(
            request.created_at
        )

        if not request.request_id.strip():
            raise MedicalValidationError(
                "request_id is required"
            )

        assistance = (
            request.assistance_request
        )

        urgency = (
            self.urgency_engine.assess(
                assistance
            )
        )

        facilities = (
            self.facility_engine.screen(
                request=assistance,
                facilities=request.facilities,
                now=now,
            )
        )

        ambulances = (
            self.ambulance_engine.screen(
                request=assistance,
                ambulances=request.ambulances,
                now=now,
            )
        )

        supply_gaps = (
            self.supply_engine.calculate(
                request.supply_requirements
            )
        )

        recommendations = (
            self.recommendation_engine.generate(
                urgency=urgency,
                facilities=facilities,
                ambulances=ambulances,
                supply_gaps=supply_gaps,
            )
        )

        missing_information = []

        if not assistance.symptoms:
            missing_information.append(
                "Clinical symptom description"
            )

        if not any(
            facility.eligible_for_consideration
            for facility in facilities
        ):
            missing_information.append(
                "Verified healthcare facility availability"
            )

        if not any(
            ambulance.eligible_for_consideration
            for ambulance in ambulances
        ):
            missing_information.append(
                "Verified ambulance availability"
            )

        duplicate = (
            self.deduplication_engine.check(
                request=assistance,
                now=now,
            )
        )

        warnings = list(
            urgency.warnings
        )

        warnings.extend([
            "Medical coordination draft only",
            "No clinical diagnosis has been performed",
            "No medicine or treatment has been prescribed",
            "No ambulance has been dispatched",
            "No hospital bed has been reserved",
            "Facility availability can change rapidly",
            "Qualified medical review is required",
            "Patient information must be protected",
        ])

        if duplicate:
            warnings.append(
                "This request ID was recently processed"
            )

        plan = MedicalResponsePlan(
            plan_id=generate_id(),
            request_id=request.request_id,
            district=assistance.district,
            incident_type=assistance.incident_type,
            urgency_assessment=urgency,
            facility_screening=facilities,
            ambulance_screening=ambulances,
            supply_gaps=supply_gaps,
            recommendations=recommendations,
            missing_information=tuple(
                missing_information
            ),
            warnings=tuple(warnings),
            status=(
                MedicalCaseStatus.PENDING_MEDICAL_REVIEW
            ),
            requires_human_medical_review=True,
            duplicate_detected=duplicate,
            created_at=now,
        )

        with self._lock:
            self._plans[
                plan.plan_id
            ] = plan

        # Audit records deliberately exclude
        # symptoms and identifying details.
        self.audit_logger.record(
            plan_id=plan.plan_id,
            action="medical_plan_created",
            actor_id="medical_response_agent",
            details={
                "incident_type": (
                    assistance.incident_type.value
                ),
                "urgency": urgency.urgency.value,
                "emergency_flag": str(
                    urgency.requires_immediate_human_escalation
                ),
            },
        )

        return plan

    def review_plan(
        self,
        plan_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: MedicalReviewDecision,
        reason: str,
    ) -> MedicalResponsePlan:
        with self._lock:
            plan = self._plans.get(
                plan_id
            )

            if plan is None:
                raise MedicalApprovalError(
                    "Plan not found"
                )

            self.review_manager.review(
                plan,
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                decision=decision,
                reason=reason,
            )

            updated = replace(
                plan,
                status=(
                    MedicalCaseStatus.REJECTED
                    if decision
                    == MedicalReviewDecision.REJECT_AS_INVALID
                    else MedicalCaseStatus.REVIEWED
                ),
            )

            self._plans[
                plan_id
            ] = updated

            return updated

    def get_plan(
        self,
        plan_id: str,
    ) -> MedicalResponsePlan | None:
        with self._lock:
            return self._plans.get(
                plan_id
            )

    def get_audit_history(
        self,
        plan_id: str,
    ) -> tuple[MedicalAuditRecord, ...]:
        return self.audit_logger.list_records(
            plan_id
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class MedicalResponseReportExporter:
    @staticmethod
    def export_json(
        plan: MedicalResponsePlan,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise MedicalExportError(
                "Report already exists"
            )

        with destination.open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                plan.to_dict(),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return destination


# ============================================================
# FACTORY
# ============================================================

def create_medical_response_agent(
    config: MedicalResponseConfig | None = None,
) -> MedicalResponseAgent:
    return MedicalResponseAgent(
        config=config
    )


# ============================================================
# EXAMPLE
# ============================================================

def main() -> None:
    now = utc_now()

    assistance_request = MedicalAssistanceRequest(
        request_id="MEDICAL-REQUEST-001",
        district="Thrissur",
        location_description="Illustrative flood relief zone",
        incident_type=(
            MedicalIncidentType.RESPIRATORY_DISTRESS
        ),
        symptoms=(
            "Difficulty breathing",
        ),
        affected_people=1,
        older_adults_count=1,
        requested_at=now,
    )

    request = MedicalResponsePlanningRequest(
        request_id=generate_id(),
        assistance_request=assistance_request,
        facilities=(
            MedicalFacility(
                facility_id="HOSPITAL-001",
                facility_name="Illustrative Hospital",
                district="Thrissur",
                status=(
                    MedicalFacilityStatus.VERIFIED_OPERATIONAL
                ),
                verified_at=now - timedelta(minutes=10),
                emergency_services=True,
                inpatient_services=True,
                available_beds=12,
            ),
        ),
        ambulances=(
            AmbulanceResource(
                ambulance_id="AMBULANCE-001",
                district="Thrissur",
                status=AmbulanceStatus.VERIFIED_AVAILABLE,
                verified_at=now - timedelta(minutes=5),
                advanced_life_support=True,
            ),
        ),
        supply_requirements=(
            MedicalSupplyRequirement(
                item_id="SUPPLY-001",
                category=MedicalSupplyCategory.FIRST_AID,
                description="Illustrative first-aid kits",
                requested_quantity=20,
                available_quantity=12,
                unit="kit",
            ),
        ),
        created_at=now,
    )

    agent = create_medical_response_agent()

    plan = agent.run(
        request,
        now=now,
    )

    print(
        json.dumps(
            plan.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    MedicalResponseReportExporter.export_json(
        plan,
        "outputs/agents/medical_response_report.json",
    )

    # Example administrative review only.
    # Does not dispatch an ambulance or
    # authorize any clinical intervention.
    reviewed = agent.review_plan(
        plan.plan_id,
        reviewer_id="medical-officer-001",
        reviewer_role="medical_officer",
        decision=MedicalReviewDecision.ACKNOWLEDGED,
        reason="Case flagged for clinical coordination",
    )

    print(
        "Review status:",
        reviewed.status.value,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "MedicalResponseError",
    "MedicalValidationError",
    "MedicalResourceError",
    "MedicalApprovalError",
    "MedicalExportError",
    "MedicalIncidentType",
    "MedicalUrgency",
    "MedicalCaseStatus",
    "MedicalFacilityStatus",
    "AmbulanceStatus",
    "MedicalSupplyCategory",
    "MedicalReviewDecision",
    "MedicalResponseConfig",
    "MedicalAssistanceRequest",
    "MedicalFacility",
    "AmbulanceResource",
    "MedicalSupplyRequirement",
    "MedicalResponsePlanningRequest",
    "MedicalUrgencyAssessment",
    "MedicalFacilityScreeningResult",
    "AmbulanceScreeningResult",
    "MedicalSupplyGap",
    "MedicalResponsePlan",
    "MedicalResponseReview",
    "MedicalAuditRecord",
    "MedicalUrgencyScreeningEngine",
    "MedicalFacilityScreeningEngine",
    "MedicalAmbulanceScreeningEngine",
    "MedicalSupplyGapEngine",
    "MedicalResponseRecommendationEngine",
    "MedicalRequestDeduplicationEngine",
    "MedicalResponseAuditLogger",
    "MedicalResponseReviewManager",
    "MedicalResponseAgent",
    "MedicalResponseReportExporter",
    "create_medical_response_agent",
]
