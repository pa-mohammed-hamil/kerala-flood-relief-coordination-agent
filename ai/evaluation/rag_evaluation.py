"""Placeholder module: ai/evaluation/rag_evaluation.py; implementation required."""

"""
Kerala Flood Relief Coordination Agent
Advanced RAG Evaluation and Benchmarking

File:
    backend/ai/evaluation/rag_evaluation.py

Python:
    3.11+

Dependencies:
    Standard library only

Features:
    - Retrieval Precision@K
    - Retrieval Recall@K
    - Hit Rate@K
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
    - Mean Average Precision (MAP)
    - Context relevance
    - Context precision and recall proxies
    - Answer faithfulness evaluation
    - Citation correctness
    - Citation coverage
    - Hallucination indicators
    - Answer correctness against ground truth
    - GraphRAG evidence evaluation
    - Hybrid retrieval benchmarking
    - Multi-query RAG benchmarking
    - Reranker evaluation
    - RAG latency and cost evaluation
    - Safety and human review gates
    - Batch evaluation
    - Regression comparison
    - JSON report export

Important:
    Semantic relevance, faithfulness, and
    correctness require independently supplied
    judgments or trusted evaluation datasets.

    This module does not independently establish
    factual truth from arbitrary generated text.

    Emergency-response answers remain advisory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class RAGEvaluationError(Exception):
    """Base RAG evaluation exception."""


class RAGValidationError(RAGEvaluationError):
    """Invalid evaluation input."""


class RAGBenchmarkError(RAGEvaluationError):
    """Benchmark evaluation failure."""


class RAGExportError(RAGEvaluationError):
    """Report export failure."""


# ============================================================
# ENUMERATIONS
# ============================================================

class RAGPipelineType(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    RERANKED = "reranked"
    MULTI_QUERY = "multi_query"
    GRAPH_RAG = "graph_rag"
    PARENT_CHILD = "parent_child"
    CORRECTIVE_RAG = "corrective_rag"
    SELF_RAG = "self_rag"
    AGENTIC_RAG = "agentic_rag"


class RAGEvaluationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_DATA = "insufficient_data"


class RAGRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RAGClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class RAGCitationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNVERIFIED = "unverified"


class RAGFindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class RAGFindingType(str, Enum):
    LOW_RETRIEVAL_RECALL = "low_retrieval_recall"
    LOW_RETRIEVAL_PRECISION = "low_retrieval_precision"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTED_CLAIM = "contradicted_claim"
    MISSING_CITATION = "missing_citation"
    INVALID_CITATION = "invalid_citation"
    UNVERIFIED_EVIDENCE = "unverified_evidence"
    INCORRECT_ANSWER = "incorrect_answer"
    UNSAFE_ANSWER = "unsafe_answer"
    MISSING_HUMAN_REVIEW = "missing_human_review"
    HIGH_LATENCY = "high_latency"
    HIGH_COST = "high_cost"
    INSUFFICIENT_GROUND_TRUTH = "insufficient_ground_truth"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator <= 0:
        return None

    return numerator / denominator


def average(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return float(statistics.mean(values))


def validate_fraction(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise RAGValidationError(
            f"{name} must be between 0 and 1"
        )


def validate_nonnegative(
    value: float,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RAGValidationError(
            f"{name} must be finite and nonnegative"
        )


def to_serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return to_serializable(
            asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(key): to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_serializable(item)
            for item in value
        ]

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RAGEvaluationConfig:
    retrieval_k: int = 10

    minimum_recall_at_k: float = 0.70

    minimum_precision_at_k: float = 0.40

    minimum_faithfulness: float = 0.90

    minimum_citation_accuracy: float = 0.90

    minimum_answer_correctness: float = 0.80

    maximum_latency_ms: float = 10000.0

    maximum_cost_usd: float = 0.10

    require_verified_sources: bool = True

    require_human_review_for_critical: bool = True

    fail_on_unsafe_answer: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.retrieval_k, int)
            or isinstance(self.retrieval_k, bool)
            or self.retrieval_k < 1
        ):
            raise RAGValidationError(
                "retrieval_k must be positive"
            )

        for name in (
            "minimum_recall_at_k",
            "minimum_precision_at_k",
            "minimum_faithfulness",
            "minimum_citation_accuracy",
            "minimum_answer_correctness",
        ):
            validate_fraction(
                getattr(self, name),
                name,
            )

        validate_nonnegative(
            self.maximum_latency_ms,
            "maximum_latency_ms",
        )

        validate_nonnegative(
            self.maximum_cost_usd,
            "maximum_cost_usd",
        )


# ============================================================
# INPUT MODELS
# ============================================================

@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    content: str

    rank: int

    retrieval_score: float | None = None

    relevance_grade: int | None = None

    source_verified: bool = False

    source_type: str = "document"

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise RAGValidationError(
                "document_id is required"
            )

        if self.rank < 1:
            raise RAGValidationError(
                "Document rank must be positive"
            )

        if (
            self.relevance_grade is not None
            and self.relevance_grade not in (0, 1, 2, 3)
        ):
            raise RAGValidationError(
                "Relevance grade must be 0, 1, 2 or 3"
            )

        if self.retrieval_score is not None:
            if not math.isfinite(self.retrieval_score):
                raise RAGValidationError(
                    "Retrieval score must be finite"
                )


@dataclass(frozen=True)
class RAGClaim:
    claim_id: str
    text: str

    status: RAGClaimStatus

    citation_ids: tuple[str, ...] = ()

    safety_critical: bool = False


@dataclass(frozen=True)
class RAGCitation:
    citation_id: str
    document_id: str

    status: RAGCitationStatus

    supports_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RAGGroundTruth:
    relevant_document_ids: frozenset[str]

    expected_answer: str | None = None

    answer_correctness: float | None = None

    def __post_init__(self) -> None:
        if self.answer_correctness is not None:
            validate_fraction(
                self.answer_correctness,
                "answer_correctness",
            )


@dataclass(frozen=True)
class RAGEvaluationInput:
    query_id: str

    question: str

    answer: str

    pipeline_type: RAGPipelineType

    retrieved_documents: tuple[
        RetrievedDocument,
        ...
    ]

    claims: tuple[RAGClaim, ...] = ()

    citations: tuple[RAGCitation, ...] = ()

    ground_truth: RAGGroundTruth | None = None

    latency_ms: float | None = None

    estimated_cost_usd: float | None = None

    risk_level: RAGRiskLevel = RAGRiskLevel.LOW

    human_review_completed: bool = False

    safety_violation_detected: bool = False

    generated_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise RAGValidationError(
                "query_id is required"
            )

        if not self.question.strip():
            raise RAGValidationError(
                "question is required"
            )

        document_ids = [
            document.document_id
            for document in self.retrieved_documents
        ]

        ranks = [
            document.rank
            for document in self.retrieved_documents
        ]

        if len(document_ids) != len(set(document_ids)):
            raise RAGValidationError(
                "Duplicate retrieved document IDs"
            )

        if len(ranks) != len(set(ranks)):
            raise RAGValidationError(
                "Duplicate document ranks"
            )

        for name in (
            "latency_ms",
            "estimated_cost_usd",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_nonnegative(
                    value,
                    name,
                )


# ============================================================
# RETRIEVAL METRICS
# ============================================================

@dataclass(frozen=True)
class RAGRetrievalMetrics:
    k: int

    retrieved_count: int

    relevant_document_count: int

    precision_at_k: float

    recall_at_k: float | None

    hit_rate_at_k: float | None

    reciprocal_rank: float | None

    average_precision: float | None

    ndcg_at_k: float | None


class RAGRetrievalEvaluator:
    """
    Computes retrieval ranking metrics.

    Relevance:
        Ground-truth document IDs determine
        binary relevance.

    NDCG:
        Uses externally supplied relevance
        grades when available, otherwise
        binary relevance.

    Missing relevance judgments should not
    be treated as independently verified.
    """

    def evaluate(
        self,
        documents: Sequence[RetrievedDocument],
        relevant_document_ids: frozenset[str],
        *,
        k: int = 10,
    ) -> RAGRetrievalMetrics:
        if k < 1:
            raise RAGValidationError(
                "k must be positive"
            )

        ranked = sorted(
            documents,
            key=lambda item: item.rank,
        )

        top_k = ranked[:k]

        binary_relevance = [
            1 if document.document_id
            in relevant_document_ids
            else 0
            for document in top_k
        ]

        relevant_retrieved = sum(
            binary_relevance
        )

        precision = (
            relevant_retrieved / k
        )

        recall = safe_divide(
            relevant_retrieved,
            len(relevant_document_ids),
        )

        hit_rate = (
            float(relevant_retrieved > 0)
            if relevant_document_ids
            else None
        )

        reciprocal_rank = None

        for position, relevant in enumerate(
            binary_relevance,
            start=1,
        ):
            if relevant:
                reciprocal_rank = 1.0 / position
                break

        if relevant_document_ids:
            relevant_so_far = 0
            precision_sum = 0.0

            for position, relevant in enumerate(
                binary_relevance,
                start=1,
            ):
                if relevant:
                    relevant_so_far += 1

                    precision_sum += (
                        relevant_so_far / position
                    )

            average_precision = (
                precision_sum
                / len(relevant_document_ids)
            )

        else:
            average_precision = None

        grades = [
            (
                document.relevance_grade
                if document.relevance_grade is not None
                else int(
                    document.document_id
                    in relevant_document_ids
                )
            )
            for document in top_k
        ]

        def dcg(
            relevance: Sequence[int],
        ) -> float:
            return sum(
                (2 ** grade - 1)
                / math.log2(index + 2)
                for index, grade in enumerate(
                    relevance
                )
            )

        actual_dcg = dcg(grades)

        ideal_grades = sorted(
            grades,
            reverse=True,
        )

        ideal_dcg = dcg(
            ideal_grades
        )

        ndcg = (
            actual_dcg / ideal_dcg
            if ideal_dcg > 0
            else None
        )

        return RAGRetrievalMetrics(
            k=k,
            retrieved_count=len(top_k),
            relevant_document_count=len(
                relevant_document_ids
            ),
            precision_at_k=precision,
            recall_at_k=recall,
            hit_rate_at_k=hit_rate,
            reciprocal_rank=reciprocal_rank,
            average_precision=average_precision,
            ndcg_at_k=ndcg,
        )


# ============================================================
# CONTEXT EVALUATION
# ============================================================

@dataclass(frozen=True)
class RAGContextMetrics:
    evaluated_documents: int

    context_relevance: float | None

    verified_source_fraction: float | None

    relevant_context_fraction: float | None


class RAGContextEvaluator:
    def evaluate(
        self,
        documents: Sequence[RetrievedDocument],
    ) -> RAGContextMetrics:
        if not documents:
            return RAGContextMetrics(
                evaluated_documents=0,
                context_relevance=None,
                verified_source_fraction=None,
                relevant_context_fraction=None,
            )

        labeled = [
            document
            for document in documents
            if document.relevance_grade is not None
        ]

        relevance = (
            average([
                document.relevance_grade / 3
                for document in labeled
            ])
            if labeled
            else None
        )

        verified = sum(
            document.source_verified
            for document in documents
        )

        relevant = (
            sum(
                document.relevance_grade > 0
                for document in labeled
            ) / len(labeled)
            if labeled
            else None
        )

        return RAGContextMetrics(
            evaluated_documents=len(documents),
            context_relevance=relevance,
            verified_source_fraction=(
                verified / len(documents)
            ),
            relevant_context_fraction=relevant,
        )


# ============================================================
# FAITHFULNESS EVALUATION
# ============================================================

@dataclass(frozen=True)
class RAGFaithfulnessMetrics:
    total_claims: int

    supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    unverifiable_claims: int

    faithfulness_score: float | None

    hallucination_indicator: float | None


class RAGFaithfulnessEvaluator:
    """
    Uses independently labeled atomic claims.

    The score is not an LLM-based entailment
    assessment and does not extract claims
    from answer text.
    """

    def evaluate(
        self,
        claims: Sequence[RAGClaim],
    ) -> RAGFaithfulnessMetrics:
        total = len(claims)

        counts = {
            status: sum(
                claim.status == status
                for claim in claims
            )
            for status in RAGClaimStatus
        }

        supported = counts[
            RAGClaimStatus.SUPPORTED
        ]

        unsupported = counts[
            RAGClaimStatus.UNSUPPORTED
        ]

        contradicted = counts[
            RAGClaimStatus.CONTRADICTED
        ]

        unverifiable = counts[
            RAGClaimStatus.UNVERIFIABLE
        ]

        faithfulness = (
            supported / total
            if total
            else None
        )

        hallucination_indicator = (
            (unsupported + contradicted) / total
            if total
            else None
        )

        return RAGFaithfulnessMetrics(
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            unverifiable_claims=unverifiable,
            faithfulness_score=faithfulness,
            hallucination_indicator=(
                hallucination_indicator
            ),
        )


# ============================================================
# CITATION EVALUATION
# ============================================================

@dataclass(frozen=True)
class RAGCitationMetrics:
    total_citations: int

    valid_citations: int
    invalid_citations: int
    unverified_citations: int

    citation_accuracy: float | None

    claim_citation_coverage: float | None


class RAGCitationEvaluator:
    def evaluate(
        self,
        claims: Sequence[RAGClaim],
        citations: Sequence[RAGCitation],
    ) -> RAGCitationMetrics:
        total = len(citations)

        valid = sum(
            citation.status
            == RAGCitationStatus.VALID
            for citation in citations
        )

        invalid = sum(
            citation.status
            == RAGCitationStatus.INVALID
            for citation in citations
        )

        unverified = sum(
            citation.status
            == RAGCitationStatus.UNVERIFIED
            for citation in citations
        )

        citation_accuracy = (
            valid / total
            if total
            else None
        )

        citation_lookup = {
            citation.citation_id: citation
            for citation in citations
        }

        covered_claims = 0

        for claim in claims:
            has_valid_citation = any(
                citation_id in citation_lookup
                and citation_lookup[
                    citation_id
                ].status == RAGCitationStatus.VALID
                and claim.claim_id in citation_lookup[
                    citation_id
                ].supports_claim_ids
                for citation_id in claim.citation_ids
            )

            if has_valid_citation:
                covered_claims += 1

        claim_coverage = (
            covered_claims / len(claims)
            if claims
            else None
        )

        return RAGCitationMetrics(
            total_citations=total,
            valid_citations=valid,
            invalid_citations=invalid,
            unverified_citations=unverified,
            citation_accuracy=citation_accuracy,
            claim_citation_coverage=claim_coverage,
        )


# ============================================================
# ANSWER CORRECTNESS
# ============================================================

@dataclass(frozen=True)
class RAGAnswerMetrics:
    answer_correctness: float | None

    expected_answer_available: bool

    generated_answer_present: bool


class RAGAnswerEvaluator:
    """
    Answer correctness is an externally
    supplied independent score.

    Expected answer text is stored for
    audit, not compared semantically here.
    """

    def evaluate(
        self,
        answer: str,
        ground_truth: RAGGroundTruth | None,
    ) -> RAGAnswerMetrics:
        return RAGAnswerMetrics(
            answer_correctness=(
                ground_truth.answer_correctness
                if ground_truth
                else None
            ),
            expected_answer_available=bool(
                ground_truth
                and ground_truth.expected_answer
            ),
            generated_answer_present=bool(
                answer.strip()
            ),
        )


# ============================================================
# LATENCY AND COST
# ============================================================

@dataclass(frozen=True)
class RAGPerformanceMetrics:
    latency_ms: float | None

    estimated_cost_usd: float | None

    latency_within_budget: bool | None

    cost_within_budget: bool | None


class RAGPerformanceEvaluator:
    def __init__(
        self,
        config: RAGEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        evaluation_input: RAGEvaluationInput,
    ) -> RAGPerformanceMetrics:
        latency = evaluation_input.latency_ms

        cost = evaluation_input.estimated_cost_usd

        return RAGPerformanceMetrics(
            latency_ms=latency,
            estimated_cost_usd=cost,
            latency_within_budget=(
                latency <= self.config.maximum_latency_ms
                if latency is not None
                else None
            ),
            cost_within_budget=(
                cost <= self.config.maximum_cost_usd
                if cost is not None
                else None
            ),
        )


# ============================================================
# FINDINGS
# ============================================================

@dataclass(frozen=True)
class RAGEvaluationFinding:
    finding_id: str

    finding_type: RAGFindingType

    severity: RAGFindingSeverity

    message: str


def make_finding(
    finding_type: RAGFindingType,
    severity: RAGFindingSeverity,
    message: str,
) -> RAGEvaluationFinding:
    return RAGEvaluationFinding(
        finding_id=new_id(),
        finding_type=finding_type,
        severity=severity,
        message=message,
    )


# ============================================================
# EVALUATION REPORT
# ============================================================

@dataclass(frozen=True)
class RAGEvaluationReport:
    evaluation_id: str

    query_id: str

    pipeline_type: RAGPipelineType

    status: RAGEvaluationStatus

    retrieval: RAGRetrievalMetrics | None

    context: RAGContextMetrics

    faithfulness: RAGFaithfulnessMetrics

    citations: RAGCitationMetrics

    answer: RAGAnswerMetrics

    performance: RAGPerformanceMetrics

    findings: tuple[
        RAGEvaluationFinding,
        ...
    ]

    requires_human_review: bool

    evaluated_at: datetime

    answer_digest: str

    warnings: tuple[str, ...]


# ============================================================
# MAIN EVALUATION ENGINE
# ============================================================

class RAGEvaluationEngine:
    """
    Combines retrieval, grounding, citation,
    answer, performance and safety checks.

    A passing score requires configured
    quality gates and complete enough
    independent ground truth.

    Critical safety failures override scores.
    """

    def __init__(
        self,
        config: RAGEvaluationConfig | None = None,
    ) -> None:
        self.config = (
            config or RAGEvaluationConfig()
        )

        self.retrieval_evaluator = (
            RAGRetrievalEvaluator()
        )

        self.context_evaluator = (
            RAGContextEvaluator()
        )

        self.faithfulness_evaluator = (
            RAGFaithfulnessEvaluator()
        )

        self.citation_evaluator = (
            RAGCitationEvaluator()
        )

        self.answer_evaluator = (
            RAGAnswerEvaluator()
        )

        self.performance_evaluator = (
            RAGPerformanceEvaluator(
                self.config
            )
        )

    def evaluate(
        self,
        item: RAGEvaluationInput,
    ) -> RAGEvaluationReport:
        findings: list[
            RAGEvaluationFinding
        ] = []

        warnings: list[str] = []

        retrieval = None

        if item.ground_truth is not None:
            retrieval = self.retrieval_evaluator.evaluate(
                item.retrieved_documents,
                item.ground_truth.relevant_document_ids,
                k=self.config.retrieval_k,
            )

            if (
                retrieval.recall_at_k is not None
                and retrieval.recall_at_k
                < self.config.minimum_recall_at_k
            ):
                findings.append(
                    make_finding(
                        RAGFindingType.LOW_RETRIEVAL_RECALL,
                        RAGFindingSeverity.HIGH,
                        "Retrieval recall is below configured threshold",
                    )
                )

            if (
                retrieval.precision_at_k
                < self.config.minimum_precision_at_k
            ):
                findings.append(
                    make_finding(
                        RAGFindingType.LOW_RETRIEVAL_PRECISION,
                        RAGFindingSeverity.WARNING,
                        "Retrieval precision is below configured threshold",
                    )
                )

        else:
            warnings.append(
                "Retrieval ground truth unavailable"
            )

        context = self.context_evaluator.evaluate(
            item.retrieved_documents
        )

        faithfulness = (
            self.faithfulness_evaluator.evaluate(
                item.claims
            )
        )

        citations = (
            self.citation_evaluator.evaluate(
                item.claims,
                item.citations,
            )
        )

        answer = self.answer_evaluator.evaluate(
            item.answer,
            item.ground_truth,
        )

        performance = (
            self.performance_evaluator.evaluate(
                item
            )
        )

        if (
            faithfulness.faithfulness_score is not None
            and faithfulness.faithfulness_score
            < self.config.minimum_faithfulness
        ):
            findings.append(
                make_finding(
                    RAGFindingType.UNSUPPORTED_CLAIM,
                    RAGFindingSeverity.HIGH,
                    "Answer faithfulness below threshold",
                )
            )

        if faithfulness.contradicted_claims > 0:
            findings.append(
                make_finding(
                    RAGFindingType.CONTRADICTED_CLAIM,
                    RAGFindingSeverity.HIGH,
                    "Contradicted claims detected",
                )
            )

        if (
            citations.citation_accuracy is not None
            and citations.citation_accuracy
            < self.config.minimum_citation_accuracy
        ):
            findings.append(
                make_finding(
                    RAGFindingType.INVALID_CITATION,
                    RAGFindingSeverity.HIGH,
                    "Citation accuracy below threshold",
                )
            )

        if (
            answer.answer_correctness is not None
            and answer.answer_correctness
            < self.config.minimum_answer_correctness
        ):
            findings.append(
                make_finding(
                    RAGFindingType.INCORRECT_ANSWER,
                    RAGFindingSeverity.HIGH,
                    "Answer correctness below threshold",
                )
            )

        if (
            self.config.require_verified_sources
            and any(
                not document.source_verified
                for document in item.retrieved_documents
            )
        ):
            findings.append(
                make_finding(
                    RAGFindingType.UNVERIFIED_EVIDENCE,
                    RAGFindingSeverity.HIGH,
                    "Unverified retrieval sources detected",
                )
            )

        if (
            item.safety_violation_detected
            and self.config.fail_on_unsafe_answer
        ):
            findings.append(
                make_finding(
                    RAGFindingType.UNSAFE_ANSWER,
                    RAGFindingSeverity.CRITICAL,
                    "Unsafe answer detected",
                )
            )

        if (
            item.risk_level == RAGRiskLevel.CRITICAL
            and self.config.require_human_review_for_critical
            and not item.human_review_completed
        ):
            findings.append(
                make_finding(
                    RAGFindingType.MISSING_HUMAN_REVIEW,
                    RAGFindingSeverity.CRITICAL,
                    "Critical RAG answer lacks human review",
                )
            )

        if performance.latency_within_budget is False:
            findings.append(
                make_finding(
                    RAGFindingType.HIGH_LATENCY,
                    RAGFindingSeverity.WARNING,
                    "RAG latency exceeded budget",
                )
            )

        if performance.cost_within_budget is False:
            findings.append(
                make_finding(
                    RAGFindingType.HIGH_COST,
                    RAGFindingSeverity.WARNING,
                    "RAG estimated cost exceeded budget",
                )
            )

        required_scores = (
            retrieval.recall_at_k
            if retrieval is not None
            else None,
            faithfulness.faithfulness_score,
            citations.citation_accuracy,
            answer.answer_correctness,
        )

        missing_ground_truth = any(
            score is None
            for score in required_scores
        )

        if missing_ground_truth:
            warnings.append(
                "One or more independent evaluation labels are missing"
            )

        critical = any(
            finding.severity
            == RAGFindingSeverity.CRITICAL
            for finding in findings
        )

        high = any(
            finding.severity
            == RAGFindingSeverity.HIGH
            for finding in findings
        )

        requires_review = (
            item.risk_level
            in (
                RAGRiskLevel.HIGH,
                RAGRiskLevel.CRITICAL,
            )
            or high
            or critical
        )

        if critical or high:
            status = RAGEvaluationStatus.FAILED

        elif missing_ground_truth:
            status = (
                RAGEvaluationStatus.INSUFFICIENT_DATA
            )

        elif requires_review:
            status = (
                RAGEvaluationStatus.REVIEW_REQUIRED
            )

        else:
            status = RAGEvaluationStatus.PASSED

        report = RAGEvaluationReport(
            evaluation_id=new_id(),
            query_id=item.query_id,
            pipeline_type=item.pipeline_type,
            status=status,
            retrieval=retrieval,
            context=context,
            faithfulness=faithfulness,
            citations=citations,
            answer=answer,
            performance=performance,
            findings=tuple(findings),
            requires_human_review=requires_review,
            evaluated_at=utc_now(),
            answer_digest=sha256_text(
                item.answer
            ),
            warnings=tuple(warnings),
        )

        logger.info(
            "RAG evaluation completed: query=%s status=%s",
            item.query_id,
            status.value,
        )

        return report


# ============================================================
# BATCH EVALUATION
# ============================================================

@dataclass(frozen=True)
class RAGBatchEvaluationReport:
    batch_id: str

    total_queries: int

    passed_queries: int
    failed_queries: int

    review_required_queries: int
    insufficient_data_queries: int

    average_recall_at_k: float | None

    average_precision_at_k: float | None

    mean_reciprocal_rank: float | None

    mean_average_precision: float | None

    mean_ndcg_at_k: float | None

    average_faithfulness: float | None

    average_citation_accuracy: float | None

    average_answer_correctness: float | None

    average_latency_ms: float | None

    reports: tuple[
        RAGEvaluationReport,
        ...
    ]

    evaluated_at: datetime


class RAGBatchEvaluator:
    def __init__(
        self,
        engine: RAGEvaluationEngine,
    ) -> None:
        self.engine = engine

    def evaluate_batch(
        self,
        items: Sequence[RAGEvaluationInput],
    ) -> RAGBatchEvaluationReport:
        reports = tuple(
            self.engine.evaluate(item)
            for item in items
        )

        def values(
            extractor: Any,
        ) -> list[float]:
            results: list[float] = []

            for report in reports:
                value = extractor(report)

                if value is not None:
                    results.append(value)

            return results

        return RAGBatchEvaluationReport(
            batch_id=new_id(),
            total_queries=len(reports),
            passed_queries=sum(
                report.status
                == RAGEvaluationStatus.PASSED
                for report in reports
            ),
            failed_queries=sum(
                report.status
                == RAGEvaluationStatus.FAILED
                for report in reports
            ),
            review_required_queries=sum(
                report.status
                == RAGEvaluationStatus.REVIEW_REQUIRED
                for report in reports
            ),
            insufficient_data_queries=sum(
                report.status
                == RAGEvaluationStatus.INSUFFICIENT_DATA
                for report in reports
            ),
            average_recall_at_k=average(
                values(
                    lambda r: (
                        r.retrieval.recall_at_k
                        if r.retrieval
                        else None
                    )
                )
            ),
            average_precision_at_k=average(
                values(
                    lambda r: (
                        r.retrieval.precision_at_k
                        if r.retrieval
                        else None
                    )
                )
            ),
            mean_reciprocal_rank=average(
                values(
                    lambda r: (
                        r.retrieval.reciprocal_rank
                        if r.retrieval
                        else None
                    )
                )
            ),
            mean_average_precision=average(
                values(
                    lambda r: (
                        r.retrieval.average_precision
                        if r.retrieval
                        else None
                    )
                )
            ),
            mean_ndcg_at_k=average(
                values(
                    lambda r: (
                        r.retrieval.ndcg_at_k
                        if r.retrieval
                        else None
                    )
                )
            ),
            average_faithfulness=average(
                values(
                    lambda r: (
                        r.faithfulness.faithfulness_score
                    )
                )
            ),
            average_citation_accuracy=average(
                values(
                    lambda r: (
                        r.citations.citation_accuracy
                    )
                )
            ),
            average_answer_correctness=average(
                values(
                    lambda r: (
                        r.answer.answer_correctness
                    )
                )
            ),
            average_latency_ms=average(
                values(
                    lambda r: (
                        r.performance.latency_ms
                    )
                )
            ),
            reports=reports,
            evaluated_at=utc_now(),
        )


# ============================================================
# PIPELINE COMPARISON
# ============================================================

@dataclass(frozen=True)
class RAGPipelineComparison:
    baseline_pipeline: RAGPipelineType

    candidate_pipeline: RAGPipelineType

    metric_name: str

    baseline_score: float
    candidate_score: float

    score_delta: float

    regressed: bool


class RAGPipelineRegressionEvaluator:
    """
    Compare metrics computed from the same
    benchmark query set and relevance labels.
    """

    def compare(
        self,
        *,
        baseline_pipeline: RAGPipelineType,
        candidate_pipeline: RAGPipelineType,
        metric_name: str,
        baseline_score: float,
        candidate_score: float,
        higher_is_better: bool = True,
        tolerance: float = 0.02,
    ) -> RAGPipelineComparison:
        validate_nonnegative(
            tolerance,
            "tolerance",
        )

        for name, value in (
            ("baseline_score", baseline_score),
            ("candidate_score", candidate_score),
        ):
            if not math.isfinite(value):
                raise RAGValidationError(
                    f"{name} must be finite"
                )

        delta = (
            candidate_score - baseline_score
        )

        regressed = (
            delta < -tolerance
            if higher_is_better
            else delta > tolerance
        )

        return RAGPipelineComparison(
            baseline_pipeline=baseline_pipeline,
            candidate_pipeline=candidate_pipeline,
            metric_name=metric_name,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            score_delta=delta,
            regressed=regressed,
        )


# ============================================================
# REPORT EXPORTER
# ============================================================

class RAGEvaluationReportExporter:
    @staticmethod
    def to_dict(
        report: RAGEvaluationReport,
    ) -> dict[str, Any]:
        return to_serializable(
            report
        )

    @classmethod
    def to_json(
        cls,
        report: RAGEvaluationReport,
        *,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            cls.to_dict(report),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def save_json(
        cls,
        report: RAGEvaluationReport,
        destination: str | Path,
    ) -> Path:
        path = Path(destination)

        if path.suffix.lower() != ".json":
            raise RAGExportError(
                "Report filename must end in .json"
            )

        try:
            with path.open(
                "x",
                encoding="utf-8",
            ) as file:
                file.write(
                    cls.to_json(report)
                )

        except OSError as exc:
            raise RAGExportError(
                "Unable to export RAG report"
            ) from exc

        return path


# ============================================================
# SERVICE LAYER
# ============================================================

class RAGEvaluationService:
    """
    Application-facing RAG evaluation service.

    Integrations:
        backend/ai/rag/
        backend/ai/llm/llm_gateway.py
        backend/ai/evaluation/hallucination_checks.py
        backend/ai/evaluation/agent_evaluation.py
        backend/ai/evaluation/benchmark_runner.py
        backend/ai/safety/source_verification.py
        backend/mlops/
    """

    def __init__(
        self,
        engine: RAGEvaluationEngine | None = None,
    ) -> None:
        self.engine = (
            engine or RAGEvaluationEngine()
        )

        self.batch_evaluator = (
            RAGBatchEvaluator(
                self.engine
            )
        )

        self.regression_evaluator = (
            RAGPipelineRegressionEvaluator()
        )

    def evaluate(
        self,
        item: RAGEvaluationInput,
    ) -> RAGEvaluationReport:
        return self.engine.evaluate(
            item
        )

    def evaluate_batch(
        self,
        items: Sequence[RAGEvaluationInput],
    ) -> RAGBatchEvaluationReport:
        return self.batch_evaluator.evaluate_batch(
            items
        )

    def compare_pipelines(
        self,
        **kwargs: Any,
    ) -> RAGPipelineComparison:
        return self.regression_evaluator.compare(
            **kwargs
        )


# ============================================================
# FACTORIES
# ============================================================

def create_rag_evaluation_engine(
    config: RAGEvaluationConfig | None = None,
) -> RAGEvaluationEngine:
    return RAGEvaluationEngine(
        config=config
    )


def create_rag_evaluation_service(
    config: RAGEvaluationConfig | None = None,
) -> RAGEvaluationService:
    return RAGEvaluationService(
        engine=create_rag_evaluation_engine(
            config=config
        )
    )


__all__ = [
    "RAGEvaluationError",
    "RAGValidationError",
    "RAGBenchmarkError",
    "RAGExportError",
    "RAGPipelineType",
    "RAGEvaluationStatus",
    "RAGRiskLevel",
    "RAGClaimStatus",
    "RAGCitationStatus",
    "RAGFindingSeverity",
    "RAGFindingType",
    "RAGEvaluationConfig",
    "RetrievedDocument",
    "RAGClaim",
    "RAGCitation",
    "RAGGroundTruth",
    "RAGEvaluationInput",
    "RAGRetrievalMetrics",
    "RAGRetrievalEvaluator",
    "RAGContextMetrics",
    "RAGContextEvaluator",
    "RAGFaithfulnessMetrics",
    "RAGFaithfulnessEvaluator",
    "RAGCitationMetrics",
    "RAGCitationEvaluator",
    "RAGAnswerMetrics",
    "RAGAnswerEvaluator",
    "RAGPerformanceMetrics",
    "RAGPerformanceEvaluator",
    "RAGEvaluationFinding",
    "RAGEvaluationReport",
    "RAGEvaluationEngine",
    "RAGBatchEvaluationReport",
    "RAGBatchEvaluator",
    "RAGPipelineComparison",
    "RAGPipelineRegressionEvaluator",
    "RAGEvaluationReportExporter",
    "RAGEvaluationService",
    "create_rag_evaluation_engine",
    "create_rag_evaluation_service",
]
