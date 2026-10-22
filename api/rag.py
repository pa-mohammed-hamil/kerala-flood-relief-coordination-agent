from __future__ import annotations

"""
backend/api/rag.py

Kerala Flood Relief Coordination Agent
Advanced Retrieval-Augmented Generation API

Python 3.11+ | FastAPI | Pydantic v2

Features:
    - Disaster-management knowledge assistant
    - Hybrid BM25 + vector retrieval
    - Semantic search
    - Query rewriting and multi-query retrieval
    - Reranking
    - Document ingestion
    - Source citations
    - Multilingual queries
    - Conversation history
    - Knowledge-base management
    - RAG evaluation
    - Groundedness and hallucination checks
    - Role-based and district-based access control
    - Prompt-injection protection boundaries

Endpoints:
    GET    /rag/health
    GET    /rag/collections
    POST   /rag/collections
    POST   /rag/documents
    GET    /rag/documents
    GET    /rag/documents/{document_id}
    DELETE /rag/documents/{document_id}
    POST   /rag/search
    POST   /rag/query
    POST   /rag/chat
    GET    /rag/conversations
    GET    /rag/conversations/{conversation_id}
    DELETE /rag/conversations/{conversation_id}
    POST   /rag/evaluate
    GET    /rag/evaluations
    GET    /rag/metrics

Important:
    RAG answers are informational and advisory.
    Official flood warnings require authorized publication.
    No source means no invented citation.
    Restricted documents require access verification.
"""


import logging
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Protocol

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

from backend.dependencies import get_current_user


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=["Advanced RAG"],
)

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

KNOWLEDGE_ROLES = {
    "knowledge_manager",
    "document_reviewer",
    "emergency_coordinator",
    "district_officer",
}

EVALUATION_ROLES = {
    "rag_evaluator",
    "ai_engineer",
    "ml_engineer",
    "risk_analyst",
}

INGEST_ROLES = (
    ADMIN_ROLES
    | KNOWLEDGE_ROLES
)

EVAL_ROLES = (
    ADMIN_ROLES
    | EVALUATION_ROLES
)

MAX_PAGE_SIZE = 100
MAX_QUERY_LENGTH = 4000
MAX_DOCUMENT_TEXT_LENGTH = 200_000


# ============================================================
# ENUMERATIONS
# ============================================================

class RetrievalMode(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GRAPH = "graph"


class RAGMode(str, Enum):
    STANDARD = "standard"
    ADVANCED = "advanced"
    MULTI_QUERY = "multi_query"
    CORRECTIVE = "corrective"
    SELF_RAG = "self_rag"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"


class DocumentVisibility(str, Enum):
    PUBLIC = "public"
    DISTRICT = "district"
    RESTRICTED = "restricted"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class AnswerLanguage(str, Enum):
    AUTO = "auto"
    ENGLISH = "en"
    MALAYALAM = "ml"
    HINDI = "hi"
    TAMIL = "ta"


class EvaluationMetric(str, Enum):
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CITATION_ACCURACY = "citation_accuracy"
    HALLUCINATION_RATE = "hallucination_rate"
    LATENCY = "latency"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class CollectionCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str = Field(
        min_length=3,
        max_length=120,
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    district_id: int | None = Field(
        default=None,
        ge=1,
    )

    visibility: DocumentVisibility = (
        DocumentVisibility.RESTRICTED
    )

    embedding_model: str | None = Field(
        default=None,
        max_length=150,
    )


class DocumentIngestRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    collection_id: str

    title: str = Field(
        min_length=3,
        max_length=255,
    )

    document_type: DocumentType

    text_content: str | None = Field(
        default=None,
        max_length=MAX_DOCUMENT_TEXT_LENGTH,
    )

    source_url: HttpUrl | None = None

    source_reference: str | None = Field(
        default=None,
        max_length=500,
    )

    district_id: int | None = Field(
        default=None,
        ge=1,
    )

    visibility: DocumentVisibility = (
        DocumentVisibility.RESTRICTED
    )

    language: AnswerLanguage = (
        AnswerLanguage.AUTO
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_document(
        self,
    ) -> "DocumentIngestRequest":
        self.collection_id = str(
            uuid.UUID(self.collection_id)
        )

        if not self.text_content and not self.source_url:
            raise ValueError(
                "Document text or source URL is required"
            )

        if len(self.metadata) > 30:
            raise ValueError(
                "Too many metadata fields"
            )

        return self


class RAGSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=2,
        max_length=MAX_QUERY_LENGTH,
    )

    collection_ids: list[str] = Field(
        min_length=1,
        max_length=20,
    )

    retrieval_mode: RetrievalMode = (
        RetrievalMode.HYBRID
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    rerank: bool = True

    district_id: int | None = Field(
        default=None,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_collections(
        self,
    ) -> "RAGSearchRequest":
        self.collection_ids = [
            str(uuid.UUID(identifier))
            for identifier in self.collection_ids
        ]

        return self


class RAGQueryRequest(RAGSearchRequest):
    rag_mode: RAGMode = RAGMode.ADVANCED

    language: AnswerLanguage = (
        AnswerLanguage.AUTO
    )

    include_citations: bool = True

    include_explanation: bool = False

    require_grounded_answer: bool = True


class RAGChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=2,
        max_length=MAX_QUERY_LENGTH,
    )

    collection_ids: list[str] = Field(
        min_length=1,
        max_length=20,
    )

    conversation_id: str | None = None

    rag_mode: RAGMode = RAGMode.ADVANCED

    retrieval_mode: RetrievalMode = (
        RetrievalMode.HYBRID
    )

    language: AnswerLanguage = (
        AnswerLanguage.AUTO
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    @model_validator(mode="after")
    def validate_ids(
        self,
    ) -> "RAGChatRequest":
        self.collection_ids = [
            str(uuid.UUID(identifier))
            for identifier in self.collection_ids
        ]

        if self.conversation_id is not None:
            self.conversation_id = str(
                uuid.UUID(self.conversation_id)
            )

        return self


class RAGEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_reference: str = Field(
        min_length=5,
        max_length=500,
    )

    collection_ids: list[str] = Field(
        min_length=1,
        max_length=20,
    )

    metrics: list[EvaluationMetric] = Field(
        default_factory=lambda: [
            EvaluationMetric.FAITHFULNESS,
            EvaluationMetric.ANSWER_RELEVANCE,
            EvaluationMetric.CONTEXT_PRECISION,
            EvaluationMetric.CITATION_ACCURACY,
        ],
    )

    model_reference: str | None = Field(
        default=None,
        max_length=150,
    )

    @model_validator(mode="after")
    def validate_ids(
        self,
    ) -> "RAGEvaluationRequest":
        self.collection_ids = [
            str(uuid.UUID(identifier))
            for identifier in self.collection_ids
        ]

        return self


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class RAGCitation(BaseModel):
    document_id: str

    chunk_id: str

    title: str

    source_reference: str | None = None

    page_number: int | None = None

    relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class RAGSearchResult(BaseModel):
    document_id: str

    chunk_id: str

    content: str

    score: float

    citation: RAGCitation


class RAGSearchResponse(BaseModel):
    query: str

    results: list[RAGSearchResult]

    retrieval_mode: RetrievalMode

    total_results: int

    latency_ms: float = Field(ge=0)


class RAGAnswerResponse(BaseModel):
    query: str

    answer: str

    citations: list[RAGCitation]

    grounded: bool

    confidence_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    language: str

    model_reference: str

    latency_ms: float = Field(ge=0)

    advisory_only: bool = True


class RAGChatResponse(RAGAnswerResponse):
    conversation_id: str

    message_id: str


class CollectionResponse(BaseModel):
    id: str

    name: str

    visibility: DocumentVisibility

    district_id: int | None = None

    document_count: int = 0

    created_at: datetime


class DocumentResponse(BaseModel):
    id: str

    collection_id: str

    title: str

    status: DocumentStatus

    created_at: datetime

    indexed_at: datetime | None = None


class ConversationResponse(BaseModel):
    id: str

    title: str

    created_at: datetime

    updated_at: datetime


class RAGEvaluationResponse(BaseModel):
    evaluation_id: str

    status: str

    metrics: dict[str, float] = Field(
        default_factory=dict,
    )

    created_at: datetime


class RAGMetricsResponse(BaseModel):
    total_queries: int

    average_latency_ms: float

    grounded_answer_rate: float

    citation_accuracy: float

    retrieval_hit_rate: float

    generated_at: datetime


class RAGHealthResponse(BaseModel):
    service: str

    status: str

    vector_store_available: bool

    embedding_model_available: bool

    reranker_available: bool

    llm_available: bool

    timestamp: datetime


# ============================================================
# SERVICE CONTRACT
# ============================================================

class RAGService(Protocol):
    """
    Concrete RAG service contract.

    Required internal components:
        - Document ingestion and validation
        - Malware and unsafe file checks
        - Chunking and embedding
        - BM25 search
        - Vector search
        - Hybrid retrieval
        - Reranking
        - Query rewriting
        - Multi-query retrieval
        - Citation generation
        - LLM answer generation
        - Groundedness verification
        - Prompt-injection detection
        - Document-level authorization
        - Conversation persistence
        - RAG evaluation
        - Audit logging

    Every method must enforce user-specific access.
    """

    async def health(self) -> RAGHealthResponse:
        ...

    async def list_collections(
        self,
        actor_id: str,
    ) -> list[CollectionResponse]:
        ...

    async def create_collection(
        self,
        payload: CollectionCreate,
        actor_id: str,
    ) -> CollectionResponse:
        ...

    async def ingest_document(
        self,
        payload: DocumentIngestRequest,
        actor_id: str,
    ) -> DocumentResponse:
        ...

    async def list_documents(
        self,
        actor_id: str,
        collection_id: str | None,
        limit: int,
        offset: int,
    ) -> list[DocumentResponse]:
        ...

    async def get_document(
        self,
        document_id: str,
        actor_id: str,
    ) -> DocumentResponse | None:
        ...

    async def delete_document(
        self,
        document_id: str,
        actor_id: str,
    ) -> bool:
        ...

    async def search(
        self,
        payload: RAGSearchRequest,
        actor_id: str,
    ) -> RAGSearchResponse:
        ...

    async def query(
        self,
        payload: RAGQueryRequest,
        actor_id: str,
    ) -> RAGAnswerResponse:
        ...

    async def chat(
        self,
        payload: RAGChatRequest,
        actor_id: str,
    ) -> RAGChatResponse:
        ...

    async def list_conversations(
        self,
        actor_id: str,
        limit: int,
        offset: int,
    ) -> list[ConversationResponse]:
        ...

    async def get_conversation(
        self,
        conversation_id: str,
        actor_id: str,
    ) -> dict[str, Any] | None:
        ...

    async def delete_conversation(
        self,
        conversation_id: str,
        actor_id: str,
    ) -> bool:
        ...

    async def evaluate(
        self,
        payload: RAGEvaluationRequest,
        actor_id: str,
    ) -> RAGEvaluationResponse:
        ...

    async def list_evaluations(
        self,
        actor_id: str,
        limit: int,
        offset: int,
    ) -> list[RAGEvaluationResponse]:
        ...

    async def metrics(
        self,
        actor_id: str,
    ) -> RAGMetricsResponse:
        ...


# ============================================================
# DEPENDENCY INJECTION
# ============================================================

def get_rag_service(
    request: Request,
) -> RAGService:
    service = getattr(
        request.app.state,
        "rag_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not configured",
        )

    return service


RAGServiceDependency = Annotated[
    RAGService,
    Depends(get_rag_service),
]


# ============================================================
# SECURITY HELPERS
# ============================================================

def validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid UUID",
        ) from None


def actor_id(user: Any) -> str:
    identifier = getattr(user, "id", None)

    if identifier is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return str(identifier)


def role_of(user: Any) -> str:
    role = getattr(user, "role", "")

    if isinstance(role, Enum):
        role = role.value

    return str(role).lower()


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
    district_id: int | None,
) -> None:
    if district_id is None:
        return

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


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=RAGHealthResponse,
)
async def rag_health(
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> RAGHealthResponse:
    return await service.health()


# ============================================================
# COLLECTIONS
# ============================================================

@router.get(
    "/collections",
    response_model=list[CollectionResponse],
)
async def list_rag_collections(
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> list[CollectionResponse]:
    return await service.list_collections(
        actor_id=actor_id(current_user),
    )


@router.post(
    "/collections",
    response_model=CollectionResponse,
    status_code=201,
)
async def create_rag_collection(
    payload: CollectionCreate,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> CollectionResponse:
    require_role(current_user, INGEST_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    return await service.create_collection(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# DOCUMENT INGESTION
# ============================================================

@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=202,
)
async def ingest_rag_document(
    payload: DocumentIngestRequest,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> DocumentResponse:
    require_role(current_user, INGEST_ROLES)

    require_district_access(
        current_user,
        payload.district_id,
    )

    # The service must also check collection ownership,
    # source URL allowlists, SSRF, unsafe content,
    # document visibility and ingestion permissions.
    return await service.ingest_document(
        payload=payload,
        actor_id=actor_id(current_user),
    )


@router.get(
    "/documents",
    response_model=list[DocumentResponse],
)
async def list_rag_documents(
    current_user: CurrentUser,
    service: RAGServiceDependency,
    collection_id: str | None = None,
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> list[DocumentResponse]:
    if collection_id is not None:
        collection_id = validate_uuid(
            collection_id
        )

    return await service.list_documents(
        actor_id=actor_id(current_user),
        collection_id=collection_id,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
)
async def get_rag_document(
    document_id: str,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> DocumentResponse:
    document = await service.get_document(
        document_id=validate_uuid(
            document_id
        ),
        actor_id=actor_id(current_user),
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return document


@router.delete(
    "/documents/{document_id}",
    status_code=204,
)
async def delete_rag_document(
    document_id: str,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> None:
    require_role(current_user, INGEST_ROLES)

    deleted = await service.delete_document(
        document_id=validate_uuid(
            document_id
        ),
        actor_id=actor_id(current_user),
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )


# ============================================================
# HYBRID SEARCH
# ============================================================

@router.post(
    "/search",
    response_model=RAGSearchResponse,
)
async def rag_search(
    payload: RAGSearchRequest,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> RAGSearchResponse:
    require_district_access(
        current_user,
        payload.district_id,
    )

    return await service.search(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# GROUNDED QUESTION ANSWERING
# ============================================================

@router.post(
    "/query",
    response_model=RAGAnswerResponse,
)
async def rag_query(
    payload: RAGQueryRequest,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> RAGAnswerResponse:
    require_district_access(
        current_user,
        payload.district_id,
    )

    response = await service.query(
        payload=payload,
        actor_id=actor_id(current_user),
    )

    if (
        payload.require_grounded_answer
        and not response.grounded
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Answer could not be verified "
                "against retrieved evidence"
            ),
        )

    if (
        payload.include_citations
        and response.answer.strip()
        and not response.citations
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Grounded answer requires "
                "verifiable source citations"
            ),
        )

    return response


# ============================================================
# CONVERSATIONAL RAG
# ============================================================

@router.post(
    "/chat",
    response_model=RAGChatResponse,
)
async def rag_chat(
    payload: RAGChatRequest,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> RAGChatResponse:
    # The service must enforce conversation ownership,
    # document permissions and history isolation.
    return await service.chat(
        payload=payload,
        actor_id=actor_id(current_user),
    )


# ============================================================
# CONVERSATIONS
# ============================================================

@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
)
async def list_rag_conversations(
    current_user: CurrentUser,
    service: RAGServiceDependency,
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> list[ConversationResponse]:
    return await service.list_conversations(
        actor_id=actor_id(current_user),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/conversations/{conversation_id}",
)
async def get_rag_conversation(
    conversation_id: str,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> dict[str, Any]:
    conversation = await service.get_conversation(
        conversation_id=validate_uuid(
            conversation_id
        ),
        actor_id=actor_id(current_user),
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return conversation


@router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
)
async def delete_rag_conversation(
    conversation_id: str,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> None:
    deleted = await service.delete_conversation(
        conversation_id=validate_uuid(
            conversation_id
        ),
        actor_id=actor_id(current_user),
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )


# ============================================================
# RAG EVALUATION
# ============================================================

@router.post(
    "/evaluate",
    response_model=RAGEvaluationResponse,
    status_code=202,
)
async def evaluate_rag(
    payload: RAGEvaluationRequest,
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> RAGEvaluationResponse:
    require_role(current_user, EVAL_ROLES)

    return await service.evaluate(
        payload=payload,
        actor_id=actor_id(current_user),
    )


@router.get(
    "/evaluations",
    response_model=list[RAGEvaluationResponse],
)
async def list_rag_evaluations(
    current_user: CurrentUser,
    service: RAGServiceDependency,
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> list[RAGEvaluationResponse]:
    require_role(current_user, EVAL_ROLES)

    return await service.list_evaluations(
        actor_id=actor_id(current_user),
        limit=limit,
        offset=offset,
    )


# ============================================================
# RAG METRICS
# ============================================================

@router.get(
    "/metrics",
    response_model=RAGMetricsResponse,
)
async def rag_metrics(
    current_user: CurrentUser,
    service: RAGServiceDependency,
) -> RAGMetricsResponse:
    require_role(current_user, EVAL_ROLES)

    return await service.metrics(
        actor_id=actor_id(current_user),
    )


__all__ = [
    "router",
    "RAGService",
    "RetrievalMode",
    "RAGMode",
    "DocumentType",
    "DocumentVisibility",
    "DocumentStatus",
    "AnswerLanguage",
    "EvaluationMetric",
    "CollectionCreate",
    "DocumentIngestRequest",
    "RAGSearchRequest",
    "RAGQueryRequest",
    "RAGChatRequest",
    "RAGEvaluationRequest",
    "RAGCitation",
    "RAGSearchResult",
    "RAGSearchResponse",
    "RAGAnswerResponse",
    "RAGChatResponse",
    "CollectionResponse",
    "DocumentResponse",
    "ConversationResponse",
    "RAGEvaluationResponse",
    "RAGMetricsResponse",
    "RAGHealthResponse",
]
