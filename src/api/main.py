"""FastAPI application exposing the Finance RAG pipeline."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.generation.answer_generator import FinanceRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[2]
_QDRANT_PATH = str(_PROJECT_ROOT / "qdrant_local")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="질문 텍스트")
    collection_name: str = Field("finance_reports", description="검색할 Qdrant 컬렉션명")


class SourceDocument(BaseModel):
    source: str
    page: Union[int, str]
    relevance_score: Optional[float]
    snippet: str


class AskResponse(BaseModel):
    answer: str
    collection_name: str
    sources: List[SourceDocument]


class CollectionsResponse(BaseModel):
    collections: List[str]


# ---------------------------------------------------------------------------
# Pipeline cache — created lazily per collection on first request
# ---------------------------------------------------------------------------

_pipelines: Dict[str, FinanceRAGPipeline] = {}
_qdrant_client: Optional[QdrantClient] = None


def _available_collections() -> List[str]:
    assert _qdrant_client is not None
    return [c.name for c in _qdrant_client.get_collections().collections]


def _get_pipeline(collection_name: str) -> FinanceRAGPipeline:
    """Return a cached pipeline for *collection_name*, creating it if needed."""
    if collection_name not in _available_collections():
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found. Available: {_available_collections()}",
        )
    if collection_name not in _pipelines:
        logger.info("Initialising pipeline for collection '%s' …", collection_name)
        _pipelines[collection_name] = FinanceRAGPipeline(
            collection_name=collection_name,
            qdrant_client=_qdrant_client,  # share the single Qdrant client
        )
        logger.info("Pipeline ready for '%s'.", collection_name)
    return _pipelines[collection_name]


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # type: ignore[type-arg]
    global _qdrant_client
    _qdrant_client = QdrantClient(path=_QDRANT_PATH)
    collections = _available_collections()
    logger.info("Qdrant connected. Available collections: %s", collections)
    yield
    _pipelines.clear()
    _qdrant_client = None
    logger.info("Pipelines released.")


app = FastAPI(
    title="Finance Insight RAG API",
    description="금융 문서 기반 질의응답 API (Qdrant + Cohere Rerank + GPT-4o-mini)",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health() -> dict:
    """서버 상태 및 로드된 파이프라인 목록 반환."""
    return {
        "status": "ok",
        "loaded_pipelines": list(_pipelines.keys()),
    }


@app.get("/collections", response_model=CollectionsResponse, tags=["System"])
def collections() -> CollectionsResponse:
    """Qdrant에 존재하는 컬렉션 목록 반환."""
    return CollectionsResponse(collections=_available_collections())


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest) -> AskResponse:
    """질문과 컬렉션명을 받아 RAG 파이프라인으로 답변과 참조 문서를 반환합니다."""
    pipeline = _get_pipeline(request.collection_name)

    logger.info("[%s] Query: %s", request.collection_name, request.query)
    result = pipeline.ask(request.query)

    sources = [
        SourceDocument(
            source=Path(doc.metadata.get("source", "unknown")).name,
            page=doc.metadata.get("page", "?"),
            relevance_score=doc.metadata.get("relevance_score"),
            snippet=doc.page_content[:200].strip(),
        )
        for doc in result["source_documents"]
    ]

    return AskResponse(
        answer=result["answer"],
        collection_name=request.collection_name,
        sources=sources,
    )
