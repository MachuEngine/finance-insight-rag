"""FastAPI application exposing the Finance RAG pipeline."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation.answer_generator import FinanceRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="질문 텍스트")


class SourceDocument(BaseModel):
    source: str
    page: Union[int, str]
    relevance_score: Optional[float]
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


# ---------------------------------------------------------------------------
# Application lifespan (pipeline is initialised once on startup)
# ---------------------------------------------------------------------------

_pipeline: Optional[FinanceRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # type: ignore[type-arg]
    global _pipeline
    logger.info("Initialising FinanceRAGPipeline …")
    _pipeline = FinanceRAGPipeline()
    logger.info("Pipeline ready.")
    yield
    _pipeline = None
    logger.info("Pipeline released.")


app = FastAPI(
    title="Finance Insight RAG API",
    description="금융 문서 기반 질의응답 API (Qdrant + Cohere Rerank + GPT-4o-mini)",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    """서버 상태 확인."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest) -> AskResponse:
    """질문을 받아 RAG 파이프라인으로 답변과 참조 문서를 반환합니다."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    logger.info("Received query: %s", request.query)
    result = _pipeline.ask(request.query)

    sources = [
        SourceDocument(
            source=Path(doc.metadata.get("source", "unknown")).name,
            page=doc.metadata.get("page", "?"),
            relevance_score=doc.metadata.get("relevance_score"),
            snippet=doc.page_content[:200].strip(),
        )
        for doc in result["source_documents"]
    ]

    return AskResponse(answer=result["answer"], sources=sources)
