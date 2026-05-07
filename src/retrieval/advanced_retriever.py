"""Two-stage retrieval: Qdrant vector search (k=10) → Cohere rerank (top_n=3)."""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

_PROJECT_ROOT = Path(__file__).parents[2]
_QDRANT_PATH = str(_PROJECT_ROOT / "qdrant_local")
_COLLECTION_NAME = "finance_reports"

logger = logging.getLogger(__name__)


class FinanceRetriever:
    """Two-stage retriever combining Qdrant vector search and Cohere reranking.

    Stage 1 — Qdrant: fetches the top-k=10 candidates by cosine similarity.
    Stage 2 — Cohere: reranks candidates and returns the top_n=3 most relevant.
    """

    def __init__(
        self,
        collection_name: str = _COLLECTION_NAME,
        client: Optional[QdrantClient] = None,
    ) -> None:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise EnvironmentError(
                "COHERE_API_KEY가 .env에 없습니다. "
                "https://cohere.com 에서 API 키를 발급받아 .env에 추가하세요."
            )

        # Accept an injected client to avoid opening a second local Qdrant instance
        client = client or QdrantClient(path=_QDRANT_PATH)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        # Stage 1: retrieve 10 candidates from Qdrant
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # Stage 2: rerank to top 3 with Cohere
        reranker = CohereRerank(
            cohere_api_key=cohere_api_key,
            model="rerank-english-v3.0",
            top_n=3,
        )

        self._retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )

    def search(self, query: str) -> list[Document]:
        """Return top-3 reranked Documents for *query*."""
        logger.info("Searching: %s", query)
        return self._retriever.invoke(query)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Berkshire Hathaway 2023 letter이 인덱싱된 상태이므로 관련 질문 사용
    query = "What were Berkshire Hathaway's major investments and financial results in 2023?"

    retriever = FinanceRetriever()
    results = retriever.search(query)

    print(f"\n{'─' * 64}")
    print(f"  Query: {query}")
    print(f"{'─' * 64}")

    for i, doc in enumerate(results, start=1):
        score = doc.metadata.get("relevance_score", None)
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")

        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)

        print(f"\n[Result {i}]  Relevance Score: {score_str}  |  {source}  p.{page}")
        print("─" * 64)
        print(doc.page_content[:400].strip())
        print("  ...")

    print(f"\n{'─' * 64}")
    print(f"  총 {len(results)}개 문서 반환 (Qdrant 10개 → Cohere rerank → top 3)")
    print(f"{'─' * 64}\n")
