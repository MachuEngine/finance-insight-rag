"""RAG answer-generation pipeline: FinanceRetriever → prompt → GPT-4o-mini."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.retrieval.advanced_retriever import FinanceRetriever

load_dotenv()

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
당신은 금융 애널리스트 보조 AI입니다.
아래 <context> 안에 제공된 문서만을 근거로 사용자의 질문에 답변하십시오.

규칙:
1. 반드시 제공된 컨텍스트 내용만 사용할 것. 외부 지식을 추가하지 말 것.
2. 컨텍스트에 답이 없다면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고만 말할 것.
3. 금융 애널리스트처럼 명확하고 논리적인 톤으로, 핵심 수치와 근거를 포함하여 답변할 것.

<context>
{context}
</context>"""


class FinanceRAGPipeline:
    """Full RAG pipeline wiring FinanceRetriever to GPT-4o-mini.

    Dependency injection: pass a pre-built *retriever* to reuse an existing
    instance (e.g. in tests or when the caller already holds one).
    If omitted, a default FinanceRetriever is created internally.
    """

    def __init__(self, retriever: Optional[FinanceRetriever] = None) -> None:
        _retriever = retriever if retriever is not None else FinanceRetriever()

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        # _retriever._retriever is the underlying LangChain BaseRetriever
        # (ContextualCompressionRetriever) required by create_retrieval_chain.
        self._chain = create_retrieval_chain(_retriever._retriever, combine_docs_chain)

    def ask(self, query: str) -> dict[str, Any]:
        """Run the RAG pipeline for *query*.

        Returns:
            answer: final answer string from the LLM.
            source_documents: list of Document objects the answer is based on.
        """
        logger.info("Query: %s", query)
        result = self._chain.invoke({"input": query})
        return {
            "answer": result["answer"],
            "source_documents": result["context"],
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    query = "이 기업의 2023년 핵심 실적은 어때?"

    pipeline = FinanceRAGPipeline()
    response = pipeline.ask(query)

    divider = "─" * 64

    print(f"\n{divider}")
    print(f"  질문: {query}")
    print(divider)
    print("\n[최종 답변]")
    print(response["answer"])

    print(f"\n{divider}")
    print(f"  [참조 출처]  총 {len(response['source_documents'])}개 문서")
    print(divider)

    for i, doc in enumerate(response["source_documents"], start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("relevance_score", None)
        score_str = f"{score:.4f}" if isinstance(score, float) else "N/A"
        print(f"  [{i}] {source}  p.{page}  (relevance: {score_str})")
        print(f"      {doc.page_content[:120].strip().replace(chr(10), ' ')} ...")

    print(f"{divider}\n")
