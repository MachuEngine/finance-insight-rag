"""RAG pipeline evaluation using RAGAS metrics (reference-free + reference-based)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

from src.generation.answer_generator import FinanceRAGPipeline

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# - Faithfulness                       : 답변이 검색된 컨텍스트에 충실한가  (0~1, 높을수록 좋음)
# - ResponseRelevancy                  : 답변이 질문에 관련되어 있는가       (0~1, 높을수록 좋음)
# - LLMContextPrecisionWithoutReference: 검색된 컨텍스트가 질문에 적절한가  (0~1, 높을수록 좋음)


def _build_metrics() -> list:
    """Initialise RAGAS metrics with the OpenAI key available at call time."""
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    return [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithoutReference(llm=llm),
    ]


@dataclass
class EvalCase:
    """A single evaluation question with an optional reference answer.

    Providing *ground_truth* enables reference-based interpretation
    but is not required — all three metrics above are reference-free.
    """

    question: str
    ground_truth: Optional[str] = None


# Default test cases for the tesla_2023 collection
TESLA_TEST_CASES: List[EvalCase] = [
    EvalCase(
        question="Tesla의 2023년 총 매출은 얼마인가?",
        ground_truth="Tesla의 2023년 총 매출은 $96.77 billion입니다.",
    ),
    EvalCase(
        question="2023년 Tesla의 차량 인도 대수는 얼마인가?",
        ground_truth="2023년 Tesla의 차량 인도 대수는 1,808,581대입니다.",
    ),
    EvalCase(
        question="Tesla의 2023년 순이익은 얼마인가?",
        ground_truth="Tesla의 2023년 순이익(Net Income)은 $15.0 billion입니다.",
    ),
    EvalCase(
        question="Cybertruck은 언제 생산을 시작했나?",
        ground_truth="Cybertruck은 2023년에 생산을 시작했습니다.",
    ),
    EvalCase(
        question="Tesla의 주요 사업 리스크는 무엇인가?",
        # ground_truth 없이 reference-free 평가
    ),
]


class RAGEvaluator:
    """Evaluate a FinanceRAGPipeline using RAGAS metrics.

    Args:
        collection_name: Qdrant collection to evaluate against.
        pipeline:        Pre-built pipeline for DI (optional).
    """

    def __init__(
        self,
        collection_name: str = "tesla_2023",
        pipeline: Optional[FinanceRAGPipeline] = None,
    ) -> None:
        self._pipeline = pipeline or FinanceRAGPipeline(collection_name=collection_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, cases: List[EvalCase]) -> dict:
        """Run evaluation and return RAGAS result object.

        The returned object supports `.to_pandas()` and direct score access.
        """
        logger.info("Building %d sample(s) via RAG pipeline …", len(cases))
        samples = [self._build_sample(c) for c in cases]

        dataset = EvaluationDataset(samples=samples)
        logger.info("Running RAGAS evaluation …")
        result = evaluate(dataset=dataset, metrics=_build_metrics())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sample(self, case: EvalCase) -> SingleTurnSample:
        """Run the RAG pipeline for one case and wrap as SingleTurnSample."""
        logger.info("  Q: %s", case.question)
        output = self._pipeline.ask(case.question)

        return SingleTurnSample(
            user_input=case.question,
            response=output["answer"],
            retrieved_contexts=[d.page_content for d in output["source_documents"]],
            reference=case.ground_truth,
        )


# ---------------------------------------------------------------------------
# __main__ — quick smoke test against the tesla_2023 collection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    evaluator = RAGEvaluator(collection_name="tesla_2023")
    result = evaluator.run(TESLA_TEST_CASES)

    df = result.to_pandas()

    divider = "─" * 72
    print(f"\n{divider}")
    print("  RAGAS Evaluation Results  —  collection: tesla_2023")
    print(divider)

    cols = ["user_input", "faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]
    display_cols = [c for c in cols if c in df.columns]
    print(df[display_cols].to_string(index=False))

    print(f"\n{divider}")
    print("  평균 점수 (Mean Scores)")
    print(divider)
    score_cols = [c for c in display_cols if c != "user_input"]
    for col in score_cols:
        print(f"  {col:<48}  {df[col].mean():.4f}")
    print(divider + "\n")
