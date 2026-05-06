# Finance Insight RAG

금융/기업 도메인에 특화된 Advanced RAG(Retrieval-Augmented Generation) 시스템입니다.
사업보고서·주주서한 등 PDF 문서를 자동으로 수집·색인하고, 사용자 질문에 대해 근거 있는 답변을 생성합니다.

## 아키텍처

```
PDF 수집          색인                    검색                   생성            API
──────────   ──────────────────   ─────────────────────   ─────────────   ────────
downloader → document_loader   → Qdrant (k=10)        → GPT-4o-mini → FastAPI
             chunker           → Cohere Rerank (top 3)
             indexer (OpenAI
             embedding)
```

| 단계 | 모듈 | 핵심 기술 |
|------|------|-----------|
| 수집 | `src/data_collection/downloader.py` | requests 스트리밍 다운로드 |
| 파싱 | `src/ingestion/document_loader.py` | PyMuPDF — 페이지·출처 메타데이터 보존 |
| 청킹 | `src/ingestion/chunker.py` | RecursiveCharacterTextSplitter (size=800, overlap=100) |
| 색인 | `src/ingestion/indexer.py` | OpenAI `text-embedding-3-small` + Qdrant 로컬 DB |
| 검색 | `src/retrieval/advanced_retriever.py` | Two-Stage: Qdrant → Cohere `rerank-english-v3.0` |
| 생성 | `src/generation/answer_generator.py` | GPT-4o-mini, temperature=0, 할루시네이션 방지 프롬프트 |
| API | `src/api/main.py` | FastAPI + lifespan 파이프라인 싱글턴 |

## 폴더 구조

```
finance-insight-rag/
├── data/
│   ├── raw/              # 다운로드된 PDF (git 제외)
│   └── processed/        # 전처리 결과물 (git 제외)
├── src/
│   ├── data_collection/
│   │   └── downloader.py
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   ├── indexer.py
│   │   └── main.py
│   ├── retrieval/
│   │   └── advanced_retriever.py
│   ├── generation/
│   │   └── answer_generator.py
│   └── api/
│       └── main.py
├── qdrant_local/         # Qdrant 로컬 DB (git 제외)
├── requirements.txt
└── .env                  # API 키 (git 제외)
```

## 시작하기

### 1. 환경 설정

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경 변수

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

### 3. PDF 수집

```bash
python src/data_collection/downloader.py
# → data/raw/sample_report.pdf 저장
```

### 4. 색인 (Ingestion)

```bash
python -m src.ingestion.main
# → qdrant_local/ 에 finance_reports 컬렉션 생성
```

### 5. API 서버 실행

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 사용법

### 상태 확인

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### 질의응답

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "이 기업의 2023년 핵심 실적은 어때?"}'
```

```json
{
  "answer": "2023년의 핵심 실적에 대한 요약은 다음과 같습니다 ...",
  "sources": [
    {
      "source": "sample_report.pdf",
      "page": 10,
      "relevance_score": 0.8331,
      "snippet": "..."
    }
  ]
}
```

Swagger UI: http://localhost:8000/docs

## 기술 스택

- **LLM**: OpenAI GPT-4o-mini
- **Embedding**: OpenAI text-embedding-3-small
- **Vector DB**: Qdrant (로컬 파일시스템 모드)
- **Reranker**: Cohere rerank-english-v3.0
- **Framework**: LangChain, FastAPI
- **PDF 파싱**: PyMuPDF
