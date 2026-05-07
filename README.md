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
| API | `src/api/main.py` | FastAPI + 컬렉션별 파이프라인 lazy cache |

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

`downloader.py`의 `REPORTS` 딕셔너리에 등록된 보고서를 key로 지정해 다운로드합니다.

```bash
# Berkshire Hathaway 2023 주주서한
python src/data_collection/downloader.py          # 기본값: berkshire_2023

# Tesla 2023 Annual Report (SEC EDGAR)
python - <<'EOF'
from src.data_collection.downloader import download_report, _SEC_HEADERS
download_report("tesla_2023", headers=_SEC_HEADERS)
EOF
```

현재 등록된 보고서:

| key | 문서 | 저장 파일명 |
|-----|------|------------|
| `berkshire_2023` | Berkshire Hathaway 2023 주주서한 | `berkshire_2023.pdf` |
| `tesla_2023` | Tesla 2023 Annual Report (SEC) | `tesla_2023_10k.pdf` |

### 4. 색인 (Ingestion)

PDF마다 별도 컬렉션으로 색인합니다. 컬렉션명은 자유롭게 지정 가능합니다.

```bash
# finance_reports 컬렉션 (Berkshire)
python -m src.ingestion.main

# tesla_2023 컬렉션 (Tesla) — Python으로 직접 지정
python - <<'EOF'
from dotenv import load_dotenv; load_dotenv()
from pathlib import Path
from src.ingestion.document_loader import PDFLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.indexer import QdrantIndexer

pdf   = Path("data/raw/tesla_2023_10k.pdf")
docs  = PDFLoader().load(pdf)
chunks = DocumentChunker().split(docs)
QdrantIndexer().index(chunks, collection_name="tesla_2023")
EOF
```

### 5. API 서버 실행

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 사용법

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/health` | 서버 상태 및 로드된 파이프라인 목록 |
| `GET` | `/collections` | Qdrant에 존재하는 컬렉션 목록 |
| `POST` | `/ask` | 질의응답 |
| `GET` | `/docs` | Swagger UI |

### 컬렉션 목록 확인

```bash
curl http://localhost:8000/collections
```

```json
{ "collections": ["finance_reports", "tesla_2023"] }
```

### 질의응답 — `collection_name`으로 컬렉션 지정

`collection_name`을 생략하면 기본값 `finance_reports`가 사용됩니다.

```bash
# Berkshire Hathaway 문서 검색 (기본값)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "2023년 핵심 실적은?",
    "collection_name": "finance_reports"
  }'

# Tesla 문서 검색
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tesla의 2023년 총 매출은 얼마인가?",
    "collection_name": "tesla_2023"
  }'
```

```json
{
  "answer": "Tesla의 2023년 총 매출은 96.77억 달러입니다.",
  "collection_name": "tesla_2023",
  "sources": [
    {
      "source": "tesla_2023_10k.pdf",
      "page": 137,
      "relevance_score": 0.9921,
      "snippet": "Total revenues of $96.77 billion..."
    }
  ]
}
```

존재하지 않는 컬렉션을 요청하면 `404`와 함께 가용 컬렉션 목록을 반환합니다.

```json
{
  "detail": "Collection 'foo' not found. Available: ['finance_reports', 'tesla_2023']"
}
```

### 파이프라인 동작 방식

컬렉션별 파이프라인은 **첫 요청 시 한 번만 초기화**되고 이후 재사용됩니다(lazy cache). `/health`에서 현재 로드된 파이프라인을 확인할 수 있습니다.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "loaded_pipelines": ["finance_reports", "tesla_2023"]
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
