"""Entry point for the Finance RAG ingestion pipeline."""

import logging
from pathlib import Path

from dotenv import load_dotenv

from .document_loader import PDFLoader
from .chunker import DocumentChunker
from .indexer import QdrantIndexer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths are resolved relative to the project root (two levels above this file)
_PROJECT_ROOT = Path(__file__).parents[2]
PDF_PATH = _PROJECT_ROOT / "data" / "raw" / "sample_report.pdf"
QDRANT_PATH = str(_PROJECT_ROOT / "qdrant_local")
COLLECTION_NAME = "finance_reports"


def run() -> None:
    """Execute the full load → chunk → index pipeline."""
    logger.info("=" * 50)
    logger.info("Finance RAG — Ingestion Pipeline")
    logger.info("=" * 50)

    # --- Step 1: Load ---
    logger.info("[1/3] Loading PDF: %s", PDF_PATH)
    loader = PDFLoader()
    documents = loader.load(PDF_PATH)
    logger.info("      Loaded %d page(s)", len(documents))

    # --- Step 2: Chunk ---
    logger.info("[2/3] Chunking documents (size=800, overlap=100)")
    chunker = DocumentChunker()
    chunks = chunker.split(documents)
    logger.info("      Created %d chunk(s)", len(chunks))

    # --- Step 3: Index ---
    logger.info("[3/3] Indexing into Qdrant — collection: '%s'", COLLECTION_NAME)
    logger.info("      DB path: %s", QDRANT_PATH)
    indexer = QdrantIndexer(db_path=QDRANT_PATH)
    count = indexer.index(chunks, COLLECTION_NAME)
    logger.info("      Indexed %d chunk(s) successfully", count)

    logger.info("=" * 50)
    logger.info("Pipeline complete.")
    logger.info("=" * 50)


if __name__ == "__main__":
    run()
