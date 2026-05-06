"""PDF document loader that preserves page-level metadata."""

from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document


class PDFLoader:
    """Load a PDF file and extract text with page-level metadata."""

    def load(self, pdf_path: Path) -> list[Document]:
        """Return one Document per non-empty page, with source and page metadata."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        documents: list[Document] = []
        with fitz.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(pdf_path), "page": page_num},
                        )
                    )
        return documents
