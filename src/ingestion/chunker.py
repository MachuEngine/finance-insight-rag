"""Text chunker optimised for financial documents."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Split documents into overlapping chunks suitable for financial text."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Prefer paragraph → line → sentence breaks before hard-splitting
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split each document into chunks, preserving source metadata."""
        return self._splitter.split_documents(documents)
