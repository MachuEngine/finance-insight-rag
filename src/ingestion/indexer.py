"""Embeds document chunks and stores them in a local Qdrant database."""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class QdrantIndexer:
    """Embed chunks with OpenAI and upsert into a local Qdrant collection."""

    # Dimension of text-embedding-3-small
    _VECTOR_SIZE = 1536

    def __init__(self, db_path: str = "./qdrant_local") -> None:
        self._client = QdrantClient(path=db_path)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def index(self, documents: list[Document], collection_name: str) -> int:
        """Embed and upsert documents. Returns the number of chunks indexed."""
        self._ensure_collection(collection_name)
        vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings,
        )
        vector_store.add_documents(documents)
        return len(documents)

    def _ensure_collection(self, name: str) -> None:
        """Create the collection if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._VECTOR_SIZE, distance=Distance.COSINE
                ),
            )
