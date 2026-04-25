"""
Dense Vector Retriever
========================
Retrieves documents by cosine similarity using bge-small-en-v1.5 embeddings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

from config.settings import get_settings


@dataclass
class RetrievedDocument:
    """A retrieved document with score and metadata."""
    id: int
    text: str
    score: float
    source_file: str = ""
    spec_number: str = ""
    section: str = ""
    content_type: str = "text"
    metadata: Dict = field(default_factory=dict)


class DenseRetriever:
    """Dense retrieval using Qdrant vector search."""

    def __init__(self, qdrant_client=None, embedding_model=None):
        self.settings = get_settings()
        self.client = qdrant_client
        self.embedding_model = embedding_model
        self._initialized = False

    def initialize(self):
        """Lazy initialization."""
        if self._initialized:
            return

        if self.client is None:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(path=self.settings.qdrant.path)

        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                self.settings.embedding.model,
                device=self.settings.embedding.device,
            )

        self._initialized = True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Retrieve top-k documents by dense vector similarity."""
        self.initialize()
        top_k = top_k or self.settings.retrieval.dense_top_k

        # Encode query
        query_embedding = self.embedding_model.encode(
            query, normalize_embeddings=True
        ).tolist()

        # Search Qdrant
        results = self.client.search(
            collection_name=self.settings.qdrant.collection,
            query_vector=query_embedding,
            limit=top_k,
        )

        documents = []
        for result in results:
            payload = result.payload or {}
            documents.append(RetrievedDocument(
                id=result.id,
                text=payload.get("text", ""),
                score=result.score,
                source_file=payload.get("source_file", ""),
                spec_number=payload.get("spec_number", ""),
                section=payload.get("section", ""),
                content_type=payload.get("content_type", "text"),
                metadata=payload,
            ))

        logger.debug(f"Dense retrieval: {len(documents)} results for query: {query[:50]}...")
        return documents
