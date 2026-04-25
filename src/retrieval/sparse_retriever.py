"""
Sparse (BM25) Retriever
=========================
Uses Qdrant's payload-based full-text search for BM25-style sparse retrieval.
"""

from typing import List, Optional

from loguru import logger

from config.settings import get_settings
from src.retrieval.dense_retriever import RetrievedDocument


class SparseRetriever:
    """BM25-style sparse retrieval using Qdrant's full-text filtering."""

    def __init__(self, qdrant_client=None):
        self.settings = get_settings()
        self.client = qdrant_client
        self._initialized = False

    def initialize(self):
        """Lazy initialization."""
        if self._initialized:
            return

        if self.client is None:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(path=self.settings.qdrant.path)

        self._initialized = True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """
        Retrieve documents using keyword-based search.
        
        Uses Qdrant scroll with payload filtering as a lightweight
        BM25 alternative. For production, upgrade to Qdrant's native
        sparse vector support.
        """
        self.initialize()
        top_k = top_k or self.settings.retrieval.sparse_top_k

        # Simple keyword extraction for filtering
        keywords = self._extract_keywords(query)

        if not keywords:
            return []

        # Use Qdrant scroll with text matching
        # NOTE: For full BM25, implement Qdrant sparse vectors
        from qdrant_client.models import Filter, FieldCondition, MatchText

        documents = []
        for keyword in keywords[:5]:  # Top 5 keywords
            try:
                results = self.client.scroll(
                    collection_name=self.settings.qdrant.collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="text", match=MatchText(text=keyword))]
                    ),
                    limit=top_k // len(keywords[:5]),
                )

                for point in results[0]:
                    payload = point.payload or {}
                    documents.append(RetrievedDocument(
                        id=point.id,
                        text=payload.get("text", ""),
                        score=0.5,  # Uniform score for keyword matches
                        source_file=payload.get("source_file", ""),
                        spec_number=payload.get("spec_number", ""),
                        section=payload.get("section", ""),
                        content_type=payload.get("content_type", "text"),
                        metadata=payload,
                    ))
            except Exception as e:
                logger.warning(f"Sparse search failed for keyword '{keyword}': {e}")

        # Deduplicate by ID
        seen = set()
        unique = []
        for doc in documents:
            if doc.id not in seen:
                seen.add(doc.id)
                unique.append(doc)

        logger.debug(f"Sparse retrieval: {len(unique)} results for query: {query[:50]}...")
        return unique[:top_k]

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from a query."""
        # Remove common stop words
        stop_words = {
            "what", "is", "the", "a", "an", "in", "of", "for", "to",
            "and", "or", "how", "does", "do", "can", "which", "that",
            "this", "with", "from", "on", "at", "by", "be", "are",
        }
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
