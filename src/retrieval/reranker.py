"""
Cross-Encoder Reranker
========================
Re-ranks retrieved documents using ms-marco-MiniLM-L-6-v2 cross-encoder.
"""

from typing import List, Optional

from loguru import logger

from config.settings import get_settings
from src.retrieval.dense_retriever import RetrievedDocument


class Reranker:
    """Cross-encoder reranker using ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model=None):
        self.settings = get_settings()
        self.model = model
        self._initialized = False

    def initialize(self):
        """Lazy initialization of cross-encoder model."""
        if self._initialized:
            return

        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.settings.reranker.model,
                device=self.settings.reranker.device,
            )
            logger.info(f"Reranker loaded: {self.settings.reranker.model}")

        self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: User query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents sorted by cross-encoder score
        """
        self.initialize()
        top_k = top_k or self.settings.reranker.top_k

        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc.text) for doc in documents]

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        reranked = sorted(documents, key=lambda d: d.score, reverse=True)

        logger.debug(
            f"Reranked {len(documents)} docs → top {top_k} "
            f"(best={reranked[0].score:.4f} if reranked else 'N/A')"
        )
        return reranked[:top_k]
