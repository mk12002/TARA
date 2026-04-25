"""
Reciprocal Rank Fusion (RRF)
==============================
Combines results from dense and sparse retrievers using RRF scoring.

Formula: RRF(d) = Σ 1/(k + rank_i(d))
Where k is a constant (default=60) and rank_i is rank in retriever i.
"""

from typing import Dict, List

from loguru import logger

from config.settings import get_settings
from src.retrieval.dense_retriever import RetrievedDocument


class HybridFusion:
    """Fuses dense and sparse retrieval results using RRF."""

    def __init__(self, k: int = 60):
        self.k = k or get_settings().retrieval.rrf_k

    def fuse(
        self,
        dense_results: List[RetrievedDocument],
        sparse_results: List[RetrievedDocument],
        top_k: int = 50,
    ) -> List[RetrievedDocument]:
        """
        Fuse results from multiple retrievers using RRF.
        
        Args:
            dense_results: Results from dense retriever (ranked by score)
            sparse_results: Results from sparse retriever (ranked by score)
            top_k: Number of results to return
            
        Returns:
            Fused and re-ranked list of documents
        """
        rrf_scores: Dict[int, float] = {}
        doc_map: Dict[int, RetrievedDocument] = {}

        # Score dense results
        for rank, doc in enumerate(dense_results, 1):
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1.0 / (self.k + rank)
            doc_map[doc.id] = doc

        # Score sparse results
        for rank, doc in enumerate(sparse_results, 1):
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1.0 / (self.k + rank)
            if doc.id not in doc_map:
                doc_map[doc.id] = doc

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.score = rrf_scores[doc_id]  # Replace score with RRF score
            fused.append(doc)

        logger.debug(
            f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(fused)} fused results"
        )
        return fused
