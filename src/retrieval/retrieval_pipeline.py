"""
Retrieval Pipeline
====================
Orchestrates the full retrieval flow: Dense → Sparse → RRF Fusion → Reranking.
"""

from typing import List, Optional

from loguru import logger

from config.settings import get_settings
from src.retrieval.dense_retriever import DenseRetriever, RetrievedDocument
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_fusion import HybridFusion
from src.retrieval.reranker import Reranker


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline combining:
    1. Dense retrieval (bge-small-en-v1.5)
    2. Sparse retrieval (Qdrant keyword matching)
    3. RRF fusion
    4. Cross-encoder reranking (MiniLM)
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[HybridFusion] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.settings = get_settings()
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or SparseRetriever()
        self.fusion = fusion or HybridFusion()
        self.reranker = reranker or Reranker()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
    ) -> List[RetrievedDocument]:
        """
        Full retrieval pipeline.
        
        Args:
            query: User query
            top_k: Final number of documents to return
            use_reranker: Whether to apply cross-encoder reranking
            
        Returns:
            Top-k retrieved and ranked documents
        """
        top_k = top_k or self.settings.retrieval.rerank_top_k

        logger.info(f"Retrieving for: {query[:80]}...")

        # Step 1: Dense retrieval
        dense_results = self.dense.retrieve(query)

        # Step 2: Sparse retrieval
        sparse_results = self.sparse.retrieve(query)

        # Step 3: RRF fusion
        fused = self.fusion.fuse(dense_results, sparse_results)

        # Step 4: Cross-encoder reranking
        if use_reranker and fused:
            final = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            final = fused[:top_k]

        logger.info(
            f"Pipeline: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(fused)} fused → {len(final)} final"
        )
        return final
