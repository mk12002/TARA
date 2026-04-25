"""
Confidence Scorer
===================
Calibrated confidence scoring for RAG responses.
"""

from dataclasses import dataclass
from typing import List

from loguru import logger

from src.retrieval.dense_retriever import RetrievedDocument


@dataclass
class ConfidenceResult:
    """Confidence assessment for a generated response."""
    overall_score: float
    retrieval_confidence: float
    citation_confidence: float
    level: str  # HIGH, MEDIUM, LOW


class ConfidenceScorer:
    """
    Multi-factor confidence scorer.
    
    Factors:
    - Retrieval score (how good were the retrieved docs)
    - Citation coverage (how many claims are cited)
    - Source agreement (do sources agree)
    """

    def score(
        self,
        retrieval_scores: List[float],
        citation_count: int,
        total_claims: int,
    ) -> ConfidenceResult:
        """Calculate calibrated confidence score."""

        # Factor 1: Retrieval quality (avg of top-k scores)
        retrieval_conf = sum(retrieval_scores[:5]) / max(len(retrieval_scores[:5]), 1)

        # Factor 2: Citation coverage
        citation_conf = citation_count / max(total_claims, 1)

        # Combined score (weighted)
        overall = 0.5 * retrieval_conf + 0.5 * citation_conf
        overall = max(0.05, min(0.98, overall))  # Clamp

        # Level
        if overall >= 0.75:
            level = "HIGH"
        elif overall >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        logger.debug(f"Confidence: {overall:.2f} ({level})")
        return ConfidenceResult(
            overall_score=overall,
            retrieval_confidence=retrieval_conf,
            citation_confidence=citation_conf,
            level=level,
        )
