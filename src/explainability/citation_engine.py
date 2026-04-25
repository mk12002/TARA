"""
Citation Engine
==================
Validates that LLM claims are grounded in retrieved context.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

from loguru import logger


@dataclass
class VerifiedCitation:
    """A verified citation linking a claim to source text."""
    claim: str
    source_id: int
    source_text: str
    is_verified: bool
    similarity: float = 0.0


class CitationEngine:
    """
    Validates LLM citations against retrieved context.
    
    This is the core of the faithfulness strategy — every claim
    must be traceable to a source document.
    """

    CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

    def verify_response(
        self,
        response: str,
        source_texts: Dict[int, str],
    ) -> List[VerifiedCitation]:
        """
        Verify all citations in a response against source texts.
        
        Args:
            response: LLM-generated response with [Source N] citations
            source_texts: Map of source_id → source text
            
        Returns:
            List of verified citations
        """
        # Split response into sentences
        sentences = re.split(r"[.!?]\s+", response)
        verified = []

        for sentence in sentences:
            cited_ids = [int(m) for m in self.CITATION_PATTERN.findall(sentence)]
            clean_sentence = self.CITATION_PATTERN.sub("", sentence).strip()

            if not clean_sentence:
                continue

            for source_id in cited_ids:
                source_text = source_texts.get(source_id, "")
                is_verified = self._check_overlap(clean_sentence, source_text)

                verified.append(VerifiedCitation(
                    claim=clean_sentence,
                    source_id=source_id,
                    source_text=source_text[:200],
                    is_verified=is_verified,
                ))

        total = len(verified)
        verified_count = sum(1 for v in verified if v.is_verified)
        logger.info(f"Citation verification: {verified_count}/{total} verified")

        return verified

    def _check_overlap(self, claim: str, source: str, threshold: float = 0.3) -> bool:
        """Check if a claim has sufficient word overlap with source text."""
        if not source:
            return False

        claim_words = set(claim.lower().split())
        source_words = set(source.lower().split())

        if not claim_words:
            return False

        overlap = len(claim_words & source_words) / len(claim_words)
        return overlap >= threshold

    def faithfulness_score(self, verified_citations: List[VerifiedCitation]) -> float:
        """Calculate faithfulness score (0-1)."""
        if not verified_citations:
            return 0.0
        return sum(1 for v in verified_citations if v.is_verified) / len(verified_citations)
