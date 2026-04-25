"""
Post-Processor
=================
Extracts citations, formats responses, and validates groundedness.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class Citation:
    """A single citation extracted from LLM response."""
    source_id: int
    claim: str
    quote: str = ""


@dataclass
class ProcessedResponse:
    """A fully processed LLM response with citations."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    source_count: int = 0
    is_grounded: bool = True


class PostProcessor:
    """Post-processes LLM responses: citations, formatting, validation."""

    CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

    def process(self, raw_response: str, total_sources: int = 0) -> ProcessedResponse:
        """
        Process a raw LLM response.
        
        Steps:
        1. Extract citations
        2. Check groundedness
        3. Format output
        """
        # Extract citation references
        cited_sources = set(int(m) for m in self.CITATION_PATTERN.findall(raw_response))

        citations = [
            Citation(source_id=sid, claim="", quote="")
            for sid in sorted(cited_sources)
        ]

        # Check groundedness — response should cite at least one source
        is_grounded = len(cited_sources) > 0

        if not is_grounded:
            logger.warning("Response contains no citations — may be ungrounded")

        return ProcessedResponse(
            answer=raw_response.strip(),
            citations=citations,
            source_count=len(cited_sources),
            is_grounded=is_grounded,
        )
