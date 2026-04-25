"""
Query Enricher
================
Expands and enhances user queries for better retrieval.
"""

from loguru import logger

from src.utils.telecom_dictionary import TELECOM_ACRONYMS


class QueryEnricher:
    """Enriches user queries with acronym expansions and synonyms."""

    def __init__(self):
        self.acronyms = TELECOM_ACRONYMS

    def enrich(self, query: str) -> str:
        """
        Enrich a query by expanding acronyms.
        
        Example: "What is PDCCH?" → "What is PDCCH (Physical Downlink Control Channel)?"
        """
        words = query.split()
        enriched_parts = []

        for word in words:
            clean = word.strip("?.,!:;()").upper()
            if clean in self.acronyms:
                enriched_parts.append(f"{word} ({self.acronyms[clean]})")
            else:
                enriched_parts.append(word)

        enriched = " ".join(enriched_parts)
        if enriched != query:
            logger.debug(f"Enriched query: {enriched[:100]}...")
        return enriched
