"""
Telecom Domain Preprocessor
==============================
Handles acronym expansion, normalization, and telecom-specific
text cleanup before chunking and embedding.
"""

import re
from typing import Dict, List

from loguru import logger
from src.utils.telecom_dictionary import TELECOM_ACRONYMS


class TelecomPreprocessor:
    """Preprocesses telecom specification text for better retrieval."""

    def __init__(self):
        self.acronyms = TELECOM_ACRONYMS
        # Common noise patterns in 3GPP specs
        self.noise_patterns = [
            re.compile(r"3GPP\s+TS\s+\d+\.\d+\s+V\d+\.\d+\.\d+.*$", re.MULTILINE),  # Headers
            re.compile(r"ETSI\s+TS\s+\d+.*$", re.MULTILINE),
            re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # Standalone page numbers
        ]

    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps."""
        text = self.clean_noise(text)
        text = self.normalize_whitespace(text)
        return text

    def clean_noise(self, text: str) -> str:
        """Remove repeated headers, footers, and page numbers."""
        for pattern in self.noise_patterns:
            text = pattern.sub("", text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def expand_acronyms(self, text: str) -> str:
        """
        Expand telecom acronyms on first occurrence.
        E.g., "UE" → "UE (User Equipment)"
        """
        expanded = set()
        words = text.split()
        result = []

        for word in words:
            clean = re.sub(r"[^A-Za-z0-9-]", "", word)
            if clean.upper() in self.acronyms and clean.upper() not in expanded:
                expansion = self.acronyms[clean.upper()]
                result.append(f"{word} ({expansion})")
                expanded.add(clean.upper())
            else:
                result.append(word)

        return " ".join(result)

    def enrich_query(self, query: str) -> str:
        """Enrich a user query with acronym expansions for better retrieval."""
        return self.expand_acronyms(query)
