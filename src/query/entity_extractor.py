"""
Telecom Entity Extractor
==========================
Extracts telecom-specific entities from user queries using regex patterns.
"""

import re
from dataclasses import dataclass, field
from typing import List

from loguru import logger


@dataclass
class ExtractedEntities:
    """Container for entities extracted from a query."""
    spec_refs: List[str] = field(default_factory=list)      # e.g., "TS 38.331"
    cell_ids: List[str] = field(default_factory=list)        # e.g., "Cell 12345"
    kpis: List[str] = field(default_factory=list)            # e.g., "throughput"
    procedures: List[str] = field(default_factory=list)      # e.g., "handover"
    technologies: List[str] = field(default_factory=list)    # e.g., "5G NR"


class EntityExtractor:
    """Regex-based telecom entity extraction (no LLM needed)."""

    SPEC_PATTERN = re.compile(r"(?:TS|TR)\s*(\d{2}\.\d{3})", re.IGNORECASE)
    CELL_PATTERN = re.compile(r"(?:cell|enb|gnb|site)\s*[#:]?\s*(\d{3,})", re.IGNORECASE)

    KPI_KEYWORDS = [
        "throughput", "latency", "rsrp", "rsrq", "sinr", "bler",
        "packet loss", "handover success", "call drop", "rrc setup",
        "prb utilization", "cqi", "mcs", "bler", "retransmission",
    ]

    PROCEDURE_KEYWORDS = [
        "handover", "rrc connection", "attach", "detach", "paging",
        "random access", "bearer setup", "pdu session", "registration",
        "authentication", "security mode", "measurement report",
    ]

    TECH_KEYWORDS = [
        "5g nr", "lte", "o-ran", "oran", "nsa", "sa", "mmwave",
        "sub-6", "massive mimo", "beamforming", "carrier aggregation",
        "dual connectivity", "network slicing", "mec",
    ]

    def extract(self, query: str) -> ExtractedEntities:
        """Extract telecom entities from a query string."""
        entities = ExtractedEntities()

        # Spec references
        entities.spec_refs = [f"TS {m}" for m in self.SPEC_PATTERN.findall(query)]

        # Cell IDs
        entities.cell_ids = self.CELL_PATTERN.findall(query)

        # Keywords
        query_lower = query.lower()
        entities.kpis = [k for k in self.KPI_KEYWORDS if k in query_lower]
        entities.procedures = [p for p in self.PROCEDURE_KEYWORDS if p in query_lower]
        entities.technologies = [t for t in self.TECH_KEYWORDS if t in query_lower]

        logger.debug(f"Extracted entities: {entities}")
        return entities
