"""
Telecom-Aware Document Chunker
================================
Custom chunking strategy that respects 3GPP section boundaries,
preserves metadata, and handles telecom-specific content types.

NOT a generic text splitter — this is purpose-built for 3GPP/O-RAN specs.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from hashlib import md5

from loguru import logger


@dataclass
class Chunk:
    """A single document chunk with rich metadata."""
    id: str
    text: str
    source_file: str
    spec_number: str = ""
    section: str = ""
    version: str = ""
    content_type: str = "text"  # text, table, procedure, figure
    metadata: Dict = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = md5(f"{self.source_file}:{self.section}:{self.text[:100]}".encode()).hexdigest()
        # Rough token estimate (words * 1.3)
        self.token_count = int(len(self.text.split()) * 1.3)


class TelecomChunker:
    """
    3GPP/O-RAN aware document chunker.
    
    Strategy:
    1. Split by section headers (e.g., "5.3.3 RRC connection establishment")
    2. Preserve section hierarchy in metadata
    3. Handle tables and procedures as atomic chunks
    4. Fall back to recursive splitting only for oversized sections
    """

    # 3GPP section header pattern: "5.3.3.1 General" or "A.2 Example"
    SECTION_PATTERN = re.compile(
        r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+(.+)$",
        re.MULTILINE
    )

    # Table markers
    TABLE_MARKERS = ["Table", "TABLE"]

    # Procedure markers
    PROCEDURE_MARKERS = [
        "procedure", "the ue shall", "the gnb shall",
        "the network shall", "step 1", "1)", "a)"
    ]

    def __init__(
        self,
        chunk_size: int = 512,      # Target tokens per chunk
        chunk_overlap: int = 50,    # Overlap tokens
        respect_sections: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sections = respect_sections

    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk a document into semantically meaningful pieces.
        
        Args:
            text: Full document text
            metadata: Document-level metadata (spec_number, version, etc.)
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        chunks = []

        if self.respect_sections:
            sections = self._split_by_sections(text)
            for section_id, section_title, section_text in sections:
                section_chunks = self._chunk_section(
                    section_text,
                    section_id=section_id,
                    section_title=section_title,
                    metadata=metadata,
                )
                chunks.extend(section_chunks)
        else:
            chunks = self._fallback_chunk(text, metadata)

        logger.info(
            f"Created {len(chunks)} chunks from document "
            f"(avg {sum(c.token_count for c in chunks) // max(len(chunks), 1)} tokens/chunk)"
        )
        return chunks

    def _split_by_sections(self, text: str) -> List[tuple]:
        """Split text into (section_id, section_title, section_text) tuples."""
        matches = list(self.SECTION_PATTERN.finditer(text))

        if not matches:
            return [("0", "Document", text)]

        sections = []
        for i, match in enumerate(matches):
            section_id = match.group(1)
            section_title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if section_text:  # Skip empty sections
                sections.append((section_id, section_title, section_text))

        return sections

    def _chunk_section(
        self,
        text: str,
        section_id: str,
        section_title: str,
        metadata: Dict,
    ) -> List[Chunk]:
        """Chunk a single section, respecting content type boundaries."""
        content_type = self._detect_content_type(text)
        estimated_tokens = int(len(text.split()) * 1.3)

        # If section fits in one chunk, keep it atomic
        if estimated_tokens <= self.chunk_size:
            return [Chunk(
                id="",
                text=text,
                source_file=metadata.get("filename", ""),
                spec_number=metadata.get("spec_number", ""),
                section=f"{section_id} {section_title}",
                version=metadata.get("version", ""),
                content_type=content_type,
                metadata={**metadata, "section_id": section_id, "section_title": section_title},
            )]

        # Section is too large — split by paragraphs
        return self._split_large_section(text, section_id, section_title, metadata)

    def _split_large_section(
        self,
        text: str,
        section_id: str,
        section_title: str,
        metadata: Dict,
    ) -> List[Chunk]:
        """Split an oversized section by paragraph boundaries."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_text = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = f"{current_text}\n\n{para}".strip() if current_text else para
            estimated = int(len(candidate.split()) * 1.3)

            if estimated > self.chunk_size and current_text:
                # Flush current chunk
                chunks.append(Chunk(
                    id="",
                    text=current_text,
                    source_file=metadata.get("filename", ""),
                    spec_number=metadata.get("spec_number", ""),
                    section=f"{section_id} {section_title}",
                    version=metadata.get("version", ""),
                    content_type=self._detect_content_type(current_text),
                    metadata={**metadata, "section_id": section_id, "section_title": section_title},
                ))
                current_text = para
            else:
                current_text = candidate

        # Don't forget the last chunk
        if current_text:
            chunks.append(Chunk(
                id="",
                text=current_text,
                source_file=metadata.get("filename", ""),
                spec_number=metadata.get("spec_number", ""),
                section=f"{section_id} {section_title}",
                version=metadata.get("version", ""),
                content_type=self._detect_content_type(current_text),
                metadata={**metadata, "section_id": section_id, "section_title": section_title},
            ))

        return chunks

    def _detect_content_type(self, text: str) -> str:
        """Detect if text is a table, procedure, or plain text."""
        text_lower = text.lower()[:200]

        for marker in self.TABLE_MARKERS:
            if marker.lower() in text_lower:
                return "table"

        for marker in self.PROCEDURE_MARKERS:
            if marker in text_lower:
                return "procedure"

        return "text"

    def _fallback_chunk(self, text: str, metadata: Dict) -> List[Chunk]:
        """Simple paragraph-based chunking as fallback."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            candidate = f"{current}\n\n{para}".strip() if current else para
            if int(len(candidate.split()) * 1.3) > self.chunk_size and current:
                chunks.append(Chunk(
                    id="", text=current,
                    source_file=metadata.get("filename", ""),
                    spec_number=metadata.get("spec_number", ""),
                    metadata=metadata,
                ))
                current = para
            else:
                current = candidate
        if current:
            chunks.append(Chunk(
                id="", text=current,
                source_file=metadata.get("filename", ""),
                spec_number=metadata.get("spec_number", ""),
                metadata=metadata,
            ))

        return chunks
