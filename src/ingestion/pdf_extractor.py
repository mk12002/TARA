"""
PDF Text Extractor for 3GPP/O-RAN Specifications
==================================================
Uses PyMuPDF for fast text extraction and pdfplumber for table extraction.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logger.warning("PyMuPDF not installed. Install with: pip install PyMuPDF")


@dataclass
class ExtractedPage:
    """A single extracted page from a PDF."""
    page_number: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """A fully extracted PDF document."""
    filename: str
    filepath: str
    total_pages: int
    pages: List[ExtractedPage]
    metadata: Dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class PDFExtractor:
    """
    Extracts text and tables from 3GPP/O-RAN specification PDFs.
    
    Uses PyMuPDF for fast text extraction with layout preservation.
    """

    # Common 3GPP header patterns
    SPEC_NUMBER_PATTERN = re.compile(r"3GPP\s+TS\s+(\d+\.\d+)")
    VERSION_PATTERN = re.compile(r"V(\d+\.\d+\.\d+)")
    RELEASE_PATTERN = re.compile(r"Release\s+(\d+)")
    SECTION_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE)

    def __init__(self):
        if fitz is None:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    def extract(self, filepath: str | Path) -> ExtractedDocument:
        """Extract text and metadata from a PDF file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PDF not found: {filepath}")

        logger.info(f"Extracting: {filepath.name}")

        doc = fitz.open(str(filepath))
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            pages.append(ExtractedPage(
                page_number=page_num + 1,
                text=text,
                metadata={"width": page.rect.width, "height": page.rect.height},
            ))

        doc.close()

        # Extract document-level metadata from first few pages
        header_text = "\n".join(p.text for p in pages[:5])
        metadata = self._extract_metadata(header_text)

        result = ExtractedDocument(
            filename=filepath.name,
            filepath=str(filepath),
            total_pages=len(pages),
            pages=pages,
            metadata=metadata,
        )

        logger.success(f"Extracted {len(pages)} pages from {filepath.name}")
        return result

    def extract_directory(self, dirpath: str | Path) -> List[ExtractedDocument]:
        """Extract all PDFs from a directory."""
        dirpath = Path(dirpath)
        pdf_files = sorted(dirpath.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs in {dirpath}")

        documents = []
        for pdf in pdf_files:
            try:
                documents.append(self.extract(pdf))
            except Exception as e:
                logger.error(f"Failed to extract {pdf.name}: {e}")

        return documents

    def _extract_metadata(self, text: str) -> Dict:
        """Extract 3GPP metadata from document header text."""
        metadata = {}

        spec_match = self.SPEC_NUMBER_PATTERN.search(text)
        if spec_match:
            metadata["spec_number"] = spec_match.group(1)

        version_match = self.VERSION_PATTERN.search(text)
        if version_match:
            metadata["version"] = version_match.group(1)

        release_match = self.RELEASE_PATTERN.search(text)
        if release_match:
            metadata["release"] = release_match.group(1)

        return metadata
