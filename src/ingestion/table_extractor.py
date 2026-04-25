"""
Table Extractor for 3GPP/O-RAN Specifications
===============================================
Uses pdfplumber for accurate table extraction from telecom spec PDFs.
"""

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")


class TableExtractor:
    """Extracts tables from 3GPP specification PDFs using pdfplumber."""

    def __init__(self):
        if pdfplumber is None:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

    def extract_tables(self, filepath: str | Path) -> List[Dict]:
        """
        Extract all tables from a PDF file.
        
        Returns:
            List of dicts with keys: page, table_index, headers, rows, raw
        """
        filepath = Path(filepath)
        tables = []

        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables() or []

                for table_idx, table in enumerate(page_tables):
                    if not table or len(table) < 2:
                        continue

                    # First row is typically headers
                    headers = [str(cell or "").strip() for cell in table[0]]
                    rows = [
                        [str(cell or "").strip() for cell in row]
                        for row in table[1:]
                    ]

                    tables.append({
                        "page": page_num,
                        "table_index": table_idx,
                        "headers": headers,
                        "rows": rows,
                        "raw": table,
                    })

        logger.info(f"Extracted {len(tables)} tables from {filepath.name}")
        return tables

    def table_to_markdown(self, table: Dict) -> str:
        """Convert an extracted table to markdown format."""
        headers = table["headers"]
        rows = table["rows"]

        # Build markdown
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            # Pad row if needed
            padded = row + [""] * (len(headers) - len(row))
            md += "| " + " | ".join(padded[:len(headers)]) + " |\n"

        return md
