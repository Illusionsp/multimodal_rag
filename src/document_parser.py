"""
document_parser.py
-------------------
Parses PDFs into typed "chunks" (narrative text, tables, images/charts).

Strategy:
  1. Run `unstructured.partition.pdf.partition_pdf` with hi_res layout
     detection + table-structure inference. This gives us:
       - NarrativeText / Title / ListItem elements (plain text)
       - Table elements with `metadata.text_as_html` (preserves merged
         cells / multi-level headers, unlike plain-text table dumps)
       - Extracted Image elements (figures, charts, photos) saved to disk
  2. Detect pages that unstructured could not read text from (scanned
     pages / pure images) and flag them for full-page OCR.
  3. Normalize everything into a common chunk schema:
        {
          "id": str,
          "type": "text" | "table" | "image",
          "content": str,          # text, markdown table, or caption
          "html": Optional[str],   # for tables, original HTML
          "image_path": Optional[str],
          "page_number": int,
          "source": str,           # filename
        }

Tables are additionally parsed into pandas DataFrames (via the preserved
HTML) so the UI can render them natively and so multi-level headers /
merged cells survive instead of collapsing into a flat text blob.
"""
from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    type: str                      # "text" | "table" | "image"
    content: str                   # text / markdown table / caption placeholder
    page_number: int
    source: str
    html: Optional[str] = None
    image_path: Optional[str] = None
    dataframe: Optional[pd.DataFrame] = field(default=None, repr=False)

    def to_metadata(self) -> dict:
        """Chroma metadata must be flat scalars only."""
        return {
            "type": self.type,
            "page_number": self.page_number,
            "source": self.source,
            "image_path": self.image_path or "",
            "has_html": bool(self.html),
        }


def _table_html_to_dataframe(html: str) -> Optional[pd.DataFrame]:
    """Convert unstructured's preserved table HTML into a DataFrame.

    pandas.read_html correctly reconstructs multi-level column headers
    (colspan/rowspan) from the HTML, which is why we keep the HTML around
    instead of relying on unstructured's flattened `.text`.
    """
    try:
        tables = pd.read_html(io.StringIO(html))
        if tables:
            return tables[0]
    except Exception as e:
        logger.warning("Failed to parse table HTML into DataFrame: %s", e)
    return None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex column header (from merged/nested table headers)
    into single readable strings, e.g. ('Region', 'North') -> 'Region - North'.
    Leaves single-level headers untouched.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [str(p) for p in tup if p and "Unnamed" not in str(p)]
            new_cols.append(" - ".join(dict.fromkeys(parts)) or "col")
        df = df.copy()
        df.columns = new_cols
    return df


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    df = _flatten_columns(df)
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class DocumentParser:
    def __init__(
        self,
        strategy: str = config.UNSTRUCTURED_STRATEGY,
        image_output_dir: Path = config.IMAGE_DIR,
    ):
        self.strategy = strategy
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_path: str) -> List[Chunk]:
        """Parse a single PDF into a flat list of Chunks."""
        from unstructured.partition.pdf import partition_pdf  # lazy import

        source_name = Path(pdf_path).name
        logger.info("Partitioning %s with strategy=%s", source_name, self.strategy)

        elements = partition_pdf(
            filename=pdf_path,
            strategy=self.strategy,
            infer_table_structure=config.INFER_TABLE_STRUCTURE,
            extract_image_block_types=config.EXTRACT_IMAGE_BLOCK_TYPES,
            extract_image_block_output_dir=str(self.image_output_dir),
            extract_image_block_to_payload=False,
        )

        chunks: List[Chunk] = []
        text_chars_per_page: dict = {}

        for el in elements:
            category = el.category
            page_number = getattr(el.metadata, "page_number", None) or 0
            text_chars_per_page[page_number] = text_chars_per_page.get(page_number, 0) + len(el.text or "")

            if category == "Table":
                html = getattr(el.metadata, "text_as_html", None)
                df = _table_html_to_dataframe(html) if html else None
                if df is not None:
                    df = _flatten_columns(df)
                content = _dataframe_to_markdown(df) if df is not None else (el.text or "")
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        type="table",
                        content=content,
                        html=html,
                        page_number=page_number,
                        source=source_name,
                        dataframe=df,
                    )
                )

            elif category == "Image":
                image_path = getattr(el.metadata, "image_path", None)
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        type="image",
                        content=el.text or "",   # often empty; filled in by chart_analyzer later
                        page_number=page_number,
                        source=source_name,
                        image_path=image_path,
                    )
                )

            elif category in ("NarrativeText", "Title", "ListItem", "UncategorizedText"):
                if el.text and el.text.strip():
                    for piece in _chunk_text(el.text.strip(), config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                        chunks.append(
                            Chunk(
                                id=str(uuid.uuid4()),
                                type="text",
                                content=piece,
                                page_number=page_number,
                                source=source_name,
                            )
                        )

        # Flag pages that unstructured essentially got no text from - these
        # are handled separately by OCRProcessor.detect_and_process_scanned_pages.
        self.scanned_pages = [
            p for p, n_chars in text_chars_per_page.items()
            if n_chars < config.SCANNED_PAGE_CHAR_THRESHOLD
        ]
        if self.scanned_pages:
            logger.info(
                "Pages with little/no extractable text (candidates for OCR): %s",
                self.scanned_pages,
            )

        logger.info(
            "Parsed %s -> %d text, %d table, %d image chunks",
            source_name,
            sum(1 for c in chunks if c.type == "text"),
            sum(1 for c in chunks if c.type == "table"),
            sum(1 for c in chunks if c.type == "image"),
        )
        return chunks
