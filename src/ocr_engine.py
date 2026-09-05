"""
ocr_engine.py
-------------
Handles text extraction for scanned pages and standalone images where the
PDF has no embedded text layer.

Two interchangeable backends (chosen via config.OCR_ENGINE):
  - "tesseract": pytesseract wrapping the system `tesseract` binary. Light,
    easy to install, good enough for clean scans.
  - "paddleocr":  PaddleOCR. Heavier install but noticeably better on noisy
    scans, rotated text, and dense infographic layouts.

Both are wrapped behind the same `OCRProcessor` interface so the rest of the
pipeline never needs to know which one is active.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List

from PIL import Image

import config
from src.document_parser import Chunk

logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(self, engine: str = config.OCR_ENGINE):
        self.engine = engine
        self._paddle = None  # lazy-loaded singleton, PaddleOCR init is slow

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _ocr_with_tesseract(self, image: Image.Image) -> str:
        import pytesseract

        return pytesseract.image_to_string(image, lang=config.TESSERACT_LANG)

    def _ocr_with_paddle(self, image_path: str) -> str:
        if self._paddle is None:
            from paddleocr import PaddleOCR

            self._paddle = PaddleOCR(use_angle_cls=True, lang=config.OCR_LANGUAGES, show_log=False)
        result = self._paddle.ocr(image_path, cls=True)
        lines = []
        for page in result or []:
            for line in page or []:
                # line = [box, (text, confidence)]
                lines.append(line[1][0])
        return "\n".join(lines)

    def extract_text(self, image_path: str) -> str:
        """Run OCR on a single image file and return raw text."""
        try:
            if self.engine == "paddleocr":
                return self._ocr_with_paddle(image_path)
            with Image.open(image_path) as img:
                return self._ocr_with_tesseract(img.convert("RGB"))
        except Exception as e:
            logger.warning("OCR failed on %s: %s", image_path, e)
            return ""

    # ------------------------------------------------------------------
    # Scanned-page fallback
    # ------------------------------------------------------------------
    def process_scanned_pages(
        self,
        pdf_path: str,
        page_numbers: List[int],
        output_dir: Path = config.IMAGE_DIR,
    ) -> List[Chunk]:
        """Rasterize specific pages of a PDF and OCR them.

        Used when `DocumentParser` detects a page with (near) zero
        extractable text, i.e. the page is a scanned image rather than
        digitally-authored text/tables.
        """
        if not page_numbers:
            return []

        from pdf2image import convert_from_path

        source_name = Path(pdf_path).name
        chunks: List[Chunk] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for page_num in page_numbers:
            try:
                images = convert_from_path(
                    pdf_path, first_page=page_num, last_page=page_num, dpi=300
                )
            except Exception as e:
                logger.warning("Could not rasterize page %d of %s: %s", page_num, source_name, e)
                continue

            for img in images:
                img_path = output_dir / f"{Path(pdf_path).stem}_scanned_p{page_num}.png"
                img.save(img_path)

                text = self._ocr_with_paddle(str(img_path)) if self.engine == "paddleocr" \
                    else self._ocr_with_tesseract(img)

                if text.strip():
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            type="text",
                            content=text.strip(),
                            page_number=page_num,
                            source=source_name,
                            image_path=str(img_path),
                        )
                    )
        logger.info("OCR fallback recovered %d text chunks from %d scanned pages", len(chunks), len(page_numbers))
        return chunks
