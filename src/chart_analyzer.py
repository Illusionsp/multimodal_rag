"""
chart_analyzer.py
------------------
Turns an extracted chart/figure/infographic image into a rich text
description that can be embedded and retrieved like any other chunk.

No open-source model reliably reads exact data values off a chart, so
instead of pretending otherwise this module builds an accurate, *searchable*
description out of three complementary signals:

  1. Caption model (BLIP) - general "what is depicted" description.
  2. OCR (reused from `OCRProcessor`) - pulls literal axis labels, legend
     entries, titles, and any numbers printed on the chart. This is where
     most of the retrievable, factual value actually comes from.
  3. Heuristic chart-type detection - simple, fast OpenCV heuristics
     (line density, shape counts, color-region segmentation) to label the
     image as bar / line / pie / scatter / table / infographic / photo.
     This is intentionally a coarse classifier, not a data extractor.

The combined description is what gets embedded, so a query like "what does
the Q3 revenue bar chart show" retrieves the right figure even though no
model actually "read" the bar heights.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

import config
from src.ocr_engine import OCRProcessor

logger = logging.getLogger(__name__)


def _detect_chart_type(image: Image.Image) -> str:
    """Cheap, dependency-light heuristic chart-type guess via OpenCV.

    This is a coarse signal only - it's appended to the description so it's
    searchable ("find the pie chart about market share"), not used for any
    numeric extraction.
    """
    try:
        import cv2

        arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0] / 2,
            param1=50, param2=40, minRadius=gray.shape[0] // 6, maxRadius=0,
        )
        if circles is not None:
            return "pie_or_donut_chart"

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=gray.shape[0] // 4, maxLineGap=10)
        if lines is not None:
            horiz = sum(1 for l in lines[:, 0] if abs(l[1] - l[3]) < 5)
            vert = sum(1 for l in lines[:, 0] if abs(l[0] - l[2]) < 5)
            diag = len(lines) - horiz - vert
            if vert > 5 and vert > diag:
                return "bar_chart"
            if diag > vert and diag > 5:
                return "line_chart"

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_blobs = sum(1 for c in contours if 5 < cv2.contourArea(c) < 200)
        if small_blobs > 40:
            return "scatter_plot"

    except Exception as e:
        logger.debug("Chart-type heuristic failed, defaulting to 'chart_or_figure': %s", e)

    return "chart_or_figure"


class ChartAnalyzer:
    def __init__(
        self,
        caption_model_name: str = config.CAPTION_MODEL_NAME,
        ocr_processor: Optional[OCRProcessor] = None,
    ):
        self.caption_model_name = caption_model_name
        self.ocr = ocr_processor or OCRProcessor()
        self._processor = None
        self._model = None

    def _load_caption_model(self):
        if self._model is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            logger.info("Loading caption model %s", self.caption_model_name)
            self._processor = BlipProcessor.from_pretrained(self.caption_model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(self.caption_model_name)

    def caption(self, image: Image.Image) -> str:
        try:
            self._load_caption_model()
            inputs = self._processor(image.convert("RGB"), return_tensors="pt")
            out = self._model.generate(**inputs, max_new_tokens=config.CHART_CAPTION_MAX_NEW_TOKENS)
            return self._processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning("Caption model failed, continuing with OCR-only description: %s", e)
            return ""

    def analyze(self, image_path: str) -> str:
        """Return a single text description combining caption + OCR + chart type."""
        try:
            image = Image.open(image_path)
        except Exception as e:
            logger.warning("Could not open image %s: %s", image_path, e)
            return ""

        chart_type = _detect_chart_type(image)
        caption = self.caption(image)
        ocr_text = self.ocr.extract_text(image_path)

        parts = [f"[{chart_type.replace('_', ' ')}]"]
        if caption:
            parts.append(f"Visual description: {caption}.")
        if ocr_text.strip():
            cleaned = " ".join(ocr_text.split())
            parts.append(f"Text detected in image (titles/axis labels/legend/values): {cleaned}")
        if len(parts) == 1:
            parts.append("No caption or text could be extracted from this image.")

        return " ".join(parts)
