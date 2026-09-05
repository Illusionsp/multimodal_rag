"""
embeddings.py
-------------
Two embedders:

  - TextEmbedder: sentence-transformers/all-MiniLM-L6-v2. Used for ALL
    retrievable content - narrative text, markdown-ified tables, and the
    text descriptions chart_analyzer produces for images. Putting
    everything in one embedding space means a single query can retrieve
    text, tables, and charts together, ranked by relevance.

  - ImageEmbedder: CLIP (via sentence-transformers). Optional secondary
    index for pure visual similarity search ("find images that look like
    this one"), kept separate because CLIP's space isn't comparable to
    MiniLM's.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, model_name: str = config.TEXT_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading text embedding model %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0].tolist()


class ImageEmbedder:
    def __init__(self, model_name: str = config.CLIP_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading CLIP model %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.model.encode(images, batch_size=config.EMBED_BATCH_SIZE, normalize_embeddings=True)

    def embed_text_query(self, text: str) -> List[float]:
        """Embed a text query into the same CLIP space, for text->image search."""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
