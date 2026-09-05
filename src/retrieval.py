"""
retrieval.py
------------
Two additions on top of plain dense (embedding) search:

  1. BM25Index - classic keyword search over the same corpus stored in
     Chroma. Dense embeddings are great for semantic similarity but often
     miss exact tokens that matter most in tables/charts - specific
     numbers, product codes, dates, column labels. BM25 catches those.

  2. reciprocal_rank_fusion - merges the dense-search ranking and the
     BM25 ranking into one candidate list without needing to normalize
     two incomparable score scales.

  3. CrossEncoderReranker - a small, free cross-encoder
     (cross-encoder/ms-marco-MiniLM-L-6-v2) that scores each fused
     candidate directly against the query text. This is the step that
     actually improves final precision - fusion produces a *reasonable*
     candidate pool, the reranker picks the *best* few from it.

Both are optional (config.ENABLE_HYBRID_SEARCH / ENABLE_RERANKING) so the
pipeline degrades gracefully to plain dense top-k if disabled or if the
extra dependencies aren't installed.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

import config

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Thin wrapper around rank_bm25, rebuilt from the vector store's full
    corpus. Fine for demo/small-to-medium document sets; for large corpora
    swap this for a persistent inverted index (e.g. Whoosh, Elasticsearch).
    """

    def __init__(self):
        self._bm25 = None
        self.ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    def build(self, ids: List[str], documents: List[str]):
        from rank_bm25 import BM25Okapi

        self.ids = ids
        self._tokenized_corpus = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        logger.info("BM25 index built over %d documents", len(ids))

    def is_ready(self) -> bool:
        return self._bm25 is not None

    def query(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.is_ready():
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0]


def reciprocal_rank_fusion(
    rankings: List[List[str]], k: int = config.RRF_K
) -> List[str]:
    """Fuse multiple ranked id lists into one, by reciprocal-rank score.

    RRF needs no score normalization across retrievers (dense cosine
    distance and BM25 scores aren't on the same scale), which is why it's
    the standard choice for hybrid search fusion.
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]


class CrossEncoderReranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s", self.model_name)
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: List[Tuple[str, str]], top_k: int) -> List[str]:
        """candidates: list of (id, text). Returns ids sorted by relevance, best first."""
        if not candidates:
            return []
        try:
            self._load()
            pairs = [(query, text) for _, text in candidates]
            scores = self._model.predict(pairs)
            ranked = sorted(zip([c[0] for c in candidates], scores), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, _ in ranked[:top_k]]
        except Exception as e:
            logger.warning("Reranking failed (%s), falling back to fusion order", e)
            return [doc_id for doc_id, _ in candidates[:top_k]]
