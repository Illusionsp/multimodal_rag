"""
rag_pipeline.py
----------------
Top-level orchestrator. Two entry points:

  - ingest(pdf_path): parse -> OCR fallback for scanned pages -> chart/image
    captioning -> embed -> store. Returns ingestion stats for the UI.

  - answer(question): embed query -> retrieve top-k chunks (text/table/
    image descriptions, optionally filtered by source/type) -> build a
    grounded prompt -> generate -> return answer + the source chunks used,
    so the UI can show exactly which page/table/chart backed the answer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import config
from src.chart_analyzer import ChartAnalyzer
from src.document_parser import Chunk, DocumentParser
from src.embeddings import TextEmbedder
<<<<<<< HEAD
=======
from src.ingest_utils import file_hash, load_manifest, save_manifest
>>>>>>> d248c77 (update readme)
from src.llm_client import LLMClient
from src.ocr_engine import OCRProcessor
from src.retrieval import BM25Index, CrossEncoderReranker, reciprocal_rank_fusion
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    source: str
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    ocr_recovered_chunks: int = 0
    skipped_duplicate: bool = False


@dataclass
class RetrievedSource:
    type: str
    page_number: int
    source: str
    content: str
    image_path: Optional[str] = None
    distance: Optional[float] = None


@dataclass
class RAGAnswer:
    answer: str
    sources: List[RetrievedSource] = field(default_factory=list)


SYSTEM_PROMPT = """You are a precise research assistant answering questions about \
uploaded documents that contain text, tables, and charts/infographics.

Rules:
- Answer ONLY using the provided CONTEXT. If the context does not contain \
the answer, say so plainly instead of guessing.
- When the context includes a markdown table, read it carefully - rows and \
columns, including multi-level headers - before answering questions about \
specific values.
- When the context includes a chart/image description, treat it as a \
paraphrased description of a visual, not verified numeric data; be upfront \
about that uncertainty if the question asks for exact figures from a chart.
- Cite which page and source document each piece of your answer came from, \
e.g. "(source.pdf, page 4)".
- Be concise and directly answer the question first, then add supporting detail.
"""


class MultimodalRAGPipeline:
    def __init__(self):
        self.parser = DocumentParser()
        self.ocr = OCRProcessor()
        self.chart_analyzer = ChartAnalyzer(ocr_processor=self.ocr)
        self.text_embedder = TextEmbedder()
        self.store = VectorStore()
        self.llm = LLMClient()
        self.bm25 = BM25Index()
        self.reranker = CrossEncoderReranker() if config.ENABLE_RERANKING else None
        self._rebuild_bm25_index()
<<<<<<< HEAD
        self._manifest = self._load_manifest()
=======
        self._manifest = load_manifest(config.INGEST_MANIFEST_PATH)
>>>>>>> d248c77 (update readme)

    # ------------------------------------------------------------------
    # Ingestion dedup
    # ------------------------------------------------------------------
<<<<<<< HEAD
    @staticmethod
    def _file_hash(pdf_path: str) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _load_manifest() -> Dict[str, str]:
        import json

        if config.INGEST_MANIFEST_PATH.exists():
            try:
                return json.loads(config.INGEST_MANIFEST_PATH.read_text())
            except Exception:
                return {}
        return {}

    def _save_manifest(self):
        import json

        config.INGEST_MANIFEST_PATH.write_text(json.dumps(self._manifest, indent=2))

=======
>>>>>>> d248c77 (update readme)
    def _rebuild_bm25_index(self):
        if not config.ENABLE_HYBRID_SEARCH:
            return
        data = self.store.get_all_text()
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        self.bm25.build(ids, docs)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, pdf_path: str) -> IngestStats:
        source_name = Path(pdf_path).name
<<<<<<< HEAD
        file_hash = self._file_hash(pdf_path)

        if self._manifest.get(source_name) == file_hash:
=======
        file_sha = file_hash(pdf_path)

        if self._manifest.get(source_name) == file_sha:
>>>>>>> d248c77 (update readme)
            logger.info("Skipping %s - identical file already ingested", source_name)
            return IngestStats(source=source_name, skipped_duplicate=True)

        chunks: List[Chunk] = self.parser.parse(pdf_path)

        # Full-page OCR fallback for pages unstructured found ~no text on.
        scanned_pages = getattr(self.parser, "scanned_pages", [])
        ocr_chunks = self.ocr.process_scanned_pages(pdf_path, scanned_pages)
        chunks.extend(ocr_chunks)

        # Turn every extracted image into a searchable text description.
        for chunk in chunks:
            if chunk.type == "image" and chunk.image_path:
                description = self.chart_analyzer.analyze(chunk.image_path)
                chunk.content = description or chunk.content or "(no description available)"

        self._embed_and_store(chunks)
        self._rebuild_bm25_index()

<<<<<<< HEAD
        self._manifest[source_name] = file_hash
        self._save_manifest()
=======
        self._manifest[source_name] = file_sha
        save_manifest(config.INGEST_MANIFEST_PATH, self._manifest)
>>>>>>> d248c77 (update readme)

        stats = IngestStats(
            source=source_name,
            text_chunks=sum(1 for c in chunks if c.type == "text"),
            table_chunks=sum(1 for c in chunks if c.type == "table"),
            image_chunks=sum(1 for c in chunks if c.type == "image"),
            ocr_recovered_chunks=len(ocr_chunks),
        )
        logger.info("Ingest complete: %s", stats)
        return stats

    def _embed_and_store(self, chunks: List[Chunk]):
        usable = [c for c in chunks if c.content and c.content.strip()]
        if not usable:
            return
        texts = [c.content for c in usable]
        embeddings = self.text_embedder.embed(texts).tolist()
        ids = [c.id for c in usable]
        metadatas = [c.to_metadata() for c in usable]
        self.store.add_text_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: int = config.DEFAULT_TOP_K,
        type_filter: Optional[List[str]] = None,
        source_filter: Optional[str] = None,
    ) -> RAGAnswer:
        sources = self._retrieve(question, top_k, type_filter, source_filter)
        if not sources:
            return RAGAnswer(
                answer="I couldn't find anything relevant in the ingested documents to answer that.",
                sources=[],
            )

        context = self._format_context(sources)
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        answer_text = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        return RAGAnswer(answer=answer_text, sources=sources)

    def _retrieve(
        self,
        question: str,
        top_k: int,
        type_filter: Optional[List[str]],
        source_filter: Optional[str],
    ) -> List[RetrievedSource]:
        candidate_k = max(top_k * config.RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
        query_embedding = self.text_embedder.embed_one(question)
        dense_results = self.store.query_text(
            query_embedding, top_k=candidate_k, type_filter=type_filter, source_filter=source_filter
        )
        dense_ids = (dense_results.get("ids") or [[]])[0]

        if not config.ENABLE_HYBRID_SEARCH or not self.bm25.is_ready():
            fused_ids = dense_ids
        else:
            bm25_hits = self.bm25.query(question, candidate_k)
            bm25_ids = [doc_id for doc_id, _ in bm25_hits]
            fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[:candidate_k]

        if not fused_ids:
            return []

        fetched = self.store.get_by_ids(fused_ids)
        id_to_doc = dict(zip(fetched.get("ids", []), fetched.get("documents", [])))
        id_to_meta = dict(zip(fetched.get("ids", []), fetched.get("metadatas", [])))
        # honor filters again in case a BM25-only hit slipped past them
        final_ids = [
            i for i in fused_ids
            if i in id_to_meta
            and (not type_filter or id_to_meta[i].get("type") in type_filter)
            and (not source_filter or id_to_meta[i].get("source") == source_filter)
        ]

        if config.ENABLE_RERANKING and self.reranker and final_ids:
            candidates = [(i, id_to_doc[i]) for i in final_ids]
            final_ids = self.reranker.rerank(question, candidates, top_k)
        else:
            final_ids = final_ids[:top_k]

        return [
            RetrievedSource(
                type=id_to_meta[i].get("type", "text"),
                page_number=id_to_meta[i].get("page_number", 0),
                source=id_to_meta[i].get("source", "unknown"),
                content=id_to_doc[i],
                image_path=id_to_meta[i].get("image_path") or None,
            )
            for i in final_ids
        ]

    @staticmethod
    def _format_context(sources: List[RetrievedSource]) -> str:
        blocks = []
        for s in sources:
            label = f"[{s.type.upper()} | {s.source} p.{s.page_number}]"
            blocks.append(f"{label}\n{s.content}")
        return "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------------
    def list_sources(self) -> List[str]:
        return self.store.list_sources()

    def reset(self):
        self.store.reset()
        self.bm25 = BM25Index()
        self._manifest = {}
<<<<<<< HEAD
        self._save_manifest()
=======
        save_manifest(config.INGEST_MANIFEST_PATH, self._manifest)
>>>>>>> d248c77 (update readme)
