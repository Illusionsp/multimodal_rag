"""
vector_store.py
----------------
Thin wrapper around a persistent ChromaDB client with two collections:

  - `documents` (config.TEXT_COLLECTION): unified MiniLM space holding
    text chunks, markdown-table chunks, and chart/image text descriptions.
    This is the primary collection every query hits.

  - `images` (config.IMAGE_COLLECTION): CLIP space holding raw image
    embeddings, used only for optional "visually similar image" search.

Metadata stored per entry lets the UI reconstruct page number, source
document, chunk type, and (for tables/images) how to re-render the original
content (dataframe / image file) rather than just showing embedded text.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

import config

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, persist_dir: str = str(config.CHROMA_DIR)):
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.text_collection = self.client.get_or_create_collection(
            name=config.TEXT_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.get_or_create_collection(
            name=config.IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_text_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
    ):
        if not ids:
            return
        self.text_collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info("Upserted %d chunks into '%s'", len(ids), config.TEXT_COLLECTION)

    def add_images(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ):
        if not ids:
            return
        self.image_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        logger.info("Upserted %d images into '%s'", len(ids), config.IMAGE_COLLECTION)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def query_text(
        self,
        query_embedding: List[float],
        top_k: int = config.DEFAULT_TOP_K,
        type_filter: Optional[List[str]] = None,
        source_filter: Optional[str] = None,
    ) -> Dict:
        where = {}
        clauses = []
        if type_filter:
            clauses.append({"type": {"$in": type_filter}})
        if source_filter:
            clauses.append({"source": source_filter})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        return self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or None,
        )

    def query_images(self, query_embedding: List[float], top_k: int = config.DEFAULT_TOP_K) -> Dict:
        return self.image_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def get_all_text(self) -> Dict:
        """Full corpus (ids/documents/metadatas) - used to (re)build the BM25 index."""
        return self.text_collection.get(include=["documents", "metadatas"])

    def get_by_ids(self, ids: List[str]) -> Dict:
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.text_collection.get(ids=ids, include=["documents", "metadatas"])

    def list_sources(self) -> List[str]:
        try:
            all_meta = self.text_collection.get(include=["metadatas"])["metadatas"]
            return sorted({m["source"] for m in all_meta if m and "source" in m})
        except Exception:
            return []

    def reset(self):
        self.client.delete_collection(config.TEXT_COLLECTION)
        self.client.delete_collection(config.IMAGE_COLLECTION)
        self.text_collection = self.client.get_or_create_collection(
            name=config.TEXT_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.get_or_create_collection(
            name=config.IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector store reset.")
