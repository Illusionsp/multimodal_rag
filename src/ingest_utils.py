"""
ingest_utils.py
----------------
Small, dependency-light helpers used by the ingestion dedup logic.
Kept separate from rag_pipeline.py (which pulls in torch/transformers/
chromadb) so this module - and the tests for it - stay fast and offline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


def file_hash(path: str) -> str:
    """SHA-256 of a file's contents, for detecting duplicate uploads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    """Load the {filename: sha256} ingestion manifest, or {} if absent/corrupt."""
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_manifest(manifest_path: Path, manifest: Dict[str, str]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2))
