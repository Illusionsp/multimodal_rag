"""
Offline unit tests for ingest_utils.py - file hashing and manifest
persistence used by ingestion dedup. No network, no PDF parsing.
"""
import json

from src.ingest_utils import file_hash, load_manifest, save_manifest


def test_file_hash_is_deterministic(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"same bytes every time")
    assert file_hash(str(f)) == file_hash(str(f))


def test_file_hash_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.pdf"
    f2 = tmp_path / "b.pdf"
    f1.write_bytes(b"content one")
    f2.write_bytes(b"content two")
    assert file_hash(str(f1)) != file_hash(str(f2))


def test_load_manifest_returns_empty_dict_when_missing(tmp_path):
    missing = tmp_path / "no_manifest.json"
    assert load_manifest(missing) == {}


def test_load_manifest_returns_empty_dict_on_corrupt_json(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text("{not valid json")
    assert load_manifest(bad) == {}


def test_save_and_load_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {"report.pdf": "abc123", "invoice.pdf": "def456"}
    save_manifest(path, manifest)
    assert json.loads(path.read_text()) == manifest
    assert load_manifest(path) == manifest
