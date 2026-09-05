"""
Offline unit tests for retrieval.py - BM25 keyword search and reciprocal
rank fusion. No network, no embedding models.
"""
import pytest

from src.retrieval import BM25Index, reciprocal_rank_fusion

pytest.importorskip("rank_bm25")


def test_bm25_ranks_exact_numeric_match_first():
    idx = BM25Index()
    ids = ["a", "b", "c", "d"]
    docs = [
        "quarterly revenue grew 12 percent in Q3",
        "the cat sat on the mat",
        "Q3 revenue table shows 42.3 million dollars total",
        "unrelated paragraph about weather patterns",
    ]
    idx.build(ids, docs)
    hits = idx.query("Q3 revenue 42.3 million", top_k=3)
    assert hits[0][0] == "c"


def test_bm25_index_not_ready_before_build():
    idx = BM25Index()
    assert not idx.is_ready()
    assert idx.query("anything", top_k=5) == []


def test_bm25_handles_empty_corpus():
    idx = BM25Index()
    idx.build([], [])
    assert idx.query("anything", top_k=5) == []


def test_reciprocal_rank_fusion_favors_items_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "z", "w"]])
    assert fused[0] == "y"  # near the top of both rankings


def test_reciprocal_rank_fusion_preserves_all_unique_ids():
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert set(fused) == {"a", "b", "c"}
