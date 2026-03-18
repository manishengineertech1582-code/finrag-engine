# tests/test_evaluation.py

"""
Unit tests for src/evaluation.py

Tests confirm that Hit@K and MRR work correctly with the metadata
keys that PyPDFLoader actually sets ('page', 'source') — not the
missing 'id' key that caused the original bug.
"""

import pytest
from src.evaluation import hit_at_k, mean_reciprocal_rank, Document


# -------------------------------------------------------
# Fixtures
# -------------------------------------------------------
def make_doc(page=None, source=None, doc_id=None, chunk_id=None):
    """Create a Document with realistic PyPDFLoader-style metadata."""
    metadata = {}
    if page is not None:
        metadata["page"] = page
    if source is not None:
        metadata["source"] = source
    if doc_id is not None:
        metadata["id"] = doc_id
    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id
    return Document(metadata=metadata)


# -------------------------------------------------------
# Hit@K Tests
# -------------------------------------------------------
class TestHitAtK:

    def test_hit_found_by_page(self):
        docs = [make_doc(page=2), make_doc(page=5), make_doc(page=8)]
        assert hit_at_k(docs, ground_truth_doc_id=5) == 1

    def test_hit_not_found(self):
        docs = [make_doc(page=1), make_doc(page=2)]
        assert hit_at_k(docs, ground_truth_doc_id=99) == 0

    def test_hit_found_by_chunk_id(self):
        docs = [make_doc(chunk_id=3), make_doc(chunk_id=7)]
        assert hit_at_k(docs, ground_truth_doc_id=7) == 1

    def test_hit_found_by_explicit_id(self):
        docs = [make_doc(doc_id="abc"), make_doc(doc_id="xyz")]
        assert hit_at_k(docs, ground_truth_doc_id="xyz") == 1

    def test_empty_docs_returns_zero(self):
        assert hit_at_k([], ground_truth_doc_id=1) == 0

    def test_none_ground_truth_returns_zero(self):
        docs = [make_doc(page=1)]
        assert hit_at_k(docs, ground_truth_doc_id=None) == 0


# -------------------------------------------------------
# MRR Tests
# -------------------------------------------------------
class TestMRR:

    def test_mrr_first_rank(self):
        docs = [make_doc(page=3), make_doc(page=7)]
        assert mean_reciprocal_rank(docs, ground_truth_doc_id=3) == pytest.approx(1.0)

    def test_mrr_second_rank(self):
        docs = [make_doc(page=1), make_doc(page=4)]
        assert mean_reciprocal_rank(docs, ground_truth_doc_id=4) == pytest.approx(0.5)

    def test_mrr_not_found(self):
        docs = [make_doc(page=1), make_doc(page=2)]
        assert mean_reciprocal_rank(docs, ground_truth_doc_id=99) == 0.0

    def test_mrr_empty_docs(self):
        assert mean_reciprocal_rank([], ground_truth_doc_id=1) == 0.0

    def test_mrr_none_ground_truth(self):
        docs = [make_doc(page=1)]
        assert mean_reciprocal_rank(docs, ground_truth_doc_id=None) == 0.0
