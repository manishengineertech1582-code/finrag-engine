# tests/test_evaluation.py
"""Unit tests for retrieval evaluation metrics."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.evaluation import hit_at_k, mean_reciprocal_rank, evaluate_retriever


def make_doc(source: str, content: str = "some content") -> Document:
    return Document(page_content=content, metadata={"source": source})


# ── hit_at_k ─────────────────────────────────────────────────────────────────

class TestHitAtK:
    def test_hit_when_correct_doc_first(self):
        docs = [make_doc("data/Manishfile.pdf"), make_doc("other.pdf")]
        assert hit_at_k(docs, "Manishfile.pdf") == 1

    def test_hit_when_correct_doc_last(self):
        docs = [make_doc("other.pdf"), make_doc("data/Manishfile.pdf")]
        assert hit_at_k(docs, "Manishfile.pdf") == 1

    def test_miss_when_correct_doc_absent(self):
        docs = [make_doc("other.pdf"), make_doc("another.pdf")]
        assert hit_at_k(docs, "Manishfile.pdf") == 0

    def test_empty_retrieved_list(self):
        assert hit_at_k([], "Manishfile.pdf") == 0

    def test_partial_source_match(self):
        docs = [make_doc("data/Manishfile.pdf")]
        assert hit_at_k(docs, "Manishfile") == 1


# ── mean_reciprocal_rank ──────────────────────────────────────────────────────

class TestMRR:
    def test_rank_1(self):
        docs = [make_doc("data/Manishfile.pdf")]
        assert mean_reciprocal_rank(docs, "Manishfile.pdf") == pytest.approx(1.0)

    def test_rank_2(self):
        docs = [make_doc("other.pdf"), make_doc("data/Manishfile.pdf")]
        assert mean_reciprocal_rank(docs, "Manishfile.pdf") == pytest.approx(0.5)

    def test_rank_3(self):
        docs = [make_doc("a.pdf"), make_doc("b.pdf"), make_doc("data/Manishfile.pdf")]
        assert mean_reciprocal_rank(docs, "Manishfile.pdf") == pytest.approx(1 / 3)

    def test_not_found_returns_zero(self):
        docs = [make_doc("other.pdf")]
        assert mean_reciprocal_rank(docs, "Manishfile.pdf") == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        assert mean_reciprocal_rank([], "Manishfile.pdf") == pytest.approx(0.0)


# ── evaluate_retriever ────────────────────────────────────────────────────────

class TestEvaluateRetriever:
    def _make_retriever(self, return_docs):
        mock = MagicMock()
        mock.get_relevant_documents.return_value = return_docs
        return mock

    def test_perfect_retriever(self):
        docs = [make_doc("data/Manishfile.pdf")]
        retriever = self._make_retriever(docs)
        dataset = [{"question": "Q1?", "source": "Manishfile.pdf"}]
        result = evaluate_retriever(retriever, dataset)
        assert result["hit_rate"] == 1.0
        assert result["mrr"] == 1.0

    def test_zero_retriever(self):
        docs = [make_doc("wrong.pdf")]
        retriever = self._make_retriever(docs)
        dataset = [{"question": "Q1?", "source": "Manishfile.pdf"}]
        result = evaluate_retriever(retriever, dataset)
        assert result["hit_rate"] == 0.0
        assert result["mrr"] == 0.0

    def test_partial_hits(self):
        good_doc = make_doc("data/Manishfile.pdf")
        bad_doc = make_doc("wrong.pdf")

        call_count = [0]
        def side_effect(query):
            call_count[0] += 1
            return [good_doc] if call_count[0] == 1 else [bad_doc]

        mock = MagicMock()
        mock.get_relevant_documents.side_effect = side_effect

        dataset = [
            {"question": "Q1?", "source": "Manishfile.pdf"},
            {"question": "Q2?", "source": "Manishfile.pdf"},
        ]
        result = evaluate_retriever(mock, dataset)
        assert result["hit_rate"] == 0.5
        assert result["total"] == 2
        assert result["hits"] == 1

    def test_empty_dataset(self):
        retriever = self._make_retriever([])
        result = evaluate_retriever(retriever, [])
        assert result["hit_rate"] == 0.0
        assert result["mrr"] == 0.0
        assert result["total"] == 0
