# tests/test_reranker.py
"""Unit tests for CrossEncoderReranker (mocked model to avoid downloading weights)."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.reranker import CrossEncoderReranker


def make_doc(content: str, source: str = "test.pdf") -> Document:
    return Document(page_content=content, metadata={"source": source})


CANDIDATES = [
    make_doc("Python is used for machine learning and data science."),
    make_doc("FastAPI is a modern REST API framework."),
    make_doc("FAISS enables fast approximate nearest-neighbour search."),
    make_doc("LangChain builds LLM-powered applications."),
    make_doc("Docker packages applications in containers."),
]


@pytest.fixture
def mock_reranker():
    """CrossEncoderReranker with the CrossEncoder model mocked out."""
    with patch("src.reranker.CrossEncoderReranker.__init__") as mock_init:
        mock_init.return_value = None
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "mock-model"

        # Simulate scores: doc at index 2 (FAISS) gets highest score
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.3, 0.95, 0.4, 0.2]
        reranker.model = mock_model
        yield reranker


class TestCrossEncoderReranker:
    def test_reranks_documents(self, mock_reranker):
        result = mock_reranker.rerank("FAISS similarity search", CANDIDATES, top_k=3)
        assert len(result) == 3

    def test_highest_score_ranked_first(self, mock_reranker):
        result = mock_reranker.rerank("FAISS similarity search", CANDIDATES, top_k=5)
        # Score 0.95 was assigned to CANDIDATES[2] (FAISS doc)
        assert "FAISS" in result[0].page_content

    def test_top_k_limits_output(self, mock_reranker):
        result = mock_reranker.rerank("query", CANDIDATES, top_k=2)
        assert len(result) == 2

    def test_rerank_score_in_metadata(self, mock_reranker):
        result = mock_reranker.rerank("query", CANDIDATES, top_k=3)
        for doc in result:
            assert "rerank_score" in doc.metadata

    def test_empty_candidates_returns_empty(self, mock_reranker):
        result = mock_reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_top_k_larger_than_candidates(self, mock_reranker):
        small = CANDIDATES[:2]
        mock_reranker.model.predict.return_value = [0.8, 0.3]
        result = mock_reranker.rerank("query", small, top_k=10)
        assert len(result) == 2  # can't return more than we have
