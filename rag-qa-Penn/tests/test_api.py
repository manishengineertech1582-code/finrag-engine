# tests/test_api.py
"""Integration tests for FastAPI endpoints using mocked pipeline."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app

client = TestClient(app)


def make_mock_chain(answer: str = "Paris is the capital of France."):
    """Build a mock qa_chain that returns a fixed answer."""
    mock_doc = Document(
        page_content="Paris is the capital of France.",
        metadata={"source": "data/Manishfile.pdf", "page": 0},
    )
    mock_chain = MagicMock()
    mock_chain.return_value = {
        "result": answer,
        "source_documents": [mock_doc],
    }
    return mock_chain


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_status_healthy(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_returns_available_retrievers(self):
        response = client.get("/health")
        data = response.json()
        assert "available_retrievers" in data
        assert "dense" in data["available_retrievers"]
        assert "bm25" in data["available_retrievers"]
        assert "hybrid" in data["available_retrievers"]

    def test_returns_chunking_strategies(self):
        response = client.get("/health")
        data = response.json()
        assert "chunking_strategies" in data
        assert "semantic" in data["chunking_strategies"]


# ── POST /ask ─────────────────────────────────────────────────────────────────

class TestAskEndpoint:
    def test_returns_answer(self):
        with patch("app.routes._get_pipeline", return_value=make_mock_chain()):
            response = client.post("/ask", json={"question": "What is the capital of France?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "Paris" in data["answer"]

    def test_returns_sources(self):
        with patch("app.routes._get_pipeline", return_value=make_mock_chain()):
            response = client.post("/ask", json={"question": "What is the capital?"})
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) > 0

    def test_returns_retriever_type(self):
        with patch("app.routes._get_pipeline", return_value=make_mock_chain()):
            response = client.post("/ask", json={
                "question": "What is AI?",
                "retriever_type": "bm25",
            })
        data = response.json()
        assert data["retriever_type"] == "bm25"

    def test_reranking_flag_reflected(self):
        with patch("app.routes._get_pipeline", return_value=make_mock_chain()):
            response = client.post("/ask", json={
                "question": "What is AI?",
                "enable_reranking": True,
            })
        data = response.json()
        assert data["reranking_applied"] is True

    def test_empty_question_returns_422(self):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_missing_question_returns_422(self):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_invalid_json_returns_422(self):
        response = client.post("/ask", content="not json",
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422
