# tests/test_chunking.py
"""Unit tests for all three chunking strategies."""

import pytest
from langchain_core.documents import Document
from src.chunking import (
    fixed_chunking,
    recursive_chunking,
    semantic_chunking,
    chunk_documents,
)

SAMPLE_TEXT = (
    "Artificial intelligence is transforming industries worldwide. "
    "Machine learning models are trained on large datasets to make predictions. "
    "Deep learning uses neural networks with many layers to learn representations. "
    "Natural language processing enables computers to understand human language. "
    "Retrieval-augmented generation combines search and generation for grounded answers. "
    "Vector databases store embeddings for fast similarity search. "
    "LangChain provides abstractions for building LLM-powered applications. "
    "FastAPI is a modern web framework for building APIs with Python. "
    "FAISS is a library for efficient similarity search on dense vectors. "
    "Cross-encoders rerank candidate passages for improved retrieval precision. "
) * 5  # ~2500 chars — enough to produce multiple chunks


@pytest.fixture
def sample_docs():
    return [Document(page_content=SAMPLE_TEXT, metadata={"source": "test.pdf", "page": 0})]


# ── Fixed chunking ────────────────────────────────────────────────────────────

class TestFixedChunking:
    def test_returns_list(self, sample_docs):
        chunks = fixed_chunking(sample_docs, chunk_size=400, chunk_overlap=50)
        assert isinstance(chunks, list)

    def test_produces_multiple_chunks(self, sample_docs):
        chunks = fixed_chunking(sample_docs, chunk_size=400, chunk_overlap=50)
        assert len(chunks) > 1, "Expected multiple chunks for a long document"

    def test_chunks_are_documents(self, sample_docs):
        chunks = fixed_chunking(sample_docs, chunk_size=400, chunk_overlap=50)
        for chunk in chunks:
            assert isinstance(chunk, Document)

    def test_metadata_preserved(self, sample_docs):
        chunks = fixed_chunking(sample_docs, chunk_size=400, chunk_overlap=50)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.pdf"

    def test_empty_input(self):
        chunks = fixed_chunking([])
        assert chunks == []


# ── Recursive chunking ────────────────────────────────────────────────────────

class TestRecursiveChunking:
    def test_returns_list(self, sample_docs):
        chunks = recursive_chunking(sample_docs, chunk_size=400, chunk_overlap=50)
        assert isinstance(chunks, list)

    def test_produces_multiple_chunks(self, sample_docs):
        chunks = recursive_chunking(sample_docs, chunk_size=400, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunks_not_exceed_size_much(self, sample_docs):
        size = 400
        chunks = recursive_chunking(sample_docs, chunk_size=size, chunk_overlap=50)
        # Allow 20% overage from splitter behaviour
        for chunk in chunks:
            assert len(chunk.page_content) <= size * 1.2, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )

    def test_metadata_preserved(self, sample_docs):
        chunks = recursive_chunking(sample_docs, chunk_size=400, chunk_overlap=50)
        for chunk in chunks:
            assert "source" in chunk.metadata


# ── Semantic chunking (without embeddings — fallback path) ────────────────────

class TestSemanticChunking:
    def test_fallback_without_embeddings(self, sample_docs):
        # Should not raise; falls back to recursive
        chunks = semantic_chunking(sample_docs, embeddings=None)
        assert len(chunks) > 0

    def test_fallback_returns_documents(self, sample_docs):
        chunks = semantic_chunking(sample_docs, embeddings=None)
        for chunk in chunks:
            assert isinstance(chunk, Document)


# ── Dispatcher ────────────────────────────────────────────────────────────────

class TestChunkDocumentsDispatcher:
    def test_fixed_strategy(self, sample_docs):
        chunks = chunk_documents(sample_docs, strategy="fixed", chunk_size=400)
        assert len(chunks) > 0

    def test_recursive_strategy(self, sample_docs):
        chunks = chunk_documents(sample_docs, strategy="recursive", chunk_size=400)
        assert len(chunks) > 0

    def test_semantic_strategy_fallback(self, sample_docs):
        chunks = chunk_documents(sample_docs, strategy="semantic", embeddings=None)
        assert len(chunks) > 0

    def test_unknown_strategy_raises(self, sample_docs):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_documents(sample_docs, strategy="magic")

    def test_case_insensitive(self, sample_docs):
        chunks = chunk_documents(sample_docs, strategy="RECURSIVE", chunk_size=400)
        assert len(chunks) > 0
