# tests/test_retriever.py
"""Unit tests for BM25 retriever (no FAISS/OpenAI needed)."""

import pytest
from langchain_core.documents import Document
from src.retriever import get_bm25_retriever


def make_doc(content: str, source: str = "test.pdf") -> Document:
    return Document(page_content=content, metadata={"source": source})


DOCS = [
    make_doc("Python is a programming language used for AI and machine learning."),
    make_doc("FastAPI is a modern web framework for building REST APIs in Python."),
    make_doc("FAISS enables efficient similarity search over dense vector embeddings."),
    make_doc("LangChain provides tools for building applications with large language models."),
    make_doc("Docker containers package applications with all their dependencies."),
]


class TestBM25Retriever:
    def test_returns_documents(self):
        retriever = get_bm25_retriever(DOCS, k=3)
        results = retriever.get_relevant_documents("Python programming")
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(d, Document) for d in results)

    def test_relevant_doc_ranked_higher(self):
        retriever = get_bm25_retriever(DOCS, k=5)
        results = retriever.get_relevant_documents("FAISS vector similarity search")
        # The FAISS doc should appear in top results
        top_contents = [r.page_content for r in results[:3]]
        assert any("FAISS" in c for c in top_contents), (
            "Expected FAISS doc in top-3 results"
        )

    def test_k_limits_results(self):
        retriever = get_bm25_retriever(DOCS, k=2)
        results = retriever.get_relevant_documents("machine learning")
        assert len(results) <= 2

    def test_callable_interface(self):
        retriever = get_bm25_retriever(DOCS, k=3)
        results = retriever("Docker containers")
        assert isinstance(results, list)

    def test_empty_corpus_raises(self):
        with pytest.raises(Exception):
            retriever = get_bm25_retriever([], k=5)
            retriever.get_relevant_documents("anything")

    def test_single_doc_corpus(self):
        single = [make_doc("The only document in this corpus.")]
        retriever = get_bm25_retriever(single, k=5)
        results = retriever.get_relevant_documents("document corpus")
        assert len(results) == 1
