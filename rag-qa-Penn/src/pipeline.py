# src/pipeline.py
"""
RAG pipeline factory.

Supports:
  retriever_type : 'dense' | 'bm25' | 'hybrid'
  enable_reranking: True | False
  chunk_strategy  : 'fixed' | 'recursive' | 'semantic'
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.retriever import get_retriever, get_bm25_retriever, get_hybrid_retriever
from src.generator import build_qa_chain


def load_pipeline(
    retriever_type: str = "dense",
    enable_reranking: bool = False,
    top_k: int = 5,
):
    """
    Build the full RAG pipeline.

    Args:
        retriever_type:   'dense' | 'bm25' | 'hybrid'
        enable_reranking: If True, wraps retriever with CrossEncoder reranker.
        top_k:            Number of docs to retrieve (pre-reranking: top_k * 3).

    Returns:
        A callable qa_chain that accepts {"query": str} and returns
        {"result": str, "source_documents": list[Document]}.
    """
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # ── Choose retriever ──────────────────────────────────────────────────────
    candidate_k = top_k * 3 if enable_reranking else top_k

    if retriever_type == "dense":
        retriever = get_retriever(vectorstore, k=candidate_k)

    elif retriever_type == "bm25":
        # Reconstruct corpus from FAISS docstore for BM25
        docs = _get_docs_from_vectorstore(vectorstore)
        retriever = get_bm25_retriever(docs, k=candidate_k)

    elif retriever_type == "hybrid":
        docs = _get_docs_from_vectorstore(vectorstore)
        retriever = get_hybrid_retriever(vectorstore, docs, k=candidate_k)

    else:
        raise ValueError(f"Unknown retriever_type '{retriever_type}'. "
                         f"Choose: 'dense', 'bm25', or 'hybrid'.")

    # ── Optional cross-encoder reranking ──────────────────────────────────────
    if enable_reranking:
        from src.reranker import get_reranker
        reranker = get_reranker()

        # Wrap retriever so reranking happens transparently
        base_retriever = retriever

        class RerankedRetriever:
            def get_relevant_documents(self, query: str) -> list[Document]:
                if hasattr(base_retriever, "get_relevant_documents"):
                    candidates = base_retriever.get_relevant_documents(query)
                else:
                    candidates = base_retriever(query)
                return reranker.rerank(query, candidates, top_k=top_k)

            def __call__(self, query):
                return self.get_relevant_documents(query)

        retriever = RerankedRetriever()
        print(f"[Pipeline] Reranking enabled (top_k={top_k})")

    print(f"[Pipeline] Ready — retriever={retriever_type}, "
          f"reranking={enable_reranking}, top_k={top_k}")

    return build_qa_chain(retriever)


def _get_docs_from_vectorstore(vectorstore) -> list[Document]:
    """Extract all stored Document objects from a FAISS vectorstore."""
    docstore = vectorstore.docstore
    index_to_docstore_id = vectorstore.index_to_docstore_id
    docs = []
    for idx in range(vectorstore.index.ntotal):
        doc_id = index_to_docstore_id.get(idx)
        if doc_id:
            doc = docstore.search(doc_id)
            if isinstance(doc, Document):
                docs.append(doc)
    return docs
