# src/retriever.py
"""
Three retriever strategies:
  - dense  : FAISS cosine similarity (default)
  - bm25   : BM25 sparse keyword retrieval (baseline)
  - hybrid : BM25 + dense fusion via Reciprocal Rank Fusion
"""

from langchain_core.documents import Document


# ── 1. Dense retriever (FAISS) ────────────────────────────────────────────────

def get_retriever(vectorstore, k: int = 5):
    """Standard dense similarity retriever backed by FAISS."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ── 2. BM25 retriever ─────────────────────────────────────────────────────────

def get_bm25_retriever(documents: list[Document], k: int = 5):
    """
    Sparse keyword retriever using BM25Okapi.
    Useful as a strong baseline and for handling exact-match queries.

    Args:
        documents: List of LangChain Document objects (the chunked corpus).
        k:         Number of documents to return.

    Returns:
        A callable retriever: query_str -> list[Document]
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "rank-bm25 is not installed. Run: pip install rank-bm25"
        )

    # Tokenise corpus once at build time
    corpus_texts = [doc.page_content for doc in documents]
    tokenised_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenised_corpus)

    class BM25Retriever:
        def __init__(self, bm25_index, docs, top_k):
            self.bm25 = bm25_index
            self.docs = docs
            self.k = top_k

        def get_relevant_documents(self, query: str) -> list[Document]:
            tokenised_query = query.lower().split()
            scores = self.bm25.get_scores(tokenised_query)
            # Get top-k indices sorted by score descending
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: self.k]
            return [self.docs[i] for i in top_indices]

        # Make it compatible with LangChain chain invoke
        def __call__(self, query: str) -> list[Document]:
            return self.get_relevant_documents(query)

    print(f"[BM25] Index built on {len(documents)} chunks")
    return BM25Retriever(bm25, documents, k)


# ── 3. Hybrid retriever (BM25 + Dense via RRF) ───────────────────────────────

def get_hybrid_retriever(vectorstore, documents: list[Document], k: int = 5):
    """
    Reciprocal Rank Fusion of dense (FAISS) and sparse (BM25) results.
    Combines lexical precision of BM25 with semantic recall of dense embeddings.

    RRF score = sum(1 / (rank + 60)) across both ranked lists.

    Args:
        vectorstore: FAISS vectorstore for dense retrieval.
        documents:   Full corpus for BM25 index construction.
        k:           Final number of documents to return.

    Returns:
        A callable: query_str -> list[Document]
    """
    dense_retriever = get_retriever(vectorstore, k=k * 2)
    bm25_retriever = get_bm25_retriever(documents, k=k * 2)

    RRF_K = 60  # standard RRF constant

    class HybridRetriever:
        def get_relevant_documents(self, query: str) -> list[Document]:
            dense_docs = dense_retriever.get_relevant_documents(query)
            bm25_docs = bm25_retriever.get_relevant_documents(query)

            # Score each unique document by its RRF rank contribution
            scores: dict[str, float] = {}
            doc_map: dict[str, Document] = {}

            for rank, doc in enumerate(dense_docs):
                key = doc.page_content[:120]  # stable identity key
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank + RRF_K)
                doc_map[key] = doc

            for rank, doc in enumerate(bm25_docs):
                key = doc.page_content[:120]
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank + RRF_K)
                doc_map[key] = doc

            ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
            return [doc_map[key] for key in ranked_keys[:k]]

        def __call__(self, query: str) -> list[Document]:
            return self.get_relevant_documents(query)

    print(f"[Hybrid] RRF retriever ready (dense + BM25, k={k})")
    return HybridRetriever()
