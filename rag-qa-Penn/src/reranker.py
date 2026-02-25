# src/reranker.py
"""
Cross-encoder reranker using sentence-transformers.

Pipeline:
  Dense/BM25 retrieval (top-k * 3 candidates)
      → CrossEncoder scores every (query, passage) pair
      → Return top-k by rerank score

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage ranking
  - ~22M params, fast inference on CPU
  - Typically +10–20% MRR over bi-encoder retrieval alone
"""

from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for passage reranking.

    Usage:
        reranker = CrossEncoderReranker()
        top_docs = reranker.rerank(query, candidate_docs, top_k=5)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            )

        print(f"[CrossEncoderReranker] Loading model: {model_name} ...")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print("[CrossEncoderReranker] Model ready.")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Score every (query, document) pair and return top_k by score.

        Args:
            query:     User's natural language question.
            documents: Candidate documents from first-stage retrieval.
            top_k:     How many to return after reranking.

        Returns:
            Reranked list of Documents (best first), length <= top_k.
        """
        if not documents:
            return []

        # Build (query, passage) pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # CrossEncoder returns a raw logit score per pair
        scores = self.model.predict(pairs)

        # Sort by score descending
        ranked = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )

        top_docs = [doc for doc, _ in ranked[:top_k]]
        top_scores = [round(float(score), 4) for _, score in ranked[:top_k]]

        print(f"[CrossEncoderReranker] Reranked {len(documents)} → {len(top_docs)} docs")
        print(f"  Top scores: {top_scores}")

        # Attach rerank score to metadata for transparency
        for doc, score in zip(top_docs, top_scores):
            doc.metadata["rerank_score"] = score

        return top_docs


# ── Convenience function ──────────────────────────────────────────────────────

_reranker_instance: CrossEncoderReranker | None = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """Return a cached singleton reranker (avoids reloading model on every request)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker(model_name)
    return _reranker_instance
