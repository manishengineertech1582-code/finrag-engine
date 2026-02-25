# src/evaluation.py
"""
Retrieval quality metrics:
  - hit_at_k          : binary — is the correct doc in the top-k results?
  - mean_reciprocal_rank : MRR — where does the correct doc rank?
  - evaluate_retriever  : run both metrics over a dataset, print comparison table
"""

from langchain_core.documents import Document


# ── Core metrics ──────────────────────────────────────────────────────────────

def hit_at_k(retrieved_docs: list[Document], ground_truth_source: str) -> int:
    """
    Returns 1 if any retrieved document matches the ground-truth source, else 0.

    Args:
        retrieved_docs:       List of retrieved LangChain Documents.
        ground_truth_source:  Expected value of doc.metadata['source'].
    """
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        if ground_truth_source in source or source in ground_truth_source:
            return 1
    return 0


def mean_reciprocal_rank(retrieved_docs: list[Document], ground_truth_source: str) -> float:
    """
    Returns 1/rank of the first document matching ground_truth_source, or 0.

    Args:
        retrieved_docs:       List of retrieved LangChain Documents (ordered).
        ground_truth_source:  Expected value of doc.metadata['source'].
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "")
        if ground_truth_source in source or source in ground_truth_source:
            return 1.0 / rank
    return 0.0


# ── Full evaluation harness ───────────────────────────────────────────────────

def evaluate_retriever(retriever, eval_dataset: list[dict], k: int = 5) -> dict:
    """
    Run Hit@k and MRR over an evaluation dataset.

    Args:
        retriever:    Any retriever with .get_relevant_documents(query) method.
        eval_dataset: List of dicts: [{"question": str, "source": str}, ...]
        k:            Top-k cutoff (informational; retriever must be pre-configured).

    Returns:
        {"hit_rate": float, "mrr": float, "total": int, "hits": int}
    """
    hits, rr_scores = [], []

    for sample in eval_dataset:
        question = sample["question"]
        ground_truth = sample["source"]

        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        else:
            docs = retriever(question)

        hits.append(hit_at_k(docs, ground_truth))
        rr_scores.append(mean_reciprocal_rank(docs, ground_truth))

    total = len(eval_dataset)
    hit_rate = sum(hits) / total if total else 0.0
    mrr = sum(rr_scores) / total if total else 0.0

    return {
        "total": total,
        "hits": sum(hits),
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
    }


def compare_retrievers(retrievers: dict, eval_dataset: list[dict], k: int = 5):
    """
    Evaluate multiple retrievers and print a comparison table.

    Args:
        retrievers:   {"label": retriever_obj, ...}
        eval_dataset: [{"question": str, "source": str}, ...]
        k:            Top-k cutoff label for display.

    Returns:
        dict of {label: metrics_dict}
    """
    print(f"\n{'='*60}")
    print(f"  Retrieval Evaluation  (k={k}, n={len(eval_dataset)} samples)")
    print(f"{'='*60}")
    print(f"  {'Retriever':<20} {'Hit Rate':>10} {'MRR':>10} {'Hits':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")

    all_results = {}
    for label, retriever in retrievers.items():
        result = evaluate_retriever(retriever, eval_dataset, k)
        all_results[label] = result
        print(f"  {label:<20} {result['hit_rate']:>10.4f} {result['mrr']:>10.4f} "
              f"{result['hits']:>5}/{result['total']}")

    print(f"{'='*60}\n")
    return all_results
