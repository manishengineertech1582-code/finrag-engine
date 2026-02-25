# src/evaluation.py

def hit_at_k(retrieved_docs, ground_truth_doc_id):
    retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs]
    return int(ground_truth_doc_id in retrieved_ids)


def mean_reciprocal_rank(retrieved_docs, ground_truth_doc_id):
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc.metadata.get("id") == ground_truth_doc_id:
            return 1 / rank
    return 0