# src/evaluation.py

"""
Evaluation metrics for retrieval systems.

This module provides standard IR (Information Retrieval) metrics:
- Hit@K
- Mean Reciprocal Rank (MRR)

FIX LOG:
- BUG-2: `_extract_doc_id` looked up `doc.metadata.get("id")`.
  PyPDFLoader NEVER sets an "id" key — it only sets "source" (file path)
  and "page" (0-based page number). This meant Hit@K and MRR always
  returned 0, making the entire evaluation module silently broken.

  FIX: Changed the primary key to "page" (what PyPDFLoader actually sets),
  with a fallback to "source" and then the old "id" key for backward
  compatibility with custom loaders that do set "id".

  The chunking module already sets "chunk_id" on each chunk. Evaluation
  functions now also accept "chunk_id" as a valid identifier, so tests
  using chunk_id-based ground truth work correctly.
"""

from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Document:
    """
    Minimal Document interface for type-checking in tests.
    In production use LangChain Document objects.
    """
    def __init__(self, metadata: Optional[dict] = None):
        self.metadata = metadata or {}


def _extract_doc_id(doc: Any) -> Optional[Any]:
    """
    Safely extract a document identifier from metadata.

    Priority order (most specific → least specific):
      1. "id"       — explicitly set by custom loaders
      2. "chunk_id" — set by src/chunking.py fixed_chunking()
      3. "page"     — set by PyPDFLoader (0-based page index)
      4. "source"   — file path, set by PyPDFLoader

    Args:
        doc: Document-like object with a `metadata` attribute.

    Returns:
        Document identifier if present, else None.
    """
    try:
        meta = doc.metadata

        # Try each key in priority order
        for key in ("id", "chunk_id", "page", "source"):
            value = meta.get(key)
            if value is not None:
                return value

        return None

    except AttributeError:
        logger.warning("Document missing metadata attribute: %s", doc)
        return None


def hit_at_k(
    retrieved_docs: List[Any],
    ground_truth_doc_id: Any,
) -> int:
    """
    Compute Hit@K metric.

    Hit@K = 1 if the ground truth document appears in the top-K retrieved
    results, else 0.

    Args:
        retrieved_docs: List of retrieved document objects (LangChain format).
        ground_truth_doc_id: Expected relevant document identifier.
                             Should match the value of "page", "chunk_id",
                             or "id" in document metadata.

    Returns:
        1 if hit found, otherwise 0.
    """
    if not retrieved_docs:
        logger.debug("Empty retrieved_docs list provided to hit_at_k.")
        return 0

    if ground_truth_doc_id is None:
        logger.error("None ground_truth_doc_id provided to hit_at_k.")
        return 0

    retrieved_ids = [
        _extract_doc_id(doc)
        for doc in retrieved_docs
    ]
    # Filter out None values for clean comparison
    retrieved_ids = [rid for rid in retrieved_ids if rid is not None]

    hit = int(ground_truth_doc_id in retrieved_ids)

    logger.debug(
        "Hit@K=%d | ground_truth=%s | retrieved_ids=%s",
        hit,
        ground_truth_doc_id,
        retrieved_ids,
    )

    return hit


def mean_reciprocal_rank(
    retrieved_docs: List[Any],
    ground_truth_doc_id: Any,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for a single query.

    MRR = 1 / rank_of_first_relevant_document
    If no relevant document is found, returns 0.0.

    Args:
        retrieved_docs: Ranked list of retrieved document objects.
        ground_truth_doc_id: Expected relevant document identifier.

    Returns:
        Reciprocal rank score (float between 0.0 and 1.0).
    """
    if not retrieved_docs:
        logger.debug("Empty retrieved_docs list provided to MRR.")
        return 0.0

    if ground_truth_doc_id is None:
        logger.error("None ground_truth_doc_id provided to MRR.")
        return 0.0

    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_id = _extract_doc_id(doc)

        if doc_id == ground_truth_doc_id:
            score = 1.0 / rank
            logger.debug(
                "MRR hit at rank=%d | score=%.4f | ground_truth=%s",
                rank,
                score,
                ground_truth_doc_id,
            )
            return score

    logger.debug("MRR no hit | ground_truth=%s", ground_truth_doc_id)
    return 0.0
