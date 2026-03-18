# src/retriever.py

"""
Retriever Module

FIX LOG:
- BUG-17: DEFAULT_TOP_K was 5. For compound questions spanning multiple
  documents or topics (e.g. "What is X AND what is Y?"), 5 chunks are
  often all retrieved from the dominant topic, leaving the second topic
  with zero coverage. Increased to 8 for better multi-topic coverage.

- BUG-18: Single-query retrieval fails for compound questions. Added
  MultiQueryRetriever support which uses the LLM to decompose a complex
  question into sub-queries, retrieves for each separately, then merges
  and deduplicates the results. This means a question about both 6GHz
  spectrum AND deep learning will retrieve relevant chunks from BOTH
  topics instead of just the dominant one.
"""

from typing import Any, Dict, Optional, Protocol
import logging
import os

logger = logging.getLogger(__name__)


class VectorStoreProtocol(Protocol):
    def as_retriever(self, **kwargs: Any) -> Any:
        ...


DEFAULT_SEARCH_TYPE = "similarity"
DEFAULT_TOP_K       = 8          # BUG-17 FIX: raised from 5 to 8

SUPPORTED_SEARCH_TYPES = {
    "similarity",
    "mmr",
    "similarity_score_threshold",
}


def get_retriever(
    vectorstore: VectorStoreProtocol,
    search_type: str = DEFAULT_SEARCH_TYPE,
    search_kwargs: Optional[Dict[str, Any]] = None,
    use_multi_query: bool = True,       # BUG-18 FIX: multi-query by default
) -> Any:
    """
    Factory function to create a retriever from a vector store.

    When use_multi_query=True (default), wraps the base retriever with
    MultiQueryRetriever. This uses the LLM to generate multiple
    sub-queries from the user's question, retrieves chunks for each,
    then merges and deduplicates. Dramatically improves recall for
    compound or multi-topic questions.

    Args:
        vectorstore:      Vector store (FAISS, Chroma, etc.)
        search_type:      "similarity", "mmr", or "similarity_score_threshold"
        search_kwargs:    {"k": int, ...}
        use_multi_query:  If True, wrap with MultiQueryRetriever

    Returns:
        Retriever compatible with LangChain chains.
    """

    if vectorstore is None:
        raise ValueError("vectorstore must be a valid object.")

    if not hasattr(vectorstore, "as_retriever"):
        raise ValueError("vectorstore must implement 'as_retriever'.")

    if search_type not in SUPPORTED_SEARCH_TYPES:
        logger.warning("Unsupported search_type '%s', falling back to 'similarity'.", search_type)
        search_type = DEFAULT_SEARCH_TYPE

    if search_kwargs is None:
        search_kwargs = {"k": DEFAULT_TOP_K}
    else:
        search_kwargs = dict(search_kwargs)

    k = search_kwargs.get("k", DEFAULT_TOP_K)
    if not isinstance(k, int) or k <= 0:
        raise ValueError("'k' must be a positive integer.")
    search_kwargs["k"] = k

    logger.info("Creating retriever | strategy=%s | k=%d | multi_query=%s",
                search_type, k, use_multi_query)

    # Base FAISS retriever
    base_retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    # BUG-18 FIX: wrap with MultiQueryRetriever for compound questions
    if use_multi_query:
        try:
            from langchain.retrievers.multi_query import MultiQueryRetriever
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
            )

            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )
            logger.info("MultiQueryRetriever created successfully.")
            return retriever

        except Exception as e:
            logger.warning("MultiQueryRetriever failed (%s), falling back to base.", e)
            return base_retriever

    return base_retriever
