# src/chunking.py

"""
This module is responsible for splitting large documents into smaller chunks
optimized for:
- LLM context windows
- Embedding quality
- Retrieval accuracy (RAG systems)

Key Responsibilities:
1. Split documents into manageable chunks
2. Preserve metadata for traceability
3. Optimize chunk size vs overlap trade-off
"""

import logging
from typing import List

# Correct LangChain import (new package structure)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------
# CONFIGURATION (Tunable Parameters)
# -------------------------------
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


# -------------------------------
# LOGGING SETUP
# -------------------------------
logger = logging.getLogger(__name__)


def validate_documents(documents: List) -> None:
    """
    Validate input documents before processing.

    Raises:
        ValueError: If documents are empty or invalid
    """
    if not documents:
        raise ValueError("No documents provided for chunking.")

    if not isinstance(documents, list):
        raise TypeError("Documents must be a list.")

    # Optional: deeper validation (can be extended)
    for doc in documents:
        if not hasattr(doc, "page_content"):
            raise ValueError("Invalid document format: missing 'page_content'.")


def create_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """
    Factory function to create a text splitter.

    Why RecursiveCharacterTextSplitter?
    - Preserves semantic boundaries better than naive splitting
    - Splits by paragraphs → sentences → words

    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks (improves context retention)

    Returns:
        Configured text splitter instance
    """

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def fixed_chunking(
    documents: List,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List:
    """
    Main chunking function used in the pipeline.

    Pipeline Step:
        Documents → Chunked Documents

    Why chunking is critical:
    - LLMs have token limits
    - Embeddings work best on focused text
    - Improves retrieval precision in RAG systems

    Args:
        documents: List of document objects (LangChain format)
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """

    try:
        logger.info("Starting document chunking...")

        # -------------------------------
        # VALIDATION
        # -------------------------------
        validate_documents(documents)

        logger.info(f"Input documents: {len(documents)}")

        # -------------------------------
        # SPLITTER CREATION
        # -------------------------------
        splitter = create_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # -------------------------------
        # CHUNKING EXECUTION
        # -------------------------------
        chunks = splitter.split_documents(documents)

        logger.info(f"Generated {len(chunks)} chunks")

        # -------------------------------
        # METADATA ENHANCEMENT (OPTIONAL)
        # -------------------------------
        # Add chunk index for traceability/debugging
        for idx, chunk in enumerate(chunks):
            if hasattr(chunk, "metadata"):
                chunk.metadata["chunk_id"] = idx

        return chunks

    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        raise