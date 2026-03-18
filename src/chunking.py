# src/chunking.py

"""
Document Chunking Module
=========================
Purpose:
    Splits large PDF pages (loaded by src/ingestion.py) into smaller,
    overlapping text chunks optimised for embedding quality and retrieval
    accuracy in the RAG pipeline. This is the second step in the
    ingestion pipeline, sitting between document loading and embedding.

Pipeline Position:
    PDF files → src/ingestion.py → [src/chunking.py] → src/embeddings.py

Why Chunking is Necessary:
    - LLMs have a limited context window (token limit)
    - Embedding models produce better vectors for focused, short text
      than for entire PDF pages
    - Smaller chunks improve retrieval precision — FAISS can pinpoint
      the exact passage that answers the question rather than returning
      an entire page

Chunking Strategy:
    Uses RecursiveCharacterTextSplitter which splits text in this order:
        Paragraphs → Sentences → Words → Characters
    This preserves semantic boundaries better than naive fixed-size splits.

Default Settings:
    CHUNK_SIZE    = 800 characters  — balances context and focus
    CHUNK_OVERLAP = 150 characters  — ensures sentences split across
                                      chunk boundaries are not lost

Metadata Preserved:
    Each chunk retains the original document's metadata (source file,
    page number) and adds a unique chunk_id for traceability.
    chunk_id is used by src/evaluation.py for retrieval quality testing.

Usage:
    from src.chunking import fixed_chunking

    # Called by create_index.py after loading PDFs
    chunks = fixed_chunking(documents)

    # Optional: custom chunk size and overlap
    chunks = fixed_chunking(documents, chunk_size=500, chunk_overlap=100)

Called by:
    create_index.py  — as part of the full PDF → vector store pipeline

Key Functions:
    validate_documents() — checks documents are non-empty and valid
    create_splitter()    — factory that builds the text splitter
    fixed_chunking()     — main entry point, returns List[Document]
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