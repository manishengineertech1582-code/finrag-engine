# src/ingestion.py

"""
Document Ingestion Module
==========================
Purpose:
    Loads PDF files from disk and converts them into LangChain Document
    objects ready for chunking and embedding. This is the first step in
    the RAG pipeline — raw PDF files go in, structured Document objects
    come out.

Pipeline Position:
    PDF files → [src/ingestion.py] → src/chunking.py → src/embeddings.py

How It Works:
    Uses LangChain's PyPDFLoader which reads each page of a PDF and
    creates one Document object per page. Each Document contains:
        - page_content : extracted text from that page
        - metadata     : {"source": "path/to/file.pdf", "page": 0}

    The "source" and "page" metadata fields are used downstream by:
        - src/evaluation.py  — for retrieval quality scoring
        - app/routes.py      — to show users which pages were retrieved
        - static/index.html  — to display source citations in the chat UI

Supported Sources:
    - PDF files via PyPDFLoader (current)
    - Extensible for DOCX, HTML, web APIs (future)

Usage:
    from src.ingestion import load_pdf, load_multiple_pdfs

    # Load a single PDF
    docs = load_pdf("data/Transformer-attention-is-all-you-need-Paper.pdf")

    # Load all PDFs at once
    docs = load_multiple_pdfs([
        "data/6ghz-details.pdf",
        "data/Hands-On-LLM.pdf",
        "data/Fundamentals of Deep Learning.pdf",
        "data/Transformer-attention-is-all-you-need-Paper.pdf",
    ])

Called by:
    create_index.py  — loads all PDFs before chunking and indexing

Key Functions:
    load_pdf()           — loads a single PDF, returns List[Document]
    load_multiple_pdfs() — loads many PDFs, skips failed files gracefully

Error Handling:
    - Invalid path      → ValueError
    - File not found    → FileNotFoundError
    - Corrupt/unreadable PDF → RuntimeError (file is skipped in batch mode)
    - Non-PDF extension → logged as warning but still attempted
"""

from typing import List
import logging
import os

from langchain_community.document_loaders import PyPDFLoader

# Configure module-level logger
logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List:
    """
    Load and parse a PDF file into LangChain Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects extracted from the PDF.

    Raises:
        ValueError: If file_path is invalid.
        FileNotFoundError: If file does not exist.
        RuntimeError: If loading fails.
    """

    # ---------------------------
    # Input Validation
    # ---------------------------
    if not file_path or not isinstance(file_path, str):
        logger.error("Invalid file_path provided: %s", file_path)
        raise ValueError("file_path must be a valid non-empty string.")

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        logger.warning("File does not have a .pdf extension: %s", file_path)

    logger.info("Loading PDF file: %s", file_path)

    # ---------------------------
    # Load PDF
    # ---------------------------
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            logger.warning("No content extracted from PDF: %s", file_path)

        logger.info(
            "Successfully loaded PDF | pages=%d | file=%s",
            len(documents),
            file_path,
        )

        return documents

    except Exception as e:
        logger.exception("Failed to load PDF: %s", file_path)
        raise RuntimeError(f"Error loading PDF: {file_path}") from e


def load_multiple_pdfs(file_paths: List[str]) -> List:
    """
    Load multiple PDF files and combine their documents.

    Args:
        file_paths: List of PDF file paths.

    Returns:
        Combined list of Document objects.

    Raises:
        ValueError: If input list is invalid.
    """

    if not file_paths or not isinstance(file_paths, list):
        logger.error("Invalid file_paths list provided.")
        raise ValueError("file_paths must be a non-empty list.")

    all_documents = []

    for path in file_paths:
        try:
            docs = load_pdf(path)
            all_documents.extend(docs)
        except Exception as e:
            logger.warning("Skipping file due to error: %s | Error: %s", path, str(e))

    logger.info("Total documents loaded from all PDFs: %d", len(all_documents))

    return all_documents