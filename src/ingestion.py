# src/ingestion.py

"""
Document Ingestion Module

This module handles loading documents from various sources.
Currently supports:
- PDF ingestion via PyPDFLoader

Designed to be extensible for additional sources (e.g., DOCX, HTML, APIs).
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