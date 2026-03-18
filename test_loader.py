# test_loader.py

"""
Test Module: PDF Loader

FIX LOG:
- BUG-5: TEST_PDF_PATH defaulted to "data/Manishfile.pdf" which does NOT
  exist in the repository. The actual PDF present is "data/6ghz-details.pdf".
  Fixed by updating the default path to the correct filename.
  The env-var override (TEST_PDF_PATH) is preserved so CI/CD can inject
  any path without changing code.
"""

import os
import logging
import pytest

from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

# FIX-5: Updated default from "data/Manishfile.pdf" → "data/6ghz-details.pdf"
TEST_PDF_PATH = os.getenv("TEST_PDF_PATH", "data/6ghz-details.pdf")


def load_pdf(file_path: str):
    """
    Helper function to load PDF documents for testing.

    Args:
        file_path: Path to PDF file.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test PDF not found: {file_path}")

    loader = PyPDFLoader(file_path)
    return loader.load()


def test_pdf_loader_success():
    """Test successful PDF loading."""
    docs = load_pdf(TEST_PDF_PATH)

    assert docs is not None, "Loaded documents should not be None"
    assert isinstance(docs, list), "Output should be a list"
    assert len(docs) > 0, "PDF should contain at least one document"

    logger.info("PDF loaded successfully with %d documents", len(docs))


def test_pdf_loader_file_not_found():
    """Test that FileNotFoundError is raised for a non-existent file."""
    invalid_path = "data/non_existent_file_xyz.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(invalid_path)


def test_pdf_document_structure():
    """Validate structure of returned LangChain Document objects."""
    docs = load_pdf(TEST_PDF_PATH)

    first_doc = docs[0]

    assert hasattr(first_doc, "page_content"), "Document missing page_content"
    assert hasattr(first_doc, "metadata"), "Document missing metadata"
    assert isinstance(first_doc.metadata, dict), "metadata must be a dict"

    logger.info("Document structure validated successfully.")


def test_pdf_metadata_has_page_key():
    """
    Ensure PyPDFLoader sets the 'page' metadata key.
    This is critical for evaluation.py Hit@K and MRR metrics.
    """
    docs = load_pdf(TEST_PDF_PATH)
    first_doc = docs[0]

    assert "page" in first_doc.metadata, (
        "PyPDFLoader must set 'page' in metadata. "
        "evaluation.py relies on this key for Hit@K / MRR."
    )
    logger.info("Metadata 'page' key confirmed: %s", first_doc.metadata)
