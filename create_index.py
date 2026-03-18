# create_index.py

"""
Indexing Pipeline — builds the FAISS vector store from PDFs.

FIX LOG:
- BUG-9: No `load_dotenv()` call. Without it, OPENAI_API_KEY is never
  loaded from .env when running `python create_index.py` directly, causing
  an authentication error on the first OpenAI API call.

  FIX: Added `load_dotenv()` at the top of the script, before any
  module that needs the API key is imported or called.
"""

import os
import logging
from typing import List

# FIX-9: Load .env FIRST so OPENAI_API_KEY is available
from dotenv import load_dotenv
load_dotenv()

from src.ingestion import load_pdf
from src.chunking import fixed_chunking
from src.embeddings import create_vector_store


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_FOLDER = "data/"
BATCH_SIZE = 5   # PDFs per batch — prevents memory overflow on large corpora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_pdf_files(folder_path: str) -> List[str]:
    """Return all .pdf files found in the given directory."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Data folder not found: {folder_path}")

    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]


def process_batch(pdf_batch: List[str]) -> List:
    """Load a batch of PDFs and attach source metadata to each document."""
    batch_documents = []

    for pdf in pdf_batch:
        try:
            logging.info("Loading PDF: %s", pdf)
            docs = load_pdf(pdf)

            for doc in docs:
                if hasattr(doc, "metadata"):
                    doc.metadata["source"] = pdf   # ensure source is always set

            batch_documents.extend(docs)

        except Exception as e:
            logging.error("Skipping %s — error: %s", pdf, e)

    return batch_documents


def deduplicate_documents(documents: List) -> List:
    """Remove duplicate documents by page_content hash."""
    seen = set()
    unique = []

    for doc in documents:
        content = getattr(doc, "page_content", str(doc))
        if content not in seen:
            seen.add(content)
            unique.append(doc)

    return unique


# -------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------
def main():
    """
    Run the full indexing pipeline:
      1. Discover PDFs
      2. Batch ingest
      3. Deduplicate
      4. Chunk
      5. Embed + index into FAISS
    """
    try:
        logging.info("Starting indexing pipeline...")

        pdf_files = get_pdf_files(DATA_FOLDER)

        if not pdf_files:
            logging.warning("No PDF files found in '%s'. Exiting.", DATA_FOLDER)
            return

        logging.info("Found %d PDF file(s).", len(pdf_files))

        all_documents = []

        for i in range(0, len(pdf_files), BATCH_SIZE):
            batch = pdf_files[i : i + BATCH_SIZE]
            logging.info("Processing batch %d / %d", i // BATCH_SIZE + 1,
                         (len(pdf_files) + BATCH_SIZE - 1) // BATCH_SIZE)
            all_documents.extend(process_batch(batch))

        logging.info("Deduplicating %d raw documents...", len(all_documents))
        all_documents = deduplicate_documents(all_documents)
        logging.info("Documents after dedup: %d", len(all_documents))

        logging.info("Chunking documents...")
        chunks = fixed_chunking(all_documents)
        logging.info("Total chunks created: %d", len(chunks))

        logging.info("Creating FAISS vector store...")
        create_vector_store(chunks)

        logging.info("Indexing pipeline completed successfully.")

    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
