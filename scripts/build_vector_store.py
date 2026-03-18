# scripts/build_vector_store.py

"""
Vector Store Builder Script

Standalone utility to build or rebuild the FAISS vector store.

FIX LOG:
- BUG-3: `from langchain.docstore.document import Document` — this import
  path was removed in LangChain 0.2+. The correct import is:
  `from langchain_core.documents import Document`

- BUG-4: `from src.embeddings import ... FAISS` — FAISS is not exported
  from src/embeddings.py. It is an internal import there. Importing it
  from embeddings caused an ImportError. Fixed by importing FAISS directly
  from langchain_community.vectorstores where it is actually defined.
"""

import os
import logging

# FIX-3: Correct Document import path for LangChain 0.3+
from langchain_core.documents import Document

# FIX-4: Import FAISS from its actual source, not re-exported from embeddings
from langchain_community.vectorstores import FAISS

from src.embeddings import create_vector_store, get_embeddings, EmbeddingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_or_rebuild_vector_store(
    persist_path: str = "vector_store",
    config: EmbeddingConfig = None,
) -> FAISS:
    """
    Load an existing FAISS vector store, or rebuild from sample documents
    if the store is missing or incompatible.

    Args:
        persist_path: Directory where the FAISS index is stored.
        config: EmbeddingConfig instance (uses defaults if None).

    Returns:
        FAISS vector store instance.
    """
    config = config or EmbeddingConfig()

    if os.path.exists(persist_path):
        try:
            logger.info("Attempting to load existing vector store from '%s'...", persist_path)
            embeddings = get_embeddings(config)
            vectorstore = FAISS.load_local(
                persist_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Vector store loaded successfully.")
            return vectorstore

        except Exception as e:
            logger.warning(
                "Failed to load vector store: %s — rebuilding...", str(e)
            )

    # -----------------------------------------------------------------
    # Rebuild: replace the sample documents below with real content
    # -----------------------------------------------------------------
    logger.info("Building new vector store from documents...")

    sample_docs = [
        Document(page_content="Hello world", metadata={"source": "sample", "page": 0}),
        Document(page_content="This is a test document.", metadata={"source": "sample", "page": 1}),
        # Replace with actual loaded documents in production
    ]

    vectorstore = create_vector_store(sample_docs, config)
    logger.info("Vector store built and saved to '%s'.", persist_path)
    return vectorstore


if __name__ == "__main__":
    store = load_or_rebuild_vector_store()
    logger.info("Done. Vector store ready.")
