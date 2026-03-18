# src/pipeline.py
"""
RAG Pipeline Loader


"""

from typing import Optional
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.retriever import get_retriever
from src.generator import build_qa_chain

logger = logging.getLogger(__name__)

# Must match EmbeddingConfig.model in embeddings.py
EMBEDDING_MODEL = "text-embedding-3-small"


def load_pipeline(
    vectorstore_path: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
):
    """
    Initialise and return a fully configured RAG QA pipeline.

    Args:
        vectorstore_path: Path to the persisted FAISS index.
        model: OpenAI chat model name (defaults to gpt-4o-mini).
        temperature: LLM temperature (0 = deterministic).

    Returns:
        Configured retrieval chain ready for inference.
    """

    vs_path = vectorstore_path or os.getenv("VECTORSTORE_PATH", "vector_store")

    if not os.path.exists(vs_path):
        logger.error("Vector store not found at: %s", vs_path)
        raise FileNotFoundError(
            f"Vector store not found at: {vs_path}. "
            "Run `python create_index.py` to build it first."
        )

    logger.info("Loading RAG pipeline | vectorstore_path=%s", vs_path)

    # --- Embeddings ---
    # BUG-16 FIX: must use SAME model that was used to build the index
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        logger.info("OpenAI embeddings initialised (model=%s).", EMBEDDING_MODEL)
    except Exception as e:
        logger.exception("Failed to initialise embeddings.")
        raise RuntimeError("Embedding initialisation failed") from e

    # --- Vector Store ---
    try:
        vectorstore = FAISS.load_local(
            vs_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS vector store loaded.")
    except Exception as e:
        logger.exception("Failed to load FAISS vector store.")
        raise RuntimeError("Vector store loading failed") from e

    # --- Retriever ---
    try:
        retriever = get_retriever(vectorstore)
        logger.info("Retriever initialised.")
    except Exception as e:
        logger.exception("Failed to initialise retriever.")
        raise RuntimeError("Retriever initialisation failed") from e

    # --- QA Chain ---
    try:
        qa_chain = build_qa_chain(
            retriever=retriever,
            model=model,
            temperature=temperature,
        )
        logger.info("QA chain built successfully.")
    except Exception as e:
        logger.exception("Failed to build QA chain.")
        raise RuntimeError("QA chain creation failed") from e

    return qa_chain
