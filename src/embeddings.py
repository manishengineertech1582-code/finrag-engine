# src/embeddings.py
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class EmbeddingConfig:
    def __init__(self, model="text-embedding-3-small", persist_path="vector_store"):
        self.model = model
        self.persist_path = persist_path


def validate_documents(docs):
    if not docs:
        raise ValueError("No documents provided.")
    for doc in docs:
        if not hasattr(doc, "page_content"):
            raise ValueError("Each document must have page_content.")


def get_embeddings(config):
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set")
    return OpenAIEmbeddings(model=config.model)


def create_vector_store(docs, config=None):
    logger.info("Starting embedding + indexing process...")
    config = config or EmbeddingConfig()
    validate_documents(docs)
    logger.info("Embedding %d documents...", len(docs))
    embeddings = get_embeddings(config)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(config.persist_path)
    logger.info("Vector store created. %d documents indexed.", len(docs))
    return vectorstore


def load_vector_store(persist_path="vector_store", config=None):
    config = config or EmbeddingConfig()
    if not os.path.exists(persist_path):
        raise FileNotFoundError(f"Vector store not found at {persist_path}")
    embeddings = get_embeddings(config)
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
