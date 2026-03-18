# src/generator.py

"""
RAG Generator Module

FIX LOG:
- BUG-1:  Removed deprecated get_relevant_documents check.
- BUG-13: Replaced deprecated RetrievalQA with create_retrieval_chain.
- BUG-19: Prompt updated to explicitly handle multi-part questions.
          Previous prompt did not instruct the model to answer EACH
          part of a compound question separately. Added instruction to
          identify and answer each sub-question individually.
"""

from typing import Any, Optional
import logging
import os

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Prompt — handles both single and multi-part questions
# -------------------------------------------------------------------
SYSTEM_PROMPT = """You are FinRAG, an expert document assistant with access \
to multiple technical documents.

Answer the user's question using ONLY the context passages provided below.

IMPORTANT RULES:
1. If the question has MULTIPLE parts (e.g. "What is X? AND What is Y?"), \
answer EACH part separately with a clear heading.
2. Read ALL context passages before answering — relevant information may be \
spread across multiple passages from different documents.
3. Synthesise and combine information across passages when needed.
4. Give a detailed, well-structured answer.
5. You MUST use the context if it contains ANY relevant information.
6. ONLY say "This information is not available in the provided documents" \
if NONE of the passages relate to the question at all.
7. Never refuse to answer when relevant content exists in the context.

Context passages:
{context}"""


PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


def build_qa_chain(
    retriever: Any,
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
):
    """
    Build a retrieval-augmented QA chain (LangChain 0.3+ API).

    Returns a chain whose .invoke({"input": question}) returns:
        {"answer": str, "context": List[Document]}
    """

    if retriever is None:
        raise ValueError("retriever must be a valid object.")

    if not hasattr(retriever, "invoke"):
        raise ValueError("retriever must implement 'invoke'.")

    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    logger.info("Initializing ChatOpenAI | model=%s | temperature=%.2f",
                model_name, temperature)

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    try:
        combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
        logger.info("Retrieval chain created successfully.")
        return qa_chain

    except Exception as e:
        logger.exception("Failed to create retrieval chain.")
        raise RuntimeError("Error building QA chain") from e


def run_query(qa_chain: Any, query: str) -> dict:
    """Execute a query against the QA chain."""

    if qa_chain is None:
        raise ValueError("qa_chain must be initialized.")
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    logger.debug("Executing query: %s", query)

    try:
        response = qa_chain.invoke({"input": query})
        logger.debug("Query executed successfully.")
        return response
    except Exception as e:
        logger.exception("Error during query execution.")
        raise RuntimeError("Failed to execute query") from e
