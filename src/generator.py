# src/generator.py

"""
RAG Generator Module
=====================
Purpose:
    Builds the LLM-powered question-answering chain that takes retrieved
    document chunks and generates a coherent, grounded answer using
    OpenAI's gpt-4o-mini model. This is the final "generation" step
    in the Retrieve → Generate RAG pipeline.

How It Works:
    1. Retrieved chunks from FAISS are injected into the system prompt
       as {context} passages.
    2. The user's question is passed as {input}.
    3. gpt-4o-mini reads the context and generates a grounded answer.
    4. If no relevant context exists, the model says so explicitly.

Key Components:
    - SYSTEM_PROMPT   : Instructs the LLM to answer strictly from context,
                        handle multi-part questions with separate headings,
                        and never hallucinate outside the provided passages.
    - PROMPT          : ChatPromptTemplate wiring system + human messages.
    - build_qa_chain(): Assembles the full retrieval chain using the
                        LangChain 0.3+ API (create_retrieval_chain +
                        create_stuff_documents_chain).
    - run_query()     : Executes a query against a built chain.

Usage:
    from src.generator import build_qa_chain, run_query

    # Build the chain (called once at startup via pipeline.py)
    qa_chain = build_qa_chain(retriever=retriever, model="gpt-4o-mini")

    # Run a query
    result = run_query(qa_chain, "What is the attention mechanism?")
    print(result["answer"])   # LLM answer
    print(result["context"])  # List of retrieved Document objects

Called by:
    src/pipeline.py   — at startup to assemble the full RAG pipeline
    app/routes.py     — indirectly via the pipeline singleton

Environment Variables:
    OPENAI_API_KEY    — required for ChatOpenAI authentication
    OPENAI_MODEL      — optional override (defaults to gpt-4o-mini)

FIX LOG:
    BUG-13: Replaced deprecated RetrievalQA with create_retrieval_chain
            (LangChain 0.3+ standard). Old chain silently dropped context
            causing "I don't know" even when relevant chunks were retrieved.
    BUG-19: Prompt updated to handle multi-part questions. Added rule to
            answer EACH part separately with a clear heading, preventing
            the LLM from answering only the first part of compound queries.
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