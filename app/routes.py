# app/routes.py

"""
API Routes Module
==================
Purpose:
    Defines the FastAPI HTTP endpoints that the chat UI and external
    clients call to ask questions. Receives a question, passes it
    through the RAG pipeline, and returns the LLM answer along with
    the source document references.

Pipeline Position:
    Browser/Client → [app/routes.py] → src/pipeline.py → FAISS → LLM → Response

Endpoints:
    POST /api/ask
        Request  : {"question": "What is the attention mechanism?"}
        Response : {"answer": "...", "sources": [{"source": "file.pdf", "page": 5}, ...]}

        - Passes the question to the RAG chain
        - Returns the LLM-generated answer
        - Returns metadata of retrieved source chunks (file + page number)
        - Displayed in the chat UI as collapsible "X sources retrieved"

Pipeline Singleton Pattern:
    The QA pipeline (FAISS + retriever + LLM chain) is expensive to
    initialise — it loads the vector store and connects to OpenAI.
    get_pipeline() builds it once on the first request and reuses the
    same instance for all subsequent requests via a module-level
    _qa_pipeline variable. This avoids reloading the vector store on
    every API call.

Error Handling:
    503 — Vector store not found (run python create_index.py first)
    503 — Pipeline failed to initialise (check server logs)
    400 — Empty question submitted
    500 — Unexpected error during query execution

Usage:
    This module is registered in app/main.py via:
        app.include_router(router)

    The endpoint is then available at:
        POST http://127.0.0.1:8000/api/ask

    Test via Swagger UI:
        http://127.0.0.1:8000/docs

Called by:
    static/index.html  — chat UI sends POST /api/ask on every question
    app/main.py        — registers this router at startup

FIX LOG:
    BUG-7:  Unhandled RuntimeError on pipeline load now returns HTTP 503
            with a helpful message instead of crashing the server.
    BUG-8:  Replaced complex hasattr singleton check with a clean
            module-level _qa_pipeline = None pattern.
    BUG-14: Updated invoke key from {"query": ...} to {"input": ...}
            and response keys from "result"/"source_documents" to
            "answer"/"context" to match the new create_retrieval_chain
            API in LangChain 0.3+.
"""

from typing import Any, List
import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.pipeline import load_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG"])


# -------------------------------------------------------------------
# Request / Response Models
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User query string")


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


# -------------------------------------------------------------------
# Pipeline Singleton
# -------------------------------------------------------------------
_qa_pipeline = None


def get_pipeline():
    global _qa_pipeline

    if _qa_pipeline is None:
        logger.info("Initialising QA pipeline (first request)...")
        try:
            _qa_pipeline = load_pipeline()
            logger.info("QA pipeline initialised successfully.")
        except FileNotFoundError as e:
            logger.error("Vector store not found: %s", str(e))
            raise HTTPException(
                status_code=503,
                detail=(
                    "Vector store not found. "
                    "Run `python create_index.py` first to build the index."
                ),
            ) from e
        except Exception as e:
            logger.exception("Failed to initialise pipeline.")
            raise HTTPException(
                status_code=503,
                detail="Pipeline initialisation failed. Check server logs.",
            ) from e

    return _qa_pipeline


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    qa_chain: Any = Depends(get_pipeline),
):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info("Received query: %s", question[:120])

    try:
        # BUG-14 FIX:
        # New create_retrieval_chain API uses {"input": ...} not {"query": ...}
        # and returns {"answer": ..., "context": [...]} not {"result": ..., "source_documents": [...]}
        result = qa_chain.invoke({"input": question})

        answer       = result.get("answer", "")
        source_docs  = result.get("context", [])   # "context" in new API

        sources = [
            doc.metadata if hasattr(doc, "metadata") else {}
            for doc in source_docs
        ]

        logger.info("Query processed. Sources returned: %d", len(sources))

        return QueryResponse(answer=answer, sources=sources)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Error processing query.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing query.",
        ) from e