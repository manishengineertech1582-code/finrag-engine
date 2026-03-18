# app/routes.py

"""
API Routes Module

FIX LOG:
- BUG-7: Unhandled RuntimeError on pipeline load -> now returns HTTP 503.
- BUG-8: Cleaner module-level singleton pattern.
- BUG-14: create_retrieval_chain (new API) returns {"answer": ..., "context": [...]}
  instead of the old RetrievalQA {"result": ..., "source_documents": [...]}.
  Updated result extraction keys to match the new response schema.
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
