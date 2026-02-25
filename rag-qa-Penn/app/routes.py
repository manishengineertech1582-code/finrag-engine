# app/routes.py
"""
FastAPI routes exposing:
  POST /ask        — Q&A with configurable retriever + reranking
  GET  /health     — index and service status
  POST /evaluate   — run Hit@k and MRR over a test set
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# Pipeline is loaded lazily on first request to avoid crash if index missing
_pipelines: dict = {}


def _get_pipeline(retriever_type: str, enable_reranking: bool):
    key = (retriever_type, enable_reranking)
    if key not in _pipelines:
        from src.pipeline import load_pipeline
        _pipelines[key] = load_pipeline(
            retriever_type=retriever_type,
            enable_reranking=enable_reranking,
        )
    return _pipelines[key]


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question to ask")
    retriever_type: str = Field(
        "dense",
        description="Retrieval strategy: 'dense' | 'bm25' | 'hybrid'",
    )
    enable_reranking: bool = Field(
        False,
        description="Apply cross-encoder reranking after first-stage retrieval",
    )

class EvalSample(BaseModel):
    question: str
    ground_truth_source: str  # expected source filename in metadata

class EvalRequest(BaseModel):
    samples: list[EvalSample]
    retriever_type: str = "dense"
    enable_reranking: bool = False
    k: int = 5


# ── POST /ask ─────────────────────────────────────────────────────────────────

@router.post("/ask")
def ask_question(request: QueryRequest):
    """
    Answer a question using the configured RAG pipeline.

    - retriever_type: 'dense' (default), 'bm25', or 'hybrid'
    - enable_reranking: cross-encoder reranking for better precision
    """
    try:
        qa_chain = _get_pipeline(request.retriever_type, request.enable_reranking)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pipeline load failed: {e}")

    try:
        result = qa_chain({"query": request.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return {
        "answer": result["result"],
        "retriever_type": request.retriever_type,
        "reranking_applied": request.enable_reranking,
        "sources": [doc.metadata for doc in result.get("source_documents", [])],
    }


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    """Returns index status and available configuration options."""
    import os
    index_exists = os.path.exists("vector_store/index.faiss")
    return {
        "status": "healthy",
        "index_loaded": index_exists,
        "available_retrievers": ["dense", "bm25", "hybrid"],
        "reranking_available": True,
        "chunking_strategies": ["fixed", "recursive", "semantic"],
    }


# ── POST /evaluate ────────────────────────────────────────────────────────────

@router.post("/evaluate")
def evaluate(request: EvalRequest):
    """
    Run retrieval evaluation over a list of Q&A samples.
    Returns Hit@k and MRR for the configured retriever.
    """
    from src.evaluation import hit_at_k, mean_reciprocal_rank
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            "vector_store", embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not load index: {e}")

    hits, rr_scores = [], []

    for sample in request.samples:
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.k})
        retrieved = retriever.get_relevant_documents(sample.question)

        hit = hit_at_k(retrieved, sample.ground_truth_source)
        rr  = mean_reciprocal_rank(retrieved, sample.ground_truth_source)
        hits.append(hit)
        rr_scores.append(rr)

    return {
        "total_samples": len(request.samples),
        "retriever_type": request.retriever_type,
        "k": request.k,
        "hit_rate": round(sum(hits) / len(hits), 4) if hits else 0,
        "mrr": round(sum(rr_scores) / len(rr_scores), 4) if rr_scores else 0,
    }
