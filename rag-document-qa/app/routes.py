# app/routes.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.pipeline import load_pipeline

router = APIRouter()

# Lazy load once
qa_chain = load_pipeline()


class QueryRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(request: QueryRequest):
    result = qa_chain.invoke({"query": request.question})

    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]],
    }