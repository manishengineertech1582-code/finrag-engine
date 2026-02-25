# app/main.py

from fastapi import FastAPI
from src.pipeline import load_pipeline

app = FastAPI()

qa_chain = load_pipeline()

@app.post("/ask")
def ask_question(question: str):
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }