# FinRAG Engine

Production-ready Retrieval-Augmented Generation (RAG) system for answering questions over PDF documents.

## Features

- Fixed and semantic chunking
- FAISS vector database
- OpenAI embeddings
- FastAPI deployment
- Docker containerization
- Retrieval evaluation (Hit@K, MRR)

## Architecture

Document → Chunking → Embeddings → FAISS → Retriever → LLM → Answer + Sources

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload