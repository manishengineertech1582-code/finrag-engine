# FinRAG Engine

**Production-ready Retrieval-Augmented Generation (RAG) system** for answering questions over PDF documents using LLMs, vector search, and modern backend infrastructure.

---

## Overview

FinRAG Engine enables intelligent question answering over unstructured documents by combining:

- Document ingestion & chunking
- Vector embeddings & similarity search
- Retrieval optimization (Hybrid + Reranking)
- LLM-based answer generation
- Evaluation metrics for retrieval quality

---

## Key Features

- PDF ingestion using PyPDFLoader  
- Multiple chunking strategies (fixed, recursive, semantic)  
- OpenAI embeddings  
- FAISS vector database for fast retrieval  
- Dense + BM25 + Hybrid (RRF) retrieval  
- Cross-encoder reranking (MiniLM)  
- LLM-powered answers (GPT-based)  
- Evaluation metrics (Hit@K, MRR)  
- FastAPI-based REST API  
- Dockerized deployment  
- CI/CD pipeline (GitHub Actions)  

---

## Architecture
            ┌──────────────┐
            │   PDF Docs   │
            └──────┬───────┘
                   ↓
           ┌──────────────┐
           │  Chunking    │
           └──────┬───────┘
                   ↓
           ┌──────────────┐
           │ Embeddings   │
           └──────┬───────┘
                   ↓
           ┌──────────────┐
           │   FAISS DB   │
           └──────┬───────┘
                   ↓
           ┌──────────────┐
           │  Retriever   │
           └──────┬───────┘
                   ↓
           ┌──────────────┐
           │  Reranker    │
           └──────┬───────┘
                   ↓
           ┌──────────────┐
           │     LLM      │
           └──────┬───────┘
                   ↓
           ┌──────────────┐
           │ Answer + Src │
           └──────────────┘

---

## Repository Structure
finrag-engine/
├── app/
│ ├── main.py # FastAPI app factory
│ └── routes.py # API endpoints
├── src/
│ ├── ingestion.py # PDF loading
│ ├── chunking.py # Chunking strategies
│ ├── embeddings.py # Embeddings + FAISS
│ ├── retriever.py # Retrieval logic
│ ├── reranker.py # Cross-encoder reranker
│ ├── generator.py # LLM QA chain
│ ├── pipeline.py # End-to-end pipeline
│ └── evaluation.py # Metrics (Hit@K, MRR)
├── tests/ # Unit & integration tests
├── data/ # Input PDFs
├── vector_store/ # FAISS index
├── Dockerfile
├── requirements.txt
├── environment.yml
└── .env


---

## Setup & Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-repo/finrag-engine.git
cd finrag-engine

2️⃣ Create Environment
Option A: pip
pip install -r requirements.txt
Option B: conda
conda env create -f environment.yml
conda activate rag

3️⃣ Set Environment Variables

Create .env file:
OPENAI_API_KEY=your_api_key_here
VECTORSTORE_PATH=vector_store

▶️ Running the Application
Local Development
uvicorn app.main:app --reload
Production (Recommended)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

🐳 Docker Deployment
Build Image
docker build -t finrag-engine .
Run Container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/vector_store:/app/vector_store \
  finrag-engine
🔌 API Endpoints

🔍 Ask Question
POST /api/ask

Request

{
  "question": "What is this document about?"
}

Response

{
  "answer": "This document explains...",
  "sources": [
    {"page": 1, "source": "file.pdf"}
  ]
}
Health Check
GET /health
Evaluation
POST /api/evaluate
Running Tests
pytest tests/

valuation Metrics
Metric	Description
Hit@K	Checks if correct doc is retrieved
MRR	Measures ranking quality

Security Best Practices
Never commit .env files
Use environment variables for secrets
Run container as non-root user
Avoid logging API keys
