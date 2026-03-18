# app/main.py

"""
Main Application Entry Point
==============================
Purpose:
    Creates and configures the FastAPI application that powers the
    FinRAG Engine. Wires together the API routes, the chat UI, logging,
    and environment variable loading into a single deployable ASGI app.

How to Run:
    # Development (with auto-reload)
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

    # Production
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

Endpoints Registered:
    POST /api/ask     — RAG question answering (via app/routes.py)
    GET  /health      — Health check for load balancers / monitoring
    GET  /            — Chat UI (served from static/index.html)
    GET  /docs        — Swagger UI for API testing (FastAPI built-in)

Request Flow:
    Browser → GET  /          → static/index.html  (chat UI)
    Browser → POST /api/ask   → app/routes.py      (RAG pipeline)
                              → src/pipeline.py    (FAISS + LLM)
                              → OpenAI API         (answer)
                              → browser            (displayed in UI)

Static File Serving:
    The chat UI (static/index.html) is mounted at "/" using FastAPI's
    StaticFiles. It MUST be mounted LAST after all API routes — if
    mounted first, it intercepts all requests including /api/ask.

Application Factory Pattern:
    create_app() builds and returns the FastAPI instance. This pattern
    allows the app to be imported cleanly in tests without side effects,
    and makes it easy to create multiple configured instances.

Lifespan Context Manager:
    Handles startup and shutdown events using the modern asynccontextmanager
    pattern (replaces deprecated @app.on_event("startup") decorators).
    Currently logs startup/shutdown — extend here to pre-load the pipeline
    at startup rather than on first request.

Environment Variables Required:
    OPENAI_API_KEY   — loaded from .env via load_dotenv() at module top
    OPENAI_MODEL     — optional, defaults to gpt-4o-mini (in generator.py)
    VECTORSTORE_PATH — optional, defaults to "vector_store" (in pipeline.py)

FIX LOG:
    BUG-6: load_dotenv() was missing. Without it, OPENAI_API_KEY was never
           read from .env when running locally, causing authentication errors
           on every pipeline call. Fixed by calling load_dotenv() at the top
           of this file before any module that needs the API key is imported.
           Also added logging.basicConfig() so all src.* loggers produce
           visible timestamped output in the terminal.
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import router

# FIX-6a: Load .env BEFORE any module that needs OPENAI_API_KEY
load_dotenv()

# FIX-6b: Configure root logger so all src.* loggers produce visible output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Application Lifecycle
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.
    Modern replacement for deprecated @app.on_event decorators.
    """
    logger.info("Starting FinRAG Engine...")
    try:
        yield
    finally:
        logger.info("Shutting down FinRAG Engine...")


# -------------------------------------------------------------------
# App Factory
# -------------------------------------------------------------------
def create_app() -> FastAPI:
    """
    Application factory pattern — preferred for production and testing.
    """
    app = FastAPI(
        title="FinRAG Engine",
        description="Production-grade RAG-based Question Answering API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {"status": "ok"}

    # Serve the chat UI — must be LAST so API routes take priority
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

    return app


# -------------------------------------------------------------------
# App Instance (for ASGI servers: uvicorn app.main:app)
# -------------------------------------------------------------------
app = create_app()