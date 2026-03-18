# app/main.py

"""
Main Application Entry Point

FIX LOG:
- BUG-6: No logging configuration and no dotenv loading.
  Without `load_dotenv()`, OPENAI_API_KEY is never read from .env
  when running locally, causing every pipeline call to fail with
  an authentication error.

  FIX: Added `load_dotenv()` call at module startup and a
  `logging.basicConfig()` so all module-level loggers produce output.
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