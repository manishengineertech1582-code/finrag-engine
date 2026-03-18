# Dockerfile
# ---------------------------------------------------------
# Production-Grade Dockerfile for FinRAG Engine
#
# FIX LOG:
# - BUG-10a: Non-root user was created AFTER `COPY . .`.
#   The copied files were therefore owned by root, so the appuser
#   process could not write to vector_store/ at runtime.
#   FIX: Create user BEFORE COPY, then chown the app directory.
#
# - BUG-10b: vector_store/ directory was not pre-created in the image.
#   If the container starts without a mounted volume, the FAISS
#   load_local() call raises FileNotFoundError immediately.
#   FIX: Pre-create the directory and set correct ownership.
# ---------------------------------------------------------

FROM python:3.10-slim

# ---------------------------
# Environment Variables
# ---------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---------------------------
# System Dependencies
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# FIX-10a: Create non-root user BEFORE any COPY
# ---------------------------
RUN useradd -m -u 1001 appuser

# ---------------------------
# Set Working Directory
# ---------------------------
WORKDIR /app

# ---------------------------
# Install Python Dependencies
# (as root — pip needs write access to site-packages)
# ---------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------
# Copy Application Code
# ---------------------------
COPY . .

# ---------------------------
# FIX-10b: Pre-create vector_store/ and set ownership so
# appuser can write the FAISS index at runtime
# ---------------------------
RUN mkdir -p /app/vector_store && \
    chown -R appuser:appuser /app

# ---------------------------
# Switch to Non-Root User
# ---------------------------
USER appuser

# ---------------------------
# Expose Port
# ---------------------------
EXPOSE 8000

# ---------------------------
# Health Check
# ---------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------
# Start Application
# ---------------------------
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
