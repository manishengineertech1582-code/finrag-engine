# app/main.py

from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="FinRAG Engine")

app.include_router(router)