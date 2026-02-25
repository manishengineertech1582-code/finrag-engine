# src/ingestion.py

from langchain.document_loaders.pypdf import PyPDFLoader




def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()