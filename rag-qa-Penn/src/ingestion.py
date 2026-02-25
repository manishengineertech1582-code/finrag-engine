# src/ingestion.py

from langchain_community.document_loaders import PyPDFLoader  # fixed: was langchain.document_loaders.pypdf




def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()