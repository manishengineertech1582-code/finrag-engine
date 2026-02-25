# test_loader.py
from langchain_community.document_loaders import PyPDFLoader  # fixed: was langchain.document_loaders

pdf_path = "data/Manishfile.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")