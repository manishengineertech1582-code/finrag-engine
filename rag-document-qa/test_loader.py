# test_loader.py
from langchain.document_loaders import PyPDFLoader

pdf_path = "data/Manishfile.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")