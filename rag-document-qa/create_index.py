# create_index.py
from src.ingestion import load_pdf
from src.chunking import fixed_chunking
from src.embeddings import create_vector_store

def main():
    pdf_path = "data/Manishfile.pdf"

    print("Loading PDF...")
    documents = load_pdf(pdf_path)

    print("Chunking...")
    chunks = fixed_chunking(documents)

    print("Creating vector store...")
    create_vector_store(chunks)

    print("Vector store created successfully!")

if __name__ == "__main__":
    main()