# create_index.py
"""
Build the FAISS vector index from a PDF document.

Usage:
  python create_index.py                          # default: recursive chunking
  python create_index.py --strategy fixed
  python create_index.py --strategy semantic
  python create_index.py --strategy recursive --chunk-size 600 --overlap 100
"""

import argparse
from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.embeddings import create_vector_store


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from PDF")
    parser.add_argument("--pdf", default="data/Manishfile.pdf", help="Path to PDF")
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=["fixed", "recursive", "semantic"],
        help="Chunking strategy",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Building FAISS Index")
    print(f"  PDF:      {args.pdf}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Chunk:    size={args.chunk_size}, overlap={args.overlap}")
    print(f"{'='*50}\n")

    print("Step 1/3  Loading PDF ...")
    documents = load_pdf(args.pdf)
    total_chars = sum(len(d.page_content) for d in documents)
    print(f"  Loaded {len(documents)} page(s), {total_chars:,} total characters\n")

    if total_chars < 100:
        print("WARNING: Very little text extracted from PDF.")
        print("  The PDF may have corrupted object pointers.")
        print("  Try: pip install pdfplumber  and update src/ingestion.py\n")

    print("Step 2/3  Chunking ...")
    embeddings_for_semantic = None
    if args.strategy == "semantic":
        from langchain_openai import OpenAIEmbeddings
        embeddings_for_semantic = OpenAIEmbeddings()

    chunks = chunk_documents(
        documents,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        embeddings=embeddings_for_semantic,
    )
    print(f"  Created {len(chunks)} chunk(s)\n")

    print("Step 3/3  Embedding & saving to FAISS ...")
    create_vector_store(chunks)
    print("\nVector store saved to vector_store/")
    print("Done!\n")


if __name__ == "__main__":
    main()
