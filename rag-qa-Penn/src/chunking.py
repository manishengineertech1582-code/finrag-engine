# src/chunking.py
"""
Three chunking strategies:
  - fixed     : CharacterTextSplitter, hard char boundaries
  - recursive : RecursiveCharacterTextSplitter (default, respects sentence breaks)
  - semantic  : SemanticChunker using embedding cosine-distance breakpoints
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


# ── 1. Fixed-size chunking ────────────────────────────────────────────────────

def fixed_chunking(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Hard character-boundary splitting.
    Fast and deterministic; may cut sentences mid-way.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )
    chunks = splitter.split_documents(documents)
    print(f"[fixed_chunking] {len(chunks)} chunks created "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ── 2. Recursive chunking ─────────────────────────────────────────────────────

def recursive_chunking(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Splits on paragraphs → sentences → words → chars in order.
    Better context preservation than fixed splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[recursive_chunking] {len(chunks)} chunks created "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ── 3. Semantic chunking ──────────────────────────────────────────────────────

def semantic_chunking(documents, embeddings=None, breakpoint_threshold: float = 95.0):
    """
    Groups sentences by embedding cosine-similarity breakpoints.
    Chunks represent coherent topics rather than fixed character counts.

    Args:
        documents:              List of LangChain Document objects.
        embeddings:             OpenAIEmbeddings instance. If None, falls back
                                to recursive_chunking with a warning.
        breakpoint_threshold:   Percentile of cosine-distance scores used to
                                detect topic boundaries (default 95th percentile).
    Returns:
        List of chunked Document objects.
    """
    if embeddings is None:
        print("[semantic_chunking] WARNING: No embeddings provided — "
              "falling back to recursive_chunking.")
        return recursive_chunking(documents)

    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        print("[semantic_chunking] WARNING: langchain-experimental not installed. "
              "Run: pip install langchain-experimental\n"
              "Falling back to recursive_chunking.")
        return recursive_chunking(documents)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold,
    )
    chunks = splitter.split_documents(documents)
    print(f"[semantic_chunking] {len(chunks)} chunks created "
          f"(threshold_percentile={breakpoint_threshold})")
    return chunks


# ── Convenience dispatcher ────────────────────────────────────────────────────

def chunk_documents(documents, strategy: str = "recursive",
                    chunk_size: int = 800, chunk_overlap: int = 150,
                    embeddings=None):
    """
    Dispatch to the right chunking strategy by name.

    Args:
        documents:    List of LangChain Document objects.
        strategy:     'fixed' | 'recursive' | 'semantic'
        chunk_size:   Used by fixed and recursive strategies.
        chunk_overlap: Used by fixed and recursive strategies.
        embeddings:   Required for semantic strategy.

    Returns:
        List of chunked Document objects.
    """
    strategy = strategy.lower().strip()
    if strategy == "fixed":
        return fixed_chunking(documents, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        return recursive_chunking(documents, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        return semantic_chunking(documents, embeddings)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose: 'fixed', 'recursive', or 'semantic'.")
