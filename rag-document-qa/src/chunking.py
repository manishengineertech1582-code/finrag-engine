# src/chunking.py

# Correct import from LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

def fixed_chunking(documents):
    """
    Splits documents into chunks of size 800 with 150 overlap.
    Returns a list of chunked document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)