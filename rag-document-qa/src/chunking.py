# src/chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

def fixed_chunking(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)