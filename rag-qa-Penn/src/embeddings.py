# src/embeddings.py

from langchain_openai import OpenAIEmbeddings           # fixed: was langchain.embeddings
from langchain_community.vectorstores import FAISS      # fixed: was langchain.vectorstores

def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_store")
    return vectorstore