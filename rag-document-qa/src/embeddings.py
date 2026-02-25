# src/embeddings.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_store")
    return vectorstore