# src/pipeline.py
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from src.retriever import get_retriever
from src.generator import build_qa_chain

def load_pipeline():
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = get_retriever(vectorstore)
    return build_qa_chain(retriever)