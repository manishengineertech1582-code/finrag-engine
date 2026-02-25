# src/generator.py

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def build_qa_chain(retriever):
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa