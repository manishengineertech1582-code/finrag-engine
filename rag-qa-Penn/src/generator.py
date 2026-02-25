# src/generator.py
from langchain_openai import ChatOpenAI                 # fixed: was langchain.chat_models
from langchain.chains import RetrievalQA

def build_qa_chain(retriever):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # cheaper than GPT-4
        temperature=0,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain