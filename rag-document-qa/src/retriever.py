# src/retriever.py

def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )