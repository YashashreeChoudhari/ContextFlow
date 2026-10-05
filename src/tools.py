from src.retrieval import retrieve_documents

def rag_search(question: str, k: int = 3):
    documents = retrieve_documents(question, k=k)

    results = []

    for document in documents:
        results.append({
            "content": document.page_content,
            "metadata": document.metadata,
        })

    return results