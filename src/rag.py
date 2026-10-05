import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from embeddings import embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

INDEX_NAME = "contextflow"

pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(INDEX_NAME):
    raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist.")

index = pc.Index(INDEX_NAME)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

def retrieve_documents(question: str,k: int = 3,source: str | None = None,):
    filter = {"source": source} if source else None

    return vector_store.similarity_search(
        question,
        k=k,
        filter=filter,
    )

def build_context(documents):
    context_parts = []

    for i, document in enumerate(documents, start=1):
        metadata = document.metadata

        source = metadata.get("source", "unknown")
        header_1 = metadata.get("header 1", "")
        header_2 = metadata.get("header 2", "")
        header_3 = metadata.get("header 3", "")

        section = " > ".join(
            part for part in [header_1, header_2, header_3] if part
        )

        context_parts.append(
            f"""
[Source {i}]
File: {source}
Section: {section}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)

def generate_answer(question: str, context: str):
    prompt = f"""
You are ContextFlow, a research assistant.

Answer the user's question using ONLY the information provided in the context.

Rules:
1. Do not invent facts.
2. Do not use information outside the context.
3. If the context does not contain enough information, say so.
4. Give a concise and clear answer.
5. Mention relevant source sections when useful.

Context:
{context}

User Question:
{question}

Answer:
"""

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            return response.text
        except Exception as error:
            print(f"\nGemini request failed (attempt {attempt}/{max_attempts})")
            print(f"Error: {error}")

            if attempt < max_attempts:
                print("Retrying in 20 seconds...")
                time.sleep(20)
            else:
                print(f"Gemini request failed after {max_attempts} attempts.")
                raise
def answer_question(
    question: str,
    k: int = 3,
    source: str | None = None,
):
    print("\nSearching knowledge base...")

    documents = retrieve_documents(
        question,
        k=k,
        source=source, )

    if not documents:
        return "I could not find relevant information in the knowledge base."

    print(f"Retrieved {len(documents)} documents.")
    context = build_context(documents)
    print("\nGenerating answer...")
    return generate_answer(question, context)

def display_sources(question: str, k: int = 3):
    documents = retrieve_documents(question, k=k)

    print("\nRetrieved Sources")
    print("=" * 60)

    for i, document in enumerate(documents, start=1):
        print(f"\nSource {i}")
        print(f"File: {document.metadata.get('source', 'unknown')}")
        print(f"Header 1: {document.metadata.get('header 1', '')}")
        print(f"Header 2: {document.metadata.get('header 2', '')}")
        print(f"Header 3: {document.metadata.get('header 3', '')}")
        print("\nContent:")
        print(document.page_content)
        print("-" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("ContextFlow RAG")
    print("=" * 60)

    question = input("\nAsk a question: ").strip()

    if not question:
        print("Please enter a question.")
        raise SystemExit

    answer = answer_question(question, k=3)

    print("\n")
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

    print("\n")
    display_sources(question, k=3)