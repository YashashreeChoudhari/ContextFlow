
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from src.embeddings import embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

INDEX_NAME = "contextflow"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)


def retrieve_documents(
    question: str,
    k: int = 3,
    source: str | None = None,
):
    metadata_filter = {"source": source} if source else None

    return vector_store.similarity_search(
        question,
        k=k,
        filter=metadata_filter,
    )
