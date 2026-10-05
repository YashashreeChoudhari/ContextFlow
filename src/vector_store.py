import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from ingestion import load_document
from chunking import chunk_markdown
from embeddings import embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")

INDEX_NAME = "contextflow"

pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(INDEX_NAME):
    raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist.")

index = pc.Index(INDEX_NAME)

document_path = PROJECT_ROOT / "data" / "sample.md"

text = load_document(str(document_path))

chunks = chunk_markdown(text)

for chunk in chunks:
    chunk.metadata["source"] = document_path.name

print(f"Loaded: {document_path.name}")
print(f"Created {len(chunks)} chunks")

index.delete(delete_all=True)

print("Old vectors deleted")

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

ids = vector_store.add_documents(
    documents=chunks
)

print(f"Uploaded {len(ids)} chunks")

stats = index.describe_index_stats()

print(stats)