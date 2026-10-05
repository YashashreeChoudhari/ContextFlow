import os
import time

from dotenv import load_dotenv
from pinecone_setup import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("PINECONE_API_KEY is not set")

pc = Pinecone(api_key=api_key)

INDEX_NAME = "contextflow"

# Gemini gemini-embedding-001 produces 3072-dimensional
# embeddings by default.
DIMENSION = 3072

if not pc.has_index(INDEX_NAME):
    print(f"Creating Pinecone index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    # Give Pinecone a moment to initialize the index.
    time.sleep(5)

else:
    print(f"Index already exists: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

print("Pinecone index is ready!")
print(index.describe_index_stats())