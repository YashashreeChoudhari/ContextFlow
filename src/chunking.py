def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The input document.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap

    return chunks

if __name__ == "__main__":
    from ingestion import load_document

    text = load_document("data/sample.txt")

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap=50
    )

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk)