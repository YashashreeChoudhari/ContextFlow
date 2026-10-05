from pathlib import Path

def load_document(path: str) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    document = load_document("data/sample.txt")
    print(document)