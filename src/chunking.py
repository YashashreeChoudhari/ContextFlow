from langchain_text_splitters import MarkdownHeaderTextSplitter


def chunk_markdown(text: str):
    """Split Markdown into chunks while preserving header metadata."""

    headers_to_split_on = [
        ("#", "header 1"),
        ("##", "header 2"),
        ("###", "header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    return splitter.split_text(text)