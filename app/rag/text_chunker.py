def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping text chunks
    while preserving document and page metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []

    for page in pages:
        text = page["text"]

        if not text:
            continue

        start = 0
        chunk_number = 1

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "document": page["document"],
                        "page": page["page"],
                        "chunk": chunk_number,
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            start += chunk_size - overlap
            chunk_number += 1

    return chunks