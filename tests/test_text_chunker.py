from app.rag.text_chunker import chunk_pages


def test_chunk_pages():
    pages = [
        {
            "document": "test.pdf",
            "page": 1,
            "text": "A" * 2500,
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=1000,
        overlap=200,
    )

    assert len(chunks) == 3

    assert chunks[0]["document"] == "test.pdf"
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk"] == 1

    assert len(chunks[0]["text"]) == 1000
    assert len(chunks[1]["text"]) == 1000
    assert len(chunks[2]["text"]) == 900