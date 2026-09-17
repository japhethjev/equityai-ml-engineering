from unittest.mock import patch

from app.rag.ingest import ingest_document


@patch("app.rag.ingest.store_chunks")
@patch("app.rag.ingest.embed_text")
@patch("app.rag.ingest.chunk_exists")
@patch("app.rag.ingest.chunk_pages")
@patch("app.rag.ingest.load_pdf")
def test_duplicate_document_skips_embeddings(
    mock_load_pdf,
    mock_chunk_pages,
    mock_chunk_exists,
    mock_embed_text,
    mock_store_chunks,
):
    """
    Existing chunks should be detected before embeddings
    are generated.

    This prevents unnecessary OpenAI API calls and
    duplicate database inserts.
    """

    # Fake PDF page
    mock_load_pdf.return_value = [
        {
            "document": "test.pdf",
            "page": 1,
            "text": "Example financial document.",
        }
    ]

    # Fake document chunk
    mock_chunk_pages.return_value = [
        {
            "document": "test.pdf",
            "page": 1,
            "chunk": 1,
            "text": "Example financial document.",
        }
    ]

    # Simulate the chunk already existing in PostgreSQL
    mock_chunk_exists.return_value = True

    # Run ingestion
    ingest_document("test.pdf")

    # Verify duplicate check occurred
    mock_chunk_exists.assert_called_once_with(
        document="test.pdf",
        page=1,
        chunk=1,
    )

    # Critical production behaviour:
    # OpenAI embeddings must NOT be generated.
    mock_embed_text.assert_not_called()

    # Nothing should be inserted into PostgreSQL.
    mock_store_chunks.assert_not_called()