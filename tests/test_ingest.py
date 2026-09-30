from unittest.mock import patch

from app.rag.ingest import ingest_document


@patch("app.rag.ingest.complete_document")
@patch("app.rag.ingest.update_document_progress")
@patch("app.rag.ingest.register_document")
@patch("app.rag.ingest.calculate_file_hash")
@patch("app.rag.ingest.store_chunks")
@patch("app.rag.ingest.embed_text")
@patch("app.rag.ingest.chunk_exists")
@patch("app.rag.ingest.chunk_pages")
@patch("app.rag.ingest.iter_pdf_pages")
def test_duplicate_document_skips_embeddings(
    mock_iter_pdf_pages,
    mock_chunk_pages,
    mock_chunk_exists,
    mock_embed_text,
    mock_store_chunks,
    mock_calculate_file_hash,
    mock_register_document,
    mock_update_document_progress,
    mock_complete_document,
):
    """
    Existing chunks should be detected before embeddings
    are generated.

    This prevents unnecessary OpenAI API calls and
    duplicate database inserts.
    """

    mock_calculate_file_hash.return_value = "test-sha256"
    mock_register_document.return_value = {
        "document_id": "11111111-1111-1111-1111-111111111111",
        "status": "processing",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": True,
    }

    # Fake PDF page
    mock_iter_pdf_pages.return_value = [
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
        document_id="11111111-1111-1111-1111-111111111111",
        page=1,
        chunk=1,
    )

    # Critical production behaviour:
    # OpenAI embeddings must NOT be generated.
    mock_embed_text.assert_not_called()

    # Nothing should be inserted into PostgreSQL.
    mock_store_chunks.assert_not_called()

@patch("app.rag.ingest.iter_pdf_pages")
@patch("app.rag.ingest.embed_text")
@patch("app.rag.ingest.register_document")
@patch("app.rag.ingest.calculate_file_hash")
def test_completed_duplicate_returns_without_reingestion(
    mock_calculate_file_hash,
    mock_register_document,
    mock_embed_text,
    mock_iter_pdf_pages,
):
    """A completed exact duplicate must not be ingested again."""

    document_id = "22222222-2222-2222-2222-222222222222"

    mock_calculate_file_hash.return_value = "completed-hash"
    mock_register_document.return_value = {
        "document_id": document_id,
        "status": "completed",
        "processed_pages": 300,
        "processed_chunks": 900,
        "last_processed_page": 300,
        "total_pages": 300,
        "total_chunks": 900,
        "is_new": False,
    }

    result = ingest_document("test.pdf")

    assert result["document_id"] == document_id
    assert result["status"] == "already_ingested"
    assert result["pages"] == 300
    assert result["total_chunks"] == 900
    assert result["inserted"] == 0

    mock_iter_pdf_pages.assert_not_called()
    mock_embed_text.assert_not_called()


@patch("app.rag.ingest.register_document")
@patch("app.rag.ingest.calculate_file_hash")
def test_processing_duplicate_is_rejected(
    mock_calculate_file_hash,
    mock_register_document,
):
    """A second worker must not process the same exact file."""

    mock_calculate_file_hash.return_value = "processing-hash"
    mock_register_document.return_value = {
        "document_id": "33333333-3333-3333-3333-333333333333",
        "status": "processing",
        "processed_pages": 50,
        "processed_chunks": 150,
        "last_processed_page": 50,
        "total_pages": None,
        "total_chunks": None,
        "is_new": False,
    }

    try:
        ingest_document("test.pdf")
    except RuntimeError as exc:
        assert "already being processed" in str(exc)
    else:
        raise AssertionError(
            "Expected concurrent ingestion to be rejected"
        )


@patch("app.rag.ingest.complete_document")
@patch("app.rag.ingest.update_document_progress")
@patch("app.rag.ingest.store_chunks")
@patch("app.rag.ingest.embed_text")
@patch("app.rag.ingest.chunk_exists")
@patch("app.rag.ingest.chunk_pages")
@patch("app.rag.ingest.iter_pdf_pages")
@patch("app.rag.ingest.register_document")
@patch("app.rag.ingest.calculate_file_hash")
def test_failed_document_resumes_without_reembedding_existing_chunks(
    mock_calculate_file_hash,
    mock_register_document,
    mock_iter_pdf_pages,
    mock_chunk_pages,
    mock_chunk_exists,
    mock_embed_text,
    mock_store_chunks,
    mock_update_document_progress,
    mock_complete_document,
):
    """A failed document reuses its UUID and skips stored chunks."""

    document_id = "44444444-4444-4444-4444-444444444444"

    mock_calculate_file_hash.return_value = "failed-hash"
    mock_register_document.return_value = {
        "document_id": document_id,
        "status": "failed",
        "processed_pages": 1,
        "processed_chunks": 1,
        "last_processed_page": 1,
        "total_pages": None,
        "total_chunks": None,
        "is_new": False,
    }

    mock_iter_pdf_pages.return_value = [
        {
            "document": "test.pdf",
            "page": 1,
            "text": "Existing page",
        },
        {
            "document": "test.pdf",
            "page": 2,
            "text": "New page",
        },
    ]

    mock_chunk_pages.side_effect = [
        [{
            "document": "test.pdf",
            "page": 1,
            "chunk": 1,
            "text": "Existing chunk",
        }],
        [{
            "document": "test.pdf",
            "page": 2,
            "chunk": 1,
            "text": "New chunk",
        }],
    ]

    # Page 1 already exists; page 2 still needs ingestion.
    mock_chunk_exists.side_effect = [True, False]
    mock_embed_text.return_value = [0.1, 0.2, 0.3]
    mock_store_chunks.return_value = {
        "inserted": 1,
        "skipped": 0,
    }

    result = ingest_document("test.pdf")

    assert result["document_id"] == document_id
    assert result["existing_chunks"] == 1
    assert result["new_chunks"] == 1
    assert result["inserted"] == 1

    # Only the missing chunk receives a new embedding.
    mock_embed_text.assert_called_once_with("New chunk")

    mock_store_chunks.assert_called_once()

    # flush_batch() clears the original pending_chunks list after
    # store_chunks() returns. Capture the call semantically rather
    # than inspecting that mutable list after it has been cleared.
    assert (
        mock_store_chunks.call_args.kwargs["document_id"]
        == document_id
    )

    # Both pending lists are cleared after store_chunks returns.
    # Behaviour is therefore verified through the embedding call,
    # store call count and document_id.
    assert mock_store_chunks.call_count == 1

    mock_complete_document.assert_called_once_with(
        document_id=document_id,
        total_pages=2,
        total_chunks=2,
    )
