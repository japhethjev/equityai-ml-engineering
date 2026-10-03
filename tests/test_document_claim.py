from unittest.mock import MagicMock, patch

import pytest

from app.rag.vector_store import (
    claim_document_for_ingestion,
)


@patch("app.rag.vector_store.get_connection")
def test_claim_queued_document(mock_get_connection):
    connection = MagicMock()
    cursor = MagicMock()

    mock_get_connection.return_value.__enter__.return_value = (
        connection
    )
    connection.cursor.return_value.__enter__.return_value = (
        cursor
    )

    cursor.fetchone.return_value = (
        "doc-123",
        "processing",
        0,
        0,
        0,
        None,
        None,
    )

    result = claim_document_for_ingestion(
        "doc-123"
    )

    assert result["document_id"] == "doc-123"
    assert result["status"] == "processing"
    assert result["claimed"] is True

    sql = cursor.execute.call_args_list[0].args[0]

    assert (
        "ingestion_status IN ('queued', 'failed')"
        in sql
    )
    assert (
        "processing_lease_expires_at"
        in sql
    )
    assert (
        "INTERVAL '15 minutes'"
        in sql
    )
    assert (
        "dispatch_lease_expires_at = NULL"
        in sql
    )

    connection.commit.assert_called_once()


@patch("app.rag.vector_store.get_connection")
def test_completed_document_is_not_claimed(
    mock_get_connection,
):
    connection = MagicMock()
    cursor = MagicMock()

    mock_get_connection.return_value.__enter__.return_value = (
        connection
    )
    connection.cursor.return_value.__enter__.return_value = (
        cursor
    )

    cursor.fetchone.side_effect = [
        None,
        (
            "doc-123",
            "completed",
            100,
            300,
            100,
            100,
            300,
        ),
    ]

    result = claim_document_for_ingestion(
        "doc-123"
    )

    assert result["status"] == "completed"
    assert result["claimed"] is False
    assert result["processed_pages"] == 100
    assert result["processed_chunks"] == 300


@patch("app.rag.vector_store.get_connection")
def test_processing_document_is_not_claimed(
    mock_get_connection,
):
    connection = MagicMock()
    cursor = MagicMock()

    mock_get_connection.return_value.__enter__.return_value = (
        connection
    )
    connection.cursor.return_value.__enter__.return_value = (
        cursor
    )

    cursor.fetchone.side_effect = [
        None,
        (
            "doc-123",
            "processing",
            10,
            30,
            10,
            None,
            None,
        ),
    ]

    result = claim_document_for_ingestion(
        "doc-123"
    )

    assert result["status"] == "processing"
    assert result["claimed"] is False


@patch("app.rag.vector_store.get_connection")
def test_missing_document_raises_error(
    mock_get_connection,
):
    connection = MagicMock()
    cursor = MagicMock()

    mock_get_connection.return_value.__enter__.return_value = (
        connection
    )
    connection.cursor.return_value.__enter__.return_value = (
        cursor
    )

    cursor.fetchone.side_effect = [
        None,
        None,
    ]

    with pytest.raises(
        ValueError,
        match="Document not found",
    ):
        claim_document_for_ingestion(
            "missing-doc"
        )


@patch("app.rag.vector_store.get_connection")
def test_expired_processing_lease_can_be_reclaimed(
    mock_get_connection,
):
    connection = MagicMock()
    cursor = MagicMock()

    mock_get_connection.return_value.__enter__.return_value = (
        connection
    )
    connection.cursor.return_value.__enter__.return_value = (
        cursor
    )

    # A successful UPDATE represents PostgreSQL determining that
    # the processing lease is absent or expired.
    cursor.fetchone.return_value = (
        "doc-stale",
        "processing",
        20,
        40,
        20,
        100,
        200,
    )

    result = claim_document_for_ingestion(
        "doc-stale"
    )

    assert result["document_id"] == "doc-stale"
    assert result["status"] == "processing"
    assert result["claimed"] is True
    assert result["processed_pages"] == 20
    assert result["processed_chunks"] == 40
    assert result["last_processed_page"] == 20

    sql = cursor.execute.call_args_list[0].args[0]

    assert (
        "ingestion_status = 'processing'"
        in sql
    )
    assert (
        "processing_lease_expires_at IS NULL"
        in sql
    )
    assert (
        "processing_lease_expires_at <= NOW()"
        in sql
    )
    assert (
        "INTERVAL '15 minutes'"
        in sql
    )

    connection.commit.assert_called_once()
