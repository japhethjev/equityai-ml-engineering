from unittest.mock import MagicMock, patch

import pytest

from app.rag.vector_store import (
    claim_document_for_dispatch,
)


@patch("app.rag.vector_store.get_connection")
def test_expired_queued_dispatch_can_be_reclaimed(
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

    cursor.fetchone.return_value = (
        "doc-123",
        "queued",
        0,
        0,
        0,
        None,
        None,
    )

    result = claim_document_for_dispatch("doc-123")

    assert result["document_id"] == "doc-123"
    assert result["status"] == "queued"
    assert result["claimed"] is True

    sql = cursor.execute.call_args_list[0].args[0]

    assert "ingestion_status = 'queued'" in sql
    assert "dispatch_lease_expires_at IS NULL" in sql
    assert "dispatch_lease_expires_at <= NOW()" in sql
    assert "INTERVAL '15 minutes'" in sql


@patch("app.rag.vector_store.get_connection")
def test_active_dispatch_lease_is_not_reclaimed(
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
            "queued",
            0,
            0,
            0,
            None,
            None,
        ),
    ]

    result = claim_document_for_dispatch("doc-123")

    assert result["status"] == "queued"
    assert result["claimed"] is False
    assert cursor.execute.call_count == 2


@patch("app.rag.vector_store.get_connection")
def test_nonqueued_document_is_not_claimed_for_dispatch(
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
            100,
            300,
        ),
    ]

    result = claim_document_for_dispatch("doc-123")

    assert result["status"] == "processing"
    assert result["claimed"] is False


@patch("app.rag.vector_store.get_connection")
def test_missing_document_dispatch_claim_raises(
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
        RuntimeError,
        match="Document registry record not found",
    ):
        claim_document_for_dispatch("missing-doc")
