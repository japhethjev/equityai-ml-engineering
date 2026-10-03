from datetime import date
from unittest.mock import MagicMock, patch

from app.rag.vector_store import (
    claim_next_orphaned_dispatch,
)


@patch("app.rag.vector_store.get_connection")
def test_claim_next_orphaned_dispatch(mock_get_connection):
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
        "hash-123",
        "report.pdf",
        "Example Plc",
        "EXM",
        "Nigeria",
        "NGX",
        "Nigeria",
        "NGN",
        "financial_report",
        "annual",
        "FY2025",
        2025,
        None,
        None,
        date(2025, 1, 1),
        date(2025, 12, 31),
        date(2026, 3, 1),
    )

    result = claim_next_orphaned_dispatch()

    assert result["document_id"] == "doc-123"
    assert result["document_hash"] == "hash-123"
    assert result["document_name"] == "report.pdf"

    assert result["metadata"]["company_name"] == (
        "Example Plc"
    )
    assert result["metadata"]["exchange"] == "NGX"
    assert result["metadata"]["fiscal_year"] == 2025

    assert result["metadata"]["period_start"] == (
        "2025-01-01"
    )
    assert result["metadata"]["period_end"] == (
        "2025-12-31"
    )
    assert result["metadata"]["publication_date"] == (
        "2026-03-01"
    )

    sql = cursor.execute.call_args.args[0]

    assert "ingestion_status = 'queued'" in sql
    assert "dispatch_lease_expires_at <= NOW()" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "LIMIT 1" in sql
    assert "INTERVAL '15 minutes'" in sql

    connection.commit.assert_called_once()


@patch("app.rag.vector_store.get_connection")
def test_claim_next_orphaned_dispatch_returns_none(
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

    cursor.fetchone.return_value = None

    result = claim_next_orphaned_dispatch()

    assert result is None
    connection.commit.assert_called_once()
