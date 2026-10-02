from unittest.mock import MagicMock, patch

import pytest

from app.rag.vector_store import (
    resolve_supporting_documents,
)


@patch("app.rag.vector_store.get_connection")
def test_supporting_document_resolver_accepts_tickerless_company(
    mock_get_connection,
):
    cursor = MagicMock()

    cursor.fetchall.return_value = [
        (
            "91e8a158-5ace-4015-8ddc-229fee7dd280",
            "dangote_refinery_valuation.pdf",
            "Dangote Petroleum Refinery FZE",
            None,
            None,
            None,
            "Nigeria",
            "NGN",
            "research",
            2026,
            None,
            None,
            None,
            None,
            None,
            1,
            True,
        )
    ]

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    mock_get_connection.return_value.__enter__.return_value = (
        connection
    )

    result = resolve_supporting_documents(
        company_name="Dangote Petroleum Refinery FZE"
    )

    assert len(result) == 1

    document = result[0]

    assert document["document_id"] == (
        "91e8a158-5ace-4015-8ddc-229fee7dd280"
    )
    assert document["document_name"] == (
        "dangote_refinery_valuation.pdf"
    )
    assert document["company_name"] == (
        "Dangote Petroleum Refinery FZE"
    )
    assert document["report_type"] == "research"
    assert document["ticker"] is None
    assert document["exchange"] is None
    assert document["currency"] == "NGN"


@patch("app.rag.vector_store.get_connection")
def test_supporting_document_resolver_accepts_ticker(
    mock_get_connection,
):
    cursor = MagicMock()

    cursor.fetchall.return_value = []

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    mock_get_connection.return_value.__enter__.return_value = (
        connection
    )

    result = resolve_supporting_documents(
        ticker="AAPL",
        exchange="NASDAQ",
    )

    assert result == []

    sql = cursor.execute.call_args.args[0]
    parameters = cursor.execute.call_args.args[1]

    assert "UPPER(ticker) = %s" in sql
    assert "UPPER(exchange) = %s" in sql
    assert "is_current = TRUE" in sql

    assert parameters == (
        "AAPL",
        "NASDAQ",
    )


def test_supporting_document_resolver_requires_issuer():
    with pytest.raises(
        ValueError,
        match="ticker or company_name is required",
    ):
        resolve_supporting_documents()


def test_supporting_document_resolver_rejects_blank_issuer():
    with pytest.raises(
        ValueError,
        match="ticker or company_name is required",
    ):
        resolve_supporting_documents(
            ticker="   ",
            company_name="   ",
        )
