from datetime import date
from unittest.mock import MagicMock, patch

from app.rag.vector_store import resolve_documents


DOCUMENT_ID = "11111111-1111-1111-1111-111111111111"


def make_connection(rows):
    cursor = MagicMock()
    cursor.fetchall.return_value = rows

    cursor_context = MagicMock()
    cursor_context.__enter__.return_value = cursor
    cursor_context.__exit__.return_value = False

    connection = MagicMock()
    connection.cursor.return_value = cursor_context

    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection
    connection_context.__exit__.return_value = False

    return connection_context, cursor


def sample_row(
    report_type="annual",
    fiscal_year=2025,
    fiscal_quarter=None,
    fiscal_half=None,
):
    return (
        DOCUMENT_ID,
        "report.pdf",
        "Example Plc",
        "ABC",
        "NGX",
        "Nigeria",
        "Nigeria",
        "NGN",
        report_type,
        fiscal_year,
        fiscal_quarter,
        fiscal_half,
        date(2025, 1, 1),
        date(2025, 12, 31),
        date(2026, 3, 20),
        1,
        True,
    )


@patch("app.rag.vector_store.get_connection")
def test_resolve_annual_report(mock_get_connection):
    connection_context, cursor = make_connection(
        [sample_row()]
    )
    mock_get_connection.return_value = connection_context

    results = resolve_documents(
        ticker="abc",
        exchange="ngx",
        report_type="annual",
        fiscal_year=2025,
    )

    assert len(results) == 1
    assert results[0]["document_id"] == DOCUMENT_ID
    assert results[0]["ticker"] == "ABC"
    assert results[0]["exchange"] == "NGX"
    assert results[0]["report_type"] == "annual"
    assert results[0]["fiscal_year"] == 2025
    assert results[0]["fiscal_quarter"] is None
    assert results[0]["is_current"] is True

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]

    assert "UPPER(ticker) = %s" in sql
    assert "UPPER(exchange) = %s" in sql
    assert "LOWER(report_type) = %s" in sql
    assert "fiscal_year = %s" in sql
    assert "is_current = TRUE" in sql

    assert params == (
        "ABC",
        "NGX",
        "annual",
        2025,
    )


@patch("app.rag.vector_store.get_connection")
def test_resolve_quarter_four_report(mock_get_connection):
    connection_context, cursor = make_connection(
        [
            sample_row(
                report_type="quarterly",
                fiscal_quarter=4,
            )
        ]
    )
    mock_get_connection.return_value = connection_context

    results = resolve_documents(
        ticker="ABC",
        exchange="NGX",
        report_type="quarterly",
        fiscal_year=2025,
        fiscal_quarter=4,
    )

    assert results[0]["report_type"] == "quarterly"
    assert results[0]["fiscal_quarter"] == 4

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]

    assert "fiscal_quarter = %s" in sql
    assert params == (
        "ABC",
        "NGX",
        "quarterly",
        2025,
        4,
    )


@patch("app.rag.vector_store.get_connection")
def test_resolve_half_year_report(mock_get_connection):
    connection_context, cursor = make_connection(
        [
            sample_row(
                report_type="half_year",
                fiscal_half=1,
            )
        ]
    )
    mock_get_connection.return_value = connection_context

    results = resolve_documents(
        ticker="ABC",
        exchange="LSE",
        report_type="half_year",
        fiscal_year=2025,
        fiscal_half=1,
    )

    assert results[0]["report_type"] == "half_year"
    assert results[0]["fiscal_half"] == 1

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]

    assert "fiscal_half = %s" in sql
    assert "UPPER(exchange) = %s" in sql

    assert params == (
        "ABC",
        "LSE",
        "half_year",
        2025,
        1,
    )


@patch("app.rag.vector_store.get_connection")
def test_exchange_disambiguates_same_ticker(
    mock_get_connection,
):
    connection_context, cursor = make_connection([])
    mock_get_connection.return_value = connection_context

    resolve_documents(
        ticker="ABC",
        exchange="NASDAQ",
        fiscal_year=2025,
    )

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]

    assert "UPPER(exchange) = %s" in sql
    assert params == (
        "ABC",
        "NASDAQ",
        2025,
    )


@patch("app.rag.vector_store.get_connection")
def test_current_only_excludes_superseded_filings(
    mock_get_connection,
):
    connection_context, cursor = make_connection([])
    mock_get_connection.return_value = connection_context

    resolve_documents(
        ticker="ABC",
        fiscal_year=2025,
        current_only=True,
    )

    sql = cursor.execute.call_args.args[0]

    assert "is_current = TRUE" in sql


@patch("app.rag.vector_store.get_connection")
def test_current_only_can_be_disabled(
    mock_get_connection,
):
    connection_context, cursor = make_connection([])
    mock_get_connection.return_value = connection_context

    resolve_documents(
        ticker="ABC",
        fiscal_year=2025,
        current_only=False,
    )

    sql = cursor.execute.call_args.args[0]

    assert "is_current = TRUE" not in sql


@patch("app.rag.vector_store.get_connection")
def test_resolve_documents_no_match_returns_empty_list(
    mock_get_connection,
):
    connection_context, _ = make_connection([])
    mock_get_connection.return_value = connection_context

    results = resolve_documents(
        ticker="ABC",
        exchange="NGX",
        fiscal_year=1999,
    )

    assert results == []


def test_resolve_documents_requires_ticker():
    try:
        resolve_documents(ticker="   ")
    except ValueError as exc:
        assert "ticker is required" in str(exc)
    else:
        raise AssertionError(
            "Expected empty ticker to be rejected"
        )


@patch("app.rag.vector_store.get_connection")
def test_resolve_documents_orders_by_financial_report_chronology(
    mock_get_connection,
):
    """
    Document resolution must order filings using actual
    reporting and publication dates rather than assuming
    calendar fiscal years or quarter chronology.
    """

    connection_context, cursor = make_connection([])
    mock_get_connection.return_value = connection_context

    resolve_documents(
        ticker="ABC",
        exchange="NGX",
    )

    sql = cursor.execute.call_args.args[0]

    assert "period_end DESC NULLS LAST" in sql
    assert "publication_date DESC NULLS LAST" in sql
    assert "filing_version DESC" in sql
    assert "created_at DESC" in sql

    period_position = sql.index(
        "period_end DESC NULLS LAST"
    )
    publication_position = sql.index(
        "publication_date DESC NULLS LAST"
    )
    version_position = sql.index(
        "filing_version DESC"
    )
    created_position = sql.index(
        "created_at DESC"
    )

    assert (
        period_position
        < publication_position
        < version_position
        < created_position
    )


@patch("app.rag.vector_store.get_connection")
def test_hybrid_search_retrieves_candidates_per_document(
    mock_get_connection,
):
    """
    Multi-document retrieval must execute an independently
    scoped candidate search for every requested filing.
    """

    document_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]

    connection_context, cursor = make_connection([])
    mock_get_connection.return_value = connection_context

    from app.rag.vector_store import hybrid_search

    hybrid_search(
        query="Compare revenue",
        query_embedding=[0.1, 0.2],
        limit=5,
        document_ids=document_ids,
    )

    assert cursor.execute.call_count == 2

    first_call = cursor.execute.call_args_list[0]
    second_call = cursor.execute.call_args_list[1]

    first_sql = first_call.args[0]
    first_params = first_call.args[1]
    second_params = second_call.args[1]

    assert (
        "WHERE dc.document_id = %s::uuid"
        in first_sql
    )

    assert "dc.embedding <=> %s" in first_sql
    assert "ts_rank(" in first_sql

    assert first_params[2] == document_ids[0]
    assert second_params[2] == document_ids[1]

    assert first_params[-1] == 5
    assert second_params[-1] == 5


@patch("app.rag.vector_store.get_connection")
def test_hybrid_search_deduplicates_document_ids(
    mock_get_connection,
):
    """
    Repeated document IDs must not cause duplicate database
    retrieval work.
    """

    document_id = (
        "11111111-1111-1111-1111-111111111111"
    )

    connection_context, cursor = make_connection([])
    mock_get_connection.return_value = connection_context

    from app.rag.vector_store import hybrid_search

    hybrid_search(
        query="Revenue",
        query_embedding=[0.1, 0.2],
        limit=5,
        document_ids=[
            document_id,
            document_id,
        ],
    )

    assert cursor.execute.call_count == 1

    params = cursor.execute.call_args.args[1]

    assert params[2] == document_id


@patch("app.rag.vector_store.get_connection")
def test_hybrid_search_without_scope_preserves_global_retrieval(
    mock_get_connection,
):
    """
    Unscoped questions must preserve the existing global
    hybrid-search behaviour.
    """

    connection_context, cursor = make_connection([])
    mock_get_connection.return_value = connection_context

    from app.rag.vector_store import hybrid_search

    hybrid_search(
        query="IPO offer price",
        query_embedding=[0.1, 0.2],
        limit=5,
        document_ids=None,
    )

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]

    assert "document_id = ANY" not in sql
    assert "dc.embedding <=> %s" in sql
    assert "ts_rank(" in sql

    assert params == (
        params[0],
        "IPO offer price",
        params[2],
        "IPO offer price",
        5,
    )


@patch("app.rag.vector_store.get_connection")
def test_resolve_historical_annual_documents(
    mock_get_connection,
):
    from app.rag.vector_store import (
        resolve_historical_documents,
    )

    rows = [
        sample_row(
            report_type="annual",
            fiscal_year=2021,
        ),
        sample_row(
            report_type="annual",
            fiscal_year=2022,
        ),
        sample_row(
            report_type="annual",
            fiscal_year=2023,
        ),
    ]

    connection_context, cursor = make_connection(
        rows
    )
    mock_get_connection.return_value = (
        connection_context
    )

    results = resolve_historical_documents(
        ticker="abc",
        exchange="ngx",
        period_type="annual",
        count=3,
    )

    assert len(results) == 3

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]

    assert "LOWER(report_type) = %s" in sql
    assert "is_current = TRUE" in sql
    assert "LIMIT %s" in sql

    assert params == (
        "ABC",
        "NGX",
        "annual",
        3,
    )


@patch("app.rag.vector_store.get_connection")
def test_resolve_historical_quarterly_documents(
    mock_get_connection,
):
    from app.rag.vector_store import (
        resolve_historical_documents,
    )

    rows = [
        sample_row(
            report_type="quarterly",
            fiscal_year=2025,
            fiscal_quarter=1,
        ),
        sample_row(
            report_type="quarterly",
            fiscal_year=2025,
            fiscal_quarter=2,
        ),
    ]

    connection_context, cursor = make_connection(
        rows
    )
    mock_get_connection.return_value = (
        connection_context
    )

    results = resolve_historical_documents(
        ticker="ABC",
        period_type="quarterly",
        count=8,
    )

    assert len(results) == 2

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]

    assert "LOWER(report_type) = %s" in sql
    assert "fiscal_quarter DESC" in sql

    assert params == (
        "ABC",
        "quarterly",
        8,
    )


@patch("app.rag.vector_store.get_connection")
def test_historical_resolver_preserves_shortfall(
    mock_get_connection,
):
    from app.rag.vector_store import (
        resolve_historical_documents,
    )

    # Database contains only three available reports,
    # even though five were requested.
    rows = [
        sample_row(fiscal_year=2023),
        sample_row(fiscal_year=2024),
        sample_row(fiscal_year=2025),
    ]

    connection_context, _ = make_connection(
        rows
    )
    mock_get_connection.return_value = (
        connection_context
    )

    results = resolve_historical_documents(
        ticker="ABC",
        period_type="annual",
        count=5,
    )

    assert len(results) == 3


def test_historical_resolver_rejects_invalid_period_type():
    from app.rag.vector_store import (
        resolve_historical_documents,
    )

    try:
        resolve_historical_documents(
            ticker="ABC",
            period_type="monthly",
            count=5,
        )
    except ValueError as exc:
        assert (
            "period_type must be annual or quarterly"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected invalid period type to fail"
        )


def test_historical_resolver_rejects_zero_count():
    from app.rag.vector_store import (
        resolve_historical_documents,
    )

    try:
        resolve_historical_documents(
            ticker="ABC",
            period_type="annual",
            count=0,
        )
    except ValueError as exc:
        assert (
            "count must be greater than zero"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected zero count to fail"
        )


@patch(
    "app.rag.vector_store.get_connection"
)
def test_get_document_chunks_returns_ordered_chunks(
    mock_get_connection,
):
    from app.rag.vector_store import (
        get_document_chunks,
    )

    document_id = (
        "11111111-1111-1111-1111-111111111111"
    )

    connection = (
        mock_get_connection.return_value
        .__enter__.return_value
    )

    cursor = (
        connection.cursor.return_value
        .__enter__.return_value
    )

    cursor.fetchall.return_value = [
        (
            document_id,
            "report.pdf",
            1,
            0,
            "2024 2025 H1 2026",
        ),
        (
            document_id,
            "report.pdf",
            1,
            1,
            "Revenue 100 120 80",
        ),
    ]

    result = get_document_chunks(
        document_id
    )

    assert result == [
        {
            "document_id": document_id,
            "document": "report.pdf",
            "page": 1,
            "chunk": 0,
            "content": "2024 2025 H1 2026",
        },
        {
            "document_id": document_id,
            "document": "report.pdf",
            "page": 1,
            "chunk": 1,
            "content": "Revenue 100 120 80",
        },
    ]

    cursor.execute.assert_called_once()

    sql, parameters = (
        cursor.execute.call_args.args
    )

    assert (
        "WHERE document_id = %s::uuid"
        in sql
    )

    assert "ORDER BY page, chunk" in sql

    assert parameters == (
        document_id,
    )


@patch(
    "app.rag.vector_store.get_connection"
)
def test_get_document_chunks_returns_empty_when_none(
    mock_get_connection,
):
    from app.rag.vector_store import (
        get_document_chunks,
    )

    connection = (
        mock_get_connection.return_value
        .__enter__.return_value
    )

    cursor = (
        connection.cursor.return_value
        .__enter__.return_value
    )

    cursor.fetchall.return_value = []

    assert get_document_chunks(
        "11111111-1111-1111-1111-111111111111"
    ) == []


def test_get_document_chunks_requires_document_id():
    from app.rag.vector_store import (
        get_document_chunks,
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="document_id is required",
    ):
        get_document_chunks(" ")


@patch("app.rag.vector_store.get_connection")
def test_list_documents_returns_ingestion_lifecycle(
    mock_get_connection,
):
    from app.rag.vector_store import list_documents

    queued_id = "11111111-1111-1111-1111-111111111111"
    processing_id = "22222222-2222-2222-2222-222222222222"
    completed_id = "33333333-3333-3333-3333-333333333333"
    failed_id = "44444444-4444-4444-4444-444444444444"

    rows = [
        (
            queued_id,
            "queued_report.pdf",
            "Queued Plc",
            "QUE",
            "LSE",
            "UK",
            "United Kingdom",
            "GBP",
            "annual",
            2025,
            None,
            None,
            date(2025, 1, 1),
            date(2025, 12, 31),
            date(2026, 3, 1),
            "queued",
            0,
            None,
            0,
            None,
            0,
            None,
            None,
            None,
        ),
        (
            processing_id,
            "large_612_page_report.pdf",
            "Large Report Plc",
            "LRG",
            "LSE",
            "UK",
            "United Kingdom",
            "GBP",
            "annual",
            2025,
            None,
            None,
            date(2025, 1, 1),
            date(2025, 12, 31),
            date(2026, 3, 15),
            "processing",
            347,
            612,
            2184,
            None,
            347,
            None,
            None,
            None,
        ),
        (
            completed_id,
            "completed_612_page_report.pdf",
            "Completed Plc",
            "CMP",
            "LSE",
            "UK",
            "United Kingdom",
            "GBP",
            "annual",
            2025,
            None,
            None,
            date(2025, 1, 1),
            date(2025, 12, 31),
            date(2026, 3, 20),
            "completed",
            612,
            612,
            3846,
            3846,
            612,
            None,
            None,
            None,
        ),
        (
            failed_id,
            "failed_large_report.pdf",
            "Failed Plc",
            "FLD",
            "LSE",
            "UK",
            "United Kingdom",
            "GBP",
            "annual",
            2025,
            None,
            None,
            date(2025, 1, 1),
            date(2025, 12, 31),
            date(2026, 3, 25),
            "failed",
            420,
            700,
            2600,
            None,
            420,
            "Embedding provider unavailable",
            None,
            None,
        ),
    ]

    connection_context, cursor = make_connection(rows)
    mock_get_connection.return_value = connection_context

    result = list_documents()

    assert len(result) == 4

    assert result[0]["status"] == "queued"
    assert result[0]["processed_pages"] == 0
    assert result[0]["total_pages"] is None

    assert result[1]["status"] == "processing"
    assert result[1]["processed_pages"] == 347
    assert result[1]["total_pages"] == 612
    assert result[1]["processed_chunks"] == 2184
    assert result[1]["pages"] == 347
    assert result[1]["chunks"] == 2184

    assert result[2]["status"] == "completed"
    assert result[2]["processed_pages"] == 612
    assert result[2]["total_pages"] == 612
    assert result[2]["processed_chunks"] == 3846
    assert result[2]["total_chunks"] == 3846

    assert result[3]["status"] == "failed"
    assert result[3]["processed_pages"] == 420
    assert result[3]["total_pages"] == 700
    assert (
        result[3]["error_message"]
        == "Embedding provider unavailable"
    )

    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args.args[0]

    assert "FROM documents" in sql
    assert "ingestion_status" in sql
    assert "processed_pages" in sql
    assert "total_pages" in sql
    assert "ORDER BY created_at DESC" in sql
