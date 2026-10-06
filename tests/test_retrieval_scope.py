from unittest.mock import patch

import pytest

from app.rag.retrieval_scope import (
    MissingCompanyError,
    ReportNotFoundError,
    resolve_retrieval_scope,
)


APPLE = {
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "market": "US",
    "country": "United States",
}

APPLE_Q2_2026 = {
    "document_id": "11111111-1111-1111-1111-111111111111",
    "document_name": "apple-q2-2026.pdf",
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "report_type": "quarterly",
    "fiscal_year": 2026,
    "fiscal_quarter": 2,
}

APPLE_FY2025 = {
    "document_id": "22222222-2222-2222-2222-222222222222",
    "document_name": "apple-fy2025.pdf",
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "report_type": "annual",
    "fiscal_year": 2025,
    "fiscal_quarter": None,
}


@patch(
    "app.rag.retrieval_scope.resolve_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_exact_quarter_resolves_document_id(
    mock_list_companies,
    mock_resolve_documents,
):
    mock_list_companies.return_value = [APPLE]
    mock_resolve_documents.return_value = [
        APPLE_Q2_2026
    ]

    scope = resolve_retrieval_scope(
        "What was AAPL revenue in Q2 2026?"
    )

    assert scope["filtered"] is True
    assert scope["company"] == APPLE
    assert scope["document_ids"] == [
        APPLE_Q2_2026["document_id"]
    ]

    mock_resolve_documents.assert_called_once_with(
        ticker="AAPL",
        exchange="NASDAQ",
        company_name="Apple Inc.",
        report_type="quarterly",
        fiscal_year=2026,
        fiscal_quarter=2,
        fiscal_half=None,
        current_only=True,
    )


@patch(
    "app.rag.retrieval_scope.resolve_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_missing_requested_report_raises(
    mock_list_companies,
    mock_resolve_documents,
):
    mock_list_companies.return_value = [APPLE]
    mock_resolve_documents.return_value = []

    with pytest.raises(ReportNotFoundError):
        resolve_retrieval_scope(
            "What was AAPL revenue in Q2 2026?"
        )


@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_period_without_company_is_rejected(
    mock_list_companies,
):
    mock_list_companies.return_value = [APPLE]

    with pytest.raises(MissingCompanyError):
        resolve_retrieval_scope(
            "What was revenue in Q2 2026?"
        )


@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_no_company_and_no_period_preserves_global_search(
    mock_list_companies,
):
    mock_list_companies.return_value = [APPLE]

    scope = resolve_retrieval_scope(
        "What is the IPO offer price?"
    )

    assert scope["filtered"] is False
    assert scope["company"] is None
    assert scope["document_ids"] is None


@patch(
    "app.rag.retrieval_scope.resolve_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_company_without_period_searches_company_documents(
    mock_list_companies,
    mock_resolve_documents,
):
    mock_list_companies.return_value = [APPLE]
    mock_resolve_documents.return_value = [
        APPLE_Q2_2026,
        APPLE_FY2025,
    ]

    scope = resolve_retrieval_scope(
        "What is AAPL debt?"
    )

    assert scope["filtered"] is True

    assert scope["document_ids"] == [
        APPLE_Q2_2026["document_id"],
        APPLE_FY2025["document_id"],
    ]


@patch(
    "app.rag.retrieval_scope.resolve_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_duplicate_document_ids_are_removed(
    mock_list_companies,
    mock_resolve_documents,
):
    mock_list_companies.return_value = [APPLE]

    mock_resolve_documents.return_value = [
        APPLE_Q2_2026,
        APPLE_Q2_2026,
    ]

    scope = resolve_retrieval_scope(
        "What is AAPL debt?"
    )

    assert scope["document_ids"] == [
        APPLE_Q2_2026["document_id"]
    ]


APPLE_FY2023 = {
    "document_id": "33333333-3333-3333-3333-333333333333",
    "document_name": "apple-fy2023.pdf",
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "report_type": "annual",
    "fiscal_year": 2023,
    "fiscal_quarter": None,
}


APPLE_FY2024 = {
    "document_id": "44444444-4444-4444-4444-444444444444",
    "document_name": "apple-fy2024.pdf",
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "report_type": "annual",
    "fiscal_year": 2024,
    "fiscal_quarter": None,
}


APPLE_Q1_2026 = {
    "document_id": "55555555-5555-5555-5555-555555555555",
    "document_name": "apple-q1-2026.pdf",
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "report_type": "quarterly",
    "fiscal_year": 2026,
    "fiscal_quarter": 1,
}


@patch(
    "app.rag.retrieval_scope.resolve_historical_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_historical_annual_window_resolves_documents(
    mock_list_companies,
    mock_resolve_historical,
):
    mock_list_companies.return_value = [APPLE]

    mock_resolve_historical.return_value = [
        APPLE_FY2023,
        APPLE_FY2024,
        APPLE_FY2025,
    ]

    scope = resolve_retrieval_scope(
        "Show AAPL gross margin for the last 5 years"
    )

    assert scope["filtered"] is True
    assert scope["historical"] is True
    assert scope["period_type"] == "annual"
    assert scope["requested_periods"] == 5
    assert scope["available_periods"] == 3

    assert scope["document_ids"] == [
        APPLE_FY2023["document_id"],
        APPLE_FY2024["document_id"],
        APPLE_FY2025["document_id"],
    ]

    mock_resolve_historical.assert_called_once_with(
        ticker="AAPL",
        exchange="NASDAQ",
        period_type="annual",
        count=5,
        current_only=True,
    )


@patch(
    "app.rag.retrieval_scope.resolve_historical_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_historical_quarter_window_resolves_documents(
    mock_list_companies,
    mock_resolve_historical,
):
    mock_list_companies.return_value = [APPLE]

    mock_resolve_historical.return_value = [
        APPLE_Q1_2026,
        APPLE_Q2_2026,
    ]

    scope = resolve_retrieval_scope(
        "Show AAPL liquidity ratios "
        "over the last 4 quarters"
    )

    assert scope["historical"] is True
    assert scope["period_type"] == "quarterly"
    assert scope["requested_periods"] == 4
    assert scope["available_periods"] == 2

    assert scope["document_ids"] == [
        APPLE_Q1_2026["document_id"],
        APPLE_Q2_2026["document_id"],
    ]

    mock_resolve_historical.assert_called_once_with(
        ticker="AAPL",
        exchange="NASDAQ",
        period_type="quarterly",
        count=4,
        current_only=True,
    )


@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_historical_window_without_company_is_rejected(
    mock_list_companies,
):
    mock_list_companies.return_value = [APPLE]

    with pytest.raises(
        MissingCompanyError
    ):
        resolve_retrieval_scope(
            "Show gross margin for the last 5 years"
        )


@patch(
    "app.rag.retrieval_scope.resolve_historical_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_empty_historical_window_raises_report_not_found(
    mock_list_companies,
    mock_resolve_historical,
):
    mock_list_companies.return_value = [APPLE]
    mock_resolve_historical.return_value = []

    with pytest.raises(
        ReportNotFoundError,
        match="historical window",
    ):
        resolve_retrieval_scope(
            "Show AAPL net margin for the last 5 years"
        )


@patch(
    "app.rag.retrieval_scope.resolve_documents"
)
@patch(
    "app.rag.retrieval_scope.resolve_historical_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_historical_request_does_not_use_standard_resolver(
    mock_list_companies,
    mock_resolve_historical,
    mock_resolve_documents,
):
    mock_list_companies.return_value = [APPLE]

    mock_resolve_historical.return_value = [
        APPLE_FY2024,
        APPLE_FY2025,
    ]

    resolve_retrieval_scope(
        "AAPL debt to equity for the last 2 years"
    )

    mock_resolve_historical.assert_called_once()
    mock_resolve_documents.assert_not_called()


@patch(
    "app.rag.retrieval_scope.resolve_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_comparative_year_can_use_latest_annual_report(
    mock_list_companies,
    mock_resolve_documents,
):
    """
    A current annual report may contain comparative prior-year
    figures. A missing separate prior-year filing must therefore
    not prevent retrieval from the available current filing.
    """

    mock_list_companies.return_value = [APPLE]

    def resolve_by_year(**kwargs):
        if kwargs["fiscal_year"] == 2025:
            return [APPLE_FY2025]

        if kwargs["fiscal_year"] == 2024:
            return []

        return []

    mock_resolve_documents.side_effect = resolve_by_year

    scope = resolve_retrieval_scope(
        "Compare AAPL annual revenue in 2025 with 2024"
    )

    assert scope["filtered"] is True
    assert scope["document_ids"] == [
        APPLE_FY2025["document_id"]
    ]


@patch(
    "app.rag.retrieval_scope.resolve_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_comparative_years_use_both_filings_when_available(
    mock_list_companies,
    mock_resolve_documents,
):
    """
    When dedicated filings exist for both comparative years,
    retrieval should use both, newest year first.
    """

    mock_list_companies.return_value = [APPLE]

    def resolve_by_year(**kwargs):
        if kwargs["fiscal_year"] == 2025:
            return [APPLE_FY2025]

        if kwargs["fiscal_year"] == 2024:
            return [APPLE_FY2024]

        return []

    mock_resolve_documents.side_effect = resolve_by_year

    scope = resolve_retrieval_scope(
        "Compare AAPL annual revenue in 2025 with 2024"
    )

    assert scope["filtered"] is True
    assert scope["document_ids"] == [
        APPLE_FY2025["document_id"],
        APPLE_FY2024["document_id"],
    ]


@patch(
    "app.rag.retrieval_scope.resolve_documents"
)
@patch(
    "app.rag.retrieval_scope.list_registered_companies"
)
def test_tickerless_company_resolves_documents_by_company_name(
    mock_list_companies,
    mock_resolve_documents,
):
    barclays = {
        "company_name": "Barclays Bank Plc",
        "ticker": None,
        "exchange": "LSE",
        "market": "UK",
        "country": "UK",
    }

    document = {
        "document_id": "barclays-2025-document",
        "document_name": "Barclays-PLC-Annual-Report-2025.pdf",
        "company_name": "Barclays Bank Plc",
        "ticker": None,
        "exchange": "LSE",
        "report_type": "annual",
        "fiscal_year": 2025,
    }

    mock_list_companies.return_value = [barclays]
    mock_resolve_documents.return_value = [document]

    scope = resolve_retrieval_scope(
        "What was Barclays PLC profit before tax in 2025?"
    )

    assert scope["filtered"] is True
    assert scope["company"]["company_name"] == "Barclays Bank Plc"
    assert scope["document_ids"] == [
        "barclays-2025-document"
    ]

    mock_resolve_documents.assert_called_once()

    kwargs = mock_resolve_documents.call_args.kwargs
    assert kwargs["ticker"] is None
    assert kwargs["company_name"] == "Barclays Bank Plc"
    assert kwargs["exchange"] == "LSE"
