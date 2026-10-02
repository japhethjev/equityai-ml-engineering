from unittest.mock import patch

import pytest

from app.rag.financial_analysis_scope import (
    FinancialAnalysisCompanyError,
    FinancialAnalysisDocumentsError,
    resolve_financial_analysis_scope,
)


APPLE = {
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "market": "US",
    "country": "United States",
}


def annual_document(
    year: int,
    document_id: str,
) -> dict:
    return {
        "document_id": document_id,
        "document_name": (
            f"apple-fy{year}.pdf"
        ),
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "exchange": "NASDAQ",
        "report_type": "annual",
        "fiscal_year": year,
        "fiscal_quarter": None,
    }


def quarterly_document(
    year: int,
    quarter: int,
    document_id: str,
) -> dict:
    return {
        "document_id": document_id,
        "document_name": (
            f"apple-q{quarter}-{year}.pdf"
        ),
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "exchange": "NASDAQ",
        "report_type": "quarterly",
        "fiscal_year": year,
        "fiscal_quarter": quarter,
    }


ANNUAL_DOCUMENTS = [
    annual_document(
        2021,
        "11111111-1111-1111-1111-111111111111",
    ),
    annual_document(
        2022,
        "22222222-2222-2222-2222-222222222222",
    ),
    annual_document(
        2023,
        "33333333-3333-3333-3333-333333333333",
    ),
    annual_document(
        2024,
        "44444444-4444-4444-4444-444444444444",
    ),
    annual_document(
        2025,
        "55555555-5555-5555-5555-555555555555",
    ),
]


@patch(
    "app.rag.financial_analysis_scope."
    "resolve_historical_documents"
)
@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_five_year_analysis_resolves_five_annual_reports(
    mock_list_companies,
    mock_resolve_historical,
):
    mock_list_companies.return_value = [
        APPLE
    ]

    mock_resolve_historical.return_value = (
        ANNUAL_DOCUMENTS
    )

    scope = resolve_financial_analysis_scope(
        "Show AAPL revenue and gross profit "
        "for the last 5 years."
    )

    assert scope is not None
    assert scope.company == APPLE
    assert scope.period_type == "annual"
    assert scope.requested_period_count == 5
    assert scope.available_period_count == 5
    assert scope.complete_history is True

    assert scope.request.metric_keys == (
        "revenue",
        "gross_profit",
    )

    assert scope.document_ids == tuple(
        document["document_id"]
        for document in ANNUAL_DOCUMENTS
    )

    mock_resolve_historical.assert_called_once_with(
        ticker="AAPL",
        exchange="NASDAQ",
        company_name="Apple Inc.",
        period_type="annual",
        count=5,
        current_only=True,
    )


@patch(
    "app.rag.financial_analysis_scope."
    "resolve_historical_documents"
)
@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_four_quarter_analysis_resolves_quarterly_reports(
    mock_list_companies,
    mock_resolve_historical,
):
    mock_list_companies.return_value = [
        APPLE
    ]

    documents = [
        quarterly_document(
            2025,
            1,
            "11111111-1111-1111-1111-111111111111",
        ),
        quarterly_document(
            2025,
            2,
            "22222222-2222-2222-2222-222222222222",
        ),
        quarterly_document(
            2025,
            3,
            "33333333-3333-3333-3333-333333333333",
        ),
        quarterly_document(
            2025,
            4,
            "44444444-4444-4444-4444-444444444444",
        ),
    ]

    mock_resolve_historical.return_value = (
        documents
    )

    scope = resolve_financial_analysis_scope(
        "Show AAPL liquidity ratios for "
        "the last 4 quarters."
    )

    assert scope is not None
    assert scope.period_type == "quarterly"
    assert scope.requested_period_count == 4
    assert scope.available_period_count == 4
    assert scope.complete_history is True

    assert scope.request.metric_keys == (
        "current_ratio",
        "quick_ratio",
    )

    mock_resolve_historical.assert_called_once_with(
        ticker="AAPL",
        exchange="NASDAQ",
        company_name="Apple Inc.",
        period_type="quarterly",
        count=4,
        current_only=True,
    )


@patch(
    "app.rag.financial_analysis_scope."
    "resolve_historical_documents"
)
@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_partial_history_is_returned_not_invented(
    mock_list_companies,
    mock_resolve_historical,
):
    mock_list_companies.return_value = [
        APPLE
    ]

    available = ANNUAL_DOCUMENTS[-3:]

    mock_resolve_historical.return_value = (
        available
    )

    scope = resolve_financial_analysis_scope(
        "Show AAPL operating cash flow "
        "for the last 5 years."
    )

    assert scope is not None

    assert scope.requested_period_count == 5
    assert scope.available_period_count == 3
    assert scope.complete_history is False

    assert scope.documents == tuple(
        available
    )

    assert scope.document_ids == tuple(
        document["document_id"]
        for document in available
    )


@patch(
    "app.rag.financial_analysis_scope."
    "resolve_historical_documents"
)
@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_duplicate_documents_are_removed(
    mock_list_companies,
    mock_resolve_historical,
):
    mock_list_companies.return_value = [
        APPLE
    ]

    duplicate = ANNUAL_DOCUMENTS[-1]

    mock_resolve_historical.return_value = [
        ANNUAL_DOCUMENTS[-2],
        duplicate,
        duplicate,
    ]

    scope = resolve_financial_analysis_scope(
        "Show AAPL revenue for the last 3 years."
    )

    assert scope is not None

    assert scope.available_period_count == 2

    assert scope.document_ids == (
        ANNUAL_DOCUMENTS[-2]["document_id"],
        duplicate["document_id"],
    )


@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_financial_analysis_requires_registered_company(
    mock_list_companies,
):
    mock_list_companies.return_value = [
        APPLE
    ]

    with pytest.raises(
        FinancialAnalysisCompanyError,
        match="registered company",
    ):
        resolve_financial_analysis_scope(
            "Show revenue for the last 5 years."
        )


@patch(
    "app.rag.financial_analysis_scope."
    "resolve_supporting_documents"
)
@patch(
    "app.rag.financial_analysis_scope."
    "resolve_historical_documents"
)
@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_no_available_reports_raises(
    mock_list_companies,
    mock_resolve_historical,
    mock_resolve_supporting,
):
    mock_list_companies.return_value = [
        APPLE
    ]

    mock_resolve_historical.return_value = []
    mock_resolve_supporting.return_value = []

    with pytest.raises(
        FinancialAnalysisDocumentsError,
        match="No registered financial reports",
    ):
        resolve_financial_analysis_scope(
            "Show AAPL revenue for the last 5 years."
        )

    mock_resolve_supporting.assert_called_once_with(
        ticker=APPLE.get("ticker"),
        exchange=APPLE.get("exchange"),
        company_name=APPLE.get("company_name"),
        current_only=True,
    )


@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_ordinary_single_fact_question_does_not_enter_analysis(
    mock_list_companies,
):
    scope = resolve_financial_analysis_scope(
        "What was AAPL revenue in 2025?"
    )

    assert scope is None

    mock_list_companies.assert_not_called()


@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_unrelated_rag_question_does_not_enter_analysis(
    mock_list_companies,
):
    scope = resolve_financial_analysis_scope(
        "What is the IPO offer price?"
    )

    assert scope is None

    mock_list_companies.assert_not_called()


@patch(
    "app.rag.financial_analysis_scope."
    "resolve_historical_documents"
)
@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_debt_equity_three_year_scope(
    mock_list_companies,
    mock_resolve_historical,
):
    mock_list_companies.return_value = [
        APPLE
    ]

    documents = ANNUAL_DOCUMENTS[-3:]

    mock_resolve_historical.return_value = (
        documents
    )

    scope = resolve_financial_analysis_scope(
        "Show AAPL debt to equity for "
        "the last 3 years."
    )

    assert scope is not None

    assert scope.request.metric_keys == (
        "debt_to_equity",
    )

    assert scope.period_type == "annual"
    assert scope.requested_period_count == 3


@patch(
    "app.rag.financial_analysis_scope."
    "list_registered_companies"
)
def test_comparison_without_historical_horizon_not_guessed(
    mock_list_companies,
):
    mock_list_companies.return_value = [
        APPLE
    ]

    scope = resolve_financial_analysis_scope(
        "Compare AAPL revenue and gross profit."
    )

    assert scope is None
