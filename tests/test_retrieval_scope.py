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
