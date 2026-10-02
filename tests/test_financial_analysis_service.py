from unittest.mock import patch

import pytest

from app.rag.financial_analysis_scope import (
    FinancialAnalysisCompanyError,
    FinancialAnalysisDocumentsError,
)
from app.rag.financial_analysis_service import (
    answer_financial_analysis_question,
)
from app.rag.financial_facts import (
    make_reported_fact,
)


@patch(
    "app.rag.financial_analysis_service."
    "resolve_financial_analysis_scope"
)
def test_non_financial_question_returns_none(
    mock_scope,
):
    mock_scope.return_value = None

    result = answer_financial_analysis_question(
        "What is the IPO offer price?"
    )

    assert result is None


@patch(
    "app.rag.financial_analysis_service."
    "build_financial_presentation"
)
@patch(
    "app.rag.financial_analysis_service."
    "execute_financial_analysis"
)
@patch(
    "app.rag.financial_analysis_service."
    "extract_financial_facts_from_document"
)
@patch(
    "app.rag.financial_analysis_service."
    "resolve_financial_analysis_scope"
)
def test_historical_financial_analysis_is_orchestrated(
    mock_scope,
    mock_extract,
    mock_execute,
    mock_present,
):
    scope = type(
        "Scope",
        (),
        {
            "company": {
                "company_name": "Example Plc",
                "ticker": "ABC",
                "exchange": "NGX",
                "currency": "NGN",
            },
            "documents": (
                {
                    "document_id": "doc-2024",
                },
                {
                    "document_id": "doc-2025",
                },
            ),
            "request": type(
                "Request",
                (),
                {
                    "metric_keys": (
                        "revenue",
                        "gross_profit_margin",
                    ),
                },
            )(),
            "requested_period_count": 2,
            "available_period_count": 2,
            "period_type": "annual",
                "complete_history": True,
        },
    )()

    mock_scope.return_value = scope

    evidence_2024 = type(
        "Evidence",
        (),
        {
            "facts": [
                make_reported_fact(
                    metric_key="revenue",
                    value="100",
                    document_id="doc-2024",
                    document_name="report-2024.pdf",
                    page=1,
                    company_name="Example Plc",
                    ticker="ABC",
                    exchange="NGX",
                    currency="NGN",
                    fiscal_year=2024,
                    period_basis="annual",
                    evidence_text="Revenue 100",
                ),
            ],
        },
    )()

    evidence_2025 = type(
        "Evidence",
        (),
        {
            "facts": [
                make_reported_fact(
                    metric_key="revenue",
                    value="200",
                    document_id="doc-2025",
                    document_name="report-2025.pdf",
                    page=1,
                    company_name="Example Plc",
                    ticker="ABC",
                    exchange="NGX",
                    currency="NGN",
                    fiscal_year=2025,
                    period_basis="annual",
                    evidence_text="Revenue 200",
                ),
            ],
        },
    )()

    mock_extract.side_effect = (
        evidence_2024,
        evidence_2025,
    )

    execution = type(
        "Execution",
        (),
        {
            "company": type(
                "CompanyAnalysis",
                (),
                {
                    "values": ("analysis-value",),
                },
            )(),
            "missing": (),
        },
    )()

    mock_execute.return_value = execution

    presentation = type(
        "Presentation",
        (),
        {
            "text": "FINANCIAL TABLE",
        },
    )()

    mock_present.return_value = presentation

    result = answer_financial_analysis_question(
        "Show ABC revenue and gross profit "
        "margin for the last 2 years.",
        request_id="request-123",
    )

    assert result is not None

    assert result["answer"] == "FINANCIAL TABLE"
    assert result["request_id"] == "request-123"
    assert result["analysis_type"] == "financial"
    assert result["complete_history"] is True

    assert result["requested_period_count"] == 2
    assert result["available_period_count"] == 2

    assert len(result["sources"]) == 2

    mock_extract.assert_any_call(
        document_id="doc-2024",
        company_name="Example Plc",
        ticker="ABC",
        exchange="NGX",
        currency="NGN",
    )

    mock_extract.assert_any_call(
        document_id="doc-2025",
        company_name="Example Plc",
        ticker="ABC",
        exchange="NGX",
        currency="NGN",
    )

    mock_execute.assert_called_once()

    execute_kwargs = (
        mock_execute.call_args.kwargs
    )

    assert (
        execute_kwargs["company_name"]
        == "Example Plc"
    )
    assert execute_kwargs["ticker"] == "ABC"
    assert execute_kwargs["metric_keys"] == (
        "revenue",
        "gross_profit_margin",
    )

    selected_facts = execute_kwargs["facts"]

    assert len(selected_facts) == 2

    assert [
        fact.fiscal_year
        for fact in selected_facts
    ] == [
        2024,
        2025,
    ]

    assert all(
        fact.period_basis == "annual"
        for fact in selected_facts
    )

    mock_present.assert_called_once()

    presented_companies = (
        mock_present.call_args.args[0]
    )

    assert len(presented_companies) == 1
    assert (
        presented_companies[0]
        is execution.company
    )


@patch(
    "app.rag.financial_analysis_service."
    "build_financial_presentation"
)
@patch(
    "app.rag.financial_analysis_service."
    "execute_financial_analysis"
)
@patch(
    "app.rag.financial_analysis_service."
    "extract_financial_facts_from_document"
)
@patch(
    "app.rag.financial_analysis_service."
    "resolve_financial_analysis_scope"
)
def test_incomplete_history_is_disclosed(
    mock_scope,
    mock_extract,
    mock_execute,
    mock_present,
):
    scope = type(
        "Scope",
        (),
        {
            "company": {
                "company_name": "Example Plc",
                "ticker": "ABC",
                "exchange": "NGX",
                "currency": "NGN",
            },
            "documents": (
                {
                    "document_id": "doc-2025",
                },
            ),
            "request": type(
                "Request",
                (),
                {
                    "metric_keys": (
                        "revenue",
                    ),
                },
            )(),
            "requested_period_count": 5,
            "available_period_count": 1,
            "period_type": "annual",
                "complete_history": False,
        },
    )()

    mock_scope.return_value = scope

    mock_extract.return_value = type(
        "Evidence",
        (),
        {
            "facts": [
                make_reported_fact(
                    metric_key="revenue",
                    value="200",
                    document_id="doc-2025",
                    document_name="report-2025.pdf",
                    page=1,
                    company_name="Example Plc",
                    ticker="ABC",
                    exchange="NGX",
                    currency="NGN",
                    fiscal_year=2025,
                    period_basis="annual",
                    evidence_text="Revenue 200",
                ),
            ],
        },
    )()

    mock_execute.return_value = type(
        "Execution",
        (),
        {
            "company": type(
                "CompanyAnalysis",
                (),
                {
                    "values": ("analysis-value",),
                },
            )(),
            "missing": (),
        },
    )()

    mock_present.return_value = type(
        "Presentation",
        (),
        {
            "text": "TABLE",
        },
    )()

    result = answer_financial_analysis_question(
        "Show ABC revenue for the last 5 years."
    )

    assert result["complete_history"] is False
    assert result["requested_period_count"] == 5
    assert result["available_period_count"] == 1

    assert "1 of 5 requested periods" in result["answer"]


@patch(
    "app.rag.financial_analysis_service."
    "resolve_financial_analysis_scope"
)
def test_company_resolution_error_is_preserved(
    mock_scope,
):
    mock_scope.side_effect = (
        FinancialAnalysisCompanyError(
            "Company required"
        )
    )

    with pytest.raises(
        FinancialAnalysisCompanyError
    ):
        answer_financial_analysis_question(
            "Show revenue for the last 5 years."
        )


@patch(
    "app.rag.financial_analysis_service."
    "resolve_financial_analysis_scope"
)
def test_document_resolution_error_is_preserved(
    mock_scope,
):
    mock_scope.side_effect = (
        FinancialAnalysisDocumentsError(
            "No reports"
        )
    )

    with pytest.raises(
        FinancialAnalysisDocumentsError
    ):
        answer_financial_analysis_question(
            "Show ABC revenue for the last 5 years."
        )
