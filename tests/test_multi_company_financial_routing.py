from unittest.mock import patch

from app.rag.financial_analysis_scope import (
    resolve_financial_analysis_scope,
)


def test_fy2025_multi_company_comparison_skips_single_company_resolution():
    question = (
        "Analyse and compare the FY2025 profit before tax reported by "
        "Barclays Bank Plc, HSBC Holdings plc, and Lloyds Bank using "
        "their uploaded annual financial reports. For each company, "
        "provide the exact profit before tax, reporting currency, "
        "financial year, source document and page number."
    )

    with (
        patch(
            "app.rag.financial_analysis_scope.list_registered_companies"
        ) as list_companies,
        patch(
            "app.rag.financial_analysis_scope.resolve_company_from_question"
        ) as resolve_company,
    ):
        result = resolve_financial_analysis_scope(question)

    assert result is None
    list_companies.assert_not_called()
    resolve_company.assert_not_called()
