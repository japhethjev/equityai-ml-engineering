import pytest

from app.rag.rag_service import (
    requests_broad_financial_analysis,
)


@pytest.mark.parametrize(
    "question",
    [
        "Analyse Dangote Cement's financial performance for FY2021.",
        "Analyze the financial health of Barclays.",
        "Discuss the major risks facing Dangote Cement.",
        "Provide a financial review of HSBC.",
        "Prepare an investment analysis of Lloyds.",
    ],
)
def test_broad_analysis_requests(question):
    assert requests_broad_financial_analysis(question)


@pytest.mark.parametrize(
    "question",
    [
        "What was Dangote Cement's revenue in FY2021?",
        "State HSBC's profit before tax for 2025.",
        "What was Barclays' dividend per share?",
        "Calculate the operating profit margin.",
    ],
)
def test_exact_metric_requests_not_classified_as_broad(question):
    assert not requests_broad_financial_analysis(question)
