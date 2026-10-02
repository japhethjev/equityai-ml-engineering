from unittest.mock import patch

import pytest

from app.rag.financial_evidence_bridge import (
    extract_financial_facts_from_document,
)


DOCUMENT_ID = (
    "11111111-1111-1111-1111-111111111111"
)


@patch(
    "app.rag.financial_evidence_bridge.get_document_chunks"
)
def test_extracts_multi_period_financial_facts(
    mock_get_chunks,
):
    mock_get_chunks.return_value = [
        {
            "document_id": DOCUMENT_ID,
            "document": "report.pdf",
            "page": 1,
            "chunk": 0,
            "content": (
                "Financial Performance\n"
                "NGN billion\n"
                "2024 2025 H1 2026\n"
                "Revenue 9,381 18,738 19,135\n"
                "Gross profit (888) 348 3,433\n"
            ),
        },
        {
            "document_id": DOCUMENT_ID,
            "document": "report.pdf",
            "page": 1,
            "chunk": 1,
            "content": (
                "2024 2025 H1 2026\n"
                "Operating profit (941) 223 3,254\n"
                "Profit after tax (2,233) (723) 2,504"
            ),
        },
    ]

    result = (
        extract_financial_facts_from_document(
            document_id=DOCUMENT_ID,
            company_name=(
                "Dangote Petroleum Refinery FZE"
            ),
            currency="NGN",
            unit="billion",
        )
    )

    assert result.document_name == "report.pdf"
    assert result.pages_examined == (1,)

    revenue = [
        fact
        for fact in result.facts
        if fact.metric_key == "revenue"
    ]

    assert len(revenue) == 3

    assert revenue[0].value == 9381
    assert revenue[0].fiscal_year == 2024
    assert revenue[0].period_basis == "annual"

    assert revenue[1].value == 18738
    assert revenue[1].fiscal_year == 2025
    assert revenue[1].period_basis == "annual"

    assert revenue[2].value == 19135
    assert revenue[2].fiscal_year == 2026
    assert revenue[2].fiscal_half == 1
    assert revenue[2].period_basis == "half_year"

    assert revenue[0].document_id == DOCUMENT_ID
    assert revenue[0].page == 1
    assert revenue[0].currency == "NGN"
    assert revenue[0].unit == "billion"


@patch(
    "app.rag.financial_evidence_bridge.get_document_chunks"
)
def test_extracts_negative_accounting_values(
    mock_get_chunks,
):
    mock_get_chunks.return_value = [
        {
            "document_id": DOCUMENT_ID,
            "document": "report.pdf",
            "page": 2,
            "chunk": 0,
            "content": (
                "2024 2025 H1 2026\n"
                "Profit after tax "
                "(2,233) (723) 2,504"
            ),
        }
    ]

    result = (
        extract_financial_facts_from_document(
            document_id=DOCUMENT_ID
        )
    )

    values = [
        fact.value
        for fact in result.facts
        if fact.metric_key
        == "profit_after_tax"
    ]

    assert values == [
        -2233,
        -723,
        2504,
    ]


@patch(
    "app.rag.financial_evidence_bridge.get_document_chunks"
)
def test_page_without_period_header_is_skipped(
    mock_get_chunks,
):
    mock_get_chunks.return_value = [
        {
            "document_id": DOCUMENT_ID,
            "document": "report.pdf",
            "page": 1,
            "chunk": 0,
            "content": (
                "Company overview\n"
                "Revenue grew strongly during the year."
            ),
        }
    ]

    result = (
        extract_financial_facts_from_document(
            document_id=DOCUMENT_ID
        )
    )

    assert result.facts == ()
    assert result.pages_examined == (1,)


@patch(
    "app.rag.financial_evidence_bridge.get_document_chunks"
)
def test_rows_before_header_are_not_extracted(
    mock_get_chunks,
):
    mock_get_chunks.return_value = [
        {
            "document_id": DOCUMENT_ID,
            "document": "report.pdf",
            "page": 1,
            "chunk": 0,
            "content": (
                "Revenue 999 999\n"
                "FY2024 FY2025\n"
                "Revenue 100 120"
            ),
        }
    ]

    result = (
        extract_financial_facts_from_document(
            document_id=DOCUMENT_ID
        )
    )

    revenue = [
        fact.value
        for fact in result.facts
        if fact.metric_key == "revenue"
    ]

    assert revenue == [100, 120]


@patch(
    "app.rag.financial_evidence_bridge.get_document_chunks"
)
def test_value_count_must_match_period_count(
    mock_get_chunks,
):
    mock_get_chunks.return_value = [
        {
            "document_id": DOCUMENT_ID,
            "document": "report.pdf",
            "page": 1,
            "chunk": 0,
            "content": (
                "FY2023 FY2024 FY2025\n"
                "Revenue 100 120"
            ),
        }
    ]

    result = (
        extract_financial_facts_from_document(
            document_id=DOCUMENT_ID
        )
    )

    assert result.facts == ()


@patch(
    "app.rag.financial_evidence_bridge.get_document_chunks"
)
def test_requires_chunks(
    mock_get_chunks,
):
    mock_get_chunks.return_value = []

    with pytest.raises(
        ValueError,
        match="No chunks found",
    ):
        extract_financial_facts_from_document(
            document_id=DOCUMENT_ID
        )


def test_requires_document_id():
    with pytest.raises(
        ValueError,
        match="document_id is required",
    ):
        extract_financial_facts_from_document(
            document_id=" "
        )


@patch(
    "app.rag.financial_evidence_bridge.get_document_chunks"
)
def test_rejects_inconsistent_document_names(
    mock_get_chunks,
):
    mock_get_chunks.return_value = [
        {
            "document_id": DOCUMENT_ID,
            "document": "one.pdf",
            "page": 1,
            "chunk": 0,
            "content": "FY2024 FY2025",
        },
        {
            "document_id": DOCUMENT_ID,
            "document": "two.pdf",
            "page": 2,
            "chunk": 0,
            "content": "FY2024 FY2025",
        },
    ]

    with pytest.raises(
        ValueError,
        match="inconsistent document names",
    ):
        extract_financial_facts_from_document(
            document_id=DOCUMENT_ID
        )


@patch(
    "app.rag.financial_evidence_bridge.get_document_chunks"
)
def test_narrative_years_before_table_do_not_override_header(
    mock_get_chunks,
):
    """
    Narrative references to years must not be mistaken for the
    authoritative financial-table period header.
    """

    mock_get_chunks.return_value = [
        {
            "document_id": DOCUMENT_ID,
            "document": "report.pdf",
            "page": 1,
            "chunk": 0,
            "content": (
                "The company recorded losses in "
                "2024 and 2025 during commissioning.\n"
                "NGN BILLION 2024 2025 H1 2026\n"
                "Revenue 9,381 18,738 19,135\n"
                "Gross profit (888) 348 3,433\n"
                "Operating profit (941) 223 3,254\n"
                "Profit after tax (2,233) (723) 2,504"
            ),
        }
    ]

    result = (
        extract_financial_facts_from_document(
            document_id=DOCUMENT_ID,
            currency="NGN",
            unit="billion",
        )
    )

    revenue = [
        fact
        for fact in result.facts
        if fact.metric_key == "revenue"
    ]

    assert len(revenue) == 3

    assert [
        fact.value
        for fact in revenue
    ] == [
        9381,
        18738,
        19135,
    ]

    assert revenue[0].fiscal_year == 2024
    assert revenue[0].period_basis == "annual"

    assert revenue[1].fiscal_year == 2025
    assert revenue[1].period_basis == "annual"

    assert revenue[2].fiscal_year == 2026
    assert revenue[2].fiscal_half == 1
    assert revenue[2].period_basis == "half_year"
