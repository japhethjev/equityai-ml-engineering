from decimal import Decimal

from app.rag.financial_fact_extractor import (
    extract_reported_facts_from_lines,
    extract_values_from_metric_line,
    parse_financial_number,
)


DOCUMENT_ID = (
    "11111111-1111-1111-1111-111111111111"
)


def test_parse_positive_financial_number():
    assert (
        parse_financial_number("18,738")
        == Decimal("18738")
    )


def test_parse_accounting_negative():
    assert (
        parse_financial_number("(2,233)")
        == Decimal("-2233")
    )


def test_extract_revenue_values():
    result = extract_values_from_metric_line(
        "Revenue 9,381 18,738 19,135"
    )

    assert result == (
        "revenue",
        [
            Decimal("9381"),
            Decimal("18738"),
            Decimal("19135"),
        ],
    )


def test_extract_profit_after_tax_values():
    result = extract_values_from_metric_line(
        "Profit after tax (2,233) (723) 2,504"
    )

    assert result == (
        "profit_after_tax",
        [
            Decimal("-2233"),
            Decimal("-723"),
            Decimal("2504"),
        ],
    )


def test_unknown_metric_is_ignored():
    assert (
        extract_values_from_metric_line(
            "Random measure 100 200"
        )
        is None
    )


def test_extract_facts_preserves_periods():
    periods = [
        {
            "fiscal_year": 2024,
            "period_basis": "annual",
        },
        {
            "fiscal_year": 2025,
            "period_basis": "annual",
        },
        {
            "fiscal_year": 2026,
            "fiscal_half": 1,
            "period_basis": "half_year",
        },
    ]

    facts = extract_reported_facts_from_lines(
        lines=[
            "Revenue 9,381 18,738 19,135",
            "Gross profit (888) 348 3,433",
        ],
        periods=periods,
        document_id=DOCUMENT_ID,
        document_name="report.pdf",
        page=1,
        company_name="Example Plc",
        ticker="ABC",
        exchange="NGX",
        currency="NGN",
        unit="billion",
    )

    assert len(facts) == 6

    revenue = [
        fact
        for fact in facts
        if fact.metric_key == "revenue"
    ]

    assert revenue[0].value == Decimal("9381")
    assert revenue[0].fiscal_year == 2024
    assert revenue[0].period_basis == "annual"

    assert revenue[1].value == Decimal("18738")
    assert revenue[1].fiscal_year == 2025

    assert revenue[2].value == Decimal("19135")
    assert revenue[2].fiscal_year == 2026
    assert revenue[2].fiscal_half == 1
    assert revenue[2].period_basis == "half_year"


def test_mismatched_column_count_is_rejected():
    facts = extract_reported_facts_from_lines(
        lines=[
            "Revenue 100 200",
        ],
        periods=[
            {
                "fiscal_year": 2023,
                "period_basis": "annual",
            },
            {
                "fiscal_year": 2024,
                "period_basis": "annual",
            },
            {
                "fiscal_year": 2025,
                "period_basis": "annual",
            },
        ],
        document_id=DOCUMENT_ID,
        document_name="report.pdf",
        page=1,
    )

    assert facts == []


def test_metric_section_heading_is_not_financial_row():
    """
    Digits embedded in a section heading must not be interpreted
    as financial values.
    """

    result = extract_values_from_metric_line(
        "REVENUE BY PRODUCT — H1 2026"
    )

    assert result is None


def test_period_before_metric_is_not_financial_row():
    """
    A period heading containing a metric name is not a metric row.
    """

    result = extract_values_from_metric_line(
        "H1 2026 REVENUE"
    )

    assert result is None


def test_metric_narrative_with_numbers_is_not_financial_row():
    """
    Narrative text following a metric label must not be treated
    as a table row merely because it contains numbers.
    """

    result = extract_values_from_metric_line(
        "Revenue grew 38% in H1 2026"
    )

    assert result is None


def test_metric_row_can_begin_with_accounting_negative():
    result = extract_values_from_metric_line(
        "Revenue (500) 750 1,200"
    )

    assert result == (
        "revenue",
        [
            Decimal("-500"),
            Decimal("750"),
            Decimal("1200"),
        ],
    )


def test_metric_row_can_begin_with_explicit_sign():
    result = extract_values_from_metric_line(
        "Revenue +100 -20 300"
    )

    assert result == (
        "revenue",
        [
            Decimal("100"),
            Decimal("-20"),
            Decimal("300"),
        ],
    )
