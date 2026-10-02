from decimal import Decimal

import pytest

from app.rag.financial_facts import (
    FinancialFact,
    make_reported_fact,
    period_label,
    same_reporting_period,
)


DOCUMENT_ID = (
    "11111111-1111-1111-1111-111111111111"
)


def make_fact(
    metric_key="revenue",
    value="1000000",
    fiscal_year=2025,
    fiscal_quarter=None,
    fiscal_half=None,
    period_basis="annual",
):
    return make_reported_fact(
        metric_key=metric_key,
        value=value,
        document_id=DOCUMENT_ID,
        document_name="report.pdf",
        page=42,
        company_name="Example Plc",
        ticker="ABC",
        exchange="NGX",
        currency="NGN",
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        fiscal_half=fiscal_half,
        period_basis=period_basis,
        evidence_text=(
            "Revenue was NGN 1,000,000."
        ),
    )


def test_reported_fact_is_created():
    fact = make_fact()

    assert isinstance(
        fact,
        FinancialFact,
    )
    assert fact.metric_key == "revenue"
    assert fact.value == Decimal("1000000")
    assert fact.source == "reported"


def test_fact_retains_document_provenance():
    fact = make_fact()

    assert fact.document_id == DOCUMENT_ID
    assert fact.document_name == "report.pdf"
    assert fact.page == 42
    assert fact.evidence_text is not None


def test_integer_value_becomes_decimal():
    fact = make_fact(
        value=500,
    )

    assert fact.value == Decimal("500")


def test_float_value_uses_safe_decimal_conversion():
    fact = make_fact(
        value=12.5,
    )

    assert fact.value == Decimal("12.5")


def test_string_decimal_is_preserved():
    fact = make_fact(
        value="123.4500",
    )

    assert fact.value == Decimal(
        "123.4500"
    )


def test_invalid_numeric_value_is_rejected():
    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        make_fact(
            value="not-a-number",
        )


def test_boolean_value_is_rejected():
    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        make_fact(
            value=True,
        )


def test_empty_metric_key_is_rejected():
    with pytest.raises(
        ValueError,
        match="metric_key is required",
    ):
        make_fact(
            metric_key="   ",
        )


def test_empty_document_id_is_rejected():
    with pytest.raises(
        ValueError,
        match="document_id is required",
    ):
        make_reported_fact(
            metric_key="revenue",
            value="100",
            document_id="   ",
            document_name="report.pdf",
            page=1,
        )


def test_invalid_page_is_rejected():
    with pytest.raises(
        ValueError,
        match="page must be greater than zero",
    ):
        make_reported_fact(
            metric_key="revenue",
            value="100",
            document_id=DOCUMENT_ID,
            document_name="report.pdf",
            page=0,
        )


def test_annual_period_label():
    fact = make_fact(
        fiscal_year=2025,
        period_basis="annual",
    )

    assert period_label(fact) == "FY2025"


def test_quarter_period_label():
    fact = make_fact(
        fiscal_year=2026,
        fiscal_quarter=2,
        period_basis="quarterly",
    )

    assert (
        period_label(fact)
        == "Q2 FY2026"
    )


def test_half_year_period_label():
    fact = make_fact(
        fiscal_year=2025,
        fiscal_half=1,
        period_basis="half_year",
    )

    assert (
        period_label(fact)
        == "H1 FY2025"
    )


def test_same_annual_period():
    left = make_fact(
        metric_key="revenue",
        fiscal_year=2025,
    )

    right = make_fact(
        metric_key="gross_profit",
        fiscal_year=2025,
    )

    assert (
        same_reporting_period(
            left,
            right,
        )
        is True
    )


def test_different_years_are_not_same_period():
    left = make_fact(
        fiscal_year=2024,
    )

    right = make_fact(
        fiscal_year=2025,
    )

    assert (
        same_reporting_period(
            left,
            right,
        )
        is False
    )


def test_quarter_and_annual_are_not_same_period():
    annual = make_fact(
        fiscal_year=2025,
        period_basis="annual",
    )

    quarter = make_fact(
        fiscal_year=2025,
        fiscal_quarter=1,
        period_basis="quarterly",
    )

    assert (
        same_reporting_period(
            annual,
            quarter,
        )
        is False
    )
