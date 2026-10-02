from decimal import Decimal

import pytest

from app.rag.financial_analysis_executor import (
    execute_financial_analysis,
)
from app.rag.financial_facts import (
    make_reported_fact,
)


def fact(
    metric_key,
    value,
    *,
    year,
    quarter=None,
    period_basis="annual",
    page=1,
):
    return make_reported_fact(
        metric_key=metric_key,
        value=value,
        document_id=f"doc-{year}-{quarter or 0}",
        document_name=f"report-{year}.pdf",
        page=page,
        company_name="Example Plc",
        ticker="ABC",
        exchange="TEST",
        currency="GBP",
        fiscal_year=year,
        fiscal_quarter=quarter,
        period_basis=period_basis,
    )


def test_reported_metric_across_three_years():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=("revenue",),
        facts=[
            fact("revenue", "100", year=2023),
            fact("revenue", "120", year=2024),
            fact("revenue", "150", year=2025),
        ],
    )

    values = result.company.values

    assert len(values) == 3

    assert tuple(
        value.period_label
        for value in values
    ) == (
        "FY2023",
        "FY2024",
        "FY2025",
    )

    assert tuple(
        value.value
        for value in values
    ) == (
        Decimal("100"),
        Decimal("120"),
        Decimal("150"),
    )

    assert result.missing == ()


def test_net_profit_margin_is_derived():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=(
            "net_profit_margin",
        ),
        facts=[
            fact(
                "profit_after_tax",
                "20",
                year=2025,
            ),
            fact(
                "revenue",
                "100",
                year=2025,
            ),
        ],
    )

    value = result.company.values[0]

    assert (
        value.metric_key
        == "net_profit_margin"
    )

    assert value.value == Decimal("20.00")
    assert value.unit == "%"
    assert value.source == "derived"

    assert tuple(
        item.metric_key
        for item in value.input_facts
    ) == (
        "profit_after_tax",
        "revenue",
    )


def test_gross_profit_margin_is_derived():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=(
            "gross_profit_margin",
        ),
        facts=[
            fact(
                "gross_profit",
                "40",
                year=2025,
            ),
            fact(
                "revenue",
                "100",
                year=2025,
            ),
        ],
    )

    assert (
        result.company.values[0].value
        == Decimal("40.00")
    )


def test_debt_to_equity_is_derived():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=("debt_to_equity",),
        facts=[
            fact(
                "total_debt",
                "60",
                year=2025,
            ),
            fact(
                "total_equity",
                "120",
                year=2025,
            ),
        ],
    )

    value = result.company.values[0]

    assert (
        value.metric_key
        == "debt_to_equity"
    )

    assert value.value == Decimal("0.50")
    assert value.source == "derived"


def test_missing_derived_input_is_not_calculated():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=(
            "net_profit_margin",
        ),
        facts=[
            fact(
                "profit_after_tax",
                "20",
                year=2025,
            ),
        ],
    )

    assert result.company.values == ()

    assert result.missing == (
        (
            "net_profit_margin",
            "FY2025",
        ),
    )


def test_missing_reported_metric_is_recorded():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=(
            "revenue",
            "gross_profit",
        ),
        facts=[
            fact(
                "revenue",
                "100",
                year=2025,
            ),
        ],
    )

    assert len(
        result.company.values
    ) == 1

    assert result.missing == (
        (
            "gross_profit",
            "FY2025",
        ),
    )


def test_multiple_requested_metrics_are_executed():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=(
            "revenue",
            "gross_profit",
            "gross_profit_margin",
        ),
        facts=[
            fact(
                "revenue",
                "100",
                year=2025,
            ),
            fact(
                "gross_profit",
                "40",
                year=2025,
            ),
        ],
    )

    assert tuple(
        value.metric_key
        for value in result.company.values
    ) == (
        "revenue",
        "gross_profit",
        "gross_profit_margin",
    )


def test_quarterly_periods_are_kept_separate():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=("revenue",),
        facts=[
            fact(
                "revenue",
                "100",
                year=2025,
                quarter=1,
                period_basis="quarterly",
            ),
            fact(
                "revenue",
                "120",
                year=2025,
                quarter=2,
                period_basis="quarterly",
            ),
        ],
    )

    assert tuple(
        value.period_label
        for value in result.company.values
    ) == (
        "Q1 FY2025",
        "Q2 FY2025",
    )


def test_derived_value_preserves_all_input_provenance():
    result = execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=(
            "current_ratio",
        ),
        facts=[
            fact(
                "current_assets",
                "200",
                year=2025,
                page=30,
            ),
            fact(
                "current_liabilities",
                "100",
                year=2025,
                page=31,
            ),
        ],
    )

    value = result.company.values[0]

    assert len(value.input_facts) == 2

    assert tuple(
        item.page
        for item in value.input_facts
    ) == (
        30,
        31,
    )


def test_duplicate_same_period_metric_is_rejected():
    with pytest.raises(
        ValueError,
        match="Multiple reported facts",
    ):
        execute_financial_analysis(
            company_name="Example Plc",
            ticker="ABC",
            metric_keys=("revenue",),
            facts=[
                fact(
                    "revenue",
                    "100",
                    year=2025,
                    page=1,
                ),
                fact(
                    "revenue",
                    "101",
                    year=2025,
                    page=2,
                ),
            ],
        )
