from decimal import Decimal

import pytest

from app.rag.financial_calculator import (
    calculate_change,
    calculate_metric,
)
from app.rag.financial_facts import (
    make_reported_fact,
)


DOCUMENT_ID = (
    "11111111-1111-1111-1111-111111111111"
)


def fact(
    metric_key,
    value,
    year=2025,
    quarter=None,
    period_basis="annual",
):
    return make_reported_fact(
        metric_key=metric_key,
        value=value,
        document_id=DOCUMENT_ID,
        document_name="report.pdf",
        page=10,
        fiscal_year=year,
        fiscal_quarter=quarter,
        period_basis=period_basis,
        currency="NGN",
    )


def test_gross_profit_margin():
    result = calculate_metric(
        "gross_profit_margin",
        [
            fact("gross_profit", "40"),
            fact("revenue", "100"),
        ],
    )

    assert result.value == Decimal("40.00")
    assert result.unit == "%"
    assert result.source == "derived"


def test_operating_profit_margin():
    result = calculate_metric(
        "operating_profit_margin",
        [
            fact("operating_profit", "25"),
            fact("revenue", "100"),
        ],
    )

    assert result.value == Decimal("25.00")


def test_net_profit_margin():
    result = calculate_metric(
        "net_profit_margin",
        [
            fact("profit_after_tax", "15"),
            fact("revenue", "120"),
        ],
    )

    assert result.value == Decimal("12.50")


def test_current_ratio():
    result = calculate_metric(
        "current_ratio",
        [
            fact("current_assets", "300"),
            fact("current_liabilities", "150"),
        ],
    )

    assert result.value == Decimal("2.00")
    assert result.unit == "x"


def test_quick_ratio():
    result = calculate_metric(
        "quick_ratio",
        [
            fact("current_assets", "300"),
            fact("inventory", "60"),
            fact("current_liabilities", "120"),
        ],
    )

    assert result.value == Decimal("2.00")


def test_cash_ratio():
    result = calculate_metric(
        "cash_ratio",
        [
            fact("cash", "50"),
            fact("current_liabilities", "100"),
        ],
    )

    assert result.value == Decimal("0.50")


def test_debt_to_equity():
    result = calculate_metric(
        "debt_to_equity",
        [
            fact("total_debt", "200"),
            fact("total_equity", "400"),
        ],
    )

    assert result.value == Decimal("0.50")


def test_debt_to_assets():
    result = calculate_metric(
        "debt_to_assets",
        [
            fact("total_debt", "200"),
            fact("total_assets", "800"),
        ],
    )

    assert result.value == Decimal("0.25")


def test_free_cash_flow():
    result = calculate_metric(
        "free_cash_flow",
        [
            fact("operating_cash_flow", "500"),
            fact("capital_expenditure", "125"),
        ],
    )

    assert result.value == Decimal("375")


def test_operating_cash_flow_margin():
    result = calculate_metric(
        "operating_cash_flow_margin",
        [
            fact("operating_cash_flow", "30"),
            fact("revenue", "120"),
        ],
    )

    assert result.value == Decimal("25.00")


def test_missing_required_fact_rejected():
    with pytest.raises(
        ValueError,
        match="Missing required financial facts",
    ):
        calculate_metric(
            "gross_profit_margin",
            [
                fact(
                    "gross_profit",
                    "40",
                )
            ],
        )


def test_mixed_reporting_periods_rejected():
    with pytest.raises(
        ValueError,
        match="same reporting period",
    ):
        calculate_metric(
            "gross_profit_margin",
            [
                fact(
                    "gross_profit",
                    "40",
                    year=2025,
                ),
                fact(
                    "revenue",
                    "100",
                    year=2024,
                ),
            ],
        )


def test_zero_denominator_rejected():
    with pytest.raises(
        ZeroDivisionError,
        match="denominator cannot be zero",
    ):
        calculate_metric(
            "current_ratio",
            [
                fact(
                    "current_assets",
                    "100",
                ),
                fact(
                    "current_liabilities",
                    "0",
                ),
            ],
        )


def test_reported_metric_cannot_be_calculated():
    with pytest.raises(
        ValueError,
        match="does not require calculation",
    ):
        calculate_metric(
            "revenue",
            [
                fact(
                    "revenue",
                    "100",
                )
            ],
        )


def test_percentage_change():
    result = calculate_change(
        fact(
            "revenue",
            "100",
            year=2024,
        ),
        fact(
            "revenue",
            "125",
            year=2025,
        ),
    )

    assert result.absolute_change == Decimal("25")
    assert result.percentage_change == Decimal("25.00")


def test_percentage_decrease():
    result = calculate_change(
        fact(
            "revenue",
            "100",
            year=2024,
        ),
        fact(
            "revenue",
            "80",
            year=2025,
        ),
    )

    assert result.absolute_change == Decimal("-20")
    assert result.percentage_change == Decimal("-20.00")


def test_change_from_negative_prior_value():
    result = calculate_change(
        fact(
            "profit_after_tax",
            "-100",
            year=2024,
        ),
        fact(
            "profit_after_tax",
            "-50",
            year=2025,
        ),
    )

    assert result.absolute_change == Decimal("50")
    assert result.percentage_change == Decimal("50.00")


def test_change_from_zero_has_no_percentage():
    result = calculate_change(
        fact(
            "revenue",
            "0",
            year=2024,
        ),
        fact(
            "revenue",
            "100",
            year=2025,
        ),
    )

    assert result.absolute_change == Decimal("100")
    assert result.percentage_change is None


def test_change_requires_same_metric():
    with pytest.raises(
        ValueError,
        match="same metric",
    ):
        calculate_change(
            fact(
                "revenue",
                "100",
                year=2024,
            ),
            fact(
                "gross_profit",
                "120",
                year=2025,
            ),
        )


def test_calculation_retains_input_provenance():
    gross_profit = fact(
        "gross_profit",
        "40",
    )

    revenue = fact(
        "revenue",
        "100",
    )

    result = calculate_metric(
        "gross_profit_margin",
        [
            gross_profit,
            revenue,
        ],
    )

    assert result.input_facts == (
        gross_profit,
        revenue,
    )

    assert (
        result.input_facts[0].document_id
        == DOCUMENT_ID
    )
