from decimal import Decimal

from app.rag.financial_analysis import (
    calculate_latest_change,
    derived_analysis_value,
    period_label,
    reported_analysis_value,
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
    half=None,
    period_basis="annual",
):
    return make_reported_fact(
        metric_key=metric_key,
        value=value,
        document_id=DOCUMENT_ID,
        document_name="report.pdf",
        page=10,
        company_name="Example Plc",
        ticker="ABC",
        exchange="NGX",
        fiscal_year=year,
        fiscal_quarter=quarter,
        fiscal_half=half,
        period_basis=period_basis,
        currency="NGN",
        unit="billion",
    )


def test_annual_period_label():
    assert (
        period_label(
            fact(
                "revenue",
                "100",
            )
        )
        == "FY2025"
    )


def test_quarter_period_label():
    assert (
        period_label(
            fact(
                "revenue",
                "100",
                year=2026,
                quarter=2,
                period_basis="quarterly",
            )
        )
        == "Q2 FY2026"
    )


def test_half_year_period_label():
    assert (
        period_label(
            fact(
                "revenue",
                "100",
                year=2026,
                half=1,
                period_basis="half_year",
            )
        )
        == "H1 FY2026"
    )


def test_reported_value_preserves_provenance():
    source = fact(
        "revenue",
        "125",
    )

    result = reported_analysis_value(
        source
    )

    assert result.value == Decimal("125")
    assert result.source == "reported"
    assert result.input_facts == (source,)


def test_derived_margin():
    result = derived_analysis_value(
        "gross_profit_margin",
        [
            fact(
                "gross_profit",
                "40",
            ),
            fact(
                "revenue",
                "100",
            ),
        ],
    )

    assert result.value == Decimal("40.00")
    assert result.unit == "%"
    assert result.source == "derived"


def test_latest_revenue_change_is_percent():
    previous = reported_analysis_value(
        fact(
            "revenue",
            "100",
            year=2024,
        )
    )

    current = reported_analysis_value(
        fact(
            "revenue",
            "125",
            year=2025,
        )
    )

    result = calculate_latest_change(
        previous,
        current,
    )

    assert result.value == Decimal("25.00")
    assert result.display_unit == "%"
    assert result.not_meaningful is False


def test_revenue_decrease_is_negative_percent():
    previous = reported_analysis_value(
        fact(
            "revenue",
            "100",
            year=2024,
        )
    )

    current = reported_analysis_value(
        fact(
            "revenue",
            "80",
            year=2025,
        )
    )

    result = calculate_latest_change(
        previous,
        current,
    )

    assert result.value == Decimal("-20.00")
    assert result.display_unit == "%"
    assert result.not_meaningful is False


def test_margin_change_uses_percentage_points():
    previous = derived_analysis_value(
        "gross_profit_margin",
        [
            fact(
                "gross_profit",
                "35",
                year=2024,
            ),
            fact(
                "revenue",
                "100",
                year=2024,
            ),
        ],
    )

    current = derived_analysis_value(
        "gross_profit_margin",
        [
            fact(
                "gross_profit",
                "42",
                year=2025,
            ),
            fact(
                "revenue",
                "100",
                year=2025,
            ),
        ],
    )

    result = calculate_latest_change(
        previous,
        current,
    )

    assert result.value == Decimal("7.00")
    assert result.display_unit == "pp"
    assert result.not_meaningful is False


def test_loss_to_profit_shows_percentage_improvement():
    previous = reported_analysis_value(
        fact(
            "profit_after_tax",
            "-100",
            year=2024,
        )
    )

    current = reported_analysis_value(
        fact(
            "profit_after_tax",
            "50",
            year=2025,
        )
    )

    result = calculate_latest_change(
        previous,
        current,
    )

    assert result.value == Decimal("150.00")
    assert result.display_unit == "%"
    assert result.not_meaningful is False


def test_profit_to_loss_shows_percentage_deterioration():
    previous = reported_analysis_value(
        fact(
            "profit_after_tax",
            "100",
            year=2024,
        )
    )

    current = reported_analysis_value(
        fact(
            "profit_after_tax",
            "-50",
            year=2025,
        )
    )

    result = calculate_latest_change(
        previous,
        current,
    )

    assert result.value == Decimal("-150.00")
    assert result.display_unit == "%"
    assert result.not_meaningful is False


def test_negative_loss_reduction_shows_improvement():
    previous = reported_analysis_value(
        fact(
            "profit_after_tax",
            "-100",
            year=2024,
        )
    )

    current = reported_analysis_value(
        fact(
            "profit_after_tax",
            "-50",
            year=2025,
        )
    )

    result = calculate_latest_change(
        previous,
        current,
    )

    assert result.value == Decimal("50.00")
    assert result.display_unit == "%"
    assert result.not_meaningful is False


def test_zero_base_has_no_percentage_change():
    previous = reported_analysis_value(
        fact(
            "revenue",
            "0",
            year=2024,
        )
    )

    current = reported_analysis_value(
        fact(
            "revenue",
            "100",
            year=2025,
        )
    )

    result = calculate_latest_change(
        previous,
        current,
    )

    assert result.value is None
    assert result.display_unit is None
    assert result.not_meaningful is True


def test_latest_change_requires_same_metric():
    previous = reported_analysis_value(
        fact(
            "revenue",
            "100",
            year=2024,
        )
    )

    current = reported_analysis_value(
        fact(
            "gross_profit",
            "50",
            year=2025,
        )
    )

    try:
        calculate_latest_change(
            previous,
            current,
        )
    except ValueError as exc:
        assert (
            "same metric"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected mismatched metrics to fail"
        )
