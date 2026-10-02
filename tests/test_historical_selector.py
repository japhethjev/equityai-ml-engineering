import pytest

from app.rag.historical_selector import (
    parse_historical_selection,
)


def test_last_five_years():
    result = parse_historical_selection(
        "Show gross profit margin for the last 5 years."
    )

    assert result is not None
    assert result.count == 5
    assert result.period_type == "annual"


def test_past_three_fiscal_years():
    result = parse_historical_selection(
        "Debt to equity for the past 3 fiscal years"
    )

    assert result is not None
    assert result.count == 3
    assert result.period_type == "annual"


def test_previous_five_years():
    result = parse_historical_selection(
        "Operating cash flow for the previous 5 years"
    )

    assert result is not None
    assert result.count == 5
    assert result.period_type == "annual"


def test_prior_three_years():
    result = parse_historical_selection(
        "Net margin for the prior 3 years"
    )

    assert result is not None
    assert result.count == 3
    assert result.period_type == "annual"


def test_last_eight_quarters():
    result = parse_historical_selection(
        "Gross margin over the last 8 quarters"
    )

    assert result is not None
    assert result.count == 8
    assert result.period_type == "quarterly"


def test_previous_four_quarters():
    result = parse_historical_selection(
        "Liquidity ratios over the previous 4 quarters"
    )

    assert result is not None
    assert result.count == 4
    assert result.period_type == "quarterly"


def test_past_six_quarters():
    result = parse_historical_selection(
        "Show operating performance for the past 6 quarters"
    )

    assert result is not None
    assert result.count == 6
    assert result.period_type == "quarterly"


def test_prior_four_quarters():
    result = parse_historical_selection(
        "Current ratio for the prior 4 quarters"
    )

    assert result is not None
    assert result.count == 4
    assert result.period_type == "quarterly"


def test_explicit_single_period_is_not_historical_window():
    result = parse_historical_selection(
        "What was revenue in FY2025?"
    )

    assert result is None


def test_explicit_quarter_is_not_historical_window():
    result = parse_historical_selection(
        "What was revenue in Q2 2026?"
    )

    assert result is None


def test_comparison_is_not_historical_window():
    result = parse_historical_selection(
        "Compare FY2024 and FY2025 revenue."
    )

    assert result is None


def test_vague_recent_years_not_inferred():
    result = parse_historical_selection(
        "Show recent years of profitability."
    )

    assert result is None


def test_historical_performance_without_count_not_inferred():
    result = parse_historical_selection(
        "Show historical financial performance."
    )

    assert result is None


def test_empty_question_returns_none():
    assert (
        parse_historical_selection("   ")
        is None
    )


def test_excessive_annual_window_rejected():
    with pytest.raises(
        ValueError,
        match="cannot exceed 20",
    ):
        parse_historical_selection(
            "Show revenue for the last 21 years"
        )


def test_excessive_quarterly_window_rejected():
    with pytest.raises(
        ValueError,
        match="cannot exceed 40",
    ):
        parse_historical_selection(
            "Show revenue for the last 41 quarters"
        )
