from app.rag.financial_analysis_request import (
    extract_metric_keys,
    extract_period_request,
    parse_financial_analysis_request,
)


def test_extract_multiple_income_statement_metrics():
    result = extract_metric_keys(
        "Show revenue, cost of sales, gross profit "
        "and operating expenses."
    )

    assert result == (
        "revenue",
        "cost_of_sales",
        "gross_profit",
        "operating_expenses",
    )


def test_margin_phrase_does_not_become_gross_profit():
    result = extract_metric_keys(
        "Show gross profit margin for the last 5 years."
    )

    assert result == (
        "gross_profit_margin",
    )


def test_net_profit_margin_is_specific_metric():
    result = extract_metric_keys(
        "What is the net profit margin trend?"
    )

    assert result == (
        "net_profit_margin",
    )


def test_liquidity_ratios_expand_to_supported_ratios():
    result = extract_metric_keys(
        "Show liquidity ratios for the last 4 quarters."
    )

    assert result == (
        "current_ratio",
        "quick_ratio",
    )


def test_profit_margins_expand_to_three_margins():
    result = extract_metric_keys(
        "Show profit margins for the last five years."
    )

    assert result == (
        "gross_profit_margin",
        "operating_profit_margin",
        "net_profit_margin",
    )


def test_operating_cash_flow_synonym():
    result = extract_metric_keys(
        "Show cash flow from operations."
    )

    assert result == (
        "operating_cash_flow",
    )


def test_debt_to_equity_synonym():
    result = extract_metric_keys(
        "Show the debt-to-equity ratio."
    )

    assert result == (
        "debt_to_equity",
    )


def test_last_five_years():
    assert extract_period_request(
        "Show revenue for the last 5 years."
    ) == (
        "annual",
        5,
    )


def test_last_five_years_written_as_word():
    assert extract_period_request(
        "Show revenue for the last five years."
    ) == (
        "annual",
        5,
    )


def test_last_eight_quarters():
    assert extract_period_request(
        "Show net margin for the last 8 quarters."
    ) == (
        "quarterly",
        8,
    )


def test_last_four_quarters_written_as_word():
    assert extract_period_request(
        "Show liquidity ratios for the last four quarters."
    ) == (
        "quarterly",
        4,
    )


def test_historical_revenue_request_routes_to_analysis():
    result = parse_financial_analysis_request(
        "Show revenue for the last 5 years."
    )

    assert result.is_financial_analysis is True
    assert result.metric_keys == (
        "revenue",
    )
    assert result.period_type == "annual"
    assert result.period_count == 5
    assert result.historical_requested is True


def test_historical_margin_request_routes_to_analysis():
    result = parse_financial_analysis_request(
        "Show gross profit margin and net profit margin "
        "for the last 5 years."
    )

    assert result.is_financial_analysis is True

    assert result.metric_keys == (
        "gross_profit_margin",
        "net_profit_margin",
    )

    assert result.period_type == "annual"
    assert result.period_count == 5


def test_quarterly_liquidity_request_routes_to_analysis():
    result = parse_financial_analysis_request(
        "Show liquidity ratios for the last 4 quarters."
    )

    assert result.is_financial_analysis is True

    assert result.metric_keys == (
        "current_ratio",
        "quick_ratio",
    )

    assert result.period_type == "quarterly"
    assert result.period_count == 4


def test_debt_equity_three_year_request():
    result = parse_financial_analysis_request(
        "Debt to equity for the last 3 years."
    )

    assert result.is_financial_analysis is True
    assert result.metric_keys == (
        "debt_to_equity",
    )
    assert result.period_type == "annual"
    assert result.period_count == 3


def test_operating_cashflow_five_year_request():
    result = parse_financial_analysis_request(
        "Show operating cash flow for the last 5 years."
    )

    assert result.is_financial_analysis is True
    assert result.metric_keys == (
        "operating_cash_flow",
    )
    assert result.period_count == 5


def test_comparison_request_detected():
    result = parse_financial_analysis_request(
        "Compare revenue, gross profit and net profit "
        "margin across these companies."
    )

    assert result.is_financial_analysis is True
    assert result.comparison_requested is True

    assert result.metric_keys == (
        "revenue",
        "gross_profit",
        "net_profit_margin",
    )


def test_single_simple_fact_remains_ordinary_rag():
    result = parse_financial_analysis_request(
        "What was revenue in 2025?"
    )

    assert result.metric_keys == (
        "revenue",
    )

    assert result.is_financial_analysis is False


def test_unrelated_question_is_not_financial_analysis():
    result = parse_financial_analysis_request(
        "What is the IPO offer price?"
    )

    assert result.is_financial_analysis is False
    assert result.metric_keys == ()


def test_percentage_change_request_is_analysis():
    result = parse_financial_analysis_request(
        "Compare revenue for the last 2 years."
    )

    assert result.is_financial_analysis is True
    assert result.comparison_requested is True
    assert result.period_count == 2
