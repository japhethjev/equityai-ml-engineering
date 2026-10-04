from app.rag.report_selector import (
    parse_report_selection,
    parse_report_selections,
)


def test_parse_fiscal_year():
    result = parse_report_selection(
        "What was revenue in FY2025?"
    )

    assert result.fiscal_year == 2025


def test_parse_annual_report():
    result = parse_report_selection(
        "What was revenue in the annual report 2025?"
    )

    assert result.report_type == "annual"
    assert result.fiscal_year == 2025
    assert result.fiscal_quarter is None


def test_parse_full_year():
    result = parse_report_selection(
        "Show full year 2025 profit."
    )

    assert result.report_type == "annual"
    assert result.fiscal_year == 2025


def test_parse_q2():
    result = parse_report_selection(
        "What was revenue in Q2 2026?"
    )

    assert result.report_type == "quarterly"
    assert result.fiscal_year == 2026
    assert result.fiscal_quarter == 2


def test_parse_q4_fiscal_year():
    result = parse_report_selection(
        "What was EBITDA in Q4 FY2025?"
    )

    assert result.report_type == "quarterly"
    assert result.fiscal_year == 2025
    assert result.fiscal_quarter == 4


def test_parse_quarter_word():
    result = parse_report_selection(
        "Revenue for quarter 3 2025"
    )

    assert result.report_type == "quarterly"
    assert result.fiscal_quarter == 3
    assert result.fiscal_year == 2025


def test_parse_h1():
    result = parse_report_selection(
        "What was profit in H1 2026?"
    )

    assert result.report_type == "half_year"
    assert result.fiscal_half == 1
    assert result.fiscal_year == 2026


def test_parse_h2_fiscal_year():
    result = parse_report_selection(
        "Show H2 FY2025 results"
    )

    assert result.report_type == "half_year"
    assert result.fiscal_half == 2
    assert result.fiscal_year == 2025


def test_parse_latest_results():
    result = parse_report_selection(
        "What are the latest results?"
    )

    assert result.latest is True


def test_no_period_returns_unrestricted_selection():
    result = parse_report_selection(
        "What is the company's debt?"
    )

    assert result.report_type is None
    assert result.fiscal_year is None
    assert result.fiscal_quarter is None
    assert result.fiscal_half is None
    assert result.latest is False


def test_compare_q2_across_two_years():
    results = parse_report_selections(
        "Compare Q2 2025 with Q2 2026"
    )

    assert len(results) == 2

    assert results[0].report_type == "quarterly"
    assert results[0].fiscal_quarter == 2
    assert results[0].fiscal_year == 2025

    assert results[1].report_type == "quarterly"
    assert results[1].fiscal_quarter == 2
    assert results[1].fiscal_year == 2026


def test_compare_q2_shorthand_across_years():
    results = parse_report_selections(
        "Compare Q2 2025 and 2026"
    )

    assert len(results) == 2

    assert results[0].fiscal_year == 2025
    assert results[1].fiscal_year == 2026

    assert all(
        result.report_type == "quarterly"
        and result.fiscal_quarter == 2
        for result in results
    )


def test_compare_two_fiscal_years():
    results = parse_report_selections(
        "Compare FY2024 and FY2025"
    )

    assert len(results) == 2

    assert results[0].report_type == "annual"
    assert results[0].fiscal_year == 2024

    assert results[1].report_type == "annual"
    assert results[1].fiscal_year == 2025


def test_compare_h1_across_two_years():
    results = parse_report_selections(
        "Compare H1 2025 with H1 2026"
    )

    assert len(results) == 2

    assert results[0].report_type == "half_year"
    assert results[0].fiscal_half == 1
    assert results[0].fiscal_year == 2025

    assert results[1].report_type == "half_year"
    assert results[1].fiscal_half == 1
    assert results[1].fiscal_year == 2026


def test_single_period_still_returns_one_selection():
    results = parse_report_selections(
        "What was revenue in Q3 2025?"
    )

    assert len(results) == 1
    assert results[0].report_type == "quarterly"
    assert results[0].fiscal_quarter == 3
    assert results[0].fiscal_year == 2025


def test_latest_report_has_no_type_constraint():
    result = parse_report_selection(
        "What is AAPL's latest report?"
    )

    assert result.latest is True
    assert result.report_type is None


def test_latest_annual_report():
    result = parse_report_selection(
        "What is AAPL's latest annual report?"
    )

    assert result.latest is True
    assert result.report_type == "annual"


def test_latest_quarterly_report():
    result = parse_report_selection(
        "What is AAPL's latest quarterly report?"
    )

    assert result.latest is True
    assert result.report_type == "quarterly"


def test_latest_half_year_report():
    result = parse_report_selection(
        "What is Barclays' latest half-year report?"
    )

    assert result.latest is True
    assert result.report_type == "half_year"


def test_annual_comparison_deduplicates_repeated_year_mentions():
    results = parse_report_selections(
        "According to HSBC Holdings plc Annual Report and Accounts "
        "2025, what was profit before tax in 2025, and how did it "
        "compare with 2024?"
    )

    assert len(results) == 2

    assert results[0].report_type == "annual"
    assert results[0].fiscal_year == 2025

    assert results[1].report_type == "annual"
    assert results[1].fiscal_year == 2024
