from app.rag.financial_period_parser import (
    find_period_header,
    parse_period_header,
)


def test_parse_bare_annual_years():
    assert parse_period_header(
        "2023 2024 2025"
    ) == [
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
    ]


def test_parse_fy_years():
    assert parse_period_header(
        "FY2023 FY2024 FY2025"
    ) == [
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
    ]


def test_parse_dangote_header():
    assert parse_period_header(
        "2024 2025 H1 2026"
    ) == [
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


def test_parse_quarterly_header():
    assert parse_period_header(
        "Q1 2026 Q2 2026 Q3 2026"
    ) == [
        {
            "fiscal_year": 2026,
            "fiscal_quarter": 1,
            "period_basis": "quarterly",
        },
        {
            "fiscal_year": 2026,
            "fiscal_quarter": 2,
            "period_basis": "quarterly",
        },
        {
            "fiscal_year": 2026,
            "fiscal_quarter": 3,
            "period_basis": "quarterly",
        },
    ]


def test_parse_cross_year_quarters():
    assert parse_period_header(
        "Q4 2025 Q1 2026"
    ) == [
        {
            "fiscal_year": 2025,
            "fiscal_quarter": 4,
            "period_basis": "quarterly",
        },
        {
            "fiscal_year": 2026,
            "fiscal_quarter": 1,
            "period_basis": "quarterly",
        },
    ]


def test_single_year_is_not_table_header():
    assert parse_period_header(
        "Results for 2025"
    ) == []


def test_narrative_date_is_not_header():
    assert parse_period_header(
        "The company was founded in 2016."
    ) == []


def test_quarter_without_year_is_rejected():
    assert parse_period_header(
        "Q1 Q2 Q3 Q4"
    ) == []


def test_half_without_year_is_rejected():
    assert parse_period_header(
        "H1 H2"
    ) == []


def test_find_header_inside_lines():
    result = find_period_header(
        [
            "Financial performance",
            "NGN billion",
            "2024 2025 H1 2026",
            "Revenue 9,381 18,738 19,135",
        ]
    )

    assert result is not None

    index, periods = result

    assert index == 2
    assert periods[-1] == {
        "fiscal_year": 2026,
        "fiscal_half": 1,
        "period_basis": "half_year",
    }
