from decimal import Decimal

from app.rag.financial_analysis import (
    derived_analysis_value,
    reported_analysis_value,
)
from app.rag.financial_facts import (
    make_reported_fact,
)
from app.rag.financial_presentation import (
    CompanyAnalysis,
    build_financial_presentation,
)


DOCUMENT_ID = (
    "11111111-1111-1111-1111-111111111111"
)


def fact(
    metric_key,
    value,
    *,
    company="Example Plc",
    ticker="ABC",
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
        company_name=company,
        ticker=ticker,
        exchange="NGX",
        fiscal_year=year,
        fiscal_quarter=quarter,
        period_basis=period_basis,
        currency="NGN",
        unit="million",
    )


def reported(
    metric_key,
    value,
    **kwargs,
):
    return reported_analysis_value(
        fact(
            metric_key,
            value,
            **kwargs,
        )
    )


def margin(
    gross_profit,
    revenue,
    **kwargs,
):
    return derived_analysis_value(
        "gross_profit_margin",
        [
            fact(
                "gross_profit",
                gross_profit,
                **kwargs,
            ),
            fact(
                "revenue",
                revenue,
                **kwargs,
            ),
        ],
    )


def test_single_company_single_period_is_prose():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
            ),
            reported(
                "gross_profit",
                "40",
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert (
        result.mode
        == "single_company_single_period"
    )

    assert result.headers == ()
    assert "Revenue was 100.00." in result.text
    assert "Gross Profit was 40.00." in result.text
    assert "|" not in result.text


def test_single_company_multi_year_has_yoy_last():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2024,
            ),
            reported(
                "revenue",
                "125",
                year=2025,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert (
        result.mode
        == "single_company_multi_period"
    )

    assert result.headers == (
        "KPI",
        "FY2024",
        "FY2025",
        "YoY %",
    )

    assert result.rows[0] == (
        "Revenue",
        "100.00",
        "125.00",
        "+25.00%",
    )


def test_single_company_quarters_use_qoq_last():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2025,
                quarter=1,
                period_basis="quarterly",
            ),
            reported(
                "revenue",
                "110",
                year=2025,
                quarter=2,
                period_basis="quarterly",
            ),
            reported(
                "revenue",
                "121",
                year=2025,
                quarter=3,
                period_basis="quarterly",
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.headers[-1] == "QoQ %"

    assert result.rows[0][-1] == "+10.00%"


def test_margin_latest_change_uses_pp():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            margin(
                "35",
                "100",
                year=2024,
            ),
            margin(
                "42",
                "100",
                year=2025,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.rows[0] == (
        "Gross Profit Margin",
        "35.00%",
        "42.00%",
        "+7.00 pp",
    )


def test_loss_to_profit_growth_is_displayed():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "profit_after_tax",
                "-100",
                year=2024,
            ),
            reported(
                "profit_after_tax",
                "50",
                year=2025,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.rows[0][-1] == "+150.00%"


def test_zero_base_change_is_unavailable():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "0",
                year=2024,
            ),
            reported(
                "revenue",
                "100",
                year=2025,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.rows[0][-1] == "—"


def test_missing_period_value_displays_dash():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2023,
            ),
            reported(
                "revenue",
                "125",
                year=2025,
            ),
            reported(
                "gross_profit",
                "50",
                year=2024,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.headers == (
        "KPI",
        "FY2023",
        "FY2024",
        "FY2025",
        "YoY %",
    )

    revenue_row = result.rows[0]

    assert revenue_row == (
        "Revenue",
        "100.00",
        "—",
        "125.00",
        "—",
    )


def test_multi_company_single_period_uses_companies_as_columns():
    company_a = CompanyAnalysis(
        company_name="Alpha Plc",
        ticker="ALP",
        values=(
            reported(
                "revenue",
                "100",
                company="Alpha Plc",
                ticker="ALP",
            ),
            reported(
                "gross_profit",
                "40",
                company="Alpha Plc",
                ticker="ALP",
            ),
        ),
    )

    company_b = CompanyAnalysis(
        company_name="Beta Plc",
        ticker="BET",
        values=(
            reported(
                "revenue",
                "150",
                company="Beta Plc",
                ticker="BET",
            ),
            reported(
                "gross_profit",
                "50",
                company="Beta Plc",
                ticker="BET",
            ),
        ),
    )

    result = build_financial_presentation(
        [
            company_a,
            company_b,
        ]
    )

    assert (
        result.mode
        == "multi_company_single_period"
    )

    assert result.headers == (
        "KPI",
        "Alpha Plc (ALP)",
        "Beta Plc (BET)",
    )

    assert result.rows[0] == (
        "Revenue",
        "100.00",
        "150.00",
    )


def test_multi_company_multi_period_groups_companies():
    company_a = CompanyAnalysis(
        company_name="Alpha Plc",
        ticker="ALP",
        values=(
            reported(
                "revenue",
                "100",
                company="Alpha Plc",
                ticker="ALP",
                year=2024,
            ),
            reported(
                "revenue",
                "120",
                company="Alpha Plc",
                ticker="ALP",
                year=2025,
            ),
        ),
    )

    company_b = CompanyAnalysis(
        company_name="Beta Plc",
        ticker="BET",
        values=(
            reported(
                "revenue",
                "200",
                company="Beta Plc",
                ticker="BET",
                year=2024,
            ),
            reported(
                "revenue",
                "250",
                company="Beta Plc",
                ticker="BET",
                year=2025,
            ),
        ),
    )

    result = build_financial_presentation(
        [
            company_a,
            company_b,
        ]
    )

    assert (
        result.mode
        == "multi_company_multi_period"
    )

    assert result.headers == (
        "Company / KPI",
        "FY2024",
        "FY2025",
        "YoY %",
    )

    assert result.rows[0][0] == (
        "**Alpha Plc (ALP)**"
    )

    assert result.rows[1] == (
        "Revenue",
        "100.00",
        "120.00",
        "+20.00%",
    )

    assert result.rows[2][0] == (
        "**Beta Plc (BET)**"
    )

    assert result.rows[3] == (
        "Revenue",
        "200.00",
        "250.00",
        "+25.00%",
    )


def test_periods_are_chronological():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "130",
                year=2025,
            ),
            reported(
                "revenue",
                "100",
                year=2023,
            ),
            reported(
                "revenue",
                "120",
                year=2024,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.headers == (
        "KPI",
        "FY2023",
        "FY2024",
        "FY2025",
        "YoY %",
    )


def test_duplicate_metric_period_is_rejected():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2025,
            ),
            reported(
                "revenue",
                "101",
                year=2025,
            ),
        ),
    )

    try:
        build_financial_presentation(
            [company]
        )
    except ValueError as exc:
        assert "Duplicate analysis value" in str(exc)
    else:
        raise AssertionError(
            "Expected duplicate metric/period to fail"
        )


def test_nonconsecutive_years_do_not_show_yoy():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2023,
            ),
            reported(
                "revenue",
                "130",
                year=2025,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.headers[-1] == "YoY %"
    assert result.rows[0][-1] == "—"


def test_consecutive_years_show_yoy():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2024,
            ),
            reported(
                "revenue",
                "125",
                year=2025,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.rows[0][-1] == "+25.00%"


def test_missing_latest_year_does_not_jump_back_for_yoy():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2023,
            ),
            reported(
                "revenue",
                "120",
                year=2024,
            ),
            reported(
                "gross_profit",
                "50",
                year=2025,
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    revenue_row = next(
        row
        for row in result.rows
        if row[0] == "Revenue"
    )

    assert revenue_row == (
        "Revenue",
        "100.00",
        "120.00",
        "—",
        "—",
    )


def test_nonconsecutive_quarters_do_not_show_qoq():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2025,
                quarter=1,
                period_basis="quarterly",
            ),
            reported(
                "revenue",
                "130",
                year=2025,
                quarter=3,
                period_basis="quarterly",
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.headers[-1] == "QoQ %"
    assert result.rows[0][-1] == "—"


def test_consecutive_quarters_show_qoq():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2025,
                quarter=2,
                period_basis="quarterly",
            ),
            reported(
                "revenue",
                "110",
                year=2025,
                quarter=3,
                period_basis="quarterly",
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.rows[0][-1] == "+10.00%"


def test_q4_to_q1_next_year_is_consecutive():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2024,
                quarter=4,
                period_basis="quarterly",
            ),
            reported(
                "revenue",
                "120",
                year=2025,
                quarter=1,
                period_basis="quarterly",
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.rows[0][-1] == "+20.00%"


def test_q4_to_q2_next_year_is_not_consecutive():
    company = CompanyAnalysis(
        company_name="Example Plc",
        ticker="ABC",
        values=(
            reported(
                "revenue",
                "100",
                year=2024,
                quarter=4,
                period_basis="quarterly",
            ),
            reported(
                "revenue",
                "120",
                year=2025,
                quarter=2,
                period_basis="quarterly",
            ),
        ),
    )

    result = build_financial_presentation(
        [company]
    )

    assert result.rows[0][-1] == "—"
