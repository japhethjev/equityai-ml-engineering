from app.rag.financial_analysis_executor import (
    execute_financial_analysis,
)
from app.rag.financial_facts import (
    make_reported_fact,
)
from app.rag.financial_narrative import (
    build_financial_narrative,
)


def fact(
    metric_key,
    value,
    *,
    year,
    half=None,
    period_basis="annual",
):
    return make_reported_fact(
        metric_key=metric_key,
        value=value,
        document_id=f"doc-{year}-{half or 0}",
        document_name="report.pdf",
        page=1,
        company_name="Example Plc",
        currency="GBP",
        unit="million",
        fiscal_year=year,
        fiscal_half=half,
        period_basis=period_basis,
    )


def company(
    metric_keys,
    facts,
):
    return execute_financial_analysis(
        company_name="Example Plc",
        ticker="ABC",
        metric_keys=metric_keys,
        facts=facts,
    ).company


def test_revenue_growth_is_narrated():
    result = build_financial_narrative(
        company(
            ("revenue",),
            [
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
            ],
        )
    )

    assert len(result.observations) == 1
    assert "25.00% increase" in result.text


def test_revenue_decline_is_narrated():
    result = build_financial_narrative(
        company(
            ("revenue",),
            [
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
            ],
        )
    )

    assert "20.00% decrease" in result.text


def test_loss_narrowing_is_narrated():
    result = build_financial_narrative(
        company(
            ("profit_after_tax",),
            [
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
            ],
        )
    )

    assert "loss narrowed" in result.text
    assert "50.00% reduction" in result.text


def test_loss_to_profit_turnaround_is_narrated():
    result = build_financial_narrative(
        company(
            ("profit_after_tax",),
            [
                fact(
                    "profit_after_tax",
                    "-50",
                    year=2024,
                ),
                fact(
                    "profit_after_tax",
                    "25",
                    year=2025,
                ),
            ],
        )
    )

    assert (
        "loss-to-profit turnaround"
        in result.text
    )


def test_margin_change_uses_percentage_points():
    result = build_financial_narrative(
        company(
            (
                "revenue",
                "gross_profit",
                "gross_profit_margin",
            ),
            [
                fact(
                    "revenue",
                    "100",
                    year=2024,
                ),
                fact(
                    "gross_profit",
                    "20",
                    year=2024,
                ),
                fact(
                    "revenue",
                    "100",
                    year=2025,
                ),
                fact(
                    "gross_profit",
                    "30",
                    year=2025,
                ),
            ],
        )
    )

    margin = next(
        observation
        for observation in result.observations
        if (
            observation.metric_key
            == "gross_profit_margin"
        )
    )

    assert "10.00 percentage-point" in margin.text


def test_annual_to_half_year_is_not_compared():
    result = build_financial_narrative(
        company(
            ("revenue",),
            [
                fact(
                    "revenue",
                    "100",
                    year=2025,
                ),
                fact(
                    "revenue",
                    "70",
                    year=2026,
                    half=1,
                    period_basis="half_year",
                ),
            ],
        )
    )

    assert result.observations == ()
    assert result.text == ""


def test_nonconsecutive_years_are_not_compared():
    result = build_financial_narrative(
        company(
            ("revenue",),
            [
                fact(
                    "revenue",
                    "100",
                    year=2023,
                ),
                fact(
                    "revenue",
                    "150",
                    year=2025,
                ),
            ],
        )
    )

    assert result.observations == ()


def test_zero_base_does_not_invent_percentage():
    result = build_financial_narrative(
        company(
            ("revenue",),
            [
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
            ],
        )
    )

    assert "zero base" in result.text
    assert "100.00% increase" not in result.text
