from app.rag.financial_facts import (
    make_reported_fact,
)
from app.rag.financial_period_selector import (
    select_financial_facts_for_periods,
)


def _fact(
    *,
    metric_key,
    value,
    fiscal_year,
    period_basis,
    fiscal_quarter=None,
    fiscal_half=None,
):
    return make_reported_fact(
        metric_key=metric_key,
        value=value,
        document_id="doc-1",
        document_name="report.pdf",
        page=1,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        fiscal_half=fiscal_half,
        period_basis=period_basis,
    )


def test_selects_latest_two_annual_periods():
    facts = [
        _fact(
            metric_key="revenue",
            value=100,
            fiscal_year=2023,
            period_basis="annual",
        ),
        _fact(
            metric_key="revenue",
            value=200,
            fiscal_year=2024,
            period_basis="annual",
        ),
        _fact(
            metric_key="revenue",
            value=300,
            fiscal_year=2025,
            period_basis="annual",
        ),
    ]

    selected = select_financial_facts_for_periods(
        facts=facts,
        period_type="annual",
        count=2,
    )

    assert [
        fact.fiscal_year
        for fact in selected
    ] == [2024, 2025]


def test_annual_selection_excludes_half_year():
    facts = [
        _fact(
            metric_key="revenue",
            value=100,
            fiscal_year=2024,
            period_basis="annual",
        ),
        _fact(
            metric_key="revenue",
            value=200,
            fiscal_year=2025,
            period_basis="annual",
        ),
        _fact(
            metric_key="revenue",
            value=300,
            fiscal_year=2026,
            fiscal_half=1,
            period_basis="half_year",
        ),
    ]

    selected = select_financial_facts_for_periods(
        facts=facts,
        period_type="annual",
        count=2,
    )

    assert [
        (
            fact.fiscal_year,
            fact.period_basis,
        )
        for fact in selected
    ] == [
        (2024, "annual"),
        (2025, "annual"),
    ]


def test_selects_all_metrics_for_selected_periods():
    facts = [
        _fact(
            metric_key="revenue",
            value=100,
            fiscal_year=2024,
            period_basis="annual",
        ),
        _fact(
            metric_key="gross_profit",
            value=20,
            fiscal_year=2024,
            period_basis="annual",
        ),
        _fact(
            metric_key="revenue",
            value=200,
            fiscal_year=2025,
            period_basis="annual",
        ),
        _fact(
            metric_key="gross_profit",
            value=50,
            fiscal_year=2025,
            period_basis="annual",
        ),
    ]

    selected = select_financial_facts_for_periods(
        facts=facts,
        period_type="annual",
        count=2,
    )

    assert len(selected) == 4

    assert {
        (
            fact.fiscal_year,
            fact.metric_key,
        )
        for fact in selected
    } == {
        (2024, "revenue"),
        (2024, "gross_profit"),
        (2025, "revenue"),
        (2025, "gross_profit"),
    }


def test_selects_latest_quarters_chronologically():
    facts = [
        _fact(
            metric_key="revenue",
            value=100,
            fiscal_year=2025,
            fiscal_quarter=3,
            period_basis="quarterly",
        ),
        _fact(
            metric_key="revenue",
            value=110,
            fiscal_year=2025,
            fiscal_quarter=4,
            period_basis="quarterly",
        ),
        _fact(
            metric_key="revenue",
            value=120,
            fiscal_year=2026,
            fiscal_quarter=1,
            period_basis="quarterly",
        ),
    ]

    selected = select_financial_facts_for_periods(
        facts=facts,
        period_type="quarterly",
        count=2,
    )

    assert [
        (
            fact.fiscal_year,
            fact.fiscal_quarter,
        )
        for fact in selected
    ] == [
        (2025, 4),
        (2026, 1),
    ]


def test_returns_partial_history_without_inventing_periods():
    facts = [
        _fact(
            metric_key="revenue",
            value=200,
            fiscal_year=2025,
            period_basis="annual",
        ),
    ]

    selected = select_financial_facts_for_periods(
        facts=facts,
        period_type="annual",
        count=5,
    )

    assert len(selected) == 1
    assert selected[0].fiscal_year == 2025


def test_ignores_facts_without_required_period_identity():
    facts = [
        _fact(
            metric_key="revenue",
            value=100,
            fiscal_year=2024,
            period_basis="annual",
        ),
        _fact(
            metric_key="revenue",
            value=999,
            fiscal_year=None,
            period_basis="annual",
        ),
    ]

    selected = select_financial_facts_for_periods(
        facts=facts,
        period_type="annual",
        count=2,
    )

    assert len(selected) == 1
    assert selected[0].fiscal_year == 2024


def test_rejects_invalid_period_type():
    facts = []

    try:
        select_financial_facts_for_periods(
            facts=facts,
            period_type="half_year",
            count=2,
        )
    except ValueError as exc:
        assert "annual or quarterly" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_rejects_non_positive_count():
    facts = []

    try:
        select_financial_facts_for_periods(
            facts=facts,
            period_type="annual",
            count=0,
        )
    except ValueError as exc:
        assert "greater than zero" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError"
        )
