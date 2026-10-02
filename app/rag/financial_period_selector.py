from app.rag.financial_facts import (
    FinancialFact,
)


def select_financial_facts_for_periods(
    *,
    facts: list[FinancialFact],
    period_type: str,
    count: int,
) -> list[FinancialFact]:
    """
    Select grounded financial facts belonging to the latest
    requested reporting periods.

    Period selection is based on fact-level period metadata,
    not the container document's report_type.

    This allows a research or valuation document containing
    multiple annual periods to support historical analysis
    without treating half-year or quarterly figures as annual
    results.

    Missing periods are never invented.
    """

    normalized_period_type = (
        period_type.strip().lower()
    )

    if normalized_period_type not in {
        "annual",
        "quarterly",
    }:
        raise ValueError(
            "period_type must be annual or quarterly"
        )

    if count <= 0:
        raise ValueError(
            "count must be greater than zero"
        )

    if normalized_period_type == "annual":
        eligible = [
            fact
            for fact in facts
            if fact.period_basis == "annual"
            and fact.fiscal_year is not None
        ]

        periods = sorted(
            {
                fact.fiscal_year
                for fact in eligible
            }
        )

        selected_periods = set(
            periods[-count:]
        )

        return [
            fact
            for fact in eligible
            if fact.fiscal_year
            in selected_periods
        ]

    eligible = [
        fact
        for fact in facts
        if fact.period_basis == "quarterly"
        and fact.fiscal_year is not None
        and fact.fiscal_quarter is not None
    ]

    periods = sorted(
        {
            (
                fact.fiscal_year,
                fact.fiscal_quarter,
            )
            for fact in eligible
        }
    )

    selected_periods = set(
        periods[-count:]
    )

    return [
        fact
        for fact in eligible
        if (
            fact.fiscal_year,
            fact.fiscal_quarter,
        )
        in selected_periods
    ]
