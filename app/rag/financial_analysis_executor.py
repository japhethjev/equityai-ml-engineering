from dataclasses import dataclass

from app.rag.financial_analysis import (
    AnalysisValue,
    derived_analysis_value,
    reported_analysis_value,
)
from app.rag.financial_facts import FinancialFact
from app.rag.financial_metrics import get_metric
from app.rag.financial_presentation import CompanyAnalysis


@dataclass(frozen=True)
class FinancialAnalysisExecution:
    """
    Deterministic result of executing requested investor KPIs
    against validated financial facts.

    Missing metrics are recorded explicitly rather than estimated.
    """

    company: CompanyAnalysis
    missing: tuple[tuple[str, str], ...]


def _fact_period_key(
    fact: FinancialFact,
) -> tuple:
    return (
        fact.fiscal_year,
        fact.fiscal_quarter,
        fact.fiscal_half,
        fact.period_basis,
    )


def _period_label_from_fact(
    fact: FinancialFact,
) -> str:
    if (
        fact.fiscal_quarter is not None
        and fact.fiscal_year is not None
    ):
        return (
            f"Q{fact.fiscal_quarter} "
            f"FY{fact.fiscal_year}"
        )

    if (
        fact.fiscal_half is not None
        and fact.fiscal_year is not None
    ):
        return (
            f"H{fact.fiscal_half} "
            f"FY{fact.fiscal_year}"
        )

    if fact.fiscal_year is not None:
        return f"FY{fact.fiscal_year}"

    return "Unspecified Period"


def _group_facts_by_period(
    facts: list[FinancialFact],
) -> dict[tuple, list[FinancialFact]]:
    grouped = {}

    for fact in facts:
        grouped.setdefault(
            _fact_period_key(fact),
            [],
        ).append(fact)

    return grouped


def _select_unique_fact(
    facts: list[FinancialFact],
    metric_key: str,
) -> FinancialFact | None:
    """
    Select one reported fact for a metric within one period.

    Multiple distinct matching facts are rejected rather than
    silently choosing one.
    """

    matches = [
        fact
        for fact in facts
        if fact.metric_key == metric_key
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            "Multiple reported facts found for "
            f"{metric_key} in the same reporting period"
        )

    return matches[0]


def _build_metric_value(
    metric_key: str,
    period_facts: list[FinancialFact],
) -> AnalysisValue | None:
    """
    Build one requested KPI for one reporting period.

    Reported metrics must exist directly.

    Derived metrics are calculated only from the canonical
    required inputs defined in the financial metric registry.
    """

    metric = get_metric(metric_key)

    if metric.kind == "reported":
        fact = _select_unique_fact(
            period_facts,
            metric_key,
        )

        if fact is None:
            return None

        return reported_analysis_value(
            fact
        )

    required_facts = []

    for required_key in metric.required_inputs:
        fact = _select_unique_fact(
            period_facts,
            required_key,
        )

        if fact is None:
            return None

        required_facts.append(fact)

    return derived_analysis_value(
        metric_key,
        required_facts,
    )


def execute_financial_analysis(
    *,
    company_name: str,
    ticker: str | None,
    metric_keys: tuple[str, ...],
    facts: list[FinancialFact],
) -> FinancialAnalysisExecution:
    """
    Execute requested financial KPIs across all available periods.

    No missing KPI is estimated or reconstructed from unrelated
    information.

    Derived metrics are calculated only when every canonical
    required input exists for the same reporting period.
    """

    grouped = _group_facts_by_period(
        facts
    )

    ordered_periods = sorted(
        grouped,
        key=lambda key: (
            key[0] if key[0] is not None else -1,
            key[1] if key[1] is not None else 0,
            key[2] if key[2] is not None else 0,
        ),
    )

    values = []
    missing = []

    for period_key in ordered_periods:
        period_facts = grouped[
            period_key
        ]

        representative = period_facts[0]

        period_label = (
            _period_label_from_fact(
                representative
            )
        )

        for metric_key in metric_keys:
            value = _build_metric_value(
                metric_key,
                period_facts,
            )

            if value is None:
                missing.append(
                    (
                        metric_key,
                        period_label,
                    )
                )
                continue

            values.append(value)

    return FinancialAnalysisExecution(
        company=CompanyAnalysis(
            company_name=company_name,
            ticker=ticker,
            values=tuple(values),
        ),
        missing=tuple(missing),
    )
