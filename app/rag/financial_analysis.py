from dataclasses import dataclass
from decimal import Decimal

from app.rag.financial_calculator import (
    calculate_metric,
)
from app.rag.financial_facts import FinancialFact
from app.rag.financial_metrics import get_metric


@dataclass(frozen=True)
class AnalysisValue:
    """
    One validated value prepared for financial analysis output.

    input_facts preserves the document provenance supporting
    either the reported or derived value.
    """

    metric_key: str
    value: Decimal
    unit: str | None
    period_label: str
    source: str
    input_facts: tuple[FinancialFact, ...]


@dataclass(frozen=True)
class LatestChange:
    """
    Latest comparable-period movement for one KPI.

    display_unit:
        "%"  -> percentage change
        "pp" -> percentage-point movement
        None -> percentage cannot be calculated
    """

    metric_key: str
    value: Decimal | None
    display_unit: str | None
    not_meaningful: bool = False


def period_label(
    fact: FinancialFact,
) -> str:
    """
    Produce the canonical presentation label for a fact.
    """

    year = fact.fiscal_year

    if (
        fact.fiscal_quarter is not None
        and year is not None
    ):
        return (
            f"Q{fact.fiscal_quarter} "
            f"FY{year}"
        )

    if (
        fact.fiscal_half is not None
        and year is not None
    ):
        return (
            f"H{fact.fiscal_half} "
            f"FY{year}"
        )

    if year is not None:
        return f"FY{year}"

    return "Unspecified Period"


def reported_analysis_value(
    fact: FinancialFact,
) -> AnalysisValue:
    """
    Convert a reported FinancialFact into an AnalysisValue
    without losing provenance.
    """

    return AnalysisValue(
        metric_key=fact.metric_key,
        value=fact.value,
        unit=fact.unit,
        period_label=period_label(fact),
        source="reported",
        input_facts=(fact,),
    )


def derived_analysis_value(
    metric_key: str,
    facts: list[FinancialFact],
) -> AnalysisValue:
    """
    Calculate a derived KPI using the existing deterministic
    financial calculator.
    """

    result = calculate_metric(
        metric_key,
        facts,
    )

    return AnalysisValue(
        metric_key=result.metric_key,
        value=result.value,
        unit=result.unit,
        period_label=period_label(
            result.input_facts[0]
        ),
        source="derived",
        input_facts=result.input_facts,
    )


def calculate_latest_change(
    previous: AnalysisValue,
    current: AnalysisValue,
) -> LatestChange:
    """
    Calculate the final YoY/QoQ presentation value.

    EquityAI investor-facing rules:

    1. Percentage KPIs such as margins use percentage-point
       movement.

    2. Absolute KPIs use percentage change.

    3. When the prior value is negative, the absolute value of
       the prior-period figure is used as the denominator. This
       preserves intuitive direction:

           -100 -> +50 = +150%
           +100 -> -50 = -150%
           -100 -> -50 = +50%

    4. When the prior value is zero, percentage change is
       mathematically undefined and is therefore not calculated.
    """

    if previous.metric_key != current.metric_key:
        raise ValueError(
            "Latest change requires the same metric"
        )

    metric = get_metric(
        current.metric_key
    )

    # -----------------------------------------------------
    # Percentage KPIs
    # Example:
    # gross margin 20% -> 25% = +5 percentage points
    # -----------------------------------------------------

    if metric.output_unit == "%":
        movement = (
            current.value
            - previous.value
        )

        return LatestChange(
            metric_key=current.metric_key,
            value=movement,
            display_unit="pp",
            not_meaningful=False,
        )

    # -----------------------------------------------------
    # Zero-base case
    # -----------------------------------------------------

    if previous.value == 0:
        return LatestChange(
            metric_key=current.metric_key,
            value=None,
            display_unit=None,
            not_meaningful=True,
        )

    # -----------------------------------------------------
    # Investor-facing percentage movement
    #
    # Absolute prior value deliberately handles losses in an
    # intuitive direction.
    # -----------------------------------------------------

    absolute_change = (
        current.value
        - previous.value
    )

    percentage_change = (
        absolute_change
        / abs(previous.value)
        * Decimal("100")
    ).quantize(
        Decimal("0.01")
    )

    return LatestChange(
        metric_key=current.metric_key,
        value=percentage_change,
        display_unit="%",
        not_meaningful=False,
    )
