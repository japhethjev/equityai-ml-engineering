from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from app.rag.financial_facts import (
    FinancialFact,
    same_reporting_period,
)
from app.rag.financial_metrics import (
    get_metric,
)


@dataclass(frozen=True)
class CalculatedMetric:
    """
    Deterministically calculated financial metric.

    input_facts preserve the provenance of every value used
    in the calculation.
    """

    metric_key: str
    value: Decimal
    unit: str | None
    input_facts: tuple[FinancialFact, ...]
    formula: str
    source: str = "derived"


@dataclass(frozen=True)
class FinancialChange:
    """
    Change in the same financial metric between two periods.
    """

    metric_key: str
    previous_value: Decimal
    current_value: Decimal
    absolute_change: Decimal
    percentage_change: Decimal | None
    previous_fact: FinancialFact
    current_fact: FinancialFact


def calculate_metric(
    metric_key: str,
    facts: list[FinancialFact],
) -> CalculatedMetric:
    """
    Calculate a supported derived financial metric.

    Inputs must:
    - contain all required canonical metrics,
    - belong to the same reporting period,
    - and have valid denominators.
    """

    metric = get_metric(metric_key)

    if metric.kind != "derived":
        raise ValueError(
            f"{metric_key} is a reported metric "
            "and does not require calculation"
        )

    facts_by_metric = {
        fact.metric_key: fact
        for fact in facts
    }

    missing = [
        required
        for required in metric.required_inputs
        if required not in facts_by_metric
    ]

    if missing:
        raise ValueError(
            "Missing required financial facts: "
            + ", ".join(missing)
        )

    input_facts = tuple(
        facts_by_metric[key]
        for key in metric.required_inputs
    )

    _validate_same_period(
        input_facts
    )

    values = {
        fact.metric_key: fact.value
        for fact in input_facts
    }

    result = _calculate(
        metric_key,
        values,
    )

    return CalculatedMetric(
        metric_key=metric.key,
        value=result,
        unit=metric.output_unit,
        input_facts=input_facts,
        formula=metric.formula or "",
    )


def _calculate(
    metric_key: str,
    values: dict[str, Decimal],
) -> Decimal:
    """
    Execute approved deterministic formulas.

    Formulas are implemented explicitly rather than evaluated
    dynamically from strings.
    """

    if metric_key == "gross_profit_margin":
        return _percentage(
            values["gross_profit"],
            values["revenue"],
        )

    if metric_key == "operating_profit_margin":
        return _percentage(
            values["operating_profit"],
            values["revenue"],
        )

    if metric_key == "net_profit_margin":
        return _percentage(
            values["profit_after_tax"],
            values["revenue"],
        )

    if metric_key == "current_ratio":
        return _ratio(
            values["current_assets"],
            values["current_liabilities"],
        )

    if metric_key == "quick_ratio":
        numerator = (
            values["current_assets"]
            - values["inventory"]
        )

        return _ratio(
            numerator,
            values["current_liabilities"],
        )

    if metric_key == "cash_ratio":
        return _ratio(
            values["cash"],
            values["current_liabilities"],
        )

    if metric_key == "debt_to_equity":
        return _ratio(
            values["total_debt"],
            values["total_equity"],
        )

    if metric_key == "debt_to_assets":
        return _ratio(
            values["total_debt"],
            values["total_assets"],
        )

    if metric_key == "free_cash_flow":
        return (
            values["operating_cash_flow"]
            - values["capital_expenditure"]
        )

    if metric_key == "operating_cash_flow_margin":
        return _percentage(
            values["operating_cash_flow"],
            values["revenue"],
        )

    raise ValueError(
        f"Calculation not implemented for metric: {metric_key}"
    )


def calculate_change(
    previous: FinancialFact,
    current: FinancialFact,
) -> FinancialChange:
    """
    Calculate absolute and percentage change between two
    observations of the same reported financial metric.

    Percentage change:

        (current - previous) / abs(previous) * 100

    Using abs(previous) gives economically intuitive direction
    when the prior period is negative.

    If the previous value is zero, percentage change is undefined
    and None is returned.
    """

    if previous.metric_key != current.metric_key:
        raise ValueError(
            "Change calculation requires the same metric"
        )

    absolute_change = (
        current.value
        - previous.value
    )

    if previous.value == 0:
        percentage_change = None
    else:
        percentage_change = (
            absolute_change
            / abs(previous.value)
            * Decimal("100")
        )

        percentage_change = _round(
            percentage_change
        )

    return FinancialChange(
        metric_key=current.metric_key,
        previous_value=previous.value,
        current_value=current.value,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        previous_fact=previous,
        current_fact=current,
    )


def _validate_same_period(
    facts: tuple[FinancialFact, ...],
) -> None:
    """
    Ensure all inputs used in a ratio belong to the same
    reporting period.
    """

    if len(facts) < 2:
        return

    first = facts[0]

    for fact in facts[1:]:
        if not same_reporting_period(
            first,
            fact,
        ):
            raise ValueError(
                "Financial calculation inputs must belong "
                "to the same reporting period"
            )


def _percentage(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:
    if denominator == 0:
        raise ZeroDivisionError(
            "Financial metric denominator cannot be zero"
        )

    return _round(
        numerator
        / denominator
        * Decimal("100")
    )


def _ratio(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:
    if denominator == 0:
        raise ZeroDivisionError(
            "Financial metric denominator cannot be zero"
        )

    return _round(
        numerator
        / denominator
    )


def _round(
    value: Decimal,
) -> Decimal:
    """
    Standardise calculated financial metrics to two decimal
    places while retaining Decimal arithmetic.
    """

    return value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
