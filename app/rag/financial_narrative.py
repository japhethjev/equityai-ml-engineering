from dataclasses import dataclass
from decimal import Decimal

from app.rag.financial_analysis import AnalysisValue
from app.rag.financial_metrics import get_metric
from app.rag.financial_presentation import (
    CompanyAnalysis,
    are_consecutive_periods,
)


@dataclass(frozen=True)
class NarrativeObservation:
    """
    One deterministic investor-facing financial observation.

    previous and current retain the validated AnalysisValue
    objects used to generate the observation.
    """

    metric_key: str
    previous: AnalysisValue
    current: AnalysisValue
    text: str


@dataclass(frozen=True)
class FinancialNarrative:
    """
    Deterministic narrative generated from validated financial
    analysis values.

    No financial values are estimated or reconstructed here.
    """

    observations: tuple[NarrativeObservation, ...]
    text: str


def _metric_label(
    metric_key: str,
) -> str:
    return get_metric(
        metric_key
    ).label.title()


def _format_number(
    value: Decimal,
) -> str:
    return f"{abs(value):,.2f}"


def _format_signed_percent(
    value: Decimal,
) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _percentage_change(
    previous: Decimal,
    current: Decimal,
) -> Decimal | None:
    """
    Investor-facing percentage movement.

    Absolute prior value is used as the denominator so movement
    from a negative base remains economically interpretable.

    Zero prior values have no meaningful percentage change.
    """

    if previous == 0:
        return None

    return (
        (
            current - previous
        )
        / abs(previous)
        * Decimal("100")
    ).quantize(
        Decimal("0.01")
    )


def _is_margin(
    value: AnalysisValue,
) -> bool:
    return (
        value.unit == "%"
        or value.metric_key.endswith(
            "_margin"
        )
    )


def _same_metric(
    left: AnalysisValue,
    right: AnalysisValue,
) -> bool:
    return (
        left.metric_key
        == right.metric_key
    )


def _profit_transition_text(
    previous: AnalysisValue,
    current: AnalysisValue,
) -> str | None:
    """
    Describe economically important sign transitions and
    loss movements before applying ordinary growth language.
    """

    label = _metric_label(
        current.metric_key
    )

    old = previous.value
    new = current.value

    if old < 0 and new > 0:
        return (
            f"{label} moved from a loss of "
            f"{_format_number(old)} in "
            f"{previous.period_label} to a profit of "
            f"{_format_number(new)} in "
            f"{current.period_label}, representing a "
            f"loss-to-profit turnaround."
        )

    if old > 0 and new < 0:
        return (
            f"{label} moved from a profit of "
            f"{_format_number(old)} in "
            f"{previous.period_label} to a loss of "
            f"{_format_number(new)} in "
            f"{current.period_label}."
        )

    if old < 0 and new < 0:
        old_loss = abs(old)
        new_loss = abs(new)

        if new_loss < old_loss:
            reduction = (
                (
                    old_loss - new_loss
                )
                / old_loss
                * Decimal("100")
            ).quantize(
                Decimal("0.01")
            )

            return (
                f"{label} remained negative, but the loss "
                f"narrowed from {_format_number(old)} in "
                f"{previous.period_label} to "
                f"{_format_number(new)} in "
                f"{current.period_label}, a "
                f"{reduction:.2f}% reduction in the "
                f"magnitude of the loss."
            )

        if new_loss > old_loss:
            increase = (
                (
                    new_loss - old_loss
                )
                / old_loss
                * Decimal("100")
            ).quantize(
                Decimal("0.01")
            )

            return (
                f"{label} remained negative and the loss "
                f"widened from {_format_number(old)} in "
                f"{previous.period_label} to "
                f"{_format_number(new)} in "
                f"{current.period_label}, a "
                f"{increase:.2f}% increase in the "
                f"magnitude of the loss."
            )

    return None


def _margin_text(
    previous: AnalysisValue,
    current: AnalysisValue,
) -> str:
    label = _metric_label(
        current.metric_key
    )

    change = (
        current.value
        - previous.value
    ).quantize(
        Decimal("0.01")
    )

    if change > 0:
        direction = "expanded"
    elif change < 0:
        direction = "contracted"
    else:
        direction = "was unchanged"

    if change == 0:
        return (
            f"{label} was unchanged at "
            f"{current.value:.2f}% between "
            f"{previous.period_label} and "
            f"{current.period_label}."
        )

    return (
        f"{label} {direction} from "
        f"{previous.value:.2f}% in "
        f"{previous.period_label} to "
        f"{current.value:.2f}% in "
        f"{current.period_label}, a "
        f"{abs(change):.2f} percentage-point "
        f"{'improvement' if change > 0 else 'decline'}."
    )


def _ordinary_change_text(
    previous: AnalysisValue,
    current: AnalysisValue,
) -> str:
    label = _metric_label(
        current.metric_key
    )

    change = _percentage_change(
        previous.value,
        current.value,
    )

    if change is None:
        return (
            f"{label} moved from zero in "
            f"{previous.period_label} to "
            f"{_format_number(current.value)} in "
            f"{current.period_label}; a percentage "
            f"change is not meaningful from a zero base."
        )

    if change > 0:
        verb = "increased"
    elif change < 0:
        verb = "decreased"
    else:
        return (
            f"{label} was unchanged at "
            f"{_format_number(current.value)} between "
            f"{previous.period_label} and "
            f"{current.period_label}."
        )

    return (
        f"{label} {verb} from "
        f"{_format_number(previous.value)} in "
        f"{previous.period_label} to "
        f"{_format_number(current.value)} in "
        f"{current.period_label}, a "
        f"{abs(change):.2f}% "
        f"{'increase' if change > 0 else 'decrease'}."
    )


def build_observation(
    previous: AnalysisValue,
    current: AnalysisValue,
) -> NarrativeObservation | None:
    """
    Build one observation only when two values are the same KPI
    and form a valid consecutive comparison.
    """

    if not _same_metric(
        previous,
        current,
    ):
        raise ValueError(
            "Narrative comparison requires the same metric"
        )

    if not are_consecutive_periods(
        previous,
        current,
    ):
        return None

    if _is_margin(current):
        text = _margin_text(
            previous,
            current,
        )
    else:
        transition = _profit_transition_text(
            previous,
            current,
        )

        if transition is not None:
            text = transition
        else:
            text = _ordinary_change_text(
                previous,
                current,
            )

    return NarrativeObservation(
        metric_key=current.metric_key,
        previous=previous,
        current=current,
        text=text,
    )


def build_financial_narrative(
    company: CompanyAnalysis,
) -> FinancialNarrative:
    """
    Generate deterministic narrative from validated analysis
    values.

    Only genuinely consecutive comparable periods are narrated.
    """

    by_metric = {}

    for value in company.values:
        by_metric.setdefault(
            value.metric_key,
            [],
        ).append(value)

    observations = []

    for metric_values in by_metric.values():

        ordered = sorted(
            metric_values,
            key=lambda value: (
                value.input_facts[0].fiscal_year
                if value.input_facts
                and value.input_facts[0].fiscal_year
                is not None
                else -1,
                value.input_facts[0].fiscal_quarter
                if value.input_facts
                and value.input_facts[0].fiscal_quarter
                is not None
                else 0,
                value.input_facts[0].fiscal_half
                if value.input_facts
                and value.input_facts[0].fiscal_half
                is not None
                else 0,
            ),
        )

        for previous, current in zip(
            ordered,
            ordered[1:],
        ):
            observation = build_observation(
                previous,
                current,
            )

            if observation is not None:
                observations.append(
                    observation
                )

    text = " ".join(
        observation.text
        for observation in observations
    )

    return FinancialNarrative(
        observations=tuple(
            observations
        ),
        text=text,
    )
