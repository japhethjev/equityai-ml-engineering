import re
from dataclasses import dataclass
from typing import Literal


PeriodType = Literal[
    "annual",
    "quarterly",
]


@dataclass(frozen=True)
class HistoricalSelection:
    """
    A requested historical reporting window.

    count:
        Number of reporting periods requested.

    period_type:
        annual means fiscal-year reports.
        quarterly means quarterly reports.
    """

    count: int
    period_type: PeriodType


_YEAR_PATTERNS = (
    r"\blast\s+(\d+)\s+(?:fiscal\s+)?years?\b",
    r"\bpast\s+(\d+)\s+(?:fiscal\s+)?years?\b",
    r"\bprevious\s+(\d+)\s+(?:fiscal\s+)?years?\b",
    r"\bprior\s+(\d+)\s+(?:fiscal\s+)?years?\b",
)

_QUARTER_PATTERNS = (
    r"\blast\s+(\d+)\s+quarters?\b",
    r"\bpast\s+(\d+)\s+quarters?\b",
    r"\bprevious\s+(\d+)\s+quarters?\b",
    r"\bprior\s+(\d+)\s+quarters?\b",
)


def parse_historical_selection(
    question: str,
) -> HistoricalSelection | None:
    """
    Extract an explicit rolling historical window.

    Examples:
        last 5 years
        past 3 fiscal years
        previous 8 quarters
        prior 4 quarters

    Returns None when no explicit historical window exists.

    This parser does not infer a period count from vague
    wording such as "historical performance" or "recent years".
    """

    text = " ".join(
        question.strip().lower().split()
    )

    if not text:
        return None

    for pattern in _YEAR_PATTERNS:
        match = re.search(
            pattern,
            text,
        )

        if match:
            count = int(
                match.group(1)
            )

            _validate_count(
                count,
                period_type="annual",
            )

            return HistoricalSelection(
                count=count,
                period_type="annual",
            )

    for pattern in _QUARTER_PATTERNS:
        match = re.search(
            pattern,
            text,
        )

        if match:
            count = int(
                match.group(1)
            )

            _validate_count(
                count,
                period_type="quarterly",
            )

            return HistoricalSelection(
                count=count,
                period_type="quarterly",
            )

    return None


def _validate_count(
    count: int,
    period_type: PeriodType,
) -> None:
    """
    Reject nonsensical or excessively broad windows.

    Limits are deliberately generous and can be revised
    later without changing the parsing interface.
    """

    if count <= 0:
        raise ValueError(
            "Historical period count must be greater than zero."
        )

    maximum = (
        20
        if period_type == "annual"
        else 40
    )

    if count > maximum:
        raise ValueError(
            f"Historical {period_type} window "
            f"cannot exceed {maximum} periods."
        )
