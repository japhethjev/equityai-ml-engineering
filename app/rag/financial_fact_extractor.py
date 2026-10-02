import re
from decimal import Decimal, InvalidOperation

from app.rag.financial_facts import (
    FinancialFact,
    make_reported_fact,
)
from app.rag.financial_metrics import (
    FINANCIAL_METRICS,
)


_NUMBER_PATTERN = re.compile(
    r"""
    (?P<open>\()?
    (?P<sign>[+-])?
    (?P<number>
        \d{1,3}(?:,\d{3})*(?:\.\d+)?
        |
        \d+(?:\.\d+)?
    )
    (?P<percent>%?)
    (?P<close>\))?
    """,
    re.VERBOSE,
)


def _normalize_text(text: str) -> str:
    return " ".join(
        text.strip().lower().split()
    )


def _reported_metrics():
    return tuple(
        metric
        for metric in FINANCIAL_METRICS.values()
        if metric.kind == "reported"
    )


def resolve_reported_metric_in_line(
    line: str,
):
    """
    Resolve a reported financial metric when a line begins
    with its canonical label or one of its known aliases.

    Longest aliases are checked first so that, for example,
    'total current assets' is preferred to shorter terms.
    """

    normalized_line = _normalize_text(line)

    candidates = []

    for metric in _reported_metrics():
        names = (
            metric.label,
            *metric.aliases,
        )

        for name in names:
            normalized_name = _normalize_text(name)

            if (
                normalized_line == normalized_name
                or normalized_line.startswith(
                    normalized_name + " "
                )
            ):
                candidates.append(
                    (
                        len(normalized_name),
                        metric,
                        normalized_name,
                    )
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, metric, matched_name = candidates[0]

    return metric, matched_name


def parse_financial_number(
    text: str,
) -> Decimal | None:
    """
    Parse the first financial number from text.

    Accounting parentheses are treated as negative values.
    Percentage symbols are ignored here; units are handled
    separately by the caller.
    """

    match = _NUMBER_PATTERN.search(text)

    if not match:
        return None

    raw_number = (
        match.group("number")
        .replace(",", "")
    )

    try:
        value = Decimal(raw_number)
    except InvalidOperation:
        return None

    if (
        match.group("sign") == "-"
        or (
            match.group("open")
            and match.group("close")
        )
    ):
        value = -value

    return value


def extract_values_from_metric_line(
    line: str,
) -> tuple[str, list[Decimal]] | None:
    """
    Extract a canonical reported metric and all numeric values
    appearing after the metric label on the same line.
    """

    resolved = resolve_reported_metric_in_line(
        line
    )

    if resolved is None:
        return None

    metric, matched_name = resolved

    normalized_line = _normalize_text(line)

    remainder = normalized_line[
        len(matched_name):
    ].lstrip()

    # A genuine financial-table row must begin its numeric
    # sequence immediately after the recognised metric label.
    #
    # Examples accepted:
    #   Revenue 9,381 18,738 19,135
    #   Revenue (500) 750 1,200
    #   Revenue +100 -20 300
    #
    # Examples rejected:
    #   REVENUE BY PRODUCT — H1 2026
    #   Revenue grew 38% in H1 2026
    first_match = _NUMBER_PATTERN.match(
        remainder
    )

    if first_match is None:
        return None

    values = []

    for match in _NUMBER_PATTERN.finditer(
        remainder
    ):
        raw_number = (
            match.group("number")
            .replace(",", "")
        )

        try:
            value = Decimal(raw_number)
        except InvalidOperation:
            continue

        if (
            match.group("sign") == "-"
            or (
                match.group("open")
                and match.group("close")
            )
        ):
            value = -value

        values.append(value)

    if not values:
        return None

    return metric.key, values


def extract_reported_facts_from_lines(
    *,
    lines: list[str],
    periods: list[dict],
    document_id: str,
    document_name: str,
    page: int,
    company_name: str | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
    currency: str | None = None,
    unit: str | None = None,
) -> list[FinancialFact]:
    """
    Convert table-like financial statement rows into grounded
    FinancialFact objects.

    The caller supplies the period columns explicitly. This is
    deliberate: the extractor must never guess which numeric
    column belongs to which reporting period.

    A metric row is accepted only when the number of extracted
    values exactly matches the number of supplied periods.
    """

    if not periods:
        raise ValueError(
            "At least one reporting period is required"
        )

    facts = []

    for line in lines:
        extracted = (
            extract_values_from_metric_line(
                line
            )
        )

        if extracted is None:
            continue

        metric_key, values = extracted

        if len(values) != len(periods):
            continue

        for value, period in zip(
            values,
            periods,
        ):
            facts.append(
                make_reported_fact(
                    metric_key=metric_key,
                    value=value,
                    document_id=document_id,
                    document_name=document_name,
                    page=page,
                    company_name=company_name,
                    ticker=ticker,
                    exchange=exchange,
                    currency=currency,
                    unit=unit,
                    fiscal_year=period.get(
                        "fiscal_year"
                    ),
                    fiscal_quarter=period.get(
                        "fiscal_quarter"
                    ),
                    fiscal_half=period.get(
                        "fiscal_half"
                    ),
                    period_basis=period.get(
                        "period_basis"
                    ),
                    evidence_text=line,
                )
            )

    return facts
