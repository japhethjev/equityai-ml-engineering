from dataclasses import dataclass
from decimal import Decimal

from app.rag.financial_analysis import (
    AnalysisValue,
    LatestChange,
    calculate_latest_change,
)
from app.rag.financial_metrics import get_metric


# =========================================================
# PRESENTATION MODELS
# =========================================================


@dataclass(frozen=True)
class CompanyAnalysis:
    """
    Validated financial analysis values for one company.

    Values must already have been extracted or calculated by
    the grounded financial-analysis pipeline.
    """

    company_name: str
    ticker: str | None
    values: tuple[AnalysisValue, ...]


@dataclass(frozen=True)
class PresentationResult:
    """
    Deterministic financial presentation result.

    mode:
        single_company_single_period
        single_company_multi_period
        multi_company_single_period
        multi_company_multi_period
    """

    mode: str
    text: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


# =========================================================
# VALUE FORMATTING
# =========================================================


def _format_decimal(
    value: Decimal,
    decimal_places: int = 2,
) -> str:
    """
    Format a Decimal without converting through float.
    """

    quantizer = Decimal("1").scaleb(
        -decimal_places
    )

    value = value.quantize(
        quantizer
    )

    return f"{value:,.{decimal_places}f}"


def format_analysis_value(
    value: AnalysisValue,
) -> str:
    """
    Format one validated KPI value.

    Percentage KPIs retain the percentage symbol.
    Ratio KPIs retain their ratio representation.
    Monetary/absolute values retain their supplied unit
    separately from the numeric amount.
    """

    metric = get_metric(
        value.metric_key
    )

    if metric.output_unit == "%":
        return (
            f"{_format_decimal(value.value)}%"
        )

    if metric.output_unit == "x":
        return (
            f"{_format_decimal(value.value)}x"
        )

    return _format_decimal(
        value.value
    )


def format_latest_change(
    change: LatestChange,
) -> str:
    """
    Format the final YoY/QoQ column.

    Zero-base percentage changes remain unavailable because
    percentage growth from zero is mathematically undefined.
    """

    if change.value is None:
        return "—"

    prefix = (
        "+"
        if change.value > 0
        else ""
    )

    formatted = _format_decimal(
        change.value
    )

    if change.display_unit == "pp":
        return f"{prefix}{formatted} pp"

    if change.display_unit == "%":
        return f"{prefix}{formatted}%"

    return f"{prefix}{formatted}"


# =========================================================
# PERIOD ORDERING
# =========================================================


def _period_sort_key(
    value: AnalysisValue,
) -> tuple:
    """
    Obtain chronological ordering from the underlying
    FinancialFact rather than parsing display labels.
    """

    fact = value.input_facts[0]

    year = (
        fact.fiscal_year
        if fact.fiscal_year is not None
        else -1
    )

    if fact.fiscal_quarter is not None:
        return (
            year,
            1,
            fact.fiscal_quarter,
        )

    if fact.fiscal_half is not None:
        return (
            year,
            2,
            fact.fiscal_half,
        )

    return (
        year,
        3,
        0,
    )


def _ordered_period_labels(
    companies: list[CompanyAnalysis],
) -> list[str]:
    """
    Return unique periods in chronological order.
    """

    representatives = {}

    for company in companies:
        for value in company.values:
            label = value.period_label

            if label not in representatives:
                representatives[label] = value

    return [
        label
        for label, _ in sorted(
            representatives.items(),
            key=lambda item: _period_sort_key(
                item[1]
            ),
        )
    ]


def _period_identity(
    value: AnalysisValue,
) -> tuple[str, int, int | None]:
    """
    Return the authoritative financial-period identity.

    Display labels are deliberately not parsed.
    """

    fact = value.input_facts[0]

    if fact.fiscal_year is None:
        return (
            "unknown",
            -1,
            None,
        )

    if fact.fiscal_quarter is not None:
        return (
            "quarterly",
            fact.fiscal_year,
            fact.fiscal_quarter,
        )

    if fact.fiscal_half is not None:
        return (
            "half_year",
            fact.fiscal_year,
            fact.fiscal_half,
        )

    return (
        "annual",
        fact.fiscal_year,
        None,
    )


def are_consecutive_periods(
    previous: AnalysisValue,
    current: AnalysisValue,
) -> bool:
    """
    Return True only for genuinely consecutive periods.

    Annual:
        FY2024 -> FY2025

    Quarterly:
        Q1 -> Q2
        Q2 -> Q3
        Q3 -> Q4
        Q4 FY2024 -> Q1 FY2025

    Half-year:
        H1 -> H2
        H2 FY2024 -> H1 FY2025
    """

    (
        previous_type,
        previous_year,
        previous_subperiod,
    ) = _period_identity(previous)

    (
        current_type,
        current_year,
        current_subperiod,
    ) = _period_identity(current)

    if previous_type != current_type:
        return False

    if previous_type == "annual":
        return (
            current_year
            == previous_year + 1
        )

    if previous_type == "quarterly":
        previous_index = (
            previous_year * 4
            + int(previous_subperiod)
        )

        current_index = (
            current_year * 4
            + int(current_subperiod)
        )

        return (
            current_index
            == previous_index + 1
        )

    if previous_type == "half_year":
        previous_index = (
            previous_year * 2
            + int(previous_subperiod)
        )

        current_index = (
            current_year * 2
            + int(current_subperiod)
        )

        return (
            current_index
            == previous_index + 1
        )

    return False


def _latest_consecutive_change(
    period_values: list[
        AnalysisValue | None
    ],
) -> LatestChange | None:
    """
    Calculate the final YoY/QoQ movement only from the final
    two displayed periods.

    Missing periods are never skipped.

    Example:

        FY2023 | FY2024 | FY2025 | YoY %
        100    | —      | 125    | —

    The engine must not compare FY2023 directly with FY2025
    and describe that movement as YoY.
    """

    if len(period_values) < 2:
        return None

    previous = period_values[-2]
    current = period_values[-1]

    if (
        previous is None
        or current is None
    ):
        return None

    if not are_consecutive_periods(
        previous,
        current,
    ):
        return None

    return calculate_latest_change(
        previous,
        current,
    )


def _change_header(
    period_labels: list[str],
) -> str:
    """
    Select the final change-column label.

    All-quarter history -> QoQ %
    Otherwise -> YoY %
    """

    if (
        period_labels
        and all(
            label.startswith("Q")
            for label in period_labels
        )
    ):
        return "QoQ %"

    return "YoY %"


# =========================================================
# KPI ORDERING
# =========================================================


PREFERRED_KPI_ORDER = (
    "revenue",
    "cost_of_sales",
    "gross_profit",
    "gross_profit_margin",
    "operating_expenses",
    "operating_profit",
    "operating_profit_margin",
    "profit_before_tax",
    "profit_after_tax",
    "net_profit_margin",
    "operating_cash_flow",
    "free_cash_flow",
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
)


def _metric_label(
    metric_key: str,
) -> str:
    """
    Return an investor-facing KPI label.

    Metric-registry labels remain canonical internally;
    presentation labels use title case for tables and prose.
    """
    return get_metric(
        metric_key
    ).label.title()


def _ordered_metric_keys(
    companies: list[CompanyAnalysis],
) -> list[str]:
    available = {
        value.metric_key
        for company in companies
        for value in company.values
    }

    ordered = [
        metric
        for metric in PREFERRED_KPI_ORDER
        if metric in available
    ]

    remaining = sorted(
        available.difference(ordered),
        key=_metric_label,
    )

    return ordered + remaining


# =========================================================
# LOOKUPS
# =========================================================


def _value_map(
    company: CompanyAnalysis,
) -> dict[tuple[str, str], AnalysisValue]:
    """
    Map (metric, period) to a validated analysis value.

    Duplicate metric/period combinations are rejected rather
    than silently choosing one source.
    """

    result = {}

    for value in company.values:
        key = (
            value.metric_key,
            value.period_label,
        )

        if key in result:
            raise ValueError(
                "Duplicate analysis value for "
                f"{value.metric_key} "
                f"{value.period_label}"
            )

        result[key] = value

    return result


def _company_display_name(
    company: CompanyAnalysis,
) -> str:
    if company.ticker:
        return (
            f"{company.company_name} "
            f"({company.ticker})"
        )

    return company.company_name


# =========================================================
# MARKDOWN RENDERING
# =========================================================


def _markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    header_line = (
        "| "
        + " | ".join(headers)
        + " |"
    )

    separator = (
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |"
    )

    body = [
        (
            "| "
            + " | ".join(row)
            + " |"
        )
        for row in rows
    ]

    return "\n".join(
        [
            header_line,
            separator,
            *body,
        ]
    )


# =========================================================
# SINGLE COMPANY + SINGLE PERIOD
# =========================================================


def _single_company_single_period(
    company: CompanyAnalysis,
) -> PresentationResult:
    metric_keys = _ordered_metric_keys(
        [company]
    )

    value_map = _value_map(
        company
    )

    periods = _ordered_period_labels(
        [company]
    )

    if len(periods) != 1:
        raise ValueError(
            "Single-period presentation requires "
            "exactly one period"
        )

    period = periods[0]

    statements = []

    for metric_key in metric_keys:
        value = value_map.get(
            (
                metric_key,
                period,
            )
        )

        if value is None:
            continue

        statements.append(
            f"{_metric_label(metric_key)} was "
            f"{format_analysis_value(value)}."
        )

    text = " ".join(statements)

    return PresentationResult(
        mode=(
            "single_company_single_period"
        ),
        text=text,
        headers=(),
        rows=(),
    )


# =========================================================
# SINGLE COMPANY + MULTIPLE PERIODS
# =========================================================


def _single_company_multi_period(
    company: CompanyAnalysis,
) -> PresentationResult:
    periods = _ordered_period_labels(
        [company]
    )

    metric_keys = _ordered_metric_keys(
        [company]
    )

    values = _value_map(
        company
    )

    change_header = _change_header(
        periods
    )

    headers = [
        "KPI",
        *periods,
        change_header,
    ]

    rows = []

    for metric_key in metric_keys:
        row = [
            _metric_label(metric_key)
        ]

        period_values = []

        for period in periods:
            value = values.get(
                (
                    metric_key,
                    period,
                )
            )

            period_values.append(
                value
            )

            row.append(
                format_analysis_value(value)
                if value is not None
                else "—"
            )

        change = _latest_consecutive_change(
            period_values
        )

        if change is not None:
            row.append(
                format_latest_change(
                    change
                )
            )
        else:
            row.append("—")

        rows.append(row)

    table = _markdown_table(
        headers,
        rows,
    )

    return PresentationResult(
        mode=(
            "single_company_multi_period"
        ),
        text=table,
        headers=tuple(headers),
        rows=tuple(
            tuple(row)
            for row in rows
        ),
    )


# =========================================================
# MULTIPLE COMPANIES + SINGLE PERIOD
# =========================================================


def _multi_company_single_period(
    companies: list[CompanyAnalysis],
) -> PresentationResult:
    periods = _ordered_period_labels(
        companies
    )

    if len(periods) != 1:
        raise ValueError(
            "Single-period peer comparison requires "
            "exactly one common period"
        )

    period = periods[0]

    metric_keys = _ordered_metric_keys(
        companies
    )

    maps = {
        _company_display_name(company):
            _value_map(company)
        for company in companies
    }

    company_names = [
        _company_display_name(company)
        for company in companies
    ]

    headers = [
        "KPI",
        *company_names,
    ]

    rows = []

    for metric_key in metric_keys:
        row = [
            _metric_label(metric_key)
        ]

        for company_name in company_names:
            value = maps[
                company_name
            ].get(
                (
                    metric_key,
                    period,
                )
            )

            row.append(
                format_analysis_value(value)
                if value is not None
                else "—"
            )

        rows.append(row)

    table = _markdown_table(
        headers,
        rows,
    )

    return PresentationResult(
        mode=(
            "multi_company_single_period"
        ),
        text=table,
        headers=tuple(headers),
        rows=tuple(
            tuple(row)
            for row in rows
        ),
    )


# =========================================================
# MULTIPLE COMPANIES + MULTIPLE PERIODS
# =========================================================


def _multi_company_multi_period(
    companies: list[CompanyAnalysis],
) -> PresentationResult:
    periods = _ordered_period_labels(
        companies
    )

    metric_keys = _ordered_metric_keys(
        companies
    )

    change_header = _change_header(
        periods
    )

    headers = [
        "Company / KPI",
        *periods,
        change_header,
    ]

    rows = []

    for company in companies:
        values = _value_map(
            company
        )

        company_name = (
            _company_display_name(
                company
            )
        )

        # Company grouping row
        rows.append(
            [
                f"**{company_name}**",
                *(
                    ""
                    for _ in periods
                ),
                "",
            ]
        )

        for metric_key in metric_keys:
            period_values = [
                values.get(
                    (
                        metric_key,
                        period,
                    )
                )
                for period in periods
            ]

            # Do not create completely empty KPI rows
            # for a company.
            if not any(
                value is not None
                for value in period_values
            ):
                continue

            row = [
                _metric_label(metric_key)
            ]

            for value in period_values:
                row.append(
                    format_analysis_value(value)
                    if value is not None
                    else "—"
                )

            change = _latest_consecutive_change(
                period_values
            )

            if change is not None:
                row.append(
                    format_latest_change(
                        change
                    )
                )
            else:
                row.append("—")

            rows.append(row)

    table = _markdown_table(
        headers,
        rows,
    )

    return PresentationResult(
        mode=(
            "multi_company_multi_period"
        ),
        text=table,
        headers=tuple(headers),
        rows=tuple(
            tuple(row)
            for row in rows
        ),
    )


# =========================================================
# PUBLIC PRESENTATION ROUTER
# =========================================================


def build_financial_presentation(
    companies: list[CompanyAnalysis],
) -> PresentationResult:
    """
    Select the deterministic EquityAI presentation contract.

    1 company + 1 period:
        prose

    1 company + multiple periods:
        KPI rows x period columns + final YoY/QoQ

    multiple companies + 1 period:
        KPI rows x company columns

    multiple companies + multiple periods:
        company groups containing KPI rows x period columns
        + final YoY/QoQ
    """

    if not companies:
        raise ValueError(
            "At least one company is required"
        )

    if any(
        not company.values
        for company in companies
    ):
        raise ValueError(
            "Each company requires at least one "
            "analysis value"
        )

    periods = _ordered_period_labels(
        companies
    )

    company_count = len(companies)
    period_count = len(periods)

    if (
        company_count == 1
        and period_count == 1
    ):
        return (
            _single_company_single_period(
                companies[0]
            )
        )

    if (
        company_count == 1
        and period_count > 1
    ):
        return (
            _single_company_multi_period(
                companies[0]
            )
        )

    if (
        company_count > 1
        and period_count == 1
    ):
        return (
            _multi_company_single_period(
                companies
            )
        )

    return _multi_company_multi_period(
        companies
    )
