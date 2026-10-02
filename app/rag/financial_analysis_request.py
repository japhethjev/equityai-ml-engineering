import re
from dataclasses import dataclass


# =========================================================
# REQUEST MODEL
# =========================================================


@dataclass(frozen=True)
class FinancialAnalysisRequest:
    """
    Structured investor-analysis request derived from a user's
    natural-language question.
    """

    is_financial_analysis: bool
    metric_keys: tuple[str, ...]
    period_type: str | None
    period_count: int | None
    comparison_requested: bool
    historical_requested: bool


# =========================================================
# METRIC VOCABULARY
# =========================================================


METRIC_PHRASES = {
    # Income statement
    "revenue": (
        "revenue",
        "sales",
        "turnover",
    ),
    "cost_of_sales": (
        "cost of sales",
        "cost of revenue",
    ),
    "gross_profit": (
        "gross profit",
    ),
    "operating_expenses": (
        "operating expenses",
        "operating expense",
        "opex",
    ),
    "operating_profit": (
        "operating profit",
        "operating income",
    ),
    "profit_before_tax": (
        "profit before tax",
        "pre-tax profit",
        "pretax profit",
        "pbt",
    ),
    "profit_after_tax": (
        "profit after tax",
        "net profit",
        "net income",
        "pat",
    ),

    # Margins
    "gross_profit_margin": (
        "gross profit margin",
        "gross margin",
    ),
    "operating_profit_margin": (
        "operating profit margin",
        "operating margin",
    ),
    "net_profit_margin": (
        "net profit margin",
        "net margin",
    ),

    # Cash flow
    "operating_cash_flow": (
        "operating cash flow",
        "operating cashflow",
        "cash flow from operations",
        "cashflow from operations",
        "cash generated from operations",
    ),
    "free_cash_flow": (
        "free cash flow",
        "free cashflow",
        "fcf",
    ),

    # Liquidity
    "current_ratio": (
        "current ratio",
    ),
    "quick_ratio": (
        "quick ratio",
        "acid test ratio",
        "acid-test ratio",
    ),

    # Leverage
    "debt_to_equity": (
        "debt to equity",
        "debt-to-equity",
        "debt equity ratio",
        "debt/equity",
        "d/e ratio",
    ),
}


METRIC_GROUPS = {
    "liquidity ratios": (
        "current_ratio",
        "quick_ratio",
    ),
    "liquidity ratio": (
        "current_ratio",
        "quick_ratio",
    ),
    "profit margins": (
        "gross_profit_margin",
        "operating_profit_margin",
        "net_profit_margin",
    ),
    "profitability margins": (
        "gross_profit_margin",
        "operating_profit_margin",
        "net_profit_margin",
    ),
    "margins": (
        "gross_profit_margin",
        "operating_profit_margin",
        "net_profit_margin",
    ),
}


# =========================================================
# NORMALISATION
# =========================================================


def _normalize(text: str) -> str:
    text = text.lower()

    text = re.sub(
        r"[^\w\s/%-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# METRIC EXTRACTION
# =========================================================


def extract_metric_keys(
    question: str,
) -> tuple[str, ...]:
    """
    Extract explicitly requested investor KPIs.

    Longer phrases receive matching priority so, for example,
    'gross profit margin' is not also interpreted as
    'gross profit'.

    Final metrics are returned in the order in which the user
    requested them.
    """

    normalized = _normalize(question)

    matches = []

    # -----------------------------------------------------
    # Metric groups
    # -----------------------------------------------------

    for phrase, metric_keys in METRIC_GROUPS.items():
        for match in re.finditer(
            r"(?<!\\w)"
            + re.escape(phrase)
            + r"(?!\\w)",
            normalized,
        ):
            for group_position, metric_key in enumerate(
                metric_keys
            ):
                matches.append(
                    (
                        match.start(),
                        group_position,
                        metric_key,
                        match.start(),
                        match.end(),
                        len(phrase),
                    )
                )

    # -----------------------------------------------------
    # Individual metric phrases
    # -----------------------------------------------------

    candidates = []

    for metric_key, phrases in METRIC_PHRASES.items():
        for phrase in phrases:
            pattern = re.compile(
                r"(?<!\\w)"
                + re.escape(phrase)
                + r"(?!\\w)"
            )

            for match in pattern.finditer(
                normalized
            ):
                candidates.append(
                    (
                        match.start(),
                        match.end(),
                        len(phrase),
                        metric_key,
                    )
                )

    # Longest phrase wins when phrases overlap at the
    # same location.
    candidates.sort(
        key=lambda item: (
            item[0],
            -item[2],
        )
    )

    accepted_spans = []

    for (
        start,
        end,
        phrase_length,
        metric_key,
    ) in candidates:

        overlaps = any(
            start < occupied_end
            and end > occupied_start
            for (
                occupied_start,
                occupied_end,
            ) in accepted_spans
        )

        if overlaps:
            continue

        accepted_spans.append(
            (
                start,
                end,
            )
        )

        matches.append(
            (
                start,
                0,
                metric_key,
                start,
                end,
                phrase_length,
            )
        )

    # -----------------------------------------------------
    # Restore user-requested order
    # -----------------------------------------------------

    matches.sort(
        key=lambda item: (
            item[0],
            item[1],
            -item[5],
        )
    )

    result = []
    seen = set()

    for match in matches:
        metric_key = match[2]

        if metric_key in seen:
            continue

        seen.add(metric_key)
        result.append(metric_key)

    return tuple(result)


# =========================================================
# PERIOD EXTRACTION
# =========================================================


def extract_period_request(
    question: str,
) -> tuple[str | None, int | None]:
    """
    Extract requests such as:

        last 5 years
        past 3 years
        previous 8 quarters
        last four quarters

    Numeric words are supported for common investor horizons.
    """

    normalized = _normalize(question)

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    number_pattern = (
        r"(\d+|"
        + "|".join(number_words)
        + r")"
    )

    annual_pattern = re.search(
        r"\b(?:last|past|previous)\s+"
        + number_pattern
        + r"\s+(?:financial\s+)?years?\b",
        normalized,
    )

    if annual_pattern:
        raw_count = annual_pattern.group(1)

        count = (
            int(raw_count)
            if raw_count.isdigit()
            else number_words[raw_count]
        )

        return (
            "annual",
            count,
        )

    quarterly_pattern = re.search(
        r"\b(?:last|past|previous)\s+"
        + number_pattern
        + r"\s+quarters?\b",
        normalized,
    )

    if quarterly_pattern:
        raw_count = quarterly_pattern.group(1)

        count = (
            int(raw_count)
            if raw_count.isdigit()
            else number_words[raw_count]
        )

        return (
            "quarterly",
            count,
        )

    return (
        None,
        None,
    )


# =========================================================
# INTENT DETECTION
# =========================================================


COMPARISON_PHRASES = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "against",
    "across companies",
    "across these companies",
)


ANALYSIS_TERMS = (
    "financial performance",
    "financial analysis",
    "trend",
    "trends",
    "historical",
    "history",
    "kpi",
    "kpis",
    "ratio",
    "ratios",
    "margin",
    "margins",
    "cash flow",
    "cashflow",
)


def explicitly_requests_comparison(
    question: str,
) -> bool:
    normalized = (
        " "
        + _normalize(question)
        + " "
    )

    return any(
        phrase in normalized
        for phrase in COMPARISON_PHRASES
    )


def parse_financial_analysis_request(
    question: str,
) -> FinancialAnalysisRequest:
    """
    Convert a natural-language investor question into a
    deterministic financial-analysis request specification.

    This function does not resolve companies or documents.
    Those remain responsibilities of the retrieval layer.
    """

    metric_keys = extract_metric_keys(
        question
    )

    (
        period_type,
        period_count,
    ) = extract_period_request(
        question
    )

    comparison_requested = (
        explicitly_requests_comparison(
            question
        )
    )

    normalized = _normalize(
        question
    )

    historical_requested = (
        period_count is not None
        or any(
            term in normalized
            for term in (
                "trend",
                "trends",
                "historical",
                "history",
                "over time",
                "across periods",
            )
        )
    )

    has_analysis_term = any(
        term in normalized
        for term in ANALYSIS_TERMS
    )

    is_financial_analysis = bool(
        metric_keys
        and (
            historical_requested
            or comparison_requested
            or has_analysis_term
            or len(metric_keys) > 1
        )
    )

    return FinancialAnalysisRequest(
        is_financial_analysis=(
            is_financial_analysis
        ),
        metric_keys=metric_keys,
        period_type=period_type,
        period_count=period_count,
        comparison_requested=(
            comparison_requested
        ),
        historical_requested=(
            historical_requested
        ),
    )
