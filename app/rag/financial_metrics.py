from dataclasses import dataclass
from typing import Literal


MetricKind = Literal[
    "reported",
    "derived",
]


@dataclass(frozen=True)
class FinancialMetric:
    """
    Canonical definition of a financial metric used by EquityAI.

    Reported metrics are expected to come directly from
    financial-report evidence.

    Derived metrics are calculated deterministically from
    grounded reported metrics.
    """

    key: str
    label: str
    kind: MetricKind
    aliases: tuple[str, ...]
    required_inputs: tuple[str, ...] = ()
    formula: str | None = None
    output_unit: str | None = None


FINANCIAL_METRICS: dict[str, FinancialMetric] = {
    # =====================================================
    # INCOME STATEMENT
    # =====================================================

    "revenue": FinancialMetric(
        key="revenue",
        label="Revenue",
        kind="reported",
        aliases=(
            "revenue",
            "sales",
            "turnover",
            "total revenue",
        ),
    ),

    "cost_of_sales": FinancialMetric(
        key="cost_of_sales",
        label="Cost of sales",
        kind="reported",
        aliases=(
            "cost of sales",
            "cost of revenue",
            "cost of goods sold",
            "cogs",
        ),
    ),

    "gross_profit": FinancialMetric(
        key="gross_profit",
        label="Gross profit",
        kind="reported",
        aliases=(
            "gross profit",
            "gross income",
        ),
    ),

    "operating_expenses": FinancialMetric(
        key="operating_expenses",
        label="Operating expenses",
        kind="reported",
        aliases=(
            "operating expenses",
            "operating expense",
            "opex",
        ),
    ),

    "operating_profit": FinancialMetric(
        key="operating_profit",
        label="Operating profit",
        kind="reported",
        aliases=(
            "operating profit",
            "operating income",
            "ebit",
        ),
    ),

    "profit_before_tax": FinancialMetric(
        key="profit_before_tax",
        label="Profit before tax",
        kind="reported",
        aliases=(
            "profit before tax",
            "profit before taxation",
            "pre-tax profit",
            "pretax profit",
            "pbt",
        ),
    ),

    "profit_after_tax": FinancialMetric(
        key="profit_after_tax",
        label="Profit after tax",
        kind="reported",
        aliases=(
            "profit after tax",
            "profit after taxation",
            "net profit",
            "net income",
            "net earnings",
            "pat",
        ),
    ),

    # =====================================================
    # BALANCE SHEET
    # =====================================================

    "cash": FinancialMetric(
        key="cash",
        label="Cash and cash equivalents",
        kind="reported",
        aliases=(
            "cash",
            "cash and cash equivalents",
            "cash equivalents",
        ),
    ),

    "current_assets": FinancialMetric(
        key="current_assets",
        label="Current assets",
        kind="reported",
        aliases=(
            "current assets",
            "total current assets",
        ),
    ),

    "inventory": FinancialMetric(
        key="inventory",
        label="Inventory",
        kind="reported",
        aliases=(
            "inventory",
            "inventories",
            "stock",
        ),
    ),

    "current_liabilities": FinancialMetric(
        key="current_liabilities",
        label="Current liabilities",
        kind="reported",
        aliases=(
            "current liabilities",
            "total current liabilities",
        ),
    ),

    "total_assets": FinancialMetric(
        key="total_assets",
        label="Total assets",
        kind="reported",
        aliases=(
            "total assets",
        ),
    ),

    "total_liabilities": FinancialMetric(
        key="total_liabilities",
        label="Total liabilities",
        kind="reported",
        aliases=(
            "total liabilities",
        ),
    ),

    "total_debt": FinancialMetric(
        key="total_debt",
        label="Total debt",
        kind="reported",
        aliases=(
            "total debt",
            "borrowings",
            "total borrowings",
            "interest-bearing debt",
        ),
    ),

    "total_equity": FinancialMetric(
        key="total_equity",
        label="Total equity",
        kind="reported",
        aliases=(
            "total equity",
            "shareholders' equity",
            "shareholders equity",
            "stockholders' equity",
            "stockholders equity",
            "equity attributable to owners",
        ),
    ),

    # =====================================================
    # CASH FLOW
    # =====================================================

    "operating_cash_flow": FinancialMetric(
        key="operating_cash_flow",
        label="Operating cash flow",
        kind="reported",
        aliases=(
            "operating cash flow",
            "cash flow from operations",
            "cash flows from operating activities",
            "net cash from operating activities",
            "net cash generated from operating activities",
        ),
    ),

    "capital_expenditure": FinancialMetric(
        key="capital_expenditure",
        label="Capital expenditure",
        kind="reported",
        aliases=(
            "capital expenditure",
            "capital expenditures",
            "capex",
            "purchase of property plant and equipment",
            "purchases of property plant and equipment",
        ),
    ),

    # =====================================================
    # DERIVED PROFITABILITY METRICS
    # =====================================================

    "gross_profit_margin": FinancialMetric(
        key="gross_profit_margin",
        label="Gross profit margin",
        kind="derived",
        aliases=(
            "gross profit margin",
            "gross margin",
        ),
        required_inputs=(
            "gross_profit",
            "revenue",
        ),
        formula="gross_profit / revenue * 100",
        output_unit="%",
    ),

    "operating_profit_margin": FinancialMetric(
        key="operating_profit_margin",
        label="Operating profit margin",
        kind="derived",
        aliases=(
            "operating profit margin",
            "operating margin",
            "ebit margin",
        ),
        required_inputs=(
            "operating_profit",
            "revenue",
        ),
        formula="operating_profit / revenue * 100",
        output_unit="%",
    ),

    "net_profit_margin": FinancialMetric(
        key="net_profit_margin",
        label="Net profit margin",
        kind="derived",
        aliases=(
            "net profit margin",
            "net margin",
            "profit after tax margin",
        ),
        required_inputs=(
            "profit_after_tax",
            "revenue",
        ),
        formula="profit_after_tax / revenue * 100",
        output_unit="%",
    ),

    # =====================================================
    # DERIVED LIQUIDITY METRICS
    # =====================================================

    "current_ratio": FinancialMetric(
        key="current_ratio",
        label="Current ratio",
        kind="derived",
        aliases=(
            "current ratio",
        ),
        required_inputs=(
            "current_assets",
            "current_liabilities",
        ),
        formula="current_assets / current_liabilities",
        output_unit="x",
    ),

    "quick_ratio": FinancialMetric(
        key="quick_ratio",
        label="Quick ratio",
        kind="derived",
        aliases=(
            "quick ratio",
            "acid-test ratio",
            "acid test ratio",
        ),
        required_inputs=(
            "current_assets",
            "inventory",
            "current_liabilities",
        ),
        formula=(
            "(current_assets - inventory) "
            "/ current_liabilities"
        ),
        output_unit="x",
    ),

    "cash_ratio": FinancialMetric(
        key="cash_ratio",
        label="Cash ratio",
        kind="derived",
        aliases=(
            "cash ratio",
        ),
        required_inputs=(
            "cash",
            "current_liabilities",
        ),
        formula="cash / current_liabilities",
        output_unit="x",
    ),

    # =====================================================
    # DERIVED LEVERAGE METRICS
    # =====================================================

    "debt_to_equity": FinancialMetric(
        key="debt_to_equity",
        label="Debt-to-equity ratio",
        kind="derived",
        aliases=(
            "debt to equity",
            "debt-to-equity",
            "debt equity ratio",
            "debt/equity",
            "d/e ratio",
        ),
        required_inputs=(
            "total_debt",
            "total_equity",
        ),
        formula="total_debt / total_equity",
        output_unit="x",
    ),

    "debt_to_assets": FinancialMetric(
        key="debt_to_assets",
        label="Debt-to-assets ratio",
        kind="derived",
        aliases=(
            "debt to assets",
            "debt-to-assets",
            "debt assets ratio",
        ),
        required_inputs=(
            "total_debt",
            "total_assets",
        ),
        formula="total_debt / total_assets",
        output_unit="x",
    ),

    # =====================================================
    # DERIVED CASH-FLOW METRICS
    # =====================================================

    "free_cash_flow": FinancialMetric(
        key="free_cash_flow",
        label="Free cash flow",
        kind="derived",
        aliases=(
            "free cash flow",
            "fcf",
        ),
        required_inputs=(
            "operating_cash_flow",
            "capital_expenditure",
        ),
        formula=(
            "operating_cash_flow - capital_expenditure"
        ),
    ),

    "operating_cash_flow_margin": FinancialMetric(
        key="operating_cash_flow_margin",
        label="Operating cash flow margin",
        kind="derived",
        aliases=(
            "operating cash flow margin",
            "cash flow margin",
            "ocf margin",
        ),
        required_inputs=(
            "operating_cash_flow",
            "revenue",
        ),
        formula="operating_cash_flow / revenue * 100",
        output_unit="%",
    ),
}


METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "profitability": (
        "gross_profit_margin",
        "operating_profit_margin",
        "net_profit_margin",
    ),
    "liquidity": (
        "current_ratio",
        "quick_ratio",
        "cash_ratio",
    ),
    "leverage": (
        "debt_to_equity",
        "debt_to_assets",
    ),
    "cash_flow": (
        "operating_cash_flow",
        "free_cash_flow",
        "operating_cash_flow_margin",
    ),
}


def get_metric(
    key: str,
) -> FinancialMetric:
    """
    Return a canonical financial metric by key.
    """

    normalized_key = key.strip().lower()

    try:
        return FINANCIAL_METRICS[normalized_key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown financial metric: {key}"
        ) from exc


def resolve_metric(
    text: str,
) -> FinancialMetric | None:
    """
    Resolve a metric from an exact canonical key, label,
    or known alias.

    This intentionally avoids fuzzy matching.
    """

    normalized = " ".join(
        text.strip().lower().split()
    )

    if not normalized:
        return None

    normalized_key = normalized.replace(
        " ",
        "_",
    ).replace(
        "-",
        "_",
    )

    if normalized_key in FINANCIAL_METRICS:
        return FINANCIAL_METRICS[normalized_key]

    for metric in FINANCIAL_METRICS.values():
        candidates = (
            metric.label.lower(),
            *(
                alias.lower()
                for alias in metric.aliases
            ),
        )

        if normalized in candidates:
            return metric

    return None


def get_metric_group(
    group: str,
) -> tuple[FinancialMetric, ...]:
    """
    Return all canonical metrics belonging to a group.
    """

    normalized_group = group.strip().lower()

    try:
        keys = METRIC_GROUPS[normalized_group]
    except KeyError as exc:
        raise KeyError(
            f"Unknown financial metric group: {group}"
        ) from exc

    return tuple(
        FINANCIAL_METRICS[key]
        for key in keys
    )
