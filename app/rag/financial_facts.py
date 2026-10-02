from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


PeriodBasis = Literal[
    "annual",
    "quarterly",
    "half_year",
    "year_to_date",
]

FactSource = Literal[
    "reported",
    "derived",
]


@dataclass(frozen=True)
class FinancialFact:
    """
    A financial value grounded in a registered document.

    Every reported value must retain enough provenance to trace
    the fact back to the source financial report.

    Derived values are added later by the deterministic financial
    analytics layer and must retain their input provenance.
    """

    metric_key: str
    value: Decimal
    document_id: str
    document_name: str
    page: int

    company_name: str | None = None
    ticker: str | None = None
    exchange: str | None = None

    currency: str | None = None
    unit: str | None = None

    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    fiscal_half: int | None = None

    period_basis: PeriodBasis | None = None
    source: FactSource = "reported"

    evidence_text: str | None = None


def make_reported_fact(
    *,
    metric_key: str,
    value: Decimal | int | float | str,
    document_id: str,
    document_name: str,
    page: int,
    company_name: str | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
    currency: str | None = None,
    unit: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
    fiscal_half: int | None = None,
    period_basis: PeriodBasis | None = None,
    evidence_text: str | None = None,
) -> FinancialFact:
    """
    Construct a reported financial fact.

    Decimal is used instead of binary floating-point so financial
    calculations can later be performed deterministically.
    """

    normalized_metric = metric_key.strip().lower()

    if not normalized_metric:
        raise ValueError(
            "metric_key is required"
        )

    normalized_document_id = (
        document_id.strip()
    )

    if not normalized_document_id:
        raise ValueError(
            "document_id is required"
        )

    normalized_document_name = (
        document_name.strip()
    )

    if not normalized_document_name:
        raise ValueError(
            "document_name is required"
        )

    if page <= 0:
        raise ValueError(
            "page must be greater than zero"
        )

    decimal_value = _to_decimal(
        value
    )

    return FinancialFact(
        metric_key=normalized_metric,
        value=decimal_value,
        document_id=normalized_document_id,
        document_name=normalized_document_name,
        page=page,
        company_name=company_name,
        ticker=ticker,
        exchange=exchange,
        currency=currency,
        unit=unit,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        fiscal_half=fiscal_half,
        period_basis=period_basis,
        source="reported",
        evidence_text=evidence_text,
    )


def _to_decimal(
    value: Decimal | int | float | str,
) -> Decimal:
    """
    Convert supported numeric input to Decimal safely.

    Floats are converted through their string representation
    rather than directly to avoid binary floating-point noise.
    """

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        raise ValueError(
            "Financial fact value must be numeric"
        )

    try:
        return Decimal(str(value))
    except Exception as exc:
        raise ValueError(
            "Financial fact value must be numeric"
        ) from exc


def period_label(
    fact: FinancialFact,
) -> str:
    """
    Produce a deterministic human-readable reporting-period label.
    """

    if (
        fact.fiscal_year is not None
        and fact.fiscal_quarter is not None
    ):
        return (
            f"Q{fact.fiscal_quarter} "
            f"FY{fact.fiscal_year}"
        )

    if (
        fact.fiscal_year is not None
        and fact.fiscal_half is not None
    ):
        return (
            f"H{fact.fiscal_half} "
            f"FY{fact.fiscal_year}"
        )

    if fact.fiscal_year is not None:
        return f"FY{fact.fiscal_year}"

    return "Unspecified period"


def same_reporting_period(
    left: FinancialFact,
    right: FinancialFact,
) -> bool:
    """
    Return True when two facts refer to the same fiscal period.

    This control will later prevent calculations such as dividing
    FY2025 profit by FY2024 revenue.
    """

    return (
        left.fiscal_year
        == right.fiscal_year
        and left.fiscal_quarter
        == right.fiscal_quarter
        and left.fiscal_half
        == right.fiscal_half
        and left.period_basis
        == right.period_basis
    )
