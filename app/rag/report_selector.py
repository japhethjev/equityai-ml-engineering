import re
from dataclasses import dataclass


@dataclass
class ReportSelection:
    """
    Structured financial-report criteria extracted from a question.
    """

    report_type: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    fiscal_half: int | None = None
    latest: bool = False


def parse_report_selection(
    question: str,
) -> ReportSelection:
    """
    Extract explicit financial reporting-period intent.

    Supported examples:
        FY2025
        FY 2025
        full year 2025
        annual report 2025
        Q1 2025
        Q4 FY2025
        quarter 2 2026
        H1 2025
        H2 FY2026
        half year 2025
        latest results
        latest report

    Company/ticker resolution is deliberately handled separately.
    """

    text = " ".join(
        question.strip().lower().split()
    )

    latest = bool(
        re.search(
            r"\b("
            r"latest|most recent|current"
            r")\b",
            text,
        )
    )

    # -------------------------------------------------
    # Quarter
    # -------------------------------------------------

    quarter_match = re.search(
        r"\bq([1-4])\b",
        text,
    )

    if not quarter_match:
        quarter_match = re.search(
            r"\bquarter\s*([1-4])\b",
            text,
        )

    fiscal_quarter = (
        int(quarter_match.group(1))
        if quarter_match
        else None
    )

    # -------------------------------------------------
    # Half year
    # -------------------------------------------------

    half_match = re.search(
        r"\bh([12])\b",
        text,
    )

    if not half_match:
        half_match = re.search(
            r"\bhalf(?:[-\s]?year)?\s*([12])\b",
            text,
        )

    fiscal_half = (
        int(half_match.group(1))
        if half_match
        else None
    )

    # -------------------------------------------------
    # Fiscal year
    # -------------------------------------------------

    year_match = re.search(
        r"\bfy\s*(20\d{2})\b",
        text,
    )

    if not year_match:
        year_match = re.search(
            r"\b(20\d{2})\b",
            text,
        )

    fiscal_year = (
        int(year_match.group(1))
        if year_match
        else None
    )

    # -------------------------------------------------
    # Report type
    # -------------------------------------------------

    report_type = None

    if fiscal_quarter is not None:
        report_type = "quarterly"

    elif fiscal_half is not None:
        report_type = "half_year"

    elif re.search(
        r"\b("
        r"quarterly|quarter[-\s]?report"
        r")\b",
        text,
    ):
        report_type = "quarterly"

    elif re.search(
        r"\b("
        r"half[-\s]?year|half[-\s]?yearly|"
        r"interim"
        r")\b",
        text,
    ):
        report_type = "half_year"

    elif re.search(
        r"\b("
        r"annual|full[-\s]?year|year[-\s]?end"
        r")\b",
        text,
    ):
        report_type = "annual"

    return ReportSelection(
        report_type=report_type,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        fiscal_half=fiscal_half,
        latest=latest,
    )


def parse_report_selections(
    question: str,
) -> list[ReportSelection]:
    """
    Extract one or more financial reporting periods.

    Comparison examples:
        Compare Q2 2025 with Q2 2026
        Compare Q2 2025 and 2026
        Compare FY2024 and FY2025
        Compare H1 2025 with H1 2026

    For ordinary single-period questions this returns a
    one-element list containing parse_report_selection().
    """

    text = " ".join(
        question.strip().lower().split()
    )

    selections: list[ReportSelection] = []

    # -------------------------------------------------
    # Explicit quarter + year pairs
    # Q2 2025 ... Q2 2026
    # -------------------------------------------------

    quarter_pairs = re.findall(
        r"\bq([1-4])\s+(?:fy\s*)?(20\d{2})\b",
        text,
    )

    if len(quarter_pairs) >= 2:
        for quarter, year in quarter_pairs:
            selections.append(
                ReportSelection(
                    report_type="quarterly",
                    fiscal_year=int(year),
                    fiscal_quarter=int(quarter),
                )
            )

        return selections

    # -------------------------------------------------
    # One quarter followed by multiple years
    # Q2 2025 and 2026
    # -------------------------------------------------

    quarter_match = re.search(
        r"\bq([1-4])\b",
        text,
    )

    years = list(
        dict.fromkeys(
            re.findall(
                r"\b(?:fy\s*)?(20\d{2})\b",
                text,
            )
        )
    )

    if quarter_match and len(years) >= 2:
        quarter = int(
            quarter_match.group(1)
        )

        for year in years:
            selections.append(
                ReportSelection(
                    report_type="quarterly",
                    fiscal_year=int(year),
                    fiscal_quarter=quarter,
                )
            )

        return selections

    # -------------------------------------------------
    # Explicit half-year + year pairs
    # H1 2025 ... H1 2026
    # -------------------------------------------------

    half_pairs = re.findall(
        r"\bh([12])\s+(?:fy\s*)?(20\d{2})\b",
        text,
    )

    if len(half_pairs) >= 2:
        for half, year in half_pairs:
            selections.append(
                ReportSelection(
                    report_type="half_year",
                    fiscal_year=int(year),
                    fiscal_half=int(half),
                )
            )

        return selections

    # -------------------------------------------------
    # One half followed by multiple years
    # H1 2025 and 2026
    # -------------------------------------------------

    half_match = re.search(
        r"\bh([12])\b",
        text,
    )

    if half_match and len(years) >= 2:
        half = int(
            half_match.group(1)
        )

        for year in years:
            selections.append(
                ReportSelection(
                    report_type="half_year",
                    fiscal_year=int(year),
                    fiscal_half=half,
                )
            )

        return selections

    # -------------------------------------------------
    # Multiple explicit fiscal years
    # FY2024 and FY2025
    # -------------------------------------------------

    fiscal_years = re.findall(
        r"\bfy\s*(20\d{2})\b",
        text,
    )

    if len(fiscal_years) >= 2:
        for year in fiscal_years:
            selections.append(
                ReportSelection(
                    report_type="annual",
                    fiscal_year=int(year),
                )
            )

        return selections

    # -------------------------------------------------
    # Full-year / annual + multiple calendar years
    # annual 2024 and 2025
    # -------------------------------------------------

    if (
        len(years) >= 2
        and re.search(
            r"\b("
            r"annual|full[-\s]?year|year[-\s]?end"
            r")\b",
            text,
        )
    ):
        for year in years:
            selections.append(
                ReportSelection(
                    report_type="annual",
                    fiscal_year=int(year),
                )
            )

        return selections

    return [
        parse_report_selection(question)
    ]
