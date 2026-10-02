import re


_YEAR = r"(?:19|20)\d{2}"

_ANNUAL_TOKEN = re.compile(
    rf"(?<!\w)(?:FY\s*)?(?P<year>{_YEAR})(?!\w)",
    re.IGNORECASE,
)

_QUARTER_TOKEN = re.compile(
    rf"(?<!\w)Q(?P<quarter>[1-4])"
    rf"(?:\s*(?:FY)?\s*(?P<year>{_YEAR}))?(?!\w)",
    re.IGNORECASE,
)

_HALF_TOKEN = re.compile(
    rf"(?<!\w)H(?P<half>[12])"
    rf"(?:\s*(?:FY)?\s*(?P<year>{_YEAR}))?(?!\w)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(
        text.replace("|", " ").split()
    )


def _period_from_match(
    *,
    year: int,
    quarter: int | None = None,
    half: int | None = None,
) -> dict:
    if quarter is not None:
        return {
            "fiscal_year": year,
            "fiscal_quarter": quarter,
            "period_basis": "quarterly",
        }

    if half is not None:
        return {
            "fiscal_year": year,
            "fiscal_half": half,
            "period_basis": "half_year",
        }

    return {
        "fiscal_year": year,
        "period_basis": "annual",
    }


def parse_period_header(
    line: str,
) -> list[dict]:
    """
    Parse an explicit financial-table period header.

    Supported examples include:
        2024 2025 H1 2026
        FY2023 FY2024 FY2025
        Q1 2026 Q2 2026 Q3 2026
        Q4 2025 Q1 2026

    The function is intentionally conservative. It returns an
    empty list when period ownership cannot be established
    explicitly from the header.

    A bare year is annual. Qn/Hn must be associated with an
    explicit year either immediately following the token or,
    for forms such as "2024 2025 H1 2026", through their own
    explicit trailing year.
    """

    text = _normalize(line)

    if not text:
        return []

    occupied: list[tuple[int, int]] = []
    tokens: list[tuple[int, dict]] = []

    # Quarter periods first so their years are not subsequently
    # interpreted as independent annual columns.
    for match in _QUARTER_TOKEN.finditer(text):
        year_text = match.group("year")

        if year_text is None:
            continue

        tokens.append(
            (
                match.start(),
                _period_from_match(
                    year=int(year_text),
                    quarter=int(
                        match.group("quarter")
                    ),
                ),
            )
        )
        occupied.append(match.span())

    # Half-year periods.
    for match in _HALF_TOKEN.finditer(text):
        year_text = match.group("year")

        if year_text is None:
            continue

        tokens.append(
            (
                match.start(),
                _period_from_match(
                    year=int(year_text),
                    half=int(
                        match.group("half")
                    ),
                ),
            )
        )
        occupied.append(match.span())

    def overlaps(
        start: int,
        end: int,
    ) -> bool:
        return any(
            start < occupied_end
            and end > occupied_start
            for occupied_start, occupied_end
            in occupied
        )

    # Remaining standalone years represent annual columns.
    for match in _ANNUAL_TOKEN.finditer(text):
        if overlaps(
            match.start(),
            match.end(),
        ):
            continue

        tokens.append(
            (
                match.start(),
                _period_from_match(
                    year=int(
                        match.group("year")
                    ),
                ),
            )
        )

    tokens.sort(
        key=lambda item: item[0]
    )

    periods = [
        period
        for _, period in tokens
    ]

    # A financial table header needs at least two periods.
    # This avoids treating arbitrary single dates in narrative
    # text as table headers.
    if len(periods) < 2:
        return []

    return periods


def find_period_header(
    lines: list[str],
) -> tuple[int, list[dict]] | None:
    """
    Return the first line that can safely be interpreted as a
    multi-period financial table header.
    """

    for index, line in enumerate(lines):
        periods = parse_period_header(line)

        if periods:
            return index, periods

    return None
