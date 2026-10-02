import re


class AmbiguousCompanyError(ValueError):
    """Raised when company identity is not unique."""


# Corporate suffixes do not materially identify an issuer.
_CORPORATE_SUFFIXES = {
    "plc",
    "plc.",
    "limited",
    "ltd",
    "ltd.",
    "fze",
    "llc",
    "inc",
    "inc.",
    "corp",
    "corporation",
}


def _contains_token(
    text: str,
    token: str,
) -> bool:
    if not token:
        return False

    return bool(
        re.search(
            rf"(?<![A-Za-z0-9])"
            rf"{re.escape(token)}"
            rf"(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _normalize_tokens(
    value: str,
) -> tuple[str, ...]:
    """
    Normalize text into lowercase alphanumeric tokens.
    """

    return tuple(
        re.findall(
            r"[A-Za-z0-9]+",
            value.lower(),
        )
    )


def _meaningful_company_tokens(
    company_name: str,
) -> tuple[str, ...]:
    """
    Return company-name tokens excluding common legal suffixes.
    """

    return tuple(
        token
        for token in _normalize_tokens(
            company_name
        )
        if token not in _CORPORATE_SUFFIXES
    )


def _company_match_strength(
    question: str,
    company_name: str,
) -> int:
    """
    Measure how specifically the question identifies a company.

    Returns the length of the longest consecutive company-name
    phrase present in the question.

    Example:

        registered:
            Dangote Petroleum Refinery FZE

        question:
            Show Dangote Petroleum Refinery revenue

        strength:
            3

    A single token such as "Dangote" is retained for ambiguity
    detection, but must never defeat a more specific phrase.
    """

    question_tokens = _normalize_tokens(
        question
    )

    company_tokens = (
        _meaningful_company_tokens(
            company_name
        )
    )

    if not question_tokens:
        return 0

    if not company_tokens:
        return 0

    question_length = len(
        question_tokens
    )

    company_length = len(
        company_tokens
    )

    maximum_size = min(
        question_length,
        company_length,
    )

    for size in range(
        maximum_size,
        0,
        -1,
    ):
        company_phrases = {
            company_tokens[
                start:start + size
            ]
            for start in range(
                company_length - size + 1
            )
        }

        for start in range(
            question_length - size + 1
        ):
            question_phrase = (
                question_tokens[
                    start:start + size
                ]
            )

            if question_phrase in company_phrases:
                return size

    return 0


def resolve_company_from_question(
    question: str,
    companies: list[dict],
) -> dict | None:
    """
    Resolve an issuer from registered EquityAI companies.

    Matching priority:

    1. Exact ticker token.
    2. Most-specific company-name phrase.
    3. Exchange disambiguation.

    Tickerless issuers are supported.

    A partial company name is accepted only when it resolves
    uniquely at the strongest matching specificity.

    Returns None when no registered issuer is identified.
    Raises AmbiguousCompanyError when multiple issuers remain.
    """

    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        return None

    # -----------------------------------------------------
    # 1. Ticker resolution
    # -----------------------------------------------------

    ticker_matches = [
        company
        for company in companies
        if company.get("ticker")
        and _contains_token(
            normalized_question,
            company["ticker"],
        )
    ]

    if ticker_matches:
        candidates = ticker_matches

    else:
        # -------------------------------------------------
        # 2. Company-name resolution
        # -------------------------------------------------

        scored_candidates = []

        for company in companies:
            company_name = company.get(
                "company_name"
            )

            if not company_name:
                continue

            strength = (
                _company_match_strength(
                    normalized_question,
                    company_name,
                )
            )

            if strength > 0:
                scored_candidates.append(
                    (
                        strength,
                        company,
                    )
                )

        if not scored_candidates:
            return None

        strongest_match = max(
            strength
            for strength, _company
            in scored_candidates
        )

        candidates = [
            company
            for strength, company
            in scored_candidates
            if strength == strongest_match
        ]

    if len(candidates) == 1:
        return candidates[0]

    # -----------------------------------------------------
    # 3. Exchange disambiguation
    # -----------------------------------------------------

    exchange_matches = [
        company
        for company in candidates
        if company.get("exchange")
        and _contains_token(
            normalized_question,
            company["exchange"],
        )
    ]

    if len(exchange_matches) == 1:
        return exchange_matches[0]

    # Duplicate registry representations of the same issuer
    # must not create false ambiguity.
    unique_identities = {
        (
            company.get("company_name"),
            company.get("ticker"),
            company.get("exchange"),
        )
        for company in candidates
    }

    if len(unique_identities) == 1:
        return candidates[0]

    raise AmbiguousCompanyError(
        "The company identifier is ambiguous across "
        "multiple registered issuers. Specify the company "
        "name, ticker, or exchange more precisely."
    )
