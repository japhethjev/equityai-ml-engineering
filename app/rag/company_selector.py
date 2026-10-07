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

    A partial company name is retained for ambiguity detection,
    but must never defeat a more specific phrase.
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
    Resolve one issuer from registered EquityAI companies.

    Matching priority:
    1. Exact ticker token.
    2. Most-specific company-name phrase.
    3. Exchange disambiguation.

    Tickerless issuers are supported.

    Returns None when no registered issuer is identified.
    Raises AmbiguousCompanyError when multiple issuers remain.
    """

    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        return None

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


def resolve_companies_from_question(
    question: str,
    companies: list[dict],
) -> list[dict]:
    """
    Resolve multiple explicitly identified registered issuers.

    Multi-company matching is deliberately stricter than the
    singular resolver. An issuer is selected when its exact
    ticker or its complete meaningful company name occurs in
    the question.

    This prevents generic shared terms such as "bank" or
    ambiguous partial names such as "Dangote" from expanding
    accidentally to several issuers.
    """

    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        return []

    question_tokens = _normalize_tokens(
        normalized_question
    )

    matches = []

    for company in companies:
        ticker = company.get("ticker")
        company_name = company.get(
            "company_name"
        )

        ticker_match = (
            bool(ticker)
            and _contains_token(
                normalized_question,
                ticker,
            )
        )

        name_match = False

        if company_name:
            company_tokens = (
                _meaningful_company_tokens(
                    company_name
                )
            )

            if company_tokens:
                company_length = len(
                    company_tokens
                )

                name_match = any(
                    question_tokens[
                        start:start + company_length
                    ]
                    == company_tokens
                    for start in range(
                        len(question_tokens)
                        - company_length
                        + 1
                    )
                )

        if ticker_match or name_match:
            matches.append(company)

    resolved = []
    seen = set()

    for company in matches:
        identity = (
            company.get("company_name"),
            company.get("ticker"),
            company.get("exchange"),
        )

        if identity in seen:
            continue

        seen.add(identity)
        resolved.append(company)

    return resolved
