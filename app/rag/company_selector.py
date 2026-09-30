import re


class AmbiguousCompanyError(ValueError):
    """Raised when company identity is not unique."""


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


def resolve_company_from_question(
    question: str,
    companies: list[dict],
) -> dict | None:
    """
    Resolve an issuer from registered EquityAI companies.

    Matching priority:
    1. Exact ticker token
    2. Company-name phrase
    3. Exchange disambiguation

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

    candidates = ticker_matches

    if not candidates:
        candidates = [
            company
            for company in companies
            if company.get("company_name")
            and company["company_name"].lower()
            in normalized_question.lower()
        ]

    if not candidates:
        return None

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
            company.get("ticker"),
            company.get("exchange"),
        )
        for company in candidates
    }

    if len(unique_identities) == 1:
        return candidates[0]

    raise AmbiguousCompanyError(
        "The company identifier is ambiguous across "
        "multiple exchanges. Specify the exchange."
    )
