from dataclasses import dataclass

from app.rag.company_selector import (
    resolve_company_from_question,
)
from app.rag.financial_analysis_request import (
    FinancialAnalysisRequest,
    parse_financial_analysis_request,
)
from app.rag.vector_store import (
    list_registered_companies,
    resolve_historical_documents,
    resolve_supporting_documents,
)


class FinancialAnalysisCompanyError(ValueError):
    """
    Raised when an investor-analysis request requires an issuer
    but no registered company can be resolved.
    """


class FinancialAnalysisDocumentsError(ValueError):
    """
    Raised when no registered reports are available for the
    requested historical analysis.
    """


@dataclass(frozen=True)
class FinancialAnalysisScope:
    """
    Fully resolved scope for a deterministic investor-analysis
    request.

    The document IDs in this object are the authoritative
    retrieval boundary for downstream RAG/fact extraction.
    """

    request: FinancialAnalysisRequest
    company: dict
    documents: tuple[dict, ...]
    document_ids: tuple[str, ...]
    requested_period_count: int
    available_period_count: int
    period_type: str
    complete_history: bool


def resolve_financial_analysis_scope(
    question: str,
) -> FinancialAnalysisScope | None:
    """
    Resolve a natural-language investor-analysis question to
    registered historical financial reports.

    Returns None when the question does not belong to the
    deterministic financial-analysis path.

    Historical requests:
        "Show AAPL revenue for the last 5 years."
        "Show GTCO liquidity ratios for the last 4 quarters."

    Only reports actually available in the registry are returned.

    If five years are requested but only three qualifying annual
    reports exist, the three available reports are returned and
    complete_history is False.

    Missing reports are never invented or substituted.
    """

    request = parse_financial_analysis_request(
        question
    )

    if not request.is_financial_analysis:
        return None

    companies = list_registered_companies()

    company = resolve_company_from_question(
        question,
        companies,
    )

    if company is None:
        raise FinancialAnalysisCompanyError(
            "A registered company or ticker is required "
            "for financial analysis."
        )

    # Historical requests explicitly specify their period basis
    # and count through the request parser.
    #
    # A comparison or analytical request without an explicit
    # historical horizon is deliberately not guessed here.
    # That will be handled by exact-period/company comparison
    # logic separately.
    if (
        request.period_type is None
        or request.period_count is None
    ):
        return None

    documents = resolve_historical_documents(
        ticker=company.get("ticker"),
        exchange=company.get("exchange"),
        company_name=company.get("company_name"),
        period_type=request.period_type,
        count=request.period_count,
        current_only=True,
    )

    if not documents:
        documents = resolve_supporting_documents(
            ticker=company.get("ticker"),
            exchange=company.get("exchange"),
            company_name=company.get("company_name"),
            current_only=True,
        )

    if not documents:
        raise FinancialAnalysisDocumentsError(
            "No registered financial reports or supporting "
            "documents are available for the requested analysis."
        )

    # Defensive deduplication. The historical resolver should
    # already return unique current filings, but the analysis
    # boundary must not pass duplicate documents downstream.
    unique_documents = []
    seen_ids = set()

    for document in documents:
        document_id = document["document_id"]

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)
        unique_documents.append(document)

    document_ids = tuple(
        document["document_id"]
        for document in unique_documents
    )

    available_period_count = len(
        unique_documents
    )

    return FinancialAnalysisScope(
        request=request,
        company=company,
        documents=tuple(unique_documents),
        document_ids=document_ids,
        requested_period_count=(
            request.period_count
        ),
        available_period_count=(
            available_period_count
        ),
        period_type=request.period_type,
        complete_history=(
            available_period_count
            >= request.period_count
        ),
    )
