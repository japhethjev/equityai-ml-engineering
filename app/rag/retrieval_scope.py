from app.rag.company_selector import (
    resolve_company_from_question,
)
from app.rag.report_selector import (
    parse_report_selections,
)
from app.rag.vector_store import (
    list_registered_companies,
    resolve_documents,
)


class MissingCompanyError(ValueError):
    """Raised when a period is specified without an issuer."""


class ReportNotFoundError(ValueError):
    """Raised when the requested filing is not registered."""


def resolve_retrieval_scope(
    question: str,
) -> dict:
    """
    Determine which registered financial reports may be searched.

    Returns:
        {
            "company": dict | None,
            "selections": list[ReportSelection],
            "documents": list[dict],
            "document_ids": list[str] | None,
            "filtered": bool,
        }

    Unrestricted retrieval remains available when the question
    contains neither a registered company nor an explicit
    reporting period.

    A reporting period without a company is rejected rather than
    searching that period across unrelated issuers.
    """

    companies = list_registered_companies()

    company = resolve_company_from_question(
        question,
        companies,
    )

    selections = parse_report_selections(
        question
    )

    has_period_constraint = any(
        selection.report_type is not None
        or selection.fiscal_year is not None
        or selection.fiscal_quarter is not None
        or selection.fiscal_half is not None
        or selection.latest
        for selection in selections
    )

    if company is None and has_period_constraint:
        raise MissingCompanyError(
            "A company or ticker is required when "
            "requesting a specific financial-report period."
        )

    # No issuer and no period:
    # preserve existing unrestricted RAG behaviour.
    if company is None:
        return {
            "company": None,
            "selections": selections,
            "documents": [],
            "document_ids": None,
            "filtered": False,
        }

    documents = []

    for selection in selections:
        matches = resolve_documents(
            ticker=company["ticker"],
            exchange=company.get("exchange"),
            report_type=selection.report_type,
            fiscal_year=selection.fiscal_year,
            fiscal_quarter=selection.fiscal_quarter,
            fiscal_half=selection.fiscal_half,
            current_only=True,
        )

        # "latest" deliberately resolves from the registry's
        # deterministic descending period/publication ordering.
        if selection.latest:
            matches = matches[:1]

        if has_period_constraint and not matches:
            raise ReportNotFoundError(
                "The requested financial report is not "
                "available in the document registry."
            )

        documents.extend(matches)

    # Deduplicate while preserving registry order.
    unique_documents = []
    seen_ids = set()

    for document in documents:
        document_id = document["document_id"]

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)
        unique_documents.append(document)

    document_ids = [
        document["document_id"]
        for document in unique_documents
    ]

    return {
        "company": company,
        "selections": selections,
        "documents": unique_documents,
        "document_ids": document_ids,
        "filtered": True,
    }
