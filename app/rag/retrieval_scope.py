from app.rag.company_selector import (
    resolve_company_from_question,
    resolve_companies_from_question,
)
from app.rag.historical_selector import (
    parse_historical_selection,
)
from app.rag.report_selector import (
    parse_report_selections,
)
from app.rag.vector_store import (
    list_registered_companies,
    resolve_documents,
    resolve_historical_documents,
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

    Supports:
    - unrestricted retrieval,
    - company-scoped retrieval,
    - explicit reporting periods,
    - multi-period comparisons,
    - historical annual windows,
    - historical quarterly windows.

    Historical requests return only reports actually available
    in the registry. A shortfall is exposed explicitly through
    requested_periods and available_periods.
    """

    companies = list_registered_companies()

    requested_companies = (
        resolve_companies_from_question(
            question,
            companies,
        )
    )

    if len(requested_companies) >= 2:
        company = None
    else:
        company = resolve_company_from_question(
            question,
            companies,
        )

    selections = parse_report_selections(
        question
    )

    historical_selection = (
        parse_historical_selection(
            question
        )
    )

    has_explicit_period_constraint = any(
        selection.report_type is not None
        or selection.fiscal_year is not None
        or selection.fiscal_quarter is not None
        or selection.fiscal_half is not None
        or selection.latest
        for selection in selections
    )

    has_historical_constraint = (
        historical_selection is not None
    )

    has_period_constraint = (
        has_explicit_period_constraint
        or has_historical_constraint
    )

    # -----------------------------------------------------
    # Multi-company explicit-period resolution
    # -----------------------------------------------------

    if len(requested_companies) >= 2:
        if historical_selection is not None:
            raise ValueError(
                "Multi-company historical-window retrieval "
                "is not supported yet."
            )

        documents = []

        for requested_company in requested_companies:
            for selection in selections:
                matches = resolve_documents(
                    ticker=requested_company.get("ticker"),
                    exchange=requested_company.get("exchange"),
                    company_name=requested_company.get(
                        "company_name"
                    ),
                    report_type=selection.report_type,
                    fiscal_year=selection.fiscal_year,
                    fiscal_quarter=selection.fiscal_quarter,
                    fiscal_half=selection.fiscal_half,
                    current_only=True,
                )

                if selection.latest:
                    matches = matches[:1]

                documents.extend(matches)

        if has_explicit_period_constraint:
            resolved_company_names = {
                document.get("company_name")
                for document in documents
            }

            missing_companies = [
                requested_company.get("company_name")
                for requested_company in requested_companies
                if requested_company.get("company_name")
                not in resolved_company_names
            ]

            if missing_companies:
                raise ReportNotFoundError(
                    "Financial reports are not available for "
                    "all requested companies: "
                    + ", ".join(missing_companies)
                )

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
            "company": None,
            "companies": requested_companies,
            "selections": selections,
            "historical_selection": None,
            "documents": unique_documents,
            "document_ids": document_ids,
            "filtered": True,
            "historical": False,
            "requested_periods": None,
            "available_periods": None,
            "period_type": None,
        }

    if company is None and has_period_constraint:
        raise MissingCompanyError(
            "A company or ticker is required when "
            "requesting a specific financial-report period."
        )

    # -----------------------------------------------------
    # No issuer and no period:
    # preserve unrestricted RAG behaviour.
    # -----------------------------------------------------

    if company is None:
        return {
            "company": None,
            "selections": selections,
            "historical_selection": None,
            "documents": [],
            "document_ids": None,
            "filtered": False,
            "historical": False,
            "requested_periods": None,
            "available_periods": None,
            "period_type": None,
        }

    # -----------------------------------------------------
    # Historical window
    # -----------------------------------------------------

    if historical_selection is not None:
        documents = resolve_historical_documents(
            ticker=company["ticker"],
            exchange=company.get("exchange"),
            period_type=(
                historical_selection.period_type
            ),
            count=historical_selection.count,
            current_only=True,
        )

        if not documents:
            raise ReportNotFoundError(
                "No financial reports are available "
                "for the requested historical window."
            )

        document_ids = [
            document["document_id"]
            for document in documents
        ]

        return {
            "company": company,
            "selections": selections,
            "historical_selection": (
                historical_selection
            ),
            "documents": documents,
            "document_ids": document_ids,
            "filtered": True,
            "historical": True,
            "requested_periods": (
                historical_selection.count
            ),
            "available_periods": len(documents),
            "period_type": (
                historical_selection.period_type
            ),
        }

    # -----------------------------------------------------
    # Existing explicit-period / company-only resolution
    # -----------------------------------------------------

    documents = []

    for selection in selections:
        matches = resolve_documents(
            ticker=company.get("ticker"),
            exchange=company.get("exchange"),
            company_name=company.get("company_name"),
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

        documents.extend(matches)

    if (
        has_explicit_period_constraint
        and not documents
    ):
        raise ReportNotFoundError(
            "The requested financial report is not "
            "available in the document registry."
        )

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
        "historical_selection": None,
        "documents": unique_documents,
        "document_ids": document_ids,
        "filtered": True,
        "historical": False,
        "requested_periods": None,
        "available_periods": None,
        "period_type": None,
    }
