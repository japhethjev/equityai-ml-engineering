from app.rag.financial_analysis_executor import (
    execute_financial_analysis,
)
from app.rag.financial_analysis_scope import (
    resolve_financial_analysis_scope,
)
from app.rag.financial_evidence_bridge import (
    extract_financial_facts_from_document,
)
from app.rag.financial_presentation import (
    build_financial_presentation,
)
from app.rag.financial_period_selector import (
    select_financial_facts_for_periods,
)


def answer_financial_analysis_question(
    question: str,
    request_id: str | None = None,
) -> dict | None:
    """
    Execute the deterministic EquityAI financial-analysis path.

    Returns None when the question does not resolve to the
    deterministic historical financial-analysis workflow.

    All reported facts originate from registered documents.
    Derived KPIs are calculated only by the deterministic
    financial-analysis layer.
    """

    scope = resolve_financial_analysis_scope(
        question
    )

    if scope is None:
        return None

    company = scope.company

    all_facts = []
    sources = []

    for document in scope.documents:
        document_id = document["document_id"]

        evidence = (
            extract_financial_facts_from_document(
                document_id=document_id,
                company_name=company.get(
                    "company_name"
                ),
                ticker=company.get("ticker"),
                exchange=company.get("exchange"),
                currency=company.get("currency"),
            )
        )

        all_facts.extend(
            evidence.facts
        )

        sources.append(
            {
                "document_id": document_id,
                "document": document.get(
                    "document_name"
                ),
                "report_type": document.get(
                    "report_type"
                ),
                "fiscal_year": document.get(
                    "fiscal_year"
                ),
                "fiscal_quarter": document.get(
                    "fiscal_quarter"
                ),
                "fiscal_half": document.get(
                    "fiscal_half"
                ),
                "period_end": document.get(
                    "period_end"
                ),
            }
        )

    selected_facts = select_financial_facts_for_periods(
        facts=all_facts,
        period_type=scope.period_type,
        count=scope.requested_period_count,
    )

    selected_periods = {
        (
            fact.fiscal_year,
            fact.fiscal_quarter,
        )
        for fact in selected_facts
        if fact.fiscal_year is not None
    }

    available_period_count = len(
        selected_periods
    )

    complete_history = (
        available_period_count
        >= scope.requested_period_count
    )

    execution = execute_financial_analysis(
        company_name=company["company_name"],
        ticker=company.get("ticker"),
        metric_keys=scope.request.metric_keys,
        facts=selected_facts,
    )

    if not execution.company.values:
        raise ValueError(
            "No requested financial metrics could be "
            "grounded in the selected reports."
        )

    presentation = (
        build_financial_presentation(
            [execution.company]
        )
    )

    answer = presentation.text

    if not complete_history:
        coverage_note = (
            "\n\nData coverage: "
            f"{available_period_count} of "
            f"{scope.requested_period_count} requested "
            "periods were available in the grounded "
            "financial evidence."
        )

        answer += coverage_note

    return {
        "answer": answer,
        "sources": sources,
        "request_id": request_id,
        "analysis_type": "financial",
        "complete_history": complete_history,
        "requested_period_count": (
            scope.requested_period_count
        ),
        "available_period_count": (
            available_period_count
        ),
        "missing_metrics": [
            {
                "metric": metric,
                "period": period,
            }
            for metric, period
            in execution.missing
        ],
    }
