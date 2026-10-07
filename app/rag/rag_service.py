import os
import time

from openai import OpenAI

from app.observability import (
    create_request_id,
    log_rag_error,
    log_rag_request,
    measure_stage,
    record_rag_metrics,
)
from app.rag.embedding_service import embed_text
from app.rag.financial_analysis_request import (
    extract_metric_keys,
)
from app.rag.financial_metrics import get_metric
from app.rag.vector_store import (
    financial_metric_search,
    hybrid_search,
)
from app.rag.reranker import (
    rerank_results,
    select_balanced_evidence,
)
from app.rag.retrieval_scope import (
    resolve_retrieval_scope,
)


ABSTENTION_TEXT = (
    "I could not find sufficient evidence "
    "in the provided documents."
)

# Retrieve enough candidates for reranking while keeping
# the final evidence set small and focused.
RERANK_CANDIDATE_LIMIT = 20


def build_targeted_retrieval_query(
    query: str,
    document: dict,
) -> str:
    """
    Build retrieval intent for one resolved financial document.

    Recognised financial metrics use authoritative document metadata
    and canonical metric labels so unrelated company names from a
    multi-company question do not dilute retrieval.

    Unrecognised questions retain the original query as a fallback.
    """
    company_name = (
        document.get("company_name") or ""
    ).strip()

    metric_keys = extract_metric_keys(query)

    if not metric_keys:
        return (
            f"Company: {company_name}\n"
            f"Question: {query}"
        )

    parts = [
        f"Company: {company_name}",
    ]

    report_type = document.get("report_type")

    if report_type:
        parts.append(
            f"Report type: {report_type}"
        )

    fiscal_year = document.get("fiscal_year")

    if fiscal_year is not None:
        parts.append(
            f"Fiscal year: {fiscal_year}"
        )

    metric_labels = [
        get_metric(metric_key).label
        for metric_key in metric_keys
    ]

    parts.append(
        "Requested metric: "
        + "; ".join(metric_labels)
    )

    return "\n".join(parts)


def get_financial_metric_terms(
    query: str,
) -> list[str]:
    """Return canonical labels and aliases for requested metrics."""
    terms = []

    for metric_key in extract_metric_keys(query):
        metric = get_metric(metric_key)
        terms.extend(
            [
                metric.label,
                *metric.aliases,
            ]
        )

    return list(
        dict.fromkeys(
            term
            for term in terms
            if term
        )
    )


def merge_retrieval_candidates(
    *candidate_sets: list[dict],
) -> list[dict]:
    """Merge retrieval lanes without duplicating the same chunk."""
    merged = []
    seen = set()

    for candidate_set in candidate_sets:
        for result in candidate_set:
            identity = (
                str(result.get("document_id")),
                result.get("page"),
                result.get("chunk"),
            )

            if identity in seen:
                continue

            seen.add(identity)
            merged.append(result)

    return merged


# =========================================================
# CALCULATION INTENT CONTROL
# =========================================================


CALCULATION_INTENT_PHRASES = (
    "calculate",
    "compute",
    "derive",
    "work out",
    "determine from",
    "estimate from",
    "using the figures",
    "using these figures",
    "based on the figures",
)


COMPARISON_INTENT_PHRASES = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "change from",
    "change between",
)


def explicitly_requests_comparison(
    query: str,
) -> bool:
    """
    Return True when the user explicitly requests a
    comparison between financial reporting periods.

    Comparison intent permits calculation of absolute
    and percentage changes, provided the underlying
    period values are directly supported by evidence.
    """

    normalized_query = (
        " "
        + " ".join(query.lower().split())
        + " "
    )

    return any(
        phrase in normalized_query
        for phrase in COMPARISON_INTENT_PHRASES
    )


def explicitly_requests_calculation(
    query: str,
) -> bool:
    """
    Return True only when the user's wording explicitly
    authorises a calculation or derivation.

    Examples that permit calculation:
    - Calculate the operating profit margin.
    - Compute the dividend yield.
    - Work out the growth rate.
    - Determine the margin from the supplied figures.

    A normal factual request such as:
    - What was the operating profit margin?

    does NOT itself authorise derivation of a metric that
    is absent from the retrieved evidence.
    """

    normalized_query = " ".join(
        query.lower().split()
    )

    return any(
        phrase in normalized_query
        for phrase in CALCULATION_INTENT_PHRASES
    )


def looks_like_unrequested_calculation(
    answer: str,
) -> bool:
    """
    Detect an answer that explicitly presents itself as
    a calculated or derived result.

    This is a defensive post-generation guard.

    It does not attempt to classify every numeric answer.
    Instead, it catches explicit evidence that the model
    performed a calculation when calculation was not
    authorised by the user.
    """

    normalized_answer = " ".join(
        answer.lower().split()
    )

    calculation_markers = (
        "calculated",
        "calculation:",
        "derived",
        "computed",
        "work out",
        "÷",
    )

    return any(
        marker in normalized_answer
        for marker in calculation_markers
    )


# =========================================================
# OPENAI CLIENT
# =========================================================


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client only when answer generation
    actually requires one.

    This allows application modules and automated tests
    to load without requiring OpenAI credentials.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    return OpenAI(api_key=api_key)


# =========================================================
# RAG QUESTION ANSWERING
# =========================================================


def answer_question(
    query: str,
    request_id: str | None = None,
) -> dict:
    """
    Answer a user question using retrieved document
    evidence only.

    If a request ID is supplied by the API layer, the same
    identifier is used throughout the RAG pipeline. When the
    function is called directly, a new request ID is generated.

    Observability captures:
    - request ID,
    - embedding latency,
    - retrieval latency,
    - reranking latency,
    - generation latency,
    - total latency,
    - retrieved and used chunk counts,
    - answer, abstention, or error outcome,
    - failed stage and exception type for errors.

    The generation step distinguishes between:
    - relevant evidence,
    - directly sufficient evidence,
    - explicitly authorised calculations,
    - and unsupported inference or derivation.
    """

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    query = query.strip()

    if not query:
        raise ValueError(
            "Question cannot be empty."
        )

    # -----------------------------------------------------
    # Determine calculation permission deterministically
    # -----------------------------------------------------

    calculation_allowed = (
        explicitly_requests_calculation(query)
        or explicitly_requests_comparison(query)
    )

    # -----------------------------------------------------
    # Observability setup
    # -----------------------------------------------------

    if request_id is None:
        request_id = create_request_id()

    request_start = time.perf_counter()

    timings = {}

    retrieved = []
    top_results = []

    # -----------------------------------------------------
    # Helper for failed RAG stages
    # -----------------------------------------------------

    def record_error(
        failed_stage: str,
        error: Exception,
    ) -> None:
        """
        Record structured ERROR telemetry without
        exposing the user's question or document
        content.
        """

        timings["total_ms"] = round(
            (
                time.perf_counter()
                - request_start
            )
            * 1000,
            2,
        )

        log_rag_error(
            request_id=request_id,
            failed_stage=failed_stage,
            error=error,
            retrieved_chunks=len(retrieved),
            used_chunks=len(top_results),
            timings=timings,
        )

        record_rag_metrics(
            outcome="ERROR",
            timings=timings,
        )

    # =====================================================
    # 1. RETRIEVAL SCOPE
    # =====================================================

    try:
        with measure_stage(
            "scope_resolution",
            timings,
        ):
            retrieval_scope = (
                resolve_retrieval_scope(query)
            )

            document_ids = retrieval_scope[
                "document_ids"
            ]

            requested_companies = (
                retrieval_scope.get("companies") or []
            )
            multi_company_comparison = (
                len(requested_companies) >= 2
            )

    except Exception as error:
        record_error(
            "scope_resolution",
            error,
        )
        raise

    # =====================================================
    # 2. EMBEDDING / RETRIEVAL / RERANKING
    # =====================================================

    if multi_company_comparison:
        ranked = []

        documents = retrieval_scope.get("documents") or []

        try:
            with measure_stage(
                "embedding",
                timings,
            ):
                targeted_queries = []

                for document in documents:
                    targeted_query = (
                        build_targeted_retrieval_query(
                            query,
                            document,
                        )
                    )

                    targeted_queries.append(
                        (
                            document,
                            targeted_query,
                            embed_text(targeted_query),
                        )
                    )

        except Exception as error:
            record_error(
                "embedding",
                error,
            )
            raise

        try:
            with measure_stage(
                "retrieval",
                timings,
            ):
                retrieved_by_document = []

                for (
                    document,
                    targeted_query,
                    targeted_embedding,
                ) in targeted_queries:
                    document_id = str(
                        document["document_id"]
                    )

                    hybrid_results = hybrid_search(
                        query=targeted_query,
                        query_embedding=targeted_embedding,
                        limit=RERANK_CANDIDATE_LIMIT,
                        document_ids=[document_id],
                    )

                    metric_terms = (
                        get_financial_metric_terms(query)
                    )

                    metric_results = []

                    if metric_terms:
                        metric_results = (
                            financial_metric_search(
                                document_id=document_id,
                                metric_terms=metric_terms,
                                fiscal_year=document.get(
                                    "fiscal_year"
                                ),
                                limit=RERANK_CANDIDATE_LIMIT,
                            )
                        )

                    document_results = (
                        merge_retrieval_candidates(
                            hybrid_results,
                            metric_results,
                        )
                    )

                    retrieved_by_document.append(
                        (
                            targeted_query,
                            document_results,
                        )
                    )

                retrieved = [
                    result
                    for _, document_results
                    in retrieved_by_document
                    for result in document_results
                ]

        except Exception as error:
            record_error(
                "retrieval",
                error,
            )
            raise

        try:
            with measure_stage(
                "reranking",
                timings,
            ):
                for (
                    targeted_query,
                    document_results,
                ) in retrieved_by_document:
                    ranked.extend(
                        rerank_results(
                            targeted_query,
                            document_results,
                        )
                    )

                ranked.sort(
                    key=lambda result: result["final_score"],
                    reverse=True,
                )

        except Exception as error:
            record_error(
                "reranking",
                error,
            )
            raise

    else:
        try:
            with measure_stage(
                "embedding",
                timings,
            ):
                query_embedding = embed_text(
                    query
                )

        except Exception as error:
            record_error(
                "embedding",
                error,
            )
            raise

        try:
            with measure_stage(
                "retrieval",
                timings,
            ):
                retrieved = hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    limit=RERANK_CANDIDATE_LIMIT,
                    document_ids=document_ids,
                )

        except Exception as error:
            record_error(
                "retrieval",
                error,
            )
            raise

        try:
            with measure_stage(
                "reranking",
                timings,
            ):
                ranked = rerank_results(
                    query,
                    retrieved,
                )

        except Exception as error:
            record_error(
                "reranking",
                error,
            )
            raise

    evidence_limit = (
        3 * len(document_ids)
        if multi_company_comparison
        else 3
    )

    top_results = select_balanced_evidence(
        ranked,
        document_ids=document_ids,
        limit=evidence_limit,
    )

    # =====================================================
    # 4. EMPTY RETRIEVAL
    # =====================================================

    if not top_results:

        timings["total_ms"] = round(
            (
                time.perf_counter()
                - request_start
            )
            * 1000,
            2,
        )

        log_rag_request(
            request_id=request_id,
            outcome="ABSTAINED",
            retrieved_chunks=len(retrieved),
            used_chunks=0,
            timings=timings,
        )

        record_rag_metrics(
            outcome="ABSTAINED",
            timings=timings,
        )

        return {
            "answer": ABSTENTION_TEXT,
            "sources": [],
            "request_id": request_id,
        }

    # =====================================================
    # 5. BUILD GROUNDED CONTEXT
    # =====================================================

    context = "\n\n".join(
        [
            (
                f"Document: {r['document']}\n"
                f"Page: {r['page']}\n"
                f"Content: {r['content']}"
            )
            for r in top_results
        ]
    )

    # =====================================================
    # 6. EVIDENCE-CONSTRAINED PROMPT
    # =====================================================

    prompt = f"""
You are a financial research assistant answering questions
from retrieved financial documents.

Your task has two stages:

STAGE 1 — EVIDENCE SUFFICIENCY

Before answering, determine whether the retrieved evidence
directly supports the exact fact requested by the user.

Evidence may be relevant to the topic without being sufficient
to answer the question.

Apply these rules strictly:

1. Use ONLY the evidence supplied below.

2. Do not use outside knowledge.

3. Do not invent, estimate, assume, extrapolate, or infer a
   factual answer that is not directly supported by the evidence.

4. The financial concept in the evidence must match the financial
   concept requested in the question.

5. Do not substitute one financial metric or concept for another.

Examples of concepts that must remain distinct include:
- IPO or offer price versus current market price
- gross proceeds versus net proceeds
- revenue versus profit
- gross profit versus operating profit
- operating profit versus profit after tax
- historical price versus current price
- forecast figures versus actual figures
- annual figures versus interim figures
- dividend per share versus earnings per share
- profit versus profit margin
- dividend per share versus dividend yield

6. Respect temporal qualifiers in the question.

Words and phrases such as:
- current
- latest
- today
- 2024
- 2025
- 2026
- H1
- H2
- interim
- full year

are material parts of the question.

Evidence from a different period must not be presented as the
requested period unless the evidence explicitly establishes the
relationship.

7. A semantically similar number is not sufficient evidence.

For example, if the question asks for a current stock market
price and the evidence contains only an IPO or offer price,
the question is NOT answerable from that evidence.

8. Direct evidence means that the requested fact or financial
   metric itself is explicitly stated in the retrieved evidence.

The presence of figures from which a requested metric could be
calculated does NOT mean that the requested metric is directly
stated.

9. Do not calculate or derive a missing financial metric from
   other figures in the evidence unless calculation permission
   for this question is explicitly ALLOWED below.

This restriction includes, but is not limited to:
- margins
- ratios
- percentages
- growth rates
- yields
- returns
- per-share measures
- valuation multiples
- differences
- averages
- totals constructed from separate figures

10. Even when all inputs required for a calculation are present,
    the calculated result is not direct evidence of the requested
    metric.

11. A question that simply asks:
    - "what is"
    - "what was"
    - "how much"
    - "state"
    - "identify"

    is a factual retrieval request.

    Such wording does NOT give permission to calculate a missing
    metric.

12. Perform a calculation only when CALCULATION PERMISSION below
    is ALLOWED.

13. If calculation permission is ALLOWED and the required inputs
    are directly supported by the retrieved evidence, you may
    calculate the result.

    Clearly identify the result as calculated or derived rather
    than directly reported.

14. If the evidence does not directly and sufficiently answer the
    exact question, and calculation permission is NOT ALLOWED,
    respond with EXACTLY:

"{ABSTENTION_TEXT}"

Do not add a source, explanation, related figure, alternative
answer, calculation, or additional sentence after this abstention
response.


STAGE 2 — ANSWER

Only if Stage 1 determines that sufficient direct evidence exists,
or calculation permission is explicitly ALLOWED:

15. Answer the exact question asked.

16. Keep the answer concise.

17. For a directly reported fact, state the supporting document
    and page.

18. For an explicitly permitted calculation, state that the result
    is calculated or derived and identify the supporting document
    and page containing the input figures.

19. Do not present related information as though it were the
    requested fact.

20. Do not change the meaning, period, financial concept, or basis
    of a figure merely because another retrieved figure appears
    semantically similar.


CALCULATION PERMISSION FOR THIS QUESTION:

{"ALLOWED" if calculation_allowed else "NOT ALLOWED"}


IMPORTANT APPLICATION CONTROL:

The calculation permission above has been determined by
application logic.

If CALCULATION PERMISSION is NOT ALLOWED:

- You MUST NOT calculate, derive, estimate, compute, reconstruct,
  or infer a missing financial metric.

- The requested financial metric itself must be explicitly stated
  in the retrieved evidence.

- Having all figures necessary to calculate the requested metric
  is NOT sufficient evidence.

- Do not divide, multiply, subtract, add, average, annualise,
  extrapolate, or otherwise transform retrieved figures to create
  the requested metric.

- A reported profit figure is NOT a reported profit margin.

- Reported revenue and reported operating profit do NOT constitute
  a directly reported operating profit margin.

- A reported EPS figure is NOT a reported dividend per share.

- A reported dividend and market price do NOT constitute a
  directly reported dividend yield.

- If the exact requested metric is not explicitly stated in the
  retrieved evidence, respond EXACTLY:

"{ABSTENTION_TEXT}"


If CALCULATION PERMISSION is ALLOWED:

- A calculation may be performed only when all required inputs
  are directly supported by the retrieved evidence.

- Clearly distinguish the calculated result from a directly
  reported figure.


MULTI-COMPANY COMPARISON CONTROL:

{"ENABLED" if multi_company_comparison else "DISABLED"}

When ENABLED:
- Treat every requested company as a separate evidence subject.
- Never use one company's evidence to answer for another company.
- Every reported figure must be supported by that company's evidence.
- Do not silently omit any requested company.
- Present the results first as a concise Markdown comparison table.
- Include Company, Period, Requested Metric(s), Source Document, and Page.
- Preserve the currency, units, scale, and terminology of each source.
- State material comparability limitations where currencies, units,
  accounting bases, or definitions differ.
- After the table, provide a brief evidence-grounded comparative analysis.
- Do not calculate metrics unless CALCULATION PERMISSION is ALLOWED.
- If evidence is insufficient, apply the Stage 1 abstention rules rather
  than filling a company's result from another company's evidence.

When DISABLED:
- Follow the normal concise answer format above.

USER QUESTION:

{query}


RETRIEVED EVIDENCE:

{context}
"""

    # =====================================================
    # 7. GROUNDED GENERATION
    # =====================================================

    try:
        with measure_stage(
            "generation",
            timings,
        ):

            client = get_openai_client()

            response = client.responses.create(
                model="gpt-5-mini",
                input=prompt,
            )

            answer = (
                response.output_text.strip()
            )

            # -------------------------------------------------
            # Defensive deterministic calculation guard
            # -------------------------------------------------
            #
            # If application logic did not permit a calculation
            # but the generated answer explicitly describes
            # itself as calculated/derived, convert the response
            # into a safe abstention.
            #
            # This provides a second layer of protection if the
            # model fails to follow the prompt instruction.
            # -------------------------------------------------

            if (
                not calculation_allowed
                and looks_like_unrequested_calculation(
                    answer
                )
            ):
                answer = ABSTENTION_TEXT

    except Exception as error:
        record_error(
            "generation",
            error,
        )
        raise

    # =====================================================
    # 8. DETERMINE OUTCOME
    # =====================================================

    if answer == ABSTENTION_TEXT:
        outcome = "ABSTAINED"
    else:
        outcome = "ANSWERED"

    # =====================================================
    # 9. TOTAL REQUEST LATENCY
    # =====================================================

    timings["total_ms"] = round(
        (
            time.perf_counter()
            - request_start
        )
        * 1000,
        2,
    )

    # =====================================================
    # 10. OBSERVABILITY SUMMARY
    # =====================================================

    log_rag_request(
        request_id=request_id,
        outcome=outcome,
        retrieved_chunks=len(retrieved),
        used_chunks=len(top_results),
        timings=timings,
    )

    record_rag_metrics(
        outcome=outcome,
        timings=timings,
    )

    # =====================================================
    # 11. RESPONSE
    # =====================================================

    return {
        "answer": answer,
        "sources": [
            {
                "document": r["document"],
                "page": r["page"],
                "chunk": r["chunk"],
                "score": r["final_score"],
            }
            for r in top_results
        ],
        "request_id": request_id,
    }