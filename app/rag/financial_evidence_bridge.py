from dataclasses import dataclass

from app.rag.financial_fact_extractor import (
    extract_reported_facts_from_lines,
)
from app.rag.financial_facts import FinancialFact
from app.rag.financial_period_parser import (
    parse_period_header,
)
from app.rag.vector_store import (
    get_document_chunks,
)


@dataclass(frozen=True)
class FinancialEvidenceResult:
    """
    Structured extraction result for one registered document.
    """

    document_id: str
    document_name: str
    facts: tuple[FinancialFact, ...]
    pages_examined: tuple[int, ...]


def _chunk_lines(
    content: str,
) -> list[str]:
    """
    Preserve extracted PDF line boundaries while removing
    blank lines.
    """

    return [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]


def _page_chunks(
    chunks: list[dict],
) -> dict[int, list[dict]]:
    """
    Group document chunks by source page while preserving
    chunk order.
    """

    grouped: dict[int, list[dict]] = {}

    for chunk in chunks:
        grouped.setdefault(
            chunk["page"],
            [],
        ).append(chunk)

    for page_chunks in grouped.values():
        page_chunks.sort(
            key=lambda item: item["chunk"]
        )

    return grouped


def _page_lines(
    chunks: list[dict],
) -> list[str]:
    """
    Reconstruct page-level evidence lines from ordered chunks.

    Duplicate identical lines are removed only when they arise
    from chunk overlap. Original ordering is preserved.
    """

    lines = []
    seen = set()

    for chunk in chunks:
        for line in _chunk_lines(
            chunk["content"]
        ):
            if line in seen:
                continue

            seen.add(line)
            lines.append(line)

    return lines


def extract_financial_facts_from_document(
    *,
    document_id: str,
    company_name: str | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
    currency: str | None = None,
    unit: str | None = None,
) -> FinancialEvidenceResult:
    """
    Extract grounded reported financial facts from all chunks
    belonging to one registered document.

    Period columns must be established from explicit table
    headers before metric rows are interpreted.

    The bridge never guesses missing periods and never performs
    financial calculations. Derived KPIs remain the responsibility
    of the deterministic financial-analysis layer.
    """

    normalized_document_id = (
        str(document_id).strip()
    )

    if not normalized_document_id:
        raise ValueError(
            "document_id is required"
        )

    chunks = get_document_chunks(
        normalized_document_id
    )

    if not chunks:
        raise ValueError(
            "No chunks found for registered document"
        )

    document_names = {
        chunk["document"]
        for chunk in chunks
    }

    if len(document_names) != 1:
        raise ValueError(
            "Document chunks contain inconsistent "
            "document names"
        )

    document_name = next(
        iter(document_names)
    )

    grouped_pages = _page_chunks(
        chunks
    )

    facts: list[FinancialFact] = []
    pages_examined = []

    for page in sorted(grouped_pages):
        pages_examined.append(page)

        lines = _page_lines(
            grouped_pages[page]
        )

        # A page can contain narrative references to years
        # before the actual financial table. Therefore we do
        # not assume that the first period-looking line is the
        # authoritative table header.
        #
        # Evaluate every candidate period header and retain the
        # candidate that produces the strongest set of grounded
        # financial facts.
        candidate_results = []

        for header_index, line in enumerate(
            lines
        ):
            periods = parse_period_header(
                line
            )

            if not periods:
                continue

            evidence_lines = lines[
                header_index + 1:
            ]

            candidate_facts = (
                extract_reported_facts_from_lines(
                    lines=evidence_lines,
                    periods=periods,
                    document_id=normalized_document_id,
                    document_name=document_name,
                    page=page,
                    company_name=company_name,
                    ticker=ticker,
                    exchange=exchange,
                    currency=currency,
                    unit=unit,
                )
            )

            if not candidate_facts:
                continue

            # Prefer the candidate that grounds the greatest
            # number of financial facts. Period count is used
            # as a deterministic secondary preference.
            candidate_results.append(
                (
                    len(candidate_facts),
                    len(periods),
                    -header_index,
                    candidate_facts,
                )
            )

        if not candidate_results:
            continue

        candidate_results.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
            reverse=True,
        )

        page_facts = candidate_results[
            0
        ][3]

        facts.extend(page_facts)

    return FinancialEvidenceResult(
        document_id=normalized_document_id,
        document_name=document_name,
        facts=tuple(facts),
        pages_examined=tuple(
            pages_examined
        ),
    )
