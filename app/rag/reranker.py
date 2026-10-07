import re

from app.rag.financial_analysis_request import extract_metric_keys
from app.rag.financial_metrics import get_metric


def normalize_text(text: str) -> str:
    """Lowercase text and remove most punctuation for comparison."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def keyword_overlap_score(query: str, content: str) -> float:
    """
    Measure how many important query words appear in the content.
    """

    query_words = set(normalize_text(query).split())
    content_words = set(normalize_text(content).split())

    if not query_words:
        return 0.0

    overlap = query_words.intersection(content_words)

    return len(overlap) / len(query_words)


def exact_phrase_bonus(query: str, content: str) -> float:
    """
    Add extra score when important phrases appear exactly.
    """

    query_normalized = normalize_text(query)
    content_normalized = normalize_text(content)

    bonus = 0.0

    important_phrases = [
        "offer price",
        "ipo offer price",
        "gross proceeds",
        "net proceeds",
        "market cap",
        "price book",
        "net margin",
    ]

    for phrase in important_phrases:
        if phrase in query_normalized and phrase in content_normalized:
            bonus += 0.4

    return bonus


def financial_evidence_bonus(
    query: str,
    content: str,
) -> float:
    """
    Reward chunks containing the requested financial metric
    together with all explicit years requested by the user.
    """

    metric_keys = extract_metric_keys(query)

    if not metric_keys:
        return 0.0

    query_years = set(
        re.findall(r"\b20\d{2}\b", query)
    )

    if not query_years:
        return 0.0

    content_normalized = normalize_text(content)
    content_years = set(
        re.findall(r"\b20\d{2}\b", content)
    )

    if not query_years.issubset(content_years):
        return 0.0

    for metric_key in metric_keys:
        metric = get_metric(metric_key)

        aliases = (
            metric.label,
            *metric.aliases,
        )

        if any(
            re.search(
                r"(?<!\w)"
                + re.escape(
                    normalize_text(alias).strip()
                )
                + r"(?!\w)",
                content_normalized,
            )
            for alias in aliases
        ):
            return 1.0

    return 0.0


def rerank_results(
    query: str,
    results: list[dict],
) -> list[dict]:
    """
    Rerank retrieved chunks using:
    - semantic similarity
    - keyword overlap
    - exact phrase matching
    """

    reranked = []

    for result in results:
        semantic_score = float(
            result.get(
                "semantic_similarity",
                result.get("similarity", 0.0),
            )
        )

        keyword_score = keyword_overlap_score(
            query,
            result["content"],
        )

        phrase_bonus = exact_phrase_bonus(
            query,
            result["content"],
        )

        evidence_bonus = financial_evidence_bonus(
            query,
            result["content"],
        )

        final_score = (
            0.55 * semantic_score
            + 0.30 * keyword_score
            + 0.15 * phrase_bonus
            + 0.15 * evidence_bonus
        )

        updated_result = result.copy()
        updated_result["keyword_overlap"] = keyword_score
        updated_result["phrase_bonus"] = phrase_bonus
        updated_result["evidence_bonus"] = evidence_bonus
        updated_result["final_score"] = final_score

        reranked.append(updated_result)

    return sorted(
        reranked,
        key=lambda x: x["final_score"],
        reverse=True,
    )

def select_balanced_evidence(
    ranked_results: list[dict],
    document_ids: list[str] | None = None,
    limit: int = 3,
) -> list[dict]:
    """
    Select final evidence while preserving balanced document depth.

    For multi-document retrieval, evidence is selected round-robin
    by requested document so one filing cannot monopolise the final
    context. Within each document, original reranking order is
    preserved.

    Single-document and unrestricted retrieval preserve the
    existing top-N behaviour.
    """
    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    if not ranked_results:
        return []

    if not document_ids or len(document_ids) <= 1:
        return ranked_results[:limit]

    requested_ids = list(
        dict.fromkeys(
            str(document_id)
            for document_id in document_ids
        )
    )

    results_by_document = {
        document_id: []
        for document_id in requested_ids
    }

    for result in ranked_results:
        result_document_id = result.get("document_id")

        if result_document_id is None:
            continue

        normalized_id = str(result_document_id)

        if normalized_id in results_by_document:
            results_by_document[normalized_id].append(result)

    selected = []
    depth = 0

    while len(selected) < limit:
        added_this_round = False

        for document_id in requested_ids:
            document_results = results_by_document[
                document_id
            ]

            if depth >= len(document_results):
                continue

            selected.append(document_results[depth])
            added_this_round = True

            if len(selected) >= limit:
                break

        if not added_this_round:
            break

        depth += 1

    return selected
