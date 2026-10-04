from app.rag.embedding_service import embed_text
from app.rag.reranker import rerank_results
from app.rag.vector_store import get_connection, hybrid_search


DOCUMENT_ID = "f7da04fc-8cc6-44ac-a5bd-e9d926cd8211"

QUERY = (
    "According to HSBC Holdings plc Annual Report and Accounts 2025, "
    "what was HSBC's reported profit before tax in 2025, "
    "and how did it compare with 2024?"
)


def excerpt(text: str, length: int = 300) -> str:
    return " ".join(text.split())[:length]


def main() -> None:
    print("=== HSBC RETRIEVAL DIAGNOSTIC ===")
    print(f"Document ID: {DOCUMENT_ID}")
    print(f"Query: {QUERY}")

    # ---------------------------------------------------------
    # 1. Find stored chunks containing likely PBT evidence.
    # READ ONLY.
    # ---------------------------------------------------------

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    page,
                    chunk,
                    content
                FROM document_chunks
                WHERE document_id = %s::uuid
                  AND (
                      content ILIKE %s
                      OR content ILIKE %s
                      OR content ILIKE %s
                  )
                ORDER BY page, chunk
                LIMIT 50
                """,
                (
                    DOCUMENT_ID,
                    "%profit before tax%",
                    "%29.9%",
                    "%32.3%",
                ),
            )

            evidence_rows = cursor.fetchall()

    print()
    print("=== STORED PBT / VALUE MATCHES ===")
    print(f"Matches: {len(evidence_rows)}")

    for page, chunk, content in evidence_rows:
        print()
        print(
            f"page={page} chunk={chunk} "
            f"content={excerpt(content)}"
        )

    # ---------------------------------------------------------
    # 2. Reproduce retrieval with a larger DIAGNOSTIC pool.
    # Production remains limit=5.
    # ---------------------------------------------------------

    print()
    print("=== EMBEDDING QUERY ===")

    query_embedding = embed_text(QUERY)

    print(
        f"Embedding dimensions: {len(query_embedding)}"
    )

    retrieved = hybrid_search(
        query=QUERY,
        query_embedding=query_embedding,
        limit=50,
        document_ids=[DOCUMENT_ID],
    )

    print()
    print("=== HYBRID TOP 50 ===")
    print(f"Candidates: {len(retrieved)}")

    for rank, result in enumerate(retrieved, start=1):
        content = result["content"]

        evidence_flag = (
            "profit before tax" in content.lower()
            or "29.9" in content
            or "32.3" in content
        )

        marker = " <<< POSSIBLE PBT EVIDENCE" if evidence_flag else ""

        print(
            f"{rank:02d}. "
            f"page={result['page']} "
            f"chunk={result['chunk']} "
            f"semantic={float(result['semantic_similarity']):.6f} "
            f"keyword={float(result['keyword_score']):.6f}"
            f"{marker}"
        )

        if evidence_flag:
            print(
                f"    {excerpt(content)}"
            )

    # ---------------------------------------------------------
    # 3. Apply the existing Python reranker to the same pool.
    # ---------------------------------------------------------

    ranked = rerank_results(
        QUERY,
        retrieved,
    )

    print()
    print("=== RERANKED TOP 50 ===")

    for rank, result in enumerate(ranked, start=1):
        content = result["content"]

        evidence_flag = (
            "profit before tax" in content.lower()
            or "29.9" in content
            or "32.3" in content
        )

        marker = " <<< POSSIBLE PBT EVIDENCE" if evidence_flag else ""

        print(
            f"{rank:02d}. "
            f"page={result['page']} "
            f"chunk={result['chunk']} "
            f"semantic={float(result['semantic_similarity']):.6f} "
            f"overlap={float(result['keyword_overlap']):.6f} "
            f"phrase={float(result['phrase_bonus']):.6f} "
            f"final={float(result['final_score']):.6f}"
            f"{marker}"
        )

        if evidence_flag:
            print(
                f"    {excerpt(content)}"
            )


if __name__ == "__main__":
    main()
