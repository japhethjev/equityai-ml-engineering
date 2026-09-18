import os

from openai import OpenAI

from app.rag.embedding_service import embed_text
from app.rag.vector_store import hybrid_search
from app.rag.reranker import rerank_results


ABSTENTION_TEXT = (
    "I could not find sufficient evidence "
    "in the provided documents."
)


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


def answer_question(query: str) -> dict:
    """
    Answer a user question using retrieved document
    evidence only.

    The generation step distinguishes between:
    - relevant evidence,
    - directly sufficient evidence,
    - and information that would require calculation,
      inference, or derivation.
    """

    query = query.strip()

    if not query:
        raise ValueError("Question cannot be empty.")

    # -----------------------------------------
    # 1. Embed the user's question
    # -----------------------------------------

    query_embedding = embed_text(query)

    # -----------------------------------------
    # 2. Hybrid retrieval
    # -----------------------------------------

    retrieved = hybrid_search(
        query=query,
        query_embedding=query_embedding,
        limit=5,
    )

    # -----------------------------------------
    # 3. Rerank retrieved evidence
    # -----------------------------------------

    ranked = rerank_results(
        query,
        retrieved,
    )

    top_results = ranked[:3]

    # -----------------------------------------
    # 4. Handle empty retrieval
    # -----------------------------------------

    if not top_results:
        return {
            "answer": ABSTENTION_TEXT,
            "sources": [],
        }

    # -----------------------------------------
    # 5. Build grounded context
    # -----------------------------------------

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

    # -----------------------------------------
    # 6. Evidence-constrained generation
    # -----------------------------------------

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
   other figures in the evidence unless the user explicitly asks
   for a calculation.

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

11. If the question simply asks "what is", "what was", "how much",
    "state", "identify", or otherwise requests a factual value,
    do not automatically interpret that as permission to calculate
    a value that is absent from the evidence.

12. Perform a calculation only when the user explicitly asks you
    to calculate, compute, estimate, derive, determine from the
    supplied figures, or work out a value.

13. If the user explicitly requests a calculation and the required
    inputs are directly supported by the retrieved evidence, you
    may calculate the result. Clearly identify it as a calculated
    or derived result rather than a directly reported figure.

14. If the evidence does not directly and sufficiently answer the
    exact question, and the user has not explicitly requested an
    allowable calculation, respond with EXACTLY:

"{ABSTENTION_TEXT}"

Do not add a source, explanation, related figure, alternative
answer, calculation, or additional sentence after this abstention
response.

STAGE 2 — ANSWER

Only if Stage 1 determines that sufficient direct evidence exists,
or the user explicitly requested an allowable calculation:

15. Answer the exact question asked.

16. Keep the answer concise.

17. For a directly reported fact, state the supporting document
    and page.

18. For an explicitly requested calculation, state that the result
    is calculated or derived and identify the supporting document
    and page containing the input figures.

19. Do not present related information as though it were the
    requested fact.

20. Do not change the meaning, period, financial concept, or basis
    of a figure merely because another retrieved figure appears
    semantically similar.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}
"""

    # -----------------------------------------
    # 7. Generate grounded answer
    # -----------------------------------------

    client = get_openai_client()

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    answer = response.output_text.strip()

    # -----------------------------------------
    # 8. Return answer and retrieval metadata
    # -----------------------------------------

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
    }