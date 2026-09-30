import os

import numpy as np
import psycopg

from pgvector.psycopg import register_vector


def get_connection():
    """
    Create and return a PostgreSQL connection
    with pgvector support registered.
    """

    connection = psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "equityai"),
        user=os.getenv("DB_USER", "equityai"),
        password=os.getenv(
            "DB_PASSWORD",
            "equityai_dev_password",
        ),
    )

    register_vector(connection)

    return connection


def chunk_exists(
    document_id: str,
    page: int,
    chunk: int,
) -> bool:
    """
    Check whether a chunk already exists for a specific
    registered document.

    document_id is the authoritative document identity.
    This prevents filings from different companies, years,
    quarters or versions interfering with one another.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM document_chunks
                    WHERE document_id = %s
                      AND page = %s
                      AND chunk = %s
                )
                """,
                (
                    document_id,
                    page,
                    chunk,
                ),
            )

            result = cursor.fetchone()

    return bool(result[0])

def store_chunk(
    chunk: dict,
    embedding: list[float],
) -> bool:
    """
    Store one document chunk and its embedding.

    Returns True if a new chunk was inserted.
    Returns False if the chunk already exists.
    """

    embedding_vector = np.array(
        embedding,
        dtype=np.float32,
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO document_chunks
                (
                    document,
                    page,
                    chunk,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s)

                ON CONFLICT (document, page, chunk)
                DO NOTHING

                RETURNING id
                """,
                (
                    chunk["document"],
                    chunk["page"],
                    chunk["chunk"],
                    chunk["text"],
                    embedding_vector,
                ),
            )

            inserted = cursor.fetchone()

        connection.commit()

    return inserted is not None


def store_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    document_id: str | None = None,
) -> dict:
    """
    Store multiple document chunks and embeddings.

    Database-level duplicate protection remains active
    even though ingestion checks for duplicates before
    generating embeddings.

    Returns counts of inserted and skipped chunks.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunks and embeddings must have the same length"
        )

    inserted_count = 0
    skipped_count = 0

    with get_connection() as connection:
        with connection.cursor() as cursor:

            for chunk, embedding in zip(
                chunks,
                embeddings,
            ):
                embedding_vector = np.array(
                    embedding,
                    dtype=np.float32,
                )

                cursor.execute(
                    """
                    INSERT INTO document_chunks
                    (
                        document,
                        page,
                        chunk,
                        content,
                        embedding,
                        document_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)

                    ON CONFLICT (document_id, page, chunk)
                    WHERE document_id IS NOT NULL
                    DO NOTHING

                    RETURNING id
                    """,
                    (
                        chunk["document"],
                        chunk["page"],
                        chunk["chunk"],
                        chunk["text"],
                        embedding_vector,
                        document_id,
                    ),
                )

                inserted = cursor.fetchone()

                if inserted is None:
                    skipped_count += 1
                else:
                    inserted_count += 1

        connection.commit()

    return {
        "inserted": inserted_count,
        "skipped": skipped_count,
    }


def search_similar_chunks(
    query_embedding: list[float],
    limit: int = 3,
) -> list[dict]:
    """
    Search the vector database and return
    the most semantically similar chunks.
    """

    query_vector = np.array(
        query_embedding,
        dtype=np.float32,
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    document,
                    page,
                    chunk,
                    content,
                    1 - (embedding <=> %s) AS similarity

                FROM document_chunks

                ORDER BY embedding <=> %s

                LIMIT %s
                """,
                (
                    query_vector,
                    query_vector,
                    limit,
                ),
            )

            rows = cursor.fetchall()

    return [
        {
            "document": row[0],
            "page": row[1],
            "chunk": row[2],
            "content": row[3],
            "similarity": row[4],
        }
        for row in rows
    ]


def hybrid_search(
    query: str,
    query_embedding: list[float],
    limit: int = 5,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Combine semantic vector similarity with PostgreSQL
    keyword matching.

    When multiple document IDs are supplied, retrieve
    candidates independently from each document so one
    filing cannot monopolise the comparison candidate set.

    Semantic similarity receives 70% weight.
    Keyword relevance receives 30% weight.
    """

    query_vector = np.array(
        query_embedding,
        dtype=np.float32,
    )

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    normalized_document_ids = (
        list(
            dict.fromkeys(
                str(document_id)
                for document_id in document_ids
            )
        )
        if document_ids
        else []
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:

            if normalized_document_ids:

                # Retrieve independently from each requested
                # filing. This guarantees candidate coverage
                # for financial comparisons.
                rows = []

                for document_id in normalized_document_ids:
                    cursor.execute(
                        """
                        SELECT
                            dc.document,
                            dc.page,
                            dc.chunk,
                            dc.content,
                            dc.document_id,

                            1 - (dc.embedding <=> %s)
                                AS semantic_similarity,

                            ts_rank(
                                to_tsvector(
                                    'english',
                                    dc.content
                                ),
                                plainto_tsquery(
                                    'english',
                                    %s
                                )
                            ) AS keyword_score

                        FROM document_chunks AS dc

                        WHERE dc.document_id = %s::uuid

                        ORDER BY
                            (
                                0.7 * (
                                    1 - (
                                        dc.embedding <=> %s
                                    )
                                )
                                +
                                0.3 * ts_rank(
                                    to_tsvector(
                                        'english',
                                        dc.content
                                    ),
                                    plainto_tsquery(
                                        'english',
                                        %s
                                    )
                                )
                            ) DESC

                        LIMIT %s
                        """,
                        (
                            query_vector,
                            query,
                            document_id,
                            query_vector,
                            query,
                            limit,
                        ),
                    )

                    rows.extend(
                        cursor.fetchall()
                    )

            else:
                cursor.execute(
                    """
                    SELECT
                        dc.document,
                        dc.page,
                        dc.chunk,
                        dc.content,
                        dc.document_id,

                        1 - (dc.embedding <=> %s)
                            AS semantic_similarity,

                        ts_rank(
                            to_tsvector('english', dc.content),
                            plainto_tsquery('english', %s)
                        ) AS keyword_score

                    FROM document_chunks AS dc

                    ORDER BY
                        (
                            0.7 * (
                                1 - (dc.embedding <=> %s)
                            )
                            +
                            0.3 * ts_rank(
                                to_tsvector(
                                    'english',
                                    dc.content
                                ),
                                plainto_tsquery(
                                    'english',
                                    %s
                                )
                            )
                        ) DESC

                    LIMIT %s
                    """,
                    (
                        query_vector,
                        query,
                        query_vector,
                        query,
                        limit,
                    ),
                )

                rows = cursor.fetchall()

    return [
        {
            "document": row[0],
            "page": row[1],
            "chunk": row[2],
            "content": row[3],
            "document_id": (
                str(row[4])
                if row[4] is not None
                else None
            ),
            "semantic_similarity": row[5],
            "keyword_score": row[6],
        }
        for row in rows
    ]


def list_documents() -> list[dict]:
    """
    Return all documents currently stored in the
    EquityAI knowledge base with page and chunk counts.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    document,
                    COUNT(DISTINCT page) AS pages,
                    COUNT(*) AS chunks
                FROM document_chunks
                GROUP BY document
                ORDER BY document
                """
            )

            rows = cursor.fetchall()

    return [
        {
            "document": row[0],
            "pages": row[1],
            "chunks": row[2],
        }
        for row in rows
    ]


def delete_document(
    document_name: str,
) -> int:
    """
    Delete all chunks and embeddings belonging
    to a document.

    Returns the number of deleted chunks.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM document_chunks
                WHERE document = %s
                """,
                (document_name,),
            )

            deleted_count = cursor.rowcount

        connection.commit()

    return deleted_count


def register_document(
    document_name: str,
    document_hash: str | None = None,
    company_name: str | None = None,
    ticker: str | None = None,
    market: str | None = None,
    exchange: str | None = None,
    country: str | None = None,
    currency: str | None = None,
    document_type: str | None = None,
    report_type: str | None = None,
    reporting_period: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
    fiscal_half: int | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
    publication_date: str | None = None,
) -> dict:
    """
    Create or retrieve a document registry record.

    Exact duplicate files are identified by SHA-256 hash.

    Returns document identity and ingestion state so callers can
    distinguish new uploads, completed duplicates and resumable
    failed/interrupted documents.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:

            if document_hash:
                cursor.execute(
                    """
                    SELECT
                        document_id,
                        ingestion_status,
                        processed_pages,
                        processed_chunks,
                        last_processed_page,
                        total_pages,
                        total_chunks
                    FROM documents
                    WHERE document_hash = %s
                    """,
                    (document_hash,),
                )

                existing = cursor.fetchone()

                if existing:
                    return {
                        "document_id": str(existing[0]),
                        "status": existing[1],
                        "processed_pages": existing[2] or 0,
                        "processed_chunks": existing[3] or 0,
                        "last_processed_page": existing[4] or 0,
                        "total_pages": existing[5],
                        "total_chunks": existing[6],
                        "is_new": False,
                    }

            cursor.execute(
                """
                INSERT INTO documents (
                    document_name,
                    document_hash,
                    company_name,
                    ticker,
                    market,
                    exchange,
                    country,
                    currency,
                    document_type,
                    report_type,
                    reporting_period,
                    fiscal_year,
                    fiscal_quarter,
                    fiscal_half,
                    period_start,
                    period_end,
                    publication_date,
                    ingestion_status
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, 'processing'
                )
                RETURNING document_id
                """,
                (
                    document_name,
                    document_hash,
                    company_name,
                    ticker,
                    market,
                    exchange,
                    country,
                    currency,
                    document_type,
                    report_type,
                    reporting_period,
                    fiscal_year,
                    fiscal_quarter,
                    fiscal_half,
                    period_start,
                    period_end,
                    publication_date,
                ),
            )

            document_id = cursor.fetchone()[0]

        connection.commit()

    return {
        "document_id": str(document_id),
        "status": "processing",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": True,
    }

def update_document_progress(
    document_id: str,
    processed_pages: int,
    processed_chunks: int,
    last_processed_page: int,
) -> None:
    """Persist ingestion progress for resumability."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE documents
                SET processed_pages = %s,
                    processed_chunks = %s,
                    last_processed_page = %s,
                    ingestion_status = 'processing',
                    error_message = NULL
                WHERE document_id = %s
                """,
                (
                    processed_pages,
                    processed_chunks,
                    last_processed_page,
                    document_id,
                ),
            )

        connection.commit()


def complete_document(
    document_id: str,
    total_pages: int,
    total_chunks: int,
) -> None:
    """Mark ingestion as successfully completed."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE documents
                SET ingestion_status = 'completed',
                    total_pages = %s,
                    processed_pages = %s,
                    total_chunks = %s,
                    processed_chunks = %s,
                    last_processed_page = %s,
                    error_message = NULL
                WHERE document_id = %s
                """,
                (
                    total_pages,
                    total_pages,
                    total_chunks,
                    total_chunks,
                    total_pages,
                    document_id,
                ),
            )

        connection.commit()


def fail_document(
    document_id: str,
    error_message: str,
) -> None:
    """Record an ingestion failure without deleting prior progress."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE documents
                SET ingestion_status = 'failed',
                    error_message = %s
                WHERE document_id = %s
                """,
                (
                    error_message[:4000],
                    document_id,
                ),
            )

        connection.commit()


def resolve_documents(
    ticker: str,
    exchange: str | None = None,
    report_type: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
    fiscal_half: int | None = None,
    current_only: bool = True,
) -> list[dict]:
    """
    Resolve financial-report metadata to registry documents.

    Resolution occurs before vector retrieval so semantically
    similar content from the wrong reporting period cannot enter
    the candidate set.

    Multiple documents may intentionally be returned.
    """

    normalized_ticker = ticker.strip().upper()

    if not normalized_ticker:
        raise ValueError("ticker is required")

    conditions = [
        "UPPER(ticker) = %s",
        "ingestion_status = 'completed'",
    ]

    parameters = [normalized_ticker]

    if exchange:
        conditions.append("UPPER(exchange) = %s")
        parameters.append(exchange.strip().upper())

    if report_type:
        conditions.append("LOWER(report_type) = %s")
        parameters.append(
            report_type.strip().lower()
        )

    if fiscal_year is not None:
        conditions.append("fiscal_year = %s")
        parameters.append(fiscal_year)

    if fiscal_quarter is not None:
        conditions.append("fiscal_quarter = %s")
        parameters.append(fiscal_quarter)

    if fiscal_half is not None:
        conditions.append("fiscal_half = %s")
        parameters.append(fiscal_half)

    if current_only:
        conditions.append("is_current = TRUE")

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT
            document_id,
            document_name,
            company_name,
            ticker,
            exchange,
            market,
            country,
            currency,
            report_type,
            fiscal_year,
            fiscal_quarter,
            fiscal_half,
            period_start,
            period_end,
            publication_date,
            filing_version,
            is_current
        FROM documents
        WHERE {where_clause}
        ORDER BY
            period_end DESC NULLS LAST,
            publication_date DESC NULLS LAST,
            filing_version DESC,
            created_at DESC
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql,
                tuple(parameters),
            )

            rows = cursor.fetchall()

    return [
        {
            "document_id": str(row[0]),
            "document_name": row[1],
            "company_name": row[2],
            "ticker": row[3],
            "exchange": row[4],
            "market": row[5],
            "country": row[6],
            "currency": row[7],
            "report_type": row[8],
            "fiscal_year": row[9],
            "fiscal_quarter": row[10],
            "fiscal_half": row[11],
            "period_start": (
                row[12].isoformat()
                if row[12]
                else None
            ),
            "period_end": (
                row[13].isoformat()
                if row[13]
                else None
            ),
            "publication_date": (
                row[14].isoformat()
                if row[14]
                else None
            ),
            "filing_version": row[15],
            "is_current": row[16],
        }
        for row in rows
    ]


def list_registered_companies() -> list[dict]:
    """
    Return distinct companies represented by completed,
    current financial reports.

    Company identity includes exchange because ticker symbols
    are not globally unique.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT
                    company_name,
                    ticker,
                    exchange,
                    market,
                    country
                FROM documents
                WHERE ingestion_status = 'completed'
                  AND is_current = TRUE
                  AND ticker IS NOT NULL
                ORDER BY
                    company_name,
                    exchange,
                    ticker
                """
            )

            rows = cursor.fetchall()

    return [
        {
            "company_name": row[0],
            "ticker": row[1],
            "exchange": row[2],
            "market": row[3],
            "country": row[4],
        }
        for row in rows
    ]
