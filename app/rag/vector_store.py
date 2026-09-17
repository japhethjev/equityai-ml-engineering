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
    document: str,
    page: int,
    chunk: int,
) -> bool:
    """
    Check whether a document chunk already exists.

    This allows the ingestion pipeline to detect
    duplicates before generating OpenAI embeddings.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM document_chunks
                    WHERE document = %s
                      AND page = %s
                      AND chunk = %s
                )
                """,
                (
                    document,
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
) -> list[dict]:
    """
    Combine semantic vector similarity
    with PostgreSQL keyword matching.

    Semantic similarity receives 70% weight.
    Keyword relevance receives 30% weight.
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

                    1 - (embedding <=> %s)
                        AS semantic_similarity,

                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', %s)
                    ) AS keyword_score

                FROM document_chunks

                ORDER BY
                    (
                        0.7 * (1 - (embedding <=> %s))
                        +
                        0.3 * ts_rank(
                            to_tsvector('english', content),
                            plainto_tsquery('english', %s)
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
            "semantic_similarity": row[4],
            "keyword_score": row[5],
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
