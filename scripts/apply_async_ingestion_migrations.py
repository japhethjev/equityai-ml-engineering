"""
Apply the database migrations required by EquityAI asynchronous
document ingestion.

Designed for controlled production deployment before the new API
task definition is activated.
"""

from pathlib import Path

from app.rag.vector_store import get_connection


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIGRATIONS = [
    PROJECT_ROOT / "db/migrations/004_async_ingestion_status.sql",
    PROJECT_ROOT / "db/migrations/005_document_processing_lease.sql",
    PROJECT_ROOT / "db/migrations/006_document_dispatch_lease.sql",
]


def apply_migration(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Migration not found: {path}")

    sql = path.read_text(encoding="utf-8")

    print(f"Applying {path.name} ...", flush=True)

    # psycopg connection context commits on success and rolls back
    # automatically if execution raises an exception.
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)

    print(f"Applied {path.name}", flush=True)


def main() -> None:
    print("Starting EquityAI async-ingestion migrations.", flush=True)

    for migration in MIGRATIONS:
        apply_migration(migration)

    print("All async-ingestion migrations applied successfully.", flush=True)


if __name__ == "__main__":
    main()
