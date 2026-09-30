from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader


def iter_pdf_pages(file_path: str) -> Iterator[dict]:
    """
    Yield PDF pages one at a time.

    No application-level page limit is imposed.
    Keeping page extraction incremental prevents the entire
    document's extracted text from being accumulated in memory.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        yield {
            "document": pdf_path.name,
            "page": page_number,
            "text": text.strip(),
        }


def load_pdf(file_path: str) -> list[dict]:
    """
    Backward-compatible helper used by existing tests/code.

    New large-document ingestion should use iter_pdf_pages()
    directly so pages are processed incrementally.
    """

    return list(iter_pdf_pages(file_path))
