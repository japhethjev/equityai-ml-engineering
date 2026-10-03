from unittest.mock import patch

import pytest

from app.rag.ingestion_preparation import (
    prepare_document_ingestion,
)


@patch(
    "app.rag.ingestion_preparation.upload_document_file"
)
@patch(
    "app.rag.ingestion_preparation.register_document"
)
@patch(
    "app.rag.ingestion_preparation.calculate_file_hash"
)
def test_prepare_new_document(
    mock_hash,
    mock_register,
    mock_upload,
    tmp_path,
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-test")

    mock_hash.return_value = "sha256-test"

    mock_register.return_value = {
        "document_id": "doc-123",
        "status": "queued",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": True,
    }

    mock_upload.return_value = {
        "bucket": "equityai-documents",
        "object_key": "documents/doc-123/report.pdf",
    }

    result = prepare_document_ingestion(
        str(pdf),
        document_name="report.pdf",
        company_name="Example Plc",
        ticker="EXM",
        exchange="NGX",
        country="Nigeria",
        currency="NGN",
        document_type="financial_report",
        report_type="annual",
        fiscal_year=2025,
        period_end="2025-12-31",
    )

    assert result["status"] == "prepared"
    assert result["document_id"] == "doc-123"
    assert result["document_hash"] == "sha256-test"

    assert (
        result["registry"]["document_hash"]
        == "sha256-test"
    )

    mock_upload.assert_called_once_with(
        file_path=str(pdf),
        document_id="doc-123",
        filename="report.pdf",
    )


@patch(
    "app.rag.ingestion_preparation.upload_document_file"
)
@patch(
    "app.rag.ingestion_preparation.register_document"
)
@patch(
    "app.rag.ingestion_preparation.calculate_file_hash"
)
def test_completed_duplicate_skips_storage(
    mock_hash,
    mock_register,
    mock_upload,
    tmp_path,
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-test")

    mock_hash.return_value = "duplicate-hash"

    mock_register.return_value = {
        "document_id": "doc-existing",
        "status": "completed",
        "processed_pages": 100,
        "processed_chunks": 300,
        "last_processed_page": 100,
        "total_pages": 100,
        "total_chunks": 300,
        "is_new": False,
    }

    result = prepare_document_ingestion(
        str(pdf),
        document_name="report.pdf",
    )

    assert result["status"] == "already_ingested"
    assert result["storage"] is None

    mock_upload.assert_not_called()


@patch(
    "app.rag.ingestion_preparation.upload_document_file"
)
@patch(
    "app.rag.ingestion_preparation.register_document"
)
@patch(
    "app.rag.ingestion_preparation.calculate_file_hash"
)
def test_processing_duplicate_is_rejected(
    mock_hash,
    mock_register,
    mock_upload,
    tmp_path,
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-test")

    mock_hash.return_value = "processing-hash"

    mock_register.return_value = {
        "document_id": "doc-processing",
        "status": "processing",
        "processed_pages": 10,
        "processed_chunks": 20,
        "last_processed_page": 10,
        "total_pages": None,
        "total_chunks": None,
        "is_new": False,
    }

    with pytest.raises(
        RuntimeError,
        match="already being processed",
    ):
        prepare_document_ingestion(
            str(pdf),
            document_name="report.pdf",
        )

    mock_upload.assert_not_called()


def test_prepare_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        prepare_document_ingestion(
            "/missing/report.pdf",
            document_name="report.pdf",
        )


@patch(
    "app.rag.ingestion_preparation.claim_document_for_dispatch"
)
@patch(
    "app.rag.ingestion_preparation.upload_document_file"
)
@patch(
    "app.rag.ingestion_preparation.register_document"
)
@patch(
    "app.rag.ingestion_preparation.calculate_file_hash"
)
def test_queued_duplicate_skips_storage(
    mock_hash,
    mock_register,
    mock_upload,
    mock_claim,
    tmp_path,
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-test")

    mock_hash.return_value = "queued-hash"

    mock_register.return_value = {
        "document_id": "doc-queued",
        "status": "queued",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": False,
    }

    mock_claim.return_value = {
        **mock_register.return_value,
        "claimed": False,
    }

    result = prepare_document_ingestion(
        str(pdf),
        document_name="report.pdf",
    )

    assert result["document_id"] == "doc-queued"
    assert result["status"] == "already_queued"
    assert result["storage"] is None

    mock_claim.assert_called_once_with(
        "doc-queued"
    )
    mock_upload.assert_not_called()


@patch(
    "app.rag.ingestion_preparation.fail_document"
)
@patch(
    "app.rag.ingestion_preparation.upload_document_file"
)
@patch(
    "app.rag.ingestion_preparation.register_document"
)
@patch(
    "app.rag.ingestion_preparation.calculate_file_hash"
)
def test_storage_failure_marks_document_failed(
    mock_hash,
    mock_register,
    mock_upload,
    mock_fail,
    tmp_path,
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-test")

    mock_hash.return_value = "storage-failure-hash"

    mock_register.return_value = {
        "document_id": "doc-storage-failure",
        "status": "queued",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": True,
    }

    mock_upload.side_effect = RuntimeError(
        "S3 unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="S3 unavailable",
    ):
        prepare_document_ingestion(
            str(pdf),
            document_name="report.pdf",
        )

    mock_fail.assert_called_once()

    kwargs = mock_fail.call_args.kwargs

    assert kwargs["document_id"] == (
        "doc-storage-failure"
    )
    assert "Failed to persist document to S3" in (
        kwargs["error_message"]
    )
    assert "S3 unavailable" in (
        kwargs["error_message"]
    )


@patch(
    "app.rag.ingestion_preparation.claim_document_for_dispatch"
)
@patch(
    "app.rag.ingestion_preparation.upload_document_file"
)
@patch(
    "app.rag.ingestion_preparation.register_document"
)
@patch(
    "app.rag.ingestion_preparation.calculate_file_hash"
)
def test_expired_queued_duplicate_is_reprepared(
    mock_hash,
    mock_register,
    mock_upload,
    mock_claim,
    tmp_path,
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-test")

    mock_hash.return_value = "queued-hash"

    mock_register.return_value = {
        "document_id": "doc-queued",
        "status": "queued",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": False,
    }

    mock_claim.return_value = {
        **mock_register.return_value,
        "claimed": True,
    }

    mock_upload.return_value = {
        "bucket": "equityai-documents",
        "object_key": (
            "documents/doc-queued/report.pdf"
        ),
    }

    result = prepare_document_ingestion(
        str(pdf),
        document_name="report.pdf",
    )

    assert result["document_id"] == "doc-queued"
    assert result["status"] == "prepared"

    mock_claim.assert_called_once_with(
        "doc-queued"
    )

    mock_upload.assert_called_once_with(
        file_path=str(pdf),
        document_id="doc-queued",
        filename="report.pdf",
    )
