from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.document_storage import (
    build_document_key,
    get_document_bucket,
    upload_document_file,
)


def test_get_document_bucket(monkeypatch):
    monkeypatch.setenv(
        "DOCUMENTS_S3_BUCKET",
        "equityai-documents",
    )

    assert get_document_bucket() == "equityai-documents"


def test_get_document_bucket_requires_configuration(
    monkeypatch,
):
    monkeypatch.delenv(
        "DOCUMENTS_S3_BUCKET",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="DOCUMENTS_S3_BUCKET",
    ):
        get_document_bucket()


def test_build_document_key():
    key = build_document_key(
        document_id="abc-123",
        filename="annual-report.pdf",
    )

    assert key == (
        "documents/abc-123/annual-report.pdf"
    )


def test_build_document_key_strips_directory_components():
    key = build_document_key(
        document_id="abc-123",
        filename="../../annual-report.pdf",
    )

    assert key == (
        "documents/abc-123/annual-report.pdf"
    )


def test_upload_document_file(tmp_path, monkeypatch):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-test")

    monkeypatch.setenv(
        "DOCUMENTS_S3_BUCKET",
        "equityai-documents",
    )

    mock_s3 = MagicMock()

    with patch(
        "app.infrastructure.document_storage."
        "boto3.client",
        return_value=mock_s3,
    ):
        result = upload_document_file(
            file_path=str(pdf),
            document_id="doc-123",
            filename="report.pdf",
        )

    mock_s3.upload_file.assert_called_once_with(
        str(pdf),
        "equityai-documents",
        "documents/doc-123/report.pdf",
        ExtraArgs={
            "ContentType": "application/pdf",
        },
    )

    assert result == {
        "bucket": "equityai-documents",
        "object_key": (
            "documents/doc-123/report.pdf"
        ),
    }


def test_upload_document_file_rejects_missing_file(
    monkeypatch,
):
    monkeypatch.setenv(
        "DOCUMENTS_S3_BUCKET",
        "equityai-documents",
    )

    with pytest.raises(FileNotFoundError):
        upload_document_file(
            file_path="/missing/report.pdf",
            document_id="doc-123",
            filename="report.pdf",
        )


def test_download_document_file(tmp_path):
    from app.infrastructure.document_storage import (
        download_document_file,
    )

    destination = tmp_path / "worker" / "report.pdf"

    mock_s3 = MagicMock()

    with patch(
        "app.infrastructure.document_storage."
        "boto3.client",
        return_value=mock_s3,
    ):
        result = download_document_file(
            bucket="equityai-documents",
            object_key="documents/doc-123/report.pdf",
            destination_path=str(destination),
        )

    mock_s3.download_file.assert_called_once_with(
        "equityai-documents",
        "documents/doc-123/report.pdf",
        str(destination),
    )

    assert result == str(destination)


def test_download_document_file_requires_bucket(
    tmp_path,
):
    from app.infrastructure.document_storage import (
        download_document_file,
    )

    with pytest.raises(
        ValueError,
        match="bucket",
    ):
        download_document_file(
            bucket="",
            object_key="documents/doc/report.pdf",
            destination_path=str(
                tmp_path / "report.pdf"
            ),
        )


def test_download_document_file_requires_object_key(
    tmp_path,
):
    from app.infrastructure.document_storage import (
        download_document_file,
    )

    with pytest.raises(
        ValueError,
        match="object_key",
    ):
        download_document_file(
            bucket="equityai-documents",
            object_key="",
            destination_path=str(
                tmp_path / "report.pdf"
            ),
        )


def test_document_object_exists_returns_true():
    from app.infrastructure.document_storage import (
        document_object_exists,
    )

    mock_s3 = MagicMock()

    with patch(
        "app.infrastructure.document_storage."
        "boto3.client",
        return_value=mock_s3,
    ):
        result = document_object_exists(
            bucket="equityai-documents",
            object_key="documents/doc-123/report.pdf",
        )

    assert result is True

    mock_s3.head_object.assert_called_once_with(
        Bucket="equityai-documents",
        Key="documents/doc-123/report.pdf",
    )


def test_document_object_exists_returns_false_for_404():
    from botocore.exceptions import ClientError

    from app.infrastructure.document_storage import (
        document_object_exists,
    )

    mock_s3 = MagicMock()

    mock_s3.head_object.side_effect = ClientError(
        {
            "Error": {
                "Code": "404",
                "Message": "Not Found",
            }
        },
        "HeadObject",
    )

    with patch(
        "app.infrastructure.document_storage."
        "boto3.client",
        return_value=mock_s3,
    ):
        result = document_object_exists(
            bucket="equityai-documents",
            object_key="documents/doc-123/report.pdf",
        )

    assert result is False


def test_document_object_exists_propagates_s3_errors():
    from botocore.exceptions import ClientError

    from app.infrastructure.document_storage import (
        document_object_exists,
    )

    mock_s3 = MagicMock()

    mock_s3.head_object.side_effect = ClientError(
        {
            "Error": {
                "Code": "AccessDenied",
                "Message": "Access denied",
            }
        },
        "HeadObject",
    )

    with patch(
        "app.infrastructure.document_storage."
        "boto3.client",
        return_value=mock_s3,
    ):
        with pytest.raises(ClientError):
            document_object_exists(
                bucket="equityai-documents",
                object_key=(
                    "documents/doc-123/report.pdf"
                ),
            )


def test_document_object_exists_validates_arguments():
    from app.infrastructure.document_storage import (
        document_object_exists,
    )

    with pytest.raises(
        ValueError,
        match="bucket",
    ):
        document_object_exists(
            bucket="",
            object_key="documents/doc/report.pdf",
        )

    with pytest.raises(
        ValueError,
        match="object_key",
    ):
        document_object_exists(
            bucket="equityai-documents",
            object_key="",
        )
