"""Durable S3 storage for uploaded EquityAI documents."""

import os
from pathlib import Path

import boto3


def get_document_bucket() -> str:
    """Return the configured production document bucket."""

    bucket = os.getenv("DOCUMENTS_S3_BUCKET", "").strip()

    if not bucket:
        raise RuntimeError(
            "DOCUMENTS_S3_BUCKET environment variable is required."
        )

    return bucket


def normalize_document_filename(filename: str) -> str:
    """Return a basename safe for use in document S3 object keys."""

    normalized = filename.strip().replace("\\", "/")
    safe_filename = Path(normalized).name

    if not safe_filename:
        raise ValueError("filename is required")

    return safe_filename


def build_document_key(
    document_id: str,
    filename: str,
) -> str:
    """
    Build a safe, deterministic S3 object key.

    document_id separates individual registry documents while
    Path.name prevents caller-supplied directory traversal.
    """

    safe_filename = normalize_document_filename(filename)

    if not document_id.strip():
        raise ValueError("document_id is required")

    return (
        f"documents/"
        f"{document_id.strip()}/"
        f"{safe_filename}"
    )


def upload_document_file(
    file_path: str,
    document_id: str,
    filename: str,
) -> dict:
    """Upload a local PDF to durable S3 document storage."""

    source_path = Path(file_path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Document file does not exist: {file_path}"
        )

    bucket = get_document_bucket()
    object_key = build_document_key(
        document_id=document_id,
        filename=filename,
    )

    s3 = boto3.client("s3")

    s3.upload_file(
        str(source_path),
        bucket,
        object_key,
        ExtraArgs={
            "ContentType": "application/pdf",
        },
    )

    return {
        "bucket": bucket,
        "object_key": object_key,
    }


def create_document_upload_url(
    upload_id: str,
    filename: str,
    expires_in: int = 900,
) -> dict:
    """
    Create a short-lived presigned S3 PUT URL for an admin PDF upload.

    The object is staged first. Registration, hashing and ingestion
    dispatch occur only after the backend confirms the staged upload.
    """
    safe_filename = normalize_document_filename(filename)

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if not upload_id.strip():
        raise ValueError("upload_id is required")

    bucket = get_document_bucket()
    object_key = (
        f"staging/{upload_id.strip()}/{safe_filename}"
    )

    s3 = boto3.client("s3")
    upload_url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": object_key,
            "ContentType": "application/pdf",
        },
        ExpiresIn=expires_in,
    )

    return {
        "bucket": bucket,
        "object_key": object_key,
        "upload_url": upload_url,
        "expires_in": expires_in,
    }


def download_document_file(
    bucket: str,
    object_key: str,
    destination_path: str,
) -> str:
    """Download a stored document from S3 for worker processing."""

    if not bucket.strip():
        raise ValueError("bucket is required")

    if not object_key.strip():
        raise ValueError("object_key is required")

    destination = Path(destination_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3 = boto3.client("s3")

    s3.download_file(
        bucket.strip(),
        object_key.strip(),
        str(destination),
    )

    return str(destination)


def document_object_exists(
    bucket: str,
    object_key: str,
) -> bool:
    """
    Return whether a document object exists in S3.

    Only a genuine S3 not-found response is converted to False.
    Other AWS errors are propagated so infrastructure failures are
    not incorrectly treated as missing documents.
    """

    if not bucket.strip():
        raise ValueError("bucket is required")

    if not object_key.strip():
        raise ValueError("object_key is required")

    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")

    try:
        s3.head_object(
            Bucket=bucket.strip(),
            Key=object_key.strip(),
        )

    except ClientError as exc:
        error = exc.response.get("Error", {})
        code = str(error.get("Code", ""))

        if code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        raise

    return True


def delete_document_object(
    bucket: str,
    object_key: str,
) -> None:
    """
    Delete a document object from S3.

    Used to remove temporary staging objects only after the worker has
    safely prepared the document or resolved it as a duplicate.
    """
    if not bucket.strip():
        raise ValueError("bucket is required")

    if not object_key.strip():
        raise ValueError("object_key is required")

    s3 = boto3.client("s3")

    s3.delete_object(
        Bucket=bucket.strip(),
        Key=object_key.strip(),
    )
