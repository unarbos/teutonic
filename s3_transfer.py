"""Helpers for Hippius transfers.

Hippius S3 has been unreliable with boto3's managed transfer stack. For
Hippius endpoints we route file transfers through the MinIO SDK instead,
while non-Hippius callers keep using the boto3 client they already have.
"""
from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from minio import Minio


HIPPIUS_HOST_SUFFIX = ".hippius.com"
MINIO_PART_SIZE = 64 * 1024**2
MINIO_PARALLEL_UPLOADS = 1


def is_hippius_endpoint(endpoint_url: str | None) -> bool:
    if not endpoint_url:
        return False
    parsed = urlparse(endpoint_url)
    host = (parsed.hostname or endpoint_url).lower()
    return host == "hippius.com" or host.endswith(HIPPIUS_HOST_SUFFIX)


def _endpoint_parts(endpoint_url: str) -> tuple[str, bool]:
    parsed = urlparse(endpoint_url)
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    secure = scheme != "http"
    endpoint = parsed.netloc or parsed.path
    if not endpoint:
        raise ValueError(f"invalid endpoint URL: {endpoint_url!r}")
    return endpoint, secure


def _boto3_credentials(client) -> tuple[str, str, str | None]:
    creds = client._request_signer._credentials
    return creds.access_key, creds.secret_key, getattr(creds, "token", None)


@lru_cache(maxsize=8)
def _minio_client_for(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    session_token: str | None,
) -> Minio:
    endpoint, secure = _endpoint_parts(endpoint_url)
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        session_token=session_token,
        secure=secure,
    )


def _minio_client_from_boto3(client) -> Minio:
    endpoint_url = getattr(getattr(client, "meta", None), "endpoint_url", None)
    if not endpoint_url:
        raise ValueError("boto3 client missing endpoint_url")
    access_key, secret_key, session_token = _boto3_credentials(client)
    return _minio_client_for(endpoint_url, access_key, secret_key, session_token)


def safe_upload_file(client, filename: str, bucket: str, key: str) -> None:
    endpoint_url = getattr(getattr(client, "meta", None), "endpoint_url", None)
    if is_hippius_endpoint(endpoint_url):
        _minio_client_from_boto3(client).fput_object(
            bucket,
            key,
            filename,
            part_size=MINIO_PART_SIZE,
            num_parallel_uploads=MINIO_PARALLEL_UPLOADS,
        )
        return
    client.upload_file(filename, bucket, key)


def safe_download_file(client, bucket: str, key: str, filename: str) -> None:
    endpoint_url = getattr(getattr(client, "meta", None), "endpoint_url", None)
    if is_hippius_endpoint(endpoint_url):
        _minio_client_from_boto3(client).fget_object(bucket, key, filename)
        return
    client.download_file(bucket, key, filename)
