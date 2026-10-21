"""Shared keep-alive transport for the many small token-shard range requests."""

import atexit
import os
from contextlib import contextmanager
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit

import httpx

# Shared across shard workers; cap sockets even when reads use nested pools.
_client = httpx.Client(
    follow_redirects=True,
    limits=httpx.Limits(max_connections=128, max_keepalive_connections=128, keepalive_expiry=60),
)
atexit.register(_client.close)


# Only this known public origin maps to the dataset bucket. Model bucket settings
# must never affect dataset reads; unrelated request URLs remain public.
DATASET_ORIGIN = "https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev"


def _request_target(request):
    url = request.full_url
    headers = dict(request.header_items())
    parsed = urlsplit(url)
    if (
        f"{parsed.scheme}://{parsed.netloc}" != DATASET_ORIGIN
        or parsed.query or parsed.fragment
        or request.get_method() != "GET"
    ):
        return url, headers, False
    endpoint = os.environ.get("TEUTONIC_EVALUATOR_R2_ENDPOINT", "").strip()
    access_key = os.environ.get("TEUTONIC_EVALUATOR_R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("TEUTONIC_EVALUATOR_R2_SECRET_ACCESS_KEY", "").strip()
    if not any((endpoint, access_key, secret_key)):
        return url, headers, False
    if not all((endpoint, access_key, secret_key)):
        raise RuntimeError("evaluator dataset R2 credentials are incomplete")
    target = urlsplit(endpoint)
    if (
        target.scheme != "https" or not target.netloc or target.username is not None
        or target.password is not None or target.query or target.fragment
    ):
        raise ValueError("evaluator R2 endpoint must be an HTTPS URL without credentials or query")
    key = unquote(parsed.path).lstrip("/")
    if not key or any(part in {".", ".."} for part in key.split("/")):
        raise ValueError("invalid dataset object key")

    from botocore.auth import S3SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    url = f"{endpoint.rstrip('/')}/datasets/{quote(key, safe='/')}"
    # Sign each attempt with fresh headers, including Range and If-Match. Header
    # signing keeps credentials out of URLs, HTTP request logs and cache keys.
    signed = AWSRequest(method="GET", url=url, headers=headers)
    credentials = Credentials(
        access_key, secret_key,
        os.environ.get("TEUTONIC_EVALUATOR_R2_SESSION_TOKEN") or None,
    )
    S3SigV4Auth(
        credentials, "s3", os.environ.get("TEUTONIC_EVALUATOR_R2_REGION", "auto"),
    ).add_auth(signed)
    return url, dict(signed.headers.items()), True


class _Response:
    def __init__(self, response):
        self.response = response
        self.status = response.status_code
        self.headers = response.headers

    def read(self, size):
        # Do not buffer an unbounded response, even if the server lies about its size.
        body = bytearray()
        for chunk in self.response.iter_raw():
            body.extend(chunk[: size - len(body)])
            if len(body) == size:
                break
        return bytes(body)


@contextmanager
def pooled_urlopen(request, timeout):
    """Provide the small urllib response interface used by RangeObject."""
    url, headers, authenticated = _request_target(request)
    try:
        with _client.stream(
            request.get_method(), url,
            headers=headers, timeout=timeout, follow_redirects=not authenticated,
        ) as response:
            if response.status_code >= 400 or (authenticated and response.is_redirect):
                raise HTTPError(
                    request.full_url, response.status_code, response.reason_phrase,
                    response.headers, None,
                )
            yield _Response(response)
    except httpx.TransportError as exc:
        # Keep RangeObject's existing bounded retry and error classification.
        raise URLError(exc) from exc
