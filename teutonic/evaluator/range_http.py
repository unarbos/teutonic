"""Shared keep-alive transport for the many small token-shard range requests."""

import atexit
from contextlib import contextmanager
from urllib.error import HTTPError, URLError

import httpx

# Shared across shard workers; cap sockets even when reads use nested pools.
_client = httpx.Client(
    follow_redirects=True,
    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32, keepalive_expiry=60),
)
atexit.register(_client.close)


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
    try:
        with _client.stream(
            request.get_method(), request.full_url,
            headers=dict(request.header_items()), timeout=timeout,
        ) as response:
            if response.status_code >= 400:
                raise HTTPError(
                    request.full_url, response.status_code, response.reason_phrase,
                    response.headers, None,
                )
            yield _Response(response)
    except httpx.TransportError as exc:
        # Keep RangeObject's existing bounded retry and error classification.
        raise URLError(exc) from exc
