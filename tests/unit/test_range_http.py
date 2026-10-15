from urllib.error import HTTPError
from urllib.request import Request

import httpx
import pytest

from teutonic.evaluator import document_index, range_http


class Stream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
        self.reads = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.reads += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_pooled_transport_retries_http_and_interrupted_body(tmp_path, monkeypatch, status):
    partial = Stream([b"ab", httpx.RemoteProtocolError("interrupted")])
    attempts = []
    delays = []

    def serve(request):
        attempts.append(request)
        assert request.headers["Range"] == "bytes=0-7"
        if len(attempts) == 1:
            return httpx.Response(status, headers={"Retry-After": "12"})
        return httpx.Response(
            206, headers={"Content-Range": "bytes 0-7/8"},
            stream=partial if len(attempts) == 2 else Stream([b"abcdefgh"]),
        )

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        monkeypatch.setattr(range_http, "_client", client)
        monkeypatch.setattr(document_index.time, "sleep", delays.append)
        monkeypatch.setattr(document_index.random, "uniform", lambda *_: 0)
        reader = document_index.RangeObject(
            {"url": "https://dataset.example/a.npy", "size_bytes": 8},
            tmp_path, range_http.pooled_urlopen,
        )
        assert reader._range(0, 7, "test") == b"abcdefgh"
        assert len(attempts) == 3 and len(delays) == 2
        assert delays == [12, 10]
        assert partial.closed
        assert reader.downloaded_bytes == 8


@pytest.mark.parametrize("status, retry_after, expected_delays", [
    (429, None, [5, 10, 20, 40]),
    (429, "300", []),
    (403, None, []),
    (404, None, []),
    (412, None, []),
])
def test_pooled_transport_retry_limits_and_permanent_failures(
    tmp_path, monkeypatch, status, retry_after, expected_delays,
):
    attempts, delays, streams = [], [], []

    def serve(request):
        attempts.append(request)
        stream = Stream([AssertionError("error body must not be downloaded")])
        streams.append(stream)
        return httpx.Response(
            status, headers={"Retry-After": retry_after} if retry_after else {}, stream=stream,
        )

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        monkeypatch.setattr(range_http, "_client", client)
        monkeypatch.setattr(document_index.time, "sleep", delays.append)
        monkeypatch.setattr(document_index.random, "uniform", lambda *_: 0)
        reader = document_index.RangeObject(
            {"url": "https://dataset.example/a.npy", "size_bytes": 8},
            tmp_path, range_http.pooled_urlopen,
        )
        error = document_index.RangeDownloadError if status == 429 else HTTPError
        with pytest.raises(error):
            reader._range(0, 7, "test")
        assert delays == expected_delays
        assert len(attempts) == len(expected_delays) + 1
        assert all(stream.closed and stream.reads == 0 for stream in streams)
        assert not reader.cache.exists()
        assert reader.downloaded_bytes == 0


def test_pooled_transport_bounds_body_and_closes_stream(monkeypatch):
    stream = Stream([b"a" * 100, AssertionError("must stop reading at limit")])
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(206, stream=stream)
    )) as client:
        monkeypatch.setattr(range_http, "_client", client)
        with range_http.pooled_urlopen(Request("https://dataset.example/a.npy"), 120) as response:
            assert response.read(9) == b"a" * 9
        assert stream.closed and stream.reads == 1


def test_pooled_transport_rejects_full_response_without_reading(tmp_path, monkeypatch):
    stream = Stream([AssertionError("must not read a full response")])
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, stream=stream)
    )) as client:
        monkeypatch.setattr(range_http, "_client", client)
        reader = document_index.RangeObject(
            {"url": "https://dataset.example/a.npy", "size_bytes": 8},
            tmp_path, range_http.pooled_urlopen,
        )
        with pytest.raises(RuntimeError, match="exact byte range"):
            reader._range(0, 7, "test")
        assert stream.closed and stream.reads == 0


@pytest.fixture
def dataset_auth(monkeypatch):
    values = {
        "ENDPOINT": "https://account.r2.cloudflarestorage.com",
        "ACCESS_KEY_ID": "test-access",
        "SECRET_ACCESS_KEY": "test-secret",
        "SESSION_TOKEN": "test-session",
        "REGION": "auto",
        "BUCKET": "models-must-not-be-used",
    }
    for key, value in values.items():
        monkeypatch.setenv(f"TEUTONIC_EVALUATOR_R2_{key}", value)


def test_authenticated_index_range_keeps_pins_and_cache_identity(
    tmp_path, monkeypatch, dataset_auth,
):
    import hashlib
    import json

    descriptor = {
        "url": range_http.DATASET_ORIGIN + "/doc_index/a%20b.npy",
        "size_bytes": 8, "etag": '"abc123"',
    }
    requests = []

    def serve(request):
        requests.append(request)
        assert str(request.url) == "https://account.r2.cloudflarestorage.com/datasets/doc_index/a%20b.npy"
        assert request.headers["Range"] == "bytes=0-7"
        assert request.headers["If-Match"] == descriptor["etag"]
        assert request.headers["Accept-Encoding"] == "identity"
        auth = request.headers["Authorization"]
        assert "Credential=test-access/" in auth
        assert "/auto/s3/aws4_request" in auth
        assert "if-match;range;" in auth
        assert request.headers["X-Amz-Security-Token"] == "test-session"
        assert "test-secret" not in str(request.url)
        return httpx.Response(206, headers={
            "Content-Range": "bytes 0-7/8", "ETag": descriptor["etag"],
        }, stream=Stream([b"abcdefgh"]))

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        monkeypatch.setattr(range_http, "_client", client)
        reader = document_index.PinnedObject(descriptor, tmp_path)
        expected_cache = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        assert reader.cache.name == expected_cache
        assert reader._range(0, 7, "0") == b"abcdefgh"
        assert reader._range(0, 7, "0") == b"abcdefgh"
        assert reader.descriptor == descriptor
        assert len(requests) == 1


@pytest.mark.parametrize("status", [403, 302])
def test_auth_failure_or_redirect_never_falls_back(monkeypatch, dataset_auth, status):
    requests = []
    url = range_http.DATASET_ORIGIN + "/shard.npy"

    def serve(request):
        requests.append(request)
        return httpx.Response(status, headers={"Location": "https://other.example/shard.npy"})

    with httpx.Client(transport=httpx.MockTransport(serve), follow_redirects=True) as client:
        monkeypatch.setattr(range_http, "_client", client)
        with pytest.raises(HTTPError) as error:
            with range_http.pooled_urlopen(Request(url), 120):
                pytest.fail("must fail before yielding response")
        assert error.value.url == url
        assert error.value.code == status
        assert len(requests) == 1


@pytest.mark.parametrize("url", [
    "https://other.example/a.npy",
    range_http.DATASET_ORIGIN + ".other.example/a.npy",
    range_http.DATASET_ORIGIN + "/a.npy?version=1",
    range_http.DATASET_ORIGIN.replace("https:", "http:") + "/a.npy",
])
def test_auth_only_applies_to_known_origin(dataset_auth, url):
    target, headers, authenticated = range_http._request_target(Request(url))
    assert target == url
    assert not authenticated
    assert "Authorization" not in headers


def test_auth_environment_is_read_lazily(monkeypatch, dataset_auth):
    request = Request(range_http.DATASET_ORIGIN + "/a.npy")
    for key in ("ENDPOINT", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY"):
        monkeypatch.delenv(f"TEUTONIC_EVALUATOR_R2_{key}")
    assert range_http._request_target(request) == (request.full_url, {}, False)
    monkeypatch.setenv("TEUTONIC_EVALUATOR_R2_ENDPOINT", "https://account.example")
    with pytest.raises(RuntimeError, match="incomplete"):
        range_http._request_target(request)
    monkeypatch.setenv("TEUTONIC_EVALUATOR_R2_ACCESS_KEY_ID", "new-access")
    monkeypatch.setenv("TEUTONIC_EVALUATOR_R2_SECRET_ACCESS_KEY", "new-secret")
    target, headers, authenticated = range_http._request_target(request)
    assert authenticated
    assert "Credential=new-access/" in headers["Authorization"]
    assert target == "https://account.example/datasets/a.npy"
