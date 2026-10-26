import gzip
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from scripts import serve_website_preview as preview


def response(body, etag='"public-v1"', encoding=None):
    result = io.BytesIO(body)
    result.headers = {"ETag": etag, "Content-Encoding": encoding}
    return result


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(preview, "monotonic", lambda: now[0])
    return now


def test_feed_cache_preserves_public_bytes_and_compresses_once(monkeypatch, clock):
    body = json.dumps({"history": ["public evaluation"] * 1000}).encode()
    upstream = Mock(return_value=response(body))
    monkeypatch.setattr(preview, "urlopen", upstream)
    cache = preview.PublicFeedCache()
    first = cache.get("/dashboard.json")
    assert cache.get("/dashboard.json") is first
    assert first.body == body
    assert gzip.decompress(first.compressed) == body
    assert len(first.compressed) < len(body) / 10
    assert first.expires_at == 105
    assert upstream.call_count == 1
    request = upstream.call_args.args[0]
    assert request.full_url == preview.PUBLIC_DASHBOARD_ORIGIN + "/dashboard.json"
    assert not request.has_header("Authorization")


def test_expired_feed_revalidates_and_changed_data_replaces_cache(monkeypatch, clock):
    upstream = Mock(side_effect=[
        response(b'{"version":1}'),
        HTTPError("url", 304, "unchanged", {}, None),
        response(b'{"version":2}', '"public-v2"'),
    ])
    monkeypatch.setattr(preview, "urlopen", upstream)
    cache = preview.PublicFeedCache()
    original = cache.get("/dashboard.json")
    clock[0] = 106
    unchanged = cache.get("/dashboard.json")
    assert unchanged.body == original.body
    assert unchanged.etag == original.etag
    assert unchanged.expires_at == 111
    assert upstream.call_args.args[0].get_header("If-none-match") == '"public-v1"'
    clock[0] = 112
    changed = cache.get("/dashboard.json")
    assert changed.body == b'{"version":2}'
    assert changed.etag != original.etag


def test_origin_gzip_and_manifest_freshness(monkeypatch, clock):
    body = json.dumps({"sources": [{"name": "category", "proportion": 1}] * 100}).encode()
    upstream = Mock(return_value=response(gzip.compress(body), encoding="gzip"))
    monkeypatch.setattr(preview, "urlopen", upstream)
    cache = preview.PublicFeedCache()
    entry = cache.get("/competitions/mix06/datasets/manifest.json")
    assert entry.body == body
    assert gzip.decompress(entry.compressed) == body
    clock[0] = 159
    assert cache.get("/competitions/mix06/datasets/manifest.json") is entry
    assert upstream.call_count == 1


@pytest.mark.parametrize("competition,prefix", [
    ("main", "king-benchmark-daily"),
    ("mix01", "competitions/mix01"),
    ("mix06", "competitions/mix06"),
])
@pytest.mark.parametrize("suffix,ttl", [
    ("all-kings/results.json", 60),
    ("all-kings/latest.json", 5),
    ("kings/reign-000001-uid-42/results.json", 5),
])
def test_public_benchmark_paths_and_running_refresh(competition, prefix, suffix, ttl):
    path = f"/benchmarks/{prefix}/{suffix}"
    assert preview.public_data_url(path) == preview.PUBLIC_BENCHMARK_ORIGIN + f"/{prefix}/{suffix}"
    assert preview.feed_cache_seconds(path) == ttl


@pytest.mark.parametrize("path", [
    "/benchmarks/competitions/mix07/all-kings/results.json",
    "/benchmarks/competitions/mix01/kings/../private/results.json",
    "/benchmarks/competitions/mix01/kings/king/model.bin",
    "/benchmarks/https://other.example/results.json",
])
def test_benchmark_proxy_rejects_unknown_files(path):
    assert preview.public_data_url(path) is None


def test_concurrent_visitors_share_one_upstream_request(monkeypatch, clock):
    upstream = Mock(side_effect=lambda *args, **kwargs: response(b'{"competitions":[]}'))
    monkeypatch.setattr(preview, "urlopen", upstream)
    cache = preview.PublicFeedCache()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cache.get("/competitions.json"), range(16)))
    assert upstream.call_count == 1
    assert all(entry is results[0] for entry in results)


def test_cache_errors_are_explicit_and_do_not_replace_good_data(monkeypatch, clock):
    upstream = Mock(side_effect=[response(b'{"version":1}'), URLError("offline")])
    monkeypatch.setattr(preview, "urlopen", upstream)
    cache = preview.PublicFeedCache()
    original = cache.get("/dashboard.json")
    clock[0] = 106
    with pytest.raises(URLError):
        cache.get("/dashboard.json")
    assert cache.entries["/dashboard.json"] is original
    with pytest.raises(ValueError):
        cache.get("/arbitrary-host-or-file")


@pytest.fixture
def handler(monkeypatch):
    captured = {}

    def server(address, handler_type):
        captured["type"] = handler_type
        return SimpleNamespace(serve_forever=lambda: None, server_close=lambda: None)

    monkeypatch.setattr(preview, "ThreadingHTTPServer", server)
    monkeypatch.setattr(sys, "argv", ["preview"])
    preview.main()
    instance = object.__new__(captured["type"])
    instance.command = "GET"
    instance.path = "/dashboard.json"
    instance.headers = {}
    instance.wfile = io.BytesIO()
    instance.send_response = Mock()
    instance.send_header = Mock()
    instance.end_headers = Mock()
    return instance


@pytest.mark.parametrize("arguments,expected", [
    ([], ("127.0.0.1", 8765)),
    (["--host", "0.0.0.0", "--port", "18765"], ("0.0.0.0", 18765)),
])
def test_bind_address_defaults_to_loopback_and_can_be_explicit(monkeypatch, arguments, expected):
    captured = []

    def server(address, handler_type):
        captured.append(address)
        return SimpleNamespace(serve_forever=lambda: None, server_close=lambda: None)

    monkeypatch.setattr(preview, "ThreadingHTTPServer", server)
    monkeypatch.setattr(sys, "argv", ["preview", *arguments])
    preview.main()
    assert captured == [expected]


def test_browser_etag_returns_no_body_and_gzip_preserves_data(handler):
    body = b'{"data":' + json.dumps(["shared categories"] * 1000).encode() + b'}'
    handler.headers = {"Accept-Encoding": "gzip"}
    handler.respond(body, "application/json", etag='W/"same"', cache_control="no-cache")
    assert gzip.decompress(handler.wfile.getvalue()) == body
    assert ("Content-Encoding", "gzip") in [call.args for call in handler.send_header.call_args_list]
    handler.wfile = io.BytesIO()
    handler.headers = {"If-None-Match": 'W/"same"', "Accept-Encoding": "gzip"}
    handler.respond(body, "application/json", etag='W/"same"', cache_control="no-cache")
    handler.send_response.assert_called_with(304)
    assert handler.wfile.getvalue() == b""


def test_gzip_opt_out_and_head_do_not_send_unwanted_bytes(handler):
    body = json.dumps({"data": ["public"] * 1000}).encode()
    handler.headers = {"Accept-Encoding": "gzip;q=0"}
    handler.respond(body, "application/json")
    assert handler.wfile.getvalue() == body
    handler.command = "HEAD"
    handler.wfile = io.BytesIO()
    handler.respond(body, "application/json")
    assert handler.wfile.getvalue() == b""
