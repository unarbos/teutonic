"""Small loopback HTTP test; no cloud credentials, model weights or GPUs needed."""

import hashlib
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from teutonic.evaluator.token_shards import NpyShard


def test_real_http_partial_download_without_disk_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEUTONIC_PRETOKENIZED_CACHE_DIR", str(tmp_path / "legacy-cache"))
    array = np.arange(1024 * 2048, dtype="<u4").reshape(1024, 2048)
    stream = io.BytesIO()
    np.save(stream, array)
    body = stream.getvalue()
    digest = hashlib.sha256(body).hexdigest()
    received = []
    connections = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            connections.add(self.client_address)
            assert self.headers.get("If-Match") is None
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            received.append((start, end))
            self.send_response(200 if self.path == "/ignore-range" else 206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("x-amz-meta-sha256", digest)
            self.end_headers()
            if self.path != "/ignore-range":
                self.wfile.write(body[start : end + 1])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        ref = SimpleNamespace(
            url=f"http://127.0.0.1:{server.server_port}/shard.npy",
            sha256=digest,
            size_bytes=len(body),
            n_tokens=array.size,
        )
        reader = NpyShard(ref)
        indices = [900, 2, 500, 50, 2]
        for i, row in zip(indices, reader.read_sequences(indices, 2048), strict=True):
            np.testing.assert_array_equal(row, array[i])
        assert reader.reader.metadata_sha256_checked
        assert reader.reader.downloaded_bytes == 33792
        calls = len(received)
        repeated = NpyShard(ref)
        for i, row in zip(indices, repeated.read_sequences(indices, 2048), strict=True):
            np.testing.assert_array_equal(row, array[i])
        assert len(received) == calls * 2
        assert repeated.reader.downloaded_bytes == 33792
        assert not list(tmp_path.iterdir())
        # A different shard on the same host reuses the shared HTTPS/HTTP pool.
        ref.url = f"http://127.0.0.1:{server.server_port}/other.npy"
        other = NpyShard(ref)
        for i, row in zip(indices, other.read_sequences(indices, 2048), strict=True):
            np.testing.assert_array_equal(row, array[i])
        assert len(connections) <= 4
        assert len(received) == 15
        ref.url = f"http://127.0.0.1:{server.server_port}/ignore-range"
        with pytest.raises(RuntimeError, match="exact byte range"):
            NpyShard(ref)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
