import ast
import asyncio
import fcntl
import io
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import httpx
import pytest

from teutonic.evaluation.index_manifest import (
    ETAG_INDEX_FORMAT,
    TOKEN_FORMAT,
    index_paths,
    pin_index_manifest,
)
from teutonic.evaluator import local_index, range_http
from teutonic.evaluator.document_index import PinnedObject
from teutonic.evaluator.local_index import IndexPrefetcher, LocalIndexStore

BODY = b"0123456789abcdef"
DESCRIPTOR = {"url": "https://index.example/d/a.tokens.npy", "etag": '"abc123"', "size_bytes": len(BODY)}


class Server:
    def __init__(self):
        self.requests = []
        self.closed = 0
        self.after_chunk = lambda: None
        self.headers = {}
        self.body = BODY
        self.status = 206

    @contextmanager
    def open(self, request, timeout):
        self.requests.append(request)
        assert request.get_header("If-match") == DESCRIPTOR["etag"]
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        body = self.body[start:end + 1]

        def chunks(size):
            for offset in range(0, len(body), min(size, 2)):
                self.after_chunk()
                yield body[offset:offset + min(size, 2)]

        try:
            yield SimpleNamespace(
                status=self.status, headers={
                    "ETag": DESCRIPTOR["etag"], "Content-Range": f"bytes {start}-{end}/{len(BODY)}",
                    **self.headers,
                }, iter_chunks=chunks, read=io.BytesIO(body).read,
            )
        finally:
            self.closed += 1


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("TEUTONIC_DOC_INDEX_STORE_DIR", str(root))
    monkeypatch.setattr(local_index, "CHUNK_BYTES", 8)
    return LocalIndexStore(root)


def test_completed_index_reused_after_restart_without_network_or_full_read(store, monkeypatch):
    server = Server()
    assert store.download(DESCRIPTOR, lambda: False, server.open)
    assert len(server.requests) == 2
    forbidden = Mock(side_effect=AssertionError("unexpected network or whole-file read"))
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    reopened = LocalIndexStore(store.root)
    assert reopened.download(DESCRIPTOR, lambda: False, forbidden)
    reader = PinnedObject(DESCRIPTOR, None, forbidden)
    assert reader.read(3, 5) == BODY[3:8]
    assert reader.downloaded_bytes == 0
    forbidden.assert_not_called()


@pytest.mark.parametrize("change", ["missing", "same-size-unverified", "pin", "size", "modified", "receipt"])
def test_untrusted_or_changed_local_file_falls_back_to_ranges(store, change):
    server = Server()
    store.download(DESCRIPTOR, lambda: False, server.open)
    directory = store.directory(DESCRIPTOR)
    descriptor = dict(DESCRIPTOR)
    if change == "missing":
        (directory / "index.bin").unlink()
    elif change == "same-size-unverified":
        (directory / "complete.json").unlink()
    elif change == "pin":
        descriptor["sha256"] = "a" * 64
    elif change == "size":
        (directory / "index.bin").write_bytes(b"x")
    elif change == "modified":
        (directory / "index.bin").write_bytes(b"x" * len(BODY))
    else:
        (directory / "complete.json").write_text("[]")
    reader = PinnedObject(descriptor, None, server.open)
    assert reader.local is None
    assert reader.read(0, len(BODY)) == BODY
    assert reader.downloaded_bytes == len(BODY)


def test_mutation_after_reader_open_falls_back(store):
    server = Server()
    store.download(DESCRIPTOR, lambda: False, server.open)
    reader = PinnedObject(DESCRIPTOR, None, server.open)
    assert reader.local is not None
    (store.directory(DESCRIPTOR) / "index.bin").write_bytes(b"x" * len(BODY))
    assert reader.read(2, 4) == BODY[2:6]
    assert reader.local is None


def test_pause_closes_stream_and_restart_resumes_at_checkpoint(store):
    server = Server()
    pause = threading.Event()
    server.after_chunk = pause.set
    assert not store.download(DESCRIPTOR, pause.is_set, server.open)
    assert server.closed == 1
    directory = store.directory(DESCRIPTOR)
    assert json.loads((directory / "partial.json").read_text())["offset"] == 2
    assert store.open(DESCRIPTOR) is None
    assert not store.download(DESCRIPTOR, pause.is_set, server.open)
    assert len(server.requests) == 1
    # Simulate a crash after an append but before its checkpoint was saved.
    with (directory / "download.part").open("ab") as stream:
        stream.write(b"bad tail")
    pause.clear()
    server.after_chunk = lambda: None
    assert LocalIndexStore(store.root).download(DESCRIPTOR, pause.is_set, server.open)
    assert server.requests[1].get_header("Range") == "bytes=2-9"
    reader = store.open(DESCRIPTOR)
    assert reader.read(0, len(BODY)) == BODY
    reader.close()


@pytest.mark.parametrize("bad", ["etag", "range", "encoding", "status", "short", "extra"])
def test_bad_transfer_never_becomes_ready(store, bad):
    server = Server()
    if bad == "etag":
        server.headers["ETag"] = '"bad"'
    elif bad == "range":
        server.headers["Content-Range"] = "bytes 1-8/16"
    elif bad == "encoding":
        server.headers["Content-Encoding"] = "gzip"
    elif bad == "status":
        server.status = 200
    elif bad == "short":
        server.body = b"short"
    else:
        @contextmanager
        def extra(request, timeout):
            with Server().open(request, timeout) as response:
                response.iter_chunks = lambda _: iter([BODY])
                yield response
        server.open = extra
    with pytest.raises((ValueError, OSError)):
        store.download(DESCRIPTOR, lambda: False, server.open)
    assert store.open(DESCRIPTOR) is None
    good = Server()
    assert store.download(DESCRIPTOR, lambda: False, good.open)
    assert good.requests[0].get_header("Range") == "bytes=0-7"


def test_failed_later_chunk_preserves_previously_committed_bytes(store):
    server = Server()
    original = server.open

    @contextmanager
    def fail_second_chunk(request, timeout):
        with original(request, timeout) as response:
            if request.get_header("Range") == "bytes=8-15":
                def broken(_size):
                    yield BODY[8:10]
                    raise OSError("connection lost")
                response.iter_chunks = broken
            yield response

    with pytest.raises(OSError, match="connection lost"):
        store.download(DESCRIPTOR, lambda: False, fail_second_chunk)
    resumed = Server()
    assert store.download(DESCRIPTOR, lambda: False, resumed.open)
    assert resumed.requests[0].get_header("Range") == "bytes=8-15"
    reader = store.open(DESCRIPTOR)
    assert reader.read(0, len(BODY)) == BODY
    reader.close()


def test_concurrent_process_lock_skips_transfer(store):
    directory = store.directory(DESCRIPTOR)
    directory.mkdir(parents=True)
    server = Server()
    with (directory / "download.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not store.download(DESCRIPTOR, lambda: False, server.open)
        assert not server.requests
    assert store.download(DESCRIPTOR, lambda: False, server.open)


def pin():
    return pin_index_manifest(json.dumps({
        "format": ETAG_INDEX_FORMAT, "token_format": TOKEN_FORMAT,
        "datasets": {"d": {"categories": ["a"], "source_manifest": {
            "url": "https://index.example/source.json", "sha256": "a" * 64,
        }}},
        "files": {path: {"etag": DESCRIPTOR["etag"], "size_bytes": len(BODY)}
                  for path in index_paths("d", "a").values()},
    }).encode(), "https://index.example/manifest.json")


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "background worker did not reach expected state"
        time.sleep(0.01)


def test_background_start_pause_resume_and_manifest_dedup(store, monkeypatch):
    inventory = pin()
    server = Server()
    fetch_started, allow_fetch = threading.Event(), threading.Event()

    def fetch(*args, **kwargs):
        fetch_started.set()
        assert allow_fetch.wait(5)
        return inventory

    monkeypatch.setattr(local_index, "fetch_index_manifest", fetch)
    manager = IndexPrefetcher(store.root, inventory["url"], inventory["sha256"], 2, opener=server.open)
    manager.start()
    try:
        assert fetch_started.wait(2)  # startup returned while network fetch is blocked
        manager.pause()
        allow_fetch.set()
        wait_for(lambda: manager.status()["manifest_ready"])
        manager.enqueue_pin(inventory)
        assert manager.status()["files_total"] == 4
        assert not server.requests
        manager.resume()
        wait_for(lambda: manager.status()["files_ready"] == 4)
        assert manager.status()["bytes_ready"] == 4 * len(BODY)
        assert manager.status()["errors"] == 0
    finally:
        allow_fetch.set()
        manager.stop()
    assert all(not t.is_alive() for t in manager.threads)


def test_active_background_transfer_yields_to_evaluation(store):
    server = Server()
    manager = IndexPrefetcher(store.root, "", "", 1, opener=server.open)
    server.after_chunk = manager.pause
    manager.enqueue_pin(pin())
    worker = threading.Thread(target=manager._worker)
    manager.threads.append(worker)
    worker.start()
    try:
        wait_for(lambda: manager.status()["paused"] and manager.status()["active_downloads"] == 0)
        assert len(server.requests) == 1
        assert server.closed == 1
        assert manager.status()["files_ready"] == 0
        server.after_chunk = lambda: None
        manager.resume()
        wait_for(lambda: manager.status()["files_ready"] == 4)
        assert any(r.get_header("Range") == "bytes=2-9" for r in server.requests)
    finally:
        manager.stop()


def test_overlapping_evaluation_handover_stays_paused(store):
    manager = IndexPrefetcher(store.root, "", "")
    manager.pause()
    manager.pause()  # next admission after previous eval releases its lock
    manager.resume()  # previous worker's finally
    assert manager.interrupted()
    manager.resume()
    assert not manager.interrupted()


def test_rate_limit_defers_background_download_without_busy_retry(store):
    opener = Mock(side_effect=HTTPError(DESCRIPTOR["url"], 429, "limited", {"Retry-After": "300"}, None))
    manager = IndexPrefetcher(store.root, "", "", 1, opener=opener)
    manager.enqueue_pin(pin())
    worker = threading.Thread(target=manager._worker)
    manager.threads.append(worker)
    started = time.monotonic()
    worker.start()
    try:
        wait_for(lambda: manager.status()["errors"] == 4 and manager.status()["active_downloads"] == 0)
        with manager.condition:
            assert all(j["retry_at"] >= started + 300 for j in manager.jobs.values())
        assert opener.call_count == 4
        assert manager.status()["files_ready"] == 0
    finally:
        manager.stop()


def test_http_transport_streams_pinned_chunks(store, monkeypatch):
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield BODY[:8]

    def serve(request):
        assert request.headers["If-Match"] == DESCRIPTOR["etag"]
        start, end = map(int, request.headers["Range"][6:].split("-"))
        class RangeStream(Stream):
            def __iter__(self):
                yield BODY[start:end + 1]
        return httpx.Response(206, headers={
            "ETag": DESCRIPTOR["etag"], "Content-Range": f"bytes {start}-{end}/{len(BODY)}",
        }, stream=RangeStream())

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        monkeypatch.setattr(range_http, "_client", client)
        assert store.download(DESCRIPTOR, lambda: False)
    reader = store.open(DESCRIPTOR)
    assert reader.read(0, len(BODY)) == BODY
    reader.close()


def engine_function(name, namespace):
    path = Path(__file__).parents[2] / "teutonic/evaluator/engine.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


@pytest.mark.parametrize("fails", [False, True])
def test_evaluation_resumes_download_after_success_or_failure(fails):
    prefetcher = Mock()
    request = SimpleNamespace(dataset={"long_documents": {"index_manifest": pin()}})
    run = Mock(side_effect=RuntimeError("failed") if fails else None)
    fn = engine_function("run_eval_with_prefetch", {"run_eval": run, "log": Mock()})
    if fails:
        with pytest.raises(RuntimeError):
            fn("eval", request, prefetcher)
    else:
        fn("eval", request, prefetcher)
    prefetcher.enqueue_pin.assert_called_once()
    prefetcher.resume.assert_called_once()


@pytest.mark.parametrize("duplicate, thread_failure", [(False, False), (True, False), (False, True)])
def test_admission_pauses_before_launch_and_recovers_launch_failure(duplicate, thread_failure):
    prefetcher = Mock()
    record, request = Mock(), SimpleNamespace(eval_id="eval")

    def start():
        prefetcher.pause.assert_called_once()
        if thread_failure:
            raise RuntimeError("cannot start thread")

    namespace = {
        "EvaluationRequestV2": SimpleNamespace(from_mapping=lambda _: request),
        "validate_protocol_versions": Mock(), "ProtocolValidationError": ValueError,
        "AttemptBusyError": type("Busy", (Exception,), {}),
        "AttemptConflictError": type("Conflict", (Exception,), {}),
        "_attempts": SimpleNamespace(start=Mock(return_value=(record, duplicate))),
        "_eval_lock": Mock(), "_index_prefetcher": prefetcher, "time": time,
        "threading": SimpleNamespace(Thread=Mock(return_value=SimpleNamespace(start=start))),
        "run_eval_with_prefetch": Mock(),
    }
    fn = engine_function("start_eval", namespace)
    if thread_failure:
        with pytest.raises(RuntimeError):
            asyncio.run(fn({}))
        prefetcher.resume.assert_called_once()
        namespace["_eval_lock"].release.assert_called_once()
        assert record.state == "failed"
    else:
        asyncio.run(fn({}))
        assert prefetcher.pause.call_count == (0 if duplicate else 1)
        prefetcher.resume.assert_not_called()


def test_cleanup_preserves_store_without_env_even_when_nested(store, monkeypatch):
    from scripts.cleanup_shard_cache import run_cleanup

    store.download(DESCRIPTOR, lambda: False, Server().open)
    # Include an ordinary cache filename to verify pruning rather than suffix luck.
    (store.root / "keep.npy").write_bytes(b"keep")
    monkeypatch.delenv("TEUTONIC_DOC_INDEX_STORE_DIR")
    result = run_cleanup(store.root.parent, 0, 0, False)
    assert result["deleted"] == 0
    assert (store.root / "keep.npy").exists()
    reader = store.open(DESCRIPTOR)
    assert reader is not None
    reader.close()


def test_env_requires_pinned_manifest_and_separate_storage(tmp_path, monkeypatch):
    monkeypatch.delenv("TEUTONIC_DOC_INDEX_STORE_DIR", raising=False)
    assert IndexPrefetcher.from_env() is None
    monkeypatch.setenv("TEUTONIC_DOC_INDEX_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("TEUTONIC_DOC_INDEX_PREFETCH_MANIFEST_URL", raising=False)
    with pytest.raises(ValueError, match="manifest URL"):
        IndexPrefetcher.from_env()
    monkeypatch.setenv("TEUTONIC_DOC_INDEX_PREFETCH_MANIFEST_URL", pin()["url"])
    monkeypatch.setenv("TEUTONIC_DOC_INDEX_PREFETCH_MANIFEST_SHA256", pin()["sha256"])
    assert IndexPrefetcher.from_env().workers == 4
    monkeypatch.setenv("TEUTONIC_MODEL_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="outside the model cache"):
        IndexPrefetcher.from_env()
    monkeypatch.setenv("TEUTONIC_MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("TEUTONIC_DOC_INDEX_PREFETCH", "0")
    assert IndexPrefetcher.from_env() is None
