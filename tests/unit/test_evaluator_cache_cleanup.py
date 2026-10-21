import ast
import multiprocessing as mp
import threading
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import cleanup_model_cache, cleanup_shard_cache
from teutonic.evaluator.cache_lock import cache_lock, document_index_cache_dir


def test_cleanup_removes_expired_ranges_and_orphaned_downloads(tmp_path, monkeypatch):
    import os

    cache = tmp_path / "pretokenized" / "ranges" / "identity"
    cache.mkdir(parents=True)
    expired = [cache / name for name in ("0-1024.range", "orphan.range.tmp", "old.npy.partial")]
    for path in expired:
        path.write_bytes(b"cached")
        os.utime(path, (1, 1))
    recent = cache / "1024-8192.range"
    recent.write_bytes(b"recent")
    monkeypatch.setattr(cleanup_shard_cache, "detect_active_files", lambda _: set())
    result = cleanup_shard_cache.run_cleanup(tmp_path, 60, 120, False)
    assert result["deleted"] == 3
    assert not any(path.exists() for path in expired)
    assert recent.exists()


def hold_cache(cache_dir, acquired, release, cleanup=False):
    with cache_lock(cache_dir, cleanup=cleanup) as locked:
        assert locked
        acquired.set()
        if not release.wait(10):
            raise TimeoutError("test did not release cache lock")


@pytest.mark.parametrize("dry_run", [False, True])
def test_cleanup_document_index_pages_and_manifests(tmp_path, monkeypatch, dry_run):
    import os

    index = tmp_path / "doc_index"
    pages = index / ("a" * 64)
    manifests = index / "manifests"
    pages.mkdir(parents=True)
    manifests.mkdir()
    expired = [
        pages / "0", pages / "range-1024-4096", pages / "tmpdownload.tmp",
        manifests / ("b" * 64 + ".json"), manifests / "tmpinterrupted",
    ]
    for path in expired:
        path.write_bytes(b"cached")
        os.utime(path, (1, 1))
    preserved = [pages / "1", pages / ".teutonic-cache.lock", index / "unrelated.json"]
    for path in preserved:
        path.write_bytes(b"keep")
    monkeypatch.setattr(cleanup_shard_cache, "detect_active_files", lambda _: set())
    result = cleanup_shard_cache.run_cleanup(tmp_path, 60, 120, dry_run)
    assert result["deleted"] == len(expired)
    assert all(path.exists() == dry_run for path in expired)
    assert all(path.exists() for path in preserved)


def test_cleanup_cli_includes_custom_document_index_cache(tmp_path, monkeypatch):
    import os
    import sys

    shards, indexes = tmp_path / "shards", tmp_path / "external-indexes"
    page = indexes / ("c" * 64) / "0"
    page.parent.mkdir(parents=True)
    page.write_bytes(b"cached")
    os.utime(page, (1, 1))
    monkeypatch.setenv("TEUTONIC_DOC_INDEX_CACHE_DIR", str(indexes))
    monkeypatch.setattr(sys, "argv", ["cleanup_shard_cache.py", "--cache-dir", str(shards)])
    monkeypatch.setattr(cleanup_shard_cache, "detect_active_files", lambda _: set())
    with cache_lock(indexes):
        cleanup_shard_cache.main()
        assert page.exists()
    cleanup_shard_cache.main()
    assert not page.exists()


@pytest.mark.parametrize("kind", ["model", "shard"])
def test_cleanup_preserves_unopened_files_during_evaluation(tmp_path, monkeypatch, kind):
    if kind == "model":
        snapshot = tmp_path / "immutable-r2" / "old-competition-king"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        victim = snapshot / "model.safetensors"
        victim.write_bytes(b"weights")
        # The single king marker refers to a different competition's snapshot.
        (tmp_path / ".current_king").write_text(str(tmp_path / "another-king"))
        monkeypatch.setattr(cleanup_model_cache, "detect_active_snapshot_dirs", lambda _: set())

        def cleanup():
            return cleanup_model_cache.run_cleanup(tmp_path, 0, 0, 0, 0, False)
    else:
        victim = tmp_path / "pretokenized" / "selected.npy"
        victim.parent.mkdir()
        victim.write_bytes(b"tokens")
        monkeypatch.setattr(cleanup_shard_cache, "detect_active_files", lambda _: set())

        def cleanup():
            return cleanup_shard_cache.run_cleanup(tmp_path, 0, 0, False)

    context = mp.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    process = context.Process(target=hold_cache, args=(tmp_path, acquired, release))
    process.start()
    try:
        assert acquired.wait(10)
        assert cleanup()["deleted"] == 0
        assert victim.exists()
        release.set()
        process.join(timeout=10)
        assert process.exitcode == 0
        assert cleanup()["deleted"] == 1
        assert not victim.exists()
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        process.close()


def test_evaluation_waits_for_cleanup_and_process_exit_releases_lock(tmp_path):
    context = mp.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    process = context.Process(target=hold_cache, args=(tmp_path, acquired, release))
    try:
        with cache_lock(tmp_path, cleanup=True) as locked:
            assert locked
            process.start()
            assert not acquired.wait(0.25)
        assert acquired.wait(10)
        process.terminate()
        process.join(timeout=10)
        with cache_lock(tmp_path, cleanup=True) as locked:
            assert locked
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        process.close()


def engine_function(name, namespace):
    # Exercise the production lifecycle without importing the GPU stack.
    path = Path(__file__).parents[2] / "teutonic/evaluator/engine.py"
    function = next(
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("external_index", [False, True])
def test_evaluator_locks_caches_before_resolution_and_releases_on_failure(tmp_path, monkeypatch, external_index):
    model_cache, shard_cache = tmp_path / "models", tmp_path / "shards"
    if external_index:
        monkeypatch.setenv("TEUTONIC_DOC_INDEX_CACHE_DIR", str(tmp_path / "indexes"))
    else:
        monkeypatch.delenv("TEUTONIC_DOC_INDEX_CACHE_DIR", raising=False)
    index_cache = document_index_cache_dir(shard_cache)
    record = SimpleNamespace(
        events=Queue(), progress=None, event=lambda kind, data: {"type": kind, "data": data},
    )

    def resolve(_artifact):
        for root in (model_cache, shard_cache, index_cache):
            with cache_lock(root, cleanup=True) as locked:
                assert not locked
        raise ValueError("fixture resolution failure")

    def cleanup():
        for root in (model_cache, shard_cache, index_cache):
            with cache_lock(root, cleanup=True) as locked:
                assert locked

    import subprocess
    from teutonic.evaluator.document_index import DocumentIndexDownloadError, RangeDownloadError

    namespace = {
        "EvaluationRequestV2": object, "ExitStack": ExitStack, "cache_lock": cache_lock,
        "document_index_cache_dir": document_index_cache_dir,
        "_attempts": SimpleNamespace(get=lambda _: record), "_eval_lock": Mock(),
        "MODEL_CACHE_DIR": model_cache, "SHARD_CACHE_DIR": shard_cache,
        "time": time, "datetime": datetime, "timezone": timezone,
        "threading": SimpleNamespace(Event=threading.Event, Thread=Mock()),
        "R2ArtifactResolver": lambda _: SimpleNamespace(resolve=resolve), "log": Mock(),
        "SafetensorsReuseLimitError": type("SafetensorsReuseLimitError", (RuntimeError,), {}),
        "DocumentIndexDownloadError": DocumentIndexDownloadError, "subprocess": subprocess,
        "RangeDownloadError": RangeDownloadError,
        "cleanup_model_cache": Mock(side_effect=cleanup),
    }
    engine_function("run_eval", namespace)(
        "fixture", SimpleNamespace(limits={"n": 30000}, king=object()),
    )
    assert record.state == "failed"
    assert record.error == "fixture resolution failure"
    namespace["cleanup_model_cache"].assert_called_once()
    namespace["_eval_lock"].release.assert_called_once()


def test_builtin_cleanup_obeys_other_evaluator_cache_lock(tmp_path):
    snapshot = tmp_path / "immutable-r2" / "old-king"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    namespace = {
        "ExitStack": ExitStack, "cache_lock": cache_lock, "MODEL_CACHE_DIR": tmp_path,
        "CACHE_HIGH_WATERMARK_GB": 1e-9, "_king_key": None, "_model_worker_pool": None,
        "log": Mock(),
    }
    import shutil
    namespace["shutil"] = shutil
    cleanup = engine_function("cleanup_model_cache", namespace)
    with cache_lock(tmp_path):
        cleanup()
        assert snapshot.exists()
    cleanup()
    assert not snapshot.exists()
    namespace["log"].warning.assert_not_called()
