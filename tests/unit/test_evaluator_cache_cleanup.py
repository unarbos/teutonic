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
from teutonic.evaluator.cache_lock import cache_lock


def hold_cache(cache_dir, acquired, release, cleanup=False):
    with cache_lock(cache_dir, cleanup=cleanup) as locked:
        assert locked
        acquired.set()
        if not release.wait(10):
            raise TimeoutError("test did not release cache lock")


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


def test_evaluator_locks_caches_before_resolution_and_releases_on_failure(tmp_path):
    model_cache, shard_cache = tmp_path / "models", tmp_path / "shards"
    record = SimpleNamespace(
        events=Queue(), progress=None, event=lambda kind, data: {"type": kind, "data": data},
    )

    def resolve(_artifact):
        for root in (model_cache, shard_cache):
            with cache_lock(root, cleanup=True) as locked:
                assert not locked
        raise ValueError("fixture resolution failure")

    def cleanup():
        for root in (model_cache, shard_cache):
            with cache_lock(root, cleanup=True) as locked:
                assert locked

    import subprocess
    from teutonic.evaluator.document_index import DocumentIndexDownloadError

    namespace = {
        "EvaluationRequestV2": object, "ExitStack": ExitStack, "cache_lock": cache_lock,
        "_attempts": SimpleNamespace(get=lambda _: record), "_eval_lock": Mock(),
        "MODEL_CACHE_DIR": model_cache, "SHARD_CACHE_DIR": shard_cache,
        "time": time, "datetime": datetime, "timezone": timezone,
        "threading": SimpleNamespace(Event=threading.Event, Thread=Mock()),
        "R2ArtifactResolver": lambda _: SimpleNamespace(resolve=resolve), "log": Mock(),
        "SafetensorsReuseLimitError": type("SafetensorsReuseLimitError", (RuntimeError,), {}),
        "DocumentIndexDownloadError": DocumentIndexDownloadError, "subprocess": subprocess,
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
