"""Exercise engine admission/lifecycle wiring without a local Torch installation."""

import ast
import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def engine_function(name, namespace):
    path = Path(__file__).parents[2] / "teutonic/evaluator/engine.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- trusted local AST
    return namespace[name]


@pytest.mark.parametrize("duplicate,fail", [(False, False), (True, False), (False, True)])
def test_admission_uses_document_runner_and_releases_failed_launch(duplicate, fail):
    record = Mock()
    request = SimpleNamespace(eval_id="test:1")
    thread = Mock()
    if fail:
        thread.start.side_effect = RuntimeError("thread failed")
    runner = Mock()
    namespace = {
        "EvaluationRequestV3": SimpleNamespace(from_mapping=lambda _: request),
        "validate_protocol_versions": Mock(), "ProtocolValidationError": ValueError,
        "AttemptBusyError": type("Busy", (Exception,), {}),
        "AttemptConflictError": type("Conflict", (Exception,), {}),
        "_attempts": SimpleNamespace(start=Mock(return_value=(record, duplicate))),
        "_eval_lock": Mock(), "time": time, "run_eval": runner,
        "threading": SimpleNamespace(Thread=Mock(return_value=thread)),
    }
    fn = engine_function("start_eval", namespace)
    if fail:
        with pytest.raises(RuntimeError):
            asyncio.run(fn({}))
        namespace["_eval_lock"].release.assert_called_once()
    else:
        asyncio.run(fn({}))
    if duplicate:
        namespace["threading"].Thread.assert_not_called()
    else:
        assert namespace["threading"].Thread.call_args.kwargs["target"] is runner


def test_production_app_has_no_window_or_index_prefetch_hook():
    root = Path(__file__).parents[2]
    engine = (root / "teutonic/evaluator/engine.py").read_text()
    app = (root / "teutonic/evaluator/app.py").read_text()
    assert "IndexPrefetcher" not in engine
    assert "run_eval_with_prefetch" not in engine
    assert "HorizonDownload.from_env" in engine
    assert "load_indexed_sequences_from_npy_shard" not in engine
    assert "from teutonic.evaluator import sources" not in app


def test_startup_preloads_and_shutdown_stops_idle_horizon_first(tmp_path, monkeypatch):
    from teutonic.evaluator import horizon_pipeline_runtime, idle_horizon, vllm_runtime

    calls = Mock()
    downloader, resident = calls.downloader, calls.resident
    monkeypatch.setattr(horizon_pipeline_runtime, "runtime_identity", lambda: {"reference": 1})
    monkeypatch.setattr(vllm_runtime, "runtime_identity", dict)
    monkeypatch.setattr(vllm_runtime, "close_all_workers", calls.close_all)
    factory = Mock(return_value=resident)
    monkeypatch.setattr(idle_horizon, "IdleHorizon", factory)
    namespace = {
        "FastAPI": object, "setup_logging": Mock(), "log": Mock(),
        "parse_gpu_ids": lambda: list(range(8)), "_model_worker_pool": None,
        "chain_config": SimpleNamespace(ARCH_MODULE="fixture"),
        "HorizonDownload": SimpleNamespace(from_env=Mock(return_value=downloader)),
        "COMPLETED_SAFETENSORS_SHA_FILE": tmp_path / "completed",
        **{key: tmp_path for key in ("MODEL_CACHE_DIR", "EVAL_RECORD_DIR", "SHARD_CACHE_DIR", "KERNEL_CACHE_DIR")},
    }
    lifespan = engine_function("lifespan", namespace)

    async def run():
        generator = lifespan(None)
        await anext(generator)
        resident.preload.assert_called_once()
        factory.assert_called_once_with(downloader, list(range(8)), {"reference": 1})
        await generator.aclose()

    asyncio.run(run())
    names = [call[0] for call in calls.mock_calls]
    assert names.index("downloader.stop") < names.index("resident.close") < names.index("close_all")
    assert namespace["_idle_horizon"] is None


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_failed_evaluation_rewarms_and_releases_admission(cleanup_failure):
    import subprocess
    import threading
    from contextlib import ExitStack, nullcontext
    from datetime import UTC, datetime

    record, resident, lock = Mock(), Mock(), Mock()
    record.progress = {}
    cleanup = Mock(side_effect=RuntimeError("cleanup failed") if cleanup_failure else None)
    namespace = {
        "EvaluationRequestV3": SimpleNamespace, "_attempts": SimpleNamespace(get=lambda _: record),
        "_eval_lock": lock, "_idle_horizon": resident, "time": time, "datetime": datetime, "UTC": UTC,
        "threading": SimpleNamespace(Event=threading.Event, Thread=Mock()), "ExitStack": ExitStack,
        "cache_lock": lambda _: nullcontext(), "MODEL_CACHE_DIR": Path("/tmp"), "SHARD_CACHE_DIR": Path("/tmp"),
        "R2ArtifactResolver": Mock(side_effect=RuntimeError("artifact unavailable")), "log": Mock(),
        "cleanup_model_cache": cleanup, "subprocess": subprocess,
        **{name: type(name, (Exception,), {}) for name in
           ("SafetensorsReuseLimitError", "DocumentIndexDownloadError", "RangeDownloadError")},
    }
    run = engine_function("run_eval", namespace)
    if cleanup_failure:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            run("fixture", SimpleNamespace())
    else:
        run("fixture", SimpleNamespace())
    assert record.state == "failed"
    resident.preload.assert_called_once()
    lock.release.assert_called_once()


def test_whole_document_batch_budget_never_crops_oversized_documents():
    batch_end = engine_function("scoring_batch_end", {})
    sequences = [[0] * n for n in (2, 6, 20, 3, 3)]
    start, batches = 0, []
    while start < len(sequences):
        end = batch_end(sequences, start, batch_size=2, token_budget=8)
        assert end > start
        batches.append(sequences[start:end])
        start = end
    assert [[len(row) for row in batch] for batch in batches] == [[2], [6], [20], [3, 3]]
    assert [row for batch in batches for row in batch] == sequences


def test_bootstrap_cap_rejects_instead_of_changing_signed_policy():
    apply = engine_function("apply_eval_limits", {"EvalRequest": SimpleNamespace,
                                                   "EVAL_BOOTSTRAP_B_CAP": 1000})
    req = SimpleNamespace(n_bootstrap=1001)
    with pytest.raises(ValueError, match="n_bootstrap"):
        apply(req)
    assert req.n_bootstrap == 1001
    req.n_bootstrap = 1000
    assert apply(req)["capped"] is False


def test_streamed_chunks_reuse_preflight_until_length_or_generation_increases():
    import math
    from queue import Empty, Queue

    pool_class = engine_function("PersistentModelWorkerPool", {"EvalRequest": SimpleNamespace,
                                                               "math": math, "Empty": Empty, "Path": Path})
    pool = object.__new__(pool_class)
    pool.specs = [{"worker_id": "king-0"}]
    pool.command_queues = {"king-0": Mock()}
    pool.result_queue = Queue()
    events = []

    def preflight(length, generation):
        pool.result_queue.put({"worker_id": "king-0", "role": "king", "type": "result",
                               "generation": generation, "sequence_indices": [0], "losses": [1.]})
        return pool.preflight_longest_document([[3] * length], generation, events.append)

    assert not preflight(8, "first")["reused"]
    reused = pool.preflight_longest_document([[3] * 4], "first", events.append)
    assert reused["reused"] and reused["input_tokens"] == 8
    assert pool.command_queues["king-0"].put.call_count == 1
    assert not preflight(16, "first")["reused"]
    assert not preflight(4, "new-model-generation")["reused"]
    assert pool.command_queues["king-0"].put.call_count == 3
