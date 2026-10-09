"""Scheduler checks without model construction or GPU allocations."""

from queue import Queue

import pytest
import torch

from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID as EOS
from teutonic.evaluator import engine
from teutonic.evaluator import sequence_parallel as parallel


class ReplyQueue:
    def __init__(self, output, spec):
        self.output, self.spec = output, spec
        self.commands = []
        self.corrupt = False

    def put(self, command):
        self.commands.append(command)
        common = {"generation": command["generation"], "worker_id": self.spec["worker_id"], "role": self.spec["role"]}
        if command["type"] == "sequence_parallel_init":
            self.output.put({**common, "type": "sequence_parallel_ready", "rank": command["rank"]})
            return
        values = [i + (0.25 if self.spec["role"] == "challenger" else 0) for i in command["sequence_indices"]]
        if self.corrupt and command["type"] == "score_parallel":
            values = [float("nan")]
        self.output.put({**common, "type": "result", "sequence_indices": command["sequence_indices"],
                         "losses": values, "scoring_mode": "sequence_parallel" if command["type"] == "score_parallel" else "single_gpu"})


@pytest.fixture
def pool():
    instance = object.__new__(engine.PersistentModelWorkerPool)
    instance.specs = engine.model_worker_specs(list(range(8)))
    instance.ready = {s["worker_id"]: {"pipeline_depth": 1} for s in instance.specs}
    instance.result_queue = Queue()
    instance.command_queues = {s["worker_id"]: ReplyQueue(instance.result_queue, s) for s in instance.specs}
    yield instance
    if getattr(instance, "parallel_directory", None) is not None:
        instance.parallel_directory.cleanup()


def document(n):
    return [1] * (n - 1) + [EOS]


def request():
    return engine.EvalRequest(king_repo="king", challenger_repo="challenger", batch_size=40,
                              long_documents={"version": "fixture"}, early_stop_enabled=True, n_bootstrap=20)


def test_exact_tested_cutoff_routes_only_larger_documents_and_preserves_order(pool):
    limit = parallel.SINGLE_GPU_MAX_TOKENS
    assert limit == 128974
    rows = [document(limit + 1), document(2048), document(limit), document(limit + 2), document(4097)]
    king, challenger, meta = pool.score(rows, "one", request(), lambda _: None)
    assert king == list(range(len(rows)))
    assert challenger == [i + .25 for i in range(len(rows))]
    assert meta["scored_tokens"] == [len(row) - 1 for row in rows]
    assert meta["sequence_parallel"]["sequence_indices"] == [0, 3]
    assert meta["long_document_preflight"]["input_tokens"] == limit + 2
    assert len(meta["long_document_preflight"]["workers"]) == 8
    assert meta["single_gpu_preflight"]["input_tokens"] == limit
    assert meta["early_stop"] is None
    for queue in pool.command_queues.values():
        for command in queue.commands:
            if command["type"] == "score":
                assert all(len(row) <= limit for row in command["token_batches"])
            elif command["type"] == "score_parallel":
                assert len(command["token_batches"]) == 1 and len(command["token_batches"][0]) > limit
    assert pool.result_queue.empty()


def test_all_large_documents_finish_without_waiting_for_regular_jobs(pool):
    rows = [document(parallel.SINGLE_GPU_MAX_TOKENS + i) for i in (1, 2, 3)]
    a, b, meta = pool.score(rows, "large", request(), lambda _: None)
    assert a == [0, 1, 2] and b == [.25, 1.25, 2.25]
    assert meta["single_gpu_preflight"] is None
    assert not any(c["type"] == "score" for q in pool.command_queues.values() for c in q.commands)


def test_groups_reused_and_later_small_only_evaluation_never_uses_parallel(pool):
    rows = [document(parallel.SINGLE_GPU_MAX_TOKENS + 1), document(2048)]
    pool.score(rows, "first", request(), lambda _: None)
    pool.score(rows, "second", request(), lambda _: None)
    pool.score([document(2048)], "third", request(), lambda _: None)
    for queue in pool.command_queues.values():
        assert sum(c["type"] == "sequence_parallel_init" for c in queue.commands) == 1
        assert all(c["type"] == "score" for c in queue.commands if c["generation"] == "third")


def test_invalid_document_or_rank_failure_is_not_skipped(pool):
    with pytest.raises(ValueError, match="complete document"):
        pool.score([[1] * (parallel.SINGLE_GPU_MAX_TOKENS + 1)], "bad-input", request(), lambda _: None)
    assert all(not q.commands for q in pool.command_queues.values())
    pool.command_queues["king-1"].corrupt = True
    with pytest.raises(RuntimeError, match="non-finite"):
        pool.score([document(parallel.SINGLE_GPU_MAX_TOKENS + 1)], "bad-rank", request(), lambda _: None)


def test_parallel_failure_closes_entire_pool_and_discards_it(monkeypatch):
    class FailedPool:
        closed = False

        def load_models(self, *args):
            return "generation"

        def score(self, *args):
            raise RuntimeError("peer failed during collective")

        def close(self, *, force=False):
            assert force
            self.closed = True

    failed = FailedPool()
    monkeypatch.setattr(engine, "_model_worker_pool", failed)
    monkeypatch.setattr(engine, "get_model_worker_pool", lambda _: failed)
    with pytest.raises(RuntimeError, match="peer failed"):
        engine.score_with_model_workers([], request(), "king", "challenger", list(range(8)), lambda _: None)
    assert failed.closed and engine._model_worker_pool is None


def test_forced_close_kills_peer_that_does_not_terminate():
    class StuckPeer:
        def __init__(self):
            self.calls = []

        def is_alive(self):
            return "kill" not in self.calls

        def terminate(self):
            self.calls.append("terminate")

        def join(self, timeout):
            self.calls.append("join")

        def kill(self):
            self.calls.append("kill")

    peer = StuckPeer()
    instance = object.__new__(engine.PersistentModelWorkerPool)
    instance.processes = {"peer": peer}
    instance.command_queues = {}
    instance.result_queue = Queue()
    instance.close(force=True)
    assert peer.calls == ["terminate", "join", "kill", "join"]


def test_prefix_attention_keeps_absolute_causal_offset_and_sink(monkeypatch):
    calls = []

    def gather(received, states):
        received[0].fill_(1)
        received[1].fill_(2)

    def backend(module, query, key, value, mask, **kwargs):
        calls.append((key, value, kwargs))
        return query, None

    monkeypatch.setattr(parallel.dist, "all_gather", gather)
    adapter = parallel.PrefixAttention(backend, 1, [5, 4])
    module = type("Attention", (), {"layer_idx": 7})()
    query = torch.ones(1, 2, 4, 8)
    adapter(module, query, query, query, None, s_aux="sink", sliding_window=128, is_causal=True)
    key, _, kwargs = calls[0]
    assert key.shape[2] == 9
    assert key[0, 0, :, 0].tolist() == [1] * 5 + [2] * 4
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 4]
    assert kwargs["cu_seq_lens_k"].tolist() == [0, 9]
    assert kwargs["max_length_k"] - kwargs["max_length_q"] == 5
    assert kwargs["s_aux"] == "sink" and kwargs["sliding_window"] == 128
    assert kwargs["is_causal"] is True and adapter.layers == {7}
