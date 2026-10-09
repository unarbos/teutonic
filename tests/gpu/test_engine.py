from __future__ import annotations

import asyncio
import hashlib
import sys
import threading
import time
import types
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from teutonic.evaluator import engine as eval_server
from teutonic.evaluator import sources
from teutonic.evaluator.engine import (
    MODEL_INSTANCES_PER_SIDE,
    MODEL_WORKER_PROCESSES,
    EvalRequest,
    ModelSequenceScorer,
    PersistentModelWorkerPool,
    checkpoint_load_key,
    kernel_cache_identity,
    load_eval_model,
    model_worker_specs,
    patch_mimo_masking_compat,
    resolved_attention_types,
    snapshot_safetensor_keys,
    validate_and_report_attention_config,
)


class ImmediateScoreQueue:
    def __init__(self, result_queue, worker_id, role, loss):
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.role = role
        self.loss = loss
        self.batch_sizes = []

    def put(self, command):
        assert command["type"] == "score"
        indices = command["sequence_indices"]
        self.batch_sizes.append(len(indices))
        self.result_queue.put({
            "type": "result",
            "generation": command["generation"],
            "worker_id": self.worker_id,
            "role": self.role,
            "sequence_indices": indices,
            "losses": [self.loss] * len(indices),
        })


def test_deterministic_execution_overrides_unsafe_settings_and_is_strict(monkeypatch):
    import os

    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":invalid:")
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        settings = eval_server.configure_deterministic_execution()
        assert settings == {
            "deterministic_algorithms": True,
            "warn_only": False,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        }
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert eval_server.configure_deterministic_execution() == settings
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cudnn.deterministic = previous[3]


def test_mixed_length_batch_budget_and_complete_scoring():
    from teutonic.evaluator.engine import scoring_batch_end

    lengths = [2048] * 40 + [8193, 2048, 4097, 90000, 2049]
    sequences = [[1] * n for n in lengths]
    assert scoring_batch_end(sequences, 0, 40, 2048) == 40
    assert scoring_batch_end(sequences, 40, 40, 2048) == 43
    assert scoring_batch_end(sequences, 43, 40, 2048) == 44
    pool = object.__new__(PersistentModelWorkerPool)
    pool.specs = [{"worker_id": "king-0", "role": "king"}, {"worker_id": "challenger-0", "role": "challenger"}]
    pool.ready = {s["worker_id"]: {"pipeline_depth": 1} for s in pool.specs}
    pool.result_queue = Queue()
    pool.command_queues = {s["worker_id"]: ImmediateScoreQueue(pool.result_queue, s["worker_id"], s["role"], 1 if s["role"] == "king" else 2) for s in pool.specs}
    req = EvalRequest(king_repo="king", challenger_repo="challenger", batch_size=40,
                      n_bootstrap=20, early_stop_enabled=True, early_stop_min_fraction=.1,
                      early_stop_check_interval=1, long_documents={"version": "fixture"})
    king, challenger, meta = pool.score(sequences, "generation-1", req, lambda event: None)
    assert len(king) == len(challenger) == len(sequences)
    assert meta["early_stop"] is None
    assert meta["scored_tokens"] == [n - 1 for n in lengths]
    assert all(q.batch_sizes == [1, 40, 3, 1, 1] for q in pool.command_queues.values())
    assert meta["long_document_preflight"]["input_tokens"] == 90000
    assert {w["worker_id"] for w in meta["long_document_preflight"]["workers"]} == {"king-0", "challenger-0"}
    pool.command_queues["king-0"].loss = float("nan")
    with pytest.raises(RuntimeError, match="non-finite.*preflight"):
        pool.score(sequences, "generation-2", req, lambda event: None)


def test_mixed_sample_audit_and_component_scores_preserve_document_identity():
    provenance = [
        {"shard_group_index": 0, "shard_index": 0, "shard_sequence_index": 7, "component": "windows"},
        {"shard_group_index": 0, "shard_index": -1, "shard_sequence_index": -1,
         "component": "long_documents", "dataset": "math-reasoning", "category": "math-dialogue",
         "length_bucket": "2049-4096", "document_row": 123, "length": 3001,
         "fragments": [{"shard_id": 4, "seq_index": 2, "seq_offset": 0, "frag_len": 2048}]},
    ]
    result = eval_server._build_sample_results([2, 4], [1, 2], provenance, [2000, 3000])
    assert result["format"] == "columnar-masked-documents-v2"
    assert result["document_row"] == [None, 123]
    assert result["component"] == ["windows", "long_documents"]
    scores = eval_server.long_document_scores([2, 4], [1, 2], provenance, [2000, 3000])
    assert scores["by_bucket"]["2049-4096"]["n_scored_tokens"] == 3000
    assert scores["by_category"]["math-reasoning/math-dialogue"]["mu_hat"] == 2


def test_worker_pool_early_stop_drains_dispatched_results():
    pool = object.__new__(PersistentModelWorkerPool)
    pool.specs = [
        {"worker_id": "king-0", "role": "king"},
        {"worker_id": "challenger-0", "role": "challenger"},
    ]
    pool.ready = {
        "king-0": {"pipeline_depth": 1},
        "challenger-0": {"pipeline_depth": 1},
    }
    pool.result_queue = Queue()
    pool.command_queues = {
        "king-0": ImmediateScoreQueue(pool.result_queue, "king-0", "king", 1.0),
        "challenger-0": ImmediateScoreQueue(
            pool.result_queue, "challenger-0", "challenger", 2.0
        ),
    }
    request = EvalRequest(
        king_repo="king",
        challenger_repo="challenger",
        delta_threshold=0.5,
        early_stop_enabled=True,
        early_stop_min_fraction=0.4,
        early_stop_advantage_quantile=0.95,
        early_stop_margin=0.0,
        early_stop_check_interval=2,
    )

    king, challenger, metadata = pool.score(
        [[index, index] for index in range(10)], "generation-1", request, lambda _event: None
    )

    assert len(king) == len(challenger) == 4
    assert metadata["early_stop"]["completed_sequences"] == 4
    assert pool.result_queue.empty()


def test_worker_pool_batches_sequences_and_preserves_the_tail():
    pool = object.__new__(PersistentModelWorkerPool)
    pool.specs = [
        {"worker_id": "king-0", "role": "king"},
        {"worker_id": "challenger-0", "role": "challenger"},
    ]
    pool.ready = {
        "king-0": {"pipeline_depth": 1},
        "challenger-0": {"pipeline_depth": 1},
    }
    pool.result_queue = Queue()
    king_queue = ImmediateScoreQueue(pool.result_queue, "king-0", "king", 1.0)
    challenger_queue = ImmediateScoreQueue(
        pool.result_queue,
        "challenger-0",
        "challenger",
        2.0,
    )
    pool.command_queues = {
        "king-0": king_queue,
        "challenger-0": challenger_queue,
    }
    request = EvalRequest(
        king_repo="king",
        challenger_repo="challenger",
        batch_size=3,
    )

    king, challenger, metadata = pool.score(
        [[index, index] for index in range(8)], "generation-1", request, lambda _event: None
    )

    assert king == [1.0] * 8
    assert challenger == [2.0] * 8
    assert metadata["early_stop"] is None
    assert king_queue.batch_sizes == [3, 3, 2]
    assert challenger_queue.batch_sizes == [3, 3, 2]


def test_batched_early_stop_uses_the_configured_check_boundary():
    pool = object.__new__(PersistentModelWorkerPool)
    pool.specs = [
        {"worker_id": "king-0", "role": "king"},
        {"worker_id": "challenger-0", "role": "challenger"},
    ]
    pool.ready = {
        "king-0": {"pipeline_depth": 1},
        "challenger-0": {"pipeline_depth": 1},
    }
    pool.result_queue = Queue()
    pool.command_queues = {
        "king-0": ImmediateScoreQueue(pool.result_queue, "king-0", "king", 1.0),
        "challenger-0": ImmediateScoreQueue(
            pool.result_queue,
            "challenger-0",
            "challenger",
            2.0,
        ),
    }
    request = EvalRequest(
        king_repo="king",
        challenger_repo="challenger",
        batch_size=3,
        delta_threshold=0.5,
        early_stop_enabled=True,
        early_stop_min_fraction=0.4,
        early_stop_advantage_quantile=0.95,
        early_stop_margin=0.0,
        early_stop_check_interval=2,
    )

    king, challenger, metadata = pool.score(
        [[index, index] for index in range(10)], "generation-1", request, lambda _event: None
    )

    assert len(king) == len(challenger) == 4
    assert metadata["early_stop"]["completed_sequences"] == 4
    assert pool.result_queue.empty()


def mimo_config(**overrides):
    pattern = [0, 1, 1, 0]
    values = {
        "model_type": "mimo_v2",
        "num_hidden_layers": len(pattern),
        "hybrid_layer_pattern": pattern,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "sliding_window": 128,
        "sliding_window_size": 128,
        "add_swa_attention_sink_bias": True,
        "_attn_implementation": "eager",
        "head_dim": 192,
        "v_head_dim": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_eager_mimo_schedule_is_resolved_in_layer_order():
    assert resolved_attention_types(mimo_config()) == [
        "full_attention",
        "sliding_window_attention",
        "sliding_window_attention",
        "full_attention",
    ]


def test_mimo_schedule_is_derived_before_model_sets_layer_types():
    config = mimo_config()
    del config.layer_types
    assert resolved_attention_types(config) == [
        "full_attention",
        "sliding_window_attention",
        "sliding_window_attention",
        "full_attention",
    ]


def test_mimo_masking_compat_removes_only_obsolete_cache_position():
    module = types.ModuleType("test_mimo_masking_compat_module")

    def current_mask(*, config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
        return config, inputs_embeds, attention_mask, past_key_values, position_ids

    module.create_causal_mask = current_mask
    module.create_sliding_window_causal_mask = current_mask
    sys.modules[module.__name__] = module
    try:
        model_type = type("TestMiMo", (), {"__module__": module.__name__})
        model = model_type()
        model.config = SimpleNamespace(model_type="mimo_v2")
        assert set(patch_mimo_masking_compat(model)) == {
            "create_causal_mask",
            "create_sliding_window_causal_mask",
        }
        assert module.create_causal_mask(
            config="config",
            inputs_embeds="embeds",
            attention_mask="mask",
            cache_position="obsolete",
            past_key_values="cache",
            position_ids="positions",
        ) == ("config", "embeds", "mask", "cache", "positions")
        assert patch_mimo_masking_compat(model) == ()
    finally:
        sys.modules.pop(module.__name__, None)


def test_supported_attention_backends_and_malformed_hybrid_config():
    assert resolved_attention_types(mimo_config(_attn_implementation="flash_attention_4"))
    with pytest.raises(RuntimeError, match="does not support attention implementation"):
        resolved_attention_types(mimo_config(_attn_implementation="sdpa"))
    with pytest.raises(RuntimeError, match="entries for 4 surviving layers"):
        resolved_attention_types(mimo_config(hybrid_layer_pattern=[0, 1]))


def test_eval_request_fixes_reference_runtime_settings():
    request = EvalRequest(king_repo="king", challenger_repo="challenger")
    assert request.attn_implementation == "eager"
    assert request.batch_size == eval_server.DEFAULT_BATCH_SIZE
    assert request.parallel_batch_size == 1
    assert request.parallel_models is True
    assert request.seq_len == 2048
    assert request.lm_head_chunk == 1024


def test_eval_request_accepts_only_eager_and_fa4():
    request = EvalRequest(
        king_repo="king",
        challenger_repo="challenger",
        attn_implementation="flash_attention_4",
    )
    assert request.attn_implementation == "flash_attention_4"
    with pytest.raises(ValueError):
        EvalRequest(
            king_repo="king",
            challenger_repo="challenger",
            attn_implementation="sdpa",
        )


def test_fa4_attention_report_preserves_asymmetric_value_head_dim():
    report = validate_and_report_attention_config(
        mimo_config(_attn_implementation="flash_attention_4"),
        "test-model",
    )
    assert report["attn_implementation"] == "flash_attention_4"
    assert report["qk_head_dim"] == 192
    assert report["v_head_dim"] == 128
    assert report["fa4_native_asymmetric_value_dim"] is True


def test_worker_topology_gives_each_side_four_distinct_single_gpu_replicas():
    gpu_ids = [7, 5, 3, 1, 6, 4, 2, 0]
    specs = model_worker_specs(gpu_ids)
    assert MODEL_INSTANCES_PER_SIDE == 4
    assert MODEL_WORKER_PROCESSES == 8
    assert [spec["gpu_ids"] for spec in specs] == [[gpu] for gpu in gpu_ids]
    assert [spec["role"] for spec in specs] == ["king"] * 4 + ["challenger"] * 4
    assert len({spec["worker_id"] for spec in specs}) == 8
    for invalid in (list(range(4)), [0] * 8, [-1] + list(range(7))):
        with pytest.raises(RuntimeError, match="exactly 8 GPUs"):
            model_worker_specs(invalid)


def test_health_reports_single_gpu_replicas():
    defaults = asyncio.run(eval_server.health())["defaults"]
    assert defaults["gpus_per_model_instance"] == 1
    assert defaults["model_instances_per_side"] == 4
    assert defaults["worker_processes"] == 8
    assert defaults["model_parallel_strategy"] == "single_gpu_replicas"
    assert defaults["sequence_pipeline_depth"] == 1


def test_eight_workers_score_each_sequence_once_per_side():
    pool = object.__new__(PersistentModelWorkerPool)
    pool.specs = model_worker_specs(list(range(8)))
    pool.ready = {spec["worker_id"]: {"pipeline_depth": 1} for spec in pool.specs}
    pool.result_queue = Queue()
    seen = {"king": [], "challenger": []}

    class IndexedScoreQueue:
        def __init__(self, spec):
            self.spec = spec

        def put(self, command):
            role = self.spec["role"]
            indices = command["sequence_indices"]
            seen[role].extend(indices)
            pool.result_queue.put({
                "type": "result", "generation": "generation",
                "worker_id": self.spec["worker_id"], "role": role,
                "sequence_indices": indices,
                "losses": [float(index) + (0.25 if role == "king" else 0.0) for index in indices],
            })

    pool.command_queues = {spec["worker_id"]: IndexedScoreQueue(spec) for spec in pool.specs}
    request = EvalRequest(king_repo="king", challenger_repo="challenger", batch_size=3)
    events = []
    king, challenger, metadata = pool.score(
        [[index, 151645 if index < 10 else index, index, index] for index in range(35)],
        "generation", request, events.append,
    )
    assert king == [index + 0.25 for index in range(35)]
    assert challenger == [float(index) for index in range(35)]
    assert sorted(seen["king"]) == sorted(seen["challenger"]) == list(range(35))
    assert pool.result_queue.empty()
    counts = [2] * 10 + [3] * 25
    assert metadata["scored_tokens"] == counts
    assert metadata["planned_scored_tokens"] == sum(counts)
    assert events[-1]["avg_king_loss"] == round(float(np.average(king, weights=counts)), 6)
    assert events[-1]["avg_challenger_loss"] == round(float(np.average(challenger, weights=counts)), 6)
    verdict = eval_server.bootstrap_verdict(king, challenger, request, counts)
    assert verdict["avg_king_loss"] == events[-1]["avg_king_loss"]
    scores = eval_server._compute_source_scores(king, challenger, ["fixture"] * 35, counts)
    assert scores["fixture"]["avg_king_loss"] == verdict["avg_king_loss"]
    assert scores["fixture"]["n_scored_tokens"] == verdict["n_scored_tokens"]


def test_checkpoint_key_excludes_randomized_sequences(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    request = EvalRequest(king_repo="king", challenger_repo="challenger", seed=1)
    first = checkpoint_load_key(str(tmp_path), request, [0, 1])
    request.seed = 999
    request.n = 17
    assert checkpoint_load_key(str(tmp_path), request, [0, 1]) == first


def test_safetensor_keys_are_read_from_index_without_loading_payloads(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.layers.1.weight":"b.safetensors",'
        '"model.layers.0.weight":"a.safetensors"}}'
    )
    assert snapshot_safetensor_keys(str(tmp_path)) == [
        "model.layers.0.weight",
        "model.layers.1.weight",
    ]


def test_duplicate_check_hashes_all_model_shards_in_parallel(monkeypatch, tmp_path):
    king = tmp_path / "king"
    challenger = tmp_path / "challenger"
    king.mkdir()
    challenger.mkdir()
    for index in range(2):
        (king / f"model-{index}.safetensors").write_bytes(f"king-{index}".encode())
        (challenger / f"model-{index}.safetensors").write_bytes(
            f"challenger-{index}".encode()
        )

    lock = threading.Lock()
    active = 0
    max_active = 0

    def measured_sha256(path):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with lock:
            active -= 1
        return digest

    phases = []
    monkeypatch.setattr(eval_server.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(eval_server, "sha256_file", measured_sha256)
    result = eval_server.reject_duplicate_safetensors(
        str(king), str(challenger), on_phase=phases.append
    )

    assert result["king_safetensors_sha256"] != result["challenger_safetensors_sha256"]
    assert max_active == 4
    assert phases[0] == {
        "phase": "duplicate_check_start",
        "king_shards": 2,
        "challenger_shards": 2,
        "hash_workers": 4,
    }
    assert phases[-1]["phase"] == "duplicate_check_done"
    assert phases[-1]["hash_workers"] == 4


@pytest.mark.parametrize("attn_implementation", ["eager", "flash_attention_4"])
def test_direct_checkpoint_loader_resolves_tied_meta_weights(tmp_path, attn_implementation):
    config = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        n_positions=16,
        vocab_size=16,
        bos_token_id=0,
        eos_token_id=1,
    )
    GPT2LMHeadModel(config).save_pretrained(tmp_path, safe_serialization=True)
    request = EvalRequest(
        king_repo="king",
        challenger_repo="challenger",
        attn_implementation=attn_implementation,
    )
    loaded = load_eval_model(str(tmp_path), config, "cpu", "tiny", request, gpu_ids=[])
    assert not any(parameter.is_meta for parameter in loaded.parameters())
    assert next(loaded.parameters()).dtype == torch.bfloat16
    assert loaded.config._attn_implementation == attn_implementation


def test_kernel_cache_is_architecture_keyed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config_a = SimpleNamespace(
        to_dict=lambda: {"model_type": "mimo_v2", "hidden_size": 4096, "num_experts": 64}
    )
    config_b = SimpleNamespace(
        to_dict=lambda: {"model_type": "mimo_v2", "hidden_size": 8192, "num_experts": 64}
    )
    assert kernel_cache_identity(config_a, [0, 1]) == kernel_cache_identity(config_a, [2, 3])
    assert kernel_cache_identity(config_a, [0, 1]) != kernel_cache_identity(config_b, [0, 1])


def test_single_gpu_scorer_serializes_batches_and_preserves_indices(monkeypatch):
    events = []
    output = Queue()

    def fake_loss(_model, token_batches, _chunk_size, **_kwargs):
        indices = tuple(tokens[0] for tokens in token_batches)
        events.append((indices, "start"))
        time.sleep(0.02)
        events.append((indices, "done"))
        return [float(tokens[0]) for tokens in token_batches]

    monkeypatch.setattr(eval_server, "compute_per_sequence_loss", fake_loss)
    request = EvalRequest(king_repo="king", challenger_repo="challenger", batch_size=2)
    scorer = ModelSequenceScorer(
        object(), request,
        {"worker_id": "king-0", "role": "king", "gpu_ids": [3]},
        output, "generation",
    )
    scorer.submit_batch([7, 2], [[7], [2]])
    scorer.submit(9, [9])
    scorer.close()
    assert scorer.depth == 1
    assert events == [((7, 2), "start"), ((7, 2), "done"), ((9,), "start"), ((9,), "done")]
    first, second = output.get_nowait(), output.get_nowait()
    assert first["sequence_indices"] == [7, 2]
    assert first["losses"] == [7.0, 2.0]
    assert second["sequence_index"] == 9
    assert second["losses"] == [9.0]
    assert first["generation"] == second["generation"] == "generation"


def test_single_gpu_scorer_reports_errors_to_worker_pool(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("test scoring failure")

    monkeypatch.setattr(eval_server, "compute_per_sequence_loss", fail)
    output = Queue()
    scorer = ModelSequenceScorer(
        object(), EvalRequest(king_repo="king", challenger_repo="challenger"),
        {"worker_id": "challenger-3", "role": "challenger", "gpu_ids": [7]},
        output, "generation",
    )
    scorer.submit(0, [0])
    scorer.close()
    result = output.get_nowait()
    assert result["type"] == "error"
    assert result["worker_id"] == "challenger-3"
    assert result["generation"] == "generation"
    assert result["error"] == "test scoring failure"


def test_oom_backoff_keeps_every_sample_and_original_order(monkeypatch):
    calls, cleared, retries = [], [], []

    def score(_model, rows, chunk, **kwargs):
        calls.append([row[0] for row in rows])
        if len(rows) > 2:
            raise eval_server.ScoringOOM("simulated allocation failure", "forward")
        return [float(row[0]) for row in rows]

    monkeypatch.setattr(eval_server, "compute_per_sequence_loss", score)
    monkeypatch.setattr(eval_server, "empty_worker_cuda_cache", lambda ids: cleared.append(ids))
    rows = [[i, i + 1] for i in range(7)]
    losses = eval_server.compute_with_oom_backoff(object(), rows, 1024, gpu_ids=[3], enabled=True, retries=retries)
    assert losses == list(map(float, range(7)))
    assert calls == [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2], [0], [1, 2], [3, 4, 5, 6], [3, 4], [5, 6]]
    assert len(cleared) == len(retries) == 3
    assert all(r["action"] == "split_batch" for r in retries)
    with pytest.raises(eval_server.ScoringOOM):
        eval_server.compute_with_oom_backoff(object(), rows, 1024, gpu_ids=[3], enabled=False, retries=[])


def test_single_document_oom_reduces_head_workspace_but_never_truncates(monkeypatch):
    observed = []
    row = list(range(8193))

    def score(_model, rows, chunk, **kwargs):
        assert rows == [row]
        observed.append(chunk)
        if chunk > 128:
            raise eval_server.ScoringOOM("simulated projection allocation failure", "lm_head")
        return [1.25]

    monkeypatch.setattr(eval_server, "compute_per_sequence_loss", score)
    monkeypatch.setattr(eval_server, "empty_worker_cuda_cache", lambda ids: None)
    retries = []
    result = eval_server.compute_with_oom_backoff(object(), [row], 1024, gpu_ids=[0], enabled=True, retries=retries)
    assert result == [1.25]
    assert observed == [1024, 512, 256, 128]
    assert all(r["action"] == "halve_lm_head_chunk" for r in retries)

    def fails_forward(*args, **kwargs):
        raise eval_server.ScoringOOM("forward exceeds memory", "forward")

    monkeypatch.setattr(eval_server, "compute_per_sequence_loss", fails_forward)
    with pytest.raises(RuntimeError, match="complete 8193-token sample cannot fit"):
        eval_server.compute_with_oom_backoff(object(), [row], 1024, gpu_ids=[0], enabled=True, retries=[])


def test_indexed_npy_loader_preserves_row_and_window_indices(tmp_path):
    request = SimpleNamespace(seq_len=3)

    matrix_path = tmp_path / "matrix.npy"
    matrix = np.arange(12, dtype=np.uint32).reshape(4, 3)
    np.save(matrix_path, matrix)
    matrix_rows = eval_server.load_indexed_sequences_from_npy_shard(
        str(matrix_path), request, np.random.default_rng(7)
    )
    assert {index for index, _sequence in matrix_rows} == set(range(4))
    assert all(sequence == matrix[index].tolist() for index, sequence in matrix_rows)

    stream_path = tmp_path / "stream.npy"
    stream = np.arange(15, dtype=np.uint32)
    np.save(stream_path, stream)
    stream_rows = eval_server.load_indexed_sequences_from_npy_shard(
        str(stream_path), request, np.random.default_rng(11)
    )
    assert {index for index, _sequence in stream_rows} == set(range(5))
    assert all(
        sequence == stream[index * 3 : (index + 1) * 3].tolist()
        for index, sequence in stream_rows
    )


def test_sampler_keeps_shard_provenance_aligned_through_shuffle(monkeypatch, tmp_path):
    request = SimpleNamespace(
        n=4,
        seq_len=2,
        vocab_size=0,
        block_hash="0x" + "a" * 64,
        hotkey="hotkey",
        dataset_sources=[
            {
                "name": "fixture",
                "proportion": 1.0,
                "target_sequences": 4,
                "shards": [
                    {
                        "url": "https://datasets.example/alpha.npy",
                        "sha256": "a" * 64,
                        "size_bytes": 100,
                        "n_tokens": 100,
                        "target_sequences": 2,
                    },
                    {
                        "url": "https://datasets.example/beta.npy",
                        "sha256": "b" * 64,
                        "size_bytes": 100,
                        "n_tokens": 100,
                        "target_sequences": 2,
                    },
                ],
            }
        ],
    )
    loaded = {
        "alpha.npy": [(7, [107, 1]), (8, [108, 1])],
        "beta.npy": [(20, [220, 1]), (21, [221, 1])],
    }

    def fake_load(shard, _request, _rng, _limit, on_phase=None):
        del on_phase
        return tmp_path, loaded[Path(shard.url).name]

    monkeypatch.setattr(sources, "_load_with_retry", fake_load)
    monkeypatch.setattr(sources.base, "dataset_seed", lambda _request: 123)
    monkeypatch.setattr(
        sources.base, "dataset_seed_material", lambda _request: "fixture-seed"
    )

    sequences, metadata = sources.sample_pretokenized_sequences(request)
    observed = {
        sequence[0]: (
            provenance["shard_group_index"],
            provenance["shard_index"],
            provenance["shard_sequence_index"],
        )
        for sequence, provenance in zip(
            sequences, metadata["_sample_provenance"], strict=True
        )
    }

    assert observed == {
        107: (0, 0, 7),
        108: (0, 0, 8),
        220: (0, 1, 20),
        221: (0, 1, 21),
    }
    assert metadata["_source_labels"] == ["fixture"] * 4


def test_sample_results_are_compact_and_index_aligned():
    result = eval_server._build_sample_results(
        [1.2, 1.3],
        [1.1, 1.4],
        [
            {
                "shard_group_index": 0,
                "shard_index": 2,
                "shard_sequence_index": 42,
            },
            {
                "shard_group_index": 1,
                "shard_index": 0,
                "shard_sequence_index": 9,
            },
        ],
        [3, 2],
    )

    assert result == {
        "format": "columnar-masked-v1",
        "n_samples": 2,
        "shard_group_index": [0, 1],
        "shard_index": [2, 0],
        "shard_sequence_index": [42, 9],
        "king_loss": [1.2, 1.3],
        "challenger_loss": [1.1, 1.4],
        "scored_tokens": [3, 2],
    }
