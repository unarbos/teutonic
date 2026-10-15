"""Replication correctness and persistent-worker loading lifecycle."""

import os
from pathlib import Path
from queue import Queue
from unittest.mock import Mock

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from teutonic.evaluator import engine, replica_loading, sequence_parallel


def tiny_checkpoint(path, seed=1):
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2Config(
        n_layer=1, n_head=2, n_embd=8, n_positions=16, vocab_size=16,
        bos_token_id=0, eos_token_id=1,
    ))
    model.save_pretrained(path, safe_serialization=True)


def test_chunks_bound_memory_and_cover_mixed_buffers_and_large_tensors():
    items = [
        ("weight", torch.arange(39, dtype=torch.bfloat16)),
        ("empty", torch.empty(0)),
        ("mask", torch.tensor([True, False, True])),
        ("scalar", torch.tensor(7, dtype=torch.int64)),
    ]
    chunks = list(replica_loading.tensor_chunks(items, chunk_bytes=16))
    assert all(sum(p.numel() * p.element_size() for p in chunk) <= 16 for chunk in chunks)
    assert all(len({p.dtype for p in chunk}) == 1 for chunk in chunks)
    assert sum(p.numel() for chunk in chunks for p in chunk) == 43
    assert torch.equal(torch.cat([p for c in chunks for p in c if p.dtype == torch.bfloat16]), items[0][1])
    with pytest.raises(ValueError, match="non-contiguous"):
        list(replica_loading.tensor_chunks([("bad", torch.zeros(2, 3).t())]))


def test_receiver_restores_ties_without_reading_checkpoint(tmp_path, monkeypatch):
    tiny_checkpoint(tmp_path)
    req = engine.EvalRequest(king_repo="king", challenger_repo="challenger")
    config, _ = engine.load_model_config(str(tmp_path), req, "replica")
    import accelerate
    monkeypatch.setattr(accelerate, "load_checkpoint_and_dispatch", Mock(side_effect=AssertionError("disk load")))
    model = engine.allocate_eval_replica(config, "cpu", "replica", req)
    assert model.transformer.wte.weight is model.lm_head.weight
    assert all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in model.parameters())
    assert not model.training and not model.config.use_cache


class LoadQueue:
    def __init__(self, pool, spec, reused):
        self.pool, self.spec, self.reused = pool, spec, reused
        self.commands = []

    def put(self, command):
        self.commands.append(command)
        common = {"worker_id": self.spec["worker_id"], "role": self.spec["role"],
                  "generation": command["generation"], "reused_model": self.reused}
        if command["type"] == "load":
            self.pool.result_queue.put({**common, "type": "load_prepared", "prepare_s": 1})
        else:
            assert command["type"] == "load_broadcast"
            self.pool.result_queue.put({**common, "type": "ready", "gpu_ids": self.spec["gpu_ids"],
                                        "pid": 1, "pipeline_depth": 1, "load_timing": {}})


@pytest.mark.parametrize("reused,expected", [
    (set(), {"king": True, "challenger": True}),
    ({"king-0", "king-1", "king-2", "king-3"}, {"king": False, "challenger": True}),
    ({"king-0"}, {"king": True, "challenger": True}),
])
def test_parent_broadcasts_only_groups_needing_weights(tmp_path, reused, expected):
    pool = object.__new__(engine.PersistentModelWorkerPool)
    pool.specs = engine.model_worker_specs(list(range(8)))
    pool.ready = {}
    pool.result_queue = Queue()
    pool.parallel_pairs = pool.parallel_directory = None
    pool.command_queues = {s["worker_id"]: LoadQueue(pool, s, s["worker_id"] in reused) for s in pool.specs}
    req = engine.EvalRequest(king_repo="king", challenger_repo="challenger")
    events = []
    pool.load_models(req, str(tmp_path), str(tmp_path), events.append)
    assert len(pool.ready) == 8
    for spec in pool.specs:
        load, broadcast = pool.command_queues[spec["worker_id"]].commands
        assert load["replica_rank"] == int(spec["worker_id"][-1])
        assert load["replica_count"] == 4
        assert broadcast["required"] == expected[spec["role"]]
    assert sum(e["phase"] == "model_worker_ready" for e in events) == 8


def test_preparation_failure_closes_all_peers(tmp_path):
    pool = object.__new__(engine.PersistentModelWorkerPool)
    pool.specs = engine.model_worker_specs(list(range(8)))
    pool.ready = {"old": {}}
    pool.parallel_pairs = pool.parallel_directory = None
    pool.command_queues = {s["worker_id"]: Mock() for s in pool.specs}
    pool.close = Mock()
    pool.result_queue = Mock()

    def failure(**kwargs):
        command = pool.command_queues["king-0"].put.call_args.args[0]
        return {"type": "error", "generation": command["generation"],
                "worker_id": "king-0", "error": "checkpoint failed", "traceback": "fixture"}

    pool.result_queue.get.side_effect = failure
    req = engine.EvalRequest(king_repo="king", challenger_repo="challenger")
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        pool.load_models(req, str(tmp_path), str(tmp_path), lambda _: None)
    pool.close.assert_called_once_with(force=True)
    assert not pool.ready


def broadcast_worker(rank, gpu_ids, snapshot, rendezvous, mismatch):
    gpu_id = gpu_ids[rank]
    torch.cuda.set_device(gpu_id)
    torch.set_num_threads(2)
    engine.configure_deterministic_execution()
    req = engine.EvalRequest(king_repo=snapshot, challenger_repo=snapshot)
    config, _ = engine.load_model_config(snapshot, req, f"test-{rank}")
    if rank == 0:
        model = engine.load_eval_model(snapshot, config, f"cuda:{gpu_id}", "source", req, [gpu_id])
        reference = {name: value.detach().cpu() for name, value in replica_loading.model_tensors(model)}
        tokens = torch.tensor([[1, 2, 3, 4]], device=f"cuda:{gpu_id}")
        with torch.inference_mode():
            logits = model(tokens).logits.cpu()
        torch.save({"state": reference, "logits": logits}, Path(snapshot) / "reference.pt")
    else:
        model = engine.allocate_eval_replica(config, f"cuda:{gpu_id}", "replica", req)
        if mismatch:
            model.register_buffer("extra", torch.ones(1, device=f"cuda:{gpu_id}"))
    if mismatch:
        try:
            with pytest.raises(RuntimeError, match="layouts differ"):
                replica_loading.broadcast_model(model, rank=rank, world_size=2,
                                                rendezvous=rendezvous, gpu_id=gpu_id)
        finally:
            torch.distributed.destroy_process_group()
        return
    # Tiny chunks deliberately exercise splitting parameters and mixed-dtype buffers.
    result = replica_loading.broadcast_model(model, rank=rank, world_size=2,
                                             rendezvous=rendezvous, gpu_id=gpu_id, chunk_bytes=128)
    assert result["chunks"] > 10
    assert not torch.distributed.is_initialized()
    assert model.transformer.wte.weight is model.lm_head.weight
    reference = torch.load(Path(snapshot) / "reference.pt", weights_only=True)
    for name, value in replica_loading.model_tensors(model):
        assert torch.equal(value.detach().cpu().reshape(-1).view(torch.uint8), reference["state"][name].reshape(-1).view(torch.uint8)), name
    with torch.inference_mode():
        logits = model(torch.tensor([[1, 2, 3, 4]], device=f"cuda:{gpu_id}")).logits.cpu()
    assert torch.equal(logits, reference["logits"])
    # Temporary loading groups must allow subsequent sequence-parallel groups.
    sequence_parallel.initialize(rank, rendezvous + "-pair", gpu_id)
    torch.distributed.barrier(device_ids=[gpu_id])
    torch.distributed.destroy_process_group()
    replica_loading.broadcast_model(model, rank=rank, world_size=2,
                                    rendezvous=rendezvous + "-reload", gpu_id=gpu_id)
    assert not torch.distributed.is_initialized()


def available_gpu_ids(count):
    # Opt in explicitly so ordinary tests never reserve another user's GPUs.
    value = os.environ.get("TEUTONIC_REPLICA_TEST_GPUS", "")
    ids = [int(gpu) for gpu in value.split(",") if gpu]
    if len(ids) < count or not torch.cuda.is_available():
        pytest.skip(f"set TEUTONIC_REPLICA_TEST_GPUS to {count} available GPUs")
    return ids[:count]


@pytest.mark.parametrize("mismatch", [False, True])
def test_nccl_copies_every_tensor_and_preserves_forward_output(tmp_path, mismatch):
    gpu_ids = available_gpu_ids(2)
    tiny_checkpoint(tmp_path)
    torch.multiprocessing.spawn(broadcast_worker, args=(
        gpu_ids, str(tmp_path), (tmp_path / "broadcast").as_uri(), mismatch,
    ), nprocs=2, join=True)


def test_live_pool_reload_reuse_and_sequence_parallel_lifecycle(tmp_path, monkeypatch):
    gpu_ids = available_gpu_ids(4)
    monkeypatch.setenv("TEUTONIC_KERNEL_CACHE_DIR", str(tmp_path / "kernels"))
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    monkeypatch.setattr(engine, "model_worker_specs", lambda ids: [
        {"worker_id": f"{role}-{rank}", "role": role, "gpu_ids": [ids[offset + rank]]}
        for role, offset in (("king", 0), ("challenger", 2)) for rank in range(2)
    ])
    for index in range(3):
        tiny_checkpoint(tmp_path / str(index), seed=index)
    snapshots = [str(tmp_path / str(index)) for index in range(3)]
    req = engine.EvalRequest(king_repo="king", challenger_repo="challenger")
    events = []
    pool = engine.PersistentModelWorkerPool(gpu_ids)
    try:
        generation = pool.load_models(req, *snapshots[:2], events.append)
        assert len(pool.ready) == 4
        starts = [e for e in events if e["phase"].endswith("_direct_gpu_load_start")]
        assert {e["worker_id"] for e in starts} == {"king-0", "challenger-0"}
        assert all(not m["reused_model"] and m["load_timing"]["broadcast"] for m in pool.ready.values())
        sequence_parallel.prepare_workers(pool, generation, events.append)
        events.clear()
        generation = pool.load_models(req, *snapshots[:2], events.append)
        assert all(m["reused_model"] and m["load_timing"]["broadcast"] is None for m in pool.ready.values())
        assert all(m["sequence_parallel_rank"] is None for m in pool.ready.values())
        assert not any(e["phase"].endswith("_direct_gpu_load_start") for e in events)
        sequence_parallel.prepare_workers(pool, generation, events.append)
        events.clear()
        pool.load_models(req, snapshots[0], snapshots[2], events.append)
        assert all(m["reused_model"] == (m["role"] == "king") for m in pool.ready.values())
        starts = [e for e in events if e["phase"].endswith("_direct_gpu_load_start")]
        assert [e["worker_id"] for e in starts] == ["challenger-0"]
        # A real load error kills waiting peers and leaves no usable partial pool.
        with pytest.raises(RuntimeError, match="no .safetensors"):
            pool.load_models(req, snapshots[0], str(tmp_path / "missing"), events.append)
        assert pool.dead_workers() == [s["worker_id"] for s in pool.specs]
        assert not pool.ready
    finally:
        pool.close(force=True)
