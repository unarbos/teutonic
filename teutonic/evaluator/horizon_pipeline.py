"""Deterministic Horizon layer pipeline and isolated document-scoring RPC.

Grouped expert math follows the supplied eval_horizon_fast.py experiment.
Torch deterministic algorithms are mandatory. Final document sums use fsum on
FP32 token CE values so head-chunk and micro-batch boundaries cannot alter the
reduction order. Validation comparisons belong in tests, never in live duels.
"""

import argparse
import hashlib
import json
import math
import os
import queue
import threading
import time
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np

BACKEND = "horizon-layer-pipeline-bf16-deterministic-v1"
PINS = {"torch": "2.15.0", "transformers": "5.15.1", "accelerate": "1.15.0"}
MAX_REQUEST_DOCUMENTS = 1024
MAX_REQUEST_TOKENS = 4 * 1024 * 1024


def settings():
    return {"dtype": "bfloat16", "pipeline_parallel_size": 8, "attention": "sdpa",
            "batch_tokens": 131072, "min_batch_tokens": 8192, "max_batch": 128, "moe_chunk": 32768,
            "head_chunk": 8192, "queue_depth": 2, "max_model_len": 524288,
            "max_request_documents": MAX_REQUEST_DOCUMENTS, "max_request_tokens": MAX_REQUEST_TOKENS,
            "seed": 42, "deterministic_algorithms": True, "reduction": "fp32-ce-cpu-fsum",
            "length_sort": "stable-longest-batches-first", "eos_token_id": 1}


def deterministic_env():
    return {"PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTORCH_ALLOC_CONF": "expandable_segments:True"}


def code_identity():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("horizon_pipeline.py", "horizon_pipeline_runtime.py", "vllm_runtime.py")}


def identity():
    versions = {name: version(name) for name in PINS}
    if any(versions[name].split("+")[0] != pin for name, pin in PINS.items()):
        raise RuntimeError(f"Horizon worker dependencies differ from tested pins: {versions}")
    import torch
    if torch.version.cuda != "13.2":
        raise RuntimeError("Horizon pipeline requires the tested Torch CUDA 13.2 build")
    return {"backend": BACKEND, "versions": versions, "torch_build": torch.__version__,
            "cuda_build": torch.version.cuda, "numpy": version("numpy"), "code": code_identity(),
            "settings": settings(), "environment": deterministic_env()}


def input_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def layer_counts(layers, devices=8):
    if devices != 8 or layers < devices:
        raise ValueError("Horizon requires eight nonempty pipeline stages")
    last = max(1, layers // devices - 4)
    rest = layers - last
    return [rest // (devices - 1) + int(i >= devices - 1 - rest % (devices - 1))
            for i in range(devices - 1)] + [last]


def validate_rows(rows, configuration, vocab_size):
    if not rows or len(rows) > configuration["max_request_documents"]:
        raise ValueError("invalid Horizon document request size")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate Horizon document IDs")
    total = 0
    for row in rows:
        tokens = row["tokens"]
        if (not isinstance(row["id"], str) or not 2 <= len(tokens) < configuration["max_model_len"]
                or tokens[-1] != configuration["eos_token_id"]
                or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens)):
            raise ValueError("invalid Horizon document tokens, context length or EOS; no truncation")
        total += len(tokens)
    if total > configuration["max_request_tokens"]:
        raise ValueError("Horizon request exceeds token bound")


def microbatches(rows, configuration):
    batches, current = [], []
    for row in sorted(rows, key=lambda r: len(r["tokens"])):
        length = len(row["tokens"])
        if current and (len(current) >= configuration["max_batch"]
                        or (len(current) + 1) * length > configuration["batch_tokens"]):
            batches.append(current)
            current = []
        # A single long document keeps full context even beyond the batch budget.
        current.append(row)
    if current:
        batches.append(current)
    return list(reversed(batches))


def padded_length(rows, configuration):
    """Avoid small-row GEMM variants without changing full-sized micro-batches."""
    return max(max(len(row["tokens"]) for row in rows),
               math.ceil(configuration["min_batch_tokens"] / len(rows)))


def reduce_document_losses(losses, rows):
    """Canonical per-document reduction, independent of neighboring documents."""
    offset, values = 0, []
    for row in rows:
        count = len(row["tokens"]) - 1
        part = losses[offset:offset + count]
        if len(part) != count or not np.all(np.isfinite(part)) or np.any(part < 0):
            raise ValueError("invalid Horizon token losses")
        values.append({"id": row["id"], "nll": math.fsum(part.tolist()),
                       "scored_tokens": count, "input_sha256": input_hash(row["tokens"])})
        offset += count
    if offset != len(losses):
        raise ValueError("Horizon target accounting differs")
    return values


def _moe_forward(self, hidden_states):
    import torch
    import torch.nn.functional as F
    residuals = hidden_states
    batch, length, hidden = hidden_states.shape
    x = hidden_states.view(-1, hidden)
    router_logits = F.linear(x, self.gate.weight)
    if self.router_score_func == "softmax":
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    elif self.router_score_func == "sigmoid":
        routing_weights = F.sigmoid(router_logits.to(torch.float32))
    else:
        raise ValueError("unsupported Horizon router")
    choice = routing_weights
    if self.gate.bias is not None:
        choice = routing_weights + self.gate.bias.to(routing_weights.dtype)
    _, selected = torch.topk(choice, self.top_k, dim=-1)
    routing_weights = torch.gather(routing_weights, dim=-1, index=selected)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = (routing_weights * self.router_scaling_factor).to(x.dtype)
    out = torch.empty(x.shape[0], hidden, dtype=x.dtype, device=x.device)
    for start in range(0, x.shape[0], self._pipeline_moe_chunk):
        stop = start + self._pipeline_moe_chunk
        xc, sc, rc = x[start:stop], selected[start:stop], routing_weights[start:stop]
        flat = sc.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token = order // self.top_k
        offsets = torch.cumsum(torch.bincount(flat, minlength=self.num_experts), 0).to(torch.int32)
        gu = torch._grouped_mm(xc[token], self.w13.transpose(1, 2), offs=offsets)
        gate, up = gu.split(self.moe_inter, dim=-1)
        activation = F.silu(gate) * up
        del gu, gate, up
        projected = torch._grouped_mm(activation, self.w2.transpose(1, 2), offs=offsets)
        del activation
        accumulator = torch.zeros(xc.shape[0], hidden, dtype=torch.float32, device=x.device)
        accumulator.index_add_(0, token, projected.float().mul_(rc.reshape(-1)[order, None].float()))
        out[start:stop] = accumulator.to(x.dtype)
        del projected, accumulator
    final = out.view(batch, length, hidden)
    if self.num_shared_experts > 0:
        final = final + self.shared_experts(residuals)
    return final, router_logits


class HorizonPipeline:
    def __init__(self, snapshot):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        self.configuration = settings()
        if torch.cuda.device_count() != 8:
            raise ValueError("Horizon pipeline requires eight visible GPUs")
        torch.manual_seed(self.configuration["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
        if cfg.model_type != "k2_horizon" or cfg.hidden_act != "silu":
            raise ValueError("unexpected Horizon architecture")
        if cfg.max_position_embeddings < self.configuration["max_model_len"]:
            raise ValueError("Horizon checkpoint context is below the scoring limit")
        self.vocab_size = cfg.vocab_size
        counts = layer_counts(cfg.num_hidden_layers)
        self.layer_devices = [gpu for gpu, count in enumerate(counts) for _ in range(count)]
        device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 7, "lm_head": 7}
        device_map.update({f"model.layers.{i}": gpu for i, gpu in enumerate(self.layer_devices)})
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True,
            attn_implementation="sdpa", device_map=device_map,
        ).eval()
        with torch.no_grad():
            for layer in self.model.model.layers:
                m = layer.mlp
                if not hasattr(m, "experts"):
                    continue
                m.w13 = torch.stack([torch.cat([e.gate_proj.weight, e.up_proj.weight], 0)
                                     for e in m.experts]).contiguous()
                m.w2 = torch.stack([e.down_proj.weight for e in m.experts]).contiguous()
                m.moe_inter = m.experts[0].gate_proj.weight.shape[0]
                m._pipeline_moe_chunk = self.configuration["moe_chunk"]
                for expert in m.experts:
                    for linear in (expert.gate_proj, expert.up_proj, expert.down_proj):
                        linear.weight.data = torch.empty(0, dtype=linear.weight.dtype, device=linear.weight.device)
                del m.experts
                m.forward = types.MethodType(_moe_forward, m)
                torch.cuda.empty_cache()
        self.stages = [[layer for i, layer in enumerate(self.model.model.layers) if self.layer_devices[i] == gpu]
                       for gpu in range(8)]

    def score(self, rows):
        import torch
        import torch.nn.functional as F
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("Horizon deterministic algorithms were disabled")
        validate_rows(rows, self.configuration, self.vocab_size)
        batches = microbatches(rows, self.configuration)
        queues = [queue.Queue(maxsize=self.configuration["queue_depth"]) for _ in range(8)]
        results = queue.Queue(maxsize=self.configuration["queue_depth"])
        errors = queue.Queue()
        cancelled = threading.Event()

        def put(target, item):
            while not cancelled.is_set():
                try:
                    target.put(item, timeout=.1)
                    return
                except queue.Full:
                    pass

        def get(target):
            while not cancelled.is_set():
                try:
                    return target.get(timeout=.1)
                except queue.Empty:
                    pass
            return None

        def feeder():
            try:
                for bid, batch in enumerate(batches):
                    if cancelled.is_set():
                        return
                    ids = np.zeros((len(batch), padded_length(batch, self.configuration)), dtype=np.int64)
                    for i, row in enumerate(batch):
                        ids[i, :len(row["tokens"])] = row["tokens"]
                    put(queues[0], (bid, ids, ids))
                put(queues[0], None)
            except BaseException as error:  # noqa: BLE001 - propagate thread failures to the RPC owner
                errors.put(error)
                cancelled.set()

        def stage(gpu):
            try:
                device = torch.device("cuda", gpu)
                torch.cuda.set_device(device)
                with torch.no_grad():
                    while not cancelled.is_set():
                        item = get(queues[gpu])
                        if item is None:
                            put(queues[gpu + 1] if gpu < 7 else results, None)
                            return
                        bid, payload, ids = item
                        hidden = self.model.model.embed_tokens(torch.from_numpy(payload).to(device)) if gpu == 0 else payload
                        length = hidden.shape[1]
                        positions = torch.arange(length, device=device).unsqueeze(0)
                        embeddings = self.model.model.rotary_emb(hidden, positions)
                        for layer in self.stages[gpu]:
                            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=None, position_ids=positions)
                        if gpu < 7:
                            next_hidden = hidden.to(torch.device("cuda", gpu + 1))
                            torch.cuda.current_stream(device).synchronize()
                            put(queues[gpu + 1], (bid, next_hidden, ids))
                            del hidden, next_hidden, item, payload
                            continue
                        hidden = self.model.model.norm(hidden)
                        targets = torch.from_numpy(ids[:, 1:]).to(device)
                        lengths = torch.tensor([len(r["tokens"]) for r in batches[bid]], device=device)
                        valid = torch.arange(length - 1, device=device)[None, :] < (lengths[:, None] - 1)
                        scored_hidden, scored_targets = hidden[:, :-1][valid], targets[valid]
                        token_losses = torch.empty_like(scored_targets, dtype=torch.float32)
                        chunk = self.configuration["head_chunk"]
                        for start in range(0, scored_hidden.shape[0], chunk):
                            logits = self.model.lm_head(scored_hidden[start:start + chunk]).float()
                            token_losses[start:start + chunk] = F.cross_entropy(
                                logits, scored_targets[start:start + chunk], reduction="none")
                            del logits
                        values = reduce_document_losses(token_losses.cpu().numpy(), batches[bid])
                        put(results, values)
                        del hidden, scored_hidden, scored_targets, token_losses, item, payload
            except BaseException as error:  # noqa: BLE001 - propagate thread failures to the RPC owner
                errors.put(error)
                cancelled.set()

        threads = [threading.Thread(target=stage, args=(gpu,), daemon=True) for gpu in range(8)]
        threads.append(threading.Thread(target=feeder, daemon=True))
        for thread in threads:
            thread.start()
        values = {}
        try:
            while True:
                if not errors.empty():
                    raise errors.get()
                result = get(results)
                if not errors.empty():
                    raise errors.get()
                if result is None:
                    break
                for row in result:
                    if row["id"] in values:
                        raise ValueError("duplicate Horizon result")
                    values[row["id"]] = row
            if set(values) != {r["id"] for r in rows}:
                raise ValueError("missing Horizon results")
            return [values[row["id"]] for row in rows]
        finally:
            cancelled.set()
            for thread in threads:
                thread.join(timeout=5)
            if any(thread.is_alive() for thread in threads):
                raise RuntimeError("Horizon pipeline thread failed to stop")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity", action="store_true")
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--response-fd", type=int)
    args = p.parse_args()
    runtime = identity()
    if args.identity:
        print(json.dumps(runtime))
        return
    if args.snapshot is None or args.response_fd is None:
        p.error("snapshot and response-fd are required")
    import ctypes
    import signal
    parent = os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "could not bind Horizon worker lifetime")
    if parent == 1 or os.getppid() != parent:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    for name, expected in deterministic_env().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"Horizon worker environment mismatch: {name}")
    with os.fdopen(args.response_fd, "w", buffering=1) as replies:
        def send(value):
            replies.write(json.dumps(value, allow_nan=False) + "\n")
        started = time.monotonic()
        model = HorizonPipeline(args.snapshot)
        send({"type": "ready", "identity": runtime, "settings": settings(),
              "load_seconds": time.monotonic() - started})
        import sys
        for line in sys.stdin:
            command = json.loads(line)
            if command["type"] == "stop":
                send({"type": "stopped"})
                return
            if command["type"] != "score":
                raise ValueError("unknown Horizon scorer command")
            started = time.monotonic()
            values = model.score(command["rows"])
            send({"type": "scores", "batch": command["batch"], "rows": values,
                  "seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
