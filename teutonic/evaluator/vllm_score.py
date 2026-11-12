"""Pinned vLLM document scorer; GPU imports stay inside the worker process.

stdin carries bounded batches, a dedicated inherited FD carries replies, and
stdout/stderr are logs. Tokenizer parity and repeatability checks belong in tests.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

BACKEND = "vllm-0.33-v1-bf16-batch-invariant-replicas-v2"
PINS = {"vllm": "0.33.0", "torch": "2.14.0", "transformers": "5.20.2"}
SINGLE_GPU_MAX_TOKENS = 128_974


def settings(role, tensor_parallel_size=None, max_model_len=None):
    if role not in {"horizon", "king", "challenger"}:
        raise ValueError("unknown scorer role")
    horizon = role == "horizon"
    tp = tensor_parallel_size if tensor_parallel_size is not None else (8 if horizon else 1)
    if tp not in ((8,) if horizon else (1, 2)):
        raise ValueError("Horizon requires TP8; MiMo requires TP1 replicas or TP2 large-document groups")
    value = {
        "trust_remote_code": True, "model_impl": "vllm", "dtype": "bfloat16",
        "tensor_parallel_size": tp, "enable_expert_parallel": True,
        "max_model_len": 524288 if horizon else (SINGLE_GPU_MAX_TOKENS + 1 if tp == 1 else 131073),
        "max_num_seqs": 32 if horizon else 16,
        "max_num_batched_tokens": 32768 if horizon else 2048 * tp,
        "kv_cache_memory_bytes": (32 if horizon or tp == 2 else 16) * 1024**3,
        "gpu_memory_utilization": 0.8, "seed": 42, "skip_tokenizer_init": True,
        "enable_prefix_caching": False, "enforce_eager": True,
        "disable_custom_all_reduce": False,
    }
    if max_model_len is not None:
        if type(max_model_len) is not int or not 3 <= max_model_len <= 1048576:
            raise ValueError("invalid scoring context limit")
        if not horizon and tp == 1 and max_model_len > SINGLE_GPU_MAX_TOKENS + 1:
            raise ValueError("large documents require the two-GPU path")
        value["max_model_len"] = max_model_len
    if not horizon:
        value["attention_config"] = {"backend": "TRITON_ATTN_DIFFKV"}
    return value


def deterministic_env(role):
    settings(role)  # Validate before selecting a role-specific plugin.
    env = {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1",
            "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8",
            "VLLM_PLUGINS": "" if role == "horizon" else "teutonic_mimo",
            "VLLM_USE_FUSED_MOE_GROUPED_TOPK": "1" if role == "horizon" else "0"}
    if role == "horizon":
        env["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
    return env


def code_identity():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("vllm_score.py", "mimo_vllm.py")}


def identity():
    versions = {name: version(name) for name in PINS}
    if any(versions[name].split("+")[0] != pin for name, pin in PINS.items()):
        raise RuntimeError(f"vLLM worker dependencies differ from validated pins: {versions}")
    from importlib.metadata import distributions, entry_points
    plugins = entry_points(group="vllm.general_plugins", name="teutonic_mimo")
    if len(plugins) != 1 or next(iter(plugins)).value != "teutonic.evaluator.mimo_vllm:register":
        raise RuntimeError("install the production teutonic_mimo plugin in the vLLM environment")
    dependencies = {item.metadata["Name"]: item.version for item in distributions()}
    import torch
    if torch.version.cuda != "13.0":
        raise RuntimeError("the validated vLLM worker requires the Torch CUDA 13.0 build")
    return {"backend": BACKEND, "versions": versions, "dependencies": dependencies,
            "torch_build": torch.__version__, "cuda_build": torch.version.cuda,
            "code": code_identity(),
            "settings": {role: settings(role) for role in ("horizon", "king", "challenger")},
            "large_document_settings": settings("king", 2),
            "environment": {role: deterministic_env(role) for role in ("horizon", "king", "challenger")}}


def input_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def score_batch(llm, params, rows, role, configuration=None):
    maximum = (configuration or settings(role))["max_model_len"]
    eos = 1 if role == "horizon" else 151645
    if (not rows or len(rows) > 128 or len({r["id"] for r in rows}) != len(rows)
            or any(not 2 <= len(r["tokens"]) < maximum or r["tokens"][-1] != eos for r in rows)):
        raise ValueError("invalid document batch, context length or EOS; no truncation allowed")
    outputs = llm.generate([{"prompt_token_ids": r["tokens"]} for r in rows], params, use_tqdm=False)
    values = []
    for row, output in zip(rows, outputs, strict=True):
        tokens, logprobs = row["tokens"], output.prompt_logprobs
        if (list(output.prompt_token_ids) != tokens or logprobs is None
                or len(logprobs) != len(tokens) or logprobs[0]):
            raise ValueError("prompt/target accounting changed")
        losses = [-entry[token].logprob for token, entry in zip(tokens[1:], logprobs[1:], strict=True)]
        if any(not math.isfinite(nll) or nll < 0 for nll in losses):
            raise ValueError("invalid token NLL")
        values.append({"id": row["id"], "nll": math.fsum(losses),
                       "scored_tokens": len(losses), "input_sha256": input_hash(tokens)})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--role", choices=["horizon", "king", "challenger"])
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--response-fd", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--max-model-len", type=int)
    args = parser.parse_args()
    runtime = identity()
    if args.identity:
        print(json.dumps(runtime))
        return
    if args.role is None or args.snapshot is None or args.response_fd is None:
        parser.error("role, snapshot and response-fd are required")
    # Workers own a dedicated session/process group. If the evaluator dies,
    # tear down this group too; no unrelated GPU process is targeted.
    import ctypes
    import signal
    parent = os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "could not bind worker lifetime to evaluator")
    if parent == 1 or os.getppid() != parent:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    for name, expected in deterministic_env(args.role).items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"worker environment mismatch: {name}")
    from vllm import LLM, SamplingParams

    with os.fdopen(args.response_fd, "w", buffering=1) as replies:
        def send(message):
            replies.write(json.dumps(message, allow_nan=False) + "\n")
        started = time.monotonic()
        configuration = settings(args.role, args.tensor_parallel_size, args.max_model_len)
        llm = LLM(model=str(args.snapshot), **configuration)
        send({"type": "ready", "identity": runtime, "settings": configuration,
              "load_seconds": time.monotonic() - started})
        params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0,
                                detokenize=False, seed=42)
        for line in sys.stdin:
            command = json.loads(line)
            if command["type"] == "stop":
                send({"type": "stopped"})
                return
            if command["type"] != "score":
                raise ValueError("unknown scorer command")
            started = time.monotonic()
            rows = score_batch(llm, params, command["rows"], args.role, configuration)
            send({"type": "scores", "batch": command["batch"], "rows": rows,
                  "seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
