#!/usr/bin/env python3
"""Paired model-eval engine used by the multi-source evaluation service.

- protocol-v2 R2 artifact refs are verified and materialized into a local cache,
- the king model is cached across evals,
- each eval has an id, status endpoint, SSE stream, phase/progress events,
- final verdicts are written to disk as JSON audit artifacts.

It evaluates configured-chain checkpoints on local or multi-source data. Model
snapshots must be self-contained: their own config/custom code is used and
compared before any weights are loaded.
"""
from __future__ import annotations

import ast
import asyncio
import gc
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
import traceback
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import chain_config
from teutonic.evaluation import (
    PROTOCOL_VERSION,
    AttemptBusyError,
    AttemptConflictError,
    EarlyStoppingPolicy,
    EvaluationAttemptRegistry,
    EvaluationRequestV2,
    ProtocolValidationError,
    challenger_futility_decision,
    paired_bootstrap_verdict,
    provisional_paired_bootstrap,
    result_provenance,
    validate_result_v2,
)
from teutonic.evaluation.protocol_v2 import DEFAULT_EVAL_BATCH_SIZE, MAX_BATCH_SIZE
from teutonic.storage.artifacts import R2ArtifactResolver

log = logging.getLogger("teutonic.evaluator.engine")
eval_log = logging.getLogger("teutonic.evaluator.scoring")

MODEL_CACHE_DIR = Path(os.environ.get("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/pair_models"))
EVAL_RECORD_DIR = Path(os.environ.get("TEUTONIC_EVAL_RECORD_DIR", "/tmp/teutonic/pair_evals"))
COMPLETED_SAFETENSORS_SHA_FILE = Path(
    os.environ.get(
        "TEUTONIC_COMPLETED_SAFETENSORS_SHA_FILE",
        _repo_root / "completed_safetensors_sha256.txt",
    )
)
MAX_COMPLETED_EVALS_PER_SAFETENSORS_SHA = 3
SHARD_CACHE_DIR = Path(
    os.environ.get(
        "TEUTONIC_SHARD_CACHE_DIR",
        os.environ.get("TEUTONIC_PARQUET_CACHE_DIR", "/tmp/teutonic/finewebedu_shards"),
    )
)
DEFAULT_BATCH_SIZE = int(
    os.environ.get("TEUTONIC_EVAL_BATCH_SIZE", str(DEFAULT_EVAL_BATCH_SIZE))
)
if not 1 <= DEFAULT_BATCH_SIZE <= MAX_BATCH_SIZE:
    raise RuntimeError(
        f"TEUTONIC_EVAL_BATCH_SIZE must be in [1, {MAX_BATCH_SIZE}], "
        f"got {DEFAULT_BATCH_SIZE}"
    )
DEFAULT_PARALLEL_BATCH_SIZE = 1
DEFAULT_ALPHA = float(os.environ.get("EVAL_ALPHA", "0.001"))
DEFAULT_SEQ_LEN = int(os.environ.get("EVAL_SEQ_LEN", "2048"))
DEFAULT_DELTA = float(os.environ.get("EVAL_DELTA", "0.0015"))
DEFAULT_BOOTSTRAP_B = int(os.environ.get("EVAL_BOOTSTRAP_B", "10000"))
DEFAULT_N = int(os.environ.get("EVAL_N", "25000"))
SUPPORTED_ATTN_IMPLEMENTATIONS = ("eager", "flash_attention_4")
DEFAULT_ATTN_IMPLEMENTATION = os.environ.get("TEUTONIC_ATTN_IMPLEMENTATION", "eager")
if DEFAULT_ATTN_IMPLEMENTATION not in SUPPORTED_ATTN_IMPLEMENTATIONS:
    raise RuntimeError(
        "TEUTONIC_ATTN_IMPLEMENTATION must be one of "
        f"{SUPPORTED_ATTN_IMPLEMENTATIONS}, got {DEFAULT_ATTN_IMPLEMENTATION!r}"
    )
EVALUATOR_VERSION = os.environ.get("TEUTONIC_EVALUATOR_VERSION", "pair-evaluator-v2")
EVALUATION_POLICY_VERSION = os.environ.get(
    "TEUTONIC_EVALUATION_POLICY_VERSION", "paired-bootstrap-v1"
)
EVALUATOR_CODE_VERSION = os.environ.get("TEUTONIC_EVALUATOR_CODE_VERSION", "")

# Server-side caps. The validator can request a larger eval_n / n_bootstrap
# in its POST body; we clamp to these to keep per-eval wall time bounded
# while clearing a backed-up duel queue. Restore via env if not needed.
EVAL_N_CAP = int(os.environ.get("EVAL_N_CAP", "25000"))
EVAL_BOOTSTRAP_B_CAP = int(os.environ.get("EVAL_BOOTSTRAP_B_CAP", "999999"))

EVAL_MAX_RUNTIME_S = int(os.environ.get("EVAL_MAX_RUNTIME_S", "0"))
DEFAULT_LM_HEAD_CHUNK = int(os.environ.get("TEUTONIC_LM_HEAD_CHUNK", "1024"))
DEFAULT_LOG_EVERY_BATCHES = int(os.environ.get("EVAL_LOG_EVERY_BATCHES", "1"))


class SafetensorsReuseLimitError(RuntimeError):
    """A challenger checkpoint has exhausted its allowed completed evaluations."""


DEFAULT_MODEL_DEVICE_MAP = os.environ.get("TEUTONIC_MODEL_DEVICE_MAP", "auto")
DEFAULT_GPU_MEMORY_FRACTION = float(os.environ.get("TEUTONIC_GPU_MEMORY_FRACTION", "0.45"))
GPUS_PER_MODEL_INSTANCE = 2
MODEL_INSTANCES_PER_SIDE = 2
MODEL_WORKER_PROCESSES = MODEL_INSTANCES_PER_SIDE * 2
KERNEL_CACHE_DIR = Path(
    os.environ.get("TEUTONIC_KERNEL_CACHE_DIR", "/tmp/teutonic/kernel_cache")
)
MODEL_LOADER_VERSION = "direct-gpu-v1"
CACHE_HIGH_WATERMARK_GB = float(os.environ.get("MODEL_CACHE_HIGH_WATERMARK_GB", "500"))

_eval_lock = threading.Lock()
_attempts = EvaluationAttemptRegistry()
_gpu_ids: list[int] = []
_king_model = None
_king_key: tuple[str, ...] | None = None
_king_device = ""
_king_gpu_ids: list[int] = []
_attention_preflight_cache: dict[str, dict] = {}
_model_worker_pool = None


class EvalRequest(BaseModel):
    king_repo: str
    challenger_repo: str
    king_digest: str = ""
    challenger_digest: str = ""
    revision: str | None = None
    block_hash: str = ""
    hotkey: str = ""
    coldkey: str = ""
    dataset_source: Literal["pretokenized_npy"] = "pretokenized_npy"
    dataset_sources: list[dict[str, Any]] = Field(default_factory=list)
    seq_len: int = Field(default=DEFAULT_SEQ_LEN, ge=2)
    vocab_size: int = 0
    attn_implementation: Literal["eager", "flash_attention_4"] = DEFAULT_ATTN_IMPLEMENTATION
    n: int = DEFAULT_N
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=MAX_BATCH_SIZE)
    alpha: float = DEFAULT_ALPHA
    delta_threshold: float = DEFAULT_DELTA
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B
    seed: int = 0xE1A
    bootstrap_seed: int = 0xB007
    early_stop_enabled: bool = False
    early_stop_min_fraction: float = 0.4
    early_stop_advantage_quantile: float = 0.95
    early_stop_margin: float = 0.0
    early_stop_check_interval: int = 100
    lm_head_chunk: int = DEFAULT_LM_HEAD_CHUNK
    log_every_batches: int = DEFAULT_LOG_EVERY_BATCHES
    model_device_map: str = DEFAULT_MODEL_DEVICE_MAP
    gpu_memory_fraction: float = DEFAULT_GPU_MEMORY_FRACTION
    parallel_models: Literal[True] = True
    parallel_batch_size: Literal[1] = DEFAULT_PARALLEL_BATCH_SIZE


def internal_request_from_v2(
    request: EvaluationRequestV2,
    king_snapshot: str,
    challenger_snapshot: str,
) -> EvalRequest:
    """Adapt protocol v2 to the unchanged scoring engine's internal request."""
    return EvalRequest(
        king_repo=king_snapshot,
        challenger_repo=challenger_snapshot,
        hotkey=str(request.miner["hotkey"]),
        coldkey=str(request.miner["coldkey"]),
        dataset_source=str(request.dataset["source"]),
        dataset_sources=list(request.dataset["sources"]),
        n=int(request.limits["n"]),
        seq_len=int(request.limits["seq_len"]),
        n_bootstrap=int(request.limits["n_bootstrap"]),
        alpha=float(request.limits["alpha"]),
        delta_threshold=float(request.limits["delta_threshold"]),
        batch_size=int(request.limits["batch_size"]),
        seed=int(request.sampling["seed"]),
        bootstrap_seed=int(request.sampling["bootstrap_seed"]),
        block_hash=str(request.sampling["block_hash"]),
        early_stop_enabled=bool(request.early_stopping["enabled"]),
        early_stop_min_fraction=float(request.early_stopping["min_fraction"]),
        early_stop_advantage_quantile=float(
            request.early_stopping["advantage_quantile"]
        ),
        early_stop_margin=float(request.early_stopping["margin"]),
        early_stop_check_interval=int(request.early_stopping["check_interval"]),
    )


def validate_protocol_versions(request: EvaluationRequestV2) -> None:
    expected = {
        "evaluator": EVALUATOR_VERSION,
        "evaluation_policy": EVALUATION_POLICY_VERSION,
    }
    if EVALUATOR_CODE_VERSION:
        expected["code"] = EVALUATOR_CODE_VERSION
    for field, actual in expected.items():
        if request.versions[field] != actual:
            raise ProtocolValidationError(
                f"versions.{field} must match deployed version {actual!r}"
            )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_gpu_ids(value: str | None = None) -> list[int]:
    value = value if value is not None else os.environ.get("EVAL_GPUS", "auto")
    if value == "auto":
        return list(range(torch.cuda.device_count()))
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def device_plan_for_gpus(gpu_ids: list[int]) -> str:
    if not torch.cuda.is_available() or not gpu_ids:
        return "cpu"
    if len(gpu_ids) == 1:
        return f"cuda:{gpu_ids[0]}"
    return "auto"


def model_worker_specs(gpu_ids: list[int]) -> list[dict]:
    """Return the fixed two-GPU, two-instance topology for each duel side."""
    if len(gpu_ids) != 8:
        raise RuntimeError(f"MiMo duel requires exactly 8 GPUs, got {gpu_ids}")
    return [
        {"worker_id": "king-0", "role": "king", "gpu_ids": gpu_ids[0:2]},
        {"worker_id": "king-1", "role": "king", "gpu_ids": gpu_ids[2:4]},
        {"worker_id": "challenger-0", "role": "challenger", "gpu_ids": gpu_ids[4:6]},
        {"worker_id": "challenger-1", "role": "challenger", "gpu_ids": gpu_ids[6:8]},
    ]


def normalize_model_ref(ref: str) -> str:
    path = Path((ref or "").strip())
    if not path.exists():
        raise FileNotFoundError("protocol-v2 evaluator accepts only materialized R2 snapshots")
    return str(path.resolve())


def dataset_seed_material(req: EvalRequest) -> str:
    block_hash = (req.block_hash or "").strip()
    hotkey = (req.hotkey or "").strip()
    if block_hash and block_hash != "default":
        return f"block_hash={block_hash}|hotkey={hotkey}"
    raise ValueError("pretokenized evaluation requires a finalized block hash")


def dataset_seed(req: EvalRequest) -> int:
    digest = hashlib.blake2b(dataset_seed_material(req).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def apply_eval_limits(req: EvalRequest, eval_id: str = "") -> dict:
    requested_n = max(0, int(req.n))
    requested_bootstrap = max(0, int(req.n_bootstrap))

    req.n = min(requested_n, EVAL_N_CAP)
    req.n_bootstrap = min(requested_bootstrap, EVAL_BOOTSTRAP_B_CAP)

    capped = req.n != requested_n or req.n_bootstrap != requested_bootstrap
    if capped:
        log.info(
            "eval %s: capped n %d->%d n_bootstrap %d->%d",
            eval_id,
            requested_n,
            req.n,
            requested_bootstrap,
            req.n_bootstrap,
        )
    return {
        "requested_n": requested_n,
        "effective_n": req.n,
        "n_cap": EVAL_N_CAP,
        "requested_n_bootstrap": requested_bootstrap,
        "effective_n_bootstrap": req.n_bootstrap,
        "n_bootstrap_cap": EVAL_BOOTSTRAP_B_CAP,
        "capped": capped,
    }


def check_eval_runtime(t0: float) -> None:
    if EVAL_MAX_RUNTIME_S > 0 and time.time() - t0 > EVAL_MAX_RUNTIME_S:
        raise TimeoutError(f"eval exceeded EVAL_MAX_RUNTIME_S={EVAL_MAX_RUNTIME_S}s")


def custom_code_files_from_config(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return []
    auto_map = config.get("auto_map") or {}
    files = []
    for value in auto_map.values():
        refs = value if isinstance(value, list) else [value]
        for ref in refs:
            if not isinstance(ref, str):
                continue
            module = ref.split("--")[-1].split(".")[0]
            if module:
                files.append(f"{module}.py")
    return sorted(set(files))


def snapshot_has_required_files(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    if not any(path.glob("*.safetensors")):
        return False
    return all((path / filename).exists() for filename in custom_code_files_from_config(path / "config.json"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def safetensors_digest_from_file_digests(file_digests: dict[str, str]) -> str:
    if not file_digests:
        raise FileNotFoundError("no .safetensors file digests found")
    h = hashlib.sha256()
    for name, digest in sorted(file_digests.items()):
        digest = digest.lower().removeprefix("sha256:")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{name}: invalid SHA-256 digest {digest!r}")
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(bytes.fromhex(digest))
    return h.hexdigest()


def snapshot_safetensors_digest(snapshot_dir: str) -> str:
    path = Path(snapshot_dir)
    shard_names = snapshot_safetensor_names(snapshot_dir)
    if not shard_names:
        raise FileNotFoundError(f"no .safetensors files found in {snapshot_dir}")
    workers = min(len(shard_names), max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        digests = dict(
            zip(
                shard_names,
                executor.map(sha256_file, (path / name for name in shard_names)),
                strict=True,
            )
        )
    return safetensors_digest_from_file_digests(digests)


def reject_duplicate_safetensors(king_snapshot: str, challenger_snapshot: str, on_phase=None) -> dict:
    snapshots = {
        "king": Path(king_snapshot),
        "challenger": Path(challenger_snapshot),
    }
    shard_names = {
        role: snapshot_safetensor_names(str(path)) for role, path in snapshots.items()
    }
    for role, names in shard_names.items():
        if not names:
            raise FileNotFoundError(f"no .safetensors files found in {snapshots[role]}")
    work = [
        (role, name, snapshots[role] / name)
        for role in ("king", "challenger")
        for name in shard_names[role]
    ]
    workers = min(len(work), max(1, os.cpu_count() or 1))
    if on_phase:
        on_phase({
            "phase": "duplicate_check_start",
            "king_shards": len(shard_names["king"]),
            "challenger_shards": len(shard_names["challenger"]),
            "hash_workers": workers,
        })
    started = time.monotonic()
    file_digests: dict[str, dict[str, str]] = {"king": {}, "challenger": {}}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(sha256_file, path): (role, name)
            for role, name, path in work
        }
        for future, (role, name) in futures.items():
            file_digests[role][name] = future.result()
    king_digest = safetensors_digest_from_file_digests(file_digests["king"])
    challenger_digest = safetensors_digest_from_file_digests(file_digests["challenger"])
    elapsed = time.monotonic() - started
    log.info(
        "duplicate check complete | king_shards=%d challenger_shards=%d workers=%d elapsed_s=%.1f",
        len(shard_names["king"]),
        len(shard_names["challenger"]),
        workers,
        elapsed,
    )
    if king_digest == challenger_digest:
        raise RuntimeError("challenger .safetensors are identical to the king")
    meta = {
        "king_safetensors_sha256": king_digest,
        "challenger_safetensors_sha256": challenger_digest,
    }
    if on_phase:
        on_phase({
            "phase": "duplicate_check_done",
            "hash_workers": workers,
            "elapsed_seconds": round(elapsed, 1),
            **{k: v[:16] for k, v in meta.items()},
        })
    return meta


def completed_safetensors_sha_uses(digest: str) -> int:
    if not COMPLETED_SAFETENSORS_SHA_FILE.exists():
        return 0
    with COMPLETED_SAFETENSORS_SHA_FILE.open() as f:
        return sum(line.strip() == digest for line in f)


def reject_reused_safetensors(digest: str, on_phase=None) -> dict:
    uses = completed_safetensors_sha_uses(digest)
    if uses >= MAX_COMPLETED_EVALS_PER_SAFETENSORS_SHA:
        raise SafetensorsReuseLimitError(
            f"challenger safetensors SHA-256 {digest} has already completed {uses} evals; "
            f"maximum allowed is {MAX_COMPLETED_EVALS_PER_SAFETENSORS_SHA}"
        )
    meta = {
        "challenger_safetensors_prior_completed_evals": uses,
        "challenger_safetensors_max_completed_evals": MAX_COMPLETED_EVALS_PER_SAFETENSORS_SHA,
    }
    if on_phase:
        on_phase({"phase": "safetensors_reuse_check_done", **meta})
    return meta


def record_completed_safetensors_sha(digest: str) -> None:
    COMPLETED_SAFETENSORS_SHA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with COMPLETED_SAFETENSORS_SHA_FILE.open("a") as f:
        f.write(f"{digest}\n")


def materialize_model(repo_or_url: str, digest: str = "", on_phase=None) -> str:
    """Accept only a complete local snapshot materialized by the R2 resolver."""
    del digest, on_phase
    path = Path(normalize_model_ref(repo_or_url))
    if not snapshot_has_required_files(path):
        raise RuntimeError(f"materialized R2 snapshot is incomplete: {path}")
    return str(path)


CONFIG_MATCH_KEYS = (
    "model_type",
    "architectures",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rope_theta",
    "tie_word_embeddings",
    "attention_bias",
)


def resolved_attention_types(config) -> list[str]:
    """Resolve MiMo's trained per-layer hybrid attention schedule in order."""
    n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    pattern = list(getattr(config, "hybrid_layer_pattern", None) or [])
    declared = list(getattr(config, "layer_types", None) or [])
    if len(pattern) != n_layers:
        raise RuntimeError(
            f"hybrid_layer_pattern has {len(pattern)} entries for {n_layers} surviving layers"
        )
    invalid = [(idx, value) for idx, value in enumerate(pattern) if value not in (0, 1)]
    if invalid:
        raise RuntimeError(f"invalid hybrid_layer_pattern entries: {invalid[:8]}")
    resolved = [
        "sliding_window_attention" if value == 1 else "full_attention"
        for value in pattern
    ]
    if declared:
        if len(declared) != n_layers:
            raise RuntimeError(
                f"layer_types has {len(declared)} entries for {n_layers} surviving layers"
            )
        normalized_declared = [
            "sliding_window_attention" if value == "sliding_attention" else value
            for value in declared
        ]
        if normalized_declared != resolved:
            mismatches = [
                {"layer": idx, "declared": got, "resolved": want}
                for idx, (got, want) in enumerate(zip(normalized_declared, resolved))
                if got != want
            ]
            raise RuntimeError(
                f"layer_types do not match hybrid_layer_pattern: {mismatches[:8]}"
            )
    if int(getattr(config, "sliding_window", 0) or 0) != 128:
        raise RuntimeError(f"MiMo sliding_window must be 128, got {config.sliding_window!r}")
    if int(getattr(config, "sliding_window_size", 0) or 0) != 128:
        raise RuntimeError(
            f"MiMo sliding_window_size must be 128, got {config.sliding_window_size!r}"
        )
    if not bool(getattr(config, "add_swa_attention_sink_bias", False)):
        raise RuntimeError("MiMo SWA learned attention sink bias must be enabled")
    implementation = str(getattr(config, "_attn_implementation", ""))
    if implementation not in SUPPORTED_ATTN_IMPLEMENTATIONS:
        raise RuntimeError(
            f"MiMo evaluation does not support attention implementation {implementation!r}"
        )
    return resolved


def validate_and_report_attention_config(config, label: str, on_phase=None) -> dict:
    if getattr(config, "model_type", "") != "mimo_v2":
        return {}
    cache_material = json.dumps(
        {
            "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "hybrid_layer_pattern": config.hybrid_layer_pattern,
            "layer_types": resolved_attention_types(config),
            "sliding_window": config.sliding_window,
            "sliding_window_size": config.sliding_window_size,
            "add_swa_attention_sink_bias": config.add_swa_attention_sink_bias,
            "attn_implementation": config._attn_implementation,
            "qk_head_dim": getattr(config, "head_dim", None),
            "v_head_dim": getattr(config, "v_head_dim", getattr(config, "head_dim", None)),
        },
        sort_keys=True,
    ).encode()
    cache_key = hashlib.sha256(cache_material).hexdigest()
    if cache_key in _attention_preflight_cache:
        cached = dict(_attention_preflight_cache[cache_key])
        cached["label"] = label
        cached["preflight_cached"] = True
        return cached
    attention_types = resolved_attention_types(config)
    qk_head_dim = int(getattr(config, "head_dim", 0) or 0)
    v_head_dim = int(getattr(config, "v_head_dim", qk_head_dim) or 0)
    implementation = str(config._attn_implementation)
    report = {
        "label": label,
        "n_layers": len(attention_types),
        "attention_types": attention_types,
        "sliding_window": int(config.sliding_window),
        "learned_swa_sink_bias": True,
        "attn_implementation": implementation,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "fa4_native_asymmetric_value_dim": (
            implementation == "flash_attention_4" and qk_head_dim != v_head_dim
        ),
        "preflight_cache_key": cache_key,
        "preflight_cached": False,
    }
    _attention_preflight_cache[cache_key] = dict(report)
    log.info("%s resolved MiMo attention config: %s", label, json.dumps(report, sort_keys=True))
    if on_phase:
        on_phase({"phase": f"{label}_attention_config_validated", **report})
    return report


def snapshot_safetensor_names(snapshot_dir: str) -> list[str]:
    path = Path(snapshot_dir)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return sorted(set(index.get("weight_map", {}).values()))
    return sorted(p.name for p in path.glob("*.safetensors"))


def snapshot_safetensor_keys(snapshot_dir: str) -> list[str]:
    """Read checkpoint tensor names without materializing any tensor payloads."""
    path = Path(snapshot_dir)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return sorted(index.get("weight_map", {}))

    from safetensors import safe_open

    keys: list[str] = []
    for shard_name in snapshot_safetensor_names(snapshot_dir):
        with safe_open(str(path / shard_name), framework="pt", device="cpu") as shard:
            keys.extend(shard.keys())
    return sorted(keys)


def reject_mtp_checkpoint_weights(snapshot_dir: str) -> None:
    mtp_keys = [name for name in snapshot_safetensor_keys(snapshot_dir) if "mtp" in name.lower()]
    if mtp_keys:
        raise RuntimeError(
            f"MTP/speculative weights are not allowed in scoring model: {mtp_keys[:8]}"
        )


def checkpoint_load_key(snapshot_dir: str, req: EvalRequest, gpu_ids: list[int]) -> str:
    """Identify immutable weights/runtime state; sampled sequences are deliberately absent."""
    path = Path(snapshot_dir).resolve()
    files = []
    identity_names = ["config.json", "model.safetensors.index.json"]
    identity_names.extend(snapshot_safetensor_names(snapshot_dir))
    for name in identity_names:
        candidate = path / name
        if candidate.exists():
            stat = candidate.stat()
            files.append((name, stat.st_size, stat.st_mtime_ns))
    material = {
        "loader": MODEL_LOADER_VERSION,
        "snapshot": str(path),
        "metadata": files,
        "gpu_ids": list(gpu_ids),
        "dtype": "bfloat16",
        "attn_implementation": req.attn_implementation,
        "model_device_map": req.model_device_map,
        "revision": req.revision,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def kernel_cache_identity(config, gpu_ids: list[int]) -> str:
    """Key reusable CUDA/Triton artifacts by architecture, never by checkpoint weights."""
    config_dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    shape_keys = (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "num_experts_per_tok",
        "num_hidden_layers",
        "vocab_size",
    )
    capabilities = []
    if torch.cuda.is_available():
        capabilities = [list(torch.cuda.get_device_capability(gpu_id)) for gpu_id in gpu_ids]
    material = {
        "shapes": {key: config_dict.get(key) for key in shape_keys},
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capabilities": capabilities,
        "grouped_moe": "torch_foreach_mm_v1",
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:24]


def configure_kernel_cache(config, gpu_ids: list[int]) -> dict:
    cache_key = kernel_cache_identity(config, gpu_ids)
    cache_dir = KERNEL_CACHE_DIR / cache_key
    triton_dir = cache_dir / "triton"
    inductor_dir = cache_dir / "torchinductor"
    cuda_dir = cache_dir / "cuda"
    for directory in (triton_dir, inductor_dir, cuda_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # Set before the first grouped-kernel launch in this persistent worker.
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["CUDA_CACHE_PATH"] = str(cuda_dir)
    return {
        "key": cache_key,
        "path": str(cache_dir),
        "scope": "architecture",
        "model_graph_compiled": False,
    }


def snapshot_meta(snapshot_dir: str) -> dict:
    path = Path(snapshot_dir)
    return {
        "path": str(path),
        "has_config": (path / "config.json").exists(),
        "python_files": sorted(p.name for p in path.glob("*.py")),
        "safetensors": snapshot_safetensor_names(snapshot_dir),
    }


def ensure_snapshot_on_path(snapshot_dir: str) -> None:
    path = str(Path(snapshot_dir).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def split_auto_map_ref(ref: str) -> tuple[str, str] | None:
    if not isinstance(ref, str) or "." not in ref:
        return None
    ref = ref.split("--")[-1]
    module_name, class_name = ref.rsplit(".", 1)
    return module_name, class_name


def config_class_from_local_auto_map(snapshot_dir: str, config_dict: dict):
    auto_map = config_dict.get("auto_map") or {}
    parsed = split_auto_map_ref(auto_map.get("AutoConfig", ""))
    if parsed is None:
        return None, None
    module_name, wanted_class_name = parsed
    module_path = Path(snapshot_dir) / f"{module_name}.py"
    if not module_path.exists():
        return None, None

    module_cache_name = f"_teutonic_{Path(snapshot_dir).name}_{module_name}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_cache_name, module_path)
    if spec is None or spec.loader is None:
        return None, None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_cache_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, wanted_class_name):
        return getattr(module, wanted_class_name), module_name

    candidates = [
        obj
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and name.endswith("Config") and obj.__module__ == module.__name__
    ]
    if len(candidates) == 1:
        log.warning(
            "auto_map points to missing %s.%s; using local config class %s.%s",
            module_name,
            wanted_class_name,
            module_name,
            candidates[0].__name__,
        )
        return candidates[0], module_name
    return None, None


def load_local_config_from_snapshot(snapshot_dir: str):
    config_path = Path(snapshot_dir) / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_class, module_name = config_class_from_local_auto_map(snapshot_dir, config_dict)
    if config_class is None:
        raise RuntimeError("could not resolve local AutoConfig class from auto_map")
    config = config_class(**config_dict)
    auto_map = dict(getattr(config, "auto_map", None) or config_dict.get("auto_map") or {})
    auto_map["AutoConfig"] = f"{module_name}.{config_class.__name__}"
    config.auto_map = auto_map
    return config


def load_model_config(snapshot_dir: str, req: EvalRequest, label: str, on_phase=None):
    from transformers import AutoConfig

    meta = snapshot_meta(snapshot_dir)
    if not meta["safetensors"]:
        raise FileNotFoundError(f"{label} snapshot has no .safetensors files: {snapshot_dir}")

    ensure_snapshot_on_path(snapshot_dir)
    if on_phase:
        on_phase({"phase": f"{label}_config_load_start", "snapshot": snapshot_dir})
    try:
        if not meta["has_config"]:
            raise FileNotFoundError(f"{snapshot_dir}/config.json is missing")
        try:
            config = AutoConfig.from_pretrained(snapshot_dir, revision=req.revision, trust_remote_code=True)
            source = "snapshot"
        except AttributeError as exc:
            config = load_local_config_from_snapshot(snapshot_dir)
            source = "snapshot_local_auto_map_compat"
            log.warning("%s AutoConfig dynamic load failed; loaded config from local code: %s", label, exc)
    except Exception as exc:
        raise RuntimeError(
            f"{label} snapshot is not self-contained enough to load its config/custom code: {exc}"
        ) from exc
    config.use_cache = False
    config._attn_implementation = req.attn_implementation
    if on_phase:
        on_phase({
            "phase": f"{label}_config_load_done",
            "source": source,
            "model_type": getattr(config, "model_type", ""),
            "attn_implementation": req.attn_implementation,
        })
    return config, {"source": source, **meta}


def config_value(config, key: str):
    if hasattr(config, key):
        return getattr(config, key)
    data = config.to_dict()
    return data.get(key)


def compare_model_configs(king_config, challenger_config) -> list[dict]:
    mismatches = []
    for key in CONFIG_MATCH_KEYS:
        king_value = config_value(king_config, key)
        challenger_value = config_value(challenger_config, key)
        if king_value is None or challenger_value is None:
            continue
        if king_value != challenger_value:
            mismatches.append({"key": key, "king": king_value, "challenger": challenger_value})
    return mismatches


def grouped_mimo_moe(
    module,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Run MiMo experts with three grouped CUDA GEMMs instead of 64 Python loops."""
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        raise RuntimeError("grouped MiMo MoE requires CUDA bfloat16 activations")
    n_tokens = hidden_states.shape[0]
    top_k = topk_indices.shape[1]
    n_experts = len(module.experts)

    flat_experts = topk_indices.reshape(-1)
    flat_tokens = torch.arange(n_tokens, device=hidden_states.device).repeat_interleave(top_k)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    sorted_tokens = flat_tokens[order]
    sorted_weights = topk_weights.reshape(-1)[order]

    counts = torch.bincount(sorted_experts, minlength=n_experts)
    active_experts = torch.nonzero(counts, as_tuple=False).flatten()
    split_sizes = counts[active_experts].tolist()
    routed_inputs = list(hidden_states[sorted_tokens].split(split_sizes, dim=0))
    experts = [module.experts[index] for index in active_experts.tolist()]

    gate_outputs = torch._foreach_mm(
        routed_inputs,
        [expert.gate_proj.weight.T for expert in experts],
    )
    up_outputs = torch._foreach_mm(
        routed_inputs,
        [expert.up_proj.weight.T for expert in experts],
    )
    activated = [module.experts[0].act_fn(gate) * up for gate, up in zip(gate_outputs, up_outputs)]
    down_outputs = torch._foreach_mm(
        activated,
        [expert.down_proj.weight.T for expert in experts],
    )

    routed_outputs = torch.cat(list(down_outputs), dim=0)
    routed_outputs = routed_outputs * sorted_weights.unsqueeze(-1)
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    final_hidden_states.index_add_(0, sorted_tokens, routed_outputs)
    return final_hidden_states.type(hidden_states.dtype)


def enable_grouped_mimo_moe(model) -> int:
    """Patch checkpoint MiMo MoE modules after strict state loading."""
    if not hasattr(torch, "_foreach_mm"):
        raise RuntimeError("this PyTorch build does not provide native grouped CUDA GEMM")
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "MiMoV2MoE":
            continue
        module.moe = types.MethodType(grouped_mimo_moe, module)
        patched += 1
    if patched == 0 and getattr(model.config, "model_type", "") == "mimo_v2":
        raise RuntimeError("MiMo checkpoint contains no patchable MiMoV2MoE modules")
    log.info("enabled grouped CUDA MoE for %d layers", patched)
    return patched


def load_safetensors_state_dict(model_dir: str) -> dict:
    from safetensors.torch import load_file

    path = Path(model_dir)
    shard_names = snapshot_safetensor_names(model_dir)
    if not shard_names:
        raise FileNotFoundError(f"no .safetensors files found in {model_dir}")

    results: dict[str, dict] = {}
    lock = threading.Lock()

    def _load(shard_name: str) -> None:
        log.info("loading shard %s", shard_name)
        shard_state = load_file(str(path / shard_name), device="cpu")
        with lock:
            results[shard_name] = shard_state

    with ThreadPoolExecutor(max_workers=len(shard_names)) as pool:
        list(pool.map(_load, shard_names))

    state: dict = {}
    for shard_name in shard_names:
        state.update(results[shard_name])
    mtp_keys = sorted(name for name in state if "mtp" in name.lower())
    if mtp_keys:
        raise RuntimeError(
            f"MTP/speculative weights are not allowed in scoring model: {mtp_keys[:8]}"
        )
    return state


def gpu_max_memory(fraction: float, gpu_ids: list[int] | None = None) -> dict:
    ids = gpu_ids if gpu_ids is not None else _gpu_ids
    if not torch.cuda.is_available() or not ids:
        return {}
    fraction = min(max(float(fraction), 0.05), 0.95)
    max_memory = {}
    for gpu_id in ids:
        props = torch.cuda.get_device_properties(gpu_id)
        gib = max(1, int((props.total_memory / (1024**3)) * fraction))
        max_memory[gpu_id] = f"{gib}GiB"
    return max_memory


def dispatch_model_across_gpus(model, req: EvalRequest, label: str, gpu_ids: list[int] | None = None, on_phase=None):
    effective_ids = gpu_ids if gpu_ids is not None else _gpu_ids
    try:
        from accelerate import dispatch_model, infer_auto_device_map
    except Exception as exc:
        raise RuntimeError(
            "model_device_map='auto' requires accelerate. Install it in the eval environment with "
            "`pip install accelerate`, or set model_device_map='single'."
        ) from exc

    no_split = list(getattr(model, "_no_split_modules", None) or [])
    max_memory = gpu_max_memory(req.gpu_memory_fraction, effective_ids)
    device_map = balanced_transformer_device_map(model, effective_ids)
    if device_map:
        if on_phase:
            used = sorted({str(device) for device in device_map.values()})
            on_phase({"phase": f"{label}_device_map_balanced", "devices": used})
        model = dispatch_model(model, device_map=device_map)
        log.info("%s dispatched with balanced device_map devices=%s", label, sorted({str(device) for device in device_map.values()}))
        return model

    if on_phase:
        on_phase({
            "phase": f"{label}_device_map_infer_start",
            "gpus": effective_ids,
            "max_memory": max_memory,
            "no_split": no_split,
        })
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory or None,
        no_split_module_classes=no_split,
        dtype=torch.bfloat16,
    )
    if on_phase:
        used = sorted({str(device) for device in device_map.values()})
        on_phase({"phase": f"{label}_dispatch_start", "devices": used})
    model = dispatch_model(model, device_map=device_map)
    log.info("%s dispatched with device_map devices=%s", label, sorted({str(device) for device in device_map.values()}))
    return model


def balanced_transformer_device_map(model, gpu_ids: list[int] | None = None) -> dict:
    ids = gpu_ids if gpu_ids is not None else _gpu_ids
    if not torch.cuda.is_available() or len(ids) < 2:
        return {}
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if core is None or layers is None:
        return {}

    n_layers = len(layers)
    if n_layers <= 0:
        return {}

    device_map = {}
    first = ids[0]
    last = ids[-1]

    for name, _module in model.named_children():
        if name != "model":
            device_map[name] = last if name == "lm_head" else first

    for name, _module in core.named_children():
        full_name = f"model.{name}"
        if name == "layers":
            continue
        # rotary_emb co-located with embed_tokens: avoids cross-GPU inv_freq transfer
        # when computing position embeddings at the start of each forward pass.
        if name in ("embed_tokens", "wte", "rotary_emb"):
            device_map[full_name] = first
        else:
            device_map[full_name] = last  # norm, etc.

    # TP remains 1: each layer lives wholly on one GPU. Distribute layers across
    # the GPUs in the model's assigned group; no cross-rank reductions.
    n_layer_gpus = len(ids)
    for layer_idx in range(n_layers):
        gpu_idx = min(n_layer_gpus - 1, (layer_idx * n_layer_gpus) // n_layers)
        device_map[f"model.layers.{layer_idx}"] = ids[gpu_idx]

    return device_map


def model_input_device(model) -> torch.device:
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed is not None:
        try:
            return next(embed.parameters()).device
        except StopIteration:
            pass
    return next(model.parameters()).device


def patch_mimo_masking_compat(model) -> tuple[str, ...]:
    """Adapt immutable MiMo code to the installed Transformers mask API."""
    if getattr(model.config, "model_type", "") != "mimo_v2":
        return ()
    module = sys.modules.get(model.__class__.__module__)
    if module is None:
        raise RuntimeError("could not resolve the loaded MiMo model module")

    patched: list[str] = []
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        function = getattr(module, name, None)
        if function is None:
            raise RuntimeError(f"MiMo model module does not expose {name}")
        if getattr(function, "_teutonic_cache_position_compat", False):
            continue
        if "cache_position" in inspect.signature(function).parameters:
            continue

        def adapter(*args, _function=function, **kwargs):
            kwargs.pop("cache_position", None)
            return _function(*args, **kwargs)

        adapter.__name__ = getattr(function, "__name__", name)
        adapter.__doc__ = getattr(function, "__doc__", None)
        adapter._teutonic_cache_position_compat = True
        setattr(module, name, adapter)
        patched.append(name)
    return tuple(patched)


def load_eval_model(snapshot_dir: str, config, device: str, label: str, req: EvalRequest, gpu_ids: list[int] | None = None, on_phase=None):
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from accelerate.utils import modeling as accelerate_modeling
    from transformers import AutoModelForCausalLM
    from transformers.initialization import no_init_weights

    if on_phase:
        on_phase({"phase": f"{label}_load_start", "device": device, "snapshot": snapshot_dir})
    t0 = time.time()
    requested_attn_implementation = req.attn_implementation
    config.use_cache = False
    # The remote MiMo class has a stale FA4 capability flag. Construct the
    # module under eager, then select the requested backend for every forward.
    config._attn_implementation = "eager"
    dtype = torch.bfloat16
    effective_ids = list(gpu_ids if gpu_ids is not None else _gpu_ids)
    reject_mtp_checkpoint_weights(snapshot_dir)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        if on_phase:
            on_phase({"phase": f"{label}_init_start", "dtype": str(dtype)})
        # Every parameter is populated by the strict checkpoint load below. Skip
        # random initialization, which is prohibitively expensive for 104B models.
        with init_empty_weights(), no_init_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    finally:
        torch.set_default_dtype(old_dtype)
    mask_compat = patch_mimo_masking_compat(model)
    if mask_compat and on_phase:
        on_phase(
            {
                "phase": f"{label}_masking_compat_enabled",
                "functions": list(mask_compat),
            }
        )
    # Empty-weight construction cannot preserve aliases created by parameter
    # assignment. Re-establish any checkpoint-declared embedding/head ties before
    # Accelerate resolves tied tensors and streams the shards.
    model.tie_weights()
    if on_phase:
        on_phase({"phase": f"{label}_direct_gpu_load_start", "dtype": str(dtype)})
    if device == "auto":
        device_map = balanced_transformer_device_map(model, effective_ids)
        if not device_map:
            raise RuntimeError(f"could not construct a fixed device map for {label} on {effective_ids}")
    else:
        device_map = {"": device}
    no_split = list(getattr(model, "_no_split_modules", None) or [])
    # Accelerate redraws a tqdm bar for every tensor in a multi-device load.
    # PM2 stores each redraw; suppress only that bar, preserving warnings and
    # our phase/progress events. Each model worker loads in its own process.
    checkpoint_progress = accelerate_modeling.tqdm
    try:
        accelerate_modeling.tqdm = partial(checkpoint_progress, disable=True)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=snapshot_dir,
            device_map=device_map,
            no_split_module_classes=no_split,
            dtype=dtype,
            offload_state_dict=False,
            force_hooks=len(set(device_map.values())) > 1,
            strict=True,
        )
    finally:
        accelerate_modeling.tqdm = checkpoint_progress
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"{label} has parameters left on meta after checkpoint load: {meta_parameters[:8]}")
    if on_phase:
        on_phase({
            "phase": f"{label}_direct_gpu_load_done",
            "devices": sorted({str(value) for value in device_map.values()}),
        })
    if torch.cuda.is_available():
        for gpu_id in effective_ids:
            torch.cuda.synchronize(gpu_id)
    if getattr(config, "model_type", "") == "mimo_v2":
        enable_grouped_mimo_moe(model)
    model.eval()
    model.config.use_cache = False
    model.config._attn_implementation = requested_attn_implementation
    params = sum(p.numel() for p in model.parameters()) / 1e9
    if on_phase:
        on_phase({"phase": f"{label}_load_done", "params_b": round(params, 3), "elapsed_s": round(time.time() - t0, 1)})
    return model


def load_model_replicas(
    snapshot_dir: str,
    config,
    label: str,
    req: EvalRequest,
    gpu_ids: list[int],
    on_phase=None,
) -> list:
    if len(gpu_ids) != 4:
        raise RuntimeError(f"{label} requires four replica GPUs, got {gpu_ids}")
    replicas = []
    for replica_idx, gpu_id in enumerate(gpu_ids):
        replicas.append(
            load_eval_model(
                snapshot_dir,
                config,
                f"cuda:{gpu_id}",
                f"{label}_replica_{replica_idx}",
                req,
                gpu_ids=[gpu_id],
                on_phase=on_phase,
            )
        )
    return replicas


def is_truncated_npy_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and "mmap length is greater than file size" in str(exc)


def read_npy_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        raw = f.read(1024)
    buf = io.BytesIO(raw)
    magic = buf.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError(f"{path} is not a .npy file")
    version = tuple(buf.read(2))
    if version[0] == 1:
        header_len = int.from_bytes(buf.read(2), "little")
    else:
        header_len = int.from_bytes(buf.read(4), "little")
    header = ast.literal_eval(buf.read(header_len).decode("latin1").strip())
    return buf.tell(), header


def shuffled_indices(rng: np.random.Generator, size: int, limit: int | None = None) -> np.ndarray:
    if limit is None or limit >= size:
        indices = np.arange(size)
        rng.shuffle(indices)
        return indices
    return rng.choice(size, size=limit, replace=False)


def load_indexed_sequences_from_npy_shard(
    path: str,
    req: EvalRequest,
    rng: np.random.Generator,
    limit: int | None = None,
) -> list[tuple[int, list[int]]]:
    """Load randomized sequences together with their shard-relative index."""
    data_offset, header = read_npy_header(path)
    dtype = np.dtype(header["descr"])
    shape = tuple(header["shape"])
    if dtype != np.dtype("<u4") and dtype != np.dtype("uint32"):
        raise ValueError(f"{path} dtype must be uint32/<u4, got {dtype}")
    if not shape:
        raise ValueError(f"{path} has invalid shape {shape}")

    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        if arr.shape[1] != req.seq_len:
            raise ValueError(
                f"{path} sequence width {arr.shape[1]} != seq_len={req.seq_len}; "
                "refusing to truncate or pad evaluation sequences"
            )
        indices = shuffled_indices(rng, arr.shape[0], limit)
        return [
            (int(i), arr[int(i)].astype(np.int64, copy=False).tolist())
            for i in indices
        ]

    if arr.ndim != 1:
        raise ValueError(f"{path} expected 1D token stream or 2D sequence matrix, got shape={arr.shape}")
    n_sequences = arr.shape[0] // req.seq_len
    if n_sequences <= 0:
        return []
    indices = shuffled_indices(rng, n_sequences, limit)
    out = []
    for i in indices:
        sequence_index = int(i)
        start = sequence_index * req.seq_len
        out.append(
            (
                sequence_index,
                arr[start : start + req.seq_len]
                .astype(np.int64, copy=False)
                .tolist(),
            )
        )
    _ = data_offset
    return out


def load_sequences_from_npy_shard(
    path: str,
    req: EvalRequest,
    rng: np.random.Generator,
    limit: int | None = None,
) -> list[list[int]]:
    """Backward-compatible token-only wrapper around the indexed loader."""
    return [
        sequence
        for _sequence_index, sequence in load_indexed_sequences_from_npy_shard(
            path,
            req,
            rng,
            limit,
        )
    ]


def lm_head_device(model) -> torch.device:
    return next(model.lm_head.parameters()).device


def model_cuda_devices(model) -> list[torch.device]:
    devices: set[torch.device] = set()
    for value in (getattr(model, "hf_device_map", None) or {}).values():
        if isinstance(value, int):
            devices.add(torch.device(f"cuda:{value}"))
        elif str(value).startswith("cuda"):
            devices.add(torch.device(value))
    for device in (model_input_device(model), lm_head_device(model)):
        if device.type == "cuda":
            devices.add(device)
    return sorted(devices, key=lambda device: device.index or 0)


@torch.no_grad()
def compute_per_sequence_loss(
    model,
    token_batches: list[list[int]],
    chunk_size: int,
    *,
    reset_peak_memory: bool = True,
) -> list[float]:
    if not token_batches:
        return []
    input_device = model_input_device(model)
    cuda_devices = model_cuda_devices(model)
    if reset_peak_memory:
        for device in cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=input_device)
    try:
        if hasattr(model, "reset_state"):
            model.reset_state()
        hidden = model.model(input_ids, use_cache=False).last_hidden_state
        head_dev = lm_head_device(model)
        if hidden.device != head_dev:
            hidden = hidden.to(head_dev)
        labels_full = input_ids if input_ids.device == head_dev else input_ids.to(head_dev)

        batch = len(token_batches)
        n_pos = labels_full.size(1) - 1
        per_token_losses = []
        for start in range(0, n_pos, chunk_size):
            end = min(start + chunk_size, n_pos)
            logits = model.lm_head(hidden[:, start:end, :])
            labels = labels_full[:, start + 1 : end + 1]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            )
            per_token_losses.append(loss.reshape(batch, -1).float())
            del logits
        # Summing once after concatenation makes the accumulation order independent
        # of lm_head_chunk, so increasing the projection chunk does not alter scores.
        total = torch.cat(per_token_losses, dim=1).sum(dim=1)
        result = (total / n_pos).float().cpu().tolist()
        peaks = {
            str(device): round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
            for device in cuda_devices
        }
        eval_log.debug(
            "eager memory | devices=%s seq_len=%d peak_allocated_gib=%s",
            [str(device) for device in cuda_devices],
            input_ids.shape[1],
            peaks,
        )
        return result
    except (torch.OutOfMemoryError, MemoryError) as exc:
        peak_detail = " ".join(
            f"{device}:allocated={torch.cuda.max_memory_allocated(device) / (1024**3):.2f}GiB,"
            f"reserved={torch.cuda.max_memory_reserved(device) / (1024**3):.2f}GiB"
            for device in cuda_devices
        )
        raise RuntimeError(
            f"OOM scoring a batch of {input_ids.shape[0]} unmodified "
            f"{input_ids.shape[1]}-token sequences; "
            f"evaluation stopped without truncation or backend fallback. {peak_detail}"
        ) from exc


class TwoGpuSequencePipeline:
    """Score batches, overlapping two single-sequence forwards when possible."""

    def __init__(self, model, req: EvalRequest, spec: dict, result_queue, generation: str):
        self.model = model
        self.req = req
        self.spec = spec
        self.result_queue = result_queue
        self.generation = generation
        self._thread_state = threading.local()
        self._stage2_lock = threading.Lock()
        self._previous_boundary = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=spec["worker_id"])
        self._hook = None
        self.boundary_layer = self._find_boundary_layer()
        if self.boundary_layer is not None:
            layers = model.model.layers
            self._hook = layers[self.boundary_layer].register_forward_pre_hook(
                self._enter_stage2,
                prepend=True,
            )

    @property
    def depth(self) -> int:
        return 2 if self.boundary_layer is not None else 1

    def _find_boundary_layer(self) -> int | None:
        if self.req.batch_size > 1 or len(self.spec["gpu_ids"]) != 2:
            return None
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            return None
        second_gpu = self.spec["gpu_ids"][1]
        for index, layer in enumerate(layers):
            parameter = next(layer.parameters(), None)
            if parameter is not None and parameter.device.type == "cuda" and parameter.device.index == second_gpu:
                return index
        return None

    def _enter_stage2(self, _module, _args) -> None:
        state = getattr(self._thread_state, "current", None)
        if state is None or state["stage2_acquired"]:
            return
        self._stage2_lock.acquire()
        state["stage2_acquired"] = True
        state["boundary"].set()

    def submit(self, sequence_index: int, token_ids: list[int]) -> None:
        self.submit_batch([sequence_index], [token_ids])

    def submit_batch(
        self,
        sequence_indices: list[int],
        token_batches: list[list[int]],
    ) -> None:
        if not sequence_indices or len(sequence_indices) != len(token_batches):
            raise ValueError("sequence indices and token batches must be non-empty and aligned")
        if self._previous_boundary is not None:
            self._previous_boundary.wait()
        state = {"boundary": threading.Event(), "stage2_acquired": False}
        self._previous_boundary = state["boundary"]
        self._executor.submit(self._score, sequence_indices, token_batches, state)

    def _score(
        self,
        sequence_indices: list[int],
        token_batches: list[list[int]],
        state: dict,
    ) -> None:
        self._thread_state.current = state
        started = time.time()
        try:
            losses = compute_per_sequence_loss(
                self.model,
                token_batches,
                self.req.lm_head_chunk,
                reset_peak_memory=False,
            )
            if len(losses) != len(sequence_indices):
                raise RuntimeError(
                    f"scorer returned {len(losses)} losses for "
                    f"{len(sequence_indices)} sequences"
                )
            result = {
                "type": "result",
                "generation": self.generation,
                "worker_id": self.spec["worker_id"],
                "role": self.spec["role"],
                "sequence_indices": sequence_indices,
                "losses": losses,
                "wall_time_s": time.time() - started,
            }
            if len(sequence_indices) == 1:
                result.update(
                    sequence_index=sequence_indices[0],
                    loss=losses[0],
                )
            self.result_queue.put(result)
        except BaseException as exc:
            self.result_queue.put({
                "type": "error",
                "generation": self.generation,
                "worker_id": self.spec["worker_id"],
                "role": self.spec["role"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        finally:
            state["boundary"].set()
            if state["stage2_acquired"]:
                self._stage2_lock.release()
            self._thread_state.current = None

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._hook is not None:
            self._hook.remove()


def empty_worker_cuda_cache(gpu_ids: list[int]) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    for gpu_id in gpu_ids:
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()


def reset_worker_peak_memory(gpu_ids: list[int]) -> None:
    if not torch.cuda.is_available():
        return
    for gpu_id in gpu_ids:
        torch.cuda.reset_peak_memory_stats(gpu_id)


def model_worker_main(spec: dict, command_queue, result_queue) -> None:
    """Own one persistent two-GPU model instance and accept load/score commands."""
    setup_logging()
    worker_id = spec["worker_id"]
    role = spec["role"]
    gpu_ids = list(spec["gpu_ids"])
    model = None
    pipeline = None
    loaded_key = None
    try:
        while True:
            command = command_queue.get()
            if command["type"] == "shutdown":
                break
            if command["type"] == "load":
                generation = command["generation"]
                try:
                    if pipeline is not None:
                        pipeline.close()
                        pipeline = None
                    req = EvalRequest(**command["request"])
                    reused = model is not None and loaded_key == command["model_key"]
                    if not reused:
                        if model is not None:
                            del model
                            model = None
                            empty_worker_cuda_cache(gpu_ids)
                        config, artifacts = load_model_config(command["snapshot"], req, worker_id)
                        attention = validate_and_report_attention_config(config, worker_id)
                        kernel_cache = configure_kernel_cache(config, gpu_ids)
                        model = load_eval_model(
                            command["snapshot"],
                            config,
                            "auto",
                            worker_id,
                            req,
                            gpu_ids=gpu_ids,
                        )
                        loaded_key = command["model_key"]
                    else:
                        artifacts = command.get("artifacts", {})
                        attention = command.get("attention", {})
                        kernel_cache = command.get("kernel_cache", {})
                    reset_worker_peak_memory(gpu_ids)
                    pipeline = TwoGpuSequencePipeline(model, req, spec, result_queue, generation)
                    result_queue.put({
                        "type": "ready",
                        "generation": generation,
                        "worker_id": worker_id,
                        "role": role,
                        "gpu_ids": gpu_ids,
                        "pid": os.getpid(),
                        "snapshot": command["snapshot"],
                        "reused_model": reused,
                        "artifacts": artifacts,
                        "attention": attention,
                        "kernel_cache": kernel_cache,
                        "pipeline_depth": pipeline.depth,
                        "pipeline_boundary_layer": pipeline.boundary_layer,
                    })
                except BaseException as exc:
                    result_queue.put({
                        "type": "error",
                        "generation": generation,
                        "worker_id": worker_id,
                        "role": role,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
            elif command["type"] == "score":
                if pipeline is None or command["generation"] != pipeline.generation:
                    raise RuntimeError(f"{worker_id} received score command before matching load")
                pipeline.submit_batch(
                    command["sequence_indices"],
                    command["token_batches"],
                )
            else:
                raise RuntimeError(f"unknown worker command: {command['type']}")
    except BaseException as exc:
        result_queue.put({
            "type": "error",
            "generation": "worker",
            "worker_id": worker_id,
            "role": role,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        if pipeline is not None:
            pipeline.close()


class PersistentModelWorkerPool:
    """Keep four model processes alive and reload only when checkpoint identity changes."""

    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = list(gpu_ids)
        self.specs = model_worker_specs(self.gpu_ids)
        context = mp.get_context("spawn")
        self.result_queue = context.Queue()
        self.command_queues = {}
        self.processes = {}
        self.ready: dict[str, dict] = {}
        for spec in self.specs:
            worker_id = spec["worker_id"]
            queue = context.Queue(maxsize=4)
            process = context.Process(
                target=model_worker_main,
                args=(spec, queue, self.result_queue),
                name=f"mimo-{worker_id}",
            )
            process.start()
            self.command_queues[worker_id] = queue
            self.processes[worker_id] = process

    def dead_workers(self) -> list[str]:
        return [worker_id for worker_id, process in self.processes.items() if not process.is_alive()]

    def load_models(
        self,
        req: EvalRequest,
        king_snapshot: str,
        challenger_snapshot: str,
        on_progress,
    ) -> str:
        generation = uuid.uuid4().hex
        request_data = req.model_dump()
        for spec in self.specs:
            snapshot = king_snapshot if spec["role"] == "king" else challenger_snapshot
            worker_id = spec["worker_id"]
            previous = self.ready.get(worker_id, {})
            self.command_queues[worker_id].put({
                "type": "load",
                "generation": generation,
                "snapshot": snapshot,
                "request": request_data,
                "model_key": checkpoint_load_key(snapshot, req, spec["gpu_ids"]),
                "artifacts": previous.get("artifacts", {}),
                "attention": previous.get("attention", {}),
                "kernel_cache": previous.get("kernel_cache", {}),
            })

        ready = {}
        while len(ready) < len(self.specs):
            try:
                message = self.result_queue.get(timeout=30)
            except Empty:
                dead = self.dead_workers()
                if dead:
                    raise RuntimeError(f"model workers exited during load: {dead}")
                on_progress({"phase": "model_workers_loading", "ready": len(ready), "total_workers": len(self.specs)})
                continue
            if message.get("generation") != generation:
                raise RuntimeError(f"stale model worker message during load: {message}")
            if message["type"] == "error":
                raise RuntimeError(
                    f"model worker {message['worker_id']} failed: {message['error']}\n{message['traceback']}"
                )
            if message["type"] != "ready":
                raise RuntimeError(f"unexpected model worker load message: {message}")
            ready[message["worker_id"]] = message
            on_progress({
                "phase": "model_worker_ready",
                "worker_id": message["worker_id"],
                "role": message["role"],
                "gpu_ids": message["gpu_ids"],
                "pid": message["pid"],
                "reused_model": message["reused_model"],
                "pipeline_depth": message["pipeline_depth"],
                "ready": len(ready),
                "total_workers": len(self.specs),
            })
        self.ready = ready
        return generation

    def score(
        self,
        sequences: list[list[int]],
        generation: str,
        req: EvalRequest,
        on_progress,
    ) -> tuple[list[float], list[float], dict]:
        king_losses: list[float | None] = [None] * len(sequences)
        challenger_losses: list[float | None] = [None] * len(sequences)
        next_index = {"king": 0, "challenger": 0}
        in_flight = {spec["worker_id"]: 0 for spec in self.specs}
        paired_done = 0
        progress_log_interval = max(1, (len(sequences) + 9) // 10)
        provisional: dict[str, Any] = {}
        early_policy = EarlyStoppingPolicy(
            enabled=req.early_stop_enabled,
            min_fraction=req.early_stop_min_fraction,
            advantage_quantile=req.early_stop_advantage_quantile,
            margin=req.early_stop_margin,
            check_interval=req.early_stop_check_interval,
        )
        next_early_check = max(1, math.ceil(len(sequences) * early_policy.min_fraction))
        early_stop_at: int | None = None
        early_stop_decision: dict[str, float] | None = None

        def fill_worker(worker_id: str, role: str) -> None:
            depth = int(self.ready[worker_id].get("pipeline_depth", 1))
            while in_flight[worker_id] < depth and next_index[role] < len(sequences):
                start = next_index[role]
                end = min(start + req.batch_size, len(sequences))
                self.command_queues[worker_id].put({
                    "type": "score",
                    "generation": generation,
                    "sequence_indices": list(range(start, end)),
                    "token_batches": sequences[start:end],
                })
                next_index[role] = end
                in_flight[worker_id] += 1

        for spec in self.specs:
            fill_worker(spec["worker_id"], spec["role"])

        started = time.time()
        while paired_done < len(sequences) and (
            early_stop_at is None or any(in_flight.values())
        ):
            try:
                message = self.result_queue.get(timeout=30)
            except Empty:
                dead = self.dead_workers()
                if dead:
                    raise RuntimeError(f"model workers exited without a result: {dead}")
                on_progress({
                    "phase": "heartbeat",
                    "done": paired_done,
                    "total": len(sequences),
                    **provisional,
                })
                continue
            if message.get("generation") != generation:
                raise RuntimeError(f"stale model worker message during scoring: {message}")
            if message["type"] == "error":
                raise RuntimeError(
                    f"model worker {message['worker_id']} failed: {message['error']}\n{message['traceback']}"
                )
            if message["type"] != "result":
                raise RuntimeError(f"unexpected model worker score message: {message}")

            worker_id = message["worker_id"]
            role = message["role"]
            in_flight[worker_id] -= 1
            indices = [int(value) for value in message["sequence_indices"]]
            losses = [float(value) for value in message["losses"]]
            if not indices or len(indices) != len(losses):
                raise RuntimeError(f"malformed batched result from {worker_id}")
            target = king_losses if role == "king" else challenger_losses
            for index, loss in zip(indices, losses, strict=True):
                if not 0 <= index < len(target):
                    raise RuntimeError(f"out-of-range {role} result for sequence {index}")
                if target[index] is not None:
                    raise RuntimeError(f"duplicate {role} result for sequence {index}")
                target[index] = loss

            previous_done = paired_done
            if early_stop_at is None:
                while (
                    paired_done < len(sequences)
                    and king_losses[paired_done] is not None
                    and challenger_losses[paired_done] is not None
                ):
                    paired_done += 1
            if paired_done != previous_done and early_stop_at is None:
                paired_king = np.asarray(king_losses[:paired_done], dtype=np.float64)
                paired_challenger = np.asarray(challenger_losses[:paired_done], dtype=np.float64)
                elapsed = max(time.time() - started, 1e-9)
                avg_king_loss = float(paired_king.mean())
                avg_challenger_loss = float(paired_challenger.mean())
                loss_delta = float((paired_king - paired_challenger).mean())
                seq_per_s = paired_done / elapsed
                crossed_log_boundary = (
                    paired_done // progress_log_interval
                    != previous_done // progress_log_interval
                )
                if paired_done == len(sequences) or crossed_log_boundary:
                    provisional = provisional_paired_bootstrap(
                        [float(value) for value in paired_king],
                        [float(value) for value in paired_challenger],
                        bootstrap_seed=req.bootstrap_seed,
                        n_bootstrap=req.n_bootstrap,
                        alpha=req.alpha,
                        delta_threshold=req.delta_threshold,
                    )
                on_progress({
                    "phase": "eval_progress",
                    "done": paired_done,
                    "total": len(sequences),
                    "mu_hat": round(loss_delta, 6),
                    "seq_per_s": round(seq_per_s, 4),
                    "avg_king_loss": round(avg_king_loss, 6),
                    "avg_challenger_loss": round(avg_challenger_loss, 6),
                    **provisional,
                })
                if paired_done == len(sequences) or crossed_log_boundary:
                    eval_log.info(
                        "loss progress | paired=%d/%d king=%.6f challenger=%.6f "
                        "king_minus_challenger=%.6f provisional_lcb=%.6f seq_per_s=%.4f",
                        paired_done,
                        len(sequences),
                        avg_king_loss,
                        avg_challenger_loss,
                        loss_delta,
                        float(provisional["provisional_lcb"]),
                        seq_per_s,
                    )
                while (
                    early_stop_at is None
                    and early_policy.enabled
                    and next_early_check < len(sequences)
                    and next_early_check <= paired_done
                ):
                    check_at = next_early_check
                    next_early_check += early_policy.check_interval
                    decision = challenger_futility_decision(
                        [float(value) for value in king_losses[:check_at]],
                        [float(value) for value in challenger_losses[:check_at]],
                        total_sequences=len(sequences),
                        delta_threshold=req.delta_threshold,
                        policy=early_policy,
                    )
                    if decision is not None:
                        early_stop_decision = decision
                        early_stop_at = check_at
                        on_progress({
                            "phase": "eval_early_stopping",
                            "done": check_at,
                            "total": len(sequences),
                            "early_stopped": True,
                            "mu_hat": round(early_stop_decision["mu_hat"], 6),
                            "mu_hat_upper_bound": round(
                                early_stop_decision["mu_hat_upper_bound"], 6
                            ),
                        })
                        eval_log.info(
                            "early stop | paired=%d/%d mu_upper=%.6f stop_threshold=%.6f "
                            "advantage_quantile=%.4f",
                            check_at,
                            len(sequences),
                            early_stop_decision["mu_hat_upper_bound"],
                            early_stop_decision["stop_threshold"],
                            early_policy.advantage_quantile,
                        )
            if early_stop_at is None:
                fill_worker(worker_id, role)

        completed = early_stop_at or len(sequences)
        return (
            [float(value) for value in king_losses[:completed]],
            [float(value) for value in challenger_losses[:completed]],
            {
                "workers": [self.ready[spec["worker_id"]] for spec in self.specs],
                "early_stop": (
                    {
                        **early_stop_decision,
                        "completed_sequences": completed,
                        **early_policy.request_dict(),
                    }
                    if early_stop_decision is not None
                    else None
                ),
            },
        )

    def close(self, *, force: bool = False) -> None:
        if force:
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()
            for process in self.processes.values():
                process.join(timeout=10)
            for queue in [*self.command_queues.values(), self.result_queue]:
                try:
                    queue.close()
                except Exception:
                    pass
            return
        for queue in self.command_queues.values():
            try:
                queue.put_nowait({"type": "shutdown"})
            except Exception:
                pass
        for process in self.processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        for queue in [*self.command_queues.values(), self.result_queue]:
            try:
                queue.close()
            except Exception:
                pass

    def status(self) -> dict:
        return {
            "running": not bool(self.dead_workers()),
            "workers": [
                {
                    "worker_id": spec["worker_id"],
                    "pid": self.processes[spec["worker_id"]].pid,
                    "alive": self.processes[spec["worker_id"]].is_alive(),
                    "loaded": spec["worker_id"] in self.ready,
                }
                for spec in self.specs
            ],
        }

    def loaded_snapshot_paths(self) -> set[Path]:
        return {
            Path(message["snapshot"]).resolve()
            for message in self.ready.values()
            if message.get("snapshot")
        }


def get_model_worker_pool(gpu_ids: list[int]) -> PersistentModelWorkerPool:
    global _model_worker_pool
    if _model_worker_pool is not None and _model_worker_pool.gpu_ids != list(gpu_ids):
        _model_worker_pool.close()
        _model_worker_pool = None
    if _model_worker_pool is None or _model_worker_pool.dead_workers():
        if _model_worker_pool is not None:
            _model_worker_pool.close(force=True)
        _model_worker_pool = PersistentModelWorkerPool(gpu_ids)
    return _model_worker_pool


def score_with_model_workers(
    sequences: list[list[int]],
    req: EvalRequest,
    king_snapshot: str,
    challenger_snapshot: str,
    gpu_ids: list[int],
    on_progress,
) -> tuple[list[float], list[float], dict]:
    """Score randomized sequences; only worker/model state persists between evals."""
    global _model_worker_pool
    pool = get_model_worker_pool(gpu_ids)
    try:
        generation = pool.load_models(req, king_snapshot, challenger_snapshot, on_progress)
        return pool.score(sequences, generation, req, on_progress)
    except Exception:
        # In particular, an eager-attention OOM must halt every queued sequence
        # immediately: never keep scoring, truncate, or switch attention backend.
        pool.close(force=True)
        _model_worker_pool = None
        raise


def bootstrap_verdict(king_losses: list[float], challenger_losses: list[float], req: EvalRequest) -> dict:
    return paired_bootstrap_verdict(
        king_losses,
        challenger_losses,
        bootstrap_seed=req.bootstrap_seed,
        n_bootstrap=req.n_bootstrap,
        alpha=req.alpha,
        delta_threshold=req.delta_threshold,
    )


def _compute_source_scores(
    king_losses: list[float],
    challenger_losses: list[float],
    source_labels: list[str] | None,
) -> dict:
    """Per-source avg_king_loss, avg_challenger_loss, mu_hat.

    Returns an empty dict when source_labels is unavailable (non-multi-source
    evals or old code paths that don't set _source_labels).
    """
    if not source_labels or len(source_labels) != len(king_losses):
        return {}
    king_arr = np.asarray(king_losses, dtype=np.float64)
    chall_arr = np.asarray(challenger_losses, dtype=np.float64)
    diff_arr = king_arr - chall_arr
    labels_arr = np.asarray(source_labels)
    scores: dict = {}
    for name in sorted(set(source_labels)):
        mask = labels_arr == name
        n = int(mask.sum())
        if n == 0:
            continue
        scores[name] = {
            "n_sequences": n,
            "avg_king_loss": round(float(king_arr[mask].mean()), 6),
            "avg_challenger_loss": round(float(chall_arr[mask].mean()), 6),
            "mu_hat": round(float(diff_arr[mask].mean()), 6),
        }
    return scores


def _build_sample_results(
    king_losses: list[float],
    challenger_losses: list[float],
    sample_provenance: list[dict[str, int]] | None,
) -> dict[str, Any]:
    """Build compact, index-aligned per-sample audit results."""
    if sample_provenance is None:
        raise RuntimeError("evaluation sampler did not provide sample provenance")
    if not (
        len(king_losses) == len(challenger_losses) == len(sample_provenance)
    ):
        raise RuntimeError(
            "sample provenance and paired losses must have identical lengths"
        )
    return {
        "format": "columnar-v1",
        "n_samples": len(king_losses),
        "shard_group_index": [
            int(item["shard_group_index"]) for item in sample_provenance
        ],
        "shard_index": [int(item["shard_index"]) for item in sample_provenance],
        "shard_sequence_index": [
            int(item["shard_sequence_index"]) for item in sample_provenance
        ],
        "king_loss": [float(value) for value in king_losses],
        "challenger_loss": [float(value) for value in challenger_losses],
    }


def _shards_used(dataset_meta: dict) -> list[dict]:
    out = []
    for source in dataset_meta.get("sources") or []:
        refs = list(source.get("used_refs") or [])
        if refs:
            out.append({"source": source.get("name") or "dataset", "refs": refs})
    used_keys = list(dataset_meta.get("used_keys") or [])
    if used_keys:
        bucket = dataset_meta.get("bucket") or ""
        refs = [f"s3://{bucket}/{key}" if bucket and not str(key).startswith(("s3://", "http://", "https://")) else key for key in used_keys]
        out.append({"source": dataset_meta.get("source") or "dataset", "refs": refs})
    return out


def _public_dataset_meta(dataset_meta: dict) -> dict:
    meta = {
        "source": dataset_meta.get("source"),
        "shards_used": _shards_used(dataset_meta),
        "min_seq_len": dataset_meta.get("min_seq_len"),
        "max_seq_len": dataset_meta.get("max_seq_len"),
        "max_model_len": dataset_meta.get("max_model_len"),
        "length_source": dataset_meta.get("length_source"),
    }
    return {k: v for k, v in meta.items() if v}


def ensure_king(req: EvalRequest, snapshot: str, config, config_source: str, device: str, gpu_ids: list[int] | None = None, on_phase=None):
    global _king_model, _king_key, _king_device, _king_gpu_ids
    repo = normalize_model_ref(req.king_repo)
    key = (repo, req.king_digest or "latest", config_source)
    effective_gpu_ids = list(gpu_ids) if gpu_ids is not None else list(_gpu_ids)
    if _king_model is not None and _king_key == key and _king_device == device and _king_gpu_ids == effective_gpu_ids:
        if on_phase:
            on_phase({"phase": "king_reuse", "repo": repo, "device": device})
        return _king_model
    if _king_model is not None:
        del _king_model
        _king_model = None
        torch.cuda.empty_cache()
    _king_model = load_model_replicas(
        snapshot,
        config,
        "king",
        req,
        effective_gpu_ids,
        on_phase=on_phase,
    )
    _king_key = key
    _king_device = device
    _king_gpu_ids = effective_gpu_ids
    write_current_king_ref(snapshot)
    return _king_model


def write_current_king_ref(snapshot: str) -> None:
    ref_file = MODEL_CACHE_DIR / ".current_king"
    try:
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        ref_file.write_text(str(Path(snapshot).resolve()))
    except Exception:
        log.warning("could not write current king ref %s", ref_file, exc_info=True)


def promote_challenger_to_king(
    req: EvalRequest,
    challenger_model,
    challenger_snapshot: str,
    config_source: str,
    device: str,
    gpu_ids: list[int],
    on_phase=None,
) -> None:
    global _king_model, _king_key, _king_device, _king_gpu_ids
    repo = normalize_model_ref(req.challenger_repo)
    _king_model = challenger_model
    _king_key = (repo, req.challenger_digest or "latest", config_source)
    _king_device = device
    _king_gpu_ids = list(gpu_ids)
    write_current_king_ref(challenger_snapshot)
    log.info("promoted challenger to king: %s@%s", repo, req.challenger_digest or "latest")
    if on_phase:
        on_phase({"phase": "king_promoted", "repo": repo, "digest": req.challenger_digest or "latest"})


def cleanup_model_cache() -> None:
    try:
        if not MODEL_CACHE_DIR.exists():
            return
        snapshots = [p for p in MODEL_CACHE_DIR.glob("*/*") if p.is_dir()]
        total = sum(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in snapshots)
        if total / 1e9 < CACHE_HIGH_WATERMARK_GB:
            return
        target = CACHE_HIGH_WATERMARK_GB * 0.7 * 1e9
        running = total
        keep = set()
        if _king_key:
            keep.add((_king_key[0].replace("/", "--"), _king_key[1].replace(":", "-")))
        loaded_paths = (
            _model_worker_pool.loaded_snapshot_paths()
            if _model_worker_pool is not None
            else set()
        )
        candidates = []
        for d in snapshots:
            try:
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                candidates.append((d.stat().st_mtime, d, size))
            except Exception:
                continue
        for _mtime, d, size in sorted(candidates):
            if running < target:
                break
            if (d.parent.name, d.name) in keep:
                continue
            if d.resolve() in loaded_paths:
                continue
            shutil.rmtree(d, ignore_errors=True)
            running -= size
            log.info("cache cleanup: deleted %s (%.1f GB)", d, size / 1e9)
    except Exception:
        log.warning("cache cleanup failed", exc_info=True)


def write_record(eval_id: str, payload: dict) -> tuple[str, str]:
    EVAL_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_RECORD_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{eval_id}.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    path.write_bytes(encoded)
    return str(path), hashlib.sha256(encoded).hexdigest()


def run_eval(eval_id: str, protocol_request: EvaluationRequestV2) -> None:
    record = _attempts.get(eval_id)
    if record is None:
        try:
            _eval_lock.release()
        except RuntimeError:
            pass
        return
    events: Queue = record.events
    record.state = "running"
    t0 = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    def on_phase(info: dict):
        previous = record.progress or {}
        normalized = dict(info)
        completed = normalized.pop(
            "done",
            normalized.get("completed_sequences", previous.get("completed_sequences", 0)),
        )
        requested = normalized.pop(
            "total",
            normalized.get(
                "requested_sequences",
                previous.get("requested_sequences", int(protocol_request.limits["n"])),
            ),
        )
        completed = max(0, int(completed))
        requested = max(0, int(requested))
        normalized["completed_sequences"] = completed
        normalized["requested_sequences"] = requested
        normalized["percent"] = round(
            (100.0 * completed / requested) if requested else 0.0,
            4,
        )
        normalized["elapsed_seconds"] = round(max(time.time() - t0, 0.0), 1)
        normalized["early_stopped"] = bool(normalized.get("early_stopped", False))
        record.progress = normalized
        events.put(record.event("progress", normalized))

    heartbeat_stop = threading.Event()

    def heartbeat_loop():
        while not heartbeat_stop.wait(30):
            current = dict(record.progress or {})
            current["heartbeat_at"] = time.time()
            on_phase(current or {"phase": "heartbeat"})

    threading.Thread(target=heartbeat_loop, daemon=True, name=f"heartbeat-{eval_id}").start()

    try:
        on_phase({"phase": "setup_start"})
        resolver = R2ArtifactResolver(MODEL_CACHE_DIR / "immutable-r2")
        on_phase({"phase": "artifact_materialization_start", "role": "king"})
        king_snapshot = resolver.resolve(protocol_request.king)
        on_phase({
            "phase": "artifact_materialization_verified",
            "role": "king",
            "digest": protocol_request.king.expected_digest,
        })
        on_phase({"phase": "artifact_materialization_start", "role": "challenger"})
        challenger_snapshot = resolver.resolve(protocol_request.challenger)
        on_phase({
            "phase": "artifact_materialization_verified",
            "role": "challenger",
            "digest": protocol_request.challenger.expected_digest,
        })
        req = internal_request_from_v2(protocol_request, king_snapshot, challenger_snapshot)
        limits_meta = apply_eval_limits(req, eval_id)
        on_phase({"phase": "limits_applied", **limits_meta})
        check_eval_runtime(t0)
        king_snapshot = materialize_model(req.king_repo, req.king_digest, on_phase=on_phase)
        check_eval_runtime(t0)
        challenger_snapshot = materialize_model(req.challenger_repo, req.challenger_digest, on_phase=on_phase)
        check_eval_runtime(t0)
        duplicate_meta = reject_duplicate_safetensors(king_snapshot, challenger_snapshot, on_phase=on_phase)
        local_challenger_digest = duplicate_meta["challenger_safetensors_sha256"]
        duplicate_meta.update(
            reject_reused_safetensors(local_challenger_digest, on_phase=on_phase)
        )
        check_eval_runtime(t0)
        king_config, king_artifacts = load_model_config(king_snapshot, req, "king", on_phase=on_phase)
        challenger_config, challenger_artifacts = load_model_config(
            challenger_snapshot,
            req,
            "challenger",
            on_phase=on_phase,
        )
        check_eval_runtime(t0)
        config_mismatches = compare_model_configs(king_config, challenger_config)
        if config_mismatches:
            raise RuntimeError(f"king/challenger config mismatch: {config_mismatches[:8]}")
        req.vocab_size = int(config_value(king_config, "vocab_size") or 0)
        if req.vocab_size <= 0:
            raise RuntimeError("model config must define a positive vocab_size")
        attention_meta = {
            "king": validate_and_report_attention_config(
                king_config,
                "king",
                on_phase=on_phase,
            ),
            "challenger": validate_and_report_attention_config(
                challenger_config,
                "challenger",
                on_phase=on_phase,
            ),
        }

        on_phase({
            "phase": "dataset_sample_start",
            "source": req.dataset_source,
            "n": req.n,
            "seq_len": req.seq_len,
        })
        sequences, dataset_meta = sample_eval_sequences(req, on_phase=on_phase)
        if not sequences:
            raise RuntimeError("pretokenized evaluation corpus is empty")
        corpus_lengths = [len(sequence) for sequence in sequences]
        corpus_max_seq_len = max(corpus_lengths)
        corpus_min_seq_len = min(corpus_lengths)
        if corpus_max_seq_len > int(king_config.max_position_embeddings):
            raise RuntimeError(
                f"corpus max sequence length {corpus_max_seq_len} exceeds checkpoint limit "
                f"{king_config.max_position_embeddings}"
            )
        king_config._eval_max_model_len = corpus_max_seq_len
        challenger_config._eval_max_model_len = corpus_max_seq_len
        dataset_meta.update({
            "min_seq_len": corpus_min_seq_len,
            "max_seq_len": corpus_max_seq_len,
            "max_model_len": corpus_max_seq_len,
            "length_source": "tokenized_corpus",
        })
        # Pop private key so it never reaches the verdict JSON or disk record.
        source_labels: list[str] | None = dataset_meta.pop("_source_labels", None)
        sample_provenance: list[dict[str, int]] | None = dataset_meta.pop(
            "_sample_provenance", None
        )
        if sample_provenance is None or len(sample_provenance) != len(sequences):
            raise RuntimeError(
                "evaluation sampler returned incomplete sample provenance"
            )
        public_dataset_meta = _public_dataset_meta(dataset_meta)
        check_eval_runtime(t0)
        on_phase({
            "phase": "dataset_sample_done",
            "digest": dataset_meta["digest"][:16],
            "min_seq_len": corpus_min_seq_len,
            "max_seq_len": corpus_max_seq_len,
            "max_model_len": corpus_max_seq_len,
        })

        worker_specs = model_worker_specs(_gpu_ids)
        on_phase({
            "phase": "parallel_models_setup",
            "workers": worker_specs,
            "gpus_per_model_instance": GPUS_PER_MODEL_INSTANCE,
            "model_instances_per_side": MODEL_INSTANCES_PER_SIDE,
            "worker_processes": MODEL_WORKER_PROCESSES,
            "tensor_parallel_size": 1,
            "model_parallel_strategy": "layer_sharding",
        })
        king_device = "cuda:0,1|cuda:2,3"
        challenger_device = "cuda:4,5|cuda:6,7"
        use_parallel = True
        king_losses, challenger_losses, worker_meta = score_with_model_workers(
            sequences,
            req,
            king_snapshot,
            challenger_snapshot,
            _gpu_ids,
            on_phase,
        )

        verdict = bootstrap_verdict(king_losses, challenger_losses, req)
        early_stop = worker_meta.get("early_stop")
        if early_stop is not None:
            verdict.update({
                "accepted": False,
                "verdict": "king",
                "early_stopped": True,
                "n_sequences": len(sequences),
                "n_sequences_evaluated": len(king_losses),
                "mu_hat_upper_bound": round(
                    float(early_stop["mu_hat_upper_bound"]), 6
                ),
                "early_stop_reason": (
                    f"projected_upper_mean={early_stop['mu_hat_upper_bound']:.6f} "
                    f"< stop_threshold={early_stop['stop_threshold']:.6f} "
                    f"after {len(king_losses)}/{len(sequences)} sequences"
                ),
                "early_stop_min_fraction": float(early_stop["min_fraction"]),
                "early_stop_advantage_quantile": float(
                    early_stop["advantage_quantile"]
                ),
                "early_stop_assumed_remaining_advantage": round(
                    float(early_stop["assumed_remaining_advantage"]), 6
                ),
                "early_stop_margin": float(early_stop["margin"]),
                "early_stop_check_interval": int(early_stop["check_interval"]),
            })
        eval_log.info(
            "loss verdict | eval_id=%s paired=%d king=%.6f challenger=%.6f "
            "king_minus_challenger=%.6f lcb=%.6f threshold=%.6f accepted=%s",
            eval_id,
            int(verdict.get("n_sequences_evaluated", verdict["n_sequences"])),
            float(verdict["avg_king_loss"]),
            float(verdict["avg_challenger_loss"]),
            float(verdict["mu_hat"]),
            float(verdict["lcb"]),
            float(verdict["delta_threshold"]),
            bool(verdict["accepted"]),
        )
        verdict["source_scores"] = _compute_source_scores(
            king_losses,
            challenger_losses,
            source_labels[: len(king_losses)] if source_labels else None,
        )
        verdict["sample_results"] = _build_sample_results(
            king_losses,
            challenger_losses,
            sample_provenance[: len(king_losses)],
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        verdict.update({
            "eval_id": eval_id,
            "king_digest": protocol_request.king.expected_digest,
            "challenger_digest": protocol_request.challenger.expected_digest,
            "king_device": king_device,
            "challenger_device": challenger_device,
            "parallel_models": use_parallel,
            "gpu_memory_fraction": req.gpu_memory_fraction,
            "limits": limits_meta,
            "model_artifacts": {
                "king": king_artifacts,
                "challenger": challenger_artifacts,
                "attention": attention_meta,
                "duplicate_check": duplicate_meta,
                "workers": worker_meta,
            },
            "max_model_len": corpus_max_seq_len,
            "gpus_per_model_instance": GPUS_PER_MODEL_INSTANCE,
            "model_instances_per_side": MODEL_INSTANCES_PER_SIDE,
            "worker_processes": MODEL_WORKER_PROCESSES,
            "tensor_parallel_size": 1,
            "model_parallel_strategy": "layer_sharding",
            "grouped_moe": True,
            "dataset": public_dataset_meta,
            "shards_used": public_dataset_meta.get("shards_used", []),
            "dataset_source": req.dataset_source,
            "wall_time_s": round(time.time() - t0, 1),
        })
        verdict.update(
            result_provenance(
                protocol_request,
                started_at=started_at,
                completed_at=completed_at,
                requested_sequences=int(protocol_request.limits["n"]),
                completed_sequences=int(
                    verdict.get("n_sequences_evaluated", verdict.get("n_sequences", 0))
                ),
                early_stopped=bool(verdict.get("early_stopped", False)),
                hardware={
                    "gpu_ids": list(_gpu_ids),
                    "workers": [spec["worker_id"] for spec in worker_specs],
                    "model_parallel_strategy": "layer_sharding",
                },
            )
        )
        record_path, result_artifact_sha256 = write_record(
            eval_id,
            {"request": protocol_request.request_payload, "verdict": verdict},
        )
        verdict["record_path"] = record_path
        verdict["result_artifact_sha256"] = result_artifact_sha256
        validate_result_v2(verdict, protocol_request)
        if verdict.get("accepted"):
            write_current_king_ref(challenger_snapshot)
            on_phase({
                "phase": "king_promoted",
                "digest": protocol_request.challenger.expected_digest,
            })
        record_completed_safetensors_sha(duplicate_meta["challenger_safetensors_sha256"])
        record.state = "completed"
        record.verdict = verdict
        events.put(record.event("verdict", verdict))
    except Exception as exc:
        log.exception("eval %s failed", eval_id)
        reason = str(exc)
        error_code = (
            "safetensors_reuse_limit"
            if isinstance(exc, SafetensorsReuseLimitError)
            else "evaluation_failed"
        )
        record.state = "failed"
        record.error = reason
        record.reason = reason
        record.error_code = error_code
        events.put(
            record.event(
                "error", {"code": error_code, "error": reason, "reason": reason}
            )
        )
    finally:
        heartbeat_stop.set()
        cleanup_model_cache()
        try:
            _eval_lock.release()
        except RuntimeError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu_ids, _model_worker_pool
    setup_logging()
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_SAFETENSORS_SHA_FILE.touch(exist_ok=True)
    SHARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _gpu_ids = parse_gpu_ids()
    log.info(
        "Pair eval server starting; arch=%s gpus=%s model_cache=%s shard_cache=%s records=%s",
        chain_config.ARCH_MODULE,
        _gpu_ids,
        MODEL_CACHE_DIR,
        SHARD_CACHE_DIR,
        EVAL_RECORD_DIR,
    )
    yield
    if _model_worker_pool is not None:
        _model_worker_pool.close()
        _model_worker_pool = None
    log.info("Pair eval server shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "arch": chain_config.ARCH_MODULE,
        "gpu_ids": _gpu_ids,
        "king_loaded": _king_key,
        "protocol_version": PROTOCOL_VERSION,
        "versions": {
            "evaluator": EVALUATOR_VERSION,
            "evaluation_policy": EVALUATION_POLICY_VERSION,
            "code": EVALUATOR_CODE_VERSION or None,
        },
        "request_features": {
            "challenger_futility_early_stopping": "observed-quantile-v1",
            "flash_attention_4": "native-asymmetric-value-head-dim-v1",
        },
        "active_evals": _attempts.active_count(),
        "cache_dir": str(MODEL_CACHE_DIR),
        "shard_cache_dir": str(SHARD_CACHE_DIR),
        "record_dir": str(EVAL_RECORD_DIR),
        "kernel_cache_dir": str(KERNEL_CACHE_DIR),
        "model_worker_pool": _model_worker_pool.status() if _model_worker_pool is not None else {"running": False, "workers": []},
        "safetensors_reuse": {
            "history_file": str(COMPLETED_SAFETENSORS_SHA_FILE),
            "max_completed_evals": MAX_COMPLETED_EVALS_PER_SAFETENSORS_SHA,
        },
        "defaults": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "alpha": DEFAULT_ALPHA,
            "seq_len": DEFAULT_SEQ_LEN,
            "attn_implementation": DEFAULT_ATTN_IMPLEMENTATION,
            "gpus_per_model_instance": GPUS_PER_MODEL_INSTANCE,
            "model_instances_per_side": MODEL_INSTANCES_PER_SIDE,
            "worker_processes": MODEL_WORKER_PROCESSES,
            "tensor_parallel_size": 1,
            "model_parallel_strategy": "layer_sharding",
            "grouped_moe": True,
            "persistent_model_workers": True,
            "direct_checkpoint_to_gpu": True,
            "sequence_pipeline_depth": 2,
            "loss_cache": False,
            "use_cache": False,
            "dtype": "bfloat16",
            "n": DEFAULT_N,
            "n_bootstrap": DEFAULT_BOOTSTRAP_B,
        },
        "caps": {
            "eval_n_cap": EVAL_N_CAP,
            "eval_bootstrap_b_cap": EVAL_BOOTSTRAP_B_CAP,
            "eval_batch_size_cap": MAX_BATCH_SIZE,
            "eval_max_runtime_s": EVAL_MAX_RUNTIME_S,
        },
    }


@app.post("/eval")
async def start_eval(payload: dict):
    try:
        request = EvaluationRequestV2.from_mapping(payload)
        validate_protocol_versions(request)
    except ProtocolValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_protocol_v2_request", "message": str(exc)},
        ) from exc
    try:
        record, duplicate = _attempts.start(
            request,
            created_at=time.time(),
            admit_new=lambda: _eval_lock.acquire(blocking=False),
        )
    except AttemptConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "attempt_conflict", "message": str(exc)},
        ) from exc
    except AttemptBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "evaluator_busy", "message": str(exc)},
        ) from exc

    if not duplicate:
        threading.Thread(
            target=run_eval,
            args=(request.eval_id, request),
            daemon=True,
            name=f"eval-{request.eval_id}",
        ).start()
    return record.response(duplicate=duplicate)


@app.get("/eval/{eval_id}")
async def get_eval(eval_id: str):
    rec = _attempts.get(eval_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="eval not found")
    return {
        **rec.response(),
        "progress": rec.progress,
        "verdict": rec.verdict,
        "error": rec.error,
        "reason": rec.reason,
        "error_code": rec.error_code,
    }


@app.get("/eval/{eval_id}/stream")
async def stream_eval(eval_id: str):
    rec = _attempts.get(eval_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="eval not found")
    event_q: Queue = rec.events

    async def generate():
        while True:
            try:
                event = event_q.get(block=False)
            except Empty:
                await asyncio.sleep(0.5)
                if rec.state in ("completed", "failed") and event_q.empty():
                    final = rec.verdict or {
                        "code": rec.error_code or "evaluation_failed",
                        "error": rec.error,
                        "reason": rec.reason or rec.error,
                    }
                    final_type = "verdict" if rec.state == "completed" else "error"
                    yield f"data: {json.dumps(rec.event(final_type, final))}\n\n"
                    break
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("verdict", "error"):
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    setup_logging()
    host = os.environ.get("EVAL_HOST", "127.0.0.1")
    port = int(os.environ.get("EVAL_PORT", "9000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
