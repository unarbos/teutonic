#!/usr/bin/env python3
"""Paired MiMo model engine for packed-text, Horizon-reference evaluation.

- protocol-v3 R2 artifact refs are verified and materialized into a local cache,
- the king model is cached across evals,
- each eval has an id, status endpoint, SSE stream, phase/progress events,
- final verdicts are written to disk as JSON audit artifacts.

It evaluates configured-chain checkpoints on pinned packed UTF-8 documents. Model
snapshots must be self-contained: their own config/custom code is used and
compared before any weights are loaded.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, asynccontextmanager, contextmanager
from datetime import UTC, datetime
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
from teutonic.evaluation.masking import (
    scored_token_counts,
)
from teutonic.evaluation.protocol_v3 import (
    DEFAULT_EVAL_BATCH_SIZE,
    MAX_BATCH_SIZE,
    PROTOCOL_VERSION,
    AttemptBusyError,
    AttemptConflictError,
    EvaluationAttemptRegistry,
    EvaluationRequestV3,
    ProtocolValidationError,
    result_provenance,
    validate_result_v3,
)
from teutonic.evaluation.reference_policy import POLICY_VERSION
from teutonic.evaluator import replica_loading, sequence_parallel
from teutonic.evaluator.cache_lock import cache_lock
from teutonic.evaluator.document_index import DocumentIndexDownloadError, RangeDownloadError
from teutonic.evaluator.document_masking import masked_model_inputs
from teutonic.evaluator.horizon import HorizonDownload
from teutonic.evaluator.module_cache import transformers_module_cache_lock
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
DEFAULT_BATCH_TOKEN_BUDGET = DEFAULT_BATCH_SIZE * 2048
DEFAULT_DELTA = float(os.environ.get("EVAL_DELTA", "0.0015"))
DEFAULT_BOOTSTRAP_B = int(os.environ.get("EVAL_BOOTSTRAP_B", "10000"))
SUPPORTED_ATTN_IMPLEMENTATIONS = ("eager", "flash_attention_4")
DEFAULT_ATTN_IMPLEMENTATION = os.environ.get("TEUTONIC_ATTN_IMPLEMENTATION", "eager")
if DEFAULT_ATTN_IMPLEMENTATION not in SUPPORTED_ATTN_IMPLEMENTATIONS:
    raise RuntimeError(
        "TEUTONIC_ATTN_IMPLEMENTATION must be one of "
        f"{SUPPORTED_ATTN_IMPLEMENTATIONS}, got {DEFAULT_ATTN_IMPLEMENTATION!r}"
    )
EVALUATOR_VERSION = os.environ.get("TEUTONIC_EVALUATOR_VERSION", "document-evaluator-v3")
EVALUATION_POLICY_VERSION = os.environ.get(
    "TEUTONIC_EVALUATION_POLICY_VERSION", POLICY_VERSION
)
if EVALUATION_POLICY_VERSION != POLICY_VERSION:
    raise RuntimeError(f"this evaluator requires policy {POLICY_VERSION!r}")
EVALUATOR_CODE_VERSION = os.environ.get("TEUTONIC_EVALUATOR_CODE_VERSION", "")

# Reject bootstrap work above the server cap without changing the requested policy.
EVAL_BOOTSTRAP_B_CAP = int(os.environ.get("EVAL_BOOTSTRAP_B_CAP", "999999"))

EVAL_MAX_RUNTIME_S = int(os.environ.get("EVAL_MAX_RUNTIME_S", "0"))
DEFAULT_LM_HEAD_CHUNK = int(os.environ.get("TEUTONIC_LM_HEAD_CHUNK", "1024"))
DEFAULT_LOG_EVERY_BATCHES = int(os.environ.get("EVAL_LOG_EVERY_BATCHES", "1"))


class SafetensorsReuseLimitError(RuntimeError):
    """A challenger checkpoint has exhausted its allowed completed evaluations."""


DEFAULT_MODEL_DEVICE_MAP = os.environ.get("TEUTONIC_MODEL_DEVICE_MAP", "auto")
DEFAULT_GPU_MEMORY_FRACTION = float(os.environ.get("TEUTONIC_GPU_MEMORY_FRACTION", "0.45"))
GPUS_PER_MODEL_INSTANCE = 1
MODEL_INSTANCES_PER_SIDE = 4
MODEL_WORKER_PROCESSES = MODEL_INSTANCES_PER_SIDE * 2
KERNEL_CACHE_DIR = Path(
    os.environ.get("TEUTONIC_KERNEL_CACHE_DIR", "/tmp/teutonic/kernel_cache")
)
MODEL_LOADER_VERSION = "load-once-broadcast-v2"
CACHE_HIGH_WATERMARK_GB = float(os.environ.get("MODEL_CACHE_HIGH_WATERMARK_GB", "500"))

_eval_lock = threading.Lock()
_horizon_downloader = None
_idle_horizon = None
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
    dataset_source: Literal["packed_text"] = "packed_text"
    packed_dataset: dict[str, Any] = Field(default_factory=dict)
    batch_token_budget: int = Field(default=DEFAULT_BATCH_TOKEN_BUDGET, ge=2, le=1_000_000)
    vocab_size: int = 0
    long_documents: dict[str, Any] | None = None
    max_document_tokens: int = 0
    attn_implementation: Literal["eager", "flash_attention_4"] = DEFAULT_ATTN_IMPLEMENTATION
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=MAX_BATCH_SIZE)
    alpha: float = DEFAULT_ALPHA
    delta_threshold: float = DEFAULT_DELTA
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B
    seed: int = 0xE1A
    bootstrap_seed: int = 0xB007
    lm_head_chunk: int = DEFAULT_LM_HEAD_CHUNK
    log_every_batches: int = DEFAULT_LOG_EVERY_BATCHES
    model_device_map: str = DEFAULT_MODEL_DEVICE_MAP
    gpu_memory_fraction: float = DEFAULT_GPU_MEMORY_FRACTION
    parallel_models: Literal[True] = True
    parallel_batch_size: Literal[1] = DEFAULT_PARALLEL_BATCH_SIZE


def internal_request_from_v3(
    request: EvaluationRequestV3,
    king_snapshot: str,
    challenger_snapshot: str,
) -> EvalRequest:
    """Adapt protocol v3 to the scoring engine's internal request."""
    return EvalRequest(
        king_repo=king_snapshot,
        challenger_repo=challenger_snapshot,
        hotkey=str(request.miner["hotkey"]),
        coldkey=str(request.miner["coldkey"]),
        dataset_source="packed_text",
        packed_dataset=dict(request.dataset),
        batch_token_budget=int(request.limits["batch_token_budget"]),
        n_bootstrap=int(request.limits["n_bootstrap"]),
        alpha=float(request.limits["alpha"]),
        delta_threshold=float(request.limits["delta_threshold"]),
        batch_size=int(request.limits["batch_size"]),
        seed=int(request.sampling["seed"]),
        bootstrap_seed=int(request.sampling["bootstrap_seed"]),
        block_hash=str(request.sampling["block_hash"]),

    )


def validate_protocol_versions(request: EvaluationRequestV3) -> None:
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


def configure_deterministic_execution() -> dict:
    """Configure before CUDA model loading; unsupported operations must fail."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


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
    """Give each duel side four independent, single-GPU model replicas."""
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or any(gpu < 0 for gpu in gpu_ids):
        raise RuntimeError(
            f"MiMo duel requires exactly 8 GPUs with distinct non-negative IDs, got {gpu_ids}"
        )
    return [
        {"worker_id": f"{role}-{replica}", "role": role, "gpu_ids": [gpu_ids[offset + replica]]}
        for role, offset in (("king", 0), ("challenger", MODEL_INSTANCES_PER_SIDE))
        for replica in range(MODEL_INSTANCES_PER_SIDE)
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
    del eval_id
    if not 1 <= req.n_bootstrap <= EVAL_BOOTSTRAP_B_CAP:
        raise ValueError(f"n_bootstrap must be in [1, {EVAL_BOOTSTRAP_B_CAP}]")
    return {
        "requested_n_bootstrap": req.n_bootstrap,
        "effective_n_bootstrap": req.n_bootstrap,
        "n_bootstrap_cap": EVAL_BOOTSTRAP_B_CAP,
        "capped": False,
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
        "scoring_policy": POLICY_VERSION,
        "document_only": req.dataset_source == "packed_text",
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
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
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
        with transformers_module_cache_lock():
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
    # These buffers scale with tokens * routed experts. Release each stage
    # before allocating the next, especially the fp32 weighted outputs below.
    del routed_inputs
    activated = [module.experts[0].act_fn(gate) * up for gate, up in zip(gate_outputs, up_outputs)]
    del gate_outputs, up_outputs
    down_outputs = torch._foreach_mm(
        activated,
        [expert.down_proj.weight.T for expert in experts],
    )
    del activated

    routed_outputs = torch.cat(list(down_outputs), dim=0)
    del down_outputs
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


def create_empty_eval_model(config, label: str, on_phase=None):
    """Construct the same meta model for checkpoint readers and broadcast receivers."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM
    from transformers.initialization import no_init_weights

    config.use_cache = False
    # Remote MiMo advertises a stale FA4 capability flag; select FA4 after construction.
    config._attn_implementation = "eager"
    dtype = torch.bfloat16
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        if on_phase:
            on_phase({"phase": f"{label}_init_start", "dtype": str(dtype)})
        # Checkpoint loading or broadcast will populate every parameter. Skip
        # random initialization, which is prohibitively expensive for 104B models.
        with transformers_module_cache_lock(), init_empty_weights(), no_init_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=dtype)
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
    # checkpoint loading or broadcast resolves the tensors.
    model.tie_weights()
    return model


def finalize_eval_model(model, req: EvalRequest):
    if getattr(model.config, "model_type", "") == "mimo_v2":
        enable_grouped_mimo_moe(model)
    model.eval()
    model.config.use_cache = False
    model.config._attn_implementation = req.attn_implementation
    return model


def allocate_eval_replica(config, device: str, label: str, req: EvalRequest, on_phase=None):
    """Allocate destinations without reading checkpoint weights or initializing them."""
    model = create_empty_eval_model(config, label, on_phase)
    model.to_empty(device=device)
    # Moving meta parameters can break embedding/head aliases again.
    model.tie_weights()
    return finalize_eval_model(model, req)


def load_eval_model(snapshot_dir: str, config, device: str, label: str, req: EvalRequest, gpu_ids: list[int] | None = None, on_phase=None):
    configure_deterministic_execution()
    from accelerate import load_checkpoint_and_dispatch
    from accelerate.utils import modeling as accelerate_modeling

    if on_phase:
        on_phase({"phase": f"{label}_load_start", "device": device, "snapshot": snapshot_dir})
    t0 = time.time()
    dtype = torch.bfloat16
    effective_ids = list(gpu_ids if gpu_ids is not None else _gpu_ids)
    reject_mtp_checkpoint_weights(snapshot_dir)
    model = create_empty_eval_model(config, label, on_phase)
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
    finalize_eval_model(model, req)
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


class ScoringOOM(RuntimeError):
    def __init__(self, message, phase):
        super().__init__(message)
        self.phase = phase


@torch.no_grad()
def compute_per_sequence_loss(
    model,
    token_batches: list[list[int]],
    chunk_size: int,
    *,
    reset_peak_memory: bool = True,
    document_only: bool = False,
) -> list[float]:
    if not token_batches:
        return []
    counts = [len(row) - 1 for row in token_batches] if document_only else scored_token_counts(token_batches)
    input_device = model_input_device(model)
    cuda_devices = model_cuda_devices(model)
    if reset_peak_memory:
        for device in cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
    lengths = [len(row) for row in token_batches]
    packed = len(set(lengths)) > 1
    rows = [[token for row in token_batches for token in row]] if packed else token_batches
    phase = "input"
    try:
        input_ids = torch.tensor(rows, dtype=torch.long, device=input_device)
        phase = "forward"
        if hasattr(model, "reset_state"):
            model.reset_state()
        with masked_model_inputs(model, input_ids, lengths if packed else None, document_only=document_only) as (layout, masked_inputs):
            hidden = model.model(input_ids, use_cache=False, **masked_inputs).last_hidden_state
        head_dev = lm_head_device(model)
        if hidden.device != head_dev:
            hidden = hidden.to(head_dev)
        labels_full = input_ids if input_ids.device == head_dev else input_ids.to(head_dev)

        batch = len(rows)
        n_pos = labels_full.size(1) - 1
        per_token_losses = []
        phase = "lm_head"
        for start in range(0, n_pos, chunk_size):
            end = min(start + chunk_size, n_pos)
            logits = model.lm_head(hidden[:, start:end, :])
            labels = labels_full[:, start + 1 : end + 1]
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            )
            per_token_losses.append(loss.reshape(batch, -1).float())
            del logits
        # Summing once after concatenation makes the accumulation order independent
        # of lm_head_chunk, so increasing the projection chunk does not alter scores.
        valid_targets = layout.valid_targets.to(head_dev)
        losses = torch.cat(per_token_losses, dim=1).masked_fill(~valid_targets, 0)
        total = sum_sample_losses(losses, lengths if packed else None)
        result = (total / torch.tensor(counts, device=head_dev)).float().cpu().tolist()
        peaks = {
            str(device): round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
            for device in cuda_devices
        }
        eval_log.debug(
            "masked scoring memory | devices=%s seq_len=%d peak_allocated_gib=%s",
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
        raise ScoringOOM(
            f"OOM during {phase}: {len(token_batches)} samples, "
            f"{sum(lengths)} input tokens, longest={max(lengths)}; {peak_detail}",
            phase=phase,
        ) from exc


def sum_sample_losses(losses, packed_lengths=None):
    if packed_lengths is None:
        return losses.sum(dim=1)
    cuts = np.cumsum(packed_lengths[:-1]).tolist()
    return torch.stack([part.sum() for part in torch.tensor_split(losses.flatten(), cuts)])


def compute_with_oom_backoff(model, token_batches, chunk_size, *, gpu_ids, enabled, retries, document_only=False):
    """Change batching/projection workspace only; never change sampled content."""
    try:
        return compute_per_sequence_loss(model, token_batches, chunk_size, reset_peak_memory=False, document_only=document_only)
    except ScoringOOM as exc:
        if not enabled:
            raise
        if len(token_batches) == 1 and (exc.phase != "lm_head" or chunk_size <= 64):
            raise RuntimeError(
                f"a complete {len(token_batches[0])}-token sample cannot fit on this worker; "
                "evaluation stopped without truncation, skipping or backend fallback. "
                f"{exc}"
            ) from exc
        failure = str(exc)
    # Outside the exception handler: release the traceback's live tensors before
    # clearing unused allocator blocks and retrying the exact same token sequences.
    empty_worker_cuda_cache(gpu_ids)
    retry = {"n_samples": len(token_batches), "input_tokens": sum(map(len, token_batches)),
             "lm_head_chunk": chunk_size, "action": "split_batch" if len(token_batches) > 1 else "halve_lm_head_chunk"}
    retries.append(retry)
    eval_log.warning("scoring OOM retry | %s | %s", retry, failure)
    if len(token_batches) > 1:
        midpoint = len(token_batches) // 2
        return (
            compute_with_oom_backoff(model, token_batches[:midpoint], chunk_size, gpu_ids=gpu_ids, enabled=True, retries=retries, document_only=document_only)
            + compute_with_oom_backoff(model, token_batches[midpoint:], chunk_size, gpu_ids=gpu_ids, enabled=True, retries=retries, document_only=document_only)
        )
    return compute_with_oom_backoff(model, token_batches, max(64, chunk_size // 2), gpu_ids=gpu_ids, enabled=True, retries=retries, document_only=document_only)


class ModelSequenceScorer:
    """Score one batch at a time on a worker's dedicated GPU."""

    def __init__(self, model, req: EvalRequest, spec: dict, result_queue, generation: str):
        self.model = model
        self.req = req
        self.spec = spec
        self.result_queue = result_queue
        self.generation = generation
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=spec["worker_id"])

    @property
    def depth(self) -> int:
        return 1

    def submit(self, sequence_index: int, token_ids: list[int]) -> None:
        self.submit_batch([sequence_index], [token_ids])

    def submit_batch(
        self,
        sequence_indices: list[int],
        token_batches: list[list[int]],
    ) -> None:
        if not sequence_indices or len(sequence_indices) != len(token_batches):
            raise ValueError("sequence indices and token batches must be non-empty and aligned")
        self._executor.submit(self._score, sequence_indices, token_batches)

    def submit_parallel(self, sequence_index, tokens, rank):
        self._executor.submit(self._score, [sequence_index], [tokens], rank)

    def _score(
        self,
        sequence_indices: list[int],
        token_batches: list[list[int]],
        parallel_rank: int | None = None,
    ) -> None:
        started = time.time()
        try:
            retries = []
            if parallel_rank is not None:
                empty_worker_cuda_cache(self.spec["gpu_ids"])
                reset_worker_peak_memory(self.spec["gpu_ids"])
                losses = [sequence_parallel.compute_loss(
                    self.model, token_batches[0], self.req.lm_head_chunk, parallel_rank,
                )]
            else:
                losses = compute_with_oom_backoff(
                    self.model,
                    token_batches,
                    self.req.lm_head_chunk,
                    gpu_ids=self.spec["gpu_ids"],
                    enabled=self.req.long_documents is not None,
                    retries=retries,
                    document_only=self.req.dataset_source == "packed_text",
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
                "oom_retries": retries,
                "scoring_mode": "sequence_parallel" if parallel_rank is not None else "single_gpu",
                "peak_allocated_gib": {
                    str(gpu_id): torch.cuda.max_memory_allocated(gpu_id) / 1024 ** 3
                    for gpu_id in self.spec["gpu_ids"]
                } if torch.cuda.is_available() else {},
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

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


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
    """Own one persistent single-GPU model instance and accept load/score commands."""
    setup_logging()
    worker_id = spec["worker_id"]
    role = spec["role"]
    gpu_ids = list(spec["gpu_ids"])
    model = None
    pipeline = None
    loaded_key = None
    parallel_rank = None
    try:
        execution = configure_deterministic_execution()
        torch.cuda.set_device(gpu_ids[0])
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
                    # Loading uses temporary per-model groups. Retire any persistent
                    # sequence-parallel pair before joining a different rank topology.
                    if parallel_rank is not None:
                        torch.distributed.destroy_process_group()
                        parallel_rank = None
                    load_started = time.monotonic()

                    def on_load_phase(event):
                        result_queue.put({"type": "load_progress", "generation": generation,
                                          "worker_id": worker_id, "role": role, **event})

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
                        if command["replica_rank"] == 0:
                            model = load_eval_model(
                                command["snapshot"], config, f"cuda:{gpu_ids[0]}",
                                worker_id, req, gpu_ids=gpu_ids, on_phase=on_load_phase,
                            )
                        else:
                            model = allocate_eval_replica(
                                config, f"cuda:{gpu_ids[0]}", worker_id, req,
                                on_phase=on_load_phase,
                            )
                    else:
                        artifacts = command.get("artifacts", {})
                        attention = command.get("attention", {})
                        kernel_cache = command.get("kernel_cache", {})
                    torch.cuda.synchronize(gpu_ids[0])
                    prepare_s = time.monotonic() - load_started
                    result_queue.put({
                        "type": "load_prepared", "generation": generation,
                        "worker_id": worker_id, "role": role, "reused_model": reused,
                        "prepare_s": prepare_s,
                    })
                    # Wait on the CPU, not inside a GPU collective during checkpoint I/O.
                    transfer = command_queue.get()
                    if (transfer.get("type") != "load_broadcast"
                            or transfer.get("generation") != generation):
                        raise RuntimeError("expected matching model broadcast command")
                    broadcast = None
                    if transfer["required"]:
                        broadcast = replica_loading.broadcast_model(
                            model, rank=command["replica_rank"],
                            world_size=command["replica_count"],
                            rendezvous=transfer["rendezvous"], gpu_id=gpu_ids[0],
                        )
                    # A receiver is reusable only after its weights arrived successfully.
                    loaded_key = command["model_key"]
                    reset_worker_peak_memory(gpu_ids)
                    pipeline = ModelSequenceScorer(model, req, spec, result_queue, generation)
                    result_queue.put({
                        "type": "ready",
                        "generation": generation,
                        "worker_id": worker_id,
                        "role": role,
                        "gpu_ids": gpu_ids,
                        "pid": os.getpid(),
                        "snapshot": command["snapshot"],
                        "reused_model": reused,
                        "model_key": loaded_key,
                        "load_timing": {"prepare_s": prepare_s, "broadcast": broadcast,
                                        "ready_s": time.monotonic() - load_started},
                        "artifacts": artifacts,
                        "attention": attention,
                        "kernel_cache": kernel_cache,
                        "execution": execution,
                        "sequence_parallel_rank": parallel_rank,
                        "pipeline_depth": pipeline.depth,
                        "pipeline_boundary_layer": None,
                    })
                except BaseException as exc:
                    loaded_key = None
                    result_queue.put({
                        "type": "error",
                        "generation": generation,
                        "worker_id": worker_id,
                        "role": role,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
            elif command["type"] == "sequence_parallel_init":
                if pipeline is None or command["generation"] != pipeline.generation:
                    raise RuntimeError("sequence-parallel initialization requires a matching loaded model")
                pipeline.close()
                empty_worker_cuda_cache(gpu_ids)
                sequence_parallel.initialize(command["rank"], command["rendezvous"], gpu_ids[0])
                parallel_rank = command["rank"]
                pipeline = ModelSequenceScorer(model, req, spec, result_queue, command["generation"])
                result_queue.put({"type": "sequence_parallel_ready", "generation": command["generation"],
                                  "worker_id": worker_id, "role": role, "rank": parallel_rank})
            elif command["type"] in ("score", "score_parallel"):
                if pipeline is None or command["generation"] != pipeline.generation:
                    raise RuntimeError(f"{worker_id} received score command before matching load")
                if command["type"] == "score_parallel":
                    if parallel_rank is None or len(command["sequence_indices"]) != 1:
                        raise RuntimeError("invalid sequence-parallel scoring command")
                    pipeline.submit_parallel(command["sequence_indices"][0], command["token_batches"][0], parallel_rank)
                else:
                    pipeline.submit_batch(command["sequence_indices"], command["token_batches"])
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


def scoring_batch_end(sequences, start, batch_size, token_budget):
    """Bound whole-document batches by count, input tokens, and attention work.

    A single whole document can exceed the token budget; it is scored alone.
    FA4 work grows quadratically within each document, so also limit the sum of
    squared lengths to the baseline batch's full-attention work.
    """
    attention_budget = token_budget ** 2 // batch_size
    tokens = attention = 0
    end = start
    while end < min(start + batch_size, len(sequences)):
        length = len(sequences[end])
        if end > start and (tokens + length > token_budget or attention + length ** 2 > attention_budget):
            break
        tokens += length
        attention += length ** 2
        end += 1
    return end


class PersistentModelWorkerPool:
    """Keep eight model processes alive and reload only when checkpoint identity changes."""

    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = list(gpu_ids)
        self.specs = model_worker_specs(self.gpu_ids)
        context = mp.get_context("spawn")
        self.result_queue = context.Queue()
        self.command_queues = {}
        self.processes = {}
        self.ready: dict[str, dict] = {}
        self.parallel_pairs = None
        self.parallel_directory = None
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
        # Pair groups are retired by each worker before preparation. Loading groups
        # exist only for this generation, leaving sequence-parallel scoring independent.
        self.parallel_pairs = None
        if self.parallel_directory is not None:
            self.parallel_directory.cleanup()
            self.parallel_directory = None
        with tempfile.TemporaryDirectory(prefix="teutonic-model-broadcast-") as directory:
            try:
                return self._load_models(req, king_snapshot, challenger_snapshot,
                                         on_progress, directory)
            except BaseException:
                self.ready = {}
                self.close(force=True)
                raise

    def _load_models(self, req, king_snapshot, challenger_snapshot, on_progress, directory):
        generation = uuid.uuid4().hex
        request_data = req.model_dump()
        groups = {}
        for spec in self.specs:
            groups.setdefault(spec["role"], []).append(spec["worker_id"])
        for spec in self.specs:
            snapshot = king_snapshot if spec["role"] == "king" else challenger_snapshot
            worker_id = spec["worker_id"]
            previous = self.ready.get(worker_id, {})
            self.command_queues[worker_id].put({
                "type": "load",
                "generation": generation,
                "replica_rank": groups[spec["role"]].index(worker_id),
                "replica_count": len(groups[spec["role"]]),
                "snapshot": snapshot,
                "request": request_data,
                "model_key": checkpoint_load_key(snapshot, req, spec["gpu_ids"]),
                "artifacts": previous.get("artifacts", {}),
                "attention": previous.get("attention", {}),
                "kernel_cache": previous.get("kernel_cache", {}),
            })

        self.ready = {}
        prepared = {}
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
            worker_id = message.get("worker_id")
            if worker_id not in self.command_queues:
                raise RuntimeError(f"unknown model worker during load: {message}")
            if message["type"] == "load_progress":
                on_progress({k: v for k, v in message.items() if k not in ("type", "generation")})
                continue
            if message["type"] == "load_prepared":
                if worker_id in prepared:
                    raise RuntimeError(f"duplicate model preparation: {worker_id}")
                prepared[worker_id] = message
                role = next(s["role"] for s in self.specs if s["worker_id"] == worker_id)
                members = groups[role]
                on_progress({"phase": "model_worker_prepared", "worker_id": worker_id,
                             "role": role, "reused_model": message["reused_model"],
                             "prepare_s": message["prepare_s"]})
                if all(member in prepared for member in members):
                    required = len(members) > 1 and any(
                        not prepared[member]["reused_model"] for member in members
                    )
                    on_progress({"phase": "model_broadcast_start", "role": role,
                                 "source_worker": members[0], "replicas": len(members),
                                 "required": required})
                    for member in members:
                        self.command_queues[member].put({
                            "type": "load_broadcast", "generation": generation,
                            "required": required,
                            "rendezvous": (Path(directory) / role).as_uri(),
                        })
                continue
            if worker_id not in prepared or worker_id in ready:
                raise RuntimeError(f"unexpected model worker readiness: {message}")
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
                "load_timing": message["load_timing"],
                "ready": len(ready),
                "total_workers": len(self.specs),
            })
        self.ready = ready
        return generation

    def preflight_longest_document(self, sequences, generation, on_progress, indices=None):
        """Check the longest sampled document on every replica before full scoring."""
        index = max(range(len(sequences)) if indices is None else indices, key=lambda i: len(sequences[i]))
        length = len(sequences[index])
        previous = getattr(self, "_document_preflight", None)
        if previous is not None and previous[0] == generation and previous[1]["input_tokens"] >= length:
            # Streamed chunks share a model generation. Recheck only when a
            # longer document arrives; scoring every chunk's maximum on all
            # eight replicas would duplicate substantial inference work.
            return {**previous[1], "sequence_index": None, "reused": True}
        on_progress({"phase": "long_document_preflight_start", "sequence_index": index,
                     "input_tokens": length, "total_workers": len(self.specs)})
        pending = {s["worker_id"] for s in self.specs}
        for worker_id in sorted(pending):
            self.command_queues[worker_id].put({
                "type": "score", "generation": generation, "sequence_indices": [index],
                "token_batches": [sequences[index]],
            })
        results = []
        while pending:
            try:
                message = self.result_queue.get(timeout=30)
            except Empty:
                dead = self.dead_workers()
                if dead:
                    raise RuntimeError(f"workers exited during long-document preflight: {dead}")
                on_progress({"phase": "long_document_preflight_wait", "pending_workers": sorted(pending)})
                continue
            if message.get("generation") != generation or message.get("worker_id") not in pending:
                raise RuntimeError("unexpected worker message during long-document preflight")
            if message["type"] == "error":
                raise RuntimeError(f"long-document preflight failed on {message['worker_id']}: {message['error']}\n{message['traceback']}")
            if message["type"] != "result" or message.get("sequence_indices") != [index] or len(message.get("losses", [])) != 1:
                raise RuntimeError("invalid long-document preflight result")
            if not math.isfinite(message["losses"][0]):
                raise RuntimeError("non-finite loss during long-document preflight")
            pending.remove(message["worker_id"])
            results.append({k: message.get(k) for k in ("worker_id", "role", "wall_time_s", "peak_allocated_gib", "oom_retries")})
            on_progress({"phase": "long_document_preflight_passed", "worker_id": message["worker_id"],
                         "input_tokens": length, "remaining_workers": len(pending)})
        result = {"sequence_index": index, "input_tokens": length,
                  "workers": sorted(results, key=lambda r: r["worker_id"]), "reused": False}
        self._document_preflight = (generation, result)
        return result

    def score(
        self,
        sequences: list[list[int]],
        generation: str,
        req: EvalRequest,
        on_progress,
    ) -> tuple[list[float], list[float], dict]:
        counts = [len(row) - 1 for row in sequences]
        parallel_indices = [i for i, row in enumerate(sequences) if len(row) > sequence_parallel.SINGLE_GPU_MAX_TOKENS]
        parallel_losses = {"king": {}, "challenger": {}}
        parallel_meta = None
        if parallel_indices:
            parallel_losses, parallel_meta = sequence_parallel.score_documents(
                self, sequences, parallel_indices, generation, on_progress,
            )
        single_indices = [i for i, row in enumerate(sequences) if len(row) <= sequence_parallel.SINGLE_GPU_MAX_TOKENS]
        single_preflight = (
            self.preflight_longest_document(sequences, generation, on_progress, single_indices)
            if req.long_documents is not None and single_indices else None
        )
        preflight = parallel_meta["preflight"] if parallel_meta is not None else single_preflight
        oom_retries = []
        total_scored_tokens = sum(counts)
        king_losses: list[float | None] = [None] * len(sequences)
        challenger_losses: list[float | None] = [None] * len(sequences)
        for index, loss in parallel_losses["king"].items():
            king_losses[index] = loss
        for index, loss in parallel_losses["challenger"].items():
            challenger_losses[index] = loss
        next_index = {"king": 0, "challenger": 0}
        in_flight = {spec["worker_id"]: 0 for spec in self.specs}
        paired_done = 0
        while paired_done < len(sequences) and king_losses[paired_done] is not None and challenger_losses[paired_done] is not None:
            paired_done += 1
        def fill_worker(worker_id: str, role: str) -> None:
            depth = int(self.ready[worker_id].get("pipeline_depth", 1))
            while in_flight[worker_id] < depth and next_index[role] < len(sequences):
                target = king_losses if role == "king" else challenger_losses
                while next_index[role] < len(sequences) and target[next_index[role]] is not None:
                    next_index[role] += 1
                if next_index[role] == len(sequences):
                    break
                start = next_index[role]
                end = scoring_batch_end(sequences, start, req.batch_size, req.batch_token_budget)
                end = next((i for i in range(start, end) if target[i] is not None), end)
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

        while paired_done < len(sequences):
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
            oom_retries.extend({"worker_id": worker_id, **r} for r in message.get("oom_retries", []))
            in_flight[worker_id] -= 1
            indices = [int(value) for value in message["sequence_indices"]]
            losses = [float(value) for value in message["losses"]]
            if not indices or len(indices) != len(losses):
                raise RuntimeError(f"malformed batched result from {worker_id}")
            if not all(math.isfinite(loss) for loss in losses):
                raise RuntimeError(f"non-finite losses from {worker_id}")
            target = king_losses if role == "king" else challenger_losses
            for index, loss in zip(indices, losses, strict=True):
                if not 0 <= index < len(target):
                    raise RuntimeError(f"out-of-range {role} result for sequence {index}")
                if target[index] is not None:
                    raise RuntimeError(f"duplicate {role} result for sequence {index}")
                target[index] = loss

            previous_done = paired_done
            while (
                paired_done < len(sequences)
                and king_losses[paired_done] is not None
                and challenger_losses[paired_done] is not None
            ):
                paired_done += 1
            if paired_done != previous_done:
                on_progress({"phase": "document_scoring", "done": paired_done, "total": len(sequences)})
            fill_worker(worker_id, role)

        completed = len(sequences)
        return (
            [float(value) for value in king_losses[:completed]],
            [float(value) for value in challenger_losses[:completed]],
            {
                "workers": [self.ready[spec["worker_id"]] for spec in self.specs],
                "scored_tokens": counts[:completed],
                "planned_scored_tokens": total_scored_tokens,
                "long_document_preflight": preflight,
                "single_gpu_preflight": single_preflight,
                "sequence_parallel": parallel_meta,
                "oom_retries": oom_retries,
                "early_stop": None,
            },
        )

    def close(self, *, force: bool = False) -> None:
        if force:
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()
            for process in self.processes.values():
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
            for queue in [*self.command_queues.values(), self.result_queue]:
                try:
                    queue.close()
                except Exception:
                    pass
            if getattr(self, "parallel_directory", None) is not None:
                self.parallel_directory.cleanup()
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
        if getattr(self, "parallel_directory", None) is not None:
            self.parallel_directory.cleanup()

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


@contextmanager
def loading_model_workers(
    req: EvalRequest,
    king_snapshot: str,
    challenger_snapshot: str,
    gpu_ids: list[int],
    on_progress,
):
    """Load verified models while the caller prepares and validates the corpus."""
    global _model_worker_pool
    pool = get_model_worker_pool(gpu_ids)
    cancelled = threading.Event()

    def loading_progress(info):
        # The loading thread owns worker shutdown. Check between load events
        # (including the loader's periodic heartbeat) if corpus preparation failed.
        if cancelled.is_set():
            raise RuntimeError("model loading cancelled after evaluation setup failed")
        on_progress(info)

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-model-load") as executor:
            loading = executor.submit(
                pool.load_models,
                req.model_copy(deep=True),
                king_snapshot,
                challenger_snapshot,
                loading_progress,
            )
            try:
                yield pool, loading
            finally:
                # Join the loader before releasing the evaluation's cache locks.
                cancelled.set()
    except BaseException:
        pool.close(force=True)
        if _model_worker_pool is pool:
            _model_worker_pool = None
        raise


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
        # An unrecoverable failure must halt every queued sample. Batching-only
        # OOM retries happen inside each scorer; content and backend never change.
        pool.close(force=True)
        _model_worker_pool = None
        raise


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
    locks = ExitStack()
    try:
        if not MODEL_CACHE_DIR.exists():
            return
        if not locks.enter_context(cache_lock(MODEL_CACHE_DIR, cleanup=True)):
            log.info("cache cleanup skipped: evaluation is using the model cache")
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
    finally:
        locks.close()


def write_record(eval_id: str, payload: dict) -> tuple[str, str]:
    EVAL_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_RECORD_DIR / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{eval_id}.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    path.write_bytes(encoded)
    return str(path), hashlib.sha256(encoded).hexdigest()


def run_eval(eval_id: str, protocol_request: EvaluationRequestV3) -> None:
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
    started_at = datetime.now(UTC).isoformat()

    downloads = None
    progress_lock = threading.RLock()

    def on_phase(info: dict):
        with progress_lock:
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
                    previous.get("requested_sequences", 0),
                ),
            )
            completed = max(0, int(completed))
            requested = max(0, int(requested))
            normalized["completed_sequences"] = completed
            normalized["requested_sequences"] = requested
            normalized["percent"] = round(
                float(normalized.get("percent", (100.0 * completed / requested) if requested else 0.0)),
                4,
            )
            normalized["elapsed_seconds"] = round(max(time.time() - t0, 0.0), 1)
            normalized["early_stopped"] = bool(normalized.get("early_stopped", False))
            if downloads is not None:
                normalized["model_downloads"] = downloads.status()
            record.progress = normalized
            events.put(record.event("progress", normalized))

    heartbeat_stop = threading.Event()

    def heartbeat_loop():
        while not heartbeat_stop.wait(30):
            with progress_lock:
                current = dict(record.progress or {})
                current["heartbeat_at"] = time.time()
                on_phase(current or {"phase": "heartbeat"})

    threading.Thread(target=heartbeat_loop, daemon=True, name=f"heartbeat-{eval_id}").start()

    cache_locks = ExitStack()
    try:
        on_phase({"phase": "setup_start"})
        cache_locks.enter_context(cache_lock(MODEL_CACHE_DIR))
        cache_locks.enter_context(cache_lock(SHARD_CACHE_DIR))
        from teutonic.evaluator.document_duel import run_document_duel
        from teutonic.evaluator.model_downloads import ModelDownloads

        # Sampling and Horizon depend only on the pinned dataset/request, not on
        # MiMo weights. All checkpoint downloads remain inside the cache locks.
        req = internal_request_from_v3(protocol_request, "", "")
        limits_meta = apply_eval_limits(req, eval_id)
        on_phase({"phase": "limits_applied", **limits_meta})
        resolver = R2ArtifactResolver(MODEL_CACHE_DIR / "immutable-r2")
        downloads = cache_locks.enter_context(
            ModelDownloads(resolver, protocol_request.king, protocol_request.challenger)
        )
        prepared = {}

        def check_preparation():
            check_eval_runtime(t0)
            downloads.check()

        def prepare_models():
            waiting = time.monotonic()
            snapshots = downloads.result()
            prepared["model_download_wait_seconds"] = time.monotonic() - waiting
            validation_started = time.monotonic()
            king_snapshot, challenger_snapshot = snapshots["king"], snapshots["challenger"]
            req.king_repo, req.challenger_repo = str(king_snapshot), str(challenger_snapshot)
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
            req.max_document_tokens = min(int(king_config.max_position_embeddings), int(challenger_config.max_position_embeddings))
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
            attention_meta = {role: {"checkpoint_validation": report,
                                     "scoring_backend": "TRITON_ATTN_DIFFKV"}
                              for role, report in attention_meta.items()}

            prepared.update(king_snapshot=king_snapshot, challenger_snapshot=challenger_snapshot,
                            king_artifacts=king_artifacts, challenger_artifacts=challenger_artifacts,
                            duplicate_meta=duplicate_meta, attention_meta=attention_meta,
                            model_validation_seconds=time.monotonic() - validation_started)
            return king_snapshot, challenger_snapshot

        verdict, public_dataset_meta, worker_meta = run_document_duel(
            sys.modules[__name__], req, None, None, on_phase,
            prepare_models=prepare_models, check_preparation=check_preparation,
        )
        king_snapshot, challenger_snapshot = prepared["king_snapshot"], prepared["challenger_snapshot"]
        king_artifacts, challenger_artifacts = prepared["king_artifacts"], prepared["challenger_artifacts"]
        duplicate_meta, attention_meta = prepared["duplicate_meta"], prepared["attention_meta"]
        public_dataset_meta.update(
            model_download_seconds=downloads.elapsed_seconds,
            model_download_wait_seconds=prepared["model_download_wait_seconds"],
            model_validation_seconds=prepared["model_validation_seconds"],
        )
        worker_specs = model_worker_specs(_gpu_ids)
        king_device = "|".join(f"cuda:{gpu}" for gpu in _gpu_ids[:4])
        challenger_device = "|".join(f"cuda:{gpu}" for gpu in _gpu_ids[4:])
        use_parallel = True
        corpus_max_seq_len = public_dataset_meta["max_seq_len"]
        completed_at = datetime.now(UTC).isoformat()
        verdict.update({
            "eval_id": eval_id,
            "king_digest": protocol_request.king.expected_digest,
            "challenger_digest": protocol_request.challenger.expected_digest,
            "king_device": king_device,
            "challenger_device": challenger_device,
            "parallel_models": use_parallel,
            "gpu_memory_fraction": 0.8,
            "limits": limits_meta,
            "model_artifacts": {
                "king": king_artifacts,
                "challenger": challenger_artifacts,
                "attention": attention_meta,
                "duplicate_check": duplicate_meta,
                "workers": worker_meta,
            },
            "max_model_len": corpus_max_seq_len,
            "gpus_per_model_instance": 1,
            "model_instances_per_side": 4,
            "worker_processes": 8,
            "tensor_parallel_size": 1,
            "model_parallel_strategy": "replicas_with_large_document_tp2",
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
                requested_sequences=public_dataset_meta["n_documents"],
                completed_sequences=int(
                    verdict.get("n_sequences_evaluated", verdict.get("n_sequences", 0))
                ),
                early_stopped=bool(verdict.get("early_stopped", False)),
                hardware={
                    "gpu_ids": list(_gpu_ids),
                    "workers": [spec["worker_id"] for spec in worker_specs],
                    "model_parallel_strategy": "replicas_with_large_document_tp2",
                    "reference_tensor_parallel_size": 1,
                    "reference_pipeline_parallel_size": 8,
                    "duel_tensor_parallel_size": 1,
                },
            )
        )
        record_path, result_artifact_sha256 = write_record(
            eval_id,
            {"request": protocol_request.request_payload, "verdict": verdict},
        )
        verdict["record_path"] = record_path
        verdict["result_artifact_sha256"] = result_artifact_sha256
        validate_result_v3(verdict, protocol_request)
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
            else "dataset_index_download"
            if isinstance(exc, DocumentIndexDownloadError)
            else "dataset_shard_download"
            if isinstance(exc, RangeDownloadError)
            else "evaluator_interrupted"
            if isinstance(exc, subprocess.CalledProcessError) and exc.returncode in {-2, -15, 130, 143}
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
        cache_locks.close()
        try:
            cleanup_model_cache()
        finally:
            try:
                # MiMo workers have exited. Load the next reference in idle time;
                # this does not delay the terminal verdict or hold up admission.
                if _idle_horizon is not None:
                    _idle_horizon.preload()
            finally:
                try:
                    _eval_lock.release()
                except RuntimeError:
                    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu_ids, _model_worker_pool, _horizon_downloader, _idle_horizon
    setup_logging()
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_SAFETENSORS_SHA_FILE.touch(exist_ok=True)
    SHARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _gpu_ids = parse_gpu_ids()
    from teutonic.evaluator.horizon_pipeline_runtime import (
        runtime_identity as reference_runtime_identity,
    )
    from teutonic.evaluator.idle_horizon import IdleHorizon
    from teutonic.evaluator.vllm_runtime import runtime_identity
    runtime_identity()  # Dependency/plugin/code preflight, without loading GPU weights.
    reference_identity = reference_runtime_identity()
    if len(_gpu_ids) != 8 or len(set(_gpu_ids)) != 8:
        raise RuntimeError("document evaluation requires eight distinct GPUs")
    log.info(
        "Pair eval server starting; arch=%s gpus=%s model_cache=%s shard_cache=%s records=%s",
        chain_config.ARCH_MODULE,
        _gpu_ids,
        MODEL_CACHE_DIR,
        SHARD_CACHE_DIR,
        EVAL_RECORD_DIR,
    )
    _horizon_downloader = HorizonDownload.from_env(MODEL_CACHE_DIR)
    if _horizon_downloader is not None:
        _horizon_downloader.start()
        _idle_horizon = IdleHorizon(_horizon_downloader, _gpu_ids, reference_identity)
        _idle_horizon.preload()
    try:
        yield
    finally:
        from teutonic.evaluator.vllm_runtime import close_all_workers
        if _horizon_downloader is not None:
            _horizon_downloader.stop()
        if _idle_horizon is not None:
            _idle_horizon.close()
            _idle_horizon = None
        close_all_workers()
        if _horizon_downloader is not None:
            _horizon_downloader = None
        if _model_worker_pool is not None:
            _model_worker_pool.close()
            _model_worker_pool = None
        log.info("Pair eval server shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    from teutonic.evaluator.horizon_pipeline import BACKEND as reference_backend
    from teutonic.evaluator.vllm_runtime import worker_status
    from teutonic.evaluator.vllm_score import BACKEND as mimo_backend
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
            "whole_documents": "packed-text",
            "early_stopping": False,
            "reference_deficit": POLICY_VERSION,
            "scoring_backend": {"horizon": reference_backend, "mimo": mimo_backend},
            "mimo_attention": "TRITON_ATTN_DIFFKV",
            "document_masking": "whole-document-native-eos-no-bos",
        },
        "active_evals": _attempts.active_count(),
        "cache_dir": str(MODEL_CACHE_DIR),
        "shard_cache_dir": str(SHARD_CACHE_DIR),
        "horizon_download": (
            _horizon_downloader.status() if _horizon_downloader is not None
            else {"state": "disabled"}
        ),
        "horizon_gpu": _idle_horizon.status() if _idle_horizon is not None else {"state": "disabled"},
        "record_dir": str(EVAL_RECORD_DIR),
        "kernel_cache_dir": str(KERNEL_CACHE_DIR),
        "model_worker_pool": worker_status(),
        "safetensors_reuse": {
            "history_file": str(COMPLETED_SAFETENSORS_SHA_FILE),
            "max_completed_evals": MAX_COMPLETED_EVALS_PER_SAFETENSORS_SHA,
        },
        "defaults": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "alpha": DEFAULT_ALPHA,
            "batch_token_budget": DEFAULT_BATCH_TOKEN_BUDGET,
            "attn_implementation": "TRITON_ATTN_DIFFKV",
            "gpus_per_model_instance": 1,
            "model_instances_per_side": 4,
            "worker_processes": 8,
            "tensor_parallel_size": 1,
            "reference_tensor_parallel_size": 1,
            "reference_pipeline_parallel_size": 8,
            "model_parallel_strategy": "replicas_with_large_document_tp2",
            "grouped_moe": True,
            "persistent_model_workers": True,
            "persistent_mimo_workers": False,
            "idle_horizon_preload": True,
            "sequence_pipeline_depth": 2,
            "reference_loss_cache": True,
            "prefix_cache": False,
            "dtype": "bfloat16",
            "n_bootstrap": DEFAULT_BOOTSTRAP_B,
        },
        "caps": {
            "eval_bootstrap_b_cap": EVAL_BOOTSTRAP_B_CAP,
            "eval_batch_size_cap": MAX_BATCH_SIZE,
            "eval_max_runtime_s": EVAL_MAX_RUNTIME_S,
        },
    }


@app.post("/eval")
async def start_eval(payload: dict):
    try:
        request = EvaluationRequestV3.from_mapping(payload)
        validate_protocol_versions(request)
    except ProtocolValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_protocol_v3_request", "message": str(exc)},
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
        try:
            threading.Thread(
                target=run_eval,
                args=(request.eval_id, request),
                daemon=True,
                name=f"eval-{request.eval_id}",
            ).start()
        except Exception:
            record.state = "failed"
            record.error = record.reason = "evaluation worker could not start"
            record.error_code = "evaluator_interrupted"
            _eval_lock.release()
            raise
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
