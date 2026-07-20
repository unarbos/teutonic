#!/usr/bin/env python3
"""Quasar pair-eval server for local Teutonic-style smoke tests.

This keeps the useful shape of eval_server.py:

- model refs are materialized from Hippius Hub / Hugging Face into a local cache,
- the king model is cached across evals,
- each eval has an id, status endpoint, SSE stream, phase/progress events,
- final verdicts are written to disk as JSON audit artifacts.

It is intentionally narrower than the production validator eval server:
it evaluates Quasar checkpoints on local FineWeb-Edu token shards. Model
snapshots are treated as self-contained by default: their own config/custom code
is used and compared before any weights are loaded. ``code_model`` remains only
as an explicit debug fallback for older, weights-only snapshots.
"""
from __future__ import annotations

import ast
import asyncio
import glob
import gc
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import types
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

log = logging.getLogger("eval_server_quasar_pair")
eval_log = logging.getLogger("eval_torch")

MODEL_CACHE_DIR = Path(os.environ.get("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/quasar_pair_models"))
EVAL_RECORD_DIR = Path(os.environ.get("TEUTONIC_EVAL_RECORD_DIR", "/tmp/teutonic/quasar_pair_evals"))
SHARD_CACHE_DIR = Path(
    os.environ.get(
        "TEUTONIC_SHARD_CACHE_DIR",
        os.environ.get("TEUTONIC_PARQUET_CACHE_DIR", "/tmp/teutonic/finewebedu_shards"),
    )
)
DEFAULT_CODE_MODEL = os.environ.get("QUASAR_CODE_MODEL", "silx-ai/Quasar-10B")
DEFAULT_PARQUET_GLOB = os.environ.get("TEUTONIC_PARQUET_GLOB", "/root/data/fineweb-edu-10BT/sample/10BT/**/*.parquet")
DEFAULT_DATASET_SOURCE = os.environ.get("TEUTONIC_DATASET_SOURCE", "s3")
DEFAULT_S3_ENDPOINT = os.environ.get("TEUTONIC_DS_ENDPOINT", "https://s3.hippius.com")
DEFAULT_S3_BUCKET = os.environ.get("TEUTONIC_DS_BUCKET", "teutonic-sn3")
DEFAULT_S3_PREFIX = os.environ.get("TEUTONIC_DS_PREFIX", "dataset/finewebedu/")
DEFAULT_S3_AUTH_SOURCE = os.environ.get("TEUTONIC_DS_AUTH_SOURCE", "env")
DEFAULT_S3_SHARD_CONTAINS = os.environ.get("TEUTONIC_DS_SHARD_CONTAINS", "/shards/")
DEFAULT_S3_SHARD_SUFFIX = os.environ.get("TEUTONIC_DS_SHARD_SUFFIX", ".npy")
DEFAULT_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "512"))
# In parallel mode each model gets only n/2 GPUs → more layers per GPU → larger MLP intermediates.
# Use a smaller batch to avoid OOM.  512 is fine for 8-GPU sequential; 256 for 4-GPU parallel.
DEFAULT_PARALLEL_BATCH_SIZE = int(os.environ.get("EVAL_PARALLEL_BATCH_SIZE", "128"))
DEFAULT_ALPHA = float(os.environ.get("EVAL_ALPHA", "0.001"))
DEFAULT_SEQ_LEN = int(os.environ.get("EVAL_SEQ_LEN", "2048"))
DEFAULT_DELTA = float(os.environ.get("EVAL_DELTA", "0.0015"))
DEFAULT_BOOTSTRAP_B = int(os.environ.get("EVAL_BOOTSTRAP_B", "10000"))
DEFAULT_N_PUBLIC = int(os.environ.get("EVAL_N_PUBLIC", "20000"))
DEFAULT_N_PRIVATE = int(os.environ.get("EVAL_N_PRIVATE", "0"))
DEFAULT_N = int(os.environ.get("EVAL_N", str(DEFAULT_N_PUBLIC + DEFAULT_N_PRIVATE)))

# Server-side caps. The validator can request a larger eval_n / n_bootstrap
# in its POST body; we clamp to these to keep per-eval wall time bounded
# while clearing a backed-up duel queue. Restore via env if not needed.
EVAL_N_CAP = int(os.environ.get("EVAL_N_CAP", "20000"))
EVAL_BOOTSTRAP_B_CAP = int(os.environ.get("EVAL_BOOTSTRAP_B_CAP", "999999"))

PROBE_ENABLED = os.environ.get("TEUTONIC_PROBE_ENABLED", "1") == "1"

EVAL_MAX_RUNTIME_S = int(os.environ.get("EVAL_MAX_RUNTIME_S", "3000"))
DEFAULT_LM_HEAD_CHUNK = int(os.environ.get("TEUTONIC_LM_HEAD_CHUNK", "32"))
DEFAULT_LOG_EVERY_BATCHES = int(os.environ.get("EVAL_LOG_EVERY_BATCHES", "1"))
DEFAULT_MODEL_DEVICE_MAP = os.environ.get("TEUTONIC_MODEL_DEVICE_MAP", "auto")
DEFAULT_PARALLEL_MODELS = os.environ.get("TEUTONIC_PARALLEL_MODELS", "1") == "1"
DEFAULT_GPU_MEMORY_FRACTION = float(os.environ.get("TEUTONIC_GPU_MEMORY_FRACTION", "0.45"))
DEFAULT_MODEL_DOWNLOAD_RETRIES = int(os.environ.get("TEUTONIC_MODEL_DOWNLOAD_RETRIES", "3"))
DEFAULT_MODEL_DOWNLOAD_RETRY_BACKOFF_S = float(os.environ.get("TEUTONIC_MODEL_DOWNLOAD_RETRY_BACKOFF_S", "15"))
MODEL_ENCRYPTION_MANIFEST_NAME = "teutonic_encryption.json"
DEFAULT_MODEL_DECRYPTION_KEY = _repo_root / "keys" / "validator_model_decryption.key"
MODEL_DECRYPTION_KEY_MODE = 0o600

# Early stopping: abort once the challenger has no mathematical chance to win.
# After at least EARLY_STOP_MIN_FRACTION of sequences are evaluated, check
# whether (d_sum + remaining * d_max_observed) / n_total < delta_threshold.
# Since LCB <= mu_hat, if that upper bound on mu_hat is below the threshold the
# challenger cannot reach it regardless of the remaining samples.
EVAL_EARLY_STOP = os.environ.get("EVAL_EARLY_STOP", "1") == "1"
EVAL_EARLY_STOP_MIN_FRACTION = float(os.environ.get("EVAL_EARLY_STOP_MIN_FRACTION", "0.4"))
EVAL_EARLY_STOP_ADVANTAGE_QUANTILE = float(os.environ.get("EVAL_EARLY_STOP_ADVANTAGE_QUANTILE", "0.95"))
DEFAULT_MODEL_DOWNLOAD_WORKERS = int(os.environ.get("TEUTONIC_MODEL_DOWNLOAD_WORKERS", "4"))
DEFAULT_S3_DOWNLOAD_RETRIES = int(os.environ.get("TEUTONIC_S3_DOWNLOAD_RETRIES", "5"))
DEFAULT_S3_DOWNLOAD_RETRY_BACKOFF_S = float(os.environ.get("TEUTONIC_S3_DOWNLOAD_RETRY_BACKOFF_S", "20"))
DEFAULT_S3_CLIENT_MAX_ATTEMPTS = int(os.environ.get("TEUTONIC_S3_CLIENT_MAX_ATTEMPTS", "10"))
DEFAULT_S3_TRANSFER_ATTEMPTS = int(os.environ.get("TEUTONIC_S3_TRANSFER_ATTEMPTS", "10"))
DEFAULT_S3_TRANSFER_CONCURRENCY = int(os.environ.get("TEUTONIC_S3_TRANSFER_CONCURRENCY", "1"))
DEFAULT_S3_TRANSFER_CHUNK_MB = int(os.environ.get("TEUTONIC_S3_TRANSFER_CHUNK_MB", "64"))
DEFAULT_ALLOW_CODE_MODEL_FALLBACK = os.environ.get("TEUTONIC_ALLOW_CODE_MODEL_FALLBACK", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CACHE_HIGH_WATERMARK_GB = float(os.environ.get("MODEL_CACHE_HIGH_WATERMARK_GB", "500"))

MODEL_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.py",
    "tokenizer*",
    "special_tokens*",
    "*.model",
    "*.tiktoken",
    "merges.txt",
    "vocab.*",
    "*.txt",
    "*.jinja",
]

_eval_lock = threading.Lock()
_evals: dict[str, dict] = {}
_gpu_ids: list[int] = []
_king_model = None
_king_key: tuple[str, ...] | None = None
_king_device = ""
_king_gpu_ids: list[int] = []


class EvalRequest(BaseModel):
    king_repo: str
    challenger_repo: str
    king_digest: str = ""
    challenger_digest: str = ""
    code_model: str = DEFAULT_CODE_MODEL
    allow_code_model_fallback: bool = DEFAULT_ALLOW_CODE_MODEL_FALLBACK
    revision: str | None = None
    block_hash: str = ""
    hotkey: str = ""
    shard_key: str = ""
    dataset_source: str = DEFAULT_DATASET_SOURCE
    parquet_glob: list[str] = Field(default_factory=lambda: [DEFAULT_PARQUET_GLOB])
    s3_endpoint: str = DEFAULT_S3_ENDPOINT
    s3_bucket: str = DEFAULT_S3_BUCKET
    s3_prefix: str = DEFAULT_S3_PREFIX
    s3_auth_source: str = DEFAULT_S3_AUTH_SOURCE
    s3_shard_contains: str = DEFAULT_S3_SHARD_CONTAINS
    s3_shard_suffix: str = DEFAULT_S3_SHARD_SUFFIX
    s3_max_shards: int = 0
    s3_doppler_project: str = "arbos"
    s3_doppler_config: str = "dev"
    seq_len: int = DEFAULT_SEQ_LEN
    n: int | None = None
    n_public: int = DEFAULT_N_PUBLIC
    n_private: int = DEFAULT_N_PRIVATE
    batch_size: int = DEFAULT_BATCH_SIZE
    alpha: float = DEFAULT_ALPHA
    delta_threshold: float = DEFAULT_DELTA
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B
    seed: int = 0xE1A
    bootstrap_seed: int = 0xB007
    text_field: str = "text"
    min_chars: int = 128
    max_chars: int = 0
    parquet_batch_size: int = 1024
    lm_head_chunk: int = DEFAULT_LM_HEAD_CHUNK
    log_every_batches: int = DEFAULT_LOG_EVERY_BATCHES
    model_device_map: str = DEFAULT_MODEL_DEVICE_MAP
    gpu_memory_fraction: float = DEFAULT_GPU_MEMORY_FRACTION
    parallel_models: bool = DEFAULT_PARALLEL_MODELS
    parallel_batch_size: int = DEFAULT_PARALLEL_BATCH_SIZE


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


def device_plan(req: EvalRequest) -> str:
    if not torch.cuda.is_available() or not _gpu_ids:
        return "cpu"
    mode = (req.model_device_map or "auto").lower()
    if mode == "single" or len(_gpu_ids) == 1:
        return f"cuda:{_gpu_ids[0]}"
    if mode != "auto":
        raise ValueError("model_device_map must be one of: auto, single")
    return "auto"


def device_plan_for_gpus(gpu_ids: list[int]) -> str:
    if not torch.cuda.is_available() or not gpu_ids:
        return "cpu"
    if len(gpu_ids) == 1:
        return f"cuda:{gpu_ids[0]}"
    return "auto"


def normalize_model_ref(ref: str) -> str:
    ref = (ref or "").strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        parsed = urlparse(ref)
        parts = [p for p in parsed.path.split("/") if p]
        if parsed.netloc.endswith("huggingface.co") and len(parts) >= 2:
            return "/".join(parts[:2])
        if "hippius.com" in parsed.netloc and parts and parts[0] == "models" and len(parts) >= 3:
            return "/".join(parts[1:3])
    return ref


def cache_key(repo: str, digest: str | None) -> str:
    material = f"{repo}@{digest or 'latest'}".encode()
    return hashlib.sha256(material).hexdigest()[:24]


def dataset_seed_material(req: EvalRequest) -> str:
    block_hash = (req.block_hash or "").strip()
    hotkey = (req.hotkey or "").strip()
    if block_hash and block_hash != "default":
        return f"block_hash={block_hash}|hotkey={hotkey}|base_seed={req.seed}"
    return f"base_seed={req.seed}"


def dataset_seed(req: EvalRequest) -> int:
    digest = hashlib.blake2b(dataset_seed_material(req).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def apply_eval_limits(req: EvalRequest, eval_id: str = "") -> dict:
    requested_n = req.n
    if requested_n is None:
        requested_n = max(0, int(req.n_public)) + max(0, int(req.n_private))
    requested_n = max(0, int(requested_n))
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


def model_decryption_key() -> Path:
    configured = os.environ.get("TEUTONIC_MODEL_DECRYPTION_KEY")
    key = Path(configured).expanduser() if configured else DEFAULT_MODEL_DECRYPTION_KEY
    key = key.resolve()
    if not key.is_file() or not os.access(key, os.R_OK):
        raise RuntimeError(
            "encrypted model snapshot found, but no private key is available; "
            "set TEUTONIC_MODEL_DECRYPTION_KEY or create keys/validator_model_decryption.key"
        )
    return key


def model_decryption_key_available() -> bool:
    configured = os.environ.get("TEUTONIC_MODEL_DECRYPTION_KEY")
    key = Path(configured).expanduser() if configured else DEFAULT_MODEL_DECRYPTION_KEY
    return key.is_file() and os.access(key, os.R_OK)


def ensure_model_decryption_key_permissions() -> None:
    configured = os.environ.get("TEUTONIC_MODEL_DECRYPTION_KEY")
    key = Path(configured).expanduser() if configured else DEFAULT_MODEL_DECRYPTION_KEY
    if not key.exists():
        return
    try:
        mode = key.stat().st_mode & 0o777
        if mode != MODEL_DECRYPTION_KEY_MODE:
            key.chmod(MODEL_DECRYPTION_KEY_MODE)
            log.info("set model decryption key permissions to 0600: %s", key)
    except OSError as exc:
        log.warning("could not set model decryption key permissions for %s: %s", key, exc)
    if not model_decryption_key_available():
        log.warning(
            "model decryption key exists but is not readable by this process: %s",
            key,
        )


def load_encryption_manifest(snapshot: Path) -> dict | None:
    manifest_path = snapshot / MODEL_ENCRYPTION_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scheme") != "age-x25519":
        raise RuntimeError(f"unsupported model encryption scheme: {manifest.get('scheme')!r}")
    if not manifest.get("files"):
        raise RuntimeError(f"{MODEL_ENCRYPTION_MANIFEST_NAME} has no encrypted files")
    return manifest


def decrypted_snapshot_path(snapshot: Path) -> Path:
    return snapshot.with_name(snapshot.name + "-decrypted")


def decrypted_snapshot_is_current(output: Path, manifest: dict) -> bool:
    if not snapshot_has_required_files(output):
        return False
    for item in manifest.get("files", []):
        dst = output / item["path"]
        if not dst.is_file():
            return False
        if dst.stat().st_size != int(item.get("plain_size", -1)):
            return False
        if sha256_file(dst) != item.get("plain_sha256"):
            return False
    return True


def decrypt_model_snapshot(snapshot_dir: str, on_phase=None) -> str:
    snapshot = Path(snapshot_dir).resolve()
    manifest = load_encryption_manifest(snapshot)
    if manifest is None:
        return str(snapshot)

    output = decrypted_snapshot_path(snapshot)
    if decrypted_snapshot_is_current(output, manifest):
        return str(output)

    age = shutil.which("age")
    if age is None:
        raise RuntimeError("encrypted model snapshot found, but `age` is not installed on PATH")
    identity = model_decryption_key()

    if on_phase:
        on_phase({"phase": "model_decrypt_start", "source": str(snapshot), "output": str(output)})
    if output.exists():
        shutil.rmtree(output)

    encrypted = {item["path"]: item for item in manifest["files"]}
    for src in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        rel = str(src.relative_to(snapshot)).replace("\\", "/")
        if rel == MODEL_ENCRYPTION_MANIFEST_NAME:
            continue
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel in encrypted:
            subprocess.run([age, "-d", "-i", str(identity), "-o", str(dst), str(src)], check=True)
            item = encrypted[rel]
            if sha256_file(dst) != item.get("plain_sha256"):
                raise RuntimeError(f"{rel}: plaintext sha256 mismatch after decrypt")
            if dst.stat().st_size != int(item.get("plain_size", -1)):
                raise RuntimeError(f"{rel}: plaintext size mismatch after decrypt")
        elif src.name.endswith(".safetensors"):
            raise RuntimeError(f"{rel}: .safetensors file is missing from {MODEL_ENCRYPTION_MANIFEST_NAME}")
        else:
            shutil.copy2(src, dst)

    if on_phase:
        on_phase({"phase": "model_decrypt_done", "files": len(encrypted), "path": str(output)})
    return str(output)


def snapshot_safetensors_digest(snapshot_dir: str) -> str:
    path = Path(snapshot_dir)
    shard_names = snapshot_safetensor_names(snapshot_dir)
    if not shard_names:
        raise FileNotFoundError(f"no .safetensors files found in {snapshot_dir}")
    h = hashlib.sha256()
    for shard_name in shard_names:
        h.update(shard_name.encode("utf-8"))
        h.update(b"\0")
        with (path / shard_name).open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
    return h.hexdigest()


def reject_duplicate_safetensors(king_snapshot: str, challenger_snapshot: str, on_phase=None) -> dict:
    if on_phase:
        on_phase({"phase": "duplicate_check_start"})
    king_digest = snapshot_safetensors_digest(king_snapshot)
    challenger_digest = snapshot_safetensors_digest(challenger_snapshot)
    if king_digest == challenger_digest:
        raise RuntimeError(
            "challenger plaintext .safetensors are identical to the king; "
            "encrypted submissions are checked after decrypt"
        )
    meta = {
        "king_safetensors_sha256": king_digest,
        "challenger_safetensors_sha256": challenger_digest,
    }
    if on_phase:
        on_phase({"phase": "duplicate_check_done", **{k: v[:16] for k, v in meta.items()}})
    return meta


def materialize_model(repo_or_url: str, digest: str = "", on_phase=None) -> str:
    """Download or reuse a model snapshot from local path, HF, or Hippius."""
    repo = normalize_model_ref(repo_or_url)
    local = Path(repo)
    if local.exists():
        return decrypt_model_snapshot(str(local.resolve()), on_phase=on_phase)

    target = MODEL_CACHE_DIR / repo.replace("/", "--") / (digest.replace(":", "-") if digest else "latest")
    if target.exists() and snapshot_has_required_files(target):
        return decrypt_model_snapshot(str(target), on_phase=on_phase)
    if target.exists():
        shutil.rmtree(target)

    if on_phase:
        on_phase({"phase": "download_start", "repo": repo, "digest": digest or "latest"})

    def download_once() -> str:
        target.mkdir(parents=True, exist_ok=True)
        # if repo_or_url.startswith("http") and "huggingface.co" in repo_or_url:
        #     from huggingface_hub import snapshot_download

        #     revision = digest[3:] if digest.startswith("hf:") else (digest or None)
        #     return snapshot_download(
        #         repo_id=repo,
        #         revision=revision,
        #         local_dir=str(target),
        #         allow_patterns=MODEL_ALLOW_PATTERNS,
        #         max_workers=DEFAULT_MODEL_DOWNLOAD_WORKERS,
        #         token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
        #     )
        # if digest.startswith("hf:"):
        #     from huggingface_hub import snapshot_download

        #     return snapshot_download(
        #         repo_id=repo,
        #         revision=digest[3:],
        #         local_dir=str(target),
        #         allow_patterns=MODEL_ALLOW_PATTERNS,
        #         max_workers=DEFAULT_MODEL_DOWNLOAD_WORKERS,
        #         token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
        #     )

        from hippius_hub import snapshot_download
        from model_store import get_hub_token

        return snapshot_download(
            repo_id=repo,
            revision=digest or None,
            local_dir=str(target),
            allow_patterns=MODEL_ALLOW_PATTERNS,
            max_workers=DEFAULT_MODEL_DOWNLOAD_WORKERS,
            token=get_hub_token(),
        )

    attempts = max(1, DEFAULT_MODEL_DOWNLOAD_RETRIES)
    last_exc: Exception | None = None
    path = ""
    for attempt in range(1, attempts + 1):
        if target.exists():
            shutil.rmtree(target)
        try:
            if on_phase:
                on_phase({
                    "phase": "download_attempt",
                    "repo": repo,
                    "digest": digest or "latest",
                    "attempt": attempt,
                    "attempts": attempts,
                    "workers": DEFAULT_MODEL_DOWNLOAD_WORKERS,
                })
            path = download_once()
            if not snapshot_has_required_files(Path(path)):
                raise RuntimeError(f"downloaded snapshot is incomplete: {path}")
            path = decrypt_model_snapshot(path, on_phase=on_phase)
            break
        except Exception as exc:
            last_exc = exc
            log.warning(
                "model download failed for %s@%s attempt %d/%d: %s",
                repo,
                digest or "latest",
                attempt,
                attempts,
                exc,
                exc_info=True,
            )
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if attempt >= attempts:
                raise RuntimeError(
                    f"failed to download model {repo}@{digest or 'latest'} after {attempts} attempts: {exc}"
                ) from exc
            delay = DEFAULT_MODEL_DOWNLOAD_RETRY_BACKOFF_S * attempt
            if on_phase:
                on_phase({
                    "phase": "download_retry_wait",
                    "repo": repo,
                    "digest": digest or "latest",
                    "attempt": attempt,
                    "attempts": attempts,
                    "sleep_s": round(delay, 1),
                    "error": str(exc),
                })
            time.sleep(delay)
    else:
        raise RuntimeError(f"failed to download model {repo}@{digest or 'latest'}: {last_exc}")

    if on_phase:
        on_phase({"phase": "download_done", "repo": repo, "path": str(path)})
    return str(path)


def prepare_remote_code(model_id: str, revision: str | None) -> str:
    path = Path(model_id)
    if path.exists():
        code_dir = str(path.resolve())
    else:
        from huggingface_hub import snapshot_download

        code_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "*.py",
                "*.json",
                "tokenizer.*",
                "*.model",
                "*.tiktoken",
                "merges.txt",
                "vocab.*",
            ],
        )
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    return code_dir


def preflight_deps() -> None:
    missing = []
    for module_name, package_hint in (("causal_conv1d", "causal-conv1d"),):
        try:
            __import__(module_name)
        except Exception:
            missing.append((module_name, package_hint))
    if missing:
        installs = " ".join(pkg for _, pkg in missing)
        modules = ", ".join(mod for mod, _ in missing)
        raise RuntimeError(f"missing required module(s): {modules}; install with `pip install {installs}`")


def patch_transformers_masking_compat() -> None:
    try:
        import transformers.masking_utils as masking_utils
    except Exception:
        return
    fn = getattr(masking_utils, "create_causal_mask", None)
    if fn is None or getattr(fn, "_quasar_compat", False):
        return
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return
    if "cache_position" in params:
        return

    def create_causal_mask_compat(*args, **kwargs):
        cache_position = kwargs.pop("cache_position", None)
        past_key_values = kwargs.get("past_key_values")
        original_get_mask_sizes = None
        if cache_position is not None and hasattr(past_key_values, "get_mask_sizes"):
            original_get_mask_sizes = past_key_values.get_mask_sizes

            def get_mask_sizes_compat(self, query_length, layer_idx):
                try:
                    return original_get_mask_sizes(cache_position, layer_idx)
                except Exception:
                    return int(query_length), 0

            past_key_values.get_mask_sizes = types.MethodType(get_mask_sizes_compat, past_key_values)
        try:
            return fn(*args, **kwargs)
        finally:
            if original_get_mask_sizes is not None:
                past_key_values.get_mask_sizes = original_get_mask_sizes

    create_causal_mask_compat._quasar_compat = True
    masking_utils.create_causal_mask = create_causal_mask_compat
    log.info("patched transformers.masking_utils.create_causal_mask for Quasar")


_triton_autotune_lock = threading.Lock()


def patch_triton_autotuner_thread_safety() -> None:
    """Make triton's Autotuner.run() safe to call from multiple threads at once.

    King and challenger share the same compiled fla kernels (e.g. l2norm_fwd_kernel
    inside chunk_gated_delta_rule), and run_eval's ThreadPoolExecutor drives their
    forward passes concurrently. triton.runtime.autotuner.Autotuner.cache is a plain
    dict with no locking, so two threads racing a cache-miss on the same kernel can
    corrupt the in-flight entry and crash with "TypeError: 'NoneType' object is not
    a mapping". King and challenger run on disjoint GPUs with independent CUDA
    streams, so serializing the brief Python-side dispatch doesn't block actual GPU
    overlap.
    """
    from triton.runtime.autotuner import Autotuner

    if getattr(Autotuner, "_quasar_thread_safe", False):
        return

    original_run = Autotuner.run

    def run_locked(self, *args, **kwargs):
        with _triton_autotune_lock:
            return original_run(self, *args, **kwargs)

    Autotuner.run = run_locked
    Autotuner._quasar_thread_safe = True
    log.info("patched triton.runtime.autotuner.Autotuner.run for thread-safety")


def patch_loaded_quasar_modules() -> None:
    try:
        import transformers.masking_utils as masking_utils
    except Exception:
        return
    patched_fn = getattr(masking_utils, "create_causal_mask", None)
    if patched_fn is None:
        return
    for name, module in list(sys.modules.items()):
        if name.endswith("modeling_qwen3_5") and hasattr(module, "create_causal_mask"):
            setattr(module, "create_causal_mask", patched_fn)


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


def snapshot_safetensor_names(snapshot_dir: str) -> list[str]:
    path = Path(snapshot_dir)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return sorted(set(index.get("weight_map", {}).values()))
    return sorted(p.name for p in path.glob("*.safetensors"))


def snapshot_meta(snapshot_dir: str) -> dict:
    path = Path(snapshot_dir)
    return {
        "path": str(path),
        "has_config": (path / "config.json").exists(),
        "has_tokenizer": any(path.glob("tokenizer*")) or any(path.glob("*.model")) or any(path.glob("vocab.*")),
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
        if not req.allow_code_model_fallback:
            raise RuntimeError(
                f"{label} snapshot is not self-contained enough to load its config/custom code: {exc}. "
                "Add config/custom-code files to the model snapshot, or set "
                "allow_code_model_fallback=true for a debug-only compatibility run."
            ) from exc
        if not req.code_model:
            raise RuntimeError(f"{label} config load failed and no code_model fallback was provided") from exc
        log.warning("%s config load from snapshot failed; falling back to code_model=%s", label, req.code_model)
        prepare_remote_code(req.code_model, req.revision)
        config = AutoConfig.from_pretrained(req.code_model, revision=req.revision, trust_remote_code=True)
        source = f"code_model:{req.code_model}"
    config.use_cache = False
    if on_phase:
        on_phase({"phase": f"{label}_config_load_done", "source": source, "model_type": getattr(config, "model_type", "")})
    return config, {"source": source, **meta}


def load_eval_tokenizer(king_snapshot: str, req: EvalRequest, on_phase=None):
    from transformers import AutoTokenizer

    source = (req.dataset_source or "s3").lower()
    tokenizer_required = source == "local"
    if source == "s3":
        return None, {"source": "not_needed_for_s3_npy"}

    ensure_snapshot_on_path(king_snapshot)
    if on_phase:
        on_phase({"phase": "tokenizer_load_start", "snapshot": king_snapshot})
    try:
        tokenizer = AutoTokenizer.from_pretrained(king_snapshot, revision=req.revision, trust_remote_code=True, use_fast=True)
        tokenizer_source = "king_snapshot"
    except Exception as exc:
        if not req.allow_code_model_fallback:
            if tokenizer_required:
                raise RuntimeError(
                    f"king snapshot tokenizer could not be loaded: {exc}. "
                    "For local parquet evals the evaluated model snapshot must provide tokenizer files, "
                    "or allow_code_model_fallback=true must be set for a debug-only run."
                ) from exc
            return None, {"source": "missing_but_not_needed"}
        tokenizer = AutoTokenizer.from_pretrained(req.code_model, revision=req.revision, trust_remote_code=True, use_fast=True)
        tokenizer_source = f"code_model:{req.code_model}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if on_phase:
        on_phase({"phase": "tokenizer_load_done", "source": tokenizer_source})
    return tokenizer, {"source": tokenizer_source}


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


def load_safetensors_state_dict(model_dir: str) -> dict:
    from concurrent.futures import ThreadPoolExecutor
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

    # Reserve the last GPU for norm + lm_head only.  It already bears the largest
    # activation spike: full hidden states (~4 GB) + logit chunk (~4 GB with chunk=32).
    # Distribute all transformer layers across GPUs 0..last-1.
    n_layer_gpus = max(1, len(ids) - 1)
    for layer_idx in range(n_layers):
        gpu_idx = min(n_layer_gpus - 1, (layer_idx * n_layer_gpus) // n_layers)
        device_map[f"model.layers.{layer_idx}"] = ids[gpu_idx]

    return device_map


def balanced_transformer_device_map_from_config(config) -> dict | None:
    if not torch.cuda.is_available() or len(_gpu_ids) < 2:
        return None
    n_layers = getattr(config, "num_hidden_layers", None)
    if not n_layers:
        return None
    first = _gpu_ids[0]
    last = _gpu_ids[-1]
    device_map: dict = {
        "model.embed_tokens": first,
        "model.norm": last,
        "lm_head": last,
    }
    for layer_idx in range(n_layers):
        gpu_idx = min(len(_gpu_ids) - 1, (layer_idx * len(_gpu_ids)) // n_layers)
        device_map[f"model.layers.{layer_idx}"] = _gpu_ids[gpu_idx]
    return device_map


def model_input_device(model) -> torch.device:
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed is not None:
        try:
            return next(embed.parameters()).device
        except StopIteration:
            pass
    return next(model.parameters()).device


def load_quasar_model(snapshot_dir: str, config, device: str, label: str, req: EvalRequest, gpu_ids: list[int] | None = None, on_phase=None):
    from transformers import AutoModelForCausalLM

    if on_phase:
        on_phase({"phase": f"{label}_load_start", "device": device, "snapshot": snapshot_dir})
    t0 = time.time()
    config.use_cache = False
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        if on_phase:
            on_phase({"phase": f"{label}_init_start", "dtype": str(dtype)})
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    finally:
        torch.set_default_dtype(old_dtype)
    if on_phase:
        on_phase({"phase": f"{label}_state_load_start"})
    state = load_safetensors_state_dict(snapshot_dir)
    info = model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    log.info("%s state loaded: missing=%d unexpected=%d", label, len(info.missing_keys), len(info.unexpected_keys))
    if on_phase:
        on_phase({"phase": f"{label}_to_device_start", "device": device, "dtype": str(dtype)})
    if device == "auto":
        effective_ids = gpu_ids if gpu_ids is not None else _gpu_ids
        log.info("%s dispatching across GPUs %s dtype=%s", label, effective_ids, dtype)
        model = model.to(dtype=dtype)
        model = dispatch_model_across_gpus(model, req, label, gpu_ids=gpu_ids, on_phase=on_phase)
    else:
        log.info("%s moving to %s dtype=%s", label, device, dtype)
        model = model.to(device=device, dtype=dtype)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    patch_loaded_quasar_modules()
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e9
    if on_phase:
        on_phase({"phase": f"{label}_load_done", "params_b": round(params, 3), "elapsed_s": round(time.time() - t0, 1)})
    return model


def expand_globs(patterns: list[str]) -> list[str]:
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        files.extend(matches)
    files = sorted({str(Path(path)) for path in files if path.endswith(".parquet")})
    if not files:
        raise FileNotFoundError(f"no parquet files matched: {patterns}")
    return files


def doppler_secret(name: str, project: str, config: str) -> str:
    return subprocess.check_output(
        ["doppler", "secrets", "get", name, "--project", project, "--config", config, "--plain"],
        text=True,
    ).strip()


def first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def s3_env_credentials(*, include_generic: bool = True) -> tuple[str, str] | None:
    access_names = [
        "TEUTONIC_DS_ACCESS_KEY",
    ]
    secret_names = [
        "TEUTONIC_DS_SECRET_KEY",
    ]
    if include_generic:
        access_names.extend((
            "HIPPIUS_ACCESS_KEY",
            "HIPPIUS_ACCESS_KEY_ID",
            "AWS_ACCESS_KEY_ID",
        ))
        secret_names.extend((
            "HIPPIUS_SECRET_KEY",
            "HIPPIUS_SECRET_ACCESS_KEY",
            "AWS_SECRET_ACCESS_KEY",
        ))
    access = first_env(tuple(access_names))
    secret = first_env(tuple(secret_names))
    if access and secret:
        return access, secret
    return None


def s3_doppler_credentials(req: EvalRequest) -> tuple[str, str]:
    return (
        doppler_secret("HIPPIUS_ACCESS_KEY", req.s3_doppler_project, req.s3_doppler_config),
        doppler_secret("HIPPIUS_SECRET_KEY", req.s3_doppler_project, req.s3_doppler_config),
    )


def s3_credentials(req: EvalRequest) -> tuple[str, str]:
    source = (req.s3_auth_source or "doppler").lower()
    if source == "env":
        creds = s3_env_credentials(include_generic=True)
        if creds:
            return creds
        raise RuntimeError("s3_auth_source='env' but no S3 credential env pair was found")
    if source == "doppler":
        try:
            return s3_doppler_credentials(req)
        except Exception as exc:
            raise RuntimeError(
                "Could not resolve S3 credentials via Doppler. Make "
                "`doppler secrets get HIPPIUS_ACCESS_KEY/HIPPIUS_SECRET_KEY "
                f"--project {req.s3_doppler_project} --config {req.s3_doppler_config}` available, "
                "or pass s3_auth_source='env' with TEUTONIC_DS_ACCESS_KEY/"
                "TEUTONIC_DS_SECRET_KEY."
            ) from exc
    if source != "auto":
        raise ValueError("s3_auth_source must be one of: doppler, env, auto")

    explicit = s3_env_credentials(include_generic=False)
    if explicit:
        return explicit
    try:
        return s3_doppler_credentials(req)
    except Exception:
        generic = s3_env_credentials(include_generic=True)
        if generic:
            return generic
        raise


def make_s3_client(req: EvalRequest):
    import boto3
    from botocore.config import Config as BotoConfig

    access, secret = s3_credentials(req)
    return boto3.client(
        "s3",
        endpoint_url=req.s3_endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": DEFAULT_S3_CLIENT_MAX_ATTEMPTS, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=300,
        ),
    )


def list_s3_shard_keys(client, req: EvalRequest) -> list[str]:
    prefix = req.s3_prefix
    keys: list[str] = []
    all_keys: list[str] = []
    counts = {
        "total": 0,
        "shards": 0,
        "npy_suffix": 0,
        "parquet_suffix": 0,
        "part_manifests": 0,
        "final_manifest": 0,
    }
    kw = {"Bucket": req.s3_bucket, "Prefix": prefix}
    while True:
        resp = client.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            all_keys.append(key)
            counts["total"] += 1
            counts["shards"] += int(req.s3_shard_contains in key)
            counts["npy_suffix"] += int(key.endswith(".npy"))
            counts["parquet_suffix"] += int(key.endswith(".parquet"))
            counts["part_manifests"] += int(key.endswith("/manifest.json") and "/parts/" in key)
            counts["final_manifest"] += int(key == prefix + "manifest.json")
            if key.endswith("/") or key.endswith("/manifest.json") or key.endswith("_SUCCESS"):
                continue
            if req.s3_shard_contains and req.s3_shard_contains not in key:
                continue
            if req.s3_shard_suffix and not key.endswith(req.s3_shard_suffix):
                continue
            if key.endswith(".json"):
                continue
            if key.endswith(".crc"):
                continue
            if key.endswith(".tmp"):
                continue
            keys.append(key)
        if not resp.get("IsTruncated"):
            break
        kw["ContinuationToken"] = resp["NextContinuationToken"]
    keys = sorted(set(keys))
    if not keys:
        sample = all_keys[:20]
        raise FileNotFoundError(
            "no shard keys matched at "
            f"s3://{req.s3_bucket}/{prefix}; counts={counts}; "
            f"filter_contains={req.s3_shard_contains!r}; "
            f"filter_suffix={req.s3_shard_suffix!r}; sample_keys={sample}"
        )
    log.info(
        "s3 dataset listed: bucket=%s prefix=%s counts=%s matched=%d sample=%s",
        req.s3_bucket,
        prefix,
        counts,
        len(keys),
        keys[:3],
    )
    return keys


def s3_cache_path(req: EvalRequest, key: str) -> Path:
    return SHARD_CACHE_DIR / req.s3_bucket / key


def s3_transfer_config():
    from boto3.s3.transfer import TransferConfig

    chunk_bytes = max(1, DEFAULT_S3_TRANSFER_CHUNK_MB) * 1024 * 1024
    return TransferConfig(
        multipart_threshold=chunk_bytes,
        multipart_chunksize=chunk_bytes,
        max_concurrency=max(1, DEFAULT_S3_TRANSFER_CONCURRENCY),
        num_download_attempts=max(1, DEFAULT_S3_TRANSFER_ATTEMPTS),
        use_threads=DEFAULT_S3_TRANSFER_CONCURRENCY > 1,
    )


def download_s3_shard(client, req: EvalRequest, key: str, on_phase=None) -> str:
    target = s3_cache_path(req, key)
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    transfer_config = s3_transfer_config()
    if on_phase:
        on_phase({"phase": "shard_download_start", "key": key})

    attempts = max(1, DEFAULT_S3_DOWNLOAD_RETRIES)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            if on_phase:
                on_phase({
                    "phase": "shard_download_attempt",
                    "key": key,
                    "attempt": attempt,
                    "attempts": attempts,
                    "transfer_attempts": DEFAULT_S3_TRANSFER_ATTEMPTS,
                    "transfer_concurrency": DEFAULT_S3_TRANSFER_CONCURRENCY,
                    "chunk_mb": DEFAULT_S3_TRANSFER_CHUNK_MB,
                })
            client.download_file(req.s3_bucket, key, str(tmp), Config=transfer_config)
            break
        except Exception as exc:
            last_exc = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            log.warning(
                "s3 shard download failed for s3://%s/%s attempt %d/%d: %s",
                req.s3_bucket,
                key,
                attempt,
                attempts,
                exc,
                exc_info=True,
            )
            if attempt >= attempts:
                raise RuntimeError(
                    f"failed to download shard s3://{req.s3_bucket}/{key} "
                    f"after {attempts} attempts: {exc}"
                ) from exc
            delay = DEFAULT_S3_DOWNLOAD_RETRY_BACKOFF_S * attempt
            if on_phase:
                on_phase({
                    "phase": "shard_download_retry_wait",
                    "key": key,
                    "attempt": attempt,
                    "attempts": attempts,
                    "sleep_s": round(delay, 1),
                    "error": str(exc),
                })
            time.sleep(delay)
    else:
        raise RuntimeError(f"failed to download shard s3://{req.s3_bucket}/{key}: {last_exc}")

    if not tmp.exists() or tmp.stat().st_size <= 0:
        raise RuntimeError(f"downloaded shard is empty: s3://{req.s3_bucket}/{key}")
    tmp.replace(target)
    if on_phase:
        on_phase({"phase": "shard_download_done", "key": key, "path": str(target)})
    return str(target)


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


def load_sequences_from_npy_shard(
    path: str,
    req: EvalRequest,
    rng: np.random.Generator,
    limit: int | None = None,
) -> list[list[int]]:
    data_offset, header = read_npy_header(path)
    dtype = np.dtype(header["descr"])
    shape = tuple(header["shape"])
    if dtype != np.dtype("<u4") and dtype != np.dtype("uint32"):
        raise ValueError(f"{path} dtype must be uint32/<u4, got {dtype}")
    if not shape:
        raise ValueError(f"{path} has invalid shape {shape}")

    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        if arr.shape[1] < req.seq_len:
            raise ValueError(f"{path} sequence width {arr.shape[1]} < seq_len={req.seq_len}")
        indices = shuffled_indices(rng, arr.shape[0], limit)
        return [arr[int(i), : req.seq_len].astype(np.int64, copy=False).tolist() for i in indices]

    if arr.ndim != 1:
        raise ValueError(f"{path} expected 1D token stream or 2D sequence matrix, got shape={arr.shape}")
    n_sequences = arr.shape[0] // req.seq_len
    if n_sequences <= 0:
        return []
    indices = shuffled_indices(rng, n_sequences, limit)
    out = []
    for i in indices:
        start = int(i) * req.seq_len
        out.append(arr[start : start + req.seq_len].astype(np.int64, copy=False).tolist())
    _ = data_offset
    return out


def sample_packed_sequences(files: list[str], tokenizer, req: EvalRequest, *, shuffle_files: bool = True) -> tuple[list[list[int]], dict]:
    import pyarrow.parquet as pq

    seed_value = dataset_seed(req)
    rng = random.Random(seed_value)
    ordered = list(files)
    if shuffle_files:
        rng.shuffle(ordered)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    sequences: list[list[int]] = []
    buffer: list[int] = []
    docs_seen = 0
    used_files: list[str] = []
    for path in ordered:
        used_files.append(path)
        pf = pq.ParquetFile(path)
        if req.text_field not in set(pf.schema_arrow.names):
            raise KeyError(f"{path} does not contain text field {req.text_field!r}")
        for batch in pf.iter_batches(batch_size=req.parquet_batch_size, columns=[req.text_field]):
            for text in batch.column(req.text_field).to_pylist():
                if not isinstance(text, str) or len(text) < req.min_chars:
                    continue
                docs_seen += 1
                if req.max_chars > 0 and len(text) > req.max_chars:
                    text = text[: req.max_chars]
                    cut = text.rfind(" ")
                    if cut > req.max_chars // 2:
                        text = text[:cut]
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    continue
                buffer.extend(ids)
                buffer.append(eos_id)
                while len(buffer) >= req.seq_len:
                    sequences.append(buffer[: req.seq_len])
                    del buffer[: req.seq_len]
                    if len(sequences) >= req.n:
                        digest = hashlib.sha256(np.asarray(sequences, dtype=np.int64).tobytes()).hexdigest()
                        return sequences, {
                            "n": len(sequences),
                            "seq_len": req.seq_len,
                            "seed": req.seed,
                            "dataset_seed": seed_value,
                            "seed_material": dataset_seed_material(req),
                            "block_hash": req.block_hash,
                            "hotkey": req.hotkey,
                            "digest": digest,
                            "source": req.dataset_source,
                            "used_files": used_files,
                            "docs_seen": docs_seen,
                        }
    raise RuntimeError(f"only produced {len(sequences)}/{req.n} sequences from {len(used_files)} files")


def sample_packed_sequences_from_s3(tokenizer, req: EvalRequest, on_phase=None) -> tuple[list[list[int]], dict]:
    client = make_s3_client(req)
    keys = list_s3_shard_keys(client, req)
    seed_value = dataset_seed(req)
    requested_shard = (req.shard_key or "").strip()
    if requested_shard and requested_shard in keys:
        keys = [requested_shard]
    elif requested_shard and requested_shard.startswith(req.s3_prefix) and requested_shard.endswith(req.s3_shard_suffix):
        keys = [requested_shard]
    rng = random.Random(seed_value)
    rng.shuffle(keys)
    if req.s3_max_shards > 0:
        keys = keys[: req.s3_max_shards]
    if on_phase:
        on_phase({
            "phase": "s3_manifest_listed",
            "bucket": req.s3_bucket,
            "prefix": req.s3_prefix,
            "shards": len(keys),
            "dataset_seed": seed_value,
            "block_hash": req.block_hash,
        })

    sequences: list[list[int]] = []
    used_keys: list[str] = []
    used_files: list[str] = []
    np_rng = np.random.default_rng(seed_value)

    for key in keys:
        local_path = download_s3_shard(client, req, key, on_phase=on_phase)
        used_keys.append(key)
        used_files.append(local_path)
        remaining = req.n - len(sequences)
        for seq in load_sequences_from_npy_shard(local_path, req, np_rng, remaining):
            sequences.append(seq)
            if len(sequences) >= req.n:
                digest = hashlib.sha256(np.asarray(sequences, dtype=np.int64).tobytes()).hexdigest()
                return sequences, {
                    "n": len(sequences),
                    "seq_len": req.seq_len,
                    "seed": req.seed,
                    "dataset_seed": seed_value,
                    "seed_material": dataset_seed_material(req),
                    "block_hash": req.block_hash,
                    "hotkey": req.hotkey,
                    "requested_shard_key": req.shard_key,
                    "digest": digest,
                    "source": "s3_npy",
                    "bucket": req.s3_bucket,
                    "prefix": req.s3_prefix,
                    "used_keys": used_keys,
                    "used_files": used_files,
                }
    raise RuntimeError(f"only produced {len(sequences)}/{req.n} sequences from {len(used_keys)} S3 npy shards")


def sample_eval_sequences(tokenizer, req: EvalRequest, on_phase=None) -> tuple[list[list[int]], dict]:
    source = (req.dataset_source or "s3").lower()
    if source == "s3":
        return sample_packed_sequences_from_s3(tokenizer, req, on_phase=on_phase)
    if source == "local":
        if tokenizer is None:
            raise RuntimeError("local parquet eval requires a tokenizer")
        files = expand_globs(req.parquet_glob)
        if on_phase:
            on_phase({"phase": "local_parquets_listed", "parquet_files": len(files)})
        return sample_packed_sequences(files, tokenizer, req)
    raise ValueError(f"unsupported dataset_source={req.dataset_source!r}; expected 's3' or 'local'")


def lm_head_device(model) -> torch.device:
    return next(model.lm_head.parameters()).device


@torch.no_grad()
def compute_per_sequence_loss(model, token_batches: list[list[int]], chunk_size: int) -> list[float]:
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=model_input_device(model))
    if hasattr(model, "reset_state"):
        model.reset_state()
    hidden = model.model(input_ids).last_hidden_state
    head_dev = lm_head_device(model)
    if hidden.device != head_dev:
        hidden = hidden.to(head_dev)
    labels_full = input_ids if input_ids.device == head_dev else input_ids.to(head_dev)

    batch = len(token_batches)
    n_pos = labels_full.size(1) - 1
    total = torch.zeros(batch, device=head_dev)
    for start in range(0, n_pos, chunk_size):
        end = min(start + chunk_size, n_pos)
        logits = model.lm_head(hidden[:, start:end, :])
        labels = labels_full[:, start + 1 : end + 1]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
        total += loss.reshape(batch, -1).sum(dim=1)
        del logits, loss
    return (total / n_pos).float().cpu().tolist()


def bootstrap_verdict(king_losses: list[float], challenger_losses: list[float], req: EvalRequest) -> dict:
    diff = np.asarray(king_losses, dtype=np.float64) - np.asarray(challenger_losses, dtype=np.float64)
    rng = np.random.default_rng(req.bootstrap_seed)
    boot = np.empty(req.n_bootstrap, dtype=np.float64)
    for i in range(req.n_bootstrap):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot[i] = diff[idx].mean()
    mu_hat = float(diff.mean())
    lcb = float(np.quantile(boot, req.alpha))
    accepted = lcb > req.delta_threshold
    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta": req.delta_threshold,
        "delta_threshold": req.delta_threshold,
        "alpha": req.alpha,
        "n_bootstrap": req.n_bootstrap,
        "n_sequences": len(diff),
        "avg_king_loss": round(float(np.mean(king_losses)), 6),
        "avg_challenger_loss": round(float(np.mean(challenger_losses)), 6),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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
    _king_model = load_quasar_model(snapshot, config, device, "king", req, gpu_ids=gpu_ids, on_phase=on_phase)
    _king_key = key
    _king_device = device
    _king_gpu_ids = effective_gpu_ids
    return _king_model


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
            shutil.rmtree(d, ignore_errors=True)
            running -= size
            log.info("cache cleanup: deleted %s (%.1f GB)", d, size / 1e9)
    except Exception:
        log.warning("cache cleanup failed", exc_info=True)


def write_record(eval_id: str, payload: dict) -> str:
    EVAL_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_RECORD_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{eval_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def run_eval(eval_id: str, req: EvalRequest) -> None:
    record = _evals[eval_id]
    events: Queue = record["events"]
    record["state"] = "running"
    t0 = time.time()

    def on_phase(info: dict):
        record["progress"] = info
        events.put({"type": "progress", "data": info})

    heartbeat_stop = threading.Event()

    def heartbeat_loop():
        while not heartbeat_stop.wait(30):
            on_phase({"phase": "heartbeat"})

    threading.Thread(target=heartbeat_loop, daemon=True, name=f"heartbeat-{eval_id}").start()

    challenger = None
    _eval_pool = None
    try:
        on_phase({"phase": "setup_start"})
        limits_meta = apply_eval_limits(req, eval_id)
        on_phase({"phase": "limits_applied", **limits_meta})
        preflight_deps()
        patch_transformers_masking_compat()
        patch_triton_autotuner_thread_safety()

        check_eval_runtime(t0)
        king_snapshot = materialize_model(req.king_repo, req.king_digest, on_phase=on_phase)
        check_eval_runtime(t0)
        challenger_snapshot = materialize_model(req.challenger_repo, req.challenger_digest, on_phase=on_phase)
        check_eval_runtime(t0)
        duplicate_meta = reject_duplicate_safetensors(king_snapshot, challenger_snapshot, on_phase=on_phase)
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

        tokenizer, tokenizer_meta = load_eval_tokenizer(king_snapshot, req, on_phase=on_phase)

        on_phase({
            "phase": "dataset_sample_start",
            "source": req.dataset_source,
            "n": req.n,
            "seq_len": req.seq_len,
        })
        sequences, dataset_meta = sample_eval_sequences(tokenizer, req, on_phase=on_phase)
        # Pop private key so it never reaches the verdict JSON or disk record.
        source_labels: list[str] | None = dataset_meta.pop("_source_labels", None)
        check_eval_runtime(t0)
        on_phase({"phase": "dataset_sample_done", "digest": dataset_meta["digest"][:16]})

        use_parallel = req.parallel_models and len(_gpu_ids) >= 4
        if use_parallel:
            mid = len(_gpu_ids) // 2
            king_gpu_ids = _gpu_ids[:mid]
            challenger_gpu_ids = _gpu_ids[mid:]
            on_phase({"phase": "parallel_models_setup", "king_gpus": king_gpu_ids, "challenger_gpus": challenger_gpu_ids})
        else:
            king_gpu_ids = _gpu_ids
            challenger_gpu_ids = _gpu_ids

        king_device = device_plan_for_gpus(king_gpu_ids)
        challenger_device = device_plan_for_gpus(challenger_gpu_ids)
        king = ensure_king(req, king_snapshot, king_config, king_artifacts["source"], king_device, gpu_ids=king_gpu_ids, on_phase=on_phase)
        check_eval_runtime(t0)
        challenger = load_quasar_model(
            challenger_snapshot,
            challenger_config,
            challenger_device,
            "challenger",
            req,
            gpu_ids=challenger_gpu_ids,
            on_phase=on_phase,
        )
        check_eval_runtime(t0)

        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor
            _eval_pool = ThreadPoolExecutor(max_workers=2)

        effective_batch_size = req.parallel_batch_size if use_parallel else req.batch_size
        king_losses: list[float] = []
        challenger_losses: list[float] = []
        king_sum = 0.0
        challenger_sum = 0.0
        eval_t0 = time.time()
        total_batches = (len(sequences) + effective_batch_size - 1) // effective_batch_size
        log_every_batches = max(1, int(req.log_every_batches or 1))
        for start in range(0, len(sequences), effective_batch_size):
            check_eval_runtime(t0)
            batch_idx = (start // effective_batch_size) + 1
            batch = sequences[start : start + effective_batch_size]
            if _eval_pool is not None:
                kfut = _eval_pool.submit(compute_per_sequence_loss, king, batch, req.lm_head_chunk)
                cfut = _eval_pool.submit(compute_per_sequence_loss, challenger, batch, req.lm_head_chunk)
                kl = kfut.result()
                cl = cfut.result()
            else:
                kl = compute_per_sequence_loss(king, batch, req.lm_head_chunk)
                cl = compute_per_sequence_loss(challenger, batch, req.lm_head_chunk)
            king_losses.extend(kl)
            challenger_losses.extend(cl)
            king_sum += float(np.sum(kl))
            challenger_sum += float(np.sum(cl))
            done = len(king_losses)
            diff_so_far = np.asarray(king_losses) - np.asarray(challenger_losses)
            mu_hat = float(diff_so_far.mean())
            seq_per_s = done / max(time.time() - eval_t0, 1e-9)
            if batch_idx % log_every_batches == 0 or done == len(sequences):
                eval_log.info(
                    "batch %d/%d | done=%d/%d | mu_hat=%.6f | %.1f seq/s",
                    batch_idx,
                    total_batches,
                    done,
                    len(sequences),
                    mu_hat,
                    seq_per_s,
                )
            on_phase({
                "phase": "eval_progress",
                "batch": batch_idx,
                "total_batches": total_batches,
                "done": done,
                "total": len(sequences),
                "mu_hat": round(mu_hat, 6),
                "seq_per_s": round(seq_per_s, 1),
                "avg_king_loss": round(king_sum / done, 6),
                "avg_challenger_loss": round(challenger_sum / done, 6),
            })

            # Early stop: can challenger still mathematically achieve LCB > delta_threshold?
            # Upper bound on final mu_hat = assume all remaining seqs give the max
            # observed per-seq advantage. Since LCB <= mu_hat, if mu_upper < delta_threshold
            # the challenger cannot win regardless of the remaining samples.
            n_total = len(sequences)
            if (EVAL_EARLY_STOP and done < n_total
                    and done >= int(n_total * EVAL_EARLY_STOP_MIN_FRACTION)):
                advantage_quantile = min(max(EVAL_EARLY_STOP_ADVANTAGE_QUANTILE, 0.0), 1.0)
                d_max = float(np.quantile(diff_so_far, advantage_quantile))
                remaining = n_total - done
                mu_upper = (float(diff_so_far.sum()) + remaining * d_max) / n_total
                if mu_upper < req.delta_threshold:
                    # Compute the real bootstrap LCB on the partial observed data
                    # using the same formula as bootstrap_verdict. The partial LCB
                    # will be <= mu_hat <= mu_upper < delta_threshold.
                    es_rng = np.random.default_rng(req.bootstrap_seed)
                    n_es = len(diff_so_far)
                    es_boot = np.empty(req.n_bootstrap, dtype=np.float64)
                    for _b in range(req.n_bootstrap):
                        _idx = es_rng.integers(0, n_es, size=n_es)
                        es_boot[_b] = diff_so_far[_idx].mean()
                    lcb_partial = float(np.quantile(es_boot, req.alpha))
                    eval_log.info(
                        "early stop at %d/%d seqs: best-case mu_hat=%.6f "
                        "lcb_partial=%.6f < delta=%.6f advantage_quantile=%.3f",
                        done, n_total, mu_upper, lcb_partial, req.delta_threshold, advantage_quantile,
                    )
                    elapsed = time.time() - t0
                    early_verdict = {
                        "accepted": False,
                        "verdict": "king",
                        "mu_hat": round(mu_hat, 6),
                        "mu_hat_upper_bound": round(mu_upper, 6),
                        "lcb": round(lcb_partial, 6),
                        "delta": req.delta_threshold,
                        "delta_threshold": req.delta_threshold,
                        "alpha": req.alpha,
                        "n_bootstrap": req.n_bootstrap,
                        "n_sequences": n_total,
                        "n_sequences_evaluated": done,
                        "avg_king_loss": round(king_sum / done, 6),
                        "avg_challenger_loss": round(challenger_sum / done, 6),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "early_stopped": True,
                        "early_stop_reason": (
                            f"best_case_mu_hat={mu_upper:.6f} < delta_threshold="
                            f"{req.delta_threshold:.6f} after {done}/{n_total} seqs"
                        ),
                        "early_stop_advantage_quantile": advantage_quantile,
                        "early_stop_assumed_remaining_advantage": round(d_max, 6),
                    }
                    early_verdict["source_scores"] = _compute_source_scores(
                        king_losses, challenger_losses,
                        source_labels[:done] if source_labels else None,
                    )
                    early_verdict.update({
                        "eval_id": eval_id,
                        "king_repo": normalize_model_ref(req.king_repo),
                        "challenger_repo": normalize_model_ref(req.challenger_repo),
                        "king_digest": req.king_digest,
                        "challenger_digest": req.challenger_digest,
                        "code_model": req.code_model,
                        "allow_code_model_fallback": req.allow_code_model_fallback,
                        "king_device": king_device,
                        "challenger_device": challenger_device,
                        "parallel_models": use_parallel,
                        "gpu_memory_fraction": req.gpu_memory_fraction,
                        "limits": limits_meta,
                        "model_artifacts": {
                            "king": king_artifacts,
                            "challenger": challenger_artifacts,
                            "tokenizer": tokenizer_meta,
                            "duplicate_check": duplicate_meta,
                        },
                        "dataset": dataset_meta,
                        "dataset_source": req.dataset_source,
                        "wall_time_s": round(elapsed, 1),
                    })
                    early_verdict["record_path"] = write_record(
                        eval_id, {"request": req.model_dump(), "verdict": early_verdict}
                    )
                    record["state"] = "completed"
                    record["verdict"] = early_verdict
                    events.put({"type": "verdict", "data": early_verdict})
                    return

        verdict = bootstrap_verdict(king_losses, challenger_losses, req)
        verdict["source_scores"] = _compute_source_scores(
            king_losses, challenger_losses, source_labels
        )
        verdict.update({
            "eval_id": eval_id,
            "king_repo": normalize_model_ref(req.king_repo),
            "challenger_repo": normalize_model_ref(req.challenger_repo),
            "king_digest": req.king_digest,
            "challenger_digest": req.challenger_digest,
            "code_model": req.code_model,
            "allow_code_model_fallback": req.allow_code_model_fallback,
            "king_device": king_device,
            "challenger_device": challenger_device,
            "parallel_models": use_parallel,
            "gpu_memory_fraction": req.gpu_memory_fraction,
            "limits": limits_meta,
            "model_artifacts": {
                "king": king_artifacts,
                "challenger": challenger_artifacts,
                "tokenizer": tokenizer_meta,
                "duplicate_check": duplicate_meta,
            },
            "dataset": dataset_meta,
            "dataset_source": req.dataset_source,
            "wall_time_s": round(time.time() - t0, 1),
        })
        verdict["record_path"] = write_record(eval_id, {"request": req.model_dump(), "verdict": verdict})
        record["state"] = "completed"
        record["verdict"] = verdict
        events.put({"type": "verdict", "data": verdict})
    except Exception as exc:
        log.exception("eval %s failed", eval_id)
        record["state"] = "failed"
        record["error"] = str(exc)
        events.put({"type": "error", "data": {"error": str(exc)}})
    finally:
        heartbeat_stop.set()
        if _eval_pool is not None:
            _eval_pool.shutdown(wait=False)
        if challenger is not None:
            del challenger
            torch.cuda.empty_cache()
        cleanup_model_cache()
        try:
            _eval_lock.release()
        except RuntimeError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu_ids
    setup_logging()
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_model_decryption_key_permissions()
    _gpu_ids = parse_gpu_ids()
    log.info(
        "Quasar pair eval server starting; gpus=%s model_cache=%s shard_cache=%s records=%s",
        _gpu_ids,
        MODEL_CACHE_DIR,
        SHARD_CACHE_DIR,
        EVAL_RECORD_DIR,
    )
    yield
    log.info("Quasar pair eval server shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu_ids": _gpu_ids,
        "king_loaded": _king_key,
        "active_evals": len(_evals),
        "cache_dir": str(MODEL_CACHE_DIR),
        "shard_cache_dir": str(SHARD_CACHE_DIR),
        "record_dir": str(EVAL_RECORD_DIR),
        "defaults": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "alpha": DEFAULT_ALPHA,
            "seq_len": DEFAULT_SEQ_LEN,
            "n": DEFAULT_N,
            "n_public": DEFAULT_N_PUBLIC,
            "n_private": DEFAULT_N_PRIVATE,
            "n_bootstrap": DEFAULT_BOOTSTRAP_B,
        },
        "caps": {
            "eval_n_cap": EVAL_N_CAP,
            "eval_bootstrap_b_cap": EVAL_BOOTSTRAP_B_CAP,
            "eval_max_runtime_s": EVAL_MAX_RUNTIME_S,
        },
        "early_stop": {
            "enabled": EVAL_EARLY_STOP,
            "min_fraction": EVAL_EARLY_STOP_MIN_FRACTION,
            "advantage_quantile": EVAL_EARLY_STOP_ADVANTAGE_QUANTILE,
        },
        "download": {
            "retries": DEFAULT_MODEL_DOWNLOAD_RETRIES,
            "retry_backoff_s": DEFAULT_MODEL_DOWNLOAD_RETRY_BACKOFF_S,
            "workers": DEFAULT_MODEL_DOWNLOAD_WORKERS,
            "allow_patterns": MODEL_ALLOW_PATTERNS,
        },
        "encryption": {
            "manifest_name": MODEL_ENCRYPTION_MANIFEST_NAME,
            "age_available": shutil.which("age") is not None,
            "private_key_available": model_decryption_key_available(),
        },
        "probe_enabled": PROBE_ENABLED,
        "allow_code_model_fallback_default": DEFAULT_ALLOW_CODE_MODEL_FALLBACK,
    }


@app.post("/eval")
async def start_eval(req: EvalRequest):
    if not _eval_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="an eval is already running")
    eval_id = uuid.uuid4().hex[:8]
    _evals[eval_id] = {
        "state": "pending",
        "progress": {},
        "verdict": None,
        "error": None,
        "request": req.model_dump(),
        "events": Queue(),
        "created_at": time.time(),
    }
    threading.Thread(target=run_eval, args=(eval_id, req), daemon=True, name=f"eval-{eval_id}").start()
    return {"eval_id": eval_id}


@app.get("/eval/{eval_id}")
async def get_eval(eval_id: str):
    if eval_id not in _evals:
        raise HTTPException(status_code=404, detail="eval not found")
    rec = _evals[eval_id]
    return {
        "eval_id": eval_id,
        "state": rec["state"],
        "progress": rec["progress"],
        "verdict": rec["verdict"],
        "error": rec["error"],
    }


@app.get("/eval/{eval_id}/stream")
async def stream_eval(eval_id: str):
    if eval_id not in _evals:
        raise HTTPException(status_code=404, detail="eval not found")
    rec = _evals[eval_id]
    event_q: Queue = rec["events"]

    async def generate():
        while True:
            try:
                event = event_q.get(block=False)
            except Empty:
                await asyncio.sleep(0.5)
                if rec["state"] in ("completed", "failed") and event_q.empty():
                    final = rec["verdict"] or {"error": rec.get("error")}
                    final_type = "verdict" if rec["state"] == "completed" else "error"
                    yield f"data: {json.dumps({'type': final_type, 'data': final})}\n\n"
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
