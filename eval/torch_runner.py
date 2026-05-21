#!/usr/bin/env python3
"""Standalone multi-GPU PyTorch eval — king-vs-challenger paired bootstrap test.

Loads model replicas across all available GPUs, fetches sequences from R2
with prefetch overlap, and computes cross-entropy loss via chunked lm_head
forward passes to minimize VRAM. Accepts the challenger only when the
bootstrapped lower confidence bound on the per-token log-loss advantage
exceeds the fixed effect floor `EVAL_DELTA` (default 0.0025 nats/token).

Usage:
    python eval_torch.py \
        --king unconst/Teutonic-I \
        --challenger unconst/Teutonic-I \
        --n 100 --batch-size 64 --seq-len 2048 --gpus 0,1,2,3,4,5,6,7

Env vars:
    TEUTONIC_R2_ENDPOINT  R2 endpoint URL
    TEUTONIC_R2_BUCKET    R2 bucket name (default: constantinople)
    TEUTONIC_R2_ACCESS_KEY R2 access key
    TEUTONIC_R2_SECRET_KEY R2 secret key
    TEUTONIC_DS_ENDPOINT   Dataset store endpoint (default: R2 endpoint)
    TEUTONIC_DS_BUCKET     Dataset store bucket (default: R2 bucket)
    TEUTONIC_DS_ACCESS_KEY Dataset store access key (default: R2 key)
    TEUTONIC_DS_SECRET_KEY Dataset store secret key (default: R2 key)
"""
import argparse
import ast
import hashlib
import io
import json
import logging
import os
import pathlib
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# Belt-and-suspenders: any unguarded socket op blocks at most this long.
socket.setdefaulttimeout(180)

import boto3
import numpy as np
import torch
import torch.nn.functional as F
from botocore.config import Config as BotoConfig
from collections import defaultdict
from transformers import AutoModelForCausalLM

# eval_server runs us with cwd=/home/const/workspace/teutonic, so the workspace
# root is not on sys.path by default. Add it so `import teutonic.quasar`
# resolves the vendored arch.
_workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

# Load the active arch module from chain.toml so AutoModelForCausalLM
# dispatches checkpoints without trust_remote_code. Idempotent.
import chain_config  # noqa: E402
chain_config.load_arch()  # noqa: E402
from eval.raw_dataset import load_raw_sequences, raw_dataset_enabled  # noqa: E402
from model_store import ModelRef, materialize_model  # noqa: E402

log = logging.getLogger("eval_torch")


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------

class R2:
    def __init__(self):
        # Sized for SHARD_DL_WORKERS parallel ranged GETs.
        # Hippius can be slow on large ranged reads — 5min read_timeout
        # avoids spurious "Read timeout on endpoint URL: None" failures.
        _pool = max(SHARD_DL_WORKERS * 2, 32)
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
            aws_access_key_id=os.environ["TEUTONIC_R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["TEUTONIC_R2_SECRET_KEY"],
            region_name="auto",
            config=BotoConfig(
                retries={"max_attempts": 5, "mode": "adaptive"},
                max_pool_connections=_pool,
                connect_timeout=30,
                read_timeout=300,
            ),
        )
        self.bucket = os.environ.get("TEUTONIC_R2_BUCKET", "constantinople")

        ds_endpoint = os.environ.get("TEUTONIC_DS_ENDPOINT")
        ds_access = os.environ.get("TEUTONIC_DS_ACCESS_KEY")
        ds_secret = os.environ.get("TEUTONIC_DS_SECRET_KEY")
        if ds_endpoint and ds_access and ds_secret:
            self.ds_client = boto3.client(
                "s3",
                endpoint_url=ds_endpoint,
                aws_access_key_id=ds_access,
                aws_secret_access_key=ds_secret,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 5, "mode": "adaptive"},
                    s3={"addressing_style": "path"},
                    max_pool_connections=_pool,
                    connect_timeout=30,
                    read_timeout=300,
                ),
            )
            self.ds_bucket = os.environ.get("TEUTONIC_DS_BUCKET", self.bucket)
            log.info("dataset store: %s bucket=%s", ds_endpoint, self.ds_bucket)
        else:
            self.ds_client = self.client
            self.ds_bucket = self.bucket

    def get(self, key):
        try:
            return json.loads(
                self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def range_get(self, key, start, end):
        return self.client.get_object(
            Bucket=self.bucket, Key=key, Range=f"bytes={start}-{end}"
        )["Body"].read()

    def ds_get(self, key):
        try:
            return json.loads(
                self.ds_client.get_object(Bucket=self.ds_bucket, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def ds_range_get(self, key, start, end):
        return self.ds_client.get_object(
            Bucket=self.ds_bucket, Key=key, Range=f"bytes={start}-{end}"
        )["Body"].read()


# ---------------------------------------------------------------------------
# Dataset (identical to validator.py)
# ---------------------------------------------------------------------------

def _read_npy_header_bytes(raw: bytes) -> tuple[int, dict]:
    """Parse enough of a .npy header to validate eval shard shape/dtype."""
    buf = io.BytesIO(raw)
    magic = buf.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("dataset shard is not a .npy file")
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    hdr = ast.literal_eval(buf.read(hl).decode("latin1").strip())
    dtype = np.dtype(hdr.get("descr"))
    if dtype != np.dtype("<u4"):
        raise ValueError(f"dataset shard dtype must be uint32/<u4, got {dtype}")
    return buf.tell(), hdr


def get_shard_info(r2, shard_key):
    header = r2.ds_range_get(shard_key, 0, 1023)
    _, hdr = _read_npy_header_bytes(header)
    n = 1
    for s in hdr["shape"]:
        n *= s
    return n


R2_FETCH_WORKERS = 32

def _parse_shard_header(r2, shard_key):
    header = r2.ds_range_get(shard_key, 0, 1023)
    off, _hdr = _read_npy_header_bytes(header)
    return off


def fetch_sequences(r2, shard_key, indices, seq_len):
    data_offset = _parse_shard_header(r2, shard_key)
    bps = seq_len * 4
    sorted_idx = sorted(set(indices))
    idx_set = set(indices)

    groups, gs, ge = [], sorted_idx[0], sorted_idx[0]
    for i in sorted_idx[1:]:
        if i - ge <= 64:
            ge = i
        else:
            groups.append((gs, ge))
            gs = ge = i
    groups.append((gs, ge))

    def _fetch_group(gs_ge):
        gs, ge = gs_ge
        chunk = r2.ds_range_get(shard_key, data_offset + gs * bps, data_offset + (ge + 1) * bps - 1)
        partial = {}
        for idx in range(gs, ge + 1):
            if idx in idx_set:
                off = (idx - gs) * bps
                partial[idx] = np.frombuffer(chunk[off : off + bps], dtype="<u4").tolist()
        return partial

    result = {}
    with ThreadPoolExecutor(max_workers=R2_FETCH_WORKERS) as pool:
        for partial in pool.map(_fetch_group, groups):
            result.update(partial)
    return result


SHARD_CACHE_DIR = os.environ.get("TEUTONIC_SHARD_CACHE", "/tmp/shard_cache")
SHARD_CACHE_MAX = int(os.environ.get("TEUTONIC_SHARD_CACHE_MAX", "10"))
SHARD_DL_WORKERS = int(os.environ.get("TEUTONIC_SHARD_DL_WORKERS", "16"))

# Per-shard lock so two callers (e.g. background prefetch + the synchronous
# download_shard inside run_bootstrap_test) don't both spend bandwidth on the
# same shard. The first one downloads; the second one sees the cache file
# exist and reads it back.
_shard_locks: dict[str, threading.Lock] = {}
_shard_locks_guard = threading.Lock()


def _parse_npy_header(raw: bytes) -> int:
    """Return the byte offset where data begins in a .npy file."""
    buf = io.BytesIO(raw)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    buf.read(hl)
    return buf.tell()


def _evict_shard_cache():
    """Keep only the most recent SHARD_CACHE_MAX files in the cache dir."""
    cache = pathlib.Path(SHARD_CACHE_DIR)
    if not cache.exists():
        return
    files = sorted(cache.glob("*.npy"), key=lambda f: f.stat().st_mtime)
    while len(files) > SHARD_CACHE_MAX:
        victim = files.pop(0)
        victim.unlink(missing_ok=True)
        log.info("evicted cached shard %s", victim.name)


def _shard_lock(shard_key: str) -> threading.Lock:
    with _shard_locks_guard:
        lock = _shard_locks.get(shard_key)
        if lock is None:
            lock = threading.Lock()
            _shard_locks[shard_key] = lock
        return lock


def prefetch_shard(r2, shard_key):
    """Background-thread shard prefetch. Idempotent; multiple callers coalesce
    on the per-shard lock and the cache file. Used by eval_server to overlap
    shard download with model loading on each new eval.
    """
    cache_name = shard_key.replace("/", "_")
    cache_path = pathlib.Path(SHARD_CACHE_DIR) / cache_name
    if cache_path.exists():
        return

    def _do():
        try:
            download_shard(r2, shard_key)
        except Exception:
            log.warning("background shard prefetch failed for %s", shard_key, exc_info=True)

    threading.Thread(
        target=_do, daemon=True, name=f"shard-prefetch-{shard_key[:30]}"
    ).start()


def download_shard(r2, shard_key):
    """Download shard with parallel streams and local disk cache.

    Per-shard lock ensures two concurrent callers don't both eat bandwidth.
    """
    cache_name = shard_key.replace("/", "_")
    cache_path = pathlib.Path(SHARD_CACHE_DIR) / cache_name

    with _shard_lock(shard_key):
        if cache_path.exists():
            t0 = time.time()
            raw = cache_path.read_bytes()
            data_offset = _parse_npy_header(raw)
            elapsed = time.time() - t0
            log.info("shard cache HIT %s: %.1f MB read in %.2fs",
                     shard_key, len(raw) / 1e6, elapsed)
            return data_offset, raw

        return _download_shard_locked(r2, shard_key, cache_path)


def _download_shard_locked(r2, shard_key, cache_path):
    """Actual download. Caller already holds the per-shard lock."""
    t0 = time.time()
    head = r2.ds_client.head_object(Bucket=r2.ds_bucket, Key=shard_key)
    total_size = head["ContentLength"]

    n_workers = min(SHARD_DL_WORKERS, max(1, total_size // (64 * 1024 * 1024)))
    chunk_size = total_size // n_workers
    chunks = [None] * n_workers

    def _dl_chunk(i):
        start = i * chunk_size
        end_byte = total_size - 1 if i == n_workers - 1 else (i + 1) * chunk_size - 1
        chunks[i] = r2.ds_client.get_object(
            Bucket=r2.ds_bucket, Key=shard_key,
            Range=f"bytes={start}-{end_byte}",
        )["Body"].read()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_dl_chunk, range(n_workers)))

    raw = b"".join(chunks)
    del chunks

    data_offset = _parse_npy_header(raw)
    elapsed = time.time() - t0
    log.info("downloaded shard %s: %.1f MB in %.1fs (%.0f MB/s, %d streams)",
             shard_key, len(raw) / 1e6, elapsed,
             len(raw) / 1e6 / elapsed if elapsed > 0 else 0, n_workers)

    try:
        pathlib.Path(SHARD_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        _evict_shard_cache()
        log.info("cached shard to %s", cache_path)
    except Exception:
        log.warning("failed to cache shard to disk", exc_info=True)

    return data_offset, raw


def extract_sequences(shard_data, data_offset, indices, seq_len):
    """Extract sequences from a locally-cached shard."""
    bps = seq_len * 4
    result = {}
    for idx in indices:
        off = data_offset + idx * bps
        result[idx] = np.frombuffer(shard_data[off : off + bps], dtype="<u4").tolist()
    return result


def _evaluator_vocab_size(evaluator) -> int | None:
    try:
        model = evaluator.primary_model
        cfg = getattr(model, "config", None)
        if cfg is None:
            return None
        vocab = int(getattr(cfg, "vocab_size", 0) or 0)
        return vocab or None
    except Exception:
        return None


def validate_sequence_cache(seq_cache, seq_len: int, vocab_size: int | None = None):
    """Fail fast on malformed token shards before model scoring."""
    if not seq_cache:
        raise ValueError("dataset shard produced no token sequences")
    max_id = -1
    for idx, seq in seq_cache.items():
        if len(seq) != seq_len:
            raise ValueError(f"sequence {idx} has length {len(seq)}, expected {seq_len}")
        if seq:
            max_id = max(max_id, max(seq))
    if vocab_size is not None and max_id >= vocab_size:
        raise ValueError(f"dataset token id {max_id} exceeds model vocab_size {vocab_size}")


# ---------------------------------------------------------------------------
# Chunked loss computation — avoids materializing full [batch, seq, vocab]
# ---------------------------------------------------------------------------

LM_HEAD_CHUNK = int(os.environ.get("TEUTONIC_LM_HEAD_CHUNK", "512"))

@torch.no_grad()
def _lm_head_device(model) -> torch.device:
    """Where lm_head's weight lives. For tied-embedding models this equals
    the embed_tokens device; for untied/sharded models this is wherever
    accelerate placed the head."""
    return next(model.lm_head.parameters()).device


@torch.no_grad()
def compute_batch_losses(model, token_batches, device, chunk_size=LM_HEAD_CHUNK):
    """Forward pass with chunked lm_head to avoid OOM on large vocabs.

    Instead of model(input_ids).logits which allocates [batch, seq, vocab],
    we get hidden states first then apply lm_head in small chunks along the
    sequence dimension. Peak VRAM drops ~7x for vocab_size=262144.

    Works for both single-GPU models (`device` is "cuda:N") and accelerate-
    sharded models (`device` is the input-embedding device). When sharded,
    `last_hidden_state` may surface on a different GPU than `lm_head`; we
    relocate it once outside the chunk loop so the chunked CE doesn't bounce
    across devices.

    The `@torch.no_grad()` is critical for sharded 80B models: without it
    PyTorch keeps every layer's activations alive for backward, blowing
    per-GPU memory by ~6x (~30 GiB/layer × 36 layers across 4 GPUs ≈ 1 TB).
    `compute_paired_losses` already had this decorator; sharded paired runs
    are now routed through this function so it needed parity.
    """
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    # Quasar (and any future stateful arch) carries a persistent latent memory
    # state across forward calls. Resetting before every batch keeps paired-CE
    # numbers exchangeable — required for the bootstrap LCB. Stock HF archs
    # (Qwen3, Gemma) do not implement reset_state and pass through harmlessly.
    if hasattr(model, "reset_state"):
        model.reset_state()
    hidden = model.model(input_ids).last_hidden_state
    lm_head = model.lm_head
    head_dev = _lm_head_device(model)
    if hidden.device != head_dev:
        hidden = hidden.to(head_dev)
    labels = input_ids if input_ids.device == head_dev else input_ids.to(head_dev)

    n_positions = labels.size(1) - 1
    total_loss = torch.zeros(len(token_batches), device=head_dev)

    for i in range(0, n_positions, chunk_size):
        end_pos = min(i + chunk_size, n_positions)
        chunk_logits = lm_head(hidden[:, i:end_pos, :])
        chunk_labels = labels[:, i + 1 : end_pos + 1]
        loss = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            reduction="none",
        )
        total_loss += loss.reshape(len(token_batches), -1).sum(dim=1)
        del chunk_logits, loss

    return (total_loss / n_positions).cpu().tolist()


# ---------------------------------------------------------------------------
# Paired losses — runs both models' lm_heads per chunk
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_paired_losses(king_model, chall_model, token_batches,
                          king_device, chall_device,
                          chunk_size=LM_HEAD_CHUNK):
    """Compute per-sequence mean cross-entropy for both models on the same tokens.

    Returns (king_losses, chall_losses) as lists of floats (nats/token).

    Sharded-safe: if either model is split across multiple GPUs (accelerate
    `device_map='auto'`), the input lives on the embed device but lm_head may
    sit on a different shard; we relocate hidden + labels to lm_head's device
    once outside the chunk loop instead of paying per-chunk transfers.
    """
    B = len(token_batches)
    input_ids_k = torch.tensor(token_batches, dtype=torch.long, device=king_device)
    input_ids_c = torch.tensor(token_batches, dtype=torch.long, device=chall_device)

    # See compute_batch_losses for rationale. Both models reset before each
    # paired batch so neither carries state from the previous call into the
    # CE numbers the bootstrap test consumes.
    if hasattr(king_model, "reset_state"):
        king_model.reset_state()
    if hasattr(chall_model, "reset_state"):
        chall_model.reset_state()

    hidden_k = king_model.model(input_ids_k).last_hidden_state
    hidden_c = chall_model.model(input_ids_c).last_hidden_state

    head_dev_k = _lm_head_device(king_model)
    head_dev_c = _lm_head_device(chall_model)
    if hidden_k.device != head_dev_k:
        hidden_k = hidden_k.to(head_dev_k)
    if hidden_c.device != head_dev_c:
        hidden_c = hidden_c.to(head_dev_c)
    labels_full_k = input_ids_k if input_ids_k.device == head_dev_k else input_ids_k.to(head_dev_k)
    labels_full_c = input_ids_c if input_ids_c.device == head_dev_c else input_ids_c.to(head_dev_c)

    n_pos = labels_full_k.size(1) - 1
    king_loss = torch.zeros(B, device=head_dev_k)
    chall_loss = torch.zeros(B, device=head_dev_c)

    for i in range(0, n_pos, chunk_size):
        end = min(i + chunk_size, n_pos)

        logits_k = king_model.lm_head(hidden_k[:, i:end, :])
        logits_c = chall_model.lm_head(hidden_c[:, i:end, :])

        labels_k = labels_full_k[:, i + 1 : end + 1]
        labels_c = labels_full_c[:, i + 1 : end + 1]
        king_loss += F.cross_entropy(
            logits_k.reshape(-1, logits_k.size(-1)), labels_k.reshape(-1),
            reduction="none",
        ).reshape(B, -1).sum(1)
        chall_loss += F.cross_entropy(
            logits_c.reshape(-1, logits_c.size(-1)), labels_c.reshape(-1),
            reduction="none",
        ).reshape(B, -1).sum(1)

        del logits_k, logits_c

    return (
        (king_loss / n_pos).cpu().tolist(),
        (chall_loss / n_pos).cpu().tolist(),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _prefetch_repo(repo, digest=None, timeout=600):
    """Materialize a Hippius Hub digest snapshot with an explicit wall-clock cap."""
    import threading

    result = {"path": None, "err": None}

    def _do():
        try:
            ref = ModelRef(repo, digest or "")
            result["path"] = materialize_model(ref, max_workers=16)
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_do, daemon=True, name=f"hippius-prefetch-{repo}")
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"prefetch of {repo}@{(digest or '')[:19]} exceeded {timeout}s")
    if result["err"] is not None:
        raise result["err"]
    return result["path"]

def _build_sharded_device_map(gpu_ids: list[int],
                              per_gpu_gib: int | None = None) -> dict:
    """device_map for accelerate when sharding ONE replica across `gpu_ids`.

    Caps every GPU not in `gpu_ids` to 0 GiB so accelerate refuses to spill
    onto our partner replica's GPUs. Per-GPU budget defaults to a generous
    240 GiB for B300 (275 GiB cards) — overridable via
    TEUTONIC_SHARD_PER_GPU_GIB env knob (or per-call arg) for B200 / smaller.
    """
    if per_gpu_gib is None:
        per_gpu_gib = int(os.environ.get("TEUTONIC_SHARD_PER_GPU_GIB", "240"))
    n_visible = torch.cuda.device_count()
    max_memory: dict = {}
    for gid in range(n_visible):
        if gid in gpu_ids:
            max_memory[gid] = f"{per_gpu_gib}GiB"
        else:
            # Hard zero so the king's device_map can't accidentally land
            # weights on the challenger's GPUs (or vice versa).
            max_memory[gid] = "0GiB"
    return max_memory


def load_model(repo, device, label="model", force_download=False, revision=None,
               shard_across_gpus: list[int] | None = None):
    """Load a model, either onto a single GPU (legacy) or sharded across a
    set of GPUs via accelerate's device_map='auto'.

    Args:
        repo:                Hippius repo id or local path to load
        device:              for single-GPU mode, a string like "cuda:0".
                             Ignored when shard_across_gpus is set.
        label:               human-readable tag for log lines
        force_download:      bypass HF cache
        revision:            pinned OCI digest
        shard_across_gpus:   if set, list of GPU indices to shard across.
                             Builds one replica via device_map='auto' with
                             max_memory caps that ban every other visible GPU.
                             Used by the LXXX 80B chain (no model fits on one
                             B200, must shard across 4 GPUs per replica).
    """
    if shard_across_gpus:
        target = f"sharded({','.join(str(g) for g in shard_across_gpus)})"
    else:
        target = device
    log.info("loading %s from %s onto %s (force_download=%s, digest=%s)",
             label, repo, target, force_download, revision[:19] if revision else None)
    t0 = time.time()
    # Pre-download with hard timeout so a stuck CDN doesn't hang the eval lock
    # for half an hour. Skip on force_download (let from_pretrained re-pull).
    if not force_download:
        try:
            _prefetch_repo(repo, digest=revision,
                           timeout=int(os.environ.get("HIPPIUS_PREFETCH_TIMEOUT", "600")))
            log.info("%s prefetch complete in %.1fs", label, time.time() - t0)
        except TimeoutError as e:
            log.error("%s prefetch timed out: %s", label, e)
            raise
        except Exception as e:
            log.warning("%s prefetch failed (%s), letting from_pretrained retry", label, e)
    if shard_across_gpus:
        device_map_arg: dict | str = "auto"
        max_memory = _build_sharded_device_map(shard_across_gpus)
        load_kwargs = {"device_map": device_map_arg, "max_memory": max_memory}
    else:
        load_kwargs = {"device_map": {"": device}}
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            local_repo = repo
            if revision and not os.path.isdir(repo):
                local_repo = _prefetch_repo(repo, digest=revision,
                                           timeout=int(os.environ.get("HIPPIUS_PREFETCH_TIMEOUT", "600")))
            model = AutoModelForCausalLM.from_pretrained(
                local_repo,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                force_download=False,
                use_safetensors=True,
                **load_kwargs,
            )
            log.info("using attn_implementation=%s", attn_impl)
            break
        except Exception as e:
            log.warning("attn %s failed (%s), trying next", attn_impl, e)
    else:
        raise RuntimeError("could not load model with any attention implementation")
    model.eval()
    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e9
    if shard_across_gpus and hasattr(model, "hf_device_map"):
        # Compact summary: how many submodules per GPU
        from collections import Counter
        per_gpu = Counter(model.hf_device_map.values())
        log.info("%s sharded: %.1fB params in %.1fs (modules/GPU: %s)",
                 label, params, elapsed, dict(per_gpu))
    else:
        log.info("%s loaded: %.1fB params in %.1fs", label, params, elapsed)
    return model


# ---------------------------------------------------------------------------
# Trainability probe — five-layer anti-finetune defense for pretraining
# ---------------------------------------------------------------------------
#
# The attack pattern: a miner takes a good model and "locks" the weights so
# it still wins the paired-CE eval but resists further fine-tuning. If this
# works, their model wins emissions forever — no one can continue-pretrain
# and improve on it. The five layers below each catch a different variant of
# the attack. ANY one tripping ⇒ the probe returns ok=False (the eval server
# rejects the candidate as untrainable).
#
# Layer 1 — Static LayerNorm/RMSNorm weight cap.
#     |weight|.max() across every *Norm module must be <= FINETUNE_NORM_WEIGHT_MAX.
#     Catches the obvious "pump LN gains by 1000x to explode gradients" trick.
#     Runs before any compute so a watermarked model burns ~0 GPU time to reject.
#
# Layer 2 — Live forward + backward, finiteness check.
#     Forward must not raise. Loss must be finite. Backward must not raise.
#     No p.grad may contain NaN/Inf. This is the property pretraining needs:
#     if the model can't even produce one finite gradient on a real batch, no
#     amount of LR tuning will save the next training run.
#
# Layer 3 — Global gradient L2 norm cap (FINETUNE_GRAD_NORM_MAX).
#     Catches generic loss-surface-rigging. Pretraining LRs are 1e-4 to 1e-3,
#     so a |grad| of 500 already moves weights by 0.05–0.5 per step — that
#     destroys any model. If the candidate's grads are this big out of the
#     box, it's brittle by construction.
#
# Layer 4 — Per-parameter-type gradient L2 norm cap.
#     Same threshold but applied per category (attn / ffn / embed / lm_head /
#     norm / bias / other). Catches surgical attacks (embedding poisoning,
#     lm_head poisoning, single-group perturbations) that blow up one bucket
#     while keeping the global norm under the cap.
#
# Layer 5 — Multi-seed rotation across random-token batches.
#     Run PROBE_SEEDS independent random batches, each going through layers
#     2-4. Random tokens are correct here because pretraining sees an
#     arbitrary input distribution and brittleness lives in parameter-space
#     geometry, not in the probe text. Multiple seeds force the candidate to
#     be fine-tunable on more than one fixed batch — a miner that hard-codes
#     a regularizer around one batch fails on the others.

# Static cap — any LN/RMSNorm tensor with |weight|.max() above this is
# rejected without running compute. Honest gemma3-style RMSNorm gains sit
# below ~1.5; allowing 30 covers exotic-but-legit scales while sitting two
# orders of magnitude under the 1000x watermark attack.
FINETUNE_NORM_WEIGHT_MAX = float(os.environ.get(
    "TEUTONIC_FINETUNE_NORM_WEIGHT_MAX", "30"
))

# Global gradient L2 norm cap. A healthy ~1B-7B model gradient on a 256-token
# random batch is typically O(1)–O(10); 500 is comfortably above honest
# models and well below the explosion that LN-pump attacks produce.
FINETUNE_GRAD_NORM_MAX = float(os.environ.get(
    "TEUTONIC_FINETUNE_GRAD_NORM_MAX", "500"
))

# Per-parameter-type cap (same numeric value as the global cap, applied
# group-by-group). Catches targeted attacks that isolate the explosion to
# one parameter group.
FINETUNE_PARAM_GROUP_GRAD_MAX = float(os.environ.get(
    "TEUTONIC_FINETUNE_PARAM_GROUP_GRAD_MAX", "500"
))

# Existing knobs kept (only batch shape + multi-seed rotation count are used;
# PROBE_LR / PROBE_STEPS / PROBE_LOSS_RATIO from the old probe are gone).
PROBE_BATCH = int(os.environ.get("TEUTONIC_PROBE_BATCH", "4"))
PROBE_SEQ_LEN = int(os.environ.get("TEUTONIC_PROBE_SEQ_LEN", "256"))
PROBE_SEEDS = int(os.environ.get("TEUTONIC_PROBE_SEEDS", "3"))
PROBE_SEED = int(os.environ.get("TEUTONIC_PROBE_SEED", str(0xC0FFEE)))
NORM_QUANT_WARN_SCORE = float(os.environ.get("TEUTONIC_NORM_QUANT_WARN", "0.5"))


def math_isfinite(x: float) -> bool:
    """Module-local isfinite that handles bf16 .float() outputs."""
    return x == x and x not in (float("inf"), float("-inf"))


def _classify_param(name: str) -> str:
    """Bucket a parameter into one of {attn, ffn, embed, lm_head, norm, bias, other}.

    The order matters: lm_head/embed checks come first (they sometimes share
    paths with `norm`), then norm, then attn/ffn structural buckets, then
    bias as a fallback for unattributed `.bias` tensors. We use a coarse
    classifier on purpose — the point is per-bucket ‖grad‖ visibility, not
    perfect taxonomy.
    """
    n = name.lower()
    if "lm_head" in n:
        return "lm_head"
    if "embed_tokens" in n or "wte" in n or n.endswith(".embedding.weight") or "embeddings" in n:
        return "embed"
    if "norm" in n or "layernorm" in n or "rmsnorm" in n:
        return "norm"
    # Quasar's HybridBlock uses ln1 / ln1_out / ln2 / ln2_out as the sandwich
    # RMSNorm names — those strings don't contain "norm" so the generic match
    # above misses them; bucket them explicitly so grad-norm logs stay coherent.
    if any(s in n for s in (
        ".ln1.", ".ln2.", ".ln1_out.", ".ln2_out.", ".embed_norm.",
    )):
        return "norm"
    # Quasar latent-memory + GLA-side projections (not learned attention).
    if any(k in n for k in (
        ".memory.", "summary_proj", "summary_query", "compress_z",
        "w_qkv_mem", "eta_channels", "w_eta",
        "c_to_hidden", "w_alpha",
    )):
        return "memory"
    # Quasar BigMac MoE — routed expert weights, DCCA bottleneck, router.
    if "router" in n or "router_weights" in n:
        return "moe_router"
    if "experts_w12" in n or "experts_w3" in n:
        return "moe_routed"
    if "shared_experts" in n:
        return "moe_shared"
    if "w_down_proj" in n or "w_up_proj" in n:
        return "moe_dcca"
    # SMEBU global stability buffers (state_dict but not parameters).
    if "moe_bias" in n or "moe_momentum" in n or "max_vio" in n or "expert_bias" in n:
        return "moe_smebu"
    if "injection_gate" in n:
        return "looped_inject"
    if any(k in n for k in (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "q_norm", "k_norm",
        "wq.", "wk.", "wv.", "wo.",
        "self_attn", "attention",
        # Quasar/GLA-style attention extras (gate / forget / decay / conv).
        "g_proj", "f_proj", "a_proj", "b_proj", ".attn.",
    )):
        return "attn"
    if any(k in n for k in (
        "gate_proj", "up_proj", "down_proj",
        "mlp", "ffn", "fc1", "fc2", "feed_forward",
        ".w1.", ".w2.", ".w3.",
        # Dense Quasar layer FFN tensor names (SwiGLUBlock).
        "ffn.gate", "ffn.up", "ffn.down",
    )):
        return "ffn"
    if n.endswith(".bias"):
        return "bias"
    return "other"


def _norm_modules(model):
    return [(n, m) for n, m in model.named_modules() if "Norm" in type(m).__name__]


def _check_norm_weight_cap(model) -> tuple[bool, str | None, float]:
    """Layer 1 — static LN/RMSNorm weight cap.

    Returns (ok, reason, max_norm_weight_seen). Walks every *Norm module's
    `.weight` and rejects if any element exceeds the cap or is non-finite.
    """
    max_seen = 0.0
    for mod_name, mod in _norm_modules(model):
        for pname, p in mod.named_parameters(recurse=False):
            if not pname.endswith("weight"):
                continue
            with torch.no_grad():
                w = float(p.detach().abs().max().item())
            if not math_isfinite(w):
                return (False,
                        f"norm_weight_non_finite:{mod_name}.{pname} |w|.max()={w}",
                        max_seen)
            if w > max_seen:
                max_seen = w
            if w > FINETUNE_NORM_WEIGHT_MAX:
                return (False,
                        (f"norm_weight_cap:{mod_name}.{pname} "
                         f"|w|.max()={w:.3e} > {FINETUNE_NORM_WEIGHT_MAX:.1f}"),
                        max_seen)
    return True, None, max_seen


def norm_quantization_score(model) -> float | None:
    """Forensic auxiliary signal: how clustered are normalization-layer norms?

    Walks every *Norm `.weight`, computes its L2 norm rounded to 4 decimals,
    and returns the fraction sharing the most common value. 1.0 means every
    norm tensor has the same L2 norm (highly suspicious). Returns None if
    the model has no norm-like modules.

    Surfaced as a warning, never a rejection reason on its own.
    """
    try:
        from collections import Counter

        rounded = []
        for _mod_name, mod in _norm_modules(model):
            for pname, p in mod.named_parameters(recurse=False):
                if not pname.endswith("weight"):
                    continue
                with torch.no_grad():
                    n = float(torch.linalg.vector_norm(p.float()).item())
                rounded.append(round(n, 4))

        if not rounded:
            return None
        _most_common, count = Counter(rounded).most_common(1)[0]
        return count / len(rounded)
    except Exception:
        log.warning("norm_quantization_score failed", exc_info=True)
        return None


def _seed_for_iteration(i: int) -> int:
    """Stable per-seed-index PRNG seed derived from PROBE_SEED."""
    return (PROBE_SEED ^ (0x9E3779B1 * (i + 1))) & 0xFFFFFFFF


def _build_probe_verdict(*, ok, reason, status,
                         max_norm_weight, per_seed,
                         norm_quant, warnings):
    """Wrap probe results into the dict shape every caller expects.

    Aggregates over `per_seed` (one entry per executed seed) and emits both
    the new diagnostic fields (status, max_norm_weight, global_grad_norm,
    param_group_grad_norms) and the legacy fields (loss_before, loss_after,
    delta, max_ratio, max_grad_norm, min_loss_before, max_loss_after,
    n_seeds, n_steps_per_seed) so existing eval_server consumers keep working.
    """
    losses = [s["loss"] for s in per_seed
              if s.get("loss") is not None and math_isfinite(s.get("loss", float("nan")))]
    grads = [s["global_grad_norm"] for s in per_seed
             if s.get("global_grad_norm") is not None
             and math_isfinite(s.get("global_grad_norm", float("nan")))]

    first_loss = losses[0] if losses else float("nan")
    min_loss = min(losses) if losses else float("nan")
    max_loss = max(losses) if losses else float("nan")
    max_grad = max(grads) if grads else float("nan")

    # Aggregate per-group grad norms: max over seeds, per category.
    agg_groups: dict[str, float] = defaultdict(float)
    for s in per_seed:
        for cat, gn in (s.get("param_group_grad_norms") or {}).items():
            if math_isfinite(gn) and gn > agg_groups[cat]:
                agg_groups[cat] = float(gn)

    return {
        # Primary verdict.
        "ok": ok,
        "status": status,
        "reason": reason,
        # Layer 1.
        "max_norm_weight": max_norm_weight,
        "norm_weight_cap": FINETUNE_NORM_WEIGHT_MAX,
        # Layers 2-4 (aggregate).
        "global_grad_norm": max_grad,
        "global_grad_norm_cap": FINETUNE_GRAD_NORM_MAX,
        "param_group_grad_norms": dict(agg_groups),
        "param_group_grad_norm_cap": FINETUNE_PARAM_GROUP_GRAD_MAX,
        # Forensic.
        "norm_quantization": norm_quant,
        "warnings": warnings or [],
        # Per-seed traces (full record for debugging / dashboard).
        "per_seed": per_seed,
        # Legacy fields preserved for the existing verdict consumers.
        "loss_before": first_loss,
        "loss_after": first_loss,  # no SGD step is taken
        "delta": 0.0,
        "max_ratio": 1.0,
        "max_grad_norm": max_grad,
        "min_loss_before": min_loss,
        "max_loss_after": max_loss,
        "n_seeds": len(per_seed),
        "n_steps_per_seed": 0,
    }


def _probe_one_seed(model, seed: int, device, vocab_size: int) -> dict:
    """Run layers 2-4 on a single random-token batch.

    Returns a per-seed dict: {ok, reason, loss, global_grad_norm,
    param_group_grad_norms, seed}. Caller is responsible for clearing
    p.grad before and after.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    vs = max(2, vocab_size or 32000)
    tokens = torch.randint(0, vs, (PROBE_BATCH, PROBE_SEQ_LEN + 1),
                           device=device, generator=g)
    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()

    base = {
        "seed": seed,
        "loss": None,
        "global_grad_norm": None,
        "param_group_grad_norms": {},
    }

    # Layer 2a — forward.
    try:
        out = model(inputs)
        logits = out.logits if hasattr(out, "logits") else out
        # Sharded-safe: when accelerate has split the model across multiple
        # GPUs, `logits` may surface on a different device than `targets`
        # (which started on `device` = embed device). With tied embeddings
        # they coincide; with untied or future variants they may not. Move
        # targets to match.
        if targets.device != logits.device:
            targets = targets.to(logits.device)
        loss_t = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    except Exception as e:
        return {**base, "ok": False,
                "reason": f"forward_raised:{type(e).__name__}:{e}"}

    loss_val = float(loss_t.detach())
    base["loss"] = loss_val
    if not math_isfinite(loss_val):
        return {**base, "ok": False, "reason": f"loss_non_finite:{loss_val}"}

    # Layer 2b — backward.
    try:
        loss_t.backward()
    except Exception as e:
        return {**base, "ok": False,
                "reason": f"backward_raised:{type(e).__name__}:{e}"}

    # Per-param NaN/Inf check.
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all().item():
            return {**base, "ok": False, "reason": f"grad_non_finite:{n}"}

    # Layer 3 — global ‖grad‖₂ cap.
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    if params_with_grad:
        global_gn = float(torch.nn.utils.clip_grad_norm_(
            params_with_grad, max_norm=float("inf")
        ))
    else:
        global_gn = 0.0
    base["global_grad_norm"] = global_gn

    if not math_isfinite(global_gn) or global_gn > FINETUNE_GRAD_NORM_MAX:
        return {**base, "ok": False,
                "reason": (f"global_grad_norm:{global_gn:.3e} > "
                           f"{FINETUNE_GRAD_NORM_MAX:.1f}")}

    # Layer 4 — per-parameter-type ‖grad‖₂ cap.
    sq_by_group: dict[str, float] = defaultdict(float)
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        cat = _classify_param(n)
        with torch.no_grad():
            sq_by_group[cat] += float((p.grad.float() ** 2).sum().item())
    group_norms: dict[str, float] = {cat: sq ** 0.5 for cat, sq in sq_by_group.items()}
    base["param_group_grad_norms"] = group_norms

    for cat, gn in group_norms.items():
        if not math_isfinite(gn) or gn > FINETUNE_PARAM_GROUP_GRAD_MAX:
            return {**base, "ok": False,
                    "reason": (f"param_group_grad:{cat} |grad|={gn:.3e} > "
                               f"{FINETUNE_PARAM_GROUP_GRAD_MAX:.1f}")}

    return {**base, "ok": True, "reason": None}


def trainability_probe(model) -> dict:
    """Five-layer anti-finetune defense; see module docstring above for the layers.

    Single-arg drop-in: same signature, same return-dict shape (legacy fields
    kept) as the previous probe. Restores model state (training mode,
    requires_grad, p.grad) on exit so the caller's evaluator is byte-identical
    after this returns.

    The probe is a property test of fine-tunability for pretraining: a
    candidate that passes can take a real CE backward step on arbitrary
    inputs without exploding, on every parameter bucket, with sane
    gradients. Any failure ⇒ ok=False, status="anti_finetune".
    """
    device = next(model.parameters()).device
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0)) or 32000

    norm_quant = norm_quantization_score(model)
    warnings: list[str] = []
    if norm_quant is not None and norm_quant >= NORM_QUANT_WARN_SCORE:
        warnings.append(
            f"norm_quantization={norm_quant:.3f} >= {NORM_QUANT_WARN_SCORE:.2f} "
            f"(suspicious clustering of normalization-tensor norms)"
        )

    # Layer 1 — static norm-weight cap (no compute).
    ok1, reason1, max_norm_w = _check_norm_weight_cap(model)
    if not ok1:
        return _build_probe_verdict(
            ok=False, reason=reason1, status="anti_finetune",
            max_norm_weight=max_norm_w, per_seed=[],
            norm_quant=norm_quant, warnings=warnings,
        )

    # Layers 2-5: snapshot training state, run multi-seed probes, restore.
    saved_rg = {n: p.requires_grad for n, p in model.named_parameters()}
    was_training = model.training

    # Enable gradient checkpointing for the probe so backward fits in VRAM
    # on disk-tight pods. With an 82B Qwen3-MoE sharded across 4 B200s
    # (TEUTONIC_SHARD_PER_GPU_GIB=120 leaves only ~58 GiB headroom per
    # GPU) the un-checkpointed backward needs ~120 GiB of activations +
    # gradients per shard and OOMs (observed live 2026-05-08 09:47 UTC,
    # eval-0173, GPU 6 hit 176/178 GiB before crashing). Gradient
    # checkpointing recomputes activations during backward instead of
    # storing them — ~3-5x memory savings, ~30% slower. Probe is only
    # ~5 s either way so the slowdown is invisible. Restored in finally.
    saved_gc_enabled: bool | None = None
    saved_use_cache: bool | None = None
    try:
        if hasattr(model, "is_gradient_checkpointing"):
            saved_gc_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
        if hasattr(getattr(model, "config", None), "use_cache"):
            saved_use_cache = bool(model.config.use_cache)
            # `use_cache=True` is incompatible with gradient checkpointing
            # in HF transformers — it silently disables checkpointing
            # otherwise (would defeat the whole point).
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                # Older transformers signature without kwargs.
                model.gradient_checkpointing_enable()
    except Exception:
        log.warning("probe: failed to enable gradient checkpointing (proceeding)",
                    exc_info=True)

    # Snapshot every buffer so any in-place mutation inside the train()-mode
    # forward (e.g. Quasar's MoE aux-loss-free bias EMA on `all_moe_bias`,
    # `all_moe_momentum`, `all_moe_max_vio`, per-MoE `max_vio`) is reverted on
    # exit. Without this, the cached king's pair-0 replica drifts away from
    # the on-disk checkpoint by one EMA step per probe, which over many
    # evals biases ~25% of bootstrap pairs (only pair 0 sees the probed
    # replicas) relative to a miner's local eval. Buffers on Quasar are
    # ~1MB total so the clone is cheap.
    saved_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    per_seed: list[dict] = []

    try:
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None

        for i in range(max(1, PROBE_SEEDS)):
            seed = _seed_for_iteration(i)
            verdict = _probe_one_seed(model, seed=seed, device=device,
                                       vocab_size=vocab_size)
            per_seed.append(verdict)

            # Clear grads between seeds so the next backward starts clean.
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

            if not verdict["ok"]:
                return _build_probe_verdict(
                    ok=False,
                    reason=f"seed{i}({seed:#010x}):{verdict['reason']}",
                    status="anti_finetune",
                    max_norm_weight=max_norm_w, per_seed=per_seed,
                    norm_quant=norm_quant, warnings=warnings,
                )

        return _build_probe_verdict(
            ok=True, reason=None, status="ok",
            max_norm_weight=max_norm_w, per_seed=per_seed,
            norm_quant=norm_quant, warnings=warnings,
        )
    finally:
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
            p.requires_grad_(saved_rg.get(n, False))
        # Restore buffers byte-for-byte. Use .copy_ in no_grad to avoid
        # disturbing autograd state and to keep buffers' device/dtype.
        with torch.no_grad():
            live_buffers = dict(model.named_buffers())
            for n, snapshot in saved_buffers.items():
                live = live_buffers.get(n)
                if live is None:
                    continue
                live.copy_(snapshot)
        # Restore gradient-checkpointing + use_cache so the king/challenger
        # evaluator path (forward-only, no_grad) doesn't pay the recompute
        # cost on every batch.
        try:
            if saved_gc_enabled is False and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
        except Exception:
            log.warning("probe: failed to restore gradient checkpointing", exc_info=True)
        try:
            if saved_use_cache is not None and hasattr(getattr(model, "config", None), "use_cache"):
                model.config.use_cache = saved_use_cache
        except Exception:
            pass
        if not was_training:
            model.eval()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU evaluator
# ---------------------------------------------------------------------------

class MultiGPUEvaluator:
    """Manages model replicas across GPUs and dispatches batches in parallel.

    Two modes:

    1. **Per-GPU mode** (default, `shard_across_gpus=False`): one full replica
       per entry in `gpu_ids`. Used for models that fit on one GPU
       (e.g. Quasar 24B on a B200). Batches are dispatched in parallel across
       replicas via the internal ThreadPoolExecutor.

    2. **Sharded mode** (`shard_across_gpus=True`): one replica spanning ALL
       GPUs in `gpu_ids` via accelerate's `device_map='auto'`. Used for
       models that don't fit on one GPU (e.g. Qwen3MoE 80B → ~152 GiB bf16,
       too big for a single B200). `compute_losses` and the paired path
       become serial — there is only one replica, so batches go through it
       end-to-end.

    Loads are SERIAL (one GPU at a time) in per-GPU mode. An earlier attempt
    to parallelize with ThreadPoolExecutor introduced silent dtype corruption:
    when two threads ran transformers.from_pretrained() concurrently the
    resulting models were left with mixed bf16/float32 Linear weights (notably
    lm_head). Forward passes then died with
    `RuntimeError: expected mat1 and mat2 to have the same dtype`.
    """

    # Sentinel device-id key for the single sharded replica in self.models.
    SHARDED_KEY = "sharded"

    def __init__(self, repo, gpu_ids, label="model", force_download=False,
                 revision=None, on_phase=None, shard_across_gpus: bool = False):
        self.gpu_ids = gpu_ids
        self.shard_across_gpus = shard_across_gpus
        self.models: dict = {}
        self.devices: dict = {}

        if len(gpu_ids) == 0:
            raise ValueError("need at least one GPU")

        if shard_across_gpus:
            if on_phase:
                try:
                    on_phase({"phase": f"{label}_load_start",
                              "gpu": gpu_ids, "done": 0, "total": 1,
                              "repo": repo, "shard": True})
                except Exception:
                    log.warning("on_phase callback raised (non-fatal)",
                                exc_info=True)
            model = load_model(repo, device=None, label=f"{label}-shard",
                               force_download=force_download,
                               revision=revision,
                               shard_across_gpus=gpu_ids)
            # In sharded mode a "device" for input_ids is the device of the
            # input embedding (where Accelerate places token IDs). lm_head's
            # device matters for the loss compute (see compute_paired_losses).
            in_device = self._infer_input_device(model, gpu_ids)
            self.models[self.SHARDED_KEY] = model
            self.devices[self.SHARDED_KEY] = in_device
            self.pool = None  # sharded mode runs serially
            if on_phase:
                try:
                    on_phase({"phase": f"{label}_load_done",
                              "gpu": gpu_ids, "done": 1, "total": 1,
                              "repo": repo, "shard": True})
                except Exception:
                    log.warning("on_phase callback raised (non-fatal)",
                                exc_info=True)
            log.info("%s evaluator ready (sharded): %d GPUs %s, input on %s",
                     label, len(gpu_ids), gpu_ids, in_device)
            return

        n = len(gpu_ids)
        for i, gid in enumerate(gpu_ids):
            if on_phase:
                try:
                    on_phase({"phase": f"{label}_load_start", "gpu": gid,
                              "done": i, "total": n, "repo": repo})
                except Exception:
                    log.warning("on_phase callback raised (non-fatal)", exc_info=True)
            self.models[gid] = load_model(repo, f"cuda:{gid}",
                                          f"{label}-gpu{gid}",
                                          force_download=force_download,
                                          revision=revision)
            self.devices[gid] = f"cuda:{gid}"
            if on_phase:
                try:
                    on_phase({"phase": f"{label}_load_done", "gpu": gid,
                              "done": i + 1, "total": n, "repo": repo})
                except Exception:
                    log.warning("on_phase callback raised (non-fatal)", exc_info=True)

        self.pool = ThreadPoolExecutor(max_workers=n)
        log.info("%s evaluator ready: %d GPUs %s", label, n, gpu_ids)

    @staticmethod
    def _infer_input_device(model, gpu_ids: list[int]) -> str:
        """Find where Accelerate placed `embed_tokens` — that's where input_ids
        must land. Falls back to gpu_ids[0] if the device map is missing."""
        try:
            for name, dev in model.hf_device_map.items():
                if "embed_tokens" in name:
                    if isinstance(dev, int):
                        return f"cuda:{dev}"
                    return str(dev)
        except Exception:
            pass
        return f"cuda:{gpu_ids[0]}"

    @property
    def primary_model(self):
        """Return the first underlying model (for trainability probe etc.).
        In sharded mode this is the single replica spanning all GPUs."""
        if self.shard_across_gpus:
            return self.models[self.SHARDED_KEY]
        return self.models[self.gpu_ids[0]]

    def compute_losses(self, token_batches):
        """Compute per-sequence CE for `token_batches`.

        Per-GPU mode: split across replicas, dispatch in parallel.
        Sharded mode: run sequentially through the one replica.
        """
        if not token_batches:
            return []

        if self.shard_across_gpus:
            return compute_batch_losses(
                self.models[self.SHARDED_KEY], token_batches,
                self.devices[self.SHARDED_KEY],
            )

        n_gpus = len(self.gpu_ids)
        per_gpu = [[] for _ in range(n_gpus)]
        idx_map = [[] for _ in range(n_gpus)]
        for i, batch in enumerate(token_batches):
            g = i % n_gpus
            per_gpu[g].append(batch)
            idx_map[g].append(i)

        futures = {}
        for g_idx, gid in enumerate(self.gpu_ids):
            if per_gpu[g_idx]:
                fut = self.pool.submit(
                    compute_batch_losses,
                    self.models[gid], per_gpu[g_idx], self.devices[gid],
                )
                futures[fut] = g_idx

        results = [None] * len(token_batches)
        for fut in as_completed(futures):
            g_idx = futures[fut]
            losses = fut.result()
            for local_i, global_i in enumerate(idx_map[g_idx]):
                results[global_i] = losses[local_i]

        return results

    def shutdown(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False)


def compute_paired_multi_gpu(king_eval, chall_eval, token_batches):
    """Compute paired CE for `token_batches` across king and challenger.

    Three modes (auto-detected from `MultiGPUEvaluator.shard_across_gpus`):

    1. **Both sharded** — one batch through king's sharded replica, then the
       same batch through challenger's sharded replica. Sequential because
       each replica already saturates its 4-GPU shard; running them concurrent
       would only help if king and challenger sat on disjoint GPU sets *and*
       per-GPU compute weren't already the bottleneck. We launch both in
       threads anyway since they DO sit on disjoint GPUs (king 0..3, chall
       4..7) — kernels can overlap and the CPU-side cost of two independent
       launches is negligible.
    2. **Both per-GPU** — pair gid-by-gid as before, parallel via thread pool.
    3. **Mixed** — error. We don't support a sharded king vs per-GPU chall
       (or vice versa); the eval-server always builds them with the same
       mode for the same chain.
    """
    if not token_batches:
        return [], []

    king_sharded = getattr(king_eval, "shard_across_gpus", False)
    chall_sharded = getattr(chall_eval, "shard_across_gpus", False)

    if king_sharded != chall_sharded:
        raise RuntimeError(
            "compute_paired_multi_gpu: king and challenger must share the same "
            f"replica mode (king_sharded={king_sharded}, chall_sharded={chall_sharded})"
        )

    if king_sharded:
        # Sharded mode: one replica per side. Use a 2-thread pool to overlap
        # king and challenger forwards (they sit on disjoint GPU sets).
        # `compute_paired_losses` walks both models inside one call, so we
        # instead split into two compute_batch_losses calls (king + chall)
        # so they can run concurrently across the disjoint GPU sets.
        king_model = king_eval.models[king_eval.SHARDED_KEY]
        chall_model = chall_eval.models[chall_eval.SHARDED_KEY]
        king_dev = king_eval.devices[king_eval.SHARDED_KEY]
        chall_dev = chall_eval.devices[chall_eval.SHARDED_KEY]
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            f_k = pool.submit(compute_batch_losses,
                              king_model, token_batches, king_dev)
            f_c = pool.submit(compute_batch_losses,
                              chall_model, token_batches, chall_dev)
            king_losses = f_k.result()
            chall_losses = f_c.result()
        finally:
            pool.shutdown(wait=False)
        return king_losses, chall_losses

    n_pairs = min(len(king_eval.gpu_ids), len(chall_eval.gpu_ids))
    per_pair = [[] for _ in range(n_pairs)]
    idx_map = [[] for _ in range(n_pairs)]
    for i, batch in enumerate(token_batches):
        p = i % n_pairs
        per_pair[p].append(batch)
        idx_map[p].append(i)

    futures = {}
    pool = ThreadPoolExecutor(max_workers=n_pairs)
    for p_idx in range(n_pairs):
        if not per_pair[p_idx]:
            continue
        k_gid = king_eval.gpu_ids[p_idx]
        c_gid = chall_eval.gpu_ids[p_idx]
        fut = pool.submit(
            compute_paired_losses,
            king_eval.models[k_gid], chall_eval.models[c_gid],
            per_pair[p_idx],
            king_eval.devices[k_gid], chall_eval.devices[c_gid],
        )
        futures[fut] = p_idx

    king_results = [None] * len(token_batches)
    chall_results = [None] * len(token_batches)
    for fut in as_completed(futures):
        p_idx = futures[fut]
        k_losses, c_losses = fut.result()
        for local_i, global_i in enumerate(idx_map[p_idx]):
            king_results[global_i] = k_losses[local_i]
            chall_results[global_i] = c_losses[local_i]

    pool.shutdown(wait=False)
    return king_results, chall_results


# ---------------------------------------------------------------------------
# Holdout sampling + paired bootstrap
# ---------------------------------------------------------------------------

# CLI fallback. Production runs eval_server.py, which gets `delta_threshold`
# per-request from the validator (computed as `c · king_loss_ema`). Kept so
# `python -m eval.torch_runner` and scripts/smoke_eval.py still work with a
# sensible default.
EVAL_DELTA = float(os.environ.get("EVAL_DELTA", "0.0025"))


def is_accepted(lcb: float, delta_threshold: float) -> bool:
    return lcb > delta_threshold


def sample_public_holdout(r2, shard_key, public_seed: bytes,
                          n_public: int, seq_len: int,
                          vocab_size: int | None = None
                          ) -> tuple[torch.Tensor, str, dict | None]:
    """Sample `n_public` sequences of `seq_len` tokens from the public corpus.

    Returns (sequences [n_public, seq_len] int64, public_indices_digest hex,
    raw_meta or None). The indices digest is sha256 of the int64 indices
    array used to sample — the second piece of the public audit triple
    (corpus_digest, seed, indices_digest).

    In raw_hippius mode the underlying parquet path is non-index-based so
    `public_indices_digest` becomes sha256 of a `b"raw:<seed_hex>"` marker;
    auditors recompute by replaying the same seed against the same corpus
    digest. The shard mode returns the real indices digest.
    """
    seed_int = int.from_bytes(public_seed, "little")
    seed_str = public_seed.hex()
    if raw_dataset_enabled():
        log.info("public holdout: raw Hippius dataset mode")
        if n_public == 0:
            marker = f"raw:{seed_str}".encode()
            return (torch.empty((0, seq_len), dtype=torch.long),
                    hashlib.sha256(marker).hexdigest(), None)
        raw_sequences, raw_meta = load_raw_sequences(
            r2, n_public, seq_len, seed_str, chain_config.SEED_TOKENIZER_REPO,
        )
        if len(raw_sequences) < n_public:
            log.warning("public holdout undersized: got %d, requested %d",
                        len(raw_sequences), n_public)
        # Bind the digest to the set of source files actually used + the seed.
        # An auditor with the same manifest + seed reproduces the same files
        # and therefore the same digest. File-bytes shifts on the upstream
        # mirror would shift the digest only via the file keys, not the bytes
        # themselves — for a stronger fingerprint, set
        # TEUTONIC_AUDIT_FINGERPRINT_N=8 to hash the first N sampled sequence
        # tensors (probabilistic content binding). Left as a flag rather than
        # default because it makes the digest sequence-order-sensitive.
        used_files = (raw_meta or {}).get("used_files", []) or []
        h = hashlib.sha256()
        for key in sorted(used_files):
            h.update(key.encode())
            h.update(b"\n")
        h.update(public_seed)
        fingerprint_n = int(os.environ.get("TEUTONIC_AUDIT_FINGERPRINT_N", "0") or "0")
        if fingerprint_n > 0 and raw_sequences:
            fp_count = min(fingerprint_n, len(raw_sequences))
            fp_tensor = torch.tensor(raw_sequences[:fp_count], dtype=torch.long)
            h.update(b"|fp|")
            h.update(fp_tensor.numpy().tobytes())
        indices_digest = h.hexdigest()
        return torch.tensor(raw_sequences, dtype=torch.long), indices_digest, raw_meta

    if n_public == 0:
        sentinel = hashlib.sha256(b"shard:empty:" + public_seed).hexdigest()
        return torch.empty((0, seq_len), dtype=torch.long), sentinel, None
    n_tokens = get_shard_info(r2, shard_key)
    n_sequences = n_tokens // seq_len
    actual_n = min(n_public, n_sequences)
    rng = np.random.Generator(np.random.PCG64(seed_int))
    indices = rng.choice(n_sequences, size=actual_n, replace=False).astype("<i8")
    indices_digest = hashlib.sha256(indices.tobytes()).hexdigest()

    log.info("public holdout: shard=%s n=%d/%d", shard_key, actual_n, n_sequences)
    data_offset, shard_data = download_shard(r2, shard_key)
    seq_cache = extract_sequences(shard_data, data_offset, indices.tolist(), seq_len)
    validate_sequence_cache(seq_cache, seq_len, vocab_size)
    # Preserve sample order so the indices_digest matches the sequence order.
    seqs = [seq_cache[i] for i in indices.tolist()]
    return torch.tensor(seqs, dtype=torch.long), indices_digest, None


def run_paired_eval(king_eval, challenger_eval,
                    holdout_seqs: torch.Tensor,
                    public_count: int,
                    boot_seed: bytes,
                    eval_alpha: float,
                    delta_threshold: float,
                    n_bootstrap: int = 10000,
                    batch_size: int = 256,
                    on_progress=None) -> dict:
    """Paired CE duel on a pre-prepared holdout tensor.

    holdout_seqs: int64 [n_total, seq_len]; first `public_count` rows are the
    public component, the rest are private. Bootstrap LCB is computed over all
    `n_total` rows; per-component means are diagnostic.
    """
    if holdout_seqs.dim() != 2:
        raise ValueError(f"holdout_seqs must be 2D, got shape {tuple(holdout_seqs.shape)}")
    n_total = holdout_seqs.shape[0]
    if not (0 <= public_count <= n_total):
        raise ValueError(f"public_count={public_count} out of range [0, {n_total}]")

    n_private = n_total - public_count
    same_evaluator = king_eval is challenger_eval
    sequences = holdout_seqs.tolist()
    batches = [sequences[i:i + batch_size] for i in range(0, n_total, batch_size)]

    log.info("paired eval: n_total=%d (public=%d private=%d) alpha=%s delta=%.6f B=%d",
             n_total, public_count, n_private, eval_alpha, delta_threshold, n_bootstrap)

    king_losses_all: list[float] = []
    chall_losses_all: list[float] = []
    king_sum = chall_sum = 0.0
    total_done = 0
    t0 = time.time()

    for bi, token_batches in enumerate(batches):
        if same_evaluator:
            king_losses = king_eval.compute_losses(token_batches)
            chall_losses = king_losses
        else:
            king_losses, chall_losses = compute_paired_multi_gpu(
                king_eval, challenger_eval, token_batches,
            )
        for kl, cl in zip(king_losses, chall_losses):
            king_losses_all.append(kl)
            chall_losses_all.append(cl)
            king_sum += kl
            chall_sum += cl
            total_done += 1

        elapsed = time.time() - t0
        sps = total_done / elapsed if elapsed > 0 else 0
        d_so_far = np.subtract(king_losses_all, chall_losses_all)
        mu_so_far = float(d_so_far.mean()) if d_so_far.size else 0.0
        log.info("batch %d/%d | done=%d/%d | mu_hat=%.6f | %.1f seq/s",
                 bi + 1, len(batches), total_done, n_total, mu_so_far, sps)
        if on_progress:
            on_progress({
                "done": total_done, "total": n_total,
                "mu_hat": round(mu_so_far, 6),
                "avg_king_loss": round(king_sum / total_done, 6),
                "avg_challenger_loss": round(chall_sum / total_done, 6),
                "seqs_per_sec": round(sps, 1),
            })

    elapsed = time.time() - t0
    king_arr = np.asarray(king_losses_all)
    chall_arr = np.asarray(chall_losses_all)
    d = king_arr - chall_arr
    mu_hat = float(d.mean())
    mu_hat_public = float(d[:public_count].mean()) if public_count else 0.0
    mu_hat_private = float(d[public_count:].mean()) if n_private else 0.0

    boot_rng = np.random.Generator(np.random.PCG64(int.from_bytes(boot_seed, "little")))
    boot_means = np.empty(n_bootstrap)
    n = len(d)
    for b in range(n_bootstrap):
        idx = boot_rng.integers(0, n, size=n)
        boot_means[b] = d[idx].mean()
    lcb = float(np.quantile(boot_means, eval_alpha))

    accepted = is_accepted(lcb, delta_threshold)
    log.info("paired result: mu_hat=%.6f (pub=%.6f priv=%.6f) lcb=%.6f delta=%.6f accepted=%s",
             mu_hat, mu_hat_public, mu_hat_private, lcb, delta_threshold, accepted)

    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "mu_hat_public": round(mu_hat_public, 6),
        "mu_hat_private": round(mu_hat_private, 6),
        "lcb": round(lcb, 6),
        "delta_threshold": delta_threshold,
        "alpha": eval_alpha,
        "n_bootstrap": n_bootstrap,
        "n_sequences": n_total,
        "n_public_seqs": public_count,
        "n_private_seqs": n_private,
        "avg_king_loss": round(king_sum / total_done, 6) if total_done else 0.0,
        "avg_challenger_loss": round(chall_sum / total_done, 6) if total_done else 0.0,
        "wall_time_s": round(elapsed, 1),
        "seqs_per_sec": round(total_done / elapsed, 1) if elapsed > 0 else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_bootstrap_test(king_eval, challenger_eval, r2, shard_key, eval_n,
                       alpha, seq_len, batch_size, seed_str,
                       n_bootstrap=10000, on_progress=None,
                       delta_threshold: float | None = None):
    """Public-only paired bootstrap test (CLI / smoke compat).

    Used by main() and scripts/smoke_eval.py. The eval-server path uses
    sample_public_holdout + sample_private_pool + run_paired_eval directly
    so it can layer the private holdout and surface audit digests.

    `delta_threshold` defaults to EVAL_DELTA for backward compat.
    """
    delta = EVAL_DELTA if delta_threshold is None else float(delta_threshold)
    public_seed = hashlib.blake2b(seed_str.encode(), digest_size=8).digest()
    boot_seed = hashlib.blake2b(seed_str.encode() + b":boot", digest_size=8).digest()

    holdout, indices_digest, raw_meta = sample_public_holdout(
        r2, shard_key, public_seed, eval_n, seq_len,
        vocab_size=_evaluator_vocab_size(king_eval),
    )
    verdict = run_paired_eval(
        king_eval, challenger_eval,
        holdout, holdout.shape[0],
        boot_seed=boot_seed, eval_alpha=alpha,
        delta_threshold=delta, n_bootstrap=n_bootstrap,
        batch_size=batch_size, on_progress=on_progress,
    )
    # Legacy field names that smoke_eval / main() print.
    verdict["delta"] = delta
    verdict["N"] = verdict["n_sequences"]
    verdict["public_indices_digest"] = indices_digest
    if raw_meta is not None:
        verdict["dataset"] = raw_meta
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_gpu_ids(gpu_str):
    if gpu_str == "auto":
        return list(range(torch.cuda.device_count()))
    return [int(x.strip()) for x in gpu_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU PyTorch model eval")
    parser.add_argument("--king", required=True, help="HF repo for king model")
    parser.add_argument("--challenger", required=True, help="HF repo for challenger model")
    parser.add_argument("--n", type=int, default=100, help="Number of sequences to evaluate")
    parser.add_argument("--alpha", type=float, default=0.001, help="Bootstrap confidence level (one-sided)")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap replicates")
    parser.add_argument("--batch-size", type=int, default=64, help="Sequences per batch (split across GPUs)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Tokens per sequence")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU IDs or 'auto' (default: auto)")
    parser.add_argument("--seed", default="test:eval", help="Seed string for deterministic sequence selection")
    parser.add_argument("--shard", default=None, help="Specific shard key (default: first shard from manifest)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for var in ["TEUTONIC_R2_ENDPOINT", "TEUTONIC_R2_ACCESS_KEY", "TEUTONIC_R2_SECRET_KEY"]:
        if var not in os.environ:
            log.error("missing env var: %s", var)
            sys.exit(1)

    gpu_ids = parse_gpu_ids(args.gpus)
    log.info("using GPUs: %s", gpu_ids)

    r2 = R2()

    if raw_dataset_enabled():
        shard_key = args.shard or "raw:hippius:fineweb-edu"
        log.info("using raw Hippius dataset mode")
    elif args.shard:
        shard_key = args.shard
    else:
        manifest = r2.ds_get("dataset/v2/manifest.json")
        if not manifest:
            manifest = r2.get("dataset/v1/manifest.json")
        if not manifest:
            log.error("could not fetch dataset manifest")
            sys.exit(1)
        shard_key = manifest["shards"][0]["key"]
        log.info("using shard: %s (%d shards available, version=%s)",
                 shard_key, len(manifest["shards"]), manifest.get("version", "v1"))

    same_model = args.king == args.challenger

    if same_model:
        log.info("king == challenger, using all %d GPUs for shared evaluator", len(gpu_ids))
        king_eval = MultiGPUEvaluator(args.king, gpu_ids, label="king")
        challenger_eval = king_eval
    else:
        mid = len(gpu_ids) // 2
        king_gpus = gpu_ids[:mid] or gpu_ids[:1]
        chall_gpus = gpu_ids[mid:] or gpu_ids[:1]
        log.info("king GPUs: %s  challenger GPUs: %s", king_gpus, chall_gpus)
        king_eval = MultiGPUEvaluator(args.king, king_gpus, label="king")
        challenger_eval = MultiGPUEvaluator(args.challenger, chall_gpus, label="challenger")

    log.info("=" * 60)
    log.info("EVAL CONFIG")
    log.info("  king:       %s", args.king)
    log.info("  challenger: %s", args.challenger)
    log.info("  GPUs:       %s (%s)", gpu_ids, "shared" if same_model else "split")
    log.info("  N=%d  alpha=%s  delta=%.6f  bootstrap=%d  batch=%d  seq_len=%d",
             args.n, args.alpha, EVAL_DELTA, args.n_bootstrap, args.batch_size, args.seq_len)
    log.info("  shard: %s", shard_key)
    log.info("  seed:  %s", args.seed)
    log.info("=" * 60)

    verdict = run_bootstrap_test(
        king_eval, challenger_eval,
        r2, shard_key, args.n, args.alpha,
        args.seq_len, args.batch_size, args.seed,
        n_bootstrap=args.n_bootstrap,
    )

    king_eval.shutdown()
    if not same_model:
        challenger_eval.shutdown()

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(json.dumps(verdict, indent=2))
    print("=" * 60)

    return 0 if not verdict["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
