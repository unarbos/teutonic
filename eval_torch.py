#!/usr/bin/env python3
"""Standalone multi-GPU PyTorch eval — king-vs-challenger paired bootstrap test.

Loads model replicas across all available GPUs, fetches sequences from R2
with prefetch overlap, and computes cross-entropy loss via chunked lm_head
forward passes to minimize VRAM. Accepts the challenger only when the
bootstrapped lower confidence bound on the per-token log-loss advantage
exceeds a configurable delta threshold.

Usage:
    python eval_torch.py \
        --king unconst/Teutonic-I \
        --challenger unconst/Teutonic-I \
        --n 100 --delta 0.01 --batch-size 64 --seq-len 2048 --gpus 0,1,2,3,4,5,6,7

Env vars:
    HF_TOKEN              HuggingFace token for gated repos
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
import hashlib
import io
import json
import logging
import os
import pathlib
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# hf-xet (the Rust chunked-CDN downloader) ignores huggingface_hub's HTTP
# timeouts and has been observed to hang for hours on partial responses,
# wedging the eval lock. Force the plain HTTPS path which honors socket
# timeouts via requests/urllib3. Must be set BEFORE huggingface_hub imports.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Conservative per-chunk read timeout for the plain HTTPS download path.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
# Belt-and-suspenders: any unguarded socket op blocks at most this long.
socket.setdefaulttimeout(180)

import boto3
import numpy as np
import torch
import torch.nn.functional as F
from botocore.config import Config as BotoConfig
from collections import defaultdict
from transformers import AutoModelForCausalLM

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

def get_shard_info(r2, shard_key):
    header = r2.ds_range_get(shard_key, 0, 1023)
    buf = io.BytesIO(header)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    hdr = eval(buf.read(hl).decode("latin1").strip())
    n = 1
    for s in hdr["shape"]:
        n *= s
    return n


R2_FETCH_WORKERS = 32

def _parse_shard_header(r2, shard_key):
    header = r2.ds_range_get(shard_key, 0, 1023)
    buf = io.BytesIO(header)
    buf.read(6)  # magic
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    buf.read(hl)
    return buf.tell()


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


def download_shard(r2, shard_key):
    """Download shard with parallel streams and local disk cache."""
    cache_name = shard_key.replace("/", "_")
    cache_path = pathlib.Path(SHARD_CACHE_DIR) / cache_name

    if cache_path.exists():
        t0 = time.time()
        raw = cache_path.read_bytes()
        data_offset = _parse_npy_header(raw)
        elapsed = time.time() - t0
        log.info("shard cache HIT %s: %.1f MB read in %.2fs",
                 shard_key, len(raw) / 1e6, elapsed)
        return data_offset, raw

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


# ---------------------------------------------------------------------------
# Chunked loss computation — avoids materializing full [batch, seq, vocab]
# ---------------------------------------------------------------------------

LM_HEAD_CHUNK = int(os.environ.get("TEUTONIC_LM_HEAD_CHUNK", "512"))

@torch.no_grad()
def compute_batch_losses(model, token_batches, device, chunk_size=LM_HEAD_CHUNK):
    """Forward pass with chunked lm_head to avoid OOM on large vocabs.

    Instead of model(input_ids).logits which allocates [batch, seq, vocab],
    we get hidden states first then apply lm_head in small chunks along the
    sequence dimension. Peak VRAM drops ~7x for vocab_size=262144.
    """
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    hidden = model.model(input_ids).last_hidden_state
    lm_head = model.lm_head

    n_positions = input_ids.size(1) - 1
    total_loss = torch.zeros(len(token_batches), device=device)

    for i in range(0, n_positions, chunk_size):
        end_pos = min(i + chunk_size, n_positions)
        chunk_logits = lm_head(hidden[:, i:end_pos, :])
        chunk_labels = input_ids[:, i + 1 : end_pos + 1]
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
    """
    B = len(token_batches)
    input_ids_k = torch.tensor(token_batches, dtype=torch.long, device=king_device)
    input_ids_c = torch.tensor(token_batches, dtype=torch.long, device=chall_device)

    hidden_k = king_model.model(input_ids_k).last_hidden_state
    hidden_c = chall_model.model(input_ids_c).last_hidden_state

    n_pos = input_ids_k.size(1) - 1
    king_loss = torch.zeros(B, device=king_device)
    chall_loss = torch.zeros(B, device=chall_device)

    for i in range(0, n_pos, chunk_size):
        end = min(i + chunk_size, n_pos)

        logits_k = king_model.lm_head(hidden_k[:, i:end, :])
        logits_c = chall_model.lm_head(hidden_c[:, i:end, :])

        labels_k = input_ids_k[:, i + 1 : end + 1]
        labels_c = input_ids_c[:, i + 1 : end + 1]
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

def _prefetch_repo(repo, revision=None, timeout=600):
    """Pre-download repo files via huggingface_hub with an explicit wall-clock
    cap. Runs in a background thread so we can abandon a hung download (the
    actual blocked socket can't be cancelled, but this lets the caller fail
    fast and let the eval-server watchdog reclaim the eval lock).
    """
    from huggingface_hub import snapshot_download
    import threading

    result = {"path": None, "err": None}

    def _do():
        try:
            result["path"] = snapshot_download(
                repo_id=repo,
                revision=revision or None,
                token=os.environ.get("HF_TOKEN") or None,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
                etag_timeout=int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "30")),
            )
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_do, daemon=True, name=f"hf-prefetch-{repo}")
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"prefetch of {repo} exceeded {timeout}s (likely stuck CDN)")
    if result["err"] is not None:
        raise result["err"]
    return result["path"]


def load_model(repo, device, label="model", force_download=False, revision=None):
    log.info("loading %s from %s onto %s (force_download=%s, revision=%s)",
             label, repo, device, force_download, revision[:12] if revision else None)
    t0 = time.time()
    # Pre-download with hard timeout so a stuck CDN doesn't hang the eval lock
    # for half an hour. Skip on force_download (let from_pretrained re-pull).
    if not force_download:
        try:
            _prefetch_repo(repo, revision=revision,
                           timeout=int(os.environ.get("HF_PREFETCH_TIMEOUT", "600")))
            log.info("%s prefetch complete in %.1fs", label, time.time() - t0)
        except TimeoutError as e:
            log.error("%s prefetch timed out: %s", label, e)
            raise
        except Exception as e:
            log.warning("%s prefetch failed (%s), letting from_pretrained retry", label, e)
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                attn_implementation=attn_impl,
                token=os.environ.get("HF_TOKEN") or None,
                force_download=force_download,
                revision=revision or None,
                use_safetensors=True,
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
    if any(k in n for k in (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "q_norm", "k_norm",
        "wq.", "wk.", "wv.", "wo.",
        "self_attn", "attention",
    )):
        return "attn"
    if any(k in n for k in (
        "gate_proj", "up_proj", "down_proj",
        "mlp", "ffn", "fc1", "fc2", "feed_forward",
        ".w1.", ".w2.", ".w3.",
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
        if not was_training:
            model.eval()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU evaluator
# ---------------------------------------------------------------------------

class MultiGPUEvaluator:
    """Manages model replicas across GPUs and dispatches batches in parallel."""

    def __init__(self, repo, gpu_ids, label="model", force_download=False, revision=None):
        self.gpu_ids = gpu_ids
        self.models = {}
        self.devices = {}

        if len(gpu_ids) == 0:
            raise ValueError("need at least one GPU")

        first_model = load_model(repo, f"cuda:{gpu_ids[0]}", f"{label}-gpu{gpu_ids[0]}",
                                 force_download=force_download, revision=revision)
        self.models[gpu_ids[0]] = first_model
        self.devices[gpu_ids[0]] = f"cuda:{gpu_ids[0]}"

        for gid in gpu_ids[1:]:
            self.models[gid] = load_model(repo, f"cuda:{gid}", f"{label}-gpu{gid}",
                                          force_download=force_download, revision=revision)
            self.devices[gid] = f"cuda:{gid}"

        self.pool = ThreadPoolExecutor(max_workers=len(gpu_ids))
        log.info("%s evaluator ready: %d GPUs %s", label, len(gpu_ids), gpu_ids)

    def compute_losses(self, token_batches):
        """Split token_batches across GPUs, compute in parallel, reassemble."""
        n_gpus = len(self.gpu_ids)
        if not token_batches:
            return []

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
        self.pool.shutdown(wait=False)


def compute_paired_multi_gpu(king_eval, chall_eval, token_batches):
    """Pair king GPUs with challenger GPUs to compute losses in parallel."""
    if not token_batches:
        return [], []

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
# Bootstrap test
# ---------------------------------------------------------------------------

def run_bootstrap_test(king_eval, challenger_eval, r2, shard_key, eval_n,
                       alpha, delta, seq_len, batch_size, seed_str,
                       n_bootstrap=10000, on_progress=None):
    """Paired bootstrap test on per-token log-loss differences.

    Scores M fixed-length blocks on both models, computes d_i = king_loss_i -
    challenger_loss_i (positive means challenger is better), then bootstraps the
    mean to get a one-sided lower confidence bound (LCB).  Accepts only if
    LCB > delta.

    Calls on_progress(info_dict) after each batch if provided.
    """
    n_tokens = get_shard_info(r2, shard_key)
    n_sequences = n_tokens // seq_len
    actual_N = min(eval_n, n_sequences)
    log.info("bootstrap test: N=%d actual_N=%d alpha=%s delta=%.6f B=%d",
             eval_n, actual_N, alpha, delta, n_bootstrap)

    seed_material = seed_str.encode()
    seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

    log.info("downloading shard %s ...", shard_key)
    data_offset, shard_data = download_shard(r2, shard_key)

    log.info("extracting %d sequences", actual_N)
    seq_cache = extract_sequences(shard_data, data_offset, eval_indices, seq_len)
    log.info("extracted %d sequences", len(seq_cache))

    batches = [
        eval_indices[i : i + batch_size]
        for i in range(0, len(eval_indices), batch_size)
    ]

    all_diffs = []
    king_sum, chall_sum = 0.0, 0.0
    total_done = 0
    t0 = time.time()

    same_evaluator = king_eval is challenger_eval

    for bi, batch_indices in enumerate(batches):
        token_batches = [seq_cache[idx] for idx in batch_indices]

        if same_evaluator:
            king_losses = king_eval.compute_losses(token_batches)
            chall_losses = king_losses
        else:
            king_losses, chall_losses = compute_paired_multi_gpu(
                king_eval, challenger_eval, token_batches,
            )

        for k_loss, c_loss in zip(king_losses, chall_losses):
            total_done += 1
            king_sum += k_loss
            chall_sum += c_loss
            all_diffs.append(k_loss - c_loss)

        elapsed = time.time() - t0
        seqs_per_sec = total_done / elapsed if elapsed > 0 else 0
        mu_hat = np.mean(all_diffs) if all_diffs else 0.0
        log.info(
            "batch %d/%d | done=%d/%d | mu_hat=%.6f | %.1f seq/s",
            bi + 1, len(batches), total_done, actual_N, mu_hat, seqs_per_sec,
        )

        if on_progress:
            on_progress({
                "done": total_done, "total": actual_N,
                "mu_hat": round(float(mu_hat), 6),
                "avg_king_loss": round(king_sum / total_done, 6),
                "avg_challenger_loss": round(chall_sum / total_done, 6),
                "seqs_per_sec": round(seqs_per_sec, 1),
            })

    elapsed = time.time() - t0
    d = np.array(all_diffs)
    mu_hat = float(d.mean())

    boot_rng = np.random.Generator(np.random.PCG64(seed ^ 0xB007))
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = boot_rng.integers(0, len(d), size=len(d))
        boot_means[b] = d[idx].mean()
    lcb = float(np.quantile(boot_means, alpha))

    accepted = lcb > delta
    log.info("bootstrap result: mu_hat=%.6f lcb=%.6f delta=%.6f accepted=%s",
             mu_hat, lcb, delta, accepted)

    verdict = {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta": delta,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "N": actual_N,
        "avg_king_loss": round(king_sum / total_done, 6) if total_done else 0,
        "avg_challenger_loss": round(chall_sum / total_done, 6) if total_done else 0,
        "wall_time_s": round(elapsed, 1),
        "seqs_per_sec": round(total_done / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
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
    parser.add_argument("--delta", type=float, default=0.01, help="Minimum effect threshold in nats/token")
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

    if args.shard:
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
             args.n, args.alpha, args.delta, args.n_bootstrap, args.batch_size, args.seq_len)
    log.info("  shard: %s", shard_key)
    log.info("  seed:  %s", args.seed)
    log.info("=" * 60)

    verdict = run_bootstrap_test(
        king_eval, challenger_eval,
        r2, shard_key, args.n, args.alpha, args.delta,
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
