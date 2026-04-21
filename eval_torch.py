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
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import boto3
import numpy as np
import torch
import torch.nn.functional as F
from botocore.config import Config as BotoConfig
from transformers import AutoModelForCausalLM

log = logging.getLogger("eval_torch")


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------

class R2:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
            aws_access_key_id=os.environ["TEUTONIC_R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["TEUTONIC_R2_SECRET_KEY"],
            region_name="auto",
            config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
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
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    s3={"addressing_style": "path"},
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

LM_HEAD_CHUNK = 256

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

def load_model(repo, device, label="model", force_download=False, revision=None):
    log.info("loading %s from %s onto %s (force_download=%s, revision=%s)",
             label, repo, device, force_download, revision[:12] if revision else None)
    t0 = time.time()
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
# Trainability probe — first-principles replacement for magnitude heuristics
# ---------------------------------------------------------------------------
#
# The reparameterization trick exploits that f(x; gamma, W) is invariant under
# (gamma, W) -> (alpha*gamma, W/alpha) for any RMSNorm followed by a Linear.
# This is a 1D symmetry of the FORWARD pass, but NOT of training dynamics:
# gradients scale as dL/dgamma ~ |W| (shrinks by alpha) and dL/dW ~ |gamma|
# (grows by alpha). One SGD step at any normal LR moves W by lr*alpha, which
# blows up the model. Honest models barely move under the same step.
#
# This directly tests the property we actually care about: can a miner
# fine-tune from this checkpoint? Trick models say no; honest models say yes.
# Random-token batches are sufficient because brittleness is in parameter-
# space geometry, not input distribution.

PROBE_LR = float(os.environ.get("TEUTONIC_PROBE_LR", "1e-5"))
PROBE_BATCH = int(os.environ.get("TEUTONIC_PROBE_BATCH", "4"))
PROBE_SEQ_LEN = int(os.environ.get("TEUTONIC_PROBE_SEQ_LEN", "256"))
PROBE_LOSS_DELTA_ABS = float(os.environ.get("TEUTONIC_PROBE_LOSS_DELTA_ABS", "5.0"))
PROBE_LOSS_DELTA_REL = float(os.environ.get("TEUTONIC_PROBE_LOSS_DELTA_REL", "2.0"))
PROBE_SEED = int(os.environ.get("TEUTONIC_PROBE_SEED", str(0xC0FFEE)))


def trainability_probe(model) -> dict:
    """Take one SGD step on a random-token batch; return a verdict dict.

    Result keys:
      ok: bool — True if model survived the step (loss didn't explode).
      reason: str | None — short rejection reason if not ok.
      loss_before: float
      loss_after: float
      delta: float (loss_after - loss_before)

    Restores all parameters to their pre-probe values via a per-param data
    snapshot (.clone() on each .data buffer) regardless of success or failure.
    """
    device = next(model.parameters()).device
    vocab_size = int(getattr(model.config, "vocab_size", 0)) or 32000

    g = torch.Generator(device=device).manual_seed(PROBE_SEED)
    tokens = torch.randint(0, vocab_size, (PROBE_BATCH, PROBE_SEQ_LEN + 1),
                           device=device, generator=g)
    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()

    snapshot = {n: p.data.clone() for n, p in model.named_parameters()}
    was_training = model.training
    saved_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}

    def _forward_loss():
        out = model(inputs)
        logits = out.logits if hasattr(out, "logits") else out
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    try:
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None

        loss_before_t = _forward_loss()
        loss_before = float(loss_before_t.detach())
        if not math_isfinite(loss_before):
            return {
                "ok": False,
                "reason": f"loss_before_non_finite:{loss_before}",
                "loss_before": loss_before,
                "loss_after": float("nan"),
                "delta": float("nan"),
            }

        loss_before_t.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-PROBE_LR)

        with torch.no_grad():
            loss_after_t = _forward_loss()
        loss_after = float(loss_after_t.detach())
        delta = loss_after - loss_before

        if not math_isfinite(loss_after):
            return {
                "ok": False,
                "reason": f"loss_after_non_finite:{loss_after}",
                "loss_before": loss_before,
                "loss_after": loss_after,
                "delta": delta,
            }

        # Reject only if BOTH absolute and relative thresholds exceeded —
        # avoids penalizing models with a small honest loss baseline (where
        # 2x of 0.05 would otherwise be a meaningless 0.1).
        explosive = (
            delta > PROBE_LOSS_DELTA_ABS
            and loss_after > PROBE_LOSS_DELTA_REL * max(loss_before, 1e-3)
        )
        if explosive:
            return {
                "ok": False,
                "reason": (f"loss_explosion:before={loss_before:.4f} "
                           f"after={loss_after:.4f} delta={delta:.4f} "
                           f"(thresh abs>{PROBE_LOSS_DELTA_ABS} "
                           f"and after>{PROBE_LOSS_DELTA_REL}x before)"),
                "loss_before": loss_before,
                "loss_after": loss_after,
                "delta": delta,
            }

        return {
            "ok": True,
            "reason": None,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "delta": delta,
        }
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(snapshot[n])
                if p.grad is not None:
                    p.grad = None
                p.requires_grad_(saved_requires_grad.get(n, False))
        if not was_training:
            model.eval()
        # Drop snapshot refs to free GPU memory promptly.
        snapshot.clear()
        torch.cuda.empty_cache()


def math_isfinite(x: float) -> bool:
    """Module-local isfinite that handles bf16 .float() outputs."""
    return x == x and x not in (float("inf"), float("-inf"))


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
