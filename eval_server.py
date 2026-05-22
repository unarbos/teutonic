#!/usr/bin/env python3
"""Eval server — persistent FastAPI service wrapping eval_torch.py.

Runs on the GPU box. Caches the king model across evals, reloads only when
the repo changes. Streams progress via SSE.

Usage:
    uvicorn eval_server:app --host 127.0.0.1 --port 9000

Env vars: same as eval_torch.py (TEUTONIC_R2_* plus Hippius Hub token env vars)
    EVAL_HOST   Bind address (default: 127.0.0.1, set to 0.0.0.0 only behind a firewall)
    TEUTONIC_EVAL_DATASET_MODE=raw_hippius to read the FineWeb-Edu Hippius
        Parquet mirror and tokenize at eval time.
"""
import asyncio
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from queue import Queue, Empty
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import chain_config  # noqa: E402
chain_config.load_arch()

from eval.torch_runner import (  # noqa: E402
    R2, MultiGPUEvaluator, parse_gpu_ids,
    trainability_probe, load_model, raw_dataset_enabled,
    sample_public_holdout, run_paired_eval, is_accepted,
)
from eval.raw_dataset import sample_private_pool  # noqa: E402
from model_store import (  # noqa: E402
    MODEL_CACHE_DIR,
    ModelRef,
    local_snapshot_path,
    sha256_safetensors,
)

log = logging.getLogger("eval_server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_gpu_ids: list[int] = []
_r2: R2 | None = None
_king_evaluator: MultiGPUEvaluator | None = None
_king_repo: str | None = None
_king_hash: str | None = None
_king_digest: str | None = None
_eval_lock = threading.Lock()
_evals: dict[str, dict] = {}

# Self-kill plumbing — see _is_cuda_fatal / _schedule_self_kill below.
# Set once a fatal-CUDA exit has been scheduled so we never schedule twice.
_self_kill_scheduled = threading.Event()

# Set whenever /eval or /probe is doing real work that competes with model
# downloads. Cleared in _run_eval's finally and in probe_endpoint's finally.
# Safe with the new self-kill: if an eval poisons CUDA and dies, the
# supervisor restarts the process which resets this event.
_gpu_busy = threading.Event()

DEFAULT_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "256"))
DEFAULT_ALPHA = float(os.environ.get("EVAL_ALPHA", "0.001"))
DEFAULT_SEQ_LEN = int(os.environ.get("EVAL_SEQ_LEN", "2048"))
DEFAULT_BOOTSTRAP_B = int(os.environ.get("EVAL_BOOTSTRAP_B", "10000"))
DEFAULT_N_PUBLIC = int(os.environ.get("EVAL_N_PUBLIC", "1000"))
DEFAULT_N_PRIVATE = int(os.environ.get("EVAL_N_PRIVATE", "1000"))

# Server-side caps. The validator can request a larger eval_n / n_bootstrap
# in its POST body; we clamp to these to keep per-eval wall time bounded
# while clearing a backed-up duel queue. Restore via env if not needed.
EVAL_N_CAP = int(os.environ.get("EVAL_N_CAP", "20000"))
EVAL_BOOTSTRAP_B_CAP = int(os.environ.get("EVAL_BOOTSTRAP_B_CAP", "999999"))

PROBE_ENABLED = os.environ.get("TEUTONIC_PROBE_ENABLED", "1") == "1"

EVAL_MAX_RUNTIME_S = int(os.environ.get("EVAL_MAX_RUNTIME_S", "1800"))

# Sharded mode: when set, build ONE replica per side via accelerate
# device_map='auto' across that side's GPU subset, instead of one full replica
# per GPU. Used by the LXXX 80B chain (152 GiB bf16 doesn't fit on a single
# B200). Default off so the live Quasar 24B chain on the production eval pod
# keeps its current per-GPU behavior unless explicitly opted in.
SHARD_ACROSS_GPUS = os.environ.get("TEUTONIC_SHARD_ACROSS_GPUS", "0") == "1"


# ---------------------------------------------------------------------------
# Fatal-CUDA self-kill
# ---------------------------------------------------------------------------
# Once a CUDA context corruption (illegal memory access / device-side assert
# / misaligned address / etc.) hits any thread on this process, the VRAM
# allocator and every cuStream are unsafe — every subsequent .from_pretrained
# / .forward / .empty_cache will keep raising. Historically (2026-05-03 14:08
# - 16:50 UTC) this poisoned the box for ~2.5 h: 10 evals in a row failed
# with "could not load model with any attention implementation" while the
# server stayed alive but degraded.
#
# The only safe recovery is to exit the process so the supervisor (see
# eval_server_loop.sh) brings it back with a fresh CUDA context. We give a
# brief delay so the in-flight SSE error event reaches the validator, then
# os._exit (NOT sys.exit, NOT regular exit) to skip atexit hooks that would
# touch the corrupted GPU state and hang.
_CUDA_FATAL_TOKENS = (
    "an illegal memory access",
    "cudaErrorIllegalAddress",
    "device-side assert",
    "CUDA error: misaligned address",
    "CUDA error: unspecified launch failure",
    "CUDA error: an illegal instruction",
    "CUBLAS_STATUS_EXECUTION_FAILED",
    "CUBLAS_STATUS_NOT_INITIALIZED",
    "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED",
    "Bus error",
    "Segmentation fault",
)

CUDA_FATAL_EXIT_DELAY_S = float(os.environ.get("CUDA_FATAL_EXIT_DELAY_S", "3"))
CUDA_FATAL_EXIT_CODE = int(os.environ.get("CUDA_FATAL_EXIT_CODE", "75"))


# sha256 over the eval-pipeline source files. Echoed back to the validator
# so an external auditor can pin the exact code that produced a verdict.
# Aggregated digest (concatenated bytes); auditor rehashes the same three
# files in the same order.
def _compute_eval_code_digest() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for relpath in ("eval/torch_runner.py", "eval/raw_dataset.py", "eval_server.py",
                    "chain_config.py", "model_store.py"):
        with open(here / relpath, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


_EVAL_CODE_DIGEST = _compute_eval_code_digest()


def _is_cuda_fatal(exc_msg: str) -> bool:
    s = str(exc_msg or "")
    return any(tok in s for tok in _CUDA_FATAL_TOKENS)


def _schedule_self_kill(reason: str, delay_s: float | None = None) -> None:
    """Schedule a hard exit because CUDA state is unrecoverable. Idempotent."""
    if _self_kill_scheduled.is_set():
        return
    _self_kill_scheduled.set()
    delay = float(delay_s if delay_s is not None else CUDA_FATAL_EXIT_DELAY_S)
    log.error("FATAL CUDA STATE: %s — exiting in %.1fs (code=%d) for "
              "supervisor restart", reason, delay, CUDA_FATAL_EXIT_CODE)

    def _die():
        try:
            time.sleep(delay)
        except Exception:
            pass
        try:
            log.error("self-killing now (cuda-fatal)")
        except Exception:
            pass
        os._exit(CUDA_FATAL_EXIT_CODE)

    threading.Thread(target=_die, daemon=True,
                     name="cuda-fatal-self-kill").start()


def _install_thread_excepthook() -> None:
    """Catch CUDA-fatal exceptions that escape our try/except blocks
    (e.g. from a daemon thread inside MultiGPUEvaluator). Without this hook,
    such exceptions just print a traceback and the process keeps running
    with corrupted CUDA state."""
    prior = threading.excepthook

    def _hook(args):
        try:
            msg = f"{args.exc_type.__name__}: {args.exc_value}"
            if _is_cuda_fatal(msg):
                _schedule_self_kill(
                    f"uncaught in thread {getattr(args.thread, 'name', '?')}: {msg}"
                )
        finally:
            prior(args)

    threading.excepthook = _hook


_install_thread_excepthook()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu_ids, _r2
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _gpu_ids = parse_gpu_ids(os.environ.get("EVAL_GPUS", "auto"))
    log.info("eval server starting with GPUs: %s", _gpu_ids)
    _r2 = R2()
    # NOTE: don't cleanup on startup — the cache may hold the current king from
    # a previous run, and re-downloading 16GB takes ~3min. After-eval cleanup
    # (in run_eval) keeps disk usage bounded between evals.
    if os.environ.get("EVAL_CLEANUP_ON_STARTUP", "0") == "1":
        _cleanup_model_cache()
    # Start the background disk-stats refresher and prime the snapshot so
    # the first /health call is fast (rather than blocking on a synchronous
    # 600 GB model cache scan from inside the request handler).
    _ensure_disk_stats_thread()
    yield
    log.info("eval server shutting down")
    if _king_evaluator:
        _king_evaluator.shutdown()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ProbeRequest(BaseModel):
    repo: str
    digest: str = ""


class EvalRequest(BaseModel):
    king_repo: str
    challenger_repo: str
    block_hash: str
    hotkey: str
    delta_threshold: float = Field(..., ge=0.0, lt=1.0)
    n_public: int = DEFAULT_N_PUBLIC
    n_private: int = DEFAULT_N_PRIVATE
    king_hash: str = ""
    king_digest: str = ""
    challenger_digest: str = ""
    shard_key: str = ""
    alpha: float = DEFAULT_ALPHA
    seq_len: int = DEFAULT_SEQ_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def _ensure_king(repo: str, king_hash: str = "", digest: str = "",
                 on_phase=None):
    """Load or reuse king evaluator. Reloads if repo, digest, or king_hash changed.

    On a fresh load, runs the trainability probe on the king. A king that fails
    the probe is a violation of an invariant (the king got there by winning an
    eval, which already required passing the probe), so we refuse to load it
    and raise — operator must intervene.

    `on_phase`, if provided, is invoked with a phase dict before/after each
    per-GPU load and around the trainability probe. Used to emit SSE
    heartbeats so the validator's stream-idle watchdog stays satisfied
    during the multi-minute king-reload that follows a coronation.
    """
    global _king_evaluator, _king_repo, _king_hash, _king_digest
    if (_king_evaluator and _king_repo == repo
            and (not digest or _king_digest == digest)
            and (not king_hash or _king_hash == king_hash)):
        log.info("reusing cached king evaluator for %s (rev=%s)",
                 repo, (_king_digest or "?")[:19])
        return _king_evaluator

    needs_reload = _king_evaluator is not None
    if needs_reload:
        log.info("king changed (%s rev=%s -> %s rev=%s), reloading",
                 _king_repo, (_king_digest or "?")[:19],
                 repo, digest[:19] if digest else "?")
        _king_evaluator.shutdown()
        _king_evaluator = None
        torch.cuda.empty_cache()

    mid = len(_gpu_ids) // 2
    king_gpus = _gpu_ids[:mid] or _gpu_ids[:1]
    new_king = MultiGPUEvaluator(repo, king_gpus, label="king",
                                  force_download=False,
                                  revision=digest or None,
                                  on_phase=on_phase,
                                  shard_across_gpus=SHARD_ACROSS_GPUS)

    if PROBE_ENABLED:
        if on_phase:
            try:
                on_phase({"phase": "king_probe_start", "repo": repo})
            except Exception:
                log.warning("on_phase callback raised (non-fatal)", exc_info=True)
        # In sharded mode there's one replica spanning all king_gpus; in
        # per-GPU mode we probe the first replica. `primary_model` abstracts
        # both.
        king_model = new_king.primary_model
        t0 = time.time()
        probe = trainability_probe(king_model)
        log.info("king trainability probe for %s: ok=%s "
                 "max_ratio=%.3f max_grad=%.2e min_before=%.4f "
                 "max_after=%.4f norm_quant=%s seeds=%d steps=%d (%.1fs)",
                 repo, probe["ok"],
                 probe.get("max_ratio", float("nan")),
                 probe.get("max_grad_norm", float("nan")),
                 probe.get("min_loss_before", float("nan")),
                 probe.get("max_loss_after", float("nan")),
                 probe.get("norm_quantization"),
                 probe.get("n_seeds", 0),
                 probe.get("n_steps_per_seed", 0),
                 time.time() - t0)
        for w in probe.get("warnings", []) or []:
            log.warning("king %s probe warning: %s", repo, w)
        if not probe["ok"]:
            log.error("KING TRAINABILITY PROBE FAILED for %s: %s. "
                      "Refusing to load this king. Operator intervention required.",
                      repo, probe["reason"])
            new_king.shutdown()
            del new_king
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"king {repo}@{(digest or '?')[:19]} failed trainability "
                f"probe: {probe['reason']}"
            )

    _king_evaluator = new_king
    _king_repo = repo
    _king_hash = king_hash or None
    _king_digest = digest or None
    return _king_evaluator


def _evict_for_challenger(target_repo: str):
    """Local Hippius cache cleanup happens in _cleanup_model_cache."""
    _ = target_repo
    _cleanup_model_cache()

def _load_challenger(repo: str, digest: str = "", on_phase=None):
    """Load challenger on the second half of GPUs.

    In sharded mode (TEUTONIC_SHARD_ACROSS_GPUS=1) the challenger occupies
    the full second-half GPU set as one accelerate-sharded replica."""
    # Force-clear any stale challenger from a previous eval before we
    # download this one. Otherwise on a disk-tight pod we can wedge at
    # ENOSPC mid-download and surface as the misleading "could not load
    # model with any attention implementation" error from the eager
    # fallback in `load_model`.
    _evict_for_challenger(repo)
    mid = len(_gpu_ids) // 2
    chall_gpus = _gpu_ids[mid:] or _gpu_ids[:1]
    return MultiGPUEvaluator(repo, chall_gpus, label="challenger",
                              revision=digest or None,
                              on_phase=on_phase,
                              shard_across_gpus=SHARD_ACROSS_GPUS)


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

MAX_EVALS_KEPT = 50
EVAL_MAX_AGE_S = 3600

CACHE_HIGH_WATERMARK_GB = float(os.environ.get("MODEL_CACHE_HIGH_WATERMARK_GB", "200"))


def _cleanup_model_cache():
    """Delete oldest local Hippius snapshots above the disk watermark."""
    try:
        root = Path(MODEL_CACHE_DIR)
        if not root.exists():
            return
        snapshots = [p for p in root.glob("**/snapshots/*") if p.is_dir()]
        total = sum(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in snapshots)
        cache_gb = total / 1e9
        if cache_gb < CACHE_HIGH_WATERMARK_GB:
            return
        keep = {(_king_repo, _king_digest)}
        candidates = []
        for d in snapshots:
            try:
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                candidates.append((d.stat().st_mtime, d, size))
            except Exception:
                continue
        target_bytes = CACHE_HIGH_WATERMARK_GB * 0.7 * 1e9
        running = total
        for _mtime, d, size in sorted(candidates):
            if running < target_bytes:
                break
            # Digest snapshots are named by digest with `:` → `-` (see
            # model_store._cache_snapshot_path); keep the loaded king.
            if _king_digest and d.name == _king_digest.replace(":", "-"):
                continue
            shutil.rmtree(d, ignore_errors=True)
            running -= size
            log.info("model cache cleanup: deleted %s (%.1f GB)", d, size / 1e9)
    except Exception:
        log.warning("model cache cleanup failed", exc_info=True)

def _prune_evals():
    """Remove old completed/failed eval records to bound memory usage."""
    try:
        now = time.time()
        to_remove = []
        for eid, rec in _evals.items():
            if rec["state"] not in ("completed", "failed"):
                continue
            age = now - rec.get("created_at", now)
            if age > EVAL_MAX_AGE_S:
                to_remove.append(eid)

        if len(_evals) - len(to_remove) > MAX_EVALS_KEPT:
            finished = sorted(
                ((eid, rec) for eid, rec in _evals.items()
                 if rec["state"] in ("completed", "failed") and eid not in to_remove),
                key=lambda x: x[1].get("created_at", 0),
            )
            excess = len(_evals) - len(to_remove) - MAX_EVALS_KEPT
            for eid, _ in finished[:excess]:
                to_remove.append(eid)

        for eid in to_remove:
            del _evals[eid]

        if to_remove:
            log.info("pruned %d old eval records, %d remaining", len(to_remove), len(_evals))
    except Exception:
        log.warning("eval pruning failed", exc_info=True)


# Disk-stats snapshot, refreshed by a dedicated background thread.
# model cache scan() walks ~600 GB of cache and can take 15-30 s under
# heavy IO; on top of that, the default asyncio executor is shared with
# the long-running /probe and /eval blocking tasks, so even
# `loop.run_in_executor(None, _get_disk_stats)` would queue behind them
# during an eval and miss the /health timeout. Running the refresh on
# its own thread decouples /health latency from anything the eval does.
_DISK_STATS_REFRESH_S = float(os.environ.get("DISK_STATS_REFRESH_S", "30"))
_disk_stats_snapshot: dict = {}
_disk_stats_thread_started = False
_disk_stats_thread_lock = threading.Lock()


def _refresh_disk_stats_once():
    stats = {}
    try:
        usage = shutil.disk_usage("/")
        stats["disk_total_gb"] = round(usage.total / 1e9, 1)
        stats["disk_used_gb"] = round(usage.used / 1e9, 1)
        stats["disk_free_gb"] = round(usage.free / 1e9, 1)
    except Exception:
        pass
    try:
        root = Path(MODEL_CACHE_DIR)
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        stats["model_cache_size_gb"] = round(sum(p.stat().st_size for p in files) / 1e9, 2)
        stats["model_cache_files"] = len(files)
    except Exception:
        pass
    return stats

def _disk_stats_loop():
    global _disk_stats_snapshot
    while True:
        try:
            _disk_stats_snapshot = _refresh_disk_stats_once()
        except Exception:
            log.warning("disk-stats refresh raised", exc_info=True)
        time.sleep(_DISK_STATS_REFRESH_S)


def _ensure_disk_stats_thread():
    """Start the background disk-stats refresher. Does NOT block on the
    first prime — that runs in the same background thread, so lifespan()
    finishes fast (boot in ~10 s instead of ~90 s waiting on a 600 GB
    model cache scan). /health responses during the first ~30-60 s will
    lack the disk_*/model_cache_* fields, which is fine — the surrounding
    `status: ok` is enough for liveness/readiness checks.

    We need the boot to be under the validator's retry budget
    (3 retries × 30 s = 90 s) so a watchdog-triggered self-kill
    doesn't cause the validator to drop in-flight evals."""
    global _disk_stats_thread_started
    with _disk_stats_thread_lock:
        if _disk_stats_thread_started:
            return
        threading.Thread(target=_disk_stats_loop, daemon=True,
                         name="disk-stats-refresher").start()
        _disk_stats_thread_started = True


def _get_disk_stats():
    """Return cached disk usage stats. Non-blocking after first call."""
    _ensure_disk_stats_thread()
    return dict(_disk_stats_snapshot)


# ---------------------------------------------------------------------------
# Eval runner (runs in a thread)
# ---------------------------------------------------------------------------

def _derive_seeds(block_hash: str, hotkey: str) -> tuple[bytes, bytes, bytes]:
    """Deterministic per-duel seeds. `public_seed` and `boot_seed` are pinned
    in the audit record; `private_seed` is validator-side replay (the private
    pool is not published, but the same validator must be able to reproduce
    its own past duel by re-running with the same block_hash + hotkey)."""
    material = block_hash.encode() + hotkey.encode()
    public_seed = hashlib.blake2b(material + b"public", digest_size=8).digest()
    boot_seed = hashlib.blake2b(material + b"boot", digest_size=8).digest()
    private_seed = hashlib.blake2b(material + b"private", digest_size=8).digest()
    return public_seed, boot_seed, private_seed


_PUBLIC_CORPUS_DIGEST_CACHE: dict = {"digest": "", "expires": 0.0, "fallback": False}
_PUBLIC_CORPUS_DIGEST_TTL_S = 60.0


def _public_corpus_digest() -> tuple[str, bool]:
    """sha256 of the actual public corpus manifest bytes. Returns (digest, fallback).

    `fallback=True` means the manifest fetch failed and the digest is derived
    from the tokenizer string only — auditors should treat the verdict as
    degraded for replay purposes.
    """
    now = time.monotonic()
    if _PUBLIC_CORPUS_DIGEST_CACHE["digest"] and now < _PUBLIC_CORPUS_DIGEST_CACHE["expires"]:
        return _PUBLIC_CORPUS_DIGEST_CACHE["digest"], _PUBLIC_CORPUS_DIGEST_CACHE["fallback"]
    digest, fallback = "", False
    try:
        from eval.raw_dataset import RawDatasetConfig
        cfg = RawDatasetConfig.from_env(chain_config.SEED_TOKENIZER_REPO)
        if _r2 is not None:
            body = _r2.ds_client.get_object(
                Bucket=_r2.ds_bucket, Key=cfg.manifest_key,
            )["Body"].read()
            digest = hashlib.sha256(body).hexdigest()
    except Exception:
        log.warning("public_corpus_digest: manifest fetch failed; using tokenizer fallback",
                    exc_info=True)
    if not digest:
        material = chain_config.SEED_TOKENIZER_REPO.encode()
        digest = hashlib.sha256(material).hexdigest()
        fallback = True
    _PUBLIC_CORPUS_DIGEST_CACHE["digest"] = digest
    _PUBLIC_CORPUS_DIGEST_CACHE["fallback"] = fallback
    _PUBLIC_CORPUS_DIGEST_CACHE["expires"] = now + _PUBLIC_CORPUS_DIGEST_TTL_S
    return digest, fallback


def _run_eval(eval_id: str, req: EvalRequest):
    record = _evals[eval_id]
    record["state"] = "running"
    _gpu_busy.set()
    event_q: Queue = record["events"]

    # Heartbeat callback: turn load-phase signals from MultiGPUEvaluator and
    # the probes into SSE `progress` events. The validator's idle watchdog
    # resets on every yielded line, so any phase event prevents the silent
    # multi-minute gap (king reload + challenger load + probe + first batch)
    # from tripping STREAM_IDLE_TIMEOUT and orphaning the eval.
    def _on_phase(info):
        try:
            event_q.put({"type": "progress", "data": info})
        except Exception:
            log.warning("failed to enqueue heartbeat event (non-fatal)", exc_info=True)

    # Periodic ticker: catches phases that don't naturally subdivide. The
    # cold-cache Hippius download inside load_model._prefetch_repo can take
    # 5-10 min and emits no per-GPU phase events of its own, so without
    # this ticker the validator's idle watchdog can still trip during a
    # one-off cold challenger fetch. Cancelled in `finally:`.
    _heartbeat_stop = threading.Event()

    def _heartbeat_loop():
        while not _heartbeat_stop.wait(30.0):
            try:
                event_q.put({"type": "progress", "data": {"phase": "heartbeat"}})
            except Exception:
                log.warning("heartbeat ticker enqueue failed (non-fatal)", exc_info=True)

    _heartbeat_thread = threading.Thread(target=_heartbeat_loop,
                                          name=f"heartbeat-{eval_id[:8]}",
                                          daemon=True)
    _heartbeat_thread.start()

    try:
        n_public = max(0, int(req.n_public))
        n_private = max(0, int(req.n_private))
        total = n_public + n_private
        if total > EVAL_N_CAP:
            # Scale both sides proportionally so we keep the public/private
            # ratio the validator asked for; integer floor + remainder goes
            # to public so the cap is hard.
            scale = EVAL_N_CAP / total
            new_pub = int(n_public * scale)
            new_priv = int(n_private * scale)
            slack = EVAL_N_CAP - (new_pub + new_priv)
            new_pub += slack
            n_public, n_private = new_pub, new_priv
        n_bootstrap = min(req.n_bootstrap, EVAL_BOOTSTRAP_B_CAP)
        if n_public != req.n_public or n_private != req.n_private or n_bootstrap != req.n_bootstrap:
            log.info("eval %s: capped n_public %d->%d n_private %d->%d n_bootstrap %d->%d",
                     eval_id, req.n_public, n_public, req.n_private, n_private,
                     req.n_bootstrap, n_bootstrap)

        public_seed, boot_seed, private_seed = _derive_seeds(req.block_hash, req.hotkey)

        # Kick off shard download in the background so it overlaps with king
        # reload + challenger load + probe. Saves ~30s/eval once the shard
        # cache is cold for that key. No-op in raw-dataset mode.
        if req.shard_key and not raw_dataset_enabled():
            try:
                from eval.torch_runner import prefetch_shard
                prefetch_shard(_r2, req.shard_key)
            except Exception:
                log.warning("shard prefetch kickoff failed (non-fatal)", exc_info=True)

        king_eval = _ensure_king(req.king_repo, req.king_hash, req.king_digest,
                                 on_phase=_on_phase)

        same_model = (req.king_repo == req.challenger_repo
                      and req.king_digest == req.challenger_digest)
        if same_model:
            challenger_eval = king_eval
        else:
            # Pre-download cleanup: free space for the ~165 GB challenger
            # before _load_challenger calls _prefetch_repo. Otherwise an
            # earlier eval's challenger sitting in cache + king + new
            # challenger can blow disk capacity, manifesting as the
            # misleading "could not load model with any attention
            # implementation" error (which is really ENOSPC during
            # safetensors mmap). See _cleanup_model_cache docstring.
            try:
                _cleanup_model_cache()
            except Exception:
                log.warning("eval %s: pre-load cleanup failed", eval_id, exc_info=True)
            challenger_eval = _load_challenger(req.challenger_repo, req.challenger_digest,
                                               on_phase=_on_phase)

        if not same_model and PROBE_ENABLED:
            _on_phase({"phase": "challenger_probe_start", "repo": req.challenger_repo})
            # In sharded mode there's one replica spanning all challenger GPUs;
            # `primary_model` resolves correctly for both per-GPU and sharded.
            chall_model = challenger_eval.primary_model
            t0 = time.time()
            probe = trainability_probe(chall_model)
            log.info("trainability probe for %s: ok=%s "
                     "max_ratio=%.3f max_grad=%.2e min_before=%.4f "
                     "max_after=%.4f norm_quant=%s seeds=%d steps=%d (%.1fs)",
                     req.challenger_repo, probe["ok"],
                     probe.get("max_ratio", float("nan")),
                     probe.get("max_grad_norm", float("nan")),
                     probe.get("min_loss_before", float("nan")),
                     probe.get("max_loss_after", float("nan")),
                     probe.get("norm_quantization"),
                     probe.get("n_seeds", 0),
                     probe.get("n_steps_per_seed", 0),
                     time.time() - t0)
            for w in probe.get("warnings", []) or []:
                log.warning("challenger %s probe warning: %s",
                            req.challenger_repo, w)
            if not probe["ok"]:
                log.warning("trainability probe REJECTED %s: %s",
                            req.challenger_repo, probe["reason"])

                challenger_eval.shutdown()
                del challenger_eval
                torch.cuda.empty_cache()

                verdict = {
                    "accepted": False,
                    "verdict": "king",
                    "rejection_reason": f"untrainable:{probe['reason']}",
                    "probe": {
                        "loss_before": probe["loss_before"],
                        "loss_after": probe["loss_after"],
                        "delta": probe["delta"],
                        "max_ratio": probe.get("max_ratio"),
                        "max_grad_norm": probe.get("max_grad_norm"),
                        "min_loss_before": probe.get("min_loss_before"),
                        "max_loss_after": probe.get("max_loss_after"),
                        "n_seeds": probe.get("n_seeds"),
                        "n_steps_per_seed": probe.get("n_steps_per_seed"),
                        "norm_quantization": probe.get("norm_quantization"),
                        "warnings": probe.get("warnings", []),
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "eval_code_digest": _EVAL_CODE_DIGEST,
                }
                record["state"] = "completed"
                record["verdict"] = verdict
                event_q.put({"type": "verdict", "data": verdict})
                return

        def _on_progress(info):
            record["progress"] = info
            event_q.put({"type": "progress", "data": info})

        vocab_size = None
        try:
            vocab_size = int(getattr(getattr(king_eval.primary_model, "config", None),
                                     "vocab_size", 0)) or None
        except Exception:
            pass

        public_seqs, public_indices_digest, raw_meta = sample_public_holdout(
            _r2, req.shard_key, public_seed, n_public, req.seq_len,
            vocab_size=vocab_size,
        )

        if n_private > 0:
            private_seqs, private_pool_digest = sample_private_pool(
                req.seq_len, n_private, chain_config.SEED_TOKENIZER_REPO,
                rng_seed=private_seed,
            )
        else:
            private_seqs = torch.zeros((0, req.seq_len), dtype=torch.int64)
            private_pool_digest = ""

        actual_public = public_seqs.shape[0]
        holdout = public_seqs if private_seqs.shape[0] == 0 else torch.cat([public_seqs, private_seqs], dim=0)

        verdict = run_paired_eval(
            king_eval, challenger_eval,
            holdout, actual_public,
            boot_seed=boot_seed,
            eval_alpha=req.alpha,
            delta_threshold=req.delta_threshold,
            n_bootstrap=n_bootstrap,
            batch_size=req.batch_size,
            on_progress=_on_progress,
        )

        _pcd, _pcd_fallback = _public_corpus_digest()
        verdict["public_corpus_digest"] = _pcd
        verdict["public_corpus_digest_fallback"] = _pcd_fallback
        verdict["public_seed"] = public_seed.hex()
        verdict["public_indices_digest"] = public_indices_digest
        verdict["private_pool_digest"] = private_pool_digest
        verdict["boot_seed"] = boot_seed.hex()
        verdict["eval_code_digest"] = _EVAL_CODE_DIGEST
        if raw_meta is not None:
            verdict["dataset"] = raw_meta

        if not same_model:
            challenger_eval.shutdown()
            del challenger_eval
            torch.cuda.empty_cache()

        record["state"] = "completed"
        record["verdict"] = verdict
        event_q.put({"type": "verdict", "data": verdict})

    except Exception as e:
        log.exception("eval %s failed", eval_id)
        record["state"] = "failed"
        record["error"] = str(e)
        event_q.put({"type": "error", "data": {"error": str(e)}})
        if _is_cuda_fatal(str(e)):
            _schedule_self_kill(f"in eval {eval_id}: {type(e).__name__}: {e}")

    finally:
        _heartbeat_stop.set()
        _gpu_busy.clear()
        try:
            _eval_lock.release()
        except RuntimeError:
            log.warning("eval %s: eval_lock was not held at release time", eval_id)
        try:
            _cleanup_model_cache()
        except Exception:
            log.warning("eval %s: model cleanup failed", eval_id, exc_info=True)
        try:
            _prune_evals()
        except Exception:
            log.warning("eval %s: prune failed", eval_id, exc_info=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/hash")
async def hash_endpoint(repo: str, digest: str = ""):
    """sha256 over cached safetensors of a Hippius digest snapshot."""
    try:
        ref = ModelRef(repo, digest)
        local_dir = local_snapshot_path(ref)
    except Exception as e:
        raise HTTPException(status_code=404,
                            detail=f"{repo}@{digest or 'missing'} not in cache: {type(e).__name__}")
    t0 = time.time()
    value = sha256_safetensors(local_dir)
    if not value:
        raise HTTPException(status_code=404, detail=f"{repo}@{digest}: no safetensors")
    n_files = len(list(Path(local_dir).glob("*.safetensors")))
    log.info("hash: %s@%s -> %s over %d files in %.1fs",
             repo, (digest or "missing")[:19], value[:16], n_files, time.time() - t0)
    return {
        "sha256": value,
        "n_files": n_files,
        "repo": repo,
        "digest": digest,
        "elapsed_s": round(time.time() - t0, 2),
    }

@app.get("/health")
async def health():
    # _get_disk_stats() reads a snapshot updated by a dedicated
    # background thread — never blocks the event loop, never queues
    # behind a long-running /probe or /eval in the executor pool.
    return {
        "status": "exiting" if _self_kill_scheduled.is_set() else "ok",
        "gpus": len(_gpu_ids),
        "gpu_ids": _gpu_ids,
        "king_loaded": _king_repo,
        "active_evals": len(_evals),
        "self_kill_scheduled": _self_kill_scheduled.is_set(),
        "gpu_busy": _gpu_busy.is_set(),
        **_get_disk_stats(),
    }


def _watchdog(eval_id: str, deadline: float):
    """Force-fail an eval that overruns EVAL_MAX_RUNTIME_S, then trigger
    a supervisor-managed restart so the next eval gets clean GPU state.

    The wedged worker thread is *not* killed by Python (no safe thread
    cancellation), so it keeps holding GPU allocations until it naturally
    finishes. Observed 2026-05-04 00:46-00:49 UTC: the previous-watchdog'd
    worker for taoism99/...-3a51b842 was still on batch 14/20 when we
    released the lock; the validator immediately dispatched the retry,
    which OOM'd during the trainability probe (GPU 4 only had 485 MiB
    free) and got falsely rejected as "untrainable" — a unfair outcome
    for the miner.
    
    Recovery is to self-kill (os._exit + supervisor restart). That brings
    up a fresh process in ~100 s with clean GPU state. The validator's
    retry loop handles the brief unavailability cleanly.
    """
    while time.time() < deadline:
        time.sleep(30)
        rec = _evals.get(eval_id)
        if rec is None or rec.get("state") in ("completed", "failed"):
            return
    rec = _evals.get(eval_id)
    if rec is None or rec.get("state") in ("completed", "failed"):
        return
    log.error("watchdog: eval %s exceeded %ds, force-failing and triggering self-kill",
              eval_id, EVAL_MAX_RUNTIME_S)
    rec["state"] = "failed"
    rec["error"] = f"watchdog timeout after {EVAL_MAX_RUNTIME_S}s"
    try:
        rec["events"].put({"type": "error",
                           "data": {"error": rec["error"]}})
    except Exception:
        pass
    try:
        _eval_lock.release()
    except RuntimeError:
        pass
    # Trigger a supervisor restart. The wedged worker keeps holding
    # GPU allocations; without a process restart the next eval can OOM.
    _schedule_self_kill(
        f"watchdog: eval {eval_id} exceeded {EVAL_MAX_RUNTIME_S}s; "
        f"restarting to release leaked GPU memory from wedged worker"
    )


def _run_probe_blocking(repo: str, digest: str) -> dict:
    """Run the trainability probe on `repo`@`digest`, return the verdict.

    Three modes (auto-selected):

    1. **Cached king reuse** — if `repo` matches the currently-loaded
       `_king_evaluator` (and digest matches if specified), reuse its
       primary_model directly. This is the common case for the validator's
       hourly `audit_incumbent_king`. trainability_probe restores p.grad
       and named_buffers in its `finally` block, so the king replica is
       byte-identical after probing.
    2. **Sharded fresh load** — when `TEUTONIC_SHARD_ACROSS_GPUS=1` and the
       model isn't the cached king, load a fresh sharded replica across all
       `_gpu_ids` (the probe holds `_eval_lock`, so no /eval is concurrent
       and we can use every GPU). Required for LXXX-scale models that don't
       fit on one GPU.
    3. **Single-GPU fresh load** — legacy path for chains where the model
       fits on one GPU (e.g. Quasar 24B on a B200/B300).

    Caller must hold `_eval_lock`.
    """
    if not _gpu_ids:
        raise RuntimeError("no GPUs available on eval server")

    global _king_evaluator, _king_repo, _king_digest

    # Mode 1: reuse cached king replica when probing the incumbent.
    if (_king_evaluator is not None
            and _king_repo == repo
            and (not digest or _king_digest == digest)):
        log.info("probe: reusing cached king %s@%s (no reload)",
                 repo, (digest or "missing")[:19])
        t0 = time.time()
        verdict = trainability_probe(_king_evaluator.primary_model)
        probe_s = time.time() - t0
        log.info("probe: %s ok=%s max_ratio=%.3f max_grad=%.2e "
                 "norm_quant=%s (cached, probe=%.1fs)",
                 repo, verdict["ok"],
                 verdict.get("max_ratio", float("nan")),
                 verdict.get("max_grad_norm", float("nan")),
                 verdict.get("norm_quantization"),
                 probe_s)
        for w in verdict.get("warnings", []) or []:
            log.warning("probe %s warning: %s", repo, w)
        verdict["timing"] = {"load_s": 0.0, "probe_s": round(probe_s, 2),
                             "total_s": round(probe_s, 2)}
        verdict["repo"] = repo
        verdict["digest"] = digest
        verdict["cached"] = True
        return verdict

    # Mode 2 + 3: load fresh.
    sharded = SHARD_ACROSS_GPUS and len(_gpu_ids) > 1
    if sharded:
        target = f"sharded({','.join(str(g) for g in _gpu_ids)})"
        log.info("probe: loading %s@%s on %s", repo, (digest or "missing")[:19], target)
    else:
        probe_gpu = _gpu_ids[-1]
        device = f"cuda:{probe_gpu}"
        log.info("probe: loading %s@%s on %s", repo, (digest or "missing")[:19], device)
    t0 = time.time()
    model = None
    try:
        if sharded:
            model = load_model(repo, device=None, label=f"probe-{repo}",
                               force_download=False,
                               revision=digest or None,
                               shard_across_gpus=_gpu_ids)
        else:
            model = load_model(repo, device, label=f"probe-{repo}",
                               force_download=False,
                               revision=digest or None)
        load_s = time.time() - t0
        t1 = time.time()
        verdict = trainability_probe(model)
        probe_s = time.time() - t1
        log.info("probe: %s ok=%s max_ratio=%.3f max_grad=%.2e "
                 "norm_quant=%s (load=%.1fs probe=%.1fs)",
                 repo, verdict["ok"],
                 verdict.get("max_ratio", float("nan")),
                 verdict.get("max_grad_norm", float("nan")),
                 verdict.get("norm_quantization"),
                 load_s, probe_s)
        for w in verdict.get("warnings", []) or []:
            log.warning("probe %s warning: %s", repo, w)
        verdict["timing"] = {
            "load_s": round(load_s, 2),
            "probe_s": round(probe_s, 2),
            "total_s": round(time.time() - t0, 2),
        }
        verdict["repo"] = repo
        verdict["digest"] = digest
        verdict["cached"] = False
        return verdict
    finally:
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


@app.post("/probe")
async def probe_endpoint(req: ProbeRequest):
    """Out-of-band trainability probe on an arbitrary repo+digest.

    Used by the validator to periodically reprobe the incumbent king
    (see audit_incumbent_king in validator.py) without going through a
    full eval. Acquires `_eval_lock` so it can't race with an in-flight
    /eval call competing for the same GPUs.
    """
    if not req.repo:
        raise HTTPException(status_code=400, detail="repo is required")
    if _self_kill_scheduled.is_set():
        raise HTTPException(status_code=503, detail="eval server is restarting")
    acquired = _eval_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=409, detail="an eval is already running")
    _gpu_busy.set()
    try:
        loop = asyncio.get_running_loop()
        verdict = await loop.run_in_executor(
            None, _run_probe_blocking, req.repo, req.digest,
        )
        return verdict
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("probe failed for %s", req.repo)
        if _is_cuda_fatal(str(exc)):
            _schedule_self_kill(f"in probe {req.repo}: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _gpu_busy.clear()
        try:
            _eval_lock.release()
        except RuntimeError:
            pass


@app.post("/eval")
async def start_eval(req: EvalRequest):
    if _self_kill_scheduled.is_set():
        raise HTTPException(status_code=503, detail="eval server is restarting")
    acquired = _eval_lock.acquire(blocking=False)
    if not acquired:
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

    thread = threading.Thread(target=_run_eval, args=(eval_id, req), daemon=True)
    thread.start()
    threading.Thread(
        target=_watchdog,
        args=(eval_id, time.time() + EVAL_MAX_RUNTIME_S),
        daemon=True,
    ).start()

    return {"eval_id": eval_id}


@app.get("/eval/{eval_id}")
async def get_eval(eval_id: str):
    if eval_id not in _evals:
        raise HTTPException(status_code=404, detail="eval not found")
    record = _evals[eval_id]
    return {
        "eval_id": eval_id,
        "state": record["state"],
        "progress": record["progress"],
        "verdict": record["verdict"],
        "error": record["error"],
    }


@app.get("/eval/{eval_id}/stream")
async def stream_eval(eval_id: str):
    if eval_id not in _evals:
        raise HTTPException(status_code=404, detail="eval not found")
    record = _evals[eval_id]
    event_q: Queue = record["events"]

    async def generate():
        while True:
            try:
                event = event_q.get(block=False)
            except Empty:
                await asyncio.sleep(0.5)
                if record["state"] in ("completed", "failed") and event_q.empty():
                    final = record["verdict"] or record.get("error")
                    final_type = "verdict" if record["state"] == "completed" else "error"
                    yield f"data: {json.dumps({'type': final_type, 'data': final})}\n\n"
                    break
                continue

            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("verdict", "error"):
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("EVAL_HOST", "127.0.0.1")
    port = int(os.environ.get("EVAL_PORT", "9000"))
    uvicorn.run("eval_server:app", host=host, port=port, log_level="info")
