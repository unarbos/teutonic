#!/usr/bin/env python3
"""vLLM-backed eval server — drop-in alternative to eval_server.py.

Hosts two persistent vLLM engines on a single multi-GPU box: the king on the
first half of EVAL_GPUS, the challenger on the second half. Both engines run
their forward passes concurrently per eval. Same FastAPI/SSE contract as
`eval_server.py` so validators can be pointed here without changes:

    POST /eval        -> {"eval_id": ...}
    GET  /eval/{id}/stream  -> SSE stream of progress / verdict events
    GET  /eval/{id}   -> JSON snapshot of current state
    POST /probe       -> {ok: ..., reason: ..., ...}
    GET  /health      -> {status, gpus, gpu_ids, king_loaded, ...}

Verdict fields match eval_torch.run_bootstrap_test exactly (with an extra
`backend: "vllm"` for telemetry).

Env vars:
    EVAL_HOST                     bind addr           (default 127.0.0.1)
    EVAL_PORT                     port                (default 9001)
    EVAL_GPUS                     "auto" | "0,1,..."  (default auto)
    EVAL_N / EVAL_ALPHA / EVAL_SEQ_LEN / EVAL_BATCH_SIZE / EVAL_BOOTSTRAP_B
    EVAL_VLLM_DTYPE / EVAL_VLLM_GPU_MEM_UTIL / EVAL_VLLM_ENFORCE_EAGER
    EVAL_VLLM_MAX_BATCH           per-engine submission chunk (default 256)
    EVAL_MAX_RUNTIME_S            watchdog (default 1800)
    TEUTONIC_PROBE_ENABLED        "1" to run HF trainability probe at load
    HF_TOKEN, TEUTONIC_R2_*       same as eval_server.py

Run:
    uvicorn eval_server_vllm:app --host 127.0.0.1 --port 9001
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from queue import Empty, Queue

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from eval_torch import R2, parse_gpu_ids, prefetch_shard
from eval_vllm import VllmEvaluator, run_bootstrap_test_vllm

log = logging.getLogger("eval_server_vllm")

# ---------------------------------------------------------------------------
# Defaults — same env vars as eval_server.py so PM2 envs stay portable
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "256"))
DEFAULT_EVAL_N = int(os.environ.get("EVAL_N", "10000"))
DEFAULT_ALPHA = float(os.environ.get("EVAL_ALPHA", "0.001"))
DEFAULT_SEQ_LEN = int(os.environ.get("EVAL_SEQ_LEN", "2048"))
DEFAULT_BOOTSTRAP_B = int(os.environ.get("EVAL_BOOTSTRAP_B", "10000"))

EVAL_MAX_RUNTIME_S = int(os.environ.get("EVAL_MAX_RUNTIME_S", "1800"))
PROBE_ENABLED = os.environ.get("TEUTONIC_PROBE_ENABLED", "0") == "1"

CACHE_HIGH_WATERMARK_GB = float(
    os.environ.get("HF_CACHE_HIGH_WATERMARK_GB", "400"),
)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_gpu_ids: list[int] = []
_king_gpu_ids: list[int] = []
_chall_gpu_ids: list[int] = []
_r2: R2 | None = None

_king: VllmEvaluator | None = None
_king_repo: str | None = None
_king_revision: str | None = None
_king_hash: str | None = None

_eval_lock = threading.Lock()
_evals: dict[str, dict] = {}

MAX_EVALS_KEPT = 50
EVAL_MAX_AGE_S = 3600


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu_ids, _king_gpu_ids, _chall_gpu_ids, _r2
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _gpu_ids = parse_gpu_ids(os.environ.get("EVAL_GPUS", "auto"))
    if not _gpu_ids:
        raise RuntimeError("no GPUs visible to eval_server_vllm")

    mid = max(1, len(_gpu_ids) // 2)
    _king_gpu_ids = _gpu_ids[:mid]
    _chall_gpu_ids = _gpu_ids[mid:] or _gpu_ids[:mid]

    log.info(
        "eval_server_vllm starting: GPUs=%s king=%s challenger=%s seq_len=%d",
        _gpu_ids, _king_gpu_ids, _chall_gpu_ids, DEFAULT_SEQ_LEN,
    )
    _r2 = R2()

    yield

    log.info("eval_server_vllm shutting down")
    if _king is not None:
        try:
            _king.shutdown()
        except Exception:
            log.warning("king shutdown failed", exc_info=True)


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ProbeRequest(BaseModel):
    repo: str
    revision: str = ""


class EvalRequest(BaseModel):
    king_repo: str
    challenger_repo: str
    block_hash: str
    hotkey: str
    shard_key: str
    king_hash: str = ""
    king_revision: str = ""
    challenger_revision: str = ""
    eval_n: int = DEFAULT_EVAL_N
    alpha: float = DEFAULT_ALPHA
    seq_len: int = DEFAULT_SEQ_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------

def _ensure_king(
    repo: str,
    king_hash: str,
    revision: str,
    seq_len: int,
    on_phase=None,
) -> VllmEvaluator:
    """Reuse the cached king if (repo, revision, hash) match; otherwise reload.

    The vLLM engine is pinned to GPUs `_king_gpu_ids` and lives in its own
    subprocess. Reload tears the subprocess down (releases all GPU memory)
    before booting a fresh one — vLLM does not currently support in-process
    weight swap for arbitrary archs.
    """
    global _king, _king_repo, _king_revision, _king_hash
    if (
        _king is not None
        and _king_repo == repo
        and (not revision or _king_revision == revision)
        and (not king_hash or _king_hash == king_hash)
        and _king.seq_len == seq_len
    ):
        log.info("reusing cached king for %s rev=%s",
                 repo, (_king_revision or "?")[:12])
        return _king

    if _king is not None:
        log.info(
            "king changed (%s rev=%s -> %s rev=%s), reloading",
            _king_repo, (_king_revision or "?")[:12],
            repo, revision[:12] if revision else "?",
        )
        try:
            _king.shutdown()
        except Exception:
            log.warning("old king shutdown failed", exc_info=True)
        _king = None

    new_king = VllmEvaluator(
        repo=repo,
        gpu_ids=_king_gpu_ids,
        seq_len=seq_len,
        label="king",
        revision=revision or None,
        on_phase=on_phase,
    )

    _king = new_king
    _king_repo = repo
    _king_revision = revision or None
    _king_hash = king_hash or None
    return _king


def _load_challenger(
    repo: str,
    revision: str,
    seq_len: int,
    on_phase=None,
) -> VllmEvaluator:
    return VllmEvaluator(
        repo=repo,
        gpu_ids=_chall_gpu_ids,
        seq_len=seq_len,
        label="challenger",
        revision=revision or None,
        on_phase=on_phase,
    )


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

def _cleanup_hf_cache():
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        cache_gb = cache_info.size_on_disk / 1e9
        if cache_gb < CACHE_HIGH_WATERMARK_GB:
            return

        all_revs = []
        for repo_info in cache_info.repos:
            for rev_info in repo_info.revisions:
                all_revs.append(
                    (rev_info.last_modified, repo_info.repo_id, rev_info),
                )
        all_revs.sort()

        running_total = cache_info.size_on_disk
        hashes_to_delete = []
        for _last, repo_id, rev_info in all_revs:
            if running_total / 1e9 < CACHE_HIGH_WATERMARK_GB * 0.7:
                break
            if (
                repo_id == _king_repo
                and (not _king_revision or rev_info.commit_hash == _king_revision)
            ):
                continue
            hashes_to_delete.append(rev_info.commit_hash)
            running_total -= rev_info.size_on_disk
            log.info("marking for deletion: %s rev %s (%.1f MB)",
                     repo_id, (rev_info.commit_hash or "")[:12],
                     rev_info.size_on_disk / 1e6)

        if hashes_to_delete:
            strategy = cache_info.delete_revisions(*hashes_to_delete)
            log.info("hf cache cleanup: deleting %d revisions, freeing %.1f MB",
                     len(hashes_to_delete), strategy.expected_freed_size / 1e6)
            strategy.execute()
    except Exception:
        log.warning("hf cache cleanup failed", exc_info=True)


def _prune_evals():
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
    except Exception:
        log.warning("eval pruning failed", exc_info=True)


def _get_disk_stats():
    stats: dict = {}
    try:
        usage = shutil.disk_usage("/")
        stats["disk_total_gb"] = round(usage.total / 1e9, 1)
        stats["disk_used_gb"] = round(usage.used / 1e9, 1)
        stats["disk_free_gb"] = round(usage.free / 1e9, 1)
    except Exception:
        pass
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        stats["hf_cache_size_gb"] = round(cache_info.size_on_disk / 1e9, 2)
        stats["hf_cache_repos"] = len(cache_info.repos)
        stats["hf_cache_revisions"] = sum(len(r.revisions) for r in cache_info.repos)
    except Exception:
        pass
    return stats


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def _run_eval(eval_id: str, req: EvalRequest):
    record = _evals[eval_id]
    record["state"] = "running"
    event_q: Queue = record["events"]

    def _on_phase(info: dict):
        try:
            event_q.put({"type": "progress", "data": info})
        except Exception:
            log.warning("on_phase enqueue failed (non-fatal)", exc_info=True)

    _heartbeat_stop = threading.Event()

    def _heartbeat_loop():
        while not _heartbeat_stop.wait(30.0):
            try:
                event_q.put({"type": "progress", "data": {"phase": "heartbeat"}})
            except Exception:
                pass

    threading.Thread(
        target=_heartbeat_loop,
        name=f"hb-{eval_id[:8]}",
        daemon=True,
    ).start()

    challenger_eval: VllmEvaluator | None = None
    same_model = (
        req.king_repo == req.challenger_repo
        and req.king_revision == req.challenger_revision
    )

    try:
        if req.shard_key:
            try:
                prefetch_shard(_r2, req.shard_key)
            except Exception:
                log.warning("shard prefetch kickoff failed (non-fatal)", exc_info=True)

        king_eval = _ensure_king(
            req.king_repo, req.king_hash, req.king_revision,
            req.seq_len, on_phase=_on_phase,
        )

        if same_model:
            challenger_eval = king_eval
        else:
            challenger_eval = _load_challenger(
                req.challenger_repo, req.challenger_revision,
                req.seq_len, on_phase=_on_phase,
            )

        if PROBE_ENABLED:
            log.info("PROBE_ENABLED=1 but vLLM probe path not implemented yet "
                     "(skipping probe; rely on miner pre-submission checks)")

        seed_str = f"{req.block_hash}:{req.hotkey}"

        def _on_progress(info: dict):
            record["progress"] = info
            event_q.put({"type": "progress", "data": info})

        verdict = run_bootstrap_test_vllm(
            king_eval, challenger_eval,
            _r2, req.shard_key, req.eval_n, req.alpha,
            req.seq_len, req.batch_size, seed_str,
            n_bootstrap=req.n_bootstrap,
            on_progress=_on_progress,
        )

        verdict["timestamp"] = datetime.now(timezone.utc).isoformat()

        record["state"] = "completed"
        record["verdict"] = verdict
        event_q.put({"type": "verdict", "data": verdict})

    except Exception as exc:
        log.exception("eval %s failed", eval_id)
        record["state"] = "failed"
        record["error"] = str(exc)
        event_q.put({"type": "error", "data": {"error": str(exc)}})

    finally:
        _heartbeat_stop.set()
        if challenger_eval is not None and not same_model:
            try:
                challenger_eval.shutdown()
            except Exception:
                log.warning("challenger shutdown failed (non-fatal)", exc_info=True)
        try:
            _eval_lock.release()
        except RuntimeError:
            log.warning("eval %s: eval_lock not held at release time", eval_id)
        try:
            _cleanup_hf_cache()
        except Exception:
            log.warning("hf cleanup failed", exc_info=True)
        try:
            _prune_evals()
        except Exception:
            log.warning("prune failed", exc_info=True)


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

def _watchdog(eval_id: str, deadline: float):
    while time.time() < deadline:
        time.sleep(30)
        rec = _evals.get(eval_id)
        if rec is None or rec.get("state") in ("completed", "failed"):
            return
    rec = _evals.get(eval_id)
    if rec is None or rec.get("state") in ("completed", "failed"):
        return
    log.error("watchdog: eval %s exceeded %ds, force-failing",
              eval_id, EVAL_MAX_RUNTIME_S)
    rec["state"] = "failed"
    rec["error"] = f"watchdog timeout after {EVAL_MAX_RUNTIME_S}s"
    try:
        rec["events"].put({"type": "error", "data": {"error": rec["error"]}})
    except Exception:
        pass
    try:
        _eval_lock.release()
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": "vllm",
        "gpus": len(_gpu_ids),
        "gpu_ids": _gpu_ids,
        "king_gpu_ids": _king_gpu_ids,
        "challenger_gpu_ids": _chall_gpu_ids,
        "king_loaded": _king_repo,
        "king_revision": _king_revision,
        "active_evals": len(_evals),
        **_get_disk_stats(),
    }


@app.post("/probe")
async def probe_endpoint(req: ProbeRequest):
    """Stub probe endpoint for API parity with eval_server.py.

    The HF-based trainability probe in `eval_torch.trainability_probe` pokes
    at HF model internals (params, grads, lm_head) which a vLLM-loaded engine
    doesn't expose. v1 returns a synthetic ok=True with `not_implemented`
    so validators that probe only their own incumbent king don't error;
    operators who require a real probe should keep eval_server.py for that
    role and only point preference-eval traffic here.
    """
    if not req.repo:
        raise HTTPException(status_code=400, detail="repo is required")
    return {
        "ok": True,
        "reason": "vllm_probe_not_implemented",
        "repo": req.repo,
        "revision": req.revision,
        "backend": "vllm",
        "timing": {"load_s": 0.0, "probe_s": 0.0, "total_s": 0.0},
        "loss_before": 0.0,
        "loss_after": 0.0,
        "delta": 0.0,
        "max_ratio": 0.0,
        "max_grad_norm": 0.0,
        "min_loss_before": 0.0,
        "max_loss_after": 0.0,
        "n_seeds": 0,
        "n_steps_per_seed": 0,
        "norm_quantization": None,
        "warnings": ["probe not implemented for vllm backend"],
    }


@app.post("/eval")
async def start_eval(req: EvalRequest):
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

    threading.Thread(
        target=_run_eval, args=(eval_id, req), daemon=True,
        name=f"eval-{eval_id}",
    ).start()
    threading.Thread(
        target=_watchdog,
        args=(eval_id, time.time() + EVAL_MAX_RUNTIME_S),
        daemon=True,
        name=f"watchdog-{eval_id}",
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
                    final_type = (
                        "verdict" if record["state"] == "completed" else "error"
                    )
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
    port = int(os.environ.get("EVAL_PORT", "9001"))
    uvicorn.run(
        "eval_server_vllm:app", host=host, port=port, log_level="info",
    )
