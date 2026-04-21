#!/usr/bin/env python3
"""Eval server — persistent FastAPI service wrapping eval_torch.py.

Runs on the GPU box. Caches the king model across evals, reloads only when
the repo changes. Streams progress via SSE.

Usage:
    uvicorn eval_server:app --host 127.0.0.1 --port 9000

Env vars: same as eval_torch.py (HF_TOKEN, TEUTONIC_R2_*)
    EVAL_HOST   Bind address (default: 127.0.0.1, set to 0.0.0.0 only behind a firewall)
"""
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
from queue import Queue, Empty

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from eval_torch import (
    R2, MultiGPUEvaluator, run_bootstrap_test, parse_gpu_ids,
    trainability_probe,
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
_king_revision: str | None = None
_eval_lock = threading.Lock()
_evals: dict[str, dict] = {}

DEFAULT_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "256"))
DEFAULT_EVAL_N = int(os.environ.get("EVAL_N", "10000"))
DEFAULT_ALPHA = float(os.environ.get("EVAL_ALPHA", "0.001"))
DEFAULT_SEQ_LEN = int(os.environ.get("EVAL_SEQ_LEN", "2048"))
DEFAULT_DELTA = float(os.environ.get("EVAL_DELTA", "0.01"))
DEFAULT_BOOTSTRAP_B = int(os.environ.get("EVAL_BOOTSTRAP_B", "10000"))

PROBE_ENABLED = os.environ.get("TEUTONIC_PROBE_ENABLED", "1") == "1"


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
    _cleanup_hf_cache()
    yield
    log.info("eval server shutting down")
    if _king_evaluator:
        _king_evaluator.shutdown()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

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
    delta: float = DEFAULT_DELTA
    seq_len: int = DEFAULT_SEQ_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def _ensure_king(repo: str, king_hash: str = "", revision: str = ""):
    """Load or reuse king evaluator. Reloads if repo, revision, or king_hash changed.

    On a fresh load, runs the trainability probe on the king. A king that fails
    the probe is a violation of an invariant (the king got there by winning an
    eval, which already required passing the probe), so we refuse to load it
    and raise — operator must intervene.
    """
    global _king_evaluator, _king_repo, _king_hash, _king_revision
    if (_king_evaluator and _king_repo == repo
            and (not revision or _king_revision == revision)
            and (not king_hash or _king_hash == king_hash)):
        log.info("reusing cached king evaluator for %s (rev=%s)",
                 repo, (_king_revision or "?")[:12])
        return _king_evaluator

    needs_reload = _king_evaluator is not None
    if needs_reload:
        log.info("king changed (%s rev=%s -> %s rev=%s), reloading",
                 _king_repo, (_king_revision or "?")[:12],
                 repo, revision[:12] if revision else "?")
        _king_evaluator.shutdown()
        _king_evaluator = None
        torch.cuda.empty_cache()

    mid = len(_gpu_ids) // 2
    king_gpus = _gpu_ids[:mid] or _gpu_ids[:1]
    new_king = MultiGPUEvaluator(repo, king_gpus, label="king",
                                  force_download=False,
                                  revision=revision or None)

    if PROBE_ENABLED:
        king_model = new_king.models[new_king.gpu_ids[0]]
        t0 = time.time()
        probe = trainability_probe(king_model)
        log.info("king trainability probe for %s: ok=%s before=%.4f "
                 "after=%.4f delta=%.4f (%.1fs)",
                 repo, probe["ok"],
                 probe["loss_before"], probe["loss_after"], probe["delta"],
                 time.time() - t0)
        if not probe["ok"]:
            log.error("KING TRAINABILITY PROBE FAILED for %s: %s. "
                      "Refusing to load this king. Operator intervention required.",
                      repo, probe["reason"])
            new_king.shutdown()
            del new_king
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"king {repo}@{(revision or '?')[:12]} failed trainability "
                f"probe: {probe['reason']}"
            )

    _king_evaluator = new_king
    _king_repo = repo
    _king_hash = king_hash or None
    _king_revision = revision or None
    return _king_evaluator


def _load_challenger(repo: str, revision: str = ""):
    """Load challenger on the second half of GPUs."""
    mid = len(_gpu_ids) // 2
    chall_gpus = _gpu_ids[mid:] or _gpu_ids[:1]
    return MultiGPUEvaluator(repo, chall_gpus, label="challenger",
                              revision=revision or None)


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

MAX_EVALS_KEPT = 50
EVAL_MAX_AGE_S = 3600

def _cleanup_hf_cache():
    """Delete HF cached models that aren't the current king."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()

        keep_repo = _king_repo
        keep_rev = _king_revision
        hashes_to_delete = []

        for repo_info in cache_info.repos:
            repo_id = repo_info.repo_id
            if repo_id == keep_repo:
                for rev_info in repo_info.revisions:
                    if keep_rev and rev_info.commit_hash == keep_rev:
                        continue
                    hashes_to_delete.append(rev_info.commit_hash)
                    log.info("marking for deletion: %s rev %s (%.1f MB)",
                             repo_id, rev_info.commit_hash[:12],
                             rev_info.size_on_disk / 1e6)
            else:
                for rev_info in repo_info.revisions:
                    hashes_to_delete.append(rev_info.commit_hash)
                    log.info("marking for deletion: %s rev %s (%.1f MB)",
                             repo_id, rev_info.commit_hash[:12],
                             rev_info.size_on_disk / 1e6)

        if not hashes_to_delete:
            log.info("hf cache cleanup: nothing to delete")
            return

        strategy = cache_info.delete_revisions(*hashes_to_delete)
        log.info("hf cache cleanup: deleting %d revisions, freeing %.1f MB",
                 len(hashes_to_delete), strategy.expected_freed_size / 1e6)
        strategy.execute()
        log.info("hf cache cleanup: done")

    except Exception:
        log.warning("hf cache cleanup failed", exc_info=True)


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


def _get_disk_stats():
    """Return disk usage stats for the health endpoint."""
    stats = {}
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
# Eval runner (runs in a thread)
# ---------------------------------------------------------------------------

def _run_eval(eval_id: str, req: EvalRequest):
    record = _evals[eval_id]
    record["state"] = "running"
    event_q: Queue = record["events"]

    try:
        king_eval = _ensure_king(req.king_repo, req.king_hash, req.king_revision)

        same_model = (req.king_repo == req.challenger_repo
                      and req.king_revision == req.challenger_revision)
        if same_model:
            challenger_eval = king_eval
        else:
            challenger_eval = _load_challenger(req.challenger_repo, req.challenger_revision)

        if not same_model and PROBE_ENABLED:
            chall_model = challenger_eval.models[challenger_eval.gpu_ids[0]]
            t0 = time.time()
            probe = trainability_probe(chall_model)
            log.info("trainability probe for %s: ok=%s before=%.4f after=%.4f "
                     "delta=%.4f (%.1fs)",
                     req.challenger_repo, probe["ok"],
                     probe["loss_before"], probe["loss_after"], probe["delta"],
                     time.time() - t0)
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
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                record["state"] = "completed"
                record["verdict"] = verdict
                event_q.put({"type": "verdict", "data": verdict})
                return

        seed_str = f"{req.block_hash}:{req.hotkey}"

        def _on_progress(info):
            record["progress"] = info
            event_q.put({"type": "progress", "data": info})

        verdict = run_bootstrap_test(
            king_eval, challenger_eval,
            _r2, req.shard_key, req.eval_n, req.alpha, req.delta,
            req.seq_len, req.batch_size, seed_str,
            n_bootstrap=req.n_bootstrap,
            on_progress=_on_progress,
        )

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

    finally:
        _cleanup_hf_cache()
        _prune_evals()
        _eval_lock.release()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpus": len(_gpu_ids),
        "gpu_ids": _gpu_ids,
        "king_loaded": _king_repo,
        "active_evals": len(_evals),
        **_get_disk_stats(),
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

    thread = threading.Thread(target=_run_eval, args=(eval_id, req), daemon=True)
    thread.start()

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
