#!/usr/bin/env python3
"""Eval server — persistent FastAPI service wrapping eval_torch.py.

Runs on the GPU box. Caches the king model across evals, reloads only when
the repo changes. Streams progress via SSE.

Usage:
    uvicorn eval_server:app --host 0.0.0.0 --port 9000

Env vars: same as eval_torch.py (HF_TOKEN, TEUTONIC_R2_*)
"""
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from queue import Queue, Empty

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from eval_torch import (
    R2, MultiGPUEvaluator, run_bootstrap_test, parse_gpu_ids,
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
    """Load or reuse king evaluator. Reloads if repo, revision, or king_hash changed."""
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
    _king_evaluator = MultiGPUEvaluator(repo, king_gpus, label="king",
                                         force_download=needs_reload,
                                         revision=revision or None)
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
    port = int(os.environ.get("EVAL_PORT", "9000"))
    uvicorn.run("eval_server:app", host="0.0.0.0", port=port, log_level="info")
