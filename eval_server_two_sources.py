#!/usr/bin/env python3
""":"
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then exec .venv/bin/python "$0" "$@"; fi
exec uv run python "$0" "$@"
":"""
from __future__ import annotations
"""Quasar pair-eval server with multiple live .npy data sources.

This is a thin wrapper around eval_server_quasar_pair.py. All dataset and
manifest logic lives in npy_sources.py; this file contains only the HTTP
server, lifespan setup, and route handlers.

Example:
    uvicorn eval_server_two_sources:app --host 127.0.0.1 --port 9011
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from queue import Empty, Queue

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import eval_server_quasar_pair as base
import npy_sources  # noqa: F401 — registers monkey-patches on import
from npy_sources import (
    DEFAULT_MANIFEST_URLS,
    DEFAULT_SHARDS_PER_SOURCE,
    DEFAULT_SOURCE_WEIGHT_MAP,
    MultiSourceEvalRequest,
    URL_CACHE_DIR,
    _manifest_source_name,
)

log = logging.getLogger("eval_server_two_sources")


@asynccontextmanager
async def lifespan(app: FastAPI):
    base.setup_logging()
    base.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base.EVAL_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    base.SHARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    URL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base.ensure_model_decryption_key_permissions()
    base._gpu_ids = base.parse_gpu_ids()
    log.info(
        "Quasar two-source eval server starting; gpus=%s shard_cache=%s url_cache=%s",
        base._gpu_ids,
        base.SHARD_CACHE_DIR,
        URL_CACHE_DIR,
    )
    yield
    log.info("Quasar two-source eval server shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu_ids": base._gpu_ids,
        "king_loaded": base._king_key,
        "active_evals": len(base._evals),
        "cache_dir": str(base.MODEL_CACHE_DIR),
        "shard_cache_dir": str(base.SHARD_CACHE_DIR),
        "url_cache_dir": str(URL_CACHE_DIR),
        "default_sources": [
            {"name": _manifest_source_name(url), "kind": "manifest", "value": url}
            for url in DEFAULT_MANIFEST_URLS
        ],
        "source_weights": DEFAULT_SOURCE_WEIGHT_MAP,
        "defaults": {
            "batch_size": base.DEFAULT_BATCH_SIZE,
            "alpha": base.DEFAULT_ALPHA,
            "seq_len": base.DEFAULT_SEQ_LEN,
            "n": base.DEFAULT_N,
            "n_public": base.DEFAULT_N_PUBLIC,
            "n_private": base.DEFAULT_N_PRIVATE,
            "n_bootstrap": base.DEFAULT_BOOTSTRAP_B,
            "shards_per_source": DEFAULT_SHARDS_PER_SOURCE,
        },
        "caps": {
            "eval_n_cap": base.EVAL_N_CAP,
            "eval_bootstrap_b_cap": base.EVAL_BOOTSTRAP_B_CAP,
            "eval_max_runtime_s": base.EVAL_MAX_RUNTIME_S,
        },
        "early_stop": {
            "enabled": base.EVAL_EARLY_STOP,
            "min_fraction": base.EVAL_EARLY_STOP_MIN_FRACTION,
            "advantage_quantile": base.EVAL_EARLY_STOP_ADVANTAGE_QUANTILE,
        },
        "encryption": {
            "manifest_name": base.MODEL_ENCRYPTION_MANIFEST_NAME,
            "age_available": base.shutil.which("age") is not None,
            "private_key_available": base.model_decryption_key_available(),
        },
    }


@app.post("/eval")
async def start_eval(req: MultiSourceEvalRequest):
    if not base._eval_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="an eval is already running")
    eval_id = uuid.uuid4().hex[:8]
    base._evals[eval_id] = {
        "state": "pending",
        "progress": {},
        "verdict": None,
        "error": None,
        "request": req.model_dump(),
        "events": Queue(),
        "created_at": time.time(),
    }
    threading.Thread(target=base.run_eval, args=(eval_id, req), daemon=True, name=f"eval-{eval_id}").start()
    return {"eval_id": eval_id}


@app.get("/eval/{eval_id}")
async def get_eval(eval_id: str):
    if eval_id not in base._evals:
        raise HTTPException(status_code=404, detail="eval not found")
    rec = base._evals[eval_id]
    return {
        "eval_id": eval_id,
        "state": rec["state"],
        "progress": rec["progress"],
        "verdict": rec["verdict"],
        "error": rec["error"],
    }


@app.get("/eval/{eval_id}/stream")
async def stream_eval(eval_id: str):
    if eval_id not in base._evals:
        raise HTTPException(status_code=404, detail="eval not found")
    rec = base._evals[eval_id]
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

    base.setup_logging()
    host = os.environ.get("EVAL_HOST", "127.0.0.1")
    port = int(os.environ.get("EVAL_PORT", "9000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
