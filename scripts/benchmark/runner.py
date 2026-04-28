#!/usr/bin/env python3
"""Continuous lm-eval-harness benchmark runner for the Teutonic king.

Watches the public dashboard.json for the current king, runs the seven
zero-shot benchmarks from Covenant-72B Table 1 (ARC-Challenge, ARC-Easy,
PIQA, OpenBookQA, HellaSwag, WinoGrande, MMLU) on every newly seen
(repo, revision) pair, and uploads the rolling time series to the same
Hippius bucket the validator writes to.

Outputs (Hippius `teutonic-sn3` bucket):
- `benchmarks.json`         rolling { updated_at, latest, history[<=N], reference }
- `benchmarks.history.jsonl` append-only log of every completed eval

Reads:
- `https://s3.hippius.com/teutonic-sn3/dashboard.json` (3 mirrors raced)

This process is independent of the validator/eval_server and only WRITES
to two new keys; it never touches king state or `dashboard.json`.

Env (all optional except the Hippius creds):

    TEUTONIC_HIPPIUS_ACCESS_KEY  (required, write)
    TEUTONIC_HIPPIUS_SECRET_KEY  (required, write)
    TEUTONIC_HIPPIUS_ENDPOINT    default https://s3.hippius.com
    TEUTONIC_HIPPIUS_BUCKET      default teutonic-sn3
    TEUTONIC_BENCH_TASKS         default arc_challenge,arc_easy,piqa,openbookqa,hellaswag,winogrande,mmlu
    TEUTONIC_BENCH_NUM_FEWSHOT   default 0
    TEUTONIC_BENCH_BATCH_SIZE    default auto
    TEUTONIC_BENCH_LIMIT         default unset (run full task)
    TEUTONIC_BENCH_HISTORY_MAX   default 200 (rows kept inline in benchmarks.json)
    TEUTONIC_BENCH_POLL_SECS     default 60
    TEUTONIC_BENCH_FAIL_COOLDOWN default 3600 (seconds before retrying a failed king)
    TEUTONIC_BENCH_STATE_DIR     default ~/.cache/teutonic-bench
    TEUTONIC_BENCH_RUN_DIR       default ~/.cache/teutonic-bench/runs
    HF_TOKEN                     optional, forwarded to lm_eval
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import boto3
import httpx
from botocore.config import Config as BotoConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bench")


HIPPIUS_ENDPOINT = os.environ.get("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com")
HIPPIUS_BUCKET = os.environ.get("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3")
HIPPIUS_ACCESS_KEY = os.environ.get("TEUTONIC_HIPPIUS_ACCESS_KEY", "")
HIPPIUS_SECRET_KEY = os.environ.get("TEUTONIC_HIPPIUS_SECRET_KEY", "")

DASHBOARD_MIRRORS = [
    "https://us-east-1.hippius.com/teutonic-sn3/dashboard.json",
    "https://eu-central-1.hippius.com/teutonic-sn3/dashboard.json",
    "https://s3.hippius.com/teutonic-sn3/dashboard.json",
]

BENCH_KEY = "benchmarks.json"
BENCH_HISTORY_KEY = "benchmarks.history.jsonl"

TASKS = [
    t.strip()
    for t in os.environ.get(
        "TEUTONIC_BENCH_TASKS",
        "arc_challenge,arc_easy,piqa,openbookqa,hellaswag,winogrande,mmlu",
    ).split(",")
    if t.strip()
]
NUM_FEWSHOT = int(os.environ.get("TEUTONIC_BENCH_NUM_FEWSHOT", "0"))
BATCH_SIZE = os.environ.get("TEUTONIC_BENCH_BATCH_SIZE", "auto")
LIMIT = os.environ.get("TEUTONIC_BENCH_LIMIT", "").strip() or None
HISTORY_MAX = int(os.environ.get("TEUTONIC_BENCH_HISTORY_MAX", "200"))
POLL_SECS = int(os.environ.get("TEUTONIC_BENCH_POLL_SECS", "60"))
FAIL_COOLDOWN = int(os.environ.get("TEUTONIC_BENCH_FAIL_COOLDOWN", "3600"))

STATE_DIR = Path(
    os.environ.get("TEUTONIC_BENCH_STATE_DIR", str(Path.home() / ".cache" / "teutonic-bench"))
)
RUN_DIR = Path(
    os.environ.get("TEUTONIC_BENCH_RUN_DIR", str(STATE_DIR / "runs"))
)
STATE_FILE = STATE_DIR / "state.json"

# Reference points from the Covenant-72B paper (Table 1, 0-shot accuracy).
# Surfaced on the dashboard as faint guide rows; never used for control flow.
REFERENCE = [
    {
        "name": "INTELLECT-1",
        "size": "10B",
        "tokens": "1T",
        "env": "Internet",
        "permissionless": False,
        "scores": {
            "arc_challenge": 0.448,
            "arc_easy": 0.718,
            "piqa": 0.774,
            "openbookqa": 0.438,
            "hellaswag": 0.703,
            "winogrande": 0.633,
            "mmlu": 0.327,
        },
    },
    {
        "name": "Psyche Consilience",
        "size": "40B",
        "tokens": "1.2T",
        "env": "Internet",
        "permissionless": False,
        "scores": {
            "arc_challenge": 0.311,
            "arc_easy": 0.558,
            "piqa": 0.761,
            "openbookqa": 0.352,
            "hellaswag": 0.637,
            "winogrande": 0.570,
            "mmlu": 0.242,
        },
    },
    {
        "name": "Covenant-72B",
        "size": "72B",
        "tokens": "1.1T",
        "env": "Internet",
        "permissionless": True,
        "scores": {
            "arc_challenge": 0.568,
            "arc_easy": 0.809,
            "piqa": 0.816,
            "openbookqa": 0.440,
            "hellaswag": 0.806,
            "winogrande": 0.759,
            "mmlu": 0.671,
        },
    },
    {
        "name": "LLM360 K2",
        "size": "65B",
        "tokens": "1.4T",
        "env": "Centralized",
        "permissionless": False,
        "scores": {
            "arc_challenge": 0.538,
            "arc_easy": 0.760,
            "piqa": 0.825,
            "openbookqa": 0.480,
            "hellaswag": 0.829,
            "winogrande": 0.764,
            "mmlu": 0.655,
        },
    },
    {
        "name": "LLaMA-2-7B",
        "size": "7B",
        "tokens": "2T",
        "env": "Centralized",
        "permissionless": False,
        "scores": {
            "arc_challenge": 0.451,
            "arc_easy": 0.738,
            "piqa": 0.787,
            "openbookqa": 0.442,
            "hellaswag": 0.762,
            "winogrande": 0.694,
            "mmlu": 0.417,
        },
    },
    {
        "name": "LLaMA-2-70B",
        "size": "70B",
        "tokens": "2T",
        "env": "Centralized",
        "permissionless": False,
        "scores": {
            "arc_challenge": 0.574,
            "arc_easy": 0.796,
            "piqa": 0.826,
            "openbookqa": 0.494,
            "hellaswag": 0.843,
            "winogrande": 0.804,
            "mmlu": 0.656,
        },
    },
]

# Per-task primary metric. Matches the conventions used in the Covenant paper:
# acc_norm for the multiple-choice tasks scored by length-normalized loglikelihood,
# acc for WinoGrande and MMLU which are single-token / well-balanced.
TASK_METRIC = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "piqa": "acc_norm",
    "openbookqa": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "mmlu": "acc",
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Hippius S3 (write side mirrors validator.py R2 helper).
# ---------------------------------------------------------------------------


def make_hippius_client():
    if not (HIPPIUS_ACCESS_KEY and HIPPIUS_SECRET_KEY):
        raise RuntimeError(
            "TEUTONIC_HIPPIUS_ACCESS_KEY / TEUTONIC_HIPPIUS_SECRET_KEY must be set"
        )
    return boto3.client(
        "s3",
        endpoint_url=HIPPIUS_ENDPOINT,
        aws_access_key_id=HIPPIUS_ACCESS_KEY,
        aws_secret_access_key=HIPPIUS_SECRET_KEY,
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=15,
            read_timeout=120,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def hippius_put_json(client, key: str, obj) -> None:
    body = json.dumps(obj, default=str, separators=(",", ":")).encode()
    client.put_object(
        Bucket=HIPPIUS_BUCKET, Key=key, Body=body, ContentType="application/json"
    )


def hippius_get_bytes(client, key: str) -> bytes | None:
    try:
        return client.get_object(Bucket=HIPPIUS_BUCKET, Key=key)["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        if "NoSuchKey" in str(e) or "404" in str(e):
            return None
        raise


def hippius_append_jsonl(client, key: str, record: dict) -> None:
    line = json.dumps(record, default=str) + "\n"
    existing = hippius_get_bytes(client, key) or b""
    client.put_object(
        Bucket=HIPPIUS_BUCKET,
        Key=key,
        Body=existing + line.encode(),
        ContentType="application/x-ndjson",
    )


# ---------------------------------------------------------------------------
# Local state.
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            log.warning("state file %s is corrupt, resetting", STATE_FILE)
    return {"seen": [], "failures": {}}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_FILE)


def seed_seen_from_history(client, state: dict) -> None:
    """Populate `seen` with (repo, revision) pairs already in the upstream
    benchmarks.history.jsonl so a fresh runner doesn't redo work."""
    if state.get("seeded_from_history"):
        return
    raw = hippius_get_bytes(client, BENCH_HISTORY_KEY)
    if raw:
        for line in raw.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            repo = row.get("hf_repo")
            rev = row.get("revision")
            if repo and rev:
                key = f"{repo}@{rev}"
                if key not in state["seen"]:
                    state["seen"].append(key)
        log.info("seeded %d (repo,rev) pairs from existing history", len(state["seen"]))
    state["seeded_from_history"] = True
    save_state(state)


# ---------------------------------------------------------------------------
# Dashboard polling.
# ---------------------------------------------------------------------------


def fetch_current_king() -> dict | None:
    """Return the king dict from whichever Hippius mirror responds first."""
    last_err = None
    for url in DASHBOARD_MIRRORS:
        try:
            r = httpx.get(url, timeout=15.0)
            r.raise_for_status()
            d = r.json()
            king = d.get("king") or {}
            if king.get("hf_repo") and king.get("king_revision"):
                return king
        except Exception as e:
            last_err = e
            continue
    if last_err:
        log.warning("all dashboard mirrors failed: %s", last_err)
    return None


# ---------------------------------------------------------------------------
# lm-eval-harness invocation.
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run_lm_eval(repo: str, revision: str, run_dir: Path) -> dict:
    """Run lm-eval-harness, return its parsed results.json contents."""
    if not cuda_available():
        # CPU-only inference on a 3B model would take days per task.
        # Treat this as a transient skip rather than a failure so the
        # runner keeps watching for the GPU to be (re)attached.
        raise RuntimeError("no CUDA device available; skipping eval until GPU is attached")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={repo},"
        f"revision={revision},"
        f"dtype=bfloat16,"
        f"trust_remote_code=False"
    )
    cmd = [
        sys.executable, "-m", "lm_eval",
        "run",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(TASKS),
        "--num_fewshot", str(NUM_FEWSHOT),
        "--batch_size", BATCH_SIZE,
        "--output_path", str(run_dir),
    ]
    if LIMIT:
        cmd += ["--limit", LIMIT]
    log.info("running: %s", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"lm_eval exited with code {proc.returncode}")
    # lm-eval writes results to <output_path>/<sanitised_model>/results_*.json
    candidates = sorted(run_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        # fallback: older harness versions wrote results.json
        candidates = sorted(run_dir.rglob("results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"no results json produced in {run_dir}")
    return json.loads(candidates[0].read_text())


def extract_scores(results_json: dict) -> dict:
    """Pull the per-task primary metric out of an lm-eval results dict.

    Handles both modern ('acc,none') and legacy ('acc') key shapes.
    """
    out = {}
    results = results_json.get("results", {})
    for task in TASKS:
        if task not in results:
            log.warning("task %s missing from results", task)
            continue
        metric = TASK_METRIC.get(task, "acc")
        node = results[task]
        for k in (f"{metric},none", metric, f"{metric},flexible-extract"):
            if k in node and isinstance(node[k], (int, float)):
                out[task] = float(node[k])
                break
        else:
            # MMLU aggregate is sometimes only available via children; fall back to acc.
            for k, v in node.items():
                if k.startswith("acc,") and isinstance(v, (int, float)):
                    out[task] = float(v)
                    break
    return out


# ---------------------------------------------------------------------------
# Result aggregation / upload.
# ---------------------------------------------------------------------------


def build_row(king: dict, scores: dict, wall_time_s: float, harness_version: str) -> dict:
    return {
        "ts": utcnow_iso(),
        "reign_number": king.get("reign_number"),
        "hotkey": king.get("hotkey"),
        "hf_repo": king.get("hf_repo"),
        "revision": king.get("king_revision"),
        "crowned_block": king.get("crowned_block"),
        "crowned_at": king.get("crowned_at"),
        "scores": scores,
        "wall_time_s": round(wall_time_s, 2),
        "harness_version": harness_version,
        "tasks": TASKS,
        "num_fewshot": NUM_FEWSHOT,
        "limit": LIMIT,
    }


def merge_rolling_doc(client, row: dict) -> dict:
    """Read existing benchmarks.json (if any), prepend this row, trim to HISTORY_MAX."""
    existing = hippius_get_bytes(client, BENCH_KEY)
    history = []
    if existing:
        try:
            doc = json.loads(existing)
            history = list(doc.get("history") or [])
        except Exception:
            log.warning("existing benchmarks.json is corrupt, starting fresh")
    history.insert(0, row)
    if len(history) > HISTORY_MAX:
        history = history[:HISTORY_MAX]
    doc = {
        "updated_at": utcnow_iso(),
        "tasks": TASKS,
        "task_metric": TASK_METRIC,
        "num_fewshot": NUM_FEWSHOT,
        "limit": LIMIT,
        "latest": row,
        "history": history,
        "reference": REFERENCE,
    }
    return doc


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------


_should_exit = False


def _handle_signal(signum, frame):  # noqa: ANN001
    global _should_exit
    log.info("received signal %s, will exit after current eval", signum)
    _should_exit = True


def get_harness_version() -> str:
    try:
        import lm_eval  # type: ignore
        return getattr(lm_eval, "__version__", "unknown")
    except Exception:
        return "unknown"


def process_king(client, state: dict, king: dict) -> None:
    repo = king["hf_repo"]
    revision = king["king_revision"]
    seen_key = f"{repo}@{revision}"

    if seen_key in state.get("seen", []):
        return

    failures = state.setdefault("failures", {})
    last_failure = failures.get(seen_key, 0)
    if last_failure and (time.time() - last_failure) < FAIL_COOLDOWN:
        return

    if not cuda_available():
        # Don't even count this as a failure: GPU may attach any minute and
        # we want to retry on the next poll tick, not after a 1h cooldown.
        log.info("waiting for CUDA device before starting eval for %s @ %s",
                 repo, revision[:12])
        return

    log.info("new king: %s @ %s (reign #%s)",
             repo, revision[:12], king.get("reign_number"))

    run_dir = RUN_DIR / f"{repo.replace('/', '_')}_{revision[:12]}_{int(time.time())}"
    t0 = time.time()
    try:
        results_json = run_lm_eval(repo, revision, run_dir)
        scores = extract_scores(results_json)
        if not scores:
            raise RuntimeError("lm_eval produced no parseable scores")
        wall = time.time() - t0
        row = build_row(king, scores, wall, get_harness_version())
        log.info("scores: %s", scores)

        doc = merge_rolling_doc(client, row)
        hippius_put_json(client, BENCH_KEY, doc)
        hippius_append_jsonl(client, BENCH_HISTORY_KEY, row)
        log.info("uploaded benchmarks.json and appended history (%.0fs total)", wall)

        state.setdefault("seen", []).append(seen_key)
        # Keep the seen list bounded.
        if len(state["seen"]) > 5000:
            state["seen"] = state["seen"][-5000:]
        failures.pop(seen_key, None)
        save_state(state)
    except Exception as e:
        log.error("eval failed for %s @ %s: %s", repo, revision[:12], e)
        log.error("traceback:\n%s", traceback.format_exc())
        failures[seen_key] = time.time()
        save_state(state)
    finally:
        # Free disk; lm-eval leaves an entire run dir per attempt.
        try:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log.info("teutonic benchmark runner starting")
    log.info("tasks=%s num_fewshot=%d batch_size=%s limit=%s",
             TASKS, NUM_FEWSHOT, BATCH_SIZE, LIMIT)
    log.info("hippius=%s/%s state=%s",
             HIPPIUS_ENDPOINT, HIPPIUS_BUCKET, STATE_DIR)

    client = make_hippius_client()

    state = load_state()
    try:
        seed_seen_from_history(client, state)
    except Exception as e:
        log.warning("could not seed seen from history: %s", e)

    while not _should_exit:
        try:
            king = fetch_current_king()
            if king:
                process_king(client, state, king)
            else:
                log.info("no king available yet")
        except Exception as e:
            log.error("loop tick failed: %s", e)
            log.error("traceback:\n%s", traceback.format_exc())

        # Sleep in 1s slices so SIGTERM exits promptly.
        for _ in range(POLL_SECS):
            if _should_exit:
                break
            time.sleep(1)

    log.info("teutonic benchmark runner exiting cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
