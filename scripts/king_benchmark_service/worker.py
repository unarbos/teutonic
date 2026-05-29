#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import select
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from common import DEFAULT_BENCHMARKS, http_json, read_json, require_bearer, utcnow_iso, write_json


class WorkerState:
    def __init__(self, root: Path, token: str, controller_url: str, callback_token: str) -> None:
        self.root = root
        self.token = token
        self.controller_url = controller_url.rstrip("/")
        self.callback_token = callback_token
        self.lock = threading.Lock()
        self.current: dict[str, Any] | None = None
        self.thread: threading.Thread | None = None
        self.state_path = root / "worker_state.json"
        root.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            payload = dict(self.current or {"status": "idle", "updated_at": utcnow_iso()})
        payload["gpu_snapshot"] = gpu_snapshot()
        return payload

    def set_current(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.current = payload
            write_json(self.state_path, payload)

    def post_event(self, event: dict[str, Any]) -> None:
        if not self.controller_url:
            return
        try:
            http_json(f"{self.controller_url}/worker-events", method="POST", payload=event, token=self.callback_token, timeout_s=20)
        except Exception as exc:
            row = {"at": utcnow_iso(), "event": event.get("event"), "error": repr(exc)}
            append_line(self.root / "callback_errors.jsonl", row)
            print(json.dumps({"event": "callback_error", **row}, sort_keys=False), flush=True)

    def post_result(self, result: dict[str, Any]) -> None:
        if not self.controller_url:
            return
        try:
            http_json(f"{self.controller_url}/worker-results", method="POST", payload=result, token=self.callback_token, timeout_s=120)
        except Exception as exc:
            row = {"at": utcnow_iso(), "event": "result", "error": repr(exc)}
            append_line(self.root / "callback_errors.jsonl", row)
            print(json.dumps({"event": "callback_error", **row}, sort_keys=False), flush=True)


def append_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=False) + "\n")


def log_event(job_root: Path, event: str, payload: dict[str, Any]) -> None:
    row = {"at": utcnow_iso(), "event": event, **payload}
    append_line(job_root / "worker_events.jsonl", row)
    print(json.dumps(row, sort_keys=False), flush=True)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    out = [part.strip() for part in value.split(",") if part.strip()]
    return out or default


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-") or "unknown"


def gpu_snapshot() -> dict[str, Any]:
    query = "index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
        if proc.returncode != 0:
            return {"error": proc.stderr.strip(), "gpus": []}
        gpus = []
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mib": parts[2],
                    "memory_total_mib": parts[3],
                    "utilization_gpu_pct": parts[4],
                    "power_draw_w": parts[5],
                    "temperature_c": parts[6],
                })
        pmon = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
        return {"gpus": gpus, "pmon": pmon.stdout.splitlines()[-16:] if pmon.stdout else []}
    except Exception as exc:
        return {"error": repr(exc), "gpus": []}


def compact_gpu_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for gpu in snapshot.get("gpus") or []:
        out.append({
            "i": gpu.get("index"),
            "mem_mib": gpu.get("memory_used_mib"),
            "util_pct": gpu.get("utilization_gpu_pct"),
            "power_w": gpu.get("power_draw_w"),
            "temp_c": gpu.get("temperature_c"),
        })
    return out


def pm2_event(event: str, payload: dict[str, Any]) -> None:
    print(json.dumps({"at": utcnow_iso(), "event": event, **payload}, sort_keys=False), flush=True)


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def eval_python(job: dict[str, Any]) -> str:
    return job.get("python") or os.environ.get("TEUTONIC_KING_BENCH_EVAL_PYTHON") or sys.executable


def cache_args(job: dict[str, Any]) -> list[str]:
    args: list[str] = []
    cache_requests = job.get("cache_requests")
    if cache_requests is None:
        cache_requests = os.environ.get("TEUTONIC_KING_BENCH_CACHE_REQUESTS", "true")
    if str(cache_requests).strip():
        args += ["--cache-requests", str(cache_requests).strip()]
    response_cache_root = job.get("response_cache_root") or os.environ.get("TEUTONIC_KING_BENCH_RESPONSE_CACHE_ROOT", "")
    if str(response_cache_root).strip():
        args += ["--response-cache-root", str(response_cache_root).strip()]
    return args


def optional_eval_args(job: dict[str, Any]) -> list[str]:
    args: list[str] = []
    if job.get("fewshot_overrides"):
        args += ["--fewshot-overrides", job["fewshot_overrides"]]
    if job.get("gen_kwargs"):
        args += ["--gen-kwargs", job["gen_kwargs"]]
    if job.get("limit") is not None:
        args += ["--limit", str(job["limit"])]
    if job.get("apply_chat_template"):
        args.append("--apply-chat-template")
    if job.get("log_samples"):
        args.append("--log-samples")
    return args


def build_eval_cmd(
    *,
    job: dict[str, Any],
    eval_script: Path,
    model_input: str,
    run_dir: Path,
    benchmarks: list[str],
    standardized_results_path: Path,
    model_args_extra: str | None,
    device: str | None,
    download_only: bool = False,
    snapshot_dir: Path | None = None,
) -> list[str]:
    cmd = [
        eval_python(job),
        str(eval_script),
        "--model-repo",
        model_input,
        "--run-dir",
        str(run_dir),
        "--benchmarks",
        ",".join(benchmarks),
        "--batch-size",
        str(os.environ.get("TEUTONIC_KING_BENCH_BATCH_SIZE") or job.get("batch_size", "auto")),
        "--dtype",
        str(job.get("dtype", "bfloat16")),
        "--standardized-results-path",
        str(standardized_results_path),
        "--resume",
    ]
    lm_eval_bin = job.get("lm_eval_bin") or os.environ.get("TEUTONIC_KING_BENCH_LM_EVAL_BIN", "")
    if str(lm_eval_bin).strip():
        cmd += ["--lm-eval-bin", str(lm_eval_bin).strip()]
    max_batch_size = job.get("max_batch_size") or os.environ.get("TEUTONIC_KING_BENCH_MAX_BATCH_SIZE", "")
    if str(max_batch_size).strip():
        cmd += ["--max-batch-size", str(max_batch_size).strip()]
    if snapshot_dir is not None:
        cmd += ["--snapshot-dir", str(snapshot_dir)]
    if model_args_extra and model_args_extra.strip():
        cmd += ["--model-args-extra", model_args_extra.strip()]
    if device and device.strip() and device.lower() not in {"auto", "none"}:
        cmd += ["--device", device.strip()]
    if download_only:
        cmd.append("--download-only")
    cmd += cache_args(job)
    cmd += optional_eval_args(job)
    return cmd


def read_child_benchmark(bench: str, child_std: Path, returncode: int, wall_time_s: float, gpu_id: str, log_path: Path) -> dict[str, Any]:
    payload = read_json(child_std, {}) or {}
    rows = payload.get("benchmarks") if isinstance(payload, dict) else None
    if isinstance(rows, list) and rows:
        row = dict(rows[0])
    else:
        row = {"name": bench, "task": None, "fewshot": None, "status": "failed", "metric": {"name": None, "value": None}}
    row.setdefault("name", bench)
    row["status"] = row.get("status") or ("completed" if returncode == 0 else "failed")
    row["returncode"] = returncode
    row["wall_time_s"] = round(wall_time_s, 2)
    row["worker_gpu_id"] = gpu_id
    row["worker_log"] = str(log_path)
    return row


def write_combined_results(
    *,
    path: Path,
    job_id: str,
    model: dict[str, Any],
    model_payload: dict[str, Any] | None,
    run_dir: Path,
    started_at: str,
    finished_at: str | None,
    benchmarks: list[str],
    completed_rows: dict[str, dict[str, Any]],
    active: dict[str, dict[str, Any]],
    pending: list[str],
    mode: str,
    gpu_ids: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for bench in benchmarks:
        if bench in completed_rows:
            rows.append(completed_rows[bench])
        elif bench in active:
            rows.append({"name": bench, "status": "running", "worker_gpu_id": active[bench]["gpu_id"], "worker_log": str(active[bench]["log_path"]), "pid": active[bench].get("pid")})
        elif bench in pending:
            rows.append({"name": bench, "status": "pending"})
    totals = {
        "requested": len(benchmarks),
        "completed": sum(1 for row in rows if row.get("status") == "completed"),
        "completed_no_metric": sum(1 for row in rows if row.get("status") == "completed_no_metric"),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
        "running": sum(1 for row in rows if row.get("status") == "running"),
        "pending": sum(1 for row in rows if row.get("status") == "pending"),
    }
    payload = {
        "schema_version": "king-benchmark-results.v1",
        "generated_at": utcnow_iso(),
        "run_id": job_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "model": model_payload or model,
        "artifacts": {
            "run_dir": str(run_dir),
            "standardized_results_json": str(path),
            "worker_events_jsonl": str(run_dir.parent / "worker_events.jsonl"),
            "gpu_snapshots_jsonl": str(run_dir.parent / "gpu_snapshots.jsonl"),
        },
        "worker": {"mode": mode, "gpu_ids": gpu_ids},
        "totals": totals,
        "benchmarks": rows,
    }
    write_json(path, payload)
    write_json(run_dir / "summary.json", payload)
    return payload



def single_gpu_lm_eval_bin(state: WorkerState) -> str:
    return os.environ.get("TEUTONIC_KING_BENCH_SINGLE_GPU_LM_EVAL_BIN", str(state.root / ".venv" / "bin" / "lm-eval"))


def accelerate_lm_eval_bin(state: WorkerState, gpu_ids: list[str]) -> str:
    configured = os.environ.get("TEUTONIC_KING_BENCH_LM_EVAL_BIN", "").strip()
    if configured:
        return configured
    accelerate = state.root / ".venv" / "bin" / "accelerate"
    gpu_arg = ",".join(gpu_ids)
    return f"{accelerate} launch --multi_gpu --num_processes {len(gpu_ids)} --gpu_ids {gpu_arg} --mixed_precision bf16 --num_cpu_threads_per_process 8 --enable_cpu_affinity --main_process_port 29600 -m lm_eval"


def run_hybrid_job(state: WorkerState, job: dict[str, Any]) -> dict[str, Any]:
    job_id = job["job_id"]
    model = job["model"]
    benchmarks = list(job.get("benchmarks") or DEFAULT_BENCHMARKS)
    mmlu_pro_names = {"MMLU-Pro", "mmlu_pro"}
    phase1 = [bench for bench in benchmarks if bench not in mmlu_pro_names]
    has_mmlu_pro = any(bench in mmlu_pro_names for bench in benchmarks)
    phase2 = ["MMLU-Pro"] if has_mmlu_pro else []

    job_root = state.root / "jobs" / job_id
    run_dir = job_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_script = Path(job.get("eval_script") or state.root / "eval_king_benchmarks.py")
    started_at = utcnow_iso()
    progress_interval = int(job.get("progress_interval_s") or os.environ.get("TEUTONIC_WORKER_PROGRESS_INTERVAL_S", "60"))
    gpu_ids = split_csv(os.environ.get("TEUTONIC_KING_BENCH_GPU_IDS"), [str(i) for i in range(8)])
    gpu_ids = gpu_ids[: max(1, int(os.environ.get("TEUTONIC_KING_BENCH_MAX_PARALLEL_BENCHMARKS", str(len(gpu_ids)))))]
    model_cache_dir = Path(os.environ.get("TEUTONIC_KING_BENCH_MODEL_CACHE_DIR", str(state.root / "model_cache"))) / slugify(model.get("king_id") or model.get("model_repo") or job_id)
    combined_path = run_dir / "standardized_results.json"
    completed_rows: dict[str, dict[str, Any]] = {}
    active: dict[str, dict[str, Any]] = {}

    log_event(job_root, "job_started", {"job_id": job_id, "benchmarks": benchmarks, "mode": "hybrid-per-gpu-then-mmlu-pro-accelerate", "gpu_ids": gpu_ids, "model_cache_dir": str(model_cache_dir), "phase1": phase1, "phase2": phase2})
    append_line(job_root / "gpu_snapshots.jsonl", {"at": utcnow_iso(), "phase": "start", "snapshot": gpu_snapshot()})

    download_std = run_dir / "download_standardized_results.json"
    download_log = job_root / "download.log"
    download_cmd = build_eval_cmd(
        job=job,
        eval_script=eval_script,
        model_input=model["model_input"],
        run_dir=run_dir,
        benchmarks=benchmarks,
        standardized_results_path=download_std,
        model_args_extra="",
        device="none",
        download_only=True,
        snapshot_dir=model_cache_dir,
    )
    with download_log.open("a") as log:
        log.write("$ " + quote_cmd(download_cmd) + "\n")
        log.flush()
        log_event(job_root, "download_started", {"cmd": quote_cmd(download_cmd), "log": str(download_log)})
        started = time.time()
        proc = subprocess.run(download_cmd, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=str(state.root), check=False)
        download_wall = round(time.time() - started, 2)
    if proc.returncode != 0:
        log_event(job_root, "download_failed", {"returncode": proc.returncode, "wall_time_s": download_wall, "log": str(download_log)})
        results = write_combined_results(
            path=combined_path,
            job_id=job_id,
            model=model,
            model_payload=(read_json(download_std, {}) or {}).get("model"),
            run_dir=run_dir,
            started_at=started_at,
            finished_at=utcnow_iso(),
            benchmarks=benchmarks,
            completed_rows={bench: {"name": bench, "status": "failed", "worker_log": str(download_log), "returncode": proc.returncode} for bench in benchmarks},
            active={},
            pending=[],
            mode="hybrid-per-gpu-then-mmlu-pro-accelerate",
            gpu_ids=gpu_ids,
        )
        return {"returncode": proc.returncode, "results": results, "artifacts": {"remote_run_dir": str(run_dir), "remote_log": str(download_log)}}

    download_payload = read_json(download_std, {}) or {}
    model_payload = download_payload.get("model") if isinstance(download_payload, dict) else None
    local_model_path = (model_payload or {}).get("local_path") or str(model_cache_dir)
    log_event(job_root, "download_completed", {"wall_time_s": download_wall, "local_model_path": local_model_path})

    def write_progress(phase: str, pending: list[str]) -> None:
        snapshot = gpu_snapshot()
        append_line(job_root / "gpu_snapshots.jsonl", {"at": utcnow_iso(), "phase": phase, "snapshot": snapshot})
        results = write_combined_results(
            path=combined_path,
            job_id=job_id,
            model=model,
            model_payload=model_payload,
            run_dir=run_dir,
            started_at=started_at,
            finished_at=None,
            benchmarks=benchmarks,
            completed_rows=completed_rows,
            active=active,
            pending=pending,
            mode="hybrid-per-gpu-then-mmlu-pro-accelerate",
            gpu_ids=gpu_ids,
        )
        event = {"event": "job_progress", "job_id": job_id, "model": model, "benchmarks": benchmarks, "status": "running", "mode": "hybrid-per-gpu-then-mmlu-pro-accelerate", "phase": phase, "updated_at": utcnow_iso(), "active": {bench: {"gpu_id": info["gpu_id"], "pid": info["pid"], "log": str(info["log_path"])} for bench, info in active.items()}, "pending": pending, "partial_results": results, "gpu_snapshot": snapshot}
        state.set_current(event)
        state.post_event(event)
        pm2_event("job_progress", {"job_id": job_id, "mode": "hybrid-per-gpu-then-mmlu-pro-accelerate", "phase": phase, "active": list(active), "pending": pending, "totals": results.get("totals"), "gpu": compact_gpu_snapshot(snapshot)})

    def start_single_gpu_benchmark(bench: str, gpu_id: str) -> dict[str, Any]:
        bench_slug = slugify(bench)
        bench_run_dir = run_dir / "benchmarks" / bench_slug
        bench_log = job_root / "benchmark_logs" / f"{bench_slug}.log"
        child_std = bench_run_dir / "standardized_results.json"
        child_job = dict(job)
        child_job["lm_eval_bin"] = single_gpu_lm_eval_bin(state)
        child_model_args = os.environ.get("TEUTONIC_KING_BENCH_PARALLEL_MODEL_ARGS_EXTRA", "")
        child_device = os.environ.get("TEUTONIC_KING_BENCH_CHILD_DEVICE", "cuda:0")
        cmd = build_eval_cmd(
            job=child_job,
            eval_script=eval_script,
            model_input=local_model_path,
            run_dir=bench_run_dir,
            benchmarks=[bench],
            standardized_results_path=child_std,
            model_args_extra=child_model_args,
            device=child_device,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        bench_log.parent.mkdir(parents=True, exist_ok=True)
        with bench_log.open("a") as log:
            log.write(f"# benchmark={bench} phase=single-gpu gpu_id={gpu_id} CUDA_VISIBLE_DEVICES={gpu_id}\n")
            log.write("$ " + quote_cmd(cmd) + "\n")
            log.flush()
        log_fh = bench_log.open("a")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, cwd=str(state.root), env=env)
        info = {"benchmark": bench, "gpu_id": gpu_id, "pid": proc.pid, "proc": proc, "log_fh": log_fh, "log_path": bench_log, "std_path": child_std, "started_monotonic": time.time(), "started_at": utcnow_iso(), "cmd": quote_cmd(cmd)}
        log_event(job_root, "benchmark_started", {k: v for k, v in info.items() if k not in {"proc", "log_fh"}})
        return info

    pending = list(phase1)
    gpu_cursor = 0
    last_progress = 0.0
    while pending or active:
        while pending and len(active) < len(gpu_ids):
            bench = pending.pop(0)
            gpu_id = gpu_ids[gpu_cursor % len(gpu_ids)]
            gpu_cursor += 1
            active[bench] = start_single_gpu_benchmark(bench, gpu_id)

        for bench, info in list(active.items()):
            proc = info["proc"]
            returncode = proc.poll()
            if returncode is None:
                continue
            info["log_fh"].close()
            wall = time.time() - info["started_monotonic"]
            row = read_child_benchmark(bench, info["std_path"], returncode, wall, info["gpu_id"], info["log_path"])
            completed_rows[bench] = row
            del active[bench]
            log_event(job_root, "benchmark_finished", {"benchmark": bench, "gpu_id": info["gpu_id"], "pid": info["pid"], "returncode": returncode, "wall_time_s": round(wall, 2), "status": row.get("status"), "log": str(info["log_path"])})

        now = time.time()
        if now - last_progress >= progress_interval or not (pending or active):
            last_progress = now
            write_progress("phase1-single-gpu", pending + phase2)
        if pending or active:
            time.sleep(5)

    if phase2:
        bench = "MMLU-Pro"
        bench_slug = slugify(bench)
        bench_run_dir = run_dir / "benchmarks" / bench_slug
        bench_log = job_root / "benchmark_logs" / f"{bench_slug}.log"
        child_std = bench_run_dir / "standardized_results.json"
        accel_job = dict(job)
        accel_job["lm_eval_bin"] = accelerate_lm_eval_bin(state, gpu_ids)
        accel_model_args = os.environ.get("TEUTONIC_KING_BENCH_ACCELERATE_MODEL_ARGS_EXTRA", "")
        accel_device = os.environ.get("TEUTONIC_KING_BENCH_ACCELERATE_DEVICE", "cuda")
        cmd = build_eval_cmd(
            job=accel_job,
            eval_script=eval_script,
            model_input=local_model_path,
            run_dir=bench_run_dir,
            benchmarks=[bench],
            standardized_results_path=child_std,
            model_args_extra=accel_model_args,
            device=accel_device,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("NCCL_DEBUG", os.environ.get("NCCL_DEBUG", "WARN"))
        bench_log.parent.mkdir(parents=True, exist_ok=True)
        with bench_log.open("a") as log:
            log.write(f"# benchmark={bench} phase=accelerate-all-gpus CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            log.write("$ " + quote_cmd(cmd) + "\n")
            log.flush()
        log_fh = bench_log.open("a")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, cwd=str(state.root), env=env)
        active[bench] = {"benchmark": bench, "gpu_id": ",".join(gpu_ids), "pid": proc.pid, "proc": proc, "log_fh": log_fh, "log_path": bench_log, "std_path": child_std, "started_monotonic": time.time(), "started_at": utcnow_iso(), "cmd": quote_cmd(cmd)}
        log_event(job_root, "benchmark_started", {k: v for k, v in active[bench].items() if k not in {"proc", "log_fh"}})
        last_progress = 0.0
        while active:
            returncode = proc.poll()
            if returncode is not None:
                log_fh.close()
                wall = time.time() - active[bench]["started_monotonic"]
                row = read_child_benchmark(bench, child_std, returncode, wall, ",".join(gpu_ids), bench_log)
                completed_rows[bench] = row
                del active[bench]
                log_event(job_root, "benchmark_finished", {"benchmark": bench, "gpu_id": ",".join(gpu_ids), "pid": proc.pid, "returncode": returncode, "wall_time_s": round(wall, 2), "status": row.get("status"), "log": str(bench_log)})
                break
            now = time.time()
            if now - last_progress >= progress_interval:
                last_progress = now
                write_progress("phase2-mmlu-pro-all-gpus", [])
            time.sleep(5)

    finished_at = utcnow_iso()
    final_results = write_combined_results(
        path=combined_path,
        job_id=job_id,
        model=model,
        model_payload=model_payload,
        run_dir=run_dir,
        started_at=started_at,
        finished_at=finished_at,
        benchmarks=benchmarks,
        completed_rows=completed_rows,
        active={},
        pending=[],
        mode="hybrid-per-gpu-then-mmlu-pro-accelerate",
        gpu_ids=gpu_ids,
    )
    append_line(job_root / "gpu_snapshots.jsonl", {"at": utcnow_iso(), "phase": "finish", "snapshot": gpu_snapshot()})
    returncode = 0 if all(row.get("status") == "completed" for row in completed_rows.values()) and len(completed_rows) == len(benchmarks) else 1
    log_event(job_root, "job_finished", {"job_id": job_id, "returncode": returncode, "totals": final_results.get("totals")})
    return {"returncode": returncode, "results": final_results, "artifacts": {"remote_run_dir": str(run_dir), "remote_log": str(job_root / "worker_events.jsonl"), "gpu_snapshots": str(job_root / "gpu_snapshots.jsonl"), "benchmark_logs": str(job_root / "benchmark_logs")}}

def run_parallel_job(state: WorkerState, job: dict[str, Any]) -> dict[str, Any]:
    job_id = job["job_id"]
    model = job["model"]
    benchmarks = list(job.get("benchmarks") or DEFAULT_BENCHMARKS)
    job_root = state.root / "jobs" / job_id
    run_dir = job_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_script = Path(job.get("eval_script") or state.root / "eval_king_benchmarks.py")
    started_at = utcnow_iso()
    progress_interval = int(job.get("progress_interval_s") or os.environ.get("TEUTONIC_WORKER_PROGRESS_INTERVAL_S", "60"))
    gpu_ids = split_csv(os.environ.get("TEUTONIC_KING_BENCH_GPU_IDS"), [str(i) for i in range(8)])
    gpu_ids = gpu_ids[: max(1, int(os.environ.get("TEUTONIC_KING_BENCH_MAX_PARALLEL_BENCHMARKS", str(len(gpu_ids)))))]
    model_cache_dir = Path(os.environ.get("TEUTONIC_KING_BENCH_MODEL_CACHE_DIR", str(state.root / "model_cache"))) / slugify(model.get("king_id") or model.get("model_repo") or job_id)
    combined_path = run_dir / "standardized_results.json"
    completed_rows: dict[str, dict[str, Any]] = {}
    pending = list(benchmarks)
    active: dict[str, dict[str, Any]] = {}

    log_event(job_root, "job_started", {"job_id": job_id, "benchmarks": benchmarks, "mode": "parallel-per-gpu", "gpu_ids": gpu_ids, "model_cache_dir": str(model_cache_dir)})
    append_line(job_root / "gpu_snapshots.jsonl", {"at": utcnow_iso(), "phase": "start", "snapshot": gpu_snapshot()})

    download_std = run_dir / "download_standardized_results.json"
    download_log = job_root / "download.log"
    download_cmd = build_eval_cmd(
        job=job,
        eval_script=eval_script,
        model_input=model["model_input"],
        run_dir=run_dir,
        benchmarks=benchmarks,
        standardized_results_path=download_std,
        model_args_extra="",
        device="none",
        download_only=True,
        snapshot_dir=model_cache_dir,
    )
    with download_log.open("a") as log:
        log.write("$ " + quote_cmd(download_cmd) + "\n")
        log.flush()
        log_event(job_root, "download_started", {"cmd": quote_cmd(download_cmd), "log": str(download_log)})
        started = time.time()
        proc = subprocess.run(download_cmd, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=str(state.root), check=False)
        download_wall = round(time.time() - started, 2)
    if proc.returncode != 0:
        log_event(job_root, "download_failed", {"returncode": proc.returncode, "wall_time_s": download_wall, "log": str(download_log)})
        results = write_combined_results(
            path=combined_path,
            job_id=job_id,
            model=model,
            model_payload=(read_json(download_std, {}) or {}).get("model"),
            run_dir=run_dir,
            started_at=started_at,
            finished_at=utcnow_iso(),
            benchmarks=benchmarks,
            completed_rows={bench: {"name": bench, "status": "failed", "worker_log": str(download_log), "returncode": proc.returncode} for bench in benchmarks},
            active={},
            pending=[],
            mode="parallel-per-gpu",
            gpu_ids=gpu_ids,
        )
        return {"returncode": proc.returncode, "results": results, "artifacts": {"remote_run_dir": str(run_dir), "remote_log": str(download_log)}}

    download_payload = read_json(download_std, {}) or {}
    model_payload = download_payload.get("model") if isinstance(download_payload, dict) else None
    local_model_path = (model_payload or {}).get("local_path") or str(model_cache_dir)
    log_event(job_root, "download_completed", {"wall_time_s": download_wall, "local_model_path": local_model_path})

    def start_benchmark(bench: str, gpu_id: str) -> dict[str, Any]:
        bench_slug = slugify(bench)
        bench_run_dir = run_dir / "benchmarks" / bench_slug
        bench_log = job_root / "benchmark_logs" / f"{bench_slug}.log"
        child_std = bench_run_dir / "standardized_results.json"
        child_model_args = os.environ.get("TEUTONIC_KING_BENCH_PARALLEL_MODEL_ARGS_EXTRA", "")
        child_device = os.environ.get("TEUTONIC_KING_BENCH_CHILD_DEVICE", "cuda:0")
        cmd = build_eval_cmd(
            job=job,
            eval_script=eval_script,
            model_input=local_model_path,
            run_dir=bench_run_dir,
            benchmarks=[bench],
            standardized_results_path=child_std,
            model_args_extra=child_model_args,
            device=child_device,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        bench_log.parent.mkdir(parents=True, exist_ok=True)
        with bench_log.open("a") as log:
            log.write(f"# benchmark={bench} gpu_id={gpu_id} CUDA_VISIBLE_DEVICES={gpu_id}\n")
            log.write("$ " + quote_cmd(cmd) + "\n")
            log.flush()
        log_fh = bench_log.open("a")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, cwd=str(state.root), env=env)
        info = {"benchmark": bench, "gpu_id": gpu_id, "pid": proc.pid, "proc": proc, "log_fh": log_fh, "log_path": bench_log, "std_path": child_std, "started_monotonic": time.time(), "started_at": utcnow_iso(), "cmd": quote_cmd(cmd)}
        log_event(job_root, "benchmark_started", {k: v for k, v in info.items() if k not in {"proc", "log_fh"}})
        return info

    last_progress = 0.0
    gpu_cursor = 0
    while pending or active:
        while pending and len(active) < len(gpu_ids):
            bench = pending.pop(0)
            gpu_id = gpu_ids[gpu_cursor % len(gpu_ids)]
            gpu_cursor += 1
            active[bench] = start_benchmark(bench, gpu_id)

        for bench, info in list(active.items()):
            proc = info["proc"]
            returncode = proc.poll()
            if returncode is None:
                continue
            info["log_fh"].close()
            wall = time.time() - info["started_monotonic"]
            row = read_child_benchmark(bench, info["std_path"], returncode, wall, info["gpu_id"], info["log_path"])
            completed_rows[bench] = row
            del active[bench]
            log_event(job_root, "benchmark_finished", {"benchmark": bench, "gpu_id": info["gpu_id"], "pid": info["pid"], "returncode": returncode, "wall_time_s": round(wall, 2), "status": row.get("status"), "log": str(info["log_path"])})

        now = time.time()
        if now - last_progress >= progress_interval or not (pending or active):
            last_progress = now
            snapshot = gpu_snapshot()
            append_line(job_root / "gpu_snapshots.jsonl", {"at": utcnow_iso(), "phase": "progress", "snapshot": snapshot})
            results = write_combined_results(
                path=combined_path,
                job_id=job_id,
                model=model,
                model_payload=model_payload,
                run_dir=run_dir,
                started_at=started_at,
                finished_at=None,
                benchmarks=benchmarks,
                completed_rows=completed_rows,
                active=active,
                pending=pending,
                mode="parallel-per-gpu",
                gpu_ids=gpu_ids,
            )
            event = {"event": "job_progress", "job_id": job_id, "model": model, "benchmarks": benchmarks, "status": "running", "updated_at": utcnow_iso(), "active": {bench: {"gpu_id": info["gpu_id"], "pid": info["pid"], "log": str(info["log_path"])} for bench, info in active.items()}, "pending": pending, "partial_results": results, "gpu_snapshot": snapshot}
            state.set_current(event)
            state.post_event(event)
        if pending or active:
            time.sleep(5)

    finished_at = utcnow_iso()
    final_results = write_combined_results(
        path=combined_path,
        job_id=job_id,
        model=model,
        model_payload=model_payload,
        run_dir=run_dir,
        started_at=started_at,
        finished_at=finished_at,
        benchmarks=benchmarks,
        completed_rows=completed_rows,
        active={},
        pending=[],
        mode="parallel-per-gpu",
        gpu_ids=gpu_ids,
    )
    append_line(job_root / "gpu_snapshots.jsonl", {"at": utcnow_iso(), "phase": "finish", "snapshot": gpu_snapshot()})
    returncode = 0 if all(row.get("status") == "completed" for row in completed_rows.values()) and len(completed_rows) == len(benchmarks) else 1
    log_event(job_root, "job_finished", {"job_id": job_id, "returncode": returncode, "totals": final_results.get("totals")})
    return {"returncode": returncode, "results": final_results, "artifacts": {"remote_run_dir": str(run_dir), "remote_log": str(job_root / "worker_events.jsonl"), "gpu_snapshots": str(job_root / "gpu_snapshots.jsonl"), "benchmark_logs": str(job_root / "benchmark_logs")}}


def run_sequential_job(state: WorkerState, job: dict[str, Any]) -> dict[str, Any]:
    job_id = job["job_id"]
    model = job["model"]
    benchmarks = list(job.get("benchmarks") or DEFAULT_BENCHMARKS)
    job_root = state.root / "jobs" / job_id
    run_dir = job_root / "run"
    log_path = job_root / "eval.log"
    job_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_script = Path(job.get("eval_script") or state.root / "eval_king_benchmarks.py")
    mode = os.environ.get("TEUTONIC_KING_BENCH_WORKER_MODE", "accelerate-data-parallel")
    snapshot_dir: Path | None = None
    if mode == "accelerate-data-parallel":
        model_args_extra = os.environ.get("TEUTONIC_KING_BENCH_ACCELERATE_MODEL_ARGS_EXTRA", "")
        device = os.environ.get("TEUTONIC_KING_BENCH_ACCELERATE_DEVICE", "cuda")
        snapshot_dir = Path(os.environ.get("TEUTONIC_KING_BENCH_MODEL_CACHE_DIR", str(state.root / "model_cache"))) / slugify(model.get("king_id") or model.get("model_repo") or job_id)
    else:
        model_args_extra = job.get("model_args_extra") or os.environ.get("TEUTONIC_KING_BENCH_MODEL_ARGS_EXTRA", "")
        device = str(job.get("device") or os.environ.get("TEUTONIC_KING_BENCH_DEVICE", "auto"))
    cmd = build_eval_cmd(
        job=job,
        eval_script=eval_script,
        model_input=model["model_input"],
        run_dir=run_dir,
        benchmarks=benchmarks,
        standardized_results_path=run_dir / "standardized_results.json",
        model_args_extra=model_args_extra,
        device=device,
        snapshot_dir=snapshot_dir,
    )
    start_snapshot = gpu_snapshot()
    log_event(job_root, "job_subprocess_starting", {"mode": mode, "cmd": quote_cmd(cmd), "log": str(log_path), "gpu": compact_gpu_snapshot(start_snapshot)})
    returncode = 1
    progress_interval = int(job.get("progress_interval_s") or os.environ.get("TEUTONIC_WORKER_PROGRESS_INTERVAL_S", "60"))
    with log_path.open("a") as log:
        log.write("# mode=" + mode + "\n")
        log.write("# gpu_snapshot_start=" + json.dumps(start_snapshot, sort_keys=False) + "\n")
        log.write("$ " + quote_cmd(cmd) + "\n")
        log.flush()
        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("NCCL_DEBUG", os.environ.get("NCCL_DEBUG", "WARN"))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(state.root),
            env=env,
            bufsize=1,
        )
        last_progress = 0.0
        stdout = proc.stdout
        while proc.poll() is None:
            if stdout is not None:
                ready, _, _ = select.select([stdout], [], [], 1.0)
                if ready:
                    line = stdout.readline()
                    if line:
                        log.write(line)
                        log.flush()
                        print(f"[eval:{job_id}] {line.rstrip()}", flush=True)
            now = time.time()
            if now - last_progress >= progress_interval:
                last_progress = now
                snapshot = gpu_snapshot()
                payload = read_json(run_dir / "standardized_results.json", {}) or {}
                append_line(job_root / "gpu_snapshots.jsonl", {"at": utcnow_iso(), "phase": "subprocess", "pid": proc.pid, "snapshot": snapshot})
                progress = {
                    "event": "job_progress",
                    "job_id": job_id,
                    "model": model,
                    "benchmarks": benchmarks,
                    "status": "running",
                    "mode": mode,
                    "pid": proc.pid,
                    "updated_at": utcnow_iso(),
                    "partial_results": payload,
                    "gpu_snapshot": snapshot,
                    "remote_log": str(log_path),
                }
                state.set_current(progress)
                state.post_event(progress)
                pm2_event("job_progress", {
                    "job_id": job_id,
                    "mode": mode,
                    "pid": proc.pid,
                    "totals": payload.get("totals") if isinstance(payload, dict) else None,
                    "gpu": compact_gpu_snapshot(snapshot),
                    "remote_log": str(log_path),
                })
        if stdout is not None:
            for line in stdout:
                log.write(line)
                log.flush()
                print(f"[eval:{job_id}] {line.rstrip()}", flush=True)
        returncode = proc.returncode or 0
        finish_snapshot = gpu_snapshot()
        log.write("# gpu_snapshot_finish=" + json.dumps(finish_snapshot, sort_keys=False) + "\n")
    results = read_json(run_dir / "standardized_results.json", {}) or {}
    log_event(job_root, "job_subprocess_finished", {"mode": mode, "returncode": returncode, "log": str(log_path), "totals": results.get("totals"), "gpu": compact_gpu_snapshot(gpu_snapshot())})
    return {"returncode": returncode, "results": results, "artifacts": {"remote_run_dir": str(run_dir), "remote_log": str(log_path), "gpu_snapshots": str(job_root / "gpu_snapshots.jsonl"), "worker_events": str(job_root / "worker_events.jsonl")}}


def run_job(state: WorkerState, job: dict[str, Any]) -> None:
    job_id = job["job_id"]
    model = job["model"]
    benchmarks = list(job.get("benchmarks") or DEFAULT_BENCHMARKS)
    started_at = utcnow_iso()
    status = {"status": "running", "job_id": job_id, "model": model, "benchmarks": benchmarks, "started_at": started_at, "updated_at": started_at}
    state.set_current(status)
    state.post_event({"event": "job_started", **status})
    returncode = 1
    result: dict[str, Any] = {"results": {}, "artifacts": {}}
    try:
        mode = os.environ.get("TEUTONIC_KING_BENCH_WORKER_MODE", "accelerate-data-parallel")
        if mode == "hybrid-dashboard" and len(benchmarks) > 1:
            result = run_hybrid_job(state, job)
        elif mode == "per-benchmark-gpu" and len(benchmarks) > 1:
            result = run_parallel_job(state, job)
        else:
            result = run_sequential_job(state, job)
        returncode = int(result.get("returncode", 1))
    except Exception as exc:
        job_root = state.root / "jobs" / job_id
        append_line(job_root / "worker_errors.jsonl", {"at": utcnow_iso(), "error": repr(exc)})
        log_event(job_root, "job_exception", {"error": repr(exc)})

    finished_at = utcnow_iso()
    final = {
        "event": "job_finished",
        "status": "completed" if returncode == 0 else "failed",
        "job_id": job_id,
        "model": model,
        "benchmarks": benchmarks,
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": returncode,
        "results": result.get("results") or {},
        "artifacts": result.get("artifacts") or {},
    }
    state.set_current(final)
    state.post_result(final)
    with state.lock:
        state.current = {"status": "idle", "last_job": final, "updated_at": utcnow_iso()}
        state.thread = None
        write_json(state.state_path, state.current)


class Handler(BaseHTTPRequestHandler):
    server_version = "TeutonicKingBenchmarkWorker/1.1"

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _auth(self) -> bool:
        state: WorkerState = self.server.state  # type: ignore[attr-defined]
        return require_bearer(dict(self.headers), state.token)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True, "at": utcnow_iso(), "gpu_snapshot": gpu_snapshot()})
            return
        if not self._auth():
            self._json(401, {"error": "unauthorized"})
            return
        if self.path == "/status":
            state: WorkerState = self.server.state  # type: ignore[attr-defined]
            self._json(200, state.snapshot())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._auth():
            self._json(401, {"error": "unauthorized"})
            return
        state: WorkerState = self.server.state  # type: ignore[attr-defined]
        if self.path == "/jobs":
            job = self._read_json()
            with state.lock:
                if state.thread is not None:
                    self._json(409, {"error": "worker busy", "current": state.current})
                    return
                if not job.get("job_id") or not isinstance(job.get("model"), dict):
                    self._json(400, {"error": "job_id and model are required"})
                    return
                thread = threading.Thread(target=run_job, args=(state, job), daemon=True)
                state.thread = thread
                state.current = {"status": "accepted", "job_id": job["job_id"], "accepted_at": utcnow_iso()}
                write_json(state.state_path, state.current)
                thread.start()
            self._json(202, {"accepted": True, "job_id": job["job_id"]})
            return
        self._json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        append_line(self.server.state.root / "access.jsonl", {"at": utcnow_iso(), "client": self.client_address[0], "message": fmt % args})  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("TEUTONIC_WORKER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TEUTONIC_WORKER_PORT", "32000")))
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("TEUTONIC_WORKER_ROOT", "/root/teutonic/king-benchmark-worker")))
    parser.add_argument("--token", default=os.environ.get("TEUTONIC_KING_BENCH_WORKER_TOKEN", ""))
    parser.add_argument("--controller-url", default=os.environ.get("TEUTONIC_KING_BENCH_CONTROLLER_URL", ""))
    parser.add_argument("--callback-token", default=os.environ.get("TEUTONIC_KING_BENCH_CONTROLLER_TOKEN") or os.environ.get("TEUTONIC_KING_BENCH_WORKER_TOKEN", ""))
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("TEUTONIC_KING_BENCH_WORKER_TOKEN is required")
    state = WorkerState(args.root, args.token, args.controller_url, args.callback_token)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state  # type: ignore[attr-defined]
    print(f"worker listening on {args.host}:{args.port} root={args.root}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
