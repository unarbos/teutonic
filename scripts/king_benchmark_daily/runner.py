#!/usr/bin/env python3
"""Daily Lium-backed benchmark service for the current Teutonic king.

This is intentionally separate from scripts/eval_king_benchmarks_lium.py: that
file remains the manual/operator runner, while this file is the PM2 service that
fetches the current king every 24h, rents one pod per benchmark, runs the full
panel, and writes dashboard-friendly JSON artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
import threading
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:  # Upload support reports a clear runtime error if missing.
    boto3 = None
    BotoConfig = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_king_benchmarks_lium as lium_eval  # noqa: E402

DASHBOARD_MIRRORS = [
    "https://us-east-1.hippius.com/teutonic-sn3/dashboard.json",
    "https://eu-central-1.hippius.com/teutonic-sn3/dashboard.json",
    "https://s3.hippius.com/teutonic-sn3/dashboard.json",
]

DEFAULT_RESULTS_ROOT = ROOT / "runs" / "king-benchmark-daily"
DEFAULT_REGISTRY = ROOT / "runs" / "lium-rentals" / "registry.json"
DEFAULT_REMOTE_BASE = "/root/king-benchmark-evals-daily"
DEFAULT_HIPPIUS_ENDPOINT = "https://s3.hippius.com"
DEFAULT_HIPPIUS_BUCKET = "teutonic-sn3"
DEFAULT_LATEST_KEY = "king-benchmark-daily/latest.json"
DEFAULT_HISTORY_KEY = "king-benchmark-daily/history.jsonl"
SCHEMA_VERSION = "teutonic-king-benchmark-daily.v1"
DEFAULT_INTERVAL_HOURS = 24.0
DEFAULT_POLL_SECONDS = 300

BENCHMARK_ORDER = [
    "MMLU",
    "MMLU-Pro",
    "BBH",
    "ARC-C",
    "TruthfulQA",
    "WinoGrande",
    "HellaSwag",
    "GSM8K",
    "MATH",
    "HumanEval",
    "MBPP",
]
DEFAULT_BENCHMARKS = [
    "MMLU",
    "MMLU-Pro",
    "BBH",
    "ARC-C",
    "TruthfulQA",
    "WinoGrande",
]


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    gpu_preferences: tuple[str, ...] = ("H200", "A100")
    batch_size: str = "auto"
    fewshot_overrides: str = ""
    log_samples: bool = False
    ttl: str = ""


# Explicit batch sizes follow the observed good settings from the manual run:
# H200 can carry MMLU-Pro batch 8; A100-80GB can carry MATH batch 4.
BENCHMARK_PROFILES: dict[str, BenchmarkProfile] = {
    "MMLU": BenchmarkProfile("MMLU", batch_size="auto"),
    "MMLU-Pro": BenchmarkProfile("MMLU-Pro", batch_size="8"),
    "BBH": BenchmarkProfile("BBH", batch_size="auto"),
    "ARC-C": BenchmarkProfile("ARC-C", batch_size="auto"),
    "TruthfulQA": BenchmarkProfile("TruthfulQA", batch_size="auto"),
    "WinoGrande": BenchmarkProfile("WinoGrande", batch_size="auto"),
    "HellaSwag": BenchmarkProfile("HellaSwag", batch_size="auto"),
    "GSM8K": BenchmarkProfile("GSM8K", batch_size="auto"),
    "MATH": BenchmarkProfile("MATH", gpu_preferences=("A100", "H200"), batch_size="4"),
    "HumanEval": BenchmarkProfile("HumanEval", batch_size="auto", log_samples=True),
    # MBPP 0-shot produced invalid all-zero results in practice; the daily
    # dashboard run records MBPP as 3-shot and marks that in the JSON.
    "MBPP": BenchmarkProfile("MBPP", batch_size="auto", fewshot_overrides="MBPP=3", log_samples=True),
}


def parse_benchmark_selection(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return list(DEFAULT_BENCHMARKS)
    canonical = {label.lower(): label for label in BENCHMARK_ORDER}
    selected: list[str] = []
    for part in raw_value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        label = canonical.get(key)
        if label is None:
            raise SystemExit(f"unknown benchmark {part!r}; expected one of: {', '.join(BENCHMARK_ORDER)}")
        if label not in selected:
            selected.append(label)
    if not selected:
        raise SystemExit("no benchmarks selected")
    return selected


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-") or "run"


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=False) + "\n")


def resolve_secret_from_env_or_doppler(
    names: list[str], *, project: str, config: str
) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    for name in names:
        value = lium_eval.doppler_secret(name, project=project, config=config).strip()
        if value:
            return value
    return ""


class HippiusPublisher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.enabled = args.upload_hippius
        self.bucket = args.hippius_bucket
        self.latest_key = args.hippius_latest_key
        self.history_key = args.hippius_history_key
        self.errors_path = args.results_root / "upload_errors.jsonl"
        self.client = None
        if not self.enabled:
            return
        access_key = resolve_secret_from_env_or_doppler(
            ["TEUTONIC_HIPPIUS_ACCESS_KEY", "HIPPIUS_ACCESS_KEY"],
            project=args.doppler_project,
            config=args.doppler_config,
        )
        secret_key = resolve_secret_from_env_or_doppler(
            ["TEUTONIC_HIPPIUS_SECRET_KEY", "HIPPIUS_SECRET_KEY"],
            project=args.doppler_project,
            config=args.doppler_config,
        )
        if not access_key or not secret_key:
            self.enabled = False
            append_jsonl(
                self.errors_path,
                {
                    "at": utcnow_iso(),
                    "error": "missing Hippius S3 credentials; latest.json was only written locally",
                    "expected_env": ["TEUTONIC_HIPPIUS_ACCESS_KEY", "TEUTONIC_HIPPIUS_SECRET_KEY"],
                },
            )
            return
        if boto3 is None or BotoConfig is None:
            self.enabled = False
            append_jsonl(
                self.errors_path,
                {"at": utcnow_iso(), "error": "boto3/botocore is not installed; latest.json was only written locally"},
            )
            return
        self.client = boto3.client(
            "s3",
            endpoint_url=args.hippius_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="decentralized",
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=15,
                read_timeout=120,
                retries={"max_attempts": 5, "mode": "adaptive"},
            ),
        )

    def put_latest(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            body = json.dumps(payload, separators=(",", ":"), sort_keys=False).encode("utf-8")
            self.client.put_object(
                Bucket=self.bucket,
                Key=self.latest_key,
                Body=body,
                ContentType="application/json",
            )
        except Exception as exc:
            append_jsonl(
                self.errors_path,
                {"at": utcnow_iso(), "key": self.latest_key, "error": repr(exc)},
            )

    def append_history(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.client is None or not self.history_key:
            return
        try:
            line = (json.dumps(payload, sort_keys=False) + "\n").encode("utf-8")
            existing = b""
            try:
                existing = self.client.get_object(Bucket=self.bucket, Key=self.history_key)["Body"].read()
            except Exception as exc:
                if "NoSuchKey" not in str(exc) and "404" not in str(exc):
                    raise
            self.client.put_object(
                Bucket=self.bucket,
                Key=self.history_key,
                Body=existing + line,
                ContentType="application/x-ndjson",
            )
        except Exception as exc:
            append_jsonl(
                self.errors_path,
                {"at": utcnow_iso(), "key": self.history_key, "error": repr(exc)},
            )


def fetch_json_url(url: str, timeout_s: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "teutonic-king-benchmark-daily/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def fetch_dashboard(mirrors: list[str], timeout_s: int) -> tuple[str, dict[str, Any]]:
    errors = []
    for url in mirrors:
        try:
            payload = fetch_json_url(url, timeout_s)
            if isinstance(payload, dict) and isinstance(payload.get("king"), dict):
                return url, payload
            errors.append(f"{url}: missing king object")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            errors.append(f"{url}: {exc!r}")
    raise RuntimeError("failed to fetch dashboard from all mirrors: " + "; ".join(errors))


def current_king_from_dashboard(dashboard: dict[str, Any]) -> dict[str, Any]:
    king = dashboard.get("king") or {}
    model_repo = king.get("model_repo") or king.get("hf_repo") or king.get("repo")
    digest = king.get("king_digest") or king.get("digest") or king.get("revision")
    if not model_repo:
        raise RuntimeError("dashboard king has no model_repo/hf_repo")
    model_input = f"{model_repo}@{digest}" if digest and str(digest).startswith(("sha256:", "hf:")) else model_repo
    return {
        "source": "dashboard",
        "model_repo": model_repo,
        "king_digest": digest,
        "model_input": model_input,
        "hotkey": king.get("hotkey"),
        "reign_number": king.get("reign_number"),
        "crowned_at": king.get("crowned_at"),
        "crowned_block": king.get("crowned_block"),
        "challenge_id": king.get("challenge_id"),
        "previous_repo": king.get("previous_repo"),
    }


def make_lium_args(base: argparse.Namespace, profile: BenchmarkProfile) -> argparse.Namespace:
    return argparse.Namespace(
        doppler_project=base.doppler_project,
        doppler_config=base.doppler_config,
        registry=base.registry,
        skip_bootstrap=base.skip_bootstrap,
        keep_on_success=base.keep_on_success,
        delete_on_failure=base.delete_on_failure,
        batch_size=profile.batch_size,
        dtype=base.dtype,
        device=base.device,
        model_args_extra=base.model_args_extra,
        gen_kwargs=base.gen_kwargs,
        fewshot_overrides=profile.fewshot_overrides or base.fewshot_overrides,
        log_samples=profile.log_samples or base.log_samples,
        limit=base.limit,
        apply_chat_template=base.apply_chat_template,
        resume=base.resume,
    )


def run_lium(cmd: list[str], env: dict[str, str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return lium_eval.run(cmd, env=env, check=check)


def run_lium_up_with_timeout(cmd: list[str], env: dict[str, str], timeout_s: int) -> None:
    print("+ " + " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        if output:
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
        print(f"[warn] lium up timed out after {timeout_s}s; will probe pod reachability", flush=True)
        return
    output = proc.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    failed = proc.returncode != 0 or lium_eval.lium_output_failed(cmd, output)
    if failed:
        raise subprocess.CalledProcessError(proc.returncode or 1, cmd, output=output)


def rent_pod_for_benchmark(
    *,
    benchmark: str,
    run_id: str,
    profile: BenchmarkProfile,
    base_args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    name = f"daily-king-{slugify(benchmark).lower()}-{run_id[-13:]}"
    errors: list[str] = []
    for gpu in profile.gpu_preferences:
        try:
            cmd = ["lium", "up", "--name", name, "--ttl", profile.ttl or base_args.ttl, "--yes", "--gpu", gpu, "--count", "1"]
            if base_args.template_id:
                cmd += ["--template_id", base_args.template_id]
            run_lium_up_with_timeout(cmd, env, base_args.rent_timeout_s)
            lium_eval.wait_for_pod(name, env, base_args.wait_timeout_s)
            lium_eval.append_rental(
                base_args.registry,
                {
                    "created_at": utcnow_iso(),
                    "status": "running",
                    "pod_name": name,
                    "owner": "king_benchmark_daily",
                    "suite_run_id": run_id,
                    "benchmark": benchmark,
                    "executor_ref": None,
                    "ttl": profile.ttl or base_args.ttl,
                    "filters": {"gpu": gpu, "count": 1, "template_id": base_args.template_id},
                },
            )
            return {"pod_name": name, "gpu": gpu, "error": None}
        except Exception as exc:
            errors.append(f"{gpu}: {exc!r}")
            # If the pod was partially created but never became usable, try to
            # delete this automation-owned name before falling back.
            try:
                lium_eval.terminate_pod(name, env, base_args.registry, reason="rent_failed")
            except Exception:
                pass
    return {"pod_name": None, "gpu": None, "error": "; ".join(errors)}


def extract_benchmark_row(result: dict[str, Any], benchmark: str, profile: BenchmarkProfile) -> dict[str, Any]:
    payload = lium_eval.read_json_if_exists(result.get("standardized_results_path"))
    bench_row = {}
    if payload and isinstance(payload.get("benchmarks"), list) and payload["benchmarks"]:
        bench_row = dict(payload["benchmarks"][0])
    metric = bench_row.get("metric")
    score = metric.get("value") if isinstance(metric, dict) else None
    return {
        "name": benchmark,
        "task": bench_row.get("task"),
        "fewshot": bench_row.get("fewshot"),
        "status": bench_row.get("status") or result.get("status"),
        "metric": metric,
        "score": score,
        "batch_size": profile.batch_size,
        "fewshot_overrides": profile.fewshot_overrides,
        "log_samples": profile.log_samples,
        "pod_name": result.get("pod_name"),
        "gpu": result.get("gpu"),
        "local_run_dir": result.get("local_run_dir"),
        "summary_json": result.get("summary_path"),
        "standardized_results_json": result.get("standardized_results_path"),
        "error": result.get("error"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"requested": len(rows), "completed": 0, "failed": 0, "running": 0, "pending": 0}
    for row in rows:
        status = row.get("status")
        if status == "completed":
            counts["completed"] += 1
        elif status in {"running", "renting"}:
            counts["running"] += 1
        elif status in {"pending"}:
            counts["pending"] += 1
        else:
            counts["failed"] += 1
    return counts


def iso_add_hours(value: str | None, hours: float) -> str | None:
    if not value:
        return None
    try:
        base = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (base + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_offset(value: str) -> timezone:
    raw = (value or "+00:00").strip()
    sign = -1 if raw.startswith("-") else 1
    raw = raw.lstrip("+-")
    hours_s, minutes_s = (raw.split(":", 1) + ["0"])[:2] if ":" in raw else (raw, "0")
    delta = timedelta(hours=int(hours_s or 0), minutes=int(minutes_s or 0))
    return timezone(sign * delta)


def parse_daily_time(value: str) -> tuple[int, int]:
    raw = (value or "00:00").strip()
    hour_s, minute_s = (raw.split(":", 1) + ["0"])[:2] if ":" in raw else (raw, "0")
    hour, minute = int(hour_s), int(minute_s)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid daily schedule time: {value!r}")
    return hour, minute


def scheduled_slot_for_local_date(local_dt: datetime, args: argparse.Namespace) -> datetime:
    hour, minute = parse_daily_time(args.daily_time)
    return local_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


def latest_scheduled_slot(now: datetime, args: argparse.Namespace) -> datetime:
    tz = parse_utc_offset(args.schedule_utc_offset)
    local_now = now.astimezone(tz)
    slot = scheduled_slot_for_local_date(local_now, args)
    if local_now < slot:
        slot = slot - timedelta(days=1)
    return slot.astimezone(timezone.utc)


def next_scheduled_slot_after(anchor_iso: str | None, args: argparse.Namespace) -> str | None:
    if not anchor_iso:
        return None
    try:
        anchor = datetime.strptime(anchor_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    tz = parse_utc_offset(args.schedule_utc_offset)
    local_anchor = anchor.astimezone(tz)
    candidate = scheduled_slot_for_local_date(local_anchor, args)
    if candidate.astimezone(timezone.utc) <= anchor:
        candidate = candidate + timedelta(days=1)
    return candidate.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_schedule(args: argparse.Namespace, started_at: str, finished_at: str | None) -> dict[str, Any]:
    anchor = finished_at or started_at
    if args.daily_time:
        return {
            "interval_hours": args.interval_hours,
            "daily_time": args.daily_time,
            "utc_offset": args.schedule_utc_offset,
            "last_state_at": anchor,
            "next_run_at": next_scheduled_slot_after(anchor, args),
            "timezone": f"UTC{args.schedule_utc_offset}",
            "source": "pm2:teutonic-king-bench-lium",
        }
    return {
        "interval_hours": args.interval_hours,
        "last_state_at": anchor,
        "next_run_at": iso_add_hours(anchor, args.interval_hours),
        "timezone": "UTC",
        "source": "pm2:teutonic-king-bench-lium",
    }


def build_payload(
    *,
    run_id: str,
    status: str,
    started_at: str,
    finished_at: str | None,
    dashboard_url: str,
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    paths: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: BENCHMARK_ORDER.index(row["name"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "generated_at": utcnow_iso(),
        "started_at": started_at,
        "finished_at": finished_at,
        "dashboard_url": dashboard_url,
        "schedule": build_schedule(args, started_at, finished_at),
        "model": model,
        "totals": summarize(ordered),
        "artifacts": paths,
        "benchmarks": ordered,
    }


PRIVATE_BENCHMARK_FIELDS = {
    "local_run_dir",
    "summary_json",
    "standardized_results_json",
    "traceback",
}


def public_payload(
    payload: dict[str, Any], publisher: HippiusPublisher | None = None
) -> dict[str, Any]:
    public = dict(payload)
    bucket = publisher.bucket if publisher is not None else DEFAULT_HIPPIUS_BUCKET
    latest_key = publisher.latest_key if publisher is not None else DEFAULT_LATEST_KEY
    history_key = publisher.history_key if publisher is not None else DEFAULT_HISTORY_KEY
    public["artifacts"] = {
        "latest_json_s3": f"s3://{bucket}/{latest_key}",
        "history_jsonl_s3": f"s3://{bucket}/{history_key}" if history_key else None,
    }
    public["benchmarks"] = [
        {key: value for key, value in row.items() if key not in PRIVATE_BENCHMARK_FIELDS}
        for row in payload.get("benchmarks", [])
    ]
    return public


def write_status(
    results_root: Path,
    run_dir: Path,
    payload: dict[str, Any],
    publisher: HippiusPublisher | None = None,
) -> dict[str, Any]:
    write_json(run_dir / "run.json", payload)
    public = public_payload(payload, publisher)
    write_json(results_root / "latest.json", public)
    if publisher is not None:
        publisher.put_latest(public)
    return public


def run_suite_once(args: argparse.Namespace) -> dict[str, Any]:
    env = lium_eval.build_lium_env(args)
    dashboard_url, dashboard = fetch_dashboard(args.dashboard_url, args.dashboard_timeout_s)
    model = current_king_from_dashboard(dashboard)
    started_at = utcnow_iso()
    run_id = utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + slugify(model["model_repo"].split("/")[-1])
    run_dir = args.results_root.expanduser().resolve() / "runs" / run_id
    paths = {
        "results_root": str(args.results_root.expanduser().resolve()),
        "run_dir": str(run_dir),
        "run_json": str(run_dir / "run.json"),
        "latest_json": str(args.results_root.expanduser().resolve() / "latest.json"),
        "history_jsonl": str(args.results_root.expanduser().resolve() / "history.jsonl"),
        "registry": str(args.registry.expanduser().resolve()),
    }
    publisher = HippiusPublisher(args)
    selected_benchmarks = parse_benchmark_selection(args.benchmarks)

    rows = [
        {
            "name": label,
            "task": None,
            "fewshot": None,
            "status": "pending",
            "metric": None,
            "score": None,
            "batch_size": BENCHMARK_PROFILES[label].batch_size,
            "fewshot_overrides": BENCHMARK_PROFILES[label].fewshot_overrides,
            "log_samples": BENCHMARK_PROFILES[label].log_samples,
            "pod_name": None,
            "gpu": None,
            "local_run_dir": None,
            "summary_json": None,
            "standardized_results_json": None,
            "error": None,
        }
        for label in selected_benchmarks
    ]
    initial_payload = build_payload(
        run_id=run_id,
        status="running",
        started_at=started_at,
        finished_at=None,
        dashboard_url=dashboard_url,
        model=model,
        rows=rows,
        paths=paths,
        args=args,
    )
    write_status(args.results_root, run_dir, initial_payload, publisher)

    if args.dry_run:
        payload = build_payload(
            run_id=run_id,
            status="dry_run",
            started_at=started_at,
            finished_at=utcnow_iso(),
            dashboard_url=dashboard_url,
            model=model,
            rows=rows,
            paths=paths,
            args=args,
        )
        public = write_status(args.results_root, run_dir, payload, publisher)
        append_jsonl(args.results_root / "history.jsonl", public)
        publisher.append_history(public)
        return payload

    row_by_name = {row["name"]: row for row in rows}
    status_lock = threading.Lock()

    def update_row(label: str, **updates: Any) -> None:
        with status_lock:
            row_by_name[label].update(updates)
            payload = build_payload(
                run_id=run_id,
                status="running",
                started_at=started_at,
                finished_at=None,
                dashboard_url=dashboard_url,
                model=model,
                rows=list(row_by_name.values()),
                paths=paths,
                args=args,
            )
            write_status(args.results_root, run_dir, payload, publisher)

    def worker(label: str) -> dict[str, Any]:
        profile = BENCHMARK_PROFILES[label]
        update_row(label, status="renting")
        rental = rent_pod_for_benchmark(
            benchmark=label,
            run_id=run_id,
            profile=profile,
            base_args=args,
            env=env,
        )
        if not rental["pod_name"]:
            return {
                "name": label,
                "status": "failed",
                "pod_name": None,
                "gpu": None,
                "error": rental["error"] or "rent failed",
            }
        update_row(label, status="running", pod_name=rental["pod_name"], gpu=rental["gpu"])
        eval_args = make_lium_args(args, profile)
        try:
            result = lium_eval.run_remote_eval_job(
                pod_name=rental["pod_name"],
                model_input=model["model_input"],
                benchmark_csv=label,
                results_root=run_dir / "benchmarks" / slugify(label),
                remote_base=f"{args.remote_base.rstrip('/')}/{run_id}",
                remote_root_override=None,
                env=env,
                args=eval_args,
            )
            result["gpu"] = rental["gpu"]
            return extract_benchmark_row(result, label, profile)
        except Exception as exc:
            return {
                "name": label,
                "task": None,
                "fewshot": None,
                "status": "failed",
                "metric": None,
                "score": None,
                "batch_size": profile.batch_size,
                "fewshot_overrides": profile.fewshot_overrides,
                "log_samples": profile.log_samples,
                "pod_name": rental["pod_name"],
                "gpu": rental["gpu"],
                "local_run_dir": None,
                "summary_json": None,
                "standardized_results_json": None,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

    with cf.ThreadPoolExecutor(max_workers=len(selected_benchmarks)) as pool:
        futures = {pool.submit(worker, label): label for label in selected_benchmarks}
        for fut in cf.as_completed(futures):
            label = futures[fut]
            row = fut.result()
            with status_lock:
                row_by_name[label].update(row)
                partial_payload = build_payload(
                    run_id=run_id,
                    status="running",
                    started_at=started_at,
                    finished_at=None,
                    dashboard_url=dashboard_url,
                    model=model,
                    rows=list(row_by_name.values()),
                    paths=paths,
                    args=args,
                )
                write_status(args.results_root, run_dir, partial_payload, publisher)

    final_rows = list(row_by_name.values())
    final_status = "completed" if all(row.get("status") == "completed" for row in final_rows) else "partial"
    payload = build_payload(
        run_id=run_id,
        status=final_status,
        started_at=started_at,
        finished_at=utcnow_iso(),
        dashboard_url=dashboard_url,
        model=model,
        rows=final_rows,
        paths=paths,
        args=args,
    )
    public = write_status(args.results_root, run_dir, payload, publisher)
    append_jsonl(args.results_root / "history.jsonl", public)
    publisher.append_history(public)
    return payload


def load_state(path: Path) -> dict[str, Any]:
    return read_json(path, {"version": 1, "last_started_at": None, "last_finished_at": None, "last_run_id": None})


def save_state(path: Path, **updates: Any) -> None:
    state = load_state(path)
    state.update(updates)
    state["updated_at"] = utcnow_iso()
    write_json(path, state)


def parse_state_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def should_run_now(state: dict[str, Any], args: argparse.Namespace) -> bool:
    last_started_dt = parse_state_time(state.get("last_started_at"))
    if args.daily_time:
        slot = latest_scheduled_slot(utcnow(), args)
        if last_started_dt is None:
            return True
        return last_started_dt < slot
    if last_started_dt is None:
        return True
    return (utcnow() - last_started_dt).total_seconds() >= args.interval_hours * 3600




def next_run_hint(state: dict[str, Any], args: argparse.Namespace) -> str | None:
    if args.daily_time:
        slot = latest_scheduled_slot(utcnow(), args)
        last_started = parse_state_time(state.get("last_started_at"))
        if last_started is not None and last_started >= slot:
            next_slot = slot + timedelta(days=1)
        else:
            next_slot = slot
        return next_slot.strftime("%Y-%m-%dT%H:%M:%SZ")
    last_started = parse_state_time(state.get("last_started_at"))
    if last_started is None:
        return utcnow_iso()
    return (last_started + timedelta(hours=args.interval_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

def service_loop(args: argparse.Namespace) -> int:
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[service] teutonic-king-bench-lium started; schedule={args.daily_time} UTC{args.schedule_utc_offset}; "
        f"benchmarks={args.benchmarks}",
        flush=True,
    )
    last_wait_log: str | None = None
    while True:
        state = load_state(args.state_file)
        if should_run_now(state, args):
            started = utcnow_iso()
            save_state(args.state_file, last_started_at=started, status="running")
            try:
                payload = run_suite_once(args)
                save_state(
                    args.state_file,
                    last_finished_at=payload.get("finished_at") or utcnow_iso(),
                    last_run_id=payload.get("run_id"),
                    last_status=payload.get("status"),
                    status="idle",
                )
            except Exception as exc:
                err = {"at": utcnow_iso(), "error": repr(exc), "traceback": traceback.format_exc()}
                save_state(args.state_file, last_error=err, status="idle")
                print(json.dumps(err, indent=2), file=sys.stderr, flush=True)
                if args.run_once:
                    raise
        else:
            hint = next_run_hint(state, args) or "unknown"
            if hint != last_wait_log:
                print(f"[service] waiting; next_run_at={hint}; state={args.state_file}", flush=True)
                last_wait_log = hint
        if args.run_once:
            return 0
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PM2-friendly daily Lium benchmark runner for the current Teutonic king.")
    parser.add_argument("--run-once", action="store_true", help="Run one benchmark suite now, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch the current king and write JSON without renting pods.")
    parser.add_argument("--benchmarks", default=os.environ.get("TEUTONIC_KING_BENCH_BENCHMARKS", ",".join(DEFAULT_BENCHMARKS)), help="Comma-separated benchmark labels to run. Defaults to the six dashboard benchmarks.")
    parser.add_argument("--interval-hours", type=float, default=float(os.environ.get("TEUTONIC_KING_BENCH_INTERVAL_HOURS", DEFAULT_INTERVAL_HOURS)))
    parser.add_argument("--daily-time", default=os.environ.get("TEUTONIC_KING_BENCH_DAILY_TIME", "00:00"), help="Daily scheduled start time in the configured UTC offset, e.g. 00:00.")
    parser.add_argument("--schedule-utc-offset", default=os.environ.get("TEUTONIC_KING_BENCH_SCHEDULE_UTC_OFFSET", "+02:00"), help="Fixed offset for --daily-time, e.g. +02:00.")
    parser.add_argument("--poll-seconds", type=int, default=int(os.environ.get("TEUTONIC_KING_BENCH_POLL_SECS", DEFAULT_POLL_SECONDS)))
    parser.add_argument("--results-root", type=Path, default=Path(os.environ.get("TEUTONIC_KING_BENCH_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)))
    parser.add_argument("--state-file", type=Path, default=Path(os.environ.get("TEUTONIC_KING_BENCH_STATE_FILE", DEFAULT_RESULTS_ROOT / "state.json")))
    parser.add_argument("--registry", type=Path, default=Path(os.environ.get("TEUTONIC_KING_BENCH_REGISTRY", DEFAULT_REGISTRY)))
    parser.add_argument("--remote-base", default=os.environ.get("TEUTONIC_KING_BENCH_REMOTE_BASE", DEFAULT_REMOTE_BASE))
    parser.add_argument("--upload-hippius", action="store_true", default=env_bool("TEUTONIC_KING_BENCH_UPLOAD_HIPPIUS", True), help="Upload latest/history JSON to Hippius S3.")
    parser.add_argument("--no-upload-hippius", action="store_false", dest="upload_hippius", help="Disable Hippius S3 upload.")
    parser.add_argument("--hippius-endpoint", default=os.environ.get("TEUTONIC_HIPPIUS_ENDPOINT", DEFAULT_HIPPIUS_ENDPOINT))
    parser.add_argument("--hippius-bucket", default=os.environ.get("TEUTONIC_HIPPIUS_BUCKET", DEFAULT_HIPPIUS_BUCKET))
    parser.add_argument("--hippius-latest-key", default=os.environ.get("TEUTONIC_KING_BENCH_LATEST_KEY", DEFAULT_LATEST_KEY))
    parser.add_argument("--hippius-history-key", default=os.environ.get("TEUTONIC_KING_BENCH_HISTORY_KEY", DEFAULT_HISTORY_KEY))
    parser.add_argument("--dashboard-url", action="append", default=None, help="Dashboard mirror URL. Repeatable; defaults to Hippius mirrors.")
    parser.add_argument("--dashboard-timeout-s", type=int, default=int(os.environ.get("TEUTONIC_KING_BENCH_DASHBOARD_TIMEOUT_S", "15")))
    parser.add_argument("--doppler-project", default=os.environ.get("TEUTONIC_KING_BENCH_DOPPLER_PROJECT", "arbos"))
    parser.add_argument("--doppler-config", default=os.environ.get("TEUTONIC_KING_BENCH_DOPPLER_CONFIG", "dev"))
    parser.add_argument("--ttl", default=os.environ.get("TEUTONIC_KING_BENCH_TTL", "36h"))
    parser.add_argument("--template-id", default=os.environ.get("TEUTONIC_KING_BENCH_TEMPLATE_ID", ""))
    parser.add_argument("--wait-timeout-s", type=int, default=int(os.environ.get("TEUTONIC_KING_BENCH_WAIT_TIMEOUT_S", "1200")))
    parser.add_argument("--rent-timeout-s", type=int, default=int(os.environ.get("TEUTONIC_KING_BENCH_RENT_TIMEOUT_S", "180")), help="Maximum time to wait for lium up before probing the named pod directly.")
    parser.add_argument("--dtype", default=os.environ.get("TEUTONIC_KING_BENCH_DTYPE", "bfloat16"))
    parser.add_argument("--device", default=os.environ.get("TEUTONIC_KING_BENCH_DEVICE", "cuda:0"))
    parser.add_argument("--model-args-extra", default=os.environ.get("TEUTONIC_KING_BENCH_MODEL_ARGS_EXTRA", ""))
    parser.add_argument("--gen-kwargs", default=os.environ.get("TEUTONIC_KING_BENCH_GEN_KWARGS", ""))
    parser.add_argument("--fewshot-overrides", default=os.environ.get("TEUTONIC_KING_BENCH_FEWSHOT_OVERRIDES", ""))
    parser.add_argument("--limit", type=int, default=int(os.environ["TEUTONIC_KING_BENCH_LIMIT"]) if os.environ.get("TEUTONIC_KING_BENCH_LIMIT") else None)
    parser.add_argument("--apply-chat-template", action="store_true", default=env_bool("TEUTONIC_KING_BENCH_APPLY_CHAT_TEMPLATE", False))
    parser.add_argument("--resume", action="store_true", default=env_bool("TEUTONIC_KING_BENCH_RESUME", False))
    parser.add_argument("--skip-bootstrap", action="store_true", default=env_bool("TEUTONIC_KING_BENCH_SKIP_BOOTSTRAP", False))
    parser.add_argument("--keep-on-success", action="store_true", default=env_bool("TEUTONIC_KING_BENCH_KEEP_ON_SUCCESS", False))
    parser.add_argument("--delete-on-failure", action="store_true", default=env_bool("TEUTONIC_KING_BENCH_DELETE_ON_FAILURE", True))
    parser.add_argument("--log-samples", action="store_true", default=env_bool("TEUTONIC_KING_BENCH_LOG_SAMPLES", False))
    args = parser.parse_args()
    args.dashboard_url = args.dashboard_url or DASHBOARD_MIRRORS
    return args


def main() -> int:
    args = parse_args()
    return service_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
