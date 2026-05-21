#!/usr/bin/env python3
"""Distributed Hugging Face -> Hippius raw-file mirror over Lium pods.

The local coordinator discovers Hugging Face Parquet files, skips objects that
already exist in Hippius at the expected size, creates short-lived presigned PUT
URLs, and fans work out to Lium pods. Workers receive upload URLs only, not the
Hippius secret key. Rerun the coordinator to recover from failed pods or expired
URLs; completed objects are skipped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from huggingface_hub import HfApi

DEFAULT_REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_SOURCE_PREFIX = "data"
DEFAULT_BUCKET = "teutonic-sn3"
DEFAULT_ENDPOINT = "https://s3.hippius.com"
DEFAULT_REMOTE_DIR = "/root/hf_hippius_mirror"
DEFAULT_PRESIGN_SECONDS = 7 * 24 * 60 * 60

WORKER_SCRIPT = r'''
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import quote

import requests


def atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def hf_url(repo: str, revision: str, path: str) -> str:
    quoted = "/".join(quote(part) for part in path.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quoted}"


def request_with_retries(method: str, url: str, *, retries: int, **kwargs):
    last = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if 200 <= resp.status_code < 300:
                return resp
            last = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        except Exception as exc:
            last = exc
        if attempt < retries:
            time.sleep(min(60, 2 ** attempt))
    raise last


def mirror_one(item: dict, assignment: dict, staging: Path, retries: int) -> dict:
    source_path = item["source_path"]
    expected = int(item["size_bytes"])
    tmp = Path(tempfile.mkstemp(prefix="mirror_", suffix=".parquet", dir=staging)[1])
    started = time.time()
    try:
        headers = {}
        hf_token = assignment.get("hf_token")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        url = hf_url(assignment["source_repo"], assignment["source_revision"], source_path)
        with requests.get(url, headers=headers, stream=True, timeout=(30, 600)) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)

        actual = tmp.stat().st_size
        if actual != expected:
            raise RuntimeError(f"download size mismatch: expected={expected} actual={actual}")

        last = None
        for attempt in range(retries + 1):
            try:
                with tmp.open("rb") as f:
                    resp = requests.put(
                        item["put_url"],
                        data=f,
                        headers={
                            "Content-Length": str(expected),
                            "Content-Type": "application/octet-stream",
                        },
                        timeout=(30, 900),
                    )
                if 200 <= resp.status_code < 300:
                    break
                last = RuntimeError(f"PUT HTTP {resp.status_code}: {resp.text[:500]}")
            except Exception as exc:
                last = exc
            if attempt < retries:
                time.sleep(min(60, 2 ** attempt))
        else:
            raise last

        return {
            "source_path": source_path,
            "dest_key": item["dest_key"],
            "size_bytes": expected,
            "ok": True,
            "seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "source_path": source_path,
            "dest_key": item["dest_key"],
            "size_bytes": expected,
            "ok": False,
            "error": repr(exc),
            "seconds": round(time.time() - started, 3),
        }
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--staging", default="/tmp/hf_hippius_mirror")
    ap.add_argument("--parallelism", type=int, default=2)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    assignment_path = Path(args.assignment)
    assignment = json.loads(assignment_path.read_text())
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)

    status_path = assignment_path.with_name("status.json")
    events_path = assignment_path.with_name("status.jsonl")
    lock = threading.Lock()
    state = {
        "worker_id": assignment["worker_id"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": None,
        "total_files": len(assignment["files"]),
        "completed_files": 0,
        "failed_files": 0,
        "completed_bytes": 0,
        "failed": [],
        "done": False,
    }
    atomic_write_json(status_path, state)

    def record(result: dict):
        with lock:
            with events_path.open("a") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")
            if result["ok"]:
                state["completed_files"] += 1
                state["completed_bytes"] += int(result["size_bytes"])
            else:
                state["failed_files"] += 1
                state["failed"].append(result)
            state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            atomic_write_json(status_path, state)
            print(json.dumps(result, sort_keys=True), flush=True)

    with cf.ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as pool:
        futures = [
            pool.submit(mirror_one, item, assignment, staging, args.retries)
            for item in assignment["files"]
        ]
        for fut in cf.as_completed(futures):
            record(fut.result())

    state["done"] = True
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(status_path, state)
    if state["failed_files"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class SourceFile:
    path: str
    size: int


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True):
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    lium_failed_markers = ("Failed:", "Failed to upload to:", "Command failed")
    lium_failed = proc.returncode == 0 and any(
        marker in (proc.stdout or "") for marker in lium_failed_markers
    )
    if check and (proc.returncode != 0 or lium_failed):
        raise subprocess.CalledProcessError(proc.returncode or 1, cmd, output=proc.stdout)
    return proc


def run_retry(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    attempts: int = 3,
    sleep_s: float = 5.0,
):
    last: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run(cmd, env=env)
        except subprocess.CalledProcessError as exc:
            last = exc
            if attempt < attempts:
                print(
                    f"retrying command after failure ({attempt}/{attempts}): "
                    + " ".join(shlex.quote(c) for c in cmd),
                    flush=True,
                )
                time.sleep(sleep_s * attempt)
    raise last  # type: ignore[misc]


def capture(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.check_output(cmd, env=env, text=True)


def doppler_secret(name: str, *, project: str, config: str) -> str:
    try:
        return subprocess.check_output(
            ["doppler", "secrets", "get", name, "--plain", "-p", project, "-c", config],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def ensure_env_from_doppler(env: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    env = dict(env)
    if not env.get("LIUM_API_KEY"):
        v = doppler_secret("LIUM_API_KEY", project=args.doppler_project, config=args.doppler_config)
        if v:
            env["LIUM_API_KEY"] = v
    if not env.get("HF_TOKEN"):
        v = doppler_secret("HF_TOKEN", project=args.doppler_project, config=args.hf_doppler_config)
        if not v:
            v = doppler_secret("HUGGINGFACE_API_KEY", project=args.doppler_project, config=args.doppler_config)
        if v:
            env["HF_TOKEN"] = v
    if not env.get("TEUTONIC_DS_ACCESS_KEY"):
        v = env.get("HIPPIUS_ACCESS_KEY") or doppler_secret(
            "HIPPIUS_ACCESS_KEY", project=args.doppler_project, config=args.doppler_config
        ) or doppler_secret(
            "HIPPIUS_ACCESS_KEY_ID", project=args.doppler_project, config=args.doppler_config
        )
        if v:
            env["TEUTONIC_DS_ACCESS_KEY"] = v
    if not env.get("TEUTONIC_DS_SECRET_KEY"):
        v = env.get("HIPPIUS_SECRET_KEY") or env.get("HIPPIUS_SECRET_ACCESS_KEY") or doppler_secret(
            "HIPPIUS_SECRET_KEY", project=args.doppler_project, config=args.doppler_config
        ) or doppler_secret(
            "HIPPIUS_SECRET_ACCESS_KEY", project=args.doppler_project, config=args.doppler_config
        )
        if v:
            env["TEUTONIC_DS_SECRET_KEY"] = v
    env.setdefault("TEUTONIC_DS_ENDPOINT", args.endpoint)
    env.setdefault("TEUTONIC_DS_BUCKET", args.bucket)
    return env


def make_s3(env: dict[str, str], args: argparse.Namespace):
    return boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=env["TEUTONIC_DS_ACCESS_KEY"],
        aws_secret_access_key=env["TEUTONIC_DS_SECRET_KEY"],
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "adaptive"},
            s3={"addressing_style": "path"},
            connect_timeout=30,
            read_timeout=300,
        ),
    )


def default_dest_prefix(repo: str, source_prefix: str) -> str:
    return f"hf-mirrors/{repo}/{source_prefix}".strip("/")


def discover_files(args: argparse.Namespace, hf_token: str) -> list[SourceFile]:
    api = HfApi(token=hf_token or None)
    entries = api.list_repo_tree(
        args.repo,
        path_in_repo=args.source_prefix,
        repo_type="dataset",
        recursive=True,
        expand=True,
    )
    files = [
        SourceFile(path=x.path, size=int(getattr(x, "size", 0) or 0))
        for x in entries
        if getattr(x, "path", "").endswith(args.extension)
    ]
    files.sort(key=lambda f: f.path)
    if args.limit:
        files = files[: args.limit]
    return files


def object_matches(s3, bucket: str, key: str, expected_size: int) -> bool:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return int(head.get("ContentLength", -1)) == expected_size


def list_existing_sizes(s3, bucket: str, prefix: str) -> dict[str, int]:
    existing: dict[str, int] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []) or []:
            existing[obj["Key"]] = int(obj["Size"])
    return existing


def dest_key(dest_prefix: str, source_prefix: str, source_path: str) -> str:
    rel = source_path[len(source_prefix) :].lstrip("/") if source_path.startswith(source_prefix) else source_path
    return f"{dest_prefix.rstrip('/')}/{rel}"


def greedy_partitions(items: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    parts: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    totals = [0 for _ in range(n)]
    for item in sorted(items, key=lambda x: int(x["size_bytes"]), reverse=True):
        idx = min(range(n), key=lambda i: totals[i])
        parts[idx].append(item)
        totals[idx] += int(item["size_bytes"])
    for part in parts:
        part.sort(key=lambda x: x["source_path"])
    return parts


def active_lium_indices(env: dict[str, str]) -> list[str]:
    out = capture(["lium", "ps"], env=env)
    m = re.search(r"\((\d+)\s+active\)", out)
    if not m:
        raise RuntimeError("could not parse active pod count from `lium ps`; pass --targets explicitly")
    return [str(i) for i in range(1, int(m.group(1)) + 1)]


def create_pods(args: argparse.Namespace, env: dict[str, str]) -> list[str]:
    targets = []
    for idx in range(args.create_pods):
        name = f"{args.pod_name_prefix}-{int(time.time())}-{idx:02d}"
        cmd = ["lium", "up", "--name", name, "--ttl", args.ttl, "--yes"]
        if args.gpu:
            cmd += ["--gpu", args.gpu]
        if args.gpu_count:
            cmd += ["--count", str(args.gpu_count)]
        if args.country:
            cmd += ["--country", args.country]
        if args.template_id:
            cmd += ["--template_id", args.template_id]
        run(cmd, env=env)
        targets.append(name)
    return targets


def wait_for_targets(targets: list[str], env: dict[str, str], timeout_s: int):
    deadline = time.time() + timeout_s
    pending = set(targets)
    while pending and time.time() < deadline:
        for target in list(pending):
            proc = subprocess.run(
                ["lium", "exec", target, "python3 --version || python --version"],
                env=env,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode == 0:
                pending.remove(target)
                print(f"target ready: {target}", flush=True)
        if pending:
            time.sleep(20)
    if pending:
        raise RuntimeError(f"targets did not become ready: {sorted(pending)}")


def write_manifest(s3, args: argparse.Namespace, dest_prefix: str, source_files: list[SourceFile]):
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": args.repo,
        "source_repo_type": "dataset",
        "source_revision": args.revision,
        "source_prefix": args.source_prefix,
        "destination_bucket": args.bucket,
        "destination_prefix": dest_prefix,
        "total_files": len(source_files),
        "total_bytes": sum(f.size for f in source_files),
        "files": [
            {
                "source_path": f.path,
                "dest_key": dest_key(dest_prefix, args.source_prefix, f.path),
                "size_bytes": f.size,
            }
            for f in source_files
        ],
    }
    body = json.dumps(manifest, indent=2, sort_keys=True).encode()
    key = f"{dest_prefix.rstrip('/')}/_manifest.json"
    s3.put_object(Bucket=args.bucket, Key=key, Body=body, ContentType="application/json")
    print(f"uploaded manifest s3://{args.bucket}/{key} bytes={len(body)}", flush=True)


def upload_worker_files(target: str, remote_dir: str, worker: Path, assignment: Path, env: dict[str, str]):
    run_retry(["lium", "exec", target, f"mkdir -p {shlex.quote(remote_dir)}"], env=env)
    run_retry(["lium", "scp", target, str(worker), f"{remote_dir}/worker.py"], env=env)
    run_retry(["lium", "scp", target, str(assignment), f"{remote_dir}/assignment.json"], env=env)
    run_retry(
        [
            "lium",
            "exec",
            target,
            (
                f"test -s {shlex.quote(remote_dir)}/worker.py && "
                f"test -s {shlex.quote(remote_dir)}/assignment.json && "
                "(python3 -c \"import importlib.util,sys; "
                "sys.exit(0 if importlib.util.find_spec('requests') else 1)\" "
                "|| (python3 -m venv "
                f"{shlex.quote(remote_dir)}/.venv && "
                f"{shlex.quote(remote_dir)}/.venv/bin/python -m pip install --quiet requests))"
            ),
        ],
        env=env,
    )


def launch_workers(targets: list[str], remote_dir: str, args: argparse.Namespace, env: dict[str, str]) -> int:
    procs: list[tuple[str, subprocess.Popen, Any]] = []
    for target in targets:
        cmd = [
            "lium",
            "exec",
            target,
            (
                f"cd {shlex.quote(remote_dir)} && "
                "PY=python3; test -x .venv/bin/python && PY=.venv/bin/python; "
                f"$PY worker.py --assignment assignment.json "
                f"--parallelism {args.remote_parallelism} --retries {args.remote_retries}"
            ),
        ]
        print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", target)
        log_path = Path(args.local_log_dir) / f"worker-{safe_target}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w")
        proc = subprocess.Popen(cmd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
        procs.append((target, proc, log))

    failed = 0
    for target, proc, log in procs:
        rc = proc.wait()
        log.close()
        print(f"worker finished target={target} rc={rc}", flush=True)
        if rc != 0:
            failed += 1
    return failed


def verify_destination(s3, args: argparse.Namespace, dest_prefix: str, files: list[SourceFile]) -> tuple[int, int]:
    missing = 0
    bad = 0
    for f in files:
        key = dest_key(dest_prefix, args.source_prefix, f.path)
        try:
            head = s3.head_object(Bucket=args.bucket, Key=key)
            if int(head.get("ContentLength", -1)) != f.size:
                bad += 1
        except ClientError:
            missing += 1
    return missing, bad


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Mirror Hugging Face dataset Parquet files to Hippius using Lium pods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--extension", default=".parquet")
    ap.add_argument("--dest-prefix", default=None)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--targets", default=None, help="Comma-separated Lium targets, or all-active.")
    ap.add_argument("--create-pods", type=int, default=0)
    ap.add_argument("--gpu", default=None, help="Lium --gpu filter when creating pods.")
    ap.add_argument("--gpu-count", type=int, default=0, help="Lium --count when creating pods.")
    ap.add_argument("--country", default=None)
    ap.add_argument("--template-id", default=None)
    ap.add_argument("--ttl", default="6h")
    ap.add_argument("--pod-name-prefix", default="fineweb-edu-mirror")
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    ap.add_argument("--remote-parallelism", type=int, default=2)
    ap.add_argument("--remote-retries", type=int, default=3)
    ap.add_argument("--presign-seconds", type=int, default=DEFAULT_PRESIGN_SECONDS)
    ap.add_argument("--wait-timeout-s", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0, help="For smoke tests; 0 means all matching files.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--pass-hf-token-to-workers", action="store_true")
    ap.add_argument("--doppler-project", default="arbos")
    ap.add_argument("--doppler-config", default="dev")
    ap.add_argument("--hf-doppler-config", default="prd")
    ap.add_argument("--local-log-dir", default="/tmp/fineweb_edu_lium_logs")
    ap.add_argument("--work-dir", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    env = ensure_env_from_doppler(os.environ, args)
    dest_prefix = args.dest_prefix or default_dest_prefix(args.repo, args.source_prefix)

    files = discover_files(args, env.get("HF_TOKEN", ""))
    total_bytes = sum(f.size for f in files)
    print(
        f"source repo={args.repo} prefix={args.source_prefix} files={len(files)} "
        f"bytes={total_bytes} dest=s3://{args.bucket}/{dest_prefix}",
        flush=True,
    )

    if args.dry_run:
        print("dry-run: not touching Lium or Hippius", flush=True)
        return

    for var in ["TEUTONIC_DS_ACCESS_KEY", "TEUTONIC_DS_SECRET_KEY"]:
        if not env.get(var):
            raise SystemExit(f"missing {var}; export it or add it to Doppler")

    s3 = make_s3(env, args)
    existing_sizes = list_existing_sizes(s3, args.bucket, dest_prefix)
    print(f"listed_existing={len(existing_sizes)}", flush=True)

    remaining = []
    skipped = 0
    for f in files:
        key = dest_key(dest_prefix, args.source_prefix, f.path)
        if existing_sizes.get(key) == f.size:
            skipped += 1
            continue
        put_url = s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": args.bucket, "Key": key, "ContentType": "application/octet-stream"},
            ExpiresIn=args.presign_seconds,
        )
        remaining.append(
            {"source_path": f.path, "dest_key": key, "size_bytes": f.size, "put_url": put_url}
        )

    print(f"skip_existing={skipped} remaining={len(remaining)}", flush=True)
    if not remaining:
        if args.write_manifest:
            write_manifest(s3, args, dest_prefix, files)
        return

    targets: list[str] = []
    if args.create_pods:
        targets.extend(create_pods(args, env))
        wait_for_targets(targets, env, args.wait_timeout_s)
    if args.targets:
        if args.targets == "all-active":
            targets.extend(active_lium_indices(env))
        else:
            targets.extend([t.strip() for t in args.targets.split(",") if t.strip()])
    targets = list(dict.fromkeys(targets))
    if not targets:
        raise SystemExit("no targets: pass --targets, --targets all-active, or --create-pods N")

    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="lium-hf-mirror-"))
    work_root.mkdir(parents=True, exist_ok=True)
    worker_path = work_root / "worker.py"
    worker_path.write_text(WORKER_SCRIPT)
    partitions = greedy_partitions(remaining, len(targets))

    assignments: list[Path] = []
    for idx, (target, part) in enumerate(zip(targets, partitions)):
        assignment = {
            "worker_id": idx,
            "target": target,
            "source_repo": args.repo,
            "source_revision": args.revision,
            "source_prefix": args.source_prefix,
            "destination_bucket": args.bucket,
            "destination_prefix": dest_prefix,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hf_token": env.get("HF_TOKEN", "") if args.pass_hf_token_to_workers else "",
            "files": part,
        }
        path = work_root / f"assignment-{idx:03d}.json"
        path.write_text(json.dumps(assignment, indent=2, sort_keys=True))
        assignments.append(path)
        print(
            f"assignment target={target} files={len(part)} "
            f"bytes={sum(int(x['size_bytes']) for x in part)}",
            flush=True,
        )

    # Lium's SSH/SCP layer is more reliable with modest setup concurrency.
    # Worker transfer parallelism still controls data-plane bandwidth.
    with cf.ThreadPoolExecutor(max_workers=min(4, len(targets))) as pool:
        futs = [
            pool.submit(upload_worker_files, target, args.remote_dir, worker_path, assignment, env)
            for target, assignment in zip(targets, assignments)
        ]
        for fut in cf.as_completed(futs):
            fut.result()

    failed_workers = launch_workers(targets, args.remote_dir, args, env)
    missing, bad = verify_destination(s3, args, dest_prefix, files)
    print(f"verify missing={missing} size_mismatch={bad} failed_workers={failed_workers}", flush=True)

    if args.write_manifest and missing == 0 and bad == 0:
        write_manifest(s3, args, dest_prefix, files)
    if failed_workers or missing or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
