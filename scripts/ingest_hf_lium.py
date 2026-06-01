#!/usr/bin/env python3
'''Distributed Hugging Face parquet ingest across Lium pods.

This coordinator is intentionally scratch-light: it only stores file lists,
logs, and manifests locally. Remote pods do the parquet download/tokenize/upload
work. For the ClimbMix path, each source parquet becomes exactly one .npy file.
'''
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_king_benchmarks_lium as lium_eval  # noqa: E402
from scripts import ingest_hf  # noqa: E402

DEFAULT_RESULTS_ROOT = ROOT / "runs" / "lium-ingest"
DEFAULT_REGISTRY = ROOT / "runs" / "lium-rentals" / "registry.json"
DEFAULT_REMOTE_BASE = "/root/teutonic-lium-ingest"
DEFAULT_DATASET = "karpathy/climbmix-400b-shuffle"
DEFAULT_TOKENIZER = "unconst/Teutonic-XXIV"
DEFAULT_DEST_PREFIX = "dataset/climbmix-400b-teutonic-xxiv"
SCHEMA_VERSION = "teutonic-lium-ingest.v1"
REGISTRY_LOCK = threading.Lock()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-") or "run"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def append_log(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(body)
        if not body.endswith("\n"):
            fh.write("\n")


def parse_gpu_specs(raw: str) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" in part:
            gpu, count_s = part.split(":", 1)
            count = int(count_s)
        else:
            gpu, count = part, 1
        if count <= 0:
            raise SystemExit(f"invalid GPU count in {item!r}")
        specs.append((gpu.strip(), count))
    if not specs:
        raise SystemExit("no GPU specs provided")
    return specs

def parse_existing_pods(raw: str | None) -> list[str]:
    if not raw:
        return []
    pods = [part.strip() for part in raw.split(",") if part.strip()]
    if len(set(pods)) != len(pods):
        raise SystemExit("--existing-pods contains duplicate pod names")
    return pods


def partition_round_robin(files: list[str], n_parts: int) -> list[list[str]]:
    parts = [[] for _ in range(n_parts)]
    for idx, file_path in enumerate(files):
        parts[idx % n_parts].append(file_path)
    return parts


def remote_paths(remote_root: str) -> dict[str, str]:
    return {
        "root": remote_root,
        "scripts_dir": f"{remote_root}/scripts",
        "ingest_script": f"{remote_root}/scripts/ingest_hf.py",
        "s3_transfer": f"{remote_root}/s3_transfer.py",
        "file_list": f"{remote_root}/files.json",
        "bootstrap": f"{remote_root}/bootstrap.sh",
        "runner": f"{remote_root}/run_ingest.sh",
        "logs_dir": f"{remote_root}/logs",
        "bootstrap_log": f"{remote_root}/logs/bootstrap.log",
        "runner_log": f"{remote_root}/logs/run_ingest.log",
        "venv": f"{remote_root}/.venv",
    }


def shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def make_bootstrap_script(remote_root: str) -> str:
    paths = remote_paths(remote_root)
    return f'''#!/usr/bin/env bash
set -euo pipefail

retry() {{
  local attempt=1
  local max_attempts=4
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      return 1
    fi
    sleep $((attempt * 10))
    attempt=$((attempt + 1))
  done
}}

export DEBIAN_FRONTEND=noninteractive
mkdir -p {shlex.quote(remote_root)} {shlex.quote(paths["scripts_dir"])} {shlex.quote(paths["logs_dir"])}
retry apt-get update
retry apt-get install -y \
  build-essential \
  ca-certificates \
  gcc \
  g++ \
  python3 \
  python3-dev \
  python3-venv

if [ ! -x {shlex.quote(paths["venv"] + "/bin/python")} ]; then
  python3 -m venv {shlex.quote(paths["venv"])}
fi

PY={shlex.quote(paths["venv"] + "/bin/python")}
PIP={shlex.quote(paths["venv"] + "/bin/pip")}

retry $PY -m pip install --upgrade pip setuptools wheel
retry $PIP install \
  boto3 \
  botocore \
  huggingface-hub \
  minio \
  numpy \
  protobuf \
  pyarrow \
  sentencepiece \
  tiktoken \
  tokenizers \
  transformers

$PY -c 'import boto3, huggingface_hub, minio, numpy, pyarrow, tokenizers, transformers; print("ingest-bootstrap-ok")'
'''


def make_run_script(
    *,
    remote_root: str,
    dataset: str,
    tokenizer: str,
    dest_prefix: str,
    run_id: str,
    part_id: str,
    args: argparse.Namespace,
    remote_env: dict[str, str],
    file_list_path: str | None = None,
    scratch_dir: str | None = None,
    progress_dir: str | None = None,
    workers: int | None = None,
    max_inflight_files: int | None = None,
) -> str:
    paths = remote_paths(remote_root)
    file_list_path = file_list_path or paths["file_list"]
    scratch_dir = scratch_dir or args.remote_scratch_dir
    progress_dir = progress_dir or args.remote_progress_dir
    workers = args.remote_workers if workers is None else workers
    max_inflight_files = args.remote_max_inflight_files if max_inflight_files is None else max_inflight_files
    cmd = [
        paths["venv"] + "/bin/python",
        paths["ingest_script"],
        "--dataset", dataset,
        "--tokenizer", tokenizer,
        "--dest-prefix", dest_prefix,
        "--file-list", file_list_path,
        "--one-npy-per-parquet",
        "--part-manifest-only",
        "--distributed-run-id", run_id,
        "--part-id", part_id,
        "--scratch-dir", scratch_dir,
        "--progress-dir", progress_dir,
        "--text-column", args.text_column,
        "--workers", str(workers),
        "--min-free-gb", str(args.remote_min_free_gb),
        "--worker-disk-gb", str(args.remote_worker_disk_gb),
        "--max-inflight-files", str(max_inflight_files),
        "--auto-max-workers", str(args.remote_auto_max_workers),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    exports = "\n".join(shell_export(name, value) for name, value in remote_env.items())
    command = " ".join(shlex.quote(part) for part in cmd)
    return f'''#!/usr/bin/env bash
set -euo pipefail
{exports}
mkdir -p {shlex.quote(scratch_dir)} {shlex.quote(progress_dir)} {shlex.quote(paths["logs_dir"])}
cd {shlex.quote(remote_root)}
{command}
'''


def resolve_secret(names: list[str], *, project: str, config: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    for name in names:
        value = lium_eval.doppler_secret(name, project=project, config=config).strip()
        if value:
            return value
    return ""


def build_remote_env(args: argparse.Namespace) -> dict[str, str]:
    hf_token = resolve_secret(["HF_TOKEN"], project=args.doppler_project, config=args.hf_doppler_config)
    access_key = resolve_secret(
        ["HIPPIUS_ACCESS_KEY", "TEUTONIC_DS_ACCESS_KEY", "TEUTONIC_HIPPIUS_ACCESS_KEY"],
        project=args.doppler_project,
        config=args.s3_doppler_config,
    )
    secret_key = resolve_secret(
        ["HIPPIUS_SECRET_KEY", "TEUTONIC_DS_SECRET_KEY", "TEUTONIC_HIPPIUS_SECRET_KEY"],
        project=args.doppler_project,
        config=args.s3_doppler_config,
    )
    if not access_key or not secret_key:
        raise SystemExit("missing Hippius credentials for remote ingest")
    env = {
        "TEUTONIC_DS_ENDPOINT": args.hippius_endpoint,
        "TEUTONIC_DS_BUCKET": args.hippius_bucket,
        "TEUTONIC_DS_ACCESS_KEY": access_key,
        "TEUTONIC_DS_SECRET_KEY": secret_key,
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return env


def make_s3_client(args: argparse.Namespace, remote_env: dict[str, str]):
    return boto3.client(
        "s3",
        endpoint_url=remote_env["TEUTONIC_DS_ENDPOINT"],
        aws_access_key_id=remote_env["TEUTONIC_DS_ACCESS_KEY"],
        aws_secret_access_key=remote_env["TEUTONIC_DS_SECRET_KEY"],
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def put_manifest_with_client(client, bucket: str, dest_prefix: str, manifest: dict[str, Any]) -> None:
    client.put_object(
        Bucket=bucket,
        Key=f"{dest_prefix.rstrip('/')}/manifest.json",
        Body=json.dumps(manifest, indent=2, sort_keys=False).encode(),
        ContentType="application/json",
    )


def run_lium_up_with_timeout(cmd: list[str], env: dict[str, str], timeout_s: int) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
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
        print(f"[warn] lium up timed out after {timeout_s}s; probing pod reachability", flush=True)
        return
    output = proc.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    failed = proc.returncode != 0 or lium_eval.lium_output_failed(cmd, output)
    if failed:
        raise subprocess.CalledProcessError(proc.returncode or 1, cmd, output=output)
    if "No executors available" in output or "No pods match targets" in output:
        raise subprocess.CalledProcessError(1, cmd, output=output)


def part_index(part_id: str) -> int:
    try:
        return int(part_id.rsplit("-", 1)[1])
    except Exception:
        return 0


def run_logged_strict(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = lium_eval.run(cmd, env=env, check=False, capture_output=True)
    output = proc.stdout or ""
    if log_path is not None:
        append_log(log_path, "$ " + " ".join(shlex.quote(part) for part in cmd))
        append_log(log_path, output)
        append_log(log_path, f"[returncode] {proc.returncode}")
    if proc.returncode != 0 or lium_eval.lium_output_failed(cmd, output):
        raise subprocess.CalledProcessError(proc.returncode or 1, cmd, output=output)
    return proc

def registry_append(args: argparse.Namespace, row: dict[str, Any]) -> None:
    if args.no_registry_updates:
        return
    with REGISTRY_LOCK:
        lium_eval.append_rental(args.registry, row)


def registry_update(args: argparse.Namespace, pod_name: str, **updates: Any) -> None:
    if args.no_registry_updates:
        return
    with REGISTRY_LOCK:
        lium_eval.update_rental(args.registry, pod_name, **updates)


def terminate_pod_for_args(
    pod_name: str,
    env: dict[str, str],
    args: argparse.Namespace,
    *,
    reason: str,
) -> None:
    lium_eval.run(["lium", "rm", pod_name], env=env)
    registry_update(args, pod_name, terminated_at=utcnow_iso(), termination_reason=reason, status="terminated")


def claim_existing_pod_for_part(
    *,
    part_id: str,
    run_id: str,
    pod_name: str,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    try:
        lium_eval.wait_for_pod(pod_name, env, args.wait_timeout_s)
        registry_append(
            args,
            {
                "created_at": utcnow_iso(),
                "status": "acquired_existing",
                "pod_name": pod_name,
                "owner": "ingest_hf_lium",
                "ingest_run_id": run_id,
                "part_id": part_id,
                "ttl": args.ttl,
                "filters": {"existing_pod": True},
            },
        )
        return {"pod_name": pod_name, "gpu": "existing", "count": None, "error": None, "existing": True}
    except Exception as exc:
        return {"pod_name": None, "gpu": "existing", "count": None, "error": f"{pod_name}: {exc!r}", "existing": True}



def rent_pod_for_part(
    *,
    part_id: str,
    run_id: str,
    gpu_specs: list[tuple[str, int]],
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    name = f"ingest-{slugify(run_id)[-13:]}-{part_id}"
    errors: list[str] = []
    if args.rent_stagger_s > 0:
        time.sleep(part_index(part_id) * args.rent_stagger_s)
    for gpu, count in gpu_specs:
        try:
            cmd = ["lium", "up", "--name", name, "--ttl", args.ttl, "--yes", "--gpu", gpu, "--count", str(count)]
            if args.template_id:
                cmd += ["--template_id", args.template_id]
            run_lium_up_with_timeout(cmd, env, args.rent_timeout_s)
            lium_eval.wait_for_pod(name, env, args.wait_timeout_s)
            registry_append(
                args,
                {
                    "created_at": utcnow_iso(),
                    "status": "running",
                    "pod_name": name,
                    "owner": "ingest_hf_lium",
                    "ingest_run_id": run_id,
                    "part_id": part_id,
                    "ttl": args.ttl,
                    "filters": {"gpu": gpu, "count": count, "template_id": args.template_id},
                },
            )
            return {"pod_name": name, "gpu": gpu, "count": count, "error": None}
        except Exception as exc:
            errors.append(f"{gpu}:{count}: {exc!r}")
            try:
                terminate_pod_for_args(name, env, args, reason="rent_failed")
            except Exception:
                pass
    return {"pod_name": None, "gpu": None, "count": None, "error": "; ".join(errors)}


def run_lium_retry(
    cmd: list[str],
    *,
    env: dict[str, str],
    attempts: int = 4,
    delay_s: int = 15,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            lium_eval.run(cmd, env=env)
            return
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                f"[warn] lium command failed attempt {attempt}/{attempts}: {exc!r}; retrying in {delay_s}s",
                flush=True,
            )
            time.sleep(delay_s)
    assert last_exc is not None
    raise last_exc


def stage_remote_files(
    *,
    pod_name: str,
    env: dict[str, str],
    local_part_dir: Path,
    remote_root: str,
    file_list: list[str],
    dataset: str,
    tokenizer: str,
    dest_prefix: str,
    run_id: str,
    part_id: str,
    remote_env: dict[str, str],
    args: argparse.Namespace,
) -> None:
    paths = remote_paths(remote_root)
    file_list_local = local_part_dir / "files.json"
    bootstrap_local = local_part_dir / "bootstrap.sh"
    runner_local = local_part_dir / "run_ingest.sh"
    ingest_local = ROOT / "scripts" / "ingest_hf.py"
    transfer_local = ROOT / "s3_transfer.py"

    write_json(file_list_local, {"files": file_list})
    write_text(bootstrap_local, make_bootstrap_script(remote_root))
    write_text(
        runner_local,
        make_run_script(
            remote_root=remote_root,
            dataset=dataset,
            tokenizer=tokenizer,
            dest_prefix=dest_prefix,
            run_id=run_id,
            part_id=part_id,
            args=args,
            remote_env=remote_env,
        ),
    )

    run_lium_retry(["lium", "exec", pod_name, f"mkdir -p {shlex.quote(paths['scripts_dir'])} {shlex.quote(paths['logs_dir'])}"], env=env)
    run_lium_retry(["lium", "scp", pod_name, str(file_list_local), paths["file_list"]], env=env)
    run_lium_retry(["lium", "scp", pod_name, str(bootstrap_local), paths["bootstrap"]], env=env)
    run_lium_retry(["lium", "scp", pod_name, str(runner_local), paths["runner"]], env=env)
    run_lium_retry(["lium", "scp", pod_name, str(ingest_local), paths["ingest_script"]], env=env)
    run_lium_retry(["lium", "scp", pod_name, str(transfer_local), paths["s3_transfer"]], env=env)
    run_lium_retry(
        ["lium", "exec", pod_name, f"chmod +x {shlex.quote(paths['bootstrap'])} {shlex.quote(paths['runner'])} {shlex.quote(paths['ingest_script'])}"],
        env=env,
    )


def run_repair_attempt(
    *,
    pod_name: str,
    env: dict[str, str],
    local_part_dir: Path,
    remote_root: str,
    repair_files: list[str],
    dataset: str,
    tokenizer: str,
    dest_prefix: str,
    run_id: str,
    base_part_id: str,
    attempt: int,
    remote_env: dict[str, str],
    args: argparse.Namespace,
    log_path: Path,
) -> str:
    paths = remote_paths(remote_root)
    repair_part_id = f"{base_part_id}-repair-{attempt:02d}"
    repair_name = f"repair-{attempt:02d}"
    local_file_list = local_part_dir / f"{repair_name}-files.json"
    local_runner = local_part_dir / f"{repair_name}.sh"
    remote_file_list = f"{remote_root}/{repair_name}-files.json"
    remote_runner = f"{remote_root}/{repair_name}.sh"
    remote_log = f"{paths['logs_dir']}/{repair_name}.log"
    scratch_dir = f"{args.remote_scratch_dir.rstrip('/')}-{base_part_id}-{repair_name}"
    progress_dir = f"{args.remote_progress_dir.rstrip('/')}-{base_part_id}-{repair_name}"
    workers = args.repair_workers or min(args.remote_workers, 4)
    max_inflight = args.repair_max_inflight_files or workers

    write_json(local_file_list, {"files": repair_files})
    write_text(
        local_runner,
        make_run_script(
            remote_root=remote_root,
            dataset=dataset,
            tokenizer=tokenizer,
            dest_prefix=dest_prefix,
            run_id=run_id,
            part_id=repair_part_id,
            args=args,
            remote_env=remote_env,
            file_list_path=remote_file_list,
            scratch_dir=scratch_dir,
            progress_dir=progress_dir,
            workers=workers,
            max_inflight_files=max_inflight,
        ),
    )
    append_log(log_path, f"[repair] attempt={attempt} part_id={repair_part_id} files={len(repair_files)}")
    run_lium_retry(["lium", "scp", pod_name, str(local_file_list), remote_file_list], env=env)
    run_lium_retry(["lium", "scp", pod_name, str(local_runner), remote_runner], env=env)
    run_lium_retry(["lium", "exec", pod_name, f"chmod +x {shlex.quote(remote_runner)}"], env=env)
    run_logged_strict(
        ["lium", "exec", pod_name, f"bash {shlex.quote(remote_runner)} > {shlex.quote(remote_log)} 2>&1"],
        env=env,
        log_path=log_path,
    )
    return ingest_hf.part_manifest_key(dest_prefix, run_id, repair_part_id)


def copy_back_logs(pod_name: str, env: dict[str, str], remote_root: str, local_part_dir: Path) -> None:
    local_part_dir.mkdir(parents=True, exist_ok=True)
    tar_name = f"/tmp/{slugify(posixpath.basename(remote_root))}-logs.tgz"
    local_tar = local_part_dir / "remote-logs.tgz"
    lium_eval.run(
        ["lium", "exec", pod_name, f"tar --ignore-failed-read -C {shlex.quote(remote_root)} -czf {shlex.quote(tar_name)} logs files.json run_ingest.sh bootstrap.sh repair-*.json repair-*.sh"],
        env=env,
        check=False,
    )
    try:
        lium_eval.run(["lium", "scp", "--download", pod_name, tar_name, str(local_tar)], env=env, check=False)
        if local_tar.exists():
            with tarfile.open(local_tar, "r:gz") as archive:
                archive.extractall(local_part_dir / "remote-artifacts")
    finally:
        lium_eval.run(["lium", "exec", pod_name, f"rm -f {shlex.quote(tar_name)}"], env=env, check=False)


def get_json_key(client, bucket: str, key: str) -> dict[str, Any]:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def list_s3_keys(client, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
    while True:
        response = client.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            keys.add(obj["Key"])
        if not response.get("IsTruncated"):
            return keys
        kwargs["ContinuationToken"] = response["NextContinuationToken"]


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def part_manifest_coverage(
    *,
    client,
    bucket: str,
    manifest_keys: list[str],
    part_files: list[str],
    dataset: str,
    tokenizer: str,
) -> dict[str, Any]:
    expected = set(part_files)
    seen: set[str] = set()
    failed_files: list[str] = []
    summaries: list[dict[str, Any]] = []
    shard_count = 0
    total_tokens = 0
    for key in manifest_keys:
        manifest = get_json_key(client, bucket, key)
        if manifest.get("source") != dataset:
            raise RuntimeError(f"part source mismatch in {key}: {manifest.get('source')!r}")
        if manifest.get("tokenizer") != tokenizer:
            raise RuntimeError(f"part tokenizer mismatch in {key}: {manifest.get('tokenizer')!r}")
        shards = manifest.get("shards", [])
        summaries.append(
            {
                "key": key,
                "status": manifest.get("status"),
                "total_shards": len(shards),
                "total_tokens": manifest.get("total_tokens"),
                "failed_files": len(manifest.get("failed_files") or []),
            }
        )
        failed_files.extend(fp for fp in manifest.get("failed_files") or [] if fp in expected)
        for shard in shards:
            source_file = shard.get("source_file")
            if source_file not in expected:
                raise RuntimeError(f"unexpected source file in {key}: {source_file!r}")
            if source_file in seen:
                raise RuntimeError(f"duplicate source file across part manifests: {source_file}")
            seen.add(source_file)
            shard_count += 1
            total_tokens += int(shard.get("n_tokens", 0))
    missing_files = [fp for fp in part_files if fp not in seen]
    pending_failed_files = [fp for fp in failed_files if fp not in seen]
    repair_files = dedupe_keep_order(pending_failed_files + missing_files)
    return {
        "complete": not repair_files,
        "missing_files": missing_files,
        "failed_files": dedupe_keep_order(pending_failed_files),
        "repair_files": repair_files,
        "manifest_summaries": summaries,
        "total_shards": shard_count,
        "total_tokens": total_tokens,
    }


def merge_part_manifests(
    *,
    run_id: str,
    part_ids: list[str],
    expected_files: list[str],
    dataset: str,
    tokenizer: str,
    dest_prefix: str,
    args: argparse.Namespace,
    remote_env: dict[str, str],
) -> dict[str, Any]:
    client = make_s3_client(args, remote_env)
    bucket = remote_env["TEUTONIC_DS_BUCKET"]
    expected = set(expected_files)
    order = {file_path: idx for idx, file_path in enumerate(expected_files)}
    part_keys = [ingest_hf.part_manifest_key(dest_prefix, run_id, part_id) for part_id in part_ids]
    shards: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_tokens = 0
    part_summaries: list[dict[str, Any]] = []
    for key in part_keys:
        manifest = get_json_key(client, bucket, key)
        if manifest.get("source") != dataset:
            raise RuntimeError(f"part source mismatch in {key}: {manifest.get('source')!r}")
        if manifest.get("tokenizer") != tokenizer:
            raise RuntimeError(f"part tokenizer mismatch in {key}: {manifest.get('tokenizer')!r}")
        manifest_shards = manifest.get("shards", [])
        part_summaries.append(
            {
                "key": key,
                "status": manifest.get("status"),
                "total_shards": len(manifest_shards),
                "total_tokens": manifest.get("total_tokens"),
                "failed_files": len(manifest.get("failed_files") or []),
            }
        )
        for shard in manifest_shards:
            source_file = shard.get("source_file")
            if source_file not in expected:
                raise RuntimeError(f"unexpected source file in {key}: {source_file!r}")
            if source_file in seen:
                raise RuntimeError(f"duplicate source file across parts: {source_file}")
            seen.add(source_file)
            shards.append(dict(shard))
            total_tokens += int(shard.get("n_tokens", 0))
    missing = [fp for fp in expected_files if fp not in seen]
    if missing:
        raise RuntimeError(f"missing {len(missing)} source files; first missing: {missing[:5]}")
    shards.sort(key=lambda shard: order[shard["source_file"]])
    expected_keys = {shard["key"] for shard in shards}
    uploaded_keys = list_s3_keys(client, bucket, f"{dest_prefix.rstrip('/')}/shards/")
    missing_objects = sorted(expected_keys - uploaded_keys)
    if missing_objects:
        raise RuntimeError(f"missing {len(missing_objects)} uploaded shard objects; first missing: {missing_objects[:5]}")
    now = utcnow_iso()
    final_manifest = {
        "version": "v4-one-npy-per-parquet-merged",
        "schema_version": SCHEMA_VERSION,
        "created": now,
        "updated": now,
        "status": "completed",
        "tokenizer": tokenizer,
        "dtype": "uint32",
        "source": dataset,
        "tokenization_mode": "one_npy_per_parquet",
        "distributed_run_id": run_id,
        "total_tokens": total_tokens,
        "total_shards": len(shards),
        "source_files": len(expected_files),
        "shard_prefix": f"{dest_prefix.rstrip('/')}/shards/",
        "part_manifests": part_keys,
        "part_summaries": part_summaries,
        "shards": shards,
    }
    put_manifest_with_client(client, bucket, dest_prefix.rstrip("/"), final_manifest)
    return final_manifest


def build_status(
    *,
    run_id: str,
    status: str,
    dataset: str,
    tokenizer: str,
    dest_prefix: str,
    rows: list[dict[str, Any]],
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "updated_at": utcnow_iso(),
        "dataset": dataset,
        "tokenizer": tokenizer,
        "dest_prefix": dest_prefix,
        "results_dir": str(run_dir),
        "registry": str(args.registry),
        "parts": rows,
        "totals": {
            "parts": len(rows),
            "completed": sum(1 for row in rows if row.get("status") == "completed"),
            "failed": sum(1 for row in rows if row.get("status") == "failed"),
            "running": sum(1 for row in rows if row.get("status") in {"renting", "running", "bootstrapping"}),
            "files": sum(int(row.get("file_count", 0)) for row in rows),
        },
    }


def discover_files(dataset: str, args: argparse.Namespace) -> list[str]:
    if args.include_prefixes:
        include_prefixes = tuple(p.strip() for p in args.include_prefixes.split(",") if p.strip())
    else:
        include_prefixes = ingest_hf.default_include_prefixes_for_dataset(dataset)
    files = ingest_hf.discover_parquet_files(
        dataset,
        os.environ.get("HF_TOKEN", ""),
        langs=args.langs.split(",") if args.langs else None,
        include_prefixes=include_prefixes,
    )
    return [fp for _config, fp in files]


def run_part(
    *,
    part_idx: int,
    part_files: list[str],
    run_id: str,
    dataset: str,
    tokenizer: str,
    dest_prefix: str,
    rental: dict[str, Any],
    lium_env: dict[str, str],
    remote_env: dict[str, str],
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    part_id = f"part-{part_idx:03d}"
    local_part_dir = run_dir / "parts" / part_id
    remote_root = f"{args.remote_base.rstrip('/')}/{run_id}/{part_id}"
    row: dict[str, Any] = {
        "part_id": part_id,
        "status": "pending",
        "file_count": len(part_files),
        "pod_name": None,
        "gpu": None,
        "gpu_count": None,
        "manifest_key": ingest_hf.part_manifest_key(dest_prefix, run_id, part_id),
        "local_dir": str(local_part_dir),
        "remote_root": remote_root,
        "error": None,
    }
    if args.dry_run:
        write_json(local_part_dir / "files.json", {"files": part_files})
        row["status"] = "dry_run"
        return row
    if not rental["pod_name"]:
        row.update(status="failed", error=rental["error"] or "rent failed")
        return row
    row.update(status="running", pod_name=rental["pod_name"], gpu=rental["gpu"], gpu_count=rental["count"])
    local_command_log = local_part_dir / "local-command.log"
    registry_update(args, rental["pod_name"], status="running_ingest", local_run_dir=str(local_part_dir), remote_root=remote_root)
    try:
        stage_remote_files(
            pod_name=rental["pod_name"],
            env=lium_env,
            local_part_dir=local_part_dir,
            remote_root=remote_root,
            file_list=part_files,
            dataset=dataset,
            tokenizer=tokenizer,
            dest_prefix=dest_prefix,
            run_id=run_id,
            part_id=part_id,
            remote_env=remote_env,
            args=args,
        )
        if not args.skip_bootstrap:
            row["status"] = "bootstrapping"
            run_logged_strict(
                ["lium", "exec", rental["pod_name"], f"bash {shlex.quote(remote_paths(remote_root)['bootstrap'])} > {shlex.quote(remote_paths(remote_root)['bootstrap_log'])} 2>&1"],
                env=lium_env,
                log_path=local_command_log,
            )
        row["status"] = "running"
        run_logged_strict(
            ["lium", "exec", rental["pod_name"], f"bash {shlex.quote(remote_paths(remote_root)['runner'])} > {shlex.quote(remote_paths(remote_root)['runner_log'])} 2>&1"],
            env=lium_env,
            log_path=local_command_log,
        )

        client = make_s3_client(args, remote_env)
        bucket = remote_env["TEUTONIC_DS_BUCKET"]
        manifest_keys = [row["manifest_key"]]
        repair_manifest_keys: list[str] = []
        for attempt in range(args.repair_attempts + 1):
            coverage = part_manifest_coverage(
                client=client,
                bucket=bucket,
                manifest_keys=manifest_keys,
                part_files=part_files,
                dataset=dataset,
                tokenizer=tokenizer,
            )
            row.update(
                manifest_keys=list(manifest_keys),
                repair_manifest_keys=list(repair_manifest_keys),
                manifest_summaries=coverage["manifest_summaries"],
                missing_files=coverage["missing_files"],
                failed_files=coverage["failed_files"],
                total_shards=coverage["total_shards"],
                total_tokens=coverage["total_tokens"],
            )
            if coverage["complete"]:
                row["status"] = "completed"
                registry_update(args, rental["pod_name"], status="ingest_complete", run_finished_at=utcnow_iso())
                return row
            if attempt >= args.repair_attempts:
                raise RuntimeError(
                    f"part incomplete after {attempt} repair attempts; "
                    f"missing={len(coverage['missing_files'])} failed={len(coverage['failed_files'])} "
                    f"first={coverage['repair_files'][:10]}"
                )
            repair_key = run_repair_attempt(
                pod_name=rental["pod_name"],
                env=lium_env,
                local_part_dir=local_part_dir,
                remote_root=remote_root,
                repair_files=coverage["repair_files"],
                dataset=dataset,
                tokenizer=tokenizer,
                dest_prefix=dest_prefix,
                run_id=run_id,
                base_part_id=part_id,
                attempt=attempt + 1,
                remote_env=remote_env,
                args=args,
                log_path=local_command_log,
            )
            repair_manifest_keys.append(repair_key)
            manifest_keys.append(repair_key)
    except Exception as exc:
        append_log(local_command_log, f"[exception] {exc!r}")
        row.update(status="failed", error=repr(exc))
        registry_update(args, rental["pod_name"], status="ingest_failed", last_error=repr(exc), run_finished_at=utcnow_iso())
        return row
    finally:
        try:
            copy_back_logs(rental["pod_name"], lium_env, remote_root, local_part_dir)
        except Exception as exc:
            append_log(local_command_log, f"[copy-back-exception] {exc!r}")
        if row.get("status") == "completed" and not args.keep_on_success:
            try:
                terminate_pod_for_args(rental["pod_name"], lium_env, args, reason="ingest_complete")
            except Exception as exc:
                append_log(local_command_log, f"[terminate-exception] {exc!r}")
        elif row.get("status") != "completed" and args.delete_on_failure:
            try:
                terminate_pod_for_args(rental["pod_name"], lium_env, args, reason="ingest_failed")
            except Exception as exc:
                append_log(local_command_log, f"[terminate-failed-exception] {exc!r}")


def run_once(args: argparse.Namespace) -> int:
    dataset = ingest_hf.normalize_hf_repo_id(args.dataset, "dataset")
    tokenizer = ingest_hf.normalize_hf_repo_id(args.tokenizer, "model")
    dest_prefix = args.dest_prefix.rstrip("/")
    run_id = args.run_id or utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + slugify(dataset.split("/")[-1])
    run_dir = args.results_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(dataset, args)
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit("no parquet files discovered")
    parts = partition_round_robin(files, args.pods)
    part_ids = [f"part-{idx:03d}" for idx in range(args.pods)]

    rows = [
        {
            "part_id": part_ids[idx],
            "status": "pending",
            "file_count": len(part_files),
            "manifest_key": ingest_hf.part_manifest_key(dest_prefix, run_id, part_ids[idx]),
        }
        for idx, part_files in enumerate(parts)
    ]
    write_json(run_dir / "input_files.json", {"files": files})
    write_json(run_dir / "plan.json", {"run_id": run_id, "parts": rows})
    write_json(run_dir / "status.json", build_status(run_id=run_id, status="running", dataset=dataset, tokenizer=tokenizer, dest_prefix=dest_prefix, rows=rows, run_dir=run_dir, args=args))

    if args.dry_run:
        for idx, part_files in enumerate(parts):
            write_json(run_dir / "parts" / part_ids[idx] / "files.json", {"files": part_files})
        write_json(run_dir / "status.json", build_status(run_id=run_id, status="dry_run", dataset=dataset, tokenizer=tokenizer, dest_prefix=dest_prefix, rows=rows, run_dir=run_dir, args=args))
        print(f"[dry-run] run_dir={run_dir}")
        print(f"[dry-run] files={len(files)} parts={args.pods}")
        return 0

    lium_env = lium_eval.build_lium_env(args)
    remote_env = build_remote_env(args)
    existing_pods = args.existing_pods_list
    gpu_specs = [] if existing_pods else parse_gpu_specs(args.gpu_specs)

    row_by_part = {row["part_id"]: dict(row) for row in rows}
    for idx, part_id in enumerate(part_ids):
        row_by_part[part_id].update(
            status="renting",
            pod_name=None,
            gpu=None,
            gpu_count=None,
            local_dir=str(run_dir / "parts" / part_id),
            remote_root=f"{args.remote_base.rstrip('/')}/{run_id}/{part_id}",
            error=None,
        )
    write_json(
        run_dir / "status.json",
        build_status(
            run_id=run_id,
            status="acquiring_pods",
            dataset=dataset,
            tokenizer=tokenizer,
            dest_prefix=dest_prefix,
            rows=[row_by_part[pid] for pid in part_ids],
            run_dir=run_dir,
            args=args,
        ),
    )

    rental_by_part: dict[str, dict[str, Any]] = {}
    rent_pool = cf.ThreadPoolExecutor(max_workers=args.pods)
    rent_future_map = {}
    try:
        if existing_pods:
            rent_future_map = {
                rent_pool.submit(
                    claim_existing_pod_for_part,
                    part_id=part_ids[idx],
                    run_id=run_id,
                    pod_name=existing_pods[idx],
                    args=args,
                    env=lium_env,
                ): part_ids[idx]
                for idx in range(args.pods)
            }
        else:
            rent_future_map = {
                rent_pool.submit(
                    rent_pod_for_part,
                    part_id=part_ids[idx],
                    run_id=run_id,
                    gpu_specs=gpu_specs,
                    args=args,
                    env=lium_env,
                ): part_ids[idx]
                for idx in range(args.pods)
            }
        for fut in cf.as_completed(rent_future_map):
            part_id = rent_future_map[fut]
            try:
                rental = fut.result()
            except Exception as exc:
                rental = {"pod_name": None, "gpu": None, "count": None, "error": repr(exc)}
            rental_by_part[part_id] = rental
            if rental.get("pod_name"):
                row_by_part[part_id].update(
                    status="acquired",
                    pod_name=rental["pod_name"],
                    gpu=rental["gpu"],
                    gpu_count=rental["count"],
                    error=None,
                )
                registry_update(args, rental["pod_name"], status="acquired", local_run_dir=str(run_dir), remote_root=row_by_part[part_id]["remote_root"])
            else:
                row_by_part[part_id].update(status="failed", error=rental.get("error") or "rent failed")
            write_json(
                run_dir / "status.json",
                build_status(
                    run_id=run_id,
                    status="acquiring_pods",
                    dataset=dataset,
                    tokenizer=tokenizer,
                    dest_prefix=dest_prefix,
                    rows=[row_by_part[pid] for pid in part_ids],
                    run_dir=run_dir,
                    args=args,
                ),
            )
    except KeyboardInterrupt:
        for fut in rent_future_map:
            fut.cancel()
        rent_pool.shutdown(wait=False, cancel_futures=True)
        for rental in rental_by_part.values():
            if rental.get("pod_name"):
                try:
                    terminate_pod_for_args(rental["pod_name"], lium_env, args, reason="acquire_interrupted")
                except Exception:
                    pass
        current_rows = [row_by_part[pid] for pid in part_ids]
        for row in current_rows:
            if row.get("status") in {"pending", "renting", "acquired"}:
                row["status"] = "interrupted"
        write_json(run_dir / "status.json", build_status(run_id=run_id, status="interrupted", dataset=dataset, tokenizer=tokenizer, dest_prefix=dest_prefix, rows=current_rows, run_dir=run_dir, args=args))
        print(f"[interrupted] wrote status to {run_dir / 'status.json'}", flush=True)
        return 130
    finally:
        rent_pool.shutdown(wait=False, cancel_futures=True)

    acquisition_failed = [row_by_part[pid] for pid in part_ids if row_by_part[pid].get("status") != "acquired"]
    if acquisition_failed:
        for rental in rental_by_part.values():
            if rental.get("pod_name"):
                try:
                    terminate_pod_for_args(rental["pod_name"], lium_env, args, reason="acquire_failed")
                except Exception:
                    pass
        current_rows = [row_by_part[pid] for pid in part_ids]
        write_json(run_dir / "status.json", build_status(run_id=run_id, status="rent_failed", dataset=dataset, tokenizer=tokenizer, dest_prefix=dest_prefix, rows=current_rows, run_dir=run_dir, args=args))
        print(f"[warn] not starting ingest because {len(acquisition_failed)} pod rentals failed", flush=True)
        return 1

    pool = cf.ThreadPoolExecutor(max_workers=args.pods)
    future_map = {}
    try:
        future_map = {
            pool.submit(
                run_part,
                part_idx=idx,
                part_files=part_files,
                run_id=run_id,
                dataset=dataset,
                tokenizer=tokenizer,
                dest_prefix=dest_prefix,
                rental=rental_by_part[part_ids[idx]],
                lium_env=lium_env,
                remote_env=remote_env,
                run_dir=run_dir,
                args=args,
            ): part_ids[idx]
            for idx, part_files in enumerate(parts)
        }
        for fut in cf.as_completed(future_map):
            part_id = future_map[fut]
            row_by_part[part_id].update(fut.result())
            current_rows = [row_by_part[pid] for pid in part_ids]
            write_json(run_dir / "status.json", build_status(run_id=run_id, status="running", dataset=dataset, tokenizer=tokenizer, dest_prefix=dest_prefix, rows=current_rows, run_dir=run_dir, args=args))
    except KeyboardInterrupt:
        for fut in future_map:
            fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        current_rows = [row_by_part[pid] for pid in part_ids]
        for row in current_rows:
            if row.get("status") in {"pending", "acquired", "running", "bootstrapping"}:
                row["status"] = "interrupted"
        write_json(run_dir / "status.json", build_status(run_id=run_id, status="interrupted", dataset=dataset, tokenizer=tokenizer, dest_prefix=dest_prefix, rows=current_rows, run_dir=run_dir, args=args))
        print(f"[interrupted] wrote status to {run_dir / 'status.json'}", flush=True)
        return 130
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    final_rows = [row_by_part[pid] for pid in part_ids]
    failed = [row for row in final_rows if row.get("status") != "completed"]
    final_status = "partial" if failed else "completed"
    final_manifest = None
    if not failed and not args.no_merge:
        merge_part_ids: list[str] = []
        marker = f"/parts/{run_id}/"
        for row in final_rows:
            manifest_keys = row.get("manifest_keys") or [row["manifest_key"]]
            for key in manifest_keys:
                if marker not in key or not key.endswith("/manifest.json"):
                    raise RuntimeError(f"cannot derive part id from manifest key: {key}")
                merge_part_ids.append(key.split(marker, 1)[1].rsplit("/manifest.json", 1)[0])
        final_manifest = merge_part_manifests(
            run_id=run_id,
            part_ids=merge_part_ids,
            expected_files=files,
            dataset=dataset,
            tokenizer=tokenizer,
            dest_prefix=dest_prefix,
            args=args,
            remote_env=remote_env,
        )
        write_json(run_dir / "merged_manifest.summary.json", {k: v for k, v in final_manifest.items() if k != "shards"})
    elif failed:
        print(f"[warn] not merging because {len(failed)} parts failed", flush=True)

    payload = build_status(run_id=run_id, status=final_status, dataset=dataset, tokenizer=tokenizer, dest_prefix=dest_prefix, rows=final_rows, run_dir=run_dir, args=args)
    if final_manifest is not None:
        payload["merged_manifest"] = {
            "key": f"{dest_prefix}/manifest.json",
            "total_shards": final_manifest["total_shards"],
            "total_tokens": final_manifest["total_tokens"],
        }
    write_json(run_dir / "status.json", payload)
    print(f"[done] status={final_status} run_dir={run_dir}")
    if final_manifest is not None:
        print(f"[done] manifest=s3://{args.hippius_bucket}/{dest_prefix}/manifest.json shards={final_manifest['total_shards']}")
    return 0 if not failed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rent Lium pods and run distributed HF parquet tokenization ingest.")
    parser.add_argument("--dataset", default=os.environ.get("TEUTONIC_LIUM_INGEST_DATASET", DEFAULT_DATASET))
    parser.add_argument("--tokenizer", default=os.environ.get("TEUTONIC_LIUM_INGEST_TOKENIZER", DEFAULT_TOKENIZER))
    parser.add_argument("--dest-prefix", default=os.environ.get("TEUTONIC_LIUM_INGEST_DEST_PREFIX", DEFAULT_DEST_PREFIX))
    parser.add_argument("--pods", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_PODS", "4")))
    parser.add_argument("--existing-pods", default=os.environ.get("TEUTONIC_LIUM_INGEST_EXISTING_PODS", ""), help="Comma-separated already-rented Lium pod names. When set, the coordinator skips lium up and uses these pods in part order.")
    parser.add_argument("--gpu-specs", default=os.environ.get("TEUTONIC_LIUM_INGEST_GPU_SPECS", "A100:1,H100:8,H200:8"), help="Comma-separated fallback specs like A100:1,H100:8,H200:8.")
    parser.add_argument("--run-id", default=os.environ.get("TEUTONIC_LIUM_INGEST_RUN_ID", ""))
    parser.add_argument("--results-root", type=Path, default=Path(os.environ.get("TEUTONIC_LIUM_INGEST_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)))
    parser.add_argument("--registry", type=Path, default=Path(os.environ.get("TEUTONIC_LIUM_INGEST_REGISTRY", DEFAULT_REGISTRY)))
    parser.add_argument("--remote-base", default=os.environ.get("TEUTONIC_LIUM_INGEST_REMOTE_BASE", DEFAULT_REMOTE_BASE))
    parser.add_argument("--remote-scratch-dir", default=os.environ.get("TEUTONIC_LIUM_INGEST_REMOTE_SCRATCH", "/var/tmp/teutonic-ingest"))
    parser.add_argument("--remote-progress-dir", default=os.environ.get("TEUTONIC_LIUM_INGEST_REMOTE_PROGRESS", "/var/tmp/teutonic-ingest-progress"))
    parser.add_argument("--remote-workers", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_REMOTE_WORKERS", "16")))
    parser.add_argument("--remote-min-free-gb", type=float, default=float(os.environ.get("TEUTONIC_LIUM_INGEST_REMOTE_MIN_FREE_GB", "20")))
    parser.add_argument("--remote-worker-disk-gb", type=float, default=float(os.environ.get("TEUTONIC_LIUM_INGEST_REMOTE_WORKER_DISK_GB", "1")))
    parser.add_argument("--remote-max-inflight-files", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_REMOTE_MAX_INFLIGHT", "16")))
    parser.add_argument("--remote-auto-max-workers", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_REMOTE_AUTO_MAX_WORKERS", "32")))
    parser.add_argument("--repair-attempts", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_REPAIR_ATTEMPTS", "2")), help="Retry only missing/failed files on the same pod before marking a part failed.")
    parser.add_argument("--repair-workers", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_REPAIR_WORKERS", "4")), help="Workers to use for targeted repair attempts. 0 means min(remote-workers, 4).")
    parser.add_argument("--repair-max-inflight-files", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_REPAIR_MAX_INFLIGHT", "4")), help="Max in-flight files for targeted repair attempts. 0 means repair-workers.")
    parser.add_argument("--text-column", default=os.environ.get("TEUTONIC_LIUM_INGEST_TEXT_COLUMN", "text"))
    parser.add_argument("--include-prefixes", default=os.environ.get("TEUTONIC_LIUM_INGEST_INCLUDE_PREFIXES", None))
    parser.add_argument("--langs", default=os.environ.get("TEUTONIC_LIUM_INGEST_LANGS", None))
    parser.add_argument("--limit-files", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_LIMIT_FILES", "0")))
    parser.add_argument("--ttl", default=os.environ.get("TEUTONIC_LIUM_INGEST_TTL", "36h"))
    parser.add_argument("--template-id", default=os.environ.get("TEUTONIC_LIUM_INGEST_TEMPLATE_ID", ""))
    parser.add_argument("--wait-timeout-s", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_WAIT_TIMEOUT_S", "1200")))
    parser.add_argument("--rent-timeout-s", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_RENT_TIMEOUT_S", "60")))
    parser.add_argument("--rent-stagger-s", type=int, default=int(os.environ.get("TEUTONIC_LIUM_INGEST_RENT_STAGGER_S", "30")), help="Seconds to stagger concurrent pod rentals by part index.")
    parser.add_argument("--doppler-project", default=os.environ.get("TEUTONIC_LIUM_INGEST_DOPPLER_PROJECT", "arbos"))
    parser.add_argument("--doppler-config", default=os.environ.get("TEUTONIC_LIUM_INGEST_DOPPLER_CONFIG", "dev"))
    parser.add_argument("--hf-doppler-config", default=os.environ.get("TEUTONIC_LIUM_INGEST_HF_DOPPLER_CONFIG", "prd"))
    parser.add_argument("--s3-doppler-config", default=os.environ.get("TEUTONIC_LIUM_INGEST_S3_DOPPLER_CONFIG", "dev"))
    parser.add_argument("--hippius-endpoint", default=os.environ.get("TEUTONIC_DS_ENDPOINT", "https://s3.hippius.com"))
    parser.add_argument("--hippius-bucket", default=os.environ.get("TEUTONIC_DS_BUCKET", "teutonic-sn3"))
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip dependency bootstrap on remote pods.")
    parser.add_argument("--keep-on-success", action="store_true", help="Do not delete successful pods.")
    parser.add_argument("--delete-on-failure", action="store_true", help="Delete failed pods after copying logs.")
    parser.add_argument("--no-registry-updates", action="store_true", help="Do not read/write the local Lium rental registry. Useful when passing already-rented pods or when another coordinator is updating the registry.")
    parser.add_argument("--no-merge", action="store_true", help="Do not merge part manifests into the final manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Discover/split files and write local plan without renting pods.")
    args = parser.parse_args()
    args.existing_pods_list = parse_existing_pods(args.existing_pods)
    pods_was_explicit = any(arg == "--pods" or arg.startswith("--pods=") for arg in sys.argv[1:])
    if args.existing_pods_list and not pods_was_explicit:
        args.pods = len(args.existing_pods_list)
    if args.pods <= 0:
        raise SystemExit("--pods must be positive")
    if args.repair_attempts < 0:
        raise SystemExit("--repair-attempts must be non-negative")
    if args.repair_workers < 0:
        raise SystemExit("--repair-workers must be non-negative")
    if args.repair_max_inflight_files < 0:
        raise SystemExit("--repair-max-inflight-files must be non-negative")
    if args.existing_pods_list and args.pods != len(args.existing_pods_list):
        raise SystemExit(f"--pods ({args.pods}) must match --existing-pods count ({len(args.existing_pods_list)})")
    return args


def main() -> int:
    args = parse_args()
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
