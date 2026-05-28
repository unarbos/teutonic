#!/usr/bin/env python3
"""Rent Lium pods, run king benchmark evals remotely, and collect artifacts."""

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_EVAL_SCRIPT = ROOT / "scripts" / "eval_king_benchmarks.py"
DEFAULT_RESULTS_ROOT = ROOT / "runs" / "king-benchmarks-lium"
DEFAULT_REGISTRY_PATH = ROOT / "runs" / "lium-rentals" / "registry.json"
DEFAULT_REMOTE_BASE = "/root/king-benchmark-evals"
DEFAULT_BENCHMARK_LABELS = [
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
DEFAULT_BENCHMARKS = ",".join(DEFAULT_BENCHMARK_LABELS)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-") or "run"


def lium_output_failed(cmd: list[str], output: str) -> bool:
    if not cmd or cmd[0] != "lium":
        return False
    markers = (
        "No pods match targets:",
        "Executor '",
        "Failed:",
        "Failed to download from:",
        "Failed to upload to:",
        "Command failed",
        "not found",
    )
    return any(marker in output for marker in markers)


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        check=False,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )
    output = proc.stdout or ""
    if capture_output and output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    failed = proc.returncode != 0 or lium_output_failed(cmd, output)
    if check and failed:
        raise subprocess.CalledProcessError(proc.returncode or 1, cmd, output=output)
    return proc


def doppler_secret(name: str, *, project: str, config: str) -> str:
    try:
        return subprocess.check_output(
            ["doppler", "secrets", "get", name, "--plain", "-p", project, "-c", config],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def build_lium_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if env.get("LIUM_API_KEY"):
        return env
    key = doppler_secret("LIUM_API_KEY", project=args.doppler_project, config=args.doppler_config)
    if not key:
        raise SystemExit(
            "missing LIUM_API_KEY; export it or add it to Doppler "
            f"({args.doppler_project}/{args.doppler_config})"
        )
    env["LIUM_API_KEY"] = key
    return env


def resolve_model_input(args: argparse.Namespace) -> str:
    model_input = getattr(args, "model_repo", None) or getattr(args, "model", None)
    if not model_input:
        raise SystemExit("provide MODEL or --model-repo")
    return model_input


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "updated_at": None, "rentals": []}
    return json.loads(path.read_text())


def write_registry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = utcnow_iso()
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def append_rental(path: Path, row: dict[str, Any]) -> None:
    payload = load_registry(path)
    payload.setdefault("rentals", []).append(row)
    write_registry(path, payload)


def update_rental(path: Path, pod_name: str, **updates: Any) -> None:
    payload = load_registry(path)
    rentals = payload.setdefault("rentals", [])
    for row in reversed(rentals):
        if row.get("pod_name") == pod_name:
            row.update(updates)
            write_registry(path, payload)
            return
    rentals.append({"pod_name": pod_name, **updates})
    write_registry(path, payload)


def list_hosts(args: argparse.Namespace) -> int:
    env = build_lium_env(args)
    cmd = ["lium", "ls", "--limit", str(args.limit), "--sort", args.sort]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    if args.count:
        cmd += ["--count", str(args.count)]
    run(cmd, env=env)
    return 0


def make_pod_name(prefix: str, executor_ref: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(prefix)}-{slugify(executor_ref)}-{stamp}"


def wait_for_pod(pod_name: str, env: dict[str, str], timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        probe = subprocess.run(
            ["lium", "exec", pod_name, "python3 --version || python --version"],
            env=env,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            print(f"[ready] pod {pod_name} is reachable", flush=True)
            return
        time.sleep(15)
    raise RuntimeError(f"timed out waiting for pod {pod_name}")


def rent_pod(args: argparse.Namespace) -> int:
    env = build_lium_env(args)
    ref_for_name = args.executor_ref or args.gpu or args.country or "auto"
    pod_name = args.name or make_pod_name(args.name_prefix, ref_for_name)
    cmd = ["lium", "up"]
    if args.executor_ref:
        cmd.append(args.executor_ref)
    cmd += ["--name", pod_name, "--ttl", args.ttl, "--yes"]
    if args.template_id:
        cmd += ["--template_id", args.template_id]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    if args.count:
        cmd += ["--count", str(args.count)]
    if args.country:
        cmd += ["--country", args.country]
    if args.ports:
        cmd += ["--ports", str(args.ports)]
    run(cmd, env=env)
    wait_for_pod(pod_name, env, args.wait_timeout_s)
    append_rental(
        args.registry,
        {
            "created_at": utcnow_iso(),
            "status": "running",
            "pod_name": pod_name,
            "executor_ref": args.executor_ref,
            "ttl": args.ttl,
            "filters": {
                "gpu": args.gpu,
                "count": args.count,
                "country": args.country,
                "template_id": args.template_id,
                "ports": args.ports,
            },
        },
    )
    print(f"[done] pod_name={pod_name}")
    print(f"[done] registry={args.registry}")
    return 0


def remote_script_paths(remote_root: str) -> dict[str, str]:
    return {
        "bootstrap": f"{remote_root}/bootstrap.sh",
        "runner": f"{remote_root}/run_eval.sh",
        "eval_script": f"{remote_root}/eval_king_benchmarks.py",
        "run_root": f"{remote_root}/run",
        "logs_dir": f"{remote_root}/logs",
        "bootstrap_log": f"{remote_root}/logs/bootstrap.log",
        "runner_log": f"{remote_root}/logs/run_eval.log",
        "venv": f"{remote_root}/.venv",
        "harness": f"{remote_root}/lm-evaluation-harness",
    }


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def append_log(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(body)
        if not body.endswith("\n"):
            fh.write("\n")


def run_logged(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = run(cmd, env=env, check=False, capture_output=True)
    if log_path is not None:
        append_log(log_path, "$ " + " ".join(shlex.quote(part) for part in cmd))
        append_log(log_path, proc.stdout or "")
        append_log(log_path, f"[returncode] {proc.returncode}")
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def terminate_pod(
    pod_name: str,
    env: dict[str, str],
    registry: Path,
    *,
    reason: str,
) -> None:
    run(["lium", "rm", pod_name], env=env)
    update_rental(
        registry,
        pod_name,
        terminated_at=utcnow_iso(),
        termination_reason=reason,
        status="terminated",
    )


def make_bootstrap_script(remote_root: str) -> str:
    paths = remote_script_paths(remote_root)
    return f"""#!/usr/bin/env bash
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
mkdir -p {shlex.quote(remote_root)}
retry apt-get update
retry apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  gcc \
  g++ \
  git \
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
  'torch>=2.6' \
  accelerate \
  antlr4-python3-runtime==4.11.1 \
  datasets \
  evaluate \
  hippius-hub \
  httpx \
  huggingface-hub \
  latex2sympy2 \
  numpy \
  protobuf \
  pyarrow \
  scikit-learn \
  scipy \
  sentencepiece \
  sympy \
  transformers \
  tiktoken

if [ ! -d {shlex.quote(paths["harness"] + "/.git")} ]; then
  retry git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git {shlex.quote(paths["harness"])}
fi

retry $PIP install -e {shlex.quote(paths["harness"] + "[math,sentencepiece]")}
$PY -c 'import datasets, evaluate, hippius_hub, httpx, lm_eval, torch, transformers; print("bootstrap-ok")'
nvidia-smi
"""


def make_run_script(
    remote_root: str,
    model_input: str,
    benchmark_csv: str,
    args: argparse.Namespace,
) -> str:
    paths = remote_script_paths(remote_root)
    cmd = [
        paths["venv"] + "/bin/python",
        paths["eval_script"],
        "--model-repo",
        model_input,
        "--run-dir",
        paths["run_root"],
        "--benchmarks",
        benchmark_csv,
        "--batch-size",
        args.batch_size,
        "--dtype",
        args.dtype,
        "--standardized-results-path",
        f"{paths['run_root']}/standardized_results.json",
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.model_args_extra:
        cmd += ["--model-args-extra", args.model_args_extra]
    if args.gen_kwargs:
        cmd += ["--gen-kwargs", args.gen_kwargs]
    if args.fewshot_overrides:
        cmd += ["--fewshot-overrides", args.fewshot_overrides]
    if args.log_samples:
        cmd.append("--log-samples")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.apply_chat_template:
        cmd.append("--apply-chat-template")
    if args.resume:
        cmd.append("--resume")
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(
        shlex.quote(part) for part in cmd
    ) + "\n"


def stage_remote_files(
    pod_name: str,
    env: dict[str, str],
    local_run_dir: Path,
    remote_root: str,
    model_input: str,
    benchmark_csv: str,
    args: argparse.Namespace,
) -> None:
    paths = remote_script_paths(remote_root)
    bootstrap_local = local_run_dir / "bootstrap.sh"
    runner_local = local_run_dir / "run_eval.sh"
    eval_local = local_run_dir / "eval_king_benchmarks.py"
    write_text(bootstrap_local, make_bootstrap_script(remote_root))
    write_text(runner_local, make_run_script(remote_root, model_input, benchmark_csv, args))
    eval_local.write_text(LOCAL_EVAL_SCRIPT.read_text())

    run(["lium", "exec", pod_name, f"mkdir -p {shlex.quote(remote_root)} {shlex.quote(paths['logs_dir'])}"], env=env)
    run(["lium", "scp", pod_name, str(bootstrap_local), paths["bootstrap"]], env=env)
    run(["lium", "scp", pod_name, str(runner_local), paths["runner"]], env=env)
    run(["lium", "scp", pod_name, str(eval_local), paths["eval_script"]], env=env)
    run(
        [
            "lium",
            "exec",
            pod_name,
            (
                f"chmod +x {shlex.quote(paths['bootstrap'])} "
                f"{shlex.quote(paths['runner'])} {shlex.quote(paths['eval_script'])}"
            ),
        ],
        env=env,
    )


def copy_back_results(
    pod_name: str,
    env: dict[str, str],
    remote_root: str,
    local_run_dir: Path,
) -> None:
    download_dir = local_run_dir / "remote-artifacts"
    download_dir.mkdir(parents=True, exist_ok=True)
    remote_root = remote_root.rstrip("/")
    remote_parent = posixpath.dirname(remote_root)
    remote_leaf = posixpath.basename(remote_root)
    remote_tar = f"/tmp/{slugify(remote_leaf)}-artifacts.tgz"
    local_tar = local_run_dir / "remote-artifacts.tgz"

    # Copy back only benchmark outputs and logs. The remote root also contains
    # the model snapshot, venv, and lm-eval checkout, which can make tar fail
    # after a successful evaluation because the archive is huge.
    excludes = [
        f"{remote_leaf}/.venv",
        f"{remote_leaf}/lm-evaluation-harness",
        f"{remote_leaf}/run/model",
    ]
    exclude_args = " ".join(
        f"--exclude={shlex.quote(pattern)}" for pattern in excludes
    )
    run(
        [
            "lium",
            "exec",
            pod_name,
            (
                f"tar -C {shlex.quote(remote_parent)} {exclude_args} -czf "
                f"{shlex.quote(remote_tar)} {shlex.quote(remote_leaf)}"
            ),
        ],
        env=env,
    )
    try:
        run(["lium", "scp", "--download", pod_name, remote_tar, str(local_tar)], env=env)
        with tarfile.open(local_tar, "r:gz") as archive:
            archive.extractall(download_dir)
    finally:
        run(
            ["lium", "exec", pod_name, f"rm -f {shlex.quote(remote_tar)}"],
            env=env,
            check=False,
        )


def run_remote_eval_job(
    *,
    pod_name: str,
    model_input: str,
    benchmark_csv: str,
    results_root: Path,
    remote_base: str,
    remote_root_override: str | None,
    env: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not LOCAL_EVAL_SCRIPT.exists():
        raise SystemExit(f"missing local evaluator script: {LOCAL_EVAL_SCRIPT}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{slugify(pod_name)}-{slugify(benchmark_csv)}-{slugify(Path(model_input).name)}"
    local_run_dir = results_root.expanduser().resolve() / run_id
    remote_root = remote_root_override or f"{remote_base.rstrip('/')}/{run_id}"
    local_run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at": utcnow_iso(),
        "pod_name": pod_name,
        "model_repo": model_input,
        "benchmarks": benchmark_csv,
        "remote_root": remote_root,
        "local_run_dir": str(local_run_dir),
    }
    write_text(local_run_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")

    update_rental(
        args.registry,
        pod_name,
        last_run_id=run_id,
        last_model=model_input,
        last_benchmarks=benchmark_csv,
        local_run_dir=str(local_run_dir),
        remote_root=remote_root,
        status="running_eval",
    )

    status = "eval_complete"
    run_started = utcnow_iso()
    summary_path = None
    standardized_path = None
    local_command_log = local_run_dir / "local-command.log"
    remote_paths = remote_script_paths(remote_root)
    try:
        stage_remote_files(pod_name, env, local_run_dir, remote_root, model_input, benchmark_csv, args)
        if not args.skip_bootstrap:
            run_logged(
                [
                    "lium",
                    "exec",
                    pod_name,
                    f"bash {shlex.quote(remote_paths['bootstrap'])} > {shlex.quote(remote_paths['bootstrap_log'])} 2>&1",
                ],
                env=env,
                log_path=local_command_log,
            )
        run_logged(
            [
                "lium",
                "exec",
                pod_name,
                f"bash {shlex.quote(remote_paths['runner'])} > {shlex.quote(remote_paths['runner_log'])} 2>&1",
            ],
            env=env,
            log_path=local_command_log,
        )
    except Exception as exc:
        status = "eval_failed"
        append_log(local_command_log, f"[exception] {exc!r}")
        raise
    finally:
        copy_error = None
        try:
            copy_back_results(pod_name, env, remote_root, local_run_dir)
        except Exception as exc:
            copy_error = exc
            append_log(local_command_log, f"[copy-back-exception] {exc!r}")
            if status == "eval_complete":
                status = "artifact_copy_failed"
        summary_candidates = list((local_run_dir / "remote-artifacts").rglob("summary.json"))
        standardized_candidates = list((local_run_dir / "remote-artifacts").rglob("standardized_results.json"))
        summary_path = str(summary_candidates[0]) if summary_candidates else None
        standardized_path = str(standardized_candidates[0]) if standardized_candidates else None
        update_rental(
            args.registry,
            pod_name,
            status=status,
            run_started_at=run_started,
            run_finished_at=utcnow_iso(),
            summary_path=summary_path,
            standardized_results_path=standardized_path,
            local_command_log=str(local_command_log),
            remote_bootstrap_log=f"{remote_paths['bootstrap_log']}",
            remote_runner_log=f"{remote_paths['runner_log']}",
        )
        if status == "eval_complete" and not args.keep_on_success:
            terminate_pod(pod_name, env, args.registry, reason="eval_complete")
        elif status != "eval_complete" and args.delete_on_failure:
            terminate_pod(pod_name, env, args.registry, reason="eval_failed")
        if copy_error is not None:
            raise copy_error
    return {
        "pod_name": pod_name,
        "benchmark": benchmark_csv,
        "status": status,
        "local_run_dir": str(local_run_dir),
        "summary_path": summary_path,
        "standardized_results_path": standardized_path,
    }


def read_json_if_exists(path_str: str | None) -> dict[str, Any] | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parse_requested_benchmarks(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return list(DEFAULT_BENCHMARK_LABELS)
    canonical = {label.lower(): label for label in DEFAULT_BENCHMARK_LABELS}
    selected = []
    for part in raw_value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in canonical:
            raise SystemExit(f"unknown benchmark label for distributed run: {part!r}")
        selected.append(canonical[key])
    if not selected:
        raise SystemExit("no benchmarks selected")
    return selected


def parse_pod_assignments(items: list[str], requested: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    canonical = {label.lower(): label for label in DEFAULT_BENCHMARK_LABELS}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --pod assignment {item!r}; expected BENCHMARK=POD_NAME")
        raw_label, pod_name = item.split("=", 1)
        key = raw_label.strip().lower()
        label = canonical.get(key)
        if not label:
            raise SystemExit(f"unknown benchmark label in --pod assignment: {raw_label!r}")
        if not pod_name.strip():
            raise SystemExit(f"missing pod name in --pod assignment: {item!r}")
        mapping[label] = pod_name.strip()
    missing = [label for label in requested if label not in mapping]
    if missing:
        raise SystemExit(f"missing --pod assignments for benchmarks: {', '.join(missing)}")
    return {label: mapping[label] for label in requested}


def summarize_suite_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"requested": len(rows), "completed": 0, "failed": 0}
    for row in rows:
        if row.get("status") == "completed":
            counts["completed"] += 1
        else:
            counts["failed"] += 1
    return counts


def write_suite_standardized_results(suite_dir: Path, payload: dict[str, Any]) -> Path:
    path = suite_dir / "standardized_results.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def run_remote_eval(args: argparse.Namespace) -> int:
    env = build_lium_env(args)
    model_input = resolve_model_input(args)
    result = run_remote_eval_job(
        pod_name=args.pod_name,
        model_input=model_input,
        benchmark_csv=args.benchmarks,
        results_root=args.results_root,
        remote_base=args.remote_base,
        remote_root_override=args.remote_root,
        env=env,
        args=args,
    )
    print(f"[done] local_run_dir={result['local_run_dir']}")
    if result["summary_path"]:
        print(f"[done] summary_json={result['summary_path']}")
    if result["standardized_results_path"]:
        print(f"[done] standardized_results_json={result['standardized_results_path']}")
    print(f"[done] registry={args.registry}")
    return 0


def run_suite(args: argparse.Namespace) -> int:
    env = build_lium_env(args)
    model_input = resolve_model_input(args)
    requested = parse_requested_benchmarks(args.benchmarks)
    assignments = parse_pod_assignments(args.pod, requested)

    suite_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + slugify(Path(model_input).name)
    suite_dir = args.results_root.expanduser().resolve() / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        suite_dir / "suite_metadata.json",
        json.dumps(
            {
                "created_at": utcnow_iso(),
                "model_repo": model_input,
                "benchmarks": requested,
                "assignments": assignments,
                "results_root": str(suite_dir),
            },
            indent=2,
        )
        + "\n",
    )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def worker(label: str, pod_name: str) -> dict[str, Any]:
        try:
            result = run_remote_eval_job(
                pod_name=pod_name,
                model_input=model_input,
                benchmark_csv=label,
                results_root=suite_dir / "benchmarks" / slugify(label),
                remote_base=f"{args.remote_base.rstrip('/')}/{suite_id}",
                remote_root_override=None,
                env=env,
                args=args,
            )
            payload = read_json_if_exists(result.get("standardized_results_path"))
            bench_row = (payload or {}).get("benchmarks", [{}])[0] if payload else {}
            return {
                "benchmark": label,
                "pod_name": pod_name,
                "status": bench_row.get("status") or result["status"],
                "metric": bench_row.get("metric"),
                "local_run_dir": result["local_run_dir"],
                "summary_json": result.get("summary_path"),
                "standardized_results_json": result.get("standardized_results_path"),
                "raw": payload,
            }
        except Exception as exc:
            return {
                "benchmark": label,
                "pod_name": pod_name,
                "status": "failed",
                "error": repr(exc),
                "local_run_dir": None,
                "summary_json": None,
                "standardized_results_json": None,
                "raw": None,
            }

    with cf.ThreadPoolExecutor(max_workers=len(requested)) as pool:
        future_map = {
            pool.submit(worker, label, assignments[label]): (label, assignments[label])
            for label in requested
        }
        for fut in cf.as_completed(future_map):
            row = fut.result()
            results.append(row)
            if row.get("status") != "completed":
                errors.append(row)

    results.sort(key=lambda row: requested.index(row["benchmark"]))
    suite_payload = {
        "schema_version": "king-benchmark-suite-results.v1",
        "generated_at": utcnow_iso(),
        "suite_run_id": suite_id,
        "model_repo": model_input,
        "totals": summarize_suite_status(results),
        "benchmarks": results,
    }
    standardized_path = write_suite_standardized_results(suite_dir, suite_payload)
    print(f"[done] suite_dir={suite_dir}")
    print(f"[done] standardized_results_json={standardized_path}")
    print(f"[done] registry={args.registry}")
    return 1 if errors else 0


def list_rentals(args: argparse.Namespace) -> int:
    payload = load_registry(args.registry)
    print(json.dumps(payload, indent=2))
    return 0


def add_model_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", nargs="?", help="Optional positional model repo/id input.")
    parser.add_argument(
        "--model-repo",
        default=None,
        help="Automation-friendly explicit king model repo/id input.",
    )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--remote-base", default=DEFAULT_REMOTE_BASE)
    parser.add_argument("--remote-root", default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-args-extra", default="")
    parser.add_argument("--gen-kwargs", default="")
    parser.add_argument("--fewshot-overrides", default="")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--keep-on-success", action="store_true", help="Do not auto-delete a pod after successful artifact collection.")
    parser.add_argument("--delete-on-failure", action="store_true", help="Delete a pod after a failed run, but only after logs and artifacts are copied back.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rent Lium hosts and run king benchmark evaluations.")
    parser.add_argument("--doppler-project", default="arbos")
    parser.add_argument("--doppler-config", default="dev")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)

    sub = parser.add_subparsers(dest="command", required=True)

    hosts = sub.add_parser("list-hosts", help="List rentable Lium executors.")
    hosts.add_argument("--limit", type=int, default=20)
    hosts.add_argument("--sort", default="price_total")
    hosts.add_argument("--gpu", default=None)
    hosts.add_argument("--count", type=int, default=0)
    hosts.set_defaults(func=list_hosts)

    rent = sub.add_parser("rent", help="Rent a specific executor and save it in the registry.")
    rent.add_argument("executor_ref", nargs="?", default=None)
    rent.add_argument("--name", default=None)
    rent.add_argument("--name-prefix", default="king-bench")
    rent.add_argument("--ttl", default="12h")
    rent.add_argument("--template-id", default=None)
    rent.add_argument("--gpu", default=None)
    rent.add_argument("--count", type=int, default=0)
    rent.add_argument("--country", default=None)
    rent.add_argument("--ports", type=int, default=0)
    rent.add_argument("--wait-timeout-s", type=int, default=1200)
    rent.set_defaults(func=rent_pod)

    runner = sub.add_parser("run", help="Run one or more benchmarks on a single rented pod.")
    runner.add_argument("pod_name")
    add_model_input_arguments(runner)
    add_run_arguments(runner)
    runner.set_defaults(func=run_remote_eval)

    suite = sub.add_parser("run-suite", help="Run one benchmark per rented pod and aggregate results.")
    suite.add_argument(
        "--pod",
        action="append",
        default=[],
        help="Benchmark-to-pod mapping in the form BENCHMARK=POD_NAME. Repeat once per benchmark.",
    )
    add_model_input_arguments(suite)
    add_run_arguments(suite)
    suite.set_defaults(func=run_suite)

    rentals = sub.add_parser("list-rentals", help="Print the saved pod registry.")
    rentals.set_defaults(func=list_rentals)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
