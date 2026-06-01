#!/usr/bin/env python3
"""Run the requested benchmark panel against a Teutonic king snapshot.

Accepts a Hippius Hub model page URL, a `project/repo` id, a pinned
`project/repo@sha256:...` ref, or a local model directory.

The script resolves public Hippius artifacts through the model-index API,
downloads the chosen snapshot once, then runs each benchmark separately via
lm-eval-harness so the requested few-shot count is preserved per benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


import httpx

try:
    from hippius_hub import snapshot_download as hippius_snapshot_download
except ImportError as exc:  # pragma: no cover - import path issue, not logic.
    raise SystemExit(
        "hippius_hub is not importable. Run this script from the Teutonic venv, for example:\n"
        "  /root/teutonic/.venv/bin/python scripts/eval_king_benchmarks.py ..."
    ) from exc

try:
    from huggingface_hub import HfApi, snapshot_download as hf_snapshot_download
except ImportError:  # pragma: no cover - optional fallback dependency.
    HfApi = None
    hf_snapshot_download = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "runs" / "king-benchmarks"
HIPPIUS_MODEL_API = "https://api.hippius.com/api/models"
HIPPIUS_MODEL_URL_RE = re.compile(r"^/models/([^/]+)/([^/]+)/?$")
HF_MODEL_URL_RE = re.compile(r"^/([^/]+)/([^/]+)(?:/(?:tree|resolve)/([^/]+))?/?$")
PINNED_REF_RE = re.compile(r"^([^/@\s]+/[^@\s]+)@(sha256:[0-9a-f]{64}|hf:[0-9a-f]{40})$")


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    task: str
    fewshot: int
    metric_candidates: tuple[str, ...]
    unsafe_code: bool = False


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("MMLU", "mmlu", 5, ("acc,none", "acc")),
    BenchmarkSpec("MMLU-Pro", "mmlu_pro", 5, ("acc,none", "acc")),
    BenchmarkSpec(
        "BBH",
        "leaderboard_bbh",
        3,
        ("acc,none", "exact_match,none", "exact_match", "acc"),
    ),
    BenchmarkSpec("ARC-C", "arc_challenge", 25, ("acc_norm,none", "acc_norm", "acc,none", "acc")),
    BenchmarkSpec("TruthfulQA", "truthfulqa_mc2", 0, ("mc2,none", "mc2", "acc,none", "acc")),
    BenchmarkSpec("WinoGrande", "winogrande", 5, ("acc,none", "acc")),
    BenchmarkSpec("HellaSwag", "hellaswag", 10, ("acc_norm,none", "acc_norm", "acc,none", "acc")),
    BenchmarkSpec(
        "GSM8K",
        "gsm8k",
        4,
        ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "exact_match"),
    ),
    BenchmarkSpec(
        "MATH",
        "minerva_math",
        4,
        ("exact_match,none", "exact_match", "math_verify,none", "math_verify"),
    ),
    BenchmarkSpec(
        "HumanEval",
        "humaneval",
        0,
        ("pass@1,create_test", "pass@1,none", "pass@1", "pass_at_1,create_test", "pass_at_1,none", "pass_at_1"),
        unsafe_code=True,
    ),
    BenchmarkSpec(
        "MBPP",
        "mbpp",
        0,
        ("pass@1,create_test", "pass@1,none", "pass@1", "pass_at_1,create_test", "pass_at_1,none", "pass_at_1"),
        unsafe_code=True,
    ),
)


@dataclass
class ModelTarget:
    source: str
    local_path: str | None
    repo_id: str | None
    revision: str | None
    digest: str | None
    primary_tag: str | None
    artifact: dict[str, Any] | None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return out.strip("-") or "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Teutonic king snapshot on the requested lm-eval benchmarks."
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Hippius Hub URL, project/repo, project/repo@sha256:..., or local model directory",
    )
    parser.add_argument(
        "--model-repo",
        default=None,
        help="Automation-friendly explicit model repo/id input. Accepts the same values as the positional model argument.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hippius revision/tag/digest override when --model is a repo id or URL",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory for run artifacts (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory. If it already has a summary.json, --resume can reuse it.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional explicit local directory for the downloaded model snapshot",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="lm-eval batch size (default: auto)",
    )
    parser.add_argument(
        "--max-batch-size",
        default=os.environ.get("TEUTONIC_KING_BENCH_MAX_BATCH_SIZE", ""),
        help="Optional lm-eval --max_batch_size when using auto batch sizing.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional lm-eval device override, e.g. cuda:0",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype passed to lm-eval hf backend (default: bfloat16)",
    )
    parser.add_argument(
        "--model-args-extra",
        default="",
        help="Extra comma-separated hf backend model_args entries for lm-eval",
    )
    parser.add_argument(
        "--gen-kwargs",
        default="",
        help="Optional lm-eval --gen_kwargs string applied to every benchmark run",
    )
    parser.add_argument(
        "--fewshot-overrides",
        default="",
        help="Comma-separated per-benchmark few-shot overrides, e.g. 'MBPP=3,MATH=4'. Matches labels or task ids.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Forward --log_samples to lm-eval so generated outputs are saved for debugging.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="Optional lm-eval sample limit for smoke runs",
    )
    parser.add_argument(
        "--lm-eval-bin",
        default=os.environ.get("LM_EVAL_BIN", ""),
        help="Explicit lm-eval executable or command string",
    )
    parser.add_argument(
        "--cache-requests",
        default=os.environ.get("TEUTONIC_KING_BENCH_CACHE_REQUESTS", "true"),
        help="lm-eval request preprocessing cache mode: true, refresh, delete, or empty to disable.",
    )
    parser.add_argument(
        "--response-cache-root",
        type=Path,
        default=Path(os.environ["TEUTONIC_KING_BENCH_RESPONSE_CACHE_ROOT"]) if os.environ.get("TEUTONIC_KING_BENCH_RESPONSE_CACHE_ROOT") else None,
        help="Optional root for per-model lm-eval SQLite response caches. Disabled by default.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Resolve and download the snapshot, then stop before lm-eval",
    )
    parser.add_argument(
        "--prefetch-tasks",
        default=os.environ.get("TEUTONIC_KING_BENCH_PREFETCH_TASKS", "true"),
        help="Load each selected lm-eval task once before evaluation to warm the datasets cache. Set empty/false/no/0 to disable.",
    )
    parser.add_argument(
        "--prefetch-timeout-s",
        type=int,
        default=int(os.environ.get("TEUTONIC_KING_BENCH_PREFETCH_TIMEOUT_S", "1800")),
        help="Per-task timeout for lm-eval task prefetching in seconds (default: 1800).",
    )
    parser.add_argument(
        "--prefetch-retries",
        type=int,
        default=int(os.environ.get("TEUTONIC_KING_BENCH_PREFETCH_RETRIES", "4")),
        help="Retries for transient Hugging Face/dataset failures while warming task caches (default: 4).",
    )
    parser.add_argument(
        "--offline-after-prefetch",
        default=os.environ.get("TEUTONIC_KING_BENCH_OFFLINE_AFTER_PREFETCH", "true"),
        help="After task prefetch succeeds, run lm-eval with HF_HUB_OFFLINE/HF_DATASETS_OFFLINE. Set false to disable.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Forward --apply_chat_template to lm-eval",
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        help="Stop after the first failed benchmark instead of continuing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing run directory and skip benchmarks already marked completed.",
    )
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Comma-separated benchmark labels or task ids to run, e.g. 'MMLU,GSM8K' or 'mmlu,gsm8k'",
    )
    parser.add_argument(
        "--standardized-results-path",
        type=Path,
        default=None,
        help="Optional explicit path for the standardized machine-readable results JSON. Defaults to <run-dir>/standardized_results.json.",
    )
    args = parser.parse_args()
    args.model = args.model_repo or args.model
    if not args.model:
        parser.error("provide MODEL or --model-repo")
    return args


def parse_model_input(raw: str) -> tuple[str | None, str | None, str | None]:
    candidate = raw.strip()
    local_path = Path(candidate).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve()), None, None

    match = PINNED_REF_RE.match(candidate)
    if match:
        repo_id, revision = match.groups()
        return None, repo_id, revision

    if candidate.startswith("http://") or candidate.startswith("https://"):
        parsed = urlparse(candidate)
        host = parsed.netloc.lower()
        if host == "hub.hippius.com":
            path_match = HIPPIUS_MODEL_URL_RE.match(parsed.path.rstrip("/") + "/")
            if not path_match:
                raise ValueError(f"unsupported Hippius Hub URL shape: {candidate}")
            project, repo = path_match.groups()
            return None, f"{project}/{repo}", None
        if host == "huggingface.co":
            path_match = HF_MODEL_URL_RE.match(parsed.path.rstrip("/") + "/")
            if not path_match:
                raise ValueError(f"unsupported Hugging Face URL shape: {candidate}")
            org, repo, revision = path_match.groups()
            return None, f"{org}/{repo}", revision
        raise ValueError(f"unsupported model URL host: {parsed.netloc}")

    if "/" not in candidate:
        raise ValueError(f"expected local dir, Hub URL, or project/repo, got: {candidate!r}")
    return None, candidate, None


def hippius_artifact_list(repo_id: str) -> list[dict[str, Any]]:
    project, repo = repo_id.split("/", 1)
    url = f"{HIPPIUS_MODEL_API}/{project}/{repo}/"
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    artifacts = payload.get("artifacts") or []
    if not artifacts:
        raise RuntimeError(f"no public artifacts returned for {repo_id}")
    return artifacts


def hippius_artifact_detail(repo_id: str, reference: str) -> dict[str, Any]:
    project, repo = repo_id.split("/", 1)
    url = f"{HIPPIUS_MODEL_API}/{project}/{repo}/{reference}/"
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json()


def hub_primary_tag(repo_id: str, digest: str) -> str | None:
    project, repo = repo_id.split("/", 1)
    url = f"https://hub.hippius.com/models/{project}/{repo}"
    try:
        response = httpx.get(url, timeout=30.0, headers={"User-Agent": "teutonic-king-benchmark/1.0"})
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    html = response.text
    pos = html.find(digest)
    if pos < 0:
        return None
    window = html[pos:pos + 3000]
    patterns = [
        r'"href":"/models/' + re.escape(project) + r'/' + re.escape(repo) + r'/([^"/]+)"',
        r'\\"href\\":\\"/models/' + re.escape(project) + r'/' + re.escape(repo) + r'/([^\\"/]+)\\"',
    ]
    for pattern in patterns:
        match = re.search(pattern, window)
        if match:
            return match.group(1)
    return None


def resolve_remote_target(raw_model: str, repo_id: str, revision: str | None) -> ModelTarget:
    if revision and revision.startswith("hf:"):
        commit = revision.removeprefix("hf:")
        return ModelTarget(
            source=raw_model,
            local_path=None,
            repo_id=repo_id,
            revision=commit,
            digest=revision,
            primary_tag=commit,
            artifact={"source": "huggingface_revision", "hf_revision": commit},
        )

    artifact: dict[str, Any]
    if revision:
        try:
            artifact = hippius_artifact_detail(repo_id, revision)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404 or not revision.startswith("sha256:"):
                raise
            primary_tag = hub_primary_tag(repo_id, revision)
            artifact = {
                "digest": revision,
                "primary_tag": primary_tag,
                "source": "hippius_registry_digest_fallback",
                "api_error": f"{exc.response.status_code} {exc.response.reason_phrase}",
            }
    else:
        artifact = hippius_artifact_list(repo_id)[0]
    digest = (artifact.get("digest") or revision or "").strip() or None
    if not digest:
        raise RuntimeError(f"artifact lookup for {repo_id} did not return a digest")
    return ModelTarget(
        source=raw_model,
        local_path=None,
        repo_id=repo_id,
        revision=revision or digest,
        digest=digest,
        primary_tag=artifact.get("primary_tag"),
        artifact=artifact,
    )


def resolve_model_target(raw_model: str, revision_override: str | None) -> ModelTarget:
    local_path, repo_id, pinned_revision = parse_model_input(raw_model)
    if local_path:
        return ModelTarget(
            source=raw_model,
            local_path=local_path,
            repo_id=None,
            revision=None,
            digest=None,
            primary_tag=None,
            artifact=None,
        )
    assert repo_id is not None
    return resolve_remote_target(raw_model, repo_id, revision_override or pinned_revision)


def find_model_root(path: Path) -> Path:
    if (path / "config.json").exists():
        return path
    candidates = [p.parent for p in path.rglob("config.json") if p.is_file()]
    if not candidates:
        return path
    candidates.sort(key=lambda p: (not (p / "model.safetensors.index.json").exists(), len(str(p))))
    return candidates[0]


def ensure_safetensors_index(path: Path) -> None:
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        return
    shards = sorted(p for p in path.glob("model-*.safetensors") if p.is_file())
    if len(shards) <= 1:
        return
    try:
        from safetensors import safe_open
    except Exception:
        return
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shards:
        total_size += shard.stat().st_size
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = shard.name
    if weight_map:
        write_json(index_path, {"metadata": {"total_size": total_size}, "weight_map": weight_map})


def is_usable_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    ensure_safetensors_index(path)
    has_single_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    has_sharded_weights = (path / "model.safetensors.index.json").exists() or (path / "pytorch_model.bin.index.json").exists()
    has_weights = has_single_weights or has_sharded_weights
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    return has_weights and has_tokenizer


def accept_existing_snapshot(target: ModelTarget, snapshot_dir: Path | None, original_error: Exception) -> ModelTarget | None:
    if snapshot_dir is None:
        return None
    model_root = find_model_root(snapshot_dir)
    if not is_usable_model_dir(model_root):
        return None
    target.local_path = str(model_root.resolve())
    artifact = dict(target.artifact or {})
    artifact.update({"source": "hippius_partial_snapshot_fallback", "hippius_error": repr(original_error)})
    target.artifact = artifact
    return target


def materialize_hippius_docker(target: ModelTarget, snapshot_dir: Path | None, original_error: Exception) -> ModelTarget:
    if not target.repo_id or not (target.revision or "").startswith("sha256:"):
        raise original_error
    docker_bin = shutil.which("docker")
    if not docker_bin:
        raise original_error
    download_dir = snapshot_dir or Path(tempfile.mkdtemp(prefix="teutonic-oci-model-"))
    download_dir.mkdir(parents=True, exist_ok=True)
    image_ref = f"registry.hippius.com/{target.repo_id}@{target.revision}"
    subprocess.run([docker_bin, "pull", image_ref], check=True)
    container_id = subprocess.check_output([docker_bin, "create", image_ref], text=True).strip()
    try:
        subprocess.run([docker_bin, "cp", f"{container_id}:/.", str(download_dir)], check=True)
    finally:
        subprocess.run([docker_bin, "rm", container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    model_root = find_model_root(download_dir)
    target.local_path = str(model_root.resolve())
    artifact = dict(target.artifact or {})
    artifact.update({"source": "hippius_docker_fallback", "image_ref": image_ref, "hippius_error": repr(original_error)})
    target.artifact = artifact
    return target


def env_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def hf_search_repo_ids(repo_name: str) -> list[str]:
    if HfApi is None or not env_enabled("TEUTONIC_KING_BENCH_HF_FALLBACK_SEARCH", True):
        return []
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        matches = api.list_models(search=repo_name, limit=int(os.environ.get("TEUTONIC_KING_BENCH_HF_FALLBACK_SEARCH_LIMIT", "25")))
    except Exception:
        return []
    out: list[str] = []
    wanted = repo_name.lower()
    for item in matches:
        model_id = getattr(item, "modelId", None) or getattr(item, "id", None)
        if not model_id or "/" not in model_id:
            continue
        if model_id.rsplit("/", 1)[-1].lower() == wanted:
            out.append(model_id)
    return out


def hf_fallback_repo_ids(repo_id: str) -> list[str]:
    org, repo = repo_id.split("/", 1)
    candidates: list[str] = []
    raw_repos = os.environ.get("TEUTONIC_KING_BENCH_HF_FALLBACK_REPOS", "")
    for item in raw_repos.split(","):
        item = item.strip()
        if not item:
            continue
        candidates.append(item.format(repo=repo, org=org, repo_id=repo_id))
    candidates.append(repo_id)
    raw_orgs = os.environ.get("TEUTONIC_KING_BENCH_HF_FALLBACK_ORGS", "")
    for item in raw_orgs.split(","):
        item = item.strip()
        if item:
            candidates.append(f"{item}/{repo}")
    candidates.extend(hf_search_repo_ids(repo))
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def materialize_hf_fallback(target: ModelTarget, snapshot_dir: Path | None, original_error: Exception) -> ModelTarget:
    if hf_snapshot_download is None:
        raise original_error
    assert target.repo_id
    hf_revision = None if (target.revision or "").startswith("sha256:") else target.revision
    errors: list[str] = []
    base_dir = snapshot_dir.parent if snapshot_dir is not None else None
    for repo_id in hf_fallback_repo_ids(target.repo_id):
        local_dir = (base_dir / f"hf-{slugify(repo_id)}") if base_dir is not None else None
        try:
            local_path = hf_snapshot_download(
                repo_id=repo_id,
                revision=hf_revision,
                local_dir=str(local_dir) if local_dir else None,
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                max_workers=8,
            )
            target.local_path = str(Path(local_path).resolve())
            target.repo_id = repo_id
            target.revision = hf_revision
            target.digest = target.digest or None
            target.primary_tag = target.primary_tag or hf_revision
            artifact = dict(target.artifact or {})
            artifact.update({"source": "huggingface_fallback", "hf_repo_id": repo_id, "hippius_error": repr(original_error)})
            target.artifact = artifact
            return target
        except Exception as exc:
            errors.append(f"{repo_id}: {exc!r}")
    raise RuntimeError("Hippius download failed and Hugging Face fallbacks also failed: " + "; ".join(errors)) from original_error


def materialize_target(target: ModelTarget, snapshot_dir: Path | None) -> ModelTarget:
    if target.local_path:
        return target
    assert target.repo_id and target.revision
    download_dir = snapshot_dir
    if download_dir is not None:
        download_dir.mkdir(parents=True, exist_ok=True)
    if target.digest and target.digest.startswith("hf:"):
        return materialize_hf_fallback(target, snapshot_dir, RuntimeError("Hugging Face revision target"))
    try:
        local_path = hippius_snapshot_download(
            repo_id=target.repo_id,
            revision=target.revision,
            local_dir=str(download_dir) if download_dir else None,
            max_workers=8,
        )
        target.local_path = str(find_model_root(Path(local_path)).resolve())
        return target
    except Exception as exc:
        existing = accept_existing_snapshot(target, snapshot_dir, exc)
        if existing is not None:
            return existing
        try:
            return materialize_hippius_docker(target, snapshot_dir, exc)
        except Exception as docker_exc:
            existing = accept_existing_snapshot(target, snapshot_dir, docker_exc)
            if existing is not None:
                return existing
            return materialize_hf_fallback(target, snapshot_dir, docker_exc)


def resolve_lm_eval_command(cli_value: str) -> list[str]:
    if cli_value.strip():
        return shlex.split(cli_value)

    for path in (shutil.which("lm-eval"), str(ROOT / ".venv" / "bin" / "lm-eval")):
        if path and Path(path).exists():
            return [path]

    candidates = [Path(sys.executable), ROOT / ".venv" / "bin" / "python"]
    for python_bin in candidates:
        if not python_bin.exists():
            continue
        probe = subprocess.run(
            [str(python_bin), "-c", "import lm_eval"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return [str(python_bin), "-m", "lm_eval"]

    raise RuntimeError(
        "lm-eval-harness is not installed. Install it first, for example:\n"
        "  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness\n"
        "  cd lm-evaluation-harness && pip install -e '.[math,sentencepiece]'\n"
        "Then rerun with --lm-eval-bin '/path/to/lm-eval' if needed."
    )


def should_prefetch_tasks(raw_value: str) -> bool:
    value = str(raw_value or "").strip().lower()
    return value not in {"", "0", "false", "no", "off", "none"}


def prefetch_task(task_name: str, timeout_s: int, retries: int) -> None:
    """Warm the datasets/task cache before accelerate fans out to many ranks."""
    code = """
import itertools
import sys
from lm_eval.tasks import TaskManager

task = sys.argv[1]
manager = TaskManager()
loaded = manager.load([task])

def walk(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from walk(value)
        return
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from walk(value)
        return
    yield obj

for task_obj in walk(loaded):
    if task_obj is None:
        continue
    download = getattr(task_obj, "download", None)
    if callable(download):
        download()
    for method_name in ("training_docs", "validation_docs", "test_docs"):
        method = getattr(task_obj, method_name, None)
        if not callable(method):
            continue
        docs = method()
        if docs is None:
            continue
        for _ in docs:
            pass
    fewshot = getattr(task_obj, "fewshot_docs", None)
    if callable(fewshot):
        try:
            for _ in itertools.islice(fewshot(), 256):
                pass
        except TypeError:
            pass
print(f"[prefetch] loaded {task}", flush=True)
"""
    attempts = max(1, int(retries))
    last_returncode: int | None = None
    for attempt in range(1, attempts + 1):
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, task_name],
                cwd=str(ROOT),
                env=env,
                check=False,
                timeout=max(1, int(timeout_s)),
            )
        except subprocess.TimeoutExpired as exc:
            if attempt >= attempts:
                raise RuntimeError(f"timed out prefetching lm-eval task {task_name!r} after {timeout_s}s") from exc
            print(f"[prefetch] retrying {task_name} after timeout ({attempt}/{attempts})", flush=True)
            time.sleep(min(60, 5 * attempt))
            continue
        if proc.returncode == 0:
            return
        last_returncode = proc.returncode
        if attempt < attempts:
            print(f"[prefetch] retrying {task_name} after returncode {proc.returncode} ({attempt}/{attempts})", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"failed to prefetch lm-eval task {task_name!r} (returncode {last_returncode})")


def prefetch_tasks(specs: list[BenchmarkSpec], timeout_s: int, retries: int) -> None:
    seen: set[str] = set()
    for spec in specs:
        if spec.task in seen:
            continue
        seen.add(spec.task)
        print(f"[prefetch] loading task cache for {spec.label} ({spec.task})", flush=True)
        prefetch_task(spec.task, timeout_s, retries)


def set_offline_cache_env(env: dict[str, str]) -> None:
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def find_results_json(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.rglob("results_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            run_dir.rglob("results.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"lm-eval did not produce a results json under {run_dir}")
    return candidates[0]


def extract_task_node(results_json: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    for container_name in ("results", "groups"):
        container = results_json.get(container_name)
        if isinstance(container, dict):
            node = container.get(task_name)
            if isinstance(node, dict):
                return node
    return None


def extract_metric(
    results_json: dict[str, Any], spec: BenchmarkSpec
) -> tuple[float | None, str | None, dict[str, Any] | None]:
    node = extract_task_node(results_json, spec.task)
    if not node:
        return None, None, None
    for key in spec.metric_candidates:
        value = node.get(key)
        if isinstance(value, (int, float)):
            return float(value), key, node
    for key, value in node.items():
        if key.startswith(("acc", "exact_match", "pass@1", "pass_at_1", "mc2")) and isinstance(value, (int, float)):
            return float(value), key, node
    return None, None, node


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def summarize_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "requested": len(rows),
        "completed": 0,
        "completed_no_metric": 0,
        "failed": 0,
    }
    for row in rows:
        status = row.get("status")
        if status == "completed":
            counts["completed"] += 1
        elif status == "completed_no_metric":
            counts["completed_no_metric"] += 1
        else:
            counts["failed"] += 1
    return counts


def standardized_results_payload(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    benchmarks = []
    for row in summary.get("benchmarks", []):
        benchmarks.append(
            {
                "name": row.get("label"),
                "task": row.get("task"),
                "fewshot": row.get("fewshot"),
                "status": row.get("status"),
                "metric": {
                    "name": row.get("metric_key"),
                    "value": row.get("score"),
                },
                "returncode": row.get("returncode"),
                "wall_time_s": row.get("wall_time_s"),
                "results_json": row.get("results_json"),
                "available_metrics": row.get("available_metrics"),
            }
        )

    return {
        "schema_version": "king-benchmark-results.v1",
        "generated_at": utcnow_iso(),
        "run_id": summary.get("run_id"),
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "model": {
            "source": summary.get("model", {}).get("source"),
            "repo_id": summary.get("model", {}).get("repo_id"),
            "revision": summary.get("model", {}).get("revision"),
            "digest": summary.get("model", {}).get("digest"),
            "primary_tag": summary.get("model", {}).get("primary_tag"),
            "local_path": summary.get("model", {}).get("local_path"),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "summary_json": str(run_dir / "summary.json"),
            "summary_markdown": str(run_dir / "SUMMARY.md"),
            "standardized_results_json": str(run_dir / "standardized_results.json"),
        },
        "totals": summarize_status_counts(summary.get("benchmarks", [])),
        "benchmarks": benchmarks,
    }


def write_standardized_results(path: Path, summary: dict[str, Any], run_dir: Path) -> None:
    payload = standardized_results_payload(summary, run_dir)
    payload["artifacts"]["standardized_results_json"] = str(path)
    write_json(path, payload)


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# King benchmark run — {summary['run_id']}",
        "",
        f"- Started: `{summary['started_at']}`",
        f"- Finished: `{summary['finished_at']}`",
        f"- Model source: `{summary['model']['source']}`",
        f"- Local snapshot: `{summary['model']['local_path']}`",
    ]
    if summary["model"].get("repo_id"):
        lines.append(f"- Repo: `{summary['model']['repo_id']}`")
    if summary["model"].get("revision"):
        lines.append(f"- Revision: `{summary['model']['revision']}`")
    if summary["model"].get("digest"):
        lines.append(f"- Digest: `{summary['model']['digest']}`")
    lines += [
        "",
        "| Benchmark | Task | Few-shot | Status | Metric | Score |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in summary["benchmarks"]:
        score = row["score"]
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else ""
        lines.append(
            f"| {row['label']} | `{row['task']}` | {row['fewshot']} | {row['status']} | "
            f"`{row.get('metric_key') or ''}` | {score_text} |"
        )
    path.write_text("\n".join(lines) + "\n")


def select_benchmarks(raw_value: str) -> list[BenchmarkSpec]:
    if not raw_value.strip():
        return list(BENCHMARKS)
    wanted = {part.strip().lower() for part in raw_value.split(",") if part.strip()}
    selected = [
        spec for spec in BENCHMARKS
        if spec.label.lower() in wanted or spec.task.lower() in wanted
    ]
    if not selected:
        raise ValueError(f"no benchmarks matched --benchmarks={raw_value!r}")
    return selected


def parse_fewshot_overrides(raw_value: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if not raw_value.strip():
        return overrides
    for part in raw_value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --fewshot-overrides item {item!r}; expected BENCHMARK=N")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"invalid few-shot value for {key!r}: {value!r}") from exc
        if parsed < 0:
            raise ValueError(f"few-shot override for {key!r} must be non-negative")
        overrides[key] = parsed
    return overrides


def apply_fewshot_overrides(
    specs: list[BenchmarkSpec],
    overrides: dict[str, int],
) -> list[BenchmarkSpec]:
    if not overrides:
        return specs
    selected: list[BenchmarkSpec] = []
    unmatched = set(overrides)
    for spec in specs:
        label_key = spec.label.lower()
        task_key = spec.task.lower()
        value = overrides.get(label_key, overrides.get(task_key))
        if value is None:
            selected.append(spec)
            continue
        unmatched.discard(label_key)
        unmatched.discard(task_key)
        selected.append(
            BenchmarkSpec(
                spec.label,
                spec.task,
                value,
                spec.metric_candidates,
                spec.unsafe_code,
            )
        )
    if unmatched:
        raise ValueError(f"few-shot override did not match a selected benchmark: {', '.join(sorted(unmatched))}")
    return selected


def load_existing_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def prepare_lm_eval_paths(model_path: str, spec: BenchmarkSpec) -> tuple[Path, Path]:
    temp_root = Path(tempfile.mkdtemp(prefix=f"king-bench-{os.getpid()}-{slugify(spec.label)}-"))
    short_model = temp_root / "model"
    short_output = temp_root / "out"
    short_output.mkdir(parents=True, exist_ok=True)
    try:
        short_model.symlink_to(Path(model_path).resolve(), target_is_directory=True)
    except OSError:
        # Fall back to the original path if the filesystem refuses symlinks.
        short_model = Path(model_path)
    return short_model, short_output


def copy_lm_eval_outputs(source_dir: Path, dest_dir: Path) -> None:
    dest = dest_dir / "lm-eval-output"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_dir, dest, dirs_exist_ok=True)


def benchmark_command(
    lm_eval_cmd: list[str],
    model_path: str,
    spec: BenchmarkSpec,
    run_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    model_args = [
        f"pretrained={model_path}",
        f"dtype={args.dtype}",
        "trust_remote_code=False",
    ]
    if args.model_args_extra.strip():
        for item in args.model_args_extra.strip().lstrip(",").split(","):
            item = item.strip()
            if item:
                model_args.append(item)

    cmd = [
        *lm_eval_cmd,
        "run",
        "--model",
        "hf",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        spec.task,
        "--num_fewshot",
        str(spec.fewshot),
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(run_dir),
    ]
    if args.cache_requests.strip():
        cmd += ["--cache_requests", args.cache_requests.strip()]
    if args.response_cache_root is not None:
        cache_id = hashlib.sha256(f"{model_path}|{spec.task}|{spec.fewshot}".encode("utf-8")).hexdigest()[:24]
        cache_prefix = args.response_cache_root / slugify(spec.label) / f"{cache_id}_"
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--use_cache", str(cache_prefix)]
    if str(args.max_batch_size).strip():
        cmd += ["--max_batch_size", str(args.max_batch_size).strip()]
    if args.device and args.device.lower() not in {"auto", "none", ""}:
        cmd += ["--device", args.device]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.gen_kwargs.strip():
        cmd += ["--gen_kwargs", args.gen_kwargs.strip()]
    if args.apply_chat_template:
        cmd.append("--apply_chat_template")
    if args.log_samples:
        cmd.append("--log_samples")
    if spec.unsafe_code:
        cmd.append("--confirm_run_unsafe_code")
    return cmd


def run_one_benchmark(
    lm_eval_cmd: list[str],
    model_path: str,
    spec: BenchmarkSpec,
    bench_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    bench_dir.mkdir(parents=True, exist_ok=True)
    lm_model_path, lm_output_dir = prepare_lm_eval_paths(model_path, spec)
    cmd = benchmark_command(lm_eval_cmd, str(lm_model_path), spec, lm_output_dir, args)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if getattr(args, "_offline_after_prefetch_ready", False):
        set_offline_cache_env(env)
    if spec.unsafe_code:
        env["HF_ALLOW_CODE_EVAL"] = "1"

    started = time.time()
    print(f"[{spec.label}] running: {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    wall = round(time.time() - started, 2)

    row: dict[str, Any] = {
        "label": spec.label,
        "task": spec.task,
        "fewshot": spec.fewshot,
        "command": cmd,
        "status": "failed" if proc.returncode else "completed",
        "returncode": proc.returncode,
        "wall_time_s": wall,
        "results_json": None,
        "metric_key": None,
        "score": None,
        "available_metrics": None,
    }
    if proc.returncode != 0:
        try:
            copy_lm_eval_outputs(lm_output_dir, bench_dir)
        except Exception:
            pass
        return row

    copy_lm_eval_outputs(lm_output_dir, bench_dir)
    results_path = find_results_json(bench_dir / "lm-eval-output")
    results_json = json.loads(results_path.read_text())
    score, metric_key, node = extract_metric(results_json, spec)
    row["results_json"] = str(results_path)
    row["metric_key"] = metric_key
    row["score"] = score
    row["available_metrics"] = sorted(node.keys()) if isinstance(node, dict) else None
    if score is None:
        row["status"] = "completed_no_metric"
    return row


def main() -> int:
    args = parse_args()
    started_at = utcnow_iso()
    selected_benchmarks = apply_fewshot_overrides(
        select_benchmarks(args.benchmarks),
        parse_fewshot_overrides(args.fewshot_overrides),
    )

    target = resolve_model_target(args.model, args.revision)

    run_id = (
        args.run_dir.expanduser().resolve().name
        if args.run_dir
        else utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + slugify(
            target.repo_id or Path(target.local_path or target.source).name
        )
    )
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir
        else args.runs_dir.expanduser().resolve() / run_id
    )
    snapshot_dir = (
        args.snapshot_dir.expanduser().resolve()
        if args.snapshot_dir
        else run_dir / "model"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    standardized_results_path = (
        args.standardized_results_path.expanduser().resolve()
        if args.standardized_results_path
        else run_dir / "standardized_results.json"
    )
    existing_summary = load_existing_summary(summary_path) if args.resume else None

    target = materialize_target(target, snapshot_dir if not target.local_path else None)

    if existing_summary:
        summary = existing_summary
        summary["model"] = asdict(target)
    else:
        summary = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": None,
            "model": asdict(target),
            "benchmarks": [],
        }
    write_json(run_dir / "summary.json", summary)
    write_standardized_results(standardized_results_path, summary, run_dir)

    if args.download_only:
        summary["finished_at"] = utcnow_iso()
        write_json(run_dir / "summary.json", summary)
        write_markdown_summary(run_dir / "SUMMARY.md", summary)
        write_standardized_results(standardized_results_path, summary, run_dir)
        print(f"[done] downloaded snapshot to {target.local_path}")
        print(f"[done] wrote summary to {run_dir / 'summary.json'}")
        print(f"[done] wrote standardized results to {standardized_results_path}")
        return 0

    lm_eval_cmd = resolve_lm_eval_command(args.lm_eval_bin)
    if should_prefetch_tasks(args.prefetch_tasks):
        prefetch_tasks(selected_benchmarks, args.prefetch_timeout_s, args.prefetch_retries)
        args._offline_after_prefetch_ready = should_prefetch_tasks(args.offline_after_prefetch)
    else:
        args._offline_after_prefetch_ready = False
    failures = 0
    assert target.local_path is not None
    existing_rows_by_task = {
        row.get("task"): row
        for row in summary.get("benchmarks", [])
        if isinstance(row, dict) and row.get("task")
    }

    for spec in selected_benchmarks:
        prior_row = existing_rows_by_task.get(spec.task)
        if args.resume and prior_row and prior_row.get("status") == "completed":
            print(f"[{spec.label}] skipping completed benchmark from existing summary", flush=True)
            continue
        row = run_one_benchmark(
            lm_eval_cmd=lm_eval_cmd,
            model_path=target.local_path,
            spec=spec,
            bench_dir=run_dir / "benchmarks" / slugify(spec.label),
            args=args,
        )
        if prior_row and prior_row in summary["benchmarks"]:
            idx = summary["benchmarks"].index(prior_row)
            summary["benchmarks"][idx] = row
        else:
            summary["benchmarks"].append(row)
        summary["finished_at"] = utcnow_iso()
        write_json(run_dir / "summary.json", summary)
        write_standardized_results(standardized_results_path, summary, run_dir)
        if row["status"] != "completed":
            failures += 1
            if args.exit_on_error:
                break

    summary["finished_at"] = utcnow_iso()
    write_json(run_dir / "summary.json", summary)
    write_markdown_summary(run_dir / "SUMMARY.md", summary)
    write_standardized_results(standardized_results_path, summary, run_dir)

    print(f"[done] run directory: {run_dir}")
    print(f"[done] summary json: {run_dir / 'summary.json'}")
    print(f"[done] summary md: {run_dir / 'SUMMARY.md'}")
    print(f"[done] standardized results json: {standardized_results_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
