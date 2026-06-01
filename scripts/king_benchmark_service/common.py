#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARKS = ["MMLU", "MMLU-Pro", "BBH", "ARC-C", "TruthfulQA", "WinoGrande"]
DASHBOARD_URLS = [
    "https://us-east-1.hippius.com/teutonic-sn3/dashboard.json",
    "https://eu-central-1.hippius.com/teutonic-sn3/dashboard.json",
    "https://s3.hippius.com/teutonic-sn3/dashboard.json",
]
SCHEMA_VERSION = "teutonic-king-benchmark-index.v1"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-") or "unknown"


def digest_short(digest: str | None) -> str:
    if not digest:
        return "no-digest"
    return slugify(digest.replace("sha256:", "sha256-"))[:80]


def king_id(model_repo: str, digest: str | None) -> str:
    return f"{slugify(model_repo)}-{digest_short(digest)}"


def model_input(model_repo: str, digest: str | None) -> str:
    if digest and str(digest).startswith(("sha256:", "hf:")):
        return f"{model_repo}@{digest}"
    return model_repo


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def bearer_token(headers: dict[str, str]) -> str:
    value = headers.get("authorization") or headers.get("Authorization") or ""
    prefix = "Bearer "
    return value[len(prefix):].strip() if value.startswith(prefix) else ""


def require_bearer(headers: dict[str, str], expected: str) -> bool:
    return bool(expected) and bearer_token(headers) == expected


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Any | None = None,
    token: str | None = None,
    timeout_s: int = 30,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "teutonic-king-benchmark-service/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def fetch_dashboard(urls: list[str] | None = None, timeout_s: int = 20) -> tuple[str, dict[str, Any]]:
    errors: list[str] = []
    for url in urls or DASHBOARD_URLS:
        try:
            payload = http_json(url, timeout_s=timeout_s)
            if isinstance(payload, dict) and isinstance(payload.get("king"), dict):
                return url, payload
            errors.append(f"{url}: missing king")
        except Exception as exc:
            errors.append(f"{url}: {exc!r}")
    raise RuntimeError("failed to fetch dashboard: " + "; ".join(errors))


def normalize_king(row: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    model_repo = row.get("model_repo") or row.get("challenger_repo") or row.get("repo") or row.get("hf_repo")
    digest = row.get("king_digest") or row.get("king_revision") or row.get("challenger_digest") or row.get("digest")
    if not model_repo:
        return None
    return {
        "king_id": king_id(model_repo, digest),
        "source": source,
        "model_repo": model_repo,
        "king_digest": digest,
        "model_input": model_input(model_repo, digest),
        "hotkey": row.get("hotkey"),
        "reign_number": row.get("reign_number"),
        "crowned_at": row.get("crowned_at") or row.get("timestamp"),
        "crowned_block": row.get("crowned_block"),
        "challenge_id": row.get("challenge_id"),
    }


def kings_from_dashboard(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates: list[tuple[str, dict[str, Any]]] = []
    if isinstance(dashboard.get("king"), dict):
        candidates.append(("current_king", dashboard["king"]))
    for row in dashboard.get("king_chain") or []:
        if isinstance(row, dict):
            candidates.append(("king_chain", row))
    for row in dashboard.get("history") or []:
        if isinstance(row, dict) and row.get("accepted") is True:
            candidates.append(("accepted_history", row))
    for source, row in candidates:
        king = normalize_king(row, source=source)
        if not king or king["king_id"] in seen:
            continue
        seen.add(king["king_id"])
        out.append(king)
    return out


def benchmark_statuses(payload: dict[str, Any] | None) -> dict[str, str]:
    statuses: dict[str, str] = {}
    if not isinstance(payload, dict):
        return statuses
    for row in payload.get("benchmarks") or []:
        if isinstance(row, dict) and row.get("name"):
            statuses[row["name"]] = row.get("status") or "unknown"
    return statuses


def missing_benchmarks(result_payload: dict[str, Any] | None, desired: list[str]) -> list[str]:
    statuses = benchmark_statuses(result_payload)
    terminal = {"completed", "missing"}
    return [name for name in desired if statuses.get(name) not in terminal]
