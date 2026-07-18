#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import socket
import struct
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None
    BotoConfig = None

from common import DEFAULT_BENCHMARKS, SCHEMA_VERSION, fetch_dashboard, http_json, kings_from_dashboard, missing_benchmarks, read_json, require_bearer, utcnow_iso, write_json


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def worker_ws_url(worker_url: str) -> tuple[str, int, str]:
    from urllib.parse import urlparse
    parsed = urlparse(worker_url.rstrip("/"))
    if parsed.scheme not in {"http", "ws"}:
        raise ValueError(f"unsupported worker websocket scheme: {parsed.scheme}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (80 if parsed.scheme in {"http", "ws"} else 443)
    base_path = parsed.path.rstrip("/")
    path = (base_path + "/events") if base_path else "/events"
    return host, port, path


def websocket_connect(host: str, port: int, path: str, token: str, timeout_s: int = 20) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout_s)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Authorization: Bearer {token}\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("websocket handshake closed")
        response += chunk
        if len(response) > 65536:
            raise ConnectionError("websocket handshake too large")
    header = response.split(b"\r\n\r\n", 1)[0].decode("iso-8859-1", errors="replace")
    if " 101 " not in header.split("\r\n", 1)[0]:
        raise ConnectionError(f"websocket handshake failed: {header.splitlines()[0] if header else 'empty response'}")
    sock.settimeout(None)
    return sock


def websocket_recv_text(sock: socket.socket) -> str | None:
    header = _recv_exact(sock, 2)
    if not header:
        return None
    b1, b2 = header
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    if opcode == 8:
        return None
    if opcode in {9, 10}:
        return ""
    if opcode != 1:
        return ""
    return payload.decode("utf-8")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("websocket closed")
        data.extend(chunk)
    return bytes(data)


class Store:
    def __init__(self, root: Path, bucket: str, endpoint: str, latest_key: str, index_key: str, history_key: str) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket = bucket
        self.endpoint = endpoint
        self.latest_key = latest_key
        self.index_key = index_key
        self.history_key = history_key
        self.latest_path = root / "latest.json"
        self.index_path = root / "kings_index.json"
        self.history_path = root / "history.jsonl"

    def s3_client(self):
        if boto3 is None or BotoConfig is None:
            return None
        access = os.environ.get("TEUTONIC_HIPPIUS_ACCESS_KEY") or os.environ.get("HIPPIUS_ACCESS_KEY")
        secret = os.environ.get("TEUTONIC_HIPPIUS_SECRET_KEY") or os.environ.get("HIPPIUS_SECRET_KEY")
        if not access or not secret:
            return None
        return boto3.client("s3", endpoint_url=self.endpoint, aws_access_key_id=access, aws_secret_access_key=secret, region_name="decentralized", config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}))

    def put_s3(self, key: str, path: Path, content_type: str = "application/json") -> bool:
        client = self.s3_client()
        if client is None:
            return False
        try:
            client.put_object(Bucket=self.bucket, Key=key, Body=path.read_bytes(), ContentType=content_type)
        except Exception as exc:
            print(f"[s3] failed to upload {key}: {exc!r}", file=sys.stderr, flush=True)
            return False
        return True

    def result_key(self, king_id: str) -> str:
        return f"king-benchmark-daily/kings/{king_id}/results.json"

    def load_index(self) -> dict[str, Any]:
        payload = read_json(self.index_path, None)
        if payload is not None:
            return payload
        client = self.s3_client()
        if client is not None:
            try:
                obj = client.get_object(Bucket=self.bucket, Key=self.index_key)
                payload = json.loads(obj["Body"].read().decode("utf-8"))
                write_json(self.index_path, payload)
                return payload
            except Exception:
                pass
        return {"schema_version": SCHEMA_VERSION, "updated_at": None, "kings": {}}

    def write_index(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = utcnow_iso()
        write_json(self.index_path, payload)
        self.put_s3(self.index_key, self.index_path)

    def write_latest(self, payload: dict[str, Any]) -> None:
        write_json(self.latest_path, payload)
        self.put_s3(self.latest_key, self.latest_path)

    def write_result(self, king_id: str, payload: dict[str, Any]) -> None:
        path = self.root / "kings" / king_id / "results.json"
        write_json(path, payload)
        self.put_s3(self.result_key(king_id), path)

    def append_history(self, payload: dict[str, Any]) -> None:
        with self.history_path.open("a") as fh:
            fh.write(json.dumps(payload, sort_keys=False) + "\n")
        self.put_s3(self.history_key, self.history_path, "application/x-ndjson")

    def remove_failed_history_for_king(self, king_id: str) -> None:
        if not self.history_path.exists():
            return
        kept: list[str] = []
        changed = False
        for line in self.history_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            payload_king = ((payload.get("model") or {}).get("king_id"))
            if payload_king == king_id and payload.get("status") == "failed":
                changed = True
                continue
            kept.append(json.dumps(payload, sort_keys=False))
        if changed:
            self.history_path.write_text(("\n".join(kept) + "\n") if kept else "")
            self.put_s3(self.history_key, self.history_path, "application/x-ndjson")


class ControllerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.store = Store(args.results_root, args.hippius_bucket, args.hippius_endpoint, args.latest_key, args.index_key, args.history_key)
        self.state_path = args.results_root / "controller_state.json"
        self.state = read_json(self.state_path, {"status": "idle", "updated_at": utcnow_iso()})

    def save(self) -> None:
        self.state["updated_at"] = utcnow_iso()
        write_json(self.state_path, self.state)

    def max_retries(self) -> int:
        return max(0, int(self.args.max_retries))

    def retry_count(self, king_id: str | None) -> int:
        if not king_id:
            return 0
        retry_counts = self.state.get("retry_counts")
        if not isinstance(retry_counts, dict):
            return 0
        try:
            return int(retry_counts.get(king_id) or 0)
        except (TypeError, ValueError):
            return 0

    def retry_exhausted(self, king_id: str | None) -> bool:
        max_retries = self.max_retries()
        return bool(max_retries and self.retry_count(king_id) >= max_retries)

    def note_final_status(self, king_id: str | None, status: str | None) -> None:
        if not king_id:
            return
        retry_counts = self.state.setdefault("retry_counts", {})
        if not isinstance(retry_counts, dict):
            retry_counts = {}
            self.state["retry_counts"] = retry_counts
        if status in {"failed", "missing"}:
            retry_counts[king_id] = self.retry_count(king_id) + 1
            self.state["last_retry_update"] = {
                "king_id": king_id,
                "status": status,
                "count": retry_counts[king_id],
                "max": self.max_retries(),
                "at": utcnow_iso(),
            }
        elif status == "completed":
            retry_counts.pop(king_id, None)

    def queue_benchmarks(self, result_payload: dict[str, Any] | None) -> list[str]:
        statuses: dict[str, str] = {}
        if isinstance(result_payload, dict):
            for row in result_payload.get("benchmarks") or []:
                if isinstance(row, dict) and row.get("name"):
                    statuses[row["name"]] = row.get("status") or "unknown"
        terminal = {"completed", "missing"}
        return [name for name in self.args.benchmarks if statuses.get(name) not in terminal]

    def build_queue(self) -> list[dict[str, Any]]:
        _, dashboard = fetch_dashboard(self.args.dashboard_url, self.args.dashboard_timeout_s)
        kings = kings_from_dashboard(dashboard)
        index = self.store.load_index()
        queued: list[dict[str, Any]] = []
        skipped_retry_exhausted: list[str] = []
        for king in kings:
            if self.retry_exhausted(king.get("king_id")):
                skipped_retry_exhausted.append(king["king_id"])
                continue
            record = (index.get("kings") or {}).get(king["king_id"], {})
            local_result = read_json(self.args.results_root / "kings" / king["king_id"] / "results.json", None)
            latest = record.get("latest_result")
            result = self.best_result(local_result, latest if isinstance(latest, dict) else None)
            missing = self.queue_benchmarks(result)
            if missing:
                queued.append({"king": king, "missing_benchmarks": missing})
        if skipped_retry_exhausted:
            self.state["last_retry_exhausted"] = {
                "count": len(skipped_retry_exhausted),
                "king_ids": skipped_retry_exhausted[:20],
                "at": utcnow_iso(),
            }
        return queued

    def next_dispatch_number(self) -> int:
        try:
            return int(self.state.get("dispatch_count") or 0) + 1
        except (TypeError, ValueError):
            return 1

    def should_run_periodic_mmlu_pro(self) -> bool:
        interval = max(0, int(self.args.mmlu_pro_interval))
        return bool(interval and self.next_dispatch_number() % interval == 0)

    def benchmark_status(self, result_payload: dict[str, Any] | None, name: str) -> str | None:
        if not isinstance(result_payload, dict):
            return None
        for row in result_payload.get("benchmarks") or []:
            if isinstance(row, dict) and row.get("name") == name:
                return row.get("status") or "unknown"
        return None

    def result_quality(self, result_payload: dict[str, Any] | None) -> tuple[int, int, int]:
        if not isinstance(result_payload, dict):
            return (-1, -1, -1)
        rows = [row for row in (result_payload.get("benchmarks") or []) if isinstance(row, dict)]
        standard_rows = [row for row in rows if row.get("name") != "MMLU-Pro"]
        completed_standard = sum(1 for row in standard_rows if row.get("status") == "completed")
        completed_total = sum(1 for row in rows if row.get("status") == "completed")
        return (completed_standard, len(standard_rows), completed_total)

    def best_result(self, *results: dict[str, Any] | None) -> dict[str, Any] | None:
        candidates = [result for result in results if isinstance(result, dict)]
        if not candidates:
            return None
        return max(candidates, key=self.result_quality)

    def standard_benchmarks_completed(self, result_payload: dict[str, Any] | None) -> bool:
        if not isinstance(result_payload, dict):
            return False
        statuses = {
            row.get("name"): row.get("status")
            for row in (result_payload.get("benchmarks") or [])
            if isinstance(row, dict) and row.get("name")
        }
        standard = [name for name in self.args.benchmarks if name != "MMLU-Pro"]
        return bool(standard) and all(statuses.get(name) == "completed" for name in standard)

    def mmlu_pro_catchup_item(self) -> dict[str, Any] | None:
        _, dashboard = fetch_dashboard(self.args.dashboard_url, self.args.dashboard_timeout_s)
        for king in kings_from_dashboard(dashboard):
            if self.retry_exhausted(king.get("king_id")):
                continue
            result = self.result_for_king(king.get("king_id"))
            if not self.standard_benchmarks_completed(result):
                continue
            status = self.benchmark_status(result, "MMLU-Pro")
            if status != "completed":
                return {"king": king, "missing_benchmarks": ["MMLU-Pro"]}
        return None

    def select_dispatch_item(self, queue: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], str]:
        if self.should_run_periodic_mmlu_pro():
            item = self.mmlu_pro_catchup_item()
            if item is not None:
                return item, ["MMLU-Pro"], "periodic_mmlu_pro"
        item = queue[0]
        benchmarks = list(item["missing_benchmarks"])
        if "MMLU-Pro" in benchmarks and len(benchmarks) > 1:
            benchmarks = [name for name in benchmarks if name != "MMLU-Pro"]
        return item, benchmarks, "standard"

    def current_dashboard_king(self) -> dict[str, Any] | None:
        try:
            _, dashboard = fetch_dashboard(self.args.dashboard_url, self.args.dashboard_timeout_s)
            kings = kings_from_dashboard(dashboard)
            return kings[0] if kings else None
        except Exception:
            return None

    def result_for_king(self, king_id: str | None) -> dict[str, Any] | None:
        if not king_id:
            return None
        local_result = read_json(self.args.results_root / "kings" / king_id / "results.json", None)
        index = self.store.load_index()
        record = (index.get("kings") or {}).get(king_id, {})
        latest = record.get("latest_result")
        return self.best_result(local_result, latest if isinstance(latest, dict) else None)

    def overlay_active_result(
        self,
        display_result: dict[str, Any] | None,
        current_job: dict[str, Any] | None,
        worker_event: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not display_result or not current_job:
            return display_result
        result_model = display_result.get("model") or {}
        job_model = current_job.get("model") or {}
        if result_model.get("king_id") != job_model.get("king_id"):
            return display_result

        active_names = [name for name in current_job.get("benchmarks") or [] if name]
        if not active_names:
            return display_result

        active_rows: dict[str, dict[str, Any]] = {}
        event_rows = (((worker_event or {}).get("partial_results") or {}).get("benchmarks") or [])
        for row in event_rows:
            if isinstance(row, dict) and row.get("name") in active_names:
                active_rows[row["name"]] = dict(row)

        patched = dict(display_result)
        rows_by_name: dict[str, dict[str, Any]] = {}
        for row in display_result.get("benchmarks") or []:
            if isinstance(row, dict) and row.get("name"):
                rows_by_name[row["name"]] = dict(row)

        for name in active_names:
            row = dict(rows_by_name.get(name) or {"name": name, "metric": {"name": None, "value": None}})
            row.update(active_rows.get(name) or {})
            if row.get("status") not in {"completed", "completed_no_metric"}:
                row["status"] = "running"
                row.setdefault("metric", {"name": None, "value": None})
            rows_by_name[name] = row

        order = list(self.args.benchmarks)
        rows = [rows_by_name.pop(name) for name in order if name in rows_by_name]
        rows.extend(rows_by_name.values())
        patched["benchmarks"] = rows
        patched["totals"] = self.result_totals(rows)
        patched["status"] = "running"
        return patched

    def latest_payload(
        self,
        *,
        status: str,
        current_job: dict[str, Any] | None = None,
        worker_event: dict[str, Any] | None = None,
        result_payload: dict[str, Any] | None = None,
        queue_size: int | None = None,
    ) -> dict[str, Any]:
        current_king = self.current_dashboard_king()
        current_king_id = (current_king or {}).get("king_id")
        result_king_id = ((result_payload or {}).get("model") or {}).get("king_id")
        display_result = result_payload if result_king_id == current_king_id else self.result_for_king(current_king_id)
        display_result = self.overlay_active_result(display_result, current_job, worker_event)
        payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": status, "generated_at": utcnow_iso()}
        if current_king:
            payload["current_king"] = current_king
        if display_result:
            payload["latest_result"] = display_result
        if queue_size is not None:
            payload["queue_size"] = queue_size
        if current_job:
            payload["active_job"] = current_job
            if ((current_job.get("model") or {}).get("king_id") == current_king_id):
                payload["current_job"] = current_job
        if worker_event:
            event_king_id = ((worker_event.get("model") or {}).get("king_id"))
            if event_king_id == current_king_id:
                payload["worker_event"] = worker_event
            else:
                payload["active_worker_event"] = worker_event
        return payload

    def handle_worker_snapshot(self, snapshot: dict[str, Any]) -> None:
        current_job = self.state.get("current_job")
        if not isinstance(current_job, dict) or not current_job.get("job_id"):
            return

        last_job = snapshot.get("last_job") if isinstance(snapshot.get("last_job"), dict) else {}
        marker = {
            "current_job_id": current_job.get("job_id"),
            "snapshot_job_id": snapshot.get("job_id"),
            "status": snapshot.get("status"),
            "updated_at": snapshot.get("updated_at"),
            "finished_at": snapshot.get("finished_at"),
            "last_job_id": last_job.get("job_id"),
            "last_job_finished_at": last_job.get("finished_at"),
        }
        if self.state.get("last_worker_snapshot_marker") == marker:
            return
        self.state["last_worker_snapshot_marker"] = marker

        job_id = current_job.get("job_id")
        status = snapshot.get("status")
        if status == "idle" and last_job.get("job_id") == job_id:
            self.handle_result(last_job)
            return
        if snapshot.get("job_id") == job_id:
            event = dict(snapshot)
            event.setdefault("event", "job_progress")
            event.setdefault("model", current_job.get("model"))
            event.setdefault("benchmarks", current_job.get("benchmarks"))
            self.handle_event(event)

    def poll_worker_once(self) -> None:
        current_job = self.state.get("current_job")
        if not isinstance(current_job, dict) or not current_job.get("job_id"):
            return
        try:
            snapshot = http_json(f"{self.args.worker_url.rstrip()}/status", token=self.args.worker_token, timeout_s=20)
        except Exception as exc:
            self.state["last_worker_poll_error"] = {"at": utcnow_iso(), "error": repr(exc)}
            self.save()
            return

        self.handle_worker_snapshot(snapshot)

    def dispatch_next(self) -> dict[str, Any]:
        queue = self.build_queue()
        if not queue:
            item = self.mmlu_pro_catchup_item()
            if item is None:
                self.state.update({"status": "idle", "last_queue_size": 0})
                self.save()
                return {"dispatched": False, "reason": "no unevaluated kings"}
            benchmarks = ["MMLU-Pro"]
            dispatch_reason = "idle_mmlu_pro_catchup"
        else:
            item, benchmarks, dispatch_reason = self.select_dispatch_item(queue)
        king = item["king"]
        dispatch_number = self.next_dispatch_number()
        job_id = f"{king['king_id']}-{utcnow_iso().replace(':','').replace('-','')}"
        job = {
            "job_id": job_id,
            "model": king,
            "benchmarks": benchmarks,
            "batch_size": "auto",
            "dtype": self.args.dtype,
            "device": self.args.device,
            "model_args_extra": self.args.model_args_extra,
            "progress_interval_s": self.args.worker_progress_interval_s,
            "dispatch_reason": dispatch_reason,
            "dispatch_number": dispatch_number,
            "retry_count": self.retry_count(king.get("king_id")),
            "max_retries": self.max_retries(),
        }
        resp = http_json(f"{self.args.worker_url.rstrip('/')}/jobs", method="POST", payload=job, token=self.args.worker_token, timeout_s=30)
        self.state.update({
            "status": "dispatched",
            "current_job": job,
            "worker_response": resp,
            "last_queue_size": len(queue),
            "dispatch_count": dispatch_number,
            "last_dispatch_reason": dispatch_reason,
        })
        self.save()
        self.mark_retry_running(job)
        self.store.write_latest(self.latest_payload(status="running", current_job=job, queue_size=len(queue)))
        return {"dispatched": True, "job": job, "worker_response": resp}

    def mark_retry_running(self, job: dict[str, Any]) -> None:
        model = job.get("model") or {}
        king_id = model.get("king_id")
        active = {name for name in (job.get("benchmarks") or []) if name}
        if not king_id or not active:
            return
        existing = self.result_for_king(king_id)
        if not isinstance(existing, dict):
            return
        patched = dict(existing)
        rows = []
        for row in patched.get("benchmarks") or []:
            if not isinstance(row, dict):
                continue
            row = dict(row)
            if row.get("name") in active and row.get("status") == "failed":
                row["status"] = "running"
                row["metric"] = {"name": None, "value": None}
                row.pop("returncode", None)
            rows.append(row)
        patched["benchmarks"] = rows
        patched["totals"] = self.result_totals(rows)
        patched["status"] = "running"
        patched["generated_at"] = utcnow_iso()
        self.store.write_result(king_id, patched)
        index = self.store.load_index()
        record = index.setdefault("kings", {}).get(king_id)
        if isinstance(record, dict):
            record["status"] = "running"
            record["updated_at"] = utcnow_iso()
            record["latest_result"] = patched
            self.store.write_index(index)
        self.store.remove_failed_history_for_king(king_id)

    def handle_event(self, payload: dict[str, Any]) -> None:
        self.state.update({"status": payload.get("status", "running"), "last_event": payload})
        self.save()
        self.store.write_latest(self.latest_payload(status=payload.get("status", "running"), current_job=self.state.get("current_job"), worker_event=payload))

    def result_totals(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "requested": len(rows),
            "completed": sum(1 for row in rows if row.get("status") == "completed"),
            "completed_no_metric": sum(1 for row in rows if row.get("status") == "completed_no_metric"),
            "failed": sum(1 for row in rows if row.get("status") == "failed"),
            "missing": sum(1 for row in rows if row.get("status") == "missing"),
            "running": sum(1 for row in rows if row.get("status") == "running"),
            "pending": sum(1 for row in rows if row.get("status") == "pending"),
        }

    def overall_status(self, current_status: str | None, rows: list[dict[str, Any]]) -> str | None:
        by_name = {row.get("name"): row.get("status") for row in rows if row.get("name")}
        desired = list(self.args.benchmarks)
        if any(status == "failed" for status in by_name.values()):
            return "failed"
        tracked = list(desired)
        if "MMLU-Pro" in by_name and "MMLU-Pro" not in tracked:
            tracked.append("MMLU-Pro")
        if tracked and all(by_name.get(name) in {"completed", "missing"} for name in tracked):
            return "missing" if any(by_name.get(name) == "missing" for name in tracked) else "completed"
        return current_status

    def merge_result_rows(self, king_id: str | None, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        existing = self.result_for_king(king_id)
        merged: dict[str, dict[str, Any]] = {}
        if isinstance(existing, dict):
            for row in existing.get("benchmarks") or []:
                if isinstance(row, dict) and row.get("name"):
                    merged[row["name"]] = dict(row)
        for row in rows:
            if row.get("name"):
                merged[row["name"]] = dict(row)
        order = list(self.args.benchmarks)
        ordered = [merged.pop(name) for name in order if name in merged]
        ordered.extend(merged.values())
        return ordered

    def normalize_finished_result(self, payload: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]], dict[str, int] | None]:
        results = payload.get("results") or {}
        rows = [dict(row) for row in (results.get("benchmarks") or []) if isinstance(row, dict)]
        status = payload.get("status")
        artifacts = payload.get("artifacts") or {}
        remote_log = str(artifacts.get("remote_log") or "").lower()
        is_download_failure = (status == "failed" and "download" in remote_log and rows and all((row.get("status") or "failed") == "failed" for row in rows))
        if status == "missing" or is_download_failure:
            status = "missing"
            for row in rows:
                row["status"] = "missing"
                row.setdefault("metric", {"name": None, "value": None})
                row.setdefault("error_type", "model_missing")
                row.setdefault("message", "model snapshot/download unavailable")
        if not rows:
            return status, rows, results.get("totals")
        model = payload.get("model") or {}
        rows = self.merge_result_rows(model.get("king_id") or payload.get("job_id"), rows)
        totals = self.result_totals(rows)
        status = self.overall_status(status, rows)
        return status, rows, totals

    def handle_result(self, payload: dict[str, Any]) -> None:
        model = payload.get("model") or {}
        king_id = model.get("king_id") or payload.get("job_id")
        status, rows, totals = self.normalize_finished_result(payload)
        result_payload = {
            "schema_version": "teutonic-king-benchmark-result.v1",
            "generated_at": utcnow_iso(),
            "job_id": payload.get("job_id"),
            "status": status,
            "model": model,
            "benchmarks": rows,
            "totals": totals,
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "worker_artifacts": payload.get("artifacts"),
        }
        self.store.write_result(king_id, result_payload)
        index = self.store.load_index()
        index.setdefault("kings", {})[king_id] = {"model": model, "status": status, "updated_at": utcnow_iso(), "result_s3": f"s3://{self.store.bucket}/{self.store.result_key(king_id)}", "latest_result": result_payload}
        self.store.write_index(index)
        self.store.append_history(result_payload)
        self.note_final_status(king_id, status)
        self.state.update({"status": "idle", "last_result": result_payload, "current_job": None})
        self.save()
        self.store.write_latest(self.latest_payload(status=status, result_payload=result_payload))


def websocket_worker_loop(state: ControllerState) -> None:
    delay = 1.0
    while True:
        sock = None
        try:
            host, port, path = worker_ws_url(state.args.worker_url)
            sock = websocket_connect(host, port, path, state.args.worker_token, timeout_s=20)
            state.state["worker_websocket"] = {"status": "connected", "at": utcnow_iso()}
            state.save()
            delay = 1.0
            while True:
                text = websocket_recv_text(sock)
                if text is None:
                    raise ConnectionError("worker websocket closed")
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    state.handle_worker_snapshot(payload)
        except Exception as exc:
            state.state["worker_websocket"] = {"status": "disconnected", "at": utcnow_iso(), "error": repr(exc), "retry_in_s": delay}
            state.save()
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

def auto_dispatch_loop(state: ControllerState) -> None:
    while True:
        try:
            snapshot = dict(state.state)
            if snapshot.get("current_job"):
                state.poll_worker_once()
            else:
                snapshot = dict(state.state)
                if snapshot.get("status") in {"idle", "failed", "completed"}:
                    state.dispatch_next()
        except Exception as exc:
            state.state.update({"status": "idle", "last_auto_dispatch_error": repr(exc)})
            state.save()
        time.sleep(state.args.auto_dispatch_interval_s)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length).decode("utf-8") if length else "{}")

    def _auth(self) -> bool:
        state: ControllerState = self.server.state  # type: ignore[attr-defined]
        return require_bearer(dict(self.headers), state.args.controller_token)

    def do_GET(self) -> None:
        state: ControllerState = self.server.state  # type: ignore[attr-defined]
        if self.path == "/health":
            self._json(200, {"ok": True, "state": state.state})
        elif self.path == "/queue":
            if not self._auth():
                self._json(401, {"error": "unauthorized"}); return
            self._json(200, {"queue": state.build_queue()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        state: ControllerState = self.server.state  # type: ignore[attr-defined]
        if not self._auth():
            self._json(401, {"error": "unauthorized"}); return
        if self.path == "/worker-events":
            state.handle_event(self._read_json())
            self._json(200, {"ok": True})
        elif self.path == "/worker-results":
            state.handle_result(self._read_json())
            self._json(200, {"ok": True})
        elif self.path == "/dispatch-next":
            self._json(200, state.dispatch_next())
        else:
            self._json(404, {"error": "not found"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("TEUTONIC_CONTROLLER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TEUTONIC_CONTROLLER_PORT", "32100")))
    parser.add_argument("--worker-url", default=os.environ.get("TEUTONIC_KING_BENCH_WORKER_URL", ""))
    parser.add_argument("--worker-token", default=os.environ.get("TEUTONIC_KING_BENCH_WORKER_TOKEN", ""))
    parser.add_argument("--controller-token", default=os.environ.get("TEUTONIC_KING_BENCH_CONTROLLER_TOKEN") or os.environ.get("TEUTONIC_KING_BENCH_WORKER_TOKEN", ""))
    parser.add_argument("--benchmarks", default=os.environ.get("TEUTONIC_KING_BENCH_BENCHMARKS", ",".join(DEFAULT_BENCHMARKS)))
    parser.add_argument("--results-root", type=Path, default=Path(os.environ.get("TEUTONIC_KING_BENCH_SERVICE_ROOT", "runs/king-benchmark-service")))
    parser.add_argument("--hippius-endpoint", default=os.environ.get("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com"))
    parser.add_argument("--hippius-bucket", default=os.environ.get("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3"))
    parser.add_argument("--latest-key", default=os.environ.get("TEUTONIC_KING_BENCH_SERVICE_LATEST_KEY", "king-benchmark-daily/all-kings/latest.json"))
    parser.add_argument("--index-key", default=os.environ.get("TEUTONIC_KING_BENCH_SERVICE_INDEX_KEY", "king-benchmark-daily/all-kings/index.json"))
    parser.add_argument("--history-key", default=os.environ.get("TEUTONIC_KING_BENCH_SERVICE_HISTORY_KEY", "king-benchmark-daily/all-kings/history.jsonl"))
    parser.add_argument("--dashboard-url", action="append", default=None)
    parser.add_argument("--dashboard-timeout-s", type=int, default=20)
    parser.add_argument("--dtype", default=os.environ.get("TEUTONIC_KING_BENCH_DTYPE", "bfloat16"))
    parser.add_argument("--device", default=os.environ.get("TEUTONIC_KING_BENCH_DEVICE", "auto"))
    parser.add_argument("--model-args-extra", default=os.environ.get("TEUTONIC_KING_BENCH_MODEL_ARGS_EXTRA", ""))
    parser.add_argument("--worker-progress-interval-s", type=int, default=int(os.environ.get("TEUTONIC_WORKER_PROGRESS_INTERVAL_S", "60")))
    parser.add_argument("--auto-dispatch", action=argparse.BooleanOptionalAction, default=env_bool("TEUTONIC_KING_BENCH_AUTO_DISPATCH", True))
    parser.add_argument("--auto-dispatch-interval-s", type=int, default=int(os.environ.get("TEUTONIC_KING_BENCH_AUTO_DISPATCH_INTERVAL_S", "60")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("TEUTONIC_KING_BENCH_MAX_RETRIES", "3")))
    parser.add_argument("--mmlu-pro-interval", type=int, default=int(os.environ.get("TEUTONIC_KING_BENCH_MMLU_PRO_INTERVAL", "3")))
    args = parser.parse_args()
    args.benchmarks = [part.strip() for part in args.benchmarks.split(",") if part.strip()]
    if not args.worker_url:
        raise SystemExit("TEUTONIC_KING_BENCH_WORKER_URL is required")
    if not args.worker_token or not args.controller_token:
        raise SystemExit("TEUTONIC_KING_BENCH_WORKER_TOKEN/CONTROLLER_TOKEN is required")
    state = ControllerState(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state  # type: ignore[attr-defined]
    if args.auto_dispatch:
        threading.Thread(target=auto_dispatch_loop, args=(state,), daemon=True).start()
        threading.Thread(target=websocket_worker_loop, args=(state,), daemon=True).start()
    print(f"controller listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("controller shutting down", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
