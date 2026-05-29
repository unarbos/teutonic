#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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

    def put_s3(self, key: str, path: Path, content_type: str = "application/json") -> None:
        client = self.s3_client()
        if client is None:
            return
        client.put_object(Bucket=self.bucket, Key=key, Body=path.read_bytes(), ContentType=content_type)

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


class ControllerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.store = Store(args.results_root, args.hippius_bucket, args.hippius_endpoint, args.latest_key, args.index_key, args.history_key)
        self.state_path = args.results_root / "controller_state.json"
        self.state = read_json(self.state_path, {"status": "idle", "updated_at": utcnow_iso()})

    def save(self) -> None:
        self.state["updated_at"] = utcnow_iso()
        write_json(self.state_path, self.state)

    def build_queue(self) -> list[dict[str, Any]]:
        _, dashboard = fetch_dashboard(self.args.dashboard_url, self.args.dashboard_timeout_s)
        kings = kings_from_dashboard(dashboard)
        index = self.store.load_index()
        queued: list[dict[str, Any]] = []
        for king in kings:
            record = (index.get("kings") or {}).get(king["king_id"], {})
            result = read_json(self.args.results_root / "kings" / king["king_id"] / "results.json", None)
            missing = missing_benchmarks(result or record.get("latest_result"), self.args.benchmarks)
            if missing:
                queued.append({"king": king, "missing_benchmarks": missing})
        return queued

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
        result = read_json(self.args.results_root / "kings" / king_id / "results.json", None)
        if result is not None:
            return result
        index = self.store.load_index()
        record = (index.get("kings") or {}).get(king_id, {})
        latest = record.get("latest_result")
        return latest if isinstance(latest, dict) else None

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

    def dispatch_next(self) -> dict[str, Any]:
        queue = self.build_queue()
        if not queue:
            self.state.update({"status": "idle", "last_queue_size": 0})
            self.save()
            return {"dispatched": False, "reason": "no unevaluated kings"}
        item = queue[0]
        king = item["king"]
        job_id = f"{king['king_id']}-{utcnow_iso().replace(':','').replace('-','')}"
        job = {
            "job_id": job_id,
            "model": king,
            "benchmarks": item["missing_benchmarks"],
            "batch_size": "auto",
            "dtype": self.args.dtype,
            "device": self.args.device,
            "model_args_extra": self.args.model_args_extra,
            "progress_interval_s": self.args.worker_progress_interval_s,
        }
        resp = http_json(f"{self.args.worker_url.rstrip('/')}/jobs", method="POST", payload=job, token=self.args.worker_token, timeout_s=30)
        self.state.update({"status": "dispatched", "current_job": job, "worker_response": resp, "last_queue_size": len(queue)})
        self.save()
        self.store.write_latest(self.latest_payload(status="running", current_job=job, queue_size=len(queue)))
        return {"dispatched": True, "job": job, "worker_response": resp}

    def handle_event(self, payload: dict[str, Any]) -> None:
        self.state.update({"status": payload.get("status", "running"), "last_event": payload})
        self.save()
        self.store.write_latest(self.latest_payload(status=payload.get("status", "running"), current_job=self.state.get("current_job"), worker_event=payload))

    def handle_result(self, payload: dict[str, Any]) -> None:
        model = payload.get("model") or {}
        king_id = model.get("king_id") or payload.get("job_id")
        result_payload = {
            "schema_version": "teutonic-king-benchmark-result.v1",
            "generated_at": utcnow_iso(),
            "job_id": payload.get("job_id"),
            "status": payload.get("status"),
            "model": model,
            "benchmarks": (payload.get("results") or {}).get("benchmarks", []),
            "totals": (payload.get("results") or {}).get("totals"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "worker_artifacts": payload.get("artifacts"),
        }
        self.store.write_result(king_id, result_payload)
        index = self.store.load_index()
        index.setdefault("kings", {})[king_id] = {"model": model, "status": payload.get("status"), "updated_at": utcnow_iso(), "result_s3": f"s3://{self.store.bucket}/{self.store.result_key(king_id)}", "latest_result": result_payload}
        self.store.write_index(index)
        self.store.write_latest(self.latest_payload(status=payload.get("status"), current_job=self.state.get("current_job"), result_payload=result_payload))
        self.store.append_history(result_payload)
        self.state.update({"status": "idle", "last_result": result_payload, "current_job": None})
        self.save()


def auto_dispatch_loop(state: ControllerState) -> None:
    while True:
        try:
            snapshot = dict(state.state)
            if snapshot.get("status") in {"idle", "failed", "completed"} and not snapshot.get("current_job"):
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
    print(f"controller listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
