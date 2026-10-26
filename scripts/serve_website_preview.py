#!/usr/bin/env python3
"""Serve local website assets with live public data, or opt in to synthetic demos."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DASHBOARD_ORIGIN = "https://pub-fedac496355c4edc9aed57189e6e190f.r2.dev"
PUBLIC_BENCHMARK_URL = (
    "https://pub-c982d552b8044578b4a79e653700ec73.r2.dev"
    "/king-benchmark-daily/all-kings/results.json"
)


def feed_cache_seconds(path: str) -> int:
    if path.endswith("/manifest.json") or path == "/benchmarks/results.json":
        return 60
    return 15 if path == "/competitions.json" else 5


@dataclass(frozen=True)
class CachedFeed:
    body: bytes
    compressed: bytes | None
    etag: str
    upstream_etag: str | None
    expires_at: float


class PublicFeedCache:
    """Share short-lived public responses and one upstream request per feed."""

    def __init__(self):
        self.entries: dict[str, CachedFeed] = {}
        self.locks: dict[str, Lock] = {}
        self.lock = Lock()

    def get(self, path: str) -> CachedFeed:
        upstream = public_data_url(path)
        if not upstream:
            raise ValueError("not a public dashboard feed")
        with self.lock:
            lock = self.locks.setdefault(path, Lock())
        with lock:
            previous = self.entries.get(path)
            if previous and previous.expires_at > monotonic():
                return previous
            headers = {
                "User-Agent": "Teutonic-local-preview/1.0",
                "Cache-Control": "no-cache",
                "Accept-Encoding": "gzip",
            }
            if previous and previous.upstream_etag:
                headers["If-None-Match"] = previous.upstream_etag
            try:
                with urlopen(Request(upstream, headers=headers), timeout=20) as response:
                    body = response.read()
                    compressed = body if response.headers.get("Content-Encoding") == "gzip" else None
                    upstream_etag = response.headers.get("ETag")
            except HTTPError as error:
                if error.code != 304 or not previous:
                    raise
                entry = replace(previous, expires_at=monotonic() + feed_cache_seconds(path))
                self.entries[path] = entry
                return entry
            if compressed:
                body = gzip.decompress(compressed)
            json.loads(body)
            if not compressed and len(body) > 1024:
                compressed = gzip.compress(body, compresslevel=5, mtime=0)
            if compressed and len(compressed) >= len(body):
                compressed = None
            entry = CachedFeed(
                body=body, compressed=compressed,
                etag='W/"' + hashlib.sha256(body).hexdigest() + '"',
                upstream_etag=upstream_etag,
                expires_at=monotonic() + feed_cache_seconds(path),
            )
            self.entries[path] = entry
            return entry


def public_data_url(path: str) -> str | None:
    """Proxy only known public feeds; all website assets come from the checkout."""
    if path == "/benchmarks/results.json":
        return PUBLIC_BENCHMARK_URL
    if path in {"/dashboard.json", "/competitions.json", "/datasets/manifest.json"}:
        return PUBLIC_DASHBOARD_ORIGIN + path
    if re.fullmatch(
        r"/competitions/(main|math|code|text|mix(?:0[1-9]|[1-9][0-9]))/"
        r"(dashboard\.json|datasets/manifest\.json)", path
    ):
        return PUBLIC_DASHBOARD_ORIGIN + path
    return None


@lru_cache(maxsize=1)
def preview_inventory() -> dict:
    """Read the small inventory at the start of the local canonical mix artifact."""
    path = ROOT / "artifacts/category-mixes/mix06.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        prefix = handle.read(65536)
    marker = '"category_inventory":'
    if marker not in prefix:
        return {}
    try:
        inventory, _ = json.JSONDecoder().raw_decode(prefix.split(marker, 1)[1].lstrip())
        return inventory
    except json.JSONDecodeError:
        return {}


def preview_payloads(established: int) -> dict:
    with (ROOT / "chain.toml").open("rb") as handle:
        config = tomllib.load(handle)
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    definitions = config["competitions"]
    mixes = [key for key in definitions if key != "main"][:established]
    legacy = [
        {"id": f"demo-old-{i}", "hotkey": f"DEMO-OLD-HOTKEY-{i}",
         "weight": .15, "crowned_block": 100 + i}
        for i in range(4)
    ][established:]
    transition = {
        "version": "mix-replacement-v1",
        "main": {"id": "demo-main", "hotkey": "DEMO-MAIN-HOTKEY", "weight": .4},
        "legacy": legacy,
        "mixes": {key: f"DEMO-{key.upper()}-HOTKEY" for key in mixes},
    }
    main_weight = (1 - .125 * established) * .4 / (.4 + sum(e["weight"] for e in legacy))
    payloads = {}
    index = []
    for key, spec in definitions.items():
        prefix = "" if key == "main" else f"competitions/{key}/"
        has_king = key == "main" or key in mixes
        chain = {
            "name": "Teutonic · LOCAL DEMO", "netuid": 3, "generation": "local-preview",
            "competition": key, "seed_repo": config["chain"]["seed_repo"],
            "seed_digest": config["seed"]["seed_digest"], "seed_repo_backend": "hf",
        }
        king = {
            "reign_number": 1, "uid": 100 + len(index),
            "hotkey": f"DEMO-{key.upper()}-HOTKEY", "coldkey": "DEMO-COLDKEY",
            "model_repo": f"demo/{key}-winner", "model_digest": "sha256:" + "a" * 64,
            "crowned_at": (now - timedelta(days=1)).isoformat(),
        } if has_king else None
        payout = {"weight": main_weight if key == "main" else .125 if has_king else None,
                  "alpha_per_hour": None, "usd_per_hour": None}
        history = []
        if key == "main":
            for i in range(12):
                loss = 2.4 - i * .012
                history.append({
                    "challenge_id": f"demo-evaluation-{i}", "uid": 200 + i,
                    "hotkey": f"DEMO-CHALLENGER-{i}", "coldkey": "DEMO-COLDKEY",
                    "model_identity": "hidden_until_promotion", "challenger_repo": None,
                    "verdict": "rejected", "mu_hat": .001, "lcb": .0005,
                    "delta_threshold": spec["delta_threshold"],
                    "avg_king_loss": loss, "avg_challenger_loss": loss - .001,
                    "n_sequences": spec["n"], "n_sequences_evaluated": spec["n"],
                    "timestamp": (now - timedelta(hours=12 - i)).isoformat(),
                    "wall_time_s": 3600, "early_stopped": False,
                })
        payloads[f"/{prefix}dashboard.json"] = {
            "schema_version": 1, "publication_id": "00000000-0000-4000-8000-000000000001",
            "generated_at": timestamp, "updated_at": timestamp, "source_watermark": 1,
            "chain": chain, "king": king, "king_payout": payout,
            "king_chain": [{**king, **payout}] if king else [],
            "stats": {}, "current_eval": None, "queue": [], "history": history,
            "dataset_versions": [], "weight_status": {"state": "local demo"},
            "service_status": {"overall": "healthy"}, "market": None,
        }
        sources = []
        for category, weight in spec["categories"].items():
            inventory = config["categories"][category]
            sources.append({
                "name": category, "dataset": inventory["dataset"], "proportion": weight, "enabled": True,
                "total_tokens": inventory["tokens"], "sequence_length": 2048,
                "total_shards": preview_inventory().get(category, {}).get("total_shards"),
                "manifest_url": config["category_datasets"][inventory["dataset"]]["manifest_url"],
                "source": inventory["dataset"],
            })
        payloads[f"/{prefix}datasets/manifest.json"] = {
            "schema_version": 1, "generated_at": timestamp, "chain": chain,
            "dataset_label": f"{key}-categories-v1", "config_version": "local-preview",
            "eval_n": spec["n"], "delta_threshold": spec["delta_threshold"],
            "sources": sources,
        }
        index.append({
            "competition": key, "active": True, "has_king": has_king,
            "reward_transition": transition if key == "main" else None,
            "eval_n": spec["n"], "document_tokens": spec["long_document_tokens"],
            "delta_threshold": spec["delta_threshold"], "dashboard": prefix + "dashboard.json",
        })
    payloads["/competitions.json"] = {"competitions": index}
    payloads["/benchmarks/results.json"] = {
        "schema_version": "teutonic-king-benchmark-all-results.v2", "kings": [],
        "generated_at": timestamp,
    }
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default: loopback)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--demo", action="store_true", help="use synthetic activity instead of live public feeds"
    )
    parser.add_argument("--mix-kings", type=int, choices=range(7), default=2,
                        help="number of MIX kings in the synthetic transition; requires --demo (0–6)")
    args = parser.parse_args()
    public_feeds = PublicFeedCache()

    class Handler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT / "website"), **kwargs)

        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            upstream = public_data_url(path)
            payloads = preview_payloads(args.mix_kings) if args.demo else {}
            if args.demo and path in payloads:
                self.respond(json.dumps(payloads[path]).encode(), "application/json")
            elif not args.demo and upstream:
                # Read the same public R2 origins as teutonic.ai's Worker. No
                # credentials, database access, or synthetic fallback is used.
                try:
                    feed = public_feeds.get(path)
                    cache_control = "public, max-age=60" if feed_cache_seconds(path) == 60 else "no-cache"
                    self.respond(
                        feed.body, "application/json", source=upstream, etag=feed.etag,
                        compressed=feed.compressed, cache_control=cache_control,
                        age=max(0, int(feed_cache_seconds(path) - (feed.expires_at - monotonic()))),
                    )
                except HTTPError as error:
                    body = json.dumps({
                        "error": "Public feed unavailable", "upstream_status": error.code
                    }).encode()
                    self.respond(body, "application/json", status=error.code)
                except (URLError, TimeoutError, OSError, ValueError):
                    self.respond(
                        b'{"error":"Public feed unavailable"}', "application/json", status=502
                    )
            elif path in {"/", "/index.html"}:
                html = (ROOT / "website/index.html").read_text()
                self.respond(html.encode(), "text/html", cache_control="no-cache")
            else:
                super().do_GET()

        def do_HEAD(self):
            path = urlsplit(self.path).path
            if public_data_url(path) or path in {"/", "/index.html"}:
                self.do_GET()
            else:
                super().do_HEAD()

        def end_headers(self):
            url = urlsplit(self.path)
            if Path(url.path).suffix in {".js", ".css", ".png", ".ico"}:
                value = "public, max-age=3600" if url.query.startswith("v=") else "no-cache"
                self.send_header("Cache-Control", value)
            super().end_headers()

        def respond(
            self, body: bytes, content_type: str, *, status: int = 200,
            source: str | None = None, etag: str | None = None,
            compressed: bytes | None = None, cache_control: str = "no-store", age: int | None = None,
        ):
            if status == 200 and cache_control != "no-store" and not etag:
                etag = 'W/"' + hashlib.sha256(body).hexdigest() + '"'
            matches = self.headers.get("If-None-Match", "").split(",")
            unchanged = etag and any(
                value.strip() == "*" or value.strip().removeprefix("W/") == etag.removeprefix("W/")
                for value in matches
            )
            if status == 200 and unchanged:
                status = 304
            accepts_gzip = False
            for encoding in self.headers.get("Accept-Encoding", "").lower().split(","):
                name, *parameters = encoding.strip().split(";")
                if name == "gzip":
                    accepts_gzip = not any(re.fullmatch(r"\s*q=0(?:\.0*)?\s*", value) for value in parameters)
            if status == 200 and accepts_gzip and len(body) > 1024:
                compressed = compressed or gzip.compress(body, compresslevel=5, mtime=0)
                if len(compressed) < len(body):
                    body = compressed
                else:
                    accepts_gzip = False
            else:
                accepts_gzip = False
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            if status != 304:
                self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.send_header("Vary", "Accept-Encoding")
            if age is not None:
                self.send_header("Age", str(age))
            if etag:
                self.send_header("ETag", etag)
            if accepts_gzip:
                self.send_header("Content-Encoding", "gzip")
            if source:
                self.send_header("X-Teutonic-Data-Source", source)
            self.end_headers()
            if status != 304 and self.command != "HEAD":
                self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    mode = "synthetic activity, real mix definitions" if args.demo else "LIVE public teutonic.ai data"
    print(f"Website: http://{args.host}:{args.port} · {mode}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
