#!/usr/bin/env python3
"""Serve website/ on localhost with synthetic activity and real chain mix weights."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mix-kings", type=int, choices=range(7), default=2,
                        help="number of MIX kings in the synthetic transition (0–6)")
    args = parser.parse_args()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT / "website"), **kwargs)

        def do_GET(self):
            path = urlsplit(self.path).path
            payloads = preview_payloads(args.mix_kings)
            if path in payloads:
                self.respond(json.dumps(payloads[path]).encode(), "application/json")
            elif path in {"/", "/index.html"}:
                html = (ROOT / "website/index.html").read_text()
                self.respond(html.encode(), "text/html")
            else:
                super().do_GET()

        def respond(self, body: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local preview: http://localhost:{args.port} · synthetic activity, real mix definitions", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
