#!/usr/bin/env python3
"""Read-only live-R2 category benchmark with local, unpublished test manifests.

Uses the evaluator's normal R2 client and existing environment credentials.
Only listing, HEAD, and GET operations are used. Nothing is published to R2.
"""

import argparse
import hashlib
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare(args):
    import numpy as np
    from botocore.exceptions import ClientError

    from teutonic.evaluation.packed_text import (
        BUCKETS,
        CATALOG_FORMAT,
        DTYPE,
        FORMAT,
        MIX_FORMAT,
        PREFIX,
        canonical,
        read_client,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    client = read_client()
    listings = {}
    for role in ("text", "index", "provenance"):
        prefix = f"{PREFIX}{role}/{args.dataset}/{args.category}/"
        listings[role] = {
            int(Path(o["Key"]).name.split(".")[0].removeprefix("part-")): o
            for page in client.get_paginator("list_objects_v2").paginate(
                Bucket="dataset", Prefix=prefix)
            for o in page.get("Contents", [])
        }
    shards = set(listings["text"])
    if not shards or any(set(values) != shards for values in listings.values()):
        raise ValueError("text/index/provenance partition sets differ or are empty")

    def read_operation(operation, **kwargs):
        for attempt in range(8):
            try:
                return operation(**kwargs)
            except ClientError as error:
                if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 429 or attempt == 7:
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))

    def partition(shard):
        objects = {}
        for role in listings:
            listed = listings[role][shard]
            head = read_operation(client.head_object, Bucket="dataset", Key=listed["Key"],
                                      IfMatch=listed["ETag"])
            sha = head.get("Metadata", {}).get("sha256", "")
            if len(sha) != 64 or head["ContentLength"] != listed["Size"]:
                raise ValueError("missing exporter SHA-256 or changed object size")
            objects[role] = {"path": listed["Key"].removeprefix(PREFIX), "role": role,
                                 "size_bytes": head["ContentLength"], "sha256": sha,
                                 "etag": head["ETag"], "dataset": args.dataset,
                                 "category": args.category, "shard_id": shard}
        index = objects["index"]
        response = read_operation(client.get_object, Bucket="dataset", Key=PREFIX + index["path"],
                                     IfMatch=index["etag"])
        with response["Body"] as body:
            raw = body.read()
        if len(raw) != index["size_bytes"] or hashlib.sha256(raw).hexdigest() != index["sha256"]:
            raise ValueError("index SHA-256 or size mismatch")
        rows = np.load(io.BytesIO(raw), allow_pickle=False)
        if rows.dtype != DTYPE or rows.ndim != 1 or not len(rows):
            raise ValueError("invalid index layout")
        lengths = rows["original_mimo_token_count"]
        if np.any(lengths[1:] < lengths[:-1]) or np.any(rows["shard_id"] != shard):
            raise ValueError("unsorted index or mismatched shard ID")
        ids = [bytes(v) for v in rows["doc_id"]]
        if any(lengths[i] == lengths[i-1] and ids[i] < ids[i-1] for i in range(1, len(rows))):
            raise ValueError("index doc-ID tie ordering differs")
        text_size = objects["text"]["size_bytes"]
        offsets, sizes = rows["byte_offset"], rows["byte_length"]
        if np.any(offsets > text_size) or np.any(sizes > text_size - offsets):
            raise ValueError("document extends beyond text object")
        for obj in objects.values():
            obj["document_count"] = len(rows)
        index.update(min_tokens=int(lengths[0]), max_tokens=int(lengths[-1]))
        buckets = []
        for name, minimum, maximum, _ in BUCKETS:
            lo, hi = int(np.searchsorted(lengths, minimum)), int(np.searchsorted(lengths, maximum, side="right"))
            if hi > lo:
                buckets.append({"dataset": args.dataset, "category": args.category, "bucket": name,
                                    "path": index["path"], "lo": lo, "hi": hi,
                                    "available_tokens": int(lengths[lo:hi].sum())})
        return list(objects.values()), buckets

    objects, partitions, totals = [], [], {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, (entries, rows) in enumerate(pool.map(partition, sorted(shards)), 1):
            objects.extend(entries)
            for row in rows:
                total = totals.setdefault(row["bucket"], {"documents": 0, "tokens": 0})
                total["documents"] += row["hi"] - row["lo"]
                total["tokens"] += row.pop("available_tokens")
                partitions.append(row)
            if count % 50 == 0:
                print(json.dumps({"verified_index_partitions": count}), flush=True)
    for name, _, _, share in BUCKETS:
        if totals.get(name, {}).get("tokens", 0) < args.token_budget * share:
            raise ValueError(f"category cannot supply requested quota: {name}")

    def save(name, value):
        raw = canonical(value)
        (args.output_dir / name).write_bytes(raw)
        return {"bucket": "dataset", "key": f"{PREFIX}local-test/{name}",
                "sha256": hashlib.sha256(raw).hexdigest()}

    pin = save("inventory.json", {"format": FORMAT, "objects": objects,
               "scope": "Unpublished category-only test inventory; not full export completion"})
    sampling = save("sampling.json", {"format": CATALOG_FORMAT, "dataset": pin,
                    "buckets": [list(b) for b in BUCKETS], "partitions": partitions})
    save("mix.json", {"format": MIX_FORMAT, "competition": "codeio-readonly-test",
         "dataset": pin, "sampling": sampling, "token_budget": args.token_budget,
         "categories": [{"dataset": args.dataset, "category": args.category, "weight": 1.0}]})
    summary = {"partitions": len(shards), "available": totals,
               "text_bytes": sum(o["size_bytes"] for o in objects if o["role"] == "text"),
               "index_bytes_verified": sum(o["size_bytes"] for o in objects if o["role"] == "index"),
               "remote_writes": 0}
    (args.output_dir / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


class GetOnlyObjects:
    """Real R2 GETs with production DatasetReader range/ETag/hash validation."""

    def __init__(self, directory):
        from teutonic.evaluation.packed_text import read_client

        self.client = read_client()
        self.allowed = {"teutonic-text/" + obj["path"] for obj in
                        json.loads((directory / "inventory.json").read_text())["objects"]
                        if obj["role"] in ("text", "index")}
        self.local = {f"teutonic-text/local-test/{name}": (directory / name).read_bytes()
                      for name in ("inventory.json", "sampling.json")}
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "bytes": 0, "text_requests": 0, "index_requests": 0}
        self.last_report = time.monotonic()

    def get_object(self, *, Bucket, Key, Range=None, IfMatch=None):
        if Bucket != "dataset":
            raise ValueError("test allows only the singular dataset bucket")
        if Key in self.local:
            if Range or IfMatch:
                raise ValueError("unexpected local-manifest range request")
            return {"Body": io.BytesIO(self.local[Key])}
        if Key not in self.allowed:
            raise ValueError("GET outside pinned test inventory")
        kwargs = {"Bucket": Bucket, "Key": Key}
        if Range:
            kwargs["Range"] = Range
        if IfMatch:
            kwargs["IfMatch"] = IfMatch
        response = self.client.get_object(**kwargs)
        with self.lock:
            self.stats["requests"] += 1
            self.stats["bytes"] += response["ContentLength"]
            self.stats["text_requests" if Key.endswith(".utf8") else "index_requests"] += 1
            if time.monotonic() - self.last_report >= 30:
                print(json.dumps({"phase": "r2_reads", **self.stats}), flush=True)
                self.last_report = time.monotonic()
        return response


def run(args):
    from types import SimpleNamespace
    from unittest.mock import patch

    from teutonic.evaluation.packed_text import validate_mix
    from teutonic.evaluator import document_duel, engine
    from teutonic.evaluator.horizon import CONFIG_SHA256, INDEX_SHA256, file_hash

    args.output_dir.mkdir(parents=True, exist_ok=False)
    client = GetOnlyObjects(args.inventory_dir)
    mix = validate_mix(json.loads((args.inventory_dir / "mix.json").read_text()))
    if args.token_budget is not None:
        mix = validate_mix({**mix, "token_budget": args.token_budget})
    snapshot = str(args.mimo_snapshot.resolve())
    challenger = str((args.challenger_snapshot or args.mimo_snapshot).resolve())
    for name, expected in (("config.json", CONFIG_SHA256), ("model.safetensors.index.json", INDEX_SHA256)):
        if file_hash(args.horizon_snapshot / name) != expected:
            raise ValueError("Horizon pin mismatch")
    engine._gpu_ids = engine.parse_gpu_ids()
    state = args.output_dir / "reference-state"
    state.mkdir()
    engine._horizon_downloader = SimpleNamespace(root=state, wait_ready=lambda cb: args.horizon_snapshot.resolve())
    req = engine.EvalRequest(king_repo=snapshot, challenger_repo=challenger, packed_dataset=mix,
                             block_hash="0x" + "a" * 64, hotkey=args.hotkey,
                             n_bootstrap=args.n_bootstrap, delta_threshold=.0015)
    config, _ = engine.load_model_config(snapshot, req, "king")
    req.max_document_tokens = config.max_position_embeddings
    req.vocab_size = config.vocab_size
    engine.validate_and_report_attention_config(config, "king")
    challenger_config, _ = engine.load_model_config(challenger, req, "challenger")
    if engine.compare_model_configs(config, challenger_config):
        raise ValueError("king/challenger model configs differ")
    original_sample = document_duel.sample_documents
    started = time.monotonic()

    def sample(*a, **kw):
        from dataclasses import asdict

        documents, metadata = original_sample(*a, **kw)
        selection = {"metadata": metadata, "documents": [asdict(d) for d in documents]}
        (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2))
        if args.expected_selection is not None:
            if selection != json.loads(args.expected_selection.read_text()):
                raise AssertionError("parallel sampler differs from serial baseline")
            progress({"phase": "serial_sample_match", "documents": len(documents),
                      "digest": metadata["digest"]})
        return documents, metadata

    def progress(info):
        event = {"elapsed_seconds": time.monotonic() - started, **info}
        with (args.output_dir / "phases.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    try:
        progress({"phase": "sampling", "token_budget": mix["token_budget"]})
        with patch.object(document_duel, "read_client", return_value=client), \
                patch.object(document_duel, "sample_documents", sample):
            verdict, metadata, workers = document_duel.run_document_duel(
                engine, req, snapshot, challenger, progress)
        report = {"seconds": time.monotonic() - started, "verdict": verdict,
                  "dataset": metadata, "workers": workers, "r2_reads": client.stats,
                  "scope": "real CodeIO R2 ranges; local unpublished manifests; production document_duel",
                  "models": {"king": snapshot, "challenger": challenger}}
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if snapshot == challenger and (verdict["accepted"] or abs(verdict["mu_hat"]) > 1e-6):
            raise AssertionError("self-duel produced a reward")
        if list(state.glob("duel-text-*")):
            raise AssertionError("temporary text cache was not cleaned")
        (args.output_dir / "passed.json").write_text(json.dumps({"passed": True, "seconds": report["seconds"]}))
    except BaseException as error:
        (args.output_dir / "failure.json").write_text(json.dumps({"error": str(error), "type": type(error).__name__}))
        raise
    finally:
        if engine._model_worker_pool is not None:
            engine._model_worker_pool.close(force=True)
        client.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", default="code-reasoning")
    p.add_argument("--category", default="codeio")
    p.add_argument("--token-budget", type=int, default=70_000_000)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--inventory-dir", type=Path, required=True)
    p.add_argument("--mimo-snapshot", type=Path, required=True)
    p.add_argument("--challenger-snapshot", type=Path)
    p.add_argument("--token-budget", type=int)
    p.add_argument("--hotkey", default="codeio-live-r2-70m")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--horizon-snapshot", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-selection", type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)(args)


if __name__ == "__main__":
    main()
