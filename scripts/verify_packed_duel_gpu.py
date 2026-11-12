#!/usr/bin/env python3
"""Run both real GPU phases against a local packed-text fixture; never access R2."""

import argparse
import hashlib
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LocalObjects:
    """Only the read API used by DatasetReader, backed by local fixture files."""

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.lock = threading.Lock()

    def add(self, path, raw, **metadata):
        target = self.root / "teutonic-text" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        return {"path": path, "size_bytes": len(raw), "sha256": sha,
                "etag": f'"{sha}"', **metadata}

    def get_object(self, *, Bucket, Key, Range=None, IfMatch=None):
        if Bucket != "dataset" or not Key.startswith("teutonic-text/"):
            raise AssertionError("unexpected or legacy bucket access")
        target = (self.root / Key).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise AssertionError("fixture path escape")
        raw = target.read_bytes()
        etag = f'"{hashlib.sha256(raw).hexdigest()}"'
        if IfMatch is not None and IfMatch != etag:
            raise AssertionError("fixture object identity mismatch")
        with self.lock:
            self.calls.append({"bucket": Bucket, "key": Key, "range": Range})
        if Range is None:
            return {"Body": io.BytesIO(raw)}
        start, end = map(int, Range.removeprefix("bytes=").split("-"))
        if not 0 <= start <= end < len(raw):
            raise AssertionError("invalid fixture range")
        if Key.endswith(".utf8"):
            time.sleep((start % 7) * 0.003)  # Exercise differing completion order.
        return {"Body": io.BytesIO(raw[start:end + 1]), "ETag": etag,
                "ContentRange": f"bytes {start}-{end}/{len(raw)}"}


def build_fixture(client, tokenizer):
    import numpy as np

    from teutonic.evaluation.packed_text import (
        DTYPE,
        FORMAT,
        MIX_FORMAT,
        DatasetReader,
        PackedDataset,
        build_catalog,
        canonical,
    )

    objects, categories = [], []
    shard = 0
    for category, weight in (("a", 0.6), ("b", 0.4)):
        categories.append({"dataset": "fixture", "category": category, "weight": weight})
        # Unequal partitions, interleaved short/long text, sorted index only.
        for count in (6, 10):
            blob, rows = bytearray(), []
            for i in range(count):
                unit = "The sky is blue. Zażółć gęślą jaźń. 你好世界. 2 + 2 = 4.\n"
                text = f"Document {category}/{shard}/{i}.\n" + unit * (28 if i % 2 else 150)
                text += "Embedded marker: <|im_end|> remains inside this document.\n"
                raw = text.encode("utf-8")
                length = len(tokenizer.encode([text])[0])
                if not (2 <= length <= 4096 if i % 2 else 4097 <= length <= 16384):
                    raise AssertionError(f"fixture length outside intended bucket: {length}")
                identity = hashlib.sha256(f"{category}/{shard}/{i}".encode()).digest()
                rows.append((identity, shard, len(blob), len(raw), length,
                             hashlib.sha256(raw).digest()))
                blob.extend(raw)
            common = {"dataset": "fixture", "category": category,
                      "shard_id": shard, "document_count": count}
            objects.append(client.add(f"text/fixture/{category}/part-{shard}.utf8",
                                      bytes(blob), role="text", **common))
            rows.sort(key=lambda row: (row[4], row[0]))
            stream = io.BytesIO()
            np.save(stream, np.array(rows, dtype=DTYPE), allow_pickle=False)
            objects.append(client.add(f"index/fixture/{category}/part-{shard}.documents.npy",
                                      stream.getvalue(), role="index", min_tokens=rows[0][4],
                                      max_tokens=rows[-1][4], **common))
            shard += 1
    manifest = client.add("manifest.json", canonical({"format": FORMAT, "objects": objects}))
    pin = {"bucket": "dataset", "key": "teutonic-text/manifest.json",
           "sha256": manifest["sha256"]}
    dataset = PackedDataset(DatasetReader(client), pin)
    catalog = build_catalog(dataset)
    sampling = client.add("sampling/fixture.json", canonical(catalog))
    mix = {"format": MIX_FORMAT, "competition": "main", "dataset": pin,
           "sampling": {"bucket": "dataset", "key": "teutonic-text/sampling/fixture.json",
                        "sha256": sampling["sha256"]},
           "token_budget": 20000, "categories": categories}
    (client.root / "mix.json").write_bytes(canonical(mix))
    return mix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimo-snapshot", type=Path, required=True)
    parser.add_argument("--horizon-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID
    from teutonic.evaluator import document_duel, engine
    from teutonic.evaluator.text_pipeline import TextTokenizer, prepared_batches

    snapshot = str(args.mimo_snapshot.resolve())
    tokenizer = TextTokenizer(snapshot, eos_token_id=DOCUMENT_EOS_TOKEN_ID)
    client = LocalObjects(args.output_dir / "objects")
    mix = build_fixture(client, tokenizer)
    engine._gpu_ids = engine.parse_gpu_ids()
    state = args.output_dir / "reference-state"
    state.mkdir()
    engine._horizon_downloader = SimpleNamespace(
        root=state, wait_ready=lambda callback: args.horizon_snapshot.resolve(),
    )
    req = engine.EvalRequest(king_repo=snapshot, challenger_repo=snapshot,
                             packed_dataset=mix, block_hash="0x" + "a" * 64,
                             hotkey="local-packed-fixture", batch_size=2,
                             n_bootstrap=1000, delta_threshold=.0015)
    config, _ = engine.load_model_config(snapshot, req, "king")
    req.max_document_tokens = config.max_position_embeddings
    req.vocab_size = config.vocab_size

    def batches(*a, **kw):
        # Force several streaming chunks without needing a large synthetic corpus.
        return prepared_batches(*a, **{**kw, "batch_documents": 2, "prefetch_batches": 2})

    reports, events = [], []

    def progress(info):
        events.append(info)
        print(json.dumps(info), flush=True)

    try:
        with patch.object(document_duel, "read_client", return_value=client), \
                patch.object(document_duel, "prepared_batches", batches):
            for name in ("cache-miss", "cache-hit"):
                client.calls.clear()
                events.clear()
                started = time.monotonic()
                verdict, metadata, workers = document_duel.run_document_duel(
                    engine, req, snapshot, snapshot, progress,
                )
                report = {"case": name, "seconds": time.monotonic() - started,
                          "verdict": verdict, "dataset": metadata, "workers": workers,
                          "reads": list(client.calls), "phases": list(events)}
                (args.output_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
                if verdict["accepted"] or abs(verdict["mu_hat"]) > 1e-6:
                    raise AssertionError("self-duel produced a reward")
                samples = verdict["sample_results"]
                if samples["king_nll"] != samples["challenger_nll"]:
                    raise AssertionError("same-checkpoint replica losses differ")
                replicas = workers["roles"]["mimo_replicas"]
                if len(replicas) != 8 or any(len(r["gpu_ids"]) != 1 for r in replicas):
                    raise AssertionError("ordinary fixture did not use eight independent GPU replicas")
                expected = metadata["n_documents"] if name == "cache-miss" else 0
                if metadata["reference_misses"] != expected:
                    raise AssertionError("unexpected reference-cache behavior")
                if sum(c["key"].endswith(".utf8") for c in client.calls) != metadata["n_documents"]:
                    raise AssertionError("selected text was not fetched exactly once per duel")
                if list(state.glob("duel-text-*")):
                    raise AssertionError("temporary text cache was not cleaned up")
                if name == "cache-hit":
                    if any(e.get("phase") == "reference_scoring" for e in events):
                        raise AssertionError("cache hit ran reference scoring")
                    if metadata["digest"] != reports[0]["dataset"]["digest"]:
                        raise AssertionError("sample changed across identical requests")
                    if verdict["sample_results"] != reports[0]["verdict"]["sample_results"]:
                        raise AssertionError("cached and uncached GPU scores differ")
                reports.append(report)
        (args.output_dir / "summary.json").write_text(json.dumps({
            "passed": True, "dataset": "local synthetic packed-text fixture; no R2 access",
            "documents": reports[0]["dataset"]["n_documents"],
            "reference_misses": [r["dataset"]["reference_misses"] for r in reports],
            "seconds": [r["seconds"] for r in reports],
            "sample_digest": reports[0]["dataset"]["digest"],
            "identical_scores": True, "self_duel_reward": 0,
        }, indent=2) + "\n")
    finally:
        if engine._model_worker_pool is not None:
            engine._model_worker_pool.close(force=True)


if __name__ == "__main__":
    main()
