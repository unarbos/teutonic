#!/usr/bin/env python3
"""Exercise production MiMo workers on local token fixtures; no R2 or promotion."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.evaluator.vllm_runtime import Worker, runtime_identity, wait_for_gpu_release


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tp", type=int, choices=(1, 2), default=1)
    p.add_argument("--length", type=int, help="synthetic long context for TP2 validation")
    p.add_argument("--previous", type=Path, help="report differences from another topology; not an equality gate")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.inputs.read_text())[:32]
    if args.length:
        tokens = rows[0]["tokens"][:-1]
        rows.append({"id": "long-fixture", "tokens": (tokens * (args.length // len(tokens) + 1))[:args.length - 1] + [151645]})
    docs = [SimpleNamespace(doc_id=r["id"]) for r in rows]
    tokens = [r["tokens"] for r in rows]
    identity = runtime_identity()
    baseline = None
    events = []
    def event(info):
        events.append(info)
        print(json.dumps(info), flush=True)
    try:
        for repeat in range(2):
            ids = list(range(args.tp))
            wait_for_gpu_release(ids)
            with Worker("king", args.snapshot, ids, args.output, identity, event,
                        worker_id=f"tp{args.tp}-load{repeat}",
                        max_model_len=max(16385, max(map(len, tokens)) + 1)) as worker:
                first = worker.score(docs, tokens)
                reverse = worker.score(docs[::-1], tokens[::-1])[::-1]
                singles = [worker.score([d], [t])[0] for d, t in zip(docs[:3], tokens[:3], strict=True)]
                if first != reverse or singles != first[:3] or (baseline is not None and first != baseline):
                    raise AssertionError("replica, order, batch or fresh-process scores differ")
                baseline = first
                (args.output / f"scores-{repeat}.json").write_text(json.dumps(first))
            wait_for_gpu_release(ids)
        comparison = None
        if args.previous:
            previous = {r["id"]: r for r in json.loads(args.previous.read_text())}
            deltas = [abs(r["nll"] - previous[r["id"]]["nll"]) / r["scored_tokens"]
                      for r in baseline if r["id"] in previous]
            comparison = {"documents": len(deltas), "max_absolute_nats_per_token": max(deltas),
                          "different_documents": sum(d != 0 for d in deltas)}
        (args.output / "passed.json").write_text(json.dumps({"passed": True, "tp": args.tp,
            "documents": len(rows), "scored_tokens": sum(len(t)-1 for t in tokens),
            "cross_topology": comparison, "events": events}))
    except BaseException as error:
        (args.output / "failure.json").write_text(json.dumps({"error": str(error), "time": time.time()}))
        raise


if __name__ == "__main__":
    main()
