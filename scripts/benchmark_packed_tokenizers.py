#!/usr/bin/env python3
"""Benchmark local UTF-8 JSONL documents against both pinned tokenizers; no network."""

import argparse
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.evaluator.text_pipeline import TextTokenizer


def benchmark(texts, snapshot, eos, repeats):
    tokenizer = TextTokenizer(snapshot, eos_token_id=eos, backend="gigatoken")

    def reference():
        return [row.ids + [eos] for row in
                tokenizer.reference.encode_batch(texts, add_special_tokens=False)]

    def timed(encode):
        start = time.perf_counter()
        value = encode()
        return value, time.perf_counter() - start

    expected, cold_reference = timed(reference)
    actual, cold_gigatoken = timed(lambda: tokenizer.encode(texts))
    if actual != expected:
        raise ValueError("Gigatoken differs from the reference tokenizer")
    reference_times, gigatoken_times = [], []
    for _ in range(repeats):
        reference_times.append(timed(reference)[1])
        gigatoken_times.append(timed(lambda: tokenizer.encode(texts))[1])
    normal, fast = statistics.median(reference_times), statistics.median(gigatoken_times)
    return {"exact_match": True, "input_tokens": sum(map(len, actual)),
            "cold_reference_seconds": cold_reference, "cold_gigatoken_seconds": cold_gigatoken,
            "warm_reference_seconds": normal, "warm_gigatoken_seconds": fast,
            "warm_speedup": normal / fast,
            "utf8_MB_per_second": sum(len(text.encode("utf-8")) for text in texts) / fast / 1e6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True, help='JSONL rows with a "text" field')
    parser.add_argument("--mimo-snapshot", type=Path, required=True)
    parser.add_argument("--horizon-snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    with args.documents.open(encoding="utf-8") as source:
        texts = [json.loads(line)["text"] for line in source]
    if not texts or args.repeats < 1 or not all(isinstance(text, str) for text in texts):
        parser.error("provide nonempty text rows and a positive repeat count")
    result = {"documents": len(texts), "utf8_bytes": sum(len(text.encode()) for text in texts),
              "gigatoken": importlib.metadata.version("gigatoken"), "repeats": args.repeats,
              "results": {name: benchmark(texts, path, eos, args.repeats)
                          for name, path, eos in [("mimo", args.mimo_snapshot, 151645),
                                                   ("horizon", args.horizon_snapshot, 1)]}}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
