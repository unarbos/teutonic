#!/usr/bin/env python3
"""GPU-only Horizon NLL verification and timing; reads local snapshots, writes a report."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import torch.nn.functional as F

    from teutonic.evaluator.horizon_score import document_nll, load_reference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 257])
    parser.add_argument("--throughput-lengths", type=int, nargs="*", default=[])
    args = parser.parse_args()

    def sync():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

    started = time.monotonic()
    model = load_reference(str(args.snapshot))
    sync()
    report = {"snapshot": str(args.snapshot), "load_seconds": time.monotonic() - started,
              "torch": torch.__version__, "device_map": {k: str(v) for k, v in model.hf_device_map.items()},
              "input": "deterministic synthetic token IDs; not corpus throughput",
              "checks": [], "throughput": []}
    generator = random.Random(789)
    eos = model.config.eos_token_id
    if type(eos) is not int:
        raise ValueError("expected one EOS ID")

    def tokens(length):
        ids = [generator.randrange(3, model.config.vocab_size) for _ in range(length - 1)] + [eos]
        if length > 4:
            ids[2] = eos  # An embedded EOS must remain part of this whole document.
        return ids

    for length in args.lengths:
        ids = tokens(length)
        with torch.inference_mode():
            tensor = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
            logits = model(tensor, use_cache=False, output_router_logits=False).logits.float()
            expected = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                       tensor[:, 1:].to(logits.device).reshape(-1), reduction="none").double().sum().item()
            del logits, tensor
        measured = document_nll(model, ids)
        error = abs(measured - expected) / (length - 1)
        # BF16 GEMM may round differently for chunked and full projection shapes.
        if not math.isfinite(measured) or error > 0.002:
            raise AssertionError(f"chunked NLL differs from forward: per-token error={error}")
        row = {"input_tokens": length, "forward_nll": expected, "chunked_nll": measured,
               "absolute_error_per_token": error, "tolerance_nats_per_token": .002}
        report["checks"].append(row)
        print(json.dumps(row), flush=True)
    for length in args.throughput_lengths:
        ids = tokens(length)
        sync()
        started = time.monotonic()
        nll = document_nll(model, ids)
        sync()
        seconds = time.monotonic() - started
        row = {"input_tokens": length, "seconds": seconds,
               "scored_tokens_per_second": (length - 1) / seconds, "nll": nll}
        report["throughput"].append(row)
        print(json.dumps(row), flush=True)
    report["peak_allocated_gib"] = [torch.cuda.max_memory_allocated(i) / 2**30
                                     for i in range(torch.cuda.device_count())]
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "load_seconds": report["load_seconds"],
                      "peak_allocated_gib": report["peak_allocated_gib"]}), flush=True)


if __name__ == "__main__":
    main()
