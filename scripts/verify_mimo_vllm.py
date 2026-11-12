#!/usr/bin/env python3
"""Compare adapted native vLLM MiMo NLL with the current evaluator.

Uses pre-tokenized local test inputs only. No dataset access or remote writes.
Every scored input includes native EOS; only the first token is unscored.
"""

import argparse
import hashlib
import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--attention-backend", default="TRITON_ATTN_DIFFKV")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_PLUGINS"] = "teutonic_mimo"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE",
                          str(args.output_dir.parent / "flashinfer-workspace"))

    import numpy as np
    from vllm import LLM, SamplingParams

    rows = json.loads(args.inputs.read_text())
    if any(len(r["tokens"]) - 1 != r["scored_tokens"] or r["tokens"][-1] != 151645 for r in rows):
        raise ValueError("fixture does not match the native-EOS scoring policy")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate fixture identity")
    settings = {"model": str(args.snapshot), "trust_remote_code": True, "model_impl": "vllm",
                    "dtype": "bfloat16", "tensor_parallel_size": args.tensor_parallel_size,
                    "enable_expert_parallel": True, "max_model_len": 16385,
                    "max_num_seqs": 16, "max_num_batched_tokens": 8192,
                    "attention_config": {"backend": args.attention_backend},
                    "kv_cache_memory_bytes": 16 * 1024**3,
                    "gpu_memory_utilization": 0.8, "seed": 42, "skip_tokenizer_init": True,
                    "enable_prefix_caching": False, "enforce_eager": True,
                    "disable_custom_all_reduce": args.disable_custom_all_reduce}
    metadata = {"settings": settings, "batch_invariant": os.environ["VLLM_BATCH_INVARIANT"],
                "fused_grouped_topk": os.environ.get("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1"),
                "v2_model_runner": os.environ["VLLM_USE_V2_MODEL_RUNNER"],
                "adapter_sha256": hashlib.sha256(Path(__file__).with_name("mimo_vllm_adapter.py").read_bytes()).hexdigest(),
                "versions": {p: version(p) for p in ("vllm", "torch", "transformers")},
                "snapshot_hashes": {name: hashlib.sha256((args.snapshot / name).read_bytes()).hexdigest()
                                    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")},
                "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                "documents": len(rows), "scored_tokens": sum(r["scored_tokens"] for r in rows)}
    (args.output_dir / "configuration.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"phase": "load", **metadata}), flush=True)
    started = time.monotonic()
    llm = LLM(**settings)
    load_seconds = time.monotonic() - started
    print(json.dumps({"phase": "loaded", "seconds": load_seconds}), flush=True)
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0,
                            detokenize=False, seed=42)
    reference, cases = {}, []

    def score(name, groups):
        started = time.monotonic()
        values = {}
        for group in groups:
            outputs = llm.generate([{"prompt_token_ids": row["tokens"]} for row in group],
                                   params, use_tqdm=False)
            if len(outputs) != len(group):
                raise AssertionError("missing output")
            for row, output in zip(group, outputs, strict=True):
                if list(output.prompt_token_ids) != row["tokens"]:
                    raise AssertionError("engine changed input IDs")
                logprobs = output.prompt_logprobs
                if len(logprobs) != len(row["tokens"]) or logprobs[0]:
                    raise AssertionError("prompt-target accounting differs")
                for pos, token in enumerate(row["tokens"][1:], 1):
                    if token not in logprobs[pos]:
                        diagnostic = {"case": name, "id": row["id"], "position": pos,
                                      "expected_token": token, "actual_tokens": list(logprobs[pos]),
                                      "input_nearby": row["tokens"][max(0, pos-3):pos+4],
                                      "returned_nearby": [list(logprobs[i] or {}) for i in
                                                          range(max(0, pos-3), min(len(logprobs), pos+4))]}
                        (args.output_dir / "invalid-target.json").write_text(json.dumps(diagnostic, indent=2))
                        raise AssertionError(f"prompt target mismatch at position {pos}")
                losses = [-entry[token].logprob for entry, token in
                          zip(logprobs[1:], row["tokens"][1:], strict=True)]
                if not all(math.isfinite(v) and v >= 0 for v in losses):
                    raise AssertionError("invalid token NLL")
                values[row["id"]] = losses
        seconds = time.monotonic() - started
        differences = [np.asarray(losses) - np.asarray(reference[key])
                       for key, losses in values.items() if key in reference]
        case = {"name": name, "seconds": seconds,
                "start_monotonic": started, "end_monotonic": started + seconds,
                "documents": len(values),
                "scored_tokens": sum(len(v) for v in values.values()),
                "max_token_nll_difference": max((float(np.max(np.abs(d))) for d in differences), default=0),
                "different_tokens": sum(int(np.count_nonzero(d)) for d in differences),
                "compared_to": "initial batched run" if reference else None}
        case["scored_tokens_per_second"] = case["scored_tokens"] / seconds
        (args.output_dir / f"{name}-token-nll.json").write_text(json.dumps(values))
        cases.append(case)
        print(json.dumps({"phase": "case_complete", **case}), flush=True)
        return values

    reference = score("batched", [rows])
    score("repeat", [rows])
    score("reverse", [list(reversed(rows))])
    by_length = sorted(rows, key=lambda r: len(r["tokens"]))
    score("batches-of-four", [by_length[i:i+4] for i in range(0, len(rows), 4)])
    single_rows = [by_length[i] for i in sorted({0, 1, len(rows)//4, len(rows)//2,
                                               3*len(rows)//4, len(rows)-2, len(rows)-1})]
    score("single-documents", [[r] for r in single_rows])
    comparisons = []
    for row in rows:
        if row.get("baseline_nll") is None:
            continue
        nll = math.fsum(reference[row["id"]])
        delta = nll - row["baseline_nll"]
        comparisons.append({"id": row["id"], "hf_nll": row["baseline_nll"], "vllm_nll": nll,
                            "delta_nll": delta, "delta_nats_per_token": delta / row["scored_tokens"],
                            "delta_bits_per_byte": delta / (math.log(2) * row["byte_length"]) if row["byte_length"] else None})
    restart = None
    if args.previous:
        old_config = json.loads((args.previous / "configuration.json").read_text())
        if old_config != metadata:
            raise ValueError("restart configuration differs")
        old = json.loads((args.previous / "batched-token-nll.json").read_text())
        differences = [np.asarray(reference[key]) - np.asarray(old[key]) for key in reference]
        restart = {"different_tokens": sum(int(np.count_nonzero(d)) for d in differences),
                   "max_token_nll_difference": max(float(np.max(np.abs(d))) for d in differences)}
    report = {**metadata, "load_seconds": load_seconds, "cases": cases,
              "batch_invariant_exact": all(c["different_tokens"] == 0 for c in cases),
              "restart": restart, "baseline_comparison": comparisons,
              "baseline_exact": all(c["delta_nll"] == 0 for c in comparisons),
              "max_abs_baseline_nats_per_token": max(abs(c["delta_nats_per_token"]) for c in comparisons),
              "max_abs_baseline_bits_per_byte": max(abs(c["delta_bits_per_byte"]) for c in comparisons if c["delta_bits_per_byte"] is not None)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "baseline_comparison"}), flush=True)
    # Report all discrepancies before failing: a flag alone is not evidence.
    if not report["batch_invariant_exact"] or restart and restart["different_tokens"]:
        raise AssertionError("vLLM token losses are not exactly invariant")


if __name__ == "__main__":
    main()
