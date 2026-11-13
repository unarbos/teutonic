#!/usr/bin/env python3
"""Compare reference reward policies on an immutable saved evaluation (no inference)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.evaluation.reference_policy import reference_verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.record.read_bytes()
    record_hash = hashlib.sha256(raw).hexdigest()
    if record_hash != args.expected_sha256:
        raise ValueError("evaluation artifact differs from the database checksum")
    record = json.loads(raw)
    request, saved = record["request"], record["verdict"]
    samples = saved["sample_results"]
    if samples["format"] != "packed-document-losses":
        raise ValueError("requires packed-document losses for all three models")
    documents = samples["documents"]
    if len(documents) != saved["n_sequences"]:
        raise ValueError("incomplete saved sample set")
    parameters = {key: request["limits"][key] for key in ("n_bootstrap", "alpha", "delta_threshold")}
    parameters["bootstrap_seed"] = request["sampling"]["bootstrap_seed"]
    inputs = [samples[key] for key in ("king_nll", "challenger_nll", "reference_nll")]
    inputs.append([doc["byte_length"] for doc in documents])
    baseline = reference_verdict(*inputs, **parameters, skip_both_zero=False)
    for key in ("mu_hat", "lcb", "accepted", "delta_threshold", "n_sequences", "scoring_policy"):
        if baseline[key] != saved[key]:
            raise ValueError(f"baseline replay differs from saved {key}: {baseline[key]} != {saved[key]}")
    candidate = reference_verdict(*inputs, **parameters, skip_both_zero=True)
    included, skipped = [], []
    for doc, k, c, r in zip(documents, *inputs[:3], strict=True):
        (included if k > r or c > r else skipped).append(doc["doc_id"])
    report = {
        "evaluation_id": request["evaluation_id"], "attempt_number": request["attempt_number"],
        "scope": "Exact saved MAIN document losses; CPU policy replay, no model inference or resampling.",
        "record_sha256": record_hash,
        "sample_results_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "policy_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "teutonic/evaluation/reference_policy.py").read_bytes()).hexdigest(),
        "parameters": parameters, "baseline_reproduces_saved_exactly": True,
        "baseline": {k: v for k, v in baseline.items() if k != "timestamp"},
        "candidate": {k: v for k, v in candidate.items() if k != "timestamp"},
        "included_doc_ids": included, "skipped_doc_ids": skipped,
        "sampled_bytes": sum(inputs[3]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"included_doc_ids", "skipped_doc_ids"}}, indent=2))


if __name__ == "__main__":
    main()
