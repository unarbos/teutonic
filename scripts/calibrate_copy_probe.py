#!/usr/bin/env python3
"""Calibrate the public behavioral-copy gate on real checkpoints.

The reference is loaded once. Candidates are loaded one at a time and their
probe metrics are optionally written to a private local JSON file. The script
always runs a reference-vs-itself repeat to measure the evaluator's numerical
floor.

Example:
    python scripts/calibrate_copy_probe.py \
      --reference https://hub.hippius.com/models/org/original/main \
      --candidate suspected=https://huggingface.co/org/model \
      --candidate independent=dendrite/teutonic-x-genesis \
      --independent-label independent \
      --output /tmp/copy-probe-calibration.json
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

import eval_server_quasar_pair as base  # noqa: E402
from eval.copy_probe import compare_models  # noqa: E402

log = logging.getLogger("calibrate_copy_probe")


def parse_labeled_ref(value: str) -> tuple[str, str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=MODEL_REF")
    label, ref = value.split("=", 1)
    label, ref = label.strip(), ref.strip()
    if not label or not ref:
        raise argparse.ArgumentTypeError("candidate must be LABEL=MODEL_REF")
    if "@" in ref:
        ref, digest = ref.rsplit("@", 1)
    else:
        digest = ""
    if not ref or (not digest and value.endswith("@")):
        raise argparse.ArgumentTypeError("candidate revision after @ cannot be empty")
    return label, ref, digest


def load_model(ref: str, digest: str, req: base.EvalRequest, label: str, device: str):
    snapshot = base.materialize_model(ref, digest)
    config, artifacts = base.load_model_config(snapshot, req, label)
    model = base.load_quasar_model(snapshot, config, device, label, req, gpu_ids=[0])
    return model, snapshot, artifacts


def compact_metrics(metrics: dict) -> str:
    return (
        f"js_mean={metrics['js_mean']:.9g} js_p95={metrics['js_p95']:.9g} "
        f"js_max={metrics['js_max']:.9g} skl_p95={metrics['symmetric_kl_p95']:.9g} "
        f"tv_p95={metrics['total_variation_p95']:.9g} "
        f"top1={metrics['top1_agreement']:.6f}"
    )


def suggested_threshold(
    results: list[dict], copy_labels: set[str], independent_labels: set[str]
) -> dict:
    positives = [
        float(row["metrics"]["js_p95"])
        for row in results
        if "metrics" in row
        and (row["label"] == "self_repeat" or row["label"] in copy_labels)
    ]
    negatives = [
        float(row["metrics"]["js_p95"])
        for row in results
        if "metrics" in row and row["label"] in independent_labels
    ]
    if not positives or not negatives:
        return {"available": False, "reason": "need both known-copy and independent cohorts"}
    positive_max = max(positives)
    negative_min = min(negatives)
    if positive_max >= negative_min:
        return {
            "available": False,
            "reason": "known-copy and independent js_p95 cohorts overlap",
            "copy_max": positive_max,
            "independent_min": negative_min,
        }
    # A logarithmic midpoint preserves orders-of-magnitude separation. The
    # numerical floor avoids suggesting literal zero after exact self-repeat.
    floor = 1e-12
    threshold = math.sqrt(max(positive_max, floor) * max(negative_min, floor))
    return {
        "available": True,
        "js_p95_max": threshold,
        "copy_max": positive_max,
        "independent_min": negative_min,
        "separation_ratio": negative_min / max(positive_max, floor),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-digest", default="")
    parser.add_argument("--candidate", action="append", type=parse_labeled_ref, default=[])
    parser.add_argument(
        "--copy-label",
        action="append",
        default=[],
        help="Candidate label assigned to the copy/derivative cohort; repeat as needed.",
    )
    parser.add_argument(
        "--independent-label",
        action="append",
        default=[],
        help="Candidate label known to be independently trained; repeat as needed.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--skip-self-repeat", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for real-model copy-probe calibration")

    base._gpu_ids = [args.gpu]
    base.preflight_deps()
    base.patch_transformers_masking_compat()
    try:
        base.patch_triton_autotuner_thread_safety()
    except Exception:
        log.warning("Triton thread-safety patch unavailable; sequential calibration is safe")

    req = base.EvalRequest(
        king_repo=args.reference,
        challenger_repo=args.reference,
        parallel_models=False,
        model_device_map="single",
    )
    device = f"cuda:{args.gpu}"
    started = time.time()
    reference, reference_snapshot, reference_artifacts = load_model(
        args.reference, args.reference_digest, req, "reference", device
    )
    results: list[dict] = []

    if not args.skip_self_repeat:
        metrics = compare_models(reference, reference)
        results.append({"label": "self_repeat", "model": args.reference, "metrics": metrics})
        print(f"self_repeat: {compact_metrics(metrics)}", flush=True)

    for label, ref, digest in args.candidate:
        req.challenger_repo = ref
        try:
            candidate, snapshot, artifacts = load_model(ref, digest, req, label, device)
            try:
                metrics = compare_models(reference, candidate)
                results.append({
                    "label": label,
                    "model": ref,
                    "digest": digest,
                    "snapshot": snapshot,
                    "artifacts": artifacts,
                    "metrics": metrics,
                })
                print(f"{label}: {compact_metrics(metrics)}", flush=True)
            finally:
                del candidate
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as exc:
            log.exception("candidate %s failed", label)
            results.append({
                "label": label,
                "model": ref,
                "digest": digest,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{label}: ERROR {type(exc).__name__}: {exc}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    calibration = suggested_threshold(
        results, set(args.copy_label), set(args.independent_label)
    )
    payload = {
        "reference": args.reference,
        "reference_digest": args.reference_digest,
        "reference_snapshot": reference_snapshot,
        "reference_artifacts": reference_artifacts,
        "copy_labels": args.copy_label,
        "independent_labels": args.independent_label,
        "results": results,
        "suggested_threshold": calibration,
        "elapsed_s": round(time.time() - started, 1),
    }
    print("suggested_threshold:", json.dumps(calibration, sort_keys=True), flush=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote private calibration record to {out}", flush=True)

    del reference
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
