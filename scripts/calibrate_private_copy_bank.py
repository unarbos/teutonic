#!/usr/bin/env python3
"""Exercise the private rolling fingerprint bank on real checkpoints.

Models are loaded one at a time.  Each fingerprint is computed while its model
is resident, then the checkpoint is released before the next model is loaded.
The secret comes only from the operator-only local secret file.

Example:
    python scripts/calibrate_private_copy_bank.py \
      --model original=https://hub.hippius.com/models/org/original/main \
      --model suspected=https://huggingface.co/org/copy \
      --bank /tmp/private-copy-probe.sqlite3 \
      --output /tmp/private-copy-probe-results.json
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

import eval_server_quasar_pair as base  # noqa: E402
from eval.copy_probe import (  # noqa: E402
    COPY_PROBE_VERSION,
    FingerprintBank,
    load_private_probe_secret,
    private_model_fingerprint,
)

log = logging.getLogger("calibrate_private_copy_bank")


def parse_labeled_ref(value: str) -> tuple[str, str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=MODEL_REF")
    label, ref = value.split("=", 1)
    label, ref = label.strip(), ref.strip()
    if not label or not ref:
        raise argparse.ArgumentTypeError("model must be LABEL=MODEL_REF")
    if "@" in ref:
        ref, digest = ref.rsplit("@", 1)
    else:
        digest = ""
    if not digest and ref.startswith(("http://", "https://")):
        parsed = urlparse(ref)
        parts = [part for part in parsed.path.split("/") if part]
        if "hippius.com" in parsed.netloc and len(parts) >= 4 and parts[0] == "models":
            digest = parts[3]
    return label, ref, digest


def public_result(label: str, model_ref: str, decision: dict) -> dict:
    result = {
        "label": label,
        "model": model_ref,
        "detected": decision["detected"],
        "stored": decision["stored"],
        "too_similar": decision["too_similar"],
        "similar_family_size": decision["similar_family_size"],
        "max_similar_models": decision["max_similar_models"],
        "compared": decision["compared"],
        "bank_size": decision["bank_size"],
        "version": decision["version"],
        "js_p95_max": decision["js_p95_max"],
    }
    if decision.get("match"):
        match = decision["match"]
        result["closest"] = {
            "model": match["model_ref"],
            "model_digest": match["model_digest"],
            "submission_id": match["submission_id"],
            "metrics": match["metrics"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_labeled_ref, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--bank-size", type=int, default=100)
    parser.add_argument("--max-similar-models", type=int, default=3)
    parser.add_argument("--js-p95-max", type=float, default=0.001)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        secret = load_private_probe_secret(REPO_ROOT / ".seed")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for real-model calibration")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    base._gpu_ids = [args.gpu]
    base.preflight_deps()
    base.patch_transformers_masking_compat()
    try:
        base.patch_triton_autotuner_thread_safety()
    except Exception:
        log.warning("Triton thread-safety patch unavailable; sequential calibration is safe")

    bank = FingerprintBank(
        args.bank,
        max_entries=args.bank_size,
        max_similar_models=args.max_similar_models,
    )
    bank.initialize()
    req = base.EvalRequest(
        king_repo=args.model[0][1],
        challenger_repo=args.model[0][1],
        parallel_models=False,
        model_device_map="single",
    )
    device = f"cuda:{args.gpu}"
    results: list[dict] = []
    started = time.time()

    for order, (label, ref, digest) in enumerate(args.model, start=1):
        model = None
        try:
            snapshot = base.materialize_model(ref, digest)
            config, artifacts = base.load_model_config(snapshot, req, label)
            model = base.load_quasar_model(
                snapshot, config, device, label, req, gpu_ids=[args.gpu]
            )
            model_digest = base.snapshot_safetensors_digest(snapshot)
            fingerprint, fingerprint_meta = private_model_fingerprint(model, secret)
            decision = bank.compare_and_store(
                fingerprint,
                fingerprint_meta,
                model_ref=base.normalize_model_ref(ref),
                model_digest=model_digest,
                submission_id=f"calibration-{order:03d}-{label}",
                submission_order=str(order),
                hotkey="",
                js_p95_max=args.js_p95_max,
            )
            result = public_result(label, ref, decision)
            result.update({
                "model_digest": model_digest,
                "snapshot": snapshot,
                "artifacts": artifacts,
            })
            results.append(result)
            closest = result.get("closest") or {}
            metrics = closest.get("metrics") or {}
            print(
                f"{label}: rejected={decision['detected']} "
                f"too_similar={decision['too_similar']} stored={decision['stored']} "
                f"family_size={decision['similar_family_size']}/"
                f"{decision['max_similar_models']} "
                f"compared={decision['compared']} bank_size={decision['bank_size']} "
                f"closest={closest.get('model', 'none')} "
                f"js_p95={metrics.get('js_p95', 'none')}",
                flush=True,
            )
        except Exception as exc:
            log.exception("model %s failed", label)
            results.append({
                "label": label,
                "model": ref,
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "version": COPY_PROBE_VERSION,
        "bank_size_limit": args.bank_size,
        "max_similar_models": args.max_similar_models,
        "js_p95_max": args.js_p95_max,
        "results": results,
        "elapsed_s": round(time.time() - started, 1),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.chmod(output, 0o600)
        print(f"wrote private calibration record to {output}", flush=True)
    return 1 if any("error" in result for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
