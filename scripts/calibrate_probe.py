#!/usr/bin/env python3
"""Calibrate the trainability probe against known-good and known-bad models.

Loads each repo onto a single GPU, runs trainability_probe, prints a table,
and shrugs out a suggested threshold based on the empirical gap.

Usage:
    cd teutonic && python scripts/calibrate_probe.py [--gpu 0]

The current king and the historical reject list (the knsimon/Teutonic-I-3xxxx
series and the iotaminer trick variants) should produce a clear bimodal
separation: honest models with delta < 0.5, trick models with delta > 50 or
non-finite.
"""
import argparse
import gc
import logging
import os
import sys
import time

import torch

# Ensure we can import eval_torch from teutonic/
HERE = os.path.dirname(os.path.abspath(__file__))
TEUTONIC_DIR = os.path.dirname(HERE)
sys.path.insert(0, TEUTONIC_DIR)

from eval_torch import load_model, trainability_probe  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibrate")


# Known-good (from recent post-fix logs that received real verdicts or passed sanity).
KNOWN_GOOD = [
    "ChrisJackieChan/Teutonic-I-7",        # current king
    "dsaddsaf/Teutonic-I-miner1",
    "22oseni/Teutonic-I-foo",
    "dasLOL/Teutonic-I-10",
    "miner-bit/Teutonic-I-5000",
    "zddos/Teutonic-I-x_a1",
    "unconst/Teutonic-I-h0",
]

# Known reparameterization-trick models (rejected for inflated layernorms).
KNOWN_TRICK = [
    "knsimon/Teutonic-I-30000",
    "knsimon/Teutonic-I-31000",
    "iotaminer/Teutonic-I-blitz-king-b1000-fixed",
    "lukewqerqwer/Teutonic-I-run1",
    "tech-dev-ai/Teutonic-I-201",
    "miner-bit/Teutonic-I-203",
]


def probe_repo(repo: str, device: str) -> dict:
    t0 = time.time()
    try:
        model = load_model(repo, device, label=repo, force_download=False)
    except Exception as e:
        return {"repo": repo, "error": f"load_failed: {type(e).__name__}: {e}",
                "elapsed_s": time.time() - t0}
    try:
        probe = trainability_probe(model)
        probe["repo"] = repo
        probe["elapsed_s"] = time.time() - t0
        return probe
    except Exception as e:
        return {"repo": repo, "error": f"probe_failed: {type(e).__name__}: {e}",
                "elapsed_s": time.time() - t0}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        if v in (float("inf"), float("-inf")):
            return f"{v:+.0e}"
        return f"{v:+.4f}"
    return str(v)


def print_row(label: str, r: dict):
    if "error" in r:
        print(f"  {label:30s} {r['repo']:48s}  ERROR: {r['error']}")
        return
    print(f"  {label:30s} {r['repo']:48s} "
          f"before={fmt(r['loss_before']):>10s} "
          f"after={fmt(r['loss_after']):>10s} "
          f"delta={fmt(r['delta']):>10s} "
          f"ok={r['ok']!s:5s} "
          f"({r['elapsed_s']:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--repos", nargs="*", default=None,
                    help="Override repo list. Default: built-in known-good + known-trick.")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    if not torch.cuda.is_available():
        sys.exit("no CUDA available")

    if args.repos:
        repos = [(r, "user") for r in args.repos]
    else:
        repos = [(r, "GOOD") for r in KNOWN_GOOD] + [(r, "TRICK") for r in KNOWN_TRICK]

    results = []
    print(f"\nProbing {len(repos)} repos on {device}\n")
    print("=" * 130)
    for repo, label in repos:
        r = probe_repo(repo, device)
        r["label"] = label
        results.append(r)
        print_row(label, r)
    print("=" * 130)

    good = [r for r in results if r.get("label") == "GOOD" and "error" not in r]
    trick = [r for r in results if r.get("label") == "TRICK" and "error" not in r]

    def _stats(rs, key):
        vals = [r[key] for r in rs if r[key] == r[key] and r[key] not in (float("inf"), float("-inf"))]
        if not vals:
            return "no finite values"
        return f"min={min(vals):+.4f} max={max(vals):+.4f}"

    print(f"\nGOOD models  delta: {_stats(good, 'delta')}")
    print(f"TRICK models delta: {_stats(trick, 'delta')}")

    good_max = max((r["delta"] for r in good
                    if r["delta"] == r["delta"] and r["delta"] not in (float("inf"), float("-inf"))),
                   default=None)
    trick_min = min((r["delta"] for r in trick
                     if r["delta"] == r["delta"] and r["delta"] not in (float("inf"), float("-inf"))),
                    default=None)
    if good_max is not None and trick_min is not None and good_max < trick_min:
        suggested = (good_max + trick_min) / 2
        print(f"\nSuggested PROBE_LOSS_DELTA_ABS: {suggested:.2f} "
              f"(midpoint of GOOD max {good_max:.4f} and TRICK min {trick_min:.4f})")
    else:
        print("\nGap is fuzzy or one cohort missing finite values. Inspect "
              "rows above and pick a threshold by hand.")


if __name__ == "__main__":
    main()
