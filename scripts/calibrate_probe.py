#!/usr/bin/env python3
"""Calibrate the trainability probe against known-good and known-bad models.

Loads each repo onto a single GPU, runs trainability_probe, prints a table,
and shrugs out a suggested threshold based on the empirical gap.

Usage:
    cd teutonic && python scripts/calibrate_probe.py [--gpu 0]

The current king and the historical reject list (the knsimon/Teutonic-I-3xxxx
series and the iotaminer trick variants) should produce a clear bimodal
separation: honest models pass all 5 layers, trick models trip Layer 1
(norm cap) or Layer 3/4 (grad cap).
"""
import argparse
import gc
import logging
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TEUTONIC_DIR = os.path.dirname(HERE)
sys.path.insert(0, TEUTONIC_DIR)

# Probe knobs are read at eval_torch import time, so any --seeds / --grad-norm-max
# overrides must be applied to os.environ before importing trainability_probe.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--seeds", type=int, default=None)
_pre.add_argument("--norm-weight-max", type=float, default=None)
_pre.add_argument("--grad-norm-max", type=float, default=None)
_pre.add_argument("--param-group-grad-max", type=float, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.seeds is not None:
    os.environ["TEUTONIC_PROBE_SEEDS"] = str(_pre_args.seeds)
if _pre_args.norm_weight_max is not None:
    os.environ["TEUTONIC_FINETUNE_NORM_WEIGHT_MAX"] = str(_pre_args.norm_weight_max)
if _pre_args.grad_norm_max is not None:
    os.environ["TEUTONIC_FINETUNE_GRAD_NORM_MAX"] = str(_pre_args.grad_norm_max)
if _pre_args.param_group_grad_max is not None:
    os.environ["TEUTONIC_FINETUNE_PARAM_GROUP_GRAD_MAX"] = str(_pre_args.param_group_grad_max)

from eval_torch import (  # noqa: E402
    load_model, trainability_probe,
    FINETUNE_NORM_WEIGHT_MAX, FINETUNE_GRAD_NORM_MAX,
    FINETUNE_PARAM_GROUP_GRAD_MAX,
    PROBE_SEEDS, PROBE_BATCH, PROBE_SEQ_LEN,
)

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


def _max_group(r: dict):
    g = r.get("param_group_grad_norms") or {}
    if not g:
        return None
    finite = [v for v in g.values() if isinstance(v, (int, float))
              and v == v and v not in (float("inf"), float("-inf"))]
    return max(finite) if finite else None


def print_row(label: str, r: dict):
    if "error" in r:
        print(f"  {label:30s} {r['repo']:48s}  ERROR: {r['error']}")
        return
    print(f"  {label:30s} {r['repo']:48s} "
          f"max_norm_w={fmt(r.get('max_norm_weight')):>10s} "
          f"global_grad={fmt(r.get('global_grad_norm')):>11s} "
          f"max_group={fmt(_max_group(r)):>11s} "
          f"nq={fmt(r.get('norm_quantization')):>8s} "
          f"ok={r['ok']!s:5s} "
          f"reason={(r.get('reason') or '')[:48]:48s} "
          f"({r['elapsed_s']:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--repos", nargs="*", default=None,
                    help="Override repo list. Default: built-in known-good + known-trick.")
    ap.add_argument("--seeds", type=int, default=None,
                    help="Independent random batches (overrides TEUTONIC_PROBE_SEEDS).")
    ap.add_argument("--norm-weight-max", type=float, default=None,
                    help="LN/RMSNorm |w|.max() cap (overrides TEUTONIC_FINETUNE_NORM_WEIGHT_MAX).")
    ap.add_argument("--grad-norm-max", type=float, default=None,
                    help="Global ||grad||_2 cap (overrides TEUTONIC_FINETUNE_GRAD_NORM_MAX).")
    ap.add_argument("--param-group-grad-max", type=float, default=None,
                    help="Per-group ||grad||_2 cap (overrides TEUTONIC_FINETUNE_PARAM_GROUP_GRAD_MAX).")
    args = ap.parse_args()
    log.info("probe knobs: seeds=%d batch=%d seq_len=%d "
             "norm_weight_max=%.2f grad_norm_max=%.1f param_group_grad_max=%.1f",
             PROBE_SEEDS, PROBE_BATCH, PROBE_SEQ_LEN,
             FINETUNE_NORM_WEIGHT_MAX, FINETUNE_GRAD_NORM_MAX,
             FINETUNE_PARAM_GROUP_GRAD_MAX)

    device = f"cuda:{args.gpu}"
    if not torch.cuda.is_available():
        sys.exit("no CUDA available")

    if args.repos:
        repos = [(r, "user") for r in args.repos]
    else:
        repos = [(r, "GOOD") for r in KNOWN_GOOD] + [(r, "TRICK") for r in KNOWN_TRICK]

    results = []
    print(f"\nProbing {len(repos)} repos on {device}\n")
    print("=" * 160)
    for repo, label in repos:
        r = probe_repo(repo, device)
        r["label"] = label
        results.append(r)
        print_row(label, r)
    print("=" * 160)

    good = [r for r in results if r.get("label") == "GOOD" and "error" not in r]
    trick = [r for r in results if r.get("label") == "TRICK" and "error" not in r]

    def _finite_vals(rs, key, derive=None):
        vals = []
        for r in rs:
            v = derive(r) if derive else r.get(key)
            if v is None:
                continue
            if isinstance(v, float):
                if v != v or v in (float("inf"), float("-inf")):
                    continue
            vals.append(v)
        return vals

    def _stats(vals):
        if not vals:
            return "no finite values"
        return f"min={min(vals):+.4f} max={max(vals):+.4f}"

    for key, derive in (
        ("max_norm_weight", None),
        ("global_grad_norm", None),
        ("max_group_grad", _max_group),
        ("norm_quantization", None),
    ):
        print(f"\nGOOD  {key:>20s}: {_stats(_finite_vals(good, key, derive))}")
        print(f"TRICK {key:>20s}: {_stats(_finite_vals(trick, key, derive))}")

    print(f"\nGOOD  ok={sum(1 for r in good if r.get('ok'))}/{len(good)}  "
          f"TRICK ok={sum(1 for r in trick if r.get('ok'))}/{len(trick)}")

    def _suggest(name, env_var, good_vals, trick_vals, factor=2.0):
        if not good_vals or not trick_vals:
            print(f"  {name}: insufficient data for both cohorts")
            return
        gmax = max(good_vals)
        tmin = min(trick_vals)
        if gmax >= tmin:
            print(f"  {name}: GOOD max {gmax:.4g} >= TRICK min {tmin:.4g} — "
                  f"cohorts overlap, no clean threshold")
            return
        midpoint = (gmax + tmin) / 2
        safe_floor = gmax * factor
        suggested = max(midpoint, safe_floor)
        print(f"  Suggested {env_var}={suggested:.4g} "
              f"(GOOD max {gmax:.4g}, TRICK min {tmin:.4g}, "
              f"midpoint {midpoint:.4g}, {factor}x safety {safe_floor:.4g})")

    print("\n--- Suggested thresholds (calibrated against current cohorts) ---")
    _suggest("norm weight cap", "TEUTONIC_FINETUNE_NORM_WEIGHT_MAX",
             _finite_vals(good, "max_norm_weight"),
             _finite_vals(trick, "max_norm_weight"),
             factor=3.0)
    _suggest("global grad cap", "TEUTONIC_FINETUNE_GRAD_NORM_MAX",
             _finite_vals(good, "global_grad_norm"),
             _finite_vals(trick, "global_grad_norm"),
             factor=2.0)
    _suggest("per-group grad cap", "TEUTONIC_FINETUNE_PARAM_GROUP_GRAD_MAX",
             _finite_vals(good, "max_group_grad", _max_group),
             _finite_vals(trick, "max_group_grad", _max_group),
             factor=2.0)


if __name__ == "__main__":
    main()
