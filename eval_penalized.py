#!/usr/bin/env python3
"""Penalized eval — regression-aware paired test for king-vs-challenger.

Instead of a plain bootstrap on d_i = king_loss_i - challenger_loss_i,
this applies a penalty to regressions before running a one-sided t-test:

    u_i = d_i                   if d_i >= 0  (challenger better)
    u_i = (1 + beta) * d_i      if d_i <  0  (challenger worse — amplified)

Equivalently: u_i = d_i - beta * max(-d_i, 0)

The challenger is accepted only if a one-sided t-test rejects
H0: E[u] <= delta at significance level alpha, AND mean(u) > delta.

Usage:
    python eval_penalized.py \\
        --king unconst/Teutonic-I \\
        --challenger unconst/Teutonic-I \\
        --n 10000 --delta 0.01 --beta 1.0 --gpus auto
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from eval_torch import (
    R2,
    MultiGPUEvaluator,
    compute_paired_multi_gpu,
    download_shard,
    extract_sequences,
    get_shard_info,
    parse_gpu_ids,
)

log = logging.getLogger("eval_penalized")


def penalize(d, beta):
    """Apply asymmetric regression penalty to paired differences.

    d: array of d_i = king_loss_i - challenger_loss_i  (positive = challenger better)
    beta: penalty multiplier for regressions (beta=1 means regressions count double)

    Returns u: penalized scores where regressions are amplified.
    """
    return d - beta * np.maximum(-d, 0.0)


def run_penalized_test(king_eval, challenger_eval, r2, shard_key, eval_n,
                       alpha, delta, beta, seq_len, batch_size, seed_str,
                       on_progress=None):
    """Penalized paired t-test on per-token log-loss differences.

    Like run_bootstrap_test but:
    1. Transforms d_i via asymmetric penalty (regressions amplified by 1+beta)
    2. Uses scipy one-sided t-test instead of bootstrap
    3. Reports regression statistics
    """
    n_tokens = get_shard_info(r2, shard_key)
    n_sequences = n_tokens // seq_len
    actual_N = min(eval_n, n_sequences)
    log.info("penalized test: N=%d actual_N=%d alpha=%s delta=%.6f beta=%.1f",
             eval_n, actual_N, alpha, delta, beta)

    seed_material = seed_str.encode()
    seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

    log.info("downloading shard %s ...", shard_key)
    data_offset, shard_data = download_shard(r2, shard_key)

    log.info("extracting %d sequences", actual_N)
    seq_cache = extract_sequences(shard_data, data_offset, eval_indices, seq_len)
    log.info("extracted %d sequences", len(seq_cache))

    batches = [
        eval_indices[i : i + batch_size]
        for i in range(0, len(eval_indices), batch_size)
    ]

    all_diffs = []
    king_sum, chall_sum = 0.0, 0.0
    total_done = 0
    t0 = time.time()

    same_evaluator = king_eval is challenger_eval

    for bi, batch_indices in enumerate(batches):
        token_batches = [seq_cache[idx] for idx in batch_indices]

        if same_evaluator:
            king_losses = king_eval.compute_losses(token_batches)
            chall_losses = king_losses
        else:
            king_losses, chall_losses = compute_paired_multi_gpu(
                king_eval, challenger_eval, token_batches,
            )

        for k_loss, c_loss in zip(king_losses, chall_losses):
            total_done += 1
            king_sum += k_loss
            chall_sum += c_loss
            all_diffs.append(k_loss - c_loss)

        elapsed = time.time() - t0
        seqs_per_sec = total_done / elapsed if elapsed > 0 else 0
        mu_hat = np.mean(all_diffs) if all_diffs else 0.0
        log.info(
            "batch %d/%d | done=%d/%d | mu_hat=%.6f | %.1f seq/s",
            bi + 1, len(batches), total_done, actual_N, mu_hat, seqs_per_sec,
        )

        if on_progress:
            on_progress({
                "done": total_done, "total": actual_N,
                "mu_hat": round(float(mu_hat), 6),
                "avg_king_loss": round(king_sum / total_done, 6),
                "avg_challenger_loss": round(chall_sum / total_done, 6),
                "seqs_per_sec": round(seqs_per_sec, 1),
            })

    elapsed = time.time() - t0
    d = np.array(all_diffs)
    mu_hat = float(d.mean())

    u = penalize(d, beta)
    mean_u = float(u.mean())

    n_regressions = int(np.sum(d < 0))
    regression_frac = n_regressions / len(d) if len(d) > 0 else 0.0

    t_stat_result = stats.ttest_1samp(u, popmean=delta, alternative="greater")
    t_stat = float(t_stat_result.statistic)
    p_value = float(t_stat_result.pvalue)

    accepted = (p_value < alpha) and (mean_u > delta)

    log.info("penalized result: mu_hat=%.6f mean_u=%.6f t_stat=%.4f p_value=%.6f "
             "delta=%.6f beta=%.1f regressions=%d/%d (%.1f%%) accepted=%s",
             mu_hat, mean_u, t_stat, p_value, delta, beta,
             n_regressions, len(d), regression_frac * 100, accepted)

    verdict = {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "mean_u": round(mean_u, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 8),
        "delta": delta,
        "alpha": alpha,
        "beta": beta,
        "n_regressions": n_regressions,
        "regression_fraction": round(regression_frac, 4),
        "N": actual_N,
        "avg_king_loss": round(king_sum / total_done, 6) if total_done else 0,
        "avg_challenger_loss": round(chall_sum / total_done, 6) if total_done else 0,
        "wall_time_s": round(elapsed, 1),
        "seqs_per_sec": round(total_done / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return verdict


def score_from_diffs(d, alpha, delta, beta):
    """Pure-numpy scoring of precomputed diffs — no GPU needed.

    Returns a verdict dict from raw d_i values. Used by experiment scripts
    to re-score the same losses under different (alpha, delta, beta) without
    re-running model inference.
    """
    u = penalize(d, beta)
    mean_u = float(u.mean())
    mu_hat = float(d.mean())

    n_regressions = int(np.sum(d < 0))
    regression_frac = n_regressions / len(d) if len(d) > 0 else 0.0

    t_stat_result = stats.ttest_1samp(u, popmean=delta, alternative="greater")
    t_stat = float(t_stat_result.statistic)
    p_value = float(t_stat_result.pvalue)

    accepted = (p_value < alpha) and (mean_u > delta)

    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "mean_u": round(mean_u, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 8),
        "delta": delta,
        "alpha": alpha,
        "beta": beta,
        "n_regressions": n_regressions,
        "regression_fraction": round(regression_frac, 4),
        "N": len(d),
    }


def score_bootstrap_from_diffs(d, alpha, delta, n_bootstrap=10000, seed=42):
    """Re-run the original bootstrap test on precomputed diffs."""
    mu_hat = float(d.mean())

    boot_rng = np.random.Generator(np.random.PCG64(seed ^ 0xB007))
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = boot_rng.integers(0, len(d), size=len(d))
        boot_means[b] = d[idx].mean()
    lcb = float(np.quantile(boot_means, alpha))

    accepted = lcb > delta

    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta": delta,
        "alpha": alpha,
        "N": len(d),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Penalized eval — regression-aware paired test")
    parser.add_argument("--king", required=True, help="HF repo for king model")
    parser.add_argument("--challenger", required=True, help="HF repo for challenger model")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0, help="Regression penalty multiplier")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--seed", default="test:penalized")
    parser.add_argument("--shard", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for var in ["TEUTONIC_R2_ENDPOINT", "TEUTONIC_R2_ACCESS_KEY", "TEUTONIC_R2_SECRET_KEY"]:
        if var not in os.environ:
            log.error("missing env var: %s", var)
            sys.exit(1)

    gpu_ids = parse_gpu_ids(args.gpus)
    r2 = R2()

    if args.shard:
        shard_key = args.shard
    else:
        manifest = r2.get("dataset/v1/manifest.json")
        if not manifest:
            log.error("could not fetch dataset manifest")
            sys.exit(1)
        shard_key = manifest["shards"][0]["key"]

    same_model = args.king == args.challenger
    if same_model:
        king_eval = MultiGPUEvaluator(args.king, gpu_ids, label="king")
        challenger_eval = king_eval
    else:
        mid = len(gpu_ids) // 2
        king_gpus = gpu_ids[:mid] or gpu_ids[:1]
        chall_gpus = gpu_ids[mid:] or gpu_ids[:1]
        king_eval = MultiGPUEvaluator(args.king, king_gpus, label="king")
        challenger_eval = MultiGPUEvaluator(args.challenger, chall_gpus, label="challenger")

    verdict = run_penalized_test(
        king_eval, challenger_eval,
        r2, shard_key, args.n, args.alpha, args.delta, args.beta,
        args.seq_len, args.batch_size, args.seed,
    )

    king_eval.shutdown()
    if not same_model:
        challenger_eval.shutdown()

    print()
    print("=" * 60)
    print("VERDICT (penalized)")
    print("=" * 60)
    print(json.dumps(verdict, indent=2))
    print("=" * 60)

    return 0 if not verdict["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
