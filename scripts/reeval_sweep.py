#!/usr/bin/env python3
"""Re-evaluate all cached challengers against fixed king losses.

Computes king losses ONCE on a deterministic 102400-sample slice, then
sweeps each challenger against the same cached losses. Saves ~50% GPU time
vs re-running full paired evals.

Run on the GPU box:
    cd /root/workspace
    python3 scripts/reeval_sweep.py [--n 102400] [--resume]
"""
import hashlib
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chain_config
chain_config.load_arch()

from eval.torch_runner import (
    R2, MultiGPUEvaluator, parse_gpu_ids,
    sample_public_holdout, compute_batch_losses,
    trainability_probe, is_accepted,
)
from eval.raw_dataset import raw_dataset_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reeval")

KING_REPO = "teutonic-miner/teutonic-q3-4b-5g6x3hrj-top3"
KING_DIGEST = "sha256:80693d7ee5bfcfe8eb4cad422380fba635d0422dbedf706cd3a1791a20749319"
MODEL_CACHE = pathlib.Path(os.environ.get("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/hippius_models"))
RESULTS_FILE = pathlib.Path("reeval_results.json")
KING_LOSSES_CACHE = pathlib.Path("reeval_king_losses.npy")
EVAL_N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 102400
RESUME = "--resume" in sys.argv
SEED = "reeval:fixed:v1"
BATCH_SIZE = 512


def discover_models():
    """Find all cached challenger snapshots."""
    models = []
    for repo_dir in sorted(MODEL_CACHE.iterdir()):
        if not repo_dir.is_dir():
            continue
        name = repo_dir.name
        if "genesis" in name or "mock" in name:
            continue
        repo = name.replace("--", "/", 1)
        if repo == KING_REPO:
            continue
        snaps = repo_dir / "snapshots"
        if not snaps.exists():
            continue
        for snap in sorted(snaps.iterdir()):
            if snap.is_dir() and snap.name.startswith("sha256-"):
                digest = snap.name.replace("sha256-", "sha256:", 1)
                models.append({"repo": repo, "digest": digest, "path": str(snap)})
    return models


def get_holdout_sequences(r2):
    """Deterministic 102400 sequences from the fixed seed."""
    public_seed = hashlib.blake2b(SEED.encode(), digest_size=8).digest()
    vocab_size = 151936  # Qwen3-4B
    holdout, indices_digest, raw_meta = sample_public_holdout(
        r2, "", public_seed, EVAL_N, 2048, vocab_size=vocab_size,
    )
    log.info("holdout: %d sequences, indices_digest=%s", holdout.shape[0], indices_digest[:16])
    return holdout


def compute_all_losses(evaluator, sequences, label="model"):
    """Compute losses for all sequences in batches."""
    seq_list = sequences.tolist()
    all_losses = []
    n_batches = (len(seq_list) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    for i in range(0, len(seq_list), BATCH_SIZE):
        batch = seq_list[i:i + BATCH_SIZE]
        losses = evaluator.compute_losses(batch)
        all_losses.extend(losses)
        done = len(all_losses)
        elapsed = time.time() - t0
        sps = done / elapsed if elapsed > 0 else 0
        if (i // BATCH_SIZE + 1) % 10 == 0 or done == len(seq_list):
            log.info("%s: %d/%d (%.1f seq/s)", label, done, len(seq_list), sps)
    return np.array(all_losses, dtype=np.float64)


def bootstrap_lcb(king_losses, chall_losses, alpha=0.001, n_bootstrap=10000):
    """Paired bootstrap LCB."""
    d = king_losses - chall_losses
    mu_hat = float(d.mean())
    boot_seed = hashlib.blake2b(f"{SEED}:boot".encode(), digest_size=8).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(boot_seed, "little")))
    n = len(d)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = d[idx].mean()
    lcb = float(np.quantile(boot_means, alpha))
    return mu_hat, lcb


def main():
    r2 = R2()
    gpu_ids = parse_gpu_ids("auto")
    log.info("GPUs: %s, EVAL_N=%d, SEED=%s", gpu_ids, EVAL_N, SEED)

    # Load or compute holdout sequences
    log.info("preparing holdout sequences...")
    sequences = get_holdout_sequences(r2)
    log.info("holdout ready: %s", sequences.shape)

    # Load or compute king losses
    if KING_LOSSES_CACHE.exists() and RESUME:
        log.info("loading cached king losses from %s", KING_LOSSES_CACHE)
        king_losses = np.load(KING_LOSSES_CACHE)
        if len(king_losses) != len(sequences):
            log.warning("king losses length mismatch (%d vs %d), recomputing",
                        len(king_losses), len(sequences))
            king_losses = None
        else:
            log.info("king losses loaded: mean=%.6f", king_losses.mean())
    else:
        king_losses = None

    if king_losses is None:
        log.info("computing king losses on ALL %d GPUs...", len(gpu_ids))
        king_eval = MultiGPUEvaluator(KING_REPO, gpu_ids, label="king",
                                       revision=KING_DIGEST)
        king_losses = compute_all_losses(king_eval, sequences, "king")
        np.save(KING_LOSSES_CACHE, king_losses)
        log.info("king losses saved: mean=%.6f", king_losses.mean())
        king_eval.shutdown()
        del king_eval
        torch.cuda.empty_cache()

    # Discover challengers
    models = discover_models()
    log.info("found %d challenger models to evaluate", len(models))

    # Load existing results
    results = {}
    if RESUME and RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    done_keys = set(results.keys())

    # Sweep challengers
    for i, m in enumerate(models):
        key = f"{m['repo']}@{m['digest'][:19]}"
        if RESUME and key in done_keys:
            log.info("[%d/%d] SKIP %s (done)", i + 1, len(models), m["repo"])
            continue

        log.info("[%d/%d] evaluating %s (%s)", i + 1, len(models), m["repo"], m["digest"][:19])
        t0 = time.time()

        try:
            chall_eval = MultiGPUEvaluator(
                m["path"], gpu_ids, label="challenger",
                revision=None,  # loading from local path
            )
            chall_losses = compute_all_losses(chall_eval, sequences, m["repo"])
            chall_eval.shutdown()
            del chall_eval
            torch.cuda.empty_cache()

            mu_hat, lcb = bootstrap_lcb(king_losses, chall_losses)
            accepted = is_accepted(lcb, 0.0025)
            elapsed = time.time() - t0

            results[key] = {
                "repo": m["repo"],
                "digest": m["digest"],
                "eval_n": EVAL_N,
                "mu_hat": round(mu_hat, 6),
                "lcb": round(lcb, 6),
                "accepted": accepted,
                "avg_king_loss": round(float(king_losses.mean()), 6),
                "avg_challenger_loss": round(float(chall_losses.mean()), 6),
                "wall_time_s": round(elapsed, 1),
            }
            status = "ACCEPTED" if accepted else "REJECTED"
            log.info("  %s mu_hat=%.6f lcb=%.6f king=%.6f chall=%.6f (%.1fs)",
                     status, mu_hat, lcb, king_losses.mean(), chall_losses.mean(), elapsed)

        except Exception as e:
            log.error("  FAILED: %s", e)
            results[key] = {
                "repo": m["repo"],
                "digest": m["digest"],
                "eval_n": EVAL_N,
                "error": str(e),
                "accepted": None,
            }

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    log.info("sweep complete. %d models evaluated. Results in %s", len(results), RESULTS_FILE)


if __name__ == "__main__":
    main()
