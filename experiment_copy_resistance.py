#!/usr/bin/env python3
"""Copy-resistance experiment: test if noise perturbations can beat the king.

Tests:
1. King vs King baseline (should be 50/50 ties)
2. King vs noise-perturbed copies at various scales
3. Consistency: re-test winners on different shards
4. Vary N to test verdict stability
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from eval_torch import R2, MultiGPUEvaluator, run_bootstrap_test, parse_gpu_ids

log = logging.getLogger("experiment")


def perturb_model(src_dir, dst_dir, noise_scale, seed=None):
    """Copy model and add gaussian noise to weights."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    rng = np.random.default_rng(seed)
    st_files = sorted(Path(dst_dir).glob("*.safetensors"))

    for st_file in st_files:
        sd = load_file(str(st_file))
        new_sd = {}
        for name, tensor in sd.items():
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                gen = torch.Generator()
                gen.manual_seed(int(rng.integers(0, 2**63)))
                noise = torch.randn(tensor.shape, generator=gen, dtype=torch.float32) * noise_scale
                new_sd[name] = (tensor.float() + noise).to(tensor.dtype)
            else:
                new_sd[name] = tensor
        save_file(new_sd, str(st_file))

    log.info("perturbed %s -> %s (noise=%.6f, seed=%s)", src_dir, dst_dir, noise_scale, seed)


def load_local_evaluator(model_dir, gpu_ids, label="model"):
    """Load a model from local directory onto specified GPUs."""
    from eval_torch import load_model
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import torch

    class LocalMultiGPUEvaluator:
        def __init__(self, model_dir, gpu_ids, label):
            self.gpu_ids = gpu_ids
            self.models = {}
            self.devices = {}
            for gid in gpu_ids:
                self.models[gid] = load_model(model_dir, f"cuda:{gid}", f"{label}-gpu{gid}")
                self.devices[gid] = f"cuda:{gid}"
            self.pool = ThreadPoolExecutor(max_workers=len(gpu_ids))
            log.info("%s evaluator ready: %d GPUs %s", label, len(gpu_ids), gpu_ids)

        def compute_losses(self, token_batches):
            from eval_torch import compute_batch_losses
            n_gpus = len(self.gpu_ids)
            if not token_batches:
                return []
            per_gpu = [[] for _ in range(n_gpus)]
            idx_map = [[] for _ in range(n_gpus)]
            for i, batch in enumerate(token_batches):
                g = i % n_gpus
                per_gpu[g].append(batch)
                idx_map[g].append(i)
            futures = {}
            for g_idx, gid in enumerate(self.gpu_ids):
                if per_gpu[g_idx]:
                    fut = self.pool.submit(compute_batch_losses, self.models[gid], per_gpu[g_idx], self.devices[gid])
                    futures[fut] = g_idx
            results = [None] * len(token_batches)
            from concurrent.futures import as_completed
            for fut in as_completed(futures):
                g_idx = futures[fut]
                losses = fut.result()
                for local_i, global_i in enumerate(idx_map[g_idx]):
                    results[global_i] = losses[local_i]
            return results

        def shutdown(self):
            self.pool.shutdown(wait=False)

    return LocalMultiGPUEvaluator(model_dir, gpu_ids, label)


def run_experiment(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    gpu_ids = parse_gpu_ids(args.gpus)
    log.info("GPUs: %s", gpu_ids)

    r2 = R2()

    manifest = r2.get("dataset/v1/manifest.json")
    if not manifest:
        log.error("could not fetch dataset manifest")
        sys.exit(1)
    n_shards = manifest["total_shards"]
    log.info("dataset: %d shards", n_shards)

    king_dir = "/tmp/experiment/king"
    if not os.path.exists(king_dir) or args.redownload:
        log.info("downloading king model from %s", args.king_repo)
        from huggingface_hub import snapshot_download
        if os.path.exists(king_dir):
            shutil.rmtree(king_dir)
        snapshot_download(args.king_repo, local_dir=king_dir,
                         token=os.environ.get("HF_TOKEN") or None)
    else:
        log.info("using cached king at %s", king_dir)

    mid = len(gpu_ids) // 2
    king_gpus = gpu_ids[:mid]
    challenger_gpus = gpu_ids[mid:]
    log.info("king GPUs: %s, challenger GPUs: %s", king_gpus, challenger_gpus)

    results = []

    # --- Experiment 1: King vs King baseline ---
    if not args.skip_baseline:
        log.info("=" * 60)
        log.info("EXPERIMENT 1: King vs King (baseline)")
        log.info("=" * 60)

        king_eval = load_local_evaluator(king_dir, king_gpus, "king")
        king_eval2 = load_local_evaluator(king_dir, challenger_gpus, "king-copy")

        for trial in range(3):
            shard_idx = (trial * 37) % n_shards
            shard_key = manifest["shards"][shard_idx]["key"]
            seed_str = f"baseline:{trial}"

            verdict = run_bootstrap_test(
                king_eval, king_eval2, r2, shard_key,
                args.n, args.alpha, args.delta, args.seq_len, args.batch_size,
                seed_str, n_bootstrap=args.n_bootstrap,
            )
            verdict["experiment"] = "baseline"
            verdict["trial"] = trial
            verdict["shard_idx"] = shard_idx
            results.append(verdict)
            log.info("BASELINE trial %d: verdict=%s mu_hat=%.6f lcb=%.6f",
                     trial, verdict["verdict"], verdict["mu_hat"], verdict["lcb"])

        king_eval2.shutdown()
        del king_eval2
        torch.cuda.empty_cache()

        king_eval.shutdown()
        del king_eval
        torch.cuda.empty_cache()

    # --- Experiment 2: Noise perturbations at various scales ---
    log.info("=" * 60)
    log.info("EXPERIMENT 2: Noise perturbations")
    log.info("=" * 60)

    noise_scales = [float(x) for x in args.noise_scales.split(",")]
    seeds_per_scale = args.seeds_per_scale

    for noise_scale in noise_scales:
        for seed in range(seeds_per_scale):
            challenger_dir = f"/tmp/experiment/challenger_n{noise_scale}_s{seed}"
            log.info("--- noise=%.6f seed=%d ---", noise_scale, seed)

            perturb_model(king_dir, challenger_dir, noise_scale, seed=seed)

            king_eval = load_local_evaluator(king_dir, king_gpus, "king")
            chall_eval = load_local_evaluator(challenger_dir, challenger_gpus,
                                              f"chall-n{noise_scale}-s{seed}")

            shard_idx = (hash(f"{noise_scale}:{seed}") % n_shards) % n_shards
            shard_key = manifest["shards"][abs(shard_idx)]["key"]
            seed_str = f"perturb:{noise_scale}:{seed}"

            verdict = run_bootstrap_test(
                king_eval, chall_eval, r2, shard_key,
                args.n, args.alpha, args.delta, args.seq_len, args.batch_size,
                seed_str, n_bootstrap=args.n_bootstrap,
            )
            verdict["experiment"] = "perturbation"
            verdict["noise_scale"] = noise_scale
            verdict["seed"] = seed
            verdict["shard_idx"] = shard_idx
            results.append(verdict)

            log.info("PERTURB noise=%.6f seed=%d: verdict=%s mu_hat=%.6f lcb=%.6f king_loss=%.6f chall_loss=%.6f",
                     noise_scale, seed, verdict["verdict"], verdict["mu_hat"],
                     verdict["lcb"], verdict["avg_king_loss"], verdict["avg_challenger_loss"])

            won = verdict["accepted"]

            # If won, re-test on different shards for consistency
            if won and not args.skip_consistency:
                log.info("  WINNER! Re-testing on %d more shards...", args.consistency_trials)
                for retrial in range(args.consistency_trials):
                    shard_idx2 = ((abs(hash(f"{noise_scale}:{seed}:{retrial+1}")) + retrial * 97) % n_shards)
                    shard_key2 = manifest["shards"][shard_idx2]["key"]
                    seed_str2 = f"consistency:{noise_scale}:{seed}:{retrial}"

                    v2 = run_bootstrap_test(
                        king_eval, chall_eval, r2, shard_key2,
                        args.n, args.alpha, args.delta, args.seq_len, args.batch_size,
                        seed_str2, n_bootstrap=args.n_bootstrap,
                    )
                    v2["experiment"] = "consistency"
                    v2["noise_scale"] = noise_scale
                    v2["seed"] = seed
                    v2["retrial"] = retrial
                    v2["shard_idx"] = shard_idx2
                    results.append(v2)
                    log.info("  CONSISTENCY retrial %d: verdict=%s mu_hat=%.6f lcb=%.6f",
                             retrial, v2["verdict"], v2["mu_hat"], v2["lcb"])

            # Test with different N values
            if not args.skip_vary_n:
                for test_n in [1000, 5000, 50000]:
                    if test_n == args.n:
                        continue
                    seed_str_n = f"vary_n:{noise_scale}:{seed}:{test_n}"
                    vn = run_bootstrap_test(
                        king_eval, chall_eval, r2, shard_key,
                        test_n, args.alpha, args.delta, args.seq_len, args.batch_size,
                        seed_str_n, n_bootstrap=args.n_bootstrap,
                    )
                    vn["experiment"] = "vary_n"
                    vn["noise_scale"] = noise_scale
                    vn["seed"] = seed
                    vn["test_n"] = test_n
                    results.append(vn)
                    log.info("  N=%d: verdict=%s mu_hat=%.6f lcb=%.6f",
                             test_n, vn["verdict"], vn["mu_hat"], vn["lcb"])

            chall_eval.shutdown()
            king_eval.shutdown()
            del chall_eval, king_eval
            torch.cuda.empty_cache()

            # Clean up challenger weights
            shutil.rmtree(challenger_dir, ignore_errors=True)

    # --- Save results ---
    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s (%d entries)", out_path, len(results))

    # --- Summary ---
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    baselines = [r for r in results if r["experiment"] == "baseline"]
    if baselines:
        mu_hats = [r["mu_hat"] for r in baselines]
        log.info("Baseline (king vs king): %d trials, mu_hat range [%.6f, %.6f]",
                 len(baselines), min(mu_hats), max(mu_hats))

    perturbations = [r for r in results if r["experiment"] == "perturbation"]
    wins = [r for r in perturbations if r["accepted"]]
    losses = [r for r in perturbations if not r["accepted"]]
    log.info("Perturbations: %d total, %d won (%.1f%%), %d lost",
             len(perturbations), len(wins),
             100 * len(wins) / len(perturbations) if perturbations else 0,
             len(losses))

    for ns in sorted(set(r.get("noise_scale", 0) for r in perturbations)):
        subset = [r for r in perturbations if r.get("noise_scale") == ns]
        w = sum(1 for r in subset if r["accepted"])
        mu_hats = [r["mu_hat"] for r in subset]
        lcbs = [r["lcb"] for r in subset]
        log.info("  noise=%.6f: %d/%d won (%.0f%%) mu_hat=[%.6f,%.6f] lcb=[%.6f,%.6f]",
                 ns, w, len(subset), 100 * w / len(subset) if subset else 0,
                 min(mu_hats), max(mu_hats), min(lcbs), max(lcbs))

    consistencies = [r for r in results if r["experiment"] == "consistency"]
    if consistencies:
        consistent_wins = sum(1 for r in consistencies if r["accepted"])
        log.info("Consistency re-tests: %d/%d still won (%.0f%%)",
                 consistent_wins, len(consistencies),
                 100 * consistent_wins / len(consistencies))

    vary_ns = [r for r in results if r["experiment"] == "vary_n"]
    if vary_ns:
        for test_n in sorted(set(r.get("test_n", 0) for r in vary_ns)):
            subset = [r for r in vary_ns if r.get("test_n") == test_n]
            w = sum(1 for r in subset if r["accepted"])
            log.info("  N=%d: %d/%d accepted", test_n, w, len(subset))


def main():
    parser = argparse.ArgumentParser(description="Copy-resistance experiment")
    parser.add_argument("--king-repo", default="unconst/Teutonic-I")
    parser.add_argument("--n", type=int, default=10000, help="Default eval N")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--delta", type=float, default=0.01, help="Minimum effect threshold in nats/token")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap replicates")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--noise-scales", default="0.0001,0.001,0.005,0.01",
                        help="Comma-separated noise scales to test")
    parser.add_argument("--seeds-per-scale", type=int, default=3,
                        help="Number of random seeds per noise scale")
    parser.add_argument("--consistency-trials", type=int, default=3,
                        help="Number of re-tests for winners")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-consistency", action="store_true")
    parser.add_argument("--skip-vary-n", action="store_true")
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--output", default="/tmp/experiment/results.json")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
