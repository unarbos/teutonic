#!/usr/bin/env python3
"""Compare bootstrap eval vs penalized eval on the same perturbation trials.

Computes model losses once per trial, then re-scores with both the original
bootstrap test and the new penalized t-test at multiple beta values. This
directly answers whether the penalized design is more resistant to copy attacks
without recomputing expensive GPU forward passes.
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_torch import (
    R2,
    compute_paired_multi_gpu,
    download_shard,
    extract_sequences,
    get_shard_info,
    load_model,
    compute_batch_losses,
    parse_gpu_ids,
)
from eval_penalized import score_from_diffs, score_bootstrap_from_diffs

log = logging.getLogger("experiment_penalized")


class LocalMultiGPUEvaluator:
    def __init__(self, model_dir, gpu_ids, label):
        from concurrent.futures import ThreadPoolExecutor
        self.gpu_ids = gpu_ids
        self.models = {}
        self.devices = {}
        for gid in gpu_ids:
            self.models[gid] = load_model(model_dir, f"cuda:{gid}", f"{label}-gpu{gid}")
            self.devices[gid] = f"cuda:{gid}"
        self.pool = ThreadPoolExecutor(max_workers=len(gpu_ids))
        log.info("%s evaluator ready: %d GPUs %s", label, len(gpu_ids), gpu_ids)

    def compute_losses(self, token_batches):
        from concurrent.futures import as_completed
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
        for fut in as_completed(futures):
            g_idx = futures[fut]
            losses = fut.result()
            for local_i, global_i in enumerate(idx_map[g_idx]):
                results[global_i] = losses[local_i]
        return results

    def shutdown(self):
        self.pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Perturbation strategies (same as experiment_gaming.py)
# ---------------------------------------------------------------------------

def perturb_model(src_dir, dst_dir, noise_scale, strategy, seed=None):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    rng = np.random.default_rng(seed)

    layer_filters = {
        "bias_only": lambda name: "bias" in name.lower(),
        "embed_only": lambda name: any(k in name.lower() for k in ["embed", "lm_head"]),
        "early_layers": lambda name: any(f"layers.{i}." in name for i in range(6)),
        "late_layers": lambda name: any(f"layers.{i}." in name for i in range(18, 24)),
    }

    for st_file in sorted(Path(dst_dir).glob("*.safetensors")):
        sd = load_file(str(st_file))
        new_sd = {}
        for name, tensor in sd.items():
            if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                new_sd[name] = tensor
                continue

            if strategy in layer_filters and not layer_filters[strategy](name):
                new_sd[name] = tensor
                continue

            gen = torch.Generator()
            gen.manual_seed(int(rng.integers(0, 2**63)))
            noise = torch.randn(tensor.shape, generator=gen, dtype=torch.float32)

            if strategy == "magnitude_scaled":
                new_sd[name] = (tensor.float() * (1.0 + noise_scale * noise)).to(tensor.dtype)
            else:
                new_sd[name] = (tensor.float() + noise_scale * noise).to(tensor.dtype)
        save_file(new_sd, str(st_file))
    return dst_dir


def collect_paired_losses(king_eval, chall_eval, r2, shard_key, eval_n, seq_len,
                          batch_size, seed_str):
    """Run GPU inference once and return raw (king_losses, chall_losses) arrays."""
    n_tokens = get_shard_info(r2, shard_key)
    n_sequences = n_tokens // seq_len
    actual_N = min(eval_n, n_sequences)

    seed_material = seed_str.encode()
    seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

    data_offset, shard_data = download_shard(r2, shard_key)
    seq_cache = extract_sequences(shard_data, data_offset, eval_indices, seq_len)

    batches = [
        eval_indices[i : i + batch_size]
        for i in range(0, len(eval_indices), batch_size)
    ]

    all_king = []
    all_chall = []
    t0 = time.time()

    same_evaluator = king_eval is chall_eval

    for bi, batch_indices in enumerate(batches):
        token_batches = [seq_cache[idx] for idx in batch_indices]

        if same_evaluator:
            king_losses = king_eval.compute_losses(token_batches)
            chall_losses = king_losses
        else:
            king_losses, chall_losses = compute_paired_multi_gpu(
                king_eval, chall_eval, token_batches,
            )

        all_king.extend(king_losses)
        all_chall.extend(chall_losses)

        if (bi + 1) % 10 == 0 or bi == len(batches) - 1:
            elapsed = time.time() - t0
            log.info("batch %d/%d | done=%d/%d | %.1f seq/s",
                     bi + 1, len(batches), len(all_king), actual_N,
                     len(all_king) / elapsed if elapsed > 0 else 0)

    elapsed = time.time() - t0
    log.info("inference complete: %d sequences in %.1fs (%.1f seq/s)",
             len(all_king), elapsed, len(all_king) / elapsed if elapsed > 0 else 0)

    return np.array(all_king), np.array(all_chall), elapsed


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
        from huggingface_hub import snapshot_download
        log.info("downloading king model from %s", args.king_repo)
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

    strategies = [s.strip() for s in args.strategies.split(",")]
    noise_scales = [float(x) for x in args.noise_scales.split(",")]
    betas = [float(x) for x in args.betas.split(",")]

    results = []

    for strategy in strategies:
        log.info("=" * 60)
        log.info("STRATEGY: %s", strategy)
        log.info("=" * 60)

        for noise_scale in noise_scales:
            king_eval = LocalMultiGPUEvaluator(king_dir, king_gpus, "king")

            for seed in range(args.trials_per_config):
                log.info("--- %s | noise=%.6f | trial=%d/%d ---",
                         strategy, noise_scale, seed + 1, args.trials_per_config)

                challenger_dir = f"/tmp/experiment/challenger_{strategy}_n{noise_scale}_s{seed}"
                perturb_model(king_dir, challenger_dir, noise_scale, strategy, seed=seed)

                chall_eval = LocalMultiGPUEvaluator(challenger_dir, challenger_gpus,
                                                    f"chall-{strategy}-{noise_scale}-{seed}")

                shard_idx = abs(hash(f"{strategy}:{noise_scale}:{seed}")) % n_shards
                shard_key = manifest["shards"][shard_idx]["key"]
                seed_str = f"penalized_cmp:{strategy}:{noise_scale}:{seed}"

                king_losses, chall_losses, wall_time = collect_paired_losses(
                    king_eval, chall_eval, r2, shard_key,
                    args.n, args.seq_len, args.batch_size, seed_str,
                )

                d = king_losses - chall_losses

                trial_record = {
                    "strategy": strategy,
                    "noise_scale": noise_scale,
                    "seed": seed,
                    "shard_idx": shard_idx,
                    "N": len(d),
                    "avg_king_loss": round(float(king_losses.mean()), 6),
                    "avg_chall_loss": round(float(chall_losses.mean()), 6),
                    "mu_hat": round(float(d.mean()), 6),
                    "d_std": round(float(d.std()), 6),
                    "n_regressions": int(np.sum(d < 0)),
                    "regression_frac": round(float(np.mean(d < 0)), 4),
                    "wall_time_s": round(wall_time, 1),
                }

                bootstrap_verdict = score_bootstrap_from_diffs(
                    d, args.alpha, args.delta, n_bootstrap=args.n_bootstrap, seed=seed,
                )
                trial_record["bootstrap"] = bootstrap_verdict

                penalized_verdicts = {}
                for beta in betas:
                    pv = score_from_diffs(d, args.alpha, args.delta, beta)
                    penalized_verdicts[f"beta_{beta}"] = pv
                trial_record["penalized"] = penalized_verdicts

                results.append(trial_record)

                boot_v = "WIN" if bootstrap_verdict["accepted"] else "lose"
                pen_strs = []
                for beta in betas:
                    pv = penalized_verdicts[f"beta_{beta}"]
                    pen_strs.append(f"b{beta}={'WIN' if pv['accepted'] else 'lose'}")

                log.info("RESULT %s noise=%.6f seed=%d: bootstrap=%s %s | "
                         "mu_hat=%.6f regr=%.1f%% king=%.4f chall=%.4f",
                         strategy, noise_scale, seed, boot_v, " ".join(pen_strs),
                         trial_record["mu_hat"],
                         trial_record["regression_frac"] * 100,
                         trial_record["avg_king_loss"],
                         trial_record["avg_chall_loss"])

                chall_eval.shutdown()
                del chall_eval
                torch.cuda.empty_cache()
                shutil.rmtree(challenger_dir, ignore_errors=True)

            king_eval.shutdown()
            del king_eval
            torch.cuda.empty_cache()

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s (%d trials)", out_path, len(results))

    print_comparison(results, strategies, noise_scales, betas, args.alpha, args.delta)


def print_comparison(results, strategies, noise_scales, betas, alpha, delta):
    print()
    print("=" * 100)
    print("BOOTSTRAP vs PENALIZED COMPARISON")
    print(f"alpha={alpha}  delta={delta}")
    print("=" * 100)

    header = f"{'Strategy':<18} {'Noise':>10} {'Trials':>7} {'Boot':>6}"
    for beta in betas:
        header += f" {'b='+str(beta):>8}"
    header += f" {'mu_hat':>10} {'regr%':>7}"
    print(header)
    print("-" * len(header))

    for strategy in strategies:
        for ns in noise_scales:
            subset = [r for r in results
                      if r["strategy"] == strategy and r["noise_scale"] == ns]
            if not subset:
                continue

            total = len(subset)
            boot_wins = sum(1 for r in subset if r["bootstrap"]["accepted"])

            row = f"{strategy:<18} {ns:>10.5f} {total:>7} {boot_wins:>6}"

            for beta in betas:
                key = f"beta_{beta}"
                pen_wins = sum(1 for r in subset if r["penalized"][key]["accepted"])
                row += f" {pen_wins:>8}"

            avg_mu = np.mean([r["mu_hat"] for r in subset])
            avg_regr = np.mean([r["regression_frac"] for r in subset]) * 100

            row += f" {avg_mu:>10.6f} {avg_regr:>6.1f}%"
            print(row)

    print()
    print("--- Detailed Comparison ---")
    print()

    any_disagreement = False
    for r in results:
        boot_accepted = r["bootstrap"]["accepted"]
        for beta in betas:
            pen_accepted = r["penalized"][f"beta_{beta}"]["accepted"]
            if boot_accepted != pen_accepted:
                any_disagreement = True
                direction = "BOOT_WIN+PEN_LOSE" if boot_accepted else "BOOT_LOSE+PEN_WIN"
                print(f"  DISAGREEMENT [{direction}]: {r['strategy']} noise={r['noise_scale']} "
                      f"seed={r['seed']} beta={beta} | "
                      f"mu_hat={r['mu_hat']:.6f} regr={r['regression_frac']*100:.1f}%")
                pv = r["penalized"][f"beta_{beta}"]
                print(f"    penalized: mean_u={pv['mean_u']:.6f} t={pv['t_stat']:.4f} "
                      f"p={pv['p_value']:.8f}")

    if not any_disagreement:
        print("  No disagreements — both methods agree on all trials.")
        print("  (Expected: neither accepts random perturbations)")

    print()
    print("--- Regression Statistics ---")
    print()

    for strategy in strategies:
        subset = [r for r in results if r["strategy"] == strategy]
        if not subset:
            continue
        regr_fracs = [r["regression_frac"] for r in subset]
        print(f"  {strategy}: regression_frac mean={np.mean(regr_fracs)*100:.1f}% "
              f"min={np.min(regr_fracs)*100:.1f}% max={np.max(regr_fracs)*100:.1f}%")

    print()
    print("--- Penalized Score Shift ---")
    print(f"  How much does the penalty reduce mean_u vs mu_hat?")
    print()
    for beta in betas:
        shifts = []
        for r in results:
            mu = r["mu_hat"]
            mean_u = r["penalized"][f"beta_{beta}"]["mean_u"]
            shifts.append(mean_u - mu)
        avg_shift = np.mean(shifts)
        print(f"  beta={beta}: avg(mean_u - mu_hat) = {avg_shift:.6f}")

    print()
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap vs Penalized eval comparison")
    parser.add_argument("--king-repo", default="kt3202/Teutonic-I-test")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--strategies", default="uniform,magnitude_scaled,bias_only,embed_only")
    parser.add_argument("--noise-scales", default="0.00001,0.0001,0.001")
    parser.add_argument("--betas", default="0,0.5,1,2,3", help="Comma-separated beta values to sweep")
    parser.add_argument("--trials-per-config", type=int, default=5)
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--output", default="/tmp/experiment/results_penalized_cmp.json")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
