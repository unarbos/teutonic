#!/usr/bin/env python3
"""Smoke test eval of Teutonic-XXIV without going through the prod eval server.

Forks smoke_eval_teutonic_viii.py. Validates: model loads at ~24B total /
~8B active, trainability probe passes, paired loss == 0 (king == challenger),
no OOM at chosen batch size, throughput estimate.

Quasar specifics:
- We pre-import teutonic.quasar so AutoModelForCausalLM resolves "quasar"
  without trust_remote_code.
- We default --batch-size to 64 (vs Teutonic-VIII's 128) because total
  weights are ~3x larger; bump if VRAM headroom permits.
- The MoE+latent-memory path only supports attn_implementation eager reliably;
  load_model already falls back through flash_attention_2 -> sdpa -> eager.

Usage on Targon (the GPU box):
    bash -c '. ~/env.sh && /root/eval-venv/bin/python smoke_eval_teutonic_xxiv.py'
"""
import argparse
import json
import logging
import os
import sys
import time

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("smoke")

# Force the workspace's vendored eval_torch (with reset_state hooks) to win
# over /root/eval_torch.py. The staging deploy places eval_torch.py next to
# this script; we put the script dir at sys.path[0] AFTER any other inserts.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
sys.path.insert(0, _script_dir)
# A 50GB MoE seed needs more than the default 600s prefetch budget.
os.environ.setdefault("HF_PREFETCH_TIMEOUT", "3600")
import teutonic.quasar  # noqa: F401

from eval_torch import (
    R2, MultiGPUEvaluator, run_bootstrap_test, parse_gpu_ids,
    trainability_probe, download_shard, get_shard_info,
)
import eval_torch as _et


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="unconst/Teutonic-XXIV")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("EVAL_BATCH_SIZE", "64")))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--shard", default=None)
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus)
    mid = max(1, len(gpu_ids) // 2)
    king_gpus = gpu_ids[:mid]
    chall_gpus = gpu_ids[mid:] or king_gpus
    log.info("gpus=%s  king=%s  challenger=%s",
             gpu_ids, king_gpus, chall_gpus)

    r2 = R2()
    if args.shard:
        shard_key = args.shard
    else:
        manifest = r2.ds_get("dataset/v2/manifest.json") \
            or r2.get("dataset/v1/manifest.json")
        if not manifest:
            log.error("could not fetch dataset manifest")
            sys.exit(1)
        shard_key = manifest["shards"][0]["key"]
    log.info("shard: %s", shard_key)

    log.info("=== loading king (%d GPUs) ===", len(king_gpus))
    t0 = time.time()
    king_eval = MultiGPUEvaluator(args.repo, king_gpus, label="king")
    log.info("king load wall: %.1fs", time.time() - t0)

    log.info("=== probing king ===")
    probe = trainability_probe(king_eval.models[king_gpus[0]])
    log.info("king probe: ok=%s before=%.4f after=%.4f delta=%.4f reason=%s",
             probe["ok"], probe["loss_before"], probe["loss_after"],
             probe["delta"], probe["reason"])
    if not probe["ok"]:
        log.error("king probe failed; aborting")
        sys.exit(1)

    same = (king_gpus == chall_gpus)
    if same:
        log.info("=== king == challenger (same evaluator) ===")
        chall_eval = king_eval
    else:
        log.info("=== loading challenger (%d GPUs) ===", len(chall_gpus))
        t0 = time.time()
        chall_eval = MultiGPUEvaluator(args.repo, chall_gpus, label="challenger")
        log.info("challenger load wall: %.1fs", time.time() - t0)
        probe2 = trainability_probe(chall_eval.models[chall_gpus[0]])
        log.info("challenger probe: ok=%s delta=%.4f", probe2["ok"], probe2["delta"])

    for g in gpu_ids:
        free, total = torch.cuda.mem_get_info(g)
        log.info("gpu %d: %.1f/%.1f GB used",
                 g, (total - free) / 1e9, total / 1e9)

    log.info("=== pre-warming shard cache for %s ===", shard_key)
    try:
        download_shard(r2, shard_key)
    except Exception as e:
        log.warning("shard pre-warm failed (%s); will rely on bootstrap test fetch", e)

    _orig_get_shard_info = _et.get_shard_info

    def _get_shard_info_cached(r2_, key):
        import pathlib
        cache_path = pathlib.Path(_et.SHARD_CACHE_DIR) / key.replace("/", "_")
        if cache_path.exists():
            raw = cache_path.read_bytes()
            offset = _et._parse_npy_header(raw)
            n_bytes = len(raw) - offset
            return n_bytes // 4
        return _orig_get_shard_info(r2_, key)

    _et.get_shard_info = _get_shard_info_cached

    log.info("=== running bootstrap test (eval_n=%d batch=%d seq=%d) ===",
             args.eval_n, args.batch_size, args.seq_len)
    t0 = time.time()
    verdict = run_bootstrap_test(
        king_eval, chall_eval,
        r2, shard_key,
        eval_n=args.eval_n,
        alpha=0.001,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed_str="smoke:teutonic-xxiv",
        n_bootstrap=1000,
    )
    elapsed = time.time() - t0
    log.info("bootstrap wall: %.1fs", elapsed)
    log.info("verdict: %s", json.dumps(verdict, indent=2))

    # king == challenger should produce mu_hat exactly 0 (same model, same
    # tokens, same RNG paths). Anything else is a state-leak bug.
    mu = verdict.get("mu_hat", float("nan"))
    if abs(mu) > 1e-6:
        log.error("paired loss diff is non-zero with king==challenger: mu_hat=%s "
                  "(state leak across forwards?)", mu)
    else:
        log.info("paired loss diff is zero — stateless contract holds")

    seq_per_s = verdict["N"] / elapsed if elapsed > 0 else 0
    log.info("throughput: %.1f seq/s -> %.0f s for 10000 seqs",
             seq_per_s, 10000 / seq_per_s if seq_per_s > 0 else float("inf"))

    king_eval.shutdown()
    if not same:
        chall_eval.shutdown()


if __name__ == "__main__":
    main()
