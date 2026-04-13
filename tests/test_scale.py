#!/usr/bin/env python3
"""Model scale-up test: ramp from 7M to 1B params on GPU.

Runs three phases per tier:
  A) Single miner sanity check (1 window)
  B) Multi-miner scoring pipeline (2 honest + 1 cheater, 5 windows)
  C) Convergence check (10 windows, eval loss must decrease)

Usage:
    TEUTONIC_DEVICE=cuda python tests/test_scale.py --tier 1
    TEUTONIC_DEVICE=cuda python tests/test_scale.py --tier 3
    TEUTONIC_DEVICE=cuda python tests/test_scale.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from teutonic.compress import TopKCompressor, compress_model_gradients
from teutonic.dataset.synthetic import SyntheticDataset
from teutonic.hparams import HParams
from teutonic.logging import setup_logging
from teutonic.model import LlamaConfig, TinyLlama
from teutonic.probe_spec import select_probe_params
from teutonic.sampler import MinerSampler
from teutonic.storage.local import LocalFileStorage
from teutonic.submission import MinerSubmission

from neurons.miner import Miner
from neurons.trainer import train_window
from neurons.validator import Validator

DEVICE = os.environ.get("TEUTONIC_DEVICE", "cpu")


# ──────────────────────────────────────────────────────────────────────────
# Tier configurations
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TierConfig:
    name: str
    cfg: LlamaConfig
    hp: HParams
    dataset_size: int
    n_phase_b_windows: int = 5
    n_phase_c_windows: int = 10


TIERS: dict[int, TierConfig] = {
    1: TierConfig(
        name="7M params",
        cfg=LlamaConfig(vocab_size=4096, hidden_dim=256, intermediate_dim=768,
                        n_layers=6, n_heads=4, seq_len=256),
        hp=HParams(max_batches=8, min_batches=2, micro_bs=4, topk=256,
                   lr=1e-3, outer_lr=0.4, n_probes=3, n_probe_params=3),
        dataset_size=4096,
    ),
    2: TierConfig(
        name="36M params",
        cfg=LlamaConfig(vocab_size=8192, hidden_dim=512, intermediate_dim=1536,
                        n_layers=8, n_heads=8, seq_len=512),
        hp=HParams(max_batches=8, min_batches=2, micro_bs=2, topk=512,
                   lr=5e-4, outer_lr=0.4, n_probes=3, n_probe_params=3),
        dataset_size=4096,
    ),
    3: TierConfig(
        name="110M params",
        cfg=LlamaConfig(vocab_size=16384, hidden_dim=768, intermediate_dim=2048,
                        n_layers=12, n_heads=12, seq_len=512),
        hp=HParams(max_batches=6, min_batches=2, micro_bs=2, topk=1024,
                   lr=3e-4, outer_lr=0.3, n_probes=2, n_probe_params=3),
        dataset_size=4096,
    ),
    4: TierConfig(
        name="271M params",
        cfg=LlamaConfig(vocab_size=32000, hidden_dim=1024, intermediate_dim=2816,
                        n_layers=16, n_heads=16, seq_len=1024),
        hp=HParams(max_batches=4, min_batches=2, micro_bs=1, topk=2048,
                   lr=2e-4, outer_lr=0.3, n_probes=2, n_probe_params=2),
        dataset_size=4096,
        n_phase_b_windows=3,
        n_phase_c_windows=5,
    ),
    5: TierConfig(
        name="941M params",
        cfg=LlamaConfig(vocab_size=32000, hidden_dim=2048, intermediate_dim=5504,
                        n_layers=16, n_heads=16, seq_len=512),
        hp=HParams(max_batches=2, min_batches=1, micro_bs=1, topk=4096,
                   lr=1e-4, outer_lr=0.2, n_probes=1, n_probe_params=2),
        dataset_size=4096,
        n_phase_b_windows=2,
        n_phase_c_windows=3,
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def make_model(cfg: LlamaConfig, seed: int = 0) -> TinyLlama:
    torch.manual_seed(seed)
    return TinyLlama(cfg)


def vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def eval_loss(model: nn.Module, dataset, n_samples: int = 32) -> float:
    model.eval()
    dev = next(model.parameters()).device
    total = 0.0
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            tokens = dataset[i].unsqueeze(0).to(dev)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            total += loss.item()
    model.train()
    return total / min(n_samples, len(dataset))


class CheatingMiner:
    def __init__(self, uid, model, dataset, storage, hparams, device=DEVICE):
        self.uid = uid
        self.model = model.to(device)
        self.dataset = dataset
        self.storage = storage
        self.hp = hparams
        self.compressor = TopKCompressor(topk=hparams.topk)
        self.device = device

    async def train_window(self, window: int) -> MinerSubmission:
        sampler = MinerSampler(self.dataset, self.uid, window,
                               max_batches=self.hp.max_batches, micro_bs=self.hp.micro_bs)
        param_info = {name: p.numel() for name, p in self.model.named_parameters()}
        pp = select_probe_params(window, self.uid, param_info,
                                 self.hp.n_probe_params, self.hp.probe_slice_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr)
        result = train_window(self.model, self.dataset, sampler, optimizer,
                              device=self.device, probe_params=pp)
        compressed = compress_model_gradients(self.model, self.compressor)
        compressed = {
            pname: {"idxs": comp["idxs"], "vals": torch.randn_like(comp["vals"]), "shape": comp["shape"]}
            for pname, comp in compressed.items()
        }
        submission = MinerSubmission(
            uid=self.uid, window=window, compressed_gradients=compressed,
            loss_ledger=result["loss_ledger"], n_batches_trained=result["n_batches_trained"],
            grad_probes=result["grad_probes"],
        )
        await self.storage.put(submission.storage_key(), submission.to_dict())
        return submission


# ──────────────────────────────────────────────────────────────────────────
# Phase A: Single miner sanity check
# ──────────────────────────────────────────────────────────────────────────

async def phase_a(tier: TierConfig, tmpdir: str) -> dict[str, Any]:
    print("\n  --- Phase A: Single Miner Sanity Check (1 window) ---")
    cfg, hp = tier.cfg, tier.hp
    dataset = SyntheticDataset(size=tier.dataset_size, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, seed=42)
    storage = LocalFileStorage(tmpdir + "/a")
    shared_init = make_model(cfg).state_dict()

    m = make_model(cfg)
    m.load_state_dict(copy.deepcopy(shared_init))
    miner = Miner(uid=1, model=m, dataset=dataset, storage=storage, hparams=hp, device=DEVICE)

    v = make_model(cfg)
    v.load_state_dict(copy.deepcopy(shared_init))
    val = Validator(uid=0, model=v, dataset=dataset, storage=storage, hparams=hp, device=DEVICE)

    reset_vram()

    sync()
    t_train = time.time()
    await miner.train_window(0)
    sync()
    t_train = time.time() - t_train

    vram_after_train = vram_gb()

    sync()
    t_eval = time.time()
    results = await val.evaluate_window(0, [1])
    sync()
    t_eval = time.time() - t_eval

    vram_after_eval = vram_gb()
    r = results[0]

    print(f"    Train:  {t_train:.2f}s")
    print(f"    Eval:   {t_eval:.2f}s")
    print(f"    VRAM:   {vram_after_eval:.2f} GB (train: {vram_after_train:.2f} GB)")
    print(f"    Score:  {r.final_score:.4f}  (loss={r.loss_score:.2f}  probe={r.probe_score:.3f}  consist={r.consistency_score:.3f})")

    ok = r.final_score > 0.9
    print(f"    {'PASS' if ok else 'FAIL'}: honest miner score {'> 0.9' if ok else f'= {r.final_score:.4f}'}")

    return {
        "train_s": t_train, "eval_s": t_eval,
        "vram_gb": vram_after_eval, "score": r.final_score,
        "loss_score": r.loss_score, "probe_score": r.probe_score,
        "consistency_score": r.consistency_score, "ok": ok,
    }


# ──────────────────────────────────────────────────────────────────────────
# Phase B: Multi-miner scoring pipeline
# ──────────────────────────────────────────────────────────────────────────

async def phase_b(tier: TierConfig, tmpdir: str) -> dict[str, Any]:
    n_windows = tier.n_phase_b_windows
    print(f"\n  --- Phase B: 2 Honest + 1 Cheater ({n_windows} windows) ---")
    cfg, hp = tier.cfg, tier.hp
    dataset = SyntheticDataset(size=tier.dataset_size, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, seed=42)
    storage = LocalFileStorage(tmpdir + "/b")
    shared_init = make_model(cfg).state_dict()

    miners = []
    for uid in [1, 2]:
        m = make_model(cfg)
        m.load_state_dict(copy.deepcopy(shared_init))
        miners.append(Miner(uid=uid, model=m, dataset=dataset, storage=storage, hparams=hp, device=DEVICE))

    cm = make_model(cfg)
    cm.load_state_dict(copy.deepcopy(shared_init))
    cheater = CheatingMiner(uid=10, model=cm, dataset=dataset, storage=storage, hparams=hp, device=DEVICE)

    vm = make_model(cfg)
    vm.load_state_dict(copy.deepcopy(shared_init))
    val = Validator(uid=0, model=vm, dataset=dataset, storage=storage, hparams=hp, device=DEVICE)

    reset_vram()
    all_uids = [1, 2, 10]

    print(f"    {'Win':>4}  {'Train':>7}  {'Eval':>7}  {'H1':>7}  {'H2':>7}  {'Cheat':>7}")
    print("    " + "-" * 48)

    total_train = 0.0
    total_eval = 0.0

    for w in range(n_windows):
        sync()
        t0 = time.time()
        for m in miners:
            await m.train_window(w)
        await cheater.train_window(w)
        sync()
        t_train = time.time() - t0

        sync()
        t0 = time.time()
        results = await val.evaluate_window(w, all_uids)
        await val.apply_best_gradients(w, results)
        sync()
        t_eval = time.time() - t0

        total_train += t_train
        total_eval += t_eval

        val_state = copy.deepcopy(val.model.state_dict())
        for m in miners:
            m.model.load_state_dict(copy.deepcopy(val_state))
        cheater.model.load_state_dict(copy.deepcopy(val_state))

        scores = {r.uid: r.final_score for r in results}
        print(f"    {w:>4}  {t_train:>6.1f}s  {t_eval:>6.1f}s  {scores[1]:>7.4f}  {scores[2]:>7.4f}  {scores[10]:>7.4f}")

    vram_peak = vram_gb()
    h1_ema = val.get_effective_score(1)
    h2_ema = val.get_effective_score(2)
    c_ema = val.get_effective_score(10)

    print(f"    Total:  train={total_train:.1f}s  eval={total_eval:.1f}s  VRAM={vram_peak:.2f}GB")
    print(f"    EMA:    H1={h1_ema:.4f}  H2={h2_ema:.4f}  Cheater={c_ema:.4f}")

    ok = h1_ema > 0.5 and h2_ema > 0.5 and c_ema == 0.0
    print(f"    {'PASS' if ok else 'FAIL'}: honest > 0.5, cheater = 0")

    return {
        "train_total_s": total_train, "eval_total_s": total_eval,
        "vram_gb": vram_peak,
        "h1_ema": h1_ema, "h2_ema": h2_ema, "c_ema": c_ema, "ok": ok,
    }


# ──────────────────────────────────────────────────────────────────────────
# Phase C: Convergence
# ──────────────────────────────────────────────────────────────────────────

async def phase_c(tier: TierConfig, tmpdir: str) -> dict[str, Any]:
    n_windows = tier.n_phase_c_windows
    print(f"\n  --- Phase C: Convergence ({n_windows} windows) ---")
    cfg, hp = tier.cfg, tier.hp
    dataset = SyntheticDataset(size=tier.dataset_size, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, seed=42)
    eval_dataset = SyntheticDataset(size=256, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, seed=7777)
    storage = LocalFileStorage(tmpdir + "/c")
    shared_init = make_model(cfg).state_dict()

    miners = []
    for uid in [1, 2]:
        m = make_model(cfg)
        m.load_state_dict(copy.deepcopy(shared_init))
        miners.append(Miner(uid=uid, model=m, dataset=dataset, storage=storage, hparams=hp, device=DEVICE))

    vm = make_model(cfg)
    vm.load_state_dict(copy.deepcopy(shared_init))
    val = Validator(uid=0, model=vm, dataset=dataset, storage=storage, hparams=hp, device=DEVICE)

    initial_loss = eval_loss(val.model, eval_dataset)

    print(f"    Initial loss: {initial_loss:.4f}")
    print(f"    {'Win':>4}  {'Loss':>10}  {'Improve':>8}")
    print("    " + "-" * 28)

    for w in range(n_windows):
        for m in miners:
            await m.train_window(w)
        results = await val.evaluate_window(w, [m.uid for m in miners])
        await val.apply_best_gradients(w, results)

        val_state = val.model.state_dict()
        for m in miners:
            m.model.load_state_dict(copy.deepcopy(val_state))

        e_loss = eval_loss(val.model, eval_dataset)
        improve = (initial_loss - e_loss) / initial_loss * 100
        print(f"    {w:>4}  {e_loss:>10.4f}  {improve:>+7.2f}%")

    final_loss = eval_loss(val.model, eval_dataset)
    reduction = (initial_loss - final_loss) / initial_loss * 100

    ok = final_loss <= initial_loss
    print(f"    Loss: {initial_loss:.4f} -> {final_loss:.4f} ({reduction:+.2f}%)")
    print(f"    {'PASS' if ok else 'FAIL'}: {'loss decreased' if ok else 'loss increased (diverging)'}")

    return {"initial_loss": initial_loss, "final_loss": final_loss, "reduction_pct": reduction, "ok": ok}


# ──────────────────────────────────────────────────────────────────────────
# Run a tier
# ──────────────────────────────────────────────────────────────────────────

async def run_tier(tier_num: int) -> dict[str, Any]:
    tier = TIERS[tier_num]
    cfg = tier.cfg
    param_count = sum(p.numel() for p in make_model(cfg).parameters())

    print("=" * 65)
    print(f"TIER {tier_num}: {tier.name} ({param_count:,} params)")
    print(f"  Config: hidden={cfg.hidden_dim} inter={cfg.intermediate_dim} "
          f"layers={cfg.n_layers} heads={cfg.n_heads} seq={cfg.seq_len}")
    print(f"  HParams: max_batches={tier.hp.max_batches} micro_bs={tier.hp.micro_bs} "
          f"topk={tier.hp.topk} n_probes={tier.hp.n_probes}")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    t_total = time.time()

    with tempfile.TemporaryDirectory() as td:
        r_a = await phase_a(tier, td)
        reset_vram()
        r_b = await phase_b(tier, td)
        reset_vram()
        r_c = await phase_c(tier, td)

    t_total = time.time() - t_total

    all_ok = r_a["ok"] and r_b["ok"] and r_c["ok"]
    status = "PASS" if all_ok else "FAIL"

    print(f"\n  {'=' * 60}")
    print(f"  TIER {tier_num} SUMMARY: [{status}]  ({t_total:.1f}s total)")
    print(f"    Phase A: {'PASS' if r_a['ok'] else 'FAIL'}  train={r_a['train_s']:.1f}s  eval={r_a['eval_s']:.1f}s  score={r_a['score']:.4f}")
    print(f"    Phase B: {'PASS' if r_b['ok'] else 'FAIL'}  H={r_b['h1_ema']:.4f}/{r_b['h2_ema']:.4f}  C={r_b['c_ema']:.4f}")
    print(f"    Phase C: {'PASS' if r_c['ok'] else 'FAIL'}  loss {r_c['initial_loss']:.4f}->{r_c['final_loss']:.4f} ({r_c['reduction_pct']:+.2f}%)")
    print(f"    Peak VRAM: {max(r_a.get('vram_gb', 0), r_b.get('vram_gb', 0)):.2f} GB")
    print(f"  {'=' * 60}")

    return {
        "tier": tier_num, "params": param_count, "total_s": t_total,
        "phase_a": r_a, "phase_b": r_b, "phase_c": r_c, "ok": all_ok,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

async def main(tiers: list[int]) -> bool:
    all_results = []
    for t in tiers:
        if t not in TIERS:
            print(f"Unknown tier {t}. Available: {list(TIERS.keys())}")
            return False
        r = await run_tier(t)
        all_results.append(r)

    if len(all_results) > 1:
        print("\n" + "=" * 85)
        print("SCALE-UP SUMMARY")
        print("=" * 85)
        print(f"{'Tier':>5}  {'Params':>12}  {'Train':>7}  {'Eval':>7}  {'VRAM':>7}  "
              f"{'Honest':>7}  {'Cheat':>7}  {'Conv%':>7}  {'Status':>6}")
        print("-" * 85)
        for r in all_results:
            a, b, c = r["phase_a"], r["phase_b"], r["phase_c"]
            print(f"{r['tier']:>5}  {r['params']:>12,}  {a['train_s']:>6.1f}s  {a['eval_s']:>6.1f}s  "
                  f"{b.get('vram_gb', 0):>6.2f}G  {b['h1_ema']:>7.4f}  {b['c_ema']:>7.4f}  "
                  f"{c['reduction_pct']:>+6.1f}%  {'PASS' if r['ok'] else 'FAIL':>6}")
        print("=" * 85)

    return all(r["ok"] for r in all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model scale-up test")
    parser.add_argument("--tier", type=int, default=None, help="Run a single tier (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all tiers sequentially")
    args = parser.parse_args()

    setup_logging(level="WARNING")

    if args.all:
        tiers = list(TIERS.keys())
    elif args.tier is not None:
        tiers = [args.tier]
    else:
        tiers = [1]

    ok = asyncio.run(main(tiers))
    sys.exit(0 if ok else 1)
