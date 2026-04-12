#!/usr/bin/env python3
"""Local test harness: 2 honest miners + 3 cheating miners + 1 validator.

Pure PoW verification: no gradient quality measurement, pure accumulation
training, validator nonce for unpredictable probe indices.

Usage:
    source .venv/bin/activate
    python run_local.py
"""

from __future__ import annotations

import asyncio
import copy
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from teutonic.clock.local import ManualClock
from teutonic.compress import TopKCompressor, compress_model_gradients
from teutonic.dataset.synthetic import SyntheticDataset
from teutonic.hparams import HParams
from teutonic.model import LlamaConfig, TinyLlama
from teutonic.probe_spec import ProbeSpec
from teutonic.sampler import MinerSampler
from teutonic.storage.local import LocalFileStorage
from teutonic.submission import MinerSubmission

from neurons.miner import Miner
from neurons.trainer import train_window
from neurons.validator import Validator


# ──────────────────────────────────────────────────────────────────────────
# Cheating miner variants
# ──────────────────────────────────────────────────────────────────────────

class CheatingMiner:
    """A miner that cheats in a configurable way."""

    def __init__(
        self,
        uid: int,
        model: nn.Module,
        dataset: SyntheticDataset,
        storage: LocalFileStorage,
        cheat_mode: str,
        hparams: HParams,
        *,
        device: str = "cpu",
    ):
        self.uid = uid
        self.model = model
        self.dataset = dataset
        self.storage = storage
        self.cheat_mode = cheat_mode
        self.hp = hparams
        self.compressor = TopKCompressor(topk=hparams.topk)
        self.device = device

    async def train_window(self, window: int) -> MinerSubmission:
        sampler = MinerSampler(
            self.dataset, self.uid, window,
            n_batches=self.hp.n_batches, micro_bs=self.hp.micro_bs,
        )

        all_indices = set(range(sampler.total_micro_batches))
        probe_spec = ProbeSpec(
            param_name=self.hp.probe_param_name,
            slice_start=self.hp.probe_slice_start,
            slice_end=self.hp.probe_slice_end,
            batch_indices=tuple(sorted(all_indices)),
        )

        train_ds = self.dataset
        if self.cheat_mode == "wrong_data":
            train_ds = SyntheticDataset(
                size=len(self.dataset), seq_len=64, vocab_size=512, seed=9999
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr)
        result = train_window(
            self.model, train_ds, sampler, optimizer,
            probe_indices=all_indices, probe_spec=probe_spec, device=self.device,
        )

        compressed = compress_model_gradients(self.model, self.compressor)

        loss_ledger = result["loss_ledger"]
        grad_probes = result["grad_probes"]

        if self.cheat_mode == "random_losses":
            loss_ledger = [random.uniform(5.0, 15.0) for _ in loss_ledger]

        if self.cheat_mode == "fake_gradients":
            grad_probes = {k: torch.randn_like(v) for k, v in grad_probes.items()}

        submission = MinerSubmission(
            uid=self.uid, window=window,
            compressed_gradients=compressed,
            loss_ledger=loss_ledger,
            grad_probes=grad_probes,
        )
        await self.storage.put(submission.storage_key(), submission.to_dict())
        return submission


# ──────────────────────────────────────────────────────────────────────────
# Harness
# ──────────────────────────────────────────────────────────────────────────

HP = HParams(n_batches=6, micro_bs=2, topk=32, lr=1e-3, outer_lr=0.4)


def make_model(cfg: LlamaConfig, seed: int = 0) -> TinyLlama:
    torch.manual_seed(seed)
    return TinyLlama(cfg)


async def run_local(n_windows: int = 3) -> None:
    cfg = LlamaConfig(
        vocab_size=512, hidden_dim=64, intermediate_dim=128,
        n_layers=2, n_heads=2, seq_len=64,
    )
    dataset = SyntheticDataset(size=2048, seq_len=64, vocab_size=512, seed=42)
    clock = ManualClock()

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalFileStorage(tmpdir)
        shared_init = make_model(cfg, seed=0).state_dict()

        honest_miners = []
        for i in range(2):
            m = make_model(cfg)
            m.load_state_dict(copy.deepcopy(shared_init))
            honest_miners.append(
                Miner(uid=i + 1, model=m, dataset=dataset,
                      storage=storage, hparams=HP, device="cpu")
            )

        cheat_modes = ["random_losses", "fake_gradients", "wrong_data"]
        cheaters = []
        for i, mode in enumerate(cheat_modes):
            m = make_model(cfg)
            m.load_state_dict(copy.deepcopy(shared_init))
            cheaters.append(
                CheatingMiner(uid=10 + i, model=m, dataset=dataset,
                              storage=storage, cheat_mode=mode,
                              hparams=HP, device="cpu")
            )

        val_model = make_model(cfg)
        val_model.load_state_dict(copy.deepcopy(shared_init))
        validator = Validator(
            uid=0, model=val_model, dataset=dataset, storage=storage,
            hparams=HP, clock=clock, device="cpu",
        )

        all_uids = [m.uid for m in honest_miners] + [c.uid for c in cheaters]

        print("=" * 72)
        print("Teutonic Local Test Harness (Pure PoW)")
        print(f"Model: {sum(p.numel() for p in val_model.parameters()):,} params")
        print(f"Dataset: {len(dataset)} sequences of length {cfg.seq_len}")
        print(f"Batches/window: {HP.n_batches} x micro_bs={HP.micro_bs}")
        print(f"Honest miners: {[m.uid for m in honest_miners]}")
        print(f"Cheating miners: {[(c.uid, c.cheat_mode) for c in cheaters]}")
        print(f"Windows: {n_windows}")
        print("=" * 72)

        for w in range(n_windows):
            clock.set_window(w)
            print(f"\n--- Window {w} ---")

            for miner in honest_miners:
                await miner.train_window(w)
            for cheater in cheaters:
                await cheater.train_window(w)

            results = await validator.evaluate_window(w, all_uids)

            print(f"{'UID':>5}  {'Type':>16}  {'Loss':>6}  {'Probe':>6}  "
                  f"{'Slash':>6}  {'Final':>7}  {'EMA':>6}  Detail")
            print("-" * 95)
            for r in results:
                if r.uid < 10:
                    mtype = "honest"
                else:
                    idx = r.uid - 10
                    mtype = cheat_modes[idx] if idx < len(cheat_modes) else "?"
                ema = validator.get_effective_score(r.uid)
                print(
                    f"{r.uid:>5}  {mtype:>16}  "
                    f"{r.loss_score:>6.2f}  {r.probe_score:>6.3f}  "
                    f"{r.slash_fraction:>6.2f}  {r.final_score:>7.4f}  "
                    f"{ema:>6.4f}  {r.reason}"
                )

            await validator.apply_best_gradients(w, results)

        # Test state persistence round-trip
        await validator.save_state()
        val_model2 = make_model(cfg)
        validator2 = Validator(
            uid=0, model=val_model2, dataset=dataset, storage=storage,
            hparams=HP, device="cpu",
        )
        loaded = await validator2.load_state()

        print("\n" + "=" * 72)
        print("FINAL SCORES (EMA)")
        print("=" * 72)
        for uid in sorted(validator.scores):
            label = "HONEST" if uid < 10 else f"CHEATER ({cheat_modes[uid - 10]})"
            print(f"  UID {uid:>3}: {validator.scores[uid]:.4f}  [{label}]")

        honest_scores = [validator.scores.get(m.uid, 0) for m in honest_miners]
        cheater_scores = [validator.scores.get(c.uid, 0) for c in cheaters]
        avg_honest = sum(honest_scores) / len(honest_scores) if honest_scores else 0
        avg_cheater = sum(cheater_scores) / len(cheater_scores) if cheater_scores else 0

        print(f"\n  Avg honest:  {avg_honest:.4f}")
        print(f"  Avg cheater: {avg_cheater:.4f}")

        # Test miner discovery
        discovered = await validator.discover_miners(0)
        print(f"\n  Miner discovery for window 0: {discovered}")

        # State persistence check
        print(f"  State persistence: {'OK' if loaded else 'FAILED'}")
        if loaded:
            print(f"  Restored scores match: {validator2.scores == validator.scores}")

        if avg_honest > avg_cheater:
            print("\n  PASS: Honest miners scored higher than cheaters")
        else:
            print("\n  WARNING: Cheaters scored >= honest miners")


if __name__ == "__main__":
    asyncio.run(run_local())
