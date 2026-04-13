#!/usr/bin/env python3
"""Statistical scoring test suite.

Runs parameterized scenarios to build a quantitative understanding of
volume-weighted scoring under diverse conditions: varying throughput,
cheating strategies, EMA dynamics, and gradient quality.

Usage:
    source .venv/bin/activate
    python tests/test_scoring.py
    python tests/test_scoring.py --scenario 3
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEVICE = os.environ.get("TEUTONIC_DEVICE", "cpu")

import numpy as np
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
from teutonic.sampler import MinerSampler
from teutonic.storage.local import LocalFileStorage
from teutonic.submission import MinerSubmission

from neurons.miner import Miner
from neurons.trainer import train_window
from neurons.validator import Validator


# ──────────────────────────────────────────────────────────────────────────
# Infrastructure
# ──────────────────────────────────────────────────────────────────────────

CFG = LlamaConfig(
    vocab_size=512, hidden_dim=64, intermediate_dim=128,
    n_layers=2, n_heads=2, seq_len=64,
)

BASE_HP = HParams(
    max_batches=8, min_batches=2, micro_bs=2, topk=32,
    lr=1e-3, outer_lr=0.4,
)


@dataclass
class Env:
    storage: LocalFileStorage
    dataset: SyntheticDataset
    shared_init: dict
    hp: HParams


def make_model(seed: int = 0) -> TinyLlama:
    torch.manual_seed(seed)
    return TinyLlama(CFG)


def make_env(tmpdir: str, hp: HParams = BASE_HP) -> Env:
    dataset = SyntheticDataset(size=2048, seq_len=64, vocab_size=512, seed=42)
    shared_init = make_model(seed=0).state_dict()
    storage = LocalFileStorage(tmpdir)
    return Env(storage=storage, dataset=dataset, shared_init=shared_init, hp=hp)


def make_miner(env: Env, uid: int, throttle: int | None = None) -> "_ThrottledMiner | Miner":
    """Create a miner. If *throttle* is set, it trains only that many batches
    while keeping the full sampler so data indices match the validator."""
    m = make_model()
    m.load_state_dict(copy.deepcopy(env.shared_init))
    if throttle is not None and throttle < env.hp.max_batches:
        return _ThrottledMiner(uid=uid, model=m, env=env, n_batches=throttle)
    return Miner(uid=uid, model=m, dataset=env.dataset, storage=env.storage, hparams=env.hp, device=DEVICE)


def make_cheater(env: Env, uid: int, cheat_mode: str, throttle: int | None = None):
    m = make_model()
    m.load_state_dict(copy.deepcopy(env.shared_init))
    return _CheatingMiner(uid=uid, model=m, dataset=env.dataset, storage=env.storage,
                          cheat_mode=cheat_mode, hparams=env.hp, device=DEVICE)


def make_validator(env: Env, uid: int = 0) -> Validator:
    m = make_model()
    m.load_state_dict(copy.deepcopy(env.shared_init))
    return Validator(uid=uid, model=m, dataset=env.dataset, storage=env.storage,
                     hparams=env.hp, device=DEVICE)


def eval_loss(model: nn.Module, dataset, n_samples: int = 64) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            tokens = dataset[i].unsqueeze(0)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            total += loss.item()
    model.train()
    return total / min(n_samples, len(dataset))


class _ThrottledMiner:
    """Honest miner that trains only *n_batches* while keeping the full sampler.

    The sampler is constructed with the network's ``max_batches`` so that
    data indices match the validator's replay.  After construction,
    ``sampler.max_batches`` is lowered so the trainer loop stops early.
    """

    def __init__(self, uid: int, model: nn.Module, env: Env, n_batches: int):
        self.uid = uid
        self.model = model
        self.env = env
        self.n_batches = n_batches
        self.compressor = TopKCompressor(topk=env.hp.topk)

    async def train_window(self, window: int) -> MinerSubmission:
        sampler = MinerSampler(
            self.env.dataset, self.uid, window,
            max_batches=self.env.hp.max_batches, micro_bs=self.env.hp.micro_bs,
        )
        sampler.max_batches = self.n_batches

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.env.hp.lr)
        result = train_window(
            self.model, self.env.dataset, sampler, optimizer,
            device=DEVICE, probe_slice_size=self.env.hp.probe_slice_size,
        )
        compressed = compress_model_gradients(self.model, self.compressor)
        submission = MinerSubmission(
            uid=self.uid, window=window, compressed_gradients=compressed,
            loss_ledger=result["loss_ledger"], n_batches_trained=result["n_batches_trained"],
            grad_probes=result["grad_probes"],
        )
        await self.env.storage.put(submission.storage_key(), submission.to_dict())
        return submission


class _CheatingMiner:
    def __init__(self, uid, model, dataset, storage, cheat_mode, hparams, device=DEVICE):
        self.uid = uid
        self.model = model
        self.dataset = dataset
        self.storage = storage
        self.cheat_mode = cheat_mode
        self.hp = hparams
        self.compressor = TopKCompressor(topk=hparams.topk)
        self.device = device

    async def train_window(self, window: int) -> MinerSubmission:
        sampler = MinerSampler(self.dataset, self.uid, window,
                               max_batches=self.hp.max_batches, micro_bs=self.hp.micro_bs)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr)
        result = train_window(self.model, self.dataset, sampler, optimizer,
                              device=self.device, probe_slice_size=self.hp.probe_slice_size)
        compressed = compress_model_gradients(self.model, self.compressor)

        loss_ledger = result["loss_ledger"]
        grad_probes = result["grad_probes"]
        n_batches_trained = result["n_batches_trained"]

        if self.cheat_mode == "fake_compressed":
            compressed = {
                pname: {"idxs": comp["idxs"], "vals": torch.randn_like(comp["vals"]), "shape": comp["shape"]}
                for pname, comp in compressed.items()
            }

        submission = MinerSubmission(
            uid=self.uid, window=window, compressed_gradients=compressed,
            loss_ledger=loss_ledger, n_batches_trained=n_batches_trained, grad_probes=grad_probes,
        )
        await self.storage.put(submission.storage_key(), submission.to_dict())
        return submission


# ──────────────────────────────────────────────────────────────────────────
# Scenario 1: Volume Curve
# ──────────────────────────────────────────────────────────────────────────

async def scenario_1_volume_curve() -> tuple[bool, str]:
    """Sweep n_batches from 1 to max_batches, one miner each, single window."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Volume Curve")
    print("=" * 70)

    max_b = BASE_HP.max_batches
    min_b = BASE_HP.min_batches

    print(f"\n  max_batches={max_b}  min_batches={min_b}")
    print(f"\n  {'Batches':>8}  {'Volume':>8}  {'Final':>8}  {'Expected':>10}  {'Match':>6}")
    print("  " + "-" * 48)

    errors = []
    with tempfile.TemporaryDirectory() as td:
        for n in range(1, max_b + 1):
            env = make_env(td)
            miner = make_miner(env, uid=1, throttle=n)
            val = make_validator(env)
            await miner.train_window(0)
            results = await val.evaluate_window(0, [1])
            r = results[0]

            volume = n / max_b
            if n < min_b:
                expected = 0.0
            else:
                expected = volume
            actual = r.final_score

            match = abs(actual - expected) < 0.05
            mark = "OK" if match else "FAIL"
            if not match:
                errors.append(f"n={n}: expected~{expected:.3f}, got {actual:.4f}")

            print(f"  {n:>8}  {volume:>8.3f}  {actual:>8.4f}  {expected:>10.3f}  {mark:>6}")

    ok = len(errors) == 0
    msg = "Volume curve is linear with floor" if ok else f"{len(errors)} mismatches: {errors[0]}"
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] {msg}")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────────
# Scenario 2: Floor Boundary
# ──────────────────────────────────────────────────────────────────────────

async def scenario_2_floor_boundary() -> tuple[bool, str]:
    """Test min_batches - 1, min_batches, min_batches + 1."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Floor Boundary")
    print("=" * 70)

    min_b = BASE_HP.min_batches
    test_points = [min_b - 1, min_b, min_b + 1]

    print(f"\n  min_batches={min_b}")
    print(f"\n  {'Batches':>8}  {'Final':>8}  {'Expect':>8}")
    print("  " + "-" * 30)

    errors = []
    with tempfile.TemporaryDirectory() as td:
        for n in test_points:
            if n < 1:
                continue
            env = make_env(td)
            miner = make_miner(env, uid=1, throttle=n)
            val = make_validator(env)
            await miner.train_window(0)
            results = await val.evaluate_window(0, [1])
            score = results[0].final_score

            if n < min_b:
                expected_label = "= 0"
                ok = score == 0.0
            else:
                expected_label = "> 0"
                ok = score > 0.0
            if not ok:
                errors.append(f"n={n}: score={score:.4f}, expected {expected_label}")

            print(f"  {n:>8}  {score:>8.4f}  {expected_label:>8}")

    ok = len(errors) == 0
    msg = "Floor boundary is sharp" if ok else errors[0]
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] {msg}")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────────
# Scenario 3: Mixed Population
# ──────────────────────────────────────────────────────────────────────────

async def scenario_3_mixed_population() -> tuple[bool, str]:
    """5 miners at different throughputs over 20 windows. Track EMA scores."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Mixed Population (20 windows)")
    print("=" * 70)

    max_b = BASE_HP.max_batches
    min_b = BASE_HP.min_batches
    fractions = [1.0, 0.75, 0.50, 0.25]
    batch_counts = [max(1, int(max_b * f)) for f in fractions]
    batch_counts.append(min_b)
    labels = ["100%", "75%", "50%", "25%", f"floor({min_b})"]
    n_windows = 10

    print(f"\n  Miners: {list(zip(labels, batch_counts))}")
    header = f"  {'Window':>6}"
    for label in labels:
        header += f"  {label:>9}"
    print(f"\n{header}")
    print("  " + "-" * (8 + 11 * len(labels)))

    with tempfile.TemporaryDirectory() as td:
        env = make_env(td)
        val = make_validator(env)
        miners = []
        for i, nb in enumerate(batch_counts):
            miners.append(make_miner(env, uid=i + 1, throttle=nb))

        ema_history: dict[int, list[float]] = {m.uid: [] for m in miners}

        for w in range(n_windows):
            for m in miners:
                await m.train_window(w)
            uids = [m.uid for m in miners]
            results = await val.evaluate_window(w, uids)
            await val.apply_best_gradients(w, results)

            val_state = copy.deepcopy(val.model.state_dict())
            for m in miners:
                m.model.load_state_dict(copy.deepcopy(val_state))

            row = f"  {w:>6}"
            for m in miners:
                ema = val.get_effective_score(m.uid)
                ema_history[m.uid].append(ema)
                row += f"  {ema:>9.4f}"
            print(row)

    print(f"\n  Final EMA scores:")
    final_emas = []
    for i, m in enumerate(miners):
        ema = ema_history[m.uid][-1]
        final_emas.append(ema)
        print(f"    {labels[i]:>10}  ({batch_counts[i]:>2} batches) -> EMA = {ema:.4f}")

    errors = []
    for i in range(len(final_emas) - 1):
        if final_emas[i] < final_emas[i + 1] - 1e-6:
            errors.append(f"{labels[i]} ({final_emas[i]:.4f}) < {labels[i+1]} ({final_emas[i+1]:.4f})")

    ok = len(errors) == 0
    ratio = final_emas[0] / final_emas[-1] if final_emas[-1] > 0 else float("inf")
    msg = f"Scores monotonically decrease with throughput. Top/bottom ratio = {ratio:.2f}x" if ok else errors[0]
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] {msg}")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────────
# Scenario 4: Volume vs Cheating Trade-off
# ──────────────────────────────────────────────────────────────────────────

async def scenario_4_cheat_vs_volume() -> tuple[bool, str]:
    """Full-volume cheater (fake_compressed) vs half-volume honest miner."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: Full-Volume Cheater vs Half-Volume Honest")
    print("=" * 70)

    max_b = BASE_HP.max_batches
    half_b = max_b // 2
    n_windows = 10

    print(f"\n  Honest miner: {half_b} batches (50% volume)")
    print(f"  Cheater:      {max_b} batches (100% volume, fake_compressed)")

    with tempfile.TemporaryDirectory() as td:
        env = make_env(td)
        val = make_validator(env)
        honest = make_miner(env, uid=1, throttle=half_b)
        cheater = make_cheater(env, uid=2, cheat_mode="fake_compressed")

        print(f"\n  {'Window':>6}  {'Honest':>8}  {'Cheater':>8}  {'Winner':>8}")
        print("  " + "-" * 38)

        for w in range(n_windows):
            await honest.train_window(w)
            await cheater.train_window(w)
            results = await val.evaluate_window(w, [1, 2])

            val_state = copy.deepcopy(val.model.state_dict())
            honest.model.load_state_dict(copy.deepcopy(val_state))
            cheater.model.load_state_dict(copy.deepcopy(val_state))

            h_ema = val.get_effective_score(1)
            c_ema = val.get_effective_score(2)
            winner = "Honest" if h_ema > c_ema else ("Tie" if h_ema == c_ema else "CHEATER")
            print(f"  {w:>6}  {h_ema:>8.4f}  {c_ema:>8.4f}  {winner:>8}")

    h_final = val.get_effective_score(1)
    c_final = val.get_effective_score(2)
    ok = h_final > c_final and c_final == 0.0
    msg = f"Honest({h_final:.4f}) beats Cheater({c_final:.4f})" if ok else f"Honest={h_final:.4f}, Cheater={c_final:.4f}"
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] {msg}")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────────
# Scenario 5: EMA Recovery
# ──────────────────────────────────────────────────────────────────────────

async def scenario_5_ema_recovery() -> tuple[bool, str]:
    """Honest miner drops below floor for 1 window, then resumes. Track recovery."""
    print("\n" + "=" * 70)
    print("SCENARIO 5: EMA Recovery After Below-Floor Window")
    print("=" * 70)

    max_b = BASE_HP.max_batches
    min_b = BASE_HP.min_batches
    drop_window = 3
    n_windows = 10

    print(f"\n  Normal: {max_b} batches | Drop window {drop_window}: {min_b - 1} batches")

    with tempfile.TemporaryDirectory() as td:
        env = make_env(td)
        val = make_validator(env)

        print(f"\n  {'Window':>6}  {'Batches':>8}  {'Raw':>8}  {'EMA':>8}  {'Note':>12}")
        print("  " + "-" * 50)

        pre_drop_ema = 0.0

        for w in range(n_windows):
            is_drop = (w == drop_window)
            nb = (min_b - 1) if is_drop else max_b
            miner = make_miner(env, uid=1, throttle=nb)

            if w > 0:
                miner.model.load_state_dict(copy.deepcopy(val.model.state_dict()))

            await miner.train_window(w)
            results = await val.evaluate_window(w, [1])
            await val.apply_best_gradients(w, results)

            raw = results[0].final_score
            ema = val.get_effective_score(1)

            if w == drop_window - 1:
                pre_drop_ema = ema

            note = ""
            if is_drop:
                note = "<- DROP"
            elif w == drop_window + 1:
                note = "<- resume"

            print(f"  {w:>6}  {nb:>8}  {raw:>8.4f}  {ema:>8.4f}  {note:>12}")

    final_ema = val.get_effective_score(1)
    recovery_pct = (final_ema / pre_drop_ema * 100) if pre_drop_ema > 0 else 0

    recovered_at = None
    for w in range(drop_window + 1, n_windows):
        if val.scores.get(1, 0) >= pre_drop_ema * 0.95:
            recovered_at = w
            break

    ok = final_ema > pre_drop_ema * 0.90
    msg = f"Pre-drop EMA={pre_drop_ema:.4f}, Final={final_ema:.4f} ({recovery_pct:.0f}% recovered)"
    if recovered_at is not None:
        msg += f", recovered at window {recovered_at}"
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] {msg}")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────────
# Scenario 6: Gradient Quality vs Volume
# ──────────────────────────────────────────────────────────────────────────

async def scenario_6_gradient_quality() -> tuple[bool, str]:
    """Compare convergence: full-throughput miners vs half-throughput miners."""
    print("\n" + "=" * 70)
    print("SCENARIO 6: Gradient Quality vs Volume (Convergence)")
    print("=" * 70)

    max_b = BASE_HP.max_batches
    half_b = max_b // 2
    n_miners = 2
    n_windows = 8

    print(f"\n  Group A: {n_miners} miners x {max_b} batches (full)")
    print(f"  Group B: {n_miners} miners x {half_b} batches (half)")

    results_map: dict[str, list[float]] = {"full": [], "half": []}

    for group_name, nb in [("full", max_b), ("half", half_b)]:
        with tempfile.TemporaryDirectory() as td:
            env = make_env(td)
            val = make_validator(env)
            miners_list = [make_miner(env, uid=i + 1, throttle=nb) for i in range(n_miners)]

            initial_loss = eval_loss(val.model, env.dataset)
            results_map[group_name].append(initial_loss)

            for w in range(n_windows):
                for m in miners_list:
                    await m.train_window(w)
                uids = [m.uid for m in miners_list]
                res = await val.evaluate_window(w, uids)
                await val.apply_best_gradients(w, res)

                val_state = copy.deepcopy(val.model.state_dict())
                for m in miners_list:
                    m.model.load_state_dict(copy.deepcopy(val_state))

                e_loss = eval_loss(val.model, env.dataset)
                results_map[group_name].append(e_loss)

    print(f"\n  {'Window':>6}  {'Full Loss':>10}  {'Half Loss':>10}  {'Delta':>8}")
    print("  " + "-" * 40)
    for w in range(n_windows + 1):
        fl = results_map["full"][w]
        hl = results_map["half"][w]
        delta = hl - fl
        label = "init" if w == 0 else str(w - 1)
        print(f"  {label:>6}  {fl:>10.4f}  {hl:>10.4f}  {delta:>+8.4f}")

    full_reduction = (results_map["full"][0] - results_map["full"][-1]) / results_map["full"][0] * 100
    half_reduction = (results_map["half"][0] - results_map["half"][-1]) / results_map["half"][0] * 100

    print(f"\n  Full-throughput loss reduction: {full_reduction:.1f}%")
    print(f"  Half-throughput loss reduction: {half_reduction:.1f}%")

    ok = full_reduction >= half_reduction * 0.8
    msg = f"Full={full_reduction:.1f}% vs Half={half_reduction:.1f}%"
    if full_reduction > half_reduction:
        msg += " (full converges faster)"
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] {msg}")
    return ok, msg


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

ALL_SCENARIOS = [
    (1, "Volume Curve", scenario_1_volume_curve),
    (2, "Floor Boundary", scenario_2_floor_boundary),
    (3, "Mixed Population", scenario_3_mixed_population),
    (4, "Cheat vs Volume", scenario_4_cheat_vs_volume),
    (5, "EMA Recovery", scenario_5_ema_recovery),
    (6, "Gradient Quality", scenario_6_gradient_quality),
]


async def main(scenario_filter: int | None = None) -> bool:
    scenarios = ALL_SCENARIOS
    if scenario_filter is not None:
        scenarios = [(n, name, fn) for n, name, fn in ALL_SCENARIOS if n == scenario_filter]
        if not scenarios:
            print(f"Unknown scenario {scenario_filter}. Available: {[n for n, _, _ in ALL_SCENARIOS]}")
            return False

    print("=" * 70)
    print(f"Scoring Statistical Test Suite ({len(scenarios)} scenarios)")
    print("=" * 70)

    passed = 0
    failed = 0
    results = []

    for num, name, fn in scenarios:
        try:
            ok, msg = await fn()
        except Exception as e:
            ok = False
            msg = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

        results.append((num, name, ok, msg))
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for num, name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Scenario {num}: {name} -- {msg}")
    print(f"\n  {passed} passed, {failed} failed out of {len(results)}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring statistical test suite")
    parser.add_argument("--scenario", type=int, default=None, help="Run a single scenario by number")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    ok = asyncio.run(main(scenario_filter=args.scenario))
    sys.exit(0 if ok else 1)
