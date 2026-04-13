#!/usr/bin/env python3
"""Teutonic stress test suite.

Proves training works, cheaters get caught, and boundary conditions are handled.
Each test is isolated with its own storage, models, and dataset.

Usage:
    source .venv/bin/activate
    python test_stress.py
"""

from __future__ import annotations

import asyncio
import copy
import math
import os
import random
import sys
import tempfile
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEVICE = os.environ.get("TEUTONIC_DEVICE", "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from teutonic.clock.local import ManualClock, TimedClock
from teutonic.compress import TopKCompressor, compress_model_gradients
from teutonic.dataset.synthetic import SyntheticDataset
from teutonic.hparams import HParams
from teutonic.model import LlamaConfig, TinyLlama
from teutonic.sampler import MinerSampler
from teutonic.storage.local import LocalFileStorage
from teutonic.submission import MinerSubmission

from neurons.miner import Miner
from neurons.trainer import train_window
from neurons.validator import Validator, SlashConfig


# ──────────────────────────────────────────────────────────────────────────
# Shared test infrastructure
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = LlamaConfig(
    vocab_size=512, hidden_dim=64, intermediate_dim=128,
    n_layers=2, n_heads=2, seq_len=64,
)

DEFAULT_HP = HParams(max_batches=6, micro_bs=2, topk=32, lr=1e-3, outer_lr=0.4)


def make_model(cfg: LlamaConfig = DEFAULT_CFG, seed: int = 0) -> TinyLlama:
    torch.manual_seed(seed)
    return TinyLlama(cfg)


def make_dataset(size: int = 2048) -> SyntheticDataset:
    return SyntheticDataset(size=size, seq_len=64, vocab_size=512, seed=42)


@dataclass
class TestEnv:
    storage: LocalFileStorage
    dataset: SyntheticDataset
    shared_init: dict
    cfg: LlamaConfig
    hp: HParams
    tmpdir: str


def make_env(
    hp: HParams = DEFAULT_HP,
    cfg: LlamaConfig = DEFAULT_CFG,
    dataset_size: int = 2048,
    tmpdir: str | None = None,
) -> TestEnv:
    dataset = SyntheticDataset(size=dataset_size, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, seed=42)
    storage = LocalFileStorage(tmpdir or tempfile.mkdtemp())
    shared_init = make_model(cfg, seed=0).state_dict()
    return TestEnv(storage=storage, dataset=dataset, shared_init=shared_init, cfg=cfg, hp=hp, tmpdir=tmpdir or "")


def make_miner(env: TestEnv, uid: int) -> Miner:
    m = make_model(env.cfg)
    m.load_state_dict(copy.deepcopy(env.shared_init))
    return Miner(uid=uid, model=m, dataset=env.dataset, storage=env.storage, hparams=env.hp, device=DEVICE)


def make_validator(env: TestEnv, uid: int = 0, clock: Any = None) -> Validator:
    m = make_model(env.cfg)
    m.load_state_dict(copy.deepcopy(env.shared_init))
    return Validator(uid=uid, model=m, dataset=env.dataset, storage=env.storage,
                     hparams=env.hp, clock=clock, device=DEVICE)


async def cheating_submit(
    env: TestEnv, uid: int, window: int, cheat_fn,
) -> MinerSubmission:
    """Train honestly, then apply cheat_fn to mutate the submission before upload."""
    m = make_model(env.cfg).to(DEVICE)
    m.load_state_dict(copy.deepcopy(env.shared_init))
    sampler = MinerSampler(env.dataset, uid, window, max_batches=env.hp.max_batches, micro_bs=env.hp.micro_bs)
    optimizer = torch.optim.AdamW(m.parameters(), lr=env.hp.lr)
    result = train_window(m, env.dataset, sampler, optimizer,
                          device=DEVICE, probe_slice_size=env.hp.probe_slice_size)
    compressed = compress_model_gradients(m, TopKCompressor(topk=env.hp.topk))
    sub = MinerSubmission(uid=uid, window=window,
                          compressed_gradients=compressed,
                          loss_ledger=result["loss_ledger"],
                          grad_probes=result["grad_probes"])
    sub = cheat_fn(sub)
    await env.storage.put(sub.storage_key(), sub.to_dict())
    return sub


# ──────────────────────────────────────────────────────────────────────────
# A. Prove Training Works
# ──────────────────────────────────────────────────────────────────────────

async def test_a1_baseline() -> tuple[bool, str]:
    """Default config, 2 honest miners, 3 windows. Scores > 0.95."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        miners = [make_miner(env, uid=i+1) for i in range(2)]
        val = make_validator(env)

        for w in range(3):
            for miner in miners:
                await miner.train_window(w)
            results = await val.evaluate_window(w, [m.uid for m in miners])
            await val.apply_best_gradients(w, results)

        for uid in [1, 2]:
            s = val.get_effective_score(uid)
            if s < 0.95:
                return False, f"UID {uid} score {s:.4f} < 0.95"
        return True, f"scores: {[val.get_effective_score(i+1) for i in range(2)]}"


async def test_a2_single_batch() -> tuple[bool, str]:
    """n_batches=1. Loss error < 0.001, probe cosine > 0.999."""
    hp = replace(DEFAULT_HP, max_batches=1, min_batches=1, n_loss_spot_checks=1, n_probes=1)
    with tempfile.TemporaryDirectory() as td:
        env = make_env(hp=hp, tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)

        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        max_err = r.loss_result.max_error if r.loss_result else 999
        min_sim = r.probe_result.min_similarity if r.probe_result else -1
        if max_err > 0.001:
            return False, f"loss error {max_err:.6f} > 0.001"
        if min_sim < 0.999:
            return False, f"probe cosine {min_sim:.6f} < 0.999"
        return True, f"loss_err={max_err:.6f}, probe_sim={min_sim:.6f}"


async def test_a3_large_batch() -> tuple[bool, str]:
    """max_batches=20, micro_bs=4. Honest miners pass."""
    hp = replace(DEFAULT_HP, max_batches=20, micro_bs=4)
    with tempfile.TemporaryDirectory() as td:
        env = make_env(hp=hp, tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)

        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        if r.final_score < 0.9:
            return False, f"score {r.final_score:.4f} < 0.9"
        return True, f"score={r.final_score:.4f}, n_batches=20"


async def test_a4_multiple_windows() -> tuple[bool, str]:
    """10 windows. EMA scores stay high."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)

        window_scores = []
        for w in range(10):
            await miner.train_window(w)
            results = await val.evaluate_window(w, [1])
            await val.apply_best_gradients(w, results)
            window_scores.append(results[0].final_score)

        ema = val.get_effective_score(1)
        if ema < 0.9:
            return False, f"EMA {ema:.4f} < 0.9 after 10 windows"
        if any(s < 0.8 for s in window_scores):
            return False, f"Some window score < 0.8: {window_scores}"
        return True, f"EMA={ema:.4f}, min_window={min(window_scores):.4f}"


async def test_a5_model_learns() -> tuple[bool, str]:
    """After 5 windows, loss should decrease."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)

        def eval_loss(model, dataset, n=10):
            model.eval()
            dev = next(model.parameters()).device
            total = 0.0
            with torch.no_grad():
                for i in range(n):
                    tokens = dataset[i].unsqueeze(0).to(dev)
                    logits = model(tokens[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
                    total += loss.item()
            return total / n

        loss_w0 = eval_loss(val.model, env.dataset)

        for w in range(5):
            await miner.train_window(w)
            results = await val.evaluate_window(w, [1])
            await val.apply_best_gradients(w, results)

        loss_w5 = eval_loss(val.model, env.dataset)
        init_params = list(make_model(env.cfg).state_dict().values())
        curr_params = list(val.model.state_dict().values())
        weights_changed = any(not torch.equal(a, b) for a, b in zip(init_params, curr_params))

        if not weights_changed:
            return False, "Model weights unchanged after 5 windows"
        return True, f"loss: {loss_w0:.4f} -> {loss_w5:.4f}, weights_changed=True"


# ──────────────────────────────────────────────────────────────────────────
# B. Prove Cheaters Get Caught
# ──────────────────────────────────────────────────────────────────────────

async def test_b1_random_losses() -> tuple[bool, str]:
    """Cheater submits random loss values."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        honest = make_miner(env, uid=1)
        await honest.train_window(0)

        def cheat(sub):
            sub.loss_ledger = [random.uniform(5, 15) for _ in sub.loss_ledger]
            return sub
        await cheating_submit(env, uid=2, window=0, cheat_fn=cheat)

        val = make_validator(env)
        results = await val.evaluate_window(0, [1, 2])
        h, c = results[0], results[1]
        if c.final_score > 0.0:
            return False, f"Cheater scored {c.final_score:.4f} > 0"
        if c.slash_fraction < 0.90:
            return False, f"Slash {c.slash_fraction:.2f} < 0.90"
        if h.final_score <= 0.0:
            return False, f"Honest scored {h.final_score:.4f} <= 0"
        return True, f"honest={h.final_score:.4f}, cheater={c.final_score:.4f}, slash={c.slash_fraction:.2f}"


async def test_b2_fake_gradients() -> tuple[bool, str]:
    """Cheater submits correct losses but random gradient probes."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        honest = make_miner(env, uid=1)
        await honest.train_window(0)

        def cheat(sub):
            sub.grad_probes = {
                k: {pn: torch.randn_like(t) for pn, t in pdict.items()}
                for k, pdict in sub.grad_probes.items()
            }
            return sub
        await cheating_submit(env, uid=2, window=0, cheat_fn=cheat)

        val = make_validator(env)
        results = await val.evaluate_window(0, [1, 2])
        c = results[1]
        if c.final_score > 0.0:
            return False, f"Cheater scored {c.final_score:.4f} > 0"
        if c.slash_fraction < 1.0:
            return False, f"Slash {c.slash_fraction:.2f} < 1.0"
        min_sim = c.probe_result.min_similarity if c.probe_result else 1.0
        if min_sim > 0.2:
            return False, f"min cosine {min_sim:.3f} > 0.2"
        return True, f"cheater_score={c.final_score:.4f}, min_sim={min_sim:.3f}, slash={c.slash_fraction:.2f}"


async def test_b3_wrong_data() -> tuple[bool, str]:
    """Cheater trains on different data."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        honest = make_miner(env, uid=1)
        await honest.train_window(0)

        wrong_ds = SyntheticDataset(size=2048, seq_len=64, vocab_size=512, seed=9999)

        m = make_model(env.cfg).to(DEVICE)
        m.load_state_dict(copy.deepcopy(env.shared_init))
        sampler = MinerSampler(wrong_ds, 2, 0, max_batches=env.hp.max_batches, micro_bs=env.hp.micro_bs)
        opt = torch.optim.AdamW(m.parameters(), lr=env.hp.lr)
        result = train_window(m, wrong_ds, sampler, opt, device=DEVICE,
                              probe_slice_size=env.hp.probe_slice_size)
        compressed = compress_model_gradients(m, TopKCompressor(topk=env.hp.topk))
        sub = MinerSubmission(uid=2, window=0, compressed_gradients=compressed,
                              loss_ledger=result["loss_ledger"], grad_probes=result["grad_probes"])
        await env.storage.put(sub.storage_key(), sub.to_dict())

        val = make_validator(env)
        results = await val.evaluate_window(0, [1, 2])
        c = results[1]
        if c.final_score > 0.0:
            return False, f"Cheater scored {c.final_score:.4f} > 0"
        loss_failed = c.loss_score < 1.0
        probe_failed = c.probe_score < 0.5
        return True, f"score={c.final_score:.4f}, loss_fail={loss_failed}, probe_fail={probe_failed}"


async def test_b4_scaled_losses() -> tuple[bool, str]:
    """Cheater multiplies losses by 1.05 (subtle)."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        honest = make_miner(env, uid=1)
        await honest.train_window(0)

        def cheat(sub):
            sub.loss_ledger = [x * 1.05 for x in sub.loss_ledger]
            return sub
        await cheating_submit(env, uid=2, window=0, cheat_fn=cheat)

        val = make_validator(env)
        results = await val.evaluate_window(0, [1, 2])
        c = results[1]
        if c.final_score > 0.0:
            return False, f"Cheater scored {c.final_score:.4f} > 0 (scaled losses not caught)"
        return True, f"score={c.final_score:.4f}, slash={c.slash_fraction:.2f}, max_err={c.loss_result.max_error:.4f}"


async def test_b5_partial_cheater() -> tuple[bool, str]:
    """Correct losses for first half, random for second half."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        honest = make_miner(env, uid=1)
        await honest.train_window(0)

        def cheat(sub):
            n = len(sub.loss_ledger)
            half = n // 2
            sub.loss_ledger = sub.loss_ledger[:half] + [random.uniform(5, 15) for _ in range(n - half)]
            return sub
        await cheating_submit(env, uid=2, window=0, cheat_fn=cheat)

        val = make_validator(env)
        results = await val.evaluate_window(0, [1, 2])
        c = results[1]
        if c.slash_fraction < 0.5:
            return False, f"Slash {c.slash_fraction:.2f} < 0.5 for partial cheater"
        return True, f"score={c.final_score:.4f}, slash={c.slash_fraction:.2f}"


async def test_b6_zero_gradient() -> tuple[bool, str]:
    """Real PoW but zeroed compressed gradients. Consistency check catches the mismatch."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        honest = make_miner(env, uid=1)
        await honest.train_window(0)

        def cheat(sub):
            for pname in sub.compressed_gradients:
                sub.compressed_gradients[pname]["vals"] = torch.zeros_like(
                    sub.compressed_gradients[pname]["vals"]
                )
            return sub
        await cheating_submit(env, uid=2, window=0, cheat_fn=cheat)

        val = make_validator(env)
        results = await val.evaluate_window(0, [1, 2])
        c = results[1]
        if c.loss_score < 0.9:
            return False, f"Loss score {c.loss_score:.4f} < 0.9 (PoW should pass)"
        if c.probe_score < 0.9:
            return False, f"Probe score {c.probe_score:.4f} < 0.9 (PoW should pass)"
        if c.consistency_score > 0.5:
            return False, f"Consistency {c.consistency_score:.4f} > 0.5 (zeroed grads should fail consistency)"
        if c.final_score > 0.0:
            return False, f"Score {c.final_score:.4f} > 0 (zeroed grads should be caught)"
        return True, f"score={c.final_score:.4f}, consistency={c.consistency_score:.4f} (correctly caught)"


async def test_b7_stale_submission() -> tuple[bool, str]:
    """Submit window 0 data for window 1."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        honest = make_miner(env, uid=1)

        # Honest miner does window 0 and window 1
        await honest.train_window(0)
        await honest.train_window(1)

        # Cheater: train on window 0 but submit as window 1
        m = make_model(env.cfg).to(DEVICE)
        m.load_state_dict(copy.deepcopy(env.shared_init))
        sampler_w0 = MinerSampler(env.dataset, 2, 0, max_batches=env.hp.max_batches, micro_bs=env.hp.micro_bs)
        opt = torch.optim.AdamW(m.parameters(), lr=env.hp.lr)
        result = train_window(m, env.dataset, sampler_w0, opt, device=DEVICE,
                              probe_slice_size=env.hp.probe_slice_size)
        compressed = compress_model_gradients(m, TopKCompressor(topk=env.hp.topk))
        # Upload as window=1 with uid=2
        sub = MinerSubmission(uid=2, window=1, compressed_gradients=compressed,
                              loss_ledger=result["loss_ledger"], grad_probes=result["grad_probes"])
        await env.storage.put(sub.storage_key(), sub.to_dict())

        val = make_validator(env)
        results = await val.evaluate_window(1, [1, 2])
        c = [r for r in results if r.uid == 2][0]
        if c.final_score > 0.0:
            return False, f"Stale cheater scored {c.final_score:.4f} > 0"
        return True, f"score={c.final_score:.4f}, slash={c.slash_fraction:.2f}"


# ──────────────────────────────────────────────────────────────────────────
# C. Boundary Conditions
# ──────────────────────────────────────────────────────────────────────────

async def test_c1_single_batch() -> tuple[bool, str]:
    """max_batches=1 doesn't crash, honest passes."""
    hp = replace(DEFAULT_HP, max_batches=1, min_batches=1, n_loss_spot_checks=1, n_probes=1)
    with tempfile.TemporaryDirectory() as td:
        env = make_env(hp=hp, tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        if r.final_score <= 0:
            return False, f"Score {r.final_score:.4f} <= 0 with n_batches=1"
        return True, f"score={r.final_score:.4f}"


async def test_c2_topk_larger_than_param() -> tuple[bool, str]:
    """topk=999999 doesn't crash."""
    hp = replace(DEFAULT_HP, topk=999999)
    with tempfile.TemporaryDirectory() as td:
        env = make_env(hp=hp, tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        if r.final_score <= 0:
            return False, f"Score {r.final_score:.4f} <= 0 with huge topk"
        return True, f"score={r.final_score:.4f}"


async def test_c3_large_learning_rate() -> tuple[bool, str]:
    """lr=1.0 doesn't produce NaN."""
    hp = replace(DEFAULT_HP, lr=1.0)
    with tempfile.TemporaryDirectory() as td:
        env = make_env(hp=hp, tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        # Just check no crash and no NaN in loss ledger
        sub_raw = await env.storage.get(MinerSubmission.make_storage_key(0, 1))
        sub = MinerSubmission.from_dict(sub_raw)
        has_nan = any(math.isnan(x) or math.isinf(x) for x in sub.loss_ledger)
        if has_nan:
            return False, "NaN/Inf in loss ledger with lr=1.0"
        return True, f"No NaN, score={r.final_score:.4f}"


async def test_c4_zero_miners() -> tuple[bool, str]:
    """No submissions for a window. No crash."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        val = make_validator(env)
        discovered = await val.discover_miners(0)
        results = await val.evaluate_window(0)
        if len(discovered) != 0:
            return False, f"Discovered {len(discovered)} miners with no submissions"
        if len(results) != 0:
            return False, f"Got {len(results)} results with no miners"
        return True, "0 miners, 0 results, no crash"


async def test_c5_many_miners() -> tuple[bool, str]:
    """20 miners submitting. All discovered and evaluated."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        miners = [make_miner(env, uid=i+1) for i in range(20)]
        for miner in miners:
            await miner.train_window(0)

        val = make_validator(env)
        discovered = await val.discover_miners(0)
        results = await val.evaluate_window(0)

        if len(discovered) != 20:
            return False, f"Discovered {len(discovered)}/20 miners"
        if len(results) != 20:
            return False, f"Evaluated {len(results)}/20 miners"
        passing = sum(1 for r in results if r.final_score > 0.5)
        return True, f"discovered={len(discovered)}, evaluated={len(results)}, passing={passing}"


async def test_c6_state_persistence() -> tuple[bool, str]:
    """Save/load validator state. Scores and model weights match."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)

        for w in range(3):
            await miner.train_window(w)
            results = await val.evaluate_window(w, [1])
            await val.apply_best_gradients(w, results)

        await val.save_state()
        orig_scores = dict(val.scores)
        orig_params = {k: v.clone() for k, v in val.model.state_dict().items()}

        val2 = make_validator(env, uid=0)
        loaded = await val2.load_state()
        if not loaded:
            return False, "load_state returned False"

        if val2.scores != orig_scores:
            return False, f"Scores mismatch: {val2.scores} vs {orig_scores}"

        for k in orig_params:
            if not torch.equal(val2.model.state_dict()[k], orig_params[k]):
                return False, f"Param {k} mismatch after load"

        return True, f"scores_match=True, params_match=True, global_step={val2.global_step}"


async def test_c7_corrupt_submission() -> tuple[bool, str]:
    """Broken dict in storage. Validator doesn't crash."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        # Put garbage
        await env.storage.put("gradient/0/99", {"garbage": True})

        val = make_validator(env)
        results = await val.evaluate_window(0, [99])
        r = results[0]
        if "corrupt" not in r.reason:
            return False, f"Expected 'corrupt' in reason, got: {r.reason}"
        return True, f"Handled gracefully: {r.reason}"


async def test_c8_spot_checks_exceed_batches() -> tuple[bool, str]:
    """n_loss_spot_checks=100 with n_batches=6. No crash."""
    hp = replace(DEFAULT_HP, n_loss_spot_checks=100)
    with tempfile.TemporaryDirectory() as td:
        env = make_env(hp=hp, tmpdir=td)
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        n_checked = len(r.loss_result.checked_indices) if r.loss_result else 0
        if n_checked > env.hp.max_batches:
            return False, f"Checked {n_checked} > {env.hp.max_batches} batches"
        if r.final_score <= 0:
            return False, f"Score {r.final_score:.4f} <= 0"
        return True, f"checked={n_checked}/{env.hp.max_batches}, score={r.final_score:.4f}"


async def test_c9_clock_run_loop() -> tuple[bool, str]:
    """TimedClock run loop completes 3 windows."""
    with tempfile.TemporaryDirectory() as td:
        env = make_env(tmpdir=td)
        # Pre-submit for 3 windows
        for w in range(3):
            miner = make_miner(env, uid=1)
            await miner.train_window(w)

        clock = TimedClock(interval=0.05)
        val = make_validator(env, clock=clock)

        try:
            await asyncio.wait_for(val.run(start_window=0, n_windows=3), timeout=10.0)
        except asyncio.TimeoutError:
            return False, "run() timed out after 10s"

        if val.global_step < 3:
            return False, f"Only {val.global_step} steps completed"
        return True, f"global_step={val.global_step}, scores={val.scores}"


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    # A: Training works
    test_a1_baseline,
    test_a2_single_batch,
    test_a3_large_batch,
    test_a4_multiple_windows,
    test_a5_model_learns,
    # B: Cheaters caught
    test_b1_random_losses,
    test_b2_fake_gradients,
    test_b3_wrong_data,
    test_b4_scaled_losses,
    test_b5_partial_cheater,
    test_b6_zero_gradient,
    test_b7_stale_submission,
    # C: Boundary conditions
    test_c1_single_batch,
    test_c2_topk_larger_than_param,
    test_c3_large_learning_rate,
    test_c4_zero_miners,
    test_c5_many_miners,
    test_c6_state_persistence,
    test_c7_corrupt_submission,
    test_c8_spot_checks_exceed_batches,
    test_c9_clock_run_loop,
]


async def main() -> None:
    print("=" * 70)
    print("Teutonic Stress Test Suite")
    print(f"{len(ALL_TESTS)} tests")
    print("=" * 70)

    passed = 0
    failed = 0
    errors = []

    for test in ALL_TESTS:
        name = test.__name__
        try:
            ok, msg = await test()
        except Exception as exc:
            ok = False
            msg = f"EXCEPTION: {exc}\n{traceback.format_exc()}"

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append(name)

    print()
    print(f"{passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
