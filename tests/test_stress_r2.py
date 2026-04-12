#!/usr/bin/env python3
"""R2 stress test suite: all 21 original tests + 3 R2-specific tests.

Runs the full PoW verification suite against live Cloudflare R2 instead of
local filesystem.  Each test gets a unique R2 prefix for isolation.
Credentials are fetched from Doppler.

Usage:
    source .venv/bin/activate
    python test_stress_r2.py
"""

from __future__ import annotations

import asyncio
import copy
import math
import random
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
from teutonic.probe_spec import ProbeSpec
from teutonic.sampler import MinerSampler
from teutonic.storage.r2 import R2Storage
from teutonic.submission import MinerSubmission

from neurons.miner import Miner
from neurons.trainer import train_window
from neurons.validator import Validator, SlashConfig


# ──────────────────────────────────────────────────────────────────────────
# R2 credentials from Doppler
# ──────────────────────────────────────────────────────────────────────────

def _doppler_get(key: str) -> str:
    result = subprocess.run(
        ["doppler", "secrets", "get", key, "--plain",
         "--project", "arbos", "--config", "dev", "--no-check-version"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"doppler failed for {key}: {result.stderr}")
    return result.stdout.strip()


_R2_ENDPOINT = None
_R2_ACCESS_KEY = None
_R2_SECRET_KEY = None
_R2_BUCKET = None


def _load_creds():
    global _R2_ENDPOINT, _R2_ACCESS_KEY, _R2_SECRET_KEY, _R2_BUCKET
    if _R2_ENDPOINT is None:
        _R2_ENDPOINT = _doppler_get("R2_URL")
        _R2_ACCESS_KEY = _doppler_get("R2_ACCESS_KEY_ID")
        _R2_SECRET_KEY = _doppler_get("R2_SECRET_ACCESS_KEY")
        _R2_BUCKET = _doppler_get("R2_BUCKET_NAME")


def _make_r2(prefix: str) -> R2Storage:
    _load_creds()
    return R2Storage(
        endpoint_url=_R2_ENDPOINT,
        access_key_id=_R2_ACCESS_KEY,
        secret_access_key=_R2_SECRET_KEY,
        bucket_name=_R2_BUCKET,
        prefix=prefix,
    )


# ──────────────────────────────────────────────────────────────────────────
# Test infrastructure (R2-backed)
# ──────────────────────────────────────────────────────────────────────────

RUN_ID = uuid.uuid4().hex[:8]

DEFAULT_CFG = LlamaConfig(
    vocab_size=512, hidden_dim=64, intermediate_dim=128,
    n_layers=2, n_heads=2, seq_len=64,
)

DEFAULT_HP = HParams(n_batches=6, micro_bs=2, topk=32, lr=1e-3, outer_lr=0.4)


def make_model(cfg: LlamaConfig = DEFAULT_CFG, seed: int = 0) -> TinyLlama:
    torch.manual_seed(seed)
    return TinyLlama(cfg)


@dataclass
class TestEnv:
    storage: R2Storage
    dataset: SyntheticDataset
    shared_init: dict
    cfg: LlamaConfig
    hp: HParams


def make_env(
    test_name: str,
    hp: HParams = DEFAULT_HP,
    cfg: LlamaConfig = DEFAULT_CFG,
    dataset_size: int = 2048,
) -> TestEnv:
    prefix = f"teutonic/stress/{RUN_ID}/{test_name}/"
    storage = _make_r2(prefix)
    dataset = SyntheticDataset(size=dataset_size, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, seed=42)
    shared_init = make_model(cfg, seed=0).state_dict()
    return TestEnv(storage=storage, dataset=dataset, shared_init=shared_init, cfg=cfg, hp=hp)


def make_miner(env: TestEnv, uid: int) -> Miner:
    m = make_model(env.cfg)
    m.load_state_dict(copy.deepcopy(env.shared_init))
    return Miner(uid=uid, model=m, dataset=env.dataset, storage=env.storage, hparams=env.hp, device="cpu")


def make_validator(env: TestEnv, uid: int = 0, clock: Any = None) -> Validator:
    m = make_model(env.cfg)
    m.load_state_dict(copy.deepcopy(env.shared_init))
    return Validator(uid=uid, model=m, dataset=env.dataset, storage=env.storage,
                     hparams=env.hp, clock=clock, device="cpu")


async def cheating_submit(env: TestEnv, uid: int, window: int, cheat_fn) -> MinerSubmission:
    m = make_model(env.cfg)
    m.load_state_dict(copy.deepcopy(env.shared_init))
    sampler = MinerSampler(env.dataset, uid, window, n_batches=env.hp.n_batches, micro_bs=env.hp.micro_bs)
    all_indices = set(range(sampler.total_micro_batches))
    probe_spec = ProbeSpec(
        param_name=env.hp.probe_param_name, slice_start=env.hp.probe_slice_start,
        slice_end=env.hp.probe_slice_end, batch_indices=tuple(sorted(all_indices)),
    )
    optimizer = torch.optim.AdamW(m.parameters(), lr=env.hp.lr)
    result = train_window(m, env.dataset, sampler, optimizer,
                          probe_indices=all_indices, probe_spec=probe_spec, device="cpu")
    compressed = compress_model_gradients(m, TopKCompressor(topk=env.hp.topk))
    sub = MinerSubmission(uid=uid, window=window, compressed_gradients=compressed,
                          loss_ledger=result["loss_ledger"], grad_probes=result["grad_probes"])
    sub = cheat_fn(sub)
    await env.storage.put(sub.storage_key(), sub.to_dict())
    return sub


async def cleanup_env(env: TestEnv) -> None:
    await env.storage.delete_prefix("")
    await env.storage.close()


# ──────────────────────────────────────────────────────────────────────────
# A. Prove Training Works (over R2)
# ──────────────────────────────────────────────────────────────────────────

async def test_a1_baseline() -> tuple[bool, str]:
    env = make_env("a1")
    try:
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
    finally:
        await cleanup_env(env)


async def test_a2_single_batch() -> tuple[bool, str]:
    hp = replace(DEFAULT_HP, n_batches=1, n_loss_spot_checks=1, n_probes=1)
    env = make_env("a2", hp=hp)
    try:
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
    finally:
        await cleanup_env(env)


async def test_a3_large_batch() -> tuple[bool, str]:
    hp = replace(DEFAULT_HP, n_batches=20, micro_bs=4)
    env = make_env("a3", hp=hp)
    try:
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        if r.final_score < 0.9:
            return False, f"score {r.final_score:.4f} < 0.9"
        return True, f"score={r.final_score:.4f}, n_batches=20"
    finally:
        await cleanup_env(env)


async def test_a4_multiple_windows() -> tuple[bool, str]:
    env = make_env("a4")
    try:
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
            return False, f"Some window score < 0.8"
        return True, f"EMA={ema:.4f}, min_window={min(window_scores):.4f}"
    finally:
        await cleanup_env(env)


async def test_a5_model_learns() -> tuple[bool, str]:
    env = make_env("a5")
    try:
        miner = make_miner(env, uid=1)
        val = make_validator(env)

        def eval_loss(model, dataset, n=10):
            model.eval()
            total = 0.0
            with torch.no_grad():
                for i in range(n):
                    tokens = dataset[i].unsqueeze(0)
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
    finally:
        await cleanup_env(env)


# ──────────────────────────────────────────────────────────────────────────
# B. Prove Cheaters Get Caught (over R2)
# ──────────────────────────────────────────────────────────────────────────

async def test_b1_random_losses() -> tuple[bool, str]:
    env = make_env("b1")
    try:
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
    finally:
        await cleanup_env(env)


async def test_b2_fake_gradients() -> tuple[bool, str]:
    env = make_env("b2")
    try:
        honest = make_miner(env, uid=1)
        await honest.train_window(0)
        def cheat(sub):
            sub.grad_probes = {k: torch.randn_like(v) for k, v in sub.grad_probes.items()}
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
    finally:
        await cleanup_env(env)


async def test_b3_wrong_data() -> tuple[bool, str]:
    env = make_env("b3")
    try:
        honest = make_miner(env, uid=1)
        await honest.train_window(0)
        wrong_ds = SyntheticDataset(size=2048, seq_len=64, vocab_size=512, seed=9999)
        m = make_model(env.cfg)
        m.load_state_dict(copy.deepcopy(env.shared_init))
        sampler = MinerSampler(wrong_ds, 2, 0, n_batches=env.hp.n_batches, micro_bs=env.hp.micro_bs)
        all_idx = set(range(sampler.total_micro_batches))
        ps = ProbeSpec(param_name=env.hp.probe_param_name, slice_start=env.hp.probe_slice_start,
                       slice_end=env.hp.probe_slice_end, batch_indices=tuple(sorted(all_idx)))
        opt = torch.optim.AdamW(m.parameters(), lr=env.hp.lr)
        result = train_window(m, wrong_ds, sampler, opt, probe_indices=all_idx, probe_spec=ps, device="cpu")
        compressed = compress_model_gradients(m, TopKCompressor(topk=env.hp.topk))
        sub = MinerSubmission(uid=2, window=0, compressed_gradients=compressed,
                              loss_ledger=result["loss_ledger"], grad_probes=result["grad_probes"])
        await env.storage.put(sub.storage_key(), sub.to_dict())
        val = make_validator(env)
        results = await val.evaluate_window(0, [1, 2])
        c = results[1]
        if c.final_score > 0.0:
            return False, f"Cheater scored {c.final_score:.4f} > 0"
        return True, f"score={c.final_score:.4f}, loss_fail={c.loss_score < 1.0}, probe_fail={c.probe_score < 0.5}"
    finally:
        await cleanup_env(env)


async def test_b4_scaled_losses() -> tuple[bool, str]:
    env = make_env("b4")
    try:
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
            return False, f"Cheater scored {c.final_score:.4f} > 0"
        return True, f"score={c.final_score:.4f}, slash={c.slash_fraction:.2f}, max_err={c.loss_result.max_error:.4f}"
    finally:
        await cleanup_env(env)


async def test_b5_partial_cheater() -> tuple[bool, str]:
    env = make_env("b5")
    try:
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
            return False, f"Slash {c.slash_fraction:.2f} < 0.5"
        return True, f"score={c.final_score:.4f}, slash={c.slash_fraction:.2f}"
    finally:
        await cleanup_env(env)


async def test_b6_zero_gradient() -> tuple[bool, str]:
    env = make_env("b6")
    try:
        honest = make_miner(env, uid=1)
        await honest.train_window(0)
        def cheat(sub):
            for pname in sub.compressed_gradients:
                sub.compressed_gradients[pname]["vals"] = torch.zeros_like(
                    sub.compressed_gradients[pname]["vals"])
            return sub
        await cheating_submit(env, uid=2, window=0, cheat_fn=cheat)
        val = make_validator(env)
        results = await val.evaluate_window(0, [1, 2])
        c = results[1]
        if c.loss_score < 0.9:
            return False, f"Loss score {c.loss_score:.4f} < 0.9"
        if c.probe_score < 0.9:
            return False, f"Probe score {c.probe_score:.4f} < 0.9"
        if c.final_score <= 0.0:
            return False, f"Score {c.final_score:.4f} <= 0"
        return True, f"score={c.final_score:.4f} (PoW passes)"
    finally:
        await cleanup_env(env)


async def test_b7_stale_submission() -> tuple[bool, str]:
    env = make_env("b7")
    try:
        honest = make_miner(env, uid=1)
        await honest.train_window(0)
        await honest.train_window(1)
        m = make_model(env.cfg)
        m.load_state_dict(copy.deepcopy(env.shared_init))
        sampler_w0 = MinerSampler(env.dataset, 2, 0, n_batches=env.hp.n_batches, micro_bs=env.hp.micro_bs)
        all_idx = set(range(sampler_w0.total_micro_batches))
        ps = ProbeSpec(param_name=env.hp.probe_param_name, slice_start=env.hp.probe_slice_start,
                       slice_end=env.hp.probe_slice_end, batch_indices=tuple(sorted(all_idx)))
        opt = torch.optim.AdamW(m.parameters(), lr=env.hp.lr)
        result = train_window(m, env.dataset, sampler_w0, opt, probe_indices=all_idx, probe_spec=ps, device="cpu")
        compressed = compress_model_gradients(m, TopKCompressor(topk=env.hp.topk))
        sub = MinerSubmission(uid=2, window=1, compressed_gradients=compressed,
                              loss_ledger=result["loss_ledger"], grad_probes=result["grad_probes"])
        await env.storage.put(sub.storage_key(), sub.to_dict())
        val = make_validator(env)
        results = await val.evaluate_window(1, [1, 2])
        c = [r for r in results if r.uid == 2][0]
        if c.final_score > 0.0:
            return False, f"Stale cheater scored {c.final_score:.4f} > 0"
        return True, f"score={c.final_score:.4f}, slash={c.slash_fraction:.2f}"
    finally:
        await cleanup_env(env)


# ──────────────────────────────────────────────────────────────────────────
# C. Boundary Conditions (over R2)
# ──────────────────────────────────────────────────────────────────────────

async def test_c1_single_batch() -> tuple[bool, str]:
    hp = replace(DEFAULT_HP, n_batches=1, n_loss_spot_checks=1, n_probes=1)
    env = make_env("c1", hp=hp)
    try:
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        if r.final_score <= 0:
            return False, f"Score {r.final_score:.4f} <= 0"
        return True, f"score={r.final_score:.4f}"
    finally:
        await cleanup_env(env)


async def test_c2_topk_larger_than_param() -> tuple[bool, str]:
    hp = replace(DEFAULT_HP, topk=999999)
    env = make_env("c2", hp=hp)
    try:
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        if r.final_score <= 0:
            return False, f"Score {r.final_score:.4f} <= 0"
        return True, f"score={r.final_score:.4f}"
    finally:
        await cleanup_env(env)


async def test_c3_large_learning_rate() -> tuple[bool, str]:
    hp = replace(DEFAULT_HP, lr=1.0)
    env = make_env("c3", hp=hp)
    try:
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        sub_raw = await env.storage.get(MinerSubmission.make_storage_key(0, 1))
        sub = MinerSubmission.from_dict(sub_raw)
        has_nan = any(math.isnan(x) or math.isinf(x) for x in sub.loss_ledger)
        if has_nan:
            return False, "NaN/Inf in loss ledger"
        return True, f"No NaN, score={r.final_score:.4f}"
    finally:
        await cleanup_env(env)


async def test_c4_zero_miners() -> tuple[bool, str]:
    env = make_env("c4")
    try:
        val = make_validator(env)
        discovered = await val.discover_miners(0)
        results = await val.evaluate_window(0)
        if len(discovered) != 0:
            return False, f"Discovered {len(discovered)} miners"
        if len(results) != 0:
            return False, f"Got {len(results)} results"
        return True, "0 miners, 0 results, no crash"
    finally:
        await cleanup_env(env)


async def test_c5_many_miners() -> tuple[bool, str]:
    env = make_env("c5")
    try:
        miners = [make_miner(env, uid=i+1) for i in range(20)]
        await asyncio.gather(*[m.train_window(0) for m in miners])
        val = make_validator(env)
        discovered = await val.discover_miners(0)
        results = await val.evaluate_window(0)
        if len(discovered) != 20:
            return False, f"Discovered {len(discovered)}/20"
        if len(results) != 20:
            return False, f"Evaluated {len(results)}/20"
        passing = sum(1 for r in results if r.final_score > 0.5)
        return True, f"discovered={len(discovered)}, evaluated={len(results)}, passing={passing}"
    finally:
        await cleanup_env(env)


async def test_c6_state_persistence() -> tuple[bool, str]:
    env = make_env("c6")
    try:
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
            return False, "Scores mismatch"
        for k in orig_params:
            if not torch.equal(val2.model.state_dict()[k], orig_params[k]):
                return False, f"Param {k} mismatch"
        return True, f"scores_match=True, params_match=True, global_step={val2.global_step}"
    finally:
        await cleanup_env(env)


async def test_c7_corrupt_submission() -> tuple[bool, str]:
    env = make_env("c7")
    try:
        await env.storage.put("gradient/0/99", {"garbage": True})
        val = make_validator(env)
        results = await val.evaluate_window(0, [99])
        r = results[0]
        if "corrupt" not in r.reason:
            return False, f"Expected 'corrupt' in reason, got: {r.reason}"
        return True, f"Handled gracefully: {r.reason}"
    finally:
        await cleanup_env(env)


async def test_c8_spot_checks_exceed_batches() -> tuple[bool, str]:
    hp = replace(DEFAULT_HP, n_loss_spot_checks=100)
    env = make_env("c8", hp=hp)
    try:
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        await miner.train_window(0)
        results = await val.evaluate_window(0, [1])
        r = results[0]
        n_checked = len(r.loss_result.checked_indices) if r.loss_result else 0
        if n_checked > env.hp.n_batches:
            return False, f"Checked {n_checked} > {env.hp.n_batches}"
        if r.final_score <= 0:
            return False, f"Score {r.final_score:.4f} <= 0"
        return True, f"checked={n_checked}/{env.hp.n_batches}, score={r.final_score:.4f}"
    finally:
        await cleanup_env(env)


async def test_c9_clock_run_loop() -> tuple[bool, str]:
    env = make_env("c9")
    try:
        for w in range(3):
            miner = make_miner(env, uid=1)
            await miner.train_window(w)
        clock = TimedClock(interval=0.05)
        val = make_validator(env, clock=clock)
        try:
            await asyncio.wait_for(val.run(start_window=0, n_windows=3), timeout=30.0)
        except asyncio.TimeoutError:
            return False, "run() timed out"
        if val.global_step < 3:
            return False, f"Only {val.global_step} steps"
        return True, f"global_step={val.global_step}"
    finally:
        await cleanup_env(env)


# ──────────────────────────────────────────────────────────────────────────
# R2-specific stress tests
# ──────────────────────────────────────────────────────────────────────────

async def test_r2_1_concurrent_miners() -> tuple[bool, str]:
    """10 miners submit simultaneously via asyncio.gather, validator evaluates all."""
    env = make_env("r2_1")
    try:
        miners = [make_miner(env, uid=i+1) for i in range(10)]
        t0 = time.perf_counter()
        await asyncio.gather(*[m.train_window(0) for m in miners])
        t_submit = time.perf_counter() - t0

        val = make_validator(env)
        t0 = time.perf_counter()
        results = await val.evaluate_window(0)
        t_eval = time.perf_counter() - t0

        if len(results) != 10:
            return False, f"Evaluated {len(results)}/10"
        passing = sum(1 for r in results if r.final_score > 0.5)
        if passing != 10:
            return False, f"Only {passing}/10 passed"
        return True, f"10 miners: submit={t_submit:.2f}s, eval={t_eval:.2f}s, all passed"
    finally:
        await cleanup_env(env)


async def test_r2_2_rapid_windows() -> tuple[bool, str]:
    """5 windows back-to-back, measure total pipeline time over R2."""
    env = make_env("r2_2")
    try:
        miner = make_miner(env, uid=1)
        val = make_validator(env)
        t0 = time.perf_counter()
        for w in range(5):
            await miner.train_window(w)
            results = await val.evaluate_window(w, [1])
            await val.apply_best_gradients(w, results)
        total = time.perf_counter() - t0
        ema = val.get_effective_score(1)
        if ema < 0.9:
            return False, f"EMA {ema:.4f} < 0.9"
        return True, f"5 windows in {total:.2f}s ({total/5:.2f}s/window), EMA={ema:.4f}"
    finally:
        await cleanup_env(env)


async def test_r2_3_large_submission() -> tuple[bool, str]:
    """Large submissions (n_batches=20, micro_bs=4). Measure R2 payload times."""
    hp = replace(DEFAULT_HP, n_batches=20, micro_bs=4)
    env = make_env("r2_3", hp=hp)
    try:
        miner = make_miner(env, uid=1)
        t0 = time.perf_counter()
        await miner.train_window(0)
        t_submit = time.perf_counter() - t0

        val = make_validator(env)
        t0 = time.perf_counter()
        results = await val.evaluate_window(0, [1])
        t_eval = time.perf_counter() - t0

        r = results[0]
        if r.final_score < 0.9:
            return False, f"Score {r.final_score:.4f} < 0.9"
        return True, f"large sub: submit={t_submit:.2f}s, eval={t_eval:.2f}s, score={r.final_score:.4f}"
    finally:
        await cleanup_env(env)


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_a1_baseline, test_a2_single_batch, test_a3_large_batch,
    test_a4_multiple_windows, test_a5_model_learns,
    test_b1_random_losses, test_b2_fake_gradients, test_b3_wrong_data,
    test_b4_scaled_losses, test_b5_partial_cheater, test_b6_zero_gradient,
    test_b7_stale_submission,
    test_c1_single_batch, test_c2_topk_larger_than_param,
    test_c3_large_learning_rate, test_c4_zero_miners, test_c5_many_miners,
    test_c6_state_persistence, test_c7_corrupt_submission,
    test_c8_spot_checks_exceed_batches, test_c9_clock_run_loop,
    test_r2_1_concurrent_miners, test_r2_2_rapid_windows,
    test_r2_3_large_submission,
]


async def main() -> None:
    _load_creds()
    print("=" * 72)
    print(f"R2 Stress Test Suite ({len(ALL_TESTS)} tests)")
    print(f"Bucket: {_R2_BUCKET}  |  Run ID: {RUN_ID}")
    print("=" * 72)

    passed = 0
    failed = 0
    errors = []
    timings: dict[str, float] = {}

    for test in ALL_TESTS:
        name = test.__name__
        t0 = time.perf_counter()
        try:
            ok, msg = await test()
        except Exception as exc:
            ok = False
            msg = f"EXCEPTION: {exc}\n{traceback.format_exc()}"
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} ({elapsed:.1f}s): {msg}")
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append(name)

    # Final cleanup of entire run prefix
    cleanup_r2 = _make_r2(f"teutonic/stress/{RUN_ID}/")
    n_cleaned = await cleanup_r2.delete_prefix("")
    print(f"\n  Cleanup: deleted {n_cleaned} leftover R2 objects")

    total_time = sum(timings.values())
    print(f"\n{passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    print(f"Total time: {total_time:.1f}s (avg {total_time/len(ALL_TESTS):.1f}s/test)")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
