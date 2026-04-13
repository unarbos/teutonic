#!/usr/bin/env python3
"""Standalone miner entrypoint for distributed convergence testing.

Runs on a GPU pod, trains each window, uploads to R2, and syncs the
model from R2 between windows (published by the validator).

Model sync is pipelined: the download for window W+1 starts immediately
after uploading window W's gradient and runs concurrently with the clock
wait.  By the time the next window starts the model is usually already
fetched.

Usage:
    python neurons/run_miner.py \
        --uid 1 --n-windows 20 --window-duration 30 \
        --t0 1713000000.0 --run-id abc12345 \
        --r2-url ... --r2-key-id ... --r2-secret ... --r2-bucket ...
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from teutonic.hparams import HParams
from teutonic.logging import setup_logging
from teutonic.model import LlamaConfig, TinyLlama
from teutonic.storage.r2 import R2Storage

from neurons.miner import Miner

logger = structlog.get_logger(__name__)

CFG = LlamaConfig(
    vocab_size=32, hidden_dim=64, intermediate_dim=192,
    n_layers=2, n_heads=2, seq_len=128,
)

HP = HParams(
    max_batches=32, micro_bs=4, topk=512,
    lr=3e-3, outer_lr=0.7, max_grad_norm=1.0,
    upload_budget_s=5.0,
)

TRAIN_SEED = 42


class PatternDataset:
    """Synthetic dataset with learnable repeating motifs."""

    def __init__(self, size: int, seq_len: int, vocab_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        seqs = []
        for _ in range(size):
            motif_len = rng.integers(4, 9)
            motif = rng.integers(0, vocab_size, size=motif_len)
            repeats = (seq_len // motif_len) + 1
            seq = np.tile(motif, repeats)[:seq_len]
            seqs.append(seq)
        self._data = torch.from_numpy(np.array(seqs, dtype=np.int64))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


class SyncTimedClock:
    """TimedClock with a shared t0 for cross-machine synchronization."""

    def __init__(self, t0: float, interval: float):
        self._t0 = t0
        self._interval = interval

    @property
    def current_window(self) -> int:
        return int((time.time() - self._t0) / self._interval)

    @property
    def window_duration(self) -> float:
        return self._interval

    def window_end_time(self, window: int) -> float:
        return self._t0 + (window + 1) * self._interval

    def window_block_hash(self, window: int) -> str:
        return hashlib.blake2b(
            f"local-block:{window}".encode(), digest_size=32
        ).hexdigest()

    async def wait_for_window(self, target: int) -> None:
        while self.current_window < target:
            await asyncio.sleep(0.05)


async def fetch_model_state(
    storage: R2Storage, window: int, timeout: float = 45.0,
) -> dict[str, Any] | None:
    """Poll R2 for model_state/{window} and return the state dict.

    Runs as a background task -- does NOT touch the model directly.
    Returns the state_dict on success, None on timeout.
    """
    key = f"model_state/{window}"
    t0 = time.monotonic()
    attempt = 0
    while time.monotonic() - t0 < timeout:
        data = await storage.get(key)
        if data is not None and "model_state_dict" in data:
            logger.info(
                "miner.fetch_model.ok",
                window=window, attempt=attempt,
                elapsed_s=round(time.monotonic() - t0, 2),
            )
            return data["model_state_dict"]
        attempt += 1
        await asyncio.sleep(0.5)
    logger.error("miner.fetch_model.timeout", window=window, timeout_s=timeout)
    return None


async def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("miner.init", uid=args.uid, device=device, run_id=args.run_id)

    dataset = PatternDataset(
        size=4096, seq_len=CFG.seq_len,
        vocab_size=CFG.vocab_size, seed=TRAIN_SEED,
    )

    storage = R2Storage(
        endpoint_url=args.r2_url,
        access_key_id=args.r2_key_id,
        secret_access_key=args.r2_secret,
        bucket_name=args.r2_bucket,
        prefix=f"teutonic/distributed/{args.run_id}/",
    )

    torch.manual_seed(0)
    model = TinyLlama(CFG).to(device)

    clock = SyncTimedClock(t0=args.t0, interval=args.window_duration)
    structlog.contextvars.bind_contextvars(role="miner", uid=args.uid)

    # Sync initial model (window 0 state published by validator before t0)
    initial_state = await fetch_model_state(storage, 0, timeout=60.0)
    if initial_state is None:
        logger.error("miner.init.model_sync_failed")
        return
    model.load_state_dict(initial_state)
    model.to(device)
    logger.info("miner.init.model_loaded", params=sum(p.numel() for p in model.parameters()))

    miner_obj = Miner(
        uid=args.uid, model=model, dataset=dataset,
        storage=storage, hparams=HP, clock=clock, device=device,
    )

    pending_sync: asyncio.Task | None = None

    try:
        for w in range(args.n_windows):
            # Finalize the background model sync from last iteration
            if pending_sync is not None:
                state_dict = await pending_sync
                if state_dict is not None:
                    miner_obj.model.load_state_dict(state_dict)
                    miner_obj.model.to(device)
                else:
                    logger.warning("miner.sync_missed", window=w)
                pending_sync = None

            await clock.wait_for_window(w)
            deadline = clock.window_end_time(w)
            await miner_obj.train_window(w, deadline=deadline)

            # Fire off background download for next window's model state
            if w + 1 < args.n_windows:
                pending_sync = asyncio.create_task(
                    fetch_model_state(storage, w + 1, timeout=args.window_duration + 15)
                )
    finally:
        if pending_sync is not None:
            pending_sync.cancel()
        logger.info("miner.shutdown", global_step=miner_obj.global_step)
        await storage.close()


def main():
    parser = argparse.ArgumentParser(description="Distributed miner")
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--n-windows", type=int, default=25)
    parser.add_argument("--window-duration", type=float, default=20.0)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--r2-url", type=str, required=True)
    parser.add_argument("--r2-key-id", type=str, required=True)
    parser.add_argument("--r2-secret", type=str, required=True)
    parser.add_argument("--r2-bucket", type=str, required=True)
    args = parser.parse_args()

    setup_logging(level="INFO")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
