#!/usr/bin/env python3
"""Convergence evaluation: proves the miner/validator loop actually reduces loss.

Runs N miners for W windows, evaluates the validator's model on a held-out
eval set after each window, and optionally logs everything to wandb.

Usage:
    source .venv/bin/activate
    python tests/test_convergence.py --backend local --wandb
    python tests/test_convergence.py --backend r2 --wandb
    python tests/test_convergence.py --backend local --n-miners 4 --n-windows 30
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

DEVICE = os.environ.get("TEUTONIC_DEVICE", "cpu")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from teutonic.hparams import HParams
from teutonic.logging import setup_logging
from teutonic.model import LlamaConfig, TinyLlama
from teutonic.storage.local import LocalFileStorage

from neurons.miner import Miner
from neurons.validator import Validator


CFG = LlamaConfig(
    vocab_size=32, hidden_dim=64, intermediate_dim=192,
    n_layers=2, n_heads=2, seq_len=128,
)

HP = HParams(
    max_batches=16, micro_bs=4, topk=512,
    lr=3e-3, outer_lr=0.7, max_grad_norm=1.0,
)

TRAIN_SEED = 42
EVAL_SEED = 7777


class PatternDataset:
    """Synthetic dataset with learnable structure.

    Each sequence repeats a short random motif (4-8 tokens) to fill
    seq_len.  The model can learn these repetition patterns easily,
    producing a clear convergence curve.
    """

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


def make_model(seed: int = 0) -> TinyLlama:
    torch.manual_seed(seed)
    return TinyLlama(CFG)


def eval_loss(model: TinyLlama, dataset: SyntheticDataset, n_samples: int = 64) -> float:
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


def _doppler_get(key: str) -> str:
    result = subprocess.run(
        ["doppler", "secrets", "get", key, "--plain",
         "--project", "arbos", "--config", "dev", "--no-check-version"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"doppler failed for {key}: {result.stderr}")
    return result.stdout.strip()


def make_r2_storage(prefix: str):
    from teutonic.storage.r2 import R2Storage
    return R2Storage(
        endpoint_url=_doppler_get("R2_URL"),
        access_key_id=_doppler_get("R2_ACCESS_KEY_ID"),
        secret_access_key=_doppler_get("R2_SECRET_ACCESS_KEY"),
        bucket_name=_doppler_get("R2_BUCKET_NAME"),
        prefix=prefix,
    )


async def run_convergence(
    backend: str,
    n_miners: int,
    n_windows: int,
    use_wandb: bool,
) -> bool:
    train_dataset = PatternDataset(size=4096, seq_len=CFG.seq_len, vocab_size=CFG.vocab_size, seed=TRAIN_SEED)
    eval_dataset = PatternDataset(size=256, seq_len=CFG.seq_len, vocab_size=CFG.vocab_size, seed=EVAL_SEED)

    shared_init = make_model(seed=0).state_dict()

    run_id = uuid.uuid4().hex[:8]
    storage = None
    tmpdir_obj = None

    if backend == "local":
        tmpdir_obj = tempfile.TemporaryDirectory()
        storage = LocalFileStorage(tmpdir_obj.name)
    elif backend == "r2":
        storage = make_r2_storage(f"teutonic/convergence/{run_id}/")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    reporter = None
    if use_wandb:
        wandb_key = _doppler_get("WANDB_API_KEY")
        import os
        os.environ["WANDB_API_KEY"] = wandb_key

        from teutonic.metrics import WandbReporter
        import dataclasses
        reporter = WandbReporter(
            project="teutonic",
            name=f"convergence-{backend}-{run_id}",
            config={
                "backend": backend,
                "n_miners": n_miners,
                "n_windows": n_windows,
                **dataclasses.asdict(HP),
                **dataclasses.asdict(CFG),
            },
            tags=["convergence", backend],
            group="convergence",
            job_type="eval",
        )

    miners = []
    for i in range(n_miners):
        m = make_model()
        m.load_state_dict(copy.deepcopy(shared_init))
        miners.append(Miner(uid=i + 1, model=m, dataset=train_dataset,
                            storage=storage, hparams=HP, device=DEVICE))

    val_model = make_model()
    val_model.load_state_dict(copy.deepcopy(shared_init))
    validator = Validator(uid=0, model=val_model, dataset=train_dataset,
                          storage=storage, hparams=HP, device=DEVICE)

    initial_loss = eval_loss(validator.model, eval_dataset)
    initial_ppl = math.exp(initial_loss)

    param_count = sum(p.numel() for p in val_model.parameters())
    print("=" * 72)
    print(f"Convergence Test | backend={backend} | run_id={run_id}")
    print(f"Model: {param_count:,} params | {n_miners} miners | {n_windows} windows")
    print(f"Initial eval loss: {initial_loss:.4f} (ppl={initial_ppl:.2f})")
    print(f"Wandb: {'enabled' if use_wandb else 'disabled'}")
    print("=" * 72)

    print(f"\n{'Window':>6}  {'Eval Loss':>10}  {'PPL':>8}  {'Train Loss':>11}  "
          f"{'Passing':>7}  {'Duration':>8}")
    print("-" * 65)

    losses: list[float] = [initial_loss]
    has_nan = False

    if reporter:
        reporter.log({
            "eval/loss": initial_loss,
            "eval/perplexity": initial_ppl,
            "meta/window": -1,
            "meta/backend": backend,
        }, step=0)

    try:
        for w in range(n_windows):
            t0 = time.monotonic()

            for miner in miners:
                await miner.train_window(w)

            uids = [m.uid for m in miners]
            results = await validator.evaluate_window(w, uids)
            await validator.apply_best_gradients(w, results)

            # Sync miners to the validator's model so verification keeps
            # working across windows (mirrors production model distribution).
            val_state = validator.model.state_dict()
            for miner in miners:
                miner.model.load_state_dict(copy.deepcopy(val_state))

            e_loss = eval_loss(validator.model, eval_dataset)
            e_ppl = math.exp(e_loss) if math.isfinite(e_loss) else float("inf")
            duration = time.monotonic() - t0

            n_passing = sum(1 for r in results if r.final_score > 0)
            finite_train_losses = []
            for r in results:
                if r.loss_result and r.loss_result.reported:
                    finite_train_losses.extend(
                        l for l in r.loss_result.reported if math.isfinite(l)
                    )
            train_mean = sum(finite_train_losses) / len(finite_train_losses) if finite_train_losses else float("nan")

            losses.append(e_loss)
            if not math.isfinite(e_loss):
                has_nan = True

            print(f"{w:>6}  {e_loss:>10.4f}  {e_ppl:>8.2f}  {train_mean:>11.4f}  "
                  f"{n_passing:>7}  {duration:>7.2f}s")

            if reporter:
                metrics: dict[str, Any] = {
                    "eval/loss": e_loss,
                    "eval/perplexity": e_ppl,
                    "train/mean_loss": train_mean,
                    "aggregate/n_passing": n_passing,
                    "meta/window": w,
                    "meta/backend": backend,
                    "meta/duration_s": round(duration, 3),
                }
                for r in results:
                    metrics[f"miner/{r.uid}/final_score"] = r.final_score
                    metrics[f"miner/{r.uid}/loss_score"] = r.loss_score
                reporter.log(metrics, step=w + 1)

    finally:
        if reporter:
            reporter.close()
        if backend == "r2" and storage is not None:
            from teutonic.storage.r2 import R2Storage
            if isinstance(storage, R2Storage):
                n_cleaned = await storage.delete_prefix("")
                print(f"\nR2 cleanup: deleted {n_cleaned} objects")
                await storage.close()
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()

    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n{'=' * 72}")
    print(f"Initial loss: {initial_loss:.4f} -> Final loss: {final_loss:.4f}")
    print(f"Reduction: {reduction:.1f}%")

    passed = True
    if has_nan:
        print("FAIL: NaN/Inf detected in eval losses")
        passed = False
    if reduction < 5.0:
        print(f"FAIL: Loss reduction {reduction:.1f}% < 5% threshold")
        passed = False

    if passed:
        print("PASS: Model is converging")
    print("=" * 72)
    return passed


def main():
    parser = argparse.ArgumentParser(description="Teutonic convergence test")
    parser.add_argument("--backend", choices=["local", "r2"], default="local")
    parser.add_argument("--n-miners", type=int, default=3)
    parser.add_argument("--n-windows", type=int, default=25)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO")

    passed = asyncio.run(run_convergence(
        backend=args.backend,
        n_miners=args.n_miners,
        n_windows=args.n_windows,
        use_wandb=args.wandb,
    ))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
