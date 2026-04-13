#!/usr/bin/env python3
"""Distributed convergence test across Lium GPU pods.

Orchestrator that:
1. Provisions pods via SSH (rsync code, install deps)
2. Publishes initial model to R2
3. Starts miners on remote GPU pods
4. Runs validator locally, publishing model updates to R2
5. Evaluates convergence and logs to wandb

Usage:
    source .venv/bin/activate
    python tests/test_distributed.py --wandb
    python tests/test_distributed.py --n-windows 25 --window-duration 15
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from teutonic.hparams import HParams
from teutonic.logging import setup_logging
from teutonic.model import LlamaConfig, TinyLlama
from teutonic.storage.r2 import R2Storage

from neurons.validator import Validator

PODS = [
    {"name": "teutonic-1", "host": "91.224.44.226", "port": 20300},
    {"name": "teutonic-2", "host": "91.224.44.207", "port": 40060},
    {"name": "teutonic-3", "host": "91.224.44.222", "port": 20100},
    {"name": "teutonic-4", "host": "91.224.44.85",  "port": 50299},
    {"name": "teutonic-5", "host": "91.224.44.81",  "port": 20409},
]

SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15"

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
EVAL_SEED = 7777


class PatternDataset:
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
        import hashlib
        return hashlib.blake2b(
            f"local-block:{window}".encode(), digest_size=32
        ).hexdigest()

    async def wait_for_window(self, target: int) -> None:
        while self.current_window < target:
            await asyncio.sleep(0.05)


def make_model(seed: int = 0) -> TinyLlama:
    torch.manual_seed(seed)
    return TinyLlama(CFG)


def eval_loss(model: TinyLlama, dataset: PatternDataset, n_samples: int = 64) -> float:
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


def ssh_cmd(pod: dict, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    full = f"ssh {SSH_OPTS} -p {pod['port']} root@{pod['host']} {cmd!r}"
    return subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)


def ssh_bg(pod: dict, cmd: str) -> subprocess.Popen:
    """Start a command on a pod in the background, streaming output."""
    full = f"ssh {SSH_OPTS} -p {pod['port']} root@{pod['host']} {cmd!r}"
    return subprocess.Popen(
        full, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )


def sync_to_pod(pod: dict, local_dir: str, remote_dir: str) -> None:
    """Tar the source tree locally and extract on the pod via SSH pipe."""
    ssh_target = f"ssh {SSH_OPTS} -p {pod['port']} root@{pod['host']}"
    ssh_cmd(pod, f"mkdir -p {remote_dir}", timeout=10)
    cmd = (
        f"tar czf - -C {local_dir} "
        f"--exclude='.venv' --exclude='__pycache__' --exclude='wandb' "
        f"--exclude='.git' --exclude='*.pyc' . "
        f"| {ssh_target} 'tar xzf - -C {remote_dir}'"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"sync to {pod['name']} failed: {result.stderr}")


def provision_pod(pod: dict) -> None:
    """Rsync code and install deps on a single pod."""
    teutonic_dir = str(Path(__file__).parent.parent)
    print(f"  [{pod['name']}] Syncing code...")
    sync_to_pod(pod, teutonic_dir, "/workspace/teutonic")

    print(f"  [{pod['name']}] Installing deps...")
    install_script = (
        "cd /workspace/teutonic && "
        "python3 -m venv --system-site-packages .venv 2>/dev/null; "
        ". .venv/bin/activate && "
        "pip install -e '.[r2]' -q 2>&1 | tail -3"
    )
    result = ssh_cmd(pod, install_script, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"install on {pod['name']} failed: {result.stderr}")
    print(f"  [{pod['name']}] Ready ({result.stdout.strip().split(chr(10))[-1]})")


async def run_distributed(
    n_miners: int,
    n_windows: int,
    window_duration: float,
    use_wandb: bool,
) -> bool:
    pods = PODS[:n_miners]

    r2_url = _doppler_get("R2_URL")
    r2_key_id = _doppler_get("R2_ACCESS_KEY_ID")
    r2_secret = _doppler_get("R2_SECRET_ACCESS_KEY")
    r2_bucket = _doppler_get("R2_BUCKET_NAME")

    run_id = uuid.uuid4().hex[:8]
    prefix = f"teutonic/distributed/{run_id}/"

    storage = R2Storage(
        endpoint_url=r2_url,
        access_key_id=r2_key_id,
        secret_access_key=r2_secret,
        bucket_name=r2_bucket,
        prefix=prefix,
    )

    train_dataset = PatternDataset(size=4096, seq_len=CFG.seq_len, vocab_size=CFG.vocab_size, seed=TRAIN_SEED)
    eval_dataset = PatternDataset(size=256, seq_len=CFG.seq_len, vocab_size=CFG.vocab_size, seed=EVAL_SEED)

    shared_init = make_model(seed=0).state_dict()

    # -- Provision pods --
    print("=" * 72)
    print(f"Distributed Convergence Test | run_id={run_id}")
    print(f"Pods: {[p['name'] for p in pods]} | {n_windows} windows @ {window_duration}s")
    print("=" * 72)
    print("\nProvisioning pods...")
    for pod in pods:
        provision_pod(pod)

    # -- Publish initial model to R2 --
    print("\nPublishing initial model to R2...")
    await storage.put("model_state/0", {"model_state_dict": shared_init})

    # -- Set up validator --
    val_model = make_model()
    val_model.load_state_dict(copy.deepcopy(shared_init))

    # t0 is in the future: larger model needs more time for initial upload + miner sync
    t0 = time.time() + 30.0

    clock = SyncTimedClock(t0=t0, interval=window_duration)

    validator = Validator(
        uid=0, model=val_model, dataset=train_dataset,
        storage=storage, hparams=HP, clock=clock, device="cpu",
    )

    reporter = None
    if use_wandb:
        wandb_key = _doppler_get("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_key
        from teutonic.metrics import WandbReporter
        reporter = WandbReporter(
            project="teutonic",
            name=f"distributed-{run_id}",
            config={
                "backend": "distributed-r2",
                "n_miners": n_miners,
                "n_windows": n_windows,
                "window_duration": window_duration,
                **dataclasses.asdict(HP),
                **dataclasses.asdict(CFG),
            },
            tags=["convergence", "distributed"],
            group="distributed",
            job_type="eval",
        )

    # -- Start miners on pods --
    print(f"\nStarting miners (t0={t0:.1f}, {t0 - time.time():.0f}s from now)...")
    for i, pod in enumerate(pods):
        uid = i + 1
        # Write a launch script to the pod, then execute it.
        # This avoids nested quoting issues with R2 URLs.
        script_lines = [
            "#!/bin/bash",
            "cd /workspace/teutonic",
            ". .venv/bin/activate",
            f"exec python neurons/run_miner.py \\",
            f"  --uid {uid} --n-windows {n_windows} \\",
            f"  --window-duration {window_duration} --t0 {t0} \\",
            f"  --run-id {run_id} \\",
            f"  --r2-url '{r2_url}' \\",
            f"  --r2-key-id '{r2_key_id}' \\",
            f"  --r2-secret '{r2_secret}' \\",
            f"  --r2-bucket '{r2_bucket}'",
        ]
        script_content = "\n".join(script_lines)
        # Upload launch script
        upload = subprocess.run(
            f"ssh {SSH_OPTS} -p {pod['port']} root@{pod['host']} "
            f"'cat > /tmp/launch_miner_{uid}.sh && chmod +x /tmp/launch_miner_{uid}.sh'",
            shell=True, input=script_content, text=True, timeout=10,
        )
        # Run it backgrounded
        subprocess.run(
            f"ssh {SSH_OPTS} -p {pod['port']} root@{pod['host']} "
            f"'nohup /tmp/launch_miner_{uid}.sh > /tmp/miner_{uid}.log 2>&1 &'",
            shell=True, timeout=10,
        )
        print(f"  [{pod['name']}] Miner UID={uid} started")

    initial_loss = eval_loss(validator.model, eval_dataset)
    initial_ppl = math.exp(initial_loss)

    param_count = sum(p.numel() for p in val_model.parameters())
    print(f"\nModel: {param_count:,} params | Initial loss: {initial_loss:.4f} (ppl={initial_ppl:.2f})")

    # Wait for t0
    wait_s = t0 - time.time()
    if wait_s > 0:
        print(f"Waiting {wait_s:.1f}s for clock t0...")
        await asyncio.sleep(wait_s)

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
            "meta/backend": "distributed-r2",
        }, step=0)

    try:
        for w in range(n_windows):
            t_start = time.monotonic()

            # Wait for window w+1 so window w's deadline has passed
            await clock.wait_for_window(w + 1)

            # Discover and evaluate miners
            uids = list(range(1, n_miners + 1))
            results = await validator.evaluate_window(w, uids)
            await validator.apply_best_gradients(w, results)

            # Publish updated model for next window
            if w + 1 < n_windows:
                await storage.put(
                    f"model_state/{w + 1}",
                    {"model_state_dict": validator.model.state_dict()},
                )

            e_loss = eval_loss(validator.model, eval_dataset)
            e_ppl = math.exp(e_loss) if math.isfinite(e_loss) else float("inf")
            duration = time.monotonic() - t_start

            n_passing = sum(1 for r in results if r.final_score > 0)
            finite_train_losses = []
            for r in results:
                if r.loss_result and r.loss_result.reported:
                    finite_train_losses.extend(
                        l for l in r.loss_result.reported if math.isfinite(l)
                    )
            train_mean = (
                sum(finite_train_losses) / len(finite_train_losses)
                if finite_train_losses else float("nan")
            )

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
                    "meta/backend": "distributed-r2",
                    "meta/duration_s": round(duration, 3),
                }
                for r in results:
                    metrics[f"miner/{r.uid}/final_score"] = r.final_score
                    metrics[f"miner/{r.uid}/loss_score"] = r.loss_score
                reporter.log(metrics, step=w + 1)

    finally:
        if reporter:
            reporter.close()

        print("\nCollecting miner logs & stopping...")
        for i, pod in enumerate(pods):
            uid = i + 1
            log_result = ssh_cmd(pod, f"tail -20 /tmp/miner_{uid}.log 2>/dev/null", timeout=10)
            if log_result.stdout.strip():
                print(f"  [{pod['name']}] Last log lines:")
                for line in log_result.stdout.strip().split("\n")[-5:]:
                    print(f"    {line}")
            ssh_cmd(pod, "pkill -f run_miner.py", timeout=10)
            print(f"  [{pod['name']}] Stopped")

        # Cleanup R2
        print("Cleaning up R2...")
        n_cleaned = await storage.delete_prefix("")
        print(f"  Deleted {n_cleaned} objects")
        await storage.close()

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
        print("PASS: Distributed training is converging")
    print("=" * 72)
    return passed


def main():
    parser = argparse.ArgumentParser(description="Distributed convergence test")
    parser.add_argument("--n-miners", type=int, default=5)
    parser.add_argument("--n-windows", type=int, default=25)
    parser.add_argument("--window-duration", type=float, default=20.0)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO")

    passed = asyncio.run(run_distributed(
        n_miners=args.n_miners,
        n_windows=args.n_windows,
        window_duration=args.window_duration,
        use_wandb=args.wandb,
    ))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
