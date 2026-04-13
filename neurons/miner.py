"""Miner: trains a model within a window deadline and uploads before it ends.

The miner trains as many batches as it can from a deterministic sequence,
uploads the gradient + proof before the window deadline, and moves on.
Captures gradient probes for ALL parameters at ALL micro-batches (since
the validator uses unpredictable entropy to select which to check).
"""

from __future__ import annotations

import asyncio
import functools
import signal
import time
from typing import Any

import structlog
import torch
import torch.nn as nn

from teutonic.compress import TopKCompressor, compress_model_gradients, decompress_and_apply
from teutonic.hparams import HParams
from teutonic.metrics import MetricsReporter, NullReporter
from teutonic.protocols import Dataset, StorageBackend, WindowClock
from teutonic.sampler import MinerSampler
from teutonic.submission import MinerSubmission

from neurons.trainer import train_window

logger = structlog.get_logger(__name__)


class Miner:
    """A single miner node that trains and submits before the window deadline."""

    def __init__(
        self,
        uid: int,
        model: nn.Module,
        dataset: Dataset,
        storage: StorageBackend,
        hparams: HParams,
        *,
        clock: WindowClock | None = None,
        device: str | torch.device = "cpu",
        reporter: MetricsReporter | None = None,
    ):
        self.uid = uid
        self.model = model
        self.dataset = dataset
        self.storage = storage
        self.hp = hparams
        self.clock = clock
        self.compressor = TopKCompressor(topk=hparams.topk)
        self.device = device
        self.global_step = 0
        self.reporter = reporter or NullReporter()

    async def train_window(self, window: int, deadline: float | None = None) -> MinerSubmission:
        """Train as many batches as possible and upload before *deadline*."""
        structlog.contextvars.bind_contextvars(window=window)
        logger.info("miner.window.start", window=window, deadline=deadline is not None)
        t0 = time.monotonic()

        sampler = MinerSampler(
            self.dataset, self.uid, window,
            max_batches=self.hp.max_batches, micro_bs=self.hp.micro_bs,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr)

        try:
            result = train_window(
                self.model, self.dataset, sampler, optimizer,
                device=self.device,
                max_grad_norm=self.hp.max_grad_norm,
                deadline=deadline,
                upload_budget_s=self.hp.upload_budget_s,
                probe_slice_size=self.hp.probe_slice_size,
            )
        except Exception:
            logger.exception("miner.window.crashed", window=window, global_step=self.global_step)
            raise

        n_batches_trained = result["n_batches_trained"]
        compressed = compress_model_gradients(self.model, self.compressor)

        submission = MinerSubmission(
            uid=self.uid, window=window,
            compressed_gradients=compressed,
            loss_ledger=result["loss_ledger"],
            n_batches_trained=n_batches_trained,
            grad_probes=result["grad_probes"],
        )

        t_upload = time.monotonic()
        await self.storage.put(submission.storage_key(), submission.to_dict())
        upload_dur = time.monotonic() - t_upload

        self.global_step += 1
        duration = time.monotonic() - t0
        logger.info(
            "miner.window.complete",
            window=window,
            global_step=self.global_step,
            n_batches_trained=n_batches_trained,
            max_batches=self.hp.max_batches,
            storage_key=submission.storage_key(),
            compressed_params=len(compressed),
            upload_s=round(upload_dur, 3),
            duration_s=round(duration, 3),
        )

        finite_losses = [l for l in result["loss_ledger"] if isinstance(l, (int, float)) and l == l]
        self.reporter.log({
            "miner/window": window,
            "miner/mean_loss": sum(finite_losses) / len(finite_losses) if finite_losses else None,
            "miner/final_loss": finite_losses[-1] if finite_losses else None,
            "miner/grad_norm": result.get("grad_norm"),
            "miner/nan_count": sum(1 for l in result["loss_ledger"] if l != l),
            "miner/n_batches_trained": n_batches_trained,
            "miner/n_probes": len(result["grad_probes"]),
            "miner/window_duration_s": round(duration, 3),
            "miner/upload_s": round(upload_dur, 3),
        }, step=self.global_step)

        return submission

    async def apply_aggregated_gradient(
        self, aggregated: dict[str, dict[str, Any]],
    ) -> None:
        decompress_and_apply(self.model, aggregated, self.compressor, self.hp.outer_lr)
        logger.info("miner.gradient.applied", n_params=len(aggregated))

    async def run(self, start_window: int = 0, n_windows: int | None = None) -> None:
        """Continuous training loop driven by the WindowClock."""
        if self.clock is None:
            raise RuntimeError("Cannot run() without a WindowClock")

        structlog.contextvars.bind_contextvars(role="miner", uid=self.uid)
        logger.info("miner.run.start", start_window=start_window, n_windows=n_windows)

        self._stop_requested = False
        loop = asyncio.get_running_loop()

        def _request_stop(sig: signal.Signals) -> None:
            logger.warning("miner.signal", signal=sig.name)
            self._stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, functools.partial(_request_stop, sig))
            except NotImplementedError:
                pass

        try:
            w = start_window
            count = 0
            while n_windows is None or count < n_windows:
                if self._stop_requested:
                    logger.info("miner.run.stopping", reason="signal")
                    break

                await self.clock.wait_for_window(w)

                if self._stop_requested:
                    break

                deadline = self.clock.window_end_time(w)
                await self.train_window(w, deadline=deadline)
                w += 1
                count += 1
        finally:
            logger.info("miner.run.shutdown", global_step=self.global_step)
            self.reporter.close()
            if hasattr(self.storage, "close"):
                try:
                    await self.storage.close()
                except Exception:
                    logger.exception("miner.storage.close_failed")
