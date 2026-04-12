"""Miner: trains a model for one window and uploads a verified submission.

Captures gradient probes for ALL micro-batches (since the validator's
nonce is unknown at training time, any index could be checked).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import signal
from typing import Any

import torch
import torch.nn as nn

from teutonic.compress import TopKCompressor, compress_model_gradients, decompress_and_apply
from teutonic.hparams import HParams
from teutonic.probe_spec import ProbeSpec
from teutonic.protocols import Dataset, StorageBackend, WindowClock
from teutonic.sampler import MinerSampler
from teutonic.submission import MinerSubmission

from neurons.trainer import train_window

logger = logging.getLogger(__name__)


class Miner:
    """A single miner node that trains and submits."""

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

    async def train_window(self, window: int) -> MinerSubmission:
        """Run one full training window and upload results."""
        sampler = MinerSampler(
            self.dataset, self.uid, window,
            n_batches=self.hp.n_batches, micro_bs=self.hp.micro_bs,
        )

        all_probe_indices = set(range(sampler.total_micro_batches))
        probe_spec = ProbeSpec(
            param_name=self.hp.probe_param_name,
            slice_start=self.hp.probe_slice_start,
            slice_end=self.hp.probe_slice_end,
            batch_indices=tuple(sorted(all_probe_indices)),
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr)

        result = train_window(
            self.model, self.dataset, sampler, optimizer,
            probe_indices=all_probe_indices,
            probe_spec=probe_spec,
            device=self.device,
            max_grad_norm=self.hp.max_grad_norm,
        )

        compressed = compress_model_gradients(self.model, self.compressor)

        submission = MinerSubmission(
            uid=self.uid, window=window,
            compressed_gradients=compressed,
            loss_ledger=result["loss_ledger"],
            grad_probes=result["grad_probes"],
        )

        await self.storage.put(submission.storage_key(), submission.to_dict())
        self.global_step += 1
        return submission

    async def apply_aggregated_gradient(
        self, aggregated: dict[str, dict[str, Any]],
    ) -> None:
        decompress_and_apply(self.model, aggregated, self.compressor, self.hp.outer_lr)

    async def run(self, start_window: int = 0, n_windows: int | None = None) -> None:
        """Continuous training loop driven by the WindowClock.

        Registers SIGINT/SIGTERM handlers to close storage before exiting.
        """
        if self.clock is None:
            raise RuntimeError("Cannot run() without a WindowClock")

        self._stop_requested = False
        loop = asyncio.get_running_loop()

        def _request_stop(sig: signal.Signals) -> None:
            logger.warning("Received %s, finishing current window then shutting down", sig.name)
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
                    logger.info("Stop requested, exiting miner run loop")
                    break

                await self.clock.wait_for_window(w)

                if self._stop_requested:
                    break

                await self.train_window(w)
                w += 1
                count += 1
        finally:
            logger.info("Miner shutting down: closing storage")
            if hasattr(self.storage, "close"):
                try:
                    await self.storage.close()
                except Exception:
                    logger.exception("Failed to close storage during shutdown")
