"""Pure accumulation training loop with loss ledger and gradient probe capture.

All micro-batches are forward+backward against the frozen start-of-window
model.  A single ``optimizer.step()`` happens at the very end.  This makes
every micro-batch loss and gradient probe exactly reproducible by the
validator replaying against its own copy of the start-of-window weights.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from teutonic.probe_spec import ProbeSpec
from teutonic.sampler import MinerSampler

logger = logging.getLogger(__name__)


def _get_param_by_name(model: nn.Module, name: str) -> nn.Parameter:
    parts = name.split(".")
    obj: Any = model
    for p in parts:
        obj = getattr(obj, p)
    return obj


def train_window(
    model: nn.Module,
    dataset: Any,
    sampler: MinerSampler,
    optimizer: torch.optim.Optimizer,
    probe_indices: set[int] | None = None,
    probe_spec: ProbeSpec | None = None,
    device: torch.device | str = "cpu",
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """Run one window of pure-accumulation training.

    All n_batches micro-batches run forward+backward against the same
    frozen weights.  Gradients accumulate.  optimizer.step() is called
    once at the end.

    Batches that produce NaN/Inf loss are skipped (gradient not accumulated,
    loss recorded as NaN in the ledger so the validator can detect it).
    """
    model.train()
    optimizer.zero_grad()

    loss_ledger: list[float] = []
    grad_probes: dict[int, torch.Tensor] = {}

    if probe_indices is None and probe_spec is not None:
        probe_indices = set(probe_spec.batch_indices)
    elif probe_indices is None:
        probe_indices = set()

    n = sampler.total_micro_batches

    for k in range(n):
        batch_idx = sampler.get_micro_batch_indices(k)
        tokens = torch.stack([dataset[int(i)] for i in batch_idx]).to(device)

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss_val = loss.item()

        if not math.isfinite(loss_val):
            logger.warning("NaN/Inf loss at micro-batch %d, skipping backward", k)
            loss_ledger.append(float("nan"))
            continue

        loss_ledger.append(loss_val)

        scaled_loss = loss / n
        scaled_loss.backward()

        if k in probe_indices and probe_spec is not None:
            param = _get_param_by_name(model, probe_spec.param_name)
            if param.grad is not None:
                grad_slice = (
                    param.grad.flatten()[probe_spec.slice_start : probe_spec.slice_end]
                    .detach()
                    .clone()
                    .cpu()
                )
                if torch.isfinite(grad_slice).all():
                    grad_probes[k] = grad_slice
                else:
                    logger.warning("Non-finite gradient probe at micro-batch %d", k)

    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return {
        "loss_ledger": loss_ledger,
        "grad_probes": grad_probes,
    }
