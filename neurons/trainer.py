"""Pure accumulation training loop with loss ledger and per-batch gradient probes.

All micro-batches are forward+backward against the frozen start-of-window
model.  A single ``optimizer.step()`` happens at the very end.  This makes
every micro-batch loss and gradient independently reproducible by the
validator replaying against its own copy of the start-of-window weights.

Since weights are frozen, each batch's gradient is independent.  Probes
capture the PER-BATCH gradient (not accumulated) for a small subset of
parameters selected deterministically from (window, uid).
"""

from __future__ import annotations

import math
import time
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from teutonic.probe_spec import ProbeParam
from teutonic.sampler import MinerSampler

logger = structlog.get_logger(__name__)


def train_window(
    model: nn.Module,
    dataset: Any,
    sampler: MinerSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_grad_norm: float = 1.0,
    deadline: float | None = None,
    upload_budget_s: float = 10.0,
    probe_params: tuple[ProbeParam, ...] = (),
) -> dict[str, Any]:
    """Run one window of pure-accumulation training.

    Captures per-batch gradient probes only for the parameters listed
    in *probe_params*.  Gradients accumulate naturally via repeated
    ``loss.backward()`` calls; the per-batch contribution is extracted
    as the delta from the previous accumulation at each probe slice.
    """
    t0 = time.monotonic()
    model.train()
    optimizer.zero_grad()

    loss_ledger: list[float] = []
    grad_probes: dict[int, dict[str, torch.Tensor]] = {}
    nan_count = 0

    max_n = sampler.total_micro_batches
    raw_grads_accumulated = 0

    prev_probe_vals: dict[str, torch.Tensor] = {}

    for k in range(max_n):
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < upload_budget_s:
                logger.info(
                    "trainer.deadline.stop",
                    batch=k, remaining_s=round(remaining, 2),
                    upload_budget_s=upload_budget_s,
                )
                break

        batch_idx = sampler.get_micro_batch_indices(k)
        tokens = torch.stack([dataset[int(i)] for i in batch_idx]).to(device)

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss_val = loss.item()

        if not math.isfinite(loss_val):
            nan_count += 1
            recent = loss_ledger[-3:] if loss_ledger else []
            logger.warning(
                "trainer.batch.nan",
                batch=k,
                recent_losses=recent,
                nan_count=nan_count,
            )
            loss_ledger.append(float("nan"))
            continue

        loss_ledger.append(loss_val)
        logger.debug("trainer.batch", batch=k, loss=round(loss_val, 5))

        loss.backward()
        raw_grads_accumulated += 1

        if probe_params:
            batch_probes: dict[str, torch.Tensor] = {}
            for pp in probe_params:
                p = _get_param_by_name(model, pp.param_name)
                if p.grad is None:
                    continue
                current = p.grad.flatten()[pp.slice_start : pp.slice_end].detach().clone()
                prev = prev_probe_vals.get(pp.param_name)
                if prev is not None:
                    delta = current - prev
                else:
                    delta = current
                prev_probe_vals[pp.param_name] = current
                if torch.isfinite(delta).all():
                    batch_probes[pp.param_name] = delta.cpu()
            if batch_probes:
                grad_probes[k] = batch_probes

    n_batches_trained = len(loss_ledger)

    if raw_grads_accumulated > 1:
        scale = 1.0 / raw_grads_accumulated
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    grad_norm = 0.0
    if max_grad_norm > 0 and raw_grads_accumulated > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()

    if raw_grads_accumulated > 0:
        optimizer.step()

    finite_losses = [l for l in loss_ledger if math.isfinite(l)]
    duration = time.monotonic() - t0
    logger.info(
        "trainer.window.complete",
        n_batches_trained=n_batches_trained,
        max_batches=max_n,
        mean_loss=round(sum(finite_losses) / len(finite_losses), 5) if finite_losses else None,
        final_loss=round(finite_losses[-1], 5) if finite_losses else None,
        nan_count=nan_count,
        grad_norm=round(grad_norm, 6),
        n_probes=len(grad_probes),
        duration_s=round(duration, 3),
    )

    return {
        "loss_ledger": loss_ledger,
        "grad_probes": grad_probes,
        "n_batches_trained": n_batches_trained,
    }


def _get_param_by_name(model: nn.Module, name: str) -> nn.Parameter:
    parts = name.split(".")
    obj: Any = model
    for p in parts:
        obj = getattr(obj, p)
    return obj
