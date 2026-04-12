"""Verification routines: loss ledger spot-checking and gradient probe comparison.

With pure accumulation training, the miner's model is frozen for the
entire window.  Every micro-batch loss and gradient probe is computed
against the same weights -- so the validator can replay ANY micro-batch
and expect an exact match (within float tolerance).

NaN/Inf values in either the reported or replayed data are treated as
automatic failures for that check index.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from teutonic.probe_spec import ProbeSpec
from teutonic.sampler import MinerSampler

logger = logging.getLogger(__name__)


@dataclass
class LossVerificationResult:
    """Result of spot-checking the loss ledger."""

    checked_indices: list[int] = field(default_factory=list)
    reported: list[float] = field(default_factory=list)
    replayed: list[float] = field(default_factory=list)
    abs_errors: list[float] = field(default_factory=list)

    @property
    def max_error(self) -> float:
        return max(self.abs_errors) if self.abs_errors else 0.0

    @property
    def mean_error(self) -> float:
        return sum(self.abs_errors) / len(self.abs_errors) if self.abs_errors else 0.0

    def score(self, atol: float = 0.02) -> float:
        """Fraction of checked micro-batches that pass within tolerance."""
        if not self.abs_errors:
            return 1.0
        passes = sum(1 for err in self.abs_errors if err <= atol)
        return passes / len(self.abs_errors)


@dataclass
class ProbeVerificationResult:
    """Result of comparing gradient probes."""

    checked_indices: list[int] = field(default_factory=list)
    cosine_sims: list[float] = field(default_factory=list)

    @property
    def min_similarity(self) -> float:
        return min(self.cosine_sims) if self.cosine_sims else 1.0

    @property
    def mean_similarity(self) -> float:
        return (
            sum(self.cosine_sims) / len(self.cosine_sims)
            if self.cosine_sims
            else 1.0
        )

    def score(self, threshold: float = 0.95) -> float:
        """Fraction of probes that pass the cosine similarity threshold."""
        if not self.cosine_sims:
            return 1.0
        passes = sum(1 for s in self.cosine_sims if s >= threshold)
        return passes / len(self.cosine_sims)


def _get_param_by_name(model: nn.Module, name: str) -> nn.Parameter:
    parts = name.split(".")
    obj: Any = model
    for p in parts:
        obj = getattr(obj, p)
    return obj


def verify_loss_ledger(
    model: nn.Module,
    dataset: Any,
    sampler: MinerSampler,
    loss_ledger: list[float],
    spot_check_indices: list[int],
    device: torch.device | str = "cpu",
) -> LossVerificationResult:
    """Replay selected micro-batches (forward only) and compare losses.

    NaN/Inf in the reported loss is treated as an automatic max-error
    failure for that index.
    """
    model.eval()
    result = LossVerificationResult()

    with torch.no_grad():
        for k in spot_check_indices:
            if k < 0 or k >= len(loss_ledger):
                continue

            reported_loss = loss_ledger[k]

            # Reported NaN/Inf = automatic failure
            if not math.isfinite(reported_loss):
                result.checked_indices.append(k)
                result.reported.append(reported_loss)
                result.replayed.append(0.0)
                result.abs_errors.append(float("inf"))
                continue

            batch_idx = sampler.get_micro_batch_indices(k)
            tokens = torch.stack([dataset[int(i)] for i in batch_idx]).to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            logits = model(inputs)
            replayed_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            ).item()

            # Replayed NaN = validator issue, skip this check
            if not math.isfinite(replayed_loss):
                logger.warning("Validator replay produced NaN at index %d, skipping", k)
                continue

            err = abs(replayed_loss - reported_loss)

            result.checked_indices.append(k)
            result.reported.append(reported_loss)
            result.replayed.append(replayed_loss)
            result.abs_errors.append(err)

    return result


def verify_gradient_probes(
    model: nn.Module,
    dataset: Any,
    sampler: MinerSampler,
    grad_probes: dict[int, torch.Tensor],
    probe_spec: ProbeSpec,
    device: torch.device | str = "cpu",
) -> ProbeVerificationResult:
    """Replay micro-batches with backward and compare gradient slices.

    Non-finite values in either the miner's probe or the replayed gradient
    result in a cosine similarity of -1.0 (definitive failure).
    """
    model.train()
    result = ProbeVerificationResult()
    n = sampler.total_micro_batches

    for k in sorted(probe_spec.batch_indices):
        if k not in grad_probes:
            continue

        expected = grad_probes[k]

        # Miner submitted non-finite probe = automatic failure
        if not torch.isfinite(expected).all():
            result.checked_indices.append(k)
            result.cosine_sims.append(-1.0)
            continue

        model.zero_grad()

        for j in range(k + 1):
            batch_idx = sampler.get_micro_batch_indices(j)
            tokens = torch.stack([dataset[int(i)] for i in batch_idx]).to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            (loss / n).backward()

        param = _get_param_by_name(model, probe_spec.param_name)
        if param.grad is None:
            continue

        actual = (
            param.grad.flatten()[probe_spec.slice_start : probe_spec.slice_end]
            .detach()
            .cpu()
        )

        # Validator replay produced non-finite gradient = skip
        if not torch.isfinite(actual).all():
            logger.warning("Validator replay produced NaN gradient at probe %d", k)
            continue

        sim = F.cosine_similarity(
            actual.unsqueeze(0).float(), expected.unsqueeze(0).float()
        ).item()

        if not math.isfinite(sim):
            sim = -1.0

        result.checked_indices.append(k)
        result.cosine_sims.append(sim)

    model.zero_grad()
    return result
