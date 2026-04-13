"""Verification routines: loss ledger spot-checking and gradient probe comparison.

With pure accumulation training, the miner's model is frozen for the
entire window.  Every micro-batch loss and gradient probe is computed
against the same weights -- so the validator can replay ANY micro-batch
and expect an exact match (within float tolerance).

Submissions are variable-length: a miner may have trained fewer than
``max_batches``.  Spot-check indices and probe indices are drawn from
[0, n_batches_trained) rather than [0, max_batches).

NaN/Inf values in either the reported or replayed data are treated as
automatic failures for that check index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from teutonic.compress import TopKCompressor
from teutonic.probe_spec import ProbeSpec
from teutonic.sampler import MinerSampler

logger = structlog.get_logger(__name__)


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
        return min(self.cosine_sims) if self.cosine_sims else 0.0

    @property
    def mean_similarity(self) -> float:
        return (
            sum(self.cosine_sims) / len(self.cosine_sims)
            if self.cosine_sims
            else 0.0
        )

    def score(self, threshold: float = 0.95) -> float:
        """Fraction of probes that pass the cosine similarity threshold."""
        if not self.cosine_sims:
            return 0.0
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

    The loss ledger may be shorter than ``sampler.max_batches`` — only
    indices within ``len(loss_ledger)`` are checked.

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

            if not math.isfinite(reported_loss):
                result.checked_indices.append(k)
                result.reported.append(reported_loss)
                result.replayed.append(0.0)
                result.abs_errors.append(float("inf"))
                logger.info("verify.loss.nan_reported", index=k)
                continue

            batch_idx = sampler.get_micro_batch_indices(k)
            tokens = torch.stack([dataset[int(i)] for i in batch_idx]).to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            logits = model(inputs)
            replayed_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            ).item()

            if not math.isfinite(replayed_loss):
                logger.warning("verify.loss.replay_nan", index=k)
                continue

            err = abs(replayed_loss - reported_loss)

            result.checked_indices.append(k)
            result.reported.append(reported_loss)
            result.replayed.append(replayed_loss)
            result.abs_errors.append(err)

            if err > 0.01:
                logger.info(
                    "verify.loss.mismatch",
                    index=k, reported=round(reported_loss, 5),
                    replayed=round(replayed_loss, 5), error=round(err, 5),
                )

    logger.info(
        "verify.loss.summary",
        n_checked=len(result.checked_indices),
        max_error=round(result.max_error, 5),
        mean_error=round(result.mean_error, 5),
    )
    return result


def verify_gradient_probes(
    model: nn.Module,
    dataset: Any,
    sampler: MinerSampler,
    grad_probes: dict[int, dict[str, torch.Tensor]],
    probe_spec: ProbeSpec,
    device: torch.device | str = "cpu",
    n_batches_trained: int | None = None,
) -> ProbeVerificationResult:
    """Replay micro-batches with backward and compare gradient slices.

    For each batch index in ``probe_spec.batch_indices``, replays batches
    0..k, then checks every parameter listed in ``probe_spec.params``
    against the miner's submitted probes.

    Missing probes for a (batch, param) pair that should exist result in
    a cosine similarity of -1.0 (definitive failure).
    """
    model.train()
    result = ProbeVerificationResult()

    if n_batches_trained is None:
        n_batches_trained = sampler.total_micro_batches

    for k in sorted(probe_spec.batch_indices):
        if k >= n_batches_trained:
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
            loss.backward()

        for pp in probe_spec.params:
            batch_dict = grad_probes.get(k)
            if batch_dict is None or pp.param_name not in batch_dict:
                result.checked_indices.append(k)
                result.cosine_sims.append(-1.0)
                logger.info(
                    "verify.probe.missing",
                    index=k, param=pp.param_name,
                )
                continue

            expected = batch_dict[pp.param_name]

            if not torch.isfinite(expected).all():
                result.checked_indices.append(k)
                result.cosine_sims.append(-1.0)
                logger.info("verify.probe.nonfinite_miner", index=k, param=pp.param_name)
                continue

            param = _get_param_by_name(model, pp.param_name)
            if param.grad is None:
                continue

            actual = (
                param.grad.flatten()[pp.slice_start : pp.slice_end]
                .detach()
                .cpu()
            )

            if not torch.isfinite(actual).all():
                logger.warning("verify.probe.replay_nan", index=k, param=pp.param_name)
                continue

            actual_norm = actual.float().norm().item()
            expected_norm = expected.float().norm().item()
            if actual_norm < 1e-12 and expected_norm < 1e-12:
                sim = 1.0
            else:
                sim = F.cosine_similarity(
                    actual.unsqueeze(0).float(), expected.unsqueeze(0).float()
                ).item()

            if not math.isfinite(sim):
                sim = -1.0

            result.checked_indices.append(k)
            result.cosine_sims.append(sim)
            logger.debug(
                "verify.probe.result",
                index=k, param=pp.param_name, cosine_sim=round(sim, 4),
            )

    model.zero_grad()
    logger.info(
        "verify.probe.summary",
        n_checked=len(result.checked_indices),
        n_params_checked=len(probe_spec.params),
        mean_sim=round(result.mean_similarity, 4),
        min_sim=round(result.min_similarity, 4),
    )
    return result


@dataclass
class GradientConsistencyResult:
    """Result of cross-checking compressed gradients against replayed gradients.

    For each checked parameter, the validator decompresses the miner's
    submitted compressed gradient, extracts the values at the top-K
    indices, and compares them to the validator's own gradient at those
    same indices.  Honest miners produce near-identical values.
    """

    param_names: list[str] = field(default_factory=list)
    cosine_sims: list[float] = field(default_factory=list)

    @property
    def min_similarity(self) -> float:
        return min(self.cosine_sims) if self.cosine_sims else 0.0

    @property
    def mean_similarity(self) -> float:
        return (
            sum(self.cosine_sims) / len(self.cosine_sims)
            if self.cosine_sims
            else 0.0
        )


def verify_gradient_consistency(
    model: nn.Module,
    dataset: Any,
    sampler: MinerSampler,
    compressed_gradients: dict[str, dict[str, Any]],
    n_batches_trained: int,
    max_grad_norm: float,
    device: torch.device | str = "cpu",
) -> GradientConsistencyResult:
    """Replay all batches, then cross-check submitted compressed gradients.

    Mirrors the miner's training pipeline: accumulate gradients over all
    batches, scale by ``1/n``, clip by ``max_grad_norm``.  Then for each
    parameter with a submitted compressed gradient, extract the values
    at the submitted top-K indices and compare to the replayed gradient
    at those same indices via cosine similarity.

    If the miner honestly compressed their real gradients, these values
    match closely.  If they substituted fake compressed gradients after
    passing probe checks, this catches them.
    """
    model.train()
    model.zero_grad()
    result = GradientConsistencyResult()

    raw_grads = 0
    for k in range(n_batches_trained):
        batch_idx = sampler.get_micro_batch_indices(k)
        tokens = torch.stack([dataset[int(i)] for i in batch_idx]).to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            continue
        loss.backward()
        raw_grads += 1

    if raw_grads == 0:
        model.zero_grad()
        return result

    if raw_grads > 1:
        scale = 1.0 / raw_grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    for name, p in model.named_parameters():
        if name not in compressed_gradients:
            continue
        if p.grad is None:
            continue

        comp = compressed_gradients[name]
        idxs = comp.get("idxs")
        submitted_vals = comp.get("vals")
        if idxs is None or submitted_vals is None:
            continue

        idxs = idxs.long().cpu()
        submitted_vals = submitted_vals.float().cpu()

        if idxs.numel() == 0:
            continue

        expected_grad = p.grad.flatten().detach().cpu()
        if idxs.max() >= expected_grad.numel():
            result.param_names.append(name)
            result.cosine_sims.append(-1.0)
            logger.warning("verify.consistency.oob", param=name)
            continue

        expected_vals = expected_grad[idxs].float()

        sub_norm = submitted_vals.norm().item()
        exp_norm = expected_vals.norm().item()
        if sub_norm < 1e-12 and exp_norm < 1e-12:
            sim = 1.0
        else:
            sim = F.cosine_similarity(
                submitted_vals.unsqueeze(0), expected_vals.unsqueeze(0)
            ).item()

        if not math.isfinite(sim):
            sim = -1.0

        result.param_names.append(name)
        result.cosine_sims.append(sim)
        logger.debug(
            "verify.consistency.result",
            param=name, cosine_sim=round(sim, 4),
        )

    model.zero_grad()
    logger.info(
        "verify.consistency.summary",
        n_params=len(result.param_names),
        mean_sim=round(result.mean_similarity, 4),
        min_sim=round(result.min_similarity, 4),
    )
    return result
