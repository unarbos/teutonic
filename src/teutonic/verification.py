"""Verification routines: loss spot-checking and per-batch gradient probes.

With pure accumulation training, the miner's model weights are frozen for
the entire window.  Every micro-batch loss and gradient is computed against
the same weights, making each batch independently verifiable with a single
forward+backward pass -- no need to replay earlier batches.

The validator spot-checks random batches (selected by block hash) and
compares both loss and per-batch gradient slices.

Gradient consistency is checked arithmetically: summing the miner's
uploaded per-batch probes and comparing against the compressed gradient
at the same positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Result of comparing per-batch gradient probes."""

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


@dataclass
class GradientConsistencyResult:
    """Result of cross-checking compressed gradients against summed probes.

    The validator sums the miner's per-batch probes across all batches
    to reconstruct the expected accumulated gradient at the probe
    positions, then compares against the compressed gradient at those
    same positions.  No replay needed -- purely arithmetic.

    When no top-K indices overlap with the probe slice (common at small
    topk), the result is empty and defaults to 1.0 (not checkable, no
    penalty).
    """

    param_names: list[str] = field(default_factory=list)
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


def _get_param_by_name(model: nn.Module, name: str) -> nn.Parameter:
    parts = name.split(".")
    obj: Any = model
    for p in parts:
        obj = getattr(obj, p)
    return obj


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity with zero-vector handling."""
    a_norm = a.float().norm().item()
    b_norm = b.float().norm().item()
    if a_norm < 1e-12 and b_norm < 1e-12:
        return 1.0
    sim = F.cosine_similarity(
        a.unsqueeze(0).float(), b.unsqueeze(0).float()
    ).item()
    return sim if math.isfinite(sim) else -1.0


def verify_loss_ledger(
    model: nn.Module,
    dataset: Any,
    sampler: MinerSampler,
    loss_ledger: list[float],
    spot_check_indices: list[int],
    device: torch.device | str = "cpu",
) -> LossVerificationResult:
    """Replay selected micro-batches (forward only) and compare losses."""
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
    """Replay individual batches and compare per-batch gradient slices.

    Since weights are frozen, each batch is independently verifiable.
    For each batch index in ``probe_spec.batch_indices``, replays JUST
    that single batch (one forward + one backward), then compares the
    per-batch gradient at each probed parameter.
    """
    model.train()
    result = ProbeVerificationResult()

    if n_batches_trained is None:
        n_batches_trained = sampler.total_micro_batches

    for k in sorted(probe_spec.batch_indices):
        if k >= n_batches_trained:
            continue

        model.zero_grad()

        batch_idx = sampler.get_micro_batch_indices(k)
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

            sim = _cosine_sim(actual, expected)

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


def verify_gradient_consistency(
    grad_probes: dict[int, dict[str, torch.Tensor]],
    compressed_gradients: dict[str, dict[str, Any]],
    probe_spec: ProbeSpec,
    n_batches_trained: int,
) -> GradientConsistencyResult:
    """Cross-check compressed gradients against summed per-batch probes.

    For each probed parameter, sums the miner's per-batch gradient probes
    across all batches to get the expected mean gradient at the probe slice.
    Then finds the compressed gradient's top-K indices that fall within
    the probe slice, and compares those values.  No model replay needed.
    """
    result = GradientConsistencyResult()

    for pp in probe_spec.params:
        probe_sum = None
        n_summed = 0
        for k in range(n_batches_trained):
            batch_dict = grad_probes.get(k)
            if batch_dict is None or pp.param_name not in batch_dict:
                continue
            probe_val = batch_dict[pp.param_name].float()
            if probe_sum is None:
                probe_sum = probe_val.clone()
            else:
                probe_sum += probe_val
            n_summed += 1

        if probe_sum is None or n_summed == 0:
            result.param_names.append(pp.param_name)
            result.cosine_sims.append(-1.0)
            continue

        if n_summed > 1:
            probe_sum /= n_summed

        comp = compressed_gradients.get(pp.param_name)
        if comp is None:
            result.param_names.append(pp.param_name)
            result.cosine_sims.append(-1.0)
            continue

        idxs = comp.get("idxs")
        vals = comp.get("vals")
        if idxs is None or vals is None:
            result.param_names.append(pp.param_name)
            result.cosine_sims.append(-1.0)
            continue

        idxs_cpu = idxs.long().cpu()
        vals_cpu = vals.float().cpu()

        mask = (idxs_cpu >= pp.slice_start) & (idxs_cpu < pp.slice_end)
        overlap_count = mask.sum().item()

        if overlap_count == 0:
            logger.debug(
                "verify.consistency.no_overlap",
                param=pp.param_name,
            )
            continue

        overlap_idxs = idxs_cpu[mask] - pp.slice_start
        comp_vals_at_overlap = vals_cpu[mask]
        probe_vals_at_overlap = probe_sum[overlap_idxs]

        sim = _cosine_sim(comp_vals_at_overlap, probe_vals_at_overlap)

        result.param_names.append(pp.param_name)
        result.cosine_sims.append(sim)
        logger.debug(
            "verify.consistency.result",
            param=pp.param_name, overlap=overlap_count,
            cosine_sim=round(sim, 4),
        )

    logger.info(
        "verify.consistency.summary",
        n_params=len(result.param_names),
        mean_sim=round(result.mean_similarity, 4),
        min_sim=round(result.min_similarity, 4),
    )
    return result
