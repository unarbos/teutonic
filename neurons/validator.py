"""Validator: pure proof-of-work verification.

Fetches miner submissions, verifies loss ledgers and gradient probes,
scores miners, and applies passing gradients.  No gradient quality
measurement -- if the miner proved it did real computation, the
gradient is accepted.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import logging
import math
import os
import re
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from teutonic.compress import TopKCompressor, decompress_and_apply
from teutonic.hparams import HParams
from teutonic.probe_spec import make_probe_spec
from teutonic.protocols import Dataset, StorageBackend, WindowClock
from teutonic.sampler import MinerSampler
from teutonic.submission import MinerSubmission
from teutonic.verification import (
    LossVerificationResult,
    ProbeVerificationResult,
    verify_gradient_probes,
    verify_loss_ledger,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Scoring & slash configuration
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SlashConfig:
    loss_slash: float = 0.90
    loss_atol: float = 0.05
    probe_hard_slash: float = 1.00
    probe_soft_slash: float = 0.50
    missing_submission_slash: float = 0.75
    probe_hard_threshold: float = 0.80
    probe_soft_threshold: float = 0.95


@dataclass
class MinerEvalResult:
    uid: int
    loss_result: LossVerificationResult | None = None
    probe_result: ProbeVerificationResult | None = None
    loss_score: float = 1.0
    probe_score: float = 1.0
    slash_fraction: float = 0.0
    final_score: float = 0.0
    reason: str = ""


# ──────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────

class Validator:
    """Pure PoW validator: loss ledger + gradient probe verification."""

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
        slash_config: SlashConfig | None = None,
    ):
        self.uid = uid
        self.model = model
        self.dataset = dataset
        self.storage = storage
        self.hp = hparams
        self.clock = clock
        self.compressor = TopKCompressor(topk=hparams.topk)
        self.device = device
        self.slash_cfg = slash_config or SlashConfig()
        self.global_step = 0

        self.scores: dict[int, float] = {}
        self.score_history: dict[int, deque[float]] = {}

    # ------------------------------------------------------------------ #
    # Nonce management
    # ------------------------------------------------------------------ #
    def _generate_nonce(self) -> str:
        return os.urandom(16).hex()

    async def _commit_nonce(self, window: int, nonce: str) -> None:
        """Commit the hash of the nonce before miners train."""
        h = hashlib.blake2b(nonce.encode(), digest_size=16).hexdigest()
        await self.storage.put(
            f"nonce_commit/{window}/{self.uid}",
            {"hash": h, "window": window},
        )

    # ------------------------------------------------------------------ #
    # Miner discovery
    # ------------------------------------------------------------------ #
    async def discover_miners(self, window: int) -> list[int]:
        """Find all UIDs that submitted for a given window."""
        keys = await self.storage.list_keys(f"gradient/{window}/")
        uids = []
        for k in keys:
            match = re.search(r"gradient/\d+/(\d+)$", k)
            if match:
                uids.append(int(match.group(1)))
        return sorted(uids)

    # ------------------------------------------------------------------ #
    # Evaluate a single miner
    # ------------------------------------------------------------------ #
    async def evaluate_miner(
        self, miner_uid: int, window: int, nonce: str
    ) -> MinerEvalResult:
        result = MinerEvalResult(uid=miner_uid)

        key = MinerSubmission.make_storage_key(window, miner_uid)
        raw = await self.storage.get(key)
        if raw is None:
            result.slash_fraction = self.slash_cfg.missing_submission_slash
            result.reason = "missing submission"
            return result

        try:
            submission = MinerSubmission.from_dict(raw)
        except (KeyError, TypeError) as exc:
            result.slash_fraction = self.slash_cfg.missing_submission_slash
            result.reason = f"corrupt submission: {exc}"
            return result

        # Validate submission data integrity
        rejection = self._validate_submission(submission)
        if rejection:
            result.slash_fraction = self.slash_cfg.missing_submission_slash
            result.reason = f"invalid submission: {rejection}"
            return result

        sampler = MinerSampler(
            self.dataset,
            miner_uid,
            window,
            n_batches=self.hp.n_batches,
            micro_bs=self.hp.micro_bs,
        )

        # 1. Loss ledger spot-check
        spot_indices = self._pick_spot_check_indices(
            miner_uid, window, sampler.total_micro_batches
        )
        loss_result = verify_loss_ledger(
            self.model, self.dataset, sampler,
            submission.loss_ledger, spot_indices, device=self.device,
        )
        result.loss_result = loss_result
        result.loss_score = loss_result.score(atol=self.slash_cfg.loss_atol)

        # 2. Gradient probe verification (uses nonce for unpredictable indices)
        probe_spec = make_probe_spec(
            window, miner_uid, sampler.total_micro_batches,
            nonce=nonce,
            param_name=self.hp.probe_param_name,
            slice_start=self.hp.probe_slice_start,
            slice_end=self.hp.probe_slice_end,
            n_probes=self.hp.n_probes,
        )

        saved_state = copy.deepcopy(self.model.state_dict())
        probe_result = verify_gradient_probes(
            self.model, self.dataset, sampler,
            submission.grad_probes, probe_spec, device=self.device,
        )
        self.model.load_state_dict(saved_state)
        result.probe_result = probe_result
        result.probe_score = probe_result.mean_similarity

        # 3. Compute slash and final score (pure PoW -- no gradient quality)
        result.slash_fraction = self._compute_slash(result)
        result.final_score = self._compute_final_score(result)
        result.reason = self._describe_result(result)

        return result

    # ------------------------------------------------------------------ #
    # Evaluate all miners for a window
    # ------------------------------------------------------------------ #
    async def evaluate_window(
        self, window: int, miner_uids: list[int] | None = None
    ) -> list[MinerEvalResult]:
        nonce = self._generate_nonce()
        await self._commit_nonce(window, nonce)

        if miner_uids is None:
            miner_uids = await self.discover_miners(window)

        results = []
        for uid in miner_uids:
            try:
                r = await asyncio.wait_for(
                    self.evaluate_miner(uid, window, nonce),
                    timeout=self.hp.eval_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("evaluate_miner timed out for UID %d window %d", uid, window)
                r = MinerEvalResult(uid=uid)
                r.slash_fraction = self.slash_cfg.missing_submission_slash
                r.reason = f"evaluation timed out after {self.hp.eval_timeout}s"
            self._record_score(uid, r.final_score)
            results.append(r)
        return results

    # ------------------------------------------------------------------ #
    # Apply aggregated gradients
    # ------------------------------------------------------------------ #
    async def apply_best_gradients(
        self, window: int, results: list[MinerEvalResult]
    ) -> None:
        passing = [r for r in results if r.final_score > 0.0]
        if not passing:
            return

        aggregated: dict[str, dict[str, Any]] = {}
        count = 0

        for r in passing:
            key = MinerSubmission.make_storage_key(window, r.uid)
            raw = await self.storage.get(key)
            if raw is None:
                continue
            try:
                sub = MinerSubmission.from_dict(raw)
            except (KeyError, TypeError):
                continue
            for pname, comp in sub.compressed_gradients.items():
                vals = comp.get("vals")
                if vals is None or (isinstance(vals, torch.Tensor) and not torch.isfinite(vals).all()):
                    logger.warning("Skipping NaN/Inf vals from UID %d param %s", r.uid, pname)
                    continue
                if pname not in aggregated:
                    aggregated[pname] = {
                        "idxs": comp["idxs"],
                        "vals": comp["vals"].float().clone(),
                        "shape": comp["shape"],
                    }
                else:
                    dense_new = self.compressor.decompress(comp, device="cpu")
                    dense_old = self.compressor.decompress(aggregated[pname], device="cpu")
                    merged = dense_old + dense_new
                    aggregated[pname] = self.compressor.compress(merged)
                    aggregated[pname]["vals"] = aggregated[pname]["vals"].float()
            count += 1

        if count > 1:
            for pname in aggregated:
                aggregated[pname]["vals"] /= count

        decompress_and_apply(self.model, aggregated, self.compressor, self.hp.outer_lr)
        self.global_step += 1

    # ------------------------------------------------------------------ #
    # Main run loop (uses WindowClock)
    # ------------------------------------------------------------------ #
    async def run(self, start_window: int = 0, n_windows: int | None = None) -> None:
        """Continuous evaluation loop driven by the WindowClock.

        Registers SIGINT/SIGTERM handlers to save state and close storage
        before exiting.  If *n_windows* is None, runs indefinitely.
        """
        if self.clock is None:
            raise RuntimeError("Cannot run() without a WindowClock")

        self._stop_requested = False
        loop = asyncio.get_running_loop()
        original_handlers: dict[int, Any] = {}

        def _request_stop(sig: signal.Signals) -> None:
            logger.warning("Received %s, finishing current window then shutting down", sig.name)
            self._stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                original_handlers[sig] = loop.add_signal_handler(
                    sig, functools.partial(_request_stop, sig)
                )
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        try:
            w = start_window
            count = 0
            while n_windows is None or count < n_windows:
                if self._stop_requested:
                    logger.info("Stop requested, exiting run loop")
                    break

                await self.clock.wait_for_window(w)

                if self._stop_requested:
                    break

                results = await self.evaluate_window(w)
                try:
                    await asyncio.wait_for(
                        self.apply_best_gradients(w, results),
                        timeout=self.hp.apply_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error("apply_best_gradients timed out for window %d", w)
                await self.save_state()
                w += 1
                count += 1
        finally:
            logger.info("Shutting down: saving state and closing storage")
            try:
                await self.save_state()
            except Exception:
                logger.exception("Failed to save state during shutdown")
            if hasattr(self.storage, "close"):
                try:
                    await self.storage.close()
                except Exception:
                    logger.exception("Failed to close storage during shutdown")

            for sig in original_handlers:
                try:
                    loop.remove_signal_handler(sig)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # State persistence
    # ------------------------------------------------------------------ #
    async def save_state(self) -> None:
        state = {
            "global_step": self.global_step,
            "scores": dict(self.scores),
            "score_history": {
                uid: list(dq) for uid, dq in self.score_history.items()
            },
            "model_state_dict": self.model.state_dict(),
        }
        await self.storage.put(f"validator_state/{self.uid}", state)

    async def load_state(self) -> bool:
        raw = await self.storage.get(f"validator_state/{self.uid}")
        if raw is None:
            return False
        self.global_step = raw.get("global_step", 0)
        self.scores = raw.get("scores", {})
        for uid, hist in raw.get("score_history", {}).items():
            uid = int(uid)
            self.score_history[uid] = deque(hist, maxlen=self.hp.score_history_len)
        msd = raw.get("model_state_dict")
        if msd is not None:
            self.model.load_state_dict(msd)
        return True

    # ------------------------------------------------------------------ #
    # Score history & EMA
    # ------------------------------------------------------------------ #
    def _record_score(self, uid: int, score: float) -> None:
        if uid not in self.score_history:
            self.score_history[uid] = deque(maxlen=self.hp.score_history_len)
        self.score_history[uid].append(score)
        self.scores[uid] = self._ema_score(uid)

    def _ema_score(self, uid: int) -> float:
        hist = self.score_history.get(uid)
        if not hist:
            return 0.0
        alpha = self.hp.score_ema_alpha
        ema = hist[0]
        for s in list(hist)[1:]:
            ema = alpha * s + (1 - alpha) * ema
        return ema

    def get_effective_score(self, uid: int) -> float:
        return self.scores.get(uid, 0.0)

    # ------------------------------------------------------------------ #
    # Submission validation
    # ------------------------------------------------------------------ #
    def _validate_submission(self, sub: MinerSubmission) -> str | None:
        """Return a rejection reason string, or None if valid."""
        # Loss ledger: must have correct length and all finite
        if len(sub.loss_ledger) != self.hp.n_batches:
            return f"loss_ledger length {len(sub.loss_ledger)} != {self.hp.n_batches}"

        nan_count = sum(1 for x in sub.loss_ledger if not math.isfinite(x))
        if nan_count > 0:
            return f"loss_ledger has {nan_count} NaN/Inf entries"

        # Gradient probes: values must be finite tensors of expected size
        expected_probe_size = self.hp.probe_slice_end - self.hp.probe_slice_start
        for k, probe in sub.grad_probes.items():
            if not isinstance(probe, torch.Tensor):
                return f"grad_probe[{k}] is not a tensor"
            if probe.dim() != 1 or probe.numel() != expected_probe_size:
                return f"grad_probe[{k}] shape {probe.shape} != ({expected_probe_size},)"
            if not torch.isfinite(probe).all():
                return f"grad_probe[{k}] contains NaN/Inf"

        # Compressed gradients: vals must be finite, idxs must be integer
        for pname, comp in sub.compressed_gradients.items():
            if "vals" not in comp or "idxs" not in comp or "shape" not in comp:
                return f"compressed_gradients[{pname}] missing keys"
            vals = comp["vals"]
            if isinstance(vals, torch.Tensor) and not torch.isfinite(vals).all():
                return f"compressed_gradients[{pname}].vals contains NaN/Inf"

        return None

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _pick_spot_check_indices(
        self, miner_uid: int, window: int, n_total: int
    ) -> list[int]:
        digest = hashlib.blake2b(
            f"losscheck:{window}:{miner_uid}".encode(), digest_size=8
        ).digest()
        seed = int.from_bytes(digest, "little")
        rng = np.random.default_rng(seed)
        k = min(self.hp.n_loss_spot_checks, n_total)
        return sorted(int(i) for i in rng.choice(n_total, size=k, replace=False))

    def _compute_slash(self, result: MinerEvalResult) -> float:
        cfg = self.slash_cfg
        slash = 0.0

        if result.loss_result is not None:
            for err in result.loss_result.abs_errors:
                if err > cfg.loss_atol:
                    slash = max(slash, cfg.loss_slash)

        if result.probe_result is not None:
            for sim in result.probe_result.cosine_sims:
                if sim < cfg.probe_hard_threshold:
                    slash = max(slash, cfg.probe_hard_slash)
                elif sim < cfg.probe_soft_threshold:
                    slash = max(slash, cfg.probe_soft_slash)

        return slash

    def _compute_final_score(self, result: MinerEvalResult) -> float:
        """Pure PoW score: did the miner do the work?"""
        if result.slash_fraction >= 1.0:
            return 0.0
        verification = result.loss_score * max(0.0, result.probe_score)
        penalty = 1.0 - result.slash_fraction
        return verification * penalty

    def _describe_result(self, result: MinerEvalResult) -> str:
        parts = []
        if result.loss_result:
            parts.append(
                f"loss={result.loss_score:.2f} "
                f"(max_err={result.loss_result.max_error:.4f})"
            )
        if result.probe_result and result.probe_result.cosine_sims:
            parts.append(
                f"probe={result.probe_score:.3f} "
                f"(min={result.probe_result.min_similarity:.3f})"
            )
        parts.append(f"slash={result.slash_fraction:.2f}")
        parts.append(f"final={result.final_score:.4f}")
        return " | ".join(parts)
