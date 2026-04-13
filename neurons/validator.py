"""Validator: pure proof-of-work verification with deadline-based discovery.

After a window ends, the validator lists submissions from storage and
filters by upload timestamp -- only gradients uploaded before the window
deadline are considered.  Submissions are variable-length: each miner
trains as many batches as it can within the window.

Verification is per-batch: since weights are frozen during a window,
each batch's loss and gradient are independently reproducible with a
single forward+backward pass.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import math
import os
import re
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn

from teutonic.compress import TopKCompressor, decompress_and_apply
from teutonic.hparams import HParams
from teutonic.metrics import MetricsReporter, NullReporter
from teutonic.probe_spec import make_probe_spec, select_probe_params
from teutonic.protocols import Dataset, StorageBackend, WindowClock
from teutonic.sampler import MinerSampler
from teutonic.submission import MinerSubmission
from teutonic.verification import (
    GradientConsistencyResult,
    LossVerificationResult,
    ProbeVerificationResult,
    verify_gradient_consistency,
    verify_gradient_probes,
    verify_loss_ledger,
)

logger = structlog.get_logger(__name__)


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
    consistency_threshold: float = 0.80
    consistency_slash: float = 1.00


@dataclass
class MinerEvalResult:
    uid: int
    n_batches_trained: int = 0
    loss_result: LossVerificationResult | None = None
    probe_result: ProbeVerificationResult | None = None
    consistency_result: GradientConsistencyResult | None = None
    loss_score: float = 1.0
    probe_score: float = 1.0
    consistency_score: float = 1.0
    slash_fraction: float = 0.0
    final_score: float = 0.0
    reason: str = ""


# ──────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────

class Validator:
    """Pure PoW validator: loss ledger + per-batch gradient probe verification.

    Uses storage timestamps to determine which submissions arrived before
    the window deadline.  No synchronous waiting for miners.
    """

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
        reporter: MetricsReporter | None = None,
    ):
        self.uid = uid
        self.model = model.to(device)
        self.dataset = dataset
        self.storage = storage
        self.hp = hparams
        self.clock = clock
        self.compressor = TopKCompressor(topk=hparams.topk)
        self.device = device
        self.slash_cfg = slash_config or SlashConfig()
        self.global_step = 0
        self.reporter = reporter or NullReporter()

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
    # Model introspection
    # ------------------------------------------------------------------ #
    def _param_info(self) -> dict[str, int]:
        """Map of param_name -> numel for all trainable parameters."""
        return {name: p.numel() for name, p in self.model.named_parameters()}

    # ------------------------------------------------------------------ #
    # Miner discovery (deadline-filtered)
    # ------------------------------------------------------------------ #
    async def discover_miners(
        self, window: int, deadline: float | None = None
    ) -> list[int]:
        """Find UIDs whose submissions were uploaded before *deadline*."""
        items = await self.storage.list_keys_with_metadata(f"gradient/{window}/")
        uids = []
        for item in items:
            if deadline is not None and item["last_modified"] > deadline:
                logger.info(
                    "validator.discover.late",
                    key=item["key"],
                    last_modified=item["last_modified"],
                    deadline=deadline,
                )
                continue
            match = re.search(r"gradient/\d+/(\d+)$", item["key"])
            if match:
                uids.append(int(match.group(1)))
        return sorted(uids)

    # ------------------------------------------------------------------ #
    # Evaluate a single miner
    # ------------------------------------------------------------------ #
    async def evaluate_miner(
        self, miner_uid: int, window: int, block_hash: str
    ) -> MinerEvalResult:
        t0 = time.monotonic()
        result = MinerEvalResult(uid=miner_uid)

        key = MinerSubmission.make_storage_key(window, miner_uid)
        raw = await self.storage.get(key)
        if raw is None:
            result.slash_fraction = self.slash_cfg.missing_submission_slash
            result.reason = "missing submission"
            logger.warning(
                "validator.miner.missing", miner_uid=miner_uid, window=window,
                slash=result.slash_fraction,
            )
            return result

        try:
            submission = MinerSubmission.from_dict(raw)
        except (KeyError, TypeError) as exc:
            result.slash_fraction = self.slash_cfg.missing_submission_slash
            result.reason = f"corrupt submission: {exc}"
            logger.warning(
                "validator.miner.corrupt", miner_uid=miner_uid, window=window,
                error=str(exc), slash=result.slash_fraction,
            )
            return result

        rejection = self._validate_submission(submission, miner_uid, window)
        if rejection:
            result.slash_fraction = self.slash_cfg.missing_submission_slash
            result.reason = f"invalid submission: {rejection}"
            logger.warning(
                "validator.miner.invalid", miner_uid=miner_uid, window=window,
                rejection=rejection, slash=result.slash_fraction,
            )
            return result

        n_trained = submission.n_batches_trained
        result.n_batches_trained = n_trained

        sampler = MinerSampler(
            self.dataset,
            miner_uid,
            window,
            max_batches=self.hp.max_batches,
            micro_bs=self.hp.micro_bs,
        )

        # Loss spot-checks (forward only, no grad)
        spot_indices = self._pick_spot_check_indices(
            miner_uid, window, n_trained, block_hash
        )
        loss_result = verify_loss_ledger(
            self.model, self.dataset, sampler,
            submission.loss_ledger, spot_indices, device=self.device,
        )
        result.loss_result = loss_result
        result.loss_score = loss_result.score(atol=self.slash_cfg.loss_atol)

        # Probe spec: params from (window, uid), batch indices from block_hash
        param_info = self._param_info()
        probe_spec = make_probe_spec(
            window, miner_uid, n_trained,
            block_hash=block_hash,
            param_info=param_info,
            n_probes=self.hp.n_probes,
            n_probe_params=self.hp.n_probe_params,
            probe_slice_size=self.hp.probe_slice_size,
        )

        # Per-batch probe verification (single fwd+bwd per checked batch)
        probe_result = verify_gradient_probes(
            self.model, self.dataset, sampler,
            submission.grad_probes, probe_spec, device=self.device,
            n_batches_trained=n_trained,
        )
        result.probe_result = probe_result
        result.probe_score = probe_result.mean_similarity

        # Consistency: sum probes arithmetically, compare to compressed grads
        consistency_result = verify_gradient_consistency(
            submission.grad_probes,
            submission.compressed_gradients,
            probe_spec,
            n_trained,
        )
        result.consistency_result = consistency_result
        result.consistency_score = consistency_result.mean_similarity

        result.slash_fraction = self._compute_slash(result)
        result.final_score = self._compute_final_score(result)
        result.reason = self._describe_result(result)

        duration = time.monotonic() - t0

        if result.slash_fraction > 0:
            logger.warning(
                "validator.miner.slashed",
                miner_uid=miner_uid,
                window=window,
                n_batches_trained=n_trained,
                slash_fraction=result.slash_fraction,
                loss_score=round(result.loss_score, 4),
                probe_score=round(result.probe_score, 4),
                final_score=round(result.final_score, 4),
                reason=result.reason,
                duration_s=round(duration, 3),
            )
        else:
            logger.info(
                "validator.miner.evaluated",
                miner_uid=miner_uid,
                window=window,
                n_batches_trained=n_trained,
                loss_score=round(result.loss_score, 4),
                probe_score=round(result.probe_score, 4),
                final_score=round(result.final_score, 4),
                duration_s=round(duration, 3),
            )

        return result

    # ------------------------------------------------------------------ #
    # Evaluate all miners for a window
    # ------------------------------------------------------------------ #
    async def evaluate_window(
        self, window: int, miner_uids: list[int] | None = None
    ) -> list[MinerEvalResult]:
        structlog.contextvars.bind_contextvars(window=window)

        if miner_uids is None:
            deadline = None
            if self.clock is not None:
                deadline = self.clock.window_end_time(window)
            miner_uids = await self.discover_miners(window, deadline=deadline)

        if self.clock is not None:
            block_hash = self.clock.window_block_hash(window)
        else:
            block_hash = hashlib.blake2b(
                f"fallback:{window}".encode(), digest_size=32
            ).hexdigest()

        logger.info(
            "validator.window.start", window=window,
            n_miners=len(miner_uids), miner_uids=miner_uids,
            block_hash=block_hash[:16],
        )
        t0 = time.monotonic()

        results = []
        for uid in miner_uids:
            try:
                r = await asyncio.wait_for(
                    self.evaluate_miner(uid, window, block_hash),
                    timeout=self.hp.eval_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "validator.miner.timeout",
                    miner_uid=uid, window=window,
                    timeout_s=self.hp.eval_timeout,
                )
                r = MinerEvalResult(uid=uid)
                r.slash_fraction = self.slash_cfg.missing_submission_slash
                r.reason = f"evaluation timed out after {self.hp.eval_timeout}s"
            self._record_score(uid, r.final_score)
            results.append(r)

        duration = time.monotonic() - t0
        n_passed = sum(1 for r in results if r.final_score > 0)
        n_slashed = sum(1 for r in results if r.slash_fraction > 0)
        logger.info(
            "validator.window.complete",
            window=window,
            n_miners=len(results),
            n_passed=n_passed,
            n_slashed=n_slashed,
            duration_s=round(duration, 3),
        )

        wb: dict[str, Any] = {
            "validator/window": window,
            "validator/n_miners": len(results),
            "validator/n_passed": n_passed,
            "validator/n_slashed": n_slashed,
            "validator/window_duration_s": round(duration, 3),
        }
        for r in results:
            wb[f"validator/eval/{r.uid}/loss_score"] = round(r.loss_score, 4)
            wb[f"validator/eval/{r.uid}/probe_score"] = round(r.probe_score, 4)
            wb[f"validator/eval/{r.uid}/consistency_score"] = round(r.consistency_score, 4)
            wb[f"validator/eval/{r.uid}/final_score"] = round(r.final_score, 4)
            wb[f"validator/eval/{r.uid}/slash_fraction"] = round(r.slash_fraction, 2)
        for uid_k, ema_v in self.scores.items():
            wb[f"validator/scores/{uid_k}"] = round(ema_v, 4)
        self.reporter.log(wb, step=self.global_step)

        return results

    # ------------------------------------------------------------------ #
    # Apply aggregated gradients
    # ------------------------------------------------------------------ #
    async def apply_best_gradients(
        self, window: int, results: list[MinerEvalResult]
    ) -> None:
        passing = [r for r in results if r.final_score > 0.0]
        if not passing:
            logger.info("validator.gradients.skip", window=window, reason="no passing miners")
            return

        aggregated: dict[str, dict[str, Any]] = {}
        count = 0
        skipped_params = 0

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
                    logger.warning(
                        "validator.gradient.nonfinite",
                        miner_uid=r.uid, param=pname, window=window,
                    )
                    skipped_params += 1
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
        logger.info(
            "validator.gradients.applied",
            window=window,
            n_passing=count,
            n_total=len(results),
            n_params=len(aggregated),
            skipped_params=skipped_params,
            global_step=self.global_step,
        )
        self.reporter.log({
            "validator/n_params_applied": len(aggregated),
            "validator/skipped_params": skipped_params,
            "validator/n_passing_grads": count,
        }, step=self.global_step)

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #
    async def run(self, start_window: int = 0, n_windows: int | None = None) -> None:
        if self.clock is None:
            raise RuntimeError("Cannot run() without a WindowClock")

        structlog.contextvars.bind_contextvars(role="validator", uid=self.uid)
        logger.info("validator.run.start", start_window=start_window, n_windows=n_windows)

        self._stop_requested = False
        loop = asyncio.get_running_loop()
        original_handlers: dict[int, Any] = {}

        def _request_stop(sig: signal.Signals) -> None:
            logger.warning("validator.signal", signal=sig.name)
            self._stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                original_handlers[sig] = loop.add_signal_handler(
                    sig, functools.partial(_request_stop, sig)
                )
            except NotImplementedError:
                pass

        try:
            w = start_window
            count = 0
            while n_windows is None or count < n_windows:
                if self._stop_requested:
                    logger.info("validator.run.stopping", reason="signal")
                    break

                await self.clock.wait_for_window(w + 1)

                if self._stop_requested:
                    break

                results = await self.evaluate_window(w)
                try:
                    await asyncio.wait_for(
                        self.apply_best_gradients(w, results),
                        timeout=self.hp.apply_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "validator.gradients.timeout",
                        window=w, timeout_s=self.hp.apply_timeout,
                    )
                await self.save_state()
                w += 1
                count += 1
        finally:
            logger.info("validator.run.shutdown", global_step=self.global_step)
            self.reporter.close()
            try:
                await self.save_state()
            except Exception:
                logger.exception("validator.state.save_failed")
            if hasattr(self.storage, "close"):
                try:
                    await self.storage.close()
                except Exception:
                    logger.exception("validator.storage.close_failed")

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
        logger.info("validator.state.saved", global_step=self.global_step, n_scores=len(self.scores))

    async def load_state(self) -> bool:
        raw = await self.storage.get(f"validator_state/{self.uid}")
        if raw is None:
            logger.info("validator.state.not_found")
            return False
        self.global_step = raw.get("global_step", 0)
        self.scores = raw.get("scores", {})
        for uid, hist in raw.get("score_history", {}).items():
            uid = int(uid)
            self.score_history[uid] = deque(hist, maxlen=self.hp.score_history_len)
        msd = raw.get("model_state_dict")
        if msd is not None:
            self.model.load_state_dict(msd)
        logger.info(
            "validator.state.loaded",
            global_step=self.global_step,
            n_scores=len(self.scores),
        )
        return True

    # ------------------------------------------------------------------ #
    # Score history & EMA
    # ------------------------------------------------------------------ #
    def _record_score(self, uid: int, score: float) -> None:
        if uid not in self.score_history:
            self.score_history[uid] = deque(maxlen=self.hp.score_history_len)
        self.score_history[uid].append(score)
        ema = self._ema_score(uid)
        self.scores[uid] = ema
        logger.debug(
            "validator.score.updated",
            miner_uid=uid, raw_score=round(score, 4), ema_score=round(ema, 4),
        )

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
    def _validate_submission(
        self, sub: MinerSubmission, miner_uid: int, window: int
    ) -> str | None:
        """Return a rejection reason string, or None if valid."""
        n = sub.n_batches_trained
        if n < 1:
            return "n_batches_trained < 1"
        if n > self.hp.max_batches:
            return f"n_batches_trained {n} > max_batches {self.hp.max_batches}"

        if len(sub.loss_ledger) != n:
            return f"loss_ledger length {len(sub.loss_ledger)} != n_batches_trained {n}"

        nan_count = sum(1 for x in sub.loss_ledger if not math.isfinite(x))
        if nan_count > 0:
            return f"loss_ledger has {nan_count} NaN/Inf entries"

        if len(sub.grad_probes) < n:
            return f"grad_probes has {len(sub.grad_probes)} entries, need {n} (one per batch trained)"

        param_info = self._param_info()
        expected_params = select_probe_params(
            window, miner_uid, param_info,
            self.hp.n_probe_params, self.hp.probe_slice_size,
        )
        expected_names = {pp.param_name for pp in expected_params}
        expected_sizes = {pp.param_name: pp.slice_end - pp.slice_start for pp in expected_params}

        for k, pdict in sub.grad_probes.items():
            if not isinstance(pdict, dict):
                return f"grad_probes[{k}] is not a dict"
            missing = expected_names - set(pdict.keys())
            if missing:
                return f"grad_probes[{k}] missing probed params: {sorted(missing)[:3]}"
            for pname in expected_names:
                probe = pdict.get(pname)
                if probe is None:
                    continue
                if not isinstance(probe, torch.Tensor):
                    return f"grad_probes[{k}][{pname}] is not a tensor"
                if probe.dim() != 1 or probe.numel() != expected_sizes[pname]:
                    return f"grad_probes[{k}][{pname}] shape {probe.shape} != ({expected_sizes[pname]},)"
                if not torch.isfinite(probe).all():
                    return f"grad_probes[{k}][{pname}] contains NaN/Inf"

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
        self, miner_uid: int, window: int, n_total: int, block_hash: str
    ) -> list[int]:
        """Pick loss spot-check indices using the window-terminating block hash."""
        digest = hashlib.blake2b(
            f"losscheck:{window}:{miner_uid}:{block_hash}".encode(),
            digest_size=8,
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

        if result.consistency_result is not None:
            for sim in result.consistency_result.cosine_sims:
                if sim < cfg.consistency_threshold:
                    slash = max(slash, cfg.consistency_slash)

        return slash

    def _compute_final_score(self, result: MinerEvalResult) -> float:
        """PoW score weighted by training volume."""
        if result.slash_fraction >= 1.0:
            return 0.0
        if result.n_batches_trained < self.hp.min_batches:
            return 0.0
        volume = result.n_batches_trained / self.hp.max_batches
        verification = (
            result.loss_score
            * max(0.0, result.probe_score)
            * max(0.0, result.consistency_score)
        )
        penalty = 1.0 - result.slash_fraction
        return volume * verification * penalty

    def _describe_result(self, result: MinerEvalResult) -> str:
        parts = []
        if result.n_batches_trained > 0:
            volume = result.n_batches_trained / self.hp.max_batches
            parts.append(f"batches={result.n_batches_trained} vol={volume:.2f}")
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
        if result.consistency_result and result.consistency_result.cosine_sims:
            parts.append(
                f"consist={result.consistency_score:.3f} "
                f"(min={result.consistency_result.min_similarity:.3f})"
            )
        parts.append(f"slash={result.slash_fraction:.2f}")
        parts.append(f"final={result.final_score:.4f}")
        return " | ".join(parts)
