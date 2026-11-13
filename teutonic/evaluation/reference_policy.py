"""Byte-weighted, parity-capped reference deficit objective. No retention vetoes."""

import math
from datetime import UTC, datetime

import numpy as np

LEGACY_POLICY_VERSION = "horizon-deficit-bpb"
ACTIVE_POLICY_VERSION = "horizon-deficit-active-bpb"
POLICY_VERSION = ACTIVE_POLICY_VERSION


def reference_verdict(king_nll, challenger_nll, reference_nll, byte_counts, *,
                      bootstrap_seed, n_bootstrap, alpha, delta_threshold, skip_both_zero=True):
    """Exclude both-zero deficits from the reward and bootstrap only.

    False retains the legacy rule for historical offline replays. Full corpus
    loss diagnostics and n_sequences retain their original meaning.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (king_nll, challenger_nll, reference_nll, byte_counts)]
    king, challenger, reference, weights = arrays
    if (king.ndim != 1 or not len(king) or any(a.shape != king.shape for a in arrays)
            or any(not np.isfinite(a).all() for a in arrays)
            or any((a < 0).any() for a in arrays[:3])
            or (weights <= 0).any() or (weights != np.floor(weights)).any()):
        raise ValueError("losses and positive byte counts must be finite aligned vectors")
    if (type(n_bootstrap) is not int or n_bootstrap < 1 or not 0 < alpha < 1
            or not math.isfinite(delta_threshold) or delta_threshold < 0):
        raise ValueError("invalid bootstrap parameters")
    scale = math.log(2) * weights
    k, c, r = king / scale, challenger / scale, reference / scale
    differences = np.maximum(k - r, 0) - np.maximum(c - r, 0)
    included = ((king > reference) | (challenger > reference)
                if skip_both_zero else np.ones(len(k), dtype=bool))
    reward, reward_weights = differences[included], weights[included]
    rng = np.random.default_rng(bootstrap_seed)
    if len(reward):
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            rows = rng.integers(len(reward), size=len(reward))
            boot[i] = np.average(reward[rows], weights=reward_weights[rows])
        lcb = float(np.quantile(boot, alpha))
        mean = float(np.average(reward, weights=reward_weights))
    else:
        # No eligible comparison provides evidence to dethrone the king.
        mean = lcb = 0.0
    accepted = lcb > delta_threshold
    result = {"accepted": accepted, "verdict": "challenger" if accepted else "king",
            "mu_hat": mean, "lcb": lcb, "delta_threshold": delta_threshold,
            "delta": delta_threshold, "alpha": alpha, "n_bootstrap": n_bootstrap,
            "n_sequences": len(k), "scoring_policy": POLICY_VERSION if skip_both_zero else LEGACY_POLICY_VERSION,
            "loss_unit": "bits/byte", "aggregation": "byte-weighted",
            "bootstrap_unit": "whole-document", "retention_gates": False,
            "avg_king_loss": float(np.average(k, weights=weights)),
            "avg_challenger_loss": float(np.average(c, weights=weights)),
            "avg_reference_loss": float(np.average(r, weights=weights)),
            "timestamp": datetime.now(UTC).isoformat()}
    if skip_both_zero:
        result.update(n_scoring_sequences=int(included.sum()),
                      n_skipped_sequences=int((~included).sum()),
                      scoring_bytes=int(reward_weights.sum()),
                      skipped_bytes=int(weights[~included].sum()),
                      sample_filter="exclude-both-zero-deficits")
    return result
