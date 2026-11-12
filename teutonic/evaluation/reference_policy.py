"""Byte-weighted, parity-capped reference deficit objective. No retention vetoes."""

import math
from datetime import UTC, datetime

import numpy as np

POLICY_VERSION = "horizon-deficit-bpb"


def reference_verdict(king_nll, challenger_nll, reference_nll, byte_counts, *,
                      bootstrap_seed, n_bootstrap, alpha, delta_threshold):
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
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        rows = rng.integers(len(k), size=len(k))
        boot[i] = np.average(differences[rows], weights=weights[rows])
    lcb = float(np.quantile(boot, alpha))
    mean = float(np.average(differences, weights=weights))
    accepted = lcb > delta_threshold
    return {"accepted": accepted, "verdict": "challenger" if accepted else "king",
            "mu_hat": mean, "lcb": lcb, "delta_threshold": delta_threshold,
            "delta": delta_threshold, "alpha": alpha, "n_bootstrap": n_bootstrap,
            "n_sequences": len(k), "scoring_policy": POLICY_VERSION,
            "loss_unit": "bits/byte", "aggregation": "byte-weighted",
            "bootstrap_unit": "whole-document", "retention_gates": False,
            "avg_king_loss": float(np.average(k, weights=weights)),
            "avg_challenger_loss": float(np.average(c, weights=weights)),
            "avg_reference_loss": float(np.average(r, weights=weights)),
            "timestamp": datetime.now(UTC).isoformat()}
