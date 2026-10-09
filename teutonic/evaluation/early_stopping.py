from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .masking import token_weights


@dataclass(frozen=True, slots=True)
class EarlyStoppingPolicy:
    """One-sided challenger-futility policy supplied by the validator."""

    enabled: bool = False
    min_fraction: float = 0.4
    advantage_quantile: float = 0.95
    margin: float = 0.0
    check_interval: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("early stopping enabled must be boolean")
        for name, value in (
            ("min_fraction", self.min_fraction),
            ("advantage_quantile", self.advantage_quantile),
            ("margin", self.margin),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"early stopping {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"early stopping {name} must be finite")
        if not 0 < float(self.min_fraction) <= 1:
            raise ValueError("early stopping min_fraction must be in (0, 1]")
        if not 0 < float(self.advantage_quantile) <= 1:
            raise ValueError("early stopping advantage_quantile must be in (0, 1]")
        if float(self.margin) < 0:
            raise ValueError("early stopping margin must be non-negative")
        if (
            isinstance(self.check_interval, bool)
            or not isinstance(self.check_interval, int)
            or self.check_interval < 1
        ):
            raise ValueError("early stopping check_interval must be a positive integer")

    def request_dict(self) -> dict[str, bool | float | int]:
        return {
            "enabled": self.enabled,
            "min_fraction": float(self.min_fraction),
            "advantage_quantile": float(self.advantage_quantile),
            "margin": float(self.margin),
            "check_interval": self.check_interval,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EarlyStoppingPolicy:
        return cls(
            enabled=value.get("enabled", False),
            min_fraction=value.get("min_fraction", 0.4),
            advantage_quantile=value.get("advantage_quantile", 0.95),
            margin=value.get("margin", 0.0),
            check_interval=value.get("check_interval", 100),
        )


def challenger_futility_decision(
    king_losses: list[float],
    challenger_losses: list[float],
    *,
    total_sequences: int,
    delta_threshold: float,
    policy: EarlyStoppingPolicy,
    scored_tokens: list[int] | None = None,
    total_scored_tokens: int | None = None,
) -> dict[str, float] | None:
    """Return the pre-merge-style one-sided futility decision, if triggered.

    The observed advantage quantile is a tunable heuristic for unseen samples,
    not a mathematical bound. A positive margin makes stopping more conservative.
    """

    if not policy.enabled:
        return None
    if not king_losses or len(king_losses) != len(challenger_losses):
        raise ValueError("early stopping requires non-empty paired losses")
    if total_sequences < len(king_losses):
        raise ValueError("early stopping observed more losses than requested")

    advantages = np.asarray(king_losses, dtype=np.float64) - np.asarray(
        challenger_losses, dtype=np.float64
    )
    weights = (
        token_weights(scored_tokens, len(advantages))
        if scored_tokens is not None else np.ones(len(advantages))
    )
    if scored_tokens is not None:
        if total_scored_tokens is None or total_scored_tokens < weights.sum():
            raise ValueError("early stopping needs the full planned scored-token count")
        total_weight = total_scored_tokens
    else:
        total_weight = total_sequences
    # Keep the observed-window quantile heuristic; project its per-token
    # advantage onto the exact number of unscored targets in the sampled corpus.
    assumed_advantage = float(np.quantile(advantages, policy.advantage_quantile))
    remaining = total_weight - weights.sum()
    projected_upper_mean = float(
        (np.dot(advantages, weights) + remaining * assumed_advantage) / total_weight
    )
    stop_threshold = float(delta_threshold) - float(policy.margin)
    if projected_upper_mean >= stop_threshold:
        return None
    return {
        "mu_hat": float(np.average(advantages, weights=weights)),
        "mu_hat_upper_bound": projected_upper_mean,
        "assumed_remaining_advantage": assumed_advantage,
        "stop_threshold": stop_threshold,
    }
