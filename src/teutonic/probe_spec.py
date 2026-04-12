"""Deterministic probe specification generation.

Probe indices are derived from (window, uid, nonce).  The nonce is a
validator secret committed before miners train and revealed at eval time,
preventing miners from predicting which micro-batches will be probed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeSpec:
    """Describes what gradient slices to capture / verify."""

    param_name: str
    slice_start: int
    slice_end: int
    batch_indices: tuple[int, ...]

    @property
    def slice_size(self) -> int:
        return self.slice_end - self.slice_start


def _deterministic_indices(
    window: int,
    uid: int,
    n_microbatches: int,
    k: int,
    nonce: str = "",
    salt: str = "probe",
) -> tuple[int, ...]:
    """Pick *k* distinct micro-batch indices from [0, n_microbatches)."""
    payload = f"{salt}:{window}:{uid}:{nonce}".encode()
    digest = hashlib.blake2b(payload, digest_size=32).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_microbatches, size=min(k, n_microbatches), replace=False)
    return tuple(sorted(int(i) for i in chosen))


def make_probe_spec(
    window: int,
    uid: int,
    n_microbatches: int,
    *,
    nonce: str = "",
    param_name: str = "layers.0.attention.wq.weight",
    slice_start: int = 0,
    slice_end: int = 128,
    n_probes: int = 3,
) -> ProbeSpec:
    """Build a probe spec.

    Without a nonce, the spec is predictable by the miner (legacy mode).
    With a nonce, the miner cannot predict which indices will be checked
    and must capture probes for all micro-batches.
    """
    indices = _deterministic_indices(
        window, uid, n_microbatches, n_probes, nonce=nonce
    )
    return ProbeSpec(
        param_name=param_name,
        slice_start=slice_start,
        slice_end=slice_end,
        batch_indices=indices,
    )
