"""Deterministic probe specification generation.

The validator selects WHICH parameters and WHICH batch indices to verify
using entropy that is unavailable to the miner during training:

- **Batch indices** are derived from (window, uid, nonce).  The nonce is
  a validator secret committed before miners train.
- **Parameter selection** is derived from the block hash of the
  window-terminating block, which is only known after the window ends.

Because miners cannot predict either, they must capture probes for ALL
parameters at ALL batch indices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeParam:
    """A single parameter + slice to probe."""

    param_name: str
    slice_start: int
    slice_end: int


@dataclass(frozen=True)
class ProbeSpec:
    """Describes which gradient slices to verify.

    ``params`` lists the randomly selected parameters (and their slice
    ranges) that the validator will check.  ``batch_indices`` lists the
    randomly selected micro-batch indices.
    """

    params: tuple[ProbeParam, ...]
    batch_indices: tuple[int, ...]


def _deterministic_indices(
    window: int,
    uid: int,
    n: int,
    k: int,
    nonce: str = "",
    salt: str = "probe",
) -> tuple[int, ...]:
    """Pick *k* distinct indices from [0, n) using PRNG seeded by hash."""
    payload = f"{salt}:{window}:{uid}:{nonce}".encode()
    digest = hashlib.blake2b(payload, digest_size=32).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=min(k, n), replace=False)
    return tuple(sorted(int(i) for i in chosen))


def _select_params(
    block_hash: str,
    window: int,
    uid: int,
    param_names: list[str],
    k: int,
) -> list[str]:
    """Select *k* random parameter names using the block hash as entropy."""
    payload = f"paramselect:{window}:{uid}:{block_hash}".encode()
    digest = hashlib.blake2b(payload, digest_size=32).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(param_names), size=min(k, len(param_names)), replace=False)
    return [param_names[int(i)] for i in chosen]


def make_probe_spec(
    window: int,
    uid: int,
    n_microbatches: int,
    *,
    nonce: str = "",
    block_hash: str = "",
    param_info: dict[str, int],
    n_probes: int = 3,
    n_probe_params: int = 3,
    probe_slice_size: int = 128,
) -> ProbeSpec:
    """Build a probe spec for the validator.

    ``param_info`` maps parameter names to their total element count.
    The block hash selects which parameters to verify; the nonce selects
    which batch indices.
    """
    batch_indices = _deterministic_indices(
        window, uid, n_microbatches, n_probes, nonce=nonce
    )

    param_names = sorted(param_info.keys())
    selected = _select_params(block_hash, window, uid, param_names, n_probe_params)

    params = tuple(
        ProbeParam(
            param_name=name,
            slice_start=0,
            slice_end=min(probe_slice_size, param_info[name]),
        )
        for name in selected
    )

    return ProbeSpec(params=params, batch_indices=batch_indices)
