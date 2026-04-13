"""Deterministic probe specification generation.

Two kinds of selection, using different entropy sources:

- **Parameter selection** is derived from ``(window, uid)`` -- known to
  the miner at training time.  This is safe because ``backward()``
  computes all parameters regardless; knowing which are checked doesn't
  help a cheater skip work.  The miner only needs to capture probes for
  the selected parameters, keeping overhead small.

- **Batch indices** are derived from the block hash of the window-
  terminating block, which is only available after the window ends.
  The miner cannot predict which batches will be spot-checked and must
  record probes for every batch.
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

    ``params`` lists the deterministically selected parameters (and
    their slice ranges).  ``batch_indices`` lists the randomly selected
    micro-batch indices for spot-checking.
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


def select_probe_params(
    window: int,
    uid: int,
    param_info: dict[str, int],
    n_probe_params: int = 3,
    probe_slice_size: int = 128,
) -> tuple[ProbeParam, ...]:
    """Select which parameters to probe -- deterministic from (window, uid).

    The miner calls this at training time to know which params to capture
    probes for.  The validator calls it at eval time and gets the same result.
    """
    param_names = sorted(param_info.keys())
    payload = f"paramselect:{window}:{uid}".encode()
    digest = hashlib.blake2b(payload, digest_size=32).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(param_names), size=min(n_probe_params, len(param_names)), replace=False)
    selected = [param_names[int(i)] for i in chosen]

    return tuple(
        ProbeParam(
            param_name=name,
            slice_start=0,
            slice_end=min(probe_slice_size, param_info[name]),
        )
        for name in selected
    )


def make_probe_spec(
    window: int,
    uid: int,
    n_microbatches: int,
    *,
    block_hash: str = "",
    param_info: dict[str, int],
    n_probes: int = 3,
    n_probe_params: int = 3,
    probe_slice_size: int = 128,
) -> ProbeSpec:
    """Build a probe spec for the validator.

    Batch indices use the block hash (unpredictable to miners).
    Parameter selection uses (window, uid) (known to miners).
    """
    batch_indices = _deterministic_indices(
        window, uid, n_microbatches, n_probes, nonce=block_hash,
    )

    params = select_probe_params(
        window, uid, param_info, n_probe_params, probe_slice_size,
    )

    return ProbeSpec(params=params, batch_indices=batch_indices)
