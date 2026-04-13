"""Deterministic batch assignment sampler with micro-batch index resolution.

For each (uid, window) pair, draws ``max_batches * micro_bs`` unique dataset
indices.  The full sequence is deterministic so both miner and validator can
reproduce it.  A miner may train on fewer than ``max_batches`` depending on
how much time remains in the window.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


def window_seed(uid: int, window: int) -> int:
    """Deterministic 32-bit seed for a (uid, window) pair."""
    return (uid * 1_000_003 ^ window) & 0xFFFF_FFFF


@runtime_checkable
class _Indexable(Protocol):
    def __len__(self) -> int: ...


class MinerSampler:
    """Deterministic sampler that assigns micro-batches to a (uid, window).

    Given a dataset of length *N*, for each (uid, window) pair this sampler
    draws ``max_batches * micro_bs`` unique indices, ordered into
    *max_batches* micro-batches of size *micro_bs* each.

    The miner trains on as many as it can before the window deadline.  The
    validator replays only the batches the miner claims to have trained.
    """

    def __init__(
        self,
        dataset: _Indexable,
        uid: int,
        window: int,
        *,
        max_batches: int,
        micro_bs: int,
    ):
        self.dataset_len = len(dataset)
        self.max_batches = max_batches
        self.micro_bs = micro_bs

        self.set_window_uid(uid, window)

    @property
    def total_micro_batches(self) -> int:
        return self.max_batches

    def set_window_uid(self, uid: int, window: int) -> None:
        self.uid = uid
        self.window = window
        self._indices = self._compute_indices()

    def _compute_indices(self) -> np.ndarray:
        wanted = self.max_batches * self.micro_bs
        if wanted > self.dataset_len:
            raise ValueError(
                f"Window needs {wanted} samples but dataset has only {self.dataset_len}"
            )
        rng = np.random.default_rng(window_seed(self.uid, self.window))
        return rng.choice(self.dataset_len, size=wanted, replace=False)

    def get_micro_batch_indices(self, k: int) -> np.ndarray:
        """Return dataset indices for the *k*-th micro-batch."""
        if k < 0 or k >= self.max_batches:
            raise IndexError(f"k={k} out of range [0, {self.max_batches})")
        start = k * self.micro_bs
        return self._indices[start : start + self.micro_bs]

    def all_indices(self) -> np.ndarray:
        return self._indices

    def __iter__(self):
        return iter(self._indices.tolist())

    def __len__(self) -> int:
        return len(self._indices)
