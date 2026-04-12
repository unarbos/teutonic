"""Synthetic token dataset for local testing."""

from __future__ import annotations

import numpy as np
import torch


class SyntheticDataset:
    """Random integer token sequences generated once from a seed.

    Each item is a 1-D int64 tensor of length *seq_len* with values in
    ``[0, vocab_size)``.  The full dataset is materialised in RAM (cheap
    for the small sizes used in testing).
    """

    def __init__(
        self,
        size: int = 4096,
        seq_len: int = 512,
        vocab_size: int = 32000,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self._data = torch.from_numpy(
            rng.integers(0, vocab_size, size=(size, seq_len), dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]
