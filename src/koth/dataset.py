"""Dataset infrastructure: tokenized shards, random batch sampling.

Adapted from templar's SharedShardedDataset for the KOTH eval pipeline.
The dataset is stored as flat 1D numpy arrays of uint32 token IDs on R2.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EvalDataset(Dataset):
    """Memory-mapped evaluation dataset from pre-tokenized .npy shards.

    Each item is a sequence of `seq_len` token IDs sliced from the flat array.
    """

    def __init__(self, shard_path: str | Path, seq_len: int):
        self.seq_len = seq_len
        shard_path = Path(shard_path)

        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        arr = np.load(str(shard_path), mmap_mode="r", allow_pickle=False)
        if arr.dtype != np.uint32:
            arr = arr.astype(np.uint32, copy=False)
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        self.tokens = torch.from_numpy(arr)
        total_tokens = self.tokens.shape[0]

        # Truncate to a multiple of seq_len
        usable = (total_tokens // seq_len) * seq_len
        self.tokens = self.tokens[:usable]
        self.n_sequences = usable // seq_len

        logger.info(
            "EvalDataset: %s, %d tokens, %d sequences of length %d",
            shard_path.name, usable, self.n_sequences, seq_len,
        )

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.n_sequences})")
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len]


def select_eval_indices(
    n_sequences: int,
    N: int,
    commit_block_hash: str,
    hotkey: str,
) -> list[int]:
    """Deterministically select N eval sequence indices.

    Uses blake2b(commit_block_hash || hotkey) as the random seed so eval
    data is unpredictable until the commit block is finalized.
    """
    seed_material = f"{commit_block_hash}:{hotkey}".encode()
    seed_hash = hashlib.blake2b(seed_material, digest_size=8).digest()
    seed = int.from_bytes(seed_hash, "little")

    rng = np.random.Generator(np.random.PCG64(seed))

    if N >= n_sequences:
        indices = list(range(n_sequences))
        rng.shuffle(indices)
        return indices

    return rng.choice(n_sequences, size=N, replace=False).tolist()


def download_shard_from_r2(
    r2_client,
    shard_key: str,
    local_path: str | Path,
) -> Path:
    """Download a dataset shard from R2 to local disk."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        logger.info("Shard already cached at %s", local_path)
        return local_path

    logger.info("Downloading shard %s -> %s", shard_key, local_path)
    r2_client.download_file(shard_key, str(local_path))
    logger.info("Downloaded shard: %.2f GB", local_path.stat().st_size / 1e9)
    return local_path
