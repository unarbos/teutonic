"""Protocol interfaces that decouple core logic from infrastructure."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class StorageBackend(Protocol):
    """Where miners put submissions and validators fetch them."""

    async def put(self, key: str, data: dict[str, Any]) -> None: ...
    async def get(self, key: str) -> dict[str, Any] | None: ...
    async def list_keys(self, prefix: str) -> list[str]: ...
    async def list_keys_with_metadata(self, prefix: str) -> list[dict[str, Any]]:
        """Return dicts with at least ``key`` and ``last_modified`` (float, UTC epoch)."""
        ...


@runtime_checkable
class WindowClock(Protocol):
    """Source of truth for the current window number and timing."""

    @property
    def current_window(self) -> int: ...

    @property
    def window_duration(self) -> float:
        """Seconds per window."""
        ...

    def window_end_time(self, window: int) -> float:
        """Monotonic timestamp when *window* ends (= start of window+1)."""
        ...

    def window_block_hash(self, window: int) -> str:
        """Return the block hash of the window-terminating block.

        This hash is only available after the window has ended and serves
        as unpredictable entropy for verification indices.  No party
        (miner or validator) can predict or control it ahead of time.
        """
        ...

    async def wait_for_window(self, target: int) -> None: ...


@runtime_checkable
class Dataset(Protocol):
    """Indexable token dataset returning (input_ids,) tensors."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> torch.Tensor: ...
