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


@runtime_checkable
class WindowClock(Protocol):
    """Source of truth for the current window number."""

    @property
    def current_window(self) -> int: ...

    async def wait_for_window(self, target: int) -> None: ...


@runtime_checkable
class Dataset(Protocol):
    """Indexable token dataset returning (input_ids,) tensors."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> torch.Tensor: ...
