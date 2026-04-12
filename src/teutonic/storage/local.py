"""Filesystem-backed storage for local testing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class LocalFileStorage:
    """Stores dicts as .pt files in a shared directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        safe = key.replace("/", "__")
        return self.root / f"{safe}.pt"

    async def put(self, key: str, data: dict[str, Any]) -> None:
        path = self._path_for(key)
        torch.save(data, path)

    async def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return torch.load(path, weights_only=True)
        except Exception:
            logger.warning("Corrupt file at %s, skipping", path)
            return None

    async def list_keys(self, prefix: str) -> list[str]:
        """Return all stored keys that start with *prefix*."""
        safe_prefix = prefix.replace("/", "__")
        results = []
        for path in self.root.glob(f"{safe_prefix}*.pt"):
            key = path.stem.replace("__", "/")
            results.append(key)
        return sorted(results)
