"""Filesystem-backed storage for local testing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)


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
        t0 = time.monotonic()
        torch.save(data, path)
        logger.debug(
            "storage.local.put",
            key=key, size_bytes=path.stat().st_size,
            duration_s=round(time.monotonic() - t0, 4),
        )

    async def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        t0 = time.monotonic()
        try:
            result = torch.load(path, weights_only=True)
            logger.debug(
                "storage.local.get",
                key=key, size_bytes=path.stat().st_size,
                duration_s=round(time.monotonic() - t0, 4),
            )
            return result
        except Exception:
            logger.warning(
                "storage.local.corrupt",
                key=key, path=str(path),
                size_bytes=path.stat().st_size if path.exists() else 0,
            )
            return None

    async def list_keys(self, prefix: str) -> list[str]:
        """Return all stored keys that start with *prefix*."""
        items = await self.list_keys_with_metadata(prefix)
        return [item["key"] for item in items]

    async def list_keys_with_metadata(self, prefix: str) -> list[dict[str, Any]]:
        """Return dicts with ``key`` and ``last_modified`` (UTC epoch from file mtime)."""
        safe_prefix = prefix.replace("/", "__")
        results: list[dict[str, Any]] = []
        for path in self.root.glob(f"{safe_prefix}*.pt"):
            key = path.stem.replace("__", "/")
            results.append({
                "key": key,
                "last_modified": path.stat().st_mtime,
            })
        return sorted(results, key=lambda x: x["key"])
