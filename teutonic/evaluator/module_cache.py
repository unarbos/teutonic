"""Coordinate Transformers custom-code cache access between model workers."""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def transformers_module_cache_lock():
    """Keep cache copies and imports atomic with respect to evaluator processes.

    Transformers uses non-atomic copies for local custom code, and its import
    lock only protects threads in one process. Lock the actual modules cache so
    the parent and spawned workers cannot import a replica's half-written file.
    Do not hold this lock while loading checkpoint weights or scoring.
    """
    from transformers.utils import HF_MODULES_CACHE

    cache_dir = Path(HF_MODULES_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep this file in place: unlinking it could let waiters lock different inodes.
    with (cache_dir / ".teutonic-import.lock").open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
