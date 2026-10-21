"""Coordinate evaluator cache use with cleanup in other processes."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


def document_index_cache_dir(shard_cache_dir: Path) -> Path:
    return Path(os.environ.get("TEUTONIC_DOC_INDEX_CACHE_DIR", shard_cache_dir / "doc_index"))


@contextmanager
def cache_lock(cache_dir: Path, *, cleanup: bool = False):
    """Evaluations share the cache; cleanup skips it when any reader holds it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Never unlink this file: all processes must lock the same inode.
    with (cache_dir / ".teutonic-cache.lock").open("a") as handle:
        mode = fcntl.LOCK_EX | fcntl.LOCK_NB if cleanup else fcntl.LOCK_SH
        try:
            fcntl.flock(handle, mode)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
