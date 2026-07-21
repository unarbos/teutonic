#!/usr/bin/env python3
"""Standalone cleanup script for the Teutonic fineweb-edu shard cache.

Removes:
  - Orphaned .tmp partial-download files left by crashed/interrupted downloads
  - Shard files (.npy) older than --max-age-hours

Safety:
  - Files newer than --min-age-hours are never touched (grace window)
  - Open file descriptors (/proc/*/fd) and memory-mapped files (/proc/*/maps) are
    detected, so any shard currently being downloaded or read (np.load mmap_mode="r")
    by the eval server is skipped
  - --dry-run shows what would be deleted without removing anything

Usage:
    python cleanup_shard_cache.py --dry-run
    python cleanup_shard_cache.py
    python cleanup_shard_cache.py --max-age-hours 3
    python cleanup_shard_cache.py --min-age-hours 0.5 --max-age-hours 3
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("cleanup_shard_cache")

DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "TEUTONIC_SHARD_CACHE_DIR",
        os.environ.get("TEUTONIC_PARQUET_CACHE_DIR", "/tmp/teutonic/finewebedu_shards"),
    )
)
DEFAULT_MIN_AGE_H = float(os.environ.get("SHARD_CACHE_MIN_AGE_H", "0.5"))
DEFAULT_MAX_AGE_H = float(os.environ.get("SHARD_CACHE_MAX_AGE_H", "3"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def find_cache_files(cache_dir: Path) -> list[Path]:
    """Return all shard files (.npy and orphaned .tmp) under cache_dir, any depth."""
    files: list[Path] = []
    try:
        for path in cache_dir.rglob("*"):
            if path.is_file() and (path.suffix == ".npy" or path.name.endswith(".npy.tmp")):
                files.append(path)
    except OSError:
        pass
    return files


def _register_path_under_cache(path_str: str, cache_dir: Path, active: set[str]) -> None:
    cache_str = str(cache_dir.resolve())
    if not path_str.startswith(cache_str):
        return
    active.add(path_str)


def detect_active_files(cache_dir: Path) -> set[str]:
    """
    Return resolved paths of files currently in use by any process.

    Checks:
    - /proc/*/fd    — open file descriptors (in-progress downloads write here)
    - /proc/*/maps  — memory-mapped regions (np.load(mmap_mode="r") reads here)
    """
    cache_dir = cache_dir.resolve()
    active: set[str] = set()

    for fd_dir in Path("/proc").glob("*/fd"):
        try:
            for fd_link in fd_dir.iterdir():
                try:
                    target = os.readlink(fd_link)
                    _register_path_under_cache(target, cache_dir, active)
                except OSError:
                    pass
        except (OSError, PermissionError):
            pass

    for maps_file in Path("/proc").glob("*/maps"):
        try:
            with open(maps_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 6:
                        _register_path_under_cache(parts[5], cache_dir, active)
        except (OSError, PermissionError):
            pass

    return active


def fmt_gb(b: int | float) -> str:
    return f"{b / 1e9:.2f} GB"


def fmt_age(age_s: float) -> str:
    if age_s < 3600:
        return f"{age_s / 60:.0f}m"
    if age_s < 86400:
        return f"{age_s / 3600:.1f}h"
    return f"{age_s / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Main cleanup logic
# ---------------------------------------------------------------------------

def run_cleanup(
    cache_dir: Path,
    min_age_s: float,
    max_age_s: float,
    dry_run: bool,
) -> dict:
    if not cache_dir.exists():
        log.info("cache dir does not exist: %s — nothing to do", cache_dir)
        return {"deleted": 0, "freed_bytes": 0}

    now = time.time()

    log.info("scanning /proc for open/mmap'd files under %s …", cache_dir)
    active_files = detect_active_files(cache_dir)
    if active_files:
        log.info("%d file(s) protected (in-use):", len(active_files))
        for f in sorted(active_files):
            log.info("  protected: %s", f)
    else:
        log.info("no protected files detected")

    files = find_cache_files(cache_dir)
    if not files:
        log.info("no shard files found under %s", cache_dir)
        return {"deleted": 0, "freed_bytes": 0}

    log.info("%d shard file(s) found", len(files))

    def file_age(f: Path) -> float:
        try:
            return now - f.stat().st_mtime
        except Exception:
            return 0.0

    def is_protected(f: Path) -> bool:
        return str(f.resolve()) in active_files

    def is_too_young(f: Path) -> bool:
        return file_age(f) < min_age_s

    deleted_count = 0
    freed_bytes = 0

    def delete(f: Path, reason: str) -> None:
        nonlocal deleted_count, freed_bytes
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        freed_bytes += size
        deleted_count += 1
        verb = "[DRY RUN] would delete" if dry_run else "deleting"
        log.info("%s  %s", verb, f)
        log.info("         reason: %s  |  size: %s  |  age: %s", reason, fmt_gb(size), fmt_age(file_age(f)))
        if not dry_run:
            try:
                f.unlink()
            except OSError as exc:
                log.warning("failed to delete %s: %s", f, exc)

    # -----------------------------------------------------------------------
    # Phase 1 — orphaned partial downloads (.tmp files)
    # -----------------------------------------------------------------------
    log.info("─── phase 1: orphaned partial downloads (.tmp) ───")
    n_before = deleted_count
    for f in list(files):
        if f.suffix != ".tmp":
            continue
        if is_protected(f):
            log.debug("skip (in-use): %s", f)
            continue
        if is_too_young(f):
            log.debug("skip (too young, age %s < min %s): %s", fmt_age(file_age(f)), fmt_age(min_age_s), f)
            continue
        delete(f, "orphaned partial download (.tmp)")
        files.remove(f)
    if deleted_count == n_before:
        log.info("no orphaned partial downloads found")

    # -----------------------------------------------------------------------
    # Phase 2 — expired shards (older than max_age_s)
    # -----------------------------------------------------------------------
    log.info("─── phase 2: expired shards (age > %s) ───", fmt_age(max_age_s))
    n_before = deleted_count
    for f in list(files):
        if is_protected(f) or is_too_young(f):
            continue
        if file_age(f) > max_age_s:
            delete(f, f"expired (age {fmt_age(file_age(f))} > max-age {fmt_age(max_age_s)})")
            files.remove(f)
    if deleted_count == n_before:
        log.info("no expired shards found")

    remaining_bytes = sum(f.stat().st_size for f in files if f.exists())
    label = "preview" if dry_run else "done"
    log.info(
        "═══ cleanup %s: %d file(s) %s, %s freed, %s remaining ═══",
        label,
        deleted_count,
        "would be deleted" if dry_run else "deleted",
        fmt_gb(freed_bytes),
        fmt_gb(remaining_bytes),
    )

    return {"deleted": deleted_count, "freed_bytes": freed_bytes, "remaining_bytes": remaining_bytes}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up expired Teutonic fineweb-edu shard cache files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        metavar="DIR",
        help="Shard cache directory to clean",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=DEFAULT_MIN_AGE_H,
        metavar="N",
        help="Never delete anything newer than N hours (safety grace window)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_H,
        metavar="N",
        help="Delete shards older than N hours",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting anything",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging (shows skipped files)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.dry_run:
        log.info("DRY RUN — no files will be deleted")

    run_cleanup(
        cache_dir=args.cache_dir.resolve(),
        min_age_s=args.min_age_hours * 3600,
        max_age_s=args.max_age_hours * 3600,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
