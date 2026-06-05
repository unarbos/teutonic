#!/usr/bin/env python3
"""Standalone cleanup script for the Teutonic Quasar pair-eval model cache.

Removes:
  - Incomplete (partial) snapshot dirs left by failed/crashed downloads
  - Snapshot dirs older than --max-age-days that are not actively in use
  - Excess snapshots beyond --keep-recent per model when total cache exceeds --watermark-gb

Safety:
  - Directories newer than --min-age-hours are never touched (grace window)
  - Open file descriptors (/proc/*/fd) and memory-mapped files (/proc/*/maps) are
    detected, so any snapshot currently loaded by the eval server is skipped
  - --dry-run shows what would be deleted without removing anything

Usage:
    python cleanup_model_cache.py --dry-run
    python cleanup_model_cache.py
    python cleanup_model_cache.py --min-age-hours 2 --max-age-days 3 --keep-recent 2
    python cleanup_model_cache.py --watermark-gb 200 --keep-recent 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

log = logging.getLogger("cleanup_model_cache")

DEFAULT_CACHE_DIR = Path(os.environ.get("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/quasar_pair_models"))
DEFAULT_MIN_AGE_H = float(os.environ.get("MODEL_CACHE_MIN_AGE_H", "1"))
DEFAULT_MAX_AGE_DAYS = float(os.environ.get("MODEL_CACHE_MAX_AGE_DAYS", str(3 / 24)))
DEFAULT_KEEP_RECENT = int(os.environ.get("MODEL_CACHE_KEEP_RECENT", "1"))
DEFAULT_WATERMARK_GB = float(os.environ.get("MODEL_CACHE_HIGH_WATERMARK_GB", "500"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def dir_size_bytes(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0


def snapshot_is_complete(path: Path) -> bool:
    """True if the directory looks like a complete model snapshot."""
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors"))


def find_snapshot_dirs(cache_dir: Path) -> list[Path]:
    """Return all snapshot dirs (two levels deep: <model-slug>/<digest>)."""
    dirs: list[Path] = []
    try:
        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for snap_dir in model_dir.iterdir():
                if snap_dir.is_dir():
                    dirs.append(snap_dir)
    except OSError:
        pass
    return dirs


def _register_path_under_cache(path_str: str, cache_dir: Path, active: set[str]) -> None:
    """If path_str is inside cache_dir, add its snapshot-level parent to active."""
    cache_str = str(cache_dir.resolve())
    if not path_str.startswith(cache_str):
        return
    try:
        rel = Path(path_str).relative_to(cache_dir)
        parts = rel.parts
        if len(parts) >= 2:
            snap = cache_dir / parts[0] / parts[1]
            active.add(str(snap.resolve()))
    except ValueError:
        pass


def detect_active_snapshot_dirs(cache_dir: Path) -> set[str]:
    """
    Return resolved paths of snapshot dirs currently in use by any process.

    Checks:
    - /proc/*/fd    — open file descriptors (config.json, tokenizer files, etc.)
    - /proc/*/maps  — memory-mapped regions (safetensors weights loaded with mmap)
    """
    cache_dir = cache_dir.resolve()
    active: set[str] = set()

    # Open file descriptors
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

    # Memory-mapped files (safetensors are loaded with mmap by default)
    for maps_file in Path("/proc").glob("*/maps"):
        try:
            with open(maps_file) as f:
                for line in f:
                    # format: addr perms offset dev inode /path
                    parts = line.split()
                    if len(parts) >= 6:
                        _register_path_under_cache(parts[5], cache_dir, active)
        except (OSError, PermissionError):
            pass

    return active


def read_king_ref(cache_dir: Path) -> str | None:
    """Read the snapshot path of the currently loaded king model written by the eval server."""
    try:
        path = (cache_dir / ".current_king").read_text().strip()
        return path or None
    except OSError:
        return None


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
    keep_recent: int,
    watermark_bytes: float,
    dry_run: bool,
) -> dict:
    if not cache_dir.exists():
        log.info("cache dir does not exist: %s — nothing to do", cache_dir)
        return {"deleted": 0, "freed_bytes": 0}

    now = time.time()

    # -----------------------------------------------------------------------
    # Detect dirs currently in use by the eval server process
    # -----------------------------------------------------------------------
    log.info("scanning /proc for open/mmap'd files under %s …", cache_dir)
    active_dirs = detect_active_snapshot_dirs(cache_dir)

    king_ref = read_king_ref(cache_dir)
    if king_ref:
        king_resolved = str(Path(king_ref).resolve())
        active_dirs.add(king_resolved)
        log.info("current king (from .current_king): %s", king_ref)
    else:
        log.info("no .current_king ref found (eval server not running or king not yet loaded)")

    if active_dirs:
        log.info("%d snapshot dir(s) protected (in-use or current king):", len(active_dirs))
        for d in sorted(active_dirs):
            log.info("  protected: %s", d)
    else:
        log.info("no protected snapshot dirs detected")

    snapshots = find_snapshot_dirs(cache_dir)
    if not snapshots:
        log.info("no snapshot dirs found under %s", cache_dir)
        return {"deleted": 0, "freed_bytes": 0}

    log.info("%d snapshot dir(s) found", len(snapshots))

    # Helpers
    def snap_age(d: Path) -> float:
        try:
            return now - d.stat().st_mtime
        except Exception:
            return 0.0

    def is_protected(d: Path) -> bool:
        return str(d.resolve()) in active_dirs

    def is_too_young(d: Path) -> bool:
        return snap_age(d) < min_age_s

    deleted_count = 0
    freed_bytes = 0
    remaining: list[Path] = list(snapshots)

    def delete(d: Path, reason: str) -> None:
        nonlocal deleted_count, freed_bytes
        size = dir_size_bytes(d)
        freed_bytes += size
        deleted_count += 1
        verb = "[DRY RUN] would delete" if dry_run else "deleting"
        log.info("%s  %s", verb, d)
        log.info("         reason: %s  |  size: %s  |  age: %s", reason, fmt_gb(size), fmt_age(snap_age(d)))
        if not dry_run:
            shutil.rmtree(d, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Phase 1 — incomplete snapshots (partial / crashed downloads)
    # -----------------------------------------------------------------------
    log.info("─── phase 1: incomplete snapshots ───")
    n_before = deleted_count
    for d in list(remaining):
        if is_protected(d):
            log.debug("skip (in-use): %s", d)
            continue
        if is_too_young(d):
            log.debug("skip (too young, age %s < min %s): %s", fmt_age(snap_age(d)), fmt_age(min_age_s), d)
            continue
        if not snapshot_is_complete(d):
            delete(d, "incomplete snapshot (missing config.json or .safetensors)")
            remaining.remove(d)
    if deleted_count == n_before:
        log.info("no incomplete snapshots found")

    # -----------------------------------------------------------------------
    # Phase 2 — expired snapshots (older than max_age_s)
    # -----------------------------------------------------------------------
    log.info("─── phase 2: expired snapshots (age > %s) ───", fmt_age(max_age_s))
    n_before = deleted_count
    for d in list(remaining):
        if is_protected(d) or is_too_young(d):
            continue
        if snap_age(d) > max_age_s:
            delete(d, f"expired (age {fmt_age(snap_age(d))} > max-age {fmt_age(max_age_s)})")
            remaining.remove(d)
    if deleted_count == n_before:
        log.info("no expired snapshots found")

    # -----------------------------------------------------------------------
    # Phase 3 — keep only N most recent snapshots per model
    # -----------------------------------------------------------------------
    log.info("─── phase 3: keep %d most-recent snapshot(s) per model ───", keep_recent)
    n_before = deleted_count
    by_model: dict[str, list[Path]] = {}
    for d in remaining:
        if is_protected(d) or is_too_young(d):
            continue
        by_model.setdefault(d.parent.name, []).append(d)

    for model_slug, dirs in by_model.items():
        if len(dirs) <= keep_recent:
            continue
        # Sort youngest-first; keep the first keep_recent, evict the rest.
        sorted_dirs = sorted(dirs, key=snap_age)  # ascending age = youngest first
        to_evict = sorted_dirs[keep_recent:]
        for d in to_evict:
            delete(d, f"excess snapshot (keeping {keep_recent} most-recent for {model_slug})")
            remaining.remove(d)

    if deleted_count == n_before:
        log.info("no excess snapshots found")

    # -----------------------------------------------------------------------
    # Phase 4 — LRU eviction above watermark
    # -----------------------------------------------------------------------
    total_bytes = sum(dir_size_bytes(d) for d in remaining)
    log.info(
        "─── phase 4: watermark check (cache: %s  |  limit: %s) ───",
        fmt_gb(total_bytes),
        fmt_gb(watermark_bytes),
    )
    if total_bytes > watermark_bytes:
        target_bytes = watermark_bytes * 0.7
        running = float(total_bytes)
        candidates: list[tuple[float, Path, int]] = []
        for d in remaining:
            if is_protected(d) or is_too_young(d):
                continue
            try:
                candidates.append((snap_age(d), d, dir_size_bytes(d)))
            except Exception:
                pass
        candidates.sort(key=lambda x: -x[0])  # oldest first
        for _age, d, size in candidates:
            if running <= target_bytes:
                break
            delete(d, f"LRU eviction (cache {fmt_gb(int(running))} > watermark {fmt_gb(watermark_bytes)})")
            remaining.remove(d)
            running -= size
        log.info("estimated cache after eviction: %s", fmt_gb(int(running)))
    else:
        log.info("cache is within watermark — no LRU eviction needed")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    remaining_bytes = sum(dir_size_bytes(d) for d in remaining)
    label = "preview" if dry_run else "done"
    log.info(
        "═══ cleanup %s: %d snapshot(s) %s, %s freed, %s remaining ═══",
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
        description="Clean up obsolete Teutonic Quasar pair-eval model cache directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        metavar="DIR",
        help="Model cache directory to clean",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=DEFAULT_MIN_AGE_H,
        metavar="N",
        help="Never delete anything newer than N hours (safety grace window)",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=DEFAULT_MAX_AGE_DAYS,
        metavar="N",
        help="Unconditionally delete complete snapshots older than N days",
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=DEFAULT_KEEP_RECENT,
        metavar="N",
        help="Keep at most N most-recent snapshots per model; evict the rest",
    )
    parser.add_argument(
        "--watermark-gb",
        type=float,
        default=DEFAULT_WATERMARK_GB,
        metavar="N",
        help="Also run LRU eviction when total cache size exceeds N GB",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting anything",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging (shows skipped dirs)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.dry_run:
        log.info("DRY RUN — no files will be deleted")

    result = run_cleanup(
        cache_dir=args.cache_dir.resolve(),
        min_age_s=args.min_age_hours * 3600,
        max_age_s=args.max_age_days * 86400,
        keep_recent=args.keep_recent,
        watermark_bytes=args.watermark_gb * 1e9,
        dry_run=args.dry_run,
    )

    sys.exit(0 if result["deleted"] >= 0 else 1)


if __name__ == "__main__":
    main()
