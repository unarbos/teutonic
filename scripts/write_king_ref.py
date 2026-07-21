#!/usr/bin/env python3
"""
Recover the current king snapshot path and write it to MODEL_CACHE_DIR/.current_king.

Sources tried in order:
  1. Eval JSON records — completed accepted challengers are authoritative
  2. GET /health on the live eval server — fallback when records are unavailable

Run this once when the eval server was started before the .current_king file was
introduced (or after the file was lost). After this, the eval server writes the
file itself on every new king load via ensure_king().

Usage:
    python scripts/write_king_ref.py
    python scripts/write_king_ref.py --dry-run
    python scripts/write_king_ref.py --eval-server http://localhost:9000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("write_king_ref")

DEFAULT_CACHE_DIR = Path(os.environ.get("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/quasar_pair_models"))
DEFAULT_RECORD_DIR = Path(os.environ.get("TEUTONIC_EVAL_RECORD_DIR", "/tmp/teutonic/quasar_pair_evals"))
DEFAULT_EVAL_SERVER = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# Source 1: live eval server /health endpoint
# ---------------------------------------------------------------------------

def king_path_from_server(server_url: str, cache_dir: Path) -> str | None:
    """
    Query GET /health and reconstruct the king snapshot path from king_loaded.

    king_loaded is the _king_key tuple: [repo, digest_or_latest, config_source].
    The snapshot dir follows the same convention as materialize_model():
        cache_dir / repo.replace("/", "--") / digest.replace(":", "-")
    """
    url = server_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        log.warning("could not reach eval server at %s: %s", url, exc)
        return None

    king_key = data.get("king_loaded")
    if not king_key:
        log.info("eval server reachable but king_loaded is null (no king loaded yet)")
        return None

    # king_key is [repo, digest_or_latest, config_source]
    try:
        repo, digest, _config_source = king_key
    except (TypeError, ValueError) as exc:
        log.warning("unexpected king_loaded format %r: %s", king_key, exc)
        return None

    digest_dir = digest.replace(":", "-") if digest and digest != "latest" else "latest"
    snapshot = cache_dir / repo.replace("/", "--") / digest_dir

    log.info("eval server reports current king:")
    log.info("  king_repo   : %s", repo)
    log.info("  king_digest : %s", digest)
    log.info("  snapshot    : %s", snapshot)

    if not snapshot.exists():
        log.warning("king snapshot dir does not exist on disk: %s", snapshot)
        log.warning("the server has it in GPU memory but disk files may have been deleted")
        # Still return it — writing the ref is the right thing so cleanup skips it
        return str(snapshot.resolve())

    return str(snapshot.resolve())


# ---------------------------------------------------------------------------
# Source 2: completed eval JSON records (fallback)
# ---------------------------------------------------------------------------

def king_path_from_records(record_dir: Path) -> tuple[str, Path] | None:
    """
    Scan eval record JSONs (newest first) and return the first valid king
    snapshot path found, together with the record file it came from.

    Record filenames are prefixed with %Y%m%dT%H%M%SZ so lexicographic sort
    gives chronological order. Only completed evals write records, so an
    in-progress eval whose king changed since the last record won't be captured
    here — that's why the live server query is tried first.
    """
    records = sorted(record_dir.glob("*.json"), reverse=True)
    if not records:
        log.error("no eval record JSON files found in %s", record_dir)
        return None

    log.info("found %d eval record(s), scanning newest first", len(records))

    for record_file in records:
        try:
            data = json.loads(record_file.read_text())
        except Exception as exc:
            log.warning("skipping %s (parse error: %s)", record_file.name, exc)
            continue

        verdict = data.get("verdict") or {}
        artifact_name = "challenger" if verdict.get("accepted") else "king"
        king_path_str = (verdict.get("model_artifacts") or {}).get(artifact_name, {}).get("path", "")

        if not king_path_str:
            log.warning("skipping %s (no model_artifacts.%s.path)", record_file.name, artifact_name)
            continue

        king_path = Path(king_path_str)
        king_repo = verdict.get(f"{artifact_name}_repo", "?")
        king_digest = verdict.get(f"{artifact_name}_digest", "latest")

        if not king_path.exists():
            log.warning("skipping %s — snapshot no longer on disk: %s", record_file.name, king_path)
            continue

        log.info("found effective king in %s (%s)", record_file.name, artifact_name)
        log.info("  king_repo   : %s", king_repo)
        log.info("  king_digest : %s", king_digest or "latest")
        log.info("  snapshot    : %s", king_path)
        return str(king_path.resolve()), record_file

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write .current_king ref file from the live eval server or eval JSON records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Model cache directory")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR, help="Eval record directory")
    parser.add_argument("--eval-server", default=DEFAULT_EVAL_SERVER, help="Eval server base URL for /health query")
    parser.add_argument("--no-server", action="store_true", help="Skip live server query, use records only")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing")
    args = parser.parse_args()

    setup_logging()

    cache_dir = args.cache_dir.resolve()
    ref_file = cache_dir / ".current_king"
    king_path: str | None = None
    source: str = ""

    if not king_path:
        record_dir = args.record_dir.resolve()
        if not record_dir.exists():
            log.error("record dir does not exist: %s", record_dir)
            raise SystemExit(1)
        log.info("checking eval records in %s …", record_dir)
        result = king_path_from_records(record_dir)
        if result:
            king_path, record_file = result
            source = f"eval record ({record_file.name})"

    if not king_path and not args.no_server:
        log.info("falling back to eval server %s …", args.eval_server)
        king_path = king_path_from_server(args.eval_server, cache_dir)
        if king_path:
            source = f"live server ({args.eval_server}/health)"

    if not king_path:
        log.error("could not determine current king from any source")
        raise SystemExit(1)

    if args.dry_run:
        log.info("[DRY RUN] would write to %s  (source: %s):", ref_file, source)
        log.info("  %s", king_path)
        return

    ref_file.parent.mkdir(parents=True, exist_ok=True)
    ref_file.write_text(king_path)
    log.info("wrote %s  (source: %s)", ref_file, source)
    log.info("  -> %s", king_path)


if __name__ == "__main__":
    main()
