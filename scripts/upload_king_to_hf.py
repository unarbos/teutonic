#!/usr/bin/env python3
"""Upload the current king model snapshot to a Hugging Face repository.

Resolves the king snapshot in this order:
  1. Most recent eval JSON record    (accepted challengers become king)
  2. MODEL_CACHE_DIR/.current_king  (written by eval server / write_king_ref.py)
  3. GET <eval-server>/health        (live king_loaded tuple fallback)

Then uploads the snapshot directory to HF using upload_folder(). It also
uploads evaluated non-king challenger snapshots after a delay.

Credentials needed:
  HF_TOKEN   — Hugging Face write token  (or --hf-token)
  HF_REPO    — Destination repo id       (or --hf-repo)   e.g. "myorg/my-model"

Usage:
    python scripts/upload_king_to_hf.py --hf-repo myorg/my-model --hf-token hf_...
    HF_TOKEN=hf_... HF_REPO=myorg/my-model python scripts/upload_king_to_hf.py
    python scripts/upload_king_to_hf.py --dry-run --hf-repo myorg/my-model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("upload_king_to_hf")

ROOT = Path(__file__).resolve().parents[1]
ENCRYPTION_MANIFEST_NAME = "teutonic_encryption.json"
DEFAULT_MODEL_DECRYPTION_KEY = ROOT / "keys" / "validator_model_decryption.key"

DEFAULT_CACHE_DIR = Path(os.environ.get("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/quasar_pair_models"))
DEFAULT_RECORD_DIR = Path(os.environ.get("TEUTONIC_EVAL_RECORD_DIR", "/tmp/teutonic/quasar_pair_evals"))
DEFAULT_EVAL_SERVER = os.environ.get("TEUTONIC_EVAL_SERVER", "http://localhost:9000")
DEFAULT_HF_NAMESPACE = os.environ.get("HF_NAMESPACE", "dendriteholdings")
DEFAULT_NON_KING_UPLOAD_DELAY_S = int(os.environ.get("TEUTONIC_NON_KING_UPLOAD_DELAY_S", str(6 * 60 * 60)))
DEFAULT_NON_KING_MOVE_DELAY_S = int(os.environ.get("TEUTONIC_NON_KING_MOVE_DELAY_S", str(60 * 60)))
DEFAULT_UPLOAD_STAGING_DIR = Path(os.environ.get("TEUTONIC_HF_UPLOAD_STAGING_DIR", "/models"))
DEFAULT_DELETE_NON_KING_AFTER_UPLOAD = os.environ.get("TEUTONIC_DELETE_NON_KING_AFTER_UPLOAD", "1").lower() not in ("0", "false", "no")

HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN")

UPLOAD_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.py",
    "tokenizer*",
    "special_tokens*",
    "*.model",
    "*.tiktoken",
    "merges.txt",
    "vocab.*",
    "*.txt",
    "*.jinja",
]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# King resolution (same sources as write_king_ref.py)
# ---------------------------------------------------------------------------

def _king_from_ref_file(cache_dir: Path) -> tuple[str, str] | None:
    """Read snapshot path from .current_king. Returns (snapshot_path, source)."""
    try:
        p = (cache_dir / ".current_king").read_text().strip()
        if p and Path(p).exists():
            return p, ".current_king file"
    except OSError:
        pass
    return None


def _king_from_server(server_url: str, cache_dir: Path) -> tuple[str, str] | None:
    """Query /health. Returns (snapshot_path, source)."""
    url = server_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        log.warning("could not reach eval server at %s: %s", url, exc)
        return None

    king_key = data.get("king_loaded")
    if not king_key:
        return None
    try:
        repo, digest, _ = king_key
    except (TypeError, ValueError):
        return None

    digest_dir = digest.replace(":", "-") if digest and digest != "latest" else "latest"
    snapshot = cache_dir / repo.replace("/", "--") / digest_dir
    if snapshot.exists():
        return str(snapshot.resolve()), f"eval server ({url})"
    log.warning("king from server does not exist on disk: %s", snapshot)
    return None


def _king_from_records(record_dir: Path) -> tuple[str, str] | None:
    """Scan eval records newest-first. Returns (snapshot_path, source)."""
    records = sorted(record_dir.glob("*.json"), reverse=True)
    for record_file in records:
        try:
            data = json.loads(record_file.read_text())
        except Exception:
            continue
        verdict = data.get("verdict") or {}
        artifact_name = "challenger" if verdict.get("accepted") else "king"
        p = (verdict.get("model_artifacts") or {}).get(artifact_name, {}).get("path", "")
        if p and Path(p).exists():
            return str(Path(p).resolve()), f"eval record ({record_file.name}, {artifact_name})"
    return None


def _meta_from_snapshot_path(snapshot: Path, cache_dir: Path) -> dict:
    """Reconstruct king_repo and king_digest from a snapshot directory path.

    Path layout: cache_dir/<repo_with_dashes>/<digest_with_dashes>
    e.g. /tmp/.../Org--model-name/sha256-abc123
    """
    try:
        repo = snapshot.parent.name.replace("--", "/")
        digest = snapshot.name.replace("-", ":", 1)  # only first dash is the sha256: separator
        return {"king_repo": repo, "king_digest": digest}
    except Exception:
        return {}


def derived_hf_repo(namespace: str, source_repo: str) -> str:
    model_name = source_repo.split("/")[-1] if "/" in source_repo else source_repo
    return f"{namespace}/{model_name}"


def safe_dir_name(value: str) -> str:
    return value.replace("/", "--").replace("\\", "--").replace(":", "-")


def resolve_king(
    cache_dir: Path,
    record_dir: Path,
    eval_server: str,
    snapshot_override: str | None,
) -> tuple[Path, str, dict]:
    """
    Return (snapshot_dir, source_description, king_meta).
    king_meta contains repo/digest info derived from the snapshot path.

    Resolution order:
      1. --snapshot CLI override
      2. Newest eval JSON record        (accepted challengers become king)
      3. MODEL_CACHE_DIR/.current_king  (written by write_king_ref.py / eval server)
      4. GET <eval-server>/health       (live king in GPU memory fallback)
    """
    if snapshot_override:
        p = Path(snapshot_override).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--snapshot path does not exist: {p}")
        return p, "CLI --snapshot override", _meta_from_snapshot_path(p, cache_dir)

    result = (
        _king_from_records(record_dir)
        or _king_from_ref_file(cache_dir)
        or _king_from_server(eval_server, cache_dir)
    )
    if result is None:
        raise RuntimeError(
            "could not determine current king snapshot. "
            "Run write_king_ref.py first, or use --snapshot to specify the directory."
        )
    snapshot_path, source = result
    p = Path(snapshot_path)
    return p, source, _meta_from_snapshot_path(p, cache_dir)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def hf_token(cli_token: str) -> str | None:
    """
    Return an HF token from (in order):
      1. --hf-token CLI argument
      2. HF_TOKEN / HUGGINGFACE_API_KEY / HUGGING_FACE_HUB_TOKEN env vars
      3. Token cached by `huggingface-cli login`  (~/.cache/huggingface/token)

    Returns None if no token is found — HfApi will then attempt anonymous access,
    which works for public repos but fails for private ones or on upload.
    """
    if cli_token:
        return cli_token
    for name in HF_TOKEN_ENV_NAMES:
        t = (os.environ.get(name) or "").strip()
        if t:
            return t
    try:
        from huggingface_hub import get_token
        cached = get_token()
        if cached:
            log.info("using token cached by huggingface-cli login")
            return cached
    except Exception:
        pass
    return None


def upload_to_hf(
    snapshot_dir: Path,
    hf_repo: str,
    token: str,
    revision: str,
    commit_message: str,
    private: bool,
    dry_run: bool,
) -> dict:
    log.info("ensuring HF repo %s exists (private=%s) …", hf_repo, private)

    files = sorted(snapshot_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    log.info(
        "uploading %d file(s) (%.2f GB) from %s …",
        file_count,
        total_bytes / 1e9,
        snapshot_dir,
    )
    log.info("  → hf.co/%s  revision=%s", hf_repo, revision)
    log.info("  commit: %s", commit_message)

    if dry_run:
        log.info("[DRY RUN] skipping actual upload")
        return {"repo": hf_repo, "revision": revision, "dry_run": True}

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=hf_repo, repo_type="model", private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=hf_repo,
        repo_type="model",
        folder_path=str(snapshot_dir),
        revision=revision,
        commit_message=commit_message,
        allow_patterns=UPLOAD_ALLOW_PATTERNS,
    )
    log.info("upload complete: %s", result)
    return {
        "repo": hf_repo,
        "revision": revision,
        "private": private,
        "commit": str(result),
        "snapshot": str(snapshot_dir),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def model_decryption_key() -> Path:
    configured = os.environ.get("TEUTONIC_MODEL_DECRYPTION_KEY")
    key = Path(configured).expanduser() if configured else DEFAULT_MODEL_DECRYPTION_KEY
    key = key.resolve()
    if not key.is_file() or not os.access(key, os.R_OK):
        raise RuntimeError(
            "encrypted king snapshot found, but no private key is available; "
            "set TEUTONIC_MODEL_DECRYPTION_KEY or create keys/validator_model_decryption.key"
        )
    return key


def load_encryption_manifest(snapshot: Path) -> dict | None:
    manifest_path = snapshot / ENCRYPTION_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scheme") != "age-x25519":
        raise RuntimeError(f"unsupported model encryption scheme: {manifest.get('scheme')!r}")
    if not manifest.get("files"):
        raise RuntimeError(f"{ENCRYPTION_MANIFEST_NAME} has no encrypted files")
    return manifest


def decrypted_upload_snapshot(snapshot_dir: Path, output_parent: Path | None = None) -> Path:
    manifest = load_encryption_manifest(snapshot_dir)
    if manifest is None:
        return snapshot_dir

    age = shutil.which("age")
    if age is None:
        raise RuntimeError("encrypted model snapshot found, but `age` is not installed on PATH")
    identity = model_decryption_key()
    if output_parent is None:
        output = snapshot_dir.with_name(snapshot_dir.name + "-decrypted")
    else:
        output = output_parent / snapshot_dir.parent.name / f"{snapshot_dir.name}-decrypted"
    if output.exists():
        shutil.rmtree(output)

    encrypted = {item["path"]: item for item in manifest["files"]}
    for src in sorted(p for p in snapshot_dir.rglob("*") if p.is_file()):
        rel = str(src.relative_to(snapshot_dir)).replace("\\", "/")
        if rel == ENCRYPTION_MANIFEST_NAME:
            continue
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel in encrypted:
            subprocess.run([age, "-d", "-i", str(identity), "-o", str(dst), str(src)], check=True)
            item = encrypted[rel]
            if sha256_file(dst) != item.get("plain_sha256"):
                raise RuntimeError(f"{rel}: plaintext sha256 mismatch after decrypt")
            if dst.stat().st_size != int(item.get("plain_size", -1)):
                raise RuntimeError(f"{rel}: plaintext size mismatch after decrypt")
        elif src.name.endswith(".safetensors"):
            raise RuntimeError(f"{rel}: .safetensors file is missing from {ENCRYPTION_MANIFEST_NAME}")
        else:
            shutil.copy2(src, dst)

    return output


def load_marker_set(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return set()
    return {str(item) for item in data} if isinstance(data, list) else set()


def save_marker_set(path: Path, markers: set[str]) -> None:
    path.write_text(json.dumps(sorted(markers), indent=2) + "\n")


def staged_snapshot_path(staging_dir: Path, source_repo: str, digest: str) -> Path:
    digest_part = safe_dir_name(digest or "latest")
    return staging_dir / "non_king" / safe_dir_name(source_repo) / digest_part


def stage_snapshot_for_upload(snapshot: Path, target: Path, dry_run: bool) -> Path:
    snapshot = snapshot.resolve()
    target = target.resolve()
    if snapshot == target or target.is_dir():
        return target
    if dry_run:
        log.info("[DRY RUN] would move non-king snapshot %s -> %s", snapshot, target)
        return snapshot
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("moving non-king snapshot to upload staging: %s -> %s", snapshot, target)
    shutil.move(str(snapshot), str(target))
    return target


def delete_snapshot_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        log.info("[DRY RUN] would delete uploaded snapshot dir %s", path)
        return
    if path.is_dir():
        log.info("deleting uploaded snapshot dir %s", path)
        shutil.rmtree(path)


def upload_non_king_models(
    record_dir: Path,
    cache_dir: Path,
    current_king_snapshots: set[Path],
    hf_namespace: str,
    token: str,
    revision: str,
    private: bool,
    dry_run: bool,
    delay_s: int,
    move_delay_s: int,
    staging_dir: Path,
    delete_after_upload: bool,
) -> list[dict]:
    marker_file = cache_dir / ".uploaded_non_king_snapshots.json"
    uploaded = load_marker_set(marker_file)
    now = time.time()
    move_before_mtime = now - max(0, move_delay_s)
    upload_before_mtime = now - max(0, delay_s)
    current_king_snapshots = {p.resolve() for p in current_king_snapshots}
    results = []

    for record_file in sorted(record_dir.glob("*.json")):
        record_mtime = record_file.stat().st_mtime
        if record_mtime > move_before_mtime:
            continue
        try:
            data = json.loads(record_file.read_text())
            request = data.get("request") or {}
            verdict = data.get("verdict") or {}
            if verdict.get("accepted") or verdict.get("verdict") == "challenger":
                continue
            artifacts = verdict.get("model_artifacts") or {}
            meta = artifacts.get("challenger") or {}
            snapshot_path = meta.get("path") or ""
            if not snapshot_path:
                continue
            snapshot = Path(snapshot_path).resolve()
        except Exception as exc:
            log.warning("skipping malformed eval record %s: %s", record_file.name, exc)
            continue

        source_repo = (
            verdict.get("challenger_repo")
            or request.get("challenger_repo")
            or snapshot.parent.name.replace("--", "/")
        )
        digest = verdict.get("challenger_digest") or request.get("challenger_digest") or snapshot.name
        marker = f"{source_repo}@{digest}"
        target = staged_snapshot_path(staging_dir, source_repo, digest)
        if not snapshot.is_dir() and target.is_dir():
            snapshot = target.resolve()
        if not snapshot.is_dir() or snapshot in current_king_snapshots or marker in uploaded:
            continue

        snapshot = stage_snapshot_for_upload(snapshot, target, dry_run)
        if record_mtime > upload_before_mtime:
            continue

        upload_snapshot_dir = decrypted_upload_snapshot(snapshot, staging_dir / "decrypted")
        if upload_snapshot_dir != snapshot:
            log.info("uploading decrypted non-king snapshot: %s", upload_snapshot_dir)

        result = upload_to_hf(
            snapshot_dir=upload_snapshot_dir,
            hf_repo=derived_hf_repo(hf_namespace, source_repo),
            token=token,
            revision=revision,
            commit_message=f"Upload challenger model: {source_repo}@{digest}",
            private=private,
            dry_run=dry_run,
        )
        results.append(result)
        if not dry_run:
            uploaded.add(marker)
            save_marker_set(marker_file, uploaded)
        if delete_after_upload:
            delete_snapshot_dir(upload_snapshot_dir, dry_run)
            if upload_snapshot_dir != snapshot:
                delete_snapshot_dir(snapshot, dry_run)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the current king model snapshot to Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("HF_REPO", ""),
        metavar="ORG/MODEL",
        help="Destination HF repo id. Defaults to <hf-namespace>/<king-model-name>.",
    )
    parser.add_argument(
        "--hf-namespace",
        default=DEFAULT_HF_NAMESPACE,
        metavar="USER_OR_ORG",
        help="HF user/org to use when auto-deriving the repo name from the king",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        metavar="TOKEN",
        help="HF write token (falls back to HF_TOKEN / HUGGINGFACE_API_KEY env vars)",
    )
    parser.add_argument(
        "--hf-revision",
        default=os.environ.get("HF_REVISION", "main"),
        metavar="BRANCH",
        help="Target branch / revision on HF",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        default=os.environ.get("HF_PRIVATE", "").lower() in ("1", "true", "yes"),
        help="Create the HF repo as private if it does not already exist",
    )
    parser.add_argument(
        "--snapshot",
        default="",
        metavar="DIR",
        help="Override: upload this directory instead of auto-resolving the king",
    )
    parser.add_argument(
        "--commit-message",
        default="",
        metavar="MSG",
        help="Custom commit message (auto-generated from king metadata if omitted)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Model cache directory (for reading .current_king)",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=DEFAULT_RECORD_DIR,
        help="Eval record directory (fallback king source)",
    )
    parser.add_argument(
        "--eval-server",
        default=DEFAULT_EVAL_SERVER,
        help="Eval server base URL (for live king query)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve king and print what would be uploaded without uploading",
    )
    parser.add_argument(
        "--non-king-upload-delay-s",
        type=int,
        default=DEFAULT_NON_KING_UPLOAD_DELAY_S,
        help="Upload non-king challenger snapshots only after this many seconds",
    )
    parser.add_argument(
        "--non-king-move-delay-s",
        type=int,
        default=DEFAULT_NON_KING_MOVE_DELAY_S,
        help="Move non-king challenger snapshots to upload staging only after this many seconds",
    )
    parser.add_argument(
        "--upload-staging-dir",
        type=Path,
        default=DEFAULT_UPLOAD_STAGING_DIR,
        help="Directory for staged/decrypted snapshots before HF upload",
    )
    parser.add_argument(
        "--keep-non-king-after-upload",
        action="store_true",
        default=not DEFAULT_DELETE_NON_KING_AFTER_UPLOAD,
        help="Keep staged non-king snapshots after successful upload",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    cache_dir = args.cache_dir.resolve()
    upload_staging_dir = args.upload_staging_dir.resolve()
    last_uploaded_file = cache_dir / ".last_uploaded_king"

    # Resolve king first so we can derive the repo name from it
    log.info("resolving current king …")
    snapshot_dir, source, king_meta = resolve_king(
        cache_dir=cache_dir,
        record_dir=args.record_dir.resolve(),
        eval_server=args.eval_server,
        snapshot_override=args.snapshot or None,
    )
    log.info("king snapshot: %s  (source: %s)", snapshot_dir, source)
    if king_meta:
        log.info("  king_repo   : %s", king_meta.get("king_repo", "?"))
        log.info("  king_digest : %s", king_meta.get("king_digest", "?"))

    upload_snapshot_dir = decrypted_upload_snapshot(snapshot_dir, upload_staging_dir / "decrypted")
    if upload_snapshot_dir != snapshot_dir:
        log.info("uploading decrypted king snapshot: %s", upload_snapshot_dir)

    # Skip upload when the actual upload snapshot hasn't changed since last run.
    snapshot_str = str(upload_snapshot_dir)
    try:
        last = last_uploaded_file.read_text().strip()
    except OSError:
        last = ""
    king_result = None
    if last == snapshot_str and not args.snapshot:
        log.info("king unchanged since last upload (%s) — skipping", snapshot_str)
        king_result = {"status": "skipped", "reason": "king unchanged", "snapshot": snapshot_str}
    else:
        # Derive HF repo name from king if not explicitly set
        hf_repo = args.hf_repo
        if not hf_repo:
            raw = king_meta.get("king_repo") or snapshot_dir.parent.name.replace("--", "/")
            hf_repo = derived_hf_repo(args.hf_namespace, raw)
            log.info("--hf-repo not set, derived from king: %s -> %s", raw, hf_repo)
        if not hf_repo:
            parser.error("could not determine HF repo name — pass --hf-repo explicitly")

        # Build commit message
        commit_message = args.commit_message
        if not commit_message:
            king_repo = king_meta.get("king_repo") or snapshot_dir.parent.name.replace("--", "/")
            king_digest = king_meta.get("king_digest") or snapshot_dir.name
            commit_message = f"Upload king model: {king_repo}@{king_digest}"
        log.info("commit message: %s", commit_message)

    # Resolve token (skip in dry-run so the script is usable without credentials)
    token = ""
    if not args.dry_run:
        token = hf_token(args.hf_token) or ""
        if token:
            log.info("authenticated as HF user")
        else:
            log.warning("no HF token found — upload may fail for private repos")

    if king_result is None:
        king_result = upload_to_hf(
            snapshot_dir=upload_snapshot_dir,
            hf_repo=hf_repo,
            token=token,
            revision=args.hf_revision,
            commit_message=commit_message,
            private=args.hf_private,
            dry_run=args.dry_run,
        )

    # Record successful king upload so next run can detect no-change.
    if not args.dry_run and king_result.get("status") != "skipped":
        try:
            last_uploaded_file.write_text(snapshot_str)
        except OSError as exc:
            log.warning("could not write %s: %s", last_uploaded_file, exc)

    non_king_results = []
    if not args.snapshot:
        non_king_results = upload_non_king_models(
            record_dir=args.record_dir.resolve(),
            cache_dir=cache_dir,
            current_king_snapshots={snapshot_dir, upload_snapshot_dir},
            hf_namespace=args.hf_namespace,
            token=token,
            revision=args.hf_revision,
            private=args.hf_private,
            dry_run=args.dry_run,
            delay_s=args.non_king_upload_delay_s,
            move_delay_s=args.non_king_move_delay_s,
            staging_dir=upload_staging_dir,
            delete_after_upload=not args.keep_non_king_after_upload,
        )

    print(json.dumps({"king": king_result, "non_king": non_king_results}, indent=2))


if __name__ == "__main__":
    main()
