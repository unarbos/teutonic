#!/usr/bin/env python3
"""Upload the current king model snapshot to a Hugging Face repository.

Resolves the king snapshot in this order:
  1. MODEL_CACHE_DIR/.current_king  (written by eval server / write_king_ref.py)
  2. GET <eval-server>/health        (live king_loaded tuple)
  3. Most recent eval JSON record    (completed-eval fallback)

Then uploads the snapshot directory to HF using upload_folder().

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
DEFAULT_HF_NAMESPACE = os.environ.get("HF_NAMESPACE", "huxdendrite")

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
        p = (verdict.get("model_artifacts") or {}).get("king", {}).get("path", "")
        if p and Path(p).exists():
            return str(Path(p).resolve()), f"eval record ({record_file.name})"
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
      2. MODEL_CACHE_DIR/.current_king  (written by write_king_ref.py / eval server)
      3. GET <eval-server>/health       (live king in GPU memory)
      4. Newest eval JSON record        (completed-eval fallback)
    """
    if snapshot_override:
        p = Path(snapshot_override).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--snapshot path does not exist: {p}")
        return p, "CLI --snapshot override", _meta_from_snapshot_path(p, cache_dir)

    result = (
        _king_from_ref_file(cache_dir)
        or _king_from_server(eval_server, cache_dir)
        or _king_from_records(record_dir)
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


def decrypted_upload_snapshot(snapshot_dir: Path) -> Path:
    manifest = load_encryption_manifest(snapshot_dir)
    if manifest is None:
        return snapshot_dir

    age = shutil.which("age")
    if age is None:
        raise RuntimeError("encrypted king snapshot found, but `age` is not installed on PATH")
    identity = model_decryption_key()
    output = snapshot_dir.with_name(snapshot_dir.name + "-decrypted")
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
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    cache_dir = args.cache_dir.resolve()
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

    upload_snapshot_dir = decrypted_upload_snapshot(snapshot_dir)
    if upload_snapshot_dir != snapshot_dir:
        log.info("uploading decrypted king snapshot: %s", upload_snapshot_dir)

    # Skip upload when the actual upload snapshot hasn't changed since last run.
    snapshot_str = str(upload_snapshot_dir)
    try:
        last = last_uploaded_file.read_text().strip()
    except OSError:
        last = ""
    if last == snapshot_str and not args.snapshot:
        log.info("king unchanged since last upload (%s) — skipping", snapshot_str)
        print(json.dumps({"status": "skipped", "reason": "king unchanged", "snapshot": snapshot_str}))
        return

    # Derive HF repo name from king if not explicitly set
    hf_repo = args.hf_repo
    if not hf_repo:
        raw = king_meta.get("king_repo") or snapshot_dir.parent.name.replace("--", "/")
        # Keep only the model-name part and prepend the configured namespace
        model_name = raw.split("/")[-1] if "/" in raw else raw
        hf_repo = f"{args.hf_namespace}/{model_name}"
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

    result = upload_to_hf(
        snapshot_dir=upload_snapshot_dir,
        hf_repo=hf_repo,
        token=token,
        revision=args.hf_revision,
        commit_message=commit_message,
        private=args.hf_private,
        dry_run=args.dry_run,
    )

    # Record successful upload so next run can detect no-change
    if not args.dry_run:
        try:
            last_uploaded_file.write_text(snapshot_str)
        except OSError as exc:
            log.warning("could not write %s: %s", last_uploaded_file, exc)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
