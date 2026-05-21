"""Hippius Hub model references and local materialization."""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hippius_hub import snapshot_download, upload_folder


REGISTRY_URL = os.environ.get("TEUTONIC_HIPPIUS_REGISTRY", "https://registry.hippius.com")
HUB_URL = os.environ.get("TEUTONIC_HIPPIUS_HUB_URL", "https://hub.hippius.com")
MODEL_CACHE_DIR = os.environ.get("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/hippius_models")
HUB_TOKEN = (
    os.environ.get("HIPPIUS_HUB_TOKEN")
    or os.environ.get("HIPPIUS_TOKEN")
    or os.environ.get("TEUTONIC_HIPPIUS_TOKEN")
    or None
)

REVEAL_VERSION = "v2"
REVEAL_V3_PREFIX = "v3"
REPO_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._/-]*$")
# Two digest shapes accepted:
#   - "sha256:<64hex>"  Hippius OCI manifest digest (challenger uploads)
#   - "hf:<40hex>"      HuggingFace commit SHA (genesis king pinned to a vanilla
#                       HF repo without a Hippius mirror)
HIPPIUS_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HF_DIGEST_RE = re.compile(r"^hf:[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^(sha256:[0-9a-f]{64}|hf:[0-9a-f]{40})$")
BARE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LEGACY_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
LEGACY_MODEL_HASH_RE = BARE_SHA256_RE
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{47,48}$")

ALLOW_PATTERNS = [
    "*.safetensors",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer*",
    "special_tokens*",
    "generation_config.json",
    "*.json",
    "*.model",
    "*.txt",
]


@dataclass(frozen=True)
class ModelRef:
    """Immutable Hippius Hub model reference."""

    repo: str
    digest: str

    def __post_init__(self) -> None:
        repo = (self.repo or "").strip()
        digest = (self.digest or "").strip()
        if not REPO_RE.match(repo):
            raise ValueError(f"invalid Hippius repo id: {self.repo!r}")
        if not DIGEST_RE.match(digest):
            raise ValueError(f"invalid Hippius OCI digest: {self.digest!r}")
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "digest", digest)

    @property
    def immutable_ref(self) -> str:
        return f"{self.repo}@{self.digest}"

    @property
    def hub_url(self) -> str:
        return f"{HUB_URL.rstrip('/')}/{self.repo}@{self.digest}"


def _normalise_digest(value: str) -> str:
    digest = (value or "").strip()
    if LEGACY_MODEL_HASH_RE.match(digest):
        digest = f"sha256:{digest}"
    if not DIGEST_RE.match(digest):
        raise ValueError(f"invalid OCI digest: {value!r}")
    return digest


def _strip_sha_prefix(digest: str) -> str:
    d = (digest or "").strip()
    if d.startswith("sha256:"):
        d = d[len("sha256:"):]
    return d.lower()


def build_reveal_payload(king_hash: str, ref: ModelRef) -> str:
    king_prefix = (king_hash or "")[:16]
    if not re.match(r"^[0-9a-fA-F]{4,16}$", king_prefix):
        raise ValueError(f"invalid king hash prefix: {king_hash!r}")
    return f"{REVEAL_VERSION}|{king_prefix.lower()}|{ref.repo}|{ref.digest}"


def parse_reveal_payload(data: str) -> tuple[str, ModelRef]:
    parts = (data or "").strip().split("|")
    if len(parts) != 4 or parts[0] != REVEAL_VERSION:
        raise ValueError("expected v2|king_hash16|repo|sha256:digest reveal")
    king_hash, repo, digest = parts[1], parts[2], _normalise_digest(parts[3])
    if not re.match(r"^[0-9a-fA-F]{4,16}$", king_hash):
        raise ValueError(f"invalid king hash prefix: {king_hash!r}")
    return king_hash.lower(), ModelRef(repo, digest)


# v3 payload: `v3|<king_digest>|<challenger_repo>|<challenger_digest>|<author_hotkey>`.
# Digests are bare 64-hex sha256 (no `sha256:` prefix). author_hotkey is the
# 48-char ss58 of the submitter, kept for cross-check against the chain-side
# iteration key (spec §4). Total ~220 chars, within the commit-reveal budget.

def build_reveal_v3(king_digest: str, challenger_ref: ModelRef, author_hotkey: str) -> str:
    king = _strip_sha_prefix(king_digest)
    if not BARE_SHA256_RE.match(king):
        raise ValueError(f"invalid king digest (need 64 hex): {king_digest!r}")
    chall = _strip_sha_prefix(challenger_ref.digest)
    hk = (author_hotkey or "").strip()
    if not SS58_RE.match(hk):
        raise ValueError(f"invalid author hotkey ss58: {author_hotkey!r}")
    return f"{REVEAL_V3_PREFIX}|{king}|{challenger_ref.repo}|{chall}|{hk}"


def parse_reveal_v3(payload: str) -> tuple[str, ModelRef, str]:
    """Returns (king_digest_64hex, ModelRef(challenger_repo, sha256:<digest>), author_hotkey)."""
    parts = (payload or "").strip().split("|")
    if len(parts) != 5 or parts[0] != REVEAL_V3_PREFIX:
        raise ValueError("expected v3|king_digest|repo|challenger_digest|author_hotkey reveal")
    king = parts[1].lower()
    if not BARE_SHA256_RE.match(king):
        raise ValueError(f"invalid v3 king digest: {parts[1]!r}")
    hk = parts[4].strip()
    if not SS58_RE.match(hk):
        raise ValueError(f"invalid v3 author hotkey: {parts[4]!r}")
    return king, ModelRef(parts[2], _normalise_digest(parts[3])), hk


def _cache_snapshot_path(ref: ModelRef) -> Path:
    repo_key = ref.repo.replace("/", "--")
    digest_key = ref.digest.replace(":", "-")
    return Path(MODEL_CACHE_DIR) / repo_key / "snapshots" / digest_key


def local_snapshot_path(ref: ModelRef) -> str:
    path = _cache_snapshot_path(ref)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return str(path)


def _call_snapshot_download(ref: ModelRef, local_dir: str | None, max_workers: int | None) -> str:
    # HF-direct path: bypass Hippius, pull straight from huggingface.co at the
    # pinned commit SHA. Used for the genesis king when we don't want to mirror
    # to Hippius first. Challenger uploads still go through the Hippius path.
    if HF_DIGEST_RE.match(ref.digest):
        from huggingface_hub import snapshot_download as hf_snapshot_download
        revision = ref.digest.split(":", 1)[1]
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        return str(hf_snapshot_download(
            repo_id=ref.repo,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=ALLOW_PATTERNS,
            max_workers=max_workers or 8,
            token=hf_token,
        ))

    # Hippius OCI path.
    kwargs = {
        "repo": ref.repo,
        "repo_id": ref.repo,
        "digest": ref.digest,
        "revision": ref.digest,
        "local_dir": local_dir,
        "allow_patterns": ALLOW_PATTERNS,
        "max_workers": max_workers,
        "token": HUB_TOKEN,
    }
    candidates = [
        ("repo", "digest", "local_dir", "allow_patterns", "max_workers", "token"),
        ("repo_id", "digest", "local_dir", "allow_patterns", "max_workers", "token"),
        ("repo", "revision", "local_dir", "allow_patterns", "max_workers", "token"),
        ("repo_id", "revision", "local_dir", "allow_patterns", "max_workers", "token"),
    ]
    last_error: Exception | None = None
    for names in candidates:
        call_kwargs = {name: kwargs[name] for name in names if kwargs[name] is not None}
        try:
            return str(snapshot_download(**call_kwargs))
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("snapshot_download failed without an exception")


def materialize_model(ref: ModelRef, local_dir: str | None = None, max_workers: int | None = None) -> str:
    """Download or reuse an immutable Hippius Hub snapshot."""
    target = Path(local_dir) if local_dir else _cache_snapshot_path(ref)
    if target.exists() and any(target.glob("*.safetensors")):
        return str(target)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _call_snapshot_download(ref, str(target), max_workers)


def ensure_ref_exists(ref: ModelRef) -> bool:
    snapshot = materialize_model(ref, max_workers=4)
    if not list(Path(snapshot).glob("*.safetensors")):
        raise FileNotFoundError(f"{ref.immutable_ref} has no .safetensors files")
    return True


def list_snapshot_files(snapshot: str | os.PathLike[str]) -> list[str]:
    root = Path(snapshot)
    return sorted(
        str(p.relative_to(root)).replace(os.sep, "/")
        for p in root.rglob("*")
        if p.is_file()
    )


def snapshot_size(snapshot: str | os.PathLike[str], files: Iterable[str] | None = None) -> int:
    root = Path(snapshot)
    paths = (root / f for f in files) if files is not None else (p for p in root.rglob("*") if p.is_file())
    total = 0
    for path in paths:
        try:
            total += Path(path).stat().st_size
        except FileNotFoundError:
            continue
    return total


def sha256_safetensors(path: str | os.PathLike[str]) -> str:
    h = __import__("hashlib").sha256()
    for p in sorted(Path(path).glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def _extract_uploaded_digest(result) -> str:
    if isinstance(result, ModelRef):
        return result.digest
    if isinstance(result, str):
        return _normalise_digest(result)
    if isinstance(result, dict):
        for key in ("digest", "manifest_digest", "oci_digest", "uploaded_digest"):
            if result.get(key):
                return _normalise_digest(str(result[key]))
    for key in ("digest", "manifest_digest", "oci_digest", "uploaded_digest"):
        value = getattr(result, key, None)
        if value:
            return _normalise_digest(str(value))
    raise ValueError(f"could not determine Hippius upload digest from {result!r}")


def upload_model_folder(
    folder_path: str | os.PathLike[str],
    repo: str,
    revision: str | None = None,
    commit_message: str | None = None,
) -> ModelRef:
    """Upload a model folder to Hippius Hub and return its immutable digest."""
    folder = str(folder_path)
    kwargs = {
        "folder_path": folder,
        "path": folder,
        "repo": repo,
        "repo_id": repo,
        "tag": revision,
        "revision": revision,
        "commit_message": commit_message,
        "allow_patterns": ALLOW_PATTERNS,
        "token": HUB_TOKEN,
    }
    candidates = [
        ("folder_path", "repo", "tag", "commit_message", "allow_patterns", "token"),
        ("folder_path", "repo_id", "tag", "commit_message", "allow_patterns", "token"),
        ("folder_path", "repo", "revision", "commit_message", "allow_patterns", "token"),
        ("path", "repo", "tag", "commit_message", "allow_patterns", "token"),
    ]
    last_error: Exception | None = None
    for names in candidates:
        call_kwargs = {name: kwargs[name] for name in names if kwargs[name] is not None}
        try:
            result = upload_folder(**call_kwargs)
            return ModelRef(repo, _extract_uploaded_digest(result))
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("upload_folder failed without an exception")
