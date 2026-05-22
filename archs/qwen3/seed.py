"""Seed a Qwen3 dense checkpoint as the genesis king.

Downloads the chosen Qwen3 model from HuggingFace, strips any `auto_map` from
`config.json` and any `.py` files (validator rejects these), uploads the result
to the configured backend, and prints the repo + digest to paste into
`chain.toml`.

Usage:
    HIPPIUS_HUB_TOKEN=... HF_TOKEN=... python -m archs.qwen3.seed

Override the defaults when needed:
    python -m archs.qwen3.seed \
        --source-repo Qwen/Qwen3-4B \
        --target-repo unconst/teutonic-q3-4b-genesis \
        --repo-backend hippius
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# Ensure repo-root is on the path when running as `python -m archs.qwen3.seed`.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chain_config  # noqa: E402
from model_store import upload_model_folder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed")

TARGET_REPO = os.environ.get("TEUTONIC_SEED_REPO_OVERRIDE", chain_config.SEED_REPO)
TARGET_BACKEND = os.environ.get(
    "TEUTONIC_SEED_REPO_BACKEND_OVERRIDE",
    getattr(chain_config, "SEED_REPO_BACKEND", "hf"),
)
_default_source_repo = "Qwen/Qwen3-4B"
if getattr(chain_config, "SEED_REPO_BACKEND", "hf") == "hf":
    _default_source_repo = chain_config.SEED_REPO
SOURCE_REPO = os.environ.get(
    "TEUTONIC_SEED_SOURCE_REPO",
    _default_source_repo,
)


def _scrub(local_dir: str) -> None:
    """Remove anything that would trip the validator's submission gates:
    `auto_map` in config.json, any `*.py` modeling code.
    """
    cfg_path = Path(local_dir) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if "auto_map" in cfg:
            log.warning("stripping auto_map from config.json")
            del cfg["auto_map"]
            cfg_path.write_text(json.dumps(cfg, indent=2))
    for py in Path(local_dir).rglob("*.py"):
        log.warning("removing modeling file: %s", py.name)
        py.unlink()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-repo", default=SOURCE_REPO,
                   help=f"Source HuggingFace repo id (default: {SOURCE_REPO})")
    p.add_argument("--hf", dest="source_repo", help=argparse.SUPPRESS)
    p.add_argument("--target-repo", default=TARGET_REPO,
                   help=f"Target repo id (default: {TARGET_REPO})")
    p.add_argument("--hippius", dest="target_repo", help=argparse.SUPPRESS)
    p.add_argument("--repo-backend", choices=("hf", "hippius"),
                   default=TARGET_BACKEND,
                   help=f"upload backend for {TARGET_REPO} (default: {TARGET_BACKEND})")
    p.add_argument("--revision", default=None, help="HF revision (default: main / latest)")
    p.add_argument("--workdir", default="/tmp/teutonic/seed",
                   help="Scratch dir for download + scrub")
    p.add_argument("--public", action="store_true",
                   help="when --repo-backend=hf, create a public repo")
    args = p.parse_args()

    work = Path(args.workdir) / args.source_repo.replace("/", "_")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    log.info("downloading %s @ %s -> %s", args.source_repo, args.revision or "main", work)
    snapshot_download(
        repo_id=args.source_repo,
        revision=args.revision,
        local_dir=str(work),
        allow_patterns=["*.safetensors", "config.json", "model.safetensors.index.json",
                        "tokenizer*", "*.json", "*.model", "*.txt"],
        max_workers=8,
        token=os.environ.get("HF_TOKEN"),
    )

    _scrub(str(work))

    sizes = sum(p.stat().st_size for p in work.rglob("*.safetensors"))
    log.info("safetensors bytes: %.2f GiB", sizes / (1 << 30))

    if args.repo_backend == "hippius":
        log.info("uploading -> Hippius %s", args.target_repo)
        ref = upload_model_folder(
            str(work),
            args.target_repo,
            commit_message=f"genesis from {args.source_repo}",
        )
        seed_repo = ref.repo
        seed_digest = ref.digest
        log.info("uploaded: %s", ref.immutable_ref)
    else:
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        private = not args.public
        log.info("creating/updating HF repo %s (private=%s)", args.target_repo, private)
        api.create_repo(args.target_repo, exist_ok=True, private=private, repo_type="model")
        log.info("uploading -> HF %s", args.target_repo)
        api.upload_folder(
            folder_path=str(work),
            repo_id=args.target_repo,
            commit_message=f"genesis from {args.source_repo}",
            allow_patterns=["*.safetensors", "config.json", "model.safetensors.index.json",
                            "tokenizer*", "*.json", "*.model", "*.txt"],
        )
        info = api.model_info(args.target_repo)
        seed_repo = args.target_repo
        seed_digest = f"hf:{info.sha}"
        log.info("uploaded: %s@%s", seed_repo, seed_digest)
    print()
    print("=" * 60)
    print(f"chain.toml [chain].seed_repo      = \"{seed_repo}\"")
    print(f"chain.toml [seed].repo_backend    = \"{args.repo_backend}\"")
    print(f"chain.toml [seed].seed_digest     = \"{seed_digest}\"")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
