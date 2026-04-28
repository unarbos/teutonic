#!/usr/bin/env python3
"""Teutonic miner — create a challenger model and submit on-chain.

1. Downloads the current king model
2. Applies a small random perturbation to the weights
3. Uploads as a challenger repo on HuggingFace
4. Computes SHA256 of the safetensors
5. Submits a reveal commitment on Bittensor chain
"""
import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import bittensor as bt
import httpx
import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("miner")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DASHBOARD_URL = os.environ.get("TEUTONIC_DASHBOARD_URL",
    "https://s3.hippius.com/teutonic-sn3/dashboard.json")
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", "unconst/Teutonic-VIII")
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")

REPO_PATTERN = re.compile(r"^[^/]+/Teutonic-VIII-.+$")

CONFIG_MATCH_KEYS = (
    "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "intermediate_size", "model_type",
)


def validate_local_config(king_dir: str, challenger_dir: str) -> str | None:
    """Compare king and challenger config.json locally.

    Returns None if OK, or a human-readable rejection reason.
    """
    king_cfg_path = Path(king_dir) / "config.json"
    chall_cfg_path = Path(challenger_dir) / "config.json"

    if not king_cfg_path.exists():
        return None  # can't validate without king config

    if not chall_cfg_path.exists():
        return "challenger config.json missing"

    with open(king_cfg_path) as f:
        king_cfg = json.load(f)
    with open(chall_cfg_path) as f:
        chall_cfg = json.load(f)

    king_arch = king_cfg.get("architectures", [])
    chall_arch = chall_cfg.get("architectures", [])
    if king_arch and chall_arch and king_arch != chall_arch:
        return f"architecture mismatch: king={king_arch} challenger={chall_arch}"

    for key in CONFIG_MATCH_KEYS:
        king_val = king_cfg.get(key)
        chall_val = chall_cfg.get(key)
        if king_val is not None and chall_val is not None and king_val != chall_val:
            return f"{key} mismatch: king={king_val} challenger={chall_val}"

    st_files = list(Path(challenger_dir).glob("*.safetensors"))
    if not st_files:
        return "no .safetensors files in challenger"

    return None


def sha256_dir(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()



def main():
    parser = argparse.ArgumentParser(description="Teutonic miner")
    parser.add_argument("--hotkey", default="h0", help="Wallet hotkey name")
    parser.add_argument("--noise", type=float, default=0.001, help="Noise scale for weight perturbation")
    parser.add_argument("--suffix", default=None, help="Challenger repo suffix (default: hotkey name)")
    parser.add_argument("--force", action="store_true", help="Bypass soft warnings (hotkey seen, king hotkey)")
    args = parser.parse_args()

    suffix = args.suffix or args.hotkey
    challenger_repo = f"unconst/Teutonic-VIII-{suffix}"

    log.info("miner starting | hotkey=%s repo=%s noise=%.4f", args.hotkey, challenger_repo, args.noise)

    # Pre-flight: repo name pattern
    if not REPO_PATTERN.match(challenger_repo):
        log.error("repo name %s does not match required pattern %s", challenger_repo, REPO_PATTERN.pattern)
        sys.exit(1)

    # Connect to chain
    wallet = bt.wallet(name=WALLET_NAME, hotkey=args.hotkey)
    subtensor = bt.subtensor(network=NETWORK)
    my_hotkey = wallet.hotkey.ss58_address
    log.info("wallet: %s", my_hotkey)

    # Pre-flight: check hotkey is registered on the subnet
    try:
        meta = subtensor.metagraph(NETUID)
        if my_hotkey not in meta.hotkeys:
            log.error("hotkey %s is NOT registered on subnet %d — register before mining", my_hotkey[:16], NETUID)
            sys.exit(1)
        uid = meta.hotkeys.index(my_hotkey)
        log.info("hotkey registered as uid=%d on subnet %d", uid, NETUID)
    except Exception:
        log.warning("could not query metagraph — skipping registration check")

    # Discover current king from dashboard
    king_repo = SEED_REPO
    king_revision = None
    dashboard = None
    try:
        resp = httpx.get(DASHBOARD_URL, timeout=15)
        resp.raise_for_status()
        dashboard = resp.json()
        king_repo = dashboard["king"]["hf_repo"]
        king_revision = dashboard["king"].get("king_revision") or None
        log.info("discovered king from dashboard: %s@%s",
                 king_repo, king_revision[:12] if king_revision else "HEAD")
    except Exception:
        log.warning("could not fetch dashboard, falling back to seed repo %s", SEED_REPO)

    # Pre-flight: check if our hotkey is the current king
    if dashboard:
        king_hotkey = dashboard.get("king", {}).get("hotkey", "")
        if king_hotkey and my_hotkey == king_hotkey:
            log.warning("your hotkey %s is the CURRENT KING — validator will skip your challenge", my_hotkey[:16])
            if not args.force:
                log.error("aborting (use --force to override)")
                sys.exit(1)
            log.warning("--force set, continuing anyway")

    # Pre-flight: check if hotkey already has a reveal (validator tracks seen hotkeys)
    try:
        all_reveals = subtensor.get_all_revealed_commitments(NETUID)
        if all_reveals and my_hotkey in all_reveals:
            log.warning("hotkey %s already has an existing reveal on-chain — "
                        "validator may skip unless running with --no-seen", my_hotkey[:16])
            if not args.force:
                log.error("aborting (use --force to override)")
                sys.exit(1)
            log.warning("--force set, continuing anyway")
    except Exception:
        log.warning("could not check existing reveals — skipping seen-hotkey check")

    # Download king model at pinned revision
    king_dir = "/tmp/teutonic/miner/king"
    if os.path.exists(king_dir):
        shutil.rmtree(king_dir)
    log.info("downloading king from %s@%s", king_repo, (king_revision or "HEAD")[:12])
    snapshot_download(king_repo, local_dir=king_dir, token=HF_TOKEN or None,
                      revision=king_revision)
    king_hash = sha256_dir(king_dir)
    log.info("king hash: %s", king_hash[:16])

    # Create challenger by perturbing weights
    challenger_dir = f"/tmp/teutonic/miner/challenger-{suffix}"
    if os.path.exists(challenger_dir):
        shutil.rmtree(challenger_dir)
    shutil.copytree(king_dir, challenger_dir)

    st_files = sorted(Path(challenger_dir).glob("*.safetensors"))
    rng = np.random.default_rng(int(time.time()))

    for st_file in st_files:
        log.info("perturbing %s", st_file.name)
        sd = load_file(str(st_file))
        new_sd = {}
        for name, tensor in sd.items():
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                noise = torch.randn_like(tensor.float()) * args.noise
                new_sd[name] = (tensor.float() + noise).to(tensor.dtype)
            else:
                new_sd[name] = tensor
        save_file(new_sd, str(st_file))

    challenger_hash = sha256_dir(challenger_dir)
    log.info("challenger hash: %s", challenger_hash[:16])

    # Pre-flight: validate challenger config matches king
    rejection = validate_local_config(king_dir, challenger_dir)
    if rejection:
        log.error("config validation failed: %s", rejection)
        sys.exit(1)
    log.info("config validation passed")

    # Upload to HuggingFace
    api = HfApi(token=HF_TOKEN)
    api.create_repo(challenger_repo, exist_ok=True, private=False)
    log.info("uploading to %s", challenger_repo)
    api.upload_folder(
        folder_path=challenger_dir,
        repo_id=challenger_repo,
        commit_message=f"Challenger from {args.hotkey} (noise={args.noise})",
        allow_patterns=["*.safetensors", "config.json", "tokenizer*", "special_tokens*"],
    )
    log.info("uploaded to https://huggingface.co/%s", challenger_repo)

    # Submit reveal commitment
    payload = f"{king_hash[:16]}:{challenger_repo}:{challenger_hash}"
    log.info("submitting reveal: %s", payload)

    success, block = subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=NETUID,
        data=payload,
        blocks_until_reveal=3,
    )

    if success:
        log.info("reveal committed at block %d", block)
        log.info("the validator will pick this up once revealed (~30 seconds)")
    else:
        log.error("reveal commitment failed")
        sys.exit(1)

    log.info("done!")


if __name__ == "__main__":
    main()
