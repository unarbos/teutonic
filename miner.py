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
import logging
import os
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
    "https://pub-0821b4e0d60149b79bad17376722bc75.r2.dev/dashboard.json")
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", "unconst/Teutonic-I")
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")


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
    args = parser.parse_args()

    suffix = args.suffix or args.hotkey
    challenger_repo = f"unconst/Teutonic-I-{suffix}"

    log.info("miner starting | hotkey=%s repo=%s noise=%.4f", args.hotkey, challenger_repo, args.noise)

    # Connect to chain
    wallet = bt.wallet(name=WALLET_NAME, hotkey=args.hotkey)
    subtensor = bt.subtensor(network=NETWORK)
    log.info("wallet: %s", wallet.hotkey.ss58_address)

    # Discover current king from dashboard
    king_repo = SEED_REPO
    king_revision = None
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
