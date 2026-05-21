#!/usr/bin/env python3
"""Teutonic miner — build a challenger and submit it on-chain.

Downloads the current king, produces a challenger by perturbing every float
tensor with low-amplitude Gaussian noise, uploads to Hippius Hub, and posts a
v3 reveal commitment binding (king_digest, challenger_repo, challenger_digest).

Noise perturbation will not clear `delta` on a mature king — it is a pipeline
test stub, not a strategy. Real dethrones come from real fine-tuning (LoRA,
full fine-tune, distillation, whatever) against the pinned king. The protocol
does not prescribe the training recipe; swap the perturb step for an actual
training loop and the rest of this script applies unchanged.
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import bittensor as bt
import httpx
import torch
from safetensors.torch import load_file, save_file

# chain_config sits at the repo root; ensure it imports regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_config  # noqa: E402
from model_store import (  # noqa: E402
    ModelRef,
    build_reveal_v3,
    materialize_model,
    upload_model_folder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("miner")

DASHBOARD_URL = os.environ.get("TEUTONIC_DASHBOARD_URL",
    "https://us-east-1.hippius.com/teutonic-sn3/dashboard.json")
SEED_REPO = os.environ.get("TEUTONIC_SEED_REPO", chain_config.SEED_REPO)
SEED_DIGEST = os.environ.get("TEUTONIC_SEED_DIGEST", getattr(chain_config, "SEED_DIGEST", ""))
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")

REPO_PATTERN = re.compile(os.environ.get("TEUTONIC_REPO_PATTERN", chain_config.REPO_PATTERN))

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


def main():
    parser = argparse.ArgumentParser(description="Teutonic miner")
    parser.add_argument("--hotkey", default="h0", help="Wallet hotkey name")
    parser.add_argument("--noise", type=float, default=0.001, help="Noise scale for weight perturbation")
    parser.add_argument("--suffix", default=None, help="Challenger repo suffix (default: hotkey name)")
    parser.add_argument("--force", action="store_true", help="Bypass soft warnings (hotkey seen, king hotkey)")
    args = parser.parse_args()

    suffix = args.suffix or args.hotkey
    namespace = chain_config.SEED_NAMESPACE or "unconst"
    challenger_repo = f"{namespace}/{chain_config.NAME}-{suffix}"

    log.info("miner starting | hotkey=%s repo=%s noise=%.4f", args.hotkey, challenger_repo, args.noise)

    # Pre-flight: repo name pattern
    if not REPO_PATTERN.match(challenger_repo):
        log.error("repo name %s does not match required pattern %s", challenger_repo, REPO_PATTERN.pattern)
        sys.exit(1)

    # Connect to chain
    wallet = bt.Wallet(name=WALLET_NAME, hotkey=args.hotkey)
    subtensor = bt.Subtensor(network=NETWORK)
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
    king_digest = SEED_DIGEST
    dashboard = None
    try:
        resp = httpx.get(DASHBOARD_URL, timeout=15)
        resp.raise_for_status()
        dashboard = resp.json()
        king = dashboard["king"]
        king_repo = king["model_repo"]
        # Dashboard publishes `king_digest`; older dashboards (pre-v3 cutover)
        # used `model_digest`. Fall back to that for back-compat.
        king_digest = king.get("king_digest") or king.get("model_digest")
        log.info("discovered king from dashboard: %s@%s", king_repo, (king_digest or "")[:19])
    except Exception:
        log.warning("could not fetch dashboard, falling back to seed model %s", SEED_REPO)
    if not king_digest:
        log.error("no king digest available; set TEUTONIC_SEED_DIGEST or wait for dashboard")
        sys.exit(1)
    king_ref = ModelRef(king_repo, king_digest)

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

    # Download king model at the immutable Hippius digest.
    king_dir = "/tmp/teutonic/miner/king"
    if os.path.exists(king_dir):
        shutil.rmtree(king_dir)
    log.info("downloading king from %s", king_ref.immutable_ref)
    materialize_model(king_ref, local_dir=king_dir, max_workers=16)

    # Create challenger by perturbing weights (stub; replace with real training).
    challenger_dir = f"/tmp/teutonic/miner/challenger-{suffix}"
    if os.path.exists(challenger_dir):
        shutil.rmtree(challenger_dir)
    shutil.copytree(king_dir, challenger_dir)

    st_files = sorted(Path(challenger_dir).glob("*.safetensors"))

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

    rejection = validate_local_config(king_dir, challenger_dir)
    if rejection:
        log.error("config validation failed: %s", rejection)
        sys.exit(1)
    log.info("config validation passed")

    log.info("uploading to Hippius Hub repo %s", challenger_repo)
    challenger_ref = upload_model_folder(
        challenger_dir,
        repo=challenger_repo,
        revision=suffix,
        commit_message=f"Challenger from {args.hotkey} (noise={args.noise})",
    )
    log.info("uploaded to %s", challenger_ref.immutable_ref)

    # On-chain reveal: v3|king_digest|repo|challenger_digest|author_hotkey.
    # Both digests are bare 64-hex sha256; king_digest is pulled straight from
    # the dashboard (full digest, not a truncation).
    payload = build_reveal_v3(king_digest, challenger_ref, my_hotkey)
    log.info("submitting reveal: %s", payload)

    resp = subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=NETUID,
        data=payload,
        blocks_until_reveal=3,
        wait_for_revealed_execution=False,
    )

    if resp.success:
        log.info("reveal committed: %s", resp.message)
        log.info("the validator will pick this up once revealed (~30 seconds)")
    else:
        log.error("reveal commitment failed: %s", resp.message)
        sys.exit(1)

    log.info("done!")


if __name__ == "__main__":
    main()
