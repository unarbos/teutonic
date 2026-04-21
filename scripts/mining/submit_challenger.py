#!/usr/bin/env python3
"""Submit a pre-built challenger to the chain.

Reads a verdict.json produced by train_challenger.py (which contains the
king_hash, uploaded HF repo, uploaded revision, challenger_hash) and posts
the bittensor reveal commitment.

Run this on the templar host where the wallet lives.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import bittensor as bt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [submit] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("submit_challenger")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", required=True,
                    help="Path to verdict.json from train_challenger.py")
    ap.add_argument("--hotkey", default="h0")
    ap.add_argument("--wallet-name", default="teutonic")
    ap.add_argument("--netuid", type=int, default=3)
    ap.add_argument("--network", default="finney")
    ap.add_argument("--blocks-until-reveal", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    v = json.loads(Path(args.verdict).read_text())
    king_hash = v["king_hash"][:16]
    repo = v.get("uploaded_repo")
    chall_hash = v.get("challenger_hash")
    if not repo or not chall_hash:
        log.error("verdict missing uploaded_repo / challenger_hash — train script must run with --upload-repo and accept the model")
        sys.exit(2)
    if not v["best"]["accepted"]:
        log.error("offline eval rejected (mu_hat=%.6f, lcb=%.6f, delta=%.6f) — refusing to burn TAO",
                  v["best"]["mu_hat"], v["best"]["lcb"], v["best"]["delta"])
        sys.exit(3)

    payload = f"{king_hash}:{repo}:{chall_hash}"
    log.info("payload: %s", payload)

    if args.dry_run:
        log.info("[dry-run] not submitting")
        return

    wallet = bt.wallet(name=args.wallet_name, hotkey=args.hotkey)
    log.info("wallet hotkey: %s", wallet.hotkey.ss58_address)

    sub = bt.subtensor(network=args.network)
    meta = sub.metagraph(args.netuid)
    if wallet.hotkey.ss58_address not in meta.hotkeys:
        log.error("hotkey not registered on netuid %d", args.netuid)
        sys.exit(4)
    uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
    log.info("registered as uid=%d", uid)

    ok, block = sub.set_reveal_commitment(
        wallet=wallet, netuid=args.netuid,
        data=payload, blocks_until_reveal=args.blocks_until_reveal,
    )
    if ok:
        log.info("reveal committed at block %d -- validator should pick up after reveal", block)
    else:
        log.error("commitment failed")
        sys.exit(5)


if __name__ == "__main__":
    main()
