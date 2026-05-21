#!/usr/bin/env python3
"""Submit a pre-built challenger to the chain.

Reads a verdict.json produced by train_challenger.py (king_hash,
uploaded_repo, uploaded_digest) and posts the bittensor reveal commitment
in the form `v2|{king_hash[:16]}|{repo}|sha256:{manifest_digest}`.

The OCI manifest digest is the immutable commitment to the file tree.

Run this on the templar host where the wallet lives.

IMPORTANT — coldkey gate (added 2026-04-29):
    The validator REJECTS any Hippius repo whose name does NOT contain the
    first 8 ss58 chars of your **coldkey** (case-insensitive substring
    match against the full "<account>/<basename>" string).

    This stops anyone from re-revealing somebody else's Hippius URL under
    their own hotkey: only YOU know your coldkey, and an imposter who
    lifts your URL ends up advertising YOUR coldkey on chain — which is
    self-incriminating.

    So an Hippius repo like:
        my-team/<chain.name>-5DhAqMpd-v3
                             ^^^^^^^^
    works (matches my coldkey 5DhAqMpd...). Without that prefix, the
    validator will record your eval as `coldkey_required` and skip it.

    This script will refuse to broadcast a reveal whose repo doesn't
    contain your coldkey prefix — fail fast locally rather than burn
    a tx for nothing.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import bittensor as bt

from model_store import ModelRef, build_reveal_payload

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [submit] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("submit_challenger")

# Must match validator.py's COLDKEY_PREFIX_LEN. If the validator side ever
# changes this, miners need to update too.
COLDKEY_PREFIX_LEN = 8


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
    repo = v.get("uploaded_repo") or v.get("model_repo")
    digest = v.get("uploaded_digest") or v.get("model_digest")
    if not repo or not digest:
        log.error("verdict missing uploaded_repo / uploaded_digest — train script must upload to Hippius Hub")
        sys.exit(2)
    try:
        model_ref = ModelRef(repo, digest)
    except ValueError as exc:
        log.error("invalid Hippius model ref: %s", exc)
        sys.exit(2)
    if not v["best"]["accepted"]:
        log.error("offline eval rejected (mu_hat=%.6f, lcb=%.6f, delta=%.6f) — refusing to burn TAO",
                  v["best"]["mu_hat"], v["best"]["lcb"], v["best"]["delta"])
        sys.exit(3)

    wallet = bt.wallet(name=args.wallet_name, hotkey=args.hotkey)
    log.info("wallet hotkey: %s", wallet.hotkey.ss58_address)

    coldkey_ss58 = wallet.coldkeypub.ss58_address
    expected_prefix = coldkey_ss58[:COLDKEY_PREFIX_LEN]
    if expected_prefix.lower() not in repo.lower():
        log.error(
            "Hippius repo '%s' does NOT contain your coldkey prefix '%s' "
            "(first %d chars of %s).\n"
            "    The validator will reject this submission with "
            "`coldkey_required` and your tx will be wasted.\n"
            "    Rename your Hippius repo or Hippius namespace so its full id "
            "contains '%s' (case-insensitive substring) anywhere — e.g.\n"
            "        %s/<chain.name>-%s-v1\n"
            "    then re-upload and rerun this script.",
            repo, expected_prefix, COLDKEY_PREFIX_LEN, coldkey_ss58,
            expected_prefix,
            repo.split("/", 1)[0] if "/" in repo else "<your-hippius-namespace>",
            expected_prefix,
        )
        sys.exit(6)
    log.info("coldkey gate ok: repo '%s' contains coldkey prefix '%s'",
             repo, expected_prefix)

    payload = build_reveal_payload(king_hash, model_ref)
    log.info("payload: %s", payload)

    if args.dry_run:
        log.info("[dry-run] not submitting")
        return

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
