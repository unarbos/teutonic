#!/usr/bin/env python3
"""Burn the validator's emission by routing 100% weight to a single UID.

Used as a fallback when the eval server is down (or otherwise unable to score
challengers) but we still want the validator hotkey to set weights every
tempo so it stays registered and emission flows somewhere deterministic
(UID 0 = subnet owner / burn UID by convention on SN3).

Pushes one `set_weights` RPC every WEIGHT_INTERVAL blocks (default 300, same
cadence the live validator uses). Reconnects the subtensor on RPC failure
so the daemon survives long websocket outages.

Env knobs:
  TEUTONIC_NETWORK        chain (default "finney")
  TEUTONIC_NETUID         subnet (default 3)
  BT_WALLET_NAME          coldkey name (default "teutonic")
  BT_WALLET_HOTKEY        hotkey name (default "default")
  BURN_UID                target uid (default 0)
  WEIGHT_INTERVAL         blocks between sets (default 300)
  BURN_POLL_SECONDS       chain poll cadence (default 12 = ~1 block)
  BURN_RPC_RETRY_DELAY    backoff on RPC errors (default 30s)
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Optional

import bittensor as bt

NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")
WALLET_HOTKEY = os.environ.get("BT_WALLET_HOTKEY", "default")
BURN_UID = int(os.environ.get("BURN_UID", "0"))
WEIGHT_INTERVAL = int(os.environ.get("WEIGHT_INTERVAL", "300"))
POLL_SECONDS = int(os.environ.get("BURN_POLL_SECONDS", "12"))
RPC_RETRY_DELAY = int(os.environ.get("BURN_RPC_RETRY_DELAY", "30"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("burn")

_stop = False


def _handle_signal(signum, _frame):
    global _stop
    log.info("received signal %s; shutting down after current cycle", signum)
    _stop = True


def _connect() -> bt.subtensor:
    log.info("connecting to %s ...", NETWORK)
    return bt.subtensor(network=NETWORK)


def _safe_block(subtensor: bt.subtensor) -> Optional[int]:
    try:
        return int(subtensor.block)
    except Exception:
        log.exception("failed to read current block")
        return None


def _set_burn_weight(subtensor: bt.subtensor, wallet: bt.wallet) -> bool:
    """Push 100% weight to BURN_UID. Returns True iff RPC reports success."""
    try:
        result = subtensor.set_weights(
            wallet=wallet,
            netuid=NETUID,
            uids=[BURN_UID],
            weights=[1.0],
        )
    except Exception:
        log.exception("set_weights RPC raised")
        return False

    # bittensor>=9 returns (success: bool, msg: str); older returns may differ.
    ok, msg = (result if isinstance(result, tuple) else (bool(result), ""))
    if ok:
        log.info("weights set: uid=%d weight=1.0 (%s)", BURN_UID, msg or "ok")
    else:
        log.error("weights NOT set: uid=%d msg=%s", BURN_UID, msg or "<no msg>")
    return ok


def _verify_target(subtensor: bt.subtensor) -> bool:
    """Sanity-check BURN_UID exists in the metagraph; log who we're burning to."""
    try:
        meta = subtensor.metagraph(NETUID)
    except Exception:
        log.exception("metagraph fetch failed; will retry on first set")
        return True  # don't block startup on a transient RPC issue
    n = len(getattr(meta, "hotkeys", []))
    if BURN_UID >= n:
        log.error("BURN_UID %d out of range (metagraph size=%d)", BURN_UID, n)
        return False
    log.info("burn target: uid=%d hotkey=%s", BURN_UID, meta.hotkeys[BURN_UID])
    return True


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info(
        "burn-weights starting: network=%s netuid=%d wallet=%s/%s burn_uid=%d "
        "interval=%d blocks",
        NETWORK, NETUID, WALLET_NAME, WALLET_HOTKEY, BURN_UID, WEIGHT_INTERVAL,
    )

    wallet = bt.wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    log.info("loaded wallet hotkey=%s", wallet.hotkey.ss58_address)

    subtensor = _connect()
    if not _verify_target(subtensor):
        return 2

    last_set_block: Optional[int] = None

    if _set_burn_weight(subtensor, wallet):
        last_set_block = _safe_block(subtensor)
    else:
        log.warning("startup weight-set failed; will retry on schedule")

    while not _stop:
        block = _safe_block(subtensor)
        if block is None:
            try:
                subtensor.close()
            except Exception:
                pass
            time.sleep(RPC_RETRY_DELAY)
            try:
                subtensor = _connect()
            except Exception:
                log.exception("reconnect failed; will retry")
            continue

        due = last_set_block is None or (block - last_set_block) >= WEIGHT_INTERVAL
        if due:
            if _set_burn_weight(subtensor, wallet):
                last_set_block = block
            else:
                time.sleep(RPC_RETRY_DELAY)
                continue
        else:
            remaining = WEIGHT_INTERVAL - (block - last_set_block)
            log.info("block=%d last_set=%d wait %d more blocks (~%ds)",
                     block, last_set_block, remaining, remaining * 12)

        for _ in range(POLL_SECONDS):
            if _stop:
                break
            time.sleep(1)

    log.info("exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
