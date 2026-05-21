#!/usr/bin/env python3
"""Set the validator hotkey's weights to 100% UID 0 on a wall-clock cadence.

This is intentionally simpler than the scoring validator and any block-height
watchdog: it never looks at challengers or prior winners, and it never derives
weights from chain state beyond verifying that the target UID exists.

Env knobs:
  TEUTONIC_NETWORK          chain (default "finney")
  TEUTONIC_NETUID           subnet (default 3)
  BT_WALLET_NAME            coldkey name (default "teutonic")
  BT_WALLET_HOTKEY          hotkey name (default "default")
  BURN_UID                  target uid (default 0)
  BURN_INTERVAL_SECONDS     seconds between successful sets (default 3600)
  BURN_RPC_RETRY_DELAY      backoff on RPC errors (default 60s)
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time

import bittensor as bt

NETWORK = os.environ.get("TEUTONIC_NETWORK", "finney")
NETUID = int(os.environ.get("TEUTONIC_NETUID", "3"))
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "teutonic")
WALLET_HOTKEY = os.environ.get("BT_WALLET_HOTKEY", "default")
BURN_UID = int(os.environ.get("BURN_UID", "0"))
INTERVAL_SECONDS = int(os.environ.get("BURN_INTERVAL_SECONDS", "3600"))
RPC_RETRY_DELAY = int(os.environ.get("BURN_RPC_RETRY_DELAY", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hourly-burn")

_stop = False


def _handle_signal(signum, _frame) -> None:
    global _stop
    log.info("received signal %s; shutting down after current cycle", signum)
    _stop = True


def _connect() -> bt.subtensor:
    log.info("connecting to %s ...", NETWORK)
    return bt.subtensor(network=NETWORK)


def _close_subtensor(subtensor: bt.subtensor) -> None:
    try:
        subtensor.close()
    except Exception:
        pass


def _verify_target(subtensor: bt.subtensor) -> bool:
    """Sanity-check BURN_UID exists in the metagraph; log who receives weight."""
    try:
        meta = subtensor.metagraph(NETUID)
    except Exception:
        log.exception("metagraph fetch failed")
        return False

    hotkeys = list(getattr(meta, "hotkeys", []))
    if BURN_UID < 0 or BURN_UID >= len(hotkeys):
        log.error("BURN_UID %d out of range (metagraph size=%d)", BURN_UID, len(hotkeys))
        return False

    log.info("burn target: uid=%d hotkey=%s", BURN_UID, hotkeys[BURN_UID])
    return True


def _set_burn_weight(subtensor: bt.subtensor, wallet: bt.wallet) -> bool:
    """Push exactly one UID with 100% weight."""
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

    ok, msg = (result if isinstance(result, tuple) else (bool(result), ""))
    if ok:
        log.info("weights set: uid=%d weight=1.0 (%s)", BURN_UID, msg or "ok")
    else:
        log.error("weights NOT set: uid=%d msg=%s", BURN_UID, msg or "<no msg>")
    return bool(ok)


def _sleep_interruptibly(seconds: int) -> None:
    deadline = time.monotonic() + max(0, seconds)
    while not _stop:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info(
        "hourly-burn starting: network=%s netuid=%d wallet=%s/%s burn_uid=%d "
        "interval=%ds",
        NETWORK, NETUID, WALLET_NAME, WALLET_HOTKEY, BURN_UID, INTERVAL_SECONDS,
    )

    wallet = bt.wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    log.info("loaded wallet hotkey=%s", wallet.hotkey.ss58_address)

    subtensor = _connect()
    try:
        if not _verify_target(subtensor):
            return 2

        while not _stop:
            if _set_burn_weight(subtensor, wallet):
                log.info("next weight set in %ds", INTERVAL_SECONDS)
                _sleep_interruptibly(INTERVAL_SECONDS)
                continue

            _close_subtensor(subtensor)
            _sleep_interruptibly(RPC_RETRY_DELAY)
            if _stop:
                break
            try:
                subtensor = _connect()
            except Exception:
                log.exception("reconnect failed; will retry")

        log.info("exited cleanly")
        return 0
    finally:
        _close_subtensor(subtensor)


if __name__ == "__main__":
    sys.exit(main())
