#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.access.contracts import Manifest, ReadySignal, ready_signal_payload
from teutonic.access.crypto import verify_hotkey_signature

from miner.common import (
    AUTH_FILE,
    MANIFEST_FILE,
    REGISTRATION_FILE,
    add_wallet_arguments,
    load_registration,
    require_current_registration,
    state_dir_from_args,
    subtensor_connection,
    wallet_from_args,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate the finalized hotkey registration and commit the uploaded manifest's "
            "ready identity on chain."
        )
    )
    parser.add_argument("--competition", choices=("main", "math", "code", "text"), default="main")
    add_wallet_arguments(parser)
    return parser.parse_args(argv)


def load_manifest(path, state) -> Manifest:
    try:
        manifest = Manifest.from_bytes(path.read_bytes())
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing uploaded manifest state: {path}") from exc
    if manifest.registration_id != state.registration_id or manifest.hotkey != state.hotkey:
        raise RuntimeError("manifest belongs to a different finalized registration")
    verify_hotkey_signature(state.hotkey, manifest.signing_payload(), manifest.signature)
    return manifest


def remove_local_auth(state_dir) -> None:
    try:
        (state_dir / AUTH_FILE).unlink()
    except FileNotFoundError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wallet = wallet_from_args(args)
    state_dir = state_dir_from_args(args, wallet)
    state = load_registration(state_dir, wallet)
    manifest = load_manifest(state_dir / MANIFEST_FILE, state)
    payload = ready_signal_payload(state.registration_id, manifest.manifest_sha256, args.competition)

    with subtensor_connection(state.network) as subtensor:
        current_state = require_current_registration(subtensor, saved=state, wallet=wallet)
        current_state.save(state_dir / REGISTRATION_FILE)
        current = str(subtensor.get_commitment(state.netuid, state.uid) or "")
        if current.startswith("r2ready:v1"):
            existing = ReadySignal.parse(
                current,
                signalling_hotkey=state.hotkey,
                block_number=0,
                extrinsic_index=0,
                event_index=0,
            )
            if (
                existing.registration_id == state.registration_id
                and existing.manifest_sha256 == manifest.manifest_sha256
                and existing.competition == args.competition
            ):
                remove_local_auth(state_dir)
                print("matching ready commitment already exists")
                return 0
            if existing.registration_id == state.registration_id:
                raise RuntimeError("registration already committed a different model manifest or competition")

        print("Submitting ready commitment and waiting for finalization", flush=True)
        result = subtensor.set_commitment(
            wallet=wallet,
            netuid=state.netuid,
            data=payload,
            raise_error=True,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        if not result.success:
            raise RuntimeError(f"ready commitment failed: {result.message}")

    remove_local_auth(state_dir)
    print(f"ready_finalized registration_id={state.registration_id}")
    print("local upload credential removed; validator processing is asynchronous")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
