#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from teutonic.access.crypto import MailboxCipher
from teutonic.config import DEFAULT_MAILBOX_PUBLIC_BASE_URL
from teutonic.credentials import latest_mailbox_object_key, mailbox_object_key

from miner.common import (
    AUTH_FILE,
    add_wallet_arguments,
    env_or_none,
    load_registration,
    signing_key,
    state_dir_from_args,
    wallet_from_args,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll the public mailbox, verify/decrypt this hotkey's scoped R2 credential, "
            "and save it locally with mode 0600."
        )
    )
    add_wallet_arguments(parser)
    parser.add_argument(
        "--mailbox-base-url",
        default=(
            env_or_none("TEUTONIC_MAILBOX_PUBLIC_BASE_URL") or DEFAULT_MAILBOX_PUBLIC_BASE_URL
        ),
    )
    parser.add_argument("--generation", type=int, help="pin a generation; default: latest")
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args(argv)


def fetch_mailbox(
    base_url: str,
    key: str,
    *,
    timeout: int,
    on_not_found: Callable[[], None] | None = None,
) -> bytes:
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        attempt = 0
        while True:
            response = client.get(
                f"{base_url.rstrip('/')}/{key}",
                params={"poll": f"{attempt}-{uuid.uuid4().hex}"},
            )
            if response.status_code == 200:
                return response.content
            if response.status_code != 404:
                response.raise_for_status()
            if on_not_found is not None:
                on_not_found()
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for the encrypted mailbox credential")
            attempt += 1
            time.sleep(2)


class ExpiredCredential(RuntimeError):
    pass


def validate_envelope(envelope: dict, state, generation: int | None) -> datetime:
    observed_generation = envelope.get("credential_generation")
    if type(observed_generation) is not int or observed_generation < 1:
        raise RuntimeError("mailbox credential contains an invalid credential_generation")
    expected = {
        "protocol_version": 1,
        "signature_scheme": "ed25519",
        "netuid": state.netuid,
        "uid": state.uid,
        "hotkey": state.hotkey,
        "registration_id": state.registration_id,
        "credential_generation": generation if generation is not None else observed_generation,
        "chain_generation": state.chain_generation,
        "registration_block": state.registration_block,
        "allowed_prefix": f"models/registrations/{state.registration_id}/",
        "credential_scope": "object-read-write",
        "revocation_event": "finalized_ready_signal",
        "submission_policy": "one_per_hotkey",
    }
    for field, value in expected.items():
        if envelope.get(field) != value:
            raise RuntimeError(f"mailbox credential contains an unexpected {field}")
    for field in (
        "validator_identity",
        "r2_endpoint",
        "private_model_bucket",
        "access_key_id",
        "secret_access_key",
        "session_token",
        "expires_at",
        "validator_signature",
    ):
        if not isinstance(envelope.get(field), str) or not envelope[field]:
            raise RuntimeError(f"mailbox credential is missing {field}")
    expires_at = datetime.fromisoformat(envelope["expires_at"].replace("Z", "+00:00"))
    if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
        raise ExpiredCredential("mailbox credential is already expired")
    return expires_at


def fetch_latest_credentials(
    base_url: str,
    state,
    miner_key,
    *,
    timeout: int,
    on_wait: Callable[[], None] | None = None,
) -> dict:
    """Discover and verify the latest envelope, waiting for overdue renewal."""
    deadline = time.monotonic() + timeout
    keys = (
        latest_mailbox_object_key(state.registration_id),
        mailbox_object_key(state.registration_id, 1),
    )
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        while True:
            for key in keys:
                response = client.get(
                    f"{base_url.rstrip('/')}/{key}",
                    params={"poll": uuid.uuid4().hex},
                    headers={"Cache-Control": "no-cache"},
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                envelope = MailboxCipher.decrypt_for_miner(response.content, miner_key)
                try:
                    validate_envelope(envelope, state, None if key == keys[0] else 1)
                except ExpiredCredential:
                    break
                return envelope
            if on_wait is not None:
                on_wait()
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for renewed upload credentials")
            time.sleep(2)


def main(
    argv: list[str] | None = None,
    *,
    on_mailbox_not_found: Callable[[], None] | None = None,
) -> int:
    args = parse_args(argv)
    if not args.mailbox_base_url:
        raise RuntimeError("--mailbox-base-url is required")
    if (args.generation is not None and args.generation < 1) or args.timeout < 1:
        raise RuntimeError("generation and timeout must be positive")
    wallet = wallet_from_args(args)
    state_dir = state_dir_from_args(args, wallet)
    state = load_registration(state_dir, wallet)
    if args.generation is None:
        envelope = fetch_latest_credentials(
            args.mailbox_base_url,
            state,
            signing_key(wallet),
            timeout=args.timeout,
            on_wait=on_mailbox_not_found,
        )
    else:
        key = mailbox_object_key(state.registration_id, args.generation)
        ciphertext = fetch_mailbox(
            args.mailbox_base_url,
            key,
            timeout=args.timeout,
            on_not_found=on_mailbox_not_found,
        )
        envelope = MailboxCipher.decrypt_for_miner(ciphertext, signing_key(wallet))
    expires_at = validate_envelope(envelope, state, args.generation)
    auth_path = state_dir / AUTH_FILE
    write_json(auth_path, envelope, secret=True)
    print(f"upload_endpoint={envelope['r2_endpoint']}")
    print(f"upload_bucket={envelope['private_model_bucket']}")
    print(f"upload_prefix={envelope['allowed_prefix']}")
    print(f"credential_expires_at={expires_at.isoformat()}")
    print(f"credential_generation={envelope['credential_generation']}")
    print(f"credential_file={auth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
