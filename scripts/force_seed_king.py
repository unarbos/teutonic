#!/usr/bin/env python3
"""Prepare a one-time reset to the seed king configured in chain.toml.

The validator must be stopped before applying this operation. The script backs
up each R2 object it changes, then clears the persisted current king. On the
next validator start, the normal startup path installs the configured seed
using the validator hotkey and current chain block.

Dry run:
    python scripts/force_seed_king.py

Apply while the validator is stopped:
    pm2 stop teutonic-validator
    python scripts/force_seed_king.py --apply --validator-stopped
    pm2 start teutonic-validator

R2 credentials may use either the TEUTONIC_R2_* names from the validator or
the R2_* names commonly exported by Doppler.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chain_config  # noqa: E402


KING_KEY = "king/current.json"
STATE_KEY = "state/validator_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def r2_settings() -> dict[str, str]:
    settings = {
        "endpoint": first_env("TEUTONIC_R2_ENDPOINT", "R2_URL"),
        "bucket": first_env("TEUTONIC_R2_BUCKET", "R2_BUCKET_NAME"),
        "access_key": first_env("TEUTONIC_R2_ACCESS_KEY", "R2_ACCESS_KEY_ID"),
        "secret_key": first_env("TEUTONIC_R2_SECRET_KEY", "R2_SECRET_ACCESS_KEY"),
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"missing R2 settings: {names}")
    return settings


def make_client(settings: dict[str, str]):
    return boto3.client(
        "s3",
        endpoint_url=settings["endpoint"],
        aws_access_key_id=settings["access_key"],
        aws_secret_access_key=settings["secret_key"],
        region_name="auto",
        config=BotoConfig(
            connect_timeout=15,
            read_timeout=45,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def get_json(client, bucket: str, key: str) -> dict:
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404", "NotFound"}:
            return {}
        raise
    data = json.loads(body)
    if not isinstance(data, dict):
        raise TypeError(f"{key} must contain a JSON object, got {type(data).__name__}")
    return data


def put_json(client, bucket: str, key: str, data: dict) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, default=str, sort_keys=True).encode(),
        ContentType="application/json",
    )


def cleared_validator_state(existing: dict, requested_at: str) -> dict:
    state = dict(existing)
    state["king"] = {}
    state["last_weight_block"] = 0
    state["last_winner_hotkey"] = None
    state["updated_at"] = requested_at
    return state


def describe_king(king: dict) -> str:
    if not king:
        return "none"
    return (
        f"repo={king.get('model_repo', '?')} "
        f"digest={(king.get('king_digest') or '')[:19]} "
        f"hotkey={(king.get('hotkey') or '')[:16]}..."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write backups and clear current-king state. Default is dry-run.",
    )
    parser.add_argument(
        "--validator-stopped",
        action="store_true",
        help="Acknowledge that teutonic-validator is stopped (required with --apply).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.apply and not args.validator_stopped:
        raise SystemExit("--apply requires --validator-stopped")

    settings = r2_settings()
    client = make_client(settings)
    current_king = get_json(client, settings["bucket"], KING_KEY)
    validator_state = get_json(client, settings["bucket"], STATE_KEY)
    state_king = validator_state.get("king") or {}
    requested_at = utc_now()
    cleared_state = cleared_validator_state(validator_state, requested_at)

    backup_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_prefix = f"backups/force-seed/{backup_stamp}"
    king_backup_key = f"{backup_prefix}/king-current.json"
    state_backup_key = f"{backup_prefix}/validator-state.json"

    print(f"R2: {settings['endpoint']} / {settings['bucket']}")
    print(f"Configured seed: {chain_config.SEED_REPO}@{chain_config.SEED_DIGEST}")
    print(f"Current {KING_KEY}: {describe_king(current_king)}")
    print(f"Embedded state king: {describe_king(state_king)}")
    print(f"Backup prefix: {backup_prefix}")

    if not args.apply:
        print("\nDry run only; no R2 objects were changed.")
        print("To apply, stop PM2 and rerun with --apply --validator-stopped.")
        return

    put_json(client, settings["bucket"], king_backup_key, current_king)
    put_json(client, settings["bucket"], state_backup_key, validator_state)
    put_json(client, settings["bucket"], KING_KEY, {})
    put_json(client, settings["bucket"], STATE_KEY, cleared_state)

    print("\nSeed reset prepared.")
    print(f"Backed up {KING_KEY} to {king_backup_key}")
    print(f"Backed up {STATE_KEY} to {state_backup_key}")
    print("Start the validator; it will install the configured seed king once.")


if __name__ == "__main__":
    main()
