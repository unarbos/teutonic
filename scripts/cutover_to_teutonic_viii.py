#!/usr/bin/env python3
"""Phase B cutover script: archive Teutonic-III dashboard + wipe live state.

Idempotent. Safe to run after `pm2 stop teutonic-validator teutonic-eval-tunnel`.
Refuses to run if the validator process is still up (best-effort check).

What it does (in order):
  1. Read live `dashboard.json` from Hippius (and R2) -> save to
     `dashboard-teutonic-iii-final.json` in both buckets (kept for posterity)
  2. Append `family_transition` event to `state/history.jsonl`
  3. DELETE these live state objects from R2:
       king/current.json
       state/queue.json
       state/seen_hotkeys.json
       state/validator_state.json
       state/dashboard_history.json
       eval/*  (all per-eval meta + verdict objects)
  4. (Does NOT delete state/history.jsonl — that's the historical event log.)
  5. Print a summary.

Usage:
    source /home/const/workspace/.venv/bin/activate
    cd /home/const/workspace/teutonic
    python scripts/cutover_to_teutonic_viii.py [--dry-run]
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cutover")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_clients():
    r2 = boto3.client(
        "s3",
        endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
        aws_access_key_id=os.environ["TEUTONIC_R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["TEUTONIC_R2_SECRET_KEY"],
        region_name="auto",
        config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"}),
    )
    r2_bucket = os.environ.get("TEUTONIC_R2_BUCKET", "constantinople")

    hippius_bucket = os.environ.get("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3")
    hippius = boto3.client(
        "s3",
        endpoint_url=os.environ.get("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com"),
        aws_access_key_id=os.environ["TEUTONIC_HIPPIUS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["TEUTONIC_HIPPIUS_SECRET_KEY"],
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            s3={"addressing_style": "path"},
        ),
    )
    return r2, r2_bucket, hippius, hippius_bucket


def check_validator_stopped():
    try:
        result = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            log.warning("pm2 jlist failed; can't verify validator is stopped")
            return
        procs = json.loads(result.stdout)
        for p in procs:
            if p["name"] in ("teutonic-validator", "teutonic-eval-tunnel"):
                status = p["pm2_env"].get("status", "?")
                if status == "online":
                    log.error("ABORTING: %s is still online (pm2 stop it first)", p["name"])
                    sys.exit(2)
    except Exception as e:
        log.warning("could not check pm2 status (%s); proceeding", e)


def archive_dashboard(hippius, hippius_bucket, r2, r2_bucket, dry_run):
    archive_key = "dashboard-teutonic-iii-final.json"
    log.info("archiving dashboard.json -> %s on Hippius and R2", archive_key)

    # Try to read once (Hippius first, fall back to R2), then write to both
    # buckets so the archive lives wherever we serve dashboards from.
    body = None
    ct = "application/json"
    for client_name, client, bucket in [("hippius", hippius, hippius_bucket),
                                         ("r2", r2, r2_bucket)]:
        try:
            obj = client.get_object(Bucket=bucket, Key="dashboard.json")
            body = obj["Body"].read()
            ct = obj.get("ContentType", "application/json")
            log.info("  %s: read dashboard.json (%d bytes)", client_name, len(body))
            break
        except Exception as e:
            log.info("  %s: no dashboard.json (%s)", client_name, e.__class__.__name__)

    if body is None:
        log.warning("no dashboard.json found anywhere; nothing to archive")
        return

    for client_name, client, bucket in [("hippius", hippius, hippius_bucket),
                                         ("r2", r2, r2_bucket)]:
        if dry_run:
            log.info("  %s: WOULD write %s (%d bytes)", client_name, archive_key, len(body))
            continue
        try:
            client.put_object(Bucket=bucket, Key=archive_key,
                              Body=body, ContentType=ct)
            log.info("  %s: wrote %s", client_name, archive_key)
        except Exception as e:
            log.warning("  %s: failed to write %s (%s)",
                        client_name, archive_key, e)


def append_transition_event(r2, r2_bucket, dry_run):
    record = {
        "event": "family_transition",
        "from": "Teutonic-III",
        "to": "Teutonic-VIII",
        "arch": "Qwen3",
        "params_b": 8.02,
        "num_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 262144,
        "tokenizer_source": "unconst/Teutonic-I",
        "timestamp": now(),
    }
    log.info("appending family_transition event to state/history.jsonl: %s", record)
    if dry_run:
        return
    line = json.dumps(record) + "\n"
    existing = b""
    try:
        existing = r2.get_object(Bucket=r2_bucket, Key="state/history.jsonl")["Body"].read()
    except Exception:
        pass
    r2.put_object(Bucket=r2_bucket, Key="state/history.jsonl",
                  Body=existing + line.encode(),
                  ContentType="application/x-ndjson")


def delete_live_state(r2, r2_bucket, dry_run):
    keys = [
        "king/current.json",
        "state/queue.json",
        "state/seen_hotkeys.json",
        "state/validator_state.json",
        "state/dashboard_history.json",
    ]
    for key in keys:
        try:
            if dry_run:
                r2.head_object(Bucket=r2_bucket, Key=key)
                log.info("WOULD delete %s", key)
            else:
                r2.delete_object(Bucket=r2_bucket, Key=key)
                log.info("deleted %s", key)
        except Exception as e:
            log.info("skipping %s (%s)", key, e.__class__.__name__)


def delete_eval_objects(r2, r2_bucket, dry_run):
    log.info("deleting eval/* objects (batched)")
    paginator = r2.get_paginator("list_objects_v2")
    total = 0
    batch = []
    for page in paginator.paginate(Bucket=r2_bucket, Prefix="eval/"):
        for obj in page.get("Contents", []):
            batch.append({"Key": obj["Key"]})
            total += 1
            if len(batch) == 1000:
                if not dry_run:
                    r2.delete_objects(Bucket=r2_bucket, Delete={"Objects": batch})
                log.info("  %s 1000 (running total %d)",
                         "would-delete" if dry_run else "deleted", total)
                batch = []
    if batch:
        if not dry_run:
            r2.delete_objects(Bucket=r2_bucket, Delete={"Objects": batch})
        log.info("  %s %d (running total %d)",
                 "would-delete" if dry_run else "deleted", len(batch), total)
    log.info("eval/* objects: %d %s", total, "would-be deleted" if dry_run else "deleted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-validator-check", action="store_true",
                    help="do not bail if validator is still running")
    args = ap.parse_args()

    if not args.skip_validator_check:
        check_validator_stopped()

    r2, r2_bucket, hippius, hippius_bucket = make_clients()

    log.info("=== Phase B cutover Teutonic-III -> Teutonic-VIII (dry_run=%s) ===",
             args.dry_run)
    log.info("R2 bucket: %s", r2_bucket)
    log.info("Hippius bucket: %s", hippius_bucket)

    archive_dashboard(hippius, hippius_bucket, r2, r2_bucket, args.dry_run)
    append_transition_event(r2, r2_bucket, args.dry_run)
    delete_live_state(r2, r2_bucket, args.dry_run)
    delete_eval_objects(r2, r2_bucket, args.dry_run)

    log.info("=== cutover %s ===", "preview complete" if args.dry_run else "complete")
    if args.dry_run:
        log.info("Re-run without --dry-run to actually perform.")


if __name__ == "__main__":
    main()
