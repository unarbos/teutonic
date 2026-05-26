#!/usr/bin/env python3
"""Re-arm failed validator submissions so they can be picked up again.

This is an operator override for cases where recent failures were caused by
validator/eval-box infrastructure rather than the miner's model. It removes
selected failed rows from dashboard history and clears the persisted burn lists
that stop the same on-chain reveal from being seen again.

Run this with the validator stopped so its in-memory caches do not overwrite the
patched R2 state on the next flush:

    pm2 stop teutonic-validator
    python3 scripts/rearm_failed_submissions.py --last 8
    python3 scripts/rearm_failed_submissions.py --hotkey hotkey123
    python3 scripts/rearm_failed_submissions.py --last 8 --apply
    pm2 start teutonic-validator
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig


def _env_first(*names: str) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _r2_client():
    endpoint = _env_first("TEUTONIC_R2_ENDPOINT", "R2_URL")
    access_key = _env_first("TEUTONIC_R2_ACCESS_KEY", "R2_ACCESS_KEY_ID")
    secret_key = _env_first("TEUTONIC_R2_SECRET_KEY", "R2_SECRET_ACCESS_KEY")
    if not endpoint or not access_key or not secret_key:
        raise SystemExit(
            "missing R2 credentials; set TEUTONIC_R2_ENDPOINT / "
            "TEUTONIC_R2_ACCESS_KEY / TEUTONIC_R2_SECRET_KEY "
            "(or R2_URL / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)"
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=BotoConfig(
            connect_timeout=15,
            read_timeout=45,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def _bucket_name() -> str:
    bucket = _env_first("TEUTONIC_R2_BUCKET", "R2_BUCKET", "R2_BUCKET_NAME")
    if not bucket:
        raise SystemExit(
            "missing bucket; set TEUTONIC_R2_BUCKET (or R2_BUCKET / R2_BUCKET_NAME)"
        )
    return bucket


def _get_json(client, bucket: str, key: str, default):
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return default
    try:
        return json.loads(body)
    except Exception as exc:
        raise SystemExit(f"could not decode {key} as JSON: {exc}") from exc


def _put_json(client, bucket: str, key: str, data) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, default=str).encode(),
        ContentType="application/json",
    )


def _append_jsonl_event(client, bucket: str, key: str, row: dict) -> None:
    try:
        existing = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        existing = b""
    body = existing + (json.dumps(row, default=str) + "\n").encode()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/x-ndjson",
    )


def _is_failed_history_row(row: dict) -> bool:
    if row.get("accepted"):
        return False
    if row.get("verdict") == "error":
        return True
    return bool(row.get("error_code"))


def _model_key(repo: str, digest: str = "") -> str:
    repo = (repo or "").strip()
    digest = (digest or "").strip()
    return f"{repo}@{digest}" if repo and digest else repo


def _row_repo(row: dict) -> str:
    return (row.get("challenger_repo") or row.get("model_repo") or "").strip()


def _row_digest(row: dict) -> str:
    return (row.get("challenger_digest") or row.get("model_digest") or "").strip()


def _row_label(row: dict) -> str:
    cid = row.get("challenge_id", "?")
    hk = (row.get("hotkey") or "")[:16]
    repo = _row_repo(row) or "?"
    digest = _row_digest(row)
    err = row.get("error_code") or row.get("verdict") or "?"
    return f"{cid} hotkey={hk} repo={repo} digest={(digest[:19] or '?')} error={err}"


def _choose_rows(history: list[dict], args) -> tuple[list[dict], list[str]]:
    if args.challenge_id:
        wanted = list(dict.fromkeys(args.challenge_id))
        by_cid = {row.get("challenge_id"): row for row in history}
        selected = [by_cid[cid] for cid in wanted if cid in by_cid]
        missing = [cid for cid in wanted if cid not in by_cid]
        if args.failed_only:
            selected = [row for row in selected if _is_failed_history_row(row)]
        if args.error_code:
            selected = [row for row in selected if row.get("error_code") == args.error_code]
        return selected, missing

    if args.hotkey:
        wanted = list(dict.fromkeys(hk.strip() for hk in args.hotkey if hk.strip()))
        seen = set()
        selected = []
        for row in history:
            hotkey = (row.get("hotkey") or "").strip()
            if hotkey not in wanted:
                continue
            if args.failed_only and not _is_failed_history_row(row):
                continue
            if args.error_code and row.get("error_code") != args.error_code:
                continue
            # Avoid selecting duplicate challenge rows if history ever contains them.
            cid = row.get("challenge_id")
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            selected.append(row)
        found_hotkeys = {(row.get("hotkey") or "").strip() for row in selected}
        missing = [hk for hk in wanted if hk not in found_hotkeys]
        return selected, missing

    selected = []
    for row in history:
        if args.failed_only and not _is_failed_history_row(row):
            continue
        if args.error_code and row.get("error_code") != args.error_code:
            continue
        selected.append(row)
        if len(selected) >= args.last:
            break
    return selected, []


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remove failed history rows and clear their submission burn state."
    )
    ap.add_argument(
        "--last",
        type=int,
        default=8,
        help="Newest N matching history rows to re-arm (default: 8).",
    )
    ap.add_argument(
        "--challenge-id",
        action="append",
        default=[],
        help="Specific challenge_id to re-arm. May be repeated.",
    )
    ap.add_argument(
        "--hotkey",
        action="append",
        default=[],
        help="Specific hotkey to re-arm. May be repeated.",
    )
    ap.add_argument(
        "--error-code",
        default="",
        help="Only select failed rows with this exact error_code.",
    )
    ap.add_argument(
        "--all-history",
        dest="failed_only",
        action="store_false",
        help="Allow selecting any history rows, not just failures.",
    )
    ap.add_argument(
        "--no-adjust-stats",
        dest="adjust_stats",
        action="store_false",
        help="Do not decrement stats.failed for the removed rows.",
    )
    ap.add_argument(
        "--reason",
        default="operator rearm after infra/gpu-side failure",
        help="Audit reason recorded in state/history.jsonl.",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write changes back to R2. Default is dry-run.",
    )
    ap.set_defaults(failed_only=True, adjust_stats=True)
    args = ap.parse_args()

    if args.last < 1:
        raise SystemExit("--last must be >= 1")

    client = _r2_client()
    bucket = _bucket_name()

    history_blob = _get_json(client, bucket, "state/dashboard_history.json", {"history": []})
    seen_blob = _get_json(client, bucket, "state/seen_hotkeys.json", {"hotkeys": []})
    completed_blob = _get_json(client, bucket, "state/completed_repos.json", {"repos": []})
    state_blob = _get_json(client, bucket, "state/validator_state.json", {})

    history = list(history_blob.get("history") or [])
    seen_hotkeys = set(seen_blob.get("hotkeys") or [])
    completed_repos = set(completed_blob.get("repos") or [])
    stats = dict(state_blob.get("stats") or {})

    selected, missing = _choose_rows(history, args)
    if missing:
        print("[warn] challenge IDs not found in history:", ", ".join(missing))
    if not selected:
        print("no matching history rows found; nothing to do")
        return

    selected_cids = {row.get("challenge_id") for row in selected}
    selected_hotkeys = sorted(
        {row.get("hotkey", "").strip() for row in selected if row.get("hotkey")}
    )
    selected_model_keys = sorted(
        {
            _model_key(_row_repo(row), _row_digest(row))
            for row in selected
            if _row_repo(row)
        }
    )

    new_history = [row for row in history if row.get("challenge_id") not in selected_cids]
    new_seen_hotkeys = sorted(hk for hk in seen_hotkeys if hk not in selected_hotkeys)
    new_completed_repos = sorted(
        repo_key for repo_key in completed_repos if repo_key not in selected_model_keys
    )

    removed_failed_rows = sum(1 for row in selected if _is_failed_history_row(row))
    failed_before = int(stats.get("failed", 0) or 0)
    failed_after = failed_before
    if args.adjust_stats:
        failed_after = max(0, failed_before - removed_failed_rows)

    print("Selected rows:")
    for row in selected:
        print(f"  - {_row_label(row)}")
    print("")
    print(f"History rows:      {len(history)} -> {len(new_history)}")
    print(f"Seen hotkeys:      {len(seen_hotkeys)} -> {len(new_seen_hotkeys)}")
    print(f"Completed repos:   {len(completed_repos)} -> {len(new_completed_repos)}")
    if args.adjust_stats:
        print(f"stats.failed:      {failed_before} -> {failed_after}")
    else:
        print(f"stats.failed:      unchanged at {failed_before}")

    if not args.apply:
        print("")
        print("[dry-run] no R2 objects were modified")
        print("restart the validator after --apply so it reloads the patched burn lists")
        return

    history_blob["history"] = new_history
    seen_blob["hotkeys"] = new_seen_hotkeys
    seen_blob["updated_at"] = _utc_now()
    completed_blob["repos"] = new_completed_repos
    completed_blob["updated_at"] = _utc_now()

    state_blob["stats"] = {**stats, "failed": failed_after} if args.adjust_stats else stats
    state_blob["updated_at"] = _utc_now()

    _put_json(client, bucket, "state/dashboard_history.json", history_blob)
    _put_json(client, bucket, "state/seen_hotkeys.json", seen_blob)
    _put_json(client, bucket, "state/completed_repos.json", completed_blob)
    _put_json(client, bucket, "state/validator_state.json", state_blob)

    _append_jsonl_event(
        client,
        bucket,
        "state/history.jsonl",
        {
            "timestamp": _utc_now(),
            "event": "submissions_rearmed",
            "challenge_ids": sorted(selected_cids),
            "hotkeys": selected_hotkeys,
            "model_keys": selected_model_keys,
            "removed_failed_rows": removed_failed_rows,
            "stats_failed_before": failed_before,
            "stats_failed_after": failed_after,
            "reason": args.reason,
            "trigger": "operator_script:rearm_failed_submissions",
        },
    )

    print("")
    print("[apply] updated:")
    print("  - state/dashboard_history.json")
    print("  - state/seen_hotkeys.json")
    print("  - state/completed_repos.json")
    print("  - state/validator_state.json")
    print("  - state/history.jsonl (audit append)")
    print("")
    print("restart the validator so it reloads the patched state and re-scans chain reveals")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
