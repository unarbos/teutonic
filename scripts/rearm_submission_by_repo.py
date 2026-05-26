#!/usr/bin/env python3
"""Re-arm validator submissions selected by model repo.

This is an operator override for cases where you want the validator to see a
specific on-chain reveal again, even if its prior evaluation already landed in
dashboard history. It removes matching rows from dashboard history and clears
the persisted burn lists that block the same reveal from being scanned again.

Run this with the validator stopped so its in-memory caches do not overwrite the
patched R2 state on the next flush:

    pm2 stop teutonic-validator
    python3 scripts/rearm_submission_by_repo.py --repo repo/name
    python3 scripts/rearm_submission_by_repo.py --repo repo/name --apply
    pm2 start teutonic-validator
"""
from __future__ import annotations

import argparse
import sys

from rearm_failed_submissions import (
    _append_jsonl_event,
    _bucket_name,
    _get_json,
    _is_failed_history_row,
    _model_key,
    _put_json,
    _r2_client,
    _row_digest,
    _row_label,
    _row_repo,
    _utc_now,
)


def _choose_rows(history: list[dict], args) -> list[dict]:
    selected = []
    for row in history:
        repo = _row_repo(row)
        digest = _row_digest(row)
        hotkey = (row.get("hotkey") or "").strip()
        if repo != args.repo:
            continue
        if args.digest and digest != args.digest:
            continue
        if args.hotkey and hotkey != args.hotkey:
            continue
        if args.failed_only and not _is_failed_history_row(row):
            continue
        selected.append(row)
        if not args.all_matches and len(selected) >= args.last:
            break
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remove history rows for a specific model repo and clear its submission burn state."
    )
    ap.add_argument(
        "--repo",
        required=True,
        help="Exact challenger_repo/model_repo to re-arm.",
    )
    ap.add_argument(
        "--digest",
        default="",
        help="Optional exact challenger/model digest filter.",
    )
    ap.add_argument(
        "--hotkey",
        default="",
        help="Optional exact hotkey filter.",
    )
    ap.add_argument(
        "--last",
        type=int,
        default=1,
        help="Newest N matching history rows to re-arm when --all-matches is not set (default: 1).",
    )
    ap.add_argument(
        "--all-matches",
        action="store_true",
        help="Re-arm every matching history row for this repo/filter set.",
    )
    ap.add_argument(
        "--failed-only",
        action="store_true",
        help="Only select failed history rows. Default includes accepted rows too.",
    )
    ap.add_argument(
        "--no-adjust-stats",
        dest="adjust_stats",
        action="store_false",
        help="Do not decrement stats.failed for removed failed rows.",
    )
    ap.add_argument(
        "--reason",
        default="operator repo-targeted rearm for forced re-evaluation",
        help="Audit reason recorded in state/history.jsonl.",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write changes back to R2. Default is dry-run.",
    )
    ap.set_defaults(adjust_stats=True)
    args = ap.parse_args()

    if args.last < 1:
        raise SystemExit("--last must be >= 1")

    client = _r2_client()
    bucket = _bucket_name()

    history_blob = _get_json(client, bucket, "state/dashboard_history.json", {"history": []})
    seen_blob = _get_json(client, bucket, "state/seen_hotkeys.json", {"hotkeys": []})
    completed_blob = _get_json(client, bucket, "state/completed_repos.json", {"repos": []})
    state_blob = _get_json(client, bucket, "state/validator_state.json", {})
    king_blob = _get_json(client, bucket, "king/current.json", {})

    history = list(history_blob.get("history") or [])
    seen_hotkeys = set(seen_blob.get("hotkeys") or [])
    completed_repos = set(completed_blob.get("repos") or [])
    stats = dict(state_blob.get("stats") or {})

    selected = _choose_rows(history, args)
    if not selected:
        target = args.repo
        if args.digest:
            target = f"{target}@{args.digest}"
        print(f"no matching history rows found for {target}")
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

    current_king_repo = (king_blob.get("model_repo") or "").strip()
    current_king_hotkey = (king_blob.get("hotkey") or "").strip()
    if current_king_repo == args.repo or (
        current_king_hotkey and current_king_hotkey in selected_hotkeys
    ):
        print("")
        print(
            "[warn] the selected repo/hotkey matches the current king; "
            "the validator still skips current-king submissions, so this alone "
            "will not force a re-eval while it remains king"
        )

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
            "event": "submission_rearmed_by_repo",
            "challenge_ids": sorted(selected_cids),
            "hotkeys": selected_hotkeys,
            "model_keys": selected_model_keys,
            "removed_failed_rows": removed_failed_rows,
            "stats_failed_before": failed_before,
            "stats_failed_after": failed_after,
            "reason": args.reason,
            "trigger": "operator_script:rearm_submission_by_repo",
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
