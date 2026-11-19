#!/usr/bin/env python3
"""Preview the latest-king bonus; --apply updates PostgreSQL, never R2 or chain."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg.rows import dict_row

from scripts.retire_mix_competitions import apply_plan
ACTIVE_COMPETITIONS = ("main", "mix01", "mix06", "mix07")  # Historical cutover roster.
from teutonic.weights.contracts import weight_payload_digest
from teutonic.weights.policy import (
    begin_latest_king_policy,
    mapped_weight_plan,
    replacement_rewards,
)
from teutonic.weights.transition import lock_drained_services


def prepare(rows, publication):
    by_key = {r["competition_key"]: r for r in rows}
    if len(by_key) != len(rows) or {k for k, r in by_key.items() if r["active"]} != set(ACTIVE_COMPETITIONS):
        raise ValueError("latest king bonus requires exactly four active competitions")
    main = by_key["main"]
    old = main["reward_transition"]
    old_hotkeys, old_shares = replacement_rewards(old)
    for key in ACTIVE_COMPETITIONS:
        row = by_key[key]
        hotkey = old["main"]["hotkey"] if key == "main" else old["mixes"].get(key)
        if not row["current_reign_id"] or row["king_hotkey"] != hotkey:
            raise ValueError(f"{key} must have a crowned king matching its reward position")
    # Every successful crown updates this pointer in the same transaction as
    # reward_transition. It orders crowns even within the same block/timestamp.
    latest = [key for key in ACTIVE_COMPETITIONS
              if by_key[key]["current_reign_id"] == main["reward_reign_id"]]
    if len(latest) != 1:
        raise ValueError("latest reward reign must identify one current active king")
    state = begin_latest_king_policy(old, latest[0])
    if not publication or publication["source_reign_id"] != main["reward_reign_id"]:
        raise ValueError("missing durable publication for the latest crown")
    actual = {}
    for hotkey, share in zip(publication["policy_hotkeys"], publication["policy_weights"] or (), strict=True):
        if not math.isfinite(share) or share < 0:
            raise ValueError("publication has invalid reward shares")
        actual[hotkey] = actual.get(hotkey, 0) + share
    expected = dict(zip(old_hotkeys, old_shares, strict=True))
    if set(actual) != set(expected) or any(abs(actual[h] - w) > 1e-9 for h, w in expected.items()):
        raise ValueError("publication differs from the current reward policy")
    hotkeys, shares = replacement_rewards(state)
    mapping = dict(zip(publication["target_hotkeys"], publication["target_uids"], strict=True))
    if not set(hotkeys) <= set(mapping):
        raise ValueError("king missing from UID mapping; reconcile validator before cutover")
    targets, uids, weights = mapped_weight_plan(hotkeys, shares, mapping, burn_uid=0)
    return {
        "changed": state != old,
        "latest_competition": latest[0],
        "latest_reign_id": str(main["reward_reign_id"]),
        "transition": state, "retired_ids": [],
        "policy_hotkeys": list(hotkeys), "policy_weights": list(shares),
        "target_hotkeys": list(targets), "target_uids": list(uids),
        "normalized_weights": list(weights), "payload_sha256": weight_payload_digest(uids, weights),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new preview/before-state path; never overwritten")
    parser.add_argument("--apply", action="store_true", help="apply with services stopped and drained")
    args = parser.parse_args()
    with psycopg.connect(os.environ["TEUTONIC_DATABASE_URL"], row_factory=dict_row) as connection:
        if not args.apply:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        rows = connection.execute(
            """SELECT c.*, k.hotkey AS king_hotkey FROM control_plane.competitions c
               LEFT JOIN control_plane.king_reigns k ON k.reign_id=c.current_reign_id
               WHERE c.netuid=%s AND c.chain_generation=%s ORDER BY c.competition_key"""
            + (" FOR UPDATE OF c" if args.apply else ""),
            (int(os.environ["TEUTONIC_NETUID"]), os.environ["TEUTONIC_CHAIN_GENERATION"]),
        ).fetchall()
        main_row = next(r for r in rows if r["competition_key"] == "main")
        if args.apply:
            lock_drained_services(connection, main_row)
        publication = connection.execute(
            "SELECT * FROM control_plane.weight_publications WHERE source_reign_id=%s"
            + (" FOR UPDATE" if args.apply else ""),
            (main_row["reward_reign_id"],),
        ).fetchone()
        plan = prepare(rows, publication)
        pending = connection.execute(
            """SELECT 1 FROM control_plane.weight_submission_attempts
               WHERE weight_publication_id=%s AND state IN
               ('claimed','submitting','submitted','included','retry_pending') LIMIT 1""",
            (publication["weight_publication_id"],),
        ).fetchone()
        if pending and args.apply:
            raise ValueError("resolve outstanding weight attempt before cutover")
        with args.output.open("x") as stream:
            json.dump({"before": {"competitions": rows, "publication": publication},
                       "plan": plan, "outstanding_weight_attempt": bool(pending),
                       "apply_requested": args.apply}, stream, indent=2, default=str)
            stream.write("\n")
        if args.apply:
            apply_plan(connection, main_row, publication, plan)
    print("Database cutover committed; no R2 or chain writes." if args.apply else
          "Read-only preview saved; no database, R2 or chain writes.")


if __name__ == "__main__":
    main()
