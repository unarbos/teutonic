#!/usr/bin/env python3
"""Preview four-competition cutover; --apply changes PostgreSQL, never R2 or chain."""

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from teutonic.competitions import ACTIVE_COMPETITIONS
from teutonic.weights.contracts import weight_payload_digest
from teutonic.weights.policy import (
    FOUR_COMPETITION_POLICY,
    begin_four_competition_policy,
    mapped_weight_plan,
    replacement_rewards,
)
from teutonic.weights.transition import lock_drained_services


def prepare(rows, publication):
    """Build a guarded plan from locked rows; no I/O or mutations."""
    by_key = {r["competition_key"]: r for r in rows}
    # MAIN retains the chain's display name (e.g. Teutonic-II-110B).
    if len(by_key) != len(rows) or any(not r["name"] or (k != "main" and r["name"] != k)
                                     for k, r in by_key.items()):
        raise ValueError("ambiguous competition identities")
    if not set(ACTIVE_COMPETITIONS) <= set(by_key):
        raise ValueError("missing active competition")
    main = by_key["main"]
    old = main["reward_transition"]
    state = begin_four_competition_policy(old)
    for key in ACTIVE_COMPETITIONS:
        row = by_key[key]
        recipient = state["main"]["hotkey"] if key == "main" else state["mixes"][key]
        if not row["active"] or not row["current_reign_id"] or row["king_hotkey"] != recipient:
            raise ValueError(f"{key} must have an active crowned king matching its reward position")
    if any(r["active"] and k not in {*ACTIVE_COMPETITIONS, "mix02", "mix03", "mix05"}
           for k, r in by_key.items()):
        raise ValueError("unexpected active competition; review cutover")
    if not publication or not publication["policy_weights"]:
        raise ValueError("missing durable reward publication")
    old_rewards = dict(zip(*replacement_rewards(old), strict=True))
    actual = {}
    for h, w in zip(publication["policy_hotkeys"], publication["policy_weights"], strict=True):
        actual[h] = actual.get(h, 0) + w
    if set(actual) != set(old_rewards) or any(abs(actual[h] - w) > 1e-9 for h, w in old_rewards.items()):
        raise ValueError("publication differs from the current reward policy")
    hotkeys, shares = replacement_rewards(state)
    # Retain the last finalized mapping; new policy introduces no recipients.
    # Validator must refresh it from the finalized metagraph before publisher restart.
    mapping = dict(zip(publication["target_hotkeys"], publication["target_uids"], strict=True))
    if not set(hotkeys) <= set(mapping):
        raise ValueError("retained king missing from UID mapping; reconcile validator before cutover")
    targets, uids, weights = mapped_weight_plan(hotkeys, shares, mapping, burn_uid=0)
    retired = [r["competition_id"] for k, r in by_key.items()
               if k in {"mix02", "mix03", "mix05"} and r["active"]]
    return {
        "changed": old["version"] != FOUR_COMPETITION_POLICY or bool(retired),
        "transition": state, "retired_ids": retired,
        "policy_hotkeys": list(hotkeys), "policy_weights": list(shares),
        "target_hotkeys": list(targets), "target_uids": list(uids),
        "normalized_weights": list(weights), "payload_sha256": weight_payload_digest(uids, weights),
    }


def apply_plan(connection, main, publication, plan):
    """Caller holds service/row locks and has written the before-state snapshot."""
    if not plan["changed"]:
        return
    connection.execute(
        "UPDATE control_plane.competitions SET reward_transition=%s, updated_at=clock_timestamp() WHERE competition_id=%s",
        (Jsonb(plan["transition"]), main["competition_id"]),
    )
    connection.execute(
        "UPDATE control_plane.competitions SET active=false, updated_at=clock_timestamp() WHERE competition_id=ANY(%s::uuid[])",
        (plan["retired_ids"],),
    )
    connection.execute(
        """UPDATE control_plane.weight_publications
           SET policy_hotkeys=%s, policy_weights=%s, target_hotkeys=%s, target_uids=%s,
               normalized_weights=%s, payload_sha256=%s, payload_revision=payload_revision+1,
               state='requested', owner_instance_id=NULL, lease_expires_at=NULL,
               next_retry_at=NULL, extrinsic_id=NULL, included_block=NULL, finalized_block=NULL,
               last_error_code=NULL, requested_at=clock_timestamp(), submitted_at=NULL,
               included_at=NULL, finalized_at=NULL, cadence_enabled=true,
               next_due_block=CASE WHEN last_attempted_block IS NULL THEN NULL
                                   ELSE last_attempted_block+cadence_blocks END,
               updated_at=clock_timestamp()
           WHERE weight_publication_id=%s""",
        (*(plan[k] for k in ("policy_hotkeys", "policy_weights", "target_hotkeys", "target_uids",
                            "normalized_weights", "payload_sha256")), publication["weight_publication_id"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new local preview/backup path; never overwritten")
    parser.add_argument("--apply", action="store_true", help="apply reviewed cutover with services stopped and drained")
    args = parser.parse_args()
    with psycopg.connect(os.environ["TEUTONIC_DATABASE_URL"], row_factory=dict_row) as connection:
        if not args.apply:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        suffix = " FOR UPDATE OF c" if args.apply else ""
        rows = connection.execute(
            """SELECT c.*, k.hotkey AS king_hotkey FROM control_plane.competitions c
               LEFT JOIN control_plane.king_reigns k ON k.reign_id=c.current_reign_id
               WHERE c.netuid=%s AND c.chain_generation=%s ORDER BY c.competition_key""" + suffix,
            (int(os.environ["TEUTONIC_NETUID"]), os.environ["TEUTONIC_CHAIN_GENERATION"]),
        ).fetchall()
        main_row = next(r for r in rows if r["competition_key"] == "main")
        if args.apply:
            lock_drained_services(connection, main_row)
        publication = connection.execute(
            "SELECT * FROM control_plane.weight_publications WHERE source_reign_id=%s"
            + (" FOR UPDATE" if args.apply else ""),
            (main_row["reward_reign_id"] or main_row["current_reign_id"],),
        ).fetchone()
        plan = prepare(rows, publication)
        pending = connection.execute(
            """SELECT 1 FROM control_plane.weight_submission_attempts
               WHERE weight_publication_id=%s AND state IN
               ('claimed','submitting','submitted','included','retry_pending') LIMIT 1""",
            (publication["weight_publication_id"],),
        ).fetchone()
        if pending:
            raise ValueError("resolve outstanding weight attempt before cutover")
        # Exclusive create preserves a recovery snapshot even if the DB commit fails.
        with args.output.open("x") as stream:
            json.dump({"before": {"competitions": rows, "publication": publication},
                       "plan": plan, "apply_requested": args.apply}, stream, indent=2, default=str)
            stream.write("\n")
        if args.apply:
            apply_plan(connection, main_row, publication, plan)
    print("Database cutover committed; no R2 or chain writes." if args.apply else
          "Read-only preview saved; no database, R2 or chain writes.")


if __name__ == "__main__":
    main()
