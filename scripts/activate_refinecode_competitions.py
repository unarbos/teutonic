#!/usr/bin/env python3
"""Preview MAIN2/MIX08/MIX09 cutover; explicit --apply writes PostgreSQL only."""

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

import chain_config
from scripts.publish_packed_evaluation import validate_bundle
from scripts.retire_mix_competitions import apply_plan
from teutonic.evaluation.configuration import DatasetManifestSnapshot, store_evaluation_configuration
from teutonic.weights.contracts import weight_payload_digest
from teutonic.weights.policy import (
    REFINECODE_COMPETITIONS, begin_refinecode_replacement, mapped_weight_plan, replacement_rewards,
)
from teutonic.weights.transition import lock_drained_services

OLD = {"main", "mix01", "mix06", "mix07"}
NEW = {"main2", "mix08", "mix09"}
SEED_MODEL = "teutonic-II-110B-A7B-5Eez-v9"


def prepare(rows, publication, seed, uid_by_hotkey):
    """Keep the historical MAIN row as control-plane parent; retire its admission."""
    by_key = {r["competition_key"]: r for r in rows}
    if (len(by_key) != len(rows) or {k for k, r in by_key.items() if r["active"]} != OLD
            or NEW & set(by_key)):
        raise ValueError("cutover requires the old four active competitions and no destination rows")
    main = by_key["main"]
    if any(r.get("main_competition_id") != main["competition_id"]
           for k, r in by_key.items() if k in OLD - {"main"}):
        raise ValueError("competitions must share the original MAIN control-plane parent")
    old = main["reward_transition"]
    expected = dict(zip(*replacement_rewards(old), strict=True))
    if not publication or publication["source_reign_id"] != main["reward_reign_id"]:
        raise ValueError("missing current durable publication")
    actual = {}
    for h, w in zip(publication["policy_hotkeys"], publication["policy_weights"], strict=True):
        if not math.isfinite(w) or w < 0:
            raise ValueError("invalid publication weight")
        actual[h] = actual.get(h, 0) + w
    if set(actual) != set(expected) or any(abs(actual[h] - w) > 1e-9 for h, w in expected.items()):
        raise ValueError("publication differs from the current reward policy")
    for key in OLD:
        row = by_key[key]
        expected_hotkey = old["main"]["hotkey"] if key == "main" else old["mixes"][key]
        if not row["current_reign_id"] or row["king_hotkey"] != expected_hotkey:
            raise ValueError("current king differs from its reward position")
    if seed["model_name"].rsplit("/", 1)[-1] != SEED_MODEL or not seed["reign_id"]:
        raise ValueError("MIX08/MIX09 starting king differs from the requested model")
    baselines = {"main2": main["current_reign_id"], "mix07": by_key["mix07"]["current_reign_id"],
                 "mix08": seed["reign_id"], "mix09": seed["reign_id"]}
    kings = {"main2": main["king_hotkey"], "mix07": by_key["mix07"]["king_hotkey"],
             "mix08": seed["hotkey"], "mix09": seed["hotkey"]}
    state = begin_refinecode_replacement(old, {k: kings[k] for k in ("main2", "mix07")},
        [{"id": k, "hotkey": by_key[k]["king_hotkey"], "reign_id": str(by_key[k]["current_reign_id"]),
          "crowned_block": by_key[k]["king_crowned_block"]} for k in ("mix01", "mix06")])
    hotkeys, shares = replacement_rewards(state)
    if not set(hotkeys) <= set(uid_by_hotkey):
        raise ValueError("starting king hotkey missing from finalized metagraph; review incentives")
    targets, uids, weights = mapped_weight_plan(hotkeys, shares, uid_by_hotkey, burn_uid=0)
    return {"changed": True, "transition": state, "starting_reigns": baselines,
            "retired_ids": [by_key[k]["competition_id"] for k in ("main", "mix01", "mix06")],
            "policy_hotkeys": list(hotkeys), "policy_weights": list(shares),
            "target_hotkeys": list(targets), "target_uids": list(uids),
            "normalized_weights": list(weights), "payload_sha256": weight_payload_digest(uids, weights)}


def configurations(directory):
    artifacts, _, mixes = validate_bundle(directory)
    if {m["competition"] for m in mixes} != NEW:
        raise ValueError("cutover bundle must contain MAIN2, MIX08 and MIX09 only; MIX07 stays unchanged")
    result = {}
    for mix in mixes:
        key = mix["competition"]
        definition = chain_config.COMPETITION_DEFAULTS[key]
        categories = [{"dataset": chain_config.CATEGORY_REGISTRY[c]["dataset"], "category": c, "weight": w}
                      for c, w in sorted(definition["categories"].items())]
        if sorted(mix["categories"], key=lambda c: c["category"]) != categories or mix["token_budget"] != definition["long_document_tokens"]:
            raise ValueError("bundle differs from reviewed chain.toml weights or budget")
        entry = next(e for e, _ in artifacts if e["key"].startswith(f"teutonic-text/splits/{key}/"))
        result[key] = dict(dataset_label=key, n=0, delta_threshold=definition["delta_threshold"],
            shards_per_dataset=1, long_documents=mix, manifests=[DatasetManifestSnapshot(
                key, f"s3://{entry['bucket']}/{entry['key']}", entry["sha256"], 1.0, mix)])
    return artifacts, result


def apply_cutover(db, root, publication, plan, configs):
    """Caller holds service/row locks; all operations belong to one transaction."""
    for key in sorted(NEW):
        row = db.execute(
            """INSERT INTO control_plane.competitions
               (netuid,chain_generation,name,competition_key,main_competition_id,starting_reign_id,active)
               VALUES (%s,%s,%s,%s,%s,%s,false) RETURNING competition_id""",
            (root["netuid"], root["chain_generation"], key, key, root["competition_id"], plan["starting_reigns"][key])).fetchone()
        db.execute(
            """INSERT INTO control_plane.evaluation_early_stopping_policies
               (competition_id,enabled,min_fraction,advantage_quantile,margin,check_interval)
               SELECT %s,enabled,min_fraction,advantage_quantile,margin,check_interval
               FROM control_plane.evaluation_early_stopping_policies WHERE competition_id=%s""",
            (row["competition_id"], root["competition_id"]))
        store_evaluation_configuration(db, netuid=root["netuid"], chain_generation=root["chain_generation"], competition=key, **configs[key])
    apply_plan(db, root, publication, plan)
    db.execute("UPDATE control_plane.competitions SET active=true WHERE main_competition_id=%s AND competition_key=ANY(%s)",
               (root["competition_id"], sorted(NEW)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new before-state/preview file")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    artifacts, configs = configurations(args.bundle_dir)
    if args.apply:
        # Admission must not point to unpublished objects. Only GET operations here.
        from teutonic.evaluation.packed_text import DatasetReader, read_client
        reader = DatasetReader(read_client())
        for entry, _ in artifacts:
            reader.json({k: entry[k] for k in ("bucket", "key", "sha256")})
    scope = (int(os.environ["TEUTONIC_NETUID"]), os.environ["TEUTONIC_CHAIN_GENERATION"])
    with psycopg.connect(os.environ["TEUTONIC_DATABASE_URL"], row_factory=dict_row) as db:
        if not args.apply:
            db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        rows = db.execute(
            """SELECT c.*, k.hotkey AS king_hotkey, k.crowned_finalized_block AS king_crowned_block FROM control_plane.competitions c
               LEFT JOIN control_plane.king_reigns k ON k.reign_id=c.current_reign_id
               WHERE c.netuid=%s AND c.chain_generation=%s ORDER BY c.competition_key"""
            + (" FOR UPDATE OF c" if args.apply else ""), scope).fetchall()
        root = next(r for r in rows if r["competition_key"] == "main")
        if args.apply:
            lock_drained_services(db, root)
            if not db.execute("SELECT 1 FROM control_plane.evaluation_early_stopping_policies WHERE competition_id=%s",
                              (root["competition_id"],)).fetchone():
                raise ValueError("missing source early-stopping policy")
        publication = db.execute("SELECT * FROM control_plane.weight_publications WHERE source_reign_id=%s"
            + (" FOR UPDATE" if args.apply else ""), (root["reward_reign_id"],)).fetchone()
        seeds = db.execute(
            """SELECT k.*, u.model_name FROM control_plane.king_reigns k
               JOIN control_plane.uploads u ON u.upload_id=k.accepted_upload_id
               JOIN control_plane.competitions c ON c.competition_id=k.competition_id
               WHERE c.netuid=%s AND c.chain_generation=%s
                 AND regexp_replace(u.model_name, '^.*/', '')=%s
               ORDER BY k.crowned_at DESC, k.reign_id""", (*scope, SEED_MODEL)).fetchall()
        if not seeds or len({(s['model_digest'], s['hotkey']) for s in seeds}) != 1:
            raise ValueError("starting model must resolve to one verified digest and owner in king history")
        assignments = db.execute(
            """SELECT a.hotkey, a.uid FROM control_plane.metagraph_uid_assignments a
               WHERE a.snapshot_id=(SELECT snapshot_id FROM control_plane.metagraph_snapshots
                 WHERE netuid=%s AND chain_generation=%s AND is_complete
                 ORDER BY finalized_block DESC LIMIT 1)""", scope).fetchall()
        plan = prepare(rows, publication, seeds[0], {a['hotkey']: a['uid'] for a in assignments})
        pending = db.execute(
            """SELECT 1 FROM control_plane.weight_submission_attempts WHERE weight_publication_id=%s
               AND state IN ('claimed','submitting','submitted','included','retry_pending') LIMIT 1""",
            (publication["weight_publication_id"],)).fetchone()
        if pending and args.apply:
            raise ValueError("resolve outstanding weight attempt before cutover")
        with args.output.open("x") as stream:
            json.dump({"before": {"competitions": rows, "publication": publication},
                       "seed": seeds[0], "plan": plan, "apply_requested": args.apply,
                       "outstanding_weight_attempt": bool(pending)}, stream, indent=2, default=str)
        if args.apply:
            apply_cutover(db, root, publication, plan, configs)
    print("Cutover committed; no R2 or chain writes." if args.apply else "Read-only preview saved; no remote writes.")


if __name__ == "__main__":
    main()
