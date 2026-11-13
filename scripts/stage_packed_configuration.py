#!/usr/bin/env python3
"""Generate guarded PostgreSQL migration SQL locally; never execute or upload it."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.evaluation.configuration import DatasetManifestSnapshot, evaluation_config_version
from teutonic.evaluation.packed_text import canonical, validate_mix


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def stage(bundle, *, netuid, generation, threshold, previous_token_budget=None, previous_configs=None):
    import math
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite non-negative bits/byte")
    if previous_token_budget is not None and (type(previous_token_budget) is not int or not 1 <= previous_token_budget <= 100_000_000):
        raise ValueError("invalid previous token budget")
    from scripts.publish_packed_evaluation import validate_bundle
    validate_bundle(bundle)
    root = Path(bundle)
    entries = json.loads((root / "pending-publication.json").read_text())
    if previous_configs is not None:
        import re
        if (not isinstance(previous_configs, dict) or not previous_configs
                or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                       for v in previous_configs.values())):
            raise ValueError("previous configs must map competitions to exact config-version hashes")
        if previous_token_budget is not None:
            raise ValueError("pin-only migration cannot change the token budget")
    staged = set()
    blocks = ["-- Apply allow_packed_text_manifests.sql first. Verify R2 publication before applying.\nBEGIN;"]
    for entry in entries:
        if not entry["key"].startswith("teutonic-text/splits/"):
            continue
        name = entry["local"]
        if Path(name).name != name:
            raise ValueError("invalid bundle filename")
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("bundle hash mismatch")
        mix = validate_mix(json.loads(raw))
        from teutonic.competitions import ACTIVE_COMPETITIONS
        if mix["competition"] not in ACTIVE_COMPETITIONS:
            raise ValueError("cannot stage a retired competition for the new rollout")
        url = f"s3://{entry['bucket']}/{entry['key']}"
        snapshot = DatasetManifestSnapshot(mix["competition"], url, entry["sha256"], 1.0, mix)
        version = evaluation_config_version(dataset_label=mix["competition"], n=0,
                    delta_threshold=threshold, manifests=[snapshot], shards_per_dataset=1, long_documents=mix)
        weights = {c["category"]: c["weight"] for c in mix["categories"]}
        sources = {c["category"]: c["dataset"] for c in mix["categories"]}
        content = literal(canonical(mix).decode())
        staged.add(mix['competition'])
        if previous_configs is not None:
            if mix['competition'] not in previous_configs:
                raise ValueError("missing previous configuration for competition")
            source_guard = f"""
   AND ec.config_version IN ({literal(previous_configs[mix['competition']])},{literal(version)})
   AND ec.delta_threshold={threshold} AND ec.dataset_label={literal(mix['competition'])}
   AND ec.shards_per_dataset=1 AND dm.position=0
   AND dm.manifest_json->>'format'='teutonic-packed-text-mix'
   AND ec.long_documents=dm.manifest_json
   AND (dm.manifest_json - 'dataset' - 'sampling')=({content}::jsonb - 'dataset' - 'sampling')
"""
        else:
            source_guard = f"""
   AND ((ec.long_documents->>'token_budget')::bigint={previous_token_budget or mix['token_budget']}
        OR (ec.long_documents={content}::jsonb AND ec.config_version={literal(version)}))
   AND ((dm.manifest_json->'category_weights'={literal(json.dumps(weights))}::jsonb
         AND dm.manifest_json->'category_sources'={literal(json.dumps(sources))}::jsonb)
        OR dm.manifest_json={content}::jsonb)
"""
        # Guard both the category weights and input-token budget against live state.
        blocks.append(f"""
DO $migration$
DECLARE cid uuid; previous uuid; eid uuid;
BEGIN
 SELECT competition_id INTO STRICT cid FROM control_plane.competitions
 WHERE netuid={int(netuid)} AND chain_generation={literal(generation)}
 AND competition_key={literal(mix['competition'])} AND active FOR UPDATE;
 IF EXISTS (SELECT 1 FROM control_plane.evaluations e WHERE e.competition_id=cid
            AND (e.state IN ('claimed','evaluating') OR
                 (e.state='retryable_failure' AND NOT EXISTS (
                    SELECT 1 FROM control_plane.uploads u WHERE u.upload_id=e.upload_id
                    AND u.state IN ('accepted','rejected','evaluation_failed')
                    AND EXISTS (SELECT 1 FROM control_plane.evaluations newer
                                WHERE newer.upload_id=e.upload_id
                                AND newer.attempt_number>e.attempt_number
                                AND newer.state IN ('completed','terminal_failure')))))) THEN
   RAISE EXCEPTION 'Drain active evaluations and resolve pending retries before cutover';
 END IF;
 SELECT evaluation_config_id INTO STRICT previous FROM control_plane.evaluation_configs
 WHERE competition_id=cid AND active FOR UPDATE;
 IF NOT EXISTS (
   SELECT 1 FROM control_plane.evaluation_configs ec
   JOIN control_plane.dataset_manifests dm USING (evaluation_config_id)
   WHERE ec.evaluation_config_id=previous AND ec.eval_n=0
   AND (SELECT count(*) FROM control_plane.dataset_manifests WHERE evaluation_config_id=previous)=1
   AND dm.sample_proportion=1
   {source_guard}
 ) THEN RAISE EXCEPTION 'Active category weights, sources or budget differ; review migration'; END IF;
 SELECT evaluation_config_id INTO eid FROM control_plane.evaluation_configs
 WHERE competition_id=cid AND config_version={literal(version)};
 IF eid IS NOT NULL AND NOT EXISTS (
   SELECT 1 FROM control_plane.evaluation_configs ec
   JOIN control_plane.dataset_manifests dm USING (evaluation_config_id)
   WHERE ec.evaluation_config_id=eid AND ec.eval_n=0 AND ec.delta_threshold={threshold}
   AND ec.long_documents={content}::jsonb AND dm.manifest_json={content}::jsonb
   AND dm.manifest_url={literal(url)} AND dm.manifest_sha256={literal(entry['sha256'])}
   AND dm.sample_proportion=1 AND dm.position=0
   AND (SELECT count(*) FROM control_plane.dataset_manifests WHERE evaluation_config_id=eid)=1
 ) THEN RAISE EXCEPTION 'Existing destination config differs from staged content'; END IF;
 IF eid IS NULL THEN
   INSERT INTO control_plane.evaluation_configs
   (competition_id, config_version, dataset_label, eval_n, delta_threshold,
    shards_per_dataset, long_documents, active)
   VALUES (cid,{literal(version)},{literal(mix['competition'])},0,{threshold},1,{content}::jsonb,false)
   RETURNING evaluation_config_id INTO eid;
   INSERT INTO control_plane.dataset_manifests
   (evaluation_config_id,position,name,manifest_url,manifest_sha256,manifest_json,sample_proportion)
   VALUES (eid,0,{literal(mix['competition'])},{literal(url)},{literal(entry['sha256'])},{content}::jsonb,1);
 END IF;
 UPDATE control_plane.evaluation_configs SET active=false WHERE competition_id=cid AND active;
 UPDATE control_plane.evaluation_configs SET active=true WHERE evaluation_config_id=eid;
 -- Only the legacy token-to-text cutover changes the early-stopping policy.
 IF EXISTS (SELECT 1 FROM control_plane.dataset_manifests
            WHERE evaluation_config_id=previous
            AND manifest_json->>'format' IS DISTINCT FROM 'teutonic-packed-text-mix') THEN
   INSERT INTO control_plane.evaluation_early_stopping_policies (competition_id, enabled)
   VALUES (cid,false) ON CONFLICT (competition_id) DO UPDATE SET enabled=false;
 END IF;
END $migration$;
""")
    if len(blocks) == 1:
        raise ValueError("bundle has no competition manifests")
    if previous_configs is not None and set(previous_configs) != staged:
        raise ValueError("previous configurations and bundle competitions differ")
    return "\n".join(blocks) + "\nCOMMIT;\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--netuid", type=int, required=True)
    p.add_argument("--chain-generation", required=True)
    p.add_argument("--delta-bpb", type=float, required=True,
                   help="explicit new-unit threshold; do not silently reuse nats/token")
    p.add_argument("--previous-token-budget", type=int,
                   help="explicit expected live budget when changing the budget at cutover")
    p.add_argument("--previous-configs", type=Path,
                   help="JSON competition-to-config-version map; allow only dataset/sampling pin changes")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(stage(args.bundle_dir, netuid=args.netuid,
                               generation=args.chain_generation, threshold=args.delta_bpb,
                               previous_token_budget=args.previous_token_budget,
                               previous_configs=json.loads(args.previous_configs.read_text())
                               if args.previous_configs else None))
    print("Prepared local SQL only; no database or R2 operations performed.")


if __name__ == "__main__":
    main()
