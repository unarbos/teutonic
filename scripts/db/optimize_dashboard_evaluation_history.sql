\set ON_ERROR_STOP on

BEGIN;

CREATE OR REPLACE VIEW control_plane.dashboard_evaluation_history
WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    SUBSTRING(encode(public.digest((e.upload_id)::text, 'sha256'::text), 'hex'::text) FROM 1 FOR 16) AS challenge_id,
    r.hotkey,
    identity.coldkey,
    r.uid,
    baseline.hotkey AS baseline_hotkey,
    COALESCE(baseline_identity.coldkey, baseline_identity_fallback.coldkey) AS baseline_coldkey,
    baseline.uid AS baseline_uid,
        CASE
            WHEN ((e.state = 'terminal_failure'::text) OR (e.verdict = 'failed'::text)) THEN 'failed'::text
            WHEN (e.verdict = 'accepted'::text) THEN 'accepted'::text
            ELSE 'rejected'::text
        END AS verdict,
    (e.verdict_summary ->> 'mu_hat'::text) AS mu_hat,
    (e.verdict_summary ->> 'lcb'::text) AS lcb,
    COALESCE((e.verdict_summary ->> 'delta'::text), (e.verdict_summary ->> 'delta_threshold'::text)) AS delta,
    (e.verdict_summary ->> 'avg_king_loss'::text) AS avg_king_loss,
    (e.verdict_summary ->> 'avg_challenger_loss'::text) AS avg_challenger_loss,
    (e.verdict_summary ->> 'wall_time_s'::text) AS wall_time_s,
    (e.verdict_summary ->> 'n_sequences_evaluated'::text) AS n_sequences_evaluated,
    (e.verdict_summary ->> 'n_sequences'::text) AS n_sequences,
    (e.verdict_summary ->> 'early_stopped'::text) AS early_stopped,
        CASE
            WHEN (e.public_error_code = ANY (ARRAY['invalid_evaluation_input'::text, 'config_rejected'::text, 'model_copy'::text, 'evaluator_busy'::text, 'evaluator_job_lost'::text, 'evaluation_failed'::text, 'protocol_invalid'::text, 'retry_exhausted'::text, 'safetensors_reuse_limit'::text])) THEN e.public_error_code
            WHEN (e.public_error_code IS NULL) THEN NULL::text
            ELSE 'evaluation_failed'::text
        END AS public_error_code,
    e.policy_version,
    e.dataset_version,
    e.completed_at,
        CASE
            WHEN (p.state = 'promoted'::text) THEN u.model_name
            ELSE NULL::text
        END AS public_model_name,
        CASE
            WHEN (p.state = 'promoted'::text) THEN p.model_digest
            ELSE NULL::bpchar
        END AS public_model_digest,
        CASE
            WHEN (p.state = 'promoted'::text) THEN p.public_prefix
            ELSE NULL::text
        END AS public_model_reference,
        CASE
            WHEN (p.state = 'promoted'::text) THEN p.disposition
            ELSE NULL::text
        END AS publication_disposition,
    COALESCE(( SELECT jsonb_agg(jsonb_build_object('source', COALESCE(NULLIF((shard_group.value ->> 'source'::text), ''::text), 'dataset'::text), 'names', COALESCE(( SELECT jsonb_agg(regexp_replace(split_part(split_part((shard_ref.value #>> '{}'::text[]), '?'::text, 1), '#'::text, 1), '^.*/'::text, ''::text) ORDER BY shard_ref.ordinality)
                   FROM jsonb_array_elements(
                        CASE
                            WHEN (jsonb_typeof((shard_group.value -> 'refs'::text)) = 'array'::text) THEN (shard_group.value -> 'refs'::text)
                            ELSE '[]'::jsonb
                        END) WITH ORDINALITY shard_ref(value, ordinality)
                  WHERE (jsonb_typeof(shard_ref.value) = 'string'::text)), '[]'::jsonb)) ORDER BY shard_group.ordinality)
           FROM jsonb_array_elements(
                CASE
                    WHEN (jsonb_typeof(COALESCE((e.verdict_summary -> 'shards_used'::text), (e.verdict_summary #> '{dataset,shards_used}'::text[]), '[]'::jsonb)) = 'array'::text) THEN COALESCE((e.verdict_summary -> 'shards_used'::text), (e.verdict_summary #> '{dataset,shards_used}'::text[]), '[]'::jsonb)
                    ELSE '[]'::jsonb
                END) WITH ORDINALITY shard_group(value, ordinality)
          WHERE (jsonb_typeof(shard_group.value) = 'object'::text)), '[]'::jsonb) AS shards_used
   FROM ((((((((((((control_plane.evaluations e
     JOIN control_plane.uploads u ON ((u.upload_id = e.upload_id)))
     JOIN control_plane.registrations r ON ((r.registration_id = u.registration_id)))
     JOIN control_plane.competitions c ON ((c.competition_id = e.competition_id)))
     JOIN control_plane.king_reigns baseline ON ((baseline.reign_id = e.claimed_king_reign_id)))
     LEFT JOIN control_plane.model_promotions p ON (((p.evaluation_id = e.evaluation_id) AND (p.state = 'promoted'::text) AND ((p.disposition = 'non_winner'::text) OR (EXISTS ( SELECT 1
           FROM control_plane.king_reigns published_winner
          WHERE ((published_winner.causing_evaluation_id = e.evaluation_id) AND (published_winner.accepted_upload_id = e.upload_id) AND (published_winner.model_digest = p.model_digest))))))))
     LEFT JOIN control_plane.metagraph_snapshots identity_snapshot ON (((identity_snapshot.netuid = r.netuid) AND (identity_snapshot.chain_generation = r.chain_generation) AND (identity_snapshot.finalized_block = r.last_seen_finalized_block) AND identity_snapshot.is_complete)))
     LEFT JOIN control_plane.metagraph_uid_assignments identity ON (((identity.snapshot_id = identity_snapshot.snapshot_id) AND (identity.uid = r.uid) AND (identity.hotkey = r.hotkey))))
     LEFT JOIN control_plane.uploads baseline_upload ON ((baseline_upload.upload_id = baseline.accepted_upload_id)))
     LEFT JOIN control_plane.registrations baseline_registration ON (((baseline_registration.registration_id = baseline_upload.registration_id) AND (baseline_registration.hotkey = baseline.hotkey))))
     LEFT JOIN control_plane.metagraph_snapshots baseline_snapshot ON (((baseline_snapshot.netuid = c.netuid) AND (baseline_snapshot.chain_generation = c.chain_generation) AND (baseline_snapshot.finalized_block = baseline_registration.last_seen_finalized_block) AND baseline_snapshot.is_complete)))
     LEFT JOIN control_plane.metagraph_uid_assignments baseline_identity ON (((baseline_identity.snapshot_id = baseline_snapshot.snapshot_id) AND (baseline_identity.uid = baseline_registration.uid) AND (baseline_identity.hotkey = baseline.hotkey))))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((baseline_identity.coldkey IS NULL) AND (snapshot.netuid = c.netuid) AND (snapshot.chain_generation = c.chain_generation) AND (assignment.hotkey = baseline.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) baseline_identity_fallback ON (true))
  WHERE (e.state = ANY (ARRAY['completed'::text, 'terminal_failure'::text]));

ALTER VIEW control_plane.dashboard_evaluation_history OWNER TO teutonic_schema_owner;
GRANT SELECT ON TABLE control_plane.dashboard_evaluation_history TO teutonic_dashboard_view;

COMMIT;
