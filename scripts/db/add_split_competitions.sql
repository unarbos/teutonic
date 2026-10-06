-- Apply with psql -v ON_ERROR_STOP=1 before deploying the new services.
BEGIN;
ALTER TABLE control_plane.competitions
    ADD COLUMN IF NOT EXISTS competition_key text NOT NULL DEFAULT 'main'
        CHECK (competition_key IN ('main', 'math', 'code', 'text')),
    ADD COLUMN IF NOT EXISTS main_competition_id uuid REFERENCES control_plane.competitions(competition_id),
    ADD COLUMN IF NOT EXISTS reward_reign_id uuid,
    ADD COLUMN IF NOT EXISTS reward_main_hotkeys text[];
ALTER TABLE control_plane.uploads
    ADD COLUMN IF NOT EXISTS competition_key text NOT NULL DEFAULT 'main'
        CHECK (competition_key IN ('main', 'math', 'code', 'text'));
ALTER TABLE control_plane.weight_publications
    ADD COLUMN IF NOT EXISTS policy_weights double precision[];
-- Preserve the exact currently paid main history, including genesis recipients.
UPDATE control_plane.competitions c
   SET reward_main_hotkeys = w.policy_hotkeys
  FROM control_plane.weight_publications w
 WHERE w.source_reign_id = c.current_reign_id
   AND c.competition_key = 'main' AND c.reward_main_hotkeys IS NULL;
CREATE OR REPLACE VIEW control_plane.dashboard_queue WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    SUBSTRING(encode(public.digest((u.upload_id)::text, 'sha256'::text), 'hex'::text) FROM 1 FOR 16) AS challenge_id,
    r.hotkey,
    identity.coldkey,
    r.uid,
    u.ready_finalized_block,
    row_number() OVER (PARTITION BY c.netuid, c.chain_generation ORDER BY u.ready_finalized_block, u.ready_extrinsic_index, u.ready_event_index, u.upload_id) AS queue_position,
        CASE
            WHEN (u.state = 'ready_for_evaluation'::text) THEN 'queued'::text
            WHEN (u.state = 'retry_pending'::text) THEN 'retrying'::text
            ELSE 'processing'::text
        END AS state,
    u.ready_at AS submitted_at
   FROM (((control_plane.uploads u
     JOIN control_plane.registrations r ON ((r.registration_id = u.registration_id)))
     JOIN control_plane.competitions c ON (((c.netuid = r.netuid) AND (c.chain_generation = r.chain_generation) AND (c.competition_key = u.competition_key))))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = r.netuid) AND (snapshot.chain_generation = r.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true))
  WHERE (u.state = ANY (ARRAY['ready_for_evaluation'::text, 'retry_pending'::text]));



CREATE OR REPLACE VIEW control_plane.dashboard_current_king WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    r.reign_number,
    r.hotkey,
    identity.coldkey,
    COALESCE(current_weights.target_uids[array_position(current_weights.target_hotkeys, r.hotkey)], r.uid) AS uid,
    public_upload.model_name AS public_model_name,
    r.model_digest AS public_model_digest,
    r.public_prefix AS public_model_reference,
    r.crowned_at,
    r.crowned_finalized_block,
    (cause.verdict_summary ->> 'mu_hat'::text) AS mu_hat,
    (cause.verdict_summary ->> 'lcb'::text) AS lcb,
    COALESCE((cause.verdict_summary ->> 'delta'::text), (cause.verdict_summary ->> 'delta_threshold'::text)) AS delta,
    (cause.verdict_summary ->> 'avg_king_loss'::text) AS avg_king_loss,
    (cause.verdict_summary ->> 'avg_challenger_loss'::text) AS avg_challenger_loss,
    (cause.verdict_summary ->> 'wall_time_s'::text) AS wall_time_s,
    current_weights.normalized_weights[array_position(current_weights.target_hotkeys, r.hotkey)] AS current_weight
   FROM (((((control_plane.competitions c
     JOIN control_plane.king_reigns r ON ((r.reign_id = c.current_reign_id)))
     LEFT JOIN control_plane.uploads public_upload ON ((public_upload.upload_id = r.accepted_upload_id)))
     LEFT JOIN control_plane.evaluations cause ON ((cause.evaluation_id = r.causing_evaluation_id)))
     LEFT JOIN control_plane.weight_publications current_weights ON ((current_weights.source_reign_id = (SELECT COALESCE(main.reward_reign_id, main.current_reign_id) FROM control_plane.competitions main WHERE main.competition_id = COALESCE(c.main_competition_id, c.competition_id)))))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = c.netuid) AND (snapshot.chain_generation = c.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true));



CREATE OR REPLACE VIEW control_plane.dashboard_king_reigns WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    r.reign_number,
    r.hotkey,
    identity.coldkey,
    COALESCE(current_weights.target_uids[array_position(current_weights.target_hotkeys, r.hotkey)], r.uid) AS uid,
    public_upload.model_name AS public_model_name,
    r.model_digest AS public_model_digest,
    r.public_prefix AS public_model_reference,
    r.crowned_at,
    r.crowned_finalized_block,
    r.ended_at,
        CASE
            WHEN (r.replacement_reason = ANY (ARRAY['accepted_challenger'::text, 'operator_seed'::text, 'policy_reset'::text])) THEN r.replacement_reason
            WHEN (r.replacement_reason IS NULL) THEN NULL::text
            ELSE 'replaced'::text
        END AS replacement_reason,
    current_weights.normalized_weights[array_position(current_weights.target_hotkeys, r.hotkey)] AS current_weight
   FROM ((((control_plane.king_reigns r
     JOIN control_plane.competitions c ON ((c.competition_id = r.competition_id)))
     LEFT JOIN control_plane.uploads public_upload ON ((public_upload.upload_id = r.accepted_upload_id)))
     LEFT JOIN control_plane.weight_publications current_weights ON ((current_weights.source_reign_id = (SELECT COALESCE(main.reward_reign_id, main.current_reign_id) FROM control_plane.competitions main WHERE main.competition_id = COALESCE(c.main_competition_id, c.competition_id)))))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = c.netuid) AND (snapshot.chain_generation = c.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true));



CREATE OR REPLACE VIEW control_plane.dashboard_weight_status WITH (security_barrier='true') AS
 SELECT DISTINCT ON (c.competition_id) c.netuid,
    c.chain_generation,
    c.name AS competition,
    weights.state,
    weights.cadence_blocks,
    weights.last_attempted_block,
    weights.next_due_block,
    attempt.state AS latest_attempt_state,
    attempt.scheduled_block AS latest_attempted_block,
    attempt.finalized_block AS latest_finalized_block,
        CASE
            WHEN (attempt.last_error_code = ANY (ARRAY['chain_unavailable'::text, 'submission_rejected'::text, 'finality_timeout'::text, 'cadence_elapsed'::text, 'newer_reign'::text])) THEN attempt.last_error_code
            WHEN (attempt.last_error_code IS NULL) THEN NULL::text
            ELSE 'weight_publication_failed'::text
        END AS public_error_code,
    weights.requested_at,
    attempt.submitted_at,
    attempt.finalized_at
   FROM ((control_plane.competitions c
     JOIN control_plane.weight_publications weights ON ((weights.source_reign_id = (SELECT COALESCE(main.reward_reign_id, main.current_reign_id) FROM control_plane.competitions main WHERE main.competition_id = COALESCE(c.main_competition_id, c.competition_id)))))
     LEFT JOIN LATERAL ( SELECT a.state,
            a.scheduled_block,
            a.finalized_block,
            a.last_error_code,
            a.submitted_at,
            a.finalized_at
           FROM control_plane.weight_submission_attempts a
          WHERE (a.weight_publication_id = weights.weight_publication_id)
          ORDER BY a.sequence DESC
         LIMIT 1) attempt ON (true))
  ORDER BY c.competition_id, weights.requested_at DESC;



CREATE OR REPLACE VIEW control_plane.dashboard_upload_failures WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    SUBSTRING(encode(public.digest((u.upload_id)::text, 'sha256'::text), 'hex'::text) FROM 1 FOR 16) AS challenge_id,
    r.hotkey,
    identity.coldkey,
    r.uid,
    baseline.hotkey AS baseline_hotkey,
    baseline_identity.coldkey AS baseline_coldkey,
    baseline.uid AS baseline_uid,
    r.state AS registration_state,
    u.upload_id,
    u.state AS upload_state,
        CASE
            WHEN (u.failure_code = ANY (ARRAY['ArtifactIntegrityError'::text, 'GenesisContractMismatch'::text, 'UploadQuotaExceeded'::text])) THEN u.failure_code
            ELSE 'verification_failed'::text
        END AS public_error_code,
    u.updated_at AS failed_at
   FROM (((control_plane.uploads u
     JOIN control_plane.registrations r ON ((r.registration_id = u.registration_id)))
     JOIN control_plane.competitions c ON (((c.netuid = r.netuid) AND (c.chain_generation = r.chain_generation) AND (c.competition_key = u.competition_key))))
     JOIN control_plane.king_reigns baseline ON ((baseline.reign_id = c.current_reign_id)))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = r.netuid) AND (snapshot.chain_generation = r.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true)
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = c.netuid) AND (snapshot.chain_generation = c.chain_generation) AND (assignment.hotkey = baseline.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) baseline_identity ON (true)
  WHERE (u.state = 'verification_failed'::text);



CREATE OR REPLACE VIEW control_plane.dashboard_stats WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    revision.revision AS source_watermark,
    ( SELECT count(*) AS count
           FROM control_plane.registrations r
          WHERE ((r.netuid = c.netuid) AND (r.chain_generation = c.chain_generation) AND (r.state = 'active'::text))) AS active_registrations,
    ( SELECT count(*) AS count
           FROM (control_plane.uploads u
             JOIN control_plane.registrations r ON ((r.registration_id = u.registration_id)))
          WHERE ((r.netuid = c.netuid) AND (r.chain_generation = c.chain_generation) AND (u.competition_key = c.competition_key) AND (u.state = ANY (ARRAY['ready_for_evaluation'::text, 'retry_pending'::text])))) AS queue_depth,
    ( SELECT count(*) AS count
           FROM control_plane.evaluations e
          WHERE ((e.competition_id = c.competition_id) AND (e.state = ANY (ARRAY['completed'::text, 'terminal_failure'::text])))) AS completed_evaluations,
    ( SELECT count(*) AS count
           FROM control_plane.king_reigns reign
          WHERE (reign.competition_id = c.competition_id)) AS reign_count
   FROM (control_plane.competitions c
     CROSS JOIN control_plane.public_state_revision revision)
  WHERE revision.singleton;



CREATE OR REPLACE VIEW control_plane.dashboard_competitions WITH (security_barrier=true) AS
SELECT c.netuid, c.chain_generation, c.name AS competition, c.competition_key,
       main.name AS main_competition, c.current_reign_id IS NOT NULL AS has_king,
       ec.config_version, ec.eval_n, ec.delta_threshold
  FROM control_plane.competitions c
  JOIN control_plane.competitions main ON main.competition_id=COALESCE(c.main_competition_id,c.competition_id)
  JOIN control_plane.evaluation_configs ec ON ec.competition_id=c.competition_id AND ec.active;
ALTER VIEW control_plane.dashboard_competitions OWNER TO teutonic_schema_owner;
GRANT SELECT ON control_plane.dashboard_competitions TO teutonic_dashboard_view;

CREATE OR REPLACE VIEW control_plane.dashboard_dataset_manifests WITH (security_barrier='true') AS
 SELECT competition.netuid,
    competition.chain_generation,
    competition.name AS competition,
    config.config_version,
    config.dataset_label,
    config.eval_n,
    config.delta_threshold,
    config.created_at AS config_created_at,
    manifest."position",
    manifest.name,
    manifest.manifest_url,
    manifest.manifest_sha256,
    manifest.sample_proportion,
    CASE WHEN manifest.manifest_json ? 'total_tokens' AND manifest.manifest_json ? 'total_shards'
         THEN manifest.manifest_json - 'shards'
         ELSE manifest.manifest_json END AS manifest_json
   FROM ((control_plane.competitions competition
     JOIN control_plane.evaluation_configs config ON (((config.competition_id = competition.competition_id) AND config.active)))
     JOIN control_plane.dataset_manifests manifest ON ((manifest.evaluation_config_id = config.evaluation_config_id)));



CREATE OR REPLACE VIEW control_plane.dashboard_dataset_versions WITH (security_barrier='true') AS
 SELECT competition.netuid,
    competition.chain_generation,
    competition.name AS competition,
    config.config_version,
    config.dataset_label,
    config.eval_n,
    config.delta_threshold,
    config.created_at AS config_created_at,
    manifest."position",
    manifest.name,
    manifest.manifest_url,
    manifest.manifest_sha256,
    manifest.sample_proportion,
    CASE WHEN manifest.manifest_json ? 'total_tokens' AND manifest.manifest_json ? 'total_shards'
         THEN manifest.manifest_json - 'shards'
         ELSE manifest.manifest_json END AS manifest_json
   FROM ((control_plane.competitions competition
     JOIN control_plane.evaluation_configs config ON (config.competition_id = competition.competition_id))
     JOIN control_plane.dataset_manifests manifest ON (manifest.evaluation_config_id = config.evaluation_config_id));



COMMIT;
