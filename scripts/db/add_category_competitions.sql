-- Additive preparation only: this does not activate MIXes or retire competitions.
BEGIN;
ALTER TABLE control_plane.competitions
    ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS reward_transition jsonb;
ALTER TABLE control_plane.competitions DROP CONSTRAINT IF EXISTS competitions_competition_key_check;
ALTER TABLE control_plane.competitions ADD CONSTRAINT competitions_competition_key_check
    CHECK (competition_key ~ '^(main|math|code|text|mix(0[1-9]|[1-9][0-9]))$');
ALTER TABLE control_plane.uploads DROP CONSTRAINT IF EXISTS uploads_competition_key_check;
ALTER TABLE control_plane.uploads ADD CONSTRAINT uploads_competition_key_check
    CHECK (competition_key ~ '^(main|math|code|text|mix(0[1-9]|[1-9][0-9]))$');
GRANT SELECT ON control_plane.competitions TO teutonic_access_controller;

CREATE OR REPLACE VIEW control_plane.dashboard_competitions WITH (security_barrier=true) AS
SELECT c.netuid, c.chain_generation, c.name AS competition, c.competition_key,
       main.name AS main_competition, c.current_reign_id IS NOT NULL AS has_king,
       ec.config_version, ec.eval_n, ec.delta_threshold,
       c.active, main.reward_transition,
       COALESCE((ec.long_documents->>'token_budget')::bigint, 0) AS document_tokens
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
         THEN manifest.manifest_json - 'shards' - '_shards_zlib_base64'
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
         THEN manifest.manifest_json - 'shards' - '_shards_zlib_base64'
         ELSE manifest.manifest_json END AS manifest_json
   FROM ((control_plane.competitions competition
     JOIN control_plane.evaluation_configs config ON (config.competition_id = competition.competition_id))
     JOIN control_plane.dataset_manifests manifest ON (manifest.evaluation_config_id = config.evaluation_config_id));
COMMIT;
