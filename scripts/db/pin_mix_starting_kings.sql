\set ON_ERROR_STOP on

-- Pin once to MAIN at application time. Re-running never moves an existing pin.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE control_plane.competitions
    ADD COLUMN IF NOT EXISTS starting_reign_id uuid
    REFERENCES control_plane.king_reigns(reign_id) ON DELETE RESTRICT;

UPDATE control_plane.competitions c
   SET starting_reign_id = main.current_reign_id
  FROM control_plane.competitions main
 WHERE c.main_competition_id = main.competition_id
   AND c.competition_key ~ '^mix[0-9]{2}$' AND c.active
   AND c.current_reign_id IS NULL AND c.starting_reign_id IS NULL
   AND main.current_reign_id IS NOT NULL;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM control_plane.competitions
         WHERE active AND competition_key ~ '^mix[0-9]{2}$'
           AND current_reign_id IS NULL AND starting_reign_id IS NULL
    ) THEN
        RAISE EXCEPTION 'cannot pin a kingless MIX without a current MAIN king';
    END IF;
END $$;

CREATE OR REPLACE VIEW control_plane.dashboard_competitions WITH (security_barrier=true) AS
SELECT c.netuid, c.chain_generation, c.name AS competition, c.competition_key,
       main.name AS main_competition, c.current_reign_id IS NOT NULL AS has_king,
       ec.config_version, ec.eval_n, ec.delta_threshold,
       c.active, main.reward_transition,
       COALESCE((ec.long_documents->>'token_budget')::bigint, 0) AS document_tokens,
       CASE WHEN starting.reign_id IS NOT NULL THEN jsonb_build_object(
           'reign_id', starting.reign_id, 'reign_number', starting.reign_number,
           'model_digest', starting.model_digest, 'model_reference', starting.public_prefix,
           'model_repo', starting_upload.model_name, 'uid', starting.uid, 'hotkey', starting.hotkey
       ) END AS starting_king
  FROM control_plane.competitions c
  JOIN control_plane.competitions main ON main.competition_id=COALESCE(c.main_competition_id,c.competition_id)
  JOIN control_plane.evaluation_configs ec ON ec.competition_id=c.competition_id AND ec.active
  LEFT JOIN control_plane.king_reigns starting ON starting.reign_id=c.starting_reign_id
  LEFT JOIN control_plane.uploads starting_upload ON starting_upload.upload_id=starting.accepted_upload_id;
ALTER VIEW control_plane.dashboard_competitions OWNER TO teutonic_schema_owner;
GRANT SELECT ON control_plane.dashboard_competitions TO teutonic_dashboard_view;

SELECT c.name AS competition, starting.reign_id AS starting_reign_id,
       starting.uid AS starting_uid, starting.model_digest AS starting_model_digest
  FROM control_plane.competitions c
  LEFT JOIN control_plane.king_reigns starting ON starting.reign_id=c.starting_reign_id
 WHERE c.active AND c.competition_key ~ '^mix[0-9]{2}$' AND c.current_reign_id IS NULL
 ORDER BY c.netuid, c.chain_generation, c.name;
COMMIT;
