\set ON_ERROR_STOP on
-- Identifier support only. No activation, retirement, config or reward changes.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE control_plane.competitions DROP CONSTRAINT IF EXISTS competitions_competition_key_check;
ALTER TABLE control_plane.competitions ADD CONSTRAINT competitions_competition_key_check
    CHECK (competition_key ~ '^(main|main2|math|code|text|mix(0[1-9]|[1-9][0-9]))$');
ALTER TABLE control_plane.uploads DROP CONSTRAINT IF EXISTS uploads_competition_key_check;
ALTER TABLE control_plane.uploads ADD CONSTRAINT uploads_competition_key_check
    CHECK (competition_key ~ '^(main|main2|math|code|text|mix(0[1-9]|[1-9][0-9]))$');
COMMIT;
