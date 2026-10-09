-- Run before deploying validator/configuration tools with long-document support.
BEGIN;
ALTER TABLE control_plane.evaluation_configs
    ADD COLUMN IF NOT EXISTS long_documents jsonb;
COMMIT;
