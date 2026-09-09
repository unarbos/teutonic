\set ON_ERROR_STOP on

BEGIN;

ALTER TABLE control_plane.evaluation_configs
    ADD COLUMN IF NOT EXISTS shards_per_dataset integer DEFAULT 4 NOT NULL;

ALTER TABLE control_plane.evaluation_configs
    ADD CONSTRAINT evaluation_configs_shards_per_dataset_check
        CHECK (shards_per_dataset > 0);

COMMIT;
