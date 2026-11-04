-- Run before activating --n 0 with configure_evaluation.py.
-- This changes validation only; budgets remain versioned configuration data.
BEGIN;
ALTER TABLE control_plane.evaluation_configs
    DROP CONSTRAINT evaluation_configs_eval_n_check;
ALTER TABLE control_plane.evaluation_configs
    ADD CONSTRAINT evaluation_configs_eval_n_check CHECK (
        eval_n >= 0 AND (
            eval_n > 0 OR COALESCE(
                jsonb_typeof(long_documents) = 'object'
                AND jsonb_typeof(long_documents->'token_budget') = 'number'
                AND (long_documents->>'token_budget')::numeric > 0,
                false
            )
        )
    );
COMMIT;
