-- Teutonic control-plane fresh-start schema.
-- Loaded by the PostgreSQL Docker entrypoint for an empty data directory.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $roles$
DECLARE
    role_name text;
BEGIN
    FOREACH role_name IN ARRAY ARRAY[
        'teutonic_schema_owner',
        'teutonic_access_controller',
        'teutonic_validator',
        'teutonic_weight_publisher',
        'teutonic_dashboard_view',
        'teutonic_auditor'
    ]
    LOOP
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = role_name) THEN
            EXECUTE format(
                'CREATE ROLE %I NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT',
                role_name
            );
        END IF;
    END LOOP;
END
$roles$;

--
-- PostgreSQL database dump
--

-- Dumped from database version 16.4
-- Dumped by pg_dump version 16.4

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: control_plane; Type: SCHEMA; Schema: -; Owner: teutonic_schema_owner
--

CREATE SCHEMA control_plane;


ALTER SCHEMA control_plane OWNER TO teutonic_schema_owner;

--
-- Name: bump_public_state_revision(); Type: FUNCTION; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE FUNCTION control_plane.bump_public_state_revision() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'control_plane', 'pg_temp'
    AS $$
BEGIN
    UPDATE public_state_revision
       SET revision = revision + 1,
           updated_at = clock_timestamp()
     WHERE singleton;
    RETURN NULL;
END
$$;


ALTER FUNCTION control_plane.bump_public_state_revision() OWNER TO teutonic_schema_owner;

--
-- Name: prevent_parent_token_reactivation(); Type: FUNCTION; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE FUNCTION control_plane.prevent_parent_token_reactivation() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'control_plane', 'pg_temp'
    AS $$
BEGIN
    IF NEW.state IN ('pending_create', 'creating', 'active')
       AND EXISTS (
           SELECT 1 FROM uploads submitted
           JOIN registrations target ON target.registration_id = NEW.registration_id
          WHERE submitted.signalling_hotkey = target.hotkey
            AND submitted.ready_at IS NOT NULL
       ) THEN
        RAISE EXCEPTION 'parent token cannot be activated after this hotkey submitted a model'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END
$$;


ALTER FUNCTION control_plane.prevent_parent_token_reactivation() OWNER TO teutonic_schema_owner;

--
-- Name: prevent_upload_access_reissue(); Type: FUNCTION; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE FUNCTION control_plane.prevent_upload_access_reissue() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'control_plane', 'pg_temp'
    AS $$
BEGIN
    IF NEW.state IN ('pending', 'publishing', 'published')
       AND EXISTS (
           SELECT 1 FROM uploads submitted
           JOIN registrations target ON target.registration_id = NEW.registration_id
          WHERE submitted.signalling_hotkey = target.hotkey
            AND submitted.ready_at IS NOT NULL
       ) THEN
        RAISE EXCEPTION 'upload access cannot be issued after this hotkey submitted a model'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END
$$;


ALTER FUNCTION control_plane.prevent_upload_access_reissue() OWNER TO teutonic_schema_owner;

--
-- Name: revoke_upload_access_after_ready(); Type: FUNCTION; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE FUNCTION control_plane.revoke_upload_access_after_ready() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'control_plane', 'pg_temp'
    AS $$
DECLARE
    token r2_parent_tokens%ROWTYPE;
    registration_hotkey text;
BEGIN
    IF NEW.ready_at IS NULL OR (TG_OP = 'UPDATE' AND OLD.ready_at IS NOT NULL) THEN
        RETURN NEW;
    END IF;

    SELECT hotkey INTO registration_hotkey
      FROM registrations
     WHERE registration_id = NEW.registration_id;

    IF NEW.signalling_hotkey IS DISTINCT FROM registration_hotkey THEN
        RAISE EXCEPTION 'submission hotkey does not own the registration'
            USING ERRCODE = '23514';
    END IF;

    SELECT * INTO token
      FROM r2_parent_tokens
     WHERE registration_id = NEW.registration_id
     FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'cannot accept submission without a registration parent token'
            USING ERRCODE = '23514';
    END IF;

    UPDATE credential_generations
       SET state = 'superseded'
     WHERE registration_id = NEW.registration_id
       AND state IN ('pending', 'publishing', 'published');

    IF token.state <> 'revoked' THEN
        UPDATE r2_parent_tokens
           SET state = 'pending_revoke',
               revocation_requested_at = COALESCE(revocation_requested_at, clock_timestamp()),
               revocation_reason = COALESCE(revocation_reason, 'model_submitted'),
               next_retry_at = clock_timestamp(),
               updated_at = clock_timestamp()
         WHERE parent_token_id = token.parent_token_id;

        INSERT INTO controller_jobs (
            registration_id,
            upload_id,
            operation,
            idempotency_key,
            state,
            next_retry_at
        ) VALUES (
            NEW.registration_id,
            NEW.upload_id,
            'revoke_parent_token',
            'revoke-parent-after-submit:' || token.parent_token_id::text,
            'pending',
            clock_timestamp()
        ) ON CONFLICT (idempotency_key) DO NOTHING;
    END IF;

    RETURN NEW;
END
$$;


ALTER FUNCTION control_plane.revoke_upload_access_after_ready() OWNER TO teutonic_schema_owner;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: chain_cursors; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.chain_cursors (
    netuid integer NOT NULL,
    chain_generation text NOT NULL,
    finalized_start_block bigint NOT NULL,
    last_finalized_block bigint NOT NULL,
    last_finalized_block_hash text NOT NULL,
    snapshot_checksum character(64) NOT NULL,
    observed_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT chain_cursors_chain_generation_check CHECK (((length(chain_generation) >= 1) AND (length(chain_generation) <= 128))),
    CONSTRAINT chain_cursors_check CHECK ((last_finalized_block >= finalized_start_block)),
    CONSTRAINT chain_cursors_finalized_start_block_check CHECK ((finalized_start_block >= 0)),
    CONSTRAINT chain_cursors_netuid_check CHECK ((netuid >= 0)),
    CONSTRAINT chain_cursors_snapshot_checksum_check CHECK ((snapshot_checksum ~ '^[0-9a-f]{64}$'::text))
);


ALTER TABLE control_plane.chain_cursors OWNER TO teutonic_schema_owner;

--
-- Name: competitions; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.competitions (
    competition_id uuid DEFAULT gen_random_uuid() NOT NULL,
    netuid integer NOT NULL,
    chain_generation text NOT NULL,
    name text NOT NULL,
    current_reign_id uuid,
    next_reign_number bigint DEFAULT 1 NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT competitions_netuid_check CHECK ((netuid >= 0)),
    CONSTRAINT competitions_next_reign_number_check CHECK ((next_reign_number > 0))
);


ALTER TABLE control_plane.competitions OWNER TO teutonic_schema_owner;

--
-- Name: evaluation_early_stopping_policies; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.evaluation_early_stopping_policies (
    competition_id uuid NOT NULL,
    enabled boolean DEFAULT true NOT NULL,
    min_fraction double precision DEFAULT 0.4 NOT NULL,
    advantage_quantile double precision DEFAULT 0.95 NOT NULL,
    margin double precision DEFAULT 0.0 NOT NULL,
    check_interval integer DEFAULT 100 NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT evaluation_early_stopping_check_interval_check CHECK ((check_interval > 0)),
    CONSTRAINT evaluation_early_stopping_min_fraction_check CHECK (((min_fraction > (0)::double precision) AND (min_fraction <= (1)::double precision))),
    CONSTRAINT evaluation_early_stopping_advantage_quantile_check CHECK (((advantage_quantile > (0)::double precision) AND (advantage_quantile <= (1)::double precision))),
    CONSTRAINT evaluation_early_stopping_margin_check CHECK ((margin >= (0)::double precision))
);


ALTER TABLE control_plane.evaluation_early_stopping_policies OWNER TO teutonic_schema_owner;

--
-- Name: evaluation_configs; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.evaluation_configs (
    evaluation_config_id uuid DEFAULT gen_random_uuid() NOT NULL,
    competition_id uuid NOT NULL,
    config_version character(64) NOT NULL,
    dataset_label text NOT NULL,
    eval_n integer NOT NULL,
    delta_threshold double precision NOT NULL,
    shards_per_dataset integer DEFAULT 4 NOT NULL,
    active boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT evaluation_configs_config_version_check CHECK ((config_version ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT evaluation_configs_dataset_label_check CHECK ((dataset_label <> ''::text)),
    CONSTRAINT evaluation_configs_delta_threshold_check CHECK (((delta_threshold >= (0)::double precision) AND (delta_threshold <= (100)::double precision))),
    CONSTRAINT evaluation_configs_eval_n_check CHECK ((eval_n > 0)),
    CONSTRAINT evaluation_configs_shards_per_dataset_check CHECK ((shards_per_dataset > 0))
);


ALTER TABLE control_plane.evaluation_configs OWNER TO teutonic_schema_owner;

--
-- Name: dataset_manifests; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.dataset_manifests (
    dataset_manifest_id uuid DEFAULT gen_random_uuid() NOT NULL,
    evaluation_config_id uuid NOT NULL,
    "position" integer NOT NULL,
    name text NOT NULL,
    manifest_url text NOT NULL,
    manifest_sha256 character(64) NOT NULL,
    manifest_json jsonb NOT NULL,
    sample_proportion double precision NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT dataset_manifests_manifest_json_check CHECK ((jsonb_typeof(manifest_json) = 'object'::text)),
    CONSTRAINT dataset_manifests_manifest_sha256_check CHECK ((manifest_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT dataset_manifests_manifest_url_check CHECK ((manifest_url ~ '^https://'::text)),
    CONSTRAINT dataset_manifests_name_check CHECK ((name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'::text)),
    CONSTRAINT dataset_manifests_position_check CHECK (("position" >= 0)),
    CONSTRAINT dataset_manifests_sample_proportion_check CHECK (((sample_proportion > (0)::double precision) AND (sample_proportion <= (1)::double precision)))
);


ALTER TABLE control_plane.dataset_manifests OWNER TO teutonic_schema_owner;

--
-- Name: controller_jobs; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.controller_jobs (
    controller_job_id uuid DEFAULT gen_random_uuid() NOT NULL,
    registration_id character(64),
    upload_id uuid,
    operation text NOT NULL,
    idempotency_key text NOT NULL,
    state text NOT NULL,
    owner_instance_id text,
    lease_expires_at timestamp with time zone,
    attempt_count integer DEFAULT 0 NOT NULL,
    next_retry_at timestamp with time zone,
    last_error_code text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    completed_at timestamp with time zone,
    payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    result jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT controller_jobs_attempt_count_check CHECK ((attempt_count >= 0)),
    CONSTRAINT controller_jobs_check CHECK (((registration_id IS NOT NULL) OR (upload_id IS NOT NULL))),
    CONSTRAINT controller_jobs_check1 CHECK (((state = ANY (ARRAY['claimed'::text, 'running'::text])) = (lease_expires_at IS NOT NULL))),
    CONSTRAINT controller_jobs_check2 CHECK (((state <> 'completed'::text) OR (completed_at IS NOT NULL))),
    CONSTRAINT controller_jobs_operation_check CHECK ((operation = ANY (ARRAY['create_parent_token'::text, 'publish_credentials'::text, 'revoke_parent_token'::text, 'abort_multipart'::text, 'cleanup_upload'::text, 'verify_upload'::text, 'create_immutable_snapshot'::text]))),
    CONSTRAINT controller_jobs_state_check CHECK ((state = ANY (ARRAY['pending'::text, 'claimed'::text, 'running'::text, 'retry_pending'::text, 'completed'::text, 'failed'::text])))
);


ALTER TABLE control_plane.controller_jobs OWNER TO teutonic_schema_owner;

--
-- Name: credential_generations; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.credential_generations (
    registration_id character(64) NOT NULL,
    generation integer NOT NULL,
    issued_at timestamp with time zone NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    bucket_name text NOT NULL,
    allowed_prefix text NOT NULL,
    allowed_actions text[],
    mailbox_object_key text NOT NULL,
    ciphertext_sha256 character(64) NOT NULL,
    state text NOT NULL,
    published_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    capability_scope text DEFAULT 'object-read-write'::text NOT NULL,
    CONSTRAINT credential_generations_broad_prefix_scope CHECK ((allowed_actions IS NULL)),
    CONSTRAINT credential_generations_capability_scope_check CHECK ((capability_scope = 'object-read-write'::text)),
    CONSTRAINT credential_generations_check CHECK ((expires_at > issued_at)),
    CONSTRAINT credential_generations_check1 CHECK ((allowed_prefix = (('models/registrations/'::text || (registration_id)::text) || '/'::text))),
    CONSTRAINT credential_generations_check2 CHECK ((mailbox_object_key = (((('mailbox/v1/'::text || (registration_id)::text) || '/generations/'::text) || lpad((generation)::text, 20, '0'::text)) || '.bin'::text))),
    CONSTRAINT credential_generations_check3 CHECK (((state <> 'published'::text) OR (published_at IS NOT NULL))),
    CONSTRAINT credential_generations_ciphertext_sha256_check CHECK ((ciphertext_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT credential_generations_generation_check CHECK ((generation > 0)),
    CONSTRAINT credential_generations_state_check CHECK ((state = ANY (ARRAY['pending'::text, 'publishing'::text, 'published'::text, 'superseded'::text, 'failed'::text])))
);


ALTER TABLE control_plane.credential_generations OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_chain; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_chain WITH (security_barrier='true') AS
 SELECT cursor.netuid,
    cursor.chain_generation,
    cursor.finalized_start_block,
    cursor.last_finalized_block,
    cursor.observed_at,
    competition.name AS competition
   FROM (control_plane.chain_cursors cursor
     LEFT JOIN control_plane.competitions competition ON (((competition.netuid = cursor.netuid) AND (competition.chain_generation = cursor.chain_generation))));


ALTER VIEW control_plane.dashboard_chain OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_contract; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_contract WITH (security_barrier='true') AS
 SELECT 1 AS schema_version;


ALTER VIEW control_plane.dashboard_contract OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_dataset_manifests; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_dataset_manifests WITH (security_barrier='true') AS
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
    manifest.manifest_json
   FROM ((control_plane.competitions competition
     JOIN control_plane.evaluation_configs config ON (((config.competition_id = competition.competition_id) AND config.active)))
     JOIN control_plane.dataset_manifests manifest ON ((manifest.evaluation_config_id = config.evaluation_config_id)));


ALTER VIEW control_plane.dashboard_dataset_manifests OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_dataset_versions; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_dataset_versions WITH (security_barrier='true') AS
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
    manifest.manifest_json
   FROM ((control_plane.competitions competition
     JOIN control_plane.evaluation_configs config ON (config.competition_id = competition.competition_id))
     JOIN control_plane.dataset_manifests manifest ON (manifest.evaluation_config_id = config.evaluation_config_id));


ALTER VIEW control_plane.dashboard_dataset_versions OWNER TO teutonic_schema_owner;

--
-- Name: evaluations; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.evaluations (
    evaluation_id uuid DEFAULT gen_random_uuid() NOT NULL,
    upload_id uuid NOT NULL,
    competition_id uuid NOT NULL,
    attempt_number integer NOT NULL,
    claimed_king_reign_id uuid NOT NULL,
    state text NOT NULL,
    owner_instance_id text,
    lease_expires_at timestamp with time zone,
    heartbeat_at timestamp with time zone,
    policy_version text NOT NULL,
    code_version text NOT NULL,
    dataset_version text NOT NULL,
    evaluator_version text,
    sampling_seed bigint NOT NULL,
    bootstrap_seed bigint NOT NULL,
    thresholds jsonb NOT NULL,
    progress_summary jsonb,
    verdict text,
    verdict_summary jsonb,
    public_error_code text,
    private_diagnostic_reference text,
    result_artifact_reference text,
    result_artifact_sha256 character(64),
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    evaluator_job_id text,
    request_sha256 character(64),
    request_payload jsonb,
    failure_class text,
    next_retry_at timestamp with time zone,
    CONSTRAINT evaluations_attempt_number_check CHECK ((attempt_number > 0)),
    CONSTRAINT evaluations_check CHECK (((state = ANY (ARRAY['claimed'::text, 'evaluating'::text])) = (lease_expires_at IS NOT NULL))),
    CONSTRAINT evaluations_check1 CHECK (((state <> 'completed'::text) OR ((verdict IS NOT NULL) AND (verdict_summary IS NOT NULL) AND (completed_at IS NOT NULL)))),
    CONSTRAINT evaluations_failure_class_check CHECK ((failure_class = ANY (ARRAY['transient_infrastructure'::text, 'deterministic_submission'::text, 'policy'::text, 'unknown'::text]))),
    CONSTRAINT evaluations_request_sha256_check CHECK ((request_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT evaluations_result_artifact_sha256_check CHECK ((result_artifact_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT evaluations_state_check CHECK ((state = ANY (ARRAY['claimed'::text, 'evaluating'::text, 'retryable_failure'::text, 'lost'::text, 'completed'::text, 'terminal_failure'::text]))),
    CONSTRAINT evaluations_verdict_check CHECK ((verdict = ANY (ARRAY['accepted'::text, 'rejected'::text, 'failed'::text])))
);


ALTER TABLE control_plane.evaluations OWNER TO teutonic_schema_owner;

--
-- Name: metagraph_snapshots; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.metagraph_snapshots (
    snapshot_id uuid DEFAULT gen_random_uuid() NOT NULL,
    netuid integer NOT NULL,
    chain_generation text NOT NULL,
    finalized_block bigint NOT NULL,
    finalized_block_hash text NOT NULL,
    snapshot_checksum character(64) NOT NULL,
    uid_count integer NOT NULL,
    is_complete boolean NOT NULL,
    observed_at timestamp with time zone NOT NULL,
    CONSTRAINT metagraph_snapshots_finalized_block_check CHECK ((finalized_block >= 0)),
    CONSTRAINT metagraph_snapshots_is_complete_check CHECK (is_complete),
    CONSTRAINT metagraph_snapshots_netuid_check CHECK ((netuid >= 0)),
    CONSTRAINT metagraph_snapshots_snapshot_checksum_check CHECK ((snapshot_checksum ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT metagraph_snapshots_uid_count_check CHECK ((uid_count >= 0))
);


ALTER TABLE control_plane.metagraph_snapshots OWNER TO teutonic_schema_owner;

--
-- Name: metagraph_uid_assignments; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.metagraph_uid_assignments (
    snapshot_id uuid NOT NULL,
    uid integer NOT NULL,
    hotkey text,
    coldkey text,
    registration_block bigint,
    CONSTRAINT metagraph_uid_assignments_check CHECK (((hotkey IS NULL) = (coldkey IS NULL))),
    CONSTRAINT metagraph_uid_assignments_registration_check CHECK (((hotkey IS NULL) = (registration_block IS NULL))),
    CONSTRAINT metagraph_uid_assignments_registration_block_check CHECK ((registration_block >= 0)),
    CONSTRAINT metagraph_uid_assignments_uid_check CHECK ((uid >= 0))
);


ALTER TABLE control_plane.metagraph_uid_assignments OWNER TO teutonic_schema_owner;

--
-- Name: registrations; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.registrations (
    registration_id character(64) NOT NULL,
    netuid integer NOT NULL,
    chain_generation text NOT NULL,
    uid integer NOT NULL,
    hotkey text NOT NULL,
    first_seen_finalized_block bigint NOT NULL,
    last_seen_finalized_block bigint NOT NULL,
    deactivated_finalized_block bigint,
    model_prefix text NOT NULL,
    state text NOT NULL,
    deactivation_reason text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    deactivated_at timestamp with time zone,
    activated_at timestamp with time zone,
    activation_finalized_block bigint,
    activation_extrinsic_index integer,
    activation_event_index integer,
    activation_payload text,
    CONSTRAINT registrations_check CHECK ((last_seen_finalized_block >= first_seen_finalized_block)),
    CONSTRAINT registrations_check1 CHECK ((model_prefix = (('models/registrations/'::text || (registration_id)::text) || '/'::text))),
    CONSTRAINT registrations_check2 CHECK ((((state = 'inactive'::text) AND (deactivated_finalized_block IS NOT NULL) AND (deactivated_at IS NOT NULL)) OR ((state <> 'inactive'::text) AND (deactivated_finalized_block IS NULL) AND (deactivated_at IS NULL)))),
    CONSTRAINT registrations_check3 CHECK (((deactivated_finalized_block IS NULL) OR (deactivated_finalized_block >= first_seen_finalized_block))),
    CONSTRAINT registrations_first_seen_finalized_block_check CHECK ((first_seen_finalized_block >= 0)),
    CONSTRAINT registrations_netuid_check CHECK ((netuid >= 0)),
    CONSTRAINT registrations_registration_id_check CHECK ((registration_id ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT registrations_activation_fields_check CHECK (((state <> 'pending_activation'::text) OR (num_nonnulls(activated_at, activation_finalized_block, activation_extrinsic_index, activation_event_index, activation_payload) = 0))),
    CONSTRAINT registrations_activation_group_check CHECK ((num_nonnulls(activated_at, activation_finalized_block, activation_extrinsic_index, activation_event_index, activation_payload) = ANY (ARRAY[0, 5]))),
    CONSTRAINT registrations_activation_finalized_block_check CHECK ((activation_finalized_block >= 0)),
    CONSTRAINT registrations_activation_extrinsic_index_check CHECK ((activation_extrinsic_index >= 0)),
    CONSTRAINT registrations_activation_event_index_check CHECK ((activation_event_index >= 0)),
    CONSTRAINT registrations_activation_payload_check CHECK ((activation_payload IS NULL OR starts_with(activation_payload, 'r2activate:v1:'::text))),
    CONSTRAINT registrations_state_check CHECK ((state = ANY (ARRAY['pending_activation'::text, 'active'::text, 'inactive'::text]))),
    CONSTRAINT registrations_uid_check CHECK ((uid >= 0))
);


ALTER TABLE control_plane.registrations OWNER TO teutonic_schema_owner;

--
-- Name: uploads; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.uploads (
    upload_id uuid DEFAULT gen_random_uuid() NOT NULL,
    registration_id character(64) NOT NULL,
    chain_generation text NOT NULL,
    signalling_hotkey text,
    ready_payload text,
    ready_finalized_block bigint,
    ready_extrinsic_index integer,
    ready_event_index integer,
    manifest_sha256 character(64),
    manifest_signature_verified boolean DEFAULT false NOT NULL,
    model_digest character(64),
    model_name text,
    object_count integer,
    total_size_bytes bigint,
    state text NOT NULL,
    failure_code text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    ready_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT uploads_check CHECK (((state = 'uploading'::text) OR (num_nonnulls(signalling_hotkey, ready_payload, ready_finalized_block, ready_extrinsic_index, ready_event_index, manifest_sha256, ready_at) = 7))),
    CONSTRAINT uploads_manifest_sha256_check CHECK ((manifest_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT uploads_model_digest_check CHECK ((model_digest ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT uploads_object_count_check CHECK ((object_count >= 0)),
    CONSTRAINT uploads_ready_event_index_check CHECK ((ready_event_index >= 0)),
    CONSTRAINT uploads_ready_extrinsic_index_check CHECK ((ready_extrinsic_index >= 0)),
    CONSTRAINT uploads_ready_finalized_block_check CHECK ((ready_finalized_block >= 0)),
    CONSTRAINT uploads_state_check CHECK ((state = ANY (ARRAY['uploading'::text, 'ready_signaled'::text, 'verifying'::text, 'immutable_snapshot_created'::text, 'ready_for_evaluation'::text, 'evaluation_claimed'::text, 'evaluating'::text, 'retry_pending'::text, 'evaluated'::text, 'rejected'::text, 'accepted_pending_promotion'::text, 'promoted'::text, 'accepted'::text, 'invalid_evaluation_input'::text, 'evaluation_failed'::text, 'cancelled_by_policy'::text, 'verification_failed'::text]))),
    CONSTRAINT uploads_total_size_bytes_check CHECK ((total_size_bytes >= 0))
);


ALTER TABLE control_plane.uploads OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_current_evaluation; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_current_evaluation WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    SUBSTRING(encode(public.digest((e.upload_id)::text, 'sha256'::text), 'hex'::text) FROM 1 FOR 16) AS challenge_id,
    r.hotkey,
    identity.coldkey,
    r.uid,
        CASE e.state
            WHEN 'claimed'::text THEN 'preparing'::text
            ELSE 'evaluating'::text
        END AS stage,
    (e.progress_summary ->> 'phase'::text) AS progress_phase,
    COALESCE(e.progress_summary ->> 'completed_sequences'::text, e.progress_summary ->> 'done'::text, '0'::text) AS completed_sequences,
    COALESCE(e.progress_summary ->> 'requested_sequences'::text, e.progress_summary ->> 'total'::text, e.request_payload #>> '{limits,n}'::text[], '0'::text) AS requested_sequences,
    COALESCE(e.progress_summary ->> 'percent'::text,
        CASE
            WHEN COALESCE((e.progress_summary ->> 'requested_sequences'::text)::numeric, (e.progress_summary ->> 'total'::text)::numeric, (e.request_payload #>> '{limits,n}'::text[])::numeric, 0::numeric) > 0::numeric
            THEN ((COALESCE((e.progress_summary ->> 'completed_sequences'::text)::numeric, (e.progress_summary ->> 'done'::text)::numeric, 0::numeric) * 100::numeric) / COALESCE((e.progress_summary ->> 'requested_sequences'::text)::numeric, (e.progress_summary ->> 'total'::text)::numeric, (e.request_payload #>> '{limits,n}'::text[])::numeric))::text
            ELSE '0'::text
        END) AS percent,
    COALESCE(e.progress_summary ->> 'elapsed_seconds'::text, GREATEST(EXTRACT(epoch FROM (e.heartbeat_at - e.started_at)), 0::numeric)::text, '0'::text) AS elapsed_seconds,
    COALESCE(e.progress_summary ->> 'early_stopped'::text, 'false'::text) AS early_stopped,
    e.policy_version,
    e.dataset_version,
    e.started_at,
    e.heartbeat_at AS last_progress_at,
    (e.progress_summary ->> 'provisional_mu_hat'::text) AS provisional_mu_hat,
    (e.progress_summary ->> 'provisional_lcb'::text) AS provisional_lcb,
    (e.progress_summary ->> 'provisional_n_sequences'::text) AS provisional_n_sequences,
    (e.progress_summary ->> 'provisional_n_bootstrap'::text) AS provisional_n_bootstrap,
    (e.request_payload #>> '{limits,delta_threshold}'::text[]) AS delta_threshold
   FROM ((((control_plane.evaluations e
     JOIN control_plane.uploads u ON ((u.upload_id = e.upload_id)))
     JOIN control_plane.registrations r ON ((r.registration_id = u.registration_id)))
     JOIN control_plane.competitions c ON ((c.competition_id = e.competition_id)))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = r.netuid) AND (snapshot.chain_generation = r.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true))
  WHERE (e.state = ANY (ARRAY['claimed'::text, 'evaluating'::text]));


ALTER VIEW control_plane.dashboard_current_evaluation OWNER TO teutonic_schema_owner;

--
-- Name: king_reigns; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.king_reigns (
    reign_id uuid DEFAULT gen_random_uuid() NOT NULL,
    competition_id uuid NOT NULL,
    reign_number bigint NOT NULL,
    accepted_upload_id uuid,
    causing_evaluation_id uuid,
    model_digest character(64) NOT NULL,
    public_bucket text NOT NULL,
    public_prefix text NOT NULL,
    hotkey text NOT NULL,
    uid integer NOT NULL,
    previous_reign_id uuid,
    crowned_at timestamp with time zone NOT NULL,
    crowned_finalized_block bigint NOT NULL,
    ended_at timestamp with time zone,
    replacement_reason text,
    operator_provenance text,
    CONSTRAINT king_reigns_check CHECK ((public_prefix = (('models/sha256/'::text || (model_digest)::text) || '/'::text))),
    CONSTRAINT king_reigns_check1 CHECK (((ended_at IS NULL) = (replacement_reason IS NULL))),
    CONSTRAINT king_reigns_check2 CHECK (((reign_number <> 0) OR (operator_provenance IS NOT NULL))),
    CONSTRAINT king_reigns_crowned_finalized_block_check CHECK ((crowned_finalized_block >= 0)),
    CONSTRAINT king_reigns_model_digest_check CHECK ((model_digest ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT king_reigns_reign_number_check CHECK ((reign_number >= 0)),
    CONSTRAINT king_reigns_uid_check CHECK ((uid >= 0))
);


ALTER TABLE control_plane.king_reigns OWNER TO teutonic_schema_owner;

--
-- Name: weight_publications; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.weight_publications (
    weight_publication_id uuid DEFAULT gen_random_uuid() NOT NULL,
    competition_id uuid NOT NULL,
    source_reign_id uuid NOT NULL,
    policy_version text NOT NULL,
    policy_hotkeys text[] NOT NULL,
    target_hotkeys text[] NOT NULL,
    target_uids integer[] NOT NULL,
    normalized_weights double precision[] NOT NULL,
    payload_sha256 character(64) NOT NULL,
    payload_revision integer DEFAULT 1 NOT NULL,
    mapping_finalized_block bigint NOT NULL,
    idempotency_key text NOT NULL,
    state text NOT NULL,
    owner_instance_id text,
    lease_expires_at timestamp with time zone,
    attempt_count integer DEFAULT 0 NOT NULL,
    next_retry_at timestamp with time zone,
    extrinsic_id text,
    included_block bigint,
    finalized_block bigint,
    last_error_code text,
    requested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    submitted_at timestamp with time zone,
    included_at timestamp with time zone,
    finalized_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    cadence_blocks integer DEFAULT 101 NOT NULL,
    cadence_enabled boolean DEFAULT true NOT NULL,
    last_attempted_block bigint,
    next_due_block bigint,
    superseded_by uuid,
    superseded_at timestamp with time zone,
    CONSTRAINT weight_due_after_attempt CHECK (((next_due_block IS NULL) OR ((last_attempted_block IS NOT NULL) AND (next_due_block > last_attempted_block)))),
    CONSTRAINT weight_publications_attempt_count_check CHECK ((attempt_count >= 0)),
    CONSTRAINT weight_publications_cadence_blocks_check CHECK ((cadence_blocks = 101)),
    CONSTRAINT weight_publications_check CHECK ((cardinality(target_uids) = cardinality(normalized_weights))),
    CONSTRAINT weight_publications_check1 CHECK (((state = ANY (ARRAY['claimed'::text, 'submitting'::text])) = (lease_expires_at IS NOT NULL))),
    CONSTRAINT weight_publications_check2 CHECK (((state <> 'finalized'::text) OR ((finalized_block IS NOT NULL) AND (finalized_at IS NOT NULL)))),
    CONSTRAINT weight_publications_finalized_block_check CHECK ((finalized_block >= 0)),
    CONSTRAINT weight_publications_included_block_check CHECK ((included_block >= 0)),
    CONSTRAINT weight_publications_last_attempted_block_check CHECK ((last_attempted_block >= 0)),
    CONSTRAINT weight_publications_next_due_block_check CHECK ((next_due_block >= 0)),
    CONSTRAINT weight_publications_payload_sha256_check CHECK ((payload_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT weight_publications_payload_revision_check CHECK ((payload_revision > 0)),
    CONSTRAINT weight_publications_mapping_finalized_block_check CHECK ((mapping_finalized_block >= 0)),
    CONSTRAINT weight_publications_state_check CHECK ((state = ANY (ARRAY['requested'::text, 'claimed'::text, 'submitting'::text, 'submitted'::text, 'included'::text, 'finalized'::text, 'retry_pending'::text, 'failed'::text, 'superseded'::text]))),
    CONSTRAINT weight_publications_target_uids_check CHECK ((cardinality(target_uids) > 0)),
    CONSTRAINT weight_publications_target_uids_check1 CHECK ((0 <= ALL (target_uids))),
    CONSTRAINT weight_publications_policy_hotkeys_check CHECK ((cardinality(policy_hotkeys) > 0)),
    CONSTRAINT weight_publications_target_hotkeys_check CHECK ((cardinality(target_hotkeys) = cardinality(target_uids))),
    CONSTRAINT weight_supersession_complete CHECK (((state <> 'superseded'::text) OR ((superseded_by IS NOT NULL) AND (superseded_at IS NOT NULL))))
);


ALTER TABLE control_plane.weight_publications OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_current_king; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_current_king WITH (security_barrier='true') AS
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
     LEFT JOIN control_plane.weight_publications current_weights ON ((current_weights.source_reign_id = r.reign_id)))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = c.netuid) AND (snapshot.chain_generation = c.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true));


ALTER VIEW control_plane.dashboard_current_king OWNER TO teutonic_schema_owner;

--
-- Name: model_promotions; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.model_promotions (
    promotion_id uuid DEFAULT gen_random_uuid() NOT NULL,
    upload_id uuid NOT NULL,
    evaluation_id uuid NOT NULL,
    model_digest character(64) NOT NULL,
    disposition text NOT NULL,
    private_bucket text NOT NULL,
    private_prefix text NOT NULL,
    public_bucket text NOT NULL,
    public_prefix text NOT NULL,
    state text NOT NULL,
    idempotency_key text NOT NULL,
    owner_instance_id text,
    lease_expires_at timestamp with time zone,
    attempt_count integer DEFAULT 0 NOT NULL,
    next_retry_at timestamp with time zone,
    expected_object_count integer NOT NULL,
    expected_size_bytes bigint NOT NULL,
    observed_object_count integer,
    observed_size_bytes bigint,
    rclone_operation_id text,
    last_error_code text,
    public_verified_at timestamp with time zone,
    private_deleted_at timestamp with time zone,
    promoted_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    expected_inventory_sha256 character(64),
    observed_inventory_sha256 character(64),
    CONSTRAINT model_promotions_attempt_count_check CHECK ((attempt_count >= 0)),
    CONSTRAINT model_promotions_check CHECK ((public_prefix = (('models/sha256/'::text || (model_digest)::text) || '/'::text))),
    CONSTRAINT model_promotions_check1 CHECK (((state = ANY (ARRAY['copying_to_public'::text, 'public_copy_verifying'::text, 'deleting_private_source'::text])) = (lease_expires_at IS NOT NULL))),
    CONSTRAINT model_promotions_check2 CHECK (((state <> 'promoted'::text) OR ((public_verified_at IS NOT NULL) AND (private_deleted_at IS NOT NULL) AND (promoted_at IS NOT NULL)))),
    CONSTRAINT model_promotions_disposition_check CHECK ((disposition = ANY (ARRAY['winner'::text, 'non_winner'::text]))),
    CONSTRAINT model_promotions_expected_inventory_sha256_check CHECK ((expected_inventory_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT model_promotions_expected_object_count_check CHECK ((expected_object_count > 0)),
    CONSTRAINT model_promotions_expected_size_bytes_check CHECK ((expected_size_bytes > 0)),
    CONSTRAINT model_promotions_model_digest_check CHECK ((model_digest ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT model_promotions_observed_inventory_sha256_check CHECK ((observed_inventory_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT model_promotions_observed_object_count_check CHECK ((observed_object_count >= 0)),
    CONSTRAINT model_promotions_observed_size_bytes_check CHECK ((observed_size_bytes >= 0)),
    CONSTRAINT model_promotions_state_check CHECK ((state = ANY (ARRAY['promotion_pending'::text, 'copying_to_public'::text, 'public_copy_verifying'::text, 'public_copy_verified'::text, 'deleting_private_source'::text, 'promoted'::text, 'retry_pending'::text, 'failed'::text])))
);


ALTER TABLE control_plane.model_promotions OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_evaluation_history; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_evaluation_history WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    SUBSTRING(encode(public.digest((e.upload_id)::text, 'sha256'::text), 'hex'::text) FROM 1 FOR 16) AS challenge_id,
    r.hotkey,
    identity.coldkey,
    r.uid,
    baseline.hotkey AS baseline_hotkey,
    baseline_identity.coldkey AS baseline_coldkey,
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
   FROM (((((((control_plane.evaluations e
     JOIN control_plane.uploads u ON ((u.upload_id = e.upload_id)))
     JOIN control_plane.registrations r ON ((r.registration_id = u.registration_id)))
     JOIN control_plane.competitions c ON ((c.competition_id = e.competition_id)))
     JOIN control_plane.king_reigns baseline ON ((baseline.reign_id = e.claimed_king_reign_id)))
     LEFT JOIN control_plane.model_promotions p ON (((p.evaluation_id = e.evaluation_id) AND (p.state = 'promoted'::text) AND ((p.disposition = 'non_winner'::text) OR (EXISTS ( SELECT 1
           FROM control_plane.king_reigns published_winner
          WHERE ((published_winner.causing_evaluation_id = e.evaluation_id) AND (published_winner.accepted_upload_id = e.upload_id) AND (published_winner.model_digest = p.model_digest))))))))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = r.netuid) AND (snapshot.chain_generation = r.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = c.netuid) AND (snapshot.chain_generation = c.chain_generation) AND (assignment.hotkey = baseline.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) baseline_identity ON (true))
  WHERE (e.state = ANY (ARRAY['completed'::text, 'terminal_failure'::text]));


ALTER VIEW control_plane.dashboard_evaluation_history OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_upload_failures; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_upload_failures WITH (security_barrier='true') AS
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
     JOIN control_plane.competitions c ON (((c.netuid = r.netuid) AND (c.chain_generation = r.chain_generation))))
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


ALTER VIEW control_plane.dashboard_upload_failures OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_king_reigns; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_king_reigns WITH (security_barrier='true') AS
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
     LEFT JOIN control_plane.weight_publications current_weights ON ((current_weights.source_reign_id = c.current_reign_id)))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = c.netuid) AND (snapshot.chain_generation = c.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true));


ALTER VIEW control_plane.dashboard_king_reigns OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_queue; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_queue WITH (security_barrier='true') AS
 SELECT c.netuid,
    c.chain_generation,
    c.name AS competition,
    SUBSTRING(encode(public.digest((u.upload_id)::text, 'sha256'::text), 'hex'::text) FROM 1 FOR 16) AS challenge_id,
    r.hotkey,
    identity.coldkey,
    r.uid,
    u.ready_finalized_block,
    row_number() OVER (PARTITION BY c.competition_id ORDER BY u.ready_finalized_block, u.ready_extrinsic_index, u.ready_event_index, u.upload_id) AS queue_position,
        CASE
            WHEN (u.state = 'ready_for_evaluation'::text) THEN 'queued'::text
            WHEN (u.state = 'retry_pending'::text) THEN 'retrying'::text
            ELSE 'processing'::text
        END AS state,
    u.ready_at AS submitted_at
   FROM (((control_plane.uploads u
     JOIN control_plane.registrations r ON ((r.registration_id = u.registration_id)))
     JOIN control_plane.competitions c ON (((c.netuid = r.netuid) AND (c.chain_generation = r.chain_generation))))
     LEFT JOIN LATERAL ( SELECT assignment.coldkey
           FROM (control_plane.metagraph_snapshots snapshot
             JOIN control_plane.metagraph_uid_assignments assignment ON ((assignment.snapshot_id = snapshot.snapshot_id)))
          WHERE ((snapshot.netuid = r.netuid) AND (snapshot.chain_generation = r.chain_generation) AND (assignment.hotkey = r.hotkey) AND snapshot.is_complete)
          ORDER BY snapshot.finalized_block DESC
         LIMIT 1) identity ON (true))
  WHERE (u.state = ANY (ARRAY['ready_for_evaluation'::text, 'retry_pending'::text]));


ALTER VIEW control_plane.dashboard_queue OWNER TO teutonic_schema_owner;

--
-- Name: service_instances; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.service_instances (
    service_name text NOT NULL,
    instance_id text NOT NULL,
    software_version text NOT NULL,
    state text NOT NULL,
    phase text NOT NULL,
    current_work_id uuid,
    started_at timestamp with time zone NOT NULL,
    heartbeat_at timestamp with time zone NOT NULL,
    restart_reason text,
    CONSTRAINT service_instances_state_check CHECK ((state = ANY (ARRAY['starting'::text, 'active'::text, 'standby'::text, 'degraded'::text, 'stopping'::text])))
);


ALTER TABLE control_plane.service_instances OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_service_health; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_service_health WITH (security_barrier='true') AS
 SELECT service_name,
        CASE
            WHEN ((clock_timestamp() - max(heartbeat_at)) > '00:01:30'::interval) THEN 'stale'::text
            WHEN bool_or((state = 'degraded'::text)) THEN 'degraded'::text
            WHEN bool_or((state = 'active'::text)) THEN 'healthy'::text
            ELSE 'offline'::text
        END AS health,
    (GREATEST((0)::numeric, EXTRACT(epoch FROM (clock_timestamp() - max(heartbeat_at)))))::bigint AS heartbeat_age_seconds,
    max(phase) FILTER (WHERE (heartbeat_at = latest_heartbeat)) AS phase
   FROM ( SELECT service_instances.service_name,
            service_instances.state,
            service_instances.phase,
            service_instances.heartbeat_at,
            max(service_instances.heartbeat_at) OVER (PARTITION BY service_instances.service_name) AS latest_heartbeat
           FROM control_plane.service_instances) instances
  GROUP BY service_name;


ALTER VIEW control_plane.dashboard_service_health OWNER TO teutonic_schema_owner;

--
-- Name: public_state_revision; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.public_state_revision (
    singleton boolean DEFAULT true NOT NULL,
    revision bigint DEFAULT 0 NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT public_state_revision_revision_check CHECK ((revision >= 0)),
    CONSTRAINT public_state_revision_singleton_check CHECK (singleton)
);


ALTER TABLE control_plane.public_state_revision OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_stats; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_stats WITH (security_barrier='true') AS
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
          WHERE ((r.netuid = c.netuid) AND (r.chain_generation = c.chain_generation) AND (u.state = ANY (ARRAY['ready_for_evaluation'::text, 'retry_pending'::text])))) AS queue_depth,
    ( SELECT count(*) AS count
           FROM control_plane.evaluations e
          WHERE ((e.competition_id = c.competition_id) AND (e.state = ANY (ARRAY['completed'::text, 'terminal_failure'::text])))) AS completed_evaluations,
    ( SELECT count(*) AS count
           FROM control_plane.king_reigns reign
          WHERE (reign.competition_id = c.competition_id)) AS reign_count
   FROM (control_plane.competitions c
     CROSS JOIN control_plane.public_state_revision revision)
  WHERE revision.singleton;


ALTER VIEW control_plane.dashboard_stats OWNER TO teutonic_schema_owner;

--
-- Name: weight_submission_attempts; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.weight_submission_attempts (
    weight_attempt_id uuid DEFAULT gen_random_uuid() NOT NULL,
    weight_publication_id uuid NOT NULL,
    sequence integer NOT NULL,
    scheduled_block bigint NOT NULL,
    idempotency_key text NOT NULL,
    payload_revision integer NOT NULL,
    target_hotkeys text[] NOT NULL,
    target_uids integer[] NOT NULL,
    normalized_weights double precision[] NOT NULL,
    payload_sha256 character(64) NOT NULL,
    state text NOT NULL,
    owner_instance_id text,
    lease_expires_at timestamp with time zone,
    try_count integer DEFAULT 0 NOT NULL,
    next_retry_at timestamp with time zone,
    publisher_mode text,
    network text,
    signer_hotkey text,
    submission_started_block bigint,
    submission_expires_block bigint,
    observed_last_update bigint,
    extrinsic_id text,
    included_block bigint,
    included_block_hash text,
    finalized_block bigint,
    finalized_block_hash text,
    last_error_code text,
    claimed_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    submitted_at timestamp with time zone,
    included_at timestamp with time zone,
    finalized_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT weight_submission_attempts_check CHECK (((state = ANY (ARRAY['claimed'::text, 'submitting'::text])) = (lease_expires_at IS NOT NULL))),
    CONSTRAINT weight_submission_attempts_check1 CHECK (((submission_expires_block IS NULL) OR ((submission_started_block IS NOT NULL) AND (submission_expires_block >= submission_started_block)))),
    CONSTRAINT weight_submission_attempts_check2 CHECK (((state <> 'finalized'::text) OR ((finalized_block IS NOT NULL) AND (finalized_at IS NOT NULL)))),
    CONSTRAINT weight_submission_attempts_finalized_block_check CHECK ((finalized_block >= 0)),
    CONSTRAINT weight_submission_attempts_included_block_check CHECK ((included_block >= 0)),
    CONSTRAINT weight_submission_attempts_observed_last_update_check CHECK ((observed_last_update >= 0)),
    CONSTRAINT weight_submission_attempts_payload_revision_check CHECK ((payload_revision > 0)),
    CONSTRAINT weight_submission_attempts_payload_sha256_check CHECK ((payload_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT weight_submission_attempts_publisher_mode_check CHECK ((publisher_mode = ANY (ARRAY['dry_run'::text, 'active'::text]))),
    CONSTRAINT weight_submission_attempts_scheduled_block_check CHECK ((scheduled_block >= 0)),
    CONSTRAINT weight_submission_attempts_sequence_check CHECK ((sequence > 0)),
    CONSTRAINT weight_submission_attempts_state_check CHECK ((state = ANY (ARRAY['claimed'::text, 'submitting'::text, 'submitted'::text, 'included'::text, 'finalized'::text, 'retry_pending'::text, 'failed'::text, 'superseded'::text]))),
    CONSTRAINT weight_submission_attempts_target_uids_check CHECK ((cardinality(target_uids) > 0)),
    CONSTRAINT weight_submission_attempts_target_uids_check1 CHECK ((0 <= ALL (target_uids))),
    CONSTRAINT weight_submission_attempts_target_hotkeys_check CHECK ((cardinality(target_hotkeys) = cardinality(target_uids))),
    CONSTRAINT weight_submission_attempts_payload_cardinality_check CHECK ((cardinality(target_uids) = cardinality(normalized_weights))),
    CONSTRAINT weight_submission_attempts_submission_expires_block_check CHECK ((submission_expires_block >= 0)),
    CONSTRAINT weight_submission_attempts_submission_started_block_check CHECK ((submission_started_block >= 0)),
    CONSTRAINT weight_submission_attempts_try_count_check CHECK ((try_count >= 0))
);


ALTER TABLE control_plane.weight_submission_attempts OWNER TO teutonic_schema_owner;

--
-- Name: dashboard_weight_status; Type: VIEW; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE VIEW control_plane.dashboard_weight_status WITH (security_barrier='true') AS
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
     JOIN control_plane.weight_publications weights ON ((weights.source_reign_id = c.current_reign_id)))
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


ALTER VIEW control_plane.dashboard_weight_status OWNER TO teutonic_schema_owner;

--
-- Name: notification_outbox; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.notification_outbox (
    notification_id uuid DEFAULT gen_random_uuid() NOT NULL,
    topic text NOT NULL,
    source_id uuid NOT NULL,
    idempotency_key text NOT NULL,
    payload jsonb NOT NULL,
    state text NOT NULL,
    owner_instance_id text,
    lease_expires_at timestamp with time zone,
    attempt_count integer DEFAULT 0 NOT NULL,
    next_retry_at timestamp with time zone,
    last_error_code text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    sent_at timestamp with time zone,
    CONSTRAINT notification_outbox_attempt_count_check CHECK ((attempt_count >= 0)),
    CONSTRAINT notification_outbox_check CHECK (((state = 'claimed'::text) = (lease_expires_at IS NOT NULL))),
    CONSTRAINT notification_outbox_check1 CHECK (((state <> 'sent'::text) OR (sent_at IS NOT NULL))),
    CONSTRAINT notification_outbox_state_check CHECK ((state = ANY (ARRAY['pending'::text, 'claimed'::text, 'sent'::text, 'retry_pending'::text, 'failed'::text])))
);


ALTER TABLE control_plane.notification_outbox OWNER TO teutonic_schema_owner;

--
-- Name: r2_parent_tokens; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.r2_parent_tokens (
    parent_token_id uuid DEFAULT gen_random_uuid() NOT NULL,
    registration_id character(64) NOT NULL,
    cloudflare_token_id text,
    access_key_id text,
    encrypted_secret bytea,
    secret_key_reference text,
    state text NOT NULL,
    attempt_count integer DEFAULT 0 NOT NULL,
    next_retry_at timestamp with time zone,
    last_error_code text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    activated_at timestamp with time zone,
    revoked_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    revocation_requested_at timestamp with time zone,
    revocation_reason text,
    token_name text,
    CONSTRAINT r2_parent_tokens_attempt_count_check CHECK ((attempt_count >= 0)),
    CONSTRAINT r2_parent_tokens_check CHECK ((num_nonnulls(encrypted_secret, secret_key_reference) <= 1)),
    CONSTRAINT r2_parent_tokens_check1 CHECK (((state <> 'active'::text) OR ((cloudflare_token_id IS NOT NULL) AND (access_key_id IS NOT NULL)))),
    CONSTRAINT r2_parent_tokens_check2 CHECK (((state <> 'revoked'::text) OR (revoked_at IS NOT NULL))),
    CONSTRAINT r2_parent_tokens_revocation_reason_check CHECK ((revocation_reason = ANY (ARRAY['model_submitted'::text, 'registration_deactivated'::text, 'operator_requested'::text]))),
    CONSTRAINT r2_parent_tokens_state_check CHECK ((state = ANY (ARRAY['pending_create'::text, 'creating'::text, 'active'::text, 'pending_revoke'::text, 'revoking'::text, 'revoked'::text, 'failed'::text])))
);


ALTER TABLE control_plane.r2_parent_tokens OWNER TO teutonic_schema_owner;

--
-- Name: upload_files; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.upload_files (
    upload_id uuid NOT NULL,
    object_path text NOT NULL,
    size_bytes bigint NOT NULL,
    sha256 character(64) NOT NULL,
    source_etag text,
    immutable_etag text,
    verified_at timestamp with time zone NOT NULL,
    CONSTRAINT upload_files_object_path_check CHECK (((object_path <> ''::text) AND (object_path <> 'manifest.json'::text) AND (object_path !~ '(^|/)\.\.(/|$)'::text) AND ("left"(object_path, 1) <> '/'::text))),
    CONSTRAINT upload_files_sha256_check CHECK ((sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT upload_files_size_bytes_check CHECK ((size_bytes >= 0))
);


ALTER TABLE control_plane.upload_files OWNER TO teutonic_schema_owner;

--
-- Name: verified_uploads; Type: TABLE; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TABLE control_plane.verified_uploads (
    upload_id uuid NOT NULL,
    immutable_bucket text NOT NULL,
    immutable_prefix text NOT NULL,
    immutable_version text,
    model_digest character(64) NOT NULL,
    manifest_sha256 character(64) NOT NULL,
    object_count integer NOT NULL,
    total_size_bytes bigint NOT NULL,
    verified_at timestamp with time zone NOT NULL,
    manifest_size_bytes bigint,
    CONSTRAINT verified_uploads_check CHECK ((immutable_prefix ~ '^models/registrations/[0-9a-f]{64}/$'::text)),
    CONSTRAINT verified_uploads_manifest_sha256_check CHECK ((manifest_sha256 ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT verified_uploads_manifest_size_bytes_check CHECK ((manifest_size_bytes > 0)),
    CONSTRAINT verified_uploads_model_digest_check CHECK ((model_digest ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT verified_uploads_object_count_check CHECK ((object_count > 0)),
    CONSTRAINT verified_uploads_total_size_bytes_check CHECK ((total_size_bytes > 0))
);


ALTER TABLE control_plane.verified_uploads OWNER TO teutonic_schema_owner;

--
-- Name: chain_cursors chain_cursors_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.chain_cursors
    ADD CONSTRAINT chain_cursors_pkey PRIMARY KEY (netuid, chain_generation);


--
-- Name: competitions competitions_netuid_chain_generation_name_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.competitions
    ADD CONSTRAINT competitions_netuid_chain_generation_name_key UNIQUE (netuid, chain_generation, name);


--
-- Name: competitions competitions_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.competitions
    ADD CONSTRAINT competitions_pkey PRIMARY KEY (competition_id);


--
-- Name: evaluation_early_stopping_policies evaluation_early_stopping_policies_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluation_early_stopping_policies
    ADD CONSTRAINT evaluation_early_stopping_policies_pkey PRIMARY KEY (competition_id);


--
-- Name: evaluation_configs evaluation_configs_competition_id_config_version_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluation_configs
    ADD CONSTRAINT evaluation_configs_competition_id_config_version_key UNIQUE (competition_id, config_version);


--
-- Name: evaluation_configs evaluation_configs_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluation_configs
    ADD CONSTRAINT evaluation_configs_pkey PRIMARY KEY (evaluation_config_id);


--
-- Name: dataset_manifests dataset_manifests_evaluation_config_id_name_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.dataset_manifests
    ADD CONSTRAINT dataset_manifests_evaluation_config_id_name_key UNIQUE (evaluation_config_id, name);


--
-- Name: dataset_manifests dataset_manifests_evaluation_config_id_position_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.dataset_manifests
    ADD CONSTRAINT dataset_manifests_evaluation_config_id_position_key UNIQUE (evaluation_config_id, "position");


--
-- Name: dataset_manifests dataset_manifests_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.dataset_manifests
    ADD CONSTRAINT dataset_manifests_pkey PRIMARY KEY (dataset_manifest_id);


--
-- Name: controller_jobs controller_jobs_idempotency_key_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.controller_jobs
    ADD CONSTRAINT controller_jobs_idempotency_key_key UNIQUE (idempotency_key);


--
-- Name: controller_jobs controller_jobs_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.controller_jobs
    ADD CONSTRAINT controller_jobs_pkey PRIMARY KEY (controller_job_id);


--
-- Name: credential_generations credential_generations_mailbox_object_key_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.credential_generations
    ADD CONSTRAINT credential_generations_mailbox_object_key_key UNIQUE (mailbox_object_key);


--
-- Name: credential_generations credential_generations_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.credential_generations
    ADD CONSTRAINT credential_generations_pkey PRIMARY KEY (registration_id, generation);


--
-- Name: evaluations evaluations_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluations
    ADD CONSTRAINT evaluations_pkey PRIMARY KEY (evaluation_id);


--
-- Name: evaluations evaluations_upload_id_attempt_number_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluations
    ADD CONSTRAINT evaluations_upload_id_attempt_number_key UNIQUE (upload_id, attempt_number);


--
-- Name: king_reigns king_reigns_competition_id_model_digest_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.king_reigns
    ADD CONSTRAINT king_reigns_competition_id_model_digest_key UNIQUE (competition_id, model_digest);


--
-- Name: king_reigns king_reigns_competition_id_reign_number_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.king_reigns
    ADD CONSTRAINT king_reigns_competition_id_reign_number_key UNIQUE (competition_id, reign_number);


--
-- Name: king_reigns king_reigns_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.king_reigns
    ADD CONSTRAINT king_reigns_pkey PRIMARY KEY (reign_id);


--
-- Name: metagraph_snapshots metagraph_snapshots_netuid_chain_generation_finalized_block_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.metagraph_snapshots
    ADD CONSTRAINT metagraph_snapshots_netuid_chain_generation_finalized_block_key UNIQUE (netuid, chain_generation, finalized_block);


--
-- Name: metagraph_snapshots metagraph_snapshots_netuid_chain_generation_snapshot_checks_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.metagraph_snapshots
    ADD CONSTRAINT metagraph_snapshots_netuid_chain_generation_snapshot_checks_key UNIQUE (netuid, chain_generation, snapshot_checksum);


--
-- Name: metagraph_snapshots metagraph_snapshots_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.metagraph_snapshots
    ADD CONSTRAINT metagraph_snapshots_pkey PRIMARY KEY (snapshot_id);


--
-- Name: metagraph_uid_assignments metagraph_uid_assignments_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.metagraph_uid_assignments
    ADD CONSTRAINT metagraph_uid_assignments_pkey PRIMARY KEY (snapshot_id, uid);


--
-- Name: model_promotions model_promotions_idempotency_key_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.model_promotions
    ADD CONSTRAINT model_promotions_idempotency_key_key UNIQUE (idempotency_key);


--
-- Name: model_promotions model_promotions_model_digest_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.model_promotions
    ADD CONSTRAINT model_promotions_model_digest_key UNIQUE (model_digest);


--
-- Name: model_promotions model_promotions_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.model_promotions
    ADD CONSTRAINT model_promotions_pkey PRIMARY KEY (promotion_id);


--
-- Name: model_promotions model_promotions_public_bucket_public_prefix_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.model_promotions
    ADD CONSTRAINT model_promotions_public_bucket_public_prefix_key UNIQUE (public_bucket, public_prefix);


--
-- Name: notification_outbox notification_outbox_idempotency_key_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.notification_outbox
    ADD CONSTRAINT notification_outbox_idempotency_key_key UNIQUE (idempotency_key);


--
-- Name: notification_outbox notification_outbox_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.notification_outbox
    ADD CONSTRAINT notification_outbox_pkey PRIMARY KEY (notification_id);


--
-- Name: public_state_revision public_state_revision_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.public_state_revision
    ADD CONSTRAINT public_state_revision_pkey PRIMARY KEY (singleton);


--
-- Name: r2_parent_tokens r2_parent_tokens_access_key_id_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.r2_parent_tokens
    ADD CONSTRAINT r2_parent_tokens_access_key_id_key UNIQUE (access_key_id);


--
-- Name: r2_parent_tokens r2_parent_tokens_cloudflare_token_id_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.r2_parent_tokens
    ADD CONSTRAINT r2_parent_tokens_cloudflare_token_id_key UNIQUE (cloudflare_token_id);


--
-- Name: r2_parent_tokens r2_parent_tokens_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.r2_parent_tokens
    ADD CONSTRAINT r2_parent_tokens_pkey PRIMARY KEY (parent_token_id);


--
-- Name: r2_parent_tokens r2_parent_tokens_registration_id_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.r2_parent_tokens
    ADD CONSTRAINT r2_parent_tokens_registration_id_key UNIQUE (registration_id);


--
-- Name: r2_parent_tokens r2_parent_tokens_token_name_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.r2_parent_tokens
    ADD CONSTRAINT r2_parent_tokens_token_name_key UNIQUE (token_name);


--
-- Name: registrations registrations_model_prefix_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.registrations
    ADD CONSTRAINT registrations_model_prefix_key UNIQUE (model_prefix);


--
-- Name: registrations registrations_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.registrations
    ADD CONSTRAINT registrations_pkey PRIMARY KEY (registration_id);


--
-- Name: service_instances service_instances_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.service_instances
    ADD CONSTRAINT service_instances_pkey PRIMARY KEY (service_name, instance_id);


--
-- Name: upload_files upload_files_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.upload_files
    ADD CONSTRAINT upload_files_pkey PRIMARY KEY (upload_id, object_path);


--
-- Name: uploads uploads_chain_generation_ready_finalized_block_ready_extrin_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.uploads
    ADD CONSTRAINT uploads_chain_generation_ready_finalized_block_ready_extrin_key UNIQUE (chain_generation, ready_finalized_block, ready_extrinsic_index, ready_event_index);


--
-- Name: uploads uploads_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.uploads
    ADD CONSTRAINT uploads_pkey PRIMARY KEY (upload_id);


--
-- Name: uploads uploads_registration_id_manifest_sha256_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.uploads
    ADD CONSTRAINT uploads_registration_id_manifest_sha256_key UNIQUE (registration_id, manifest_sha256);


--
-- Name: verified_uploads verified_uploads_immutable_prefix_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.verified_uploads
    ADD CONSTRAINT verified_uploads_immutable_prefix_key UNIQUE (immutable_prefix);


--
-- Name: verified_uploads verified_uploads_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.verified_uploads
    ADD CONSTRAINT verified_uploads_pkey PRIMARY KEY (upload_id);


--
-- Name: weight_publications weight_publications_idempotency_key_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_publications
    ADD CONSTRAINT weight_publications_idempotency_key_key UNIQUE (idempotency_key);


--
-- Name: weight_publications weight_publications_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_publications
    ADD CONSTRAINT weight_publications_pkey PRIMARY KEY (weight_publication_id);


--
-- Name: weight_publications weight_publications_source_reign_id_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_publications
    ADD CONSTRAINT weight_publications_source_reign_id_key UNIQUE (source_reign_id);


--
-- Name: weight_submission_attempts weight_submission_attempts_idempotency_key_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_submission_attempts
    ADD CONSTRAINT weight_submission_attempts_idempotency_key_key UNIQUE (idempotency_key);


--
-- Name: weight_submission_attempts weight_submission_attempts_pkey; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_submission_attempts
    ADD CONSTRAINT weight_submission_attempts_pkey PRIMARY KEY (weight_attempt_id);


--
-- Name: weight_submission_attempts weight_submission_attempts_weight_publication_id_sequence_key; Type: CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_submission_attempts
    ADD CONSTRAINT weight_submission_attempts_weight_publication_id_sequence_key UNIQUE (weight_publication_id, sequence);


--
-- Name: controller_jobs_claimable; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE INDEX controller_jobs_claimable ON control_plane.controller_jobs USING btree (next_retry_at, created_at, controller_job_id) WHERE (state = ANY (ARRAY['pending'::text, 'retry_pending'::text]));


--
-- Name: evaluations_one_successful_policy_king; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE UNIQUE INDEX evaluations_one_successful_policy_king ON control_plane.evaluations USING btree (upload_id, policy_version, claimed_king_reign_id) WHERE (state = 'completed'::text);


--
-- Name: evaluations_recovery; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE INDEX evaluations_recovery ON control_plane.evaluations USING btree (lease_expires_at, created_at) WHERE (state = ANY (ARRAY['claimed'::text, 'evaluating'::text]));


--
-- Name: evaluations_retry_due; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE INDEX evaluations_retry_due ON control_plane.evaluations USING btree (next_retry_at, created_at) WHERE (state = 'retryable_failure'::text);


--
-- Name: evaluation_configs_one_active; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE UNIQUE INDEX evaluation_configs_one_active ON control_plane.evaluation_configs USING btree (competition_id) WHERE active;


--
-- Name: king_reigns_one_current; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE UNIQUE INDEX king_reigns_one_current ON control_plane.king_reigns USING btree (competition_id) WHERE (ended_at IS NULL);


--
-- Name: model_promotions_claimable; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE INDEX model_promotions_claimable ON control_plane.model_promotions USING btree (next_retry_at, created_at, promotion_id) WHERE (state = ANY (ARRAY['promotion_pending'::text, 'retry_pending'::text, 'public_copy_verified'::text, 'copying_to_public'::text, 'public_copy_verifying'::text, 'deleting_private_source'::text]));


--
-- Name: notification_outbox_due; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE INDEX notification_outbox_due ON control_plane.notification_outbox USING btree (next_retry_at, created_at) WHERE (state = ANY (ARRAY['pending'::text, 'retry_pending'::text]));


--
-- Name: registrations_one_active_hotkey; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE UNIQUE INDEX registrations_one_active_hotkey ON control_plane.registrations USING btree (netuid, chain_generation, hotkey) WHERE (state <> 'inactive'::text);


--
-- Name: registrations_one_active_uid; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE UNIQUE INDEX registrations_one_active_uid ON control_plane.registrations USING btree (netuid, chain_generation, uid) WHERE (state <> 'inactive'::text);


--
-- Name: uploads_evaluation_order; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE INDEX uploads_evaluation_order ON control_plane.uploads USING btree (ready_finalized_block, ready_extrinsic_index, ready_event_index, upload_id) WHERE (state = 'ready_for_evaluation'::text);


--
-- Name: uploads_one_submission_per_hotkey; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE UNIQUE INDEX uploads_one_submission_per_hotkey ON control_plane.uploads USING btree (signalling_hotkey) WHERE (ready_at IS NOT NULL);


--
-- Name: weight_submission_attempts_claimable; Type: INDEX; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE INDEX weight_submission_attempts_claimable ON control_plane.weight_submission_attempts USING btree (next_retry_at, scheduled_block, weight_attempt_id) WHERE (state = ANY (ARRAY['claimed'::text, 'submitting'::text, 'submitted'::text, 'included'::text, 'retry_pending'::text]));


--
-- Name: chain_cursors chain_cursors_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER chain_cursors_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.chain_cursors FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: competitions competitions_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER competitions_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.competitions FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: credential_generations credential_generations_prevent_reissue; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER credential_generations_prevent_reissue BEFORE INSERT OR UPDATE OF state ON control_plane.credential_generations FOR EACH ROW EXECUTE FUNCTION control_plane.prevent_upload_access_reissue();


--
-- Name: evaluations evaluations_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER evaluations_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.evaluations FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: king_reigns king_reigns_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER king_reigns_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.king_reigns FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: metagraph_snapshots metagraph_snapshots_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER metagraph_snapshots_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.metagraph_snapshots FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: metagraph_uid_assignments metagraph_uid_assignments_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER metagraph_uid_assignments_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.metagraph_uid_assignments FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: model_promotions model_promotions_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER model_promotions_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.model_promotions FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: notification_outbox notification_outbox_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER notification_outbox_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.notification_outbox FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: r2_parent_tokens r2_parent_tokens_prevent_reactivation; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER r2_parent_tokens_prevent_reactivation BEFORE INSERT OR UPDATE OF state ON control_plane.r2_parent_tokens FOR EACH ROW EXECUTE FUNCTION control_plane.prevent_parent_token_reactivation();


--
-- Name: registrations registrations_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER registrations_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.registrations FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: service_instances service_instances_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER service_instances_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.service_instances FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: uploads uploads_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER uploads_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.uploads FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: uploads uploads_revoke_access_after_ready; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER uploads_revoke_access_after_ready AFTER INSERT OR UPDATE OF ready_at ON control_plane.uploads FOR EACH ROW EXECUTE FUNCTION control_plane.revoke_upload_access_after_ready();


--
-- Name: weight_publications weight_publications_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER weight_publications_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.weight_publications FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: weight_submission_attempts weight_submission_attempts_revision; Type: TRIGGER; Schema: control_plane; Owner: teutonic_schema_owner
--

CREATE TRIGGER weight_submission_attempts_revision AFTER INSERT OR DELETE OR UPDATE ON control_plane.weight_submission_attempts FOR EACH STATEMENT EXECUTE FUNCTION control_plane.bump_public_state_revision();


--
-- Name: competitions competitions_current_reign_fk; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.competitions
    ADD CONSTRAINT competitions_current_reign_fk FOREIGN KEY (current_reign_id) REFERENCES control_plane.king_reigns(reign_id) DEFERRABLE INITIALLY DEFERRED;


--
-- Name: evaluation_configs evaluation_configs_competition_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluation_configs
    ADD CONSTRAINT evaluation_configs_competition_id_fkey FOREIGN KEY (competition_id) REFERENCES control_plane.competitions(competition_id) ON DELETE RESTRICT;


--
-- Name: evaluation_early_stopping_policies evaluation_early_stopping_policies_competition_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluation_early_stopping_policies
    ADD CONSTRAINT evaluation_early_stopping_policies_competition_id_fkey FOREIGN KEY (competition_id) REFERENCES control_plane.competitions(competition_id) ON DELETE RESTRICT;


--
-- Name: dataset_manifests dataset_manifests_evaluation_config_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.dataset_manifests
    ADD CONSTRAINT dataset_manifests_evaluation_config_id_fkey FOREIGN KEY (evaluation_config_id) REFERENCES control_plane.evaluation_configs(evaluation_config_id) ON DELETE CASCADE;


--
-- Name: controller_jobs controller_jobs_registration_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.controller_jobs
    ADD CONSTRAINT controller_jobs_registration_id_fkey FOREIGN KEY (registration_id) REFERENCES control_plane.registrations(registration_id) ON DELETE RESTRICT;


--
-- Name: controller_jobs controller_jobs_upload_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.controller_jobs
    ADD CONSTRAINT controller_jobs_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES control_plane.uploads(upload_id) ON DELETE RESTRICT;


--
-- Name: credential_generations credential_generations_registration_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.credential_generations
    ADD CONSTRAINT credential_generations_registration_id_fkey FOREIGN KEY (registration_id) REFERENCES control_plane.registrations(registration_id) ON DELETE RESTRICT;


--
-- Name: evaluations evaluations_claimed_king_reign_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluations
    ADD CONSTRAINT evaluations_claimed_king_reign_id_fkey FOREIGN KEY (claimed_king_reign_id) REFERENCES control_plane.king_reigns(reign_id) ON DELETE RESTRICT;


--
-- Name: evaluations evaluations_competition_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluations
    ADD CONSTRAINT evaluations_competition_id_fkey FOREIGN KEY (competition_id) REFERENCES control_plane.competitions(competition_id) ON DELETE RESTRICT;


--
-- Name: evaluations evaluations_upload_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.evaluations
    ADD CONSTRAINT evaluations_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES control_plane.verified_uploads(upload_id) ON DELETE RESTRICT;


--
-- Name: king_reigns king_reigns_accepted_upload_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.king_reigns
    ADD CONSTRAINT king_reigns_accepted_upload_id_fkey FOREIGN KEY (accepted_upload_id) REFERENCES control_plane.verified_uploads(upload_id) ON DELETE RESTRICT;


--
-- Name: king_reigns king_reigns_causing_evaluation_fk; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.king_reigns
    ADD CONSTRAINT king_reigns_causing_evaluation_fk FOREIGN KEY (causing_evaluation_id) REFERENCES control_plane.evaluations(evaluation_id) DEFERRABLE INITIALLY DEFERRED;


--
-- Name: king_reigns king_reigns_competition_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.king_reigns
    ADD CONSTRAINT king_reigns_competition_id_fkey FOREIGN KEY (competition_id) REFERENCES control_plane.competitions(competition_id) ON DELETE RESTRICT;


--
-- Name: king_reigns king_reigns_previous_reign_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.king_reigns
    ADD CONSTRAINT king_reigns_previous_reign_id_fkey FOREIGN KEY (previous_reign_id) REFERENCES control_plane.king_reigns(reign_id) ON DELETE RESTRICT;


--
-- Name: metagraph_uid_assignments metagraph_uid_assignments_snapshot_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.metagraph_uid_assignments
    ADD CONSTRAINT metagraph_uid_assignments_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES control_plane.metagraph_snapshots(snapshot_id) ON DELETE RESTRICT;


--
-- Name: model_promotions model_promotions_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.model_promotions
    ADD CONSTRAINT model_promotions_evaluation_id_fkey FOREIGN KEY (evaluation_id) REFERENCES control_plane.evaluations(evaluation_id) ON DELETE RESTRICT;


--
-- Name: model_promotions model_promotions_upload_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.model_promotions
    ADD CONSTRAINT model_promotions_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES control_plane.verified_uploads(upload_id) ON DELETE RESTRICT;


--
-- Name: r2_parent_tokens r2_parent_tokens_registration_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.r2_parent_tokens
    ADD CONSTRAINT r2_parent_tokens_registration_id_fkey FOREIGN KEY (registration_id) REFERENCES control_plane.registrations(registration_id) ON DELETE RESTRICT;


--
-- Name: upload_files upload_files_upload_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.upload_files
    ADD CONSTRAINT upload_files_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES control_plane.uploads(upload_id) ON DELETE RESTRICT;


--
-- Name: uploads uploads_registration_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.uploads
    ADD CONSTRAINT uploads_registration_id_fkey FOREIGN KEY (registration_id) REFERENCES control_plane.registrations(registration_id) ON DELETE RESTRICT;


--
-- Name: verified_uploads verified_uploads_upload_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.verified_uploads
    ADD CONSTRAINT verified_uploads_upload_id_fkey FOREIGN KEY (upload_id) REFERENCES control_plane.uploads(upload_id) ON DELETE RESTRICT;


--
-- Name: weight_publications weight_publications_competition_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_publications
    ADD CONSTRAINT weight_publications_competition_id_fkey FOREIGN KEY (competition_id) REFERENCES control_plane.competitions(competition_id) ON DELETE RESTRICT;


--
-- Name: weight_publications weight_publications_source_reign_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_publications
    ADD CONSTRAINT weight_publications_source_reign_id_fkey FOREIGN KEY (source_reign_id) REFERENCES control_plane.king_reigns(reign_id) ON DELETE RESTRICT;


--
-- Name: weight_publications weight_publications_superseded_by_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_publications
    ADD CONSTRAINT weight_publications_superseded_by_fkey FOREIGN KEY (superseded_by) REFERENCES control_plane.weight_publications(weight_publication_id);


--
-- Name: weight_submission_attempts weight_submission_attempts_weight_publication_id_fkey; Type: FK CONSTRAINT; Schema: control_plane; Owner: teutonic_schema_owner
--

ALTER TABLE ONLY control_plane.weight_submission_attempts
    ADD CONSTRAINT weight_submission_attempts_weight_publication_id_fkey FOREIGN KEY (weight_publication_id) REFERENCES control_plane.weight_publications(weight_publication_id) ON DELETE RESTRICT;


--
-- Name: SCHEMA control_plane; Type: ACL; Schema: -; Owner: teutonic_schema_owner
--

GRANT USAGE ON SCHEMA control_plane TO teutonic_access_controller;
GRANT USAGE ON SCHEMA control_plane TO teutonic_validator;
GRANT USAGE ON SCHEMA control_plane TO teutonic_weight_publisher;
GRANT USAGE ON SCHEMA control_plane TO teutonic_dashboard_view;
GRANT USAGE ON SCHEMA control_plane TO teutonic_auditor;


--
-- Name: FUNCTION bump_public_state_revision(); Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

REVOKE ALL ON FUNCTION control_plane.bump_public_state_revision() FROM PUBLIC;


--
-- Name: TABLE chain_cursors; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.chain_cursors TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.chain_cursors TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.chain_cursors TO teutonic_auditor;


--
-- Name: TABLE competitions; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.competitions TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.competitions TO teutonic_auditor;
GRANT SELECT ON TABLE control_plane.competitions TO teutonic_weight_publisher;


--
-- Name: TABLE evaluation_early_stopping_policies; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.evaluation_early_stopping_policies TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.evaluation_early_stopping_policies TO teutonic_auditor;


--
-- Name: TABLE evaluation_configs; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.evaluation_configs TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.evaluation_configs TO teutonic_auditor;


--
-- Name: TABLE dataset_manifests; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dataset_manifests TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.dataset_manifests TO teutonic_auditor;


--
-- Name: TABLE controller_jobs; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.controller_jobs TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.controller_jobs TO teutonic_auditor;


--
-- Name: TABLE credential_generations; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.credential_generations TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.credential_generations TO teutonic_auditor;


--
-- Name: TABLE dashboard_chain; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_chain TO teutonic_dashboard_view;


--
-- Name: TABLE dashboard_contract; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_contract TO teutonic_dashboard_view;


--
-- Name: TABLE dashboard_dataset_manifests; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_dataset_manifests TO teutonic_dashboard_view;


--
-- Name: TABLE dashboard_dataset_versions; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_dataset_versions TO teutonic_dashboard_view;


--
-- Name: TABLE evaluations; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.evaluations TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.evaluations TO teutonic_auditor;


--
-- Name: TABLE metagraph_snapshots; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.metagraph_snapshots TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.metagraph_snapshots TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.metagraph_snapshots TO teutonic_auditor;


--
-- Name: TABLE metagraph_uid_assignments; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.metagraph_uid_assignments TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.metagraph_uid_assignments TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.metagraph_uid_assignments TO teutonic_auditor;


--
-- Name: TABLE registrations; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.registrations TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.registrations TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.registrations TO teutonic_auditor;


--
-- Name: TABLE uploads; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.uploads TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.uploads TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.uploads TO teutonic_auditor;


--
-- Name: COLUMN uploads.state; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT UPDATE(state) ON TABLE control_plane.uploads TO teutonic_validator;


--
-- Name: COLUMN uploads.failure_code; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT UPDATE(failure_code) ON TABLE control_plane.uploads TO teutonic_validator;


--
-- Name: COLUMN uploads.updated_at; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT UPDATE(updated_at) ON TABLE control_plane.uploads TO teutonic_validator;


--
-- Name: TABLE dashboard_current_evaluation; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_current_evaluation TO teutonic_dashboard_view;


--
-- Name: TABLE king_reigns; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.king_reigns TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.king_reigns TO teutonic_auditor;
GRANT SELECT ON TABLE control_plane.king_reigns TO teutonic_weight_publisher;


--
-- Name: TABLE weight_publications; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.weight_publications TO teutonic_validator;
GRANT SELECT,UPDATE ON TABLE control_plane.weight_publications TO teutonic_weight_publisher;
GRANT SELECT ON TABLE control_plane.weight_publications TO teutonic_auditor;


--
-- Name: TABLE dashboard_current_king; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_current_king TO teutonic_dashboard_view;


--
-- Name: TABLE model_promotions; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.model_promotions TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.model_promotions TO teutonic_auditor;


--
-- Name: TABLE dashboard_evaluation_history; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_evaluation_history TO teutonic_dashboard_view;


--
-- Name: TABLE dashboard_upload_failures; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_upload_failures TO teutonic_dashboard_view;


--
-- Name: TABLE dashboard_king_reigns; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_king_reigns TO teutonic_dashboard_view;


--
-- Name: TABLE dashboard_queue; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_queue TO teutonic_dashboard_view;


--
-- Name: TABLE service_instances; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE control_plane.service_instances TO teutonic_access_controller;
GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE control_plane.service_instances TO teutonic_validator;
GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE control_plane.service_instances TO teutonic_weight_publisher;
GRANT SELECT ON TABLE control_plane.service_instances TO teutonic_auditor;


--
-- Name: TABLE dashboard_service_health; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_service_health TO teutonic_dashboard_view;


--
-- Name: TABLE public_state_revision; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.public_state_revision TO teutonic_auditor;


--
-- Name: TABLE dashboard_stats; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_stats TO teutonic_dashboard_view;


--
-- Name: TABLE weight_submission_attempts; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.weight_submission_attempts TO teutonic_weight_publisher;
GRANT SELECT ON TABLE control_plane.weight_submission_attempts TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.weight_submission_attempts TO teutonic_auditor;


--
-- Name: TABLE dashboard_weight_status; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT ON TABLE control_plane.dashboard_weight_status TO teutonic_dashboard_view;


--
-- Name: TABLE notification_outbox; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.notification_outbox TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.notification_outbox TO teutonic_auditor;


--
-- Name: TABLE r2_parent_tokens; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.r2_parent_tokens TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.r2_parent_tokens TO teutonic_auditor;


--
-- Name: TABLE upload_files; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.upload_files TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.upload_files TO teutonic_auditor;
GRANT SELECT ON TABLE control_plane.upload_files TO teutonic_validator;


--
-- Name: TABLE verified_uploads; Type: ACL; Schema: control_plane; Owner: teutonic_schema_owner
--

GRANT SELECT,INSERT,UPDATE ON TABLE control_plane.verified_uploads TO teutonic_access_controller;
GRANT SELECT ON TABLE control_plane.verified_uploads TO teutonic_validator;
GRANT SELECT ON TABLE control_plane.verified_uploads TO teutonic_auditor;


--
-- PostgreSQL database dump complete
--


REVOKE ALL ON SCHEMA control_plane FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA control_plane FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA control_plane FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA control_plane FROM PUBLIC;

ALTER DEFAULT PRIVILEGES FOR ROLE teutonic_schema_owner IN SCHEMA control_plane
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE teutonic_schema_owner IN SCHEMA control_plane
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE teutonic_schema_owner IN SCHEMA control_plane
    REVOKE ALL ON FUNCTIONS FROM PUBLIC;


INSERT INTO control_plane.public_state_revision (singleton)
VALUES (true);

COMMIT;
