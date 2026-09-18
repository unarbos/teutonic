from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone

try:
    import psycopg
    from psycopg import errors
except ImportError:
    psycopg = None
    errors = None

from teutonic.dashboard.contracts import canonical_dashboard_json
from teutonic.dashboard.projection import DashboardProjectionRepository

DATABASE_URL = os.environ.get("TEUTONIC_TEST_DATABASE_URL")
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


@unittest.skipUnless(DATABASE_URL and psycopg, "PostgreSQL integration database required")
class DashboardViewIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.owner = psycopg.connect(DATABASE_URL, autocommit=True)
        if cls.owner.execute("SELECT current_database()").fetchone()[0] != "teutonic_test":
            raise RuntimeError("refusing Phase 8 tests outside teutonic_test")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.owner.close()

    def setUp(self) -> None:
        self.owner.execute(
            """
            TRUNCATE TABLE
                control_plane.weight_submission_attempts,
                control_plane.notification_outbox,
                control_plane.model_promotions,
                control_plane.evaluations,
                control_plane.weight_publications,
                control_plane.king_reigns,
                control_plane.competitions,
                control_plane.controller_jobs,
                control_plane.verified_uploads,
                control_plane.upload_files,
                control_plane.uploads,
                control_plane.credential_generations,
                control_plane.r2_parent_tokens,
                control_plane.registrations,
                control_plane.metagraph_uid_assignments,
                control_plane.metagraph_snapshots,
                control_plane.chain_cursors,
                control_plane.service_instances
            RESTART IDENTITY CASCADE
            """
        )
        self._seed()
        self.dashboard = psycopg.connect(DATABASE_URL, autocommit=True)
        self.dashboard.execute("SET ROLE teutonic_dashboard_view")
        self.repository = DashboardProjectionRepository(
            self.dashboard,
            netuid=306,
            chain_generation="test",
            competition="quasar",
            chain_name="Teutonic Testnet",
            seed_repo="owner/genesis",
            seed_digest="hf:" + "b" * 40,
            seed_repo_backend="hf",
        )
        self.assertTrue(self.repository.acquire_lock())

    def tearDown(self) -> None:
        self.repository.release_lock()
        self.dashboard.close()

    def _seed(self) -> None:
        hotkey = "5" + "H" * 47
        coldkey = "5" + "C" * 47
        king_hotkey = "5" + "K" * 47
        self.king_hotkey = king_hotkey
        snapshot = self.owner.execute(
            """
            INSERT INTO control_plane.metagraph_snapshots (
                netuid, chain_generation, finalized_block, finalized_block_hash,
                snapshot_checksum, uid_count, is_complete, observed_at
            ) VALUES (306, 'test', 1000, '0x1000', %s, 2, true, %s)
            RETURNING snapshot_id
            """,
            ("1" * 64, NOW),
        ).fetchone()[0]
        with self.owner.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO control_plane.metagraph_uid_assignments
                    (snapshot_id, uid, hotkey, coldkey, registration_block)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (snapshot, 0, king_hotkey, "5" + "G" * 47, 900),
                    (snapshot, 7, hotkey, coldkey, 950),
                ],
            )
        self.owner.execute(
            """
            INSERT INTO control_plane.chain_cursors (
                netuid, chain_generation, finalized_start_block, last_finalized_block,
                last_finalized_block_hash, snapshot_checksum, observed_at
            ) VALUES (306, 'test', 900, 1000, '0x1000', %s, %s)
            """,
            ("1" * 64, NOW),
        )
        registration = "2" * 64
        self.owner.execute(
            """
            INSERT INTO control_plane.registrations (
                registration_id, netuid, chain_generation, uid, hotkey,
                first_seen_finalized_block, last_seen_finalized_block, model_prefix, state
            ) VALUES (%s, 306, 'test', 7, %s, 900, 1000, %s, 'active')
            """,
            (registration, hotkey, f"models/registrations/{registration}/"),
        )
        self.owner.execute(
            """
            INSERT INTO control_plane.r2_parent_tokens (
                registration_id, cloudflare_token_id, access_key_id, state, activated_at
            ) VALUES (%s, 'phase8-parent-token', 'phase8-parent-access', 'active', %s)
            """,
            (registration, NOW),
        )
        model_digest = "3" * 64
        upload = self.owner.execute(
            """
            INSERT INTO control_plane.uploads (
                registration_id, chain_generation, signalling_hotkey, ready_payload,
                ready_finalized_block, ready_extrinsic_index, ready_event_index,
                manifest_sha256, manifest_signature_verified, model_digest, model_name,
                object_count, total_size_bytes, state, ready_at
            ) VALUES (%s, 'test', %s, 'r2ready:v1', 901, 0, 0, %s, true, %s,
                      'owner/challenger', 1, 100, 'evaluated', %s)
            RETURNING upload_id
            """,
            (registration, hotkey, "4" * 64, model_digest, NOW),
        ).fetchone()[0]
        self.owner.execute(
            """
            INSERT INTO control_plane.verified_uploads (
                upload_id, immutable_bucket, immutable_prefix, model_digest,
                manifest_sha256, object_count, total_size_bytes, verified_at
            ) VALUES (%s, 'DO-NOT-LEAK-private-models', %s, %s, %s, 1, 100, %s)
            """,
            (
                upload,
                f"models/registrations/{registration}/",
                model_digest,
                "4" * 64,
                NOW,
            ),
        )
        competition = self.owner.execute(
            """
            INSERT INTO control_plane.competitions (netuid, chain_generation, name)
            VALUES (306, 'test', 'quasar') RETURNING competition_id
            """
        ).fetchone()[0]
        evaluation_config = self.owner.execute(
            """
            INSERT INTO control_plane.evaluation_configs (
                competition_id, config_version, dataset_label, eval_n,
                delta_threshold, active
            ) VALUES (%s, %s, 'fixture-datasets', 2000, 0.5, true)
            RETURNING evaluation_config_id
            """,
            (competition, "7" * 64),
        ).fetchone()[0]
        self.owner.execute(
            """
            INSERT INTO control_plane.dataset_manifests (
                evaluation_config_id, "position", name, manifest_url,
                manifest_sha256, manifest_json, sample_proportion
            ) VALUES (%s, 0, 'fixture', 'https://datasets.example/fixture/manifest.json',
                      %s, %s::jsonb, 1.0)
            """,
            (
                evaluation_config,
                "8" * 64,
                '{"shards":[{"key":"shards/part-000.npy","sha256":"' + "9" * 64
                + '","size_bytes":4096,"n_tokens":4096}]}',
            ),
        )
        reign = self.owner.execute(
            """
            INSERT INTO control_plane.king_reigns (
                competition_id, reign_number, model_digest, public_bucket, public_prefix,
                hotkey, uid, crowned_at, crowned_finalized_block, operator_provenance
            ) VALUES (%s, 0, %s, 'teutonic-models', %s, %s, 0, %s, 900, 'seed')
            RETURNING reign_id
            """,
            (competition, "5" * 64, f"models/sha256/{'5' * 64}/", king_hotkey, NOW),
        ).fetchone()[0]
        self.owner.execute(
            "UPDATE control_plane.competitions SET current_reign_id = %s WHERE competition_id = %s",
            (reign, competition),
        )
        evaluation = self.owner.execute(
            """
            INSERT INTO control_plane.evaluations (
                upload_id, competition_id, attempt_number, claimed_king_reign_id,
                state, policy_version, code_version, dataset_version,
                sampling_seed, bootstrap_seed, thresholds, verdict, verdict_summary,
                private_diagnostic_reference, result_artifact_reference, completed_at
            ) VALUES (%s, %s, 1, %s, 'completed', 'policy-v1', 'code-v1', 'dataset-v1',
                      1, 2, '{}'::jsonb, 'rejected',
                      %s::jsonb, 'traceback:http://validator-internal:9000 secret_access_key',
                      's3://DO-NOT-LEAK/results.json', %s)
            RETURNING evaluation_id
            """,
            (
                upload,
                competition,
                reign,
                '{"mu_hat":0.01,"lcb":0.005,"delta_threshold":0.02,"avg_king_loss":2.1,'
                '"avg_challenger_loss":2.09,"wall_time_s":42,"n_sequences":64,'
                '"source_scores":{"fixture":{"n_sequences":64,"avg_king_loss":2.1,'
                '"avg_challenger_loss":2.09,"mu_hat":0.01}},'
                '"shards_used":[{"source":"fixture","refs":['
                '"https://datasets.example/private/path/part-000.npy?secret=never"]}]}',
                NOW,
            ),
        ).fetchone()[0]
        self.owner.execute(
            """
            INSERT INTO control_plane.weight_publications (
                competition_id, source_reign_id, policy_version, policy_hotkeys,
                target_hotkeys, target_uids, normalized_weights, payload_sha256,
                mapping_finalized_block, idempotency_key, state
            ) VALUES (%s, %s, 'weights-v1', %s, %s, ARRAY[0],
                      ARRAY[1.0]::double precision[], %s, 900,
                      'phase8-weight', 'requested')
            """,
            (competition, reign, [king_hotkey], [king_hotkey], "6" * 64),
        )
        self.owner.execute(
            """
            INSERT INTO control_plane.service_instances (
                service_name, instance_id, software_version, state, phase, started_at, heartbeat_at
            ) VALUES ('validator', 'validator-a', 'phase8-fixture', 'active', 'idle', %s, %s)
            """,
            (NOW, NOW),
        )
        self.ids = {
            "registration": registration,
            "upload": upload,
            "evaluation": evaluation,
            "model_digest": model_digest,
        }

    def test_projection_is_schema_valid_complete_and_secret_free(self):
        payload = self.repository.project(now=NOW)
        body = canonical_dashboard_json(payload)
        text = body.decode()
        self.assertEqual(payload["source_watermark"] > 0, True)
        self.assertEqual(len(payload["history"]), 1)
        self.assertEqual(payload["history"][0]["delta"], 0.02)
        self.assertEqual(
            payload["history"][0]["shards_used"],
            [{"source": "fixture", "names": ["part-000.npy"]}],
        )
        self.assertEqual(
            payload["history"][0]["source_scores"],
            [],
        )
        self.assertEqual(payload["history"][0]["model_identity"], "hidden_until_promotion")
        self.assertIsNone(payload["history"][0]["challenger_repo"])
        self.assertEqual(payload["history"][0]["coldkey"], "5" + "C" * 47)
        self.assertEqual(payload["history"][0]["baseline_coldkey"], "5" + "G" * 47)
        self.assertEqual(payload["king"]["coldkey"], "5" + "G" * 47)
        self.assertEqual(len(payload["dataset_versions"]), 1)
        self.assertEqual(payload["dataset_versions"][0]["config_version"], "7" * 64)
        self.assertEqual(payload["dataset_versions"][0]["dataset_label"], "fixture-datasets")
        self.assertEqual(payload["dataset_versions"][0]["sources"][0]["name"], "fixture")
        for marker in (
            "DO-NOT-LEAK",
            "secret_access_key",
            "validator-internal",
            "result_artifact_reference",
            "private_diagnostic_reference",
            "immutable_bucket",
            "datasets.example/private",
            "secret=never",
        ):
            self.assertNotIn(marker.lower(), text.lower())

    def test_privileged_dashboard_process_projects_sanitized_source_scores(self):
        self.repository.release_lock()
        privileged = DashboardProjectionRepository(
            self.owner,
            netuid=306,
            chain_generation="test",
            competition="quasar",
            chain_name="Teutonic Testnet",
            seed_repo="owner/genesis",
            seed_digest="hf:" + "b" * 40,
            seed_repo_backend="hf",
        )
        try:
            self.assertTrue(privileged.acquire_lock())
            scores = privileged.project(now=NOW)["history"][0]["source_scores"]
        finally:
            privileged.release_lock()
            self.assertTrue(self.repository.acquire_lock())
        self.assertEqual(
            scores,
            [{
                "source": "fixture",
                "n_sequences": 64,
                "avg_king_loss": 2.1,
                "avg_challenger_loss": 2.09,
                "mu_hat": 0.01,
            }],
        )

    def test_reuse_limit_failure_is_projected_with_public_reason(self):
        self.owner.execute(
            """
            UPDATE control_plane.evaluations
               SET state = 'terminal_failure', verdict = 'failed',
                   failure_class = 'unknown',
                   public_error_code = 'safetensors_reuse_limit',
                   verdict_summary = '{"error_code":"safetensors_reuse_limit"}'::jsonb,
                   completed_at = %s
             WHERE evaluation_id = %s
            """,
            (NOW, self.ids["evaluation"]),
        )

        entry = self.repository.project(now=NOW)["history"][0]
        self.assertEqual(entry["verdict"], "error")
        self.assertEqual(entry["error_code"], "safetensors_reuse_limit")
        self.assertEqual(
            entry["error_message"],
            "This model checkpoint has reached the allowed evaluation reuse limit.",
        )

    def test_model_copy_failure_is_projected_with_specific_public_reason(self):
        self.owner.execute(
            """
            UPDATE control_plane.evaluations
               SET state = 'terminal_failure', verdict = 'failed',
                   failure_class = 'policy', public_error_code = 'model_copy',
                   verdict_summary = '{"error_code":"model_copy"}'::jsonb,
                   completed_at = %s
             WHERE evaluation_id = %s
            """,
            (NOW, self.ids["evaluation"]),
        )

        entry = self.repository.project(now=NOW)["history"][0]
        self.assertEqual(entry["verdict"], "error")
        self.assertEqual(entry["error_code"], "model_copy")
        self.assertEqual(
            entry["error_message"],
            "The challenger's model weights are identical to the current king.",
        )

    def test_upload_verification_failure_is_projected_as_hidden_history_error(self):
        registration = "a" * 64
        hotkey = "5" + "F" * 47
        snapshot = self.owner.execute(
            """
            SELECT snapshot_id
              FROM control_plane.metagraph_snapshots
             WHERE netuid = 306 AND chain_generation = 'test'
            """
        ).fetchone()[0]
        self.owner.execute(
            """
            INSERT INTO control_plane.metagraph_uid_assignments
                (snapshot_id, uid, hotkey, coldkey, registration_block)
            VALUES (%s, 170, %s, %s, 950)
            """,
            (snapshot, hotkey, "5" + "D" * 47),
        )
        self.owner.execute(
            """
            INSERT INTO control_plane.registrations (
                registration_id, netuid, chain_generation, uid, hotkey,
                first_seen_finalized_block, last_seen_finalized_block, model_prefix, state
            ) VALUES (%s, 306, 'test', 170, %s, 900, 1000, %s, 'active')
            """,
            (registration, hotkey, f"models/registrations/{registration}/"),
        )
        self.owner.execute(
            """
            INSERT INTO control_plane.r2_parent_tokens (
                registration_id, cloudflare_token_id, access_key_id, state, activated_at
            ) VALUES (%s, 'failed-upload-parent-token', 'failed-upload-parent-access',
                      'active', %s)
            """,
            (registration, NOW),
        )
        upload_id = "9452d08b-b3bb-4bc9-9c45-01889fff6fa8"
        self.owner.execute(
            """
            INSERT INTO control_plane.uploads (
                upload_id, registration_id, chain_generation, signalling_hotkey,
                ready_payload, ready_finalized_block, ready_extrinsic_index,
                ready_event_index, manifest_sha256, state, failure_code, ready_at, updated_at
            ) VALUES (%s, %s, 'test', %s, 'r2ready:v1', 902, 0, 0, %s,
                      'verification_failed', 'ArtifactIntegrityError', %s, %s)
            """,
            (upload_id, registration, hotkey, "a" * 64, NOW, NOW),
        )

        projected = self.repository.project(now=NOW)
        failure = next(item for item in projected["history"] if item.get("upload_id") == upload_id)
        self.assertEqual(failure["uid"], 170)
        self.assertEqual(failure["registration_state"], "active")
        self.assertEqual(failure["upload_state"], "verification_failed")
        self.assertEqual(failure["error_code"], "ArtifactIntegrityError")
        self.assertEqual(failure["verdict"], "error")
        self.assertEqual(failure["model_identity"], "hidden_until_promotion")
        self.assertIsNone(failure["policy_version"])
        self.assertIsNone(failure["dataset_version"])
        canonical_dashboard_json(projected)

    def test_current_evaluation_projects_provisional_bootstrap_metrics(self):
        self.owner.execute(
            """
            UPDATE control_plane.evaluations
               SET state = 'evaluating', verdict = NULL, verdict_summary = NULL,
                   completed_at = NULL, started_at = %s, heartbeat_at = %s,
                   lease_expires_at = %s,
                   request_payload = %s::jsonb, progress_summary = %s::jsonb
             WHERE evaluation_id = %s
            """,
            (
                NOW,
                NOW,
                NOW + timedelta(minutes=5),
                '{"limits":{"n":2000,"delta_threshold":0.5}}',
                '{"phase":"eval_progress","completed_sequences":400,'
                '"requested_sequences":2000,"percent":20.0,'
                '"provisional_mu_hat":0.72,"provisional_lcb":0.61,'
                '"provisional_n_sequences":400,"provisional_n_bootstrap":1000}',
                self.ids["evaluation"],
            ),
        )
        projected = self.repository.project(now=NOW)
        current = projected["current_eval"]
        self.assertIsNotNone(current)
        self.assertIsNone(current["model_digest"])
        self.assertEqual(current["provisional_mu_hat"], 0.72)
        self.assertEqual(current["provisional_lcb"], 0.61)
        self.assertEqual(current["provisional_n_sequences"], 400)
        self.assertEqual(current["provisional_n_bootstrap"], 1000)
        self.assertEqual(current["delta_threshold"], 0.5)
        canonical_dashboard_json(projected)

        self.repository.release_lock()
        owner_repository = DashboardProjectionRepository(
            self.owner,
            netuid=306,
            chain_generation="test",
            competition="quasar",
            chain_name="Teutonic Testnet",
            seed_repo="owner/genesis",
            seed_digest="hf:" + "b" * 40,
            seed_repo_backend="hf",
        )
        self.assertTrue(owner_repository.acquire_lock())
        try:
            owner_current = owner_repository.project(now=NOW)["current_eval"]
            self.assertEqual(owner_current["model_digest"], self.ids["model_digest"])
        finally:
            owner_repository.release_lock()
            self.assertTrue(self.repository.acquire_lock())

    def test_current_king_uses_remapped_uid_from_latest_weight_revision(self):
        self.owner.execute(
            """
            UPDATE control_plane.weight_publications
               SET target_hotkeys = %s, target_uids = ARRAY[19], payload_revision = 2,
                   mapping_finalized_block = 1001
            """,
            ([self.king_hotkey],),
        )
        payload = self.repository.project(now=NOW)
        self.assertEqual(payload["king"]["uid"], 19)
        self.assertEqual(payload["king_payout"]["weight"], 1.0)
        self.assertEqual(payload["king_chain"][0]["uid"], 19)

    def test_king_chain_uses_weights_and_uids_from_current_plan(self):
        competition, current_reign = self.owner.execute(
            "SELECT competition_id, current_reign_id FROM control_plane.competitions"
        ).fetchone()
        eligible_hotkey = "5" + "P" * 47
        excluded_hotkey = "5" + "O" * 47
        self.owner.execute(
            "UPDATE control_plane.king_reigns SET reign_number = 2 WHERE reign_id = %s",
            (current_reign,),
        )
        self.owner.execute(
            """
            INSERT INTO control_plane.king_reigns (
                competition_id, reign_number, model_digest, public_bucket, public_prefix,
                hotkey, uid, crowned_at, crowned_finalized_block, ended_at,
                replacement_reason, operator_provenance
            ) VALUES
                (%s, 0, %s, 'teutonic-models', %s, %s, 3, %s, 700, %s,
                 'accepted_challenger', 'seed'),
                (%s, 1, %s, 'teutonic-models', %s, %s, 4, %s, 800, %s,
                 'accepted_challenger', NULL)
            """,
            (
                competition,
                "7" * 64,
                f"models/sha256/{'7' * 64}/",
                excluded_hotkey,
                NOW,
                NOW,
                competition,
                "8" * 64,
                f"models/sha256/{'8' * 64}/",
                eligible_hotkey,
                NOW,
                NOW,
            ),
        )
        self.owner.execute(
            """
            UPDATE control_plane.weight_publications
               SET policy_hotkeys = %s, target_hotkeys = %s,
                   target_uids = ARRAY[19, 23],
                   normalized_weights = ARRAY[0.5, 0.5]::double precision[],
                   payload_revision = 2, mapping_finalized_block = 1001
             WHERE source_reign_id = %s
            """,
            (
                [self.king_hotkey, eligible_hotkey],
                [self.king_hotkey, eligible_hotkey],
                current_reign,
            ),
        )

        chain = {
            reign["hotkey"]: reign for reign in self.repository.project(now=NOW)["king_chain"]
        }
        self.assertEqual(chain[self.king_hotkey]["weight"], 0.5)
        self.assertEqual(chain[self.king_hotkey]["uid"], 19)
        self.assertEqual(chain[eligible_hotkey]["weight"], 0.5)
        self.assertEqual(chain[eligible_hotkey]["uid"], 23)
        self.assertIsNone(chain[excluded_hotkey]["weight"])
        self.assertEqual(chain[excluded_hotkey]["uid"], 3)

    def test_fresh_database_projects_a_valid_empty_state(self):
        self.owner.execute(
            """
            TRUNCATE TABLE
                control_plane.weight_submission_attempts,
                control_plane.notification_outbox,
                control_plane.model_promotions,
                control_plane.evaluations,
                control_plane.weight_publications,
                control_plane.king_reigns,
                control_plane.competitions,
                control_plane.controller_jobs,
                control_plane.verified_uploads,
                control_plane.upload_files,
                control_plane.uploads,
                control_plane.credential_generations,
                control_plane.r2_parent_tokens,
                control_plane.registrations,
                control_plane.metagraph_uid_assignments,
                control_plane.metagraph_snapshots,
                control_plane.chain_cursors,
                control_plane.service_instances
            RESTART IDENTITY CASCADE
            """
        )
        empty = self.repository.project(now=NOW)
        canonical_dashboard_json(empty)
        self.assertIsNone(empty["king"])
        self.assertIsNone(empty["current_eval"])
        self.assertEqual(empty["history"], [])
        self.assertEqual(empty["queue"], [])

    def test_chain_and_identity_sources_advance_public_watermark(self):
        before = self.repository.project(now=NOW)["source_watermark"]
        self.owner.execute(
            """
            UPDATE control_plane.chain_cursors
               SET last_finalized_block = last_finalized_block + 1,
                   last_finalized_block_hash = '0x1001'
             WHERE netuid = 306 AND chain_generation = 'test'
            """
        )
        after = self.repository.project(now=NOW)["source_watermark"]
        self.assertGreater(after, before)

    def test_identity_appears_only_after_promotion_is_committed(self):
        before = self.repository.project(now=NOW)
        self.assertIsNone(before["history"][0]["challenger_digest"])
        digest = self.ids["model_digest"]
        self.owner.execute(
            """
            INSERT INTO control_plane.model_promotions (
                upload_id, evaluation_id, model_digest, disposition,
                private_bucket, private_prefix, public_bucket, public_prefix,
                state, idempotency_key, expected_object_count, expected_size_bytes,
                public_verified_at, private_deleted_at, promoted_at
            ) VALUES (%s, %s, %s, 'non_winner', 'private-models', %s,
                      'teutonic-models', %s, 'promoted', 'phase8-promotion', 1, 100, %s, %s, %s)
            """,
            (
                self.ids["upload"],
                self.ids["evaluation"],
                digest,
                f"models/registrations/{self.ids['registration']}/",
                f"models/sha256/{digest}/",
                NOW,
                NOW,
                NOW,
            ),
        )
        after = self.repository.project(now=NOW)
        history = after["history"][0]
        self.assertEqual(history["model_identity"], "public")
        self.assertEqual(history["challenger_repo"], "owner/challenger")
        self.assertEqual(history["publication_disposition"], "non_winner")

    def test_winner_identity_waits_for_committed_crown(self):
        digest = self.ids["model_digest"]
        self.owner.execute(
            "UPDATE control_plane.evaluations SET verdict = 'accepted' WHERE evaluation_id = %s",
            (self.ids["evaluation"],),
        )
        self.owner.execute(
            """
            INSERT INTO control_plane.model_promotions (
                upload_id, evaluation_id, model_digest, disposition,
                private_bucket, private_prefix, public_bucket, public_prefix,
                state, idempotency_key, expected_object_count, expected_size_bytes,
                public_verified_at, private_deleted_at, promoted_at
            ) VALUES (%s, %s, %s, 'winner', 'private-models', %s,
                      'teutonic-models', %s, 'promoted', 'phase8-winner-promotion',
                      1, 100, %s, %s, %s)
            """,
            (
                self.ids["upload"],
                self.ids["evaluation"],
                digest,
                f"models/registrations/{self.ids['registration']}/",
                f"models/sha256/{digest}/",
                NOW,
                NOW,
                NOW,
            ),
        )
        before_crown = self.repository.project(now=NOW)["history"][0]
        self.assertEqual(before_crown["model_identity"], "hidden_until_promotion")

        competition, previous = self.owner.execute(
            "SELECT competition_id, current_reign_id FROM control_plane.competitions"
        ).fetchone()
        self.owner.execute(
            """
            UPDATE control_plane.king_reigns
               SET ended_at = %s, replacement_reason = 'accepted_challenger'
             WHERE reign_id = %s
            """,
            (NOW, previous),
        )
        crowned = self.owner.execute(
            """
            INSERT INTO control_plane.king_reigns (
                competition_id, reign_number, accepted_upload_id, causing_evaluation_id,
                model_digest, public_bucket, public_prefix, hotkey, uid, previous_reign_id,
                crowned_at, crowned_finalized_block
            ) SELECT %s, 1, %s, %s, %s, 'teutonic-models', %s,
                     registration.hotkey, registration.uid, %s, %s, 1001
                FROM control_plane.uploads upload
                JOIN control_plane.registrations registration
                  ON registration.registration_id = upload.registration_id
               WHERE upload.upload_id = %s
            RETURNING reign_id
            """,
            (
                competition,
                self.ids["upload"],
                self.ids["evaluation"],
                digest,
                f"models/sha256/{digest}/",
                previous,
                NOW,
                self.ids["upload"],
            ),
        ).fetchone()[0]
        self.owner.execute(
            "UPDATE control_plane.competitions SET current_reign_id = %s WHERE competition_id = %s",
            (crowned, competition),
        )
        after_crown = self.repository.project(now=NOW)["history"][0]
        self.assertEqual(after_crown["model_identity"], "public")
        self.assertEqual(after_crown["publication_disposition"], "winner")

    def test_dashboard_role_can_only_read_approved_views(self):
        approved = self.dashboard.execute(
            """
            SELECT table_name
              FROM information_schema.role_table_grants
             WHERE grantee = 'teutonic_dashboard_view'
               AND table_schema = 'control_plane'
               AND privilege_type = 'SELECT'
             ORDER BY table_name
            """
        ).fetchall()
        names = {row[0] for row in approved}
        self.assertEqual(
            names,
            {
                "dashboard_chain",
                "dashboard_contract",
                "dashboard_current_evaluation",
                "dashboard_current_king",
                "dashboard_dataset_manifests",
                "dashboard_dataset_versions",
                "dashboard_evaluation_history",
                "dashboard_king_reigns",
                "dashboard_queue",
                "dashboard_service_health",
                "dashboard_stats",
                "dashboard_upload_failures",
                "dashboard_weight_status",
            },
        )
        with self.assertRaises(errors.InsufficientPrivilege):
            self.dashboard.execute("SELECT * FROM control_plane.evaluations").fetchall()
        with self.assertRaises(errors.InsufficientPrivilege):
            self.dashboard.execute("SELECT * FROM control_plane.r2_parent_tokens").fetchall()

    def test_global_dataset_manifest_is_projected_from_active_postgres_config(self):
        manifest = self.repository.project_dataset_manifest(now=NOW)
        self.assertEqual(manifest["config_version"], "7" * 64)
        self.assertEqual(manifest["eval_n"], 2000)
        self.assertEqual(manifest["delta_threshold"], 0.5)
        self.assertEqual(manifest["sources"][0]["name"], "fixture")
        self.assertEqual(manifest["sources"][0]["total_tokens"], 4096)
        self.assertEqual(manifest["sources"][0]["total_shards"], 1)
        self.assertIsNone(manifest["sources"][0]["sequence_length"])
        self.assertEqual(
            set(manifest["sources"][0]),
            {
                "name",
                "proportion",
                "manifest_url",
                "manifest_sha256",
                "source_repo",
                "tokenizer",
                "dtype",
                "tokenization_mode",
                "sequence_length",
                "total_tokens",
                "total_shards",
                "estimated_sequences",
            },
        )


if __name__ == "__main__":
    unittest.main()
