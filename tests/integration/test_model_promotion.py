from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

try:
    import psycopg
except ImportError:
    psycopg = None

from teutonic.promotion import (
    ObservedObject,
    PromotionInvariantError,
    PromotionRepository,
    PromotionWorker,
)

DATABASE_URL = os.environ.get("TEUTONIC_TEST_DATABASE_URL")
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


class SimulatedCrash(BaseException):
    pass


class MemoryInspector:
    def __init__(self):
        self.objects: dict[tuple[str, str], ObservedObject] = {}
        self.body_reads = 0

    def inventory(self, bucket, prefix):
        return {
            key[len(prefix) :]: value
            for (stored_bucket, key), value in self.objects.items()
            if stored_bucket == bucket and key.startswith(prefix)
        }


class MemoryRclone:
    def __init__(self, inspector):
        self.inspector = inspector
        self.copy_calls = 0
        self.delete_calls = 0

    def copy(
        self,
        *,
        source_bucket,
        source_prefix,
        destination_bucket,
        destination_prefix,
        probe_path,
        expected_object_count,
        heartbeat=None,
    ):
        self.copy_calls += 1
        if expected_object_count < 1 or not probe_path:
            raise AssertionError("promotion did not specify its route-validation probe")
        if heartbeat is not None:
            heartbeat()
        source = list(self.inspector.inventory(source_bucket, source_prefix).values())
        for item in source:
            self.inspector.objects[(destination_bucket, destination_prefix + item.path)] = item

    def delete_source(self, *, bucket, prefix, heartbeat=None):
        self.delete_calls += 1
        if heartbeat is not None:
            heartbeat()
        for stored_bucket, key in list(self.inspector.objects):
            if stored_bucket == bucket and key.startswith(prefix):
                del self.inspector.objects[(stored_bucket, key)]


@unittest.skipUnless(DATABASE_URL and psycopg, "TEUTONIC_TEST_DATABASE_URL and psycopg required")
class ModelPromotionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)
        cls.second_connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls):
        cls.second_connection.close()
        cls.connection.close()

    def setUp(self):
        self.repository = PromotionRepository(
            self.connection,
            netuid=306,
            chain_generation="test",
            competition="quasar",
            instance_id="promotion-worker-a",
        )
        self.repository.acquire_lock()
        self.inspector = MemoryInspector()
        self.executor = MemoryRclone(self.inspector)
        self._reset_and_seed()

    def tearDown(self):
        self.repository.release_lock()

    def _reset_and_seed(self, *, disposition="winner"):
        self.connection.execute(
            """
            TRUNCATE TABLE
                control_plane.notification_outbox,
                control_plane.weight_publications,
                control_plane.model_promotions,
                control_plane.evaluations,
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
                control_plane.chain_cursors
            RESTART IDENTITY CASCADE
            """
        )
        self.inspector.objects.clear()
        self.executor.copy_calls = 0
        self.executor.delete_calls = 0
        registration = "1" * 64
        model_digest = "2" * 64
        manifest_digest = "3" * 64
        prefix = f"models/registrations/{registration}/"
        self.connection.execute(
            """
            INSERT INTO control_plane.registrations (
                registration_id, netuid, chain_generation, uid, hotkey,
                first_seen_finalized_block, last_seen_finalized_block,
                model_prefix, state
            ) VALUES (%s, 306, 'test', 7, 'miner-hotkey', 100, 101, %s, 'active')
            """,
            (registration, f"models/registrations/{registration}/"),
        )
        self.connection.execute(
            """
            INSERT INTO control_plane.r2_parent_tokens (
                registration_id, cloudflare_token_id, access_key_id, state, activated_at
            ) VALUES (%s, 'promotion-token', 'promotion-access', 'active', %s)
            """,
            (registration, NOW),
        )
        upload = self.connection.execute(
            """
            INSERT INTO control_plane.uploads (
                registration_id, chain_generation, signalling_hotkey, ready_payload,
                ready_finalized_block, ready_extrinsic_index, ready_event_index,
                manifest_sha256, manifest_signature_verified, model_digest, model_name,
                object_count, total_size_bytes, state, ready_at
            ) VALUES (
                %s, 'test', 'miner-hotkey', 'r2ready:v1', 101, 0, 0,
                %s, true, %s, 'phase6/model', 2, 18, %s, %s
            ) RETURNING upload_id
            """,
            (
                registration,
                manifest_digest,
                model_digest,
                "accepted_pending_promotion" if disposition == "winner" else "rejected",
                NOW,
            ),
        ).fetchone()[0]
        files = (
            ("config.json", 2, "4" * 64),
            ("weights/model.safetensors", 16, "5" * 64),
        )
        with self.connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO control_plane.upload_files (
                    upload_id, object_path, size_bytes, sha256, verified_at
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                [(upload, path, size, digest, NOW) for path, size, digest in files],
            )
        self.connection.execute(
            """
            INSERT INTO control_plane.verified_uploads (
                upload_id, immutable_bucket, immutable_prefix, immutable_version,
                model_digest, manifest_sha256, manifest_size_bytes,
                object_count, total_size_bytes, verified_at
            ) VALUES (%s, 'private-models', %s, 'snapshot-v1', %s, %s, 100, 2, 18, %s)
            """,
            (upload, prefix, model_digest, manifest_digest, NOW),
        )
        competition = self.connection.execute(
            """
            INSERT INTO control_plane.competitions (netuid, chain_generation, name)
            VALUES (306, 'test', 'quasar') RETURNING competition_id
            """
        ).fetchone()[0]
        king = self.connection.execute(
            """
            INSERT INTO control_plane.king_reigns (
                competition_id, reign_number, model_digest, public_bucket, public_prefix,
                hotkey, uid, crowned_at, crowned_finalized_block, operator_provenance
            ) VALUES (%s, 0, %s, 'public-models', %s, 'genesis', 0, %s, 100, 'fixture')
            RETURNING reign_id
            """,
            (competition, "a" * 64, f"models/sha256/{'a' * 64}/", NOW),
        ).fetchone()[0]
        self.connection.execute(
            "UPDATE control_plane.competitions SET current_reign_id = %s WHERE competition_id = %s",
            (king, competition),
        )
        evaluation = self.connection.execute(
            """
            INSERT INTO control_plane.evaluations (
                upload_id, competition_id, attempt_number, claimed_king_reign_id,
                state, policy_version, code_version, dataset_version,
                sampling_seed, bootstrap_seed, thresholds, verdict, verdict_summary,
                completed_at
            ) VALUES (
                %s, %s, 1, %s, 'completed', 'policy-v1', 'code-v1', 'data-v1',
                1, 2, '{}'::jsonb, %s, '{}'::jsonb, %s
            ) RETURNING evaluation_id
            """,
            (
                upload,
                competition,
                king,
                "accepted" if disposition == "winner" else "rejected",
                NOW,
            ),
        ).fetchone()[0]
        promotion = self.connection.execute(
            """
            INSERT INTO control_plane.model_promotions (
                upload_id, evaluation_id, model_digest, disposition,
                private_bucket, private_prefix, public_bucket, public_prefix,
                state, idempotency_key, next_retry_at,
                expected_object_count, expected_size_bytes
            ) VALUES (
                %s, %s, %s, %s, 'private-models', %s, 'public-models', %s,
                'promotion_pending', %s, %s, 2, 18
            ) RETURNING promotion_id
            """,
            (
                upload,
                evaluation,
                model_digest,
                disposition,
                prefix,
                f"models/sha256/{model_digest}/",
                f"promote-model:{upload}",
                NOW,
            ),
        ).fetchone()[0]
        expected = {
            path: ObservedObject(path, size, digest) for path, size, digest in files
        }
        expected["manifest.json"] = ObservedObject("manifest.json", 100, manifest_digest)
        for path, item in expected.items():
            self.inspector.objects[("private-models", prefix + path)] = item
        self.upload_id = str(upload)
        self.promotion_id = str(promotion)
        self.king_id = str(king)
        self.private_prefix = prefix
        self.public_prefix = f"models/sha256/{model_digest}/"

    def _worker(self, *, after_stage=None, on_winner=None):
        return PromotionWorker(
            self.repository,
            self.executor,
            self.inspector,
            clock=lambda: NOW,
            retry_base_delay=timedelta(0),
            after_stage=after_stage,
            on_winner_promoted=on_winner,
        )


    def _specialize_fixture(self, key):
        main_id = self.connection.execute(
            "SELECT competition_id FROM control_plane.competitions WHERE name='quasar'"
        ).fetchone()[0]
        split_id = self.connection.execute(
            """INSERT INTO control_plane.competitions
                   (netuid,chain_generation,name,competition_key,main_competition_id)
               VALUES (306,'test',%s,%s,%s) RETURNING competition_id""",
            (key, key, main_id),
        ).fetchone()[0]
        self.connection.execute(
            "UPDATE control_plane.evaluations SET competition_id=%s WHERE upload_id=%s",
            (split_id, self.upload_id),
        )
        self.connection.execute(
            "UPDATE control_plane.uploads SET competition_key=%s WHERE upload_id=%s",
            (key, self.upload_id),
        )
        return split_id

    def _coordinator(self):
        from teutonic.validator import ValidatorRepository
        from teutonic.validator.runtime import CrownCoordinator, FinalizedMetagraph

        class Chain:
            def snapshot(self):
                return FinalizedMetagraph(110, {"genesis": 0, "miner-hotkey": 7, "second-hotkey": 8})

        repository = ValidatorRepository(
            self.connection, netuid=306, chain_generation="test", competition="quasar",
            instance_id="promotion-regression", public_model_bucket="public-models",
        )
        return CrownCoordinator(repository, Chain())

    def _seed_owner_snapshot(self, block, uid, hotkey, coldkey):
        snapshot = self.connection.execute(
            """INSERT INTO control_plane.metagraph_snapshots
                   (netuid,chain_generation,finalized_block,finalized_block_hash,
                    snapshot_checksum,uid_count,is_complete,observed_at)
               VALUES (306,'test',%s,%s,%s,1,true,%s) RETURNING snapshot_id""",
            (block, f"0x{block:064x}", f"{block:064x}", NOW),
        ).fetchone()[0]
        if coldkey is None:
            return
        self.connection.execute(
            """INSERT INTO control_plane.metagraph_uid_assignments
                   (snapshot_id,uid,hotkey,coldkey,registration_block)
               VALUES (%s,%s,%s,%s,100)""",
            (snapshot, uid, hotkey, coldkey),
        )

    def _seed_reused_submission(self, key="code", *, coldkey="original-coldkey",
                                missing_original=False, altered_inventory=False):
        """Reuse a checkpoint under a new hotkey and deliver the verdict twice."""
        from teutonic.validator import ValidatorRepository

        main_id = self.connection.execute(
            "SELECT competition_id FROM control_plane.competitions WHERE name='quasar'"
        ).fetchone()[0]
        split_id = self.connection.execute(
            """INSERT INTO control_plane.competitions
                   (netuid,chain_generation,name,competition_key,main_competition_id)
               VALUES (306,'test',%s,%s,%s) RETURNING competition_id""",
            (key, key, main_id),
        ).fetchone()[0]
        registration = "6" * 64
        prefix = f"models/registrations/{registration}/"
        self.connection.execute(
            """INSERT INTO control_plane.registrations
                   (registration_id,netuid,chain_generation,uid,hotkey,
                    first_seen_finalized_block,last_seen_finalized_block,model_prefix,state)
               VALUES (%s,306,'test',8,'second-hotkey',100,102,%s,'active')""",
            (registration, prefix),
        )
        self.connection.execute(
            """INSERT INTO control_plane.r2_parent_tokens
                   (registration_id,cloudflare_token_id,access_key_id,state,activated_at)
               VALUES (%s,'second-token','second-access','active',%s)""",
            (registration, NOW),
        )
        upload = self.connection.execute(
            """INSERT INTO control_plane.uploads
                   (registration_id,chain_generation,signalling_hotkey,ready_payload,
                    ready_finalized_block,ready_extrinsic_index,ready_event_index,
                    manifest_sha256,manifest_signature_verified,model_digest,model_name,
                    object_count,total_size_bytes,state,ready_at,competition_key)
               SELECT %s,chain_generation,'second-hotkey','r2ready:v1',102,0,0,
                      %s,true,model_digest,'second/model',object_count,total_size_bytes,
                      'evaluation_claimed',%s,%s
                 FROM control_plane.uploads WHERE upload_id=%s RETURNING upload_id""",
            (registration, "7" * 64, NOW, key, self.upload_id),
        ).fetchone()[0]
        self.connection.execute(
            """INSERT INTO control_plane.verified_uploads
                   (upload_id,immutable_bucket,immutable_prefix,model_digest,manifest_sha256,
                    manifest_size_bytes,object_count,total_size_bytes,verified_at)
               SELECT %s,immutable_bucket,%s,model_digest,%s,100,object_count,total_size_bytes,%s
                 FROM control_plane.verified_uploads WHERE upload_id=%s""",
            (upload, prefix, "7" * 64, NOW, self.upload_id),
        )
        self.connection.execute(
            """INSERT INTO control_plane.upload_files (upload_id,object_path,size_bytes,sha256,verified_at)
               SELECT %s,object_path,size_bytes,sha256,verified_at FROM control_plane.upload_files
                WHERE upload_id=%s""", (upload, self.upload_id),
        )
        if not missing_original:
            self._seed_owner_snapshot(101, 7, "miner-hotkey", "original-coldkey")
        self._seed_owner_snapshot(102, 8, "second-hotkey", coldkey)
        if altered_inventory:
            # A README/config edit or renamed shard changes the full inventory
            # digest but does not turn someone else's weights into a new model.
            for table in ("uploads", "verified_uploads"):
                self.connection.execute(
                    f"UPDATE control_plane.{table} SET model_digest=%s WHERE upload_id=%s",
                    ("e" * 64, upload),
                )
            self.connection.execute(
                """UPDATE control_plane.upload_files SET object_path='renamed.safetensors'
                    WHERE upload_id=%s AND object_path LIKE '%%.safetensors'""", (upload,),
            )
            self.connection.execute(
                """UPDATE control_plane.upload_files SET sha256=%s
                    WHERE upload_id=%s AND object_path='config.json'""", ("d" * 64, upload),
            )
        evaluation = self.connection.execute(
            """INSERT INTO control_plane.evaluations
                   (upload_id,competition_id,attempt_number,claimed_king_reign_id,state,
                    owner_instance_id,lease_expires_at,policy_version,code_version,dataset_version,
                    sampling_seed,bootstrap_seed,thresholds)
               VALUES (%s,%s,1,%s,'claimed','reuse-validator',%s,'policy-v1','code-v1',
                       'data-v1',1,2,'{}'::jsonb) RETURNING evaluation_id""",
            (upload, split_id, self.king_id, NOW + timedelta(minutes=5)),
        ).fetchone()[0]
        validator = ValidatorRepository(
            self.connection, netuid=306, chain_generation="test", competition="quasar",
            instance_id="reuse-validator", public_model_bucket="public-models",
        )
        validator.acquire_lock()
        try:
            result = {"accepted": True, "result_artifact_sha256": "f" * 64}
            validator.complete_verdict(str(evaluation), result=result, now=NOW, publish_non_winning=False)
            # Repeated delivery must not add another promotion or change ownership.
            validator.complete_verdict(str(evaluation), result=result, now=NOW, publish_non_winning=False)
        finally:
            validator.release_lock()
        return upload, evaluation, split_id

    def test_main_worker_promotes_and_crowns_each_specialist(self):
        for key in ("math", "code", "text"):
            with self.subTest(competition=key):
                self._reset_and_seed()
                split_id = self._specialize_fixture(key)
                coordinator = self._coordinator()
                worker = self._worker(on_winner=coordinator)
                self.assertTrue(worker.run_one(propagate=True))
                row = self.connection.execute(
                    """SELECT k.hotkey,k.accepted_upload_id,c.current_reign_id
                         FROM control_plane.competitions c
                         JOIN control_plane.king_reigns k ON k.reign_id=c.current_reign_id
                        WHERE c.competition_id=%s""", (split_id,),
                ).fetchone()
                self.assertEqual(row[:2], ("miner-hotkey", UUID(self.upload_id)))
                self.assertEqual(str(self.connection.execute(
                    "SELECT current_reign_id FROM control_plane.competitions WHERE name='quasar'"
                ).fetchone()[0]), self.king_id)
                self.assertEqual(self.connection.execute(
                    "SELECT state FROM control_plane.uploads WHERE upload_id=%s", (self.upload_id,)
                ).fetchone()[0], "accepted")
                self.assertEqual(self.executor.copy_calls, 1)
                self.assertFalse(worker.run_one(propagate=True))
                self.assertEqual(coordinator(self.promotion_id), str(row[2]))

    def test_specialist_copy_and_crown_recover_after_each_worker_crash_stage(self):
        stages = ("copy_completed", "public_verification_started", "public_copy_verified",
                  "private_deletion_started", "private_source_deleted", "promoted")
        for stage in stages:
            with self.subTest(stage=stage):
                self._reset_and_seed()
                self._specialize_fixture("math")
                coordinator = self._coordinator()

                def crash_here(observed_stage, _claim, stop_at=stage):
                    if observed_stage == stop_at:
                        raise SimulatedCrash(stop_at)

                with self.assertRaises(SimulatedCrash):
                    self._worker(after_stage=crash_here, on_winner=coordinator).run_one()
                self.assertTrue(self._worker(on_winner=coordinator).run_one(propagate=True))
                self.assertEqual(self.connection.execute(
                    "SELECT state FROM control_plane.uploads WHERE upload_id=%s", (self.upload_id,)
                ).fetchone()[0], "accepted")
                self.assertEqual(self.connection.execute(
                    "SELECT count(*) FROM control_plane.king_reigns WHERE causing_evaluation_id IS NOT NULL"
                ).fetchone()[0], 1)
                self.assertEqual(self.executor.copy_calls, 1)
                self.assertEqual(self.executor.delete_calls, 1)

    def test_same_coldkey_reuse_crowns_new_hotkey_without_recopying(self):
        self._specialize_fixture("math")
        worker = self._worker(on_winner=self._coordinator())
        self.assertTrue(worker.run_one(propagate=True))
        original_public = dict(self.inspector.inventory("public-models", self.public_prefix))
        upload, evaluation, split_id = self._seed_reused_submission()
        self.assertEqual(self.repository.pending_winner_crowns(), (self.promotion_id,))
        self.assertTrue(worker.run_one(propagate=True))
        self.assertEqual(self.connection.execute(
            """SELECT hotkey,uid,accepted_upload_id,causing_evaluation_id
                 FROM control_plane.king_reigns WHERE competition_id=%s""", (split_id,),
        ).fetchone(), ("second-hotkey", 8, upload, evaluation))
        self.assertEqual(self.connection.execute(
            "SELECT state FROM control_plane.uploads WHERE upload_id=%s", (upload,)
        ).fetchone()[0], "accepted")
        self.assertEqual(self.connection.execute(
            "SELECT upload_id FROM control_plane.model_promotions"
        ).fetchone()[0], UUID(self.upload_id))
        self.assertEqual(self.executor.copy_calls, 1)
        self.assertEqual(self.executor.delete_calls, 1)
        self.assertEqual(self.inspector.inventory("public-models", self.public_prefix), original_public)
        weights = self.connection.execute(
            """SELECT policy_hotkeys,policy_weights FROM control_plane.weight_publications
                WHERE source_reign_id=(SELECT reward_reign_id FROM control_plane.competitions WHERE name='quasar')"""
        ).fetchone()
        self.assertEqual(dict(zip(*weights)), {"genesis": 0.7, "miner-hotkey": 0.15, "second-hotkey": 0.15})
        self.assertFalse(worker.run_one(propagate=True))

    def test_foreign_or_unverified_owner_cannot_replace_published_winner(self):
        for case in ("foreign", "renamed_weights", "missing_original", "missing_candidate"):
            with self.subTest(case=case):
                self._reset_and_seed()
                self._specialize_fixture("math")
                worker = self._worker(on_winner=self._coordinator())
                self.assertTrue(worker.run_one(propagate=True))
                original_promotion = self.connection.execute(
                    "SELECT upload_id,evaluation_id FROM control_plane.model_promotions"
                ).fetchone()
                _upload, evaluation, split = self._seed_reused_submission(
                    coldkey=None if case == "missing_candidate" else "foreign-coldkey",
                    missing_original=case == "missing_original",
                    altered_inventory=case == "renamed_weights",
                )
                self.assertEqual(self.connection.execute(
                    """SELECT e.state,e.verdict,e.public_error_code,u.state
                         FROM control_plane.evaluations e JOIN control_plane.uploads u USING(upload_id)
                        WHERE e.evaluation_id=%s""", (evaluation,),
                ).fetchone(), ("terminal_failure", "failed", "model_copy", "evaluation_failed"))
                self.assertEqual(self.connection.execute(
                    "SELECT upload_id,evaluation_id FROM control_plane.model_promotions"
                ).fetchone(), original_promotion)
                self.assertIsNone(self.connection.execute(
                    "SELECT current_reign_id FROM control_plane.competitions WHERE competition_id=%s",
                    (split,),
                ).fetchone()[0])
                self.assertFalse(worker.run_one(propagate=True))
                self.assertEqual(self.connection.execute(
                    "SELECT count(*) FROM control_plane.weight_publications"
                ).fetchone()[0], 1)
                self.assertEqual(self.executor.copy_calls, 1)

    def test_crown_rechecks_owner_and_does_not_transfer_rewards(self):
        self._specialize_fixture("math")
        worker = self._worker(on_winner=self._coordinator())
        self.assertTrue(worker.run_one(propagate=True))
        upload, _evaluation, split = self._seed_reused_submission()
        # Simulate an accepted verdict bypassing the earlier ownership boundary.
        self.connection.execute(
            """UPDATE control_plane.metagraph_uid_assignments SET coldkey='foreign-coldkey'
                WHERE hotkey='second-hotkey'"""
        )
        self.assertTrue(worker.run_one(propagate=True))
        self.assertEqual(self.connection.execute(
            "SELECT state,failure_code FROM control_plane.uploads WHERE upload_id=%s", (upload,),
        ).fetchone(), ("evaluation_failed", "model_copy"))
        self.assertIsNone(self.connection.execute(
            "SELECT current_reign_id FROM control_plane.competitions WHERE competition_id=%s",
            (split,),
        ).fetchone()[0])
        self.assertEqual(self.connection.execute(
            "SELECT count(*) FROM control_plane.weight_publications"
        ).fetchone()[0], 1)
        self.assertFalse(worker.run_one(propagate=True))

    def test_later_coldkey_assignment_does_not_change_original_ownership(self):
        from psycopg.rows import dict_row

        from teutonic.validator.model_ownership import model_ownership_error

        self._specialize_fixture("math")
        self.assertTrue(self._worker(on_winner=self._coordinator()).run_one(propagate=True))
        upload, _, _ = self._seed_reused_submission(coldkey="foreign-coldkey")
        self._seed_owner_snapshot(110, 7, "miner-hotkey", "foreign-coldkey")
        with self.connection.cursor(row_factory=dict_row) as cursor:
            self.assertEqual(model_ownership_error(cursor, str(upload)),
                             "published_model_owner_mismatch")

    def test_reused_non_winner_artifact_can_finish_copying_for_a_new_winner(self):
        for timing in ("before_copy", "during_copy", "after_copy"):
            with self.subTest(timing=timing):
                self._reset_and_seed(disposition="non_winner")
                self._specialize_fixture("math")
                result = []
                if timing == "before_copy":
                    result.append(self._seed_reused_submission())
                elif timing == "after_copy":
                    self.assertTrue(self._worker().run_one(propagate=True))
                    result.append(self._seed_reused_submission())

                def change_during_copy(stage, _claim, when=timing, submissions=result):
                    if stage == "copy_completed" and when == "during_copy":
                        submissions.append(self._seed_reused_submission())

                worker = self._worker(after_stage=change_during_copy, on_winner=self._coordinator())
                self.assertTrue(worker.run_one(propagate=True))
                if timing == "during_copy":
                    # The in-flight claim began as a non-winner. Reconciliation
                    # must still crown the newly associated winner next cycle.
                    self.assertTrue(worker.run_one(propagate=True))
                upload, evaluation, _ = result[0]
                self.assertEqual(self.connection.execute(
                    "SELECT hotkey,accepted_upload_id FROM control_plane.king_reigns WHERE causing_evaluation_id=%s",
                    (evaluation,),
                ).fetchone(), ("second-hotkey", upload))
                self.assertEqual(self.connection.execute(
                    "SELECT state FROM control_plane.uploads WHERE upload_id=%s", (self.upload_id,)
                ).fetchone()[0], "rejected")
                self.assertEqual(self.connection.execute(
                    "SELECT state FROM control_plane.uploads WHERE upload_id=%s", (upload,)
                ).fetchone()[0], "accepted")
                self.assertEqual(self.executor.copy_calls, 1)
                self.assertFalse(worker.run_one(propagate=True))

    def test_promotion_lock_covers_all_competitions_in_the_generation(self):
        from teutonic.promotion.repository import PromotionWorkerLockUnavailable

        second = PromotionRepository(
            self.second_connection, netuid=306, chain_generation="test", competition="math",
            instance_id="duplicate-worker",
        )
        with self.assertRaises(PromotionWorkerLockUnavailable):
            second.acquire_lock()

    def test_crown_rejects_artifact_that_differs_from_evaluated_model(self):
        from teutonic.validator.repository import SchedulerInvariantError

        self._specialize_fixture("math")
        self.assertTrue(self._worker().run_one(propagate=True))
        self.connection.execute(
            "UPDATE control_plane.verified_uploads SET model_digest=%s WHERE upload_id=%s",
            ("f" * 64, self.upload_id),
        )
        with self.assertRaisesRegex(SchedulerInvariantError, "differs from evaluated model"):
            self._coordinator()(self.promotion_id)
        self.assertEqual(self.connection.execute(
            "SELECT count(*) FROM control_plane.king_reigns"
        ).fetchone()[0], 1)
        self.assertEqual(self.connection.execute(
            "SELECT count(*) FROM control_plane.weight_publications"
        ).fetchone()[0], 0)

    def test_main_worker_does_not_claim_specialist_from_another_root(self):
        split = self._specialize_fixture("math")
        other = self.connection.execute(
            "INSERT INTO control_plane.competitions(netuid,chain_generation,name) VALUES(306,'test','unrelated') RETURNING competition_id"
        ).fetchone()[0]
        self.connection.execute(
            "UPDATE control_plane.competitions SET main_competition_id=%s WHERE competition_id=%s",
            (other, split),
        )
        self.assertFalse(self._worker().run_one(propagate=True))
        self.assertEqual(self.executor.copy_calls, 0)

    def test_winner_promotes_only_after_verified_copy_and_private_deletion(self):
        before = self.connection.execute(
            """
            SELECT public_model_name, public_model_digest, public_model_reference
              FROM control_plane.dashboard_evaluation_history
             WHERE challenge_id IS NOT NULL
            """
        ).fetchone()
        self.assertEqual(before, (None, None, None))
        callbacks = []
        self.assertTrue(self._worker(on_winner=callbacks.append).run_one(propagate=True))

        promotion = self.connection.execute(
            """
            SELECT state, observed_object_count, observed_size_bytes,
                   public_verified_at IS NOT NULL, private_deleted_at IS NOT NULL
              FROM control_plane.model_promotions WHERE promotion_id = %s
            """,
            (self.promotion_id,),
        ).fetchone()
        self.assertEqual(promotion, ("promoted", 2, 18, True, True))
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(self.inspector.inventory("private-models", self.private_prefix), {})
        self.assertEqual(len(self.inspector.inventory("public-models", self.public_prefix)), 3)
        self.assertEqual(self.inspector.body_reads, 0)
        after = self.connection.execute(
            """
            SELECT public_model_name, public_model_digest, public_model_reference
              FROM control_plane.dashboard_evaluation_history
             WHERE challenge_id IS NOT NULL
            """
        ).fetchone()
        # A winning copy is still hidden until the separate crown callback commits
        # a matching king_reigns row. Phase 8 tests the post-crown disclosure.
        self.assertEqual(after, (None, None, None))

    def test_crown_callback_failure_retries_without_recopying_model(self):
        calls = []

        def crown(promotion_id):
            calls.append(promotion_id)
            if len(calls) == 1:
                raise RuntimeError("simulated crown transaction outage")
            self.connection.execute(
                "UPDATE control_plane.uploads SET state = 'accepted' WHERE upload_id = %s",
                (self.upload_id,),
            )

        worker = self._worker(on_winner=crown)
        self.assertTrue(worker.run_one())
        self.assertEqual(
            self.connection.execute("SELECT state FROM control_plane.uploads").fetchone()[0],
            "promoted",
        )
        self.assertTrue(worker.run_one(propagate=True))
        self.assertEqual(calls, [self.promotion_id, self.promotion_id])
        self.assertEqual(self.executor.copy_calls, 1)
        self.assertEqual(self.executor.delete_calls, 1)
        self.assertEqual(
            self.connection.execute("SELECT state FROM control_plane.uploads").fetchone()[0],
            "accepted",
        )

    def test_forced_crash_at_every_stage_reconciles_idempotently(self):
        stages = (
            "copy_completed",
            "public_verification_started",
            "public_copy_verified",
            "private_deletion_started",
            "private_source_deleted",
            "promoted",
        )
        for stage in stages:
            with self.subTest(stage=stage):
                self._reset_and_seed()
                crashed = False

                def crash_here(observed_stage, _claim):
                    nonlocal crashed
                    if observed_stage == stage and not crashed:
                        crashed = True
                        raise SimulatedCrash(stage)

                with self.assertRaises(SimulatedCrash):
                    self._worker(after_stage=crash_here).run_one()
                self.assertTrue(crashed)
                self._worker().run_one(propagate=True)
                state = self.connection.execute(
                    "SELECT state FROM control_plane.model_promotions WHERE promotion_id = %s",
                    (self.promotion_id,),
                ).fetchone()[0]
                self.assertEqual(state, "promoted")
                self.assertEqual(
                    self.inspector.inventory("private-models", self.private_prefix), {}
                )
                self.assertEqual(
                    len(self.inspector.inventory("public-models", self.public_prefix)), 3
                )

    def test_digest_collision_fails_without_deleting_source_or_changing_king(self):
        self.inspector.objects[("public-models", self.public_prefix + "config.json")] = ObservedObject(
            "config.json", 2, "f" * 64
        )
        self._worker().run_one()
        state = self.connection.execute(
            "SELECT state, last_error_code FROM control_plane.model_promotions"
        ).fetchone()
        self.assertEqual(state, ("failed", "PromotionCollisionError"))
        self.assertEqual(len(self.inspector.inventory("private-models", self.private_prefix)), 3)
        self.assertEqual(
            str(
                self.connection.execute(
                    "SELECT current_reign_id FROM control_plane.competitions"
                ).fetchone()[0]
            ),
            self.king_id,
        )
        self.assertEqual(
            self.connection.execute("SELECT state FROM control_plane.uploads").fetchone()[0],
            "accepted_pending_promotion",
        )

    def test_expired_copy_lease_is_reclaimed_by_a_new_worker(self):
        crashed = False

        def crash_after_copy(stage, _claim):
            nonlocal crashed
            if stage == "copy_completed" and not crashed:
                crashed = True
                raise SimulatedCrash(stage)

        with self.assertRaises(SimulatedCrash):
            self._worker(after_stage=crash_after_copy).run_one()
        self.repository.release_lock()
        replacement = PromotionRepository(
            self.second_connection,
            netuid=306,
            chain_generation="test",
            competition="quasar",
            instance_id="promotion-worker-b",
        )
        replacement.acquire_lock()
        try:
            worker = PromotionWorker(
                replacement,
                self.executor,
                self.inspector,
                clock=lambda: NOW + timedelta(minutes=3),
            )
            self.assertTrue(worker.run_one(propagate=True))
            self.assertEqual(
                self.connection.execute(
                    "SELECT state FROM control_plane.model_promotions"
                ).fetchone()[0],
                "promoted",
            )
        finally:
            replacement.release_lock()
            self.repository.acquire_lock()

    def test_restricted_validator_role_can_complete_promotion(self):
        self.repository.release_lock()
        self.second_connection.execute("SET ROLE teutonic_validator")
        restricted = PromotionRepository(
            self.second_connection,
            netuid=306,
            chain_generation="test",
            competition="quasar",
            instance_id="promotion-worker-restricted",
        )
        restricted.acquire_lock()
        try:
            worker = PromotionWorker(restricted, self.executor, self.inspector, clock=lambda: NOW)
            self.assertTrue(worker.run_one(propagate=True))
            self.assertEqual(
                self.connection.execute(
                    "SELECT state FROM control_plane.model_promotions"
                ).fetchone()[0],
                "promoted",
            )
        finally:
            restricted.release_lock()
            self.second_connection.execute("RESET ROLE")
            self.repository.acquire_lock()

    def test_non_winner_promotion_never_changes_verdict_king_or_weights(self):
        self._reset_and_seed(disposition="non_winner")
        callbacks = []
        self._worker(on_winner=callbacks.append).run_one(propagate=True)
        self.assertEqual(callbacks, [])
        self.assertEqual(
            self.connection.execute("SELECT state FROM control_plane.uploads").fetchone()[0],
            "rejected",
        )
        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM control_plane.king_reigns").fetchone()[0],
            1,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.weight_publications"
            ).fetchone()[0],
            0,
        )

    def test_invalid_or_mismatched_verdict_is_ineligible_for_promotion(self):
        self.connection.execute(
            "UPDATE control_plane.evaluations SET verdict = 'rejected'"
        )
        with self.assertRaises(PromotionInvariantError):
            self._worker().run_one()
        self.assertEqual(self.executor.copy_calls, 0)
        self.assertEqual(self.executor.delete_calls, 0)
        self.assertEqual(
            len(self.inspector.inventory("private-models", self.private_prefix)), 3
        )
