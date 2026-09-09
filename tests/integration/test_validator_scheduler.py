from __future__ import annotations

import asyncio
import hashlib
import os
import unittest
from datetime import datetime, timedelta, timezone

try:
    import psycopg
except ImportError:
    psycopg = None

from teutonic.evaluation import (
    EarlyStoppingPolicy,
    EvaluatorJobNotFoundError,
    EvaluationRequestV2,
    result_provenance,
)
from teutonic.evaluation.configuration import DatasetManifestSnapshot, canonical_manifest_bytes
from teutonic.validator import (
    EvaluationPolicyConfig,
    SchedulerLockUnavailable,
    ValidatorRepository,
    ValidatorScheduler,
)


DATABASE_URL = os.environ.get("TEUTONIC_TEST_DATABASE_URL")
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def policy(**overrides) -> EvaluationPolicyConfig:
    manifest = {
        "shards": [{
            "key": "shards/part-000.npy",
            "sha256": "a" * 64,
            "size_bytes": 16384,
            "n_tokens": 16384,
        }]
    }
    snapshot = DatasetManifestSnapshot(
        name="fixture",
        manifest_url="https://datasets.example/fixture/manifest.json",
        manifest_sha256=hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
        proportion=1.0,
        manifest=manifest,
    )
    values = {
        "policy_version": "quasar-paired-v1",
        "code_version": "phase5-test",
        "dataset_version": "b" * 64,
        "evaluator_version": "teutonic-evaluator-v2",
        "sampling_seed": 7,
        "bootstrap_seed": 11,
        "n": 32,
        "seq_len": 64,
        "n_bootstrap": 128,
        "alpha": 0.05,
        "delta_threshold": 0.0015,
        "dataset_source": "pretokenized_npy",
        "dataset_label": "fixture-v1",
        "shards_per_dataset": 4,
        "dataset_manifests": (snapshot,),
        "lease": timedelta(minutes=2),
        "retry_base_delay": timedelta(seconds=5),
        "max_attempts": 3,
    }
    values.update(overrides)
    return EvaluationPolicyConfig(**values)


def terminal_result(request_payload, *, accepted: bool):
    request = EvaluationRequestV2.from_mapping(request_payload)
    result = result_provenance(
        request,
        started_at=NOW.isoformat(),
        completed_at=(NOW + timedelta(minutes=1)).isoformat(),
        requested_sequences=32,
        completed_sequences=32,
        early_stopped=False,
        hardware={"worker": "fixture-gpu"},
    )
    result.update(
        {
            "accepted": accepted,
            "verdict": "challenger" if accepted else "king",
            "mu_hat": 0.003 if accepted else 0.0,
            "lcb": 0.002 if accepted else -0.001,
            "delta_threshold": 0.0015,
            "avg_king_loss": 1.2,
            "avg_challenger_loss": 1.197 if accepted else 1.201,
            "wall_time_s": 60.0,
            "result_artifact_sha256": "f" * 64,
        }
    )
    return result


class FakeEvaluator:
    def __init__(self, result):
        self.result = result
        self.started = []

    async def health(self):
        return {"status": "ok"}

    async def start_attempt(self, request):
        parsed = EvaluationRequestV2.from_mapping(request)
        self.started.append(parsed.request_payload)
        return {"eval_id": parsed.eval_id, "duplicate": False}

    async def events(self, eval_id):
        request = EvaluationRequestV2.from_mapping(self.started[-1])
        yield {
            "evaluation_id": request.evaluation_id,
            "attempt_number": request.attempt_number,
            "type": "progress",
            "data": {
                "phase": "scoring",
                "done": 16,
                "total": 32,
                "provisional_mu_hat": 0.003,
                "provisional_lcb": 0.002,
                "provisional_n_sequences": 16,
                "provisional_n_bootstrap": 128,
                "private_worker_hostname": "must-not-persist",
            },
        }
        yield {
            "evaluation_id": request.evaluation_id,
            "attempt_number": request.attempt_number,
            "type": "verdict",
            "data": self.result(request.request_payload),
        }

    async def status(self, eval_id):
        return {"eval_id": eval_id, "state": "running"}


class RecoveryEvaluator:
    def __init__(self, request, *, lost=False):
        self.request = request
        self.lost = lost

    async def health(self):
        return {"status": "ok"}

    async def status(self, eval_id):
        if self.lost:
            raise EvaluatorJobNotFoundError(eval_id)
        return {
            "eval_id": eval_id,
            "state": "completed",
            "verdict": terminal_result(self.request, accepted=False),
        }


class UnavailableEvaluator:
    async def health(self):
        raise ConnectionError("GPU tunnel unavailable")


class ReuseLimitEvaluator(FakeEvaluator):
    async def events(self, eval_id):
        request = EvaluationRequestV2.from_mapping(self.started[-1])
        yield {
            "evaluation_id": request.evaluation_id,
            "attempt_number": request.attempt_number,
            "type": "error",
            "data": {
                "error": (
                    "challenger safetensors SHA-256 private-digest has already completed "
                    "3 evals; maximum allowed is 3"
                ),
            },
        }


class LegacyDuplicateEvaluator(FakeEvaluator):
    async def events(self, eval_id):
        request = EvaluationRequestV2.from_mapping(self.started[-1])
        yield {
            "evaluation_id": request.evaluation_id,
            "attempt_number": request.attempt_number,
            "type": "error",
            "data": {
                "code": "evaluation_failed",
                "error": "challenger .safetensors are identical to the king",
            },
        }


class DispatchOutageEvaluator(FakeEvaluator):
    def __init__(self, result):
        super().__init__(result)
        self.dispatches = []

    async def start_attempt(self, request):
        parsed = EvaluationRequestV2.from_mapping(request)
        self.dispatches.append(parsed.request_payload)
        if len(self.dispatches) == 1:
            raise ConnectionError("connecterror: GPU tunnel unavailable")
        self.started.append(parsed.request_payload)
        return {"eval_id": parsed.eval_id, "duplicate": False}


@unittest.skipUnless(DATABASE_URL and psycopg, "TEUTONIC_TEST_DATABASE_URL and psycopg required")
class ValidatorSchedulerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)
        cls.second_connection = psycopg.connect(DATABASE_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.second_connection.close()
        cls.connection.close()

    def setUp(self) -> None:
        self.connection.execute(
            """
            TRUNCATE TABLE
                control_plane.service_instances,
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
        for block in (100, 101, 103):
            snapshot = self.connection.execute(
                """
                INSERT INTO control_plane.metagraph_snapshots (
                    netuid, chain_generation, finalized_block, finalized_block_hash,
                    snapshot_checksum, uid_count, is_complete, observed_at
                ) VALUES (306, 'test', %s, %s, %s, 4, true, %s)
                RETURNING snapshot_id
                """,
                (block, f"0x{block:064x}", f"{block:064x}", NOW),
            ).fetchone()[0]
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO control_plane.metagraph_uid_assignments
                        (snapshot_id, uid, hotkey, coldkey, registration_block)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    [
                        (snapshot, uid, f"hotkey-{uid}", f"coldkey-{uid}", block)
                        for uid in (1, 2, 3, 4)
                    ],
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
            ) VALUES (%s, 0, %s, 'public-models', %s, 'genesis-hotkey', 0, %s, 100, 'test')
            RETURNING reign_id
            """,
            (competition, "a" * 64, f"models/sha256/{'a' * 64}/", NOW),
        ).fetchone()[0]
        self.connection.execute(
            "UPDATE control_plane.competitions SET current_reign_id = %s WHERE competition_id = %s",
            (king, competition),
        )
        self.competition_id = competition
        self.connection.execute(
            """
            INSERT INTO control_plane.evaluation_early_stopping_policies (
                competition_id, enabled, min_fraction, advantage_quantile,
                margin, check_interval
            ) VALUES (%s, true, 0.4, 0.95, 0.0, 100)
            """,
            (competition,),
        )
        self.genesis_reign_id = king
        self.uploads = [
            self._seed_upload(uid=uid, block=block, extrinsic=extrinsic, event=event)
            for uid, block, extrinsic, event in (
                (1, 103, 0, 0),
                (2, 101, 2, 1),
                (3, 101, 2, 0),
            )
        ]
        self.repository = self._repository(self.connection, "validator-a")
        self.repository.acquire_lock()

    def tearDown(self) -> None:
        self.repository.release_lock()

    def _repository(self, connection, instance_id):
        return ValidatorRepository(
            connection,
            netuid=306,
            chain_generation="test",
            competition="quasar",
            instance_id=instance_id,
            public_model_bucket="public-models",
        )

    def _seed_upload(self, *, uid, block, extrinsic, event):
        registration_id = f"{uid:064x}"
        digest = f"{uid + 1:064x}"
        self.connection.execute(
            """
            INSERT INTO control_plane.registrations (
                registration_id, netuid, chain_generation, uid, hotkey,
                first_seen_finalized_block, last_seen_finalized_block, model_prefix, state
            ) VALUES (%s, 306, 'test', %s, %s, 100, 103, %s, 'active')
            """,
            (registration_id, uid, f"hotkey-{uid}", f"models/registrations/{registration_id}/"),
        )

        self.connection.execute(
            """
            INSERT INTO control_plane.r2_parent_tokens (
                registration_id, cloudflare_token_id, access_key_id, state, activated_at
            ) VALUES (%s, %s, %s, 'active', %s)
            """,
            (registration_id, f"cloudflare-token-{uid}", f"access-key-{uid}", NOW),
        )
        upload = self.connection.execute(
            """
            INSERT INTO control_plane.uploads (
                registration_id, chain_generation, signalling_hotkey, ready_payload,
                ready_finalized_block, ready_extrinsic_index, ready_event_index,
                manifest_sha256, manifest_signature_verified, model_digest, model_name,
                object_count, total_size_bytes, state, ready_at
            ) VALUES (%s, 'test', %s, 'r2ready:v1', %s, %s, %s, %s, true, %s,
                      %s, 2, 1024, 'ready_for_evaluation', %s)
            RETURNING upload_id
            """,
            (
                registration_id,
                f"hotkey-{uid}",
                block,
                extrinsic,
                event,
                f"{uid + 10:064x}",
                digest,
                f"model-{uid}",
                NOW,
            ),
        ).fetchone()[0]
        self.connection.execute(
            """
            INSERT INTO control_plane.verified_uploads (
                upload_id, immutable_bucket, immutable_prefix, model_digest,
                manifest_sha256, object_count, total_size_bytes, verified_at
            ) VALUES (%s, 'private-models', %s, %s, %s, 2, 1024, %s)
            """,
            (
                upload,
                f"models/registrations/{registration_id}/",
                digest,
                f"{uid + 10:064x}",
                NOW,
            ),
        )
        return upload

    def test_early_stopping_policy_is_loaded_and_bound_to_claim(self):
        loaded = self.repository.load_early_stopping_policy()
        self.assertEqual(
            loaded,
            EarlyStoppingPolicy(
                enabled=True,
                min_fraction=0.4,
                advantage_quantile=0.95,
                margin=0.0,
                check_interval=100,
            ),
        )
        claim = self.repository.claim_next(
            now=NOW,
            policy=policy(early_stopping=loaded),
        )
        self.assertIsNotNone(claim)
        parsed = EvaluationRequestV2.from_mapping(claim.request)
        self.assertEqual(parsed.early_stopping, loaded.request_dict())
        row = self.connection.execute(
            "SELECT thresholds FROM control_plane.evaluations WHERE evaluation_id = %s",
            (claim.evaluation_id,),
        ).fetchone()
        self.assertEqual(row[0]["early_stopping"], loaded.request_dict())

    def _accept_and_promote(self, claim):
        self.repository.complete_verdict(
            claim.evaluation_id,
            result=terminal_result(claim.request, accepted=True),
            now=NOW + timedelta(minutes=1),
            publish_non_winning=False,
        )
        promotion = self.connection.execute(
            "SELECT promotion_id FROM control_plane.model_promotions WHERE evaluation_id = %s",
            (claim.evaluation_id,),
        ).fetchone()[0]
        self.connection.execute(
            """
            UPDATE control_plane.model_promotions
               SET state = 'promoted', public_verified_at = %s, private_deleted_at = %s,
                   promoted_at = %s
             WHERE promotion_id = %s
            """,
            (NOW, NOW, NOW, promotion),
        )
        return promotion


    def test_next_submission_waits_until_winner_is_crowned(self) -> None:
        first = self.repository.claim_next(now=NOW, policy=policy())
        self.assertIsNone(self.repository.claim_next(now=NOW, policy=policy()))
        promotion = self._accept_and_promote(first)
        self.assertIsNone(self.repository.claim_next(now=NOW, policy=policy()))
        self.connection.execute(
            "UPDATE control_plane.uploads SET state = 'promoted' WHERE upload_id = %s",
            (first.upload_id,),
        )
        self.assertIsNone(self.repository.claim_next(now=NOW, policy=policy()))
        reign = self.repository.crown_promoted_winner(
            str(promotion),
            now=NOW,
            crowned_finalized_block=110,
            policy_hotkeys=["hotkey-3"],
            target_hotkeys=["hotkey-3"],
            target_uids=[3],
            normalized_weights=[1.0],
        )
        second = self.repository.claim_next(now=NOW, policy=policy())
        self.assertEqual(second.upload_id, str(self.uploads[1]))
        self.assertEqual(second.claimed_king_reign_id, reign)

    def test_claim_order_and_competition_lock_are_deterministic(self) -> None:
        standby = self._repository(self.second_connection, "validator-b")
        with self.assertRaises(SchedulerLockUnavailable):
            standby.acquire_lock()

        observed = []
        sampling_identities = []
        for _ in range(3):
            claim = self.repository.claim_next(now=NOW, policy=policy())
            self.assertIsNotNone(claim)
            observed.append(claim.upload_id)
            sampling_identities.append(
                (
                    claim.request["miner"]["hotkey"],
                    claim.request["sampling"]["block_hash"],
                )
            )
            self.repository.fail_attempt(
                claim.evaluation_id,
                now=NOW,
                failure_class="deterministic_submission",
                public_error_code="fixture_invalid",
                retry=False,
            )
        self.assertEqual(
            observed, [str(self.uploads[2]), str(self.uploads[1]), str(self.uploads[0])]
        )
        self.assertEqual(
            sampling_identities,
            [
                ("hotkey-3", f"0x{101:064x}"),
                ("hotkey-2", f"0x{101:064x}"),
                ("hotkey-1", f"0x{103:064x}"),
            ],
        )

    def test_dedicated_validator_role_can_claim_and_commit_its_owned_state(self) -> None:
        self.repository.release_lock()
        self.second_connection.execute("SET ROLE teutonic_validator")
        restricted = self._repository(self.second_connection, "validator-restricted")
        restricted.acquire_lock()
        try:
            claim = restricted.claim_next(now=NOW, policy=policy())
            self.assertIsNotNone(claim)
            restricted.fail_attempt(
                claim.evaluation_id,
                now=NOW,
                failure_class="deterministic_submission",
                public_error_code="fixture_invalid",
                retry=False,
            )
            self.assertEqual(
                self.connection.execute(
                    "SELECT state FROM control_plane.uploads WHERE upload_id = %s",
                    (claim.upload_id,),
                ).fetchone()[0],
                "invalid_evaluation_input",
            )
        finally:
            restricted.release_lock()
            self.second_connection.execute("RESET ROLE")
            self.repository.acquire_lock()

    def test_expired_lease_is_adopted_and_retry_creates_append_only_attempt(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        self.repository.start_evaluating(
            claim.evaluation_id,
            evaluator_job_id=claim.eval_id,
            now=NOW,
            lease=timedelta(seconds=1),
        )
        self.repository.release_lock()
        standby = self._repository(self.second_connection, "validator-b")
        standby.acquire_lock()
        try:
            candidates = standby.recovery_candidates(now=NOW + timedelta(seconds=2))
            self.assertEqual([item.evaluation_id for item in candidates], [claim.evaluation_id])
            standby.adopt(claim.evaluation_id, now=NOW + timedelta(seconds=2), lease=policy().lease)
            standby.fail_attempt(
                claim.evaluation_id,
                now=NOW + timedelta(seconds=2),
                failure_class="transient_infrastructure",
                public_error_code="worker_lost",
                retry=True,
                retry_delay=timedelta(seconds=5),
            )
            self.assertIsNone(standby.claim_next(now=NOW + timedelta(seconds=6), policy=policy()))
            retried = standby.claim_next(now=NOW + timedelta(seconds=7), policy=policy())
            self.assertEqual(retried.upload_id, claim.upload_id)
            self.assertEqual(retried.attempt_number, 2)
            self.assertEqual(retried.request["miner"]["hotkey"], claim.request["miner"]["hotkey"])
            self.assertEqual(
                retried.request["sampling"]["block_hash"],
                claim.request["sampling"]["block_hash"],
            )
            attempts = self.connection.execute(
                """
                SELECT attempt_number, state
                  FROM control_plane.evaluations
                 WHERE upload_id = %s
                 ORDER BY attempt_number
                """,
                (claim.upload_id,),
            ).fetchall()
            self.assertEqual(attempts, [(1, "retryable_failure"), (2, "claimed")])
        finally:
            standby.release_lock()
            self.repository.acquire_lock()

    def test_restart_recovers_terminal_evaluator_result_before_retrying(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        self.repository.start_evaluating(
            claim.evaluation_id,
            evaluator_job_id=claim.eval_id,
            now=NOW,
            lease=timedelta(seconds=1),
        )
        self.repository.release_lock()
        restarted = self._repository(self.second_connection, "validator-after-restart")
        restarted.acquire_lock()
        try:
            scheduler = ValidatorScheduler(
                restarted,
                RecoveryEvaluator(claim.request),
                policy=policy(),
                preflight=lambda _request: None,
                clock=lambda: NOW + timedelta(seconds=2),
            )
            self.assertEqual(asyncio.run(scheduler.reconcile()), 1)
            row = self.connection.execute(
                "SELECT state, verdict FROM control_plane.evaluations WHERE evaluation_id = %s",
                (claim.evaluation_id,),
            ).fetchone()
            self.assertEqual(row, ("completed", "rejected"))
            self.assertEqual(
                self.connection.execute(
                    "SELECT count(*) FROM control_plane.evaluations WHERE upload_id = %s",
                    (claim.upload_id,),
                ).fetchone()[0],
                1,
            )
        finally:
            restarted.release_lock()
            self.repository.acquire_lock()

    def test_restart_retries_only_after_matching_evaluator_job_is_lost(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        self.repository.start_evaluating(
            claim.evaluation_id,
            evaluator_job_id=claim.eval_id,
            now=NOW,
            lease=timedelta(seconds=1),
        )
        self.repository.release_lock()
        restarted = self._repository(self.second_connection, "validator-after-restart")
        restarted.acquire_lock()
        try:
            scheduler = ValidatorScheduler(
                restarted,
                RecoveryEvaluator(claim.request, lost=True),
                policy=policy(),
                preflight=lambda _request: None,
                clock=lambda: NOW + timedelta(seconds=2),
            )
            self.assertEqual(asyncio.run(scheduler.reconcile()), 1)
            row = self.connection.execute(
                """
                SELECT state, failure_class, public_error_code
                  FROM control_plane.evaluations
                 WHERE evaluation_id = %s
                """,
                (claim.evaluation_id,),
            ).fetchone()
            self.assertEqual(
                row, ("retryable_failure", "transient_infrastructure", "evaluator_job_lost")
            )
        finally:
            restarted.release_lock()
            self.repository.acquire_lock()

    def test_unavailable_evaluator_does_not_claim_queued_upload(self) -> None:
        async def preflight(_request):
            return None

        scheduler = ValidatorScheduler(
            self.repository,
            UnavailableEvaluator(),
            policy=policy(),
            preflight=preflight,
            clock=lambda: NOW,
        )
        self.assertFalse(asyncio.run(scheduler.run_once()))
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.evaluations"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.uploads WHERE state = 'ready_for_evaluation'"
            ).fetchone()[0],
            3,
        )

    def test_reuse_limit_failure_persists_stable_public_reason(self) -> None:
        async def preflight(_request):
            return None

        scheduler = ValidatorScheduler(
            self.repository,
            ReuseLimitEvaluator(lambda request: terminal_result(request, accepted=False)),
            policy=policy(),
            preflight=preflight,
            clock=lambda: NOW,
        )
        self.assertTrue(asyncio.run(scheduler.run_once()))
        row = self.connection.execute(
            """
            SELECT e.evaluation_id, e.state, e.failure_class, e.public_error_code,
                   e.private_diagnostic_reference, u.state, u.failure_code
              FROM control_plane.evaluations e
              JOIN control_plane.uploads u USING (upload_id)
             WHERE e.attempt_number = 1
             ORDER BY e.created_at
             LIMIT 1
            """
        ).fetchone()
        self.assertEqual(row[1:4], ("terminal_failure", "policy", "safetensors_reuse_limit"))
        self.assertEqual(row[4], f"diagnostic:{row[0]}:RuntimeError")
        self.assertEqual(row[5:], ("evaluation_failed", "safetensors_reuse_limit"))
        self.assertNotIn("private-digest", str(row))

    def test_legacy_duplicate_failure_persists_model_copy(self) -> None:
        async def preflight(_request):
            return None

        scheduler = ValidatorScheduler(
            self.repository,
            LegacyDuplicateEvaluator(lambda request: terminal_result(request, accepted=False)),
            policy=policy(),
            preflight=preflight,
            clock=lambda: NOW,
        )
        self.assertTrue(asyncio.run(scheduler.run_once()))
        row = self.connection.execute(
            """
            SELECT e.state, e.failure_class, e.public_error_code,
                   e.verdict_summary, u.state, u.failure_code
              FROM control_plane.evaluations e
              JOIN control_plane.uploads u USING (upload_id)
             WHERE e.attempt_number = 1
             ORDER BY e.created_at
             LIMIT 1
            """
        ).fetchone()
        self.assertEqual(row[:3], ("terminal_failure", "policy", "model_copy"))
        self.assertEqual(row[3], {"error_code": "model_copy"})
        self.assertEqual(row[4:], ("evaluation_failed", "model_copy"))

    def test_dispatch_outage_reuses_attempt_without_spending_retry_budget(self) -> None:
        current = [NOW]
        evaluator = DispatchOutageEvaluator(
            lambda request: terminal_result(request, accepted=False)
        )

        async def preflight(_request):
            return None

        scheduler = ValidatorScheduler(
            self.repository,
            evaluator,
            policy=policy(max_attempts=1),
            preflight=preflight,
            clock=lambda: current[0],
        )
        self.assertTrue(asyncio.run(scheduler.run_once()))
        deferred = self.connection.execute(
            """
            SELECT evaluation_id, attempt_number, state, started_at
              FROM control_plane.evaluations
            """
        ).fetchone()
        self.assertEqual(deferred[1:], (1, "retryable_failure", None))

        current[0] += timedelta(seconds=6)
        self.assertTrue(asyncio.run(scheduler.run_once()))
        completed = self.connection.execute(
            """
            SELECT evaluation_id, attempt_number, state, started_at
              FROM control_plane.evaluations
            """
        ).fetchall()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][0], deferred[0])
        self.assertEqual(completed[0][1:3], (1, "completed"))
        self.assertIsNotNone(completed[0][3])
        self.assertEqual(
            (
                evaluator.dispatches[0]["evaluation_id"],
                evaluator.dispatches[0]["attempt_number"],
            ),
            (
                evaluator.dispatches[1]["evaluation_id"],
                evaluator.dispatches[1]["attempt_number"],
            ),
        )

    def test_restart_redispatches_unaccepted_claim_with_same_attempt(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy(max_attempts=1))
        self.repository.release_lock()
        restarted = self._repository(self.second_connection, "validator-after-dispatch-crash")
        restarted.acquire_lock()
        try:
            evaluator = FakeEvaluator(
                lambda request: terminal_result(request, accepted=False)
            )

            async def preflight(_request):
                return None

            scheduler = ValidatorScheduler(
                restarted,
                evaluator,
                policy=policy(max_attempts=1),
                preflight=preflight,
                clock=lambda: NOW + timedelta(minutes=3),
            )
            self.assertEqual(asyncio.run(scheduler.reconcile()), 1)
            attempts = self.connection.execute(
                """
                SELECT evaluation_id::text, attempt_number, state
                  FROM control_plane.evaluations
                 WHERE upload_id = %s
                """,
                (claim.upload_id,),
            ).fetchall()
            self.assertEqual(attempts, [(claim.evaluation_id, 1, "completed")])
            self.assertEqual(
                (
                    evaluator.started[0]["evaluation_id"],
                    evaluator.started[0]["attempt_number"],
                ),
                (claim.evaluation_id, claim.attempt_number),
            )
        finally:
            restarted.release_lock()
            self.repository.acquire_lock()

    def test_restart_waits_for_unavailable_gpu_without_consuming_accepted_attempt(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        self.repository.start_evaluating(
            claim.evaluation_id,
            evaluator_job_id=claim.eval_id,
            now=NOW,
            lease=timedelta(seconds=1),
        )
        self.repository.release_lock()
        restarted = self._repository(self.second_connection, "validator-with-gpu-outage")
        restarted.acquire_lock()
        try:
            async def preflight(_request):
                return None

            scheduler = ValidatorScheduler(
                restarted,
                UnavailableEvaluator(),
                policy=policy(),
                preflight=preflight,
                clock=lambda: NOW + timedelta(seconds=2),
            )
            self.assertEqual(asyncio.run(scheduler.reconcile()), 0)
            attempts = self.connection.execute(
                """
                SELECT attempt_number, state, public_error_code
                  FROM control_plane.evaluations
                 WHERE upload_id = %s
                """,
                (claim.upload_id,),
            ).fetchall()
            self.assertEqual(attempts, [(1, "evaluating", None)])
        finally:
            restarted.release_lock()
            self.repository.acquire_lock()

    def test_v2_dispatch_persists_bounded_progress_and_one_accepted_verdict(self) -> None:
        evaluator = FakeEvaluator(lambda request: terminal_result(request, accepted=True))

        async def preflight(_request):
            return None

        scheduler = ValidatorScheduler(
            self.repository, evaluator, policy=policy(), preflight=preflight, clock=lambda: NOW
        )
        self.assertTrue(asyncio.run(scheduler.run_once()))
        row = self.connection.execute(
            """
            SELECT e.state, e.verdict, e.progress_summary, u.state,
                   e.verdict_summary ->> 'delta'
              FROM control_plane.evaluations e
              JOIN control_plane.uploads u ON u.upload_id = e.upload_id
            """
        ).fetchone()
        self.assertEqual(row[0:2], ("completed", "accepted"))
        self.assertNotIn("private_worker_hostname", row[2])
        self.assertEqual(
            row[2],
            {
                "phase": "scoring",
                "completed_sequences": 16,
                "requested_sequences": 32,
                "percent": 50.0,
                "provisional_mu_hat": 0.003,
                "provisional_lcb": 0.002,
                "provisional_n_sequences": 16,
                "provisional_n_bootstrap": 128,
            },
        )
        self.assertEqual(row[3], "accepted_pending_promotion")
        self.assertEqual(row[4], "0.0015")
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.model_promotions"
            ).fetchone()[0],
            1,
        )

    def test_seeded_queue_runs_across_logical_days_without_mutable_r2_state(self) -> None:
        current = NOW
        evaluator = FakeEvaluator(lambda request: terminal_result(request, accepted=False))

        async def preflight(request):
            self.assertEqual(request["challenger"]["kind"], "r2-prefix")
            self.assertIn("models/registrations/", request["challenger"]["prefix"])
            self.assertNotIn("/sha256/", request["challenger"]["prefix"])
            return None

        for _ in range(3):
            scheduler = ValidatorScheduler(
                self.repository,
                evaluator,
                policy=policy(),
                preflight=preflight,
                clock=lambda current=current: current,
            )
            self.assertTrue(asyncio.run(scheduler.run_once()))
            current += timedelta(days=1)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.uploads WHERE state = 'rejected'"
            ).fetchone()[0],
            3,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.uploads WHERE state = 'ready_for_evaluation'"
            ).fetchone()[0],
            0,
        )

    def test_duplicate_verdict_and_crown_create_one_logical_outcome(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        result = terminal_result(claim.request, accepted=True)
        self.repository.complete_verdict(
            claim.evaluation_id, result=result, now=NOW, publish_non_winning=False
        )
        self.repository.complete_verdict(
            claim.evaluation_id, result=result, now=NOW, publish_non_winning=False
        )
        promotion = self.connection.execute(
            "SELECT promotion_id FROM control_plane.model_promotions"
        ).fetchone()[0]
        self.connection.execute(
            """
            UPDATE control_plane.model_promotions
               SET state = 'promoted', public_verified_at = %s,
                   private_deleted_at = %s, promoted_at = %s
            """,
            (NOW, NOW, NOW),
        )
        first = self.repository.crown_promoted_winner(
            str(promotion),
            now=NOW,
            crowned_finalized_block=110,
            policy_hotkeys=["hotkey-3", "genesis-hotkey"],
            target_hotkeys=["hotkey-3", "genesis-hotkey"],
            target_uids=[1, 2],
            normalized_weights=[0.75, 0.25],
        )
        second = self.repository.crown_promoted_winner(
            str(promotion),
            now=NOW,
            crowned_finalized_block=110,
            policy_hotkeys=["hotkey-3", "genesis-hotkey"],
            target_hotkeys=["hotkey-3", "genesis-hotkey"],
            target_uids=[1, 2],
            normalized_weights=[0.75, 0.25],
        )
        self.assertEqual(second, first)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.king_reigns"
            ).fetchone()[0],
            2,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.weight_publications"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.notification_outbox"
            ).fetchone()[0],
            1,
        )

    def test_promoted_winner_precedes_recent_kings_in_frozen_weight_policy(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        promotion = self._accept_and_promote(claim)
        self.assertEqual(
            self.repository.promotion_weight_hotkeys(str(promotion), limit=5),
            (claim.request["miner"]["hotkey"], "genesis-hotkey"),
        )

    def test_promoted_winner_displaces_one_initial_weight_hotkey(self) -> None:
        starter_hotkeys = [f"starter-hotkey-{uid}" for uid in (110, 115, 143, 224, 226)]
        self.connection.execute(
            """
            INSERT INTO control_plane.weight_publications (
                competition_id, source_reign_id, policy_version, policy_hotkeys,
                target_hotkeys, target_uids, normalized_weights, payload_sha256,
                mapping_finalized_block, idempotency_key, state
            ) VALUES (%s, %s, 'genesis-equal-v1', %s, %s,
                      ARRAY[110, 115, 143, 224, 226],
                      ARRAY[0.2, 0.2, 0.2, 0.2, 0.2], %s, 103,
                      'initial-weight-policy-test', 'requested')
            """,
            (
                self.competition_id,
                self.genesis_reign_id,
                starter_hotkeys,
                starter_hotkeys,
                "7" * 64,
            ),
        )
        claim = self.repository.claim_next(now=NOW, policy=policy())
        promotion = self._accept_and_promote(claim)
        self.assertEqual(
            self.repository.promotion_weight_hotkeys(str(promotion), limit=5),
            (claim.request["miner"]["hotkey"], *starter_hotkeys[:4]),
        )

    def test_current_reign_weight_plan_revises_after_winner_uid_remap(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        promotion = self._accept_and_promote(claim)
        reign = self.repository.crown_promoted_winner(
            str(promotion),
            now=NOW,
            crowned_finalized_block=110,
            policy_hotkeys=[claim.request["miner"]["hotkey"], "genesis-hotkey"],
            target_hotkeys=["burn:uid:0"],
            target_uids=[0],
            normalized_weights=[1.0],
        )
        self.assertIsNotNone(reign)
        current = self.repository.current_weight_policy()
        self.assertEqual(current["payload_revision"], 1)
        self.assertTrue(
            self.repository.refresh_current_weight_plan(
                publication_id=current["publication_id"],
                expected_revision=1,
                mapping_finalized_block=120,
                target_hotkeys=[claim.request["miner"]["hotkey"]],
                target_uids=[17],
                normalized_weights=[1.0],
                now=NOW + timedelta(minutes=2),
            )
        )
        row = self.connection.execute(
            """
            SELECT policy_hotkeys, target_hotkeys, target_uids, payload_revision,
                   mapping_finalized_block, state
              FROM control_plane.weight_publications
            """
        ).fetchone()
        self.assertEqual(
            row,
            (
                [claim.request["miner"]["hotkey"], "genesis-hotkey"],
                [claim.request["miner"]["hotkey"]],
                [17],
                2,
                120,
                "requested",
            ),
        )

    def test_validator_service_heartbeat_is_upserted(self) -> None:
        self.repository.heartbeat_service(
            now=NOW,
            phase="evaluating",
            software_version="release-1",
        )
        self.repository.heartbeat_service(
            now=NOW + timedelta(seconds=30),
            phase="idle",
            software_version="release-1",
        )
        row = self.connection.execute(
            """
            SELECT state, phase, heartbeat_at
              FROM control_plane.service_instances
             WHERE service_name = 'validator' AND instance_id = 'validator-a'
            """
        ).fetchone()
        self.assertEqual(row, ("active", "idle", NOW + timedelta(seconds=30)))

    def test_rejected_verdict_is_terminal_without_publication_by_default(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        self.repository.complete_verdict(
            claim.evaluation_id,
            result=terminal_result(claim.request, accepted=False),
            now=NOW,
            publish_non_winning=False,
        )
        state = self.connection.execute(
            "SELECT state FROM control_plane.uploads WHERE upload_id = %s", (claim.upload_id,)
        ).fetchone()[0]
        self.assertEqual(state, "rejected")
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.model_promotions"
            ).fetchone()[0],
            0,
        )

    def test_stale_king_cannot_be_replaced_by_old_baseline_verdict(self) -> None:
        claim = self.repository.claim_next(now=NOW, policy=policy())
        promotion = self._accept_and_promote(claim)
        self.connection.execute(
            """
            UPDATE control_plane.king_reigns
               SET ended_at = %s, replacement_reason = 'test_interleaving'
             WHERE reign_id = %s
            """,
            (NOW, self.genesis_reign_id),
        )
        replacement = self.connection.execute(
            """
            INSERT INTO control_plane.king_reigns (
                competition_id, reign_number, model_digest, public_bucket, public_prefix,
                hotkey, uid, previous_reign_id, crowned_at, crowned_finalized_block
            ) VALUES (%s, 1, %s, 'public-models', %s, 'other-hotkey', 99, %s, %s, 109)
            RETURNING reign_id
            """,
            (
                self.competition_id,
                "e" * 64,
                f"models/sha256/{'e' * 64}/",
                self.genesis_reign_id,
                NOW,
            ),
        ).fetchone()[0]
        self.connection.execute(
            """
            UPDATE control_plane.competitions
               SET current_reign_id = %s, next_reign_number = 2
             WHERE competition_id = %s
            """,
            (replacement, self.competition_id),
        )
        crowned = self.repository.crown_promoted_winner(
            str(promotion),
            now=NOW,
            crowned_finalized_block=110,
            policy_hotkeys=["hotkey-3"],
            target_hotkeys=["hotkey-3"],
            target_uids=[1],
            normalized_weights=[1.0],
        )
        self.assertIsNone(crowned)
        upload_state = self.connection.execute(
            "SELECT state, failure_code FROM control_plane.uploads WHERE upload_id = %s",
            (claim.upload_id,),
        ).fetchone()
        self.assertEqual(upload_state, ("retry_pending", "stale_king"))
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.weight_publications"
            ).fetchone()[0],
            0,
        )

        retry = self.repository.claim_next(now=NOW, policy=policy())
        self.assertEqual(retry.upload_id, claim.upload_id)
        self.assertEqual(retry.request["challenger"]["bucket"], "public-models")
        self.assertEqual(
            retry.request["challenger"]["prefix"],
            f"models/sha256/{retry.request['challenger']['expected_digest']}/",
        )
        self.repository.complete_verdict(
            retry.evaluation_id,
            result=terminal_result(retry.request, accepted=True),
            now=NOW,
            publish_non_winning=False,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM control_plane.uploads WHERE upload_id = %s",
                (claim.upload_id,),
            ).fetchone()[0],
            "promoted",
        )
        recrowned = self.repository.crown_promoted_winner(
            str(promotion),
            now=NOW,
            crowned_finalized_block=111,
            policy_hotkeys=["hotkey-3"],
            target_hotkeys=["hotkey-3"],
            target_uids=[3],
            normalized_weights=[1.0],
        )
