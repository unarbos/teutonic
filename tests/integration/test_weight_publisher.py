from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone

try:
    import psycopg
except ImportError:
    psycopg = None

from teutonic.weights import ChainObservation, SubmissionReceipt, weight_payload_digest
from teutonic.weights.repository import WeightPublicationRepository
from teutonic.weights.service import WeightPublisher


DATABASE_URL = os.environ.get("TEUTONIC_TEST_DATABASE_URL")
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


class FakeChain:
    network = "fake-test"
    signer_hotkey = "fake-validator-hotkey"
    mortality_period = 128

    def __init__(self, block: int = 1_000) -> None:
        self.block = block
        self.last_update = 0
        self.last_payload = None
        self.submit_calls = 0
        self.inclusion_delay = 1

    def current_block(self) -> int:
        return self.block

    def advance(self, blocks: int) -> None:
        self.block += blocks

    def observe(self, plan):
        return ChainObservation(
            current_block=self.block,
            finalized_block=self.block,
            uid_count=16,
            validator_uid=0,
            last_update=self.last_update,
            weights_match=self.last_payload == plan.payload_sha256,
        )

    def submit(self, plan):
        self.submit_calls += 1
        self.block += self.inclusion_delay
        self.last_update = self.block
        self.last_payload = plan.payload_sha256
        identity = f"0xfake{self.submit_calls}"
        return SubmissionReceipt(
            True,
            extrinsic_id=identity,
            included_block=self.block,
            included_block_hash=f"0xblock{self.block}",
            finalized_block=self.block,
            finalized_block_hash=f"0xblock{self.block}",
        )

    def find_finalized(self, extrinsic_id: str, *, start_block: int):
        return None


@unittest.skipUnless(DATABASE_URL and psycopg, "PostgreSQL integration database required")
class WeightPublisherIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)
        if cls.connection.execute("SELECT current_database()").fetchone()[0] != "teutonic_test":
            raise RuntimeError("refusing Phase 7 tests outside teutonic_test")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def setUp(self) -> None:
        self.connection.execute(
            """
            TRUNCATE TABLE control_plane.weight_submission_attempts,
                           control_plane.weight_publications,
                           control_plane.king_reigns,
                           control_plane.competitions
            RESTART IDENTITY CASCADE
            """
        )
        self.chain = FakeChain()
        self.competition_id, self.reign_id, self.publication_id = self._seed_reign(0)
        self.repository = self._repository("publisher-a")
        self.repository.acquire_lock()

    def tearDown(self) -> None:
        self.repository.release_lock()

    def _repository(self, instance: str) -> WeightPublicationRepository:
        return WeightPublicationRepository(
            self.connection,
            netuid=306,
            chain_generation="test",
            competition="quasar",
            instance_id=instance,
        )

    def _seed_reign(self, number: int, *, current: bool = True, digest_valid: bool = True):
        if number == 0:
            competition = self.connection.execute(
                """
                INSERT INTO control_plane.competitions (netuid, chain_generation, name)
                VALUES (306, 'test', 'quasar') RETURNING competition_id
                """
            ).fetchone()[0]
            previous = None
        else:
            competition = self.competition_id
            previous = self.connection.execute(
                "SELECT current_reign_id FROM control_plane.competitions WHERE competition_id = %s",
                (competition,),
            ).fetchone()[0]
            self.connection.execute(
                """
                UPDATE control_plane.king_reigns
                   SET ended_at = %s, replacement_reason = 'fixture_new_reign'
                 WHERE reign_id = %s
                """,
                (NOW, previous),
            )
        digest = f"{number + 1:x}" * 64
        reign = self.connection.execute(
            """
            INSERT INTO control_plane.king_reigns (
                competition_id, reign_number, model_digest, public_bucket, public_prefix,
                hotkey, uid, previous_reign_id, crowned_at, crowned_finalized_block,
                operator_provenance
            ) VALUES (%s, %s, %s, 'public-models', %s, %s, %s, %s, %s, %s, 'fixture')
            RETURNING reign_id
            """,
            (
                competition,
                number,
                digest,
                f"models/sha256/{digest}/",
                f"king-{number}",
                number + 1,
                previous,
                NOW,
                100 + number,
            ),
        ).fetchone()[0]
        if current:
            self.connection.execute(
                """
                UPDATE control_plane.competitions
                   SET current_reign_id = %s
                 WHERE competition_id = %s
                """,
                (reign, competition),
            )
        payload = weight_payload_digest([number + 1], [1.0])
        publication = self.connection.execute(
            """
            INSERT INTO control_plane.weight_publications (
                competition_id, source_reign_id, policy_version, policy_hotkeys,
                target_hotkeys, target_uids, normalized_weights, payload_sha256,
                mapping_finalized_block, idempotency_key, state
            ) VALUES (
                %s, %s, 'policy-v1', %s, %s, %s, ARRAY[1.0], %s, 100, %s, 'requested'
            )
            RETURNING weight_publication_id
            """,
            (
                competition,
                reign,
                [f"king-{number}"],
                [f"king-{number}"],
                [number + 1],
                payload if digest_valid else "f" * 64,
                f"publish-weights:{reign}",
            ),
        ).fetchone()[0]
        return competition, reign, publication

    def worker(self, **kwargs) -> WeightPublisher:
        return WeightPublisher(
            self.repository,
            self.chain,
            publisher_mode="dry_run",
            retry_base_delay=timedelta(0),
            clock=lambda: NOW,
            **kwargs,
        )

    def test_finality_is_recorded_and_duplicate_poll_is_idle(self) -> None:
        stages = []
        worker = self.worker(after_stage=lambda name, _plan, _receipt: stages.append(name))
        self.assertTrue(worker.run_one(propagate=True))
        self.assertFalse(worker.run_one(propagate=True))
        row = self.connection.execute(
            """
            SELECT p.state, p.next_due_block, a.state, a.extrinsic_id,
                   a.included_block, a.finalized_block
              FROM control_plane.weight_publications p
              JOIN control_plane.weight_submission_attempts a USING (weight_publication_id)
            """
        ).fetchone()
        self.assertEqual(row, ("finalized", 1102, "finalized", "0xfake1", 1001, 1001))
        self.assertEqual(
            stages,
            ["submitting", "external_returned", "submitted", "included", "finalized"],
        )
        self.assertEqual(self.chain.submit_calls, 1)

    def test_current_plan_is_attempted_101_blocks_after_inclusion(self) -> None:
        worker = self.worker()
        worker.run_one(propagate=True)
        self.chain.advance(99)
        self.assertFalse(worker.run_one(propagate=True))
        self.chain.advance(1)
        self.assertFalse(worker.run_one(propagate=True))
        self.chain.advance(1)
        self.assertTrue(worker.run_one(propagate=True))
        attempts = self.connection.execute(
            """
            SELECT sequence, scheduled_block, state
              FROM control_plane.weight_submission_attempts ORDER BY sequence
            """
        ).fetchall()
        self.assertEqual(attempts, [(1, 1000, "finalized"), (2, 1102, "finalized")])
        self.assertEqual(self.chain.submit_calls, 2)

    def test_inclusion_latency_does_not_schedule_inside_chain_rate_limit(self) -> None:
        self.chain.inclusion_delay = 3
        worker = self.worker()
        self.assertTrue(worker.run_one(propagate=True))
        self.chain.advance(100)
        self.assertFalse(worker.run_one(propagate=True))
        self.assertEqual(self.chain.submit_calls, 1)
        self.chain.advance(1)
        self.assertTrue(worker.run_one(propagate=True))
        self.assertEqual(self.chain.submit_calls, 2)

    def test_uid_remap_revision_keeps_prior_attempt_payload_immutable(self) -> None:
        worker = self.worker()
        self.assertTrue(worker.run_one(propagate=True))
        remapped_digest = weight_payload_digest([7], [1.0])
        self.connection.execute(
            """
            UPDATE control_plane.weight_publications
               SET target_uids = ARRAY[7], normalized_weights = ARRAY[1.0],
                   target_hotkeys = ARRAY['king-0'],
                   payload_sha256 = %s, payload_revision = 2,
                   mapping_finalized_block = 1001, state = 'requested',
                   next_due_block = 1001, finalized_block = NULL,
                   submitted_at = NULL, included_at = NULL, finalized_at = NULL
             WHERE weight_publication_id = %s
            """,
            (remapped_digest, self.publication_id),
        )
        self.assertTrue(worker.run_one(propagate=True))
        attempts = self.connection.execute(
            """
            SELECT sequence, payload_revision, target_hotkeys, target_uids, payload_sha256
              FROM control_plane.weight_submission_attempts
             ORDER BY sequence
            """
        ).fetchall()
        self.assertEqual(attempts[0][0:4], (1, 1, ["king-0"], [1]))
        self.assertEqual(attempts[1][0:4], (2, 2, ["king-0"], [7]))
        self.assertNotEqual(attempts[0][4], attempts[1][4])

    def test_restart_after_external_return_does_not_resubmit(self) -> None:
        def crash(name, _plan, _receipt):
            if name == "external_returned":
                raise RuntimeError("fault injection after chain acceptance")

        self.worker(after_stage=crash).run_one()
        state = self.connection.execute(
            "SELECT state FROM control_plane.weight_submission_attempts"
        ).fetchone()[0]
        self.assertEqual(state, "retry_pending")
        self.assertTrue(self.worker().run_one(propagate=True))
        self.assertEqual(self.chain.submit_calls, 1)
        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM control_plane.weight_submission_attempts"
            ).fetchone()[0],
            "finalized",
        )

    def test_restart_after_submitted_ack_does_not_resubmit(self) -> None:
        def crash(name, _plan, _receipt):
            if name == "submitted":
                raise RuntimeError("fault injection after submitted acknowledgement")

        self.worker(after_stage=crash).run_one()
        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM control_plane.weight_submission_attempts"
            ).fetchone()[0],
            "retry_pending",
        )
        self.assertTrue(self.worker().run_one(propagate=True))
        self.assertEqual(self.chain.submit_calls, 1)

    def test_new_reign_preempts_uncertain_old_attempt_immediately(self) -> None:
        def crash(name, _plan, _receipt):
            if name == "external_returned":
                raise RuntimeError("uncertain old attempt")

        self.worker(after_stage=crash).run_one()
        _competition, new_reign, new_publication = self._seed_reign(1)
        self.assertTrue(self.worker().run_one(propagate=True))
        parents = self.connection.execute(
            """
            SELECT weight_publication_id, state, superseded_by
              FROM control_plane.weight_publications
            """
        ).fetchall()
        old = next(row for row in parents if row[0] == self.publication_id)
        new = next(row for row in parents if row[0] == new_publication)
        self.assertEqual(old[1:], ("superseded", new_publication))
        self.assertEqual(new[1], "finalized")
        states = self.connection.execute(
            "SELECT state FROM control_plane.weight_submission_attempts ORDER BY claimed_at"
        ).fetchall()
        self.assertEqual(states, [("superseded",), ("finalized",)])
        self.assertEqual(self.chain.submit_calls, 2)

    def test_invalid_frozen_payload_disables_cadence(self) -> None:
        self.connection.execute(
            "UPDATE control_plane.weight_publications SET payload_sha256 = %s",
            ("f" * 64,),
        )
        self.worker().run_one()
        self.assertEqual(
            self.connection.execute(
                "SELECT state, cadence_enabled FROM control_plane.weight_publications"
            ).fetchone(),
            ("failed", False),
        )
        self.chain.advance(101)
        self.assertFalse(self.worker().run_one())
        self.assertEqual(self.chain.submit_calls, 0)


if __name__ == "__main__":
    unittest.main()
