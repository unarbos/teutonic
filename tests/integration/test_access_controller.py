from __future__ import annotations

import hashlib
import io
import os
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

try:
    import psycopg
except ImportError:
    psycopg = None

from botocore.exceptions import ClientError
from nacl.signing import SigningKey

from teutonic.access import (
    AccessControllerJobRunner,
    AccessControllerRepository,
    MailboxCipher,
    MailboxStore,
    Manifest,
    ManifestFile,
    MetagraphSnapshot,
    R2UploadController,
    ReadySignal,
    UidAssignment,
    encode_ss58_public_key,
)
from teutonic.access.cloudflare import ParentToken
from teutonic.access.crypto import SecretCipher, encode_signature
from teutonic.access.repository import ControllerInvariantError, ControllerLockUnavailable
from teutonic.credentials import (
    ActivationSignal,
    activation_message,
    activation_signal_payload,
    mailbox_object_key,
    latest_mailbox_object_key,
)
from teutonic.storage.artifacts import model_digest_from_inventory


DATABASE_URL = os.environ.get("TEUTONIC_TEST_DATABASE_URL")
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.multipart: list[dict[str, str]] = []

    @staticmethod
    def etag(value: bytes) -> str:
        return f'"{hashlib.md5(value, usedforsecurity=False).hexdigest()}"'

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        value = Body if isinstance(Body, bytes) else Body.read()
        self.objects[(Bucket, Key)] = value
        self.metadata[(Bucket, Key)] = dict(kwargs.get("Metadata") or {})
        return {"ETag": self.etag(value)}

    def get_object(self, *, Bucket, Key):
        try:
            value = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            ) from exc
        return {
            "Body": io.BytesIO(value),
            "ETag": self.etag(value),
            "Metadata": self.metadata.get((Bucket, Key), {}),
        }

    def head_object(self, *, Bucket, Key):
        value = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(value),
            "ETag": self.etag(value),
            "Metadata": self.metadata.get((Bucket, Key), {}),
        }

    def list_objects_v2(self, *, Bucket, Prefix, **kwargs):
        return {
            "Contents": [
                {"Key": key, "Size": len(value), "ETag": self.etag(value)}
                for (bucket, key), value in sorted(self.objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ],
            "IsTruncated": False,
        }

    def list_multipart_uploads(self, *, Bucket, Prefix, **kwargs):
        return {
            "Uploads": [item for item in self.multipart if item["Key"].startswith(Prefix)],
            "IsTruncated": False,
        }

    def list_parts(self, *, Bucket, Key, UploadId, **kwargs):
        upload = next(
            item
            for item in self.multipart
            if item["Key"] == Key and item["UploadId"] == UploadId
        )
        return {"Parts": upload.get("Parts", []), "IsTruncated": False}

    def copy_object(self, *, Bucket, Key, CopySource, **kwargs):
        value = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.objects[(Bucket, Key)] = value
        self.metadata[(Bucket, Key)] = dict(kwargs.get("Metadata") or {})
        return {"CopyObjectResult": {"ETag": self.etag(value)}}

    def abort_multipart_upload(self, *, Bucket, Key, UploadId):
        self.multipart = [
            item
            for item in self.multipart
            if (item["Key"], item["UploadId"]) != (Key, UploadId)
        ]

    def delete_objects(self, *, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop((Bucket, item["Key"]), None)
        return {"Deleted": Delete["Objects"]}


class FakeTokenGateway:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.revoked: list[str] = []

    def create_parent_token(self, name: str) -> ParentToken:
        self.created.append(name)
        token_id = f"parent-{len(self.created)}"
        return ParentToken(token_id, token_id, f"secret-{len(self.created)}")

    def revoke_parent_token(self, token_id: str) -> None:
        if token_id not in self.revoked:
            self.revoked.append(token_id)


def signed_manifest(miner: SigningKey, registration: str, files: dict[str, bytes]) -> Manifest:
    inventory = tuple(
        ManifestFile(path=path, size=len(value), sha256=hashlib.sha256(value).hexdigest())
        for path, value in sorted(files.items())
    )
    manifest = Manifest(
        registration_id=registration,
        hotkey=encode_ss58_public_key(bytes(miner.verify_key)),
        model_name="phase4/integration",
        files=inventory,
        model_digest=model_digest_from_inventory(
            [(item.path, item.size, item.sha256) for item in inventory]
        ),
        signature="unsigned",
    )
    return replace(
        manifest,
        signature=encode_signature(miner.sign(manifest.signing_payload()).signature),
    )


@unittest.skipUnless(DATABASE_URL and psycopg, "TEUTONIC_TEST_DATABASE_URL and psycopg required")
class AccessControllerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = psycopg.connect(DATABASE_URL, autocommit=True)
        if cls.connection.execute("SELECT current_database()").fetchone()[0] != "teutonic_test":
            raise RuntimeError("refusing access-controller tests outside teutonic_test")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def setUp(self) -> None:
        self.connection.execute(
            """
            TRUNCATE TABLE
                control_plane.controller_jobs, control_plane.verified_uploads,
                control_plane.upload_files, control_plane.uploads,
                control_plane.credential_generations, control_plane.r2_parent_tokens,
                control_plane.registrations,
                control_plane.metagraph_uid_assignments,
                control_plane.metagraph_snapshots, control_plane.chain_cursors
            RESTART IDENTITY CASCADE
            """
        )
        self.miner = SigningKey.generate()
        self.hotkey = encode_ss58_public_key(bytes(self.miner.verify_key))
        self.repository = AccessControllerRepository(
            self.connection, finalized_start_block=100
        )
        self.repository.acquire_lock()
        self.s3 = FakeS3()
        self.gateway = FakeTokenGateway()
        self.validator = SigningKey.generate()
        self.runner = AccessControllerJobRunner(
            self.repository,
            token_gateway=self.gateway,
            upload_controller=R2UploadController(
                self.s3,
                private_model_bucket="private",
                genesis_contract_files={
                    "config.json": hashlib.sha256(b"{}").hexdigest()
                },
            ),
            mailbox_store=MailboxStore(self.s3, bucket="mailbox"),
            secret_cipher=SecretCipher(b"p" * 32),
            mailbox_cipher=MailboxCipher(self.validator),
            account_id="account",
            r2_endpoint="https://account.r2.cloudflarestorage.com",
            private_model_bucket="private",
            instance_id="phase4-test",
            clock=lambda: NOW,
        )

    def tearDown(self) -> None:
        self.repository.release_lock()

    def snapshot(
        self, block: int, hotkey: str | None, registration_block: int = 100
    ) -> MetagraphSnapshot:
        assignments = (
            (UidAssignment(42, hotkey, hotkey, registration_block),)
            if hotkey is not None
            else (UidAssignment(42, None, None),)
        )
        return MetagraphSnapshot(
            netuid=3,
            chain_generation="phase4-chain",
            finalized_block=block,
            finalized_block_hash=f"0x{block}",
            assignments=assignments,
            observed_at=NOW + timedelta(seconds=block - 100),
        )

    def activate(self) -> str:
        result = self.repository.apply_finalized_snapshot(self.snapshot(100, self.hotkey))
        registration = result.created_registrations[0]
        message = activation_message(
            netuid=3,
            uid=42,
            hotkey=self.hotkey,
            registration_id=registration,
            registration_block=100,
            chain_generation="phase4-chain",
        )
        payload = activation_signal_payload(self.miner.sign(message.encode()).signature)
        self.repository.accept_activation_signal(
            ActivationSignal.parse(
                payload,
                netuid=3,
                chain_generation="phase4-chain",
                signalling_hotkey=self.hotkey,
                block_number=100,
                extrinsic_index=1,
                event_index=2,
            ),
            now=NOW,
        )
        self.runner.run_until_idle(propagate=True)
        return registration

    def renew(self, now):
        return self.repository.schedule_credential_renewals(
            now=now, netuid=3, chain_generation="phase4-chain", private_model_bucket="private",
        )

    def test_automatic_renewal_and_expired_credentials_catchup(self):
        registration = self.activate()
        original = MailboxCipher.decrypt_for_test(
            self.s3.objects[("mailbox", latest_mailbox_object_key(registration))], self.miner,
        )
        self.assertEqual(self.renew(NOW + timedelta(days=5)), 0)
        now = NOW + timedelta(days=6)
        self.assertEqual(self.renew(now), 1)
        self.assertEqual(self.renew(now), 0)
        self.assertEqual(self.repository.request_credential_rotation(
            registration, now=now, credential_ttl=timedelta(days=7), private_model_bucket="private",
        ), 2)
        self.runner.clock = lambda: now
        self.runner.run_until_idle(propagate=True)
        renewed = MailboxCipher.decrypt_for_test(
            self.s3.objects[("mailbox", latest_mailbox_object_key(registration))], self.miner,
        )
        self.assertEqual(renewed['credential_generation'], 2)
        self.assertEqual(renewed['access_key_id'], original['access_key_id'])
        self.assertEqual(renewed['allowed_prefix'], original['allowed_prefix'])
        self.assertNotEqual(renewed['secret_access_key'], original['secret_access_key'])
        self.assertNotEqual(renewed['session_token'], original['session_token'])
        self.assertEqual(datetime.fromisoformat(renewed['expires_at']), now + timedelta(days=7))
        self.assertEqual(self.renew(now), 0)
        now = NOW + timedelta(days=14)
        self.assertEqual(self.renew(now), 1)
        self.runner.run_until_idle(propagate=True)
        self.assertEqual(MailboxCipher.decrypt_for_test(
            self.s3.objects[("mailbox", latest_mailbox_object_key(registration))], self.miner,
        )['credential_generation'], 3)

    def test_expired_publication_job_does_not_block_fresh_generation(self):
        registration = self.activate()
        self.repository.request_credential_rotation(
            registration, now=NOW, credential_ttl=timedelta(days=7), private_model_bucket="private",
        )
        now = NOW + timedelta(days=9)
        self.runner.clock = lambda: now
        self.assertEqual(self.renew(now), 0)
        self.runner.run_until_idle(propagate=True)
        self.assertNotIn(("mailbox", mailbox_object_key(registration, 2)), self.s3.objects)
        self.assertEqual(self.renew(now), 1)
        self.runner.run_until_idle(propagate=True)
        self.assertEqual(MailboxCipher.decrypt_for_test(
            self.s3.objects[("mailbox", latest_mailbox_object_key(registration))], self.miner,
        )['credential_generation'], 3)

    def test_renewal_does_not_restore_quota_revoked_or_deregistered_authority(self):
        registration = self.activate()
        self.assertEqual(self.renew(NOW + timedelta(days=6)), 1)
        self.repository.request_upload_quota_revocation(
            registration, observed_bytes=2, limit_bytes=1, now=NOW,
        )
        self.assertEqual(self.renew(NOW + timedelta(days=10)), 0)
        self.runner.run_until_idle(propagate=True)
        self.assertEqual(self.renew(NOW + timedelta(days=10)), 0)
        self.assertNotIn(("mailbox", latest_mailbox_object_key(registration)), self.s3.objects)
        self.assertNotIn(("mailbox", mailbox_object_key(registration, 2)), self.s3.objects)
        self.assertEqual(self.connection.execute(
            "SELECT count(*) FROM control_plane.controller_jobs WHERE state = 'retry_pending'"
        ).fetchone()[0], 0)
        self.repository.apply_finalized_snapshot(self.snapshot(101, None))
        self.assertEqual(self.renew(NOW + timedelta(days=10)), 0)

    def test_latest_publication_retry_preserves_generation_and_ciphertext(self):
        registration = self.activate()
        now = NOW + timedelta(days=6)
        self.runner.clock = lambda: now
        self.assertEqual(self.renew(now), 1)
        publish_latest = self.runner.mailbox_store.publish_latest
        def fail(*args):
            raise RuntimeError('temporary mailbox failure')
        self.runner.mailbox_store.publish_latest = fail
        self.runner.run_until_idle()
        ciphertext = self.s3.objects[("mailbox", mailbox_object_key(registration, 2))]
        self.assertEqual(self.renew(now), 0)
        self.runner.mailbox_store.publish_latest = publish_latest
        now += timedelta(seconds=10)
        self.runner.run_until_idle(propagate=True)
        self.assertEqual(self.s3.objects[("mailbox", latest_mailbox_object_key(registration))], ciphertext)
        self.assertEqual(self.renew(now), 0)

    def test_full_one_shot_lifecycle_replay_rotation_and_immutable_snapshot(self) -> None:
        partial = replace(self.snapshot(100, self.hotkey), complete=False)
        with self.assertRaises(ControllerInvariantError):
            self.repository.apply_finalized_snapshot(partial)
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM control_plane.registrations"
            ).fetchone()[0],
            0,
        )

        registration = self.activate()
        replay = self.repository.apply_finalized_snapshot(self.snapshot(100, self.hotkey))
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.created_registrations, ())

        generation = self.repository.request_credential_rotation(
            registration,
            now=NOW + timedelta(minutes=1),
            credential_ttl=timedelta(days=7),
            private_model_bucket="private",
        )
        self.assertEqual(generation, 2)
        self.runner.run_until_idle(propagate=True)
        generations = self.connection.execute(
            """
            SELECT generation, state FROM control_plane.credential_generations
             WHERE registration_id = %s ORDER BY generation
            """,
            (registration,),
        ).fetchall()
        self.assertEqual(generations, [(1, "superseded"), (2, "published")])
        mailbox = self.s3.objects[("mailbox", mailbox_object_key(registration, 2))]
        envelope = MailboxCipher.decrypt_for_test(mailbox, self.miner)
        self.assertEqual(envelope["credential_generation"], 2)
        self.assertEqual(envelope["submission_policy"], "one_per_hotkey")

        self.repository.apply_finalized_snapshot(self.snapshot(101, self.hotkey))
        files = {"config.json": b"{}", "model.bin": b"immutable-phase-four"}
        manifest = signed_manifest(self.miner, registration, files)
        prefix = f"models/registrations/{registration}/"
        for path, value in files.items():
            self.s3.put_object(
                Bucket="private",
                Key=f"{prefix}{path}",
                Body=value,
                Metadata={"sha256": hashlib.sha256(value).hexdigest()},
            )
        self.s3.put_object(
            Bucket="private",
            Key=f"{prefix}manifest.json",
            Body=manifest.as_bytes(),
            Metadata={"sha256": manifest.manifest_sha256},
        )
        self.s3.multipart.append(
            {
                "Key": f"{prefix}unfinished.bin",
                "UploadId": "unfinished-after-upload",
                "Parts": [],
            }
        )
        self.repository.request_credential_rotation(
            registration, now=NOW + timedelta(minutes=2),
            credential_ttl=timedelta(days=7), private_model_bucket="private",
        )
        upload_id = self.repository.accept_ready_signal(
            ReadySignal.parse(
                f"r2ready:v1|{registration}|{manifest.manifest_sha256}",
                signalling_hotkey=self.hotkey,
                block_number=101,
                extrinsic_index=0,
                event_index=0,
            ),
            now=NOW + timedelta(minutes=2),
        )
        with self.assertRaises(Exception):
            self.repository.accept_ready_signal(
                ReadySignal.parse(
                    f"r2ready:v1|{registration}|{manifest.manifest_sha256}",
                    signalling_hotkey=self.hotkey,
                    block_number=101,
                    extrinsic_index=1,
                    event_index=0,
                ),
                now=NOW + timedelta(minutes=2),
            )
        self.runner.run_until_idle(propagate=True)

        state = self.connection.execute(
            "SELECT state FROM control_plane.uploads WHERE upload_id = %s", (upload_id,)
        ).fetchone()[0]
        self.assertEqual(state, "ready_for_evaluation")
        token_state = self.connection.execute(
            "SELECT state FROM control_plane.r2_parent_tokens WHERE registration_id = %s",
            (registration,),
        ).fetchone()[0]
        self.assertEqual(token_state, "revoked")
        self.assertEqual(len(self.gateway.revoked), 1)
        self.assertNotIn(("mailbox", mailbox_object_key(registration, 3)), self.s3.objects)
        self.assertEqual(self.connection.execute(
            "SELECT count(*) FROM control_plane.controller_jobs WHERE state = 'retry_pending'"
        ).fetchone()[0], 0)
        self.assertEqual(self.renew(NOW + timedelta(days=10)), 0)
        self.assertNotIn(("mailbox", latest_mailbox_object_key(registration)), self.s3.objects)
        self.assertEqual(self.s3.multipart, [])
        self.assertNotIn(
            ("mailbox", mailbox_object_key(registration, 1)),
            self.s3.objects,
        )
        self.assertNotIn(
            ("mailbox", mailbox_object_key(registration, 2)),
            self.s3.objects,
        )
        self.assertIn(
            ("private", f"{prefix}model.bin"),
            self.s3.objects,
        )
        with self.assertRaises(ControllerInvariantError):
            self.repository.request_credential_rotation(
                registration,
                now=NOW + timedelta(minutes=3),
                credential_ttl=timedelta(days=7),
                private_model_bucket="private",
            )

        self.repository.apply_finalized_snapshot(self.snapshot(102, None))
        replacement = self.repository.apply_finalized_snapshot(
            self.snapshot(103, self.hotkey, registration_block=103)
        )
        replacement_id = replacement.created_registrations[0]
        replacement_state = self.connection.execute(
            "SELECT state FROM control_plane.registrations WHERE registration_id = %s",
            (replacement_id,),
        ).fetchone()[0]
        self.assertEqual(replacement_state, "inactive")
        self.runner.run_until_idle(propagate=True)
        incomplete_jobs = self.connection.execute(
            "SELECT count(*) FROM control_plane.controller_jobs WHERE state <> 'completed'"
        ).fetchone()[0]
        self.assertEqual(incomplete_jobs, 0)

    def test_expired_controller_job_lease_is_recovered(self) -> None:
        registration = self.activate()
        job_ids = self.connection.execute(
            """
            INSERT INTO control_plane.controller_jobs (
                registration_id, operation, idempotency_key, state,
                owner_instance_id, lease_expires_at
            ) VALUES
                (%s, 'cleanup_upload', %s, 'running', 'dead-instance', %s),
                (%s, 'abort_multipart', %s, 'claimed', 'dead-instance', %s)
            RETURNING controller_job_id
            """,
            (
                registration,
                f"restart-cleanup:{registration}",
                NOW - timedelta(seconds=1),
                registration,
                f"restart-abort:{registration}",
                NOW - timedelta(seconds=1),
            ),
        ).fetchall()
        self.assertEqual(self.repository.recover_expired_jobs(now=NOW), 2)
        self.runner.run_until_idle(propagate=True)
        states = self.connection.execute(
            "SELECT state FROM control_plane.controller_jobs WHERE controller_job_id = ANY(%s)",
            ([row[0] for row in job_ids],),
        ).fetchall()
        self.assertEqual(states, [("completed",), ("completed",)])

    def test_upload_quota_revokes_credentials_and_cleans_completed_and_multipart_bytes(
        self,
    ) -> None:
        registration = self.activate()
        prefix = f"models/registrations/{registration}/"
        self.runner.upload_controller.max_upload_bytes = 10
        self.s3.put_object(
            Bucket="private",
            Key=f"{prefix}completed.bin",
            Body=b"123456",
        )
        self.s3.multipart.append(
            {
                "Key": f"{prefix}uploading.bin",
                "UploadId": "multipart-1",
                "Parts": [{"PartNumber": 1, "Size": 4}, {"PartNumber": 2, "Size": 3}],
            }
        )

        self.assertEqual(self.runner.enforce_upload_quotas(), 1)
        token = self.connection.execute(
            """
            SELECT state, revocation_reason
              FROM control_plane.r2_parent_tokens
             WHERE registration_id = %s
            """,
            (registration,),
        ).fetchone()
        self.assertEqual(token, ("pending_revoke", "operator_requested"))
        jobs = self.connection.execute(
            """
            SELECT operation, payload ->> 'reason',
                   (payload ->> 'observed_bytes')::bigint,
                   (payload ->> 'limit_bytes')::bigint
              FROM control_plane.controller_jobs
             WHERE registration_id = %s
               AND idempotency_key LIKE '%%-after-upload-quota:%%'
             ORDER BY created_at
            """,
            (registration,),
        ).fetchall()
        self.assertEqual(
            jobs,
            [
                ("revoke_parent_token", "upload_quota_exceeded", 13, 10),
                ("abort_multipart", "upload_quota_exceeded", 13, 10),
                ("cleanup_upload", "upload_quota_exceeded", 13, 10),
            ],
        )

        self.runner.run_until_idle(propagate=True)
        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM control_plane.r2_parent_tokens WHERE registration_id = %s",
                (registration,),
            ).fetchone()[0],
            "revoked",
        )
        self.assertEqual(self.gateway.revoked, ["parent-1"])
        self.assertFalse(
            any(key.startswith(prefix) for bucket, key in self.s3.objects if bucket == "private")
        )
        self.assertEqual(self.s3.multipart, [])
        self.assertEqual(self.runner.enforce_upload_quotas(), 0)

    def test_reuse_limit_failure_queues_idempotent_private_model_cleanup(self) -> None:
        registration = self.activate()
        prefix = f"models/registrations/{registration}/"
        model_key = f"{prefix}model.safetensors"
        self.s3.put_object(Bucket="private", Key=model_key, Body=b"reused-model")
        manifest_sha256 = "b" * 64
        upload_id = self.connection.execute(
            """
            INSERT INTO control_plane.uploads (
                registration_id, chain_generation, signalling_hotkey,
                ready_payload, ready_finalized_block, ready_extrinsic_index,
                ready_event_index, manifest_sha256, state, failure_code, ready_at
            ) VALUES (
                %s, 'phase4-chain', %s, %s, 101, 0, 0, %s,
                'evaluation_failed', 'evaluation_failed', %s
            )
            RETURNING upload_id
            """,
            (
                registration,
                self.hotkey,
                f"r2ready:v1|{registration}|{manifest_sha256}",
                manifest_sha256,
                NOW,
            ),
        ).fetchone()[0]

        self.assertEqual(self.runner.schedule_reuse_limit_cleanups(), 0)
        self.connection.execute(
            """
            UPDATE control_plane.uploads
               SET failure_code = 'safetensors_reuse_limit'
             WHERE upload_id = %s
            """,
            (upload_id,),
        )
        self.assertEqual(self.runner.schedule_reuse_limit_cleanups(), 1)
        self.assertEqual(self.runner.schedule_reuse_limit_cleanups(), 0)
        cleanup = self.connection.execute(
            """
            SELECT operation, state, payload ->> 'reason'
              FROM control_plane.controller_jobs
             WHERE idempotency_key = %s
            """,
            (f"cleanup-upload-after-reuse-limit:{upload_id}",),
        ).fetchone()
        self.assertEqual(
            cleanup,
            ("cleanup_upload", "pending", "safetensors_reuse_limit"),
        )

        self.runner.run_until_idle(propagate=True)

        self.assertNotIn(("private", model_key), self.s3.objects)
        completed = self.connection.execute(
            """
            SELECT state, (result ->> 'deleted')::integer
              FROM control_plane.controller_jobs
             WHERE idempotency_key = %s
            """,
            (f"cleanup-upload-after-reuse-limit:{upload_id}",),
        ).fetchone()
        self.assertEqual(completed, ("completed", 1))

    def test_advisory_lock_allows_only_one_controller(self) -> None:
        with psycopg.connect(DATABASE_URL, autocommit=True) as second_connection:
            contender = AccessControllerRepository(
                second_connection,
                finalized_start_block=100,
            )
            with self.assertRaises(ControllerLockUnavailable):
                contender.acquire_lock()


if __name__ == "__main__":
    unittest.main()
