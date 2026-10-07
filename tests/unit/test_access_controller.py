from __future__ import annotations

import hashlib
import io
import unittest
from dataclasses import replace
from datetime import datetime, timezone

from nacl.signing import SigningKey

from teutonic.access import (
    AccessControllerJobRunner,
    GenesisContractMismatch,
    MailboxCipher,
    MailboxStore,
    Manifest,
    ManifestFile,
    MetagraphSnapshot,
    R2UploadController,
    ReadySignal,
    UidAssignment,
    UploadQuotaExceeded,
    encode_ss58_public_key,
    ready_signal_payload,
)
from teutonic.access.crypto import SecretCipher, encode_signature
from teutonic.storage.artifacts import ArtifactIntegrityError, model_digest_from_inventory


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.multipart: list[dict[str, str]] = []
        self.get_keys: list[str] = []
        self.head_keys: list[str] = []

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.read()
        self.metadata[(Bucket, Key)] = dict(kwargs.get("Metadata") or {})
        return {"ETag": self._etag(self.objects[(Bucket, Key)])}

    def get_object(self, *, Bucket, Key):
        self.get_keys.append(Key)
        value = self.objects[(Bucket, Key)]
        return {
            "Body": io.BytesIO(value),
            "ETag": self._etag(value),
            "Metadata": self.metadata.get((Bucket, Key), {}),
        }

    def head_object(self, *, Bucket, Key):
        self.head_keys.append(Key)
        value = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(value),
            "ETag": self._etag(value),
            "Metadata": self.metadata.get((Bucket, Key), {}),
        }

    def list_objects_v2(self, *, Bucket, Prefix, **kwargs):
        contents = [
            {"Key": key, "Size": len(value), "ETag": self._etag(value)}
            for (bucket, key), value in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def list_multipart_uploads(self, *, Bucket, Prefix, **kwargs):
        uploads = [item for item in self.multipart if item["Key"].startswith(Prefix)]
        return {"Uploads": uploads, "IsTruncated": False}

    def list_parts(self, *, Bucket, Key, UploadId, **kwargs):
        upload = next(
            item
            for item in self.multipart
            if item["Key"] == Key and item["UploadId"] == UploadId
        )
        return {"Parts": upload.get("Parts", []), "IsTruncated": False}

    def copy_object(self, *, Bucket, Key, CopySource, **kwargs):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.metadata[(Bucket, Key)] = dict(kwargs.get("Metadata") or {})
        return {"CopyObjectResult": {"ETag": self._etag(self.objects[(Bucket, Key)])}}

    def delete_objects(self, *, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop((Bucket, item["Key"]), None)
            self.metadata.pop((Bucket, item["Key"]), None)
        return {"Deleted": Delete["Objects"]}

    @staticmethod
    def _etag(value: bytes) -> str:
        return f'"{hashlib.md5(value, usedforsecurity=False).hexdigest()}"'


class FakeRevocationRepository:
    def __init__(self, registration: str, keys: tuple[str, ...]) -> None:
        self.registration = registration
        self.keys = keys
        self.state = "active"

    def registration_context(self, registration: str):
        if registration != self.registration:
            raise AssertionError("unexpected registration")
        return {
            "token_state": self.state,
            "cloudflare_token_id": "parent-token",
        }

    def record_parent_token_revoked(self, registration: str, *, now) -> None:
        if registration != self.registration:
            raise AssertionError("unexpected registration")
        self.state = "revoked"

    def mailbox_object_keys(self, registration: str) -> tuple[str, ...]:
        if registration != self.registration:
            raise AssertionError("unexpected registration")
        return self.keys

    def revoked_mailbox_object_keys(self) -> tuple[str, ...]:
        return self.keys if self.state == "revoked" else ()


class FakeTokenGateway:
    def __init__(self) -> None:
        self.revoked: list[str] = []

    def revoke_parent_token(self, token_id: str) -> None:
        self.revoked.append(token_id)


def signed_manifest(key: SigningKey, registration: str, files: dict[str, bytes]) -> Manifest:
    inventory = tuple(
        ManifestFile(path=path, size=len(value), sha256=hashlib.sha256(value).hexdigest())
        for path, value in sorted(files.items())
    )
    manifest = Manifest(
        registration_id=registration,
        hotkey=encode_ss58_public_key(bytes(key.verify_key)),
        model_name="phase4/test-model",
        files=inventory,
        model_digest=model_digest_from_inventory(
            [(item.path, item.size, item.sha256) for item in inventory]
        ),
        signature="unsigned",
    )
    return replace(manifest, signature=encode_signature(key.sign(manifest.signing_payload()).signature))


class AccessControllerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.miner = SigningKey.generate()
        self.registration = "a" * 64
        self.hotkey = encode_ss58_public_key(bytes(self.miner.verify_key))

    def test_finalized_snapshot_and_ready_signal_are_deterministic(self) -> None:
        snapshot = MetagraphSnapshot(
            netuid=3,
            chain_generation="genesis-1",
            finalized_block=100,
            finalized_block_hash="0x100",
            assignments=(UidAssignment(42, self.hotkey, self.hotkey),),
            observed_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
        )
        replay = replace(snapshot)
        self.assertEqual(snapshot.checksum, replay.checksum)
        signal = ReadySignal.parse(
            f"r2ready:v1|{self.registration}|{'b' * 64}",
            signalling_hotkey=self.hotkey,
            block_number=100,
            extrinsic_index=2,
            event_index=3,
        )
        self.assertEqual(signal.manifest_sha256, "b" * 64)
        compact = ready_signal_payload(self.registration, "b" * 64)
        self.assertLessEqual(len(compact.encode()), 128)
        compact_signal = ReadySignal.parse(
            compact,
            signalling_hotkey=self.hotkey,
            block_number=100,
            extrinsic_index=2,
            event_index=3,
        )
        self.assertEqual(compact_signal.registration_id, self.registration)
        self.assertEqual(compact_signal.manifest_sha256, "b" * 64)
        self.assertEqual(compact_signal.raw_payload, compact)
        with self.assertRaises(ValueError):
            ReadySignal.parse(
                "r2ready:v2|bad",
                signalling_hotkey=self.hotkey,
                block_number=100,
                extrinsic_index=2,
                event_index=3,
            )

    def test_mailbox_is_signed_and_encrypted_to_the_miner(self) -> None:
        validator = SigningKey.generate()
        cipher = MailboxCipher(validator)
        ciphertext, envelope = cipher.create_ciphertext(
            hotkey=self.hotkey,
            netuid=3,
            uid=42,
            registration_id=self.registration,
            generation=1,
            endpoint="https://account.r2.cloudflarestorage.com",
            private_model_bucket="private",
            allowed_prefix=f"models/registrations/{self.registration}/",
            access_key_id="access",
            secret_access_key="secret",
            session_token="session",
            expires_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
            chain_generation="genesis-1",
            registration_block=100,
        )
        decrypted = cipher.decrypt_for_test(ciphertext, self.miner)
        self.assertEqual(decrypted, envelope)
        secret_cipher = SecretCipher(b"x" * 32)
        self.assertEqual(secret_cipher.decrypt(secret_cipher.encrypt("parent-secret")), "parent-secret")

    def test_model_cleanup_raises_when_r2_reports_delete_errors(self) -> None:
        s3 = FakeS3()
        prefix = f"models/registrations/{self.registration}/"
        s3.put_object(Bucket="private", Key=f"{prefix}model.bin", Body=b"model")

        def fail_delete(**_kwargs):
            return {"Errors": [{"Key": f"{prefix}model.bin", "Code": "InternalError"}]}

        s3.delete_objects = fail_delete
        controller = R2UploadController(
            s3,
            private_model_bucket="private",
            genesis_contract_files={"config.json": "b" * 64},
        )

        with self.assertRaisesRegex(RuntimeError, "R2 failed to delete"):
            controller.cleanup_model_prefix(prefix)

    def test_mailbox_deletion_is_exact_and_idempotent(self) -> None:
        s3 = FakeS3()
        store = MailboxStore(s3, bucket="dashboard")
        first = f"mailbox/v1/{self.registration}/generations/{1:020d}.bin"
        second = f"mailbox/v1/{self.registration}/generations/{2:020d}.bin"
        s3.put_object(Bucket="dashboard", Key=first, Body=b"first")
        s3.put_object(Bucket="dashboard", Key=second, Body=b"second")
        latest = f"mailbox/v1/{self.registration}/latest.bin"
        store.publish_latest(self.registration, b"second")
        s3.put_object(Bucket="dashboard", Key="dashboard.json", Body=b"public")

        self.assertEqual(store.delete((first, second)), 3)
        self.assertEqual(store.delete((first, second)), 3)
        self.assertNotIn(("dashboard", latest), s3.objects)
        self.assertNotIn(("dashboard", first), s3.objects)
        self.assertNotIn(("dashboard", second), s3.objects)
        self.assertEqual(s3.objects[("dashboard", "dashboard.json")], b"public")
        with self.assertRaisesRegex(Exception, "non-mailbox"):
            store.delete(("dashboard.json",))

    def test_revocation_removes_mailbox_and_replay_repairs_it(self) -> None:
        key = f"mailbox/v1/{self.registration}/generations/{1:020d}.bin"
        s3 = FakeS3()
        s3.put_object(Bucket="dashboard", Key=key, Body=b"credential")
        repository = FakeRevocationRepository(self.registration, (key,))
        gateway = FakeTokenGateway()
        runner = AccessControllerJobRunner(
            repository,
            token_gateway=gateway,
            upload_controller=None,
            mailbox_store=MailboxStore(s3, bucket="dashboard"),
            secret_cipher=None,
            mailbox_cipher=None,
            account_id="account",
            r2_endpoint="https://account.r2.cloudflarestorage.com",
            private_model_bucket="private",
            instance_id="test",
        )

        first = runner._revoke_parent(
            {"registration_id": self.registration},
            now=datetime(2026, 8, 20, tzinfo=timezone.utc),
        )
        self.assertTrue(first["revoked"])
        self.assertEqual(first["mailbox_credentials_removed"], 2)
        self.assertEqual(gateway.revoked, ["parent-token"])
        self.assertNotIn(("dashboard", key), s3.objects)

        s3.put_object(Bucket="dashboard", Key=key, Body=b"stale")
        replay = runner._revoke_parent(
            {"registration_id": self.registration},
            now=datetime(2026, 8, 20, tzinfo=timezone.utc),
        )
        self.assertTrue(replay["replayed"])
        self.assertEqual(gateway.revoked, ["parent-token"])
        self.assertNotIn(("dashboard", key), s3.objects)

    def test_r2_upload_is_revoked_verified_in_place_and_reverified(self) -> None:
        files = {"config.json": b"{}", "weights/model.bin": b"phase-four"}
        manifest = signed_manifest(self.miner, self.registration, files)
        prefix = f"models/registrations/{self.registration}/"
        s3 = FakeS3()
        for path, value in files.items():
            s3.put_object(
                Bucket="private",
                Key=f"{prefix}{path}",
                Body=value,
                Metadata={"sha256": hashlib.sha256(value).hexdigest()},
            )
        s3.put_object(
            Bucket="private",
            Key=f"{prefix}manifest.json",
            Body=manifest.as_bytes(),
            Metadata={"sha256": manifest.manifest_sha256},
        )
        controller = R2UploadController(
            s3,
            private_model_bucket="private",
            genesis_contract_files={
                "config.json": hashlib.sha256(files["config.json"]).hexdigest()
            },
        )
        verified = controller.verify_manifest(
            model_prefix=prefix,
            registration_id=self.registration,
            hotkey=self.hotkey,
            expected_manifest_sha256=manifest.manifest_sha256,
        )
        immutable = controller.create_immutable_snapshot(
            model_prefix=prefix, verified=verified
        )
        self.assertEqual(immutable.prefix, prefix)
        self.assertEqual(immutable.bucket, "private")
        self.assertEqual(set(immutable.etags), set(files))
        self.assertEqual(immutable.manifest_size, len(manifest.as_bytes()))
        self.assertEqual(
            s3.get_keys,
            [f"{prefix}manifest.json", f"{prefix}config.json"],
        )
        self.assertEqual(s3.head_keys, [f"{prefix}weights/model.bin"])
        for path, value in files.items():
            self.assertEqual(s3.objects[("private", f"{prefix}{path}")], value)

    def test_verifier_rejects_undeclared_objects_and_multipart_state(self) -> None:
        files = {"model.bin": b"phase-four"}
        manifest = signed_manifest(self.miner, self.registration, files)
        prefix = f"models/registrations/{self.registration}/"
        s3 = FakeS3()
        s3.put_object(
            Bucket="private",
            Key=f"{prefix}model.bin",
            Body=files["model.bin"],
            Metadata={"sha256": hashlib.sha256(files["model.bin"]).hexdigest()},
        )
        s3.put_object(
            Bucket="private",
            Key=f"{prefix}manifest.json",
            Body=manifest.as_bytes(),
            Metadata={"sha256": manifest.manifest_sha256},
        )
        s3.put_object(Bucket="private", Key=f"{prefix}undeclared.bin", Body=b"bad")
        controller = R2UploadController(
            s3,
            private_model_bucket="private",
            genesis_contract_files={
                "model.bin": hashlib.sha256(files["model.bin"]).hexdigest()
            },
        )
        with self.assertRaises(ArtifactIntegrityError):
            controller.verify_manifest(
                model_prefix=prefix,
                registration_id=self.registration,
                hotkey=self.hotkey,
                expected_manifest_sha256=manifest.manifest_sha256,
            )
        del s3.objects[("private", f"{prefix}undeclared.bin")]
        s3.multipart.append({"Key": f"{prefix}unfinished.bin", "UploadId": "upload-1"})
        with self.assertRaises(ArtifactIntegrityError):
            controller.verify_manifest(
                model_prefix=prefix,
                registration_id=self.registration,
                hotkey=self.hotkey,
                expected_manifest_sha256=manifest.manifest_sha256,
            )

    def test_upload_usage_counts_completed_objects_and_multipart_parts(self) -> None:
        other_registration = "b" * 64
        prefix = f"models/registrations/{self.registration}/"
        s3 = FakeS3()
        s3.put_object(Bucket="private", Key=f"{prefix}completed.bin", Body=b"123456")
        s3.put_object(
            Bucket="private",
            Key=f"models/registrations/{other_registration}/ignored.bin",
            Body=b"ignored",
        )
        s3.multipart.append(
            {
                "Key": f"{prefix}uploading.bin",
                "UploadId": "upload-1",
                "Parts": [{"PartNumber": 1, "Size": 4}, {"PartNumber": 2, "Size": 3}],
            }
        )
        controller = R2UploadController(
            s3,
            private_model_bucket="private",
            genesis_contract_files={"config.json": hashlib.sha256(b"{}").hexdigest()},
        )
        self.assertEqual(
            controller.registration_upload_usage(
                (self.registration, other_registration)
            ),
            {self.registration: 13, other_registration: 7},
        )

    def test_verifier_rejects_upload_above_hard_byte_limit_before_reading_objects(
        self,
    ) -> None:
        files = {"config.json": b"{}", "model.bin": b"weights"}
        manifest = signed_manifest(self.miner, self.registration, files)
        prefix = f"models/registrations/{self.registration}/"
        s3 = FakeS3()
        for path, value in files.items():
            s3.put_object(
                Bucket="private",
                Key=f"{prefix}{path}",
                Body=value,
                Metadata={"sha256": hashlib.sha256(value).hexdigest()},
            )
        s3.put_object(
            Bucket="private",
            Key=f"{prefix}manifest.json",
            Body=manifest.as_bytes(),
            Metadata={"sha256": manifest.manifest_sha256},
        )
        total_bytes = sum(len(value) for value in files.values()) + len(
            manifest.as_bytes()
        )
        controller = R2UploadController(
            s3,
            private_model_bucket="private",
            genesis_contract_files={
                "config.json": hashlib.sha256(files["config.json"]).hexdigest()
            },
            max_upload_bytes=total_bytes - 1,
        )
        with self.assertRaises(UploadQuotaExceeded):
            controller.verify_manifest(
                model_prefix=prefix,
                registration_id=self.registration,
                hotkey=self.hotkey,
                expected_manifest_sha256=manifest.manifest_sha256,
            )
        self.assertEqual(s3.get_keys, [])

    def test_verifier_rejects_missing_sha256_metadata_before_evaluation(self) -> None:
        files = {"model.bin": b"phase-four"}
        manifest = signed_manifest(self.miner, self.registration, files)
        prefix = f"models/registrations/{self.registration}/"
        s3 = FakeS3()
        s3.put_object(Bucket="private", Key=f"{prefix}model.bin", Body=files["model.bin"])
        s3.put_object(
            Bucket="private",
            Key=f"{prefix}manifest.json",
            Body=manifest.as_bytes(),
            Metadata={"sha256": manifest.manifest_sha256},
        )
        with self.assertRaisesRegex(ArtifactIntegrityError, "metadata"):
            R2UploadController(
                s3,
                private_model_bucket="private",
                genesis_contract_files={
                    "model.bin": hashlib.sha256(files["model.bin"]).hexdigest()
                },
            ).verify_manifest(
                model_prefix=prefix,
                registration_id=self.registration,
                hotkey=self.hotkey,
                expected_manifest_sha256=manifest.manifest_sha256,
            )

    def test_verifier_rejects_missing_or_changed_genesis_files_before_object_reads(self) -> None:
        genesis = b"immutable-genesis-config"
        files = {"config.json": b"miner-changed-config", "model.bin": b"weights"}
        manifest = signed_manifest(self.miner, self.registration, files)
        prefix = f"models/registrations/{self.registration}/"
        s3 = FakeS3()
        s3.put_object(
            Bucket="private",
            Key=f"{prefix}manifest.json",
            Body=manifest.as_bytes(),
            Metadata={"sha256": manifest.manifest_sha256},
        )
        controller = R2UploadController(
            s3,
            private_model_bucket="private",
            genesis_contract_files={"config.json": hashlib.sha256(genesis).hexdigest()},
        )
        with self.assertRaisesRegex(GenesisContractMismatch, "changed=\\['config.json'\\]"):
            controller.verify_manifest(
                model_prefix=prefix,
                registration_id=self.registration,
                hotkey=self.hotkey,
                expected_manifest_sha256=manifest.manifest_sha256,
            )

        missing_manifest = signed_manifest(
            self.miner, self.registration, {"model.bin": b"weights"}
        )
        s3.put_object(
            Bucket="private",
            Key=f"{prefix}manifest.json",
            Body=missing_manifest.as_bytes(),
            Metadata={"sha256": missing_manifest.manifest_sha256},
        )
        with self.assertRaisesRegex(GenesisContractMismatch, "missing=\\['config.json'\\]"):
            controller.verify_manifest(
                model_prefix=prefix,
                registration_id=self.registration,
                hotkey=self.hotkey,
                expected_manifest_sha256=missing_manifest.manifest_sha256,
            )

    def test_verifier_hashes_genesis_contract_bytes_instead_of_trusting_metadata(self) -> None:
        expected = b"immutable-genesis-config"
        changed = b"x" * len(expected)
        expected_digest = hashlib.sha256(expected).hexdigest()
        inventory = (
            ManifestFile("config.json", len(expected), expected_digest),
            ManifestFile(
                "model.bin", len(b"weights"), hashlib.sha256(b"weights").hexdigest()
            ),
        )
        manifest = Manifest(
            registration_id=self.registration,
            hotkey=self.hotkey,
            model_name="forged-contract-test",
            files=inventory,
            model_digest=model_digest_from_inventory(
                [(item.path, item.size, item.sha256) for item in inventory]
            ),
            signature="unsigned",
        )
        manifest = replace(
            manifest,
            signature=encode_signature(
                self.miner.sign(manifest.signing_payload()).signature
            ),
        )
        prefix = f"models/registrations/{self.registration}/"
        s3 = FakeS3()
        for path, value, declared_digest in (
            ("config.json", changed, expected_digest),
            ("model.bin", b"weights", hashlib.sha256(b"weights").hexdigest()),
        ):
            s3.put_object(
                Bucket="private",
                Key=f"{prefix}{path}",
                Body=value,
                Metadata={"sha256": declared_digest},
            )
        s3.put_object(
            Bucket="private",
            Key=f"{prefix}manifest.json",
            Body=manifest.as_bytes(),
            Metadata={"sha256": manifest.manifest_sha256},
        )

        with self.assertRaisesRegex(
            GenesisContractMismatch, "genesis contract bytes differ"
        ):
            R2UploadController(
                s3,
                private_model_bucket="private",
                genesis_contract_files={"config.json": expected_digest},
            ).verify_manifest(
                model_prefix=prefix,
                registration_id=self.registration,
                hotkey=self.hotkey,
                expected_manifest_sha256=manifest.manifest_sha256,
            )


if __name__ == "__main__":
    unittest.main()
