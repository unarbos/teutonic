from __future__ import annotations

import base64
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

from botocore.exceptions import ClientError

from teutonic.credentials import latest_mailbox_object_key, mailbox_object_key
from teutonic.storage.artifacts import ArtifactIntegrityError
from teutonic.storage.r2_credentials import create_local_temporary_credentials

from .crypto import MailboxCipher, SecretCipher
from .repository import AccessControllerRepository, ControllerInvariantError
from .storage import R2UploadController


log = logging.getLogger("teutonic.access-controller.jobs")


class MailboxStore:
    """Publish encrypted generations and remove them after authority revocation."""

    _KEY = re.compile(r"^mailbox/v1/[0-9a-f]{64}/(?:generations/[0-9]{20}|latest)\.bin$")

    def __init__(self, s3_client: Any, *, bucket: str) -> None:
        if not bucket:
            raise ValueError("mailbox bucket is required")
        self.s3 = s3_client
        self.bucket = bucket

    @staticmethod
    def _digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def publish(self, key: str, ciphertext: bytes) -> None:
        try:
            existing = self.s3.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = exc.response.get("Error", {}).get("Code")
            if status != 404 and code not in {"NoSuchKey", "NotFound"}:
                raise
        else:
            body = existing["Body"]
            try:
                observed = body.read()
            finally:
                body.close()
            if self._digest(observed) != self._digest(ciphertext):
                raise ControllerInvariantError("mailbox generation already contains different bytes")
            return
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=ciphertext,
            ContentType="application/octet-stream",
            Metadata={"sha256": self._digest(ciphertext)},
        )

    def publish_latest(self, registration: str, ciphertext: bytes) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=latest_mailbox_object_key(registration),
            Body=ciphertext,
            ContentType="application/octet-stream",
            CacheControl="no-store",
            Metadata={"sha256": self._digest(ciphertext)},
        )

    def delete(self, keys: Iterable[str]) -> int:
        selected = sorted(set(keys))
        if any(not self._KEY.fullmatch(key) for key in selected):
            raise ControllerInvariantError("refusing to delete a non-mailbox object")
        selected = sorted(
            set(selected) | {latest_mailbox_object_key(key.split("/")[2]) for key in selected}
        )
        for offset in range(0, len(selected), 1000):
            batch = selected[offset : offset + 1000]
            response = self.s3.delete_objects(
                Bucket=self.bucket,
                Delete={
                    "Objects": [{"Key": key} for key in batch],
                    "Quiet": True,
                },
            )
            if response.get("Errors"):
                raise RuntimeError("R2 failed to delete one or more mailbox credentials")
        return len(selected)


class AccessControllerJobRunner:
    def __init__(
        self,
        repository: AccessControllerRepository,
        *,
        token_gateway: Any,
        upload_controller: R2UploadController,
        mailbox_store: MailboxStore,
        secret_cipher: SecretCipher,
        mailbox_cipher: MailboxCipher,
        account_id: str,
        r2_endpoint: str,
        private_model_bucket: str,
        instance_id: str,
        clock: Callable[[], datetime] | None = None,
        lease: timedelta = timedelta(minutes=2),
        retry_delay: timedelta = timedelta(seconds=5),
    ) -> None:
        self.repository = repository
        self.token_gateway = token_gateway
        self.upload_controller = upload_controller
        self.mailbox_store = mailbox_store
        self.secret_cipher = secret_cipher
        self.mailbox_cipher = mailbox_cipher
        self.account_id = account_id
        self.r2_endpoint = r2_endpoint.rstrip("/")
        self.private_model_bucket = private_model_bucket
        self.instance_id = instance_id
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lease = lease
        self.retry_delay = retry_delay

    @staticmethod
    def _timestamp(value: Any) -> datetime:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            raise ControllerInvariantError("job timestamp is not timezone-aware")
        return parsed

    def run_one(self, *, propagate: bool = False) -> bool:
        now = self.clock()
        job = self.repository.claim_job(
            instance_id=self.instance_id, now=now, lease=self.lease
        )
        if job is None:
            return False
        job_id = str(job["controller_job_id"])
        operation = str(job["operation"])
        log.info(
            "controller job claimed operation=%s job=%s registration=%s upload=%s",
            operation,
            job_id,
            job.get("registration_id") or "-",
            job.get("upload_id") or "-",
        )
        self.repository.set_job_running(job_id, instance_id=self.instance_id, now=now)
        try:
            result = self._dispatch(job, now=now)
            self.repository.complete_job(
                job_id,
                instance_id=self.instance_id,
                now=self.clock(),
                result=result,
            )
            log.info("controller job completed operation=%s job=%s", operation, job_id)
        except ArtifactIntegrityError as exc:
            error_code = type(exc).__name__
            if job.get("upload_id"):
                self.repository.mark_upload_verification_failed(
                    str(job["upload_id"]), error_code=error_code, now=self.clock()
                )
            self.repository.fail_job(
                job_id,
                instance_id=self.instance_id,
                now=self.clock(),
                error_code=error_code,
            )
            log.error(
                "controller job failed operation=%s job=%s error=%s retry=false",
                operation,
                job_id,
                error_code,
            )
            if propagate:
                raise
        except Exception as exc:
            error_code = type(exc).__name__
            self.repository.retry_job(
                job_id,
                instance_id=self.instance_id,
                now=self.clock(),
                delay=self.retry_delay,
                error_code=error_code,
            )
            log.warning(
                "controller job retry scheduled operation=%s job=%s error=%s",
                operation,
                job_id,
                error_code,
                exc_info=True,
            )
            if propagate:
                raise
        return True

    def run_until_idle(self, *, maximum_jobs: int = 100, propagate: bool = False) -> int:
        count = 0
        while count < maximum_jobs and self.run_one(propagate=propagate):
            count += 1
        if count == maximum_jobs:
            raise RuntimeError("controller job drain exceeded its safety limit")
        return count

    def reconcile_revoked_mailboxes(self) -> int:
        """Remove mailbox credentials left by revocations completed before deletion existed."""
        return self.mailbox_store.delete(
            self.repository.revoked_mailbox_object_keys()
        )

    def enforce_upload_quotas(self) -> int:
        """Revoke active upload authorities whose R2 usage exceeds the hard limit."""
        authorities = self.repository.active_upload_authorities()
        for registration, prefix in authorities.items():
            if prefix != f"models/registrations/{registration}/":
                raise ControllerInvariantError(
                    "active upload authority has a non-canonical model prefix"
                )
        usage = self.upload_controller.registration_upload_usage(authorities)
        revoked = 0
        for registration, observed_bytes in sorted(usage.items()):
            if observed_bytes <= self.upload_controller.max_upload_bytes:
                continue
            requested = self.repository.request_upload_quota_revocation(
                registration,
                observed_bytes=observed_bytes,
                limit_bytes=self.upload_controller.max_upload_bytes,
                now=self.clock(),
            )
            if requested:
                revoked += 1
                log.warning(
                    "upload quota exceeded registration=%s bytes=%d limit=%d "
                    "action=revoke_abort_cleanup",
                    registration,
                    observed_bytes,
                    self.upload_controller.max_upload_bytes,
                )
        return revoked

    def schedule_reuse_limit_cleanups(self) -> int:
        """Schedule deletion of model prefixes that exhausted evaluation reuse."""
        return self.repository.enqueue_reuse_limit_cleanups()

    def _dispatch(self, job: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        operation = job["operation"]
        if operation == "create_parent_token":
            return self._create_parent(job, now=now)
        if operation == "publish_credentials":
            return self._publish_credentials(job, now=now)
        if operation == "revoke_parent_token":
            return self._revoke_parent(job, now=now)
        if operation == "verify_upload":
            return self._verify_upload(job, now=now)
        if operation == "create_immutable_snapshot":
            return self._create_snapshot(job, now=now)
        if operation == "abort_multipart":
            context = self.repository.registration_context(str(job["registration_id"]))
            return {
                "aborted": self.upload_controller.abort_multipart_uploads(
                    context["model_prefix"]
                )
            }
        if operation == "cleanup_upload":
            context = self.repository.registration_context(str(job["registration_id"]))
            return {
                "deleted": self.upload_controller.cleanup_model_prefix(
                    context["model_prefix"]
                )
            }
        raise ControllerInvariantError(f"unknown controller job operation {operation!r}")

    def _create_parent(self, job: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        registration = str(job["registration_id"])
        context = self.repository.registration_context(registration)
        if context["token_state"] == "active":
            return {"cloudflare_token_id": context["cloudflare_token_id"], "replayed": True}
        parent = self.token_gateway.create_parent_token(context["token_name"])
        self.repository.record_parent_token_active(
            registration,
            cloudflare_token_id=parent.token_id,
            access_key_id=parent.access_key_id,
            encrypted_secret=self.secret_cipher.encrypt(parent.secret_access_key),
            now=now,
            credential_ttl=timedelta(days=7),
            private_model_bucket=self.private_model_bucket,
        )
        return {"cloudflare_token_id": parent.token_id}

    def _publish_credentials(self, job: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        registration = str(job["registration_id"])
        context = self.repository.registration_context(registration)
        if context["state"] != "active" or context["token_state"] != "active":
            # Ready or deregistration can arrive while a renewal is queued.
            # Finish obsolete work without publishing or retrying revoked access.
            return {"skipped": "upload_authority_inactive"}
        payload = dict(job["payload"])
        generation = int(payload["generation"])
        issued_at = self._timestamp(payload["issued_at"])
        expires_at = self._timestamp(payload["expires_at"])
        if expires_at <= now:
            # A long outage may leave a stale publication job. Complete it so the
            # renewal scan can allocate a fresh generation, never reuse its payload.
            return {"expired": True, "generation": generation}
        ttl = int((expires_at - issued_at).total_seconds())
        parent_secret = self.secret_cipher.decrypt(bytes(context["encrypted_secret"]))
        credentials = create_local_temporary_credentials(
            endpoint=self.r2_endpoint,
            account_id=self.account_id,
            parent_access_key_id=context["access_key_id"],
            parent_secret_access_key=parent_secret,
            bucket=payload["bucket"],
            prefix=context["model_prefix"],
            ttl_seconds=ttl,
            issued_at_unix=int(issued_at.timestamp()),
        )

        checkpoint = dict(job.get("result") or {})
        encoded = checkpoint.get("ciphertext")
        if encoded:
            ciphertext = base64.b64decode(encoded, validate=True)
        else:
            ciphertext, _ = self.mailbox_cipher.create_ciphertext(
                hotkey=context["hotkey"],
                netuid=context["netuid"],
                uid=context["uid"],
                registration_id=registration,
                generation=generation,
                endpoint=self.r2_endpoint,
                private_model_bucket=payload["bucket"],
                allowed_prefix=context["model_prefix"],
                access_key_id=credentials.access_key_id,
                secret_access_key=credentials.secret_access_key,
                session_token=credentials.session_token,
                expires_at=expires_at,
                chain_generation=context["chain_generation"],
                registration_block=context["first_seen_finalized_block"],
            )
            checkpoint = {
                "ciphertext": base64.b64encode(ciphertext).decode(),
                "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
            }
            self.repository.checkpoint_job_result(
                str(job["controller_job_id"]),
                instance_id=self.instance_id,
                result=checkpoint,
                now=now,
            )

        digest = hashlib.sha256(ciphertext).hexdigest()
        if checkpoint.get("ciphertext_sha256") != digest:
            raise ControllerInvariantError("credential job ciphertext checkpoint is corrupt")
        self.repository.record_credential_publishing(
            registration,
            generation=generation,
            issued_at=issued_at,
            expires_at=expires_at,
            bucket=payload["bucket"],
            ciphertext_sha256=digest,
        )
        key = mailbox_object_key(registration, generation)
        self.mailbox_store.publish(key, ciphertext)
        self.mailbox_store.publish_latest(registration, ciphertext)
        self.repository.record_credential_published(registration, generation=generation, now=now)
        return {"ciphertext_sha256": digest, "mailbox_object_key": key}

    def _revoke_parent(self, job: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        registration = str(job["registration_id"])
        context = self.repository.registration_context(registration)
        replayed = context["token_state"] == "revoked"
        if not replayed:
            if context["cloudflare_token_id"]:
                self.token_gateway.revoke_parent_token(context["cloudflare_token_id"])
            self.repository.record_parent_token_revoked(registration, now=now)
        deleted = self.mailbox_store.delete(
            self.repository.mailbox_object_keys(registration)
        )
        return {
            "revoked": not replayed,
            "replayed": replayed,
            "mailbox_credentials_removed": deleted,
        }

    def _verify_upload(self, job: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        upload_id = str(job["upload_id"])
        context = self.repository.upload_context(upload_id)
        if context["token_state"] != "revoked":
            raise ControllerInvariantError(
                "private model upload cannot be verified before access is revoked"
            )
        aborted = self.upload_controller.abort_multipart_uploads(context["model_prefix"])
        verified = self.upload_controller.verify_manifest(
            model_prefix=context["model_prefix"],
            registration_id=str(context["registration_id"]),
            hotkey=context["hotkey"],
            expected_manifest_sha256=str(context["manifest_sha256"]),
        )
        self.repository.record_verified_manifest(
            upload_id,
            verified.manifest,
            source_etags=verified.source_etags,
            now=now,
        )
        return {"model_digest": verified.manifest.model_digest, "aborted": aborted}

    def _create_snapshot(self, job: dict[str, Any], *, now: datetime) -> dict[str, Any]:
        upload_id = str(job["upload_id"])
        context = self.repository.upload_context(upload_id)
        if context["token_state"] != "revoked":
            raise ControllerInvariantError(
                "private model upload cannot be finalized before access is revoked"
            )
        verified = self.upload_controller.verify_manifest(
            model_prefix=context["model_prefix"],
            registration_id=str(context["registration_id"]),
            hotkey=context["hotkey"],
            expected_manifest_sha256=str(context["manifest_sha256"]),
        )
        if verified.manifest.model_digest != str(context["model_digest"]):
            raise ArtifactIntegrityError("model digest changed after initial verification")
        immutable = self.upload_controller.create_immutable_snapshot(
            model_prefix=context["model_prefix"], verified=verified
        )
        self.repository.commit_immutable_snapshot(
            upload_id,
            bucket=immutable.bucket,
            prefix=immutable.prefix,
            version=immutable.version,
            manifest_size=immutable.manifest_size,
            immutable_etags=immutable.etags,
            now=now,
        )
        return {"bucket": immutable.bucket, "prefix": immutable.prefix}
