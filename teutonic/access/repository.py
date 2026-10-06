from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from teutonic.credentials import ActivationSignal, activation_message
from teutonic.credentials.contracts import mailbox_object_key, registration_id

from .contracts import Manifest, MetagraphSnapshot, ReadySignal
from .crypto import verify_hotkey_signature


CONTROLLER_ADVISORY_LOCK_ID = 8_451_120_703_004_001
EVALUATION_REUSE_LIMIT_ERROR_CODE = "safetensors_reuse_limit"


class ControllerLockUnavailable(RuntimeError):
    pass


class ControllerInvariantError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    snapshot_id: str
    replayed: bool
    created_registrations: tuple[str, ...]
    deactivated_registrations: tuple[str, ...]


class AccessControllerRepository:
    def __init__(
        self,
        connection,
        *,
        finalized_start_block: int,
    ) -> None:
        self.connection = connection
        self.finalized_start_block = finalized_start_block
        self._lock_held = False

    def acquire_lock(self) -> None:
        acquired = self.connection.execute(
            "SELECT pg_try_advisory_lock(%s)", (CONTROLLER_ADVISORY_LOCK_ID,)
        ).fetchone()[0]
        if not acquired:
            raise ControllerLockUnavailable("another access controller holds the advisory lock")
        self._lock_held = True

    def release_lock(self) -> None:
        if self._lock_held:
            self.connection.execute(
                "SELECT pg_advisory_unlock(%s)", (CONTROLLER_ADVISORY_LOCK_ID,)
            )
            self._lock_held = False

    def _require_lock(self) -> None:
        if not self._lock_held:
            raise ControllerLockUnavailable("access controller advisory lock is not held")

    def last_finalized_block(self, *, netuid: int, chain_generation: str) -> int | None:
        """Return the durable scanner cursor while the controller lock is held."""
        self._require_lock()
        row = self.connection.execute(
            """
            SELECT last_finalized_block
              FROM control_plane.chain_cursors
             WHERE netuid = %s AND chain_generation = %s
            """,
            (netuid, chain_generation),
        ).fetchone()
        return None if row is None else int(row[0])

    def _enqueue_job(
        self,
        cursor,
        *,
        operation: str,
        idempotency_key: str,
        registration_id: str | None = None,
        upload_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO control_plane.controller_jobs (
                registration_id, upload_id, operation, idempotency_key,
                state, next_retry_at, payload
            ) VALUES (%s, %s, %s, %s, 'pending', NULL, %s::jsonb)
            ON CONFLICT (idempotency_key) DO NOTHING
            """,
            (registration_id, upload_id, operation, idempotency_key, Jsonb(dict(payload or {}))),
        )

    def apply_finalized_snapshot(self, snapshot: MetagraphSnapshot) -> SnapshotResult:
        self._require_lock()
        if not snapshot.complete:
            raise ControllerInvariantError("partial metagraph snapshots cannot change lifecycle state")
        if snapshot.finalized_block < self.finalized_start_block:
            raise ControllerInvariantError("snapshot predates the configured finalized start block")

        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT snapshot_id, finalized_block_hash, snapshot_checksum
                  FROM control_plane.metagraph_snapshots
                 WHERE netuid = %s AND chain_generation = %s AND finalized_block = %s
                """,
                (snapshot.netuid, snapshot.chain_generation, snapshot.finalized_block),
            )
            existing = cursor.fetchone()
            if existing is not None:
                if (
                    existing["finalized_block_hash"] != snapshot.finalized_block_hash
                    or existing["snapshot_checksum"] != snapshot.checksum
                ):
                    raise ControllerInvariantError("finalized block replay conflicts with stored snapshot")
                return SnapshotResult(str(existing["snapshot_id"]), True, (), ())

            cursor.execute(
                """
                SELECT last_finalized_block
                  FROM control_plane.chain_cursors
                 WHERE netuid = %s AND chain_generation = %s
                 FOR UPDATE
                """,
                (snapshot.netuid, snapshot.chain_generation),
            )
            chain_cursor = cursor.fetchone()
            if chain_cursor and snapshot.finalized_block <= chain_cursor["last_finalized_block"]:
                raise ControllerInvariantError("snapshot is stale relative to the durable chain cursor")

            cursor.execute(
                """
                INSERT INTO control_plane.metagraph_snapshots (
                    netuid, chain_generation, finalized_block, finalized_block_hash,
                    snapshot_checksum, uid_count, is_complete, observed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, true, %s)
                RETURNING snapshot_id
                """,
                (
                    snapshot.netuid,
                    snapshot.chain_generation,
                    snapshot.finalized_block,
                    snapshot.finalized_block_hash,
                    snapshot.checksum,
                    len(snapshot.assignments),
                    snapshot.observed_at,
                ),
            )
            snapshot_id = cursor.fetchone()["snapshot_id"]
            for assignment in snapshot.assignments:
                registration_block = (
                    assignment.registration_block
                    if assignment.registration_block is not None
                    else snapshot.finalized_block
                )
                cursor.execute(
                    """
                    INSERT INTO control_plane.metagraph_uid_assignments
                        (snapshot_id, uid, hotkey, coldkey, registration_block)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        snapshot_id,
                        assignment.uid,
                        assignment.hotkey,
                        assignment.coldkey,
                        registration_block if assignment.hotkey is not None else None,
                    ),
                )

            current = {
                assignment.uid: assignment
                for assignment in snapshot.assignments
                if assignment.hotkey is not None
            }
            cursor.execute(
                """
                SELECT registration_id, uid, hotkey, first_seen_finalized_block
                  FROM control_plane.registrations
                 WHERE netuid = %s AND chain_generation = %s AND state <> 'inactive'
                 FOR UPDATE
                """,
                (snapshot.netuid, snapshot.chain_generation),
            )
            active = cursor.fetchall()
            deactivated: list[str] = []
            for row in active:
                assignment = current.get(row["uid"])
                assignment_block = (
                    assignment.registration_block
                    if assignment is not None and assignment.registration_block is not None
                    else snapshot.finalized_block
                )
                if (
                    assignment is not None
                    and assignment.hotkey == row["hotkey"]
                    and assignment_block == row["first_seen_finalized_block"]
                ):
                    cursor.execute(
                        """
                        UPDATE control_plane.registrations
                           SET last_seen_finalized_block = %s, updated_at = clock_timestamp()
                         WHERE registration_id = %s
                        """,
                        (snapshot.finalized_block, row["registration_id"]),
                    )
                    continue
                reason = "uid_replaced" if assignment is not None else "deregistered"
                cursor.execute(
                    """
                    UPDATE control_plane.registrations
                       SET state = 'inactive', deactivated_finalized_block = %s,
                           deactivation_reason = %s, deactivated_at = clock_timestamp(),
                           updated_at = clock_timestamp()
                     WHERE registration_id = %s
                    """,
                    (snapshot.finalized_block, reason, row["registration_id"]),
                )
                registration = str(row["registration_id"])
                deactivated.append(registration)
                self._enqueue_job(
                    cursor,
                    registration_id=registration,
                    operation="revoke_parent_token",
                    idempotency_key=f"revoke-parent-after-deactivation:{registration}",
                    payload={"reason": "registration_deactivated"},
                )
                self._enqueue_job(
                    cursor,
                    registration_id=registration,
                    operation="abort_multipart",
                    idempotency_key=f"abort-multipart-after-deactivation:{registration}",
                )
                self._enqueue_job(
                    cursor,
                    registration_id=registration,
                    operation="cleanup_upload",
                    idempotency_key=f"cleanup-upload-after-deactivation:{registration}",
                )

            deactivated_set = set(deactivated)
            active_by_uid = {
                row["uid"]: (row["hotkey"], row["first_seen_finalized_block"])
                for row in active
                if str(row["registration_id"]) not in deactivated_set
            }
            created: list[str] = []
            for uid, assignment in sorted(current.items()):
                registration_block = (
                    assignment.registration_block
                    if assignment.registration_block is not None
                    else snapshot.finalized_block
                )
                if active_by_uid.get(uid) == (assignment.hotkey, registration_block):
                    continue
                identifier = registration_id(
                    netuid=snapshot.netuid,
                    uid=uid,
                    hotkey=assignment.hotkey,
                    registration_block=registration_block,
                    chain_generation=snapshot.chain_generation,
                )
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM control_plane.uploads WHERE signalling_hotkey = %s
                    ) AS consumed
                    """,
                    (assignment.hotkey,),
                )
                consumed = cursor.fetchone()["consumed"]
                if consumed:
                    cursor.execute(
                        """
                        INSERT INTO control_plane.registrations (
                            registration_id, netuid, chain_generation, uid, hotkey,
                            first_seen_finalized_block, last_seen_finalized_block,
                            deactivated_finalized_block, model_prefix, state,
                            deactivation_reason, deactivated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'inactive',
                                  'submission_consumed', clock_timestamp())
                        ON CONFLICT (registration_id) DO NOTHING
                        """,
                        (
                            identifier,
                            snapshot.netuid,
                            snapshot.chain_generation,
                            uid,
                            assignment.hotkey,
                            registration_block,
                            registration_block,
                            snapshot.finalized_block,
                            f"models/registrations/{identifier}/",
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO control_plane.registrations (
                            registration_id, netuid, chain_generation, uid, hotkey,
                            first_seen_finalized_block, last_seen_finalized_block,
                            model_prefix, state
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending_activation')
                        ON CONFLICT (registration_id) DO NOTHING
                        """,
                        (
                            identifier,
                            snapshot.netuid,
                            snapshot.chain_generation,
                            uid,
                            assignment.hotkey,
                            registration_block,
                            snapshot.finalized_block,
                            f"models/registrations/{identifier}/",
                        ),
                    )
                if cursor.rowcount == 1:
                    created.append(identifier)

            cursor.execute(
                """
                INSERT INTO control_plane.chain_cursors (
                    netuid, chain_generation, finalized_start_block, last_finalized_block,
                    last_finalized_block_hash, snapshot_checksum, observed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (netuid, chain_generation) DO UPDATE SET
                    last_finalized_block = EXCLUDED.last_finalized_block,
                    last_finalized_block_hash = EXCLUDED.last_finalized_block_hash,
                    snapshot_checksum = EXCLUDED.snapshot_checksum,
                    observed_at = EXCLUDED.observed_at,
                    updated_at = clock_timestamp()
                """,
                (
                    snapshot.netuid,
                    snapshot.chain_generation,
                    self.finalized_start_block,
                    snapshot.finalized_block,
                    snapshot.finalized_block_hash,
                    snapshot.checksum,
                    snapshot.observed_at,
                ),
            )
        return SnapshotResult(str(snapshot_id), False, tuple(created), tuple(deactivated))

    def accept_activation_signal(
        self, signal: ActivationSignal, *, now: datetime
    ) -> str:
        """Verify a finalized Ed25519 activation commitment and enqueue credentials."""
        self._require_lock()
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT *
                  FROM control_plane.registrations
                 WHERE netuid = %s AND chain_generation = %s AND hotkey = %s
                   AND first_seen_finalized_block <= %s
                   AND (deactivated_finalized_block IS NULL OR %s < deactivated_finalized_block)
                 ORDER BY first_seen_finalized_block DESC
                 LIMIT 1
                 FOR UPDATE
                """,
                (
                    signal.netuid,
                    signal.chain_generation,
                    signal.signalling_hotkey,
                    signal.block_number,
                    signal.block_number,
                ),
            )
            row = cursor.fetchone()
            if row is None or row["state"] not in {"pending_activation", "active"}:
                raise ControllerInvariantError("hotkey has no registration eligible for activation")
            cursor.execute(
                """
                SELECT assignment.hotkey, assignment.registration_block
                  FROM control_plane.metagraph_snapshots snapshot
                  JOIN control_plane.metagraph_uid_assignments assignment USING (snapshot_id)
                 WHERE snapshot.netuid = %s AND snapshot.chain_generation = %s
                   AND snapshot.finalized_block = %s AND assignment.uid = %s
                """,
                (row["netuid"], row["chain_generation"], signal.block_number, row["uid"]),
            )
            assignment = cursor.fetchone()
            if (
                assignment is None
                or assignment["hotkey"] != signal.signalling_hotkey
                or assignment["registration_block"] != row["first_seen_finalized_block"]
            ):
                raise ControllerInvariantError(
                    "hotkey did not own the registration at the activation block"
                )
            registration = str(row["registration_id"])
            message = activation_message(
                netuid=row["netuid"],
                uid=row["uid"],
                hotkey=row["hotkey"],
                registration_id=registration,
                registration_block=row["first_seen_finalized_block"],
                chain_generation=row["chain_generation"],
            )
            verify_hotkey_signature(row["hotkey"], message.encode(), signal.signature)
            if row["state"] == "active":
                return registration
            cursor.execute(
                """
                UPDATE control_plane.registrations
                   SET state = 'active', activated_at = %s,
                       activation_finalized_block = %s,
                       activation_extrinsic_index = %s,
                       activation_event_index = %s,
                       activation_payload = %s,
                       updated_at = %s
                 WHERE registration_id = %s
                """,
                (
                    now,
                    signal.block_number,
                    signal.extrinsic_index,
                    signal.event_index,
                    signal.raw_payload,
                    now,
                    registration,
                ),
            )
            token_name = f"teutonic-registration-{registration}"
            cursor.execute(
                """
                INSERT INTO control_plane.r2_parent_tokens (registration_id, token_name, state)
                VALUES (%s, %s, 'pending_create')
                ON CONFLICT (registration_id) DO NOTHING
                """,
                (registration, token_name),
            )
            self._enqueue_job(
                cursor,
                registration_id=registration,
                operation="create_parent_token",
                idempotency_key=f"create-parent:{registration}",
            )
            return registration

    def registration_context(self, registration: str) -> dict[str, Any]:
        self._require_lock()
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT registration.*,
                       token.parent_token_id, token.token_name, token.cloudflare_token_id,
                       token.access_key_id, token.encrypted_secret, token.state AS token_state
                  FROM control_plane.registrations registration
                  LEFT JOIN control_plane.r2_parent_tokens token USING (registration_id)
                 WHERE registration.registration_id = %s
                """,
                (registration,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ControllerInvariantError("registration does not exist")
            return dict(row)

    def record_parent_token_active(
        self,
        registration: str,
        *,
        cloudflare_token_id: str,
        access_key_id: str,
        encrypted_secret: bytes,
        now: datetime,
        credential_ttl: timedelta,
        private_model_bucket: str,
    ) -> None:
        self._require_lock()
        expires_at = now + credential_ttl
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT registration.state, registration.hotkey,
                       token.state AS token_state
                  FROM control_plane.registrations registration
                  JOIN control_plane.r2_parent_tokens token USING (registration_id)
                 WHERE registration.registration_id = %s
                 FOR UPDATE OF registration, token
                """,
                (registration,),
            )
            row = cursor.fetchone()
            if row is None or row["state"] != "active":
                raise ControllerInvariantError("registration is not active")
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM control_plane.uploads WHERE signalling_hotkey = %s)",
                (row["hotkey"],),
            )
            if cursor.fetchone()["exists"]:
                raise ControllerInvariantError("hotkey submission eligibility is permanently consumed")
            if row["token_state"] not in {"pending_create", "creating", "active"}:
                raise ControllerInvariantError("parent token is not creatable")
            cursor.execute(
                """
                UPDATE control_plane.r2_parent_tokens
                   SET cloudflare_token_id = %s, access_key_id = %s,
                       encrypted_secret = %s, state = 'active', activated_at = %s,
                       next_retry_at = NULL, last_error_code = NULL,
                       updated_at = %s
                 WHERE registration_id = %s
                """,
                (
                    cloudflare_token_id,
                    access_key_id,
                    encrypted_secret,
                    now,
                    now,
                    registration,
                ),
            )
            self._enqueue_job(
                cursor,
                registration_id=registration,
                operation="publish_credentials",
                idempotency_key=f"publish-credentials:{registration}:1",
                payload={
                    "generation": 1,
                    "issued_at": now.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "bucket": private_model_bucket,
                },
            )

    def request_credential_rotation(
        self,
        registration: str,
        *,
        now: datetime,
        credential_ttl: timedelta,
        private_model_bucket: str,
    ) -> int:
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT registration.state, registration.hotkey, token.state AS token_state,
                       COALESCE((
                           SELECT MAX(generation.generation)
                             FROM control_plane.credential_generations generation
                            WHERE generation.registration_id = registration.registration_id
                       ), 0) AS generation
                  FROM control_plane.registrations registration
                  JOIN control_plane.r2_parent_tokens token USING (registration_id)
                 WHERE registration.registration_id = %s
                 FOR UPDATE OF registration, token
                """,
                (registration,),
            )
            row = cursor.fetchone()
            if row is None or row["state"] != "active" or row["token_state"] != "active":
                raise ControllerInvariantError("registration upload authority is not active")
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM control_plane.uploads WHERE signalling_hotkey = %s)",
                (row["hotkey"],),
            )
            if cursor.fetchone()["exists"]:
                raise ControllerInvariantError("hotkey submission eligibility is permanently consumed")
            generation = int(row["generation"]) + 1
            self._enqueue_job(
                cursor,
                registration_id=registration,
                operation="publish_credentials",
                idempotency_key=f"publish-credentials:{registration}:{generation}",
                payload={
                    "generation": generation,
                    "issued_at": now.isoformat(),
                    "expires_at": (now + credential_ttl).isoformat(),
                    "bucket": private_model_bucket,
                },
            )
            return generation

    def record_credential_publishing(
        self,
        registration: str,
        *,
        generation: int,
        issued_at: datetime,
        expires_at: datetime,
        bucket: str,
        ciphertext_sha256: str,
    ) -> None:
        self._require_lock()
        key = mailbox_object_key(registration, generation)
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO control_plane.credential_generations (
                    registration_id, generation, issued_at, expires_at, bucket_name,
                    allowed_prefix, allowed_actions, mailbox_object_key,
                    ciphertext_sha256, state
                ) VALUES (%s, %s, %s, %s, %s, %s, NULL, %s, %s, 'publishing')
                ON CONFLICT (registration_id, generation) DO UPDATE SET
                    ciphertext_sha256 = EXCLUDED.ciphertext_sha256
                WHERE credential_generations.state IN ('pending', 'publishing', 'published')
                  AND credential_generations.ciphertext_sha256 = EXCLUDED.ciphertext_sha256
                  AND credential_generations.issued_at = EXCLUDED.issued_at
                  AND credential_generations.expires_at = EXCLUDED.expires_at
                  AND credential_generations.bucket_name = EXCLUDED.bucket_name
                  AND credential_generations.allowed_prefix = EXCLUDED.allowed_prefix
                  AND credential_generations.mailbox_object_key = EXCLUDED.mailbox_object_key
                """,
                (
                    registration,
                    generation,
                    issued_at,
                    expires_at,
                    bucket,
                    f"models/registrations/{registration}/",
                    key,
                    ciphertext_sha256,
                ),
            )
            if cursor.rowcount != 1:
                raise ControllerInvariantError("credential generation conflicts with durable state")

    def record_credential_published(
        self,
        registration: str,
        *,
        generation: int,
        now: datetime,
    ) -> None:
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE control_plane.credential_generations
                   SET state = 'superseded'
                 WHERE registration_id = %s AND generation < %s
                   AND state IN ('pending', 'publishing', 'published')
                """,
                (registration, generation),
            )
            cursor.execute(
                """
                UPDATE control_plane.credential_generations
                   SET state = 'published', published_at = COALESCE(published_at, %s)
                 WHERE registration_id = %s AND generation = %s
                   AND state IN ('publishing', 'published')
                """,
                (now, registration, generation),
            )
            if cursor.rowcount != 1:
                raise ControllerInvariantError("credential generation is not publishable")

    def record_parent_token_revoked(self, registration: str, *, now: datetime) -> None:
        self._require_lock()
        updated = self.connection.execute(
            """
            UPDATE control_plane.r2_parent_tokens
               SET state = 'revoked', encrypted_secret = NULL,
                   secret_key_reference = NULL, revoked_at = COALESCE(revoked_at, %s),
                   next_retry_at = NULL, updated_at = %s
             WHERE registration_id = %s AND state <> 'revoked'
            """,
            (now, now, registration),
        )
        if updated.rowcount not in {0, 1}:
            raise ControllerInvariantError("parent token revocation updated multiple rows")

    def mailbox_object_keys(self, registration: str) -> tuple[str, ...]:
        self._require_lock()
        rows = self.connection.execute(
            """
            SELECT mailbox_object_key
              FROM control_plane.credential_generations
             WHERE registration_id = %s
             ORDER BY generation
            """,
            (registration,),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def revoked_mailbox_object_keys(self) -> tuple[str, ...]:
        self._require_lock()
        rows = self.connection.execute(
            """
            SELECT generation.mailbox_object_key
              FROM control_plane.credential_generations generation
              JOIN control_plane.r2_parent_tokens token USING (registration_id)
             WHERE token.state = 'revoked'
             ORDER BY generation.registration_id, generation.generation
            """
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def active_upload_authorities(self) -> dict[str, str]:
        """Return registration prefixes whose parent upload token is active."""
        self._require_lock()
        rows = self.connection.execute(
            """
            SELECT registration.registration_id, registration.model_prefix
              FROM control_plane.registrations registration
              JOIN control_plane.r2_parent_tokens token USING (registration_id)
             WHERE registration.state = 'active' AND token.state = 'active'
             ORDER BY registration.registration_id
            """
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def request_upload_quota_revocation(
        self,
        registration: str,
        *,
        observed_bytes: int,
        limit_bytes: int,
        now: datetime,
    ) -> bool:
        """Durably revoke and clean an active prefix that exceeded its byte quota."""
        self._require_lock()
        if observed_bytes <= limit_bytes or limit_bytes < 1 or now.tzinfo is None:
            raise ValueError("upload quota revocation requires a valid over-limit observation")
        payload = {
            "reason": "upload_quota_exceeded",
            "observed_bytes": observed_bytes,
            "limit_bytes": limit_bytes,
        }
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            token = cursor.execute(
                """
                SELECT token.parent_token_id, token.state
                  FROM control_plane.r2_parent_tokens token
                  JOIN control_plane.registrations registration USING (registration_id)
                 WHERE token.registration_id = %s AND registration.state = 'active'
                 FOR UPDATE OF token
                """,
                (registration,),
            ).fetchone()
            if token is None or token["state"] != "active":
                return False
            cursor.execute(
                """
                UPDATE control_plane.r2_parent_tokens
                   SET state = 'pending_revoke',
                       revocation_requested_at = COALESCE(revocation_requested_at, %s),
                       revocation_reason = COALESCE(revocation_reason, 'operator_requested'),
                       next_retry_at = %s, updated_at = %s
                 WHERE parent_token_id = %s
                """,
                (now, now, now, token["parent_token_id"]),
            )
            cursor.execute(
                """
                UPDATE control_plane.credential_generations
                   SET state = 'superseded'
                 WHERE registration_id = %s
                   AND state IN ('pending', 'publishing', 'published')
                """,
                (registration,),
            )
            for operation in (
                "revoke_parent_token",
                "abort_multipart",
                "cleanup_upload",
            ):
                self._enqueue_job(
                    cursor,
                    registration_id=registration,
                    operation=operation,
                    idempotency_key=f"{operation}-after-upload-quota:{registration}",
                    payload=payload,
                )
        return True

    def enqueue_reuse_limit_cleanups(self) -> int:
        """Durably schedule R2 cleanup for checkpoints rejected by the reuse policy."""
        self._require_lock()
        rows = self.connection.execute(
            """
            INSERT INTO control_plane.controller_jobs (
                registration_id, upload_id, operation, idempotency_key,
                state, next_retry_at, payload
            )
            SELECT upload.registration_id, upload.upload_id, 'cleanup_upload',
                   'cleanup-upload-after-reuse-limit:' || upload.upload_id::text,
                   'pending', NULL,
                   jsonb_build_object('reason', %s::text)
              FROM control_plane.uploads upload
             WHERE upload.state = 'evaluation_failed'
               AND upload.failure_code = %s
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING controller_job_id
            """,
            (
                EVALUATION_REUSE_LIMIT_ERROR_CODE,
                EVALUATION_REUSE_LIMIT_ERROR_CODE,
            ),
        ).fetchall()
        return len(rows)

    def accept_ready_signal(self, signal: ReadySignal, *, now: datetime) -> str:
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT upload_id, registration_id, signalling_hotkey, ready_payload, manifest_sha256
                  FROM control_plane.uploads
                 WHERE chain_generation = (
                           SELECT chain_generation FROM control_plane.registrations
                            WHERE registration_id = %s
                       )
                   AND ready_finalized_block = %s
                   AND ready_extrinsic_index = %s
                   AND ready_event_index = %s
                """,
                (
                    signal.registration_id,
                    signal.block_number,
                    signal.extrinsic_index,
                    signal.event_index,
                ),
            )
            replay = cursor.fetchone()
            if replay is not None:
                if (
                    str(replay["registration_id"]) != signal.registration_id
                    or replay["signalling_hotkey"] != signal.signalling_hotkey
                    or replay["ready_payload"] != signal.raw_payload
                    or replay["manifest_sha256"] != signal.manifest_sha256
                ):
                    raise ControllerInvariantError("ready signal replay conflicts with stored upload")
                return str(replay["upload_id"])

            cursor.execute(
                "SELECT * FROM control_plane.registrations WHERE registration_id = %s FOR UPDATE",
                (signal.registration_id,),
            )
            registration = cursor.fetchone()
            if registration is None or registration["hotkey"] != signal.signalling_hotkey:
                raise ControllerInvariantError("ready signal hotkey does not own registration")
            if registration["state"] != "active":
                raise ControllerInvariantError("ready signal registration is not active")
            activation_position = (
                registration["activation_finalized_block"],
                registration["activation_extrinsic_index"],
                registration["activation_event_index"],
            )
            if None in activation_position or (
                signal.block_number,
                signal.extrinsic_index,
                signal.event_index,
            ) <= activation_position:
                raise ControllerInvariantError("ready signal does not follow activation")
            if signal.block_number < registration["first_seen_finalized_block"] or (
                registration["deactivated_finalized_block"] is not None
                and signal.block_number >= registration["deactivated_finalized_block"]
            ):
                raise ControllerInvariantError("ready signal is outside registration eligibility")
            cursor.execute(
                """
                SELECT assignment.hotkey, assignment.registration_block
                  FROM control_plane.metagraph_snapshots snapshot
                  JOIN control_plane.metagraph_uid_assignments assignment USING (snapshot_id)
                 WHERE snapshot.netuid = %s AND snapshot.chain_generation = %s
                   AND snapshot.finalized_block = %s AND assignment.uid = %s
                """,
                (
                    registration["netuid"],
                    registration["chain_generation"],
                    signal.block_number,
                    registration["uid"],
                ),
            )
            assignment = cursor.fetchone()
            if (
                assignment is None
                or assignment["hotkey"] != signal.signalling_hotkey
                or assignment["registration_block"]
                != registration["first_seen_finalized_block"]
            ):
                raise ControllerInvariantError("hotkey did not occupy the UID at the ready block")
            cursor.execute(
                """
                INSERT INTO control_plane.uploads (
                    registration_id, chain_generation, signalling_hotkey, ready_payload,
                    ready_finalized_block, ready_extrinsic_index, ready_event_index,
                    manifest_sha256, state, ready_at, competition_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'ready_signaled', %s, %s)
                RETURNING upload_id
                """,
                (
                    signal.registration_id,
                    registration["chain_generation"],
                    signal.signalling_hotkey,
                    signal.raw_payload,
                    signal.block_number,
                    signal.extrinsic_index,
                    signal.event_index,
                    signal.manifest_sha256,
                    now,
                    signal.competition,
                ),
            )
            upload_id = cursor.fetchone()["upload_id"]
            self._enqueue_job(
                cursor,
                registration_id=signal.registration_id,
                upload_id=str(upload_id),
                operation="verify_upload",
                idempotency_key=f"verify-upload:{upload_id}",
            )
            return str(upload_id)

    def record_verified_manifest(
        self,
        upload_id: str,
        manifest: Manifest,
        *,
        source_etags: Mapping[str, str | None],
        now: datetime,
    ) -> None:
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM control_plane.upload_files WHERE upload_id = %s",
                (upload_id,),
            )
            for item in manifest.files:
                cursor.execute(
                    """
                    INSERT INTO control_plane.upload_files (
                        upload_id, object_path, size_bytes, sha256, source_etag, verified_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        upload_id,
                        item.path,
                        item.size,
                        item.sha256,
                        source_etags.get(item.path),
                        now,
                    ),
                )
            cursor.execute(
                """
                UPDATE control_plane.uploads
                   SET manifest_signature_verified = true, model_digest = %s,
                       model_name = %s, object_count = %s, total_size_bytes = %s,
                       state = 'verifying', updated_at = %s
                 WHERE upload_id = %s
                """,
                (
                    manifest.model_digest,
                    manifest.model_name,
                    len(manifest.files),
                    sum(item.size for item in manifest.files),
                    now,
                    upload_id,
                ),
            )
            self._enqueue_job(
                cursor,
                upload_id=upload_id,
                registration_id=manifest.registration_id,
                operation="create_immutable_snapshot",
                idempotency_key=f"create-immutable-snapshot:{upload_id}",
            )

    def upload_context(self, upload_id: str) -> dict[str, Any]:
        self._require_lock()
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT upload.*, registration.hotkey, registration.model_prefix,
                       token.state AS token_state
                  FROM control_plane.uploads upload
                  JOIN control_plane.registrations registration USING (registration_id)
                  JOIN control_plane.r2_parent_tokens token USING (registration_id)
                 WHERE upload.upload_id = %s
                """,
                (upload_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ControllerInvariantError("upload does not exist")
            return dict(row)

    def mark_upload_verification_failed(
        self, upload_id: str, *, error_code: str, now: datetime
    ) -> None:
        self._require_lock()
        self.connection.execute(
            """
            UPDATE control_plane.uploads
               SET state = 'verification_failed', failure_code = %s, updated_at = %s
             WHERE upload_id = %s AND state NOT IN ('ready_for_evaluation', 'evaluated')
            """,
            (error_code, now, upload_id),
        )

    def commit_immutable_snapshot(
        self,
        upload_id: str,
        *,
        bucket: str,
        prefix: str,
        version: str | None,
        manifest_size: int,
        immutable_etags: Mapping[str, str | None],
        now: datetime,
    ) -> None:
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM control_plane.uploads WHERE upload_id = %s FOR UPDATE",
                (upload_id,),
            )
            upload = cursor.fetchone()
            if upload is None or not upload["manifest_signature_verified"]:
                raise ControllerInvariantError("upload manifest has not been verified")
            cursor.execute(
                """
                INSERT INTO control_plane.verified_uploads (
                    upload_id, immutable_bucket, immutable_prefix, immutable_version,
                    model_digest, manifest_sha256, manifest_size_bytes,
                    object_count, total_size_bytes, verified_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (upload_id) DO UPDATE SET
                    immutable_version = EXCLUDED.immutable_version,
                    manifest_size_bytes = EXCLUDED.manifest_size_bytes,
                    verified_at = EXCLUDED.verified_at
                """,
                (
                    upload_id,
                    bucket,
                    prefix,
                    version,
                    upload["model_digest"],
                    upload["manifest_sha256"],
                    manifest_size,
                    upload["object_count"],
                    upload["total_size_bytes"],
                    now,
                ),
            )
            for path, etag in immutable_etags.items():
                cursor.execute(
                    "UPDATE control_plane.upload_files SET immutable_etag = %s WHERE upload_id = %s AND object_path = %s",
                    (etag, upload_id, path),
                )
            cursor.execute(
                """
                UPDATE control_plane.uploads
                   SET state = 'ready_for_evaluation', updated_at = %s
                 WHERE upload_id = %s
                """,
                (now, upload_id),
            )

    def recover_expired_jobs(self, *, now: datetime) -> int:
        self._require_lock()
        result = self.connection.execute(
            """
            UPDATE control_plane.controller_jobs
               SET state = 'retry_pending', owner_instance_id = NULL,
                   lease_expires_at = NULL, next_retry_at = %s,
                   last_error_code = COALESCE(last_error_code, 'lease_expired'),
                   updated_at = %s
             WHERE state IN ('claimed', 'running') AND lease_expires_at <= %s
            """,
            (now, now, now),
        )
        return result.rowcount

    def claim_job(
        self,
        *,
        instance_id: str,
        now: datetime,
        lease: timedelta,
    ) -> dict[str, Any] | None:
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT * FROM control_plane.controller_jobs
                 WHERE state IN ('pending', 'retry_pending')
                   AND (state = 'pending' OR next_retry_at IS NULL OR next_retry_at <= %s)
                   AND (
                       operation NOT IN ('verify_upload', 'create_immutable_snapshot')
                       OR EXISTS (
                           SELECT 1
                             FROM control_plane.r2_parent_tokens token
                            WHERE token.registration_id = controller_jobs.registration_id
                              AND token.state = 'revoked'
                       )
                   )
                 ORDER BY next_retry_at NULLS FIRST, created_at, controller_job_id
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1
                """,
                (now,),
            )
            job = cursor.fetchone()
            if job is None:
                return None
            cursor.execute(
                """
                UPDATE control_plane.controller_jobs
                   SET state = 'claimed', owner_instance_id = %s,
                       lease_expires_at = %s, attempt_count = attempt_count + 1,
                       updated_at = %s
                 WHERE controller_job_id = %s
                RETURNING *
                """,
                (instance_id, now + lease, now, job["controller_job_id"]),
            )
            return dict(cursor.fetchone())

    def set_job_running(self, job_id: str, *, instance_id: str, now: datetime) -> None:
        self._require_lock()
        result = self.connection.execute(
            """
            UPDATE control_plane.controller_jobs SET state = 'running', updated_at = %s
             WHERE controller_job_id = %s AND owner_instance_id = %s AND state = 'claimed'
            """,
            (now, job_id, instance_id),
        )
        if result.rowcount != 1:
            raise ControllerInvariantError("controller job lease is not owned by this instance")

    def checkpoint_job_result(
        self,
        job_id: str,
        *,
        instance_id: str,
        result: Mapping[str, Any],
        now: datetime,
    ) -> None:
        self._require_lock()
        updated = self.connection.execute(
            """
            UPDATE control_plane.controller_jobs SET result = %s::jsonb, updated_at = %s
             WHERE controller_job_id = %s AND owner_instance_id = %s
               AND state IN ('claimed', 'running')
            """,
            (Jsonb(dict(result)), now, job_id, instance_id),
        )
        if updated.rowcount != 1:
            raise ControllerInvariantError("controller job checkpoint lost its lease")

    def complete_job(
        self,
        job_id: str,
        *,
        instance_id: str,
        now: datetime,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        self._require_lock()
        updated = self.connection.execute(
            """
            UPDATE control_plane.controller_jobs
               SET state = 'completed', result = %s::jsonb, completed_at = %s,
                   lease_expires_at = NULL, owner_instance_id = NULL,
                   next_retry_at = NULL, updated_at = %s
             WHERE controller_job_id = %s AND owner_instance_id = %s
               AND state IN ('claimed', 'running')
            """,
            (Jsonb(dict(result or {})), now, now, job_id, instance_id),
        )
        if updated.rowcount != 1:
            raise ControllerInvariantError("controller job completion lost its lease")

    def retry_job(
        self,
        job_id: str,
        *,
        instance_id: str,
        now: datetime,
        delay: timedelta,
        error_code: str,
    ) -> None:
        self._require_lock()
        updated = self.connection.execute(
            """
            UPDATE control_plane.controller_jobs
               SET state = 'retry_pending', last_error_code = %s,
                   next_retry_at = %s, lease_expires_at = NULL,
                   owner_instance_id = NULL, updated_at = %s
             WHERE controller_job_id = %s AND owner_instance_id = %s
               AND state IN ('claimed', 'running')
            """,
            (error_code, now + delay, now, job_id, instance_id),
        )
        if updated.rowcount != 1:
            raise ControllerInvariantError("controller job retry lost its lease")

    def fail_job(
        self,
        job_id: str,
        *,
        instance_id: str,
        now: datetime,
        error_code: str,
    ) -> None:
        self._require_lock()
        updated = self.connection.execute(
            """
            UPDATE control_plane.controller_jobs
               SET state = 'failed', last_error_code = %s,
                   lease_expires_at = NULL, owner_instance_id = NULL,
                   next_retry_at = NULL, updated_at = %s
             WHERE controller_job_id = %s AND owner_instance_id = %s
               AND state IN ('claimed', 'running')
            """,
            (error_code, now, job_id, instance_id),
        )
        if updated.rowcount != 1:
            raise ControllerInvariantError("controller job failure lost its lease")
