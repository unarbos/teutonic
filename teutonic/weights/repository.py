from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

from psycopg.rows import dict_row

from .contracts import SubmissionReceipt, WeightPlan


class WeightPublisherLockUnavailable(RuntimeError):
    pass


class WeightLeaseLostError(RuntimeError):
    pass


def weight_publisher_lock_key(netuid: int, chain_generation: str, competition: str) -> int:
    material = f"teutonic-weight-publisher-v1|{netuid}|{chain_generation}|{competition}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=True)


class WeightPublicationRepository:
    def __init__(
        self,
        connection,
        *,
        netuid: int,
        chain_generation: str,
        competition: str,
        instance_id: str,
    ) -> None:
        self.connection = connection
        self.netuid = netuid
        self.chain_generation = chain_generation
        self.competition = competition
        self.instance_id = instance_id
        self._lock_key = weight_publisher_lock_key(netuid, chain_generation, competition)
        self._lock_held = False

    def acquire_lock(self) -> None:
        acquired = self.connection.execute(
            "SELECT pg_try_advisory_lock(%s)", (self._lock_key,)
        ).fetchone()[0]
        if not acquired:
            raise WeightPublisherLockUnavailable("another weight publisher holds the worker lock")
        self._lock_held = True

    def release_lock(self) -> None:
        if self._lock_held:
            self.connection.execute("SELECT pg_advisory_unlock(%s)", (self._lock_key,))
            self._lock_held = False

    def _require_lock(self) -> None:
        if not self._lock_held:
            raise WeightPublisherLockUnavailable("weight publisher lock is not held")

    def claim_next(
        self, *, now: datetime, lease: timedelta, current_block: int
    ) -> WeightPlan | None:
        """Claim the current reign's recovery or cadence attempt.

        Every older publication is superseded first, including uncertain attempts. The
        current plan is therefore never delayed by an obsolete reign. An uncertain
        attempt of the current plan is reconciled until its 101-block cadence boundary;
        at that boundary it is superseded by a fresh, same-payload attempt.
        """
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT w.*, COALESCE(c.reward_reign_id, c.current_reign_id) AS current_reign_id, r.reign_number
                  FROM control_plane.competitions c
                  JOIN control_plane.weight_publications w
                    ON w.source_reign_id = COALESCE(c.reward_reign_id, c.current_reign_id)
                  JOIN control_plane.king_reigns r ON r.reign_id = w.source_reign_id
                 WHERE c.netuid = %s AND c.chain_generation = %s AND c.name = %s
                 FOR UPDATE OF w
                """,
                (self.netuid, self.chain_generation, self.competition),
            )
            publication = cursor.fetchone()
            if publication is None:
                return None
            cursor.execute(
                """
                UPDATE control_plane.weight_publications old
                   SET state = 'superseded', cadence_enabled = false,
                       owner_instance_id = NULL, lease_expires_at = NULL,
                       next_retry_at = NULL, superseded_by = %s,
                       superseded_at = %s, updated_at = clock_timestamp()
                 WHERE old.competition_id = %s AND old.weight_publication_id <> %s
                   AND old.state <> 'superseded'
                """,
                (
                    publication["weight_publication_id"],
                    now,
                    publication["competition_id"],
                    publication["weight_publication_id"],
                ),
            )
            cursor.execute(
                """
                UPDATE control_plane.weight_submission_attempts a
                   SET state = 'superseded', owner_instance_id = NULL,
                       lease_expires_at = NULL, next_retry_at = NULL,
                       last_error_code = 'newer_reign', updated_at = clock_timestamp()
                  FROM control_plane.weight_publications old
                 WHERE a.weight_publication_id = old.weight_publication_id
                   AND old.competition_id = %s AND old.weight_publication_id <> %s
                   AND a.state NOT IN ('finalized', 'failed', 'superseded')
                """,
                (publication["competition_id"], publication["weight_publication_id"]),
            )
            cursor.execute(
                """
                SELECT *
                  FROM control_plane.weight_submission_attempts
                 WHERE weight_publication_id = %s
                   AND state IN (
                       'claimed', 'submitting', 'submitted', 'included', 'retry_pending'
                   )
                 ORDER BY sequence DESC
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1
                """,
                (publication["weight_publication_id"],),
            )
            attempt = cursor.fetchone()
            if attempt is not None:
                cadence_elapsed = (
                    attempt["submission_started_block"] is not None
                    and current_block
                    >= int(attempt["submission_started_block"]) + int(publication["cadence_blocks"])
                )
                if cadence_elapsed:
                    cursor.execute(
                        """
                        UPDATE control_plane.weight_submission_attempts
                           SET state = 'superseded', owner_instance_id = NULL,
                               lease_expires_at = NULL, next_retry_at = NULL,
                               last_error_code = 'cadence_elapsed',
                               updated_at = clock_timestamp()
                         WHERE weight_attempt_id = %s
                        """,
                        (attempt["weight_attempt_id"],),
                    )
                    attempt = None
                elif attempt["state"] == "retry_pending" and (
                    attempt["next_retry_at"] is not None and attempt["next_retry_at"] > now
                ):
                    return None
                elif attempt["state"] in {"claimed", "submitting"} and (
                    attempt["owner_instance_id"] != self.instance_id
                    and attempt["lease_expires_at"] > now
                ):
                    return None
            if attempt is None:
                if not publication["cadence_enabled"]:
                    return None
                next_due = publication["next_due_block"]
                if publication["last_attempted_block"] is not None and (
                    next_due is None or current_block < int(next_due)
                ):
                    return None
                sequence = cursor.execute(
                    """
                    SELECT COALESCE(max(sequence), 0) + 1 AS next_sequence
                      FROM control_plane.weight_submission_attempts
                     WHERE weight_publication_id = %s
                    """,
                    (publication["weight_publication_id"],),
                ).fetchone()["next_sequence"]
                cursor.execute(
                    """
                    INSERT INTO control_plane.weight_submission_attempts (
                        weight_publication_id, sequence, scheduled_block,
                        idempotency_key, payload_revision, target_hotkeys, target_uids,
                        normalized_weights, payload_sha256, state,
                        owner_instance_id, lease_expires_at, try_count, claimed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'claimed', %s, %s, 1, %s
                    )
                    RETURNING *
                    """,
                    (
                        publication["weight_publication_id"],
                        sequence,
                        current_block,
                        f"{publication['idempotency_key']}:attempt:{sequence}",
                        publication["payload_revision"],
                        publication["target_hotkeys"],
                        publication["target_uids"],
                        publication["normalized_weights"],
                        publication["payload_sha256"],
                        self.instance_id,
                        now + lease,
                        now,
                    ),
                )
                attempt = cursor.fetchone()
                previous_state = "claimed"
            else:
                previous_state = attempt["state"]
                cursor.execute(
                    """
                    UPDATE control_plane.weight_submission_attempts
                       SET state = 'claimed', owner_instance_id = %s,
                           lease_expires_at = %s, next_retry_at = NULL,
                           try_count = try_count + 1, updated_at = clock_timestamp()
                     WHERE weight_attempt_id = %s
                     RETURNING *
                    """,
                    (self.instance_id, now + lease, attempt["weight_attempt_id"]),
                )
                attempt = cursor.fetchone()
            cursor.execute(
                """
                UPDATE control_plane.weight_publications
                   SET state = 'claimed', owner_instance_id = %s, lease_expires_at = %s,
                       next_retry_at = NULL, updated_at = clock_timestamp()
                 WHERE weight_publication_id = %s
                """,
                (self.instance_id, now + lease, publication["weight_publication_id"]),
            )
        return self._plan(publication, attempt, previous_state)

    @staticmethod
    def _plan(publication, attempt, previous_state: str) -> WeightPlan:
        return WeightPlan(
            publication_id=str(publication["weight_publication_id"]),
            competition_id=str(publication["competition_id"]),
            source_reign_id=str(publication["source_reign_id"]),
            current_reign_id=str(publication["current_reign_id"]),
            reign_number=int(publication["reign_number"]),
            policy_version=publication["policy_version"],
            payload_revision=int(attempt["payload_revision"]),
            target_hotkeys=tuple(str(value) for value in attempt["target_hotkeys"]),
            target_uids=tuple(int(uid) for uid in attempt["target_uids"]),
            normalized_weights=tuple(float(weight) for weight in attempt["normalized_weights"]),
            payload_sha256=attempt["payload_sha256"],
            idempotency_key=publication["idempotency_key"],
            previous_state=previous_state,
            attempt_count=int(attempt["try_count"]),
            attempt_id=str(attempt["weight_attempt_id"]),
            attempt_sequence=int(attempt["sequence"]),
            scheduled_block=int(attempt["scheduled_block"]),
            submission_started_block=attempt["submission_started_block"],
            submission_expires_block=attempt["submission_expires_block"],
            observed_last_update=attempt["observed_last_update"],
            extrinsic_id=attempt["extrinsic_id"],
        )

    def is_current(self, plan: WeightPlan) -> bool:
        row = self.connection.execute(
            """
            SELECT COALESCE(c.reward_reign_id, c.current_reign_id) = %s
              FROM control_plane.competitions c
             WHERE c.competition_id = %s
            """,
            (plan.source_reign_id, plan.competition_id),
        ).fetchone()
        return bool(row and row[0])

    def mark_submitting(
        self,
        plan: WeightPlan,
        *,
        now: datetime,
        lease: timedelta,
        publisher_mode: str,
        network: str,
        signer_hotkey: str,
        started_block: int,
        expires_block: int,
        observed_last_update: int,
        cadence_blocks: int = 101,
    ) -> None:
        self._require_lock()
        if plan.attempt_id is None:
            raise WeightLeaseLostError("weight plan has no claimed attempt")
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE control_plane.weight_submission_attempts
                   SET state = 'submitting', lease_expires_at = %s,
                       publisher_mode = COALESCE(publisher_mode, %s),
                       network = COALESCE(network, %s),
                       signer_hotkey = COALESCE(signer_hotkey, %s),
                       submission_started_block = %s, submission_expires_block = %s,
                       observed_last_update = %s, last_error_code = NULL,
                       updated_at = clock_timestamp()
                 WHERE weight_attempt_id = %s AND owner_instance_id = %s
                   AND state = 'claimed'
                   AND (publisher_mode IS NULL OR publisher_mode = %s)
                   AND (network IS NULL OR network = %s)
                   AND (signer_hotkey IS NULL OR signer_hotkey = %s)
                 RETURNING weight_attempt_id
                """,
                (
                    now + lease,
                    publisher_mode,
                    network,
                    signer_hotkey,
                    started_block,
                    expires_block,
                    observed_last_update,
                    plan.attempt_id,
                    self.instance_id,
                    publisher_mode,
                    network,
                    signer_hotkey,
                ),
            ).fetchone()
            if row is None:
                raise WeightLeaseLostError("weight attempt could not enter submitting state")
            self.connection.execute(
                """
                UPDATE control_plane.weight_publications
                   SET state = 'submitting', lease_expires_at = %s,
                       attempt_count = attempt_count + 1,
                       last_attempted_block = %s, next_due_block = %s,
                       extrinsic_id = NULL, included_block = NULL, finalized_block = NULL,
                       submitted_at = NULL, included_at = NULL, finalized_at = NULL,
                       last_error_code = NULL, updated_at = clock_timestamp()
                 WHERE weight_publication_id = %s AND owner_instance_id = %s
                """,
                (
                    now + lease,
                    started_block,
                    started_block + cadence_blocks,
                    plan.publication_id,
                    self.instance_id,
                ),
            )

    def mark_submitted(
        self, plan: WeightPlan, *, receipt: SubmissionReceipt, now: datetime
    ) -> None:
        self._transition(
            plan,
            expected=("submitting", "claimed"),
            state="submitted",
            now=now,
            receipt=receipt,
        )

    def mark_included(self, plan: WeightPlan, *, receipt: SubmissionReceipt, now: datetime) -> None:
        if receipt.included_block is not None:
            self._transition(plan, expected=("submitted",), state="included", now=now,
                             receipt=receipt)

    def _transition(
        self,
        plan: WeightPlan,
        *,
        expected: tuple[str, ...],
        state: str,
        now: datetime,
        receipt: SubmissionReceipt,
    ) -> None:
        if plan.attempt_id is None:
            raise WeightLeaseLostError("weight plan has no claimed attempt")
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE control_plane.weight_submission_attempts
                   SET state = %s, lease_expires_at = NULL,
                       extrinsic_id = COALESCE(%s, extrinsic_id),
                       included_block = COALESCE(%s, included_block),
                       included_block_hash = COALESCE(%s, included_block_hash),
                       submitted_at = COALESCE(submitted_at, %s),
                       included_at = CASE WHEN %s = 'included' THEN COALESCE(included_at, %s)
                                          ELSE included_at END,
                       updated_at = clock_timestamp()
                 WHERE weight_attempt_id = %s AND owner_instance_id = %s
                   AND state = ANY(%s)
                 RETURNING weight_attempt_id
                """,
                (
                    state,
                    receipt.extrinsic_id,
                    receipt.included_block,
                    receipt.included_block_hash,
                    now,
                    state,
                    now,
                    plan.attempt_id,
                    self.instance_id,
                    list(expected),
                ),
            ).fetchone()
            if row is None:
                raise WeightLeaseLostError(f"weight transition to {state} lost its attempt")
            self.connection.execute(
                """
                UPDATE control_plane.weight_publications
                   SET state = %s, lease_expires_at = NULL,
                       extrinsic_id = COALESCE(%s, extrinsic_id),
                       included_block = COALESCE(%s, included_block),
                       submitted_at = COALESCE(submitted_at, %s),
                       included_at = CASE WHEN %s = 'included' THEN COALESCE(included_at, %s)
                                          ELSE included_at END,
                       updated_at = clock_timestamp()
                 WHERE weight_publication_id = %s AND owner_instance_id = %s
                """,
                (
                    state,
                    receipt.extrinsic_id,
                    receipt.included_block,
                    now,
                    state,
                    now,
                    plan.publication_id,
                    self.instance_id,
                ),
            )

    def mark_finalized(
        self, plan: WeightPlan, *, receipt: SubmissionReceipt, now: datetime
    ) -> None:
        if receipt.finalized_block is None or plan.attempt_id is None:
            raise ValueError("finalized receipt or claimed attempt is missing")
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE control_plane.weight_submission_attempts
                   SET state = 'finalized', owner_instance_id = NULL, lease_expires_at = NULL,
                       extrinsic_id = COALESCE(%s, extrinsic_id),
                       included_block = COALESCE(included_block, %s),
                       included_block_hash = COALESCE(included_block_hash, %s),
                       finalized_block = %s, finalized_block_hash = %s,
                       submitted_at = COALESCE(submitted_at, %s),
                       included_at = COALESCE(included_at, %s), finalized_at = %s,
                       next_retry_at = NULL, last_error_code = NULL,
                       updated_at = clock_timestamp()
                 WHERE weight_attempt_id = %s AND owner_instance_id = %s
                   AND state IN ('claimed', 'submitting', 'submitted', 'included')
                 RETURNING weight_attempt_id
                """,
                (
                    receipt.extrinsic_id,
                    receipt.included_block or receipt.finalized_block,
                    receipt.included_block_hash or receipt.finalized_block_hash,
                    receipt.finalized_block,
                    receipt.finalized_block_hash,
                    now,
                    now,
                    now,
                    plan.attempt_id,
                    self.instance_id,
                ),
            ).fetchone()
            if row is None:
                raise WeightLeaseLostError("weight finality acknowledgement lost its attempt")
            self.connection.execute(
                """
                UPDATE control_plane.weight_publications
                   SET state = 'finalized', owner_instance_id = NULL, lease_expires_at = NULL,
                       extrinsic_id = COALESCE(%s, extrinsic_id),
                       included_block = COALESCE(included_block, %s),
                       finalized_block = %s,
                       submitted_at = COALESCE(submitted_at, %s),
                       included_at = COALESCE(included_at, %s), finalized_at = %s,
                       next_retry_at = NULL, last_error_code = NULL,
                       updated_at = clock_timestamp()
                 WHERE weight_publication_id = %s AND owner_instance_id = %s
                """,
                (
                    receipt.extrinsic_id,
                    receipt.included_block or receipt.finalized_block,
                    receipt.finalized_block,
                    now,
                    now,
                    now,
                    plan.publication_id,
                    self.instance_id,
                ),
            )

    def retry_or_fail(
        self,
        plan: WeightPlan,
        *,
        now: datetime,
        error_code: str,
        retry_delay: timedelta,
        max_attempts: int,
        terminal: bool = False,
    ) -> str:
        if plan.attempt_id is None:
            raise WeightLeaseLostError("weight plan has no claimed attempt")
        state = "failed" if terminal or plan.attempt_count >= max_attempts else "retry_pending"
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE control_plane.weight_submission_attempts
                   SET state = %s, owner_instance_id = NULL, lease_expires_at = NULL,
                       next_retry_at = %s, last_error_code = %s,
                       updated_at = clock_timestamp()
                 WHERE weight_attempt_id = %s AND owner_instance_id = %s
                   AND state IN ('claimed', 'submitting', 'submitted', 'included')
                 RETURNING weight_attempt_id
                """,
                (
                    state,
                    None if state == "failed" else now + retry_delay,
                    error_code,
                    plan.attempt_id,
                    self.instance_id,
                ),
            ).fetchone()
            if row is None:
                raise WeightLeaseLostError("weight retry transition lost its attempt")
            self.connection.execute(
                """
                UPDATE control_plane.weight_publications
                   SET state = %s,
                       cadence_enabled = CASE WHEN %s THEN false ELSE cadence_enabled END,
                       owner_instance_id = NULL, lease_expires_at = NULL,
                       next_retry_at = %s, last_error_code = %s,
                       updated_at = clock_timestamp()
                 WHERE weight_publication_id = %s AND owner_instance_id = %s
                """,
                (
                    state,
                    terminal,
                    None if state == "failed" else now + retry_delay,
                    error_code,
                    plan.publication_id,
                    self.instance_id,
                ),
            )
        return state

    def supersede_claim(self, plan: WeightPlan, *, now: datetime) -> None:
        if plan.attempt_id is None:
            raise WeightLeaseLostError("weight plan has no claimed attempt")
        with self.connection.transaction():
            newer = self.connection.execute(
                "SELECT COALESCE(reward_reign_id, current_reign_id) FROM control_plane.competitions WHERE competition_id = %s",
                (plan.competition_id,),
            ).fetchone()
            if newer is None or str(newer[0]) == plan.source_reign_id:
                raise WeightLeaseLostError("weight plan is still current")
            self.connection.execute(
                """
                UPDATE control_plane.weight_submission_attempts
                   SET state = 'superseded', owner_instance_id = NULL,
                       lease_expires_at = NULL, next_retry_at = NULL,
                       last_error_code = 'newer_reign', updated_at = clock_timestamp()
                 WHERE weight_attempt_id = %s AND owner_instance_id = %s AND state = 'claimed'
                """,
                (plan.attempt_id, self.instance_id),
            )
            self.connection.execute(
                """
                UPDATE control_plane.weight_publications old
                   SET state = 'superseded', cadence_enabled = false,
                       owner_instance_id = NULL, lease_expires_at = NULL,
                       superseded_by = newer.weight_publication_id,
                       superseded_at = %s, last_error_code = 'newer_reign',
                       updated_at = clock_timestamp()
                  FROM control_plane.weight_publications newer
                 WHERE old.weight_publication_id = %s
                   AND newer.source_reign_id = %s
                """,
                (now, plan.publication_id, str(newer[0])),
            )

    def heartbeat_service(self, *, now: datetime, phase: str, software_version: str) -> None:
        with self.connection.transaction():
            self.connection.execute(
                """
                INSERT INTO control_plane.service_instances (
                    service_name, instance_id, software_version, state, phase,
                    current_work_id, started_at, heartbeat_at
                ) VALUES ('weight-publisher', %s, %s, 'active', %s, NULL, %s, %s)
                ON CONFLICT (service_name, instance_id) DO UPDATE
                   SET software_version = EXCLUDED.software_version, state = 'active',
                       phase = EXCLUDED.phase, heartbeat_at = EXCLUDED.heartbeat_at
                """,
                (self.instance_id, software_version, phase, now, now),
            )
