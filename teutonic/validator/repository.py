from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from teutonic.evaluation import EarlyStoppingPolicy, EvaluationRequestV2
from teutonic.evaluation.configuration import DatasetManifestSnapshot, EvaluationSettings

from .contracts import ClaimedEvaluation, EvaluationPolicyConfig, RecoveryCandidate


class SchedulerLockUnavailable(RuntimeError):
    pass


class SchedulerInvariantError(RuntimeError):
    pass


class LeaseLostError(RuntimeError):
    pass


def scheduler_lock_key(netuid: int, chain_generation: str, competition: str) -> int:
    material = f"teutonic-validator-v1|{netuid}|{chain_generation}|{competition}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=True)


def _json_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


class ValidatorRepository:
    """Short PostgreSQL transactions for the order-sensitive validator workflow."""

    def __init__(
        self,
        connection,
        *,
        netuid: int,
        chain_generation: str,
        competition: str,
        instance_id: str,
        public_model_bucket: str,
    ) -> None:
        self.connection = connection
        self.netuid = netuid
        self.chain_generation = chain_generation
        self.competition = competition
        self.instance_id = instance_id
        self.public_model_bucket = public_model_bucket
        self._lock_key = scheduler_lock_key(netuid, chain_generation, competition)
        self._lock_held = False

    def acquire_lock(self) -> None:
        acquired = self.connection.execute(
            "SELECT pg_try_advisory_lock(%s)", (self._lock_key,)
        ).fetchone()[0]
        if not acquired:
            raise SchedulerLockUnavailable("another scheduler holds the competition lock")
        self._lock_held = True

    def release_lock(self) -> None:
        if self._lock_held:
            self.connection.execute("SELECT pg_advisory_unlock(%s)", (self._lock_key,))
            self._lock_held = False

    def _require_lock(self) -> None:
        if not self._lock_held:
            raise SchedulerLockUnavailable("competition scheduler lock is not held")

    def load_evaluation_settings(self) -> EvaluationSettings:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT ec.config_version, ec.dataset_label, ec.eval_n,
                       ec.delta_threshold, ec.shards_per_dataset, dm.position, dm.name,
                       dm.manifest_url, dm.manifest_sha256, dm.manifest_json,
                       dm.sample_proportion
                  FROM control_plane.competitions c
                  JOIN control_plane.evaluation_configs ec
                    ON ec.competition_id = c.competition_id AND ec.active
                  JOIN control_plane.dataset_manifests dm
                    ON dm.evaluation_config_id = ec.evaluation_config_id
                 WHERE c.netuid = %s AND c.chain_generation = %s AND c.name = %s
                 ORDER BY dm.position
                """,
                (self.netuid, self.chain_generation, self.competition),
            )
            rows = cursor.fetchall()
        if not rows:
            raise SchedulerInvariantError("competition has no active evaluation configuration")
        first = rows[0]
        manifests = tuple(
            DatasetManifestSnapshot(
                name=str(row["name"]),
                manifest_url=str(row["manifest_url"]),
                manifest_sha256=str(row["manifest_sha256"]),
                proportion=float(row["sample_proportion"]),
                manifest=row["manifest_json"],
            )
            for row in rows
        )
        return EvaluationSettings(
            config_version=str(first["config_version"]),
            dataset_label=str(first["dataset_label"]),
            n=int(first["eval_n"]),
            delta_threshold=float(first["delta_threshold"]),
            manifests=manifests,
            shards_per_dataset=int(first["shards_per_dataset"]),
        )

    def load_early_stopping_policy(self) -> EarlyStoppingPolicy:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT policy.enabled, policy.min_fraction,
                       policy.advantage_quantile, policy.margin,
                       policy.check_interval
                  FROM control_plane.competitions competition
                  JOIN control_plane.evaluation_early_stopping_policies policy
                    ON policy.competition_id = competition.competition_id
                 WHERE competition.netuid = %s
                   AND competition.chain_generation = %s
                   AND competition.name = %s
                """,
                (self.netuid, self.chain_generation, self.competition),
            )
            row = cursor.fetchone()
        if row is None:
            raise SchedulerInvariantError(
                "competition has no early-stopping policy; apply the database setup script"
            )
        return EarlyStoppingPolicy(
            enabled=bool(row["enabled"]),
            min_fraction=float(row["min_fraction"]),
            advantage_quantile=float(row["advantage_quantile"]),
            margin=float(row["margin"]),
            check_interval=int(row["check_interval"]),
        )

    def claim_next(
        self, *, now: datetime, policy: EvaluationPolicyConfig
    ) -> ClaimedEvaluation | None:
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor(
            row_factory=dict_row
        ) as cursor:
            cursor.execute(
                """
                SELECT c.competition_id, c.current_reign_id,
                       k.model_digest AS king_digest, k.public_bucket AS king_bucket,
                       k.public_prefix AS king_prefix,
                       u.upload_id, u.registration_id, u.model_digest, u.model_name,
                       u.ready_finalized_block, u.ready_extrinsic_index, u.ready_event_index,
                       ready_snapshot.finalized_block_hash AS ready_finalized_block_hash,
                       COALESCE(promoted.public_bucket, vu.immutable_bucket)
                           AS challenger_bucket,
                       COALESCE(promoted.public_prefix, vu.immutable_prefix)
                           AS challenger_prefix,
                       r.hotkey, r.uid,
                       assignment.coldkey,
                       previous.evaluation_id AS previous_evaluation_id,
                       previous.attempt_number AS previous_attempt,
                       previous.state AS previous_state,
                       previous.started_at AS previous_started_at,
                       previous.next_retry_at,
                       previous.claimed_king_reign_id AS previous_king_reign_id,
                       previous.request_payload AS previous_request
                  FROM control_plane.competitions c
                  JOIN control_plane.king_reigns k ON k.reign_id = c.current_reign_id
                  JOIN LATERAL (
                      SELECT candidate.*
                        FROM control_plane.uploads candidate
                        JOIN control_plane.registrations candidate_registration
                          ON candidate_registration.registration_id = candidate.registration_id
                       WHERE candidate_registration.netuid = c.netuid
                         AND candidate.chain_generation = c.chain_generation
                         AND candidate.state IN ('ready_for_evaluation', 'retry_pending')
                         AND NOT EXISTS (
                             SELECT 1
                               FROM control_plane.uploads pending
                               JOIN control_plane.registrations pending_registration
                                 ON pending_registration.registration_id = pending.registration_id
                              WHERE pending_registration.netuid = c.netuid
                                AND pending.chain_generation = c.chain_generation
                                AND pending.state IN (
                                    'evaluation_claimed', 'evaluating',
                                    'accepted_pending_promotion', 'promoted'
                                )
                         )
                       ORDER BY candidate.ready_finalized_block,
                                candidate.ready_extrinsic_index,
                                candidate.ready_event_index,
                                candidate.upload_id
                       FOR UPDATE OF candidate SKIP LOCKED
                       LIMIT 1
                  ) u ON true
                  JOIN control_plane.metagraph_snapshots ready_snapshot
                    ON ready_snapshot.netuid = c.netuid
                   AND ready_snapshot.chain_generation = c.chain_generation
                   AND ready_snapshot.finalized_block = u.ready_finalized_block
                   AND ready_snapshot.is_complete
                  JOIN control_plane.verified_uploads vu ON vu.upload_id = u.upload_id
                  LEFT JOIN control_plane.model_promotions promoted
                    ON promoted.model_digest = u.model_digest
                   AND promoted.state = 'promoted'
                  JOIN control_plane.registrations r ON r.registration_id = u.registration_id
                  LEFT JOIN LATERAL (
                      SELECT e.evaluation_id, e.attempt_number, e.state, e.started_at,
                             e.next_retry_at, e.claimed_king_reign_id, e.request_payload
                        FROM control_plane.evaluations e
                       WHERE e.upload_id = u.upload_id
                       ORDER BY e.attempt_number DESC
                       LIMIT 1
                  ) previous ON true
                  LEFT JOIN LATERAL (
                      SELECT a.coldkey
                        FROM control_plane.metagraph_snapshots s
                        JOIN control_plane.metagraph_uid_assignments a
                          ON a.snapshot_id = s.snapshot_id
                       WHERE s.netuid = r.netuid AND s.chain_generation = r.chain_generation
                         AND a.uid = r.uid AND a.hotkey = r.hotkey AND s.is_complete
                       ORDER BY s.finalized_block DESC
                       LIMIT 1
                  ) assignment ON true
                 WHERE c.netuid = %s AND c.chain_generation = %s AND c.name = %s
                 FOR UPDATE OF c
                """,
                (self.netuid, self.chain_generation, self.competition),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            if row["current_reign_id"] is None:
                raise SchedulerInvariantError("competition has no current king")
            if row["next_retry_at"] is not None and row["next_retry_at"] > now:
                return None

            # Dispatch failures happen before the evaluator accepts the job. Reclaim the
            # same durable attempt and request identity instead of spending a miner retry.
            # Reposting this request is safe because the evaluator binds eval_id to the
            # original request and returns the existing job after an ambiguous response.
            if (
                row["previous_state"] == "retryable_failure"
                and row["previous_started_at"] is None
            ):
                request = EvaluationRequestV2.from_mapping(row["previous_request"])
                evaluation_id = str(row["previous_evaluation_id"])
                cursor.execute(
                    """
                    UPDATE control_plane.evaluations
                       SET state = 'claimed', owner_instance_id = %s,
                           lease_expires_at = %s, heartbeat_at = %s,
                           failure_class = NULL, public_error_code = NULL,
                           private_diagnostic_reference = NULL, next_retry_at = NULL,
                           updated_at = clock_timestamp()
                     WHERE evaluation_id = %s AND state = 'retryable_failure'
                    """,
                    (self.instance_id, now + policy.lease, now, evaluation_id),
                )
                cursor.execute(
                    """
                    UPDATE control_plane.uploads
                       SET state = 'evaluation_claimed', failure_code = NULL,
                           updated_at = clock_timestamp()
                     WHERE upload_id = %s AND state = 'retry_pending'
                    """,
                    (row["upload_id"],),
                )
                return ClaimedEvaluation(
                    evaluation_id=evaluation_id,
                    upload_id=str(row["upload_id"]),
                    attempt_number=int(row["previous_attempt"]),
                    competition_id=str(row["competition_id"]),
                    claimed_king_reign_id=str(row["previous_king_reign_id"]),
                    request=request.request_payload,
                )

            attempt_number = int(row["previous_attempt"] or 0) + 1
            if attempt_number > policy.max_attempts:
                cursor.execute(
                    """
                    UPDATE control_plane.uploads
                       SET state = 'evaluation_failed', failure_code = 'attempt_limit_exhausted',
                           updated_at = clock_timestamp()
                     WHERE upload_id = %s
                    """,
                    (row["upload_id"],),
                )
                return None

            cursor.execute("SELECT gen_random_uuid() AS evaluation_id")
            evaluation_id = str(cursor.fetchone()["evaluation_id"])
            request = EvaluationRequestV2.from_mapping(
                {
                    "protocol_version": "teutonic-evaluator-v2",
                    "evaluation_id": evaluation_id,
                    "attempt_number": attempt_number,
                    "king": {
                        "kind": "r2-prefix",
                        "bucket": row["king_bucket"],
                        "prefix": row["king_prefix"],
                        "expected_digest": row["king_digest"],
                    },
                    "challenger": {
                        "kind": "r2-prefix",
                        "bucket": row["challenger_bucket"],
                        "prefix": row["challenger_prefix"],
                        "expected_digest": row["model_digest"],
                    },
                    "miner": {
                        "hotkey": row["hotkey"],
                        "coldkey": row["coldkey"] or "unavailable",
                        "uid": row["uid"],
                        "netuid": self.netuid,
                        "challenge_id": str(row["upload_id"]),
                    },
                    "versions": {
                        "evaluation_policy": policy.policy_version,
                        "dataset": policy.dataset_version,
                        "code": policy.code_version,
                        "evaluator": policy.evaluator_version,
                    },
                    "sampling": {
                        "seed": policy.sampling_seed,
                        "bootstrap_seed": policy.bootstrap_seed,
                        "block_hash": row["ready_finalized_block_hash"],
                    },
                    "limits": policy.thresholds,
                    "early_stopping": policy.early_stopping.request_dict(),
                    "dataset": policy.dataset_request(
                        block_hash=str(row["ready_finalized_block_hash"]),
                        hotkey=str(row["hotkey"]),
                    ),
                }
            )
            cursor.execute(
                """
                INSERT INTO control_plane.evaluations (
                    evaluation_id, upload_id, competition_id, attempt_number,
                    claimed_king_reign_id, state, owner_instance_id, lease_expires_at,
                    heartbeat_at, policy_version, code_version, dataset_version,
                    evaluator_version, sampling_seed, bootstrap_seed,
                    thresholds, request_sha256, request_payload
                ) VALUES (
                    %s, %s, %s, %s, %s, 'claimed', %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s::jsonb, %s, %s::jsonb
                )
                """,
                (
                    evaluation_id,
                    row["upload_id"],
                    row["competition_id"],
                    attempt_number,
                    row["current_reign_id"],
                    self.instance_id,
                    now + policy.lease,
                    now,
                    policy.policy_version,
                    policy.code_version,
                    policy.dataset_version,
                    policy.evaluator_version,
                    policy.sampling_seed,
                    policy.bootstrap_seed,
                    Jsonb(policy.persisted_thresholds),
                    request.request_sha256,
                    Jsonb(dict(request.request_payload)),
                ),
            )
            cursor.execute(
                """
                UPDATE control_plane.uploads
                   SET state = 'evaluation_claimed', failure_code = NULL,
                       updated_at = clock_timestamp()
                 WHERE upload_id = %s
                """,
                (row["upload_id"],),
            )
        return ClaimedEvaluation(
            evaluation_id=evaluation_id,
            upload_id=str(row["upload_id"]),
            attempt_number=attempt_number,
            competition_id=str(row["competition_id"]),
            claimed_king_reign_id=str(row["current_reign_id"]),
            request=request.request_payload,
        )

    def start_evaluating(
        self, evaluation_id: str, *, evaluator_job_id: str, now: datetime, lease: timedelta
    ) -> None:
        self._require_lock()
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE control_plane.evaluations
                   SET state = 'evaluating', evaluator_job_id = %s,
                       started_at = COALESCE(started_at, %s),
                       heartbeat_at = %s, lease_expires_at = %s, updated_at = clock_timestamp()
                 WHERE evaluation_id = %s AND owner_instance_id = %s
                   AND state IN ('claimed', 'evaluating')
                 RETURNING upload_id
                """,
                (
                    evaluator_job_id,
                    now,
                    now,
                    now + lease,
                    evaluation_id,
                    self.instance_id,
                ),
            ).fetchone()
            if row is None:
                raise LeaseLostError("evaluation lease is no longer owned")
            self.connection.execute(
                """
                UPDATE control_plane.uploads
                   SET state = 'evaluating', updated_at = clock_timestamp()
                 WHERE upload_id = %s AND state IN ('evaluation_claimed', 'evaluating')
                """,
                (row[0],),
            )

    def heartbeat(
        self,
        evaluation_id: str,
        *,
        now: datetime,
        lease: timedelta,
        progress: Mapping[str, Any] | None = None,
    ) -> None:
        self._require_lock()
        bounded = _bounded_progress(progress) if progress is not None else None
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE control_plane.evaluations
                   SET heartbeat_at = %s, lease_expires_at = %s,
                       progress_summary = COALESCE(%s::jsonb, progress_summary),
                       updated_at = clock_timestamp()
                 WHERE evaluation_id = %s AND owner_instance_id = %s
                   AND state IN ('claimed', 'evaluating') AND lease_expires_at >= %s
                 RETURNING evaluation_id
                """,
                (
                    now,
                    now + lease,
                    Jsonb(bounded) if bounded is not None else None,
                    evaluation_id,
                    self.instance_id,
                    now,
                ),
            ).fetchone()
            if row is None:
                raise LeaseLostError("evaluation heartbeat rejected after lease loss")

    def complete_verdict(
        self,
        evaluation_id: str,
        *,
        result: Mapping[str, Any],
        now: datetime,
        publish_non_winning: bool,
        private_diagnostic_reference: str | None = None,
        result_artifact_reference: str | None = None,
    ) -> str:
        self._require_lock()
        accepted = bool(result["accepted"])
        verdict = "accepted" if accepted else "rejected"
        upload_state = "accepted_pending_promotion" if accepted else "rejected"
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT e.*, vu.model_digest, vu.immutable_bucket, vu.immutable_prefix,
                       vu.object_count, vu.total_size_bytes
                  FROM control_plane.evaluations e
                  JOIN control_plane.verified_uploads vu ON vu.upload_id = e.upload_id
                 WHERE e.evaluation_id = %s
                 FOR UPDATE OF e
                """,
                (evaluation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise SchedulerInvariantError("unknown evaluation")
            if row["state"] == "completed":
                if row["verdict"] != verdict or row["verdict_summary"] != dict(result):
                    raise SchedulerInvariantError(
                        "terminal evaluation replay conflicts with verdict"
                    )
                return str(row["verdict"])
            if row["owner_instance_id"] != self.instance_id or row["state"] not in {
                "claimed",
                "evaluating",
            }:
                raise LeaseLostError("evaluation is no longer owned")
            cursor.execute(
                """
                UPDATE control_plane.evaluations
                   SET state = 'completed', verdict = %s, verdict_summary = %s::jsonb,
                       private_diagnostic_reference = %s, result_artifact_reference = %s,
                       result_artifact_sha256 = %s, completed_at = %s,
                       lease_expires_at = NULL, heartbeat_at = %s, updated_at = clock_timestamp()
                 WHERE evaluation_id = %s
                """,
                (
                    verdict,
                    Jsonb(dict(result)),
                    private_diagnostic_reference,
                    result_artifact_reference,
                    result["result_artifact_sha256"],
                    now,
                    now,
                    evaluation_id,
                ),
            )
            cursor.execute(
                """
                UPDATE control_plane.uploads
                   SET state = %s, failure_code = NULL, updated_at = clock_timestamp()
                 WHERE upload_id = %s
                """,
                (upload_state, row["upload_id"]),
            )
            if accepted or publish_non_winning:
                disposition = "winner" if accepted else "non_winner"
                cursor.execute(
                    """
                    INSERT INTO control_plane.model_promotions (
                        upload_id, evaluation_id, model_digest, disposition,
                        private_bucket, private_prefix, public_bucket, public_prefix,
                        state, idempotency_key, next_retry_at,
                        expected_object_count, expected_size_bytes
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        'promotion_pending', %s, %s, %s, %s
                    ) ON CONFLICT (model_digest) DO UPDATE SET
                        evaluation_id = EXCLUDED.evaluation_id,
                        disposition = 'winner',
                        updated_at = clock_timestamp()
                    WHERE EXCLUDED.disposition = 'winner'
                      AND model_promotions.state = 'promoted'
                    RETURNING state
                    """,
                    (
                        row["upload_id"],
                        evaluation_id,
                        row["model_digest"],
                        disposition,
                        row["immutable_bucket"],
                        row["immutable_prefix"],
                        self.public_model_bucket,
                        f"models/sha256/{row['model_digest']}/",
                        f"promote-model:{row['upload_id']}",
                        now,
                        row["object_count"],
                        row["total_size_bytes"],
                    ),
                )
                promotion = cursor.fetchone()
                if accepted and promotion is not None and promotion["state"] == "promoted":
                    cursor.execute(
                        """
                        UPDATE control_plane.uploads
                           SET state = 'promoted', updated_at = clock_timestamp()
                         WHERE upload_id = %s AND state = 'accepted_pending_promotion'
                        """,
                        (row["upload_id"],),
                    )
        return verdict

    def fail_attempt(
        self,
        evaluation_id: str,
        *,
        now: datetime,
        failure_class: str,
        public_error_code: str,
        retry: bool,
        retry_delay: timedelta = timedelta(0),
        private_diagnostic_reference: str | None = None,
    ) -> str:
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM control_plane.evaluations WHERE evaluation_id = %s FOR UPDATE",
                (evaluation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise SchedulerInvariantError("unknown evaluation")
            if row["state"] in {"completed", "terminal_failure", "retryable_failure"}:
                return str(row["state"])
            if row["owner_instance_id"] != self.instance_id:
                raise LeaseLostError("evaluation is no longer owned")
            target = "retryable_failure" if retry else "terminal_failure"
            upload_state = (
                "retry_pending"
                if retry
                else "invalid_evaluation_input"
                if failure_class == "deterministic_submission"
                else "evaluation_failed"
            )
            verdict = None if retry else "failed"
            summary = None if retry else {"error_code": public_error_code}
            cursor.execute(
                """
                UPDATE control_plane.evaluations
                   SET state = %s, failure_class = %s, public_error_code = %s,
                       private_diagnostic_reference = %s, verdict = %s,
                       verdict_summary = %s::jsonb, next_retry_at = %s,
                       completed_at = CASE WHEN %s = 'terminal_failure' THEN %s ELSE NULL END,
                       lease_expires_at = NULL, heartbeat_at = %s, updated_at = clock_timestamp()
                 WHERE evaluation_id = %s
                """,
                (
                    target,
                    failure_class,
                    public_error_code,
                    private_diagnostic_reference,
                    verdict,
                    Jsonb(summary) if summary is not None else None,
                    now + retry_delay if retry else None,
                    target,
                    now,
                    now,
                    evaluation_id,
                ),
            )
            cursor.execute(
                """
                UPDATE control_plane.uploads
                   SET state = %s, failure_code = %s, updated_at = clock_timestamp()
                 WHERE upload_id = %s
                """,
                (upload_state, public_error_code, row["upload_id"]),
            )
        return target

    def defer_dispatch(
        self,
        evaluation_id: str,
        *,
        now: datetime,
        public_error_code: str,
        retry_delay: timedelta,
        private_diagnostic_reference: str | None = None,
    ) -> None:
        """Defer a job the evaluator has not accepted without consuming its attempt."""
        self._require_lock()
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT upload_id, state, started_at, owner_instance_id
                  FROM control_plane.evaluations
                 WHERE evaluation_id = %s
                 FOR UPDATE
                """,
                (evaluation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise SchedulerInvariantError("unknown evaluation")
            if row["state"] == "retryable_failure" and row["started_at"] is None:
                return
            if row["owner_instance_id"] != self.instance_id or row["state"] != "claimed":
                raise LeaseLostError("unaccepted evaluation is no longer owned")
            if row["started_at"] is not None:
                raise SchedulerInvariantError("accepted evaluation cannot be deferred as dispatch")
            cursor.execute(
                """
                UPDATE control_plane.evaluations
                   SET state = 'retryable_failure',
                       failure_class = 'transient_infrastructure',
                       public_error_code = %s, private_diagnostic_reference = %s,
                       next_retry_at = %s, owner_instance_id = NULL,
                       lease_expires_at = NULL, heartbeat_at = %s,
                       updated_at = clock_timestamp()
                 WHERE evaluation_id = %s
                """,
                (
                    public_error_code,
                    private_diagnostic_reference,
                    now + retry_delay,
                    now,
                    evaluation_id,
                ),
            )
            cursor.execute(
                """
                UPDATE control_plane.uploads
                   SET state = 'retry_pending', failure_code = %s,
                       updated_at = clock_timestamp()
                 WHERE upload_id = %s
                """,
                (public_error_code, row["upload_id"]),
            )

    def recovery_candidates(self, *, now: datetime) -> tuple[RecoveryCandidate, ...]:
        self._require_lock()
        rows = self.connection.execute(
            """
            SELECT e.evaluation_id, e.upload_id, e.attempt_number, e.state,
                   e.evaluator_job_id, e.owner_instance_id, e.competition_id,
                   e.claimed_king_reign_id, e.request_payload
              FROM control_plane.evaluations e
              JOIN control_plane.competitions c ON c.competition_id = e.competition_id
             WHERE c.netuid = %s AND c.chain_generation = %s AND c.name = %s
               AND e.state IN ('claimed', 'evaluating')
               AND (e.lease_expires_at < %s OR e.owner_instance_id = %s)
             ORDER BY e.created_at, e.evaluation_id
            """,
            (self.netuid, self.chain_generation, self.competition, now, self.instance_id),
        ).fetchall()
        return tuple(
            RecoveryCandidate(
                evaluation_id=str(row[0]),
                upload_id=str(row[1]),
                attempt_number=int(row[2]),
                state=str(row[3]),
                evaluator_job_id=str(row[4] or f"{row[0]}:{row[2]}"),
                owner_instance_id=row[5],
                competition_id=str(row[6]),
                claimed_king_reign_id=str(row[7]),
                request=row[8] or {},
            )
            for row in rows
        )

    def adopt(self, evaluation_id: str, *, now: datetime, lease: timedelta) -> None:
        self._require_lock()
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE control_plane.evaluations
                   SET owner_instance_id = %s, heartbeat_at = %s, lease_expires_at = %s,
                       updated_at = clock_timestamp()
                 WHERE evaluation_id = %s AND state IN ('claimed', 'evaluating')
                 RETURNING evaluation_id
                """,
                (self.instance_id, now, now + lease, evaluation_id),
            ).fetchone()
            if row is None:
                raise LeaseLostError("evaluation cannot be adopted")

    def promotion_weight_hotkeys(self, promotion_id: str, *, limit: int = 5) -> tuple[str, ...]:
        """Return the promoted challenger followed by the current durable policy.

        UID resolution deliberately happens against a finalized metagraph in the
        runtime. PostgreSQL supplies hotkeys so deregistration and UID remaps do not
        silently redirect weight. Older databases fall back to recent reign ordering.
        """
        if limit < 1:
            raise ValueError("weight hotkey limit must be positive")
        with self.connection.cursor(row_factory=dict_row) as cursor:
            row = cursor.execute(
                """
                SELECT u.signalling_hotkey, e.competition_id
                  FROM control_plane.model_promotions p
                  JOIN control_plane.evaluations e ON e.evaluation_id = p.evaluation_id
                  JOIN control_plane.uploads u ON u.upload_id = p.upload_id
                 WHERE p.promotion_id = %s
                   AND p.disposition = 'winner' AND p.state = 'promoted'
                """,
                (promotion_id,),
            ).fetchone()
            if row is None:
                raise SchedulerInvariantError("only a promoted winner has weight targets")
            current_policy = cursor.execute(
                """
                SELECT weights.policy_hotkeys
                  FROM control_plane.competitions competition
                  JOIN control_plane.weight_publications weights
                    ON weights.source_reign_id = competition.current_reign_id
                 WHERE competition.competition_id = %s
                """,
                (row["competition_id"],),
            ).fetchone()
            if current_policy is not None:
                previous_hotkeys = tuple(current_policy["policy_hotkeys"])
            else:
                reigns = cursor.execute(
                    """
                    SELECT hotkey
                      FROM control_plane.king_reigns
                     WHERE competition_id = %s
                     ORDER BY reign_number DESC
                     LIMIT %s
                    """,
                    (row["competition_id"], limit),
                ).fetchall()
                previous_hotkeys = tuple(item["hotkey"] for item in reigns)
        ordered = [row["signalling_hotkey"], *previous_hotkeys]
        return tuple(dict.fromkeys(ordered))[:limit]

    def current_weight_policy(self) -> Mapping[str, Any] | None:
        self._require_lock()
        with self.connection.cursor(row_factory=dict_row) as cursor:
            row = cursor.execute(
                """
                SELECT w.weight_publication_id, w.payload_revision,
                       w.mapping_finalized_block, w.policy_hotkeys
                  FROM control_plane.competitions c
                  JOIN control_plane.weight_publications w
                    ON w.source_reign_id = c.current_reign_id
                 WHERE c.netuid = %s AND c.chain_generation = %s AND c.name = %s
                """,
                (self.netuid, self.chain_generation, self.competition),
            ).fetchone()
        if row is None:
            return None
        return {
            "publication_id": str(row["weight_publication_id"]),
            "payload_revision": int(row["payload_revision"]),
            "mapping_finalized_block": int(row["mapping_finalized_block"]),
            "policy_hotkeys": tuple(str(value) for value in row["policy_hotkeys"]),
        }

    def refresh_current_weight_plan(
        self,
        *,
        publication_id: str,
        expected_revision: int,
        mapping_finalized_block: int,
        target_hotkeys: Sequence[str],
        target_uids: Sequence[int],
        normalized_weights: Sequence[float],
        now: datetime,
    ) -> bool:
        """Create a new frozen payload revision after a finalized UID remap.

        An in-flight attempt always keeps its original payload. Refresh is deferred
        until that attempt becomes terminal, and every attempt stores its own payload.
        """
        self._require_lock()
        if (
            len(target_uids) != len(normalized_weights)
            or len(target_hotkeys) != len(target_uids)
            or not target_uids
        ):
            raise ValueError("weight target and value arrays must be non-empty and equal length")
        payload = {
            "target_uids": list(target_uids),
            "normalized_weights": list(normalized_weights),
        }
        digest = _json_digest(payload)
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            row = cursor.execute(
                """
                SELECT w.*
                  FROM control_plane.competitions c
                  JOIN control_plane.weight_publications w
                    ON w.source_reign_id = c.current_reign_id
                 WHERE c.netuid = %s AND c.chain_generation = %s AND c.name = %s
                   AND w.weight_publication_id = %s
                 FOR UPDATE OF w
                """,
                (
                    self.netuid,
                    self.chain_generation,
                    self.competition,
                    publication_id,
                ),
            ).fetchone()
            if row is None or int(row["payload_revision"]) != expected_revision:
                return False
            if mapping_finalized_block <= int(row["mapping_finalized_block"]):
                return False
            if digest == row["payload_sha256"]:
                mapping_changed = list(target_hotkeys) != list(row["target_hotkeys"])
                cursor.execute(
                    """
                    UPDATE control_plane.weight_publications
                       SET target_hotkeys = %s,
                           payload_revision = payload_revision + %s,
                           mapping_finalized_block = %s,
                           updated_at = clock_timestamp()
                     WHERE weight_publication_id = %s
                    """,
                    (
                        list(target_hotkeys),
                        1 if mapping_changed else 0,
                        mapping_finalized_block,
                        publication_id,
                    ),
                )
                return mapping_changed
            active = cursor.execute(
                """
                SELECT 1
                  FROM control_plane.weight_submission_attempts
                 WHERE weight_publication_id = %s
                   AND state IN ('claimed', 'submitting', 'submitted', 'included', 'retry_pending')
                 LIMIT 1
                """,
                (publication_id,),
            ).fetchone()
            if active is not None:
                return False
            cursor.execute(
                """
                UPDATE control_plane.weight_publications
                   SET target_hotkeys = %s, target_uids = %s,
                       normalized_weights = %s, payload_sha256 = %s,
                       payload_revision = payload_revision + 1,
                       mapping_finalized_block = %s, state = 'requested',
                       owner_instance_id = NULL, lease_expires_at = NULL,
                       next_retry_at = NULL, extrinsic_id = NULL,
                       included_block = NULL, finalized_block = NULL,
                       last_error_code = NULL, requested_at = %s,
                       submitted_at = NULL, included_at = NULL, finalized_at = NULL,
                       cadence_enabled = true,
                       next_due_block = CASE
                           WHEN last_attempted_block IS NULL THEN NULL
                           ELSE GREATEST(last_attempted_block + 1, %s)
                       END,
                       updated_at = clock_timestamp()
                 WHERE weight_publication_id = %s
                """,
                (
                    list(target_hotkeys),
                    list(target_uids),
                    list(normalized_weights),
                    digest,
                    mapping_finalized_block,
                    now,
                    mapping_finalized_block,
                    publication_id,
                ),
            )
        return True

    def heartbeat_service(
        self,
        *,
        now: datetime,
        phase: str,
        software_version: str,
        state: str = "active",
        current_work_id: str | None = None,
        restart_reason: str | None = None,
    ) -> None:
        with self.connection.transaction():
            self.connection.execute(
                """
                INSERT INTO control_plane.service_instances (
                    service_name, instance_id, software_version, state, phase,
                    current_work_id, started_at, heartbeat_at, restart_reason
                ) VALUES ('validator', %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (service_name, instance_id) DO UPDATE
                   SET software_version = EXCLUDED.software_version,
                       state = EXCLUDED.state, phase = EXCLUDED.phase,
                       current_work_id = EXCLUDED.current_work_id,
                       heartbeat_at = EXCLUDED.heartbeat_at,
                       restart_reason = EXCLUDED.restart_reason
                """,
                (
                    self.instance_id,
                    software_version,
                    state,
                    phase,
                    current_work_id,
                    now,
                    now,
                    restart_reason,
                ),
            )

    def crown_promoted_winner(
        self,
        promotion_id: str,
        *,
        now: datetime,
        crowned_finalized_block: int,
        policy_hotkeys: Sequence[str],
        target_hotkeys: Sequence[str],
        target_uids: Sequence[int],
        normalized_weights: Sequence[float],
    ) -> str | None:
        if (
            len(target_uids) != len(normalized_weights)
            or len(target_hotkeys) != len(target_uids)
            or not target_uids
            or not policy_hotkeys
        ):
            raise ValueError("weight target and value arrays must be non-empty and equal length")
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT p.*, e.competition_id, e.claimed_king_reign_id, e.policy_version, e.verdict,
                       u.signalling_hotkey, r.uid, c.current_reign_id, c.next_reign_number
                  FROM control_plane.model_promotions p
                  JOIN control_plane.evaluations e ON e.evaluation_id = p.evaluation_id
                  JOIN control_plane.uploads u ON u.upload_id = p.upload_id
                  JOIN control_plane.registrations r ON r.registration_id = u.registration_id
                  JOIN control_plane.competitions c ON c.competition_id = e.competition_id
                 WHERE p.promotion_id = %s
                 FOR UPDATE OF p, e, u, c
                """,
                (promotion_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise SchedulerInvariantError("unknown promotion")
            if row["state"] != "promoted" or row["disposition"] != "winner":
                raise SchedulerInvariantError("only a promoted winner can be crowned")
            existing = cursor.execute(
                "SELECT reign_id FROM control_plane.king_reigns WHERE causing_evaluation_id = %s",
                (row["evaluation_id"],),
            ).fetchone()
            if existing is not None:
                return str(existing["reign_id"])
            if row["verdict"] != "accepted":
                raise SchedulerInvariantError("promotion has no accepted verdict")
            if row["current_reign_id"] != row["claimed_king_reign_id"]:
                cursor.execute(
                    """
                    UPDATE control_plane.uploads
                       SET state = 'retry_pending', failure_code = 'stale_king',
                           updated_at = clock_timestamp()
                     WHERE upload_id = %s
                    """,
                    (row["upload_id"],),
                )
                return None
            previous = row["current_reign_id"]
            reign_number = row["next_reign_number"]
            cursor.execute(
                """
                UPDATE control_plane.king_reigns
                   SET ended_at = %s, replacement_reason = 'accepted_challenger'
                 WHERE reign_id = %s AND ended_at IS NULL
                """,
                (now, previous),
            )
            cursor.execute(
                """
                INSERT INTO control_plane.king_reigns (
                    competition_id, reign_number, accepted_upload_id, causing_evaluation_id,
                    model_digest, public_bucket, public_prefix, hotkey, uid,
                    previous_reign_id, crowned_at, crowned_finalized_block
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING reign_id
                """,
                (
                    row["competition_id"],
                    reign_number,
                    row["upload_id"],
                    row["evaluation_id"],
                    row["model_digest"],
                    row["public_bucket"],
                    row["public_prefix"],
                    row["signalling_hotkey"],
                    row["uid"],
                    previous,
                    now,
                    crowned_finalized_block,
                ),
            )
            reign_id = str(cursor.fetchone()["reign_id"])
            cursor.execute(
                """
                UPDATE control_plane.competitions
                   SET current_reign_id = %s, next_reign_number = next_reign_number + 1,
                       updated_at = clock_timestamp()
                 WHERE competition_id = %s
                """,
                (reign_id, row["competition_id"]),
            )
            cursor.execute(
                """
                UPDATE control_plane.uploads
                   SET state = 'accepted', updated_at = clock_timestamp()
                 WHERE upload_id = %s
                """,
                (row["upload_id"],),
            )
            weight_payload = {
                "target_uids": list(target_uids),
                "normalized_weights": list(normalized_weights),
            }
            cursor.execute(
                """
                INSERT INTO control_plane.weight_publications (
                    competition_id, source_reign_id, policy_version, policy_hotkeys,
                    target_hotkeys, target_uids, normalized_weights, payload_sha256,
                    mapping_finalized_block, idempotency_key, state
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'requested')
                ON CONFLICT (source_reign_id) DO NOTHING
                """,
                (
                    row["competition_id"],
                    reign_id,
                    row["policy_version"],
                    list(policy_hotkeys),
                    list(target_hotkeys),
                    list(target_uids),
                    list(normalized_weights),
                    _json_digest(weight_payload),
                    crowned_finalized_block,
                    f"publish-weights:{reign_id}",
                ),
            )
            cursor.execute(
                """
                INSERT INTO control_plane.notification_outbox (
                    topic, source_id, idempotency_key, payload, state, next_retry_at
                ) VALUES ('king_crowned', %s, %s, %s::jsonb, 'pending', %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (
                    reign_id,
                    f"notify-king-crowned:{reign_id}",
                    Jsonb({"reign_id": reign_id, "competition_id": str(row["competition_id"])}),
                    now,
                ),
            )
        return reign_id


def _bounded_progress(progress: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(progress)
    if "completed_sequences" not in normalized and "done" in normalized:
        normalized["completed_sequences"] = normalized["done"]
    if "requested_sequences" not in normalized and "total" in normalized:
        normalized["requested_sequences"] = normalized["total"]
    if (
        "percent" not in normalized
        and "completed_sequences" in normalized
        and "requested_sequences" in normalized
    ):
        completed = max(0, int(normalized["completed_sequences"]))
        requested = max(0, int(normalized["requested_sequences"]))
        normalized["percent"] = 100.0 * completed / requested if requested else 0.0
    allowed = {
        "phase",
        "completed_sequences",
        "requested_sequences",
        "percent",
        "elapsed_seconds",
        "early_stopped",
    }
    result = {key: normalized[key] for key in allowed if key in normalized}
    for key in ("provisional_mu_hat", "provisional_lcb"):
        if key not in normalized:
            continue
        try:
            value = float(normalized[key])
        except (TypeError, ValueError) as exc:
            raise SchedulerInvariantError(f"progress {key} must be numeric") from exc
        if not math.isfinite(value):
            raise SchedulerInvariantError(f"progress {key} must be finite")
        result[key] = value
    for key in ("provisional_n_sequences", "provisional_n_bootstrap"):
        if key not in normalized:
            continue
        try:
            value = int(normalized[key])
        except (TypeError, ValueError) as exc:
            raise SchedulerInvariantError(f"progress {key} must be an integer") from exc
        if value < 1:
            raise SchedulerInvariantError(f"progress {key} must be positive")
        result[key] = value
    encoded = json.dumps(result, separators=(",", ":"), ensure_ascii=True)
    if len(encoded.encode()) > 4096:
        raise SchedulerInvariantError("progress summary exceeds 4096 bytes")
    return result
