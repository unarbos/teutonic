"""Bind published checkpoint reuse to its original finalized coldkey."""


def model_ownership_error(cursor, upload_id: str) -> str | None:
    # Full inventory digests change when a README or shard filename changes.
    # Compare the multiset of safetensors sizes/hashes as well, ignoring names.
    # p.upload_id remains the original artifact source even on a later win.
    row = cursor.execute(
        """
        WITH candidate AS (
            SELECT u.upload_id, u.ready_finalized_block, u.signalling_hotkey,
                   r.netuid, r.chain_generation, r.uid, vu.model_digest
              FROM control_plane.uploads u
              JOIN control_plane.registrations r USING (registration_id)
              JOIN control_plane.verified_uploads vu USING (upload_id)
             WHERE u.upload_id = %s
        ), candidate_weights AS (
            SELECT jsonb_agg(jsonb_build_array(sha256, size_bytes)
                             ORDER BY sha256, size_bytes) AS fingerprints
              FROM control_plane.upload_files
             WHERE upload_id = %s AND object_path LIKE '%%.safetensors'
        )
        SELECT p.upload_id AS original_upload_id,
               (SELECT a.coldkey FROM control_plane.metagraph_snapshots s
                  JOIN control_plane.metagraph_uid_assignments a USING (snapshot_id)
                 WHERE s.netuid=candidate.netuid AND s.chain_generation=candidate.chain_generation
                   AND s.finalized_block=candidate.ready_finalized_block AND s.is_complete
                   AND a.uid=candidate.uid AND a.hotkey=candidate.signalling_hotkey
               ) AS candidate_coldkey,
               (SELECT a.coldkey FROM control_plane.metagraph_snapshots s
                  JOIN control_plane.metagraph_uid_assignments a USING (snapshot_id)
                 WHERE s.netuid=original_registration.netuid
                   AND s.chain_generation=original_registration.chain_generation
                   AND s.finalized_block=original.ready_finalized_block AND s.is_complete
                   AND a.uid=original_registration.uid AND a.hotkey=original.signalling_hotkey
               ) AS original_coldkey
          FROM candidate CROSS JOIN candidate_weights
          JOIN control_plane.model_promotions p ON (
               p.model_digest=candidate.model_digest
               OR (candidate_weights.fingerprints IS NOT NULL
                   AND candidate_weights.fingerprints=(
                       SELECT jsonb_agg(jsonb_build_array(f.sha256, f.size_bytes)
                                        ORDER BY f.sha256, f.size_bytes)
                         FROM control_plane.upload_files f
                        WHERE f.upload_id=p.upload_id AND f.object_path LIKE '%%.safetensors'
                   )))
          JOIN control_plane.uploads original ON original.upload_id=p.upload_id
          JOIN control_plane.registrations original_registration
            ON original_registration.registration_id=original.registration_id
         ORDER BY p.created_at, p.promotion_id
         LIMIT 1
        """,
        (upload_id, upload_id),
    ).fetchone()
    if row is None or str(row["original_upload_id"]) == str(upload_id):
        return None
    if not row["original_coldkey"] or not row["candidate_coldkey"]:
        return "published_model_owner_unverified"
    if row["original_coldkey"] != row["candidate_coldkey"]:
        return "published_model_owner_mismatch"
    return None


def reject_model_copy(cursor, *, upload_id, evaluation_id, reason, now):
    """Record a terminal policy failure and release the shared evaluation queue."""
    cursor.execute(
        """UPDATE control_plane.evaluations
              SET state='terminal_failure', verdict='failed', failure_class='policy',
                  public_error_code='model_copy', private_diagnostic_reference=%s,
                  verdict_summary=COALESCE(verdict_summary,'{}'::jsonb)
                      || jsonb_build_object('accepted',false,'policy_error',%s::text),
                  lease_expires_at=NULL, next_retry_at=NULL, completed_at=%s,
                  updated_at=clock_timestamp()
            WHERE evaluation_id=%s AND upload_id=%s""",
        (reason, reason, now, evaluation_id, upload_id),
    )
    cursor.execute(
        """UPDATE control_plane.uploads
              SET state='evaluation_failed', failure_code='model_copy', updated_at=clock_timestamp()
            WHERE upload_id=%s""",
        (upload_id,),
    )
