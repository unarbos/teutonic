"""Small, independent publication of queue and evaluation progress.

This path deliberately never reads historical evaluations, dataset manifests,
market APIs or the full dashboard object. One snapshot covers every competition.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from time import monotonic

from jsonschema import Draft202012Validator, FormatChecker
from psycopg.rows import dict_row

from .contracts import SCHEMA_PATH, DashboardContractError, _reject_non_finite
from .projection import DASHBOARD_LOCK_ID, DashboardProjectionError, DashboardProjectionRepository

LIVE_KEY = "dashboard-live.json"


@lru_cache(maxsize=1)
def _validator():
    definitions = json.loads(SCHEMA_PATH.read_text())["$defs"]
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "generated_at", "netuid", "generation",
                     "source_watermark", "competitions"],
        "properties": {
            "schema_version": {"const": 1},
            "generated_at": {"$ref": "#/$defs/timestamp"},
            "netuid": {"type": "integer", "minimum": 0},
            "generation": {"type": "string", "minLength": 1},
            "source_watermark": {"type": "integer", "minimum": 0},
            "competitions": {"type": "array", "minItems": 1, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["key", "name", "queue", "current_eval", "service_status"],
                "properties": {
                    "key": {"type": "string", "pattern": "^(main|math|code|text|mix[0-9]{2})$"},
                    "name": {"type": "string", "minLength": 1},
                    "queue": {"type": "array", "items": {"$ref": "#/$defs/queue_entry"}},
                    "current_eval": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/current_eval"}]},
                    "service_status": {"$ref": "#/$defs/service_status"},
                },
            }},
        },
        "$defs": definitions,
    }
    return Draft202012Validator(schema, format_checker=FormatChecker())


def canonical_live_json(payload):
    _reject_non_finite(payload)
    error = next(_validator().iter_errors(payload), None)
    if error is not None:
        raise DashboardContractError(f"dashboard-live invalid: {error.message}")
    keys = [item["key"] for item in payload["competitions"]]
    if len(set(keys)) != len(keys) or "main" not in keys:
        raise DashboardContractError("dashboard-live needs unique competitions including main")
    if sum(item["current_eval"] is not None for item in payload["competitions"]) > 1:
        raise DashboardContractError("dashboard-live has multiple active evaluations")
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


class LiveDashboardRepository(DashboardProjectionRepository):
    # The slow publisher has its own lock and cannot block live publication.
    lock_id = DASHBOARD_LOCK_ID + 1

    def project(self, *, now=None):
        if not self._lock_held:
            raise DashboardProjectionError("live publisher lock is not held")
        generated = now or datetime.now(UTC)
        scope = (self.netuid, self.chain_generation)
        started = monotonic()
        with self.connection.transaction(), self.connection.cursor(row_factory=dict_row) as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cur.execute("SET LOCAL statement_timeout = '3s'")
            competitions = cur.execute(
                "SELECT competition, competition_key FROM control_plane.dashboard_competitions "
                "WHERE netuid=%s AND chain_generation=%s AND main_competition=%s "
                "ORDER BY competition_key", (*scope, self.competition),
            ).fetchall()
            if not competitions:
                raise DashboardProjectionError("no configured competitions for live publication")
            names = [row["competition"] for row in competitions]
            queues = cur.execute(
                "SELECT * FROM control_plane.dashboard_queue WHERE netuid=%s "
                "AND chain_generation=%s AND competition=ANY(%s) ORDER BY queue_position",
                (*scope, names),
            ).fetchall()
            current = cur.execute(
                "SELECT * FROM control_plane.dashboard_current_evaluation WHERE netuid=%s "
                "AND chain_generation=%s AND competition=ANY(%s)", (*scope, names),
            ).fetchall()
            if len(current) > 1:
                raise DashboardProjectionError("multiple active evaluations in live snapshot")
            # Match the existing publisher's optional digest enrichment. The
            # restricted dashboard role can still publish without base-table access.
            pending = queues + current
            for row in pending:
                row["model_digest"] = None
            if pending and cur.execute(
                "SELECT has_table_privilege(current_user, "
                "'control_plane.verified_uploads', 'SELECT') AS allowed"
            ).fetchone()["allowed"]:
                digests = cur.execute(
                    "SELECT substring(encode(public.digest(upload_id::text, 'sha256'), 'hex') "
                    "FROM 1 FOR 16) AS challenge_id, model_digest "
                    "FROM control_plane.verified_uploads WHERE "
                    "substring(encode(public.digest(upload_id::text, 'sha256'), 'hex') "
                    "FROM 1 FOR 16)=ANY(%s)",
                    ([row["challenge_id"] for row in pending],),
                ).fetchall()
                by_id = {row["challenge_id"]: row["model_digest"] for row in digests}
                for row in pending:
                    row["model_digest"] = by_id.get(row["challenge_id"])
            stats = cur.execute(
                "SELECT source_watermark FROM control_plane.dashboard_stats WHERE netuid=%s "
                "AND chain_generation=%s AND competition=%s", (*scope, self.competition),
            ).fetchone()
            weights = cur.execute(
                "SELECT * FROM control_plane.dashboard_weight_status WHERE netuid=%s "
                "AND chain_generation=%s AND competition=%s", (*scope, self.competition),
            ).fetchone()
            services = cur.execute(
                "SELECT * FROM control_plane.dashboard_service_health ORDER BY service_name"
            ).fetchall()
        # Do not label a slow, obsolete projection as a fresh live update.
        if monotonic() - started > 5:
            raise DashboardProjectionError("live projection exceeded five-second deadline")
        result = []
        for competition in competitions:
            name = competition["competition"]
            active = next((row for row in current if row["competition"] == name), None)
            active = self._current(active) if active else None
            result.append({
                "key": competition["competition_key"], "name": name,
                "queue": [self._queue(row) for row in queues if row["competition"] == name],
                "current_eval": active,
                "service_status": self._services(services, active, self._weight(weights), generated),
            })
        return {
            "schema_version": 1, "generated_at": generated.astimezone(UTC).isoformat(),
            "netuid": self.netuid, "generation": self.chain_generation,
            "source_watermark": int(stats["source_watermark"]), "competitions": result,
        }


def publish_live(client, bucket, payload):
    body = canonical_live_json(payload)
    if len(body) > 1024 * 1024:
        raise DashboardContractError("live publication exceeds 1 MiB")
    client.put_object(
        Bucket=bucket, Key=LIVE_KEY, Body=body,
        ContentType="application/json; charset=utf-8",
        CacheControl="public, max-age=5, must-revalidate",
    )
    return len(body)
