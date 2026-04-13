"""Validator state management -- writes live state and event history to R2."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .r2 import R2Client

logger = logging.getLogger(__name__)

STATE_KEY = "state/validator_state.json"
HISTORY_KEY = "state/history.jsonl"
QUEUE_KEY = "state/queue.json"
KING_KEY = "king/current.json"
SEEN_HOTKEYS_KEY = "state/seen_hotkeys.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ValidatorState:
    """Manages the validator's live state and event history in R2.

    All state is persisted to R2 so a dashboard can read it with zero
    additional infrastructure.
    """

    def __init__(self, r2: R2Client):
        self.r2 = r2
        self.king: dict[str, Any] = {}
        self.current_eval: dict[str, Any] | None = None
        self.queue: list[dict[str, Any]] = []
        self.seen_hotkeys: set[str] = set()
        self.stats = {
            "total_challenges": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "total_invalid": 0,
            "total_stale": 0,
            "avg_king_loss": 0.0,
            "avg_challenger_loss": 0.0,
            "uptime_since": _now(),
        }

    def load(self) -> None:
        """Load existing state from R2 (on startup)."""
        saved = self.r2.get_json(STATE_KEY)
        if saved:
            self.king = saved.get("king", {})
            self.current_eval = saved.get("current_eval")
            self.queue = saved.get("queue", [])
            self.stats = saved.get("stats", self.stats)
            logger.info("Loaded validator state from R2: king reign %s", self.king.get("reign_number"))

        king_data = self.r2.get_json(KING_KEY)
        if king_data:
            self.king = king_data

        queue_data = self.r2.get_json(QUEUE_KEY)
        if queue_data:
            self.queue = queue_data.get("pending", [])

        seen_data = self.r2.get_json(SEEN_HOTKEYS_KEY)
        if seen_data:
            self.seen_hotkeys = set(seen_data.get("hotkeys", []))
            logger.info("Loaded %d seen hotkeys from R2", len(self.seen_hotkeys))

    def _flush(self) -> None:
        """Write current state snapshot to R2."""
        self.r2.put_json(STATE_KEY, {
            "king": self.king,
            "current_eval": self.current_eval,
            "queue": self.queue,
            "stats": self.stats,
            "updated_at": _now(),
        })

    def _append_event(self, event: dict) -> None:
        event.setdefault("timestamp", _now())
        self.r2.append_jsonl(HISTORY_KEY, event)

    def _flush_queue(self) -> None:
        self.r2.put_json(QUEUE_KEY, {"pending": self.queue, "updated_at": _now()})

    def _flush_king(self) -> None:
        self.r2.put_json(KING_KEY, self.king)

    def _flush_seen_hotkeys(self) -> None:
        self.r2.put_json(SEEN_HOTKEYS_KEY, {
            "hotkeys": sorted(self.seen_hotkeys),
            "updated_at": _now(),
        })

    def mark_seen(self, hotkey: str) -> None:
        """Record a hotkey as permanently seen (never re-evaluate)."""
        self.seen_hotkeys.add(hotkey)
        self._flush_seen_hotkeys()

    def is_seen(self, hotkey: str) -> bool:
        return hotkey in self.seen_hotkeys

    # -- Public event methods --

    def set_initial_king(
        self, hotkey: str, hf_repo: str, king_hash: str, block: int
    ) -> None:
        self.king = {
            "hotkey": hotkey,
            "hf_repo": hf_repo,
            "king_hash": king_hash,
            "reign_number": 0,
            "crowned_at": _now(),
            "crowned_block": block,
            "challenge_id": "seed",
        }
        self._flush_king()
        self._flush()
        self._append_event({
            "event": "king_initialized",
            "hotkey": hotkey,
            "hf_repo": hf_repo,
            "king_hash": king_hash,
        })

    def enqueue_challenge(
        self,
        challenge_id: str,
        hotkey: str,
        hf_repo: str,
        commit_block: int,
        king_hash: str,
        model_hash: str,
    ) -> None:
        entry = {
            "challenge_id": challenge_id,
            "hotkey": hotkey,
            "hf_repo": hf_repo,
            "commit_block": commit_block,
            "king_hash": king_hash,
            "model_hash": model_hash,
            "queued_at": _now(),
        }
        self.queue.append(entry)
        self.stats["total_challenges"] += 1
        self._flush_queue()
        self._flush()
        self._append_event({"event": "challenge_queued", **entry})

    def dequeue(self) -> dict[str, Any] | None:
        if not self.queue:
            return None
        return self.queue.pop(0)

    def eval_started(
        self, challenge_id: str, hotkey: str, hf_repo: str, pod_name: str
    ) -> None:
        self.current_eval = {
            "challenge_id": challenge_id,
            "hotkey": hotkey,
            "hf_repo": hf_repo,
            "status": "running",
            "started_at": _now(),
            "progress": {"s": 0, "n": 0, "N": 0, "K": 0},
            "pod_name": pod_name,
        }
        self._flush_queue()
        self._flush()
        self._append_event({
            "event": "eval_started",
            "challenge_id": challenge_id,
            "hotkey": hotkey,
            "pod_name": pod_name,
        })

    def eval_progress(self, s: int, n: int, N: int, K: int) -> None:
        """Update live eval progress (called periodically)."""
        if self.current_eval:
            self.current_eval["progress"] = {"s": s, "n": n, "N": N, "K": K}
            self._flush()

    def eval_completed(
        self,
        challenge_id: str,
        hotkey: str,
        accepted: bool,
        verdict: dict,
    ) -> None:
        if accepted:
            self.stats["total_accepted"] += 1
        else:
            self.stats["total_rejected"] += 1

        if verdict.get("avg_king_loss"):
            self.stats["avg_king_loss"] = verdict["avg_king_loss"]
        if verdict.get("avg_challenger_loss"):
            self.stats["avg_challenger_loss"] = verdict["avg_challenger_loss"]

        self.current_eval = None
        self._flush()
        self._append_event({
            "event": "eval_completed",
            "challenge_id": challenge_id,
            "hotkey": hotkey,
            "accepted": accepted,
            **verdict,
        })

    def king_changed(
        self,
        challenge_id: str,
        new_hotkey: str,
        new_hf_repo: str,
        new_king_hash: str,
        block: int,
        avg_loss: float,
    ) -> None:
        old_hotkey = self.king.get("hotkey")
        reign = self.king.get("reign_number", 0) + 1
        self.king = {
            "hotkey": new_hotkey,
            "hf_repo": new_hf_repo,
            "king_hash": new_king_hash,
            "reign_number": reign,
            "crowned_at": _now(),
            "crowned_block": block,
            "challenge_id": challenge_id,
        }
        self._flush_king()
        self._flush()
        self._append_event({
            "event": "king_changed",
            "challenge_id": challenge_id,
            "new_king_hotkey": new_hotkey,
            "old_king_hotkey": old_hotkey,
            "reign_number": reign,
            "avg_king_loss": avg_loss,
        })

    def submission_rejected(
        self, challenge_id: str, hotkey: str, reason: str, detail: str = ""
    ) -> None:
        self.stats["total_invalid"] += 1
        self._flush()
        self._append_event({
            "event": "submission_rejected",
            "challenge_id": challenge_id,
            "hotkey": hotkey,
            "reason": reason,
            "detail": detail,
        })

    def submission_stale(self, challenge_id: str, hotkey: str) -> None:
        self.stats["total_stale"] += 1
        self._flush()
        self._append_event({
            "event": "submission_stale",
            "challenge_id": challenge_id,
            "hotkey": hotkey,
            "reason": "king_hash_mismatch",
        })
