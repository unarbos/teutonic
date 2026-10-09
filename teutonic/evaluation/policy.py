from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import numpy as np

from .masking import token_weights


GENERIC_CONFIG_LOCK_KEYS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "model_type",
    "tie_word_embeddings",
    "rope_theta",
    "max_position_embeddings",
    "max_seq_len",
)


def paired_bootstrap_verdict(
    king_losses: list[float],
    challenger_losses: list[float],
    *,
    bootstrap_seed: int,
    n_bootstrap: int,
    alpha: float,
    delta_threshold: float,
    now: Callable[[], str] | None = None,
    scored_tokens: list[int] | None = None,
) -> dict[str, Any]:
    """Resample paired windows, recomputing the weighted ratio in every draw.

    Production masked scoring supplies scored_tokens. Omitting counts retains
    the historical equal-window calculation for existing records and callers.
    """
    if not king_losses or len(king_losses) != len(challenger_losses):
        raise ValueError("bootstrap requires non-empty paired losses")
    diff = np.asarray(king_losses, dtype=np.float64) - np.asarray(
        challenger_losses, dtype=np.float64
    )
    weights = token_weights(scored_tokens, len(diff)) if scored_tokens is not None else None
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot[i] = (
            diff[idx].mean() if weights is None
            else np.average(diff[idx], weights=weights[idx])
        )
    mu_hat = float(np.average(diff, weights=weights))
    lcb = float(np.quantile(boot, alpha))
    accepted = lcb > delta_threshold
    timestamp = now() if now is not None else datetime.now(timezone.utc).isoformat()
    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta": delta_threshold,
        "delta_threshold": delta_threshold,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "n_sequences": len(diff),
        "avg_king_loss": round(float(np.average(king_losses, weights=weights)), 6),
        "avg_challenger_loss": round(float(np.average(challenger_losses, weights=weights)), 6),
        "timestamp": timestamp,
    }


def provisional_paired_bootstrap(
    king_losses: list[float],
    challenger_losses: list[float],
    *,
    bootstrap_seed: int,
    n_bootstrap: int,
    alpha: float,
    delta_threshold: float,
    scored_tokens: list[int] | None = None,
) -> dict[str, Any]:
    """Compute an explicitly provisional checkpoint from partial paired losses."""
    if not king_losses or len(king_losses) != len(challenger_losses):
        raise ValueError("provisional bootstrap requires non-empty paired losses")
    if n_bootstrap < 1:
        raise ValueError("provisional bootstrap count must be positive")
    verdict = paired_bootstrap_verdict(
        king_losses,
        challenger_losses,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        delta_threshold=delta_threshold,
        scored_tokens=scored_tokens,
    )
    return {
        "provisional_mu_hat": verdict["mu_hat"],
        "provisional_lcb": verdict["lcb"],
        "provisional_n_sequences": verdict["n_sequences"],
        "provisional_n_bootstrap": n_bootstrap,
    }


def validate_config_lock(
    king_config: Mapping[str, Any],
    challenger_config: Mapping[str, Any],
    *,
    extra_lock_keys: tuple[str, ...] = (),
) -> str | None:
    """Return the legacy architecture/shape-lock rejection, if any."""
    king_arch = king_config.get("architectures", [])
    challenger_arch = challenger_config.get("architectures", [])
    if king_arch and challenger_arch and king_arch != challenger_arch:
        return f"architecture mismatch: king={king_arch} challenger={challenger_arch}"

    sentinel = object()
    for key in GENERIC_CONFIG_LOCK_KEYS + extra_lock_keys:
        king_value = king_config.get(key, sentinel)
        challenger_value = challenger_config.get(key, sentinel)
        if king_value != challenger_value:
            king_display = king_value if king_value is not sentinel else "<absent>"
            challenger_display = (
                challenger_value if challenger_value is not sentinel else "<absent>"
            )
            return (
                f"{key} mismatch: king={king_display} "
                f"challenger={challenger_display}"
            )
    return None


def _parse_registry_timestamp(value: str | datetime | None) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        try:
            parsed = parsedate_to_datetime(value)
        except Exception:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _trusted_timestamp_source(source: object) -> bool:
    return bool(source) and not str(source).startswith("untrusted:")


def decide_model_copy(
    *,
    challenger_repo: str,
    challenger_digest: str,
    king_repo: str,
    king_digest: str,
    challenger_info: Mapping[str, Any] | None = None,
    king_info: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the deterministic exact-layer copy decision from fetched metadata."""
    if not king_repo or not king_digest:
        return None
    if challenger_repo == king_repo and challenger_digest == king_digest:
        return {
            "action": "reject",
            "reason": (
                f"challenger is identical to the current king "
                f"(same repo {challenger_repo!r} and digest {challenger_digest[:19]})"
            ),
            "challenger_committed_at": None,
            "king_committed_at": None,
        }
    if challenger_info is None or king_info is None:
        return None

    challenger_layers = dict(challenger_info.get("safetensor_layers") or {})
    king_layers = dict(king_info.get("safetensor_layers") or {})
    if not challenger_layers or len(challenger_layers) != len(king_layers):
        return None
    if any(king_layers.get(title) != digest for title, digest in challenger_layers.items()):
        return None

    challenger_timestamp = challenger_info.get("committed_at")
    king_timestamp = king_info.get("committed_at")
    challenger_source = challenger_info.get("timestamp_source")
    king_source = king_info.get("timestamp_source")
    challenger_time = _parse_registry_timestamp(challenger_timestamp)
    king_time = _parse_registry_timestamp(king_timestamp)
    base_reason = (
        f"all {len(challenger_layers)} .safetensors layers have identical blob digests; "
        f"challenger pushed_at={challenger_timestamp} source={challenger_source}, "
        f"king pushed_at={king_timestamp} source={king_source}"
    )
    result_metadata = {
        "challenger_committed_at": challenger_timestamp,
        "king_committed_at": king_timestamp,
        "challenger_timestamp_source": challenger_source,
        "king_timestamp_source": king_source,
    }

    if (
        challenger_time is None
        or king_time is None
        or not _trusted_timestamp_source(challenger_source)
    ):
        return {
            "action": "reject",
            "reason": (
                "model is a copy of the king "
                f"(trusted challenger timestamp unavailable): {base_reason}"
            ),
            **result_metadata,
        }
    if challenger_time < king_time:
        return {
            "action": "crown_earlier",
            "reason": (
                "model is identical to the king but has an earlier registry-observed push "
                f"time ({challenger_timestamp} < {king_timestamp}); displacing king with "
                f"original author. {base_reason}"
            ),
            **result_metadata,
        }
    return {
        "action": "reject",
        "reason": f"model is a copy of the king (not earlier than king): {base_reason}",
        **result_metadata,
    }


def normalize_verdict(
    verdict: Mapping[str, Any],
    *,
    challenge_id: str,
    challenger_digest: str,
) -> dict[str, Any]:
    """Attach validator-owned identity fields without changing evaluator output."""
    normalized = dict(verdict)
    normalized["challenge_id"] = challenge_id
    normalized["challenger_digest"] = challenger_digest
    return normalized


def _verdict_shards_used(verdict: Mapping[str, Any]) -> list[dict[str, Any]]:
    shards = verdict.get("shards_used")
    if shards:
        return list(shards)
    dataset = verdict.get("dataset") if isinstance(verdict.get("dataset"), dict) else {}
    return list(dataset.get("shards_used") or [])


def build_verdict_history_entry(
    verdict: Mapping[str, Any],
    *,
    challenger_repo: str,
    hotkey: str,
    uid: int | None,
    coldkey: str | None,
    now: Callable[[], str],
) -> dict[str, Any]:
    """Reproduce the legacy public history projection for an evaluator verdict."""
    king_loss = verdict.get("avg_king_loss", 0)
    challenger_loss = verdict.get("avg_challenger_loss", 0)
    delta = verdict.get("delta", verdict.get("delta_threshold", 0))
    entry: dict[str, Any] = {
        "challenge_id": verdict.get("challenge_id"),
        "hotkey": hotkey,
        "uid": uid,
        "coldkey": coldkey,
        "challenger_repo": challenger_repo,
        "challenger_digest": verdict.get("challenger_digest", ""),
        "accepted": verdict.get("accepted", False),
        "verdict": verdict.get("verdict", "unknown"),
        "mu_hat": verdict.get("mu_hat", 0),
        "lcb": verdict.get("lcb", 0),
        "delta": delta,
        "avg_king_loss": king_loss,
        "avg_challenger_loss": challenger_loss,
        "best_loss": min(king_loss, challenger_loss) if (king_loss or challenger_loss) else 0,
        "wall_time_s": verdict.get("wall_time_s", 0),
        "timestamp": verdict.get("timestamp", now()),
    }
    optional_fields = (
        "challenger_committed_at",
        "king_committed_at",
        "challenger_timestamp_source",
        "king_timestamp_source",
        "source_scores",
    )
    if verdict.get("rejection_reason"):
        entry["rejection_reason"] = verdict["rejection_reason"]
    for field in optional_fields:
        value = verdict.get(field)
        if value is not None and (value or field.endswith("_at")):
            entry[field] = value
    shards_used = _verdict_shards_used(verdict)
    if shards_used:
        dataset = verdict.get("dataset") if isinstance(verdict.get("dataset"), dict) else {}
        entry["shards_used"] = shards_used
        entry["dataset"] = {
            "source": verdict.get("dataset_source") or dataset.get("source"),
            "shards_used": shards_used,
        }
    if verdict.get("early_stopped"):
        entry["early_stopped"] = True
        entry["n_sequences"] = verdict.get("n_sequences")
        entry["n_sequences_evaluated"] = verdict.get("n_sequences_evaluated")
    return entry


def build_failure_history_entry(
    submission: Mapping[str, Any],
    *,
    error_code: str,
    error_detail: object = "",
    uid: int | None,
    coldkey: str | None,
    now: Callable[[], str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproduce the legacy terminal-failure public history projection."""
    record: dict[str, Any] = {
        "challenge_id": submission.get("challenge_id", "?"),
        "hotkey": submission.get("hotkey", ""),
        "uid": uid,
        "coldkey": coldkey,
        "challenger_repo": submission.get("model_repo", ""),
        "challenger_digest": submission.get("model_digest", ""),
        "accepted": False,
        "verdict": "error",
        "error_code": error_code,
        "error_detail": str(error_detail),
        "mu_hat": 0,
        "lcb": 0,
        "delta": 0,
        "avg_king_loss": 0,
        "avg_challenger_loss": 0,
        "best_loss": 0,
        "wall_time_s": 0,
        "timestamp": now(),
    }
    if extra:
        record.update(extra)
    return record


def classify_eval_error(exc: BaseException | str) -> tuple[bool, str]:
    """Return the legacy retry decision and stable reason marker."""
    if isinstance(exc, asyncio.CancelledError):
        return True, "validator_cancelled"
    # httpx transport exceptions frequently have an empty message (notably
    # when an evaluator disappears mid-stream). Include the concrete exception
    # type so retry classification does not depend on optional error text.
    text = str(exc).lower()
    if isinstance(exc, BaseException):
        text = f"{type(exc).__name__.lower()} {text}"
    if ("stuck cdn" in text) or ("prefetch" in text and "exceeded" in text):
        return False, "prefetch_exhausted"
    if "safetensors_reuse_limit" in text:
        return False, "safetensors_reuse_limit"
    if "model_copy" in text:
        return False, "model_copy"
    for marker in ("dataset_index_download", "evaluator_interrupted"):
        if marker in text:
            return True, marker
    if (
        "failed to download shard" in text
        or "s3 shard download failed" in text
        or "retriesexceeded" in text
        or "max retries exceeded" in text
    ):
        return True, "dataset_shard_download"
    if text.startswith("eval server error") or "'eval server error'" in text:
        return False, "eval_server_reported"
    transient_markers = (
        "internal error",
        "stream idle",
        "watchdog timeout",
        "timed out",
        "timeout",
        "server disconnected",
        "connection reset",
        "connecterror",
        "readerror",
        "remoteprotocolerror",
        "streamconsumed",
        "streamclosed",
        "streamerror",
        "peer closed connection",
        "incomplete chunked",
        "incompleteread",
        "503",
        "502",
        "504",
    )
    for marker in transient_markers:
        if marker in text:
            return True, marker
    return False, ""
