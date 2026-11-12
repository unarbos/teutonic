from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Queue
from typing import Any

from .early_stopping import EarlyStoppingPolicy
from .packed_text import validate_mix

PROTOCOL_VERSION = "teutonic-evaluator-v3"
# Validated for single-GPU BF16 MiMo replicas on B300 with 2048 tokens and FA4.
DEFAULT_EVAL_BATCH_SIZE = 40
MAX_BATCH_SIZE = 1024
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")
_PRIVATE_MODEL_PREFIX_RE = re.compile(r"^models/registrations/[0-9a-f]{64}/$")
_FORBIDDEN_KEY_PARTS = (
    "access_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "session_token",
    "signed_url",
)


class ProtocolValidationError(ValueError):
    pass


class AttemptConflictError(RuntimeError):
    pass


class AttemptBusyError(RuntimeError):
    pass


def _required_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{field_name} must be an object")
    return value


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _exact_keys(value: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ProtocolValidationError(f"{field_name} contains unknown fields: {unknown}")


def _reject_credentials(value: object, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                raise ProtocolValidationError(f"{path}.{key} may not contain credentials")
            _reject_credentials(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_credentials(nested, f"{path}[{index}]")


def _normalize_digest(value: object, field_name: str) -> str:
    digest = _required_string(value, field_name).lower().removeprefix("sha256:")
    if not _DIGEST_RE.fullmatch(digest):
        raise ProtocolValidationError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class R2Artifact:
    bucket: str
    prefix: str
    expected_digest: str

    @classmethod
    def from_mapping(cls, value: object, field_name: str) -> R2Artifact:
        data = _required_mapping(value, field_name)
        _exact_keys(data, {"kind", "bucket", "prefix", "expected_digest"}, field_name)
        if data.get("kind") != "r2-prefix":
            raise ProtocolValidationError(f"{field_name}.kind must be 'r2-prefix'")
        bucket = _required_string(data.get("bucket"), f"{field_name}.bucket")
        prefix = _required_string(data.get("prefix"), f"{field_name}.prefix").strip("/") + "/"
        if ".." in prefix.split("/"):
            raise ProtocolValidationError(f"{field_name}.prefix may not contain '..'")
        expected_digest = _normalize_digest(
            data.get("expected_digest"), f"{field_name}.expected_digest"
        )
        public_prefix = f"models/sha256/{expected_digest}/"
        if prefix != public_prefix and not _PRIVATE_MODEL_PREFIX_RE.fullmatch(prefix):
            raise ProtocolValidationError(
                f"{field_name}.prefix must identify a public digest or private registration"
            )
        return cls(bucket=bucket, prefix=prefix, expected_digest=expected_digest)

    def public_dict(self) -> dict[str, str]:
        return {
            "kind": "r2-prefix",
            "bucket": self.bucket,
            "prefix": self.prefix,
            "expected_digest": self.expected_digest,
        }


@dataclass(frozen=True)
class EvaluationRequestV3:
    evaluation_id: str
    attempt_number: int
    king: R2Artifact
    challenger: R2Artifact
    miner: Mapping[str, Any]
    versions: Mapping[str, str]
    sampling: Mapping[str, int | str]
    limits: Mapping[str, int | float]
    early_stopping: Mapping[str, bool | float | int]
    dataset: Mapping[str, Any]
    request_payload: Mapping[str, Any]

    @property
    def eval_id(self) -> str:
        return f"{self.evaluation_id}:{self.attempt_number}"

    @property
    def request_sha256(self) -> str:
        canonical = json.dumps(
            self.request_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> EvaluationRequestV3:
        data = _required_mapping(value, "request")
        _reject_credentials(data)
        _exact_keys(
            data,
            {
                "protocol_version",
                "evaluation_id",
                "attempt_number",
                "king",
                "challenger",
                "miner",
                "versions",
                "sampling",
                "limits",
                "early_stopping",
                "dataset",
            },
            "request",
        )
        if data.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolValidationError(
                f"protocol_version must be {PROTOCOL_VERSION!r}"
            )
        evaluation_id = _required_string(data.get("evaluation_id"), "evaluation_id")
        if not _IDENTIFIER_RE.fullmatch(evaluation_id):
            raise ProtocolValidationError("evaluation_id contains unsupported characters")
        attempt_number = data.get("attempt_number")
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            raise ProtocolValidationError("attempt_number must be an integer >= 1")

        miner = _required_mapping(data.get("miner"), "miner")
        _exact_keys(miner, {"hotkey", "coldkey", "uid", "netuid", "challenge_id"}, "miner")
        _required_string(miner.get("hotkey"), "miner.hotkey")
        _required_string(miner.get("coldkey"), "miner.coldkey")
        for key in ("uid", "netuid"):
            if isinstance(miner.get(key), bool) or not isinstance(miner.get(key), int):
                raise ProtocolValidationError(f"miner.{key} must be an integer")

        versions = _required_mapping(data.get("versions"), "versions")
        _exact_keys(
            versions,
            {"evaluation_policy", "dataset", "code", "evaluator"},
            "versions",
        )
        normalized_versions = {
            key: _required_string(versions.get(key), f"versions.{key}")
            for key in ("evaluation_policy", "dataset", "code", "evaluator")
        }

        sampling = _required_mapping(data.get("sampling"), "sampling")
        _exact_keys(sampling, {"seed", "bootstrap_seed", "block_hash"}, "sampling")
        normalized_sampling: dict[str, int | str] = {}
        for key in ("seed", "bootstrap_seed"):
            number = sampling.get(key)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ProtocolValidationError(f"sampling.{key} must be an integer >= 0")
            normalized_sampling[key] = number
        block_hash = _required_string(sampling.get("block_hash"), "sampling.block_hash").lower()
        if not _BLOCK_HASH_RE.fullmatch(block_hash):
            raise ProtocolValidationError(
                "sampling.block_hash must be a lowercase 32-byte 0x-prefixed hash"
            )
        normalized_sampling["block_hash"] = block_hash

        limits = _required_mapping(data.get("limits"), "limits")
        _exact_keys(
            limits,
            {"batch_token_budget", "n_bootstrap", "alpha", "delta_threshold", "batch_size"},
            "limits",
        )
        normalized_limits: dict[str, int | float] = {}
        for key, minimum in (("batch_token_budget", 2), ("n_bootstrap", 1), ("batch_size", 1)):
            number = limits.get(key)
            if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
                raise ProtocolValidationError(f"limits.{key} must be an integer >= {minimum}")
            normalized_limits[key] = number
        if normalized_limits["batch_size"] > MAX_BATCH_SIZE:
            raise ProtocolValidationError(
                f"limits.batch_size must be <= {MAX_BATCH_SIZE}"
            )
        for key in ("alpha", "delta_threshold"):
            number = limits.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ProtocolValidationError(f"limits.{key} must be numeric")
            normalized_limits[key] = float(number)
        if not 0 < normalized_limits["alpha"] < 1:
            raise ProtocolValidationError("limits.alpha must be between 0 and 1")

        raw_early_stopping = data.get("early_stopping")
        if raw_early_stopping is None:
            early_stopping = EarlyStoppingPolicy()
            normalized_early_stopping = early_stopping.request_dict()
        else:
            early_stopping_data = _required_mapping(raw_early_stopping, "early_stopping")
            _exact_keys(
                early_stopping_data,
                {
                    "enabled",
                    "min_fraction",
                    "advantage_quantile",
                    "margin",
                    "check_interval",
                },
                "early_stopping",
            )
            try:
                early_stopping = EarlyStoppingPolicy.from_mapping(early_stopping_data)
            except ValueError as exc:
                raise ProtocolValidationError(str(exc)) from exc
            normalized_early_stopping = early_stopping.request_dict()

        if early_stopping.enabled:
            raise ProtocolValidationError("early stopping is disabled for reference-deficit evaluation")
        if not math.isfinite(normalized_limits["delta_threshold"]) or normalized_limits["delta_threshold"] < 0:
            raise ProtocolValidationError("delta_threshold must be finite and non-negative bits/byte")
        if normalized_limits["batch_token_budget"] > 1_000_000:
            raise ProtocolValidationError("batch_token_budget exceeds bound")
        try:
            normalized_dataset = deepcopy(validate_mix(dict(_required_mapping(data.get("dataset"), "dataset"))))
        except (ValueError, TypeError, KeyError) as exc:
            raise ProtocolValidationError(str(exc)) from exc

        normalized_payload = {
            "protocol_version": PROTOCOL_VERSION,
            "evaluation_id": evaluation_id,
            "attempt_number": attempt_number,
            "king": R2Artifact.from_mapping(data.get("king"), "king").public_dict(),
            "challenger": R2Artifact.from_mapping(
                data.get("challenger"), "challenger"
            ).public_dict(),
            "miner": dict(miner),
            "versions": normalized_versions,
            "sampling": normalized_sampling,
            "limits": normalized_limits,
            "dataset": normalized_dataset,
        }
        if raw_early_stopping is not None:
            normalized_payload["early_stopping"] = normalized_early_stopping
        return cls(
            evaluation_id=evaluation_id,
            attempt_number=attempt_number,
            king=R2Artifact.from_mapping(data.get("king"), "king"),
            challenger=R2Artifact.from_mapping(data.get("challenger"), "challenger"),
            miner=dict(miner),
            versions=normalized_versions,
            sampling=normalized_sampling,
            limits=normalized_limits,
            early_stopping=normalized_early_stopping,
            dataset=normalized_dataset,
            request_payload=normalized_payload,
        )


@dataclass
class EvaluationAttempt:
    request: EvaluationRequestV3
    state: str = "pending"
    progress: dict[str, Any] = field(default_factory=dict)
    verdict: dict[str, Any] | None = None
    error: str | None = None
    reason: str | None = None
    error_code: str | None = None
    created_at: float = 0.0
    events: Queue = field(default_factory=Queue)

    def response(self, *, duplicate: bool | None = None) -> dict[str, Any]:
        response: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "eval_id": self.request.eval_id,
            "evaluation_id": self.request.evaluation_id,
            "attempt_number": self.request.attempt_number,
            "request_sha256": self.request.request_sha256,
            "state": self.state,
        }
        if duplicate is not None:
            response["duplicate"] = duplicate
        if self.state == "completed":
            response["verdict"] = self.verdict
        elif self.state == "failed":
            response.update(
                {"error": self.error, "reason": self.reason, "error_code": self.error_code}
            )
        return response

    def event(self, event_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "eval_id": self.request.eval_id,
            "evaluation_id": self.request.evaluation_id,
            "attempt_number": self.request.attempt_number,
            "type": event_type,
            "data": dict(data),
        }


class EvaluationAttemptRegistry:
    """Thread-safe in-memory idempotency boundary for evaluator attempts."""

    def __init__(self) -> None:
        self._records: dict[str, EvaluationAttempt] = {}
        self._lock = threading.Lock()

    def start(
        self,
        request: EvaluationRequestV3,
        *,
        created_at: float = 0.0,
        admit_new: Callable[[], bool] | None = None,
    ) -> tuple[EvaluationAttempt, bool]:
        with self._lock:
            existing = self._records.get(request.eval_id)
            if existing is not None:
                if existing.request.request_sha256 != request.request_sha256:
                    raise AttemptConflictError(
                        "evaluation attempt ID is already bound to a different request"
                    )
                return existing, True
            if admit_new is not None and not admit_new():
                raise AttemptBusyError("an eval is already running")
            record = EvaluationAttempt(request=request, created_at=created_at)
            self._records[request.eval_id] = record
            return record, False

    def get(self, eval_id: str) -> EvaluationAttempt | None:
        with self._lock:
            return self._records.get(eval_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def active_count(self) -> int:
        with self._lock:
            return sum(
                record.state not in {"completed", "failed"}
                for record in self._records.values()
            )


def result_provenance(
    request: EvaluationRequestV3,
    *,
    started_at: str,
    completed_at: str,
    requested_sequences: int,
    completed_sequences: int,
    early_stopped: bool,
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_id": request.evaluation_id,
        "attempt_number": request.attempt_number,
        "request_sha256": request.request_sha256,
        "king_artifact_digest": request.king.expected_digest,
        "challenger_artifact_digest": request.challenger.expected_digest,
        "miner": dict(request.miner),
        "versions": dict(request.versions),
        "dataset_identity": dict(request.dataset),
        "sampling": {
            **dict(request.sampling),
            "requested_sequences": requested_sequences,
            "completed_sequences": completed_sequences,
            "early_stopped": early_stopped,
        },
        "configured_limits": dict(request.limits),
        "configured_early_stopping": dict(request.early_stopping),
        "started_at": started_at,
        "completed_at": completed_at,
        "hardware": dict(hardware),
    }


def validate_result_v3(result: Mapping[str, Any], request: EvaluationRequestV3) -> None:
    """Fail closed when a terminal evaluator result lacks required audit provenance."""
    from .reference_policy import POLICY_VERSION
    required = {
        "protocol_version",
        "evaluation_id",
        "attempt_number",
        "request_sha256",
        "king_artifact_digest",
        "challenger_artifact_digest",
        "miner",
        "versions",
        "dataset_identity",
        "sampling",
        "configured_limits",
        "started_at",
        "completed_at",
        "hardware",
        "accepted",
        "verdict",
        "mu_hat",
        "lcb",
        "delta_threshold",
        "avg_king_loss",
        "avg_challenger_loss",
        "wall_time_s",
        "result_artifact_sha256",
    }
    if "early_stopping" in request.request_payload:
        required.add("configured_early_stopping")
    missing = sorted(required - set(result))
    if missing:
        raise ProtocolValidationError(f"protocol v3 result is missing fields: {missing}")
    for name in ("mu_hat", "lcb", "delta_threshold", "avg_king_loss", "avg_challenger_loss", "avg_reference_loss"):
        value = result.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ProtocolValidationError(f"result {name} must be finite")
    if (result.get("scoring_policy") != POLICY_VERSION or result.get("loss_unit") != "bits/byte"
            or result.get("aggregation") != "byte-weighted" or result.get("retention_gates") is not False):
        raise ProtocolValidationError("result does not use the reference-deficit objective")
    if result["delta_threshold"] != request.limits["delta_threshold"]:
        raise ProtocolValidationError("result threshold differs from request")
    accepted = result["lcb"] > result["delta_threshold"]
    if result["accepted"] is not accepted or result["verdict"] != ("challenger" if accepted else "king"):
        raise ProtocolValidationError("result decision differs from its confidence bound")
    if result["dataset_identity"] != dict(request.dataset) or result["configured_limits"] != dict(request.limits):
        raise ProtocolValidationError("result dataset or limits differ from request")
    sampling = result["sampling"]
    if (not isinstance(sampling, Mapping) or sampling.get("early_stopped") is not False
            or type(sampling.get("requested_sequences")) is not int
            or sampling["requested_sequences"] <= 0
            or sampling.get("completed_sequences") != sampling["requested_sequences"]):
        raise ProtocolValidationError("result must score every selected document")
    expected_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_id": request.evaluation_id,
        "attempt_number": request.attempt_number,
        "request_sha256": request.request_sha256,
        "king_artifact_digest": request.king.expected_digest,
        "challenger_artifact_digest": request.challenger.expected_digest,
    }
    for field_name, expected in expected_identity.items():
        if result.get(field_name) != expected:
            raise ProtocolValidationError(
                f"protocol v3 result {field_name} does not match its request"
            )
    if result.get("versions") != dict(request.versions):
        raise ProtocolValidationError("protocol v3 result versions do not match its request")
    if "early_stopping" in request.request_payload and result.get(
        "configured_early_stopping"
    ) != dict(request.early_stopping):
        raise ProtocolValidationError(
            "protocol v3 result early-stopping policy does not match its request"
        )
    result_sampling = result.get("sampling")
    if isinstance(result_sampling, Mapping) and result_sampling.get("early_stopped"):
        if result.get("accepted") is not False or result.get("verdict") != "king":
            raise ProtocolValidationError(
                "protocol v3 early stopping may only reject a challenger"
            )
        if not request.early_stopping.get("enabled"):
            raise ProtocolValidationError(
                "protocol v3 result stopped early when the request disabled it"
            )
    artifact_digest = result.get("result_artifact_sha256")
    if not isinstance(artifact_digest, str) or not _DIGEST_RE.fullmatch(artifact_digest):
        raise ProtocolValidationError("result_artifact_sha256 must be a lowercase SHA-256 digest")
