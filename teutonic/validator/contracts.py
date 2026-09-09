from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping

from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    EvaluationSettings,
    pretokenized_dataset_request,
)
from teutonic.evaluation.early_stopping import EarlyStoppingPolicy
from teutonic.evaluation.protocol_v2 import DEFAULT_EVAL_BATCH_SIZE, MAX_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class EvaluationPolicyConfig:
    policy_version: str
    code_version: str
    dataset_version: str
    evaluator_version: str
    sampling_seed: int
    bootstrap_seed: int
    n: int
    seq_len: int
    n_bootstrap: int
    alpha: float
    delta_threshold: float
    dataset_source: str
    dataset_label: str
    shards_per_dataset: int
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    dataset_manifests: tuple[DatasetManifestSnapshot, ...] = ()
    early_stopping: EarlyStoppingPolicy = field(default_factory=EarlyStoppingPolicy)
    lease: timedelta = timedelta(minutes=2)
    retry_base_delay: timedelta = timedelta(seconds=30)
    max_attempts: int = 3
    publish_non_winning_models: bool = False

    def __post_init__(self) -> None:
        if not all(
            (
                self.policy_version,
                self.code_version,
                self.dataset_version,
                self.evaluator_version,
            )
        ):
            raise ValueError("all evaluation version identities are required")
        if self.lease <= timedelta(0) or self.retry_base_delay < timedelta(0):
            raise ValueError("lease must be positive and retry delay cannot be negative")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not 1 <= self.batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}]")
        if self.early_stopping.enabled and self.early_stopping.check_interval > self.n:
            raise ValueError("early stopping check_interval cannot exceed evaluation n")
        if self.dataset_source != "pretokenized_npy" or not self.dataset_manifests:
            raise ValueError("evaluation needs database-backed pre-tokenized manifests")
        if self.shards_per_dataset < 1:
            raise ValueError("shards_per_dataset must be positive")

    @property
    def thresholds(self) -> dict[str, int | float]:
        return {
            "n": self.n,
            "seq_len": self.seq_len,
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
            "delta_threshold": self.delta_threshold,
            "batch_size": self.batch_size,
        }

    @property
    def persisted_thresholds(self) -> dict[str, Any]:
        return {
            **self.thresholds,
            "early_stopping": self.early_stopping.request_dict(),
        }

    def dataset_request(self, *, block_hash: str, hotkey: str) -> dict[str, Any]:
        settings = EvaluationSettings(
            config_version=self.dataset_version,
            dataset_label=self.dataset_label,
            n=self.n,
            delta_threshold=self.delta_threshold,
            manifests=self.dataset_manifests,
            shards_per_dataset=self.shards_per_dataset,
        )
        return pretokenized_dataset_request(
            settings,
            block_hash=block_hash,
            hotkey=hotkey,
            seq_len=self.seq_len,
        )

    def retry_delay(self, attempt_number: int) -> timedelta:
        exponent = max(0, min(attempt_number - 1, 8))
        return self.retry_base_delay * (2**exponent)


@dataclass(frozen=True, slots=True)
class ClaimedEvaluation:
    evaluation_id: str
    upload_id: str
    attempt_number: int
    competition_id: str
    claimed_king_reign_id: str
    request: Mapping[str, Any]

    @property
    def eval_id(self) -> str:
        return f"{self.evaluation_id}:{self.attempt_number}"


@dataclass(frozen=True, slots=True)
class RecoveryCandidate:
    evaluation_id: str
    upload_id: str
    attempt_number: int
    state: str
    evaluator_job_id: str
    owner_instance_id: str | None
    competition_id: str
    claimed_king_reign_id: str
    request: Mapping[str, Any]
