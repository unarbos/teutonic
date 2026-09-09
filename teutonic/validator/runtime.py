from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from teutonic.evaluation.configuration import EvaluationSettings
from teutonic.evaluation.early_stopping import EarlyStoppingPolicy
from teutonic.evaluation.protocol_v2 import DEFAULT_EVAL_BATCH_SIZE

from .contracts import EvaluationPolicyConfig


def _required(source: Mapping[str, str], name: str) -> str:
    value = source.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def _boolean(source: Mapping[str, str], name: str, default: str = "false") -> bool:
    value = source.get(name, default).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def evaluation_policy_from_env(
    source: Mapping[str, str] | None = None,
    *,
    settings: EvaluationSettings | None = None,
    early_stopping: EarlyStoppingPolicy | None = None,
) -> EvaluationPolicyConfig:
    env = os.environ if source is None else source
    if settings is None:
        raise RuntimeError("active PostgreSQL evaluation configuration is required")
    return EvaluationPolicyConfig(
        policy_version=_required(env, "TEUTONIC_EVALUATION_POLICY_VERSION"),
        code_version=_required(env, "TEUTONIC_EVALUATOR_CODE_VERSION"),
        dataset_version=settings.config_version,
        evaluator_version=_required(env, "TEUTONIC_EVALUATOR_VERSION"),
        sampling_seed=0,
        bootstrap_seed=int(env.get("TEUTONIC_EVAL_BOOTSTRAP_SEED", "0")),
        n=settings.n,
        seq_len=int(env.get("TEUTONIC_EVAL_SEQ_LEN", "2048")),
        n_bootstrap=int(env.get("TEUTONIC_EVAL_BOOTSTRAP_B", "10000")),
        alpha=float(env.get("TEUTONIC_EVAL_ALPHA", "0.001")),
        delta_threshold=settings.delta_threshold,
        dataset_source="pretokenized_npy",
        dataset_label=settings.dataset_label,
        shards_per_dataset=settings.shards_per_dataset,
        batch_size=int(
            env.get("TEUTONIC_EVAL_BATCH_SIZE", str(DEFAULT_EVAL_BATCH_SIZE))
        ),
        dataset_manifests=settings.manifests,
        early_stopping=early_stopping or EarlyStoppingPolicy(),
        lease=timedelta(seconds=int(env.get("TEUTONIC_EVALUATION_LEASE_SECONDS", "120"))),
        retry_base_delay=timedelta(
            seconds=int(env.get("TEUTONIC_EVALUATION_RETRY_SECONDS", "30"))
        ),
        max_attempts=int(env.get("TEUTONIC_EVALUATION_MAX_ATTEMPTS", "3")),
        publish_non_winning_models=_boolean(
            env, "TEUTONIC_PUBLISH_NON_WINNING_MODELS"
        ),
    )


@dataclass(frozen=True, slots=True)
class FinalizedMetagraph:
    block: int
    uid_by_hotkey: Mapping[str, int]


class BittensorFinalizedMetagraphReader:
    """Read chain state without constructing or loading a signing wallet."""

    def __init__(self, *, network: str, netuid: int) -> None:
        import bittensor as bt

        self.netuid = netuid
        self.subtensor = bt.Subtensor(network=network)
        self._cached: FinalizedMetagraph | None = None

    def snapshot(self) -> FinalizedMetagraph:
        finalized_hash = self.subtensor.substrate.get_chain_finalised_head()
        block = _as_int(self.subtensor.substrate.get_block_number(finalized_hash))
        if self._cached is not None and self._cached.block == block:
            return self._cached
        metagraph = self.subtensor.metagraph(self.netuid, block=block, lite=True)
        self._cached = FinalizedMetagraph(
            block=block,
            uid_by_hotkey={str(hotkey): uid for uid, hotkey in enumerate(metagraph.hotkeys)},
        )
        return self._cached


def equal_weight_plan(
    ordered_hotkeys: Sequence[str], uid_by_hotkey: Mapping[str, int], *, burn_uid: int
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[float, ...]]:
    target_hotkeys: list[str] = []
    target_uids: list[int] = []
    for hotkey in ordered_hotkeys:
        uid = uid_by_hotkey.get(hotkey)
        if uid is not None and uid not in target_uids:
            target_hotkeys.append(hotkey)
            target_uids.append(int(uid))
    if not target_uids:
        target_hotkeys = [f"burn:uid:{burn_uid}"]
        target_uids = [burn_uid]
    share = 1.0 / len(target_uids)
    return (
        tuple(target_hotkeys),
        tuple(target_uids),
        tuple(share for _uid in target_uids),
    )


class CrownCoordinator:
    def __init__(
        self,
        repository: Any,
        chain: Any,
        *,
        burn_uid: int = 0,
        king_chain_size: int = 5,
    ) -> None:
        if burn_uid < 0 or king_chain_size < 1:
            raise ValueError("crown weight policy is invalid")
        self.repository = repository
        self.chain = chain
        self.burn_uid = burn_uid
        self.king_chain_size = king_chain_size

    def __call__(self, promotion_id: str) -> str | None:
        hotkeys = self.repository.promotion_weight_hotkeys(
            promotion_id, limit=self.king_chain_size
        )
        snapshot = self.chain.snapshot()
        target_hotkeys, target_uids, weights = equal_weight_plan(
            hotkeys, snapshot.uid_by_hotkey, burn_uid=self.burn_uid
        )
        return self.repository.crown_promoted_winner(
            promotion_id,
            now=datetime.now(timezone.utc),
            crowned_finalized_block=snapshot.block,
            policy_hotkeys=hotkeys,
            target_hotkeys=target_hotkeys,
            target_uids=target_uids,
            normalized_weights=weights,
        )

    def reconcile_current_weight_plan(self) -> bool:
        policy = self.repository.current_weight_policy()
        if policy is None:
            return False
        snapshot = self.chain.snapshot()
        if snapshot.block <= policy["mapping_finalized_block"]:
            return False
        target_hotkeys, target_uids, weights = equal_weight_plan(
            policy["policy_hotkeys"], snapshot.uid_by_hotkey, burn_uid=self.burn_uid
        )
        return self.repository.refresh_current_weight_plan(
            publication_id=policy["publication_id"],
            expected_revision=policy["payload_revision"],
            mapping_finalized_block=snapshot.block,
            target_hotkeys=target_hotkeys,
            target_uids=target_uids,
            normalized_weights=weights,
            now=datetime.now(timezone.utc),
        )
