from __future__ import annotations

import unittest
import hashlib
from datetime import timedelta

from teutonic.evaluation import EarlyStoppingPolicy
from teutonic.evaluation.configuration import DatasetManifestSnapshot, canonical_manifest_bytes
from teutonic.promotion import promotion_worker_lock_key
from teutonic.validator import EvaluationPolicyConfig, scheduler_lock_key
from teutonic.validator.repository import SchedulerInvariantError, _bounded_progress


class ValidatorSchedulerPolicyTests(unittest.TestCase):
    def test_provisional_progress_metrics_are_bounded_and_private_fields_are_dropped(self):
        progress = _bounded_progress(
            {
                "phase": "eval_progress",
                "done": 16,
                "total": 32,
                "provisional_mu_hat": 0.003,
                "provisional_lcb": 0.002,
                "provisional_n_sequences": 16,
                "provisional_n_bootstrap": 128,
                "private_worker_hostname": "never-persist",
            }
        )
        self.assertEqual(progress["provisional_lcb"], 0.002)
        self.assertEqual(progress["provisional_n_sequences"], 16)
        self.assertNotIn("private_worker_hostname", progress)
        with self.assertRaises(SchedulerInvariantError):
            _bounded_progress({"provisional_lcb": float("nan")})

    def _policy(self, **overrides):
        manifest = {
            "shards": [{
                "key": "shards/part-000.npy",
                "sha256": "a" * 64,
                "size_bytes": 16384,
                "n_tokens": 16384,
            }]
        }
        snapshot = DatasetManifestSnapshot(
            name="fixture",
            manifest_url="https://datasets.example/fixture/manifest.json",
            manifest_sha256=hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
            proportion=1.0,
            manifest=manifest,
        )
        values = {
            "policy_version": "policy-v1",
            "code_version": "code-v1",
            "dataset_version": "b" * 64,
            "evaluator_version": "evaluator-v2",
            "sampling_seed": 1,
            "bootstrap_seed": 2,
            "n": 32,
            "seq_len": 64,
            "n_bootstrap": 100,
            "alpha": 0.05,
            "delta_threshold": 0.0015,
            "dataset_source": "pretokenized_npy",
            "dataset_label": "fixture-v1",
            "shards_per_dataset": 4,
            "dataset_manifests": (snapshot,),
            "retry_base_delay": timedelta(seconds=5),
        }
        values.update(overrides)
        return EvaluationPolicyConfig(**values)

    def test_retry_backoff_is_bounded_and_deterministic(self) -> None:
        policy = self._policy()
        self.assertEqual(policy.retry_delay(1), timedelta(seconds=5))
        self.assertEqual(policy.retry_delay(3), timedelta(seconds=20))
        self.assertEqual(policy.retry_delay(100), timedelta(seconds=1280))

    def test_scheduler_lock_key_scopes_network_generation_and_competition(self) -> None:
        first = scheduler_lock_key(306, "test", "quasar")
        self.assertEqual(first, scheduler_lock_key(306, "test", "quasar"))
        self.assertNotEqual(first, scheduler_lock_key(307, "test", "quasar"))
        self.assertNotEqual(first, scheduler_lock_key(306, "test-2", "quasar"))
        self.assertNotEqual(first, scheduler_lock_key(306, "test", "mimo"))

    def test_promotion_worker_uses_an_independent_lock(self) -> None:
        promotion = promotion_worker_lock_key(306, "test", "quasar")
        self.assertEqual(
            promotion,
            promotion_worker_lock_key(306, "test", "quasar"),
        )
        self.assertNotEqual(promotion, scheduler_lock_key(306, "test", "quasar"))

    def test_invalid_policy_configuration_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self._policy(max_attempts=0)
        with self.assertRaises(ValueError):
            self._policy(dataset_source="unknown")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            self._policy(batch_size=0)
        with self.assertRaisesRegex(ValueError, "check_interval"):
            self._policy(
                n=32,
                early_stopping=EarlyStoppingPolicy(enabled=True, check_interval=100),
            )
