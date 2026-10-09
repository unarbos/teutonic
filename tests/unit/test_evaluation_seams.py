from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from teutonic.evaluation import (
    EarlyStoppingPolicy,
    EvaluatorBusyError,
    EvaluatorConflictError,
    EvaluatorJobNotFoundError,
    HttpEvaluatorClient,
    build_failure_history_entry,
    build_verdict_history_entry,
    classify_eval_error,
    challenger_futility_decision,
    decide_model_copy,
    normalize_verdict,
    paired_bootstrap_verdict,
    provisional_paired_bootstrap,
    validate_config_lock,
)
from teutonic.validator.service import _evaluator_error_code


FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "evaluation_behavior_v1.json").read_text()
)


def legacy_fixture_projection(case, identity, now):
    """Frozen subset of the pre-extraction State projection used by these fixtures."""
    if case["kind"] == "verdict":
        verdict = {
            **case["input"],
            "challenge_id": identity["challenge_id"],
            "challenger_digest": identity["challenger_digest"],
        }
        king_loss = verdict.get("avg_king_loss", 0)
        challenger_loss = verdict.get("avg_challenger_loss", 0)
        return {
            "challenge_id": verdict.get("challenge_id"),
            "hotkey": identity["hotkey"],
            "uid": identity["uid"],
            "coldkey": identity["coldkey"],
            "challenger_repo": identity["challenger_repo"],
            "challenger_digest": verdict.get("challenger_digest", ""),
            "accepted": verdict.get("accepted", False),
            "verdict": verdict.get("verdict", "unknown"),
            "mu_hat": verdict.get("mu_hat", 0),
            "lcb": verdict.get("lcb", 0),
            "delta": verdict.get("delta", verdict.get("delta_threshold", 0)),
            "avg_king_loss": king_loss,
            "avg_challenger_loss": challenger_loss,
            "best_loss": min(king_loss, challenger_loss) if (king_loss or challenger_loss) else 0,
            "wall_time_s": verdict.get("wall_time_s", 0),
            "timestamp": verdict.get("timestamp", now()),
        }
    if case["kind"] == "failure":
        return {
            "challenge_id": identity["challenge_id"],
            "hotkey": identity["hotkey"],
            "uid": identity["uid"],
            "coldkey": identity["coldkey"],
            "challenger_repo": identity["challenger_repo"],
            "challenger_digest": identity["challenger_digest"],
            "accepted": False,
            "verdict": "error",
            "error_code": case["error_code"],
            "error_detail": case["error_detail"],
            "mu_hat": 0,
            "lcb": 0,
            "delta": 0,
            "avg_king_loss": 0,
            "avg_challenger_loss": 0,
            "best_loss": 0,
            "wall_time_s": 0,
            "timestamp": now(),
        }
    return None


def public_aggregate(entries):
    return {
        "accepted": sum(entry["accepted"] is True for entry in entries),
        "rejected": sum(entry["verdict"] == "king" for entry in entries),
        "failed": sum(entry["verdict"] == "error" for entry in entries),
    }


def legacy_bootstrap_verdict(policy_input, now):
    """Frozen pre-extraction implementation from the evaluator engine."""
    king_losses = policy_input["king_losses"]
    challenger_losses = policy_input["challenger_losses"]
    diff = np.asarray(king_losses, dtype=np.float64) - np.asarray(
        challenger_losses, dtype=np.float64
    )
    rng = np.random.default_rng(policy_input["bootstrap_seed"])
    boot = np.empty(policy_input["n_bootstrap"], dtype=np.float64)
    for i in range(policy_input["n_bootstrap"]):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot[i] = diff[idx].mean()
    mu_hat = float(diff.mean())
    lcb = float(np.quantile(boot, policy_input["alpha"]))
    accepted = lcb > policy_input["delta_threshold"]
    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta": policy_input["delta_threshold"],
        "delta_threshold": policy_input["delta_threshold"],
        "alpha": policy_input["alpha"],
        "n_bootstrap": policy_input["n_bootstrap"],
        "n_sequences": len(diff),
        "avg_king_loss": round(float(np.mean(king_losses)), 6),
        "avg_challenger_loss": round(float(np.mean(challenger_losses)), 6),
        "timestamp": now(),
    }


class EvaluationPolicyRegressionTests(unittest.TestCase):
    def test_evaluator_infrastructure_codes_remain_retryable_through_protocol(self):
        for code in ("dataset_index_download", "evaluator_interrupted"):
            with self.subTest(code=code):
                public_code = _evaluator_error_code({"code": code, "error": "private detail"})
                self.assertEqual(
                    classify_eval_error(RuntimeError(f"eval server error: {public_code}")),
                    (True, code),
                )

    def test_duplicate_safetensors_maps_to_model_copy_without_evaluator_change(self) -> None:
        self.assertEqual(
            _evaluator_error_code(
                {
                    "code": "evaluation_failed",
                    "error": "challenger .safetensors are identical to the king",
                }
            ),
            "model_copy",
        )
        self.assertEqual(
            classify_eval_error(RuntimeError("eval server error: model_copy")),
            (False, "model_copy"),
        )

    def test_evaluator_reuse_limit_code_supports_rolling_upgrade(self) -> None:
        self.assertEqual(
            _evaluator_error_code({"code": "safetensors_reuse_limit"}),
            "safetensors_reuse_limit",
        )
        self.assertEqual(
            _evaluator_error_code(
                {
                    "error": (
                        "challenger safetensors SHA-256 digest has already completed 3 evals; "
                        "maximum allowed is 3"
                    )
                }
            ),
            "safetensors_reuse_limit",
        )

    def test_safetensors_reuse_limit_has_a_stable_public_error_code(self) -> None:
        self.assertEqual(
            classify_eval_error("eval server error: safetensors_reuse_limit"),
            (False, "safetensors_reuse_limit"),
        )

    def test_provisional_bootstrap_uses_configured_count_and_marks_its_sample(self) -> None:
        king = [1.2, 1.1, 1.3, 1.25]
        challenger = [1.0, 1.05, 1.1, 1.15]
        progress = provisional_paired_bootstrap(
            king,
            challenger,
            bootstrap_seed=17,
            n_bootstrap=10_000,
            alpha=0.1,
            delta_threshold=0.05,
        )
        expected = paired_bootstrap_verdict(
            king,
            challenger,
            bootstrap_seed=17,
            n_bootstrap=10_000,
            alpha=0.1,
            delta_threshold=0.05,
        )
        self.assertEqual(
            progress,
            {
                "provisional_mu_hat": expected["mu_hat"],
                "provisional_lcb": expected["lcb"],
                "provisional_n_sequences": 4,
                "provisional_n_bootstrap": 10_000,
            },
        )

    def test_provisional_bootstrap_rejects_unpaired_or_empty_losses(self) -> None:
        parameters = {
            "bootstrap_seed": 17,
            "n_bootstrap": 128,
            "alpha": 0.1,
            "delta_threshold": 0.05,
        }
        with self.assertRaises(ValueError):
            provisional_paired_bootstrap([], [], **parameters)
        with self.assertRaises(ValueError):
            provisional_paired_bootstrap([1.0], [1.0, 2.0], **parameters)

    def test_early_stopping_is_one_sided_challenger_futility(self) -> None:
        policy = EarlyStoppingPolicy(
            enabled=True,
            min_fraction=0.4,
            advantage_quantile=0.95,
            margin=0.0,
            check_interval=100,
        )
        decision = challenger_futility_decision(
            [1.0] * 8,
            [2.0] * 8,
            total_sequences=20,
            delta_threshold=0.5,
            policy=policy,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision["mu_hat_upper_bound"], -1.0)

        # A clearly winning challenger never triggers this rejection-only rule.
        self.assertIsNone(
            challenger_futility_decision(
                [2.0] * 8,
                [1.0] * 8,
                total_sequences=20,
                delta_threshold=0.5,
                policy=policy,
            )
        )

    def test_early_stopping_margin_makes_rejection_more_conservative(self) -> None:
        losses = ([1.0] * 8, [1.4] * 8)
        aggressive = EarlyStoppingPolicy(enabled=True, margin=0.0)
        conservative = EarlyStoppingPolicy(enabled=True, margin=1.0)
        self.assertIsNotNone(
            challenger_futility_decision(
                *losses,
                total_sequences=20,
                delta_threshold=0.5,
                policy=aggressive,
            )
        )
        self.assertIsNone(
            challenger_futility_decision(
                *losses,
                total_sequences=20,
                delta_threshold=0.5,
                policy=conservative,
            )
        )

    def test_empty_message_transport_errors_are_retryable(self) -> None:
        cases = (
            (httpx.ConnectError(""), "connecterror"),
            (httpx.ReadError(""), "readerror"),
            (httpx.ReadTimeout(""), "timeout"),
            (httpx.RemoteProtocolError(""), "remoteprotocolerror"),
        )
        for error, expected_marker in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(classify_eval_error(error), (True, expected_marker))

    def test_legacy_behavior_fixtures(self) -> None:
        identity = FIXTURE["identity"]
        now = lambda: FIXTURE["fixed_now"]
        observed_names = set()
        legacy_entries = []
        extracted_entries = []
        for case in FIXTURE["cases"]:
            observed_names.add(case["name"])
            if case["kind"] == "verdict":
                normalized = normalize_verdict(
                    case["input"],
                    challenge_id=identity["challenge_id"],
                    challenger_digest=identity["challenger_digest"],
                )
                actual = build_verdict_history_entry(
                    normalized,
                    challenger_repo=identity["challenger_repo"],
                    hotkey=identity["hotkey"],
                    uid=identity["uid"],
                    coldkey=identity["coldkey"],
                    now=now,
                )
            elif case["kind"] == "failure":
                actual = build_failure_history_entry(
                    {
                        "challenge_id": identity["challenge_id"],
                        "hotkey": identity["hotkey"],
                        "model_repo": identity["challenger_repo"],
                        "model_digest": identity["challenger_digest"],
                    },
                    error_code=case["error_code"],
                    error_detail=case["error_detail"],
                    uid=identity["uid"],
                    coldkey=identity["coldkey"],
                    now=now,
                )
            else:
                transient, marker = classify_eval_error(case["input"])
                actual = {"transient": transient, "marker": marker}
            legacy = legacy_fixture_projection(case, identity, now)
            if legacy is not None:
                legacy_entries.append(legacy)
                extracted_entries.append(actual)
                self.assertEqual(actual, legacy, case["name"])
            for key, expected in case["expected"].items():
                self.assertEqual(actual[key], expected, f"{case['name']}:{key}")

            if case["name"] in {"accepted", "rejected"}:
                policy_input = case["policy_input"]
                policy_actual = paired_bootstrap_verdict(
                    policy_input["king_losses"],
                    policy_input["challenger_losses"],
                    bootstrap_seed=policy_input["bootstrap_seed"],
                    n_bootstrap=policy_input["n_bootstrap"],
                    alpha=policy_input["alpha"],
                    delta_threshold=policy_input["delta_threshold"],
                    now=now,
                )
                self.assertEqual(policy_actual, legacy_bootstrap_verdict(policy_input, now))
                for key, expected in case["policy_expected"].items():
                    self.assertEqual(policy_actual[key], expected, f"{case['name']}:policy:{key}")
            elif case["name"] == "copy":
                policy_actual = decide_model_copy(**case["policy_input"])
                self.assertIsNotNone(policy_actual)
                for key, expected in case["policy_expected"].items():
                    self.assertEqual(policy_actual[key], expected, f"copy:policy:{key}")
            elif case["name"] == "invalid_config":
                policy_actual = validate_config_lock(**case["policy_input"])
                self.assertEqual(policy_actual, case["policy_expected"])
        self.assertEqual(
            observed_names,
            {"accepted", "rejected", "copy", "invalid_config", "transient_error", "terminal_error"},
        )
        self.assertEqual(public_aggregate(extracted_entries), public_aggregate(legacy_entries))
        self.assertEqual(
            public_aggregate(extracted_entries),
            {"accepted": 1, "rejected": 1, "failed": 2},
        )

    def test_policy_package_is_independent_from_legacy_main_loop(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import teutonic.evaluation; "
                    "assert 'validator' not in sys.modules; "
                    "assert 'boto3' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_copy_policy_crowns_the_registry_observed_earlier_model(self) -> None:
        decision = decide_model_copy(
            challenger_repo="miner/original",
            challenger_digest="sha256:challenger",
            king_repo="king/copy",
            king_digest="sha256:king",
            challenger_info={
                "safetensor_layers": {"model.safetensors": "same"},
                "committed_at": datetime(2026, 8, 17, tzinfo=timezone.utc),
                "timestamp_source": "harbor_artifact.push_time",
            },
            king_info={
                "safetensor_layers": {"model.safetensors": "same"},
                "committed_at": "2026-08-18T00:00:00Z",
                "timestamp_source": "harbor_artifact.push_time",
            },
        )
        self.assertEqual(decision["action"], "crown_earlier")

    def test_config_lock_preserves_absent_value_marker(self) -> None:
        self.assertEqual(
            validate_config_lock({"vocab_size": 100}, {}),
            "vocab_size mismatch: king=100 challenger=<absent>",
        )


class HttpEvaluatorClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_start_status_and_stream_contract(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if request.url.path == "/eval" and request.method == "POST":
                return httpx.Response(200, json={"eval_id": "eval-1"})
            if request.url.path == "/eval/eval-1":
                return httpx.Response(200, json={"eval_id": "eval-1", "state": "running"})
            if request.url.path == "/eval/eval-1/stream":
                return httpx.Response(
                    200,
                    text='data: {"type":"progress","data":{"done":1}}\n\n'
                    'data: {"type":"verdict","data":{"accepted":true}}\n\n',
                )
            raise AssertionError(request.url.path)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            async with HttpEvaluatorClient("http://evaluator", client=transport) as client:
                self.assertEqual(await client.health(), {"status": "ok"})
                eval_id = await client.start({"king_repo": "king", "challenger_repo": "challenger"})
                self.assertEqual(eval_id, "eval-1")
                self.assertEqual((await client.status(eval_id))["state"], "running")
                events = [event async for event in client.events(eval_id)]
        self.assertEqual([event["type"] for event in events], ["progress", "verdict"])
        self.assertEqual(
            requests,
            [
                ("GET", "/health"),
                ("POST", "/eval"),
                ("GET", "/eval/eval-1"),
                ("GET", "/eval/eval-1/stream"),
            ],
        )

    async def test_busy_start_has_a_stable_error(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(409, json={"detail": "busy"})
        )
        async with httpx.AsyncClient(transport=transport) as raw_client:
            async with HttpEvaluatorClient("http://evaluator", client=raw_client) as client:
                with self.assertRaises(EvaluatorBusyError):
                    await client.start({})

    async def test_conflicting_and_lost_attempts_have_distinct_errors(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    409,
                    json={"detail": {"code": "attempt_conflict"}},
                )
            return httpx.Response(404, json={"detail": "eval not found"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as raw_client:
            async with HttpEvaluatorClient("http://evaluator", client=raw_client) as client:
                with self.assertRaises(EvaluatorConflictError):
                    await client.start({})
                with self.assertRaises(EvaluatorJobNotFoundError):
                    await client.status("lost-evaluation:1")
                with self.assertRaises(EvaluatorJobNotFoundError):
                    _ = [event async for event in client.events("lost-evaluation:1")]
