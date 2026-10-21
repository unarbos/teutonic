from __future__ import annotations

import io
import json
import math
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import httpx
from botocore.exceptions import ClientError

from teutonic.dashboard.contracts import (
    DashboardContractError,
    canonical_dashboard_json,
    canonical_dataset_manifest_json,
    validate_dashboard,
)
from teutonic.dashboard.market import (
    COINGECKO_PRICE_URL,
    KEYLESS_MARKET_SOURCE,
    KeylessMarketClient,
    MarketClient,
    MarketDataError,
    select_market,
)
from teutonic.dashboard.storage import DashboardObjectStore
from teutonic.dashboard.service import DashboardViewService

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
DIGEST = "a" * 64


def payload() -> dict:
    stamp = "2026-08-18T12:00:00Z"
    return {
        "schema_version": 1,
        "publication_id": "11111111-1111-4111-8111-111111111111",
        "generated_at": stamp,
        "updated_at": stamp,
        "source_watermark": 7,
        "chain": {
            "name": "Teutonic",
            "netuid": 306,
            "generation": "test",
            "competition": "quasar",
            "seed_repo": "owner/genesis",
            "seed_digest": "hf:" + "b" * 40,
            "seed_repo_backend": "hf",
            "finalized_start_block": None,
            "last_finalized_block": None,
            "observed_at": None,
        },
        "king": None,
        "king_payout": {"weight": None, "alpha_per_hour": None, "usd_per_hour": None},
        "king_chain": [],
        "stats": {
            "active_registrations": 0,
            "queue_depth": 0,
            "completed_evaluations": 0,
            "reign_count": 0,
        },
        "current_eval": None,
        "queue": [],
        "history": [],
        "dataset_versions": [],
        "weight_status": {
            "state": None,
            "cadence_blocks": None,
            "last_attempted_block": None,
            "next_due_block": None,
            "latest_attempt_state": None,
            "latest_finalized_block": None,
            "error_code": None,
            "requested_at": None,
            "submitted_at": None,
            "finalized_at": None,
        },
        "service_status": {
            "overall": "offline",
            "validator_phase": None,
            "validator_heartbeat_age_seconds": None,
            "current_evaluation_age_seconds": None,
            "weight_delayed": False,
            "services": [],
        },
        "market": None,
    }


def market(*, fetched_at: str = "2026-08-18T12:00:00Z", stale: bool = False) -> dict:
    return {
        "source": "market-fixture-v1",
        "fetched_at": fetched_at,
        "stale": stale,
        "tao_price_usd": 412.5,
        "tao_change_24h": -1.25,
        "sn3_alpha_price_tao": 0.031,
        "sn3_reg_burn_tao": 0.5,
    }


def dataset_manifest() -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-08-18T12:00:00Z",
        "chain": {
            "name": "Teutonic",
            "netuid": 306,
            "generation": "test",
            "competition": "quasar",
        },
        "config_version": "a" * 64,
        "dataset_label": "fixture",
        "eval_n": 2000,
        "delta_threshold": 0.5,
        "sampling": {
            "algorithm": "blake2b-64-block-hash-hotkey-v1",
            "inputs": ["block_hash", "hotkey"],
        },
        "sources": [{
            "name": "fixture",
            "proportion": 1.0,
            "manifest_url": "https://datasets.example/fixture/manifest.json",
            "manifest_sha256": "b" * 64,
            "source_repo": "owner/dataset",
            "tokenizer": "owner/tokenizer",
            "dtype": "uint32",
            "tokenization_mode": "seq_packed_shards",
            "sequence_length": 2048,
            "total_tokens": 2_048_000,
            "total_shards": 4,
            "estimated_sequences": 1000,
        }],
    }


class FakeS3:
    def __init__(self) -> None:
        self.objects = {}
        self.put_calls = 0

    @staticmethod
    def _missing():
        return ClientError({"Error": {"Code": "NoSuchKey"}}, "HeadObject")

    def head_object(self, *, Bucket, Key):
        record = self.objects.get((Bucket, Key))
        if record is None:
            raise self._missing()
        return {
            "ContentLength": len(record["Body"]),
            "Metadata": dict(record["Metadata"]),
        }

    def get_object(self, *, Bucket, Key):
        record = self.objects.get((Bucket, Key))
        if record is None:
            raise self._missing()
        return {"Body": io.BytesIO(record["Body"])}

    def put_object(self, **kwargs):
        self.put_calls += 1
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = dict(kwargs)
        return {"ETag": "fixture"}


class DashboardContractTests(unittest.TestCase):
    def test_empty_payload_is_strict_and_valid(self):
        body = canonical_dashboard_json(payload())
        self.assertNotIn(b"NaN", body)
        self.assertEqual(json.loads(body)["schema_version"], 1)

    def test_current_evaluation_accepts_provisional_bootstrap_metrics(self):
        fixture = payload()
        fixture["current_eval"] = {
            "challenge_id": "0123456789abcdef",
            "model_digest": "c" * 64,
            "hotkey": "public-hotkey",
            "coldkey": "public-coldkey",
            "uid": 7,
            "model_identity": "hidden_until_promotion",
            "stage": "eval_progress",
            "progress": 400,
            "total": 2000,
            "percent": 20.0,
            "elapsed_seconds": 120.0,
            "early_stopped": False,
            "policy_version": "policy-v1",
            "dataset_version": "dataset-v1",
            "started_at": "2026-08-18T11:58:00Z",
            "last_progress_at": "2026-08-18T12:00:00Z",
            "provisional_mu_hat": 0.72,
            "provisional_lcb": 0.61,
            "provisional_n_sequences": 400,
            "provisional_n_bootstrap": 1000,
            "delta_threshold": 0.5,
        }
        parsed = json.loads(canonical_dashboard_json(fixture))
        self.assertEqual(parsed["current_eval"]["provisional_lcb"], 0.61)

    def test_queue_entry_exposes_the_model_digest(self):
        fixture = payload()
        fixture["queue"] = [{
            "challenge_id": "0123456789abcdef",
            "model_digest": "d" * 64,
            "hotkey": "public-hotkey",
            "coldkey": "public-coldkey",
            "uid": 7,
            "model_identity": "hidden_until_promotion",
            "block": 123,
            "queue_position": 1,
            "state": "queued",
            "submitted_at": "2026-08-18T12:00:00Z",
        }]
        parsed = json.loads(canonical_dashboard_json(fixture))
        self.assertEqual(parsed["queue"][0]["model_digest"], "d" * 64)

        fixture["queue"][0]["model_digest"] = None
        canonical_dashboard_json(fixture)

        del fixture["queue"][0]["model_digest"]
        with self.assertRaises(DashboardContractError):
            canonical_dashboard_json(fixture)

    def test_dashboard_accepts_sanitized_dataset_version_catalog(self):
        fixture = payload()
        fixture["dataset_versions"] = [{
            "config_version": "a" * 64,
            "dataset_label": "fixture-v1",
            "eval_n": 2000,
            "delta_threshold": 0.5,
            "created_at": "2026-08-18T12:00:00Z",
            "sources": dataset_manifest()["sources"],
        }]
        parsed = json.loads(canonical_dashboard_json(fixture))
        self.assertEqual(parsed["dataset_versions"][0]["sources"][0]["name"], "fixture")
        source = fixture["dataset_versions"][0]["sources"][0]
        source["dataset"] = "dclm-baseline-1.0"
        parsed = json.loads(canonical_dashboard_json(fixture))
        self.assertEqual(
            parsed["dataset_versions"][0]["sources"][0]["dataset"],
            "dclm-baseline-1.0",
        )
        for invalid_dataset in ("", None, 123):
            source["dataset"] = invalid_dataset
            with self.assertRaises(DashboardContractError):
                canonical_dashboard_json(fixture)
        source["dataset"] = "dclm-baseline-1.0"
        fixture["dataset_versions"][0]["private_manifest"] = {"secret": "never"}
        with self.assertRaises(DashboardContractError):
            canonical_dashboard_json(fixture)

    def test_global_dataset_manifest_is_canonical_and_strict(self):
        body = canonical_dataset_manifest_json(dataset_manifest())
        parsed = json.loads(body)
        self.assertEqual(parsed["eval_n"], 2000)
        self.assertEqual(parsed["sources"][0]["total_tokens"], 2_048_000)
        self.assertEqual(parsed["sources"][0]["estimated_sequences"], 1000)
        self.assertNotIn("manifest", parsed["sources"][0])
        self.assertNotIn("shards", parsed["sources"][0])
        invalid = dataset_manifest()
        invalid["private_credentials"] = "never"
        with self.assertRaises(DashboardContractError):
            canonical_dataset_manifest_json(invalid)
        invalid = dataset_manifest()
        invalid["sources"][0]["manifest"] = {"shards": []}
        with self.assertRaises(DashboardContractError):
            canonical_dataset_manifest_json(invalid)

    def test_non_finite_and_unknown_fields_are_rejected(self):
        invalid = payload()
        invalid["king_payout"]["weight"] = math.inf
        with self.assertRaises(DashboardContractError):
            canonical_dashboard_json(invalid)
        invalid = payload()
        invalid["private_bucket"] = "teutonic-private-models"
        with self.assertRaises(DashboardContractError):
            validate_dashboard(invalid)

    def test_public_fixture_has_no_secret_bearing_shapes(self):
        fixture = payload()
        fixture["history"] = [
            {
                "challenge_id": "0123456789abcdef",
                "hotkey": "public-hotkey",
                "coldkey": "public-coldkey",
                "uid": 7,
                "baseline_hotkey": "king-hotkey",
                "baseline_coldkey": None,
                "baseline_uid": 0,
                "verdict": "error",
                "accepted": False,
                "mu_hat": None,
                "lcb": None,
                "delta": None,
                "avg_king_loss": None,
                "avg_challenger_loss": None,
                "wall_time_s": None,
                "n_sequences_evaluated": None,
                "n_sequences": None,
                "early_stopped": False,
                "shards_used": [],
                "source_scores": [],
                "error_code": "evaluation_failed",
                "error_message": "The evaluation could not be completed.",
                "policy_version": "policy-v1",
                "dataset_version": "dataset-v1",
                "timestamp": "2026-08-18T12:00:00Z",
                "challenger_repo": None,
                "challenger_digest": None,
                "model_reference": None,
                "publication_disposition": None,
                "model_identity": "hidden_until_promotion",
            }
        ]
        upload_failure = deepcopy(fixture["history"][0])
        upload_failure.update(
            {
                "uid": 170,
                "error_code": "ArtifactIntegrityError",
                "error_message": "The uploaded model artifacts failed integrity verification.",
                "policy_version": None,
                "dataset_version": None,
                "registration_state": "active",
                "upload_id": "9452d08b-b3bb-4bc9-9c45-01889fff6fa8",
                "upload_state": "verification_failed",
            }
        )
        fixture["history"].append(upload_failure)
        text = canonical_dashboard_json(fixture).decode()
        self.assertIn("ArtifactIntegrityError", text)
        self.assertIn("9452d08b-b3bb-4bc9-9c45-01889fff6fa8", text)
        forbidden = (
            "secret_access_key",
            "access_key_id",
            "parent_token",
            "private-models",
            "models/registrations/",
            "immutable_bucket",
            "traceback",
            "http://validator-internal",
        )
        for marker in forbidden:
            self.assertNotIn(marker, text.lower())


class MarketTests(unittest.TestCase):
    def test_fetch_validates_source_shape_ranges_and_timestamp(self):
        response = httpx.Response(200, json=market())
        client = MarketClient(
            "https://market.example/v1/dashboard",
            source="market-fixture-v1",
            transport=httpx.MockTransport(lambda _request: response),
        )
        self.assertFalse(client.fetch(now=NOW)["stale"])
        bad = market()
        bad["tao_price_usd"] = -1
        client = MarketClient(
            "https://market.example/v1/dashboard",
            source="market-fixture-v1",
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=bad)),
        )
        with self.assertRaises(MarketDataError):
            client.fetch(now=NOW)

    def test_keyless_client_combines_coingecko_and_subnet_chain_data(self):
        requests = []

        def handler(request):
            requests.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "bittensor": {
                        "usd": 192.10,
                        "usd_24h_change": -0.53,
                        "last_updated_at": int(NOW.timestamp()),
                    }
                },
            )

        class Balance:
            def __init__(self, tao):
                self.tao = tao

        class Subtensor:
            def __init__(self):
                self.price_calls = []
                self.recycle_calls = []

            def get_subnet_price(self, netuid):
                self.price_calls.append(netuid)
                return Balance(0.0276)

            def recycle(self, netuid):
                self.recycle_calls.append(netuid)
                return Balance(0.001525)

        subtensor = Subtensor()
        client = KeylessMarketClient(
            netuid=3,
            transport=httpx.MockTransport(handler),
            subtensor=subtensor,
        )
        first = client.fetch(now=NOW)
        cached = client.fetch(now=NOW + timedelta(seconds=30))

        self.assertEqual(requests, [COINGECKO_PRICE_URL])
        self.assertEqual(subtensor.price_calls, [3])
        self.assertEqual(subtensor.recycle_calls, [3])
        self.assertEqual(first, cached)
        self.assertEqual(first["source"], KEYLESS_MARKET_SOURCE)
        self.assertEqual(first["tao_price_usd"], 192.10)
        self.assertEqual(first["tao_change_24h"], -0.53)
        self.assertEqual(first["sn3_alpha_price_tao"], 0.0276)
        self.assertEqual(first["sn3_reg_burn_tao"], 0.001525)

    def test_keyless_client_rejects_stale_coingecko_quote(self):
        response = httpx.Response(
            200,
            json={
                "bittensor": {
                    "usd": 192.10,
                    "usd_24h_change": -0.53,
                    "last_updated_at": int((NOW - timedelta(minutes=11)).timestamp()),
                }
            },
        )

        class Subtensor:
            def get_subnet_price(self, _netuid):
                return 0.0276

            def recycle(self, _netuid):
                return 0.001525

        client = KeylessMarketClient(
            netuid=3,
            transport=httpx.MockTransport(lambda _request: response),
            subtensor=Subtensor(),
        )
        with self.assertRaisesRegex(MarketDataError, "already stale"):
            client.fetch(now=NOW)

    def test_keyless_client_converts_last_tempo_emission_to_alpha_per_hour(self):
        class Balance:
            def __init__(self, tao):
                self.tao = tao

        class MetagraphInfo:
            tempo = 360
            hotkeys = ["king-hotkey", "prior-hotkey"]
            emission = [Balance(29.520164689), Balance(0)]

        class Subtensor:
            def __init__(self):
                self.calls = []

            def get_metagraph_info(self, netuid):
                self.calls.append(netuid)
                return MetagraphInfo()

        subtensor = Subtensor()
        client = KeylessMarketClient(netuid=3, subtensor=subtensor)
        first = client.fetch_payouts(["king-hotkey", "missing"], now=NOW)
        cached = client.fetch_payouts(["king-hotkey"], now=NOW + timedelta(seconds=30))

        self.assertEqual(subtensor.calls, [3])
        self.assertAlmostEqual(first["king-hotkey"], 24.600137240833333)
        self.assertNotIn("missing", first)
        self.assertEqual(first, cached)

    def test_failure_reuses_bounded_stale_market_then_expires_it(self):
        previous = market(fetched_at="2026-08-18T11:30:00Z")
        selected = select_market(None, previous, now=NOW, maximum_stale=timedelta(hours=1))
        self.assertTrue(selected["stale"])
        self.assertIsNone(
            select_market(
                None,
                previous,
                now=NOW + timedelta(hours=1),
                maximum_stale=timedelta(hours=1),
            )
        )


class DashboardStorageTests(unittest.TestCase):
    def test_complete_single_put_is_verified_and_unchanged_is_skipped(self):
        s3 = FakeS3()
        store = DashboardObjectStore(s3, bucket="teutonic-dash")
        body = canonical_dashboard_json(payload())
        first = store.publish(body, source_watermark=7)
        second = store.publish(body, source_watermark=7)
        self.assertEqual(first.state, "published")
        self.assertEqual(second.state, "unchanged")
        self.assertEqual(s3.put_calls, 1)
        stored = s3.objects[("teutonic-dash", "dashboard.json")]
        self.assertEqual(stored["ContentType"], "application/json; charset=utf-8")
        self.assertEqual(store.previous_payload()["source_watermark"], 7)

    def test_older_watermark_cannot_replace_newer_object(self):
        s3 = FakeS3()
        store = DashboardObjectStore(s3, bucket="teutonic-dash")
        body = canonical_dashboard_json(payload())
        store.publish(body, source_watermark=8)
        older = deepcopy(payload())
        older["source_watermark"] = 7
        result = store.publish(canonical_dashboard_json(older), source_watermark=7)
        self.assertEqual(result.state, "stale_skipped")
        self.assertEqual(s3.put_calls, 1)

    def test_global_dataset_manifest_is_published_to_stable_public_key(self):
        s3 = FakeS3()
        store = DashboardObjectStore(s3, bucket="teutonic-dash")
        body = canonical_dataset_manifest_json(dataset_manifest())
        first = store.publish_dataset_manifest(body, config_version="a" * 64)
        second = store.publish_dataset_manifest(body, config_version="a" * 64)
        self.assertEqual(first.state, "published")
        self.assertEqual(second.state, "unchanged")
        stored = s3.objects[("teutonic-dash", "datasets/manifest.json")]
        self.assertEqual(stored["Metadata"]["config-version"], "a" * 64)


class DashboardServiceTests(unittest.TestCase):
    def test_chain_emissions_populate_hourly_alpha_and_usd_payouts(self):
        class PayoutMarket:
            def fetch_payouts(self, hotkeys, *, now):
                self.hotkeys = hotkeys
                self.now = now
                return {"king-hotkey": 24.6}

        candidate = {
            "market": market(),
            "king": {"hotkey": "king-hotkey"},
            "king_payout": {
                "weight": 0.2,
                "alpha_per_hour": None,
                "usd_per_hour": None,
            },
            "king_chain": [
                {
                    "hotkey": "king-hotkey",
                    "alpha_per_hour": None,
                    "usd_per_hour": None,
                },
                {
                    "hotkey": "deregistered-hotkey",
                    "alpha_per_hour": None,
                    "usd_per_hour": None,
                },
            ],
        }
        client = PayoutMarket()
        service = DashboardViewService(None, None, market_client=client)
        service._apply_payouts(candidate, now=NOW)

        expected_usd = 24.6 * market()["sn3_alpha_price_tao"] * market()["tao_price_usd"]
        self.assertEqual(client.hotkeys, ["king-hotkey", "deregistered-hotkey"])
        self.assertEqual(candidate["king_payout"]["alpha_per_hour"], 24.6)
        self.assertAlmostEqual(candidate["king_payout"]["usd_per_hour"], expected_usd)
        self.assertEqual(candidate["king_chain"][0]["alpha_per_hour"], 24.6)
        self.assertAlmostEqual(candidate["king_chain"][0]["usd_per_hour"], expected_usd)
        self.assertIsNone(candidate["king_chain"][1]["alpha_per_hour"])
        self.assertIsNone(candidate["king_chain"][1]["usd_per_hour"])

    def test_market_only_refresh_preserves_existing_dashboard_snapshot(self):
        previous = payload()

        class Store:
            def previous_payload(self):
                return deepcopy(previous)

            def publish(self, body, *, source_watermark):
                self.published = json.loads(body)
                return source_watermark

        class LiveMarket:
            def fetch(self, *, now):
                return market(fetched_at=now.isoformat().replace("+00:00", "Z"))

        store = Store()
        result = DashboardViewService(
            None, store, market_client=LiveMarket()
        ).publish_market_only(now=NOW)

        self.assertEqual(result, previous["source_watermark"])
        self.assertEqual(store.published["publication_id"], previous["publication_id"])
        self.assertEqual(store.published["generated_at"], previous["generated_at"])
        self.assertEqual(store.published["market"]["tao_price_usd"], 412.5)

    def test_market_failure_does_not_block_new_database_publication(self):
        previous = payload()
        previous["market"] = market(fetched_at="2026-08-18T11:30:00Z")

        class Repository:
            project_calls = 0

            def project(self, *, now):
                self.project_calls += 1
                candidate = payload()
                candidate["source_watermark"] = 8
                return candidate

            def project_dataset_manifest(self, *, now):
                return dataset_manifest()

        class Store:
            def __init__(self):
                self.published = None

            def previous_payload(self):
                return previous

            def publish(self, body, *, source_watermark):
                self.published = json.loads(body)
                return source_watermark

            def publish_dataset_manifest(self, body, *, config_version):
                self.dataset = json.loads(body)
                return config_version

        class FailedMarket:
            def fetch(self, *, now):
                raise RuntimeError("private endpoint details must not escape")

        repository = Repository()
        store = Store()
        dashboard_result, dataset_result, active = DashboardViewService(
            repository, store, market_client=FailedMarket()
        ).publish_once(now=NOW)
        self.assertEqual(dashboard_result, 8)
        self.assertEqual(dataset_result, "a" * 64)
        self.assertFalse(active)
        self.assertEqual(repository.project_calls, 1)
        self.assertEqual(store.published["source_watermark"], 8)
        self.assertEqual(store.dataset["eval_n"], 2000)
        self.assertTrue(store.published["market"]["stale"])
        self.assertNotIn("private endpoint", json.dumps(store.published))


if __name__ == "__main__":
    unittest.main()
