import hashlib
import tomllib
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import chain_config
from scripts.configure_evaluation import parse_args, prepare_configuration
from teutonic.access.contracts import ReadySignal, ready_signal_payload
from teutonic.competitions import ACTIVE_COMPETITIONS, normalize_competition
from teutonic.dashboard.projection import _dataset_sources
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    EvaluationSettings,
    canonical_manifest_bytes,
    pretokenized_dataset_request,
)
from teutonic.evaluation.mixes import build_mix_manifests, validate_definitions, write_bundle
from teutonic.weights.policy import replacement_after_win, replacement_rewards

LEGACY_CONFIG = tomllib.loads((Path(__file__).parents[1] / "fixtures/legacy_evaluation.toml").read_text())

def original_sources():
    result = []
    for dataset, spec in chain_config.CATEGORY_DATASETS.items():
        shards = [
            {
                "key": f"shards/{dataset}__{category}--part-{i}.npy",
                "sha256": "a" * 64,
                "n_tokens": 100_000 * 2048,
                "size_bytes": 100_000 * 2048 * 4 + 128,
            }
            for category, entry in chain_config.CATEGORY_REGISTRY.items()
            if entry["dataset"] == dataset
            for i in range(5)
        ]
        manifest = {"shards": shards, "seq_len": 2048}
        result.append(
            DatasetManifestSnapshot(
                dataset,
                spec["manifest_url"],
                hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
                1.0,
                manifest,
            )
        )
    return result


@pytest.fixture
def bundle(tmp_path):
    outputs = build_mix_manifests(original_sources())
    write_bundle(outputs, tmp_path)
    return tmp_path, outputs


@pytest.mark.parametrize("key", ACTIVE_COMPETITIONS)
def test_every_competition_uses_identical_flat_planning(key, bundle):
    directory, outputs = bundle
    args = parse_args(
        ["--initialize", "--bundle-dir", str(directory), "--long-document-tokens", "0"]
    )
    config, version = prepare_configuration(None, key, args)
    settings = EvaluationSettings(config_version=version, **config)
    request = pretokenized_dataset_request(
        settings, block_hash="0x123", hotkey="miner", seq_len=2048
    )
    assert request == pretokenized_dataset_request(
        settings, block_hash="0x123", hotkey="miner", seq_len=2048
    )
    (source,) = request["sources"]
    assert source["target_sequences"] == 20000
    assert len({s["url"] for s in source["shards"]}) == len(source["shards"])
    by_url = {s["url"]: s["category"] for s in outputs[key]["shards"]}
    counts = Counter()
    for shard in source["shards"]:
        counts[by_url[shard["url"]]] += shard["target_sequences"]
    weights = chain_config.COMPETITION_DEFAULTS[key]["categories"]
    assert set(counts) == set(weights)
    assert sum(counts.values()) == 20000
    assert all(abs(counts[c] - 20000 * w) < 1.000001 for c, w in weights.items())


def test_mix02_preserves_two_token_proportional_halves_without_overlap():
    validate_definitions()
    weights = chain_config.COMPETITION_DEFAULTS["mix02"]["categories"]
    code_categories = set(LEGACY_CONFIG["splits"]["code"]["categories"])
    remainder = set(weights) - code_categories
    assert remainder == {"format-rewrites", "games", "general", "planning"}
    for group in (code_categories, remainder):
        total = sum(chain_config.CATEGORY_REGISTRY[c]["tokens"] for c in group)
        assert sum(weights[c] for c in group) == pytest.approx(0.5)
        for c in group:
            assert weights[c] == pytest.approx(
                0.5 * chain_config.CATEGORY_REGISTRY[c]["tokens"] / total
            )
    assert weights["reasoning"] == pytest.approx(0.156458, abs=1e-6)


def test_changed_bundle_weights_rejected(bundle):
    directory, outputs = bundle
    bad = deepcopy(outputs)
    weights = bad["mix02"]["category_weights"]
    weights["reasoning"] += 0.01
    weights["planning"] -= 0.01
    write_bundle(bad, directory)
    with pytest.raises(ValueError, match="weights differ"):
        prepare_configuration(
            None,
            "mix02",
            parse_args(
                ["--initialize", "--bundle-dir", str(directory), "--long-document-tokens", "0"]
            ),
        )


def test_dashboard_uses_category_inventory_without_large_shard_list(bundle):
    _, outputs = bundle
    manifest = dict(outputs["mix02"])
    manifest.pop("shards")
    rows = _dataset_sources(
        [
            {
                "manifest_json": manifest,
                "name": "mix02",
                "sample_proportion": 1.0,
                "manifest_url": "https://example.com/mix02.json",
                "manifest_sha256": "a" * 64,
            }
        ]
    )
    assert len(rows) == 14
    assert sum(r["proportion"] for r in rows) == pytest.approx(1)
    assert all(r["total_shards"] == 5 for r in rows)


def test_compressed_database_manifest_preserves_original_hash_and_detects_tampering(bundle):
    from teutonic.evaluation.configuration import manifest_for_storage

    _, outputs = bundle
    manifest = outputs["mix06"]
    digest = hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()
    stored = manifest_for_storage(manifest)
    assert "shards" not in stored and "_shards_zlib_base64" in stored
    snap = DatasetManifestSnapshot("mix06", "https://example.com/mix06.json", digest, 1, stored)
    assert snap.manifest == manifest
    stored["total_tokens"] += 1
    with pytest.raises(ValueError, match="content does not match"):
        DatasetManifestSnapshot("mix06", "https://example.com/mix06.json", digest, 1, stored)


@pytest.mark.parametrize("key", [*ACTIVE_COMPETITIONS, "mix99"])
def test_mix_commitment_roundtrip_and_size(key):
    payload = ready_signal_payload("a" * 64, "b" * 64, key.upper())
    assert len(payload.encode()) <= 128
    signal = ReadySignal.parse(
        payload, signalling_hotkey="miner", block_number=1, extrinsic_index=0, event_index=0
    )
    assert signal.competition == key
    assert normalize_competition(key.upper()) == key


@pytest.mark.parametrize("key", ["mix00", "mix100", "../mix01", "code"])
def test_new_cli_rejects_invalid_or_retired_names(key):
    with pytest.raises(ValueError):
        normalize_competition(key)


def transition():
    return {
        "version": "mix-replacement-v1",
        "main": {"id": "main", "hotkey": "main", "weight": 0.4},
        "legacy": [{"id": f"old-{i}", "hotkey": f"old-{i}", "weight": 0.15} for i in range(4)],
        "mixes": {},
        "evictions": [],
    }


def test_reward_transition_displaces_one_position_only_on_first_win():
    state = transition()
    assert dict(zip(*replacement_rewards(state))) == pytest.approx(
        {"main": 0.4, **{f"old-{i}": 0.15 for i in range(4)}}
    )
    new = replacement_after_win(state, "mix02", "winner")
    rewards = dict(zip(*replacement_rewards(new)))
    assert "old-0" not in rewards
    assert rewards["winner"] == 0.125
    assert sum(rewards.values()) == pytest.approx(1)
    assert state == transition()
    newer = replacement_after_win(new, "mix02", "replacement")
    assert len(newer["legacy"]) == 3
    assert len(newer["evictions"]) == 1
    assert "winner" not in dict(zip(*replacement_rewards(newer)))
    for key in ACTIVE_COMPETITIONS[1:]:
        newer = replacement_after_win(newer, key, key)
    assert not newer["legacy"]
    assert dict(zip(*replacement_rewards(newer))) == {
        "main": 0.25,
        **{key: 0.125 for key in ACTIVE_COMPETITIONS[1:]},
    }


def test_new_main_replaces_old_main_without_entering_legacy_roster():
    state = replacement_after_win(transition(), "main", "latest-main")
    assert state["legacy"] == transition()["legacy"]
    rewards = dict(zip(*replacement_rewards(state)))
    assert rewards["latest-main"] == 0.4
    assert "main" not in rewards


def test_same_hotkey_positions_aggregate_without_removing_other_entitlements():
    state = transition()
    state["legacy"][0]["hotkey"] = "main"
    state = replacement_after_win(state, "mix01", "main")
    rewards = dict(zip(*replacement_rewards(state)))
    assert rewards["main"] == pytest.approx(0.125 + 0.875 * 0.4 / 0.85)
    assert sum(rewards.values()) == pytest.approx(1)


def test_immutable_publisher_never_overwrites_existing_objects():
    import io

    from botocore.exceptions import ClientError

    from scripts.publish_category_artifacts import publish_immutable

    class Store:
        def __init__(self):
            self.body = None
            self.writes = 0

        def head_object(self, **kwargs):
            if self.body is None:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            return {}

        def put_object(self, **kwargs):
            assert kwargs["IfNoneMatch"] == "*"
            self.writes += 1
            self.body = kwargs["Body"]

        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(self.body)}

    client = Store()
    body = b'{"test":true}'
    digest = hashlib.sha256(body).hexdigest()
    publish_immutable(client, "datasets", "new-version.json", body, digest)
    publish_immutable(client, "datasets", "new-version.json", body, digest)
    assert client.writes == 1
    client.body = b"different existing data"
    with pytest.raises(ValueError, match="immutable object checksum mismatch"):
        publish_immutable(client, "datasets", "new-version.json", body, digest)
    assert client.writes == 1
