import hashlib
from collections import Counter
from dataclasses import replace

import pytest

import chain_config
from scripts.configure_evaluation import parse_args, prepare_configuration
from scripts.publish_split_manifests import build_manifests
from teutonic.access.contracts import ReadySignal, ready_signal_payload
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    EvaluationSettings,
    canonical_manifest_bytes,
    pretokenized_dataset_request,
    validate_dataset_manifest,
)
from teutonic.weights.policy import competition_rewards, mapped_weight_plan


def snapshot(name, manifest, proportion=1.0):
    return DatasetManifestSnapshot(
        name,
        f"https://datasets.example/{name}/manifest.json",
        hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
        proportion,
        manifest,
    )


def source_snapshots():
    sources = {}
    for spec in chain_config.SPLIT_DEFAULTS.values():
        for category, config in spec["categories"].items():
            dataset = config["dataset"]
            sources.setdefault(dataset, []).append(
                {
                    "key": f"shards/{dataset}__{category}--fixture.npy",
                    "sha256": "a" * 64,
                    "n_tokens": 100_000 * 2048,
                    "size_bytes": 100_000 * 2048 * 4 + 128,
                }
            )
    return [snapshot(name, {"shards": shards}) for name, shards in sources.items()]


def split_snapshots(selected):
    manifests = build_manifests(source_snapshots(), chain_config.SPLIT_DEFAULTS)
    return tuple(snapshot(k, v, 0.7 if k == selected else 0.15) for k, v in manifests.items())


@pytest.mark.parametrize("selected", ["math", "code", "text"])
def test_live_definitions_cover_24_categories_and_exact_quotas(selected):
    manifests = split_snapshots(selected)
    assert sum(len(s.manifest["category_weights"]) for s in manifests) == 24
    settings = EvaluationSettings("a" * 64, "fixture", 30000, 0.003, manifests, 4)
    request = pretokenized_dataset_request(
        settings, block_hash="0x1", hotkey="hotkey", seq_len=2048
    )
    assert request == pretokenized_dataset_request(
        settings, block_hash="0x1", hotkey="hotkey", seq_len=2048
    )
    for source in request["sources"]:
        count = 21000 if source["name"] == selected else 4500
        assert source["target_sequences"] == count
        assert sum(s["target_sequences"] for s in source["shards"]) == count
        manifest = next(s for s in manifests if s.name == source["name"])
        category_by_url = {s["url"]: s["category"] for s in manifest.manifest["shards"]}
        counts = Counter()
        for shard in source["shards"]:
            counts[category_by_url[shard["url"]]] += shard["target_sequences"]
        assert set(counts) == set(manifest.manifest["category_weights"])
        for category, share in manifest.manifest["category_weights"].items():
            assert abs(counts[category] - count * share) < 1.00000001


@pytest.mark.parametrize("competition", ["main", "math", "code", "text"])
def test_ready_selection_roundtrip_within_chain_commitment_size(competition):
    payload = ready_signal_payload("a" * 64, "b" * 64, competition)
    assert len(payload.encode()) <= 128
    parsed = ReadySignal.parse(
        payload, signalling_hotkey="hotkey", block_number=1, extrinsic_index=0, event_index=0
    )
    assert parsed.competition == competition
    assert parsed.registration_id == "a" * 64
    assert parsed.manifest_sha256 == "b" * 64
    with pytest.raises(ValueError):
        ReadySignal.parse(
            payload + ":unknown",
            signalling_hotkey="hotkey",
            block_number=1,
            extrinsic_index=0,
            event_index=0,
        )


def test_threshold_update_never_fetches_or_replaces_manifests():
    current = {
        "dataset_label": "math",
        "n": 30000,
        "delta_threshold": 0.003,
        "shards_per_dataset": 4,
        "manifests": split_snapshots("math"),
    }

    def forbidden(**kwargs):
        raise AssertionError("threshold-only update attempted network access")

    changed, version = prepare_configuration(
        current,
        "math",
        parse_args(["--competition", "math", "--delta-threshold", ".005"]),
        forbidden,
    )
    assert changed["delta_threshold"] == 0.005
    assert changed["manifests"] == current["manifests"]
    assert len(version) == 64
    refreshed, _ = prepare_configuration(
        current,
        "math",
        parse_args(["--refresh-manifests"]),
        lambda **kwargs: replace(
            next(s for s in current["manifests"] if s.name == kwargs["name"]), **kwargs
        ),
    )
    assert refreshed["delta_threshold"] == 0.003


def test_manifest_missing_or_extra_category_fails_closed():
    manifest = dict(split_snapshots("math")[0].manifest)
    manifest["shards"] = manifest["shards"][1:]
    with pytest.raises(RuntimeError, match="declared categories"):
        validate_dataset_manifest(manifest)


@pytest.mark.parametrize(
    "k,main_shares",
    [(0, [0.2] * 5), (1, [0.25, 0.2, 0.2, 0.2]), (2, [0.3, 0.2, 0.2]), (3, [0.4, 0.15])],
)
def test_gradual_rewards_and_main_replacement(k, main_shares):
    history = ["main-5", "main-4", "main-3", "main-2", "main-1"]
    splits = dict(zip(("math", "code", "text")[:k], ("math-king", "code-king", "text-king")[:k]))
    hotkeys, shares = competition_rewards(history, splits)
    assert list(shares) == pytest.approx(main_shares + [0.15] * k)
    assert len(hotkeys) == 5
    assert sum(shares) == pytest.approx(1)
    next_hotkeys, next_shares = competition_rewards(["new-main", *history][:5], splits)
    assert next_hotkeys[0] == "new-main"
    assert history[len(main_shares) - 1] not in next_hotkeys
    assert next_shares == shares
    if k:
        splits["math"] = "new-math"
        replaced, weights = competition_rewards(history, splits)
        assert "math-king" not in replaced
        assert "new-math" in replaced
        assert weights == shares


def test_uid_remap_preserves_relative_shares_and_burn_fallback():
    hotkeys, uids, weights = mapped_weight_plan(
        ["a", "b", "gone"], [0.4, 0.15, 0.45], {"a": 7, "b": 9}, burn_uid=0
    )
    assert hotkeys == ("a", "b")
    assert uids == (7, 9)
    assert weights == pytest.approx((0.4 / 0.55, 0.15 / 0.55))
    assert mapped_weight_plan(["gone"], [1], {}, burn_uid=0) == (("burn:uid:0",), (0,), (1.0,))
