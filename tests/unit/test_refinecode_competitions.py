import hashlib
import io
from copy import deepcopy

import numpy as np
import pytest

import chain_config
from scripts.prepare_packed_evaluation import prepare
from scripts.publish_packed_evaluation import run, validate_bundle
from teutonic.access.contracts import ready_signal_payload, ReadySignal
from teutonic.competitions import ACTIVE_COMPETITIONS, normalize_competition
from teutonic.evaluation.mixes import validate_definitions
from teutonic.evaluation.packed_text import (
    COLLECTION_FORMAT, DTYPE, DatasetReader, PackedDataset, build_catalog, canonical, digest, validate_pin,
)
from teutonic.evaluator.text_sampling import sample_documents
from teutonic.weights.policy import (
    FOUR_COMPETITION_POLICY, LATEST_KING_POLICY, begin_refinecode_policy,
    begin_latest_king_policy, replacement_after_win, replacement_rewards,
)


def test_weights_and_baselines():
    validate_definitions()
    assert ACTIVE_COMPETITIONS == ("main2", "mix07", "mix08", "mix09")
    specs, registry = chain_config.COMPETITION_DEFAULTS, chain_config.CATEGORY_REGISTRY
    total = sum(registry[c]["tokens"] for c in specs["main2"]["categories"])
    assert set(specs["main2"]["categories"]) == set(specs["main"]["categories"]) | {"the-stack-v2"}
    assert specs["main2"]["categories"] == {c: registry[c]["tokens"]/total for c in specs["main2"]["categories"]}
    assert specs["mix08"]["categories"] == {"dclm-baseline-1.0": .5, "the-stack-v2": .5}
    assert specs["mix09"]["categories"] == {"the-stack-v2": 1}
    assert specs["main2"]["starting_king_competition"] == "main"
    for key in ("mix08", "mix09"):
        assert specs[key]["starting_king_model"] == "teutonic-II-110B-A7B-5Eez-v9"
    for key in ACTIVE_COMPETITIONS:
        assert specs[key]["incentive"] == (.4 if key == "main2" else .2)
        assert specs[key]["delta_threshold"] == .0005
        assert specs[key]["long_document_tokens"] == 60_000_000
    assert all(specs[k]["incentive"] == 0 for k in ("main", "mix01", "mix06"))


def test_ready_signal_round_trip():
    payload = ready_signal_payload("a"*64, "b"*64, "main2")
    assert payload.endswith(":main2")
    assert normalize_competition("MAIN2") == "main2"
    # Same wire encoding as the existing MIXes, with an explicit destination.
    assert ReadySignal.parse(payload, signalling_hotkey="miner", block_number=1, extrinsic_index=0, event_index=0).competition == "main2"


def test_miner_defaults_to_main2_and_rejects_retired_admission():
    from miner.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(["ready"]).competition == "main2"
    for key in ACTIVE_COMPETITIONS:
        assert parser.parse_args(["ready", "--competition", key.upper()]).competition == key
    for key in ("main", "mix01", "mix06"):
        with pytest.raises(SystemExit):
            parser.parse_args(["ready", "--competition", key])


def old_policy(version=FOUR_COMPETITION_POLICY):
    return {"version": version, "main": {"id": "main", "hotkey": "old", "weight": .4},
            "mixes": {"mix01": "one", "mix06": "six", "mix07": "seven"}, "legacy": [],
            **({"latest_competition": "mix01"} if version == LATEST_KING_POLICY else {})}


def test_new_reward_roster_and_historical_replay():
    old = old_policy()
    assert dict(zip(*replacement_rewards(old))) == {"old": .4, "one": .2, "six": .2, "seven": .2}
    assert dict(zip(*replacement_rewards(old_policy(LATEST_KING_POLICY)))) == {
        "old": .2, "one": .4, "six": .2, "seven": .2}
    saved = deepcopy(old)
    kings = {"main2": "old", "mix07": "seven", "mix08": "v9", "mix09": "v9"}
    state = begin_refinecode_policy(old, kings)
    assert old == saved
    assert dict(zip(*replacement_rewards(state))) == pytest.approx({"old": .4, "seven": .2, "v9": .4})
    for key in ACTIVE_COMPETITIONS:
        state = replacement_after_win(state, key, key)
        rewards = dict(zip(*replacement_rewards(state)))
        assert rewards[key] == pytest.approx(.4 if key == "main2" else .2)
        assert rewards["main2"] == pytest.approx(.4)
        assert "latest_competition" not in state
        assert sum(rewards.values()) == pytest.approx(1)
    for key in ("main", "mix01", "mix06"):
        with pytest.raises(ValueError, match="retired"):
            replacement_after_win(state, key, "retired")
    with pytest.raises(ValueError):
        begin_latest_king_policy(state, "main")


def test_cutover_pins_requested_model_and_preserves_history():
    from scripts.activate_refinecode_competitions import prepare as cutover
    state = old_policy()
    roster = {"main": "old", **state["mixes"]}
    rows = [{"competition_key": k, "competition_id": k, "active": True,
             "main_competition_id": None if k == "main" else "main",
             "current_reign_id": k+"-reign", "king_hotkey": h, "king_crowned_block": {"main": 4, "mix01": 2, "mix06": 1, "mix07": 3}[k],
             "reward_transition": state, "reward_reign_id": "main-reign"} for k, h in roster.items()]
    h, w = replacement_rewards(state)
    publication = {"source_reign_id": "main-reign", "policy_hotkeys": h, "policy_weights": w}
    seed = {"model_name": "namespace/teutonic-II-110B-A7B-5Eez-v9", "reign_id": "seed-reign", "hotkey": "v9"}
    mapping = {"old": 1, "seven": 2, "one": 4, "six": 5}
    before = deepcopy((rows, publication, seed))
    plan = cutover(rows, publication, seed, mapping)
    assert (rows, publication, seed) == before
    assert plan["starting_reigns"] == {"main2": "main-reign", "mix07": "mix07-reign",
                                      "mix08": "seed-reign", "mix09": "seed-reign"}
    assert plan["retired_ids"] == ["main", "mix01", "mix06"]
    assert plan["policy_weights"] == pytest.approx([.4, .2, .2, .2])
    assert [e["id"] for e in plan["transition"]["legacy"]] == ["mix06", "mix01"]
    assert "v9" not in plan["policy_hotkeys"]
    for bad in ("wrong-seed", "missing-uid", "wrong-parent", "nan-publication"):
        r, p, s = deepcopy(before)
        m = dict(mapping)
        if bad == "wrong-seed": s["model_name"] = "another-model"
        elif bad == "missing-uid": m.pop("one")
        elif bad == "wrong-parent": r[1]["main_competition_id"] = "elsewhere"
        else: p["policy_weights"] = [float("nan"), *p["policy_weights"][1:]]
        with pytest.raises(ValueError):
            cutover(r, p, s, m)


class TwoBuckets:
    """Identical object paths and shard IDs, different bytes in each source."""
    def __init__(self):
        self.objects, self.calls = {}, []

    def add(self, bucket, key, raw):
        self.objects[bucket, key] = raw
        return {"bucket": bucket, "key": key, "sha256": hashlib.sha256(raw).hexdigest()}

    def get_object(self, *, Bucket, Key, Range=None, IfMatch=None):
        self.calls.append((Bucket, Key, Range))
        raw = self.objects[Bucket, Key]
        etag = '"' + hashlib.md5(raw).hexdigest() + '"'
        if not Range:
            return {"Body": io.BytesIO(raw)}
        assert etag == IfMatch
        lo, hi = map(int, Range.removeprefix("bytes=").split("-"))
        return {"Body": io.BytesIO(raw[lo:hi+1]), "ETag": etag,
                "ContentRange": f"bytes {lo}-{hi}/{len(raw)}"}


def collection_fixture():
    client, sources = TwoBuckets(), {}
    for name, bucket, prefix, fmt in [
        ("original", "dataset", "teutonic-text/", "teutonic-packed-text-v2"),
        ("refinecode", "code", "refinecode-text-v2/", "refinecode-packed-text-v2"),
    ]:
        rows, blob = [], b""
        for i in range(80):
            text = f"{name}:{i} żółw".encode()
            rows.append((hashlib.sha256(text).digest(), 0, len(blob), len(text),
                         2000 if i < 40 else 5000, hashlib.sha256(text).digest()))
            blob += text
        rows.sort(key=lambda r: (r[4], r[0]))
        data = io.BytesIO()
        np.save(data, np.array(rows, dtype=DTYPE), allow_pickle=False)
        objects = []
        for path, raw, role in [("same/documents.npy", data.getvalue(), "index"), ("same/text.utf8", blob, "text")]:
            pin = client.add(bucket, prefix+path, raw)
            objects.append({"path": path, "role": role, "dataset": name, "category": name,
                            "shard_id": 0, "document_count": 80, "min_tokens": 2000, "max_tokens": 5000,
                            "size_bytes": len(raw), "sha256": pin["sha256"],
                            "etag": '"'+hashlib.md5(raw).hexdigest()+'"'})
        sources[name] = client.add(bucket, prefix+"manifest.json", canonical({"format": fmt, "objects": objects}))
    manifest = {"format": COLLECTION_FORMAT, "sources": sources}
    pin = client.add("dataset", f"teutonic-text/collections/{digest(manifest)}/manifest.json", canonical(manifest))
    return client, PackedDataset(DatasetReader(client), pin)


def test_collection_routes_colliding_shards_and_samples_deterministically(tmp_path):
    client, dataset = collection_fixture()
    catalog = build_catalog(dataset)
    definitions = {"main2": {"categories": {"original": .5, "refinecode": .5}, "long_document_tokens": 40_000}}
    registry = {name: {"dataset": name} for name in ("original", "refinecode")}
    prepare(dataset, definitions, registry, tmp_path, catalog=catalog)
    _, _, mixes = validate_bundle(tmp_path)
    first, _ = sample_documents(dataset, catalog, mixes[0], "test", index_workers=1)
    second, _ = sample_documents(dataset, catalog, mixes[0], "test", index_workers=8)
    assert first == second
    assert {d.dataset for d in first} == {"original", "refinecode"}
    for doc in first:
        assert dataset.text(doc).startswith(doc.dataset + ":")
    text_calls = {(b, k) for b, k, _ in client.calls if k.endswith(".utf8")}
    assert text_calls == {("dataset", "teutonic-text/same/text.utf8"), ("code", "refinecode-text-v2/same/text.utf8")}
    # The collection has not been published yet; validation loads it from the bundle.
    del client.objects[dataset.pin['bucket'], dataset.pin['key']]
    assert run(tmp_path, client, publish=False)["state"] == "validated-read-only"


@pytest.mark.parametrize("bucket,key", [("code", "refinecode-text/manifest.json"),
                                         ("datasets", "teutonic-text/manifest.json"),
                                         ("code", "refinecode-text-v2/../secret")])
def test_unapproved_roots_rejected(bucket, key):
    with pytest.raises(ValueError):
        validate_pin({"bucket": bucket, "key": key, "sha256": "a"*64})
