import copy
import hashlib
import io
import json
import tomllib
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

import chain_config
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    EvaluationSettings,
    canonical_manifest_bytes,
    evaluation_config_version,
    pretokenized_dataset_request,
)
from teutonic.evaluation.long_documents import (
    LONG_DOCUMENT_VERSION,
    validate_long_documents,
)
from teutonic.evaluation.long_documents import (
    build_long_document_config as build_config,
)
from teutonic.evaluator.document_index import (
    FRAG_DTYPE,
    PAGE_SIZE,
    PTR_DTYPE,
    DocumentIndex,
    DocumentIndexDownloadError,
    NpyVector,
    PinnedObject,
    sample_category,
)
from teutonic.evaluator.long_documents import extend_with_long_documents, sequence_digest

LEGACY_CONFIG = tomllib.loads((Path(__file__).parents[1] / "fixtures/legacy_evaluation.toml").read_text())

def snapshot(name, shards, proportion=1, **extra):
    manifest = {"shards": shards, **extra}
    return DatasetManifestSnapshot(
        name,
        f"https://datasets.example/{name}/manifest.json",
        hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
        proportion,
        manifest,
    )


def descriptor(url):
    return {"url": url, "etag": '"abc123"', "size_bytes": 100}


def build_long_document_config(sources, describe=descriptor, index_base="https://index.example/doc_index", **kwargs):
    from teutonic.evaluation.index_manifest import (
        INDEX_FORMAT,
        TOKEN_FORMAT,
        index_paths,
        pin_index_manifest,
    )
    from teutonic.evaluation.long_documents import category_mix
    files, datasets = {}, {}
    originals = {s.name: s for s in all_sources()}
    originals.update({s.name: s for s in sources if "category_weights" not in s.manifest})
    for c in category_mix(sources):
        original = originals[c['dataset']]
        item = datasets.setdefault(c['dataset'], {'categories': [], 'source_manifest': {'url': original.manifest_url, 'sha256': original.manifest_sha256}})
        item['categories'].append(c['category'])
        for path in index_paths(c['dataset'], c['category']).values():
            d = describe(index_base + '/' + path)
            files[path] = {'etag': d['etag'], 'size_bytes': d['size_bytes'], 'sha256': hashlib.sha256(path.encode()).hexdigest()}
    raw = json.dumps({'format': INDEX_FORMAT, 'token_format': TOKEN_FORMAT, 'datasets': datasets, 'files': files}).encode()
    return build_config(sources, index_manifest=pin_index_manifest(raw, index_base+'/manifest.json'), **kwargs)


def all_sources():
    groups = {}
    for spec in LEGACY_CONFIG["splits"].values():
        for category, item in spec["categories"].items():
            dataset = item["dataset"]
            groups.setdefault(dataset, []).append(
                {
                    "key": f"shards/{dataset}__{category}--fixture.npy",
                    "sha256": "a" * 64,
                    "n_tokens": 2048 * 100000,
                    "size_bytes": 2048 * 100000 * 4 + 128,
                }
            )
    proportions = {s["name"]: s["proportion"] for s in LEGACY_CONFIG["evaluation"]["datasets"]}
    return tuple(snapshot(k, v, proportions[k]) for k, v in groups.items())


@pytest.mark.parametrize("competition", ["main", "math", "code", "text"])
def test_competition_mix_uses_existing_source_and_category_weights(competition):
    from scripts.publish_split_manifests import build_manifests

    sources = all_sources()
    if competition != "main":
        manifests = build_manifests(sources, LEGACY_CONFIG["splits"])
        sources = tuple(
            snapshot(k, v.pop("shards"), 0.7 if k == competition else 0.15, **v)
            for k, v in manifests.items()
        )
    config = build_long_document_config(sources, describe=descriptor)
    assert config["max_document_tokens"] == 16384
    assert config["version"] == LONG_DOCUMENT_VERSION
    assert len(config["categories"]) == 24
    assert sum(c["weight"] for c in config["categories"]) == pytest.approx(1)
    for s in sources:
        entries = [c for c in config["categories"] if c["source"] == s.name]
        assert sum(c["weight"] for c in entries) == pytest.approx(s.proportion)
        for c in entries:
            expected = s.manifest.get("category_weights", {}).get(c["category"], 1 / len(entries))
            assert c["weight"] == pytest.approx(s.proportion * expected)
            assert f"/doc_index/{c['dataset']}/" in c["files"]["tokens"]["url"]
    args = {
        "dataset_label": competition,
        "n": 30000,
        "delta_threshold": 0.003,
        "manifests": sources,
        "shards_per_dataset": 4,
    }
    version = evaluation_config_version(**args, long_documents=config)
    assert version != evaluation_config_version(**args)
    changed = copy.deepcopy(config)
    changed["categories"][0]["files"]["tokens"]["etag"] = '"def456"'
    with pytest.raises(ValueError, match="differ"):
        evaluation_config_version(**args, long_documents=changed)
    settings = EvaluationSettings(config_version=version, **args, long_documents=config)
    request = pretokenized_dataset_request(settings, block_hash="b", hotkey="h", seq_len=2048)
    assert request["long_documents"] == config
    assert sum(s["target_sequences"] for s in request["sources"]) == 30000


def npy_bytes(array):
    out = io.BytesIO()
    np.save(out, array)
    return out.getvalue()


class RangeServer:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []
        self.wrong_range = False
        self.wrong_etag = False

    def describe(self, url):
        body = self.objects[url]
        return {"url": url, "size_bytes": len(body), "etag": f'"{hashlib.md5(body).hexdigest()}"'}

    def open(self, request, timeout):
        self.calls.append(request)
        d = self.describe(request.full_url)
        if request.get_header("If-match") != d["etag"]:
            raise HTTPError(request.full_url, 412, "changed", {}, None)
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        response = io.BytesIO(self.objects[request.full_url][start : end + 1])
        response.status = 200 if self.wrong_range else 206
        response.headers = {
            "ETag": '"bad"' if self.wrong_etag else d["etag"],
            "Content-Range": f"bytes {start}-{end}/{d['size_bytes']}",
        }
        return response


def test_range_reader_pins_versions_checks_ranges_and_recovers_corrupt_cache(tmp_path):
    server = RangeServer({"https://index.example/x": bytes(range(256)) * 1024})
    d = server.describe("https://index.example/x")
    reader = PinnedObject(d, tmp_path, server.open)
    assert reader.read(PAGE_SIZE - 3, 8) == server.objects[d["url"]][PAGE_SIZE - 3 : PAGE_SIZE + 5]
    assert len(server.calls) == 2
    cached = PinnedObject(d, tmp_path, lambda *a, **k: pytest.fail("cache miss"))
    assert cached.read(PAGE_SIZE - 3, 8) == reader.read(PAGE_SIZE - 3, 8)
    (reader.cache / "0").write_bytes(b"corrupt")
    repaired = PinnedObject(d, tmp_path, server.open)
    assert repaired.read(0, 3) == b"\x00\x01\x02"
    server.wrong_range = True
    with pytest.raises(RuntimeError, match="range"):
        reader.read(2 * PAGE_SIZE, 1)
    server.wrong_range = False
    server.wrong_etag = True
    with pytest.raises(RuntimeError, match="changed"):
        reader.read(2 * PAGE_SIZE, 1)
    server.wrong_etag = False
    server.objects[d["url"]] = b"new contents"
    with pytest.raises(HTTPError):
        reader.read(2 * PAGE_SIZE, 1)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, None])
def test_range_reader_retries_transient_failures_without_changing_pins(tmp_path, monkeypatch, status):
    from teutonic.evaluator import document_index

    server = RangeServer({"https://index.example/x": b"abcdef"})
    calls, delays = [], []

    def open_with_failure(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            if status is None:
                raise URLError("connection reset")
            raise HTTPError(request.full_url, status, "temporary", {"Retry-After": "12"}, None)
        return server.open(request, timeout)

    monkeypatch.setattr(document_index.time, "sleep", delays.append)
    monkeypatch.setattr(document_index.random, "uniform", lambda *_: 0)
    reader = PinnedObject(server.describe("https://index.example/x"), tmp_path, open_with_failure)
    assert reader.read(1, 3) == b"bcd"
    assert delays == [5 if status is None else 12]
    assert calls[0] is calls[1]
    assert reader.read(1, 3) == b"bcd"
    assert len(calls) == 2


def test_range_reader_exhaustion_is_bounded_and_does_not_cache_failure(tmp_path, monkeypatch):
    from teutonic.evaluator import document_index

    server = RangeServer({"https://index.example/x": b"abcdef"})
    calls, delays = [], []

    def unavailable(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 429, "rate limited", {}, None)

    monkeypatch.setattr(document_index.time, "sleep", delays.append)
    monkeypatch.setattr(document_index.random, "uniform", lambda *_: 0)
    reader = PinnedObject(server.describe("https://index.example/x"), tmp_path, unavailable)
    with pytest.raises(DocumentIndexDownloadError, match="exhausted") as error:
        reader.read(0, 3)
    assert isinstance(error.value.__cause__, HTTPError)
    assert len(calls) == 5
    assert delays == [5, 10, 20, 40]
    assert not reader.cache.exists()


@pytest.mark.parametrize("status", [403, 404, 412])
def test_range_reader_does_not_retry_permanent_http_failures(tmp_path, monkeypatch, status):
    from teutonic.evaluator import document_index

    server = RangeServer({"https://index.example/x": b"abcdef"})

    def unavailable(request, timeout):
        raise HTTPError(request.full_url, status, "permanent", {}, None)

    monkeypatch.setattr(document_index.time, "sleep", lambda _: pytest.fail("unexpected retry"))
    reader = PinnedObject(server.describe("https://index.example/x"), tmp_path, unavailable)
    with pytest.raises(HTTPError) as error:
        reader.read(0, 3)
    assert error.value.code == status


@pytest.mark.parametrize("retry_after, delay", [
    ("Fri, 09 Oct 2026 20:00:30 GMT", 30),
    ("invalid", 5),
    ("300", None),
])
def test_range_reader_retry_after_dates_and_long_cooldowns(tmp_path, monkeypatch, retry_after, delay):
    from datetime import datetime

    from teutonic.evaluator import document_index

    server = RangeServer({"https://index.example/x": b"abcdef"})
    calls, delays = [], []

    def open_with_failure(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "limited", {"Retry-After": retry_after}, None)
        return server.open(request, timeout)

    monkeypatch.setattr(document_index, "datetime", SimpleNamespace(
        now=lambda _: datetime(2026, 10, 9, 20, 0, tzinfo=UTC),
    ))
    monkeypatch.setattr(document_index.time, "sleep", delays.append)
    monkeypatch.setattr(document_index.random, "uniform", lambda *_: 0)
    reader = PinnedObject(server.describe("https://index.example/x"), tmp_path, open_with_failure)
    if delay is None:
        with pytest.raises(DocumentIndexDownloadError, match="cooldown"):
            reader.read(0, 3)
        assert delays == []
    else:
        assert reader.read(0, 3) == b"abc"
        assert delays == [delay]


def test_npy_index_rejects_wrong_dtype_and_truncation(tmp_path):
    for body in (npy_bytes(np.array([1], dtype="<i8")), npy_bytes(np.array([1], dtype="<i4"))[:-1]):
        server = RangeServer({"https://index.example/x": body})
        with pytest.raises(ValueError):
            NpyVector(
                PinnedObject(server.describe("https://index.example/x"), tmp_path, server.open),
                np.dtype("<i4"),
            )


class MemoryLengths:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, row):
        return self.values[row]

    def search(self, value, right=False):
        return int(np.searchsorted(self.values, value, side="right" if right else "left"))


@pytest.mark.parametrize("populations", [(5000, 500), (500, 5000)])
def test_sampler_allocates_tokens_40_60_independently_of_bucket_population(populations):
    values = (
        [0, 1, 512, 1024, 1536, 2048]
        + [2049] * (populations[0] // 2)
        + [4096] * (populations[0] // 2)
        + [4097] * (populations[1] // 2)
        + [16384] * (populations[1] // 2)
        + [16385] * 1000
        + [99999] * 9
    )
    index = SimpleNamespace(tokens=MemoryLengths(values), locate=lambda row: [{"row": row}])
    rows, meta = sample_category(index, token_budget=1_000_000, seed=123, max_tokens=16384)
    assert (rows, meta) == sample_category(
        index, token_budget=1_000_000, seed=123, max_tokens=16384
    )
    assert len({d["document_row"] for d in rows}) == len(rows)
    assert [b["population"] for b in meta["buckets"]] == list(populations)
    assert [b["target_tokens"] for b in meta["buckets"]] == [400_000, 600_000]
    assert [b["token_weight"] for b in meta["buckets"]] == [0.4, 0.6]
    assert all(2049 <= d["length"] <= 16384 for d in rows)
    assert meta["excluded_under_length_limit"] == 6
    assert meta["excluded_over_length_limit"] == 1009
    assert meta["max_document_tokens"] == 16384
    assert abs(meta["sampled_tokens"] - 1_000_000) <= 4096 + 16384
    for b, maximum in zip(meta["buckets"], [4096, 16384], strict=True):
        assert abs(b["sampled_tokens"] - b["target_tokens"]) <= maximum
        selected = [d for d in rows if d["length_bucket"] == b["bucket"]]
        assert b["sampled_tokens"] == sum(d["length"] for d in selected)
        assert b["sampled_documents"] == len(selected)
    tiny, tiny_meta = sample_category(index, token_budget=1, seed=123, max_tokens=16384)
    assert {d["length_bucket"] for d in tiny} == {
        "2049-4096", "4097-16384"
    }
    assert tiny_meta["sampled_tokens"] > tiny_meta["target_tokens"]


@pytest.mark.parametrize("short", [2049, 4096])
@pytest.mark.parametrize("long", [4097, 8192, 8193, 16384])
def test_sampler_includes_exact_bucket_edges_and_excludes_outside_range(short, long):
    eligible = [short, long]
    lengths = [0, 1, 2048, *eligible, 16385, 128974]
    def locate(row):
        assert 2049 <= lengths[row] <= 16384
        return [{"row": row}]
    index = SimpleNamespace(tokens=MemoryLengths(lengths), locate=locate)
    rows, meta = sample_category(index, token_budget=1, seed=15, max_tokens=1048576)
    assert sorted(d["length"] for d in rows) == eligible
    assert meta["sampled_tokens"] == sum(eligible)
    assert meta["excluded_over_length_limit"] == 2
    assert meta["excluded_under_length_limit"] == 3


@pytest.mark.parametrize("lengths", [[2049], [4097], [2048, 16385]])
def test_sampler_rejects_missing_required_bucket(lengths):
    index = SimpleNamespace(tokens=MemoryLengths(lengths), locate=lambda row: [{"row": row}])
    with pytest.raises(ValueError, match="required bucket"):
        sample_category(index, token_budget=100_000, seed=1, max_tokens=16384)


def test_seventy_million_budget_allocates_28m_and_42m_input_tokens():
    index = SimpleNamespace(
        tokens=MemoryLengths([2500] * 20_000 + [10_000] * 20_000),
        locate=lambda row: [{"row": row}],
    )
    rows, meta = sample_category(index, token_budget=70_000_000, seed=123, max_tokens=16384)
    assert [b["sampled_tokens"] for b in meta["buckets"]] == [28_000_000, 42_000_000]
    assert [b["sampled_documents"] for b in meta["buckets"]] == [11_200, 4_200]
    assert meta["sampled_tokens"] == 70_000_000
    assert len({d["document_row"] for d in rows}) == len(rows) == 15_400


def test_sampler_rejects_insufficient_context_or_complete_documents():
    index = SimpleNamespace(tokens=MemoryLengths([2049, 8193]), locate=lambda row: None)
    with pytest.raises(ValueError, match="context limit"):
        sample_category(index, token_budget=1, seed=1, max_tokens=8192)
    with pytest.raises(ValueError, match="not enough complete documents"):
        sample_category(index, token_budget=1, seed=1, max_tokens=16384)
    index.locate = lambda row: [{"row": row}]
    with pytest.raises(ValueError, match="not enough complete documents"):
        sample_category(index, token_budget=1_000_000, seed=1, max_tokens=16384)


def make_index_fixture(tmp_path):
    # Original shard rows are 2048 tokens; documents cross both rows and shards.
    lengths = [2, 512, 513, 1024, 1025, 1536, 1537, 2048, 2049, 4096, 16384, 16385, 20000]
    documents = [
        np.concatenate([np.full(n - 1, i + 1, dtype="<u4"), np.array([151645], dtype="<u4")])
        for i, n in enumerate(lengths)
    ]
    stream = np.concatenate(documents)
    stream = np.pad(stream, (0, -len(stream) % 8192))
    shards, names, paths = [], [], {}
    for i, start in enumerate(range(0, len(stream), 8192)):
        key = f"shards/code-reasoning__codeio--{i}.npy"
        path = tmp_path / f"shard-{i}.npy"
        np.save(path, stream[start : start + 8192])
        names.append(key)
        paths[key] = path
        shards.append(
            {
                "key": key,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_tokens": 8192,
                "size_bytes": path.stat().st_size,
            }
        )
    fragments, pointers, cursor = [], [], 0
    for length in lengths:
        first = len(fragments)
        left = length
        while left:
            shard, inside = divmod(cursor, 8192)
            row, offset = divmod(inside, 2048)
            size = min(left, 2048 - offset)
            fragments.append((shard, row, offset, size))
            left -= size
            cursor += size
        pointers.append((first, len(fragments) - first))
    base = "https://index.example/doc_index/code-reasoning/"
    objects = {
        base + "codeio.tokens.npy": npy_bytes(np.array(lengths, dtype="<i4")),
        base + "codeio.frag_ptr.npy": npy_bytes(np.array(pointers, dtype=PTR_DTYPE)),
        base + "codeio.fragments.bin": np.array(fragments, dtype=FRAG_DTYPE).tobytes(),
        base + "shard_names.json": json.dumps(names).encode(),
    }
    server = RangeServer(objects)
    snap = snapshot("code-reasoning", shards)
    config = build_long_document_config(
        (snap,),
        token_budget=15000,
        index_base="https://index.example/doc_index",
        describe=server.describe,
    )
    factory = lambda c, cache: DocumentIndex(
        c, cache, lambda d, root: PinnedObject(d, root, server.open)
    )
    return config, snap, factory, documents, paths, server


@pytest.mark.parametrize("flat_mix", [False, True])
@pytest.mark.parametrize("local_files", [0, 2, 4])
@pytest.mark.parametrize("documents_only", [False, True])
def test_complete_document_reconstruction_shuffle_and_audit_without_model(tmp_path, monkeypatch, flat_mix, local_files, documents_only):
    import teutonic.evaluator as evaluator_package

    config, snap, factory, documents, paths, _server = make_index_fixture(tmp_path)
    if flat_mix:
        from teutonic.evaluation.index_manifest import ETAG_INDEX_FORMAT, pin_index_manifest
        from teutonic.evaluation.mixes import build_mix_manifests
        manifest = build_mix_manifests((snap,), {"mix02": {"categories": {"codeio": 1.0}}})["mix02"]
        snap = snapshot("mix02", manifest.pop("shards"), **manifest)
        inventory = json.loads(config["index_manifest"]["content"])
        inventory["format"] = ETAG_INDEX_FORMAT
        for entry in inventory["files"].values():
            entry.pop("sha256")
        pin = pin_index_manifest(json.dumps(inventory).encode(), config["index_manifest"]["url"])
        config = build_config((snap,), token_budget=15000, index_manifest=pin)
    fake_sources = SimpleNamespace(
        base=SimpleNamespace(SHARD_CACHE_DIR=tmp_path),
        ShardRef=lambda *args: SimpleNamespace(
            source=args[0], url=args[1], sha256=args[2], size_bytes=args[3], n_tokens=args[4]
        ),
    )
    monkeypatch.setattr(evaluator_package, "sources", fake_sources, raising=False)
    req = SimpleNamespace(
        long_documents=config,
        max_document_tokens=16384,
        vocab_size=160000,
        dataset_sources=[{"name": snap.name, "proportion": 1}],
    )
    meta = {
        "dataset_seed": 10,
        "digest": "base",
        "sources": [{"name": snap.name}],
        "_source_labels": [snap.name],
        "_sample_provenance": [
            {"shard_group_index": 0, "shard_index": 0, "shard_sequence_index": 0}
        ],
    }
    windows = [] if documents_only else [[12, 13]]
    if documents_only:
        meta["_source_labels"] = []
        meta["_sample_provenance"] = []
    kwargs = {
        "index_factory": factory,
        "manifest_loader": lambda *args: snap,
    }
    from test_token_shards import ShardServer

    from teutonic.evaluator.token_shards import NpyShard

    server = ShardServer({
        f"https://datasets.example/code-reasoning/{key}": path.read_bytes()
        for key, path in paths.items()
    })
    kwargs["shard_reader"] = lambda ref: NpyShard(ref, opener=server.open)
    result, audit = extend_with_long_documents(req, windows, meta, **kwargs)
    first_transferred = server.transferred
    if local_files:
        from teutonic.evaluator.local_index import LocalIndexStore

        root = tmp_path / "persistent-indexes"
        store = LocalIndexStore(root)
        descriptors = list(config["categories"][0]["files"].values())[:local_files]

        def download(request, timeout):
            response = _server.open(request, timeout)
            response.iter_chunks = lambda size: iter(lambda: response.read(size), b"")
            return response

        for descriptor in descriptors:
            assert store.download(descriptor, lambda: False, download)
        monkeypatch.setenv("TEUTONIC_DOC_INDEX_STORE_DIR", str(root))

        def local_object(descriptor, cache):
            reader = PinnedObject(descriptor, cache, _server.open)
            assert (reader.local is not None) == (descriptor in descriptors)
            return reader

        kwargs["index_factory"] = lambda category, cache: DocumentIndex(category, cache, local_object)
    second, again = extend_with_long_documents(req, windows, meta, **kwargs)
    assert result == second and audit == again
    full_dataset_bytes = sum(path.stat().st_size for path in paths.values())
    assert first_transferred < full_dataset_bytes
    assert server.transferred - first_transferred < full_dataset_bytes
    assert audit["long_documents"]["n_documents"] >= 3
    assert sum(p["component"] == "windows" for p in audit["_sample_provenance"]) == len(windows)
    assert audit["n"] == len(result)
    from teutonic.evaluation.masking import scored_token_counts
    assert all(n > 0 for n in scored_token_counts(result))
    for tokens, p in zip(result, audit["_sample_provenance"], strict=True):
        if p["component"] == "windows":
            assert tokens == [12, 13]
        else:
            assert tokens == documents[p["document_row"]].tolist()
            assert 2049 <= len(tokens) <= 16384
            assert p["shard_index"] == -1
    assert audit["digest"] == sequence_digest(result)
    assert any(len(tokens) == 16384 for tokens in result)
    buckets = audit["long_documents"]["categories"][0]["buckets"]
    assert [b["target_tokens"] for b in buckets] == [6000, 9000]
    req.max_document_tokens = 8192
    with pytest.raises(ValueError, match="context limit"):
        extend_with_long_documents(req, windows, meta, **kwargs)
    req.max_document_tokens = 16384
    assert sequence_digest([[1, 2], [3, 4, 5]]) != sequence_digest([[1, 2, 3], [4, 5]])
    req.vocab_size = 100
    with pytest.raises(ValueError, match="vocabulary"):
        extend_with_long_documents(req, windows, meta, **kwargs)


def test_incomplete_documents_are_rejected_by_index(tmp_path):
    config, _snap, factory, _documents, _paths, server = make_index_fixture(tmp_path)
    category = config["categories"][0]
    url = category["files"]["tokens"]["url"]
    lengths = np.load(io.BytesIO(server.objects[url]))
    lengths[-1] += 1
    server.objects[url] = npy_bytes(lengths)
    category["files"]["tokens"] = server.describe(url)
    index = factory(category, tmp_path)
    assert index.locate(len(lengths) - 1) is None
    assert index.locate(1) is not None


def test_protocol_keeps_index_identity_in_request_hash():
    from test_evaluator_protocol_v2 import request_payload

    from teutonic.evaluation import EvaluationRequestV2, ProtocolValidationError

    config = build_long_document_config(all_sources(), describe=descriptor)
    settings = EvaluationSettings("f" * 64, "fixture", 25000, 0.003, all_sources(), 4, config)
    payload = request_payload()
    payload["dataset"] = pretokenized_dataset_request(
        settings, block_hash="b", hotkey="h", seq_len=2048
    )
    parsed = EvaluationRequestV2.from_mapping(payload)
    assert parsed.dataset["long_documents"] == config
    payload["dataset"]["long_documents"]["token_budget"] += 1
    assert EvaluationRequestV2.from_mapping(payload).request_sha256 != parsed.request_sha256
    payload["dataset"]["long_documents"]["categories"][0]["weight"] = -1
    with pytest.raises(ProtocolValidationError):
        EvaluationRequestV2.from_mapping(payload)


def test_documents_only_configuration_policy_and_request():
    from test_evaluator_protocol_v2 import request_payload

    from teutonic.evaluation import EvaluationRequestV2, ProtocolValidationError
    from teutonic.evaluation.early_stopping import EarlyStoppingPolicy
    from teutonic.validator.contracts import EvaluationPolicyConfig

    sources = all_sources()
    config = build_long_document_config(sources)
    settings = EvaluationSettings("f" * 64, "fixture", 0, 0.003, sources, 4, config)
    policy = EvaluationPolicyConfig(
        policy_version="fixture", code_version="fixture", dataset_version=settings.config_version,
        evaluator_version="fixture", sampling_seed=0, bootstrap_seed=0,
        n=0, seq_len=2048, n_bootstrap=100, alpha=0.01, delta_threshold=0.003,
        dataset_source="pretokenized_npy", dataset_label="fixture", shards_per_dataset=4,
        dataset_manifests=sources, long_documents=config,
        early_stopping=EarlyStoppingPolicy(enabled=True),
    )
    payload = request_payload()
    payload["limits"] = policy.thresholds
    payload["dataset"] = policy.dataset_request(block_hash="b", hotkey="h")
    parsed = EvaluationRequestV2.from_mapping(payload)
    assert parsed.limits["n"] == 0
    assert all(s["target_sequences"] == 0 and s["shards"] == [] for s in parsed.dataset["sources"])
    assert parsed.dataset["long_documents"] == config
    for invalid in (None, {}):
        payload["dataset"]["long_documents"] = invalid
        with pytest.raises(ProtocolValidationError):
            EvaluationRequestV2.from_mapping(payload)
    with pytest.raises(ValueError, match="zero windows"):
        EvaluationSettings("f" * 64, "fixture", 0, 0.003, sources, 4, None)


@pytest.mark.parametrize("mutation", ["weak_etag", "weight", "duplicate", "context"])
def test_configuration_rejects_ambiguous_or_invalid_pins(mutation):
    config = build_long_document_config(all_sources(), describe=descriptor)
    if mutation == "weak_etag":
        config["categories"][0]["files"]["tokens"]["etag"] = 'W/"abc"'
    elif mutation == "weight":
        config["categories"][0]["weight"] = float("nan")
    elif mutation == "duplicate":
        config["categories"].append(config["categories"][0])
    else:
        config["max_document_tokens"] = 8193
    with pytest.raises(ValueError):
        validate_long_documents(config)


@pytest.mark.parametrize("cap", [0, 2048, 4096, 8192, 16383, 16385, 128974, True])
def test_configuration_rejects_caps_outside_current_policy(cap):
    config = build_long_document_config(all_sources())
    config["max_document_tokens"] = cap
    with pytest.raises(ValueError, match="16384"):
        validate_long_documents(config)


@pytest.mark.parametrize("version", ["long-documents-v1", "long-documents-2k-8k-v1", "whole-documents-0-8k-v2"])
def test_previous_policy_is_not_silently_replayed(version):
    config = build_long_document_config(all_sources())
    config.update(version=version)
    with pytest.raises(ValueError, match="unsupported"):
        validate_long_documents(config)


def test_budget_changes_preserve_pins_and_threshold_updates_do_not_fetch(monkeypatch):
    from scripts import configure_evaluation as cli
    from teutonic.evaluation.mixes import build_mix_manifests

    manifest = build_mix_manifests(all_sources(), {"main": chain_config.COMPETITION_DEFAULTS["main"]})["main"]
    sources = (snapshot("main", manifest.pop("shards"), **manifest),)
    config = build_long_document_config(sources, describe=descriptor)
    current = {
        "dataset_label": "main",
        "n": 30000,
        "delta_threshold": 0.003,
        "manifests": sources,
        "shards_per_dataset": 4,
        "long_documents": config,
    }
    monkeypatch.setattr(
        cli, "build_long_document_config", lambda *a, **kw: pytest.fail("repinned index")
    )
    updated, _ = cli.prepare_configuration(
        current, "main", cli.parse_args(["--long-document-tokens", "7000000"])
    )
    assert updated["long_documents"]["token_budget"] == 7000000
    assert updated["long_documents"]["categories"] == config["categories"]

    previous = copy.deepcopy(current)
    previous["long_documents"].update(version="whole-documents-0-8k-v2", max_document_tokens=8192)
    documents_only, version = cli.prepare_configuration(
        previous, "main", cli.parse_args(["--n", "0", "--long-document-tokens", "70000000"])
    )
    assert documents_only["n"] == 0
    assert documents_only["long_documents"]["token_budget"] == 70_000_000
    assert documents_only["long_documents"]["max_document_tokens"] == 16384
    assert documents_only["long_documents"]["version"] == LONG_DOCUMENT_VERSION
    assert documents_only["long_documents"]["index_manifest"] == config["index_manifest"]
    assert documents_only["long_documents"]["categories"] == config["categories"]
    assert version != evaluation_config_version(**current)
    with pytest.raises(ValueError, match="zero windows"):
        cli.prepare_configuration(
            documents_only, "main", cli.parse_args(["--long-document-tokens", "0"])
        )
    assert config["token_budget"] == 6000000
    unchanged, _ = cli.prepare_configuration(
        current, "main", cli.parse_args(["--delta-threshold", ".005"])
    )
    assert unchanged["long_documents"] == config
    disabled, _ = cli.prepare_configuration(
        current, "main", cli.parse_args(["--long-document-tokens", "0"])
    )
    assert disabled["long_documents"] is None

    previous = copy.deepcopy(current)
    previous["long_documents"].update(version="long-documents-v1", max_document_tokens=0)
    updated, _ = cli.prepare_configuration(
        previous, "main", cli.parse_args(["--long-document-tokens", "6000000"])
    )
    assert updated["long_documents"]["version"] == LONG_DOCUMENT_VERSION
    assert updated["long_documents"]["max_document_tokens"] == 16384
    assert updated["long_documents"]["index_manifest"] == config["index_manifest"]
    assert updated["long_documents"]["categories"] == config["categories"]


def test_inventory_pin_hashes_exact_bytes_and_rejects_tampering():
    from teutonic.evaluation.index_manifest import pin_index_manifest, validate_index_manifest_pin

    config = build_long_document_config(all_sources())
    pin = config['index_manifest']
    raw = pin['content'].encode()
    with pytest.raises(ValueError, match='SHA-256'):
        validate_index_manifest_pin({**pin, 'sha256': None})
    assert pin['sha256'] == hashlib.sha256(raw).hexdigest()
    assert pin_index_manifest(raw, pin['url'], pin['sha256']) == pin
    with pytest.raises(ValueError, match='SHA-256'):
        pin_index_manifest(raw + b'\n', pin['url'], pin['sha256'])
    bad = {**pin, 'content': pin['content'].replace('151645', '151644')}
    with pytest.raises(ValueError, match='SHA-256'):
        validate_index_manifest_pin(bad)
    manifest = json.loads(pin['content'])
    manifest['files'].pop(next(iter(manifest['files'])))
    with pytest.raises(ValueError, match='missing'):
        pin_index_manifest(json.dumps(manifest).encode(), pin['url'])


def test_pinned_inventory_is_replayable_without_refetching(monkeypatch):
    from teutonic.evaluation import index_manifest

    config = build_long_document_config(all_sources())
    monkeypatch.setattr(index_manifest, 'fetch_index_manifest', lambda *a, **kw: pytest.fail('re-fetched mutable manifest'))
    assert build_config(all_sources(), index_manifest=config['index_manifest']) == config
    with pytest.raises(ValueError, match='differ'):
        changed = copy.deepcopy(config)
        changed['categories'][0]['files']['tokens']['sha256'] = 'f' * 64
        validate_long_documents(changed)


def test_source_manifest_lineage_is_checked():
    config = build_long_document_config(all_sources())
    sources = list(all_sources())
    source = sources[0]
    sources[0] = snapshot(source.name, list(source.manifest['shards']) + [
        {**source.manifest['shards'][0], 'key': source.manifest['shards'][0]['key'].replace('fixture', 'another')}
    ], source.proportion)
    with pytest.raises(ValueError, match='source manifest differs'):
        build_config(sources, index_manifest=config['index_manifest'])


def test_inventory_fetch_checks_expected_hash():
    from teutonic.evaluation.index_manifest import fetch_index_manifest

    pin = build_long_document_config(all_sources())['index_manifest']
    calls = []
    def open_manifest(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(pin['content'].encode())
    assert fetch_index_manifest(pin['url'], pin['sha256'], opener=open_manifest) == pin
    assert calls == [pin['url']]
    with pytest.raises(ValueError, match='SHA-256'):
        fetch_index_manifest(pin['url'], '0' * 64, opener=open_manifest)


def test_inventory_builder_hashes_parallel_ranges_in_file_order():
    from scripts.build_document_index_manifest import hash_remote_ranges

    body = bytes(range(256)) * 1024
    server = RangeServer({'https://index.example/x': body})
    descriptor = server.describe('https://index.example/x')
    assert hash_remote_ranges(descriptor, 3, chunk_bytes=10000, opener=server.open) == hashlib.sha256(body).hexdigest()
    server.wrong_range = True
    with pytest.raises(ValueError, match='range'):
        hash_remote_ranges(descriptor, 3, chunk_bytes=10000, opener=server.open)


def test_metadata_inventory_build_never_downloads_index_bodies(tmp_path, monkeypatch):
    from scripts import build_document_index_manifest as builder
    from teutonic.evaluation.index_manifest import ETAG_INDEX_FORMAT, pin_index_manifest
    sources = tmp_path / "sources.json"
    output = tmp_path / "inventory.json"
    sources.write_text(json.dumps({"datasets": {"dclm-baseline-1.0": {
        "categories": ["dclm-baseline-1.0"],
        "source_manifest": {"url": "https://datasets.example/dclm/manifest.json", "sha256": "a" * 64}}}}))
    monkeypatch.setattr(builder, "describe_index_file", lambda url: {
        "url": url, "etag": '"abc123-20"', "size_bytes": 1000})
    monkeypatch.setattr(builder, "urlopen", lambda *a, **kw: pytest.fail("index body download"))
    monkeypatch.setattr(builder, "hash_remote_ranges", lambda *a, **kw: pytest.fail("remote hashing"))
    args = ["--sources", str(sources), "--output", str(output),
            "--manifest-url", "https://datasets.example/doc_index/new.json"]
    with pytest.raises(SystemExit):
        builder.main(args)
    builder.main([*args, "--metadata-only"])
    manifest = json.loads(output.read_text())
    assert manifest["format"] == ETAG_INDEX_FORMAT
    assert len(manifest["files"]) == 4
    assert all(set(f) == {"etag", "size_bytes"} for f in manifest["files"].values())
    pin_index_manifest(output.read_bytes(), args[-1])
    manifest["format"] = "teutonic-document-index-v1"
    with pytest.raises(ValueError, match="SHA-256"):
        pin_index_manifest(json.dumps(manifest).encode(), args[-1])
