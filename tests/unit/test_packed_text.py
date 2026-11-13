import hashlib
import io
from dataclasses import replace

import numpy as np
import pytest

from teutonic.evaluation.packed_text import (
    BUCKET,
    CATALOG_FORMAT,
    DTYPE,
    FORMAT,
    INDEX_PAGE_BYTES,
    MIX_FORMAT,
    PREFIX,
    DatasetReader,
    IndexPages,
    PackedDataset,
    build_catalog,
    canonical,
    validate_mix,
    validate_pin,
)
from teutonic.evaluation.reference_policy import reference_verdict
from teutonic.evaluator.text_pipeline import prepared_batches
from teutonic.evaluator.text_sampling import sample_documents


class ReadOnlyR2:
    def __init__(self):
        self.objects, self.calls = {}, []

    def get_object(self, *, Bucket, Key, Range=None, IfMatch=None):
        assert Bucket == "dataset", "legacy bucket access"
        self.calls.append((Key, Range))
        raw = self.objects[Key]
        etag = '"' + hashlib.md5(raw).hexdigest() + '"'
        if IfMatch is not None:
            assert IfMatch == etag
        if Range is None:
            return {"Body": io.BytesIO(raw)}
        start, end = map(int, Range.removeprefix("bytes=").split("-"))
        return {"Body": io.BytesIO(raw[start:end+1]), "ETag": etag,
                "ContentRange": f"bytes {start}-{end}/{len(raw)}"}

    def add(self, path, raw, **metadata):
        self.objects[PREFIX + path] = raw
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw), "etag": '"' + hashlib.md5(raw).hexdigest() + '"', **metadata}


def fixture():
    client, objects = ReadOnlyR2(), []
    # Unequal partitions containing both bucket edges. Physical order differs from index order.
    for shard, lengths in enumerate(([4097, 2, 4096, 16384], [4097]*36 + [4096]*36)):
        blob, rows = b"", []
        for i, length in enumerate(lengths):
            text = f"żółw {shard} {i}\n".encode()
            identity = hashlib.sha256(f"{shard}:{i}".encode()).digest()
            rows.append((identity, shard, len(blob), len(text), length, hashlib.sha256(text).digest()))
            blob += text
        common = {"dataset": "test", "category": "a", "shard_id": shard, "document_count": len(rows)}
        objects.append(client.add(f"text/test/a/part-{shard}.utf8", blob, role="text", **common))
        rows.sort(key=lambda row: (row[4], row[0]))
        stream = io.BytesIO()
        np.save(stream, np.array(rows, dtype=DTYPE), allow_pickle=False)
        objects.append(client.add(f"index/test/a/part-{shard}.documents.npy", stream.getvalue(),
                                  role="index", min_tokens=min(lengths), max_tokens=max(lengths), **common))
    raw = canonical({"format": FORMAT, "objects": objects})
    client.add("manifest.json", raw)
    pin = {"bucket": BUCKET, "key": PREFIX + "manifest.json", "sha256": hashlib.sha256(raw).hexdigest()}
    dataset = PackedDataset(DatasetReader(client), pin)
    catalog = build_catalog(dataset)
    mix = {"format": MIX_FORMAT, "competition": "main", "dataset": pin,
           "sampling": {"bucket": BUCKET, "key": PREFIX + "sampling/test.json", "sha256": "a"*64},
           "token_budget": 40000, "categories": [{"dataset": "test", "category": "a", "weight": 1}]}
    return client, dataset, catalog, mix


def test_manifest_ranges_and_utf8_hash():
    client, dataset, catalog, _ = fixture()
    assert DTYPE.itemsize == 92
    assert catalog["format"] == CATALOG_FORMAT
    assert not any(key.endswith(".utf8") for key, _ in client.calls)
    for obj in dataset.objects.values():
        if obj["role"] == "index":
            for i in range(obj["document_count"]):
                doc = dataset.document(obj["path"], i)
                assert dataset.text(doc).startswith("żółw")
                with pytest.raises(ValueError, match="hash"):
                    dataset.text(replace(doc, text_sha256="0"*64))


def test_deterministic_partition_sampling_and_population():
    _, dataset, catalog, mix = fixture()
    docs, meta = sample_documents(dataset, catalog, mix, 123)
    assert (docs, meta) == sample_documents(dataset, catalog, mix, 123)
    assert len({d.doc_id for d in docs}) == len(docs)
    assert [x["population"] for x in meta["categories"]] == [38, 38]
    # One draw from each bucket at tiny budgets: partitions selected by document count.
    mix["token_budget"] = 1
    large = 0
    for seed in range(200):
        docs, _ = sample_documents(dataset, catalog, mix, seed)
        large += sum(d.shard_id == 1 for d in docs)
    assert 350 < large < 400  # expected ~379, not 200 from uniform shard sampling


@pytest.mark.parametrize("seed,budget,expected", [
    (0, 1, "7c0a43a443a92a3ae55ab9683df4fe4752e326cebaa057d7575e557e93dfe04d"),
    (123, 40000, "34e91b95f30cf44a47a13ed2414a9d026a67ba67db76fd02593b6dd2d9dad725"),
    (731, 120000, "c4a553fe92c7db497d752af95aa5a8bd130067a491d06c58e85c8983fa57ac8a"),
    (999, 190000, "3fd5a8bfd4ccb7f1c4e0fe99e542e263c32f7916196913e94e47e3ec23f7577d"),
])
@pytest.mark.parametrize("batch_size", [1, 3, 64, 2048])
def test_parallel_sampling_preserves_previous_sampler_digests(seed, budget, expected, batch_size):
    # Golden digests recorded from the serial implementation, before this change.
    _, dataset, catalog, mix = fixture()
    mix["token_budget"] = budget
    _, meta = sample_documents(dataset, catalog, mix, seed, candidate_batch_size=batch_size)
    assert meta["digest"] == expected


def test_small_quota_does_not_prefetch_thousands_of_unused_index_rows(monkeypatch):
    _, dataset, catalog, mix = fixture()
    template = dataset.document(catalog['partitions'][0]['path'], 0)
    for row in catalog['partitions']:
        row['lo'], row['hi'] = (0, 100000) if row['bucket'] == '2-4096' else (100000, 200000)
        dataset.objects[row['path']]['document_count'] = 200000
    mix['token_budget'] = 10000
    requested = []

    def documents(references, executor):
        requested.append(len(references))
        return [replace(template, doc_id=hashlib.sha256(f'{path}:{number}'.encode()).hexdigest(),
                        row=number,
                        original_mimo_token_count=2000 if number < 100000 else 8000)
                for path, number in references]

    monkeypatch.setattr(dataset, 'documents', documents)
    selected = sample_documents(dataset, catalog, mix, 123)
    assert sum(requested) < 100
    requested.clear()
    assert sample_documents(dataset, catalog, mix, 123, candidate_batch_size=2048) == selected
    assert sum(requested) > 1000


def test_grouped_index_reads_overlap_and_restore_request_order(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    client, original, _, _ = fixture()
    dataset = PackedDataset(DatasetReader(client), original.pin)
    references = [(path, number) for number in (1, 0) for path, obj in dataset.objects.items()
                  if obj["role"] == "index"]
    expected = [original.document(*ref) for ref in references]
    get = client.get_object
    barrier = threading.Barrier(2)
    def blocked_get(**kwargs):
        if kwargs.get("Range"):
            barrier.wait(timeout=5)  # Serial reads would fail this check.
        return get(**kwargs)
    monkeypatch.setattr(client, "get_object", blocked_get)
    client.calls.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert dataset.documents(references, pool) == expected
    assert len(client.calls) == 2  # Both tiny indexes fit in their cached header pages.
    monkeypatch.setattr(client, "get_object", get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert dataset.documents(references[::-1], pool) == expected[::-1]
    assert len(client.calls) == 2


def test_shared_index_cache_coalesces_ranges_and_bounds_memory():
    client = ReadOnlyR2()
    raw = bytes(range(256)) * (INDEX_PAGE_BYTES * 6 // 256)
    obj = client.add("index/test/pages.npy", raw)
    pages = IndexPages(DatasetReader(client), max_bytes=2 * INDEX_PAGE_BYTES)
    result = pages.read(obj, [1, 2, 3])
    assert [result[i] for i in (1, 2, 3)] == [raw[:INDEX_PAGE_BYTES]] * 3
    assert len(client.calls) == 1
    assert client.calls[0][1] == f"bytes={INDEX_PAGE_BYTES}-{4 * INDEX_PAGE_BYTES - 1}"
    assert pages.bytes <= pages.max_bytes
    pages.read(obj, [2, 3])
    assert len(client.calls) == 1
    pages.read(obj, [0, 5])
    assert len(client.calls) == 3  # Gaps are not downloaded.
    assert pages.bytes <= pages.max_bytes


def test_grouped_rows_cross_page_boundaries():
    from teutonic.evaluation.packed_text import Partition

    client = ReadOnlyR2()
    rows = np.zeros(1600, dtype=DTYPE)
    rows["original_mimo_token_count"] = np.arange(len(rows)) + 2
    stream = io.BytesIO()
    np.save(stream, rows, allow_pickle=False)
    obj = client.add("index/test/crossing.npy", stream.getvalue(), document_count=len(rows))
    partition = Partition(DatasetReader(client), obj)
    crossing = (INDEX_PAGE_BYTES - partition.offset) // DTYPE.itemsize
    assert (partition.offset + crossing * DTYPE.itemsize) % INDEX_PAGE_BYTES + DTYPE.itemsize > INDEX_PAGE_BYTES
    selected = [crossing + 1, 1599, crossing, 0, crossing]
    assert [int(row["original_mimo_token_count"]) for row in partition.rows(selected)] == [i + 2 for i in selected]


def test_corrupt_catalog_and_legacy_bucket_fail_closed():
    _, dataset, catalog, mix = fixture()
    bad = {**dataset.pin, "bucket": "datasets"}
    with pytest.raises(ValueError):
        validate_pin(bad)
    with pytest.raises(ValueError):
        validate_mix({**mix, "shards": []})
    catalog["partitions"][0]["hi"] = 9999
    with pytest.raises(ValueError):
        sample_documents(dataset, catalog, mix, 1)


def test_pipeline_order_bounds_and_no_truncation():
    _, dataset, catalog, mix = fixture()
    docs, _ = sample_documents(dataset, catalog, mix, 123)
    class Tokenizer:
        def encode(self, texts):
            return [[ord(c) for c in t] for t in texts]
    batches = list(prepared_batches(dataset, docs, Tokenizer(), batch_documents=2))
    assert [d for group, _ in batches for d in group] == docs
    with pytest.raises(ValueError, match="no truncation"):
        list(prepared_batches(dataset, docs, Tokenizer(), max_tokens=2))
    with pytest.raises(ValueError, match="byte bound"):
        list(prepared_batches(dataset, docs, Tokenizer(), max_document_bytes=1))


def verdict(k, c, r, sizes):
    scale = np.log(2) * np.asarray(sizes)
    return reference_verdict(np.asarray(k)*scale, np.asarray(c)*scale, np.asarray(r)*scale, sizes,
                             bootstrap_seed=3, n_bootstrap=100, alpha=.05, delta_threshold=.0015)


def test_deficit_reward_and_no_retention_veto():
    result = verdict([.7, 1.3], [.95, 1.1], [1., 1.], [10, 100])
    assert result["mu_hat"] == pytest.approx(.2)
    assert result["n_skipped_sequences"] == 1
    assert not result["retention_gates"]
    assert verdict([1.3]*10, [.8]*10, [1.]*10, [100]*10)["mu_hat"] == pytest.approx(.3)
    assert verdict([.9], [1.1], [1.], [100])["mu_hat"] == pytest.approx(-.1)
    assert verdict([.7], [.95], [1.], [100])["mu_hat"] == 0
    assert verdict([1.01]*10, [1.008]*10, [1.]*10, [100]*10)["accepted"]
    assert not verdict([1.01]*10, [1.009]*10, [1.]*10, [100]*10)["accepted"]


@pytest.mark.parametrize("sizes", [[0], [-1], [float("nan")]])
def test_invalid_loss_weights(sizes):
    with pytest.raises(ValueError):
        verdict([1.], [1.], [1.], sizes)
