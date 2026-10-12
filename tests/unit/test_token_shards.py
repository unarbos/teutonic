import ast
import base64
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pytest

from teutonic.evaluator import document_index
from teutonic.evaluator.document_index import RangeDownloadError
from teutonic.evaluator.token_shards import MAX_RANGE_BYTES, NpyShard


class ShardServer:
    def __init__(self, objects, *, metadata=True):
        self.objects = objects
        self.metadata = metadata
        self.calls = []
        self.transferred = 0
        self.body_reads = 0
        self.status = 206
        self.wrong_range = False
        self.truncate = False
        self.digest_override = None
        self.checksum_type = None
        self.checksum = None

    def open(self, request, timeout):
        assert timeout == 120
        assert request.get_method() == "GET"
        assert request.get_header("If-match") is None
        assert request.get_header("Accept-encoding") == "identity"
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        body = self.objects[request.full_url]
        assert 0 <= start <= end < len(body)
        self.calls.append((start, end))
        server = self

        class Response(io.BytesIO):
            def read(self, size=-1):
                assert 0 < size <= MAX_RANGE_BYTES + 1
                data = super().read(size)
                server.transferred += len(data)
                server.body_reads += 1
                return data

        response = Response(body[start : end + 1 - int(self.truncate)])
        response.status = self.status
        response.headers = {
            "Content-Range": f"bytes {start}-{end}/{len(body) + int(self.wrong_range)}",
        }
        if self.metadata:
            response.headers["x-amz-meta-sha256"] = (
                self.digest_override or hashlib.sha256(body).hexdigest()
            )
        if self.checksum_type:
            response.headers["x-amz-checksum-type"] = self.checksum_type
            response.headers["x-amz-checksum-sha256"] = self.checksum
        return response


def shard_fixture(array, tmp_path, *, metadata=True):
    stream = io.BytesIO()
    np.save(stream, array)
    body = stream.getvalue()
    ref = SimpleNamespace(
        source="fixture",
        url="https://datasets.example/shard.npy",
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
        n_tokens=array.size,
    )
    server = ShardServer({ref.url: body}, metadata=metadata)
    return ref, server


def old_loader():
    """Exercise the actual legacy loader without importing torch/transformers."""
    path = Path(__file__).parents[2] / "teutonic/evaluator/engine.py"
    tree = ast.parse(path.read_text())
    names = {"read_npy_header", "shuffled_indices", "load_indexed_sequences_from_npy_shard"}
    functions = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
        type_ignores=[],
    )
    namespace = {"ast": ast, "io": io, "np": np, "EvalRequest": SimpleNamespace}
    exec(compile(functions, str(path), "exec"), namespace)  # noqa: S102 - only checked-in code
    return SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.mark.parametrize("layout", ["stream", "matrix", "fortran"])
@pytest.mark.parametrize("limit", [3, 1024])
def test_range_sampling_matches_actual_local_loader(tmp_path, layout, limit):
    from test_evaluator_sources import load_sources_module

    array = np.arange(1024 * 64, dtype="<u4").reshape(1024, 64)
    if layout == "stream":
        array = np.append(array.reshape(-1), np.array([42, 43], dtype="<u4"))
    elif layout == "fortran":
        array = np.asfortranarray(array)
    ref, server = shard_fixture(array, tmp_path)
    path = tmp_path / "reference.npy"
    path.write_bytes(server.objects[ref.url])
    legacy = old_loader()
    sources = load_sources_module(tmp_path)
    sources.base.shuffled_indices = legacy.shuffled_indices
    sources.open_shard = lambda ref: NpyShard(ref, opener=server.open)
    req = SimpleNamespace(seq_len=64)
    expected = legacy.load_indexed_sequences_from_npy_shard(
        str(path), req, np.random.default_rng(123), limit
    )
    events = []
    actual = sources._load_with_retry(ref, req, np.random.default_rng(123), limit, events.append)
    assert actual == expected
    assert events[-1]["metadata_sha256_checked"] is True
    assert events[-1]["size_bytes"] == server.transferred
    if limit == 3 and layout != "fortran":
        assert server.transferred < ref.size_bytes // 10
    calls = len(server.calls)
    repeated = sources._load_with_retry(ref, req, np.random.default_rng(123), limit)
    assert repeated == expected
    assert len(server.calls) == calls * 2
    assert sorted(tmp_path.iterdir()) == [path]  # Only the legacy reference file exists.


def test_sparse_large_shard_download_without_disk_cache(tmp_path):
    array = np.arange(4096 * 2048, dtype="<u4").reshape(4096, 2048)
    ref, server = shard_fixture(array, tmp_path, metadata=False)
    reader = NpyShard(ref, opener=server.open)
    indices = [4000, 2, 2048, 100, 2]
    rows = reader.read_sequences(indices, 2048)
    for index, row in zip(indices, rows):
        np.testing.assert_array_equal(row, array[index])
    assert server.transferred == 1024 + 4 * 2048 * 4
    assert server.transferred < ref.size_bytes / 900
    assert not reader.reader.metadata_sha256_checked
    assert reader.reader.cache is None
    assert not list(tmp_path.iterdir())
    before = len(server.calls)
    again = NpyShard(ref, opener=server.open)
    for index, row in zip(indices, again.read_sequences(indices, 2048)):
        np.testing.assert_array_equal(row, array[index])
    assert len(server.calls) == before * 2
    assert again.reader.downloaded_bytes == 33792
    assert not list(tmp_path.iterdir())


def test_token_reader_ignores_legacy_cache_and_never_accesses_disk(tmp_path, monkeypatch):
    from test_evaluator_sources import load_sources_module

    array = np.arange(10000, dtype="<u4")
    ref, server = shard_fixture(array, tmp_path)
    descriptor = {"url": ref.url, "size_bytes": ref.size_bytes, "sha256": ref.sha256}
    legacy = document_index.RangeObject(descriptor, tmp_path / "ranges", server.open)
    legacy._range(0, 1023, "0-1024")
    legacy._range(128 + 2500 * 4, 128 + 2503 * 4 - 1, "10128-12")
    calls = len(server.calls)
    monkeypatch.setenv("TEUTONIC_PRETOKENIZED_CACHE_DIR", str(tmp_path))
    sources = load_sources_module(tmp_path)
    sources.NpyShard = lambda shard: NpyShard(shard, opener=server.open)

    def disk_access(*_args, **_kwargs):
        pytest.fail("token range reader accessed the disk cache")

    # The legacy cache is valid and populated, but the token reader must ignore it.
    with monkeypatch.context() as no_disk:
        for name in ("exists", "read_bytes", "mkdir", "replace"):
            no_disk.setattr(Path, name, disk_access)
        no_disk.setattr(document_index.tempfile, "NamedTemporaryFile", disk_access)
        reader = sources.open_shard(ref)
        np.testing.assert_array_equal(reader.read_tokens([(2500, 3)])[0], array[2500:2503])
    assert len(server.calls) == calls + 2
    assert reader.reader.cache is None


@pytest.mark.parametrize("failure", ["full_response", "wrong_range", "metadata"])
def test_rejects_bad_response_without_reading_body(tmp_path, failure):
    ref, server = shard_fixture(np.arange(10000, dtype="<u4"), tmp_path)
    if failure == "full_response":
        server.status = 200
    elif failure == "wrong_range":
        server.wrong_range = True
    else:
        server.digest_override = "0" * 64
    with pytest.raises(RuntimeError, match="range|SHA-256"):
        NpyShard(ref, opener=server.open)
    assert server.body_reads == 0
    assert not list(tmp_path.rglob("*.range"))


@pytest.mark.parametrize(
    "checksum_type, mismatch", [("FULL_OBJECT", False), ("FULL_OBJECT", True), ("COMPOSITE", True)]
)
def test_whole_object_metadata_checksum(tmp_path, checksum_type, mismatch):
    ref, server = shard_fixture(np.arange(10000, dtype="<u4"), tmp_path, metadata=False)
    server.checksum_type = checksum_type
    server.checksum = base64.b64encode(bytes.fromhex("0" * 64 if mismatch else ref.sha256)).decode()
    if checksum_type == "FULL_OBJECT" and mismatch:
        with pytest.raises(RuntimeError, match="SHA-256"):
            NpyShard(ref, opener=server.open)
    else:
        reader = NpyShard(ref, opener=server.open)
        assert reader.reader.metadata_sha256_checked == (checksum_type == "FULL_OBJECT")


def test_truncated_ranges_retry_then_fail_without_cached_partial(tmp_path, monkeypatch):
    ref, server = shard_fixture(np.arange(10000, dtype="<u4"), tmp_path)
    server.truncate = True
    monkeypatch.setattr(document_index.time, "sleep", lambda _: None)
    with pytest.raises(RangeDownloadError, match="exhausted"):
        NpyShard(ref, opener=server.open)
    assert len(server.calls) == 5
    assert not list(tmp_path.rglob("*.range*"))


def test_transient_http_failure_recovers(tmp_path, monkeypatch):
    ref, server = shard_fixture(np.arange(10000, dtype="<u4"), tmp_path)
    calls = []

    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(ref.url, 503, "busy", {}, None)
        return server.open(request, timeout)

    monkeypatch.setattr(document_index.time, "sleep", lambda _: None)
    reader = NpyShard(ref, opener=opener)
    np.testing.assert_array_equal(reader.read_tokens([(2500, 3)])[0], [2500, 2501, 2502])
    assert len(calls) == 3


@pytest.mark.parametrize("invalid", ["dtype", "tokens", "size", "dimensions"])
def test_validates_npy_descriptor(tmp_path, invalid):
    array = np.arange(10000, dtype="<u4")
    if invalid == "dtype":
        array = array.astype("<u2")
    elif invalid == "dimensions":
        array = array.reshape(10, 10, 100)
    ref, server = shard_fixture(array, tmp_path)
    if invalid == "tokens":
        ref.n_tokens += 1
    elif invalid == "size":
        # Correct Content-Range but invalid NPY data size.
        server.objects[ref.url] += b"extra"
        ref.size_bytes += 5
        ref.sha256 = hashlib.sha256(server.objects[ref.url]).hexdigest()
    with pytest.raises(ValueError, match="dtype|token count|size|matrix"):
        NpyShard(ref, opener=server.open)


def test_bounds_and_sequence_width_checked_before_token_reads(tmp_path):
    ref, server = shard_fixture(np.arange(1024, dtype="<u4").reshape(16, 64), tmp_path)
    reader = NpyShard(ref, opener=server.open)
    for span in [(-1, 2), (1023, 2), (0, 0)]:
        with pytest.raises(ValueError, match="outside"):
            reader.read_tokens([span])
    with pytest.raises(ValueError, match="width"):
        reader.read_sequences([0], 32)
    assert len(server.calls) == 1


def test_dense_reads_remain_bounded_and_preserve_order(tmp_path):
    array = np.arange(1025 * 2048, dtype="<u4").reshape(1025, 2048)
    ref, server = shard_fixture(array, tmp_path)
    reader = NpyShard(ref, opener=server.open)
    indices = np.arange(1025)[::-1]
    rows = reader.read_sequences(indices, 2048)
    for i, row in zip(indices, rows, strict=True):
        np.testing.assert_array_equal(row, array[i])
    assert len(server.calls) == 3  # header plus two bounded data ranges
    assert max(end - start + 1 for start, end in server.calls) == MAX_RANGE_BYTES


def test_dense_scattered_windows_merge_but_sparse_fragments_stay_small(tmp_path):
    array = np.arange(4096 * 2048, dtype="<u4").reshape(4096, 2048)
    ref, server = shard_fixture(array, tmp_path)
    reader = NpyShard(ref, opener=server.open)
    # 1/8 coverage: formerly one request per row (gaps exceed 16 KiB).
    indices = np.arange(0, 4096, 8)[::-1]
    for i, row in zip(indices, reader.read_sequences(indices, 2048), strict=True):
        np.testing.assert_array_equal(row, array[i])
    assert len(server.calls) == 5  # header + four ranges, bounded to 8 MiB each
    before = server.transferred
    spans = [(80 * 2048, 200), (88 * 2048, 200)]
    for (start, length), tokens in zip(spans, reader.read_tokens(spans), strict=True):
        np.testing.assert_array_equal(tokens, array.reshape(-1)[start : start + length])
    assert server.transferred - before == 400 * 4
    assert len(server.calls) == 7


@pytest.mark.parametrize("version", [(2, 0), (3, 0)])
def test_newer_npy_headers(tmp_path, version):
    array = np.arange(10000, dtype="<u4")
    ref, server = shard_fixture(array, tmp_path)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=version)
    server.objects[ref.url] = stream.getvalue()
    ref.size_bytes = len(stream.getvalue())
    ref.sha256 = hashlib.sha256(stream.getvalue()).hexdigest()
    reader = NpyShard(ref, opener=server.open)
    np.testing.assert_array_equal(reader.read_tokens([(9876, 10)])[0], array[9876:9886])


def test_oversized_header_rejected_before_further_downloads(tmp_path):
    ref, server = shard_fixture(np.arange(10000, dtype="<u4"), tmp_path, metadata=False)
    raw = bytearray(server.objects[ref.url])
    raw[8:10] = (60000).to_bytes(2, "little")
    server.objects[ref.url] = bytes(raw)
    with pytest.raises(ValueError, match="header length"):
        NpyShard(ref, opener=server.open)
    assert server.transferred == 1024
