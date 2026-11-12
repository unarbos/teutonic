"""Pinned packed-text inventories and document-only competition manifests.

All dataset access is read-only and restricted to the singular `dataset` bucket.
No publishing or deletion operations belong in this module.
"""

import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np

FORMAT = "teutonic-packed-text-v2"
MIX_FORMAT = "teutonic-packed-text-mix"
CATALOG_FORMAT = "teutonic-packed-text-sampling"
BUCKET = "dataset"
PREFIX = "teutonic-text/"
BUCKETS = (("2-4096", 2, 4096, 0.4), ("4097-16384", 4097, 16384, 0.6))
DTYPE = np.dtype([
    ("doc_id", "V32"), ("shard_id", "<u4"), ("byte_offset", "<u8"),
    ("byte_length", "<u8"), ("original_mimo_token_count", "<i8"),
    ("text_sha256", "V32"),
])
SHA = re.compile(r"^[0-9a-f]{64}$")
INDEX_PAGE_BYTES = 65536


class IndexPages:
    """Shared, byte-bounded index cache; coalesce adjacent missing pages per GET."""

    def __init__(self, reader, max_bytes=256 * 1024 * 1024):
        if max_bytes < INDEX_PAGE_BYTES:
            raise ValueError("index cache must hold at least one page")
        self.reader, self.max_bytes = reader, max_bytes
        self.cache, self.bytes = OrderedDict(), 0
        self.lock = threading.Lock()

    def read(self, obj, numbers):
        result, missing = {}, []
        identity = (obj["path"], obj["etag"])
        with self.lock:
            for number in sorted(set(numbers)):
                if not 0 <= number * INDEX_PAGE_BYTES < obj["size_bytes"]:
                    raise IndexError(number)
                key = (*identity, number)
                if key in self.cache:
                    result[number] = self.cache[key]
                    self.cache.move_to_end(key)
                else:
                    missing.append(number)
        # Each range is <=1 MiB, with no download of unrequested intervening pages.
        groups = []
        for number in missing:
            if groups and number == groups[-1][-1] + 1 and len(groups[-1]) < 16:
                groups[-1].append(number)
            else:
                groups.append([number])
        for group in groups:
            start = group[0] * INDEX_PAGE_BYTES
            size = min((group[-1] + 1) * INDEX_PAGE_BYTES, obj["size_bytes"]) - start
            raw = self.reader.read(obj, start, size)
            with self.lock:
                for number in group:
                    offset = (number - group[0]) * INDEX_PAGE_BYTES
                    value = raw[offset:offset + INDEX_PAGE_BYTES]
                    result[number] = value
                    key = (*identity, number)
                    old = self.cache.pop(key, b"")
                    self.bytes += len(value) - len(old)
                    self.cache[key] = value
                while self.bytes > self.max_bytes:
                    _, value = self.cache.popitem(last=False)
                    self.bytes -= len(value)
        return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid dataset path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError("dataset path must be a normalized relative path")
    return value


def validate_pin(pin):
    if not isinstance(pin, dict) or set(pin) != {"bucket", "key", "sha256"}:
        raise ValueError("dataset pin needs bucket, key and SHA-256")
    if pin["bucket"] != BUCKET or not pin["key"].startswith(PREFIX):
        raise ValueError("evaluation data must be under dataset/teutonic-text/")
    relative_path(pin["key"])
    if not isinstance(pin["sha256"], str) or not SHA.fullmatch(pin["sha256"]):
        raise ValueError("invalid dataset SHA-256")
    return dict(pin)


def validate_mix(value):
    required = {"format", "competition", "dataset", "sampling", "categories", "token_budget"}
    if not isinstance(value, dict) or set(value) != required or value["format"] != MIX_FORMAT:
        raise ValueError("expected a document-only packed-text mix")
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", value["competition"]):
        raise ValueError("invalid competition name")
    validate_pin(value["dataset"])
    validate_pin(value["sampling"])
    if type(value["token_budget"]) is not int or not 1 <= value["token_budget"] <= 100_000_000:
        raise ValueError("invalid document input-token budget")
    categories = value["categories"]
    if not isinstance(categories, list) or not categories:
        raise ValueError("empty category mixture")
    seen = set()
    for row in categories:
        if not isinstance(row, dict) or set(row) != {"dataset", "category", "weight"}:
            raise ValueError("invalid mixture category")
        identity = (row["dataset"], row["category"])
        if any(not isinstance(x, str) or not x for x in identity) or identity in seen:
            raise ValueError("duplicate or invalid category")
        seen.add(identity)
        if type(row["weight"]) not in (int, float) or not math.isfinite(row["weight"]) or row["weight"] <= 0:
            raise ValueError("invalid category weight")
    if not math.isclose(sum(x["weight"] for x in categories), 1, abs_tol=1e-9, rel_tol=0):
        raise ValueError("category weights must sum to one")
    return value


def read_client(*, max_pool_connections=32):
    import boto3
    from botocore.config import Config

    def env(name):
        return os.environ.get("TEUTONIC_EVALUATOR_" + name) or os.environ.get(name)

    endpoint = env("R2_ENDPOINT") or os.environ.get("TEUTONIC_R2_ENDPOINT")
    if not endpoint:
        raise ValueError("R2 endpoint is not configured")
    return boto3.client(
        "s3", endpoint_url=endpoint, region_name=os.environ.get("TEUTONIC_R2_REGION", "auto"),
        aws_access_key_id=env("R2_ACCESS_KEY_ID"), aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        aws_session_token=env("R2_SESSION_TOKEN"),
        config=Config(max_pool_connections=max_pool_connections, retries={"mode": "standard", "max_attempts": 5}),
    )


class DatasetReader:
    def __init__(self, client):
        self.client = client
        self._guard = threading.Lock()
        self._objects = {}

    @contextmanager
    def _object(self, key):
        # Include the body read in the critical section: R2 limits simultaneous
        # reads of one object. Remove idle locks so a large corpus stays bounded.
        with self._guard:
            entry = self._objects.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._guard:
                entry[1] -= 1
                if not entry[1]:
                    del self._objects[key]

    def _read(self, key, consume, **kwargs):
        from botocore.exceptions import ClientError, ConnectionClosedError, ReadTimeoutError

        with self._object(key):
            for attempt in range(8):
                try:
                    response = self.client.get_object(Bucket=BUCKET, Key=key, **kwargs)
                    # StreamingBody.__enter__ exposes the raw urllib3 stream,
                    # bypassing botocore's timeout translation. Keep the wrapper.
                    with closing(response["Body"]) as body:
                        return consume(response, body)
                except ClientError as exc:
                    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                    code = exc.response.get("Error", {}).get("Code")
                    if attempt == 7 or (status not in {429, 500, 502, 503, 504}
                                       and code not in {"ServiceUnavailable", "SlowDown"}):
                        raise
                except (ConnectionClosedError, ReadTimeoutError):
                    if attempt == 7:
                        raise
                delay = min(30, 2**attempt)
                logging.getLogger(__name__).warning(
                    "Retrying pinned R2 read key=%s attempt=%d delay_seconds=%d",
                    key, attempt + 1, delay,
                )
                time.sleep(delay)

    def json(self, pin):
        pin = validate_pin(pin)
        raw = self._read(pin["key"], lambda response, body: body.read())
        if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("manifest hash mismatch")
        return json.loads(raw)

    def read(self, obj, offset, length):
        path = relative_path(obj["path"])
        if offset < 0 or length < 0 or offset + length > obj["size_bytes"]:
            raise ValueError("range outside object")
        if not length:
            return b""
        end = offset + length - 1
        def consume(response, body):
            if response.get("ContentRange") != f"bytes {offset}-{end}/{obj['size_bytes']}":
                raise ValueError("server did not honor exact range")
            if response.get("ETag") != obj["etag"]:
                raise ValueError("dataset object changed")
            raw = body.read(length + 1)
            if len(raw) != length:
                raise ValueError("short or oversized range response")
            return raw
        return self._read(PREFIX + path, consume, Range=f"bytes={offset}-{end}", IfMatch=obj["etag"])


class Partition:
    """Sparse NPY reads pinned by the completed export's object ETag."""

    def __init__(self, reader, obj, *, pages=None):
        self.reader, self.obj = reader, obj
        self.pages = pages if pages is not None else IndexPages(reader, 16 * INDEX_PAGE_BYTES)
        # Fixed-width records; never allow object arrays or pickle payloads.
        header = io.BytesIO(self.pages.read(obj, [0])[0])
        version = np.lib.format.read_magic(header)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(header)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(header)
        else:
            raise ValueError("unsupported index NPY version")
        if dtype != DTYPE or fortran or len(shape) != 1:
            raise ValueError("invalid packed-text index layout")
        self.count, self.offset = shape[0], header.tell()
        if self.count != obj["document_count"] or self.offset + self.count * DTYPE.itemsize != obj["size_bytes"]:
            raise ValueError("index count/size mismatch")
    def row(self, number):
        return self.rows([number])[0]

    def rows(self, numbers):
        locations, required = [], set()
        for number in numbers:
            if not 0 <= number < self.count:
                raise IndexError(number)
            offset = self.offset + number * DTYPE.itemsize
            first, last = offset // INDEX_PAGE_BYTES, (offset + DTYPE.itemsize - 1) // INDEX_PAGE_BYTES
            locations.append((offset % INDEX_PAGE_BYTES, first, last))
            required.update(range(first, last + 1))
        pages = self.pages.read(self.obj, required)
        result = []
        for offset, first, last in locations:
            raw = b"".join(pages[i] for i in range(first, last + 1))
            result.append(np.frombuffer(raw[offset:offset + DTYPE.itemsize], dtype=DTYPE)[0])
        return result

    def bound(self, length, right=False):
        lo, hi = 0, self.count
        while lo < hi:
            mid = (lo + hi) // 2
            found = int(self.row(mid)["original_mimo_token_count"])
            if found < length or (right and found == length):
                lo = mid + 1
            else:
                hi = mid
        return lo


@dataclass(frozen=True)
class Document:
    doc_id: str
    dataset: str
    category: str
    partition: str
    row: int
    shard_id: int
    byte_offset: int
    byte_length: int
    original_mimo_token_count: int
    text_sha256: str


class PackedDataset:
    def __init__(self, reader, pin, *, text_cache=None):
        self.reader, self.pin = reader, validate_pin(pin)
        self.text_cache = Path(text_cache) if text_cache is not None else None
        if self.text_cache is not None:
            self.text_cache.mkdir(parents=True, exist_ok=True)
        self.manifest = reader.json(pin)
        if self.manifest.get("format") != FORMAT:
            raise ValueError("unsupported packed-text dataset")
        self.objects, self.texts = {}, {}
        for obj in self.manifest["objects"]:
            path = relative_path(obj["path"])
            if path in self.objects or not SHA.fullmatch(obj["sha256"]) or not obj.get("etag"):
                raise ValueError("invalid or duplicate dataset object")
            self.objects[path] = obj
            if obj["role"] == "text":
                if obj["shard_id"] in self.texts:
                    raise ValueError("duplicate text shard ID")
                self.texts[obj["shard_id"]] = obj
        self.index_pages = IndexPages(reader)
        # Partition objects retain only header metadata; page memory is bounded globally.
        self.partition = lru_cache(maxsize=4096)(self._partition)

    def _partition(self, path):
        obj = self.objects[path]
        if obj["role"] != "index":
            raise ValueError("expected an index partition")
        return Partition(self.reader, obj, pages=self.index_pages)

    def document(self, path, number):
        part = self.partition(path)
        return self._document(part, number, part.row(number))

    def documents(self, references, executor):
        """Group candidate rows by partition; return results in requested RNG order."""
        groups = {}
        for position, (path, number) in enumerate(references):
            groups.setdefault(path, []).append((position, number))

        def fetch(item):
            path, entries = item
            part = self.partition(path)
            rows = part.rows([number for _, number in entries])
            return [(position, self._document(part, number, row))
                    for (position, number), row in zip(entries, rows, strict=True)]

        result = [None] * len(references)
        for entries in executor.map(fetch, groups.items()):
            for position, doc in entries:
                result[position] = doc
        return result

    def _document(self, part, number, row):
        shard = int(row["shard_id"])
        obj = part.obj
        text = self.texts[shard]
        if any(text[k] != obj[k] for k in ("dataset", "category", "shard_id")):
            raise ValueError("index/text partition mismatch")
        return Document(
            bytes(row["doc_id"]).hex(), obj["dataset"], obj["category"], obj["path"], number, shard,
            int(row["byte_offset"]), int(row["byte_length"]),
            int(row["original_mimo_token_count"]), bytes(row["text_sha256"]).hex(),
        )

    def text(self, doc):
        cached = self.text_cache / doc.text_sha256 if self.text_cache is not None else None
        raw = cached.read_bytes() if cached is not None and cached.exists() else None
        if raw is not None and (len(raw) != doc.byte_length or hashlib.sha256(raw).hexdigest() != doc.text_sha256):
            cached.unlink(missing_ok=True)
            raw = None
        if raw is None:
            raw = self.reader.read(self.texts[doc.shard_id], doc.byte_offset, doc.byte_length)
        if len(raw) != doc.byte_length or hashlib.sha256(raw).hexdigest() != doc.text_sha256:
            raise ValueError("document text hash mismatch")
        text = raw.decode("utf-8", errors="strict")
        if cached is not None and not cached.exists():
            with tempfile.NamedTemporaryFile(dir=self.text_cache, delete=False) as out:
                temporary = Path(out.name)
                out.write(raw)
            temporary.replace(cached)
        return text


def build_catalog(dataset):
    """One-time boundary discovery; no text downloads and no remote writes."""
    rows = []
    for path, obj in sorted(dataset.objects.items()):
        if obj["role"] != "index":
            continue
        for name, minimum, maximum, _weight in BUCKETS:
            if obj["max_tokens"] < minimum or obj["min_tokens"] > maximum:
                continue
            lo = 0 if obj["min_tokens"] >= minimum else dataset.partition(path).bound(minimum)
            hi = obj["document_count"] if obj["max_tokens"] <= maximum else dataset.partition(path).bound(maximum, right=True)
            if hi > lo:
                rows.append({"dataset": obj["dataset"], "category": obj["category"],
                             "bucket": name, "path": path, "lo": lo, "hi": hi})
    return {"format": CATALOG_FORMAT, "dataset": dataset.pin,
            "buckets": [list(x) for x in BUCKETS], "partitions": rows}
