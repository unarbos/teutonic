"""Read pinned document indexes by HTTP range, without downloading 150+ GB."""

import hashlib
import io
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from teutonic.evaluation.long_documents import (
    LENGTH_BUCKETS,
    MAX_DOCUMENT_TOKENS,
    validate_index_file,
)

FRAG_DTYPE = np.dtype(
    [("shard_id", "<i4"), ("seq_index", "<u2"), ("seq_offset", "<u2"), ("frag_len", "<u2")]
)
PTR_DTYPE = np.dtype([("frag_offset", "<i8"), ("n_fragments", "<u2")])
PAGE_SIZE = 64 * 1024


class PinnedObject:
    def __init__(self, descriptor, cache_dir, opener=urlopen):
        validate_index_file(descriptor)
        self.descriptor = descriptor
        self.size = descriptor["size_bytes"]
        self.opener = opener
        identity = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        self.cache = Path(cache_dir) / identity
        self.page = lru_cache(maxsize=64)(self._page)

    def _page(self, number):
        start = number * PAGE_SIZE
        end = min(start + PAGE_SIZE, self.size) - 1
        if start < 0 or start > end:
            raise ValueError("index byte range is out of bounds")
        return self._range(start, end, str(number))

    def _range(self, start, end, cache_key):
        target = self.cache / cache_key
        if target.exists():
            blob = target.read_bytes()
            if (
                len(blob) == end - start + 1 + 32
                and hashlib.sha256(blob[32:]).digest() == blob[:32]
            ):
                return blob[32:]
        request = Request(
            self.descriptor["url"],
            headers={
                "User-Agent": "teutonic-eval/1.0",
                "Range": f"bytes={start}-{end}",
                "If-Match": self.descriptor["etag"],
                "Accept-Encoding": "identity",
            },
        )
        with self.opener(request, timeout=120) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
            ):
                raise RuntimeError("index server did not honor the exact byte range")
            if response.headers.get("ETag") != self.descriptor["etag"]:
                raise RuntimeError("document index changed since configuration was pinned")
            body = response.read(end - start + 2)
        if len(body) != end - start + 1:
            raise RuntimeError("truncated document index range")
        self.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.cache, delete=False) as tmp:
            path = Path(tmp.name)
            tmp.write(hashlib.sha256(body).digest() + body)
        try:
            path.replace(target)
        finally:
            path.unlink(missing_ok=True)
        return body

    def read(self, offset, size):
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError("index byte range is out of bounds")
        if size == 0:
            return b""
        # JSON shard-name inventories are tens of MB: use one range instead of
        # hundreds of serial page requests. Numeric tables still use sparse pages.
        if size > PAGE_SIZE:
            return self._range(offset, offset + size - 1, f"range-{offset}-{size}")
        first, last = offset // PAGE_SIZE, (offset + size - 1) // PAGE_SIZE
        data = b"".join(self.page(n) for n in range(first, last + 1))
        return data[offset % PAGE_SIZE : offset % PAGE_SIZE + size]


class NpyVector:
    def __init__(self, reader, expected_dtype):
        self.reader = reader
        stream = io.BytesIO(reader.read(0, min(PAGE_SIZE, reader.size)))
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported document index NPY version: {version}")
        if len(shape) != 1 or fortran or dtype != expected_dtype:
            raise ValueError("document index has an unexpected dtype or shape")
        self.length, self.dtype, self.offset = shape[0], dtype, stream.tell()
        if self.offset + self.length * dtype.itemsize != reader.size:
            raise ValueError("document index NPY size differs from its header")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return np.frombuffer(
            self.reader.read(self.offset + index * self.dtype.itemsize, self.dtype.itemsize),
            dtype=self.dtype,
        )[0]

    def search(self, value, *, right=False):
        lo, hi = 0, len(self)
        while lo < hi:
            mid = (lo + hi) // 2
            token_length = int(self[mid])
            if token_length < value or (right and token_length == value):
                lo = mid + 1
            else:
                hi = mid
        return lo


class DocumentIndex:
    def __init__(self, category, cache_dir, object_factory=PinnedObject):
        self.category = category
        files = {key: object_factory(value, cache_dir) for key, value in category["files"].items()}
        self.tokens = NpyVector(files["tokens"], np.dtype("<i4"))
        self.pointers = NpyVector(files["frag_ptr"], PTR_DTYPE)
        self.fragments = files["fragments"]
        if len(self.tokens) != len(self.pointers) or self.fragments.size % FRAG_DTYPE.itemsize:
            raise ValueError("document index tables are not aligned")
        names = files["shard_names"]
        self.shard_names = json.loads(names.read(0, names.size))
        if not isinstance(self.shard_names, list) or not all(
            isinstance(x, str) for x in self.shard_names
        ):
            raise ValueError("invalid shard_names table")

    def locate(self, row):
        pointer = self.pointers[row]
        offset, count = int(pointer["frag_offset"]), int(pointer["n_fragments"])
        if count < 1:
            return None
        fragments = np.frombuffer(
            self.fragments.read(offset * FRAG_DTYPE.itemsize, count * FRAG_DTYPE.itemsize),
            dtype=FRAG_DTYPE,
        )
        if int(fragments["frag_len"].sum()) != int(self.tokens[row]):
            return None  # source tokenization discarded the tail of this document
        if (fragments["shard_id"] < 0).any() or (
            fragments["shard_id"] >= len(self.shard_names)
        ).any():
            raise ValueError("document fragment has an invalid shard ID")
        if (fragments["frag_len"] == 0).any() or (
            fragments["seq_offset"].astype("int64") + fragments["frag_len"] > 2048
        ).any():
            raise ValueError("document fragment extends beyond its source row")
        return [
            dict(zip(FRAG_DTYPE.names, map(int, fragment), strict=True)) for fragment in fragments
        ]


def sample_category(index, *, token_budget, seed, max_tokens):
    """Balance document COUNTS by bucket prevalence, stopping near a token quota.

    Every eligible nonempty bucket gets one document. Remaining draws go to the
    most underrepresented bucket; the final whole document is included only if
    it brings the total closer to the quota. No document is truncated.
    """
    rng = np.random.default_rng(seed)
    buckets = []
    for name, minimum, maximum in LENGTH_BUCKETS:
        lo = index.tokens.search(minimum)
        natural_hi = (
            len(index.tokens) if maximum is None else index.tokens.search(maximum, right=True)
        )
        hi = min(natural_hi, index.tokens.search(max_tokens, right=True))
        hi = max(lo, hi)
        buckets.append(
            {
                "bucket": name,
                "lo": lo,
                "hi": hi,
                "population": hi - lo,
                "excluded_over_context_limit": natural_hi - hi,
                "sampled_documents": 0,
                "sampled_tokens": 0,
                "incomplete_skipped": 0,
            }
        )
    active = [i for i, b in enumerate(buckets) if b["population"]]
    if not active:
        raise ValueError("category contains no eligible documents longer than 2048")
    population = sum(b["population"] for b in buckets)
    used = [set() for _ in buckets]
    selected, total = [], 0

    def draw(which):
        bucket = buckets[which]
        while len(used[which]) < bucket["population"]:
            row = int(rng.integers(bucket["lo"], bucket["hi"]))
            if row in used[which]:
                continue
            used[which].add(row)
            fragments = index.locate(row)
            if fragments is None:
                bucket["incomplete_skipped"] += 1
                continue
            return {
                "document_row": row,
                "length": int(index.tokens[row]),
                "length_bucket": bucket["bucket"],
                "fragments": fragments,
            }
        raise ValueError(f"not enough complete documents in bucket {bucket['bucket']}")

    def take(which, document):
        nonlocal total
        selected.append(document)
        total += document["length"]
        buckets[which]["sampled_documents"] += 1
        buckets[which]["sampled_tokens"] += document["length"]

    for which in active:
        take(which, draw(which))
    while total < token_budget:
        which = max(
            active,
            key=lambda i: (
                buckets[i]["population"] / population * (len(selected) + 1)
                - buckets[i]["sampled_documents"]
            ),
        )
        document = draw(which)
        if abs(total + document["length"] - token_budget) >= token_budget - total:
            break
        take(which, document)
    return selected, {
        "target_tokens": token_budget,
        "sampled_tokens": total,
        "n_documents": len(selected),
        "max_document_tokens": min(max_tokens, MAX_DOCUMENT_TOKENS),
        "excluded_over_length_limit": len(index.tokens) - index.tokens.search(min(max_tokens, MAX_DOCUMENT_TOKENS), right=True),
        "buckets": buckets,
    }
