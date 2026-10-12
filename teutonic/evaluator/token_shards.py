"""Read sampled uint32 NPY tokens into memory without writing shard caches."""

import io
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from teutonic.evaluator.document_index import RangeObject
from teutonic.evaluator.range_http import pooled_urlopen

# Merge nearby reads to avoid one HTTP request per row, with bounded overfetch.
MERGE_GAP = 16 * 1024
DENSE_MERGE_GAP = 256 * 1024
MAX_RANGE_BYTES = 8 * 1024 * 1024


class NpyShard:
    def __init__(self, shard, opener=pooled_urlopen):
        self.reader = RangeObject(
            {"url": shard.url, "size_bytes": shard.size_bytes, "sha256": shard.sha256},
            opener=opener,
        )
        # Ordinary numeric headers are small; bound even malformed header reads.
        prefix = self._read(0, min(1024, shard.size_bytes))
        stream = io.BytesIO(prefix)
        version = np.lib.format.read_magic(stream)
        if version not in {(1, 0), (2, 0), (3, 0)}:
            raise ValueError(f"unsupported token shard NPY version: {version}")
        length_bytes = 2 if version == (1, 0) else 4
        header_length = int.from_bytes(stream.read(length_bytes), "little")
        header_end = 8 + length_bytes + header_length
        if header_length > 10000 or header_end > shard.size_bytes:
            raise ValueError("invalid token shard NPY header length")
        if header_end > len(prefix):
            prefix += self._read(len(prefix), header_end - len(prefix))
        stream = io.BytesIO(prefix)
        np.lib.format.read_magic(stream)
        # v3 changes header text encoding only; numeric dtype/shape headers are ASCII.
        read_header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        self.shape, self.fortran, self.dtype = read_header(stream)
        self.offset = stream.tell()
        if self.dtype != np.dtype("<u4"):
            raise ValueError("token shard dtype must be uint32/<u4")
        if len(self.shape) not in (1, 2) or any(n <= 0 for n in self.shape):
            raise ValueError("token shard expected 1D token stream or 2D sequence matrix")
        self.n_tokens = math.prod(self.shape)
        if self.n_tokens != shard.n_tokens:
            raise ValueError("token shard token count differs from validator descriptor")
        if self.offset + self.n_tokens * self.dtype.itemsize != shard.size_bytes:
            raise ValueError("token shard NPY size differs from validator descriptor")

    def _read(self, start, size):
        if start < 0 or size <= 0 or start + size > self.reader.size:
            raise ValueError("token shard byte range is out of bounds")
        return self.reader._range(start, start + size - 1, f"{start}-{size}")

    def _read_spans(self, spans):
        """Fetch sorted/coalesced byte spans, then restore caller order."""
        # Dense window samples otherwise produce hundreds of tiny requests per
        # shard. Trade extra bytes for fewer round trips at >= 1/16 coverage.
        # Sparse document fragments retain the small overfetch allowance.
        merge_gap = (
            DENSE_MERGE_GAP
            if sum(size for _start, size in spans) >= self.n_tokens * 4 / 16
            else MERGE_GAP
        )
        groups = []
        for i, (start, size) in sorted(enumerate(spans), key=lambda item: item[1][0]):
            end = start + size
            if size <= 0 or size > MAX_RANGE_BYTES:
                raise ValueError("token shard read exceeds range limit")
            if (
                groups
                and start <= groups[-1][1] + merge_gap
                and max(end, groups[-1][1]) - groups[-1][0] <= MAX_RANGE_BYTES
            ):
                groups[-1][1] = max(groups[-1][1], end)
                groups[-1][2].append((i, start, size))
            else:
                groups.append([start, end, [(i, start, size)]])
        result = [None] * len(spans)

        def fetch(group):
            start, end, members = group
            body = self._read(start, end - start)
            return [
                (i, np.frombuffer(body, dtype=self.dtype, count=size // 4, offset=offset - start))
                for i, offset, size in members
            ]

        if len(groups) <= 1:
            batches = map(fetch, groups)
            for batch in batches:
                for i, array in batch:
                    result[i] = array
        else:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for batch in pool.map(fetch, groups):
                    for i, array in batch:
                        result[i] = array
        return result

    def read_tokens(self, spans):
        """Read (start, length) spans in logical row-major token order."""
        for start, length in spans:
            if start < 0 or length <= 0 or start + length > self.n_tokens:
                raise ValueError("token span is outside its shard")
        if len(self.shape) == 1 or not self.fortran:
            return self._read_spans(
                [(self.offset + start * 4, length * 4) for start, length in spans]
            )
        # Preserve the old loader's behavior for Fortran-ordered matrices too.
        rows, width = self.shape
        physical = [
            (self.offset + ((i % width) * rows + i // width) * 4, 4)
            for start, length in spans
            for i in range(start, start + length)
        ]
        values = self._read_spans(physical)
        result, cursor = [], 0
        for _start, length in spans:
            result.append(np.concatenate(values[cursor : cursor + length]))
            cursor += length
        return result

    def sequence_count(self, width):
        if width <= 0 or (len(self.shape) == 2 and self.shape[1] != width):
            raise ValueError(f"token shard sequence width differs from seq_len={width}")
        return self.n_tokens // width

    def read_sequences(self, indices, width):
        self.sequence_count(width)
        return self.read_tokens([(int(i) * width, width) for i in indices])
