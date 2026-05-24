"""Raw Hippius Parquet dataset loader for eval-time tokenization.

This is an opt-in bridge for mirrors such as:
  hf-mirrors/HuggingFaceFW/fineweb-edu/data/

The existing production path consumes pretokenized uint32 .npy shards. Raw mode
downloads one or more Parquet files from the Hippius mirror, tokenizes text with
a standard HF tokenizer, packs fixed-length token windows, and hands those token
IDs to the same paired-loss/bootstrap machinery.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from s3_transfer import safe_download_file

log = logging.getLogger("eval_raw_dataset")

PRIVATE_POOL_DEFAULT_DIR = "/var/teutonic/private_pool"

RAW_MODE_VALUES = {"raw", "raw_hippius", "fineweb", "fineweb_edu"}
DEFAULT_PREFIX = "hf-mirrors/HuggingFaceFW/fineweb-edu/data/"
DEFAULT_MANIFEST = f"{DEFAULT_PREFIX}_manifest.json"


def raw_dataset_enabled() -> bool:
    return os.environ.get("TEUTONIC_EVAL_DATASET_MODE", "").lower() in RAW_MODE_VALUES


@dataclass(frozen=True)
class RawDatasetConfig:
    manifest_key: str
    prefix: str
    explicit_keys: tuple[str, ...]
    tokenizer_repo: str
    text_column: str
    cache_dir: pathlib.Path
    max_files_per_eval: int
    list_fallback: bool

    @classmethod
    def from_env(cls, default_tokenizer_repo: str) -> "RawDatasetConfig":
        prefix = os.environ.get("TEUTONIC_RAW_DATASET_PREFIX", DEFAULT_PREFIX).strip("/")
        manifest_key = os.environ.get(
            "TEUTONIC_RAW_DATASET_MANIFEST",
            f"{prefix}/_manifest.json",
        ).strip("/")
        tokenizer_repo = os.environ.get(
            "TEUTONIC_RAW_TOKENIZER_REPO",
            default_tokenizer_repo,
        )
        if not tokenizer_repo:
            raise RuntimeError(
                "raw dataset mode needs TEUTONIC_RAW_TOKENIZER_REPO or chain seed tokenizer"
            )
        return cls(
            manifest_key=manifest_key,
            prefix=prefix,
            explicit_keys=tuple(
                key.strip() for key in os.environ.get("TEUTONIC_RAW_DATASET_KEYS", "").split(",")
                if key.strip()
            ),
            tokenizer_repo=tokenizer_repo,
            text_column=os.environ.get("TEUTONIC_RAW_TEXT_COLUMN", "text"),
            cache_dir=pathlib.Path(
                os.environ.get("TEUTONIC_RAW_CACHE", "/tmp/teutonic_raw_dataset")
            ),
            max_files_per_eval=int(os.environ.get("TEUTONIC_RAW_MAX_FILES_PER_EVAL", "8")),
            list_fallback=os.environ.get("TEUTONIC_RAW_LIST_FALLBACK", "1") == "1",
        )


def load_raw_sequences(
    r2,
    eval_n: int,
    seq_len: int,
    seed_str: str,
    default_tokenizer_repo: str,
) -> tuple[list[list[int]], dict]:
    """Return fixed-length token sequences sampled from raw mirrored Parquet."""
    cfg = RawDatasetConfig.from_env(default_tokenizer_repo)
    files = _load_file_list(r2, cfg)
    if not files:
        raise RuntimeError(
            f"raw dataset has no parquet files at manifest={cfg.manifest_key!r} "
            f"prefix={cfg.prefix!r}"
        )

    seed = int.from_bytes(hashlib.blake2b(seed_str.encode(), digest_size=8).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    start_idx = int(rng.integers(0, len(files)))
    ordered = files[start_idx:] + files[:start_idx]

    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_repo, token=token, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.sep_token_id

    sequences: list[list[int]] = []
    used_files: list[str] = []
    docs_seen = 0
    token_offset = 0
    token_pool = np.empty(0, dtype=np.uint32)

    for item in ordered[: cfg.max_files_per_eval]:
        key = item["key"]
        tokens = _get_tokenized_npy(r2, cfg, key, tokenizer, eos_id)
        used_files.append(key)
        docs_seen += len(tokens) // 500  # rough estimate
        token_pool = np.concatenate([token_pool[token_offset:], tokens])
        token_offset = 0
        n_windows = len(token_pool) // seq_len
        if n_windows > 0:
            usable = n_windows * seq_len
            windows = token_pool[:usable].reshape(n_windows, seq_len)
            for row in windows:
                sequences.append(row.tolist())
                if len(sequences) >= eval_n:
                    return sequences, _meta(cfg, files, used_files, docs_seen)
            token_offset = usable

    if not sequences:
        raise RuntimeError(
            f"raw dataset tokenization produced no {seq_len}-token windows "
            f"from {len(used_files)} files"
        )
    log.warning(
        "raw dataset produced only %d/%d requested sequences from %d files",
        len(sequences), eval_n, len(used_files),
    )
    return sequences, _meta(cfg, files, used_files, docs_seen)


def _meta(
    cfg: RawDatasetConfig,
    files: list[dict],
    used_files: list[str],
    docs_seen: int,
) -> dict:
    return {
        "mode": "raw_hippius",
        "manifest": cfg.manifest_key,
        "prefix": cfg.prefix,
        "tokenizer": cfg.tokenizer_repo,
        "total_files": len(files),
        "used_files": used_files,
        "docs_seen": docs_seen,
    }


def _load_file_list(r2, cfg: RawDatasetConfig) -> list[dict]:
    if cfg.explicit_keys:
        return [{"key": key, "size_bytes": 0} for key in cfg.explicit_keys]

    manifest = r2.ds_get(cfg.manifest_key)
    if manifest:
        files = manifest.get("files") or manifest.get("shards") or []
        out = []
        for item in files:
            key = item.get("dest_key") or item.get("key")
            if key and key.endswith(".parquet"):
                out.append({"key": key, "size_bytes": int(item.get("size_bytes", 0))})
        out.sort(key=lambda x: x["key"])
        return out

    if not cfg.list_fallback:
        return []

    log.warning("raw dataset manifest %s unavailable; listing prefix %s", cfg.manifest_key, cfg.prefix)
    paginator = r2.ds_client.get_paginator("list_objects_v2")
    max_listed = int(os.environ.get("TEUTONIC_RAW_MAX_LISTED_FILES", "10000"))
    files: list[dict] = []
    for page in paginator.paginate(Bucket=r2.ds_bucket, Prefix=cfg.prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith(".parquet"):
                files.append({"key": key, "size_bytes": int(obj.get("Size", 0))})
                if len(files) >= max_listed:
                    files.sort(key=lambda x: x["key"])
                    return files
    files.sort(key=lambda x: x["key"])
    return files


def _download_parquet(r2, cfg: RawDatasetConfig, key: str) -> pathlib.Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    path = cfg.cache_dir / f"{digest}.parquet"
    if path.exists() and path.stat().st_size > 0:
        return path

    fd, tmp_name = tempfile.mkstemp(prefix=f"{digest}.", suffix=".tmp", dir=cfg.cache_dir)
    os.close(fd)
    tmp = pathlib.Path(tmp_name)
    try:
        log.info("downloading raw parquet s3://%s/%s", r2.ds_bucket, key)
        safe_download_file(r2.ds_client, r2.ds_bucket, key, str(tmp))
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _iter_parquet_texts(path: pathlib.Path, text_column: str) -> Iterable[str]:
    import pyarrow.parquet as pq
    import pyarrow.types as patypes

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    column = text_column
    if column not in schema.names:
        string_columns = [
            field.name for field in schema
            if patypes.is_string(field.type) or patypes.is_large_string(field.type)
        ]
        if not string_columns:
            raise RuntimeError(f"{path} has no string column; columns={schema.names}")
        column = string_columns[0]
        log.warning("%s missing column %r; using %r", path.name, text_column, column)

    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=[column])
        for value in table.column(column).to_pylist():
            if isinstance(value, str) and value:
                yield value



def _get_tokenized_npy(
    r2, cfg: RawDatasetConfig, key: str, tokenizer, eos_id: int | None,
) -> np.ndarray:
    """Return a flat uint32 token array for a parquet file, cached on disk.

    Cache key: sha256(parquet_key + tokenizer_repo). The parquet file is
    downloaded (or served from cache) and every text document is tokenized
    with EOS separators. Result is saved as a .npy for instant reload on
    subsequent evals. Eliminates ~90% of the dataset preparation time.
    """
    cache_key = hashlib.sha256(
        f"{key}|{cfg.tokenizer_repo}".encode()
    ).hexdigest()[:24]
    npy_path = cfg.cache_dir / f"{cache_key}.tokens.npy"
    if npy_path.exists() and npy_path.stat().st_size > 0:
        log.info("tokenized cache HIT %s (%s)", key.rsplit("/", 1)[-1], npy_path.name)
        return np.load(npy_path)

    log.info("tokenizing %s (cache miss)", key.rsplit("/", 1)[-1])
    local_path = _download_parquet(r2, cfg, key)
    all_ids: list[int] = []
    for text in _iter_parquet_texts(local_path, cfg.text_column):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            ids.append(int(eos_id))
        all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint32)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_npy = npy_path.with_suffix(".tmp.npy")
    np.save(tmp_npy, arr)
    tmp_npy.replace(npy_path)
    log.info("tokenized cache WRITE %s: %d tokens (%.1f MB)",
             npy_path.name, len(arr), arr.nbytes / 1e6)
    return arr


_FILE_HASH_CACHE: dict[str, tuple[int, float, str]] = {}


def _hash_file(path: pathlib.Path) -> str:
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        stat = None
    if stat is not None:
        cached = _FILE_HASH_CACHE.get(key)
        if cached is not None and cached[0] == stat.st_size and cached[1] == stat.st_mtime:
            return cached[2]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if stat is not None:
        _FILE_HASH_CACHE[key] = (stat.st_size, stat.st_mtime, digest)
    return digest


def _pool_digest(file_digests: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(file_digests)).encode()).hexdigest()


def sample_private_pool(
    seq_len: int,
    n_sequences: int,
    tokenizer_repo: str,
    rng_seed: bytes | None = None,
) -> tuple[torch.Tensor, str]:
    """Sample n_sequences x seq_len from the validator's private pool directory.

    Pool location: env TEUTONIC_PRIVATE_POOL_DIR (default /var/teutonic/private_pool).
    Pool format: directory of *.parquet files; each parquet has a `text` column.
    Pool digest is sha256 over sorted({sha256(file) for file in pool_dir}).

    Sampling: enumerate parquet files, weight uniformly by row count, sample
    without replacement at the row level, tokenize, pack into fixed-length
    windows. Tokenization matches the public-corpus path.

    rng_seed is derived deterministically by the eval server from
    `blake2b(block_hash || hotkey || b"private")` so the validator can replay
    its own verdicts. Defaults to `os.urandom(8)` only if the caller omits it
    (smoke-test / CLI path).
    """
    pool_dir = pathlib.Path(os.environ.get("TEUTONIC_PRIVATE_POOL_DIR", PRIVATE_POOL_DEFAULT_DIR))
    files = sorted(pool_dir.glob("*.parquet")) if pool_dir.exists() else []
    if not files:
        raise RuntimeError("private pool empty; configure TEUTONIC_PRIVATE_POOL_DIR")

    file_digests = [_hash_file(p) for p in files]
    pool_digest = _pool_digest(file_digests)

    if n_sequences == 0:
        return torch.empty((0, seq_len), dtype=torch.long), pool_digest

    import pyarrow.parquet as pq

    row_counts = [pq.ParquetFile(p).metadata.num_rows for p in files]
    total_rows = sum(row_counts)
    if total_rows == 0:
        raise RuntimeError(f"private pool has 0 rows across {len(files)} files")

    seed = rng_seed if rng_seed is not None else os.urandom(8)
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed, "little")))

    # Sample more rows than strictly required — packed tokens may underfill
    # `n_sequences * seq_len` if rows are short. 4x is comfortably safe for
    # CommonCrawl/GitHub/ArXiv-grade text at seq_len=2048.
    target_tokens = n_sequences * seq_len
    n_rows_target = min(total_rows, max(n_sequences * 4, n_sequences + 8))
    row_choice = rng.choice(total_rows, size=n_rows_target, replace=False)
    row_choice.sort()

    cumulative = np.cumsum([0] + row_counts)
    rows_per_file: dict[int, set[int]] = {}
    for r in row_choice:
        f_idx = int(np.searchsorted(cumulative, r, side="right") - 1)
        rows_per_file.setdefault(f_idx, set()).add(int(r - cumulative[f_idx]))

    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, token=token, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.sep_token_id

    column = os.environ.get("TEUTONIC_RAW_TEXT_COLUMN", "text")
    sequences: list[list[int]] = []
    remainder: list[int] = []

    for f_idx, want_rows in rows_per_file.items():
        pf = pq.ParquetFile(files[f_idx])
        cur_row = 0
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=[column])
            n_rg = table.num_rows
            for i, value in enumerate(table.column(column).to_pylist()):
                if cur_row + i in want_rows and isinstance(value, str) and value:
                    ids = tokenizer.encode(value, add_special_tokens=False)
                    if eos_id is not None:
                        ids.append(int(eos_id))
                    remainder.extend(ids)
                    while len(remainder) >= seq_len:
                        sequences.append(remainder[:seq_len])
                        remainder = remainder[seq_len:]
                        if len(sequences) >= n_sequences:
                            break
                if len(sequences) >= n_sequences:
                    break
            cur_row += n_rg
            if len(sequences) >= n_sequences:
                break
        if len(sequences) >= n_sequences:
            break

    if len(sequences) < n_sequences:
        raise RuntimeError(
            f"private pool yielded only {len(sequences)}/{n_sequences} "
            f"sequences from {len(files)} files (~{target_tokens} tokens requested)"
        )

    if not sequences:
        return torch.empty((0, seq_len), dtype=torch.long), pool_digest
    return torch.tensor(sequences, dtype=torch.long), pool_digest
