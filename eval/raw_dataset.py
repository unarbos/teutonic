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

log = logging.getLogger("eval_raw_dataset")

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
    token_remainder: list[int] = []
    used_files: list[str] = []
    docs_seen = 0

    for item in ordered[: cfg.max_files_per_eval]:
        key = item["key"]
        local_path = _download_parquet(r2, cfg, key)
        used_files.append(key)
        for text in _iter_parquet_texts(local_path, cfg.text_column):
            docs_seen += 1
            ids = tokenizer.encode(text, add_special_tokens=False)
            if eos_id is not None:
                ids.append(int(eos_id))
            token_remainder.extend(ids)
            while len(token_remainder) >= seq_len:
                sequences.append(token_remainder[:seq_len])
                token_remainder = token_remainder[seq_len:]
                if len(sequences) >= eval_n:
                    return sequences, _meta(cfg, files, used_files, docs_seen)

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
        r2.ds_client.download_file(r2.ds_bucket, key, str(tmp))
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
