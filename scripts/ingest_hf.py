#!/usr/bin/env python3
"""Stream a HuggingFace dataset, tokenize, pack into .npy shards, and upload
to Hippius using parallel workers. Maintains a live manifest so validators can
use shards as soon as they appear.

Each worker independently downloads a parquet file, tokenizes it, packs tokens
into seq_len windows, and uploads 2 GB shards directly to Hippius. A shared
atomic counter coordinates shard numbering across workers.

Supports any parquet-based HF dataset with a 'text' column. Tested with:
  - uonlp/CulturaX (6.3T tokens, 167 languages, English-first)
  - nvidia/Nemotron-CC-v2 (10.3TB, cleaned Common Crawl)

Usage:
    python scripts/ingest_hf.py --dataset uonlp/CulturaX [--langs en,de,fr] [--workers 16]
    python scripts/ingest_hf.py --dataset nvidia/Nemotron-CC-v2 [--workers 16]

Env vars:
    HF_TOKEN                    HuggingFace token (gated datasets)
    TEUTONIC_DS_ENDPOINT        Hippius S3 endpoint (default: https://s3.hippius.com)
    TEUTONIC_DS_BUCKET          Hippius bucket (default: teutonic-sn3)
    TEUTONIC_DS_ACCESS_KEY      Hippius access key
    TEUTONIC_DS_SECRET_KEY      Hippius secret key
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import signal
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pyarrow.parquet as pq
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

log = logging.getLogger("ingest_hf")

TOKENIZER_NAME = "unsloth/gemma-3-1b-it"
DEST_PREFIX = "dataset/v2"
DTYPE = np.dtype("<u4")
BYTES_PER_TOKEN = DTYPE.itemsize  # 4

DS_ENDPOINT = os.environ.get("TEUTONIC_DS_ENDPOINT", "https://s3.hippius.com")
DS_BUCKET = os.environ.get("TEUTONIC_DS_BUCKET", "teutonic-sn3")
DS_ACCESS_KEY = os.environ.get("TEUTONIC_DS_ACCESS_KEY", "")
DS_SECRET_KEY = os.environ.get("TEUTONIC_DS_SECRET_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.warning("received signal %s, will stop after current work", signum)
    _shutdown = True


# ---------------------------------------------------------------------------
# Hippius S3
# ---------------------------------------------------------------------------

def make_client():
    return boto3.client(
        "s3",
        endpoint_url=DS_ENDPOINT,
        aws_access_key_id=DS_ACCESS_KEY,
        aws_secret_access_key=DS_SECRET_KEY,
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            s3={"addressing_style": "path"},
        ),
    )


def get_manifest(client) -> dict | None:
    try:
        body = client.get_object(
            Bucket=DS_BUCKET, Key=f"{DEST_PREFIX}/manifest.json"
        )["Body"].read()
        return json.loads(body)
    except Exception:
        return None


def put_manifest(client, manifest: dict):
    body = json.dumps(manifest, indent=2).encode()
    client.put_object(
        Bucket=DS_BUCKET,
        Key=f"{DEST_PREFIX}/manifest.json",
        Body=body,
        ContentType="application/json",
    )


def flush_shard(buf: bytearray, shard_idx: int, client, dry_run: bool) -> dict:
    """Write token buffer as .npy shard, upload, return shard info dict."""
    key = f"{DEST_PREFIX}/shards/shard_{shard_idx:06d}.npy"
    n_tokens = len(buf) // BYTES_PER_TOKEN

    arr = np.frombuffer(buf, dtype=DTYPE)
    tmp_path = f"/tmp/ingest_shard_{shard_idx:06d}.npy"
    np.save(tmp_path, arr)

    file_hasher = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            file_hasher.update(chunk)

    size_bytes = Path(tmp_path).stat().st_size
    sha = file_hasher.hexdigest()

    if not dry_run:
        client.upload_file(tmp_path, DS_BUCKET, key)

    Path(tmp_path).unlink(missing_ok=True)

    return {
        "key": key,
        "n_tokens": n_tokens,
        "size_bytes": size_bytes,
        "sha256": sha,
    }


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_and_pack(tokenizer, text: str, remainder: list[int],
                      seq_len: int) -> tuple[bytes, list[int]]:
    """Tokenize text, prepend leftover tokens from previous sample, and pack
    into full seq_len windows. Returns (packed_bytes, new_remainder)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    all_tokens = remainder + ids

    n_full = len(all_tokens) // seq_len
    packed_count = n_full * seq_len

    if packed_count == 0:
        return b"", all_tokens

    arr = np.array(all_tokens[:packed_count], dtype=DTYPE)
    return arr.tobytes(), all_tokens[packed_count:]


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

CULTURAX_LANG_PRIORITY = [
    "en", "de", "fr", "es", "it", "pt", "nl", "ru", "zh", "ja", "ko",
    "pl", "cs", "sv", "da", "no", "fi", "ro", "hu", "el", "bg", "tr",
]

NEMOTRON_CONFIG_PRIORITY = [
    "High-Quality", "Medium-High-Quality", "Medium-Quality",
    "Diverse-QA", "High-Quality-Synthetic", "Translated-Diverse-QA",
]


def discover_parquet_files(dataset_repo: str, token: str,
                           langs: list[str] | None = None) -> list[tuple[str, str]]:
    """Return [(config_name, file_path), ...] ordered by priority."""
    api = HfApi(token=token)
    all_files = api.list_repo_files(dataset_repo, repo_type="dataset")

    config_files: dict[str, list[str]] = {}
    for f in all_files:
        if f.endswith(".parquet") and "/" in f:
            config = f.split("/")[0]
            config_files.setdefault(config, []).append(f)

    for config in config_files:
        config_files[config].sort()

    is_culturax = "CulturaX" in dataset_repo
    if langs:
        ordered_configs = [l for l in langs if l in config_files]
    elif is_culturax:
        ordered_configs = [l for l in CULTURAX_LANG_PRIORITY if l in config_files]
        for c in sorted(config_files):
            if c not in ordered_configs:
                ordered_configs.append(c)
    else:
        ordered_configs = [p for p in NEMOTRON_CONFIG_PRIORITY if p in config_files]
        for c in sorted(config_files):
            if c not in ordered_configs:
                ordered_configs.append(c)

    result = []
    for config in ordered_configs:
        files = config_files[config]
        log.info("config %s: %d parquet files", config, len(files))
        for f in files:
            result.append((config, f))

    log.info("total: %d parquet files across %d configs", len(result), len(ordered_configs))
    return result


def iter_parquet_texts(parquet_path: str):
    """Yield text strings from a local parquet file, row-group by row-group."""
    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["text"])
        texts = table.column("text").to_pylist()
        for text in texts:
            if text:
                yield text


# ---------------------------------------------------------------------------
# Shared shard counter (set via worker initializer, accessed as global)
# ---------------------------------------------------------------------------

_shard_counter: multiprocessing.Value | None = None


def next_shard_idx() -> int:
    """Atomically allocate the next shard index."""
    with _shard_counter.get_lock():
        idx = _shard_counter.value
        _shard_counter.value += 1
        return idx


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------

def worker_process_file(
    file_path: str,
    dataset_repo: str,
    token: str,
    seq_len: int,
    shard_size_bytes: int,
    dry_run: bool,
) -> tuple[str, list[dict], int, int]:
    """Download a parquet file, tokenize, pack, and upload shards directly.

    Returns (file_path, shard_info_list, n_samples, n_tokens).
    """
    worker_log = logging.getLogger(f"worker.{os.getpid()}")
    worker_log.info("starting %s", file_path)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=token)
    client = make_client()

    tmp_dir = tempfile.mkdtemp(prefix="ingest_hf_")
    try:
        try:
            local_path = hf_hub_download(
                dataset_repo, file_path, repo_type="dataset", token=token,
                local_dir=tmp_dir,
            )
        except Exception as e:
            worker_log.error("download failed %s: %s", file_path, e)
            raise

        buf = bytearray()
        remainder: list[int] = []
        n_samples = 0
        n_tokens = 0
        shard_infos: list[dict] = []

        for text in iter_parquet_texts(local_path):
            packed, remainder = tokenize_and_pack(tokenizer, text, remainder, seq_len)
            if not packed:
                continue

            buf.extend(packed)
            n_samples += 1
            n_tokens += len(packed) // BYTES_PER_TOKEN

            while len(buf) >= shard_size_bytes:
                shard_data = bytearray(buf[:shard_size_bytes])
                buf = bytearray(buf[shard_size_bytes:])
                idx = next_shard_idx()
                info = flush_shard(shard_data, idx, client, dry_run)
                shard_infos.append(info)
                worker_log.info(
                    "uploaded shard %d from %s (%s tokens)",
                    idx, file_path.split("/")[-1], f"{info['n_tokens']:,}",
                )

        if buf:
            idx = next_shard_idx()
            info = flush_shard(buf, idx, client, dry_run)
            shard_infos.append(info)
            worker_log.info(
                "uploaded partial shard %d from %s (%s tokens)",
                idx, file_path.split("/")[-1], f"{info['n_tokens']:,}",
            )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    worker_log.info(
        "finished %s: %d samples, %s tokens, %d shards",
        file_path, n_samples, f"{n_tokens:,}", len(shard_infos),
    )
    return file_path, shard_infos, n_samples, n_tokens


def _worker_init(counter: multiprocessing.Value):
    """Initializer for worker processes — set shared counter, configure logging,
    and ignore signals (main process handles shutdown)."""
    global _shard_counter
    _shard_counter = counter
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


# ---------------------------------------------------------------------------
# Main ingestion loop (parallel)
# ---------------------------------------------------------------------------

def ingest(dataset_repo: str, shard_size_gb: float = 2.0, seq_len: int = 2048,
           dry_run: bool = False, langs: list[str] | None = None,
           workers: int = 16):
    global _shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    shard_size_bytes = int(shard_size_gb * 1024**3)
    shard_size_bytes = (shard_size_bytes // BYTES_PER_TOKEN) * BYTES_PER_TOKEN

    for var in ["TEUTONIC_DS_ACCESS_KEY", "TEUTONIC_DS_SECRET_KEY"]:
        if not os.environ.get(var):
            log.error("missing env var: %s", var)
            sys.exit(1)

    token = HF_TOKEN
    if not token:
        log.error("missing HF_TOKEN for gated dataset")
        sys.exit(1)

    client = make_client()

    manifest = get_manifest(client)
    if manifest:
        start_shard_idx = manifest["total_shards"]
        log.info("resuming: %d existing shards, %s tokens already ingested",
                 start_shard_idx, f"{manifest['total_tokens']:,}")
    else:
        start_shard_idx = 0
        manifest = None
        log.info("starting fresh (no existing manifest)")

    shard_counter = multiprocessing.Value("i", start_shard_idx)

    log.info("dataset: %s", dataset_repo)
    log.info("discovering parquet files...")
    parquet_files = discover_parquet_files(dataset_repo, token, langs)

    state_path = Path(f"/tmp/ingest_state_{dataset_repo.replace('/', '_')}.json")
    completed_files: set[str] = set()
    if state_path.exists() and start_shard_idx > 0:
        try:
            state = json.loads(state_path.read_text())
            completed_files = set(state.get("completed_files", []))
            log.info("resume state: %d files already processed", len(completed_files))
        except Exception:
            pass

    pending_files = [
        (config, fp) for config, fp in parquet_files if fp not in completed_files
    ]
    log.info(
        "%d files to process (%d already done), using %d workers",
        len(pending_files), len(completed_files), workers,
    )

    total_samples = 0
    total_tokens_ingested = 0
    total_shards_uploaded = 0
    failed_files: list[str] = []
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(shard_counter,),
    ) as pool:
        futures = {}
        for _config, file_path in pending_files:
            if _shutdown:
                break
            fut = pool.submit(
                worker_process_file,
                file_path, dataset_repo, token, seq_len,
                shard_size_bytes, dry_run,
            )
            futures[fut] = file_path

        for fut in as_completed(futures):
            file_path = futures[fut]
            try:
                fp, shard_infos, n_samples, n_tokens = fut.result()

                for info in shard_infos:
                    if manifest is None:
                        manifest = {
                            "version": "v2",
                            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "tokenizer": TOKENIZER_NAME,
                            "dtype": "uint32",
                            "source": dataset_repo,
                            "total_tokens": 0,
                            "total_shards": 0,
                            "shard_prefix": f"{DEST_PREFIX}/shards/",
                            "shards": [],
                        }
                    manifest["shards"].append(info)
                    manifest["total_shards"] = len(manifest["shards"])
                    manifest["total_tokens"] = sum(
                        s["n_tokens"] for s in manifest["shards"]
                    )
                    manifest["updated"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )

                if shard_infos and not dry_run:
                    put_manifest(client, manifest)

                total_samples += n_samples
                total_tokens_ingested += n_tokens
                total_shards_uploaded += len(shard_infos)

                completed_files.add(fp)
                state_path.write_text(json.dumps({
                    "completed_files": sorted(completed_files),
                    "shard_idx": shard_counter.value,
                    "total_samples": total_samples,
                    "total_tokens": total_tokens_ingested,
                }))

                elapsed = time.time() - t0
                rate = total_tokens_ingested / elapsed if elapsed > 0 else 0
                log.info(
                    "completed %s: +%d shards +%s tokens | "
                    "total: %d shards %s tokens %.0f tok/s %d/%d files",
                    fp.split("/")[-1], len(shard_infos), f"{n_tokens:,}",
                    total_shards_uploaded, f"{total_tokens_ingested:,}",
                    rate, len(completed_files), len(parquet_files),
                )

            except Exception as e:
                log.error("FAILED %s: %s", file_path, e)
                failed_files.append(file_path)

            if _shutdown:
                log.warning("shutdown requested, cancelling remaining work")
                for remaining_fut in futures:
                    remaining_fut.cancel()
                break

    elapsed = time.time() - t0
    rate = total_tokens_ingested / elapsed if elapsed > 0 else 0
    log.info(
        "ingestion %s: %d shards, %s samples, %s tokens in %.0fs (%.0f tok/s)",
        "stopped (signal)" if _shutdown else "complete",
        total_shards_uploaded, f"{total_samples:,}",
        f"{total_tokens_ingested:,}", elapsed, rate,
    )
    if failed_files:
        log.warning("%d files failed (will be retried on next run): %s",
                    len(failed_files), failed_files[:20])


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Ingest HuggingFace dataset to Hippius shards")
    parser.add_argument("--dataset", default="uonlp/CulturaX",
                        help="HF dataset repo (default: uonlp/CulturaX)")
    parser.add_argument("--langs", default=None,
                        help="Comma-separated language codes for CulturaX (default: en first, then others)")
    parser.add_argument("--shard-size-gb", type=float, default=2.0)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel worker processes (default: 16)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    langs = args.langs.split(",") if args.langs else None

    ingest(
        dataset_repo=args.dataset,
        shard_size_gb=args.shard_size_gb,
        seq_len=args.seq_len,
        dry_run=args.dry_run,
        langs=langs,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
