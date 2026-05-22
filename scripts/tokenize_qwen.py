#!/usr/bin/env python3
"""Re-tokenize CulturaX with the Qwen3 tokenizer for the Teutonic-LXXX
dataset (v3).

Mirrors `scripts/reshard_dataset.py`'s on-disk format so the eval pipeline
can read v3 shards opaquely (1D uint32 npy, ~536 M tokens/shard, sha256-
manifested). Switches the source corpus and the tokenizer; everything
downstream stays the same.

Status: **NOT executed in the LXXX sandbox session** — wall is multi-day
on a single 64-core box, multi-TB on disk, and nontrivial Hippius write
credentials. This script is the operator handoff so a separate compute
job can run it during Phase 3 of the runbook.

Usage (Phase 3 of LXXX_RUNBOOK):
    python scripts/tokenize_qwen.py \
        --tokenizer Qwen/Qwen3-30B-A3B \
        --source uonlp/CulturaX \
        --shard-tokens 536870912 \
        --dest s3://teutonic-sn3/dataset/v3/ \
        --staging /mnt/local-ssd/v3-staging \
        --workers 60 \
        [--resume]

Resume support: if --resume and existing v3 shards live at the dest, the
script discovers them via S3 ListObjectsV2 and starts numbering at the
next index, mirroring `reshard_dataset.discover_existing_shards`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s3_transfer import safe_upload_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [tokenize_qwen] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tokenize_qwen")


@dataclass
class ShardInfo:
    key: str
    n_tokens: int
    size_bytes: int
    sha256: str


@dataclass
class State:
    shard_idx: int = 0
    buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<u4"))
    shards: list[ShardInfo] = field(default_factory=list)


def _tokenize_chunk(args):
    """Worker: tokenize a chunk of (text, language) tuples into uint32 ids.

    Each worker holds its own tokenizer (transformers tokenizers are not
    fork-safe in general; reload after fork). We append eos_token_id between
    documents — matches v2's contiguous-document layout.
    """
    chunk_id, texts, tokenizer_repo = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_repo, use_fast=True)
    eos = tok.eos_token_id
    out: list[int] = []
    for txt in texts:
        if not txt:
            continue
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        out.extend(ids)
        out.append(eos)
    arr = np.asarray(out, dtype="<u4")
    return chunk_id, arr


def _flush_shard(state: State, shard_size: int, dest: str,
                 staging: Path, dry_run: bool, s3_client) -> bool:
    """If state.buf has at least shard_size tokens, write the next shard
    out to staging then upload to S3 (or just stage if dry_run)."""
    if state.buf.size < shard_size:
        return False
    n_take = (state.buf.size // shard_size) * shard_size  # full shards only
    while n_take >= shard_size:
        chunk = state.buf[:shard_size]
        state.buf = state.buf[shard_size:]
        n_take -= shard_size

        key = f"dataset/v3/shards/shard_{state.shard_idx:06d}.npy"
        local = staging / Path(key).name
        np.save(local, chunk, allow_pickle=False)
        size_bytes = local.stat().st_size

        h = hashlib.sha256()
        with open(local, "rb") as f:
            while True:
                blk = f.read(1 << 20)
                if not blk:
                    break
                h.update(blk)
        sha = h.hexdigest()

        log.info("shard %06d ready: %d tokens, %.2f GB, sha=%s",
                 state.shard_idx, shard_size, size_bytes / 1e9, sha[:12])

        if not dry_run and dest.startswith("s3://"):
            bucket, _, prefix = dest[5:].partition("/")
            obj_key = f"{prefix.rstrip('/')}/shards/shard_{state.shard_idx:06d}.npy"
            safe_upload_file(s3_client, str(local), bucket, obj_key)
            log.info("uploaded -> s3://%s/%s", bucket, obj_key)

        state.shards.append(ShardInfo(
            key=key, n_tokens=shard_size, size_bytes=size_bytes, sha256=sha,
        ))
        state.shard_idx += 1
        local.unlink()  # reclaim disk after each shard
    return True


def _write_manifest(state: State, tokenizer: str, dest: str,
                    dry_run: bool, s3_client):
    manifest = {
        "version": "v3",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": tokenizer,
        "dtype": "uint32",
        "source": "uonlp/CulturaX",
        "total_tokens": sum(s.n_tokens for s in state.shards),
        "total_shards": len(state.shards),
        "shard_prefix": "dataset/v3/shards/",
        "shards": [s.__dict__ for s in state.shards],
    }
    log.info("manifest: %d shards, %.0f B tokens",
             manifest["total_shards"], manifest["total_tokens"] / 1e9)
    if dry_run:
        path = Path("/tmp/v3-manifest.json")
        path.write_text(json.dumps(manifest, indent=2))
        log.info("dry-run wrote manifest -> %s", path)
        return
    if dest.startswith("s3://"):
        bucket, _, prefix = dest[5:].partition("/")
        key = f"{prefix.rstrip('/')}/manifest.json"
        body = json.dumps(manifest).encode()
        s3_client.put_object(Bucket=bucket, Key=key, Body=body,
                             ContentType="application/json")
        log.info("uploaded manifest -> s3://%s/%s", bucket, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--source", default="uonlp/CulturaX",
                    help="HF dataset name (CulturaX is the v2 source; matching "
                         "for apples-to-apples loss curves)")
    ap.add_argument("--source-config", default=None,
                    help="HF dataset config (CulturaX has per-language configs; "
                         "default = stream all languages mixed)")
    ap.add_argument("--source-split", default="train")
    ap.add_argument("--shard-tokens", type=int, default=536_870_912,
                    help="tokens per shard (default 2 GiB at uint32 = 0.5 G tok)")
    ap.add_argument("--dest", required=True,
                    help="s3://bucket/prefix or local path")
    ap.add_argument("--staging", default="/tmp/v3-staging",
                    help="local scratch dir for one shard at a time")
    ap.add_argument("--workers", type=int, default=mp.cpu_count() - 4)
    ap.add_argument("--chunk-docs", type=int, default=512,
                    help="docs per worker task")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="cap total shards (0 = no cap, run until source exhausted)")
    ap.add_argument("--resume", action="store_true",
                    help="discover existing v3 shards in dest and continue")
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenize + stage but do not upload to S3")
    args = ap.parse_args()

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)

    s3 = None
    if args.dest.startswith("s3://") and not args.dry_run:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("TEUTONIC_DS_ENDPOINT", "https://s3.hippius.com"),
            aws_access_key_id=os.environ["TEUTONIC_DS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["TEUTONIC_DS_SECRET_KEY"],
        )

    state = State()

    if args.resume and s3 is not None:
        bucket, _, prefix = args.dest[5:].partition("/")
        log.info("resume: scanning s3://%s/%s/shards/ for existing shards",
                 bucket, prefix.rstrip("/"))
        paginator = s3.get_paginator("list_objects_v2")
        existing = []
        for page in paginator.paginate(Bucket=bucket,
                                        Prefix=f"{prefix.rstrip('/')}/shards/shard_"):
            for obj in page.get("Contents", []) or []:
                existing.append(obj)
        existing.sort(key=lambda o: o["Key"])
        if existing:
            state.shard_idx = len(existing)
            log.info("resume: found %d existing shards, starting at %06d",
                     len(existing), state.shard_idx)
            for o in existing:
                # We don't re-sha; trust the shards already on S3.
                state.shards.append(ShardInfo(
                    key=o["Key"], n_tokens=args.shard_tokens,
                    size_bytes=o["Size"], sha256="(unverified-resume)",
                ))

    log.info("loading source dataset %s (config=%s, split=%s)",
             args.source, args.source_config, args.source_split)
    from datasets import load_dataset
    ds = load_dataset(args.source, name=args.source_config,
                      split=args.source_split, streaming=True)

    pool = mp.Pool(args.workers)
    log.info("tokenizing with %s on %d workers, shard_size=%d tokens",
             args.tokenizer, args.workers, args.shard_tokens)

    pending: list = []
    chunk_buf: list[str] = []
    chunk_id = 0

    def _drain():
        nonlocal pending
        results = []
        if pending:
            for fut in pending:
                results.append(fut.get())
        pending = []
        results.sort(key=lambda x: x[0])
        for _cid, arr in results:
            state.buf = np.concatenate([state.buf, arr])
        _flush_shard(state, args.shard_tokens, args.dest, staging,
                     args.dry_run, s3)

    t0 = time.time()
    n_docs = 0
    for row in ds:
        text = row.get("text") or row.get("content") or ""
        chunk_buf.append(text)
        n_docs += 1
        if len(chunk_buf) >= args.chunk_docs:
            pending.append(pool.apply_async(
                _tokenize_chunk, ((chunk_id, chunk_buf, args.tokenizer),)))
            chunk_buf = []
            chunk_id += 1
            if len(pending) >= args.workers * 2:
                _drain()
                if args.max_shards and state.shard_idx >= args.max_shards:
                    log.info("hit --max-shards=%d, stopping", args.max_shards)
                    break
            if n_docs % (args.chunk_docs * args.workers) == 0:
                rate = n_docs / max(time.time() - t0, 1)
                log.info("ingested %d docs (%.0f docs/s), shards_done=%d",
                         n_docs, rate, state.shard_idx)

    if chunk_buf:
        pending.append(pool.apply_async(
            _tokenize_chunk, ((chunk_id, chunk_buf, args.tokenizer),)))

    _drain()
    pool.close()
    pool.join()

    if state.buf.size > 0:
        log.info("tail buffer has %d tokens (less than shard size %d) — DROPPED",
                 state.buf.size, args.shard_tokens)

    _write_manifest(state, args.tokenizer, args.dest, args.dry_run, s3)
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    sys.exit(main())
