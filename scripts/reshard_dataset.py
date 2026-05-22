#!/usr/bin/env python3
"""Reshard the legacy 2D anneal .npy files into smaller 1D flat shards with a manifest.

Streams each legacy file from R2 via range requests, flattens to 1D uint32,
writes ~2 GB chunks as new shards, computes SHA256 per shard, and builds
a manifest.json. Uploads everything to dataset/v1/ on the destination bucket.

Supports resuming from a previous run by reading existing shards from R2.

Usage:
    python scripts/reshard_dataset.py [--dry-run] [--shard-size-gb 2] [--resume]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config as BotoConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s3_transfer import safe_upload_file

DTYPE = np.dtype("<u4")
BYTES_PER_TOKEN = DTYPE.itemsize  # 4

R2_CFG = {
    "endpoint": os.environ["TEUTONIC_R2_ENDPOINT"],
    "access_key": os.environ["TEUTONIC_R2_ACCESS_KEY"],
    "secret_key": os.environ["TEUTONIC_R2_SECRET_KEY"],
    "bucket": os.environ.get("TEUTONIC_R2_BUCKET", "constantinople"),
}

LEGACY_SHARDS = [
    "anneal/anneal_000000.npy",
    "anneal/anneal_000002.npy",
    "anneal/anneal_000004.npy",
    "anneal/anneal_000005.npy",
]

DEST_PREFIX = "dataset/v1"
DOWNLOAD_CHUNK = 128 * 1024 * 1024  # 128 MB per range request
TOKENS_PER_SHARD = 536_870_912  # tokens in a 2 GB shard (2^29)


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_CFG["endpoint"],
        aws_access_key_id=R2_CFG["access_key"],
        aws_secret_access_key=R2_CFG["secret_key"],
        region_name="auto",
        config=BotoConfig(
            connect_timeout=30,
            read_timeout=120,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def get_object_size(client, key: str) -> int:
    resp = client.head_object(Bucket=R2_CFG["bucket"], Key=key)
    return resp["ContentLength"]


def read_npy_header(client, key: str) -> tuple[tuple, np.dtype, int]:
    """Read a .npy file header via range request. Returns (shape, dtype, header_bytes)."""
    header_data = client.get_object(
        Bucket=R2_CFG["bucket"], Key=key, Range="bytes=0-1023"
    )["Body"].read()

    buf = io.BytesIO(header_data)
    magic = buf.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError(f"Not a valid .npy file: {key}")

    version = struct.unpack("BB", buf.read(2))
    if version[0] == 1:
        header_len = struct.unpack("<H", buf.read(2))[0]
    else:
        header_len = struct.unpack("<I", buf.read(4))[0]

    header_str = buf.read(header_len).decode("latin1").strip()
    header_offset = buf.tell()

    header_dict = eval(header_str)  # noqa: S307 — numpy header is trusted
    shape = tuple(header_dict["shape"])
    dtype = np.dtype(header_dict["descr"])

    return shape, dtype, header_offset


@dataclass
class ShardInfo:
    key: str
    n_tokens: int
    size_bytes: int
    sha256: str


@dataclass
class ReshardState:
    shard_size_bytes: int
    shard_idx: int = 0
    shards: list[ShardInfo] = field(default_factory=list)
    total_tokens: int = 0

    _buf: bytearray = field(default_factory=bytearray, repr=False)
    _hasher: hashlib._Hash = field(default_factory=lambda: hashlib.sha256(), repr=False)
    _tokens_in_buf: int = 0

    def flush_shard(self, client, dry_run: bool) -> ShardInfo | None:
        if not self._buf:
            return None

        key = f"{DEST_PREFIX}/shards/shard_{self.shard_idx:06d}.npy"
        n_tokens = self._tokens_in_buf

        arr = np.frombuffer(self._buf, dtype=DTYPE)
        tmp_path = f"/tmp/teutonic_reshard_{self.shard_idx:06d}.npy"
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
        info = ShardInfo(key=key, n_tokens=n_tokens, size_bytes=size_bytes, sha256=sha)

        if not dry_run:
            safe_upload_file(client, tmp_path, R2_CFG["bucket"], key)
            print(f"  Uploaded {key} ({size_bytes / 1e9:.2f} GB, {n_tokens:,} tokens)")
        else:
            print(f"  [DRY RUN] Would upload {key} ({size_bytes / 1e9:.2f} GB, {n_tokens:,} tokens)")

        Path(tmp_path).unlink(missing_ok=True)
        self.shards.append(info)
        self.total_tokens += n_tokens
        self.shard_idx += 1

        self._buf = bytearray()
        self._hasher = hashlib.sha256()
        self._tokens_in_buf = 0

        return info

    def add_data(self, data: bytes, client, dry_run: bool):
        """Add raw token bytes, flushing shards as needed."""
        offset = 0
        while offset < len(data):
            remaining_capacity = self.shard_size_bytes - len(self._buf)
            take = min(remaining_capacity, len(data) - offset)
            self._buf.extend(data[offset : offset + take])
            self._tokens_in_buf += take // BYTES_PER_TOKEN
            offset += take

            if len(self._buf) >= self.shard_size_bytes:
                self.flush_shard(client, dry_run)


def discover_existing_shards(client) -> list[ShardInfo]:
    """List existing v1 shards on R2 for resume support."""
    paginator = client.get_paginator("list_objects_v2")
    shards = []
    for page in paginator.paginate(Bucket=R2_CFG["bucket"], Prefix=f"{DEST_PREFIX}/shards/shard_"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size = obj["Size"]
            # Extract shard index from key
            try:
                idx = int(key.split("shard_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            n_tokens = (size - 128) // BYTES_PER_TOKEN  # approximate (header is ~128 bytes)
            shards.append(ShardInfo(key=key, n_tokens=n_tokens, size_bytes=size, sha256=""))
    shards.sort(key=lambda s: s.key)
    return shards


def reshard(shard_size_gb: float = 2.0, dry_run: bool = False, resume: bool = False):
    shard_size_bytes = int(shard_size_gb * 1024**3)
    shard_size_bytes = (shard_size_bytes // BYTES_PER_TOKEN) * BYTES_PER_TOKEN
    tokens_per_output_shard = shard_size_bytes // BYTES_PER_TOKEN

    client = make_client()

    # Calculate total source tokens per file
    source_info = []
    print("Reading source file headers...")
    for legacy_key in LEGACY_SHARDS:
        file_size = get_object_size(client, legacy_key)
        shape, dtype, header_offset = read_npy_header(client, legacy_key)
        total_elements = 1
        for s in shape:
            total_elements *= s
        data_bytes = file_size - header_offset
        source_info.append({
            "key": legacy_key, "shape": shape, "dtype": dtype,
            "header_offset": header_offset, "file_size": file_size,
            "data_bytes": data_bytes, "n_tokens": total_elements,
        })
        print(f"  {legacy_key}: {shape}, {total_elements:,} tokens, {data_bytes / 1e9:.1f} GB")

    # Resume logic
    existing_shards = []
    start_shard_idx = 0
    tokens_to_skip = 0

    if resume:
        existing_shards = discover_existing_shards(client)
        if existing_shards:
            start_shard_idx = len(existing_shards)
            tokens_to_skip = start_shard_idx * tokens_per_output_shard
            print(f"\nRESUME: Found {len(existing_shards)} existing shards")
            print(f"  Tokens already processed: {tokens_to_skip:,}")
            print(f"  Will start at shard_{start_shard_idx:06d}")
        else:
            print("\nRESUME: No existing shards found, starting fresh")

    state = ReshardState(shard_size_bytes=shard_size_bytes, shard_idx=start_shard_idx)
    # Carry forward existing shard info for manifest
    for es in existing_shards:
        state.shards.append(es)
        state.total_tokens += es.n_tokens

    print(f"\nResharding {len(LEGACY_SHARDS)} legacy files -> ~{shard_size_gb} GB shards")
    print(f"Destination: {R2_CFG['bucket']}/{DEST_PREFIX}/")
    if dry_run:
        print("DRY RUN — nothing will be uploaded\n")

    t0 = time.time()
    total_source_bytes = 0
    global_tokens_seen = 0

    for src in source_info:
        legacy_key = src["key"]
        header_offset = src["header_offset"]
        file_size = src["file_size"]
        data_bytes = src["data_bytes"]
        file_tokens = src["n_tokens"]

        # Skip fully-consumed files
        if tokens_to_skip >= file_tokens:
            tokens_to_skip -= file_tokens
            global_tokens_seen += file_tokens
            print(f"\nSkipping {legacy_key} (fully processed)")
            continue

        # Partially-consumed file: skip into it
        skip_data_bytes = 0
        if tokens_to_skip > 0:
            skip_data_bytes = tokens_to_skip * BYTES_PER_TOKEN
            print(f"\nResuming {legacy_key} at byte {skip_data_bytes:,} ({tokens_to_skip:,} tokens skipped)")
            global_tokens_seen += tokens_to_skip
            tokens_to_skip = 0
        else:
            print(f"\nProcessing {legacy_key}...")

        print(f"  Shape: {src['shape']}, dtype: {src['dtype']}, data: {data_bytes / 1e9:.1f} GB")

        offset = header_offset + skip_data_bytes
        effective_data = data_bytes - skip_data_bytes
        downloaded = 0

        while offset < file_size:
            end = min(offset + DOWNLOAD_CHUNK - 1, file_size - 1)
            resp = client.get_object(
                Bucket=R2_CFG["bucket"], Key=legacy_key,
                Range=f"bytes={offset}-{end}",
            )
            chunk = resp["Body"].read()

            usable = (len(chunk) // BYTES_PER_TOKEN) * BYTES_PER_TOKEN
            if usable < len(chunk):
                end = offset + usable - 1

            state.add_data(chunk[:usable], client, dry_run)
            downloaded += usable
            offset = end + 1

            pct = downloaded / effective_data * 100
            print(f"  {downloaded / 1e9:.1f} / {effective_data / 1e9:.1f} GB ({pct:.0f}%) — {state.shard_idx} shards so far", end="\r")

        total_source_bytes += downloaded
        print(f"\n  Done with {legacy_key}: {downloaded / 1e9:.1f} GB processed")

    # Flush any remaining data in the buffer
    state.flush_shard(client, dry_run)

    elapsed = time.time() - t0

    # Build manifest with ALL shards (existing + new)
    # Re-read existing shard sizes for accurate manifest
    if resume and existing_shards:
        print("\nRe-reading existing shard metadata for manifest...")
        all_shards_info = []
        for es in existing_shards:
            all_shards_info.append({
                "key": es.key, "n_tokens": tokens_per_output_shard,
                "size_bytes": es.size_bytes, "sha256": es.sha256,
            })
        for s in state.shards[len(existing_shards):]:
            all_shards_info.append({
                "key": s.key, "n_tokens": s.n_tokens,
                "size_bytes": s.size_bytes, "sha256": s.sha256,
            })
    else:
        all_shards_info = [
            {"key": s.key, "n_tokens": s.n_tokens, "size_bytes": s.size_bytes, "sha256": s.sha256}
            for s in state.shards
        ]

    total_tokens = sum(s["n_tokens"] for s in all_shards_info)

    manifest = {
        "version": "v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": "google/gemma-3-1b",
        "dtype": "uint32",
        "total_tokens": total_tokens,
        "total_shards": len(all_shards_info),
        "shard_prefix": f"{DEST_PREFIX}/shards/",
        "shards": all_shards_info,
    }

    manifest_key = f"{DEST_PREFIX}/manifest.json"
    if not dry_run:
        body = json.dumps(manifest, indent=2).encode()
        client.put_object(Bucket=R2_CFG["bucket"], Key=manifest_key, Body=body, ContentType="application/json")
        print(f"\nManifest uploaded to {manifest_key}")
    else:
        print(f"\n[DRY RUN] Would upload manifest to {manifest_key}")

    print(f"\nResharding complete:")
    print(f"  Total shards: {len(all_shards_info)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  New data processed: {total_source_bytes / 1e9:.1f} GB")
    print(f"  Time (this run): {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")

    manifest_path = Path("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Local manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Reshard legacy dataset into v1 format")
    parser.add_argument("--shard-size-gb", type=float, default=2.0, help="Target shard size in GB")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without uploading")
    parser.add_argument("--resume", action="store_true", help="Resume from existing shards on R2")
    args = parser.parse_args()

    reshard(shard_size_gb=args.shard_size_gb, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
