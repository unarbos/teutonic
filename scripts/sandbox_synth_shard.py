#!/usr/bin/env python3
"""Generate a synthetic uint32 token shard valid for an arbitrary vocab and
upload it to the dataset S3 bucket. Used by the LXXX sandbox smoke when the
real v2 (Gemma3) shards have token IDs outside the Qwen3 vocab range
(causes a CUDA device-side assert in the embedding lookup).

The shard is uniform random in [0, vocab_size). Loss numbers from this
shard are meaningless — the test value is purely "did sharded loading +
paired forward + bootstrap math produce a finite verdict". Both models
see identical tokens, so mu_hat reflects only the perturbation noise.

Usage:
    source .venv/bin/activate
    source /root/.creds/hf_token.env
    python scripts/sandbox_synth_shard.py \
        --tokens 40_000_000 \
        --vocab-size 151936 \
        --key dataset/lxxx-smoke/shard_smoke.npy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s3_transfer import safe_upload_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=40_000_000,
                    help="number of uint32 tokens to generate (~160 MB at default)")
    ap.add_argument("--vocab-size", type=int, default=151936)
    ap.add_argument("--key", default="dataset/lxxx-smoke/shard_smoke.npy",
                    help="S3 object key under TEUTONIC_DS_BUCKET")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-out", default="/tmp/lxxx-smoke-shard.npy")
    ap.add_argument("--no-upload", action="store_true",
                    help="skip S3 upload (just write local file)")
    args = ap.parse_args()

    print(f"generating {args.tokens:,} tokens uniform in [0, {args.vocab_size})", flush=True)
    rng = np.random.default_rng(args.seed)
    arr = rng.integers(0, args.vocab_size, size=args.tokens, dtype="<u4")
    print(f"max={arr.max()} min={arr.min()} bytes={arr.nbytes:,}", flush=True)

    out = Path(args.local_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr, allow_pickle=False)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)", flush=True)

    if args.no_upload:
        print("--no-upload set, skipping S3 push", flush=True)
        return 0

    import boto3
    endpoint = os.environ["TEUTONIC_DS_ENDPOINT"]
    bucket = os.environ["TEUTONIC_DS_BUCKET"]
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["TEUTONIC_DS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["TEUTONIC_DS_SECRET_KEY"],
        region_name="decentralized",
    )
    print(f"uploading to s3://{bucket}/{args.key} ...", flush=True)
    t0 = time.time()
    safe_upload_file(s3, str(out), bucket, args.key)
    print(f"upload done in {time.time()-t0:.1f}s -> s3://{bucket}/{args.key}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
