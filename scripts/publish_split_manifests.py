#!/usr/bin/env python3
"""Build the three split inventories from source manifests; optionally publish to R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chain_config
from teutonic.evaluation.categories import DEFAULT_RULES_PATH, category_of, load_rules
from teutonic.evaluation.configuration import (
    _shard_url,
    canonical_manifest_bytes,
    fetch_dataset_manifest,
    validate_dataset_manifest,
)


def build_manifests(sources, definitions):
    rules = load_rules(DEFAULT_RULES_PATH)
    outputs = {
        name: {
            "format": "teutonic-split-v1",
            "split": name,
            "category_weights": {k: c["weight"] for k, c in spec["categories"].items()},
            "category_sources": {k: c["dataset"] for k, c in spec["categories"].items()},
            "source_manifests": [],
            "shards": [],
        }
        for name, spec in definitions.items()
    }
    assignments = {}
    for name, spec in definitions.items():
        for category, config in spec["categories"].items():
            identity = (config["dataset"], category)
            if identity in assignments:
                raise ValueError(f"category assigned twice: {identity}")
            assignments[identity] = name
    for source in sources:
        used = set()
        for shard in source.manifest["shards"]:
            ref = next(
                str(shard[k]) for k in ("url", "href", "uri", "key", "path", "name") if shard.get(k)
            )
            category = category_of(rules.get(source.name), str(shard.get("source_file") or ref))
            split = assignments.get((source.name, category))
            if split is None:
                raise ValueError(f"unassigned source category: {source.name}/{category}")
            outputs[split]["shards"].append(
                {
                    "url": _shard_url(source.manifest_url, ref),
                    "category": category,
                    "dataset": source.name,
                    "sha256": shard["sha256"],
                    "n_tokens": shard["n_tokens"],
                    "size_bytes": shard["size_bytes"],
                }
            )
            used.add(split)
        for split in used:
            outputs[split]["source_manifests"].append(
                {"name": source.name, "url": source.manifest_url, "sha256": source.manifest_sha256}
            )
            for field in ("tokenizer", "dtype", "seq_len", "tokenization_mode"):
                value = source.manifest.get(field)
                if value is not None:
                    previous = outputs[split].get(field)
                    if previous is not None and previous != value:
                        raise ValueError(f"incompatible {field} across source manifests")
                    outputs[split][field] = value
    for output in outputs.values():
        output["total_tokens"] = sum(s["n_tokens"] for s in output["shards"])
        output["total_shards"] = len(output["shards"])
        validate_dataset_manifest(output)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--bucket", default=os.environ.get("TEUTONIC_DATASETS_BUCKET"))
    args = parser.parse_args(argv)
    if args.publish and not args.bucket:
        parser.error("--publish requires --bucket or TEUTONIC_DATASETS_BUCKET")
    sources = (fetch_dataset_manifest(**item) for item in chain_config.EVALUATION_DATASETS)
    outputs = build_manifests(sources, chain_config.SPLIT_DEFAULTS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = None
    if args.publish:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ.get("R2_SESSION_TOKEN"),
            region_name="auto",
        )
    for name, manifest in outputs.items():
        body = canonical_manifest_bytes(manifest)
        digest = hashlib.sha256(body).hexdigest()
        (args.output_dir / f"{name}.json").write_bytes(body)
        if client:
            # Keep immutable versions for audit, plus the conventional discovery URL.
            for key in (f"splits/{name}/versions/{digest}.json", f"splits/{name}/manifest.json"):
                client.put_object(
                    Bucket=args.bucket,
                    Key=key,
                    Body=body,
                    ContentType="application/json",
                    Metadata={"sha256": digest},
                )
                observed = client.get_object(Bucket=args.bucket, Key=key)["Body"].read()
                if hashlib.sha256(observed).hexdigest() != digest:
                    raise RuntimeError(f"published manifest verification failed: {key}")
        print(
            json.dumps(
                {
                    "split": name,
                    "sha256": digest,
                    "shards": len(manifest["shards"]),
                    "categories": len(manifest["category_weights"]),
                    "published": bool(client),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
