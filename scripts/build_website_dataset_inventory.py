#!/usr/bin/env python3
"""Build a small public inventory from a hash-verified packed dataset manifest.

Reads R2 only; writes the requested local JSON file. No publication or DB edits.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.evaluation.packed_text import FORMAT, DatasetReader, read_client


def inventory(manifest, pin):
    if manifest.get("format") != FORMAT:
        raise ValueError("expected a packed-text dataset manifest")
    shards, documents, sizes = Counter(), Counter(), Counter()
    for obj in manifest["objects"]:
        if obj["role"] == "text":
            key = obj["dataset"], obj["category"]
            shards[key] += 1
            documents[key] += obj["document_count"]
            sizes[key] += obj["size_bytes"]
    categories = manifest["categories"]
    keys = [(c["dataset"], c["category"]) for c in categories]
    if len(set(keys)) != len(keys) or set(keys) != set(shards):
        raise ValueError("category and text-shard inventories differ")
    sources = []
    for c in sorted(categories, key=lambda c: (c["dataset"], c["category"])):
        key = c["dataset"], c["category"]
        if documents[key] != c["documents"] or sizes[key] != c["utf8_bytes"]:
            raise ValueError("category totals differ from text-shard inventory")
        sources.append({
            "name": c["category"], "dataset": c["dataset"],
            "total_tokens": c["original_mimo_tokens"],
            "total_documents": c["documents"], "total_shards": shards[key],
            "total_utf8_bytes": c["utf8_bytes"],
            "tokenization_mode": "packed-text", "dtype": "utf-8",
            "manifest_url": f"s3://{pin['bucket']}/{pin['key']}",
            "manifest_sha256": pin["sha256"],
        })
    if (sum(documents.values()) != manifest["documents"]
            or sum(sizes.values()) != manifest["utf8_bytes"]):
        raise ValueError("dataset totals differ from text-shard inventory")
    return {"schema_version": 1, "format": "teutonic-packed-text-inventory-v1",
            "dataset": pin, "dataset_label": "Full packed-text dataset",
            "token_count_basis": "original_mimo_token_count",
            "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(args.publication.read_text())["dataset"]
    result = inventory(DatasetReader(read_client()).json(pin), pin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(result['sources'])} categories; wrote {args.output}")


if __name__ == "__main__":
    main()
