#!/usr/bin/env python3
"""Validate and publish immutable packed manifests; use --dry-run for read-only checks."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.publish_category_artifacts import publish_immutable
from teutonic.evaluation.packed_text import (
    BUCKET,
    BUCKETS,
    CATALOG_FORMAT,
    PREFIX,
    DatasetReader,
    PackedDataset,
    read_client,
    validate_mix,
    validate_pin,
)


def validate_bundle(directory):
    root = Path(directory).resolve()
    entries = json.loads((root / "pending-publication.json").read_text())
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("bundle needs a sampling catalog and competition manifests")
    artifacts, catalogs, mixes = [], [], []
    keys, names = set(), set()
    for entry in entries:
        if set(entry) != {"local", "bucket", "key", "sha256"}:
            raise ValueError("invalid publication entry")
        pin = validate_pin({k: entry[k] for k in ("bucket", "key", "sha256")})
        name = entry["local"]
        path = root / name
        if Path(name).name != name or path.is_symlink() or pin["key"] in keys or name in names:
            raise ValueError("duplicate or unsafe publication path")
        keys.add(pin["key"])
        names.add(name)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("bundle hash mismatch")
        value = json.loads(raw)
        if value.get("format") == CATALOG_FORMAT:
            if pin["key"] != f"{PREFIX}sampling/{pin['sha256']}/manifest.json":
                raise ValueError("catalog must use its immutable hash path")
            validate_pin(value["dataset"])
            if value.get("buckets") != [list(b) for b in BUCKETS] or not value.get("partitions"):
                raise ValueError("invalid sampling catalog")
            catalogs.append((pin, value))
        else:
            validate_mix(value)
            if pin["key"] != f"{PREFIX}splits/{value['competition']}/{pin['sha256']}.json":
                raise ValueError("mix must use its immutable competition/hash path")
            mixes.append(value)
        artifacts.append((entry, raw))
    if len(catalogs) != 1 or not mixes or len({m["competition"] for m in mixes}) != len(mixes):
        raise ValueError("bundle needs exactly one catalog and unique competitions")
    catalog_pin, catalog = catalogs[0]
    for mix in mixes:
        if mix["sampling"] != catalog_pin or mix["dataset"] != catalog["dataset"]:
            raise ValueError("bundle dependency pins differ")
    return artifacts, catalog, mixes


def run(directory, client, *, publish=True):
    artifacts, catalog, mixes = validate_bundle(directory)
    dataset = PackedDataset(DatasetReader(client), catalog["dataset"])
    available, seen = set(), set()
    for row in catalog["partitions"]:
        obj = dataset.objects.get(row["path"], {})
        identity = (row["dataset"], row["category"], row["bucket"])
        key = (*identity, row["path"])
        if (key in seen or obj.get("role") != "index"
                or identity[:2] != (obj.get("dataset"), obj.get("category"))
                or row["bucket"] not in {b[0] for b in BUCKETS}
                or any(type(row[k]) is not int for k in ("lo", "hi"))
                or not 0 <= row["lo"] < row["hi"] <= obj["document_count"]):
            raise ValueError("catalog partition differs from pinned inventory")
        seen.add(key)
        available.add(identity)
    for mix in mixes:
        for category in mix["categories"]:
            if any((category["dataset"], category["category"], b[0]) not in available for b in BUCKETS):
                raise ValueError("missing category length bucket")
    # All local content/dependencies are checked before the first conditional PUT.
    if publish:
        for entry, raw in sorted(artifacts, key=lambda pair: "/splits/" in pair[0]["key"]):
            publish_immutable(client, BUCKET, entry["key"], raw, entry["sha256"])
    receipt = {"state": "published-and-verified" if publish else "validated-read-only",
               "dataset": dataset.pin, "objects": [entry for entry, _ in artifacts],
               "budgets": {mix["competition"]: mix["token_budget"] for mix in mixes}}
    (Path(directory) / ("publication.json" if publish else "validation.json")).write_text(
        json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", dest="publish", action="store_true", help="publish immutable objects (default)")
    mode.add_argument("--dry-run", dest="publish", action="store_false", help="validate with read-only R2 access")
    parser.set_defaults(publish=True)
    args = parser.parse_args()
    print(json.dumps(run(args.bundle_dir, read_client(), publish=args.publish), indent=2))


if __name__ == "__main__":
    main()
