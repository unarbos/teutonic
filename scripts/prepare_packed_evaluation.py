#!/usr/bin/env python3
"""Prepare local catalog and migrated competition manifests. No remote writes."""

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teutonic.evaluation.packed_text import (
    BUCKET,
    COLLECTION_FORMAT,
    MIX_FORMAT,
    PREFIX,
    DatasetReader,
    PackedDataset,
    build_catalog,
    canonical,
    combine_catalogs,
    read_client,
    validate_mix,
    validate_collection,
)


def prepare(dataset, definitions, registry, output, *, catalog=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(dataset) if catalog is None else catalog
    if catalog['dataset'] != dataset.pin:
        raise ValueError('catalog and dataset pins differ')
    raw = canonical(catalog)
    catalog_hash = hashlib.sha256(raw).hexdigest()
    catalog_key = f"{PREFIX}sampling/{catalog_hash}/manifest.json"
    (output / "sampling.json").write_bytes(raw)
    catalog_pin = {"bucket": BUCKET, "key": catalog_key, "sha256": catalog_hash}
    entries = []
    if dataset.manifest.get("format") == COLLECTION_FORMAT:
        collection_raw = canonical(dataset.manifest)
        if hashlib.sha256(collection_raw).hexdigest() != dataset.pin["sha256"]:
            raise ValueError("collection pin differs from canonical content")
        (output / "collection.json").write_bytes(collection_raw)
        entries.append({"local": "collection.json", **dataset.pin})
    entries.append({"local": "sampling.json", **catalog_pin})
    available = {(r["dataset"], r["category"], r["bucket"]) for r in catalog["partitions"]}
    for key, definition in sorted(definitions.items()):
        categories = [{"category": cat, "dataset": registry[cat]["dataset"], "weight": weight}
                      for cat, weight in sorted(definition["categories"].items())]
        for cat in categories:
            for bucket in ("2-4096", "4097-16384"):
                if (cat["dataset"], cat["category"], bucket) not in available:
                    raise ValueError(f"missing required bucket for {key}/{cat['category']}/{bucket}")
        mix = validate_mix({"format": MIX_FORMAT, "competition": key, "dataset": dataset.pin,
                            "sampling": catalog_pin, "categories": categories,
                            "token_budget": definition["long_document_tokens"]})
        raw = canonical(mix)
        sha = hashlib.sha256(raw).hexdigest()
        name = f"{key}.json"
        (output / name).write_bytes(raw)
        entries.append({"local": name, "bucket": BUCKET,
                        "key": f"{PREFIX}splits/{key}/{sha}.json", "sha256": sha})
    # This is a reviewable upload inventory, deliberately not an upload command.
    (output / "pending-publication.json").write_text(json.dumps(entries, indent=2) + "\n")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-pin", type=Path)
    source.add_argument("--collection", type=Path, help="local collection of pinned source datasets")
    parser.add_argument("--source-catalog", action="append", default=[], metavar="NAME=PATH",
                        help="reuse an existing source catalog; missing catalogs are built read-only")
    parser.add_argument("--definitions", type=Path, default=Path("chain.toml"),
                        help="reviewed TOML; use active DB weights for a production migration")
    from teutonic.competitions import ACTIVE_COMPETITIONS
    parser.add_argument("--competition", action="append", required=True, choices=ACTIVE_COMPETITIONS,
                        help="explicitly select competitions; never auto-reactivate retired mixes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--page-bytes", type=int, default=4096)
    parser.add_argument("--catalog-state", type=Path)
    parser.add_argument("--dataset-manifest-file", type=Path,
                        help="reuse a local manifest after checking the dataset-pin SHA-256")
    args = parser.parse_args()
    if args.collection and (args.dataset_manifest_file or args.catalog_state):
        parser.error("collection preparation uses per-source manifests and checkpoints")
    config = tomllib.loads(args.definitions.read_text())
    definitions = {key: config["competitions"][key] for key in args.competition}
    collection = json.loads(args.collection.read_text()) if args.collection else None
    if collection is not None:
        validate_collection(collection)
        sha = hashlib.sha256(canonical(collection)).hexdigest()
        pin = {"bucket": BUCKET, "key": f"{PREFIX}collections/{sha}/manifest.json", "sha256": sha}
    else:
        pin = json.loads(args.dataset_pin.read_text())
    class Reader(DatasetReader):
        def json(self, requested):
            if collection is not None and requested == pin:
                return collection
            if args.dataset_manifest_file is None:
                return super().json(requested)
            raw = args.dataset_manifest_file.read_bytes()
            if requested != pin or hashlib.sha256(raw).hexdigest() != pin['sha256']:
                raise ValueError('local dataset manifest pin mismatch')
            return json.loads(raw)
    dataset = PackedDataset(Reader(read_client(max_pool_connections=max(32,args.workers))), pin)
    from teutonic.evaluation.packed_catalog import build_resumable_catalog
    def build(source, state):
        return build_resumable_catalog(source, state, workers=args.workers, page_bytes=args.page_bytes,
            on_progress=lambda event: print(json.dumps(event), flush=True))
    if collection is not None:
        supplied = {}
        for item in args.source_catalog:
            name, path = item.split("=", 1)
            if name not in dataset.sources or name in supplied:
                raise ValueError("unknown or duplicate source catalog")
            supplied[name] = json.loads(Path(path).read_text())
        catalogs = {name: supplied[name] if name in supplied else
                    build(source, args.output_dir/f"catalog-{name}.sqlite")
                    for name, source in dataset.sources.items()}
        catalog = combine_catalogs(pin, collection, catalogs)
    else:
        if args.source_catalog:
            raise ValueError("source catalogs require a collection")
        catalog = build(dataset, args.catalog_state or args.output_dir/'catalog-state.sqlite')
    entries = prepare(dataset, definitions, config["categories"], args.output_dir, catalog=catalog)
    print(json.dumps({"prepared_files": len(entries), "remote_writes": 0}))


if __name__ == "__main__":
    main()
