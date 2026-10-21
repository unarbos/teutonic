#!/usr/bin/env python3
"""Prepare a local immutable mix bundle; publication is explicit and version-only."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chain_config
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    canonical_manifest_bytes,
    fetch_dataset_manifest,
)
from teutonic.evaluation.mixes import build_mix_manifests, validate_definitions, write_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, help="read DATASET.json locally instead of fetching"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true", help="upload immutable version URLs only")
    parser.add_argument("--bucket", default=os.environ.get("TEUTONIC_DATASETS_BUCKET", "datasets"))
    args = parser.parse_args(argv)
    validate_definitions()
    sources = []
    for name, spec in chain_config.CATEGORY_DATASETS.items():
        if args.source_dir:
            manifest = json.loads((args.source_dir / f"{name}.json").read_text())
            source = DatasetManifestSnapshot(
                name,
                spec["manifest_url"],
                hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
                1.0,
                manifest,
            )
        else:
            source = fetch_dataset_manifest(
                name=name, manifest_url=spec["manifest_url"], proportion=1.0
            )
        sources.append(source)
    entries = write_bundle(build_mix_manifests(sources), args.output_dir)
    # Inventory input is independent of any competition mix, including MAIN.
    inventory = {
        "datasets": {
            s.name: {
                "categories": sorted(
                    c for c, v in chain_config.CATEGORY_REGISTRY.items() if v["dataset"] == s.name
                ),
                "source_manifest": {"url": s.manifest_url, "sha256": s.manifest_sha256},
            }
            for s in sources
        }
    }
    (args.output_dir / "index-sources.json").write_text(json.dumps(inventory, indent=2) + "\n")
    if args.publish:
        from scripts.publish_category_artifacts import publish_bundle

        publish_bundle(args.output_dir, args.bucket)
    print(json.dumps({"competitions": entries, "published": args.publish}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
