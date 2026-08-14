#!/usr/bin/env python3
"""Print the current dataset bundle digest for chain.toml [dataset].bundle_digest.

Changing the eval mixture is a scoring-contract change, so it is meant to be a
reviewed commit: run this, eyeball the source list and weights it prints, and
paste the digest into chain.toml.

    python scripts/pin_dataset_bundle.py            # default bundle from chain.toml
    python scripts/pin_dataset_bundle.py --url ...  # a candidate bundle
    python scripts/pin_dataset_bundle.py --check    # exit 1 if it drifted from the pin
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing npy_sources pulls in the eval server (torch). Import the bundle
# helpers only, via the module, to keep the failure mode obvious if that breaks.
from npy_sources import (  # noqa: E402
    DEFAULT_BUNDLE_MANIFEST_URL,
    EXPECTED_BUNDLE_DIGEST,
    bundle_to_sources,
    fetch_bundle_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_BUNDLE_MANIFEST_URL)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the live bundle no longer matches chain.toml's pin",
    )
    args = parser.parse_args()

    bundle, digest = fetch_bundle_manifest(args.url)
    manifest_urls, weight_map = bundle_to_sources(bundle)

    print(f"url    = {args.url}")
    print(f"digest = {digest}")
    print(f"sources ({len(manifest_urls)}):")
    total = 0.0
    for url in manifest_urls:
        name = url.rstrip("/").split("/")[-2] if "/" in url else url
        weight = weight_map.get(name, 0.0)
        total += weight
        print(f"  {weight:>6.3f}  {url}")
    print(f"weight sum = {total:.3f}")

    if args.check:
        if not EXPECTED_BUNDLE_DIGEST:
            print("\nchain.toml [dataset].bundle_digest is empty — nothing to check", file=sys.stderr)
            return 1
        if digest != EXPECTED_BUNDLE_DIGEST:
            print(
                f"\nDRIFT: pinned={EXPECTED_BUNDLE_DIGEST} live={digest}",
                file=sys.stderr,
            )
            return 1
        print("\nmatches chain.toml pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
