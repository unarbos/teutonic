#!/usr/bin/env python3
"""Build the local Gigatoken Horizon patch; public downloads and local writes only."""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True, help='new or empty local directory')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    patch_dir = root/'patches/gigatoken-horizon'
    pin = json.loads((patch_dir/'source-pin.json').read_text())
    with urllib.request.urlopen(pin['url'], timeout=120) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('upstream source archive SHA-256 mismatch')
    destination = args.build_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('--build-dir must be empty; existing work is never overwritten')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        archive.extractall(destination, filter='data')
    source = destination/'gigatoken-0.11.0'
    subprocess.run(['patch', '-p1', '--batch', '--forward', '-i', str(patch_dir/'horizon.patch')],
                   cwd=source, check=True)
    # Pin the compiler used for the verified wheel. An installed Rust toolchain
    # is required; this script does not install one or modify the shell profile.
    (source/'rust-toolchain.toml').write_text('[toolchain]\nchannel = "nightly-2026-11-10"\n')
    subprocess.run([sys.executable, '-m', 'maturin', 'build', '--release', '--locked',
                    '--interpreter', sys.executable, '--out', str(args.output_dir.resolve())],
                   cwd=source, env={**os.environ, 'CARGO_BUILD_JOBS': os.environ.get('CARGO_BUILD_JOBS','4')}, check=True)


if __name__ == '__main__':
    main()
