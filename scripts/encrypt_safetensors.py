#!/usr/bin/env python3
"""Encrypt every .safetensors file in a model folder with an age recipient key."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_recipient(value: str) -> str:
    path = Path(value)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                if line.lower().startswith("age1"):
                    return line
                raise SystemExit(f"{path}: expected age1 recipient, got {line!r}")
        raise SystemExit(f"{path}: no age1 recipient found")
    if not value.lower().startswith("age1"):
        raise SystemExit("--recipient must be an age1 public key or a file containing one")
    return value


def ensure_age() -> None:
    if shutil.which("age") is None:
        raise SystemExit("missing `age` CLI. Install age first, then rerun.")


def assert_output_path_ok(source: Path, output: Path) -> None:
    if source == output:
        raise SystemExit("--output must differ from --folder")
    try:
        output.relative_to(source)
    except ValueError:
        return
    raise SystemExit("--output cannot be inside --folder")


def encrypt_file(src: Path, dst: Path, recipient: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["age", "-r", recipient, "-o", str(dst), str(src)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Model folder containing .safetensors files")
    parser.add_argument("--recipient", required=True, help="age1 public key, or path to public key file")
    parser.add_argument("--output", help="Encrypted output folder, default: <folder>-encrypted")
    parser.add_argument("--force", action="store_true", help="Delete existing output folder first")
    args = parser.parse_args()

    ensure_age()
    recipient = load_recipient(args.recipient)
    source = Path(args.folder).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"not a directory: {source}")

    output = Path(args.output).expanduser().resolve() if args.output else source.with_name(source.name + "-encrypted")
    assert_output_path_ok(source, output)
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {output} (use --force to replace)")
        shutil.rmtree(output)

    files = []
    for src in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = src.relative_to(source)
        dst = output / rel
        if src.name.endswith(".safetensors"):
            plain_sha256 = sha256_file(src)
            encrypt_file(src, dst, recipient)
            files.append({
                "path": str(rel).replace("\\", "/"),
                "plain_size": src.stat().st_size,
                "encrypted_size": dst.stat().st_size,
                "plain_sha256": plain_sha256,
                "encrypted_sha256": sha256_file(dst),
            })
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if not files:
        raise SystemExit(f"no .safetensors files found in {source}")

    manifest = {
        "version": 1,
        "scheme": "age-x25519",
        "recipient": recipient,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (output / "teutonic_encryption.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"encrypted {len(files)} safetensors files -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
