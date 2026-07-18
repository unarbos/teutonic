#!/usr/bin/env python3
"""Decrypt .safetensors files previously encrypted by encrypt_safetensors.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


MANIFEST_NAME = "teutonic_encryption.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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


def decrypt_file(src: Path, dst: Path, identity: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["age", "-d", "-i", str(identity), "-o", str(dst), str(src)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Encrypted model folder")
    parser.add_argument("--identity", required=True, help="age private key file")
    parser.add_argument("--output", help="Decrypted output folder, default: <folder>-decrypted")
    parser.add_argument("--force", action="store_true", help="Delete existing output folder first")
    args = parser.parse_args()

    ensure_age()
    source = Path(args.folder).expanduser().resolve()
    identity = Path(args.identity).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"not a directory: {source}")
    if not identity.is_file():
        raise SystemExit(f"not a private key file: {identity}")

    manifest_path = source / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"missing {MANIFEST_NAME} in {source}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scheme") != "age-x25519":
        raise SystemExit(f"unsupported encryption scheme: {manifest.get('scheme')!r}")

    output = Path(args.output).expanduser().resolve() if args.output else source.with_name(source.name + "-decrypted")
    assert_output_path_ok(source, output)
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {output} (use --force to replace)")
        shutil.rmtree(output)

    entries = {item["path"]: item for item in manifest.get("files", [])}
    if not entries:
        raise SystemExit(f"{MANIFEST_NAME} has no encrypted files")

    for src in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = str(src.relative_to(source)).replace("\\", "/")
        dst = output / rel
        if rel in entries:
            decrypt_file(src, dst, identity)
            item = entries[rel]
            got_sha = sha256_file(dst)
            if got_sha != item.get("plain_sha256"):
                raise SystemExit(f"{rel}: plaintext sha256 mismatch after decrypt")
            if dst.stat().st_size != int(item.get("plain_size", -1)):
                raise SystemExit(f"{rel}: plaintext size mismatch after decrypt")
        elif src.name.endswith(".safetensors"):
            raise SystemExit(f"{rel}: .safetensors file is missing from {MANIFEST_NAME}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"decrypted {len(entries)} safetensors files -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
