#!/usr/bin/env python3
"""Split a single model.safetensors into Transformers-compatible shards."""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

from safetensors.torch import load_file, save_file


SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgt]?b?)?$", re.I)
UNIT_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000**2,
    "mb": 1000**2,
    "g": 1000**3,
    "gb": 1000**3,
    "t": 1000**4,
    "tb": 1000**4,
}


def parse_size(value: str) -> int:
    match = SIZE_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid size: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    return int(number * UNIT_MULTIPLIERS[unit])


def tensor_nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def shard_state_dict(state: dict, max_shard_size: int, weights_name: str):
    shards: list[OrderedDict] = []
    current = OrderedDict()
    current_size = 0
    weight_map = {}
    total_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            shards.append(current)
            current = OrderedDict()
            current_size = 0

    for name, tensor in state.items():
        size = tensor_nbytes(tensor)
        total_size += size
        if current and current_size + size > max_shard_size:
            flush()
        current[name] = tensor
        current_size += size

    flush()

    total_shards = len(shards)
    shard_files = OrderedDict()
    stem = weights_name.removesuffix(".safetensors")
    for idx, shard in enumerate(shards, start=1):
        filename = f"{stem}-{idx:05d}-of-{total_shards:05d}.safetensors"
        shard_files[filename] = shard
        for weight_name in shard:
            weight_map[weight_name] = filename

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    return shard_files, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="/root/models/teutonic-5gbdp8ba-test-dv00",
        help="Directory containing model.safetensors.",
    )
    parser.add_argument("--input", default="model.safetensors")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--delete-original", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    src = model_dir / args.input
    if not src.exists():
        raise FileNotFoundError(src)

    max_shard_size = parse_size(args.max_shard_size)
    existing = sorted(model_dir.glob("model-*-of-*.safetensors"))
    index_path = model_dir / "model.safetensors.index.json"
    if (existing or index_path.exists()) and not args.overwrite:
        raise RuntimeError(
            "shards/index already exist; pass --overwrite if you really want to rewrite them"
        )

    print(f"loading {src}")
    state = load_file(str(src), device="cpu")
    print(f"loaded {len(state)} tensors; max_shard_size={max_shard_size} bytes")

    shard_files, index = shard_state_dict(state, max_shard_size, "model.safetensors")
    print(f"writing {len(shard_files)} shard(s)")
    for filename, shard in shard_files.items():
        out = model_dir / filename
        print(f"  {filename}: {len(shard)} tensor(s)")
        save_file(shard, str(out), metadata={"format": "pt"})

    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"wrote {index_path}")

    if args.delete_original:
        src.unlink()
        print(f"deleted {src}")
    else:
        print(f"kept original {src}; delete it after verifying the shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
