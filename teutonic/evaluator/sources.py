from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from teutonic.evaluator import engine as base


log = logging.getLogger("teutonic.evaluator.sources")
URL_CACHE_DIR = Path(
    os.environ.get(
        "TEUTONIC_PRETOKENIZED_CACHE_DIR",
        str(base.SHARD_CACHE_DIR / "pretokenized"),
    )
)


@dataclass(frozen=True, slots=True)
class ShardRef:
    source: str
    url: str
    sha256: str
    size_bytes: int
    n_tokens: int


def _cache_path(shard: ShardRef) -> Path:
    filename = Path(shard.url.split("?", 1)[0]).name or "shard.npy"
    return URL_CACHE_DIR / f"{shard.sha256[:24]}-{filename}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, shard: ShardRef) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == shard.size_bytes
        and _sha256_file(path) == shard.sha256
    )


def materialize_shard(shard: ShardRef, on_phase=None) -> Path:
    target = _cache_path(shard)
    if _verified(target, shard):
        return target
    target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    if on_phase:
        on_phase({"phase": "pretokenized_shard_download_start", "url": shard.url})
    request = Request(shard.url, headers={"User-Agent": "teutonic-eval/1.0"})
    try:
        with urlopen(request, timeout=600) as response, partial.open("xb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        if not _verified(partial, shard):
            raise RuntimeError("downloaded shard differs from validator descriptor")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if on_phase:
        on_phase(
            {
                "phase": "pretokenized_shard_download_done",
                "url": shard.url,
                "sha256": shard.sha256,
                "size_bytes": shard.size_bytes,
            }
        )
    return target


def _load_with_retry(
    shard: ShardRef,
    req: base.EvalRequest,
    rng: np.random.Generator,
    limit: int,
    on_phase=None,
) -> tuple[Path, list[tuple[int, list[int]]]]:
    path = materialize_shard(shard, on_phase=on_phase)
    try:
        return path, base.load_indexed_sequences_from_npy_shard(
            str(path), req, rng, limit
        )
    except Exception as exc:
        if not base.is_truncated_npy_error(exc):
            raise
        path.unlink(missing_ok=True)
        path = materialize_shard(shard, on_phase=on_phase)
        return path, base.load_indexed_sequences_from_npy_shard(
            str(path), req, rng, limit
        )


def _source_seed(dataset_seed: int, source: str) -> int:
    digest = hashlib.blake2b(f"{dataset_seed}:{source}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _even_quotas(target: int, n_shards: int) -> list[int]:
    """Split a source target as evenly as possible across its shards."""
    if n_shards < 1:
        raise ValueError("cannot split an evaluation target across zero shards")
    base, extra = divmod(target, n_shards)
    return [base + (1 if index < extra else 0) for index in range(n_shards)]


def sample_pretokenized_sequences(
    req: base.EvalRequest, on_phase=None
) -> tuple[list[list[int]], dict]:
    dataset_seed = base.dataset_seed(req)
    sources = req.dataset_sources
    if not sources:
        raise ValueError("validator request contains no pre-tokenized dataset sources")
    if sum(int(source["target_sequences"]) for source in sources) != req.n:
        raise ValueError("validator source targets do not sum to evaluation sample count")

    sequences: list[list[int]] = []
    source_labels: list[str] = []
    sample_provenance: list[dict[str, int]] = []
    source_meta: list[dict] = []
    for shard_group_index, source in enumerate(sources):
        name = str(source["name"])
        target = int(source["target_sequences"])
        rng = np.random.default_rng(_source_seed(dataset_seed, name))
        refs = [
            ShardRef(
                source=name,
                url=str(raw["url"]),
                sha256=str(raw["sha256"]),
                size_bytes=int(raw["size_bytes"]),
                n_tokens=int(raw["n_tokens"]),
            )
            for raw in source["shards"]
        ]
        # Every shard the validator selected contributes its share, so no single
        # shard's content drives the source's contribution to the verdict. A
        # validator that stratifies by dataset category sends each shard's count
        # explicitly, because category weights cannot be carried by shard count
        # alone; otherwise the target is split evenly across the shards sent.
        declared = [raw.get("target_sequences") for raw in source["shards"]]
        raw_targets_explicit = all(isinstance(value, int) and value > 0 for value in declared)
        if raw_targets_explicit:
            if sum(declared) != target:
                raise ValueError(
                    f"source {name!r} shard targets sum to {sum(declared)}, "
                    f"expected {target}"
                )
            quotas = [int(value) for value in declared]
        else:
            refs = refs[:target] if target < len(refs) else refs
            quotas = _even_quotas(target, len(refs))
        loaded_per_shard: list[list[tuple[int, list[int]]]] = []
        used_shards: list[dict] = []
        for shard, quota in zip(refs, quotas, strict=True):
            load_limit = int(quota * 1.5) + 8 if req.vocab_size > 0 else quota
            _local_path, loaded = _load_with_retry(
                shard, req, rng, load_limit, on_phase=on_phase
            )
            if req.vocab_size > 0:
                loaded = [
                    (sequence_index, sequence)
                    for sequence_index, sequence in loaded
                    if max(sequence) < req.vocab_size
                ]
            if raw_targets_explicit and len(loaded) < quota:
                raise RuntimeError(
                    f"source {name!r} shard {shard.url!r} produced {len(loaded)}/{quota} "
                    "valid sequences; refusing to change declared category allocation"
                )
            loaded_per_shard.append(loaded)
            used_shards.append(
                {
                    "url": shard.url,
                    "sha256": shard.sha256,
                }
            )
        selected: list[tuple[list[int], dict[str, int]]] = []
        for shard_index, (loaded, quota) in enumerate(
            zip(loaded_per_shard, quotas, strict=True)
        ):
            selected.extend(
                (
                    sequence,
                    {
                        "shard_group_index": shard_group_index,
                        "shard_index": shard_index,
                        "shard_sequence_index": sequence_index,
                    },
                )
                for sequence_index, sequence in loaded[:quota]
            )
        # A shard that came up short after vocab filtering is covered by the
        # spare sequences its siblings already loaded.
        if len(selected) < target:
            for shard_index, (loaded, quota) in enumerate(
                zip(loaded_per_shard, quotas, strict=True)
            ):
                if len(selected) >= target:
                    break
                selected.extend(
                    (
                        sequence,
                        {
                            "shard_group_index": shard_group_index,
                            "shard_index": shard_index,
                            "shard_sequence_index": sequence_index,
                        },
                    )
                    for sequence_index, sequence in loaded[
                        quota : quota + target - len(selected)
                    ]
                )
        if len(selected) < target:
            raise RuntimeError(
                f"source {name!r} produced {len(selected)}/{target} requested sequences"
            )
        taken = selected[:target]
        sequences.extend(sequence for sequence, _provenance in taken)
        sample_provenance.extend(provenance for _sequence, provenance in taken)
        source_labels.extend([name] * target)
        source_meta.append(
            {
                "name": name,
                "proportion": float(source["proportion"]),
                "target_sequences": target,
                "n_sequences": len(taken),
                "used_shards": used_shards,
                "used_refs": [item["url"] for item in used_shards],
            }
        )
        if on_phase:
            on_phase(
                {
                    "phase": "pretokenized_source_sampled",
                    "source": name,
                    "target_sequences": target,
                    "n_sequences": len(taken),
                    "used_shards": len(used_shards),
                }
            )

    tagged = list(zip(sequences, source_labels, sample_provenance, strict=True))
    random.Random(dataset_seed).shuffle(tagged)
    sequences = [sequence for sequence, _source, _provenance in tagged]
    source_labels = [source for _sequence, source, _provenance in tagged]
    sample_provenance = [
        provenance for _sequence, _source, provenance in tagged
    ]
    digest = hashlib.sha256(np.asarray(sequences, dtype=np.int64).tobytes()).hexdigest()
    return sequences, {
        "n": len(sequences),
        "seq_len": req.seq_len,
        "dataset_seed": dataset_seed,
        "seed_material": base.dataset_seed_material(req),
        "block_hash": req.block_hash,
        "hotkey": req.hotkey,
        "digest": digest,
        "source": "pretokenized_npy",
        "sources": source_meta,
        "_source_labels": source_labels,
        "_sample_provenance": sample_provenance,
    }


def sample_eval_sequences(req: base.EvalRequest, on_phase=None):
    if req.dataset_source != "pretokenized_npy":
        raise ValueError("GPU evaluator accepts only validator-provided pre-tokenized NPY shards")
    return sample_pretokenized_sequences(req, on_phase=on_phase)


base.sample_eval_sequences = sample_eval_sequences
