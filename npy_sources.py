from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from pydantic import BaseModel, Field

import eval_server_quasar_pair as base

log = logging.getLogger("eval_server_two_sources")

# ---------------------------------------------------------------------------
# Source registry – env-configurable defaults
# ---------------------------------------------------------------------------

_raw_manifest_urls = os.environ.get("TEUTONIC_MANIFEST_URLS", "").strip()
DEFAULT_MANIFEST_URLS: list[str] = (
    [u.strip() for u in _raw_manifest_urls.split(",") if u.strip()]
    if _raw_manifest_urls
    else [
        "https://eu-central-1.hippius.com/teutonic-sn3/dataset/automathtext-v2-quasar-10b/manifest.json",
        "https://us-east-1.hippius.com/teutonic-sn3/dataset/quasar-sn3-retok/manifest.json",
        "https://us-east-1.hippius.com/teutonic-sn3/dataset/ultradata-math-quasar-10b/manifest.json",
        "https://eu-central-1.hippius.com/teutonic-sn3/dataset/finewebedu/manifest.json",
    ]
)

# Fixed per-source sampling weights matched by substring against the source name.
# Override via TEUTONIC_SOURCE_WEIGHT_MAP="pattern1=w1,pattern2=w2,..."
# or TEUTONIC_SOURCE_WEIGHTS="w1,w2,..." (positional, aligned to npy_manifests order).
_raw_weight_map = os.environ.get("TEUTONIC_SOURCE_WEIGHT_MAP", "").strip()
DEFAULT_SOURCE_WEIGHT_MAP: dict[str, float] = (
    {
        k.strip(): float(v.strip())
        for pair in _raw_weight_map.split(",")
        if "=" in pair
        for k, v in [pair.split("=", 1)]
    }
    if _raw_weight_map
    else {
        "automathtext-v2": 0.35,
        "quasar-sn3": 0.05,
        "ultradata-math": 0.35,
        "finewebedu": 0.25,
    }
)

_raw_weights = os.environ.get("TEUTONIC_SOURCE_WEIGHTS", "")
DEFAULT_SOURCE_WEIGHTS: list[float] = (
    [float(w) for w in _raw_weights.split(",") if w.strip()]
    if _raw_weights.strip()
    else []
)
# vocab_size of the eval model — sequences with any token_id >= this value are dropped
# before inference to prevent CUDA device-side assert from embedding OOB access.
DEFAULT_VOCAB_SIZE: int = int(os.environ.get("TEUTONIC_VOCAB_SIZE", "248320"))
DEFAULT_MAX_SEQS_PER_SHARD: int = int(os.environ.get("TEUTONIC_MAX_SEQS_PER_SHARD", "0"))
URL_CACHE_DIR = Path(
    os.environ.get(
        "TEUTONIC_MULTI_SOURCE_CACHE_DIR",
        str(base.SHARD_CACHE_DIR / "_multi_source_urls"),
    )
)
MULTI_SOURCE_NAMES = {"multi", "multi_npy", "multi_source_npy", "two_sources"}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class NpyDataSource(BaseModel):
    name: str
    kind: str
    value: str | list[str]
    enabled: bool = True
    max_shards: int = 0


class MultiSourceEvalRequest(base.EvalRequest):
    dataset_source: str = "multi_source_npy"
    npy_manifests: list[str] = Field(default_factory=lambda: list(DEFAULT_MANIFEST_URLS))
    npy_sources: list[NpyDataSource] = Field(default_factory=list)
    source_mix_policy: str = "balanced"
    source_weights: list[float] = Field(default_factory=lambda: list(DEFAULT_SOURCE_WEIGHTS))
    max_seqs_per_shard: int = DEFAULT_MAX_SEQS_PER_SHARD
    vocab_size: int = DEFAULT_VOCAB_SIZE


@dataclass(frozen=True)
class ShardRef:
    source: str
    ref: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_multi_source(req: base.EvalRequest) -> bool:
    return (req.dataset_source or "").lower() in MULTI_SOURCE_NAMES


def text_from_ref(ref: str, req: MultiSourceEvalRequest) -> str:
    ref = str(ref).strip()
    parsed = urlparse(ref)
    if parsed.scheme in ("http", "https"):
        request = Request(ref, headers={"User-Agent": "teutonic-eval/1.0"})
        with urlopen(request, timeout=120) as resp:
            return resp.read().decode("utf-8")
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        s3_req = req.model_copy(update={"s3_bucket": bucket})
        client = base.make_s3_client(s3_req)
        return client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    path = Path(ref)
    if path.exists():
        return path.read_text()
    client = base.make_s3_client(req)
    return client.get_object(Bucket=req.s3_bucket, Key=ref.lstrip("/"))["Body"].read().decode("utf-8")


def public_url_from_manifest_key(manifest_ref: str, key: str) -> str | None:
    parsed = urlparse(manifest_ref)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        return None
    bucket = path_parts[0]
    key = key.lstrip("/")
    if key.startswith(f"{bucket}/"):
        return f"{parsed.scheme}://{parsed.netloc}/{key}"
    return f"{parsed.scheme}://{parsed.netloc}/{bucket}/{key}"


def normalize_manifest_ref(value: str, manifest: dict, manifest_ref: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme in ("http", "https", "s3") or Path(value).is_absolute():
        return value
    shard_prefix = str(manifest.get("shard_prefix") or manifest.get("prefix") or "").strip("/")
    key = value.lstrip("/")
    if shard_prefix and not key.startswith(f"{shard_prefix}/"):
        key = f"{shard_prefix}/{key}"
    public_url = public_url_from_manifest_key(manifest_ref, key)
    return public_url or key


def refs_from_manifest(source_name: str, manifest_ref: str, req: MultiSourceEvalRequest) -> list[ShardRef]:
    manifest = json.loads(text_from_ref(manifest_ref, req))
    if isinstance(manifest, dict):
        raw_shards = manifest.get("shards", [])
        manifest_dict = manifest
    elif isinstance(manifest, list):
        raw_shards = manifest
        manifest_dict = {}
    else:
        raise ValueError(f"manifest source {source_name!r} must be a JSON object or list")
    refs: list[ShardRef] = []
    for entry in raw_shards:
        value = ""
        if isinstance(entry, str):
            value = entry
        elif isinstance(entry, dict):
            for key in ("url", "href", "uri", "key", "path", "name"):
                if entry.get(key):
                    value = str(entry[key])
                    break
        normalized = normalize_manifest_ref(value, manifest_dict, manifest_ref)
        if normalized.endswith(".npy"):
            refs.append(ShardRef(source_name, normalized))
    if not refs:
        raise FileNotFoundError(f"manifest source {source_name!r} produced no .npy refs from {manifest_ref!r}")
    return refs


def refs_from_links_text(source_name: str, text: str) -> list[ShardRef]:
    refs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value = line.split()[0]
        if value.endswith(".npy"):
            refs.append(ShardRef(source_name, value))
    if not refs:
        raise FileNotFoundError(f"links source {source_name!r} produced no .npy refs")
    return refs


def refs_from_source(spec: NpyDataSource, req: MultiSourceEvalRequest) -> list[ShardRef]:
    kind = spec.kind.lower()
    if kind in ("manifest", "manifest_json"):
        refs = refs_from_manifest(spec.name, str(spec.value), req)
    elif kind in ("manifests", "manifest_list"):
        url_list = spec.value if isinstance(spec.value, list) else [str(spec.value)]
        refs = []
        for url in url_list:
            url = url.strip()
            if url:
                refs.extend(refs_from_manifest(spec.name, url, req))
    elif kind in ("links_file", "file"):
        refs = refs_from_links_text(spec.name, text_from_ref(str(spec.value), req))
    elif kind in ("url_list", "links"):
        if isinstance(spec.value, list):
            refs = [
                ShardRef(spec.name, str(item).strip())
                for item in spec.value
                if str(item).strip().endswith(".npy")
            ]
        else:
            refs = refs_from_links_text(spec.name, str(spec.value))
    else:
        raise ValueError(
            f"unsupported npy source kind={spec.kind!r}; expected manifest, manifests, links_file, or url_list"
        )
    if spec.max_shards > 0:
        refs = refs[: spec.max_shards]
    return refs


def _manifest_source_name(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 2 and parts[-1].endswith(".json"):
        return parts[-2]
    if parts:
        return parts[-1].replace(".json", "")
    return "manifest"


def default_sources(req: MultiSourceEvalRequest) -> list[NpyDataSource]:
    if req.npy_sources:
        return [source for source in req.npy_sources if source.enabled]
    return [
        NpyDataSource(name=_manifest_source_name(url), kind="manifest", value=url)
        for url in req.npy_manifests
        if url
    ]


def source_seed(base_seed: int, source_name: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{source_name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def source_targets(total: int, source_count: int, weights: list[float] | None = None) -> list[int]:
    if not weights or len(weights) != source_count:
        base_n = total // source_count
        remainder = total % source_count
        return [base_n + (1 if idx < remainder else 0) for idx in range(source_count)]
    total_w = sum(weights)
    raw = [total * w / total_w for w in weights]
    targets = [int(r) for r in raw]
    remainder = total - sum(targets)
    order = sorted(range(source_count), key=lambda i: -(raw[i] - targets[i]))
    for i in order[:remainder]:
        targets[i] += 1
    return targets


def url_cache_path(ref: str) -> Path:
    parsed = urlparse(ref)
    filename = Path(parsed.path).name or "shard.npy"
    digest = hashlib.sha256(ref.encode()).hexdigest()[:24]
    return URL_CACHE_DIR / f"{digest}-{filename}"


def download_url_ref(ref: str, on_phase=None) -> str:
    target = url_cache_path(ref)
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if on_phase:
        on_phase({"phase": "url_shard_download_start", "url": ref})
    request = Request(ref, headers={"User-Agent": "teutonic-eval/1.0"})
    with urlopen(request, timeout=600) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded URL shard is empty: {ref}")
    tmp.replace(target)
    if on_phase:
        on_phase({"phase": "url_shard_download_done", "url": ref, "path": str(target)})
    return str(target)


def materialize_shard(ref: ShardRef, req: MultiSourceEvalRequest, on_phase=None) -> str:
    parsed = urlparse(ref.ref)
    if parsed.scheme in ("http", "https"):
        return download_url_ref(ref.ref, on_phase=on_phase)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        s3_req = req.model_copy(update={"s3_bucket": bucket})
        client = base.make_s3_client(s3_req)
        return base.download_s3_shard(client, s3_req, key, on_phase=on_phase)
    path = Path(ref.ref)
    if path.exists():
        return str(path.resolve())
    client = base.make_s3_client(req)
    return base.download_s3_shard(client, req, ref.ref.lstrip("/"), on_phase=on_phase)


def static_source_weights(source_names: list[str]) -> list[float]:
    """Return fixed per-source weights by matching source names against DEFAULT_SOURCE_WEIGHT_MAP."""
    n = len(source_names)
    if n == 0:
        return []
    weights = []
    for name in source_names:
        weight = next(
            (w for pattern, w in DEFAULT_SOURCE_WEIGHT_MAP.items() if pattern in name),
            None,
        )
        weights.append(weight if weight is not None else 1.0 / n)
    return weights


def sample_balanced_multi_source(req: MultiSourceEvalRequest, on_phase=None) -> tuple[list[list[int]], dict]:
    seed_value = base.dataset_seed(req)
    rng = random.Random(seed_value)
    specs = default_sources(req)
    if not specs:
        raise ValueError("multi-source eval needs at least one npy source")

    refs_by_source: list[tuple[NpyDataSource, list[ShardRef]]] = []
    for spec in specs:
        refs = refs_from_source(spec, req)
        rng.shuffle(refs)
        refs_by_source.append((spec, refs))

    if req.s3_max_shards > 0:
        refs_by_source = [(spec, refs[: req.s3_max_shards]) for spec, refs in refs_by_source]

    if req.source_weights and len(req.source_weights) == len(refs_by_source):
        weights = req.source_weights
    else:
        weights = static_source_weights([spec.name for spec, _ in refs_by_source])
    targets = source_targets(int(req.n or base.DEFAULT_N), len(refs_by_source), weights=weights)
    log.info(
        "source_weights=%s",
        {spec.name: round(weights[idx], 4) for idx, (spec, _) in enumerate(refs_by_source)},
    )
    if on_phase:
        on_phase({
            "phase": "multi_source_listed",
            "dataset_seed": seed_value,
            "sources": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "shards": len(refs),
                    "weight": round(weights[idx], 4),
                    "target_sequences": targets[idx],
                }
                for idx, (spec, refs) in enumerate(refs_by_source)
            ],
        })

    sequences: list[list[int]] = []
    source_meta = []
    for target, (spec, refs) in zip(targets, refs_by_source):
        source_sequences: list[list[int]] = []
        used_refs: list[str] = []
        used_files: list[str] = []
        np_rng = np.random.default_rng(source_seed(seed_value, spec.name))
        for shard_ref in refs:
            if len(source_sequences) >= target:
                break
            local_path = materialize_shard(shard_ref, req, on_phase=on_phase)
            used_refs.append(shard_ref.ref)
            used_files.append(local_path)
            remaining = target - len(source_sequences)
            per_shard = min(remaining, req.max_seqs_per_shard) if req.max_seqs_per_shard > 0 else remaining
            # When vocab filtering is active, load with headroom so filtered-out
            # sequences don't leave us short.  The outer taken[:target] still caps
            # the final count; load_sequences_from_npy_shard clamps to shard size.
            load_limit = (int(per_shard * 1.5) + 8) if req.vocab_size > 0 else per_shard
            loaded = base.load_sequences_from_npy_shard(local_path, req, np_rng, load_limit)
            if req.vocab_size > 0:
                valid = [seq for seq in loaded if max(seq) < req.vocab_size]
                n_dropped = len(loaded) - len(valid)
                if n_dropped:
                    log.warning(
                        "source %r shard %s: dropped %d/%d seqs (token_id >= vocab_size=%d)",
                        spec.name,
                        shard_ref.ref.split("/")[-1],
                        n_dropped,
                        len(loaded),
                        req.vocab_size,
                    )
                source_sequences.extend(valid)
            else:
                source_sequences.extend(loaded)

        if len(source_sequences) < target:
            raise RuntimeError(
                f"source {spec.name!r} only produced {len(source_sequences)}/{target} sequences "
                f"from {len(used_refs)} shards"
            )
        taken = source_sequences[:target]
        sequences.extend(taken)
        log.info(
            "source %r: %d/%d seqs from %d shard(s): %s",
            spec.name,
            len(taken),
            target,
            len(used_refs),
            ", ".join(ref.split("/")[-1] for ref in used_refs),
        )
        if on_phase:
            on_phase({
                "phase": "source_sampled",
                "source": spec.name,
                "n_sequences": len(taken),
                "target_sequences": target,
                "used_shards": used_refs,
            })
        source_meta.append({
            "name": spec.name,
            "kind": spec.kind,
            "target_sequences": target,
            "n_sequences": len(taken),
            "available_shards": len(refs),
            "used_refs": used_refs,
            "used_files": used_files,
        })

    rng.shuffle(sequences)
    digest = hashlib.sha256(np.asarray(sequences, dtype=np.int64).tobytes()).hexdigest()
    log.info(
        "multi_source sample ready: n=%d seed=%d digest=%s sources=[%s]",
        len(sequences),
        seed_value,
        digest[:12],
        ", ".join(f"{m['name']}:{m['n_sequences']}seq/{len(m['used_refs'])}shards" for m in source_meta),
    )
    return sequences, {
        "n": len(sequences),
        "seq_len": req.seq_len,
        "seed": req.seed,
        "dataset_seed": seed_value,
        "seed_material": base.dataset_seed_material(req),
        "block_hash": req.block_hash,
        "hotkey": req.hotkey,
        "digest": digest,
        "source": "multi_source_npy",
        "source_mix_policy": req.source_mix_policy,
        "sources": source_meta,
    }


# ---------------------------------------------------------------------------
# Overrides that replace base.sample_eval_sequences / base.load_eval_tokenizer
# ---------------------------------------------------------------------------

_base_sample_eval_sequences = base.sample_eval_sequences
_base_load_eval_tokenizer = base.load_eval_tokenizer


def sample_eval_sequences(tokenizer, req: base.EvalRequest, on_phase=None):
    if is_multi_source(req):
        return sample_balanced_multi_source(req, on_phase=on_phase)
    return _base_sample_eval_sequences(tokenizer, req, on_phase=on_phase)


def load_eval_tokenizer(king_snapshot: str, req: base.EvalRequest, on_phase=None):
    if is_multi_source(req):
        return None, {"source": "not_needed_for_multi_source_npy"}
    return _base_load_eval_tokenizer(king_snapshot, req, on_phase=on_phase)


base.sample_eval_sequences = sample_eval_sequences
base.load_eval_tokenizer = load_eval_tokenizer
