from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from pydantic import BaseModel, Field

import eval_server_quasar_pair as base

# chain_config sits at the repo root; ensure it imports regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_config  # noqa: E402

log = logging.getLogger("eval_server_two_sources")

# ---------------------------------------------------------------------------
# Source registry – pinned dataset bundle, resolved once at server startup
# ---------------------------------------------------------------------------
#
# The eval mixture IS the measuring stick. `delta_threshold` is a fixed constant
# (0.0015 nats), so changing which corpora are sampled — or their mix weights —
# moves the bar every challenger is judged against, while the recorded verdict
# still says plain "accepted: true". Two consequences drive the design here:
#
#   1. There is deliberately NO hardcoded fallback mixture. A manifest we cannot
#      fetch or cannot verify must stop the eval server. The validator then
#      falls back to burn weights, which is honest and reversible; scoring
#      against a silently different mixture corrupts the king chain forever.
#   2. The bundle is pinned by digest in chain.toml, so changing the mixture is
#      a reviewed commit rather than an invisible edit to a bucket object.
#
# Resolution is explicit (`ensure_bundle_resolved`, called from the server
# lifespan) rather than an import side effect: importing this module must not do
# network I/O, and a failure should surface as a refused startup rather than a
# half-initialised module.

# Precedence: env override > chain.toml > built-in default. Only the digest
# gates correctness, so an unset chain.toml URL is not fatal.
DEFAULT_BUNDLE_MANIFEST_URL = (
    os.environ.get("TEUTONIC_BUNDLE_MANIFEST_URL", "").strip()
    or chain_config.DATASET_BUNDLE_URL
    or "https://s3.hippius.com/teutonic-sn3/dataset/all-datasets.manifest.json"
)
EXPECTED_BUNDLE_DIGEST = (
    os.environ.get("TEUTONIC_BUNDLE_DIGEST", "").strip() or chain_config.DATASET_BUNDLE_DIGEST
)

# Bounded retry: a transient blip should not take the eval box down, but an
# unreachable manifest must eventually raise rather than degrade.
BUNDLE_FETCH_ATTEMPTS = max(1, int(os.environ.get("TEUTONIC_BUNDLE_FETCH_ATTEMPTS", "3")))
BUNDLE_FETCH_BACKOFF_S = float(os.environ.get("TEUTONIC_BUNDLE_FETCH_BACKOFF_S", "5"))
BUNDLE_FETCH_TIMEOUT_S = float(os.environ.get("TEUTONIC_BUNDLE_FETCH_TIMEOUT_S", "30"))


class BundleManifestError(RuntimeError):
    """The dataset bundle could not be fetched, parsed, or matched to its pin."""


def bundle_digest(raw: bytes) -> str:
    """sha256 over the exact manifest bytes, in chain.toml's digest format."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def fetch_bundle_manifest(
    url: str,
    *,
    attempts: int = BUNDLE_FETCH_ATTEMPTS,
    backoff_s: float = BUNDLE_FETCH_BACKOFF_S,
    timeout_s: float = BUNDLE_FETCH_TIMEOUT_S,
) -> tuple[dict, str]:
    """Fetch and parse the dataset bundle manifest at `url`.

    Returns (parsed_manifest, digest). Retries a bounded number of times on any
    failure and then raises — never returns a substitute mixture.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": "teutonic-eval/1.0"})
            with urlopen(request, timeout=timeout_s) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8")), bundle_digest(raw)
        except Exception as exc:
            last_exc = exc
            log.warning(
                "dataset bundle fetch attempt %d/%d failed for %s: %s", attempt, attempts, url, exc
            )
            if attempt < attempts:
                time.sleep(backoff_s * attempt)
    raise BundleManifestError(
        f"could not fetch dataset bundle manifest {url} after {attempts} attempts"
    ) from last_exc


def verify_bundle_digest(actual: str, expected: str) -> None:
    """Enforce the chain.toml pin. Empty pin logs loudly but allows startup."""
    if not expected:
        log.warning(
            "dataset bundle is UNPINNED (chain.toml [dataset].bundle_digest is empty); "
            "resolved digest=%s — set it to make mixture drift fatal",
            actual,
        )
        return
    if actual != expected:
        raise BundleManifestError(
            f"dataset bundle digest mismatch: expected={expected} actual={actual}; "
            "the eval mixture changed under a pinned chain.toml — refusing to score"
        )


def resolve_bundle_sources(
    url: str = DEFAULT_BUNDLE_MANIFEST_URL,
    expected_digest: str = EXPECTED_BUNDLE_DIGEST,
    **fetch_kwargs,
) -> tuple[list[str], dict[str, float], str]:
    """Fetch, verify, and unpack the bundle into (manifest_urls, weights, digest)."""
    bundle, digest = fetch_bundle_manifest(url, **fetch_kwargs)
    verify_bundle_digest(digest, expected_digest)
    manifest_urls, weight_map = bundle_to_sources(bundle)
    if not manifest_urls:
        raise BundleManifestError(f"dataset bundle {url} has no enabled sources")
    return manifest_urls, weight_map, digest


def bundle_to_sources(bundle: dict) -> tuple[list[str], dict[str, float]]:
    """Extract (manifest_urls, name->weight map) from a bundle's enabled sources."""
    manifest_urls: list[str] = []
    weight_map: dict[str, float] = {}
    for entry in bundle.get("sources", []):
        if not entry.get("enabled", True):
            continue
        url = entry.get("manifest_url")
        name = entry.get("name")
        if not url or not name:
            continue
        manifest_urls.append(url)
        if entry.get("weight") is not None:
            weight = float(entry["weight"])
            weight_map[name] = weight
            parts = [part for part in urlparse(url).path.split("/") if part]
            url_source_name = parts[-2] if len(parts) >= 2 else ""
            if url_source_name and url_source_name != name:
                # The bundle name may use hyphens while its manifest directory
                # uses underscores. Default sources are named from the URL.
                weight_map[url_source_name] = weight
    return manifest_urls, weight_map


_raw_manifest_urls = os.environ.get("TEUTONIC_MANIFEST_URLS", "").strip()
_raw_weight_map = os.environ.get("TEUTONIC_SOURCE_WEIGHT_MAP", "").strip()

# Populated by ensure_bundle_resolved(); empty until then. Env overrides are
# applied eagerly because they need no network.
DEFAULT_MANIFEST_URLS: list[str] = [
    u.strip() for u in _raw_manifest_urls.split(",") if u.strip()
]

# Fixed per-source sampling weights matched by substring against the source name.
# Override via TEUTONIC_SOURCE_WEIGHT_MAP="pattern1=w1,pattern2=w2,..."
# or TEUTONIC_SOURCE_WEIGHTS="w1,w2,..." (positional, aligned to npy_manifests order).
DEFAULT_SOURCE_WEIGHT_MAP: dict[str, float] = {
    k.strip(): float(v.strip())
    for pair in _raw_weight_map.split(",")
    if "=" in pair
    for k, v in [pair.split("=", 1)]
}

# Provenance of the active mixture, stamped into every verdict so a decision can
# be audited against the exact source list that produced it.
RESOLVED_BUNDLE: dict[str, object] = {}


def ensure_bundle_resolved(*, force: bool = False) -> dict:
    """Resolve the dataset mixture into the module defaults. Idempotent.

    Raises BundleManifestError if the bundle is unreachable or fails its pin —
    the caller (server lifespan) must let that abort startup rather than serve
    evals against an unknown mixture.
    """
    if RESOLVED_BUNDLE and not force:
        return dict(RESOLVED_BUNDLE)

    if _raw_manifest_urls or _raw_weight_map:
        # An explicit env override means the operator does not want the bundle
        # consulted at all. Legitimate for local dev; recorded so a verdict
        # produced this way is never mistaken for a pinned-mixture verdict.
        log.warning(
            "dataset sources come from env overrides (%d manifests); bundle manifest not consulted",
            len(DEFAULT_MANIFEST_URLS),
        )
        RESOLVED_BUNDLE.update({"origin": "env_override", "url": "", "digest": "", "pinned": False})
        return dict(RESOLVED_BUNDLE)

    urls, weights, digest = resolve_bundle_sources(
        DEFAULT_BUNDLE_MANIFEST_URL, EXPECTED_BUNDLE_DIGEST
    )
    # Mutated in place, not rebound: eval_server_two_sources imports these names
    # by value, so rebinding here would leave that module holding stale objects.
    DEFAULT_MANIFEST_URLS.clear()
    DEFAULT_MANIFEST_URLS.extend(urls)
    DEFAULT_SOURCE_WEIGHT_MAP.clear()
    DEFAULT_SOURCE_WEIGHT_MAP.update(weights)
    RESOLVED_BUNDLE.update({
        "origin": "bundle",
        "url": DEFAULT_BUNDLE_MANIFEST_URL,
        "digest": digest,
        "pinned": bool(EXPECTED_BUNDLE_DIGEST),
        "n_sources": len(urls),
    })
    log.info(
        "dataset bundle resolved: %d sources digest=%s pinned=%s",
        len(urls),
        digest,
        bool(EXPECTED_BUNDLE_DIGEST),
    )
    return dict(RESOLVED_BUNDLE)


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
DEFAULT_SHARDS_PER_SOURCE: int = int(os.environ.get("TEUTONIC_SHARDS_PER_SOURCE", "2"))
URL_CACHE_DIR = Path(
    os.environ.get(
        "TEUTONIC_MULTI_SOURCE_CACHE_DIR",
        str(base.SHARD_CACHE_DIR / "_multi_source_urls"),
    )
)
MULTI_SOURCE_NAMES = {"multi", "multi_npy", "multi_source_npy", "two_sources"}
MANIFEST_SHARD_URL_OVERRIDES: dict[tuple[str, str], str] = {
    (
        "https://s3.hippius.com/tokens-here/dataset/quasar-synth-v1/manifest.json",
        "dataset/quasar-synth-run/shards/shard_000000.npy",
    ): "https://s3.hippius.com/tokens-here/dataset/quasar-synth-v1/shards/shard_000000.npy",
}

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
    shards_per_source: int = DEFAULT_SHARDS_PER_SOURCE
    vocab_size: int = DEFAULT_VOCAB_SIZE


@dataclass(frozen=True)
class ShardRef:
    source: str
    ref: str
    seq_len: int | None = None


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
    manifest_url = manifest_ref.split("?", 1)[0]
    override = MANIFEST_SHARD_URL_OVERRIDES.get((manifest_url, key))
    if override:
        return override
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
    manifest_seq_len = int(
        manifest_dict.get("seq_len") or manifest_dict.get("sequence_length") or 0
    )
    refs: list[ShardRef] = []
    for entry in raw_shards:
        value = ""
        shard_seq_len = manifest_seq_len
        if isinstance(entry, str):
            value = entry
        elif isinstance(entry, dict):
            shard_seq_len = int(
                entry.get("seq_len")
                or entry.get("sequence_length")
                or manifest_seq_len
                or 0
            )
            for key in ("url", "href", "uri", "key", "path", "name"):
                if entry.get(key):
                    value = str(entry[key])
                    break
        normalized = normalize_manifest_ref(value, manifest_dict, manifest_ref)
        if normalized.endswith(".npy"):
            refs.append(ShardRef(source_name, normalized, shard_seq_len or None))
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
    if not req.npy_manifests:
        # Reachable only if the server started without resolving the bundle.
        # Refuse rather than sample from an empty/unknown mixture.
        raise BundleManifestError(
            "no dataset sources available — ensure_bundle_resolved() was not run or resolved empty"
        )
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


def can_redownload_shard(ref: ShardRef) -> bool:
    parsed = urlparse(ref.ref)
    return parsed.scheme in ("http", "https", "s3") or not Path(ref.ref).exists()


def load_materialized_shard_with_retry(
    ref: ShardRef,
    req: MultiSourceEvalRequest,
    rng: np.random.Generator,
    limit: int | None,
    on_phase=None,
) -> tuple[str, list[list[int]]]:
    local_path = materialize_shard(ref, req, on_phase=on_phase)
    try:
        return local_path, base.load_sequences_from_npy_shard(
            local_path,
            req,
            rng,
            limit,
            source_seq_len=ref.seq_len,
        )
    except Exception as exc:
        if not base.is_truncated_npy_error(exc) or not can_redownload_shard(ref):
            raise
        log.warning("cached npy shard is truncated; deleting and redownloading: %s", local_path)
        Path(local_path).unlink(missing_ok=True)
        local_path = materialize_shard(ref, req, on_phase=on_phase)
        return local_path, base.load_sequences_from_npy_shard(
            local_path,
            req,
            rng,
            limit,
            source_seq_len=ref.seq_len,
        )


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
                    "source_seq_len": refs[0].seq_len if refs else None,
                    "target_shards": req.shards_per_source,
                    "weight": round(weights[idx], 4),
                    "target_sequences": targets[idx],
                }
                for idx, (spec, refs) in enumerate(refs_by_source)
            ],
        })

    sequences: list[list[int]] = []
    source_labels: list[str] = []  # parallel to sequences; same index → same source
    source_meta = []
    for target, (spec, refs) in zip(targets, refs_by_source):
        source_sequences: list[list[int]] = []
        used_refs: list[str] = []
        used_files: list[str] = []
        np_rng = np.random.default_rng(source_seed(seed_value, spec.name))
        target_shards = min(req.shards_per_source, len(refs)) if req.shards_per_source > 0 else len(refs)
        shard_targets = source_targets(target, target_shards)
        for shard_idx, shard_ref in enumerate(refs):
            if len(source_sequences) >= target:
                break
            remaining = target - len(source_sequences)
            shard_target = shard_targets[shard_idx] if shard_idx < target_shards else remaining
            per_shard = min(shard_target, req.max_seqs_per_shard) if req.max_seqs_per_shard > 0 else shard_target
            # When vocab filtering is active, load with headroom so filtered-out
            # sequences don't leave us short.  The outer taken[:target] still caps
            # the final count; load_sequences_from_npy_shard clamps to shard size.
            load_limit = (int(per_shard * 1.5) + 8) if req.vocab_size > 0 else per_shard
            local_path, loaded = load_materialized_shard_with_retry(
                shard_ref, req, np_rng, load_limit, on_phase=on_phase
            )
            used_refs.append(shard_ref.ref)
            used_files.append(local_path)
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
        source_labels.extend([spec.name] * len(taken))
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
            "source_seq_len": refs[0].seq_len if refs else None,
            "target_sequences": target,
            "n_sequences": len(taken),
            "available_shards": len(refs),
            "used_refs": used_refs,
            "used_files": used_files,
        })

    # Shuffle sequences and source_labels together using the same permutation.
    # Shuffling a Python list with rng.shuffle uses Fisher-Yates keyed only on
    # the list length and the current RNG state, not on element values — so
    # zip-shuffling [(seq, label), ...] produces the identical permutation as
    # the previous bare rng.shuffle(sequences) call, preserving the digest.
    tagged = list(zip(sequences, source_labels))
    rng.shuffle(tagged)
    sequences = [s for s, _ in tagged]
    source_labels = [lb for _, lb in tagged]
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
        "seq_len": base.effective_seq_len(req),
        "base_seq_len": req.seq_len,
        "seq_len_multiplier": req.seq_len_multiplier,
        "seed": req.seed,
        "dataset_seed": seed_value,
        "seed_material": base.dataset_seed_material(req),
        "block_hash": req.block_hash,
        "hotkey": req.hotkey,
        "digest": digest,
        "source": "multi_source_npy",
        "source_mix_policy": req.source_mix_policy,
        # Which mixture produced this sample. Recorded in the verdict so an
        # accept/reject can be replayed against the exact source list.
        "bundle": dict(RESOLVED_BUNDLE),
        "sources": source_meta,
        # Private key: parallel list of source names for each sequence in the
        # shuffled order. Consumed by eval_server to compute per-source scores;
        # popped before writing to disk so it never appears in verdict JSON.
        "_source_labels": source_labels,
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
