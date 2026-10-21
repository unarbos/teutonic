"""Compile explicit category lists into immutable, flat evaluation manifests."""

import hashlib
import json
import math
from pathlib import Path

import chain_config

from .categories import DEFAULT_RULES_PATH, category_of, load_rules
from .configuration import (
    DatasetManifestSnapshot,
    _shard_url,
    canonical_manifest_bytes,
    fetch_dataset_manifest,
    validate_dataset_manifest,
)

MIX_FORMAT = "teutonic-category-mix-v1"
_RULES = load_rules(DEFAULT_RULES_PATH)


def validate_definitions():
    from teutonic.competitions import ACTIVE_COMPETITIONS

    if set(chain_config.COMPETITION_DEFAULTS) != set(ACTIVE_COMPETITIONS):
        raise ValueError("configure MAIN and MIX01–MIX06 before activation")
    for key, spec in chain_config.COMPETITION_DEFAULTS.items():
        weights = spec["categories"]
        if not weights or set(weights) - set(chain_config.CATEGORY_REGISTRY):
            raise ValueError(f"unknown or empty category list: {key}")
        if any(not math.isfinite(w) or w <= 0 for w in weights.values()) or not math.isclose(
            sum(weights.values()), 1, abs_tol=1e-9, rel_tol=0
        ):
            raise ValueError(f"invalid category weights: {key}")
        identities = [(chain_config.CATEGORY_REGISTRY[c]["dataset"], c) for c in weights]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate category: {key}")
        if spec["n"] < 1 or not 1 <= spec["long_document_tokens"] <= 100_000_000:
            raise ValueError(f"invalid evaluation budgets: {key}")
        if spec["incentive"] != (0.25 if key == "main" else 0.125):
            raise ValueError(f"unsupported incentive allocation: {key}")


def original_category(dataset, shard):
    if dataset == "dclm-baseline-1.0":
        return dataset
    ref = shard.get("source_file") or next(
        str(shard[k]) for k in ("url", "href", "uri", "key", "path", "name") if shard.get(k)
    )
    return category_of(_RULES.get(dataset), ref)


def build_mix_manifests(sources, definitions=None, registry=None):
    definitions = chain_config.COMPETITION_DEFAULTS if definitions is None else definitions
    registry = chain_config.CATEGORY_REGISTRY if registry is None else registry
    sources = {s.name: s for s in sources}
    grouped = {}
    for name, snapshot in sources.items():
        for shard in snapshot.manifest["shards"]:
            cat = original_category(name, shard)
            if cat not in registry or registry[cat]["dataset"] != name:
                raise ValueError(f"unregistered source category: {name}/{cat}")
            if shard["size_bytes"] <= shard["n_tokens"] * 4:
                raise ValueError(f"invalid NPY token count or size in {name}/{cat}")
            ref = next(
                str(shard[k]) for k in ("url", "href", "uri", "key", "path", "name") if shard.get(k)
            )
            grouped.setdefault(cat, []).append(
                {
                    "url": _shard_url(snapshot.manifest_url, ref),
                    "dataset": name,
                    "category": cat,
                    **{k: shard[k] for k in ("sha256", "size_bytes", "n_tokens")},
                }
            )
    outputs = {}
    for key, spec in definitions.items():
        weights = dict(spec["categories"])
        if not weights or set(weights) - set(grouped):
            raise ValueError(f"missing category inventory for {key}")
        names = sorted({registry[c]["dataset"] for c in weights})
        manifest = {
            "format": MIX_FORMAT,
            "competition": key,
            "category_weights": weights,
            "category_inventory": {
                c: {
                    "total_tokens": sum(s["n_tokens"] for s in grouped[c]),
                    "total_shards": len(grouped[c]),
                }
                for c in weights
            },
            "category_sources": {c: registry[c]["dataset"] for c in weights},
            "source_manifests": [
                {"name": n, "url": sources[n].manifest_url, "sha256": sources[n].manifest_sha256}
                for n in names
            ],
            "shards": [s for cat in sorted(weights) for s in grouped[cat]],
        }
        for field in ("tokenizer", "dtype", "seq_len", "tokenization_mode"):
            values = {sources[n].manifest.get(field) for n in names} - {None}
            if len(values) > 1:
                raise ValueError(f"incompatible source {field}")
            if values:
                manifest[field] = values.pop()
        manifest["total_tokens"] = sum(s["n_tokens"] for s in manifest["shards"])
        manifest["total_shards"] = len(manifest["shards"])
        validate_dataset_manifest(manifest)
        outputs[key] = manifest
    return outputs


def load_bundle_snapshot(directory, key):
    directory = Path(directory)
    entry = json.loads((directory / "bundle.json").read_text())["competitions"][key]
    # Only read the conventional local name, never a path supplied by a bundle.
    manifest = json.loads((directory / f"{key}.json").read_text())
    return DatasetManifestSnapshot(key, entry["url"], entry["sha256"], 1.0, manifest)


def competition_snapshot(key, fetch=fetch_dataset_manifest, bundle_dir=None):
    if bundle_dir is not None:
        snapshot = load_bundle_snapshot(bundle_dir, key)
    else:
        snapshot = fetch(
            name=key,
            manifest_url=chain_config.COMPETITION_DEFAULTS[key]["manifest_url"],
            proportion=1.0,
        )
    manifest = snapshot.manifest
    if manifest.get("format") != MIX_FORMAT or manifest.get("competition") != key:
        raise ValueError(f"expected category manifest for {key}")
    if manifest["category_weights"] != chain_config.COMPETITION_DEFAULTS[key]["categories"]:
        raise ValueError(f"published category weights differ from chain.toml for {key}")
    if manifest["category_sources"] != {
        c: chain_config.CATEGORY_REGISTRY[c]["dataset"] for c in manifest["category_weights"]
    }:
        raise ValueError(f"published category identities differ for {key}")
    return snapshot


def version_manifest_url(key, digest):
    configured = chain_config.COMPETITION_DEFAULTS[key]["manifest_url"]
    base = (
        configured.split("/versions/", 1)[0]
        if "/versions/" in configured
        else configured.rsplit("/", 1)[0]
    )
    return base + f"/versions/{digest}.json"


def write_bundle(outputs, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    for key, manifest in outputs.items():
        body = canonical_manifest_bytes(manifest)
        digest = hashlib.sha256(body).hexdigest()
        url = version_manifest_url(key, digest)
        (directory / f"{key}.json").write_bytes(body)
        entries[key] = {"url": url, "sha256": digest}
    (directory / "bundle.json").write_text(json.dumps({"competitions": entries}, indent=2) + "\n")
    return entries
