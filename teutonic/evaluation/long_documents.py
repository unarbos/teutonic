"""Frozen configuration for the additional complete-document evaluation sample."""

import math
import re
from collections import Counter
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .categories import DEFAULT_RULES_PATH, category_of, load_rules

LONG_DOCUMENT_VERSION = "long-documents-2k-8k-v1"
DEFAULT_INDEX_BASE = "https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev/doc_index"
DEFAULT_LONG_DOCUMENT_TOKENS = 6_000_000
MAX_DOCUMENT_TOKENS = 8192
LENGTH_BUCKETS = (("2049-4096", 2049, 4096), ("4097-8192", 4097, 8192))
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def category_mix(manifests):
    """MAIN uses manifest shard counts; splits use their frozen category weights."""
    rules = load_rules(DEFAULT_RULES_PATH)
    result = []
    for snapshot in manifests:
        manifest = snapshot.manifest
        explicit = manifest.get("category_weights")
        if explicit is not None:
            datasets = manifest.get("category_sources", {})
            if set(datasets) != set(explicit):
                raise ValueError(
                    "long-document sampling requires category_sources in split manifests"
                )
            weights = explicit
        else:
            counts = Counter(
                category_of(
                    rules.get(snapshot.name),
                    str(
                        shard.get("source_file")
                        or next(
                            shard[k]
                            for k in ("url", "href", "uri", "key", "path", "name")
                            if shard.get(k)
                        )
                    ),
                )
                for shard in manifest["shards"]
            )
            weights = {k: n / sum(counts.values()) for k, n in counts.items()}
            datasets = {k: snapshot.name for k in weights}
        for category, weight in sorted(weights.items()):
            if not _NAME.fullmatch(category) or not _NAME.fullmatch(datasets[category]):
                raise ValueError("cannot map manifest category to a document index")
            result.append(
                {
                    "source": snapshot.name,
                    "dataset": datasets[category],
                    "category": category,
                    "weight": snapshot.proportion * weight,
                }
            )
    return result


def describe_index_file(url, opener=urlopen):
    try:
        with opener(
            Request(url, method="HEAD", headers={"User-Agent": "teutonic-eval/1.0"}), timeout=120
        ) as response:
            etag = response.headers.get("ETag", "")
            size = int(response.headers.get("Content-Length", "0"))
    except OSError as exc:
        raise RuntimeError(f"cannot pin document index file {url}: {exc}") from exc
    descriptor = {"url": url, "etag": etag, "size_bytes": size}
    validate_index_file(descriptor)
    return descriptor


def _https(url):
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and not parsed.username
        and not parsed.password
    )


def validate_index_file(value):
    if not isinstance(value, dict) or set(value) not in ({"url", "etag", "size_bytes"}, {"url", "etag", "size_bytes", "sha256"}):
        raise ValueError("index file needs URL, ETag and size_bytes")
    if not isinstance(value["url"], str) or not _https(value["url"]):
        raise ValueError("index file URL must be HTTPS")
    if "sha256" in value and not _SHA.fullmatch(value["sha256"]):
        raise ValueError("invalid index file SHA-256")
    tag = value["etag"]
    if not isinstance(tag, str) or not re.fullmatch(r'"[0-9a-fA-F-]+"', tag):
        raise ValueError("index file requires a strong R2 ETag")
    if type(value["size_bytes"]) is not int or value["size_bytes"] <= 0:
        raise ValueError("index file size must be positive")


def validate_long_documents(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "version",
        "token_budget",
        "max_document_tokens",
        "manifests",
        "categories",
        "index_manifest",
    }:
        raise ValueError("invalid long-document configuration fields")
    if value["version"] != LONG_DOCUMENT_VERSION:
        raise ValueError("unsupported long-document sampling version")
    if type(value["token_budget"]) is not int or not 1 <= value["token_budget"] <= 100_000_000:
        raise ValueError("long-document token budget must be in [1, 100000000]")
    cap = value["max_document_tokens"]
    if type(cap) is not int or cap != MAX_DOCUMENT_TOKENS:
        raise ValueError("long-document maximum must be 8192 for the 2k-8k sampling policy")
    from .index_manifest import inventory_files, validate_index_manifest_pin

    validate_index_manifest_pin(value["index_manifest"])
    manifests = value["manifests"]
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("long documents require pinned source manifests")
    names = set()
    for manifest in manifests:
        if set(manifest) != {"name", "url", "sha256"} or not _NAME.fullmatch(manifest["name"]):
            raise ValueError("invalid long-document manifest descriptor")
        if (
            manifest["name"] in names
            or not _https(manifest["url"])
            or not _SHA.fullmatch(manifest["sha256"])
        ):
            raise ValueError("invalid or duplicate long-document source manifest")
        names.add(manifest["name"])
    categories = value["categories"]
    if not isinstance(categories, list) or not categories:
        raise ValueError("long documents require categories")
    identities = set()
    for item in categories:
        if set(item) != {"source", "dataset", "category", "weight", "files"}:
            raise ValueError("invalid long-document category fields")
        if item["source"] not in names or any(
            not _NAME.fullmatch(item[k]) for k in ("dataset", "category")
        ):
            raise ValueError("invalid long-document category identity")
        identity = (item["dataset"], item["category"])
        if identity in identities:
            raise ValueError("duplicate long-document category")
        identities.add(identity)
        w = item["weight"]
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0:
            raise ValueError("long-document category weight must be positive")
        if set(item["files"]) != {"tokens", "frag_ptr", "fragments", "shard_names"}:
            raise ValueError("long-document index files are incomplete")
        if item["files"] != inventory_files(value["index_manifest"], item["dataset"], item["category"]):
            raise ValueError("category index files differ from pinned index manifest")
    if not math.isclose(sum(c["weight"] for c in categories), 1.0, abs_tol=1e-9, rel_tol=0):
        raise ValueError("long-document category weights must sum to one")
    return value


def build_long_document_config(
    manifests,
    *,
    token_budget=DEFAULT_LONG_DOCUMENT_TOKENS,
    max_document_tokens=MAX_DOCUMENT_TOKENS,
    index_manifest=None,
    index_manifest_url=None,
):
    from .index_manifest import (
        DEFAULT_INDEX_MANIFEST_URL,
        fetch_index_manifest,
        inventory_files,
        validate_index_manifest_pin,
    )

    pin = index_manifest if index_manifest is not None else fetch_index_manifest(index_manifest_url or DEFAULT_INDEX_MANIFEST_URL)
    inventory = validate_index_manifest_pin(pin)
    categories = category_mix(manifests)
    for snapshot in manifests:
        # MAIN sources must match the tokenization manifests linked by the index.
        # Split sources are separately pinned; reconstruction verifies every shard
        # against those frozen split manifests.
        if snapshot.name in inventory["datasets"]:
            source = inventory["datasets"][snapshot.name]["source_manifest"]
            if source != {"url": snapshot.manifest_url, "sha256": snapshot.manifest_sha256}:
                raise ValueError("index source manifest differs from evaluation source")
    for item in categories:
        item["files"] = inventory_files(pin, item["dataset"], item["category"])
    return validate_long_documents(
        {
            "version": LONG_DOCUMENT_VERSION,
            "token_budget": token_budget,
            "max_document_tokens": max_document_tokens,
            "index_manifest": pin,
            "categories": categories,
            "manifests": [
                {"name": s.name, "url": s.manifest_url, "sha256": s.manifest_sha256}
                for s in manifests
            ],
        }
    )
