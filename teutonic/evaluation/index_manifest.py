"""Inventory manifests for document indexes, pinned by the exact published bytes."""

import hashlib
import json
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .long_documents import _NAME, _SHA, DEFAULT_INDEX_BASE, _https, validate_index_file

DEFAULT_INDEX_MANIFEST_URL = DEFAULT_INDEX_BASE + "/manifest.json"
INDEX_FORMAT = "teutonic-document-index-v1"
TOKEN_FORMAT = {
    "eos_token_id": 151645,
    "length_includes_eos": True,
    "shard_row_tokens": 2048,
    "tokens_dtype": "<i4",
    "frag_ptr_dtype": [["frag_offset", "<i8"], ["n_fragments", "<u2"]],
    "fragments_dtype": [["shard_id", "<i4"], ["seq_index", "<u2"], ["seq_offset", "<u2"], ["frag_len", "<u2"]],
}


def index_paths(dataset, category):
    return {
        "tokens": f"{dataset}/{category}.tokens.npy",
        "frag_ptr": f"{dataset}/{category}.frag_ptr.npy",
        "fragments": f"{dataset}/{category}.fragments.bin",
        "shard_names": f"{dataset}/shard_names.json",
    }


def validate_inventory(manifest, manifest_url):
    if not isinstance(manifest, dict) or set(manifest) != {"format", "token_format", "datasets", "files"}:
        raise ValueError("invalid document-index manifest fields")
    if manifest["format"] != INDEX_FORMAT or manifest["token_format"] != TOKEN_FORMAT:
        raise ValueError("unsupported document-index binary/token format")
    datasets, files = manifest["datasets"], manifest["files"]
    if not isinstance(datasets, dict) or not datasets or not isinstance(files, dict):
        raise ValueError("document-index manifest requires datasets and files")
    expected = set()
    for dataset, value in datasets.items():
        if not _NAME.fullmatch(dataset) or set(value) != {"categories", "source_manifest"}:
            raise ValueError("invalid index dataset")
        categories = value["categories"]
        if not isinstance(categories, list) or not categories or len(set(categories)) != len(categories) or any(not _NAME.fullmatch(c) for c in categories):
            raise ValueError("invalid index categories")
        source = value["source_manifest"]
        if set(source) != {"url", "sha256"} or not _https(source["url"]) or not _SHA.fullmatch(source["sha256"]):
            raise ValueError("invalid index source manifest")
        for category in categories:
            expected.update(index_paths(dataset, category).values())
    if set(files) != expected:
        raise ValueError("document-index inventory has missing or unexpected files")
    for path, value in files.items():
        if set(value) != {"size_bytes", "etag", "sha256"} or not _SHA.fullmatch(value["sha256"]):
            raise ValueError("index inventory file requires SHA-256, ETag and size")
        validate_index_file({"url": urljoin(manifest_url, path), **value})
    return manifest


def pin_index_manifest(raw, url=DEFAULT_INDEX_MANIFEST_URL, expected_sha256=None):
    if not _https(url):
        raise ValueError("document-index manifest URL must be HTTPS")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("document-index manifest SHA-256 mismatch")
    content = raw.decode("utf-8")
    validate_inventory(json.loads(content), url)
    return {"url": url, "sha256": digest, "content": content}


def validate_index_manifest_pin(pin):
    if not isinstance(pin, dict) or set(pin) != {"url", "sha256", "content"} or not isinstance(pin["content"], str) or not isinstance(pin["sha256"], str) or not _SHA.fullmatch(pin["sha256"]):
        raise ValueError("index manifest pin requires URL, SHA-256 and exact content")
    pin_index_manifest(pin["content"].encode("utf-8"), pin["url"], pin["sha256"])
    return json.loads(pin["content"])


def fetch_index_manifest(url=DEFAULT_INDEX_MANIFEST_URL, expected_sha256=None, opener=urlopen):
    with opener(Request(url, headers={"User-Agent": "teutonic-eval/1.0"}), timeout=120) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("document-index manifest exceeds 4 MiB")
    return pin_index_manifest(raw, url, expected_sha256)


def inventory_files(pin, dataset, category):
    manifest = validate_index_manifest_pin(pin)
    if dataset not in manifest["datasets"] or category not in manifest["datasets"][dataset]["categories"]:
        raise ValueError(f"index manifest does not cover {dataset}/{category}")
    return {key: {"url": urljoin(pin["url"], path), **manifest["files"][path]}
            for key, path in index_paths(dataset, category).items()}
