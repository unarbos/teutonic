"""Seeded long-document sampling and reconstruction for every competition."""

import hashlib
import json
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from teutonic.evaluation.categories import apportion, category_seed
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    _shard_url,
    canonical_manifest_bytes,
    fetch_dataset_manifest,
)
from teutonic.evaluation.long_documents import (
    MAX_DOCUMENT_TOKENS,
    category_mix,
    validate_long_documents,
)
from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID
from teutonic.evaluator.document_index import DocumentIndex, sample_category


def sequence_digest(sequences):
    """Length-delimited digest, unambiguous for variable-length samples."""
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(len(sequence).to_bytes(8, "little"))
        digest.update(np.asarray(sequence, dtype="<i8").tobytes())
    return digest.hexdigest()


def load_pinned_manifest(descriptor, proportion, cache_dir):
    cache = Path(cache_dir) / "manifests"
    target = cache / f"{descriptor['sha256']}.json"
    if target.exists():
        manifest = json.loads(target.read_text())
        return DatasetManifestSnapshot(
            descriptor["name"], descriptor["url"], descriptor["sha256"], proportion, manifest
        )
    snapshot = fetch_dataset_manifest(
        name=descriptor["name"], manifest_url=descriptor["url"], proportion=proportion
    )
    if snapshot.manifest_sha256 != descriptor["sha256"]:
        raise ValueError("long-document source manifest changed since configuration was pinned")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
        tmp.write(canonical_manifest_bytes(snapshot.manifest))
        path = Path(tmp.name)
    try:
        path.replace(target)
    finally:
        path.unlink(missing_ok=True)
    return snapshot


def extend_with_long_documents(
    req,
    sequences,
    metadata,
    on_phase=None,
    *,
    index_factory=DocumentIndex,
    manifest_loader=load_pinned_manifest,
    shard_materializer=None,
):
    from teutonic.evaluator import sources

    config = validate_long_documents(req.long_documents)
    if config is None:
        return sequences, metadata
    if req.max_document_tokens < MAX_DOCUMENT_TOKENS:
        raise ValueError(
            "long-document sampling requires a checkpoint context limit of at least 8192"
        )
    cap = config["max_document_tokens"]
    cache = Path(
        os.environ.get("TEUTONIC_DOC_INDEX_CACHE_DIR", sources.base.SHARD_CACHE_DIR / "doc_index")
    )
    proportions = {source["name"]: source["proportion"] for source in req.dataset_sources}
    snapshots = [manifest_loader(d, proportions[d["name"]], cache) for d in config["manifests"]]
    declared_mix = [{k: v for k, v in c.items() if k != "files"} for c in config["categories"]]
    if declared_mix != category_mix(snapshots):
        raise ValueError("long-document category mix differs from the pinned manifests")
    refs = {}
    for snapshot in snapshots:
        for shard in snapshot.manifest["shards"]:
            reference = next(
                str(shard[k]) for k in ("url", "href", "uri", "key", "path", "name") if shard.get(k)
            )
            url = _shard_url(snapshot.manifest_url, reference)
            dataset = shard.get("dataset", snapshot.name)
            # Index entries use original dataset keys, including for specialist manifests.
            path = unquote(urlparse(url).path)
            marker = f"/{dataset}/"
            if marker not in path:
                raise ValueError(f"shard URL does not identify original dataset {dataset}")
            key = path.split(marker, 1)[1]
            identity = (dataset, key)
            ref = sources.ShardRef(
                snapshot.name, url, shard["sha256"], shard["size_bytes"], shard["n_tokens"]
            )
            if identity in refs and refs[identity] != ref:
                raise ValueError("conflicting source shard descriptors")
            refs[identity] = ref
    quotas = apportion(
        config["token_budget"],
        len(config["categories"]),
        [c["weight"] for c in config["categories"]],
    )
    seed = metadata["dataset_seed"]

    def plan(entry):
        category, quota = entry
        index = index_factory(category, cache)
        selected, summary = sample_category(
            index,
            token_budget=quota,
            max_tokens=cap,
            seed=category_seed(
                seed,
                f"long-documents|{category['source']}|{category['dataset']}|{category['category']}",
            ),
        )
        for document in selected:
            document.update(
                source=category["source"],
                dataset=category["dataset"],
                category=category["category"],
            )
            for fragment in document["fragments"]:
                key = index.shard_names[fragment["shard_id"]].removeprefix(
                    f"{category['dataset']}/"
                )
                if (category["dataset"], key) not in refs:
                    raise ValueError(
                        "document index references a shard absent from the frozen manifest"
                    )
                fragment["shard_key"] = key
        return selected, {**declared_mix[config["categories"].index(category)], **summary}

    with ThreadPoolExecutor(max_workers=8) as pool:
        plans = list(pool.map(plan, zip(config["categories"], quotas, strict=True)))
    documents = [d for selected, _summary in plans for d in selected]
    needed = sorted({(d["dataset"], f["shard_key"]) for d in documents for f in d["fragments"]})
    if on_phase:
        on_phase(
            {
                "phase": "long_documents_planned",
                "documents": len(documents),
                "input_tokens": sum(d["length"] for d in documents),
                "unique_shards": len(needed),
                "referenced_shard_bytes": sum(refs[k].size_bytes for k in needed),
            }
        )
    materialize = shard_materializer or sources.materialize_shard
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = dict(zip(needed, pool.map(lambda k: materialize(refs[k]), needed), strict=True))

    @lru_cache(maxsize=32)
    def open_shard(identity):
        array = np.load(paths[identity], mmap_mode="r", allow_pickle=False)
        if (
            array.dtype != np.dtype("<u4")
            or array.size != refs[identity].n_tokens
            or array.size % 2048
        ):
            raise ValueError("indexed shard has invalid dtype or token count")
        if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[1] != 2048):
            raise ValueError("indexed shard has invalid dimensions")
        return array.reshape(-1, 2048)

    labels = list(metadata["_source_labels"])
    provenance = [{**p, "component": "windows"} for p in metadata["_sample_provenance"]]
    all_sequences = list(sequences)
    source_groups = {s["name"]: i for i, s in enumerate(metadata["sources"])}
    for document in documents:
        pieces = []
        for fragment in document["fragments"]:
            array = open_shard((document["dataset"], fragment["shard_key"]))
            row, offset, length = (fragment[k] for k in ("seq_index", "seq_offset", "frag_len"))
            if row >= array.shape[0]:
                raise ValueError("document fragment row is outside its shard")
            pieces.append(array[row, offset : offset + length])
        tokens = np.concatenate(pieces)
        if len(tokens) != document["length"] or tokens[-1] != DOCUMENT_EOS_TOKEN_ID:
            raise ValueError("indexed document has wrong length or missing terminal EOS")
        if req.vocab_size <= 0 or int(tokens.max()) >= req.vocab_size:
            raise ValueError("indexed document contains out-of-vocabulary tokens")
        all_sequences.append(tokens.tolist())
        labels.append(document["source"])
        provenance.append(
            {
                **document,
                "component": "long_documents",
                "shard_group_index": source_groups[document["source"]],
                "shard_index": -1,
                "shard_sequence_index": -1,
            }
        )
    tagged = list(zip(all_sequences, labels, provenance, strict=True))
    random.Random(category_seed(seed, "combined-window-document-order-v1")).shuffle(tagged)
    all_sequences, labels, provenance = map(list, zip(*tagged, strict=True))
    long_meta = {
        "version": config["version"],
        "index_manifest": config["index_manifest"],
        "target_tokens": config["token_budget"],
        "input_tokens": sum(d["length"] for d in documents),
        "n_documents": len(documents),
        "context_limit": cap,
        "categories": [summary for _selected, summary in plans],
        "index_files": [
            {"dataset": c["dataset"], "category": c["category"], "files": c["files"]}
            for c in config["categories"]
        ],
        "used_shards": [
            {
                "dataset": dataset,
                "key": key,
                "url": refs[(dataset, key)].url,
                "sha256": refs[(dataset, key)].sha256,
            }
            for dataset, key in needed
        ],
        "early_stopping": "disabled: complete the full mixed-length sample",
    }
    if on_phase:
        on_phase(
            {
                "phase": "long_documents_sampled",
                "documents": len(documents),
                "input_tokens": long_meta["input_tokens"],
            }
        )
    return all_sequences, {
        **metadata,
        "n": len(all_sequences),
        "window_digest": metadata["digest"],
        "digest": sequence_digest(all_sequences),
        "digest_format": "length-prefixed-int64-le-v1",
        "long_documents": long_meta,
        "_source_labels": labels,
        "_sample_provenance": provenance,
    }
