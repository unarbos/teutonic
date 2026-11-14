"""Whole-document sampling, independent of shard sizes and fetch completion order."""

import bisect
import copy
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from teutonic.evaluation.categories import apportion, category_seed
from teutonic.evaluation.packed_text import BUCKETS, CATALOG_FORMAT, digest, validate_mix


def sample_documents(dataset, catalog, mix, seed, *, candidate_batch_size=64, index_workers=64):
    if candidate_batch_size < 1 or not 1 <= index_workers <= 64:
        raise ValueError("invalid sampling read bounds")
    validate_mix(mix)
    if mix["dataset"] != dataset.pin or catalog.get("dataset") != dataset.pin:
        raise ValueError("dataset/catalog/mix pins differ")
    if catalog.get("format") != CATALOG_FORMAT or catalog.get("buckets") != [list(x) for x in BUCKETS]:
        raise ValueError("unsupported sampling catalog")
    intervals = {}
    seen = set()
    for row in catalog["partitions"]:
        identity = (row["dataset"], row["category"], row["bucket"])
        key = (*identity, row["path"])
        obj = dataset.objects.get(row["path"], {})
        if key in seen or obj.get("role") != "index":
            raise ValueError("duplicate or invalid catalog partition")
        if (obj["dataset"], obj["category"]) != identity[:2]:
            raise ValueError("catalog category mismatch")
        if any(type(row[x]) is not int for x in ("lo", "hi")) or not 0 <= row["lo"] < row["hi"] <= obj["document_count"]:
            raise ValueError("catalog row range out of bounds")
        seen.add(key)
        intervals.setdefault(identity, []).append(row)
    categories = sorted(mix["categories"], key=lambda c: (c["dataset"], c["category"]))
    quotas = apportion(mix["token_budget"], len(categories), [c["weight"] for c in categories])
    def sample_category(item):
        category, quota = item
        selected, summaries, document_ids = [], [], set()
        rng = np.random.default_rng(category_seed(seed, f"documents|{category['dataset']}|{category['category']}"))
        targets = apportion(quota, len(BUCKETS), [b[3] for b in BUCKETS])
        for (name, minimum, maximum, _weight), target in zip(BUCKETS, targets, strict=True):
            rows = sorted(intervals.get((category["dataset"], category["category"], name), []), key=lambda x: x["path"])
            ends, population = [], 0
            for row in rows:
                population += row["hi"] - row["lo"]
                ends.append(population)
            if not population:
                raise ValueError(f"empty required bucket: {category}/{name}")
            used, tokens, count = set(), 0, 0
            while count == 0 or tokens < target:
                if len(used) == population:
                    raise ValueError("not enough documents to meet token quota")
                # Look ahead without changing consumed RNG draws. The checkpoint
                # after the final considered candidate seeds the next length bucket.
                # Keep lookahead small: sparse MIX categories can finish after a
                # handful of rows scattered across a very large partition set.
                references, states = [], []
                batch_size = min(candidate_batch_size, population - len(used),
                                 max(1, (target - tokens) // minimum + 1))
                while len(references) < batch_size:
                    ordinal = int(rng.integers(population))
                    if ordinal in used:
                        continue
                    used.add(ordinal)
                    part = bisect.bisect_right(ends, ordinal)
                    before = ends[part - 1] if part else 0
                    row = rows[part]
                    references.append((row["path"], row["lo"] + ordinal - before))
                    states.append(copy.deepcopy(rng.bit_generator.state))
                finished = False
                for doc, state in zip(dataset.documents(references, reads), states, strict=True):
                    length = doc.original_mimo_token_count
                    if not minimum <= length <= maximum:
                        raise ValueError("catalog does not match index length bucket")
                    if not doc.byte_length:
                        raise ValueError("sampled empty text cannot be scored in bits/byte")
                    if count and abs(length - (target - tokens)) >= target - tokens:
                        finished = True
                    else:
                        if doc.doc_id in document_ids:
                            raise ValueError("duplicate document ID in exported population")
                        document_ids.add(doc.doc_id)
                        selected.append(doc)
                        count += 1
                        tokens += length
                        finished = tokens >= target
                    if finished:
                        rng.bit_generator.state = state
                        break
                if finished:
                    break
            summaries.append({**category, "bucket": name, "population": population,
                              "target_tokens": target, "sampled_tokens": tokens, "documents": count})
        return selected, summaries

    # Category RNGs are independent; consume results in sorted category order.
    selected, summaries, document_ids = [], [], set()
    # One shared I/O executor bounds total requests even across many categories.
    with ThreadPoolExecutor(max_workers=index_workers) as reads, \
            ThreadPoolExecutor(max_workers=min(8, len(categories))) as pool:
        for docs, stats in pool.map(sample_category, zip(categories, quotas, strict=True)):
            for doc in docs:
                if doc.doc_id in document_ids:
                    raise ValueError("duplicate document ID across categories")
                document_ids.add(doc.doc_id)
            selected.extend(docs)
            summaries.extend(stats)
    random.Random(category_seed(seed, "packed-document-order")).shuffle(selected)
    return selected, {"digest": digest([asdict(d) for d in selected]), "categories": summaries,
                      "n_documents": len(selected), "input_tokens": sum(d.original_mimo_token_count for d in selected)}
