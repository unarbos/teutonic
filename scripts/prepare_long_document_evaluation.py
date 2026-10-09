#!/usr/bin/env python3
"""Read public manifests/indexes and prepare a long-document config; no DB or GPU writes."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chain_config
from teutonic.evaluation.categories import apportion, category_seed
from teutonic.evaluation.configuration import fetch_dataset_manifest
from teutonic.evaluation.index_manifest import (
    DEFAULT_INDEX_MANIFEST_URL,
    fetch_index_manifest,
    pin_index_manifest,
)
from teutonic.evaluation.long_documents import (
    DEFAULT_LONG_DOCUMENT_TOKENS,
    build_long_document_config,
)
from teutonic.evaluator.document_index import DocumentIndex, sample_category


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", choices=("main", "math", "code", "text"), default="main")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_LONG_DOCUMENT_TOKENS)
    parser.add_argument("--index-manifest-url", default=DEFAULT_INDEX_MANIFEST_URL)
    parser.add_argument("--index-manifest-file", type=Path)
    parser.add_argument("--index-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan-context-limit",
        type=int,
        help="also plan document IDs, without downloading token shards",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--index-cache", type=Path, default=Path("/tmp/teutonic-doc-index"))
    args = parser.parse_args()
    if args.competition == "main":
        definitions = chain_config.EVALUATION_DATASETS
    else:
        definitions = [
            {
                "name": k,
                "manifest_url": v["manifest_url"],
                "proportion": 0.7 if k == args.competition else 0.15,
            }
            for k, v in chain_config.SPLIT_DEFAULTS.items()
        ]
    snapshots = tuple(fetch_dataset_manifest(**d) for d in definitions)
    config = build_long_document_config(
        snapshots, token_budget=args.token_budget,
        index_manifest=(pin_index_manifest(args.index_manifest_file.read_bytes(), args.index_manifest_url, args.index_manifest_sha256)
                        if args.index_manifest_file else fetch_index_manifest(args.index_manifest_url, args.index_manifest_sha256))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Pinned {len(config['categories'])} categories to {args.output}", flush=True)
    if args.plan_context_limit:
        categories = config["categories"]
        quotas = apportion(args.token_budget, len(categories), [c["weight"] for c in categories])

        def plan(entry):
            c, quota = entry
            idx = DocumentIndex(c, args.index_cache)
            rows, summary = sample_category(
                idx,
                token_budget=quota,
                max_tokens=min(args.plan_context_limit, config["max_document_tokens"]),
                seed=category_seed(
                    args.seed, f"long-documents|{c['source']}|{c['dataset']}|{c['category']}"
                ),
            )
            result = {
                "source": c["source"],
                "dataset": c["dataset"],
                "category": c["category"],
                **summary,
                "max_sample_length": max(d["length"] for d in rows),
                "documents": rows,
            }
            print(
                json.dumps({k: v for k, v in result.items() if k not in {"documents", "buckets"}}),
                flush=True,
            )
            return result

        with ThreadPoolExecutor(max_workers=8) as pool:
            plans = list(pool.map(plan, zip(categories, quotas, strict=True)))
        report = {
            "competition": args.competition,
            "seed": args.seed,
            "context_limit": args.plan_context_limit,
            "max_document_tokens": min(args.plan_context_limit, config["max_document_tokens"]),
            "input_tokens": sum(p["sampled_tokens"] for p in plans),
            "n_documents": sum(p["n_documents"] for p in plans),
            "categories": plans,
        }
        output = args.output.with_suffix(".plan.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "categories"}), flush=True)


if __name__ == "__main__":
    main()
