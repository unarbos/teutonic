#!/usr/bin/env python3
"""Explicit administration of versioned PostgreSQL evaluation configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg.rows import dict_row

import chain_config
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    EvaluationSettings,
    evaluation_config_version,
    fetch_dataset_manifest,
    pretokenized_dataset_request,
    store_evaluation_configuration,
)

COMPETITIONS = ("main", "math", "code", "text")


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--competition", choices=COMPETITIONS, default="main")
    group.add_argument(
        "--all", action="store_true", help="atomically configure all four competitions"
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="allow missing configurations to use chain.toml defaults",
    )
    parser.add_argument("--delta-threshold", type=float)
    parser.add_argument("--n", type=int)
    parser.add_argument("--shards-per-dataset", type=int)
    parser.add_argument("--manifest", action="append", default=[], metavar="NAME=HTTPS_URL")
    parser.add_argument(
        "--refresh-manifests",
        action="store_true",
        help="explicitly re-fetch all configured manifest URLs",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_active(connection, competition_id):
    config = connection.execute(
        "SELECT * FROM control_plane.evaluation_configs WHERE competition_id = %s AND active",
        (competition_id,),
    ).fetchone()
    if config is None:
        return None
    rows = connection.execute(
        "SELECT * FROM control_plane.dataset_manifests WHERE evaluation_config_id = %s ORDER BY position",
        (config["evaluation_config_id"],),
    ).fetchall()
    return {
        "dataset_label": config["dataset_label"],
        "n": config["eval_n"],
        "delta_threshold": config["delta_threshold"],
        "shards_per_dataset": config["shards_per_dataset"],
        "manifests": tuple(
            DatasetManifestSnapshot(
                name=r["name"],
                manifest_url=r["manifest_url"],
                manifest_sha256=r["manifest_sha256"],
                proportion=r["sample_proportion"],
                manifest=r["manifest_json"],
            )
            for r in rows
        ),
    }


def initial_configuration(key, fetch):
    if key == "main":
        defaults = {
            "dataset_label": chain_config.EVALUATION_DATASET_LABEL,
            "n": chain_config.EVALUATION_N,
            "delta_threshold": chain_config.EVALUATION_DELTA_THRESHOLD,
            "shards_per_dataset": chain_config.EVALUATION_SHARDS_PER_DATASET,
        }
        sources = chain_config.EVALUATION_DATASETS
    else:
        split = chain_config.SPLIT_DEFAULTS[key]
        defaults = {
            "dataset_label": f"{key}-70-15-15",
            "n": split["n"],
            "delta_threshold": split["delta_threshold"],
            "shards_per_dataset": chain_config.EVALUATION_SHARDS_PER_DATASET,
        }
        sources = tuple(
            {
                "name": k,
                "manifest_url": chain_config.SPLIT_DEFAULTS[k]["manifest_url"],
                "proportion": 0.7 if k == key else 0.15,
            }
            for k in COMPETITIONS[1:]
        )
    return {**defaults, "manifests": tuple(fetch(**item) for item in sources)}


def prepare_configuration(current, key, args, fetch=None):
    if fetch is None:
        fetch = fetch_dataset_manifest
    overrides = {}
    for item in args.manifest:
        name, separator, url = item.partition("=")
        if not separator or not name or name in overrides:
            raise ValueError("--manifest requires unique NAME=HTTPS_URL entries")
        overrides[name] = url
    cache = {}

    def download(**source):
        source["manifest_url"] = overrides.get(source["name"], source["manifest_url"])
        identity = (source["name"], source["manifest_url"])
        if identity not in cache:
            cache[identity] = fetch(**source)
        return replace(cache[identity], proportion=source["proportion"])

    if current is None:
        if not args.initialize:
            raise ValueError(f"{key} has no active configuration; use --initialize")
        result = initial_configuration(key, download)
    else:
        result = dict(current)
        result["manifests"] = tuple(
            download(name=s.name, manifest_url=s.manifest_url, proportion=s.proportion)
            if args.refresh_manifests or s.name in overrides
            else s
            for s in current["manifests"]
        )
    unknown = set(overrides) - {s.name for s in result["manifests"]}
    if unknown:
        raise ValueError(f"unknown manifest names for {key}: {sorted(unknown)}")
    for field in ("delta_threshold", "n", "shards_per_dataset"):
        if getattr(args, field) is not None:
            result[field] = getattr(args, field)
    if key != "main":
        snapshots = result["manifests"]
        if {s.name for s in snapshots} != set(COMPETITIONS[1:]):
            raise ValueError("specialist configurations require math, code, and text manifests")
        for s in snapshots:
            if s.manifest.get("format") != "teutonic-split-v1" or s.manifest.get("split") != s.name:
                raise ValueError(f"invalid specialist manifest: {s.name}")
            expected = 0.7 if s.name == key else 0.15
            if abs(s.proportion - expected) > 1e-9:
                raise ValueError("specialist proportions must be 70/15/15")
        categories = [c for s in snapshots for c in s.manifest["category_weights"]]
        if len(categories) != len(set(categories)):
            raise ValueError("a category occurs in multiple split manifests")
    settings = EvaluationSettings(config_version=evaluation_config_version(**result), **result)
    # Validates actual category coverage, rounding and shard capacity before activation.
    pretokenized_dataset_request(
        settings, block_hash="configuration-validation", hotkey="validation", seq_len=2048
    )
    return result, settings.config_version


def main(argv=None) -> int:
    args = parse_args(argv)
    keys = COMPETITIONS if args.all else (args.competition,)
    main_name = required("TEUTONIC_COMPETITION")
    netuid, generation = int(required("TEUTONIC_NETUID")), required("TEUTONIC_CHAIN_GENERATION")
    # One transaction makes --all atomic and serializes administrative updates.
    with psycopg.connect(required("TEUTONIC_DATABASE_URL"), row_factory=dict_row) as connection:
        main = connection.execute(
            "SELECT * FROM control_plane.competitions WHERE netuid=%s AND chain_generation=%s AND name=%s FOR UPDATE",
            (netuid, generation, main_name),
        ).fetchone()
        if main is None or main["competition_key"] != "main":
            raise RuntimeError("TEUTONIC_COMPETITION must identify the existing main competition")
        prepared = []
        fetched = {}

        def cached_fetch(**source):
            identity = (source["name"], source["manifest_url"])
            if identity not in fetched:
                fetched[identity] = fetch_dataset_manifest(**source)
            return replace(fetched[identity], proportion=source["proportion"])

        for key in keys:
            name = main_name if key == "main" else key
            row = connection.execute(
                "SELECT * FROM control_plane.competitions WHERE netuid=%s AND chain_generation=%s AND name=%s FOR UPDATE",
                (netuid, generation, name),
            ).fetchone()
            if (
                row
                and key != "main"
                and (
                    row["competition_key"] != key
                    or row["main_competition_id"] != main["competition_id"]
                )
            ):
                raise ValueError(f"competition name already belongs to another scope: {name}")
            current = load_active(connection, row["competition_id"]) if row else None
            config, version = prepare_configuration(current, key, args, cached_fetch)
            print(
                json.dumps(
                    {
                        "competition": key,
                        "config_version": version,
                        "n": config["n"],
                        "delta_threshold": config["delta_threshold"],
                        "previous_delta_threshold": current["delta_threshold"] if current else None,
                        "dry_run": args.dry_run,
                        "manifests": [
                            {
                                "name": s.name,
                                "url": s.manifest_url,
                                "sha256": s.manifest_sha256,
                                "proportion": s.proportion,
                            }
                            for s in config["manifests"]
                        ],
                    }
                )
            )
            prepared.append((key, name, row, config))
        if args.dry_run:
            connection.rollback()
            return 0
        for key, name, row, config in prepared:
            if row is None:
                row = connection.execute(
                    "INSERT INTO control_plane.competitions(netuid,chain_generation,name,competition_key,main_competition_id) VALUES (%s,%s,%s,%s,%s) RETURNING competition_id",
                    (netuid, generation, name, key, main["competition_id"]),
                ).fetchone()
            connection.execute(
                "INSERT INTO control_plane.evaluation_early_stopping_policies(competition_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (row["competition_id"],),
            )
            store_evaluation_configuration(
                connection, netuid=netuid, chain_generation=generation, competition=name, **config
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
