#!/usr/bin/env python3
"""Explicit administration of versioned PostgreSQL evaluation configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

import chain_config
from teutonic.competitions import (
    ACTIVE_COMPETITIONS,
    BOOTSTRAP_COMPETITIONS,
    CONFIGURED_COMPETITIONS,
    normalize_competition,
)
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    EvaluationSettings,
    evaluation_config_version,
    fetch_dataset_manifest,
    pretokenized_dataset_request,
    store_evaluation_configuration,
)
from teutonic.evaluation.index_manifest import (
    DEFAULT_INDEX_MANIFEST_URL,
    fetch_index_manifest,
    pin_index_manifest,
)
from teutonic.evaluation.long_documents import (
    LONG_DOCUMENT_VERSION,
    MAX_DOCUMENT_TOKENS,
    build_long_document_config,
)
from teutonic.evaluation.mixes import competition_snapshot, validate_definitions
from teutonic.weights.transition import prepare_mix07_replacement, snapshot_transition

COMPETITIONS = CONFIGURED_COMPETITIONS
LEGACY_SPLITS = ("math", "code", "text")


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--competition", type=normalize_competition, choices=COMPETITIONS, default="main")
    group.add_argument(
        "--all", action="store_true", help="atomically configure all active competition definitions"
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="allow missing configurations to use chain.toml defaults",
    )
    parser.add_argument("--replace-definition", action="store_true", help="replace categories and policy defaults from chain.toml")
    parser.add_argument("--bundle-dir", type=Path, help="prepared immutable manifests from prepare_competition_mixes.py")
    parser.add_argument("--cutover", action="store_true", help="initial legacy migration: snapshot rewards and activate MAIN/MIX01–06")
    parser.add_argument(
        "--replace-mix04", action="store_true",
        help="activate MIX07 from MIX05's king; retire MIX04 while retaining its reward slot until MIX07 wins",
    )
    parser.add_argument("--delta-threshold", type=float)
    parser.add_argument("--n", type=int, help="window sample count; 0 requires document evaluation")
    parser.add_argument("--shards-per-dataset", type=int)
    parser.add_argument("--manifest", action="append", default=[], metavar="NAME=HTTPS_URL")
    parser.add_argument(
        "--refresh-manifests",
        action="store_true",
        help="explicitly re-fetch all configured manifest URLs",
    )
    parser.add_argument("--long-document-tokens", type=int, help="whole-document token budget; 0 disables documents")
    parser.add_argument("--doc-index-manifest-url", default=DEFAULT_INDEX_MANIFEST_URL)
    parser.add_argument("--doc-index-manifest-file", type=Path, help="use exact local manifest bytes before publication")
    parser.add_argument("--doc-index-manifest-sha256", help="require this exact manifest hash")
    parser.add_argument("--max-document-tokens", type=int, choices=(MAX_DOCUMENT_TOKENS,), default=None,
                        help=f"whole-document selection cap ({MAX_DOCUMENT_TOKENS} for the current policy)")
    parser.add_argument("--refresh-doc-index", action="store_true", help="fetch and pin the current index manifest")
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
        "long_documents": config["long_documents"],
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


def initial_configuration(key, fetch, bundle_dir=None):
    if key not in chain_config.COMPETITION_DEFAULTS:
        raise ValueError(f"no active competition definition: {key}")
    spec = chain_config.COMPETITION_DEFAULTS[key]
    return {"dataset_label": f"{key}-categories-v1", "n": spec["n"],
            "delta_threshold": spec["delta_threshold"],
            "shards_per_dataset": chain_config.EVALUATION_SHARDS_PER_DATASET,
            "manifests": (competition_snapshot(key, fetch, bundle_dir),)}


def prepare_configuration(current, key, args, fetch=None, *, index_manifest=None):
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

    replacing = getattr(args, "replace_definition", False)
    if current is None or replacing:
        if current is None and not args.initialize:
            raise ValueError(f"{key} has no active configuration; use --initialize")
        result = initial_configuration(key, download, getattr(args, "bundle_dir", None))
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
    if key in COMPETITIONS:
        snapshots = result["manifests"]
        if len(snapshots) != 1 or snapshots[0].name != key or snapshots[0].proportion != 1:
            raise ValueError("category competitions require one flat manifest")
        manifest = snapshots[0].manifest
        if manifest.get("format") != "teutonic-category-mix-v1" or manifest.get("competition") != key:
            raise ValueError(f"invalid category competition manifest: {key}")
        if current and not replacing and (
            manifest["category_weights"] != current["manifests"][0].manifest.get("category_weights")
            or manifest["category_sources"] != current["manifests"][0].manifest.get("category_sources")
        ):
            raise ValueError("changing category definitions requires --replace-definition")
    for field in ("delta_threshold", "n", "shards_per_dataset"):
        if getattr(args, field) is not None:
            result[field] = getattr(args, field)
    if key in LEGACY_SPLITS:
        snapshots = result["manifests"]
        if {s.name for s in snapshots} != set(LEGACY_SPLITS):
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
    previous_long = result.get("long_documents")
    budget = args.long_document_tokens
    if budget is None and (current is None or replacing) and key in chain_config.COMPETITION_DEFAULTS:
        budget = chain_config.COMPETITION_DEFAULTS[key]["long_document_tokens"]
    if budget is not None and budget < 0:
        raise ValueError("--long-document-tokens cannot be negative")
    if budget == 0:
        result["long_documents"] = None
    elif budget is not None or args.refresh_doc_index or args.max_document_tokens is not None or (previous_long and (args.refresh_manifests or args.manifest)):
        if budget is None and not previous_long:
            raise ValueError("enable long documents with --long-document-tokens first")
        cap = args.max_document_tokens if args.max_document_tokens is not None else MAX_DOCUMENT_TOKENS
        if previous_long and not (args.refresh_doc_index or args.refresh_manifests or args.manifest):
            # Changing a budget must not silently select a newer index snapshot.
            result["long_documents"] = {
                **deepcopy(previous_long),
                "version": LONG_DOCUMENT_VERSION,
                "token_budget": budget or previous_long["token_budget"],
                "max_document_tokens": cap,
            }
        else:
            result["long_documents"] = build_long_document_config(
                result["manifests"], token_budget=budget or previous_long["token_budget"],
                max_document_tokens=cap,
                index_manifest=(pin_index_manifest(args.doc_index_manifest_file.read_bytes(), args.doc_index_manifest_url, args.doc_index_manifest_sha256)
                                if args.doc_index_manifest_file else index_manifest
                                if index_manifest is not None and not args.refresh_doc_index
                                else fetch_index_manifest(args.doc_index_manifest_url, args.doc_index_manifest_sha256)),
            )
    settings = EvaluationSettings(config_version=evaluation_config_version(**result), **result)
    # Validates actual category coverage, rounding and shard capacity before activation.
    pretokenized_dataset_request(
        settings, block_hash="configuration-validation", hotkey="validation", seq_len=2048
    )
    return result, settings.config_version


def main(argv=None) -> int:
    args = parse_args(argv)
    keys = (BOOTSTRAP_COMPETITIONS if args.cutover else ACTIVE_COMPETITIONS) if args.all else (args.competition,)
    validate_definitions()
    if args.cutover and not (args.all and args.initialize and args.replace_definition):
        raise ValueError("--cutover requires --all --initialize --replace-definition")
    if args.replace_mix04 and (
        args.all or args.cutover or args.competition != "mix07" or not args.initialize
        or args.n != 0 or not args.long_document_tokens or args.long_document_tokens < 0
    ):
        raise ValueError("--replace-mix04 requires --competition MIX07 --initialize --n 0 --long-document-tokens BUDGET")
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
        transition = None
        if args.cutover:
            transition = snapshot_transition(connection, main)
            print(json.dumps({"reward_transition": transition, "dry_run": args.dry_run}))
        elif not main["reward_transition"]:
            raise ValueError("initialize the category competitions with --cutover")
        rotation = prepare_mix07_replacement(connection, main) if args.replace_mix04 else None
        if rotation:
            print(json.dumps({
                "replacement": "mix04->mix07",
                "starting_reign_id": str(rotation["starting_reign_id"]),
                "starting_source": "mix05", "dry_run": args.dry_run,
                "rewards": "MIX04 retains its slot until MIX07's first crown",
            }))
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
            if key == "mix07" and row is None and rotation is None:
                raise ValueError("create MIX07 with --replace-mix04 to pin MIX05 and preserve MIX04 rewards")
            config, version = prepare_configuration(
                current, key, args,
                index_manifest=rotation["index_manifest"] if rotation else None,
            )
            print(
                json.dumps(
                    {
                        "competition": key,
                        "config_version": version,
                        "n": config["n"],
                        "delta_threshold": config["delta_threshold"],
                        "previous_delta_threshold": current["delta_threshold"] if current else None,
                        "dry_run": args.dry_run,
                        "long_document_tokens": (config.get("long_documents") or {}).get("token_budget"),
                        "category_weights": {s.name: s.manifest.get("category_weights") for s in config["manifests"]},
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
            if key.startswith("mix") and not (
                row and (row["current_reign_id"] or row["starting_reign_id"])
            ) and not rotation and main["current_reign_id"] is None:
                raise ValueError("a kingless MIX requires a MAIN king to pin its starting point")
            if args.dry_run:
                del current, config
                continue
            # Store one large inventory at a time, within the same outer transaction.
            # Any later validation/write failure still rolls the entire update back.
            if row is None:
                row = connection.execute(
                    "INSERT INTO control_plane.competitions(netuid,chain_generation,name,competition_key,main_competition_id,active) VALUES (%s,%s,%s,%s,%s,false) RETURNING competition_id",
                    (netuid, generation, name, key, main["competition_id"]),
                ).fetchone()
            if key.startswith("mix"):
                connection.execute(
                    """UPDATE control_plane.competitions SET starting_reign_id=%s
                       WHERE competition_id=%s AND starting_reign_id IS NULL
                         AND current_reign_id IS NULL""",
                    (rotation["starting_reign_id"] if rotation else main["current_reign_id"], row["competition_id"]),
                )
            connection.execute(
                "INSERT INTO control_plane.evaluation_early_stopping_policies(competition_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (row["competition_id"],),
            )
            store_evaluation_configuration(
                connection, netuid=netuid, chain_generation=generation, competition=name, **config
            )
            del current, config
        if args.dry_run:
            connection.rollback()
            return 0
        if args.cutover:
            connection.execute(
                "UPDATE control_plane.competitions SET reward_transition=%s WHERE competition_id=%s",
                (Jsonb(transition), main["competition_id"]),
            )
            connection.execute(
                """UPDATE control_plane.competitions SET active=(competition_key=ANY(%s))
                   WHERE competition_id=%s OR main_competition_id=%s""",
                (list(keys), main["competition_id"], main["competition_id"]),
            )
        if rotation:
            connection.execute(
                "UPDATE control_plane.competitions SET reward_transition=%s,updated_at=clock_timestamp() WHERE competition_id=%s",
                (Jsonb(rotation["reward_transition"]), main["competition_id"]),
            )
            connection.execute(
                "UPDATE control_plane.competitions SET active=false,updated_at=clock_timestamp() WHERE competition_id=%s",
                (rotation["retired_competition_id"],),
            )
            connection.execute(
                "UPDATE control_plane.competitions SET active=true,updated_at=clock_timestamp() WHERE competition_id=%s",
                (row["competition_id"],),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
