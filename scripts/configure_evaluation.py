#!/usr/bin/env python3
from __future__ import annotations

import logging
import os

import psycopg
from psycopg.rows import dict_row

import chain_config
from teutonic.evaluation.configuration import (
    fetch_dataset_manifest,
    store_evaluation_configuration,
)


log = logging.getLogger("teutonic.evaluation-config")


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    snapshots = tuple(
        fetch_dataset_manifest(
            name=str(item["name"]),
            manifest_url=str(item["manifest_url"]),
            proportion=float(item["proportion"]),
        )
        for item in chain_config.EVALUATION_DATASETS
    )
    with psycopg.connect(required("TEUTONIC_DATABASE_URL"), row_factory=dict_row) as connection:
        stored = store_evaluation_configuration(
            connection,
            netuid=int(required("TEUTONIC_NETUID")),
            chain_generation=required("TEUTONIC_CHAIN_GENERATION"),
            competition=required("TEUTONIC_COMPETITION"),
            dataset_label=chain_config.EVALUATION_DATASET_LABEL,
            n=chain_config.EVALUATION_N,
            delta_threshold=chain_config.EVALUATION_DELTA_THRESHOLD,
            manifests=snapshots,
            shards_per_dataset=chain_config.EVALUATION_SHARDS_PER_DATASET,
        )
    log.info(
        "%s evaluation config=%s n=%s delta=%s datasets=%s",
        "created" if stored.created else "activated existing",
        stored.config_version,
        chain_config.EVALUATION_N,
        chain_config.EVALUATION_DELTA_THRESHOLD,
        ",".join(item.name for item in snapshots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
