import hashlib
import io
import os
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import patch

import psycopg
import pytest
from psycopg.rows import dict_row

import chain_config
from scripts import configure_evaluation as admin
from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    canonical_manifest_bytes,
    store_evaluation_configuration,
)

DATABASE_URL = os.environ.get("TEUTONIC_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="PostgreSQL integration database required")


def cleanup(connection):
    connection.execute(
        "DELETE FROM control_plane.dataset_manifests WHERE evaluation_config_id IN (SELECT ec.evaluation_config_id FROM control_plane.evaluation_configs ec JOIN control_plane.competitions c USING(competition_id) WHERE c.netuid=998)"
    )
    connection.execute(
        "DELETE FROM control_plane.evaluation_configs WHERE competition_id IN (SELECT competition_id FROM control_plane.competitions WHERE netuid=998)"
    )
    connection.execute(
        "DELETE FROM control_plane.evaluation_early_stopping_policies WHERE competition_id IN (SELECT competition_id FROM control_plane.competitions WHERE netuid=998)"
    )
    connection.execute(
        "DELETE FROM control_plane.competitions WHERE netuid=998 AND competition_key<>'main'"
    )
    connection.execute("DELETE FROM control_plane.competitions WHERE netuid=998")


@pytest.fixture
def database():
    with psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row) as connection:
        assert (
            connection.execute("SELECT current_database() AS name").fetchone()["name"]
            == "teutonic_test"
        )
        cleanup(connection)
        connection.execute(
            "INSERT INTO control_plane.competitions(netuid,chain_generation,name) VALUES (998,'split-admin-test','existing-main')"
        )
        try:
            yield connection
        finally:
            cleanup(connection)


def fixture_fetch(**source):
    name = source["name"]
    weights = {
        category: spec["weight"]
        for category, spec in chain_config.SPLIT_DEFAULTS[name]["categories"].items()
    }
    manifest = {
        "format": "teutonic-split-v1",
        "split": name,
        "category_weights": weights,
        "shards": [
            {
                "url": f"https://datasets.example/{name}/{category}.npy",
                "category": category,
                "sha256": "a" * 64,
                "n_tokens": 30000 * 2048,
                "size_bytes": 30000 * 2048 * 4 + 128,
            }
            for category in weights
        ],
    }
    return DatasetManifestSnapshot(
        **source,
        manifest_sha256=hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
        manifest=manifest,
    )


def seed_main(database):
    main = replace(
        fixture_fetch(name="math", manifest_url="https://datasets.example/main.json", proportion=1),
        name="original-main",
    )
    store_evaluation_configuration(
        database,
        netuid=998,
        chain_generation="split-admin-test",
        competition="existing-main",
        dataset_label="original",
        n=30000,
        delta_threshold=0.012,
        manifests=[main],
        shards_per_dataset=4,
    )


def run_admin(*args):
    with (
        patch.dict(
            os.environ,
            {
                "TEUTONIC_DATABASE_URL": DATABASE_URL,
                "TEUTONIC_NETUID": "998",
                "TEUTONIC_CHAIN_GENERATION": "split-admin-test",
                "TEUTONIC_COMPETITION": "existing-main",
            },
        ),
        redirect_stdout(io.StringIO()),
    ):
        return admin.main(list(args))


def policies(database):
    return database.execute(
        "SELECT c.competition_key, ec.config_version, ec.delta_threshold FROM control_plane.competitions c JOIN control_plane.evaluation_configs ec USING(competition_id) WHERE c.netuid=998 AND ec.active ORDER BY c.competition_key"
    ).fetchall()


def test_initialize_dry_run_targeted_update_and_atomic_all(database):
    seed_main(database)
    original = policies(database)
    with patch.object(admin, "fetch_dataset_manifest", side_effect=fixture_fetch) as fetch:
        assert run_admin("--all", "--initialize", "--dry-run") == 0
        assert policies(database) == original
        assert (
            database.execute(
                "SELECT count(*) AS n FROM control_plane.competitions WHERE netuid=998"
            ).fetchone()["n"]
            == 1
        )
        assert run_admin("--all", "--initialize") == 0
        assert fetch.call_count == 6
    rows = policies(database)
    assert len(rows) == 4
    assert next(r for r in rows if r["competition_key"] == "main") == original[0]
    assert all(r["delta_threshold"] == 0.003 for r in rows if r["competition_key"] != "main")
    with patch.object(
        admin, "fetch_dataset_manifest", side_effect=AssertionError("unexpected network request")
    ):
        assert run_admin("--competition", "math", "--delta-threshold", ".004") == 0
    changed = policies(database)
    assert [r for r in changed if r["competition_key"] != "math"] == [
        r for r in rows if r["competition_key"] != "math"
    ]
    assert next(r for r in changed if r["competition_key"] == "math")["delta_threshold"] == 0.004
    real_store = admin.store_evaluation_configuration

    def fail_last(connection, **kwargs):
        if kwargs["competition"] == "text":
            raise RuntimeError("simulated activation failure")
        return real_store(connection, **kwargs)

    with (
        patch.object(admin, "store_evaluation_configuration", side_effect=fail_last),
        pytest.raises(RuntimeError, match="simulated"),
    ):
        run_admin("--all", "--delta-threshold", ".006")
    assert policies(database) == changed
    assert run_admin("--all", "--delta-threshold", ".006") == 0
    assert all(r["delta_threshold"] == 0.006 for r in policies(database))
