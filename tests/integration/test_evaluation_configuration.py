from __future__ import annotations

import hashlib
import os
import unittest

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    dict_row = None

from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    canonical_manifest_bytes,
    store_evaluation_configuration,
)
from teutonic.validator.repository import ValidatorRepository


DATABASE_URL = os.environ.get("TEUTONIC_TEST_DATABASE_URL")


def snapshot(marker: str) -> DatasetManifestSnapshot:
    manifest = {
        "shards": [{
            "key": "shards/part-000.npy",
            "sha256": marker * 64,
            "size_bytes": 4096,
            "n_tokens": 4096,
        }]
    }
    return DatasetManifestSnapshot(
        name=f"fixture-{marker}",
        manifest_url=f"https://datasets.example/{marker}/manifest.json",
        manifest_sha256=hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
        proportion=1.0,
        manifest=manifest,
    )


@unittest.skipUnless(DATABASE_URL and psycopg, "PostgreSQL integration database required")
class EvaluationConfigurationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        self.connection.execute(
            """
            INSERT INTO control_plane.competitions (netuid, chain_generation, name)
            VALUES (999, 'evaluation-config-test', 'fixture')
            """
        )

    def tearDown(self) -> None:
        self.connection.rollback()
        self.connection.close()

    def test_snapshots_are_immutable_and_activation_is_atomic(self) -> None:
        first = store_evaluation_configuration(
            self.connection,
            netuid=999,
            chain_generation="evaluation-config-test",
            competition="fixture",
            dataset_label="first",
            n=2000,
            delta_threshold=0.5,
            manifests=(snapshot("a"),),
            shards_per_dataset=4,
        )
        second = store_evaluation_configuration(
            self.connection,
            netuid=999,
            chain_generation="evaluation-config-test",
            competition="fixture",
            dataset_label="second",
            n=100,
            delta_threshold=0.25,
            manifests=(snapshot("b"),),
            shards_per_dataset=4,
        )
        self.assertNotEqual(first.config_version, second.config_version)
        rows = self.connection.execute(
            """
            SELECT config_version, active
              FROM control_plane.evaluation_configs
             WHERE competition_id = (
                   SELECT competition_id FROM control_plane.competitions
                    WHERE netuid = 999 AND chain_generation = 'evaluation-config-test'
                      AND name = 'fixture'
             )
             ORDER BY created_at, config_version
            """
        ).fetchall()
        self.assertEqual(sum(bool(row["active"]) for row in rows), 1)
        self.assertTrue(
            next(row for row in rows if row["config_version"] == second.config_version)["active"]
        )

        repository = ValidatorRepository(
            self.connection,
            netuid=999,
            chain_generation="evaluation-config-test",
            competition="fixture",
            instance_id="test",
            public_model_bucket="public-models",
        )
        settings = repository.load_evaluation_settings()
        self.assertEqual(settings.config_version, second.config_version)
        self.assertEqual(settings.n, 100)
        self.assertEqual(settings.delta_threshold, 0.25)
        self.assertEqual(settings.manifests[0].name, "fixture-b")


if __name__ == "__main__":
    unittest.main()
