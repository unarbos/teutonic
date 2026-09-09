from __future__ import annotations

import hashlib
import unittest

from teutonic.evaluation.configuration import (
    DatasetManifestSnapshot,
    EvaluationConfigurationError,
    EvaluationSettings,
    canonical_manifest_bytes,
    pretokenized_dataset_request,
    validate_dataset_manifest,
)


def snapshot(name: str, proportion: float, marker: str) -> DatasetManifestSnapshot:
    manifest = {
        "shard_prefix": f"{name}/shards/",
        "shards": [
            {
                "key": f"{name}/shards/part-{index:03d}.npy",
                "sha256": marker * 63 + str(index),
                "size_bytes": 8192,
                "n_tokens": 4096,
            }
            for index in range(3)
        ],
    }
    return DatasetManifestSnapshot(
        name=name,
        manifest_url=f"https://datasets.example/{name}/manifest.json",
        manifest_sha256=hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest(),
        proportion=proportion,
        manifest=manifest,
    )


class EvaluationConfigurationTests(unittest.TestCase):
    def test_manifest_requires_verified_pretokenized_shard_descriptors(self):
        with self.assertRaises(EvaluationConfigurationError):
            validate_dataset_manifest({"shards": [{"key": "part.npy"}]})
        with self.assertRaises(EvaluationConfigurationError):
            validate_dataset_manifest(
                {
                    "shards": [{
                        "key": "part.parquet",
                        "sha256": "a" * 64,
                        "size_bytes": 1,
                        "n_tokens": 1,
                    }]
                }
            )

    def test_validator_request_has_concrete_https_shards_and_exact_targets(self):
        settings = EvaluationSettings(
            config_version="f" * 64,
            dataset_label="global-v1",
            n=7,
            delta_threshold=0.5,
            manifests=(snapshot("alpha", 0.6, "a"), snapshot("beta", 0.4, "b")),
            shards_per_dataset=4,
        )
        request = pretokenized_dataset_request(
            settings,
            block_hash="0x" + "1" * 64,
            hotkey="5ExampleHotkey",
            seq_len=64,
        )
        self.assertEqual(request["source"], "pretokenized_npy")
        self.assertEqual(
            [source["target_sequences"] for source in request["sources"]],
            [4, 3],
        )
        for source in request["sources"]:
            for shard in source["shards"]:
                self.assertTrue(shard["url"].startswith("https://datasets.example/"))
                self.assertTrue(shard["url"].endswith(".npy"))
                self.assertEqual(set(shard), {"url", "sha256", "size_bytes", "n_tokens"})

    def test_shard_selection_is_deterministic_for_block_hash_and_hotkey(self):
        settings = EvaluationSettings(
            config_version="f" * 64,
            dataset_label="global-v1",
            n=1,
            delta_threshold=0.5,
            manifests=(snapshot("alpha", 1.0, "a"),),
            shards_per_dataset=4,
        )
        kwargs = {
            "settings": settings,
            "block_hash": "0x" + "2" * 64,
            "hotkey": "5ExampleHotkey",
            "seq_len": 64,
        }
        first = pretokenized_dataset_request(**kwargs)
        self.assertEqual(first, pretokenized_dataset_request(**kwargs))
        variants = {
            tuple(item["url"] for item in first["sources"][0]["shards"]),
            tuple(
                item["url"]
                for item in pretokenized_dataset_request(
                    **{**kwargs, "block_hash": "0x" + "3" * 64}
                )["sources"][0]["shards"]
            ),
            tuple(
                item["url"]
                for item in pretokenized_dataset_request(
                    **{**kwargs, "hotkey": "5AnotherHotkey"}
                )["sources"][0]["shards"]
            ),
        }
        self.assertGreater(len(variants), 1)


if __name__ == "__main__":
    unittest.main()
