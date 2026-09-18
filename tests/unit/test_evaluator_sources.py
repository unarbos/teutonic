from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def load_sources_module(tmp_path: Path):
    base = ModuleType("teutonic.evaluator.engine")
    base.SHARD_CACHE_DIR = tmp_path
    base.dataset_seed = lambda _request: 123
    base.dataset_seed_material = lambda _request: "fixture-seed"
    base.is_truncated_npy_error = lambda _exc: False

    module_path = Path(__file__).parents[2] / "teutonic" / "evaluator" / "sources.py"
    spec = importlib.util.spec_from_file_location(
        "teutonic.evaluator._test_sources", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "teutonic.evaluator.engine": base,
            "teutonic.evaluator._test_sources": module,
        },
    ):
        spec.loader.exec_module(module)
    return module


def test_sampler_keeps_shard_provenance_aligned_through_shuffle(tmp_path) -> None:
    sources = load_sources_module(tmp_path)
    request = SimpleNamespace(
        n=4,
        seq_len=2,
        vocab_size=0,
        block_hash="0x" + "a" * 64,
        hotkey="hotkey",
        dataset_sources=[
            {
                "name": "fixture",
                "proportion": 1.0,
                "target_sequences": 4,
                "shards": [
                    {
                        "url": "https://datasets.example/alpha.npy",
                        "sha256": "a" * 64,
                        "size_bytes": 100,
                        "n_tokens": 100,
                        "target_sequences": 2,
                    },
                    {
                        "url": "https://datasets.example/beta.npy",
                        "sha256": "b" * 64,
                        "size_bytes": 100,
                        "n_tokens": 100,
                        "target_sequences": 2,
                    },
                ],
            }
        ],
    )
    loaded = {
        "alpha.npy": [(7, [107, 1]), (8, [108, 1])],
        "beta.npy": [(20, [220, 1]), (21, [221, 1])],
    }

    def fake_load(shard, _request, _rng, _limit, on_phase=None):
        del on_phase
        return tmp_path, loaded[Path(shard.url).name]

    sources._load_with_retry = fake_load
    sequences, metadata = sources.sample_pretokenized_sequences(request)
    observed = {
        sequence[0]: (
            provenance["shard_group_index"],
            provenance["shard_index"],
            provenance["shard_sequence_index"],
        )
        for sequence, provenance in zip(
            sequences, metadata["_sample_provenance"], strict=True
        )
    }

    assert observed == {
        107: (0, 0, 7),
        108: (0, 0, 8),
        220: (0, 1, 20),
        221: (0, 1, 21),
    }
    assert metadata["_source_labels"] == ["fixture"] * 4
