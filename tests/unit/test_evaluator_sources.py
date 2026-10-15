from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def load_sources_module(tmp_path: Path):
    base = ModuleType("teutonic.evaluator.engine")
    from test_token_shards import old_loader

    base.shuffled_indices = old_loader().shuffled_indices
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
    import teutonic.evaluator

    with patch.object(teutonic.evaluator, "engine", base, create=True), patch.dict(
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

    def fake_load(shard, _request, _rng, _limit, on_phase=None, *, indices=None):
        del on_phase
        return loaded[Path(shard.url).name]

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


def test_parallel_sampling_matches_serial_local_files(tmp_path):
    import hashlib
    import random
    import threading

    import numpy as np

    from test_token_shards import NpyShard, old_loader, shard_fixture

    sources = load_sources_module(tmp_path)
    legacy = old_loader()
    refs, servers, paths = [], {}, {}
    for i in range(4):
        array = np.arange((40 + i) * 8, dtype="<u4").reshape(40 + i, 8) + i * 1000
        ref, server = shard_fixture(array, tmp_path)
        body = server.objects.pop(ref.url)
        ref.url = f"https://datasets.example/{i}.npy"
        server.objects[ref.url] = body
        path = tmp_path / f"{i}.npy"
        path.write_bytes(body)
        refs.append(ref)
        servers[ref.url], paths[ref.url] = server, path
    request = SimpleNamespace(
        n=20, seq_len=8, vocab_size=10000, block_hash="block", hotkey="hotkey",
        dataset_sources=[{
            "name": "fixture", "proportion": 1.0, "target_sequences": 20,
            "shards": [{**vars(ref), "target_sequences": 5} for ref in refs],
        }],
    )
    rng = np.random.default_rng(sources._source_seed(123, "fixture"))
    expected = []
    for i, ref in enumerate(refs):
        rows = legacy.load_indexed_sequences_from_npy_shard(str(paths[ref.url]), request, rng, 15)
        expected.extend((tokens, {
            "shard_group_index": 0, "shard_index": i, "shard_sequence_index": index,
        }) for index, tokens in rows[:5])
    random.Random(123).shuffle(expected)
    # All four files must be in flight simultaneously. The timeout makes a
    # regression to serial loading fail without hanging the test suite.
    barrier = threading.Barrier(4, timeout=10)

    def open_shard(ref):
        barrier.wait()
        return NpyShard(ref, opener=servers[ref.url].open)

    sources.open_shard = open_shard
    sequences, metadata = sources.sample_pretokenized_sequences(request)
    assert sequences == [tokens for tokens, _ in expected]
    assert metadata["_sample_provenance"] == [pin for _, pin in expected]
    assert metadata["digest"] == hashlib.sha256(np.asarray(sequences, dtype=np.int64).tobytes()).hexdigest()
    assert metadata["sources"][0]["used_refs"] == [ref.url for ref in refs]
