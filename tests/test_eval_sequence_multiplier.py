from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import eval_server_quasar_pair as server


class SequenceMultiplierTests(unittest.TestCase):
    def request(self, **updates) -> server.EvalRequest:
        values = {
            "king_repo": "king",
            "challenger_repo": "challenger",
            "seq_len": 3,
            "seq_len_multiplier": 4,
        }
        values.update(updates)
        return server.EvalRequest(**values)

    def save_array(self, directory: str, name: str, array: np.ndarray) -> str:
        path = Path(directory) / name
        np.save(path, array.astype(np.uint32))
        return str(path)

    def test_default_is_four_times_2048(self):
        req = server.EvalRequest(king_repo="king", challenger_repo="challenger")

        self.assertEqual(req.seq_len_multiplier, 4)
        self.assertEqual(server.effective_seq_len(req), 8192)

    def test_2d_sequences_only_sample_complete_merged_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            array = np.arange(7 * 3, dtype=np.uint32).reshape(7, 3)
            path = self.save_array(directory, "matrix.npy", array)

            sequences = server.load_sequences_from_npy_shard(
                path, self.request(), np.random.default_rng(123)
            )

        self.assertEqual(len(sequences), 4)
        self.assertTrue(all(len(sequence) == 12 for sequence in sequences))
        self.assertEqual({sequence[0] for sequence in sequences}, {0, 3, 6, 9})
        self.assertIn(list(range(9, 21)), sequences)

    def test_1d_stream_only_samples_complete_merged_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_array(directory, "stream.npy", np.arange(23, dtype=np.uint32))

            sequences = server.load_sequences_from_npy_shard(
                path, self.request(), np.random.default_rng(123)
            )

        self.assertEqual(len(sequences), 4)
        self.assertTrue(all(len(sequence) == 12 for sequence in sequences))
        self.assertEqual({sequence[0] for sequence in sequences}, {0, 3, 6, 9})
        self.assertIn(list(range(9, 21)), sequences)

    def test_shard_without_enough_base_sequences_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_array(
                directory,
                "short.npy",
                np.arange(3 * 3, dtype=np.uint32).reshape(3, 3),
            )

            sequences = server.load_sequences_from_npy_shard(
                path, self.request(), np.random.default_rng(123)
            )

        self.assertEqual(sequences, [])

    def test_batch_size_is_divided_with_floor_of_one(self):
        req = self.request()

        self.assertEqual(server.scaled_batch_size(512, req), 128)
        self.assertEqual(server.scaled_batch_size(3, req), 1)


if __name__ == "__main__":
    unittest.main()
