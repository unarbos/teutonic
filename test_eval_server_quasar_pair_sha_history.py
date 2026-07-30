import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import eval_server_quasar_pair as server


class SafetensorsShaHistoryTest(unittest.TestCase):
    def test_remote_hub_digests_match_downloaded_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            weights = model_dir / "model.safetensors"
            weights.write_bytes(b"test weights")
            file_sha = server.sha256_file(weights)
            expected = server.snapshot_safetensors_digest(str(model_dir))

            hf_file = SimpleNamespace(
                path="model.safetensors",
                lfs=SimpleNamespace(sha256=file_sha),
                blob_id=None,
            )
            with patch("huggingface_hub.HfApi") as api:
                api.return_value.list_repo_tree.return_value = [hf_file]
                self.assertEqual(
                    server.remote_snapshot_safetensors_digest(
                        "https://huggingface.co/example/model",
                        "hf:revision",
                    ),
                    expected,
                )

            hippius_file = SimpleNamespace(
                path="",
                rfilename="model.safetensors",
                lfs=None,
                blob_id=f"sha256:{file_sha}",
            )
            info = SimpleNamespace(siblings=[hippius_file])
            with (
                patch("hippius_hub.model_info", return_value=info),
                patch("model_store.get_hub_token", return_value="token"),
            ):
                self.assertEqual(
                    server.remote_snapshot_safetensors_digest(
                        "example/model",
                        "sha256:revision",
                    ),
                    expected,
                )

    def test_fourth_completed_eval_is_rejected(self):
        original = server.COMPLETED_SAFETENSORS_SHA_FILE
        digest = "a" * 64
        try:
            with tempfile.TemporaryDirectory() as tmp:
                server.COMPLETED_SAFETENSORS_SHA_FILE = Path(tmp) / "completed.txt"
                for prior_uses in range(3):
                    self.assertEqual(
                        server.reject_reused_safetensors(digest)[
                            "challenger_safetensors_prior_completed_evals"
                        ],
                        prior_uses,
                    )
                    server.record_completed_safetensors_sha(digest)

                with self.assertRaisesRegex(RuntimeError, digest):
                    server.reject_reused_safetensors(digest)
        finally:
            server.COMPLETED_SAFETENSORS_SHA_FILE = original


if __name__ == "__main__":
    unittest.main()
