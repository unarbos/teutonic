import tempfile
import unittest
from pathlib import Path

import eval_server_quasar_pair as server


class SafetensorsShaHistoryTest(unittest.TestCase):
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
