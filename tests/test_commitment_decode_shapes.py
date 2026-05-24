import ast
import unittest
from pathlib import Path

from bittensor.core.chain_data.utils import decode_revealed_commitment


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "validator.py"


def _load_decode_commitment_pair():
    tree = ast.parse(VALIDATOR_PATH.read_text(), filename=str(VALIDATOR_PATH))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_commitment_pair":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(module, str(VALIDATOR_PATH), "exec"), namespace)
            return namespace["_decode_commitment_pair"]
    raise AssertionError("_decode_commitment_pair not found in validator.py")


def _scale_compact_prefix(length: int) -> bytes:
    if length < 1 << 6:
        return bytes([(length << 2) | 0b00])
    if length < 1 << 14:
        encoded = (length << 2) | 0b01
        return encoded.to_bytes(2, "little")
    if length < 1 << 30:
        encoded = (length << 2) | 0b10
        return encoded.to_bytes(4, "little")
    raise ValueError("length too large for this test helper")


def _raw_commitment_bytes(message: str) -> bytes:
    payload = message.encode("utf-8")
    return _scale_compact_prefix(len(payload)) + payload


class CommitmentDecodeShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._decode_commitment_pair = staticmethod(_load_decode_commitment_pair())

    def test_validator_decoder_accepts_latin1_wrapped_raw_bytes(self):
        message = "v4|repo|sha256:abc123|5Hotkey"
        raw = _raw_commitment_bytes(message)
        pair = ("5ExampleHotkey", [(raw.decode("latin-1"), 12345)])

        hotkey, entries = self._decode_commitment_pair(pair)

        self.assertEqual(hotkey, "5ExampleHotkey")
        self.assertEqual(entries, [(12345, message)])

    def test_sdk_decoder_accepts_hex_serialized_payload(self):
        message = "v4|repo|sha256:abc123|5Hotkey"
        raw = _raw_commitment_bytes(message)

        block, decoded = decode_revealed_commitment((f"0x{raw.hex()}", 12345))

        self.assertEqual(block, 12345)
        self.assertEqual(decoded, message)

    def test_validator_decoder_accepts_hex_serialized_payload(self):
        message = "v4|repo|sha256:abc123|5Hotkey"
        raw = _raw_commitment_bytes(message)
        pair = ("5ExampleHotkey", [(f"0x{raw.hex()}", 12345)])

        _, entries = self._decode_commitment_pair(pair)

        self.assertEqual(entries, [(12345, message)])


if __name__ == "__main__":
    unittest.main()
