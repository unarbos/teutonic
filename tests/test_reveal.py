"""On-chain reveal parsing — the boundary between untrusted chain bytes and the
validator's model-fetch path. A permissive parse here means fetching whatever a
miner names; a wrong reject silently drops an honest submission.
"""
from __future__ import annotations

import pytest

from model_store import (
    DIGEST_RE,
    ModelRef,
    build_reveal_v3,
    build_reveal_v4,
    parse_reveal_v3,
    parse_reveal_v4,
)

HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
SHA = "sha256:" + "a" * 64
HF = "hf:" + "b" * 40


def test_v4_round_trip():
    ref = ModelRef("miner/teutonic-x", SHA)
    payload = build_reveal_v4(ref, HOTKEY)
    assert payload == f"v4|miner/teutonic-x|{SHA}|{HOTKEY}"
    parsed, hotkey = parse_reveal_v4(payload)
    assert parsed.repo == ref.repo
    assert parsed.digest == SHA
    assert hotkey == HOTKEY


def test_v4_round_trip_hf_digest():
    parsed, _ = parse_reveal_v4(build_reveal_v4(ModelRef("miner/teutonic-x", HF), HOTKEY))
    assert parsed.digest == HF


def test_v3_round_trip():
    ref = ModelRef("miner/teutonic-x", SHA)
    king_digest, parsed, hotkey = parse_reveal_v3(build_reveal_v3(SHA, ref, HOTKEY))
    assert king_digest == SHA
    assert parsed.repo == ref.repo
    assert hotkey == HOTKEY


def test_build_rejects_non_ss58_hotkey():
    with pytest.raises(ValueError):
        build_reveal_v4(ModelRef("miner/teutonic-x", SHA), "not-an-ss58")


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "v4",
        f"v4|miner/teutonic-x|{SHA}",  # too few fields
        f"v4|miner/teutonic-x|{SHA}|{HOTKEY}|extra",  # too many
        f"v3|miner/teutonic-x|{SHA}|{HOTKEY}",  # wrong version prefix
        f"v4|miner/teutonic-x|{SHA}|not-an-ss58",
    ],
)
def test_v4_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        parse_reveal_v4(payload)


def test_v4_rejects_v3_payload():
    with pytest.raises(ValueError):
        parse_reveal_v4(build_reveal_v3(SHA, ModelRef("miner/teutonic-x", SHA), HOTKEY))


@pytest.mark.parametrize("digest", [SHA, HF])
def test_digest_re_accepts_both_supported_shapes(digest):
    assert DIGEST_RE.match(digest)


@pytest.mark.parametrize(
    "digest",
    [
        "a" * 64,  # unprefixed
        "sha256:" + "a" * 63,  # short
        "sha256:" + "A" * 64,  # uppercase hex
        "hf:" + "b" * 39,
        "md5:" + "a" * 32,
        "sha256:" + "z" * 64,
    ],
)
def test_digest_re_rejects_bad_shapes(digest):
    assert not DIGEST_RE.match(digest)
