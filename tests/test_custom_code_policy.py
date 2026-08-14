"""Quasar auto_map validation — the remote-code gate.

Quasar checkpoints ship their own modeling code, so the validator executes files
that a miner uploaded. `_validate_quasar_auto_map` is the check that keeps
auto_map pointed at the two known local modules instead of an arbitrary hub repo.
"""
from __future__ import annotations

import pytest

from validator import QUASAR_EXPECTED_AUTO_MAP, _validate_quasar_auto_map


def test_expected_auto_map_passes():
    assert _validate_quasar_auto_map(dict(QUASAR_EXPECTED_AUTO_MAP)) is None


@pytest.mark.parametrize("auto_map", [None, "", [], 42])
def test_non_dict_auto_map_is_rejected(auto_map):
    assert _validate_quasar_auto_map(auto_map) == "quasar config must provide auto_map"


def test_missing_key_is_rejected():
    auto_map = dict(QUASAR_EXPECTED_AUTO_MAP)
    auto_map.pop("AutoConfig")
    assert "keys mismatch" in _validate_quasar_auto_map(auto_map)


def test_extra_key_is_rejected():
    auto_map = dict(QUASAR_EXPECTED_AUTO_MAP)
    auto_map["AutoModelForSequenceClassification"] = "modeling_qwen3_5.Whatever"
    assert "keys mismatch" in _validate_quasar_auto_map(auto_map)


def test_wrong_target_class_is_rejected():
    auto_map = dict(QUASAR_EXPECTED_AUTO_MAP)
    auto_map["AutoModelForCausalLM"] = "modeling_qwen3_5.EvilForCausalLM"
    assert "mismatch" in _validate_quasar_auto_map(auto_map)


def test_hub_qualified_module_is_rejected():
    """`repo--module.Class` would resolve code from someone else's repo."""
    auto_map = dict(QUASAR_EXPECTED_AUTO_MAP)
    auto_map["AutoConfig"] = "attacker/repo--configuration_qwen3_5.QuasarConfig"
    assert _validate_quasar_auto_map(auto_map) is not None


def test_path_traversal_module_is_rejected():
    auto_map = dict(QUASAR_EXPECTED_AUTO_MAP)
    auto_map["AutoConfig"] = "../../etc/configuration_qwen3_5.QuasarConfig"
    assert _validate_quasar_auto_map(auto_map) is not None
