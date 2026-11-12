from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_category_competitions import original_sources, transition

import chain_config
from scripts.legacy import prepare_competition_mixes as prepare
from teutonic.competitions import ACTIVE_COMPETITIONS
from teutonic.evaluation.configuration import canonical_manifest_bytes
from teutonic.evaluation.mixes import competition_snapshot
from teutonic.weights import transition as maintenance
from teutonic.weights.policy import (
    begin_mix07_replacement,
    replacement_after_win,
    replacement_rewards,
    validate_transition,
)


def incumbent_state(all_kings=False):
    state = transition()
    keys = [f"mix{i:02d}" for i in range(1, 7)] if all_kings else ["mix04", "mix05"]
    for key in keys:
        state = replacement_after_win(state, key, key + "-king")
    return state


@pytest.mark.parametrize("all_kings", [False, True])
def test_mix07_takes_only_mix04_slot_on_first_crown(all_kings):
    old = incumbent_state(all_kings)
    pending = begin_mix07_replacement(old)
    assert replacement_rewards(old) == replacement_rewards(pending)
    assert pending == begin_mix07_replacement(pending)
    assert old["version"] == "mix-replacement-v1"
    # MAIN and MIX05 wins during the wait must not remove MIX04's share.
    pending = replacement_after_win(pending, "main", "new-main")
    pending = replacement_after_win(pending, "mix05", "new-mix05")
    before = dict(zip(*replacement_rewards(pending)))
    assert before["mix04-king"] == 0.125
    crowned = replacement_after_win(pending, "mix07", "new-mix07")
    after = dict(zip(*replacement_rewards(crowned)))
    assert after == {**{h: w for h, w in before.items() if h != "mix04-king"}, "new-mix07": 0.125}
    assert crowned["legacy"] == pending["legacy"]
    assert len(crowned["evictions"]) == len(pending["evictions"]) + 1
    assert pending["mixes"]["mix04"] == "mix04-king"
    again = replacement_after_win(crowned, "mix07", "next-mix07")
    assert again["legacy"] == crowned["legacy"]
    assert again["evictions"] == crowned["evictions"]
    assert begin_mix07_replacement(again) == again
    assert sum(replacement_rewards(again)[1]) == pytest.approx(1)


def test_overlapping_hotkeys_keep_other_reward_entitlements():
    old = incumbent_state(True)
    old["mixes"]["mix04"] = old["main"]["hotkey"]
    old["mixes"]["mix05"] = old["main"]["hotkey"]
    pending = begin_mix07_replacement(old)
    assert dict(zip(*replacement_rewards(pending)))["main"] == 0.5
    new = replacement_after_win(pending, "mix07", "winner")
    assert dict(zip(*replacement_rewards(new)))["main"] == 0.375
    assert dict(zip(*replacement_rewards(new)))["winner"] == 0.125


def test_replacement_rejects_double_slots_and_uncutover_or_retired_wins():
    old = incumbent_state(True)
    with pytest.raises(ValueError):
        replacement_after_win(old, "mix07", "winner")
    with pytest.raises(ValueError, match="rewarded king"):
        begin_mix07_replacement(transition())
    pending = begin_mix07_replacement(old)
    with pytest.raises(ValueError, match="retired MIX04"):
        replacement_after_win(pending, "mix04", "winner")
    pending["mixes"]["mix07"] = "winner"
    with pytest.raises(ValueError, match="share one reward slot"):
        validate_transition(pending)


def test_mix07_manifest_is_only_dclm_and_local_preparation_needs_only_dclm(tmp_path, monkeypatch):
    assert "mix07" in ACTIVE_COMPETITIONS and "mix04" not in ACTIVE_COMPETITIONS
    definition = chain_config.COMPETITION_DEFAULTS["mix07"]
    assert definition["categories"] == {"dclm-baseline-1.0": 1.0}
    dclm = next(s for s in original_sources() if s.name == "dclm-baseline-1.0")
    (tmp_path / "dclm-baseline-1.0.json").write_bytes(canonical_manifest_bytes(dclm.manifest))
    monkeypatch.setattr(prepare, "fetch_dataset_manifest", Mock(side_effect=AssertionError("network")))
    output = tmp_path / "bundle"
    prepare.main(["--competition", "MIX07", "--source-dir", str(tmp_path), "--output-dir", str(output)])
    snap = competition_snapshot("mix07", bundle_dir=output)
    assert snap.manifest["category_weights"] == {"dclm-baseline-1.0": 1.0}
    assert {s["dataset"] for s in snap.manifest["shards"]} == {"dclm-baseline-1.0"}
    assert snap.manifest_url.endswith(f"/versions/{snap.manifest_sha256}.json")
    assert not (output / "mix04.json").exists()


class ReplacementDatabase:
    def __init__(self):
        self.rows = [
            {"competition_key": "mix04", "name": "mix04", "competition_id": "four",
             "active": True, "current_reign_id": "four-king", "king_hotkey": "mix04-king"},
            {"competition_key": "mix05", "name": "mix05", "competition_id": "five",
             "active": True, "current_reign_id": "five-king", "king_hotkey": "mix05-king"},
        ]
        self.pin = {"url": "https://example.com/pinned-index.json"}

    def execute(self, sql, parameters):
        if "FROM control_plane.competitions" in sql:
            assert "FOR UPDATE" in sql
            return SimpleNamespace(fetchall=lambda: deepcopy(self.rows))
        assert "SELECT long_documents" in sql
        return SimpleNamespace(fetchone=lambda: {"long_documents": {"index_manifest": self.pin}})


def test_cutover_pins_actual_mix05_king_once_and_preserves_document_index(monkeypatch):
    database = ReplacementDatabase()
    guard = Mock()
    monkeypatch.setattr(maintenance, "lock_drained_services", guard)
    main = {"competition_id": "main", "reward_transition": incumbent_state()}
    plan = maintenance.prepare_mix07_replacement(database, main)
    guard.assert_called_once_with(database, main)
    assert plan["starting_reign_id"] == "five-king"
    assert plan["index_manifest"] == database.pin
    assert replacement_rewards(plan["reward_transition"]) == replacement_rewards(main["reward_transition"])
    database.rows[0]["active"] = False
    database.rows[1]["current_reign_id"] = "later-five-king"
    database.rows.append({"competition_key": "mix07", "name": "mix07", "competition_id": "seven",
                          "active": True, "current_reign_id": None, "starting_reign_id": "five-king"})
    main["reward_transition"] = plan["reward_transition"]
    repeated = maintenance.prepare_mix07_replacement(database, main)
    assert repeated["starting_reign_id"] == "five-king"
    assert repeated["reward_transition"] == plan["reward_transition"]


@pytest.mark.parametrize("mutation", ["no_mix05_king", "wrong_reward", "existing_mix07"])
def test_cutover_rejects_ambiguous_starting_king_or_reward_state(monkeypatch, mutation):
    database = ReplacementDatabase()
    monkeypatch.setattr(maintenance, "lock_drained_services", Mock())
    if mutation == "no_mix05_king":
        database.rows[1]["current_reign_id"] = None
    elif mutation == "wrong_reward":
        database.rows[0]["king_hotkey"] = "different-king"
    else:
        database.rows.append({"competition_key": "mix07", "name": "mix07"})
    with pytest.raises(ValueError):
        maintenance.prepare_mix07_replacement(database, {"competition_id": "main", "reward_transition": incumbent_state()})
