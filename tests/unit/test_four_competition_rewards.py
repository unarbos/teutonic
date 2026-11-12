from copy import deepcopy

import pytest
from test_mix07_replacement import incumbent_state

import chain_config
from scripts.retire_mix_competitions import prepare
from teutonic.competitions import ACTIVE_COMPETITIONS, ACTIVE_INCENTIVES
from teutonic.weights.policy import (
    begin_four_competition_policy,
    begin_mix07_replacement,
    replacement_after_win,
    replacement_rewards,
)


def current():
    return replacement_after_win(begin_mix07_replacement(incumbent_state(True)), "mix07", "seven")


def test_cutover_preserves_history_and_pays_only_four_positions():
    old = current()
    saved = deepcopy(old)
    new = begin_four_competition_policy(old)
    assert old == saved
    assert dict(zip(*replacement_rewards(new))) == {
        "main": .4, "mix01-king": .2, "mix06-king": .2, "seven": .2,
    }
    assert set(new["retired_rewards"]) == {"mix02", "mix03", "mix05"}
    assert begin_four_competition_policy(new) == new
    assert ACTIVE_COMPETITIONS == ("main", "mix01", "mix06", "mix07")
    assert {k: v["incentive"] for k, v in chain_config.COMPETITION_DEFAULTS.items() if v["incentive"]} == ACTIVE_INCENTIVES


@pytest.mark.parametrize("key", ["mix02", "mix03", "mix04", "mix05", "math", "mix99"])
def test_retired_promotions_cannot_reintroduce_reward_slots(key):
    with pytest.raises(ValueError, match="retired"):
        replacement_after_win(begin_four_competition_policy(current()), key, "winner")


def test_shared_hotkey_retains_other_entitlements_and_new_wins_transfer_only_one():
    state = current()
    state["mixes"]["mix02"] = state["mixes"]["mix01"] = "main"
    new = begin_four_competition_policy(state)
    assert dict(zip(*replacement_rewards(new)))["main"] == pytest.approx(.6)
    new = replacement_after_win(new, "mix01", "new-one")
    assert dict(zip(*replacement_rewards(new)))["main"] == .4
    new = replacement_after_win(new, "main", "new-main")
    assert dict(zip(*replacement_rewards(new))) == {
        "new-main": .4, "new-one": .2, "mix06-king": .2, "seven": .2,
    }


def test_cutover_refuses_kingless_mix_or_legacy_positions():
    with pytest.raises(ValueError, match="crowned"):
        begin_four_competition_policy(begin_mix07_replacement(incumbent_state(True)))
    state = current()
    state["mixes"].pop("mix03")
    state["legacy"] = [{"id": "legacy", "hotkey": "legacy", "weight": .1}]
    with pytest.raises(ValueError, match="legacy"):
        begin_four_competition_policy(state)


def database_fixture():
    state = current()
    recipients = {"main": state["main"]["hotkey"], **state["mixes"]}
    rows = [{"competition_key": k, "name": k, "competition_id": k, "active": True,
             "current_reign_id": k, "king_hotkey": h, "reward_transition": state if k == "main" else None}
            for k, h in recipients.items()]
    h, w = replacement_rewards(state)
    publication = {"policy_hotkeys": h, "policy_weights": w, "target_hotkeys": h,
                   "target_uids": list(range(len(h)))}
    return rows, publication


def test_migration_plans_both_activation_and_durable_weight_payload():
    rows, publication = database_fixture()
    plan = prepare(rows, publication)
    assert set(plan["retired_ids"]) == {"mix02", "mix03", "mix05"}
    assert plan["policy_weights"] == [.4, .2, .2, .2]
    assert plan["normalized_weights"] == [.4, .2, .2, .2]
    assert plan["changed"]
    for row in rows:
        if row["competition_key"] == "main":
            row["reward_transition"] = plan["transition"]
        if row["competition_id"] in plan["retired_ids"]:
            row["active"] = False
    publication.update({k: plan[k] for k in publication})
    assert not prepare(rows, publication)["changed"]


@pytest.mark.parametrize("bad", ["king", "policy", "mapping"])
def test_migration_rejects_inconsistent_live_snapshot(bad):
    rows, publication = database_fixture()
    if bad == "king":
        rows[0]["king_hotkey"] = "different"
    elif bad == "policy":
        publication["policy_weights"] = [.5] * len(publication["policy_hotkeys"])
    else:
        publication["target_hotkeys"] = publication["target_hotkeys"][1:]
        publication["target_uids"] = publication["target_uids"][1:]
    with pytest.raises(ValueError):
        prepare(rows, publication)
