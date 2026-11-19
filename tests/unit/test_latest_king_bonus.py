from copy import deepcopy

import pytest

from scripts.activate_latest_king_bonus import prepare
from scripts.retire_mix_competitions import apply_plan

ACTIVE_COMPETITIONS = ("main", "mix01", "mix06", "mix07")  # V4 replay roster.
from teutonic.weights.contracts import WeightPlan
from teutonic.weights.policy import (
    FOUR_COMPETITION_POLICY,
    LATEST_KING_POLICY,
    begin_four_competition_policy,
    begin_latest_king_policy,
    mapped_weight_plan,
    replacement_after_win,
    replacement_rewards,
    validate_transition,
)


def old_state():
    return {"version": FOUR_COMPETITION_POLICY,
            "main": {"id": "main", "hotkey": "main-king", "weight": .4},
            "mixes": {k: k + "-king" for k in ACTIVE_COMPETITIONS if k != "main"},
            "legacy": [], "evictions": []}


@pytest.mark.parametrize("latest", ACTIVE_COMPETITIONS)
def test_each_current_king_can_receive_the_bonus(latest):
    old = old_state()
    saved = deepcopy(old)
    state = begin_latest_king_policy(old, latest)
    assert old == saved
    assert dict(zip(*replacement_rewards(old))) == {
        "main-king": .4, "mix01-king": .2, "mix06-king": .2, "mix07-king": .2}
    rewards = dict(zip(*replacement_rewards(state)))
    assert rewards == {k + "-king": .4 if k == latest else .2 for k in ACTIVE_COMPETITIONS}
    assert sum(rewards.values()) == pytest.approx(1)
    assert begin_latest_king_policy(state, latest) == state


def test_bonus_moves_only_with_a_new_crown_and_replaces_outgoing_recipient():
    state = begin_latest_king_policy(old_state(), "mix07")
    for key in ("mix01", "main", "mix06", "mix06", "mix07"):
        before = deepcopy(state)
        winner = key + "-new-" + str(len(state["main"]["hotkey"]))
        # Preview does not change durable state; persistence happens at crown.
        preview = replacement_after_win(state, key, winner)
        assert state == before
        state = preview
        assert state["latest_competition"] == key
        rewards = dict(zip(*replacement_rewards(state)))
        assert rewards[winner] == .4
        assert sorted(rewards.values()) == [.2, .2, .2, .4]
        assert sum(rewards.values()) == pytest.approx(1)


def test_shared_hotkeys_receive_the_sum_of_base_positions_and_one_bonus():
    state = old_state()
    state["mixes"]["mix01"] = "main-king"
    state = begin_latest_king_policy(state, "mix01")
    assert dict(zip(*replacement_rewards(state)))["main-king"] == pytest.approx(.6)
    state = replacement_after_win(state, "mix06", "six-new")
    assert dict(zip(*replacement_rewards(state))) == {
        "main-king": .4, "six-new": .4, "mix07-king": .2}
    # A new model from the same hotkey still moves the competition's bonus.
    state = replacement_after_win(state, "main", "main-king")
    assert dict(zip(*replacement_rewards(state)))["main-king"] == pytest.approx(.6)
    h, w = replacement_rewards(state)
    _, _, mapped = mapped_weight_plan(h, w, {"six-new": 6, "mix07-king": 7}, burn_uid=0)
    assert mapped == (.5, .5)  # Existing UID eligibility normalization is unchanged.


@pytest.mark.parametrize("key", [None, "", "mix02", "mix03", "mix04", "mix05", "unknown"])
def test_missing_or_retired_bonus_recipient_is_rejected(key):
    with pytest.raises(ValueError, match="active competition"):
        begin_latest_king_policy(old_state(), key)
    with pytest.raises(ValueError):
        replacement_after_win(begin_latest_king_policy(old_state(), "main"), key, "winner")


def test_policy_cannot_silently_downgrade_or_change_bonus_on_reapply():
    state = begin_latest_king_policy(old_state(), "mix07")
    with pytest.raises(ValueError, match="downgrade"):
        begin_four_competition_policy(state)
    with pytest.raises(ValueError, match="persisted"):
        begin_latest_king_policy(state, "main")
    state.pop("latest_competition")
    with pytest.raises(ValueError):
        validate_transition(state)


def fixture():
    state = old_state()
    rows = [{"competition_id": key, "competition_key": key, "active": True,
             "current_reign_id": key + "-reign", "king_hotkey": key + "-king"}
            for key in ACTIVE_COMPETITIONS]
    rows[0].update(reward_transition=state, reward_reign_id="mix06-reign")
    h, w = replacement_rewards(state)
    publication = {"source_reign_id": "mix06-reign", "weight_publication_id": "publication",
                   "policy_hotkeys": list(h), "policy_weights": list(w),
                   "target_hotkeys": list(h), "target_uids": [1, 2, 3, 4]}
    return rows, publication


def test_cutover_uses_latest_crown_pointer_and_creates_a_valid_publisher_payload():
    rows, publication = fixture()
    saved = deepcopy((rows, publication))
    plan = prepare(rows, publication)
    assert (rows, publication) == saved
    assert plan["latest_competition"] == "mix06"
    assert plan["policy_weights"] == [.2, .2, .4, .2]
    assert plan["retired_ids"] == []
    WeightPlan(publication_id="publication", competition_id="main",
               source_reign_id="mix06-reign", current_reign_id="mix06-reign", reign_number=1,
               policy_version=LATEST_KING_POLICY, payload_revision=2,
               target_hotkeys=tuple(plan["target_hotkeys"]), target_uids=tuple(plan["target_uids"]),
               normalized_weights=tuple(plan["normalized_weights"]), payload_sha256=plan["payload_sha256"],
               idempotency_key="publish-weights:mix06-reign", previous_state="requested", attempt_count=0).validate(uid_count=5)
    rows[0]["reward_transition"] = plan["transition"]
    publication.update({k: plan[k] for k in publication if k in plan})
    assert not prepare(rows, publication)["changed"]


@pytest.mark.parametrize("bad", ["active", "king", "latest", "publication", "shares", "mapping"])
def test_cutover_rejects_inconsistent_live_state(bad):
    rows, publication = fixture()
    if bad == "active": rows[1]["active"] = False
    elif bad == "king": rows[1]["king_hotkey"] = "wrong"
    elif bad == "latest": rows[0]["reward_reign_id"] = "retired-reign"
    elif bad == "publication": publication["source_reign_id"] = "other-reign"
    elif bad == "shares": publication["policy_weights"][0] = .5
    elif bad == "mapping": publication["target_hotkeys"].pop(); publication["target_uids"].pop()
    with pytest.raises(ValueError):
        prepare(rows, publication)


def test_activation_updates_policy_and_payload_without_changing_competitions():
    rows, publication = fixture()
    statements = []
    class Connection:
        def execute(self, sql, params): statements.append((sql, params))
    plan = prepare(rows, publication)
    apply_plan(Connection(), rows[0], publication, plan)
    assert len(statements) == 2
    assert "reward_transition=" in statements[0][0]
    assert "payload_revision=payload_revision+1" in statements[1][0]
    assert all("active=false" not in sql for sql, _ in statements)
    statements.clear()
    apply_plan(Connection(), rows[0], publication, {**plan, "changed": False})
    assert not statements
