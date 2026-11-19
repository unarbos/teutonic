"""Gradual main/specialist reward policy, independent of UID mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy

MAIN_SHARES = {
    0: (20, 20, 20, 20, 20),
    1: (25, 20, 20, 20),
    2: (30, 20, 20),
    3: (40, 15),
}

FOUR_COMPETITION_POLICY = "four-competition-v3"
LATEST_KING_POLICY = "latest-king-bonus-v4"
REFINECODE_POLICY = "refinecode-competitions-v5"
REFINECODE_REPLACEMENT_POLICY = "refinecode-replacement-v6"
REFINECODE_COMPETITIONS = ("main2", "mix07", "mix08", "mix09")


def competition_rewards(main_history: Sequence[str], split_kings: Mapping[str, str]):
    if set(split_kings) - {"math", "code", "text"}:
        raise ValueError("unknown specialist competition")
    shares = MAIN_SHARES[len(split_kings)]
    main = tuple(dict.fromkeys(main_history))[: len(shares)]
    if not main:
        raise ValueError("main reward history is empty")
    # Bootstrap fixtures may contain fewer than five recipients. Preserve each
    # specialist's 15% and distribute the main budget across available main slots.
    scale = sum(shares) / sum(shares[: len(main)])
    pairs = [(h, s * scale / 100) for h, s in zip(main, shares)]
    pairs += [(split_kings[k], 0.15) for k in ("math", "code", "text") if k in split_kings]
    combined = {}
    for hotkey, weight in pairs:
        combined[hotkey] = combined.get(hotkey, 0.0) + weight
    return tuple(combined), tuple(combined.values())


def validate_transition(state):
    MIX_COMPETITIONS = tuple(f"mix{i:02d}" for i in range(1, 8))  # Historical policies.
    if isinstance(state, dict) and state.get("version") == REFINECODE_REPLACEMENT_POLICY:
        kings, legacy, evictions = state.get("kings"), state.get("legacy"), state.get("evictions")
        if (not isinstance(kings, dict) or not {"main2", "mix07"} <= set(kings)
                or set(kings) - set(REFINECODE_COMPETITIONS)
                or any(not isinstance(h, str) or not h for h in kings.values())
                or not isinstance(legacy, list) or not isinstance(evictions, list)
                or len(kings) + len(legacy) != 4 or len(evictions) != len(kings) - 2):
            raise ValueError("invalid gradual RefineCode reward roster")
        if any(not isinstance(e, dict) or set(e) != {"competition", "entry"} for e in evictions):
            raise ValueError("invalid RefineCode eviction record")
        entries = legacy + [e["entry"] for e in evictions]
        if (len(entries) != 2 or any(not isinstance(e, dict) for e in entries)
                or {e.get("id") for e in entries} != {"mix01", "mix06"}
                or any(not isinstance(e.get("hotkey"), str) or not e["hotkey"]
                       or not isinstance(e.get("reign_id"), str) or not e["reign_id"]
                       or type(e.get("crowned_block")) is not int or e["crowned_block"] < 0
                       for e in entries)
                or {e["competition"] for e in evictions} != set(kings) - {"main2", "mix07"}):
            raise ValueError("invalid retired RefineCode reward positions")
        ordered = sorted(entries, key=lambda e: (e["crowned_block"], e["reign_id"]))
        if [e["entry"] for e in evictions] + legacy != ordered:
            raise ValueError("retired rewards must be displaced oldest first")
        return
    if isinstance(state, dict) and state.get("version") == REFINECODE_POLICY:
        kings = state.get("kings")
        if (not isinstance(kings, dict) or set(kings) != set(REFINECODE_COMPETITIONS)
                or any(not isinstance(h, str) or not h for h in kings.values())):
            raise ValueError("RefineCode rewards require MAIN2/MIX07/MIX08/MIX09 kings")
        return
    if isinstance(state, dict) and state.get("version") in {FOUR_COMPETITION_POLICY, LATEST_KING_POLICY}:
        if (set(state.get("mixes", {})) != {"mix01", "mix06", "mix07"}
                or not state.get("main", {}).get("hotkey")
                or any(not isinstance(h, str) or not h for h in state["mixes"].values())
                or state.get("legacy") != []):
            raise ValueError("four-competition rewards require MAIN and three crowned MIX kings")
        if state["version"] == LATEST_KING_POLICY and state.get("latest_competition") not in {
            "main", "mix01", "mix06", "mix07"
        }:
            raise ValueError("latest king bonus requires an active competition")
        return
    if not isinstance(state, dict) or state.get("version") not in {
        "mix-replacement-v1", "mix-successor-v2"
    }:
        raise ValueError("invalid reward transition")
    entries = [state["main"], *state["legacy"]]
    if len({e["id"] for e in entries}) != len(entries) or len(state["legacy"]) > 6:
        raise ValueError("invalid legacy reward roster")
    if any(not e["hotkey"] or not math.isfinite(e["weight"]) or e["weight"] <= 0 for e in entries):
        raise ValueError("invalid incumbent reward allocation")
    if set(state["mixes"]) - set(MIX_COMPETITIONS) or any(not h for h in state["mixes"].values()):
        raise ValueError("invalid MIX reward roster")
    if len(state["mixes"]) > 6 or {"mix04", "mix07"} <= set(state["mixes"]):
        raise ValueError("MIX04 and MIX07 share one reward slot")
    if state["version"] == "mix-replacement-v1" and "mix07" in state["mixes"]:
        raise ValueError("MIX07 requires the successor transition")
    if state["version"] == "mix-successor-v2" and (
        state.get("successors") != {"mix07": "mix04"}
        or not {"mix04", "mix07"} & set(state["mixes"])
    ):
        raise ValueError("invalid MIX07 successor transition")
    if len(state["mixes"]) == 6 and state["legacy"]:
        raise ValueError("all MIX winners must displace the legacy roster")


def replacement_rewards(state):
    """Calculate the versioned allocation, retaining historical policy semantics."""
    validate_transition(state)
    if state["version"] == REFINECODE_REPLACEMENT_POLICY:
        combined = {}
        pairs = [(state["kings"][k], .4 if k == "main2" else .2)
                 for k in REFINECODE_COMPETITIONS if k in state["kings"]]
        pairs += [(entry["hotkey"], .2) for entry in state["legacy"]]
        for hotkey, share in pairs:
            combined[hotkey] = combined.get(hotkey, 0) + share
        return tuple(combined), tuple(combined.values())
    if state["version"] == REFINECODE_POLICY:
        combined = {}
        for key in REFINECODE_COMPETITIONS:
            hotkey = state["kings"][key]
            combined[hotkey] = combined.get(hotkey, 0) + (.4 if key == "main2" else .2)
        return tuple(combined), tuple(combined.values())
    if state["version"] in {FOUR_COMPETITION_POLICY, LATEST_KING_POLICY}:
        recipients = {"main": state["main"]["hotkey"], **state["mixes"]}
        # V3 must keep its original meaning for old durable states and replays.
        shares = ({"main": .4, "mix01": .2, "mix06": .2, "mix07": .2}
                  if state["version"] == FOUR_COMPETITION_POLICY else
                  {"main": .2, "mix01": .2, "mix06": .2, "mix07": .2})
        if state["version"] == LATEST_KING_POLICY:
            shares[state["latest_competition"]] += .2
        combined = {}
        for key, share in shares.items():
            hotkey = recipients[key]
            combined[hotkey] = combined.get(hotkey, 0) + share
        return tuple(combined), tuple(combined.values())
    entries = [state["main"], *state["legacy"]]
    remaining = 1 - 0.125 * len(state["mixes"])
    total = sum(e["weight"] for e in entries)
    combined = {}
    for entry in entries:
        hotkey = entry["hotkey"]
        combined[hotkey] = combined.get(hotkey, 0) + remaining * entry["weight"] / total
    for key, hotkey in sorted(state["mixes"].items()):
        combined[hotkey] = combined.get(hotkey, 0) + 0.125
    return tuple(combined), tuple(combined.values())


def replacement_after_win(state, competition, hotkey):
    """Pure preview used both before promotion and in the crowning transaction."""
    MIX_COMPETITIONS = tuple(f"mix{i:02d}" for i in range(1, 8))  # Historical policies.
    validate_transition(state)
    result = deepcopy(state)
    if result["version"] == REFINECODE_REPLACEMENT_POLICY:
        if competition not in REFINECODE_COMPETITIONS or not isinstance(hotkey, str) or not hotkey:
            raise ValueError("retired competition or empty winner cannot receive rewards")
        if competition not in result["kings"]:
            removed = result["legacy"].pop(0)
            result["evictions"].append({"competition": competition, "entry": removed})
        result["kings"][competition] = hotkey
        validate_transition(result)
        return result
    if result["version"] == REFINECODE_POLICY:
        if competition not in result["kings"] or not isinstance(hotkey, str) or not hotkey:
            raise ValueError("retired competition or empty winner cannot receive rewards")
        result["kings"][competition] = hotkey
        validate_transition(result)
        return result
    if result["version"] in {FOUR_COMPETITION_POLICY, LATEST_KING_POLICY}:
        if not isinstance(hotkey, str) or not hotkey:
            raise ValueError("winner hotkey is empty")
        if competition == "main":
            result["main"]["hotkey"] = hotkey
        elif competition in result["mixes"]:
            result["mixes"][competition] = hotkey
        else:
            raise ValueError("retired or unknown competition cannot receive rewards")
        if result["version"] == LATEST_KING_POLICY:
            result["latest_competition"] = competition
        validate_transition(result)
        return result
    if competition == "main":
        result["main"]["hotkey"] = hotkey
    elif competition in MIX_COMPETITIONS:
        if result["version"] == "mix-successor-v2" and competition == "mix04":
            raise ValueError("retired MIX04 cannot crown another king")
        predecessor = result.get("successors", {}).get(competition)
        if predecessor in result["mixes"]:
            incumbent = result["mixes"].pop(predecessor)
            result["evictions"].append({
                "competition": competition,
                "entry": {"id": predecessor, "hotkey": incumbent, "weight": 0.125},
            })
        elif competition not in result["mixes"] and result["legacy"]:
            removed = result["legacy"].pop(0)
            result["evictions"].append({"competition": competition, "entry": removed})
        result["mixes"][competition] = hotkey
    else:
        raise ValueError("retired or unknown competition cannot receive rewards")
    validate_transition(result)
    return result


def begin_mix07_replacement(state):
    """Retain MIX04's full reward slot until MIX07 is actually crowned."""
    validate_transition(state)
    if state["version"] in {FOUR_COMPETITION_POLICY, LATEST_KING_POLICY, REFINECODE_POLICY, REFINECODE_REPLACEMENT_POLICY}:
        raise ValueError("MIX07 replacement is already complete; four-competition policy is active")
    result = deepcopy(state)
    if result["version"] == "mix-successor-v2":
        return result
    if "mix04" not in result["mixes"]:
        raise ValueError("MIX04 must have a rewarded king before replacement")
    result.update(version="mix-successor-v2", successors={"mix07": "mix04"})
    validate_transition(result)
    return result


def begin_four_competition_policy(state):
    """Retire three reward slots without inventing kings or changing old state."""
    validate_transition(state)
    if state["version"] in {LATEST_KING_POLICY, REFINECODE_POLICY, REFINECODE_REPLACEMENT_POLICY}:
        raise ValueError("cannot downgrade the latest king policy through the old cutover")
    if state["version"] == FOUR_COMPETITION_POLICY:
        return deepcopy(state)
    if state["legacy"] or not {"mix01", "mix06", "mix07"} <= set(state["mixes"]):
        raise ValueError("cutover requires crowned MIX01/06/07 and no surviving legacy positions")
    result = deepcopy(state)
    result["version"] = FOUR_COMPETITION_POLICY
    result["mixes"] = {k: state["mixes"][k] for k in ("mix01", "mix06", "mix07")}
    result["retired_rewards"] = {
        k: h for k, h in state["mixes"].items() if k not in result["mixes"]
    }
    result.pop("successors", None)
    validate_transition(result)
    return result


def begin_refinecode_policy(state, kings):
    """Explicit cutover; never reinterpret an older durable reward policy."""
    validate_transition(state)
    if state["version"] not in {FOUR_COMPETITION_POLICY, LATEST_KING_POLICY, REFINECODE_POLICY}:
        raise ValueError("RefineCode cutover requires an existing four-competition policy")
    result = {"version": REFINECODE_POLICY, "kings": dict(kings)}
    validate_transition(result)
    if state["version"] == REFINECODE_POLICY and result != state:
        raise ValueError("RefineCode policy is already active with different recipients")
    return result


def begin_refinecode_replacement(state, kings, retired):
    """Starting opponents earn no new slot until a challenger is crowned."""
    validate_transition(state)
    if state["version"] not in {FOUR_COMPETITION_POLICY, LATEST_KING_POLICY, REFINECODE_POLICY}:
        raise ValueError("gradual RefineCode cutover requires an unmigrated policy")
    if set(kings) != {"main2", "mix07"}:
        raise ValueError("only MAIN2 and MIX07 initially own active reward slots")
    result = {"version": REFINECODE_REPLACEMENT_POLICY, "kings": dict(kings),
              "legacy": sorted(deepcopy(retired), key=lambda e: (e["crowned_block"], e["reign_id"])),
              "evictions": []}
    validate_transition(result)
    if state["version"] == REFINECODE_POLICY:
        expected = {k: state["kings"][k] for k in kings}
    else:
        expected = {"main2": state["main"]["hotkey"], "mix07": state["mixes"]["mix07"]}
        if any(e["hotkey"] != state["mixes"][e["id"]] for e in retired):
            raise ValueError("retired positions differ from the existing reward policy")
    if kings != expected:
        raise ValueError("continuing kings differ from the existing reward policy")
    return result


def begin_latest_king_policy(state, latest_competition):
    """Activate the bonus using the durable last-crowned competition."""
    validate_transition(state)
    if state["version"] not in {FOUR_COMPETITION_POLICY, LATEST_KING_POLICY}:
        raise ValueError("latest king bonus requires the four-competition policy")
    result = deepcopy(state)
    result.update(version=LATEST_KING_POLICY, latest_competition=latest_competition)
    validate_transition(result)
    if state["version"] == LATEST_KING_POLICY and result != state:
        raise ValueError("latest king differs from the persisted bonus recipient")
    return result


def mapped_weight_plan(hotkeys, shares, uid_by_hotkey, *, burn_uid):
    if len(hotkeys) != len(shares) or not hotkeys:
        raise ValueError("reward hotkeys and shares must have matching non-empty lengths")
    if any(not math.isfinite(share) or share < 0 for share in shares):
        raise ValueError("reward shares must be finite and nonnegative")
    recipients = {}
    for hotkey, share in zip(hotkeys, shares, strict=True):
        uid = uid_by_hotkey.get(hotkey)
        if uid is not None:
            if uid in recipients and recipients[uid][0] != hotkey:
                raise ValueError("multiple hotkeys resolve to one UID")
            recipients[uid] = (hotkey, recipients.get(uid, (hotkey, 0))[1] + share)
    if not recipients:
        return (f"burn:uid:{burn_uid}",), (burn_uid,), (1.0,)
    total = sum(share for _, share in recipients.values())
    if total <= 0:
        raise ValueError("reward total must be positive")
    return (
        tuple(h for h, _ in recipients.values()),
        tuple(recipients),
        tuple(share / total for _, share in recipients.values()),
    )
