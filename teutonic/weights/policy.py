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
    from teutonic.competitions import MIX_COMPETITIONS
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
    """Allocate 12.5% per MIX; normalize the remainder over surviving positions."""
    validate_transition(state)
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
    from teutonic.competitions import MIX_COMPETITIONS
    validate_transition(state)
    result = deepcopy(state)
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
    result = deepcopy(state)
    if result["version"] == "mix-successor-v2":
        return result
    if "mix04" not in result["mixes"]:
        raise ValueError("MIX04 must have a rewarded king before replacement")
    result.update(version="mix-successor-v2", successors={"mix07": "mix04"})
    validate_transition(result)
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
