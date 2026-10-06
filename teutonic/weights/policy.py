"""Gradual main/specialist reward policy, independent of UID mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

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
