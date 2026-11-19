"""Competition identifiers shared by wire contracts, CLI and publishers."""

import re

LEGACY_COMPETITIONS = ("math", "code", "text")
MIX_COMPETITIONS = tuple(f"mix{i:02d}" for i in range(1, 10))
CONFIGURED_COMPETITIONS = ("main", "main2", *MIX_COMPETITIONS)
# Keep every definition for history; rollout accepts only these four competitions.
ACTIVE_INCENTIVES = {"main2": 0.4, "mix07": 0.2, "mix08": 0.2, "mix09": 0.2}
ACTIVE_COMPETITIONS = tuple(ACTIVE_INCENTIVES)
# This historical migration must not silently create future competitions.
BOOTSTRAP_COMPETITIONS = ("main", *(f"mix{i:02d}" for i in range(1, 7)))


def normalize_competition(value: str) -> str:
    key = value.strip().lower()
    if key in {"main", "main2"} or re.fullmatch(r"mix(?:0[1-9]|[1-9][0-9])", key):
        return key
    raise ValueError("competition must be MAIN, MAIN2 or MIX01–MIX99")
