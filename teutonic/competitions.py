"""Competition identifiers shared by wire contracts, CLI and publishers."""

import re

LEGACY_COMPETITIONS = ("math", "code", "text")
MIX_COMPETITIONS = tuple(f"mix{i:02d}" for i in range(1, 8))
CONFIGURED_COMPETITIONS = ("main", *MIX_COMPETITIONS)
# Keep every definition for history; rollout accepts only these four competitions.
ACTIVE_INCENTIVES = {"main": 0.4, "mix01": 0.2, "mix06": 0.2, "mix07": 0.2}
ACTIVE_COMPETITIONS = tuple(ACTIVE_INCENTIVES)
BOOTSTRAP_COMPETITIONS = tuple(k for k in CONFIGURED_COMPETITIONS if k != "mix07")


def normalize_competition(value: str) -> str:
    key = value.strip().lower()
    if key == "main" or re.fullmatch(r"mix(?:0[1-9]|[1-9][0-9])", key):
        return key
    raise ValueError("competition must be MAIN or MIX01–MIX99")
