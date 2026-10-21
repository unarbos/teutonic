"""Competition identifiers shared by wire contracts, CLI and publishers."""

import re

LEGACY_COMPETITIONS = ("math", "code", "text")
MIX_COMPETITIONS = tuple(f"mix{i:02d}" for i in range(1, 7))
ACTIVE_COMPETITIONS = ("main", *MIX_COMPETITIONS)


def normalize_competition(value: str) -> str:
    key = value.strip().lower()
    if key == "main" or re.fullmatch(r"mix(?:0[1-9]|[1-9][0-9])", key):
        return key
    raise ValueError("competition must be MAIN or MIX01–MIX99")
