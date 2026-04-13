"""Reveal-commit scanning: parse revealed commitments, track seen hotkeys."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

COMMIT_SEPARATOR = ":"


@dataclass
class RevealCommit:
    """A revealed on-chain commitment from a miner."""

    hotkey: str
    block: int
    king_hash: str
    hf_repo: str
    model_hash: str
    raw: str = ""

    @classmethod
    def parse(cls, hotkey: str, block: int, raw_data: str) -> RevealCommit | None:
        """Parse a revealed commitment string.

        Expected format: {king_hash}:{hf_repo_id}:{model_sha256}
        """
        parts = raw_data.split(COMMIT_SEPARATOR, maxsplit=2)
        if len(parts) != 3:
            logger.warning(
                "Invalid reveal format from %s: expected 3 parts, got %d",
                hotkey[:16], len(parts),
            )
            return None

        king_hash, hf_repo, model_hash = parts
        if not king_hash or not hf_repo or not model_hash:
            logger.warning("Empty field in reveal from %s", hotkey[:16])
            return None

        return cls(
            hotkey=hotkey,
            block=block,
            king_hash=king_hash.strip(),
            hf_repo=hf_repo.strip(),
            model_hash=model_hash.strip(),
            raw=raw_data,
        )

    def is_stale(self, current_king_hash: str) -> bool:
        """Check if this commit targets a king that is no longer current."""
        return self.king_hash != current_king_hash[:len(self.king_hash)]


class RevealScanner:
    """Scans the chain for newly revealed miner commitments.

    Uses get_all_revealed_commitments() and permanently tracks every hotkey
    that has been returned so it is never re-processed.
    """

    def __init__(self, subtensor, netuid: int, seen_hotkeys: set[str] | None = None):
        self.subtensor = subtensor
        self.netuid = netuid
        self.seen_hotkeys: set[str] = seen_hotkeys or set()

    def scan(self) -> list[RevealCommit]:
        """Return new reveals sorted by block, skipping already-seen hotkeys.

        Each hotkey is added to seen_hotkeys the moment its reveal is returned,
        guaranteeing it will never be evaluated again.
        """
        try:
            all_reveals = self.subtensor.get_all_revealed_commitments(self.netuid)
        except Exception:
            logger.exception("Failed to fetch revealed commitments")
            return []

        if not all_reveals:
            return []

        new_commits: list[RevealCommit] = []

        for hotkey, entries in all_reveals.items():
            if hotkey in self.seen_hotkeys:
                continue

            if not entries:
                continue

            # Take the most recent reveal (highest block) for this hotkey
            latest_block, latest_data = max(entries, key=lambda e: e[0])

            commit = RevealCommit.parse(hotkey, latest_block, latest_data)
            if commit is None:
                continue

            self.seen_hotkeys.add(hotkey)
            new_commits.append(commit)

        new_commits.sort(key=lambda c: c.block)
        return new_commits
