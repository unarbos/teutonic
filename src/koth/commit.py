"""On-chain commit parsing, staleness detection, ordering, deduplication."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

COMMIT_SEPARATOR = ":"


@dataclass
class MinerCommit:
    """Parsed on-chain commit from a miner."""

    hotkey: str
    uid: int
    block: int
    king_hash: str
    hf_repo: str
    model_hash: str
    raw: str = ""

    @classmethod
    def parse(cls, hotkey: str, uid: int, block: int, raw_commitment: str) -> MinerCommit | None:
        """Parse a raw commitment string into a MinerCommit.

        Expected format: {king_hash}:{hf_repo_id}:{model_sha256}
        """
        parts = raw_commitment.split(COMMIT_SEPARATOR, maxsplit=2)
        if len(parts) != 3:
            logger.warning(
                "Invalid commit format from %s (uid %d): expected 3 parts, got %d",
                hotkey, uid, len(parts),
            )
            return None

        king_hash, hf_repo, model_hash = parts
        if not king_hash or not hf_repo or not model_hash:
            logger.warning("Empty field in commit from %s (uid %d)", hotkey, uid)
            return None

        return cls(
            hotkey=hotkey,
            uid=uid,
            block=block,
            king_hash=king_hash.strip(),
            hf_repo=hf_repo.strip(),
            model_hash=model_hash.strip(),
            raw=raw_commitment,
        )

    def is_stale(self, current_king_hash: str) -> bool:
        """Check if this commit targets a king that is no longer current."""
        return self.king_hash != current_king_hash[:len(self.king_hash)]


def deduplicate_commits(commits: list[MinerCommit]) -> list[MinerCommit]:
    """Keep only the latest commit per hotkey, ordered by block number."""
    by_hotkey: dict[str, MinerCommit] = {}
    for c in sorted(commits, key=lambda x: x.block):
        by_hotkey[c.hotkey] = c
    return sorted(by_hotkey.values(), key=lambda x: x.block)


class CommitScanner:
    """Scans the Bittensor chain for new miner commits."""

    def __init__(self, subtensor, netuid: int):
        self.subtensor = subtensor
        self.netuid = netuid
        self._last_scanned_block = 0
        self._seen_commits: set[str] = set()

    def scan(self) -> list[MinerCommit]:
        """Scan for new commits since last scan.

        Returns commits ordered by block number, deduplicated by hotkey.
        """
        metagraph = self.subtensor.metagraph(self.netuid)
        commits: list[MinerCommit] = []

        for uid in range(len(metagraph.hotkeys)):
            hotkey = metagraph.hotkeys[uid]
            try:
                raw = self.subtensor.get_commitment(self.netuid, uid)
            except Exception:
                continue

            if not raw:
                continue

            commit_key = f"{hotkey}:{raw}"
            if commit_key in self._seen_commits:
                continue

            current_block = self.subtensor.block
            commit = MinerCommit.parse(hotkey, uid, current_block, raw)
            if commit:
                commits.append(commit)
                self._seen_commits.add(commit_key)

        return deduplicate_commits(commits)
