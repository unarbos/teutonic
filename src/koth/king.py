"""King model lifecycle: HuggingFace operations, hashing, forking."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .validation import sha256_of_directory

logger = logging.getLogger(__name__)

SAFETENSORS_PATTERNS = ["*.safetensors"]
IGNORE_PATTERNS = [
    "*.bin", "*.pt", "*.pkl", "*.pickle", "*.py", "*.pyc",
    "*.sh", "*.bat", "*.js", "*.ts",
    "__pycache__/*", ".git/*",
]


class KingManager:
    """Manages the king model on HuggingFace and local cache."""

    def __init__(self, hf_repo: str, cache_dir: str, hf_token: str = ""):
        self.hf_repo = hf_repo
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token or None
        self.api = HfApi(token=self.hf_token)
        self._king_hash: str = ""

    @property
    def king_hash(self) -> str:
        return self._king_hash

    @property
    def king_dir(self) -> Path:
        return self.cache_dir / "current"

    def ensure_repo_exists(self) -> None:
        """Create the king HF repo if it doesn't exist."""
        try:
            self.api.create_repo(
                repo_id=self.hf_repo,
                repo_type="model",
                exist_ok=True,
                private=False,
            )
        except Exception:
            logger.info("King repo %s already exists or creation failed (OK if exists)", self.hf_repo)

    def download_king(self) -> Path:
        """Download the current king from HuggingFace."""
        self.king_dir.mkdir(parents=True, exist_ok=True)
        local = snapshot_download(
            repo_id=self.hf_repo,
            local_dir=str(self.king_dir),
            token=self.hf_token,
            allow_patterns=SAFETENSORS_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
        self._king_hash = sha256_of_directory(local)
        logger.info("Downloaded king from %s, hash=%s", self.hf_repo, self._king_hash[:16])
        return Path(local)

    def upload_seed(self, model_dir: str | Path, commit_message: str = "Initial seed king") -> str:
        """Upload an initial seed model to the king repo."""
        model_dir = Path(model_dir)
        self.ensure_repo_exists()

        self.api.upload_folder(
            folder_path=str(model_dir),
            repo_id=self.hf_repo,
            commit_message=commit_message,
            allow_patterns=["*.safetensors", "config.json", "tokenizer*", "special_tokens*"],
        )

        # Copy to local cache
        if self.king_dir.exists():
            shutil.rmtree(self.king_dir)
        self.king_dir.mkdir(parents=True)
        for f in model_dir.glob("*.safetensors"):
            shutil.copy2(f, self.king_dir / f.name)
        for f in model_dir.glob("config.json"):
            shutil.copy2(f, self.king_dir / f.name)

        self._king_hash = sha256_of_directory(self.king_dir)
        logger.info("Uploaded seed king to %s, hash=%s", self.hf_repo, self._king_hash[:16])
        return self._king_hash

    def fork_winner(
        self,
        winner_dir: str | Path,
        miner_hotkey: str,
        challenge_id: str,
    ) -> str:
        """Fork a winning challenger's weights into the king repo.

        Uploads the winner's safetensors to the canonical king repo as a new
        commit, giving a git history of every king.
        """
        winner_dir = Path(winner_dir)

        commit_msg = f"King #{self.king_hash[:8]} dethroned by {miner_hotkey[:16]} ({challenge_id})"

        self.api.upload_folder(
            folder_path=str(winner_dir),
            repo_id=self.hf_repo,
            commit_message=commit_msg,
            allow_patterns=["*.safetensors"],
        )

        # Update local cache
        if self.king_dir.exists():
            shutil.rmtree(self.king_dir)
        self.king_dir.mkdir(parents=True)
        for f in winner_dir.glob("*.safetensors"):
            shutil.copy2(f, self.king_dir / f.name)
        # Preserve config from existing king
        # (architecture doesn't change, only weights)

        self._king_hash = sha256_of_directory(self.king_dir)
        logger.info(
            "Forked winner into king repo %s, new hash=%s",
            self.hf_repo, self._king_hash[:16],
        )
        return self._king_hash

    def download_challenger(self, hf_repo: str, target_dir: str | Path | None = None) -> Path:
        """Download a challenger model from a miner's HF repo.

        Only downloads .safetensors files -- ignores all code, configs, etc.
        """
        if target_dir is None:
            target_dir = self.cache_dir / "challenger"
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        local = snapshot_download(
            repo_id=hf_repo,
            local_dir=str(target_dir),
            token=self.hf_token,
            allow_patterns=SAFETENSORS_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
        logger.info("Downloaded challenger from %s to %s", hf_repo, local)
        return Path(local)
