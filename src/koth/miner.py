"""Reference miner for King of the Hill.

Downloads the current king, trains on pretraining data, uploads the
improved model to the miner's own HuggingFace repo, and commits
on-chain.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import bittensor as bt
import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file as load_safetensors, save_file

from .validation import sha256_of_directory

logger = logging.getLogger(__name__)

SAFETENSORS_PATTERNS = ["*.safetensors"]
IGNORE_PATTERNS = ["*.bin", "*.pt", "*.pkl", "*.py", "__pycache__/*", ".git/*"]


class Miner:
    """Reference KOTH miner."""

    def __init__(
        self,
        king_repo: str,
        miner_repo: str,
        netuid: int = 3,
        network: str = "finney",
        wallet_name: str = "default",
        wallet_hotkey: str = "default",
        hf_token: str = "",
        cache_dir: str = "/tmp/koth/miner",
        learning_rate: float = 1e-5,
        train_steps: int = 100,
        sequence_length: int = 2048,
        batch_size: int = 4,
        dataset_path: str = "",
    ):
        self.king_repo = king_repo
        self.miner_repo = miner_repo
        self.hf_token = hf_token or None
        self.cache_dir = Path(cache_dir)
        self.api = HfApi(token=self.hf_token)

        self.lr = learning_rate
        self.train_steps = train_steps
        self.seq_len = sequence_length
        self.batch_size = batch_size
        self.dataset_path = dataset_path

        self.wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        self.subtensor = bt.subtensor(network=network)
        self.netuid = netuid

    def run(self) -> None:
        """Full miner loop: download king, train, upload, commit."""
        logger.info("Starting KOTH miner")

        # 1. Download king
        king_dir = self._download_king()
        king_hash = sha256_of_directory(king_dir)
        logger.info("Downloaded king, hash=%s", king_hash[:16])

        # 2. Train
        model_dir = self._train(king_dir)

        # 3. Upload to own HF repo
        model_hash = self._upload(model_dir)

        # 4. Commit on-chain
        self._commit(king_hash, model_hash)

        logger.info("Miner submission complete.")

    def _download_king(self) -> Path:
        target = self.cache_dir / "king"
        target.mkdir(parents=True, exist_ok=True)
        local = snapshot_download(
            repo_id=self.king_repo,
            local_dir=str(target),
            token=self.hf_token,
            allow_patterns=SAFETENSORS_PATTERNS + ["config.json", "tokenizer*"],
            ignore_patterns=IGNORE_PATTERNS,
        )
        return Path(local)

    def _train(self, king_dir: Path) -> Path:
        """Train on pretraining data starting from the king weights.

        This is a minimal reference implementation. Real miners would
        use more sophisticated training setups.
        """
        logger.info("Training for %d steps with lr=%e", self.train_steps, self.lr)

        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers required for training: pip install transformers")

        model = AutoModelForCausalLM.from_pretrained(
            str(king_dir),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=False,
        )
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.1)

        # Load dataset
        if self.dataset_path:
            tokens = np.load(self.dataset_path, mmap_mode="r", allow_pickle=False)
            if tokens.dtype != np.uint32:
                tokens = tokens.astype(np.uint32, copy=False)
            if tokens.ndim != 1:
                tokens = tokens.reshape(-1)
            tokens_t = torch.from_numpy(tokens)
        else:
            logger.warning("No dataset path provided, using random data for training")
            tokens_t = torch.randint(0, 32000, (self.seq_len * self.train_steps * self.batch_size,))

        n_sequences = len(tokens_t) // self.seq_len
        rng = np.random.Generator(np.random.PCG64(int(time.time())))

        for step in range(self.train_steps):
            indices = rng.choice(n_sequences, size=self.batch_size, replace=False)
            batch = torch.stack([
                tokens_t[i * self.seq_len : (i + 1) * self.seq_len]
                for i in indices
            ]).to("cuda", dtype=torch.long)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % 10 == 0:
                logger.info("Step %d/%d, loss=%.4f", step + 1, self.train_steps, loss.item())

        # Save trained model
        output_dir = self.cache_dir / "trained"
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir), safe_serialization=True)

        # Copy config files from king
        import shutil
        for f in king_dir.glob("config.json"):
            shutil.copy2(f, output_dir / f.name)
        for f in king_dir.glob("tokenizer*"):
            shutil.copy2(f, output_dir / f.name)

        logger.info("Training complete. Model saved to %s", output_dir)
        return output_dir

    def _upload(self, model_dir: Path) -> str:
        """Upload trained model to miner's HF repo."""
        try:
            self.api.create_repo(repo_id=self.miner_repo, exist_ok=True, private=False)
        except Exception:
            pass

        self.api.upload_folder(
            folder_path=str(model_dir),
            repo_id=self.miner_repo,
            commit_message="KOTH challenger submission",
            allow_patterns=["*.safetensors", "config.json", "tokenizer*", "special_tokens*"],
        )

        model_hash = sha256_of_directory(model_dir)
        logger.info("Uploaded model to %s, hash=%s", self.miner_repo, model_hash[:16])
        return model_hash

    def _commit(self, king_hash: str, model_hash: str) -> None:
        """Commit on-chain: king_hash:hf_repo:model_hash."""
        # Truncate king hash to 16 chars for the commit
        payload = f"{king_hash[:16]}:{self.miner_repo}:{model_hash}"
        logger.info("Committing: %s", payload[:80])

        self.subtensor.commit(
            wallet=self.wallet,
            netuid=self.netuid,
            data=payload,
        )
        logger.info("Commit successful.")
